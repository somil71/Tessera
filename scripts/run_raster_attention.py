"""Exact blockwise spatial self-attention for the validated ONNX fixture pattern."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "64"

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import rasterio
from rasterio.windows import Window

from job_memory import JobMemoryLimit
from run_raster_fixed import (configure_gdal_cache, own_peak_commit_bytes,
                              own_peak_working_set_bytes)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit-mib", type=int, required=True)
    parser.add_argument("--time-limit-minutes", type=float, default=30.0)
    parser.add_argument("--band-order", help="Comma-separated 1-based source band indexes")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_model(model):
    onnx.checker.check_model(model)
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    inputs = [item for item in graph.input if item.name not in initializers]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ValueError("attention schedule needs one image input and output")
    for value in (inputs[0], graph.output[0]):
        tensor = value.type.tensor_type
        dims = tensor.shape.dim
        if tensor.elem_type != TensorProto.FLOAT or len(dims) != 4:
            raise ValueError("attention schedule needs NCHW FP32 tensors")
        if dims[0].HasField("dim_value") and dims[0].dim_value != 1:
            raise ValueError("attention schedule needs batch size one")
        if any(dims[axis].HasField("dim_value") for axis in (2, 3)):
            raise ValueError("attention schedule needs dynamic spatial dimensions")
    channels = inputs[0].type.tensor_type.shape.dim[1].dim_value
    if channels < 1 or graph.output[0].type.tensor_type.shape.dim[1].dim_value != channels:
        raise ValueError("attention input/output channel counts must match")
    nodes = list(graph.node)
    if [node.op_type for node in nodes] != [
        "Shape", "Reshape", "Transpose", "MatMul", "Softmax", "MatMul",
        "Transpose", "Reshape"
    ]:
        raise ValueError("unsupported attention operator sequence")
    if any(len(node.output) != 1 for node in nodes):
        raise ValueError("attention nodes must have one output")
    if [len(node.input) for node in nodes] != [1, 2, 1, 2, 1, 2, 1, 2]:
        raise ValueError("unsupported attention node input counts")
    names = [node.output[0] for node in nodes]
    image = inputs[0].name
    expected_inputs = [
        [image], [image, nodes[1].input[1]], [names[1]],
        [names[2], names[1]], [names[3]], [names[4], names[2]],
        [names[5]], [names[6], names[0]],
    ]
    if [list(node.input) for node in nodes] != expected_inputs:
        raise ValueError("unsupported attention dataflow")
    if graph.output[0].name != names[-1]:
        raise ValueError("attention output is not the final reshape")
    flat_shape = initializers.get(nodes[1].input[1])
    if flat_shape is None or tuple(numpy_helper.to_array(flat_shape).reshape(-1)) != (1, channels, -1):
        raise ValueError("unsupported attention flatten shape")
    allowed_attrs = [set(), {"allowzero"}, {"perm"}, set(), {"axis"},
                     set(), {"perm"}, {"allowzero"}]
    for index, node in enumerate(nodes):
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if set(attrs) - allowed_attrs[index]:
            raise ValueError(f"unsupported attention attributes at node {index}")
        if node.op_type == "Reshape" and attrs.get("allowzero", 0) != 0:
            raise ValueError("attention reshape allowzero is unsupported")
        if node.op_type == "Transpose" and tuple(attrs.get("perm", [])) != (0, 2, 1):
            raise ValueError("attention transpose must swap pixels and channels")
        if node.op_type == "Softmax" and attrs.get("axis", -1) not in (-1, 2):
            raise ValueError("attention softmax must run over all keys")
    return channels


def online_attention(features, output, query_block, key_block):
    pixels, channels = features.shape
    for query_start in range(0, pixels, query_block):
        query_end = min(pixels, query_start+query_block)
        query = np.array(features[query_start:query_end], dtype=np.float32, copy=True)
        qcount = query_end-query_start
        maximum = np.full(qcount, -np.inf, dtype=np.float64)
        denominator = np.zeros(qcount, dtype=np.float64)
        numerator = np.zeros((qcount, channels), dtype=np.float64)
        for key_start in range(0, pixels, key_block):
            key_end = min(pixels, key_start+key_block)
            keys = np.array(features[key_start:key_end], dtype=np.float32, copy=True)
            scores = query @ keys.T
            new_maximum = np.maximum(maximum, scores.max(axis=1))
            old_scale = np.exp(maximum-new_maximum)
            probabilities = np.exp(scores.astype(np.float64)-new_maximum[:, None])
            denominator = old_scale*denominator + probabilities.sum(axis=1)
            numerator = old_scale[:, None]*numerator + probabilities @ keys.astype(np.float64)
            maximum = new_maximum
        output[query_start:query_end] = (numerator/denominator[:, None]).astype(np.float32)


def estimate_cost(pixels, channels, query_block, key_block):
    pilot = np.random.default_rng(8701).normal(size=(1024, channels)).astype(np.float32)
    result = np.empty_like(pilot)
    samples = []
    for _ in range(3):
        began = time.perf_counter()
        online_attention(pilot, result, query_block, key_block)
        samples.append(time.perf_counter()-began)
    pilot_seconds = sorted(samples)[1]
    measured_rate = 1024**2 / max(pilot_seconds, 1e-6)
    # Cap extrapolation at 20 million score cells/s, then add a 2x allowance.
    assumed_rate = min(measured_rate, 20_000_000)
    predicted = 2 * pixels**2 / assumed_rate
    return predicted, measured_rate, pilot_seconds


def read_features(source, bands, scale, offset, features, channels):
    with rasterio.open(source) as src:
        rows_per_read = max(1, (32*1024*1024)//(src.width*channels*4))
        reads = 0
        for row in range(0, src.height, rows_per_read):
            count = min(rows_per_read, src.height-row)
            window = Window(0, row, src.width, count)
            if np.any(src.read_masks(window=window) != 255):
                raise ValueError(f"no-data pixel in raster rows {row}:{row+count}")
            block = src.read(indexes=bands, window=window, out_dtype="float32")
            if not np.isfinite(block).all():
                raise ValueError(f"non-finite input in raster rows {row}:{row+count}")
            block = block*scale+offset
            if not np.isfinite(block).all():
                raise ValueError(f"non-finite preprocessed input in rows {row}:{row+count}")
            features[row*src.width:(row+count)*src.width] = (
                block.transpose(1, 2, 0).reshape(-1, channels)
            )
            reads += 1
    features.flush()
    return reads


def write_scores(source, partial, scores, channels):
    with rasterio.open(source) as src:
        profile = dict(driver="GTiff", width=src.width, height=src.height,
                       count=channels, dtype="float32", crs=src.crs,
                       transform=src.transform, nodata=None, compress="deflate",
                       predictor=3, BIGTIFF="IF_SAFER")
        with rasterio.open(partial, "w", **profile) as dst:
            dst.update_tags(**src.tags())
            for band in range(1, channels+1):
                dst.set_band_description(band, f"score_{band}")
            rows_per_write = max(1, (32*1024*1024)//(src.width*channels*4))
            for row in range(0, src.height, rows_per_write):
                count = min(rows_per_write, src.height-row)
                data = np.array(scores[row*src.width:(row+count)*src.width], copy=True)
                data = data.reshape(count, src.width, channels).transpose(2, 0, 1)
                dst.write(data, window=Window(0, row, src.width, count))


def run(args, external_job=None):
    started = time.perf_counter()
    model_path = args.model.resolve(strict=True)
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if output in {model_path, source}:
        raise ValueError("output must differ from model and input")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    if args.memory_limit_mib < 1:
        raise ValueError("memory limit must be positive")
    time_limit = getattr(args, "time_limit_minutes", 30.0)
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time limit must be positive and finite")
    if not np.isfinite(args.scale) or not np.isfinite(args.offset):
        raise ValueError("scale and offset must be finite")
    gdal_cachemax_mib = configure_gdal_cache(args.memory_limit_mib)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.stem}.attention-{uuid4().hex}"
    work.mkdir()
    feature_path = work / "features.f32"
    score_path = work / "scores.f32"
    partial = output.parent / f".{output.stem}.partial-{uuid4().hex}.tif"
    failure_path = output.with_suffix(".failure.json")
    job = external_job
    feature_map = None
    score_map = None
    published = False
    estimate = None
    try:
        if job is None:
            job = JobMemoryLimit(args.memory_limit_mib*1024*1024)
        model = onnx.load(model_path)
        channels = validate_model(model)
        with rasterio.open(source) as src:
            if src.crs is None or src.count != channels:
                raise ValueError("raster requires a CRS and the model's band count")
            height, width = src.height, src.width
            crs, transform = str(src.crs), list(src.transform)
        pixels = height*width
        bands = (tuple(int(part) for part in args.band_order.split(","))
                 if args.band_order else tuple(range(1, channels+1)))
        if sorted(bands) != list(range(1, channels+1)):
            raise ValueError("band order must permute source bands")
        query_block, key_block = 128, 256
        estimate, rate, pilot_seconds = estimate_cost(
            pixels, channels, query_block, key_block
        )
        if estimate > time_limit*60:
            raise RuntimeError(
                f"estimated attention runtime {estimate:.1f} s exceeds "
                f"the {time_limit*60:.0f} s limit"
            )
        raw_bytes = pixels*channels*4
        free_bytes = shutil.disk_usage(output.parent).free
        if free_bytes < 3*raw_bytes+64*1024*1024:
            raise RuntimeError("insufficient free disk for attention staging and output")
        feature_map = np.memmap(feature_path, dtype="<f4", mode="w+",
                                shape=(pixels, channels))
        reads = read_features(source, bands, args.scale, args.offset,
                              feature_map, channels)
        score_map = np.memmap(score_path, dtype="<f4", mode="w+",
                              shape=(pixels, channels))
        attention_started = time.perf_counter()
        online_attention(feature_map, score_map, query_block, key_block)
        attention_seconds = time.perf_counter()-attention_started
        score_map.flush()
        write_scores(source, partial, score_map, channels)
        output_file_bytes = partial.stat().st_size
        os.replace(partial, output)
        published = True
        report = dict(model=str(model_path), input=str(source), output=str(output),
                      strategy="blockwise-exact-attention", shape=[height, width],
                      input_bands=channels, score_bands=channels,
                      crs=crs, transform=transform, band_order=bands,
                      scale=args.scale, offset=args.offset,
                      query_block=query_block, key_block=key_block,
                      gdal_cachemax_mib=gdal_cachemax_mib,
                      estimated_attention_seconds=estimate,
                      pilot_seconds=pilot_seconds,
                      measured_pilot_score_cells_per_second=rate,
                      attention_seconds=attention_seconds,
                      elapsed_seconds=time.perf_counter()-started,
                      memory_limit_bytes=args.memory_limit_mib*1024*1024,
                      memory_limit_enforced=True,
                      job_peak_committed_bytes=job.peak_job_bytes(),
                      job_peak_conservative_upper_bound_bytes=job.conservative_peak_bytes(),
                      scheduler_peak_commit_bytes=own_peak_commit_bytes(),
                      scheduler_peak_working_set_bytes=own_peak_working_set_bytes(),
                      input_staging_bytes=raw_bytes, score_staging_bytes=raw_bytes,
                      output_file_bytes=output_file_bytes,
                      peak_temp_storage_bytes=2*raw_bytes+output_file_bytes,
                      raster_read_windows=reads, numerical_error=None)
        report.update(job.resource_snapshot())
        output.with_suffix(".report.json").write_text(
            json.dumps(report, indent=2)+"\n", encoding="utf-8"
        )
        failure_path.unlink(missing_ok=True)
        print(f"wrote {output}; {pixels} pixels of blockwise attention")
    except Exception as exc:
        failure = dict(status="failed", error=str(exc),
                       estimated_attention_seconds=estimate,
                       elapsed_seconds=time.perf_counter()-started,
                       memory_limit_bytes=args.memory_limit_mib*1024*1024,
                       job_peak_committed_bytes=job.peak_job_bytes() if job else None,
                       output_published=published)
        failure_path.write_text(json.dumps(failure, indent=2)+"\n", encoding="utf-8")
        raise
    finally:
        if feature_map is not None:
            feature_map.flush()
            del feature_map
        if score_map is not None:
            score_map.flush()
            del score_map
        gc.collect()
        for path in (feature_path, score_path, partial):
            path.unlink(missing_ok=True)
        work.rmdir()
        if job is not None and external_job is None:
            job.close()


if __name__ == "__main__":
    run(arguments())
