"""Two-pass GeoTIFF inference for the supported global spatial mean pattern."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import rasterio
from rasterio.windows import Window

from job_memory import JobMemoryLimit
from run_raster_fixed import DEFAULT_DLL, DEFAULT_EXE, model_contract, run as run_pointwise


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit-mib", type=int, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--band-order", help="Comma-separated 1-based source band indexes")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--ort-dll", type=Path, default=DEFAULT_DLL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_reduction(model):
    onnx.checker.check_model(model)
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    inputs = [item for item in graph.input if item.name not in initializers]
    if len(inputs) != 1 or len(graph.output) != 1 or len(graph.node) < 2:
        raise ValueError("global-mean schedule needs one image input and output")
    image = inputs[0]
    dims = image.type.tensor_type.shape.dim
    if image.type.tensor_type.elem_type != TensorProto.FLOAT or len(dims) != 4:
        raise ValueError("global-mean input must be NCHW FP32")
    if dims[0].HasField("dim_value") and dims[0].dim_value != 1:
        raise ValueError("global-mean schedule needs batch size one")
    if any(dims[axis].HasField("dim_value") for axis in (2, 3)):
        raise ValueError("global-mean schedule needs dynamic spatial dimensions")
    channels = dims[1].dim_value
    if channels < 1:
        raise ValueError("global-mean channel count must be fixed")
    reduction = graph.node[0]
    if reduction.op_type != "ReduceMean" or list(reduction.input[:1]) != [image.name]:
        raise ValueError("first node must reduce the image")
    if len(reduction.output) != 1 or len(reduction.input) != 2:
        raise ValueError("unsupported global-mean reduction wiring")
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in reduction.attribute}
    if set(attrs) - {"keepdims", "noop_with_empty_axes"}:
        raise ValueError("unsupported global-mean reduction attributes")
    if attrs.get("keepdims", 1) != 1 or attrs.get("noop_with_empty_axes", 0) != 0:
        raise ValueError("global mean must retain spatial singleton dimensions")
    axes = initializers.get(reduction.input[1])
    if axes is None or tuple(numpy_helper.to_array(axes).reshape(-1)) != (2, 3):
        raise ValueError("global mean must reduce spatial axes 2 and 3")
    if not any(reduction.output[0] in node.input for node in graph.node[1:]):
        raise ValueError("global mean has no downstream consumer")
    if graph.output[0].name == reduction.output[0]:
        raise ValueError("global mean output must be a per-pixel raster")
    return channels, reduction.output[0]


def streaming_mean(source, channels, bands, tile_size, scale, offset):
    sums = np.zeros(channels, dtype=np.float64)
    windows = 0
    with rasterio.open(source) as src:
        if src.count != channels or src.crs is None:
            raise ValueError("raster requires a CRS and the model's band count")
        height, width = src.height, src.width
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                window = Window(col, row, min(tile_size, width-col),
                                min(tile_size, height-row))
                if np.any(src.read_masks(window=window) != 255):
                    raise ValueError(f"no-data pixel in window at row {row}, col {col}")
                block = src.read(indexes=bands, window=window, out_dtype="float32")
                if not np.isfinite(block).all():
                    raise ValueError(f"non-finite input at row {row}, col {col}")
                block = block * scale + offset
                if not np.isfinite(block).all():
                    raise ValueError(f"non-finite preprocessed input at row {row}, col {col}")
                sums += block.sum(axis=(1, 2), dtype=np.float64)
                windows += 1
    return (sums / (height * width)).astype(np.float32), windows


def transform(model, reduction_output, mean, path, tile_size):
    updated = copy.deepcopy(model)
    del updated.graph.node[0]
    updated.graph.initializer.append(
        numpy_helper.from_array(mean.reshape(1, len(mean), 1, 1),
                                name=reduction_output)
    )
    onnx.checker.check_model(updated)
    onnx.save(updated, path)
    model_contract(path, tile_size)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, external_job=None):
    started = time.perf_counter()
    model_path = args.model.resolve(strict=True)
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if output in {model_path, source}:
        raise ValueError("output must differ from model and input")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    if args.tile_size < 1 or args.threads < 1 or args.memory_limit_mib < 1:
        raise ValueError("tile size, threads, and memory limit must be positive")
    if not np.isfinite(args.scale) or not np.isfinite(args.offset):
        raise ValueError("scale and offset must be finite")
    model = onnx.load(model_path)
    channels, reduction_output = validate_reduction(model)
    if args.band_order:
        bands = tuple(int(part) for part in args.band_order.split(","))
    else:
        bands = tuple(range(1, channels + 1))
    if sorted(bands) != list(range(1, channels + 1)):
        raise ValueError("band order must be a permutation of source bands")
    output.parent.mkdir(parents=True, exist_ok=True)
    transformed = output.parent / f".{output.stem}.mean-{uuid4().hex}.onnx"
    failure_path = output.with_suffix(".failure.json")
    job = external_job
    published = False
    try:
        if job is None:
            job = JobMemoryLimit(args.memory_limit_mib * 1024 * 1024)
        reduction_started = time.perf_counter()
        mean, reduction_windows = streaming_mean(
            source, channels, bands, args.tile_size, args.scale, args.offset
        )
        reduction_seconds = time.perf_counter() - reduction_started
        transformed_hash = transform(model, reduction_output, mean,
                                     transformed, args.tile_size)
        raster_args = copy.copy(args)
        raster_args.model = transformed
        raster_args.cut_after = 0
        raster_args.worker_mode = "pipe"
        run_pointwise(raster_args, external_job=job)
        published = True
        report_path = output.with_suffix(".report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(model=str(model_path), strategy="two-pass-global-mean",
                      transformed_model_sha256=transformed_hash,
                      global_mean_fp32=mean.tolist(),
                      reduction_windows=reduction_windows,
                      reduction_seconds=reduction_seconds,
                      elapsed_seconds=time.perf_counter()-started,
                      job_peak_committed_bytes=job.peak_job_bytes(),
                      job_peak_conservative_upper_bound_bytes=job.conservative_peak_bytes())
        report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        failure_path.unlink(missing_ok=True)
    except Exception as exc:
        failure = dict(status="failed", error=str(exc),
                       elapsed_seconds=time.perf_counter()-started,
                       memory_limit_bytes=args.memory_limit_mib*1024*1024,
                       job_peak_committed_bytes=job.peak_job_bytes() if job else None,
                       output_published=published)
        failure_path.write_text(json.dumps(failure, indent=2)+"\n", encoding="utf-8")
        raise
    finally:
        transformed.unlink(missing_ok=True)
        if job is not None and external_job is None:
            job.close()


if __name__ == "__main__":
    run(arguments())
