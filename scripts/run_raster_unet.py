"""Exact window schedule for the frozen local U-Net ONNX topology.

This engineering path requires even raster dimensions and even output tile size.
The input halo is clipped at the real image edge, preserving Conv padding there.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import onnx
from onnx import TensorProto, helper
import rasterio
from rasterio.windows import Window

from run_raster_fixed import DEFAULT_DLL, DEFAULT_EXE, execute, own_peak_commit_bytes, read_exact
from job_memory import JobMemoryLimit


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cut-after", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit-mib", type=int,
                        help="Enforce a Windows job-wide committed-memory budget")
    parser.add_argument("--band-order", help="Comma-separated 1-based source band indexes")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--ort-dll", type=Path, default=DEFAULT_DLL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_model(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    inputs = [item for item in graph.input if item.name not in initializers]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ValueError("U-Net schedule requires one input and one output")
    for value in (inputs[0], graph.output[0]):
        tensor = value.type.tensor_type
        if tensor.elem_type != TensorProto.FLOAT or len(tensor.shape.dim) != 4:
            raise ValueError("U-Net schedule requires NCHW FP32")
        if tensor.shape.dim[0].HasField("dim_value") and tensor.shape.dim[0].dim_value != 1:
            raise ValueError("U-Net schedule requires batch size one")
        for axis in (2, 3):
            if tensor.shape.dim[axis].HasField("dim_value"):
                raise ValueError("U-Net schedule requires dynamic spatial dimensions")
    ops = [node.op_type for node in graph.node]
    if ops != ["Conv", "Relu", "MaxPool", "Conv", "Relu", "ConvTranspose", "Add", "Conv"]:
        raise ValueError("unsupported U-Net topology")
    nodes = graph.node
    expected_edges = [
        ("image", "enc"), ("enc", "skip"), ("skip", "pooled"),
        ("pooled", "bottleneck"), ("bottleneck", "deep"),
        ("deep", "up"), ("up", "merged"), ("merged", "logits"),
    ]
    if [(node.input[0], node.output[0]) for node in nodes] != expected_edges:
        raise ValueError("unsupported U-Net wiring")
    if list(nodes[6].input) != ["up", "skip"]:
        raise ValueError("unsupported U-Net skip connection")
    allowed_attrs = {
        0: {"kernel_shape", "pads", "strides", "dilations", "auto_pad", "group"},
        1: set(),
        2: {"kernel_shape", "strides", "pads", "dilations", "ceil_mode", "auto_pad"},
        3: {"kernel_shape", "pads", "strides", "dilations", "auto_pad", "group"},
        4: set(),
        5: {"kernel_shape", "strides", "pads", "dilations", "auto_pad", "group", "output_padding"},
        6: set(),
        7: {"kernel_shape", "pads", "strides", "dilations", "auto_pad", "group"},
    }
    for index, node in enumerate(nodes):
        if len(node.output) != 1 or any(attr.name not in allowed_attrs[index] for attr in node.attribute):
            raise ValueError(f"unsupported U-Net attributes at node {index}")
    for index, (kernel, pads, strides, dilation) in {
        0: ((3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
        3: ((3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
        7: ((1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
    }.items():
        node = nodes[index]
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        weight = initializers.get(node.input[1])
        if (weight is None or tuple(weight.dims[2:]) != kernel
            or tuple(attrs.get("pads", [0, 0, 0, 0])) != pads
            or tuple(attrs.get("strides", [1, 1])) != strides
            or tuple(attrs.get("dilations", [1, 1])) != dilation
            or attrs.get("auto_pad", b"NOTSET") != b"NOTSET"
            or attrs.get("group", 1) != 1):
            raise ValueError(f"unsupported U-Net Conv at node {index}")
    pool = {attr.name: helper.get_attribute_value(attr) for attr in nodes[2].attribute}
    up = {attr.name: helper.get_attribute_value(attr) for attr in nodes[5].attribute}
    if (tuple(pool.get("kernel_shape", [])) != (2, 2)
        or tuple(pool.get("strides", [1, 1])) != (2, 2)
        or tuple(pool.get("pads", [0, 0, 0, 0])) != (0, 0, 0, 0)
        or tuple(pool.get("dilations", [1, 1])) != (1, 1)
        or pool.get("auto_pad", b"NOTSET") != b"NOTSET"
        or pool.get("ceil_mode", 0) != 0
        or tuple(up.get("kernel_shape", [])) != (2, 2)
        or tuple(up.get("strides", [1, 1])) != (2, 2)
        or tuple(up.get("pads", [0, 0, 0, 0])) != (0, 0, 0, 0)
        or tuple(up.get("dilations", [1, 1])) != (1, 1)
        or up.get("auto_pad", b"NOTSET") != b"NOTSET"
        or tuple(up.get("output_padding", [0, 0])) != (0, 0)
        or up.get("group", 1) != 1):
        raise ValueError("unsupported U-Net pooling or upsampling")
    channels = inputs[0].type.tensor_type.shape.dim[1].dim_value
    scores = graph.output[0].type.tensor_type.shape.dim[1].dim_value
    if channels < 1 or scores < 1:
        raise ValueError("channel dimensions must be fixed and positive")
    return channels, scores


class Worker:
    def __init__(self, exe, env, model, directory, shape, cut, channels, threads,
                 job=None):
        self.height, self.width = shape
        self.directory = directory
        directory.mkdir()
        self.head = directory / "head.onnx"
        self.tail = directory / "tail.onnx"
        self.process = None
        try:
            execute([exe, "split", model, "--cut-after", str(cut), "--height",
                     str(self.height), "--width", str(self.width), "--out-dir", directory], env, job)
            self.process = subprocess.Popen(
                [exe, *(["--await-start"] if job is not None else []),
                 "serve-split-pipe", self.head, self.tail, "--channels", str(channels),
                 "--height", str(self.height), "--width", str(self.width),
                 "--threads", str(threads)],
                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0,
            )
            if job is not None:
                job.assign_process(self.process)
                self.process.stdin.write(b"\x01")
                self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line or json.loads(line).get("ready") is not True:
                exit_code = self.process.wait(timeout=5)
                raise RuntimeError(
                    f"Rust worker failed to start for {shape} under the requested memory budget: "
                    f"exit {exit_code}; {self.process.stderr.read()}"
                )
        except Exception:
            if self.process is not None:
                self.process.terminate()
                self.process.wait(timeout=5)
                self.process.stdin.close()
                self.process.stdout.close()
                self.process.stderr.close()
            self.head.unlink(missing_ok=True)
            self.tail.unlink(missing_ok=True)
            directory.rmdir()
            raise

    def infer(self, tile, output_channels):
        payload = np.ascontiguousarray(tile, dtype="<f4").tobytes()
        self.process.stdin.write((json.dumps({"input_bytes": len(payload)}) + "\n").encode() + payload)
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Rust worker stopped: {self.process.stderr.read().decode(errors='replace')}")
        metric = json.loads(line)
        if metric.get("ok") is not True:
            raise RuntimeError(f"Rust worker failed: {metric.get('error')}")
        expected = output_channels * self.height * self.width * 4
        if metric.get("output_bytes") != expected:
            raise ValueError("Rust worker returned wrong output size")
        scores = np.frombuffer(read_exact(self.process.stdout, expected), dtype="<f4")
        return scores.reshape(output_channels, self.height, self.width), metric

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        self.process.stdout.close()
        self.process.stderr.close()
        self.head.unlink(missing_ok=True)
        self.tail.unlink(missing_ok=True)
        self.directory.rmdir()


def run(args, external_job=None):
    started = time.perf_counter()
    model = args.model.resolve(strict=True)
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if output in (model, source):
        raise ValueError("output must differ from model and input")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    if args.tile_size < 2 or args.tile_size % 2 or args.threads < 1:
        raise ValueError("U-Net tile size must be positive and even; threads must be positive")
    if args.memory_limit_mib is not None and args.memory_limit_mib < 1:
        raise ValueError("memory limit must be positive")
    if not np.isfinite(args.scale) or not np.isfinite(args.offset):
        raise ValueError("scale and offset must be finite")
    channels, output_channels = validate_model(model)
    with rasterio.open(source) as source_check:
        expected_crs = source_check.crs
        expected_shape = (source_check.height, source_check.width)
        if expected_crs is None or source_check.count != channels:
            raise ValueError("raster requires a CRS and the model's band count")
        if source_check.height % 2 or source_check.width % 2:
            raise ValueError("U-Net whole-image reference requires even raster dimensions")
    exe = args.exe.resolve(strict=True)
    dll = args.ort_dll.resolve(strict=True)
    env = os.environ.copy()
    env["ORT_DYLIB_PATH"] = str(dll)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.stem}.work-{uuid4().hex}"
    work.mkdir()
    partial = output.parent / f".{output.stem}.partial-{uuid4().hex}.tif"
    failure_path = output.with_suffix(".failure.json")
    worker = None
    job = external_job
    published = False
    shapes = {}
    peak_child = 0
    peak_scheduler = 0
    tiles = 0
    try:
        if job is None and args.memory_limit_mib is not None:
            job = JobMemoryLimit(args.memory_limit_mib * 1024 * 1024)
        with rasterio.open(source) as src:
            if (src.crs != expected_crs or src.count != channels
                    or (src.height, src.width) != expected_shape):
                raise RuntimeError("raster metadata changed or could not be read under the memory limit")
            bands = (tuple(int(part) for part in args.band_order.split(","))
                     if args.band_order else tuple(range(1, channels + 1)))
            if sorted(bands) != list(range(1, channels + 1)):
                raise ValueError("band order must permute the source bands")
            profile = dict(driver="GTiff", width=src.width, height=src.height,
                           count=output_channels, dtype="float32", crs=src.crs,
                           transform=src.transform, nodata=None, compress="deflate",
                           predictor=3, BIGTIFF="IF_SAFER")
            with rasterio.open(partial, "w", **profile) as dst:
                dst.update_tags(**src.tags())
                for band in range(1, output_channels + 1):
                    dst.set_band_description(band, f"score_{band}")
                for row in range(0, src.height, args.tile_size):
                    for col in range(0, src.width, args.tile_size):
                        core_h = min(args.tile_size, src.height - row)
                        core_w = min(args.tile_size, src.width - col)
                        # Four pixels cover the encoder Conv, pool, bottleneck
                        # Conv and decoder path. Even boundaries preserve pool phase.
                        top = max(0, row - 4)
                        left = max(0, col - 4)
                        bottom = min(src.height, row + core_h + 4)
                        right = min(src.width, col + core_w + 4)
                        shape = (bottom - top, right - left)
                        shapes.setdefault(shape, []).append((row, col, core_h, core_w, top, left))
                # Output windows may be written in any order. Grouping by halo
                # shape reuses one session pair without holding several workers.
                for shape, jobs in shapes.items():
                    worker = Worker(exe, env, model,
                        work / f"shape_{shape[0]}x{shape[1]}", shape,
                        args.cut_after, channels, args.threads, job)
                    for row, col, core_h, core_w, top, left in jobs:
                        window = Window(left, top, shape[1], shape[0])
                        if np.any(src.read_masks(window=window) != 255):
                            raise ValueError(f"no-data pixel near row {row}, col {col}")
                        tile = src.read(indexes=bands, window=window, out_dtype="float32")
                        if not np.isfinite(tile).all():
                            raise ValueError(f"non-finite input near row {row}, col {col}")
                        tile = tile * args.scale + args.offset
                        scores, metric = worker.infer(tile, output_channels)
                        peak_child = max(peak_child, int(metric["peak_commit_bytes"]))
                        dst.write(scores[:, row-top:row-top+core_h, col-left:col-left+core_w],
                                  window=Window(col, row, core_w, core_h))
                        peak_scheduler = max(peak_scheduler, own_peak_commit_bytes())
                        tiles += 1
                    worker.close()
                    worker = None
        os.replace(partial, output)
        published = True
        report = dict(model=str(model), input=str(source), output=str(output),
                      shape=[src.height, src.width], crs=str(src.crs),
                      transform=list(src.transform), tile_size=args.tile_size,
                      tile_count=tiles, halo_pixels=4, alignment_pixels=2,
                      worker_shapes=[list(shape) for shape in shapes],
                      cut_after=args.cut_after, band_order=bands,
                      scale=args.scale, offset=args.offset,
                      elapsed_seconds=time.perf_counter()-started,
                      max_child_peak_commit_bytes=peak_child,
                      scheduler_peak_commit_bytes=peak_scheduler,
                      combined_process_peak_upper_bound_bytes=peak_child+peak_scheduler,
                      memory_limit_bytes=job.budget_bytes if job else None,
                      job_peak_committed_bytes=job.peak_job_bytes() if job else None,
                      job_peak_conservative_upper_bound_bytes=(
                          job.conservative_peak_bytes() if job else None),
                      raw_bridge_bytes=0, numerical_error=None,
                      memory_limit_enforced=job is not None)
        output.with_suffix(".report.json").write_text(json.dumps(report, indent=2)+"\n")
        failure_path.unlink(missing_ok=True)
        print(f"wrote {output}; {tiles} U-Net tiles")
    except Exception as exc:
        failure = dict(status="failed", error=str(exc),
                       elapsed_seconds=time.perf_counter()-started,
                       memory_limit_bytes=(args.memory_limit_mib * 1024 * 1024
                                           if args.memory_limit_mib is not None else None),
                       job_peak_committed_bytes=job.peak_job_bytes() if job else None,
                       job_peak_conservative_upper_bound_bytes=(
                           job.conservative_peak_bytes() if job else None),
                       output_published=published)
        failure_path.write_text(json.dumps(failure, indent=2)+"\n")
        raise
    finally:
        if worker is not None:
            worker.close()
        partial.unlink(missing_ok=True)
        work.rmdir()
        if job is not None and external_job is None:
            job.close()


if __name__ == "__main__":
    run(arguments())
