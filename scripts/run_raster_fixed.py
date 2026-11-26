"""Fixed window schedule for pixelwise ONNX models on GeoTIFF rasters.

Rasterio handles geospatial I/O. The Rust CLI performs each tile's ONNX execution.
This is a bounded development path, not the memory-limited planner.
"""

import argparse
import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

# Raster window assembly does not use BLAS. Prevent OpenBLAS from reserving
# hundreds of megabytes for its default thread pool before NumPy is imported.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "64"

import numpy as np
import onnx
from onnx import TensorProto, helper
import rasterio
from rasterio.windows import Window

from job_memory import JobMemoryLimit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "target" / "debug" / "raster-compiler.exe"


def _resolve_default_dll():
    venv_dll = (
        ROOT / ".venv" / "Lib" / "site-packages" / "onnxruntime" / "capi" / "onnxruntime.dll"
    )
    if venv_dll.exists():
        return venv_dll
    try:
        import onnxruntime
        pkg_dll = Path(onnxruntime.__file__).parent / "capi" / "onnxruntime.dll"
        if pkg_dll.exists():
            return pkg_dll
    except Exception:
        pass
    return venv_dll


DEFAULT_DLL = _resolve_default_dll()


def configure_gdal_cache(memory_limit_mib):
    """Bound GDAL's raster block cache before the first raster read."""
    if memory_limit_mib is None:
        cache_mib = 64
    else:
        # GDAL's cache competes with the scheduler and ONNX worker within the
        # same Windows Job Object. A large cap here can push combined commit
        # over the budget even when an individual worker probe succeeded.
        cache_mib = max(8, min(256, memory_limit_mib // 8))
    os.environ["GDAL_CACHEMAX"] = str(cache_mib)
    return cache_mib


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cut-after", type=int, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit-mib", type=int,
                        help="Enforce a Windows job-wide committed-memory budget")
    parser.add_argument(
        "--worker-mode", choices=("pipe", "persistent", "fresh"), default="pipe"
    )
    parser.add_argument("--band-order", help="Comma-separated 1-based source band indexes")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--ort-dll", type=Path, default=DEFAULT_DLL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_contract(path, tile_size):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    inputs = [item for item in graph.input if item.name not in initializers]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ValueError("model needs one image input and one raster output")
    for value in (inputs[0], graph.output[0]):
        tensor = value.type.tensor_type
        if tensor.elem_type != TensorProto.FLOAT or len(tensor.shape.dim) != 4:
            raise ValueError(f"{value.name} must be NCHW FP32")
        dims = tensor.shape.dim
        if dims[0].HasField("dim_value") and dims[0].dim_value != 1:
            raise ValueError(f"{value.name} must have batch size 1")
        for axis in (2, 3):
            if dims[axis].HasField("dim_value") and dims[axis].dim_value != tile_size:
                raise ValueError(f"{value.name} has a fixed spatial size incompatible with tile size")
    input_channels = inputs[0].type.tensor_type.shape.dim[1].dim_value
    output_channels = graph.output[0].type.tensor_type.shape.dim[1].dim_value
    if input_channels < 1 or output_channels < 1:
        raise ValueError("input and output channel counts must be fixed positive values")
    for index, node in enumerate(graph.node):
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if node.op_type == "Conv":
            weight = initializers.get(node.input[1]) if len(node.input) > 1 else None
            if (
                weight is None
                or tuple(weight.dims[2:]) != (1, 1)
                or attrs.get("kernel_shape", [1, 1]) != [1, 1]
                or attrs.get("strides", [1, 1]) != [1, 1]
                or attrs.get("dilations", [1, 1]) != [1, 1]
                or attrs.get("pads", [0, 0, 0, 0]) != [0, 0, 0, 0]
                or attrs.get("auto_pad", b"NOTSET") != b"NOTSET"
                or attrs.get("group", 1) != 1
            ):
                raise ValueError(f"node {index} Conv is not a supported pointwise convolution")
        elif node.op_type in {"Relu", "Identity", "Sigmoid"}:
            if len(node.input) != 1 or len(node.output) != 1 or attrs:
                raise ValueError(f"node {index} has unsupported pointwise attributes or arity")
            continue
        elif node.op_type == "BatchNormalization":
            if (len(node.input) != 5 or len(node.output) != 1
                or any(name not in initializers for name in node.input[1:])
                or any(len(initializers[name].dims) != 1 for name in node.input[1:])
                or len({initializers[name].dims[0] for name in node.input[1:]}) != 1
                or attrs.get("training_mode", 0) != 0
                or not set(attrs).issubset({"epsilon", "momentum", "training_mode"})):
                raise ValueError(f"node {index} BatchNormalization is not supported inference mode")
        elif node.op_type in {"Add", "Sub", "Mul"}:
            if len(node.input) != 2 or len(node.output) != 1 or attrs:
                raise ValueError(f"node {index} has unsupported pointwise attributes or arity")
            for name in node.input:
                if name in initializers:
                    dims = list(initializers[name].dims)
                    if len(dims) > 4 or any(dim != 1 for dim in dims[-2:]):
                        raise ValueError(f"node {index} uses a spatially varying constant")
        else:
            raise ValueError(f"node {index} ({node.op_type}) is not independent across pixels")
    return input_channels, output_channels


def execute(command, env, job=None):
    if job is None:
        result = subprocess.run(command, env=env, text=True, capture_output=True)
    else:
        guarded = [command[0], "--await-start", *command[1:]]
        process = subprocess.Popen(guarded, env=env, text=True,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        try:
            job.assign_process(process)
            stdout, stderr = process.communicate("\x01")
        except Exception:
            process.terminate()
            process.communicate()
            raise
        result = subprocess.CompletedProcess(guarded, process.returncode, stdout, stderr)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def read_exact(stream, count):
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ValueError(f"Rust worker returned only {count - remaining} of {count} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@lru_cache(maxsize=1)
def memory_api():
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("page_fault_count", ctypes.c_ulong),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
            ("private_usage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    return Counters, kernel32, psapi


def own_peak_commit_bytes():
    """Return this Python process's lifetime peak committed bytes on Windows."""
    Counters, kernel32, psapi = memory_api()
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.peak_pagefile_usage)


def own_peak_working_set_bytes():
    """Return this Python process's lifetime peak resident bytes on Windows."""
    Counters, kernel32, psapi = memory_api()
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.peak_working_set_size)


def run(args, external_job=None):
    started = time.perf_counter()
    model = args.model.resolve(strict=True)
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if output in {model, source}:
        raise ValueError("output must differ from model and input")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
    if args.tile_size < 1 or args.threads < 1 or args.cut_after < 0:
        raise ValueError("tile size and threads must be positive; cut index must be nonnegative")
    if args.memory_limit_mib is not None and args.memory_limit_mib < 1:
        raise ValueError("memory limit must be positive")
    if not np.isfinite(args.scale) or not np.isfinite(args.offset):
        raise ValueError("scale and offset must be finite")
    gdal_cachemax_mib = configure_gdal_cache(args.memory_limit_mib)
    input_channels, output_channels = model_contract(model, args.tile_size)
    with rasterio.open(source) as source_check:
        expected_crs = source_check.crs
        expected_shape = (source_check.height, source_check.width)
        if expected_crs is None:
            raise ValueError("input GeoTIFF has no CRS")
        if source_check.count != input_channels:
            raise ValueError(
                f"model requires {input_channels} bands, raster has {source_check.count}"
            )
    exe = args.exe.resolve(strict=True)
    dll = args.ort_dll.resolve(strict=True)
    env = os.environ.copy()
    env["ORT_DYLIB_PATH"] = str(dll)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.stem}.work-{uuid4().hex}"
    work.mkdir()
    temporary_output = output.parent / f".{output.stem}.partial-{uuid4().hex}.tif"
    report_path = output.with_suffix(".report.json")
    failure_path = output.with_suffix(".failure.json")
    tile_input = work / "tile.f32"
    tile_output = work / "scores.f32"
    head = work / "head.onnx"
    tail = work / "tail.onnx"
    tiles = 0
    max_child_commit = 0
    max_scheduler_commit = 0
    raw_bridge_bytes = 0
    worker = None
    job = external_job
    published = False
    try:
        if job is None and args.memory_limit_mib is not None:
            job = JobMemoryLimit(args.memory_limit_mib * 1024 * 1024)
        execute(
            [
                exe,
                "split",
                model,
                "--cut-after",
                str(args.cut_after),
                "--height",
                str(args.tile_size),
                "--width",
                str(args.tile_size),
                "--out-dir",
                work,
            ],
            env,
            job,
        )
        if args.worker_mode in {"persistent", "pipe"}:
            worker = subprocess.Popen(
                [
                    exe,
                    *(["--await-start"] if job is not None else []),
                    "serve-split-pipe" if args.worker_mode == "pipe" else "serve-split",
                    head,
                    tail,
                    "--channels",
                    str(input_channels),
                    "--height",
                    str(args.tile_size),
                    "--width",
                    str(args.tile_size),
                    "--threads",
                    str(args.threads),
                ],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=args.worker_mode != "pipe",
                bufsize=0 if args.worker_mode == "pipe" else 1,
            )
            if job is not None:
                job.assign_process(worker)
                worker.stdin.write(b"\x01" if args.worker_mode == "pipe" else "\x01")
                worker.stdin.flush()
            ready_line = worker.stdout.readline()
            if not ready_line:
                exit_code = worker.wait(timeout=5)
                raise RuntimeError(
                    f"Rust worker failed to start under the requested memory budget "
                    f"(exit {exit_code}): {worker.stderr.read()}"
                )
            ready = json.loads(ready_line)
            if ready.get("ready") is not True:
                raise RuntimeError(f"unexpected Rust worker response: {ready}")
        with rasterio.open(source) as src:
            if (src.crs != expected_crs or src.count != input_channels
                    or (src.height, src.width) != expected_shape):
                raise RuntimeError("raster metadata changed or could not be read under the memory limit")
            if args.band_order:
                bands = tuple(int(part) for part in args.band_order.split(","))
            else:
                bands = tuple(range(1, input_channels + 1))
            if sorted(bands) != list(range(1, input_channels + 1)):
                raise ValueError("band order must be a permutation of the source bands")
            profile = {
                "driver": "GTiff",
                "width": src.width,
                "height": src.height,
                "count": output_channels,
                "dtype": "float32",
                "crs": src.crs,
                "transform": src.transform,
                "nodata": None,
                "compress": "deflate",
                "predictor": 3,
                "BIGTIFF": "IF_SAFER",
            }
            with rasterio.open(temporary_output, "w", **profile) as dst:
                dst.update_tags(**src.tags())
                for band in range(1, output_channels + 1):
                    dst.set_band_description(band, f"score_{band}")
                for row in range(0, src.height, args.tile_size):
                    for col in range(0, src.width, args.tile_size):
                        height = min(args.tile_size, src.height - row)
                        width = min(args.tile_size, src.width - col)
                        window = Window(col, row, width, height)
                        if np.any(src.read_masks(window=window) != 255):
                            raise ValueError(f"no-data pixel in window at row {row}, col {col}")
                        actual = src.read(indexes=bands, window=window, out_dtype="float32")
                        if not np.isfinite(actual).all():
                            raise ValueError(f"non-finite input pixel in window at row {row}, col {col}")
                        tile = np.zeros(
                            (input_channels, args.tile_size, args.tile_size), dtype=np.float32
                        )
                        tile[:, :height, :width] = actual * args.scale + args.offset
                        if args.worker_mode == "pipe":
                            payload = tile.astype("<f4", copy=False).tobytes(order="C")
                            worker.stdin.write(
                                (json.dumps({"input_bytes": len(payload)}) + "\n").encode()
                                + payload
                            )
                            worker.stdin.flush()
                            line = worker.stdout.readline()
                            if not line:
                                raise RuntimeError(
                                    f"Rust worker stopped: {worker.stderr.read().decode(errors='replace')}"
                                )
                            metric = json.loads(line)
                            if metric.get("ok") is not True:
                                raise RuntimeError(f"Rust worker failed: {metric.get('error')}")
                            expected_bytes = output_channels * args.tile_size * args.tile_size * 4
                            if metric["output_bytes"] != expected_bytes:
                                raise ValueError(
                                    f"Rust announced {metric['output_bytes']} output bytes, expected {expected_bytes}"
                                )
                            output_data = read_exact(worker.stdout, expected_bytes)
                            scores = np.frombuffer(output_data, dtype="<f4")
                        elif worker is None:
                            tile.tofile(tile_input)
                            stdout = execute(
                                [
                                    exe,
                                    "run-split",
                                    head,
                                    tail,
                                    "--input-f32",
                                    tile_input,
                                    "--output-f32",
                                    tile_output,
                                    "--channels",
                                    str(input_channels),
                                    "--height",
                                    str(args.tile_size),
                                    "--width",
                                    str(args.tile_size),
                                    "--threads",
                                    str(args.threads),
                                ],
                                env,
                                job,
                            )
                            metrics = next(
                                line
                                for line in stdout.splitlines()
                                if line.startswith("mode=split")
                            )
                            metric = dict(part.split("=", 1) for part in metrics.split())
                        else:
                            tile.tofile(tile_input)
                            worker.stdin.write(
                                json.dumps(
                                    {
                                        "input_f32": str(tile_input),
                                        "output_f32": str(tile_output),
                                    }
                                )
                                + "\n"
                            )
                            worker.stdin.flush()
                            line = worker.stdout.readline()
                            if not line:
                                raise RuntimeError(
                                    f"Rust worker stopped: {worker.stderr.read()}"
                                )
                            metric = json.loads(line)
                            if metric.get("ok") is not True:
                                raise RuntimeError(f"Rust worker failed: {metric.get('error')}")
                        max_child_commit = max(max_child_commit, int(metric["peak_commit_bytes"]))
                        expected = output_channels * args.tile_size * args.tile_size
                        if args.worker_mode != "pipe":
                            scores = np.fromfile(tile_output, dtype="<f4")
                        if scores.size != expected:
                            raise ValueError(f"Rust produced {scores.size} scores, expected {expected}")
                        scores = scores.reshape(output_channels, args.tile_size, args.tile_size)
                        dst.write(scores[:, :height, :width], window=window)
                        tiles += 1
                        if args.worker_mode != "pipe":
                            raw_bridge_bytes += tile_input.stat().st_size + tile_output.stat().st_size
                        max_scheduler_commit = max(
                            max_scheduler_commit, own_peak_commit_bytes()
                        )
        os.replace(temporary_output, output)
        published = True
        report = {
            "model": str(model),
            "input": str(source),
            "output": str(output),
            "shape": [src.height, src.width],
            "input_bands": input_channels,
            "score_bands": output_channels,
            "crs": str(src.crs),
            "transform": list(src.transform),
            "tile_size": args.tile_size,
            "gdal_cachemax_mib": gdal_cachemax_mib,
            "tile_count": tiles,
            "worker_mode": args.worker_mode,
            "cut_after": args.cut_after,
            "band_order": bands,
            "scale": args.scale,
            "offset": args.offset,
            "elapsed_seconds": time.perf_counter() - started,
            "max_child_peak_commit_bytes": max_child_commit,
            "scheduler_peak_commit_bytes": max_scheduler_commit,
            "combined_process_peak_upper_bound_bytes": max_child_commit
            + max_scheduler_commit,
            "memory_limit_bytes": (args.memory_limit_mib * 1024 * 1024
                                   if args.memory_limit_mib is not None else None),
            "job_peak_committed_bytes": job.peak_job_bytes() if job else None,
            "job_peak_conservative_upper_bound_bytes":
                job.conservative_peak_bytes() if job else None,
            "raw_bridge_bytes": raw_bridge_bytes,
            "numerical_error": None,
            "memory_limit_enforced": job is not None,
            "notes": "Separate process peaks are not simultaneous. The job counter measures the job-wide peak; its conservative upper bound adds pre-assignment scheduler commitment. Pipe mode keeps tile data in memory.",
        }
        if job is not None:
            report.update(job.resource_snapshot())
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        failure_path.unlink(missing_ok=True)
        print(f"wrote {output} and {report_path}; {tiles} tiles")
    except Exception as exc:
        failure = {
            "status": "failed",
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
            "memory_limit_bytes": (args.memory_limit_mib * 1024 * 1024
                                   if args.memory_limit_mib is not None else None),
            "job_peak_committed_bytes": job.peak_job_bytes() if job else None,
            "job_peak_conservative_upper_bound_bytes":
                job.conservative_peak_bytes() if job else None,
            "output_published": published,
        }
        failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if worker is not None:
            if worker.poll() is None:
                worker.stdin.close()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.wait(timeout=5)
            worker.stdout.close()
            worker.stderr.close()
        temporary_output.unlink(missing_ok=True)
        for path in (tile_input, tile_output, head, tail):
            path.unlink(missing_ok=True)
        work.rmdir()
        if job is not None and external_job is None:
            job.close()


if __name__ == "__main__":
    run(arguments())
