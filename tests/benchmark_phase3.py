"""Measure full and split U-Net inference in separate Windows processes."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import time

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "target" / "debug" / "raster-compiler.exe"
MODEL = ROOT / "models" / "evaluation" / "unet_dev.onnx"
OUT = ROOT / "runs" / "phase3_benchmark"
OUT.mkdir(parents=True, exist_ok=True)
DLL = ROOT / ".venv" / "Lib" / "site-packages" / "onnxruntime" / "capi" / "onnxruntime.dll"
ENV = os.environ.copy()
ENV["ORT_DYLIB_PATH"] = str(DLL)
HEIGHT = 512
WIDTH = 512
THREADS = 4
CUTS = (2, 4, 5)
REPEATS = 3


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    result = subprocess.run(args, env=ENV, check=True, capture_output=True, text=True)
    metric_line = next(line for line in result.stdout.splitlines() if line.startswith("mode="))
    return {
        key: int(value) if re.fullmatch(r"\d+", value) else value
        for key, value in (part.split("=", 1) for part in metric_line.split())
    }


image = np.random.default_rng(3101).standard_normal((1, 3, HEIGHT, WIDTH)).astype(np.float32)
input_path = OUT / "input.f32"
image.tofile(input_path)
common = [
    "--input-f32",
    str(input_path),
    "--channels",
    "3",
    "--height",
    str(HEIGHT),
    "--width",
    str(WIDTH),
    "--threads",
    str(THREADS),
]
reference_path = OUT / "full.f32"
full_command = [EXE, "run", MODEL, *common, "--output-f32", reference_path]
split_commands = {}
boundary_bytes = {}
graph_text = subprocess.run(
    [EXE, "graph", MODEL, "--height", str(HEIGHT), "--width", str(WIDTH)],
    check=True,
    capture_output=True,
    text=True,
).stdout
for cut in CUTS:
    cut_line = next(
        line for line in graph_text.splitlines() if line.strip().startswith(f"after node {cut}:")
    )
    boundary_bytes[f"cut_{cut}"] = int(re.search(r"boundary bytes=(\d+)", cut_line).group(1))
    directory = OUT / f"cut_{cut}"
    subprocess.run(
        [
            EXE,
            "split",
            MODEL,
            "--cut-after",
            str(cut),
            "--height",
            str(HEIGHT),
            "--width",
            str(WIDTH),
            "--out-dir",
            directory,
        ],
        check=True,
        capture_output=True,
    )
    split_commands[cut] = [
        EXE,
        "run-split",
        directory / "head.onnx",
        directory / "tail.onnx",
        *common,
        "--output-f32",
        directory / "output.f32",
    ]

results = {"full": [], **{f"cut_{cut}": [] for cut in CUTS}}
errors = {}
for repeat in range(REPEATS):
    order = ["full", *(f"cut_{cut}" for cut in CUTS)]
    if repeat % 2:
        order.reverse()
    for label in order:
        command = full_command if label == "full" else split_commands[int(label.split("_")[1])]
        metrics = run(command)
        metrics["repeat"] = repeat
        results[label].append(metrics)
        print(label, metrics, flush=True)

reference = np.fromfile(reference_path, dtype="<f4")
assert reference.size == 2 * HEIGHT * WIDTH
for cut in CUTS:
    actual = np.fromfile(OUT / f"cut_{cut}" / "output.f32", dtype="<f4")
    assert actual.shape == reference.shape
    absolute = np.abs(actual - reference)
    mask = np.abs(reference) >= 1e-3
    max_abs = float(absolute.max())
    max_rel = float(np.max(absolute[mask] / np.abs(reference[mask]))) if mask.any() else 0.0
    errors[f"cut_{cut}"] = {"max_abs": max_abs, "max_rel": max_rel}
    assert max_abs <= 1e-4 and max_rel <= 1e-3, (cut, max_abs, max_rel)

report = {
    "model": str(MODEL),
    "model_sha256": sha256(MODEL),
    "input_sha256": sha256(input_path),
    "input_shape": [1, 3, HEIGHT, WIDTH],
    "threads": THREADS,
    "repeats": REPEATS,
    "onnxruntime_version": ort.__version__,
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "processor": platform.processor(),
    "logical_cpus": os.cpu_count(),
    "recorded_at_unix_seconds": time(),
    "metric_definitions": {
        "peak_commit_bytes": "GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX.PeakPagefileUsage",
        "peak_working_set_bytes": "GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX.PeakWorkingSetSize",
        "total_ms": "Rust process time from input read through output write, excludes process startup",
    },
    "runs": results,
    "boundary_bytes": boundary_bytes,
    "errors": errors,
}
report_path = OUT / "report.json"
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(report_path)
