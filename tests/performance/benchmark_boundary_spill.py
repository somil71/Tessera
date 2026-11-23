"""Compare RAM and disk graph-boundary materialization on one U-Net cut."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "target" / "debug" / "raster-compiler.exe"
MODEL = ROOT / "models" / "evaluation" / "unet_dev.onnx"
DLL = ROOT / ".venv" / "Lib" / "site-packages" / "onnxruntime" / "capi" / "onnxruntime.dll"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--cut-after", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def invoke(command, env):
    started = time.perf_counter()
    result = subprocess.run([str(item) for item in command], env=env,
                            capture_output=True, text=True, check=True)
    metric_line = next(line for line in result.stdout.splitlines() if line.startswith("mode="))
    metrics = {key: int(value) if re.fullmatch(r"\d+", value) else value
               for key, value in (part.split("=", 1) for part in metric_line.split())}
    metrics["wall_seconds"] = time.perf_counter() - started
    return metrics


def run(args):
    if args.size < 2 or args.size % 2 or args.repeats < 1 or args.threads < 1:
        raise ValueError("size must be positive and even; repeats and threads positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(3101).standard_normal(
        (1, 3, args.size, args.size), dtype=np.float32
    )
    input_path = out / "input.f32"
    image.tofile(input_path)
    reference = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"]).run(
        None, {"image": image}
    )[0]
    split_dir = out / "split"
    subprocess.run([EXE, "split", MODEL, "--cut-after", str(args.cut_after),
                    "--height", str(args.size), "--width", str(args.size),
                    "--out-dir", split_dir], check=True, capture_output=True)
    env = os.environ.copy()
    env["ORT_DYLIB_PATH"] = str(DLL)
    results = {"memory": [], "disk": []}
    for repeat in range(args.repeats):
        order = ("memory", "disk") if repeat % 2 == 0 else ("disk", "memory")
        for storage in order:
            output = out / f"{repeat:02d}_{storage}.f32"
            command = [EXE, "run-split", split_dir / "head.onnx",
                       split_dir / "tail.onnx", "--input-f32", input_path,
                       "--output-f32", output, "--channels", "3",
                       "--height", str(args.size), "--width", str(args.size),
                       "--threads", str(args.threads)]
            if storage == "disk":
                command.extend(["--spill-boundary", "--spill-dir", out])
            metric = invoke(command, env)
            actual = np.fromfile(output, dtype="<f4").reshape(reference.shape)
            delta = np.abs(actual-reference)
            eligible = np.abs(reference) >= 1e-3
            metric["max_absolute_error"] = float(delta.max())
            metric["max_eligible_relative_error"] = (
                float(np.max(delta[eligible]/np.abs(reference[eligible])))
                if np.any(eligible) else 0.0
            )
            metric["output_sha256"] = sha256(output)
            assert metric["max_absolute_error"] <= 1e-4
            assert metric["max_eligible_relative_error"] <= 1e-3
            assert not list(out.glob(".raster-compiler-spill-*"))
            results[storage].append(metric)
    report = dict(model=str(MODEL), model_sha256=sha256(MODEL),
                  input_sha256=sha256(input_path), size=args.size,
                  cut_after=args.cut_after, repeats=args.repeats,
                  threads=args.threads, runs=results,
                  median={storage: {
                      key: statistics.median(item[key] for item in metrics)
                      for key in ("wall_seconds", "total_ms", "peak_commit_bytes",
                                  "peak_working_set_bytes")
                  } for storage, metrics in results.items()})
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report["median"], indent=2))
    print(path)


if __name__ == "__main__":
    run(arguments())
