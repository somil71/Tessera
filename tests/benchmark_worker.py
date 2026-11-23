"""Compare fresh and persistent Rust workers on the same pointwise raster."""

import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = ROOT / "scripts" / "run_raster_fixed.py"
MODEL = ROOT / "models" / "pointwise_dev.onnx"
SOURCE = ROOT / "runs" / "raster_smoke" / "input.tif"
OUT = ROOT / "runs" / "worker_benchmark"
OUT.mkdir(parents=True, exist_ok=True)
REPEATS = 3
TILE_SIZE = 16


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


results = {"fresh": [], "persistent": [], "pipe": []}
for repeat in range(REPEATS):
    modes = ("fresh", "persistent", "pipe")
    order = modes[repeat:] + modes[:repeat]
    for mode in order:
        output = OUT / f"{mode}.tif"
        subprocess.run(
            [
                PYTHON,
                RUNNER,
                "--model",
                MODEL,
                "--input",
                SOURCE,
                "--output",
                output,
                "--cut-after",
                "0",
                "--tile-size",
                str(TILE_SIZE),
                "--worker-mode",
                mode,
                "--overwrite",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(output.with_suffix(".report.json").read_text())
        if mode == "pipe":
            assert report["raw_bridge_bytes"] == 0
        results[mode].append(
            {
                "repeat": repeat,
                "elapsed_seconds": report["elapsed_seconds"],
                "max_child_peak_commit_bytes": report["max_child_peak_commit_bytes"],
                "scheduler_peak_commit_bytes": report["scheduler_peak_commit_bytes"],
                "combined_process_peak_upper_bound_bytes": report[
                    "combined_process_peak_upper_bound_bytes"
                ],
                "raw_bridge_bytes": report["raw_bridge_bytes"],
                "tile_count": report["tile_count"],
            }
        )
    with rasterio.open(OUT / "fresh.tif") as fresh, rasterio.open(
        OUT / "persistent.tif"
    ) as persistent, rasterio.open(OUT / "pipe.tif") as pipe:
        assert np.array_equal(fresh.read(), persistent.read())
        assert np.array_equal(fresh.read(), pipe.read())
        assert fresh.crs == persistent.crs == pipe.crs
        assert fresh.transform == persistent.transform == pipe.transform

report = {
    "model_sha256": sha256(MODEL),
    "input_sha256": sha256(SOURCE),
    "tile_size": TILE_SIZE,
    "repeats": REPEATS,
    "runs": results,
    "medians": {
        mode: {
            "elapsed_seconds": statistics.median(
                run["elapsed_seconds"] for run in mode_results
            ),
            "max_child_peak_commit_bytes": statistics.median(
                run["max_child_peak_commit_bytes"] for run in mode_results
            ),
            "scheduler_peak_commit_bytes": statistics.median(
                run["scheduler_peak_commit_bytes"] for run in mode_results
            ),
            "combined_process_peak_upper_bound_bytes": statistics.median(
                run["combined_process_peak_upper_bound_bytes"] for run in mode_results
            ),
        }
        for mode, mode_results in results.items()
    },
    "limitation": "The combined figure sums separate process peaks and is an upper bound, not a measured simultaneous job-wide peak.",
}
path = OUT / "report.json"
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report["medians"], indent=2))
print(path)
