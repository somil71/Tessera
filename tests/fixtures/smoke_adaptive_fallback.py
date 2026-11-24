"""Force full-run failures to verify adaptive RAM search recovery."""

from argparse import Namespace
import json
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "64"

import numpy as np
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_raster_auto as auto


OUT = ROOT / "runs" / "adaptive_fallback_smoke"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT / "models" / "evaluation" / "unet_dev.onnx"
SOURCE = OUT / "input.tif"
OUTPUT = OUT / "scores.tif"
OUTPUT.unlink(missing_ok=True)
IMAGE = np.random.default_rng(9285).standard_normal((3, 32, 48), dtype=np.float32)
with rasterio.open(SOURCE, "w", driver="GTiff", width=48, height=32,
                   count=3, dtype="float32", crs="EPSG:32643",
                   transform=from_origin(500000, 2000000, 10, 10)) as dst:
    dst.write(IMAGE)

original_profile = auto.profile_candidate
original_run = auto.run_unet


def measured_candidate(*args, **kwargs):
    trial = original_profile(*args, **kwargs)
    if trial["status"] == "ok" and trial["storage_mode"] == "memory":
        # Make tile 32 best and tiles 16/8 clearly slower, independent of CPU jitter.
        trial["predicted_execution_seconds"] = 32/trial["tile_size"]
    return trial


def fail_initial_full_runs(args, external_job=None):
    if args.tile_size >= 8:
        raise RuntimeError("synthetic full-raster failure before publishing output")
    return original_run(args, external_job=external_job)


auto.profile_candidate = measured_candidate
auto.run_unet = fail_initial_full_runs
args = Namespace(model=MODEL, input=SOURCE, output=OUTPUT,
                 memory_limit_mib=128, threads=2, band_order=None,
                 scale=1.0, offset=0.0, exe=auto.DEFAULT_EXE,
                 ort_dll=auto.DEFAULT_DLL, max_tile_size=64,
                 explore_spill=False, overwrite=False)
auto.run(args)

report = json.loads(OUTPUT.with_suffix(".report.json").read_text())
assert report["ram_search_stopped_after_tile"] >= 8
assert report["unprobed_ram_tile_sizes"] == []
assert report["selected_plan"]["tile_size"] < 8
assert report["selected_plan"]["storage_mode"] == "memory"
assert any(item["tile_size"] == 64 and item["status"] == "failed"
           for item in report["planning_candidates"])
assert any(item["tile_size"] >= 8 for item in report["failed_execution_attempts"])
assert report["memory_limit_enforced"] is True
import onnxruntime as ort
reference = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"]).run(
    None, {"image": IMAGE[np.newaxis]})[0][0]
with rasterio.open(OUTPUT) as raster:
    actual = raster.read()
assert np.max(np.abs(actual-reference)) <= 1e-4
print(f"recovered with RAM tile {report['selected_plan']['tile_size']} after "
      f"{len(report['failed_execution_attempts'])} forced full-run failures")
