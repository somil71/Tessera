"""Exercise U-Net disk-spill calibration under a binding 128 MiB cap."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "storage_planner_smoke"
OUT.mkdir(parents=True, exist_ok=True)
PYTHON = sys.executable
MODEL = ROOT / "models" / "evaluation" / "unet_dev.onnx"
SOURCE = OUT / "input.tif"
IMAGE = np.random.default_rng(5201).standard_normal(
    (3, 512, 512), dtype=np.float32
)
with rasterio.open(SOURCE, "w", driver="GTiff", width=512, height=512,
                   count=3, dtype="float32", crs="EPSG:32643",
                   transform=from_origin(500000, 2000000, 10, 10)) as dst:
    dst.write(IMAGE)
    dst.update_tags(test_origin="storage-planner-smoke")

reference = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"]).run(
    None, {"image": IMAGE[np.newaxis]}
)[0][0]


def run(script, output, extra):
    output.unlink(missing_ok=True)
    result = subprocess.run(
        [PYTHON, script, "--model", MODEL, "--input", SOURCE,
         "--output", output, "--memory-limit-mib", "128",
         "--threads", "4", *extra],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["memory_limit_enforced"] is True
    assert report["job_enforced_limit_bytes"] <= 128 * 1024 * 1024
    assert report["job_peak_private_commit_sampled_bytes"] <= 128 * 1024 * 1024
    with rasterio.open(output) as raster:
        actual = raster.read()
        assert raster.crs == rasterio.crs.CRS.from_epsg(32643)
        assert raster.tags()["test_origin"] == "storage-planner-smoke"
    assert np.max(np.abs(actual-reference)) <= 1e-4
    return report


fixed = run(ROOT / "scripts" / "run_raster_unet.py", OUT / "disk.tif",
            ["--tile-size", "512", "--cut-after", "4",
             "--storage-mode", "disk"])
assert fixed["boundary_spill_write_bytes"] > 0

automatic = run(ROOT / "scripts" / "run_raster_auto.py", OUT / "automatic.tif",
                ["--max-tile-size", "512", "--explore-spill"])
disk_trials = [item for item in automatic["planning_candidates"]
               if item["storage_mode"] == "disk"]
assert disk_trials and any(item["status"] == "ok" for item in disk_trials)
assert automatic["selected_plan"]["storage_mode"] in ("memory", "disk")
assert isinstance(automatic["job_peak_counter_exceeds_budget"], bool)
print(f"automatic {automatic['selected_plan']['storage_mode']} tile "
      f"{automatic['selected_plan']['tile_size']}; "
      f"{len(disk_trials)} disk candidates; "
      f"raw peak counter exceeded cap: {automatic['job_peak_counter_exceeds_budget']}")
