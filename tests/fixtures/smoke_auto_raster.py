"""Exercise automatic tile/cut selection on all supported raster schedules."""

import json
import os
from pathlib import Path
import subprocess

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = ROOT / "scripts" / "run_raster_auto.py"
OUT = ROOT / "runs" / "auto_raster_smoke"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = OUT / "input.tif"
TRANSFORM = from_origin(500000, 2000000, 10, 10)
IMAGE = np.random.default_rng(3102).normal(size=(3, 32, 48)).astype(np.float32)

with rasterio.open(SOURCE, "w", driver="GTiff", width=48, height=32,
                   count=3, dtype="float32", crs="EPSG:32643",
                   transform=TRANSFORM) as dst:
    dst.write(IMAGE)
    dst.update_tags(test_origin="auto-smoke")

for family, model in (
    ("pointwise", ROOT / "models" / "pointwise_dev.onnx"),
    ("local_unet", ROOT / "models" / "evaluation" / "unet_dev.onnx"),
    ("global_mean", ROOT / "models" / "evaluation" / "global_mean_dev.onnx"),
):
    output = OUT / f"{family}.tif"
    preprocessing = (["--band-order", "3,2,1", "--scale", "0.5",
                      "--offset", "-0.1"] if family == "pointwise" else [])
    result = subprocess.run(
        [PYTHON, RUNNER, "--model", model, "--input", SOURCE,
         "--output", output, "--memory-limit-mib", "128",
         "--max-tile-size", "32", "--overwrite", *preprocessing],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    reference_input = (IMAGE[[2, 1, 0]]*0.5-0.1
                       if family == "pointwise" else IMAGE)
    reference = session.run(None, {"image": reference_input[np.newaxis]})[0][0]
    with rasterio.open(output) as raster:
        actual = raster.read()
        assert raster.crs == rasterio.crs.CRS.from_epsg(32643)
        assert raster.transform == TRANSFORM
        assert raster.tags()["test_origin"] == "auto-smoke"
        assert raster.dtypes == ("float32", "float32")
    max_abs = float(np.max(np.abs(actual-reference)))
    eligible = np.abs(reference) >= 1e-3
    max_rel = (float(np.max(np.abs(actual[eligible]-reference[eligible]) /
                            np.abs(reference[eligible]))) if np.any(eligible) else 0.0)
    assert max_abs <= 1e-4 and max_rel <= 1e-3, (family, max_abs, max_rel)
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["strategy"] == f"auto-{family}"
    assert report["memory_limit_enforced"] is True
    assert report["job_peak_conservative_upper_bound_bytes"] <= 128*1024*1024
    assert report["selected_plan"] in report["planning_candidates"]
    assert len(report["planning_candidates"]) > 1
    print(f"{family}: tile {report['selected_plan']['tile_size']}, "
          f"cut {report['selected_plan']['cut_after']}, max_abs={max_abs:.3g}")

print("automatic raster plans passed all three references")

small_output = OUT / "too_small.tif"
small_output.unlink(missing_ok=True)
failed = subprocess.run(
    [PYTHON, RUNNER, "--model", ROOT / "models" / "pointwise_dev.onnx",
     "--input", SOURCE, "--output", small_output,
     "--memory-limit-mib", "60", "--max-tile-size", "32"],
    capture_output=True, text=True,
)
assert failed.returncode != 0 and not small_output.exists()
failure = json.loads(small_output.with_suffix(".plan.failure.json").read_text())
assert failure["status"] == "failed"
assert failure["memory_limit_bytes"] == 60*1024*1024
assert failure["planning_candidates"]
print("no feasible plan: reported without publishing output")
