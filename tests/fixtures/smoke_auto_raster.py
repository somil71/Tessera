"""Exercise automatic tile/cut selection on all supported raster schedules."""

import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import onnx
import onnxruntime as ort
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from local_graph import LocalPlan
from run_raster_auto import local_probe_shape, unique_probe_tiles

large_halo = LocalPlan(320, 320, 32, 32, ())
assert local_probe_shape(1024, 1024, 512, large_halo) == (832, 832)
assert unique_probe_tiles([512, 256, 128], 1024, 1024, large_halo) == [512, 128]

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
    if family == "pointwise":
        pointwise_reference = reference.copy()
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
    assert report["planning_memory_limit_enforced"] is True
    assert report["overall_job_enforced_limit_bytes"] <= 128*1024*1024
    assert report["overall_job_peak_private_commit_sampled_bytes"] <= 128*1024*1024
    assert report["job_peak_committed_bytes"] <= 128*1024*1024
    assert report["selected_plan"]["execution_job_peak_committed_bytes"] <= 128*1024*1024
    assert report["selected_plan"]["overall_job_active_process_count_before_execution"] == 1
    assert report["selected_plan"]["overall_job_active_process_count_after_execution"] == 1
    assert report["gdal_cachemax_mib"] == 16
    assert report["job_peak_working_set_sampled_bytes"] > 0
    assert report["job_peak_private_commit_sampled_bytes"] <= 128*1024*1024
    assert report["working_set_sample_count"] > 0
    assert report["job_io_read_transfer_bytes"] >= 0
    assert report["job_io_write_transfer_bytes"] >= 0
    assert report["job_peak_conservative_upper_bound_bytes"] <= 128*1024*1024
    assert report["selected_plan"] in report["planning_candidates"]
    assert len(report["planning_candidates"]) > 1
    print(f"{family}: tile {report['selected_plan']['tile_size']}, "
          f"cut {report['selected_plan']['cut_after']}, max_abs={max_abs:.3g}")

print("automatic raster plans passed all three references")

reference_path = OUT / "pointwise_oracle.tif"
with rasterio.open(OUT / "pointwise.tif") as existing:
    reference_profile = existing.profile
with rasterio.open(reference_path, "w", **reference_profile) as oracle:
    oracle.write(pointwise_reference)
checked_output = OUT / "pointwise_checked.tif"
checked = subprocess.run(
    [PYTHON, RUNNER, "--model", ROOT / "models" / "pointwise_dev.onnx",
     "--input", SOURCE, "--output", checked_output,
     "--reference-raster", reference_path, "--memory-limit-mib", "128",
     "--max-tile-size", "32", "--overwrite", "--band-order", "3,2,1",
     "--scale", "0.5", "--offset", "-0.1"],
    capture_output=True, text=True,
)
assert checked.returncode == 0, checked.stderr
checked_report = json.loads(checked_output.with_suffix(".report.json").read_text())
assert checked_report["numerical_error"]["passed"] is True
assert checked_report["numerical_error"]["max_absolute_error"] == 0
assert checked_report["numerical_error"]["reference_sha256"]

wrong_reference = OUT / "pointwise_wrong_oracle.tif"
altered = pointwise_reference.copy()
altered[0, 0, 0] += 0.01
with rasterio.open(wrong_reference, "w", **reference_profile) as oracle:
    oracle.write(altered)
failed_output = OUT / "pointwise_reference_failure.tif"
failed = subprocess.run(
    [PYTHON, RUNNER, "--model", ROOT / "models" / "pointwise_dev.onnx",
     "--input", SOURCE, "--output", failed_output,
     "--reference-raster", wrong_reference, "--memory-limit-mib", "128",
     "--max-tile-size", "32", "--overwrite", "--band-order", "3,2,1",
     "--scale", "0.5", "--offset", "-0.1"],
    capture_output=True, text=True,
)
assert failed.returncode != 0
failed_report = json.loads(failed_output.with_suffix(".report.json").read_text())
assert failed_report["numerical_error"]["passed"] is False
assert failed_report["numerical_error"]["max_absolute_error"] >= 0.01 - 1e-6
assert json.loads(failed_output.with_suffix(".plan.failure.json").read_text())["status"] == "failed"
print("optional score oracle: exact match and numerical failure recorded")

attention_model = ROOT / "models" / "evaluation" / "attention_holdout.onnx"
attention_source = OUT / "attention_input.tif"
attention_output = OUT / "block_attention.tif"
attention_image = np.random.default_rng(4101).normal(size=(4, 8, 10)).astype(np.float32)
with rasterio.open(attention_source, "w", driver="GTiff", width=10, height=8,
                   count=4, dtype="float32", crs="EPSG:32643",
                   transform=TRANSFORM) as dst:
    dst.write(attention_image)
attention_run = subprocess.run(
    [PYTHON, RUNNER, "--model", attention_model, "--input", attention_source,
     "--output", attention_output, "--memory-limit-mib", "128", "--overwrite"],
    capture_output=True, text=True,
)
assert attention_run.returncode == 0, attention_run.stderr
reference = ort.InferenceSession(attention_model, providers=["CPUExecutionProvider"]).run(
    None, {"image": attention_image[np.newaxis]}
)[0][0]
with rasterio.open(attention_output) as raster:
    actual = raster.read()
    assert raster.crs == rasterio.crs.CRS.from_epsg(32643)
    assert raster.transform == TRANSFORM
error = float(np.max(np.abs(actual-reference)))
assert error <= 1e-4
attention_report = json.loads(attention_output.with_suffix(".report.json").read_text())
assert attention_report["strategy"] == "auto-block_attention"
assert attention_report["gdal_cachemax_mib"] == 16
assert attention_report["job_peak_working_set_sampled_bytes"] > 0
assert attention_report["job_peak_private_commit_sampled_bytes"] <= 128*1024*1024
assert attention_report["working_set_sample_count"] > 0
assert attention_report["job_peak_conservative_upper_bound_bytes"] <= 128*1024*1024
attention_reference = OUT / "attention_oracle.tif"
with rasterio.open(attention_output) as existing:
    attention_profile = existing.profile
with rasterio.open(attention_reference, "w", **attention_profile) as oracle:
    oracle.write(reference)
attention_checked = subprocess.run(
    [PYTHON, RUNNER, "--model", attention_model, "--input", attention_source,
     "--output", attention_output, "--reference-raster", attention_reference,
     "--memory-limit-mib", "128", "--overwrite"],
    capture_output=True, text=True,
)
assert attention_checked.returncode == 0, attention_checked.stderr
assert json.loads(attention_output.with_suffix(".report.json").read_text())[
    "numerical_error"]["passed"] is True
print(f"block_attention: max_abs={error:.3g}")

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

unsupported_model = OUT / "unsupported_sigmoid.onnx"
unsupported = onnx.load(ROOT / "models" / "pointwise_dev.onnx")
unsupported.graph.node[1].op_type = "Tanh"
onnx.checker.check_model(unsupported)
onnx.save(unsupported, unsupported_model)
unsupported_output = OUT / "unsupported_scores.tif"
unsupported_output.unlink(missing_ok=True)
rejected = subprocess.run(
    [PYTHON, RUNNER, "--model", unsupported_model, "--input", SOURCE,
     "--output", unsupported_output, "--memory-limit-mib", "128"],
    capture_output=True, text=True,
)
assert rejected.returncode != 0 and not unsupported_output.exists()
unsupported_failure = json.loads(
    unsupported_output.with_suffix(".plan.failure.json").read_text()
)
assert "first unsupported node 1 (Tanh)" in unsupported_failure["error"]
print("unsupported operator: identified before inference")
