"""Diff exact U-Net windows against whole-image ONNX Runtime inference."""

import json
import os
from pathlib import Path
import subprocess

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import onnx
import onnxruntime as ort
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
MODEL = ROOT / "models" / "evaluation" / "unet_dev.onnx"
RUNNER = ROOT / "scripts" / "run_raster_unet.py"
OUT = ROOT / "runs" / "unet_raster_smoke"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = OUT / "input.tif"
OUTPUT = OUT / "scores.tif"
TRANSFORM = from_origin(500000, 2000000, 10, 10)
IMAGE = np.random.default_rng(20261123).normal(size=(3, 38, 54)).astype(np.float32)

with rasterio.open(SOURCE, "w", driver="GTiff", width=54, height=38,
                   count=3, dtype="float32", crs="EPSG:32643",
                   transform=TRANSFORM) as dst:
    dst.write(IMAGE)
    dst.update_tags(test_origin="unet-smoke")

session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
for tile_size, order, scale, offset in (
    (6, None, 1.0, 0.0),
    (8, None, 1.0, 0.0),
    (16, "3,2,1", 0.5, -0.1),
    (24, None, 1.0, 0.0),
):
    ordered = IMAGE if order is None else IMAGE[[2, 1, 0]]
    reference = session.run(None, {"image": (ordered * scale + offset)[np.newaxis]})[0][0]
    preprocessing = ([] if order is None else ["--band-order", order,
                     "--scale", str(scale), "--offset", str(offset)])
    result = subprocess.run([PYTHON, RUNNER, "--model", MODEL, "--input", SOURCE,
                             "--output", OUTPUT, "--tile-size", str(tile_size),
                             "--memory-limit-mib", "128", "--overwrite", *preprocessing],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with rasterio.open(OUTPUT) as raster:
        actual = raster.read()
        assert raster.crs == rasterio.crs.CRS.from_epsg(32643)
        assert raster.transform == TRANSFORM
        assert raster.tags()["test_origin"] == "unet-smoke"
        assert raster.count == 2 and raster.dtypes == ("float32", "float32")
    max_abs = float(np.max(np.abs(actual - reference)))
    assert max_abs <= 1e-4, (tile_size, max_abs)
    report = json.loads(OUTPUT.with_suffix(".report.json").read_text())
    assert report["tile_count"] == ((38+tile_size-1)//tile_size)*((54+tile_size-1)//tile_size)
    assert report["raw_bridge_bytes"] == 0
    assert report["memory_limit_enforced"] is True
    assert report["job_peak_conservative_upper_bound_bytes"] <= 128 * 1024 * 1024
    print(f"U-Net tile {tile_size}: max_abs={max_abs:.3g}, shapes={len(report['worker_shapes'])}")

print("exact U-Net raster tiling passed")

# A supported user model need not use the development fixture's tensor names.
renamed_model = OUT / "renamed_unet.onnx"
renamed = onnx.load(MODEL)
names = {value.name for value in renamed.graph.input}
names.update(value.name for value in renamed.graph.output)
names.update(value.name for value in renamed.graph.initializer)
names.update(name for node in renamed.graph.node for name in (*node.input, *node.output) if name)
mapping = {name: f"user_tensor_{index}" for index, name in enumerate(sorted(names))}
for value in (*renamed.graph.input, *renamed.graph.output, *renamed.graph.value_info):
    if value.name in mapping:
        value.name = mapping[value.name]
for value in renamed.graph.initializer:
    value.name = mapping[value.name]
for node in renamed.graph.node:
    for index, name in enumerate(node.input):
        if name in mapping:
            node.input[index] = mapping[name]
    for index, name in enumerate(node.output):
        if name in mapping:
            node.output[index] = mapping[name]
onnx.checker.check_model(renamed)
onnx.save(renamed, renamed_model)
renamed_session = ort.InferenceSession(renamed_model, providers=["CPUExecutionProvider"])
renamed_input = renamed_session.get_inputs()[0].name
renamed_reference = renamed_session.run(None, {renamed_input: IMAGE[np.newaxis]})[0][0]
renamed_output = OUT / "renamed_scores.tif"
result = subprocess.run([PYTHON, RUNNER, "--model", renamed_model, "--input", SOURCE,
                         "--output", renamed_output, "--tile-size", "16",
                         "--memory-limit-mib", "128", "--overwrite"],
                        capture_output=True, text=True)
assert result.returncode == 0, result.stderr
with rasterio.open(renamed_output) as raster:
    assert np.array_equal(raster.read(), renamed_reference)
print("renamed U-Net tensors: matched whole-image reference")
auto_output = OUT / "renamed_auto_scores.tif"
result = subprocess.run([PYTHON, ROOT / "scripts" / "run_raster_auto.py",
                         "--model", renamed_model, "--input", SOURCE,
                         "--output", auto_output, "--memory-limit-mib", "128",
                         "--max-tile-size", "32", "--overwrite"],
                        capture_output=True, text=True)
assert result.returncode == 0, result.stderr
with rasterio.open(auto_output) as raster:
    assert np.array_equal(raster.read(), renamed_reference)
auto_report = json.loads(auto_output.with_suffix(".report.json").read_text())
assert auto_report["strategy"] == "auto-local_unet"
print("renamed U-Net: automatic schedule matched whole-image reference")

disk_output = OUT / "disk_scores.tif"
result = subprocess.run([PYTHON, RUNNER, "--model", MODEL, "--input", SOURCE,
                         "--output", disk_output, "--tile-size", "16",
                         "--cut-after", "4", "--storage-mode", "disk",
                         "--memory-limit-mib", "128", "--overwrite"],
                        capture_output=True, text=True)
assert result.returncode == 0, result.stderr
with rasterio.open(disk_output) as raster:
    assert np.array_equal(raster.read(), reference)
disk_report = json.loads(disk_output.with_suffix(".report.json").read_text())
assert disk_report["boundary_storage"] == "disk"
assert disk_report["boundary_spill_write_bytes"] > 0
assert disk_report["boundary_spill_write_bytes"] == disk_report["boundary_spill_read_bytes"]
assert disk_report["job_peak_conservative_upper_bound_bytes"] <= 128 * 1024 * 1024
print("disk-spill U-Net tiles: matched whole-image reference")

bad_source = OUT / "odd_width.tif"
with rasterio.open(bad_source, "w", driver="GTiff", width=53, height=38,
                   count=3, dtype="float32", crs="EPSG:32643",
                   transform=TRANSFORM) as dst:
    dst.write(IMAGE[:, :, :53])
bad_output = OUT / "should_not_exist.tif"
bad_output.unlink(missing_ok=True)
failed = subprocess.run([PYTHON, RUNNER, "--model", MODEL, "--input", bad_source,
                         "--output", bad_output, "--tile-size", "8"],
                        capture_output=True, text=True)
assert failed.returncode != 0 and "violate graph alignment" in failed.stderr
assert not bad_output.exists()
print("odd-size whole-image contract: rejected before output")
