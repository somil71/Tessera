"""Check windowed GeoTIFF assembly and georeferencing against whole-model output."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = ROOT / "scripts" / "run_raster_fixed.py"
MODEL = ROOT / "models" / "pointwise_dev.onnx"
OUT = ROOT / "runs" / "raster_smoke"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = OUT / "input.tif"
OUTPUT = OUT / "scores.tif"
TRANSFORM = from_origin(500000, 2000000, 10, 10)
IMAGE = np.random.default_rng(3101).standard_normal((3, 37, 53)).astype(np.float32)


with rasterio.open(
    SOURCE,
    "w",
    driver="GTiff",
    width=53,
    height=37,
    count=3,
    dtype="float32",
    crs="EPSG:32643",
    transform=TRANSFORM,
    tiled=True,
    blockxsize=16,
    blockysize=16,
) as dst:
    dst.write(IMAGE)
    dst.update_tags(test_origin="phase4-smoke")

session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
for tile_size, band_order, scale, offset in (
    (16, None, 1.0, 0.0),
    (11, "3,2,1", 0.5, -0.1),
):
    ordered = IMAGE if band_order is None else IMAGE[[2, 1, 0]]
    reference = session.run(None, {"image": (ordered * scale + offset)[np.newaxis, ...]})[0][0]
    preprocess = (
        []
        if band_order is None
        else ["--band-order", band_order, "--scale", str(scale), "--offset", str(offset)]
    )
    subprocess.run(
        [
            PYTHON,
            RUNNER,
            "--model",
            MODEL,
            "--input",
            SOURCE,
            "--output",
            OUTPUT,
            "--cut-after",
            "0",
            "--tile-size",
            str(tile_size),
            "--overwrite",
            *preprocess,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with rasterio.open(OUTPUT) as result:
        actual = result.read()
        assert result.count == 2
        assert result.crs == rasterio.crs.CRS.from_epsg(32643)
        assert result.transform == TRANSFORM
        assert result.tags()["test_origin"] == "phase4-smoke"
    absolute_error = float(np.max(np.abs(actual - reference)))
    assert absolute_error <= 1e-4, (tile_size, absolute_error)
    report = json.loads(OUTPUT.with_suffix(".report.json").read_text())
    assert report["tile_count"] == ((37 + tile_size - 1) // tile_size) * (
        (53 + tile_size - 1) // tile_size
    )
    assert report["max_child_peak_commit_bytes"] > 0
    assert report["worker_mode"] == "pipe"
    assert report["raw_bridge_bytes"] == 0
    assert report["combined_process_peak_upper_bound_bytes"] >= report[
        "max_child_peak_commit_bytes"
    ]
    print(f"tile {tile_size}: {report['tile_count']} windows, max_abs={absolute_error:.3g}")

bad_source = OUT / "input_with_nodata.tif"
bad_image = IMAGE.copy()
bad_image[0, 0, 0] = -9999.0
with rasterio.open(
    bad_source,
    "w",
    driver="GTiff",
    width=53,
    height=37,
    count=3,
    dtype="float32",
    crs="EPSG:32643",
    transform=TRANSFORM,
    nodata=-9999.0,
) as dst:
    dst.write(bad_image)
bad_output = OUT / "should_not_exist.tif"
bad_output.unlink(missing_ok=True)
failed = subprocess.run(
    [
        PYTHON,
        RUNNER,
        "--model",
        MODEL,
        "--input",
        bad_source,
        "--output",
        bad_output,
        "--cut-after",
        "0",
        "--tile-size",
        "16",
    ],
    capture_output=True,
    text=True,
)
assert failed.returncode != 0 and "no-data pixel" in failed.stderr
assert not bad_output.exists()
print("no-data input: rejected without publishing an output")

unsupported_output = OUT / "unsupported_should_not_exist.tif"
unsupported_output.unlink(missing_ok=True)
unsupported = subprocess.run(
    [
        PYTHON,
        RUNNER,
        "--model",
        ROOT / "models" / "evaluation" / "unet_dev.onnx",
        "--input",
        SOURCE,
        "--output",
        unsupported_output,
        "--cut-after",
        "4",
        "--tile-size",
        "16",
    ],
    capture_output=True,
    text=True,
)
assert unsupported.returncode != 0 and "not a supported pointwise" in unsupported.stderr
assert not unsupported_output.exists()
print("spatial U-Net: rejected before raster execution")
