"""Compare blockwise attention GeoTIFF logits to whole-image ONNX Runtime."""

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
RUNNER = ROOT / "scripts" / "run_raster_attention.py"
MODEL = ROOT / "models" / "evaluation" / "attention_holdout.onnx"
OUT = ROOT / "runs" / "attention_raster_smoke"
OUT.mkdir(parents=True, exist_ok=True)
TRANSFORM = from_origin(500000, 2000000, 10, 10)
session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])

for height, width, seed, order, scale, offset in (
    (8, 10, 4101, "4,3,2,1", 0.5, -0.1),
    (64, 64, 4102, None, 1.0, 0.0),
    (96, 96, 4103, None, 1.0, 0.0),
):
    image = np.random.default_rng(seed).normal(size=(4, height, width)).astype(np.float32)
    source = OUT / f"input_{height}_{width}.tif"
    output = OUT / f"scores_{height}_{width}.tif"
    with rasterio.open(source, "w", driver="GTiff", width=width, height=height,
                       count=4, dtype="float32", crs="EPSG:32643",
                       transform=TRANSFORM) as dst:
        dst.write(image)
        dst.update_tags(test_origin="attention-smoke")
    ordered = image if order is None else image[[3, 2, 1, 0]]
    reference = session.run(None, {
        "image": (ordered*scale+offset)[np.newaxis]
    })[0][0]
    preprocessing = ([] if order is None else ["--band-order", order,
                     "--scale", str(scale), "--offset", str(offset)])
    result = subprocess.run(
        [PYTHON, RUNNER, "--model", MODEL, "--input", source,
         "--output", output, "--memory-limit-mib", "128",
         "--overwrite", *preprocessing], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    with rasterio.open(output) as raster:
        actual = raster.read()
        assert raster.count == 4 and raster.dtypes == ("float32",)*4
        assert raster.crs == rasterio.crs.CRS.from_epsg(32643)
        assert raster.transform == TRANSFORM
        assert raster.tags()["test_origin"] == "attention-smoke"
    max_abs = float(np.max(np.abs(actual-reference)))
    eligible = np.abs(reference) >= 1e-3
    max_rel = (float(np.max(np.abs(actual[eligible]-reference[eligible]) /
                            np.abs(reference[eligible]))) if np.any(eligible) else 0.0)
    assert max_abs <= 1e-4 and max_rel <= 1e-3, (height, width, max_abs, max_rel)
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["memory_limit_enforced"] is True
    assert report["job_peak_conservative_upper_bound_bytes"] <= 128*1024*1024
    assert report["peak_temp_storage_bytes"] > 0
    print(f"attention {height}x{width}: max_abs={max_abs:.3g}, max_rel={max_rel:.3g}")

print("blockwise attention raster inference passed")

huge_source = OUT / "infeasible_8192.tif"
huge_output = OUT / "infeasible_should_not_exist.tif"
huge_output.unlink(missing_ok=True)
with rasterio.open(huge_source, "w", driver="GTiff", width=8192, height=8192,
                   count=4, dtype="float32", crs="EPSG:32643",
                   transform=TRANSFORM, tiled=True, blockxsize=256,
                   blockysize=256, compress="deflate"):
    pass
infeasible = subprocess.run(
    [PYTHON, RUNNER, "--model", MODEL, "--input", huge_source,
     "--output", huge_output, "--memory-limit-mib", "128"],
    capture_output=True, text=True,
)
assert infeasible.returncode != 0
assert not huge_output.exists()
failure = json.loads(huge_output.with_suffix(".failure.json").read_text())
assert failure["estimated_attention_seconds"] > 30*60
assert failure["output_published"] is False
print("8192x8192 attention: reported infeasible before raster staging")

bad_source = OUT / "input_with_nodata.tif"
bad_output = OUT / "nodata_should_not_exist.tif"
bad_output.unlink(missing_ok=True)
bad_image = np.zeros((4, 8, 10), dtype=np.float32)
bad_image[0, 0, 0] = -9999.0
with rasterio.open(bad_source, "w", driver="GTiff", width=10, height=8,
                   count=4, dtype="float32", crs="EPSG:32643",
                   transform=TRANSFORM, nodata=-9999.0) as dst:
    dst.write(bad_image)
bad = subprocess.run(
    [PYTHON, RUNNER, "--model", MODEL, "--input", bad_source,
     "--output", bad_output, "--memory-limit-mib", "128"],
    capture_output=True, text=True,
)
assert bad.returncode != 0 and "no-data pixel" in bad.stderr
assert not bad_output.exists()
assert json.loads(bad_output.with_suffix(".failure.json").read_text())["output_published"] is False
print("attention no-data: rejected without output")
