"""Compare two-pass global mean raster logits to whole-image ONNX Runtime."""

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
RUNNER = ROOT / "scripts" / "run_raster_global_mean.py"
MODEL = ROOT / "models" / "evaluation" / "global_mean_dev.onnx"
OUT = ROOT / "runs" / "global_mean_raster_smoke"
OUT.mkdir(parents=True, exist_ok=True)
TRANSFORM = from_origin(500000, 2000000, 10, 10)
session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])

for height, width, tile_size, bands, scale, offset in (
    (33, 47, 8, None, 1.0, 0.0),
    (33, 47, 13, "3,2,1", 0.5, -0.1),
    (96, 80, 24, None, 1.0, 0.0),
):
    image = np.random.default_rng(3102 + height + tile_size).normal(
        size=(3, height, width)
    ).astype(np.float32)
    source = OUT / f"input_{height}_{width}.tif"
    output = OUT / f"scores_{height}_{width}_{tile_size}.tif"
    with rasterio.open(source, "w", driver="GTiff", width=width, height=height,
                       count=3, dtype="float32", crs="EPSG:32643",
                       transform=TRANSFORM) as dst:
        dst.write(image)
        dst.update_tags(test_origin="global-mean-smoke")
    ordered = image if bands is None else image[[2, 1, 0]]
    reference = session.run(None, {
        "image": (ordered * scale + offset)[np.newaxis]
    })[0][0]
    preprocessing = ([] if bands is None else ["--band-order", bands,
                     "--scale", str(scale), "--offset", str(offset)])
    result = subprocess.run(
        [PYTHON, RUNNER, "--model", MODEL, "--input", source,
         "--output", output, "--tile-size", str(tile_size),
         "--memory-limit-mib", "128", "--overwrite", *preprocessing],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    with rasterio.open(output) as raster:
        actual = raster.read()
        assert raster.count == 2 and raster.dtypes == ("float32", "float32")
        assert raster.crs == rasterio.crs.CRS.from_epsg(32643)
        assert raster.transform == TRANSFORM
        assert raster.tags()["test_origin"] == "global-mean-smoke"
    max_abs = float(np.max(np.abs(actual - reference)))
    relative = np.abs(actual - reference) / np.maximum(np.abs(reference), 1e-30)
    eligible = np.abs(reference) >= 1e-3
    max_rel = float(np.max(relative[eligible])) if np.any(eligible) else 0.0
    assert max_abs <= 1e-4 and max_rel <= 1e-3, (max_abs, max_rel)
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["strategy"] == "two-pass-global-mean"
    assert report["memory_limit_enforced"] is True
    assert report["job_peak_conservative_upper_bound_bytes"] <= 128*1024*1024
    assert report["reduction_windows"] == ((height+tile_size-1)//tile_size)*(
        (width+tile_size-1)//tile_size)
    print(f"global mean {height}x{width}, tile {tile_size}: "
          f"max_abs={max_abs:.3g}, max_rel={max_rel:.3g}")

print("two-pass global mean raster inference passed")

bad_source = OUT / "input_with_nodata.tif"
bad_output = OUT / "should_not_exist.tif"
bad_output.unlink(missing_ok=True)
bad_image = image.copy()
bad_image[0, 0, 0] = -9999.0
with rasterio.open(bad_source, "w", driver="GTiff", width=width, height=height,
                   count=3, dtype="float32", crs="EPSG:32643",
                   transform=TRANSFORM, nodata=-9999.0) as dst:
    dst.write(bad_image)
rejected = subprocess.run(
    [PYTHON, RUNNER, "--model", MODEL, "--input", bad_source,
     "--output", bad_output, "--tile-size", "24",
     "--memory-limit-mib", "128"], capture_output=True, text=True,
)
assert rejected.returncode != 0 and "no-data pixel" in rejected.stderr
assert not bad_output.exists()
failure = json.loads(bad_output.with_suffix(".failure.json").read_text())
assert failure["output_published"] is False
print("global reduction no-data: rejected before output")
