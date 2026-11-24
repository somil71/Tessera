"""Check streamed per-band preprocessing, metadata, and rejection paths."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare_raster.py"
with tempfile.TemporaryDirectory() as temp:
    directory = Path(temp)
    source = directory / "source.tif"
    image = np.arange(3*17*23, dtype=np.float32).reshape(3, 17, 23) / 100
    with rasterio.open(source, "w", driver="GTiff", width=23, height=17,
                       count=3, dtype="float32", crs="EPSG:32643",
                       transform=from_origin(500000, 2000000, 10, 10)) as dst:
        dst.write(image)
        dst.update_tags(origin="preprocessing-smoke")
    spec = directory / "spec.json"
    spec.write_text(json.dumps({"bands": [
        {"source": 3, "scale": 0.5, "offset": -1, "clip_min": 0, "clip_max": 4, "name": "third"},
        {"source": 1, "scale": -2, "offset": 2, "name": "first"},
    ]}), encoding="utf-8")
    output = directory / "prepared.tif"
    command = [sys.executable, SCRIPT, "--input", source, "--output", output,
               "--spec", spec, "--window-size", "7", "--memory-limit-mib", "128"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with rasterio.open(output) as got:
        expected = np.stack([np.clip(image[2]*0.5-1, 0, 4), image[0]*-2+2])
        np.testing.assert_allclose(got.read(), expected, atol=1e-6, rtol=0)
        assert got.crs == rasterio.crs.CRS.from_epsg(32643)
        assert got.transform == from_origin(500000, 2000000, 10, 10)
        assert got.tags()["origin"] == "preprocessing-smoke"
        assert got.descriptions == ("third", "first")
    report = json.loads(output.with_suffix(".preprocess.json").read_text())
    assert report["windows"] == 12 and report["memory_limit_enforced"]
    assert report["job_peak_private_commit_sampled_bytes"] <= 128*1024*1024
    invalid = directory / "invalid.json"
    invalid.write_text('{"bands":[{"source":4}]}', encoding="utf-8")
    failed = subprocess.run([*command[:], "--report", directory / "bad.json"],
                            capture_output=True, text=True)
    assert failed.returncode != 0 and "new paths" in failed.stderr
    bad_command = [sys.executable, SCRIPT, "--input", source,
                   "--output", directory / "bad.tif", "--spec", invalid,
                   "--memory-limit-mib", "128"]
    failed = subprocess.run(bad_command, capture_output=True, text=True)
    assert failed.returncode != 0 and "valid 1-based band index" in failed.stderr
    assert not (directory / "bad.tif").exists()
    nonfinite = directory / "nonfinite.tif"
    bad_image = image.copy()
    bad_image[2, 8, 8] = np.nan
    with rasterio.open(nonfinite, "w", driver="GTiff", width=23, height=17,
                       count=3, dtype="float32", crs="EPSG:32643",
                       transform=from_origin(500000, 2000000, 10, 10)) as dst:
        dst.write(bad_image)
    failed = subprocess.run([sys.executable, SCRIPT, "--input", nonfinite,
                             "--output", directory / "nonfinite-output.tif",
                             "--spec", spec, "--window-size", "7",
                             "--memory-limit-mib", "128"], capture_output=True, text=True)
    assert failed.returncode != 0 and "non-finite source pixel" in failed.stderr
    assert not (directory / "nonfinite-output.tif").exists()
    print("streamed preprocessing: values, metadata, cap, and rejection passed")
