"""Run a branched local ONNX graph through fixed and automatic GeoTIFF paths."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, ROOT / "tests" / "fixtures" / "smoke_local_graph.py"],
               check=True, capture_output=True, text=True)
MODEL = ROOT / "runs" / "local_graph_smoke" / "branched.onnx"
OUT = ROOT / "runs" / "generic_local_raster_smoke"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = OUT / "input.tif"
IMAGE = np.random.default_rng(9301).standard_normal((3, 37, 53), dtype=np.float32)
with rasterio.open(SOURCE, "w", driver="GTiff", width=53, height=37,
                   count=3, dtype="float32", crs="EPSG:32643",
                   transform=from_origin(500000, 2000000, 10, 10)) as dst:
    dst.write(IMAGE)
    dst.update_tags(test_origin="generic-local-smoke")
reference = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"]).run(
    None, {"image": IMAGE[np.newaxis]})[0][0]

for runner, name, extra in (
    ("run_raster_local.py", "fixed", ["--tile-size", "16", "--cut-after", "3"]),
    ("run_raster_local.py", "disk", ["--tile-size", "16", "--cut-after", "3",
                                      "--storage-mode", "disk"]),
    ("run_raster_auto.py", "automatic", ["--max-tile-size", "32"]),
):
    output = OUT / f"{name}.tif"
    output.unlink(missing_ok=True)
    result = subprocess.run([sys.executable, ROOT / "scripts" / runner,
                             "--model", MODEL, "--input", SOURCE,
                             "--output", output, "--memory-limit-mib", "128",
                             "--threads", "2", *extra], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with rasterio.open(output) as dst:
        actual = dst.read()
        assert dst.crs == rasterio.crs.CRS.from_epsg(32643)
        assert dst.tags()["test_origin"] == "generic-local-smoke"
    error = float(np.max(np.abs(actual-reference)))
    assert error <= 1e-4, (name, error)
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["memory_limit_enforced"] is True
    assert report["job_peak_private_commit_sampled_bytes"] <= 128*1024*1024
    assert report["halo_pixels"] == [2, 2]
    assert report["schedule"] == "local_graph"
    if name == "disk":
        assert report["boundary_storage"] == "disk"
        assert report["boundary_spill_write_bytes"] > 0
    if name == "automatic":
        assert report["strategy"] == "auto-local_graph"
    print(f"{name}: max_abs={error:.3g}, tile_count={report['tile_count']}")

shrinking = onnx.load(MODEL)
shrinking.graph.node[0].attribute[0].ints[:] = [0, 0, 0, 0]
onnx.save(shrinking, OUT / "shrinking.onnx")
bad_output = OUT / "shrinking.tif"
bad_output.unlink(missing_ok=True)
bad = subprocess.run([sys.executable, ROOT / "scripts" / "run_raster_local.py",
                      "--model", OUT / "shrinking.onnx", "--input", SOURCE,
                      "--output", bad_output, "--tile-size", "16", "--cut-after", "3",
                      "--memory-limit-mib", "128"], capture_output=True, text=True)
assert bad.returncode != 0 and not bad_output.exists()
assert "output differs from input raster dimensions" in bad.stderr
