"""Exercise whole-image oracle success, numerical failure, and size guard."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import rasterio


ROOT = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, ROOT / "tests" / "fixtures" / "smoke_generic_local_raster.py"],
               check=True, capture_output=True, text=True)
model = ROOT / "runs" / "local_graph_smoke" / "branched.onnx"
source = ROOT / "runs" / "generic_local_raster_smoke" / "input.tif"
candidate = ROOT / "runs" / "generic_local_raster_smoke" / "automatic.tif"
oracle = ROOT / "tests" / "performance" / "whole_image_oracle.py"


def invoke(directory, name, selected, *extra):
    return subprocess.run([sys.executable, oracle, "--model", model, "--input", source,
                           "--candidate", selected, "--reference-output", directory / f"{name}.tif",
                           "--report", directory / f"{name}.json", *extra],
                          capture_output=True, text=True)


with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    good = invoke(directory, "good", candidate)
    assert good.returncode == 0, good.stderr
    good_report = json.loads((directory / "good.json").read_text())
    assert good_report["passed"] and good_report["comparison"]["max_absolute_error"] == 0
    with rasterio.open(directory / "good.tif") as dst:
        assert dst.crs == rasterio.crs.CRS.from_epsg(32643)
        assert dst.tags()["test_origin"] == "generic-local-smoke"
        assert dst.descriptions == ("score_1", "score_2")
    with rasterio.open(candidate) as src:
        profile = src.profile
        scores = src.read()
    scores[0, 0, 0] += 0.01
    altered = directory / "altered.tif"
    with rasterio.open(altered, "w", **profile) as dst:
        dst.write(scores)
    bad = invoke(directory, "bad", altered)
    assert bad.returncode != 0, bad.stdout + bad.stderr
    bad_report = json.loads((directory / "bad.json").read_text())
    assert not bad_report["passed"]
    assert bad_report["comparison"]["max_absolute_error_at"] == [0, 0, 0]
    guard = invoke(directory, "guard", candidate, "--max-pixels", "100")
    assert guard.returncode != 0 and "exceeds max-pixels" in guard.stderr
    assert not (directory / "guard.tif").exists()
    print("whole-image oracle: exact candidate, perturbed score, and size guard passed")
