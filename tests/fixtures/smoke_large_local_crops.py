"""Check large-local crop oracle on correct and altered scheduled scores."""

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
oracle = ROOT / "tests" / "performance" / "verify_large_local_crops.py"


def invoke(output, report):
    return subprocess.run([sys.executable, oracle, "--model", model, "--input", source,
                           "--candidate", output, "--report", report, "--crop-size", "16"],
                          capture_output=True, text=True)


with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    good = invoke(candidate, directory / "good.json")
    assert good.returncode == 0, good.stderr
    good_report = json.loads((directory / "good.json").read_text())
    assert good_report["passed"] and len(good_report["crops"]) == 3
    with rasterio.open(candidate) as src:
        profile = src.profile
        scores = src.read()
    scores[0, 5, 5] += 0.01
    altered = directory / "altered.tif"
    with rasterio.open(altered, "w", **profile) as dst:
        dst.write(scores)
    bad = invoke(altered, directory / "bad.json")
    assert bad.returncode != 0, bad.stdout + bad.stderr
    bad_report = json.loads((directory / "bad.json").read_text())
    assert not bad_report["passed"]
    assert bad_report["max_absolute_error"] > 0.009
    print("large-local crop oracle: exact and altered outputs passed")
