"""Exercise live ConstantOfShape padding through a tiled GeoTIFF schedule."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from local_graph import analyze_local
from run_raster_auto import model_family

out = ROOT / "runs" / "smoke_cos"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1024)
weight = numpy_helper.from_array(
    rng.standard_normal((2, 3, 3, 3), dtype=np.float32)*0.1, name="weight")
reshape_to_matrix = numpy_helper.from_array(np.array([1, 2], dtype=np.int64),
                                              name="reshape_to_matrix")
reshape_to_vector = numpy_helper.from_array(np.array([2], dtype=np.int64),
                                              name="reshape_to_vector")
nodes = [
    helper.make_node("Identity", ["image"], ["image_passthrough"]),
    helper.make_node("Constant", [], ["shape_two"],
                     value=numpy_helper.from_array(np.array([2], dtype=np.int64))),
    helper.make_node("ConstantOfShape", ["shape_two"], ["zeros"],
                     value=helper.make_tensor("zero", TensorProto.INT64, [1], [0])),
    helper.make_node("Reshape", ["zeros", "reshape_to_matrix"], ["zero_matrix"]),
    helper.make_node("Transpose", ["zero_matrix"], ["zero_column"], perm=[1, 0]),
    helper.make_node("Reshape", ["zero_column", "reshape_to_vector"], ["zero_vector"]),
    helper.make_node("Constant", [], ["ones"],
                     value=numpy_helper.from_array(np.array([1, 1], dtype=np.int64))),
    helper.make_node("Concat", ["zero_vector", "ones", "zero_vector", "ones"],
                     ["pads"], axis=0),
    helper.make_node("Pad", ["image_passthrough", "pads"], ["padded"], mode="constant"),
    helper.make_node("Conv", ["padded", "weight"], ["conv"]),
    helper.make_node("Relu", ["conv"], ["logits"]),
]
model = helper.make_model(
    helper.make_graph(nodes, "live-constant-padding",
                      [helper.make_tensor_value_info("image", TensorProto.FLOAT,
                                                     [1, 3, "h", "w"])],
                      [helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                                     [1, 2, "h", "w"])],
                      initializer=[weight, reshape_to_matrix, reshape_to_vector]),
    opset_imports=[helper.make_opsetid("", 18)],
)
model.ir_version = 13
onnx.checker.check_model(model)
model_path = out / "model.onnx"
onnx.save(model, model_path)

plan = analyze_local(model)
assert (plan.halo_y, plan.halo_x) == (1, 1)
assert (plan.alignment_y, plan.alignment_x) == (1, 1)
assert model_family(model_path, model)[:3] == ("local_graph", 3, 2)

image = rng.standard_normal((3, 37, 53), dtype=np.float32)
source = out / "input.tif"
with rasterio.open(source, "w", driver="GTiff", width=53, height=37,
                   count=3, dtype="float32", crs="EPSG:32643",
                   transform=from_origin(500000, 2000000, 10, 10)) as raster:
    raster.write(image)
reference = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]).run(
    None, {"image": image[np.newaxis]})[0][0]

output = out / "scores.tif"
output.unlink(missing_ok=True)
result = subprocess.run(
    [sys.executable, ROOT / "scripts" / "run_raster_local.py",
     "--model", model_path, "--input", source, "--output", output,
     "--tile-size", "16", "--cut-after", "0", "--memory-limit-mib", "128",
     "--threads", "2"], capture_output=True, text=True)
assert result.returncode == 0, result.stderr
with rasterio.open(output) as raster:
    actual = raster.read()
    assert raster.crs == rasterio.crs.CRS.from_epsg(32643)
    assert raster.transform == from_origin(500000, 2000000, 10, 10)
assert np.max(np.abs(actual-reference)) <= 1e-4
report = json.loads(output.with_suffix(".report.json").read_text())
assert report["tile_count"] > 1
assert report["memory_limit_enforced"] is True
automatic = out / "automatic.tif"
automatic.unlink(missing_ok=True)
result = subprocess.run(
    [sys.executable, ROOT / "scripts" / "run_raster_auto.py",
     "--model", model_path, "--input", source, "--output", automatic,
     "--max-tile-size", "16", "--memory-limit-mib", "128", "--threads", "2"],
    capture_output=True, text=True)
assert result.returncode == 0, result.stderr
with rasterio.open(automatic) as raster:
    actual_auto = raster.read()
    assert raster.crs == rasterio.crs.CRS.from_epsg(32643)
    assert raster.transform == from_origin(500000, 2000000, 10, 10)
assert np.max(np.abs(actual_auto-reference)) <= 1e-4
auto_report = json.loads(automatic.with_suffix(".report.json").read_text())
assert auto_report["strategy"] == "auto-local_graph"
assert auto_report["job_peak_committed_bytes"] <= 128*1024*1024

bad = onnx.ModelProto()
bad.CopyFrom(model)
bad.graph.initializer.append(numpy_helper.from_array(
    np.array([0, 1, 1, 1, 0, 0, 1, 1], dtype=np.int64), name="channel_pads"))
next(node for node in bad.graph.node if node.op_type == "Pad").input[1] = "channel_pads"
try:
    analyze_local(bad)
except ValueError as error:
    assert "spatial-only" in str(error)
else:
    raise AssertionError("channel padding must fail local analysis")
print("live ConstantOfShape/Pad tiled scores matched whole-image ONNX Runtime")
