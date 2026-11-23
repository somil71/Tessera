"""Compare expanded pointwise ONNX operators against whole-image inference."""

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
OUT = ROOT / "runs" / "pointwise_operator_smoke"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = OUT / "pointwise_bn_sigmoid_mul.onnx"
SOURCE = OUT / "input.tif"
IMAGE = np.random.default_rng(8217).standard_normal((3, 37, 53), dtype=np.float32)

weights = np.random.default_rng(8218)
initializers = [numpy_helper.from_array(value, name=name) for name, value in (
    ("w1", weights.standard_normal((3, 3, 1, 1), dtype=np.float32)*0.1),
    ("scale", np.array([1.1, 0.9, 1.2], dtype=np.float32)),
    ("bias", np.array([0.1, -0.2, 0.3], dtype=np.float32)),
    ("mean", np.array([0.2, 0.0, -0.1], dtype=np.float32)),
    ("variance", np.array([0.8, 1.3, 0.9], dtype=np.float32)),
    ("factor", np.array([[[[1.0]], [[0.7]], [[1.2]]]], dtype=np.float32)),
    ("w2", weights.standard_normal((2, 3, 1, 1), dtype=np.float32)*0.1),
)]
nodes = [
    helper.make_node("Conv", ["image", "w1"], ["conv"]),
    helper.make_node("BatchNormalization", ["conv", "scale", "bias", "mean", "variance"],
                     ["normalized"], epsilon=1e-5),
    helper.make_node("Sigmoid", ["normalized"], ["gated"]),
    helper.make_node("Mul", ["gated", "factor"], ["scaled"]),
    helper.make_node("Conv", ["scaled", "w2"], ["logits"]),
]
graph = helper.make_graph(nodes, "pointwise-expanded",
    [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "h", "w"])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, "h", "w"])],
    initializer=initializers)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 13
onnx.checker.check_model(model)
onnx.save(model, MODEL)

with rasterio.open(SOURCE, "w", driver="GTiff", width=53, height=37,
                   count=3, dtype="float32", crs="EPSG:32643",
                   transform=from_origin(500000, 2000000, 10, 10)) as dst:
    dst.write(IMAGE)
    dst.update_tags(test_origin="expanded-pointwise")

reference = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"]).run(
    None, {"image": IMAGE[np.newaxis]})[0][0]
for runner, name, extra in (
    ("run_raster_fixed.py", "fixed", ["--cut-after", "2", "--tile-size", "16"]),
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
        assert dst.tags()["test_origin"] == "expanded-pointwise"
    error = float(np.max(np.abs(actual-reference)))
    assert error <= 1e-4, (name, error)
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["memory_limit_enforced"] is True
    assert report["job_peak_private_commit_sampled_bytes"] <= 128*1024*1024
    print(f"{name}: max_abs={error:.3g}")

# Training-mode BatchNormalization is not a valid independent-pixel inference schedule.
training = onnx.load(MODEL)
training.graph.node[1].attribute.add().CopyFrom(helper.make_attribute("training_mode", 1))
onnx.save(training, OUT / "training_bn.onnx")
sys.path.insert(0, str(ROOT / "scripts"))
from run_raster_fixed import model_contract
try:
    model_contract(OUT / "training_bn.onnx", 16)
except ValueError as exc:
    assert "BatchNormalization is not supported inference mode" in str(exc)
else:
    raise AssertionError("training-mode BatchNormalization was accepted")
