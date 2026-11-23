"""Check derived local halos against whole-image ONNX Runtime scores."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from local_graph import LocalPlan, analyze_local


unet = analyze_local(onnx.load(ROOT / "models" / "evaluation" / "unet_dev.onnx"))
assert unet == LocalPlan(4, 4, 2, 2,
    ("Conv", "Relu", "MaxPool", "Conv", "Relu", "ConvTranspose", "Add", "Conv"))
try:
    analyze_local(onnx.load(ROOT / "models" / "evaluation" / "global_mean_dev.onnx"))
except ValueError as exc:
    assert "no modeled local dependency" in str(exc)
else:
    raise AssertionError("spatial global mean was accepted as local")

rng = np.random.default_rng(9284)
weights = [numpy_helper.from_array(rng.standard_normal(shape, dtype=np.float32)*0.1,
                                   name=name) for name, shape in (
    ("w1", (4, 3, 3, 3)), ("w2", (4, 4, 3, 3)), ("w3", (2, 8, 1, 1)))]
nodes = [
    helper.make_node("Conv", ["image", "w1"], ["conv1"], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["conv1"], ["active"]),
    helper.make_node("Conv", ["active", "w2"], ["conv2"], pads=[1, 1, 1, 1]),
    helper.make_node("Concat", ["conv1", "conv2"], ["joined"], axis=1),
    helper.make_node("Conv", ["joined", "w3"], ["logits"]),
]
graph = helper.make_graph(nodes, "two-local-convs-with-branch",
    [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "h", "w"])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, "h", "w"])],
    initializer=weights)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 13
model_path = ROOT / "runs" / "local_graph_smoke" / "branched.onnx"
model_path.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, model_path)
plan = analyze_local(model)
assert plan.halo_y == plan.halo_x == 2
assert plan.alignment_y == plan.alignment_x == 1
split_dir = model_path.parent / "split"
split_dir.mkdir(exist_ok=True)
subprocess.run([ROOT / "target" / "debug" / "raster-compiler.exe", "split",
                model_path, "--cut-after", "3", "--height", "37", "--width", "53",
                "--out-dir", split_dir], check=True, capture_output=True, text=True)

session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
image = rng.standard_normal((3, 37, 53), dtype=np.float32)
reference = session.run(None, {"image": image[np.newaxis]})[0][0]
for tile in (8, 12):
    actual = np.empty_like(reference)
    for row in range(0, image.shape[1], tile):
        for col in range(0, image.shape[2], tile):
            core_h = min(tile, image.shape[1]-row)
            core_w = min(tile, image.shape[2]-col)
            top = max(0, row-plan.halo_y)
            left = max(0, col-plan.halo_x)
            bottom = min(image.shape[1], row+core_h+plan.halo_y)
            right = min(image.shape[2], col+core_w+plan.halo_x)
            scores = session.run(None, {"image": image[np.newaxis, :, top:bottom, left:right]})[0][0]
            actual[:, row:row+core_h, col:col+core_w] = scores[
                :, row-top:row-top+core_h, col-left:col-left+core_w]
    error = float(np.max(np.abs(actual-reference)))
    assert error <= 1e-4, (tile, error)
    print(f"branched Conv/Concat tile {tile}: max_abs={error:.3g}")

nonlocal_model = onnx.ModelProto()
nonlocal_model.CopyFrom(model)
nonlocal_model.graph.node[3].attribute[0].i = 2
try:
    analyze_local(nonlocal_model)
except ValueError as exc:
    assert "channel-axis Concat" in str(exc)
else:
    raise AssertionError("spatial Concat was accepted")
