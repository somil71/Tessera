"""Create an exploratory pixelwise model for raster I/O tests."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "pointwise_dev.onnx"
OUT.parent.mkdir(parents=True, exist_ok=True)

weights = np.array(
    [
        [[[0.5]], [[-0.25]], [[0.1]]],
        [[[0.2]], [[0.3]], [[-0.4]]],
    ],
    dtype=np.float32,
)
graph = helper.make_graph(
    [
        helper.make_node("Conv", ["image", "kernel"], ["features"]),
        helper.make_node("Relu", ["features"], ["logits"]),
    ],
    "pointwise_dev",
    [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "height", "width"])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, "height", "width"])],
    [numpy_helper.from_array(weights, "kernel")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(OUT)
