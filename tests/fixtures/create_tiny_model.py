"""Create a small Conv -> Relu -> Add ONNX model for local smoke testing."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


root = Path(__file__).resolve().parents[2]
output = root / "models" / "tiny_skip.onnx"
output.parent.mkdir(exist_ok=True)

input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 8, 8])
output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, 8, 8])
weights = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32) / 9, "kernel")

graph = helper.make_graph(
    [
        helper.make_node("Conv", ["image", "kernel"], ["features"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["features"], ["activated"]),
        helper.make_node("Add", ["activated", "image"], ["logits"]),
    ],
    "tiny_skip",
    [input_info],
    [output_info],
    [weights],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, output)
print(output)
