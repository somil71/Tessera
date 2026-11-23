"""Generate the frozen development and held-out ONNX model fixtures."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261122)


def weights(name, shape):
    values = RNG.normal(0.0, 0.05, shape).astype(np.float32)
    return numpy_helper.from_array(values, name)


def save(name, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = OUT / f"{name}.onnx"
    onnx.save(model, path)
    print(path)


image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "height", "width"])
unet_output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, "height", "width"])
save(
    "unet_dev",
    [
        helper.make_node("Conv", ["image", "enc_kernel"], ["enc"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["enc"], ["skip"]),
        helper.make_node("MaxPool", ["skip"], ["pooled"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["pooled", "bottleneck_kernel"], ["bottleneck"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["bottleneck"], ["deep"]),
        helper.make_node("ConvTranspose", ["deep", "up_kernel"], ["up"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Add", ["up", "skip"], ["merged"]),
        helper.make_node("Conv", ["merged", "head_kernel"], ["logits"]),
    ],
    [image],
    [unet_output],
    [
        weights("enc_kernel", (8, 3, 3, 3)),
        weights("bottleneck_kernel", (16, 8, 3, 3)),
        weights("up_kernel", (16, 8, 2, 2)),
        weights("head_kernel", (2, 8, 1, 1)),
    ],
)

global_output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, "height", "width"])
save(
    "global_mean_dev",
    [
        helper.make_node("ReduceMean", ["image", "axes"], ["mean"], keepdims=1),
        helper.make_node("Sub", ["image", "mean"], ["centered"]),
        helper.make_node("Conv", ["centered", "global_kernel"], ["logits"]),
    ],
    [image],
    [global_output],
    [numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes"), weights("global_kernel", (2, 3, 1, 1))],
)

attention_output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4, "height", "width"])
attention_input = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 4, "height", "width"])
save(
    "attention_holdout",
    [
        helper.make_node("Shape", ["image"], ["original_shape"]),
        helper.make_node("Reshape", ["image", "flat_shape"], ["channels_by_pixel"]),
        helper.make_node("Transpose", ["channels_by_pixel"], ["pixels_by_channel"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["pixels_by_channel", "channels_by_pixel"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["probabilities"], axis=-1),
        helper.make_node("MatMul", ["probabilities", "pixels_by_channel"], ["attended"]),
        helper.make_node("Transpose", ["attended"], ["attended_channels"], perm=[0, 2, 1]),
        helper.make_node("Reshape", ["attended_channels", "original_shape"], ["logits"]),
    ],
    [attention_input],
    [attention_output],
    [numpy_helper.from_array(np.array([1, 4, -1], dtype=np.int64), "flat_shape")],
)
