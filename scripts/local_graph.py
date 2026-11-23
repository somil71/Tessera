"""Conservative spatial dependency analysis for local NCHW ONNX graphs.

This module derives an aligned input halo. It deliberately rejects operators
whose dependency or output phase has not been modeled; it does not execute a
raster or by itself establish a supported inference contract.
"""

from dataclasses import dataclass
import math

import onnx
from onnx import TensorProto, helper


POINTWISE = {"Relu", "Identity", "Sigmoid", "BatchNormalization",
             "Add", "Sub", "Mul"}


@dataclass(frozen=True)
class LocalPlan:
    halo_y: int
    halo_x: int
    alignment_y: int
    alignment_x: int
    operators: tuple[str, ...]


def _attrs(node):
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def _pair(value, label):
    result = tuple(int(item) for item in value)
    if len(result) != 2 or any(item < 1 for item in result):
        raise ValueError(f"{label} must contain two positive dimensions")
    return result


def _spatial_op(node, initializers):
    attrs = _attrs(node)
    op = node.op_type
    if attrs.get("auto_pad", b"NOTSET") != b"NOTSET":
        raise ValueError(f"{op} auto_pad is not modeled")
    pads = tuple(attrs.get("pads", [0, 0, 0, 0]))
    if len(pads) != 4 or any(item < 0 for item in pads):
        raise ValueError(f"{op} pads must contain four nonnegative values")
    strides = _pair(attrs.get("strides", [1, 1]), f"{op} strides")
    dilations = _pair(attrs.get("dilations", [1, 1]), f"{op} dilations")
    if op in {"Conv", "ConvTranspose"}:
        if len(node.input) < 2 or node.input[1] not in initializers:
            raise ValueError(f"{op} requires a constant kernel")
        weight = initializers[node.input[1]]
        if len(weight.dims) != 4:
            raise ValueError(f"{op} requires a 2D kernel")
        kernel = _pair(attrs.get("kernel_shape", weight.dims[2:]), f"{op} kernel")
        if tuple(weight.dims[2:]) != kernel:
            raise ValueError(f"{op} kernel declaration differs from weights")
    else:
        kernel = _pair(attrs.get("kernel_shape", []), "MaxPool kernel")
        if attrs.get("ceil_mode", 0) != 0 or len(node.output) != 1:
            raise ValueError("MaxPool ceil mode or index output is not modeled")
    if op == "ConvTranspose":
        if "output_shape" in attrs or tuple(attrs.get("output_padding", [0, 0])) != (0, 0):
            raise ValueError("ConvTranspose output shape/padding is not modeled")
    allowed = ({"kernel_shape", "strides", "dilations", "pads", "auto_pad", "group"}
               if op == "Conv" else
               {"kernel_shape", "strides", "dilations", "pads", "auto_pad", "group",
                "output_padding"} if op == "ConvTranspose" else
               {"kernel_shape", "strides", "dilations", "pads", "auto_pad",
                "ceil_mode", "storage_order"})
    if not set(attrs).issubset(allowed):
        raise ValueError(f"{op} has unmodeled attributes")
    return kernel, strides, dilations, pads


def analyze_local(model):
    """Return halo and phase alignment for a same-resolution local graph."""
    onnx.checker.check_model(model)
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    inputs = [item for item in graph.input if item.name not in initializers]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ValueError("local analysis requires one image input and one raster output")
    for value in (inputs[0], graph.output[0]):
        tensor = value.type.tensor_type
        if tensor.elem_type != TensorProto.FLOAT or len(tensor.shape.dim) != 4:
            raise ValueError("local analysis requires NCHW FP32 tensors")
    image = inputs[0].name
    output = graph.output[0].name
    steps = {image: (1, 1)}
    node_specs = []
    alignment = [1, 1]
    for index, node in enumerate(graph.node):
        if node.domain not in ("", "ai.onnx"):
            raise ValueError(f"node {index} has an unmodeled operator domain")
        if len(node.output) != 1 or not node.output[0]:
            raise ValueError(f"node {index} has unmodeled outputs")
        op = node.op_type
        data_inputs = [name for name in node.input if name in steps]
        if not data_inputs:
            raise ValueError(f"node {index} ({op}) has no image-dependent input")
        if len({steps[name] for name in data_inputs}) != 1:
            raise ValueError(f"node {index} ({op}) merges different spatial scales")
        step = steps[data_inputs[0]]
        spec = None
        if op in {"Conv", "ConvTranspose", "MaxPool"}:
            if node.input[0] not in steps:
                raise ValueError(f"node {index} ({op}) has no spatial input")
            spec = _spatial_op(node, initializers)
            strides = spec[1]
            if op == "ConvTranspose":
                if any(step[axis] % strides[axis] for axis in range(2)):
                    raise ValueError(f"node {index} upsampling has fractional image phase")
                next_step = tuple(step[axis] // strides[axis] for axis in range(2))
            else:
                next_step = tuple(step[axis] * strides[axis] for axis in range(2))
            alignment = [math.lcm(alignment[axis], next_step[axis]) for axis in range(2)]
        elif op in POINTWISE:
            attrs = _attrs(node)
            if op == "BatchNormalization":
                if (len(node.input) != 5 or any(name not in initializers for name in node.input[1:])
                    or any(len(initializers[name].dims) != 1 for name in node.input[1:])
                    or len({initializers[name].dims[0] for name in node.input[1:]}) != 1
                    or attrs.get("training_mode", 0) != 0
                    or not set(attrs).issubset({"epsilon", "momentum", "training_mode"})):
                    raise ValueError("BatchNormalization must use constant inference parameters")
            elif op in {"Add", "Sub", "Mul"}:
                if len(node.input) != 2 or attrs:
                    raise ValueError(f"node {index} ({op}) has unmodeled arity or attributes")
                for name in node.input:
                    if name in initializers:
                        dims = list(initializers[name].dims)
                        if len(dims) > 4 or any(dim != 1 for dim in dims[-2:]):
                            raise ValueError(f"node {index} has a spatially varying constant")
            elif len(node.input) != 1 or attrs:
                raise ValueError(f"node {index} ({op}) has unmodeled arity or attributes")
            next_step = step
        elif op == "Concat":
            if _attrs(node) != {"axis": 1} or len(data_inputs) != len(node.input):
                raise ValueError("only channel-axis Concat is local")
            next_step = step
        else:
            raise ValueError(f"node {index} ({op}) has no modeled local dependency")
        steps[node.output[0]] = next_step
        node_specs.append((node, data_inputs, spec))
    if steps.get(output) != (1, 1):
        raise ValueError("output must have the input spatial resolution")

    def dependency(axis, pixel):
        needed = {output: (pixel, pixel + 1)}
        for node, data_inputs, spec in reversed(node_specs):
            interval = needed.pop(node.output[0], None)
            if interval is None:
                continue
            low, high = interval
            if spec is not None:
                kernel, strides, dilations, pads = spec
                stride, dilation, pad = strides[axis], dilations[axis], pads[axis]
                if node.op_type == "ConvTranspose":
                    low, high = ((low + pad - (kernel[axis]-1)*dilation + stride-1)//stride,
                                 (high-1 + pad)//stride + 1)
                else:
                    low, high = (low*stride - pad,
                                 (high-1)*stride - pad + (kernel[axis]-1)*dilation + 1)
            for name in data_inputs:
                old = needed.get(name)
                needed[name] = (min(old[0], low), max(old[1], high)) if old else (low, high)
        if image not in needed:
            raise ValueError("output does not depend on the image")
        return needed[image]

    halo = []
    for axis, period in enumerate(alignment):
        if period > 4096:
            raise ValueError("alignment period is too large to validate")
        negative = positive = 0
        for pixel in range(period):
            low, high = dependency(axis, pixel)
            shifted = dependency(axis, pixel + period)
            if shifted != (low + period, high + period):
                raise ValueError("spatial dependency is not translation-periodic")
            negative = max(negative, pixel - low)
            positive = max(positive, high - pixel - 1)
        halo.append(max(negative, positive))
    return LocalPlan(halo[0], halo[1], alignment[0], alignment[1],
                     tuple(node.op_type for node in graph.node))
