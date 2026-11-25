"""Conservative spatial dependency analysis for local NCHW ONNX graphs.

This module derives an aligned input halo. It deliberately rejects operators
whose dependency or output phase has not been modeled; it does not execute a
raster or by itself establish a supported inference contract.
"""

from dataclasses import dataclass
import math

import onnx
import numpy as np
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


POINTWISE = {"Relu", "Identity", "Sigmoid", "BatchNormalization",
             "Add", "Sub", "Mul", "Div"}
SHAPE_ONLY = {"Constant", "Gather", "Unsqueeze", "Slice", "Cast",
              "ConstantOfShape", "Reshape", "Transpose"}
PURE_CONSTANT = SHAPE_ONLY | {"Concat"}


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


def _constant_value(model, name, element_type):
    """Evaluate a small initializer/constant-only branch, never image values."""
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output}
    selected = set()

    def visit(value):
        if value in initializers:
            return
        node = producers.get(value)
        if node is None or node.op_type not in PURE_CONSTANT:
            raise ValueError(f"{name} is not produced by a constant-only branch")
        if node.output[0] in selected:
            return
        if node.op_type == "Concat" and _attrs(node).get("axis") != 0:
            raise ValueError(f"{name} uses a non-vector constant Concat")
        for source in node.input:
            if source:
                visit(source)
        selected.add(node.output[0])

    visit(name)
    nodes = [node for node in graph.node if node.output[0] in selected]
    for node in nodes:
        if node.op_type == "ConstantOfShape":
            requested = np.asarray(_constant_value(model, node.input[0], TensorProto.INT64))
            if (requested.dtype != np.int64 or requested.ndim != 1
                    or requested.size > 4 or np.any(requested < 0)
                    or math.prod(int(dim) for dim in requested) > 64):
                raise ValueError("constant Pad shape branch requests too many elements")
    used_initializers = [item for item in graph.initializer
                         if item.name == name or any(item.name in node.input for node in nodes)]
    mini = helper.make_model(
        helper.make_graph(nodes, "constant_padding", [],
                          [helper.make_tensor_value_info(name, element_type, [None])],
                          initializer=used_initializers),
        opset_imports=list(model.opset_import),
    )
    mini.ir_version = model.ir_version
    onnx.checker.check_model(mini)
    return ReferenceEvaluator(mini).run(None, {})[0]


def _pad_spec(node, model):
    attrs = _attrs(node)
    if attrs.get("mode", b"constant") != b"constant" or set(attrs) - {"mode"}:
        raise ValueError("Pad requires constant mode without other attributes")
    if len(node.input) not in (2, 3) or not node.input[1]:
        raise ValueError("Pad requires explicit pad widths")
    pads = np.asarray(_constant_value(model, node.input[1], TensorProto.INT64))
    if pads.dtype != np.int64 or pads.shape != (8,):
        raise ValueError("Pad widths must be an eight-element int64 vector")
    if (np.any(pads < 0) or np.any(pads > 4096)
            or np.any(pads[[0, 1, 4, 5]] != 0)):
        raise ValueError("Pad requires bounded nonnegative spatial-only widths")
    if len(node.input) == 3 and node.input[2]:
        value = np.asarray(_constant_value(model, node.input[2], TensorProto.FLOAT))
        if value.dtype != np.float32 or value.size != 1 or not np.isfinite(value).all():
            raise ValueError("Pad value must be a finite FP32 scalar")
    return int(pads[2]), int(pads[3]), int(pads[6]), int(pads[7])


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
        if (len(node.input) not in (2, 3) or node.input[1] not in initializers
            or (len(node.input) == 3 and node.input[2] not in initializers)):
            raise ValueError(f"{op} requires a constant kernel")
        weight = initializers[node.input[1]]
        if len(weight.dims) != 4:
            raise ValueError(f"{op} requires a 2D kernel")
        kernel = _pair(attrs.get("kernel_shape", weight.dims[2:]), f"{op} kernel")
        if tuple(weight.dims[2:]) != kernel:
            raise ValueError(f"{op} kernel declaration differs from weights")
    else:
        if len(node.input) != 1:
            raise ValueError("MaxPool requires one spatial input")
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


def _concrete_shapes(model, height, width):
    """Infer tensor shapes with a concrete batch-one image, without executing it."""
    concrete = onnx.ModelProto()
    concrete.CopyFrom(model)
    initializers = {item.name for item in concrete.graph.initializer}
    inputs = [item for item in concrete.graph.input if item.name not in initializers]
    if len(inputs) != 1:
        raise ValueError("concrete shape inference needs one image input")
    dims = inputs[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        raise ValueError("concrete shape inference needs NCHW input")
    for axis, value in ((0, 1), (2, height), (3, width)):
        dims[axis].ClearField("dim_param")
        dims[axis].dim_value = value
    inferred = onnx.shape_inference.infer_shapes(concrete, data_prop=True)
    shapes = {}
    for value in (*inferred.graph.input, *inferred.graph.value_info,
                  *inferred.graph.output):
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 4 and all(dim.HasField("dim_value") and dim.dim_value > 0
                                  for dim in dims):
            shapes[value.name] = tuple(dim.dim_value for dim in dims)
    return shapes


def _resize_factors(node, shape_sets):
    attrs = _attrs(node)
    if (attrs.get("mode", b"nearest") != b"nearest"
            or attrs.get("coordinate_transformation_mode", b"half_pixel") != b"asymmetric"
            or attrs.get("nearest_mode", b"round_prefer_floor") != b"floor"
            or set(attrs) - {"mode", "coordinate_transformation_mode", "nearest_mode",
                             "cubic_coeff_a"}):
        raise ValueError("Resize requires nearest/asymmetric/floor semantics")
    if len(node.input) != 4 or node.input[1:3] != ["", ""] or not node.input[3]:
        raise ValueError("Resize requires a dynamic sizes input only")
    factors = None
    for shapes in shape_sets:
        source, target = shapes.get(node.input[0]), shapes.get(node.output[0])
        if source is None or target is None or source[:2] != target[:2]:
            raise ValueError("Resize spatial input/output shapes were not inferred")
        if any(target[axis] % source[axis] for axis in (2, 3)):
            raise ValueError("Resize needs an integer spatial upsample factor")
        sample = tuple(target[axis] // source[axis] for axis in (2, 3))
        if any(factor < 2 for factor in sample):
            raise ValueError("Resize must upsample both spatial axes")
        if factors is not None and factors != sample:
            raise ValueError("Resize factor changed across aligned input sizes")
        factors = sample
    return factors


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
    resize_shape_sets = ([_concrete_shapes(model, side, side) for side in (128, 256)]
                         if any(node.op_type == "Resize" for node in graph.node) else [])
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
        if op == "Shape":
            if len(node.input) != 1 or _attrs(node):
                raise ValueError(f"node {index} Shape has unmodeled inputs or attributes")
            continue
        if op in SHAPE_ONLY or (op == "Concat" and _attrs(node).get("axis") == 0):
            if data_inputs:
                raise ValueError(f"node {index} ({op}) applies shape-only logic to image values")
            continue
        if not data_inputs:
            raise ValueError(f"node {index} ({op}) has no image-dependent input")
        if len({steps[name] for name in data_inputs}) != 1:
            raise ValueError(f"node {index} ({op}) merges different spatial scales")
        step = steps[data_inputs[0]]
        spec = None
        resize_factor = None
        pad_spec = None
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
            elif op in {"Add", "Sub", "Mul", "Div"}:
                if len(node.input) != 2 or attrs:
                    raise ValueError(f"node {index} ({op}) has unmodeled arity or attributes")
                for name in node.input:
                    if name not in steps and name not in initializers:
                        raise ValueError(f"node {index} ({op}) has an unmodeled nonconstant input")
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
        elif op == "Resize":
            if list(node.input[:1]) != [data_inputs[0]] or len(data_inputs) != 1:
                raise ValueError(f"node {index} Resize needs one spatial input")
            resize_factor = _resize_factors(node, resize_shape_sets)
            if any(step[axis] % resize_factor[axis] for axis in range(2)):
                raise ValueError(f"node {index} Resize has fractional image phase")
            next_step = tuple(step[axis] // resize_factor[axis] for axis in range(2))
            alignment = [math.lcm(alignment[axis], step[axis]) for axis in range(2)]
        elif op == "Pad":
            if node.input[0] not in steps or len(data_inputs) != 1:
                raise ValueError(f"node {index} Pad needs one spatial input")
            pad_spec = _pad_spec(node, model)
            next_step = step
        else:
            raise ValueError(f"node {index} ({op}) has no modeled local dependency")
        steps[node.output[0]] = next_step
        node_specs.append((node, data_inputs, spec, resize_factor, pad_spec))
    if steps.get(output) != (1, 1):
        raise ValueError("output must have the input spatial resolution")

    def dependency(axis, pixel):
        needed = {output: (pixel, pixel + 1)}
        for node, data_inputs, spec, resize_factor, pad_spec in reversed(node_specs):
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
            elif resize_factor is not None:
                factor = resize_factor[axis]
                low, high = low // factor, (high-1) // factor + 1
            elif pad_spec is not None:
                before = pad_spec[axis]
                low, high = low-before, high-before
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
        radius = max(negative, positive)
        halo.append(((radius + period - 1) // period) * period)
    return LocalPlan(halo[0], halo[1], alignment[0], alignment[1],
                     tuple(node.op_type for node in graph.node))


def local_channels(model):
    """Return fixed input/output channel counts for a local raster graph."""
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    if len(inputs) != 1 or len(model.graph.output) != 1:
        raise ValueError("local raster needs one input and one output")
    channels = []
    for value in (inputs[0], model.graph.output[0]):
        dims = value.type.tensor_type.shape.dim
        if len(dims) != 4 or not dims[1].HasField("dim_value") or dims[1].dim_value < 1:
            raise ValueError("local raster needs fixed positive channel counts")
        if dims[0].HasField("dim_value") and dims[0].dim_value != 1:
            raise ValueError("local raster needs batch size one")
        if any(dims[axis].HasField("dim_value") for axis in (2, 3)):
            raise ValueError("local raster needs dynamic spatial dimensions")
        channels.append(dims[1].dim_value)
    return tuple(channels)


def local_output_shape(model, height, width):
    """Calculate spatial output dimensions, rejecting inconsistent merges."""
    if height < 1 or width < 1:
        raise ValueError("local raster dimensions must be positive")
    if any(node.op_type == "Resize" for node in model.graph.node):
        shapes = _concrete_shapes(model, height, width)
        for index, node in enumerate(model.graph.node):
            if node.op_type == "Resize":
                try:
                    _resize_factors(node, [shapes])
                except ValueError as error:
                    raise ValueError(f"node {index} Resize shape invalid: {error}") from error
        output = shapes.get(model.graph.output[0].name)
        if output is None:
            raise ValueError("local raster output shape was not inferred")
        return output[2:]
    initializers = {item.name: item for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    shapes = {inputs[0].name: (height, width)}
    for index, node in enumerate(model.graph.node):
        if node.op_type == "Shape":
            continue
        if (node.op_type in SHAPE_ONLY
                or (node.op_type == "Concat" and _attrs(node).get("axis") == 0)):
            if any(name in shapes for name in node.input):
                raise ValueError(f"node {index} applies shape-only logic to image values")
            continue
        data = [shapes[name] for name in node.input if name in shapes]
        if not data or len(set(data)) != 1:
            raise ValueError(f"node {index} has inconsistent spatial inputs")
        shape = data[0]
        if node.op_type in {"Conv", "ConvTranspose", "MaxPool"}:
            kernel, strides, dilations, pads = _spatial_op(node, initializers)
            result = []
            for axis in range(2):
                effective = dilations[axis]*(kernel[axis]-1)+1
                if node.op_type == "ConvTranspose":
                    length = (shape[axis]-1)*strides[axis]+effective-pads[axis]-pads[axis+2]
                else:
                    length = ((shape[axis]+pads[axis]+pads[axis+2]-effective)//strides[axis]+1)
                if length < 1:
                    raise ValueError(f"node {index} produces an empty spatial axis")
                result.append(length)
            shape = tuple(result)
        elif node.op_type == "Pad":
            before_y, before_x, after_y, after_x = _pad_spec(node, model)
            shape = (shape[0]+before_y+after_y, shape[1]+before_x+after_x)
        shapes[node.output[0]] = shape
    return shapes[model.graph.output[0].name]


def validate_local_raster(model, height, width, cropped_shapes=()):
    """Check dependency and same-size output for the raster and each tile shape."""
    plan = analyze_local(model)
    channels = local_channels(model)
    if height % plan.alignment_y or width % plan.alignment_x:
        raise ValueError("raster dimensions violate local graph alignment")
    if local_output_shape(model, height, width) != (height, width):
        raise ValueError("local graph output differs from input raster dimensions")
    for shape in cropped_shapes:
        if local_output_shape(model, *shape) != shape:
            raise ValueError(f"local graph changes cropped tile shape {shape}")
    return channels, plan
