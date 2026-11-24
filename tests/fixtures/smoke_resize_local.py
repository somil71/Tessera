"""Validate exact local analysis for shape-built nearest Resize sizes."""

from pathlib import Path
import json
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
from local_graph import analyze_local, local_output_shape, validate_local_raster

OUTPUT = ROOT / "runs" / "resize_local_smoke"
OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL = OUTPUT / "shape_resize.onnx"
initializers = [numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), "kernel")]


def constant(name, values):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(
        np.asarray(values, dtype=np.int64)))


nodes = [
    helper.make_node("Conv", ["image", "kernel"], ["down"],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]),
    helper.make_node("Shape", ["image"], ["shape"]),
    constant("starts", [0]), constant("ends", [2]), constant("axes", [0]),
    helper.make_node("Slice", ["shape", "starts", "ends", "axes"], ["batch_channel"]),
    constant("height_index", 2), constant("width_index", 3),
    helper.make_node("Gather", ["shape", "height_index"], ["height"], axis=0),
    helper.make_node("Gather", ["shape", "width_index"], ["width"], axis=0),
    constant("unsqueeze_axis", [0]),
    helper.make_node("Unsqueeze", ["height", "unsqueeze_axis"], ["height_vec"]),
    helper.make_node("Unsqueeze", ["width", "unsqueeze_axis"], ["width_vec"]),
    helper.make_node("Concat", ["height_vec", "width_vec"], ["spatial"], axis=0),
    helper.make_node("Cast", ["spatial"], ["spatial_int"], to=TensorProto.INT64),
    helper.make_node("Concat", ["batch_channel", "spatial_int"], ["sizes"], axis=0),
    helper.make_node("Resize", ["down", "", "", "sizes"], ["logits"],
                     mode="nearest", coordinate_transformation_mode="asymmetric",
                     nearest_mode="floor"),
]
graph = helper.make_graph(nodes, "shape-resize-local",
                          [helper.make_tensor_value_info("image", TensorProto.FLOAT,
                                                         [1, 1, "height", "width"])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                                         [1, 1, "height", "width"])],
                          initializer=initializers)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, MODEL)

channels, plan = validate_local_raster(model, 66, 98, ((18, 18), (20, 20)))
assert channels == (1, 1)
assert (plan.alignment_y, plan.alignment_x) == (2, 2)
assert (plan.halo_y, plan.halo_x) == (2, 2)
assert local_output_shape(model, 66, 98) == (66, 98)
bad_mode = onnx.ModelProto()
bad_mode.CopyFrom(model)
mode = next(attr for attr in bad_mode.graph.node[-1].attribute
            if attr.name == "coordinate_transformation_mode")
mode.s = b"half_pixel"
try:
    analyze_local(bad_mode)
except ValueError as error:
    assert "nearest/asymmetric/floor" in str(error)
else:
    raise AssertionError("half-pixel Resize should have been rejected")
try:
    local_output_shape(model, 65, 98)
except ValueError as error:
    assert "integer spatial upsample" in str(error)
else:
    raise AssertionError("odd resize ratio should have been rejected")

session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
image = np.random.default_rng(9801).standard_normal((1, 1, 66, 98), dtype=np.float32)
reference = session.run(None, {"image": image})[0][0]
actual = np.empty_like(reference)
for row in range(0, 66, 16):
    for col in range(0, 98, 16):
        core_h, core_w = min(16, 66-row), min(16, 98-col)
        top, left = max(0, row-plan.halo_y), max(0, col-plan.halo_x)
        bottom, right = min(66, row+core_h+plan.halo_y), min(98, col+core_w+plan.halo_x)
        crop = image[:, :, top:bottom, left:right]
        scores = session.run(None, {"image": crop})[0][0]
        actual[:, row:row+core_h, col:col+core_w] = scores[:,
            row-top:row-top+core_h, col-left:col-left+core_w]
error = float(np.max(np.abs(actual-reference)))
assert error == 0.0, error
source = OUTPUT / "input.tif"
with rasterio.open(source, "w", driver="GTiff", width=98, height=66,
                   count=1, dtype="float32", crs="EPSG:32643",
                   transform=from_origin(500000, 2000000, 10, 10)) as dst:
    dst.write(image[0])
for runner, name, extra in (
    ("run_raster_local.py", "fixed", ["--tile-size", "16", "--cut-after", "0"]),
    ("run_raster_auto.py", "automatic", ["--max-tile-size", "32"]),
):
    output = OUTPUT / f"{name}.tif"
    output.unlink(missing_ok=True)
    result = subprocess.run([sys.executable, ROOT / "scripts" / runner,
                             "--model", MODEL, "--input", source, "--output", output,
                             "--memory-limit-mib", "128", "--threads", "2", *extra],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with rasterio.open(output) as dst:
        got = dst.read()
        assert dst.crs == rasterio.crs.CRS.from_epsg(32643)
    np.testing.assert_array_equal(got, reference)
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["memory_limit_enforced"]
    assert report["halo_pixels"] == [2, 2]
print(f"shape-built nearest Resize: halo {plan.halo_y}, alignment {plan.alignment_y}, max_abs={error}")
