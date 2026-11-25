"""Apply the frozen logits-only projection and create amendment 6 input rasters."""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "models" / "holdout" / "sar3-source.onnx"
PROJECTED = ROOT / "models" / "holdout" / "sar3-logits.onnx"
INPUT_DIR = ROOT / "data" / "holdout" / "sar3"
RECORD = ROOT / "runs" / "holdout" / "sar3" / "frozen-artifacts.json"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_logits():
    model = onnx.load(SOURCE)
    outputs = {value.name: value for value in model.graph.output}
    if "logits" not in outputs or len(outputs) != 2:
        raise ValueError("source must declare logits and one derived output")
    producer = {name: node for node in model.graph.node for name in node.output}
    needed = set()

    def visit(name):
        node = producer.get(name)
        if node is None or node.output[0] in needed:
            return
        needed.add(node.output[0])
        for parent in node.input:
            if parent:
                visit(parent)

    visit("logits")
    selected = [node for node in model.graph.node if node.output[0] in needed]
    if not selected or len(selected) == len(model.graph.node):
        raise ValueError("logits projection did not remove a derived-output branch")
    logits = onnx.ValueInfoProto()
    logits.CopyFrom(outputs["logits"])
    del model.graph.node[:]
    model.graph.node.extend(selected)
    del model.graph.output[:]
    model.graph.output.append(logits)
    onnx.checker.check_model(model)
    onnx.save(model, PROJECTED)
    return len(selected)


def make_input(side, seed):
    path = INPUT_DIR / f"sar3-{side}.tif"
    image = np.random.default_rng(seed).standard_normal((3, side, side), dtype=np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=side, width=side,
                       count=3, dtype="float32", crs="EPSG:32643",
                       transform=from_origin(500000, 2000000, 10, 10)) as raster:
        raster.write(image)
    return path


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    PROJECTED.parent.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    selected_nodes = project_logits()
    inputs = [make_input(512, 7207), make_input(1024, 7208)]
    result = dict(source=str(SOURCE), source_sha256=sha256(SOURCE),
                  projected=str(PROJECTED), projected_sha256=sha256(PROJECTED),
                  selected_nodes=selected_nodes,
                  inputs=[dict(path=str(path), sha256=sha256(path)) for path in inputs])
    RECORD.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(RECORD)


if __name__ == "__main__":
    main()
