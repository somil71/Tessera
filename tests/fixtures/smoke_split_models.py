"""Split frozen fixtures and compare chained ONNX Runtime outputs to unsplit runs."""

from pathlib import Path
import subprocess

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "target" / "debug" / "raster-compiler.exe"
MODELS = ROOT / "models" / "evaluation"
OUT = ROOT / "runs" / "split_smoke"
CASES = (
    ("unet_dev", (1, 3, 32, 48)),
    ("global_mean_dev", (1, 3, 33, 47)),
    ("attention_holdout", (1, 4, 8, 10)),
)


for name, shape in CASES:
    source = MODELS / f"{name}.onnx"
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 1000
    reference = ort.InferenceSession(source, providers=["CPUExecutionProvider"])
    expected = reference.run(None, {"image": image})[0]
    cuts = len(onnx.load(source).graph.node) - 1
    for cut in range(cuts):
        directory = OUT / name / str(cut)
        subprocess.run(
            [
                EXE,
                "split",
                source,
                "--cut-after",
                str(cut),
                "--height",
                str(shape[2]),
                "--width",
                str(shape[3]),
                "--out-dir",
                directory,
            ],
            check=True,
            capture_output=True,
        )
        head_path = directory / "head.onnx"
        tail_path = directory / "tail.onnx"
        onnx.checker.check_model(head_path)
        onnx.checker.check_model(tail_path)
        head = ort.InferenceSession(head_path, providers=["CPUExecutionProvider"])
        tail = ort.InferenceSession(tail_path, providers=["CPUExecutionProvider"])
        boundary_values = head.run(None, {"image": image})
        boundary = dict(zip((item.name for item in head.get_outputs()), boundary_values))
        actual = tail.run(None, boundary)[0]
        absolute_error = np.max(np.abs(actual - expected))
        assert np.allclose(actual, expected, atol=1e-4, rtol=1e-3), (
            name,
            cut,
            absolute_error,
        )
        print(f"{name} cut {cut}: max absolute error {absolute_error:.3g}")
