"""Confirm each frozen fixture executes with ONNX Runtime on CPU."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


models = Path(__file__).resolve().parents[2] / "models" / "evaluation"
cases = (
    ("unet_dev", (1, 3, 32, 48), (1, 2, 32, 48)),
    ("global_mean_dev", (1, 3, 33, 47), (1, 2, 33, 47)),
    ("attention_holdout", (1, 4, 8, 10), (1, 4, 8, 10)),
)

for name, input_shape, output_shape in cases:
    session = ort.InferenceSession(models / f"{name}.onnx", providers=["CPUExecutionProvider"])
    image = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) / 1000
    output = session.run(None, {"image": image})[0]
    assert output.shape == output_shape, (name, output.shape)
    assert np.isfinite(output).all(), name
    print(f"{name}: {output.shape}, finite")
