"""Verify repeated requests through every split-model worker session."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "target" / "debug" / "raster-compiler.exe"
MODELS = ROOT / "models" / "evaluation"
OUT = ROOT / "runs" / "worker_smoke"
OUT.mkdir(parents=True, exist_ok=True)
ENV = os.environ.copy()
ENV["ORT_DYLIB_PATH"] = str(
    ROOT / ".venv" / "Lib" / "site-packages" / "onnxruntime" / "capi" / "onnxruntime.dll"
)
CASES = (
    ("unet_dev", (1, 3, 32, 48)),
    ("global_mean_dev", (1, 3, 33, 47)),
    ("attention_holdout", (1, 4, 8, 10)),
)


def read_exact(stream, count):
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise AssertionError(f"received only {count - remaining} of {count} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


for name, shape in CASES:
    model = MODELS / f"{name}.onnx"
    reference = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    images = (
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 1000,
        np.random.default_rng(3101).standard_normal(shape).astype(np.float32),
    )
    expected = [reference.run(None, {"image": image})[0] for image in images]
    count = len(onnx.load(model).graph.node) - 1
    for cut in range(count):
        directory = OUT / name / str(cut)
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                EXE,
                "split",
                model,
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
        for transport in ("file", "pipe"):
            worker = subprocess.Popen(
                [
                    EXE,
                    "serve-split-pipe" if transport == "pipe" else "serve-split",
                    directory / "head.onnx",
                    directory / "tail.onnx",
                    "--channels",
                    str(shape[1]),
                    "--height",
                    str(shape[2]),
                    "--width",
                    str(shape[3]),
                ],
                env=ENV,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=transport == "file",
                bufsize=1 if transport == "file" else 0,
            )
            try:
                ready = json.loads(worker.stdout.readline())
                assert ready["ready"] is True
                for index, image in enumerate(images):
                    input_path = directory / f"input_{index}.f32"
                    output_path = directory / f"output_{index}.f32"
                    if transport == "file":
                        image.tofile(input_path)
                        worker.stdin.write(
                            json.dumps(
                                {"input_f32": str(input_path), "output_f32": str(output_path)}
                            )
                            + "\n"
                        )
                    else:
                        payload = image.astype("<f4", copy=False).tobytes()
                        worker.stdin.write(
                            (json.dumps({"input_bytes": len(payload)}) + "\n").encode()
                            + payload
                        )
                    worker.stdin.flush()
                    response = json.loads(worker.stdout.readline())
                    assert response["ok"] is True, (name, cut, transport, response)
                    if transport == "file":
                        actual = np.fromfile(output_path, dtype="<f4")
                    else:
                        payload = read_exact(worker.stdout, response["output_bytes"])
                        actual = np.frombuffer(payload, dtype="<f4")
                    actual = actual.reshape(expected[index].shape)
                    absolute = np.abs(actual - expected[index])
                    mask = np.abs(expected[index]) >= 1e-3
                    max_abs = float(absolute.max())
                    max_rel = (
                        float(np.max(absolute[mask] / np.abs(expected[index][mask])))
                        if mask.any()
                        else 0.0
                    )
                    assert max_abs <= 1e-4 and max_rel <= 1e-3, (
                        name,
                        cut,
                        transport,
                        index,
                        max_abs,
                        max_rel,
                    )
                print(f"{name} cut {cut} {transport}: two requests matched reference")
            finally:
                worker.stdin.close()
                worker.wait(timeout=10)
                worker.stdout.close()
                worker.stderr.close()
            assert worker.returncode == 0, (name, cut, transport, worker.returncode)
