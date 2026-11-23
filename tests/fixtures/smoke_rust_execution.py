"""Check Rust whole-model and split execution against the frozen CPU reference."""

import os
from pathlib import Path
import subprocess

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "target" / "debug" / "raster-compiler.exe"
MODELS = ROOT / "models" / "evaluation"
OUT = ROOT / "runs" / "rust_execution_smoke"
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


def error(actual, expected):
    absolute = np.abs(actual - expected)
    mask = np.abs(expected) >= 1e-3
    max_abs = float(absolute.max())
    max_rel = float(np.max(absolute[mask] / np.abs(expected[mask]))) if mask.any() else 0.0
    return max_abs, max_rel


for name, shape in CASES:
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    source = MODELS / f"{name}.onnx"
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 1000
    input_path = directory / "input.f32"
    image.tofile(input_path)
    expected = ort.InferenceSession(source, providers=["CPUExecutionProvider"]).run(
        None, {"image": image}
    )[0]

    base = [
        "--input-f32",
        str(input_path),
        "--channels",
        str(shape[1]),
        "--height",
        str(shape[2]),
        "--width",
        str(shape[3]),
        "--threads",
        "4",
    ]
    full_path = directory / "full.f32"
    subprocess.run(
        [EXE, "run", source, *base, "--output-f32", full_path],
        env=ENV,
        check=True,
        capture_output=True,
    )
    full = np.fromfile(full_path, dtype="<f4").reshape(expected.shape)
    full_abs, full_rel = error(full, expected)
    assert full_abs <= 1e-4 and full_rel <= 1e-3, (name, "full", full_abs, full_rel)

    cuts = len(onnx.load(source).graph.node) - 1
    for cut in range(cuts):
        split_dir = directory / f"cut_{cut}"
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
                split_dir,
            ],
            check=True,
            capture_output=True,
        )
        split_path = split_dir / "output.f32"
        subprocess.run(
            [
                EXE,
                "run-split",
                split_dir / "head.onnx",
                split_dir / "tail.onnx",
                *base,
                "--output-f32",
                split_path,
            ],
            env=ENV,
            check=True,
            capture_output=True,
        )
        actual = np.fromfile(split_path, dtype="<f4").reshape(expected.shape)
        max_abs, max_rel = error(actual, expected)
        assert max_abs <= 1e-4 and max_rel <= 1e-3, (
            name,
            cut,
            max_abs,
            max_rel,
        )
        print(f"{name} cut {cut}: max_abs={max_abs:.3g}, max_rel={max_rel:.3g}")
        disk_path = split_dir / "spilled.f32"
        spilled = subprocess.run(
            [EXE, "run-split", split_dir / "head.onnx", split_dir / "tail.onnx",
             *base, "--output-f32", disk_path, "--spill-boundary",
             "--spill-dir", split_dir],
            env=ENV, check=True, capture_output=True, text=True,
        )
        spilled_output = np.fromfile(disk_path, dtype="<f4").reshape(expected.shape)
        spill_abs, spill_rel = error(spilled_output, expected)
        assert spill_abs <= 1e-4 and spill_rel <= 1e-3, (
            name, cut, "spill", spill_abs, spill_rel
        )
        assert np.array_equal(spilled_output, actual)
        assert "boundary_storage=disk" in spilled.stdout
        assert "spill_write_bytes=" in spilled.stdout
        assert not list(split_dir.glob(".raster-compiler-spill-*"))
