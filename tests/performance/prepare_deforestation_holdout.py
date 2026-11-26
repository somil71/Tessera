"""Download and prepare the independent Deforestation U-Net holdout artifacts."""

import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
MODEL_URL = (
    "https://huggingface.co/ItsArca/deforestation-unet-onnx/resolve/"
    "e16cba63ca6d426105bc8ee6aed64e89a43b3e5e/unet_deforestation.onnx"
)
MODEL_PATH = ROOT / "models" / "holdout" / "deforestation-unet.onnx"
INPUT_DIR = ROOT / "data" / "holdout" / "deforestation"
INPUT_PATH = INPUT_DIR / "deforestation-512.tif"
RECORD_PATH = ROOT / "runs" / "holdout" / "deforestation" / "frozen-artifacts.json"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.is_file():
        print(f"Downloading model from {MODEL_URL}...")
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, MODEL_PATH.open("wb") as out_file:
            while chunk := response.read(65536):
                out_file.write(chunk)
    print(f"Model ready at {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")


def make_input(side=512, seed=8307):
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_PATH.is_file():
        print(f"Generating 10-band synthetic input at {INPUT_PATH}...")
        # 10 bands normalized synthetic values
        image = np.random.default_rng(seed).standard_normal((10, side, side), dtype=np.float32)
        with rasterio.open(
            INPUT_PATH,
            "w",
            driver="GTiff",
            height=side,
            width=side,
            count=10,
            dtype="float32",
            crs="EPSG:32643",
            transform=from_origin(500000, 2000000, 10, 10),
        ) as raster:
            raster.write(image)
    print(f"Input ready at {INPUT_PATH}")


def main():
    download_model()
    make_input(512, 8307)
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": str(MODEL_PATH),
        "model_sha256": sha256(MODEL_PATH),
        "model_bytes": MODEL_PATH.stat().st_size,
        "input": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "input_bytes": INPUT_PATH.stat().st_size,
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Frozen artifact record written to:", RECORD_PATH)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
