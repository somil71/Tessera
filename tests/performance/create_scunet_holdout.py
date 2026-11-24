"""Generate the three reserved SCUNet FP32 rasters from amendment 3."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


CASES = ((5201, 64), (5202, 128), (5203, 512))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for seed, size in CASES:
        path = output_dir / f"scunet-{seed}-{size}.tif"
        if path.exists():
            raise FileExistsError(path)
        pixels = np.random.default_rng(seed).random((3, size, size), dtype=np.float32)
        with rasterio.open(path, "w", driver="GTiff", width=size, height=size,
                           count=3, dtype="float32", crs="EPSG:32643",
                           transform=from_origin(500000, 2000000, 1, 1),
                           nodata=None, tiled=True, blockxsize=256,
                           blockysize=256, compress="deflate", predictor=3) as dst:
            dst.write(pixels)
            dst.descriptions = ("R", "G", "B")
            dst.update_tags(seed=seed, generator="numpy-default_rng-random-float32")
        cases.append(dict(seed=seed, shape=[3, size, size], path=str(path.resolve()),
                          sha256=sha256(path), pixels_sha256=hashlib.sha256(
                              pixels.tobytes(order="C")).hexdigest()))
    report = output_dir / "generation.json"
    report.write_text(json.dumps(dict(schema="raster-compiler-scunet-inputs-v1",
                                      numpy_version=np.__version__, cases=cases),
                                 indent=2)+"\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
