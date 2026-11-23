"""Generate a deterministic development GeoTIFF without holding it in RAM."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--height", type=int, default=8192)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--rows-per-write", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args):
    started = time.perf_counter()
    output = args.output.resolve()
    if args.height < 1 or args.width < 1 or args.bands < 1 or args.rows_per_write < 1:
        raise ValueError("raster dimensions and write rows must be positive")
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    logical_hash = hashlib.sha256()
    transform = from_origin(500000, 2000000, 10, 10)
    with rasterio.open(
        output, "w", driver="GTiff", width=args.width, height=args.height,
        count=args.bands, dtype="float32", crs="EPSG:32643",
        transform=transform, tiled=True, blockxsize=256, blockysize=256,
        compress="deflate", predictor=3, BIGTIFF="IF_SAFER",
    ) as dst:
        dst.update_tags(seed=str(args.seed), generator="chunked-standard-normal-fp32")
        for row in range(0, args.height, args.rows_per_write):
            count = min(args.rows_per_write, args.height-row)
            pixels = rng.standard_normal(
                size=(args.bands, count, args.width), dtype=np.float32
            )
            logical_hash.update(pixels.tobytes(order="C"))
            dst.write(pixels, window=Window(0, row, args.width, count))
    with output.open("rb") as file:
        file_hash = hashlib.file_digest(file, "sha256").hexdigest()
    report = dict(output=str(output), seed=args.seed,
                  shape=[args.bands, args.height, args.width],
                  rows_per_write=args.rows_per_write,
                  logical_chunk_stream_sha256=logical_hash.hexdigest(),
                  geotiff_sha256=file_hash,
                  geotiff_bytes=output.stat().st_size,
                  elapsed_seconds=time.perf_counter()-started)
    output.with_suffix(".generation.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(arguments())
