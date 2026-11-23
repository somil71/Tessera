"""Compare two georeferenced FP32 score rasters in bounded windows."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "64"

import numpy as np
import rasterio
from rasterio.windows import Window


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def run(args):
    started = time.perf_counter()
    if args.window_size < 1:
        raise ValueError("window size must be positive")
    reference_path = args.reference.resolve(strict=True)
    candidate_path = args.candidate.resolve(strict=True)
    def sha256(path):
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()
    max_abs = 0.0
    max_rel = 0.0
    changed = 0
    windows = 0
    with rasterio.open(reference_path) as ref, rasterio.open(candidate_path) as got:
        for field in ("width", "height", "count", "crs", "transform"):
            if getattr(ref, field) != getattr(got, field):
                raise ValueError(f"raster {field} differs")
        if any(dtype != "float32" for dtype in ref.dtypes+got.dtypes):
            raise ValueError("score rasters must be FP32")
        for row in range(0, ref.height, args.window_size):
            for col in range(0, ref.width, args.window_size):
                window = Window(col, row, min(args.window_size, ref.width-col),
                                min(args.window_size, ref.height-row))
                expected = ref.read(window=window)
                actual = got.read(window=window)
                if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                    raise ValueError(f"non-finite score at row {row}, col {col}")
                delta = np.abs(actual-expected)
                max_abs = max(max_abs, float(delta.max()))
                eligible = np.abs(expected) >= 1e-3
                if np.any(eligible):
                    max_rel = max(max_rel, float(np.max(
                        delta[eligible]/np.abs(expected[eligible])
                    )))
                changed += int(np.count_nonzero(delta))
                windows += 1
        shape = [ref.count, ref.height, ref.width]
        crs = str(ref.crs)
        transform = list(ref.transform)
    report = dict(reference=str(reference_path), candidate=str(candidate_path),
                  reference_sha256=sha256(reference_path),
                  candidate_sha256=sha256(candidate_path),
                  shape=shape, crs=crs, transform=transform,
                  windows=windows, max_absolute_error=max_abs,
                  max_eligible_relative_error=max_rel,
                  differing_values=changed,
                  absolute_tolerance=args.absolute_tolerance,
                  relative_tolerance=args.relative_tolerance,
                  passed=max_abs <= args.absolute_tolerance and
                         max_rel <= args.relative_tolerance,
                  elapsed_seconds=time.perf_counter()-started)
    args.report.resolve().write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError("raster scores exceeded the numerical tolerances")


if __name__ == "__main__":
    run(arguments())
