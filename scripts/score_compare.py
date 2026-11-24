"""Bounded comparison of georeferenced FP32 score rasters."""

import hashlib
from pathlib import Path
import time

import numpy as np
import rasterio
from rasterio.windows import Window


def compare_scores(reference: Path, candidate: Path, *, window_size: int = 256,
                   absolute_tolerance: float = 1e-4,
                   relative_tolerance: float = 1e-3) -> dict:
    """Return numerical and metadata evidence without loading entire rasters."""
    started = time.perf_counter()
    if window_size < 1:
        raise ValueError("window size must be positive")
    reference = reference.resolve(strict=True)
    candidate = candidate.resolve(strict=True)
    max_abs = 0.0
    max_rel = 0.0
    changed = 0
    windows = 0
    with rasterio.open(reference) as ref, rasterio.open(candidate) as got:
        for field in ("width", "height", "count", "crs", "transform"):
            if getattr(ref, field) != getattr(got, field):
                raise ValueError(f"raster {field} differs")
        if any(dtype != "float32" for dtype in ref.dtypes+got.dtypes):
            raise ValueError("score rasters must be FP32")
        for row in range(0, ref.height, window_size):
            for col in range(0, ref.width, window_size):
                window = Window(col, row, min(window_size, ref.width-col),
                                min(window_size, ref.height-row))
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

    def sha256(path):
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    return dict(reference=str(reference), candidate=str(candidate),
                reference_sha256=sha256(reference),
                candidate_sha256=sha256(candidate),
                shape=shape, crs=crs, transform=transform,
                windows=windows, max_absolute_error=max_abs,
                max_eligible_relative_error=max_rel,
                differing_values=changed,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                passed=max_abs <= absolute_tolerance and
                       max_rel <= relative_tolerance,
                elapsed_seconds=time.perf_counter()-started)
