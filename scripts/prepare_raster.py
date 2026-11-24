"""Stream declared per-band FP32 preprocessing into a georeferenced GeoTIFF."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "64"

import numpy as np
import rasterio
from rasterio.windows import Window

from job_memory import JobMemoryLimit
from run_raster_fixed import configure_gdal_cache


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--memory-limit-mib", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_spec(path, source_count):
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec) != {"bands"} or not isinstance(spec["bands"], list) or not spec["bands"]:
        raise ValueError("spec must contain a nonempty bands list only")
    bands = []
    for index, item in enumerate(spec["bands"], 1):
        if not isinstance(item, dict) or set(item) - {"source", "scale", "offset", "clip_min", "clip_max", "name"}:
            raise ValueError(f"band {index} has unsupported spec fields")
        source = item.get("source")
        if type(source) is not int or not 1 <= source <= source_count:
            raise ValueError(f"band {index} source must be a valid 1-based band index")
        scale, offset = item.get("scale", 1.0), item.get("offset", 0.0)
        low, high = item.get("clip_min"), item.get("clip_max")
        if not all(type(value) in (int, float) and np.isfinite(value)
                   for value in (scale, offset)):
            raise ValueError(f"band {index} scale and offset must be finite numbers")
        if any(value is not None and (type(value) not in (int, float) or not np.isfinite(value))
               for value in (low, high)) or (low is not None and high is not None and low > high):
            raise ValueError(f"band {index} clipping bounds are invalid")
        name = item.get("name", f"band_{index}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"band {index} name must be a nonempty string")
        bands.append(dict(source=source, scale=float(scale), offset=float(offset),
                          clip_min=low, clip_max=high, name=name))
    return {"bands": bands}


def run(args):
    started = time.perf_counter()
    if args.memory_limit_mib < 1 or args.window_size < 1:
        raise ValueError("memory limit and window size must be positive")
    source = args.input.resolve(strict=True)
    spec_path = args.spec.resolve(strict=True)
    output = args.output.resolve()
    report_path = (args.report or output.with_suffix(".preprocess.json")).resolve()
    if len({source, spec_path, output, report_path}) != 4:
        raise ValueError("input, spec, output, and report paths must differ")
    if output.exists() or report_path.exists():
        raise FileExistsError("output and report must be new paths")
    configure_gdal_cache(args.memory_limit_mib)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.stem}.partial-{uuid4().hex}.tif"
    job = JobMemoryLimit(args.memory_limit_mib * 1024 * 1024)
    try:
        with rasterio.open(source) as src:
            if src.crs is None or src.nodata is not None:
                raise ValueError("source requires a CRS and no declared nodata")
            spec = validate_spec(spec_path, src.count)
            bands = spec["bands"]
            indexes = tuple(item["source"] for item in bands)
            profile = dict(driver="GTiff", width=src.width, height=src.height,
                           count=len(bands), dtype="float32", crs=src.crs,
                           transform=src.transform, nodata=None, compress="deflate",
                           tiled=True, blockxsize=256, blockysize=256)
            windows = 0
            with rasterio.open(temporary, "w", **profile) as dst:
                dst.update_tags(**src.tags())
                for index, item in enumerate(bands, 1):
                    dst.set_band_description(index, item["name"])
                for row in range(0, src.height, args.window_size):
                    for col in range(0, src.width, args.window_size):
                        window = Window(col, row, min(args.window_size, src.width-col),
                                        min(args.window_size, src.height-row))
                        data = src.read(indexes=indexes, window=window, out_dtype="float32")
                        masks = src.read_masks(indexes=indexes, window=window)
                        if not np.all(masks == 255) or not np.isfinite(data).all():
                            raise ValueError(f"masked or non-finite source pixel at row {row}, col {col}")
                        for index, item in enumerate(bands):
                            data[index] *= item["scale"]
                            data[index] += item["offset"]
                            if item["clip_min"] is not None or item["clip_max"] is not None:
                                np.clip(data[index], item["clip_min"], item["clip_max"],
                                        out=data[index])
                        if not np.isfinite(data).all():
                            raise ValueError(f"preprocessing produced non-finite pixel at row {row}, col {col}")
                        dst.write(data, window=window)
                        windows += 1
        os.replace(temporary, output)
        report = dict(schema="raster-compiler-preprocessing-v1",
                      input=str(source), input_sha256=sha256(source),
                      spec=str(spec_path), spec_sha256=sha256(spec_path),
                      normalized_spec=spec, output=str(output),
                      output_sha256=sha256(output), shape=[len(bands), profile["height"], profile["width"]],
                      crs=str(profile["crs"]), transform=list(profile["transform"]),
                      windows=windows, memory_limit_bytes=args.memory_limit_mib*1024*1024,
                      memory_limit_enforced=True,
                      job_peak_committed_bytes=job.peak_job_bytes(),
                      job_peak_private_commit_sampled_bytes=job.peak_private_commit_sampled_bytes,
                      job_peak_working_set_sampled_bytes=job.peak_working_set_sampled_bytes,
                      elapsed_seconds=time.perf_counter()-started)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        temporary.unlink(missing_ok=True)
        job.close()


if __name__ == "__main__":
    run(arguments())
