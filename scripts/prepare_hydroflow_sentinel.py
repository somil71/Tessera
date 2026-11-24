"""Build a georeferenced 12-band HydroFlow trial from one pinned Sentinel-2 scene."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

os.environ["GDAL_CACHEMAX"] = "64"

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform

from job_memory import JobMemoryLimit
from run_raster_fixed import configure_gdal_cache


ITEM_ID = "S2A_43PFL_20250328_1_L2A"
ITEM_URL = f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{ITEM_ID}"
ITEM_SHA256 = "29bea2196389eaf4503a56452a5df99ea9878be998907f0241b54c041379fe48"
ASSETS = ("coastal", "blue", "green", "red", "rededge1", "rededge2",
          "rededge3", "nir", "nir08", "nir09", "swir16", "swir22")
NAMES = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
         "B8A", "B09", "B11", "B12")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("models/public/hydroflow-config.json"))
    parser.add_argument("--row", type=int, default=1650)
    parser.add_argument("--col", type=int, default=3160)
    parser.add_argument("--size", type=int, default=1088)
    parser.add_argument("--memory-limit-mib", type=int, default=2048)
    return parser.parse_args()


def run(args):
    if args.size < 1 or args.row < 0 or args.col < 0 or args.memory_limit_mib < 1:
        raise ValueError("size and memory cap must be positive; row and col must be nonnegative")
    output = args.output.resolve()
    spec_output = args.spec_output.resolve()
    config_path = args.config.resolve(strict=True)
    report_path = output.with_suffix(".source.json")
    if len({output, spec_output, config_path, report_path}) != 4:
        raise ValueError("all file paths must be distinct")
    if any(path.exists() for path in (output, spec_output, report_path)):
        raise FileExistsError("output, spec, and source report must be new paths")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    calibration = config["calibration"]
    if len(calibration["band_names"]) != len(ASSETS):
        raise ValueError("expected twelve HydroFlow calibration bands")
    configure_gdal_cache(args.memory_limit_mib)
    job = JobMemoryLimit(args.memory_limit_mib * 1024 * 1024)
    temporary = output.parent / f".{output.stem}.partial-{uuid4().hex}.tif"
    try:
        with urlopen(ITEM_URL, timeout=30) as response:
            item_bytes = response.read()
        if hashlib.sha256(item_bytes).hexdigest() != ITEM_SHA256:
            raise ValueError("pinned STAC item metadata changed")
        item = json.loads(item_bytes)
        if item["id"] != ITEM_ID:
            raise ValueError("STAC item ID changed")
        sources = []
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                   CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                   GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_MAX_RETRY="2")
        output.parent.mkdir(parents=True, exist_ok=True)
        spec_output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.Env(**env):
            blue = item["assets"]["blue"]["href"]
            with rasterio.open(blue) as reference:
                if args.row + args.size > reference.height or args.col + args.size > reference.width:
                    raise ValueError("crop exceeds reference image")
                region = Window(args.col, args.row, args.size, args.size)
                transform = window_transform(region, reference.transform)
                crs = reference.crs
            profile = dict(driver="GTiff", width=args.size, height=args.size,
                           count=len(ASSETS), dtype="uint16", crs=crs, transform=transform,
                           nodata=None, compress="deflate", tiled=True,
                           blockxsize=256, blockysize=256)
            with rasterio.open(temporary, "w", **profile) as target:
                target.update_tags(source_item=ITEM_ID, source_url=ITEM_URL,
                                   source_datetime=item["properties"]["datetime"],
                                   reference_asset="blue", reference_row=args.row,
                                   reference_col=args.col, resampling="bilinear")
                for index, (key, name) in enumerate(zip(ASSETS, NAMES), 1):
                    asset = item["assets"][key]
                    if asset["eo:bands"][0]["name"] != name:
                        raise ValueError(f"unexpected {key} band identity")
                    with rasterio.open(asset["href"]) as source:
                        if source.crs != crs or source.count != 1:
                            raise ValueError(f"unexpected {name} source projection or count")
                        with WarpedVRT(source, crs=crs, transform=transform,
                                       width=args.size, height=args.size,
                                       resampling=Resampling.bilinear, nodata=0) as vrt:
                            for row in range(0, args.size, 256):
                                for col in range(0, args.size, 256):
                                    window = Window(col, row, min(256, args.size-col),
                                                    min(256, args.size-row))
                                    pixels = vrt.read(1, window=window)
                                    mask = vrt.read_masks(1, window=window)
                                    if not np.all(mask == 255):
                                        raise ValueError(f"nodata in {name} at {row},{col}")
                                    target.write(pixels, index, window=window)
                    target.set_band_description(index, name)
                    sources.append(dict(name=name, key=key, href=asset["href"],
                                        native_resolution_m=asset["raster:bands"][0]["spatial_resolution"]))
        spec = {"bands": []}
        for index, name in enumerate(calibration["band_names"], 1):
            low = calibration["band_mins"][index-1]
            high = calibration["band_maxs"][index-1]
            denom = high - low + 1e-6
            spec["bands"].append(dict(source=index, name=name,
                                      scale=1.0/denom, offset=-low/denom,
                                      clip_min=0.0, clip_max=1.0))
        os.replace(temporary, output)
        spec_output.write_text(json.dumps(spec, indent=2)+"\n", encoding="utf-8")
        report = dict(schema="raster-compiler-stac-source-v1", item_id=ITEM_ID,
                      item_url=ITEM_URL, item_sha256=hashlib.sha256(item_bytes).hexdigest(),
                      datetime=item["properties"]["datetime"],
                      scene_cloud_cover_percent=item["properties"]["eo:cloud_cover"],
                      sources=sources, reference_row=args.row, reference_col=args.col,
                      width=args.size, height=args.size, crs=str(crs),
                      transform=list(transform), resampling="bilinear",
                      output=str(output), output_sha256=sha256(output),
                      spec=str(spec_output), spec_sha256=sha256(spec_output),
                      config_sha256=sha256(config_path),
                      memory_limit_bytes=args.memory_limit_mib*1024*1024,
                      job_peak_committed_bytes=job.peak_job_bytes())
        report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        temporary.unlink(missing_ok=True)
        job.close()


if __name__ == "__main__":
    run(arguments())
