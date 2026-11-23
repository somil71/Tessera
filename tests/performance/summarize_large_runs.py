"""Audit completed 8192x8192 development runs and write a compact manifest."""

import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy
import onnx
import onnxruntime
import rasterio


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
CAP = 2 * 1024**3


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def family(name, seed):
    source = DATA / f"large_{name}_{seed}.tif"
    generation = load(source.with_suffix(".generation.json"))
    assert generation["shape"] == [3, 8192, 8192]
    assert generation["seed"] == seed
    assert file_hash(source) == generation["geotiff_sha256"]
    auto_path = RUNS / f"large_{name}_{'auto_budgetcache' if name == 'unet' else 'auto'}.tif"
    manual_path = RUNS / f"large_{name}_tile256.tif"
    auto = load(auto_path.with_suffix(".report.json"))
    manual = load(manual_path.with_suffix(".report.json"))
    diff_name = ("large_unet_budgetcache_partition_diff.json" if name == "unet"
                 else "large_global_partition_diff.json")
    difference = load(RUNS / diff_name)
    assert difference["passed"] and difference["differing_values"] == 0
    assert Path(difference["reference"]).resolve() == auto_path.resolve()
    assert Path(difference["candidate"]).resolve() == manual_path.resolve()
    auto_hash = file_hash(auto_path)
    manual_hash = file_hash(manual_path)
    assert difference["reference_sha256"] == auto_hash
    assert difference["candidate_sha256"] == manual_hash
    for report in (auto, manual):
        assert report["memory_limit_enforced"] is True
        assert report["memory_limit_bytes"] == CAP
        assert report["job_peak_conservative_upper_bound_bytes"] <= CAP
        assert report["shape"] == [8192, 8192]
        assert report["gdal_cachemax_mib"] in (64, 1536)
    with rasterio.open(auto_path) as raster:
        assert raster.count == 2 and raster.dtypes == ("float32", "float32")
        assert raster.tags()["seed"] == str(seed)
        assert raster.crs == rasterio.crs.CRS.from_epsg(32643)
    return dict(
        input=dict(path=str(source), sha256=generation["geotiff_sha256"],
                   bytes=generation["geotiff_bytes"], seed=seed),
        automatic=dict(path=str(auto_path), sha256=auto_hash,
                       bytes=auto_path.stat().st_size,
                       tile_size=auto["tile_size"], cut_after=auto["cut_after"],
                       planning_seconds=auto["planning_seconds"],
                       total_seconds=auto["total_elapsed_seconds"],
                       peak_job_commit_bytes=auto["job_peak_committed_bytes"],
                       gdal_cache_mib=auto["gdal_cachemax_mib"]),
        alternate=dict(path=str(manual_path), sha256=manual_hash,
                       bytes=manual_path.stat().st_size,
                       tile_size=manual["tile_size"], cut_after=manual["cut_after"],
                       total_seconds=manual["elapsed_seconds"],
                       peak_job_commit_bytes=manual["job_peak_committed_bytes"],
                       gdal_cache_mib=manual["gdal_cachemax_mib"]),
        partition_comparison=dict(max_absolute_error=difference["max_absolute_error"],
                                  max_eligible_relative_error=difference["max_eligible_relative_error"],
                                  differing_values=difference["differing_values"]),
    )


manifest = dict(
    status="development_evidence_not_held_out",
    machine=dict(platform=platform.platform(), processor=platform.processor()),
    versions=dict(python=sys.version.split()[0], numpy=numpy.__version__,
                  onnx=onnx.__version__, onnxruntime=onnxruntime.__version__,
                  rasterio=rasterio.__version__),
    memory_cap_bytes=CAP,
    unet=family("unet", 3101),
    global_mean=family("global", 3102),
)
path = RUNS / "large_development_summary.json"
path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
print(path)
