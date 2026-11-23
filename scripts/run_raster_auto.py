"""Profile tile and graph-cut candidates, then execute a supported GeoTIFF model."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import time
from uuid import uuid4

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import onnx
import rasterio

from job_memory import JobMemoryLimit
from run_raster_attention import (run as run_attention,
                                  validate_model as validate_attention)
from run_raster_fixed import (DEFAULT_DLL, DEFAULT_EXE, model_contract,
                              own_peak_commit_bytes, run as run_pointwise)
from run_raster_global_mean import (run as run_global_mean, transform,
                                    validate_reduction)
from run_raster_unet import Worker, run as run_unet, validate_model as validate_unet


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit-mib", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--band-order", help="Comma-separated 1-based source band indexes")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--ort-dll", type=Path, default=DEFAULT_DLL)
    parser.add_argument("--max-tile-size", type=int, default=512,
                        help="Optional upper bound on automatic tile search")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_family(model_path, model):
    try:
        channels, scores = model_contract(model_path, 16)
        return "pointwise", channels, scores, None
    except ValueError as pointwise_error:
        try:
            channels, scores = validate_unet(model_path)
            return "local_unet", channels, scores, None
        except ValueError as unet_error:
            try:
                channels, mean_name = validate_reduction(model)
                scores = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
                if scores < 1:
                    raise ValueError("output channel count must be fixed")
                return "global_mean", channels, scores, mean_name
            except ValueError as global_error:
                try:
                    channels = validate_attention(model)
                    return "block_attention", channels, channels, None
                except ValueError as attention_error:
                    raise ValueError(
                        "model does not match a supported raster schedule: "
                        f"pointwise: {pointwise_error}; U-Net: {unet_error}; "
                        f"global mean: {global_error}; attention: {attention_error}"
                    ) from attention_error


def candidate_tiles(height, width, maximum, family):
    if maximum < 2:
        raise ValueError("maximum tile size must be at least two")
    top = min(maximum, 1 << (max(height, width) - 1).bit_length())
    top = 1 << (top.bit_length() - 1)
    minimum = 2
    sizes = []
    while top >= minimum:
        sizes.append(top)
        top //= 2
    return sizes


def worker_shape_count(height, width, tile, family):
    if family != "local_unet":
        return 1
    heights = set()
    for row in range(0, height, tile):
        core_h = min(tile, height-row)
        top = max(0, row-4)
        bottom = min(height, row+core_h+4)
        heights.add(bottom-top)
    widths = set()
    for col in range(0, width, tile):
        core_w = min(tile, width-col)
        left = max(0, col-4)
        right = min(width, col+core_w+4)
        widths.add(right-left)
    return len(heights)*len(widths)


def profile_candidate(model, directory, family, tile, cut, channels, scores,
                      height, width, exe, env, threads, job):
    shape = ((min(height, tile+8), min(width, tile+8))
             if family == "local_unet" else (tile, tile))
    candidate = dict(tile_size=tile, cut_after=cut, probe_shape=list(shape),
                     tile_count=math.ceil(height/tile)*math.ceil(width/tile),
                     worker_shape_count=worker_shape_count(height, width, tile, family))
    worker = None
    try:
        startup = time.perf_counter()
        worker = Worker(exe, env, model, directory, shape, cut,
                        channels, threads, job)
        startup_ms = (time.perf_counter()-startup)*1000
        tile_data = np.zeros((channels, *shape), dtype=np.float32)
        worker.infer(tile_data, scores)
        durations = []
        metric = None
        for _ in range(3):
            began = time.perf_counter()
            _, metric = worker.infer(tile_data, scores)
            durations.append((time.perf_counter()-began)*1000)
        inference_ms = statistics.median(durations)
        upper = own_peak_commit_bytes() + int(metric["peak_commit_bytes"])
        candidate.update(status="ok", startup_ms=startup_ms,
                         median_probe_ms=inference_ms,
                         child_peak_commit_bytes=int(metric["peak_commit_bytes"]),
                         separate_peak_upper_bound_bytes=upper,
                         predicted_execution_seconds=(
                             candidate["tile_count"]*inference_ms/1000
                             + startup_ms/1000*candidate["worker_shape_count"]
                         ))
    except Exception as exc:
        candidate.update(status="failed", error=str(exc))
    finally:
        if worker is not None:
            worker.close()
    return candidate


def run(args):
    started = time.perf_counter()
    model_path = args.model.resolve(strict=True)
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if output in {model_path, source}:
        raise ValueError("output must differ from model and input")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    if args.memory_limit_mib < 1 or args.threads < 1 or args.max_tile_size < 2:
        raise ValueError("memory limit, threads, and maximum tile size must be positive")
    if not np.isfinite(args.scale) or not np.isfinite(args.offset):
        raise ValueError("scale and offset must be finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.stem}.plan-{uuid4().hex}"
    work.mkdir()
    probe_model = model_path
    generated_model = None
    job = None
    trials = []
    attempts = []
    selected = None
    family = None
    try:
        job = JobMemoryLimit(args.memory_limit_mib*1024*1024)
        model = onnx.load(model_path)
        family, channels, scores, mean_name = model_family(model_path, model)
        if family != "global_mean" and len(model.graph.node) < 2:
            raise ValueError("split worker requires at least two model nodes")
        with rasterio.open(source) as src:
            if src.crs is None or src.count != channels:
                raise ValueError("raster requires a CRS and the model's band count")
            height, width = src.height, src.width
        if family == "local_unet" and (height % 2 or width % 2):
            raise ValueError("local U-Net requires even raster dimensions")
        if args.band_order:
            bands = tuple(int(part) for part in args.band_order.split(","))
            if sorted(bands) != list(range(1, channels+1)):
                raise ValueError("band order must permute source bands")
        if family == "block_attention":
            run_attention(args, external_job=job)
            report_path = output.with_suffix(".report.json")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            selected = dict(query_block=report["query_block"],
                            key_block=report["key_block"],
                            estimated_attention_seconds=report["estimated_attention_seconds"])
            report.update(strategy="auto-block_attention", selected_plan=selected,
                          planning_candidates=[selected],
                          planning_seconds=report["pilot_seconds"],
                          total_elapsed_seconds=time.perf_counter()-started,
                          job_peak_committed_bytes=job.peak_job_bytes(),
                          job_peak_conservative_upper_bound_bytes=job.conservative_peak_bytes())
            report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
            output.with_suffix(".plan.failure.json").unlink(missing_ok=True)
            print(f"auto plan: block attention {selected['query_block']}x{selected['key_block']}")
            return
        exe = args.exe.resolve(strict=True)
        dll = args.ort_dll.resolve(strict=True)
        env = os.environ.copy()
        env["ORT_DYLIB_PATH"] = str(dll)
        if family == "global_mean":
            generated_model = work / "global_probe.onnx"
            transform(model, mean_name, np.zeros(channels, dtype=np.float32),
                      generated_model, 16)
            probe_model = generated_model
        tiles = candidate_tiles(height, width, args.max_tile_size, family)
        cuts = range(len(model.graph.node)-1)
        if family == "global_mean":
            cuts = (0,)
        for tile in tiles:
            for cut in cuts:
                directory = work / f"tile_{tile}_cut_{cut}"
                result = profile_candidate(probe_model, directory, family, tile,
                                           cut, channels, scores, height, width,
                                           exe, env, args.threads, job)
                trials.append(result)
        feasible = [trial for trial in trials if trial["status"] == "ok"]
        if not feasible:
            raise RuntimeError("no tile/cut candidate completed under the memory budget")
        fastest = min(trial["predicted_execution_seconds"] for trial in feasible)
        near = sorted(
            (trial for trial in feasible
             if trial["predicted_execution_seconds"] <= 1.10*fastest),
            key=lambda trial: (-trial["tile_size"],
                               trial["predicted_execution_seconds"],
                               trial["separate_peak_upper_bound_bytes"]),
        )
        slower = sorted(
            (trial for trial in feasible
             if trial["predicted_execution_seconds"] > 1.10*fastest),
            key=lambda trial: (trial["predicted_execution_seconds"],
                               trial["separate_peak_upper_bound_bytes"]),
        )
        feasible = near + slower
        for candidate in feasible:
            run_args = copy.copy(args)
            run_args.tile_size = candidate["tile_size"]
            run_args.cut_after = candidate["cut_after"]
            run_args.worker_mode = "pipe"
            try:
                if family == "pointwise":
                    run_pointwise(run_args, external_job=job)
                elif family == "local_unet":
                    run_unet(run_args, external_job=job)
                else:
                    run_global_mean(run_args, external_job=job)
                selected = candidate
                break
            except Exception as exc:
                attempts.append(dict(tile_size=candidate["tile_size"],
                                     cut_after=candidate["cut_after"], error=str(exc)))
                failure_path = output.with_suffix(".failure.json")
                if failure_path.exists():
                    failure = json.loads(failure_path.read_text(encoding="utf-8"))
                    if failure.get("output_published"):
                        raise RuntimeError("execution published output before failing; see failure report") from exc
        if selected is None:
            raise RuntimeError("all profiled plans failed during complete-raster execution")
        report_path = output.with_suffix(".report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(strategy=f"auto-{family}",
                      selection_rule="Within 10% predicted runtime, prefer fewer tiles",
                      planning_seconds=time.perf_counter()-started-report["elapsed_seconds"],
                      total_elapsed_seconds=time.perf_counter()-started,
                      selected_plan=selected,
                      planning_candidates=trials,
                      failed_execution_attempts=attempts,
                      job_peak_committed_bytes=job.peak_job_bytes(),
                      job_peak_conservative_upper_bound_bytes=job.conservative_peak_bytes())
        report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        output.with_suffix(".plan.failure.json").unlink(missing_ok=True)
        print(f"auto plan: {family}, tile {selected['tile_size']}, cut {selected['cut_after']}; "
              f"{len(trials)} candidates")
    except Exception as exc:
        failure = dict(status="failed", error=str(exc), family=family,
                       planning_candidates=trials,
                       failed_execution_attempts=attempts,
                       memory_limit_bytes=args.memory_limit_mib*1024*1024,
                       job_peak_committed_bytes=job.peak_job_bytes() if job else None)
        output.with_suffix(".plan.failure.json").write_text(
            json.dumps(failure, indent=2)+"\n", encoding="utf-8")
        raise
    finally:
        if generated_model is not None:
            generated_model.unlink(missing_ok=True)
        work.rmdir()
        if job is not None:
            job.close()


if __name__ == "__main__":
    run(arguments())
