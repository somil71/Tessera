"""Profile tile and graph-cut candidates, then execute a supported GeoTIFF model."""

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time
from uuid import uuid4

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "64"

import numpy as np
import onnx
import rasterio

from job_memory import JobMemoryLimit
from local_graph import analyze_local, local_channels, validate_local_raster
from run_raster_attention import (run as run_attention,
                                  validate_model as validate_attention)
from run_raster_fixed import (DEFAULT_DLL, DEFAULT_EXE, configure_gdal_cache, model_contract,
                              own_peak_commit_bytes, run as run_pointwise)
from run_raster_global_mean import (run as run_global_mean, transform,
                                    validate_reduction)
from run_raster_unet import Worker, run as run_unet, validate_model as validate_unet

DISK_PROBE_MIN_EXPECTED_RUNTIME_SECONDS = 10.0
RAM_SLOW_TILE_FACTOR = 1.5
RAM_SLOW_TILE_STREAK = 2
MAX_LARGE_GRAPH_CUTS = 4
LARGE_PROBE_PIXELS = 1_048_576


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
    parser.add_argument("--explore-spill", action="store_true",
                        help="Profile larger disk-spill U-Net tiles even for short predicted RAM runs")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_family(model_path, model):
    supported_ops = {"Conv", "ConvTranspose", "Relu", "Identity", "Sigmoid",
                     "BatchNormalization", "Add", "Sub", "Mul", "Concat",
                     "MaxPool", "ReduceMean", "Shape", "Reshape", "Transpose",
                     "MatMul", "Softmax", "Constant", "Gather", "Unsqueeze",
                     "Slice", "Cast", "Resize"}
    for index, node in enumerate(model.graph.node):
        if node.op_type not in supported_ops:
            raise ValueError(
                f"first unsupported node {index} ({node.op_type}): "
                "operator has no validated raster schedule"
            )
    try:
        channels, scores = model_contract(model_path, 16)
        return "pointwise", channels, scores, None
    except ValueError as pointwise_error:
        try:
            channels, scores = validate_unet(model_path)
            return "local_unet", channels, scores, None
        except ValueError as unet_error:
            try:
                channels, scores = local_channels(model)
                analyze_local(model)
                return "local_graph", channels, scores, None
            except ValueError as local_error:
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
                            f"local: {local_error}; global mean: {global_error}; "
                            f"attention: {attention_error}"
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


def candidate_cuts(model, family):
    """Sample large local graphs at data stages; preserve exhaustive small graphs."""
    count = len(model.graph.node)
    if count < 2:
        return []
    if family != "local_graph" or count <= 32:
        return list(range(count - 1))
    spatial = {"Conv", "ConvTranspose", "MaxPool", "Resize", "MatMul",
               "Softmax", "ReduceMean"}
    data_nodes = [index for index, node in enumerate(model.graph.node[:-1])
                  if node.op_type in spatial]
    if not data_nodes:
        data_nodes = list(range(count-1))
    selected = {0, count-2}
    resize_nodes = [index for index, node in enumerate(model.graph.node[:-1])
                    if node.op_type == "Resize"]
    if resize_nodes:
        selected.add(resize_nodes[0])
        before = next((candidate for candidate in reversed(data_nodes)
                       if candidate < resize_nodes[0]), None)
        if before is not None:
            selected.add(before)
    for fraction in (8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15):
        if len(selected) >= MAX_LARGE_GRAPH_CUTS:
            break
        selected.add(data_nodes[round(fraction*(len(data_nodes)-1)/16)])
    return sorted(selected)


def unique_probe_tiles(tiles, height, width, local_plan):
    """Retain the largest tile for each halo-expanded probe shape."""
    seen = set()
    selected = []
    for tile in tiles:
        shape = (min(height, tile+2*local_plan.halo_y),
                 min(width, tile+2*local_plan.halo_x))
        if shape not in seen:
            selected.append(tile)
            seen.add(shape)
    return selected


def worker_shape_count(height, width, tile, family, local_plan=None):
    if family not in {"local_unet", "local_graph"}:
        return 1
    heights = set()
    for row in range(0, height, tile):
        core_h = min(tile, height-row)
        top = max(0, row-local_plan.halo_y)
        bottom = min(height, row+core_h+local_plan.halo_y)
        heights.add(bottom-top)
    widths = set()
    for col in range(0, width, tile):
        core_w = min(tile, width-col)
        left = max(0, col-local_plan.halo_x)
        right = min(width, col+core_w+local_plan.halo_x)
        widths.add(right-left)
    return len(heights)*len(widths)


def profile_candidate(model, directory, family, tile, cut, channels, scores,
                      height, width, exe, env, threads, job, storage_mode="memory",
                      local_plan=None):
    shape = ((min(height, tile+2*local_plan.halo_y),
              min(width, tile+2*local_plan.halo_x))
             if family in {"local_unet", "local_graph"} else (tile, tile))
    candidate = dict(tile_size=tile, cut_after=cut, storage_mode=storage_mode,
                     probe_shape=list(shape),
                     timed_probe_repetitions=(1 if shape[0]*shape[1] > LARGE_PROBE_PIXELS
                                              else 3),
                     tile_count=math.ceil(height/tile)*math.ceil(width/tile),
                     worker_shape_count=worker_shape_count(
                         height, width, tile, family, local_plan))
    worker = None
    try:
        startup = time.perf_counter()
        worker = Worker(exe, env, model, directory, shape, cut,
                        channels, threads, job, storage_mode)
        startup_ms = (time.perf_counter()-startup)*1000
        tile_data = np.zeros((channels, *shape), dtype=np.float32)
        worker.infer(tile_data, scores)
        durations = []
        metric = None
        for _ in range(candidate["timed_probe_repetitions"]):
            began = time.perf_counter()
            _, metric = worker.infer(tile_data, scores)
            durations.append((time.perf_counter()-began)*1000)
        inference_ms = statistics.median(durations)
        upper = own_peak_commit_bytes() + int(metric["peak_commit_bytes"])
        candidate.update(status="ok", startup_ms=startup_ms,
                         median_probe_ms=inference_ms,
                         child_peak_commit_bytes=int(metric["peak_commit_bytes"]),
                         probe_spill_write_bytes=int(metric.get("spill_write_bytes", 0)),
                         probe_spill_read_bytes=int(metric.get("spill_read_bytes", 0)),
                         probe_spill_peak_storage_bytes=int(
                             metric.get("spill_peak_storage_bytes", 0)),
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


def rank_candidates(feasible):
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
    return near + slower


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
    configure_gdal_cache(args.memory_limit_mib)
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
        local_plan = (analyze_local(model) if family in {"local_unet", "local_graph"}
                      else None)
        if family != "global_mean" and len(model.graph.node) < 2:
            raise ValueError("split worker requires at least two model nodes")
        with rasterio.open(source) as src:
            if src.crs is None or src.count != channels:
                raise ValueError("raster requires a CRS and the model's band count")
            height, width = src.height, src.width
        if family in {"local_unet", "local_graph"}:
            if height % local_plan.alignment_y or width % local_plan.alignment_x:
                raise ValueError("raster dimensions violate local graph alignment")
            if family == "local_graph":
                validate_local_raster(model, height, width)
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
            report.update(job.resource_snapshot())
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
        equivalent_tiles = []
        if local_plan is not None:
            tiles = [tile for tile in tiles if tile % local_plan.alignment_y == 0
                     and tile % local_plan.alignment_x == 0]
            if not tiles:
                raise ValueError("maximum tile size is below local graph alignment")
            original_tiles = tiles
            tiles = unique_probe_tiles(tiles, height, width, local_plan)
            equivalent_tiles = [tile for tile in original_tiles if tile not in tiles]
        cuts = candidate_cuts(model, family)
        if family == "global_mean":
            cuts = (0,)
        node_count = len(model.graph.node)
        sampled_cuts = len(cuts) < node_count-1 and family == "local_graph"
        if family in {"local_unet", "local_graph"}:
            del model
            gc.collect()
        probed_disk_tiles = set()
        probed_ram_tiles = set()
        attempted_plans = set()
        ram_search_stopped_after = None
        restored_equivalent_tiles = []
        rounds = (3 + int(sampled_cuts) + int(bool(equivalent_tiles))
                  if local_plan is not None else 2)
        for round_index in range(rounds):
            if round_index == 0:
                slow_tile_streak = 0
                for tile in tiles:
                    probed_ram_tiles.add(tile)
                    tile_trials = []
                    for cut in cuts:
                        directory = work / f"tile_{tile}_cut_{cut}"
                        trial = profile_candidate(
                            probe_model, directory, family, tile, cut, channels,
                            scores, height, width, exe, env, args.threads, job,
                            "memory", local_plan)
                        trials.append(trial)
                        tile_trials.append(trial)
                        if sampled_cuts and trial["status"] != "ok":
                            disk_trial = profile_candidate(
                                probe_model, work / f"tile_{tile}_cut_{cut}_disk",
                                family, tile, cut, channels, scores, height,
                                width, exe, env, args.threads, job, "disk",
                                local_plan)
                            trials.append(disk_trial)
                            tile_trials.append(disk_trial)
                            probed_disk_tiles.add(tile)
                        if sampled_cuts and any(item["status"] == "ok"
                                                for item in tile_trials):
                            break
                    feasible_ram = [trial for trial in trials
                                    if trial["storage_mode"] == "memory" and
                                    trial["status"] == "ok"]
                    feasible_tile = [trial for trial in tile_trials
                                     if trial["status"] == "ok"]
                    if feasible_tile and feasible_ram:
                        best = min(trial["predicted_execution_seconds"]
                                   for trial in feasible_ram)
                        tile_best = min(trial["predicted_execution_seconds"]
                                        for trial in feasible_tile)
                        slow_tile_streak = (slow_tile_streak + 1 if
                                            tile_best > RAM_SLOW_TILE_FACTOR*best else 0)
                    else:
                        slow_tile_streak = 0
                    if slow_tile_streak >= RAM_SLOW_TILE_STREAK:
                        ram_search_stopped_after = tile
                        break
                    if sampled_cuts and feasible_tile:
                        break
                if local_plan is not None:
                    ram_ok = [trial for trial in trials if trial["status"] == "ok"]
                    best_ram = (min(ram_ok, key=lambda trial:
                                trial["predicted_execution_seconds"])
                                if ram_ok else None)
                    if best_ram is None:
                        disk_tiles = [tile for tile in tiles if tile not in probed_disk_tiles]
                    elif (args.explore_spill or
                          best_ram["predicted_execution_seconds"] >=
                          DISK_PROBE_MIN_EXPECTED_RUNTIME_SECONDS):
                        disk_tiles = [tile for tile in tiles
                                      if tile > best_ram["tile_size"]
                                      and tile not in probed_disk_tiles]
                    else:
                        disk_tiles = []
                else:
                    disk_tiles = []
            elif round_index == 1:
                # A probe may pass while the full GeoTIFF fails under the cap.
                # In that case, search untried smaller RAM tiles first.
                for tile in tiles:
                    if tile in probed_ram_tiles:
                        continue
                    probed_ram_tiles.add(tile)
                    for cut in cuts:
                        directory = work / f"tile_{tile}_cut_{cut}"
                        trials.append(profile_candidate(
                            probe_model, directory, family, tile, cut, channels,
                            scores, height, width, exe, env, args.threads, job,
                            "memory", local_plan))
                disk_tiles = []
            elif round_index == 2:
                disk_tiles = [tile for tile in tiles if tile not in probed_disk_tiles]
            elif round_index == 3 and sampled_cuts:
                # Only reached when sampled structural cuts found no full run.
                # Search every remaining graph cut before declaring no plan.
                remaining = [cut for cut in range(node_count-1)
                             if cut not in cuts]
                for tile in tiles:
                    for cut in remaining:
                        for storage_mode in ("memory", "disk"):
                            directory = work / f"tile_{tile}_cut_{cut}_{storage_mode}"
                            trials.append(profile_candidate(
                                probe_model, directory, family, tile, cut,
                                channels, scores, height, width, exe, env,
                                args.threads, job, storage_mode, local_plan))
                disk_tiles = []
            else:
                # Probe smaller tiles with the same maximum halo window only
                # after every larger full-raster attempt has failed.
                restored_equivalent_tiles = list(equivalent_tiles)
                for tile in equivalent_tiles:
                    for cut in range(node_count-1):
                        for storage_mode in ("memory", "disk"):
                            directory = work / f"tile_{tile}_cut_{cut}_{storage_mode}"
                            trials.append(profile_candidate(
                                probe_model, directory, family, tile, cut,
                                channels, scores, height, width, exe, env,
                                args.threads, job, storage_mode, local_plan))
                disk_tiles = []
            for tile in disk_tiles:
                probed_disk_tiles.add(tile)
                for cut in cuts:
                    directory = work / f"tile_{tile}_cut_{cut}"
                    trials.append(profile_candidate(
                        probe_model, directory, family, tile, cut, channels,
                        scores, height, width, exe, env, args.threads, job,
                        "disk", local_plan))
            feasible = [trial for trial in trials
                        if trial["status"] == "ok" and
                        (trial["storage_mode"], trial["tile_size"],
                         trial["cut_after"]) not in attempted_plans]
            for candidate in rank_candidates(feasible) if feasible else []:
                attempted_plans.add((candidate["storage_mode"],
                                     candidate["tile_size"], candidate["cut_after"]))
                run_args = copy.copy(args)
                run_args.tile_size = candidate["tile_size"]
                run_args.cut_after = candidate["cut_after"]
                run_args.worker_mode = "pipe"
                run_args.storage_mode = candidate["storage_mode"]
                try:
                    if family == "pointwise":
                        run_pointwise(run_args, external_job=job)
                    elif family in {"local_unet", "local_graph"}:
                        run_args.general_local = family == "local_graph"
                        run_unet(run_args, external_job=job)
                    else:
                        run_global_mean(run_args, external_job=job)
                    selected = candidate
                    break
                except Exception as exc:
                    attempts.append(dict(tile_size=candidate["tile_size"],
                                         cut_after=candidate["cut_after"],
                                         storage_mode=candidate["storage_mode"],
                                         error=str(exc)))
                    failure_path = output.with_suffix(".failure.json")
                    if failure_path.exists():
                        failure = json.loads(failure_path.read_text(encoding="utf-8"))
                        if failure.get("output_published"):
                            raise RuntimeError("execution published output before failing; see failure report") from exc
            if selected is not None:
                break
        if selected is None:
            if any(trial["status"] == "ok" for trial in trials):
                raise RuntimeError("all profiled plans failed during complete-raster execution")
            raise RuntimeError("no tile/cut/storage candidate completed under the memory budget")
        report_path = output.with_suffix(".report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(strategy=f"auto-{family}",
                      selection_rule="Profile RAM plans from large to small; stop after two smaller tile sizes each predict over 1.5 times the best runtime. For local graphs over 32 nodes, defer smaller tiles with an identical halo-expanded probe shape, sample structural cuts, try disk after each failed RAM probe, and execute the first feasible large-graph candidate to limit calibration cost. If sampled plans fail, search all remaining cuts and then restore deferred tile sizes. Profile larger spill tiles for smaller graphs when the fastest predicted RAM run is at least 10 s. Probes over one million pixels use one timed repetition after warmup. Rank measured plans by predicted runtime; within 10%, prefer fewer tiles",
                      ram_search_stopped_after_tile=ram_search_stopped_after,
                      cut_candidates=list(cuts),
                      equivalent_probe_tile_sizes_skipped=equivalent_tiles,
                      equivalent_probe_tile_sizes_restored=restored_equivalent_tiles,
                      unprobed_ram_tile_sizes=[tile for tile in tiles
                                               if tile not in probed_ram_tiles],
                      planning_seconds=time.perf_counter()-started-report["elapsed_seconds"],
                      total_elapsed_seconds=time.perf_counter()-started,
                      selected_plan=selected,
                      planning_candidates=trials,
                      failed_execution_attempts=attempts,
                      job_peak_committed_bytes=job.peak_job_bytes(),
                      job_peak_conservative_upper_bound_bytes=job.conservative_peak_bytes())
        report.update(job.resource_snapshot())
        report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        output.with_suffix(".plan.failure.json").unlink(missing_ok=True)
        print(f"auto plan: {family}, tile {selected['tile_size']}, cut {selected['cut_after']}, "
              f"storage {selected['storage_mode']}; "
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
