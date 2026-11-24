"""Run a predeclared manual schedule list and retain every baseline attempt."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from run_benchmark_pair import RUNNERS, COMPARE


LOCAL_FAMILIES = {"local_unet", "local_graph"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-raster", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=RUNNERS, required=True)
    parser.add_argument("--memory-limit-mib", type=int, required=True)
    parser.add_argument("--max-threads", type=int, required=True)
    parser.add_argument("--time-allowance-minutes", type=float, default=240.0)
    parser.add_argument("--tuner", required=True)
    parser.add_argument("--band-order")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    return parser.parse_args()


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def load_candidates(path, family, max_threads):
    source = json.loads(path.read_text(encoding="utf-8"))
    candidates = source.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate file needs a nonempty candidates list")
    checked = []
    seen = set()
    for index, item in enumerate(candidates):
        if not isinstance(item, dict) or set(item) != {
                "tile_size", "cut_after", "threads", "storage_mode"}:
            raise ValueError(f"candidate {index} has missing or extra fields")
        tile, cut, threads, storage = (item[key] for key in
                                       ("tile_size", "cut_after", "threads", "storage_mode"))
        if (type(tile) is not int or tile < 2 or type(cut) is not int or cut < 0 or
                type(threads) is not int or not 1 <= threads <= max_threads or
                storage not in ("memory", "disk") or
                (family not in LOCAL_FAMILIES and storage != "memory")):
            raise ValueError(f"candidate {index} has an invalid schedule")
        key = (tile, cut, threads, storage)
        if key in seen:
            raise ValueError(f"duplicate candidate {index}: {key}")
        seen.add(key)
        checked.append(dict(tile_size=tile, cut_after=cut,
                            threads=threads, storage_mode=storage))
    return checked


def invoke(command, path):
    started = time.perf_counter()
    result = subprocess.run([str(part) for part in command],
                            capture_output=True, text=True)
    record = dict(command=[str(part) for part in command],
                  exit_code=result.returncode,
                  wall_seconds=time.perf_counter()-started,
                  stdout=result.stdout, stderr=result.stderr)
    write_json(path, record)
    return record


def run(args):
    started = time.perf_counter()
    if (args.memory_limit_mib < 1 or args.max_threads < 1 or
            args.time_allowance_minutes <= 0 or not args.tuner.strip()):
        raise ValueError("cap, thread limit, allowance, and tuner must be positive/nonempty")
    model = args.model.resolve(strict=True)
    source = args.input.resolve(strict=True)
    reference = args.reference_raster.resolve(strict=True)
    candidate_file = args.candidate_file.resolve(strict=True)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"tuning directory must be empty: {out}")
    candidates = load_candidates(candidate_file, args.family, args.max_threads)
    out.mkdir(parents=True, exist_ok=True)
    cap = args.memory_limit_mib*1024*1024
    manifest = dict(schema="raster-compiler-manual-tuning-v1", status="running",
                    tuner=args.tuner, family=args.family,
                    model=str(model), model_sha256=sha256(model),
                    input=str(source), input_sha256=sha256(source),
                    reference_raster=str(reference), reference_sha256=sha256(reference),
                    candidate_file=str(candidate_file),
                    candidate_file_sha256=sha256(candidate_file),
                    time_allowance_minutes=args.time_allowance_minutes,
                    memory_limit_bytes=cap, max_threads=args.max_threads,
                    band_order=args.band_order, scale=args.scale, offset=args.offset,
                    environment=dict(platform=platform.platform(),
                                     processor=platform.processor(),
                                     python=sys.version.split()[0],
                                     onnx=metadata.version("onnx"),
                                     onnxruntime=metadata.version("onnxruntime"),
                                     numpy=metadata.version("numpy"),
                                     rasterio=metadata.version("rasterio")),
                    declared_candidates=candidates, attempts=[])
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2],
                              capture_output=True, text=True)
    manifest["source_commit"] = (revision.stdout.strip()
                                 if revision.returncode == 0 else None)
    summary_path = out / "tuning.json"
    write_json(summary_path, manifest)
    for index, schedule in enumerate(candidates):
        if (time.perf_counter()-started)/60 >= args.time_allowance_minutes:
            manifest["stopped_after_allowance"] = True
            break
        stem = f"{index:03d}"
        output = out / f"{stem}.tif"
        command = [sys.executable, RUNNERS[args.family],
                   "--model", model, "--input", source, "--output", output,
                   "--memory-limit-mib", args.memory_limit_mib,
                   "--tile-size", schedule["tile_size"],
                   "--cut-after", schedule["cut_after"],
                   "--threads", schedule["threads"],
                   "--scale", args.scale, "--offset", args.offset]
        if args.family in LOCAL_FAMILIES:
            command.extend(["--storage-mode", schedule["storage_mode"]])
        if args.band_order:
            command.extend(["--band-order", args.band_order])
        attempt = dict(index=index, schedule=schedule, output=str(output),
                       process_record=f"{stem}.process.json", status="running")
        manifest["attempts"].append(attempt)
        write_json(summary_path, manifest)
        process = invoke(command, out / f"{stem}.process.json")
        attempt.update(exit_code=process["exit_code"],
                       wall_seconds=process["wall_seconds"],
                       process_record_sha256=sha256(out / f"{stem}.process.json"))
        report_path = output.with_suffix(".report.json")
        if process["exit_code"] != 0 or not output.exists() or not report_path.exists():
            attempt.update(status="failed", reason="runner failed or omitted output/report")
            write_json(summary_path, manifest)
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        attempt.update(report=str(report_path), report_sha256=sha256(report_path),
                       output_sha256=sha256(output),
                       enforced_job_limit_bytes=report.get("job_enforced_limit_bytes"),
                       sampled_private_peak_bytes=report.get(
                           "job_peak_private_commit_sampled_bytes"),
                       raw_job_peak_bytes=report.get("job_peak_committed_bytes"))
        if (report.get("memory_limit_enforced") is not True or
                report.get("memory_limit_bytes") != cap or
                not isinstance(attempt["enforced_job_limit_bytes"], int) or
                attempt["enforced_job_limit_bytes"] > cap or
                not isinstance(attempt["sampled_private_peak_bytes"], int) or
                attempt["sampled_private_peak_bytes"] > cap or
                report.get("job_peak_counter_exceeds_budget") is True):
            attempt.update(status="failed", reason="memory evidence failed")
            write_json(summary_path, manifest)
            continue
        comparison = out / f"{stem}.comparison.json"
        compared = invoke([sys.executable, COMPARE, reference, output,
                           "--report", comparison],
                          out / f"{stem}.comparison.process.json")
        attempt["comparison"] = str(comparison)
        attempt["comparison_process_sha256"] = sha256(
            out / f"{stem}.comparison.process.json")
        if comparison.exists():
            attempt["comparison_sha256"] = sha256(comparison)
        if (compared["exit_code"] != 0 or not comparison.exists() or
                json.loads(comparison.read_text(encoding="utf-8")).get("passed") is not True):
            attempt.update(status="failed", reason="reference raster mismatch")
        else:
            attempt["status"] = "valid"
        write_json(summary_path, manifest)
    valid = [item for item in manifest["attempts"] if item["status"] == "valid"]
    manifest["elapsed_seconds"] = time.perf_counter()-started
    manifest["allowance_exceeded"] = (
        manifest["elapsed_seconds"] > args.time_allowance_minutes*60)
    manifest["unattempted_candidates"] = len(candidates)-len(manifest["attempts"])
    if valid:
        selected = min(valid, key=lambda item: item["wall_seconds"])
        manifest.update(status="completed", selected_attempt_index=selected["index"],
                        selected_schedule=selected["schedule"],
                        selected_wall_seconds=selected["wall_seconds"])
    else:
        manifest["status"] = "no_valid_schedule"
    write_json(summary_path, manifest)
    if not valid:
        raise RuntimeError(f"no valid manual schedule; inspect {summary_path}")
    print(summary_path)


if __name__ == "__main__":
    run(arguments())
