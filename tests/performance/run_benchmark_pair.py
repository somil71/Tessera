"""Alternate automatic and frozen manual schedules; retain every raw run.

This measures a supplied baseline. It does not tune or certify that baseline.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {
    "pointwise": ROOT / "scripts" / "run_raster_fixed.py",
    "local_unet": ROOT / "scripts" / "run_raster_unet.py",
    "local_graph": ROOT / "scripts" / "run_raster_local.py",
    "global_mean": ROOT / "scripts" / "run_raster_global_mean.py",
}
AUTO = ROOT / "scripts" / "run_raster_auto.py"
COMPARE = ROOT / "tests" / "performance" / "compare_rasters.py"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=RUNNERS, required=True)
    parser.add_argument("--memory-limit-mib", type=int, required=True)
    parser.add_argument("--max-threads", type=int, required=True)
    parser.add_argument("--auto-threads", type=int, required=True)
    parser.add_argument("--baseline-threads", type=int, required=True)
    parser.add_argument("--baseline-tile-size", type=int, required=True)
    parser.add_argument("--baseline-cut-after", type=int, default=0)
    parser.add_argument("--baseline-storage-mode", choices=("memory", "disk"),
                        default="memory")
    parser.add_argument("--max-auto-tile-size", type=int, default=512)
    parser.add_argument("--baseline-tuning-log", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5,
                        help="Repeated pairs after the first-run pair")
    parser.add_argument("--band-order")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    return parser.parse_args()


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_tuning_manifest(path, args, model_hash, input_hash, cap):
    try:
        tuning = read_json(path)
    except (ValueError, UnicodeError):
        if args.repeats >= 5:
            raise ValueError("five-repeat evaluation requires a structured tuning manifest")
        return False
    if not isinstance(tuning, dict) or tuning.get("schema") != "raster-compiler-manual-tuning-v1":
        if args.repeats >= 5:
            raise ValueError("five-repeat evaluation requires a v1 tuning manifest")
        return False
    selected = tuning.get("selected_schedule", {})
    attempts = tuning.get("attempts", [])
    selected_attempt = next((item for item in attempts
                             if item.get("index") == tuning.get("selected_attempt_index")),
                            None)
    def artifact_matches(file_path, expected_hash):
        return (isinstance(file_path, str) and isinstance(expected_hash, str) and
                Path(file_path).is_file() and sha256(Path(file_path)) == expected_hash)

    expected = dict(tile_size=args.baseline_tile_size,
                    cut_after=args.baseline_cut_after,
                    threads=args.baseline_threads,
                    storage_mode=args.baseline_storage_mode)
    checks = [
        tuning.get("status") == "completed",
        tuning.get("model_sha256") == model_hash,
        tuning.get("input_sha256") == input_hash,
        tuning.get("family") == args.family,
        tuning.get("memory_limit_bytes") == cap,
        tuning.get("max_threads") == args.max_threads,
        tuning.get("band_order") == args.band_order,
        tuning.get("scale") == args.scale,
        tuning.get("offset") == args.offset,
        selected == expected,
        tuning.get("selected_attempt_index") is not None,
        selected_attempt is not None and selected_attempt.get("status") == "valid" and
            selected_attempt.get("schedule") == expected,
        isinstance(attempts, list) and
            len(attempts) <= len(tuning.get("declared_candidates", [])),
        artifact_matches(tuning.get("candidate_file"),
                         tuning.get("candidate_file_sha256")),
        artifact_matches(tuning.get("reference_raster"),
                         tuning.get("reference_sha256")),
        all(item.get("schedule") == tuning.get("declared_candidates", [])[index] and
            artifact_matches(str(path.parent / item.get("process_record", "")),
                             item.get("process_record_sha256"))
            for index, item in enumerate(attempts)
            if index < len(tuning.get("declared_candidates", []))),
    ]
    if selected_attempt is not None:
        checks.extend([
            artifact_matches(selected_attempt.get("output"),
                             selected_attempt.get("output_sha256")),
            artifact_matches(selected_attempt.get("report"),
                             selected_attempt.get("report_sha256")),
            artifact_matches(selected_attempt.get("comparison"),
                             selected_attempt.get("comparison_sha256")),
        ])
    if args.repeats >= 5:
        checks.extend([
            tuning.get("time_allowance_minutes") == 240,
            tuning.get("allowance_exceeded") is False,
            isinstance(tuning.get("elapsed_seconds"), (int, float)) and
                tuning["elapsed_seconds"] <= 240*60,
        ])
    if not all(checks):
        raise ValueError("tuning manifest does not match the frozen benchmark pair")
    return True


def invoke(command, record_path):
    started = time.perf_counter()
    result = subprocess.run([str(item) for item in command],
                            capture_output=True, text=True)
    record = dict(command=[str(item) for item in command],
                  exit_code=result.returncode,
                  wall_seconds=time.perf_counter()-started,
                  stdout=result.stdout, stderr=result.stderr)
    record_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    return record


def run(args):
    model = args.model.resolve(strict=True)
    source = args.input.resolve(strict=True)
    tuning_log = args.baseline_tuning_log.resolve(strict=True)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"benchmark directory must be empty: {out}")
    if args.repeats < 1 or args.memory_limit_mib < 1:
        raise ValueError("repeats and memory limit must be positive")
    if not (1 <= args.auto_threads <= args.max_threads and
            1 <= args.baseline_threads <= args.max_threads):
        raise ValueError("runner threads must be within the declared maximum")
    if args.baseline_tile_size < 2 or args.max_auto_tile_size < 2:
        raise ValueError("tile sizes must be at least two")
    out.mkdir(parents=True, exist_ok=True)
    cap = args.memory_limit_mib * 1024 * 1024
    model_hash = sha256(model)
    input_hash = sha256(source)
    tuning_manifest_validated = validate_tuning_manifest(
        tuning_log, args, model_hash, input_hash, cap)
    shared = ["--model", str(model), "--input", str(source),
              "--memory-limit-mib", str(args.memory_limit_mib),
              "--scale", str(args.scale), "--offset", str(args.offset)]
    if args.band_order:
        shared.extend(["--band-order", args.band_order])
    manifest = dict(status="running", model=str(model), model_sha256=model_hash,
                    input=str(source), input_sha256=input_hash,
                    baseline_tuning_log=str(tuning_log),
                    baseline_tuning_log_sha256=sha256(tuning_log),
                    tuning_manifest_validated=tuning_manifest_validated,
                    family=args.family, memory_limit_bytes=cap,
                    max_threads=args.max_threads,
                    auto_threads=args.auto_threads,
                    baseline_threads=args.baseline_threads,
                    baseline_tile_size=args.baseline_tile_size,
                    baseline_cut_after=args.baseline_cut_after,
                    baseline_storage_mode=args.baseline_storage_mode,
                    repeats=args.repeats, pairs=[])
    summary_path = out / "benchmark.json"
    summary_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    for index in range(args.repeats + 1):
        # Reverse order every pair so one method does not always receive the
        # warmer filesystem cache. First pair remains explicitly identifiable.
        order = ("auto", "baseline") if index % 2 == 0 else ("baseline", "auto")
        pair = dict(index=index, kind="first" if index == 0 else "repeat",
                    order=list(order), runs={})
        manifest["pairs"].append(pair)
        for method in order:
            stem = f"{index:02d}_{method}"
            output = out / f"{stem}.tif"
            command = [sys.executable, AUTO if method == "auto" else RUNNERS[args.family],
                       *shared, "--output", str(output), "--threads",
                       str(args.auto_threads if method == "auto" else args.baseline_threads)]
            if method == "auto":
                command.extend(["--max-tile-size", str(args.max_auto_tile_size)])
            else:
                command.extend(["--tile-size", str(args.baseline_tile_size),
                                "--cut-after", str(args.baseline_cut_after)])
                if args.family in {"local_unet", "local_graph"}:
                    command.extend(["--storage-mode", args.baseline_storage_mode])
            raw = invoke(command, out / f"{stem}.process.json")
            run = dict(output=str(output), process_record=f"{stem}.process.json",
                       exit_code=raw["exit_code"], wall_seconds=raw["wall_seconds"])
            report_path = output.with_suffix(".report.json")
            if raw["exit_code"] == 0 and report_path.exists() and output.exists():
                report = read_json(report_path)
                run.update(report=str(report_path), output_sha256=sha256(output),
                           report_sha256=sha256(report_path),
                           reported_seconds=report.get("total_elapsed_seconds",
                                                       report.get("elapsed_seconds")),
                           peak_job_commit_bytes=report.get("job_peak_committed_bytes"),
                           peak_private_commit_sampled_bytes=report.get(
                               "job_peak_private_commit_sampled_bytes"),
                           peak_counter_exceeds_budget=report.get(
                               "job_peak_counter_exceeds_budget"),
                           enforced_job_limit_bytes=report.get("job_enforced_limit_bytes"),
                           peak_job_working_set_sampled_bytes=report.get(
                               "job_peak_working_set_sampled_bytes"),
                           job_io_read_transfer_bytes=report.get("job_io_read_transfer_bytes"),
                           job_io_write_transfer_bytes=report.get("job_io_write_transfer_bytes"),
                           conservative_peak_bytes=report.get(
                               "job_peak_conservative_upper_bound_bytes"))
                if method == "auto":
                    run.update(overall_peak_job_commit_bytes=report.get(
                                   "overall_job_peak_committed_bytes"),
                               overall_peak_private_commit_sampled_bytes=report.get(
                                   "overall_job_peak_private_commit_sampled_bytes"),
                               overall_peak_counter_exceeds_budget=report.get(
                                   "overall_job_peak_counter_exceeds_budget"),
                               overall_enforced_job_limit_bytes=report.get(
                                   "overall_job_enforced_limit_bytes"))
                if (report.get("memory_limit_enforced") is not True or
                        report.get("memory_limit_bytes") != cap or
                        run["conservative_peak_bytes"] is None or
                        run["enforced_job_limit_bytes"] is None or
                        run["enforced_job_limit_bytes"] > cap or
                        run["peak_private_commit_sampled_bytes"] is None or
                        run["peak_private_commit_sampled_bytes"] > cap):
                    run["validation_error"] = "job-wide memory evidence failed"
                elif (method == "auto" and
                      (report.get("planning_memory_limit_enforced") is not True or
                       not isinstance(run["overall_enforced_job_limit_bytes"], int) or
                       run["overall_enforced_job_limit_bytes"] > cap or
                       not isinstance(run["overall_peak_private_commit_sampled_bytes"], int) or
                       run["overall_peak_private_commit_sampled_bytes"] > cap)):
                    run["validation_error"] = "planning job memory evidence failed"
                elif (not isinstance(run["peak_job_working_set_sampled_bytes"], int) or
                      run["peak_job_working_set_sampled_bytes"] <= 0 or
                      not isinstance(run["job_io_read_transfer_bytes"], int) or
                      not isinstance(run["job_io_write_transfer_bytes"], int)):
                    run["validation_error"] = "job resource accounting is missing"
                elif run["peak_counter_exceeds_budget"] is True:
                    run["validation_error"] = (
                        "selected execution job peak exceeded the cap"
                    )
                elif (method == "auto" and
                      run["overall_peak_counter_exceeds_budget"] is True and
                      not any(item.get("status") == "failed" for item in
                              report.get("planning_candidates", [])) and
                      not report.get("failed_execution_attempts")):
                    run["validation_error"] = (
                        "outer planning job peak exceeded the cap without a recorded failed attempt"
                    )
            else:
                run["validation_error"] = "runner failed or omitted output/report"
            pair["runs"][method] = run
            summary_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
            if "validation_error" in run:
                manifest["status"] = "failed"
                summary_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
                raise RuntimeError(f"{stem}: {run['validation_error']}; see process record")
        comparison = out / f"{index:02d}_comparison.json"
        raw = invoke([sys.executable, COMPARE,
                      pair["runs"]["baseline"]["output"],
                      pair["runs"]["auto"]["output"],
                      "--report", comparison],
                     out / f"{index:02d}_comparison.process.json")
        pair["comparison"] = str(comparison)
        pair["comparison_exit_code"] = raw["exit_code"]
        compared = read_json(comparison) if comparison.exists() else {}
        if (raw["exit_code"] != 0 or not compared.get("passed") or
                compared.get("reference_sha256") != pair["runs"]["baseline"]["output_sha256"] or
                compared.get("candidate_sha256") != pair["runs"]["auto"]["output_sha256"]):
            manifest["status"] = "failed"
            summary_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
            raise RuntimeError(f"pair {index}: output comparison failed")
        summary_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    repeated = manifest["pairs"][1:]
    auto_median = statistics.median(pair["runs"]["auto"]["wall_seconds"]
                                    for pair in repeated)
    baseline_median = statistics.median(pair["runs"]["baseline"]["wall_seconds"]
                                        for pair in repeated)
    manifest.update(status="completed",
                    first_run_seconds={method: manifest["pairs"][0]["runs"][method]["wall_seconds"]
                                       for method in ("auto", "baseline")},
                    repeated_median_seconds=dict(auto=auto_median,
                                                 baseline=baseline_median),
                    runtime_ratio=auto_median/baseline_median,
                    runtime_ratio_band=(
                        "within_1_25" if auto_median <= 1.25*baseline_median
                        else "above_1_5" if auto_median > 1.5*baseline_median
                        else "between_1_25_and_1_5"))
    summary_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    run(arguments())
