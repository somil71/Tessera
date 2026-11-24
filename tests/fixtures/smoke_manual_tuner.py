"""Check that tuning logs and benchmark selection agree on one model/input pair."""

import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, ROOT / "tests" / "fixtures" /
                "smoke_generic_local_raster.py"], check=True, capture_output=True, text=True)
fixture = ROOT / "runs" / "generic_local_raster_smoke"
work = ROOT / "runs" / "manual_tuner_smoke" / uuid4().hex
work.mkdir(parents=True)
model = ROOT / "runs" / "local_graph_smoke" / "branched.onnx"
source = fixture / "input.tif"
reference = fixture / "fixed.tif"
candidates = work / "candidates.json"
candidates.write_text(json.dumps({"candidates": [
    {"tile_size": 16, "cut_after": 3, "threads": 2, "storage_mode": "memory"},
    {"tile_size": 8, "cut_after": 0, "threads": 2, "storage_mode": "memory"},
]}, indent=2)+"\n", encoding="utf-8")
tuning_dir = work / "tuning"
subprocess.run([sys.executable, ROOT / "tests" / "performance" /
                "tune_manual_baseline.py", "--model", model, "--input", source,
                "--reference-raster", reference, "--candidate-file", candidates,
                "--output-dir", tuning_dir, "--family", "local_graph",
                "--memory-limit-mib", "128", "--max-threads", "2",
                "--time-allowance-minutes", "1", "--tuner", "development-smoke"],
               check=True, capture_output=True, text=True)
tuning = json.loads((tuning_dir / "tuning.json").read_text())
assert tuning["status"] == "completed"
assert len(tuning["attempts"]) == 2
assert all(item["status"] == "valid" for item in tuning["attempts"])
selected = tuning["selected_schedule"]
benchmark = work / "benchmark"
subprocess.run([sys.executable, ROOT / "tests" / "performance" /
                "run_benchmark_pair.py", "--model", model, "--input", source,
                "--output-dir", benchmark, "--family", "local_graph",
                "--memory-limit-mib", "128", "--max-threads", "2",
                "--auto-threads", "2", "--baseline-threads", str(selected["threads"]),
                "--baseline-tile-size", str(selected["tile_size"]),
                "--baseline-cut-after", str(selected["cut_after"]),
                "--baseline-storage-mode", selected["storage_mode"],
                "--max-auto-tile-size", "32",
                "--baseline-tuning-log", tuning_dir / "tuning.json",
                "--repeats", "1"], check=True, capture_output=True, text=True)
result = json.loads((benchmark / "benchmark.json").read_text())
assert result["status"] == "completed"
assert result["tuning_manifest_validated"] is True
assert len(result["pairs"]) == 2
base = [sys.executable, ROOT / "tests" / "performance" / "run_benchmark_pair.py",
        "--model", model, "--input", source, "--family", "local_graph",
        "--memory-limit-mib", "128", "--max-threads", "2", "--auto-threads", "2",
        "--baseline-threads", str(selected["threads"]),
        "--baseline-cut-after", str(selected["cut_after"]),
        "--baseline-storage-mode", selected["storage_mode"],
        "--baseline-tuning-log", tuning_dir / "tuning.json"]
wrong_allowance = subprocess.run([*base, "--output-dir", work / "wrong_allowance",
                                  "--baseline-tile-size", str(selected["tile_size"]),
                                  "--repeats", "5"], capture_output=True, text=True)
assert wrong_allowance.returncode != 0
assert "tuning manifest does not match" in wrong_allowance.stderr
wrong_schedule = subprocess.run([*base, "--output-dir", work / "wrong_schedule",
                                 "--baseline-tile-size", "32", "--repeats", "1"],
                                capture_output=True, text=True)
assert wrong_schedule.returncode != 0
assert "tuning manifest does not match" in wrong_schedule.stderr
candidates.write_text(candidates.read_text(encoding="utf-8")+"\n", encoding="utf-8")
wrong_artifact = subprocess.run([*base, "--output-dir", work / "wrong_artifact",
                                 "--baseline-tile-size", str(selected["tile_size"]),
                                 "--repeats", "1"], capture_output=True, text=True)
assert wrong_artifact.returncode != 0
assert "tuning manifest does not match" in wrong_artifact.stderr
print(f"manual tuner retained {len(tuning['attempts'])} valid attempts; "
      f"benchmark selected {selected}; {work}")
