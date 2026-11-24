# Auditable manual baseline tuning

Recorded 2026-11-24. `tests/performance/tune_manual_baseline.py` executes a **human-declared** list of fixed schedules under a stated wall-clock allowance. It does not design candidates or make the tuner independent of the planner author. Every attempted configuration keeps its command, stdout/stderr, output GeoTIFF, runner report, resource counters, and comparison against a supplied reference raster. The manifest hashes the model, input, reference, candidate file, records, and selected artifacts. The fastest valid completed attempt becomes the provisional manual baseline.

The candidate file is JSON. Keep the order chosen before running:

```json
{
  "candidates": [
    {"tile_size": 128, "cut_after": 3, "threads": 2, "storage_mode": "memory"},
    {"tile_size": 256, "cut_after": 0, "threads": 2, "storage_mode": "memory"},
    {"tile_size": 256, "cut_after": 3, "threads": 2, "storage_mode": "disk"}
  ]
}
```

For a real evaluation, freeze and verify the model, input, candidate file, reference raster, tuner identity, memory cap, thread limit, and four-hour allowance before starting. The reference raster must itself have an established correctness basis; comparison to it does not create one. A valid run must finish with the configured hard Job Object limit, sampled private commitment at or below the cap, and output matching the reference. Raw peak-counter overshoot without a recorded failed attempt is not accepted for a fixed candidate. A candidate may run past the allowance because the tuner does not forcibly terminate a Windows process tree; the manifest flags such an overrun, and the five-repeat benchmark rejects it.

```powershell
.\.venv\Scripts\python.exe tests\performance\tune_manual_baseline.py `
  --model models\your_model.onnx --input data\your_image.tif `
  --reference-raster runs\frozen_reference.tif `
  --candidate-file runs\manual_candidates.json `
  --output-dir runs\manual_tuning --family local_graph `
  --memory-limit-mib 1024 --max-threads 2 `
  --time-allowance-minutes 240 --tuner "Name or team"
```

`runs\manual_tuning\tuning.json` records the selected tile, cut, threads, and storage mode. Pass that path unchanged to the [alternated benchmark runner](evaluation-runner.md) with matching `--baseline-*` flags. For five repeated pairs, the benchmark requires the v1 structured manifest, an exact four-hour allowance with no overrun, matching model/input hashes and configuration, a valid selected attempt, and intact candidate/reference/selected artifacts and process records. This check enforces provenance; it cannot prove that a human search was competent or independent.

`tests/fixtures/smoke_manual_tuner.py` exercises two tiny development schedules with a one-minute allowance and one benchmark repeat. It also verifies rejection of an altered schedule, a modified candidate file, and a five-repeat run using that smoke allowance. This is tool validation only. No four-hour tuned baseline or independent held-out model-machine pair has been evaluated yet.
