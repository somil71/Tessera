# Alternated benchmark runner

`tests/performance/run_benchmark_pair.py` runs one model/input/memory-cap pair with an automatic schedule and one **preselected** manual schedule. It makes a first-run pair, then five repeated pairs by default. It reverses execution order for each pair, keeps each output and raw process record, checks the enforced job-wide committed-memory bound using [amendment 2](evaluation-amendment-2.md), and compares every automatic output against its paired baseline output in bounded raster windows. The summary records wall-clock medians and their ratio. The coordinator's wall time includes Python startup, model loading, automatic planning, raster reads, and output writes.

This runner does **not** tune the manual schedule, certify the four-hour tuning allowance, establish that a model was independently held out, or compute a whole-image numerical oracle. Supply and retain a tuning log made before benchmarking, with every attempted manual configuration, its results, the tuner, and the time allowance. Freeze a genuinely new held-out model and its input hashes before using this for the feasibility decision. A ratio band from a development smoke run is not a protocol pass or failure.

Example for a preselected U-Net schedule:

```powershell
.\.venv\Scripts\python.exe tests\performance\run_benchmark_pair.py `
  --model models\evaluation\unet_dev.onnx `
  --input data\large_unet_3101.tif `
  --output-dir runs\unet_benchmark `
  --family local_unet `
  --memory-limit-mib 2048 `
  --max-threads 4 --auto-threads 4 --baseline-threads 4 `
  --baseline-tile-size 256 --baseline-cut-after 4 `
  --baseline-tuning-log runs\unet_tuning_log.json
```

The output directory must be empty. `benchmark.json` is updated after each run, so a failure retains the completed evidence and its process stdout/stderr. The script handles pointwise, frozen U-Net, general local graph, and global-mean schedules. For either local family, `--baseline-storage-mode memory|disk` fixes the manual boundary-storage choice. Attention needs a separately defined manual block schedule before its baseline can be evaluated. For each model-machine pair, keep `--repeats 5` and document CPU affinity, machine configuration, temporary storage, input/model hashes, and tuning history. The full pass/fail rule remains in the [evaluation protocol](evaluation-protocol.md).
