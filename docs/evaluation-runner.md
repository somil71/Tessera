# Alternated benchmark runner

`tests/performance/run_benchmark_pair.py` runs one model/input/memory-cap pair with an automatic schedule and one **preselected** manual schedule. It makes a first-run pair, then five repeated pairs by default. It reverses execution order for each pair, keeps each output and raw process record, checks the enforced job-wide committed-memory bound using [amendment 2](evaluation-amendment-2.md), and compares every automatic output against its paired baseline output in bounded raster windows. The summary records wall-clock medians and their ratio. The coordinator's wall time includes Python startup, model loading, automatic planning, raster reads, and output writes.

This runner does **not** tune the manual schedule, establish that a model was independently held out, or compute a whole-image numerical oracle. Use the [manual tuning tool](manual-baseline-tuning.md) with a predeclared candidate list and a frozen reference raster. For five repeated pairs, the benchmark checks its structured manifest, four-hour allowance, selected schedule, matching hashes, and retained artifacts. It cannot certify the candidate search's independence or quality. Freeze a genuinely new held-out model and its input hashes before using this for the feasibility decision. A ratio band from a development smoke run is not a protocol pass or failure.

## Small whole-image numerical oracle

For a crop that fits in RAM, run the model **unsplit** in ONNX Runtime before evaluating a scheduled output. The oracle enforces a pixel-count guard, requires a georeferenced raster without masked or non-finite pixels, and writes the raw FP32 score raster with matching CRS, transform, and source tags. It records hashes and ONNX Runtime version. With `--candidate`, it checks every raw score against the protocol's absolute and eligible-relative tolerances and reports the location of the largest error. A failed comparison exits nonzero but retains the reference and report for diagnosis.

```powershell
.\.venv\Scripts\python.exe tests\performance\whole_image_oracle.py `
  --model models\evaluation\unet_dev.onnx `
  --input data\small_unet_crop.tif `
  --reference-output runs\oracle\unet_reference.tif `
  --candidate runs\small_unet_auto.tif `
  --report runs\oracle\unet_report.json `
  --threads 2 --max-pixels 1048576
```

Supply the same `--band-order`, `--scale`, and `--offset` used by the scheduled runner. For reference generation without an existing candidate, omit `--candidate`. The reference output and report must be new paths. The tool does not enforce a memory cap: the small-crop bound is explicit, while the scheduled large-image runs use the Windows Job Object cap.

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
