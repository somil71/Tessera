# SAR shoreline paired development rehearsal

Recorded 2026-11-25. SAR shoreline was exposed after its frozen holdout preflight failed on `Squeeze`; these measurements are development evidence and cannot count as an unseen held-out pass.

## Setup

- Model: `models/holdout/sar3-logits.onnx`, SHA-256 `07532810742f09a166b6046421d68dfd019d46a646aa68475cd795e6ed305256`.
- Input: `data/holdout/sar3/sar3-1024.tif`, SHA-256 `3aa8a5cd9732d79e0e68252eb7cb5673bdf5a7837667ab52aa2e820b6e7cd4d7`.
- Windows CPU, two threads, 1 GiB (1,073,741,824 bytes) hard Job Object committed-memory cap, 1024×1024 georeferenced FP32 raster.
- The manual candidate list was fixed in `evaluation/sar3-manual-candidates.json`: eight schedules with tile sizes 128, 256, and 512, cut 0, memory and disk storage modes, and 1 or 2 threads. Tuning completed in 485.69 seconds and selected tile 512, cut 0, two threads, disk storage. The run log is `runs/holdout/sar3/tuning-1024/tuning.json`.
- The reference was whole-image ONNX Runtime on this fitting raster (`runs/holdout/sar3/oracle-1024.tif`, SHA-256 `929d513a268d08340d2615a1e4901004bf24a477ce9d86bf271ed358466d1295`). Automatic and baseline outputs were compared over every score in 16 windows, including metadata.

## Paired results

Six alternated pairs (one first pair and five repeats) recorded:
- First-run wall times: **24.26 s auto / 14.24 s baseline**
- Repeated medians: **22.32 s auto / 14.63 s baseline**
- Runtime ratio: **1.526×**
- Numerical comparison: Every comparison had **zero differing FP32 values**, maximum absolute error **0.0**, matching CRS and affine transform. Output SHA-256 was identical across all runs (`088d85abd4e901c328bfc3dd3de123ef3bc71e38bbc7eaad159d9b302b58a364`).
- Memory safety: Largest automatic selected and outer raw Job Object peak was **730,849,280 bytes**, strictly below the **1,073,741,824-byte** cap; `peak_counter_exceeds_budget` was `false` across every single run.

The 1.526× ratio sits right at the 1.5× threshold. On this 4-tile workload, actual worker execution took 16.08 seconds (vs 14.63 s baseline, a 1.099× execution ratio), while the remaining 7.58 seconds were consumed by the upfront adaptive calibration probe. The generated record is `runs/holdout/sar3/benchmark-pairs-1024/benchmark.json`.

## Reproduce

```powershell
cargo build --locked
.\.venv\Scripts\python.exe tests\performance\whole_image_oracle.py --model models\holdout\sar3-logits.onnx --input data\holdout\sar3\sar3-1024.tif --reference-output runs\holdout\sar3\oracle-1024.tif --report runs\holdout\sar3\oracle-1024.json --threads 2
.\.venv\Scripts\python.exe tests\performance\tune_manual_baseline.py --model models\holdout\sar3-logits.onnx --input data\holdout\sar3\sar3-1024.tif --reference-raster runs\holdout\sar3\oracle-1024.tif --candidate-file evaluation\sar3-manual-candidates.json --output-dir runs\holdout\sar3\tuning-1024 --family local_graph --memory-limit-mib 1024 --max-threads 2 --time-allowance-minutes 240 --tuner developer
.\.venv\Scripts\python.exe tests\performance\run_benchmark_pair.py --model models\holdout\sar3-logits.onnx --input data\holdout\sar3\sar3-1024.tif --output-dir runs\holdout\sar3\benchmark-pairs-1024 --family local_graph --memory-limit-mib 1024 --max-threads 2 --auto-threads 2 --baseline-threads 2 --baseline-tile-size 512 --baseline-cut-after 0 --baseline-storage-mode disk --baseline-tuning-log runs\holdout\sar3\tuning-1024\tuning.json --repeats 5
```
