# FraXet paired development rehearsal

Recorded 2026-11-24. FraXet was exposed after its frozen holdout preflight failed on `Div`; these measurements are development evidence and cannot count as an unseen held-out pass.

## Setup

- Model: `models/holdout/fraxet-unet-rgb.onnx`, SHA-256 `b50e3c714a0447fae3da175c999c239b503cb98c0c3363e8d790f7c0fd7948fe`.
- Input: `data/holdout/fraxet/fraxet-6203-512.tif`, SHA-256 `2c47ec9ddf763c71c36f121335d231b54ae2b26ca9a84c55fcf545b6f6cee68c`.
- Windows CPU, two threads, 1 GiB hard Job Object committed-memory cap, 512×512 georeferenced FP32 raster.
- The manual candidate list was fixed in `evaluation/fraxet-manual-candidates.json`: ten schedules and a 240-minute allowance. All ten completed and were cap-valid. Tuning took 164.62 seconds and selected tile 512, cut 0, two threads, memory storage. The run log is `runs/fraxet-development-tuning-512/tuning.json` (generated, ignored by Git).
- The reference was whole-image ONNX Runtime on this fitting raster. Automatic and baseline outputs were compared over every score in four windows, including metadata.

## Paired results

The first strict five-pair run, before the direct one-tile path, recorded median auto/baseline wall ratio **1.847×**. It passed correctness and the selected execution cap, but failed the proposed 1.25× performance target. The main cost was redundant profiling of a full-image tile before executing the same schedule. Its generated record is `runs/fraxet-development-pairs-512/benchmark.json`.

A guarded direct path now attempts a single full-image RAM tile under the hard cap when it covers a local or pointwise raster. If that attempt fails before publishing output, the normal calibrated search continues and records the failure. Large local graphs still use a single cold timed inference when calibration is needed, avoiding repeated session commitment during a probe. A forced-failure fixture verified fallback to a smaller RAM tile and exact output.

After the change, six alternated pairs (one first pair and five repeats) recorded first-run wall times **4.531 s auto / 3.761 s baseline**, and repeated medians **4.508 s auto / 3.676 s baseline**, ratio **1.227×**. All six automatic runs selected the direct schedule without manual tile or cut input. Every comparison had zero differing FP32 values, maximum absolute and eligible-relative error 0, matching CRS and transform. The largest automatic selected and outer raw Job Object peak was **927,260,672 bytes**, below the **1,073,741,824-byte** cap; largest sampled working set was **836,857,856 bytes**. The generated record is `runs/fraxet-development-pairs-direct-512/benchmark.json`.

The improvement is meaningful but narrow: a fitting one-tile raster is an easy case for a manual baseline. The 1.227× ratio has little margin to the 1.25× threshold, and this input was used while developing the planner. It does not validate larger FraXet rasters, other models, disk scheduling, a second machine, or independent user adoption.

## Reproduce

```powershell
.\.venv\Scripts\python.exe tests\performance\tune_manual_baseline.py --model models\holdout\fraxet-unet-rgb.onnx --input data\holdout\fraxet\fraxet-6203-512.tif --reference-raster runs\holdout\fraxet\dev-oracle-512.tif --output-dir runs\fraxet-development-tuning-512 --family local_graph --memory-limit-mib 1024 --max-threads 2 --candidate-file evaluation\fraxet-manual-candidates.json --time-allowance-minutes 240 --tuner developer
.\.venv\Scripts\python.exe tests\performance\run_benchmark_pair.py --model models\holdout\fraxet-unet-rgb.onnx --input data\holdout\fraxet\fraxet-6203-512.tif --output-dir runs\fraxet-development-pairs-direct-512 --family local_graph --memory-limit-mib 1024 --max-threads 2 --auto-threads 2 --baseline-threads 2 --baseline-tile-size 512 --baseline-cut-after 0 --baseline-storage-mode memory --baseline-tuning-log runs\fraxet-development-tuning-512\tuning.json --repeats 5
```

The generated tuning and benchmark directories are retained locally and ignored by Git. The predeclared candidate list and commands above are versioned; rerunning with different package or machine versions may change times.
