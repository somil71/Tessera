# Blockwise exact attention on a GeoTIFF

Recorded 2026-11-23. `scripts/run_raster_attention.py` validates the exposed development fixture's Shape → Reshape → Transpose → MatMul → Softmax → MatMul → Transpose → Reshape pattern. It stores the preprocessed input and FP32 output in temporary disk-backed arrays, then computes attention in query and key blocks using a stable running softmax maximum, denominator, and weighted-value sum. No full pixel-by-pixel score matrix is materialized. It writes georeferenced FP32 score bands and removes staging files after the run. The Windows Job Object cap includes model validation, raster I/O, staging, and attention computation. This is support for this checked self-attention pattern, not arbitrary transformer models.

`tests/fixtures/smoke_attention_raster.py` compared 8×10, 64×64, and 96×96 rasters with whole-image ONNX Runtime FP32 logits under a 128 MiB job cap. The largest maximum absolute error was **1.74×10⁻⁵** and the largest eligible-relative error was **1.01×10⁻⁴**, below the frozen `1e-4` and `1e-3` limits. The 8×10 case also checked band order and scale/offset. Output retained CRS, transform, FP32 channel count, and a source tag; a no-data input was rejected without output. The 96×96 run reported a peak job commitment of 89,608,192 bytes, peak process working set of 85,557,248 bytes, and peak temporary storage of 434,460 bytes. The auto CLI also selected this path and matched an 8×10 whole-image reference.

Before reading all pixels, the path calibrates a 1024-pixel blockwise pilot, caps extrapolated throughput at 20 million score cells per second, and applies a factor-of-two time allowance. It rejects a predicted run over 30 minutes. An 8192×8192 metadata-only GeoTIFF produced an infeasible report before input staging or output publication. The estimate is a conservative scheduling heuristic, not a runtime guarantee. Feasible attention images may still take longer than estimated, and disk traffic grows with repeated key-block scans.

The model file is historically named `attention_holdout.onnx`, but it was exposed during development and is **not** a valid independent holdout. [Evaluation amendment 1](evaluation-amendment-1.md) still requires a new frozen model and machine pair. These tests establish implementation correctness for the exposed fixture only. They do not establish the held-out performance result or general transformer support.

Reproduce with:

```powershell
.\.venv\Scripts\python.exe tests\fixtures\create_eval_models.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_attention_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_auto_raster.py
```
