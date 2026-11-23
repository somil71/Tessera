# Two-pass global mean raster inference

Recorded 2026-11-23. `scripts/run_raster_global_mean.py` validates an ONNX graph whose first node is `ReduceMean` of its NCHW FP32 input over spatial axes 2 and 3 with singleton dimensions retained. It streams bounded GeoTIFF windows through the declared band order, scale, and offset, summing each channel in FP64 and storing the final mean as FP32. It then replaces that reduction node with an ONNX initializer and runs the remaining pointwise graph through the existing Rust pipe worker. The same Windows Job Object cap remains active across both passes and every Rust child process. Models outside this checked pattern are rejected.

The protocol's 33×47 and 96×80 development sizes were compared with whole-image ONNX Runtime output. Tile sizes 8, 13, and 24 passed the frozen `1e-4` absolute and `1e-3` eligible-relative tolerances. The largest observed absolute error was **1.49×10⁻⁸**; the largest eligible-relative error was **1.92×10⁻⁶**. The 13-pixel case also exercised band permutation and scale/offset. Output retained CRS, affine transform, FP32 score bands, and the source metadata tag. A no-data pixel was rejected during the reduction pass without publishing output. All tests used a 128 MiB job-wide committed-memory cap.

This is an exact global dependency schedule for the supported fixture, within the numerical contract. It still requires a user-selected tile size and fixed cut. Its FP64 accumulation can differ from ONNX Runtime's internal reduction order by a small FP32 amount. The result does not establish the large-image performance target, generic global-op support, or the held-out feasibility decision.

Reproduce with:

```powershell
cargo build
.\.venv\Scripts\python.exe tests\fixtures\create_eval_models.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_global_mean_raster.py
```
