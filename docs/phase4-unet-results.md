# Exact local U-Net raster schedule

Recorded 2026-11-23. `scripts/run_raster_unet.py` adds a fixed spatial schedule for the frozen `unet_dev.onnx` architecture: 3×3 encoder convolution, ReLU, 2×2 stride-two max pool, 3×3 bottleneck convolution, ReLU, 2×2 stride-two transpose convolution, skip addition, and 1×1 head convolution. It validates this topology and attributes before inference. Other local models are not yet automatically analyzed.

Each output tile starts on an even coordinate. Its input window includes a four-pixel halo on all sides, clipped to the actual raster bounds. Even alignment preserves the max-pool phase; clipping makes convolution padding at the image edge match the whole-image execution. The worker receives the full halo window, and the scheduler writes only the core scores. Edge windows have different shapes, so the scheduler groups tiles by shape and keeps only one Rust split worker alive at a time. This avoids the unaccounted memory cost of several concurrent ONNX Runtime sessions.

`tests/fixtures/smoke_unet_raster.py` compared a 38×54, three-band GeoTIFF against whole-image ONNX Runtime logits for output tile sizes 6, 8, 16, and 24. Maximum absolute FP32 logit error was **0** in all four cases. The 16-pixel case also checked band permutation and scale/offset. CRS, transform, score count, FP32 type, and source tag survived output assembly. An odd-width raster was rejected without publishing an output because the fixture's pool and skip path requires even whole-image dimensions. The prior pixelwise raster tests still pass.

The reported child peak is now the maximum of workers run **one at a time**; the Python scheduler peak is separate. Their sum is a conservative bound, not a Windows Job Object measurement or an enforced 2 GiB cap. The reference comparison is on a small engineering raster. There is no proof yet for a real user model, a large raster, a held-out model, or planner-chosen tile size and graph cut.

Reproduce with:

```powershell
cargo build
.\.venv\Scripts\python.exe tests\fixtures\create_eval_models.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_unet_raster.py
```
