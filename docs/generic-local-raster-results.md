# General local raster execution

Recorded 2026-11-23. This is development evidence, not the frozen independent holdout or tuned-baseline result.

The new `scripts/run_raster_local.py` uses the local graph analyzer to derive halo and alignment, validates one NCHW FP32 input and output with dynamic spatial dimensions, checks full-raster and cropped-tile output dimensions, then runs graph-cut workers under an optional Windows Job Object cap. `scripts/run_raster_auto.py` now considers this `local_graph` family, measures tile/cut/storage candidates under its required cap, and records the selected plan. The alternating benchmark wrapper accepts the family and a fixed baseline storage mode.

The 37×53 branched Conv/Concat engineering model passed whole-image ONNX Runtime comparison through fixed RAM, fixed disk-boundary, and automatic GeoTIFF paths at 128 MiB. All three outputs had zero maximum absolute error, retained EPSG:32643 and the source tag, and reported derived halo `[2, 2]`. The fixed disk path recorded positive boundary-spill bytes. A variant whose Conv padding shrinks the output was rejected before publishing a raster. An untuned one-repeat benchmark-wrapper smoke completed; it is not baseline evidence for the feasibility decision.

A separate deterministic 4096×4096 three-band development raster (seed 9501, input GeoTIFF SHA-256 `20b8b5fbcf43aae18325590bbedf985a72b9fcdcebd0fe5136f3831bf7bb9539`) exercised a larger job-capped path:

| Run | Plan | Total time | Job peak counter | Sampled private peak |
| --- | --- | ---: | ---: | ---: |
| Automatic | tile 256, cut 0, RAM | 29.47 s, including 8.97 s planning | 152.38 MB | 133.84 MB |
| Fixed alternate | tile 128, cut 3, RAM | 21.05 s | 101.46 MB | 101.46 MB |

Both runs configured a 134,217,728-byte hard job limit. The automatic search recorded 36 candidates and four failed probes; its raw peak counter exceeded the configured limit after those failures. The 50 ms sampled private peak was below the limit. See [evaluation amendment 2](evaluation-amendment-2.md) for the counter interpretation and its limits. A separate fixed tile 512 attempt failed on an 8,454,272-byte Rust allocation and published no output; its versioned failure archive remains in the ignored `runs/` directory.

The fixed alternate and automatic outputs were compared across all 256 nonoverlapping 256×256 windows. They had the same CRS and transform, zero differing FP32 values, and zero maximum absolute/eligible-relative error. Their TIFF file hashes differ because file layout/compression can vary. This cross-partition check is not a whole-image ONNX Runtime oracle at 4096²; the 37×53 test supplies that oracle for this model.

The current local analyzer covers explicit-pad Conv, ConvTranspose, MaxPool, pointwise operators, and channel Concat. It rejects spatial global operations and unmodeled attributes. The public FTW-Q model is still unsupported at Resize, and the current implementation does not prove arbitrary ONNX graph support. The new local path still needs an independent supported real model, repeated held-out evaluation, and a tuned baseline.

Reproduce the fixture and large development case:

```powershell
cargo build --locked
.\.venv\Scripts\python.exe tests\fixtures\smoke_generic_local_raster.py
.\.venv\Scripts\python.exe tests\performance\create_large_raster.py --output data\generic_local_4096_dev.tif --seed 9501 --height 4096 --width 4096 --bands 3
.\.venv\Scripts\python.exe scripts\run_raster_auto.py --model runs\local_graph_smoke\branched.onnx --input data\generic_local_4096_dev.tif --output runs\generic_local_4096_auto.tif --memory-limit-mib 128 --threads 2 --max-tile-size 512
.\.venv\Scripts\python.exe scripts\run_raster_local.py --model runs\local_graph_smoke\branched.onnx --input data\generic_local_4096_dev.tif --output runs\generic_local_4096_fixed.tif --memory-limit-mib 128 --threads 2 --tile-size 128 --cut-after 3
.\.venv\Scripts\python.exe tests\performance\compare_rasters.py runs\generic_local_4096_fixed.tif runs\generic_local_4096_auto.tif --report runs\generic_local_4096_comparison.json --window-size 256
```
