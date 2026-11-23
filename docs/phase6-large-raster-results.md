# 8192×8192 development raster runs

Recorded 2026-11-23 on a Windows 11 laptop with an Intel Core i5-13450HX (10 cores, 16 logical processors). The runs used Python 3.11.6, NumPy 2.4.6, ONNX 1.23.2, ONNX Runtime 1.31.0, Rasterio 1.4.4, four inference threads, and a Windows Job Object committed-memory cap of **2,147,483,648 bytes**. `tests/performance/create_large_raster.py` generated two deterministic three-band FP32 GeoTIFFs in 128-row chunks. The U-Net input used seed 3101 and SHA-256 `c63951fc350bb245ab012d29f0eff0dd1db7609e1af29ff864b006705546538d`; global mean used seed 3102 and SHA-256 `a7f917c2b5d429247cea0449e0ae1147235507c044f3877ef737e35d72ca633c`. The input files were about 740 MB each.

| Model and schedule | Tile / cut | Total elapsed | Peak job committed | GDAL cache |
| --- | ---: | ---: | ---: | ---: |
| U-Net automatic | 512 / 0 | 113.0 s, including 13.9 s planning | 1,764.6 MB | 1,536 MiB |
| U-Net alternate | 256 / 4 | 130.6 s | 208.7 MB | 64 MiB |
| Global mean automatic | 512 / 0 | 102.7 s, including 2.0 s planning | 1,685.5 MB | 1,536 MiB |
| Global mean alternate | 256 / 0 | 113.9 s | 1,681.7 MB | 1,536 MiB |

Each run completed below the 2 GiB cap and produced a georeferenced two-band FP32 GeoTIFF. `tests/performance/compare_rasters.py` streamed 1,024 windows per model to compare automatic and alternate partitions. **Both comparisons found zero differing FP32 values**, zero maximum absolute error, and zero eligible-relative error, with matching CRS and affine transform. Each comparison records both output SHA-256 hashes; the audit script verifies that those hashes still match the present files. The global-mean FP32 channel means also matched across tile sizes. This is a strong large-image partition check, but a whole-image 8192×8192 reference was not computed; small whole-image references remain the numerical oracle.

The first U-Net automatic run used GDAL's default raster block cache and peaked at 1,764.4 MB in 111.7 s. An explicit 64 MiB cache reduced its peak to 295.3 MB but raised total time to 130.3 s. The final automatic policy reserves at least 256 MiB for Python, ONNX Runtime, and tiles, and allows up to 1,536 MiB for GDAL under a 2 GiB budget. That run peaked at 1,764.6 MB in 113.0 s. [GDAL documents](https://gdal.org/en/stable/user/configoptions.html) that `GDAL_CACHEMAX` controls its raster block cache and defaults to 5% of usable physical RAM. The present cache policy is a heuristic; it should become a measured planner choice when evaluating larger user models.

These are **single development runs**, not the frozen feasibility decision. The 256-pixel schedules were selected for a partition check, not tuned for four hours, and the runs were not alternated five times. No independent held-out model-machine pair or real analyst trial has been completed. The generated inputs, outputs, raw reports, and `runs/large_development_summary.json` live in ignored local directories; the scripts and exact reproduction commands are versioned here.

```powershell
.\.venv\Scripts\python.exe tests\performance\create_large_raster.py --output data\large_unet_3101.tif --seed 3101
.\.venv\Scripts\python.exe tests\performance\create_large_raster.py --output data\large_global_3102.tif --seed 3102
.\.venv\Scripts\python.exe scripts\run_raster_auto.py --model models\evaluation\unet_dev.onnx --input data\large_unet_3101.tif --output runs\large_unet_auto_budgetcache.tif --memory-limit-mib 2048 --threads 4
.\.venv\Scripts\python.exe scripts\run_raster_unet.py --model models\evaluation\unet_dev.onnx --input data\large_unet_3101.tif --output runs\large_unet_tile256.tif --tile-size 256 --cut-after 4 --memory-limit-mib 2048 --threads 4
.\.venv\Scripts\python.exe scripts\run_raster_auto.py --model models\evaluation\global_mean_dev.onnx --input data\large_global_3102.tif --output runs\large_global_auto.tif --memory-limit-mib 2048 --threads 4
.\.venv\Scripts\python.exe scripts\run_raster_global_mean.py --model models\evaluation\global_mean_dev.onnx --input data\large_global_3102.tif --output runs\large_global_tile256.tif --tile-size 256 --memory-limit-mib 2048 --threads 4
.\.venv\Scripts\python.exe tests\performance\compare_rasters.py runs\large_unet_auto_budgetcache.tif runs\large_unet_tile256.tif --report runs\large_unet_budgetcache_partition_diff.json
.\.venv\Scripts\python.exe tests\performance\compare_rasters.py runs\large_global_auto.tif runs\large_global_tile256.tif --report runs\large_global_partition_diff.json
.\.venv\Scripts\python.exe tests\performance\summarize_large_runs.py
```
