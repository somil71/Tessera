# Georeferenced Sentinel-2 development trial

Recorded 2026-11-24. This is a development correctness trial on genuine georeferenced pixels. It is **not** the independent holdout, a water-map accuracy evaluation, or a pass of the frozen performance protocol.

## Source and preparation

The source is [Earth Search Sentinel-2 L2A item `S2A_43PFL_20250328_1_L2A`](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_43PFL_20250328_1_L2A), acquired 2025-03-28 05:36:23 UTC with item-level cloud cover 2.480388%. `scripts/prepare_hydroflow_sentinel.py` pins the item JSON SHA-256 `29bea2196389eaf4503a56452a5df99ea9878be998907f0241b54c041379fe48` and records the exact COG URLs for B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, and B12. It aligns the 20 m and 60 m bands to the B02 10 m grid with bilinear resampling, rejects any masked pixel, and writes a 12-band raw GeoTIFF. The crop starts at B02 row 1650, column 3160 and is 1088×1088 pixels in EPSG:32643 with transform `(10, 0, 631600, 0, -10, 1083540)`.

The [HydroFlow model](https://huggingface.co/egyx/hydroflow-water-segmentation) is pinned at revision `47c399ee0cf612759d3ad664239705dddfb7b85b`; its FP32 ONNX SHA-256 is `22a283eecd573b2700817a018dbbbb17871e8dfaedf55e9f8f0a779924338b94`. The source script writes a per-band specification from the model's published min/max calibration, using `clip((raw-min)/(max-min+1e-6), 0, 1)`. `prepare_raster.py` applies it in bounded windows under the same 2 GiB Windows Job Object limit.

From the repository root, after following the Windows quickstart and downloading the pinned HydroFlow model and config:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_hydroflow_sentinel.py --output data\public\sentinel2-hydroflow-1088-raw.tif --spec-output data\public\sentinel2-hydroflow-1088-spec.json --size 1088 --memory-limit-mib 2048
.\.venv\Scripts\python.exe scripts\prepare_raster.py --input data\public\sentinel2-hydroflow-1088-raw.tif --output data\public\sentinel2-hydroflow-1088-normalized.tif --spec data\public\sentinel2-hydroflow-1088-spec.json --memory-limit-mib 2048
.\.venv\Scripts\python.exe scripts\run_raster_auto.py --model models\public\hydroflow-best-model.onnx --input data\public\sentinel2-hydroflow-1088-normalized.tif --output data\public\sentinel2-hydroflow-1088-scores.tif --memory-limit-mib 2048 --threads 2
.\.venv\Scripts\python.exe tests\performance\whole_image_oracle.py --model models\public\hydroflow-best-model.onnx --input data\public\sentinel2-hydroflow-1088-normalized.tif --reference-output data\public\sentinel2-hydroflow-1088-oracle.tif --candidate data\public\sentinel2-hydroflow-1088-scores.tif --report data\public\sentinel2-hydroflow-1088-oracle.json --threads 2 --max-pixels 2000000
```

The data files are ignored by Git. The raw raster SHA-256 was `9711fabaf08e0e13a2296b592fdb580a94b9e6aa866aa510ca3f7a62c67be39f`; the normalized raster SHA-256 was `0066013137464eef9d1a76a16fd9c1afa04dd909707459d0081bba36b203f2ca`. Source and preprocessing reports next to the rasters record full provenance and observed commitment.

## Observed result

The automatic runner chose a 512-pixel tile, graph cut 0, and disk boundary storage, producing nine local tiles. With the reduced 256 MiB GDAL cache, planning plus execution took 59.88 seconds; execution took 43.25 seconds. The independent unsplit ONNX Runtime oracle agreed at **every one of 1,183,744 FP32 scores** (`max_absolute_error=0`, `differing_values=0`). The score raster retained EPSG:32643 and the source transform.

The auto run's job peak counter was 2,199,146,496 bytes, above the requested 2,147,483,648-byte limit, although its sampled private commitment peaked at 2,142,068,736 bytes and the job limit remained enforced. Its first RAM probe failed an allocation before the selected disk plan ran. To distinguish that calibration attempt from the actual execution, the selected fixed plan was run separately with `scripts/run_raster_local.py --tile-size 512 --cut-after 0 --storage-mode disk --memory-limit-mib 2048 --threads 2`. It took 41.51 seconds, peaked at **1,689,866,240 bytes** of job commitment, and matched the oracle exactly with the same CRS and transform. The automatic peak counter **does not meet** a strict interpretation of the frozen cap gate; this trial does not establish a protocol pass. Calibration should be isolated or made to avoid over-budget probes before that claim can be made.

The model's supplied unreferenced example has B08 values 64–255, B11 values 10–80, and B12 values 0–97. In the real Sentinel scene, these three bands almost entirely clip to 1 after the published calibration (B08 99.98%, B11 100%, B12 100%). This indicates a source-domain mismatch or undocumented upstream band preparation. The scores verify scheduled execution only; they are **not** evidence of useful water segmentation. A map-quality evaluation requires a demonstrated compatible input contract and ground truth.
