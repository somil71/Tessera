# Streaming raster preprocessing

`scripts/prepare_raster.py` turns a georeferenced GeoTIFF into an FP32 GeoTIFF using a declared operation for each output band. It reads and writes bounded windows, runs under a required Windows Job Object committed-memory cap, rejects declared nodata, masks, and non-finite pixels, and copies the source CRS, transform, and dataset tags. It writes a report with source/spec/output hashes, the normalized specification, elapsed time, and job memory counters. The output and report must be new paths; an incomplete temporary TIFF is removed on failure.

A specification has one object per output band. `source` is a 1-based input band index. For each pixel, the command computes `clip(float32(source) * scale + offset, clip_min, clip_max)`. Scale defaults to 1 and offset to 0. Either clipping bound may be omitted. Repeated or reordered source bands are allowed.

```json
{
  "bands": [
    {"source": 3, "scale": 0.00392156862745098, "clip_min": 0, "clip_max": 1, "name": "red"},
    {"source": 2, "scale": 0.00392156862745098, "clip_min": 0, "clip_max": 1, "name": "green"},
    {"source": 1, "scale": 0.00392156862745098, "clip_min": 0, "clip_max": 1, "name": "blue"}
  ]
}
```

```powershell
.\.venv\Scripts\python.exe scripts\prepare_raster.py `
  --input data\raw.tif --output data\normalized.tif `
  --spec data\preprocessing.json --memory-limit-mib 2048

.\.venv\Scripts\python.exe scripts\run_raster_auto.py `
  --model models\model.onnx --input data\normalized.tif `
  --output runs\scores.tif --memory-limit-mib 2048
```

The preparation and inference commands each enforce the declared limit in their own process. The prepared raster occupies additional disk space; retain its SHA-256 from the preprocessing report alongside the inference report so the exact input can be identified later. The automatic runner still validates the model topology. This command does not make unsupported ONNX operators executable.

The [HydroFlow compatibility probe](real-model-compatibility.md#hydroflow-development-probe) motivated per-band arithmetic and clipping: its author requires different calibration for each Sentinel-2 band. For a HydroFlow band with upstream `min` and `max`, set `scale = 1 / (max - min + 1e-6)`, `offset = -min / (max - min + 1e-6)`, `clip_min = 0`, and `clip_max = 1`. Record the resulting JSON file and its hash. The supplied HydroFlow sample TIFF has no CRS, so it is not a valid input to this georeferenced workflow.

## Development measurement

On 2026-11-24, a 4096×4096 three-band georeferenced development raster was processed in 256 windows under a **128 MiB hard cap**. The report recorded 69,488,640 bytes peak job commitment and 9.98 seconds including input and output hashing. An independent bounded-window comparison found zero FP32 differences from the declared scale-and-clip expression across all 256 windows, with matching CRS, transform, and dataset tags.

The prepared raster then passed preflight for the development general-local graph. Automatic inference finished in 32.23 seconds, selecting 128-pixel tiles and a RAM boundary; sampled peak private commitment was 134,148,096 bytes under the 134,217,728-byte hard limit. Its two-band FP32 output retained the input CRS, transform, and tags. A [crop-based oracle](evaluation-runner.md#large-local-crop-spot-check) compared 124×124 interior score regions at the top-left, center, and bottom-right against unsplit ONNX Runtime; all had zero FP32 differences. The raw spec, rasters, outputs, and reports are ignored under `runs/` and `data/`. This is development evidence on a synthetic model, not the held-out performance or external real-model result.
