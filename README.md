# Raster Compiler

Rust tooling for planning inference on images that exceed an available memory budget. The first target is CPU inference for ONNX vision models on large GeoTIFF rasters.

The [v1 support contract](docs/v1-contract.md) and [frozen evaluation protocol](docs/evaluation-protocol.md) define the product and its acceptance test. The [evaluation amendment](docs/evaluation-amendment-1.md) records that the original attention holdout was exposed in development and must be replaced for the final evaluation. Phases 1–3 are complete as development milestones. Phase 4 has fixed GeoTIFF schedules for pixelwise models, the [frozen local U-Net](docs/phase4-unet-results.md), and the [global-mean fixture](docs/phase4-global-mean-results.md), using a [persistent pipe worker](docs/phase4-worker-results.md). The [development status](docs/development-status.md) records what is complete, why it was built, what was measured, and what remains.

## Current state

The CLI loads an ONNX model, reports its input and output tensors, validates graph wiring, lists tensor producers and consumers, estimates tensor sizes for an image shape, and reports live tensors and their byte totals at every possible cut. It rejects inconsistent resolved output shapes and leaves unsupported shape patterns as `unknown`. Cut totals cover boundary tensors only; they are not peak process memory. The CLI can write two ONNX subgraphs at a chosen cut and execute them sequentially on a raw FP32 input. Python raster runners handle three validated model patterns and can enforce a Windows Job Object committed-memory cap. The project does not yet choose tile sizes, cuts, or storage decisions automatically.

```powershell
cargo run -- inspect path\to\model.onnx
cargo run -- graph path\to\model.onnx
cargo run -- graph path\to\model.onnx --height 1024 --width 1024
cargo run -- split path\to\model.onnx --cut-after 4 --height 1024 --width 1024 --out-dir runs\split
cargo run -- run path\to\model.onnx --input-f32 data\input.f32 --output-f32 runs\full.f32 --channels 3 --height 1024 --width 1024
cargo run -- run-split runs\split\head.onnx runs\split\tail.onnx --input-f32 data\input.f32 --output-f32 runs\split.f32 --channels 3 --height 1024 --width 1024
```

The `ort` dependency loads ONNX Runtime dynamically. On this machine, the Rust target is `x86_64-pc-windows-gnu`, which has no matching prebuilt ONNX Runtime package in `ort`. To run `inspect`, provide a compatible `onnxruntime.dll` and set `ORT_DYLIB_PATH` to its full path. `graph` reads the ONNX file without loading ONNX Runtime. Keep local models in `models/`, rasters in `data/`, and experiment outputs in `runs/`. These directories are excluded from Git.

For a local smoke test on Windows, create a virtual environment and a tiny model:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install onnx==1.23.2 onnxruntime==1.31.0
.\.venv\Scripts\python.exe tests\fixtures\create_tiny_model.py
$env:ORT_DYLIB_PATH = (Resolve-Path .\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll).Path
cargo run -- inspect models\tiny_skip.onnx
cargo run -- graph models\tiny_skip.onnx
cargo run -- graph models\tiny_skip.onnx --height 8 --width 8
```

The cut report for this fixture includes `activated` and the original `image` input, because the final `Add` node uses both. Tensor sizes are estimates based on supported operator patterns and exclude ONNX Runtime's internal workspace allocations. Unsupported shapes display as `unknown`. Generate and smoke-test the phase 1 evaluation models with:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy==2.4.6
.\.venv\Scripts\python.exe tests\fixtures\create_eval_models.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_eval_models.py
cargo build
.\.venv\Scripts\python.exe tests\fixtures\smoke_split_models.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_rust_execution.py
.\.venv\Scripts\python.exe tests\benchmark_phase3.py
.\.venv\Scripts\python.exe tests\fixtures\create_pointwise_model.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_unet_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_global_mean_raster.py
```

Run the fixed raster prototype with a pixelwise ONNX model and a georeferenced GeoTIFF:

```powershell
.\.venv\Scripts\python.exe scripts\run_raster_fixed.py --model models\pointwise_dev.onnx --input data\input.tif --output runs\scores.tif --cut-after 0 --tile-size 256
```

The wrapper supports `--band-order 3,2,1`, `--scale`, and `--offset`. It rejects declared no-data and non-finite pixels. It writes a JSON report beside the output. The wrapper uses Rasterio 1.4.4; install all pinned Python dependencies with `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`.
Use `--worker-mode persistent` to compare the file-backed worker or `--worker-mode fresh` for a fresh Rust process per tile. The default `pipe` mode writes no raw tile bridge files. Pass `--memory-limit-mib 2048` to enforce a [Windows job-wide committed-memory cap](docs/phase4-job-memory-results.md) across the Python scheduler and Rust processes. This argument remains optional for development comparisons; the future planner CLI will require it.

For the frozen U-Net topology and an even-size raster, run the fixed exact spatial schedule:

```powershell
.\.venv\Scripts\python.exe scripts\run_raster_unet.py --model models\evaluation\unet_dev.onnx --input data\input.tif --output runs\unet_scores.tif --tile-size 256
```

This path validates the fixture topology, aligns tiles to the pool stride, reads a four-pixel halo, and clips halo windows at real image edges. It does not choose tile size or cut automatically.

For the supported global-mean pattern, stream the reduction pass and execute the transformed pointwise graph under one job-wide cap:

```powershell
.\.venv\Scripts\python.exe scripts\run_raster_global_mean.py --model models\evaluation\global_mean_dev.onnx --input data\input.tif --output runs\global_scores.tif --tile-size 256 --memory-limit-mib 2048
```

## Build order

1. Fix reference models, numerical tolerances, and baseline rules for the feasibility experiment.
2. Partition one U-Net into subgraphs and measure intermediate tensors and backend allocations.
3. Add windowed raster input, output assembly, and a fixed execution schedule.
4. Plan keep, spill, and recompute choices under an enforced Windows Job Object memory limit.
5. Add one model with a cross-image operation and evaluate unseen workloads.

Each supported model must have an explicit inference contract. A run report will record output error, peak committed memory, working set, runtime, and disk traffic.
