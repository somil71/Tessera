# Windows CPU quickstart

This workflow is for a 64-bit Windows machine with Python 3.11, a Rust toolchain, and enough free disk space for the input, FP32 output, and temporary files. It describes the current source checkout; a clean-machine installation has not yet been independently verified.

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
cargo build --locked
.\.venv\Scripts\python.exe scripts\doctor.py
```

The doctor checks Python x64 and pinned package versions, the Rust executable, and the ONNX Runtime DLL used by Rust. The Python runners set the Rust runtime's DLL path automatically. If `py -3.11` is unavailable, install Python 3.11 x64 and rerun the commands. `cargo build --locked` requires Cargo on `PATH`.

For a supported ONNX model and georeferenced GeoTIFF, run preflight and then automatic inference:

```powershell
.\.venv\Scripts\python.exe scripts\doctor.py `
  --model models\your_model.onnx --input data\your_image.tif `
  --output runs\scores.tif --memory-limit-mib 2048 `
  --report runs\preflight.json

.\.venv\Scripts\python.exe scripts\run_raster_auto.py `
  --model models\your_model.onnx --input data\your_image.tif `
  --output runs\scores.tif --memory-limit-mib 2048 `
  --threads 4
```

Add `--band-order 3,2,1`, `--scale`, or `--offset` only when those values match the model's declared preprocessing. The output is a georeferenced FP32 score GeoTIFF, with `runs\scores.report.json` beside it. Inspect `strategy`, `selected_plan`, `memory_limit_enforced`, `job_peak_conservative_upper_bound_bytes`, `memory_limit_bytes`, `total_elapsed_seconds`, and the [resource-accounting fields](runtime-resource-accounting.md). The doctor verifies basic compatibility and available disk space; it does **not** prove a feasible schedule, numerical correctness, or a model's intended whole-image behavior.

If the run cannot plan or execute within the cap, inspect `runs\scores.plan.failure.json` and any `runs\scores.failure.json`. The runner should not publish a partial score raster as a completed result. Supported model patterns and their limits are in the [v1 contract](v1-contract.md) and [development status](development-status.md). To check the installation with the repository's engineering fixtures:

```powershell
.\.venv\Scripts\python.exe tests\fixtures\create_eval_models.py
.\.venv\Scripts\python.exe tests\fixtures\create_pointwise_model.py
cargo test
.\.venv\Scripts\python.exe tests\fixtures\smoke_auto_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_job_memory.py
```

For a new model, establish its intended inference dimensions and preprocessing before treating a whole-image ONNX Runtime run as a numerical reference. A model trained for fixed patches may have different behavior when given the whole raster. The [evaluation protocol](evaluation-protocol.md) defines how the project compares FP32 scores, enforces memory, and evaluates automatic planning against a tuned baseline.
