# Development status

Updated 2026-11-23. This is the state of the repository, not a claim that the v1 product or feasibility experiment has passed.

## Purpose and acceptance target

Raster Compiler aims to run an ONNX vision model on a GeoTIFF too large for normal whole-image inference on a Windows CPU machine. The intended product accepts a model, raster, preprocessing declaration, and committed-memory budget; it chooses an execution plan, writes georeferenced FP32 scores, and records correctness, time, memory, and disk use. The [v1 contract](v1-contract.md) defines supported inputs and the [evaluation protocol](evaluation-protocol.md) freezes the engineering fixtures, references, memory caps, and pass criteria. The main research question is whether automatic scheduling can stay within the memory cap without an unacceptable runtime penalty against a tuned manual baseline.

## Completed work

| Stage | What was built | Why it exists | Evidence |
| --- | --- | --- | --- |
| Phase 1: foundation | Rust CLI, ONNX Runtime loading, generated ONNX fixtures, frozen support and evaluation documents | Defines the target user, exact correctness reference, model families, and baseline rules before planner work | Three engineering model fixtures pass ONNX Runtime smoke execution; the originally designated attention holdout was later exposed in development |
| Phase 2: graph analysis | ONNX wiring validation, producers and consumers, supported shape inference, tensor byte estimates, live tensors and byte totals at each cut | A planner needs explicit dependencies and boundary sizes before it can choose where to split or store tensors | Five Rust unit tests and graph inspection of U-Net, global mean, and attention fixtures |
| Phase 3: subgraph execution | Head/tail model extraction, sequential Rust ONNX Runtime execution, boundary materialization, whole-model comparison, Windows process memory readings at execution stages | Tests whether a graph cut is both numerically correct and actually cheaper in process memory | All 16 internal cuts across three fixtures matched whole-model smoke outputs; [Phase 3 report](phase3-results.md) measures a larger U-Net |
| Phase 4: fixed raster prototype | Python Rasterio window schedulers feeding persistent Rust split workers through pipes; pixelwise windows and an exact four-pixel-halo schedule for the frozen local U-Net; georeferenced FP32 output and JSON reports | Connects the compiler core to a real raster format and establishes exact local spatial tiling | Pixelwise 37×53 and U-Net 38×54 rasters matched whole-image outputs exactly across tested tile sizes; CRS, transform, preprocessing, and rejection cases passed. See [U-Net results](phase4-unet-results.md) |

The source of each piece is [graph.rs](../src/graph.rs), [split.rs](../src/split.rs), [execute.rs](../src/execute.rs), [memory.rs](../src/memory.rs), and [run_raster_fixed.py](../scripts/run_raster_fixed.py). Test fixtures and benchmark scripts live in `tests/`.

## Architecture and execution path

1. The Rust `graph` command parses ONNX without loading ONNX Runtime. It checks topological wiring, infers shapes for the frozen operator patterns, and prints the tensors crossing each candidate cut. Unknown shape patterns remain unknown rather than receiving a false byte estimate.
2. The Rust `split` command turns a chosen cut into `head.onnx` and `tail.onnx`. Each half receives only the weights it needs. Boundary tensors receive concrete ONNX type and shape declarations for the supplied image size.
3. Rust `run` executes one model on raw little-endian NCHW FP32 data. Rust `run-split` executes the head, copies required boundary tensors, releases the head session, then loads and executes the tail. It writes raw FP32 scores. Keeping the original input alive through the head run fixed an aliasing error discovered during testing.
4. The fixed pixelwise raster prototype validates that each operation is independent at every pixel. A second, topology-checked U-Net path aligns output tiles to the pool stride, adds a four-pixel input halo, clips at true image edges, and crops the returned scores. Both send raw tile bytes to persistent Rust `serve-split-pipe` workers and write georeferenced FP32 windows. Rasterio copies the source CRS and transform into the output profile. [Rasterio's window API](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html) supplies the windowed I/O.

Python currently owns GeoTIFF I/O because Rasterio has a Windows wheel in the pinned Python 3.11 environment, while native GDAL is not installed in the workspace. The first version used a raw-file bridge and one Rust process per tile. The default now keeps both ONNX Runtime sessions loaded and sends tile data over pipes. On the 12-window smoke workload, median elapsed time fell from 1.647 s with fresh processes to 0.240 s with pipes; Rust worker peak committed memory rose from 16.95 to 18.22 MiB. See [worker measurements](phase4-worker-results.md). The Python scheduler's own peak and Rust worker peak are reported separately; their sum is only an upper bound until a Windows Job Object measures them together.

## What the measurements say

On the frozen U-Net at 1×3×512×512, three fresh-process runs per configuration gave median peak committed memory of **55.07 MiB** for the whole model and **63.00–69.05 MiB** for three manual cuts. Median total time was **120 ms** whole and **125–133 ms** split. All compared outputs matched exactly in that benchmark. The split consumed more committed memory, largely because materializing boundary tensors while head outputs still exist raises the process peak. Peak working set differs from committed memory; cut 4 had a lower working-set peak but a higher committed-memory peak. The planned limit applies to committed memory, so cut 4 is not a memory win. See [Phase 3 results](phase3-results.md) for the full table and measurement definition.

The raster prototype's test output matched a whole-image pointwise ONNX Runtime reference exactly for both tested tile sizes, including uneven right and bottom edges. The EPSG:32643 CRS, affine transform, two FP32 score bands, and a source metadata tag were preserved. The scheduler rejected a no-data pixel before publishing an output and rejected the spatial U-Net before starting raster execution.

The separate U-Net raster path matched whole-image logits exactly at tile sizes 6, 8, 16, and 24 on a 38×54 raster, including a reordered and scaled input. It preserved the georeferencing and rejected an odd-width input without publishing output. It groups shape-specific tiles so that only one Rust worker remains live at a time. See [the U-Net report](phase4-unet-results.md).

## Remaining gaps

- There is no automatic cut, keep/spill/recompute planner. Manual cuts are exploratory.
- There is no Windows Job Object limit yet. Rust reports each child process's peak committed bytes and working set; the raster report does not include the Python scheduler in a job-wide peak.
- Raster execution has two fixed paths: pixelwise models and the validated development U-Net topology. There is no generic receptive-field analyzer yet. Global mean needs a whole-image reduction pass. Global attention needs a blockwise exact schedule. Unsupported models are rejected rather than silently miscomputed.
- The default raster path has a persistent Rust worker and no raw tile bridge, but Python still owns GeoTIFF I/O. It is a correctness prototype, not a large-raster performance result.
- The output report has no numerical error estimate for an arbitrary user raster because a whole-image reference may not fit. The test harness calculates error on small reference cases.
- There is no independent user trial, real remote-sensing model/raster trial, tuned manual baseline, held-out 1 GiB run, or second physical machine. The originally designated attention holdout was exposed during development; [evaluation amendment 1](evaluation-amendment-1.md) requires a new independent holdout. The frozen feasibility decision has not been made.

## Next execution plan

1. Generalize the local-model halo and alignment analysis beyond the frozen U-Net topology, and integrate it with automatic tile and cut selection.
2. Add the global-mean model with an explicit first reduction pass, then the blockwise attention path with a cost estimate and an infeasibility report when appropriate.
3. Implement a planner that chooses keep, spill, or recompute using measured memory and disk costs. Enforce a job-wide committed-memory cap with a Windows Job Object, including the Python raster scheduler and Rust worker.
4. Freeze a new independent holdout under a versioned protocol, then run the development and held-out evaluation, retain failures and raw reports, compare against the four-hour tuned baseline, and test a real user model and raster.

## Reproduce the current checks

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tests\fixtures\create_eval_models.py
.\.venv\Scripts\python.exe tests\fixtures\create_pointwise_model.py
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
.\.venv\Scripts\python.exe tests\fixtures\smoke_split_models.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_rust_execution.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_unet_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_worker.py
.\.venv\Scripts\python.exe tests\benchmark_phase3.py
.\.venv\Scripts\python.exe tests\benchmark_worker.py
```
