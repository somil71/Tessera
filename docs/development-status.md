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
| Phase 4: fixed raster prototype | Python Rasterio schedulers for pixelwise, exact local U-Net, and two-pass global-mean models; persistent Rust pipe workers; georeferenced FP32 output; optional Windows Job Object cap and JSON reports | Connects the compiler core to a real raster format, handles local and one global dependency pattern, and tests job-wide committed-memory enforcement | Pixelwise 37×53 and U-Net 38×54 outputs matched exactly; global mean 33×47 and 96×80 passed frozen numerical tolerances, all under 128 MiB. A 48 MiB job rejected a 64 MiB commit. See [global-mean results](phase4-global-mean-results.md) and [job memory results](phase4-job-memory-results.md) |

The source of each piece is [graph.rs](../src/graph.rs), [split.rs](../src/split.rs), [execute.rs](../src/execute.rs), [memory.rs](../src/memory.rs), and [run_raster_fixed.py](../scripts/run_raster_fixed.py). Test fixtures and benchmark scripts live in `tests/`.

## Architecture and execution path

1. The Rust `graph` command parses ONNX without loading ONNX Runtime. It checks topological wiring, infers shapes for the frozen operator patterns, and prints the tensors crossing each candidate cut. Unknown shape patterns remain unknown rather than receiving a false byte estimate.
2. The Rust `split` command turns a chosen cut into `head.onnx` and `tail.onnx`. Each half receives only the weights it needs. Boundary tensors receive concrete ONNX type and shape declarations for the supplied image size.
3. Rust `run` executes one model on raw little-endian NCHW FP32 data. Rust `run-split` executes the head, copies required boundary tensors, releases the head session, then loads and executes the tail. It writes raw FP32 scores. Keeping the original input alive through the head run fixed an aliasing error discovered during testing.
4. The fixed pixelwise raster prototype validates that each operation is independent at every pixel. A second, topology-checked U-Net path aligns output tiles to the pool stride, adds a four-pixel input halo, clips at true image edges, and crops the returned scores. Both send raw tile bytes to persistent Rust `serve-split-pipe` workers and write georeferenced FP32 windows. Rasterio copies the source CRS and transform into the output profile. [Rasterio's window API](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html) supplies the windowed I/O.
5. The global-mean path first streams every raster window to compute the full-image channel mean, replaces the spatial reduction with that FP32 constant, then executes the remaining pointwise graph under the same job-wide cap. Its output uses the same georeferenced FP32 writer.

Python currently owns GeoTIFF I/O because Rasterio has a Windows wheel in the pinned Python 3.11 environment, while native GDAL is not installed in the workspace. The first version used a raw-file bridge and one Rust process per tile. The default now keeps both ONNX Runtime sessions loaded and sends tile data over pipes. On the 12-window smoke workload, median elapsed time fell from 1.647 s with fresh processes to 0.240 s with pipes; Rust worker peak committed memory rose from 16.95 to 18.22 MiB. See [worker measurements](phase4-worker-results.md). The newer optional memory budget joins the Python scheduler and all Rust model processes to a Windows Job Object and records its peak committed memory; [the cap test](phase4-job-memory-results.md) also verifies an over-budget commit fails.

## What the measurements say

On the frozen U-Net at 1×3×512×512, three fresh-process runs per configuration gave median peak committed memory of **55.07 MiB** for the whole model and **63.00–69.05 MiB** for three manual cuts. Median total time was **120 ms** whole and **125–133 ms** split. All compared outputs matched exactly in that benchmark. The split consumed more committed memory, largely because materializing boundary tensors while head outputs still exist raises the process peak. Peak working set differs from committed memory; cut 4 had a lower working-set peak but a higher committed-memory peak. The planned limit applies to committed memory, so cut 4 is not a memory win. See [Phase 3 results](phase3-results.md) for the full table and measurement definition.

The raster prototype's test output matched a whole-image pointwise ONNX Runtime reference exactly for both tested tile sizes, including uneven right and bottom edges. The EPSG:32643 CRS, affine transform, two FP32 score bands, and a source metadata tag were preserved. The scheduler rejected a no-data pixel before publishing an output and rejected the spatial U-Net before starting raster execution.

The separate U-Net raster path matched whole-image logits exactly at tile sizes 6, 8, 16, and 24 on a 38×54 raster, including a reordered and scaled input. It preserved the georeferencing and rejected an odd-width input without publishing output. It groups shape-specific tiles so that only one Rust worker remains live at a time. See [the U-Net report](phase4-unet-results.md).

The two-pass global-mean path passed the protocol's 33×47 and 96×80 correctness sizes at tile sizes 8, 13, and 24. Its largest absolute error against whole-image ONNX Runtime logits was 1.49×10⁻⁸ and largest eligible-relative error 1.92×10⁻⁶. It rejected no-data during the reduction pass before publishing output. See [the global-mean report](phase4-global-mean-results.md).

## Remaining gaps

- There is no automatic cut, keep/spill/recompute planner. Manual cuts are exploratory.
- The two fixed raster runners can enforce a Windows Job Object committed-memory cap when requested, but there is no automatic choice of tile size or graph cut. The final v1 CLI must require the cap and choose a feasible plan. Job-wide peak working set is not yet measured.
- Raster execution has three fixed paths: pixelwise models, the validated development U-Net topology, and a first-node spatial global mean. There is no generic receptive-field or global-operation analyzer yet. Global attention needs a blockwise exact schedule. Unsupported models are rejected rather than silently miscomputed.
- The default raster path has a persistent Rust worker and no raw tile bridge, but Python still owns GeoTIFF I/O. It is a correctness prototype, not a large-raster performance result.
- The output report has no numerical error estimate for an arbitrary user raster because a whole-image reference may not fit. The test harness calculates error on small reference cases.
- There is no independent user trial, real remote-sensing model/raster trial, tuned manual baseline, held-out 1 GiB run, or second physical machine. The originally designated attention holdout was exposed during development; [evaluation amendment 1](evaluation-amendment-1.md) requires a new independent holdout. The frozen feasibility decision has not been made.

## Next execution plan

1. Generalize the local-model halo and alignment analysis beyond the frozen U-Net topology, and integrate it with automatic tile and cut selection.
2. Add the blockwise attention path with a cost estimate and an infeasibility report when appropriate.
3. Implement a planner that chooses tile size, graph cut, and keep/spill/recompute using measured memory and disk costs. Make the existing Job Object cap mandatory in the final CLI and add job-wide working-set measurement.
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
.\.venv\Scripts\python.exe tests\fixtures\smoke_global_mean_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_job_memory.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_worker.py
.\.venv\Scripts\python.exe tests\benchmark_phase3.py
.\.venv\Scripts\python.exe tests\benchmark_worker.py
```
