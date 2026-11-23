# Development status

Updated 2026-11-23. This is the state of the repository, not a claim that the v1 product or feasibility experiment has passed.

## Purpose and acceptance target

Raster Compiler aims to run an ONNX vision model on a GeoTIFF too large for normal whole-image inference on a Windows CPU machine. The intended product accepts a model, raster, preprocessing declaration, and committed-memory budget; it chooses an execution plan, writes georeferenced FP32 scores, and records correctness, time, memory, and disk use. The [v1 contract](v1-contract.md) defines supported inputs and the [evaluation protocol](evaluation-protocol.md) freezes the engineering fixtures, references, memory caps, and pass criteria. The main research question is whether automatic scheduling can stay within the memory cap without an unacceptable runtime penalty against a tuned manual baseline.

## Completed work

| Stage | What was built | Why it exists | Evidence |
| --- | --- | --- | --- |
| Phase 1: foundation | Rust CLI, ONNX Runtime loading, generated ONNX fixtures, frozen support and evaluation documents | Defines the target user, exact correctness reference, model families, and baseline rules before planner work | Three development/held-out model fixtures pass ONNX Runtime smoke execution |
| Phase 2: graph analysis | ONNX wiring validation, producers and consumers, supported shape inference, tensor byte estimates, live tensors and byte totals at each cut | A planner needs explicit dependencies and boundary sizes before it can choose where to split or store tensors | Five Rust unit tests and graph inspection of U-Net, global mean, and attention fixtures |
| Phase 3: subgraph execution | Head/tail model extraction, sequential Rust ONNX Runtime execution, boundary materialization, whole-model comparison, Windows process memory readings at execution stages | Tests whether a graph cut is both numerically correct and actually cheaper in process memory | All 16 internal cuts across three fixtures matched whole-model smoke outputs; [Phase 3 report](phase3-results.md) measures a larger U-Net |
| Phase 4: fixed raster prototype | Python Rasterio window scheduler feeding the Rust split runner, edge padding and cropping for pixelwise models, georeferenced FP32 GeoTIFF output, band order, scale/offset, no-data rejection, a JSON run report | Connects the compiler core to a real raster format while limiting the first tiled path to models for which independent windows are exact | 37×53 raster passed 16-pixel and 11-pixel window assembly, preprocessing, CRS/transform preservation, and whole-image output comparison; invalid no-data and spatial U-Net cases were rejected |

The source of each piece is [graph.rs](../src/graph.rs), [split.rs](../src/split.rs), [execute.rs](../src/execute.rs), [memory.rs](../src/memory.rs), and [run_raster_fixed.py](../scripts/run_raster_fixed.py). Test fixtures and benchmark scripts live in `tests/`.

## Architecture and execution path

1. The Rust `graph` command parses ONNX without loading ONNX Runtime. It checks topological wiring, infers shapes for the frozen operator patterns, and prints the tensors crossing each candidate cut. Unknown shape patterns remain unknown rather than receiving a false byte estimate.
2. The Rust `split` command turns a chosen cut into `head.onnx` and `tail.onnx`. Each half receives only the weights it needs. Boundary tensors receive concrete ONNX type and shape declarations for the supplied image size.
3. Rust `run` executes one model on raw little-endian NCHW FP32 data. Rust `run-split` executes the head, copies required boundary tensors, releases the head session, then loads and executes the tail. It writes raw FP32 scores. Keeping the original input alive through the head run fixed an aliasing error discovered during testing.
4. The fixed raster prototype validates that each model operation is independent at every pixel. It reads one GeoTIFF window at a time, applies declared band order and scale/offset, pads partial edge windows, calls `run-split`, crops the result, and writes the matching output window. Rasterio copies the source CRS and transform into the output profile. [Rasterio's window API](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html) supplies the windowed I/O.

Python currently owns GeoTIFF I/O because Rasterio has a Windows wheel in the pinned Python 3.11 environment, while native GDAL is not installed in the workspace. The raw-file bridge and one Rust process per tile are deliberately simple and measurable. They are expensive: the 20-window, 37×53 smoke run took about 2.35 seconds even though each inference is tiny. A persistent worker or integrated raster runtime is needed before this is a practical large-raster product.

## What the measurements say

On the frozen U-Net at 1×3×512×512, three fresh-process runs per configuration gave median peak committed memory of **55.07 MiB** for the whole model and **63.00–69.05 MiB** for three manual cuts. Median total time was **120 ms** whole and **125–133 ms** split. All compared outputs matched exactly in that benchmark. The split consumed more committed memory, largely because materializing boundary tensors while head outputs still exist raises the process peak. Peak working set differs from committed memory; cut 4 had a lower working-set peak but a higher committed-memory peak. The planned limit applies to committed memory, so cut 4 is not a memory win. See [Phase 3 results](phase3-results.md) for the full table and measurement definition.

The raster prototype's test output matched a whole-image pointwise ONNX Runtime reference exactly for both tested tile sizes, including uneven right and bottom edges. The EPSG:32643 CRS, affine transform, two FP32 score bands, and a source metadata tag were preserved. The scheduler rejected a no-data pixel before publishing an output and rejected the spatial U-Net before starting raster execution.

## Remaining gaps

- There is no automatic cut, keep/spill/recompute planner. Manual cuts are exploratory.
- There is no Windows Job Object limit yet. Rust reports each child process's peak committed bytes and working set; the raster report does not include the Python scheduler in a job-wide peak.
- The raster path accepts only pixelwise models. U-Net needs exact halo, stride-aligned windows, and edge handling. Global mean needs a whole-image reduction pass. Global attention needs a blockwise exact schedule. The current validator rejects these rather than returning incorrect maps.
- The raster path launches a new Rust/ONNX Runtime process for every tile and uses temporary raw files. It is a correctness prototype, not a large-raster performance result.
- The output report has no numerical error estimate for an arbitrary user raster because a whole-image reference may not fit. The test harness calculates error on small reference cases.
- There is no independent user trial, real remote-sensing model/raster trial, tuned manual baseline, held-out 1 GiB run, or second physical machine. The frozen feasibility decision has not been made.

## Next execution plan

1. Keep ONNX Runtime sessions alive across raster windows, eliminate the raw-file/process-per-tile bridge, and measure combined scheduler plus backend committed memory.
2. Add a fixed exact schedule for local convolutions: receptive-field halo, stride/pooling alignment, image-edge padding, output cropping, and multi-tile diff against a whole-image reference. Support the frozen U-Net first.
3. Add the global-mean model with an explicit first reduction pass, then the blockwise attention path with a cost estimate and an infeasibility report when appropriate.
4. Implement a planner that chooses keep, spill, or recompute using measured memory and disk costs. Enforce a job-wide committed-memory cap with a Windows Job Object.
5. Run the frozen development and held-out protocol, retain failures and raw reports, compare against the four-hour tuned baseline, then test a real user model and raster.

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
.\.venv\Scripts\python.exe tests\benchmark_phase3.py
```
