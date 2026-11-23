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
| Phase 5: automatic calibration and attention prototype | A required-cap CLI profiles tile sizes and cuts for the three local/reduction schedules and routes the validated global-attention pattern to blockwise online softmax with a pre-run cost gate | Removes developer-selected local schedules and avoids materializing the quadratic attention score matrix | Three local/reduction auto outputs matched references under 128 MiB; attention 8×10, 64×64, and 96×96 passed frozen tolerances. An 8192×8192 attention case was rejected before staging. See [planner results](phase5-auto-plan-results.md) and [attention results](phase5-attention-results.md) |
| Phase 6: large development rasters | Chunked deterministic 8192×8192 input generation, budgeted GDAL cache, bounded-window output comparison, and a hash-checked audit manifest | Tests actual large GeoTIFF execution and ties comparison results to the output files that were measured | U-Net and global-mean automatic runs finished under the 2 GiB job cap; alternate legal partitions had zero differing FP32 values in 1,024 windows each. These are single development runs, not the independent feasibility decision. See [large-raster results](phase6-large-raster-results.md) |

The source of each piece is [graph.rs](../src/graph.rs), [split.rs](../src/split.rs), [execute.rs](../src/execute.rs), [memory.rs](../src/memory.rs), and [run_raster_fixed.py](../scripts/run_raster_fixed.py). Test fixtures and benchmark scripts live in `tests/`.

## Architecture and execution path

1. The Rust `graph` command parses ONNX without loading ONNX Runtime. It checks topological wiring, infers shapes for the frozen operator patterns, and prints the tensors crossing each candidate cut. Unknown shape patterns remain unknown rather than receiving a false byte estimate.
2. The Rust `split` command turns a chosen cut into `head.onnx` and `tail.onnx`. Each half receives only the weights it needs. Boundary tensors receive concrete ONNX type and shape declarations for the supplied image size.
3. Rust `run` executes one model on raw little-endian NCHW FP32 data. Rust `run-split` executes the head, copies required boundary tensors, releases the head session, then loads and executes the tail. It writes raw FP32 scores. Keeping the original input alive through the head run fixed an aliasing error discovered during testing.
4. The fixed pixelwise raster prototype validates that each operation is independent at every pixel. A second, topology-checked U-Net path aligns output tiles to the pool stride, adds a four-pixel input halo, clips at true image edges, and crops the returned scores. The U-Net checker now follows graph wiring rather than requiring the development fixture's tensor names; a renamed ONNX graph matched whole-image output under both fixed and automatic schedules. Both paths send raw tile bytes to persistent Rust `serve-split-pipe` workers and write georeferenced FP32 windows. Rasterio copies the source CRS and transform into the output profile. [Rasterio's window API](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html) supplies the windowed I/O.
5. The global-mean path first streams every raster window to compute the full-image channel mean, replaces the spatial reduction with that FP32 constant, then executes the remaining pointwise graph under the same job-wide cap. Its output uses the same georeferenced FP32 writer.
6. The automatic runner calibrates legal tile and cut choices using the Rust worker under the same job cap, ranks them by measured cost and raster geometry, then executes a feasible choice through one of the three fixed paths. It records probe costs and failed attempts in the output report.
7. The attention path validates the exposed Shape/Reshape/MatMul/Softmax graph, stages FP32 pixel vectors in disk-backed arrays, and computes stable query/key blocks without storing the full score matrix. It estimates the quadratic cost before input staging and reports cases above the 30-minute gate as infeasible.

Python currently owns GeoTIFF I/O because Rasterio has a Windows wheel in the pinned Python 3.11 environment, while native GDAL is not installed in the workspace. The first version used a raw-file bridge and one Rust process per tile. The default now keeps both ONNX Runtime sessions loaded and sends tile data over pipes. On the 12-window smoke workload, median elapsed time fell from 1.647 s with fresh processes to 0.240 s with pipes; Rust worker peak committed memory rose from 16.95 to 18.22 MiB. See [worker measurements](phase4-worker-results.md). The newer optional memory budget joins the Python scheduler and all Rust model processes to a Windows Job Object and records its peak committed memory; [the cap test](phase4-job-memory-results.md) also verifies an over-budget commit fails.

## What the measurements say

On the frozen U-Net at 1×3×512×512, three fresh-process runs per configuration gave median peak committed memory of **55.07 MiB** for the whole model and **63.00–69.05 MiB** for three manual cuts. Median total time was **120 ms** whole and **125–133 ms** split. All compared outputs matched exactly in that benchmark. The split consumed more committed memory, largely because materializing boundary tensors while head outputs still exist raises the process peak. Peak working set differs from committed memory; cut 4 had a lower working-set peak but a higher committed-memory peak. The planned limit applies to committed memory, so cut 4 is not a memory win. See [Phase 3 results](phase3-results.md) for the full table and measurement definition.

The raster prototype's test output matched a whole-image pointwise ONNX Runtime reference exactly for both tested tile sizes, including uneven right and bottom edges. The EPSG:32643 CRS, affine transform, two FP32 score bands, and a source metadata tag were preserved. The scheduler rejected a no-data pixel before publishing an output and rejected the spatial U-Net before starting raster execution.

The separate U-Net raster path matched whole-image logits exactly at tile sizes 6, 8, 16, and 24 on a 38×54 raster, including a reordered and scaled input. It preserved the georeferencing and rejected an odd-width input without publishing output. It groups shape-specific tiles so that only one Rust worker remains live at a time. See [the U-Net report](phase4-unet-results.md).

The two-pass global-mean path passed the protocol's 33×47 and 96×80 correctness sizes at tile sizes 8, 13, and 24. Its largest absolute error against whole-image ONNX Runtime logits was 1.49×10⁻⁸ and largest eligible-relative error 1.92×10⁻⁶. It rejected no-data during the reduction pass before publishing output. See [the global-mean report](phase4-global-mean-results.md).

The first automatic planner run covered all three supported model families on a 32×48 raster under 128 MiB without manual tile or cut selection. Pointwise and U-Net outputs matched exactly; global mean differed by at most 1.49×10⁻⁸. Its U-Net path profiled 35 candidates, so planning time is material on this small test and remains part of the final performance comparison. See [the planner report](phase5-auto-plan-results.md).

The blockwise attention path passed whole-image comparisons at 8×10, 64×64, and 96×96 under 128 MiB. Largest maximum absolute error was 1.74×10⁻⁵ and largest eligible-relative error 1.01×10⁻⁴. The 96×96 run recorded 89.61 MB peak job commitment and 434,460 bytes of peak temporary storage. A metadata-only 8192×8192 raster was rejected on cost before staging. The historical attention model is exposed development material, not the new holdout. See [attention results](phase5-attention-results.md).

Two deterministic 8192×8192 three-band development GeoTIFFs now exercise the actual raster path. The automatic U-Net run took 113.0 s, including 13.9 s of planning, and peaked at 1,764.6 MB committed under the 2 GiB cap. The automatic global-mean run took 102.7 s, including 2.0 s of planning, and peaked at 1,685.5 MB. Alternate legal partitions produced zero differing FP32 output values, with matching CRS and transforms. Comparison reports include SHA-256 hashes of both outputs, and the audit script checks those hashes against current files. These are single development runs; see [Phase 6 results](phase6-large-raster-results.md).

## Remaining gaps

- An empirical planner now chooses tile size and graph cut for the three fixed schedules. It does not choose keep/spill/recompute, estimate disk costs, or analyze arbitrary supported graphs. Its ranking must still be measured against the tuned baseline.
- The fixed runners can optionally enforce a Windows Job Object committed-memory cap; the automatic CLI requires it. Current reports include a 50 ms sampled sum of live job-process working sets and cumulative job I/O transfer bytes. The sampled working-set maximum can miss shorter peaks, and transfer bytes include more than physical disk activity. Exact peak working set and disk-only bytes remain open.
- Raster execution has four validated patterns: pixelwise models, the development U-Net topology, first-node spatial global mean, and the exposed self-attention fixture. There is no generic receptive-field, global-operation, or transformer-graph analyzer. Unsupported models are rejected rather than silently miscomputed.
- The automatic runner identifies the first operator outside the validated schedule set before inference. A graph that uses only familiar operators but has an unsupported arrangement still receives the combined family-validation reasons rather than a single offending node.
- The default raster path has a persistent Rust worker and no raw tile bridge, but Python still owns GeoTIFF I/O. Large development runs exist, but repeated held-out performance results do not.
- The output report has no numerical error estimate for an arbitrary user raster because a whole-image reference may not fit. The test harness calculates error on small reference cases.
- There is no independent user trial, real remote-sensing model/raster trial, tuned manual baseline, held-out 1 GiB run, or second physical machine. The originally designated attention holdout was exposed during development; [evaluation amendment 1](evaluation-amendment-1.md) requires a new independent holdout. The frozen feasibility decision has not been made.
- An [alternated benchmark runner](evaluation-runner.md) now preserves paired automatic/manual runs and their outputs, but its smoke test used one repeat and an untuned development schedule. The required five repeats, four-hour tuning history, independent holdout, and decision are outstanding.
- A [Windows quickstart](windows-quickstart.md) and `scripts/doctor.py` now check the pinned environment and basic model/raster compatibility. They have been exercised on this development machine, not a clean independent installation.

## Next execution plan

1. Generalize the local-model halo and alignment analysis beyond the frozen U-Net topology, and add measured keep/spill/recompute planning.
2. Generalize attention scheduling beyond the exposed fixture and measure repeated disk-block traffic under realistic feasible rasters.
3. Extend the empirical planner with keep/spill/recompute and disk costs. Add job-wide working-set measurement and a required cap for every final user path.
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
.\.venv\Scripts\python.exe tests\fixtures\smoke_attention_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_auto_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_job_memory.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_worker.py
.\.venv\Scripts\python.exe tests\benchmark_phase3.py
.\.venv\Scripts\python.exe tests\benchmark_worker.py
.\.venv\Scripts\python.exe tests\performance\summarize_large_runs.py
```
