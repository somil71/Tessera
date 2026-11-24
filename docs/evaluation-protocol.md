# Frozen feasibility and evaluation protocol

**Amendments:** The model historically named `attention_holdout.onnx` was exposed during development and is no longer eligible as unseen held-out evidence. See [evaluation amendment 1](evaluation-amendment-1.md). [Evaluation amendment 2](evaluation-amendment-2.md) records how to interpret a Windows job peak counter that exceeds the enforced cap after failed allocation probes. [Evaluation amendment 3](evaluation-amendment-3.md) froze SCUNet, which [failed preflight](scunet-holdout-result.md). [Amendment 4](evaluation-amendment-4.md) froze FraXet, which also [failed preflight](fraxet-holdout-result.md). Neither establishes a protocol pass.

Frozen on 2026-11-22 before planner implementation. Changes to models, tolerances, baselines, or pass criteria must be versioned and accompanied by a new held-out evaluation. All times below refer to completed runs, including model load, planning, input reads, and output writes.

## Models

Generate the three versioned ONNX fixtures with `tests/fixtures/create_eval_models.py` using Python 3.11, NumPy 2.4.6, and ONNX 1.23.2. The script embeds fixed seeded FP32 weights. The generated files stay out of Git and must match these SHA-256 hashes:

| Model | Role | SHA-256 |
| --- | --- | --- |
| `unet_dev.onnx` | Development: variable-size U-Net with skip connection | `568D9D37B85AED0349221AB1EFE13C9D03AE3ED40D6F94657233DC61FA79CCBE` |
| `global_mean_dev.onnx` | Development: spatial ReduceMean followed by centering and convolution | `95C318EF50694C2A226B9F0BE612E938BB4228988AFBCE6718A002E9B8FC7719` |
| `attention_holdout.onnx` | Held out until the planner and attention support are frozen | `CC86BB4A65F65CAD9126F8249F1F1C7C84D01732AE07AF9C540C1F4DF20A66F5` |

These are synthetic engineering fixtures. They establish execution semantics and stress memory planning; they do not establish map quality or usefulness to a mapping analyst. A later independent-user trial uses a real model and raster under the v1 contract.

## Inputs and reference

- Generate deterministic FP32 input rasters from development seeds 3101 and 3102 and held-out seeds 4101, 4102, and 4103. Record exact dimensions, band order, preprocessing, and hashes in each run report. Do not tune the planner on held-out seeds or the attention model.
- Development correctness cases use U-Net dimensions 32×48 and 66×98, plus global-mean dimensions 33×47 and 96×80. The U-Net dimensions are even because its pooling path halves each spatial axis.
- Held-out correctness cases use attention dimensions 8×10, 64×64, and 96×96. Reserve input seeds 4101, 4102, and 4103. The attention model's computation grows quadratically, so record a pre-run time estimate and mark a case infeasible if the estimate exceeds 30 minutes on the test CPU.
- For each supported model, the reference is unsplit ONNX Runtime CPU execution on an input that fits in RAM. Compare raw FP32 outputs elementwise. Require maximum absolute error ≤ `1e-4` **and** maximum relative error ≤ `1e-3` for elements with reference magnitude ≥ `1e-3`. Record both maxima, including edges and chosen partition boundaries.
- The large-image performance cases use U-Net 8192×8192 and global-mean 8192×8192. They may be reduced only in a newly versioned protocol if the reference/baseline cannot finish on the available machine. Correctness of large runs is additionally checked by repeat runs with different legal partition boundaries. Small whole-image references remain the primary numerical oracle.

## Memory and baseline

- The execution process and its children run in a Windows Job Object with a 2 GiB job-wide committed-memory limit. Record peak job committed memory and peak working set separately. The CLI, ONNX Runtime sessions, weights, temporary buffers, and input/output machinery count against the limit. Report mapped-file and filesystem cache observations where measurable.
- The development machine is the current i5-13450HX laptop with 10 cores, 16 logical processors, and 34,053,414,912 bytes of installed RAM. Development uses a 2 GiB cap and at most four execution threads. Reserve the held-out 1 GiB cap and two-thread configuration until the planner is frozen. A second physical machine, if available, is an additional held-out machine pair rather than a replacement for this configuration.
- The manual baseline uses the same ONNX Runtime version and supported graph transformations. It may choose tile size, cuts, keep/spill/recompute decisions, and thread count. Its tuner gets four hours per model-machine pair, with all attempted schedules retained. A benchmark author other than the planner author should tune it when possible.
- Both planner and baseline run with the same CPU affinity, thread limit, image, memory cap, and temporary storage. Alternate their run order. Report first-run time and median of five repeated completed runs, plus planner profiling time, disk reads/writes, peak temporary storage, and any failure.

## Decision

A clear pass requires all held-out cases that finish within the pre-run time limit to satisfy the numerical contract and memory limit, with planner median runtime ≤ 1.25× the tuned baseline on each case and no manual schedule choice. A numerical violation or memory-limit failure is a failure. Runtime > 1.5× baseline on a majority of held-out pairs is a performance failure. Results between the pass and failure thresholds are inconclusive and require a documented bottleneck analysis.

The 30-minute attention cutoff is a feasibility result, not a pass. If every attention case is infeasible, v1 cannot claim demonstrated attention support. A cross-machine claim requires a second physical machine or a documented remote runner with a distinct hardware profile; one laptop does not establish it.

Store the exact commands, tool versions, model and input hashes, raw outputs, baseline tuning log, and failed runs with the final report.
