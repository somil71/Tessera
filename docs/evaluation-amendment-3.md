# Evaluation amendment 3: replacement independent model

Recorded 2026-11-24 **before downloading, opening, or executing the replacement model**. This amends the exposed attention holdout described in [amendment 1](evaluation-amendment-1.md). All numerical tolerances, memory and timing rules, and decision thresholds in the [frozen protocol](evaluation-protocol.md) remain in force. This amendment records a new external model and model-machine pair; it does not claim that the current planner supports or passes it.

## Reserved model

Use the independently authored [Heliosoph SCUNet ONNX export](https://huggingface.co/Heliosoph/scunet-onnx), variant `scunet_color_25`. Pin repository revision `6d11417ee2fbcc73783c502a238ac115097754fe`. The upstream model card describes one NCHW FP32 input `[batch,3,H,W]`, one NCHW FP32 output `[batch,3,H,W]`, dynamic height and width divisible by eight, RGB values in `[0,1]`, and an `.onnx` graph with a required `.onnx.data` sibling. Do not clip or quantize output scores for numerical comparison.

The repository API lists these exact blobs, to be checked after download:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `scunet_color_25.onnx` | 3,795,638 | `76170f1fcb84d3b3d38c4e7133a658bbbb70504e3dc5e4e4d1f7b0c395c093be` |
| `scunet_color_25.onnx.data` | 73,138,176 | `f46aa0f533e5bb8b2485a6531a205e270f4a7be2447880a12d97f5f00b1e70c9` |

The planner author may use the published model card and repository metadata to determine eligibility, but must not inspect the graph, run preflight, or tune on these files until the implementation and candidate schedules are frozen in a commit. Any such exposure moves this model to development status and requires another replacement holdout.

## Reserved inputs and machine pair

- Generate input pixels with pinned NumPy 2.4.6 as `np.random.default_rng(seed).random((3, height, width), dtype=np.float32)`. Write bands in RGB order to an FP32 GeoTIFF with CRS `EPSG:32643`, `from_origin(500000, 2000000, 1, 1)`, and no nodata. Record file hashes after generation. Use identity band order, scale 1, and offset 0. These synthetic rasters test execution correctness and scheduling, not denoising quality.
- Whole-image correctness inputs: seed `5201` at 64×64 and seed `5202` at 128×128. The unsplit ONNX Runtime CPU output is the reference. Apply both frozen error thresholds to every raw score, including partition edges.
- Performance input: seed `5203` at 512×512. Before running, record a bounded pre-run time estimate. If it exceeds 30 minutes, record the case as infeasible under the existing rule; do not silently shrink it. Retain any first failure.
- Reserve the development laptop's previously untested **1 GiB hard Windows Job Object committed-memory cap and two execution threads** as this model-machine configuration. Pin CPU affinity, temporary drive, ONNX Runtime version, and machine specifications in the run manifest before trials. This is a held-out *configuration* on the same physical laptop, not cross-machine evidence.

## Evaluation sequence

1. Commit the implementation, schedule search, and the human-declared manual candidate list before accessing the model bytes. Record exact Git commit and hashes.
2. Download the pinned graph and external data, verify both hashes, and generate the three reserved rasters. Keep the raw assets outside Git under ignored `models/` and `data/`.
3. Run the small unsplit [whole-image oracle](evaluation-runner.md#small-whole-image-numerical-oracle) and the automatic runner; compare raw outputs at both sizes. Retain all failures and diagnostic reports.
4. Give the manual tuner the frozen four-hour allowance and fixed candidate list. Keep every attempt. Then alternate a first-run pair and five repeated benchmark pairs at the same cap, threads, affinity, and storage device.
5. Apply the existing pass, fail, or inconclusive rules without adjusting these inputs or thresholds after seeing results. An unsupported graph or an over-cap run is a failed held-out case. A different independent user's real model and georeferenced raster trial is still required for product completion.

This model provides an independent architecture test; its natural-image denoising task does not substitute for the required remote-sensing user trial. SCUNet may contain operators outside the current product set. That uncertainty is the reason to preserve it as a genuinely unseen test rather than tune toward its graph.
