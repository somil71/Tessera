# Large local graph calibration experiment

Recorded 2026-11-24 on the exposed HydroFlow development model. This is **single-run development evidence**, not the newly frozen independent holdout or the protocol's tuned manual baseline. The model and synthetic 1088×1088 twelve-band input are pinned in the [real-model compatibility probe](real-model-compatibility.md#hydroflow-development-probe). Every run used two execution threads and a hard 2 GiB Windows Job Object committed-memory limit. Whole-image ONNX Runtime scores and GeoTIFF georeferencing were compared after each successful automatic run; all 1,183,744 FP32 scores matched exactly.

| Schedule | Probed candidates | Planning (s) | Total (s) | Outcome |
| --- | ---: | ---: | ---: | --- |
| Fixed RAM, tile 256, cut 0 | Manual schedule | — | 55.17 | 25 tiles; exact scores |
| Initial automatic search | 32 | 155.81 | 197.41 | Disk, tile 512, cut 0; exact scores |
| Four structural cuts, model released before probing | 8 | 59.50 | 101.42 | Disk, tile 512, cut 0; exact scores |
| Stop after first feasible structural cut | 2 | 27.52 | 69.30 | Disk, tile 512, cut 0; exact scores |
| One timed measurement for probes over 1M pixels | 2 | 16.76 | 58.71 | Disk, tile 512, cut 0; exact scores |

The final automatic run's sampled peak private commitment was 2,143,637,504 bytes, below the enforced 2,147,483,648-byte limit. The raw `PeakJobMemoryUsed` counter reached 2,200,666,112 bytes **after a failed RAM probe**; the [Windows counter amendment](evaluation-amendment-2.md) explains this observed discrepancy. The selected disk probe succeeded and the complete output was published. The selected plan wrote and read graph-boundary spill data and used nine tiles; the 58.71-second total includes model load, planning, inference, raster I/O, and output publication.

The planner now keeps exhaustive cut search for graphs with at most 32 nodes. For a larger local graph it samples up to four structural cuts, tries disk after a failed RAM probe, and stops when a full schedule succeeds. It defers smaller tiles with the same halo-expanded probe shape but restores them if larger full runs fail. If the sampled cuts fail, it searches the remaining cuts before declaring no plan. This favors total single-run time over extensive calibration. The fallback may still be expensive, and the first feasible large-graph schedule may not have the fastest execution phase.

The final automatic total was 1.064× the **single fixed development run** (58.71/55.17). This is not a protocol result: the baseline had no four-hour tuning history, the runs were not alternated five times, and the input was synthetic. The frozen external holdout in [amendment 3](evaluation-amendment-3.md) remains unopened.
