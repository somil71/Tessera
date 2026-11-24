# Automatic tile and cut calibration

Recorded 2026-11-23. `scripts/run_raster_auto.py` accepts a model, GeoTIFF, output, preprocessing, and required job-wide committed-memory budget. It calibrates three raster schedules: pixelwise, frozen local U-Net, and first-node spatial global mean. It profiles power-of-two tile sizes up to 512 by default and every legal internal graph cut for the pointwise and U-Net models. Each probe loads a Rust split worker under the same Windows Job Object, runs one warmup plus three timed FP32 tile requests, and records startup time, median inference time, child committed-memory peak, and a conservative scheduler-plus-child bound. A global-mean probe uses a transformed model with a placeholder mean; the full run computes the actual image-wide mean.

The estimated full-run cost is measured per-tile time times raster tile count plus worker startup time times the number of distinct halo shapes. Within 10% of the best predicted time, the planner favors larger tiles to reduce window I/O and measurement-noise sensitivity. It then tries the ranked plans on the full raster, retaining failed attempts. The selected output report contains every probe, its chosen plan, planning time, total time, and job memory peak. If no candidate finishes under the budget, a `.plan.failure.json` record is written and no output is published.

`tests/fixtures/smoke_auto_raster.py` ran all three families on a 32×48 GeoTIFF under a 128 MiB job cap without tile or cut arguments. Pointwise and U-Net outputs matched whole-image ONNX Runtime logits exactly; global mean's maximum absolute error was 1.49×10⁻⁸, within the frozen tolerance. The pointwise case also checked band reordering and scale/offset. A 60 MiB cap produced a no-feasible-plan report and no output raster. These are small engineering cases, not the frozen 8192×8192 performance or independent held-out evaluation.

At the time of this Phase 5 measurement, this was an empirical tile-and-cut planner without keep/spill/recompute, disk cost calibration, or generic local analysis. It probed up to 35 U-Net candidates on the small test raster; that planning overhead was included in total runtime. Later [storage planning](phase8-storage-planner-results.md), [general local execution](generic-local-raster-results.md), and [adaptive calibration](adaptive-calibration-results.md) expanded the implementation. The candidate cost model remains a ranking heuristic, not a guaranteed runtime bound. Final performance must be compared with the four-hour tuned baseline under the frozen protocol.

The later [attention implementation](phase5-attention-results.md) is now routed through the same automatic CLI. It uses a separate blockwise cost gate rather than tile-and-cut calibration. The original three-family measurements above remain unchanged.

Reproduce with:

```powershell
cargo build
.\.venv\Scripts\python.exe tests\fixtures\create_eval_models.py
.\.venv\Scripts\python.exe tests\fixtures\create_pointwise_model.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_auto_raster.py
```
