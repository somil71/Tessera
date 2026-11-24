# Adaptive RAM calibration

Recorded 2026-11-24. This is a development optimization, not a held-out performance result.

The automatic planner still starts with the largest legal tile and measures all graph cuts at each tile size. After a feasible RAM plan exists, it stops probing smaller tiles when two successive sizes each predict runtime greater than 1.5 times the fastest measured plan. It records the stopping tile and unprobed RAM sizes. If all attempted full-raster executions fail, it profiles those remaining RAM sizes, retries, and only then searches the remaining disk-boundary sizes for local graphs. Failed attempts remain in the run report.

On the same 4096×4096 development GeoTIFF and 128 MiB cap used in [general local raster results](generic-local-raster-results.md), the earlier exhaustive RAM calibration measured 36 candidates and selected RAM tile 256/cut 0. The adaptive run selected the same plan, measured 24 candidates, and stopped after tile 16. It left tiles 8, 4, and 2 available for recovery. Planning time was 5.86 s versus 8.97 s; total reported time was 25.32 s versus 29.47 s. These are single runs, so the time difference is an observation, not a stable performance estimate. The two resulting GeoTIFF files had identical SHA-256 hashes, CRS, transforms, and FP32 values in all 256 comparison windows.

`tests/fixtures/smoke_adaptive_fallback.py` makes the first 21 U-Net full-run attempts fail deterministically after measurement. The planner then profiles the previously skipped smaller RAM sizes, selects tile 4, and matches the whole-image ONNX Runtime logits within the frozen 1e-4 absolute tolerance. The test verifies that no unprobed RAM sizes remain in the final report. The existing automatic raster and storage-planner fixtures also pass.

The 1.5 factor and two-size stopping rule are engineering heuristics. They reduce calibration work without guaranteeing a globally fastest plan; the tuned independent baseline and repeated held-out measurements remain the decision gate.
