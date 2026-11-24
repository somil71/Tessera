# FTP-B3 independent holdout preflight

Recorded 2026-11-24 after [amendment 5](evaluation-amendment-5.md) and the two artifact hashes were committed. The preflight ran on the pinned 50,311,056-byte ONNX model and deterministic eight-band 512×512 GeoTIFF under the declared 1 GiB budget. The generated complete record is `runs/holdout/ftp-b3/preflight.json` (ignored by Git).

`scripts/doctor.py` reported **failed** before raster inference: first unsupported node **2**, `ConstantOfShape`, has no validated raster schedule. The model declares one FP32 NCHW input `[batch, 8, height, width]` and one FP32 NCHW output `[batch, 3, height, width]`. The input has eight FP32 bands and EPSG:32643. Model and input hashes matched the frozen amendment.

The graph contains 959 nodes and additional operators outside the current validated raster schedules, including `GlobalAveragePool`, `Gather`, `Pad`, `Slice`, and `Unsqueeze`. Supporting only `ConstantOfShape` would not establish that the model can run exactly under the cap. No automatic plan, numerical oracle, tuned baseline, or paired benchmark was run. This is a **failed independent holdout compatibility result**, not a feasibility pass.

The graph is now exposed development material. Any subsequent support work on FTP-B3 needs a newly frozen independent holdout for the decision test.
