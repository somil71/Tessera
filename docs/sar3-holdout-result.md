# SAR shoreline independent holdout preflight

Recorded 2026-11-25. [Amendment 6](evaluation-amendment-6.md), the logits-only projection algorithm, source revision, input seeds, cap, numerical thresholds, and paired benchmark rule were committed before model download or graph access. The downloaded source and generated projected model and rasters matched the hashes committed in `d80b4ca`. The complete generated doctor record is `runs/holdout/sar3/preflight.json` (ignored by Git).

The projected model declares one FP32 NCHW three-band input and one FP32 NCHW two-logit output, both with dynamic spatial dimensions. The 512×512 three-band GeoTIFF has EPSG:32643, no nodata, and the frozen SHA-256 hash. The model has 194 selected logits-ancestor nodes. The installation doctor reported **failed** before inference: first unsupported node **1**, `Squeeze`, has no validated raster schedule.

No automatic schedule, numerical oracle, manual tuning, or five-pair benchmark was run. This is a **failed independent compatibility result** and cannot establish the feasibility pass. The graph is now exposed development material. Subsequent inspection showed the `Squeeze` operations are part of shape calculations; supporting that pattern later will require another newly frozen model for an unseen decision test.
