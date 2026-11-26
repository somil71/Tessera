# Deforestation U-Net independent holdout preflight

Recorded 2026-11-25. [Amendment 7](evaluation-amendment-7.md), the repository revision, input seeds, cap, and numerical thresholds were committed before model download or graph inspection. The downloaded model (7,717,953 bytes, SHA-256 `a6c0329aaef4b0758d90cbfd80164bb045b8d3dcfbf82072b96d1d0566e96e8a`) and generated 10-band 512×512 GeoTIFF (`deforestation-512.tif`, 10,489,262 bytes, SHA-256 `ef66fc99aff2a1d3961caa7f42f6136877c4106c9133c0a0e41f626f13a2c0e1`) matched the hashes in `runs/holdout/deforestation/frozen-artifacts.json`.

The model consists of 38 nodes entirely within the supported local operator set (`15 Conv`, `14 Relu`, `3 MaxPool`, `3 Concat`, `3 ConvTranspose`). However, the exported ONNX graph specifies fixed input spatial dimensions `[1, 10, 512, 512]` and fixed output dimensions `[1, 3, 512, 512]` rather than dynamic spatial dimensions `dim_param`.

`scripts/doctor.py` reported **failed** before inference under the 1 GiB Job Object budget:
`model does not match a supported raster schedule: local: local raster needs dynamic spatial dimensions`.

No automatic plan, numerical oracle, manual tuning, or 5-pair benchmark was run. This is a **failed independent compatibility result** under the frozen evaluation protocol. The model is now exposed development material.
