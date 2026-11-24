# Evaluation amendment 4: second replacement holdout

Recorded 2026-11-24 before downloading, opening, preflighting, or executing this model. The [SCUNet replacement failed unseen preflight](scunet-holdout-result.md) and is exposed development material. This amendment reserves a new external model; it does not erase that failure. The original protocol's numerical tolerances, 1 GiB/two-thread held-out configuration, four-hour manual allowance, five alternated repeated pairs, 30-minute pre-run cutoff, and decision thresholds remain in force.

## Model and independent task

Use the independently authored [`ayoubft/fraXteX` RGB U-Net ONNX](https://huggingface.co/ayoubft/fraXteX), file `rgb/unet-rgb.onnx` at repository revision `890ccb47e6a8e015126bab5dab13cbeca39066de`. The repository API reports 124,145,004 bytes and SHA-256 `b50e3c714a0447fae3da175c999c239b503cb98c0c3363e8d790f7c0fd7948fe`; verify after download. The model card describes a three-channel, dynamic multiple-of-16 RGB input of raw 0–255 FP32 values, preprocessing embedded in the graph, and one-channel sigmoid fracture probability output. This geological imagery task is external to the implementation fixtures and HydroFlow work. The model card and repository metadata are the only sources consulted before freeze; **no ONNX graph bytes were accessed**.

## Reserved inputs, machine, and baseline

Generate RGB pixels with pinned NumPy 2.4.6 as `np.random.default_rng(seed).integers(0, 256, size=(3, height, width), dtype=np.uint8).astype(np.float32)`. Write them in RGB order to FP32 GeoTIFF with CRS EPSG:32643, `from_origin(500000, 2000000, 1, 1)`, and no nodata. These synthetic values test the execution contract, not fracture-map quality. Whole-image correctness cases are seed 6201 at 64×64 and seed 6202 at 128×128. The performance case is seed 6203 at 512×512. Before running that case, record a bounded pre-run time estimate and mark it infeasible if above 30 minutes.

Use the same Windows i5-13450HX laptop, hard 1,073,741,824-byte Job Object committed-memory limit, two automatic execution threads, full 16-logical-processor affinity mask `0xFFFF`, and E: for outputs and spill. Python 3.11.6, NumPy 2.4.6, ONNX 1.23.2, ONNX Runtime CPU 1.31.0, and rasterio 1.4.4 are pinned. Record all artifact hashes and exact Git commit before download.

The predeclared manual list is [`evaluation/fraxet-manual-candidates.json`](../evaluation/fraxet-manual-candidates.json). It uses legal multiple-of-16 tiles 64, 128, 256, and 512; cut 0; RAM and disk storage; and two one-thread alternatives. It is fixed before graph access. Keep each failed attempt and do not revise the candidate list in response to results. An unsupported graph or over-budget execution is a failed held-out case, and later support changes require another fresh holdout for a new pass claim.
