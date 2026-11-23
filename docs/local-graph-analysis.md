# Local graph dependency analysis

Recorded 2026-11-23. `scripts/local_graph.py` derives a conservative input halo and tile-origin alignment from a validated NCHW FP32 ONNX graph. It walks supported graph operations backward from one output pixel to the input coordinates that can affect it. It checks that this dependency repeats after the derived spatial alignment period. The rules cover explicit-pad Conv, ConvTranspose, MaxPool, pointwise operations, and channel-axis Concat. Unsupported attributes and global operations are rejected.

For the existing U-Net fixture, the analyzer derives a four-pixel halo and two-pixel alignment on each axis. The fixed U-Net raster runner and automatic U-Net probes now use these derived values. Its earlier whole-image correctness tests still pass, including uneven edge tiles and disk-boundary execution.

`tests/fixtures/smoke_local_graph.py` adds a separate two-convolution branched graph with channel Concat. The analyzer derives halo 2 and alignment 1. Two tiled ONNX Runtime executions match whole-image logits exactly on a 37×53 image. Rust `split` resolves the Concat boundary shape at 37×53. The test also confirms that a spatial global-mean graph and spatial-axis Concat are rejected as local.

The [general local raster runner](generic-local-raster-results.md) now consumes this analysis. It validates output dimensions for the full image and all cropped shapes and has a separate branched-graph whole-image correctness fixture. Its accepted operator/attribute set is bounded; a graph with spatial reduction, Resize, or an unmodeled operator is rejected. Broader numerical validation on independently authored models remains required.

The inverse-window rules follow the official [ONNX Conv](https://onnx.ai/onnx/operators/onnx__Conv.html), [ConvTranspose](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html), and [MaxPool](https://onnx.ai/onnx/operators/onnx__MaxPool.html) operator definitions. Exact behavior is limited to the accepted attributes in code.

```powershell
cargo build --locked
.\.venv\Scripts\python.exe tests\fixtures\create_eval_models.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_local_graph.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_unet_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_auto_raster.py
```
