# Local graph dependency analysis

Recorded 2026-11-23 and extended 2026-11-24. `scripts/local_graph.py` derives a conservative input halo and tile-origin alignment from a validated NCHW FP32 ONNX graph. It walks supported graph operations backward from one output pixel to the input coordinates that can affect it. It checks that this dependency repeats after the derived spatial alignment period. The rules cover explicit-pad Conv, ConvTranspose, MaxPool, pointwise operations, channel-axis Concat, and a bounded nearest-neighbor Resize pattern. Unsupported attributes and global operations are rejected.

For the existing U-Net fixture, the analyzer derives a four-pixel halo and two-pixel alignment on each axis. The fixed U-Net raster runner and automatic U-Net probes now use these derived values. Its earlier whole-image correctness tests still pass, including uneven edge tiles and disk-boundary execution.

`tests/fixtures/smoke_local_graph.py` adds a separate two-convolution branched graph with channel Concat. The analyzer derives halo 2 and alignment 1. Two tiled ONNX Runtime executions match whole-image logits exactly on a 37×53 image. Rust `split` resolves the Concat boundary shape at 37×53. The test also confirms that a spatial global-mean graph and spatial-axis Concat are rejected as local.

For Resize, the accepted attributes are `nearest`, `asymmetric`, and `floor`, with a dynamic `sizes` input. The analyzer separates Shape/Constant/Gather/Unsqueeze/Slice/Cast and axis-zero Concat shape plumbing from image values, uses ONNX shape inference with concrete 128² and 256² inputs to verify a stable integer upsample factor, and rounds the derived halo to the tile-origin alignment. Full-raster and cropped tile shapes are checked again for the same factor. Other Resize semantics and image-value use of shape-only operators are rejected. This is a bounded spatial dependency proof, not a general ONNX shape interpreter.

For shape graphs that ONNX cannot infer through a dynamic Resize, the analyzer may derive an expected output shape from a channel Concat sibling. It then executes only the Resize size calculation, replacing `Shape` of known tensors with concrete dimensions, and requires the requested four dimensions to match that expected shape. Large convolution initializer payloads are omitted from shape-inference probes while their dimensions remain available. The exposed SAR development run and its limits are recorded in [the SAR result](sar3-holdout-result.md).

`tests/fixtures/smoke_resize_local.py` builds a downsample/shape-plumbing/Resize graph. Its 66×98 image produces halo 2 and alignment 2; a 65-pixel input is rejected because its resize ratio is noninteger. Whole-image, fixed GeoTIFF, and automatic GeoTIFF scores match exactly under a 128 MiB cap. This establishes end-to-end behavior for that pattern, not for every graph that contains Resize.

The [general local raster runner](generic-local-raster-results.md) consumes the analysis. It validates output dimensions for the full image and all cropped shapes. The exposed HydroFlow graph passes local dependency analysis with halo 544 and alignment 32. A fixed 25-tile 1088² synthetic-input run under a 2 GiB cap matched its whole-image ONNX Runtime scores exactly; see the [external-model probe](real-model-compatibility.md#hydroflow-development-probe). Broader automatic-planner and independently held-out numerical validation remain required.

The inverse-window rules follow the official [ONNX Conv](https://onnx.ai/onnx/operators/onnx__Conv.html), [ConvTranspose](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html), [MaxPool](https://onnx.ai/onnx/operators/onnx__MaxPool.html), and [Resize](https://onnx.ai/onnx/operators/onnx__Resize.html) operator definitions. Concrete shapes use [ONNX shape inference](https://onnx.ai/onnx/api/shape_inference.html), whose data propagation has documented limits. Exact behavior is limited to the accepted attributes in code and the verified shape classes.

```powershell
cargo build --locked
.\.venv\Scripts\python.exe tests\fixtures\create_eval_models.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_local_graph.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_resize_local.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_unet_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_auto_raster.py
```
