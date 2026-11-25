# Version 1 support contract

Raster Compiler v1 targets Windows x64 CPU inference. A user supplies an ONNX model, a GeoTIFF raster, an output path, and a maximum committed-memory budget. The output is a georeferenced GeoTIFF of raw per-pixel FP32 scores plus a machine-readable run report. A class map may be derived from those scores, but accuracy comparisons use the raw scores.

## Accepted inputs

- ONNX models with one NCHW FP32 image input and one NCHW FP32 raster output. Batch size is one. Height and width may be dynamic if the model supports them.
- GeoTIFF inputs with the number of bands required by the model. Preprocessing is declared explicitly by the user: band order, scale, offset, and no-data handling. Defaults are no band reordering, scale 1, offset 0, and an error on no-data pixels.
- A Windows Job Object committed-memory budget and a temporary directory with enough free space for the proposed plan.

## Operator support target

The v1 planner supports the operators exercised by the frozen fixtures: Conv, ConvTranspose, Relu, Add, Sub, MaxPool, ReduceMean over spatial axes, Shape, Reshape, Transpose, MatMul, and Softmax. The independent-pixel and general local schedules additionally validate inference-mode BatchNormalization, Sigmoid, and Mul; the general local schedule validates channel-axis Concat. Its constant-padding path also validates shape-only ConstantOfShape, Reshape, and Transpose leading to spatial-only constant-mode Pad. Initializers, broadcasting, and dynamic spatial dimensions are supported for the specified patterns. Other operator versions, layouts, axes, data types, and control-flow nodes require explicit compatibility checks. The tool must report the first unsupported node before inference and identify its operator and reason.

Whole-image execution is the reference only for models whose declared inference contract permits the requested image dimensions. A model trained and intended for fixed patches may require a different reference and cannot be silently evaluated against whole-image output.

## Completion criteria

The CLI must automatically select a plan, execute it under the declared committed-memory limit, write a complete georeferenced raster, and produce an auditable report of numerical error, runtime, peak committed memory, peak working set, disk bytes, and temporary storage. Interrupted runs must either resume safely or fail with a clear reason. A clean Windows installation and an independent user's supported model and raster must work without developer-selected schedules.

This contract defines a bounded v1 product. Passing the feasibility experiment in [evaluation-protocol.md](evaluation-protocol.md) permits work toward these completion criteria; the present scaffold does not yet satisfy them.
