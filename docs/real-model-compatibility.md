# Public remote-sensing model compatibility probe

Recorded 2026-11-23. This is a development compatibility probe, not the independent holdout or a map-quality test.

The external [FieldsoftheWorld FTW-Q model](https://huggingface.co/fieldsoftheworld/ftw-q) is an agricultural-field boundary U-Net with an ONNX FP32 export, dynamic spatial dimensions constrained to multiples of 32, 16 input bands, and three output logits. Its model card specifies Sentinel-2 quarterly mosaics and preprocessing. We pinned Hugging Face revision `fc182b29879d5bcf20dc4e28cb758731817fbf3f`; the downloaded `ftw-q-unet-efb3.onnx` was 51,037,786 bytes with SHA-256 `d14285f91870a79833aac3529650f385afec605ed415acdf2227a02a0a88d036`. The downloaded file is ignored under `models/public/`.

The graph has 439 nodes. Its operator inventory includes Conv, Sigmoid, Mul, ReduceMean, Add, Shape, Concat, Relu, Resize, and BatchNormalization. `scripts/doctor.py` rejected a 16-band 32×32 synthetic compatibility raster before inference, naming the first unsupported node: **node 5, BatchNormalization**. This is the expected honest result under the current [v1 support contract](v1-contract.md), but it means the project has not yet completed a real remote-sensing model/raster run. The synthetic raster did not use the model card's reflectance preprocessing and was not used to assess prediction quality.

The probe demonstrates a concrete support gap: useful external segmentation models can contain common operators and topology beyond the frozen engineering fixtures. Supporting this model would require validated operator semantics, shape/halo analysis, numerical comparisons, and memory planning for its actual graph. The model is now exposed to development and is ineligible as an unseen holdout.
