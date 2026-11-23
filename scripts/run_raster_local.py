"""Fixed aligned schedule for validated local NCHW FP32 ONNX graphs."""

from run_raster_unet import arguments, run


if __name__ == "__main__":
    args = arguments(__doc__)
    args.general_local = True
    run(args)
