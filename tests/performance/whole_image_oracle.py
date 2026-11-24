"""Create and optionally verify an unsplit ONNX Runtime score-raster reference."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "64"

import numpy as np
import onnxruntime as ort
import rasterio


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--band-order", help="Comma-separated 1-based source band indexes")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    started = time.perf_counter()
    if args.threads < 1 or args.max_pixels < 1:
        raise ValueError("threads and max-pixels must be positive")
    if (not np.isfinite(args.scale) or not np.isfinite(args.offset)
            or not np.isfinite(args.absolute_tolerance)
            or not np.isfinite(args.relative_tolerance)
            or args.absolute_tolerance < 0 or args.relative_tolerance < 0):
        raise ValueError("preprocessing and tolerances must be finite and valid")
    model = args.model.resolve(strict=True)
    source = args.input.resolve(strict=True)
    reference = args.reference_output.resolve()
    report_path = args.report.resolve()
    candidate = args.candidate.resolve(strict=True) if args.candidate else None
    if len({model, source, reference, report_path, candidate} - {None}) != 4 + int(candidate is not None):
        raise ValueError("model, input, output, candidate, and report paths must differ")
    if reference.exists() or report_path.exists():
        raise FileExistsError("reference output and report must be new paths")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise ValueError("oracle requires one FP32 input and one FP32 output")
    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError("input raster requires a CRS")
        if src.width * src.height > args.max_pixels:
            raise ValueError("input exceeds max-pixels; oracle is for small reference crops")
        if src.nodata is not None:
            raise ValueError("declared no-data is unsupported")
        dims = inputs[0].shape
        if len(dims) != 4 or (isinstance(dims[0], int) and dims[0] != 1):
            raise ValueError("model input must be batch-one NCHW")
        if isinstance(dims[1], int) and dims[1] != src.count:
            raise ValueError("model input channel count differs from source raster")
        if any(isinstance(dim, int) and dim != size for dim, size in zip(dims[2:], (src.height, src.width))):
            raise ValueError("model spatial input shape differs from source raster")
        bands = (tuple(int(part) for part in args.band_order.split(","))
                 if args.band_order else tuple(range(1, src.count + 1)))
        if sorted(bands) != list(range(1, src.count + 1)):
            raise ValueError("band order must permute the source bands")
        data = src.read(indexes=bands, out_dtype="float32")
        if not np.isfinite(data).all() or not np.all(src.read_masks() == 255):
            raise ValueError("masked or non-finite source pixels are unsupported")
        data = np.asarray(data * args.scale + args.offset, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("preprocessing produced non-finite values")
        scores = session.run([outputs[0].name], {inputs[0].name: data[np.newaxis]})[0]
        if scores.dtype != np.float32 or scores.ndim != 4 or scores.shape[0] != 1 or scores.shape[2:] != data.shape[1:]:
            raise ValueError("model output must be batch-one FP32 CHW at input resolution")
        scores = scores[0]
        if not np.isfinite(scores).all():
            raise ValueError("model produced non-finite scores")
        profile = dict(driver="GTiff", width=src.width, height=src.height,
                       count=scores.shape[0], dtype="float32", crs=src.crs,
                       transform=src.transform, nodata=None, compress="deflate")
        tags = src.tags()
        crs = str(src.crs)
        transform = list(src.transform)
    comparison = None
    if candidate:
        with rasterio.open(candidate) as got:
            if (got.width, got.height, got.count, got.crs, got.transform) != (
                    profile["width"], profile["height"], profile["count"], profile["crs"], profile["transform"]):
                raise ValueError("candidate shape or georeferencing differs")
            if any(dtype != "float32" for dtype in got.dtypes):
                raise ValueError("candidate must contain FP32 scores")
            actual = got.read()
        if not np.isfinite(actual).all():
            raise ValueError("candidate has non-finite scores")
        delta = np.abs(actual.astype(np.float64) - scores.astype(np.float64))
        where = np.unravel_index(int(np.argmax(delta)), delta.shape)
        eligible = np.abs(scores) >= 1e-3
        max_abs = float(delta[where])
        max_rel = float(np.max(delta[eligible] / np.abs(scores[eligible]))) if np.any(eligible) else 0.0
        comparison = dict(candidate=str(candidate), candidate_sha256=digest(candidate),
                          max_absolute_error=max_abs, max_eligible_relative_error=max_rel,
                          max_absolute_error_at=[int(i) for i in where],
                          differing_values=int(np.count_nonzero(delta)),
                          absolute_tolerance=args.absolute_tolerance,
                          relative_tolerance=args.relative_tolerance,
                          passed=max_abs <= args.absolute_tolerance and max_rel <= args.relative_tolerance)
    reference.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference, "w", **profile) as dst:
        dst.write(scores)
        dst.update_tags(**tags)
        for band in range(1, scores.shape[0] + 1):
            dst.set_band_description(band, f"score_{band}")
    report = dict(schema="raster-compiler-whole-image-oracle-v1",
                  model=str(model), model_sha256=digest(model), input=str(source),
                  input_sha256=digest(source), reference=str(reference),
                  reference_sha256=digest(reference), shape=list(scores.shape),
                  crs=crs, transform=transform, band_order=list(bands),
                  scale=args.scale, offset=args.offset, threads=args.threads,
                  onnxruntime_version=ort.__version__, rasterio_version=rasterio.__version__,
                  comparison=comparison, passed=comparison is None or comparison["passed"],
                  elapsed_seconds=time.perf_counter() - started)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError("candidate scores exceeded numerical tolerances")


if __name__ == "__main__":
    try:
        run(arguments())
    except (ValueError, FileExistsError, AssertionError) as error:
        print(f"oracle failed: {error}", file=sys.stderr)
        sys.exit(1)
