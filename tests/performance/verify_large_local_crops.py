"""Check deterministic large-raster crop interiors against unsplit local ONNX inference."""

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
import onnx
import onnxruntime as ort
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from local_graph import analyze_local, local_channels


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--band-order")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    started = time.perf_counter()
    if args.crop_size < 1 or args.threads < 1:
        raise ValueError("crop size and threads must be positive")
    if not all(np.isfinite(value) for value in (args.scale, args.offset,
                                                args.absolute_tolerance, args.relative_tolerance)):
        raise ValueError("preprocessing and tolerances must be finite")
    if args.absolute_tolerance < 0 or args.relative_tolerance < 0:
        raise ValueError("tolerances must be nonnegative")
    model_path = args.model.resolve(strict=True)
    source_path = args.input.resolve(strict=True)
    candidate_path = args.candidate.resolve(strict=True)
    report_path = args.report.resolve()
    if len({model_path, source_path, candidate_path, report_path}) != 4 or report_path.exists():
        raise ValueError("all paths must differ and the report must be new")
    model = onnx.load(model_path)
    plan = analyze_local(model)
    input_channels, output_channels = local_channels(model)
    if (args.crop_size % plan.alignment_y or args.crop_size % plan.alignment_x
            or args.crop_size <= 2*plan.halo_y or args.crop_size <= 2*plan.halo_x):
        raise ValueError("crop size must be aligned and larger than twice the derived halo")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("model must have one image input and one raster output")
    crops = []
    max_abs = max_rel = 0.0
    with rasterio.open(source_path) as source, rasterio.open(candidate_path) as candidate:
        if (source.width, source.height, source.crs, source.transform) != (
                candidate.width, candidate.height, candidate.crs, candidate.transform):
            raise ValueError("candidate georeferencing or dimensions differ")
        if source.crs is None or source.count != input_channels or candidate.count != output_channels:
            raise ValueError("source CRS or input/output band counts differ")
        if any(dtype != "float32" for dtype in candidate.dtypes):
            raise ValueError("candidate scores must be FP32")
        if source.width < args.crop_size or source.height < args.crop_size:
            raise ValueError("raster is smaller than one oracle crop")
        bands = (tuple(int(part) for part in args.band_order.split(","))
                 if args.band_order else tuple(range(1, input_channels+1)))
        if sorted(bands) != list(range(1, input_channels+1)):
            raise ValueError("band order must permute source bands")
        last_row = ((source.height-args.crop_size)//plan.alignment_y)*plan.alignment_y
        last_col = ((source.width-args.crop_size)//plan.alignment_x)*plan.alignment_x
        locations = list(dict.fromkeys(((0, 0),
                                       (((source.height-args.crop_size)//2//plan.alignment_y)*plan.alignment_y,
                                        ((source.width-args.crop_size)//2//plan.alignment_x)*plan.alignment_x),
                                       (last_row, last_col))))
        for row, col in locations:
            window = Window(col, row, args.crop_size, args.crop_size)
            image = source.read(indexes=bands, window=window, out_dtype="float32")
            if not np.all(source.read_masks(indexes=bands, window=window) == 255) or not np.isfinite(image).all():
                raise ValueError(f"crop at row {row}, col {col} has masked or non-finite input")
            image = np.asarray(image*args.scale+args.offset, dtype=np.float32)
            reference = session.run([outputs[0].name], {inputs[0].name: image[np.newaxis]})[0]
            if reference.shape != (1, output_channels, args.crop_size, args.crop_size):
                raise ValueError("whole-crop ONNX output has an unexpected shape")
            actual = candidate.read(window=window)
            interior = (slice(None), slice(plan.halo_y, args.crop_size-plan.halo_y),
                        slice(plan.halo_x, args.crop_size-plan.halo_x))
            expected = reference[0][interior]
            got = actual[interior]
            if not np.isfinite(expected).all() or not np.isfinite(got).all():
                raise ValueError(f"crop at row {row}, col {col} has non-finite scores")
            delta = np.abs(got.astype(np.float64)-expected.astype(np.float64))
            eligible = np.abs(expected) >= 1e-3
            crop_abs = float(delta.max())
            crop_rel = float(np.max(delta[eligible]/np.abs(expected[eligible]))) if np.any(eligible) else 0.0
            max_abs = max(max_abs, crop_abs)
            max_rel = max(max_rel, crop_rel)
            crops.append(dict(row=row, col=col, checked_shape=list(expected.shape),
                              max_absolute_error=crop_abs,
                              max_eligible_relative_error=crop_rel,
                              differing_values=int(np.count_nonzero(delta))))
    report = dict(schema="raster-compiler-large-local-crop-oracle-v1",
                  model=str(model_path), model_sha256=sha256(model_path),
                  input=str(source_path), input_sha256=sha256(source_path),
                  candidate=str(candidate_path), candidate_sha256=sha256(candidate_path),
                  crop_size=args.crop_size, halo=[plan.halo_y, plan.halo_x],
                  alignment=[plan.alignment_y, plan.alignment_x],
                  band_order=list(bands), scale=args.scale, offset=args.offset,
                  crops=crops, max_absolute_error=max_abs,
                  max_eligible_relative_error=max_rel,
                  absolute_tolerance=args.absolute_tolerance,
                  relative_tolerance=args.relative_tolerance,
                  passed=max_abs <= args.absolute_tolerance and max_rel <= args.relative_tolerance,
                  elapsed_seconds=time.perf_counter()-started)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError("crop interiors exceeded numerical tolerances")


if __name__ == "__main__":
    run(arguments())
