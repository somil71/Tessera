"""Compare two georeferenced FP32 score rasters in bounded windows."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "64"

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from score_compare import compare_scores


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def run(args):
    report = compare_scores(args.reference, args.candidate,
                            window_size=args.window_size,
                            absolute_tolerance=args.absolute_tolerance,
                            relative_tolerance=args.relative_tolerance)
    args.report.resolve().write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError("raster scores exceeded the numerical tolerances")


if __name__ == "__main__":
    run(arguments())
