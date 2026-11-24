"""Check a Windows CPU installation and, optionally, a model/raster pair."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
PINS = {"numpy": "2.4.6", "onnx": "1.23.2",
        "onnxruntime": "1.31.0", "rasterio": "1.4.4"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_contract(value):
    tensor = value.type.tensor_type
    dims = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
            for dim in tensor.shape.dim]
    return {"name": value.name, "element_type": int(tensor.elem_type),
            "shape": dims}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--memory-limit-mib", type=int)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def check(args):
    errors = []
    checks = {}
    checks["platform"] = platform.platform()
    checks["python"] = sys.version.split()[0]
    checks["python_bits"] = struct.calcsize("P") * 8
    if os.name != "nt":
        errors.append("Windows is required for Job Object memory enforcement")
    if sys.version_info[:2] != (3, 11) or checks["python_bits"] != 64:
        errors.append("Python 3.11 x64 is required for the pinned environment")
    checks["packages"] = {}
    for name, expected in PINS.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        checks["packages"][name] = dict(installed=installed, required=expected)
        if installed != expected:
            errors.append(f"{name} must be version {expected}; found {installed}")
    exe = ROOT / "target" / "debug" / "raster-compiler.exe"
    dll = ROOT / ".venv" / "Lib" / "site-packages" / "onnxruntime" / "capi" / "onnxruntime.dll"
    checks["rust_executable"] = dict(path=str(exe), exists=exe.is_file())
    checks["onnxruntime_dll"] = dict(path=str(dll), exists=dll.is_file())
    checks["cargo"] = shutil.which("cargo")
    if not exe.is_file():
        errors.append("Rust executable is missing; run cargo build")
    if not dll.is_file():
        errors.append("ONNX Runtime DLL is missing from .venv; install requirements-dev.txt")
    if args.model is not None or args.input is not None:
        if args.memory_limit_mib is None:
            errors.append("--memory-limit-mib is required for a model/raster preflight")
        if args.model is None or args.input is None:
            errors.append("--model and --input must be supplied together")
        elif not args.model.is_file() or not args.input.is_file():
            errors.append("model and input must be existing files")
        elif all(checks["packages"][name]["installed"] == version
                 for name, version in PINS.items()):
            try:
                import onnx
                import rasterio
                from local_graph import validate_local_raster
                from run_raster_auto import model_family

                model_path = args.model.resolve(strict=True)
                source = args.input.resolve(strict=True)
                if args.output and args.output.resolve() in (model_path, source):
                    errors.append("output must differ from model and input")
                model = onnx.load(model_path)
                initializers = {item.name for item in model.graph.initializer}
                checks["model_graph"] = {
                    "path": str(model_path), "sha256": sha256(model_path),
                    "inputs": [tensor_contract(value) for value in model.graph.input
                               if value.name not in initializers],
                    "outputs": [tensor_contract(value) for value in model.graph.output],
                    "node_count": len(model.graph.node),
                    "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
                }
                with rasterio.open(source) as raster:
                    shape = [raster.height, raster.width]
                    raster_checks = dict(path=str(source), shape=shape,
                                         sha256=sha256(source), bands=raster.count,
                                         crs=str(raster.crs), dtype=list(raster.dtypes),
                                         nodata=raster.nodata)
                    checks["raster"] = raster_checks
                    if raster.crs is None:
                        errors.append("input GeoTIFF needs a CRS")
                family, channels, scores, _ = model_family(model_path, model)
                if raster_checks["bands"] != channels:
                    errors.append(f"model needs {channels} bands; raster has {raster_checks['bands']}")
                if family == "local_unet" and any(size % 2 for size in shape):
                    errors.append("supported U-Net requires even raster dimensions")
                if family == "local_graph":
                    validate_local_raster(model, *shape)
                checks["model"] = dict(path=str(model_path), family=family,
                                       input_bands=channels, score_bands=scores)
                checks["uncompressed_fp32_output_bytes"] = (
                    shape[0] * shape[1] * scores * 4
                )
            except Exception as exc:
                errors.append(f"model/raster preflight failed: {exc}")
    if args.memory_limit_mib is not None and args.memory_limit_mib < 1:
        errors.append("memory limit must be positive")
    target = (args.output or ROOT / "runs" / "scores.tif").resolve()
    ancestor = target.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    checks["output_volume"] = dict(path=str(ancestor),
                                   free_bytes=shutil.disk_usage(ancestor).free)
    checks["memory_limit_mib"] = args.memory_limit_mib
    return dict(status="ready" if not errors else "failed", checks=checks,
                errors=errors,
                note="Preflight does not prove a feasible schedule or numerical correctness")


def main():
    args = arguments()
    result = check(args)
    payload = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
