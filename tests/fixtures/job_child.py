"""Inspect inherited job settings and attempt one explicit memory commit."""

import ctypes
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from job_memory import winapi


input()
kernel, _ = winapi()
kernel.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.c_uint32, ctypes.c_uint32]
kernel.VirtualAlloc.restype = ctypes.c_void_p
kernel.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
kernel.VirtualFree.restype = ctypes.c_int
size = int(sys.argv[1])
address = kernel.VirtualAlloc(None, size, 0x1000 | 0x2000, 0x04)
print(json.dumps({
    "requested_bytes": size,
    "allocated": bool(address),
    "error": ctypes.get_last_error() if not address else None,
}))
if address:
    kernel.VirtualFree(address, 0, 0x8000)
