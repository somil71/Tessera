# Persistent raster worker measurement

Recorded 2026-11-23. The fixed pixelwise GeoTIFF scheduler now defaults to a Rust worker that keeps both ONNX Runtime sessions loaded and exchanges raw FP32 tile bytes through stdin/stdout pipes. The older file-backed persistent worker and fresh-process-per-tile mode remain available for comparison. The pipe protocol uses a JSON request line followed by a fixed-size input payload, and a JSON response line followed by the output payload. The Python client reads until every announced byte arrives; a single pipe read is not assumed to return the full tile.

`tests/fixtures/smoke_worker.py` sent two different inputs through every internal cut of the U-Net, global-mean, and attention engineering fixtures using both file and pipe transports. All 32 worker configurations matched the whole-model reference within the frozen numerical tolerances. `tests/fixtures/smoke_raster.py` assembled a 37×53 GeoTIFF with 16-pixel and 11-pixel windows; both outputs matched the whole-image pointwise reference exactly and retained CRS, transform, and tested metadata. The pipe mode wrote zero raw tile bridge bytes.

`tests/benchmark_worker.py` compared three modes on the same 37×53, three-band GeoTIFF, a 16-pixel tile size, 12 windows, and four inference threads. It alternated mode order over three runs. The medians from `runs/worker_benchmark/report.json` were:

| Mode | Elapsed for full raster | Rust child peak committed | Python scheduler peak committed | Sum of separate peaks |
| --- | ---: | ---: | ---: | ---: |
| Fresh process per tile | 1.647 s | 16.95 MiB | 51.84 MiB | 68.80 MiB |
| Persistent sessions, raw files | 0.309 s | 18.15 MiB | 52.00 MiB | 70.20 MiB |
| Persistent sessions, pipes | 0.240 s | 18.22 MiB | 51.80 MiB | 70.05 MiB |

Pipe mode was about 6.9× faster than fresh-process mode on this small workload. Its Rust worker used about 1.3 MiB more committed memory because it kept sessions loaded. The last column adds the two processes' separate lifetime peaks; it is a conservative upper bound, not a measurement of their simultaneous sum and not a Windows Job Object reading. The numbers are small-fixture development measurements, not evidence that the 2 GiB product target is met.

The first combined-memory reading found that NumPy's default OpenBLAS thread pool made the Python scheduler commit over 500 MiB before raster work. The scheduler now sets `OPENBLAS_NUM_THREADS=1` before importing NumPy because window assembly does not use BLAS. A separate import check fell from about 523 MiB to about 18 MiB. This is a process-level memory fix, not a model execution optimization.

Reproduce with:

```powershell
cargo build
.\.venv\Scripts\python.exe tests\fixtures\smoke_worker.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_raster.py
.\.venv\Scripts\python.exe tests\benchmark_worker.py
```
