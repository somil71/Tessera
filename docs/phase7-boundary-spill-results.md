# Graph-boundary disk spill experiment

The Rust `run-split` command can write its FP32 or INT64 graph-boundary tensors to temporary files, release the head ONNX Runtime session and input tensor, load the tail session, and then reload the boundary. It deletes those files on normal completion or an ordinary error. The command reports spill bytes and process memory. This experiment preceded [U-Net raster planner integration](phase8-storage-planner-results.md).

`tests/fixtures/smoke_rust_execution.py` exercised every internal cut of the development U-Net, global-mean, and exposed attention fixtures in both memory and disk modes. All 16 disk-spill outputs were identical to the in-memory split outputs and whole-model ONNX Runtime references on the smoke inputs. The test also checked that temporary directories were removed.

`tests/performance/benchmark_boundary_spill.py` compared three fresh-process runs per mode at cut 4 of the development U-Net. Each size used the same seeded input and four inference threads; run order alternated. Times include input read, model loads, boundary movement, inference, and output write inside Rust, but not process startup or model splitting. These are development measurements without a Job Object cap.

| Input | Mode | Median elapsed | Median peak committed | Median peak working set | Boundary written/read |
| --- | --- | ---: | ---: | ---: | ---: |
| 512×512 | RAM | 125 ms | 67.03 MB | 73.45 MB | 0 |
| 512×512 | Disk | 403 ms | 54.50 MB | 70.49 MB | 12.58 MB each way |
| 1024×1024 | RAM | 323 ms | 218.37 MB | 188.76 MB | 0 |
| 1024×1024 | Disk | 1410 ms | 168.03 MB | 166.36 MB | 50.33 MB each way |

Disk spill reduced measured process peak commitment by roughly 12.5 MB and 50.3 MB at the two sizes, while increasing elapsed time about 3.2× and 4.4×. This supports offering spill **only when RAM plans cannot satisfy the cap or when its measured total cost is competitive**. It does not establish a planner win. The current code still has to hold all boundary tensors in RAM during tail execution; disk spill removes their overlap with tail-session loading, not their eventual memory use. It does not implement recomputation. A hard process kill can leave its temporary files behind; clean cancellation and recovery remain to be designed.

```powershell
cargo build --locked
.\.venv\Scripts\python.exe tests\fixtures\smoke_rust_execution.py
.\.venv\Scripts\python.exe tests\performance\benchmark_boundary_spill.py --size 512 --cut-after 4 --repeats 3 --output-dir runs\boundary_spill_512
.\.venv\Scripts\python.exe tests\performance\benchmark_boundary_spill.py --size 1024 --cut-after 4 --repeats 3 --output-dir runs\boundary_spill_1024
```

The raw development reports live in the ignored `runs/` directory. The benchmark script and commands are versioned for reproduction.
