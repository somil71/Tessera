# Phase 3: split execution and memory characterization

Phase 3 is complete as a feasibility milestone: the Rust CLI can split a model at a chosen node, run the two models in sequence on a raw NCHW FP32 input, and report process memory and timing. It does not yet choose a cut, read GeoTIFF windows, spill tensors, or enforce a memory cap.

## Correctness

`tests/fixtures/smoke_rust_execution.py` runs the whole ONNX model and every internal cut of the three frozen fixtures. All 16 split runs passed ONNX Runtime CPU execution. Each split output had maximum absolute and relative error 0 against the whole-model Python ONNX Runtime reference on the smoke inputs. The test uses monotone inputs, which exposed an earlier input-aliasing bug; the runner now retains the original input while the head runs and materializes boundary tensors before releasing the head session.

## U-Net measurement

`tests/benchmark_phase3.py` generated a seeded 1×3×512×512 FP32 input, ran each configuration in a fresh process three times with four inference threads, alternated execution order, and wrote raw data to `runs/phase3_benchmark/report.json`. The input SHA-256 is `ae0d8587cdb71ec44be6ba5af36bda2ac1988b5f67fab9f67e69bb4afe52a076`; the model SHA-256 is `568d9d37b85aed0349221ab1efe13c9d03ae3ed40d6f94657233dc61fa79ccbe`. The test machine reported Windows `10.0.26300` and 16 logical CPUs.

| Execution | Boundary bytes | Median total time | Median peak committed | Median peak working set | Max error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole model | — | 120 ms | 55.07 MiB | 74.19 MiB | reference |
| Split after node 2 | 10 MiB | 133 ms | 63.00 MiB | 82.21 MiB | 0 |
| Split after node 4 | 12 MiB | 125 ms | 63.84 MiB | 71.90 MiB | 0 |
| Split after node 5 | 16 MiB | 129 ms | 69.05 MiB | 88.15 MiB | 0 |

The split runner loads the head, executes it, copies boundary tensors, releases the head session, then loads and executes the tail. Loading both sessions simultaneously previously raised the tested split peaks to about 85–98 MiB. Sequential loading reduced them, but every measured split still used more committed memory than the whole model. At cut 4, the boundary copy itself raised current committed memory from about 54 MiB after the head run to about 67 MiB. The measured result is a warning for the future planner: a small boundary estimate alone does not predict process peak memory.

The memory counters are process-wide. `peak_commit_bytes` is Windows `PROCESS_MEMORY_COUNTERS_EX.PeakPagefileUsage`; `peak_working_set_bytes` is `PeakWorkingSetSize`, obtained through `GetProcessMemoryInfo` ([Microsoft structure reference](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex), [function reference](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo)). Stage snapshots report current private committed bytes. They include the runtime, model state, Rust buffers, and I/O; they do not isolate ONNX Runtime allocations. These are development measurements, not the held-out feasibility result: the workload is small, the cuts are manual, the reported time excludes process startup and model splitting, and no Job Object limit was applied.

## Reproduce

```powershell
cargo build
.\.venv\Scripts\python.exe tests\fixtures\smoke_rust_execution.py
.\.venv\Scripts\python.exe tests\benchmark_phase3.py
```
