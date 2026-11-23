# Windows job-wide committed-memory cap

Recorded 2026-11-23. Both fixed raster runners accept `--memory-limit-mib`. The Python scheduler joins a Windows Job Object before it launches any Rust model process. Every Rust split and inference process waits for a start byte while the scheduler verifies or assigns its job membership. The job applies `JOB_OBJECT_LIMIT_JOB_MEMORY`; the report includes the configured limit, measured `PeakJobMemoryUsed`, and a conservative upper bound that adds any scheduler commitment missing from the first job reading. Microsoft documents that [job memory limits make an over-limit commit fail](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information) and defines the [peak job counter](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information).

The explicit allocation test (`tests/fixtures/smoke_job_memory.py`) created a 48 MiB job. It verified child membership, accepted an 8 MiB `VirtualAlloc` commit, and rejected a 64 MiB commit with Windows error 1455. The peak job counter recorded an attempted high-water value above the limit after this failed request; a failed-run peak must therefore not be treated as proof that the allocation succeeded. Successful capped raster runs at 128 MiB stayed below the limit and matched whole-image FP32 references exactly in the pixelwise and U-Net smoke suites. A 64 MiB pixelwise run failed during worker startup, wrote a `.failure.json` record, and published no output raster.

The memory cap now covers the scheduler and its model processes on this Windows host. The fixed schedules still require user-selected tile size and graph cut, and the command accepts an omitted cap for development comparisons. The planner and final v1 CLI must require a budget, select a feasible schedule automatically, and report a job-wide peak working set plus disk and temporary-storage metrics. The current test uses small engineering models; it does not establish the 2 GiB held-out result.

Reproduce with:

```powershell
cargo build
.\.venv\Scripts\python.exe tests\fixtures\smoke_job_memory.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_raster.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_unet_raster.py
```
