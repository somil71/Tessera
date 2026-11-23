# Windows runtime resource accounting

The scheduler and its Rust children join one memory-limited Windows Job Object. The `job_peak_committed_bytes` and conservative upper bound remain the enforcement evidence. The conservative value adds scheduler commitment that existed before assignment. Successful automatic runs must keep it at or below the requested budget.

Current reports also include:

- `job_peak_working_set_sampled_bytes`: the largest observed sum of live job-process working sets, sampled every 50 ms and once at report generation. `working_set_sample_count` and `working_set_incomplete_samples` describe coverage. This is **not an exact peak**; short spikes can be missed.
- `job_io_read_transfer_bytes`, `job_io_write_transfer_bytes`, and `job_io_other_transfer_bytes`: cumulative Windows Job Object I/O transfer counters. They include processes that have exited. These are **not physical disk-only bytes**; model loads, raster I/O, pipes, and other I/O may contribute.
- The attention path separately reports the sizes of its disk-backed staging arrays and output. Other schedules still need a peak temporary-storage measurement.

These meanings follow Microsoft's [Job Object accounting](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_and_io_accounting_information), [I/O counters](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-io_counters), [process ID list](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_process_id_list), and [process memory counters](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo). The remaining v1 measurement gap is a defensible disk-only traffic estimate and exact or explicitly bounded working-set peak.
