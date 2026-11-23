# Evaluation amendment 2: Windows peak counter interpretation

Recorded 2026-11-23, before selecting or running the replacement held-out model. The original [evaluation protocol](evaluation-protocol.md) requires a hard committed-memory cap and a recorded peak. The 128 MiB development storage search exposed a measurement ambiguity: the automatic run completed after eight failed probes, and `PeakJobMemoryUsed` reported 152,948,736 bytes against a configured 134,217,728-byte hard cap. A 50 ms sampled sum of live process private commitment peaked at 128,393,216 bytes. The selected raster output matched the alternate schedule exactly.

Microsoft documents that [`JOB_OBJECT_LIMIT_JOB_MEMORY`](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information) rejects an attempt to commit memory beyond the job-wide limit. It documents [`PeakJobMemoryUsed`](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information) as the job's peak memory usage, but does not specify whether a rejected commit can transiently raise that counter. The observed counter value alone therefore does not establish that a commit above the hard limit succeeded. The provisional-accounting explanation remains an inference, not a documented Windows guarantee.

For the final evaluation, record all of the following without clipping or discarding an overshooting counter:

1. The requested cap, the successfully configured Job Object limit, and successful job membership for scheduler and children. A missing cap or escaped process is a memory failure.
2. The raw `PeakJobMemoryUsed` value and whether it exceeds the requested cap.
3. The 50 ms sampled sum of current private commitment across live job processes, its sample count, and incomplete-sample count. This sample is not an exact historical peak.
4. Every failed probe and execution attempt. If the raw peak counter exceeds the cap without a recorded failed attempt, mark the memory measurement inconclusive for investigation.

A completed run with the hard limit active, sampled private commitment at or below the requested cap, and documented rejected probes can satisfy the memory criterion even if the raw peak counter overshoots. This clarifies evidence interpretation; it does not relax the cap, numerical tolerances, baseline allowance, or runtime thresholds. The replacement held-out evaluation remains required under [amendment 1](evaluation-amendment-1.md).
