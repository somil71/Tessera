"""Verify a nested Windows Job Object caps the scheduler and children."""

from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from job_memory import JobMemoryLimit


limit = 48 * 1024 * 1024
job = JobMemoryLimit(limit)
try:
    assert job.job_limit_bytes <= limit
    assert job.unaccounted_starting_bytes >= 0
    child = ROOT / "tests" / "fixtures" / "job_child.py"
    def attempt(bytes_to_allocate):
        process = subprocess.Popen(
            [sys.executable, child, str(bytes_to_allocate)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        inherited = job.contains_process(process)
        job.assign_process(process)
        stdout, stderr = process.communicate("go\n", timeout=10)
        assert process.returncode == 0, stderr
        return inherited, json.loads(stdout)

    inherited, small_state = attempt(8 * 1024 * 1024)
    assert small_state["allocated"] is True
    _, large_state = attempt(64 * 1024 * 1024)
    assert large_state["allocated"] is False, large_state
    resources = job.resource_snapshot()
    assert resources["job_peak_working_set_sampled_bytes"] > 0
    assert 0 < resources["job_peak_private_commit_sampled_bytes"] <= limit
    assert resources["working_set_sample_count"] >= 1
    assert resources["job_io_read_transfer_bytes"] >= 0
    assert resources["job_io_write_transfer_bytes"] >= 0
    print(f"job cap: {limit} bytes; child inherited: {inherited}; "
          f"8 MiB accepted, 64 MiB rejected (Win32 error {large_state['error']}); "
          f"sampled working set {resources['job_peak_working_set_sampled_bytes']} bytes")
finally:
    job.close()
