"""Windows Job Object committed-memory cap for raster scheduler processes.

The caller joins before launching Rust children. Any existing commitment not
visible in the new job's first reading is reserved from the requested budget.
"""

import ctypes


JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


class BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_int64),
        ("per_job_user_time_limit", ctypes.c_int64),
        ("limit_flags", ctypes.c_uint32),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.c_uint32),
        ("scheduling_class", ctypes.c_uint32),
    ]


class IoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operation_count", ctypes.c_uint64),
        ("write_operation_count", ctypes.c_uint64),
        ("other_operation_count", ctypes.c_uint64),
        ("read_transfer_count", ctypes.c_uint64),
        ("write_transfer_count", ctypes.c_uint64),
        ("other_transfer_count", ctypes.c_uint64),
    ]


class ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("basic_limit_information", BasicLimitInformation),
        ("io_info", IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


class ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("page_fault_count", ctypes.c_uint32),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
        ("private_usage", ctypes.c_size_t),
    ]


def winapi():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                ctypes.c_void_p, ctypes.c_uint32]
    kernel.SetInformationJobObject.restype = ctypes.c_int
    kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.AssignProcessToJobObject.restype = ctypes.c_int
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_int)]
    kernel.IsProcessInJob.restype = ctypes.c_int
    kernel.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                  ctypes.c_void_p, ctypes.c_uint32,
                                                  ctypes.c_void_p]
    kernel.QueryInformationJobObject.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_uint32]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    return kernel, psapi


def check(ok, operation):
    if not ok:
        raise OSError(ctypes.get_last_error(), f"{operation} failed")


class JobMemoryLimit:
    def __init__(self, budget_bytes):
        if budget_bytes <= 0:
            raise ValueError("memory budget must be positive")
        self.kernel, psapi = winapi()
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        check(psapi.GetProcessMemoryInfo(self.kernel.GetCurrentProcess(),
                                         ctypes.byref(counters), counters.cb),
              "GetProcessMemoryInfo")
        self.starting_private_bytes = int(counters.private_usage)
        if budget_bytes <= self.starting_private_bytes:
            raise ValueError(
                f"memory budget {budget_bytes} bytes is below scheduler's existing "
                f"commitment of {self.starting_private_bytes} bytes"
            )
        self.budget_bytes = budget_bytes
        self.handle = self.kernel.CreateJobObjectW(None, None)
        check(self.handle, "CreateJobObjectW")
        try:
            check(self.kernel.AssignProcessToJobObject(
                self.handle, self.kernel.GetCurrentProcess()), "AssignProcessToJobObject")
            self.initial_job_peak_bytes = self.peak_job_bytes()
            self.unaccounted_starting_bytes = max(
                0, self.starting_private_bytes - self.initial_job_peak_bytes
            )
            self.job_limit_bytes = budget_bytes - self.unaccounted_starting_bytes
            limits = ExtendedLimitInformation()
            limits.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_JOB_MEMORY
            limits.job_memory_limit = self.job_limit_bytes
            check(self.kernel.SetInformationJobObject(
                self.handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(limits), ctypes.sizeof(limits)), "SetInformationJobObject")
        except Exception:
            self.close()
            raise

    def peak_job_bytes(self):
        limits = ExtendedLimitInformation()
        check(self.kernel.QueryInformationJobObject(
            self.handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits), ctypes.sizeof(limits), None),
            "QueryInformationJobObject")
        return int(limits.peak_job_memory_used)

    def contains_process(self, process):
        result = ctypes.c_int()
        check(self.kernel.IsProcessInJob(process._handle, self.handle,
                                         ctypes.byref(result)), "IsProcessInJob")
        return bool(result.value)

    def assign_process(self, process):
        if not self.contains_process(process):
            check(self.kernel.AssignProcessToJobObject(self.handle, process._handle),
                  "AssignProcessToJobObject(child)")
        if not self.contains_process(process):
            raise RuntimeError("child process did not join the memory-limited job")

    def conservative_peak_bytes(self):
        return self.unaccounted_starting_bytes + self.peak_job_bytes()

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
