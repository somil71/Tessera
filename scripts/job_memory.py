"""Windows Job Object committed-memory cap for raster scheduler processes.

The caller joins before launching Rust children. Any existing commitment not
visible in the new job's first reading is reserved from the requested budget.
"""

import ctypes
import os
import threading


JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
JOB_OBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION_CLASS = 8
JOB_OBJECT_BASIC_PROCESS_ID_LIST_CLASS = 3
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
ERROR_INSUFFICIENT_BUFFER = 122


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


class BasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("total_user_time", ctypes.c_int64),
        ("total_kernel_time", ctypes.c_int64),
        ("this_period_total_user_time", ctypes.c_int64),
        ("this_period_total_kernel_time", ctypes.c_int64),
        ("total_page_fault_count", ctypes.c_uint32),
        ("total_processes", ctypes.c_uint32),
        ("active_processes", ctypes.c_uint32),
        ("total_terminated_processes", ctypes.c_uint32),
    ]


class BasicAndIoAccountingInformation(ctypes.Structure):
    _fields_ = [("basic", BasicAccountingInformation), ("io", IoCounters)]


class ProcessIdListHeader(ctypes.Structure):
    _fields_ = [("assigned", ctypes.c_uint32), ("returned", ctypes.c_uint32)]


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
    kernel.GetCurrentProcessId.restype = ctypes.c_uint32
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
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
        self.psapi = psapi
        self.working_set_sample_interval_seconds = 0.05
        self.peak_working_set_sampled_bytes = 0
        self.peak_private_commit_sampled_bytes = 0
        self.working_set_sample_count = 0
        self.working_set_incomplete_samples = 0
        self._sampler_stop = threading.Event()
        self._sampler = None
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
            self._sample_working_set()
            self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
            self._sampler.start()
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

    def io_transfer_bytes(self):
        accounting = BasicAndIoAccountingInformation()
        check(self.kernel.QueryInformationJobObject(
            self.handle, JOB_OBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION_CLASS,
            ctypes.byref(accounting), ctypes.sizeof(accounting), None),
            "QueryInformationJobObject(I/O accounting)")
        return dict(read=int(accounting.io.read_transfer_count),
                    write=int(accounting.io.write_transfer_count),
                    other=int(accounting.io.other_transfer_count))

    def _process_ids(self):
        capacity = 8
        while True:
            buffer = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
            ok = self.kernel.QueryInformationJobObject(
                self.handle, JOB_OBJECT_BASIC_PROCESS_ID_LIST_CLASS,
                buffer, ctypes.sizeof(buffer), None)
            if not ok:
                if ctypes.get_last_error() == ERROR_INSUFFICIENT_BUFFER:
                    capacity *= 2
                    continue
                check(ok, "QueryInformationJobObject(process IDs)")
            header = ProcessIdListHeader.from_buffer(buffer)
            if header.returned < header.assigned:
                capacity = max(capacity * 2, header.assigned)
                continue
            ids = (ctypes.c_size_t * capacity).from_buffer(buffer, 8)
            return [int(ids[index]) for index in range(header.returned)]

    def _sample_working_set(self):
        total_working_set = 0
        total_private_commit = 0
        incomplete = False
        current_pid = os.getpid()
        for pid in self._process_ids():
            current = pid == current_pid
            handle = (self.kernel.GetCurrentProcess() if current else
                      self.kernel.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid))
            if not handle:
                incomplete = True  # The child may have exited during enumeration.
                continue
            try:
                counters = ProcessMemoryCountersEx()
                counters.cb = ctypes.sizeof(counters)
                if self.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters),
                                                   counters.cb):
                    total_working_set += int(counters.working_set_size)
                    total_private_commit += int(counters.private_usage)
                else:
                    incomplete = True
            finally:
                if not current:
                    self.kernel.CloseHandle(handle)
        self.peak_working_set_sampled_bytes = max(self.peak_working_set_sampled_bytes,
                                                  total_working_set)
        self.peak_private_commit_sampled_bytes = max(
            self.peak_private_commit_sampled_bytes, total_private_commit)
        self.working_set_sample_count += 1
        self.working_set_incomplete_samples += int(incomplete)

    def _sample_loop(self):
        while not self._sampler_stop.wait(self.working_set_sample_interval_seconds):
            try:
                self._sample_working_set()
            except OSError:
                self.working_set_incomplete_samples += 1

    def resource_snapshot(self):
        self._sample_working_set()
        io = self.io_transfer_bytes()
        return dict(job_peak_working_set_sampled_bytes=self.peak_working_set_sampled_bytes,
                    job_peak_private_commit_sampled_bytes=(
                        self.peak_private_commit_sampled_bytes),
                    job_peak_counter_exceeds_budget=(
                        self.peak_job_bytes() > self.job_limit_bytes),
                    job_peak_conservative_exceeds_budget=(
                        self.conservative_peak_bytes() > self.budget_bytes),
                    job_enforced_limit_bytes=self.job_limit_bytes,
                    working_set_sample_interval_ms=int(
                        self.working_set_sample_interval_seconds * 1000),
                    working_set_sample_count=self.working_set_sample_count,
                    working_set_incomplete_samples=self.working_set_incomplete_samples,
                    job_io_read_transfer_bytes=io["read"],
                    job_io_write_transfer_bytes=io["write"],
                    job_io_other_transfer_bytes=io["other"])

    def close(self):
        if hasattr(self, "_sampler_stop"):
            self._sampler_stop.set()
            if self._sampler is not None:
                self._sampler.join(timeout=1)
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
