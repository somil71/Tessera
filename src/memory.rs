pub struct MemorySnapshot {
    pub commit_bytes: usize,
    pub working_set_bytes: usize,
    pub peak_commit_bytes: usize,
    pub peak_working_set_bytes: usize,
}

#[cfg(windows)]
pub fn process_snapshot() -> Result<MemorySnapshot, std::io::Error> {
    use std::mem::size_of;

    #[repr(C)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCountersEx,
            size: u32,
        ) -> i32;
    }

    let mut counters = ProcessMemoryCountersEx {
        cb: size_of::<ProcessMemoryCountersEx>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
        private_usage: 0,
    };
    // SAFETY: the pseudo handle identifies this process and counters points to a valid struct.
    let success = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            size_of::<ProcessMemoryCountersEx>() as u32,
        )
    };
    if success == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(MemorySnapshot {
        commit_bytes: counters.private_usage,
        working_set_bytes: counters.working_set_size,
        peak_commit_bytes: counters.peak_pagefile_usage,
        peak_working_set_bytes: counters.peak_working_set_size,
    })
}

#[cfg(not(windows))]
pub fn process_snapshot() -> Result<MemorySnapshot, std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "process memory peaks are available on Windows only",
    ))
}
