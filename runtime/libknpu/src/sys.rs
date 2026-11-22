// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The four C library functions the runtime calls. The Rust standard
//! library already links the C library, so they are declared here instead
//! of coming from an external crate (docs/design.md D41). The constants and
//! types are those of glibc on 64-bit Linux (aarch64 and x86_64);
//! tests/uapi.rs checks them against the C headers.

use std::ffi::{c_int, c_long, c_ulong, c_void};

pub const PROT_READ: c_int = 0x1;
pub const PROT_WRITE: c_int = 0x2;
pub const MAP_SHARED: c_int = 0x01;
pub const CLOCK_MONOTONIC: c_int = 1;

/// `struct timespec` (`time_t` and `long` are 64-bit).
#[repr(C)]
#[derive(Default)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: c_long,
}

/// `off_t` with 64-bit file offsets.
pub type OffT = i64;

unsafe extern "C" {
    pub fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    pub fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: OffT,
    ) -> *mut c_void;
    pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
    pub fn clock_gettime(clock: c_int, tp: *mut Timespec) -> c_int;
}

/// What mmap returns on failure.
pub const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
