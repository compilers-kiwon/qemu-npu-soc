// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The C library functions the runtime calls: those of the device (ioctl,
//! mmap, munmap, clock_gettime) and those that load the CPU code of a model
//! (memfd_create and the dlopen family, docs/design.md 10). The Rust
//! standard library already links the C library, so they are declared here
//! instead of coming from an external crate (docs/design.md D41). The
//! constants and types are those of glibc on 64-bit Linux (aarch64 and
//! x86_64); tests/uapi.rs checks them against the C headers.

use std::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void};

pub const PROT_READ: c_int = 0x1;
pub const PROT_WRITE: c_int = 0x2;
pub const MAP_SHARED: c_int = 0x01;
pub const CLOCK_MONOTONIC: c_int = 1;
pub const MFD_CLOEXEC: c_uint = 0x0001;
pub const RTLD_NOW: c_int = 0x00002;
pub const RTLD_LOCAL: c_int = 0;
/// What dladdr1 returns in `extra_info`: the symbol table entry.
pub const RTLD_DL_SYMENT: c_int = 1;
/// Symbol types, the low four bits of `st_info`.
pub const STT_OBJECT: u8 = 1;
pub const STT_FUNC: u8 = 2;

/// `struct timespec` (`time_t` and `long` are 64-bit).
#[repr(C)]
#[derive(Default)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: c_long,
}

/// `off_t` with 64-bit file offsets.
pub type OffT = i64;

/// `Dl_info`, what dladdr1 says about an address.
#[repr(C)]
pub struct DlInfo {
    pub dli_fname: *const c_char,
    pub dli_fbase: *mut c_void,
    pub dli_sname: *const c_char,
    pub dli_saddr: *mut c_void,
}

/// `Elf64_Sym`, an entry of a symbol table.
#[repr(C)]
pub struct Elf64Sym {
    pub st_name: u32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: u16,
    pub st_value: u64,
    pub st_size: u64,
}

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
    pub fn memfd_create(name: *const c_char, flags: c_uint) -> c_int;
    pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    pub fn dladdr1(
        addr: *const c_void,
        info: *mut DlInfo,
        extra_info: *mut *mut c_void,
        flags: c_int,
    ) -> c_int;
    pub fn dlerror() -> *mut c_char;
    pub fn dlclose(handle: *mut c_void) -> c_int;
}

/// What mmap returns on failure.
pub const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
