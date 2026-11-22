// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The knpu UAPI (linux/include/uapi/drm/knpu_accel.h) and the one generic
//! DRM ioctl the runtime needs (GEM_CLOSE), written by hand.
//! tests/uapi.rs compiles the C header and compares the sizes, field
//! offsets and ioctl numbers with these definitions.

use std::mem::size_of;

/// ioctl number encoding of asm-generic/ioctl.h (aarch64 and x86_64).
const IOC_WRITE: u64 = 1;
const IOC_READ: u64 = 2;

const fn ioc(dir: u64, nr: u32, size: usize) -> u64 {
    (dir << 30) | ((size as u64) << 16) | ((b'd' as u64) << 8) | nr as u64
}

/// DRM_IOWR(nr, type)
const fn drm_iowr(nr: u32, size: usize) -> u64 {
    ioc(IOC_READ | IOC_WRITE, nr, size)
}

/// DRM_IOW(nr, type)
const fn drm_iow(nr: u32, size: usize) -> u64 {
    ioc(IOC_WRITE, nr, size)
}

const DRM_COMMAND_BASE: u32 = 0x40;

/// `struct drm_gem_close` (drm.h)
#[repr(C)]
#[derive(Default)]
pub struct GemClose {
    pub handle: u32,
    pub pad: u32,
}

/// `struct drm_knpu_npu_info`
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct NpuInfo {
    pub id: u32,
    pub version: u32,
    pub config: u32,
    pub sram_size: u32,
}

/// `struct drm_knpu_query`
#[repr(C)]
#[derive(Default)]
pub struct Query {
    pub type_: u32,
    pub size: u32,
    pub pointer: u64,
}

/// `struct drm_knpu_bo_create`
#[repr(C)]
#[derive(Default)]
pub struct BoCreate {
    pub size: u64,
    pub flags: u32,
    pub handle: u32,
}

/// `struct drm_knpu_bo_mmap_offset`
#[repr(C)]
#[derive(Default)]
pub struct BoMmapOffset {
    pub handle: u32,
    pub pad: u32,
    pub offset: u64,
}

/// `struct drm_knpu_bo_wait`
#[repr(C)]
#[derive(Default)]
pub struct BoWait {
    pub handle: u32,
    pub pad: u32,
    pub timeout_ns: i64,
}

/// `struct drm_knpu_cmdstream_create`
#[repr(C)]
#[derive(Default)]
pub struct CmdstreamCreate {
    pub size: u32,
    pub flags: u32,
    pub data: u64,
    pub handle: u32,
    pub pad: u32,
}

pub const MAX_REGIONS: usize = 8;

/// `struct drm_knpu_submit`
#[repr(C)]
#[derive(Default)]
pub struct Submit {
    pub cmdstream: u32,
    pub perfmon_id: u32,
    pub region_bo_handles: [u32; MAX_REGIONS],
}

/// `struct drm_knpu_perfmon_create`
#[repr(C)]
#[derive(Default)]
pub struct PerfmonCreate {
    pub id: u32,
    pub pad: u32,
}

/// `struct drm_knpu_perfmon_destroy`
#[repr(C)]
#[derive(Default)]
pub struct PerfmonDestroy {
    pub id: u32,
    pub pad: u32,
}

/// `struct drm_knpu_perfmon_get_values`
#[repr(C)]
#[derive(Default)]
pub struct PerfmonGetValues {
    pub id: u32,
    pub pad: u32,
    pub values_ptr: u64,
}

pub const QUERY_NPU_INFO: u32 = 0;

pub const PERF_MACS: usize = 0;
pub const PERF_DMA_RD_BYTES: usize = 1;
pub const PERF_DMA_WR_BYTES: usize = 2;
pub const PERF_OPS: usize = 3;
pub const PERF_COUNTERS: usize = 4;

pub const fn version_major(version: u32) -> u32 {
    version >> 16
}
pub const fn version_minor(version: u32) -> u32 {
    version & 0xffff
}
pub const fn config_num_regions(config: u32) -> u32 {
    config & 0xf
}
pub const fn config_round_double(config: u32) -> bool {
    (config >> 8) & 1 != 0
}
pub const fn config_round_single(config: u32) -> bool {
    (config >> 9) & 1 != 0
}

pub const DRM_IOCTL_GEM_CLOSE: u64 = drm_iow(0x09, size_of::<GemClose>());

pub const DRM_IOCTL_KNPU_QUERY: u64 = drm_iowr(DRM_COMMAND_BASE, size_of::<Query>());
pub const DRM_IOCTL_KNPU_BO_CREATE: u64 =
    drm_iowr(DRM_COMMAND_BASE + 1, size_of::<BoCreate>());
pub const DRM_IOCTL_KNPU_BO_MMAP_OFFSET: u64 =
    drm_iowr(DRM_COMMAND_BASE + 2, size_of::<BoMmapOffset>());
pub const DRM_IOCTL_KNPU_BO_WAIT: u64 = drm_iowr(DRM_COMMAND_BASE + 3, size_of::<BoWait>());
pub const DRM_IOCTL_KNPU_CMDSTREAM_CREATE: u64 =
    drm_iowr(DRM_COMMAND_BASE + 4, size_of::<CmdstreamCreate>());
pub const DRM_IOCTL_KNPU_SUBMIT: u64 = drm_iowr(DRM_COMMAND_BASE + 5, size_of::<Submit>());
pub const DRM_IOCTL_KNPU_PERFMON_CREATE: u64 =
    drm_iowr(DRM_COMMAND_BASE + 6, size_of::<PerfmonCreate>());
pub const DRM_IOCTL_KNPU_PERFMON_DESTROY: u64 =
    drm_iowr(DRM_COMMAND_BASE + 7, size_of::<PerfmonDestroy>());
pub const DRM_IOCTL_KNPU_PERFMON_GET_VALUES: u64 =
    drm_iowr(DRM_COMMAND_BASE + 8, size_of::<PerfmonGetValues>());
