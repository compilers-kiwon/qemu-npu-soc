// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Compares the hand-written UAPI and C library definitions with the C
//! headers: tests/uapi.c, compiled with the host C compiler ($CC or cc)
//! against linux/include/uapi/drm/knpu_accel.h, prints sizes, field offsets
//! and constants. The ioctl encoding and these layouts are the same on
//! x86_64 and aarch64.

use std::collections::BTreeMap;
use std::mem::{offset_of, size_of};
use std::path::Path;
use std::process::Command;

use knpu::{sys, uapi};

fn c_values() -> BTreeMap<String, u64> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let uapi_dir = crate_dir.join("../../linux/include/uapi/drm");
    let exe = Path::new(env!("CARGO_TARGET_TMPDIR")).join("uapi-values");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let status = Command::new(&cc)
        .args(["-std=c11", "-Wall", "-Werror", "-o"])
        .arg(&exe)
        .arg("-I")
        .arg(&uapi_dir)
        // knpu_accel.h includes "drm.h" (linux-libc-dev on Debian/Ubuntu)
        .args(["-idirafter", "/usr/include/drm"])
        .arg(crate_dir.join("tests/uapi.c"))
        .status()
        .expect("cannot run the C compiler");
    assert!(status.success(), "{cc} could not compile tests/uapi.c");
    let out = Command::new(&exe).output().expect("cannot run uapi-values");
    assert!(out.status.success());
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| {
            let (key, value) = line.rsplit_once(' ').unwrap();
            (key.to_string(), value.parse().unwrap())
        })
        .collect()
}

#[test]
fn layout_and_numbers_match_the_c_headers() {
    macro_rules! rust {
        ($($key:expr => $value:expr),* $(,)?) => {
            BTreeMap::from([$(($key.to_string(), $value as u64)),*])
        };
    }
    use uapi::*;
    let rust = rust! {
        "size drm_gem_close" => size_of::<GemClose>(),
        "offset drm_gem_close.pad" => offset_of!(GemClose, pad),
        "size drm_knpu_npu_info" => size_of::<NpuInfo>(),
        "offset drm_knpu_npu_info.version" => offset_of!(NpuInfo, version),
        "offset drm_knpu_npu_info.config" => offset_of!(NpuInfo, config),
        "offset drm_knpu_npu_info.sram_size" => offset_of!(NpuInfo, sram_size),
        "size drm_knpu_query" => size_of::<Query>(),
        "offset drm_knpu_query.size" => offset_of!(Query, size),
        "offset drm_knpu_query.pointer" => offset_of!(Query, pointer),
        "size drm_knpu_bo_create" => size_of::<BoCreate>(),
        "offset drm_knpu_bo_create.flags" => offset_of!(BoCreate, flags),
        "offset drm_knpu_bo_create.handle" => offset_of!(BoCreate, handle),
        "size drm_knpu_bo_mmap_offset" => size_of::<BoMmapOffset>(),
        "offset drm_knpu_bo_mmap_offset.pad" => offset_of!(BoMmapOffset, pad),
        "offset drm_knpu_bo_mmap_offset.offset" => offset_of!(BoMmapOffset, offset),
        "size drm_knpu_bo_wait" => size_of::<BoWait>(),
        "offset drm_knpu_bo_wait.pad" => offset_of!(BoWait, pad),
        "offset drm_knpu_bo_wait.timeout_ns" => offset_of!(BoWait, timeout_ns),
        "size drm_knpu_cmdstream_create" => size_of::<CmdstreamCreate>(),
        "offset drm_knpu_cmdstream_create.flags" => offset_of!(CmdstreamCreate, flags),
        "offset drm_knpu_cmdstream_create.data" => offset_of!(CmdstreamCreate, data),
        "offset drm_knpu_cmdstream_create.handle" => offset_of!(CmdstreamCreate, handle),
        "offset drm_knpu_cmdstream_create.pad" => offset_of!(CmdstreamCreate, pad),
        "size drm_knpu_submit" => size_of::<Submit>(),
        "offset drm_knpu_submit.perfmon_id" => offset_of!(Submit, perfmon_id),
        "offset drm_knpu_submit.region_bo_handles" => offset_of!(Submit, region_bo_handles),
        "size drm_knpu_perfmon_create" => size_of::<PerfmonCreate>(),
        "offset drm_knpu_perfmon_create.pad" => offset_of!(PerfmonCreate, pad),
        "size drm_knpu_perfmon_destroy" => size_of::<PerfmonDestroy>(),
        "offset drm_knpu_perfmon_destroy.pad" => offset_of!(PerfmonDestroy, pad),
        "size drm_knpu_perfmon_get_values" => size_of::<PerfmonGetValues>(),
        "offset drm_knpu_perfmon_get_values.pad" => offset_of!(PerfmonGetValues, pad),
        "offset drm_knpu_perfmon_get_values.values_ptr" => offset_of!(PerfmonGetValues, values_ptr),
        "value DRM_IOCTL_GEM_CLOSE" => DRM_IOCTL_GEM_CLOSE,
        "value DRM_IOCTL_KNPU_QUERY" => DRM_IOCTL_KNPU_QUERY,
        "value DRM_IOCTL_KNPU_BO_CREATE" => DRM_IOCTL_KNPU_BO_CREATE,
        "value DRM_IOCTL_KNPU_BO_MMAP_OFFSET" => DRM_IOCTL_KNPU_BO_MMAP_OFFSET,
        "value DRM_IOCTL_KNPU_BO_WAIT" => DRM_IOCTL_KNPU_BO_WAIT,
        "value DRM_IOCTL_KNPU_CMDSTREAM_CREATE" => DRM_IOCTL_KNPU_CMDSTREAM_CREATE,
        "value DRM_IOCTL_KNPU_SUBMIT" => DRM_IOCTL_KNPU_SUBMIT,
        "value DRM_IOCTL_KNPU_PERFMON_CREATE" => DRM_IOCTL_KNPU_PERFMON_CREATE,
        "value DRM_IOCTL_KNPU_PERFMON_DESTROY" => DRM_IOCTL_KNPU_PERFMON_DESTROY,
        "value DRM_IOCTL_KNPU_PERFMON_GET_VALUES" => DRM_IOCTL_KNPU_PERFMON_GET_VALUES,
        "value DRM_KNPU_QUERY_NPU_INFO" => QUERY_NPU_INFO,
        "value DRM_KNPU_MAX_REGIONS" => MAX_REGIONS,
        "value DRM_KNPU_PERF_MACS" => PERF_MACS,
        "value DRM_KNPU_PERF_DMA_RD_BYTES" => PERF_DMA_RD_BYTES,
        "value DRM_KNPU_PERF_DMA_WR_BYTES" => PERF_DMA_WR_BYTES,
        "value DRM_KNPU_PERF_OPS" => PERF_OPS,
        "value DRM_KNPU_PERF_COUNTERS" => PERF_COUNTERS,
        "value PROT_READ" => sys::PROT_READ,
        "value PROT_WRITE" => sys::PROT_WRITE,
        "value MAP_SHARED" => sys::MAP_SHARED,
        "value CLOCK_MONOTONIC" => sys::CLOCK_MONOTONIC,
        "size timespec" => size_of::<sys::Timespec>(),
        "offset timespec.tv_nsec" => offset_of!(sys::Timespec, tv_nsec),
        "size off_t" => size_of::<sys::OffT>(),
    };
    let c = c_values();
    for (key, value) in &c {
        assert_eq!(rust.get(key), Some(value), "{key}: C {value}");
    }
    assert_eq!(rust.len(), c.len(), "Rust checks something the C side does not print");
}
