// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The knpu accel device: buffers, command streams, jobs and performance
//! monitors (linux/include/uapi/drm/knpu_accel.h), and the CPU steps of a
//! model after its jobs (cpu.rs).

use std::ffi::{c_ulong, c_void};
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::{ptr, slice};

use crate::cpu::CpuCode;
use crate::model::{Model, RegionRole, Tensor};
use crate::{Error, Result, sys, uapi};

/// "KNPU", the ID register (docs/npu-spec.md 3.1).
const KNPU_ID: u32 = 0x4B4E_5055;

/// How long the runtime waits for one job. The driver's own job timeout
/// (500 ms) ends a hung job well before.
const WAIT_NS: i64 = 10_000_000_000;

/// What `KNPU_QUERY` reports about the NPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceInfo {
    pub version_major: u32,
    pub version_minor: u32,
    pub num_regions: u32,
    pub round_double: bool,
    pub round_single: bool,
    pub sram_size: u32,
}

/// Performance counters of one job (docs/npu-spec.md 11).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    pub macs: u64,
    pub dma_rd_bytes: u64,
    pub dma_wr_bytes: u64,
    pub ops: u64,
}

/// Result of one inference.
#[derive(Debug, Clone)]
pub struct Output {
    pub data: Vec<u8>,
    pub counters: Counters,
}

/// An open knpu device (`/dev/accel/accelN`).
pub struct Device {
    file: File,
    info: DeviceInfo,
}

impl Device {
    /// Opens the device and reads its information.
    pub fn open(path: impl AsRef<Path>) -> Result<Device> {
        let path = path.as_ref();
        let file = OpenOptions::new().read(true).write(true).open(path).map_err(|e| {
            Error::Io { what: format!("cannot open {}", path.display()), source: e }
        })?;
        let mut dev = Device {
            file,
            info: DeviceInfo {
                version_major: 0,
                version_minor: 0,
                num_regions: 0,
                round_double: false,
                round_single: false,
                sram_size: 0,
            },
        };
        let mut npu = uapi::NpuInfo::default();
        let mut q = uapi::Query {
            type_: uapi::QUERY_NPU_INFO,
            size: size_of::<uapi::NpuInfo>() as u32,
            pointer: &mut npu as *mut _ as u64,
        };
        dev.ioctl(uapi::DRM_IOCTL_KNPU_QUERY, &mut q, "KNPU_QUERY")?;
        if npu.id != KNPU_ID {
            return Err(Error::Mismatch(format!(
                "{} is not a kn1-npu (ID {:#x})",
                path.display(),
                npu.id
            )));
        }
        dev.info = DeviceInfo {
            version_major: uapi::version_major(npu.version),
            version_minor: uapi::version_minor(npu.version),
            num_regions: uapi::config_num_regions(npu.config),
            round_double: uapi::config_round_double(npu.config),
            round_single: uapi::config_round_single(npu.config),
            sram_size: npu.sram_size,
        };
        Ok(dev)
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Creates the region buffers (constant regions filled in) and the
    /// command stream of a model without CPU code. The kernel validates the
    /// stream here. A model with CPU sections needs
    /// [`Device::load_with_cpu_code`].
    pub fn load(&self, model: &Model) -> Result<Loaded<'_>> {
        if model.cpu_code.is_some() {
            return Err(Error::Mismatch(
                "the model has CPU code, which runs in this process; \
                 load it with Device::load_with_cpu_code"
                    .to_string(),
            ));
        }
        // SAFETY: the model has no CPU code
        unsafe { self.load_with_cpu_code(model) }
    }

    /// Like [`Device::load`], and also loads the CPU code of the model, if
    /// it has any (docs/model-format.md 12, step 2): its CPU steps then run
    /// after every NPU job.
    ///
    /// # Safety
    ///
    /// The CPU code runs in this process: its initialisation routines here,
    /// its layer functions on every [`Loaded::run`]. Nothing checks what it
    /// does (docs/model-format.md 11), so the model file must be trusted
    /// like an executable.
    pub unsafe fn load_with_cpu_code(&self, model: &Model) -> Result<Loaded<'_>> {
        model.check_device(&self.info)?;
        let input = model.input()?.clone();
        let output = model.output()?.clone();

        let mut regions = Vec::new();
        for r in &model.regions {
            let buf = self.create_buffer(r.size as u64)?;
            if let (RegionRole::Const, Some(content)) = (r.role, &r.content) {
                buf.write(0, content);
            }
            regions.push((r.index, buf));
        }

        let cpu = match &model.cpu_code {
            // SAFETY: the caller trusts the CPU code
            Some(code) => Some(unsafe { CpuCode::load(code, &model.cpu_steps) }?),
            None => None,
        };

        let mut args = uapi::CmdstreamCreate {
            size: model.cmds.len() as u32,
            data: model.cmds.as_ptr() as u64,
            ..Default::default()
        };
        self.ioctl(uapi::DRM_IOCTL_KNPU_CMDSTREAM_CREATE, &mut args, "KNPU_CMDSTREAM_CREATE")?;
        let cmdstream = Handle { dev: self, handle: args.handle };

        Ok(Loaded { dev: self, regions, cmdstream, input, output, cpu })
    }

    fn ioctl<T>(&self, request: u64, arg: &mut T, what: &str) -> Result<()> {
        loop {
            // SAFETY: arg is the structure this request takes, and every
            // pointer inside it points to memory that lives through the call
            let r = unsafe {
                sys::ioctl(self.file.as_raw_fd(), request as c_ulong, arg as *mut T)
            };
            if r == 0 {
                return Ok(());
            }
            let e = io::Error::last_os_error();
            // Like libdrm's drmIoctl(), retry interrupted calls
            if matches!(e.raw_os_error(), Some(libc_errno::EINTR | libc_errno::EAGAIN)) {
                continue;
            }
            return Err(Error::Io { what: what.to_string(), source: e });
        }
    }

    fn create_buffer(&self, size: u64) -> Result<Buffer<'_>> {
        let mut create = uapi::BoCreate { size, ..Default::default() };
        self.ioctl(uapi::DRM_IOCTL_KNPU_BO_CREATE, &mut create, "KNPU_BO_CREATE")?;
        let handle = Handle { dev: self, handle: create.handle };
        let mut off = uapi::BoMmapOffset { handle: create.handle, ..Default::default() };
        self.ioctl(uapi::DRM_IOCTL_KNPU_BO_MMAP_OFFSET, &mut off, "KNPU_BO_MMAP_OFFSET")?;
        let len = create.size as usize;
        // SAFETY: a fresh shared mapping of the buffer through the device
        let ptr = unsafe {
            sys::mmap(
                ptr::null_mut(),
                len,
                sys::PROT_READ | sys::PROT_WRITE,
                sys::MAP_SHARED,
                self.file.as_raw_fd(),
                off.offset as sys::OffT,
            )
        };
        if ptr == sys::MAP_FAILED {
            return Err(Error::Io {
                what: "mmap of a buffer".to_string(),
                source: io::Error::last_os_error(),
            });
        }
        Ok(Buffer { handle, ptr: ptr.cast(), len })
    }
}

/// errno values the runtime tells apart (asm-generic/errno-base.h,
/// asm-generic/errno.h; the same on aarch64 and x86_64).
mod libc_errno {
    pub const EINTR: i32 = 4;
    pub const EIO: i32 = 5;
    pub const EAGAIN: i32 = 11;
    pub const ETIME: i32 = 62;
}

/// A GEM handle, closed when dropped.
struct Handle<'d> {
    dev: &'d Device,
    handle: u32,
}

impl Drop for Handle<'_> {
    fn drop(&mut self) {
        let mut args = uapi::GemClose { handle: self.handle, pad: 0 };
        let _ = self.dev.ioctl(uapi::DRM_IOCTL_GEM_CLOSE, &mut args, "GEM_CLOSE");
    }
}

/// A buffer object mapped into this process.
struct Buffer<'d> {
    handle: Handle<'d>,
    ptr: *mut u8,
    len: usize,
}

impl Buffer<'_> {
    fn write(&self, offset: usize, data: &[u8]) {
        assert!(offset + data.len() <= self.len);
        // SAFETY: the range is inside the mapping
        unsafe { ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len()) }
    }

    fn read(&self, offset: usize, len: usize) -> Vec<u8> {
        assert!(offset + len <= self.len);
        let mut out = vec![0; len];
        // SAFETY: the range is inside the mapping
        unsafe { ptr::copy_nonoverlapping(self.ptr.add(offset), out.as_mut_ptr(), len) }
        out
    }

    /// The whole mapping, for the CPU steps; only between jobs, when the
    /// device does not use the buffer.
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is len bytes and lives as long as self, and
        // &mut self keeps out every other access from this process
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        // SAFETY: the mapping was made by create_buffer with this length;
        // the handle is closed after the unmapping
        unsafe { sys::munmap(self.ptr.cast::<c_void>(), self.len) };
    }
}

/// A model loaded on a device, ready to run.
pub struct Loaded<'d> {
    dev: &'d Device,
    regions: Vec<(u32, Buffer<'d>)>,
    cmdstream: Handle<'d>,
    input: Tensor,
    output: Tensor,
    /// The CPU code and steps (Device::load_with_cpu_code).
    cpu: Option<CpuCode>,
}

impl Loaded<'_> {
    /// Runs one inference: copies the input into the arena, submits the job
    /// with a performance monitor, waits, runs the CPU steps if the model
    /// has any and returns the output.
    pub fn run(&mut self, input: &[u8]) -> Result<Output> {
        if input.len() != self.input.size as usize {
            return Err(Error::Mismatch(format!(
                "the input has {} bytes, the model takes {}",
                input.len(),
                self.input.size
            )));
        }
        self.buffer(self.input.region).write(self.input.offset as usize, input);

        let dev = self.dev;
        let mut create = uapi::PerfmonCreate::default();
        dev.ioctl(uapi::DRM_IOCTL_KNPU_PERFMON_CREATE, &mut create, "KNPU_PERFMON_CREATE")?;
        let perfmon = Perfmon { dev, id: create.id };

        let mut submit = uapi::Submit {
            cmdstream: self.cmdstream.handle,
            perfmon_id: perfmon.id,
            ..Default::default()
        };
        for (index, buf) in &self.regions {
            submit.region_bo_handles[*index as usize] = buf.handle.handle;
        }
        dev.ioctl(uapi::DRM_IOCTL_KNPU_SUBMIT, &mut submit, "KNPU_SUBMIT")?;

        let mut wait = uapi::BoWait {
            handle: self.buffer(self.output.region).handle.handle,
            timeout_ns: monotonic_ns()? + WAIT_NS,
            ..Default::default()
        };
        match dev.ioctl(uapi::DRM_IOCTL_KNPU_BO_WAIT, &mut wait, "KNPU_BO_WAIT") {
            Err(Error::Io { source, .. }) if source.raw_os_error() == Some(libc_errno::EIO) => {
                return Err(Error::JobFailed);
            }
            Err(Error::Io { source, .. }) if source.raw_os_error() == Some(libc_errno::ETIME) => {
                return Err(Error::Timeout);
            }
            other => other?,
        }

        let mut values = [0u64; uapi::PERF_COUNTERS];
        let mut get = uapi::PerfmonGetValues {
            id: perfmon.id,
            values_ptr: values.as_mut_ptr() as u64,
            ..Default::default()
        };
        dev.ioctl(uapi::DRM_IOCTL_KNPU_PERFMON_GET_VALUES, &mut get, "KNPU_PERFMON_GET_VALUES")?;

        // The CPU steps, after the job (docs/model-format.md 12, step 5)
        if let Some(cpu) = &self.cpu {
            let mut regions: Vec<(u32, &mut [u8])> =
                self.regions.iter_mut().map(|(index, buf)| (*index, buf.bytes_mut())).collect();
            cpu.run(&mut regions);
        }

        let out = self.buffer(self.output.region);
        Ok(Output {
            data: out.read(self.output.offset as usize, self.output.size as usize),
            counters: Counters {
                macs: values[uapi::PERF_MACS],
                dma_rd_bytes: values[uapi::PERF_DMA_RD_BYTES],
                dma_wr_bytes: values[uapi::PERF_DMA_WR_BYTES],
                ops: values[uapi::PERF_OPS],
            },
        })
    }

    fn buffer(&self, region: u32) -> &Buffer<'_> {
        // Model::parse checked that every tensor is in a region
        &self.regions.iter().find(|(index, _)| *index == region).unwrap().1
    }
}

/// A performance monitor, destroyed when dropped.
struct Perfmon<'d> {
    dev: &'d Device,
    id: u32,
}

impl Drop for Perfmon<'_> {
    fn drop(&mut self) {
        let mut args = uapi::PerfmonDestroy { id: self.id, pad: 0 };
        let request = uapi::DRM_IOCTL_KNPU_PERFMON_DESTROY;
        let _ = self.dev.ioctl(request, &mut args, "KNPU_PERFMON_DESTROY");
    }
}

/// CLOCK_MONOTONIC in nanoseconds, the clock of KNPU_BO_WAIT's deadline.
fn monotonic_ns() -> Result<i64> {
    let mut ts = sys::Timespec::default();
    // SAFETY: ts is a valid timespec
    if unsafe { sys::clock_gettime(sys::CLOCK_MONOTONIC, &mut ts) } != 0 {
        return Err(Error::Io {
            what: "clock_gettime".to_string(),
            source: io::Error::last_os_error(),
        });
    }
    Ok(ts.tv_sec * 1_000_000_000 + ts.tv_nsec as i64)
}

