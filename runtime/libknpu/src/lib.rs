// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! libknpu: user-space runtime of the kn1-npu (docs/design.md 10).
//!
//! [`Model::parse`] reads and checks a model.knpu file
//! (docs/model-format.md), [`Device::open`] opens the knpu accel device,
//! [`Device::load`] creates the region buffers and the command stream of a
//! model ([`Device::load_with_cpu_code`], which is unsafe, also loads the
//! CPU code of a model that has it), and [`Loaded::run`] runs one
//! inference, the NPU job and then the CPU steps, and returns the output
//! with the job's performance counters.

#[doc(hidden)]
pub mod cpu;
mod device;
mod model;
#[doc(hidden)]
pub mod sys;
#[doc(hidden)]
pub mod uapi;

pub use device::{Counters, Device, DeviceInfo, Loaded, Output};
pub use model::{CpuStep, Direction, Model, Region, RegionRole, StepKind, Tensor};

use std::fmt;
use std::io;

/// Errors of the runtime.
#[derive(Debug)]
pub enum Error {
    /// The model file breaks a rule of docs/model-format.md, its CPU code
    /// included.
    Model(String),
    /// The model does not fit the device or the way it is loaded (CPU code
    /// needs Device::load_with_cpu_code), or the data does not fit the
    /// model.
    Mismatch(String),
    /// A system call failed.
    Io { what: String, source: io::Error },
    /// The device stopped with an error, or the driver timed the job out;
    /// the kernel log holds the error code and offset.
    JobFailed,
    /// The job did not finish before the runtime's deadline.
    Timeout,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Model(msg) => write!(f, "bad model file: {msg}"),
            Error::Mismatch(msg) => write!(f, "{msg}"),
            Error::Io { what, source } => write!(f, "{what}: {source}"),
            Error::JobFailed => {
                write!(f, "the NPU job failed (see the kernel log)")
            }
            Error::Timeout => write!(f, "the NPU job did not finish in time"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Result type of the runtime.
pub type Result<T> = std::result::Result<T, Error>;
