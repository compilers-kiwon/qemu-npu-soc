// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The CPU code of a model (docs/model-format.md 9, 10; docs/design.md 10):
//! the shared object loaded from a memfd with dlopen, the symbols of the
//! CPU steps checked (docs/model-format.md 11, rule 7), and the steps run
//! on the arena after the NPU job.

use std::ffi::{CStr, CString, c_void};
use std::fs::File;
use std::io::{self, Write};
use std::mem::{self, MaybeUninit};
use std::os::fd::FromRawFd;
use std::ptr;

use crate::model::{CpuStep, StepKind};
use crate::{Error, Result, sys};

/// A CPU step with its symbol looked up.
enum Action {
    CopyIn { region: u32, offset: usize, size: usize, array: *mut u8 },
    Call(unsafe extern "C" fn()),
    CopyOut { region: u32, offset: usize, size: usize, array: *const u8 },
}

/// The shared object of a model, loaded, and its CPU steps.
pub struct CpuCode {
    handle: *mut c_void,
    actions: Vec<Action>,
    // The memfd stays open while the object is loaded. dlopen returns an
    // object already loaded under the same path name, and a closed fd's
    // number, so its /proc/self/fd path, goes to the next file opened: the
    // next model's code would come back as this one. Fields drop after
    // Drop::drop, so the memfd closes after dlclose.
    _memfd: File,
}

/// The message of this thread's last dl* error.
fn dl_error() -> String {
    // SAFETY: dlerror returns null or a NUL-terminated string that stays
    // valid until the next dl* call of this thread
    let msg = unsafe { sys::dlerror() };
    if msg.is_null() {
        return "unknown error".to_string();
    }
    // SAFETY: see above
    unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned()
}

fn io_error(what: &str) -> Error {
    Error::Io { what: what.to_string(), source: io::Error::last_os_error() }
}

impl CpuCode {
    /// Writes the shared object `code` to a memfd, dlopens it and looks up
    /// the symbol of every step: a call must name a function and a copy an
    /// object of the step's size (rule 7). [`crate::Model::parse`] has
    /// checked the steps themselves; this does not look at the ELF header.
    ///
    /// # Safety
    ///
    /// The object's initialisation routines run here, its functions on
    /// every [`CpuCode::run`] and its termination routines when it is
    /// dropped. Nothing checks what they do (docs/model-format.md 11), so
    /// the code must be trusted like an executable.
    pub unsafe fn load(code: &[u8], steps: &[CpuStep]) -> Result<CpuCode> {
        // SAFETY: the name is NUL-terminated
        let fd = unsafe { sys::memfd_create(c"knpu-cpu".as_ptr(), sys::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(io_error("memfd_create"));
        }
        // SAFETY: a new file descriptor that nothing else owns
        let mut memfd = unsafe { File::from_raw_fd(fd) };
        memfd.write_all(code).map_err(|e| Error::Io {
            what: "writing the CPU code to a memfd".to_string(),
            source: e,
        })?;
        let path = CString::new(format!("/proc/self/fd/{fd}")).unwrap();
        // SAFETY: the path is NUL-terminated; the caller trusts the code
        // whose initialisation routines run here
        let handle = unsafe { sys::dlopen(path.as_ptr(), sys::RTLD_NOW | sys::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(Error::Model(format!("cannot load the CPU code: {}", dl_error())));
        }
        // Dropping it on an error below closes the object
        let mut cpu = CpuCode { handle, actions: Vec::new(), _memfd: memfd };
        for (i, step) in steps.iter().enumerate() {
            let action = cpu.resolve(i, step)?;
            cpu.actions.push(action);
        }
        Ok(cpu)
    }

    /// Looks up the symbol of step `i` and checks it (rule 7).
    fn resolve(&self, i: usize, step: &CpuStep) -> Result<Action> {
        let bad = |what: &str| Error::Model(format!("CPU step {i}: {} {what}", step.name));
        // Model::parse checked that the name is a C identifier
        let name = CString::new(step.name.as_str()).unwrap();
        // SAFETY: the object is loaded and the name is NUL-terminated
        let addr = unsafe { sys::dlsym(self.handle, name.as_ptr()) };
        if addr.is_null() {
            return Err(bad("is not in the CPU code"));
        }
        let mut info = MaybeUninit::<sys::DlInfo>::uninit();
        let mut entry: *const sys::Elf64Sym = ptr::null();
        // SAFETY: info and entry are where dladdr1 writes what it finds
        let found = unsafe {
            sys::dladdr1(addr, info.as_mut_ptr(), (&raw mut entry).cast(), sys::RTLD_DL_SYMENT)
        };
        if found == 0 || entry.is_null() {
            return Err(bad("has no dynamic symbol"));
        }
        // SAFETY: dladdr1 filled info, and entry points into the symbol
        // table of the loaded object
        let (info, entry) = unsafe { (info.assume_init(), &*entry) };
        // dladdr1 reports a symbol at the address. If another name shares
        // it, the entry is not this symbol's and says nothing about it
        // SAFETY: dli_sname is null or a NUL-terminated name in the object
        let same = !info.dli_sname.is_null()
            && unsafe { CStr::from_ptr(info.dli_sname) } == name.as_c_str();
        if !same {
            return Err(bad("shares its address with another symbol"));
        }
        let typ = entry.st_info & 0xF;
        let (region, offset, size) = (step.region, step.offset as usize, step.size as usize);
        match step.kind {
            StepKind::Call if typ == sys::STT_FUNC => {
                // SAFETY: a function of the object; the CPU code's functions
                // are `void f(void)` (docs/model-format.md 9)
                Ok(Action::Call(unsafe {
                    mem::transmute::<*mut c_void, unsafe extern "C" fn()>(addr)
                }))
            }
            StepKind::Call => Err(bad("is not a function")),
            _ if typ != sys::STT_OBJECT || entry.st_size != u64::from(step.size) => {
                Err(bad(&format!("is not an object of {size} bytes")))
            }
            StepKind::CopyIn => Ok(Action::CopyIn { region, offset, size, array: addr.cast() }),
            StepKind::CopyOut => Ok(Action::CopyOut { region, offset, size, array: addr.cast() }),
        }
    }

    /// Runs the steps in order on the regions, given as (index, bytes).
    /// [`crate::Model::parse`] has checked that every copy is inside its
    /// region; a region the steps name must be here.
    pub fn run(&self, regions: &mut [(u32, &mut [u8])]) {
        fn bytes<'a>(regions: &'a mut [(u32, &mut [u8])], index: u32) -> &'a mut [u8] {
            let found = regions.iter_mut().find(|(i, _)| *i == index);
            found.expect("a region of the CPU steps").1
        }
        for action in &self.actions {
            match *action {
                Action::CopyIn { region, offset, size, array } => {
                    let from = &bytes(regions, region)[offset..offset + size];
                    // SAFETY: array is an object of `size` bytes (load
                    // checked its symbol), apart from the region
                    unsafe { ptr::copy_nonoverlapping(from.as_ptr(), array, size) }
                }
                // SAFETY: the caller of load trusts the code
                Action::Call(f) => unsafe { f() },
                Action::CopyOut { region, offset, size, array } => {
                    let to = &mut bytes(regions, region)[offset..offset + size];
                    // SAFETY: as for CopyIn
                    unsafe { ptr::copy_nonoverlapping(array, to.as_mut_ptr(), size) }
                }
            }
        }
    }
}

impl Drop for CpuCode {
    fn drop(&mut self) {
        // SAFETY: the handle came from dlopen, and the actions, which point
        // into the object, go with self
        unsafe { sys::dlclose(self.handle) };
    }
}
