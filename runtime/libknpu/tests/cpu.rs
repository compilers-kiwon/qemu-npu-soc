// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! The CPU code loader (src/cpu.rs) with shared objects for the host:
//! tests/cpu_code.c, compiled with the host C compiler ($CC or cc) and
//! linked like the CPU code of a model (-Bsymbolic), stands in for it.
//! Model::parse takes AArch64 code only, so these tests call the loader
//! directly; the AArch64 code itself runs in V3 and V6 (docs/design.md 10).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard};

use knpu::cpu::CpuCode;
use knpu::{CpuStep, Error, StepKind};

/// One test at a time: a test counts the open memfds, and the shared
/// objects are built into the same files.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// tests/cpu_code.c built as a shared object whose layer8 adds `add`.
fn shared_object(add: i32) -> Vec<u8> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let so = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("cpu_code_{add}.so"));
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let status = Command::new(&cc)
        .args(["-std=c89", "-Wall", "-Werror", "-O2", "-fPIC", "-shared", "-Wl,-Bsymbolic"])
        .arg(format!("-DADD={add}"))
        .arg("-o")
        .arg(&so)
        .arg(crate_dir.join("tests/cpu_code.c"))
        .status()
        .expect("cannot run the C compiler");
    assert!(status.success(), "{cc} could not compile tests/cpu_code.c");
    std::fs::read(&so).unwrap()
}

/// A step; copies are in region 1.
fn step(kind: StepKind, name: &str, offset: u32, size: u32) -> CpuStep {
    let region = if kind == StepKind::Call { 0 } else { 1 };
    CpuStep { kind, region, name: name.to_string(), offset, size }
}

/// The steps of the MNIST model's softmax layer: ten bytes at 0 into act7,
/// layer8, act8 to 16.
fn steps() -> Vec<CpuStep> {
    vec![
        step(StepKind::CopyIn, "act7", 0, 10),
        step(StepKind::Call, "layer8", 0, 0),
        step(StepKind::CopyOut, "act8", 16, 10),
    ]
}

fn load(code: &[u8], steps: &[CpuStep]) -> knpu::Result<CpuCode> {
    // SAFETY: tests/cpu_code.c, or bytes that dlopen refuses
    unsafe { CpuCode::load(code, steps) }
}

/// Runs the steps on a 32-byte region 1 that starts with `input` and
/// returns the region.
fn run(code: &CpuCode, input: &[i8]) -> Vec<i8> {
    let mut arena = [0u8; 32];
    for (byte, &value) in arena.iter_mut().zip(input) {
        *byte = value as u8;
    }
    code.run(&mut [(1, &mut arena[..])]);
    arena.iter().map(|&b| b as i8).collect()
}

/// The memfds of CPU code open in this process.
fn memfds() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .unwrap()
        .filter_map(|entry| std::fs::read_link(entry.ok()?.path()).ok())
        .filter(|target| target.to_string_lossy().starts_with("/memfd:knpu-cpu"))
        .count()
}

#[test]
fn runs_the_steps() {
    let _serial = serial();
    let code = load(&shared_object(3), &steps()).unwrap();
    let input: Vec<i8> = (-5..5).collect();
    let arena = run(&code, &input);
    assert_eq!(arena[16..26], input.iter().map(|v| v + 3).collect::<Vec<_>>()[..]);
    // Only the copy out writes to the region
    assert_eq!(arena[..10], input[..]);
    assert!(arena[10..16].iter().chain(&arena[26..]).all(|&b| b == 0));
    // Every run copies its input in
    assert_eq!(run(&code, &[100; 10])[16..26], [103; 10]);
}

#[test]
fn keeps_loaded_models_apart() {
    let _serial = serial();
    // Each memfd stays open while its object is loaded, so the second does
    // not get the /proc/self/fd path of the first, for which dlopen would
    // return the first object
    let one = load(&shared_object(1), &steps()).unwrap();
    let two = load(&shared_object(2), &steps()).unwrap();
    assert_eq!(run(&one, &[0; 10])[16], 1);
    assert_eq!(run(&two, &[0; 10])[16], 2);
    // Once one is unloaded, its path is free for the next
    drop(one);
    let three = load(&shared_object(3), &steps()).unwrap();
    assert_eq!(run(&three, &[0; 10])[16], 3);
    assert_eq!(run(&two, &[0; 10])[16], 2);
}

#[test]
fn checks_the_symbols() {
    let _serial = serial();
    let so = shared_object(1);
    let cases = [
        ("no such symbol", step(StepKind::CopyIn, "act9", 0, 10)),
        ("call of an array", step(StepKind::Call, "act7", 0, 0)),
        ("copy of a function", step(StepKind::CopyIn, "layer8", 0, 10)),
        ("copy in of another size", step(StepKind::CopyIn, "act7", 0, 9)),
        ("copy out of another size", step(StepKind::CopyOut, "act8", 16, 11)),
    ];
    for (what, bad) in cases {
        match load(&so, &[bad]) {
            Err(Error::Model(msg)) => println!("{what}: {msg}"),
            Err(e) => panic!("{what}: wrong error: {e}"),
            Ok(_) => panic!("{what}: accepted"),
        }
    }
}

#[test]
fn rejects_code_that_does_not_load() {
    let _serial = serial();
    let so = shared_object(1);
    for (what, code) in [("not ELF", &b"not a shared object"[..]), ("ELF header only", &so[..64])] {
        match load(code, &steps()) {
            Err(Error::Model(msg)) if msg.starts_with("cannot load the CPU code") => {
                println!("{what}: {msg}")
            }
            Err(e) => panic!("{what}: wrong error: {e}"),
            Ok(_) => panic!("{what}: accepted"),
        }
    }
}

#[test]
fn closes_the_memfd() {
    let _serial = serial();
    let so = shared_object(1);
    let before = memfds();
    let code = load(&so, &steps()).unwrap();
    assert_eq!(memfds(), before + 1);
    drop(code);
    assert_eq!(memfds(), before);
    // Also when the code does not load or a step does not check out
    assert!(load(b"not a shared object", &steps()).is_err());
    assert!(load(&so, &[step(StepKind::Call, "act7", 0, 0)]).is_err());
    assert_eq!(memfds(), before);
}
