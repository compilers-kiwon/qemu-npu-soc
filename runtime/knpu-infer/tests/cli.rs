// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! knpu-infer's command line and the checks it makes before it opens the
//! device (the device itself is exercised by V6 in the guest).

use std::path::PathBuf;
use std::process::{Command, Output};

fn vectors() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/conv/vectors")
}

fn tmp(name: &str, data: &[u8]) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, data).unwrap();
    path
}

fn knpu_infer(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_knpu-infer")).args(args).output().unwrap()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn u32_at(d: &[u8], at: usize) -> usize {
    u32::from_le_bytes(d[at..at + 4].try_into().unwrap()) as usize
}

fn put32(d: &mut [u8], at: usize, v: usize) {
    d[at..at + 4].copy_from_slice(&(v as u32).to_le_bytes());
}

/// The conv test model as format minor 1 with CPU sections
/// (docs/model-format.md 9, 10): CPU_CODE is just an ELF header for
/// `machine`, the part Model::parse reads, and CPU_STEPS has three steps.
fn model_with_cpu_code(machine: u8) -> Vec<u8> {
    let conv = std::fs::read(vectors().join("conv-double-64k.knpu")).unwrap();
    let table = u32_at(&conv, 16);
    let mut sections: Vec<(usize, Vec<u8>)> = (0..usize::from(conv[14]))
        .map(|i| table + 16 * i)
        .map(|at| {
            let (off, size) = (u32_at(&conv, at + 4), u32_at(&conv, at + 8));
            (u32_at(&conv, at), conv[off..off + size].to_vec())
        })
        .collect();
    let mut elf = vec![0; 64];
    elf[..7].copy_from_slice(b"\x7fELF\x02\x01\x01"); // 64-bit, little-endian
    elf[16..20].copy_from_slice(&[3, 0, machine, 0]); // ET_DYN
    // COPY_IN act7, CALL layer8 and COPY_OUT act8 as (kind, region, where
    // the name is, offset, size), then the names
    let mut steps = vec![0; 16];
    put32(&mut steps, 0, 3);
    for (kind, region, name, offset, size) in
        [(0, 1, 0, 0, 10), (1, 0, 5, 0, 0), (2, 1, 12, 1568, 10)]
    {
        let at = steps.len();
        steps.extend_from_slice(&[kind, region]);
        steps.resize(at + 16, 0);
        put32(&mut steps, at + 4, name);
        put32(&mut steps, at + 8, offset);
        put32(&mut steps, at + 12, size);
    }
    steps.extend_from_slice(b"act7\0layer8\0act8\0");
    sections.extend([(5, elf), (6, steps)]);

    // The header with minor 1, the table after it, the sections 16-byte
    // aligned
    let mut d = conv[..32].to_vec();
    d[6] = 1;
    d[14] = sections.len() as u8;
    put32(&mut d, 16, 32);
    d.resize(32 + 16 * sections.len(), 0);
    for (i, (kind, body)) in sections.iter().enumerate() {
        d.resize(d.len().next_multiple_of(16), 0);
        let off = d.len();
        put32(&mut d, 32 + 16 * i, *kind);
        put32(&mut d, 32 + 16 * i + 4, off);
        put32(&mut d, 32 + 16 * i + 8, body.len());
        d.extend_from_slice(body);
    }
    d
}

#[test]
fn usage() {
    let out = knpu_infer(&[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("usage: knpu-infer"));
    let out = knpu_infer(&["--help".as_ref()]);
    assert_eq!(out.status.code(), Some(0));
    let out = knpu_infer(&["--bogus".as_ref(), "a".as_ref(), "b".as_ref()]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("unknown option --bogus"));
}

#[test]
fn checks_before_the_device() {
    let model = vectors().join("conv-double-64k.knpu");
    let input = vectors().join("input-0.bin");
    let expect = vectors().join("expected-double-0.bin");
    let data = std::fs::read(&model).unwrap();
    let truncated = tmp("truncated.knpu", &data[..data.len() / 2]);
    let short = tmp("short.bin", &[0; 100]);
    let missing = PathBuf::from("/nonexistent/accel0");

    // A truncated model file is rejected
    let out = knpu_infer(&[truncated.as_os_str(), input.as_os_str()]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("bad model file"), "{}", stderr(&out));

    // An input or expected output of the wrong size is rejected
    let out = knpu_infer(&[model.as_os_str(), short.as_os_str()]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("the model takes 1568"), "{}", stderr(&out));
    let out = knpu_infer(&[
        model.as_os_str(),
        input.as_os_str(),
        "--expect".as_ref(),
        short.as_os_str(),
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("the model makes 3136"), "{}", stderr(&out));

    // With good files the next step is the device
    let out = knpu_infer(&[
        "--device".as_ref(),
        missing.as_os_str(),
        model.as_os_str(),
        input.as_os_str(),
        "--expect".as_ref(),
        expect.as_os_str(),
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("cannot open /nonexistent/accel0"), "{}", stderr(&out));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("model_ops=4"), "{stdout}");
    assert!(stdout.contains("cpu_steps=0"), "{stdout}");
    assert!(stdout.contains("input_shape=1x14x14x8"), "{stdout}");
}

#[test]
fn reads_models_with_cpu_code() {
    let input = vectors().join("input-0.bin");
    let missing = PathBuf::from("/nonexistent/accel0");
    let model = tmp("cpu.knpu", &model_with_cpu_code(183));
    let x86 = tmp("cpu-x86_64.knpu", &model_with_cpu_code(62));

    // A good model goes on to the device
    let out = knpu_infer(&[
        "--device".as_ref(),
        missing.as_os_str(),
        model.as_os_str(),
        input.as_os_str(),
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("cannot open /nonexistent/accel0"), "{}", stderr(&out));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("format=0.1"), "{stdout}");
    assert!(stdout.contains("cpu_steps=3"), "{stdout}");

    // CPU code for another machine is a bad model file
    let out = knpu_infer(&[x86.as_os_str(), input.as_os_str()]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("CPU_CODE is not for AArch64"), "{}", stderr(&out));
}
