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
    assert!(stdout.contains("input_shape=1x14x14x8"), "{stdout}");
}
