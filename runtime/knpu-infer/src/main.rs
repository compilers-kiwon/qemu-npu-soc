// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! knpu-infer: runs a model.knpu file on the kn1-npu (docs/design.md 10).
//!
//! Prints key=value lines: the model, the device, the job's performance
//! counters and, with --expect, whether the output matches. Exit status: 0
//! done (and matching), 1 output differs from --expect, 2 error.

use std::fs;
use std::process::ExitCode;

use knpu::{Device, Model};

const USAGE: &str = "\
usage: knpu-infer [--device PATH] MODEL INPUT [--expect FILE] [--output FILE]

Runs MODEL (a model.knpu file) on the kn1-npu with INPUT, the int8 bytes of
the model input, and prints the job's performance counters.

  --device PATH   accel device (default /dev/accel/accel0)
  --expect FILE   compare the output with the bytes in FILE
  --output FILE   write the output bytes to FILE

Exit status: 0 done, 1 the output differs from --expect, 2 error.";

struct Args {
    device: String,
    model: String,
    input: String,
    expect: Option<String>,
    output: Option<String>,
}

/// Parses the command line; Ok(None) asks for the help text.
fn parse_args(args: &[String]) -> Result<Option<Args>, String> {
    let mut device = "/dev/accel/accel0".to_string();
    let (mut expect, mut output) = (None, None);
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| {
            it.next().cloned().ok_or_else(|| format!("{name} needs a value"))
        };
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--device" => device = value("--device")?,
            "--expect" => expect = Some(value("--expect")?),
            "--output" => output = Some(value("--output")?),
            s if s.starts_with('-') && s.len() > 1 => {
                return Err(format!("unknown option {s}"));
            }
            _ => positional.push(arg.clone()),
        }
    }
    let [model, input]: [String; 2] = positional
        .try_into()
        .map_err(|_| "need a MODEL and an INPUT file".to_string())?;
    Ok(Some(Args { device, model, input, expect, output }))
}

fn read(path: &str, what: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| format!("cannot read {what} {path}: {e}"))
}

fn shape(dims: &[u32]) -> String {
    dims.iter().map(u32::to_string).collect::<Vec<_>>().join("x")
}

/// Runs the model; Ok(false) when the output differs from --expect.
fn run(args: &Args) -> Result<bool, String> {
    // Everything that can be checked without the device comes first
    let model = Model::parse(&read(&args.model, "model")?)
        .map_err(|e| format!("{}: {e}", args.model))?;
    let (input_t, output_t) = (
        model.input().map_err(|e| e.to_string())?,
        model.output().map_err(|e| e.to_string())?,
    );
    let input = read(&args.input, "input")?;
    if input.len() != input_t.size as usize {
        return Err(format!(
            "input {} has {} bytes, the model takes {}",
            args.input,
            input.len(),
            input_t.size
        ));
    }
    let expect = match &args.expect {
        Some(path) => {
            let data = read(path, "expected output")?;
            if data.len() != output_t.size as usize {
                return Err(format!(
                    "expected output {path} has {} bytes, the model makes {}",
                    data.len(),
                    output_t.size
                ));
            }
            Some(data)
        }
        None => None,
    };
    println!(
        "model={} format=0.{} sram_size={} command_bytes={} model_ops={}",
        args.model,
        model.format_minor,
        model.sram_size,
        model.cmds.len(),
        model.op_count()
    );
    println!(
        "input_shape={} input_scale={:?} input_zero_point={} \
         output_shape={} output_scale={:?} output_zero_point={}",
        shape(&input_t.shape),
        input_t.scale,
        input_t.zero_point,
        shape(&output_t.shape),
        output_t.scale,
        output_t.zero_point
    );

    let device = Device::open(&args.device).map_err(|e| e.to_string())?;
    let info = device.info();
    println!(
        "device={} npu={}.{} device_sram_size={}",
        args.device, info.version_major, info.version_minor, info.sram_size
    );
    let mut loaded = device.load(&model).map_err(|e| e.to_string())?;
    let out = loaded.run(&input).map_err(|e| e.to_string())?;
    let c = out.counters;
    println!(
        "macs={} dma_rd_bytes={} dma_wr_bytes={} ops={}",
        c.macs, c.dma_rd_bytes, c.dma_wr_bytes, c.ops
    );

    if let Some(path) = &args.output {
        fs::write(path, &out.data).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    let Some(expect) = expect else {
        return Ok(true);
    };
    let differing: Vec<usize> =
        (0..expect.len()).filter(|&i| out.data[i] != expect[i]).collect();
    match differing.first() {
        None => {
            println!("match=yes");
            Ok(true)
        }
        Some(&i) => {
            println!(
                "match=no differing={} first={} got={} expected={}",
                differing.len(),
                i,
                out.data[i] as i8,
                expect[i] as i8
            );
            Ok(false)
        }
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(Some(args)) => args,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("knpu-infer: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("knpu-infer: {e}");
            ExitCode::from(2)
        }
    }
}
