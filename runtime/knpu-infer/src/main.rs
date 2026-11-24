// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! knpu-infer: runs a model.knpu file on the kn1-npu (docs/design.md 10).
//!
//! Prints key=value lines: the model, the device, the job's performance
//! counters, the top-1 class of a classification output and, with
//! --expect, whether the output matches. Exit status: 0 done (and
//! matching), 1 output differs from --expect, 2 error.

use std::fs;
use std::process::ExitCode;

use knpu::{Device, Model};

const USAGE: &str = "\
usage: knpu-infer [--device PATH] MODEL INPUT [--expect FILE] [--output FILE]

Runs MODEL (a model.knpu file) on the kn1-npu with INPUT, the int8 bytes of
the model input, and prints the job's performance counters and, if the
output is a classification vector, the index of its largest element
(top1). The CPU code of MODEL, if it has any, runs in this process: run
only model files you trust like programs.

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

/// The index of the largest int8 element, the first one on a tie, of an
/// output that is a classification vector: two or more elements, and every
/// dimension but the last 1 (docs/design.md 10).
fn top1(shape: &[u32], data: &[u8]) -> Option<usize> {
    let (last, outer) = shape.split_last()?;
    if *last < 2 || outer.iter().any(|&d| d != 1) {
        return None;
    }
    let mut best = 0;
    for (i, &v) in data.iter().enumerate() {
        if v as i8 > data[best] as i8 {
            best = i;
        }
    }
    Some(best)
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
        "model={} format=0.{} sram_size={} command_bytes={} model_ops={} cpu_steps={}",
        args.model,
        model.format_minor,
        model.sram_size,
        model.cmds.len(),
        model.op_count(),
        model.cpu_steps.len()
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
    // SAFETY: knpu-infer runs the model file it is given, CPU code
    // included, as a shell runs the program it is given: the user trusts it
    // (docs/model-format.md 11, the usage text)
    let mut loaded =
        unsafe { device.load_with_cpu_code(&model) }.map_err(|e| e.to_string())?;
    let out = loaded.run(&input).map_err(|e| e.to_string())?;
    let c = out.counters;
    println!(
        "macs={} dma_rd_bytes={} dma_wr_bytes={} ops={}",
        c.macs, c.dma_rd_bytes, c.dma_wr_bytes, c.ops
    );
    if let Some(class) = top1(&output_t.shape, &out.data) {
        println!("top1={class}");
    }

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

#[cfg(test)]
mod tests {
    use super::top1;

    #[test]
    fn top1_compares_int8_values() {
        // As u8, 0x80 (-128) and 0xff (-1) would be the largest
        assert_eq!(top1(&[1, 4], &[0x80, 0x7f, 0x01, 0xff]), Some(1));
        assert_eq!(top1(&[4], &[0xff, 0x01, 0x80, 0x00]), Some(1));
        assert_eq!(top1(&[1, 3], &[0x80, 0x80, 0xff]), Some(2));
    }

    #[test]
    fn top1_takes_the_first_of_equal_elements() {
        assert_eq!(top1(&[1, 1, 1, 3], &[5, 9, 9]), Some(1));
        assert_eq!(top1(&[1, 10], &[0x80; 10]), Some(0));
    }

    #[test]
    fn top1_needs_a_classification_vector() {
        assert_eq!(top1(&[1, 1], &[3]), None);
        assert_eq!(top1(&[2, 3], &[0; 6]), None);
        assert_eq!(top1(&[1, 14, 14, 16], &[0; 3136]), None);
    }
}
