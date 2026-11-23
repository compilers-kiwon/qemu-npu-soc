#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V2: the softmax that tflite2c writes against TFLite Micro.

The MNIST logits reach few of the paths of TFLite Micro's int8 Softmax, so
this test builds its own models: one int8 SOFTMAX with a 1x1xRxD input,
written with the object API of TFLite Micro's Python schema, for a range of
input scales, betas and row lengths D. For each model and rounding mode,
TFLite Micro (tflm_run_<mode>) and the tflite2c C, built by gcc (-std=c89
-pedantic-errors) with UBSan, run the same rows; the outputs must be equal
and UBSan must report nothing.

The rows are seeded random bytes, rows at the edges (all equal, all -128,
all 127, one high element among low ones) and rows that together hold
every difference from the row maximum, -255 to 0, so also those around
diff_min, below which TFLite Micro skips an element.
"""

import argparse
import importlib.util
import os
import random
import struct
import subprocess
import sys

import flatbuffers

SEED = 20261123

# (input scale, beta, row length): typical logit scales, scales so large
# that the input multiplier is capped (from 16 in the single rounding build
# and 32 in the double one), betas other than 1 and long rows
CASES = [(s, 1.0, 10) for s in (0.001, 0.02, 0.0625, 0.14331382513046265,
                                0.3, 1.0, 4.0, 15.9, 16.0, 31.9, 32.0, 100.0)]
CASES += [(0.1, b, 10) for b in (0.1, 0.5, 1.7, 2.0)]
CASES += [(0.05, 1.0, 1), (0.05, 1.0, 2), (0.07, 1.0, 37), (0.2, 1.0, 100),
          (0.05, 1.0, 511), (2.0, 1.0, 511)]
RANDOM_ROWS = 64


def load_schema(path):
    spec = importlib.util.spec_from_file_location("schema_py_generated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def softmax_model(schema, flatbuffers, rows, depth, scale, zero_point, beta):
    """A .tflite model with one SOFTMAX over 1 x 1 x rows x depth."""
    def tensor(name, buffer, scale, zero_point):
        t = schema.TensorT()
        t.name, t.shape, t.buffer = name, [1, 1, rows, depth], buffer
        t.type = schema.TensorType.INT8
        t.quantization = schema.QuantizationParametersT()
        t.quantization.scale = [scale]
        t.quantization.zeroPoint = [zero_point]
        return t

    code = schema.OperatorCodeT()
    code.builtinCode = schema.BuiltinOperator.SOFTMAX
    code.deprecatedBuiltinCode = schema.BuiltinOperator.SOFTMAX
    code.version = 1
    op = schema.OperatorT()
    op.opcodeIndex, op.inputs, op.outputs = 0, [0], [1]
    op.builtinOptionsType = schema.BuiltinOptions.SoftmaxOptions
    op.builtinOptions = schema.SoftmaxOptionsT()
    op.builtinOptions.beta = beta
    sub = schema.SubGraphT()
    sub.tensors = [tensor("input", 1, scale, zero_point),
                   tensor("output", 2, 1.0 / 256, -128)]
    sub.inputs, sub.outputs, sub.operators = [0], [1], [op]
    model = schema.ModelT()
    model.version = 3
    model.operatorCodes = [code]
    model.subgraphs = [sub]
    model.buffers = [schema.BufferT() for _ in range(3)]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def rows_for(rng, depth):
    rows = [[rng.randint(-128, 127) for _ in range(depth)]
            for _ in range(RANDOM_ROWS)]
    rows += [[0] * depth, [-128] * depth, [127] * depth,
             [127] + [-128] * (depth - 1), [-128] * (depth - 1) + [127]]
    # Rows of 127 followed by -128, -127, ... 126 in turn
    values = list(range(-128, 127))
    while depth > 1 and values:
        rows.append([127] + values[:depth - 1])
        rows[-1] += [127] * (depth - len(rows[-1]))
        values = values[depth - 1:]
    return bytes(v & 0xFF for row in rows for v in row), len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for opt in ("tflite2c", "schema", "cc", "tflm-double", "tflm-single",
                "work-dir"):
        ap.add_argument(f"--{opt}", required=True)
    args = ap.parse_args()
    work = args.work_dir
    os.makedirs(work, exist_ok=True)
    schema = load_schema(args.schema)

    rng = random.Random(SEED)
    cflags = ["-std=c89", "-pedantic-errors", "-Wall", "-Wno-unknown-pragmas",
              "-Wno-unused-const-variable", "-O1", "-fsanitize=undefined",
              "-fno-sanitize-recover=undefined"]
    failures = passes = 0
    for k, (scale, beta, depth) in enumerate(CASES):
        scale, beta = f32(scale), f32(beta)
        zero_point = rng.randint(-128, 127)
        data, rows = rows_for(rng, depth)
        name = f"softmax-{k}"
        model = os.path.join(work, f"{name}.tflite")
        with open(model, "wb") as f:
            f.write(softmax_model(schema, flatbuffers, rows, depth, scale,
                                  zero_point, beta))
        inp = os.path.join(work, f"{name}-input.bin")
        with open(inp, "wb") as f:
            f.write(data)
        for mode in ("double", "single"):
            tflm = args.tflm_double if mode == "double" else args.tflm_single
            want = os.path.join(work, f"{name}-{mode}-tflm.bin")
            subprocess.run([tflm, model, inp, want], check=True)
            src = os.path.join(work, f"{name}-{mode}.c")
            main_c = os.path.join(work, f"{name}-main.c")
            exe = os.path.join(work, f"{name}-{mode}")
            got = os.path.join(work, f"{name}-{mode}-gcc.bin")
            subprocess.run([sys.executable, args.tflite2c, "--schema",
                            args.schema, "--round", mode, model, "-o", src,
                            "--main", main_c], check=True)
            subprocess.run([args.cc, *cflags, "-o", exe, src, main_c],
                           check=True)
            r = subprocess.run([exe, inp, got], capture_output=True,
                               text=True)
            with open(want, "rb") as f:
                expected = f.read()
            ok = r.returncode == 0 and not r.stderr
            if ok:
                with open(got, "rb") as f:
                    ok = f.read() == expected
            what = (f"{name} {mode}: scale {scale!r}, beta {beta!r}, "
                    f"{rows} rows of {depth}")
            if ok:
                passes += 1
                print(f"PASS {what}")
            else:
                failures += 1
                print(f"FAIL {what}: exit {r.returncode} {r.stderr.strip()}")
    print(f"{passes} passed, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
