#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V2 and V3 of the MNIST CNN of M2 (docs/plan.md M2.1 to M2.8).

So far (M2.1) the model and its references:

1. models/mnist/mnist.tflite has the operators of docs/design.md 6.6 and
   its nominal MACs and parameters.
2. TFLite Micro (tflm_run_<mode>, V2) runs the fifteen test inputs in both
   rounding modes and writes the outputs and the logits (the input of the
   final SOFTMAX).
3. At least one input gives different outputs in the two rounding modes,
   so that a wrong rounding mode shows in V3 and V6.

Inputs: the ten digit images (models/mnist/digits/), three random inputs,
all -128 and all 127. The random inputs come from a seeded stream: the
first two, and the first later one whose outputs differ in the two
rounding modes. The two modes differ only at rounding ties, so few inputs
show the difference in the outputs (about one in eight random ones); the
digit images and the first random inputs do not.
"""

import argparse
import importlib.util
import os
import random
import subprocess
import sys

SEED = 20261123
SEARCH_LIMIT = 200   # random inputs tried for one that tells the modes apart
IN_BYTES = 28 * 28
CLASSES = 10

# docs/design.md 6.6
OPERATORS = ["CONV_2D", "AVERAGE_POOL_2D", "CONV_2D", "AVERAGE_POOL_2D",
             "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX"]
MACS = 214_304
PARAMS = 27_562


class Checker:
    def __init__(self):
        self.failures = 0
        self.passes = 0

    def check(self, cond, what):
        if cond:
            self.passes += 1
        else:
            self.failures += 1
            print(f"FAIL: {what}")


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True,
                          **kw)


def write(path, data):
    with open(path, "wb") as f:
        f.write(data)
    return path


def read(path):
    with open(path, "rb") as f:
        return f.read()


def load_schema(path):
    spec = importlib.util.spec_from_file_location("schema_py_generated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(path, schema):
    """Operator names, nominal MACs and parameter count of a .tflite file."""
    model = schema.Model.GetRootAs(read(path), 0)
    sub = model.Subgraphs(0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items()
             if isinstance(v, int)}

    def size(index):
        t = sub.Tensors(index)
        n = 1
        for j in range(t.ShapeLength()):
            n *= t.Shape(j)
        return n

    ops, macs, params = [], 0, 0
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        name = names.get(max(code.BuiltinCode(), code.DeprecatedBuiltinCode()),
                         "?")
        ops.append(name)
        if name in ("CONV_2D", "FULLY_CONNECTED"):
            inputs = [op.Inputs(j) for j in range(op.InputsLength())]
            out = sub.Tensors(op.Outputs(0))
            positions = 1   # output pixels (H x W) a filter is applied at
            if name == "CONV_2D":
                positions = out.Shape(1) * out.Shape(2)
            macs += positions * size(inputs[1])
            params += size(inputs[1])
            if len(inputs) > 2 and inputs[2] >= 0:
                params += size(inputs[2])
    return ops, macs, params


def top1(data):
    """Index of the largest int8 element (the first one on a tie)."""
    values = [b - 256 if b > 127 else b for b in data]
    return values.index(max(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for opt in ("model", "digits", "schema", "tflm-double", "tflm-single",
                "work-dir"):
        ap.add_argument(f"--{opt}", required=True)
    args = ap.parse_args()
    work = args.work_dir
    os.makedirs(work, exist_ok=True)
    c = Checker()

    ops, macs, params = summary(args.model, load_schema(args.schema))
    c.check(ops == OPERATORS, f"operators {ops}")
    c.check(macs == MACS, f"nominal MACs {macs}")
    c.check(params == PARAMS, f"parameters {params}")

    names = [f"digit-{d}" for d in range(10)]
    inputs = [read(os.path.join(args.digits, f"{n}.bin")) for n in names]
    for n, inp in zip(names, inputs):
        c.check(len(inp) == IN_BYTES, f"{n}: {len(inp)} bytes")

    def tflm(mode, name, data):
        """V2: the final output and the logits of one input."""
        program = args.tflm_double if mode == "double" else args.tflm_single
        out = os.path.join(work, f"{mode}-{name}-tflm.bin")
        logits = os.path.join(work, f"{mode}-{name}-tflm-logits.bin")
        inp = write(os.path.join(work, f"{name}.bin"), data)
        run([program, args.model, inp, out, logits])
        return read(out), read(logits)

    rng = random.Random(SEED)

    def random_input():
        return bytes(rng.randrange(256) for _ in range(IN_BYTES))

    names += ["random-0", "random-1"]
    inputs += [random_input(), random_input()]
    for i in range(2, 2 + SEARCH_LIMIT):
        candidate = random_input()
        if (tflm("double", "candidate", candidate)[0]
                != tflm("single", "candidate", candidate)[0]):
            break
    else:
        c.check(False, f"none of {SEARCH_LIMIT} random inputs gives different "
                       "outputs in the two rounding modes")
        print(f"{c.passes} passed, {c.failures} failed")
        sys.exit(1)
    print(f"random-2 is input {i} of the seeded stream")
    names += ["random-2", "all-min", "all-max"]
    inputs += [candidate, bytes([0x80]) * IN_BYTES, bytes([0x7F]) * IN_BYTES]

    # V2: final outputs and logits in both rounding modes
    golden = {}
    for mode in ("double", "single"):
        for n, inp in zip(names, inputs):
            golden[mode, n] = tflm(mode, n, inp)
            c.check(len(golden[mode, n][0]) == CLASSES,
                    f"{mode} {n}: TFLM output size")
            c.check(len(golden[mode, n][1]) == CLASSES,
                    f"{mode} {n}: TFLM logits size")

    differ = [n for n in names
              if golden["double", n][0] != golden["single", n][0]]
    c.check(len(differ) > 0,
            "some input gives different outputs in the two rounding modes")
    print("outputs differ between the rounding modes for: "
          + (", ".join(differ) or "none"))
    print("top-1 (double): " + ", ".join(
        f"{n}={top1(golden['double', n][0])}" for n in names))

    print(f"{c.passes} passed, {c.failures} failed")
    sys.exit(1 if c.failures else 0)


if __name__ == "__main__":
    main()
