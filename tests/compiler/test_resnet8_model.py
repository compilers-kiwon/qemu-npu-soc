#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V2 and V3 of the ResNet-8 of M3 (docs/plan.md M3.1 to M3.7).

So far (M3.1 to M3.3) the model, its references and the generator:

1. models/resnet8/resnet8.tflite has the operators of docs/plan.md M3.1
   and the nominal MACs and parameters of docs/design.md 6.6.
2. TFLite Micro (tflm_run_<mode>, V2) runs the fifteen test inputs in both
   rounding modes and writes the outputs and the logits (the input of the
   final SOFTMAX).
3. At least one input gives different outputs in the two rounding modes,
   so that a wrong rounding mode shows in V3 and V6.
4. tflite2c writes the model as C89 for each rounding mode, in full and
   with --npu-only. gcc (-std=c89 -pedantic-errors) builds it, also with
   UBSan (-fsanitize=undefined), and for every input the outputs and the
   logits must equal TFLite Micro's; the --npu-only C gives the logits as
   its output. UBSan must report nothing.
5. lcc (aarch64 mlir driver, -A -Wf-pragma=knpu) takes the same C without
   warnings.

Inputs: the ten photographs (models/resnet8/photos/, one per CIFAR-10
class), three random inputs, all -128 and all 127. The random inputs come
from a seeded stream: the first two, and the first later one whose outputs
differ in the two rounding modes.
"""

import argparse
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mnist_model import (Checker, load_schema, read, run,  # noqa: E402
                              summary, top1, write)

SEED = 20261124
SEARCH_LIMIT = 200   # random inputs tried for one that tells the modes apart
IN_BYTES = 32 * 32 * 3
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
           "horse", "ship", "truck")

# docs/plan.md M3.1: the converter leaves no RESHAPE before the FC
OPERATORS = (["CONV_2D"] * 3 + ["ADD"] + (["CONV_2D"] * 3 + ["ADD"]) * 2
             + ["AVERAGE_POOL_2D", "FULLY_CONNECTED", "SOFTMAX"])
# docs/design.md 6.6
MACS = 32_562_192
PARAMS = 203_746


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for opt in ("model", "photos", "tflite2c", "schema", "lcc-dir", "cc",
                "tflm-double", "tflm-single", "work-dir"):
        ap.add_argument(f"--{opt}", required=True)
    args = ap.parse_args()
    work = args.work_dir
    os.makedirs(work, exist_ok=True)
    c = Checker()

    ops, macs, params = summary(args.model, load_schema(args.schema))
    c.check(ops == OPERATORS, f"operators {ops}")
    c.check(macs == MACS, f"nominal MACs {macs}")
    c.check(params == PARAMS, f"parameters {params}")

    names = [f"photo-{cls}" for cls in CLASSES]
    inputs = [read(os.path.join(args.photos, f"{n}.bin")) for n in names]
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
            c.check(len(golden[mode, n][0]) == len(CLASSES),
                    f"{mode} {n}: TFLM output size")
            c.check(len(golden[mode, n][1]) == len(CLASSES),
                    f"{mode} {n}: TFLM logits size")

    differ = [n for n in names
              if golden["double", n][0] != golden["single", n][0]]
    c.check(len(differ) > 0,
            "some input gives different outputs in the two rounding modes")
    print("outputs differ between the rounding modes for: "
          + (", ".join(differ) or "none"))
    print("top-1 (double): " + ", ".join(
        f"{n}={CLASSES[top1(golden['double', n][0])]}" for n in names))

    # V2: the tflite2c C on gcc, with and without UBSan, and on lcc
    lcc = os.path.join(args.lcc_dir, "lcc")
    env = dict(os.environ, LCCDIR=args.lcc_dir)
    cflags = ["-std=c89", "-pedantic-errors", "-Wall", "-Wno-unknown-pragmas",
              "-Wno-unused-const-variable", "-O1"]
    builds = (("gcc", cflags),
              ("ubsan", cflags + ["-fsanitize=undefined",
                                  "-fno-sanitize-recover=undefined"]))
    for mode in ("double", "single"):
        for npu_only in (False, True):
            tag = f"{mode}-npu" if npu_only else mode
            src = os.path.join(work, f"resnet8-{tag}.c")
            main_c = os.path.join(work, f"main-{tag}.c")
            mlir = os.path.join(work, f"resnet8-{tag}.mlir")
            run([sys.executable, args.tflite2c, "--schema", args.schema,
                 "--round", mode, args.model, "-o", src, "--main", main_c]
                + (["--npu-only"] if npu_only else []))
            r = subprocess.run([lcc, "-A", "-Wf-pragma=knpu", "-S", src,
                                "-o", mlir],
                               env=env, capture_output=True, text=True)
            c.check(r.returncode == 0 and not r.stderr,
                    f"{tag}: lcc -A: {r.stderr.strip()}")
            for build, flags in builds:
                exe = os.path.join(work, f"resnet8-{tag}-{build}")
                run([args.cc, *flags, "-o", exe, src, main_c])
                for n in names:
                    out, logits = (os.path.join(work, f"{tag}-{n}-{build}{x}")
                                   for x in (".bin", "-logits.bin"))
                    r = subprocess.run([exe, os.path.join(work, f"{n}.bin"),
                                        out, logits],
                                       capture_output=True, text=True)
                    c.check(r.returncode == 0 and not r.stderr,
                            f"{tag} {build} {n}: exit {r.returncode} "
                            f"{r.stderr.strip()}")
                    want, want_logits = golden[mode, n]
                    c.check(read(out) == (want_logits if npu_only else want),
                            f"{tag} {build} {n}: output differs from V2")
                    c.check(read(logits) == want_logits,
                            f"{tag} {build} {n}: logits differ from V2")

    print(f"{c.passes} passed, {c.failures} failed")
    sys.exit(1 if c.failures else 0)


if __name__ == "__main__":
    main()
