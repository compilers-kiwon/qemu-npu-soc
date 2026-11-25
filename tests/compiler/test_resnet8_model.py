#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V2 and V3 of the ResNet-8 of M3 (docs/plan.md M3.1 to M3.7).

So far (M3.1 to M3.4) the model, its references, the generator and the
compiled model for a 128 KiB SRAM:

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
6. V3: knpu-compile turns lcc's output of the full C into model.knpu for a
   128 KiB SRAM, where every layer fits whole (one tile), with the
   overlapped and the serial (--overlap=false) schedule; the softmax layer
   becomes CPU code. knpu_run runs the NPU part on hw/npu-core for every
   input: it must leave V2's logits in the arena, PERF_MACS must be the
   model's nominal MACs and the DMA and operation counters what the command
   stream implies. cpu_run, an aarch64 program on qemu-aarch64 user mode,
   runs the CPU steps on that arena: the output must equal V2's. The
   estimated cycles (--report) are fewer with overlap. The time knpu_run
   takes for a job, the whole process, is printed: QEMU runs a job on the
   shared core in its main loop (docs/design.md D32).
7. Compiled without wait commands (--insert-waits=false), the model stops
   with a synchronization error.

Inputs: the ten photographs (models/resnet8/photos/, one per CIFAR-10
class), three random inputs, all -128 and all 127. The random inputs come
from a seeded stream: the first two, and the first later one whose outputs
differ in the two rounding modes.
"""

import argparse
import os
import random
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_file  # noqa: E402
from test_conv_model import ERR_SYNC, run_npu  # noqa: E402
from test_mnist_model import (Checker, load_schema, read, run,  # noqa: E402
                              stream_counters, summary, top1, write)

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
# docs/plan.md M3.4: every layer fits in SRAM whole
SRAM_SIZE = 131072


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for opt in ("model", "photos", "tflite2c", "schema", "lcc-dir", "cc",
                "knpu-compile", "knpu-run", "tflm-double", "tflm-single",
                "cpu-run", "qemu", "qemu-ld-prefix", "work-dir"):
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

    # V3: the whole model, the NPU part on the shared core and the CPU
    # steps on qemu-aarch64
    for mode in ("double", "single"):
        mlir = os.path.join(work, f"resnet8-{mode}.mlir")
        cycles = {}
        for overlap in (True, False):
            tag = f"{mode}-128k" + ("" if overlap else "-serial")
            knpu = os.path.join(work, f"resnet8-{tag}.knpu")
            text = run([args.knpu_compile, f"--sram-size={SRAM_SIZE}",
                        f"--overlap={'true' if overlap else 'false'}",
                        "--report", mlir, "-o", knpu]).stdout
            total = re.search(r"total cycles (\d+)", text)
            cycles[overlap] = int(total.group(1)) if total else None
            model = model_file.read(read(knpu))
            c.check(model.sram_size == SRAM_SIZE, f"{tag}: header SRAM size")
            counters = stream_counters(model)
            logits = next(s for s in model.cpu_steps if s.kind == "copy_in")
            out = next(t for t in model.tensors if t.direction == "output")
            seconds = []
            for n, inp in zip(names, inputs):
                start = time.perf_counter()
                result, arena = run_npu(args, model, inp, work,
                                        f"npu-{tag}-{n}", arena=True)
                seconds.append(time.perf_counter() - start)
                c.check(result["err_code"] == 0 and result["macs"] == MACS,
                        f"{tag} {n}: err_code {result['err_code']:#x}, "
                        f"PERF_MACS {result['macs']}")
                c.check((result["dma_rd_bytes"], result["dma_wr_bytes"],
                         result["ops"]) == counters,
                        f"{tag} {n}: counters differ from the stream")
                if arena is None:
                    continue
                c.check(arena[logits.offset:logits.offset + logits.size]
                        == golden[mode, n][1],
                        f"{tag} {n}: logits in the arena differ from V2")
                arena_in = write(os.path.join(work, f"cpu-{tag}-{n}-in.bin"),
                                 arena)
                arena_out = os.path.join(work, f"cpu-{tag}-{n}-out.bin")
                r = subprocess.run([args.qemu, "-L", args.qemu_ld_prefix,
                                    args.cpu_run, knpu, arena_in, arena_out],
                                   capture_output=True, text=True)
                c.check(r.returncode == 0 and not r.stderr,
                        f"{tag} {n}: cpu_run exit {r.returncode} "
                        f"{r.stderr.strip()}")
                if r.returncode == 0:
                    c.check(read(arena_out)[out.offset:out.offset + out.size]
                            == golden[mode, n][0],
                            f"{tag} {n}: output differs from V2")
            rd, wr, ops = counters
            print(f"{tag}: DMA reads {rd} and writes {wr} bytes, {ops} "
                  f"operations; knpu_run takes "
                  f"{statistics.median(seconds) * 1000:.0f} ms a job "
                  f"(median of {len(seconds)})")
        c.check(None not in cycles.values() and cycles[True] < cycles[False],
                f"{mode}-128k: estimated cycles {cycles}")
        print(f"{mode}-128k: estimated cycles {cycles[True]} overlapped, "
              f"{cycles[False]} serial")

        tag = f"{mode}-128k-nowait"
        knpu = os.path.join(work, f"resnet8-{tag}.knpu")
        run([args.knpu_compile, f"--sram-size={SRAM_SIZE}",
             "--insert-waits=false", mlir, "-o", knpu])
        result, _ = run_npu(args, model_file.read(read(knpu)), inputs[0],
                            work, f"npu-{tag}")
        c.check(result["err_code"] == ERR_SYNC,
                f"{tag}: SYNC error without waits "
                f"(err_code {result['err_code']:#x})")

    print(f"{c.passes} passed, {c.failures} failed")
    sys.exit(1 if c.failures else 0)


if __name__ == "__main__":
    main()
