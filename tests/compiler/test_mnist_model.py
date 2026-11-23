#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V2 and V3 of the MNIST CNN of M2 (docs/plan.md M2.1 to M2.8).

So far (M2.1 to M2.4) the model, its references, the generator and the NPU
part of the compiled model:

1. models/mnist/mnist.tflite has the operators of docs/design.md 6.6 and
   its nominal MACs and parameters.
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
6. V3: knpu-compile turns lcc's output of the --npu-only C into model.knpu
   for a 64 KiB and a 32 KiB SRAM, with the overlapped and the serial
   (--overlap=false) schedule. knpu_run executes it on hw/npu-core for
   every input: the logits must equal V2's (so a fully connected layer run
   as a 1x1 convolution equals FullyConnected), PERF_MACS must be the
   model's nominal MACs and the DMA and operation counters what the command
   stream implies. With overlap the activations stay in SRAM: only the
   input and the constants are read and only the logits written. The
   estimated cycles (--report) are fewer with overlap.
7. Compiled without wait commands (--insert-waits=false), the model stops
   with a synchronization error.

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
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_file  # noqa: E402
from test_conv_model import ERR_SYNC, run_npu  # noqa: E402

SEED = 20261123
SEARCH_LIMIT = 200   # random inputs tried for one that tells the modes apart
IN_BYTES = 28 * 28
CLASSES = 10

# docs/design.md 6.6
OPERATORS = ["CONV_2D", "AVERAGE_POOL_2D", "CONV_2D", "AVERAGE_POOL_2D",
             "RESHAPE", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX"]
MACS = 214_304
PARAMS = 27_562
SRAM_SIZES = {65536: "64k", 32768: "32k"}

# Command numbers (hw/npu-core/knpu_cmd.h)
OP_STOP, OP_CONV, OP_ADD, OP_AVGPOOL = 0x0000, 0x0002, 0x0003, 0x0004
OP_DMA = 0x0010
SET_DMA_COUNT, SET_DMA_SRC, SET_DMA_LEN = 0x0108, 0x4010, 0x4012
CMD_PAYLOAD, SPACE_SRAM = 0x4000, 1


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


def stream_counters(model):
    """PERF_DMA_RD_BYTES, PERF_DMA_WR_BYTES and PERF_OPS that the command
    stream implies (docs/npu-spec.md 11)."""
    words = model.words()
    src_sram, length, count = True, 0, 1   # the SET state after START
    rd = wr = ops = 0
    i = 0
    while i < len(words):
        cmd, param = words[i] & 0xFFFF, words[i] >> 16
        payload = words[i + 1] if cmd & CMD_PAYLOAD else None
        i += 2 if cmd & CMD_PAYLOAD else 1
        if cmd == SET_DMA_SRC:
            src_sram = param >> 8 == SPACE_SRAM
        elif cmd == SET_DMA_LEN:
            length = payload
        elif cmd == SET_DMA_COUNT:
            count = param
        elif cmd == OP_DMA:
            ops += 1
            if src_sram:
                wr += length * count
            else:
                rd += length * count
        elif cmd in (OP_CONV, OP_ADD, OP_AVGPOOL):
            ops += 1
        elif cmd == OP_STOP:
            break
    return rd, wr, ops


def top1(data):
    """Index of the largest int8 element (the first one on a tie)."""
    values = [b - 256 if b > 127 else b for b in data]
    return values.index(max(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for opt in ("model", "digits", "tflite2c", "schema", "lcc-dir", "cc",
                "knpu-compile", "knpu-run", "tflm-double", "tflm-single",
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
            src = os.path.join(work, f"mnist-{tag}.c")
            main_c = os.path.join(work, f"main-{tag}.c")
            run([sys.executable, args.tflite2c, "--schema", args.schema,
                 "--round", mode, args.model, "-o", src, "--main", main_c]
                + (["--npu-only"] if npu_only else []))
            r = subprocess.run([lcc, "-A", "-Wf-pragma=knpu", "-S", src,
                                "-o", os.path.join(work, f"mnist-{tag}.mlir")],
                               env=env, capture_output=True, text=True)
            c.check(r.returncode == 0 and not r.stderr,
                    f"{tag}: lcc -A: {r.stderr.strip()}")
            for build, flags in builds:
                exe = os.path.join(work, f"mnist-{tag}-{build}")
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

    # V3: the NPU part on the shared core
    for mode in ("double", "single"):
        mlir = os.path.join(work, f"mnist-{mode}-npu.mlir")
        for sram, size_name in SRAM_SIZES.items():
            cycles = {}
            for overlap in (True, False):
                tag = f"{mode}-{size_name}" + ("" if overlap else "-serial")
                knpu = os.path.join(work, f"mnist-{tag}.knpu")
                text = run([args.knpu_compile, f"--sram-size={sram}",
                            f"--overlap={'true' if overlap else 'false'}",
                            "--report", mlir, "-o", knpu]).stdout
                total = re.search(r"total cycles (\d+)", text)
                cycles[overlap] = int(total.group(1)) if total else None
                model = model_file.read(read(knpu))
                c.check(model.sram_size == sram, f"{tag}: header SRAM size")
                rd, wr, ops = stream_counters(model)
                if overlap:
                    const = model.region(0).size
                    c.check((rd, wr) == (IN_BYTES + const, CLASSES),
                            f"{tag}: DMA reads {rd} and writes {wr}; only "
                            f"the input and {const} constant bytes are read "
                            f"and the logits written")
                for n, inp in zip(names, inputs):
                    result, out = run_npu(args, model, inp, work,
                                          f"npu-{tag}-{n}")
                    c.check(result["err_code"] == 0,
                            f"{tag} {n}: err_code {result['err_code']:#x} "
                            f"at {result['err_offset']:#x}")
                    c.check(out == golden[mode, n][1],
                            f"{tag} {n}: logits differ from V2")
                    c.check(result["macs"] == MACS,
                            f"{tag} {n}: PERF_MACS {result['macs']}")
                    c.check((result["dma_rd_bytes"], result["dma_wr_bytes"],
                             result["ops"]) == (rd, wr, ops),
                            f"{tag} {n}: counters differ from the stream")
            c.check(None not in cycles.values()
                    and cycles[True] < cycles[False],
                    f"{mode}-{size_name}: estimated cycles {cycles}")
            print(f"{mode}-{size_name}: estimated cycles {cycles[True]} "
                  f"overlapped, {cycles[False]} serial")

        tag = f"{mode}-64k-nowait"
        knpu = os.path.join(work, f"mnist-{tag}.knpu")
        run([args.knpu_compile, "--insert-waits=false", mlir, "-o", knpu])
        result, _ = run_npu(args, model_file.read(read(knpu)), inputs[0],
                            work, f"npu-{tag}")
        c.check(result["err_code"] == ERR_SYNC,
                f"{tag}: SYNC error without waits "
                f"(err_code {result['err_code']:#x})")

    print(f"{c.passes} passed, {c.failures} failed")
    sys.exit(1 if c.failures else 0)


if __name__ == "__main__":
    main()
