#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V3: the compiled convolution model on the shared core against V2.

For both rounding modes:

1. tflite2c writes models/conv/conv.tflite as C89; gcc (-std=c89
   -pedantic-errors) builds it, and its outputs must equal TFLite Micro's
   (tflm_run_<mode>, V2) for every input.
2. lcc (aarch64 mlir driver, -Wf-pragma=knpu) lowers the same C to MLIR;
   knpu-compile turns it into model.knpu for a 64 KiB SRAM (one tile) and a
   4 KiB SRAM (several tiles, double-buffered). model_file.py reads the file
   and knpu_run executes it on hw/npu-core; the outputs must equal V2 and
   the counters must match the model, independently of tiling.
3. The streams contain wait commands; compiled without them
   (--insert-waits=false) the device stops with a synchronization error.
4. With --vectors, the committed test vectors of V6 (models/conv/vectors/:
   the four models, five inputs and their V2 outputs per rounding mode) must
   equal what this run makes; --update-vectors writes them instead.

Inputs: seeded random bytes plus all -128 and all 127.
"""

import argparse
import glob
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_file  # noqa: E402

SEED = 20261122
QBASE = 0x8000_0000
REGION_ADDR = [0x1_0000_0000 + n * 0x1000_0000 for n in range(8)]
ERR_SYNC = 0x07
OP_DMA_WAIT, OP_KERNEL_WAIT, OP_CONV = 0x0011, 0x0012, 0x0002

# The test model (models/conv/make_model.py)
H, W, IC, OC, K = 14, 14, 8, 16, 3
MACS = H * W * OC * K * K * IC   # nominal MACs, SAME padding
OUT_BYTES = H * W * OC

# Inputs that go into the test vectors: the first three random ones (0 and 2
# give different outputs in the two rounding modes) and the two edge cases
VECTOR_INPUTS = (0, 1, 2, 8, 9)
SRAM_NAMES = {65536: "64k", 4096: "4k"}


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


def run_npu(args, model, inp, work, tag, arena=False):
    """Runs model.knpu on knpu_run; returns (result dict, output bytes), or
    with arena the bytes of the output's whole region instead."""
    cmd = [args.knpu_run, "--sram-size", str(model.sram_size),
           "--queue", f"{QBASE:#x}:{len(model.cmds):#x}",
           "--mem", f"{QBASE:#x}:{len(model.cmds):#x}:"
           + write(os.path.join(work, f"{tag}-queue.bin"), model.cmds)]
    for r in model.regions:
        if r.role == "const":
            data = r.content
        else:
            buf = bytearray(r.size)
            for t in model.tensors:
                if t.direction == "input" and t.region == r.index:
                    buf[t.offset:t.offset + t.size] = inp
            data = bytes(buf)
        path = write(os.path.join(work, f"{tag}-region{r.index}.bin"), data)
        cmd += ["--region", f"{r.index}:{REGION_ADDR[r.index]:#x}",
                "--mem", f"{REGION_ADDR[r.index]:#x}:{r.size:#x}:{path}"]
    out = next(t for t in model.tensors if t.direction == "output")
    start, size = REGION_ADDR[out.region] + out.offset, out.size
    if arena:
        start, size = REGION_ADDR[out.region], model.region(out.region).size
    dump = os.path.join(work, f"{tag}-out.bin")
    cmd += ["--dump", f"{start:#x}:{size:#x}:{dump}"]
    text = run(cmd).stdout
    result = {k: int(v, 0) for k, v in (f.split("=") for f in text.split())}
    data = read(dump) if result["err_code"] == 0 else None
    return result, data


def check_vectors(c, directory, vectors, update):
    """Compares (or with update writes) the test vectors in directory."""
    if update:
        os.makedirs(directory, exist_ok=True)
        for name, data in vectors.items():
            write(os.path.join(directory, name), data)
        print(f"wrote {len(vectors)} test vectors to {directory}")
        return
    hint = "(make them again with --update-vectors)"
    for name, data in sorted(vectors.items()):
        path = os.path.join(directory, name)
        c.check(os.path.exists(path) and read(path) == data,
                f"test vector {name} differs from this run {hint}")
    present = {os.path.basename(p) for p in
               glob.glob(os.path.join(directory, "*.knpu")) +
               glob.glob(os.path.join(directory, "*.bin"))}
    for name in sorted(present - vectors.keys()):
        c.check(False, f"test vector {name} is not made any more {hint}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for opt in ("model", "tflite2c", "schema", "lcc-dir", "cc",
                "knpu-compile", "knpu-run", "tflm-double", "tflm-single",
                "work-dir"):
        ap.add_argument(f"--{opt}", required=True)
    ap.add_argument("--vectors", help="directory of the V6 test vectors")
    ap.add_argument("--update-vectors", action="store_true",
                    help="write the test vectors instead of comparing")
    args = ap.parse_args()
    if args.update_vectors and not args.vectors:
        ap.error("--update-vectors needs --vectors")
    work = args.work_dir
    os.makedirs(work, exist_ok=True)
    c = Checker()

    rng = random.Random(SEED)
    n_in = H * W * IC
    inputs = [bytes(rng.randrange(256) for _ in range(n_in))
              for _ in range(8)]
    inputs += [bytes([0x80]) * n_in, bytes([0x7F]) * n_in]
    vectors = {f"input-{n}.bin": inputs[i]
               for n, i in enumerate(VECTOR_INPUTS)}

    lcc = os.path.join(args.lcc_dir, "lcc")
    env = dict(os.environ, LCCDIR=args.lcc_dir)
    for mode in ("double", "single"):
        tflm = args.tflm_double if mode == "double" else args.tflm_single
        src = os.path.join(work, f"conv_{mode}.c")
        main_c = os.path.join(work, "main.c")
        run([sys.executable, args.tflite2c, "--schema", args.schema,
             "--round", mode, args.model, "-o", src, "--main", main_c])

        # V2 reference outputs
        golden = []
        for i, inp in enumerate(inputs):
            out = os.path.join(work, f"{mode}-{i}-tflm.bin")
            run([tflm, args.model,
                 write(os.path.join(work, f"in-{i}.bin"), inp), out])
            golden.append(read(out))
            c.check(len(golden[-1]) == OUT_BYTES, f"{mode}: TFLM output size")
        for n, i in enumerate(VECTOR_INPUTS):
            vectors[f"expected-{mode}-{n}.bin"] = golden[i]

        # The generated C on the CPU
        exe = os.path.join(work, f"conv_{mode}")
        run([args.cc, "-std=c89", "-pedantic-errors", "-Wall",
             "-Wno-unknown-pragmas", "-Wno-unused-const-variable", "-O1",
             "-o", exe, src, main_c])
        for i, inp in enumerate(inputs):
            out = os.path.join(work, f"{mode}-{i}-cpu.bin")
            run([exe, os.path.join(work, f"in-{i}.bin"), out])
            c.check(read(out) == golden[i],
                    f"{mode}: gcc build of the generated C, input {i}")

        # lcc -> knpu-compile -> shared core
        mlir = os.path.join(work, f"conv_{mode}.mlir")
        run([lcc, "-Wf-pragma=knpu", "-S", src, "-o", mlir], env=env)
        for sram in (65536, 4096):
            for waits in (True, False):
                tag = f"{mode}-{sram}" + ("" if waits else "-nowait")
                knpu = os.path.join(work, f"{tag}.knpu")
                run([args.knpu_compile, f"--sram-size={sram}",
                     f"--insert-waits={'true' if waits else 'false'}",
                     mlir, "-o", knpu])
                model = model_file.read(read(knpu))
                ops = [w & 0xFFFF for w in model.words()]
                convs = ops.count(OP_CONV)
                n_waits = ops.count(OP_DMA_WAIT) + ops.count(OP_KERNEL_WAIT)
                if sram == 65536:
                    c.check(convs == 1, f"{tag}: one tile ({convs})")
                else:
                    c.check(convs > 1, f"{tag}: several tiles ({convs})")
                c.check(model.sram_size == sram, f"{tag}: header SRAM size")

                if not waits:
                    c.check(n_waits == 0, f"{tag}: no wait commands")
                    result, _ = run_npu(args, model, inputs[0], work, tag)
                    c.check(result["err_code"] == ERR_SYNC,
                            f"{tag}: SYNC error without waits "
                            f"(err_code {result['err_code']:#x})")
                    continue

                c.check(n_waits > 0, f"{tag}: wait commands present")
                vectors[f"conv-{mode}-{SRAM_NAMES[sram]}.knpu"] = read(knpu)
                print(f"{tag}: {convs} tiles, {n_waits} waits, "
                      f"{len(model.cmds)} command bytes")
                for i, inp in enumerate(inputs):
                    result, out = run_npu(args, model, inp, work,
                                          f"{tag}-{i}")
                    c.check(result["err_code"] == 0,
                            f"{tag}: input {i} runs "
                            f"(err_code {result['err_code']:#x} at "
                            f"{result['err_offset']:#x})")
                    c.check(out == golden[i], f"{tag}: input {i} equals V2")
                    c.check(result["macs"] == MACS,
                            f"{tag}: PERF_MACS {result['macs']}")
                    c.check(result["dma_wr_bytes"] == OUT_BYTES,
                            f"{tag}: PERF_DMA_WR_BYTES "
                            f"{result['dma_wr_bytes']}")

    if args.vectors:
        check_vectors(c, args.vectors, vectors, args.update_vectors)
    print(f"{c.passes} passed, {c.failures} failed")
    sys.exit(1 if c.failures else 0)


if __name__ == "__main__":
    main()
