#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V3: int8 average pooling on the shared core against TFLite Micro (V2).

For every case the test makes a seeded random input, asks both
tflm_golden_<mode> builds for the reference output, runs a command stream
(DMA in, OP_AVGPOOL, DMA out) on knpu_run and compares the output bytes and
the performance counters. Average pooling does not depend on the rounding
mode, so the two references must agree as well. The cases cover windows
from 1x1 to the whole input, stride 1, 2 and 3, VALID and SAME padding
(the windows at the edges then hold fewer elements), non-square windows,
clamps, extreme inputs, inputs that put many averages exactly on .5, and
the two pooling layers of the MNIST CNN. The streams also set zero points
and a rounding mode, which OP_AVGPOOL ignores (docs/npu-spec.md 8.4).

The error cases check the shape rule, the SRAM range, the overlap rule and
the synchronization rule of OP_AVGPOOL (spec 6.4 and 9) with their error
codes and ERR_OFFSET.
"""

import argparse
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knpu_asm import assemble  # noqa: E402
from test_conv import Machine, same_padding, valid_padding  # noqa: E402
from test_errors import (BAD_PARAM, DMA_IN, DMA_OUT, NONE,  # noqa: E402
                         REGION1, SRAM_RANGE, SYNC, fmt, offset_of)


class Case:
    def __init__(self, name, rng, h, w, c, k, s, padding, mode,
                 clamp=(-128, 127), values=None):
        self.name, self.mode, self.clamp = name, mode, clamp
        self.h, self.w, self.c = h, w, c
        self.kh, self.kw = k
        self.sh, self.sw = s
        pad = same_padding if padding == "SAME" else valid_padding
        self.oh, self.pt, self.pb = pad(h, self.kh, self.sh)
        self.ow, self.pl, self.pr = pad(w, self.kw, self.sw)
        # Set in the stream to show that OP_AVGPOOL does not use them
        self.ifm_zp = rng.randint(-128, 127)
        self.ofm_zp = rng.randint(-128, 127)
        values = values or range(-128, 128)
        self.input = bytes(rng.choice(values) & 0xFF
                           for _ in range(h * w * c))

    def golden(self, tool, work_dir, mode):
        inp = os.path.join(work_dir, f"{self.name}-input.bin")
        with open(inp, "wb") as f:
            f.write(self.input)
        out = os.path.join(work_dir, f"{self.name}-golden-{mode}.bin")
        subprocess.run([
            tool, "--op", "avgpool",
            "--ifm", f"{self.h},{self.w},{self.c}",
            "--ofm", f"{self.oh},{self.ow},{self.c}",
            "--kernel", f"{self.kh},{self.kw}",
            "--stride", f"{self.sh},{self.sw}",
            "--pad", f"{self.pt},{self.pl}",
            "--clamp", f"{self.clamp[0]},{self.clamp[1]}",
            "--input", inp, "--output", out,
        ], check=True)
        with open(out, "rb") as f:
            return f.read()

    def program(self):
        """Region 1 holds the input and then the output."""
        out_size = self.oh * self.ow * self.c
        out_off = (len(self.input) + 15) & ~15
        src = f"""
            SET_DMA_SRC region1, 0
            SET_DMA_DST sram, 0
            SET_DMA_LEN {len(self.input)}
            OP_DMA
            SET_IFM_BASE 0
            SET_IFM_SHAPE {self.h}, {self.w}, {self.c}
            SET_IFM_ZP {self.ifm_zp}
            SET_KERNEL {self.kh}, {self.kw}
            SET_STRIDE {self.sh}, {self.sw}
            SET_PAD {self.pt}, {self.pl}, {self.pb}, {self.pr}
            SET_OFM_BASE {out_off}
            SET_OFM_SHAPE {self.oh}, {self.ow}, {self.c}
            SET_OFM_ZP {self.ofm_zp}
            SET_CLAMP {self.clamp[0]}, {self.clamp[1]}
            SET_ROUND_MODE {self.mode}
            OP_DMA_WAIT 0
            OP_AVGPOOL
            SET_DMA_SRC sram, {out_off}
            SET_DMA_DST region1, {out_off}
            SET_DMA_LEN {out_size}
            OP_KERNEL_WAIT 0
            OP_DMA
            OP_STOP
        """
        regions = [(1, out_off + out_size, self.input)]
        dumps = [(1, out_off, out_size)]
        return assemble(src), regions, dumps


def make_cases(seed):
    rng = random.Random(seed)
    cases = []

    def add(name, *args, **kwargs):
        mode = ("double", "single")[len(cases) % 2]
        cases.append(Case(name, rng, *args, mode, **kwargs))

    for k, s in (((2, 2), (2, 2)), ((2, 2), (1, 1)), ((3, 3), (1, 1)),
                 ((3, 3), (2, 2))):
        for padding in ("VALID", "SAME"):
            for i in range(2):
                add(f"k{k[0]}-s{s[0]}-{padding}-{i}",
                    rng.randint(k[0], 12), rng.randint(k[1], 12),
                    rng.randint(1, 16), k, s, padding)
    add("mnist-pool1", 26, 26, 8, (2, 2), (2, 2), "VALID")
    add("mnist-pool2", 11, 11, 16, (2, 2), (2, 2), "VALID")
    add("k1", 5, 4, 3, (1, 1), (1, 1), "VALID")
    add("whole-input", 5, 7, 6, (5, 7), (1, 1), "VALID")
    add("k3x1-s2x1", 9, 8, 4, (3, 1), (2, 1), "SAME")
    add("k4-s3-SAME", 10, 11, 3, (4, 4), (3, 3), "SAME")
    add("k7-SAME", 6, 5, 2, (7, 7), (1, 1), "SAME")
    add("relu", 8, 8, 5, (3, 3), (2, 2), "SAME", clamp=(-5, 127))
    add("narrow", 8, 8, 5, (3, 3), (1, 1), "SAME", clamp=(-20, 30))
    add("all-min", 6, 6, 3, (3, 3), (1, 1), "SAME", values=[-128])
    add("all-max", 6, 6, 3, (3, 3), (1, 1), "SAME", values=[127])
    add("extremes", 6, 6, 3, (3, 3), (1, 1), "SAME", values=[-128, 127])
    add("ties", 8, 8, 4, (2, 2), (2, 2), "VALID", values=[-1, 0, 1])
    add("ties-SAME", 7, 7, 4, (3, 3), (1, 1), "SAME", values=[-1, 0, 1])
    return cases


# Average pooling 2x2, stride 2, from 4x4x2 to 2x2x2. SRAM: IFM 0x000
# (32 bytes), OFM 0x100 (8 bytes).
POOL = """
    SET_IFM_BASE 0x000
    SET_IFM_SHAPE 4, 4, 2
    SET_KERNEL 2, 2
    SET_STRIDE 2, 2
    SET_OFM_BASE 0x100
    SET_OFM_SHAPE 2, 2, 2
""".strip().splitlines()

# (name, lines, expected error, index of the failing line, Machine.run
#  arguments)
ERROR_CASES = [
    ("good stream", POOL + ["OP_AVGPOOL", "OP_STOP"], NONE, None, {}),
    ("reserved PARAM", POOL + [".word 0x00010004"], BAD_PARAM, 6, {}),

    # Shape (spec 6.4)
    ("shapes not set", ["OP_AVGPOOL"], BAD_PARAM, 0, {}),
    ("OFM shape not set", POOL[:4] + ["OP_AVGPOOL"], BAD_PARAM, 4, {}),
    ("channel counts differ", POOL + ["SET_OFM_SHAPE 2, 2, 3",
                                      "OP_AVGPOOL"], BAD_PARAM, 7, {}),
    ("OFM shape does not match", POOL + ["SET_OFM_SHAPE 2, 1, 2",
                                         "OP_AVGPOOL"], BAD_PARAM, 7, {}),
    ("padding not below the kernel", POOL + ["SET_PAD 0, 2, 0, 0",
                                             "OP_AVGPOOL"], BAD_PARAM, 7,
     {}),
    ("SAME 3x3 shape", POOL + ["SET_KERNEL 3, 3", "SET_STRIDE 1, 1",
                               "SET_PAD 1, 1, 1, 1", "SET_OFM_SHAPE 4, 4, 2",
                               "OP_AVGPOOL", "OP_STOP"], NONE, None, {}),
    ("shape checked before the SRAM range", POOL + ["SET_IFM_BASE 0xFFF0",
                                                    "SET_OFM_SHAPE 2, 2, 3",
                                                    "OP_AVGPOOL"],
     BAD_PARAM, 8, {}),

    # SRAM range and overlap (spec 6.4)
    ("IFM beyond the SRAM", POOL + ["SET_IFM_BASE 0xFFF0", "OP_AVGPOOL"],
     SRAM_RANGE, 7, {}),
    ("OFM beyond a small SRAM", POOL + ["OP_AVGPOOL"], SRAM_RANGE, 6,
     {"sram_size": 0x104}),
    ("OFM overlaps the IFM", POOL + ["SET_OFM_BASE 0x01C", "OP_AVGPOOL"],
     BAD_PARAM, 7, {}),
    ("OFM right after the IFM", POOL + ["SET_OFM_BASE 0x020", "OP_AVGPOOL",
                                        "OP_STOP"], NONE, None, {}),
    ("range checked before the overlap", POOL + ["SET_IFM_BASE 0xFFF0",
                                                 "SET_OFM_BASE 0xFFF8",
                                                 "OP_AVGPOOL"],
     SRAM_RANGE, 8, {}),

    # Synchronization (spec 9)
    ("avgpool reads a DMA in flight", fmt(DMA_IN, 0) + POOL
     + ["OP_AVGPOOL"], SYNC, 10, {"regions": [REGION1]}),
    ("avgpool after OP_DMA_WAIT", fmt(DMA_IN, 0) + POOL
     + ["OP_DMA_WAIT 0", "OP_AVGPOOL", "OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
    ("DMA reads an avgpool output in flight", POOL + ["OP_AVGPOOL"]
     + fmt(DMA_OUT, 0x100), SYNC, 10, {"regions": [REGION1]}),
    ("DMA overwrites an avgpool input in flight", POOL + ["OP_AVGPOOL"]
     + fmt(DMA_IN, 0), SYNC, 10, {"regions": [REGION1]}),
    ("DMA after OP_KERNEL_WAIT", POOL + ["OP_AVGPOOL", "OP_KERNEL_WAIT 0"]
     + fmt(DMA_OUT, 0x100) + ["OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
    ("DMA next to an avgpool in flight", POOL + ["OP_AVGPOOL"]
     + fmt(DMA_IN, 0x800) + ["OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
]


def check_case(machine, case, golden_tool, work_dir):
    """Problems of one golden case; an empty list when it passes."""
    expected = case.golden(golden_tool["double"], work_dir, "double")
    single = case.golden(golden_tool["single"], work_dir, "single")
    stream, regions, dumps = case.program()
    res, (got,) = machine.run(stream, regions, dumps)
    problems = []
    if single != expected:
        problems.append("the two TFLite Micro builds differ")
    if res["err_code"] != 0:
        problems.append(f"error {res['err_code']:#x} at "
                        f"{res['err_offset']:#x}")
    if got != expected:
        diff = sum(a != b for a, b in zip(got, expected))
        problems.append(f"{diff} of {len(expected)} output bytes differ")
    want = {"macs": 0, "dma_rd_bytes": len(case.input),
            "dma_wr_bytes": len(expected), "ops": 3}
    for key, value in want.items():
        if res[key] != value:
            problems.append(f"{key} {res[key]} != {value}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--knpu-run", required=True)
    parser.add_argument("--golden-double", required=True)
    parser.add_argument("--golden-single", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    machine = Machine(args.knpu_run, args.work_dir)
    golden_tool = {"double": args.golden_double,
                   "single": args.golden_single}
    failures = 0
    cases = make_cases(args.seed)
    for case in cases:
        problems = check_case(machine, case, golden_tool, args.work_dir)
        if problems:
            failures += 1
            print(f"FAIL {case.name}: " + "; ".join(problems))
        else:
            print(f"PASS {case.name}: {case.h}x{case.w}x{case.c} -> "
                  f"{case.oh}x{case.ow}x{case.c}, kernel {case.kh}x"
                  f"{case.kw}, stride {case.sh}x{case.sw}, pad {case.pt},"
                  f"{case.pl},{case.pb},{case.pr}")

    for name, lines, code, index, kwargs in ERROR_CASES:
        res, _ = machine.run(assemble("\n".join(lines)), **kwargs)
        offset = 0 if index is None else offset_of(lines, index)
        got = (res["err_code"], res["err_offset"])
        if got == (code, offset):
            print(f"PASS {name}: error {code:#x} at {offset:#x}")
        else:
            failures += 1
            print(f"FAIL {name}: expected error {code:#x} at {offset:#x}, "
                  f"got {got[0]:#x} at {got[1]:#x}")

    total = len(cases) + len(ERROR_CASES)
    print(f"{total - failures} of {total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
