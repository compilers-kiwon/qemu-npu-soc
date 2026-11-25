#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V3: int8 addition on the shared core against TFLite Micro (V2).

For every case the test makes two seeded random inputs and the parameters
of an int8 ADD, asks both tflm_golden_<mode> builds for the reference
output, runs a command stream (DMA in, OP_ADD, DMA out) on knpu_run once
in each rounding mode and compares the output bytes with the reference of
that mode, and the performance counters. Most cases derive the multipliers
and shifts from input and output scales as TFLite Micro does
(add_common.cc). The others use equal input scales, the extreme
multipliers and shifts, or multipliers and shifts that put many input or
output rescalings exactly on .5, where the two rounding modes differ (such
a case fails if its two references agree). The cases also cover zero
points, clamps (ReLU and narrower), extreme inputs, the same input twice,
in-place additions (OFM_BASE equal to IFM_BASE, IFM2_BASE or both) and the
shapes of the three ADD layers of the ResNet-8 of M3 (docs/design.md 6.6).
The streams set a kernel, stride and padding, which OP_ADD ignores
(docs/npu-spec.md 6.4).

The error cases check the shape rule, the SRAM range, the overlap rule
with its in-place exception and the synchronization rule of OP_ADD (spec
6.4 and 9) with their error codes and ERR_OFFSET.
"""

import argparse
import math
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knpu_asm import assemble  # noqa: E402
from test_conv import Machine  # noqa: E402
from test_errors import (BAD_PARAM, DMA_IN, DMA_OUT, NONE,  # noqa: E402
                         REGION1, SRAM_RANGE, SYNC, fmt, offset_of)

MODES = ("double", "single")

# Input and output shapes of the three ADD layers of the ResNet-8
RESNET8_ADDS = ((32, 32, 26), (16, 16, 52), (8, 8, 104))


def quantize_multiplier(real):
    """(multiplier, shift) of a real multiplier in (0, 1), as TFLite Micro's
    QuantizeMultiplierSmallerThanOneExp (quantization_util.cc) makes them"""
    q, shift = math.frexp(real)
    m = math.floor(q * (1 << 31) + 0.5)     # TfLiteRound, q > 0
    if m == 1 << 31:
        m //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return m, shift


def tflite_scales(s1, s2, so):
    """The three (multiplier, shift) pairs of an int8 ADD with input scales
    s1, s2 and output scale so (add_common.cc, left_shift 20)"""
    twice_max = 2 * max(s1, s2)
    return (quantize_multiplier(s1 / twice_max),
            quantize_multiplier(s2 / twice_max),
            quantize_multiplier(twice_max / ((1 << 20) * so)))


class Case:
    def __init__(self, name, rng, h, w, c, kind="tflite", clamp=None,
                 values=None, same_input=False, in_place=None,
                 sram_size=None):
        self.name, self.kind = name, kind
        self.h, self.w, self.c = h, w, c
        self.same_input, self.in_place = same_input, in_place
        self.sram_size = sram_size

        self.ifm_zp = rng.randint(-128, 127)
        self.ifm2_zp = rng.randint(-128, 127)
        self.ofm_zp = rng.randint(-128, 127)
        if clamp == "relu":
            self.clamp = (self.ofm_zp, 127)
        else:
            self.clamp = clamp or (-128, 127)

        if kind == "tflite":
            s1 = 10 ** rng.uniform(-3, 0)
            s2 = s1 * 10 ** rng.uniform(-1.5, 1.5)
            so = (s1 + s2) * rng.uniform(0.5, 1.5)
            self.scales = tflite_scales(s1, s2, so)
        elif kind == "equal":
            s = 10 ** rng.uniform(-3, 0)
            self.scales = tflite_scales(s, s, 2 * s * rng.uniform(0.5, 1.5))
        elif kind == "output-ties":
            # The inputs scaled by 1/2 exactly and (d1 + d2) / 2 at the
            # output, d = input - zero point: the odd sums end in .5
            self.scales = ((1 << 30, 0), (1 << 30, 0), (1 << 30, -19))
        elif kind == "input-ties":
            # d / 2 at each input, which ends in .5 for odd d, and the sum
            # passed through unchanged
            self.scales = ((1 << 30, -20), (1 << 30, -20), ((1 << 31) - 1, 0))
        elif kind == "extreme":
            self.scales = tuple(
                (rng.choice([0, 1, 1 << 30, (1 << 31) - 1]),
                 rng.choice([0, -1, -15, -20, -31]))
                for _ in range(3))
        else:
            raise ValueError(kind)

        values = values or range(-128, 128)
        size = h * w * c
        self.input1 = bytes(rng.choice(values) & 0xFF for _ in range(size))
        self.input2 = self.input1 if same_input else bytes(
            rng.choice(values) & 0xFF for _ in range(size))
        # Set in the stream to show that OP_ADD does not use them
        self.kernel = (rng.randint(1, 5), rng.randint(1, 5))
        self.stride = (rng.randint(1, 3), rng.randint(1, 3))
        self.pad = [rng.randint(0, 15) for _ in range(4)]

    def golden(self, tool, work_dir, mode):
        files = []
        for i, data in enumerate((self.input1, self.input2), 1):
            files.append(os.path.join(work_dir, f"{self.name}-input{i}.bin"))
            with open(files[-1], "wb") as f:
                f.write(data)
        out = os.path.join(work_dir, f"{self.name}-golden-{mode}.bin")
        (m1, s1), (m2, s2), (mo, so) = self.scales
        subprocess.run([
            tool, "--op", "add",
            "--ifm", f"{self.h},{self.w},{self.c}",
            "--ifm-zp", str(self.ifm_zp), "--ifm2-zp", str(self.ifm2_zp),
            "--ofm-zp", str(self.ofm_zp),
            "--ifm-scale", f"{m1},{s1}", "--ifm2-scale", f"{m2},{s2}",
            "--ofm-scale", f"{mo},{so}",
            "--clamp", f"{self.clamp[0]},{self.clamp[1]}",
            "--input", files[0], "--input2", files[1], "--output", out,
        ], check=True)
        with open(out, "rb") as f:
            return f.read()

    def program(self, mode):
        """Region 1 holds the inputs and then the output."""
        size = len(self.input1)
        step = (size + 15) & ~15
        sram1, sram2 = 0, 0 if self.same_input else step
        sram_out = {None: sram2 + step, "ifm": sram1,
                    "ifm2": sram2}[self.in_place]
        arena = self.input1 if self.same_input else (
            self.input1 + bytes(step - size) + self.input2)
        out_off = (len(arena) + 15) & ~15
        dma_in2 = "" if self.same_input else f"""
            SET_DMA_SRC region1, {step}
            SET_DMA_DST sram, {sram2}
            OP_DMA"""
        (m1, s1), (m2, s2), (mo, so) = self.scales
        src = f"""
            SET_DMA_SRC region1, 0
            SET_DMA_DST sram, {sram1}
            SET_DMA_LEN {size}
            OP_DMA{dma_in2}
            SET_IFM_BASE {sram1}
            SET_IFM2_BASE {sram2}
            SET_IFM_SHAPE {self.h}, {self.w}, {self.c}
            SET_IFM_ZP {self.ifm_zp}
            SET_IFM2_ZP {self.ifm2_zp}
            SET_IFM_SCALE {m1}, {s1}
            SET_IFM2_SCALE {m2}, {s2}
            SET_KERNEL {self.kernel[0]}, {self.kernel[1]}
            SET_STRIDE {self.stride[0]}, {self.stride[1]}
            SET_PAD {', '.join(map(str, self.pad))}
            SET_OFM_BASE {sram_out}
            SET_OFM_SHAPE {self.h}, {self.w}, {self.c}
            SET_OFM_ZP {self.ofm_zp}
            SET_OFM_SCALE {mo}, {so}
            SET_CLAMP {self.clamp[0]}, {self.clamp[1]}
            SET_ROUND_MODE {mode}
            OP_DMA_WAIT 0
            OP_ADD
            SET_DMA_SRC sram, {sram_out}
            SET_DMA_DST region1, {out_off}
            SET_DMA_LEN {size}
            OP_KERNEL_WAIT 0
            OP_DMA
            OP_STOP
        """
        regions = [(1, out_off + size, arena)]
        dumps = [(1, out_off, size)]
        return assemble(src), regions, dumps


def make_cases(seed):
    rng = random.Random(seed)
    cases = []

    def add(name, *args, **kwargs):
        cases.append(Case(name, rng, *args, **kwargs))

    for i in range(10):
        add(f"tflite-{i}", rng.randint(1, 9), rng.randint(1, 9),
            rng.randint(1, 24))
    for i in range(2):
        add(f"equal-scales-{i}", rng.randint(1, 9), rng.randint(1, 9),
            rng.randint(1, 24), kind="equal")
    for i in range(4):
        add(f"extreme-scales-{i}", 6, 5, 7, kind="extreme")
    add("output-ties", 7, 6, 5, kind="output-ties")
    add("input-ties", 7, 6, 5, kind="input-ties")
    add("relu", 6, 5, 8, clamp="relu")
    add("narrow", 6, 5, 8, clamp=(-20, 30))
    add("all-min", 4, 4, 3, values=[-128])
    add("all-max", 4, 4, 3, values=[127])
    add("extremes", 6, 6, 3, values=[-128, 127])
    add("extremes-extreme-scales", 6, 6, 3, kind="extreme",
        values=[-128, 127])
    add("same-input", 5, 5, 4, same_input=True)
    add("in-place-ifm", 5, 5, 4, in_place="ifm")
    add("in-place-ifm2", 5, 5, 4, in_place="ifm2")
    add("in-place-both", 5, 5, 4, same_input=True, in_place="ifm")
    add("in-place-ties", 7, 6, 5, kind="output-ties", in_place="ifm2")
    for i, (h, w, c) in enumerate(RESNET8_ADDS, 1):
        # Two inputs and the output of the first do not fit 64 KiB
        add(f"resnet8-add{i}", h, w, c, clamp="relu", sram_size=128 * 1024)
    return cases


# Addition of two 4x4x2 tensors. SRAM: IFM 0x000, IFM2 0x100, OFM 0x200
# (32 bytes each). The scales stay (0, 0), which is valid.
ADD = """
    SET_IFM_BASE 0x000
    SET_IFM2_BASE 0x100
    SET_IFM_SHAPE 4, 4, 2
    SET_OFM_BASE 0x200
    SET_OFM_SHAPE 4, 4, 2
""".strip().splitlines()

# (name, lines, expected error, index of the failing line, Machine.run
#  arguments)
ERROR_CASES = [
    ("good stream", ADD + ["OP_ADD", "OP_STOP"], NONE, None, {}),
    ("reserved PARAM", ADD + [".word 0x00010003"], BAD_PARAM, 5, {}),

    # Shape (spec 6.4)
    ("shapes not set", ["OP_ADD"], BAD_PARAM, 0, {}),
    ("OFM shape not set", ADD[:4] + ["OP_ADD"], BAD_PARAM, 4, {}),
    ("IFM shape not set", ADD[:2] + ADD[3:] + ["OP_ADD"], BAD_PARAM, 4, {}),
    ("heights differ", ADD + ["SET_OFM_SHAPE 2, 4, 2", "OP_ADD"], BAD_PARAM,
     6, {}),
    ("same size, other shape", ADD + ["SET_OFM_SHAPE 4, 8, 1", "OP_ADD"],
     BAD_PARAM, 6, {}),
    ("channel counts differ", ADD + ["SET_OFM_SHAPE 4, 4, 1", "OP_ADD"],
     BAD_PARAM, 6, {}),
    ("kernel, stride and padding are not used",
     ADD + ["SET_KERNEL 3, 3", "SET_STRIDE 2, 2", "SET_PAD 2, 2, 2, 2",
            "OP_ADD", "OP_STOP"], NONE, None, {}),
    ("shape checked before the SRAM range", ADD + ["SET_IFM2_BASE 0xFFF0",
                                                   "SET_OFM_SHAPE 4, 4, 3",
                                                   "OP_ADD"],
     BAD_PARAM, 7, {}),

    # SRAM range and overlap (spec 6.4)
    ("IFM beyond the SRAM", ADD + ["SET_IFM_BASE 0xFFF0", "OP_ADD"],
     SRAM_RANGE, 6, {}),
    ("IFM2 beyond the SRAM", ADD + ["SET_IFM2_BASE 0xFFF0", "OP_ADD"],
     SRAM_RANGE, 6, {}),
    ("OFM beyond a small SRAM", ADD + ["OP_ADD"], SRAM_RANGE, 5,
     {"sram_size": 0x21F}),
    ("OFM at the end of a small SRAM", ADD + ["OP_ADD", "OP_STOP"], NONE,
     None, {"sram_size": 0x220}),
    ("OFM overlaps the IFM", ADD + ["SET_OFM_BASE 0x010", "OP_ADD"],
     BAD_PARAM, 6, {}),
    ("OFM overlaps the IFM2", ADD + ["SET_OFM_BASE 0x0F8", "OP_ADD"],
     BAD_PARAM, 6, {}),
    ("OFM right after the IFM2", ADD + ["SET_OFM_BASE 0x120", "OP_ADD",
                                        "OP_STOP"], NONE, None, {}),
    ("in place on the IFM", ADD + ["SET_OFM_BASE 0x000", "OP_ADD",
                                   "OP_STOP"], NONE, None, {}),
    ("in place on the IFM2", ADD + ["SET_OFM_BASE 0x100", "OP_ADD",
                                    "OP_STOP"], NONE, None, {}),
    ("in place on both inputs", ADD + ["SET_IFM2_BASE 0x000",
                                       "SET_OFM_BASE 0x000", "OP_ADD",
                                       "OP_STOP"], NONE, None, {}),
    ("in place on the IFM, overlapping the IFM2",
     ADD + ["SET_IFM2_BASE 0x010", "SET_OFM_BASE 0x000", "OP_ADD"],
     BAD_PARAM, 7, {}),
    ("in place on the IFM2, overlapping the IFM",
     ADD + ["SET_IFM_BASE 0x0F0", "SET_OFM_BASE 0x100", "OP_ADD"],
     BAD_PARAM, 7, {}),
    ("inputs that overlap each other", ADD + ["SET_IFM2_BASE 0x010",
                                              "OP_ADD", "OP_STOP"],
     NONE, None, {}),
    ("range checked before the overlap", ADD + ["SET_IFM_BASE 0xFFF0",
                                                "SET_OFM_BASE 0xFFF8",
                                                "OP_ADD"],
     SRAM_RANGE, 7, {}),

    # Synchronization (spec 9)
    ("add reads a DMA in flight", fmt(DMA_IN, 0x100) + ADD + ["OP_ADD"],
     SYNC, 9, {"regions": [REGION1]}),
    ("add after OP_DMA_WAIT", fmt(DMA_IN, 0x100) + ADD
     + ["OP_DMA_WAIT 0", "OP_ADD", "OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
    ("DMA reads an add output in flight", ADD + ["OP_ADD"]
     + fmt(DMA_OUT, 0x200), SYNC, 9, {"regions": [REGION1]}),
    ("DMA overwrites an add input in flight", ADD + ["OP_ADD"]
     + fmt(DMA_IN, 0x100), SYNC, 9, {"regions": [REGION1]}),
    ("DMA reads an in-place output in flight", ADD
     + ["SET_OFM_BASE 0x000", "OP_ADD"] + fmt(DMA_OUT, 0x000), SYNC, 10,
     {"regions": [REGION1]}),
    ("DMA after OP_KERNEL_WAIT", ADD + ["OP_ADD", "OP_KERNEL_WAIT 0"]
     + fmt(DMA_OUT, 0x200) + ["OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
    ("DMA next to an add in flight", ADD + ["OP_ADD"]
     + fmt(DMA_IN, 0x800) + ["OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
]


def check_case(machine, case, golden_tool, work_dir):
    """Problems of one golden case and the number of output bytes in which
    the two rounding modes differ"""
    golden = {mode: case.golden(golden_tool[mode], work_dir, mode)
              for mode in MODES}
    problems = []
    for mode in MODES:
        stream, regions, dumps = case.program(mode)
        res, (got,) = machine.run(stream, regions, dumps,
                                  sram_size=case.sram_size)
        expected = golden[mode]
        if res["err_code"] != 0:
            problems.append(f"{mode}: error {res['err_code']:#x} at "
                            f"{res['err_offset']:#x}")
        if got != expected:
            diff = sum(a != b for a, b in zip(got, expected))
            problems.append(f"{mode}: {diff} of {len(expected)} output "
                            "bytes differ")
        inputs = 1 if case.same_input else 2
        want = {"macs": 0, "dma_rd_bytes": inputs * len(case.input1),
                "dma_wr_bytes": len(expected), "ops": inputs + 2}
        for key, value in want.items():
            if res[key] != value:
                problems.append(f"{mode}: {key} {res[key]} != {value}")
    differ = sum(a != b for a, b in zip(golden["double"], golden["single"]))
    if case.kind.endswith("ties") and not differ:
        problems.append("the two rounding modes give the same output")
    return problems, differ


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--knpu-run", required=True)
    parser.add_argument("--golden-double", required=True)
    parser.add_argument("--golden-single", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()

    machine = Machine(args.knpu_run, args.work_dir)
    golden_tool = {"double": args.golden_double,
                   "single": args.golden_single}
    failures = 0
    cases = make_cases(args.seed)
    for case in cases:
        problems, differ = check_case(machine, case, golden_tool,
                                      args.work_dir)
        if problems:
            failures += 1
            print(f"FAIL {case.name}: " + "; ".join(problems))
        else:
            print(f"PASS {case.name}: {case.h}x{case.w}x{case.c}, scales "
                  f"{case.scales}, rounding modes differ in {differ} of "
                  f"{len(case.input1)} bytes")

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
