#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V3: error paths and synchronization rules of the shared core.

Every case is a short command stream with the expected error code and
ERR_OFFSET (docs/npu-spec.md 10), including the synchronization errors of
section 9 and streams that are correct only because of the implicit wait.
The example stream of spec section 12 is checked word for word.
"""

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knpu_asm import assemble  # noqa: E402
from test_conv import QBASE, Machine  # noqa: E402

NONE, UNKNOWN_CMD, TRUNCATED, NO_STOP, BAD_QUEUE = 0, 1, 2, 3, 4
BAD_PARAM, SRAM_RANGE, SYNC, BUS = 5, 6, 7, 8

# 1x1 convolution 4x4x2 -> 4x4x2. SRAM: IFM 0x000, weights 0x100,
# bias 0x200, requantization table 0x300 (zeros are valid), OFM 0x400.
CONV = """
    SET_IFM_BASE 0x000
    SET_IFM_SHAPE 4, 4, 2
    SET_WEIGHT_BASE 0x100
    SET_BIAS_BASE 0x200
    SET_RQ_TABLE 0x300
    SET_OFM_BASE 0x400
    SET_OFM_SHAPE 4, 4, 2
""".strip().splitlines()

# DMA 32 bytes between region 1 and the SRAM offset given by format()
DMA_IN = ["SET_DMA_SRC region1, 0", "SET_DMA_DST sram, {}",
          "SET_DMA_LEN 32", "OP_DMA"]
DMA_OUT = ["SET_DMA_SRC sram, {}", "SET_DMA_DST region1, 0x100",
           "SET_DMA_LEN 32", "OP_DMA"]
REGION1 = (1, 0x200, bytes(range(32)))


def fmt(lines, *args):
    return [line.format(*args) for line in lines]


# (name, lines, expected error, index of the failing line or "qsize",
#  Machine.run arguments)
CASES = [
    ("good stream", CONV + ["OP_CONV", "OP_STOP"], NONE, None, {}),

    # Queue (spec 3.9)
    ("QBASE not aligned", ["OP_STOP"], BAD_QUEUE, 0,
     {"qbase": QBASE + 2}),
    ("QSIZE zero", ["OP_STOP"], BAD_QUEUE, 0, {"qsize": 0}),
    ("QSIZE not a multiple of 4", ["OP_NOP", "OP_STOP"], BAD_QUEUE, 0,
     {"qsize": 6}),
    ("command fetch fails", ["OP_NOP"] * 3, BUS, 2,
     {"qsize": 16, "queue_mem": 8}),
    ("payload fetch fails", ["OP_NOP", "SET_IFM_BASE 0"], BUS, 1,
     {"qsize": 16, "queue_mem": 8}),

    # Decoding (spec 4)
    ("unknown command", ["OP_NOP", ".word 0x0099", "OP_STOP"],
     UNKNOWN_CMD, 1, {}),
    ("bit 15 set", ["OP_NOP", ".word 0x8000", "OP_STOP"], UNKNOWN_CMD, 1,
     {}),
    ("OP_ADD not implemented yet", ["OP_ADD", "OP_STOP"], UNKNOWN_CMD, 0,
     {}),
    ("SET_OFM_SCALE not implemented yet", ["SET_OFM_SCALE 1, 0", "OP_STOP"],
     UNKNOWN_CMD, 0, {}),
    ("payload outside QSIZE", ["OP_NOP", ".word 0x4000"], TRUNCATED, 1, {}),
    ("no OP_STOP", ["OP_NOP", "OP_NOP"], NO_STOP, "qsize", {}),
    ("words after OP_STOP are not read", ["OP_STOP", ".word 0x0099"],
     NONE, None, {}),

    # Field checks of SET commands (spec 5, 6.1)
    ("reserved PARAM of OP_NOP", [".word 0x00010001", "OP_STOP"],
     BAD_PARAM, 0, {}),
    ("reserved PARAM of a wait", ["OP_DMA_WAIT 0", ".word 0x00100011"],
     BAD_PARAM, 1, {}),
    ("zero point out of int8", ["SET_IFM_ZP 128"], BAD_PARAM, 0, {}),
    ("zero kernel size", ["SET_KERNEL 0, 3"], BAD_PARAM, 0, {}),
    ("zero stride", ["SET_STRIDE 1, 0"], BAD_PARAM, 0, {}),
    ("clamp minimum above maximum", ["SET_CLAMP 10, -10"], BAD_PARAM, 0,
     {}),
    ("rounding mode 2", [".word 0x00020107"], BAD_PARAM, 0, {}),
    ("zero DMA row count", ["SET_DMA_COUNT 0"], BAD_PARAM, 0, {}),
    ("bias base not aligned", ["SET_BIAS_BASE 0x202"], BAD_PARAM, 0, {}),
    ("table base not aligned", ["SET_RQ_TABLE 0x301"], BAD_PARAM, 0, {}),
    ("zero shape", ["SET_IFM_SHAPE 4, 0, 2"], BAD_PARAM, 0, {}),
    ("region 8", ["SET_DMA_SRC region8, 0"], BAD_PARAM, 0, {}),
    ("SRAM with a region number", [".word 0x01014010", ".word 0"],
     BAD_PARAM, 0, {}),
    ("reserved PARAM of a base", [".word 0x00014000", ".word 0"],
     BAD_PARAM, 0, {}),

    # OP_CONV checks (spec 6.4)
    ("OFM shape does not match", CONV + ["SET_OFM_SHAPE 3, 4, 2",
                                         "OP_CONV"], BAD_PARAM, 8, {}),
    ("padding not below the kernel", CONV + ["SET_PAD 1, 0, 0, 0",
                                             "OP_CONV"], BAD_PARAM, 8, {}),
    ("SAME 3x3 shape", CONV + ["SET_KERNEL 3, 3", "SET_PAD 1, 1, 1, 1",
                               "OP_CONV", "OP_STOP"], NONE, None, {}),
    ("OFM shape not set", CONV[:5] + ["OP_CONV"], BAD_PARAM, 5, {}),
    ("weights beyond the SRAM", CONV + ["SET_WEIGHT_BASE 0xFFFE",
                                        "OP_CONV"], SRAM_RANGE, 8, {}),
    ("OFM beyond a small SRAM", CONV + ["OP_CONV"], SRAM_RANGE, 7,
     {"sram_size": 0x410}),
    ("OFM overlaps the IFM", CONV + ["SET_OFM_BASE 0x010", "OP_CONV"],
     BAD_PARAM, 8, {}),
    ("OFM overlaps the table", CONV + ["SET_OFM_BASE 0x30C", "OP_CONV"],
     BAD_PARAM, 8, {}),

    # DMA checks (spec 6.3)
    ("DMA between two SRAM spaces", ["SET_DMA_LEN 4", "OP_DMA"],
     BAD_PARAM, 1, {}),
    ("DMA beyond the SRAM", fmt(DMA_IN, 0xFFF0), SRAM_RANGE, 3,
     {"regions": [REGION1]}),
    ("2-D DMA beyond the SRAM", fmt(DMA_IN, 0xFF00)[:3]
     + ["SET_DMA_COUNT 9", "OP_DMA"], SRAM_RANGE, 4,
     {"regions": [REGION1]}),
    ("DMA to unmapped memory", fmt(DMA_IN, 0), BUS, 3,
     {"region_addr": {1: 0x7000_0000}}),

    # Synchronization (spec 9)
    ("conv reads a DMA in flight", fmt(DMA_IN, 0) + CONV + ["OP_CONV"],
     SYNC, 11, {"regions": [REGION1]}),
    ("conv after OP_DMA_WAIT", fmt(DMA_IN, 0) + CONV
     + ["OP_DMA_WAIT 0", "OP_CONV", "OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
    ("DMA reads a conv output in flight", CONV + ["OP_CONV"]
     + fmt(DMA_OUT, 0x400), SYNC, 11, {"regions": [REGION1]}),
    ("DMA overwrites a conv input in flight", CONV + ["OP_CONV"]
     + fmt(DMA_IN, 0x100), SYNC, 11, {"regions": [REGION1]}),
    ("DMA after OP_KERNEL_WAIT", CONV + ["OP_CONV", "OP_KERNEL_WAIT 0"]
     + fmt(DMA_OUT, 0x400) + ["OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
    ("DMA next to a conv in flight", CONV + ["OP_CONV"]
     + fmt(DMA_IN, 0x800) + ["OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
    ("OP_DMA_WAIT 1 leaves the newest DMA", fmt(DMA_IN, 0x800)
     + fmt(DMA_IN, 0) + CONV + ["OP_DMA_WAIT 1", "OP_CONV"], SYNC, 16,
     {"regions": [REGION1]}),
    ("OP_DMA_WAIT 1 retires the oldest DMA", fmt(DMA_IN, 0)
     + fmt(DMA_IN, 0x800) + CONV + ["OP_DMA_WAIT 1", "OP_CONV",
                                    "OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
    ("third DMA retires the first", fmt(DMA_IN, 0) + fmt(DMA_IN, 0x800)
     + fmt(DMA_IN, 0x900) + CONV + ["OP_CONV", "OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
    ("third DMA keeps the second", fmt(DMA_IN, 0x800) + fmt(DMA_IN, 0)
     + fmt(DMA_IN, 0x900) + CONV + ["OP_CONV"], SYNC, 19,
     {"regions": [REGION1]}),
    ("third conv retires the first", CONV + ["OP_CONV",
                                             "SET_OFM_BASE 0x800",
                                             "OP_CONV",
                                             "SET_OFM_BASE 0x900",
                                             "OP_CONV"]
     + fmt(DMA_OUT, 0x400) + ["OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
    ("third conv keeps the second", CONV + ["OP_CONV",
                                            "SET_OFM_BASE 0x800",
                                            "OP_CONV",
                                            "SET_OFM_BASE 0x900",
                                            "OP_CONV"]
     + fmt(DMA_OUT, 0x800), SYNC, 15, {"regions": [REGION1]}),
    ("DMAs in one queue never conflict", fmt(DMA_IN, 0)
     + fmt(DMA_OUT, 0) + ["OP_STOP"], NONE, None,
     {"regions": [REGION1]}),
]


def offset_of(lines, index):
    return len(assemble("\n".join(lines[:index])))


# docs/npu-spec.md section 12, word for word
SPEC_EXAMPLE = """
    SET_DMA_SRC region0, 0
    SET_DMA_DST sram, 0x100
    SET_DMA_LEN 0x70
    OP_DMA
    SET_DMA_SRC region1, 0
    SET_DMA_DST sram, 0
    SET_DMA_LEN 64
    OP_DMA
    SET_IFM_BASE 0
    SET_IFM_SHAPE 8, 8, 1
    SET_IFM_ZP -128
    SET_WEIGHT_BASE 0x100
    SET_BIAS_BASE 0x140
    SET_RQ_TABLE 0x150
    SET_KERNEL 3, 3
    SET_PAD 1, 1, 1, 1
    SET_OFM_BASE 0x200
    SET_OFM_SHAPE 8, 8, 4
    OP_DMA_WAIT 0
    OP_CONV
    SET_DMA_SRC sram, 0x200
    SET_DMA_DST region1, 0x40
    SET_DMA_LEN 256
    OP_KERNEL_WAIT 0
    OP_DMA
    OP_STOP
"""
SPEC_WORDS = [
    0x00004010, 0, 0x01004011, 0x100, 0x00004012, 0x70, 0x00000010,
    0x00014010, 0, 0x01004011, 0, 0x00004012, 0x40, 0x00000010,
    0x00004000, 0, 0x00014006, 0x00080008, 0xFF800100,
    0x00004003, 0x100, 0x00004004, 0x140, 0x00004005, 0x150,
    0x03030103, 0x11110105, 0x00004002, 0x200, 0x00044007, 0x00080008,
    0x00000011, 0x00000002, 0x01004010, 0x200, 0x00014011, 0x40,
    0x00004012, 0x100, 0x00000012, 0x00000010, 0x00000000,
]


def check_spec_example(machine):
    stream = assemble(SPEC_EXAMPLE)
    problems = []
    if stream != struct.pack(f"<{len(SPEC_WORDS)}I", *SPEC_WORDS):
        problems.append("assembled words differ from the spec table")
    if len(stream) != 0xA8:
        problems.append(f"QSIZE {len(stream):#x} != 0xa8")
    regions = [(0, 0x70, bytes(0x70)), (1, 0x140, bytes(64))]
    res, _ = machine.run(stream, regions)
    want = {"err_code": 0, "macs": 2304, "dma_rd_bytes": 176,
            "dma_wr_bytes": 256, "ops": 4}
    problems += [f"{k} {res[k]} != {v}" for k, v in want.items()
                 if res[k] != v]

    # Without the OP_DMA_WAIT at 0x7C, OP_CONV fails there (spec 12)
    lines = SPEC_EXAMPLE.strip().splitlines()
    no_wait = assemble("\n".join(ln for ln in lines if "DMA_WAIT" not in ln))
    res, _ = machine.run(no_wait, regions)
    if (res["err_code"], res["err_offset"]) != (SYNC, 0x7C):
        problems.append("without OP_DMA_WAIT: error "
                        f"{res['err_code']:#x} at {res['err_offset']:#x}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--knpu-run", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    machine = Machine(args.knpu_run, args.work_dir)
    failures = 0
    for name, lines, code, index, kwargs in CASES:
        stream = assemble("\n".join(lines))
        res, _ = machine.run(stream, **kwargs)
        if index is None:
            offset = 0
        elif index == "qsize":
            offset = len(stream)
        else:
            offset = offset_of(lines, index)
        got = (res["err_code"], res["err_offset"])
        if got == (code, offset):
            print(f"PASS {name}: error {code:#x} at {offset:#x}")
        else:
            failures += 1
            print(f"FAIL {name}: expected error {code:#x} at {offset:#x}, "
                  f"got {got[0]:#x} at {got[1]:#x}")

    problems = check_spec_example(machine)
    if problems:
        failures += 1
        print("FAIL spec example: " + "; ".join(problems))
    else:
        print("PASS spec example: words, counters and the missing wait")

    total = len(CASES) + 1
    print(f"{total - failures} of {total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
