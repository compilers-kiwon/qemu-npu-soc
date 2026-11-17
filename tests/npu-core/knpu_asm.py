#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Assembler for kn1-npu command streams (docs/npu-spec.md v0).

One command per line, operands separated by commas, '#' starts a comment:

    SET_DMA_SRC region0, 0x0      # space: sram or region0..region7
    SET_DMA_DST sram, 0x100
    SET_DMA_LEN 0x70
    OP_DMA
    SET_IFM_SHAPE 8, 8, 1         # H, W, C
    SET_PAD 1, 1, 1, 1            # top, left, bottom, right
    SET_ROUND_MODE single         # double or single
    OP_DMA_WAIT 0
    .word 0xdeadbeef              # raw word, for tests of bad streams

The assembler only checks that each value fits its field. Whether the
value is allowed (a zero kernel size, an unaligned bias) is left to the
device, so tests can build invalid streams.

As a script: knpu_asm.py INPUT.s -o OUTPUT.bin
"""

import argparse
import struct
import sys

# name: (CMD, operand kinds); see spec 6.1
COMMANDS = {
    "OP_STOP": (0x0000, ""),
    "OP_NOP": (0x0001, ""),
    "OP_CONV": (0x0002, ""),
    "OP_ADD": (0x0003, ""),
    "OP_AVGPOOL": (0x0004, ""),
    "OP_DMA": (0x0010, ""),
    "OP_DMA_WAIT": (0x0011, "count"),
    "OP_KERNEL_WAIT": (0x0012, "count"),
    "SET_IFM_ZP": (0x0100, "zp"),
    "SET_IFM2_ZP": (0x0101, "zp"),
    "SET_OFM_ZP": (0x0102, "zp"),
    "SET_KERNEL": (0x0103, "pair8"),
    "SET_STRIDE": (0x0104, "pair8"),
    "SET_PAD": (0x0105, "pad"),
    "SET_CLAMP": (0x0106, "clamp"),
    "SET_ROUND_MODE": (0x0107, "round"),
    "SET_DMA_COUNT": (0x0108, "u16"),
    "SET_IFM_BASE": (0x4000, "u32"),
    "SET_IFM2_BASE": (0x4001, "u32"),
    "SET_OFM_BASE": (0x4002, "u32"),
    "SET_WEIGHT_BASE": (0x4003, "u32"),
    "SET_BIAS_BASE": (0x4004, "u32"),
    "SET_RQ_TABLE": (0x4005, "u32"),
    "SET_IFM_SHAPE": (0x4006, "shape"),
    "SET_OFM_SHAPE": (0x4007, "shape"),
    "SET_IFM_SCALE": (0x4008, "scale"),
    "SET_IFM2_SCALE": (0x4009, "scale"),
    "SET_OFM_SCALE": (0x400A, "scale"),
    "SET_DMA_SRC": (0x4010, "space"),
    "SET_DMA_DST": (0x4011, "space"),
    "SET_DMA_LEN": (0x4012, "u32"),
    "SET_DMA_STRIDE": (0x4013, "u32"),
}

PAYLOAD = 0x4000
ROUND = {"double": 0, "single": 1}


class AsmError(Exception):
    pass


def _int(text):
    try:
        return int(text, 0)
    except ValueError:
        raise AsmError(f"not a number: {text!r}")


def _field(value, lo, hi, what):
    if not lo <= value <= hi:
        raise AsmError(f"{what} {value} is outside [{lo}, {hi}]")
    return value


def _space(text):
    text = text.lower()
    if text == "sram":
        return 0x100
    if text.startswith("region"):
        return _field(_int(text[6:]), 0, 255, "region")
    raise AsmError(f"unknown DMA space {text!r}")


def _encode(kind, ops):
    """Return (param, payload) for the operand kind."""
    counts = {"": 0, "count": 1, "zp": 1, "u16": 1, "u32": 1, "round": 1,
              "pair8": 2, "clamp": 2, "scale": 2, "space": 2, "shape": 3,
              "pad": 4}
    if len(ops) != counts[kind]:
        raise AsmError(f"expected {counts[kind]} operands, got {len(ops)}")
    if kind == "":
        return 0, None
    if kind == "round":
        if ops[0].lower() not in ROUND:
            raise AsmError("rounding mode must be double or single")
        return ROUND[ops[0].lower()], None
    if kind == "space":
        return _space(ops[0]), _field(_int(ops[1]), 0, 2**32 - 1, "offset")
    v = [_int(o) for o in ops]
    if kind == "count":
        return _field(v[0], 0, 15, "count"), None
    if kind == "zp":
        return _field(v[0], -32768, 32767, "zero point") & 0xFFFF, None
    if kind == "u16":
        return _field(v[0], 0, 0xFFFF, "value"), None
    if kind == "u32":
        return 0, _field(v[0], 0, 2**32 - 1, "value")
    if kind == "pair8":
        return (_field(v[0], 0, 255, "value") << 8 |
                _field(v[1], 0, 255, "value")), None
    if kind == "clamp":
        lo = _field(v[0], -128, 127, "clamp minimum") & 0xFF
        hi = _field(v[1], -128, 127, "clamp maximum") & 0xFF
        return hi << 8 | lo, None
    if kind == "scale":
        return (_field(v[1], -32768, 32767, "shift") & 0xFFFF,
                _field(v[0], 0, 2**32 - 1, "multiplier"))
    if kind == "shape":
        h, w, c = v
        return (_field(c, 0, 0xFFFF, "C"),
                _field(h, 0, 0xFFFF, "H") << 16 | _field(w, 0, 0xFFFF, "W"))
    if kind == "pad":
        p = [_field(x, 0, 15, "padding") for x in v]
        return p[0] | p[1] << 4 | p[2] << 8 | p[3] << 12, None
    raise AssertionError(kind)


def assemble(source):
    """Assemble source text and return the command stream as bytes."""
    words = []
    for lineno, line in enumerate(source.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, rest = line.partition(" ")
        ops = [o.strip() for o in rest.split(",")] if rest.strip() else []
        try:
            if name == ".word":
                words += [_field(_int(o), 0, 2**32 - 1, "word") for o in ops]
                continue
            if name.upper() not in COMMANDS:
                raise AsmError(f"unknown command {name!r}")
            cmd, kind = COMMANDS[name.upper()]
            param, payload = _encode(kind, ops)
        except AsmError as e:
            raise AsmError(f"line {lineno}: {e}") from None
        words.append(param << 16 | cmd)
        if cmd & PAYLOAD:
            words.append(payload)
    return struct.pack(f"<{len(words)}I", *words)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    with open(args.input) as f:
        source = f.read()
    try:
        data = assemble(source)
    except AsmError as e:
        print(f"{args.input}: {e}", file=sys.stderr)
        return 1
    with open(args.output, "wb") as f:
        f.write(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
