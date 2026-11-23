#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Reader of model.knpu v0, written from docs/model-format.md alone.

It checks the rules of section 11 of that document except those that need
the device (2, 3) or the loaded CPU code (7), so it also tests the
compiler's writer against the format document.

    model_file.py model.knpu      # print a summary
"""

import re
import struct
import sys

MAGIC = b"KNPU"
CMDS, REGIONS, CONST, TENSORS, CPU_CODE, CPU_STEPS = 1, 2, 3, 4, 5, 6
KINDS = {CMDS: "CMDS", REGIONS: "REGIONS", CONST: "CONST",
         TENSORS: "TENSORS", CPU_CODE: "CPU_CODE", CPU_STEPS: "CPU_STEPS"}
NO_CONTENT = 0xFFFFFFFF
STEP_KINDS = {0: "copy_in", 1: "call", 2: "copy_out"}
ET_DYN, EM_AARCH64 = 3, 183


class FormatError(Exception):
    pass


class Region:
    def __init__(self, index, role, size, content):
        self.index, self.role, self.size = index, role, size
        self.content = content  # bytes of a constant region, else None


class Tensor:
    def __init__(self, direction, region, offset, size, shape, scale,
                 zero_point):
        self.direction = direction  # "input" or "output"
        self.region, self.offset, self.size = region, offset, size
        self.shape, self.scale, self.zero_point = shape, scale, zero_point


class Step:
    def __init__(self, kind, region, name, offset, size):
        self.kind = kind  # "copy_in", "call" or "copy_out"
        self.region, self.name = region, name
        self.offset, self.size = offset, size


class Model:
    def __init__(self):
        self.format_minor = 0
        self.sram_size = 0
        self.spec_major = 0
        self.cmds = b""
        self.regions = []
        self.tensors = []
        self.cpu_code = None  # bytes of the shared object, if any
        self.cpu_steps = []

    def words(self):
        return list(struct.unpack(f"<{len(self.cmds) // 4}I", self.cmds))

    def region(self, index):
        return next(r for r in self.regions if r.index == index)


def _check(cond, msg):
    if not cond:
        raise FormatError(msg)


def read(data):
    """Parses and checks model.knpu bytes; returns a Model."""
    m = Model()
    _check(len(data) > 32, "file is not larger than the header")
    (magic, major, minor, sram, spec_major, nsec, table,
     reserved) = struct.unpack_from("<4sHHIHHI12s", data, 0)
    _check(magic == MAGIC, "bad magic")
    _check(major == 0, f"unknown format_major {major}")
    _check(reserved == bytes(12), "reserved header bytes are not 0")
    _check(sram % 4096 == 0 and 4096 <= sram <= 16 << 20,
           f"sram_size {sram} out of range")
    _check(nsec >= 1, "no sections")
    _check(table % 16 == 0, "section table is not 16-byte aligned")
    m.format_minor, m.sram_size, m.spec_major = minor, sram, spec_major

    # Section table and layout (sections 2 and 4)
    _check(table + 16 * nsec <= len(data), "section table outside the file")
    spans = [(0, 32, "header"), (table, table + 16 * nsec, "section table")]
    sections = []
    for i in range(nsec):
        kind, off, size, res = struct.unpack_from("<IIII", data,
                                                  table + 16 * i)
        _check(res == 0, f"section {i}: reserved field is not 0")
        _check(kind in KINDS, f"section {i}: unknown kind {kind}")
        _check(off % 16 == 0, f"section {i}: offset not 16-byte aligned")
        _check(off + size <= len(data), f"section {i}: outside the file")
        spans.append((off, off + size, f"section {i}"))
        sections.append((kind, data[off:off + size]))
    spans.sort()
    for (b0, e0, n0), (b1, e1, n1) in zip(spans, spans[1:]):
        _check(e0 <= b1, f"{n0} overlaps {n1}")
        _check(data[e0:b1] == bytes(b1 - e0), f"gap after {n0} is not 0")
    _check(spans[-1][1] == len(data),
           "file does not end at the last section")

    counts = {k: sum(1 for kind, _ in sections if kind == k) for k in KINDS}
    _check(counts[CMDS] == 1 and counts[REGIONS] == 1 and
           counts[TENSORS] == 1, "need exactly one CMDS, REGIONS, TENSORS")
    _check(counts[CPU_CODE] == counts[CPU_STEPS] <= 1,
           "need both or none of CPU_CODE and CPU_STEPS, at most one each")
    _check(counts[CPU_CODE] == 0 or minor >= 1,
           "CPU sections need format_minor 1")

    by_kind = {kind: body for kind, body in sections}
    # CMDS (section 5)
    m.cmds = by_kind[CMDS]
    _check(len(m.cmds) >= 4 and len(m.cmds) % 4 == 0,
           "CMDS size must be a positive multiple of 4")
    _check(m.words()[-1] == 0x0000_0000, "CMDS must end with OP_STOP")

    # REGIONS (section 6)
    body = by_kind[REGIONS]
    _check(len(body) % 16 == 0, "REGIONS size is not a multiple of 16")
    used_const = []
    last = -1
    for i in range(0, len(body), 16):
        index, role, size, content = struct.unpack_from("<IIII", body, i)
        _check(index <= 7, f"region index {index} out of range")
        _check(index > last, "regions not in increasing index order")
        last = index
        _check(role in (0, 1), f"region {index}: bad role {role}")
        _check(size >= 1, f"region {index}: size 0")
        if role == 0:
            _check(content < nsec and sections[content][0] == CONST,
                   f"region {index}: content is not a CONST section")
            blob = sections[content][1]
            _check(len(blob) == size,
                   f"region {index}: size differs from its CONST section")
            used_const.append(content)
            m.regions.append(Region(index, "const", size, blob))
        else:
            _check(content == NO_CONTENT,
                   f"region {index}: an arena has no content")
            m.regions.append(Region(index, "arena", size, None))
    const_sections = [i for i, (k, _) in enumerate(sections) if k == CONST]
    _check(sorted(used_const) == const_sections,
           "every CONST section belongs to exactly one region")

    # TENSORS (section 8)
    body = by_kind[TENSORS]
    _check(len(body) % 48 == 0, "TENSORS size is not a multiple of 48")
    seen_output = False
    for i in range(0, len(body), 48):
        (direction, ttype, rank, res8, region, offset, size, d0, d1, d2, d3,
         scale, zp, res) = struct.unpack_from("<BBBBIII4IfiQ", body, i)
        dims = [d0, d1, d2, d3]
        _check(res8 == 0 and res == 0, "reserved tensor bytes are not 0")
        _check(direction in (0, 1), "bad tensor direction")
        _check(direction == 1 or not seen_output,
               "inputs must come before outputs")
        seen_output |= direction == 1
        _check(ttype == 0, "tensor type must be int8")
        _check(1 <= rank <= 4, "tensor rank out of range")
        _check(all(d >= 1 for d in dims[:rank]) and
               all(d == 0 for d in dims[rank:]), "bad tensor shape")
        n = 1
        for d in dims[:rank]:
            n *= d
        _check(size == n, "tensor size differs from its shape")
        _check(scale > 0 and scale != float("inf"), "bad tensor scale")
        _check(-128 <= zp <= 127, "tensor zero point out of range")
        regions = [r for r in m.regions if r.index == region]
        _check(regions and regions[0].role == "arena",
               "tensor is not in an arena region")
        _check(offset + size <= regions[0].size, "tensor outside its region")
        m.tensors.append(Tensor("output" if direction else "input", region,
                                offset, size, dims[:rank], scale, zp))
    ins = [t for t in m.tensors if t.direction == "input"]
    outs = [t for t in m.tensors if t.direction == "output"]
    for a in ins:
        for b in outs:
            _check(a.region != b.region or a.offset + a.size <= b.offset or
                   b.offset + b.size <= a.offset,
                   "an input overlaps an output")

    if counts[CPU_CODE]:
        read_cpu(m, by_kind[CPU_CODE], by_kind[CPU_STEPS])
    return m


def read_cpu(m, code, body):
    """CPU_CODE and CPU_STEPS (sections 9 and 10)."""
    _check(len(code) >= 64 and code[:4] == b"\x7fELF",
           "CPU_CODE is not an ELF file")
    _check(code[4] == 2 and code[5] == 1,
           "CPU_CODE is not 64-bit little-endian ELF")
    e_type, e_machine = struct.unpack_from("<HH", code, 16)
    _check(e_type == ET_DYN, "CPU_CODE is not a shared object")
    _check(e_machine == EM_AARCH64, "CPU_CODE is not for AArch64")
    m.cpu_code = code

    _check(len(body) >= 16, "CPU_STEPS is smaller than its header")
    count, = struct.unpack_from("<I", body, 0)
    _check(body[4:16] == bytes(12), "reserved CPU_STEPS bytes are not 0")
    _check(count >= 1, "no CPU steps")
    names_at = 16 + 16 * count
    _check(len(body) > names_at, "CPU_STEPS has no name table")
    names = body[names_at:]
    _check(names[-1] == 0, "the name table does not end with NUL")
    for i in range(count):
        kind, region, res, name, offset, size = struct.unpack_from(
            "<BBHIII", body, 16 + 16 * i)
        _check(kind in STEP_KINDS, f"step {i}: bad kind {kind}")
        _check(res == 0, f"step {i}: reserved field is not 0")
        _check(name < len(names), f"step {i}: name outside the name table")
        sym = names[name:names.index(0, name)].decode("ascii", "replace")
        _check(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", sym) is not None,
               f"step {i}: name {sym!r} is not a C identifier")
        if STEP_KINDS[kind] == "call":
            _check(region == 0 and offset == 0 and size == 0,
                   f"step {i}: a call has region, offset and size 0")
        else:
            regions = [r for r in m.regions if r.index == region]
            _check(regions and regions[0].role == "arena",
                   f"step {i}: not in an arena region")
            _check(size >= 1 and offset + size <= regions[0].size,
                   f"step {i}: copy outside its region")
        m.cpu_steps.append(Step(STEP_KINDS[kind], region, sym, offset, size))


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: model_file.py model.knpu")
    with open(sys.argv[1], "rb") as f:
        m = read(f.read())
    print(f"model.knpu 0.{m.format_minor}: sram_size {m.sram_size}, "
          f"spec {m.spec_major}, {len(m.cmds)} bytes of commands")
    for r in m.regions:
        print(f"region {r.index}: {r.role}, {r.size} bytes")
    for t in m.tensors:
        print(f"{t.direction}: region {t.region} offset {t.offset} "
              f"shape {t.shape} scale {t.scale!r} zero_point {t.zero_point}")
    if m.cpu_code is not None:
        print(f"CPU code: {len(m.cpu_code)} bytes")
    for s in m.cpu_steps:
        print(f"CPU step {s.kind} {s.name}" + ("" if s.kind == "call" else
              f" region {s.region} offset {s.offset} size {s.size}"))


if __name__ == "__main__":
    main()
