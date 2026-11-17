#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V3: int8 convolutions on the shared core against TFLite Micro (V2).

For every case the test makes seeded random tensors and per-channel
requantization, asks tflm_golden_<mode> for the reference output, runs a
command stream (DMA in, OP_CONV, DMA out) on knpu_run and compares the
output bytes and the performance counters. The cases cover 1x1 and 3x3
kernels, stride 1 and 2, VALID and SAME padding and both rounding modes,
plus edge values of the requantization (including multipliers that put
many results exactly on .5, where the two rounding modes differ), a
non-square kernel and 2-D DMA.
"""

import argparse
import os
import random
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knpu_asm import assemble  # noqa: E402

QBASE = 0x8000_0000
REGION_ADDR = [0x1_0000_0000 + n * 0x1000_0000 for n in range(8)]


class Machine:
    """Runs one job on knpu_run with regions placed in a sparse memory."""

    def __init__(self, knpu_run, work_dir):
        self.knpu_run = knpu_run
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.count = 0

    def _file(self, name, data=None):
        path = os.path.join(self.work_dir, f"{self.count}-{name}")
        if data is not None:
            with open(path, "wb") as f:
                f.write(data)
        return path

    def run(self, stream, regions=(), dumps=(), qbase=QBASE, qsize=None,
            queue_mem=None, region_addr=None, sram_size=None):
        """Run @stream.

        regions: (n, size, data) tuples; region n is a --mem block at
        REGION_ADDR[n] (or region_addr[n]) holding data, zero-filled to
        size. dumps: (n, offset, size) tuples read back after the job.
        Returns (result dict, list of dumped bytes).
        """
        self.count += 1
        addr = list(REGION_ADDR)
        for n, a in (region_addr or {}).items():
            addr[n] = a
        qsize = len(stream) if qsize is None else qsize
        qmem = len(stream) if queue_mem is None else queue_mem
        queue = stream[:qmem] + bytes(max(qmem - len(stream), 0))
        cmd = [self.knpu_run,
               "--queue", f"{qbase:#x}:{qsize:#x}",
               "--mem", f"{QBASE:#x}:{qmem:#x}:"
               + self._file("stream.bin", queue)]
        if sram_size is not None:
            cmd += ["--sram-size", str(sram_size)]
        for n in range(8):
            cmd += ["--region", f"{n}:{addr[n]:#x}"]
        for n, size, data in regions:
            cmd += ["--mem", f"{REGION_ADDR[n]:#x}:{size:#x}:"
                    + self._file(f"region{n}.bin", data)]
        outs = []
        for i, (n, off, size) in enumerate(dumps):
            path = self._file(f"dump{i}.bin")
            outs.append(path)
            cmd += ["--dump", f"{REGION_ADDR[n] + off:#x}:{size:#x}:{path}"]
        out = subprocess.run(cmd, check=True, capture_output=True,
                             text=True).stdout
        result = {k: int(v, 0) for k, v in
                  (f.split("=") for f in out.split())}
        data = []
        for path in outs:
            with open(path, "rb") as f:
                data.append(f.read())
        return result, data


def same_padding(size, k, s):
    out = (size + s - 1) // s
    total = max((out - 1) * s + k - size, 0)
    return out, total // 2, total - total // 2


def valid_padding(size, k, s):
    return (size - k) // s + 1, 0, 0


class Case:
    def __init__(self, name, rng, h, w, ic, oc, k, s, padding, mode,
                 edge=None, dma_2d=False):
        self.name, self.mode, self.dma_2d = name, mode, dma_2d
        self.h, self.w, self.ic, self.oc = h, w, ic, oc
        self.kh, self.kw = k
        self.sh, self.sw = s
        pad = same_padding if padding == "SAME" else valid_padding
        self.oh, self.pt, self.pb = pad(h, self.kh, self.sh)
        self.ow, self.pl, self.pr = pad(w, self.kw, self.sw)

        self.ifm_zp = rng.randint(-128, 127)
        self.ofm_zp = rng.randint(-128, 127)
        if edge == "relu":
            self.clamp = (self.ofm_zp, 127)
        elif edge == "narrow":
            self.clamp = (-20, 30)
        else:
            self.clamp = (-128, 127)

        self.input = bytes(rng.randrange(256) for _ in range(h * w * ic))
        wmax = 2 if edge in ("positive-shift", "ties") else 127
        self.weights = struct.pack(
            f"{oc * self.kh * self.kw * ic}b",
            *(rng.randint(-wmax, wmax)
              for _ in range(oc * self.kh * self.kw * ic)))
        bias, rq = [], []
        for c in range(oc):
            if edge == "positive-shift":
                bias.append(rng.randint(-1000, 1000))
                m = rng.randint(1 << 30, (1 << 31) - 1)
                sh = rng.choice([1, 2, 0])
            elif edge == "ties":
                # m = 0.5 and small accumulators put many results on .5,
                # where the two rounding modes differ
                bias.append(rng.randint(-64, 64))
                m = 1 << 30
                sh = -rng.randint(1, 3)
            elif edge == "extreme":
                bias.append(rng.randint(-(1 << 20), 1 << 20))
                m = rng.choice([0, 1, (1 << 31) - 1, 1 << 30])
                sh = rng.choice([0, -1, -31, -15])
            else:
                bias.append(rng.randint(-(1 << 15), 1 << 15))
                m = rng.randint(1 << 30, (1 << 31) - 1)
                sh = -rng.randint(7, 12)
            rq += [m, sh]
        self.bias = struct.pack(f"<{oc}i", *bias)
        self.rq = struct.pack(f"<{2 * oc}i", *rq)

    @property
    def macs(self):
        return self.oh * self.ow * self.oc * self.kh * self.kw * self.ic

    def golden(self, tool, work_dir):
        files = {}
        for key in ("input", "weights", "bias", "rq"):
            files[key] = os.path.join(work_dir, f"{self.name}-{key}.bin")
            with open(files[key], "wb") as f:
                f.write(getattr(self, key))
        out = os.path.join(work_dir, f"{self.name}-golden.bin")
        subprocess.run([
            tool,
            "--ifm", f"{self.h},{self.w},{self.ic}",
            "--ofm", f"{self.oh},{self.ow},{self.oc}",
            "--kernel", f"{self.kh},{self.kw}",
            "--stride", f"{self.sh},{self.sw}",
            "--pad", f"{self.pt},{self.pl}",
            "--ifm-zp", str(self.ifm_zp), "--ofm-zp", str(self.ofm_zp),
            "--clamp", f"{self.clamp[0]},{self.clamp[1]}",
            "--input", files["input"], "--weights", files["weights"],
            "--bias", files["bias"], "--rq", files["rq"], "--output", out,
        ], check=True)
        with open(out, "rb") as f:
            return f.read()

    def program(self):
        """Region 0 constants, region 1 input then output (spec 12)."""
        align = lambda n: (n + 15) & ~15  # noqa: E731
        w_size = len(self.weights)
        b_off = align(w_size)
        rq_off = b_off + len(self.bias)
        consts = (self.weights + bytes(b_off - w_size) + self.bias + self.rq)

        in_row = self.w * self.ic
        out_row = self.ow * self.oc
        # 2-D DMA: rows sit in the region with a gap after each one
        in_pitch = in_row + 7 if self.dma_2d else in_row
        out_pitch = out_row + 5 if self.dma_2d else out_row
        arena_in = b"".join(self.input[y * in_row:(y + 1) * in_row]
                            + b"\xa5" * (in_pitch - in_row)
                            for y in range(self.h))
        out_off = align(len(arena_in))

        sram_in, sram_c = 0, align(len(self.input))
        sram_out = align(sram_c + len(consts))
        dma_in = (f"SET_DMA_LEN {in_row}\nSET_DMA_COUNT {self.h}\n"
                  f"SET_DMA_STRIDE {in_pitch}" if self.dma_2d else
                  f"SET_DMA_LEN {len(self.input)}\nSET_DMA_COUNT 1")
        dma_out = (f"SET_DMA_LEN {out_row}\nSET_DMA_COUNT {self.oh}\n"
                   f"SET_DMA_STRIDE {out_pitch}" if self.dma_2d else
                   f"SET_DMA_LEN {self.oh * out_row}\nSET_DMA_COUNT 1")
        src = f"""
            SET_DMA_SRC region0, 0
            SET_DMA_DST sram, {sram_c}
            SET_DMA_LEN {len(consts)}
            OP_DMA
            SET_DMA_SRC region1, 0
            SET_DMA_DST sram, {sram_in}
            {dma_in}
            OP_DMA
            SET_IFM_BASE {sram_in}
            SET_IFM_SHAPE {self.h}, {self.w}, {self.ic}
            SET_IFM_ZP {self.ifm_zp}
            SET_WEIGHT_BASE {sram_c}
            SET_BIAS_BASE {sram_c + b_off}
            SET_RQ_TABLE {sram_c + rq_off}
            SET_KERNEL {self.kh}, {self.kw}
            SET_STRIDE {self.sh}, {self.sw}
            SET_PAD {self.pt}, {self.pl}, {self.pb}, {self.pr}
            SET_OFM_BASE {sram_out}
            SET_OFM_SHAPE {self.oh}, {self.ow}, {self.oc}
            SET_OFM_ZP {self.ofm_zp}
            SET_CLAMP {self.clamp[0]}, {self.clamp[1]}
            SET_ROUND_MODE {self.mode}
            OP_DMA_WAIT 0
            OP_CONV
            SET_DMA_SRC sram, {sram_out}
            SET_DMA_DST region1, {out_off}
            {dma_out}
            OP_KERNEL_WAIT 0
            OP_DMA
            OP_STOP
        """
        out_size = (self.oh - 1) * out_pitch + out_row
        regions = [(0, len(consts), consts),
                   (1, out_off + self.oh * out_pitch, arena_in)]
        dumps = [(1, out_off, out_size)]
        return assemble(src), regions, dumps, out_row, out_pitch, consts


def make_cases(seed):
    rng = random.Random(seed)
    cases = []
    for mode in ("double", "single"):
        for k in (1, 3):
            for s in (1, 2):
                for padding in ("VALID", "SAME"):
                    for i in range(2):
                        h, w = rng.randint(k, 11), rng.randint(k, 11)
                        cases.append(Case(
                            f"{mode}-k{k}-s{s}-{padding}-{i}", rng, h, w,
                            rng.randint(1, 16), rng.randint(1, 16),
                            (k, k), (s, s), padding, mode))
        for edge in ("relu", "narrow", "extreme", "positive-shift", "ties"):
            cases.append(Case(f"{mode}-{edge}", rng, 7, 6, 5, 9, (3, 3),
                              (1, 1), "SAME", mode, edge=edge))
        cases.append(Case(f"{mode}-k3x1-s2x1", rng, 9, 8, 4, 6, (3, 1),
                          (2, 1), "SAME", mode))
        cases.append(Case(f"{mode}-dma2d", rng, 6, 5, 3, 4, (3, 3), (2, 2),
                          "SAME", mode, dma_2d=True))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--knpu-run", required=True)
    parser.add_argument("--golden-double", required=True)
    parser.add_argument("--golden-single", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()

    machine = Machine(args.knpu_run, args.work_dir)
    golden_tool = {"double": args.golden_double,
                   "single": args.golden_single}
    failures = 0
    cases = make_cases(args.seed)
    for case in cases:
        expected = case.golden(golden_tool[case.mode], args.work_dir)
        stream, regions, dumps, out_row, out_pitch, consts = case.program()
        res, (raw,) = machine.run(stream, regions, dumps)
        got = b"".join(raw[y * out_pitch:y * out_pitch + out_row]
                       for y in range(case.oh))
        gaps = b"".join(raw[y * out_pitch + out_row:(y + 1) * out_pitch]
                        for y in range(case.oh - 1))
        problems = []
        if res["err_code"] != 0:
            problems.append(f"error {res['err_code']:#x} at "
                            f"{res['err_offset']:#x}")
        if got != expected:
            diff = sum(a != b for a, b in zip(got, expected))
            problems.append(f"{diff} of {len(expected)} output bytes differ")
        if gaps.strip(b"\0"):
            problems.append("2-D DMA wrote between output rows")
        want = {"macs": case.macs,
                "dma_rd_bytes": len(consts) + len(case.input),
                "dma_wr_bytes": case.oh * out_row, "ops": 4}
        for key, value in want.items():
            if res[key] != value:
                problems.append(f"{key} {res[key]} != {value}")
        if problems:
            failures += 1
            print(f"FAIL {case.name}: " + "; ".join(problems))
        else:
            print(f"PASS {case.name}: {case.h}x{case.w}x{case.ic} -> "
                  f"{case.oh}x{case.ow}x{case.oc}, "
                  f"pad {case.pt},{case.pl},{case.pb},{case.pr}")
    print(f"{len(cases) - failures} of {len(cases)} cases match TFLite Micro")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
