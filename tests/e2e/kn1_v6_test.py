#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V6: end to end on kn1-evb (docs/design.md 11).

Boots the Buildroot images through the firmware chain like the boot smoke
test, then in the guest:

  1. loads the knpu module
  2. runs knpu-infer on the four test models in /usr/share/knpu/conv/
     (double and single rounding, compiled for a 64 KiB and a 4 KiB SRAM)
     with the five inputs, comparing every output with TFLite Micro's
     (--expect), and checks the job's counters: the nominal MACs and the
     written bytes of the layer whatever the tiling, and one OP per OP
     command of the stream
  3. checks that an output of the other rounding mode is told apart, and
     that a truncated model file is rejected
  4. checks that the kernel logged no warning, then powers off

UART0 goes through a UNIX socket and UART1 to a file, both kept in the log
directory as v6-uart0.log and v6-uart1.log.
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "boot"))
import kn1_boot_test as boot  # noqa: E402

check = boot.check
TestFailure = boot.TestFailure

DIR = "/usr/share/knpu/conv"
MODES = ("double", "single")
SRAMS = ("64k", "4k")
INPUTS = 5

# The test model (models/conv/make_model.py): 14x14x8 -> 16, 3x3, SAME
MACS = 14 * 14 * 16 * 3 * 3 * 8
OUT_BYTES = 14 * 14 * 16


def infer(con, args):
    """Run knpu-infer; return its key=value fields and exit status."""
    out = con.run(f"knpu-infer {args}; echo status=$?", timeout=60)
    fields = {}
    for line in out.splitlines():
        for word in line.split():
            if "=" in word:
                key, value = word.split("=", 1)
                fields[key] = value
    return fields, out


def run_model(con, model, mode):
    ops = set()
    for n in range(INPUTS):
        v, out = infer(con, f"{DIR}/{model} {DIR}/input-{n}.bin "
                            f"--expect {DIR}/expected-{mode}-{n}.bin")
        what = f"{model} input {n}"
        if v.get("status") != "0" or v.get("match") != "yes":
            print(out)
        check(v.get("status") == "0" and v.get("match") == "yes",
              f"{what}: output equals TFLite Micro ({mode} rounding)")
        check(v.get("macs") == str(MACS) and
              v.get("dma_wr_bytes") == str(OUT_BYTES) and
              v.get("ops") == v.get("model_ops"),
              f"{what}: counters macs={v.get('macs')} "
              f"dma_wr_bytes={v.get('dma_wr_bytes')} ops={v.get('ops')} "
              f"(model_ops={v.get('model_ops')})")
        ops.add((v.get("dma_rd_bytes"), v.get("ops")))
    check(len(ops) == 1, f"{model}: the same counters for every input")


def test_v6(args):
    uart0_log = os.path.join(args.log_dir, "v6-uart0.log")
    uart1_log = os.path.join(args.log_dir, "v6-uart1.log")
    with tempfile.TemporaryDirectory(prefix="kn1-v6-") as tmp, \
            open(uart0_log, "wb") as log:
        sock_path = os.path.join(tmp, "uart0.sock")
        qemu = boot.start_qemu(args, sock_path, uart1_log)
        sock = None
        try:
            sock = boot.connect(qemu, sock_path)
            con = boot.Console(qemu, sock, log)
            boot.boot_to_shell(con, args.timeout, "boot")

            out = con.run("modprobe knpu; echo rc=$?; "
                          "test -c /dev/accel/accel0 && echo node=yes")
            check("rc=0" in out and "node=yes" in out,
                  "modprobe knpu creates /dev/accel/accel0")

            for mode in MODES:
                for sram in SRAMS:
                    run_model(con, f"conv-{mode}-{sram}.knpu", mode)

            # Inputs 0 and 2 give different outputs in the two rounding
            # modes, so a model run with the wrong mode is caught
            v, _ = infer(con, f"{DIR}/conv-double-64k.knpu "
                              f"{DIR}/input-0.bin "
                              f"--expect {DIR}/expected-single-0.bin")
            check(v.get("status") == "1" and v.get("match") == "no",
                  "the double-rounding model does not match the "
                  f"single-rounding output ({v.get('differing')} bytes "
                  "differ)")

            con.run(f"head -c 1000 {DIR}/conv-double-64k.knpu "
                    "> /tmp/truncated.knpu")
            v, out = infer(con, f"/tmp/truncated.knpu {DIR}/input-0.bin")
            check(v.get("status") == "2" and "bad model file" in out,
                  "knpu-infer rejects a truncated model file")

            out = con.run("echo warnings=$(dmesg | grep -cE "
                          "'WARNING:|BUG:|Oops|refcount_t:')")
            check("warnings=0" in out, "the kernel logged no warnings")

            con.send("poweroff")
            status = con.wait_exit(120)
            check(status == 0, f"poweroff ends QEMU (status {status})")
        finally:
            if sock:
                sock.close()
            if qemu.poll() is None:
                qemu.kill()
                qemu.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default="output/images",
                        help="Buildroot images directory "
                             "(default: %(default)s)")
    parser.add_argument("--host-bin", default="output/host/bin",
                        help="directory holding qemu-system-aarch64 "
                             "(default: %(default)s)")
    parser.add_argument("--log-dir", default="output/test-logs",
                        help="where to write v6-uart0.log and v6-uart1.log "
                             "(default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds allowed for the boot "
                             "(default: %(default)s)")
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    try:
        for path in (os.path.join(args.host_bin, "qemu-system-aarch64"),
                     os.path.join(args.images, "flash.bin"),
                     os.path.join(args.images, "disk.img")):
            if not os.path.exists(path):
                raise TestFailure(f"{path} does not exist")
        test_v6(args)
    except TestFailure as e:
        print(f"FAIL: {e}")
        print(f"Console logs are in {args.log_dir}/")
        return 1
    print("All V6 checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
