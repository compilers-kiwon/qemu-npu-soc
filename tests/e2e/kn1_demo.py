#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Demo: the ResNet-8 classifies ten photographs on kn1-evb.

Boots the Buildroot images through the firmware chain without a human,
like the boot smoke test, loads the knpu module and runs knpu-infer with
the ResNet-8 compiled for a 64 KiB SRAM (double rounding) on the ten
photographs in /usr/share/knpu/resnet8/ (models/resnet8/photos/), one of
each CIFAR-10 class (docs/design.md 10). Prints for each photograph the
class of top1 and whether the output equals TFLite Micro's, then powers
off. Exits with status 1 when an output differs or a step fails.

UART0 goes through a UNIX socket and UART1 to a file, both kept in the log
directory as demo-uart0.log and demo-uart1.log. Only the Python standard
library is used.
"""

import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "boot"))
import kn1_boot_test as boot  # noqa: E402
import kn1_v6_test as v6  # noqa: E402

NET = v6.RESNET8
MODEL = "resnet8-double-64k.knpu"
MODE = "double"


def classify(con):
    """Run the model on each photograph; True if every output equals
    TFLite Micro's."""
    print(f"{MODEL} (64 KiB SRAM, {MODE} rounding)")
    print(f"  {'photo':<12}{'top1':<12}output")
    right = equal = 0
    for cls in NET.classes:
        name = f"photo-{cls}"
        expected = f"{NET.dir}/expected-{MODE}-{name}.bin"
        v, out = v6.infer(con, f"{NET.dir}/{MODEL} {NET.dir}/{name}.bin "
                               f"--expect {expected}")
        top1 = v.get("top1", "")
        label = "-"
        if top1.isdigit() and int(top1) < len(NET.classes):
            label = NET.classes[int(top1)]
        if v.get("status") == "0" and v.get("match") == "yes":
            result = "equals TFLite Micro"
            equal += 1
        elif v.get("status") == "1":
            result = (f"differs from TFLite Micro "
                      f"({v.get('differing')} bytes)")
        else:
            result = f"knpu-infer failed (status {v.get('status')})"
            print(out)
        if label == cls:
            right += 1
        print(f"  {cls:<12}{label:<12}{result}")
    n = len(NET.classes)
    print(f"{right} of {n} photographs classified as their class")
    print(f"{equal} of {n} outputs equal TFLite Micro")
    return equal == n


def demo(args):
    """Boot, classify, power off; True if all went well."""
    uart0_log = os.path.join(args.log_dir, "demo-uart0.log")
    uart1_log = os.path.join(args.log_dir, "demo-uart1.log")
    with tempfile.TemporaryDirectory(prefix="kn1-demo-") as tmp, \
            open(uart0_log, "wb") as log:
        sock_path = os.path.join(tmp, "uart0.sock")
        print("boot: kn1-evb in QEMU through TF-A, U-Boot and Linux")
        qemu = boot.start_qemu(args, sock_path, uart1_log)
        sock = None
        try:
            sock = boot.connect(qemu, sock_path)
            con = boot.Console(qemu, sock, log)
            boot.boot_to_shell(con, args.timeout, "boot")

            out = con.run("modprobe knpu; echo rc=$?; "
                          "test -c /dev/accel/accel0 && echo node=yes")
            if "rc=0" not in out or "node=yes" not in out:
                raise boot.TestFailure(
                    "modprobe knpu did not create /dev/accel/accel0")
            print("modprobe knpu: /dev/accel/accel0")

            ok = classify(con)

            con.send("poweroff")
            status = con.wait_exit(120)
            print(f"poweroff: QEMU exited with status {status}")
            return ok and status == 0
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
                        help="where to write demo-uart0.log and "
                             "demo-uart1.log (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds allowed for the boot "
                             "(default: %(default)s)")
    args = parser.parse_args()
    # The lines show up one by one also when stdout is a pipe
    sys.stdout.reconfigure(line_buffering=True)

    for path in (os.path.join(args.host_bin, "qemu-system-aarch64"),
                 os.path.join(args.images, "flash.bin"),
                 os.path.join(args.images, "disk.img")):
        if not os.path.exists(path):
            print(f"FAIL: {path} does not exist "
                  "(build the image first: make)")
            return 1
    os.makedirs(args.log_dir, exist_ok=True)

    try:
        ok = demo(args)
    except boot.TestFailure as e:
        print(f"FAIL: {e}")
        ok = False
    logs = os.path.relpath(args.log_dir)
    if logs.startswith(".."):
        logs = os.path.abspath(args.log_dir)
    print(f"Console logs are in {logs}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
