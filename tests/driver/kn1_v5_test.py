#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""V5: knpu driver test on kn1-evb (docs/design.md 11).

Boots the Buildroot images through the firmware chain like the boot smoke
test, then in the guest:

  1. loads the knpu module and checks that it probed the NPU and took the
     board's CMA pool above 4 GiB
  2. runs knpu-test (tests/driver/knpu_test.c): ioctls, validator and
     submit checks, buffers above 4 GiB, device errors, timeout recovery,
     performance monitors and runtime PM
  3. unloads the module and checks that the device node is gone and the
     NPU clock is off
  4. loads the module again and reruns knpu-test
  5. checks that the kernel logged no warning, then powers off

UART0 goes through a UNIX socket and UART1 to a file, both kept in the log
directory as v5-uart0.log and v5-uart1.log.
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

CLK_ENABLED = "/sys/kernel/debug/clk/npu_clk/clk_enable_count"


def values(con, command):
    """Run commands that print key=value lines and return them as a dict."""
    out = con.run(command)
    return dict(line.split("=", 1) for line in out.splitlines()
                if "=" in line)


def load_module(con, what):
    v = values(con, "modprobe knpu; echo rc=$?; "
               "test -c /dev/accel/accel0 && echo node=yes")
    check(v.get("rc") == "0" and v.get("node") == "yes",
          f"{what}: modprobe knpu creates /dev/accel/accel0")


def run_knpu_test(con, what):
    out = con.run("knpu-test; echo status=$?", timeout=300)
    lines = out.splitlines()
    for line in lines:
        if line.startswith("FAIL") or line.startswith("All "):
            print(f"  guest: {line}")
    status = next((line.split("=", 1)[1] for line in lines
                   if line.startswith("status=")), "?")
    passed = sum(line.startswith("PASS") for line in lines)
    check(status == "0", f"{what}: knpu-test passed {passed} checks "
          f"(exit status {status})")


def test_v5(args):
    uart0_log = os.path.join(args.log_dir, "v5-uart0.log")
    uart1_log = os.path.join(args.log_dir, "v5-uart1.log")
    with tempfile.TemporaryDirectory(prefix="kn1-v5-") as tmp, \
            open(uart0_log, "wb") as log:
        sock_path = os.path.join(tmp, "uart0.sock")
        qemu = boot.start_qemu(args, sock_path, uart1_log)
        sock = None
        try:
            sock = boot.connect(qemu, sock_path)
            con = boot.Console(qemu, sock, log)
            boot.boot_to_shell(con, args.timeout, "boot")
            con.run("mount -t debugfs none /sys/kernel/debug 2>/dev/null")

            load_module(con, "first load")
            v = values(con, "dmesg | grep -q 'knpu .*kn1-npu v0.0, 64 KiB "
                       "SRAM, 8 regions' && echo probe=yes; "
                       "dmesg | grep -q 'created CMA memory pool at "
                       "0x000000013c000000, size 64 MiB' && echo pool=yes; "
                       "dmesg | grep -q 'knpu .*assigned reserved memory "
                       "node npu-pool@13c000000' && echo assigned=yes")
            check(v.get("probe") == "yes",
                  "the driver identified a kn1-npu v0.0 with 64 KiB SRAM")
            check(v.get("pool") == "yes" and v.get("assigned") == "yes",
                  "the NPU uses the 64 MiB CMA pool at 0x13c000000")

            run_knpu_test(con, "first load")

            v = values(con, "rmmod knpu; echo rc=$?; "
                       "test -e /dev/accel/accel0 || echo gone=yes; "
                       f"echo clk=$(cat {CLK_ENABLED})")
            check(v.get("rc") == "0" and v.get("gone") == "yes",
                  "rmmod knpu removes /dev/accel/accel0")
            check(v.get("clk") == "0", "the NPU clock is off after rmmod")

            load_module(con, "second load")
            run_knpu_test(con, "second load")

            v = values(con, "echo warnings=$(dmesg | grep -cE "
                       "'WARNING:|BUG:|Oops|refcount_t:')")
            check(v.get("warnings") == "0", "the kernel logged no warnings")

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
                        help="where to write v5-uart0.log and v5-uart1.log "
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
        test_v5(args)
    except TestFailure as e:
        print(f"FAIL: {e}")
        print(f"Console logs are in {args.log_dir}/")
        return 1
    print("All V5 checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
