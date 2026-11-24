#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Boot smoke test for kn1-evb through the full firmware chain.

Boots the Buildroot images (flash.bin, disk.img) on the kn1-evb QEMU
machine without touching the console until Linux asks for a login, then
checks the M1.2 completion conditions:

  1. BL1 -> BL2 -> BL31 -> U-Boot -> extlinux -> Linux login, unattended
  2. four CPUs, 4 GiB of DRAM, the PL031 RTC holding the host time and
     the SBSA watchdog device
  3. reboot goes back through BL1 to a login prompt
  4. poweroff makes QEMU exit with status 0
  5. the machine refuses a RAM size other than 4 GiB

UART0 (U-Boot and Linux) is attached through a UNIX socket, UART1 (TF-A)
goes to a file. Both logs are kept in the log directory. Only the Python
standard library is used.
"""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time

LOGIN = "login:"
PROMPT = "# "
MARK = "KN1_%s"


class TestFailure(Exception):
    pass


class Console:
    """UART0 of a running QEMU, reached through a UNIX socket."""

    def __init__(self, qemu, sock, log):
        self.qemu = qemu
        self.sock = sock
        self.log = log
        self.text = ""
        self.pos = 0

    def _read(self, timeout):
        self.sock.settimeout(timeout)
        try:
            data = self.sock.recv(4096)
        except socket.timeout:
            return True
        if not data:
            return False
        self.log.write(data)
        self.log.flush()
        self.text += data.decode(errors="replace").replace("\r", "")
        return True

    def expect(self, pattern, timeout):
        """Wait for pattern after the current position and move past it."""
        deadline = time.monotonic() + timeout
        while True:
            index = self.text.find(pattern, self.pos)
            if index >= 0:
                self.pos = index + len(pattern)
                return self.text[:index]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TestFailure(f"timed out waiting for {pattern!r}")
            if not self._read(min(remaining, 1.0)):
                raise TestFailure(
                    f"console closed while waiting for {pattern!r}")

    def wait_exit(self, timeout):
        """Keep logging the console until QEMU exits; return its status."""
        deadline = time.monotonic() + timeout
        while self.qemu.poll() is None:
            if time.monotonic() > deadline:
                raise TestFailure("QEMU did not exit")
            if not self._read(0.5):
                time.sleep(0.1)
        return self.qemu.wait()

    def send(self, line):
        self.sock.sendall(line.encode() + b"\r")

    def run(self, command, timeout=30):
        """Run a shell command and return its output between two markers."""
        begin, end = MARK % "BEGIN", MARK % "END"
        marker = f"printf '{MARK}\\n'"
        self.send(f"{marker} BEGIN; {command}; {marker} END")
        self.expect(begin + "\n", timeout)
        start = self.pos
        self.expect(end, timeout)
        return self.text[start:self.pos - len(end)]


def start_qemu(args, sock_path, uart1_log, extra=()):
    cmd = [
        os.path.join(args.host_bin, "qemu-system-aarch64"),
        "-M", "kn1-evb", "-display", "none",
        "-chardev", f"socket,id=uart0,path={sock_path},server=on,wait=on",
        "-serial", "chardev:uart0",
        "-serial", f"file:{uart1_log}",
        "-bios", os.path.join(args.images, "flash.bin"),
        "-drive", "file=%s,if=none,format=raw,id=hd0,snapshot=on"
        % os.path.join(args.images, "disk.img"),
        "-device", "virtio-blk-device,drive=hd0",
        *extra,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)


def connect(qemu, sock_path):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if qemu.poll() is not None:
            raise TestFailure("QEMU exited before the console was connected")
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(sock_path)
            return sock
        except OSError:
            sock.close()
            time.sleep(0.1)
    raise TestFailure("could not connect to the QEMU console socket")


def check(condition, message):
    if not condition:
        raise TestFailure(message)
    print(f"PASS: {message}")


def boot_to_shell(con, timeout, what):
    before = con.expect(LOGIN, timeout)
    boot = before[before.rfind("U-Boot 20"):] if "U-Boot 20" in before else ""
    check("with extlinux" in boot,
          f"{what}: U-Boot boots Linux through extlinux")
    con.send("root")
    con.expect(PROMPT, 30)
    # Keep kernel messages off the console from here on, as Buildroot's
    # runtime tests do (support/testing/infra/emulator.py): they can land
    # in the middle of a command's output. The tests read the kernel log
    # with dmesg.
    con.run("dmesg -n 1")
    print(f"PASS: {what}: logged in on UART0")


def check_system(con):
    out = con.run(
        'echo "nproc=$(nproc)"; '
        "echo \"memkb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)\"; "
        'echo "rtc=$(cat /sys/class/rtc/rtc0/name)"; '
        'echo "epoch=$(date +%s)"; '
        'echo "wdt=$(test -c /dev/watchdog0 && echo yes)"; '
        'echo "cmdline=$(cat /proc/cmdline)"')
    host_epoch = int(time.time())
    values = dict(line.split("=", 1)
                  for line in out.splitlines() if "=" in line)

    check(values.get("nproc") == "4", "four CPUs are online (PSCI CPU_ON)")
    memkb = int(values.get("memkb", "0"))
    check(3584 * 1024 < memkb <= 4096 * 1024,
          f"MemTotal {memkb} kB fits 4 GiB of DRAM")
    check("pl031" in values.get("rtc", ""), "rtc0 is the PL031")
    guest_epoch = int(values.get("epoch", "0"))
    check(abs(guest_epoch - host_epoch) < 600,
          "system time set from the RTC is within 10 minutes of the host")
    check(values.get("wdt") == "yes", "/dev/watchdog0 exists")
    check("root=/dev/vda1" in values.get("cmdline", ""),
          "kernel command line comes from extlinux.conf")


def count(path, text):
    with open(path, errors="replace") as f:
        return f.read().count(text)


def test_boot(args):
    uart0_log = os.path.join(args.log_dir, "uart0.log")
    uart1_log = os.path.join(args.log_dir, "uart1.log")
    with tempfile.TemporaryDirectory(prefix="kn1-") as tmp, \
            open(uart0_log, "wb") as log:
        sock_path = os.path.join(tmp, "uart0.sock")
        qemu = start_qemu(args, sock_path, uart1_log)
        sock = None
        try:
            sock = connect(qemu, sock_path)
            con = Console(qemu, sock, log)

            boot_to_shell(con, args.timeout, "first boot")
            check(count(uart1_log, "BL31: v") == 1, "TF-A ran BL31 on UART1")
            check_system(con)

            con.send("reboot")
            boot_to_shell(con, args.timeout, "after reboot")
            check(count(uart1_log, "kn1: system reset") == 1
                  and count(uart1_log, "BL1: Booting BL2") == 2,
                  "reboot went through PSCI SYSTEM_RESET and BL1")

            con.send("poweroff")
            status = con.wait_exit(120)
            check(status == 0,
                  f"poweroff ends QEMU with status 0 (got {status})")
            check(count(uart1_log, "kn1: system off") == 1,
                  "poweroff went through PSCI SYSTEM_OFF")
        finally:
            if sock:
                sock.close()
            if qemu.poll() is None:
                qemu.kill()
                qemu.wait()


def test_ram_size(args):
    cmd = [os.path.join(args.host_bin, "qemu-system-aarch64"),
           "-M", "kn1-evb", "-m", "2G", "-display", "none",
           "-bios", os.path.join(args.images, "flash.bin")]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=30)
    except subprocess.TimeoutExpired:
        raise TestFailure("QEMU kept running with -m 2G")
    check(result.returncode != 0 and "fixed at 4 GiB" in result.stderr,
          "the machine refuses -m 2G")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default="output/images",
                        help="Buildroot images directory "
                             "(default: %(default)s)")
    parser.add_argument("--host-bin", default="output/host/bin",
                        help="directory holding qemu-system-aarch64 "
                             "(default: %(default)s)")
    parser.add_argument("--log-dir", default="output/test-logs",
                        help="where to write uart0.log and uart1.log "
                             "(default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds allowed for each boot "
                             "(default: %(default)s)")
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    try:
        for path in (os.path.join(args.host_bin, "qemu-system-aarch64"),
                     os.path.join(args.images, "flash.bin"),
                     os.path.join(args.images, "disk.img")):
            if not os.path.exists(path):
                raise TestFailure(f"{path} does not exist")
        test_boot(args)
        test_ram_size(args)
    except TestFailure as e:
        print(f"FAIL: {e}")
        print(f"Console logs are in {args.log_dir}/")
        return 1
    print("All boot checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
