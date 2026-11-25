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
  4. checks that /usr/share/knpu/mnist/ holds the MNIST test vectors and
     digit images of this tree (models/mnist/vectors/, models/mnist/digits/),
     as the expected values of 5. come from those files
  5. runs knpu-infer on the four MNIST CNN models (the whole model with its
     CPU code, double and single rounding, 64 KiB and 32 KiB SRAM) with the
     fifteen inputs: the output must equal TFLite Micro's, top1 must be the
     index of its largest element, PERF_MACS the nominal MACs and the DMA
     and OP counters what the model's command stream implies. An output of
     the other rounding mode must be told apart. Prints the top1 of the
     digit images.
  6. does the same as 4. and 5. for the ResNet-8: the test vectors and
     photographs in /usr/share/knpu/resnet8/ (models/resnet8/vectors/,
     models/resnet8/photos/), and the four models with the fifteen inputs.
     Prints the classes of the photographs.
  7. checks that the kernel logged no warning, then powers off

UART0 goes through a UNIX socket and UART1 to a file, both kept in the log
directory as v6-uart0.log and v6-uart1.log.
"""

import argparse
import hashlib
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "boot"))
sys.path.insert(0, os.path.join(HERE, "..", "compiler"))
import kn1_boot_test as boot  # noqa: E402
import model_file  # noqa: E402
import test_mnist_model as mnist  # noqa: E402
import test_resnet8_model as resnet8  # noqa: E402

check = boot.check
TestFailure = boot.TestFailure

DIR = "/usr/share/knpu/conv"
MODES = ("double", "single")
SRAMS = ("64k", "4k")
INPUTS = 5

# The test model (models/conv/make_model.py): 14x14x8 -> 16, 3x3, SAME
MACS = 14 * 14 * 16 * 3 * 3 * 8
OUT_BYTES = 14 * 14 * 16


class Classifier:
    """The MNIST CNN or the ResNet-8. knpu-models installs the files of
    models/<name>/vectors/ and of its images (models/<name>/<images_dir>/)
    in /usr/share/knpu/<name>/. The inputs are those of V3
    (tests/compiler/test_<name>_model.py): an image of each class, named
    <prefix>-<class>, then five more."""

    def __init__(self, name, title, images_dir, what, prefix, macs,
                 classes):
        self.name = name
        self.title = title
        self.dir = f"/usr/share/knpu/{name}"
        self.tree = [os.path.join(HERE, "..", "..", "models", name, d)
                     for d in ("vectors", images_dir)]
        self.images_dir = images_dir
        self.what = what
        self.images = [f"{prefix}-{c}" for c in classes]
        self.inputs = self.images + ["random-0", "random-1", "random-2",
                                     "all-min", "all-max"]
        self.macs = macs
        self.classes = classes


MNIST = Classifier("mnist", "MNIST", "digits", "digit images", "digit",
                   mnist.MACS, [str(d) for d in range(10)])
RESNET8 = Classifier("resnet8", "ResNet-8", "photos", "photographs",
                     "photo", resnet8.MACS, resnet8.CLASSES)
CLASSIFIER_SRAMS = ("64k", "32k")


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


def read(path):
    with open(path, "rb") as f:
        return f.read()


def check_files(con, net):
    """run_classifier takes its expected values from the files of this
    tree, so the image must hold the same files."""
    want = {}
    for directory in net.tree:
        for name in os.listdir(directory):
            want[name] = hashlib.md5(
                read(os.path.join(directory, name))).hexdigest()
    got = {}
    for line in con.run(f"(cd {net.dir} && md5sum *)").splitlines():
        digest, sep, name = line.partition("  ")
        if sep and len(digest) == 32:
            got[name] = digest
    differ = sorted(n for n in want.keys() | got.keys()
                    if want.get(n) != got.get(n))
    what = (f"{net.dir}/ holds the {len(want)} files of "
            f"models/{net.name}/vectors/ and "
            f"models/{net.name}/{net.images_dir}/")
    if differ:
        what += (f"; {len(differ)} differ, the first {differ[0]} (install "
                 "them again: make br-knpu-models-rebuild all)")
    check(not differ, what)


def run_classifier(con, net, model, mode):
    stream = mnist.stream_counters(
        model_file.read(read(os.path.join(net.tree[0], model))))
    want = [str(n) for n in (net.macs, *stream)]
    tops = {}
    for name in net.inputs:
        expected = f"expected-{mode}-{name}.bin"
        v, out = infer(con, f"{net.dir}/{model} {net.dir}/{name}.bin "
                            f"--expect {net.dir}/{expected}")
        what = f"{model} {name}"
        if v.get("status") != "0" or v.get("match") != "yes":
            print(out)
        top1 = mnist.top1(read(os.path.join(net.tree[0], expected)))
        check(v.get("status") == "0" and v.get("match") == "yes" and
              v.get("top1") == str(top1),
              f"{what}: output equals TFLite Micro ({mode} rounding), "
              f"top1={v.get('top1')}")
        got = [v.get(k) for k in ("macs", "dma_rd_bytes", "dma_wr_bytes",
                                  "ops")]
        check(got == want,
              f"{what}: counters macs dma_rd_bytes dma_wr_bytes ops = "
              f"{' '.join(map(str, got))} (from the model: {' '.join(want)})")
        tops[name] = net.classes[top1]
    print(f"{model}: top1 of the {net.what} "
          + " ".join(f"{n}={tops[n]}" for n in net.images))


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

            for net in (MNIST, RESNET8):
                check_files(con, net)
                for mode in MODES:
                    for sram in CLASSIFIER_SRAMS:
                        run_classifier(con, net,
                                       f"{net.name}-{mode}-{sram}.knpu", mode)

                # random-2 gives different outputs in the two rounding
                # modes (tests/compiler/test_<name>_model.py)
                v, _ = infer(con, f"{net.dir}/{net.name}-double-64k.knpu "
                                  f"{net.dir}/random-2.bin --expect "
                                  f"{net.dir}/expected-single-random-2.bin")
                check(v.get("status") == "1" and v.get("match") == "no",
                      f"the double-rounding {net.title} model does not "
                      "match the single-rounding output "
                      f"({v.get('differing')} bytes differ)")

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
