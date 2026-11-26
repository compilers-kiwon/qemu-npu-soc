#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""License of every file against the table of docs/design.md 12.4.

Reads the report of `reuse lint --json`, which gives the SPDX license
expression of each file from its header or from REUSE.toml, and compares it
with the first row of TABLE that matches the path. Fails when a file has
another expression or none, and when a row matches no file. Only the Python
standard library is used.

usage: tests/license/check-licenses.py <report of reuse lint --json>
"""

import fnmatch
import json
import sys

# The table of docs/design.md 12.4. A pattern ending in / is a directory,
# the others are fnmatch patterns; the first row that matches applies.
TABLE = [
    # Inputs from elsewhere keep their license: MNIST (its test images, the
    # CNN trained on it, the model files compiled from that and the outputs
    # of TFLite Micro) and the photographs from Wikimedia Commons
    ("models/mnist/digits/*", "CC-BY-SA-3.0"),
    ("models/mnist/mnist.tflite", "CC-BY-SA-3.0"),
    ("models/mnist/vectors/*.knpu", "CC-BY-SA-3.0"),
    ("models/mnist/vectors/expected-*.bin", "CC-BY-SA-3.0"),
    ("models/resnet8/photos/*", "CC0-1.0"),
    # Code for upstream trees follows each upstream
    ("hw/qemu/", "GPL-2.0-or-later"),
    ("hw/npu-core/", "MIT"),
    ("linux/knpu/", "GPL-2.0-only OR MIT"),
    ("linux/include/", "MIT"),
    ("linux/bindings/", "GPL-2.0-only OR BSD-2-Clause"),
    ("linux/dts/", "GPL-2.0-or-later OR MIT"),
    ("firmware/tf-a/", "BSD-3-Clause"),
    ("firmware/u-boot/", "GPL-2.0-or-later"),
    # Buildroot's license; a patch for an upstream tree would follow that
    # tree and need a row of its own
    ("br2-external/", "GPL-2.0-or-later"),
    # Everything else (D18)
    ("", "Apache-2.0 WITH LLVM-exception"),
]


def row_of(path):
    for i, (pattern, _) in enumerate(TABLE):
        if not pattern or pattern.endswith("/"):
            if path.startswith(pattern):
                return i
        elif fnmatch.fnmatchcase(path, pattern):
            return i
    raise AssertionError("the last row matches every path")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <report of reuse lint --json>")
    with open(sys.argv[1], encoding="utf-8") as f:
        report = json.load(f)
    if report.get("lint_version") != "1.0":
        sys.exit(f"FAIL: unknown version {report.get('lint_version')} of "
                 f"the reuse lint report")

    count = [0] * len(TABLE)
    failed = False
    for entry in sorted(report["files"], key=lambda e: e["path"]):
        path = entry["path"]
        found = [e["value"] for e in entry["spdx_expressions"]]
        i = row_of(path)
        count[i] += 1
        pattern, want = TABLE[i]
        if found != [want]:
            print(f"FAIL: {path}: {' and '.join(found) or 'no license'}, "
                  f"expected {want} ({pattern or 'everything else'})")
            failed = True

    for (pattern, want), n in zip(TABLE, count):
        print(f"{n:4d} file(s) {pattern or 'everything else'}: {want}")
        if n == 0:
            print(f"FAIL: no file matches {pattern}")
            failed = True
    if failed:
        sys.exit("FAIL: licenses differ from docs/design.md 12.4")
    print("The license of every file follows docs/design.md 12.4.")


if __name__ == "__main__":
    main()
