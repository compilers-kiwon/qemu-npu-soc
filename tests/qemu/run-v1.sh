#!/bin/sh
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# V1 (docs/design.md 11): the kn1-npu qtest and QEMU's generic
# device-introspect-test and qom-test for aarch64, run from the build tree
# that Buildroot keeps for the host-qemu-kn1 package.
#
#   tests/qemu/run-v1.sh [BUILDROOT_OUTPUT_DIR]     (default: output)

set -eu

out=${1:-output}
build=
for dir in "$out"/build/host-qemu-kn1-*/build; do
	build=$dir
done
if [ ! -x "$build/qemu-system-aarch64" ]; then
	echo "run-v1.sh: no host-qemu-kn1 build under $out/build" >&2
	exit 1
fi

QTEST_QEMU_BINARY=$build/qemu-system-aarch64
export QTEST_QEMU_BINARY

status=0
for test in kn1-npu-test device-introspect-test qom-test; do
	echo "== $test"
	if ! "$build/tests/qtest/$test"; then
		echo "FAIL: $test" >&2
		status=1
	fi
done
exit $status
