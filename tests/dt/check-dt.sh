#!/bin/sh
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Schema checks for the kn1-npu binding and the kn1-evb device tree. Unpacks
# a Linux source tarball into a temporary directory, adds linux/bindings/
# and linux/dts/ of this repository, then runs dt_binding_check on the
# binding and CHECK_DTBS on kiwon/kn1-evb.dtb. Any message fails the check,
# except that no schema matches the board's root compatible (design 13.3).
#
# Needs dt-schema (dt-doc-validate, dt-validate, ...), yamllint, flex, bison
# and a host C compiler.
#
# usage: tests/dt/check-dt.sh <linux-7.3.1.tar.xz>

set -eu

if [ $# -ne 1 ] || [ ! -f "$1" ]; then
	echo "usage: $0 <linux source tarball>" >&2
	exit 2
fi

top=$(cd "$(dirname "$0")/../.." && pwd)
tarball=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

echo "Unpacking $(basename "$tarball")"
tar -xf "$tarball" -C "$work"
src=$(find "$work" -mindepth 1 -maxdepth 1 -type d -name 'linux-*')
build=$work/build

cp -r "$top/linux/bindings/." "$src/Documentation/devicetree/bindings/"
mkdir -p "$src/arch/arm64/boot/dts/kiwon"
cp "$top"/linux/dts/kiwon/* "$src/arch/arm64/boot/dts/kiwon/"

# make -s prints nothing but the checkers' messages
log=$work/check.log
run() {
	echo "$*"
	if ! make -C "$src" O="$build" -s "$@" >> "$log" 2>&1; then
		cat "$log"
		echo "FAIL: make $*"
		exit 1
	fi
}

run dt_binding_check DT_SCHEMA_FILES=npu/kiwon,kn1-npu.yaml
run ARCH=arm64 defconfig
run ARCH=arm64 CHECK_DTBS=y kiwon/kn1-evb.dtb

cat "$log"
known="/: failed to match any schema with compatible: \['kiwon,kn1-evb', 'kiwon,kn1'\]"
# Kernel Makefile notices such as the host assembler warning are not ours
if grep -v -e "$known" -e 'Makefile:[0-9]*: ' "$log" | grep -q .; then
	echo "FAIL: the checks reported problems"
	exit 1
fi
echo "Binding and device tree checks passed."
