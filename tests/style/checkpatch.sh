#!/bin/sh
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Upstream style checks (docs/design.md 11). Unpacks checkpatch.pl and its
# settings from the source tarballs the build downloads, puts the files of
# this repository where they go in each upstream tree, and checks them as new
# files with that upstream's rules:
#
#   QEMU    hw/qemu/, hw/npu-core/ (as hw/misc/) and the glue patches
#   Linux   linux/: the driver (as drivers/accel/knpu/), the UAPI header,
#           the device tree and the binding
#   U-Boot  firmware/u-boot/ and its glue patch, with U-Boot's
#           .checkpatch.conf
#   TF-A    firmware/tf-a/, with Linux's checkpatch.pl and TF-A's
#           .checkpatch.conf
#
# QEMU's checkpatch.pl -f checks a file as a patch that adds it. Linux's -f
# (also in U-Boot's and the one TF-A uses) lowers long lines, misspellings
# and the SPDX comment style to checks that only --strict shows, so for
# those three the script writes such patches itself.
#
# Any report (error, warning or check) fails, except the allowed ones listed
# at the end with their reasons. The tarballs are the ones named in the
# Buildroot .hash files and must match those hashes.
#
# usage: tests/style/checkpatch.sh <Buildroot download directory>

set -eu

if [ $# -ne 1 ] || [ ! -d "$1" ]; then
	echo "usage: $0 <Buildroot download directory>" >&2
	exit 2
fi

top=$(cd "$(dirname "$0")/../.." && pwd)
dl=$(cd "$1" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# checkpatch.pl also looks for .checkpatch.conf in $CHECKPATCH_CONFIG_DIR and
# $HOME; only the upstream's own file may count.
unset CHECKPATCH_CONFIG_DIR
HOME=$work
export HOME

# unpack <dir> <download subdirectory> <hash file> <member>...: checks the
# tarball named in the hash file against it and unpacks the members (without
# the top directory) into $work/<dir>
unpack() {
	dir=$work/$1
	hash=$top/$3
	file=$(awk '$1 == "sha256" && $3 ~ /\.tar\./ { print $3 }' "$hash")
	if [ -z "$file" ] || [ "$(echo "$file" | wc -l)" -ne 1 ]; then
		echo "FAIL: $3 does not name exactly one tarball" >&2
		exit 1
	fi
	tarball=$dl/$2/$file
	if [ ! -f "$tarball" ]; then
		echo "FAIL: $tarball is missing" >&2
		exit 1
	fi
	echo "Unpacking checkpatch.pl and its settings from $file"
	if ! awk -v f="$file" -v p="$tarball" \
	    '$1 == "sha256" && $3 == f { print $2 "  " p }' "$hash" |
		sha256sum --check --quiet -; then
		echo "FAIL: $file does not match $3" >&2
		exit 1
	fi
	shift 3
	mkdir -p "$dir"
	tar -xf "$tarball" -C "$dir" --strip-components=1 \
	    --wildcards --no-wildcards-match-slash "$@"
}

# newfiles <dir> <file>...: writes a patch that adds each file of the tree
# in $work/<dir> as $work/<dir>-new/<n>.patch
newfiles() {
	out=$work/$1-new
	cd "$work/$1"
	shift
	mkdir "$out"
	i=0
	for f; do
		i=$((i + 1))
		# Without the diff --git line checkpatch.pl takes the patch for
		# one with an empty commit message
		{
			echo "diff --git a/$f b/$f"
			diff -u --label /dev/null --label "b/$f" /dev/null "$f" ||
				[ $? -eq 1 ]
		} > "$out/$(printf %03d $i).patch"
	done
	cd "$work"
}

# check <name> <dir> <checkpatch.pl> <option or file>...: runs checkpatch.pl
# in $work/<dir> and adds its reports to $work/report as "<name>: <report>"
check() {
	name=$1
	cd "$work/$2"
	shift 2
	n=-1 # the first argument is checkpatch.pl
	for arg; do
		case $arg in
		-*) ;;
		*) n=$((n + 1)) ;;
		esac
	done
	echo "$name: checking $n file(s)"
	status=0
	perl "$@" > "$work/out" 2>&1 || status=$?
	cd "$work"
	# Status 1 only means that there are reports
	if [ "$status" -gt 1 ]; then
		cat "$work/out"
		echo "FAIL: checkpatch.pl of $name stopped with status $status"
		exit 1
	fi
	# --terse puts each report on one line; drop the per-file totals
	grep -v -e '^total: ' -e '^$' "$work/out" |
		sed -e "s|^$top/||" -e "s|^|$name: |" >> "$work/report" || true
}

: > "$work/report"

# QEMU, as br2-external/package/qemu-kn1/qemu-kn1.mk puts the files
unpack qemu qemu-kn1 br2-external/package/qemu-kn1/qemu-kn1.hash \
	'*/scripts/checkpatch.pl'
cp -r "$top/hw/qemu/hw" "$top/hw/qemu/include" "$top/hw/qemu/tests" \
	"$work/qemu/"
cp "$top/hw/npu-core/knpu_cmd.h" "$top/hw/npu-core/knpu_core.h" \
	"$top/hw/npu-core/knpu_core.c" "$work/qemu/hw/misc/"
files=$(cd "$work/qemu" && find hw include tests -type f | sort)
# $files becomes one argument per file (the names have no spaces)
check QEMU qemu scripts/checkpatch.pl --no-tree --terse -f $files
check QEMU qemu scripts/checkpatch.pl --no-tree --terse \
	"$top"/hw/qemu/patches/*.patch

# Linux: the driver and the UAPI header where they would sit in the kernel
# tree, the device tree and the binding as the kernel build and
# tests/dt/check-dt.sh put them
unpack linux linux br2-external/patches/linux/linux.hash \
	'*/scripts/checkpatch.pl' '*/scripts/spelling.txt' \
	'*/scripts/const_structs.checkpatch'
mkdir -p "$work/linux/drivers/accel" "$work/linux/arch/arm64/boot" \
	"$work/linux/Documentation/devicetree"
cp -r "$top/linux/knpu" "$work/linux/drivers/accel/knpu"
cp -r "$top/linux/include" "$work/linux/include"
cp -r "$top/linux/dts" "$work/linux/arch/arm64/boot/dts"
cp -r "$top/linux/bindings" "$work/linux/Documentation/devicetree/bindings"
files=$(cd "$work/linux" && find drivers include arch Documentation -type f |
	sort)
newfiles linux $files
check Linux linux scripts/checkpatch.pl --no-tree --showfile --terse \
	--show-types "$work"/linux-new/*.patch

# U-Boot, as br2-external/boot/uboot/uboot.mk puts the files; the options
# are in U-Boot's .checkpatch.conf
unpack uboot uboot br2-external/patches/uboot/uboot.hash \
	'*/scripts/checkpatch.pl' '*/scripts/spelling.txt' \
	'*/scripts/const_structs.checkpatch' '*/.checkpatch.conf'
mkdir -p "$work/uboot/board" "$work/uboot/configs"
cp -r "$top/firmware/u-boot/board/kiwon" "$work/uboot/board/kiwon"
cp "$top/firmware/u-boot/configs/kn1-evb_defconfig" "$work/uboot/configs/"
files=$(cd "$work/uboot" && find board configs -type f | sort)
newfiles uboot $files
check U-Boot uboot scripts/checkpatch.pl --showfile --terse --show-types \
	"$work"/uboot-new/*.patch
check U-Boot uboot scripts/checkpatch.pl --terse --show-types \
	"$top"/firmware/u-boot/patches/*.patch

# TF-A, as br2-external/boot/arm-trusted-firmware/arm-trusted-firmware.mk
# puts the files; like TF-A's make checkpatch, Linux's checkpatch.pl with
# TF-A's .checkpatch.conf
unpack tfa arm-trusted-firmware \
	br2-external/patches/arm-trusted-firmware/arm-trusted-firmware.hash \
	'*/.checkpatch.conf'
mkdir -p "$work/tfa/plat/kiwon"
cp -r "$top/firmware/tf-a/plat/kiwon/kn1" "$work/tfa/plat/kiwon/kn1"
files=$(cd "$work/tfa" && find plat -type f | sort)
newfiles tfa $files
check TF-A tfa ../linux/scripts/checkpatch.pl --terse --show-types \
	"$work"/tfa-new/*.patch

# allow <count> <report>: exactly <count> reports match <report> (an
# extended regular expression from the start of the line); they are taken
# out of $work/report
failed=0
allow() {
	grep -E -e "^$2" "$work/report" | sed 's/^/Allowed: /' || true
	n=$(grep -c -E -e "^$2" "$work/report" || true)
	if [ "$n" -ne "$1" ]; then
		echo "FAIL: $1 allowed report(s) expected, $n found: $2"
		failed=1
	fi
	grep -v -E -e "^$2" "$work/report" > "$work/rest" || true
	mv "$work/rest" "$work/report"
}

# D33: the command interpreter that the QEMU device shares with the host
# tools is MIT. QEMU accepts MIT and asks for the reason in the commit
# message.
mit="WARNING: Saw acceptable license 'MIT' "
allow 1 "QEMU: hw/misc/knpu_cmd\.h:[0-9]+: $mit"
allow 1 "QEMU: hw/misc/knpu_core\.h:[0-9]+: $mit"
allow 1 "QEMU: hw/misc/knpu_core\.c:[0-9]+: $mit"
# The DRAM size and banks come from the device tree through the same two
# fdtdec functions that U-Boot's qemu-arm board uses
# (board/emulation/qemu-arm/qemu-arm.c).
allow 2 "U-Boot: board/kiwon/kn1-evb/kn1-evb\.c:[0-9]+: WARNING:LIVETREE: "

if [ -s "$work/report" ]; then
	sed 's/^/FAIL: /' "$work/report"
	failed=1
fi
if [ $failed -ne 0 ]; then
	echo "FAIL: the upstream style checks reported problems"
	exit 1
fi
echo "Upstream style checks passed."
