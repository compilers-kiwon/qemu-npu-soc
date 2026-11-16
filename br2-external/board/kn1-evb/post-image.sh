#!/bin/bash
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Buildroot post-image script: build the boot flash image and the boot
# disk image, and put a start-qemu.sh next to the images.

set -e

BOARD_DIR="$(dirname "$0")"

# Boot flash image for kn1-evb: BL1 at the reset vector, the FIP (BL2,
# BL31, U-Boot) at 256 KiB, which is PLAT_KN1_FIP_BASE in the TF-A port.
FLASH="${BINARIES_DIR}/flash.bin"
rm -f "${FLASH}"
dd if="${BINARIES_DIR}/bl1.bin" of="${FLASH}" bs=4096 conv=notrunc status=none
dd if="${BINARIES_DIR}/fip.bin" of="${FLASH}" bs=4096 seek=64 conv=notrunc \
    status=none

# Boot disk: GPT with the ext4 root file system as its only partition.
# Run from the Buildroot top directory, as all post-image scripts are.
support/scripts/genimage.sh -c "${BOARD_DIR}/genimage.cfg"

sed -e "s|@HOST_DIR@|${HOST_DIR}|g" \
    "${BOARD_DIR}/start-qemu.sh.in" > "${BINARIES_DIR}/start-qemu.sh"
chmod +x "${BINARIES_DIR}/start-qemu.sh"
