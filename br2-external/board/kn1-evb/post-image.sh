#!/bin/bash
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Buildroot post-image script: put a start-qemu.sh next to the images.

set -e

BOARD_DIR="$(dirname "$0")"

sed -e "s|@HOST_DIR@|${HOST_DIR}|g" \
    "${BOARD_DIR}/start-qemu.sh.in" > "${BINARIES_DIR}/start-qemu.sh"
chmod +x "${BINARIES_DIR}/start-qemu.sh"
