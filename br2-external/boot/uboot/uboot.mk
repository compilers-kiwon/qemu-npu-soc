################################################################################
#
# uboot (kn1-evb board port hook)
#
################################################################################

# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later

# This is not a package. It extends Buildroot's uboot package so that the
# kn1-evb board port kept in firmware/u-boot/ is copied into the U-Boot
# tree and its Kconfig glue patch is applied after the sources are
# extracted and patched (docs/design.md, 7.2 and 12.3).

UBOOT_KN1_SRC_DIR = $(BR2_EXTERNAL_KN1_PATH)/../firmware/u-boot

define UBOOT_KN1_ADD_BOARD
	rm -rf $(@D)/board/kiwon
	cp -r $(UBOOT_KN1_SRC_DIR)/board/kiwon $(@D)/board/kiwon
	cp $(UBOOT_KN1_SRC_DIR)/configs/kn1-evb_defconfig $(@D)/configs/
	$(APPLY_PATCHES) $(@D) $(UBOOT_KN1_SRC_DIR)/patches \*.patch
endef
UBOOT_POST_PATCH_HOOKS += UBOOT_KN1_ADD_BOARD
