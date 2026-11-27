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

# legal-info saves the tarball and the applied patches but not the files
# copied in above or the device trees Buildroot copies in from
# BR2_TARGET_UBOOT_CUSTOM_DTS_PATH; they go next to them, at their paths in
# the U-Boot tree (docs/design.md 12.4)
define UBOOT_KN1_SAVE_BOARD
	mkdir -p $(UBOOT_REDIST_SOURCES_DIR)/board \
		$(UBOOT_REDIST_SOURCES_DIR)/configs \
		$(UBOOT_REDIST_SOURCES_DIR)/arch/$(UBOOT_ARCH)/dts
	cp -r $(UBOOT_KN1_SRC_DIR)/board/kiwon $(UBOOT_REDIST_SOURCES_DIR)/board/
	cp $(UBOOT_KN1_SRC_DIR)/configs/kn1-evb_defconfig \
		$(UBOOT_REDIST_SOURCES_DIR)/configs/
	cp $(UBOOT_CUSTOM_DTS_PATH) \
		$(UBOOT_REDIST_SOURCES_DIR)/arch/$(UBOOT_ARCH)/dts/
endef
UBOOT_POST_LEGAL_INFO_HOOKS += UBOOT_KN1_SAVE_BOARD
