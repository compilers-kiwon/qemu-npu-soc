################################################################################
#
# arm-trusted-firmware (kn1 platform port hook)
#
################################################################################

# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later

# This is not a package. It extends Buildroot's arm-trusted-firmware
# package so that the kn1 platform port kept in firmware/tf-a/ is copied
# into the TF-A tree after the sources are extracted and patched
# (docs/design.md, D34: no glue patch, the directory is copied).

ARM_TRUSTED_FIRMWARE_KN1_PLAT_DIR = \
	$(BR2_EXTERNAL_KN1_PATH)/../firmware/tf-a/plat/kiwon/kn1

define ARM_TRUSTED_FIRMWARE_KN1_COPY_PLAT
	rm -rf $(@D)/plat/kiwon
	mkdir -p $(@D)/plat/kiwon
	cp -r $(ARM_TRUSTED_FIRMWARE_KN1_PLAT_DIR) $(@D)/plat/kiwon/kn1
endef
ARM_TRUSTED_FIRMWARE_POST_PATCH_HOOKS += ARM_TRUSTED_FIRMWARE_KN1_COPY_PLAT

# legal-info saves the tarball but not the port copied in above; it goes
# next to it, at its path in the TF-A tree (docs/design.md 12.4)
define ARM_TRUSTED_FIRMWARE_KN1_SAVE_PLAT
	mkdir -p $(ARM_TRUSTED_FIRMWARE_REDIST_SOURCES_DIR)/plat/kiwon
	cp -r $(ARM_TRUSTED_FIRMWARE_KN1_PLAT_DIR) \
		$(ARM_TRUSTED_FIRMWARE_REDIST_SOURCES_DIR)/plat/kiwon/kn1
endef
ARM_TRUSTED_FIRMWARE_POST_LEGAL_INFO_HOOKS += ARM_TRUSTED_FIRMWARE_KN1_SAVE_PLAT
