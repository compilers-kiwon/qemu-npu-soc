################################################################################
#
# linux (kn1 device tree legal-info hook)
#
################################################################################

# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later

# This is not a package. It extends Buildroot's linux package: legal-info
# saves the kernel tarball and the applied patches but not the device trees
# BR2_LINUX_KERNEL_CUSTOM_DTS_DIR copies in from linux/dts/; they go next to
# them, at their paths in the kernel tree (docs/design.md 5.5 and 12.4).

define LINUX_KN1_SAVE_DTS
	mkdir -p $(LINUX_REDIST_SOURCES_DIR)/arch/$(KERNEL_ARCH)/boot/dts
	cp -r $(LINUX_KERNEL_CUSTOM_DTS_DIR)/. \
		$(LINUX_REDIST_SOURCES_DIR)/arch/$(KERNEL_ARCH)/boot/dts/
endef
LINUX_POST_LEGAL_INFO_HOOKS += LINUX_KN1_SAVE_DTS
