################################################################################
#
# knpu
#
################################################################################

# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later

# The kn1-npu driver, built out of tree from linux/knpu/. The command
# encoding is shared with the QEMU device (hw/npu-core/knpu_cmd.h).

# Buildroot's version for sources from a local directory; legal-info
# leaves out a package without a version (docs/design.md 12.4)
KNPU_VERSION = custom
KNPU_SITE = $(BR2_EXTERNAL_KN1_PATH)/../linux
KNPU_SITE_METHOD = local
KNPU_LICENSE = GPL-2.0-only or MIT (driver), MIT (UAPI header)
KNPU_LICENSE_FILES = LICENSES/GPL-2.0-only.txt LICENSES/MIT.txt
KNPU_INSTALL_STAGING = YES
KNPU_MODULE_SUBDIRS = knpu

KNPU_NPU_CORE_DIR = $(BR2_EXTERNAL_KN1_PATH)/../hw/npu-core

define KNPU_COPY_CMD_HEADER
	cp $(KNPU_NPU_CORE_DIR)/knpu_cmd.h $(@D)/knpu/
endef
KNPU_POST_RSYNC_HOOKS += KNPU_COPY_CMD_HEADER

# legal-info takes the license texts from the build directory; linux/ has
# none of its own
define KNPU_COPY_LICENSE_FILES
	$(foreach f,$(KNPU_LICENSE_FILES), \
		$(INSTALL) -D -m 0644 $(BR2_EXTERNAL_KN1_PATH)/../$(f) \
			$(@D)/$(f)$(sep))
endef
KNPU_POST_RSYNC_HOOKS += KNPU_COPY_LICENSE_FILES

# legal-info saves linux/ as the source, without the header copied in
define KNPU_SAVE_CMD_HEADER
	$(INSTALL) -D -m 0644 $(KNPU_NPU_CORE_DIR)/knpu_cmd.h \
		$(KNPU_REDIST_SOURCES_DIR)/knpu/knpu_cmd.h
endef
KNPU_POST_LEGAL_INFO_HOOKS += KNPU_SAVE_CMD_HEADER

define KNPU_LINUX_CONFIG_FIXUPS
	$(call KCONFIG_ENABLE_OPT,CONFIG_DRM_ACCEL)
endef

# The GEM DMA helpers and the GPU scheduler have no prompt, so only another
# driver can select them: kn1_defconfig builds the Ethos-U driver as a
# module for this (docs/design.md 8.1).
KNPU_KERNEL_OPTS = CONFIG_DRM_ACCEL CONFIG_DRM_GEM_DMA_HELPER CONFIG_DRM_SCHED
# The timeout recovery calls pm_runtime_force_resume(), an empty stub
# without CONFIG_PM_SLEEP (docs/design.md 8.2)
KNPU_KERNEL_OPTS += CONFIG_PM_SLEEP

define KNPU_CHECK_KERNEL_CONFIG
	$(foreach opt,$(KNPU_KERNEL_OPTS), \
		grep -Eq '^$(opt)=[ym]$$' $(LINUX_DIR)/.config || \
			{ echo "knpu: the kernel needs $(opt)"; exit 1; }$(sep))
endef
KNPU_PRE_BUILD_HOOKS += KNPU_CHECK_KERNEL_CONFIG

define KNPU_INSTALL_STAGING_CMDS
	$(INSTALL) -D -m 0644 $(@D)/include/uapi/drm/knpu_accel.h \
		$(STAGING_DIR)/usr/include/drm/knpu_accel.h
endef

$(eval $(kernel-module))
$(eval $(generic-package))
