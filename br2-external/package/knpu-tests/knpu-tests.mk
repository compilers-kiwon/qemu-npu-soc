################################################################################
#
# knpu-tests
#
################################################################################

# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later

# The guest side of V5 (docs/design.md 11). The host side,
# tests/driver/kn1_v5_test.py, boots the image and runs it.

KNPU_TESTS_SITE = $(BR2_EXTERNAL_KN1_PATH)/../tests/driver
KNPU_TESTS_SITE_METHOD = local
KNPU_TESTS_LICENSE = Apache-2.0 WITH LLVM-exception
KNPU_TESTS_DEPENDENCIES = knpu

define KNPU_TESTS_COPY_CMD_HEADER
	cp $(BR2_EXTERNAL_KN1_PATH)/../hw/npu-core/knpu_cmd.h $(@D)/
endef
KNPU_TESTS_POST_RSYNC_HOOKS += KNPU_TESTS_COPY_CMD_HEADER

define KNPU_TESTS_BUILD_CMDS
	$(TARGET_CC) $(TARGET_CFLAGS) $(TARGET_LDFLAGS) \
		-Wall -Wextra -Werror -o $(@D)/knpu-test $(@D)/knpu_test.c
endef

define KNPU_TESTS_INSTALL_TARGET_CMDS
	$(INSTALL) -D -m 0755 $(@D)/knpu-test $(TARGET_DIR)/usr/bin/knpu-test
endef

$(eval $(generic-package))
