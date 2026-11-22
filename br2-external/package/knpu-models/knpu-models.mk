################################################################################
#
# knpu-models
#
################################################################################

# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later

# The test vectors of V6 (docs/design.md 10, 11): the test model compiled to
# model.knpu, its inputs and TFLite Micro's outputs, from models/conv/vectors/
# of this repository.

KNPU_MODELS_SITE = $(BR2_EXTERNAL_KN1_PATH)/../models/conv/vectors
KNPU_MODELS_SITE_METHOD = local
KNPU_MODELS_LICENSE = Apache-2.0 WITH LLVM-exception

define KNPU_MODELS_INSTALL_TARGET_CMDS
	$(INSTALL) -d -m 0755 $(TARGET_DIR)/usr/share/knpu/conv
	$(INSTALL) -m 0644 $(@D)/*.knpu $(@D)/*.bin \
		$(TARGET_DIR)/usr/share/knpu/conv/
endef

$(eval $(generic-package))
