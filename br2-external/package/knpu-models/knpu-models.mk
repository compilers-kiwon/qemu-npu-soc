################################################################################
#
# knpu-models
#
################################################################################

# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later

# The test vectors of V6 (docs/design.md 10, 11) from models/ of this
# repository: the convolution test model and the MNIST CNN compiled to
# model.knpu, their inputs and TFLite Micro's outputs, and the MNIST digit
# images. The MNIST models, outputs and digit images are under MNIST's
# license, the other files under the repository's (docs/design.md 12.4).

KNPU_MODELS_SITE = $(BR2_EXTERNAL_KN1_PATH)/../models
KNPU_MODELS_SITE_METHOD = local
KNPU_MODELS_LICENSE = Apache-2.0 WITH LLVM-exception, \
	CC-BY-SA-3.0 (MNIST models, outputs and digit images)

define KNPU_MODELS_INSTALL_TARGET_CMDS
	$(INSTALL) -d -m 0755 $(TARGET_DIR)/usr/share/knpu/conv \
		$(TARGET_DIR)/usr/share/knpu/mnist
	$(INSTALL) -m 0644 $(@D)/conv/vectors/* $(TARGET_DIR)/usr/share/knpu/conv/
	$(INSTALL) -m 0644 $(@D)/mnist/vectors/* $(@D)/mnist/digits/* \
		$(TARGET_DIR)/usr/share/knpu/mnist/
endef

$(eval $(generic-package))
