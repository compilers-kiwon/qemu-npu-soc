################################################################################
#
# knpu-models
#
################################################################################

# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later

# The test vectors of V6 (docs/design.md 10, 11) from models/ of this
# repository: the convolution test model, the MNIST CNN and the ResNet-8
# compiled to model.knpu, their inputs and TFLite Micro's outputs, the MNIST
# digit images and the ResNet-8 photographs. The MNIST models, outputs and
# digit images are under MNIST's license, the photographs under CC0-1.0,
# the other files under the repository's (docs/design.md 12.4).

# Buildroot's version for sources from a local directory; legal-info
# leaves out a package without a version (docs/design.md 12.4)
KNPU_MODELS_VERSION = custom
KNPU_MODELS_SITE = $(BR2_EXTERNAL_KN1_PATH)/../models
KNPU_MODELS_SITE_METHOD = local
KNPU_MODELS_LICENSE = Apache-2.0 WITH LLVM-exception, \
	CC-BY-SA-3.0 (MNIST models, outputs and digit images), \
	CC0-1.0 (ResNet-8 photographs)
KNPU_MODELS_LICENSE_FILES = LICENSES/Apache-2.0.txt \
	LICENSES/LLVM-exception.txt LICENSES/CC-BY-SA-3.0.txt \
	LICENSES/CC0-1.0.txt

# legal-info takes the license texts from the build directory; models/
# has none of its own
define KNPU_MODELS_COPY_LICENSE_FILES
	$(foreach f,$(KNPU_MODELS_LICENSE_FILES), \
		$(INSTALL) -D -m 0644 $(BR2_EXTERNAL_KN1_PATH)/../$(f) \
			$(@D)/$(f)$(sep))
endef
KNPU_MODELS_POST_RSYNC_HOOKS += KNPU_MODELS_COPY_LICENSE_FILES

define KNPU_MODELS_INSTALL_TARGET_CMDS
	$(INSTALL) -d -m 0755 $(TARGET_DIR)/usr/share/knpu/conv \
		$(TARGET_DIR)/usr/share/knpu/mnist \
		$(TARGET_DIR)/usr/share/knpu/resnet8
	$(INSTALL) -m 0644 $(@D)/conv/vectors/* $(TARGET_DIR)/usr/share/knpu/conv/
	$(INSTALL) -m 0644 $(@D)/mnist/vectors/* $(@D)/mnist/digits/* \
		$(TARGET_DIR)/usr/share/knpu/mnist/
	$(INSTALL) -m 0644 $(@D)/resnet8/vectors/* $(@D)/resnet8/photos/* \
		$(TARGET_DIR)/usr/share/knpu/resnet8/
endef

$(eval $(generic-package))
