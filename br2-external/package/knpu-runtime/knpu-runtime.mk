################################################################################
#
# knpu-runtime
#
################################################################################

# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later

# libknpu and knpu-infer (docs/design.md 10): the cargo workspace in
# runtime/ of this repository. It uses no external crates, so the local
# site method, which does no vendoring, builds it offline.

# Buildroot's version for sources from a local directory; legal-info
# leaves out a package without a version (docs/design.md 12.4)
KNPU_RUNTIME_VERSION = custom
KNPU_RUNTIME_SITE = $(BR2_EXTERNAL_KN1_PATH)/../runtime
KNPU_RUNTIME_SITE_METHOD = local
KNPU_RUNTIME_LICENSE = Apache-2.0 WITH LLVM-exception
KNPU_RUNTIME_LICENSE_FILES = LICENSES/Apache-2.0.txt \
	LICENSES/LLVM-exception.txt
# The workspace root has no package to install; build and install the tool
KNPU_RUNTIME_SUBDIR = knpu-infer

# legal-info takes the license texts from the build directory; runtime/
# has none of its own
define KNPU_RUNTIME_COPY_LICENSE_FILES
	$(foreach f,$(KNPU_RUNTIME_LICENSE_FILES), \
		$(INSTALL) -D -m 0644 $(BR2_EXTERNAL_KN1_PATH)/../$(f) \
			$(@D)/$(f)$(sep))
endef
KNPU_RUNTIME_POST_RSYNC_HOOKS += KNPU_RUNTIME_COPY_LICENSE_FILES

$(eval $(cargo-package))
