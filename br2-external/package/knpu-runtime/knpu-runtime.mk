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

KNPU_RUNTIME_SITE = $(BR2_EXTERNAL_KN1_PATH)/../runtime
KNPU_RUNTIME_SITE_METHOD = local
KNPU_RUNTIME_LICENSE = Apache-2.0 WITH LLVM-exception
# The workspace root has no package to install; build and install the tool
KNPU_RUNTIME_SUBDIR = knpu-infer

$(eval $(cargo-package))
