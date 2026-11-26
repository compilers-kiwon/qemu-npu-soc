################################################################################
#
# qemu-kn1
#
################################################################################

# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: GPL-2.0-or-later

# QEMU for the build host with the kiwon-kn1 SoC and the kn1-evb machine.
# The upstream tarball is patched with hw/qemu/patches/; the kn1 sources
# and qtests from hw/qemu/ and the shared NPU core from hw/npu-core/ are
# copied into the tree. Modelled on Buildroot's host-qemu.

QEMU_KN1_VERSION = 11.1.3
QEMU_KN1_SOURCE = qemu-$(QEMU_KN1_VERSION).tar.xz
QEMU_KN1_SITE = https://download.qemu.org
QEMU_KN1_LICENSE = GPL-2.0, LGPL-2.1, MIT, BSD-3-Clause, BSD-2-Clause, Others/BSD-1c
QEMU_KN1_LICENSE_FILES = COPYING COPYING.LIB

# Our SoC, board and glue patch live outside the br2-external tree
QEMU_KN1_KN1_SRC_DIR = $(BR2_EXTERNAL_KN1_PATH)/../hw/qemu
QEMU_KN1_NPU_CORE_DIR = $(BR2_EXTERNAL_KN1_PATH)/../hw/npu-core

HOST_QEMU_KN1_DEPENDENCIES = \
	host-dtc \
	host-libglib2 \
	host-meson \
	host-pixman \
	host-pkgconf \
	host-python3 \
	host-python-distlib \
	host-python-setuptools \
	host-python-wheel \
	host-slirp \
	host-zlib

HOST_QEMU_KN1_TARGETS = aarch64-softmmu
HOST_QEMU_KN1_CFLAGS = $(HOST_CFLAGS) -I$(HOST_DIR)/include/libfdt

define HOST_QEMU_KN1_ADD_KN1_SOURCES
	cp -r $(QEMU_KN1_KN1_SRC_DIR)/hw/. $(@D)/hw/
	cp -r $(QEMU_KN1_KN1_SRC_DIR)/include/. $(@D)/include/
	cp -r $(QEMU_KN1_KN1_SRC_DIR)/tests/. $(@D)/tests/
	cp $(addprefix $(QEMU_KN1_NPU_CORE_DIR)/,knpu_cmd.h knpu_core.h knpu_core.c) \
		$(@D)/hw/misc/
	$(APPLY_PATCHES) $(@D) $(QEMU_KN1_KN1_SRC_DIR)/patches \*.patch
endef
HOST_QEMU_KN1_POST_PATCH_HOOKS += HOST_QEMU_KN1_ADD_KN1_SOURCES

# legal-info saves the tarball and the applied patches but not the files
# copied in above; they go next to them, at their paths in the QEMU tree
# (docs/design.md 12.4)
define HOST_QEMU_KN1_SAVE_KN1_SOURCES
	cp -r $(addprefix $(QEMU_KN1_KN1_SRC_DIR)/,hw include tests) \
		$(HOST_QEMU_KN1_REDIST_SOURCES_DIR)/
	cp $(addprefix $(QEMU_KN1_NPU_CORE_DIR)/,knpu_cmd.h knpu_core.h knpu_core.c) \
		$(HOST_QEMU_KN1_REDIST_SOURCES_DIR)/hw/misc/
endef
HOST_QEMU_KN1_POST_LEGAL_INFO_HOOKS += HOST_QEMU_KN1_SAVE_KN1_SOURCES

# Override CPP, as it expects to be able to call it like it'd
# call the compiler (same as Buildroot's host-qemu).
define HOST_QEMU_KN1_CONFIGURE_CMDS
	unset TARGET_DIR; \
	cd $(@D); $(HOST_CONFIGURE_OPTS) CPP="$(HOSTCC) -E" \
		./configure \
		--target-list="$(HOST_QEMU_KN1_TARGETS)" \
		--prefix="$(HOST_DIR)" \
		--interp-prefix=$(STAGING_DIR) \
		--cc="$(HOSTCC)" \
		--host-cc="$(HOSTCC)" \
		--extra-cflags="$(HOST_QEMU_KN1_CFLAGS)" \
		--extra-ldflags="$(HOST_LDFLAGS)" \
		--python=$(HOST_DIR)/bin/python3 \
		--ninja=$(HOST_DIR)/bin/ninja \
		--disable-alsa \
		--disable-bpf \
		--disable-bzip2 \
		--disable-cap-ng \
		--disable-containers \
		--disable-coreaudio \
		--disable-curl \
		--disable-dbus-display \
		--disable-docs \
		--disable-download \
		--disable-dsound \
		--disable-guest-agent \
		--disable-jack \
		--disable-libssh \
		--disable-libusb \
		--disable-linux-aio \
		--disable-linux-io-uring \
		--disable-linux-user \
		--disable-netmap \
		--disable-oss \
		--disable-pa \
		--disable-pipewire \
		--disable-plugins \
		--disable-png \
		--disable-sdl \
		--disable-selinux \
		--disable-vde \
		--disable-vhost-user-blk-server \
		--disable-virtfs \
		--disable-vnc-jpeg \
		--disable-vnc-sasl \
		--disable-zstd \
		--enable-fdt \
		--enable-slirp \
		--enable-system \
		--enable-tools
endef

define HOST_QEMU_KN1_BUILD_CMDS
	unset TARGET_DIR; \
	$(HOST_MAKE_ENV) $(MAKE) -C $(@D)
endef

define HOST_QEMU_KN1_INSTALL_CMDS
	unset TARGET_DIR; \
	$(HOST_MAKE_ENV) $(MAKE) -C $(@D) install
endef

$(eval $(host-generic-package))
