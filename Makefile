# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Entry points for building and testing (README.md, docs/design.md 11);
# CI calls the same targets. `make help` lists them.

# Buildroot output directory (made absolute: Buildroot runs in buildroot/)
O ?= output
override O := $(abspath $(O))
# The MLIR CMake package of the compiler (apt.llvm.org LLVM 23)
MLIR_DIR ?= /usr/lib/llvm-23/lib/cmake/mlir

BUILD_DIR := $(CURDIR)/build
PY_VENV := $(BUILD_DIR)/py-venv
FLATBUFFERS_VERSION := 25.9.23
PY_STAMP := $(PY_VENV)/.flatbuffers-$(FLATBUFFERS_VERSION)

# Buildroot stops when PATH has spaces (buildroot/support/dependencies/
# dependencies.sh), and WSL adds Windows directories that have them, so
# Buildroot runs with PATH without those entries.
BR_PATH := $(shell printf '%s' "$$PATH" | tr ':' '\n' | \
                   grep -v '[[:space:]]' | paste -sd: -)
BR_MAKE = PATH='$(BR_PATH)' $(MAKE) -C buildroot O='$(O)' \
          BR2_EXTERNAL='$(CURDIR)/br2-external'

TEST_ARGS = --images '$(O)/images' --host-bin '$(O)/host/bin' \
            --log-dir '$(O)/test-logs'

# The tests boot QEMU one at a time. Buildroot and the CMake builds still
# run their own jobs in parallel.
.NOTPARALLEL:

.PHONY: all defconfig run test test-boot test-v1 test-v5 test-v6 check \
        check-npu-core compiler check-compiler update-vectors \
        check-runtime help FORCE

# The full image in $(O)/images
all: $(O)/.config
	$(BR_MAKE)

# The first time, and again when the defconfig changed
$(O)/.config: br2-external/configs/kn1_evb_defconfig
	$(BR_MAKE) kn1_evb_defconfig

defconfig:
	$(BR_MAKE) kn1_evb_defconfig

run:
	'$(O)/images/start-qemu.sh'

# A Buildroot target: make br-linux-rebuild, make br-menuconfig, ...
br-%: FORCE
	$(BR_MAKE) $*

FORCE:

# Tests on the image (docs/design.md 11)
test: test-boot test-v1 test-v5 test-v6

test-boot:
	python3 tests/boot/kn1_boot_test.py $(TEST_ARGS)

test-v1:
	tests/qemu/run-v1.sh '$(O)'

test-v5:
	python3 tests/driver/kn1_v5_test.py $(TEST_ARGS)

test-v6:
	python3 tests/e2e/kn1_v6_test.py $(TEST_ARGS)

# Host tests
check: check-npu-core check-compiler check-runtime

check-npu-core:
	cmake -S tests/npu-core -B '$(BUILD_DIR)/npu-core'
	cmake --build '$(BUILD_DIR)/npu-core' --parallel
	ctest --test-dir '$(BUILD_DIR)/npu-core' --output-on-failure

# The generator's Python package, in a venv of its own
$(PY_STAMP):
	rm -rf '$(PY_VENV)'
	python3 -m venv '$(PY_VENV)'
	'$(PY_VENV)/bin/pip' install --quiet flatbuffers==$(FLATBUFFERS_VERSION)
	touch '$@'

compiler: $(PY_STAMP)
	cmake -S compiler -B '$(BUILD_DIR)/compiler' -G Ninja \
	      -DMLIR_DIR='$(MLIR_DIR)' \
	      -DPython3_EXECUTABLE='$(PY_VENV)/bin/python3'
	cmake --build '$(BUILD_DIR)/compiler'

check-compiler: compiler
	ctest --test-dir '$(BUILD_DIR)/compiler' --output-on-failure

# Writes models/conv/vectors/ again; commit them when they change
update-vectors: compiler
	cmake --build '$(BUILD_DIR)/compiler' --target update-conv-vectors

# Rust 1.97.1: the cargo of the full build if there is one, else the
# system's
check-runtime:
	cd runtime && PATH='$(O)/host/bin':"$$PATH" cargo test --locked --offline

help:
	@echo 'Image (Buildroot, in $(O)):'
	@echo '  make                 build the full image'
	@echo '  make defconfig       load kn1_evb_defconfig again'
	@echo '  make run             boot the image in QEMU'
	@echo '  make br-<target>     a Buildroot target, e.g. br-linux-rebuild'
	@echo 'Tests on the image:'
	@echo '  make test            test-boot, test-v1, test-v5 and test-v6'
	@echo 'Host tests (in $(BUILD_DIR)):'
	@echo '  make check           check-npu-core, check-compiler and check-runtime'
	@echo '  make compiler        build the compiler without testing it'
	@echo '  make update-vectors  write models/conv/vectors/ again'
	@echo 'Variables: O=<Buildroot output directory>, MLIR_DIR=<MLIR CMake package>'
