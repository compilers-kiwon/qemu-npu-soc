#
# Copyright (c) 2013-2025, Arm Limited and Contributors. All rights reserved.
# Copyright (c) 2023-2025, Linaro Limited and Contributors. All rights reserved.
# Copyright (c) 2026, BayLibre SAS
# Copyright (c) 2026, Kiwon Kwon. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Kiwon kn1: one cluster of four Cortex-A72 cores, GICv3, PL011 UARTs, a
# secure PL061 GPIO that drives power off and reset, boot flash at the
# reset vector and 16 MiB of secure RAM. The SoC is modelled by the QEMU
# "kn1-evb" machine. Derived from plat/qemu/qemu with the optional
# features (OP-TEE, SPM, RME, SDEI, trusted and measured boot,
# semihosting, transfer list) left out.

PLAT_KN1_PATH		:=	plat/kiwon/kn1

include lib/xlat_tables_v2/xlat_tables.mk
include drivers/arm/gic/v3/gicv3.mk

PLAT_INCLUDES		:=	-I${PLAT_KN1_PATH}/			\
				-I${PLAT_KN1_PATH}/include

KN1_CPU_LIBS		:=	lib/cpus/aarch64/cortex_a72.S

PLAT_BL_COMMON_SOURCES	:=	${PLAT_KN1_PATH}/kn1_common.c		\
				${PLAT_KN1_PATH}/kn1_console.c		\
				drivers/arm/pl011/aarch64/pl011_console.S \
				${XLAT_TABLES_LIB_SRCS}

KN1_IO_SOURCES		:=	drivers/io/io_fip.c			\
				drivers/io/io_memmap.c			\
				drivers/io/io_storage.c			\
				${PLAT_KN1_PATH}/kn1_io_storage.c

BL1_SOURCES		+=	${KN1_CPU_LIBS}				\
				${KN1_IO_SOURCES}			\
				${PLAT_KN1_PATH}/aarch64/plat_helpers.S	\
				${PLAT_KN1_PATH}/kn1_bl1_setup.c

BL2_SOURCES		+=	${KN1_IO_SOURCES}			\
				common/desc_image_load.c		\
				${PLAT_KN1_PATH}/aarch64/plat_helpers.S	\
				${PLAT_KN1_PATH}/kn1_bl2_setup.c	\
				${PLAT_KN1_PATH}/kn1_bl2_mem_params_desc.c \
				${PLAT_KN1_PATH}/kn1_image_load.c

BL31_SOURCES		+=	${KN1_CPU_LIBS}				\
				${GICV3_SOURCES}			\
				drivers/arm/pl061/pl061_gpio.c		\
				drivers/gpio/gpio.c			\
				plat/common/plat_gicv3.c		\
				plat/common/plat_hold_pen.c		\
				plat/common/plat_psci_common.c		\
				${PLAT_KN1_PATH}/aarch64/plat_helpers.S	\
				${PLAT_KN1_PATH}/kn1_bl31_setup.c	\
				${PLAT_KN1_PATH}/kn1_gicv3.c		\
				${PLAT_KN1_PATH}/kn1_pm.c		\
				${PLAT_KN1_PATH}/kn1_topology.c

# Cortex-A72 is an Armv8.0-A core that sits behind a CCI, so it has no
# hardware-assisted coherency and EL3 keeps its shared data in coherent
# memory, as the Arm FVP port does for the same core.
ARM_ARCH_MAJOR		:=	8
ARM_ARCH_MINOR		:=	0
HW_ASSISTED_COHERENCY	:=	0
USE_COHERENT_MEM	:=	1

SEPARATE_CODE_AND_RODATA :=	1
