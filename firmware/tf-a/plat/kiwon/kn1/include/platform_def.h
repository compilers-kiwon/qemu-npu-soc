/*
 * Copyright (c) 2015-2025, Arm Limited and Contributors. All rights reserved.
 * Copyright (c) 2026, BayLibre SAS
 * Copyright (c) 2026, Kiwon Kwon. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef PLATFORM_DEF_H
#define PLATFORM_DEF_H

#include <arch.h>
#include <common/tbbr/tbbr_img_def.h>
#include <drivers/arm/gic_common.h>
#include <lib/utils_def.h>
#include <plat/common/common_def.h>

#define PLATFORM_STACK_SIZE		0x1000

/* One cluster of four Cortex-A72 cores; the count is fixed by the SoC. */
#define PLATFORM_CLUSTER_COUNT		U(1)
#define PLATFORM_MAX_CPUS_PER_CLUSTER	U(4)
#define PLATFORM_CLUSTER0_CORE_COUNT	PLATFORM_MAX_CPUS_PER_CLUSTER
#define PLATFORM_CORE_COUNT		PLATFORM_CLUSTER0_CORE_COUNT

#define KN1_PRIMARY_CPU			U(0)

#define PLAT_NUM_PWR_DOMAINS		(PLATFORM_CLUSTER_COUNT + \
					 PLATFORM_CORE_COUNT)
#define PLAT_MAX_PWR_LVL		MPIDR_AFFLVL1

#define PLAT_MAX_RET_STATE		U(1)
#define PLAT_MAX_OFF_STATE		U(2)

/* Local power state for power domains in Run state. */
#define PLAT_LOCAL_STATE_RUN		U(0)
/* Local power state for retention. Valid only for CPU power domains. */
#define PLAT_LOCAL_STATE_RET		U(1)
/* Local power state for OFF/power-down. Valid for CPU and cluster domains. */
#define PLAT_LOCAL_STATE_OFF		2

/*
 * Macros used to parse state information from State-ID if it is using the
 * recommended encoding for State-ID.
 */
#define PLAT_LOCAL_PSTATE_WIDTH		4
#define PLAT_LOCAL_PSTATE_MASK		((1 << PLAT_LOCAL_PSTATE_WIDTH) - 1)

/* Cortex-A72 cache line: 64 bytes. */
#define CACHE_WRITEBACK_SHIFT		6
#define CACHE_WRITEBACK_GRANULE		(1 << CACHE_WRITEBACK_SHIFT)

/*
 * SoC memory map (see docs/design.md section 5.2). The boot flash holds
 * BL1 at the reset vector and the FIP at KN1_FIP_OFFSET. TF-A uses the
 * first 1 MiB of the 16 MiB secure RAM; the rest is unused for now.
 */
#define KN1_FLASH_BASE			0x00000000
#define KN1_FLASH_SIZE			0x04000000
#define KN1_FIP_OFFSET			0x00040000

#define KN1_SEC_RAM_BASE		0x0e000000
#define KN1_SEC_RAM_SIZE		0x01000000

#define KN1_DRAM_BASE			ULL(0x40000000)
#define KN1_DRAM_SIZE			ULL(0x100000000)

#define SEC_SRAM_BASE			KN1_SEC_RAM_BASE
#define SEC_SRAM_SIZE			0x00100000

/* Secure PL061 GPIO: line 0 is power off, line 1 is reset (design D23). */
#define SECURE_GPIO_BASE		0x090b0000
#define SECURE_GPIO_SIZE		0x00001000
#define SECURE_GPIO_POWEROFF		0
#define SECURE_GPIO_RESET		1

/*
 * TF-A lives in the secure SRAM, partition it here. The first page is
 * shared between the primary and the secondary cores for the hold pen.
 */
#define SHARED_RAM_BASE			SEC_SRAM_BASE
#define SHARED_RAM_SIZE			0x00001000

#define PLAT_KN1_HOLD_BASE		SHARED_RAM_BASE
#define PLAT_KN1_HOLD_SIZE		(PLATFORM_CORE_COUNT * HOLD_SLOT_SIZE)

#define BL_RAM_BASE			(SHARED_RAM_BASE + SHARED_RAM_SIZE)
#define BL_RAM_SIZE			(SEC_SRAM_SIZE - SHARED_RAM_SIZE)

/*
 * BL1 specific defines.
 *
 * BL1 RO data lives in the boot flash below the FIP. BL1 RW data is
 * relocated from flash to RAM at runtime, at the top of the secure SRAM.
 * BL1_RW_BASE is calculated using the current BL1 RW debug size plus a
 * little space for growth.
 */
#define BL1_RO_BASE			KN1_FLASH_BASE
#define BL1_RO_LIMIT			(KN1_FLASH_BASE + KN1_FIP_OFFSET)
#define BL1_RW_BASE			(BL1_RW_LIMIT - 0x12000)
#define BL1_RW_LIMIT			(BL_RAM_BASE + BL_RAM_SIZE)

/*
 * BL2 specific defines.
 *
 * Put BL2 just below BL31. BL2_BASE is calculated using the current BL2
 * debug size plus a little space for growth.
 */
#define BL2_BASE			(BL31_BASE - 0x35000)
#define BL2_LIMIT			BL31_BASE

/*
 * BL31 specific defines.
 *
 * Put BL31 at the top of the secure SRAM. BL31_BASE is calculated using
 * the current BL31 debug size plus a little space for growth.
 */
#define BL31_BASE			(BL31_LIMIT - 0x70000)
#define BL31_LIMIT			(BL_RAM_BASE + BL_RAM_SIZE)
#define BL31_PROGBITS_LIMIT		BL1_RW_BASE

/* BL33 (U-Boot) is loaded 512 MiB into DRAM. */
#define NS_IMAGE_OFFSET			(KN1_DRAM_BASE + 0x20000000)
#define NS_IMAGE_MAX_SIZE		(KN1_DRAM_SIZE - 0x20000000)

/* DRAM ends at 5 GiB, so the address space needs 33 bits. */
#define PLAT_PHY_ADDR_SPACE_SIZE	(1ULL << 33)
#define PLAT_VIRT_ADDR_SPACE_SIZE	(1ULL << 33)
#define MAX_MMAP_REGIONS		12
#define MAX_XLAT_TABLES			8
#define MAX_IO_DEVICES			2
#define MAX_IO_HANDLES			4

/*
 * PL011 UARTs. UART0 is the non-secure console for the normal world;
 * UART1 is secure and used by TF-A. The design fixes the UART clock at
 * 24 MHz (the fixed clock in kn1.dtsi).
 */
#define KN1_UART0_BASE			0x09000000
#define KN1_UART1_BASE			0x09040000
#define KN1_UART_CLK_IN_HZ		24000000

#define PLAT_KN1_BOOT_UART_BASE		KN1_UART1_BASE
#define PLAT_KN1_BOOT_UART_CLK_IN_HZ	KN1_UART_CLK_IN_HZ

#define PLAT_KN1_CRASH_UART_BASE	KN1_UART1_BASE
#define PLAT_KN1_CRASH_UART_CLK_IN_HZ	KN1_UART_CLK_IN_HZ

#define PLAT_KN1_CONSOLE_BAUDRATE	115200

#define PLAT_KN1_FIP_BASE		(KN1_FLASH_BASE + KN1_FIP_OFFSET)
#define PLAT_KN1_FIP_MAX_SIZE		(KN1_FLASH_SIZE - KN1_FIP_OFFSET)

/* Device windows: the GIC, then the UARTs, RTC and GPIO. */
#define DEVICE0_BASE			0x08000000
#define DEVICE0_SIZE			0x01000000
#define DEVICE1_BASE			0x09000000
#define DEVICE1_SIZE			0x00100000

/*
 * GICv3 related constants
 */
#define GICD_BASE			0x08000000
#define GICR_BASE			0x080a0000

#define KN1_IRQ_SEC_SGI_0		8
#define KN1_IRQ_SEC_SGI_1		9
#define KN1_IRQ_SEC_SGI_2		10
#define KN1_IRQ_SEC_SGI_3		11
#define KN1_IRQ_SEC_SGI_4		12
#define KN1_IRQ_SEC_SGI_5		13
#define KN1_IRQ_SEC_SGI_6		14
#define KN1_IRQ_SEC_SGI_7		15

/* The secure SGIs are the only Group 1 Secure interrupts configured. */
#define PLATFORM_G1S_PROPS(grp)						\
	INTR_PROP_DESC(KN1_IRQ_SEC_SGI_0, GIC_HIGHEST_SEC_PRIORITY,	\
					  grp, GIC_INTR_CFG_EDGE),	\
	INTR_PROP_DESC(KN1_IRQ_SEC_SGI_1, GIC_HIGHEST_SEC_PRIORITY,	\
					  grp, GIC_INTR_CFG_EDGE),	\
	INTR_PROP_DESC(KN1_IRQ_SEC_SGI_2, GIC_HIGHEST_SEC_PRIORITY,	\
					  grp, GIC_INTR_CFG_EDGE),	\
	INTR_PROP_DESC(KN1_IRQ_SEC_SGI_3, GIC_HIGHEST_SEC_PRIORITY,	\
					  grp, GIC_INTR_CFG_EDGE),	\
	INTR_PROP_DESC(KN1_IRQ_SEC_SGI_4, GIC_HIGHEST_SEC_PRIORITY,	\
					  grp, GIC_INTR_CFG_EDGE),	\
	INTR_PROP_DESC(KN1_IRQ_SEC_SGI_5, GIC_HIGHEST_SEC_PRIORITY,	\
					  grp, GIC_INTR_CFG_EDGE),	\
	INTR_PROP_DESC(KN1_IRQ_SEC_SGI_6, GIC_HIGHEST_SEC_PRIORITY,	\
					  grp, GIC_INTR_CFG_EDGE),	\
	INTR_PROP_DESC(KN1_IRQ_SEC_SGI_7, GIC_HIGHEST_SEC_PRIORITY,	\
					  grp, GIC_INTR_CFG_EDGE)

#endif /* PLATFORM_DEF_H */
