/* SPDX-License-Identifier: GPL-2.0-only OR MIT */
/* Copyright (c) 2026 Kiwon Kwon */
/* kn1-npu registers (docs/npu-spec.md 3) */

#ifndef __KNPU_DEVICE_H__
#define __KNPU_DEVICE_H__

#include <linux/bits.h>

#define KNPU_REG_ID			0x000
#define KNPU_REG_VERSION		0x004
#define   KNPU_VERSION_MAJOR_MASK	GENMASK(31, 16)
#define   KNPU_VERSION_MINOR_MASK	GENMASK(15, 0)
#define KNPU_REG_CONFIG			0x008
#define   KNPU_CONFIG_NUM_REGIONS_MASK	GENMASK(3, 0)
#define   KNPU_CONFIG_ROUND_DOUBLE	BIT(8)
#define   KNPU_CONFIG_ROUND_SINGLE	BIT(9)
#define KNPU_REG_SRAM_SIZE		0x00c
#define KNPU_REG_CTRL			0x010
#define   KNPU_CTRL_START		BIT(0)
#define   KNPU_CTRL_SOFT_RESET		BIT(1)
#define KNPU_REG_STATUS			0x014
#define   KNPU_STATUS_BUSY		BIT(0)
#define   KNPU_STATUS_DONE		BIT(1)
#define   KNPU_STATUS_ERROR		BIT(2)
#define   KNPU_STATUS_ERR_CODE_MASK	GENMASK(15, 8)
#define KNPU_REG_ERR_OFFSET		0x018
#define KNPU_REG_IRQ_STATUS		0x020
#define KNPU_REG_IRQ_ENABLE		0x024
#define   KNPU_IRQ_DONE			BIT(0)
#define   KNPU_IRQ_ERROR		BIT(1)
#define KNPU_REG_QBASE_LO		0x030
#define KNPU_REG_QBASE_HI		0x034
#define KNPU_REG_QSIZE			0x038
#define KNPU_REG_REGION_BASE_LO(n)	(0x100 + 8 * (n))
#define KNPU_REG_REGION_BASE_HI(n)	(0x104 + 8 * (n))
#define KNPU_REG_PERF_CTRL		0x200
#define   KNPU_PERF_ENABLE		BIT(0)
#define   KNPU_PERF_CLEAR		BIT(1)
#define KNPU_REG_PERF_LO(n)		(0x208 + 8 * (n))
#define KNPU_REG_PERF_HI(n)		(0x20c + 8 * (n))

#endif /* __KNPU_DEVICE_H__ */
