/*
 * Copyright (c) 2015-2025, Arm Limited and Contributors. All rights reserved.
 * Copyright (c) 2026, Kiwon Kwon. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <platform_def.h>

#include <lib/xlat_tables/xlat_tables_v2.h>

#include "kn1_private.h"

#define MAP_DEVICE0	MAP_REGION_FLAT(DEVICE0_BASE,			\
					DEVICE0_SIZE,			\
					MT_DEVICE | MT_RW | EL3_PAS)

#define MAP_DEVICE1	MAP_REGION_FLAT(DEVICE1_BASE,			\
					DEVICE1_SIZE,			\
					MT_DEVICE | MT_RW | EL3_PAS)

#define MAP_SHARED_RAM	MAP_REGION_FLAT(SHARED_RAM_BASE,		\
					SHARED_RAM_SIZE,		\
					MT_DEVICE | MT_RW | EL3_PAS)

#define MAP_NS_DRAM0	MAP_REGION_FLAT(KN1_DRAM_BASE, KN1_DRAM_SIZE,	\
					MT_MEMORY | MT_RW | MT_NS)

#define MAP_FLASH0	MAP_REGION_FLAT(KN1_FLASH_BASE, KN1_FLASH_SIZE,	\
					MT_MEMORY | MT_RO | EL3_PAS)

/*
 * Table of regions for various BL stages to map using the MMU.
 * This doesn't include TZRAM as the 'mem_layout' argument passed to
 * setup_page_tables() will give the available subset of that.
 */
#ifdef IMAGE_BL1
static const mmap_region_t plat_kn1_mmap[] = {
	MAP_FLASH0,
	MAP_SHARED_RAM,
	MAP_DEVICE0,
	MAP_DEVICE1,
	{0}
};
#endif
#ifdef IMAGE_BL2
static const mmap_region_t plat_kn1_mmap[] = {
	MAP_FLASH0,
	MAP_SHARED_RAM,
	MAP_DEVICE0,
	MAP_DEVICE1,
	MAP_NS_DRAM0,
	{0}
};
#endif
#ifdef IMAGE_BL31
static const mmap_region_t plat_kn1_mmap[] = {
	MAP_SHARED_RAM,
	MAP_DEVICE0,
	MAP_DEVICE1,
	{0}
};
#endif

/*******************************************************************************
 * Returns kn1 platform specific memory map regions.
 ******************************************************************************/
const mmap_region_t *plat_kn1_get_mmap(void)
{
	return plat_kn1_mmap;
}
