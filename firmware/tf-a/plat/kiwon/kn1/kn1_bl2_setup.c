/*
 * Copyright (c) 2015-2025, Arm Limited and Contributors. All rights reserved.
 * Copyright (c) 2026, Kiwon Kwon. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <assert.h>

#include <platform_def.h>

#include <arch_features.h>
#include <arch_helpers.h>
#include <common/bl_common.h>
#include <common/debug.h>
#include <common/desc_image_load.h>
#include <lib/utils.h>
#include <plat/common/platform.h>

#include "kn1_private.h"

#define MAP_BL2_TOTAL		MAP_REGION_FLAT(			\
					bl2_tzram_layout.total_base,	\
					bl2_tzram_layout.total_size,	\
					MT_MEMORY | MT_RW | EL3_PAS)

#define MAP_BL2_RO		MAP_REGION_FLAT(			\
					BL_CODE_BASE,			\
					BL_CODE_END - BL_CODE_BASE,	\
					MT_CODE | EL3_PAS),		\
				MAP_REGION_FLAT(			\
					BL_RO_DATA_BASE,		\
					BL_RO_DATA_END			\
						- BL_RO_DATA_BASE,	\
					MT_RO_DATA | EL3_PAS)

#define MAP_BL_COHERENT_RAM	MAP_REGION_FLAT(			\
					BL_COHERENT_RAM_BASE,		\
					BL_COHERENT_RAM_END		\
						- BL_COHERENT_RAM_BASE,	\
					MT_DEVICE | MT_RW | EL3_PAS)

/* Data structure which holds the extents of the trusted SRAM for BL2 */
static meminfo_t bl2_tzram_layout __aligned(CACHE_WRITEBACK_GRANULE);

void bl2_early_platform_setup2(u_register_t arg0, u_register_t arg1,
			       u_register_t arg2, u_register_t arg3)
{
	meminfo_t *mem_layout = (void *)arg1;

	/* Initialize the console to provide early debug support */
	kn1_console_init();

	/* Setup the BL2 memory layout */
	bl2_tzram_layout = *mem_layout;

	plat_kn1_io_setup();
}

void bl2_platform_setup(void)
{
	/*
	 * This is where a TrustZone address space controller and other
	 * security related peripherals would be configured. The kn1 device
	 * tree is static and already describes PSCI, so BL2 edits no device
	 * tree: U-Boot has its own (OF_SEPARATE) and loads the kernel's from
	 * the boot disk.
	 */
}

void bl2_plat_arch_setup(void)
{
	const mmap_region_t bl_regions[] = {
		MAP_BL2_TOTAL,
		MAP_BL2_RO,
		MAP_BL_COHERENT_RAM,
		{0}
	};

	setup_page_tables(bl_regions, plat_kn1_get_mmap());
	enable_mmu_el1(0);
}

/*******************************************************************************
 * Gets SPSR for BL33 entry
 ******************************************************************************/
static uint32_t kn1_get_spsr_for_bl33_entry(void)
{
	unsigned int mode;

	/* Figure out what mode we enter the non-secure world in */
	mode = (el_implemented(2) != EL_IMPL_NONE) ? MODE_EL2 : MODE_EL1;

	return SPSR_64(mode, MODE_SP_ELX, DISABLE_ALL_EXCEPTIONS);
}

/*******************************************************************************
 * This function can be used by the platforms to update/use image
 * information for given `image_id`.
 ******************************************************************************/
int bl2_plat_handle_post_image_load(unsigned int image_id)
{
	bl_mem_params_node_t *bl_mem_params = get_bl_mem_params_node(image_id);

	assert(bl_mem_params);

	if (image_id == BL33_IMAGE_ID) {
		bl_mem_params->ep_info.spsr = kn1_get_spsr_for_bl33_entry();

		/* BL33 expects to receive the primary CPU MPID (through x0) */
		bl_mem_params->ep_info.args.arg0 = 0xffff & read_mpidr();
	}

	return 0;
}

uintptr_t plat_get_ns_image_entrypoint(void)
{
	return NS_IMAGE_OFFSET;
}
