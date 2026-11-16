/*
 * Copyright (c) 2015-2026, Arm Limited and Contributors. All rights reserved.
 * Copyright (c) 2026, Kiwon Kwon. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <assert.h>

#include <platform_def.h>

#include <arch.h>
#include <arch_helpers.h>
#include <common/bl_common.h>
#include <drivers/arm/pl061_gpio.h>
#include <drivers/console.h>
#include <plat/common/platform.h>

#include "kn1_private.h"

#define MAP_BL31_TOTAL		MAP_REGION_FLAT(			\
					BL31_BASE,			\
					BL31_END - BL31_BASE,		\
					MT_MEMORY | MT_RW | EL3_PAS)
#define MAP_BL31_RO		MAP_REGION_FLAT(			\
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

/*
 * Placeholder variable for copying the arguments that have been passed to
 * BL31 from BL2.
 */
static entry_point_info_t bl33_image_ep_info;

/*******************************************************************************
 * Perform any BL31 early platform setup.  Here is an opportunity to copy
 * parameters passed by the calling EL (S-EL1 in BL2 & EL3 in BL1) before
 * they are lost (potentially). This needs to be done before the MMU is
 * initialized so that the memory layout can be used while creating page
 * tables. BL2 has flushed this information to memory, so we are guaranteed
 * to pick up good data.
 ******************************************************************************/
void bl31_early_platform_setup2(u_register_t arg0, u_register_t arg1,
				u_register_t arg2, u_register_t arg3)
{
	/* Initialize the console to provide early debug support */
	kn1_console_init();

	/*
	 * Check params passed from BL2
	 */
	bl_params_t *params_from_bl2 = (bl_params_t *)arg0;

	assert(params_from_bl2);
	assert(params_from_bl2->h.type == PARAM_BL_PARAMS);
	assert(params_from_bl2->h.version >= VERSION_2);

	bl_params_node_t *bl_params = params_from_bl2->head;

	/*
	 * Copy the BL33 entry point information. It is stored in Secure
	 * RAM, in BL2's address space.
	 */
	while (bl_params) {
		if (bl_params->image_id == BL33_IMAGE_ID)
			bl33_image_ep_info = *bl_params->ep_info;

		bl_params = bl_params->next_params_info;
	}

	if (!bl33_image_ep_info.pc)
		panic();
}

void bl31_plat_arch_setup(void)
{
	const mmap_region_t bl_regions[] = {
		MAP_BL31_TOTAL,
		MAP_BL31_RO,
		MAP_BL_COHERENT_RAM,
		{0}
	};

	setup_page_tables(bl_regions, plat_kn1_get_mmap());

	enable_mmu_el3(0);
}

static void kn1_gpio_init(void)
{
	pl061_gpio_init();
	pl061_gpio_register(SECURE_GPIO_BASE, 0);
}

void bl31_platform_setup(void)
{
	plat_kn1_gic_init();
	kn1_gpio_init();
}

unsigned int plat_get_syscnt_freq2(void)
{
	return read_cntfrq_el0();
}

/*******************************************************************************
 * Return a pointer to the 'entry_point_info' structure of the next image
 * for the security state specified. BL33 corresponds to the non-secure
 * image type. kn1 has no BL32, so a NULL pointer is returned for the
 * secure image type.
 ******************************************************************************/
entry_point_info_t *bl31_plat_get_next_image_ep_info(uint32_t type)
{
	assert(sec_state_is_valid(type));

	if (type != NON_SECURE)
		return NULL;

	/* None of the images can have 0x0 as the entrypoint */
	if (bl33_image_ep_info.pc)
		return &bl33_image_ep_info;
	else
		return NULL;
}

void bl31_plat_runtime_setup(void)
{
	console_flush();
	console_switch_state(CONSOLE_FLAG_RUNTIME);
}
