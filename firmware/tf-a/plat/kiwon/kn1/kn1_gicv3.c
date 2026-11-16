/*
 * Copyright (c) 2019, Linaro Limited and Contributors. All rights reserved.
 * Copyright (c) 2026, Kiwon Kwon. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <platform_def.h>

#include <drivers/arm/gic_common.h>
#include <drivers/arm/gicv3.h>
#include <plat/common/platform.h>

#include "kn1_private.h"

static const interrupt_prop_t kn1_interrupt_props[] = {
	PLATFORM_G1S_PROPS(INTR_GROUP1S)
};

static uintptr_t kn1_rdistif_base_addrs[PLATFORM_CORE_COUNT];

static unsigned int kn1_mpidr_to_core_pos(unsigned long mpidr)
{
	return (unsigned int)plat_core_pos_by_mpidr(mpidr);
}

static const gicv3_driver_data_t kn1_gicv3_driver_data = {
	.gicd_base = GICD_BASE,
	.gicr_base = GICR_BASE,
	.interrupt_props = kn1_interrupt_props,
	.interrupt_props_num = ARRAY_SIZE(kn1_interrupt_props),
	.rdistif_num = PLATFORM_CORE_COUNT,
	.rdistif_base_addrs = kn1_rdistif_base_addrs,
	.mpidr_to_core_pos = kn1_mpidr_to_core_pos
};

void plat_kn1_gic_init(void)
{
	gicv3_driver_init(&kn1_gicv3_driver_data);
	gicv3_distif_init();
	gicv3_rdistif_init(plat_my_core_pos());
	gicv3_cpuif_enable(plat_my_core_pos());
}

void kn1_pwr_gic_on_finish(void)
{
	gicv3_rdistif_init(plat_my_core_pos());
	gicv3_cpuif_enable(plat_my_core_pos());
}

void kn1_pwr_gic_off(void)
{
	gicv3_cpuif_disable(plat_my_core_pos());
	gicv3_rdistif_off(plat_my_core_pos());
}
