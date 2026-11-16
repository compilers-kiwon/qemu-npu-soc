/*
 * Copyright (c) 2015-2023, Arm Limited and Contributors. All rights reserved.
 * Copyright (c) 2026, Kiwon Kwon. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef KN1_PRIVATE_H
#define KN1_PRIVATE_H

#include <stdint.h>

#include <lib/xlat_tables/xlat_tables_v2.h>

void plat_kn1_io_setup(void);
unsigned int plat_kn1_calc_core_pos(u_register_t mpidr);
const mmap_region_t *plat_kn1_get_mmap(void);

void kn1_console_init(void);

void plat_kn1_gic_init(void);
void kn1_pwr_gic_on_finish(void);
void kn1_pwr_gic_off(void);

#endif /* KN1_PRIVATE_H */
