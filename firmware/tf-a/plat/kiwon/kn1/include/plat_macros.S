/*
 * Copyright (c) 2016, ARM Limited and Contributors. All rights reserved.
 * Copyright (c) 2026, Kiwon Kwon. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef PLAT_MACROS_S
#define PLAT_MACROS_S

	/* ---------------------------------------------
	 * The below required platform porting macro
	 * prints out relevant platform registers
	 * whenever an unhandled exception is taken in
	 * BL31. The kn1 port prints nothing beyond the
	 * generic crash report.
	 * ---------------------------------------------
	 */
	.macro plat_crash_print_regs
	.endm

#endif /* PLAT_MACROS_S */
