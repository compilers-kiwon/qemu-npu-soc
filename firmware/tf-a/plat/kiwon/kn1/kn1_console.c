/*
 * Copyright (c) 2015-2018, ARM Limited and Contributors. All rights reserved.
 * Copyright (c) 2026, Kiwon Kwon. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <platform_def.h>

#include <drivers/arm/pl011.h>
#include <drivers/console.h>

#include "kn1_private.h"

static console_t console;

/* TF-A talks on the secure UART1; UART0 belongs to the normal world. */
void kn1_console_init(void)
{
	(void)console_pl011_register(PLAT_KN1_BOOT_UART_BASE,
				     PLAT_KN1_BOOT_UART_CLK_IN_HZ,
				     PLAT_KN1_CONSOLE_BAUDRATE, &console);

	console_set_scope(&console, CONSOLE_FLAG_BOOT |
			  CONSOLE_FLAG_RUNTIME);
}
