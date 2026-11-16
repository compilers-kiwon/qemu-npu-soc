/*
 * Copyright (c) 2015-2019, ARM Limited and Contributors. All rights reserved.
 * Copyright (c) 2026, BayLibre SAS
 * Copyright (c) 2026, Kiwon Kwon. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <assert.h>

#include <platform_def.h>

#include <arch_helpers.h>
#include <common/debug.h>
#include <drivers/gpio.h>
#include <lib/psci/psci.h>
#include <plat/common/plat_hold_pen.h>
#include <plat/common/platform.h>

#include "kn1_private.h"

/*
 * The secure entry point to be used on warm reset.
 */
static uintptr_t secure_entrypoint;

/* Make composite power state parameter till power level 0 */
#if PSCI_EXTENDED_STATE_ID

#define kn1_make_pwrstate_lvl0(lvl0_state, pwr_lvl, type) \
		(((lvl0_state) << PSTATE_ID_SHIFT) | \
		 ((type) << PSTATE_TYPE_SHIFT))
#else
#define kn1_make_pwrstate_lvl0(lvl0_state, pwr_lvl, type) \
		(((lvl0_state) << PSTATE_ID_SHIFT) | \
		 ((pwr_lvl) << PSTATE_PWR_LVL_SHIFT) | \
		 ((type) << PSTATE_TYPE_SHIFT))
#endif /* PSCI_EXTENDED_STATE_ID */

#define kn1_make_pwrstate_lvl1(lvl1_state, lvl0_state, pwr_lvl, type) \
		(((lvl1_state) << PLAT_LOCAL_PSTATE_WIDTH) | \
		 kn1_make_pwrstate_lvl0(lvl0_state, pwr_lvl, type))

/*
 *  The table storing the valid idle power states. Ensure that the
 *  array entries are populated in ascending order of state-id to
 *  enable us to use binary search during power state validation.
 *  The table must be terminated by a NULL entry.
 */
static const unsigned int kn1_pm_idle_states[] = {
	/* State-id - 0x01 */
	kn1_make_pwrstate_lvl1(PLAT_LOCAL_STATE_RUN, PLAT_LOCAL_STATE_RET,
			       MPIDR_AFFLVL0, PSTATE_TYPE_STANDBY),
	/* State-id - 0x02 */
	kn1_make_pwrstate_lvl1(PLAT_LOCAL_STATE_RUN, PLAT_LOCAL_STATE_OFF,
			       MPIDR_AFFLVL0, PSTATE_TYPE_POWERDOWN),
	/* State-id - 0x22 */
	kn1_make_pwrstate_lvl1(PLAT_LOCAL_STATE_OFF, PLAT_LOCAL_STATE_OFF,
			       MPIDR_AFFLVL1, PSTATE_TYPE_POWERDOWN),
	0,
};

/*******************************************************************************
 * Platform handler called to check the validity of the power state
 * parameter. The power state parameter has to be a composite power state.
 ******************************************************************************/
static int kn1_validate_power_state(unsigned int power_state,
				    psci_power_state_t *req_state)
{
	unsigned int state_id;
	int i;

	assert(req_state);

	/*
	 *  Currently we are using a linear search for finding the matching
	 *  entry in the idle power state array. This can be made a binary
	 *  search if the number of entries justify the additional complexity.
	 */
	for (i = 0; !!kn1_pm_idle_states[i]; i++) {
		if (power_state == kn1_pm_idle_states[i])
			break;
	}

	/* Return error if entry not found in the idle state array */
	if (!kn1_pm_idle_states[i])
		return PSCI_E_INVALID_PARAMS;

	i = 0;
	state_id = psci_get_pstate_id(power_state);

	/* Parse the State ID and populate the state info parameter */
	while (state_id) {
		req_state->pwr_domain_state[i++] = state_id &
						PLAT_LOCAL_PSTATE_MASK;
		state_id >>= PLAT_LOCAL_PSTATE_WIDTH;
	}

	return PSCI_E_SUCCESS;
}

/*******************************************************************************
 * Platform handler called when a CPU is about to enter standby.
 ******************************************************************************/
static void kn1_cpu_standby(plat_local_state_t cpu_state)
{
	assert(cpu_state == PLAT_LOCAL_STATE_RET);

	/*
	 * Enter standby state
	 * dsb is good practice before using wfi to enter low power states
	 */
	dsb();
	wfi();
}

/*******************************************************************************
 * Platform handler called when a power domain is about to be turned on. The
 * mpidr determines the CPU to be turned on. All cores leave reset together
 * on kn1, so the secondaries wait in the hold pen until signalled here.
 ******************************************************************************/
static int kn1_pwr_domain_on(u_register_t mpidr)
{
	unsigned int pos = plat_core_pos_by_mpidr(mpidr);

	plat_hold_pen_signal((struct hold_slot *)PLAT_KN1_HOLD_BASE,
			     pos, secure_entrypoint);

	return PSCI_E_SUCCESS;
}

/*******************************************************************************
 * Platform handler called when a power domain is about to be turned off. The
 * target_state encodes the power state that each level should transition to.
 ******************************************************************************/
static void kn1_pwr_domain_off(const psci_power_state_t *target_state)
{
	kn1_pwr_gic_off();
}

void __dead2 plat_secondary_cold_boot_setup(void);

static void __dead2
kn1_pwr_domain_pwr_down_wfi(const psci_power_state_t *target_state)
{
	disable_mmu_el3();
	plat_secondary_cold_boot_setup();
}

/*******************************************************************************
 * Platform handler called when a power domain is about to be suspended. The
 * target_state encodes the power state that each level should transition to.
 ******************************************************************************/
static void kn1_pwr_domain_suspend(const psci_power_state_t *target_state)
{
	assert(0);
}

/*******************************************************************************
 * Platform handler called when a power domain has just been powered on after
 * being turned off earlier. The target_state encodes the low power state that
 * each level has woken up from.
 ******************************************************************************/
static void kn1_pwr_domain_on_finish(const psci_power_state_t *target_state)
{
	assert(target_state->pwr_domain_state[MPIDR_AFFLVL0] ==
					PLAT_LOCAL_STATE_OFF);

	kn1_pwr_gic_on_finish();
}

/*******************************************************************************
 * Platform handler called when a power domain has just been powered on after
 * having been suspended earlier. The target_state encodes the low power state
 * that each level has woken up from.
 ******************************************************************************/
static void kn1_pwr_domain_suspend_finish(const psci_power_state_t *target_state)
{
	assert(0);
}

/*******************************************************************************
 * Platform handlers to shutdown/reboot the system. Both drive a line of the
 * secure GPIO; on the board the lines go to the power controller (design
 * D23), which acts on the rising edge. The controller cuts or resets the
 * power asynchronously (QEMU's gpio-pwr queues the request for its main
 * loop), so the CPU waits for it rather than reporting a failure.
 ******************************************************************************/
static void __dead2 kn1_wait_for_power_controller(void)
{
	for (;;) {
		wfi();
	}
}

static void __dead2 kn1_system_off(void)
{
	NOTICE("kn1: system off\n");
	gpio_set_direction(SECURE_GPIO_POWEROFF, GPIO_DIR_OUT);
	gpio_set_value(SECURE_GPIO_POWEROFF, GPIO_LEVEL_LOW);
	gpio_set_value(SECURE_GPIO_POWEROFF, GPIO_LEVEL_HIGH);
	kn1_wait_for_power_controller();
}

static void __dead2 kn1_system_reset(void)
{
	NOTICE("kn1: system reset\n");
	gpio_set_direction(SECURE_GPIO_RESET, GPIO_DIR_OUT);
	gpio_set_value(SECURE_GPIO_RESET, GPIO_LEVEL_LOW);
	gpio_set_value(SECURE_GPIO_RESET, GPIO_LEVEL_HIGH);
	kn1_wait_for_power_controller();
}

static const plat_psci_ops_t plat_kn1_psci_pm_ops = {
	.cpu_standby = kn1_cpu_standby,
	.pwr_domain_on = kn1_pwr_domain_on,
	.pwr_domain_off = kn1_pwr_domain_off,
	.pwr_domain_pwr_down = kn1_pwr_domain_pwr_down_wfi,
	.pwr_domain_suspend = kn1_pwr_domain_suspend,
	.pwr_domain_on_finish = kn1_pwr_domain_on_finish,
	.pwr_domain_suspend_finish = kn1_pwr_domain_suspend_finish,
	.system_off = kn1_system_off,
	.system_reset = kn1_system_reset,
	.validate_power_state = kn1_validate_power_state,
};

int plat_setup_psci_ops(uintptr_t sec_entrypoint,
			const plat_psci_ops_t **psci_ops)
{
	secure_entrypoint = sec_entrypoint;
	plat_hold_pen_init((struct hold_slot *)PLAT_KN1_HOLD_BASE,
			   PLATFORM_CORE_COUNT);
	*psci_ops = &plat_kn1_psci_pm_ops;

	return 0;
}
