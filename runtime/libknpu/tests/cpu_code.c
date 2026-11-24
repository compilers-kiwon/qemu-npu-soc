/*
 * A shared object in the shape of the CPU code of a model, for
 * tests/cpu.rs: int8 arrays and a function without arguments that the CPU
 * steps name (docs/model-format.md 9). ADD, given when it is compiled,
 * tells two builds apart.
 *
 * SPDX-FileCopyrightText: 2026 Kiwon Kwon
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

signed char act7[10];
signed char act8[10];

void layer8(void)
{
	int i;

	for (i = 0; i < 10; i++)
		act8[i] = (signed char)(act7[i] + ADD);
}
