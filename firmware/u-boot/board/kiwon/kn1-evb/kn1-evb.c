// SPDX-License-Identifier: GPL-2.0-or-later
// SPDX-FileCopyrightText: 2026 Kiwon Kwon
/*
 * Kiwon kn1-evb board. U-Boot runs as the TF-A BL33 in non-secure EL2 and
 * takes everything it needs to know about the board from its own control
 * device tree (kn1-evb.dts, OF_SEPARATE).
 */

#include <fdtdec.h>
#include <init.h>
#include <asm/armv8/mmu.h>
#include <asm/global_data.h>
#include <linux/sizes.h>

DECLARE_GLOBAL_DATA_PTR;

#define KN1_MEM_MAP_DRAM	1

static struct mm_region kn1_evb_mem_map[] = {
	{
		/*
		 * Peripherals up to the start of DRAM. The boot flash below
		 * them belongs to the secure world and is not mapped.
		 */
		.virt = 0x08000000UL,
		.phys = 0x08000000UL,
		.size = 0x38000000UL,
		.attrs = PTE_BLOCK_MEMTYPE(MT_DEVICE_NGNRNE) |
			 PTE_BLOCK_NON_SHARE |
			 PTE_BLOCK_PXN | PTE_BLOCK_UXN
	}, {
		/* DRAM; base and size are filled in from the device tree */
		.virt = 0x40000000UL,
		.phys = 0x40000000UL,
		.size = SZ_4G,
		.attrs = PTE_BLOCK_MEMTYPE(MT_NORMAL) |
			 PTE_BLOCK_INNER_SHARE
	}, {
		/* List terminator */
		0,
	}
};

struct mm_region *mem_map = kn1_evb_mem_map;

int dram_init(void)
{
	int ret;

	/*
	 * fdtdec_setup_mem_size_base() would take the first /memory node,
	 * which is the secure on-chip RAM (status = "disabled"). The
	 * _lowest() variant only looks at enabled memory nodes.
	 */
	ret = fdtdec_setup_mem_size_base_lowest();
	if (ret)
		return ret;

	kn1_evb_mem_map[KN1_MEM_MAP_DRAM].virt = gd->ram_base;
	kn1_evb_mem_map[KN1_MEM_MAP_DRAM].phys = gd->ram_base;
	kn1_evb_mem_map[KN1_MEM_MAP_DRAM].size = gd->ram_size;

	return 0;
}

int dram_init_banksize(void)
{
	return fdtdec_setup_memory_banksize();
}

/*
 * Keep U-Boot, the device trees and the images it loads below 4 GiB.
 * Relocating to the top of DRAM overlaps the NPU CMA pool in its last
 * 64 MiB (kn1-evb.dts): LMB cannot reserve the pool and the kernel's
 * device tree lands inside it, where Linux hands the pages to CMA. TI's
 * AM65x and J721E boards limit U-Boot the same way.
 */
phys_addr_t board_get_usable_ram_top(phys_size_t total_size)
{
	if (gd->ram_top > SZ_4G)
		return SZ_4G;

	return gd->ram_top;
}
