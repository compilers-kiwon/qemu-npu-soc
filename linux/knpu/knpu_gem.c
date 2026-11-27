// SPDX-License-Identifier: GPL-2.0-only OR MIT
/* Copyright (c) 2026 Kiwon Kwon */
/* Buffer objects and the command stream validator, after ethosu_gem.c */

#include <linux/err.h>
#include <linux/overflow.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include <drm/drm_file.h>
#include <drm/drm_gem.h>
#include <drm/drm_prime.h>
#include <drm/drm_print.h>

#include "knpu_drv.h"
#include "knpu_gem.h"

static void knpu_gem_free_object(struct drm_gem_object *obj)
{
	struct knpu_gem_object *bo = to_knpu_bo(obj);

	kfree(bo->info);
	drm_gem_dma_free(&bo->base);
}

static int knpu_gem_mmap(struct drm_gem_object *obj, struct vm_area_struct *vma)
{
	/* A validated command stream must stay as it was checked */
	if (to_knpu_bo(obj)->info)
		return -EINVAL;

	return drm_gem_dma_object_mmap(obj, vma);
}

static struct dma_buf *knpu_gem_export(struct drm_gem_object *obj, int flags)
{
	/* Nor may another device write it through a dma-buf */
	if (to_knpu_bo(obj)->info)
		return ERR_PTR(-EINVAL);

	return drm_gem_prime_export(obj, flags);
}

static const struct drm_gem_object_funcs knpu_gem_funcs = {
	.free = knpu_gem_free_object,
	.print_info = drm_gem_dma_object_print_info,
	.get_sg_table = drm_gem_dma_object_get_sg_table,
	.vmap = drm_gem_dma_object_vmap,
	.export = knpu_gem_export,
	.mmap = knpu_gem_mmap,
	.vm_ops = &drm_gem_dma_vm_ops,
};

struct drm_gem_object *knpu_gem_create_object(struct drm_device *ddev,
					      size_t size)
{
	struct knpu_gem_object *bo;

	bo = kzalloc_obj(*bo);
	if (!bo)
		return ERR_PTR(-ENOMEM);

	bo->base.base.funcs = &knpu_gem_funcs;
	return &bo->base.base;
}

int knpu_gem_create_with_handle(struct drm_file *file,
				struct drm_device *ddev,
				u64 *size, u32 *handle)
{
	struct drm_gem_dma_object *mem;
	int ret;

	mem = drm_gem_dma_create(ddev, *size);
	if (IS_ERR(mem))
		return PTR_ERR(mem);

	ret = drm_gem_handle_create(file, &mem->base, handle);
	if (!ret)
		*size = mem->base.size;

	/* The handle holds the reference now */
	drm_gem_object_put(&mem->base);
	return ret;
}

/* One side of a DMA: the SRAM or a region (spec 6.3) */
struct knpu_dma_side {
	bool sram;
	u8 region;
	u32 offset;
};

/* The SET state the validator tracks, with the defaults of spec 5 */
struct knpu_dma_state {
	struct knpu_dma_side src;
	struct knpu_dma_side dst;
	u32 len;
	u32 count;
	u32 stride;
};

static int knpu_set_space(struct knpu_dma_side *side, u16 param, u32 offset)
{
	u8 kind = param >> 8;
	u8 region = param & 0xff;

	switch (kind) {
	case KNPU_SPACE_REGION:
		if (region >= KNPU_NUM_REGIONS)
			return -EINVAL;
		side->sram = false;
		break;
	case KNPU_SPACE_SRAM:
		if (region)
			return -EINVAL;
		side->sram = true;
		break;
	default:
		return -EINVAL;
	}

	side->region = region;
	side->offset = offset;
	return 0;
}

/*
 * Record the bytes an OP_DMA touches. The region side starts at the region
 * offset and has COUNT rows of LEN bytes, STRIDE bytes apart; the SRAM side
 * is LEN * COUNT contiguous bytes.
 */
static int knpu_check_dma(struct drm_device *ddev,
			  struct knpu_cmdstream_info *info,
			  const struct knpu_dma_state *dma, u32 offset)
{
	struct knpu_device *kdev = to_knpu_device(ddev);
	const struct knpu_dma_side *sram, *region;
	u64 total, end;

	/* Exactly one side is the SRAM; the device refuses anything else */
	if (dma->src.sram == dma->dst.sram) {
		drm_dbg(ddev, "cmd 0x%x: DMA needs exactly one SRAM side\n",
			offset);
		return -EINVAL;
	}
	sram = dma->src.sram ? &dma->src : &dma->dst;
	region = dma->src.sram ? &dma->dst : &dma->src;

	if (check_mul_overflow((u64)dma->len, (u64)dma->count, &total) ||
	    check_add_overflow(total, (u64)sram->offset, &end) ||
	    end > kdev->npu_info.sram_size) {
		drm_dbg(ddev, "cmd 0x%x: SRAM range 0x%x+0x%llx is outside the SRAM\n",
			offset, sram->offset, total);
		return -EINVAL;
	}

	if (!dma->len || !dma->count)
		return 0;

	if (check_mul_overflow((u64)(dma->count - 1), (u64)dma->stride, &end) ||
	    check_add_overflow(end, (u64)region->offset, &end) ||
	    check_add_overflow(end, (u64)dma->len, &end)) {
		drm_dbg(ddev, "cmd 0x%x: region range overflows\n", offset);
		return -EINVAL;
	}

	info->region_size[region->region] =
		max(info->region_size[region->region], end);
	if (!dma->dst.sram)
		info->output_region[region->region] = true;

	drm_dbg(ddev, "cmd 0x%x: DMA %s region %u, 0x%x..0x%llx\n", offset,
		dma->dst.sram ? "from" : "to", region->region, region->offset,
		end);
	return 0;
}

/*
 * Validate a stream that has already been copied into the kernel, so user
 * space cannot change it while or after it is checked.
 */
static int knpu_validate(struct drm_device *ddev,
			 struct knpu_cmdstream_info *info,
			 const u32 *words, u32 nwords)
{
	struct knpu_dma_state dma = {
		.src = { .sram = true },
		.dst = { .sram = true },
		.count = 1,
	};
	bool stop = false;
	u32 i;
	int ret;

	for (i = 0; i < nwords; i++) {
		u32 offset = i * 4;
		u16 cmd = words[i] & KNPU_CMD_MASK;
		u16 param = KNPU_CMD_PARAM(words[i]);
		u32 payload = 0;

		if (cmd & KNPU_CMD_PAYLOAD) {
			if (i + 1 >= nwords) {
				drm_dbg(ddev, "cmd 0x%x: payload is missing\n",
					offset);
				return -EINVAL;
			}
			payload = words[++i];
		}

		switch (cmd) {
		case KNPU_OP_STOP:
			/* Words after OP_STOP would never run */
			if (i != nwords - 1) {
				drm_dbg(ddev, "cmd 0x%x: OP_STOP is not the last word\n",
					offset);
				return -EINVAL;
			}
			stop = true;
			break;
		case KNPU_OP_DMA:
			ret = knpu_check_dma(ddev, info, &dma, offset);
			if (ret)
				return ret;
			break;
		case KNPU_SET_DMA_SRC:
		case KNPU_SET_DMA_DST:
			ret = knpu_set_space(cmd == KNPU_SET_DMA_SRC ?
					     &dma.src : &dma.dst, param, payload);
			if (ret) {
				drm_dbg(ddev, "cmd 0x%x: bad DMA space 0x%x\n",
					offset, param);
				return ret;
			}
			break;
		case KNPU_SET_DMA_LEN:
			dma.len = payload;
			break;
		case KNPU_SET_DMA_COUNT:
			dma.count = param;
			break;
		case KNPU_SET_DMA_STRIDE:
			dma.stride = payload;
			break;
		/*
		 * Operations only touch the SRAM, which the device bounds
		 * itself (SRAM_RANGE), and missing waits are reported by the
		 * device (SYNC). Their fields are checked by the device too.
		 */
		case KNPU_OP_NOP:
		case KNPU_OP_CONV:
		case KNPU_OP_ADD:
		case KNPU_OP_AVGPOOL:
		case KNPU_OP_DMA_WAIT:
		case KNPU_OP_KERNEL_WAIT:
		case KNPU_SET_IFM_ZP:
		case KNPU_SET_IFM2_ZP:
		case KNPU_SET_OFM_ZP:
		case KNPU_SET_KERNEL:
		case KNPU_SET_STRIDE:
		case KNPU_SET_PAD:
		case KNPU_SET_CLAMP:
		case KNPU_SET_ROUND_MODE:
		case KNPU_SET_IFM_BASE:
		case KNPU_SET_IFM2_BASE:
		case KNPU_SET_OFM_BASE:
		case KNPU_SET_WEIGHT_BASE:
		case KNPU_SET_BIAS_BASE:
		case KNPU_SET_RQ_TABLE:
		case KNPU_SET_IFM_SHAPE:
		case KNPU_SET_OFM_SHAPE:
		case KNPU_SET_IFM_SCALE:
		case KNPU_SET_IFM2_SCALE:
		case KNPU_SET_OFM_SCALE:
			break;
		default:
			drm_dbg(ddev, "cmd 0x%x: unknown command 0x%04x\n",
				offset, cmd);
			return -EINVAL;
		}
	}

	if (!stop) {
		drm_dbg(ddev, "command stream does not end with OP_STOP\n");
		return -EINVAL;
	}

	return 0;
}

int knpu_gem_cmdstream_create(struct drm_file *file,
			      struct drm_device *ddev,
			      u32 size, u64 data, u32 *handle)
{
	struct knpu_cmdstream_info *info __free(kfree) = NULL;
	u32 *words __free(kvfree) = NULL;
	struct drm_gem_dma_object *mem;
	struct knpu_gem_object *bo;
	int ret;

	if (!size || size % 4)
		return -EINVAL;

	info = kzalloc_obj(*info);
	if (!info)
		return -ENOMEM;
	info->cmd_size = size;

	words = vmemdup_user(u64_to_user_ptr(data), size);
	if (IS_ERR(words))
		return PTR_ERR(no_free_ptr(words));

	ret = knpu_validate(ddev, info, words, size / 4);
	if (ret)
		return ret;

	mem = drm_gem_dma_create(ddev, size);
	if (IS_ERR(mem))
		return PTR_ERR(mem);

	memcpy(mem->vaddr, words, size);
	bo = to_knpu_bo(&mem->base);
	bo->info = no_free_ptr(info);

	ret = drm_gem_handle_create(file, &mem->base, handle);

	/* The handle holds the reference now */
	drm_gem_object_put(&mem->base);
	return ret;
}
