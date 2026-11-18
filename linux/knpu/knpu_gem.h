/* SPDX-License-Identifier: GPL-2.0-only OR MIT */
/* Copyright (c) 2026 Kiwon Kwon */

#ifndef __KNPU_GEM_H__
#define __KNPU_GEM_H__

#include <drm/drm_gem_dma_helper.h>

#include "knpu_cmd.h"

/* What the validator learned about a command stream (design 8.4) */
struct knpu_cmdstream_info {
	u32 cmd_size;
	/* Bytes of each region the stream touches, from offset 0 */
	u64 region_size[KNPU_NUM_REGIONS];
	/* Regions that a DMA writes to */
	bool output_region[KNPU_NUM_REGIONS];
};

struct knpu_gem_object {
	struct drm_gem_dma_object base;

	/* Set only for command stream buffers */
	struct knpu_cmdstream_info *info;
};

static inline struct knpu_gem_object *to_knpu_bo(struct drm_gem_object *obj)
{
	return container_of(to_drm_gem_dma_obj(obj), struct knpu_gem_object,
			    base);
}

struct drm_gem_object *knpu_gem_create_object(struct drm_device *ddev,
					      size_t size);
int knpu_gem_create_with_handle(struct drm_file *file,
				struct drm_device *ddev,
				u64 *size, u32 *handle);
int knpu_gem_cmdstream_create(struct drm_file *file,
			      struct drm_device *ddev,
			      u32 size, u64 data, u32 *handle);

#endif /* __KNPU_GEM_H__ */
