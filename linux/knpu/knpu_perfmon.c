// SPDX-License-Identifier: GPL-2.0-only OR MIT
/* Copyright (c) 2026 Kiwon Kwon */
/* Performance monitors, after ethosu_perfmon.c and v3d_perfmon.c */

#include <linux/slab.h>
#include <linux/uaccess.h>

#include <drm/drm_file.h>

#include "knpu_drv.h"

static void knpu_perfmon_get(struct knpu_perfmon *perfmon)
{
	if (perfmon)
		refcount_inc(&perfmon->refcnt);
}

void knpu_perfmon_put(struct knpu_perfmon *perfmon)
{
	if (perfmon && refcount_dec_and_test(&perfmon->refcnt))
		kfree(perfmon);
}

/* Called when a job submitted with @perfmon has finished */
void knpu_perfmon_add(struct knpu_perfmon *perfmon,
		      const u64 values[DRM_KNPU_PERF_COUNTERS])
{
	int i;

	spin_lock(&perfmon->lock);
	for (i = 0; i < DRM_KNPU_PERF_COUNTERS; i++)
		perfmon->values[i] += values[i];
	spin_unlock(&perfmon->lock);
}

struct knpu_perfmon *knpu_perfmon_find(struct knpu_file_priv *priv, u32 id)
{
	struct knpu_perfmon *perfmon;

	xa_lock(&priv->perfmons);
	perfmon = xa_load(&priv->perfmons, id);
	knpu_perfmon_get(perfmon);
	xa_unlock(&priv->perfmons);

	return perfmon;
}

void knpu_perfmon_open_file(struct knpu_file_priv *priv)
{
	xa_init_flags(&priv->perfmons, XA_FLAGS_ALLOC1);
}

void knpu_perfmon_close_file(struct knpu_file_priv *priv)
{
	struct knpu_perfmon *perfmon;
	unsigned long id;

	xa_for_each(&priv->perfmons, id, perfmon)
		knpu_perfmon_put(perfmon);

	xa_destroy(&priv->perfmons);
}

int knpu_ioctl_perfmon_create(struct drm_device *ddev, void *data,
			      struct drm_file *file)
{
	struct knpu_file_priv *priv = file->driver_priv;
	struct drm_knpu_perfmon_create *args = data;
	struct knpu_perfmon *perfmon;
	u32 id;
	int ret;

	if (args->pad)
		return -EINVAL;

	perfmon = kzalloc_obj(*perfmon);
	if (!perfmon)
		return -ENOMEM;

	refcount_set(&perfmon->refcnt, 1);
	spin_lock_init(&perfmon->lock);

	ret = xa_alloc(&priv->perfmons, &id, perfmon, xa_limit_32b,
		       GFP_KERNEL);
	if (ret < 0) {
		kfree(perfmon);
		return ret;
	}

	args->id = id;
	return 0;
}

int knpu_ioctl_perfmon_destroy(struct drm_device *ddev, void *data,
			       struct drm_file *file)
{
	struct knpu_file_priv *priv = file->driver_priv;
	struct drm_knpu_perfmon_destroy *args = data;
	struct knpu_perfmon *perfmon;

	if (args->pad)
		return -EINVAL;

	perfmon = xa_erase(&priv->perfmons, args->id);
	if (!perfmon)
		return -ENOENT;

	knpu_perfmon_put(perfmon);
	return 0;
}

int knpu_ioctl_perfmon_get_values(struct drm_device *ddev, void *data,
				  struct drm_file *file)
{
	struct knpu_file_priv *priv = file->driver_priv;
	struct drm_knpu_perfmon_get_values *args = data;
	u64 values[DRM_KNPU_PERF_COUNTERS];
	struct knpu_perfmon *perfmon;

	if (args->pad)
		return -EINVAL;

	perfmon = knpu_perfmon_find(priv, args->id);
	if (!perfmon)
		return -ENOENT;

	spin_lock(&perfmon->lock);
	memcpy(values, perfmon->values, sizeof(values));
	spin_unlock(&perfmon->lock);
	knpu_perfmon_put(perfmon);

	if (copy_to_user(u64_to_user_ptr(args->values_ptr), values,
			 sizeof(values)))
		return -EFAULT;

	return 0;
}
