/* SPDX-License-Identifier: GPL-2.0-only OR MIT */
/* Copyright (c) 2026 Kiwon Kwon */

#ifndef __KNPU_DRV_H__
#define __KNPU_DRV_H__

#include <linux/clk.h>
#include <linux/mutex.h>
#include <linux/refcount.h>
#include <linux/spinlock.h>
#include <linux/xarray.h>

#include <drm/drm_device.h>
#include <drm/gpu_scheduler.h>
#include <drm/knpu_accel.h>

#include "knpu_cmd.h"

struct knpu_job;

/* What the last job wrote to the device, shown in debugfs (last_submit) */
struct knpu_submit_regs {
	bool valid;
	u64 qbase;
	u32 qsize;
	u64 region_base[KNPU_NUM_REGIONS];
};

struct knpu_device {
	struct drm_device base;

	void __iomem *regs;
	struct clk_bulk_data *clks;
	int num_clks;
	int irq;

	struct drm_knpu_npu_info npu_info;

	/* Job scheduling (knpu_job.c): one hardware queue */
	struct drm_gpu_scheduler sched;
	struct mutex sched_lock;	/* serializes arming and pushing jobs */
	spinlock_t fence_lock;
	u64 fence_context;
	u64 emit_seqno;
	struct knpu_job *in_flight_job;

	/* debugfs: the next job is not started, so it times out */
	bool inject_hang;

	spinlock_t last_submit_lock;	/* protects last_submit */
	struct knpu_submit_regs last_submit;
};

#define to_knpu_device(drm_dev) \
	container_of(drm_dev, struct knpu_device, base)

struct knpu_file_priv {
	struct knpu_device *kdev;
	struct drm_sched_entity sched_entity;
	struct xarray perfmons;
};

/*
 * A performance monitor adds up the device counters of the jobs submitted
 * with it. Jobs hold a reference until they are freed.
 */
struct knpu_perfmon {
	refcount_t refcnt;
	spinlock_t lock;		/* protects values */
	u64 values[DRM_KNPU_PERF_COUNTERS];
};

/* knpu_perfmon.c */
void knpu_perfmon_open_file(struct knpu_file_priv *priv);
void knpu_perfmon_close_file(struct knpu_file_priv *priv);
struct knpu_perfmon *knpu_perfmon_find(struct knpu_file_priv *priv, u32 id);
void knpu_perfmon_put(struct knpu_perfmon *perfmon);
void knpu_perfmon_add(struct knpu_perfmon *perfmon,
		      const u64 values[DRM_KNPU_PERF_COUNTERS]);
int knpu_ioctl_perfmon_create(struct drm_device *ddev, void *data,
			      struct drm_file *file);
int knpu_ioctl_perfmon_destroy(struct drm_device *ddev, void *data,
			       struct drm_file *file);
int knpu_ioctl_perfmon_get_values(struct drm_device *ddev, void *data,
				  struct drm_file *file);

#endif /* __KNPU_DRV_H__ */
