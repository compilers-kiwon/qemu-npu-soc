/* SPDX-License-Identifier: GPL-2.0-only OR MIT */
/* Copyright (c) 2026 Kiwon Kwon */

#ifndef __KNPU_JOB_H__
#define __KNPU_JOB_H__

#include <linux/kref.h>

#include <drm/gpu_scheduler.h>

#include "knpu_cmd.h"

struct knpu_device;
struct knpu_file_priv;
struct knpu_perfmon;

struct knpu_job {
	struct drm_sched_job base;
	struct knpu_device *kdev;

	struct drm_gem_object *cmd_bo;
	struct drm_gem_object *region_bo[KNPU_NUM_REGIONS];
	u8 region_bo_num[KNPU_NUM_REGIONS];
	u8 region_cnt;

	struct knpu_perfmon *perfmon;

	/* Signaled by the scheduler once the job is done */
	struct dma_fence *inference_done_fence;

	/* Signaled by the interrupt handler when the device finishes */
	struct dma_fence *done_fence;

	struct kref refcount;
};

int knpu_ioctl_submit(struct drm_device *ddev, void *data,
		      struct drm_file *file);

int knpu_job_init(struct knpu_device *kdev);
void knpu_job_fini(struct knpu_device *kdev);
int knpu_job_open(struct knpu_file_priv *priv);
void knpu_job_close(struct knpu_file_priv *priv);

#endif /* __KNPU_JOB_H__ */
