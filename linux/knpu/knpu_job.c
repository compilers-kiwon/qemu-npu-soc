// SPDX-License-Identifier: GPL-2.0-only OR MIT
/* Copyright (c) 2026 Kiwon Kwon */
/* Jobs, interrupts and timeout recovery, after ethosu_job.c */

#include <linux/bitfield.h>
#include <linux/interrupt.h>
#include <linux/platform_device.h>
#include <linux/pm_runtime.h>
#include <linux/slab.h>

#include <drm/drm_file.h>
#include <drm/drm_gem.h>
#include <drm/drm_print.h>

#include "knpu_device.h"
#include "knpu_drv.h"
#include "knpu_gem.h"
#include "knpu_job.h"

#define KNPU_JOB_TIMEOUT_MS	500

static const char * const knpu_err_names[] = {
	[KNPU_ERR_NONE]		= "NONE",
	[KNPU_ERR_UNKNOWN_CMD]	= "UNKNOWN_CMD",
	[KNPU_ERR_TRUNCATED]	= "TRUNCATED",
	[KNPU_ERR_NO_STOP]	= "NO_STOP",
	[KNPU_ERR_BAD_QUEUE]	= "BAD_QUEUE",
	[KNPU_ERR_BAD_PARAM]	= "BAD_PARAM",
	[KNPU_ERR_SRAM_RANGE]	= "SRAM_RANGE",
	[KNPU_ERR_SYNC]		= "SYNC",
	[KNPU_ERR_BUS]		= "BUS",
};

static const char *knpu_err_name(u32 code)
{
	if (code < ARRAY_SIZE(knpu_err_names))
		return knpu_err_names[code];
	return "unknown";
}

static struct knpu_job *to_knpu_job(struct drm_sched_job *sched_job)
{
	return container_of(sched_job, struct knpu_job, base);
}

static const char *knpu_fence_get_driver_name(struct dma_fence *fence)
{
	return "knpu";
}

static const char *knpu_fence_get_timeline_name(struct dma_fence *fence)
{
	return "knpu-npu";
}

static const struct dma_fence_ops knpu_fence_ops = {
	.get_driver_name = knpu_fence_get_driver_name,
	.get_timeline_name = knpu_fence_get_timeline_name,
};

/* Program the device for @job and start it (docs/npu-spec.md 2) */
static void knpu_job_hw_submit(struct knpu_device *kdev, struct knpu_job *job)
{
	struct drm_gem_dma_object *cmd_bo = to_drm_gem_dma_obj(job->cmd_bo);
	struct knpu_cmdstream_info *info = to_knpu_bo(job->cmd_bo)->info;
	struct knpu_submit_regs regs = { .valid = true };
	int i;

	for (i = 0; i < job->region_cnt; i++) {
		dma_addr_t addr = to_drm_gem_dma_obj(job->region_bo[i])->dma_addr;
		u8 region = job->region_bo_num[i];

		writel_relaxed(lower_32_bits(addr),
			       kdev->regs + KNPU_REG_REGION_BASE_LO(region));
		writel_relaxed(upper_32_bits(addr),
			       kdev->regs + KNPU_REG_REGION_BASE_HI(region));
		regs.region_base[region] = addr;
	}

	writel_relaxed(lower_32_bits(cmd_bo->dma_addr),
		       kdev->regs + KNPU_REG_QBASE_LO);
	writel_relaxed(upper_32_bits(cmd_bo->dma_addr),
		       kdev->regs + KNPU_REG_QBASE_HI);
	writel_relaxed(info->cmd_size, kdev->regs + KNPU_REG_QSIZE);
	writel_relaxed(job->perfmon ? KNPU_PERF_ENABLE | KNPU_PERF_CLEAR : 0,
		       kdev->regs + KNPU_REG_PERF_CTRL);

	regs.qbase = cmd_bo->dma_addr;
	regs.qsize = info->cmd_size;
	spin_lock(&kdev->last_submit_lock);
	kdev->last_submit = regs;
	spin_unlock(&kdev->last_submit_lock);

	if (READ_ONCE(kdev->inject_hang)) {
		WRITE_ONCE(kdev->inject_hang, false);
		dev_warn(kdev->base.dev, "debugfs inject_hang: job not started\n");
		return;
	}

	/* writel orders the buffer and register writes before START */
	writel(KNPU_CTRL_START, kdev->regs + KNPU_REG_CTRL);
}

static int knpu_acquire_object_fences(struct knpu_job *job)
{
	struct knpu_cmdstream_info *info = to_knpu_bo(job->cmd_bo)->info;
	int i, ret;

	for (i = 0; i < job->region_cnt; i++) {
		struct drm_gem_object *bo = job->region_bo[i];
		bool write = info->output_region[job->region_bo_num[i]];

		ret = dma_resv_reserve_fences(bo->resv, 1);
		if (ret)
			return ret;

		ret = drm_sched_job_add_implicit_dependencies(&job->base, bo,
							      write);
		if (ret)
			return ret;
	}

	return 0;
}

/*
 * Every attached buffer gets the job's fence, so BO_WAIT on an input also
 * waits until the device has stopped reading it.
 */
static void knpu_attach_object_fences(struct knpu_job *job)
{
	struct knpu_cmdstream_info *info = to_knpu_bo(job->cmd_bo)->info;
	int i;

	for (i = 0; i < job->region_cnt; i++)
		dma_resv_add_fence(job->region_bo[i]->resv,
				   job->inference_done_fence,
				   info->output_region[job->region_bo_num[i]] ?
				   DMA_RESV_USAGE_WRITE : DMA_RESV_USAGE_READ);
}

static int knpu_job_push(struct knpu_job *job)
{
	struct knpu_device *kdev = job->kdev;
	struct ww_acquire_ctx acquire_ctx;
	int ret;

	ret = drm_gem_lock_reservations(job->region_bo, job->region_cnt,
					&acquire_ctx);
	if (ret)
		return ret;

	ret = knpu_acquire_object_fences(job);
	if (ret)
		goto out;

	/* Put by knpu_job_cleanup() */
	ret = pm_runtime_resume_and_get(kdev->base.dev);
	if (!ret) {
		guard(mutex)(&kdev->sched_lock);

		drm_sched_job_arm(&job->base);
		job->inference_done_fence =
			dma_fence_get(&job->base.s_fence->finished);
		kref_get(&job->refcount); /* put by the scheduler */
		drm_sched_entity_push_job(&job->base);
		knpu_attach_object_fences(job);
	}

out:
	drm_gem_unlock_reservations(job->region_bo, job->region_cnt,
				    &acquire_ctx);
	return ret;
}

static void knpu_job_err_cleanup(struct knpu_job *job)
{
	int i;

	knpu_perfmon_put(job->perfmon);

	for (i = 0; i < job->region_cnt; i++)
		drm_gem_object_put(job->region_bo[i]);

	drm_gem_object_put(job->cmd_bo);

	if (job->done_fence) {
		if (dma_fence_was_initialized(job->done_fence))
			dma_fence_put(job->done_fence);
		else
			dma_fence_free(job->done_fence);
	}

	kfree(job);
}

static void knpu_job_cleanup(struct kref *ref)
{
	struct knpu_job *job = container_of(ref, struct knpu_job, refcount);

	pm_runtime_put_autosuspend(job->kdev->base.dev);
	dma_fence_put(job->inference_done_fence);
	knpu_job_err_cleanup(job);
}

static void knpu_job_put(struct knpu_job *job)
{
	kref_put(&job->refcount, knpu_job_cleanup);
}

static void knpu_job_free(struct drm_sched_job *sched_job)
{
	struct knpu_job *job = to_knpu_job(sched_job);

	drm_sched_job_cleanup(sched_job);
	knpu_job_put(job);
}

static struct dma_fence *knpu_job_run(struct drm_sched_job *sched_job)
{
	struct knpu_job *job = to_knpu_job(sched_job);
	struct knpu_device *kdev = job->kdev;
	struct dma_fence *fence = job->done_fence;

	if (unlikely(job->base.s_fence->finished.error))
		return NULL;

	dma_fence_init(fence, &knpu_fence_ops, &kdev->fence_lock,
		       kdev->fence_context, ++kdev->emit_seqno);
	dma_fence_get(fence);

	WRITE_ONCE(kdev->in_flight_job, job);
	knpu_job_hw_submit(kdev, job);

	return fence;
}

static void knpu_read_counters(struct knpu_device *kdev,
			       u64 values[DRM_KNPU_PERF_COUNTERS])
{
	int i;

	for (i = 0; i < DRM_KNPU_PERF_COUNTERS; i++)
		values[i] = readl_relaxed(kdev->regs + KNPU_REG_PERF_LO(i)) |
			(u64)readl_relaxed(kdev->regs + KNPU_REG_PERF_HI(i)) << 32;
}

static irqreturn_t knpu_irq_thread(int irq, void *data)
{
	struct knpu_device *kdev = data;
	u32 status = readl_relaxed(kdev->regs + KNPU_REG_STATUS);
	struct knpu_job *job = READ_ONCE(kdev->in_flight_job);

	if (!job || !(status & (KNPU_STATUS_DONE | KNPU_STATUS_ERROR)))
		return IRQ_HANDLED;

	if (job->perfmon) {
		u64 values[DRM_KNPU_PERF_COUNTERS];

		knpu_read_counters(kdev, values);
		knpu_perfmon_add(job->perfmon, values);
	}

	if (status & KNPU_STATUS_ERROR) {
		u32 code = FIELD_GET(KNPU_STATUS_ERR_CODE_MASK, status);

		dev_err_ratelimited(kdev->base.dev,
				    "job failed: %s (%u) at command offset 0x%x\n",
				    knpu_err_name(code), code,
				    readl_relaxed(kdev->regs + KNPU_REG_ERR_OFFSET));
		dma_fence_set_error(job->done_fence, -EIO);
	}

	WRITE_ONCE(kdev->in_flight_job, NULL);
	dma_fence_signal(job->done_fence);

	return IRQ_HANDLED;
}

static irqreturn_t knpu_irq_handler(int irq, void *data)
{
	struct knpu_device *kdev = data;
	u32 pending = readl_relaxed(kdev->regs + KNPU_REG_IRQ_STATUS);

	if (!pending)
		return IRQ_NONE;

	/* IRQ_STATUS is write-1-to-clear; the level line drops with it */
	writel_relaxed(pending, kdev->regs + KNPU_REG_IRQ_STATUS);
	return IRQ_WAKE_THREAD;
}

static enum drm_gpu_sched_stat knpu_job_timedout(struct drm_sched_job *bad)
{
	struct knpu_job *job = to_knpu_job(bad);
	struct knpu_device *kdev = job->kdev;
	u32 status;

	/*
	 * QEMU runs a job in one go on its main loop, so a long job can make
	 * the guest clock jump past the timeout just as the job finishes.
	 * A finished job whose interrupt is pending or being handled is not
	 * a hang.
	 */
	synchronize_irq(kdev->irq);
	if (dma_fence_is_signaled(job->done_fence) ||
	    readl_relaxed(kdev->regs + KNPU_REG_IRQ_STATUS))
		return DRM_GPU_SCHED_STAT_NO_HANG;

	status = readl_relaxed(kdev->regs + KNPU_REG_STATUS);
	dev_err(kdev->base.dev, "job timed out (STATUS 0x%x), resetting the NPU\n",
		status);

	drm_sched_stop(&kdev->sched, bad);

	WRITE_ONCE(kdev->in_flight_job, NULL);
	synchronize_irq(kdev->irq);
	if (!dma_fence_is_signaled(job->done_fence)) {
		dma_fence_set_error(job->done_fence, -ETIMEDOUT);
		dma_fence_signal(job->done_fence);
	}

	/* Runtime resume soft-resets the device */
	pm_runtime_force_suspend(kdev->base.dev);
	pm_runtime_force_resume(kdev->base.dev);

	/* The bad job finishes with -ECANCELED */
	drm_sched_start(&kdev->sched, 0);

	return DRM_GPU_SCHED_STAT_RESET;
}

static const struct drm_sched_backend_ops knpu_sched_ops = {
	.run_job = knpu_job_run,
	.timedout_job = knpu_job_timedout,
	.free_job = knpu_job_free,
};

int knpu_job_init(struct knpu_device *kdev)
{
	struct device *dev = kdev->base.dev;
	struct drm_sched_init_args args = {
		.ops = &knpu_sched_ops,
		.num_rqs = DRM_SCHED_PRIORITY_COUNT,
		.credit_limit = 1,
		.timeout = msecs_to_jiffies(KNPU_JOB_TIMEOUT_MS),
		.name = dev_name(dev),
		.dev = dev,
	};
	int ret;

	spin_lock_init(&kdev->fence_lock);
	ret = devm_mutex_init(dev, &kdev->sched_lock);
	if (ret)
		return ret;

	kdev->irq = platform_get_irq(to_platform_device(dev), 0);
	if (kdev->irq < 0)
		return kdev->irq;

	ret = devm_request_threaded_irq(dev, kdev->irq, knpu_irq_handler,
					knpu_irq_thread, 0, KBUILD_MODNAME,
					kdev);
	if (ret)
		return dev_err_probe(dev, ret, "failed to request irq\n");

	kdev->fence_context = dma_fence_context_alloc(1);

	ret = drm_sched_init(&kdev->sched, &args);
	if (ret)
		return dev_err_probe(dev, ret, "failed to create scheduler\n");

	return 0;
}

void knpu_job_fini(struct knpu_device *kdev)
{
	drm_sched_fini(&kdev->sched);
}

int knpu_job_open(struct knpu_file_priv *priv)
{
	struct drm_gpu_scheduler *sched = &priv->kdev->sched;

	return drm_sched_entity_init(&priv->sched_entity,
				     DRM_SCHED_PRIORITY_NORMAL, &sched, 1,
				     NULL);
}

void knpu_job_close(struct knpu_file_priv *priv)
{
	drm_sched_entity_destroy(&priv->sched_entity);
}

int knpu_ioctl_submit(struct drm_device *ddev, void *data,
		      struct drm_file *file)
{
	struct knpu_device *kdev = to_knpu_device(ddev);
	struct knpu_file_priv *priv = file->driver_priv;
	struct drm_knpu_submit *args = data;
	struct knpu_cmdstream_info *info;
	struct knpu_job *job;
	int i, j, ret;

	job = kzalloc_obj(*job);
	if (!job)
		return -ENOMEM;

	kref_init(&job->refcount);
	job->kdev = kdev;

	if (args->perfmon_id) {
		job->perfmon = knpu_perfmon_find(priv, args->perfmon_id);
		if (!job->perfmon) {
			ret = -ENOENT;
			goto out_put_job;
		}
	}

	job->done_fence = kzalloc_obj(*job->done_fence);
	if (!job->done_fence) {
		ret = -ENOMEM;
		goto out_put_job;
	}

	ret = drm_sched_job_init(&job->base, &priv->sched_entity, 1, NULL,
				 file->client_id);
	if (ret)
		goto out_put_job;

	job->cmd_bo = drm_gem_object_lookup(file, args->cmdstream);
	if (!job->cmd_bo) {
		ret = -ENOENT;
		goto out_cleanup_job;
	}
	info = to_knpu_bo(job->cmd_bo)->info;
	if (!info) {
		drm_dbg(ddev, "handle %u is not a command stream\n",
			args->cmdstream);
		ret = -EINVAL;
		goto out_cleanup_job;
	}

	for (i = 0; i < KNPU_NUM_REGIONS; i++) {
		u32 handle = args->region_bo_handles[i];
		struct drm_gem_object *gem;

		if (!handle) {
			if (!info->region_size[i])
				continue;
			drm_dbg(ddev, "region %d is used but has no buffer\n", i);
			ret = -EINVAL;
			goto out_cleanup_job;
		}

		if (!info->region_size[i]) {
			drm_dbg(ddev, "region %d is not used by the stream\n", i);
			ret = -EINVAL;
			goto out_cleanup_job;
		}

		gem = drm_gem_object_lookup(file, handle);
		if (!gem) {
			ret = -ENOENT;
			goto out_cleanup_job;
		}
		job->region_bo[job->region_cnt] = gem;
		job->region_bo_num[job->region_cnt] = i;
		job->region_cnt++;

		if (to_knpu_bo(gem)->info) {
			drm_dbg(ddev, "region %d: a command stream is not a region buffer\n",
				i);
			ret = -EINVAL;
			goto out_cleanup_job;
		}

		for (j = 0; j < job->region_cnt - 1; j++) {
			if (job->region_bo[j] == gem) {
				drm_dbg(ddev, "region %d: buffer already attached\n",
					i);
				ret = -EINVAL;
				goto out_cleanup_job;
			}
		}

		/* The stream must not reach past the end of the buffer */
		if (info->region_size[i] > gem->size) {
			drm_dbg(ddev, "region %d needs 0x%llx bytes, buffer has 0x%zx\n",
				i, info->region_size[i], gem->size);
			ret = -EOVERFLOW;
			goto out_cleanup_job;
		}
	}

	ret = knpu_job_push(job);
	if (!ret) {
		knpu_job_put(job);
		return 0;
	}

out_cleanup_job:
	drm_sched_job_cleanup(&job->base);
out_put_job:
	knpu_job_err_cleanup(job);
	return ret;
}
