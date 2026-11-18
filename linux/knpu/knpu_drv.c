// SPDX-License-Identifier: GPL-2.0-only OR MIT
/* Copyright (c) 2026 Kiwon Kwon */
/* DRM accel driver for the kn1-npu, after drivers/accel/ethosu */

#include <linux/bitfield.h>
#include <linux/debugfs.h>
#include <linux/dma-mapping.h>
#include <linux/dma-resv.h>
#include <linux/io.h>
#include <linux/module.h>
#include <linux/of_reserved_mem.h>
#include <linux/platform_device.h>
#include <linux/pm_runtime.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include <drm/drm_accel.h>
#include <drm/drm_drv.h>
#include <drm/drm_file.h>
#include <drm/drm_gem.h>
#include <drm/drm_ioctl.h>
#include <drm/drm_utils.h>

#include "knpu_device.h"
#include "knpu_drv.h"
#include "knpu_gem.h"
#include "knpu_job.h"

static int knpu_ioctl_query(struct drm_device *ddev, void *data,
			    struct drm_file *file)
{
	struct knpu_device *kdev = to_knpu_device(ddev);
	struct drm_knpu_query *args = data;

	if (args->type != DRM_KNPU_QUERY_NPU_INFO)
		return -EINVAL;

	if (!args->pointer) {
		args->size = sizeof(kdev->npu_info);
		return 0;
	}

	if (args->size < offsetofend(struct drm_knpu_npu_info, sram_size))
		return -EINVAL;

	return copy_struct_to_user(u64_to_user_ptr(args->pointer), args->size,
				   &kdev->npu_info, sizeof(kdev->npu_info),
				   NULL);
}

static int knpu_ioctl_bo_create(struct drm_device *ddev, void *data,
				struct drm_file *file)
{
	struct drm_knpu_bo_create *args = data;
	int cookie, ret;

	if (!args->size || args->flags)
		return -EINVAL;

	if (!drm_dev_enter(ddev, &cookie))
		return -ENODEV;

	ret = knpu_gem_create_with_handle(file, ddev, &args->size,
					  &args->handle);

	drm_dev_exit(cookie);
	return ret;
}

static int knpu_ioctl_bo_mmap_offset(struct drm_device *ddev, void *data,
				     struct drm_file *file)
{
	struct drm_knpu_bo_mmap_offset *args = data;
	struct drm_gem_object *obj;

	if (args->pad)
		return -EINVAL;

	obj = drm_gem_object_lookup(file, args->handle);
	if (!obj)
		return -ENOENT;

	args->offset = drm_vma_node_offset_addr(&obj->vma_node);
	drm_gem_object_put(obj);
	return 0;
}

/*
 * -EIO if the newest job of some submitter that used @obj failed. Signaled
 * fences stay in the reservation object until a newer one replaces them,
 * so older fences of the same context are skipped. The unlocked iterator
 * would skip every signaled fence, hence the lock.
 */
static int knpu_bo_status(struct drm_gem_object *obj)
{
	struct dma_resv_iter cursor, other;
	struct dma_fence *fence, *f;
	int ret;

	ret = dma_resv_lock_interruptible(obj->resv, NULL);
	if (ret)
		return ret;

	dma_resv_for_each_fence(&cursor, obj->resv, DMA_RESV_USAGE_READ, fence) {
		bool newest = true;

		dma_resv_for_each_fence(&other, obj->resv, DMA_RESV_USAGE_READ, f) {
			if (f != fence && f->context == fence->context &&
			    dma_fence_is_later(f, fence)) {
				newest = false;
				break;
			}
		}

		if (newest && dma_fence_get_status(fence) < 0) {
			ret = -EIO;
			break;
		}
	}

	dma_resv_unlock(obj->resv);
	return ret;
}

/* Like drm_gem_dma_resv_wait(), but a failed job makes the wait fail */
static int knpu_ioctl_bo_wait(struct drm_device *ddev, void *data,
			      struct drm_file *file)
{
	struct drm_knpu_bo_wait *args = data;
	unsigned long timeout = drm_timeout_abs_to_jiffies(args->timeout_ns);
	struct drm_gem_object *obj;
	int cookie;
	long ret;

	if (args->pad)
		return -EINVAL;

	if (!drm_dev_enter(ddev, &cookie))
		return -ENODEV;

	obj = drm_gem_object_lookup(file, args->handle);
	if (!obj) {
		ret = -ENOENT;
		goto out_dev_exit;
	}

	ret = dma_resv_wait_timeout(obj->resv, DMA_RESV_USAGE_READ, true,
				    timeout);
	if (ret == 0)
		ret = -ETIME;
	else if (ret > 0)
		ret = knpu_bo_status(obj);

	drm_gem_object_put(obj);
out_dev_exit:
	drm_dev_exit(cookie);
	return ret;
}

static int knpu_ioctl_cmdstream_create(struct drm_device *ddev, void *data,
				       struct drm_file *file)
{
	struct drm_knpu_cmdstream_create *args = data;
	int cookie, ret;

	if (!args->data || args->flags || args->pad)
		return -EINVAL;

	if (!drm_dev_enter(ddev, &cookie))
		return -ENODEV;

	ret = knpu_gem_cmdstream_create(file, ddev, args->size, args->data,
					&args->handle);

	drm_dev_exit(cookie);
	return ret;
}

static int knpu_open(struct drm_device *ddev, struct drm_file *file)
{
	struct knpu_file_priv *priv;
	int ret;

	if (!try_module_get(THIS_MODULE))
		return -EINVAL;

	priv = kzalloc_obj(*priv);
	if (!priv) {
		ret = -ENOMEM;
		goto err_put_mod;
	}
	priv->kdev = to_knpu_device(ddev);

	ret = knpu_job_open(priv);
	if (ret)
		goto err_free;

	knpu_perfmon_open_file(priv);
	file->driver_priv = priv;
	return 0;

err_free:
	kfree(priv);
err_put_mod:
	module_put(THIS_MODULE);
	return ret;
}

static void knpu_postclose(struct drm_device *ddev, struct drm_file *file)
{
	struct knpu_file_priv *priv = file->driver_priv;

	knpu_job_close(priv);
	knpu_perfmon_close_file(priv);
	kfree(priv);
	module_put(THIS_MODULE);
}

static const struct drm_ioctl_desc knpu_drm_driver_ioctls[] = {
#define KNPU_IOCTL(n, func) \
	DRM_IOCTL_DEF_DRV(KNPU_##n, knpu_ioctl_##func, 0)

	KNPU_IOCTL(QUERY, query),
	KNPU_IOCTL(BO_CREATE, bo_create),
	KNPU_IOCTL(BO_MMAP_OFFSET, bo_mmap_offset),
	KNPU_IOCTL(BO_WAIT, bo_wait),
	KNPU_IOCTL(CMDSTREAM_CREATE, cmdstream_create),
	KNPU_IOCTL(SUBMIT, submit),
	KNPU_IOCTL(PERFMON_CREATE, perfmon_create),
	KNPU_IOCTL(PERFMON_DESTROY, perfmon_destroy),
	KNPU_IOCTL(PERFMON_GET_VALUES, perfmon_get_values),
};

DEFINE_DRM_ACCEL_FOPS(knpu_drm_driver_fops);

/*
 * knpu driver version:
 * - 1.0 - initial interface
 */
static const struct drm_driver knpu_drm_driver = {
	.driver_features = DRIVER_COMPUTE_ACCEL | DRIVER_GEM,
	.open = knpu_open,
	.postclose = knpu_postclose,
	.ioctls = knpu_drm_driver_ioctls,
	.num_ioctls = ARRAY_SIZE(knpu_drm_driver_ioctls),
	.fops = &knpu_drm_driver_fops,
	.name = "knpu",
	.desc = "Kiwon kn1-npu accel driver",
	.major = 1,
	.minor = 0,

	.gem_create_object = knpu_gem_create_object,
};

/* SOFT_RESET returns every register except SRAM_SIZE to its reset value */
static int knpu_hw_reset(struct knpu_device *kdev)
{
	u32 status;

	writel(KNPU_CTRL_SOFT_RESET, kdev->regs + KNPU_REG_CTRL);
	status = readl(kdev->regs + KNPU_REG_STATUS);
	if (status) {
		dev_err(kdev->base.dev, "STATUS is 0x%x after soft reset\n",
			status);
		return -EIO;
	}

	writel(KNPU_IRQ_DONE | KNPU_IRQ_ERROR, kdev->regs + KNPU_REG_IRQ_ENABLE);
	return 0;
}

static int knpu_device_resume(struct device *dev)
{
	struct knpu_device *kdev = dev_get_drvdata(dev);
	int ret;

	ret = clk_bulk_prepare_enable(kdev->num_clks, kdev->clks);
	if (ret)
		return ret;

	ret = knpu_hw_reset(kdev);
	if (ret)
		clk_bulk_disable_unprepare(kdev->num_clks, kdev->clks);

	return ret;
}

static int knpu_device_suspend(struct device *dev)
{
	struct knpu_device *kdev = dev_get_drvdata(dev);

	clk_bulk_disable_unprepare(kdev->num_clks, kdev->clks);
	return 0;
}

/* Power the device up, identify it and leave runtime PM holding it on */
static int knpu_init(struct knpu_device *kdev)
{
	struct device *dev = kdev->base.dev;
	struct drm_knpu_npu_info *info = &kdev->npu_info;
	int ret;

	ret = knpu_device_resume(dev);
	if (ret)
		return ret;

	info->id = readl(kdev->regs + KNPU_REG_ID);
	info->version = readl(kdev->regs + KNPU_REG_VERSION);
	info->config = readl(kdev->regs + KNPU_REG_CONFIG);
	info->sram_size = readl(kdev->regs + KNPU_REG_SRAM_SIZE);

	/* A different MAJOR is a different programming model (spec 3.2) */
	if (info->id != KNPU_ID ||
	    FIELD_GET(KNPU_VERSION_MAJOR_MASK, info->version) !=
	    KNPU_VERSION_MAJOR ||
	    FIELD_GET(KNPU_CONFIG_NUM_REGIONS_MASK, info->config) !=
	    KNPU_NUM_REGIONS) {
		dev_err(dev, "unsupported NPU: ID 0x%08x VERSION 0x%08x CONFIG 0x%08x\n",
			info->id, info->version, info->config);
		ret = -ENODEV;
		goto err_suspend;
	}

	pm_runtime_set_autosuspend_delay(dev, 50);
	pm_runtime_use_autosuspend(dev);
	ret = devm_pm_runtime_set_active_enabled(dev);
	if (ret)
		goto err_suspend;
	pm_runtime_get_noresume(dev);

	dev_info(dev, "kn1-npu v%lu.%lu, %u KiB SRAM, %lu regions, rounding:%s%s\n",
		 FIELD_GET(KNPU_VERSION_MAJOR_MASK, info->version),
		 FIELD_GET(KNPU_VERSION_MINOR_MASK, info->version),
		 info->sram_size / 1024,
		 FIELD_GET(KNPU_CONFIG_NUM_REGIONS_MASK, info->config),
		 info->config & KNPU_CONFIG_ROUND_DOUBLE ? " double" : "",
		 info->config & KNPU_CONFIG_ROUND_SINGLE ? " single" : "");
	return 0;

err_suspend:
	knpu_device_suspend(dev);
	return ret;
}

/* debugfs: what the last job wrote to the device */
static int knpu_last_submit_show(struct seq_file *m, void *unused)
{
	struct knpu_device *kdev = m->private;
	struct knpu_submit_regs regs;
	int i;

	spin_lock(&kdev->last_submit_lock);
	regs = kdev->last_submit;
	spin_unlock(&kdev->last_submit_lock);

	if (!regs.valid)
		return 0;

	seq_printf(m, "qbase 0x%016llx\n", regs.qbase);
	seq_printf(m, "qsize 0x%x\n", regs.qsize);
	for (i = 0; i < KNPU_NUM_REGIONS; i++)
		seq_printf(m, "region%d 0x%016llx\n", i, regs.region_base[i]);

	return 0;
}
DEFINE_SHOW_ATTRIBUTE(knpu_last_submit);

static void knpu_debugfs_init(struct knpu_device *kdev)
{
	struct dentry *root = kdev->base.debugfs_root;

	/* Fault injection for the timeout test: 1 skips the next START */
	debugfs_create_bool("inject_hang", 0600, root, &kdev->inject_hang);
	debugfs_create_file("last_submit", 0400, root, kdev,
			    &knpu_last_submit_fops);
}

static int knpu_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct knpu_device *kdev;
	int ret;

	kdev = devm_drm_dev_alloc(dev, &knpu_drm_driver, struct knpu_device,
				  base);
	if (IS_ERR(kdev))
		return PTR_ERR(kdev);
	platform_set_drvdata(pdev, kdev);
	spin_lock_init(&kdev->last_submit_lock);

	/* The address registers are 64 bits and the DRAM goes past 4 GiB */
	ret = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(64));
	if (ret)
		return ret;

	/* Buffers come from the board's NPU pool when it has one */
	ret = devm_of_reserved_mem_device_init(dev);
	if (ret && ret != -ENODEV)
		return dev_err_probe(dev, ret, "cannot use memory-region\n");

	kdev->regs = devm_platform_ioremap_resource(pdev, 0);
	if (IS_ERR(kdev->regs))
		return PTR_ERR(kdev->regs);

	kdev->num_clks = devm_clk_bulk_get_all(dev, &kdev->clks);
	if (kdev->num_clks < 0)
		return dev_err_probe(dev, kdev->num_clks, "cannot get clocks\n");

	ret = knpu_job_init(kdev);
	if (ret)
		return ret;

	ret = knpu_init(kdev);
	if (ret)
		goto err_job_fini;

	ret = drm_dev_register(&kdev->base, 0);
	if (ret) {
		pm_runtime_put_sync_suspend(dev);
		goto err_job_fini;
	}

	knpu_debugfs_init(kdev);
	pm_runtime_put_autosuspend(dev);
	return 0;

err_job_fini:
	knpu_job_fini(kdev);
	return ret;
}

static void knpu_remove(struct platform_device *pdev)
{
	struct knpu_device *kdev = platform_get_drvdata(pdev);

	drm_dev_unregister(&kdev->base);
	knpu_job_fini(kdev);
}

static const struct of_device_id knpu_of_match[] = {
	{ .compatible = "kiwon,kn1-npu" },
	{}
};
MODULE_DEVICE_TABLE(of, knpu_of_match);

static DEFINE_RUNTIME_DEV_PM_OPS(knpu_pm_ops, knpu_device_suspend,
				 knpu_device_resume, NULL);

static struct platform_driver knpu_driver = {
	.probe = knpu_probe,
	.remove = knpu_remove,
	.driver = {
		.name = "knpu",
		.pm = pm_ptr(&knpu_pm_ops),
		.of_match_table = knpu_of_match,
	},
};
module_platform_driver(knpu_driver);

MODULE_AUTHOR("Kiwon Kwon");
MODULE_DESCRIPTION("Kiwon kn1-npu accel driver");
MODULE_LICENSE("Dual MIT/GPL");
