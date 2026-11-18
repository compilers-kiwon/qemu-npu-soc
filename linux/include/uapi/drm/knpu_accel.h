/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 Kiwon Kwon */
#ifndef _KNPU_ACCEL_H_
#define _KNPU_ACCEL_H_

#include "drm.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * DOC: kn1-npu accel UAPI
 *
 * The kn1-npu runs command streams made of SET, OP and wait commands
 * (docs/npu-spec.md). Operations only touch the NPU's internal SRAM; DMA
 * commands move data between that SRAM and up to eight regions, each of
 * which is a buffer object attached at submit time.
 *
 * A command stream is copied into a kernel-owned buffer and validated by
 * KNPU_CMDSTREAM_CREATE, so user space cannot change it afterwards. The
 * validator records how many bytes of each region the stream touches, and
 * KNPU_SUBMIT checks the attached buffers against those sizes.
 */

/**
 * enum drm_knpu_ioctl_id - IOCTL IDs
 *
 * Place new ioctls at the end, don't re-order, don't replace or remove
 * entries. Use the DRM_IOCTL_KNPU_xxx definitions instead of these IDs.
 */
enum drm_knpu_ioctl_id {
	/** @DRM_KNPU_QUERY: Query device information. */
	DRM_KNPU_QUERY = 0,

	/** @DRM_KNPU_BO_CREATE: Create a buffer object. */
	DRM_KNPU_BO_CREATE,

	/** @DRM_KNPU_BO_MMAP_OFFSET: Get the fake offset to mmap a buffer. */
	DRM_KNPU_BO_MMAP_OFFSET,

	/** @DRM_KNPU_BO_WAIT: Wait for the jobs using a buffer object. */
	DRM_KNPU_BO_WAIT,

	/** @DRM_KNPU_CMDSTREAM_CREATE: Copy and validate a command stream. */
	DRM_KNPU_CMDSTREAM_CREATE,

	/** @DRM_KNPU_SUBMIT: Run a command stream with its regions. */
	DRM_KNPU_SUBMIT,

	/** @DRM_KNPU_PERFMON_CREATE: Create a performance monitor. */
	DRM_KNPU_PERFMON_CREATE,

	/** @DRM_KNPU_PERFMON_DESTROY: Destroy a performance monitor. */
	DRM_KNPU_PERFMON_DESTROY,

	/** @DRM_KNPU_PERFMON_GET_VALUES: Read a performance monitor. */
	DRM_KNPU_PERFMON_GET_VALUES,
};

/**
 * enum drm_knpu_query_type - Query type
 *
 * Place new types at the end, don't re-order, don't remove or replace.
 */
enum drm_knpu_query_type {
	/** @DRM_KNPU_QUERY_NPU_INFO: Query struct drm_knpu_npu_info. */
	DRM_KNPU_QUERY_NPU_INFO = 0,
};

/**
 * struct drm_knpu_npu_info - NPU information
 *
 * The first three fields are the ID, VERSION and CONFIG registers.
 */
struct drm_knpu_npu_info {
	/** @id: ID register, "KNPU" (0x4B4E5055). */
	__u32 id;

	/** @version: VERSION register. */
	__u32 version;
#define DRM_KNPU_VERSION_MAJOR(x)		((x) >> 16)
#define DRM_KNPU_VERSION_MINOR(x)		((x) & 0xffff)

	/** @config: CONFIG register. */
	__u32 config;
#define DRM_KNPU_CONFIG_NUM_REGIONS(x)		((x) & 0xf)
#define DRM_KNPU_CONFIG_ROUND_DOUBLE(x)		(((x) >> 8) & 0x1)
#define DRM_KNPU_CONFIG_ROUND_SINGLE(x)		(((x) >> 9) & 0x1)
#define DRM_KNPU_CONFIG_DMA_DEPTH(x)		(((x) >> 16) & 0xf)
#define DRM_KNPU_CONFIG_KERNEL_DEPTH(x)		(((x) >> 20) & 0xf)

	/** @sram_size: Size of the NPU's internal SRAM in bytes. */
	__u32 sram_size;
};

/**
 * struct drm_knpu_query - Arguments passed to DRM_IOCTL_KNPU_QUERY
 */
struct drm_knpu_query {
	/** @type: The query type (enum drm_knpu_query_type). */
	__u32 type;

	/**
	 * @size: Size of the queried structure.
	 *
	 * If @pointer is 0, the driver sets @size to the size of the structure
	 * it knows. Otherwise the driver copies min(@size, that size) bytes to
	 * @pointer and zero-fills the rest of @size.
	 */
	__u32 size;

	/** @pointer: User pointer to the structure, or 0. */
	__u64 pointer;
};

/**
 * struct drm_knpu_bo_create - Arguments passed to DRM_IOCTL_KNPU_BO_CREATE
 *
 * Buffers are physically contiguous; the SoC has no IOMMU.
 */
struct drm_knpu_bo_create {
	/**
	 * @size: Requested size in bytes. The page-aligned allocated size is
	 * returned here.
	 */
	__u64 size;

	/** @flags: Must be 0. */
	__u32 flags;

	/** @handle: Returned handle for the object. Handles are nonzero. */
	__u32 handle;
};

/**
 * struct drm_knpu_bo_mmap_offset - Arguments passed to
 * DRM_IOCTL_KNPU_BO_MMAP_OFFSET
 *
 * Command stream buffers cannot be mapped.
 */
struct drm_knpu_bo_mmap_offset {
	/** @handle: Handle of the object to map. */
	__u32 handle;

	/** @pad: Must be 0. */
	__u32 pad;

	/** @offset: The fake offset to pass to mmap(). */
	__u64 offset;
};

/**
 * struct drm_knpu_bo_wait - Arguments passed to DRM_IOCTL_KNPU_BO_WAIT
 *
 * Waits until every submitted job that reads or writes the buffer has
 * finished. Returns 0 on success, -ETIME when @timeout_ns passed first, and
 * -EIO when the last job that used the buffer failed: the device stopped
 * with an error or the job timed out. The kernel log holds the error code
 * and the offset of the failing command.
 */
struct drm_knpu_bo_wait {
	/** @handle: Handle of the object to wait for. */
	__u32 handle;

	/** @pad: Must be 0. */
	__u32 pad;

	/** @timeout_ns: Absolute CLOCK_MONOTONIC time in nanoseconds. */
	__s64 timeout_ns;
};

/**
 * struct drm_knpu_cmdstream_create - Arguments passed to
 * DRM_IOCTL_KNPU_CMDSTREAM_CREATE
 *
 * Returns -EINVAL when the validator rejects the stream.
 */
struct drm_knpu_cmdstream_create {
	/** @size: Size of the stream in bytes, a nonzero multiple of 4. */
	__u32 size;

	/** @flags: Must be 0. */
	__u32 flags;

	/** @data: User pointer to the command words. */
	__u64 data;

	/** @handle: Returned handle for the command stream buffer. */
	__u32 handle;

	/** @pad: Must be 0. */
	__u32 pad;
};

#define DRM_KNPU_MAX_REGIONS	8

/**
 * struct drm_knpu_submit - Arguments passed to DRM_IOCTL_KNPU_SUBMIT
 *
 * Every region the command stream uses needs a buffer at least as large
 * as the validator measured (-EOVERFLOW otherwise); regions it does not use
 * must be left 0. A buffer can be attached to one region only, and a command
 * stream buffer cannot be a region buffer.
 */
struct drm_knpu_submit {
	/** @cmdstream: Handle returned by DRM_IOCTL_KNPU_CMDSTREAM_CREATE. */
	__u32 cmdstream;

	/** @perfmon_id: Performance monitor to count into, or 0. */
	__u32 perfmon_id;

	/** @region_bo_handles: Buffer handle for each region, or 0. */
	__u32 region_bo_handles[DRM_KNPU_MAX_REGIONS];
};

/* Counters of a performance monitor, in the order GET_VALUES returns them */
#define DRM_KNPU_PERF_MACS		0
#define DRM_KNPU_PERF_DMA_RD_BYTES	1
#define DRM_KNPU_PERF_DMA_WR_BYTES	2
#define DRM_KNPU_PERF_OPS		3
#define DRM_KNPU_PERF_COUNTERS		4

/**
 * struct drm_knpu_perfmon_create - Arguments passed to
 * DRM_IOCTL_KNPU_PERFMON_CREATE
 *
 * A performance monitor adds up the device counters of every job submitted
 * with it. The counters are fixed (see docs/npu-spec.md 11).
 */
struct drm_knpu_perfmon_create {
	/** @id: Returned monitor ID. IDs are nonzero. */
	__u32 id;

	/** @pad: Must be 0. */
	__u32 pad;
};

/**
 * struct drm_knpu_perfmon_destroy - Arguments passed to
 * DRM_IOCTL_KNPU_PERFMON_DESTROY
 */
struct drm_knpu_perfmon_destroy {
	/** @id: Monitor to destroy. */
	__u32 id;

	/** @pad: Must be 0. */
	__u32 pad;
};

/**
 * struct drm_knpu_perfmon_get_values - Arguments passed to
 * DRM_IOCTL_KNPU_PERFMON_GET_VALUES
 *
 * Copies DRM_KNPU_PERF_COUNTERS __u64 values to @values_ptr. No implicit
 * synchronization is done: wait for the jobs first.
 */
struct drm_knpu_perfmon_get_values {
	/** @id: Monitor to read. */
	__u32 id;

	/** @pad: Must be 0. */
	__u32 pad;

	/** @values_ptr: User pointer to DRM_KNPU_PERF_COUNTERS __u64. */
	__u64 values_ptr;
};

/**
 * DRM_IOCTL_KNPU() - Build a knpu IOCTL number
 * @__access: Access type. Must be R, W or RW.
 * @__id: One of the DRM_KNPU_xxx id.
 * @__type: Suffix of the type being passed to the IOCTL.
 *
 * Don't use this macro directly, use the DRM_IOCTL_KNPU_xxx values instead.
 */
#define DRM_IOCTL_KNPU(__access, __id, __type) \
	DRM_IO ## __access(DRM_COMMAND_BASE + DRM_KNPU_ ## __id, \
			   struct drm_knpu_ ## __type)

enum {
	DRM_IOCTL_KNPU_QUERY =
		DRM_IOCTL_KNPU(WR, QUERY, query),
	DRM_IOCTL_KNPU_BO_CREATE =
		DRM_IOCTL_KNPU(WR, BO_CREATE, bo_create),
	DRM_IOCTL_KNPU_BO_MMAP_OFFSET =
		DRM_IOCTL_KNPU(WR, BO_MMAP_OFFSET, bo_mmap_offset),
	DRM_IOCTL_KNPU_BO_WAIT =
		DRM_IOCTL_KNPU(WR, BO_WAIT, bo_wait),
	DRM_IOCTL_KNPU_CMDSTREAM_CREATE =
		DRM_IOCTL_KNPU(WR, CMDSTREAM_CREATE, cmdstream_create),
	DRM_IOCTL_KNPU_SUBMIT =
		DRM_IOCTL_KNPU(WR, SUBMIT, submit),
	DRM_IOCTL_KNPU_PERFMON_CREATE =
		DRM_IOCTL_KNPU(WR, PERFMON_CREATE, perfmon_create),
	DRM_IOCTL_KNPU_PERFMON_DESTROY =
		DRM_IOCTL_KNPU(WR, PERFMON_DESTROY, perfmon_destroy),
	DRM_IOCTL_KNPU_PERFMON_GET_VALUES =
		DRM_IOCTL_KNPU(WR, PERFMON_GET_VALUES, perfmon_get_values),
};

#if defined(__cplusplus)
}
#endif

#endif /* _KNPU_ACCEL_H_ */
