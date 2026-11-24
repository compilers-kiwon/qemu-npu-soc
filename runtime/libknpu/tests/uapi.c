/*
 * Prints the sizes, field offsets and constants of the knpu UAPI and of the
 * C library definitions the runtime declares itself, for tests/uapi.rs.
 *
 * SPDX-FileCopyrightText: 2026 Kiwon Kwon
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/* memfd_create's flags, dladdr1 and Dl_info are GNU extensions */
#define _GNU_SOURCE

#include <dlfcn.h>
#include <elf.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>

#include "knpu_accel.h"

#define SIZE(t) printf("size %s %zu\n", #t, sizeof(struct t))
#define OFFSET(t, f) printf("offset %s.%s %zu\n", #t, #f, offsetof(struct t, f))
/* The same for a typedef name */
#define TSIZE(t) printf("size %s %zu\n", #t, sizeof(t))
#define TOFFSET(t, f) printf("offset %s.%s %zu\n", #t, #f, offsetof(t, f))
#define VALUE(n) printf("value %s %llu\n", #n, (unsigned long long)(n))

int main(void)
{
	SIZE(drm_gem_close);
	OFFSET(drm_gem_close, pad);

	SIZE(drm_knpu_npu_info);
	OFFSET(drm_knpu_npu_info, version);
	OFFSET(drm_knpu_npu_info, config);
	OFFSET(drm_knpu_npu_info, sram_size);
	SIZE(drm_knpu_query);
	OFFSET(drm_knpu_query, size);
	OFFSET(drm_knpu_query, pointer);
	SIZE(drm_knpu_bo_create);
	OFFSET(drm_knpu_bo_create, flags);
	OFFSET(drm_knpu_bo_create, handle);
	SIZE(drm_knpu_bo_mmap_offset);
	OFFSET(drm_knpu_bo_mmap_offset, pad);
	OFFSET(drm_knpu_bo_mmap_offset, offset);
	SIZE(drm_knpu_bo_wait);
	OFFSET(drm_knpu_bo_wait, pad);
	OFFSET(drm_knpu_bo_wait, timeout_ns);
	SIZE(drm_knpu_cmdstream_create);
	OFFSET(drm_knpu_cmdstream_create, flags);
	OFFSET(drm_knpu_cmdstream_create, data);
	OFFSET(drm_knpu_cmdstream_create, handle);
	OFFSET(drm_knpu_cmdstream_create, pad);
	SIZE(drm_knpu_submit);
	OFFSET(drm_knpu_submit, perfmon_id);
	OFFSET(drm_knpu_submit, region_bo_handles);
	SIZE(drm_knpu_perfmon_create);
	OFFSET(drm_knpu_perfmon_create, pad);
	SIZE(drm_knpu_perfmon_destroy);
	OFFSET(drm_knpu_perfmon_destroy, pad);
	SIZE(drm_knpu_perfmon_get_values);
	OFFSET(drm_knpu_perfmon_get_values, pad);
	OFFSET(drm_knpu_perfmon_get_values, values_ptr);

	VALUE(DRM_IOCTL_GEM_CLOSE);
	VALUE(DRM_IOCTL_KNPU_QUERY);
	VALUE(DRM_IOCTL_KNPU_BO_CREATE);
	VALUE(DRM_IOCTL_KNPU_BO_MMAP_OFFSET);
	VALUE(DRM_IOCTL_KNPU_BO_WAIT);
	VALUE(DRM_IOCTL_KNPU_CMDSTREAM_CREATE);
	VALUE(DRM_IOCTL_KNPU_SUBMIT);
	VALUE(DRM_IOCTL_KNPU_PERFMON_CREATE);
	VALUE(DRM_IOCTL_KNPU_PERFMON_DESTROY);
	VALUE(DRM_IOCTL_KNPU_PERFMON_GET_VALUES);
	VALUE(DRM_KNPU_QUERY_NPU_INFO);
	VALUE(DRM_KNPU_MAX_REGIONS);
	VALUE(DRM_KNPU_PERF_MACS);
	VALUE(DRM_KNPU_PERF_DMA_RD_BYTES);
	VALUE(DRM_KNPU_PERF_DMA_WR_BYTES);
	VALUE(DRM_KNPU_PERF_OPS);
	VALUE(DRM_KNPU_PERF_COUNTERS);

	VALUE(PROT_READ);
	VALUE(PROT_WRITE);
	VALUE(MAP_SHARED);
	VALUE(CLOCK_MONOTONIC);
	SIZE(timespec);
	OFFSET(timespec, tv_nsec);
	printf("size off_t %zu\n", sizeof(off_t));

	VALUE(MFD_CLOEXEC);
	VALUE(RTLD_NOW);
	VALUE(RTLD_LOCAL);
	VALUE(RTLD_DL_SYMENT);
	VALUE(STT_OBJECT);
	VALUE(STT_FUNC);
	TSIZE(Dl_info);
	TOFFSET(Dl_info, dli_fbase);
	TOFFSET(Dl_info, dli_sname);
	TOFFSET(Dl_info, dli_saddr);
	TSIZE(Elf64_Sym);
	TOFFSET(Elf64_Sym, st_info);
	TOFFSET(Elf64_Sym, st_other);
	TOFFSET(Elf64_Sym, st_shndx);
	TOFFSET(Elf64_Sym, st_value);
	TOFFSET(Elf64_Sym, st_size);
	return 0;
}
