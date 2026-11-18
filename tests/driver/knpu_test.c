// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/*
 * V5: knpu driver test, run inside the kn1-evb guest (docs/design.md 11).
 *
 * Needs the knpu module loaded and debugfs mounted. Checks the ioctls,
 * the command stream validator and the submit checks (each wrong stream or
 * buffer must be refused), that buffers live above 4 GiB, that device
 * errors and a timed-out job come back as -EIO and leave the NPU usable,
 * the performance monitors and runtime PM. Prints PASS/FAIL lines and
 * exits with 1 if anything failed.
 */

#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/klog.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <time.h>
#include <unistd.h>

#include <drm/knpu_accel.h>

#include "knpu_cmd.h"

#define DEVICE		"/dev/accel/accel0"
#define DEBUGFS		"/sys/kernel/debug"
#define RUNTIME_STATUS	"/sys/class/accel/accel0/device/power/runtime_status"
#define CLK_ENABLED	DEBUGFS "/clk/npu_clk/clk_enable_count"
#define FOUR_GIB	0x100000000ull

static int failures;
static int checks;
static char debugfs_dir[256];

static void check(bool ok, const char *fmt, ...)
{
	va_list ap;

	checks++;
	if (!ok)
		failures++;
	printf("%s: ", ok ? "PASS" : "FAIL");
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
	putchar('\n');
	fflush(stdout);
}

static void fatal(const char *what)
{
	printf("FAIL: %s: %s\n", what, strerror(errno));
	exit(1);
}

/* ioctl() that returns 0 or -errno */
static int xioctl(int fd, unsigned long req, void *arg)
{
	int ret;

	do {
		ret = ioctl(fd, req, arg);
	} while (ret < 0 && (errno == EINTR || errno == EAGAIN));

	return ret < 0 ? -errno : 0;
}

static const char *err_name(int err)
{
	switch (-err) {
	case 0: return "0";
	case EINVAL: return "-EINVAL";
	case ENOENT: return "-ENOENT";
	case EOVERFLOW: return "-EOVERFLOW";
	case EFAULT: return "-EFAULT";
	case EIO: return "-EIO";
	case ETIME: return "-ETIME";
	default: return strerror(-err);
	}
}

static void check_err(int got, int want, const char *what)
{
	check(got == want, "%s: %s (got %s)", what, err_name(want),
	      err_name(got));
}

/* Command stream builder */
struct stream {
	uint32_t w[128];
	unsigned int n;
};

static uint32_t cmd(struct stream *s, uint16_t op, uint16_t param)
{
	uint32_t offset = s->n * 4;

	s->w[s->n++] = KNPU_CMD_WORD(op, param);
	return offset;
}

static uint32_t cmdp(struct stream *s, uint16_t op, uint16_t param,
		     uint32_t payload)
{
	uint32_t offset = cmd(s, op, param);

	s->w[s->n++] = payload;
	return offset;
}

static void dma(struct stream *s, uint16_t src, uint32_t src_off,
		uint16_t dst, uint32_t dst_off, uint32_t len, uint16_t count,
		uint32_t stride)
{
	cmdp(s, KNPU_SET_DMA_SRC, src, src_off);
	cmdp(s, KNPU_SET_DMA_DST, dst, dst_off);
	cmdp(s, KNPU_SET_DMA_LEN, 0, len);
	cmd(s, KNPU_SET_DMA_COUNT, count);
	cmdp(s, KNPU_SET_DMA_STRIDE, 0, stride);
	cmd(s, KNPU_OP_DMA, 0);
}

#define SRAM		KNPU_SPACE(KNPU_SPACE_SRAM, 0)
#define REGION(n)	KNPU_SPACE(KNPU_SPACE_REGION, n)

/* Buffer objects */
struct bo {
	uint32_t handle;
	uint64_t size;
	uint8_t *map;
};

static void bo_new(int fd, uint64_t size, struct bo *bo)
{
	struct drm_knpu_bo_create create = { .size = size };
	struct drm_knpu_bo_mmap_offset off = { 0 };

	if (xioctl(fd, DRM_IOCTL_KNPU_BO_CREATE, &create))
		fatal("BO_CREATE");
	off.handle = create.handle;
	if (xioctl(fd, DRM_IOCTL_KNPU_BO_MMAP_OFFSET, &off))
		fatal("BO_MMAP_OFFSET");
	bo->map = mmap(NULL, create.size, PROT_READ | PROT_WRITE, MAP_SHARED,
		       fd, off.offset);
	if (bo->map == MAP_FAILED)
		fatal("mmap");
	bo->handle = create.handle;
	bo->size = create.size;
}

static void bo_free(int fd, struct bo *bo)
{
	struct drm_gem_close close = { .handle = bo->handle };

	munmap(bo->map, bo->size);
	xioctl(fd, DRM_IOCTL_GEM_CLOSE, &close);
}

static void handle_close(int fd, uint32_t handle)
{
	struct drm_gem_close close = { .handle = handle };

	xioctl(fd, DRM_IOCTL_GEM_CLOSE, &close);
}

static int cmdstream_create(int fd, const struct stream *s, uint32_t *handle)
{
	struct drm_knpu_cmdstream_create args = {
		.size = s->n * 4,
		.data = (uintptr_t)s->w,
	};
	int ret = xioctl(fd, DRM_IOCTL_KNPU_CMDSTREAM_CREATE, &args);

	if (!ret && handle)
		*handle = args.handle;
	else if (!ret)
		handle_close(fd, args.handle);
	return ret;
}

static int submit(int fd, uint32_t stream, const uint32_t regions[8],
		  uint32_t perfmon)
{
	struct drm_knpu_submit args = {
		.cmdstream = stream,
		.perfmon_id = perfmon,
	};

	memcpy(args.region_bo_handles, regions, sizeof(args.region_bo_handles));
	return xioctl(fd, DRM_IOCTL_KNPU_SUBMIT, &args);
}

static int64_t now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static int bo_wait(int fd, uint32_t handle, int64_t timeout_ns)
{
	struct drm_knpu_bo_wait args = {
		.handle = handle,
		.timeout_ns = now_ns() + timeout_ns,
	};

	return xioctl(fd, DRM_IOCTL_KNPU_BO_WAIT, &args);
}

/* Create, submit and wait; returns the first error */
static int run(int fd, const struct stream *s, const uint32_t regions[8],
	       uint32_t perfmon, uint32_t wait_handle)
{
	uint32_t handle;
	int ret;

	ret = cmdstream_create(fd, s, &handle);
	if (ret)
		return ret;
	ret = submit(fd, handle, regions, perfmon);
	handle_close(fd, handle);
	if (ret)
		return ret;
	return bo_wait(fd, wait_handle, 5000000000ll);
}

/* Files in sysfs and debugfs */
static bool read_file(const char *path, char *buf, size_t size)
{
	FILE *f = fopen(path, "r");
	size_t n;

	if (!f)
		return false;
	n = fread(buf, 1, size - 1, f);
	buf[n] = '\0';
	fclose(f);
	return true;
}

static bool write_file(const char *path, const char *text)
{
	FILE *f = fopen(path, "w");
	bool ok;

	if (!f)
		return false;
	ok = fputs(text, f) >= 0;
	return fclose(f) == 0 && ok;
}

static void find_debugfs(void)
{
	glob_t g;

	if (glob(DEBUGFS "/accel/*/last_submit", 0, NULL, &g)) {
		mount("debugfs", DEBUGFS, "debugfs", 0, NULL);
		if (glob(DEBUGFS "/accel/*/last_submit", 0, NULL, &g))
			return;
	}
	snprintf(debugfs_dir, sizeof(debugfs_dir), "%s", g.gl_pathv[0]);
	*strrchr(debugfs_dir, '/') = '\0';
	globfree(&g);
}

static bool debugfs_path(const char *name, char *path, size_t size)
{
	if (!debugfs_dir[0])
		return false;
	snprintf(path, size, "%s/%s", debugfs_dir, name);
	return true;
}

/* Value of "<key> 0x..." in debugfs last_submit, or ~0 */
static uint64_t last_submit(const char *key)
{
	char path[300], buf[1024], *p;

	if (!debugfs_path("last_submit", path, sizeof(path)) ||
	    !read_file(path, buf, sizeof(buf)))
		return ~0ull;
	for (p = buf; p && *p; p = strchr(p, '\n') ? strchr(p, '\n') + 1 : NULL) {
		size_t len = strlen(key);

		if (!strncmp(p, key, len) && p[len] == ' ')
			return strtoull(p + len + 1, NULL, 16);
	}
	return ~0ull;
}

/* Number of kernel log lines containing @text */
static int klog_count(const char *text)
{
	static char buf[1 << 20];
	int n = klogctl(10 /* SYSLOG_ACTION_SIZE_BUFFER */, NULL, 0);
	int count = 0;
	char *p;

	if (n <= 0 || n > (int)sizeof(buf) - 1)
		n = sizeof(buf) - 1;
	n = klogctl(3 /* SYSLOG_ACTION_READ_ALL */, buf, n);
	if (n < 0)
		return -1;
	buf[n] = '\0';
	for (p = buf; (p = strstr(p, text)); p += strlen(text))
		count++;
	return count;
}

/*
 * A 1x1 convolution: IFM 2x2x4 at SRAM 0, weights 2x1x1x4 at 16, bias at
 * 24, requantization table at 32, OFM 2x2x2 at 64. Region 0 holds the 48
 * bytes of inputs and constants, region 1 receives the 8 output bytes.
 */
#define CONV_IN_BYTES	48
#define CONV_OUT_BYTES	8
#define CONV_IFM_ZP	1
#define CONV_OFM_ZP	(-3)
#define RQ_ONE		0x7fffffff	/* rescale(x) == x for small x */

static const int8_t conv_ifm[16] = {
	1, 2, 3, 4, -1, -2, -3, -4, 10, 0, -10, 5, 7, 7, 7, 7,
};
static const int8_t conv_w[8] = { 1, -1, 2, 0, 3, 1, -2, -1 };
static const int32_t conv_bias[2] = { 5, -7 };

static void conv_fill(uint8_t *in)
{
	int32_t rq[4] = { RQ_ONE, 0, RQ_ONE, 0 };

	memcpy(in, conv_ifm, 16);
	memcpy(in + 16, conv_w, 8);
	memcpy(in + 24, conv_bias, 8);
	memcpy(in + 32, rq, 16);
}

static void conv_expect(int8_t out[CONV_OUT_BYTES])
{
	int px, oc, ic;

	for (px = 0; px < 4; px++) {
		for (oc = 0; oc < 2; oc++) {
			int32_t acc = conv_bias[oc];

			for (ic = 0; ic < 4; ic++)
				acc += conv_w[oc * 4 + ic] *
				       (conv_ifm[px * 4 + ic] - CONV_IFM_ZP);
			acc += CONV_OFM_ZP;
			out[px * 2 + oc] = acc < -128 ? -128 : acc > 127 ? 127 : acc;
		}
	}
}

/* Returns the offset of OP_CONV; drop a wait to get a SYNC error */
static uint32_t conv_stream(struct stream *s, bool dma_wait, bool kernel_wait)
{
	uint32_t conv;

	s->n = 0;
	dma(s, REGION(0), 0, SRAM, 0, CONV_IN_BYTES, 1, 0);
	if (dma_wait)
		cmd(s, KNPU_OP_DMA_WAIT, 0);
	cmdp(s, KNPU_SET_IFM_BASE, 0, 0);
	cmdp(s, KNPU_SET_WEIGHT_BASE, 0, 16);
	cmdp(s, KNPU_SET_BIAS_BASE, 0, 24);
	cmdp(s, KNPU_SET_RQ_TABLE, 0, 32);
	cmdp(s, KNPU_SET_OFM_BASE, 0, 64);
	cmdp(s, KNPU_SET_IFM_SHAPE, 4, 2u << 16 | 2);
	cmdp(s, KNPU_SET_OFM_SHAPE, 2, 2u << 16 | 2);
	cmd(s, KNPU_SET_IFM_ZP, (uint16_t)CONV_IFM_ZP);
	cmd(s, KNPU_SET_OFM_ZP, (uint16_t)CONV_OFM_ZP);
	conv = cmd(s, KNPU_OP_CONV, 0);
	if (kernel_wait)
		cmd(s, KNPU_OP_KERNEL_WAIT, 0);
	dma(s, SRAM, 64, REGION(1), 0, CONV_OUT_BYTES, 1, 0);
	cmd(s, KNPU_OP_STOP, 0);
	return conv;
}

static struct drm_knpu_npu_info info;

static void test_query(int fd)
{
	struct drm_knpu_query q = { .type = DRM_KNPU_QUERY_NPU_INFO };
	uint32_t small;
	int ret;

	ret = xioctl(fd, DRM_IOCTL_KNPU_QUERY, &q);
	check(!ret && q.size == sizeof(info),
	      "QUERY without a pointer returns the size (%u)", q.size);

	q.pointer = (uintptr_t)&info;
	q.size = sizeof(info);
	ret = xioctl(fd, DRM_IOCTL_KNPU_QUERY, &q);
	check(!ret && info.id == KNPU_ID, "QUERY: ID 0x%08x is \"KNPU\"",
	      info.id);
	check(DRM_KNPU_VERSION_MAJOR(info.version) == 0,
	      "QUERY: VERSION %u.%u", DRM_KNPU_VERSION_MAJOR(info.version),
	      DRM_KNPU_VERSION_MINOR(info.version));
	check(DRM_KNPU_CONFIG_NUM_REGIONS(info.config) == 8 &&
	      DRM_KNPU_CONFIG_ROUND_DOUBLE(info.config) &&
	      DRM_KNPU_CONFIG_ROUND_SINGLE(info.config),
	      "QUERY: 8 regions, double and single rounding");
	check(info.sram_size >= 4096 && info.sram_size % 4096 == 0,
	      "QUERY: SRAM is %u KiB", info.sram_size / 1024);

	q.pointer = (uintptr_t)&small;
	q.size = sizeof(small);
	check_err(xioctl(fd, DRM_IOCTL_KNPU_QUERY, &q), -EINVAL,
		  "QUERY with a short buffer");
	q.type = 1;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_QUERY, &q), -EINVAL,
		  "QUERY of an unknown type");
}

static void test_bo(int fd)
{
	struct drm_knpu_bo_create create = { .size = 100 };
	struct drm_knpu_bo_mmap_offset off = { 0 };
	struct drm_knpu_bo_wait wait = { 0 };
	int ret;

	ret = xioctl(fd, DRM_IOCTL_KNPU_BO_CREATE, &create);
	check(!ret && create.handle && create.size == 4096,
	      "BO_CREATE rounds 100 bytes up to a page");
	handle_close(fd, create.handle);

	create.size = 0;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_BO_CREATE, &create), -EINVAL,
		  "BO_CREATE of 0 bytes");
	create.size = 4096;
	create.flags = 1;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_BO_CREATE, &create), -EINVAL,
		  "BO_CREATE with flags");

	off.handle = 0x7fffffff;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_BO_MMAP_OFFSET, &off), -ENOENT,
		  "BO_MMAP_OFFSET of an unknown handle");

	wait.handle = 0x7fffffff;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_BO_WAIT, &wait), -ENOENT,
		  "BO_WAIT on an unknown handle");
	wait.pad = 1;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_BO_WAIT, &wait), -EINVAL,
		  "BO_WAIT with pad set");
}

/* Region 0 -> SRAM as four 64-byte rows 256 bytes apart -> region 1 */
static void test_dma_job(int fd)
{
	uint32_t regions[8] = { 0 };
	struct stream s = { .n = 0 };
	struct bo in, out;
	bool same = true;
	int i, ret;

	bo_new(fd, 4096, &in);
	bo_new(fd, 4096, &out);
	for (i = 0; i < 4096; i++)
		in.map[i] = i * 7 + 3;
	memset(out.map, 0, 4096);

	dma(&s, REGION(0), 0, SRAM, 0, 64, 4, 256);
	dma(&s, SRAM, 0, REGION(1), 0x100, 256, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);

	regions[0] = in.handle;
	regions[1] = out.handle;
	ret = run(fd, &s, regions, 0, out.handle);
	check_err(ret, 0, "2-D DMA job from region 0 to region 1");
	for (i = 0; i < 256; i++)
		same &= out.map[0x100 + i] == in.map[(i / 64) * 256 + i % 64];
	check(same && out.map[0xff] == 0 && out.map[0x200] == 0,
	      "the rows arrive in region 1 at 0x100 and nothing else changes");

	/* The board's NPU pool lies above 4 GiB (kn1-evb.dts) */
	check(last_submit("qbase") >= FOUR_GIB &&
	      last_submit("qbase") != ~0ull,
	      "the command stream address 0x%llx is above 4 GiB",
	      (unsigned long long)last_submit("qbase"));
	check(last_submit("region0") >= FOUR_GIB &&
	      last_submit("region1") >= FOUR_GIB &&
	      last_submit("region2") == 0,
	      "region 0 and 1 bases 0x%llx and 0x%llx are above 4 GiB",
	      (unsigned long long)last_submit("region0"),
	      (unsigned long long)last_submit("region1"));
	check(last_submit("qsize") == s.n * 4, "QSIZE is the stream size");

	bo_free(fd, &in);
	bo_free(fd, &out);
}

static void test_conv_job_and_perfmon(int fd)
{
	struct drm_knpu_perfmon_create create = { 0 };
	struct drm_knpu_perfmon_get_values get = { 0 };
	struct drm_knpu_perfmon_destroy destroy = { 0 };
	uint32_t regions[8] = { 0 };
	int8_t expect[CONV_OUT_BYTES];
	struct stream s;
	uint64_t v[DRM_KNPU_PERF_COUNTERS];
	struct bo in, out;
	uint32_t handle;
	int ret;

	bo_new(fd, 4096, &in);
	bo_new(fd, 4096, &out);
	conv_fill(in.map);
	memset(out.map, 0x55, 4096);
	conv_expect(expect);
	regions[0] = in.handle;
	regions[1] = out.handle;

	ret = xioctl(fd, DRM_IOCTL_KNPU_PERFMON_CREATE, &create);
	check(!ret && create.id, "PERFMON_CREATE returns id %u", create.id);

	conv_stream(&s, true, true);
	ret = run(fd, &s, regions, create.id, out.handle);
	check_err(ret, 0, "1x1 convolution job");
	check(!memcmp(out.map, expect, CONV_OUT_BYTES),
	      "the convolution output matches the reference");

	get.id = create.id;
	get.values_ptr = (uintptr_t)v;
	ret = xioctl(fd, DRM_IOCTL_KNPU_PERFMON_GET_VALUES, &get);
	check(!ret && v[DRM_KNPU_PERF_MACS] == 32 &&
	      v[DRM_KNPU_PERF_DMA_RD_BYTES] == CONV_IN_BYTES &&
	      v[DRM_KNPU_PERF_DMA_WR_BYTES] == CONV_OUT_BYTES &&
	      v[DRM_KNPU_PERF_OPS] == 3,
	      "perfmon: MACs %llu, DMA read %llu, DMA write %llu, ops %llu",
	      (unsigned long long)v[0], (unsigned long long)v[1],
	      (unsigned long long)v[2], (unsigned long long)v[3]);

	/* A job without the monitor leaves it alone; one with it adds up */
	run(fd, &s, regions, 0, out.handle);
	run(fd, &s, regions, create.id, out.handle);
	ret = xioctl(fd, DRM_IOCTL_KNPU_PERFMON_GET_VALUES, &get);
	check(!ret && v[DRM_KNPU_PERF_MACS] == 64 &&
	      v[DRM_KNPU_PERF_OPS] == 6,
	      "perfmon adds up only the jobs submitted with it");

	ret = cmdstream_create(fd, &s, &handle);
	check_err(ret ? ret : submit(fd, handle, regions, 0x7fffffff), -ENOENT,
		  "SUBMIT with an unknown perfmon");
	if (!ret)
		handle_close(fd, handle);
	destroy.id = create.id;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_PERFMON_DESTROY, &destroy), 0,
		  "PERFMON_DESTROY");
	check_err(xioctl(fd, DRM_IOCTL_KNPU_PERFMON_GET_VALUES, &get), -ENOENT,
		  "PERFMON_GET_VALUES after destroy");
	check_err(xioctl(fd, DRM_IOCTL_KNPU_PERFMON_DESTROY, &destroy),
		  -ENOENT, "PERFMON_DESTROY twice");

	bo_free(fd, &in);
	bo_free(fd, &out);
}

static void expect_invalid(int fd, const struct stream *s, const char *what)
{
	check_err(cmdstream_create(fd, s, NULL), -EINVAL, what);
}

static void test_validator(int fd)
{
	struct drm_knpu_cmdstream_create args = { 0 };
	uint32_t sram = info.sram_size;
	struct stream s;

	s.n = 0;
	cmd(&s, KNPU_OP_STOP, 0);
	check_err(cmdstream_create(fd, &s, NULL), 0, "a lone OP_STOP is valid");

	/* Every command of spec v0 is known, also those M1 does not run */
	s.n = 0;
	cmd(&s, KNPU_OP_NOP, 0);
	cmd(&s, KNPU_OP_ADD, 0);
	cmd(&s, KNPU_OP_AVGPOOL, 0);
	cmd(&s, KNPU_OP_KERNEL_WAIT, 1);
	cmd(&s, KNPU_SET_IFM2_ZP, 3);
	cmd(&s, KNPU_SET_KERNEL, 0x0303);
	cmd(&s, KNPU_SET_STRIDE, 0x0101);
	cmd(&s, KNPU_SET_PAD, 0x1111);
	cmd(&s, KNPU_SET_CLAMP, 0x7f80);
	cmd(&s, KNPU_SET_ROUND_MODE, KNPU_ROUND_SINGLE);
	cmdp(&s, KNPU_SET_IFM2_BASE, 0, 0x100);
	cmdp(&s, KNPU_SET_IFM_SCALE, 0xffff, 0x40000000);
	cmdp(&s, KNPU_SET_IFM2_SCALE, 0xffff, 0x40000000);
	cmdp(&s, KNPU_SET_OFM_SCALE, 0xffff, 0x40000000);
	cmd(&s, KNPU_OP_STOP, 0);
	check_err(cmdstream_create(fd, &s, NULL), 0,
		  "every spec v0 command is accepted");

	args.data = (uintptr_t)s.w;
	args.size = 0;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_CMDSTREAM_CREATE, &args), -EINVAL,
		  "an empty stream");
	args.size = 6;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_CMDSTREAM_CREATE, &args), -EINVAL,
		  "a stream size that is not a multiple of 4");
	args.size = 4;
	args.flags = 1;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_CMDSTREAM_CREATE, &args), -EINVAL,
		  "CMDSTREAM_CREATE with flags");
	args.flags = 0;
	args.data = 0;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_CMDSTREAM_CREATE, &args), -EINVAL,
		  "CMDSTREAM_CREATE without data");
	args.data = 16;
	check_err(xioctl(fd, DRM_IOCTL_KNPU_CMDSTREAM_CREATE, &args), -EFAULT,
		  "CMDSTREAM_CREATE from a bad pointer");

	s.n = 0;
	cmd(&s, 0x0005, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_invalid(fd, &s, "unknown command 0x0005");

	s.n = 0;
	cmd(&s, 0x8000, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_invalid(fd, &s, "a command word with bit 15 set");

	s.n = 0;
	cmd(&s, KNPU_OP_STOP, 0);
	cmd(&s, KNPU_SET_DMA_LEN, 0);
	expect_invalid(fd, &s, "a payload command cut off at the end");

	s.n = 0;
	cmd(&s, KNPU_OP_NOP, 0);
	expect_invalid(fd, &s, "a stream without OP_STOP");

	s.n = 0;
	cmd(&s, KNPU_OP_STOP, 0);
	cmd(&s, KNPU_OP_NOP, 0);
	expect_invalid(fd, &s, "words after OP_STOP");

	s.n = 0;
	cmdp(&s, KNPU_SET_DMA_LEN, 0, 16);
	cmd(&s, KNPU_OP_DMA, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_invalid(fd, &s, "DMA from SRAM to SRAM (the default spaces)");

	s.n = 0;
	dma(&s, REGION(0), 0, REGION(1), 0, 16, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_invalid(fd, &s, "DMA from a region to a region");

	s.n = 0;
	cmdp(&s, KNPU_SET_DMA_SRC, KNPU_SPACE(2, 0), 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_invalid(fd, &s, "DMA space of an unknown kind");

	s.n = 0;
	cmdp(&s, KNPU_SET_DMA_SRC, REGION(8), 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_invalid(fd, &s, "DMA region 8");

	s.n = 0;
	cmdp(&s, KNPU_SET_DMA_DST, KNPU_SPACE(KNPU_SPACE_SRAM, 1), 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_invalid(fd, &s, "SRAM space with a region number");

	s.n = 0;
	dma(&s, REGION(0), 0, SRAM, sram - 16, 32, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_invalid(fd, &s, "DMA into SRAM past SRAM_SIZE");

	s.n = 0;
	dma(&s, SRAM, 0, REGION(0), 0, sram / 2, 3, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_invalid(fd, &s, "DMA of LEN x COUNT larger than the SRAM");

	s.n = 0;
	dma(&s, REGION(0), 0, SRAM, sram - 16, 16, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	check_err(cmdstream_create(fd, &s, NULL), 0,
		  "DMA ending exactly at SRAM_SIZE is valid");
}

static void test_submit_checks(int fd)
{
	uint32_t regions[8] = { 0 };
	uint32_t exact, two_d;
	struct stream s;
	struct bo a, b;
	int ret;

	bo_new(fd, 8192, &a);
	bo_new(fd, 4096, &b);

	/* Needs 8192 bytes of region 0 and 16 of region 1 */
	s.n = 0;
	dma(&s, REGION(0), 8192 - 64, SRAM, 0, 64, 1, 0);
	dma(&s, SRAM, 0, REGION(1), 0, 16, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	ret = cmdstream_create(fd, &s, &exact);
	check_err(ret, 0, "stream that needs exactly 8 KiB of region 0");

	regions[0] = a.handle;
	regions[1] = b.handle;
	check_err(submit(fd, exact, regions, 0), 0,
		  "SUBMIT with buffers that are just large enough");
	bo_wait(fd, b.handle, 5000000000ll);

	regions[0] = b.handle;
	regions[1] = a.handle;
	check_err(submit(fd, exact, regions, 0), -EOVERFLOW,
		  "SUBMIT with a region 0 buffer one page too small");

	/* 16 rows 4 KiB apart need 15 * 4096 + 16 bytes */
	s.n = 0;
	dma(&s, REGION(0), 0, SRAM, 0, 16, 16, 4096);
	cmd(&s, KNPU_OP_STOP, 0);
	ret = cmdstream_create(fd, &s, &two_d);
	check_err(ret, 0, "stream with a 2-D DMA");
	memset(regions, 0, sizeof(regions));
	regions[0] = a.handle;
	check_err(submit(fd, two_d, regions, 0), -EOVERFLOW,
		  "SUBMIT where the DMA stride runs past the buffer");

	memset(regions, 0, sizeof(regions));
	regions[0] = a.handle;
	check_err(submit(fd, exact, regions, 0), -EINVAL,
		  "SUBMIT without a buffer for a used region");

	regions[1] = b.handle;
	regions[2] = b.handle;
	check_err(submit(fd, exact, regions, 0), -EINVAL,
		  "SUBMIT with a buffer on an unused region");

	regions[2] = 0;
	regions[1] = a.handle;
	check_err(submit(fd, exact, regions, 0), -EINVAL,
		  "SUBMIT with one buffer on two regions");

	regions[1] = 0x7fffffff;
	check_err(submit(fd, exact, regions, 0), -ENOENT,
		  "SUBMIT with an unknown region buffer");

	regions[1] = exact;
	check_err(submit(fd, exact, regions, 0), -EINVAL,
		  "SUBMIT with a command stream as a region buffer");

	regions[1] = b.handle;
	check_err(submit(fd, a.handle, regions, 0), -EINVAL,
		  "SUBMIT of a plain buffer as the command stream");
	check_err(submit(fd, 0x7fffffff, regions, 0), -ENOENT,
		  "SUBMIT of an unknown command stream");

	{
		struct drm_knpu_bo_mmap_offset off = { .handle = exact };
		void *p = MAP_FAILED;

		if (!xioctl(fd, DRM_IOCTL_KNPU_BO_MMAP_OFFSET, &off))
			p = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
				 MAP_SHARED, fd, off.offset);
		check(p == MAP_FAILED, "a command stream cannot be mapped");
		if (p != MAP_FAILED)
			munmap(p, 4096);
	}

	handle_close(fd, exact);
	handle_close(fd, two_d);
	bo_free(fd, &a);
	bo_free(fd, &b);
}

/* Streams the validator accepts but the device stops (spec 10) */
static void test_device_errors(int fd)
{
	uint32_t regions[8] = { 0 };
	int8_t expect[CONV_OUT_BYTES];
	char text[96];
	struct stream s;
	struct bo in, out;
	uint32_t conv;
	int before, ret;

	bo_new(fd, 4096, &in);
	bo_new(fd, 4096, &out);
	conv_fill(in.map);
	regions[0] = in.handle;
	regions[1] = out.handle;

	/* OP_CONV reads SRAM that the DMA before it may still be writing */
	conv = conv_stream(&s, false, true);
	snprintf(text, sizeof(text), "job failed: SYNC (7) at command offset 0x%x",
		 conv);
	before = klog_count(text);
	ret = run(fd, &s, regions, 0, out.handle);
	check_err(ret, -EIO, "a stream with a missing DMA wait fails");
	check(klog_count(text) == before + 1, "the kernel log has \"%s\"",
	      text);

	s.n = 0;
	cmd(&s, KNPU_SET_KERNEL, 0);
	dma(&s, SRAM, 0, REGION(1), 0, 16, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	memset(regions, 0, sizeof(regions));
	regions[1] = out.handle;
	snprintf(text, sizeof(text),
		 "job failed: BAD_PARAM (5) at command offset 0x0");
	before = klog_count(text);
	ret = run(fd, &s, regions, 0, out.handle);
	check_err(ret, -EIO, "a stream with a kernel size of 0 fails");
	check(klog_count(text) == before + 1, "the kernel log has \"%s\"",
	      text);

	/* The device needs no reset after an error */
	regions[0] = in.handle;
	conv_stream(&s, true, true);
	conv_expect(expect);
	memset(out.map, 0, 4096);
	ret = run(fd, &s, regions, 0, out.handle);
	check(!ret && !memcmp(out.map, expect, CONV_OUT_BYTES),
	      "the next job after the errors runs correctly");

	bo_free(fd, &in);
	bo_free(fd, &out);
}

static void test_timeout(int fd)
{
	uint32_t regions[8] = { 0 };
	char path[300], buf[16];
	int8_t expect[CONV_OUT_BYTES];
	struct stream s;
	struct bo in, out;
	int64_t start, ms;
	int before, ret;

	if (!debugfs_path("inject_hang", path, sizeof(path)) ||
	    !write_file(path, "1")) {
		check(false, "debugfs inject_hang is writable");
		return;
	}

	bo_new(fd, 4096, &in);
	bo_new(fd, 4096, &out);
	conv_fill(in.map);
	regions[0] = in.handle;
	regions[1] = out.handle;
	conv_stream(&s, true, true);

	before = klog_count("job timed out");
	start = now_ns();
	ret = run(fd, &s, regions, 0, out.handle);
	ms = (now_ns() - start) / 1000000;
	check_err(ret, -EIO, "a job that never starts fails");
	check(ms >= 400 && ms < 5000, "it fails after the 500 ms timeout (%lld ms)",
	      (long long)ms);
	check(klog_count("job timed out") == before + 1,
	      "the kernel log reports the timeout");
	check(read_file(path, buf, sizeof(buf)) && buf[0] == 'N',
	      "inject_hang clears itself");

	/* Soft reset during recovery left the NPU usable */
	conv_expect(expect);
	memset(out.map, 0, 4096);
	ret = run(fd, &s, regions, 0, out.handle);
	check(!ret && !memcmp(out.map, expect, CONV_OUT_BYTES),
	      "the job after the reset runs correctly");

	bo_free(fd, &in);
	bo_free(fd, &out);
}

static void test_runtime_pm(void)
{
	char buf[32];

	/* Autosuspend is 50 ms after the last job */
	usleep(300000);
	check(read_file(RUNTIME_STATUS, buf, sizeof(buf)) &&
	      !strncmp(buf, "suspended", 9),
	      "the idle NPU is runtime suspended");
	check(read_file(CLK_ENABLED, buf, sizeof(buf)) && atoi(buf) == 0,
	      "the idle NPU's clock is off");
}

int main(void)
{
	int fd = open(DEVICE, O_RDWR | O_CLOEXEC);

	if (fd < 0)
		fatal("open " DEVICE);
	find_debugfs();
	check(debugfs_dir[0], "knpu debugfs directory found (%s)",
	      debugfs_dir[0] ? debugfs_dir : "none");

	test_query(fd);
	test_bo(fd);
	test_dma_job(fd);
	test_conv_job_and_perfmon(fd);
	test_validator(fd);
	test_submit_checks(fd);
	test_device_errors(fd);
	test_timeout(fd);
	close(fd);
	test_runtime_pm();

	if (failures) {
		printf("%d of %d knpu checks failed.\n", failures, checks);
		return 1;
	}
	printf("All %d knpu checks passed.\n", checks);
	return 0;
}
