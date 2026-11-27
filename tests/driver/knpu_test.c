// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/*
 * V5: knpu driver test, run inside the kn1-evb guest (docs/design.md 11).
 *
 * Needs the knpu module loaded, debugfs mounted and root (for drm.debug
 * and /dev/kmsg). Checks the ioctls, the command stream validator and the
 * submit checks (each wrong stream or buffer must be refused, for the
 * reason the driver logs with drm.debug on), random changes of a valid
 * stream against a reference validator, that buffers live above 4 GiB,
 * that device errors and a timed-out job come back as -EIO and leave the
 * NPU usable, the write fences of output buffers, the performance
 * monitors, runtime PM and that the kernel logged no warnings. Prints
 * PASS/FAIL lines and exits with 1 if anything failed.
 */

#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <poll.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <time.h>
#include <unistd.h>

#include <drm/knpu_accel.h>
#include <linux/dma-buf.h>

#include "knpu_cmd.h"

#define DEVICE		"/dev/accel/accel0"
#define DEBUGFS		"/sys/kernel/debug"
#define RUNTIME_STATUS	"/sys/class/accel/accel0/device/power/runtime_status"
#define CLK_ENABLED	DEBUGFS "/clk/npu_clk/clk_enable_count"
#define DRM_DEBUG_PARAM	"/sys/module/drm/parameters/debug"
#define DRM_UT_DRIVER	0x2ul
#define FOUR_GIB	0x100000000ull
#define ARRAY_SIZE(a)	(sizeof(a) / sizeof((a)[0]))

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

/* Returns the offset of OP_DMA */
static uint32_t dma(struct stream *s, uint16_t src, uint32_t src_off,
		    uint16_t dst, uint32_t dst_off, uint32_t len,
		    uint16_t count, uint32_t stride)
{
	cmdp(s, KNPU_SET_DMA_SRC, src, src_off);
	cmdp(s, KNPU_SET_DMA_DST, dst, dst_off);
	cmdp(s, KNPU_SET_DMA_LEN, 0, len);
	cmd(s, KNPU_SET_DMA_COUNT, count);
	cmdp(s, KNPU_SET_DMA_STRIDE, 0, stride);
	return cmd(s, KNPU_OP_DMA, 0);
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

/*
 * The kernel log, read through /dev/kmsg from its oldest record on. The
 * validator tests overwrite the 128 KiB log buffer with debug messages
 * more than once, so a dmesg after the test only shows its end: knpu-test
 * reads every record as it comes and looks for warnings itself. The
 * messages since klog_mark() are kept for the checks.
 */
static int klog_fd = -1;
static char klog_text[1 << 16];
static size_t klog_len;
static int klog_records, klog_lost, klog_warnings;
static char klog_warning[256];

static void klog_read(void)
{
	static char rec[8192];

	while (klog_fd >= 0) {
		ssize_t n = read(klog_fd, rec, sizeof(rec) - 1);
		size_t len;
		char *msg;

		if (n < 0 && errno == EPIPE) {
			/* Overwritten before it could be read */
			klog_lost++;
			continue;
		}
		if (n <= 0)
			break;
		rec[n] = '\0';
		klog_records++;

		/* "<level>,<seq>,<usec>,<flags>;<text>\n", then " KEY=value" */
		msg = strchr(rec, ';');
		if (!msg)
			continue;
		msg++;
		msg[strcspn(msg, "\n")] = '\0';

		if ((strstr(msg, "WARNING:") || strstr(msg, "BUG:") ||
		     strstr(msg, "Oops") || strstr(msg, "refcount_t:")) &&
		    !klog_warnings++)
			snprintf(klog_warning, sizeof(klog_warning), "%s", msg);

		len = strlen(msg);
		if (klog_len + len + 2 > sizeof(klog_text))
			continue;
		memcpy(klog_text + klog_len, msg, len);
		klog_len += len;
		klog_text[klog_len++] = '\n';
		klog_text[klog_len] = '\0';
	}
}

/* Keep only the messages that come after this point */
static void klog_mark(void)
{
	klog_read();
	klog_len = 0;
	klog_text[0] = '\0';
}

/* Number of times @text appears in the messages since klog_mark() */
static int klog_count(const char *text)
{
	int count = 0;
	char *p;

	klog_read();
	for (p = klog_text; (p = strstr(p, text)); p += strlen(text))
		count++;
	return count;
}

/*
 * The knpu driver's debug messages since klog_mark(), one per line,
 * without the "knpu <device>: [drm:<function> [knpu]] " in front.
 */
static const char *klog_dbg(void)
{
	static const char tag[] = " [knpu]] ";
	static char out[sizeof(klog_text)];
	const char *line, *eol, *msg;
	size_t used = 0;

	klog_read();
	for (line = klog_text; *line; line = eol + 1) {
		eol = strchr(line, '\n');
		msg = strstr(line, tag);
		if (!msg || msg > eol)
			continue;
		msg += sizeof(tag) - 1;
		memcpy(out + used, msg, eol + 1 - msg);
		used += eol + 1 - msg;
	}
	out[used] = '\0';
	return out;
}

/* Whether the knpu driver logged exactly @msg since klog_mark() */
static bool logged(const char *msg)
{
	size_t len = strlen(msg);
	const char *line, *eol;

	for (line = klog_dbg(); *line; line = eol + 1) {
		eol = strchr(line, '\n');
		if ((size_t)(eol - line) == len && !strncmp(line, msg, len))
			return true;
	}
	return false;
}

/* A message a check expects, formatted into one of a few buffers */
static const char *expected(const char *fmt, ...)
	__attribute__((format(printf, 1, 2)));

static const char *expected(const char *fmt, ...)
{
	static char buf[4][128];
	static unsigned int next;
	char *b = buf[next++ % 4];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(b, sizeof(buf[0]), fmt, ap);
	va_end(ap);
	return b;
}

/* drm.debug: its driver messages (0x2) give the validator's reasons */
static char drm_debug_was[32];

static bool drm_debug_on(void)
{
	char text[32];

	if (!read_file(DRM_DEBUG_PARAM, drm_debug_was, sizeof(drm_debug_was)))
		return false;
	drm_debug_was[strcspn(drm_debug_was, "\n")] = '\0';
	snprintf(text, sizeof(text), "%lu",
		 strtoul(drm_debug_was, NULL, 0) | DRM_UT_DRIVER);
	return write_file(DRM_DEBUG_PARAM, text);
}

static bool drm_debug_restore(void)
{
	char text[32];

	return write_file(DRM_DEBUG_PARAM, drm_debug_was) &&
	       read_file(DRM_DEBUG_PARAM, text, sizeof(text)) &&
	       strtoul(text, NULL, 0) == strtoul(drm_debug_was, NULL, 0);
}

/* CMDSTREAM_CREATE refuses the stream with -EINVAL and logs @msg */
static void expect_refused(int fd, const struct stream *s, const char *what,
			   const char *msg)
{
	bool seen;
	int ret;

	klog_mark();
	ret = cmdstream_create(fd, s, NULL);
	seen = logged(msg);
	check(ret == -EINVAL && seen, "%s: -EINVAL, \"%s\" (got %s, %s)", what,
	      msg, err_name(ret), seen ? "logged" : "not logged");
}

/* CMDSTREAM_CREATE takes the stream and logs @msg, an OP_DMA's range */
static void expect_valid(int fd, const struct stream *s, uint32_t *handle,
			 const char *what, const char *msg)
{
	uint32_t h = 0;
	bool seen;
	int ret;

	klog_mark();
	ret = cmdstream_create(fd, s, handle ? &h : NULL);
	seen = logged(msg);
	if (handle)
		*handle = ret ? 0 : h;
	check(!ret && seen, "%s: valid, \"%s\" (got %s, %s)", what, msg,
	      err_name(ret), seen ? "logged" : "not logged");
}

/* SUBMIT returns @want and logs @msg, or nothing if @msg is NULL */
static void expect_submit(int fd, uint32_t stream, const uint32_t regions[8],
			  int want, const char *what, const char *msg)
{
	bool seen;
	int ret;

	klog_mark();
	ret = submit(fd, stream, regions, 0);
	if (!msg) {
		seen = !*klog_dbg();
		check(ret == want && seen,
		      "%s: %s, nothing logged (got %s, %s)", what,
		      err_name(want), err_name(ret),
		      seen ? "nothing logged" : "logged");
		return;
	}
	seen = logged(msg);
	check(ret == want && seen, "%s: %s, \"%s\" (got %s, %s)", what,
	      err_name(want), msg, err_name(ret),
	      seen ? "logged" : "not logged");
}

/* CMDSTREAM_CREATE refuses the arguments before looking at the words */
static void expect_args_refused(int fd, struct drm_knpu_cmdstream_create *args,
				int want, const char *what)
{
	bool quiet;
	int ret;

	klog_mark();
	ret = xioctl(fd, DRM_IOCTL_KNPU_CMDSTREAM_CREATE, args);
	quiet = !*klog_dbg();
	check(ret == want && quiet, "%s: %s, nothing logged (got %s, %s)", what,
	      err_name(want), err_name(ret),
	      quiet ? "nothing logged" : "logged");
	if (!ret)
		handle_close(fd, args->handle);
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

/* Refused streams must name the rule and the command (docs/design.md 8.4) */
static void test_validator(int fd)
{
	struct drm_knpu_cmdstream_create args = { 0 };
	uint32_t sram = info.sram_size;
	struct stream s;
	uint32_t at;

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

	/* The words are valid, only the arguments are wrong */
	s.n = 0;
	cmd(&s, KNPU_OP_STOP, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	args.data = (uintptr_t)s.w;
	args.size = 0;
	expect_args_refused(fd, &args, -EINVAL, "an empty stream");
	args.size = 6;
	expect_args_refused(fd, &args, -EINVAL,
			    "a stream size that is not a multiple of 4");
	args.size = 4;
	args.flags = 1;
	expect_args_refused(fd, &args, -EINVAL, "CMDSTREAM_CREATE with flags");
	args.flags = 0;
	args.pad = 1;
	expect_args_refused(fd, &args, -EINVAL,
			    "CMDSTREAM_CREATE with pad set");
	args.pad = 0;
	args.data = 0;
	expect_args_refused(fd, &args, -EINVAL,
			    "CMDSTREAM_CREATE without data");
	args.data = 16;
	expect_args_refused(fd, &args, -EFAULT,
			    "CMDSTREAM_CREATE from a bad pointer");

	s.n = 0;
	at = cmd(&s, 0x0005, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "unknown command 0x0005",
		       expected("cmd 0x%x: unknown command 0x0005", at));

	s.n = 0;
	at = cmd(&s, 0x8000, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "a command word with bit 15 set",
		       expected("cmd 0x%x: unknown command 0x8000", at));

	s.n = 0;
	at = cmdp(&s, 0x4020, 0, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "an unknown command with the payload bit set",
		       expected("cmd 0x%x: unknown command 0x4020", at));

	s.n = 0;
	at = cmd(&s, KNPU_SET_DMA_LEN, 0);
	expect_refused(fd, &s, "a payload command cut off at the end",
		       expected("cmd 0x%x: payload is missing", at));

	s.n = 0;
	cmd(&s, KNPU_OP_NOP, 0);
	expect_refused(fd, &s, "a stream without OP_STOP",
		       "command stream does not end with OP_STOP");

	s.n = 0;
	at = cmd(&s, KNPU_OP_STOP, 0);
	cmd(&s, KNPU_OP_NOP, 0);
	expect_refused(fd, &s, "words after OP_STOP",
		       expected("cmd 0x%x: OP_STOP is not the last word", at));

	s.n = 0;
	cmdp(&s, KNPU_SET_DMA_LEN, 0, 16);
	at = cmd(&s, KNPU_OP_DMA, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "DMA from SRAM to SRAM (the default spaces)",
		       expected("cmd 0x%x: DMA needs exactly one SRAM side",
				at));

	s.n = 0;
	at = dma(&s, REGION(0), 0, REGION(1), 0, 16, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "DMA from a region to a region",
		       expected("cmd 0x%x: DMA needs exactly one SRAM side",
				at));

	/* The DMA state carries over from one OP_DMA to the next */
	s.n = 0;
	dma(&s, REGION(0), 0, SRAM, 0, 16, 1, 0);
	cmdp(&s, KNPU_SET_DMA_SRC, SRAM, 0x100);
	at = cmd(&s, KNPU_OP_DMA, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "a second OP_DMA after the source became SRAM",
		       expected("cmd 0x%x: DMA needs exactly one SRAM side",
				at));

	s.n = 0;
	dma(&s, REGION(2), 0x100, SRAM, 0, 16, 4, 64);
	cmdp(&s, KNPU_SET_DMA_LEN, 0, 32);
	at = cmd(&s, KNPU_OP_DMA, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_valid(fd, &s, NULL, "a second OP_DMA with only LEN changed",
		     expected("cmd 0x%x: DMA from region 2, 0x100..0x1e0", at));

	/* Unset, DST is the SRAM at 0, COUNT 1 and STRIDE 0 */
	s.n = 0;
	cmdp(&s, KNPU_SET_DMA_SRC, REGION(0), 0x100);
	cmdp(&s, KNPU_SET_DMA_LEN, 0, 16);
	at = cmd(&s, KNPU_OP_DMA, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_valid(fd, &s, NULL,
		     "an OP_DMA with the default DST, COUNT and STRIDE",
		     expected("cmd 0x%x: DMA from region 0, 0x100..0x110", at));

	s.n = 0;
	at = cmdp(&s, KNPU_SET_DMA_SRC, KNPU_SPACE(2, 0), 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "DMA source space of an unknown kind",
		       expected("cmd 0x%x: bad DMA space 0x200", at));

	s.n = 0;
	at = cmdp(&s, KNPU_SET_DMA_DST, KNPU_SPACE(2, 0), 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "DMA destination space of an unknown kind",
		       expected("cmd 0x%x: bad DMA space 0x200", at));

	s.n = 0;
	at = cmdp(&s, KNPU_SET_DMA_SRC, REGION(8), 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "DMA source region 8",
		       expected("cmd 0x%x: bad DMA space 0x8", at));

	s.n = 0;
	at = cmdp(&s, KNPU_SET_DMA_DST, REGION(8), 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "DMA destination region 8",
		       expected("cmd 0x%x: bad DMA space 0x8", at));

	s.n = 0;
	at = cmdp(&s, KNPU_SET_DMA_DST, KNPU_SPACE(KNPU_SPACE_SRAM, 1), 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "SRAM space with a region number",
		       expected("cmd 0x%x: bad DMA space 0x101", at));

	s.n = 0;
	at = dma(&s, REGION(0), 0, SRAM, sram - 16, 32, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "DMA into SRAM past SRAM_SIZE",
		       expected("cmd 0x%x: SRAM range 0x%x+0x20 is outside the SRAM",
				at, sram - 16));

	s.n = 0;
	at = dma(&s, SRAM, 0, REGION(0), 0, sram / 2, 3, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "DMA of LEN x COUNT larger than the SRAM",
		       expected("cmd 0x%x: SRAM range 0x0+0x%x is outside the SRAM",
				at, sram / 2 * 3));

	/* Sums that wrap around in 32 bits must not pass as small ones */
	s.n = 0;
	at = dma(&s, REGION(0), 0, SRAM, 0, 0x80000000, 2, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "DMA of LEN x COUNT = 4 GiB",
		       expected("cmd 0x%x: SRAM range 0x0+0x100000000 is outside the SRAM",
				at));

	s.n = 0;
	at = dma(&s, REGION(0), 0, SRAM, 0xffffff00, 0x200, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_refused(fd, &s, "DMA into SRAM at 0xffffff00",
		       expected("cmd 0x%x: SRAM range 0xffffff00+0x200 is outside the SRAM",
				at));

	s.n = 0;
	at = dma(&s, REGION(0), 0, SRAM, sram - 16, 16, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_valid(fd, &s, NULL, "DMA ending exactly at SRAM_SIZE",
		     expected("cmd 0x%x: DMA from region 0, 0x0..0x10", at));
}

/*
 * Random streams: the convolution stream with a few words changed, dropped
 * or added. The kernel and a reference validator must judge each alike,
 * down to the messages. Every stream has its own seed, so a failing one
 * can be made again from RANDOM_SEED and its number.
 */
#define RANDOM_STREAMS	1000
#define RANDOM_SEED	0x4b4e5055ull	/* "KNPU" */

static const uint16_t known_commands[] = {
	KNPU_OP_STOP, KNPU_OP_NOP, KNPU_OP_CONV, KNPU_OP_ADD,
	KNPU_OP_AVGPOOL, KNPU_OP_DMA, KNPU_OP_DMA_WAIT, KNPU_OP_KERNEL_WAIT,
	KNPU_SET_IFM_ZP, KNPU_SET_IFM2_ZP, KNPU_SET_OFM_ZP, KNPU_SET_KERNEL,
	KNPU_SET_STRIDE, KNPU_SET_PAD, KNPU_SET_CLAMP, KNPU_SET_ROUND_MODE,
	KNPU_SET_DMA_COUNT, KNPU_SET_IFM_BASE, KNPU_SET_IFM2_BASE,
	KNPU_SET_OFM_BASE, KNPU_SET_WEIGHT_BASE, KNPU_SET_BIAS_BASE,
	KNPU_SET_RQ_TABLE, KNPU_SET_IFM_SHAPE, KNPU_SET_OFM_SHAPE,
	KNPU_SET_IFM_SCALE, KNPU_SET_IFM2_SCALE, KNPU_SET_OFM_SCALE,
	KNPU_SET_DMA_SRC, KNPU_SET_DMA_DST, KNPU_SET_DMA_LEN,
	KNPU_SET_DMA_STRIDE,
};

static bool known_command(uint16_t op)
{
	size_t i;

	for (i = 0; i < ARRAY_SIZE(known_commands); i++)
		if (known_commands[i] == op)
			return true;
	return false;
}

struct dma_side {
	bool sram;
	uint8_t region;
	uint32_t offset;
};

/* The messages the reference validator expects, one per line */
static char ref_log[4096];

static void ref_say(const char *fmt, ...)
	__attribute__((format(printf, 1, 2)));

static void ref_say(const char *fmt, ...)
{
	size_t used = strlen(ref_log);
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(ref_log + used, sizeof(ref_log) - used, fmt, ap);
	va_end(ap);
	used = strlen(ref_log);
	if (used + 1 < sizeof(ref_log))
		strcpy(ref_log + used, "\n");
}

/*
 * docs/design.md 8.4 written again: puts the messages the validator should
 * log for the stream in ref_log and returns whether the stream is valid.
 */
static bool ref_validate(const uint32_t *w, unsigned int n, uint32_t sram_size)
{
	struct dma_side src = { .sram = true }, dst = { .sram = true };
	uint32_t len = 0, count = 1, stride = 0;
	unsigned int i;

	ref_log[0] = '\0';
	for (i = 0; i < n; i++) {
		uint32_t at = i * 4, payload = 0;
		uint16_t op = w[i] & KNPU_CMD_MASK;
		uint16_t param = KNPU_CMD_PARAM(w[i]);
		uint8_t kind = param >> 8, number = param & 0xff;
		struct dma_side *side, *sram, *region;
		uint64_t total, end;

		if (op & KNPU_CMD_PAYLOAD) {
			if (i + 1 == n) {
				ref_say("cmd 0x%x: payload is missing", at);
				return false;
			}
			payload = w[++i];
		}
		if (!known_command(op)) {
			ref_say("cmd 0x%x: unknown command 0x%04x", at, op);
			return false;
		}

		switch (op) {
		case KNPU_OP_STOP:
			if (i + 1 != n) {
				ref_say("cmd 0x%x: OP_STOP is not the last word",
					at);
				return false;
			}
			return true;
		case KNPU_SET_DMA_SRC:
		case KNPU_SET_DMA_DST:
			side = op == KNPU_SET_DMA_SRC ? &src : &dst;
			side->sram = kind == KNPU_SPACE_SRAM;
			side->region = number;
			side->offset = payload;
			if (side->sram ? number : kind != KNPU_SPACE_REGION ||
			    number >= KNPU_NUM_REGIONS) {
				ref_say("cmd 0x%x: bad DMA space 0x%x", at,
					param);
				return false;
			}
			break;
		case KNPU_SET_DMA_LEN:
			len = payload;
			break;
		case KNPU_SET_DMA_COUNT:
			count = param;
			break;
		case KNPU_SET_DMA_STRIDE:
			stride = payload;
			break;
		case KNPU_OP_DMA:
			if (src.sram == dst.sram) {
				ref_say("cmd 0x%x: DMA needs exactly one SRAM side",
					at);
				return false;
			}
			sram = src.sram ? &src : &dst;
			region = src.sram ? &dst : &src;
			total = (uint64_t)len * count;
			if (sram->offset + total > sram_size) {
				ref_say("cmd 0x%x: SRAM range 0x%x+0x%llx is outside the SRAM",
					at, sram->offset,
					(unsigned long long)total);
				return false;
			}
			if (!len || !count)
				break;
			end = region->offset + (uint64_t)(count - 1) * stride +
			      len;
			ref_say("cmd 0x%x: DMA %s region %u, 0x%x..0x%llx",
				at, dst.sram ? "from" : "to", region->region,
				region->offset, (unsigned long long)end);
			break;
		default:
			/* The device checks the other commands' fields */
			break;
		}
	}
	ref_say("command stream does not end with OP_STOP");
	return false;
}

/* xorshift64*, seeded through a splitmix64 step */
static uint64_t rng_state;

static void rng_seed(uint64_t seed)
{
	uint64_t z = seed + 0x9e3779b97f4a7c15ull;

	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
	rng_state = (z ^ (z >> 31)) | 1;
}

static uint32_t rng(void)
{
	rng_state ^= rng_state >> 12;
	rng_state ^= rng_state << 25;
	rng_state ^= rng_state >> 27;
	return (rng_state * 0x2545f4914f6cdd1dull) >> 32;
}

/* A PARAM that is likely to matter: spaces, region numbers, counts */
static uint16_t random_param(void)
{
	static const uint16_t params[] = {
		0, 1, 2, 7, 8, 0xff, KNPU_SPACE(KNPU_SPACE_SRAM, 0),
		KNPU_SPACE(KNPU_SPACE_SRAM, 1), KNPU_SPACE(2, 0), 0xffff,
	};

	return rng() % 2 ? params[rng() % ARRAY_SIZE(params)] : rng();
}

/* A word that is likely to matter to the validator */
static uint32_t random_word(uint32_t sram)
{
	const uint32_t values[] = {
		0, 1, 16, 0x40, 0xff0, sram - 16, sram, sram + 16, 0x7fffffff,
		0x80000000, 0xffffff00, 0xffffffff,
	};
	uint16_t op = known_commands[rng() % ARRAY_SIZE(known_commands)];

	switch (rng() % 5) {
	case 0:
		return rng();
	case 1:
		return values[rng() % ARRAY_SIZE(values)];
	case 2:
		/* Next to a known command: the payload bit or bit 15 flipped */
		op ^= rng() % 2 ? KNPU_CMD_PAYLOAD : 0x8000;
		return KNPU_CMD_WORD(op, random_param());
	default:
		return KNPU_CMD_WORD(op, random_param());
	}
}

/* Half of the edits go to the DMA commands and their payloads */
static unsigned int pick_word(const struct stream *s)
{
	unsigned int dma[ARRAY_SIZE(s->w)], n = 0, i;

	for (i = 0; i < s->n; i++) {
		switch (s->w[i] & KNPU_CMD_MASK) {
		case KNPU_SET_DMA_SRC:
		case KNPU_SET_DMA_DST:
		case KNPU_SET_DMA_LEN:
		case KNPU_SET_DMA_STRIDE:
			if (i + 1 < s->n)
				dma[n++] = i + 1;
			/* fall through */
		case KNPU_SET_DMA_COUNT:
		case KNPU_OP_DMA:
			dma[n++] = i;
			break;
		}
	}
	return n && rng() % 2 ? dma[rng() % n] : rng() % s->n;
}

/*
 * One to three edits: change a word, flip one of its bits, give it another
 * PARAM, drop it, add a word before it or cut the stream after it.
 */
static void mutate(struct stream *s, uint32_t sram)
{
	int edits = 1 + rng() % 3;

	while (edits--) {
		unsigned int at = pick_word(s);

		switch (rng() % 6) {
		case 0:
			s->w[at] = random_word(sram);
			break;
		case 1:
			s->w[at] ^= 1u << (rng() % 32);
			break;
		case 2:
			s->w[at] = KNPU_CMD_WORD(s->w[at] & KNPU_CMD_MASK,
						 random_param());
			break;
		case 3:
			if (s->n > 1) {
				memmove(&s->w[at], &s->w[at + 1],
					(s->n - at - 1) * sizeof(s->w[0]));
				s->n--;
			}
			break;
		case 4:
			memmove(&s->w[at + 1], &s->w[at],
				(s->n - at) * sizeof(s->w[0]));
			s->w[at] = random_word(sram);
			s->n++;
			break;
		default:
			s->n = at + 1;
			break;
		}
	}
}

static void print_lines(const char *title, const char *text)
{
	const char *line, *eol;

	printf("  %s:\n", title);
	for (line = text; *line; line = eol + 1) {
		eol = strchr(line, '\n');
		printf("    %.*s\n", (int)(eol - line), line);
	}
}

static void test_random_streams(int fd)
{
	static const char *const rules[][2] = {
		{ "payload is missing", "payload" },
		{ "unknown command", "unknown" },
		{ "OP_STOP is not the last word", "stop" },
		{ "does not end with OP_STOP", "no stop" },
		{ "exactly one SRAM side", "sides" },
		{ "is outside the SRAM", "SRAM range" },
		{ "bad DMA space", "space" },
	};
	int refused[ARRAY_SIZE(rules)] = { 0 };
	int valid = 0, ranges = 0, wrong = 0, i;
	char summary[160] = "";
	struct stream base, s;
	unsigned int j;
	size_t r;

	conv_stream(&base, true, true);
	for (i = 0; i < RANDOM_STREAMS; i++) {
		const char *got, *p;
		bool ok;
		int ret;

		rng_seed(RANDOM_SEED + i);
		s = base;
		mutate(&s, info.sram_size);
		ok = ref_validate(s.w, s.n, info.sram_size);

		klog_mark();
		ret = cmdstream_create(fd, &s, NULL);
		got = klog_dbg();

		valid += ok;
		for (p = ref_log; (p = strstr(p, "..0x")); p++)
			ranges++;
		for (r = 0; !ok && r < ARRAY_SIZE(rules); r++)
			if (strstr(ref_log, rules[r][0]))
				refused[r]++;

		if (ret == (ok ? 0 : -EINVAL) && !strcmp(got, ref_log))
			continue;
		if (wrong++ >= 3)
			continue;
		printf("random stream %d (seed 0x%llx + %d): kernel %s, "
		       "reference %s\n", i, RANDOM_SEED, i, err_name(ret),
		       ok ? "0" : "-EINVAL");
		printf("  words:");
		for (j = 0; j < s.n; j++)
			printf("%s %08x", j && !(j % 8) ? "\n        " : "",
			       s.w[j]);
		printf("\n");
		print_lines("kernel messages", got);
		print_lines("reference messages", ref_log);
	}

	for (r = 0; r < ARRAY_SIZE(rules); r++)
		snprintf(summary + strlen(summary),
			 sizeof(summary) - strlen(summary), "%s %s %d",
			 r ? "," : "", rules[r][1], refused[r]);
	check(!wrong, "%d random streams (seed 0x%llx): kernel and reference "
	      "%s %d; %d valid, %d DMA ranges; refused for%s", RANDOM_STREAMS,
	      RANDOM_SEED, wrong ? "disagree on" : "agree on",
	      wrong ? wrong : RANDOM_STREAMS, valid, ranges, summary);
}

static void test_submit_checks(int fd)
{
	uint32_t regions[8] = { 0 };
	uint32_t exact, two_d, far, many, last;
	struct stream s;
	struct bo a, b;
	uint32_t at;
	int ret;

	bo_new(fd, 8192, &a);
	bo_new(fd, 4096, &b);

	/* Needs 8192 bytes of region 0 and 16 of region 1 */
	s.n = 0;
	at = dma(&s, REGION(0), 8192 - 64, SRAM, 0, 64, 1, 0);
	dma(&s, SRAM, 0, REGION(1), 0, 16, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_valid(fd, &s, &exact,
		     "stream that needs exactly 8 KiB of region 0",
		     expected("cmd 0x%x: DMA from region 0, 0x1fc0..0x2000",
			      at));

	regions[0] = a.handle;
	regions[1] = b.handle;
	expect_submit(fd, exact, regions, 0,
		      "SUBMIT with buffers that are just large enough", NULL);
	bo_wait(fd, b.handle, 5000000000ll);

	regions[0] = b.handle;
	regions[1] = a.handle;
	expect_submit(fd, exact, regions, -EOVERFLOW,
		      "SUBMIT with a region 0 buffer one page too small",
		      "region 0 needs 0x2000 bytes, buffer has 0x1000");

	/* 16 rows 4 KiB apart need 15 * 4096 + 16 bytes */
	s.n = 0;
	dma(&s, REGION(0), 0, SRAM, 0, 16, 16, 4096);
	cmd(&s, KNPU_OP_STOP, 0);
	ret = cmdstream_create(fd, &s, &two_d);
	check_err(ret, 0, "stream with a 2-D DMA");
	memset(regions, 0, sizeof(regions));
	regions[0] = a.handle;
	expect_submit(fd, two_d, regions, -EOVERFLOW,
		      "SUBMIT where the DMA stride runs past the buffer",
		      "region 0 needs 0xf010 bytes, buffer has 0x2000");

	/* Region ends above 4 GiB, which 32-bit sums would make small */
	s.n = 0;
	at = dma(&s, REGION(0), 0xfffff000, SRAM, 0, 0x2000, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_valid(fd, &s, &far, "stream that reads region 0 at 0xfffff000",
		     expected("cmd 0x%x: DMA from region 0, 0xfffff000..0x100001000",
			      at));
	expect_submit(fd, far, regions, -EOVERFLOW,
		      "SUBMIT of it with an 8 KiB buffer",
		      "region 0 needs 0x100001000 bytes, buffer has 0x2000");
	handle_close(fd, far);

	s.n = 0;
	at = dma(&s, REGION(0), 0, SRAM, 0, 16, 3, 0x80000000);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_valid(fd, &s, &far, "stream with rows 2 GiB apart in region 0",
		     expected("cmd 0x%x: DMA from region 0, 0x0..0x100000010",
			      at));
	expect_submit(fd, far, regions, -EOVERFLOW,
		      "SUBMIT of it with an 8 KiB buffer",
		      "region 0 needs 0x100000010 bytes, buffer has 0x2000");
	handle_close(fd, far);

	/* Of several OP_DMA on one region, the furthest end counts */
	s.n = 0;
	dma(&s, REGION(0), 0, SRAM, 0, 0x100, 1, 0);
	at = dma(&s, REGION(0), 0x1f00, SRAM, 0, 0x100, 1, 0);
	dma(&s, REGION(0), 0x200, SRAM, 0, 0x100, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_valid(fd, &s, &many, "stream with three OP_DMA on region 0",
		     expected("cmd 0x%x: DMA from region 0, 0x1f00..0x2000",
			      at));
	regions[0] = b.handle;
	expect_submit(fd, many, regions, -EOVERFLOW,
		      "SUBMIT of it with a 4 KiB buffer (8 KiB needed)",
		      "region 0 needs 0x2000 bytes, buffer has 0x1000");
	handle_close(fd, many);

	/* The last region */
	s.n = 0;
	at = dma(&s, REGION(7), 4096 - 16, SRAM, 0, 16, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	expect_valid(fd, &s, &last, "stream that reads region 7",
		     expected("cmd 0x%x: DMA from region 7, 0xff0..0x1000",
			      at));
	memset(regions, 0, sizeof(regions));
	expect_submit(fd, last, regions, -EINVAL,
		      "SUBMIT of it without a region 7 buffer",
		      "region 7 is used but has no buffer");
	regions[7] = b.handle;
	expect_submit(fd, last, regions, 0,
		      "SUBMIT of it with a 4 KiB region 7 buffer", NULL);
	check_err(bo_wait(fd, b.handle, 5000000000ll), 0,
		  "the job on region 7 finishes");
	handle_close(fd, last);

	memset(regions, 0, sizeof(regions));
	regions[0] = a.handle;
	expect_submit(fd, exact, regions, -EINVAL,
		      "SUBMIT without a buffer for a used region",
		      "region 1 is used but has no buffer");

	regions[1] = b.handle;
	regions[2] = b.handle;
	expect_submit(fd, exact, regions, -EINVAL,
		      "SUBMIT with a buffer on an unused region",
		      "region 2 is not used by the stream");

	regions[2] = 0;
	regions[1] = a.handle;
	expect_submit(fd, exact, regions, -EINVAL,
		      "SUBMIT with one buffer on two regions",
		      "region 1: buffer already attached");

	regions[1] = 0x7fffffff;
	expect_submit(fd, exact, regions, -ENOENT,
		      "SUBMIT with an unknown region buffer", NULL);

	regions[1] = exact;
	expect_submit(fd, exact, regions, -EINVAL,
		      "SUBMIT with a command stream as a region buffer",
		      "region 1: a command stream is not a region buffer");

	regions[1] = b.handle;
	expect_submit(fd, a.handle, regions, -EINVAL,
		      "SUBMIT of a plain buffer as the command stream",
		      expected("handle %u is not a command stream", a.handle));
	expect_submit(fd, 0x7fffffff, regions, -ENOENT,
		      "SUBMIT of an unknown command stream", NULL);

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
	{
		struct drm_prime_handle prime = {
			.handle = exact,
			.flags = DRM_CLOEXEC,
			.fd = -1,
		};
		int ret = xioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime);

		check_err(ret, -EINVAL, "PRIME_HANDLE_TO_FD of a command stream");
		if (!ret)
			close(prime.fd);
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
	int ret;

	bo_new(fd, 4096, &in);
	bo_new(fd, 4096, &out);
	conv_fill(in.map);
	regions[0] = in.handle;
	regions[1] = out.handle;

	/* OP_CONV reads SRAM that the DMA before it may still be writing */
	conv = conv_stream(&s, false, true);
	snprintf(text, sizeof(text), "job failed: SYNC (7) at command offset 0x%x",
		 conv);
	klog_mark();
	ret = run(fd, &s, regions, 0, out.handle);
	check_err(ret, -EIO, "a stream with a missing DMA wait fails");
	check(klog_count(text) == 1, "the kernel log has \"%s\"", text);

	s.n = 0;
	cmd(&s, KNPU_SET_KERNEL, 0);
	dma(&s, SRAM, 0, REGION(1), 0, 16, 1, 0);
	cmd(&s, KNPU_OP_STOP, 0);
	memset(regions, 0, sizeof(regions));
	regions[1] = out.handle;
	snprintf(text, sizeof(text),
		 "job failed: BAD_PARAM (5) at command offset 0x0");
	klog_mark();
	ret = run(fd, &s, regions, 0, out.handle);
	check_err(ret, -EIO, "a stream with a kernel size of 0 fails");
	check(klog_count(text) == 1, "the kernel log has \"%s\"", text);

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

/*
 * Whether a reader of the buffer has to wait for a job: export it as a
 * dma-buf and take the fences a reader waits for, the write fences, as a
 * sync file. 1 if one is pending, 0 if not, -1 on errors.
 */
static int reader_waits(int fd, uint32_t handle)
{
	struct drm_prime_handle prime = {
		.handle = handle,
		.flags = DRM_CLOEXEC,
		.fd = -1,
	};
	struct dma_buf_export_sync_file sync = {
		.flags = DMA_BUF_SYNC_READ,
		.fd = -1,
	};
	struct pollfd p = { .events = POLLIN };
	int ret = -1;

	if (xioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime))
		return -1;
	if (!xioctl(prime.fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &sync)) {
		p.fd = sync.fd;
		switch (poll(&p, 1, 0)) {
		case 0:
			ret = 1;
			break;
		case 1:
			ret = 0;
			break;
		}
		close(sync.fd);
	}
	close(prime.fd);
	return ret;
}

static void test_timeout(int fd)
{
	uint32_t regions[8] = { 0 };
	char path[300], buf[16];
	int8_t expect[CONV_OUT_BYTES];
	struct stream s;
	struct bo in, out;
	int64_t start, ms;
	uint32_t handle;
	int ret;

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

	klog_mark();
	start = now_ns();
	ret = cmdstream_create(fd, &s, &handle);
	if (!ret) {
		ret = submit(fd, handle, regions, 0);
		handle_close(fd, handle);
	}

	/*
	 * While the job is held, its fence is a write fence on the output
	 * region, which a reader has to wait for, and a read fence on the
	 * input region (docs/design.md 8.4).
	 */
	check(!ret && reader_waits(fd, out.handle) == 1,
	      "a reader of the held job's output buffer has to wait (dma-buf)");
	check(!ret && reader_waits(fd, in.handle) == 0,
	      "a reader of its input buffer does not have to wait");

	if (!ret)
		ret = bo_wait(fd, out.handle, 5000000000ll);
	ms = (now_ns() - start) / 1000000;
	check_err(ret, -EIO, "a job that never starts fails");
	check(ms >= 400 && ms < 5000, "it fails after the 500 ms timeout (%lld ms)",
	      (long long)ms);
	check(klog_count("job timed out") == 1,
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
	klog_fd = open("/dev/kmsg", O_RDONLY | O_NONBLOCK | O_CLOEXEC);
	check(klog_fd >= 0, "the kernel log can be read from /dev/kmsg");
	klog_read();

	test_query(fd);
	test_bo(fd);
	test_dma_job(fd);
	test_conv_job_and_perfmon(fd);

	check(drm_debug_on(), "drm.debug %s turns on the driver messages (0x2)",
	      drm_debug_was);
	test_validator(fd);
	test_random_streams(fd);
	test_submit_checks(fd);
	check(drm_debug_restore(), "drm.debug is %s again", drm_debug_was);

	test_device_errors(fd);
	test_timeout(fd);
	close(fd);
	test_runtime_pm();

	klog_read();
	check(!klog_warnings && !klog_lost,
	      "the kernel log has no warnings (%d records read, %d lost)%s%s",
	      klog_records, klog_lost, klog_warnings ? "; first: " : "",
	      klog_warning);

	if (failures) {
		printf("%d of %d knpu checks failed.\n", failures, checks);
		return 1;
	}
	printf("All %d knpu checks passed.\n", checks);
	return 0;
}
