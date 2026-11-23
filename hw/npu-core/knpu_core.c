/*
 * kn1-npu shared core: executes one job of a command stream
 *
 * Section numbers in comments refer to docs/npu-spec.md v0.
 *
 * Copyright (c) 2026 Kiwon Kwon
 *
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "knpu_core.h"

typedef struct KnpuShape {
    uint32_t h, w, c;
} KnpuShape;

/* SET state (spec 5); reset to these defaults at every job start */
typedef struct KnpuState {
    uint32_t ifm_base, ofm_base, weight_base, bias_base, rq_table;
    KnpuShape ifm, ofm;
    int32_t ifm_zp, ofm_zp;
    uint32_t kh, kw, sh, sw;
    uint32_t pad_top, pad_left, pad_bottom, pad_right;
    int32_t clamp_min, clamp_max;
    int round_mode;
    uint8_t dma_src_kind, dma_src_region;
    uint8_t dma_dst_kind, dma_dst_region;
    uint32_t dma_src_off, dma_dst_off;
    uint32_t dma_len, dma_count, dma_stride;
} KnpuState;

/* A byte range [start, end) of the SRAM touched by an operation */
typedef struct KnpuRange {
    uint64_t start, end;
    bool write;
} KnpuRange;

#define KNPU_MAX_RANGES 5

typedef struct KnpuAccess {
    KnpuRange r[KNPU_MAX_RANGES];
    int n;
} KnpuAccess;

/* Operations in progress, oldest first (spec 9) */
typedef struct KnpuQueue {
    KnpuAccess op[KNPU_DMA_DEPTH > KNPU_KERNEL_DEPTH ?
                  KNPU_DMA_DEPTH : KNPU_KERNEL_DEPTH];
    int n, depth;
} KnpuQueue;

typedef struct KnpuCtx {
    uint8_t *sram;
    uint32_t sram_size;
    const KnpuMemOps *ops;
    void *opaque;
    const KnpuJob *job;
    KnpuState st;
    KnpuQueue dma_q, kernel_q;
    KnpuPerf perf;
} KnpuCtx;

static uint32_t ld32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static void state_reset(KnpuState *st)
{
    memset(st, 0, sizeof(*st));
    st->kh = st->kw = 1;
    st->sh = st->sw = 1;
    st->clamp_min = -128;
    st->clamp_max = 127;
    st->round_mode = KNPU_ROUND_DOUBLE;
    st->dma_src_kind = st->dma_dst_kind = KNPU_SPACE_SRAM;
    st->dma_count = 1;
}

/* Spec 8.1 */
int32_t knpu_rescale(int32_t x, int32_t multiplier, int shift,
                     int round_mode)
{
    if (round_mode == KNPU_ROUND_SINGLE) {
        int t = 31 - shift;
        int64_t r = ((int64_t)x * multiplier + ((int64_t)1 << (t - 1))) >> t;
        return (int32_t)(uint32_t)(uint64_t)r;
    } else {
        int ls = shift > 0 ? shift : 0;
        int rs = shift > 0 ? 0 : -shift;
        int32_t x1 = (int32_t)((uint32_t)x << ls);
        int64_t p = (int64_t)x1 * multiplier;
        int64_t nudge = p >= 0 ? ((int64_t)1 << 30) : 1 - ((int64_t)1 << 30);
        /* SaturatingRoundingDoublingHighMul; m >= 0 so it never saturates */
        int32_t h = (int32_t)((p + nudge) / ((int64_t)1 << 31));
        /* RoundingDivideByPOT */
        int32_t mask = (int32_t)(((int64_t)1 << rs) - 1);
        int32_t thr = (mask >> 1) + (h < 0 ? 1 : 0);
        return (h >> rs) + ((h & mask) > thr ? 1 : 0);
    }
}

static int32_t clamp32(int32_t v, int32_t lo, int32_t hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

static bool ranges_conflict(const KnpuAccess *a, const KnpuAccess *b)
{
    for (int i = 0; i < a->n; i++) {
        for (int j = 0; j < b->n; j++) {
            const KnpuRange *ra = &a->r[i], *rb = &b->r[j];
            if ((ra->write || rb->write) &&
                ra->start < rb->end && rb->start < ra->end) {
                return true;
            }
        }
    }
    return false;
}

static void queue_wait(KnpuQueue *q, int remaining)
{
    if (q->n > remaining) {
        memmove(&q->op[0], &q->op[q->n - remaining],
                remaining * sizeof(q->op[0]));
        q->n = remaining;
    }
}

/*
 * Issue an asynchronous operation (spec 9): make room in its own queue,
 * check it against the other queue, then record it. Returns false on a
 * synchronization error; the caller performs the operation otherwise.
 */
static bool issue(KnpuQueue *own, const KnpuQueue *other,
                  const KnpuAccess *acc)
{
    if (own->n == own->depth) {
        queue_wait(own, own->depth - 1);
    }
    for (int i = 0; i < other->n; i++) {
        if (ranges_conflict(acc, &other->op[i])) {
            return false;
        }
    }
    own->op[own->n++] = *acc;
    return true;
}

static void add_range(KnpuAccess *acc, uint64_t start, uint64_t len,
                      bool write)
{
    acc->r[acc->n].start = start;
    acc->r[acc->n].end = start + len;
    acc->r[acc->n].write = write;
    acc->n++;
}

static bool in_sram(const KnpuCtx *c, const KnpuRange *r)
{
    return r->end <= c->sram_size;
}

static int exec_dma(KnpuCtx *c)
{
    const KnpuState *st = &c->st;
    bool to_sram;
    uint8_t region;
    uint32_t sram_off, region_off;
    uint64_t total = (uint64_t)st->dma_len * st->dma_count;
    KnpuAccess acc = { .n = 0 };

    if (st->dma_src_kind == KNPU_SPACE_REGION &&
        st->dma_dst_kind == KNPU_SPACE_SRAM) {
        to_sram = true;
        region = st->dma_src_region;
        region_off = st->dma_src_off;
        sram_off = st->dma_dst_off;
    } else if (st->dma_src_kind == KNPU_SPACE_SRAM &&
               st->dma_dst_kind == KNPU_SPACE_REGION) {
        to_sram = false;
        region = st->dma_dst_region;
        region_off = st->dma_dst_off;
        sram_off = st->dma_src_off;
    } else {
        return KNPU_ERR_BAD_PARAM;
    }

    add_range(&acc, sram_off, total, to_sram);
    if (!in_sram(c, &acc.r[0])) {
        return KNPU_ERR_SRAM_RANGE;
    }
    if (!issue(&c->dma_q, &c->kernel_q, &acc)) {
        return KNPU_ERR_SYNC;
    }

    for (uint32_t row = 0; row < st->dma_count; row++) {
        uint64_t addr = c->job->region_base[region] + region_off +
                        (uint64_t)row * st->dma_stride;
        uint8_t *p = c->sram + sram_off + (uint64_t)row * st->dma_len;
        bool ok = to_sram ? c->ops->read(c->opaque, addr, p, st->dma_len)
                          : c->ops->write(c->opaque, addr, p, st->dma_len);
        if (!ok) {
            return KNPU_ERR_BUS;
        }
    }

    if (to_sram) {
        c->perf.dma_rd_bytes += total;
    } else {
        c->perf.dma_wr_bytes += total;
    }
    c->perf.ops++;
    return KNPU_ERR_NONE;
}

/* Shape rule of the window operations (spec 6.4) */
static bool window_shape_ok(const KnpuState *st)
{
    if (!st->ifm.c || !st->ofm.c) {
        return false;
    }
    if (st->pad_top >= st->kh || st->pad_bottom >= st->kh ||
        st->pad_left >= st->kw || st->pad_right >= st->kw) {
        return false;
    }

    uint32_t ph = st->ifm.h + st->pad_top + st->pad_bottom;
    uint32_t pw = st->ifm.w + st->pad_left + st->pad_right;
    if (ph < st->kh || pw < st->kw) {
        return false;
    }
    return st->ofm.h == (ph - st->kh) / st->sh + 1 &&
           st->ofm.w == (pw - st->kw) / st->sw + 1;
}

/*
 * SRAM range and overlap checks of an operation (spec 6.4 steps 2 and 3).
 * The OFM write is the last range of @acc and must not overlap the reads.
 */
static int check_ranges(const KnpuCtx *c, const KnpuAccess *acc)
{
    const KnpuRange *o = &acc->r[acc->n - 1];

    for (int i = 0; i < acc->n; i++) {
        if (!in_sram(c, &acc->r[i])) {
            return KNPU_ERR_SRAM_RANGE;
        }
    }
    for (int i = 0; i < acc->n - 1; i++) {
        const KnpuRange *r = &acc->r[i];
        if (o->start < r->end && r->start < o->end) {
            return KNPU_ERR_BAD_PARAM;
        }
    }
    return KNPU_ERR_NONE;
}

static int exec_conv(KnpuCtx *c)
{
    const KnpuState *st = &c->st;
    const KnpuShape *in = &st->ifm, *out = &st->ofm;
    KnpuAccess acc = { .n = 0 };

    if (!window_shape_ok(st)) {
        return KNPU_ERR_BAD_PARAM;
    }

    /* Reads first, the OFM write last (spec 6.4 table) */
    add_range(&acc, st->ifm_base, (uint64_t)in->h * in->w * in->c, false);
    add_range(&acc, st->weight_base,
              (uint64_t)out->c * st->kh * st->kw * in->c, false);
    add_range(&acc, st->bias_base, 4 * (uint64_t)out->c, false);
    add_range(&acc, st->rq_table, 8 * (uint64_t)out->c, false);
    add_range(&acc, st->ofm_base, (uint64_t)out->h * out->w * out->c, true);

    int err = check_ranges(c, &acc);
    if (err != KNPU_ERR_NONE) {
        return err;
    }

    const uint8_t *rq = c->sram + st->rq_table;
    for (uint32_t oc = 0; oc < out->c; oc++) {
        int32_t m = (int32_t)ld32(rq + 8 * oc);
        int32_t s = (int32_t)ld32(rq + 8 * oc + 4);
        if (m < 0 || s < -31 || s > 30) {
            return KNPU_ERR_BAD_PARAM;
        }
    }

    if (!issue(&c->kernel_q, &c->dma_q, &acc)) {
        return KNPU_ERR_SYNC;
    }

    /* Spec 8.2, the same loop as TFLite Micro ConvPerChannel */
    const int8_t *ifm = (const int8_t *)(c->sram + st->ifm_base);
    const int8_t *wts = (const int8_t *)(c->sram + st->weight_base);
    const uint8_t *bias = c->sram + st->bias_base;
    int8_t *ofm = (int8_t *)(c->sram + st->ofm_base);

    for (uint32_t oy = 0; oy < out->h; oy++) {
        for (uint32_t ox = 0; ox < out->w; ox++) {
            for (uint32_t oc = 0; oc < out->c; oc++) {
                uint32_t acc32 = 0;     /* int32 with wrap-around */
                for (uint32_t ky = 0; ky < st->kh; ky++) {
                    int64_t iy = (int64_t)oy * st->sh - st->pad_top + ky;
                    if (iy < 0 || iy >= in->h) {
                        continue;
                    }
                    for (uint32_t kx = 0; kx < st->kw; kx++) {
                        int64_t ix = (int64_t)ox * st->sw - st->pad_left + kx;
                        if (ix < 0 || ix >= in->w) {
                            continue;
                        }
                        const int8_t *x = ifm + ((uint64_t)iy * in->w + ix) *
                                                in->c;
                        const int8_t *w = wts +
                            (((uint64_t)oc * st->kh + ky) * st->kw + kx) *
                            in->c;
                        for (uint32_t ic = 0; ic < in->c; ic++) {
                            acc32 += (uint32_t)(w[ic] * (x[ic] - st->ifm_zp));
                        }
                    }
                }
                acc32 += ld32(bias + 4 * oc);
                int32_t v = knpu_rescale((int32_t)acc32,
                                         (int32_t)ld32(rq + 8 * oc),
                                         (int32_t)ld32(rq + 8 * oc + 4),
                                         st->round_mode);
                v = (int32_t)((uint32_t)v + (uint32_t)st->ofm_zp);
                ofm[((uint64_t)oy * out->w + ox) * out->c + oc] =
                    (int8_t)clamp32(v, st->clamp_min, st->clamp_max);
            }
        }
    }

    c->perf.macs += (uint64_t)out->h * out->w * out->c *
                    st->kh * st->kw * in->c;
    c->perf.ops++;
    return KNPU_ERR_NONE;
}

static int exec_avgpool(KnpuCtx *c)
{
    const KnpuState *st = &c->st;
    const KnpuShape *in = &st->ifm, *out = &st->ofm;
    KnpuAccess acc = { .n = 0 };

    if (!window_shape_ok(st) || out->c != in->c) {
        return KNPU_ERR_BAD_PARAM;
    }

    add_range(&acc, st->ifm_base, (uint64_t)in->h * in->w * in->c, false);
    add_range(&acc, st->ofm_base, (uint64_t)out->h * out->w * out->c, true);

    int err = check_ranges(c, &acc);
    if (err != KNPU_ERR_NONE) {
        return err;
    }
    if (!issue(&c->kernel_q, &c->dma_q, &acc)) {
        return KNPU_ERR_SYNC;
    }

    /*
     * Spec 8.4, the same loop as TFLite Micro AveragePool (int8). The zero
     * points, the scales and ROUND_MODE are not used. Every window holds
     * at least one input element (spec 6.4), so n >= 1.
     */
    const int8_t *ifm = (const int8_t *)(c->sram + st->ifm_base);
    int8_t *ofm = (int8_t *)(c->sram + st->ofm_base);

    for (uint32_t oy = 0; oy < out->h; oy++) {
        for (uint32_t ox = 0; ox < out->w; ox++) {
            for (uint32_t ch = 0; ch < out->c; ch++) {
                int32_t sum = 0, n = 0;
                for (uint32_t ky = 0; ky < st->kh; ky++) {
                    int64_t iy = (int64_t)oy * st->sh - st->pad_top + ky;
                    if (iy < 0 || iy >= in->h) {
                        continue;
                    }
                    for (uint32_t kx = 0; kx < st->kw; kx++) {
                        int64_t ix = (int64_t)ox * st->sw - st->pad_left + kx;
                        if (ix < 0 || ix >= in->w) {
                            continue;
                        }
                        sum += ifm[((uint64_t)iy * in->w + ix) * in->c + ch];
                        n++;
                    }
                }
                /* Rounds half away from zero */
                int32_t avg = sum > 0 ? (sum + n / 2) / n : (sum - n / 2) / n;
                ofm[((uint64_t)oy * out->w + ox) * out->c + ch] =
                    (int8_t)clamp32(avg, st->clamp_min, st->clamp_max);
            }
        }
    }

    c->perf.ops++;
    return KNPU_ERR_NONE;
}

static bool zp_ok(uint16_t param)
{
    int16_t v = (int16_t)param;
    return v >= -128 && v <= 127;
}

static bool space_ok(uint16_t param)
{
    uint8_t kind = param >> 8, region = param & 0xFF;
    return (kind == KNPU_SPACE_REGION && region < KNPU_NUM_REGIONS) ||
           (kind == KNPU_SPACE_SRAM && region == 0);
}

/*
 * Execute one command. Returns a KNPU_ERR_* code; sets *stop when the
 * command ends the job.
 */
static int exec_cmd(KnpuCtx *c, uint16_t cmd, uint16_t param,
                    uint32_t payload, bool *stop)
{
    KnpuState *st = &c->st;

    switch (cmd) {
    case KNPU_OP_STOP:
    case KNPU_OP_NOP:
    case KNPU_OP_CONV:
    case KNPU_OP_AVGPOOL:
    case KNPU_OP_DMA:
        if (param) {
            return KNPU_ERR_BAD_PARAM;
        }
        break;
    case KNPU_OP_DMA_WAIT:
    case KNPU_OP_KERNEL_WAIT:
        if (param & 0xFFF0) {
            return KNPU_ERR_BAD_PARAM;
        }
        break;
    case KNPU_SET_IFM_BASE:
    case KNPU_SET_OFM_BASE:
    case KNPU_SET_WEIGHT_BASE:
    case KNPU_SET_BIAS_BASE:
    case KNPU_SET_RQ_TABLE:
    case KNPU_SET_DMA_LEN:
    case KNPU_SET_DMA_STRIDE:
        if (param) {
            return KNPU_ERR_BAD_PARAM;
        }
        break;
    default:
        break;
    }

    switch (cmd) {
    case KNPU_OP_STOP:
        c->dma_q.n = 0;
        c->kernel_q.n = 0;
        *stop = true;
        return KNPU_ERR_NONE;
    case KNPU_OP_NOP:
        return KNPU_ERR_NONE;
    case KNPU_OP_CONV:
        return exec_conv(c);
    case KNPU_OP_AVGPOOL:
        return exec_avgpool(c);
    case KNPU_OP_DMA:
        return exec_dma(c);
    case KNPU_OP_DMA_WAIT:
        queue_wait(&c->dma_q, param);
        return KNPU_ERR_NONE;
    case KNPU_OP_KERNEL_WAIT:
        queue_wait(&c->kernel_q, param);
        return KNPU_ERR_NONE;

    case KNPU_SET_IFM_ZP:
    case KNPU_SET_OFM_ZP:
        if (!zp_ok(param)) {
            return KNPU_ERR_BAD_PARAM;
        }
        *(cmd == KNPU_SET_IFM_ZP ? &st->ifm_zp : &st->ofm_zp) =
            (int16_t)param;
        return KNPU_ERR_NONE;
    case KNPU_SET_KERNEL:
    case KNPU_SET_STRIDE:
        if (!(param >> 8) || !(param & 0xFF)) {
            return KNPU_ERR_BAD_PARAM;
        }
        if (cmd == KNPU_SET_KERNEL) {
            st->kh = param >> 8;
            st->kw = param & 0xFF;
        } else {
            st->sh = param >> 8;
            st->sw = param & 0xFF;
        }
        return KNPU_ERR_NONE;
    case KNPU_SET_PAD:
        st->pad_top = param & 0xF;
        st->pad_left = (param >> 4) & 0xF;
        st->pad_bottom = (param >> 8) & 0xF;
        st->pad_right = (param >> 12) & 0xF;
        return KNPU_ERR_NONE;
    case KNPU_SET_CLAMP: {
        int8_t lo = (int8_t)(param & 0xFF), hi = (int8_t)(param >> 8);
        if (lo > hi) {
            return KNPU_ERR_BAD_PARAM;
        }
        st->clamp_min = lo;
        st->clamp_max = hi;
        return KNPU_ERR_NONE;
    }
    case KNPU_SET_ROUND_MODE:
        if (param != KNPU_ROUND_DOUBLE && param != KNPU_ROUND_SINGLE) {
            return KNPU_ERR_BAD_PARAM;
        }
        st->round_mode = param;
        return KNPU_ERR_NONE;
    case KNPU_SET_DMA_COUNT:
        if (!param) {
            return KNPU_ERR_BAD_PARAM;
        }
        st->dma_count = param;
        return KNPU_ERR_NONE;

    case KNPU_SET_IFM_BASE:
        st->ifm_base = payload;
        return KNPU_ERR_NONE;
    case KNPU_SET_OFM_BASE:
        st->ofm_base = payload;
        return KNPU_ERR_NONE;
    case KNPU_SET_WEIGHT_BASE:
        st->weight_base = payload;
        return KNPU_ERR_NONE;
    case KNPU_SET_BIAS_BASE:
    case KNPU_SET_RQ_TABLE:
        if (payload & 3) {
            return KNPU_ERR_BAD_PARAM;
        }
        *(cmd == KNPU_SET_BIAS_BASE ? &st->bias_base : &st->rq_table) =
            payload;
        return KNPU_ERR_NONE;
    case KNPU_SET_IFM_SHAPE:
    case KNPU_SET_OFM_SHAPE: {
        KnpuShape s = { payload >> 16, payload & 0xFFFF, param };
        if (!s.h || !s.w || !s.c) {
            return KNPU_ERR_BAD_PARAM;
        }
        *(cmd == KNPU_SET_IFM_SHAPE ? &st->ifm : &st->ofm) = s;
        return KNPU_ERR_NONE;
    }
    case KNPU_SET_DMA_SRC:
    case KNPU_SET_DMA_DST:
        if (!space_ok(param)) {
            return KNPU_ERR_BAD_PARAM;
        }
        if (cmd == KNPU_SET_DMA_SRC) {
            st->dma_src_kind = param >> 8;
            st->dma_src_region = param & 0xFF;
            st->dma_src_off = payload;
        } else {
            st->dma_dst_kind = param >> 8;
            st->dma_dst_region = param & 0xFF;
            st->dma_dst_off = payload;
        }
        return KNPU_ERR_NONE;
    case KNPU_SET_DMA_LEN:
        if (!payload) {
            return KNPU_ERR_BAD_PARAM;
        }
        st->dma_len = payload;
        return KNPU_ERR_NONE;
    case KNPU_SET_DMA_STRIDE:
        st->dma_stride = payload;
        return KNPU_ERR_NONE;

    default:
        /*
         * Unknown commands, and for now also OP_ADD and the SET commands
         * only it uses (IFM2_BASE, IFM2_ZP, *_SCALE).
         */
        return KNPU_ERR_UNKNOWN_CMD;
    }
}

static bool cmd_known(uint16_t cmd)
{
    switch (cmd) {
    case KNPU_OP_STOP: case KNPU_OP_NOP: case KNPU_OP_CONV:
    case KNPU_OP_AVGPOOL: case KNPU_OP_DMA:
    case KNPU_OP_DMA_WAIT: case KNPU_OP_KERNEL_WAIT:
    case KNPU_SET_IFM_ZP: case KNPU_SET_OFM_ZP: case KNPU_SET_KERNEL:
    case KNPU_SET_STRIDE: case KNPU_SET_PAD: case KNPU_SET_CLAMP:
    case KNPU_SET_ROUND_MODE: case KNPU_SET_DMA_COUNT:
    case KNPU_SET_IFM_BASE: case KNPU_SET_OFM_BASE:
    case KNPU_SET_WEIGHT_BASE: case KNPU_SET_BIAS_BASE:
    case KNPU_SET_RQ_TABLE: case KNPU_SET_IFM_SHAPE: case KNPU_SET_OFM_SHAPE:
    case KNPU_SET_DMA_SRC: case KNPU_SET_DMA_DST: case KNPU_SET_DMA_LEN:
    case KNPU_SET_DMA_STRIDE:
        return true;
    default:
        return false;
    }
}

void knpu_core_run(uint8_t *sram, uint32_t sram_size,
                   const KnpuMemOps *ops, void *opaque,
                   const KnpuJob *job, KnpuResult *res)
{
    KnpuCtx c = {
        .sram = sram, .sram_size = sram_size, .ops = ops, .opaque = opaque,
        .job = job,
        .dma_q = { .depth = KNPU_DMA_DEPTH },
        .kernel_q = { .depth = KNPU_KERNEL_DEPTH },
    };
    uint32_t off = 0;
    int err = KNPU_ERR_NONE;
    bool stop = false;

    state_reset(&c.st);

    /* Spec 3.9 */
    if ((job->qbase & 3) || !job->qsize || (job->qsize & 3)) {
        err = KNPU_ERR_BAD_QUEUE;
        goto out;
    }

    while (!stop) {
        uint8_t buf[4];
        uint32_t word, payload = 0;
        uint16_t cmd;

        /* Spec 10: fetch, UNKNOWN_CMD, TRUNCATED, payload fetch, command */
        if (off >= job->qsize) {
            err = KNPU_ERR_NO_STOP;
            off = job->qsize;
            break;
        }
        if (!ops->read(opaque, job->qbase + off, buf, 4)) {
            err = KNPU_ERR_BUS;
            break;
        }
        word = ld32(buf);
        cmd = word & KNPU_CMD_MASK;
        if (!cmd_known(cmd)) {
            err = KNPU_ERR_UNKNOWN_CMD;
            break;
        }
        if (cmd & KNPU_CMD_PAYLOAD) {
            if (job->qsize - off < 8) {
                err = KNPU_ERR_TRUNCATED;
                break;
            }
            if (!ops->read(opaque, job->qbase + off + 4, buf, 4)) {
                err = KNPU_ERR_BUS;
                break;
            }
            payload = ld32(buf);
        }
        err = exec_cmd(&c, cmd, KNPU_CMD_PARAM(word), payload, &stop);
        if (err != KNPU_ERR_NONE) {
            break;
        }
        off += (cmd & KNPU_CMD_PAYLOAD) ? 8 : 4;
    }

out:
    res->err_code = err;
    res->err_offset = err == KNPU_ERR_NONE ? 0 :
                      err == KNPU_ERR_BAD_QUEUE ? 0 : off;
    res->perf = c.perf;
}
