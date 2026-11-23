/*
 * QTest for the Kiwon kn1-npu device on the kn1-evb board
 *
 * Covers the registers (reset values, access rules), the completion and
 * error interrupts as seen by the GIC (SPI 32), jobs that move data
 * through the system bus (above 4 GiB as well), every implemented command,
 * the error paths, the performance counters, CTRL.SOFT_RESET and the
 * sram-size property. The arithmetic itself is checked bit for bit against
 * TensorFlow Lite Micro by the host tests of the qemu-npu-soc repository.
 *
 * Copyright (c) 2026 Kiwon Kwon
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/bswap.h"
#include "libqtest.h"
#include "hw/misc/knpu_cmd.h"

#define NPU_BASE            0x0b000000ULL
#define NPU_SPI             32

#define R_ID                0x000
#define R_VERSION           0x004
#define R_CONFIG            0x008
#define R_SRAM_SIZE         0x00c
#define R_CTRL              0x010
#define R_STATUS            0x014
#define R_ERR_OFFSET        0x018
#define R_IRQ_STATUS        0x020
#define R_IRQ_ENABLE        0x024
#define R_QBASE_LO          0x030
#define R_QBASE_HI          0x034
#define R_QSIZE             0x038
#define R_REGION_LO(n)      (0x100 + 8 * (n))
#define R_REGION_HI(n)      (0x104 + 8 * (n))
#define R_PERF_CTRL         0x200
#define R_PERF_MACS         0x208
#define R_PERF_DMA_RD       0x210
#define R_PERF_DMA_WR       0x218
#define R_PERF_OPS          0x220

#define CTRL_START          1
#define CTRL_SOFT_RESET     2
#define STATUS_BUSY         1
#define STATUS_DONE         2
#define STATUS_ERROR        4
#define IRQ_DONE            1
#define IRQ_ERROR           2
#define PERF_ENABLE         1
#define PERF_CLEAR          2

/* DRAM is 0x4000_0000..0x1_3fff_ffff; region 1 sits above 4 GiB */
#define QUEUE_ADDR          0x40100000ULL
#define REGION0_ADDR        0x40200000ULL
#define REGION1_ADDR        0x100000000ULL
#define UNMAPPED_ADDR       0x200000000ULL

#define SPACE_SRAM          KNPU_SPACE(KNPU_SPACE_SRAM, 0)
#define SPACE_REGION(n)     KNPU_SPACE(KNPU_SPACE_REGION, n)

/* Identity requantization: rescale(x) == x in both rounding modes */
#define RQ_ONE              0x7fffffff

typedef struct Prog {
    uint32_t w[256];
    unsigned n;
} Prog;

/* Append a command; returns its byte offset for ERR_OFFSET checks */
static uint32_t op(Prog *p, uint16_t cmd, uint16_t param)
{
    g_assert_cmpuint(p->n, <, ARRAY_SIZE(p->w));
    p->w[p->n++] = KNPU_CMD_WORD(cmd, param);
    return (p->n - 1) * 4;
}

static uint32_t set(Prog *p, uint16_t cmd, uint16_t param, uint32_t payload)
{
    uint32_t off = op(p, cmd, param);

    g_assert(cmd & KNPU_CMD_PAYLOAD);
    p->w[p->n++] = payload;
    return off;
}

static uint32_t rd(QTestState *qts, uint64_t reg)
{
    return qtest_readl(qts, NPU_BASE + reg);
}

static void wr(QTestState *qts, uint64_t reg, uint32_t val)
{
    qtest_writel(qts, NPU_BASE + reg, val);
}

static uint64_t rd64(QTestState *qts, uint64_t reg)
{
    return rd(qts, reg) | (uint64_t)rd(qts, reg + 4) << 32;
}

static void put_le32(QTestState *qts, uint64_t addr, const uint32_t *v,
                     unsigned n)
{
    for (unsigned i = 0; i < n; i++) {
        uint8_t b[4];

        stl_le_p(b, v[i]);
        qtest_memwrite(qts, addr + 4 * i, b, 4);
    }
}

static QTestState *start_machine(const char *extra)
{
    QTestState *qts = qtest_initf("-machine kn1-evb %s", extra);

    qtest_irq_intercept_in(qts, "/machine/soc/gic");
    return qts;
}

static void set_regions(QTestState *qts)
{
    wr(qts, R_REGION_LO(0), (uint32_t)REGION0_ADDR);
    wr(qts, R_REGION_HI(0), REGION0_ADDR >> 32);
    wr(qts, R_REGION_LO(1), (uint32_t)REGION1_ADDR);
    wr(qts, R_REGION_HI(1), REGION1_ADDR >> 32);
}

/* Load @p as the command stream, start it and wait; returns STATUS */
static uint32_t run(QTestState *qts, const Prog *p)
{
    uint32_t status;
    int i;

    put_le32(qts, QUEUE_ADDR, p->w, p->n);
    wr(qts, R_QBASE_LO, (uint32_t)QUEUE_ADDR);
    wr(qts, R_QBASE_HI, QUEUE_ADDR >> 32);
    wr(qts, R_QSIZE, p->n * 4);
    wr(qts, R_CTRL, CTRL_START);

    /* The bottom half runs from the main loop between qtest commands */
    for (i = 0; i < 1000; i++) {
        status = rd(qts, R_STATUS);
        if (!(status & STATUS_BUSY)) {
            return status;
        }
    }
    g_assert_not_reached();
}

static void assert_error(QTestState *qts, const Prog *p, uint8_t code,
                         uint32_t offset)
{
    uint32_t status = run(qts, p);

    g_assert_cmphex(status, ==, STATUS_ERROR | code << 8);
    g_assert_cmphex(rd(qts, R_ERR_OFFSET), ==, offset);
    g_assert_cmphex(rd(qts, R_IRQ_STATUS) & IRQ_ERROR, ==, IRQ_ERROR);
    wr(qts, R_IRQ_STATUS, IRQ_DONE | IRQ_ERROR);
}

static void test_reset_values(void)
{
    QTestState *qts = start_machine("");
    int n;

    g_assert_cmphex(rd(qts, R_ID), ==, 0x4b4e5055);
    g_assert_cmphex(rd(qts, R_VERSION), ==, 0);
    g_assert_cmphex(rd(qts, R_CONFIG), ==, 0x00220308);
    g_assert_cmphex(rd(qts, R_SRAM_SIZE), ==, 64 * 1024);
    g_assert_cmphex(rd(qts, R_CTRL), ==, 0);
    g_assert_cmphex(rd(qts, R_STATUS), ==, 0);
    g_assert_cmphex(rd(qts, R_ERR_OFFSET), ==, 0);
    g_assert_cmphex(rd(qts, R_IRQ_STATUS), ==, 0);
    g_assert_cmphex(rd(qts, R_IRQ_ENABLE), ==, 0);
    g_assert_cmphex(rd(qts, R_QBASE_LO), ==, 0);
    g_assert_cmphex(rd(qts, R_QBASE_HI), ==, 0);
    g_assert_cmphex(rd(qts, R_QSIZE), ==, 0);
    for (n = 0; n < 8; n++) {
        g_assert_cmphex(rd64(qts, R_REGION_LO(n)), ==, 0);
    }
    g_assert_cmphex(rd(qts, R_PERF_CTRL), ==, 0);
    for (n = 0; n < 4; n++) {
        g_assert_cmphex(rd64(qts, R_PERF_MACS + 8 * n), ==, 0);
    }
    g_assert_false(qtest_get_irq(qts, NPU_SPI));
    qtest_quit(qts);
}

static void test_access_rules(void)
{
    QTestState *qts = start_machine("");

    /* RO registers ignore writes; undefined offsets are RAZ/WI */
    wr(qts, R_ID, 0);
    wr(qts, R_SRAM_SIZE, 0x1000);
    wr(qts, R_STATUS, 0xffffffff);
    wr(qts, R_PERF_MACS, 0x1234);
    g_assert_cmphex(rd(qts, R_ID), ==, 0x4b4e5055);
    g_assert_cmphex(rd(qts, R_SRAM_SIZE), ==, 64 * 1024);
    g_assert_cmphex(rd(qts, R_STATUS), ==, 0);
    g_assert_cmphex(rd(qts, R_PERF_MACS), ==, 0);
    wr(qts, 0x01c, 0xffffffff);
    wr(qts, 0x140, 0xffffffff);
    wr(qts, 0x1fc, 0xffffffff);
    g_assert_cmphex(rd(qts, 0x01c), ==, 0);
    g_assert_cmphex(rd(qts, 0x140), ==, 0);
    g_assert_cmphex(rd(qts, 0x1fc), ==, 0);
    g_assert_cmphex(rd(qts, 0xfffc), ==, 0);

    /* Reserved bits read as zero */
    wr(qts, R_IRQ_ENABLE, 0xffffffff);
    g_assert_cmphex(rd(qts, R_IRQ_ENABLE), ==, IRQ_DONE | IRQ_ERROR);
    wr(qts, R_PERF_CTRL, 0xffffffff);
    g_assert_cmphex(rd(qts, R_PERF_CTRL), ==, PERF_ENABLE);

    /* 64-bit registers are two independent halves */
    wr(qts, R_QBASE_LO, 0x89abcdef);
    wr(qts, R_QBASE_HI, 0x01234567);
    wr(qts, R_REGION_HI(7), 0xfedcba98);
    wr(qts, R_REGION_LO(7), 0x76543210);
    g_assert_cmphex(rd64(qts, R_QBASE_LO), ==, 0x0123456789abcdefULL);
    g_assert_cmphex(rd64(qts, R_REGION_LO(7)), ==, 0xfedcba9876543210ULL);
    qtest_quit(qts);
}

/*
 * A 1x1 convolution from 4x4x1 to 4x4x2 with weights (1, -1), bias (0, 10)
 * and identity requantization: out[y][x] = (in, 10 - in).
 * Region 0: weights at 0x00, bias at 0x10, table at 0x20 (0x30 bytes).
 * Region 1: input at 0x00 (16 bytes), output at 0x40 (32 bytes).
 * SRAM: input 0x000, constants 0x100, output 0x200.
 */
static void conv_program(Prog *p)
{
    set(p, KNPU_SET_DMA_SRC, SPACE_REGION(0), 0);
    set(p, KNPU_SET_DMA_DST, SPACE_SRAM, 0x100);
    set(p, KNPU_SET_DMA_LEN, 0, 0x30);
    op(p, KNPU_OP_DMA, 0);
    set(p, KNPU_SET_DMA_SRC, SPACE_REGION(1), 0);
    set(p, KNPU_SET_DMA_DST, SPACE_SRAM, 0);
    set(p, KNPU_SET_DMA_LEN, 0, 16);
    op(p, KNPU_OP_DMA, 0);
    set(p, KNPU_SET_IFM_BASE, 0, 0);
    set(p, KNPU_SET_IFM_SHAPE, 1, 4 << 16 | 4);
    set(p, KNPU_SET_WEIGHT_BASE, 0, 0x100);
    set(p, KNPU_SET_BIAS_BASE, 0, 0x110);
    set(p, KNPU_SET_RQ_TABLE, 0, 0x120);
    set(p, KNPU_SET_OFM_BASE, 0, 0x200);
    set(p, KNPU_SET_OFM_SHAPE, 2, 4 << 16 | 4);
    op(p, KNPU_OP_DMA_WAIT, 0);
    op(p, KNPU_OP_CONV, 0);
    set(p, KNPU_SET_DMA_SRC, SPACE_SRAM, 0x200);
    set(p, KNPU_SET_DMA_DST, SPACE_REGION(1), 0x40);
    set(p, KNPU_SET_DMA_LEN, 0, 32);
    op(p, KNPU_OP_KERNEL_WAIT, 0);
    op(p, KNPU_OP_DMA, 0);
    op(p, KNPU_OP_STOP, 0);
}

static void conv_data(QTestState *qts)
{
    static const uint32_t consts[12] = {
        [0] = 0x0000ff01,                   /* weights 1, -1 */
        [4] = 0, [5] = 10,                  /* bias */
        [8] = RQ_ONE, [9] = 0, [10] = RQ_ONE, [11] = 0,
    };
    uint8_t in[16];
    int i;

    for (i = 0; i < 16; i++) {
        in[i] = (uint8_t)(i * 9 - 70);
    }
    put_le32(qts, REGION0_ADDR, consts, ARRAY_SIZE(consts));
    qtest_memwrite(qts, REGION1_ADDR, in, sizeof(in));
    qtest_memset(qts, REGION1_ADDR + 0x40, 0x5a, 32);
}

static void check_conv_output(QTestState *qts)
{
    uint8_t out[32];
    int i;

    qtest_memread(qts, REGION1_ADDR + 0x40, out, sizeof(out));
    for (i = 0; i < 16; i++) {
        int8_t x = (int8_t)(i * 9 - 70);

        g_assert_cmpint((int8_t)out[2 * i], ==, x);
        g_assert_cmpint((int8_t)out[2 * i + 1], ==, 10 - x);
    }
}

static void test_conv_job(void)
{
    QTestState *qts = start_machine("");
    Prog p = { .n = 0 };

    conv_program(&p);
    conv_data(qts);
    set_regions(qts);
    wr(qts, R_PERF_CTRL, PERF_ENABLE);

    g_assert_cmphex(run(qts, &p), ==, STATUS_DONE);
    g_assert_cmphex(rd(qts, R_ERR_OFFSET), ==, 0);
    g_assert_cmphex(rd(qts, R_IRQ_STATUS), ==, IRQ_DONE);
    check_conv_output(qts);

    g_assert_cmpuint(rd64(qts, R_PERF_MACS), ==, 4 * 4 * 2);
    g_assert_cmpuint(rd64(qts, R_PERF_DMA_RD), ==, 0x30 + 16);
    g_assert_cmpuint(rd64(qts, R_PERF_DMA_WR), ==, 32);
    g_assert_cmpuint(rd64(qts, R_PERF_OPS), ==, 4);
    qtest_quit(qts);
}

/*
 * The commands the first job does not use: OP_NOP, SET_IFM_ZP,
 * SET_OFM_ZP, SET_KERNEL, SET_STRIDE, SET_PAD, SET_CLAMP,
 * SET_ROUND_MODE, SET_DMA_COUNT and SET_DMA_STRIDE. A 3x3 convolution,
 * stride 2, padding 1, over a 5x5x2 input read with 2-D DMA (rows 16 bytes
 * apart), one output channel, identity requantization.
 */
static void test_all_commands(void)
{
    QTestState *qts = start_machine("");
    Prog p = { .n = 0 };
    const int ifm_zp = 3, ofm_zp = -5, lo = -20, hi = 20;
    int8_t in[5][5][2], w[3][3][2];
    uint8_t consts[64] = { 0 };
    uint8_t row[16], out[9];
    int y, x, c, oy, ox, ky, kx;

    for (y = 0; y < 5; y++) {
        for (x = 0; x < 5; x++) {
            for (c = 0; c < 2; c++) {
                in[y][x][c] = (int8_t)((y * 7 + x * 3 + c * 5) % 23 - 11);
            }
            memcpy(row + 2 * x, in[y][x], 2);
        }
        memset(row + 10, 0xee, 6);
        qtest_memwrite(qts, REGION1_ADDR + 16 * y, row, sizeof(row));
    }
    for (ky = 0; ky < 3; ky++) {
        for (kx = 0; kx < 3; kx++) {
            for (c = 0; c < 2; c++) {
                w[ky][kx][c] = (int8_t)((ky * 3 + kx + c) % 5 - 2);
            }
        }
    }
    memcpy(consts, w, sizeof(w));       /* 18 bytes at 0x00 */
    stl_le_p(consts + 0x20, 7);         /* bias */
    stl_le_p(consts + 0x30, RQ_ONE);    /* table: multiplier, shift 0 */
    qtest_memwrite(qts, REGION0_ADDR, consts, sizeof(consts));

    set(&p, KNPU_SET_DMA_SRC, SPACE_REGION(0), 0);
    set(&p, KNPU_SET_DMA_DST, SPACE_SRAM, 0x400);
    set(&p, KNPU_SET_DMA_LEN, 0, sizeof(consts));
    op(&p, KNPU_OP_DMA, 0);
    set(&p, KNPU_SET_DMA_SRC, SPACE_REGION(1), 0);
    set(&p, KNPU_SET_DMA_DST, SPACE_SRAM, 0);
    set(&p, KNPU_SET_DMA_LEN, 0, 10);
    op(&p, KNPU_SET_DMA_COUNT, 5);
    set(&p, KNPU_SET_DMA_STRIDE, 0, 16);
    op(&p, KNPU_OP_DMA, 0);
    op(&p, KNPU_OP_NOP, 0);
    set(&p, KNPU_SET_IFM_SHAPE, 2, 5 << 16 | 5);
    op(&p, KNPU_SET_IFM_ZP, (uint16_t)ifm_zp);
    set(&p, KNPU_SET_WEIGHT_BASE, 0, 0x400);
    set(&p, KNPU_SET_BIAS_BASE, 0, 0x420);
    set(&p, KNPU_SET_RQ_TABLE, 0, 0x430);
    op(&p, KNPU_SET_KERNEL, 3 << 8 | 3);
    op(&p, KNPU_SET_STRIDE, 2 << 8 | 2);
    op(&p, KNPU_SET_PAD, 0x1111);
    set(&p, KNPU_SET_OFM_BASE, 0, 0x800);
    set(&p, KNPU_SET_OFM_SHAPE, 1, 3 << 16 | 3);
    op(&p, KNPU_SET_OFM_ZP, (uint16_t)ofm_zp);
    op(&p, KNPU_SET_CLAMP, (uint8_t)hi << 8 | (uint8_t)lo);
    op(&p, KNPU_SET_ROUND_MODE, KNPU_ROUND_SINGLE);
    op(&p, KNPU_OP_DMA_WAIT, 0);
    op(&p, KNPU_OP_CONV, 0);
    op(&p, KNPU_OP_KERNEL_WAIT, 0);
    set(&p, KNPU_SET_DMA_SRC, SPACE_SRAM, 0x800);
    set(&p, KNPU_SET_DMA_DST, SPACE_REGION(1), 0x100);
    set(&p, KNPU_SET_DMA_LEN, 0, 9);
    op(&p, KNPU_SET_DMA_COUNT, 1);
    op(&p, KNPU_OP_DMA, 0);
    op(&p, KNPU_OP_STOP, 0);

    set_regions(qts);
    g_assert_cmphex(run(qts, &p), ==, STATUS_DONE);

    qtest_memread(qts, REGION1_ADDR + 0x100, out, sizeof(out));
    for (oy = 0; oy < 3; oy++) {
        for (ox = 0; ox < 3; ox++) {
            int acc = 7;

            for (ky = 0; ky < 3; ky++) {
                for (kx = 0; kx < 3; kx++) {
                    y = oy * 2 - 1 + ky;
                    x = ox * 2 - 1 + kx;
                    if (y < 0 || y >= 5 || x < 0 || x >= 5) {
                        continue;
                    }
                    for (c = 0; c < 2; c++) {
                        acc += w[ky][kx][c] * (in[y][x][c] - ifm_zp);
                    }
                }
            }
            acc = MIN(MAX(acc + ofm_zp, lo), hi);
            g_assert_cmpint((int8_t)out[oy * 3 + ox], ==, acc);
        }
    }
    qtest_quit(qts);
}

/*
 * A 3x3 average pooling, stride 2, padding 1, from 5x5x2 to 3x3x2: the
 * windows at the edges hold four or six input elements, several averages
 * end in .5 and the clamp cuts both ends. The zero points and the rounding
 * mode are set but OP_AVGPOOL does not use them. Region 1: input at 0x00
 * (50 bytes), output at 0x40 (18 bytes). SRAM: input 0x000, output 0x100.
 */
static void test_avgpool_job(void)
{
    QTestState *qts = start_machine("");
    Prog p = { .n = 0 };
    const int lo = -100, hi = 60;
    int8_t in[5][5][2];
    uint8_t out[3][3][2];
    int y, x, c, oy, ox, ky, kx;

    for (y = 0; y < 5; y++) {
        for (x = 0; x < 5; x++) {
            for (c = 0; c < 2; c++) {
                in[y][x][c] = (int8_t)((y * 37 + x * 11 + c * 101) % 255 -
                                       127);
            }
        }
    }
    qtest_memwrite(qts, REGION1_ADDR, in, sizeof(in));
    qtest_memset(qts, REGION1_ADDR + 0x40, 0x5a, sizeof(out));

    set(&p, KNPU_SET_DMA_SRC, SPACE_REGION(1), 0);
    set(&p, KNPU_SET_DMA_DST, SPACE_SRAM, 0);
    set(&p, KNPU_SET_DMA_LEN, 0, sizeof(in));
    op(&p, KNPU_OP_DMA, 0);
    set(&p, KNPU_SET_IFM_BASE, 0, 0);
    set(&p, KNPU_SET_IFM_SHAPE, 2, 5 << 16 | 5);
    op(&p, KNPU_SET_IFM_ZP, 9);
    op(&p, KNPU_SET_KERNEL, 3 << 8 | 3);
    op(&p, KNPU_SET_STRIDE, 2 << 8 | 2);
    op(&p, KNPU_SET_PAD, 0x1111);
    set(&p, KNPU_SET_OFM_BASE, 0, 0x100);
    set(&p, KNPU_SET_OFM_SHAPE, 2, 3 << 16 | 3);
    op(&p, KNPU_SET_OFM_ZP, (uint16_t)-7);
    op(&p, KNPU_SET_CLAMP, (uint8_t)hi << 8 | (uint8_t)lo);
    op(&p, KNPU_SET_ROUND_MODE, KNPU_ROUND_SINGLE);
    op(&p, KNPU_OP_DMA_WAIT, 0);
    op(&p, KNPU_OP_AVGPOOL, 0);
    set(&p, KNPU_SET_DMA_SRC, SPACE_SRAM, 0x100);
    set(&p, KNPU_SET_DMA_DST, SPACE_REGION(1), 0x40);
    set(&p, KNPU_SET_DMA_LEN, 0, sizeof(out));
    op(&p, KNPU_OP_KERNEL_WAIT, 0);
    op(&p, KNPU_OP_DMA, 0);
    op(&p, KNPU_OP_STOP, 0);

    set_regions(qts);
    wr(qts, R_PERF_CTRL, PERF_ENABLE);
    g_assert_cmphex(run(qts, &p), ==, STATUS_DONE);

    qtest_memread(qts, REGION1_ADDR + 0x40, out, sizeof(out));
    for (oy = 0; oy < 3; oy++) {
        for (ox = 0; ox < 3; ox++) {
            for (c = 0; c < 2; c++) {
                int sum = 0, n = 0, avg;

                for (ky = 0; ky < 3; ky++) {
                    for (kx = 0; kx < 3; kx++) {
                        y = oy * 2 - 1 + ky;
                        x = ox * 2 - 1 + kx;
                        if (y >= 0 && y < 5 && x >= 0 && x < 5) {
                            sum += in[y][x][c];
                            n++;
                        }
                    }
                }
                /* Half away from zero (docs/npu-spec.md 8.4) */
                avg = sum > 0 ? (sum + n / 2) / n : (sum - n / 2) / n;
                g_assert_cmpint((int8_t)out[oy][ox][c], ==,
                                MIN(MAX(avg, lo), hi));
            }
        }
    }

    /* One operation without MACs between the two DMAs */
    g_assert_cmpuint(rd64(qts, R_PERF_OPS), ==, 3);
    g_assert_cmpuint(rd64(qts, R_PERF_MACS), ==, 0);
    g_assert_cmpuint(rd64(qts, R_PERF_DMA_RD), ==, sizeof(in));
    g_assert_cmpuint(rd64(qts, R_PERF_DMA_WR), ==, sizeof(out));
    qtest_quit(qts);
}

static void test_interrupts(void)
{
    QTestState *qts = start_machine("");
    Prog good = { .n = 0 }, bad = { .n = 0 };

    op(&good, KNPU_OP_STOP, 0);
    op(&bad, 0x0099, 0);

    /* Masked: the status bit is set, the line stays low */
    g_assert_cmphex(run(qts, &good), ==, STATUS_DONE);
    g_assert_cmphex(rd(qts, R_IRQ_STATUS), ==, IRQ_DONE);
    g_assert_false(qtest_get_irq(qts, NPU_SPI));

    /* Unmasking a pending DONE raises SPI 32; RW1C lowers it */
    wr(qts, R_IRQ_ENABLE, IRQ_DONE);
    g_assert_true(qtest_get_irq(qts, NPU_SPI));
    wr(qts, R_IRQ_STATUS, IRQ_ERROR);
    g_assert_true(qtest_get_irq(qts, NPU_SPI));
    wr(qts, R_IRQ_STATUS, IRQ_DONE);
    g_assert_false(qtest_get_irq(qts, NPU_SPI));
    g_assert_cmphex(rd(qts, R_IRQ_STATUS), ==, 0);

    /* An error job with only DONE enabled does not interrupt */
    g_assert_cmphex(run(qts, &bad), ==,
                    STATUS_ERROR | KNPU_ERR_UNKNOWN_CMD << 8);
    g_assert_cmphex(rd(qts, R_IRQ_STATUS), ==, IRQ_ERROR);
    g_assert_false(qtest_get_irq(qts, NPU_SPI));
    wr(qts, R_IRQ_ENABLE, IRQ_DONE | IRQ_ERROR);
    g_assert_true(qtest_get_irq(qts, NPU_SPI));
    wr(qts, R_IRQ_STATUS, IRQ_ERROR);
    g_assert_false(qtest_get_irq(qts, NPU_SPI));

    /* Completion interrupt of an enabled job; START clears STATUS only */
    g_assert_cmphex(run(qts, &good), ==, STATUS_DONE);
    g_assert_true(qtest_get_irq(qts, NPU_SPI));
    g_assert_cmphex(run(qts, &good), ==, STATUS_DONE);
    g_assert_cmphex(rd(qts, R_IRQ_STATUS), ==, IRQ_DONE);

    /* Soft reset lowers the line */
    wr(qts, R_CTRL, CTRL_SOFT_RESET);
    g_assert_false(qtest_get_irq(qts, NPU_SPI));
    qtest_quit(qts);
}

static void test_errors(void)
{
    QTestState *qts = start_machine("");
    Prog p;
    uint32_t off;

    set_regions(qts);

    /* Unknown command */
    p.n = 0;
    op(&p, KNPU_OP_NOP, 0);
    off = op(&p, 0x0099, 0);
    op(&p, KNPU_OP_STOP, 0);
    assert_error(qts, &p, KNPU_ERR_UNKNOWN_CMD, off);

    /* Commands that run past QSIZE */
    p.n = 0;
    op(&p, KNPU_OP_NOP, 0);
    off = op(&p, KNPU_SET_IFM_BASE, 0);     /* payload word missing */
    assert_error(qts, &p, KNPU_ERR_TRUNCATED, off);
    p.n = 0;
    op(&p, KNPU_OP_NOP, 0);
    op(&p, KNPU_OP_NOP, 0);
    assert_error(qts, &p, KNPU_ERR_NO_STOP, 8);

    /* Empty queue */
    p.n = 0;
    assert_error(qts, &p, KNPU_ERR_BAD_QUEUE, 0);

    /* A field out of range */
    p.n = 0;
    off = op(&p, KNPU_SET_KERNEL, 0x0003);
    assert_error(qts, &p, KNPU_ERR_BAD_PARAM, off);

    /* SRAM out of range */
    p.n = 0;
    set(&p, KNPU_SET_DMA_SRC, SPACE_REGION(1), 0);
    set(&p, KNPU_SET_DMA_DST, SPACE_SRAM, 0xfff0);
    set(&p, KNPU_SET_DMA_LEN, 0, 0x20);
    off = op(&p, KNPU_OP_DMA, 0);
    assert_error(qts, &p, KNPU_ERR_SRAM_RANGE, off);

    /* The convolution job without its OP_DMA_WAIT */
    {
        Prog full = { .n = 0 };
        unsigned i;

        conv_program(&full);
        p.n = 0;
        off = 0;
        for (i = 0; i < full.n; i++) {
            uint16_t cmd = full.w[i] & KNPU_CMD_MASK;

            if (cmd == KNPU_OP_DMA_WAIT) {
                continue;
            }
            if (cmd == KNPU_OP_CONV) {
                off = p.n * 4;
            }
            p.w[p.n++] = full.w[i];
            if (cmd & KNPU_CMD_PAYLOAD) {
                p.w[p.n++] = full.w[++i];
            }
        }
        assert_error(qts, &p, KNPU_ERR_SYNC, off);
    }

    /* DMA to an address with nothing behind it */
    wr(qts, R_REGION_LO(2), (uint32_t)UNMAPPED_ADDR);
    wr(qts, R_REGION_HI(2), UNMAPPED_ADDR >> 32);
    p.n = 0;
    set(&p, KNPU_SET_DMA_SRC, SPACE_REGION(2), 0);
    set(&p, KNPU_SET_DMA_DST, SPACE_SRAM, 0);
    set(&p, KNPU_SET_DMA_LEN, 0, 4);
    off = op(&p, KNPU_OP_DMA, 0);
    op(&p, KNPU_OP_STOP, 0);
    assert_error(qts, &p, KNPU_ERR_BUS, off);

    /* DMA into the NPU's own registers is refused, not re-entered */
    wr(qts, R_REGION_LO(2), (uint32_t)NPU_BASE);
    wr(qts, R_REGION_HI(2), NPU_BASE >> 32);
    p.n = 0;
    set(&p, KNPU_SET_DMA_SRC, SPACE_SRAM, 0);
    set(&p, KNPU_SET_DMA_DST, SPACE_REGION(2), R_CTRL);
    set(&p, KNPU_SET_DMA_LEN, 0, 4);
    off = op(&p, KNPU_OP_DMA, 0);
    op(&p, KNPU_OP_STOP, 0);
    assert_error(qts, &p, KNPU_ERR_BUS, off);

    /* A queue that is not all backed by memory */
    p.n = 0;
    op(&p, KNPU_OP_STOP, 0);
    put_le32(qts, QUEUE_ADDR, p.w, p.n);
    wr(qts, R_QBASE_LO, (uint32_t)UNMAPPED_ADDR);
    wr(qts, R_QBASE_HI, UNMAPPED_ADDR >> 32);
    wr(qts, R_QSIZE, 4);
    wr(qts, R_CTRL, CTRL_START);
    while (rd(qts, R_STATUS) & STATUS_BUSY) {
        /* wait for the bottom half */
    }
    g_assert_cmphex(rd(qts, R_STATUS), ==,
                    STATUS_ERROR | KNPU_ERR_BUS << 8);
    g_assert_cmphex(rd(qts, R_ERR_OFFSET), ==, 0);
    qtest_quit(qts);
}

static void test_perf_counters(void)
{
    QTestState *qts = start_machine("");
    Prog p = { .n = 0 }, bad = { .n = 0 };
    uint32_t off;

    conv_program(&p);
    conv_data(qts);
    set_regions(qts);

    /* Disabled: nothing is counted */
    g_assert_cmphex(run(qts, &p), ==, STATUS_DONE);
    g_assert_cmpuint(rd64(qts, R_PERF_OPS), ==, 0);
    g_assert_cmpuint(rd64(qts, R_PERF_MACS), ==, 0);

    /* Enabled: two jobs add up */
    wr(qts, R_PERF_CTRL, PERF_ENABLE);
    run(qts, &p);
    run(qts, &p);
    g_assert_cmpuint(rd64(qts, R_PERF_MACS), ==, 2 * 32);
    g_assert_cmpuint(rd64(qts, R_PERF_DMA_RD), ==, 2 * 64);
    g_assert_cmpuint(rd64(qts, R_PERF_DMA_WR), ==, 2 * 32);
    g_assert_cmpuint(rd64(qts, R_PERF_OPS), ==, 2 * 4);

    /* A failing job counts the commands before the failure only */
    wr(qts, R_PERF_CTRL, PERF_ENABLE | PERF_CLEAR);
    g_assert_cmpuint(rd64(qts, R_PERF_OPS), ==, 0);
    set(&bad, KNPU_SET_DMA_SRC, SPACE_REGION(1), 0);
    set(&bad, KNPU_SET_DMA_DST, SPACE_SRAM, 0);
    set(&bad, KNPU_SET_DMA_LEN, 0, 16);
    op(&bad, KNPU_OP_DMA, 0);
    set(&bad, KNPU_SET_DMA_DST, SPACE_SRAM, 0xfff8);
    off = op(&bad, KNPU_OP_DMA, 0);
    assert_error(qts, &bad, KNPU_ERR_SRAM_RANGE, off);
    g_assert_cmpuint(rd64(qts, R_PERF_OPS), ==, 1);
    g_assert_cmpuint(rd64(qts, R_PERF_DMA_RD), ==, 16);

    /* CLEAR alone stops counting as well */
    wr(qts, R_PERF_CTRL, PERF_CLEAR);
    g_assert_cmphex(rd(qts, R_PERF_CTRL), ==, 0);
    g_assert_cmpuint(rd64(qts, R_PERF_DMA_RD), ==, 0);
    qtest_quit(qts);
}

static void test_soft_reset(void)
{
    QTestState *qts = start_machine("");
    Prog p = { .n = 0 }, dump = { .n = 0 };
    uint8_t out[16];
    unsigned i;

    conv_program(&p);
    conv_data(qts);
    set_regions(qts);
    wr(qts, R_IRQ_ENABLE, IRQ_DONE);
    wr(qts, R_PERF_CTRL, PERF_ENABLE);
    g_assert_cmphex(run(qts, &p), ==, STATUS_DONE);
    g_assert_true(qtest_get_irq(qts, NPU_SPI));

    wr(qts, R_CTRL, CTRL_SOFT_RESET | CTRL_START);
    g_assert_cmphex(rd(qts, R_STATUS), ==, 0);
    g_assert_cmphex(rd(qts, R_IRQ_STATUS), ==, 0);
    g_assert_cmphex(rd(qts, R_IRQ_ENABLE), ==, 0);
    g_assert_cmphex(rd(qts, R_PERF_CTRL), ==, 0);
    g_assert_cmpuint(rd64(qts, R_PERF_OPS), ==, 0);
    g_assert_cmphex(rd64(qts, R_QBASE_LO), ==, 0);
    g_assert_cmphex(rd(qts, R_QSIZE), ==, 0);
    g_assert_cmphex(rd64(qts, R_REGION_LO(1)), ==, 0);
    g_assert_cmphex(rd(qts, R_SRAM_SIZE), ==, 64 * 1024);
    g_assert_false(qtest_get_irq(qts, NPU_SPI));

    /* The SRAM is cleared: the input left at 0x000 reads back as zeros */
    set_regions(qts);
    set(&dump, KNPU_SET_DMA_SRC, SPACE_SRAM, 0);
    set(&dump, KNPU_SET_DMA_DST, SPACE_REGION(1), 0x80);
    set(&dump, KNPU_SET_DMA_LEN, 0, sizeof(out));
    op(&dump, KNPU_OP_DMA, 0);
    op(&dump, KNPU_OP_STOP, 0);
    qtest_memset(qts, REGION1_ADDR + 0x80, 0x5a, sizeof(out));
    g_assert_cmphex(run(qts, &dump), ==, STATUS_DONE);
    qtest_memread(qts, REGION1_ADDR + 0x80, out, sizeof(out));
    for (i = 0; i < sizeof(out); i++) {
        g_assert_cmphex(out[i], ==, 0);
    }
    qtest_quit(qts);
}

static void test_sram_size(void)
{
    QTestState *qts = start_machine("-global kiwon-kn1-npu.sram-size=32768");
    Prog p = { .n = 0 };
    uint32_t off;

    g_assert_cmphex(rd(qts, R_SRAM_SIZE), ==, 32 * 1024);
    set_regions(qts);
    set(&p, KNPU_SET_DMA_SRC, SPACE_REGION(1), 0);
    set(&p, KNPU_SET_DMA_DST, SPACE_SRAM, 0x7ff0);
    set(&p, KNPU_SET_DMA_LEN, 0, 0x10);
    op(&p, KNPU_OP_DMA, 0);
    set(&p, KNPU_SET_DMA_LEN, 0, 0x11);
    off = op(&p, KNPU_OP_DMA, 0);
    op(&p, KNPU_OP_STOP, 0);
    assert_error(qts, &p, KNPU_ERR_SRAM_RANGE, off);
    qtest_quit(qts);
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);

    qtest_add_func("kn1-npu/reset-values", test_reset_values);
    qtest_add_func("kn1-npu/access-rules", test_access_rules);
    qtest_add_func("kn1-npu/conv-job", test_conv_job);
    qtest_add_func("kn1-npu/all-commands", test_all_commands);
    qtest_add_func("kn1-npu/avgpool-job", test_avgpool_job);
    qtest_add_func("kn1-npu/interrupts", test_interrupts);
    qtest_add_func("kn1-npu/errors", test_errors);
    qtest_add_func("kn1-npu/perf-counters", test_perf_counters);
    qtest_add_func("kn1-npu/soft-reset", test_soft_reset);
    qtest_add_func("kn1-npu/sram-size", test_sram_size);

    return g_test_run();
}
