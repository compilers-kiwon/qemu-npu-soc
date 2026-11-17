/*
 * Kiwon kn1-npu: int8 convolution NPU of the kn1 SoC
 *
 * The programming model is docs/npu-spec.md v0 in the qemu-npu-soc
 * repository. This file holds the registers, the interrupt and the job
 * scheduling; the command stream itself is executed by the shared core
 * (knpu_core.c), which the host tests of the repository use as well.
 *
 * CTRL.START latches the job and schedules a bottom half, which runs the
 * whole command stream and then raises the completion interrupt, so the
 * completion is always asynchronous to the MMIO write. Time is not
 * modelled; the performance counters count work, not cycles.
 *
 * Copyright (c) 2026 Kiwon Kwon
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/log.h"
#include "qemu/main-loop.h"
#include "hw/core/irq.h"
#include "hw/core/qdev-properties.h"
#include "hw/misc/kiwon-kn1-npu.h"
#include "migration/vmstate.h"
#include "knpu_core.h"
#include "trace.h"

/* Register offsets */
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
#define R_REGION_BASE       0x100   /* + 8n: LO, + 8n + 4: HI */
#define R_REGION_BASE_END   (R_REGION_BASE + 8 * KN1_NPU_NUM_REGIONS)
#define R_PERF_CTRL         0x200
#define R_PERF_BASE         0x208   /* + 8n: LO, + 8n + 4: HI */
#define R_PERF_END          (R_PERF_BASE + 8 * KN1_NPU_NUM_PERF)

#define CTRL_START          BIT(0)
#define CTRL_SOFT_RESET     BIT(1)

#define STATUS_BUSY         BIT(0)
#define STATUS_DONE         BIT(1)
#define STATUS_ERROR        BIT(2)
#define STATUS_ERR_SHIFT    8

#define IRQ_DONE            BIT(0)
#define IRQ_ERROR           BIT(1)
#define IRQ_MASK            (IRQ_DONE | IRQ_ERROR)

#define PERF_ENABLE         BIT(0)
#define PERF_CLEAR          BIT(1)

#define CONFIG_VALUE        (KN1_NPU_NUM_REGIONS | BIT(8) | BIT(9) | \
                             KNPU_DMA_DEPTH << 16 | KNPU_KERNEL_DEPTH << 20)

#define SRAM_SIZE_MIN       (4 * KiB)
#define SRAM_SIZE_MAX       (16 * MiB)

static void kn1_npu_update_irq(KiwonKn1NpuState *s)
{
    qemu_set_irq(s->irq, !!(s->irq_status & s->irq_enable));
}

static bool kn1_npu_busy(KiwonKn1NpuState *s)
{
    return s->status & STATUS_BUSY;
}

static bool kn1_npu_mem_read(void *opaque, uint64_t addr, void *buf,
                             uint64_t len)
{
    KiwonKn1NpuState *s = opaque;

    return address_space_read(&s->dma_as, addr, MEMTXATTRS_UNSPECIFIED,
                              buf, len) == MEMTX_OK;
}

static bool kn1_npu_mem_write(void *opaque, uint64_t addr, const void *buf,
                              uint64_t len)
{
    KiwonKn1NpuState *s = opaque;

    return address_space_write(&s->dma_as, addr, MEMTXATTRS_UNSPECIFIED,
                               buf, len) == MEMTX_OK;
}

static const KnpuMemOps kn1_npu_mem_ops = {
    .read = kn1_npu_mem_read,
    .write = kn1_npu_mem_write,
};

static void kn1_npu_run_job(void *opaque)
{
    KiwonKn1NpuState *s = opaque;
    KnpuJob job = {
        .qbase = s->job_qbase,
        .qsize = s->job_qsize,
    };
    KnpuResult res;

    memcpy(job.region_base, s->job_region_base, sizeof(job.region_base));
    knpu_core_run(s->sram, s->sram_size, &kn1_npu_mem_ops, s, &job, &res);

    if (s->perf_ctrl & PERF_ENABLE) {
        s->perf[KN1_NPU_PERF_MACS] += res.perf.macs;
        s->perf[KN1_NPU_PERF_DMA_RD_BYTES] += res.perf.dma_rd_bytes;
        s->perf[KN1_NPU_PERF_DMA_WR_BYTES] += res.perf.dma_wr_bytes;
        s->perf[KN1_NPU_PERF_OPS] += res.perf.ops;
    }
    if (res.err_code == KNPU_ERR_NONE) {
        s->status = STATUS_DONE;
        s->irq_status |= IRQ_DONE;
    } else {
        s->status = STATUS_ERROR | res.err_code << STATUS_ERR_SHIFT;
        s->irq_status |= IRQ_ERROR;
    }
    s->err_offset = res.err_offset;
    trace_kn1_npu_job_done(res.err_code, res.err_offset, res.perf.macs,
                           res.perf.ops);
    kn1_npu_update_irq(s);
}

static void kn1_npu_start(KiwonKn1NpuState *s)
{
    if (kn1_npu_busy(s)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: CTRL.START while a job is running\n", __func__);
        return;
    }
    s->status = STATUS_BUSY;
    s->err_offset = 0;
    s->job_qbase = s->qbase;
    s->job_qsize = s->qsize;
    memcpy(s->job_region_base, s->region_base, sizeof(s->region_base));
    trace_kn1_npu_job_start(s->qbase, s->qsize);
    qemu_bh_schedule(s->bh);
}

/* Hardware reset and CTRL.SOFT_RESET: every register but SRAM_SIZE */
static void kn1_npu_reset(KiwonKn1NpuState *s)
{
    qemu_bh_cancel(s->bh);
    s->status = 0;
    s->err_offset = 0;
    s->irq_status = 0;
    s->irq_enable = 0;
    s->qbase = 0;
    s->qsize = 0;
    memset(s->region_base, 0, sizeof(s->region_base));
    s->perf_ctrl = 0;
    memset(s->perf, 0, sizeof(s->perf));
    s->job_qbase = 0;
    s->job_qsize = 0;
    memset(s->job_region_base, 0, sizeof(s->job_region_base));
    /* The spec leaves the SRAM contents undefined; clear them */
    if (s->sram) {
        memset(s->sram, 0, s->sram_size);
    }
    kn1_npu_update_irq(s);
}

static uint64_t kn1_npu_read(void *opaque, hwaddr offset, unsigned size)
{
    KiwonKn1NpuState *s = opaque;
    uint64_t val = 0;

    switch (offset) {
    case R_ID:
        val = KNPU_ID;
        break;
    case R_VERSION:
        val = KNPU_VERSION_MAJOR << 16 | KNPU_VERSION_MINOR;
        break;
    case R_CONFIG:
        val = CONFIG_VALUE;
        break;
    case R_SRAM_SIZE:
        val = s->sram_size;
        break;
    case R_CTRL:
        break;
    case R_STATUS:
        val = s->status;
        break;
    case R_ERR_OFFSET:
        val = s->err_offset;
        break;
    case R_IRQ_STATUS:
        val = s->irq_status;
        break;
    case R_IRQ_ENABLE:
        val = s->irq_enable;
        break;
    case R_QBASE_LO:
        val = extract64(s->qbase, 0, 32);
        break;
    case R_QBASE_HI:
        val = extract64(s->qbase, 32, 32);
        break;
    case R_QSIZE:
        val = s->qsize;
        break;
    case R_REGION_BASE ... R_REGION_BASE_END - 1:
        val = extract64(s->region_base[(offset - R_REGION_BASE) / 8],
                        (offset & 4) ? 32 : 0, 32);
        break;
    case R_PERF_CTRL:
        val = s->perf_ctrl;
        break;
    case R_PERF_BASE ... R_PERF_END - 1:
        val = extract64(s->perf[(offset - R_PERF_BASE) / 8],
                        (offset & 4) ? 32 : 0, 32);
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: read of undefined offset 0x%" HWADDR_PRIx "\n",
                      __func__, offset);
        break;
    }

    trace_kn1_npu_read(offset, val);
    return val;
}

static void kn1_npu_write(void *opaque, hwaddr offset, uint64_t val,
                          unsigned size)
{
    KiwonKn1NpuState *s = opaque;

    trace_kn1_npu_write(offset, val);

    switch (offset) {
    case R_CTRL:
        if (val & CTRL_SOFT_RESET) {
            kn1_npu_reset(s);
        } else if (val & CTRL_START) {
            kn1_npu_start(s);
        }
        return;
    case R_IRQ_STATUS:
        s->irq_status &= ~val;
        kn1_npu_update_irq(s);
        return;
    case R_IRQ_ENABLE:
        s->irq_enable = val & IRQ_MASK;
        kn1_npu_update_irq(s);
        return;
    case R_QBASE_LO:
    case R_QBASE_HI:
    case R_QSIZE:
    case R_REGION_BASE ... R_REGION_BASE_END - 1:
    case R_PERF_CTRL:
        if (kn1_npu_busy(s)) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "%s: write to 0x%" HWADDR_PRIx " while busy\n",
                          __func__, offset);
            return;
        }
        break;
    case R_ID:
    case R_VERSION:
    case R_CONFIG:
    case R_SRAM_SIZE:
    case R_STATUS:
    case R_ERR_OFFSET:
    case R_PERF_BASE ... R_PERF_END - 1:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: write to read-only offset 0x%" HWADDR_PRIx "\n",
                      __func__, offset);
        return;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: write to undefined offset 0x%" HWADDR_PRIx "\n",
                      __func__, offset);
        return;
    }

    switch (offset) {
    case R_QBASE_LO:
        s->qbase = deposit64(s->qbase, 0, 32, val);
        break;
    case R_QBASE_HI:
        s->qbase = deposit64(s->qbase, 32, 32, val);
        break;
    case R_QSIZE:
        s->qsize = val;
        break;
    case R_REGION_BASE ... R_REGION_BASE_END - 1: {
        uint64_t *base = &s->region_base[(offset - R_REGION_BASE) / 8];

        *base = deposit64(*base, (offset & 4) ? 32 : 0, 32, val);
        break;
    }
    case R_PERF_CTRL:
        if (val & PERF_CLEAR) {
            memset(s->perf, 0, sizeof(s->perf));
        }
        s->perf_ctrl = val & PERF_ENABLE;
        break;
    default:
        g_assert_not_reached();
    }
}

static const MemoryRegionOps kn1_npu_ops = {
    .read = kn1_npu_read,
    .write = kn1_npu_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    /* Other sizes and alignments are bus errors (spec 3) */
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4,
        .unaligned = false,
    },
};

static void kn1_npu_reset_hold(Object *obj, ResetType type)
{
    kn1_npu_reset(KIWON_KN1_NPU(obj));
}

static void kn1_npu_realize(DeviceState *dev, Error **errp)
{
    KiwonKn1NpuState *s = KIWON_KN1_NPU(dev);

    if (!s->dma_mr) {
        error_setg(errp, "%s: the 'memory' link property is not set",
                   TYPE_KIWON_KN1_NPU);
        return;
    }
    if (s->sram_size < SRAM_SIZE_MIN || s->sram_size > SRAM_SIZE_MAX ||
        s->sram_size % (4 * KiB)) {
        error_setg(errp, "%s: sram-size must be a multiple of 4 KiB "
                   "between 4 KiB and 16 MiB (got %" PRIu32 ")",
                   TYPE_KIWON_KN1_NPU, s->sram_size);
        return;
    }

    address_space_init(&s->dma_as, s->dma_mr, "kn1-npu-dma");
    s->sram = g_malloc0(s->sram_size);
    s->bh = aio_bh_new_guarded(qemu_get_aio_context(), kn1_npu_run_job, s,
                               &dev->mem_reentrancy_guard);

    memory_region_init_io(&s->iomem, OBJECT(s), &kn1_npu_ops, s,
                          TYPE_KIWON_KN1_NPU, KN1_NPU_MMIO_SIZE);
    sysbus_init_mmio(SYS_BUS_DEVICE(s), &s->iomem);
    sysbus_init_irq(SYS_BUS_DEVICE(s), &s->irq);
}

static int kn1_npu_post_load(void *opaque, int version_id)
{
    KiwonKn1NpuState *s = opaque;

    /* A job latched before the migration has not run yet */
    if (kn1_npu_busy(s)) {
        qemu_bh_schedule(s->bh);
    }
    return 0;
}

static const VMStateDescription vmstate_kn1_npu = {
    .name = TYPE_KIWON_KN1_NPU,
    .version_id = 1,
    .minimum_version_id = 1,
    .post_load = kn1_npu_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT32(status, KiwonKn1NpuState),
        VMSTATE_UINT32(err_offset, KiwonKn1NpuState),
        VMSTATE_UINT32(irq_status, KiwonKn1NpuState),
        VMSTATE_UINT32(irq_enable, KiwonKn1NpuState),
        VMSTATE_UINT64(qbase, KiwonKn1NpuState),
        VMSTATE_UINT32(qsize, KiwonKn1NpuState),
        VMSTATE_UINT64_ARRAY(region_base, KiwonKn1NpuState,
                             KN1_NPU_NUM_REGIONS),
        VMSTATE_UINT32(perf_ctrl, KiwonKn1NpuState),
        VMSTATE_UINT64_ARRAY(perf, KiwonKn1NpuState, KN1_NPU_NUM_PERF),
        VMSTATE_UINT64(job_qbase, KiwonKn1NpuState),
        VMSTATE_UINT32(job_qsize, KiwonKn1NpuState),
        VMSTATE_UINT64_ARRAY(job_region_base, KiwonKn1NpuState,
                             KN1_NPU_NUM_REGIONS),
        VMSTATE_VBUFFER_UINT32(sram, KiwonKn1NpuState, 0, NULL, sram_size),
        VMSTATE_END_OF_LIST()
    }
};

static const Property kn1_npu_properties[] = {
    DEFINE_PROP_UINT32("sram-size", KiwonKn1NpuState, sram_size,
                       KN1_NPU_SRAM_DEFAULT),
    DEFINE_PROP_LINK("memory", KiwonKn1NpuState, dma_mr,
                     TYPE_MEMORY_REGION, MemoryRegion *),
};

static void kn1_npu_class_init(ObjectClass *oc, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    ResettableClass *rc = RESETTABLE_CLASS(oc);

    dc->realize = kn1_npu_realize;
    dc->desc = "Kiwon kn1 NPU";
    dc->vmsd = &vmstate_kn1_npu;
    rc->phases.hold = kn1_npu_reset_hold;
    device_class_set_props(dc, kn1_npu_properties);
}

static const TypeInfo kn1_npu_types[] = {
    {
        .name = TYPE_KIWON_KN1_NPU,
        .parent = TYPE_SYS_BUS_DEVICE,
        .instance_size = sizeof(KiwonKn1NpuState),
        .class_init = kn1_npu_class_init,
    },
};

DEFINE_TYPES(kn1_npu_types)
