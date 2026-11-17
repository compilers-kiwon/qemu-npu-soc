/*
 * Kiwon kn1-npu: int8 convolution NPU of the kn1 SoC
 *
 * Copyright (c) 2026 Kiwon Kwon
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef HW_MISC_KIWON_KN1_NPU_H
#define HW_MISC_KIWON_KN1_NPU_H

#include "hw/core/sysbus.h"
#include "qemu/units.h"
#include "qom/object.h"
#include "system/memory.h"

#define TYPE_KIWON_KN1_NPU "kiwon-kn1-npu"
OBJECT_DECLARE_SIMPLE_TYPE(KiwonKn1NpuState, KIWON_KN1_NPU)

#define KN1_NPU_MMIO_SIZE       0x10000
#define KN1_NPU_NUM_REGIONS     8
#define KN1_NPU_SRAM_DEFAULT    (64 * KiB)

enum {
    KN1_NPU_PERF_MACS,
    KN1_NPU_PERF_DMA_RD_BYTES,
    KN1_NPU_PERF_DMA_WR_BYTES,
    KN1_NPU_PERF_OPS,
    KN1_NPU_NUM_PERF,
};

struct KiwonKn1NpuState {
    SysBusDevice parent_obj;

    MemoryRegion iomem;
    qemu_irq irq;
    QEMUBH *bh;

    /* Properties */
    MemoryRegion *dma_mr;       /* "memory": the bus the NPU masters */
    uint32_t sram_size;         /* "sram-size" */

    AddressSpace dma_as;
    uint8_t *sram;

    /* Registers (docs/npu-spec.md 3) */
    uint32_t status;
    uint32_t err_offset;
    uint32_t irq_status;
    uint32_t irq_enable;
    uint64_t qbase;
    uint32_t qsize;
    uint64_t region_base[KN1_NPU_NUM_REGIONS];
    uint32_t perf_ctrl;
    uint64_t perf[KN1_NPU_NUM_PERF];

    /* The job latched by CTRL.START, run by the bottom half */
    uint64_t job_qbase;
    uint32_t job_qsize;
    uint64_t job_region_base[KN1_NPU_NUM_REGIONS];
};

#endif
