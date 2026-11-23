/*
 * kn1-npu shared core: executes one job of a command stream
 *
 * The core implements docs/npu-spec.md v0 sections 4 to 11: command
 * decoding, SET state, DMA, the arithmetic of the operations, the
 * asynchronous queues with their synchronization check, errors and the
 * performance counters. Registers, interrupts and timing belong to the
 * caller (the QEMU device or a host tool), which also owns the SRAM and
 * reaches system memory through KnpuMemOps.
 *
 * Implemented so far: OP_STOP, OP_NOP, OP_CONV, OP_AVGPOOL, OP_DMA, the
 * two wait commands and the SET commands they use. OP_ADD and the SET
 * commands only it uses stop the job with KNPU_ERR_UNKNOWN_CMD until it is
 * implemented.
 *
 * Copyright (c) 2026 Kiwon Kwon
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef KNPU_CORE_H
#define KNPU_CORE_H

#include <stdbool.h>
#include <stdint.h>

#include "knpu_cmd.h"

/*
 * Access to the system address space. Each call returns false when the
 * bus transaction fails; the core then stops with KNPU_ERR_BUS.
 */
typedef struct KnpuMemOps {
    bool (*read)(void *opaque, uint64_t addr, void *buf, uint64_t len);
    bool (*write)(void *opaque, uint64_t addr, const void *buf,
                  uint64_t len);
} KnpuMemOps;

typedef struct KnpuPerf {
    uint64_t macs;
    uint64_t dma_rd_bytes;
    uint64_t dma_wr_bytes;
    uint64_t ops;
} KnpuPerf;

typedef struct KnpuJob {
    uint64_t qbase;
    uint32_t qsize;
    uint64_t region_base[KNPU_NUM_REGIONS];
} KnpuJob;

typedef struct KnpuResult {
    uint8_t err_code;       /* KNPU_ERR_NONE when the job reached OP_STOP */
    uint32_t err_offset;    /* ERR_OFFSET, 0 without an error */
    KnpuPerf perf;          /* what the completed commands add (spec 11) */
} KnpuResult;

/*
 * Run the command stream of @job to its end. @sram holds @sram_size bytes
 * and keeps its contents between jobs. The caller adds @res->perf to its
 * counters when PERF_CTRL.ENABLE is set.
 */
void knpu_core_run(uint8_t *sram, uint32_t sram_size,
                   const KnpuMemOps *ops, void *opaque,
                   const KnpuJob *job, KnpuResult *res);

/*
 * The requantization of spec 8.1, exported for tests. @shift is in
 * [-31, 30] and @multiplier in [0, 2^31 - 1].
 */
int32_t knpu_rescale(int32_t x, int32_t multiplier, int shift,
                     int round_mode);

#endif /* KNPU_CORE_H */
