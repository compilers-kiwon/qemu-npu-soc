/*
 * kn1-npu command stream and register encoding (docs/npu-spec.md v0)
 *
 * Copyright (c) 2026 Kiwon Kwon
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef KNPU_CMD_H
#define KNPU_CMD_H

/* Programming model version (VERSION register) */
#define KNPU_ID                 0x4B4E5055u     /* "KNPU" */
#define KNPU_VERSION_MAJOR      0
#define KNPU_VERSION_MINOR      0

#define KNPU_NUM_REGIONS        8
#define KNPU_DMA_DEPTH          2
#define KNPU_KERNEL_DEPTH       2

/* Command word: PARAM[31:16], 0[15], L[14], OPCODE[13:0] (spec 4) */
#define KNPU_CMD_MASK           0xFFFFu
#define KNPU_CMD_PAYLOAD        0x4000u
#define KNPU_CMD_PARAM(w)       ((uint16_t)((w) >> 16))
#define KNPU_CMD_WORD(cmd, param) \
    (((uint32_t)(uint16_t)(param) << 16) | (uint32_t)(cmd))

/* Commands without a payload word (spec 6.1) */
#define KNPU_OP_STOP            0x0000
#define KNPU_OP_NOP             0x0001
#define KNPU_OP_CONV            0x0002
#define KNPU_OP_ADD             0x0003
#define KNPU_OP_AVGPOOL         0x0004
#define KNPU_OP_DMA             0x0010
#define KNPU_OP_DMA_WAIT        0x0011
#define KNPU_OP_KERNEL_WAIT     0x0012
#define KNPU_SET_IFM_ZP         0x0100
#define KNPU_SET_IFM2_ZP        0x0101
#define KNPU_SET_OFM_ZP         0x0102
#define KNPU_SET_KERNEL         0x0103
#define KNPU_SET_STRIDE         0x0104
#define KNPU_SET_PAD            0x0105
#define KNPU_SET_CLAMP          0x0106
#define KNPU_SET_ROUND_MODE     0x0107
#define KNPU_SET_DMA_COUNT      0x0108

/* Commands followed by a payload word */
#define KNPU_SET_IFM_BASE       0x4000
#define KNPU_SET_IFM2_BASE      0x4001
#define KNPU_SET_OFM_BASE       0x4002
#define KNPU_SET_WEIGHT_BASE    0x4003
#define KNPU_SET_BIAS_BASE      0x4004
#define KNPU_SET_RQ_TABLE       0x4005
#define KNPU_SET_IFM_SHAPE      0x4006
#define KNPU_SET_OFM_SHAPE      0x4007
#define KNPU_SET_IFM_SCALE      0x4008
#define KNPU_SET_IFM2_SCALE     0x4009
#define KNPU_SET_OFM_SCALE      0x400A
#define KNPU_SET_DMA_SRC        0x4010
#define KNPU_SET_DMA_DST        0x4011
#define KNPU_SET_DMA_LEN        0x4012
#define KNPU_SET_DMA_STRIDE     0x4013

/* SET_DMA_SRC / SET_DMA_DST space: PARAM[15:8] kind, PARAM[7:0] region */
#define KNPU_SPACE_REGION       0
#define KNPU_SPACE_SRAM         1
#define KNPU_SPACE(kind, region) (((kind) << 8) | (region))

/* SET_ROUND_MODE */
#define KNPU_ROUND_DOUBLE       0
#define KNPU_ROUND_SINGLE       1

/* STATUS.ERR_CODE (spec 10) */
#define KNPU_ERR_NONE           0x00
#define KNPU_ERR_UNKNOWN_CMD    0x01
#define KNPU_ERR_TRUNCATED      0x02
#define KNPU_ERR_NO_STOP        0x03
#define KNPU_ERR_BAD_QUEUE      0x04
#define KNPU_ERR_BAD_PARAM      0x05
#define KNPU_ERR_SRAM_RANGE     0x06
#define KNPU_ERR_SYNC           0x07
#define KNPU_ERR_BUS            0x08

#endif /* KNPU_CMD_H */
