// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: knpu-translate --knpu-to-model %s | od -A d -t x4 -v -w4 | FileCheck %s

// OP_ADD with its SET commands (docs/npu-spec.md 6.1, 8.3): the IFM, IFM2
// and OFM bases, one shape set as both IFM_SHAPE and OFM_SHAPE, the three
// zero points and scales, clamp and ROUND_MODE, but no kernel, stride or
// padding, which the operation does not use. A scale is its shift in PARAM
// (int16) and its multiplier in the payload. The first addition adds the
// input to itself; the second adds the first one's output to the input in
// place (OFM_BASE = IFM2_BASE) and sets only what changed. Three sections
// (no constant region): CMDS at 80 (164 bytes), REGIONS at 256, TENSORS at
// 272.

module attributes {knpu.sram_size = 65536 : i64} {
  knpu.region 1 "arena" size 32
  knpu.io "input" region 1 [0] shape [1, 2, 2, 4] scale 1.0 zero_point -128
  knpu.io "output" region 1 [16] shape [1, 2, 2, 4] scale 1.0 zero_point 5
  func.func @knpu_model() {
    knpu.dma "load" region 1 [0] sram [0] length 16 count 1 stride 0
    knpu.dma_wait 0
    knpu.elementwise {ifm = 0 : i64, ifm2 = 0 : i64, ofm = 16 : i64, shape = array<i64: 2, 2, 4>, ifm_zp = -128 : i64, ifm2_zp = -128 : i64, ofm_zp = 5 : i64, ifm_scale = array<i64: 1073741824, 0>, ifm2_scale = array<i64: 1073741824, 0>, ofm_scale = array<i64: 1127607261, -17>, clamp = array<i64: 5, 127>, round = "single"}
    knpu.elementwise {ifm = 16 : i64, ifm2 = 0 : i64, ofm = 0 : i64, shape = array<i64: 2, 2, 4>, ifm_zp = 5 : i64, ifm2_zp = -128 : i64, ofm_zp = 5 : i64, ifm_scale = array<i64: 1073741824, 0>, ifm2_scale = array<i64: 1073741824, 0>, ofm_scale = array<i64: 1127607261, -17>, clamp = array<i64: 5, 127>, round = "single"}
    knpu.kernel_wait 0
    knpu.dma "store" region 1 [16] sram [0] length 16 count 1 stride 0
    knpu.stop
    return
  }
}

// CHECK: 0000000 55504e4b
// CHECK-NEXT: 0000004 00000000
// CHECK-NEXT: 0000008 00010000
// CHECK-NEXT: 0000012 00030000
// CHECK-NEXT: 0000016 00000020
// CHECK: 0000032 00000001
// CHECK-NEXT: 0000036 00000050
// CHECK-NEXT: 0000040 000000a4
// CHECK: 0000048 00000002
// CHECK-NEXT: 0000052 00000100
// CHECK: 0000064 00000004
// CHECK-NEXT: 0000068 00000110
// Load of the input: SET_DMA_SRC region 1 at 0, SET_DMA_LEN 16, OP_DMA
// CHECK: 0000080 00014010
// CHECK-NEXT: 0000084 00000000
// CHECK-NEXT: 0000088 00004012
// CHECK-NEXT: 0000092 00000010
// CHECK-NEXT: 0000096 00000010
// First addition: SET_IFM_BASE 0, SET_IFM_SHAPE 2x2x4, SET_IFM_ZP -128,
// SET_IFM_SCALE (2^30, 0), SET_IFM2_BASE 0, SET_IFM2_ZP -128,
// SET_IFM2_SCALE (2^30, 0), SET_OFM_BASE 16, SET_OFM_SHAPE 2x2x4,
// SET_OFM_ZP 5, SET_OFM_SCALE (1127607261, -17), SET_CLAMP (5, 127),
// SET_ROUND_MODE single, then the wait and OP_ADD
// CHECK-NEXT: 0000100 00004000
// CHECK-NEXT: 0000104 00000000
// CHECK-NEXT: 0000108 00044006
// CHECK-NEXT: 0000112 00020002
// CHECK-NEXT: 0000116 ff800100
// CHECK-NEXT: 0000120 00004008
// CHECK-NEXT: 0000124 40000000
// CHECK-NEXT: 0000128 00004001
// CHECK-NEXT: 0000132 00000000
// CHECK-NEXT: 0000136 ff800101
// CHECK-NEXT: 0000140 00004009
// CHECK-NEXT: 0000144 40000000
// CHECK-NEXT: 0000148 00004002
// CHECK-NEXT: 0000152 00000010
// CHECK-NEXT: 0000156 00044007
// CHECK-NEXT: 0000160 00020002
// CHECK-NEXT: 0000164 00050102
// CHECK-NEXT: 0000168 ffef400a
// CHECK-NEXT: 0000172 4335ebdd
// CHECK-NEXT: 0000176 7f050106
// CHECK-NEXT: 0000180 00010107
// CHECK-NEXT: 0000184 00000011
// CHECK-NEXT: 0000188 00000003
// Second addition, in place: SET_IFM_BASE 16, SET_IFM_ZP 5,
// SET_OFM_BASE 0, OP_ADD
// CHECK-NEXT: 0000192 00004000
// CHECK-NEXT: 0000196 00000010
// CHECK-NEXT: 0000200 00050100
// CHECK-NEXT: 0000204 00004002
// CHECK-NEXT: 0000208 00000000
// CHECK-NEXT: 0000212 00000003
// Store of the output (the length is already 16), after its wait, and
// OP_STOP
// CHECK-NEXT: 0000216 01004010
// CHECK-NEXT: 0000220 00000000
// CHECK-NEXT: 0000224 00014011
// CHECK-NEXT: 0000228 00000010
// CHECK-NEXT: 0000232 00000012
// CHECK-NEXT: 0000236 00000010
// CHECK-NEXT: 0000240 00000000
// CHECK: 0000368
