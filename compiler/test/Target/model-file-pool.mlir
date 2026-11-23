// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: knpu-translate --knpu-to-model %s | od -A d -t x4 -v -w4 | FileCheck %s

// OP_AVGPOOL with its SET commands (docs/npu-spec.md 6.1, 8.4): the IFM
// and OFM bases and shapes, kernel, stride, padding and clamp, but no zero
// points or ROUND_MODE, which the operation does not use. Three sections
// (no constant region): CMDS at 80 (112 bytes), REGIONS at 192, TENSORS at
// 208.

module attributes {knpu.sram_size = 65536 : i64} {
  knpu.region 1 "arena" size 80
  knpu.io "input" region 1 [0] shape [1, 4, 4, 4] scale 1.0 zero_point 0
  knpu.io "output" region 1 [64] shape [1, 2, 2, 4] scale 1.0 zero_point 0
  func.func @knpu_model() {
    knpu.dma "load" region 1 [0] sram [0] length 64 count 1 stride 0
    knpu.dma_wait 0
    knpu.pool {ifm = 0 : i64, ifm_shape = array<i64: 4, 4, 4>, ofm = 64 : i64, ofm_shape = array<i64: 2, 2, 4>, kernel = array<i64: 3, 3>, stride = array<i64: 2, 2>, pad = array<i64: 1, 1, 1, 1>, clamp = array<i64: -100, 100>}
    knpu.kernel_wait 0
    knpu.dma "store" region 1 [64] sram [64] length 16 count 1 stride 0
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
// CHECK-NEXT: 0000040 00000070
// CHECK: 0000048 00000002
// CHECK-NEXT: 0000052 000000c0
// CHECK: 0000064 00000004
// CHECK-NEXT: 0000068 000000d0
// Load of the input: SET_DMA_SRC region 1 at 0, SET_DMA_LEN 64 (the
// destination SRAM 0 and the count 1 are the state after START), OP_DMA
// CHECK: 0000080 00014010
// CHECK-NEXT: 0000084 00000000
// CHECK-NEXT: 0000088 00004012
// CHECK-NEXT: 0000092 00000040
// CHECK-NEXT: 0000096 00000010
// SET_IFM_BASE 0, SET_IFM_SHAPE 4x4x4, SET_KERNEL 3x3, SET_STRIDE 2x2,
// SET_PAD 1 on every side, SET_OFM_BASE 64, SET_OFM_SHAPE 2x2x4,
// SET_CLAMP (-100, 100), then the wait and OP_AVGPOOL
// CHECK-NEXT: 0000100 00004000
// CHECK-NEXT: 0000104 00000000
// CHECK-NEXT: 0000108 00044006
// CHECK-NEXT: 0000112 00040004
// CHECK-NEXT: 0000116 03030103
// CHECK-NEXT: 0000120 02020104
// CHECK-NEXT: 0000124 11110105
// CHECK-NEXT: 0000128 00004002
// CHECK-NEXT: 0000132 00000040
// CHECK-NEXT: 0000136 00044007
// CHECK-NEXT: 0000140 00020002
// CHECK-NEXT: 0000144 649c0106
// CHECK-NEXT: 0000148 00000011
// CHECK-NEXT: 0000152 00000004
// Store of the output, after its wait, and OP_STOP
// CHECK-NEXT: 0000156 01004010
// CHECK-NEXT: 0000160 00000040
// CHECK-NEXT: 0000164 00014011
// CHECK-NEXT: 0000168 00000040
// CHECK-NEXT: 0000172 00004012
// CHECK-NEXT: 0000176 00000010
// CHECK-NEXT: 0000180 00000012
// CHECK-NEXT: 0000184 00000010
// CHECK-NEXT: 0000188 00000000
// CHECK-NEXT: 0000192 00000001
// CHECK: 0000304
