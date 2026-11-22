// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: knpu-opt %s -split-input-file --knpu-insert-waits | FileCheck %s

// The stream of docs/npu-spec.md 12 without its waits: the convolution
// reads what both DMAs write, the output DMA reads what the convolution
// writes.

module attributes {knpu.sram_size = 65536 : i64} {
  func.func @knpu_model() {
    knpu.dma "load" region 0 [0] sram [256] length 112 count 1 stride 0
    knpu.dma "load" region 1 [0] sram [0] length 64 count 1 stride 0
    knpu.conv {ifm = 0 : i64, ifm_shape = array<i64: 8, 8, 1>, ifm_zp = -128 : i64, ofm = 512 : i64, ofm_shape = array<i64: 8, 8, 4>, ofm_zp = 0 : i64, weight = 256 : i64, bias = 320 : i64, rq_table = 336 : i64, kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, clamp = array<i64: -128, 127>, round = "double"}
    knpu.dma "store" region 1 [64] sram [512] length 256 count 1 stride 0
    knpu.stop
    return
  }
}

// CHECK-LABEL: func.func @knpu_model
// CHECK-NEXT: knpu.dma "load" region 0[0]
// CHECK-NEXT: knpu.dma "load" region 1[0]
// CHECK-NEXT: knpu.dma_wait 0
// CHECK-NEXT: knpu.conv
// CHECK-NEXT: knpu.kernel_wait 0
// CHECK-NEXT: knpu.dma "store"
// CHECK-NEXT: knpu.stop

// -----

// Five double-buffered tiles in 4 KiB (the test model's layer). Queue depth
// 2 completes the oldest DMA when a third is issued (implicit wait), so the
// convolutions of tiles 1 to 3 need no wait: their input loads are already
// done. Tile 0 and 4 wait for their input load and leave the newer DMA
// running (dma_wait 1). Every output DMA waits for its convolution.

module attributes {knpu.sram_size = 4096 : i64} {
  func.func @knpu_model() {
    knpu.dma "load" region 0 [0] sram [0] length 1344 count 1 stride 0
    knpu.dma "load" region 1 [0] sram [1344] length 448 count 1 stride 0
    knpu.dma "load" region 1 [224] sram [1904] length 560 count 1 stride 0
    knpu.conv {ifm = 1344 : i64, ifm_shape = array<i64: 4, 14, 8>, ifm_zp = -128 : i64, ofm = 2464 : i64, ofm_shape = array<i64: 3, 14, 16>, ofm_zp = -128 : i64, weight = 0 : i64, bias = 1152 : i64, rq_table = 1216 : i64, kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>, pad = array<i64: 1, 1, 0, 1>, clamp = array<i64: -128, 127>, round = "double"}
    knpu.dma "store" region 1 [1568] sram [2464] length 672 count 1 stride 0
    knpu.dma "load" region 1 [560] sram [1344] length 560 count 1 stride 0
    knpu.conv {ifm = 1904 : i64, ifm_shape = array<i64: 5, 14, 8>, ifm_zp = -128 : i64, ofm = 3136 : i64, ofm_shape = array<i64: 3, 14, 16>, ofm_zp = -128 : i64, weight = 0 : i64, bias = 1152 : i64, rq_table = 1216 : i64, kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, clamp = array<i64: -128, 127>, round = "double"}
    knpu.dma "store" region 1 [2240] sram [3136] length 672 count 1 stride 0
    knpu.dma "load" region 1 [896] sram [1904] length 560 count 1 stride 0
    knpu.conv {ifm = 1344 : i64, ifm_shape = array<i64: 5, 14, 8>, ifm_zp = -128 : i64, ofm = 2464 : i64, ofm_shape = array<i64: 3, 14, 16>, ofm_zp = -128 : i64, weight = 0 : i64, bias = 1152 : i64, rq_table = 1216 : i64, kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, clamp = array<i64: -128, 127>, round = "double"}
    knpu.dma "store" region 1 [2912] sram [2464] length 672 count 1 stride 0
    knpu.dma "load" region 1 [1232] sram [1344] length 336 count 1 stride 0
    knpu.conv {ifm = 1904 : i64, ifm_shape = array<i64: 5, 14, 8>, ifm_zp = -128 : i64, ofm = 3136 : i64, ofm_shape = array<i64: 3, 14, 16>, ofm_zp = -128 : i64, weight = 0 : i64, bias = 1152 : i64, rq_table = 1216 : i64, kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, clamp = array<i64: -128, 127>, round = "double"}
    knpu.dma "store" region 1 [3584] sram [3136] length 672 count 1 stride 0
    knpu.conv {ifm = 1344 : i64, ifm_shape = array<i64: 3, 14, 8>, ifm_zp = -128 : i64, ofm = 2464 : i64, ofm_shape = array<i64: 2, 14, 16>, ofm_zp = -128 : i64, weight = 0 : i64, bias = 1152 : i64, rq_table = 1216 : i64, kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>, pad = array<i64: 0, 1, 1, 1>, clamp = array<i64: -128, 127>, round = "double"}
    knpu.dma "store" region 1 [4256] sram [2464] length 448 count 1 stride 0
    knpu.stop
    return
  }
}

// CHECK-LABEL: func.func @knpu_model
// CHECK-NEXT: knpu.dma "load" region 0[0]
// CHECK-NEXT: knpu.dma "load" region 1[0]
// CHECK-NEXT: knpu.dma "load" region 1[224]
// CHECK-NEXT: knpu.dma_wait 1
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1344
// CHECK-NEXT: knpu.kernel_wait 0
// CHECK-NEXT: knpu.dma "store" region 1[1568]
// CHECK-NEXT: knpu.dma "load" region 1[560]
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1904
// CHECK-NEXT: knpu.kernel_wait 0
// CHECK-NEXT: knpu.dma "store" region 1[2240]
// CHECK-NEXT: knpu.dma "load" region 1[896]
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1344
// CHECK-NEXT: knpu.kernel_wait 0
// CHECK-NEXT: knpu.dma "store" region 1[2912]
// CHECK-NEXT: knpu.dma "load" region 1[1232]
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1904
// CHECK-NEXT: knpu.kernel_wait 0
// CHECK-NEXT: knpu.dma "store" region 1[3584]
// CHECK-NEXT: knpu.dma_wait 1
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1344
// CHECK-NEXT: knpu.kernel_wait 0
// CHECK-NEXT: knpu.dma "store" region 1[4256]
// CHECK-NEXT: knpu.stop

// -----

// Waits already in the stream are followed, not duplicated.

module attributes {knpu.sram_size = 65536 : i64} {
  func.func @knpu_model() {
    knpu.dma "load" region 0 [0] sram [256] length 112 count 1 stride 0
    knpu.dma "load" region 1 [0] sram [0] length 64 count 1 stride 0
    knpu.dma_wait 0
    knpu.conv {ifm = 0 : i64, ifm_shape = array<i64: 8, 8, 1>, ifm_zp = -128 : i64, ofm = 512 : i64, ofm_shape = array<i64: 8, 8, 4>, ofm_zp = 0 : i64, weight = 256 : i64, bias = 320 : i64, rq_table = 336 : i64, kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, clamp = array<i64: -128, 127>, round = "double"}
    knpu.kernel_wait 0
    knpu.dma "store" region 1 [64] sram [512] length 256 count 1 stride 0
    knpu.stop
    return
  }
}

// CHECK-LABEL: func.func @knpu_model
// CHECK-NEXT: knpu.dma "load" region 0[0]
// CHECK-NEXT: knpu.dma "load" region 1[0]
// CHECK-NEXT: knpu.dma_wait 0
// CHECK-NEXT: knpu.conv
// CHECK-NEXT: knpu.kernel_wait 0
// CHECK-NEXT: knpu.dma "store"
// CHECK-NEXT: knpu.stop
