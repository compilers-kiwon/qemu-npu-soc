// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: not knpu-opt %s -split-input-file --knpu-estimate 2> %t.err | FileCheck %s
// RUN: FileCheck %s --check-prefix=ERR < %t.err
// RUN: not knpu-opt %s -split-input-file --knpu-estimate="macs-per-cycle=64 dma-bytes-per-cycle=8 pool-elements-per-cycle=32" 2>/dev/null | FileCheck %s --check-prefix=SLOW
// RUN: not knpu-opt %s -split-input-file --knpu-estimate="dma-bytes-per-cycle=0" 2>&1 | FileCheck %s --check-prefix=RATE

// Cycle estimates worked out by hand (docs/design.md 9.1). By default a
// cycle does 256 MACs, moves 16 DMA bytes or adds 256 pool elements;
// lengths round up. The SLOW checks use 64 MACs, 8 bytes and 32 elements
// per cycle.

// The stream of docs/npu-spec.md 12. The constants (112 bytes, 7 cycles)
// and the input (64 bytes, 4) load one after the other; the convolution
// (8*8*4*3*3*1 = 2,304 MACs, 9 cycles) waits for both and the output store
// (256 bytes, 16 cycles) for the convolution; with the slow rates they
// take 14, 8, 36 and 32 cycles.

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

// CHECK-LABEL: func.func @knpu_model() attributes {knpu.total_cycles = 36 : i64}
// CHECK-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 0, 7>}
// CHECK-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 7, 11>}
// CHECK-NEXT: knpu.dma_wait 0
// CHECK-NEXT: knpu.conv {{{.*}}knpu.cycles = array<i64: 11, 20>
// CHECK-NEXT: knpu.kernel_wait 0
// CHECK-NEXT: knpu.dma "store" {{.*}} {knpu.cycles = array<i64: 20, 36>}
// CHECK-NEXT: knpu.stop

// SLOW-LABEL: func.func @knpu_model() attributes {knpu.total_cycles = 90 : i64}
// SLOW-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 0, 14>}
// SLOW-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 14, 22>}
// SLOW-NEXT: knpu.dma_wait 0
// SLOW-NEXT: knpu.conv {{{.*}}knpu.cycles = array<i64: 22, 58>
// SLOW-NEXT: knpu.kernel_wait 0
// SLOW-NEXT: knpu.dma "store" {{.*}} {knpu.cycles = array<i64: 58, 90>}

// -----

// Three loads in a row (32 bytes, 2 cycles each) fill the DMA queue (depth
// 2): the third is issued when the first ends, at 2, and starts when the
// second ends, at 4. The pooling (2*2*256*2*2 = 4,096 window elements, 16
// cycles) needs no wait, so it is issued at 2 as well and runs alongside;
// the stop waits for everything. With the slow rates the loads take 4
// cycles (the third is issued at 4 and runs from 8) and the pooling 128,
// from 4.

module attributes {knpu.sram_size = 65536 : i64} {
  func.func @knpu_model() {
    knpu.dma "load" region 1 [0] sram [0] length 32 count 1 stride 0
    knpu.dma "load" region 1 [32] sram [32] length 32 count 1 stride 0
    knpu.dma "load" region 1 [64] sram [64] length 32 count 1 stride 0
    knpu.pool {ifm = 1024 : i64, ifm_shape = array<i64: 4, 4, 256>, ofm = 8192 : i64, ofm_shape = array<i64: 2, 2, 256>, kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, clamp = array<i64: -128, 127>}
    knpu.stop
    return
  }
}

// CHECK-LABEL: func.func @knpu_model() attributes {knpu.total_cycles = 18 : i64}
// CHECK-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 0, 2>}
// CHECK-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 2, 4>}
// CHECK-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 4, 6>}
// CHECK-NEXT: knpu.pool {{{.*}}knpu.cycles = array<i64: 2, 18>
// CHECK-NEXT: knpu.stop

// SLOW-LABEL: func.func @knpu_model() attributes {knpu.total_cycles = 132 : i64}
// SLOW-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 0, 4>}
// SLOW-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 4, 8>}
// SLOW-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 8, 12>}
// SLOW-NEXT: knpu.pool {{{.*}}knpu.cycles = array<i64: 4, 132>

// -----

// Lengths round up: 100 bytes take 7 cycles, 1,000 MACs 4. A strided DMA
// moves length * count bytes (3 rows of 8: 2 cycles).

module attributes {knpu.sram_size = 65536 : i64} {
  func.func @knpu_model() {
    knpu.dma "load" region 1 [0] sram [0] length 100 count 1 stride 0
    knpu.dma_wait 0
    knpu.conv {ifm = 0 : i64, ifm_shape = array<i64: 1, 1, 100>, ifm_zp = 0 : i64, ofm = 2048 : i64, ofm_shape = array<i64: 1, 1, 10>, ofm_zp = 0 : i64, weight = 128 : i64, bias = 1152 : i64, rq_table = 1216 : i64, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, clamp = array<i64: -128, 127>, round = "double"}
    knpu.kernel_wait 0
    knpu.dma "store" region 1 [128] sram [2048] length 8 count 3 stride 16
    knpu.stop
    return
  }
}

// CHECK-LABEL: func.func @knpu_model() attributes {knpu.total_cycles = 13 : i64}
// CHECK-NEXT: knpu.dma "load" {{.*}} {knpu.cycles = array<i64: 0, 7>}
// CHECK-NEXT: knpu.dma_wait 0
// CHECK-NEXT: knpu.conv {{{.*}}knpu.cycles = array<i64: 7, 11>
// CHECK-NEXT: knpu.kernel_wait 0
// CHECK-NEXT: knpu.dma "store" {{.*}} {knpu.cycles = array<i64: 11, 13>}

// -----

// ERR: no knpu_model command function (run knpu-schedule first)
// RATE: knpu-estimate rates must be positive
func.func @knpu_model(%in: tensor<1x1x1x1xi8>) -> tensor<1x1x1x1xi8> {
  return %in : tensor<1x1x1x1xi8>
}
