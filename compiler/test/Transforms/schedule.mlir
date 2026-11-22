// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: not knpu-opt %s -split-input-file --knpu-tile=sram-size=4096 --knpu-schedule=sram-size=4096 2> %t.err | FileCheck %s
// RUN: FileCheck %s --check-prefix=ERR < %t.err

// One layer, one tile: the shapes of docs/npu-spec.md 12. Region 0 holds
// weights (36 bytes, padded to 48), bias and the requantization table; the
// arena holds the input at 0 and the output at 64. SRAM: constants at 0,
// input at 96, output at 160.

func.func @knpu_model(
    %in: tensor<1x8x8x1xi8> {knpu.scale = 0.5 : f32, knpu.zero_point = -128 : i64})
    -> (tensor<1x8x8x4xi8> {knpu.scale = 0.25 : f32, knpu.zero_point = 0 : i64}) {
  %w = arith.constant dense<1> : tensor<4x3x3x1xi8>
  %b = arith.constant dense<[1, 2, 3, -4]> : tensor<4xi32>
  %rq = arith.constant dense<[[1073741824, -1], [1073741824, -2], [1073741824, -3], [2147483647, 30]]> : tensor<4x2xi32>
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = -128 : i64,
      output_zp = 0 : i64, clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x1xi8>, tensor<4x3x3x1xi8>, tensor<4xi32>,
         tensor<4x2xi32>) -> tensor<1x8x8x4xi8>
  return %0 : tensor<1x8x8x4xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 0 "const" size 96 content dense<[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, -4, -1, -1, -1, 0, 0, 0, 64, -1, -1, -1, -1, 0, 0, 0, 64, -2, -1, -1, -1, 0, 0, 0, 64, -3, -1, -1, -1, -1, -1, -1, 127, 30, 0, 0, 0]> : tensor<96xi8>
// CHECK: knpu.region 1 "arena" size 320
// CHECK: knpu.io "input" region 1[0] shape [1, 8, 8, 1] scale 5.000000e-01 zero_point -128
// CHECK: knpu.io "output" region 1[64] shape [1, 8, 8, 4] scale 2.500000e-01 zero_point 0
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[0] length 96 count 1 stride 0
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[96] length 64 count 1 stride 0
// CHECK-NEXT: knpu.conv {bias = 48 : i64, clamp = array<i64: -128, 127>, ifm = 96 : i64, ifm_shape = array<i64: 8, 8, 1>, ifm_zp = -128 : i64, kernel = array<i64: 3, 3>, ofm = 160 : i64, ofm_shape = array<i64: 8, 8, 4>, ofm_zp = 0 : i64, pad = array<i64: 1, 1, 1, 1>, round = "double", rq_table = 64 : i64, stride = array<i64: 1, 1>, weight = 0 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[64] sram[160] length 256 count 1 stride 0
// CHECK-NEXT: knpu.stop
// CHECK-NEXT: return

// -----

// Two layers: the intermediate tensor gets its own place in the arena, each
// layer its own constants in region 0, loaded before the layer runs.

func.func @knpu_model(
    %in: tensor<1x2x2x1xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x2x2x2xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w1 = arith.constant dense<1> : tensor<2x1x1x1xi8>
  %w2 = arith.constant dense<2> : tensor<2x1x1x2xi8>
  %b = arith.constant dense<0> : tensor<2xi32>
  %rq = arith.constant dense<1073741824> : tensor<2x2xi32>
  %0 = knpu.conv2d %in, %w1, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 3 : i64,
      clamp = array<i64: -128, 127>, round = "single"}
      : (tensor<1x2x2x1xi8>, tensor<2x1x1x1xi8>, tensor<2xi32>,
         tensor<2x2xi32>) -> tensor<1x2x2x2xi8>
  %1 = knpu.conv2d %0, %w2, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 3 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "single"}
      : (tensor<1x2x2x2xi8>, tensor<2x1x1x2xi8>, tensor<2xi32>,
         tensor<2x2xi32>) -> tensor<1x2x2x2xi8>
  return %1 : tensor<1x2x2x2xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 0 "const" size 96
// CHECK: knpu.region 1 "arena" size 40
// CHECK: knpu.io "input" region 1[0]
// CHECK: knpu.io "output" region 1[32]
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[0] length 48
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[48] length 4
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 48 : i64{{.*}}ofm = 64 : i64
// CHECK-NEXT: knpu.dma "store" region 1[16] sram[64] length 8
// CHECK-NEXT: knpu.dma "load" region 0[48] sram[0] length 48
// CHECK-NEXT: knpu.dma "load" region 1[16] sram[48] length 8
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 48 : i64{{.*}}ofm = 64 : i64
// CHECK-NEXT: knpu.dma "store" region 1[32] sram[64] length 8
// CHECK-NEXT: knpu.stop

// -----

// Five tiles in 4 KiB (see tile.mlir): input buffers at 1344 and 1904,
// output buffers at 2464 and 3136; the next tile's input is loaded before
// the current tile's convolution.

func.func @knpu_model(
    %in: tensor<1x14x14x8xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = -128 : i64})
    -> (tensor<1x14x14x16xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = -128 : i64}) {
  %w = arith.constant dense<1> : tensor<16x3x3x8xi8>
  %b = arith.constant dense<0> : tensor<16xi32>
  %rq = arith.constant dense<1073741824> : tensor<16x2xi32>
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = -128 : i64,
      output_zp = -128 : i64, clamp = array<i64: -128, 127>,
      round = "double"}
      : (tensor<1x14x14x8xi8>, tensor<16x3x3x8xi8>, tensor<16xi32>,
         tensor<16x2xi32>) -> tensor<1x14x14x16xi8>
  return %0 : tensor<1x14x14x16xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 0 "const" size 1344
// CHECK: knpu.region 1 "arena" size 4704
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[0] length 1344
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[1344] length 448
// CHECK-NEXT: knpu.dma "load" region 1[224] sram[1904] length 560
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1344 : i64, ifm_shape = array<i64: 4, 14, 8>{{.*}}ofm = 2464 : i64, ofm_shape = array<i64: 3, 14, 16>{{.*}}pad = array<i64: 1, 1, 0, 1>
// CHECK-NEXT: knpu.dma "store" region 1[1568] sram[2464] length 672
// CHECK-NEXT: knpu.dma "load" region 1[560] sram[1344] length 560
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1904 : i64{{.*}}ofm = 3136 : i64
// CHECK-NEXT: knpu.dma "store" region 1[2240] sram[3136] length 672
// CHECK-NEXT: knpu.dma "load" region 1[896] sram[1904] length 560
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1344 : i64{{.*}}ofm = 2464 : i64
// CHECK-NEXT: knpu.dma "store" region 1[2912] sram[2464] length 672
// CHECK-NEXT: knpu.dma "load" region 1[1232] sram[1344] length 336
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1904 : i64{{.*}}ofm = 3136 : i64
// CHECK-NEXT: knpu.dma "store" region 1[3584] sram[3136] length 672
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1344 : i64, ifm_shape = array<i64: 3, 14, 8>{{.*}}ofm = 2464 : i64, ofm_shape = array<i64: 2, 14, 16>{{.*}}pad = array<i64: 0, 1, 1, 1>
// CHECK-NEXT: knpu.dma "store" region 1[4256] sram[2464] length 448
// CHECK-NEXT: knpu.stop

// -----

// ERR: no knpu_model function in graph form (run knpu-import first)
func.func @other() {
  return
}
