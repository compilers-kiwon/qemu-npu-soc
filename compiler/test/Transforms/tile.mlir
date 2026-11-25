// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: knpu-opt %s -split-input-file --knpu-tile=sram-size=65536 | FileCheck %s --check-prefix=BIG
// RUN: not knpu-opt %s -split-input-file --knpu-tile=sram-size=4096 2> %t.err | FileCheck %s --check-prefix=SMALL
// RUN: FileCheck %s --check-prefix=ERR < %t.err
// RUN: not knpu-opt %s -split-input-file --knpu-tile="sram-size=4096 overlap=false" 2>/dev/null | FileCheck %s --check-prefix=SERIAL

// The test model's layer (14x14x8 -> 16, 3x3, SAME): constants 1,344
// bytes, input 1,568, output 3,136. In 64 KiB it stays whole. In 4 KiB,
// two input and two output buffers of a 3-row tile fit (5 input rows):
// 1344 + 2 * 560 + 2 * 672 = 3808 bytes. Without overlap one buffer each
// of a 7-row tile fits (9 input rows): 1344 + 1008 + 1568 = 3920 bytes.

func.func @conv(%in: tensor<1x14x14x8xi8>, %w: tensor<16x3x3x8xi8>,
                %b: tensor<16xi32>, %rq: tensor<16x2xi32>)
    -> tensor<1x14x14x16xi8> {
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = -128 : i64,
      output_zp = -128 : i64, clamp = array<i64: -128, 127>,
      round = "double"}
      : (tensor<1x14x14x8xi8>, tensor<16x3x3x8xi8>, tensor<16xi32>,
         tensor<16x2xi32>) -> tensor<1x14x14x16xi8>
  return %0 : tensor<1x14x14x16xi8>
}

// BIG-LABEL: func.func @conv
// BIG-NOT: tensor.extract_slice
// BIG: knpu.conv2d %arg0
// BIG-SAME: pad = array<i64: 1, 1, 1, 1>
// BIG-NOT: knpu.conv2d

// SMALL-LABEL: func.func @conv
// SMALL: %[[E:.*]] = tensor.empty() : tensor<1x14x14x16xi8>
// SMALL: %[[S0:.*]] = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 4, 14, 8] [1, 1, 1, 1]
// SMALL: %[[T0:.*]] = knpu.conv2d %[[S0]], %arg1, %arg2, %arg3 {{{.*}}pad = array<i64: 1, 1, 0, 1>{{.*}}} : {{.*}} -> tensor<1x3x14x16xi8>
// SMALL: %[[R0:.*]] = tensor.insert_slice %[[T0]] into %[[E]][0, 0, 0, 0] [1, 3, 14, 16] [1, 1, 1, 1]
// SMALL: tensor.extract_slice %arg0[0, 2, 0, 0] [1, 5, 14, 8]
// SMALL: knpu.conv2d {{.*}}pad = array<i64: 0, 1, 0, 1>
// SMALL: tensor.insert_slice {{.*}} into %[[R0]][0, 3, 0, 0] [1, 3, 14, 16]
// SMALL: tensor.extract_slice %arg0[0, 5, 0, 0] [1, 5, 14, 8]
// SMALL: tensor.extract_slice %arg0[0, 8, 0, 0] [1, 5, 14, 8]
// SMALL: %[[S4:.*]] = tensor.extract_slice %arg0[0, 11, 0, 0] [1, 3, 14, 8]
// SMALL: knpu.conv2d %[[S4]], {{.*}}pad = array<i64: 0, 1, 1, 1>{{.*}} -> tensor<1x2x14x16xi8>
// SMALL: %[[R4:.*]] = tensor.insert_slice {{.*}}[0, 12, 0, 0] [1, 2, 14, 16]
// SMALL: return %[[R4]]

// SERIAL-LABEL: func.func @conv
// SERIAL: tensor.extract_slice %arg0[0, 0, 0, 0] [1, 8, 14, 8]
// SERIAL: knpu.conv2d {{.*}} -> tensor<1x7x14x16xi8>
// SERIAL: tensor.extract_slice %arg0[0, 6, 0, 0] [1, 8, 14, 8]
// SERIAL: knpu.conv2d {{.*}} -> tensor<1x7x14x16xi8>
// SERIAL-NOT: knpu.conv2d
// SERIAL: return

// -----

// Stride 2 (32x16x8 -> 16x8x8, 3x3, SAME: pad top 0, bottom 1). Constants
// take 672 bytes; in 4 KiB a tile has 4 output rows and reads 9 input rows
// (672 + 2 * 1152 + 2 * 256 = 3488). The last tile's window runs past the
// input and keeps the bottom padding.

func.func @stride2(%in: tensor<1x32x16x8xi8>, %w: tensor<8x3x3x8xi8>,
                   %b: tensor<8xi32>, %rq: tensor<8x2xi32>)
    -> tensor<1x16x8x8xi8> {
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 2, 2>,
      pad = array<i64: 0, 0, 1, 1>, input_zp = 0 : i64,
      output_zp = 0 : i64, clamp = array<i64: -128, 127>, round = "single"}
      : (tensor<1x32x16x8xi8>, tensor<8x3x3x8xi8>, tensor<8xi32>,
         tensor<8x2xi32>) -> tensor<1x16x8x8xi8>
  return %0 : tensor<1x16x8x8xi8>
}

// BIG-LABEL: func.func @stride2
// BIG-NOT: tensor.extract_slice

// SMALL-LABEL: func.func @stride2
// SMALL: tensor.extract_slice %arg0[0, 0, 0, 0] [1, 9, 16, 8]
// SMALL: knpu.conv2d {{.*}}pad = array<i64: 0, 0, 0, 1>{{.*}} -> tensor<1x4x8x8xi8>
// SMALL: tensor.extract_slice %arg0[0, 8, 0, 0] [1, 9, 16, 8]
// SMALL: tensor.extract_slice %arg0[0, 16, 0, 0] [1, 9, 16, 8]
// SMALL: tensor.extract_slice %arg0[0, 24, 0, 0] [1, 8, 16, 8]
// SMALL: knpu.conv2d {{.*}}pad = array<i64: 0, 0, 1, 1>{{.*}} -> tensor<1x4x8x8xi8>
// SMALL-NOT: knpu.conv2d
// SMALL: return

// -----

// Weights alone (36,864 bytes) exceed 4 KiB: output-channel tiles would be
// needed.

func.func @too_big(%in: tensor<1x8x8x64xi8>, %w: tensor<64x3x3x64xi8>,
                   %b: tensor<64xi32>, %rq: tensor<64x2xi32>)
    -> tensor<1x8x8x64xi8> {
  // ERR: does not fit in 4096 bytes of SRAM even with one output row per tile
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = 0 : i64,
      output_zp = 0 : i64, clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x64xi8>, tensor<64x3x3x64xi8>, tensor<64xi32>,
         tensor<64x2xi32>) -> tensor<1x8x8x64xi8>
  return %0 : tensor<1x8x8x64xi8>
}

// BIG-LABEL: func.func @too_big
// BIG-NOT: tensor.extract_slice

// -----

// Average pooling (32x32x4 -> 16x16x4, 2x2, stride 2) tiles like a
// convolution without constants. In 4 KiB a tile of t output rows needs
// 2 * 256t + 2 * 64t bytes double-buffered (t = 6) and 256t + 64t with one
// buffer each (t = 12).

func.func @pool(%in: tensor<1x32x32x4xi8>) -> tensor<1x16x16x4xi8> {
  %0 = knpu.avgpool %in {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>,
      pad = array<i64: 0, 0, 0, 0>, clamp = array<i64: -128, 127>}
      : (tensor<1x32x32x4xi8>) -> tensor<1x16x16x4xi8>
  return %0 : tensor<1x16x16x4xi8>
}

// BIG-LABEL: func.func @pool
// BIG-NOT: tensor.extract_slice
// BIG: knpu.avgpool %arg0

// SMALL-LABEL: func.func @pool
// SMALL: tensor.extract_slice %arg0[0, 0, 0, 0] [1, 12, 32, 4]
// SMALL: knpu.avgpool {{.*}} -> tensor<1x6x16x4xi8>
// SMALL: tensor.extract_slice %arg0[0, 12, 0, 0] [1, 12, 32, 4]
// SMALL: knpu.avgpool {{.*}} -> tensor<1x6x16x4xi8>
// SMALL: tensor.extract_slice %arg0[0, 24, 0, 0] [1, 8, 32, 4]
// SMALL: knpu.avgpool {{.*}} -> tensor<1x4x16x4xi8>
// SMALL-NOT: knpu.avgpool
// SMALL: return

// SERIAL-LABEL: func.func @pool
// SERIAL: tensor.extract_slice %arg0[0, 0, 0, 0] [1, 24, 32, 4]
// SERIAL: knpu.avgpool {{.*}} -> tensor<1x12x16x4xi8>
// SERIAL: tensor.extract_slice %arg0[0, 24, 0, 0] [1, 8, 32, 4]
// SERIAL: knpu.avgpool {{.*}} -> tensor<1x4x16x4xi8>
// SERIAL-NOT: knpu.avgpool
// SERIAL: return

// -----

// A fully connected layer (a 1x1 convolution with one output row) is not
// split: its weights alone (8,192 bytes) exceed 4 KiB.

func.func @fc_too_big(%in: tensor<1x1x1x4096xi8>, %w: tensor<2x1x1x4096xi8>,
                      %b: tensor<2xi32>, %rq: tensor<2x2xi32>)
    -> tensor<1x1x1x2xi8> {
  // ERR: does not fit in 4096 bytes of SRAM even with one output row per tile
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64,
      output_zp = 0 : i64, clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x1x1x4096xi8>, tensor<2x1x1x4096xi8>, tensor<2xi32>,
         tensor<2x2xi32>) -> tensor<1x1x1x2xi8>
  return %0 : tensor<1x1x1x2xi8>
}

// BIG-LABEL: func.func @fc_too_big
// BIG-NOT: tensor.extract_slice

// -----

// An addition of two 16x16x8 tensors (2,048 bytes each) is split into tiles
// of the same rows of both inputs and the output: two buffers of each, 640
// bytes for five rows, with overlap; one of each, 1,280 bytes for ten rows,
// without.

func.func @add(%a: tensor<1x16x16x8xi8>, %b: tensor<1x16x16x8xi8>)
    -> tensor<1x16x16x8xi8> {
  %0 = knpu.add %a, %b {input_zp = 0 : i64, input2_zp = 0 : i64,
      output_zp = 0 : i64, input_scale = array<i64: 1073741824, 0>,
      input2_scale = array<i64: 1073741824, 0>,
      output_scale = array<i64: 1073741824, -19>,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x16x16x8xi8>, tensor<1x16x16x8xi8>) -> tensor<1x16x16x8xi8>
  return %0 : tensor<1x16x16x8xi8>
}

// BIG-LABEL: func.func @add
// BIG-NOT: tensor.extract_slice
// BIG: knpu.add %arg0, %arg1

// SMALL-LABEL: func.func @add
// SMALL: %[[E:.*]] = tensor.empty() : tensor<1x16x16x8xi8>
// SMALL: %[[A0:.*]] = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 5, 16, 8] [1, 1, 1, 1]
// SMALL: %[[B0:.*]] = tensor.extract_slice %arg1[0, 0, 0, 0] [1, 5, 16, 8] [1, 1, 1, 1]
// SMALL: %[[T0:.*]] = knpu.add %[[A0]], %[[B0]] {{.*}} -> tensor<1x5x16x8xi8>
// SMALL: tensor.insert_slice %[[T0]] into %[[E]][0, 0, 0, 0] [1, 5, 16, 8]
// SMALL: tensor.extract_slice %arg0[0, 5, 0, 0] [1, 5, 16, 8]
// SMALL: tensor.extract_slice %arg0[0, 10, 0, 0] [1, 5, 16, 8]
// SMALL: tensor.extract_slice %arg0[0, 15, 0, 0] [1, 1, 16, 8]
// SMALL: tensor.extract_slice %arg1[0, 15, 0, 0] [1, 1, 16, 8]
// SMALL: knpu.add {{.*}} -> tensor<1x1x16x8xi8>
// SMALL: tensor.insert_slice {{.*}}[0, 15, 0, 0] [1, 1, 16, 8]
// SMALL-NOT: knpu.add
// SMALL: return

// SERIAL-LABEL: func.func @add
// SERIAL: knpu.add {{.*}} -> tensor<1x10x16x8xi8>
// SERIAL: knpu.add {{.*}} -> tensor<1x6x16x8xi8>
// SERIAL-NOT: knpu.add
// SERIAL: return
