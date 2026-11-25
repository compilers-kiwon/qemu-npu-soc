// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: not knpu-opt %s -split-input-file --knpu-tile=sram-size=4096 --knpu-schedule=sram-size=4096 2> %t.err | FileCheck %s
// RUN: FileCheck %s --check-prefix=ERR < %t.err
// RUN: not knpu-opt %s -split-input-file --knpu-tile="sram-size=4096 overlap=false" --knpu-schedule="sram-size=4096 overlap=false" 2>/dev/null | FileCheck %s --check-prefix=SERIAL

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

// Two layers. The intermediate tensor stays in SRAM (at 48) for the second
// layer, whose constants are loaded (to 0) while the first layer runs; only
// the model input and output are in the arena. Serially (SERIAL), the
// intermediate tensor goes to the arena and back, and each layer loads its
// constants to 0 before it runs.

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
// CHECK: knpu.region 1 "arena" size 24
// CHECK: knpu.io "input" region 1[0]
// CHECK: knpu.io "output" region 1[16]
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[64] length 48
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[112] length 4
// CHECK-NEXT: knpu.dma "load" region 0[48] sram[0] length 48
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 112 : i64{{.*}}ofm = 48 : i64{{.*}}weight = 64 : i64}
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 48 : i64{{.*}}ofm = 64 : i64{{.*}}weight = 0 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[16] sram[64] length 8
// CHECK-NEXT: knpu.stop

// SERIAL-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// SERIAL: knpu.region 1 "arena" size 320
// SERIAL-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// SERIAL: knpu.region 0 "const" size 96
// SERIAL: knpu.region 1 "arena" size 40
// SERIAL: knpu.io "input" region 1[0]
// SERIAL: knpu.io "output" region 1[32]
// SERIAL-LABEL: func.func @knpu_model() {
// SERIAL-NEXT: knpu.dma "load" region 0[0] sram[0] length 48
// SERIAL-NEXT: knpu.dma "load" region 1[0] sram[48] length 4
// SERIAL-NEXT: knpu.conv {{{.*}}ifm = 48 : i64{{.*}}ofm = 64 : i64
// SERIAL-NEXT: knpu.dma "store" region 1[16] sram[64] length 8
// SERIAL-NEXT: knpu.dma "load" region 0[48] sram[0] length 48
// SERIAL-NEXT: knpu.dma "load" region 1[16] sram[48] length 8
// SERIAL-NEXT: knpu.conv {{{.*}}ifm = 48 : i64{{.*}}ofm = 64 : i64
// SERIAL-NEXT: knpu.dma "store" region 1[32] sram[64] length 8
// SERIAL-NEXT: knpu.stop

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

// conv, avgpool, reshape and fc (a 1x1 convolution) as knpu-import makes
// them. The conv output (at 48) and the pool output (at 80, read by the fc
// through the reshape) stay in SRAM; the arena holds only the model input
// and output. The pool has no constants, so the fc constants are loaded
// (to 0) while the conv runs.

func.func @knpu_model(
    %in: tensor<1x4x4x1xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x1x1x2xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w1 = arith.constant dense<1> : tensor<2x1x1x1xi8>
  %b1 = arith.constant dense<0> : tensor<2xi32>
  %rq1 = arith.constant dense<1073741824> : tensor<2x2xi32>
  %0 = knpu.conv2d %in, %w1, %b1, %rq1 {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x4x4x1xi8>, tensor<2x1x1x1xi8>, tensor<2xi32>,
         tensor<2x2xi32>) -> tensor<1x4x4x2xi8>
  %1 = knpu.avgpool %0 {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>,
      pad = array<i64: 0, 0, 0, 0>, clamp = array<i64: -128, 127>}
      : (tensor<1x4x4x2xi8>) -> tensor<1x2x2x2xi8>
  %shape = arith.constant dense<[1, 1, 1, 8]> : tensor<4xindex>
  %2 = tensor.reshape %1(%shape)
      : (tensor<1x2x2x2xi8>, tensor<4xindex>) -> tensor<1x1x1x8xi8>
  %w3 = arith.constant dense<2> : tensor<2x1x1x8xi8>
  %b3 = arith.constant dense<0> : tensor<2xi32>
  %rq3 = arith.constant dense<1073741824> : tensor<2x2xi32>
  %3 = knpu.conv2d %2, %w3, %b3, %rq3 {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x1x1x8xi8>, tensor<2x1x1x8xi8>, tensor<2xi32>,
         tensor<2x2xi32>) -> tensor<1x1x1x2xi8>
  return %3 : tensor<1x1x1x2xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 0 "const" size 96
// CHECK: knpu.region 1 "arena" size 18
// CHECK: knpu.io "output" region 1[16] shape [1, 1, 1, 2]
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[80] length 48
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[128] length 16
// CHECK-NEXT: knpu.dma "load" region 0[48] sram[0] length 48
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 128 : i64{{.*}}ofm = 48 : i64{{.*}}weight = 80 : i64}
// CHECK-NEXT: knpu.pool {clamp = array<i64: -128, 127>, ifm = 48 : i64, ifm_shape = array<i64: 4, 4, 2>, kernel = array<i64: 2, 2>, ofm = 80 : i64, ofm_shape = array<i64: 2, 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 80 : i64, ifm_shape = array<i64: 1, 1, 8>{{.*}}ofm = 48 : i64{{.*}}weight = 0 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[16] sram[48] length 2
// CHECK-NEXT: knpu.stop

// -----

// A cpu layer after the last NPU layer. Its input goes to the arena (at 16)
// and its output gets a place there (at 32, the model output); it becomes
// the CPU steps: copy in, call, copy out.

func.func @knpu_model(
    %in: tensor<1x2x2x1xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x2x2x2xi8> {knpu.scale = 0.00390625 : f32, knpu.zero_point = -128 : i64}) {
  %w = arith.constant dense<1> : tensor<2x1x1x1xi8>
  %b = arith.constant dense<0> : tensor<2xi32>
  %rq = arith.constant dense<1073741824> : tensor<2x2xi32>
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x2x2x1xi8>, tensor<2x1x1x1xi8>, tensor<2xi32>,
         tensor<2x2xi32>) -> tensor<1x2x2x2xi8>
  %1 = knpu.cpu @layer2(%0) {in_array = @act1, out_array = @act2}
      : (tensor<1x2x2x2xi8>) -> tensor<1x2x2x2xi8>
  return %1 : tensor<1x2x2x2xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 1 "arena" size 40
// CHECK: knpu.io "output" region 1[32] shape [1, 2, 2, 2] scale 3.906250e-03 zero_point -128
// CHECK-NEXT: knpu.cpu_step "copy_in" @act1 arena[16] size 8
// CHECK-NEXT: knpu.cpu_step "call" @layer2
// CHECK-NEXT: knpu.cpu_step "copy_out" @act2 arena[32] size 8
// CHECK-LABEL: func.func @knpu_model() {
// CHECK: knpu.conv
// CHECK-NEXT: knpu.dma "store" region 1[16] sram[64] length 8
// CHECK-NEXT: knpu.stop

// -----

// Too little SRAM to load ahead: the second layer's constants (1,248
// bytes) do not fit next to the first layer's constants (1,344), input
// (512) and output (1,024, kept in SRAM at 0) in 4 KiB. They are loaded
// after the first layer runs, to where its constants were.

func.func @knpu_model(
    %in: tensor<1x8x8x8xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x4x4x8xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w1 = arith.constant dense<1> : tensor<16x3x3x8xi8>
  %b1 = arith.constant dense<0> : tensor<16xi32>
  %rq1 = arith.constant dense<1073741824> : tensor<16x2xi32>
  %0 = knpu.conv2d %in, %w1, %b1, %rq1 {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x8xi8>, tensor<16x3x3x8xi8>, tensor<16xi32>,
         tensor<16x2xi32>) -> tensor<1x8x8x16xi8>
  %w2 = arith.constant dense<1> : tensor<8x3x3x16xi8>
  %b2 = arith.constant dense<0> : tensor<8xi32>
  %rq2 = arith.constant dense<1073741824> : tensor<8x2xi32>
  %1 = knpu.conv2d %0, %w2, %b2, %rq2 {stride = array<i64: 2, 2>,
      pad = array<i64: 0, 0, 1, 1>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x16xi8>, tensor<8x3x3x16xi8>, tensor<8xi32>,
         tensor<8x2xi32>) -> tensor<1x4x4x8xi8>
  return %1 : tensor<1x4x4x8xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[1024] length 1344
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[2368] length 512
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 2368 : i64{{.*}}ofm = 0 : i64{{.*}}weight = 1024 : i64}
// CHECK-NEXT: knpu.dma "load" region 0[1344] sram[1024] length 1248
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 0 : i64{{.*}}ofm = 2272 : i64{{.*}}weight = 1024 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[512] sram[2272] length 128
// CHECK-NEXT: knpu.stop

// -----

// ERR: no knpu_model function in graph form (run knpu-import first)
func.func @other() {
  return
}

// -----

func.func @knpu_model(
    %in: tensor<1x2x2x1xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x2x2x1xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w = arith.constant dense<1> : tensor<1x1x1x1xi8>
  %b = arith.constant dense<0> : tensor<1xi32>
  %rq = arith.constant dense<1073741824> : tensor<1x2xi32>
  %0 = knpu.cpu @layer1(%in) {in_array = @act0, out_array = @act1}
      : (tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
  // ERR: cannot run on the NPU after a cpu layer
  %1 = knpu.conv2d %0, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x2x2x1xi8>, tensor<1x1x1x1xi8>, tensor<1xi32>,
         tensor<1x2xi32>) -> tensor<1x2x2x1xi8>
  return %1 : tensor<1x2x2x1xi8>
}

// -----

// A residual connection (M3): %0 feeds the second conv and the add, so it
// stays in SRAM (at 48) until the add, like the second conv's output (at
// 64); the add reads both there and writes its own buffer (at 0, where the
// second conv's constants were), never in place. Serially (SERIAL) the add
// loads both inputs from the arena.

func.func @knpu_model(
    %in: tensor<1x2x2x1xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x2x2x2xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = -128 : i64}) {
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
      pad = array<i64: 0, 0, 0, 0>, input_zp = 3 : i64, output_zp = -4 : i64,
      clamp = array<i64: -128, 127>, round = "single"}
      : (tensor<1x2x2x2xi8>, tensor<2x1x1x2xi8>, tensor<2xi32>,
         tensor<2x2xi32>) -> tensor<1x2x2x2xi8>
  %2 = knpu.add %0, %1 {input_zp = 3 : i64, input2_zp = -4 : i64,
      output_zp = -128 : i64, input_scale = array<i64: 1073741824, 0>,
      input2_scale = array<i64: 1359676495, -2>,
      output_scale = array<i64: 1127607261, -17>,
      clamp = array<i64: -128, 127>, round = "single"}
      : (tensor<1x2x2x2xi8>, tensor<1x2x2x2xi8>) -> tensor<1x2x2x2xi8>
  return %2 : tensor<1x2x2x2xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 1 "arena" size 24
// CHECK: knpu.io "output" region 1[16] shape [1, 2, 2, 2]
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[64] length 48
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[112] length 4
// CHECK-NEXT: knpu.dma "load" region 0[48] sram[0] length 48
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 112 : i64{{.*}}ofm = 48 : i64{{.*}}weight = 64 : i64}
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 48 : i64{{.*}}ofm = 64 : i64{{.*}}weight = 0 : i64}
// CHECK-NEXT: knpu.elementwise {clamp = array<i64: -128, 127>, ifm = 48 : i64, ifm2 = 64 : i64, ifm2_scale = array<i64: 1359676495, -2>, ifm2_zp = -4 : i64, ifm_scale = array<i64: 1073741824, 0>, ifm_zp = 3 : i64, ofm = 0 : i64, ofm_scale = array<i64: 1127607261, -17>, ofm_zp = -128 : i64, round = "single", shape = array<i64: 2, 2, 2>}
// CHECK-NEXT: knpu.dma "store" region 1[16] sram[0] length 8
// CHECK-NEXT: knpu.stop

// SERIAL: knpu.io "output" region 1[48] shape [1, 2, 2, 2] scale 1.000000e+00 zero_point -128
// SERIAL: knpu.dma "store" region 1[32] sram[64] length 8
// SERIAL-NEXT: knpu.dma "load" region 1[16] sram[0] length 8
// SERIAL-NEXT: knpu.dma "load" region 1[32] sram[16] length 8
// SERIAL-NEXT: knpu.elementwise {{{.*}}ifm = 0 : i64, ifm2 = 16 : i64{{.*}}ofm = 32 : i64
// SERIAL-NEXT: knpu.dma "store" region 1[48] sram[32] length 8
// SERIAL-NEXT: knpu.stop

// -----

// An add of a tensor to itself loads it once and reads it as both inputs.

func.func @knpu_model(
    %in: tensor<1x2x2x4xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x2x2x4xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %0 = knpu.add %in, %in {input_zp = 0 : i64, input2_zp = 0 : i64,
      output_zp = 0 : i64, input_scale = array<i64: 1073741824, 0>,
      input2_scale = array<i64: 1073741824, 0>,
      output_scale = array<i64: 1073741824, -19>,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x2x2x4xi8>, tensor<1x2x2x4xi8>) -> tensor<1x2x2x4xi8>
  return %0 : tensor<1x2x2x4xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[0] length 16
// CHECK-NEXT: knpu.elementwise {{{.*}}ifm = 0 : i64, ifm2 = 0 : i64{{.*}}ofm = 16 : i64
// CHECK-NEXT: knpu.dma "store" region 1[16] sram[16] length 16
// CHECK-NEXT: knpu.stop

// SERIAL: knpu.io "output" region 1[16] shape [1, 2, 2, 4]
// SERIAL: knpu.dma "load" region 1[0] sram[0] length 16
// SERIAL-NEXT: knpu.elementwise {{{.*}}ifm = 0 : i64, ifm2 = 0 : i64{{.*}}ofm = 16 : i64
// SERIAL-NEXT: knpu.dma "store" region 1[16] sram[16] length 16

// -----

// An add in two tiles of two rows, as knpu-tile will make them (M3.5): a
// tile reads the same rows of both inputs. The conv output goes through the
// arena because tiles read it. With overlap the inputs are double-buffered
// (rows 0-1 at 0 and 128, rows 2-3 at 64 and 192, loaded before the first
// tile's add); serially each tile loads its rows of both inputs.

func.func @knpu_model(
    %in: tensor<1x4x4x8xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x4x4x8xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w = arith.constant dense<1> : tensor<8x1x1x8xi8>
  %b = arith.constant dense<0> : tensor<8xi32>
  %rq = arith.constant dense<1073741824> : tensor<8x2xi32>
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x4x4x8xi8>, tensor<8x1x1x8xi8>, tensor<8xi32>,
         tensor<8x2xi32>) -> tensor<1x4x4x8xi8>
  %e = tensor.empty() : tensor<1x4x4x8xi8>
  %a0 = tensor.extract_slice %in[0, 0, 0, 0] [1, 2, 4, 8] [1, 1, 1, 1]
      : tensor<1x4x4x8xi8> to tensor<1x2x4x8xi8>
  %b0 = tensor.extract_slice %0[0, 0, 0, 0] [1, 2, 4, 8] [1, 1, 1, 1]
      : tensor<1x4x4x8xi8> to tensor<1x2x4x8xi8>
  %t0 = knpu.add %a0, %b0 {input_zp = 0 : i64, input2_zp = 0 : i64,
      output_zp = 0 : i64, input_scale = array<i64: 1073741824, 0>,
      input2_scale = array<i64: 1073741824, 0>,
      output_scale = array<i64: 1073741824, -19>,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x2x4x8xi8>, tensor<1x2x4x8xi8>) -> tensor<1x2x4x8xi8>
  %r0 = tensor.insert_slice %t0 into %e[0, 0, 0, 0] [1, 2, 4, 8] [1, 1, 1, 1]
      : tensor<1x2x4x8xi8> into tensor<1x4x4x8xi8>
  %a1 = tensor.extract_slice %in[0, 2, 0, 0] [1, 2, 4, 8] [1, 1, 1, 1]
      : tensor<1x4x4x8xi8> to tensor<1x2x4x8xi8>
  %b1 = tensor.extract_slice %0[0, 2, 0, 0] [1, 2, 4, 8] [1, 1, 1, 1]
      : tensor<1x4x4x8xi8> to tensor<1x2x4x8xi8>
  %t1 = knpu.add %a1, %b1 {input_zp = 0 : i64, input2_zp = 0 : i64,
      output_zp = 0 : i64, input_scale = array<i64: 1073741824, 0>,
      input2_scale = array<i64: 1073741824, 0>,
      output_scale = array<i64: 1073741824, -19>,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x2x4x8xi8>, tensor<1x2x4x8xi8>) -> tensor<1x2x4x8xi8>
  %r1 = tensor.insert_slice %t1 into %r0[0, 2, 0, 0] [1, 2, 4, 8] [1, 1, 1, 1]
      : tensor<1x2x4x8xi8> into tensor<1x4x4x8xi8>
  return %r1 : tensor<1x4x4x8xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 1 "arena" size 384
// CHECK-LABEL: func.func @knpu_model() {
// CHECK: knpu.dma "store" region 1[128] sram[288] length 128
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[0] length 64
// CHECK-NEXT: knpu.dma "load" region 1[128] sram[128] length 64
// CHECK-NEXT: knpu.dma "load" region 1[64] sram[64] length 64
// CHECK-NEXT: knpu.dma "load" region 1[192] sram[192] length 64
// CHECK-NEXT: knpu.elementwise {{{.*}}ifm = 0 : i64, ifm2 = 128 : i64{{.*}}ofm = 256 : i64{{.*}}shape = array<i64: 2, 4, 8>}
// CHECK-NEXT: knpu.dma "store" region 1[256] sram[256] length 64
// CHECK-NEXT: knpu.elementwise {{{.*}}ifm = 64 : i64, ifm2 = 192 : i64{{.*}}ofm = 320 : i64{{.*}}shape = array<i64: 2, 4, 8>}
// CHECK-NEXT: knpu.dma "store" region 1[320] sram[320] length 64
// CHECK-NEXT: knpu.stop

// SERIAL: knpu.io "output" region 1[256] shape [1, 4, 4, 8]
// SERIAL: knpu.dma "store" region 1[128] sram[288] length 128
// SERIAL-NEXT: knpu.dma "load" region 1[0] sram[0] length 64
// SERIAL-NEXT: knpu.dma "load" region 1[128] sram[64] length 64
// SERIAL-NEXT: knpu.elementwise {{{.*}}ifm = 0 : i64, ifm2 = 64 : i64{{.*}}ofm = 128 : i64
// SERIAL-NEXT: knpu.dma "store" region 1[256] sram[128] length 64
// SERIAL-NEXT: knpu.dma "load" region 1[64] sram[0] length 64
// SERIAL-NEXT: knpu.dma "load" region 1[192] sram[64] length 64
// SERIAL-NEXT: knpu.elementwise {{{.*}}ifm = 0 : i64, ifm2 = 64 : i64{{.*}}ofm = 128 : i64
// SERIAL-NEXT: knpu.dma "store" region 1[320] sram[128] length 64

// -----

// An add whose inputs and output (3 x 2,048 bytes) do not fit in 4 KiB is
// split by knpu-tile into tiles of the same rows of both inputs and the
// output: 10 rows with overlap, the next tile's inputs loaded in front of
// the current tile's addition; 21 rows without.

func.func @knpu_model(
    %in: tensor<1x32x32x2xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x32x32x2xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w = arith.constant dense<1> : tensor<2x1x1x2xi8>
  %b = arith.constant dense<0> : tensor<2xi32>
  %rq = arith.constant dense<1073741824> : tensor<2x2xi32>
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x32x32x2xi8>, tensor<2x1x1x2xi8>, tensor<2xi32>,
         tensor<2x2xi32>) -> tensor<1x32x32x2xi8>
  %1 = knpu.add %in, %0 {input_zp = 0 : i64, input2_zp = 0 : i64,
      output_zp = 0 : i64, input_scale = array<i64: 1073741824, 0>,
      input2_scale = array<i64: 1073741824, 0>,
      output_scale = array<i64: 1073741824, -19>,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x32x32x2xi8>, tensor<1x32x32x2xi8>) -> tensor<1x32x32x2xi8>
  return %1 : tensor<1x32x32x2xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.io "output" region 1[4096] shape [1, 32, 32, 2]
// CHECK: knpu.dma "store" region 1[3968] sram[1968] length 128
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[0] length 640
// CHECK-NEXT: knpu.dma "load" region 1[2048] sram[1280] length 640
// CHECK-NEXT: knpu.dma "load" region 1[640] sram[640] length 640
// CHECK-NEXT: knpu.dma "load" region 1[2688] sram[1920] length 640
// CHECK-NEXT: knpu.elementwise {{{.*}}ifm = 0 : i64, ifm2 = 1280 : i64{{.*}}ofm = 2560 : i64{{.*}}shape = array<i64: 10, 32, 2>}
// CHECK-NEXT: knpu.dma "store" region 1[4096] sram[2560] length 640
// CHECK-NEXT: knpu.dma "load" region 1[1280] sram[0] length 640
// CHECK-NEXT: knpu.dma "load" region 1[3328] sram[1280] length 640
// CHECK-NEXT: knpu.elementwise {{{.*}}ifm = 640 : i64, ifm2 = 1920 : i64{{.*}}ofm = 3200 : i64
// CHECK: knpu.elementwise {{{.*}}shape = array<i64: 2, 32, 2>}
// CHECK-NEXT: knpu.dma "store" region 1[6016] sram[3200] length 128
// CHECK-NEXT: knpu.stop

// SERIAL: knpu.io "output" region 1[4096] shape [1, 32, 32, 2]
// SERIAL: knpu.elementwise {{{.*}}shape = array<i64: 21, 32, 2>}
// SERIAL: knpu.elementwise {{{.*}}shape = array<i64: 11, 32, 2>}

// -----

func.func @knpu_model(
    %in: tensor<1x4x4x2xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x2x4x2xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %a = tensor.extract_slice %in[0, 0, 0, 0] [1, 2, 4, 2] [1, 1, 1, 1]
      : tensor<1x4x4x2xi8> to tensor<1x2x4x2xi8>
  %b = tensor.extract_slice %in[0, 2, 0, 0] [1, 2, 4, 2] [1, 1, 1, 1]
      : tensor<1x4x4x2xi8> to tensor<1x2x4x2xi8>
  // ERR: inputs must be read from the same row
  %0 = knpu.add %a, %b {input_zp = 0 : i64, input2_zp = 0 : i64,
      output_zp = 0 : i64, input_scale = array<i64: 1073741824, 0>,
      input2_scale = array<i64: 1073741824, 0>,
      output_scale = array<i64: 1073741824, -19>,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x2x4x2xi8>, tensor<1x2x4x2xi8>) -> tensor<1x2x4x2xi8>
  return %0 : tensor<1x2x4x2xi8>
}

// -----

// Output-channel tiles of six channels (see tile.mlir): the whole input
// (at 1888) is loaded once; each tile's constants, the channel range of the
// weights, bias and requantization table, are three DMAs into one of two
// buffers (at 0 and 944), the next tile's in front of the current tile's
// convolution; each output goes to its channels of the arena tensor with a
// 2-D DMA of 64 rows of 6 bytes, 32 bytes apart. Serially the tiles have
// thirteen channels and one buffer of each.

func.func @knpu_model(
    %in: tensor<1x8x8x16xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x8x8x32xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w = arith.constant dense<1> : tensor<32x3x3x16xi8>
  %b = arith.constant dense<0> : tensor<32xi32>
  %rq = arith.constant dense<1073741824> : tensor<32x2xi32>
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = 0 : i64,
      output_zp = 0 : i64, clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x16xi8>, tensor<32x3x3x16xi8>, tensor<32xi32>,
         tensor<32x2xi32>) -> tensor<1x8x8x32xi8>
  return %0 : tensor<1x8x8x32xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 0 "const" size 4992
// CHECK: knpu.io "output" region 1[1024] shape [1, 8, 8, 32]
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[0] length 864
// CHECK-NEXT: knpu.dma "load" region 0[4608] sram[864] length 24
// CHECK-NEXT: knpu.dma "load" region 0[4736] sram[896] length 48
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[1888] length 1024
// CHECK-NEXT: knpu.dma "load" region 0[864] sram[944] length 864
// CHECK-NEXT: knpu.dma "load" region 0[4632] sram[1808] length 24
// CHECK-NEXT: knpu.dma "load" region 0[4784] sram[1840] length 48
// CHECK-NEXT: knpu.conv {bias = 864 : i64{{.*}}ifm = 1888 : i64{{.*}}ofm = 2912 : i64, ofm_shape = array<i64: 8, 8, 6>{{.*}}rq_table = 896 : i64{{.*}}weight = 0 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[1024] sram[2912] length 6 count 64 stride 32
// CHECK-NEXT: knpu.dma "load" region 0[1728] sram[0] length 864
// CHECK-NEXT: knpu.dma "load" region 0[4656] sram[864] length 24
// CHECK-NEXT: knpu.dma "load" region 0[4832] sram[896] length 48
// CHECK-NEXT: knpu.conv {bias = 1808 : i64{{.*}}ofm = 3296 : i64{{.*}}weight = 944 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[1030] sram[3296] length 6 count 64 stride 32
// CHECK: knpu.dma "load" region 0[4320] sram[944] length 288
// CHECK-NEXT: knpu.dma "load" region 0[4728] sram[1232] length 8
// CHECK-NEXT: knpu.dma "load" region 0[4976] sram[1248] length 16
// CHECK: knpu.conv {bias = 1232 : i64{{.*}}ofm_shape = array<i64: 8, 8, 2>{{.*}}rq_table = 1248 : i64{{.*}}weight = 944 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[1054] sram[3296] length 2 count 64 stride 32
// CHECK-NEXT: knpu.stop

// SERIAL: knpu.io "output" region 1[1024] shape [1, 8, 8, 32]
// SERIAL: knpu.dma "load" region 0[0] sram[0] length 1872
// SERIAL-NEXT: knpu.dma "load" region 0[4608] sram[1872] length 52
// SERIAL-NEXT: knpu.dma "load" region 0[4736] sram[1936] length 104
// SERIAL-NEXT: knpu.dma "load" region 1[0] sram[2048] length 1024
// SERIAL-NEXT: knpu.conv {{{.*}}ofm_shape = array<i64: 8, 8, 13>
// SERIAL-NEXT: knpu.dma "store" region 1[1024] sram[3072] length 13 count 64 stride 32
// SERIAL-NEXT: knpu.dma "load" region 0[1872] sram[0] length 1872
// SERIAL: knpu.dma "store" region 1[1050] sram[3072] length 6 count 64 stride 32
// SERIAL-NEXT: knpu.stop

// -----

// Output-channel tiles on one-row tiles (see tile.mlir): the two tiles of a
// row read its input, loaded once into one of two buffers (at 864 and
// 2464), the next row's in front of the first tile's convolution; the
// constants of the two channel ranges (two buffers, at 0 and 432) are
// loaded again for every row. The six buffers fill the 4 KiB. Serially the
// tiles are one-row tiles of all channels.

func.func @knpu_model(
    %in: tensor<1x3x8x200xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x3x8x4xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w = arith.constant dense<1> : tensor<4x1x1x200xi8>
  %b = arith.constant dense<0> : tensor<4xi32>
  %rq = arith.constant dense<1073741824> : tensor<4x2xi32>
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64,
      output_zp = 0 : i64, clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x3x8x200xi8>, tensor<4x1x1x200xi8>, tensor<4xi32>,
         tensor<4x2xi32>) -> tensor<1x3x8x4xi8>
  return %0 : tensor<1x3x8x4xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 1 "arena" size 4896
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[0] length 400
// CHECK-NEXT: knpu.dma "load" region 0[800] sram[400] length 8
// CHECK-NEXT: knpu.dma "load" region 0[816] sram[416] length 16
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[864] length 1600
// CHECK-NEXT: knpu.dma "load" region 1[1600] sram[2464] length 1600
// CHECK-NEXT: knpu.dma "load" region 0[400] sram[432] length 400
// CHECK-NEXT: knpu.dma "load" region 0[808] sram[832] length 8
// CHECK-NEXT: knpu.dma "load" region 0[832] sram[848] length 16
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 864 : i64{{.*}}ofm = 4064 : i64{{.*}}weight = 0 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[4800] sram[4064] length 2 count 8 stride 4
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[0] length 400
// CHECK-NEXT: knpu.dma "load" region 0[800] sram[400] length 8
// CHECK-NEXT: knpu.dma "load" region 0[816] sram[416] length 16
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 864 : i64{{.*}}ofm = 4080 : i64{{.*}}weight = 432 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[4802] sram[4080] length 2 count 8 stride 4
// CHECK-NEXT: knpu.dma "load" region 1[3200] sram[864] length 1600
// CHECK: knpu.conv {{{.*}}ifm = 2464 : i64{{.*}}weight = 0 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[4832] sram[4064] length 2 count 8 stride 4
// CHECK: knpu.dma "store" region 1[4866] sram[4080] length 2 count 8 stride 4
// CHECK-NEXT: knpu.stop

// SERIAL: knpu.io "output" region 1[4800] shape [1, 3, 8, 4]
// SERIAL: knpu.dma "load" region 0[0] sram[0] length 848
// SERIAL-NEXT: knpu.dma "load" region 1[0] sram[848] length 1600
// SERIAL-NEXT: knpu.conv {{{.*}}ofm_shape = array<i64: 1, 8, 4>
// SERIAL-NEXT: knpu.dma "store" region 1[4800] sram[2448] length 32 count 1
// SERIAL-NEXT: knpu.dma "load" region 1[1600] sram[848] length 1600

// -----

// A tensor that a whole layer reads first and a tiled layer later (M3.6):
// the first conv's output stays in SRAM (at 0) until the second conv, which
// reads it there while it is stored to the arena (at 1536), and the add,
// split into row tiles (its inputs and output do not fit in 4 KiB at once),
// loads its rows from the arena.

func.func @knpu_model(
    %in: tensor<1x32x24x2xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x32x24x2xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w = arith.constant dense<1> : tensor<2x1x1x2xi8>
  %b = arith.constant dense<0> : tensor<2xi32>
  %rq = arith.constant dense<1073741824> : tensor<2x2xi32>
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x32x24x2xi8>, tensor<2x1x1x2xi8>, tensor<2xi32>,
         tensor<2x2xi32>) -> tensor<1x32x24x2xi8>
  %1 = knpu.conv2d %0, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x32x24x2xi8>, tensor<2x1x1x2xi8>, tensor<2xi32>,
         tensor<2x2xi32>) -> tensor<1x32x24x2xi8>
  %2 = knpu.add %0, %1 {input_zp = 0 : i64, input2_zp = 0 : i64,
      output_zp = 0 : i64, input_scale = array<i64: 1073741824, 0>,
      input2_scale = array<i64: 1073741824, 0>,
      output_scale = array<i64: 1073741824, -19>,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x32x24x2xi8>, tensor<1x32x24x2xi8>) -> tensor<1x32x24x2xi8>
  return %2 : tensor<1x32x24x2xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 1 "arena" size 6144
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[1584] length 48
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[1632] length 1536
// CHECK-NEXT: knpu.dma "load" region 0[48] sram[1536] length 48
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1632 : i64{{.*}}ofm = 0 : i64{{.*}}weight = 1584 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[1536] sram[0] length 1536
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 0 : i64{{.*}}ofm = 1584 : i64{{.*}}weight = 1536 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[3072] sram[1584] length 1536
// CHECK-NEXT: knpu.dma "load" region 1[1536] sram[0] length 672
// CHECK-NEXT: knpu.dma "load" region 1[3072] sram[1344] length 672

// -----

// Three convs whose 1,536-byte outputs, kept in SRAM, leave room in 4 KiB
// to load only one later layer's 864-byte constants ahead (M3.6). Dropping
// that load moves fewer bytes without overlap than sending an output
// through the arena and back, so the outputs stay in SRAM (the arena holds
// only the model input and output); the second conv's constants are loaded
// while the first runs, the third's after the second.

func.func @knpu_model(
    %in: tensor<1x8x8x1xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x8x8x24xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w1 = arith.constant dense<1> : tensor<24x1x1x1xi8>
  %w2 = arith.constant dense<1> : tensor<24x1x1x24xi8>
  %b = arith.constant dense<0> : tensor<24xi32>
  %rq = arith.constant dense<1073741824> : tensor<24x2xi32>
  %0 = knpu.conv2d %in, %w1, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x1xi8>, tensor<24x1x1x1xi8>, tensor<24xi32>,
         tensor<24x2xi32>) -> tensor<1x8x8x24xi8>
  %1 = knpu.conv2d %0, %w2, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x24xi8>, tensor<24x1x1x24xi8>, tensor<24xi32>,
         tensor<24x2xi32>) -> tensor<1x8x8x24xi8>
  %2 = knpu.conv2d %1, %w2, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x24xi8>, tensor<24x1x1x24xi8>, tensor<24xi32>,
         tensor<24x2xi32>) -> tensor<1x8x8x24xi8>
  return %2 : tensor<1x8x8x24xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 1 "arena" size 1600
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[1536] length 320
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[1856] length 64
// CHECK-NEXT: knpu.dma "load" region 0[320] sram[3072] length 864
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1856 : i64{{.*}}ofm = 0 : i64{{.*}}weight = 1536 : i64}
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 0 : i64{{.*}}ofm = 1536 : i64{{.*}}weight = 3072 : i64}
// CHECK-NEXT: knpu.dma "load" region 0[1184] sram[3072] length 864
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1536 : i64{{.*}}ofm = 0 : i64{{.*}}weight = 3072 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[64] sram[0] length 1536
// CHECK-NEXT: knpu.stop

// -----

// The same with 768-byte outputs and 1,408-byte constants (M3.6): again
// only one later layer's constants can be loaded ahead. Making the layout
// fit drops both loads ahead, one after the other; the second conv's is
// then restored, as dropping the third's alone leaves room for it.

func.func @knpu_model(
    %in: tensor<1x4x6x1xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x4x6x32xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w1 = arith.constant dense<1> : tensor<32x1x1x1xi8>
  %w2 = arith.constant dense<1> : tensor<32x1x1x32xi8>
  %b = arith.constant dense<0> : tensor<32xi32>
  %rq = arith.constant dense<1073741824> : tensor<32x2xi32>
  %0 = knpu.conv2d %in, %w1, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x4x6x1xi8>, tensor<32x1x1x1xi8>, tensor<32xi32>,
         tensor<32x2xi32>) -> tensor<1x4x6x32xi8>
  %1 = knpu.conv2d %0, %w2, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x4x6x32xi8>, tensor<32x1x1x32xi8>, tensor<32xi32>,
         tensor<32x2xi32>) -> tensor<1x4x6x32xi8>
  %2 = knpu.conv2d %1, %w2, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x4x6x32xi8>, tensor<32x1x1x32xi8>, tensor<32xi32>,
         tensor<32x2xi32>) -> tensor<1x4x6x32xi8>
  return %2 : tensor<1x4x6x32xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 1 "arena" size 800
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[2176] length 416
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[2592] length 24
// CHECK-NEXT: knpu.dma "load" region 0[416] sram[0] length 1408
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 2592 : i64{{.*}}ofm = 1408 : i64{{.*}}weight = 2176 : i64}
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1408 : i64{{.*}}ofm = 2176 : i64{{.*}}weight = 0 : i64}
// CHECK-NEXT: knpu.dma "load" region 0[1824] sram[0] length 1408
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 2176 : i64{{.*}}ofm = 1408 : i64{{.*}}weight = 0 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[32] sram[1408] length 768
// CHECK-NEXT: knpu.stop

// -----

// Four convs, 1 -> 24 -> 32 -> 24 -> 32 channels (M3.6). Placed in the
// order they were made, the buffers of single steps do not fit with every
// output kept in SRAM, and the third conv's output would go through the
// arena; placed largest first, they fit, and every output stays in SRAM.

func.func @knpu_model(
    %in: tensor<1x6x8x1xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64})
    -> (tensor<1x6x8x32xi8> {knpu.scale = 1.0 : f32, knpu.zero_point = 0 : i64}) {
  %w1 = arith.constant dense<1> : tensor<24x1x1x1xi8>
  %w2 = arith.constant dense<1> : tensor<32x1x1x24xi8>
  %w3 = arith.constant dense<1> : tensor<24x1x1x32xi8>
  %b24 = arith.constant dense<0> : tensor<24xi32>
  %rq24 = arith.constant dense<1073741824> : tensor<24x2xi32>
  %b32 = arith.constant dense<0> : tensor<32xi32>
  %rq32 = arith.constant dense<1073741824> : tensor<32x2xi32>
  %0 = knpu.conv2d %in, %w1, %b24, %rq24 {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x6x8x1xi8>, tensor<24x1x1x1xi8>, tensor<24xi32>,
         tensor<24x2xi32>) -> tensor<1x6x8x24xi8>
  %1 = knpu.conv2d %0, %w2, %b32, %rq32 {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x6x8x24xi8>, tensor<32x1x1x24xi8>, tensor<32xi32>,
         tensor<32x2xi32>) -> tensor<1x6x8x32xi8>
  %2 = knpu.conv2d %1, %w3, %b24, %rq24 {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x6x8x32xi8>, tensor<24x1x1x32xi8>, tensor<24xi32>,
         tensor<24x2xi32>) -> tensor<1x6x8x24xi8>
  %3 = knpu.conv2d %2, %w2, %b32, %rq32 {stride = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x6x8x24xi8>, tensor<32x1x1x24xi8>, tensor<32xi32>,
         tensor<32x2xi32>) -> tensor<1x6x8x32xi8>
  return %3 : tensor<1x6x8x32xi8>
}

// CHECK-LABEL: module attributes {knpu.sram_size = 4096 : i64}
// CHECK: knpu.region 1 "arena" size 1584
// CHECK-LABEL: func.func @knpu_model() {
// CHECK-NEXT: knpu.dma "load" region 0[0] sram[0] length 320
// CHECK-NEXT: knpu.dma "load" region 1[0] sram[320] length 48
// CHECK-NEXT: knpu.dma "load" region 0[320] sram[2688] length 1152
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 320 : i64{{.*}}ofm = 1536 : i64{{.*}}weight = 0 : i64}
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1536 : i64{{.*}}ofm = 0 : i64{{.*}}weight = 2688 : i64}
// CHECK-NEXT: knpu.dma "load" region 0[1472] sram[2688] length 1056
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 0 : i64{{.*}}ofm = 1536 : i64{{.*}}weight = 2688 : i64}
// CHECK-NEXT: knpu.dma "load" region 0[2528] sram[2688] length 1152
// CHECK-NEXT: knpu.conv {{{.*}}ifm = 1536 : i64{{.*}}ofm = 0 : i64{{.*}}weight = 2688 : i64}
// CHECK-NEXT: knpu.dma "store" region 1[48] sram[0] length 1536
// CHECK-NEXT: knpu.stop
