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
