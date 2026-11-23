// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: knpu-opt %s | knpu-opt | FileCheck %s

// Graph and command operations print and parse back unchanged.

// CHECK-LABEL: func.func @graph
func.func @graph(%in: tensor<1x8x8x1xi8>, %w: tensor<4x3x3x1xi8>,
                 %b: tensor<4xi32>, %rq: tensor<4x2xi32>)
    -> tensor<1x8x8x4xi8> {
  // CHECK: knpu.conv2d %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {clamp = array<i64: -128, 127>, input_zp = -128 : i64, output_zp = 0 : i64, pad = array<i64: 1, 1, 1, 1>, round = "double", stride = array<i64: 1, 1>} : (tensor<1x8x8x1xi8>, tensor<4x3x3x1xi8>, tensor<4xi32>, tensor<4x2xi32>) -> tensor<1x8x8x4xi8>
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = -128 : i64,
      output_zp = 0 : i64, clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x1xi8>, tensor<4x3x3x1xi8>, tensor<4xi32>,
         tensor<4x2xi32>) -> tensor<1x8x8x4xi8>
  return %0 : tensor<1x8x8x4xi8>
}

module attributes {knpu.sram_size = 65536 : i64} {
  // CHECK: knpu.region 0 "const" size 4 content dense<[1, 2, 3, 4]> : tensor<4xi8>
  knpu.region 0 "const" size 4 content dense<[1, 2, 3, 4]> : tensor<4xi8>
  // CHECK: knpu.region 1 "arena" size 320
  knpu.region 1 "arena" size 320
  // CHECK: knpu.io "input" region 1[0] shape [1, 8, 8, 1] scale 1.000000e+00 zero_point -128
  knpu.io "input" region 1 [0] shape [1, 8, 8, 1] scale 1.0 zero_point -128
  // CHECK-LABEL: func.func @commands
  func.func @commands() {
    // CHECK: knpu.dma "load" region 1[0] sram[0] length 64 count 1 stride 0
    knpu.dma "load" region 1 [0] sram [0] length 64 count 1 stride 0
    // CHECK: knpu.dma "store" region 1[64] sram[512] length 32 count 8 stride 64
    knpu.dma "store" region 1 [64] sram [512] length 32 count 8 stride 64
    // CHECK: knpu.dma_wait 1
    knpu.dma_wait 1
    // CHECK: knpu.conv {bias = 320 : i64
    knpu.conv {ifm = 0 : i64, ifm_shape = array<i64: 8, 8, 1>,
               ifm_zp = -128 : i64, ofm = 512 : i64,
               ofm_shape = array<i64: 8, 8, 4>, ofm_zp = 0 : i64,
               weight = 256 : i64, bias = 320 : i64, rq_table = 336 : i64,
               kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>,
               pad = array<i64: 1, 1, 1, 1>, clamp = array<i64: -128, 127>,
               round = "single"}
    // CHECK: knpu.kernel_wait 0
    knpu.kernel_wait 0
    // CHECK: knpu.stop
    knpu.stop
    return
  }
}

// CHECK-LABEL: func.func @graph_m2
func.func @graph_m2(%in: tensor<1x4x4x2xi8>) -> tensor<1x2x2x2xi8> {
  // CHECK: knpu.avgpool %{{.*}} {clamp = array<i64: -100, 100>, kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x4x4x2xi8>) -> tensor<1x2x2x2xi8>
  %0 = knpu.avgpool %in {kernel = array<i64: 3, 3>, stride = array<i64: 2, 2>,
      pad = array<i64: 1, 1, 1, 1>, clamp = array<i64: -100, 100>}
      : (tensor<1x4x4x2xi8>) -> tensor<1x2x2x2xi8>
  // CHECK: knpu.cpu @layer2(%{{.*}}) {in_array = @act1, out_array = @act2} : (tensor<1x2x2x2xi8>) -> tensor<1x2x2x2xi8>
  %1 = knpu.cpu @layer2(%0) {in_array = @act1, out_array = @act2}
      : (tensor<1x2x2x2xi8>) -> tensor<1x2x2x2xi8>
  return %1 : tensor<1x2x2x2xi8>
}

module attributes {knpu.sram_size = 65536 : i64} {
  // CHECK: knpu.cpu_step "copy_in" @act1 arena[16] size 8
  knpu.cpu_step "copy_in" @act1 arena [16] size 8
  // CHECK: knpu.cpu_step "call" @layer2{{$}}
  knpu.cpu_step "call" @layer2
  // CHECK: knpu.cpu_step "copy_out" @act2 arena[32] size 8
  knpu.cpu_step "copy_out" @act2 arena [32] size 8
  // CHECK: knpu.cpu_code dense<[127, 69, 76, 70]> : tensor<4xi8>
  knpu.cpu_code dense<[127, 69, 76, 70]> : tensor<4xi8>
  // CHECK-LABEL: func.func @pool
  func.func @pool() {
    // CHECK: knpu.pool {clamp = array<i64: -100, 100>, ifm = 0 : i64, ifm_shape = array<i64: 4, 4, 2>, kernel = array<i64: 3, 3>, ofm = 32 : i64, ofm_shape = array<i64: 2, 2, 2>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}
    knpu.pool {ifm = 0 : i64, ifm_shape = array<i64: 4, 4, 2>, ofm = 32 : i64,
               ofm_shape = array<i64: 2, 2, 2>, kernel = array<i64: 3, 3>,
               stride = array<i64: 2, 2>, pad = array<i64: 1, 1, 1, 1>,
               clamp = array<i64: -100, 100>}
    return
  }
}
