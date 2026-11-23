// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: knpu-opt %s -split-input-file -verify-diagnostics

// The verifiers enforce the field ranges and the window conditions of
// docs/npu-spec.md 6.4.

func.func @output_shape(%in: tensor<1x8x8x1xi8>, %w: tensor<4x3x3x1xi8>,
                        %b: tensor<4xi32>, %rq: tensor<4x2xi32>) {
  // expected-error @+1 {{output must be 8x8 for this input}}
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x1xi8>, tensor<4x3x3x1xi8>, tensor<4xi32>,
         tensor<4x2xi32>) -> tensor<1x7x8x4xi8>
  return
}

// -----

func.func @pad_too_big(%in: tensor<1x8x8x1xi8>, %w: tensor<4x3x3x1xi8>,
                       %b: tensor<4xi32>, %rq: tensor<4x2xi32>) {
  // expected-error @+1 {{padding must be smaller than the kernel}}
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 3, 1, 1, 1>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x1xi8>, tensor<4x3x3x1xi8>, tensor<4xi32>,
         tensor<4x2xi32>) -> tensor<1x10x8x4xi8>
  return
}

// -----

func.func @channels(%in: tensor<1x8x8x2xi8>, %w: tensor<4x3x3x1xi8>,
                    %b: tensor<4xi32>, %rq: tensor<4x2xi32>) {
  // expected-error @+1 {{weight input channels must match the input}}
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "double"}
      : (tensor<1x8x8x2xi8>, tensor<4x3x3x1xi8>, tensor<4xi32>,
         tensor<4x2xi32>) -> tensor<1x8x8x4xi8>
  return
}

// -----

func.func @clamp(%in: tensor<1x8x8x1xi8>, %w: tensor<4x3x3x1xi8>,
                 %b: tensor<4xi32>, %rq: tensor<4x2xi32>) {
  // expected-error @+1 {{clamp must be (min, max) in -128..127 with min <= max}}
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: 10, -10>, round = "double"}
      : (tensor<1x8x8x1xi8>, tensor<4x3x3x1xi8>, tensor<4xi32>,
         tensor<4x2xi32>) -> tensor<1x8x8x4xi8>
  return
}

// -----

func.func @round(%in: tensor<1x8x8x1xi8>, %w: tensor<4x3x3x1xi8>,
                 %b: tensor<4xi32>, %rq: tensor<4x2xi32>) {
  // expected-error @+1 {{round must be "double" or "single"}}
  %0 = knpu.conv2d %in, %w, %b, %rq {stride = array<i64: 1, 1>,
      pad = array<i64: 1, 1, 1, 1>, input_zp = 0 : i64, output_zp = 0 : i64,
      clamp = array<i64: -128, 127>, round = "nearest"}
      : (tensor<1x8x8x1xi8>, tensor<4x3x3x1xi8>, tensor<4xi32>,
         tensor<4x2xi32>) -> tensor<1x8x8x4xi8>
  return
}

// -----

module attributes {knpu.sram_size = 4096 : i64} {
  func.func @dma_outside_sram() {
    // expected-error @+1 {{SRAM side [4000, 4100) is outside the 4096-byte SRAM}}
    knpu.dma "load" region 1 [0] sram [4000] length 100 count 1 stride 0
    return
  }
}

// -----

func.func @dma_region() {
  // expected-error @+1 {{region must be in 0..7}}
  knpu.dma "load" region 8 [0] sram [0] length 1 count 1 stride 0
  return
}

// -----

func.func @conv_overlap() {
  // expected-error @+1 {{OFM overlaps the IFM}}
  knpu.conv {ifm = 0 : i64, ifm_shape = array<i64: 8, 8, 1>,
             ifm_zp = 0 : i64, ofm = 32 : i64,
             ofm_shape = array<i64: 8, 8, 4>, ofm_zp = 0 : i64,
             weight = 512 : i64, bias = 640 : i64, rq_table = 656 : i64,
             kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>,
             pad = array<i64: 1, 1, 1, 1>, clamp = array<i64: -128, 127>,
             round = "double"}
  return
}

// -----

func.func @conv_bias_alignment() {
  // expected-error @+1 {{bias and requantization table offsets must be multiples of 4}}
  knpu.conv {ifm = 0 : i64, ifm_shape = array<i64: 8, 8, 1>,
             ifm_zp = 0 : i64, ofm = 1024 : i64,
             ofm_shape = array<i64: 8, 8, 4>, ofm_zp = 0 : i64,
             weight = 512 : i64, bias = 642 : i64, rq_table = 656 : i64,
             kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>,
             pad = array<i64: 1, 1, 1, 1>, clamp = array<i64: -128, 127>,
             round = "double"}
  return
}

// -----

func.func @wait_range() {
  // expected-error @+1 {{pending must be in 0..15}}
  knpu.dma_wait 16
  return
}

// -----

// expected-error @+1 {{a constant region needs its content}}
knpu.region 0 "const" size 4

// -----

// expected-error @+1 {{content must be a 1-D i8 tensor of size elements}}
knpu.region 0 "const" size 8 content dense<[1, 2, 3, 4]> : tensor<4xi8>

// -----

// expected-error @+1 {{scale must be positive and finite}}
knpu.io "input" region 1 [0] shape [1, 8, 8, 1] scale 0.0 zero_point 0

// -----

func.func @avgpool_channels(%in: tensor<1x4x4x2xi8>) {
  // expected-error @+1 {{output channels must match the input}}
  %0 = knpu.avgpool %in {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>,
      pad = array<i64: 0, 0, 0, 0>, clamp = array<i64: -128, 127>}
      : (tensor<1x4x4x2xi8>) -> tensor<1x2x2x3xi8>
  return
}

// -----

func.func @avgpool_output_shape(%in: tensor<1x5x5x2xi8>) {
  // expected-error @+1 {{output must be 2x2 for this input, kernel, stride and padding}}
  %0 = knpu.avgpool %in {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>,
      pad = array<i64: 0, 0, 0, 0>, clamp = array<i64: -128, 127>}
      : (tensor<1x5x5x2xi8>) -> tensor<1x3x3x2xi8>
  return
}

// -----

module attributes {knpu.sram_size = 4096 : i64} {
  func.func @pool_outside_sram() {
    // expected-error @+1 {{OFM [4090, 4098) is outside the 4096-byte SRAM}}
    knpu.pool {ifm = 0 : i64, ifm_shape = array<i64: 4, 4, 2>,
               ofm = 4090 : i64, ofm_shape = array<i64: 2, 2, 2>,
               kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>,
               pad = array<i64: 0, 0, 0, 0>, clamp = array<i64: -128, 127>}
    return
  }
}

// -----

func.func @pool_overlap() {
  // expected-error @+1 {{OFM overlaps the IFM}}
  knpu.pool {ifm = 0 : i64, ifm_shape = array<i64: 4, 4, 2>, ofm = 16 : i64,
             ofm_shape = array<i64: 2, 2, 2>, kernel = array<i64: 2, 2>,
             stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>,
             clamp = array<i64: -128, 127>}
  return
}

// -----

// expected-error @+1 {{kind must be "copy_in", "call" or "copy_out"}}
knpu.cpu_step "jump" @f

// -----

// expected-error @+1 {{a call has no arena offset or size}}
knpu.cpu_step "call" @f arena [0] size 4

// -----

// expected-error @+1 {{a copy needs an arena offset and a size}}
knpu.cpu_step "copy_in" @a
