// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: knpu-opt %s -split-input-file --knpu-import -verify-diagnostics

// knpu-import reports pragma errors at the function that carries the
// pragma, naming the key.

// A key is missing.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
// expected-error @+1 {{#pragma knpu conv2d: missing key 'clamp'}}
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// An unknown key.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
// expected-error @+1 {{#pragma knpu conv2d: unknown key 'dilation'}}
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>, dilation = 2}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// The weight array has the wrong size for the shapes.

module {
llvm.mlir.global internal constant @W("\01\02") : !llvm.array<2 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
// expected-error @+1 {{#pragma knpu conv2d: 'w' must be a const array of 1 bytes}}
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// An activation is not a writable array of the shape's size.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
// expected-error @+1 {{#pragma knpu conv2d: 'in' must not be const}}
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @W, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @W, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// A multiplier is negative.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00\80") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
// expected-error @+1 {{#pragma knpu conv2d: 'mult' values must be in 0..2^31-1}}
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// A shift is out of range.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\1F\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
// expected-error @+1 {{#pragma knpu conv2d: 'shift' values must be in -31..30}}
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// A name that is not a global variable.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
// expected-error @+1 {{#pragma knpu conv2d: 'bias' names 'l1', which is not a global variable}}
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @l1, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// The layer's in_zp differs from the model's.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
// expected-error @+1 {{#pragma knpu conv2d: 'in_zp' differs from in_zp of the model}}
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 5, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// The scale is not a float.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
// expected-error @+1 {{#pragma knpu model: 'in_scale' must name a const float with an initializer}}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @W, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// The rounding mode is unknown.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
// expected-error @+1 {{#pragma knpu model: 'round' must be "double" or "single"}}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "nearest"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// The layer shapes break the window conditions.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<2 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
// expected-error @+1 {{output must be 2x1 for this input, kernel, stride and padding}}
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 2, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// An operation the NPU path does not have.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
// expected-error @+1 {{#pragma knpu add is not supported}}
func.func @l1() attributes {knpu.op = "add", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// The model function does more than call layers.

module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @knpu_model() attributes {knpu.op = "model", knpu.args = {in = @a0, out = @a1, in_scale = @SC, in_zp = 0, out_scale = @SC, out_zp = 0, round = "double"}} {
  // expected-error @+1 {{the model function may only call layer functions}}
  %c = arith.constant 0 : i32
  func.call @l1() : () -> ()
  func.return
}
}

// -----

// No model function.

// expected-error @+1 {{no function with #pragma knpu model}}
module {
llvm.mlir.global internal constant @W("\01") : !llvm.array<1 x i8>
llvm.mlir.global internal constant @B("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @M("\00\00\00@") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @S("\00\00\00\00") : !llvm.array<4 x i8>
llvm.mlir.global internal constant @SC(1.0 : f32) : f32
llvm.mlir.global external @a0() : !llvm.array<1 x i8>
llvm.mlir.global external @a1() : !llvm.array<1 x i8>
func.func @l1() attributes {knpu.op = "conv2d", knpu.args = {in = @a0, out = @a1, w = @W, bias = @B, mult = @M, shift = @S, in_shape = array<i64: 1, 1, 1>, out_shape = array<i64: 1, 1, 1>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0, out_zp = 0, clamp = array<i64: -128, 127>}} {
  func.return
}
}
