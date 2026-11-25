// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: knpu-opt %s -split-input-file --knpu-import | FileCheck %s

// Globals and attributes as lcc v4.2-llvm.3 writes them for #pragma knpu
// (lcc -S -Wf-pragma=knpu); the layer bodies are cut down to one helper
// call. Constants are little-endian byte strings, scales f32 globals.

module attributes {llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
llvm.mlir.global internal constant @"L1_W"("\01\FE\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F\10\11\12\FF\FE\FD\FC\FB\FA\F9\F8\F7\F6\F5\F4\F3\F2\F1\F0\EF\EE") {alignment = 1 : i64} : !llvm.array<36 x i8>
llvm.mlir.global internal constant @"L1_B"("d\00\00\008\FF\FF\FF") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"L1_M"("\9Ay\82Z\00\00\00@") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"L1_S"("\F9\FF\FF\FF\F8\FF\FF\FF") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"ACT0_SCALE"(0x3B808081 : f32) {alignment = 4 : i64} : f32
llvm.mlir.global internal constant @"ACT1_SCALE"(0x3CA3D70A : f32) {alignment = 4 : i64} : f32
func.func @"helper"(%a0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
  func.return %a0 : i32
}
func.func @"layer1"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "conv2d", knpu.args = {in = @"act0", out = @"act1", w = @"L1_W", bias = @"L1_B", mult = @"L1_M", shift = @"L1_S", in_shape = array<i64: 4, 4, 2>, out_shape = array<i64: 4, 4, 2>, kernel = array<i64: 3, 3>, stride = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, in_zp = -128 : i64, out_zp = -5 : i64, clamp = array<i64: -5, 127>}} {
  %c1 = arith.constant 1 : i32
  %r = func.call @"helper"(%c1) : (i32) -> i32
  func.return
}
func.func @"knpu_model"() attributes {knpu.op = "model", knpu.args = {in = @"act0", out = @"act1", in_scale = @"ACT0_SCALE", in_zp = -128 : i64, out_scale = @"ACT1_SCALE", out_zp = -5 : i64, round = "double"}} {
  func.call @"layer1"() : () -> ()
  cf.br ^L14
^L14:
  func.return
}
llvm.mlir.global external @"act1"() {alignment = 1 : i64} : !llvm.array<32 x i8> {
  %v1 = llvm.mlir.zero : !llvm.array<32 x i8>
  llvm.return %v1 : !llvm.array<32 x i8>
}
llvm.mlir.global external @"act0"() {alignment = 1 : i64} : !llvm.array<32 x i8> {
  %v1 = llvm.mlir.zero : !llvm.array<32 x i8>
  llvm.return %v1 : !llvm.array<32 x i8>
}
}

// The layer, its helper and every global the pragmas named are gone.
// CHECK-NOT: llvm.mlir.global
// CHECK-NOT: @helper
// CHECK-NOT: @layer1
// CHECK-LABEL: func.func @knpu_model(
// CHECK-SAME: %[[IN:.*]]: tensor<1x4x4x2xi8> {knpu.scale = 0.00392156886 : f32, knpu.zero_point = -128 : i64})
// CHECK-SAME: -> (tensor<1x4x4x2xi8> {knpu.scale = 2.000000e-02 : f32, knpu.zero_point = -5 : i64})
// CHECK: %[[W:.*]] = arith.constant dense<{{\[\[\[\[}}1, -2], [3, 4], {{.*}}, [-17, -18]]]]> : tensor<2x3x3x2xi8>
// CHECK: %[[B:.*]] = arith.constant dense<[100, -200]> : tensor<2xi32>
// CHECK: %[[RQ:.*]] = arith.constant dense<{{\[\[}}1518500250, -7], [1073741824, -8]]> : tensor<2x2xi32>
// CHECK: %[[OUT:.*]] = knpu.conv2d %[[IN]], %[[W]], %[[B]], %[[RQ]] {clamp = array<i64: -5, 127>, input_zp = -128 : i64, output_zp = -5 : i64, pad = array<i64: 1, 1, 1, 1>, round = "double", stride = array<i64: 1, 1>}
// CHECK: return %[[OUT]] : tensor<1x4x4x2xi8>
// CHECK-NOT: llvm.mlir.global

// -----

// Two layers in a chain; an all-zero bias; single rounding.

module {
llvm.mlir.global internal constant @"W1"("\01\01") {alignment = 1 : i64} : !llvm.array<2 x i8>
llvm.mlir.global internal constant @"W2"("\02\02\02\02") {alignment = 1 : i64} : !llvm.array<4 x i8>
llvm.mlir.global internal constant @"Z2"("\00\00\00\00\00\00\00\00") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"M2"("\00\00\00@\00\00\00@") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"S2"("\00\00\00\00\FF\FF\FF\FF") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"SC"(0x3F800000 : f32) {alignment = 4 : i64} : f32
func.func @"l1"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "conv2d", knpu.args = {in = @"a0", out = @"a1", w = @"W1", bias = @"Z2", mult = @"M2", shift = @"S2", in_shape = array<i64: 2, 2, 1>, out_shape = array<i64: 2, 2, 2>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0 : i64, out_zp = 3 : i64, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @"l2"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "conv2d", knpu.args = {in = @"a1", out = @"a2", w = @"W2", bias = @"Z2", mult = @"M2", shift = @"S2", in_shape = array<i64: 2, 2, 2>, out_shape = array<i64: 2, 2, 2>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 3 : i64, out_zp = 0 : i64, clamp = array<i64: 0, 127>}} {
  func.return
}
func.func @"knpu_model"() attributes {knpu.op = "model", knpu.args = {in = @"a0", out = @"a2", in_scale = @"SC", in_zp = 0 : i64, out_scale = @"SC", out_zp = 0 : i64, round = "single"}} {
  func.call @"l1"() : () -> ()
  func.call @"l2"() : () -> ()
  func.return
}
llvm.mlir.global external @"a0"() {alignment = 1 : i64} : !llvm.array<4 x i8>
llvm.mlir.global external @"a1"() {alignment = 1 : i64} : !llvm.array<8 x i8>
llvm.mlir.global external @"a2"() {alignment = 1 : i64} : !llvm.array<8 x i8>
}

// CHECK-LABEL: func.func @knpu_model(
// CHECK-SAME: %[[IN:.*]]: tensor<1x2x2x1xi8>
// CHECK-SAME: -> (tensor<1x2x2x2xi8> {knpu.scale = 1.000000e+00 : f32, knpu.zero_point = 0 : i64})
// CHECK-DAG: %[[Z:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK-DAG: %[[RQ:.*]] = arith.constant dense<{{\[\[}}1073741824, 0], [1073741824, -1]]> : tensor<2x2xi32>
// CHECK: %[[A1:.*]] = knpu.conv2d %[[IN]], %{{.*}}, %{{.*}}, %{{.*}} {{{.*}}output_zp = 3 : i64{{.*}}round = "single"{{.*}}} : {{.*}} -> tensor<1x2x2x2xi8>
// CHECK: %[[A2:.*]] = knpu.conv2d %[[A1]], %{{.*}}, %{{.*}}, %{{.*}} {clamp = array<i64: 0, 127>, input_zp = 3 : i64
// CHECK: return %[[A2]]

// -----

// avgpool, reshape, fc and a cpu layer. The fc layer becomes a 1x1
// knpu.conv2d of the reshaped 1x1x1x4 input; its weights [2, 4] are the
// OHWI tensor 2x1x1x4. The cpu layer becomes knpu.cpu; its function stays
// for the CPU code, with external linkage and without the pragma, and so do
// the helper it calls and the arrays it uses.

module {
llvm.mlir.global internal constant @"W3"("\01\02\03\04\FF\FE\FD\FC") {alignment = 1 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"B3"("\0A\00\00\00\F6\FF\FF\FF") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"M3"("\00\00\00@\00\00\00@") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"S3"("\FE\FF\FF\FF\FD\FF\FF\FF") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"SC"(0x3F800000 : f32) {alignment = 4 : i64} : f32
llvm.mlir.global internal constant @"SC4"(0x3B800000 : f32) {alignment = 4 : i64} : f32
func.func @"pool_helper"(%a0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
  func.return %a0 : i32
}
func.func @"cpu_helper"(%a0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
  func.return %a0 : i32
}
func.func @"l1"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "avgpool", knpu.args = {in = @"a0", out = @"a1", in_shape = array<i64: 4, 4, 1>, out_shape = array<i64: 2, 2, 1>, kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, zp = -128 : i64, clamp = array<i64: -128, 127>}} {
  %c1 = arith.constant 1 : i32
  %r = func.call @"pool_helper"(%c1) : (i32) -> i32
  func.return
}
func.func @"l2"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "reshape", knpu.args = {in = @"a1", out = @"a2", in_shape = array<i64: 2, 2, 1>, out_shape = array<i64: 1, 1, 4>}} {
  func.return
}
func.func @"l3"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "fc", knpu.args = {in = @"a2", out = @"a3", w = @"W3", bias = @"B3", mult = @"M3", shift = @"S3", in_size = 4 : i64, out_size = 2 : i64, in_zp = -128 : i64, out_zp = 5 : i64, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @"l4"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "cpu", knpu.args = {in = @"a3", out = @"a4", in_shape = array<i64: 1, 1, 2>, out_shape = array<i64: 1, 1, 2>, in_zp = 5 : i64, out_zp = -128 : i64}} {
  %p = llvm.mlir.addressof @"a3" : !llvm.ptr
  %q = llvm.mlir.addressof @"a4" : !llvm.ptr
  %v = llvm.load %p : !llvm.ptr -> i8
  llvm.store %v, %q : i8, !llvm.ptr
  %c1 = arith.constant 1 : i32
  %r = func.call @"cpu_helper"(%c1) : (i32) -> i32
  func.return
}
func.func @"knpu_model"() attributes {knpu.op = "model", knpu.args = {in = @"a0", out = @"a4", in_scale = @"SC", in_zp = -128 : i64, out_scale = @"SC4", out_zp = -128 : i64, round = "single"}} {
  func.call @"l1"() : () -> ()
  func.call @"l2"() : () -> ()
  func.call @"l3"() : () -> ()
  func.call @"l4"() : () -> ()
  func.return
}
llvm.mlir.global external @"a0"() {alignment = 1 : i64} : !llvm.array<16 x i8>
llvm.mlir.global external @"a1"() {alignment = 1 : i64} : !llvm.array<4 x i8>
llvm.mlir.global external @"a2"() {alignment = 1 : i64} : !llvm.array<4 x i8>
llvm.mlir.global external @"a3"() {alignment = 1 : i64} : !llvm.array<2 x i8>
llvm.mlir.global external @"a4"() {alignment = 1 : i64} : !llvm.array<2 x i8>
}

// CHECK-NOT: @pool_helper
// CHECK: func.func @cpu_helper(
// CHECK: func.func @l4() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-LABEL: func.func @knpu_model(
// CHECK-SAME: %[[IN:.*]]: tensor<1x4x4x1xi8> {knpu.scale = 1.000000e+00 : f32, knpu.zero_point = -128 : i64})
// CHECK-SAME: -> (tensor<1x1x1x2xi8> {knpu.scale = 3.906250e-03 : f32, knpu.zero_point = -128 : i64})
// CHECK: %[[P:.*]] = knpu.avgpool %[[IN]] {clamp = array<i64: -128, 127>, kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x4x4x1xi8>) -> tensor<1x2x2x1xi8>
// CHECK: %[[SHAPE:.*]] = arith.constant dense<[1, 1, 1, 4]> : tensor<4xindex>
// CHECK: %[[R:.*]] = tensor.reshape %[[P]](%[[SHAPE]]) : (tensor<1x2x2x1xi8>, tensor<4xindex>) -> tensor<1x1x1x4xi8>
// CHECK: %[[W:.*]] = arith.constant dense<{{\[\[\[\[}}1, 2, 3, 4]]], {{\[\[\[}}-1, -2, -3, -4]]]]> : tensor<2x1x1x4xi8>
// CHECK: %[[B:.*]] = arith.constant dense<[10, -10]> : tensor<2xi32>
// CHECK: %[[RQ:.*]] = arith.constant dense<{{\[\[}}1073741824, -2], [1073741824, -3]]> : tensor<2x2xi32>
// CHECK: %[[F:.*]] = knpu.conv2d %[[R]], %[[W]], %[[B]], %[[RQ]] {clamp = array<i64: -128, 127>, input_zp = -128 : i64, output_zp = 5 : i64, pad = array<i64: 0, 0, 0, 0>, round = "single", stride = array<i64: 1, 1>} : (tensor<1x1x1x4xi8>, tensor<2x1x1x4xi8>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x1x1x2xi8>
// CHECK: %[[C:.*]] = knpu.cpu @l4(%[[F]]) {in_array = @a3, out_array = @a4} : (tensor<1x1x1x2xi8>) -> tensor<1x1x1x2xi8>
// CHECK: return %[[C]]
// CHECK-NOT: @a0
// CHECK: llvm.mlir.global external @a3()
// CHECK-NEXT: llvm.mlir.global external @a4()

// -----

// An fc layer that reads a 1x2x2 activation as its vector of 4: the input
// is reshaped to 1x1x1x4.

module {
llvm.mlir.global internal constant @"W1"("\01\01") {alignment = 1 : i64} : !llvm.array<2 x i8>
llvm.mlir.global internal constant @"Z2"("\00\00\00\00\00\00\00\00") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"M2"("\00\00\00@\00\00\00@") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"S2"("\00\00\00\00\FF\FF\FF\FF") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"W2"("\01\02\03\04") {alignment = 1 : i64} : !llvm.array<4 x i8>
llvm.mlir.global internal constant @"Z1"("\00\00\00\00") {alignment = 4 : i64} : !llvm.array<4 x i8>
llvm.mlir.global internal constant @"M1"("\00\00\00@") {alignment = 4 : i64} : !llvm.array<4 x i8>
llvm.mlir.global internal constant @"S1"("\00\00\00\00") {alignment = 4 : i64} : !llvm.array<4 x i8>
llvm.mlir.global internal constant @"SC"(0x3F800000 : f32) {alignment = 4 : i64} : f32
func.func @"l1"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "conv2d", knpu.args = {in = @"a0", out = @"a1", w = @"W1", bias = @"Z2", mult = @"M2", shift = @"S2", in_shape = array<i64: 1, 2, 1>, out_shape = array<i64: 1, 2, 2>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0 : i64, out_zp = 3 : i64, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @"l2"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "fc", knpu.args = {in = @"a1", out = @"a2", w = @"W2", bias = @"Z1", mult = @"M1", shift = @"S1", in_size = 4 : i64, out_size = 1 : i64, in_zp = 3 : i64, out_zp = 0 : i64, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @"knpu_model"() attributes {knpu.op = "model", knpu.args = {in = @"a0", out = @"a2", in_scale = @"SC", in_zp = 0 : i64, out_scale = @"SC", out_zp = 0 : i64, round = "double"}} {
  func.call @"l1"() : () -> ()
  func.call @"l2"() : () -> ()
  func.return
}
llvm.mlir.global external @"a0"() {alignment = 1 : i64} : !llvm.array<2 x i8>
llvm.mlir.global external @"a1"() {alignment = 1 : i64} : !llvm.array<4 x i8>
llvm.mlir.global external @"a2"() {alignment = 1 : i64} : !llvm.array<1 x i8>
}

// CHECK-LABEL: func.func @knpu_model(
// CHECK: %[[A1:.*]] = knpu.conv2d {{.*}} -> tensor<1x1x2x2xi8>
// CHECK: %[[SHAPE:.*]] = arith.constant dense<[1, 1, 1, 4]> : tensor<4xindex>
// CHECK: %[[R:.*]] = tensor.reshape %[[A1]](%[[SHAPE]]) : (tensor<1x1x2x2xi8>, tensor<4xindex>) -> tensor<1x1x1x4xi8>
// CHECK: %[[A2:.*]] = knpu.conv2d %[[R]], {{.*}} : (tensor<1x1x1x4xi8>, tensor<1x1x1x4xi8>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x1x1x1xi8>
// CHECK: return %[[A2]]

// -----

// A residual connection: a1 feeds the second conv and the add, which adds
// the conv's output to it. The add keeps its zero points, scales and clamp;
// the rounding mode comes from the model.

module {
llvm.mlir.global internal constant @"W1"("\01\01") {alignment = 1 : i64} : !llvm.array<2 x i8>
llvm.mlir.global internal constant @"W2"("\02\02\02\02") {alignment = 1 : i64} : !llvm.array<4 x i8>
llvm.mlir.global internal constant @"Z2"("\00\00\00\00\00\00\00\00") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"M2"("\00\00\00@\00\00\00@") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"S2"("\00\00\00\00\FF\FF\FF\FF") {alignment = 4 : i64} : !llvm.array<8 x i8>
llvm.mlir.global internal constant @"SC"(0x3F800000 : f32) {alignment = 4 : i64} : f32
func.func @"l1"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "conv2d", knpu.args = {in = @"a0", out = @"a1", w = @"W1", bias = @"Z2", mult = @"M2", shift = @"S2", in_shape = array<i64: 2, 2, 1>, out_shape = array<i64: 2, 2, 2>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 0 : i64, out_zp = 3 : i64, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @"l2"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "conv2d", knpu.args = {in = @"a1", out = @"a2", w = @"W2", bias = @"Z2", mult = @"M2", shift = @"S2", in_shape = array<i64: 2, 2, 2>, out_shape = array<i64: 2, 2, 2>, kernel = array<i64: 1, 1>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, in_zp = 3 : i64, out_zp = -4 : i64, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @"l3"() attributes {llvm.linkage = #llvm.linkage<internal>, knpu.op = "add", knpu.args = {in = @"a1", in2 = @"a2", out = @"a3", shape = array<i64: 2, 2, 2>, in_zp = 3 : i64, in2_zp = -4 : i64, out_zp = -128 : i64, in_mult = 1073741824 : i64, in_shift = 0 : i64, in2_mult = 1359676495 : i64, in2_shift = -2 : i64, out_mult = 1127607261 : i64, out_shift = -17 : i64, clamp = array<i64: -128, 127>}} {
  func.return
}
func.func @"knpu_model"() attributes {knpu.op = "model", knpu.args = {in = @"a0", out = @"a3", in_scale = @"SC", in_zp = 0 : i64, out_scale = @"SC", out_zp = -128 : i64, round = "single"}} {
  func.call @"l1"() : () -> ()
  func.call @"l2"() : () -> ()
  func.call @"l3"() : () -> ()
  func.return
}
llvm.mlir.global external @"a0"() {alignment = 1 : i64} : !llvm.array<4 x i8>
llvm.mlir.global external @"a1"() {alignment = 1 : i64} : !llvm.array<8 x i8>
llvm.mlir.global external @"a2"() {alignment = 1 : i64} : !llvm.array<8 x i8>
llvm.mlir.global external @"a3"() {alignment = 1 : i64} : !llvm.array<8 x i8>
}

// CHECK-LABEL: func.func @knpu_model(
// CHECK-SAME: %[[IN:.*]]: tensor<1x2x2x1xi8>
// CHECK-SAME: -> (tensor<1x2x2x2xi8> {knpu.scale = 1.000000e+00 : f32, knpu.zero_point = -128 : i64})
// CHECK: %[[A1:.*]] = knpu.conv2d %[[IN]], {{.*}} -> tensor<1x2x2x2xi8>
// CHECK: %[[A2:.*]] = knpu.conv2d %[[A1]], {{.*}} -> tensor<1x2x2x2xi8>
// CHECK: %[[A3:.*]] = knpu.add %[[A1]], %[[A2]] {clamp = array<i64: -128, 127>, input2_scale = array<i64: 1359676495, -2>, input2_zp = -4 : i64, input_scale = array<i64: 1073741824, 0>, input_zp = 3 : i64, output_scale = array<i64: 1127607261, -17>, output_zp = -128 : i64, round = "single"} : (tensor<1x2x2x2xi8>, tensor<1x2x2x2xi8>) -> tensor<1x2x2x2xi8>
// CHECK: return %[[A3]]
// CHECK-NOT: llvm.mlir.global
