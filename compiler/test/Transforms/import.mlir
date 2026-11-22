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
