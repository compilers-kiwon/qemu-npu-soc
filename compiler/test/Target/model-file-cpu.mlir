// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: not knpu-translate --knpu-to-model -split-input-file %s 2> %t.err | od -A d -t x4 -v -w4 | FileCheck %s
// RUN: FileCheck %s --check-prefix=ERR < %t.err

// The CPU sections of format minor 1 (docs/model-format.md 9, 10). The
// writer copies knpu.cpu_code as it is and lays out the CPU steps with the
// index of the arena region and a table of NUL-terminated names. Five
// sections (no constant region): CMDS at 112 (4 bytes), REGIONS at 128,
// TENSORS at 144, CPU_CODE at 240 (6 bytes), CPU_STEPS at 256 (16 + 3 x 16
// + 17 bytes), end of file at 337.

module attributes {knpu.sram_size = 4096 : i64} {
  knpu.region 1 "arena" size 48
  knpu.io "input" region 1 [0] shape [1, 1, 1, 16] scale 1.0 zero_point 0
  knpu.io "output" region 1 [32] shape [1, 1, 1, 16] scale 0.5 zero_point -128
  knpu.cpu_step "copy_in" @act0 arena [0] size 16
  knpu.cpu_step "call" @layer1
  knpu.cpu_step "copy_out" @act1 arena [32] size 16
  knpu.cpu_code dense<[127, 69, 76, 70, 2, 1]> : tensor<6xi8>
  func.func @knpu_model() {
    knpu.stop
    return
  }
}

// Header: format_minor 1, five sections
// CHECK:      0000000 55504e4b
// CHECK-NEXT: 0000004 00010000
// CHECK-NEXT: 0000008 00001000
// CHECK-NEXT: 0000012 00050000
// CHECK-NEXT: 0000016 00000020
// Section table: kind, offset, size
// CHECK:      0000032 00000001
// CHECK-NEXT: 0000036 00000070
// CHECK-NEXT: 0000040 00000004
// CHECK:      0000048 00000002
// CHECK-NEXT: 0000052 00000080
// CHECK-NEXT: 0000056 00000010
// CHECK:      0000064 00000004
// CHECK-NEXT: 0000068 00000090
// CHECK-NEXT: 0000072 00000060
// CHECK:      0000080 00000005
// CHECK-NEXT: 0000084 000000f0
// CHECK-NEXT: 0000088 00000006
// CHECK:      0000096 00000006
// CHECK-NEXT: 0000100 00000100
// CHECK-NEXT: 0000104 00000051
// CMDS: OP_STOP
// CHECK:      0000112 00000000
// CPU_CODE: the six bytes, then zeros up to the next section
// CHECK:      0000240 464c457f
// CHECK-NEXT: 0000244 00000102
// CHECK-NEXT: 0000248 00000000
// CHECK-NEXT: 0000252 00000000
// CPU_STEPS: three steps
// CHECK-NEXT: 0000256 00000003
// CHECK-NEXT: 0000260 00000000
// CHECK-NEXT: 0000264 00000000
// CHECK-NEXT: 0000268 00000000
// COPY_IN, region 1, name 0 (act0), arena offset 0, 16 bytes
// CHECK-NEXT: 0000272 00000100
// CHECK-NEXT: 0000276 00000000
// CHECK-NEXT: 0000280 00000000
// CHECK-NEXT: 0000284 00000010
// CALL, name 5 (layer1)
// CHECK-NEXT: 0000288 00000001
// CHECK-NEXT: 0000292 00000005
// CHECK-NEXT: 0000296 00000000
// CHECK-NEXT: 0000300 00000000
// COPY_OUT, region 1, name 12 (act1), arena offset 32, 16 bytes
// CHECK-NEXT: 0000304 00000102
// CHECK-NEXT: 0000308 0000000c
// CHECK-NEXT: 0000312 00000020
// CHECK-NEXT: 0000316 00000010
// Names: "act0\0layer1\0act1\0". The section (81 bytes) ends with the last
// NUL; the newlines after it come from -split-input-file.
// CHECK-NEXT: 0000320 30746361
// CHECK-NEXT: 0000324 79616c00
// CHECK-NEXT: 0000328 00317265
// CHECK-NEXT: 0000332 31746361
// CHECK-NEXT: 0000336 {{[0-9a-f]+}}00

// -----

// CPU steps without the CPU code
// ERR: error: CPU code needs one knpu.cpu_code and at least one knpu.cpu_step
module attributes {knpu.sram_size = 4096 : i64} {
  knpu.region 1 "arena" size 16
  knpu.cpu_step "call" @layer1
  func.func @knpu_model() {
    knpu.stop
    return
  }
}

// -----

// The CPU code without CPU steps
// ERR: error: CPU code needs one knpu.cpu_code and at least one knpu.cpu_step
module attributes {knpu.sram_size = 4096 : i64} {
  knpu.region 1 "arena" size 16
  knpu.cpu_code dense<[127, 69, 76, 70]> : tensor<4xi8>
  func.func @knpu_model() {
    knpu.stop
    return
  }
}

// -----

// ERR: error: 'knpu.cpu_step' op copies outside the arena
module attributes {knpu.sram_size = 4096 : i64} {
  knpu.region 1 "arena" size 16
  knpu.cpu_step "copy_in" @act0 arena [8] size 16
  knpu.cpu_step "call" @layer1
  knpu.cpu_code dense<[127, 69, 76, 70]> : tensor<4xi8>
  func.func @knpu_model() {
    knpu.stop
    return
  }
}

// -----

// ERR: error: 'knpu.cpu_step' op symbol must be a C identifier
module attributes {knpu.sram_size = 4096 : i64} {
  knpu.region 1 "arena" size 16
  knpu.cpu_step "call" @"layer-1"
  knpu.cpu_code dense<[127, 69, 76, 70]> : tensor<4xi8>
  func.func @knpu_model() {
    knpu.stop
    return
  }
}

// -----

// ERR: error: 'knpu.region' op CPU steps need a single arena region
module attributes {knpu.sram_size = 4096 : i64} {
  knpu.region 1 "arena" size 16
  knpu.region 2 "arena" size 16
  knpu.cpu_step "call" @layer1
  knpu.cpu_code dense<[127, 69, 76, 70]> : tensor<4xi8>
  func.func @knpu_model() {
    knpu.stop
    return
  }
}
