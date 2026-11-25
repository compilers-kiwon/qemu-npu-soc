/*
 * SPDX-FileCopyrightText: 2026 Kiwon Kwon
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * knpu-compile schedules the tiled model with both SRAM placement rules of
 * knpu-schedule and keeps the one with fewer estimated cycles, the current
 * rule (placement=cost) on a tie (docs/design.md 9.4). Two models of 1x1
 * convolutions in 4 KiB, through the pinned lcc release; the constants are
 * written with macros only to keep this file short.
 * - By default one on which the rule before M3.6 (placement=largest) is
 *   faster: 4x6 activations, 1 -> 32 -> 32 -> 32 channels, an add of the
 *   first and the third convolution's outputs, then 32 -> 8.
 * - With -DTIE one on which both rules estimate the same cycles with
 *   different schedules: 4x4 activations, 1 -> 24 -> 32 -> 24 -> 32.
 *
 * RUN: %lcc -Wf-pragma=knpu -S %s -o %t.mlir
 * RUN: knpu-compile --sram-size=4096 --report %t.mlir -o %t.knpu | FileCheck %s
 * RUN: knpu-compile --sram-size=4096 --placement=cost --report %t.mlir -o %t.cost.knpu | FileCheck %s --check-prefix=COST
 * RUN: knpu-compile --sram-size=4096 --overlap=false --report %t.mlir -o %t.serial.knpu | FileCheck %s --check-prefix=SERIAL
 * RUN: knpu-compile --sram-size=4096 --emit-mlir %t.mlir -o %t.best.mlir
 * RUN: knpu-compile --sram-size=4096 --placement=largest --emit-mlir %t.mlir -o %t.largest.mlir
 * RUN: diff %t.best.mlir %t.largest.mlir
 * RUN: %lcc -Wf-pragma=knpu -DTIE -S %s -o %t.tie.mlir
 * RUN: knpu-compile --sram-size=4096 --report %t.tie.mlir -o %t.tie.knpu | FileCheck %s --check-prefix=TIE
 * RUN: knpu-compile --sram-size=4096 --emit-mlir %t.tie.mlir -o %t.tie.best.mlir
 * RUN: knpu-compile --sram-size=4096 --placement=cost --emit-mlir %t.tie.mlir -o %t.tie.cost.mlir
 * RUN: knpu-compile --sram-size=4096 --placement=largest --emit-mlir %t.tie.mlir -o %t.tie.largest.mlir
 * RUN: diff %t.tie.best.mlir %t.tie.cost.mlir
 * RUN: not diff %t.tie.best.mlir %t.tie.largest.mlir
 *
 * CHECK:       total cycles 435
 * CHECK-NEXT:  SRAM placement largest (cost: 505 cycles)
 *
 * COST:        total cycles 505
 * COST-NEXT:   SRAM placement cost{{$}}
 *
 * Without overlap there is nothing to give up and no rule to report.
 * SERIAL:      total cycles {{[0-9]+}}
 * SERIAL-NOT:  SRAM placement
 *
 * TIE:         total cycles 311
 * TIE-NEXT:    SRAM placement cost (largest: 311 cycles)
 */

#define ONE8 1, 1, 1, 1, 1, 1, 1, 1
#define ONE32 ONE8, ONE8, ONE8, ONE8
#define ONE256 ONE32, ONE32, ONE32, ONE32, ONE32, ONE32, ONE32, ONE32
#define ZERO8 0, 0, 0, 0, 0, 0, 0, 0
#define ZERO32 ZERO8, ZERO8, ZERO8, ZERO8
#define HALF8 1073741824, 1073741824, 1073741824, 1073741824, \
	1073741824, 1073741824, 1073741824, 1073741824
#define HALF32 HALF8, HALF8, HALF8, HALF8

static const float ACT0_SCALE = 1.0f;
static const float OUT_SCALE = 1.0f;

static int knpu_rescale(int x, int m, int s)
{
	long t = 31 - s;

	return (int)(((long)x * (long)m + (1L << (t - 1))) >> t);
}

static int knpu_clamp(int x)
{
	return x < -128 ? -128 : x > 127 ? 127 : x;
}

#ifndef TIE

static const signed char L1_W[32] = { ONE32 };
static const int L1_B[32] = { ZERO32 };
static const int L1_M[32] = { HALF32 };
static const int L1_S[32] = { ZERO32 };
static const signed char L2_W[1024] = { ONE256, ONE256, ONE256, ONE256 };
static const int L2_B[32] = { ZERO32 };
static const int L2_M[32] = { HALF32 };
static const int L2_S[32] = { ZERO32 };
static const signed char L3_W[1024] = { ONE256, ONE256, ONE256, ONE256 };
static const int L3_B[32] = { ZERO32 };
static const int L3_M[32] = { HALF32 };
static const int L3_S[32] = { ZERO32 };
static const signed char L5_W[256] = { ONE256 };
static const int L5_B[8] = { ZERO8 };
static const int L5_M[8] = { HALF8 };
static const int L5_S[8] = { ZERO8 };
signed char act0[24];
signed char act1[768];
signed char act2[768];
signed char act3[768];
signed char act4[768];
signed char act5[192];

static void layer1(void)
{
#pragma knpu conv2d in=act0 out=act1 w=L1_W bias=L1_B mult=L1_M shift=L1_S in_shape=(4,6,1) out_shape=(4,6,32) kernel=(1,1) stride=(1,1) pad=(0,0,0,0) in_zp=0 out_zp=0 clamp=(-128,127)
	int p, oc;

	for (p = 0; p < 24; p++)
		for (oc = 0; oc < 32; oc++)
			act1[p * 32 + oc] = (signed char)knpu_clamp(knpu_rescale(
			    L1_B[oc] + L1_W[oc] * act0[p], L1_M[oc], L1_S[oc]));
}

static void layer2(void)
{
#pragma knpu conv2d in=act1 out=act2 w=L2_W bias=L2_B mult=L2_M shift=L2_S in_shape=(4,6,32) out_shape=(4,6,32) kernel=(1,1) stride=(1,1) pad=(0,0,0,0) in_zp=0 out_zp=0 clamp=(-128,127)
	int p, oc, ic, acc;

	for (p = 0; p < 24; p++)
		for (oc = 0; oc < 32; oc++) {
			acc = L2_B[oc];
			for (ic = 0; ic < 32; ic++)
				acc += L2_W[oc * 32 + ic] * act1[p * 32 + ic];
			act2[p * 32 + oc] = (signed char)knpu_clamp(
			    knpu_rescale(acc, L2_M[oc], L2_S[oc]));
		}
}

static void layer3(void)
{
#pragma knpu conv2d in=act2 out=act3 w=L3_W bias=L3_B mult=L3_M shift=L3_S in_shape=(4,6,32) out_shape=(4,6,32) kernel=(1,1) stride=(1,1) pad=(0,0,0,0) in_zp=0 out_zp=0 clamp=(-128,127)
	int p, oc, ic, acc;

	for (p = 0; p < 24; p++)
		for (oc = 0; oc < 32; oc++) {
			acc = L3_B[oc];
			for (ic = 0; ic < 32; ic++)
				acc += L3_W[oc * 32 + ic] * act2[p * 32 + ic];
			act3[p * 32 + oc] = (signed char)knpu_clamp(
			    knpu_rescale(acc, L3_M[oc], L3_S[oc]));
		}
}

static void layer4(void)
{
#pragma knpu add in=act1 in2=act3 out=act4 shape=(4,6,32) in_zp=0 in2_zp=0 out_zp=0 in_mult=1073741824 in_shift=0 in2_mult=1073741824 in2_shift=0 out_mult=1073741824 out_shift=-19 clamp=(-128,127)
	int i, x1, x2;

	for (i = 0; i < 768; i++) {
		x1 = knpu_rescale(act1[i] * (1 << 20), 1073741824, 0);
		x2 = knpu_rescale(act3[i] * (1 << 20), 1073741824, 0);
		act4[i] = (signed char)knpu_clamp(
		    knpu_rescale(x1 + x2, 1073741824, -19));
	}
}

static void layer5(void)
{
#pragma knpu conv2d in=act4 out=act5 w=L5_W bias=L5_B mult=L5_M shift=L5_S in_shape=(4,6,32) out_shape=(4,6,8) kernel=(1,1) stride=(1,1) pad=(0,0,0,0) in_zp=0 out_zp=0 clamp=(-128,127)
	int p, oc, ic, acc;

	for (p = 0; p < 24; p++)
		for (oc = 0; oc < 8; oc++) {
			acc = L5_B[oc];
			for (ic = 0; ic < 32; ic++)
				acc += L5_W[oc * 32 + ic] * act4[p * 32 + ic];
			act5[p * 8 + oc] = (signed char)knpu_clamp(
			    knpu_rescale(acc, L5_M[oc], L5_S[oc]));
		}
}

void knpu_model(void)
{
#pragma knpu model in=act0 out=act5 in_scale=ACT0_SCALE in_zp=0 out_scale=OUT_SCALE out_zp=0 round="double"
	layer1();
	layer2();
	layer3();
	layer4();
	layer5();
}

#else

static const signed char L1_W[24] = { ONE8, ONE8, ONE8 };
static const int L1_B[24] = { ZERO8, ZERO8, ZERO8 };
static const int L1_M[24] = { HALF8, HALF8, HALF8 };
static const int L1_S[24] = { ZERO8, ZERO8, ZERO8 };
static const signed char L2_W[768] = { ONE256, ONE256, ONE256 };
static const int L2_B[32] = { ZERO32 };
static const int L2_M[32] = { HALF32 };
static const int L2_S[32] = { ZERO32 };
static const signed char L3_W[768] = { ONE256, ONE256, ONE256 };
static const int L3_B[24] = { ZERO8, ZERO8, ZERO8 };
static const int L3_M[24] = { HALF8, HALF8, HALF8 };
static const int L3_S[24] = { ZERO8, ZERO8, ZERO8 };
static const signed char L4_W[768] = { ONE256, ONE256, ONE256 };
static const int L4_B[32] = { ZERO32 };
static const int L4_M[32] = { HALF32 };
static const int L4_S[32] = { ZERO32 };
signed char act0[16];
signed char act1[384];
signed char act2[512];
signed char act3[384];
signed char act4[512];

static void layer1(void)
{
#pragma knpu conv2d in=act0 out=act1 w=L1_W bias=L1_B mult=L1_M shift=L1_S in_shape=(4,4,1) out_shape=(4,4,24) kernel=(1,1) stride=(1,1) pad=(0,0,0,0) in_zp=0 out_zp=0 clamp=(-128,127)
	int p, oc;

	for (p = 0; p < 16; p++)
		for (oc = 0; oc < 24; oc++)
			act1[p * 24 + oc] = (signed char)knpu_clamp(knpu_rescale(
			    L1_B[oc] + L1_W[oc] * act0[p], L1_M[oc], L1_S[oc]));
}

static void layer2(void)
{
#pragma knpu conv2d in=act1 out=act2 w=L2_W bias=L2_B mult=L2_M shift=L2_S in_shape=(4,4,24) out_shape=(4,4,32) kernel=(1,1) stride=(1,1) pad=(0,0,0,0) in_zp=0 out_zp=0 clamp=(-128,127)
	int p, oc, ic, acc;

	for (p = 0; p < 16; p++)
		for (oc = 0; oc < 32; oc++) {
			acc = L2_B[oc];
			for (ic = 0; ic < 24; ic++)
				acc += L2_W[oc * 24 + ic] * act1[p * 24 + ic];
			act2[p * 32 + oc] = (signed char)knpu_clamp(
			    knpu_rescale(acc, L2_M[oc], L2_S[oc]));
		}
}

static void layer3(void)
{
#pragma knpu conv2d in=act2 out=act3 w=L3_W bias=L3_B mult=L3_M shift=L3_S in_shape=(4,4,32) out_shape=(4,4,24) kernel=(1,1) stride=(1,1) pad=(0,0,0,0) in_zp=0 out_zp=0 clamp=(-128,127)
	int p, oc, ic, acc;

	for (p = 0; p < 16; p++)
		for (oc = 0; oc < 24; oc++) {
			acc = L3_B[oc];
			for (ic = 0; ic < 32; ic++)
				acc += L3_W[oc * 32 + ic] * act2[p * 32 + ic];
			act3[p * 24 + oc] = (signed char)knpu_clamp(
			    knpu_rescale(acc, L3_M[oc], L3_S[oc]));
		}
}

static void layer4(void)
{
#pragma knpu conv2d in=act3 out=act4 w=L4_W bias=L4_B mult=L4_M shift=L4_S in_shape=(4,4,24) out_shape=(4,4,32) kernel=(1,1) stride=(1,1) pad=(0,0,0,0) in_zp=0 out_zp=0 clamp=(-128,127)
	int p, oc, ic, acc;

	for (p = 0; p < 16; p++)
		for (oc = 0; oc < 32; oc++) {
			acc = L4_B[oc];
			for (ic = 0; ic < 24; ic++)
				acc += L4_W[oc * 24 + ic] * act3[p * 24 + ic];
			act4[p * 32 + oc] = (signed char)knpu_clamp(
			    knpu_rescale(acc, L4_M[oc], L4_S[oc]));
		}
}

void knpu_model(void)
{
#pragma knpu model in=act0 out=act4 in_scale=ACT0_SCALE in_zp=0 out_scale=OUT_SCALE out_zp=0 round="double"
	layer1();
	layer2();
	layer3();
	layer4();
}

#endif
