/*
 * SPDX-FileCopyrightText: 2026 Kiwon Kwon
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * The pragmas of M2 in the shape tflite2c writes them (docs/design.md 9.3):
 * a convolution, an average pooling, a reshape, a fully connected layer
 * and a cpu layer, through the pinned lcc release (lcc -S
 * -Wf-pragma=knpu), knpu-import and the NPU passes. The model file with
 * the CPU code is M2.5's (docs/plan.md).
 *
 * RUN: %lcc -Wf-pragma=knpu -S %s -o %t.mlir
 * RUN: knpu-opt --knpu-import %t.mlir | FileCheck %s
 * RUN: knpu-opt --knpu-import --knpu-tile=sram-size=4096 --knpu-schedule=sram-size=4096 --knpu-insert-waits %t.mlir | FileCheck %s --check-prefix=CMD
 *
 * The NPU layers, their constants and the helper only they call are gone;
 * the cpu layer stays with external linkage, with its helper and arrays.
 * CHECK-NOT:   @knpu_rescale
 * CHECK-NOT:   @layer1
 * CHECK:       func.func @knpu_halve(
 * CHECK:       func.func @layer5() attributes {llvm.linkage = #llvm.linkage<external>} {
 * CHECK-LABEL: func.func @knpu_model(
 * CHECK-SAME:  %[[IN:.*]]: tensor<1x6x6x1xi8> {knpu.scale = 0.00392156886 : f32, knpu.zero_point = -128 : i64})
 * CHECK-SAME:  -> (tensor<1x1x1x3xi8> {knpu.scale = 3.906250e-03 : f32, knpu.zero_point = -128 : i64})
 * CHECK:       %[[C:.*]] = knpu.conv2d %[[IN]], {{.*}} -> tensor<1x4x4x2xi8>
 * CHECK:       %[[P:.*]] = knpu.avgpool %[[C]] {clamp = array<i64: -128, 127>, kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x4x4x2xi8>) -> tensor<1x2x2x2xi8>
 * CHECK:       %[[R:.*]] = tensor.reshape %[[P]](%{{.*}}) : (tensor<1x2x2x2xi8>, tensor<4xindex>) -> tensor<1x1x1x8xi8>
 * CHECK:       %[[W:.*]] = arith.constant dense<{{\[\[\[\[}}1, 2, 3, 4, 5, 6, 7, 8]]], {{.*}}> : tensor<3x1x1x8xi8>
 * CHECK:       %[[RQ:.*]] = arith.constant dense<{{\[\[}}1073741824, -6], [1073741824, -6], [2147483647, -10]]> : tensor<3x2xi32>
 * CHECK:       %[[F:.*]] = knpu.conv2d %[[R]], %[[W]], %{{.*}}, %[[RQ]] {clamp = array<i64: -128, 127>, input_zp = -128 : i64, output_zp = 7 : i64, pad = array<i64: 0, 0, 0, 0>, round = "double", stride = array<i64: 1, 1>} : {{.*}} -> tensor<1x1x1x3xi8>
 * CHECK:       %[[Y:.*]] = knpu.cpu @layer5(%[[F]]) {in_array = @act4, out_array = @act5} : (tensor<1x1x1x3xi8>) -> tensor<1x1x1x3xi8>
 * CHECK:       return %[[Y]]
 * CHECK-DAG:   llvm.mlir.global external @act4()
 * CHECK-DAG:   llvm.mlir.global external @act5()
 *
 * In 4 KiB the conv and pool outputs stay in SRAM and the fc constants
 * load while the conv runs; the fc output goes to the arena for the CPU
 * steps.
 * CMD:         knpu.region 1 "arena" size 67
 * CMD:         knpu.io "input" region 1[0] shape [1, 6, 6, 1]
 * CMD:         knpu.io "output" region 1[64] shape [1, 1, 1, 3]
 * CMD-NEXT:    knpu.cpu_step "copy_in" @act4 arena[48] size 3
 * CMD-NEXT:    knpu.cpu_step "call" @layer5
 * CMD-NEXT:    knpu.cpu_step "copy_out" @act5 arena[64] size 3
 * CMD-LABEL:   func.func @knpu_model() {
 * CMD-NEXT:    knpu.dma "load" region 0[0] sram[112] length 64
 * CMD-NEXT:    knpu.dma "load" region 1[0] sram[176] length 36
 * CMD-NEXT:    knpu.dma "load" region 0[64] sram[0] length 72
 * CMD-NEXT:    knpu.dma_wait 1
 * CMD-NEXT:    knpu.conv {{{.*}}ifm = 176 : i64{{.*}}ofm = 80 : i64
 * CMD-NEXT:    knpu.pool {{{.*}}ifm = 80 : i64{{.*}}ofm = 112 : i64
 * CMD-NEXT:    knpu.dma_wait 0
 * CMD-NEXT:    knpu.conv {{{.*}}ifm = 112 : i64, ifm_shape = array<i64: 1, 1, 8>{{.*}}ofm = 80 : i64
 * CMD-NEXT:    knpu.kernel_wait 0
 * CMD-NEXT:    knpu.dma "store" region 1[48] sram[80] length 3
 * CMD-NEXT:    knpu.stop
 */

static const signed char L1_W[18] = {
	1, -2, 3, -4, 5, -6, 7, -8, 9,
	-9, 8, -7, 6, -5, 4, -3, 2, -1
};
static const int L1_B[2] = { 50, -50 };
static const int L1_M[2] = { 1518500250, 1073741824 };
static const int L1_S[2] = { -7, -8 };
static const signed char L4_W[24] = {
	1, 2, 3, 4, 5, 6, 7, 8,
	-1, -2, -3, -4, -5, -6, -7, -8,
	10, 0, -10, 0, 10, 0, -10, 0
};
static const int L4_B[3] = { 100, 0, -100 };
static const int L4_M[3] = { 1073741824, 1073741824, 2147483647 };
static const int L4_S[3] = { -6, -6, -10 };
static const float ACT0_SCALE = 0.0039215689f;
static const float ACT5_SCALE = 0.00390625f;
signed char act0[36];
signed char act1[32];
signed char act2[8];
signed char act3[8];
signed char act4[3];
signed char act5[3];

static int knpu_rescale(int x, int m, int s)
{
	long t = 31 - s;

	return (int)(((long)x * (long)m + (1L << (t - 1))) >> t);
}

static int knpu_halve(int x)
{
	return x / 2;
}

static void layer1(void)
{
#pragma knpu conv2d in=act0 out=act1 w=L1_W bias=L1_B mult=L1_M shift=L1_S in_shape=(6,6,1) out_shape=(4,4,2) kernel=(3,3) stride=(1,1) pad=(0,0,0,0) in_zp=-128 out_zp=-128 clamp=(-128,127)
	int oy, ox, oc, ky, kx, acc;

	for (oy = 0; oy < 4; oy++)
		for (ox = 0; ox < 4; ox++)
			for (oc = 0; oc < 2; oc++) {
				acc = 0;
				for (ky = 0; ky < 3; ky++)
					for (kx = 0; kx < 3; kx++)
						acc += L1_W[(oc * 3 + ky) * 3 + kx]
						    * (act0[(oy + ky) * 6 + ox + kx] - (-128));
				acc += L1_B[oc];
				acc = knpu_rescale(acc, L1_M[oc], L1_S[oc]) + (-128);
				if (acc < -128)
					acc = -128;
				if (acc > 127)
					acc = 127;
				act1[(oy * 4 + ox) * 2 + oc] = (signed char)acc;
			}
}

static void layer2(void)
{
#pragma knpu avgpool in=act1 out=act2 in_shape=(4,4,2) out_shape=(2,2,2) kernel=(2,2) stride=(2,2) pad=(0,0,0,0) zp=-128 clamp=(-128,127)
	int oy, ox, c, ky, kx, sum;

	for (oy = 0; oy < 2; oy++)
		for (ox = 0; ox < 2; ox++)
			for (c = 0; c < 2; c++) {
				sum = 0;
				for (ky = 0; ky < 2; ky++)
					for (kx = 0; kx < 2; kx++)
						sum += act1[((oy * 2 + ky) * 4 + ox * 2 + kx)
						    * 2 + c];
				act2[(oy * 2 + ox) * 2 + c] = (signed char)(sum > 0
				    ? (sum + 2) / 4 : (sum - 2) / 4);
			}
}

static void layer3(void)
{
#pragma knpu reshape in=act2 out=act3 in_shape=(2,2,2) out_shape=(1,1,8)
	int i;

	for (i = 0; i < 8; i++)
		act3[i] = act2[i];
}

static void layer4(void)
{
#pragma knpu fc in=act3 out=act4 w=L4_W bias=L4_B mult=L4_M shift=L4_S in_size=8 out_size=3 in_zp=-128 out_zp=7 clamp=(-128,127)
	int o, i, acc;

	for (o = 0; o < 3; o++) {
		acc = L4_B[o];
		for (i = 0; i < 8; i++)
			acc += L4_W[o * 8 + i] * (act3[i] - (-128));
		acc = knpu_rescale(acc, L4_M[o], L4_S[o]) + 7;
		if (acc < -128)
			acc = -128;
		if (acc > 127)
			acc = 127;
		act4[o] = (signed char)acc;
	}
}

static void layer5(void)
{
#pragma knpu cpu in=act4 out=act5 in_shape=(1,1,3) out_shape=(1,1,3) in_zp=7 out_zp=-128
	int i;

	for (i = 0; i < 3; i++)
		act5[i] = (signed char)knpu_halve(act4[i] - 7);
}

void knpu_model(void)
{
#pragma knpu model in=act0 out=act5 in_scale=ACT0_SCALE in_zp=-128 out_scale=ACT5_SCALE out_zp=-128 round="double"
	layer1();
	layer2();
	layer3();
	layer4();
	layer5();
}
