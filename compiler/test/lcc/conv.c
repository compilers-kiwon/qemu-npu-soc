/*
 * SPDX-FileCopyrightText: 2026 Kiwon Kwon
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * One convolution layer in the shape tflite2c writes (docs/design.md 9.3),
 * through the pinned lcc release: lcc -S -Wf-pragma=knpu, then knpu-import
 * and the whole knpu-compile pipeline.
 *
 * RUN: %lcc -Wf-pragma=knpu -S %s -o %t.mlir
 * RUN: knpu-opt --knpu-import %t.mlir | FileCheck %s
 * RUN: knpu-compile --sram-size=4096 --emit-mlir %t.mlir | FileCheck %s --check-prefix=CMD
 * RUN: knpu-compile --sram-size=4096 %t.mlir -o %t.knpu
 *
 * CHECK-NOT:   @layer1
 * CHECK-NOT:   @knpu_rescale
 * CHECK-LABEL: func.func @knpu_model(
 * CHECK-SAME:  %[[IN:.*]]: tensor<1x4x4x2xi8> {knpu.scale = 0.00392156886 : f32, knpu.zero_point = -128 : i64})
 * CHECK-SAME:  -> (tensor<1x4x4x3xi8> {knpu.scale = 2.000000e-02 : f32, knpu.zero_point = -5 : i64})
 * CHECK-DAG:   %[[B:.*]] = arith.constant dense<[100, -200, 0]> : tensor<3xi32>
 * CHECK-DAG:   %[[RQ:.*]] = arith.constant dense<{{\[\[}}1518500250, -7], [1073741824, -8], [2147483647, 30]]> : tensor<3x2xi32>
 * CHECK:       knpu.conv2d %[[IN]], %{{.*}}, %[[B]], %[[RQ]] {clamp = array<i64: -5, 127>, input_zp = -128 : i64, output_zp = -5 : i64, pad = array<i64: 1, 1, 1, 1>, round = "single", stride = array<i64: 1, 1>} : {{.*}} -> tensor<1x4x4x3xi8>
 * CHECK-NOT:   llvm.mlir.global
 *
 * CMD:         knpu.region 0 "const" size 104
 * CMD:         knpu.region 1 "arena" size 80
 * CMD:         knpu.io "input" region 1[0] shape [1, 4, 4, 2]
 * CMD:         knpu.io "output" region 1[32] shape [1, 4, 4, 3]
 * CMD:         knpu.dma_wait 0
 * CMD-NEXT:    knpu.conv
 * CMD-NEXT:    knpu.kernel_wait 0
 * CMD-NEXT:    knpu.dma "store" region 1[32]
 * CMD-NEXT:    knpu.stop
 */

static const signed char L1_W[54] = {
	1, -2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
	-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
	-17, -18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127
};
static const int L1_B[3] = { 100, -200, 0 };
static const int L1_M[3] = { 1518500250, 1073741824, 2147483647 };
static const int L1_S[3] = { -7, -8, 30 };
static const float ACT0_SCALE = 0.0039215689f;
static const float ACT1_SCALE = 0.02f;
signed char act0[32];
signed char act1[48];

static int knpu_rescale(int x, int m, int s)
{
	long t = 31 - s;

	return (int)(((long)x * (long)m + (1L << (t - 1))) >> t);
}

static void layer1(void)
{
#pragma knpu conv2d in=act0 out=act1 w=L1_W bias=L1_B mult=L1_M shift=L1_S in_shape=(4,4,2) out_shape=(4,4,3) kernel=(3,3) stride=(1,1) pad=(1,1,1,1) in_zp=-128 out_zp=-5 clamp=(-5,127)
	int oy, ox, oc, ky, kx, ic, iy, ix, acc;

	for (oy = 0; oy < 4; oy++)
		for (ox = 0; ox < 4; ox++)
			for (oc = 0; oc < 3; oc++) {
				acc = 0;
				for (ky = 0; ky < 3; ky++) {
					iy = oy - 1 + ky;
					if (iy < 0 || iy >= 4)
						continue;
					for (kx = 0; kx < 3; kx++) {
						ix = ox - 1 + kx;
						if (ix < 0 || ix >= 4)
							continue;
						for (ic = 0; ic < 2; ic++)
							acc += L1_W[((oc * 3 + ky) * 3 + kx) * 2 + ic]
							    * (act0[(iy * 4 + ix) * 2 + ic] - (-128));
					}
				}
				acc += L1_B[oc];
				acc = knpu_rescale(acc, L1_M[oc], L1_S[oc]) + (-5);
				if (acc < -5)
					acc = -5;
				if (acc > 127)
					acc = 127;
				act1[(oy * 4 + ox) * 3 + oc] = (signed char)acc;
			}
}

void knpu_model(void)
{
#pragma knpu model in=act0 out=act1 in_scale=ACT0_SCALE in_zp=-128 out_scale=ACT1_SCALE out_zp=-5 round="single"
	layer1();
}
