#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""tflite2c: write an int8 .tflite model as C89 for lcc and gcc.

Every layer becomes one parameterless static function whose body is the
C89 loop TFLite Micro's reference kernel runs, and whose first line is the
layer's `#pragma knpu` (docs/design.md 9.2, 9.3). knpu_model() calls the
layers in order and carries `#pragma knpu model`. Requantization multipliers
and shifts, and the softmax parameters, are computed with TFLite Micro's
formulas for the chosen rounding mode.

CONV_2D, AVERAGE_POOL_2D, RESHAPE and FULLY_CONNECTED are NPU layers
(pragmas conv2d, avgpool, reshape and fc). SOFTMAX is a CPU layer (pragma
cpu): its body is TFLite Micro's int8 Softmax with the gemmlowp fixed-point
functions it uses, ported to C89. CPU layers must come after the last NPU
layer.

    tflite2c.py --schema schema_py_generated.py --round double \\
        model.tflite -o model.c [--main main.c] [--npu-only]

The schema is TFLite Micro's flatc-generated Python module
(tensorflow/lite/micro/python/schema_py_generated.py); it needs the PyPI
package flatbuffers. --main writes a separate C file for gcc that runs the
model, `main INPUT OUTPUT [NPU_OUTPUT]`: it reads the input bytes, calls
knpu_model() and writes the output bytes, and with NPU_OUTPUT also the
input of the first CPU layer, which is the output of the NPU layers (the
logits before a softmax). --npu-only leaves the CPU layers out, so the
output of the last NPU layer becomes the model output.
"""

import argparse
import importlib.util
import math
import struct
import sys

INT8_MIN, INT8_MAX = -128, 127

# Softmax (reference/softmax.h): Q5.26 input differences, Q12.19 sums
SCALED_DIFF_INTEGER_BITS = 5
ACCUMULATION_INTEGER_BITS = 12


def load_schema(path):
    spec = importlib.util.spec_from_file_location("schema_py_generated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fail(msg):
    sys.exit(f"tflite2c: {msg}")


# --------------------------------------------------------------------------
# TFLite Micro arithmetic
# --------------------------------------------------------------------------

def round_half_away(x):
    """TfLiteRound (std::round) of a double."""
    f = math.floor(abs(x))
    r = f + 1 if abs(x) - f >= 0.5 else f
    return int(math.copysign(r, x))


def to_f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def quantize_multiplier(m, single_rounding):
    """QuantizeMultiplier (kernels/internal/quantization_util.cc)."""
    if m == 0.0:
        return 0, 0
    q, shift = math.frexp(m)
    q_fixed = round_half_away(q * (1 << 31))
    assert q_fixed <= (1 << 31)
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    assert q_fixed <= 2**31 - 1
    if shift < -31:
        shift, q_fixed = 0, 0
    if single_rounding and shift > 30:
        shift, q_fixed = 30, 2**31 - 1
    return q_fixed, shift


def softmax_scaling(beta, input_scale, single_rounding):
    """CalculateSoftmaxParams (kernels/softmax_common.cc) for int8: the
    input multiplier and left shift of PreprocessSoftmaxScaling and diff_min
    from CalculateInputRadius (kernels/internal/quantization_util.cc)."""
    max_real = (1 << 30) - 1.0 if single_rounding else (1 << 31) - 1.0
    real = min(beta * input_scale * (1 << (31 - SCALED_DIFF_INTEGER_BITS)),
               max_real)
    if not real > 1.0:
        fail("SOFTMAX input scale times beta is too small")
    m, shift = quantize_multiplier(real, single_rounding)
    if shift < 0:
        fail("SOFTMAX input multiplier must not shift right")
    radius = (1.0 * ((1 << SCALED_DIFF_INTEGER_BITS) - 1)
              * (1 << (31 - SCALED_DIFF_INTEGER_BITS)) / (1 << shift))
    return m, shift, -math.floor(radius)


def quantize(scale, zero_point, f):
    """Quantize() of kernels/kernel_util.cc: float arithmetic."""
    tmp = round_half_away(to_f32(to_f32(f) / scale))
    return zero_point + tmp


def activation_range(activation, scale, zero_point, schema):
    """CalculateActivationRangeQuantized for int8 outputs."""
    act = schema.ActivationFunctionType
    qmin, qmax = INT8_MIN, INT8_MAX
    if activation == act.NONE:
        return qmin, qmax
    if activation == act.RELU:
        return max(qmin, quantize(scale, zero_point, 0.0)), qmax
    if activation == act.RELU6:
        return (max(qmin, quantize(scale, zero_point, 0.0)),
                min(qmax, quantize(scale, zero_point, 6.0)))
    if activation == act.RELU_N1_TO_1:
        return (max(qmin, quantize(scale, zero_point, -1.0)),
                min(qmax, quantize(scale, zero_point, 1.0)))
    fail(f"unsupported fused activation {activation}")


def padding(schema, mode, size, k, stride):
    """ComputePaddingHeightWidth: (output size, before, after)."""
    if mode == schema.Padding.SAME:
        out = (size + stride - 1) // stride
    elif mode == schema.Padding.VALID:
        out = (size + stride - k) // stride
    else:
        fail(f"unsupported padding {mode}")
    total = max((out - 1) * stride + k - size, 0)
    return out, total // 2, total - total // 2


# --------------------------------------------------------------------------
# Model reading
# --------------------------------------------------------------------------

class Tensor:
    # Element accessors rather than ...AsNumpy(), which needs numpy
    def __init__(self, model, sub, index, schema):
        t = sub.Tensors(index)
        self.index = index
        self.shape = [t.Shape(j) for j in range(t.ShapeLength())]
        self.type = t.Type()
        q = t.Quantization()
        self.scales = [q.Scale(j) for j in range(q.ScaleLength())] if q else []
        self.zero_points = ([q.ZeroPoint(j)
                             for j in range(q.ZeroPointLength())]
                            if q else [])
        buf = model.Buffers(t.Buffer())
        n = buf.DataLength()
        self.data = bytes(buf.Data(j) for j in range(n)) if n else None

    def int8(self):
        return list(struct.unpack(f"{len(self.data)}b", self.data))

    def int32(self):
        return list(struct.unpack(f"<{len(self.data) // 4}i", self.data))

    def size(self):
        """Number of elements."""
        return math.prod(self.shape)

    def hwc(self, what):
        """(H, W, C) of a 1xHxWxC tensor; a 1xN tensor is (1, 1, N)."""
        if len(self.shape) == 4 and self.shape[0] == 1:
            return tuple(self.shape[1:])
        if len(self.shape) == 2 and self.shape[0] == 1:
            return 1, 1, self.shape[1]
        fail(f"{what} must be 1xHxWxC or 1xN")


def builtin_options(op, cls):
    opts = cls()
    table = op.BuiltinOptions()
    opts.Init(table.Bytes, table.Pos)
    return opts


class Conv:
    """One CONV_2D with everything the C and the pragma need."""

    cpu = False

    def __init__(self, model, sub, op, schema, round_mode):
        tensors = [Tensor(model, sub, op.Inputs(i), schema)
                   for i in range(op.InputsLength())]
        if len(tensors) != 3:
            fail("CONV_2D needs input, filter and bias")
        inp, filt, bias = tensors
        out = Tensor(model, sub, op.Outputs(0), schema)
        opts = builtin_options(op, schema.Conv2DOptions)

        tt = schema.TensorType
        if (inp.type, filt.type, bias.type, out.type) != (
                tt.INT8, tt.INT8, tt.INT32, tt.INT8):
            fail("CONV_2D must be int8 with an int32 bias")
        if (opts.DilationHFactor(), opts.DilationWFactor()) != (1, 1):
            fail("CONV_2D dilation must be 1")
        if len(inp.shape) != 4 or inp.shape[0] != 1:
            fail("CONV_2D input must be 1xHxWxC")
        if filt.data is None or bias.data is None:
            fail("CONV_2D filter and bias must be constant")
        if any(z != 0 for z in filt.zero_points):
            fail("CONV_2D filter zero points must be 0")

        self.input, self.output = inp, out
        _, self.ih, self.iw, self.ic = inp.shape
        self.oc, self.kh, self.kw, kic = filt.shape
        if kic != self.ic:
            fail("CONV_2D filter channels differ from the input")
        self.sh, self.sw = opts.StrideH(), opts.StrideW()
        self.oh, self.pt, self.pb = padding(schema, opts.Padding(), self.ih,
                                            self.kh, self.sh)
        self.ow, self.pl, self.pr = padding(schema, opts.Padding(), self.iw,
                                            self.kw, self.sw)
        if out.shape != [1, self.oh, self.ow, self.oc]:
            fail("CONV_2D output shape differs from its padding")

        self.in_zp = inp.zero_points[0]
        self.out_zp = out.zero_points[0]
        self.weights = filt.int8()
        self.bias = bias.int32()

        # PopulateConvolutionQuantizationParams (kernels/kernel_util.cc)
        single = round_mode == "single"
        in_scale, out_scale = float(inp.scales[0]), float(out.scales[0])
        self.mult, self.shift = [], []
        for c in range(self.oc):
            f = float(filt.scales[c] if len(filt.scales) > 1
                      else filt.scales[0])
            m, s = quantize_multiplier(in_scale * f / out_scale, single)
            self.mult.append(m)
            self.shift.append(s)
        self.clamp = activation_range(opts.FusedActivationFunction(),
                                      out_scale, self.out_zp, schema)


class AvgPool:
    """One AVERAGE_POOL_2D."""

    cpu = False

    def __init__(self, model, sub, op, schema, round_mode):
        inp = Tensor(model, sub, op.Inputs(0), schema)
        out = Tensor(model, sub, op.Outputs(0), schema)
        opts = builtin_options(op, schema.Pool2DOptions)

        tt = schema.TensorType
        if (inp.type, out.type) != (tt.INT8, tt.INT8):
            fail("AVERAGE_POOL_2D must be int8")
        if len(inp.shape) != 4 or inp.shape[0] != 1:
            fail("AVERAGE_POOL_2D input must be 1xHxWxC")
        # PoolingPrepare (kernels/pooling_common.cc)
        if (abs(float(inp.scales[0]) - float(out.scales[0])) > 1.0e-6
                or inp.zero_points[0] != out.zero_points[0]):
            fail("AVERAGE_POOL_2D input and output quantization must match")

        self.input, self.output = inp, out
        _, self.ih, self.iw, self.c = inp.shape
        self.kh, self.kw = opts.FilterHeight(), opts.FilterWidth()
        self.sh, self.sw = opts.StrideH(), opts.StrideW()
        self.oh, self.pt, self.pb = padding(schema, opts.Padding(), self.ih,
                                            self.kh, self.sh)
        self.ow, self.pl, self.pr = padding(schema, opts.Padding(), self.iw,
                                            self.kw, self.sw)
        if out.shape != [1, self.oh, self.ow, self.c]:
            fail("AVERAGE_POOL_2D output shape differs from its padding")
        self.zp = inp.zero_points[0]
        self.clamp = activation_range(opts.FusedActivationFunction(),
                                      float(out.scales[0]),
                                      out.zero_points[0], schema)


class Reshape:
    """One RESHAPE: the same bytes in another shape."""

    cpu = False

    def __init__(self, model, sub, op, schema, round_mode):
        inp = Tensor(model, sub, op.Inputs(0), schema)
        out = Tensor(model, sub, op.Outputs(0), schema)
        tt = schema.TensorType
        if (inp.type, out.type) != (tt.INT8, tt.INT8):
            fail("RESHAPE must be int8")
        if inp.size() != out.size():
            fail("RESHAPE must keep the number of elements")
        self.input, self.output = inp, out
        self.in_shape = inp.hwc("RESHAPE input")
        self.out_shape = out.hwc("RESHAPE output")


class FullyConnected:
    """One FULLY_CONNECTED: per-channel or per-tensor, bias optional."""

    cpu = False

    def __init__(self, model, sub, op, schema, round_mode):
        n = op.InputsLength()
        if n not in (2, 3):
            fail("FULLY_CONNECTED needs input, weights and an optional bias")
        inp = Tensor(model, sub, op.Inputs(0), schema)
        filt = Tensor(model, sub, op.Inputs(1), schema)
        bias = (Tensor(model, sub, op.Inputs(2), schema)
                if n == 3 and op.Inputs(2) >= 0 else None)
        out = Tensor(model, sub, op.Outputs(0), schema)
        opts = builtin_options(op, schema.FullyConnectedOptions)

        tt = schema.TensorType
        if ((inp.type, filt.type, out.type) != (tt.INT8, tt.INT8, tt.INT8)
                or (bias is not None and bias.type != tt.INT32)):
            fail("FULLY_CONNECTED must be int8 with an int32 bias")
        if (opts.WeightsFormat()
                != schema.FullyConnectedOptionsWeightsFormat.DEFAULT):
            fail("FULLY_CONNECTED weights must be in the default format")
        if filt.data is None or (bias is not None and bias.data is None):
            fail("FULLY_CONNECTED weights and bias must be constant")
        if any(z != 0 for z in filt.zero_points):
            fail("FULLY_CONNECTED weight zero points must be 0")
        if len(filt.shape) != 2:
            fail("FULLY_CONNECTED weights must be [outputs, inputs]")
        self.out_size, self.in_size = filt.shape
        if inp.size() != self.in_size or out.size() != self.out_size:
            fail("FULLY_CONNECTED must take one input vector")
        if len(filt.scales) not in (1, self.out_size):
            fail("FULLY_CONNECTED needs one weight scale or one per output")

        self.input, self.output = inp, out
        self.in_zp = inp.zero_points[0]
        self.out_zp = out.zero_points[0]
        self.weights = filt.int8()
        # TFLite Micro skips a missing bias; adding zeros is the same
        self.bias = bias.int32() if bias is not None else [0] * self.out_size

        # CalculateOpDataFullyConnected (kernels/fully_connected_common.cc):
        # the same multiplier for per-channel and per-tensor weights
        single = round_mode == "single"
        in_scale, out_scale = float(inp.scales[0]), float(out.scales[0])
        self.mult, self.shift = [], []
        for o in range(self.out_size):
            f = float(filt.scales[o] if len(filt.scales) > 1
                      else filt.scales[0])
            m, s = quantize_multiplier(in_scale * f / out_scale, single)
            self.mult.append(m)
            self.shift.append(s)
        self.clamp = activation_range(opts.FusedActivationFunction(),
                                      out_scale, self.out_zp, schema)


class Softmax:
    """One SOFTMAX: a CPU layer."""

    cpu = True

    def __init__(self, model, sub, op, schema, round_mode):
        inp = Tensor(model, sub, op.Inputs(0), schema)
        out = Tensor(model, sub, op.Outputs(0), schema)
        opts = builtin_options(op, schema.SoftmaxOptions)

        tt = schema.TensorType
        if (inp.type, out.type) != (tt.INT8, tt.INT8):
            fail("SOFTMAX must be int8")
        if out.shape != inp.shape:
            fail("SOFTMAX output shape differs from its input")
        # CalculateSoftmaxParams (kernels/softmax_common.cc)
        if out.zero_points[0] != -128 or float(out.scales[0]) != 1.0 / 256:
            fail("SOFTMAX output must have scale 1/256 and zero point -128")

        self.input, self.output = inp, out
        self.in_shape = inp.hwc("SOFTMAX input")
        self.depth = inp.shape[-1]
        self.outer = inp.size() // self.depth
        # The sum of exponentials stays below 2^28, so that the output
        # shift is at most 31 (reference/softmax.h checks it)
        if self.depth > 511:
            fail("SOFTMAX rows must have at most 511 elements")
        self.in_zp = inp.zero_points[0]
        self.out_zp = out.zero_points[0]
        self.mult, self.shift, self.diff_min = softmax_scaling(
            float(opts.Beta()), float(inp.scales[0]), round_mode == "single")


def read_model(path, schema, round_mode):
    with open(path, "rb") as f:
        buf = f.read()
    model = schema.Model.GetRootAs(buf, 0)
    if model.SubgraphsLength() != 1:
        fail("the model must have one subgraph")
    sub = model.Subgraphs(0)
    if sub.InputsLength() != 1 or sub.OutputsLength() != 1:
        fail("the model must have one input and one output")
    ops = schema.BuiltinOperator
    kinds = {ops.CONV_2D: Conv, ops.AVERAGE_POOL_2D: AvgPool,
             ops.RESHAPE: Reshape, ops.FULLY_CONNECTED: FullyConnected,
             ops.SOFTMAX: Softmax}
    layers = []
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        if builtin not in kinds:
            name = next((k for k, v in vars(ops).items() if v == builtin),
                        str(builtin))
            fail(f"operator {name} is not supported yet")
        layers.append(kinds[builtin](model, sub, op, schema, round_mode))
    model_in = Tensor(model, sub, sub.Inputs(0), schema)
    model_out = Tensor(model, sub, sub.Outputs(0), schema)
    return layers, model_in, model_out


# --------------------------------------------------------------------------
# C89 writing
# --------------------------------------------------------------------------

def c_array(decl, values, per_line=16):
    lines = [f"{decl} = {{"]
    for i in range(0, len(values), per_line):
        chunk = ", ".join(str(v) for v in values[i:i + per_line])
        sep = "," if i + per_line < len(values) else ""
        lines.append(f"\t{chunk}{sep}")
    lines.append("};")
    return "\n".join(lines)


def c_float(x):
    # The float32 value printed exactly as a double: strtod gives that
    # double and the conversion to float is exact
    return f"{float(x)!r}f"


def c_block(*lines):
    """C lines given as (depth, text): text indented by depth tabs."""
    return "\n".join("\t" * d + t if t else "" for d, t in lines)


RESCALE = {
    # MultiplyByQuantizedMultiplier, double rounding (kernels/internal/
    # common.cc): SaturatingRoundingDoublingHighMul then RoundingDivideByPOT
    "double": c_block(
        (0, "static int knpu_rescale(int x, int m, int s)"),
        (0, "{"),
        (1, "int ls = s > 0 ? s : 0;"),
        (1, "int rs = s > 0 ? 0 : -s;"),
        (1, "long p, nudge;"),
        (1, "int h, mask, thr;"),
        (0, ""),
        (1, "p = (long)(int)((unsigned)x << ls) * (long)m;"),
        (1, "nudge = p >= 0 ? (1L << 30) : (1L - (1L << 30));"),
        (1, "h = (int)((p + nudge) / (1L << 31));"),
        (1, "mask = (int)((1L << rs) - 1);"),
        (1, "thr = (mask >> 1) + (h < 0 ? 1 : 0);"),
        (1, "return (h >> rs) + ((h & mask) > thr ? 1 : 0);"),
        (0, "}")),
    # MultiplyByQuantizedMultiplier, TFLITE_SINGLE_ROUNDING
    "single": c_block(
        (0, "static int knpu_rescale(int x, int m, int s)"),
        (0, "{"),
        (1, "long t = 31 - s;"),
        (0, ""),
        (1, "return (int)(((long)x * (long)m + (1L << (t - 1))) >> t);"),
        (0, "}")),
}

# The gemmlowp fixed-point functions (kernels/internal/fixedpoint.h) and
# GetReciprocal (kernels/internal/common.h) that TFLite Micro's int8
# Softmax uses, for 32-bit raw values. A Qm.n value has m integer bits.
SOFTMAX_HELPERS = c_block(
    (0, "/* SaturatingRoundingDoublingHighMul */"),
    (0, "static int knpu_srdhm(int a, int b)"),
    (0, "{"),
    (1, "long ab = (long)a * (long)b;"),
    (1, "long nudge = ab >= 0 ? (1L << 30) : (1L - (1L << 30));"),
    (0, ""),
    (1, "if (a == b && a == -2147483647 - 1)"),
    (2, "return 2147483647;"),
    (1, "return (int)((ab + nudge) / (1L << 31));"),
    (0, "}"),
    (0, ""),
    (0, "/* RoundingDivideByPOT, 0 <= e <= 31 */"),
    (0, "static int knpu_rdivpot(int x, int e)"),
    (0, "{"),
    (1, "int mask = (int)((1L << e) - 1);"),
    (1, "int thr = (mask >> 1) + (x < 0 ? 1 : 0);"),
    (0, ""),
    (1, "return (x >> e) + ((x & mask) > thr ? 1 : 0);"),
    (0, "}"),
    (0, ""),
    (0, "/* SaturatingRoundingMultiplyByPOT, e > 0: x * 2^e saturated */"),
    (0, "static int knpu_satmulpot(int x, int e)"),
    (0, "{"),
    (1, "long w = (long)x * (1L << e);"),
    (0, ""),
    (1, "if (w > 2147483647L)"),
    (2, "return 2147483647;"),
    (1, "if (w < -2147483647L - 1)"),
    (2, "return -2147483647 - 1;"),
    (1, "return (int)w;"),
    (0, "}"),
    (0, ""),
    (0, "/* exp_on_interval_between_negative_one_quarter_and_0_excl,"),
    (0, "   Q0.31 in and out */"),
    (0, "static int knpu_exp_quarter(int a)"),
    (0, "{"),
    (1, "int x, x2, x3, x4, x4_over_4, poly;"),
    (0, ""),
    (1, "x = a + (1 << 28);"),
    (1, "x2 = knpu_srdhm(x, x);"),
    (1, "x3 = knpu_srdhm(x2, x);"),
    (1, "x4 = knpu_srdhm(x2, x2);"),
    (1, "x4_over_4 = knpu_rdivpot(x4, 2);"),
    (1, "poly = knpu_rdivpot(knpu_srdhm(x4_over_4 + x3, 715827883) + x2, 1);"),
    (1, "return 1895147668 + knpu_srdhm(1895147668, x + poly);"),
    (0, "}"),
    (0, ""),
    (0, "/* exp_on_negative_values: Q5.26 in, Q0.31 out */"),
    (0, "static int knpu_exp_neg(int a)"),
    (0, "{"),
    (1, "int quarter = 1 << 24;"),
    (1, "int amq = (a & (quarter - 1)) - quarter;"),
    (1, "int rem = amq - a;"),
    (1, "int result = knpu_exp_quarter(knpu_satmulpot(amq, 5));"),
    (0, ""),
    (1, "if (rem & (1 << 24))"),
    (2, "result = knpu_srdhm(result, 1672461947);"),
    (1, "if (rem & (1 << 25))"),
    (2, "result = knpu_srdhm(result, 1302514674);"),
    (1, "if (rem & (1 << 26))"),
    (2, "result = knpu_srdhm(result, 790015084);"),
    (1, "if (rem & (1 << 27))"),
    (2, "result = knpu_srdhm(result, 290630308);"),
    (1, "if (rem & (1 << 28))"),
    (2, "result = knpu_srdhm(result, 39332535);"),
    (1, "if (rem & (1 << 29))"),
    (2, "result = knpu_srdhm(result, 720401);"),
    (1, "if (rem & (1 << 30))"),
    (2, "result = knpu_srdhm(result, 242);"),
    (1, "return a == 0 ? 2147483647 : result;"),
    (0, "}"),
    (0, ""),
    (0, "/* one_over_one_plus_x_for_x_in_0_1: Q0.31 in and out,"),
    (0, "   Newton-Raphson division in Q2.29 */"),
    (0, "static int knpu_one_over_one_plus_x(int a)"),
    (0, "{"),
    (1, "long sum = (long)a + 2147483647L;"),
    (1, "int half_den = (int)((sum + (sum >= 0 ? 1 : -1)) / 2);"),
    (1, "int x, i, t;"),
    (0, ""),
    (1, "x = 1515870810 + knpu_srdhm(half_den, -1010580540);"),
    (1, "for (i = 0; i < 3; i++) {"),
    (2, "t = (1 << 29) - knpu_srdhm(half_den, x);"),
    (2, "x = x + knpu_satmulpot(knpu_srdhm(x, t), 2);"),
    (1, "}"),
    (1, "return knpu_satmulpot(x, 1);"),
    (0, "}"),
    (0, ""),
    (0, "/* GetReciprocal: x > 0 with x_integer_digits integer bits */"),
    (0, "static int knpu_reciprocal(int x, int x_integer_digits, int *nbits)"),
    (0, "{"),
    (1, "unsigned u = (unsigned)x;"),
    (1, "int headroom_plus_one = 0;"),
    (0, ""),
    (1, "while (!(u & 0x80000000u)) {"),
    (2, "u <<= 1;"),
    (2, "headroom_plus_one++;"),
    (1, "}"),
    (1, "*nbits = x_integer_digits - headroom_plus_one;"),
    (1, "return knpu_one_over_one_plus_x((int)(u - 0x80000000u));"),
    (0, "}"))


def conv_function(n, c, act_in, act_out):
    p = f"L{n}"
    pragma = (
        f"#pragma knpu conv2d in={act_in} out={act_out} w={p}_W "
        f"bias={p}_B mult={p}_M shift={p}_S "
        f"in_shape=({c.ih},{c.iw},{c.ic}) "
        f"out_shape=({c.oh},{c.ow},{c.oc}) "
        f"kernel=({c.kh},{c.kw}) stride=({c.sh},{c.sw}) "
        f"pad=({c.pt},{c.pl},{c.pb},{c.pr}) "
        f"in_zp={c.in_zp} out_zp={c.out_zp} "
        f"clamp=({c.clamp[0]},{c.clamp[1]})")
    w_index = f"((oc * {c.kh} + ky) * {c.kw} + kx) * {c.ic} + ic"
    in_index = f"(iy * {c.iw} + ix) * {c.ic} + ic"
    out_index = f"(oy * {c.ow} + ox) * {c.oc} + oc"
    # ConvPerChannel (reference/integer_ops/conv.h)
    return c_block(
        (0, f"static void layer{n}(void)"),
        (0, "{"),
        (0, pragma),
        (1, "int oy, ox, oc, ky, kx, ic, iy, ix, acc;"),
        (0, ""),
        (1, f"for (oy = 0; oy < {c.oh}; oy++)"),
        (2, f"for (ox = 0; ox < {c.ow}; ox++)"),
        (3, f"for (oc = 0; oc < {c.oc}; oc++) {{"),
        (4, "acc = 0;"),
        (4, f"for (ky = 0; ky < {c.kh}; ky++) {{"),
        (5, f"iy = oy * {c.sh} - {c.pt} + ky;"),
        (5, f"if (iy < 0 || iy >= {c.ih})"),
        (6, "continue;"),
        (5, f"for (kx = 0; kx < {c.kw}; kx++) {{"),
        (6, f"ix = ox * {c.sw} - {c.pl} + kx;"),
        (6, f"if (ix < 0 || ix >= {c.iw})"),
        (7, "continue;"),
        (6, f"for (ic = 0; ic < {c.ic}; ic++)"),
        (7, f"acc += {p}_W[{w_index}]"),
        (7, f"    * ({act_in}[{in_index}] - ({c.in_zp}));"),
        (5, "}"),
        (4, "}"),
        (4, f"acc += {p}_B[oc];"),
        (4, f"acc = knpu_rescale(acc, {p}_M[oc], {p}_S[oc]) + ({c.out_zp});"),
        (4, f"if (acc < {c.clamp[0]})"),
        (5, f"acc = {c.clamp[0]};"),
        (4, f"if (acc > {c.clamp[1]})"),
        (5, f"acc = {c.clamp[1]};"),
        (4, f"{act_out}[{out_index}] = (signed char)acc;"),
        (3, "}"),
        (0, "}"))


def avgpool_function(n, a, act_in, act_out):
    pragma = (
        f"#pragma knpu avgpool in={act_in} out={act_out} "
        f"in_shape=({a.ih},{a.iw},{a.c}) out_shape=({a.oh},{a.ow},{a.c}) "
        f"kernel=({a.kh},{a.kw}) stride=({a.sh},{a.sw}) "
        f"pad=({a.pt},{a.pl},{a.pb},{a.pr}) zp={a.zp} "
        f"clamp=({a.clamp[0]},{a.clamp[1]})")
    in_index = f"(iy * {a.iw} + ix) * {a.c} + c"
    out_index = f"(oy * {a.ow} + ox) * {a.c} + c"
    # AveragePool (reference/integer_ops/pooling.h): the elements of the
    # window inside the input, averaged with rounding half away from zero
    return c_block(
        (0, f"static void layer{n}(void)"),
        (0, "{"),
        (0, pragma),
        (1, "int oy, ox, c, ky, kx, iy, ix, acc, count;"),
        (0, ""),
        (1, f"for (oy = 0; oy < {a.oh}; oy++)"),
        (2, f"for (ox = 0; ox < {a.ow}; ox++)"),
        (3, f"for (c = 0; c < {a.c}; c++) {{"),
        (4, "acc = 0;"),
        (4, "count = 0;"),
        (4, f"for (ky = 0; ky < {a.kh}; ky++) {{"),
        (5, f"iy = oy * {a.sh} - {a.pt} + ky;"),
        (5, f"if (iy < 0 || iy >= {a.ih})"),
        (6, "continue;"),
        (5, f"for (kx = 0; kx < {a.kw}; kx++) {{"),
        (6, f"ix = ox * {a.sw} - {a.pl} + kx;"),
        (6, f"if (ix < 0 || ix >= {a.iw})"),
        (7, "continue;"),
        (6, f"acc += {act_in}[{in_index}];"),
        (6, "count++;"),
        (5, "}"),
        (4, "}"),
        (4, "acc = acc > 0 ? (acc + count / 2) / count"),
        (4, "    : (acc - count / 2) / count;"),
        (4, f"if (acc < {a.clamp[0]})"),
        (5, f"acc = {a.clamp[0]};"),
        (4, f"if (acc > {a.clamp[1]})"),
        (5, f"acc = {a.clamp[1]};"),
        (4, f"{act_out}[{out_index}] = (signed char)acc;"),
        (3, "}"),
        (0, "}"))


def reshape_function(n, r, act_in, act_out):
    pragma = (
        f"#pragma knpu reshape in={act_in} out={act_out} "
        f"in_shape=({r.in_shape[0]},{r.in_shape[1]},{r.in_shape[2]}) "
        f"out_shape=({r.out_shape[0]},{r.out_shape[1]},{r.out_shape[2]})")
    # Reshape (kernels/reshape.cc) copies the bytes
    return c_block(
        (0, f"static void layer{n}(void)"),
        (0, "{"),
        (0, pragma),
        (1, "int i;"),
        (0, ""),
        (1, f"for (i = 0; i < {r.input.size()}; i++)"),
        (2, f"{act_out}[i] = {act_in}[i];"),
        (0, "}"))


def fc_function(n, f, act_in, act_out):
    p = f"L{n}"
    pragma = (
        f"#pragma knpu fc in={act_in} out={act_out} w={p}_W "
        f"bias={p}_B mult={p}_M shift={p}_S "
        f"in_size={f.in_size} out_size={f.out_size} "
        f"in_zp={f.in_zp} out_zp={f.out_zp} "
        f"clamp=({f.clamp[0]},{f.clamp[1]})")
    # FullyConnectedPerChannel and FullyConnected
    # (reference/integer_ops/fully_connected.h)
    return c_block(
        (0, f"static void layer{n}(void)"),
        (0, "{"),
        (0, pragma),
        (1, "int o, i, acc;"),
        (0, ""),
        (1, f"for (o = 0; o < {f.out_size}; o++) {{"),
        (2, "acc = 0;"),
        (2, f"for (i = 0; i < {f.in_size}; i++)"),
        (3, f"acc += {p}_W[o * {f.in_size} + i]"),
        (3, f"    * ({act_in}[i] - ({f.in_zp}));"),
        (2, f"acc += {p}_B[o];"),
        (2, f"acc = knpu_rescale(acc, {p}_M[o], {p}_S[o]) + ({f.out_zp});"),
        (2, f"if (acc < {f.clamp[0]})"),
        (3, f"acc = {f.clamp[0]};"),
        (2, f"if (acc > {f.clamp[1]})"),
        (3, f"acc = {f.clamp[1]};"),
        (2, f"{act_out}[o] = (signed char)acc;"),
        (1, "}"),
        (0, "}"))


def softmax_function(n, s, act_in, act_out, round_mode):
    p = f"L{n}"
    shape = ",".join(str(d) for d in s.in_shape)
    pragma = (
        f"#pragma knpu cpu in={act_in} out={act_out} "
        f"in_shape=({shape}) out_shape=({shape}) "
        f"in_zp={s.in_zp} out_zp={s.out_zp}")
    # MultiplyByQuantizedMultiplierGreaterThanOne (kernels/internal/
    # common.h) of the input difference, a Q5.26 value
    if round_mode == "single":
        scaled = f"knpu_rescale(diff, {p}_IN_MULT, {p}_IN_SHIFT)"
    else:
        scaled = (f"knpu_srdhm((int)((unsigned)diff << {p}_IN_SHIFT), "
                  f"{p}_IN_MULT)")
    d = s.depth
    x = f"{act_in}[i * {d} + c]"
    y = f"{act_out}[i * {d} + c]"
    # Softmax (reference/softmax.h) with int8 input and output: the sum of
    # exponentials is Q12.19, its reciprocal Q0.31, and the output shift
    # nbits + 31 - 8 for 8-bit outputs
    return c_block(
        (0, f"static void layer{n}(void)"),
        (0, "{"),
        (0, pragma),
        (1, "int i, c, max, diff, sum, nbits, scale, out;"),
        (0, ""),
        (1, f"for (i = 0; i < {s.outer}; i++) {{"),
        (2, "max = -128;"),
        (2, f"for (c = 0; c < {d}; c++)"),
        (3, f"if ({x} > max)"),
        (4, f"max = {x};"),
        (2, "sum = 0;"),
        (2, f"for (c = 0; c < {d}; c++) {{"),
        (3, f"diff = {x} - max;"),
        (3, f"if (diff >= {p}_DIFF_MIN)"),
        (4, f"sum += knpu_rdivpot(knpu_exp_neg({scaled}), "
            f"{ACCUMULATION_INTEGER_BITS});"),
        (2, "}"),
        (2, f"scale = knpu_reciprocal(sum, {ACCUMULATION_INTEGER_BITS}, "
            "&nbits);"),
        (2, f"for (c = 0; c < {d}; c++) {{"),
        (3, f"diff = {x} - max;"),
        (3, f"if (diff < {p}_DIFF_MIN) {{"),
        (4, f"{y} = -128;"),
        (4, "continue;"),
        (3, "}"),
        (3, f"out = knpu_srdhm(scale, knpu_exp_neg({scaled}));"),
        (3, "out = knpu_rdivpot(out, nbits + 23) - 128;"),
        (3, "if (out < -128)"),
        (4, "out = -128;"),
        (3, "if (out > 127)"),
        (4, "out = 127;"),
        (3, f"{y} = (signed char)out;"),
        (2, "}"),
        (1, "}"),
        (0, "}"))


def write_model_c(src_name, layers, model_in, model_out, round_mode):
    out = [f"/* Generated by tflite2c from {src_name}; do not edit. */", ""]
    # Activation arrays: act0 is the model input, actN the output of layer N
    acts = [model_in] + [layer.output for layer in layers]
    for n, layer in enumerate(layers, 1):
        p = f"L{n}"
        if isinstance(layer, (Conv, FullyConnected)):
            out.append(c_array(f"static const signed char {p}_W"
                               f"[{len(layer.weights)}]", layer.weights))
            size = len(layer.mult)
            out.append(c_array(f"static const int {p}_B[{size}]",
                               layer.bias, 8))
            out.append(c_array(f"static const int {p}_M[{size}]",
                               layer.mult, 8))
            out.append(c_array(f"static const int {p}_S[{size}]",
                               layer.shift, 8))
            out.append("")
        elif isinstance(layer, Softmax):
            out.append(f"static const int {p}_IN_MULT = {layer.mult};")
            out.append(f"static const int {p}_IN_SHIFT = {layer.shift};")
            out.append(f"static const int {p}_DIFF_MIN = {layer.diff_min};")
            out.append("")
    last = len(acts) - 1
    if acts[last].index != model_out.index:
        fail("the last layer must write the model output")
    out.append(f"static const float ACT0_SCALE = "
               f"{c_float(model_in.scales[0])};")
    out.append(f"static const float ACT{last}_SCALE = "
               f"{c_float(model_out.scales[0])};")
    for n, t in enumerate(acts):
        out.append(f"signed char act{n}[{t.size()}];")
    out.append("")
    if any(isinstance(layer, (Conv, FullyConnected)) for layer in layers) or (
            round_mode == "single"
            and any(isinstance(layer, Softmax) for layer in layers)):
        out.append(RESCALE[round_mode])
        out.append("")
    if any(isinstance(layer, Softmax) for layer in layers):
        out.append(SOFTMAX_HELPERS)
        out.append("")
    for n, layer in enumerate(layers, 1):
        if layer.input.index != acts[n - 1].index:
            fail("layers must form a chain from the model input")
        act_in, act_out = f"act{n - 1}", f"act{n}"
        if isinstance(layer, Conv):
            out.append(conv_function(n, layer, act_in, act_out))
        elif isinstance(layer, AvgPool):
            out.append(avgpool_function(n, layer, act_in, act_out))
        elif isinstance(layer, Reshape):
            out.append(reshape_function(n, layer, act_in, act_out))
        elif isinstance(layer, FullyConnected):
            out.append(fc_function(n, layer, act_in, act_out))
        else:
            out.append(softmax_function(n, layer, act_in, act_out,
                                        round_mode))
        out.append("")
    pragma = (f"#pragma knpu model in=act0 out=act{last} "
              f"in_scale=ACT0_SCALE in_zp={model_in.zero_points[0]} "
              f"out_scale=ACT{last}_SCALE "
              f"out_zp={model_out.zero_points[0]} round=\"{round_mode}\"")
    out.append(c_block(
        (0, "void knpu_model(void)"),
        (0, "{"),
        (0, pragma),
        *((1, f"layer{n}();") for n in range(1, len(layers) + 1)),
        (0, "}")))
    return "\n".join(out) + "\n"


def write_main_c(src_name, in_size, out, npu_out):
    """main() for gcc; out and npu_out are (array name, size) pairs."""
    arrays = []
    for array in (("act0", in_size), out, npu_out):
        if array not in arrays:
            arrays.append(array)

    def save(argi, name):
        return [(1, f"f = fopen(argv[{argi}], \"wb\");"),
                (1, f"if (!f || fwrite({name}, 1, sizeof {name}, f) != "
                    f"sizeof {name} ||"),
                (1, "    fclose(f) != 0)"),
                (2, "return 1;")]

    return c_block(
        (0, f"/* Generated by tflite2c from {src_name}; do not edit. */"),
        (0, "/* Runs knpu_model() on a CPU: "
            "main INPUT OUTPUT [NPU_OUTPUT] */"),
        (0, ""),
        (0, "#include <stdio.h>"),
        (0, ""),
        *((0, f"extern signed char {name}[{size}];") for name, size in arrays),
        (0, "void knpu_model(void);"),
        (0, ""),
        (0, "int main(int argc, char **argv)"),
        (0, "{"),
        (1, "FILE *f;"),
        (0, ""),
        (1, "if (argc != 3 && argc != 4)"),
        (2, "return 2;"),
        (1, "f = fopen(argv[1], \"rb\");"),
        (1, "if (!f || fread(act0, 1, sizeof act0, f) != sizeof act0 ||"),
        (1, "    fgetc(f) != EOF)"),
        (2, "return 1;"),
        (1, "fclose(f);"),
        (1, "knpu_model();"),
        *save(2, out[0]),
        (1, "if (argc == 3)"),
        (2, "return 0;"),
        *save(3, npu_out[0]),
        (1, "return 0;"),
        (0, "}")) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Write an int8 .tflite model as C89 with #pragma knpu.")
    ap.add_argument("model", help=".tflite file")
    ap.add_argument("-o", "--output", required=True, help="model C file")
    ap.add_argument("--main", help="also write a main() C file for gcc")
    ap.add_argument("--npu-only", action="store_true",
                    help="leave out the CPU layers; the output of the last "
                         "NPU layer becomes the model output")
    ap.add_argument("--schema", required=True,
                    help="TFLite Micro's schema_py_generated.py")
    ap.add_argument("--round", choices=["double", "single"],
                    default="double",
                    help="requantization rounding of the target build "
                         "(TFLITE_SINGLE_ROUNDING)")
    args = ap.parse_args()

    schema = load_schema(args.schema)
    layers, model_in, model_out = read_model(args.model, schema, args.round)
    if not layers:
        fail("the model has no operators")
    n_npu = next((i for i, layer in enumerate(layers) if layer.cpu),
                 len(layers))
    if not all(layer.cpu for layer in layers[n_npu:]):
        fail("CPU layers must come after the last NPU layer")
    # The output of the NPU layers: the model input if there is none
    npu_out = layers[n_npu - 1].output if n_npu else model_in
    if args.npu_only:
        if not n_npu:
            fail("the model has no NPU layer")
        layers, model_out = layers[:n_npu], npu_out
    name = args.model.replace("\\", "/").rsplit("/", 1)[-1]
    # Generated C uses LF line ends only (lcc reads #pragma lines verbatim)
    with open(args.output, "w", newline="\n") as f:
        f.write(write_model_c(name, layers, model_in, model_out, args.round))
    if args.main:
        with open(args.main, "w", newline="\n") as f:
            f.write(write_main_c(name, model_in.size(),
                                 (f"act{len(layers)}", model_out.size()),
                                 (f"act{n_npu}", npu_out.size())))


if __name__ == "__main__":
    main()
