#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""tflite2c: write an int8 .tflite model as C89 for lcc and gcc.

Every NPU layer becomes one parameterless static function whose body is the
C89 loop TFLite Micro's reference kernel runs, and whose first line is the
layer's `#pragma knpu` (docs/design.md 9.2, 9.3). knpu_model() calls the
layers in order and carries `#pragma knpu model`. Requantization multipliers
and shifts are computed with TFLite Micro's formulas for the chosen rounding
mode.

    tflite2c.py --schema schema_py_generated.py --round double \\
        model.tflite -o model.c [--main main.c]

The schema is TFLite Micro's flatc-generated Python module
(tensorflow/lite/micro/python/schema_py_generated.py); it needs the PyPI
package flatbuffers. --main writes a separate C file that reads the input
bytes from a file, calls knpu_model() and writes the output bytes, so that
gcc can run the model.
"""

import argparse
import importlib.util
import math
import struct
import sys

INT8_MIN, INT8_MAX = -128, 127


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


class Conv:
    """One CONV_2D with everything the C and the pragma need."""

    def __init__(self, model, sub, op, schema, round_mode):
        tensors = [Tensor(model, sub, op.Inputs(i), schema)
                   for i in range(op.InputsLength())]
        if len(tensors) != 3:
            fail("CONV_2D needs input, filter and bias")
        inp, filt, bias = tensors
        out = Tensor(model, sub, op.Outputs(0), schema)
        opts = schema.Conv2DOptions()
        table = op.BuiltinOptions()
        opts.Init(table.Bytes, table.Pos)

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
    layers = []
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        if builtin != ops.CONV_2D:
            name = next((k for k, v in vars(ops).items() if v == builtin),
                        str(builtin))
            fail(f"operator {name} is not supported yet")
        layers.append(Conv(model, sub, op, schema, round_mode))
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


def write_model_c(src_name, layers, model_in, model_out, round_mode):
    out = [f"/* Generated by tflite2c from {src_name}; do not edit. */", ""]
    # Activation arrays: act0 is the model input, actN the output of layer N
    acts = [model_in] + [layer.output for layer in layers]
    for n, layer in enumerate(layers, 1):
        p = f"L{n}"
        out.append(c_array(f"static const signed char {p}_W"
                           f"[{len(layer.weights)}]", layer.weights))
        out.append(c_array(f"static const int {p}_B[{layer.oc}]",
                           layer.bias, 8))
        out.append(c_array(f"static const int {p}_M[{layer.oc}]",
                           layer.mult, 8))
        out.append(c_array(f"static const int {p}_S[{layer.oc}]",
                           layer.shift, 8))
        out.append("")
    last = len(acts) - 1
    if acts[last].index != model_out.index:
        fail("the last layer must write the model output")
    out.append(f"static const float ACT0_SCALE = "
               f"{c_float(model_in.scales[0])};")
    out.append(f"static const float ACT{last}_SCALE = "
               f"{c_float(model_out.scales[0])};")
    for n, t in enumerate(acts):
        size = t.shape[1] * t.shape[2] * t.shape[3]
        out.append(f"signed char act{n}[{size}];")
    out.append("")
    out.append(RESCALE[round_mode])
    out.append("")
    for n, layer in enumerate(layers, 1):
        if layer.input.index != acts[n - 1].index:
            fail("layers must form a chain from the model input")
        out.append(conv_function(n, layer, f"act{n - 1}", f"act{n}"))
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


def write_main_c(src_name, in_size, out_size, last):
    out = f"act{last}"
    return c_block(
        (0, f"/* Generated by tflite2c from {src_name}; do not edit. */"),
        (0, "/* Runs knpu_model() on a CPU: main INPUT OUTPUT */"),
        (0, ""),
        (0, "#include <stdio.h>"),
        (0, ""),
        (0, f"extern signed char act0[{in_size}];"),
        (0, f"extern signed char {out}[{out_size}];"),
        (0, "void knpu_model(void);"),
        (0, ""),
        (0, "int main(int argc, char **argv)"),
        (0, "{"),
        (1, "FILE *f;"),
        (0, ""),
        (1, "if (argc != 3)"),
        (2, "return 2;"),
        (1, "f = fopen(argv[1], \"rb\");"),
        (1, "if (!f || fread(act0, 1, sizeof act0, f) != sizeof act0 ||"),
        (1, "    fgetc(f) != EOF)"),
        (2, "return 1;"),
        (1, "fclose(f);"),
        (1, "knpu_model();"),
        (1, "f = fopen(argv[2], \"wb\");"),
        (1, f"if (!f || fwrite({out}, 1, sizeof {out}, f) != sizeof {out})"),
        (2, "return 1;"),
        (1, "return fclose(f) != 0;"),
        (0, "}")) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Write an int8 .tflite model as C89 with #pragma knpu.")
    ap.add_argument("model", help=".tflite file")
    ap.add_argument("-o", "--output", required=True, help="model C file")
    ap.add_argument("--main", help="also write a main() C file for gcc")
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
    name = args.model.replace("\\", "/").rsplit("/", 1)[-1]
    # Generated C uses LF line ends only (lcc reads #pragma lines verbatim)
    with open(args.output, "w", newline="\n") as f:
        f.write(write_model_c(name, layers, model_in, model_out, args.round))
    if args.main:
        in_size = model_in.shape[1] * model_in.shape[2] * model_in.shape[3]
        out_size = (model_out.shape[1] * model_out.shape[2]
                    * model_out.shape[3])
        with open(args.main, "w", newline="\n") as f:
            f.write(write_main_c(name, in_size, out_size, len(layers)))


if __name__ == "__main__":
    main()
