// V2: golden output of one int8 convolution from the TensorFlow Lite Micro
// reference kernel (reference_integer_ops::ConvPerChannel). The rounding
// mode is fixed at build time by TFLITE_SINGLE_ROUNDING, so CMake builds
// this file twice: tflm_golden_double and tflm_golden_single.
//
//   tflm_golden --ifm H,W,C --ofm H,W,C --kernel KH,KW --stride SH,SW
//               --pad TOP,LEFT --ifm-zp Z --ofm-zp Z --clamp MIN,MAX
//               --input F --weights F --bias F --rq F --output F
//
// The binary files use the SRAM layouts of docs/npu-spec.md 7: NHWC int8
// input, OHWI int8 weights, int32 bias and (multiplier, shift) int32 pairs,
// all little endian. TFLite only uses the top and left padding.
//
// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/conv.h"

// TFLite Micro leaves MicroPrintf to the platform.
void MicroPrintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fputc('\n', stderr);
}

namespace {

[[noreturn]] void Die(const std::string& msg) {
  std::fprintf(stderr, "tflm_golden: %s\n", msg.c_str());
  std::exit(2);
}

std::vector<int> Ints(const std::string& s, size_t count) {
  std::vector<int> v;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t comma = s.find(',', pos);
    std::string part = s.substr(pos, comma == std::string::npos
                                         ? std::string::npos
                                         : comma - pos);
    char* end;
    long x = std::strtol(part.c_str(), &end, 0);
    if (part.empty() || *end != '\0') Die("bad number list: " + s);
    v.push_back(static_cast<int>(x));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (v.size() != count) Die("expected " + std::to_string(count) +
                             " values: " + s);
  return v;
}

std::vector<uint8_t> Load(const std::string& path, size_t size) {
  std::vector<uint8_t> buf(size);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) Die("cannot open " + path);
  size_t got = std::fread(buf.data(), 1, size, f);
  bool extra = std::fgetc(f) != EOF;
  std::fclose(f);
  if (got != size || extra) Die("size of " + path + " is not " +
                                std::to_string(size));
  return buf;
}

int32_t Le32(const uint8_t* p) {
  return static_cast<int32_t>(static_cast<uint32_t>(p[0]) |
                              static_cast<uint32_t>(p[1]) << 8 |
                              static_cast<uint32_t>(p[2]) << 16 |
                              static_cast<uint32_t>(p[3]) << 24);
}

// RuntimeShape has no initializer-list constructor; RuntimeShape({n}) would
// pick the dimensions-count constructor.
tflite::RuntimeShape Shape(std::vector<int32_t> dims) {
  return tflite::RuntimeShape(static_cast<int>(dims.size()), dims.data());
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> opt;
  for (int i = 1; i + 1 < argc; i += 2) opt[argv[i]] = argv[i + 1];
  if (argc % 2 != 1) Die("options come in pairs");
  for (const char* key :
       {"--ifm", "--ofm", "--kernel", "--stride", "--pad", "--ifm-zp",
        "--ofm-zp", "--clamp", "--input", "--weights", "--bias", "--rq",
        "--output"}) {
    if (!opt.count(key)) Die(std::string("missing ") + key);
  }

  const std::vector<int> ifm = Ints(opt["--ifm"], 3);
  const std::vector<int> ofm = Ints(opt["--ofm"], 3);
  const std::vector<int> kernel = Ints(opt["--kernel"], 2);
  const std::vector<int> stride = Ints(opt["--stride"], 2);
  const std::vector<int> pad = Ints(opt["--pad"], 2);
  const std::vector<int> clamp = Ints(opt["--clamp"], 2);
  const int ifm_zp = Ints(opt["--ifm-zp"], 1)[0];
  const int ofm_zp = Ints(opt["--ofm-zp"], 1)[0];
  const int ic = ifm[2], oc = ofm[2];

  std::vector<uint8_t> input =
      Load(opt["--input"], static_cast<size_t>(ifm[0]) * ifm[1] * ic);
  std::vector<uint8_t> weights = Load(
      opt["--weights"], static_cast<size_t>(oc) * kernel[0] * kernel[1] * ic);
  std::vector<uint8_t> bias_raw = Load(opt["--bias"], 4 * oc);
  std::vector<uint8_t> rq_raw = Load(opt["--rq"], 8 * oc);

  std::vector<int32_t> bias(oc), multiplier(oc), shift(oc);
  for (int c = 0; c < oc; ++c) {
    bias[c] = Le32(&bias_raw[4 * c]);
    multiplier[c] = Le32(&rq_raw[8 * c]);
    shift[c] = Le32(&rq_raw[8 * c + 4]);
  }

  // Parameters as TFLite Micro's conv kernel fills them (conv_common.cc)
  tflite::ConvParams params = {};
  params.padding_values.height = static_cast<int16_t>(pad[0]);
  params.padding_values.width = static_cast<int16_t>(pad[1]);
  params.stride_height = static_cast<int16_t>(stride[0]);
  params.stride_width = static_cast<int16_t>(stride[1]);
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.input_offset = -ifm_zp;
  params.weights_offset = 0;
  params.output_offset = ofm_zp;
  params.quantized_activation_min = clamp[0];
  params.quantized_activation_max = clamp[1];

  std::vector<int8_t> output(static_cast<size_t>(ofm[0]) * ofm[1] * oc);
  tflite::micro::reference_integer_ops::ConvPerChannel(
      params, multiplier.data(), shift.data(),
      Shape({1, ifm[0], ifm[1], ic}),
      reinterpret_cast<const int8_t*>(input.data()),
      Shape({oc, kernel[0], kernel[1], ic}),
      reinterpret_cast<const int8_t*>(weights.data()),
      Shape({oc}), bias.data(),
      Shape({1, ofm[0], ofm[1], oc}), output.data());

  FILE* f = std::fopen(opt["--output"].c_str(), "wb");
  if (!f || std::fwrite(output.data(), 1, output.size(), f) != output.size() ||
      std::fclose(f) != 0) {
    Die("cannot write " + opt["--output"]);
  }
  return 0;
}
