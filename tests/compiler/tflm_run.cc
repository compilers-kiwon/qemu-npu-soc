// V2: runs a .tflite model with the TensorFlow Lite Micro interpreter.
//
//   tflm_run_<mode> MODEL INPUT OUTPUT
//
// INPUT holds the int8 bytes of the model's input tensor; the int8 bytes of
// the output tensor are written to OUTPUT. The program is built once per
// rounding mode (TFLITE_SINGLE_ROUNDING 0 and 1).
//
// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

namespace {

alignas(16) uint8_t arena[1 << 20];

[[noreturn]] void die(const char *msg, const char *arg = nullptr) {
  std::fprintf(stderr, "tflm_run: %s%s%s\n", msg, arg ? ": " : "",
               arg ? arg : "");
  std::exit(2);
}

std::vector<uint8_t> readFile(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f)
    die("cannot open", path);
  std::vector<uint8_t> data;
  uint8_t buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0)
    data.insert(data.end(), buf, buf + n);
  if (std::ferror(f))
    die("cannot read", path);
  std::fclose(f);
  return data;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4)
    die("usage: tflm_run MODEL INPUT OUTPUT");

  // FlatBuffers wants the model aligned; vector storage comes from malloc
  std::vector<uint8_t> modelData = readFile(argv[1]);
  std::vector<uint8_t> input = readFile(argv[2]);
  const tflite::Model *model = tflite::GetModel(modelData.data());

  // The operators of the two demo models (docs/design.md 9.2)
  tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddFullyConnected();
  resolver.AddAdd();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddSoftmax();

  tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof arena);
  if (interpreter.AllocateTensors() != kTfLiteOk)
    die("AllocateTensors failed");
  if (interpreter.inputs_size() != 1 || interpreter.outputs_size() != 1)
    die("the model must have one input and one output");

  TfLiteTensor *in = interpreter.input(0);
  if (in->type != kTfLiteInt8 || in->bytes != input.size())
    die("input must be the int8 bytes of the input tensor", argv[2]);
  std::memcpy(in->data.int8, input.data(), input.size());
  if (interpreter.Invoke() != kTfLiteOk)
    die("Invoke failed");

  const TfLiteTensor *out = interpreter.output(0);
  if (out->type != kTfLiteInt8)
    die("the output must be int8");
  FILE *f = std::fopen(argv[3], "wb");
  if (!f || std::fwrite(out->data.int8, 1, out->bytes, f) != out->bytes ||
      std::fclose(f) != 0)
    die("cannot write", argv[3]);
  return 0;
}
