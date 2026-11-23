// V2: runs a .tflite model with the TensorFlow Lite Micro interpreter.
//
//   tflm_run_<mode> MODEL INPUT OUTPUT [LAST_INPUT]
//
// INPUT holds the int8 bytes of the model's input tensor; the int8 bytes of
// the output tensor are written to OUTPUT. With LAST_INPUT, the int8 bytes
// of the last operator's first input are written there too (the logits of a
// classifier that ends in SOFTMAX); the interpreter then keeps every tensor
// (preserve_all_tensors). The program is built once per rounding mode
// (TFLITE_SINGLE_ROUNDING 0 and 1).
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

void writeFile(const char *path, const void *data, size_t size) {
  FILE *f = std::fopen(path, "wb");
  if (!f || std::fwrite(data, 1, size, f) != size || std::fclose(f) != 0)
    die("cannot write", path);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4 && argc != 5)
    die("usage: tflm_run MODEL INPUT OUTPUT [LAST_INPUT]");
  bool keepAll = argc == 5;

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

  tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof arena,
                                       nullptr, nullptr, keepAll);
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
  writeFile(argv[3], out->data.int8, out->bytes);

  if (keepAll) {
    const auto *ops = model->subgraphs()->Get(0)->operators();
    const tflite::Operator *last =
        ops && ops->size() ? ops->Get(ops->size() - 1) : nullptr;
    if (!last || !last->inputs() || last->inputs()->size() == 0)
      die("the model has no last operator with an input");
    const TfLiteEvalTensor *t = interpreter.GetTensor(last->inputs()->Get(0));
    if (!t || t->type != kTfLiteInt8)
      die("the last operator's first input must be int8");
    size_t bytes = 1;
    for (int i = 0; i < t->dims->size; ++i)
      bytes *= static_cast<size_t>(t->dims->data[i]);
    writeFile(argv[4], t->data.int8, bytes);
  }
  return 0;
}
