// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-import: builds the graph form from the #pragma knpu attributes that
// lcc puts on functions (docs/design.md 9.3). The pass never looks at the
// function bodies: the C loops of the NPU layers are only for running the
// same model on a CPU, and the functions of cpu layers stay for the CPU
// code.

#include "knpu/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::knpu {
#define GEN_PASS_DEF_KNPUIMPORT
#include "knpu/Transforms/Passes.h.inc"
} // namespace mlir::knpu

using namespace mlir;
using namespace mlir::knpu;

namespace {

constexpr llvm::StringLiteral kOpAttr = "knpu.op";
constexpr llvm::StringLiteral kArgsAttr = "knpu.args";

/// The knpu.args of one function, checked key by key. Errors are reported
/// at the function and name the pragma key.
class PragmaArgs {
public:
  PragmaArgs(func::FuncOp func, StringRef op, DictionaryAttr dict,
             SymbolTable &symbols)
      : func(func), op(op), dict(dict), symbols(symbols) {}

  InFlightDiagnostic error() {
    return func.emitError() << "#pragma knpu " << op << ": ";
  }
  InFlightDiagnostic error(StringRef key) {
    return error() << "'" << key << "' ";
  }

  /// Every key in @keys is present and there are no others.
  LogicalResult expectKeys(ArrayRef<StringRef> keys) {
    for (StringRef k : keys)
      if (!dict.get(k))
        return error() << "missing key '" << k << "'";
    for (NamedAttribute a : dict)
      if (!llvm::is_contained(keys, a.getName().getValue()))
        return error() << "unknown key '" << a.getName().getValue() << "'";
    return success();
  }

  FailureOr<LLVM::GlobalOp> global(StringRef key) {
    auto ref = dyn_cast<FlatSymbolRefAttr>(dict.get(key));
    if (!ref)
      return error(key) << "must name a file-scope object";
    auto g = symbols.lookup<LLVM::GlobalOp>(ref.getValue());
    if (!g)
      return error(key) << "names '" << ref.getValue()
                        << "', which is not a global variable";
    return g;
  }

  FailureOr<int64_t> integer(StringRef key) {
    auto v = dyn_cast<IntegerAttr>(dict.get(key));
    if (!v)
      return error(key) << "must be an integer";
    return v.getInt();
  }

  FailureOr<SmallVector<int64_t>> list(StringRef key, size_t n) {
    auto v = dyn_cast<DenseI64ArrayAttr>(dict.get(key));
    if (!v || v.size() != static_cast<int64_t>(n))
      return error(key) << "must be a list of " << n << " integers";
    return SmallVector<int64_t>(v.asArrayRef());
  }

  FailureOr<StringRef> string(StringRef key) {
    auto v = dyn_cast<StringAttr>(dict.get(key));
    if (!v)
      return error(key) << "must be a string";
    return v.getValue();
  }

private:
  func::FuncOp func;
  StringRef op;
  DictionaryAttr dict;
  SymbolTable &symbols;
};

/// Size in bytes of an integer or (nested) integer array type.
static std::optional<int64_t> byteSize(Type t) {
  if (auto arr = dyn_cast<LLVM::LLVMArrayType>(t)) {
    auto inner = byteSize(arr.getElementType());
    if (!inner)
      return std::nullopt;
    return *inner * static_cast<int64_t>(arr.getNumElements());
  }
  if (auto it = dyn_cast<IntegerType>(t))
    if (it.getWidth() % 8 == 0)
      return it.getWidth() / 8;
  return std::nullopt;
}

/// Initial bytes of a constant global. lcc writes initialized globals as a
/// byte string in target byte order; an all-zero global may come as a zero
/// initializer region instead.
static std::optional<std::string> initialBytes(LLVM::GlobalOp g,
                                               int64_t size) {
  if (auto s = dyn_cast_or_null<StringAttr>(g.getValueOrNull())) {
    if (static_cast<int64_t>(s.getValue().size()) != size)
      return std::nullopt;
    return s.getValue().str();
  }
  Region &init = g.getInitializerRegion();
  if (!g.getValueOrNull() && init.hasOneBlock() &&
      init.front().getOperations().size() == 2 &&
      isa<LLVM::ZeroOp>(init.front().front()))
    return std::string(size, '\0');
  return std::nullopt;
}

static int32_t readLE32(const std::string &bytes, size_t i) {
  uint32_t v = 0;
  for (int b = 3; b >= 0; --b)
    v = (v << 8) | static_cast<uint8_t>(bytes[4 * i + b]);
  return static_cast<int32_t>(v);
}

static bool isInt8(int64_t v) { return v >= -128 && v <= 127; }

static int64_t elements(ArrayRef<int64_t> hwc) {
  return hwc[0] * hwc[1] * hwc[2];
}

enum class Kind { Conv, Fc, AvgPool, Reshape, Cpu };

/// One layer function, checked. Shapes are (H, W, C); a fully connected
/// layer's are (1, 1, size), with a 1x1 kernel.
struct Layer {
  Kind kind;
  func::FuncOp func;
  StringRef op; // pragma operation
  StringRef in, out;
  SmallVector<int64_t> inShape, outShape, kernel, stride, pad, clamp;
  int64_t inZp = 0, outZp = 0; // a reshape's come from its input
  std::vector<int8_t> weight;
  std::vector<int32_t> bias, rq; // rq: multiplier, shift per channel
  SmallVector<LLVM::GlobalOp> globals;

  bool onNpu() const { return kind != Kind::Cpu; }
  /// The key that holds the input zero point
  StringRef inZpKey() const { return kind == Kind::AvgPool ? "zp" : "in_zp"; }
};

/// in and out: writable arrays of the layer's shapes.
static LogicalResult readActivations(PragmaArgs &args, Layer &l) {
  auto in = args.global("in"), out = args.global("out");
  if (failed(in) || failed(out))
    return failure();
  l.in = in->getSymName();
  l.out = out->getSymName();
  if (l.in == l.out)
    return args.error() << "'in' and 'out' must be different arrays";
  for (auto [key, g, shape] : {std::tuple{"in", *in, &l.inShape},
                               std::tuple{"out", *out, &l.outShape}}) {
    int64_t n = elements(*shape);
    if (g.getConstant())
      return args.error(key) << "must not be const";
    auto size = byteSize(g.getGlobalType());
    if (!size || *size != n)
      return args.error(key) << "must be an array of " << n << " bytes";
  }
  l.globals.append({*in, *out});
  return success();
}

/// w, bias, mult and shift: const arrays with initializers of the sizes the
/// layer's shapes give.
static LogicalResult readConstants(PragmaArgs &args, Layer &l,
                                   int64_t weightBytes) {
  auto w = args.global("w"), bias = args.global("bias"),
       mult = args.global("mult"), shift = args.global("shift");
  if (failed(w) || failed(bias) || failed(mult) || failed(shift))
    return failure();
  auto bytesOf = [&](StringRef key, LLVM::GlobalOp g,
                     int64_t size) -> std::optional<std::string> {
    auto gsize = byteSize(g.getGlobalType());
    if (!g.getConstant() || !gsize || *gsize != size) {
      args.error(key) << "must be a const array of " << size << " bytes";
      return std::nullopt;
    }
    auto bytes = initialBytes(g, size);
    if (!bytes)
      args.error(key) << "must have an initializer";
    return bytes;
  };
  int64_t oc = l.outShape[2];
  auto wBytes = bytesOf("w", *w, weightBytes);
  auto bBytes = bytesOf("bias", *bias, 4 * oc);
  auto mBytes = bytesOf("mult", *mult, 4 * oc);
  auto sBytes = bytesOf("shift", *shift, 4 * oc);
  if (!wBytes || !bBytes || !mBytes || !sBytes)
    return failure();

  l.weight.assign(wBytes->begin(), wBytes->end());
  for (int64_t c = 0; c < oc; ++c) {
    int32_t m = readLE32(*mBytes, c), s = readLE32(*sBytes, c);
    if (m < 0)
      return args.error("mult") << "values must be in 0..2^31-1";
    if (s < -31 || s > 30)
      return args.error("shift") << "values must be in -31..30";
    l.bias.push_back(readLE32(*bBytes, c));
    l.rq.push_back(m);
    l.rq.push_back(s);
  }
  l.globals.append({*w, *bias, *mult, *shift});
  return success();
}

static LogicalResult checkZeroPoints(PragmaArgs &args, const Layer &l) {
  if (!isInt8(l.inZp) || !isInt8(l.outZp))
    return args.error() << "zero points must be in -128..127";
  return success();
}

static LogicalResult checkPositive(PragmaArgs &args,
                                   ArrayRef<SmallVector<int64_t>> lists,
                                   StringRef what) {
  for (const SmallVector<int64_t> &list : lists)
    for (int64_t v : list)
      if (v < 1)
        return args.error() << what << " must be positive";
  return success();
}

/// #pragma knpu conv2d
static FailureOr<Layer> readConv(PragmaArgs &args, Layer l) {
  if (failed(args.expectKeys({"in", "out", "w", "bias", "mult", "shift",
                              "in_shape", "out_shape", "kernel", "stride",
                              "pad", "in_zp", "out_zp", "clamp"})))
    return failure();
  auto inShape = args.list("in_shape", 3), outShape = args.list("out_shape", 3),
       kernel = args.list("kernel", 2), stride = args.list("stride", 2),
       pad = args.list("pad", 4), clamp = args.list("clamp", 2);
  auto inZp = args.integer("in_zp"), outZp = args.integer("out_zp");
  if (failed(inShape) || failed(outShape) || failed(kernel) ||
      failed(stride) || failed(pad) || failed(clamp) || failed(inZp) ||
      failed(outZp))
    return failure();
  l.inShape = *inShape;
  l.outShape = *outShape;
  l.kernel = *kernel;
  l.stride = *stride;
  l.pad = *pad;
  l.clamp = *clamp;
  l.inZp = *inZp;
  l.outZp = *outZp;
  if (failed(checkZeroPoints(args, l)) ||
      failed(checkPositive(args, {l.inShape, l.outShape, l.kernel, l.stride},
                           "shapes, kernel and stride")) ||
      failed(readActivations(args, l)) ||
      failed(readConstants(args, l,
                           l.outShape[2] * l.kernel[0] * l.kernel[1] *
                               l.inShape[2])))
    return failure();
  return l;
}

/// #pragma knpu fc: a 1x1 convolution of a 1x1xIN input (docs/design.md
/// D7). The weights [OUT, IN] are already OHWI.
static FailureOr<Layer> readFc(PragmaArgs &args, Layer l) {
  if (failed(args.expectKeys({"in", "out", "w", "bias", "mult", "shift",
                              "in_size", "out_size", "in_zp", "out_zp",
                              "clamp"})))
    return failure();
  auto inSize = args.integer("in_size"), outSize = args.integer("out_size"),
       inZp = args.integer("in_zp"), outZp = args.integer("out_zp");
  auto clamp = args.list("clamp", 2);
  if (failed(inSize) || failed(outSize) || failed(inZp) || failed(outZp) ||
      failed(clamp))
    return failure();
  l.inShape = {1, 1, *inSize};
  l.outShape = {1, 1, *outSize};
  l.kernel = {1, 1};
  l.stride = {1, 1};
  l.pad = {0, 0, 0, 0};
  l.clamp = *clamp;
  l.inZp = *inZp;
  l.outZp = *outZp;
  if (failed(checkZeroPoints(args, l)) ||
      failed(checkPositive(args, {l.inShape, l.outShape}, "sizes")) ||
      failed(readActivations(args, l)) ||
      failed(readConstants(args, l, *outSize * *inSize)))
    return failure();
  return l;
}

/// #pragma knpu avgpool
static FailureOr<Layer> readAvgPool(PragmaArgs &args, Layer l) {
  if (failed(args.expectKeys({"in", "out", "in_shape", "out_shape", "kernel",
                              "stride", "pad", "zp", "clamp"})))
    return failure();
  auto inShape = args.list("in_shape", 3), outShape = args.list("out_shape", 3),
       kernel = args.list("kernel", 2), stride = args.list("stride", 2),
       pad = args.list("pad", 4), clamp = args.list("clamp", 2);
  auto zp = args.integer("zp");
  if (failed(inShape) || failed(outShape) || failed(kernel) ||
      failed(stride) || failed(pad) || failed(clamp) || failed(zp))
    return failure();
  l.inShape = *inShape;
  l.outShape = *outShape;
  l.kernel = *kernel;
  l.stride = *stride;
  l.pad = *pad;
  l.clamp = *clamp;
  l.inZp = l.outZp = *zp;
  if (failed(checkZeroPoints(args, l)) ||
      failed(checkPositive(args, {l.inShape, l.outShape, l.kernel, l.stride},
                           "shapes, kernel and stride")) ||
      failed(readActivations(args, l)))
    return failure();
  return l;
}

/// #pragma knpu reshape: the same bytes in another shape
static FailureOr<Layer> readReshape(PragmaArgs &args, Layer l) {
  if (failed(args.expectKeys({"in", "out", "in_shape", "out_shape"})))
    return failure();
  auto inShape = args.list("in_shape", 3), outShape = args.list("out_shape", 3);
  if (failed(inShape) || failed(outShape))
    return failure();
  l.inShape = *inShape;
  l.outShape = *outShape;
  if (failed(checkPositive(args, {l.inShape, l.outShape}, "shapes")))
    return failure();
  if (elements(l.inShape) != elements(l.outShape))
    return args.error() << "'in_shape' and 'out_shape' must have the same "
                           "number of elements";
  if (failed(readActivations(args, l)))
    return failure();
  return l;
}

/// #pragma knpu cpu: only the inputs and outputs of a CPU layer
static FailureOr<Layer> readCpu(PragmaArgs &args, Layer l) {
  if (failed(args.expectKeys(
          {"in", "out", "in_shape", "out_shape", "in_zp", "out_zp"})))
    return failure();
  auto inShape = args.list("in_shape", 3), outShape = args.list("out_shape", 3);
  auto inZp = args.integer("in_zp"), outZp = args.integer("out_zp");
  if (failed(inShape) || failed(outShape) || failed(inZp) || failed(outZp))
    return failure();
  l.inShape = *inShape;
  l.outShape = *outShape;
  l.inZp = *inZp;
  l.outZp = *outZp;
  if (failed(checkZeroPoints(args, l)) ||
      failed(checkPositive(args, {l.inShape, l.outShape}, "shapes")) ||
      failed(readActivations(args, l)))
    return failure();
  return l;
}

class ImportPass : public knpu::impl::KnpuImportBase<ImportPass> {
public:
  void runOnOperation() override;

private:
  void eraseDeadInternalSymbols(ModuleOp module);
};

void ImportPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbols(module);
  MLIRContext *ctx = &getContext();

  if (auto dl = module->getAttrOfType<StringAttr>("llvm.data_layout"))
    if (dl.getValue().starts_with("E")) {
      module.emitError("knpu-import reads constants as little-endian; the "
                       "module is big-endian");
      return signalPassFailure();
    }

  // The marked functions
  func::FuncOp model;
  llvm::StringMap<func::FuncOp> marked;
  for (auto func : module.getOps<func::FuncOp>()) {
    auto op = func->getAttrOfType<StringAttr>(kOpAttr);
    if (!op)
      continue;
    if (op.getValue() == "model") {
      if (model) {
        func.emitError("more than one #pragma knpu model function");
        return signalPassFailure();
      }
      model = func;
    } else {
      marked[func.getSymName()] = func;
    }
  }
  if (!model) {
    module.emitError("no function with #pragma knpu model");
    return signalPassFailure();
  }

  auto argsOf = [&](func::FuncOp f) {
    auto d = f->getAttrOfType<DictionaryAttr>(kArgsAttr);
    return d ? d : DictionaryAttr::get(ctx);
  };

  // Model arguments
  PragmaArgs margs(model, "model", argsOf(model), symbols);
  if (failed(margs.expectKeys({"in", "out", "in_scale", "in_zp", "out_scale",
                               "out_zp", "round"})))
    return signalPassFailure();
  auto mIn = margs.global("in"), mOut = margs.global("out");
  auto mInScale = margs.global("in_scale"), mOutScale = margs.global("out_scale");
  auto mInZp = margs.integer("in_zp"), mOutZp = margs.integer("out_zp");
  auto round = margs.string("round");
  if (failed(mIn) || failed(mOut) || failed(mInScale) || failed(mOutScale) ||
      failed(mInZp) || failed(mOutZp) || failed(round))
    return signalPassFailure();
  if (*round != "double" && *round != "single") {
    margs.error("round") << "must be \"double\" or \"single\"";
    return signalPassFailure();
  }
  auto scaleOf = [&](StringRef key, LLVM::GlobalOp g) -> std::optional<float> {
    auto f = dyn_cast_or_null<FloatAttr>(g.getValueOrNull());
    if (!g.getConstant() || !f || !f.getType().isF32()) {
      margs.error(key) << "must name a const float with an initializer";
      return std::nullopt;
    }
    float v = f.getValue().convertToFloat();
    if (!(v > 0.0f) || std::isinf(v)) {
      margs.error(key) << "must be positive and finite";
      return std::nullopt;
    }
    return v;
  };
  auto inScale = scaleOf("in_scale", *mInScale);
  auto outScale = scaleOf("out_scale", *mOutScale);
  if (!inScale || !outScale)
    return signalPassFailure();

  // The model body: calls of layer functions in order
  SmallVector<func::CallOp> calls;
  Region &body = model.getBody();
  for (auto [i, block] : llvm::enumerate(body)) {
    for (Operation &op : block) {
      if (auto call = dyn_cast<func::CallOp>(op)) {
        calls.push_back(call);
      } else if (auto br = dyn_cast<cf::BranchOp>(op)) {
        if (br.getDest() != block.getNextNode() || br.getNumOperands()) {
          op.emitError("the model function must be straight-line code");
          return signalPassFailure();
        }
      } else if (!isa<func::ReturnOp>(op)) {
        op.emitError("the model function may only call layer functions");
        return signalPassFailure();
      }
    }
  }

  // The layers in call order; cpu layers only after the last NPU layer
  std::vector<Layer> layers;
  bool afterCpu = false;
  for (func::CallOp call : calls) {
    auto it = marked.find(call.getCallee());
    if (it == marked.end()) {
      call.emitError() << "'" << call.getCallee()
                       << "' is not a #pragma knpu layer function";
      return signalPassFailure();
    }
    func::FuncOp f = it->second;
    StringRef op = f->getAttrOfType<StringAttr>(kOpAttr).getValue();
    PragmaArgs args(f, op, argsOf(f), symbols);
    Layer l;
    l.func = f;
    l.op = op;
    FailureOr<Layer> layer = failure();
    if (op == "conv2d") {
      l.kind = Kind::Conv;
      layer = readConv(args, l);
    } else if (op == "fc") {
      l.kind = Kind::Fc;
      layer = readFc(args, l);
    } else if (op == "avgpool") {
      l.kind = Kind::AvgPool;
      layer = readAvgPool(args, l);
    } else if (op == "reshape") {
      l.kind = Kind::Reshape;
      layer = readReshape(args, l);
    } else if (op == "cpu") {
      l.kind = Kind::Cpu;
      layer = readCpu(args, l);
    } else {
      f.emitError() << "#pragma knpu " << op << " is not supported";
      return signalPassFailure();
    }
    if (failed(layer))
      return signalPassFailure();
    if (layer->onNpu() && afterCpu) {
      args.error() << "comes after a cpu layer; cpu layers must follow the "
                      "last NPU layer";
      return signalPassFailure();
    }
    afterCpu |= !layer->onNpu();
    layers.push_back(std::move(*layer));
  }
  for (auto &entry : marked)
    if (llvm::none_of(layers,
                      [&](auto &l) { return l.func == entry.second; })) {
      func::FuncOp f = entry.second;
      f.emitError("#pragma knpu layer function is not called by the model "
                  "function");
      return signalPassFailure();
    }

  // Data flow by array name: what each array holds after every layer
  struct Tensor {
    SmallVector<int64_t> shape; // H, W, C
    int64_t zp;
    int index; // -1: model input, else producing layer
  };
  llvm::StringMap<Tensor> live;
  StringRef inName = mIn->getSymName(), outName = mOut->getSymName();
  std::optional<SmallVector<int64_t>> inShape;
  for (auto [i, l] : llvm::enumerate(layers)) {
    PragmaArgs args(l.func, l.op, argsOf(l.func), symbols);
    if (l.in == inName && !live.count(inName)) {
      if (l.kind == Kind::Reshape) {
        l.inZp = *mInZp;
      } else if (l.inZp != *mInZp) {
        args.error(l.inZpKey()) << "differs from in_zp of the model";
        return signalPassFailure();
      }
      inShape = l.inShape;
      live[inName] = {l.inShape, l.inZp, -1};
    }
    auto it = live.find(l.in);
    if (it == live.end()) {
      args.error("in") << "is neither the model input nor the output of an "
                          "earlier layer";
      return signalPassFailure();
    }
    // A fully connected layer reads the bytes of any shape as its vector
    bool shapeOk = l.kind == Kind::Fc
                       ? elements(it->second.shape) == l.inShape[2]
                       : it->second.shape == l.inShape;
    if (l.kind == Kind::Reshape)
      l.inZp = l.outZp = it->second.zp;
    if (!shapeOk || it->second.zp != l.inZp) {
      args.error("in") << "shape or zero point differs from where '" << l.in
                       << "' was written";
      return signalPassFailure();
    }
    live[l.out] = {l.outShape, l.outZp, static_cast<int>(i)};
  }
  if (!inShape) {
    model.emitError() << "no layer reads the model input '" << inName << "'";
    return signalPassFailure();
  }
  auto result = live.find(outName);
  if (result == live.end() || result->second.index < 0) {
    model.emitError() << "no layer writes the model output '" << outName
                      << "'";
    return signalPassFailure();
  }
  if (result->second.zp != *mOutZp) {
    margs.error("out_zp") << "differs from out_zp of the layer writing '"
                          << outName << "'";
    return signalPassFailure();
  }

  // Build the graph function in place of the model function
  OpBuilder b(model);
  Location loc = model.getLoc();
  Type i8 = b.getIntegerType(8), i32 = b.getIntegerType(32);
  auto actType = [&](ArrayRef<int64_t> hwc) {
    return RankedTensorType::get({1, hwc[0], hwc[1], hwc[2]}, i8);
  };
  auto fn = func::FuncOp::create(
      b, loc, kModelFunc,
      b.getFunctionType({actType(*inShape)},
                        {actType(result->second.shape)}));
  fn.setArgAttr(0, kScaleAttr, b.getF32FloatAttr(*inScale));
  fn.setArgAttr(0, kZeroPointAttr, b.getI64IntegerAttr(*mInZp));
  fn.setResultAttr(0, kScaleAttr, b.getF32FloatAttr(*outScale));
  fn.setResultAttr(0, kZeroPointAttr, b.getI64IntegerAttr(*mOutZp));

  b.setInsertionPointToStart(fn.addEntryBlock());
  llvm::StringMap<Value> values;
  values[inName] = fn.getArgument(0);
  for (Layer &l : layers) {
    // The layer function's name stays in the location of everything the
    // layer becomes, down to its commands
    Location lloc =
        NameLoc::get(b.getStringAttr(l.func.getSymName()), l.func.getLoc());
    auto reshape = [&](Value v, RankedTensorType type) -> Value {
      if (v.getType() == type)
        return v;
      Value shape =
          arith::ConstantOp::create(b, lloc, b.getIndexTensorAttr(
                                                 type.getShape()));
      return tensor::ReshapeOp::create(b, lloc, type, v, shape);
    };
    Value in = values[l.in], out;
    switch (l.kind) {
    case Kind::Conv:
    case Kind::Fc: {
      int64_t oc = l.outShape[2];
      auto constant = [&](RankedTensorType t, auto data) {
        return arith::ConstantOp::create(
                   b, lloc, DenseElementsAttr::get(t, llvm::ArrayRef(data)))
            .getResult();
      };
      Value w = constant(RankedTensorType::get(
                             {oc, l.kernel[0], l.kernel[1], l.inShape[2]}, i8),
                         l.weight);
      Value bias = constant(RankedTensorType::get({oc}, i32), l.bias);
      Value rq = constant(RankedTensorType::get({oc, 2}, i32), l.rq);
      out = Conv2DOp::create(
          b, lloc, actType(l.outShape), reshape(in, actType(l.inShape)), w,
          bias, rq, b.getDenseI64ArrayAttr(l.stride),
          b.getDenseI64ArrayAttr(l.pad), b.getI64IntegerAttr(l.inZp),
          b.getI64IntegerAttr(l.outZp), b.getDenseI64ArrayAttr(l.clamp),
          b.getStringAttr(*round));
      break;
    }
    case Kind::AvgPool:
      out = AvgPool2DOp::create(b, lloc, actType(l.outShape), in,
                                b.getDenseI64ArrayAttr(l.kernel),
                                b.getDenseI64ArrayAttr(l.stride),
                                b.getDenseI64ArrayAttr(l.pad),
                                b.getDenseI64ArrayAttr(l.clamp));
      break;
    case Kind::Reshape:
      out = reshape(in, actType(l.outShape));
      break;
    case Kind::Cpu:
      out = CpuOp::create(b, lloc, actType(l.outShape), in,
                          FlatSymbolRefAttr::get(l.func.getSymNameAttr()),
                          FlatSymbolRefAttr::get(ctx, l.in),
                          FlatSymbolRefAttr::get(ctx, l.out));
      break;
    }
    if (Operation *def = out.getDefiningOp(); def && failed(verify(def)))
      return signalPassFailure();
    values[l.out] = out;
  }
  func::ReturnOp::create(b, loc, ValueRange{values[outName]});

  // Remove what the graph replaced. The functions of cpu layers stay for
  // the CPU code, with external linkage so that it exports them.
  SetVector<Operation *> globals;
  globals.insert(*mIn);
  globals.insert(*mOut);
  globals.insert(*mInScale);
  globals.insert(*mOutScale);
  for (Layer &l : layers)
    for (LLVM::GlobalOp g : l.globals)
      globals.insert(g);
  model.erase();
  for (auto &entry : marked) {
    func::FuncOp f = entry.second;
    if (f->getAttrOfType<StringAttr>(kOpAttr).getValue() == "cpu") {
      f->removeAttr(kOpAttr);
      f->removeAttr(kArgsAttr);
      f->setAttr("llvm.linkage",
                 LLVM::LinkageAttr::get(ctx, LLVM::Linkage::External));
      continue;
    }
    if (!SymbolTable::symbolKnownUseEmpty(f, module)) {
      f.emitError("#pragma knpu layer function is also used outside the "
                  "model function");
      return signalPassFailure();
    }
    f.erase();
  }
  for (Operation *g : globals)
    if (SymbolTable::symbolKnownUseEmpty(g, module))
      g->erase();
  eraseDeadInternalSymbols(module);
}

/// Erases functions and globals with internal linkage that nothing uses any
/// more, such as helpers only the layer function bodies called.
void ImportPass::eraseDeadInternalSymbols(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : llvm::make_early_inc_range(*module.getBody())) {
      bool internal = false;
      if (auto g = dyn_cast<LLVM::GlobalOp>(op))
        internal = g.getLinkage() == LLVM::Linkage::Internal;
      else if (auto f = dyn_cast<func::FuncOp>(op))
        if (auto l = f->getAttrOfType<LLVM::LinkageAttr>("llvm.linkage"))
          internal = l.getLinkage() == LLVM::Linkage::Internal;
      if (internal && SymbolTable::symbolKnownUseEmpty(&op, module)) {
        op.erase();
        changed = true;
      }
    }
  }
}

} // namespace
