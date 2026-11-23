// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "knpu/Dialect/Knpu/KnpuOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::knpu;

#define GET_OP_CLASSES
#include "knpu/Dialect/Knpu/KnpuOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

std::string mlir::knpu::checkWindowShape(const WindowShape &s) {
  // Field widths of SET_IFM_SHAPE / SET_OFM_SHAPE, SET_KERNEL, SET_STRIDE
  // and SET_PAD (docs/npu-spec.md 6.1)
  for (int64_t v : {s.ih, s.iw, s.ic, s.oh, s.ow, s.oc})
    if (v < 1 || v > 0xFFFF)
      return "shape dimensions must be in 1..65535";
  for (int64_t v : {s.kh, s.kw, s.sh, s.sw})
    if (v < 1 || v > 0xFF)
      return "kernel and stride must be in 1..255";
  for (int64_t v : {s.pt, s.pl, s.pb, s.pr})
    if (v < 0 || v > 0xF)
      return "padding must be in 0..15";
  // Window conditions (docs/npu-spec.md 6.4)
  if (s.pt >= s.kh || s.pb >= s.kh || s.pl >= s.kw || s.pr >= s.kw)
    return "padding must be smaller than the kernel";
  if (s.ih + s.pt + s.pb < s.kh || s.iw + s.pl + s.pr < s.kw)
    return "padded input is smaller than the kernel";
  int64_t oh = (s.ih + s.pt + s.pb - s.kh) / s.sh + 1;
  int64_t ow = (s.iw + s.pl + s.pr - s.kw) / s.sw + 1;
  if (s.oh != oh || s.ow != ow)
    return llvm::formatv("output must be {0}x{1} for this input, kernel, "
                         "stride and padding",
                         oh, ow)
        .str();
  return "";
}

static bool isInt8(int64_t v) { return v >= -128 && v <= 127; }

static LogicalResult verifyClamp(Operation *op, ArrayRef<int64_t> clamp) {
  if (clamp.size() != 2 || !isInt8(clamp[0]) || !isInt8(clamp[1]) ||
      clamp[0] > clamp[1])
    return op->emitOpError("clamp must be (min, max) in -128..127 with "
                           "min <= max");
  return success();
}

static LogicalResult verifyQuant(Operation *op, int64_t inZp, int64_t outZp,
                                 ArrayRef<int64_t> clamp, StringRef round) {
  if (!isInt8(inZp) || !isInt8(outZp))
    return op->emitOpError("zero points must be in -128..127");
  if (failed(verifyClamp(op, clamp)))
    return failure();
  if (round != "double" && round != "single")
    return op->emitOpError("round must be \"double\" or \"single\"");
  return success();
}

/// The shapes of a graph-level window operation: input and output must be
/// static 1xHxWxC tensors.
static LogicalResult verifyActivations(Operation *op, ShapedType in,
                                       ShapedType out) {
  if (!in.hasStaticShape() || !out.hasStaticShape())
    return op->emitOpError("operands and result must have static shapes");
  if (in.getDimSize(0) != 1 || out.getDimSize(0) != 1)
    return op->emitOpError("batch size must be 1");
  return success();
}

/// SRAM size from the enclosing module, or 0 when it is not known yet.
static int64_t sramSizeOf(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return 0;
  auto attr = module->getAttrOfType<IntegerAttr>(kSramSizeAttr);
  return attr ? attr.getInt() : 0;
}

namespace {
struct Range {
  int64_t begin, size;
  const char *what;
};
} // namespace

static LogicalResult verifySramRanges(Operation *op, ArrayRef<Range> ranges) {
  int64_t sram = sramSizeOf(op);
  for (const Range &r : ranges) {
    if (r.begin < 0 || r.begin > 0xFFFFFFFFll)
      return op->emitOpError() << r.what << " offset out of range";
    if (sram && r.begin + r.size > sram)
      return op->emitOpError()
             << r.what << " [" << r.begin << ", " << r.begin + r.size
             << ") is outside the " << sram << "-byte SRAM";
  }
  return success();
}

/// SRAM range and overlap checks of an operation command
/// (docs/npu-spec.md 6.4): everything inside SRAM, and the OFM apart from
/// everything the command reads.
static LogicalResult verifyCommandRanges(Operation *op, ArrayRef<Range> reads,
                                         Range ofm) {
  if (failed(verifySramRanges(op, reads)) ||
      failed(verifySramRanges(op, {ofm})))
    return failure();
  for (const Range &r : reads)
    if (ofm.begin < r.begin + r.size && r.begin < ofm.begin + ofm.size)
      return op->emitOpError() << "OFM overlaps the " << r.what;
  return success();
}

//===----------------------------------------------------------------------===//
// Graph operations
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::verify() {
  auto in = getInput().getType();
  auto w = getWeight().getType();
  auto b = getBias().getType();
  auto rq = getRq().getType();
  auto out = getOutput().getType();
  for (ShapedType t : {ShapedType(in), ShapedType(w), ShapedType(b),
                       ShapedType(rq), ShapedType(out)})
    if (!t.hasStaticShape())
      return emitOpError("operands and result must have static shapes");
  if (in.getDimSize(0) != 1 || out.getDimSize(0) != 1)
    return emitOpError("batch size must be 1");
  int64_t oc = w.getDimSize(0);
  if (w.getDimSize(3) != in.getDimSize(3))
    return emitOpError("weight input channels must match the input");
  if (out.getDimSize(3) != oc || b.getDimSize(0) != oc ||
      rq.getDimSize(0) != oc || rq.getDimSize(1) != 2)
    return emitOpError("output, bias and requantization table must have one "
                       "entry per weight output channel");
  if (getStride().size() != 2 || getPad().size() != 4)
    return emitOpError("stride must have 2 values and pad 4");

  WindowShape s{in.getDimSize(1),  in.getDimSize(2),  in.getDimSize(3),
                out.getDimSize(1), out.getDimSize(2), oc,
                w.getDimSize(1),   w.getDimSize(2),   getStride()[0],
                getStride()[1],    getPad()[0],       getPad()[1],
                getPad()[2],       getPad()[3]};
  std::string err = checkWindowShape(s);
  if (!err.empty())
    return emitOpError(err);
  return verifyQuant(*this, getInputZp(), getOutputZp(), getClamp(),
                     getRound());
}

LogicalResult AvgPool2DOp::verify() {
  auto in = getInput().getType();
  auto out = getOutput().getType();
  if (failed(verifyActivations(*this, in, out)))
    return failure();
  if (out.getDimSize(3) != in.getDimSize(3))
    return emitOpError("output channels must match the input");
  if (getKernel().size() != 2 || getStride().size() != 2 ||
      getPad().size() != 4)
    return emitOpError("kernel and stride must have 2 values and pad 4");

  WindowShape s{in.getDimSize(1),  in.getDimSize(2),  in.getDimSize(3),
                out.getDimSize(1), out.getDimSize(2), out.getDimSize(3),
                getKernel()[0],    getKernel()[1],    getStride()[0],
                getStride()[1],    getPad()[0],       getPad()[1],
                getPad()[2],       getPad()[3]};
  std::string err = checkWindowShape(s);
  if (!err.empty())
    return emitOpError(err);
  return verifyClamp(*this, getClamp());
}

LogicalResult CpuOp::verify() {
  return verifyActivations(*this, getInput().getType(),
                           getOutput().getType());
}

//===----------------------------------------------------------------------===//
// Command operations
//===----------------------------------------------------------------------===//

LogicalResult DmaOp::verify() {
  // I64Attr getters return uint64_t; compare as signed values
  int64_t region = getRegion(), length = getLength(), count = getCount(),
          stride = getStride(), regionOffset = getRegionOffset(),
          sramOffset = getSramOffset();
  if (getDir() != "load" && getDir() != "store")
    return emitOpError("dir must be \"load\" or \"store\"");
  if (region < 0 || region > 7)
    return emitOpError("region must be in 0..7");
  if (length < 1 || length > 0xFFFFFFFFll)
    return emitOpError("length must be in 1..2^32-1");
  if (count < 1 || count > 0xFFFF)
    return emitOpError("count must be in 1..65535");
  if (stride < 0 || stride > 0xFFFFFFFFll)
    return emitOpError("stride must be in 0..2^32-1");
  if (regionOffset < 0 || regionOffset > 0xFFFFFFFFll)
    return emitOpError("region offset must be in 0..2^32-1");
  return verifySramRanges(*this,
                          {{sramOffset, length * count, "SRAM side"}});
}

LogicalResult ConvOp::verify() {
  if (getIfmShape().size() != 3 || getOfmShape().size() != 3 ||
      getKernel().size() != 2 || getStride().size() != 2 ||
      getPad().size() != 4)
    return emitOpError("shapes need 3 values, kernel and stride 2, pad 4");
  WindowShape s{getIfmShape()[0], getIfmShape()[1], getIfmShape()[2],
                getOfmShape()[0], getOfmShape()[1], getOfmShape()[2],
                getKernel()[0],   getKernel()[1],   getStride()[0],
                getStride()[1],   getPad()[0],      getPad()[1],
                getPad()[2],      getPad()[3]};
  std::string err = checkWindowShape(s);
  if (!err.empty())
    return emitOpError(err);
  if (failed(verifyQuant(*this, getIfmZp(), getOfmZp(), getClamp(),
                         getRound())))
    return failure();
  int64_t ifm = getIfm(), ofm = getOfm(), weight = getWeight(),
          bias = getBias(), rq = getRqTable();
  if (bias % 4 || rq % 4)
    return emitOpError("bias and requantization table offsets must be "
                       "multiples of 4");
  return verifyCommandRanges(*this,
                             {{ifm, s.ih * s.iw * s.ic, "IFM"},
                              {weight, s.oc * s.kh * s.kw * s.ic, "weights"},
                              {bias, 4 * s.oc, "bias"},
                              {rq, 8 * s.oc, "requantization table"}},
                             {ofm, s.oh * s.ow * s.oc, "OFM"});
}

LogicalResult PoolOp::verify() {
  if (getIfmShape().size() != 3 || getOfmShape().size() != 3 ||
      getKernel().size() != 2 || getStride().size() != 2 ||
      getPad().size() != 4)
    return emitOpError("shapes need 3 values, kernel and stride 2, pad 4");
  WindowShape s{getIfmShape()[0], getIfmShape()[1], getIfmShape()[2],
                getOfmShape()[0], getOfmShape()[1], getOfmShape()[2],
                getKernel()[0],   getKernel()[1],   getStride()[0],
                getStride()[1],   getPad()[0],      getPad()[1],
                getPad()[2],      getPad()[3]};
  std::string err = checkWindowShape(s);
  if (!err.empty())
    return emitOpError(err);
  if (s.oc != s.ic)
    return emitOpError("output channels must match the input");
  if (failed(verifyClamp(*this, getClamp())))
    return failure();
  int64_t ifm = getIfm(), ofm = getOfm();
  return verifyCommandRanges(*this, {{ifm, s.ih * s.iw * s.ic, "IFM"}},
                             {ofm, s.oh * s.ow * s.oc, "OFM"});
}

LogicalResult DmaWaitOp::verify() {
  int64_t pending = getPending();
  if (pending < 0 || pending > 15)
    return emitOpError("pending must be in 0..15");
  return success();
}

LogicalResult KernelWaitOp::verify() {
  int64_t pending = getPending();
  if (pending < 0 || pending > 15)
    return emitOpError("pending must be in 0..15");
  return success();
}

//===----------------------------------------------------------------------===//
// Model file description
//===----------------------------------------------------------------------===//

LogicalResult RegionOp::verify() {
  int64_t index = getIndex(), size = getSize();
  if (index < 0 || index > 7)
    return emitOpError("index must be in 0..7");
  if (size < 1 || size > 0xFFFFFFFFll)
    return emitOpError("size must be in 1..2^32-1");
  if (getRole() == "arena") {
    if (getContent())
      return emitOpError("an arena region has no content");
    return success();
  }
  if (getRole() != "const")
    return emitOpError("role must be \"const\" or \"arena\"");
  auto content = getContent();
  if (!content)
    return emitOpError("a constant region needs its content");
  auto type = llvm::cast<ShapedType>(content->getType());
  if (type.getRank() != 1 || !type.getElementType().isInteger(8) ||
      type.getDimSize(0) != size)
    return emitOpError("content must be a 1-D i8 tensor of size elements");
  return success();
}

LogicalResult IoOp::verify() {
  if (getDirection() != "input" && getDirection() != "output")
    return emitOpError("direction must be \"input\" or \"output\"");
  if (getShape().empty() || getShape().size() > 4)
    return emitOpError("shape must have 1 to 4 dimensions");
  for (int64_t d : getShape())
    if (d < 1)
      return emitOpError("dimensions must be at least 1");
  float scale = getScale().convertToFloat();
  if (!(scale > 0.0f) || std::isinf(scale))
    return emitOpError("scale must be positive and finite");
  if (!isInt8(getZeroPoint()))
    return emitOpError("zero point must be in -128..127");
  int64_t region = getRegion(), offset = getOffset();
  if (region < 0 || region > 7 || offset < 0 || offset > 0xFFFFFFFFll)
    return emitOpError("bad region or offset");
  return success();
}

LogicalResult CpuStepOp::verify() {
  StringRef kind = getKind();
  bool copy = kind == "copy_in" || kind == "copy_out";
  if (!copy && kind != "call")
    return emitOpError("kind must be \"copy_in\", \"call\" or \"copy_out\"");
  if (!copy) {
    if (getOffset() || getSize())
      return emitOpError("a call has no arena offset or size");
    return success();
  }
  if (!getOffset() || !getSize())
    return emitOpError("a copy needs an arena offset and a size");
  int64_t offset = *getOffset(), size = *getSize();
  if (offset < 0 || offset > 0xFFFFFFFFll)
    return emitOpError("arena offset must be in 0..2^32-1");
  if (size < 1 || size > 0xFFFFFFFFll)
    return emitOpError("size must be in 1..2^32-1");
  return success();
}

LogicalResult CpuCodeOp::verify() {
  auto type = llvm::cast<ShapedType>(getCode().getType());
  if (type.getRank() != 1 || !type.getElementType().isInteger(8) ||
      type.getDimSize(0) < 1)
    return emitOpError("code must be a non-empty 1-D i8 tensor");
  return success();
}
