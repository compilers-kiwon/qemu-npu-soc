// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-schedule: places tensors in the arena (region 1), constants in
// region 0 and both in SRAM, and lowers the graph to command operations.
//
// Per layer the stream is: DMA of the layer's constants, DMA of tile 0's
// input, then for every tile k the DMA of tile k+1's input (into the other
// input buffer), the convolution of tile k and the DMA of its output. Wait
// commands are left to knpu-insert-waits.

#include "knpu/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::knpu {
#define GEN_PASS_DEF_KNPUSCHEDULE
#include "knpu/Transforms/Passes.h.inc"
} // namespace mlir::knpu

using namespace mlir;
using namespace mlir::knpu;

namespace {

constexpr int64_t kConstRegion = 0;
constexpr int64_t kArenaRegion = 1;

/// One knpu.conv2d: rows [srcRow, srcRow + inRows) of the arena tensor src
/// in, rows [dstRow, dstRow + outRows) of the arena tensor dst out.
struct Tile {
  Conv2DOp op;
  Value src;
  int64_t srcRow, inRows;
  int64_t dstRow, outRows;
};

/// The tiles writing one arena tensor.
struct Layer {
  Value src, weight, bias, rq;
  SmallVector<Tile> tiles;
  int64_t constOffset = 0; // in region 0
};

/// Static slice [0, row, 0, 0] [1, rows, W, C] [1, 1, 1, 1] of a 1xHxWxC
/// tensor; returns (row, rows).
template <typename SliceOp>
std::optional<std::pair<int64_t, int64_t>> rowSlice(SliceOp op,
                                                    ShapedType full) {
  ArrayRef<int64_t> off = op.getStaticOffsets(), size = op.getStaticSizes(),
                    stride = op.getStaticStrides();
  if (off.size() != 4 || llvm::any_of(stride, [](int64_t s) { return s != 1; }))
    return std::nullopt;
  if (ShapedType::isDynamic(off[1]) || ShapedType::isDynamic(size[1]) ||
      off[0] != 0 || off[2] != 0 || off[3] != 0 || size[0] != 1 ||
      size[2] != full.getDimSize(2) || size[3] != full.getDimSize(3))
    return std::nullopt;
  return std::make_pair(off[1], size[1]);
}

std::vector<int8_t> denseBytes(Value v, bool *ok) {
  std::vector<int8_t> out;
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  auto attr = cst ? dyn_cast<DenseIntElementsAttr>(cst.getValue()) : nullptr;
  if (!attr) {
    *ok = false;
    return out;
  }
  unsigned width = attr.getElementType().getIntOrFloatBitWidth();
  for (const APInt &x : attr.getValues<APInt>()) {
    uint64_t u = x.getZExtValue();
    for (unsigned i = 0; i < width / 8; ++i) // little-endian
      out.push_back(static_cast<int8_t>(u >> (8 * i)));
  }
  return out;
}

int64_t bytesOf(Value v) {
  return cast<ShapedType>(v.getType()).getNumElements();
}

class SchedulePass : public knpu::impl::KnpuScheduleBase<SchedulePass> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  LogicalResult collect(func::FuncOp fn);

  llvm::SetVector<Value> arena;          // in placement order
  llvm::MapVector<Value, Layer> layers;  // keyed by the written tensor
};

LogicalResult SchedulePass::collect(func::FuncOp fn) {
  for (Value arg : fn.getArguments())
    arena.insert(arg);

  for (Operation &op : fn.getBody().front()) {
    if (isa<arith::ConstantOp, tensor::EmptyOp, tensor::ExtractSliceOp,
            tensor::InsertSliceOp, func::ReturnOp>(op))
      continue;
    auto conv = dyn_cast<Conv2DOp>(op);
    if (!conv)
      return op.emitOpError("cannot be scheduled on the NPU");

    Tile t{conv, conv.getInput(), 0, conv.getInput().getType().getDimSize(1),
           0, conv.getOutput().getType().getDimSize(1)};
    if (auto slice = conv.getInput().getDefiningOp<tensor::ExtractSliceOp>()) {
      auto rows = rowSlice(slice, slice.getSourceType());
      if (!rows)
        return slice.emitOpError("must take whole rows of its source");
      t.src = slice.getSource();
      t.srcRow = rows->first;
    }
    if (!arena.contains(t.src) && !layers.count(t.src))
      return conv.emitOpError("input must be the model input or a whole "
                              "layer output");

    // The written tensor: the result itself, or the end of the
    // tensor.insert_slice chain the result goes into
    Value dst = conv.getOutput();
    if (dst.hasOneUse()) {
      if (auto ins = dyn_cast<tensor::InsertSliceOp>(*dst.user_begin());
          ins && ins.getSource() == dst) {
        auto rows = rowSlice(ins, ins.getDestType());
        if (!rows)
          return ins.emitOpError("must fill whole rows of its destination");
        t.dstRow = rows->first;
        dst = ins.getResult();
        while (dst.hasOneUse()) {
          auto next = dyn_cast<tensor::InsertSliceOp>(*dst.user_begin());
          if (!next || next.getDest() != dst)
            break;
          dst = next.getResult();
        }
      }
    }

    Layer &layer = layers[dst];
    if (layer.tiles.empty()) {
      layer.src = t.src;
      layer.weight = conv.getWeight();
      layer.bias = conv.getBias();
      layer.rq = conv.getRq();
    } else if (layer.src != t.src || layer.weight != conv.getWeight() ||
               layer.bias != conv.getBias() || layer.rq != conv.getRq()) {
      return conv.emitOpError("tiles of one output must share their input "
                              "and constants");
    }
    layer.tiles.push_back(t);
  }

  // Every layer output must be covered exactly once, tile after tile
  for (auto &[dst, layer] : layers) {
    llvm::sort(layer.tiles, [](const Tile &a, const Tile &b) {
      return a.dstRow < b.dstRow;
    });
    int64_t next = 0;
    for (Tile &t : layer.tiles) {
      if (t.dstRow != next)
        return t.op.emitOpError("tiles must cover the output rows once");
      next += t.outRows;
    }
    if (next != cast<ShapedType>(dst.getType()).getDimSize(1))
      return layer.tiles.front().op.emitOpError(
          "tiles must cover the output rows once");
    arena.insert(dst);
  }

  auto ret = cast<func::ReturnOp>(fn.getBody().front().getTerminator());
  for (Value v : ret.getOperands())
    if (!layers.count(v))
      return ret.emitOpError("must return layer outputs");
  if (layers.empty())
    return fn.emitOpError("has no layer to schedule");
  return success();
}

void SchedulePass::runOnOperation() {
  ModuleOp module = getOperation();
  auto fn = module.lookupSymbol<func::FuncOp>(kModelFunc);
  if (!fn || fn.getNumArguments() == 0) {
    module.emitError("no knpu_model function in graph form "
                     "(run knpu-import first)");
    return signalPassFailure();
  }
  if (failed(collect(fn)))
    return signalPassFailure();

  OpBuilder b(fn);

  // Arena: model inputs, then the layer outputs in order
  llvm::DenseMap<Value, int64_t> arenaOffset;
  int64_t arenaSize = 0;
  for (Value v : arena) {
    arenaSize = alignTo16(arenaSize);
    arenaOffset[v] = arenaSize;
    arenaSize += bytesOf(v);
  }

  // Region 0: per layer weights, bias and requantization table, each
  // 16-byte aligned (convConstBytes)
  std::vector<int8_t> constData;
  for (auto &[dst, layer] : layers) {
    bool ok = true;
    std::vector<int8_t> parts[] = {denseBytes(layer.weight, &ok),
                                   denseBytes(layer.bias, &ok),
                                   denseBytes(layer.rq, &ok)};
    if (!ok) {
      layer.tiles.front().op.emitOpError("constants must be dense");
      return signalPassFailure();
    }
    constData.resize(alignTo16(constData.size()));
    layer.constOffset = constData.size();
    for (auto [i, part] : llvm::enumerate(parts)) {
      if (i)
        constData.resize(alignTo16(constData.size()));
      llvm::append_range(constData, part);
    }
  }

  // The command function
  auto program = func::FuncOp::create(b, fn.getLoc(), kModelFunc,
                                      b.getFunctionType({}, {}));
  b.setInsertionPointToStart(program.addEntryBlock());
  auto dma = [&](Location loc, StringRef dir, int64_t region,
                 int64_t regionOffset, int64_t sramOffset, int64_t length) {
    DmaOp::create(b, loc, b.getStringAttr(dir), b.getI64IntegerAttr(region),
                  b.getI64IntegerAttr(regionOffset),
                  b.getI64IntegerAttr(sramOffset),
                  b.getI64IntegerAttr(length), b.getI64IntegerAttr(1),
                  b.getI64IntegerAttr(0));
  };

  for (auto &[dst, layer] : layers) {
    Conv2DOp first = layer.tiles.front().op;
    auto inType = first.getInput().getType();
    auto wType = first.getWeight().getType();
    auto outType = cast<ShapedType>(dst.getType());
    int64_t iw = inType.getDimSize(2), ic = inType.getDimSize(3);
    int64_t ow = outType.getDimSize(2), oc = outType.getDimSize(3);
    int64_t kh = wType.getDimSize(1), kw = wType.getDimSize(2);

    int64_t wBytes = oc * kh * kw * ic;
    int64_t constBytes = convConstBytes(oc, kh, kw, ic);
    int64_t maxIn = 0, maxOut = 0;
    for (const Tile &t : layer.tiles) {
      maxIn = std::max(maxIn, t.inRows * iw * ic);
      maxOut = std::max(maxOut, t.outRows * ow * oc);
    }
    bool db = layer.tiles.size() > 1;
    int64_t need = convSramBytes(constBytes, maxIn, maxOut, db);
    if (need > sramSize) {
      first.emitOpError() << "needs " << need << " bytes of SRAM but "
                          << static_cast<int64_t>(sramSize)
                          << " are given (run knpu-tile with the same "
                             "sram-size)";
      return signalPassFailure();
    }

    // SRAM: constants at 0, then input and output buffers
    int64_t inBuf[2], outBuf[2];
    inBuf[0] = alignTo16(constBytes);
    inBuf[1] = inBuf[0] + (db ? alignTo16(maxIn) : 0);
    outBuf[0] = inBuf[0] + (db ? 2 : 1) * alignTo16(maxIn);
    outBuf[1] = outBuf[0] + (db ? alignTo16(maxOut) : 0);

    dma(first.getLoc(), "load", kConstRegion, layer.constOffset, 0,
        constBytes);
    int64_t srcBase = arenaOffset[layer.src], dstBase = arenaOffset[dst];
    auto load = [&](size_t k) {
      Tile t = layer.tiles[k];
      dma(t.op.getLoc(), "load", kArenaRegion, srcBase + t.srcRow * iw * ic,
          inBuf[k % 2], t.inRows * iw * ic);
    };
    load(0);
    for (size_t k = 0; k < layer.tiles.size(); ++k) {
      Tile t = layer.tiles[k];
      if (k + 1 < layer.tiles.size())
        load(k + 1);
      Conv2DOp op = t.op;
      ConvOp::create(
          b, op.getLoc(), b.getI64IntegerAttr(inBuf[k % 2]),
          b.getDenseI64ArrayAttr({t.inRows, iw, ic}), op.getInputZpAttr(),
          b.getI64IntegerAttr(outBuf[k % 2]),
          b.getDenseI64ArrayAttr({t.outRows, ow, oc}), op.getOutputZpAttr(),
          b.getI64IntegerAttr(0), b.getI64IntegerAttr(alignTo16(wBytes)),
          b.getI64IntegerAttr(alignTo16(wBytes) + alignTo16(4 * oc)),
          b.getDenseI64ArrayAttr({kh, kw}), op.getStrideAttr(),
          op.getPadAttr(), op.getClampAttr(), op.getRoundAttr());
      dma(op.getLoc(), "store", kArenaRegion, dstBase + t.dstRow * ow * oc,
          outBuf[k % 2], t.outRows * ow * oc);
    }
  }
  StopOp::create(b, fn.getLoc());
  func::ReturnOp::create(b, fn.getLoc());

  // Region table and model inputs and outputs, ahead of the program
  b.setInsertionPoint(program);
  if (!constData.empty()) {
    auto type = RankedTensorType::get(
        {static_cast<int64_t>(constData.size())}, b.getIntegerType(8));
    RegionOp::create(b, fn.getLoc(), b.getI64IntegerAttr(kConstRegion),
                     b.getStringAttr("const"),
                     b.getI64IntegerAttr(constData.size()),
                     DenseElementsAttr::get(type, llvm::ArrayRef(constData)));
  }
  RegionOp::create(b, fn.getLoc(), b.getI64IntegerAttr(kArenaRegion),
                   b.getStringAttr("arena"), b.getI64IntegerAttr(arenaSize),
                   ElementsAttr());
  auto io = [&](StringRef dir, Value v, DictionaryAttr attrs) {
    auto scale = attrs ? attrs.getAs<FloatAttr>(kScaleAttr) : nullptr;
    auto zp = attrs ? attrs.getAs<IntegerAttr>(kZeroPointAttr) : nullptr;
    if (!scale || !zp) {
      fn.emitOpError() << "model " << dir << "s need " << kScaleAttr
                       << " and " << kZeroPointAttr << " attributes";
      return failure();
    }
    IoOp::create(b, fn.getLoc(), b.getStringAttr(dir),
                 b.getI64IntegerAttr(kArenaRegion),
                 b.getI64IntegerAttr(arenaOffset[v]),
                 b.getDenseI64ArrayAttr(cast<ShapedType>(v.getType()).getShape()),
                 b.getF32FloatAttr(scale.getValueAsDouble()), zp);
    return success();
  };
  auto ret = cast<func::ReturnOp>(fn.getBody().front().getTerminator());
  for (auto [i, arg] : llvm::enumerate(fn.getArguments()))
    if (failed(io("input", arg, fn.getArgAttrDict(i))))
      return signalPassFailure();
  for (auto [i, v] : llvm::enumerate(ret.getOperands()))
    if (failed(io("output", v, fn.getResultAttrDict(i))))
      return signalPassFailure();

  module->setAttr(kSramSizeAttr, b.getI64IntegerAttr(sramSize));
  fn.erase();
}

} // namespace
