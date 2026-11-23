// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-schedule: places constants in region 0 and tensors in the arena
// (region 1), lays out SRAM by the lifetimes of its buffers and lowers the
// graph to command operations (docs/design.md 9.4).
//
// The NPU layers run in program order as steps. With overlap (the default):
// - An activation that whole (one-tile) layers read stays in SRAM from the
//   layer that writes it to the last layer that reads it. Model inputs and
//   outputs, tensors of cpu layers and tensors of tiled layers go through
//   the arena.
// - Constants are double-buffered: the constants of a layer are loaded
//   while the previous layer with constants runs, in front of that layer's
//   first operation and after its own loads.
// - In a tiled layer the next tile's input is loaded in front of the
//   current tile's operation; inputs and outputs have two buffers each.
// Without overlap (the serial schedule to compare with) every layer loads
// its constants and every tile its input, and stores its output, one
// buffer each.
//
// SRAM layout: every buffer lives from a first to a last step. Buffers
// that live over several steps are placed first, largest first (as the
// greedy memory planner of TFLite Micro does), then the buffers of single
// steps in the order they were made (constants, inputs, outputs), each at
// the lowest 16-byte aligned offset that no buffer alive at the same time
// uses. Constants loaded ahead keep apart, where room allows, from the
// buffers of the two layers before the one they are loaded in, which may
// still be running (operation queue depth 2), so that the load need not
// wait for them. When the layout does not fit, the largest buffer alive
// with the one that failed and loaded ahead or kept in SRAM is demoted one
// level (ahead without the margin, then not ahead; kept, then through the
// arena) and the layout is made again. Wait commands are left to
// knpu-insert-waits.

#include "knpu/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"

#include <array>
#include <numeric>

namespace mlir::knpu {
#define GEN_PASS_DEF_KNPUSCHEDULE
#include "knpu/Transforms/Passes.h.inc"
} // namespace mlir::knpu

using namespace mlir;
using namespace mlir::knpu;

namespace {

constexpr int64_t kConstRegion = 0;
constexpr int64_t kArenaRegion = 1;

/// One knpu.conv2d or knpu.avgpool: rows [srcRow, srcRow + inRows) of the
/// layer's input in, rows [dstRow, dstRow + outRows) of its output out.
struct Tile {
  Operation *op;
  int64_t srcRow, inRows;
  int64_t dstRow, outRows;
};

/// The tiles writing one tensor: a step of the schedule.
struct Layer {
  Value src, dst; // tensors with reshapes looked through
  Value weight, bias, rq; // convolutions only
  SmallVector<Tile> tiles;
  int64_t constOffset = 0; // in region 0

  bool conv() const { return static_cast<bool>(weight); }
  Location loc() const { return tiles.front().op->getLoc(); }
};

/// A buffer in SRAM, alive from step first to step last.
struct Buffer {
  int64_t size, first, last;
  int choice = -1; // the demotable choice it comes from (see Plan)
  int64_t offset = -1;

  bool alive(const Buffer &o) const {
    return first <= o.last && o.first <= last;
  }
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

/// (H, W, C) of a 1xHxWxC value
std::array<int64_t, 3> hwc(Value v) {
  auto t = cast<ShapedType>(v.getType());
  return {t.getDimSize(1), t.getDimSize(2), t.getDimSize(3)};
}

class SchedulePass : public knpu::impl::KnpuScheduleBase<SchedulePass> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  LogicalResult collect(func::FuncOp fn);
  Value root(Value v) const {
    auto it = alias.find(v);
    return it == alias.end() ? v : it->second;
  }
  bool doubleBuffered(const Layer &l) const {
    return overlap && l.tiles.size() > 1;
  }
  void makeBuffers();
  int placeBuffers();
  LogicalResult planSram();

  SmallVector<Value> inputs;              // model inputs
  llvm::DenseMap<Value, Value> alias;     // tensor.reshape result -> tensor
  std::vector<Layer> layers;              // the steps, in program order
  SmallVector<CpuOp> cpus;
  SmallVector<Value> outputs;             // returned values

  // Choices the SRAM layout may demote, per step, and their levels:
  // ahead[j] for loading layer j's constants during step aheadFrom[j]
  // (2: ahead with the margin, 1: ahead, 0: at layer j); keep[j] for
  // keeping layer j's output in SRAM until step lastUse[j] (1: kept, 0:
  // through the arena). -1: no such choice.
  SmallVector<int> ahead, aheadFrom, keep, lastUse;

  // The buffers of the current plan; per step its constants (-1: none),
  // input and output buffers (empty: the tensor stays in SRAM)
  std::vector<Buffer> buffers;
  SmallVector<int> constBuf;
  SmallVector<SmallVector<int, 2>> inBufs, outBufs;
  llvm::DenseMap<Value, int> keptBuf; // tensor kept in SRAM -> buffer
};

LogicalResult SchedulePass::collect(func::FuncOp fn) {
  llvm::MapVector<Value, Layer> byDst;
  for (Value arg : fn.getArguments())
    inputs.push_back(arg);
  auto isTensor = [&](Value v) {
    return llvm::is_contained(inputs, v) || byDst.count(v) ||
           llvm::any_of(cpus, [&](CpuOp c) { return c.getOutput() == v; });
  };

  for (Operation &op : fn.getBody().front()) {
    if (isa<arith::ConstantOp, tensor::EmptyOp, tensor::ExtractSliceOp,
            tensor::InsertSliceOp, func::ReturnOp>(op))
      continue;
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(op)) {
      alias[reshape.getResult()] = root(reshape.getSource());
      continue;
    }
    if (auto cpu = dyn_cast<CpuOp>(op)) {
      if (!isTensor(root(cpu.getInput())))
        return cpu.emitOpError("input must be the model input or a whole "
                               "layer output");
      cpus.push_back(cpu);
      continue;
    }
    if (!isa<Conv2DOp, AvgPool2DOp>(op))
      return op.emitOpError("cannot be scheduled on the NPU");
    if (!cpus.empty())
      return op.emitOpError("cannot run on the NPU after a cpu layer");

    Value input = op.getOperand(0), output = op.getResult(0);
    Tile t{&op, 0, cast<ShapedType>(input.getType()).getDimSize(1), 0,
           cast<ShapedType>(output.getType()).getDimSize(1)};
    Value src = input;
    if (auto slice = input.getDefiningOp<tensor::ExtractSliceOp>()) {
      auto rows = rowSlice(slice, slice.getSourceType());
      if (!rows)
        return slice.emitOpError("must take whole rows of its source");
      src = slice.getSource();
      t.srcRow = rows->first;
    }
    src = root(src);
    if (!isTensor(src))
      return op.emitOpError("input must be the model input or a whole "
                            "layer output");

    // The written tensor: the result itself, or the end of the
    // tensor.insert_slice chain the result goes into
    Value dst = output;
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

    Value weight, bias, rq;
    if (auto conv = dyn_cast<Conv2DOp>(op)) {
      weight = conv.getWeight();
      bias = conv.getBias();
      rq = conv.getRq();
    }
    Layer &layer = byDst[dst];
    if (layer.tiles.empty()) {
      layer.src = src;
      layer.dst = dst;
      layer.weight = weight;
      layer.bias = bias;
      layer.rq = rq;
    } else if (layer.src != src || layer.weight != weight ||
               layer.bias != bias || layer.rq != rq ||
               layer.tiles.front().op->getName() != op.getName()) {
      return op.emitOpError("tiles of one output must share their input "
                            "and constants");
    }
    layer.tiles.push_back(t);
  }

  // Every layer output must be covered exactly once, tile after tile
  for (auto &[dst, layer] : byDst) {
    llvm::sort(layer.tiles, [](const Tile &a, const Tile &b) {
      return a.dstRow < b.dstRow;
    });
    int64_t next = 0;
    for (Tile &t : layer.tiles) {
      if (t.dstRow != next)
        return t.op->emitOpError("tiles must cover the output rows once");
      next += t.outRows;
    }
    if (next != cast<ShapedType>(dst.getType()).getDimSize(1))
      return layer.tiles.front().op->emitOpError(
          "tiles must cover the output rows once");
    layers.push_back(layer);
  }

  auto ret = cast<func::ReturnOp>(fn.getBody().front().getTerminator());
  for (Value v : ret.getOperands()) {
    if (!isTensor(root(v)) || llvm::is_contained(inputs, root(v)))
      return ret.emitOpError("must return layer outputs");
    outputs.push_back(v);
  }
  if (layers.empty())
    return fn.emitOpError("has no layer to schedule");
  return success();
}

/// The buffers the current levels of the choices give, in the order they
/// are made: per step its constants, inputs and outputs.
void SchedulePass::makeBuffers() {
  buffers.clear();
  keptBuf.clear();
  size_t n = layers.size();
  constBuf.assign(n, -1);
  inBufs.assign(n, {});
  outBufs.assign(n, {});
  auto add = [&](int64_t size, int64_t first, int64_t last, int choice) {
    buffers.push_back({alignTo16(size), first, last, choice});
    return static_cast<int>(buffers.size() - 1);
  };
  for (size_t i = 0; i < n; ++i) {
    const Layer &l = layers[i];
    auto [ih, iw, ic] = hwc(l.tiles.front().op->getOperand(0));
    auto [oh, ow, oc] = hwc(l.dst);
    if (l.conv()) {
      auto w = cast<ShapedType>(l.weight.getType());
      int64_t bytes = convConstBytes(oc, w.getDimSize(1), w.getDimSize(2), ic);
      int64_t first = i;
      if (ahead[i] > 0)
        first = std::max<int64_t>(aheadFrom[i] - (ahead[i] == 2 ? 2 : 0), 0);
      constBuf[i] = add(bytes, first, i, ahead[i] > 0 ? 2 * i : -1);
    }
    int copies = doubleBuffered(l) ? 2 : 1;
    int64_t maxIn = 0, maxOut = 0;
    for (const Tile &t : l.tiles) {
      maxIn = std::max(maxIn, t.inRows * iw * ic);
      maxOut = std::max(maxOut, t.outRows * ow * oc);
    }
    if (!keptBuf.count(l.src))
      for (int c = 0; c < copies; ++c)
        inBufs[i].push_back(add(maxIn, i, i, -1));
    if (keep[i] > 0) {
      keptBuf[l.dst] = add(bytesOf(l.dst), i, lastUse[i], 2 * i + 1);
    } else {
      for (int c = 0; c < copies; ++c)
        outBufs[i].push_back(add(maxOut, i, i, -1));
    }
  }
}

/// Places the buffers; returns the index of one that does not fit, or -1.
int SchedulePass::placeBuffers() {
  SmallVector<int> order(buffers.size());
  std::iota(order.begin(), order.end(), 0);
  auto spans = [&](int k) { return buffers[k].last > buffers[k].first; };
  llvm::stable_sort(order, [&](int a, int b) {
    if (spans(a) != spans(b))
      return spans(a);
    return spans(a) && buffers[a].size > buffers[b].size;
  });
  SmallVector<int> placed;
  for (int k : order) {
    Buffer &buf = buffers[k];
    SmallVector<std::pair<int64_t, int64_t>> busy;
    for (int j : placed)
      if (buffers[j].alive(buf))
        busy.push_back({buffers[j].offset,
                        buffers[j].offset + buffers[j].size});
    llvm::sort(busy);
    int64_t offset = 0;
    for (auto [begin, end] : busy) {
      if (offset + buf.size <= begin)
        break;
      offset = std::max(offset, end);
    }
    if (offset + buf.size > sramSize)
      return k;
    buf.offset = offset;
    placed.push_back(k);
  }
  return -1;
}

LogicalResult SchedulePass::planSram() {
  size_t n = layers.size();
  ahead.assign(n, -1);
  aheadFrom.assign(n, -1);
  keep.assign(n, -1);
  lastUse.assign(n, -1);
  if (overlap) {
    // Constants: during the previous layer with constants
    int previous = -1;
    for (size_t i = 0; i < n; ++i) {
      if (!layers[i].conv())
        continue;
      if (previous >= 0) {
        ahead[i] = 2;
        aheadFrom[i] = previous;
      }
      previous = i;
    }
    // Outputs that only whole layers read, and are not model outputs
    for (size_t i = 0; i < n; ++i) {
      const Layer &l = layers[i];
      bool readElsewhere =
          l.tiles.size() > 1 ||
          llvm::any_of(outputs, [&](Value v) { return root(v) == l.dst; }) ||
          llvm::any_of(cpus, [&](CpuOp c) {
            return root(c.getInput()) == l.dst;
          });
      for (size_t j = i + 1; j < n && !readElsewhere; ++j)
        if (layers[j].src == l.dst) {
          readElsewhere = layers[j].tiles.size() > 1;
          lastUse[i] = j;
        }
      if (!readElsewhere && lastUse[i] >= 0)
        keep[i] = 1;
    }
  }

  for (;;) {
    makeBuffers();
    int failed = placeBuffers();
    if (failed < 0)
      return success();
    // Demote the largest choice alive with the buffer that did not fit
    int best = -1;
    for (auto [k, buf] : llvm::enumerate(buffers))
      if (buf.choice >= 0 && buf.alive(buffers[failed]) &&
          (best < 0 || buf.size > buffers[best].size))
        best = k;
    if (best < 0)
      return layers.front().tiles.front().op->emitOpError()
             << "buffers do not fit in " << static_cast<int64_t>(sramSize)
             << " bytes of SRAM";
    int choice = buffers[best].choice;
    --(choice % 2 ? keep : ahead)[choice / 2];
  }
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

  // Every layer must fit on its own, as knpu-tile made it
  for (const Layer &l : layers) {
    auto [ih, iw, ic] = hwc(l.tiles.front().op->getOperand(0));
    auto [oh, ow, oc] = hwc(l.dst);
    int64_t constBytes = 0;
    if (l.conv()) {
      auto w = cast<ShapedType>(l.weight.getType());
      constBytes = convConstBytes(oc, w.getDimSize(1), w.getDimSize(2), ic);
    }
    int64_t maxIn = 0, maxOut = 0;
    for (const Tile &t : l.tiles) {
      maxIn = std::max(maxIn, t.inRows * iw * ic);
      maxOut = std::max(maxOut, t.outRows * ow * oc);
    }
    int64_t need = layerSramBytes(constBytes, maxIn, maxOut,
                                  doubleBuffered(l));
    if (need > sramSize) {
      l.tiles.front().op->emitOpError()
          << "needs " << need << " bytes of SRAM but "
          << static_cast<int64_t>(sramSize)
          << " are given (run knpu-tile with the same sram-size and "
             "overlap)";
      return signalPassFailure();
    }
  }
  if (failed(planSram()))
    return signalPassFailure();

  OpBuilder b(fn);

  // Arena: model inputs, then in program order the layer outputs that do
  // not stay in SRAM and the outputs of cpu layers
  llvm::DenseMap<Value, int64_t> arenaOffset;
  int64_t arenaSize = 0;
  auto place = [&](Value v) {
    arenaSize = alignTo16(arenaSize);
    arenaOffset[v] = arenaSize;
    arenaSize += bytesOf(v);
  };
  for (Value v : inputs)
    place(v);
  for (const Layer &l : layers)
    if (!keptBuf.count(l.dst))
      place(l.dst);
  for (CpuOp c : cpus)
    place(c.getOutput());

  // Region 0: per layer weights, bias and requantization table, each
  // 16-byte aligned (convConstBytes)
  std::vector<int8_t> constData;
  for (Layer &layer : layers) {
    if (!layer.conv())
      continue;
    bool ok = true;
    std::vector<int8_t> parts[] = {denseBytes(layer.weight, &ok),
                                   denseBytes(layer.bias, &ok),
                                   denseBytes(layer.rq, &ok)};
    if (!ok) {
      layer.tiles.front().op->emitOpError("constants must be dense");
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
  auto loadConstants = [&](const Layer &l, int step) {
    auto w = cast<ShapedType>(l.weight.getType());
    int64_t bytes = convConstBytes(w.getDimSize(0), w.getDimSize(1),
                                   w.getDimSize(2), w.getDimSize(3));
    dma(l.loc(), "load", kConstRegion, l.constOffset,
        buffers[constBuf[step]].offset, bytes);
  };

  for (auto [i, layer] : llvm::enumerate(layers)) {
    auto [ih, iw, ic] = hwc(layer.tiles.front().op->getOperand(0));
    auto [oh, ow, oc] = hwc(layer.dst);
    bool db = doubleBuffered(layer);
    auto keptIn = keptBuf.find(layer.src), keptOut = keptBuf.find(layer.dst);
    bool inKept = keptIn != keptBuf.end(), outKept = keptOut != keptBuf.end();
    int64_t srcBase = inKept ? 0 : arenaOffset[layer.src];
    int64_t dstBase = outKept ? 0 : arenaOffset[layer.dst];

    if (layer.conv() && ahead[i] <= 0)
      loadConstants(layer, i);
    auto load = [&](size_t k) {
      const Tile &t = layer.tiles[k];
      dma(t.op->getLoc(), "load", kArenaRegion,
          srcBase + t.srcRow * iw * ic,
          buffers[inBufs[i][k % inBufs[i].size()]].offset,
          t.inRows * iw * ic);
    };
    if (!inKept && db)
      load(0);
    for (size_t k = 0; k < layer.tiles.size(); ++k) {
      const Tile &t = layer.tiles[k];
      if (!inKept) {
        if (!db)
          load(k);
        else if (k + 1 < layer.tiles.size())
          load(k + 1);
      }
      // In front of the layer's first operation: the constants of the next
      // layer with constants, loaded ahead
      if (k == 0)
        for (size_t j = i + 1; j < layers.size(); ++j)
          if (ahead[j] > 0 && aheadFrom[j] == static_cast<int>(i))
            loadConstants(layers[j], j);

      int64_t ifm = inKept ? buffers[keptIn->second].offset +
                                 t.srcRow * iw * ic
                           : buffers[inBufs[i][k % inBufs[i].size()]].offset;
      int64_t ofm = outKept ? buffers[keptOut->second].offset +
                                  t.dstRow * ow * oc
                            : buffers[outBufs[i][k % outBufs[i].size()]].offset;
      if (auto conv = dyn_cast<Conv2DOp>(t.op)) {
        auto w = conv.getWeight().getType();
        int64_t kh = w.getDimSize(1), kw = w.getDimSize(2);
        int64_t base = buffers[constBuf[i]].offset;
        int64_t biasAt = base + alignTo16(oc * kh * kw * ic);
        ConvOp::create(
            b, conv.getLoc(), b.getI64IntegerAttr(ifm),
            b.getDenseI64ArrayAttr({t.inRows, iw, ic}),
            conv.getInputZpAttr(), b.getI64IntegerAttr(ofm),
            b.getDenseI64ArrayAttr({t.outRows, ow, oc}),
            conv.getOutputZpAttr(), b.getI64IntegerAttr(base),
            b.getI64IntegerAttr(biasAt),
            b.getI64IntegerAttr(biasAt + alignTo16(4 * oc)),
            b.getDenseI64ArrayAttr({kh, kw}), conv.getStrideAttr(),
            conv.getPadAttr(), conv.getClampAttr(), conv.getRoundAttr());
      } else {
        auto pool = cast<AvgPool2DOp>(t.op);
        PoolOp::create(b, pool.getLoc(), b.getI64IntegerAttr(ifm),
                       b.getDenseI64ArrayAttr({t.inRows, iw, ic}),
                       b.getI64IntegerAttr(ofm),
                       b.getDenseI64ArrayAttr({t.outRows, ow, oc}),
                       pool.getKernelAttr(), pool.getStrideAttr(),
                       pool.getPadAttr(), pool.getClampAttr());
      }
      if (!outKept)
        dma(t.op->getLoc(), "store", kArenaRegion,
            dstBase + t.dstRow * ow * oc, ofm, t.outRows * ow * oc);
    }
  }
  StopOp::create(b, fn.getLoc());
  func::ReturnOp::create(b, fn.getLoc());

  // Region table, model inputs and outputs and the CPU steps, ahead of the
  // program
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
                 b.getI64IntegerAttr(arenaOffset[root(v)]),
                 b.getDenseI64ArrayAttr(cast<ShapedType>(v.getType()).getShape()),
                 b.getF32FloatAttr(scale.getValueAsDouble()), zp);
    return success();
  };
  for (auto [i, arg] : llvm::enumerate(fn.getArguments()))
    if (failed(io("input", arg, fn.getArgAttrDict(i))))
      return signalPassFailure();
  for (auto [i, v] : llvm::enumerate(outputs))
    if (failed(io("output", v, fn.getResultAttrDict(i))))
      return signalPassFailure();
  for (CpuOp c : cpus) {
    auto step = [&](StringRef kind, FlatSymbolRefAttr symbol, Value v) {
      IntegerAttr offset, size;
      if (v) {
        offset = b.getI64IntegerAttr(arenaOffset[root(v)]);
        size = b.getI64IntegerAttr(bytesOf(v));
      }
      CpuStepOp::create(b, c.getLoc(), b.getStringAttr(kind), symbol, offset,
                        size);
    };
    step("copy_in", c.getInArrayAttr(), c.getInput());
    step("call", c.getCalleeAttr(), Value());
    step("copy_out", c.getOutArrayAttr(), c.getOutput());
  }

  module->setAttr(kSramSizeAttr, b.getI64IntegerAttr(sramSize));
  fn.erase();
}

} // namespace
