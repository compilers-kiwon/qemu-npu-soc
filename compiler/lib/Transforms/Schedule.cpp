// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-schedule: places constants in region 0 and tensors in the arena
// (region 1), lays out SRAM by the lifetimes of its buffers and lowers the
// graph to command operations (docs/design.md 9.4).
//
// The NPU layers run in program order as steps. An add reads two tensors,
// each from SRAM or loaded like the input of a convolution, and writes a
// third: the NPU could add in place (docs/npu-spec.md 6.4), but the output
// always gets a buffer of its own. With overlap (the default):
// - An activation that a whole (one-tile) layer writes and whole layers
//   read stays in SRAM from that layer to the last layer that reads it,
//   also when several layers read it (a residual connection). If a later
//   layer that reads it is tiled, or SRAM is short, it stays until the
//   first layer that reads it, which reads it there while it is stored to
//   the arena (both only read that SRAM, docs/npu-spec.md 9), and the
//   later layers load it. Model inputs and outputs, tensors of cpu layers
//   and the outputs of tiled layers go through the arena.
// - Constants are double-buffered: the constants of a layer are loaded
//   while the previous layer with constants runs, in front of that layer's
//   first operation and after its own loads.
// - In a tiled layer the next tile's input is loaded in front of the
//   current tile's operation; inputs and outputs have two buffers each.
// - The output-channel tiles of a convolution that cover the same output
//   rows (a row band) read the input of those rows once. Their constants,
//   the channel range of the weights, bias and requantization table, have
//   two buffers as well: the next tile's are loaded in front of the current
//   tile's operation. Their outputs go to the arena with 2-D DMAs, a row of
//   the tile's channels for each output pixel (docs/npu-spec.md 6.3).
// Without overlap (the serial schedule to compare with) every layer loads
// its constants (an output-channel tile its own) and every row band its
// input, and stores its output, one buffer each.
//
// SRAM layout: every buffer lives from a first to a last step. Buffers
// that live over several steps are placed first, largest first (as the
// greedy memory planner of TFLite Micro does), then the buffers of single
// steps in the order they were made (constants, inputs, outputs), each at
// the lowest 16-byte aligned offset that no buffer alive at the same time
// uses; if that does not fit, the buffers of single steps go largest first
// as well. Constants loaded ahead keep apart, where room allows, from the
// buffers of the two layers before the one they are loaded in, which may
// still be running (operation queue depth 2), so that the load need not
// wait for them. When the layout does not fit, a choice alive with the
// buffer that failed is demoted one level (constants: ahead without the
// margin, then not ahead; an activation: kept until the last layer that
// reads it, then until the first, then through the arena) and the layout
// is made again. Of the choices whose demotion frees SRAM during the steps
// of that buffer, the one demoted is the one that then moves the fewest
// bytes without overlap; if there is none, the largest. Once the layout
// fits, the demotions it turns out not to need are undone, the costliest
// first. Wait commands are left to knpu-insert-waits.

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

/// One knpu.conv2d, knpu.avgpool or knpu.add: rows [srcRow, srcRow + inRows)
/// of each of the layer's inputs in, rows [dstRow, dstRow + outRows) and
/// channels [dstChan, dstChan + chans) of its output out. A convolution
/// tile uses the constants of its output channels.
struct Tile {
  Operation *op;
  int64_t srcRow, inRows;
  int64_t dstRow, outRows;
  int64_t dstChan, chans;
};

/// The tiles writing one tensor: a step of the schedule.
struct Layer {
  // Inputs (two for an add) and output: tensors with reshapes looked through
  SmallVector<Value, 2> srcs;
  Value dst;
  Value weight, bias, rq; // convolutions only
  SmallVector<Tile> tiles; // by output row, then output channel
  // Row bands: tiles of the same output rows, which read the same input
  // rows; per tile its band, per band its first tile
  SmallVector<int> band;
  SmallVector<size_t> bandStart;
  bool channels = false; // output-channel tiles
  int64_t constOffset = 0; // in region 0

  bool conv() const { return static_cast<bool>(weight); }
  bool reads(Value v) const { return llvm::is_contained(srcs, v); }
  /// Input j is input 0 again (an add of a tensor to itself): it has no
  /// buffer or load of its own
  bool repeats(size_t j) const { return j > 0 && srcs[j] == srcs[0]; }
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

/// Static slice [0, row, 0, chan] [1, rows, W, chans] [1, 1, 1, 1] of a
/// 1xHxWxC tensor: whole rows, or a channel range of them; returns (row,
/// rows, chan, chans).
template <typename SliceOp>
std::optional<std::array<int64_t, 4>> boxSlice(SliceOp op, ShapedType full) {
  ArrayRef<int64_t> off = op.getStaticOffsets(), size = op.getStaticSizes(),
                    stride = op.getStaticStrides();
  if (off.size() != 4 || llvm::any_of(stride, [](int64_t s) { return s != 1; }))
    return std::nullopt;
  if (llvm::any_of(off, ShapedType::isDynamic) ||
      llvm::any_of(size, ShapedType::isDynamic) || off[0] != 0 ||
      off[2] != 0 || size[0] != 1 || size[2] != full.getDimSize(2))
    return std::nullopt;
  return std::array<int64_t, 4>{off[1], size[1], off[3], size[3]};
}

/// The output channels [chan, chan + chans) of a convolution constant
/// (OHWI weights, bias or requantization table) that a tile uses: a
/// tensor.extract_slice of the layer's constant along its first dimension,
/// or the constant itself. Returns (the layer's constant, chan, chans).
std::optional<std::tuple<Value, int64_t, int64_t>> chanRange(Value v) {
  auto slice = v.getDefiningOp<tensor::ExtractSliceOp>();
  if (!slice)
    return std::make_tuple(v, int64_t(0),
                           cast<ShapedType>(v.getType()).getDimSize(0));
  ArrayRef<int64_t> off = slice.getStaticOffsets(),
                    size = slice.getStaticSizes(),
                    stride = slice.getStaticStrides();
  ShapedType full = slice.getSourceType();
  if (llvm::any_of(stride, [](int64_t s) { return s != 1; }) ||
      llvm::any_of(off, ShapedType::isDynamic) ||
      llvm::any_of(size, ShapedType::isDynamic))
    return std::nullopt;
  for (int64_t d = 1; d < full.getRank(); ++d)
    if (off[d] != 0 || size[d] != full.getDimSize(d))
      return std::nullopt;
  return std::make_tuple(slice.getSource(), off[0], size[0]);
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
  /// The sizes of a layer's buffers and how many of each its tiles use at
  /// once: the constants of a tile (all of the layer's or those of an
  /// output-channel range), each input tile and the output tile
  struct Sizes {
    int64_t constBytes = 0, inBytes = 0, outBytes = 0;
    int constCopies = 0, inCopies = 1, outCopies = 1;
  };
  Sizes sizesOf(const Layer &l) const;
  /// Whether tensor v is in SRAM at step i, kept there by the layer that
  /// writes it
  bool inSram(Value v, int64_t i) const {
    auto it = keptBuf.find(v);
    return it != keptBuf.end() && buffers[it->second].last >= i;
  }
  void makeBuffers();
  int placeBuffers();
  std::pair<int64_t, bool> demotion(int choice, int64_t first,
                                    int64_t last) const;
  LogicalResult planSram();

  SmallVector<Value> inputs;              // model inputs
  llvm::DenseMap<Value, Value> alias;     // tensor.reshape result -> tensor
  std::vector<Layer> layers;              // the steps, in program order
  SmallVector<CpuOp> cpus;
  SmallVector<Value> outputs;             // returned values

  // Choices the SRAM layout may demote, per step, and their levels:
  // ahead[j] for loading layer j's constants during step aheadFrom[j]
  // (2: ahead with the margin, 1: ahead, 0: at layer j); keep[j] for
  // keeping layer j's output in SRAM (2: until step lastUse[j], the last
  // layer that reads it; 1: until step firstUse[j], the first, and stored
  // to the arena for the layers after it; 0: through the arena). -1: no
  // such choice.
  SmallVector<int> ahead, aheadFrom, keep, firstUse, lastUse;

  // The buffers of the current plan; per step its constant buffers (none,
  // one, or two for output-channel tiles), the buffers of each input and the
  // output buffers (empty: the tensor stays in SRAM)
  std::vector<Buffer> buffers;
  SmallVector<SmallVector<int, 2>> constBufs;
  SmallVector<std::array<SmallVector<int, 2>, 2>> inBufs;
  SmallVector<SmallVector<int, 2>> outBufs;
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
    if (!isa<Conv2DOp, AvgPool2DOp, AddOp>(op))
      return op.emitOpError("cannot be scheduled on the NPU");
    if (!cpus.empty())
      return op.emitOpError("cannot run on the NPU after a cpu layer");

    Value output = op.getResult(0);
    auto outType = cast<ShapedType>(output.getType());
    Tile t{&op, 0, cast<ShapedType>(op.getOperand(0).getType()).getDimSize(1),
           0, outType.getDimSize(1), 0, outType.getDimSize(3)};
    // The inputs, of which a tile reads the same rows
    SmallVector<Value, 2> srcs;
    for (unsigned j = 0, e = isa<AddOp>(op) ? 2 : 1; j < e; ++j) {
      Value src = op.getOperand(j);
      int64_t row = 0;
      if (auto slice = src.getDefiningOp<tensor::ExtractSliceOp>()) {
        auto box = boxSlice(slice, slice.getSourceType());
        if (!box || (*box)[2] != 0 ||
            (*box)[3] != slice.getSourceType().getDimSize(3))
          return slice.emitOpError("must take whole rows of its source");
        src = slice.getSource();
        row = (*box)[0];
      }
      if (j == 0)
        t.srcRow = row;
      else if (row != t.srcRow)
        return op.emitOpError("inputs must be read from the same row");
      src = root(src);
      if (!isTensor(src))
        return op.emitOpError("input must be the model input or a whole "
                              "layer output");
      srcs.push_back(src);
    }

    // The written tensor: the result itself, or the end of the
    // tensor.insert_slice chain the result goes into
    Value dst = output;
    if (dst.hasOneUse()) {
      if (auto ins = dyn_cast<tensor::InsertSliceOp>(*dst.user_begin());
          ins && ins.getSource() == dst) {
        auto box = boxSlice(ins, ins.getDestType());
        if (!box)
          return ins.emitOpError("must fill whole rows of its destination, "
                                 "or a channel range of them");
        t.dstRow = (*box)[0];
        t.dstChan = (*box)[2];
        dst = ins.getResult();
        while (dst.hasOneUse()) {
          auto next = dyn_cast<tensor::InsertSliceOp>(*dst.user_begin());
          if (!next || next.getDest() != dst)
            break;
          dst = next.getResult();
        }
      }
    }

    // A convolution tile's constants: the layer's, or the same channel
    // range of each as the tile's output
    Value weight, bias, rq;
    if (auto conv = dyn_cast<Conv2DOp>(op)) {
      auto w = chanRange(conv.getWeight()), b = chanRange(conv.getBias()),
           q = chanRange(conv.getRq());
      for (auto &range : {w, b, q})
        if (!range || std::get<1>(*range) != t.dstChan ||
            std::get<2>(*range) != t.chans)
          return op.emitOpError("constants must be those of the output "
                                "channels of the tile");
      weight = std::get<0>(*w);
      bias = std::get<0>(*b);
      rq = std::get<0>(*q);
    }
    Layer &layer = byDst[dst];
    if (layer.tiles.empty()) {
      layer.srcs = srcs;
      layer.dst = dst;
      layer.weight = weight;
      layer.bias = bias;
      layer.rq = rq;
    } else if (layer.srcs != srcs || layer.weight != weight ||
               layer.bias != bias || layer.rq != rq ||
               layer.tiles.front().op->getName() != op.getName()) {
      return op.emitOpError("tiles of one output must share their input "
                            "and constants");
    }
    layer.tiles.push_back(t);
  }

  // Every layer output must be covered exactly once: row band after row
  // band, each by tiles side by side over all channels that read the same
  // input rows
  for (auto &[dst, layer] : byDst) {
    llvm::sort(layer.tiles, [](const Tile &a, const Tile &b) {
      return std::tie(a.dstRow, a.dstChan) < std::tie(b.dstRow, b.dstChan);
    });
    auto [h, w, c] = hwc(dst);
    int64_t row = 0;
    for (size_t k = 0; k < layer.tiles.size();) {
      const Tile &first = layer.tiles[k];
      if (first.dstRow != row)
        return first.op->emitOpError("tiles must cover the output rows once");
      layer.bandStart.push_back(k);
      int64_t chan = 0;
      for (; k < layer.tiles.size() && layer.tiles[k].dstRow == row; ++k) {
        const Tile &t = layer.tiles[k];
        if (t.dstChan != chan || t.outRows != first.outRows)
          return t.op->emitOpError("tiles must cover the output channels "
                                   "of their rows once");
        if (t.srcRow != first.srcRow || t.inRows != first.inRows)
          return t.op->emitOpError("tiles of the same output rows must read "
                                   "the same input rows");
        layer.band.push_back(layer.bandStart.size() - 1);
        layer.channels |= t.chans != c;
        chan += t.chans;
      }
      if (chan != c)
        return first.op->emitOpError("tiles must cover the output channels "
                                     "of their rows once");
      row += first.outRows;
    }
    if (row != h)
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

SchedulePass::Sizes SchedulePass::sizesOf(const Layer &l) const {
  Sizes s;
  auto [ih, iw, ic] = hwc(l.tiles.front().op->getOperand(0));
  auto [oh, ow, oc] = hwc(l.dst);
  int64_t chans = 0;
  for (const Tile &t : l.tiles) {
    s.inBytes = std::max(s.inBytes, t.inRows * iw * ic);
    s.outBytes = std::max(s.outBytes, t.outRows * ow * t.chans);
    chans = std::max(chans, t.chans);
  }
  bool db = doubleBuffered(l);
  if (l.conv()) {
    auto w = cast<ShapedType>(l.weight.getType());
    s.constBytes = convConstBytes(chans, w.getDimSize(1), w.getDimSize(2), ic);
    s.constCopies = l.channels && db ? 2 : 1;
  }
  // Inputs are double-buffered across row bands, outputs across tiles
  s.inCopies = overlap && l.bandStart.size() > 1 ? 2 : 1;
  s.outCopies = db ? 2 : 1;
  return s;
}

/// The buffers the current levels of the choices give, in the order they
/// are made: per step its constants, inputs and outputs.
void SchedulePass::makeBuffers() {
  buffers.clear();
  keptBuf.clear();
  size_t n = layers.size();
  constBufs.assign(n, {});
  inBufs.assign(n, {});
  outBufs.assign(n, {});
  auto add = [&](int64_t size, int64_t first, int64_t last, int choice) {
    buffers.push_back({alignTo16(size), first, last, choice});
    return static_cast<int>(buffers.size() - 1);
  };
  for (size_t i = 0; i < n; ++i) {
    const Layer &l = layers[i];
    Sizes s = sizesOf(l);
    if (l.conv()) {
      int64_t first = i;
      if (ahead[i] > 0)
        first = std::max<int64_t>(aheadFrom[i] - (ahead[i] == 2 ? 2 : 0), 0);
      constBufs[i].push_back(
          add(s.constBytes, first, i, ahead[i] > 0 ? 2 * i : -1));
      // The second buffer of output-channel tiles' constants
      if (s.constCopies > 1)
        constBufs[i].push_back(add(s.constBytes, i, i, -1));
    }
    for (auto [j, src] : llvm::enumerate(l.srcs))
      if (!inSram(src, i) && !l.repeats(j))
        for (int c = 0; c < s.inCopies; ++c)
          inBufs[i][j].push_back(add(s.inBytes, i, i, -1));
    if (keep[i] > 0) {
      keptBuf[l.dst] = add(bytesOf(l.dst), i,
                           keep[i] == 2 ? lastUse[i] : firstUse[i], 2 * i + 1);
    } else {
      for (int c = 0; c < s.outCopies; ++c)
        outBufs[i].push_back(add(s.outBytes, i, i, -1));
    }
  }
}

/// Places the buffers; returns the index of one that does not fit, or -1.
/// Buffers of several steps go first, largest first, then those of single
/// steps in the order they were made; if that does not fit, the single
/// steps' largest first too.
int SchedulePass::placeBuffers() {
  auto attempt = [&](bool bigFirst) {
    SmallVector<int> order(buffers.size());
    std::iota(order.begin(), order.end(), 0);
    auto spans = [&](int k) { return buffers[k].last > buffers[k].first; };
    llvm::stable_sort(order, [&](int a, int b) {
      if (spans(a) != spans(b))
        return spans(a);
      return (bigFirst || spans(a)) && buffers[a].size > buffers[b].size;
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
  };
  int failed = attempt(false);
  if (failed >= 0 && attempt(true) < 0)
    return -1;
  return failed;
}

/// Demoting a choice by one level: the bytes it then moves without overlap,
/// and whether it frees SRAM at one of the steps [first, last].
std::pair<int64_t, bool> SchedulePass::demotion(int choice, int64_t first,
                                                int64_t last) const {
  int64_t i = choice / 2;
  // Whether a step of [lo, hi] within [first, last] is not a step of a
  // layer that still needs the bytes
  auto frees = [&](int64_t lo, int64_t hi, auto needs) {
    for (int64_t s = std::max(lo, first); s <= std::min(hi, last); ++s)
      if (!needs(s))
        return true;
    return false;
  };
  auto none = [](int64_t) { return false; };
  if (choice % 2 == 0) {
    // Layer i's constants: loaded at the step they are loaded during, then
    // at layer i
    int64_t from = aheadFrom[i];
    if (ahead[i] == 2)
      return {0, frees(from - 2, from - 1, none)};
    return {buffers[constBufs[i][0]].size, frees(from, i - 1, none)};
  }
  // Layer i's output: kept until the first layer that reads it, the later
  // ones loading it; or through the arena, stored and loaded again before
  // the first
  Value v = layers[i].dst;
  int64_t bytes = bytesOf(v);
  if (keep[i] == 2 && firstUse[i] < lastUse[i]) {
    int64_t loads = 0;
    for (int64_t j = firstUse[i] + 1; j <= lastUse[i]; ++j)
      loads += layers[j].reads(v);
    return {loads * bytes, frees(firstUse[i] + 1, lastUse[i], [&](int64_t s) {
              return layers[s].reads(v);
            })};
  }
  return {2 * bytes, frees(i + 1, firstUse[i] - 1, none)};
}

LogicalResult SchedulePass::planSram() {
  size_t n = layers.size();
  ahead.assign(n, -1);
  aheadFrom.assign(n, -1);
  keep.assign(n, -1);
  firstUse.assign(n, -1);
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
    // Outputs of whole layers that are not model outputs and that a whole
    // layer reads first: kept until the last layer that reads them if only
    // whole layers do, else until the first
    for (size_t i = 0; i < n; ++i) {
      const Layer &l = layers[i];
      if (l.tiles.size() > 1 ||
          llvm::any_of(outputs, [&](Value v) { return root(v) == l.dst; }) ||
          llvm::any_of(cpus, [&](CpuOp c) {
            return root(c.getInput()) == l.dst;
          }))
        continue;
      bool whole = true;
      for (size_t j = i + 1; j < n; ++j)
        if (layers[j].reads(l.dst)) {
          if (firstUse[i] < 0)
            firstUse[i] = j;
          lastUse[i] = j;
          whole = whole && layers[j].tiles.size() == 1;
        }
      if (firstUse[i] >= 0 && layers[firstUse[i]].tiles.size() == 1)
        keep[i] = whole ? 2 : 1;
    }
  }

  auto demote = [&](int choice) {
    int i = choice / 2;
    if (choice % 2 == 0)
      --ahead[i];
    // With one layer reading it, kept until the first is kept until the last
    else if (--keep[i] == 1 && firstUse[i] == lastUse[i])
      keep[i] = 0;
  };
  auto promote = [&](int choice) {
    int i = choice / 2;
    if (choice % 2 == 0)
      ++ahead[i];
    else if (++keep[i] == 1 && firstUse[i] == lastUse[i])
      keep[i] = 2;
  };
  // The demotions made, with the bytes each then moves without overlap
  SmallVector<std::pair<int, int64_t>> demoted;
  for (;;) {
    makeBuffers();
    int failed = placeBuffers();
    if (failed < 0)
      break;
    // Demote one level, of the choices alive with the buffer that did not
    // fit and whose demotion frees SRAM during its steps, the one that moves
    // the fewest bytes without overlap; if there is none, the largest (its
    // buffers may then be placed elsewhere)
    const Buffer &f = buffers[failed];
    int best = -1, largest = -1;
    int64_t bestCost = 0;
    for (auto [k, buf] : llvm::enumerate(buffers)) {
      if (buf.choice < 0 || !buf.alive(f))
        continue;
      if (largest < 0 || buf.size > buffers[largest].size)
        largest = k;
      auto [cost, frees] = demotion(buf.choice, f.first, f.last);
      if (frees && (best < 0 || cost < bestCost ||
                    (cost == bestCost && buf.size > buffers[best].size))) {
        best = k;
        bestCost = cost;
      }
    }
    if (best < 0)
      best = largest;
    if (best < 0)
      return layers.front().tiles.front().op->emitOpError()
             << "buffers do not fit in " << static_cast<int64_t>(sramSize)
             << " bytes of SRAM";
    int choice = buffers[best].choice;
    demoted.push_back({choice, demotion(choice, 0, -1).first});
    demote(choice);
  }
  // Promote back, the costliest first, the demotions that the layout turns
  // out not to need (a later demotion may have freed the SRAM they did)
  llvm::stable_sort(demoted, [](const auto &a, const auto &b) {
    return a.second > b.second;
  });
  for (const auto &d : demoted) {
    promote(d.first);
    makeBuffers();
    if (placeBuffers() >= 0)
      demote(d.first);
  }
  makeBuffers();
  if (placeBuffers() >= 0)
    return layers.front().tiles.front().op->emitOpError(
        "internal error: the SRAM layout no longer fits");
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

  // Every layer must fit on its own, as knpu-tile made it. The second input
  // of an add has buffers of its own unless it is the first.
  for (const Layer &l : layers) {
    Sizes s = sizesOf(l);
    int64_t inputs = l.srcs.size() > 1 && !l.repeats(1) ? 2 : 1;
    int64_t need = tileSramBytes(s.constBytes, s.constCopies, s.inBytes,
                                 inputs * s.inCopies, s.outBytes, s.outCopies);
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
  // not stay in SRAM until the last layer that reads them and the outputs
  // of cpu layers
  llvm::DenseMap<Value, int64_t> arenaOffset;
  int64_t arenaSize = 0;
  auto place = [&](Value v) {
    arenaSize = alignTo16(arenaSize);
    arenaOffset[v] = arenaSize;
    arenaSize += bytesOf(v);
  };
  for (Value v : inputs)
    place(v);
  for (auto [i, l] : llvm::enumerate(layers))
    if (keep[i] < 2)
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
                 int64_t regionOffset, int64_t sramOffset, int64_t length,
                 int64_t count = 1, int64_t stride = 0) {
    DmaOp::create(b, loc, b.getStringAttr(dir), b.getI64IntegerAttr(region),
                  b.getI64IntegerAttr(regionOffset),
                  b.getI64IntegerAttr(sramOffset),
                  b.getI64IntegerAttr(length), b.getI64IntegerAttr(count),
                  b.getI64IntegerAttr(stride));
  };
  // The constants of layer l's tile t into SRAM at sram: all of the
  // layer's in one DMA, or for an output-channel tile the channel range of
  // the weights, bias and requantization table in one DMA each, laid out
  // as convConstBytes
  auto loadConstants = [&](const Layer &l, const Tile &t, int64_t sram) {
    auto w = cast<ShapedType>(l.weight.getType());
    int64_t oc = w.getDimSize(0), kh = w.getDimSize(1), kw = w.getDimSize(2),
            ic = w.getDimSize(3);
    if (!l.channels) {
      dma(l.loc(), "load", kConstRegion, l.constOffset, sram,
          convConstBytes(oc, kh, kw, ic));
      return;
    }
    int64_t perChan = kh * kw * ic;
    int64_t biasAt = l.constOffset + alignTo16(oc * perChan);
    int64_t rqAt = biasAt + alignTo16(4 * oc);
    int64_t sramBias = sram + alignTo16(t.chans * perChan);
    int64_t sramRq = sramBias + alignTo16(4 * t.chans);
    Location loc = t.op->getLoc();
    dma(loc, "load", kConstRegion, l.constOffset + t.dstChan * perChan, sram,
        t.chans * perChan);
    dma(loc, "load", kConstRegion, biasAt + 4 * t.dstChan, sramBias,
        4 * t.chans);
    dma(loc, "load", kConstRegion, rqAt + 8 * t.dstChan, sramRq, 8 * t.chans);
  };
  // The first tile's constants of layer j, in its first constant buffer
  auto loadFirstConstants = [&](size_t j) {
    loadConstants(layers[j], layers[j].tiles.front(),
                  buffers[constBufs[j][0]].offset);
  };

  for (auto [i, layer] : llvm::enumerate(layers)) {
    auto [ih, iw, ic] = hwc(layer.tiles.front().op->getOperand(0));
    auto [oh, ow, oc] = hwc(layer.dst);
    size_t tiles = layer.tiles.size(), bands = layer.bandStart.size();
    const SmallVector<int, 2> &consts = constBufs[i];
    auto keptOut = keptBuf.find(layer.dst);
    bool outKept = keptOut != keptBuf.end();
    // An output kept until the first layer that reads it goes to the arena
    // as well, for the layers after it
    bool outStored = keep[i] < 2;
    int64_t dstBase = outStored ? arenaOffset[layer.dst] : 0;

    if (layer.conv() && ahead[i] <= 0)
      loadFirstConstants(i);
    // The input DMAs of row band r: one per input that has buffers, that
    // is, that is not in SRAM already
    auto load = [&](size_t r) {
      const Tile &t = layer.tiles[layer.bandStart[r]];
      for (auto [j, src] : llvm::enumerate(layer.srcs))
        if (const auto &bufs = inBufs[i][j]; !bufs.empty())
          dma(t.op->getLoc(), "load", kArenaRegion,
              arenaOffset[src] + t.srcRow * iw * ic,
              buffers[bufs[r % bufs.size()]].offset, t.inRows * iw * ic);
    };
    // Double-buffered inputs of several row bands: the next band's input is
    // loaded in front of the operation of the current band's first tile
    bool dbIn = overlap && bands > 1;
    if (dbIn)
      load(0);
    for (size_t k = 0; k < tiles; ++k) {
      const Tile &t = layer.tiles[k];
      size_t r = layer.band[k];
      if (k == layer.bandStart[r]) {
        if (!dbIn)
          load(r);
        else if (r + 1 < bands)
          load(r + 1);
      }
      // Output-channel tiles: the next tile's constants, or with one buffer
      // this tile's
      if (layer.channels) {
        if (consts.size() > 1 && k + 1 < tiles)
          loadConstants(layer, layer.tiles[k + 1],
                        buffers[consts[(k + 1) % 2]].offset);
        else if (consts.size() == 1 && k > 0)
          loadConstants(layer, t, buffers[consts[0]].offset);
      }
      // In front of the layer's first operation: the constants of the next
      // layer with constants, loaded ahead
      if (k == 0)
        for (size_t j = i + 1; j < layers.size(); ++j)
          if (ahead[j] > 0 && aheadFrom[j] == static_cast<int>(i))
            loadFirstConstants(j);

      // Where tile k finds input j
      auto inAt = [&](size_t j) {
        if (layer.repeats(j))
          j = 0;
        if (inSram(layer.srcs[j], i))
          return buffers[keptBuf.lookup(layer.srcs[j])].offset +
                 t.srcRow * iw * ic;
        const auto &bufs = inBufs[i][j];
        return buffers[bufs[r % bufs.size()]].offset;
      };
      int64_t ifm = inAt(0);
      int64_t ofm = outKept ? buffers[keptOut->second].offset +
                                  t.dstRow * ow * oc
                            : buffers[outBufs[i][k % outBufs[i].size()]].offset;
      if (auto conv = dyn_cast<Conv2DOp>(t.op)) {
        auto w = conv.getWeight().getType();
        int64_t kh = w.getDimSize(1), kw = w.getDimSize(2);
        int64_t base = buffers[consts[k % consts.size()]].offset;
        int64_t biasAt = base + alignTo16(t.chans * kh * kw * ic);
        ConvOp::create(
            b, conv.getLoc(), b.getI64IntegerAttr(ifm),
            b.getDenseI64ArrayAttr({t.inRows, iw, ic}),
            conv.getInputZpAttr(), b.getI64IntegerAttr(ofm),
            b.getDenseI64ArrayAttr({t.outRows, ow, t.chans}),
            conv.getOutputZpAttr(), b.getI64IntegerAttr(base),
            b.getI64IntegerAttr(biasAt),
            b.getI64IntegerAttr(biasAt + alignTo16(4 * t.chans)),
            b.getDenseI64ArrayAttr({kh, kw}), conv.getStrideAttr(),
            conv.getPadAttr(), conv.getClampAttr(), conv.getRoundAttr());
      } else if (auto pool = dyn_cast<AvgPool2DOp>(t.op)) {
        PoolOp::create(b, pool.getLoc(), b.getI64IntegerAttr(ifm),
                       b.getDenseI64ArrayAttr({t.inRows, iw, ic}),
                       b.getI64IntegerAttr(ofm),
                       b.getDenseI64ArrayAttr({t.outRows, ow, oc}),
                       pool.getKernelAttr(), pool.getStrideAttr(),
                       pool.getPadAttr(), pool.getClampAttr());
      } else {
        auto sum = cast<AddOp>(t.op);
        ElementwiseOp::create(
            b, sum.getLoc(), b.getI64IntegerAttr(ifm),
            b.getI64IntegerAttr(inAt(1)), b.getI64IntegerAttr(ofm),
            b.getDenseI64ArrayAttr({t.outRows, ow, oc}), sum.getInputZpAttr(),
            sum.getInput2ZpAttr(), sum.getOutputZpAttr(),
            sum.getInputScaleAttr(), sum.getInput2ScaleAttr(),
            sum.getOutputScaleAttr(), sum.getClampAttr(), sum.getRoundAttr());
      }
      // The output of an output-channel tile goes to its channels of each
      // pixel: a 2-D DMA with a row of the tile's channels per pixel and the
      // channel count as the stride (rows at most 65535 per DMA)
      if (!outStored)
        continue;
      int64_t at = dstBase + t.dstRow * ow * oc;
      if (!layer.channels) {
        dma(t.op->getLoc(), "store", kArenaRegion, at, ofm,
            t.outRows * ow * oc);
        continue;
      }
      for (int64_t p = 0, pixels = t.outRows * ow; p < pixels; p += 0xFFFF)
        dma(t.op->getLoc(), "store", kArenaRegion, at + p * oc + t.dstChan,
            ofm + p * t.chans, t.chans, std::min<int64_t>(0xFFFF, pixels - p),
            oc);
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
