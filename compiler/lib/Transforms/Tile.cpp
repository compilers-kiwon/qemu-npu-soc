// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-tile: splits a layer into tiles when its constants, inputs and output
// do not fit in SRAM at once (docs/design.md 9.4). A convolution or average
// pooling is split into output-row tiles (H tiles) and an addition into
// tiles of the same rows of both inputs and the output. A convolution that
// does not fit even with one-row tiles is split into output-channel tiles,
// each with the weights, bias and requantization table of its channels;
// they read the whole input if it fits beside them, and are row tiles as
// well if it does not.

#include "knpu/Transforms/Passes.h"

#include "mlir/IR/Builders.h"

namespace mlir::knpu {
#define GEN_PASS_DEF_KNPUTILE
#include "knpu/Transforms/Passes.h.inc"
} // namespace mlir::knpu

using namespace mlir;
using namespace mlir::knpu;

namespace {

SmallVector<OpFoldResult> indices(OpBuilder &b, ArrayRef<int64_t> values) {
  return llvm::map_to_vector(values, [&](int64_t v) -> OpFoldResult {
    return b.getIndexAttr(v);
  });
}

/// Rows [row, row + rows) of a 1xHxWxC tensor
Value rowSlice(OpBuilder &b, Location loc, Value v, int64_t row,
               int64_t rows) {
  auto t = cast<RankedTensorType>(v.getType());
  int64_t w = t.getDimSize(2), c = t.getDimSize(3);
  return tensor::ExtractSliceOp::create(
      b, loc, RankedTensorType::get({1, rows, w, c}, t.getElementType()), v,
      indices(b, {0, row, 0, 0}), indices(b, {1, rows, w, c}),
      indices(b, {1, 1, 1, 1}));
}

/// Output channels [chan, chan + chans) of a convolution constant, whose
/// first dimension is the output channel (OHWI weights, bias and
/// requantization table)
Value chanSlice(OpBuilder &b, Location loc, Value v, int64_t chan,
                int64_t chans) {
  auto t = cast<RankedTensorType>(v.getType());
  SmallVector<int64_t> offsets(t.getRank(), 0), strides(t.getRank(), 1);
  SmallVector<int64_t> sizes(t.getShape());
  offsets[0] = chan;
  sizes[0] = chans;
  return tensor::ExtractSliceOp::create(
      b, loc, RankedTensorType::get(sizes, t.getElementType()), v,
      indices(b, offsets), indices(b, sizes), indices(b, strides));
}

class TilePass : public knpu::impl::KnpuTileBase<TilePass> {
public:
  using Base::Base;

  void runOnOperation() override {
    SmallVector<Operation *> ops;
    getOperation().walk([&](Operation *op) {
      if (isa<Conv2DOp, AvgPool2DOp, AddOp>(op))
        ops.push_back(op);
    });
    for (Operation *op : ops) {
      auto add = dyn_cast<AddOp>(op);
      if (failed(add ? tileAdd(add) : tileWindow(op)))
        return signalPassFailure();
    }
  }

private:
  LogicalResult tileWindow(Operation *op);
  LogicalResult tileAdd(AddOp op);
};

/// A convolution or average pooling
LogicalResult TilePass::tileWindow(Operation *op) {
  auto conv = dyn_cast<Conv2DOp>(op);
  auto pool = dyn_cast<AvgPool2DOp>(op);
  Value input = op->getOperand(0);
  auto inType = cast<RankedTensorType>(input.getType());
  auto outType = cast<RankedTensorType>(op->getResult(0).getType());
  int64_t ih = inType.getDimSize(1), iw = inType.getDimSize(2),
          ic = inType.getDimSize(3);
  int64_t oh = outType.getDimSize(1), ow = outType.getDimSize(2),
          oc = outType.getDimSize(3);
  int64_t kh, kw;
  if (conv) {
    auto wType = conv.getWeight().getType();
    kh = wType.getDimSize(1);
    kw = wType.getDimSize(2);
  } else {
    kh = pool.getKernel()[0];
    kw = pool.getKernel()[1];
  }
  int64_t sh = conv ? conv.getStride()[0] : pool.getStride()[0];
  DenseI64ArrayAttr padAttr = conv ? conv.getPadAttr() : pool.getPadAttr();
  ArrayRef<int64_t> pad = padAttr.asArrayRef();
  // The constants of an output-channel range (none for a pooling), and the
  // input a tile of t output rows reads: at most (t - 1) * SH + KH rows
  auto constBytes = [&](int64_t chans) {
    return conv ? convConstBytes(chans, kh, kw, ic) : 0;
  };
  auto inBytes = [&](int64_t t) {
    return std::min(ih, (t - 1) * sh + kh) * iw * ic;
  };

  if (layerSramBytes(constBytes(oc), ih * iw * ic, oh * ow * oc, false) <=
      sramSize)
    return success();

  // The most output rows per tile with double-buffered input and output
  // (one buffer each without overlap)
  int64_t rows = 0, chans = oc;
  for (int64_t t = oh - 1; t >= 1 && !rows; --t)
    if (layerSramBytes(constBytes(oc), inBytes(t), t * ow * oc, overlap) <=
        sramSize)
      rows = t;

  // A convolution that does not fit even so: output-channel tiles with the
  // most channels, whose constants are double-buffered too. They read the
  // whole input once if it fits beside them; if not, they are row tiles as
  // well, with the most rows.
  int64_t copies = overlap ? 2 : 1;
  for (int64_t t = oh; conv && !rows && t >= 1; --t)
    for (int64_t n = oc - 1; n >= 1 && !rows; --n)
      if (tileSramBytes(constBytes(n), copies, inBytes(t), t < oh ? copies : 1,
                        t * ow * n, copies) <= sramSize) {
        rows = t;
        chans = n;
      }
  if (!rows)
    return op->emitOpError()
           << "does not fit in " << static_cast<int64_t>(sramSize)
           << " bytes of SRAM even with one output row "
           << (conv ? "and one output channel " : "") << "per tile";

  OpBuilder b(op);
  Location loc = op->getLoc();
  Type i8 = outType.getElementType();
  // The constants of each output-channel range: all of a convolution's when
  // it has only row tiles
  struct Range {
    int64_t chan, chans;
    Value weight, bias, rq;
  };
  SmallVector<Range> ranges;
  for (int64_t c0 = 0; c0 < oc; c0 += chans) {
    int64_t n = std::min(chans, oc - c0);
    if (!conv)
      ranges.push_back({c0, n, {}, {}, {}});
    else if (n == oc)
      ranges.push_back(
          {0, oc, conv.getWeight(), conv.getBias(), conv.getRq()});
    else
      ranges.push_back({c0, n, chanSlice(b, loc, conv.getWeight(), c0, n),
                        chanSlice(b, loc, conv.getBias(), c0, n),
                        chanSlice(b, loc, conv.getRq(), c0, n)});
  }
  Value acc = tensor::EmptyOp::create(b, loc, outType.getShape(), i8);
  for (int64_t o0 = 0; o0 < oh; o0 += rows) {
    int64_t n = std::min(rows, oh - o0);
    // Input rows the tile's windows cover, before clipping to the input;
    // tiles of all output rows read the whole input with the layer's
    // padding
    int64_t first = o0 * sh - pad[0];
    int64_t last = (o0 + n - 1) * sh - pad[0] + kh - 1;
    int64_t r0 = std::max<int64_t>(first, 0);
    int64_t r1 = std::min(last, ih - 1);
    Value slice = input;
    DenseI64ArrayAttr tilePad = padAttr;
    if (n < oh) {
      slice = rowSlice(b, loc, input, r0, r1 - r0 + 1);
      tilePad =
          b.getDenseI64ArrayAttr({r0 - first, pad[1], last - r1, pad[3]});
    }
    for (const Range &r : ranges) {
      auto tileType = RankedTensorType::get({1, n, ow, r.chans}, i8);
      Value tile;
      if (conv)
        tile = Conv2DOp::create(b, loc, tileType, slice, r.weight, r.bias,
                                r.rq, conv.getStrideAttr(), tilePad,
                                conv.getInputZpAttr(), conv.getOutputZpAttr(),
                                conv.getClampAttr(), conv.getRoundAttr());
      else
        tile = AvgPool2DOp::create(b, loc, tileType, slice,
                                   pool.getKernelAttr(), pool.getStrideAttr(),
                                   tilePad, pool.getClampAttr());
      acc = tensor::InsertSliceOp::create(b, loc, tile, acc,
                                          indices(b, {0, o0, 0, r.chan}),
                                          indices(b, {1, n, ow, r.chans}),
                                          indices(b, {1, 1, 1, 1}));
    }
  }
  op->getResult(0).replaceAllUsesWith(acc);
  op->erase();
  return success();
}

/// An addition: tiles of the same rows of both inputs and the output
LogicalResult TilePass::tileAdd(AddOp op) {
  auto type = cast<RankedTensorType>(op.getType());
  int64_t h = type.getDimSize(1), w = type.getDimSize(2),
          c = type.getDimSize(3);
  // Two inputs and the output of t rows, in as many copies
  auto bytes = [&](int64_t t, int64_t copies) {
    return tileSramBytes(0, 0, t * w * c, 2 * copies, t * w * c, copies);
  };
  if (bytes(h, 1) <= sramSize)
    return success();
  int64_t rows = 0;
  for (int64_t t = h - 1; t >= 1 && !rows; --t)
    if (bytes(t, overlap ? 2 : 1) <= sramSize)
      rows = t;
  if (!rows)
    return op.emitOpError()
           << "does not fit in " << static_cast<int64_t>(sramSize)
           << " bytes of SRAM even with one output row per tile";

  OpBuilder b(op);
  Location loc = op.getLoc();
  Value acc =
      tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType());
  for (int64_t r0 = 0; r0 < h; r0 += rows) {
    int64_t n = std::min(rows, h - r0);
    Value in = rowSlice(b, loc, op.getInput(), r0, n);
    // An addition of a tensor to itself reads one slice twice
    Value in2 = op.getInput2() == op.getInput()
                    ? in
                    : rowSlice(b, loc, op.getInput2(), r0, n);
    Value tile = AddOp::create(
        b, loc, in.getType(), in, in2, op.getInputZpAttr(),
        op.getInput2ZpAttr(), op.getOutputZpAttr(), op.getInputScaleAttr(),
        op.getInput2ScaleAttr(), op.getOutputScaleAttr(), op.getClampAttr(),
        op.getRoundAttr());
    acc = tensor::InsertSliceOp::create(b, loc, tile, acc,
                                        indices(b, {0, r0, 0, 0}),
                                        indices(b, {1, n, w, c}),
                                        indices(b, {1, 1, 1, 1}));
  }
  op.getResult().replaceAllUsesWith(acc);
  op.erase();
  return success();
}

} // namespace
