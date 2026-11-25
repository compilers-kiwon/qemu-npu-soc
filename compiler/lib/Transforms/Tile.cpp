// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-tile: splits a layer into tiles when its constants, inputs and output
// do not fit in SRAM at once (docs/design.md 9.4). A convolution or average
// pooling is split into output-row tiles (H tiles) and an addition into
// tiles of the same rows of both inputs and the output.

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
  int64_t kh, kw, constBytes = 0;
  if (conv) {
    auto wType = conv.getWeight().getType();
    kh = wType.getDimSize(1);
    kw = wType.getDimSize(2);
    constBytes = convConstBytes(oc, kh, kw, ic);
  } else {
    kh = pool.getKernel()[0];
    kw = pool.getKernel()[1];
  }
  int64_t sh = conv ? conv.getStride()[0] : pool.getStride()[0];
  ArrayRef<int64_t> pad = conv ? conv.getPad() : pool.getPad();

  if (layerSramBytes(constBytes, ih * iw * ic, oh * ow * oc, false) <=
      sramSize)
    return success();

  // The most output rows per tile with double-buffered input and output
  // (one buffer each without overlap). A tile of t rows reads at most
  // (t - 1) * SH + KH input rows.
  int64_t rows = 0;
  for (int64_t t = oh - 1; t >= 1 && !rows; --t) {
    int64_t inRows = std::min(ih, (t - 1) * sh + kh);
    if (layerSramBytes(constBytes, inRows * iw * ic, t * ow * oc, overlap) <=
        sramSize)
      rows = t;
  }
  if (!rows)
    return op->emitOpError()
           << "does not fit in " << static_cast<int64_t>(sramSize)
           << " bytes of SRAM even with one output row per tile "
              "(output-channel tiles are not supported yet)";

  OpBuilder b(op);
  Location loc = op->getLoc();
  Type i8 = outType.getElementType();
  Value acc = tensor::EmptyOp::create(b, loc, outType.getShape(), i8);
  for (int64_t o0 = 0; o0 < oh; o0 += rows) {
    int64_t n = std::min(rows, oh - o0);
    // Input rows the tile's windows cover, before clipping to the input
    int64_t first = o0 * sh - pad[0];
    int64_t last = (o0 + n - 1) * sh - pad[0] + kh - 1;
    int64_t r0 = std::max<int64_t>(first, 0);
    int64_t r1 = std::min(last, ih - 1);
    Value slice = rowSlice(b, loc, input, r0, r1 - r0 + 1);
    auto tileType = RankedTensorType::get({1, n, ow, oc}, i8);
    auto tilePad = b.getDenseI64ArrayAttr({r0 - first, pad[1], last - r1,
                                           pad[3]});
    Value tile;
    if (conv)
      tile = Conv2DOp::create(b, loc, tileType, slice, conv.getWeight(),
                              conv.getBias(), conv.getRq(),
                              conv.getStrideAttr(), tilePad,
                              conv.getInputZpAttr(), conv.getOutputZpAttr(),
                              conv.getClampAttr(), conv.getRoundAttr());
    else
      tile = AvgPool2DOp::create(b, loc, tileType, slice,
                                 pool.getKernelAttr(), pool.getStrideAttr(),
                                 tilePad, pool.getClampAttr());
    acc = tensor::InsertSliceOp::create(b, loc, tile, acc,
                                        indices(b, {0, o0, 0, 0}),
                                        indices(b, {1, n, ow, oc}),
                                        indices(b, {1, 1, 1, 1}));
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
