// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-tile: splits a convolution into output-row tiles (H tiles) when its
// constants, input and output do not fit in SRAM at once.

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

class TilePass : public knpu::impl::KnpuTileBase<TilePass> {
public:
  using Base::Base;

  void runOnOperation() override {
    SmallVector<Conv2DOp> convs;
    getOperation().walk([&](Conv2DOp op) { convs.push_back(op); });
    for (Conv2DOp op : convs)
      if (failed(tile(op)))
        return signalPassFailure();
  }

private:
  LogicalResult tile(Conv2DOp op);
};

LogicalResult TilePass::tile(Conv2DOp op) {
  auto inType = op.getInput().getType();
  auto outType = op.getOutput().getType();
  auto wType = op.getWeight().getType();
  int64_t ih = inType.getDimSize(1), iw = inType.getDimSize(2),
          ic = inType.getDimSize(3);
  int64_t oh = outType.getDimSize(1), ow = outType.getDimSize(2),
          oc = outType.getDimSize(3);
  int64_t kh = wType.getDimSize(1), kw = wType.getDimSize(2);
  int64_t sh = op.getStride()[0];
  ArrayRef<int64_t> pad = op.getPad();

  int64_t constBytes = convConstBytes(oc, kh, kw, ic);
  if (convSramBytes(constBytes, ih * iw * ic, oh * ow * oc, false) <=
      sramSize)
    return success();

  // The most output rows per tile with double-buffered input and output.
  // A tile of t rows reads at most (t - 1) * SH + KH input rows.
  int64_t rows = 0;
  for (int64_t t = oh - 1; t >= 1 && !rows; --t) {
    int64_t inRows = std::min(ih, (t - 1) * sh + kh);
    if (convSramBytes(constBytes, inRows * iw * ic, t * ow * oc, true) <=
        sramSize)
      rows = t;
  }
  if (!rows)
    return op.emitOpError()
           << "does not fit in " << static_cast<int64_t>(sramSize)
           << " bytes of SRAM even with one output row per tile "
              "(output-channel tiles are not supported yet)";

  OpBuilder b(op);
  Location loc = op.getLoc();
  Type i8 = outType.getElementType();
  Value acc = tensor::EmptyOp::create(b, loc, outType.getShape(), i8);
  for (int64_t o0 = 0; o0 < oh; o0 += rows) {
    int64_t n = std::min(rows, oh - o0);
    // Input rows the tile's windows cover, before clipping to the input
    int64_t first = o0 * sh - pad[0];
    int64_t last = (o0 + n - 1) * sh - pad[0] + kh - 1;
    int64_t r0 = std::max<int64_t>(first, 0);
    int64_t r1 = std::min(last, ih - 1);
    auto sliceType = RankedTensorType::get({1, r1 - r0 + 1, iw, ic}, i8);
    Value slice = tensor::ExtractSliceOp::create(
        b, loc, sliceType, op.getInput(), indices(b, {0, r0, 0, 0}),
        indices(b, {1, r1 - r0 + 1, iw, ic}), indices(b, {1, 1, 1, 1}));
    auto tileType = RankedTensorType::get({1, n, ow, oc}, i8);
    Value tile = Conv2DOp::create(
        b, loc, tileType, slice, op.getWeight(), op.getBias(), op.getRq(),
        op.getStrideAttr(),
        b.getDenseI64ArrayAttr({r0 - first, pad[1], last - r1, pad[3]}),
        op.getInputZpAttr(), op.getOutputZpAttr(), op.getClampAttr(),
        op.getRoundAttr());
    acc = tensor::InsertSliceOp::create(b, loc, tile, acc,
                                        indices(b, {0, o0, 0, 0}),
                                        indices(b, {1, n, ow, oc}),
                                        indices(b, {1, 1, 1, 1}));
  }
  op.getOutput().replaceAllUsesWith(acc);
  op.erase();
  return success();
}

} // namespace
