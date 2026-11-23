// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-tile: splits a convolution or average pooling into output-row tiles
// (H tiles) when its constants, input and output do not fit in SRAM at
// once.

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
    SmallVector<Operation *> ops;
    getOperation().walk([&](Operation *op) {
      if (isa<Conv2DOp, AvgPool2DOp>(op))
        ops.push_back(op);
    });
    for (Operation *op : ops)
      if (failed(tile(op)))
        return signalPassFailure();
  }

private:
  LogicalResult tile(Operation *op);
};

LogicalResult TilePass::tile(Operation *op) {
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
    auto sliceType = RankedTensorType::get({1, r1 - r0 + 1, iw, ic}, i8);
    Value slice = tensor::ExtractSliceOp::create(
        b, loc, sliceType, input, indices(b, {0, r0, 0, 0}),
        indices(b, {1, r1 - r0 + 1, iw, ic}), indices(b, {1, 1, 1, 1}));
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

} // namespace
