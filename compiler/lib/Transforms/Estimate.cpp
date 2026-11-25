// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-estimate: a cycle estimate of the command stream (docs/design.md
// 9.1). The device does not model time (docs/design.md D1), so this is a
// model for comparing schedules: the command processor issues commands in
// order and stops only when the queue of a DMA or operation is full or at
// a wait command (docs/npu-spec.md 9); each queue runs its jobs one after
// another, for a time given by assumed rates.

#include "knpu/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <vector>

#include "knpu_cmd.h"

namespace mlir::knpu {
#define GEN_PASS_DEF_KNPUESTIMATE
#include "knpu/Transforms/Passes.h.inc"
} // namespace mlir::knpu

using namespace mlir;
using namespace mlir::knpu;

namespace {

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

/// The end cycles of a queue's jobs, in the order they were issued. Jobs of
/// one queue end in that order.
struct Queue {
  std::vector<int64_t> ends;
  size_t depth;

  /// The cycle from which at most @pending jobs are unfinished.
  int64_t until(int64_t pending) const {
    if (static_cast<int64_t>(ends.size()) <= pending)
      return 0;
    return ends[ends.size() - pending - 1];
  }
};

class EstimatePass : public knpu::impl::KnpuEstimateBase<EstimatePass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fn = module.lookupSymbol<func::FuncOp>(kModelFunc);
    if (!fn || fn.getNumArguments() != 0) {
      module.emitError("no knpu_model command function "
                       "(run knpu-schedule first)");
      return signalPassFailure();
    }
    if (macsPerCycle < 1 || dmaBytesPerCycle < 1 || poolElementsPerCycle < 1 ||
        addElementsPerCycle < 1) {
      module.emitError("knpu-estimate rates must be positive");
      return signalPassFailure();
    }

    Builder b(&getContext());
    Queue dma{{}, KNPU_DMA_DEPTH}, kernel{{}, KNPU_KERNEL_DEPTH};
    int64_t now = 0; // when the command processor issues the next command
    for (Operation &op : fn.getBody().front()) {
      if (auto w = dyn_cast<DmaWaitOp>(op)) {
        now = std::max(now, dma.until(w.getPending()));
        continue;
      }
      if (auto w = dyn_cast<KernelWaitOp>(op)) {
        now = std::max(now, kernel.until(w.getPending()));
        continue;
      }
      if (isa<StopOp>(op)) {
        now = std::max({now, dma.until(0), kernel.until(0)});
        continue;
      }

      Queue *queue;
      int64_t length;
      if (auto d = dyn_cast<DmaOp>(op)) {
        queue = &dma;
        length = ceilDiv(d.getLength() * d.getCount(), dmaBytesPerCycle);
      } else if (auto c = dyn_cast<ConvOp>(op)) {
        // Nominal MACs, as PERF_MACS counts them (docs/npu-spec.md 11)
        ArrayRef<int64_t> in = c.getIfmShape(), out = c.getOfmShape(),
                          k = c.getKernel();
        queue = &kernel;
        length = ceilDiv(out[0] * out[1] * out[2] * k[0] * k[1] * in[2],
                         macsPerCycle);
      } else if (auto p = dyn_cast<PoolOp>(op)) {
        ArrayRef<int64_t> out = p.getOfmShape(), k = p.getKernel();
        queue = &kernel;
        length = ceilDiv(out[0] * out[1] * out[2] * k[0] * k[1],
                         poolElementsPerCycle);
      } else if (auto e = dyn_cast<ElementwiseOp>(op)) {
        ArrayRef<int64_t> s = e.getShape();
        queue = &kernel;
        length = ceilDiv(s[0] * s[1] * s[2], addElementsPerCycle);
      } else {
        continue;
      }
      // A full queue: the processor waits for its oldest unfinished job
      now = std::max(now, queue->until(queue->depth - 1));
      int64_t start = std::max(now, queue->until(0));
      queue->ends.push_back(start + length);
      op.setAttr(kCyclesAttr, b.getDenseI64ArrayAttr({start, start + length}));
    }
    fn->setAttr(kTotalCyclesAttr, b.getI64IntegerAttr(now));
  }
};

} // namespace
