// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-insert-waits: follows the DMA and operation queues the way the NPU
// does (docs/npu-spec.md 9) and inserts a wait command only before a
// command that would conflict with a job still in flight on the other queue.
// The wait leaves as many jobs in flight as the conflict allows, like the
// wait computation of Arm Vela (get_wait_dependency).

#include "knpu/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <deque>

#include "knpu_cmd.h"

namespace mlir::knpu {
#define GEN_PASS_DEF_KNPUINSERTWAITS
#include "knpu/Transforms/Passes.h.inc"
} // namespace mlir::knpu

using namespace mlir;
using namespace mlir::knpu;

namespace {

struct Span {
  int64_t begin, end; // [begin, end) in SRAM
};

/// An asynchronous command with the SRAM it reads and writes.
struct Job {
  SmallVector<Span, 4> reads, writes;
};

bool overlap(const Span &a, const Span &b) {
  return a.begin < b.end && b.begin < a.end;
}

/// Two jobs conflict when their SRAM ranges overlap and one of them writes.
bool conflicts(const Job &a, const Job &b) {
  for (const Span &w : a.writes) {
    for (const Span &s : b.reads)
      if (overlap(w, s))
        return true;
    for (const Span &s : b.writes)
      if (overlap(w, s))
        return true;
  }
  for (const Span &w : b.writes)
    for (const Span &s : a.reads)
      if (overlap(w, s))
        return true;
  return false;
}

Job describe(DmaOp op) {
  Job j;
  int64_t base = op.getSramOffset();
  Span s{base, base + static_cast<int64_t>(op.getLength() * op.getCount())};
  // A load writes SRAM, a store reads it
  (op.getDir() == "load" ? j.writes : j.reads).push_back(s);
  return j;
}

Job describe(ConvOp op) {
  Job j;
  ArrayRef<int64_t> in = op.getIfmShape(), out = op.getOfmShape(),
                    k = op.getKernel();
  int64_t ic = in[2], oc = out[2];
  int64_t ifm = op.getIfm(), ofm = op.getOfm(), w = op.getWeight(),
          bias = op.getBias(), rq = op.getRqTable();
  j.reads = {{ifm, ifm + in[0] * in[1] * ic},
             {w, w + oc * k[0] * k[1] * ic},
             {bias, bias + 4 * oc},
             {rq, rq + 8 * oc}};
  j.writes = {{ofm, ofm + out[0] * out[1] * oc}};
  return j;
}

Job describe(PoolOp op) {
  Job j;
  ArrayRef<int64_t> in = op.getIfmShape(), out = op.getOfmShape();
  int64_t ifm = op.getIfm(), ofm = op.getOfm();
  j.reads = {{ifm, ifm + in[0] * in[1] * in[2]}};
  j.writes = {{ofm, ofm + out[0] * out[1] * out[2]}};
  return j;
}

void drain(std::deque<Job> &queue, int64_t pending) {
  while (static_cast<int64_t>(queue.size()) > pending)
    queue.pop_front();
}

class InsertWaitsPass : public knpu::impl::KnpuInsertWaitsBase<InsertWaitsPass> {
public:
  void runOnOperation() override {
    auto fn = getOperation().lookupSymbol<func::FuncOp>(kModelFunc);
    if (!fn || fn.getNumArguments() != 0) {
      getOperation().emitError("no knpu_model command function "
                               "(run knpu-schedule first)");
      return signalPassFailure();
    }

    std::deque<Job> dmaQueue, kernelQueue;
    for (Operation &op : llvm::make_early_inc_range(fn.getBody().front())) {
      if (auto w = dyn_cast<DmaWaitOp>(op)) {
        drain(dmaQueue, w.getPending());
        continue;
      }
      if (auto w = dyn_cast<KernelWaitOp>(op)) {
        drain(kernelQueue, w.getPending());
        continue;
      }
      if (isa<StopOp>(op)) {
        dmaQueue.clear();
        kernelQueue.clear();
        continue;
      }

      Job job;
      if (auto d = dyn_cast<DmaOp>(op))
        job = describe(d);
      else if (auto c = dyn_cast<ConvOp>(op))
        job = describe(c);
      else if (auto p = dyn_cast<PoolOp>(op))
        job = describe(p);
      else
        continue;
      bool isDma = isa<DmaOp>(op);
      std::deque<Job> &own = isDma ? dmaQueue : kernelQueue;
      std::deque<Job> &other = isDma ? kernelQueue : dmaQueue;
      size_t depth = isDma ? KNPU_DMA_DEPTH : KNPU_KERNEL_DEPTH;

      // A full queue completes its oldest job first (implicit wait)
      if (own.size() >= depth)
        own.pop_front();

      // Wait for the newest conflicting job on the other queue
      int64_t newest = -1;
      for (auto [i, j] : llvm::enumerate(other))
        if (conflicts(job, j))
          newest = i;
      if (newest >= 0) {
        int64_t pending = static_cast<int64_t>(other.size()) - newest - 1;
        OpBuilder b(&op);
        if (isDma)
          KernelWaitOp::create(b, op.getLoc(), b.getI64IntegerAttr(pending));
        else
          DmaWaitOp::create(b, op.getLoc(), b.getI64IntegerAttr(pending));
        drain(other, pending);
      }
      own.push_back(std::move(job));
    }
  }
};

} // namespace
