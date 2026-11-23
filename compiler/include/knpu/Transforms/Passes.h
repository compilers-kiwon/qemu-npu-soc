// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KNPU_TRANSFORMS_PASSES_H
#define KNPU_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

#include "knpu/Dialect/Knpu/KnpuOps.h"

namespace mlir::knpu {

#define GEN_PASS_DECL
#include "knpu/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "knpu/Transforms/Passes.h.inc"

inline int64_t alignTo16(int64_t v) { return (v + 15) & ~int64_t(15); }

/// Bytes of a convolution layer's constants as region 0 and SRAM hold them:
/// weights, bias and requantization table, each starting 16-byte aligned.
inline int64_t convConstBytes(int64_t oc, int64_t kh, int64_t kw,
                              int64_t ic) {
  return alignTo16(oc * kh * kw * ic) + alignTo16(4 * oc) + 8 * oc;
}

/// SRAM bytes a layer needs on its own: its constants (none for a pooling
/// layer) followed by one input and one output buffer, or by two of each
/// when its tiles are double-buffered. knpu-tile chooses tiles with this;
/// knpu-schedule lays out a layer's buffers in the same order.
inline int64_t layerSramBytes(int64_t constBytes, int64_t inBytes,
                              int64_t outBytes, bool doubleBuffered) {
  int64_t n = doubleBuffered ? 2 : 1;
  return alignTo16(constBytes) + n * alignTo16(inBytes) +
         n * alignTo16(outBytes);
}

/// Attributes knpu-estimate adds: (start, end) cycle of a DMA or operation
/// command, and the cycle the whole job ends at on the model function.
inline constexpr llvm::StringLiteral kCyclesAttr = "knpu.cycles";
inline constexpr llvm::StringLiteral kTotalCyclesAttr = "knpu.total_cycles";

} // namespace mlir::knpu

#endif // KNPU_TRANSFORMS_PASSES_H
