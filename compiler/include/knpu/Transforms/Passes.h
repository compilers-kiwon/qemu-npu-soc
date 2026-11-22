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

/// SRAM bytes a convolution layer needs: its constants followed by one
/// input and one output buffer, or by two of each when the layer has more
/// than one tile (double buffering). knpu-tile chooses tiles with this and
/// knpu-schedule lays out SRAM the same way.
inline int64_t convSramBytes(int64_t constBytes, int64_t inBytes,
                             int64_t outBytes, bool doubleBuffered) {
  int64_t n = doubleBuffered ? 2 : 1;
  return alignTo16(constBytes) + n * alignTo16(inBytes) +
         n * alignTo16(outBytes);
}

} // namespace mlir::knpu

#endif // KNPU_TRANSFORMS_PASSES_H
