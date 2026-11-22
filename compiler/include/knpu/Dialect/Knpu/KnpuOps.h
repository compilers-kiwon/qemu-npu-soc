// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KNPU_DIALECT_KNPU_KNPUOPS_H
#define KNPU_DIALECT_KNPU_KNPUOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "knpu/Dialect/Knpu/KnpuDialect.h"

#define GET_OP_CLASSES
#include "knpu/Dialect/Knpu/KnpuOps.h.inc"

namespace mlir::knpu {

/// Module attribute holding the SRAM size the program was compiled for.
inline constexpr llvm::StringLiteral kSramSizeAttr = "knpu.sram_size";

/// Name of the model function: tensor graph after knpu-import, command
/// program after knpu-schedule.
inline constexpr llvm::StringLiteral kModelFunc = "knpu_model";

/// Argument and result attributes of the graph-form model function.
inline constexpr llvm::StringLiteral kScaleAttr = "knpu.scale";
inline constexpr llvm::StringLiteral kZeroPointAttr = "knpu.zero_point";

/// Shapes of one windowed NPU operation (docs/npu-spec.md 6.4).
struct WindowShape {
  int64_t ih, iw, ic;
  int64_t oh, ow, oc;
  int64_t kh, kw;
  int64_t sh, sw;
  int64_t pt, pl, pb, pr;
};

/// Checks the field ranges of the SET commands and the window conditions of
/// docs/npu-spec.md 6.4. Returns an empty string or the first violation.
std::string checkWindowShape(const WindowShape &s);

} // namespace mlir::knpu

#endif // KNPU_DIALECT_KNPU_KNPUOPS_H
