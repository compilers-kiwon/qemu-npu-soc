// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KNPU_TARGET_MODELFILE_H
#define KNPU_TARGET_MODELFILE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::knpu {

/// Writes a scheduled module (command operations, knpu.region, knpu.io and
/// the knpu.sram_size attribute) as model.knpu v0 (docs/model-format.md).
LogicalResult writeModelFile(ModuleOp module, llvm::raw_ostream &os);

/// Registers the "knpu-to-model" translation for knpu-translate.
void registerToModelFileTranslation();

} // namespace mlir::knpu

#endif // KNPU_TARGET_MODELFILE_H
