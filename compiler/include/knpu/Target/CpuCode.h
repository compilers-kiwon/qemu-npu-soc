// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KNPU_TARGET_CPUCODE_H
#define KNPU_TARGET_CPUCODE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <string>
#include <vector>

namespace mlir::knpu {

/// Moves the cpu layers out of a scheduled module into a module of their
/// own, the CPU module (docs/design.md 9.4): the functions and globals the
/// knpu.cpu_step operations name and everything they use, in module order,
/// with the module attributes that are not knpu.* (the target triple and
/// data layout lcc wrote). Returns a null module when there are no CPU
/// steps.
FailureOr<OwningOpRef<ModuleOp>> extractCpuModule(ModuleOp module);

/// The programs that compile the CPU module (docs/design.md 9.4).
struct CpuTools {
  std::string lccOpt, mlirTranslate, clang, ldLld;
};

/// Compiles the CPU module into an AArch64 shared object in a temporary
/// directory: lcc-opt --lcc-to-linalg --lcc-lower --emit-bytecode,
/// mlir-translate --mlir-to-llvmir, clang --target=aarch64-linux-gnu -O2
/// -fPIC -c and ld.lld -shared -Bsymbolic. Returns the bytes of the shared
/// object; errors are reported at the module.
FailureOr<std::vector<char>> compileCpuModule(ModuleOp cpu,
                                              const CpuTools &tools);

/// Adds the shared object to the scheduled module as knpu.cpu_code, in
/// front of the model function.
void addCpuCode(ModuleOp module, ArrayRef<char> code);

} // namespace mlir::knpu

#endif // KNPU_TARGET_CPUCODE_H
