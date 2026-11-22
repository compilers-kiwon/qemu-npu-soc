// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "knpu/Dialect/Knpu/KnpuDialect.h"
#include "knpu/Dialect/Knpu/KnpuOps.h"

using namespace mlir;
using namespace mlir::knpu;

#include "knpu/Dialect/Knpu/KnpuOpsDialect.cpp.inc"

void KnpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "knpu/Dialect/Knpu/KnpuOps.cpp.inc"
      >();
}
