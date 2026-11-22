# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_dialect_library(KnpuDialect
  KnpuDialect.cpp
  KnpuOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${KNPU_SOURCE_DIR}/include/knpu/Dialect/Knpu

  DEPENDS
  MLIRKnpuOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
