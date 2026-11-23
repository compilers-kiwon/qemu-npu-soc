# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Command numbers and queue depths of the NPU (hw/npu-core, MIT)
include_directories(${KNPU_REPO_DIR}/hw/npu-core)

add_mlir_library(KnpuTransforms
  Import.cpp
  Tile.cpp
  Schedule.cpp
  InsertWaits.cpp
  Estimate.cpp

  ADDITIONAL_HEADER_DIRS
  ${KNPU_SOURCE_DIR}/include/knpu/Transforms

  DEPENDS
  KnpuPassesIncGen

  LINK_LIBS PUBLIC
  KnpuDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRTensorDialect
)
