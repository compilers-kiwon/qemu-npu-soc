# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Command numbers come from the shared NPU core (hw/npu-core, MIT), the same
# header the QEMU device and the Linux driver use
include_directories(${KNPU_REPO_DIR}/hw/npu-core)

add_mlir_library(KnpuTarget
  CpuCode.cpp
  ModelFile.cpp

  ADDITIONAL_HEADER_DIRS
  ${KNPU_SOURCE_DIR}/include/knpu/Target

  DEPENDS
  MLIRKnpuOpsIncGen

  LINK_LIBS PUBLIC
  KnpuDialect
  MLIRBytecodeWriter
  MLIRFuncDialect
  MLIRIR
  MLIRTranslateLib
)
