# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_llvm_executable(knpu-compile knpu-compile.cpp)
llvm_update_compile_flags(knpu-compile)
target_link_libraries(knpu-compile PRIVATE
  KnpuDialect
  KnpuTarget
  KnpuTransforms
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRParser
  MLIRPass
)
mlir_check_all_link_libraries(knpu-compile)
