# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_llvm_executable(knpu-opt knpu-opt.cpp)
llvm_update_compile_flags(knpu-opt)
target_link_libraries(knpu-opt PRIVATE
  KnpuDialect
  KnpuTransforms
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIROptLib
  MLIRTensorDialect
)
mlir_check_all_link_libraries(knpu-opt)
