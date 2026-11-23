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

# The default tools of the CPU code (compiler/cmake/deps.cmake); lcc-opt is
# built before knpu-compile
target_compile_definitions(knpu-compile PRIVATE
  KNPU_LCC_OPT="${KNPU_LCC_OPT}"
  KNPU_MLIR_TRANSLATE="${KNPU_MLIR_TRANSLATE}"
  KNPU_CLANG="${KNPU_CLANG}"
  KNPU_LD_LLD="${KNPU_LD_LLD}"
)
add_dependencies(knpu-compile lcc-opt)
