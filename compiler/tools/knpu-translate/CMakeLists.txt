# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_llvm_executable(knpu-translate knpu-translate.cpp)
llvm_update_compile_flags(knpu-translate)
target_link_libraries(knpu-translate PRIVATE
  KnpuTarget
  MLIRTranslateLib
)
mlir_check_link_libraries(knpu-translate)
