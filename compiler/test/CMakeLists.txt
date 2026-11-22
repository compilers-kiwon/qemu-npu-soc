# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# V4: lit / FileCheck tests of the dialect, the passes, the model file writer
# and one compile of real lcc output

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(KNPU_TEST_DEPENDS knpu-opt knpu-translate knpu-compile)
add_lit_testsuite(check-knpu "Running the knpu compiler tests (V4)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${KNPU_TEST_DEPENDS}
)

# The same suite under ctest
add_test(NAME v4_lit
         COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR})
