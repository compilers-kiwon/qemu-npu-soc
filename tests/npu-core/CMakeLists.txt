# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Host tests of the kn1-npu shared core (docs/design.md 11):
#   V2  tflm_golden_double / tflm_golden_single: TFLite Micro reference
#       convolution and average pooling, built once per rounding mode
#   V3  knpu_run executes command streams on hw/npu-core; the Python tests
#       compare its results with V2 and check the error paths
#
#   cmake -S tests/npu-core -B build/npu-core
#   cmake --build build/npu-core
#   ctest --test-dir build/npu-core --output-on-failure

cmake_minimum_required(VERSION 3.20)
project(knpu_host_tests LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)

set(NPU_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../hw/npu-core)

# Shared core and the host runner
add_library(knpu_core STATIC ${NPU_CORE_DIR}/knpu_core.c)
target_include_directories(knpu_core PUBLIC ${NPU_CORE_DIR})
target_compile_options(knpu_core PRIVATE -Wall -Wextra -Werror)

add_executable(knpu_run knpu_run.c)
target_link_libraries(knpu_run PRIVATE knpu_core)
target_compile_options(knpu_run PRIVATE -Wall -Wextra -Werror)

# Pinned TensorFlow Lite Micro sources, downloaded once and hash-checked
include(${CMAKE_CURRENT_SOURCE_DIR}/tflm-files.cmake)
set(TFLM_DIR ${CMAKE_CURRENT_BINARY_DIR}/tflite-micro)
list(LENGTH TFLM_FILES n)
math(EXPR last "${n} - 1")
foreach(i RANGE 0 ${last} 2)
  math(EXPR j "${i} + 1")
  list(GET TFLM_FILES ${i} path)
  list(GET TFLM_FILES ${j} sha256)
  if(NOT EXISTS ${TFLM_DIR}/${path})
    message(STATUS "Downloading TFLite Micro ${path}")
  endif()
  file(DOWNLOAD ${TFLM_URL}/${TFLM_COMMIT}/${path} ${TFLM_DIR}/${path}
       EXPECTED_HASH SHA256=${sha256} STATUS st TLS_VERIFY ON)
  list(GET st 0 code)
  if(NOT code EQUAL 0)
    message(FATAL_ERROR "Could not download ${path}: ${st}")
  endif()
endforeach()

foreach(mode double single)
  if(mode STREQUAL single)
    set(flag 1)
  else()
    set(flag 0)
  endif()
  add_executable(tflm_golden_${mode} tflm_golden.cc
                 ${TFLM_DIR}/tensorflow/lite/micro/kernels/internal/common.cc)
  target_include_directories(tflm_golden_${mode} PRIVATE ${TFLM_DIR})
  target_compile_definitions(tflm_golden_${mode} PRIVATE
                             TFLITE_SINGLE_ROUNDING=${flag}
                             KNPU_GOLDEN_MODE="${mode}")
endforeach()

enable_testing()
set(TOOLS
  --knpu-run $<TARGET_FILE:knpu_run>
  --golden-double $<TARGET_FILE:tflm_golden_double>
  --golden-single $<TARGET_FILE:tflm_golden_single>)
add_test(NAME v3_conv
         COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/test_conv.py
                 ${TOOLS} --work-dir ${CMAKE_CURRENT_BINARY_DIR}/work-conv)
add_test(NAME v3_avgpool
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_avgpool.py
                 ${TOOLS} --work-dir ${CMAKE_CURRENT_BINARY_DIR}/work-avgpool)
add_test(NAME v3_errors
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_errors.py
                 --knpu-run $<TARGET_FILE:knpu_run>
                 --work-dir ${CMAKE_CURRENT_BINARY_DIR}/work-errors)
set_tests_properties(v3_conv v3_avgpool v3_errors PROPERTIES
                     ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
