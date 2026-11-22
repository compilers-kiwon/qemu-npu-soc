# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# knpu model compiler (docs/design.md 9), an out-of-tree MLIR project laid
# out like mlir/examples/standalone. It also builds the compiler's end-to-end
# tests in tests/compiler.
#
#   cmake -S compiler -B build/compiler -G Ninja \
#         -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir
#   cmake --build build/compiler
#   cmake --build build/compiler --target check-knpu      # V4 (lit)
#   ctest --test-dir build/compiler --output-on-failure   # V2, V3

cmake_minimum_required(VERSION 3.23)
project(knpu_compiler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(NOT LLVM_EXTERNAL_LIT)
  find_program(LLVM_EXTERNAL_LIT NAMES lit llvm-lit
               HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
endif()
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)

set(KNPU_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(KNPU_BINARY_DIR ${PROJECT_BINARY_DIR})
set(KNPU_REPO_DIR ${PROJECT_SOURCE_DIR}/..)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${KNPU_SOURCE_DIR}/include ${KNPU_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
separate_arguments(llvm_definitions NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${llvm_definitions})

# Pinned downloads: lcc, the TFLite Python schema (docs/design.md 12.2)
include(cmake/deps.cmake)

add_subdirectory(include/knpu/Dialect/Knpu)
add_subdirectory(include/knpu/Transforms)
add_subdirectory(lib/Dialect/Knpu)
add_subdirectory(lib/Transforms)
add_subdirectory(lib/Target)
add_subdirectory(tools/knpu-opt)
add_subdirectory(tools/knpu-translate)
add_subdirectory(tools/knpu-compile)

set(KNPU_TFLITE2C ${KNPU_SOURCE_DIR}/tflite2c/tflite2c.py)

enable_testing()
add_subdirectory(test)
add_subdirectory(${KNPU_REPO_DIR}/tests/compiler
                 ${KNPU_BINARY_DIR}/tests-compiler)
