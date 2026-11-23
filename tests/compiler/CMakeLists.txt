# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# End-to-end tests of the compiler (docs/design.md 11), built as part of
# compiler/:
#   V2  tflm_run_double / tflm_run_single: the TFLite Micro interpreter,
#       built once per rounding mode, runs the .tflite model itself
#   V3  test_conv_model.py: the tflite2c C on gcc, and lcc -> knpu-compile
#       -> knpu_run (shared NPU core), both compared with V2
#   V3  test_mnist_model.py: the MNIST CNN of M2 (so far its V2 outputs and
#       logits, from TFLite Micro and from the tflite2c C on gcc, and the
#       NPU part through lcc -> knpu-compile -> knpu_run, docs/plan.md M2.1
#       to M2.4)
#   V2  test_softmax.py: the tflite2c softmax on gcc against TFLite Micro,
#       over input scales, betas and row lengths

include(${CMAKE_CURRENT_SOURCE_DIR}/tflm-deps.cmake)

# The interpreter and kernel sources from TFLite Micro's own CMake list
function(knpu_tflm_sources out)
  set(PROJECT_SOURCE_DIR ${TFLM_SOURCE_DIR})
  include(${TFLM_SOURCE_DIR}/cmake/generated/tflm_sources.cmake)
  set(${out} ${TFLM_MICRO_CORE_SRCS} ${TFLM_MICRO_KERNEL_SRCS}
             ${TFLM_SYSTEM_DEFAULT_SRCS} PARENT_SCOPE)
endfunction()
knpu_tflm_sources(TFLM_SRCS)

foreach(mode double single)
  if(mode STREQUAL single)
    set(flag 1)
  else()
    set(flag 0)
  endif()
  # Compile definitions as in TFLite Micro's CMakeLists.txt
  add_library(tflm_${mode} STATIC EXCLUDE_FROM_ALL ${TFLM_SRCS})
  target_include_directories(tflm_${mode} SYSTEM PUBLIC
                             ${TFLM_SOURCE_DIR} ${FLATBUFFERS_DIR}/include)
  target_compile_definitions(tflm_${mode}
    PUBLIC TFLITE_SINGLE_ROUNDING=${flag}
           FLATBUFFERS_LOCALE_INDEPENDENT=0 FLATBUFFERS_PREFER_PRINTF=1
    PRIVATE KERNELS_OPTIMIZED_FOR_SPEED TF_LITE_USE_CTIME)
  target_compile_options(tflm_${mode} PRIVATE -w)
  target_link_libraries(tflm_${mode} PUBLIC m)

  add_executable(tflm_run_${mode} tflm_run.cc)
  target_link_libraries(tflm_run_${mode} PRIVATE tflm_${mode})
endforeach()

# The shared NPU core and its host runner (as in tests/npu-core)
set(NPU_CORE_DIR ${KNPU_REPO_DIR}/hw/npu-core)
add_executable(knpu_run ${KNPU_REPO_DIR}/tests/npu-core/knpu_run.c
                        ${NPU_CORE_DIR}/knpu_core.c)
target_include_directories(knpu_run PRIVATE ${NPU_CORE_DIR})
set_target_properties(knpu_run PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

set(CONV_MODEL_TEST
  ${CMAKE_CURRENT_SOURCE_DIR}/test_conv_model.py
  --model ${KNPU_REPO_DIR}/models/conv/conv.tflite
  --tflite2c ${KNPU_TFLITE2C}
  --schema ${KNPU_TFLITE_SCHEMA}
  --lcc-dir ${KNPU_LCC_DIR}
  --cc ${CMAKE_C_COMPILER}
  --knpu-compile $<TARGET_FILE:knpu-compile>
  --knpu-run $<TARGET_FILE:knpu_run>
  --tflm-double $<TARGET_FILE:tflm_run_double>
  --tflm-single $<TARGET_FILE:tflm_run_single>
  --work-dir ${CMAKE_CURRENT_BINARY_DIR}/work-conv
  --vectors ${KNPU_REPO_DIR}/models/conv/vectors)

add_test(NAME v3_conv_model
         COMMAND Python3::Interpreter ${CONV_MODEL_TEST})
set_tests_properties(v3_conv_model PROPERTIES
                     ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)

# Writes the V6 test vectors (models/conv/vectors/) from this build; commit
# them when the compiler's output changes
add_custom_target(update-conv-vectors
  COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
          ${Python3_EXECUTABLE} ${CONV_MODEL_TEST} --update-vectors
  DEPENDS knpu-compile knpu_run tflm_run_double tflm_run_single
  USES_TERMINAL)

set(MNIST_MODEL_TEST
  ${CMAKE_CURRENT_SOURCE_DIR}/test_mnist_model.py
  --model ${KNPU_REPO_DIR}/models/mnist/mnist.tflite
  --digits ${KNPU_REPO_DIR}/models/mnist/digits
  --tflite2c ${KNPU_TFLITE2C}
  --schema ${KNPU_TFLITE_SCHEMA}
  --lcc-dir ${KNPU_LCC_DIR}
  --cc ${CMAKE_C_COMPILER}
  --knpu-compile $<TARGET_FILE:knpu-compile>
  --knpu-run $<TARGET_FILE:knpu_run>
  --tflm-double $<TARGET_FILE:tflm_run_double>
  --tflm-single $<TARGET_FILE:tflm_run_single>
  --work-dir ${CMAKE_CURRENT_BINARY_DIR}/work-mnist)

add_test(NAME v3_mnist_model
         COMMAND Python3::Interpreter ${MNIST_MODEL_TEST})
add_test(NAME v2_softmax
         COMMAND Python3::Interpreter
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_softmax.py
                 --tflite2c ${KNPU_TFLITE2C}
                 --schema ${KNPU_TFLITE_SCHEMA}
                 --cc ${CMAKE_C_COMPILER}
                 --tflm-double $<TARGET_FILE:tflm_run_double>
                 --tflm-single $<TARGET_FILE:tflm_run_single>
                 --work-dir ${CMAKE_CURRENT_BINARY_DIR}/work-softmax)
set_tests_properties(v3_mnist_model v2_softmax PROPERTIES
                     ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
