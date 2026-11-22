# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TensorFlow Lite Micro and the FlatBuffers headers it needs, pinned and
# checked by SHA-256 (docs/design.md 12.2, 14). FlatBuffers is the version
# TFLite Micro pins in its MODULE.bazel, with the same archive hash.
# knpu_download comes from compiler/cmake/deps.cmake.

set(TFLM_COMMIT fd2a85dfb3bedac81920e8667565b1044c6b6a43)
set(TFLM_SHA256
    267dba17d64ff017583669e1f398a43f6bec88071517da3721d002b445b399c1)
set(FLATBUFFERS_VERSION 25.9.23)
set(FLATBUFFERS_SHA256
    9102253214dea6ae10c2ac966ea1ed2155d22202390b532d1dea64935c518ada)

function(knpu_fetch_archive url sha256 name out_dir)
  set(tarball ${KNPU_DEPS_DIR}/${name}.tar.gz)
  knpu_download(${url} ${sha256} ${tarball})
  if(NOT EXISTS ${KNPU_DEPS_DIR}/${name})
    file(ARCHIVE_EXTRACT INPUT ${tarball} DESTINATION ${KNPU_DEPS_DIR})
  endif()
  set(${out_dir} ${KNPU_DEPS_DIR}/${name} PARENT_SCOPE)
endfunction()

knpu_fetch_archive(
  https://github.com/tensorflow/tflite-micro/archive/${TFLM_COMMIT}.tar.gz
  ${TFLM_SHA256} tflite-micro-${TFLM_COMMIT} TFLM_SOURCE_DIR)
knpu_fetch_archive(
  https://github.com/google/flatbuffers/archive/refs/tags/v${FLATBUFFERS_VERSION}.tar.gz
  ${FLATBUFFERS_SHA256} flatbuffers-${FLATBUFFERS_VERSION} FLATBUFFERS_DIR)
