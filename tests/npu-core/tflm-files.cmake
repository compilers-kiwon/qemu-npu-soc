# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TensorFlow Lite Micro files that the V2 reference build needs, pinned to
# one commit and checked by SHA-256 (docs/design.md 14). The list is the
# include closure of the int8 reference convolution and average pooling and
# of common.cc.

set(TFLM_COMMIT fd2a85dfb3bedac81920e8667565b1044c6b6a43)
set(TFLM_URL https://raw.githubusercontent.com/tensorflow/tflite-micro)

set(TFLM_FILES
  tensorflow/lite/micro/compatibility.h
  c6f9ec2dfb70af1f1138feece2d80faf74e07bf9144a58c5ba36e9eeb921b4e8
  tensorflow/lite/micro/micro_log.h
  892c5bee78a8e0a13b3ef5c335d4cca9b5fec62163c8487b73f1579798dffd61
  tensorflow/lite/micro/kernels/internal/common.cc
  3638e8f6ef7995f43d1ac1c7bd7d83952b0a1ec7a58086c22db1c5ad3ef266c2
  tensorflow/lite/micro/kernels/internal/common.h
  2742b4738307c5647bc1da0c77db125faa315b0fd90933f2a370f89d94c9c06e
  tensorflow/lite/micro/kernels/internal/cppmath.h
  7a4eebbb914018ae43dc4fae834d521b42cfb793d03fccf90fb94a8beccbc648
  tensorflow/lite/micro/kernels/internal/fixedpoint.h
  c19cb30b01ae9452c434140371502a7891bcc07a85f3d30a262e0444a11dc189
  tensorflow/lite/micro/kernels/internal/runtime_shape.h
  ace53bea7c2eb9c6e025dc2cd7b23d42977ce3860cbd9018c4c3bbe6de47b92c
  tensorflow/lite/micro/kernels/internal/types.h
  aa30efcc27f0933f2005e211cc098bd63f821ae65e8f228abfd3361804e1110f
  tensorflow/lite/micro/kernels/internal/reference/integer_ops/conv.h
  94fd76303db2a5a22a77a7056a205b8ff02496d5efe996439c4f556c4843719a
  tensorflow/lite/micro/kernels/internal/reference/integer_ops/pooling.h
  c7fe4656d2991d8912ec1b2754b034c63f68c45fed79b00e44a761ac8d88e3af
)
