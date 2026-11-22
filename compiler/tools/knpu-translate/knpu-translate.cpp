// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-translate: writes model.knpu from scheduled knpu commands
// (--knpu-to-model).

#include "knpu/Target/ModelFile.h"

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::knpu::registerToModelFileTranslation();
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "knpu translation tool"));
}
