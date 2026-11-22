// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-opt: runs knpu passes on MLIR files (lit tests, debugging).

#include "knpu/Dialect/Knpu/KnpuDialect.h"
#include "knpu/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::knpu::registerKnpuPasses();

  // lcc output uses llvm, func, cf and arith; the graph adds tensor
  mlir::DialectRegistry registry;
  registry.insert<mlir::knpu::KnpuDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::tensor::TensorDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "knpu optimizer driver\n", registry));
}
