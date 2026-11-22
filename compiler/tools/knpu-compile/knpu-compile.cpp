// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-compile: compiles the .mlir that lcc writes for a tflite2c model
// (lcc -S -Wf-pragma=knpu) into model.knpu. It runs knpu-import, knpu-tile,
// knpu-schedule and knpu-insert-waits and then the model file writer.
//
//   knpu-compile model.mlir -o model.knpu [--sram-size=BYTES]
//                [--emit-mlir] [--insert-waits=false]

#include "knpu/Dialect/Knpu/KnpuDialect.h"
#include "knpu/Target/ModelFile.h"
#include "knpu/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFile(cl::Positional,
                                      cl::desc("<lcc output .mlir>"),
                                      cl::init("-"));
static cl::opt<std::string> outputFile("o", cl::desc("Output file"),
                                       cl::value_desc("filename"),
                                       cl::init("-"));
static cl::opt<int64_t> sramSize("sram-size",
                                 cl::desc("SRAM size of the NPU in bytes"),
                                 cl::init(65536));
static cl::opt<bool> emitMlir("emit-mlir",
                              cl::desc("Write the scheduled MLIR instead of "
                                       "model.knpu"),
                              cl::init(false));
static cl::opt<bool> insertWaits(
    "insert-waits",
    cl::desc("Insert wait commands (turning this off is only for showing "
             "that the device then stops with a synchronization error)"),
    cl::init(true));

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "knpu model compiler\n");

  if (sramSize < 4096 || sramSize > (16 << 20) || sramSize % 4096) {
    llvm::errs() << "knpu-compile: --sram-size must be a multiple of 4096 "
                    "in 4096..16777216\n";
    return 1;
  }

  DialectRegistry registry;
  registry.insert<knpu::KnpuDialect, arith::ArithDialect,
                  cf::ControlFlowDialect, func::FuncDialect,
                  LLVM::LLVMDialect, tensor::TensorDialect>();
  MLIRContext context(registry);

  std::string error;
  auto input = openInputFile(inputFile, &error);
  if (!input) {
    llvm::errs() << "knpu-compile: " << error << "\n";
    return 1;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  SourceMgrDiagnosticHandler diagnostics(sourceMgr, &context);
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return 1;

  PassManager pm(&context);
  if (failed(applyPassManagerCLOptions(pm)))
    return 1;
  pm.addPass(knpu::createKnpuImport());
  knpu::KnpuTileOptions tileOptions;
  tileOptions.sramSize = sramSize;
  pm.addPass(knpu::createKnpuTile(tileOptions));
  knpu::KnpuScheduleOptions scheduleOptions;
  scheduleOptions.sramSize = sramSize;
  pm.addPass(knpu::createKnpuSchedule(scheduleOptions));
  if (insertWaits)
    pm.addPass(knpu::createKnpuInsertWaits());
  if (failed(pm.run(*module)))
    return 1;

  auto output = openOutputFile(outputFile, &error);
  if (!output) {
    llvm::errs() << "knpu-compile: " << error << "\n";
    return 1;
  }
  if (emitMlir)
    module->print(output->os());
  else if (failed(knpu::writeModelFile(*module, output->os())))
    return 1;
  output->keep();
  return 0;
}
