// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// knpu-compile: compiles the .mlir that lcc writes for a tflite2c model
// (lcc -S -Wf-pragma=knpu) into model.knpu. It runs knpu-import, knpu-tile,
// knpu-schedule, knpu-insert-waits and knpu-estimate and then the model
// file writer. knpu-schedule runs with both of its SRAM placement rules,
// each followed by knpu-insert-waits and knpu-estimate on a copy of the
// tiled module, and the one with fewer estimated cycles is kept (the
// current rule on a tie): the current rule chooses greedily and is slower
// than the one before it on some models. With cpu layers it moves their
// functions into the CPU module and compiles that with lcc-opt,
// mlir-translate, clang and ld.lld into the CPU code of the model
// (docs/design.md 9.4).
//
//   knpu-compile model.mlir -o model.knpu [--sram-size=BYTES]
//                [--overlap=false] [--placement=cost|largest]
//                [--report] [--emit-mlir]
//                [--emit-cpu-mlir] [--insert-waits=false]
//                [--macs-per-cycle=N] [--dma-bytes-per-cycle=N]
//                [--pool-elements-per-cycle=N]
//                [--add-elements-per-cycle=N] [--lcc-opt=PATH]
//                [--mlir-translate=PATH] [--clang=PATH] [--ld-lld=PATH]

#include "knpu/Dialect/Knpu/KnpuDialect.h"
#include "knpu/Target/CpuCode.h"
#include "knpu/Target/ModelFile.h"
#include "knpu/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
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
static cl::opt<bool> overlap(
    "overlap",
    cl::desc("Keep activations in SRAM and load constants and tiles ahead "
             "(false: the serial schedule, for comparison)"),
    cl::init(true));
static cl::opt<knpu::Placement> placement(
    "placement",
    cl::desc("Lay out SRAM with this rule of knpu-schedule instead of the "
             "one that gives fewer estimated cycles"),
    cl::values(clEnumValN(knpu::Placement::Cost, "cost",
                          "the current rule (kept on a tie)"),
               clEnumValN(knpu::Placement::Largest, "largest",
                          "the rule before M3.6")));
static cl::opt<bool> report(
    "report",
    cl::desc("Print the estimated cycles per layer and in total to stdout"),
    cl::init(false));
static cl::opt<bool> emitMlir("emit-mlir",
                              cl::desc("Write the scheduled MLIR instead of "
                                       "model.knpu"),
                              cl::init(false));
static cl::opt<bool> emitCpuMlir(
    "emit-cpu-mlir",
    cl::desc("Write the CPU module (the cpu layers that lcc-opt compiles) "
             "instead of model.knpu"),
    cl::init(false));
static cl::opt<bool> insertWaits(
    "insert-waits",
    cl::desc("Insert wait commands (turning this off is only for showing "
             "that the device then stops with a synchronization error)"),
    cl::init(true));
static cl::opt<int64_t> macsPerCycle(
    "macs-per-cycle", cl::desc("Cycle estimate: MACs per cycle"),
    cl::init(256));
static cl::opt<int64_t> dmaBytesPerCycle(
    "dma-bytes-per-cycle", cl::desc("Cycle estimate: DMA bytes per cycle"),
    cl::init(16));
static cl::opt<int64_t> poolElementsPerCycle(
    "pool-elements-per-cycle",
    cl::desc("Cycle estimate: average pooling window elements per cycle"),
    cl::init(256));
static cl::opt<int64_t> addElementsPerCycle(
    "add-elements-per-cycle",
    cl::desc("Cycle estimate: addition output elements per cycle"),
    cl::init(256));
// The tools of the CPU code; CMake puts in the ones it built and found
static cl::opt<std::string> lccOpt("lcc-opt", cl::desc("lcc-opt to run"),
                                   cl::value_desc("path"),
                                   cl::init(KNPU_LCC_OPT));
static cl::opt<std::string>
    mlirTranslate("mlir-translate", cl::desc("mlir-translate to run"),
                  cl::value_desc("path"), cl::init(KNPU_MLIR_TRANSLATE));
static cl::opt<std::string> clang("clang", cl::desc("clang to run"),
                                  cl::value_desc("path"),
                                  cl::init(KNPU_CLANG));
static cl::opt<std::string> ldLld("ld-lld", cl::desc("ld.lld to run"),
                                  cl::value_desc("path"),
                                  cl::init(KNPU_LD_LLD));

/// The name of the layer function a command came from (knpu-import puts it
/// in the location), or "?".
static StringRef layerName(Location loc) {
  if (auto name = dyn_cast<NameLoc>(loc))
    return name.getName().getValue();
  return "?";
}

/// The estimated cycles of the whole job (knpu-estimate), or 0.
static int64_t totalCycles(ModuleOp module) {
  auto fn = module.lookupSymbol<func::FuncOp>(knpu::kModelFunc);
  auto total =
      fn ? fn->getAttrOfType<IntegerAttr>(knpu::kTotalCyclesAttr) : nullptr;
  return total ? total.getInt() : 0;
}

static StringRef placementName(knpu::Placement rule) {
  return rule == knpu::Placement::Cost ? "cost" : "largest";
}

/// Per layer, in the order of their first operation command: the
/// operations, the bytes and the cycles (from knpu-estimate) of the layer's
/// commands. The DMAs of constants loaded ahead count for the layer that
/// uses them. Then the total and, with overlap, the SRAM placement rule
/// kept, with the total of the other rule when both were tried.
static void printReport(ModuleOp module, knpu::Placement rule,
                        std::optional<int64_t> otherCycles,
                        llvm::raw_ostream &os) {
  auto fn = module.lookupSymbol<func::FuncOp>(knpu::kModelFunc);
  struct Row {
    std::string op;
    int64_t tiles = 0, dmaBytes = 0;
    int64_t opStart = -1, opEnd = 0, dmaStart = -1, dmaEnd = 0;
  };
  llvm::MapVector<StringRef, Row> rows;
  for (Operation &op : fn.getBody().front()) {
    auto cycles = op.getAttrOfType<DenseI64ArrayAttr>(knpu::kCyclesAttr);
    if (!cycles || !isa<knpu::ConvOp, knpu::PoolOp, knpu::ElementwiseOp>(op))
      continue;
    Row &row = rows[layerName(op.getLoc())];
    row.op = isa<knpu::ConvOp>(op)   ? "conv"
             : isa<knpu::PoolOp>(op) ? "pool"
                                     : "add";
    ++row.tiles;
    if (row.opStart < 0)
      row.opStart = cycles[0];
    row.opEnd = std::max(row.opEnd, cycles[1]);
  }
  for (Operation &op : fn.getBody().front()) {
    auto cycles = op.getAttrOfType<DenseI64ArrayAttr>(knpu::kCyclesAttr);
    auto dma = dyn_cast<knpu::DmaOp>(op);
    if (!cycles || !dma)
      continue;
    Row &row = rows[layerName(op.getLoc())];
    row.dmaBytes += dma.getLength() * dma.getCount();
    if (row.dmaStart < 0)
      row.dmaStart = cycles[0];
    row.dmaEnd = std::max(row.dmaEnd, cycles[1]);
  }

  auto span = [](int64_t start, int64_t end) {
    return start < 0 ? std::string("-")
                     : std::to_string(start) + ".." + std::to_string(end);
  };
  os << "estimated cycles (" << macsPerCycle << " MACs, " << dmaBytesPerCycle
     << " DMA bytes, " << poolElementsPerCycle << " pool elements, "
     << addElementsPerCycle << " add elements per cycle; "
     << (overlap ? "overlapped" : "serial") << " schedule, " << sramSize
     << "-byte SRAM)\n";
  os << llvm::format("  %-12s %-5s %5s %13s %10s %13s\n", "layer", "op",
                     "tiles", "operations", "DMA bytes", "DMA");
  for (auto &[name, row] : rows)
    os << llvm::format("  %-12s %-5s %5lld %13s %10lld %13s\n",
                       name.str().c_str(),
                       row.op.empty() ? "-" : row.op.c_str(),
                       static_cast<long long>(row.tiles),
                       span(row.opStart, row.opEnd).c_str(),
                       static_cast<long long>(row.dmaBytes),
                       span(row.dmaStart, row.dmaEnd).c_str());
  os << "  total cycles " << totalCycles(module) << "\n";
  if (!overlap)
    return;
  os << "  SRAM placement " << placementName(rule);
  if (otherCycles)
    os << " (" << (rule == knpu::Placement::Cost ? "largest" : "cost")
       << ": " << *otherCycles << " cycles)";
  os << "\n";
}

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
  if (report && outputFile == "-") {
    llvm::errs() << "knpu-compile: --report prints to stdout; write the "
                    "output to a file with -o\n";
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
  tileOptions.overlap = overlap;
  pm.addPass(knpu::createKnpuTile(tileOptions));
  if (failed(pm.run(*module)))
    return 1;

  // knpu-schedule with one SRAM placement rule, then knpu-insert-waits and
  // knpu-estimate, on a copy of the tiled module
  knpu::KnpuEstimateOptions estimateOptions;
  estimateOptions.macsPerCycle = macsPerCycle;
  estimateOptions.dmaBytesPerCycle = dmaBytesPerCycle;
  estimateOptions.poolElementsPerCycle = poolElementsPerCycle;
  estimateOptions.addElementsPerCycle = addElementsPerCycle;
  auto schedule = [&](knpu::Placement rule) -> OwningOpRef<ModuleOp> {
    OwningOpRef<ModuleOp> copy(module->clone());
    PassManager spm(&context);
    if (failed(applyPassManagerCLOptions(spm)))
      return nullptr;
    knpu::KnpuScheduleOptions scheduleOptions;
    scheduleOptions.sramSize = sramSize;
    scheduleOptions.overlap = overlap;
    scheduleOptions.placement = rule;
    spm.addPass(knpu::createKnpuSchedule(scheduleOptions));
    if (insertWaits)
      spm.addPass(knpu::createKnpuInsertWaits());
    spm.addPass(knpu::createKnpuEstimate(estimateOptions));
    if (failed(spm.run(*copy)))
      return nullptr;
    return copy;
  };
  // Both rules unless --placement chooses one; without overlap there is
  // nothing to give up and both give the serial schedule
  bool fixed = placement.getNumOccurrences() > 0;
  knpu::Placement rule = fixed ? placement.getValue() : knpu::Placement::Cost;
  OwningOpRef<ModuleOp> scheduled = schedule(rule);
  if (!scheduled)
    return 1;
  std::optional<int64_t> otherCycles;
  if (!fixed && overlap) {
    OwningOpRef<ModuleOp> other = schedule(knpu::Placement::Largest);
    if (!other)
      return 1;
    if (totalCycles(*other) < totalCycles(*scheduled)) {
      std::swap(scheduled, other);
      rule = knpu::Placement::Largest;
    }
    otherCycles = totalCycles(*other);
  }
  module = std::move(scheduled);

  // The cpu layers leave the module for the CPU module, which becomes the
  // AArch64 shared object of knpu.cpu_code (docs/design.md 9.4)
  OwningOpRef<ModuleOp> cpu;
  if (!emitMlir) {
    auto extracted = knpu::extractCpuModule(*module);
    if (failed(extracted))
      return 1;
    cpu = std::move(*extracted);
    if (emitCpuMlir && !cpu) {
      llvm::errs() << "knpu-compile: the model has no cpu layers\n";
      return 1;
    }
    if (cpu && !emitCpuMlir) {
      auto code = knpu::compileCpuModule(
          *cpu, {lccOpt, mlirTranslate, clang, ldLld});
      if (failed(code))
        return 1;
      knpu::addCpuCode(*module, *code);
    }
  }

  auto output = openOutputFile(outputFile, &error);
  if (!output) {
    llvm::errs() << "knpu-compile: " << error << "\n";
    return 1;
  }
  if (emitMlir)
    module->print(output->os());
  else if (emitCpuMlir)
    cpu->print(output->os());
  else if (failed(knpu::writeModelFile(*module, output->os())))
    return 1;
  output->keep();
  if (report)
    printReport(*module, rule, otherCycles, llvm::outs());
  return 0;
}
