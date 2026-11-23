// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The CPU path of knpu-compile (docs/design.md 9.4, D45). The functions of
// the cpu layers, which knpu-import keeps with external linkage, move into
// a module of their own together with the functions and globals they use.
// lcc-opt, mlir-translate, clang and ld.lld turn that module into the
// AArch64 shared object of the CPU_CODE section (docs/model-format.md 9).

#include "knpu/Target/CpuCode.h"

#include "knpu/Dialect/Knpu/KnpuOps.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

using namespace mlir;
using namespace mlir::knpu;

FailureOr<OwningOpRef<ModuleOp>>
mlir::knpu::extractCpuModule(ModuleOp module) {
  SymbolTable symbols(module);
  llvm::SetVector<Operation *> used;
  auto use = [&](Operation *user, StringAttr name) -> LogicalResult {
    Operation *def = symbols.lookup(name);
    if (!def)
      return user->emitOpError()
             << "uses @" << name.getValue()
             << ", which the module does not define";
    if (auto fn = dyn_cast<func::FuncOp>(def);
        fn && fn.getSymName() == kModelFunc)
      return user->emitOpError("the CPU code cannot use the model function");
    used.insert(def);
    return success();
  };

  for (CpuStepOp step : module.getOps<CpuStepOp>())
    if (failed(use(step, step.getSymbolAttr().getAttr())))
      return failure();
  if (used.empty())
    return OwningOpRef<ModuleOp>();
  // The symbols those use, transitively; the set grows while it is walked
  for (size_t i = 0; i < used.size(); ++i) {
    Operation *op = used[i];
    auto uses = SymbolTable::getSymbolUses(op);
    if (!uses)
      return op->emitOpError("may hold symbol tables of unknown operations");
    for (const SymbolTable::SymbolUse &u : *uses)
      if (failed(use(u.getUser(), u.getSymbolRef().getRootReference())))
        return failure();
  }

  OwningOpRef<ModuleOp> cpu = ModuleOp::create(module.getLoc());
  for (NamedAttribute attr : module->getDiscardableAttrDictionary())
    if (!attr.getName().getValue().starts_with("knpu."))
      (*cpu)->setAttr(attr.getName(), attr.getValue());
  Block *body = cpu->getBody();
  for (Operation &op : llvm::make_early_inc_range(*module.getBody()))
    if (used.contains(&op))
      op.moveBefore(body, body->end());
  return cpu;
}

/// Runs one tool and waits for it; the tool writes its own messages to
/// stderr.
static LogicalResult runTool(ModuleOp cpu, StringRef name, StringRef program,
                             ArrayRef<StringRef> args) {
  SmallVector<StringRef> argv{program};
  argv.append(args.begin(), args.end());
  std::string message;
  bool notRun = false;
  int code = llvm::sys::ExecuteAndWait(program, argv, std::nullopt, {}, 0, 0,
                                       &message, &notRun);
  if (notRun)
    return emitError(cpu.getLoc()) << "cannot run " << name << " (" << program
                           << "): " << message;
  if (code != 0) {
    InFlightDiagnostic diag = emitError(cpu.getLoc())
                              << name << " failed (exit code " << code << ")";
    if (!message.empty())
      diag << ": " << message;
    return diag;
  }
  return success();
}

FailureOr<std::vector<char>>
mlir::knpu::compileCpuModule(ModuleOp cpu, const CpuTools &tools) {
  SmallString<128> dir;
  if (std::error_code ec =
          llvm::sys::fs::createUniqueDirectory("knpu-cpu", dir))
    return emitError(cpu.getLoc()) << "cannot make a temporary directory: "
                           << ec.message();
  llvm::scope_exit removeDir(
      [&] { (void)llvm::sys::fs::remove_directories(dir); });
  auto file = [&](StringRef name) {
    SmallString<128> path(dir);
    llvm::sys::path::append(path, name);
    return std::string(path);
  };
  std::string input = file("cpu.mlirbc"), lowered = file("lowered.mlirbc"),
              ir = file("cpu.ll"), object = file("cpu.o"),
              shared = file("cpu.so");

  {
    std::error_code ec;
    llvm::raw_fd_ostream os(input, ec);
    if (ec)
      return emitError(cpu.getLoc()) << "cannot write " << input << ": "
                             << ec.message();
    if (failed(writeBytecodeToFile(cpu, os)))
      return emitError(cpu.getLoc(), "cannot write the CPU module as bytecode");
  }
  if (failed(runTool(cpu, "lcc-opt", tools.lccOpt,
                     {"--lcc-to-linalg", "--lcc-lower", "--emit-bytecode",
                      input, "-o", lowered})) ||
      failed(runTool(cpu, "mlir-translate", tools.mlirTranslate,
                     {"--mlir-to-llvmir", lowered, "-o", ir})) ||
      failed(runTool(cpu, "clang", tools.clang,
                     {"--target=aarch64-linux-gnu", "-O2", "-fPIC", "-c", ir,
                      "-o", object})) ||
      failed(runTool(cpu, "ld.lld", tools.ldLld,
                     {"-shared", object, "-o", shared})))
    return failure();

  auto buffer = llvm::MemoryBuffer::getFile(shared, /*IsText=*/false,
                                            /*RequiresNullTerminator=*/false);
  if (!buffer)
    return emitError(cpu.getLoc()) << "cannot read " << shared << ": "
                           << buffer.getError().message();
  StringRef bytes = (*buffer)->getBuffer();
  return std::vector<char>(bytes.begin(), bytes.end());
}

void mlir::knpu::addCpuCode(ModuleOp module, ArrayRef<char> code) {
  OpBuilder b(module.getContext());
  if (auto fn = module.lookupSymbol<func::FuncOp>(kModelFunc))
    b.setInsertionPoint(fn);
  else
    b.setInsertionPointToEnd(module.getBody());
  auto type = RankedTensorType::get({static_cast<int64_t>(code.size())},
                                    b.getIntegerType(8));
  CpuCodeOp::create(b, module.getLoc(),
                    DenseElementsAttr::getFromRawBuffer(type, code));
}
