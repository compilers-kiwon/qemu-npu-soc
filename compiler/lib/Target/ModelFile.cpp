// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Writes model.knpu v0 (docs/model-format.md) from command operations. The
// encoder tracks the SET state of the NPU (docs/npu-spec.md 5) and emits a
// SET command only when its value changes. Base addresses and shapes are
// always set before their first use. A wait command is emitted right before
// the OP word of the next command, after that command's SET commands.
// A model with cpu layers also gets the CPU_CODE and CPU_STEPS sections of
// format minor 1 from knpu.cpu_code and knpu.cpu_step.

#include "knpu/Target/ModelFile.h"

#include "knpu/Dialect/Knpu/KnpuOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>

#include "knpu_cmd.h"

using namespace mlir;
using namespace mlir::knpu;

namespace {

/// Command stream encoder with SET deduplication.
class Encoder {
public:
  Encoder() {
    // SET state after START (docs/npu-spec.md 5); bases and shapes are
    // treated as unknown so they are always set before their first use
    last[KNPU_SET_IFM_ZP] = {0, 0};
    last[KNPU_SET_OFM_ZP] = {0, 0};
    last[KNPU_SET_KERNEL] = {0x0101, 0};
    last[KNPU_SET_STRIDE] = {0x0101, 0};
    last[KNPU_SET_PAD] = {0, 0};
    last[KNPU_SET_CLAMP] = {clampParam(-128, 127), 0};
    last[KNPU_SET_ROUND_MODE] = {KNPU_ROUND_DOUBLE, 0};
    last[KNPU_SET_DMA_SRC] = {KNPU_SPACE(KNPU_SPACE_SRAM, 0), 0};
    last[KNPU_SET_DMA_DST] = {KNPU_SPACE(KNPU_SPACE_SRAM, 0), 0};
    last[KNPU_SET_DMA_COUNT] = {1, 0};
    last[KNPU_SET_DMA_STRIDE] = {0, 0};
  }

  static uint16_t clampParam(int64_t min, int64_t max) {
    return static_cast<uint16_t>(((max & 0xFF) << 8) | (min & 0xFF));
  }

  void set(uint16_t cmd, uint16_t param, uint32_t payload = 0) {
    auto it = last.find(cmd);
    if (it != last.end() && it->second == std::make_pair(param, payload))
      return;
    last[cmd] = {param, payload};
    words.push_back(KNPU_CMD_WORD(cmd, param));
    if (cmd & KNPU_CMD_PAYLOAD)
      words.push_back(payload);
  }

  /// Wait commands wait for the next command's OP word.
  void wait(uint16_t cmd, int64_t pending) {
    pendingWaits.push_back(KNPU_CMD_WORD(cmd, pending));
  }

  void op(uint16_t cmd) {
    llvm::append_range(words, pendingWaits);
    pendingWaits.clear();
    words.push_back(KNPU_CMD_WORD(cmd, 0));
  }

  void dma(DmaOp d) {
    uint16_t region = KNPU_SPACE(KNPU_SPACE_REGION, d.getRegion());
    uint16_t sram = KNPU_SPACE(KNPU_SPACE_SRAM, 0);
    uint32_t regionOff = d.getRegionOffset(), sramOff = d.getSramOffset();
    bool load = d.getDir() == "load";
    set(KNPU_SET_DMA_SRC, load ? region : sram, load ? regionOff : sramOff);
    set(KNPU_SET_DMA_DST, load ? sram : region, load ? sramOff : regionOff);
    set(KNPU_SET_DMA_LEN, 0, d.getLength());
    set(KNPU_SET_DMA_COUNT, d.getCount());
    if (d.getCount() > 1)
      set(KNPU_SET_DMA_STRIDE, 0, d.getStride());
    op(KNPU_OP_DMA);
  }

  void conv(ConvOp c) {
    ArrayRef<int64_t> in = c.getIfmShape(), out = c.getOfmShape(),
                      k = c.getKernel(), s = c.getStride(), p = c.getPad();
    set(KNPU_SET_IFM_BASE, 0, c.getIfm());
    set(KNPU_SET_IFM_SHAPE, in[2], (in[0] << 16) | in[1]);
    set(KNPU_SET_IFM_ZP, static_cast<uint16_t>(c.getIfmZp()));
    set(KNPU_SET_WEIGHT_BASE, 0, c.getWeight());
    set(KNPU_SET_BIAS_BASE, 0, c.getBias());
    set(KNPU_SET_RQ_TABLE, 0, c.getRqTable());
    set(KNPU_SET_KERNEL, (k[0] << 8) | k[1]);
    set(KNPU_SET_STRIDE, (s[0] << 8) | s[1]);
    set(KNPU_SET_PAD, (p[3] << 12) | (p[2] << 8) | (p[1] << 4) | p[0]);
    set(KNPU_SET_OFM_BASE, 0, c.getOfm());
    set(KNPU_SET_OFM_SHAPE, out[2], (out[0] << 16) | out[1]);
    set(KNPU_SET_OFM_ZP, static_cast<uint16_t>(c.getOfmZp()));
    set(KNPU_SET_CLAMP, clampParam(c.getClamp()[0], c.getClamp()[1]));
    set(KNPU_SET_ROUND_MODE,
        c.getRound() == "single" ? KNPU_ROUND_SINGLE : KNPU_ROUND_DOUBLE);
    op(KNPU_OP_CONV);
  }

  /// OP_AVGPOOL uses neither the zero points nor ROUND_MODE
  void pool(PoolOp p) {
    ArrayRef<int64_t> in = p.getIfmShape(), out = p.getOfmShape(),
                      k = p.getKernel(), s = p.getStride(), pd = p.getPad();
    set(KNPU_SET_IFM_BASE, 0, p.getIfm());
    set(KNPU_SET_IFM_SHAPE, in[2], (in[0] << 16) | in[1]);
    set(KNPU_SET_KERNEL, (k[0] << 8) | k[1]);
    set(KNPU_SET_STRIDE, (s[0] << 8) | s[1]);
    set(KNPU_SET_PAD, (pd[3] << 12) | (pd[2] << 8) | (pd[1] << 4) | pd[0]);
    set(KNPU_SET_OFM_BASE, 0, p.getOfm());
    set(KNPU_SET_OFM_SHAPE, out[2], (out[0] << 16) | out[1]);
    set(KNPU_SET_CLAMP, clampParam(p.getClamp()[0], p.getClamp()[1]));
    op(KNPU_OP_AVGPOOL);
  }

  std::vector<uint32_t> words;

private:
  std::map<uint16_t, std::pair<uint16_t, uint32_t>> last;
  std::vector<uint32_t> pendingWaits;
};

/// Little-endian byte buffer.
class Bytes {
public:
  void u8(uint8_t v) { data.push_back(v); }
  void u16(uint16_t v) { u8(v), u8(v >> 8); }
  void u32(uint32_t v) { u16(v), u16(v >> 16); }
  void i32(int32_t v) { u32(static_cast<uint32_t>(v)); }
  void f32(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u32(u);
  }
  void zeros(size_t n) { data.insert(data.end(), n, 0); }
  void align16() { zeros((16 - data.size() % 16) % 16); }
  void put32(size_t at, uint32_t v) {
    for (int i = 0; i < 4; ++i)
      data[at + i] = static_cast<uint8_t>(v >> (8 * i));
  }
  size_t size() const { return data.size(); }
  std::vector<uint8_t> data;
};

enum SectionKind : uint32_t {
  kCmds = 1,
  kRegions = 2,
  kConst = 3,
  kTensors = 4,
  kCpuCode = 5,
  kCpuSteps = 6,
};

/// A symbol name of CPU_STEPS: a non-empty C identifier
/// (docs/model-format.md 10).
bool isCIdentifier(StringRef name) {
  return !name.empty() && !llvm::isDigit(name.front()) &&
         llvm::all_of(name,
                      [](char c) { return llvm::isAlnum(c) || c == '_'; });
}

} // namespace

LogicalResult mlir::knpu::writeModelFile(ModuleOp module,
                                         llvm::raw_ostream &os) {
  auto sramAttr = module->getAttrOfType<IntegerAttr>(kSramSizeAttr);
  if (!sramAttr)
    return module.emitError() << "missing " << kSramSizeAttr
                              << " (run knpu-schedule first)";
  int64_t sramSize = sramAttr.getInt();
  if (sramSize < 4096 || sramSize > (16 << 20) || sramSize % 4096)
    return module.emitError("SRAM size must be a multiple of 4 KiB in "
                            "4 KiB..16 MiB");

  func::FuncOp program;
  SmallVector<RegionOp> regions;
  SmallVector<IoOp> ios;
  SmallVector<CpuStepOp> steps;
  SmallVector<CpuCodeOp> codes;
  for (Operation &op : *module.getBody()) {
    if (auto r = dyn_cast<RegionOp>(op))
      regions.push_back(r);
    else if (auto io = dyn_cast<IoOp>(op))
      ios.push_back(io);
    else if (auto s = dyn_cast<CpuStepOp>(op))
      steps.push_back(s);
    else if (auto c = dyn_cast<CpuCodeOp>(op))
      codes.push_back(c);
    else if (auto f = dyn_cast<func::FuncOp>(op);
             f && f.getSymName() == kModelFunc && !program)
      program = f;
    else
      return op.emitError("model.knpu v0 has no place for this operation");
  }
  if (!program || program.getNumArguments() || program.getNumResults())
    return module.emitError("no knpu_model command function");

  // Command stream
  Encoder enc;
  Operation *lastCmd = nullptr;
  for (Operation &op : program.getBody().front()) {
    if (auto d = dyn_cast<DmaOp>(op))
      enc.dma(d);
    else if (auto c = dyn_cast<ConvOp>(op))
      enc.conv(c);
    else if (auto p = dyn_cast<PoolOp>(op))
      enc.pool(p);
    else if (auto w = dyn_cast<DmaWaitOp>(op))
      enc.wait(KNPU_OP_DMA_WAIT, w.getPending());
    else if (auto w = dyn_cast<KernelWaitOp>(op))
      enc.wait(KNPU_OP_KERNEL_WAIT, w.getPending());
    else if (isa<StopOp>(op))
      enc.op(KNPU_OP_STOP);
    else if (!isa<func::ReturnOp>(op))
      return op.emitError("is not an NPU command");
    if (!isa<func::ReturnOp>(op))
      lastCmd = &op;
  }
  if (!lastCmd || !isa<StopOp>(lastCmd))
    return program.emitError("the command stream must end with knpu.stop");

  // Regions, in index order; constant regions get a CONST section each
  llvm::sort(regions, [](RegionOp a, RegionOp b) {
    return a.getIndex() < b.getIndex();
  });
  for (size_t i = 1; i < regions.size(); ++i)
    if (regions[i].getIndex() == regions[i - 1].getIndex())
      return regions[i].emitOpError("duplicate region index");
  llvm::DenseMap<int64_t, RegionOp> regionByIndex;
  for (RegionOp r : regions)
    regionByIndex[r.getIndex()] = r;

  // Tensors: inputs first, then outputs (docs/model-format.md 8)
  std::stable_partition(ios.begin(), ios.end(), [](IoOp io) {
    return io.getDirection() == "input";
  });
  for (IoOp io : ios) {
    auto it = regionByIndex.find(io.getRegion());
    if (it == regionByIndex.end() || it->second.getRole() != "arena")
      return io.emitOpError("must be in an arena region");
    int64_t size = 1;
    for (int64_t d : io.getShape())
      size *= d;
    if (io.getOffset() + size > it->second.getSize())
      return io.emitOpError("is outside its region");
  }

  // Layout: header, section table, CMDS, REGIONS, CONST..., TENSORS and
  // with cpu layers CPU_CODE, CPU_STEPS
  struct Section {
    uint32_t kind;
    Bytes body;
  };
  std::vector<Section> sections;
  Section cmds{kCmds, {}};
  for (uint32_t w : enc.words)
    cmds.body.u32(w);
  sections.push_back(std::move(cmds));

  size_t firstConst = 2;
  Section regionTable{kRegions, {}};
  size_t nextConst = firstConst;
  for (RegionOp r : regions) {
    regionTable.body.u32(r.getIndex());
    regionTable.body.u32(r.getRole() == "const" ? 0 : 1);
    regionTable.body.u32(r.getSize());
    regionTable.body.u32(r.getRole() == "const" ? nextConst++ : 0xFFFFFFFFu);
  }
  sections.push_back(std::move(regionTable));
  for (RegionOp r : regions) {
    if (r.getRole() != "const")
      continue;
    Section c{kConst, {}};
    for (const APInt &v : r.getContent()->getValues<APInt>())
      c.body.u8(static_cast<uint8_t>(v.getZExtValue()));
    sections.push_back(std::move(c));
  }

  Section tensors{kTensors, {}};
  for (IoOp io : ios) {
    ArrayRef<int64_t> shape = io.getShape();
    int64_t size = 1;
    for (int64_t d : shape)
      size *= d;
    tensors.body.u8(io.getDirection() == "input" ? 0 : 1);
    tensors.body.u8(0); // int8
    tensors.body.u8(shape.size());
    tensors.body.u8(0);
    tensors.body.u32(io.getRegion());
    tensors.body.u32(io.getOffset());
    tensors.body.u32(size);
    for (size_t i = 0; i < 4; ++i)
      tensors.body.u32(i < shape.size() ? shape[i] : 0);
    tensors.body.f32(io.getScale().convertToFloat());
    tensors.body.i32(io.getZeroPoint());
    tensors.body.zeros(8);
  }
  sections.push_back(std::move(tensors));

  // CPU code and steps, format minor 1 (docs/model-format.md 9, 10). The
  // copies use the arena region, the one knpu.cpu_step calls "arena".
  bool cpu = !steps.empty() || !codes.empty();
  if (cpu) {
    if (steps.empty() || codes.size() != 1)
      return module.emitError("CPU code needs one knpu.cpu_code and at "
                              "least one knpu.cpu_step");
    RegionOp arena;
    for (RegionOp r : regions) {
      if (r.getRole() != "arena")
        continue;
      if (arena)
        return r.emitOpError("CPU steps need a single arena region");
      arena = r;
    }
    if (!arena)
      return steps.front().emitOpError("needs an arena region");

    Section code{kCpuCode, {}};
    for (const APInt &v : codes.front().getCode().getValues<APInt>())
      code.body.u8(static_cast<uint8_t>(v.getZExtValue()));
    sections.push_back(std::move(code));

    Section stepTable{kCpuSteps, {}};
    stepTable.body.u32(steps.size());
    stepTable.body.zeros(12);
    Bytes names;
    for (CpuStepOp s : steps) {
      StringRef name = s.getSymbol();
      if (!isCIdentifier(name))
        return s.emitOpError("symbol must be a C identifier");
      StringRef kind = s.getKind();
      bool copy = kind != "call";
      int64_t offset = copy ? *s.getOffset() : 0;
      int64_t size = copy ? *s.getSize() : 0;
      if (offset + size > static_cast<int64_t>(arena.getSize()))
        return s.emitOpError("copies outside the arena");
      stepTable.body.u8(kind == "copy_in" ? 0 : kind == "call" ? 1 : 2);
      stepTable.body.u8(copy ? arena.getIndex() : 0);
      stepTable.body.u16(0);
      stepTable.body.u32(names.size());
      stepTable.body.u32(offset);
      stepTable.body.u32(size);
      for (char c : name)
        names.u8(c);
      names.u8(0);
    }
    llvm::append_range(stepTable.body.data, names.data);
    sections.push_back(std::move(stepTable));
  }

  Bytes file;
  file.u8('K'), file.u8('N'), file.u8('P'), file.u8('U');
  file.u16(0); // format_major
  file.u16(cpu ? 1 : 0); // format_minor
  file.u32(sramSize);
  file.u16(KNPU_VERSION_MAJOR);
  file.u16(sections.size());
  file.u32(32); // section_table
  file.zeros(12);
  size_t table = file.size();
  file.zeros(16 * sections.size());
  for (auto [i, s] : llvm::enumerate(sections)) {
    file.align16();
    size_t at = table + 16 * i;
    file.put32(at, s.kind);
    file.put32(at + 4, file.size());
    file.put32(at + 8, s.body.size());
    llvm::append_range(file.data, s.body.data);
  }
  if (file.size() > 0xFFFFFFFFull)
    return module.emitError("model.knpu would exceed 4 GiB");
  os.write(reinterpret_cast<const char *>(file.data.data()), file.size());
  return success();
}

void mlir::knpu::registerToModelFileTranslation() {
  TranslateFromMLIRRegistration reg(
      "knpu-to-model", "write model.knpu from scheduled knpu commands",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        auto module = dyn_cast<ModuleOp>(op);
        if (!module)
          return op->emitError("expected a module");
        return writeModelFile(module, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<KnpuDialect, func::FuncDialect>();
      });
}
