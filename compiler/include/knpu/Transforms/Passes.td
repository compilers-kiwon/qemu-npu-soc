// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KNPU_TRANSFORMS_PASSES_TD
#define KNPU_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def KnpuImport : Pass<"knpu-import", "::mlir::ModuleOp"> {
  let summary = "Turn the functions lcc marked with #pragma knpu into knpu "
                "graph operations";
  let description = [{
    Reads the `knpu.op` / `knpu.args` attributes that lcc puts on the layer
    functions and on the model function (docs/design.md 9.3) and builds
    `func.func @knpu_model(tensor) -> tensor` with one operation per layer
    call: `knpu.conv2d` for conv2d and fc (a 1x1 convolution),
    `knpu.avgpool`, `knpu.add`, `tensor.reshape` and `knpu.cpu`. Layers are
    connected by array name, so that one tensor may feed several layers (a
    residual connection). Constant globals become dense constants; the
    requantization table is the (multiplier, shift) pair of every output
    channel. The converted NPU layer functions, the `knpu.*` attributes and
    the globals nothing uses any more are removed. The functions of cpu
    layers stay, with external linkage, for the CPU code; they may only
    follow the last NPU layer.
  }];
  let dependentDialects = [
    "::mlir::knpu::KnpuDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def KnpuTile : Pass<"knpu-tile", "::mlir::ModuleOp"> {
  let summary = "Split layers into output-row and output-channel tiles that "
                "fit in SRAM";
  let description = [{
    A layer whose constants, inputs and output fit in `sram-size` stays
    whole. Otherwise its output rows are split into tiles so that the
    constants and two buffers of each input and of the output (double
    buffering) fit, or one of each with `overlap=false`; every tile is a
    `knpu.conv2d`, `knpu.avgpool` or `knpu.add` on `tensor.extract_slice`s
    of the inputs (an add reads the same rows of both) with its own padding,
    put together with `tensor.insert_slice`. A convolution that does not
    fit even with one-row tiles, such as a fully connected layer with large
    weights, is split into output-channel tiles with the most channels: each
    takes the channel range of the weights, bias and requantization table,
    whose two buffers then count as well, and writes those channels of the
    output. The tiles read the whole input if it fits beside them, and are
    output-row tiles as well otherwise. A layer that does not fit even so
    is an error.
  }];
  let options = [
    Option<"sramSize", "sram-size", "int64_t", "65536",
           "SRAM size of the target NPU in bytes">,
    Option<"overlap", "overlap", "bool", "true",
           "Size tiles for double-buffered inputs and outputs">,
  ];
  let dependentDialects = ["::mlir::tensor::TensorDialect"];
}

def KnpuSchedule : Pass<"knpu-schedule", "::mlir::ModuleOp"> {
  let summary = "Place tensors in the arena and SRAM and lower to commands";
  let description = [{
    Places the constants in region 0 and the model inputs and outputs, the
    tensors exchanged with cpu layers and the activations that do not stay
    in SRAM in the arena (region 1, docs/design.md D40), lays out SRAM by
    the lifetimes of its buffers and lowers every layer to command
    operations; an add reads two tensors and writes a third, never in place.
    With `overlap` (the default) an activation stays in SRAM until the last
    layer that reads it, or until the first one while it is stored to the
    arena for the others; the constants of the next layer that has
    constants are loaded while the current one runs, and the next tile's
    input is loaded ahead of the current tile's operation. Where SRAM is
    short these choices are given up one level at a time, first those that
    then move the fewest bytes without overlap. `placement=largest` gives
    the rule before that instead, which knpu-compile compares with: an
    activation stays until the last layer that reads it or goes through the
    arena, and the largest choice is given up first, for good.
    Output-channel tiles of the same output rows read their input once,
    load the next tile's constants ahead and write their outputs to the
    arena with 2-D DMAs. `overlap=false` gives the serial schedule to
    compare with: every layer and tile goes to the arena and back, one
    buffer each. Adds the `knpu.region`, `knpu.io` and `knpu.cpu_step`
    operations and the `knpu.sram_size` module attribute. No wait commands
    are inserted (knpu-insert-waits).
  }];
  let options = [
    Option<"sramSize", "sram-size", "int64_t", "65536",
           "SRAM size of the target NPU in bytes">,
    Option<"overlap", "overlap", "bool", "true",
           "Keep activations in SRAM and load ahead (false: serial)">,
    Option<"placement", "placement", "::mlir::knpu::Placement",
           "::mlir::knpu::Placement::Cost",
           "Which choice to give up first where SRAM is short",
           [{::llvm::cl::values(
             clEnumValN(::mlir::knpu::Placement::Cost, "cost",
                        "the one that then moves the fewest bytes"),
             clEnumValN(::mlir::knpu::Placement::Largest, "largest",
                        "the largest (the rule before M3.6)"))}]>,
  ];
}

def KnpuInsertWaits : Pass<"knpu-insert-waits", "::mlir::ModuleOp"> {
  let summary = "Insert the wait commands the queue rules require";
  let description = [{
    Follows the asynchronous queues of docs/npu-spec.md 9 (depth 2, implicit
    wait of the oldest job, conflicts on overlapping SRAM with the other
    queue) through the command stream and inserts `knpu.dma_wait` or
    `knpu.kernel_wait` only before commands that would conflict, waiting for
    no more jobs than needed.
  }];
}

def KnpuEstimate : Pass<"knpu-estimate", "::mlir::ModuleOp"> {
  let summary = "Estimate the cycles of the command stream";
  let description = [{
    Follows the command stream through the queue model of docs/npu-spec.md
    9 (docs/design.md 9.1): the command processor issues the commands in
    order and waits only when a queue is full or at a wait command; each
    queue runs its jobs one after another. A DMA takes its bytes divided by
    `dma-bytes-per-cycle`, a convolution its nominal MACs divided by
    `macs-per-cycle`, an average pooling its window elements
    (OH*OW*C*KH*KW) divided by `pool-elements-per-cycle` and an addition
    its output elements (H*W*C) divided by `add-elements-per-cycle`, rounded
    up. The rates are assumptions for comparing schedules, not those of a
    chip. Every DMA and operation command gets `knpu.cycles` (start, end)
    and the model function `knpu.total_cycles`, the cycle the job ends at.
  }];
  let options = [
    Option<"macsPerCycle", "macs-per-cycle", "int64_t", "256",
           "Multiply-accumulates per cycle">,
    Option<"dmaBytesPerCycle", "dma-bytes-per-cycle", "int64_t", "16",
           "Bytes a DMA moves per cycle">,
    Option<"poolElementsPerCycle", "pool-elements-per-cycle", "int64_t",
           "256", "Window elements an average pooling adds per cycle">,
    Option<"addElementsPerCycle", "add-elements-per-cycle", "int64_t",
           "256", "Output elements an addition computes per cycle">,
  ];
}

#endif // KNPU_TRANSFORMS_PASSES_TD
