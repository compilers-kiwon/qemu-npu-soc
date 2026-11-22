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
    `func.func @knpu_model(tensor) -> tensor` with one `knpu.conv2d` per
    layer call. Constant globals become dense constants; the requantization
    table is the (multiplier, shift) pair of every output channel. The
    converted functions, the `knpu.*` attributes and the globals nothing uses
    any more are removed.
  }];
  let dependentDialects = [
    "::mlir::knpu::KnpuDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
}

def KnpuTile : Pass<"knpu-tile", "::mlir::ModuleOp"> {
  let summary = "Split convolutions into output-row tiles that fit in SRAM";
  let description = [{
    A layer whose constants, input and output fit in `sram-size` stays whole.
    Otherwise its output rows are split into tiles so that the constants and
    two input and two output buffers (double buffering) fit; every tile is a
    `knpu.conv2d` on a `tensor.extract_slice` of the input with its own
    padding, put together with `tensor.insert_slice`.
  }];
  let options = [
    Option<"sramSize", "sram-size", "int64_t", "65536",
           "SRAM size of the target NPU in bytes">,
  ];
  let dependentDialects = ["::mlir::tensor::TensorDialect"];
}

def KnpuSchedule : Pass<"knpu-schedule", "::mlir::ModuleOp"> {
  let summary = "Place tensors in the arena and SRAM and lower to commands";
  let description = [{
    Places the model inputs and outputs in the arena (region 1) and the
    constants in region 0 (docs/design.md D40), then lowers every layer to
    command operations: a DMA of its constants, the input DMA of the next
    tile ahead of the current tile's convolution, and the output DMA. Adds
    the `knpu.region` and `knpu.io` operations and the `knpu.sram_size`
    module attribute. No wait commands are inserted (knpu-insert-waits).
  }];
  let options = [
    Option<"sramSize", "sram-size", "int64_t", "65536",
           "SRAM size of the target NPU in bytes">,
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

#endif // KNPU_TRANSFORMS_PASSES_TD
