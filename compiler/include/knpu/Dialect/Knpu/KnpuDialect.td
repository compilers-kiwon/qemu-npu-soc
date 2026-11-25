// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KNPU_DIALECT_TD
#define KNPU_DIALECT_TD

include "mlir/IR/OpBase.td"

def Knpu_Dialect : Dialect {
  let name = "knpu";
  let summary = "kn1-npu graph and command operations";
  let description = [{
    The dialect has two levels (docs/design.md 9.4).

    Graph operations (`knpu.conv2d`, `knpu.avgpool`, `knpu.add`,
    `knpu.cpu`) take and return tensor values. The `knpu-import` pass builds
    them from the functions that lcc marked with `knpu.op` / `knpu.args`
    (`#pragma knpu`, docs/design.md 9.3).

    Command operations (`knpu.dma`, `knpu.conv`, `knpu.pool`,
    `knpu.elementwise`, `knpu.dma_wait`, `knpu.kernel_wait`, `knpu.stop`)
    follow the commands of the NPU specification (docs/npu-spec.md) almost
    one to one (`knpu.elementwise` is OP_ADD); the values of the SET commands
    are attributes. `knpu.region`, `knpu.io`, `knpu.cpu_step` and
    `knpu.cpu_code` describe the region table, the model inputs and outputs,
    the CPU steps and the CPU code of `model.knpu` (docs/model-format.md).
  }];
  let cppNamespace = "::mlir::knpu";
}

class Knpu_Op<string mnemonic, list<Trait> traits = []>
    : Op<Knpu_Dialect, mnemonic, traits>;

#endif // KNPU_DIALECT_TD
