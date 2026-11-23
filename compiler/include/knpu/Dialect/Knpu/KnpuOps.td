// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KNPU_OPS_TD
#define KNPU_OPS_TD

include "knpu/Dialect/Knpu/KnpuDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Graph operations
//===----------------------------------------------------------------------===//

def Knpu_Conv2DOp : Knpu_Op<"conv2d", [Pure]> {
  let summary = "int8 convolution with per-channel requantization";
  let description = [{
    `ConvPerChannel` of TFLite Micro (docs/npu-spec.md 8.2): input 1xHxWxC
    (NHWC), weight OCxKHxKWxIC (OHWI), bias OC, requantization table OCx2
    (multiplier, shift per output channel), output 1xOHxOWxOC.

    `stride` is (SH, SW), `pad` is (top, left, bottom, right), `clamp` is
    (min, max) and `round` is "double" or "single". The shapes satisfy the
    window conditions of docs/npu-spec.md 6.4.

    A fully connected layer is a 1x1 convolution of a 1x1x1xIC input
    (docs/design.md D7).
  }];
  let arguments = (ins
    TensorRankOf<[I8], [4]>:$input,
    TensorRankOf<[I8], [4]>:$weight,
    TensorRankOf<[I32], [1]>:$bias,
    TensorRankOf<[I32], [2]>:$rq,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$pad,
    I64Attr:$input_zp,
    I64Attr:$output_zp,
    DenseI64ArrayAttr:$clamp,
    StrAttr:$round
  );
  let results = (outs TensorRankOf<[I8], [4]>:$output);
  let assemblyFormat = [{
    $input `,` $weight `,` $bias `,` $rq attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Knpu_AvgPool2DOp : Knpu_Op<"avgpool", [Pure]> {
  let summary = "int8 average pooling";
  let description = [{
    `AveragePool` of TFLite Micro (docs/npu-spec.md 8.4): input 1xHxWxC
    (NHWC), output 1xOHxOWxC. The input and output have the same
    quantization, so the operation has no zero point or rounding mode.

    `kernel` is (KH, KW); `stride`, `pad` and `clamp` are as in
    `knpu.conv2d`. The shapes satisfy the window conditions of
    docs/npu-spec.md 6.4.
  }];
  let arguments = (ins
    TensorRankOf<[I8], [4]>:$input,
    DenseI64ArrayAttr:$kernel,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$pad,
    DenseI64ArrayAttr:$clamp
  );
  let results = (outs TensorRankOf<[I8], [4]>:$output);
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def Knpu_CpuOp : Knpu_Op<"cpu"> {
  let summary = "A layer that runs on the CPU after the NPU job";
  let description = [{
    Calls `callee`, the C function of a `#pragma knpu cpu` layer, which
    reads the global array `in_array` and writes `out_array`
    (docs/design.md 9.3). knpu-import keeps the function for the CPU code;
    the compiler knows only its inputs and outputs.
  }];
  let arguments = (ins
    TensorRankOf<[I8], [4]>:$input,
    FlatSymbolRefAttr:$callee,
    FlatSymbolRefAttr:$in_array,
    FlatSymbolRefAttr:$out_array
  );
  let results = (outs TensorRankOf<[I8], [4]>:$output);
  let assemblyFormat = [{
    $callee `(` $input `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Command operations
//===----------------------------------------------------------------------===//

def Knpu_DmaOp : Knpu_Op<"dma"> {
  let summary = "OP_DMA with its SET_DMA_* state";
  let description = [{
    `dir` is "load" (region to SRAM) or "store" (SRAM to region). The SRAM
    side is `length * count` contiguous bytes at `sram_offset`; region row r
    starts at `region_offset + r * stride` (docs/npu-spec.md 6.3).
  }];
  let arguments = (ins
    StrAttr:$dir,
    I64Attr:$region,
    I64Attr:$region_offset,
    I64Attr:$sram_offset,
    I64Attr:$length,
    I64Attr:$count,
    I64Attr:$stride
  );
  let assemblyFormat = [{
    $dir `region` $region `[` $region_offset `]` `sram` `[` $sram_offset `]`
    `length` $length `count` $count `stride` $stride attr-dict
  }];
  let hasVerifier = 1;
}

def Knpu_ConvOp : Knpu_Op<"conv"> {
  let summary = "OP_CONV with its SET state";
  let description = [{
    All `ifm`, `ofm`, `weight`, `bias` and `rq_table` values are SRAM
    offsets. Shapes are (H, W, C), `kernel` is (KH, KW); the other
    attributes are as in `knpu.conv2d`.
  }];
  let arguments = (ins
    I64Attr:$ifm,
    DenseI64ArrayAttr:$ifm_shape,
    I64Attr:$ifm_zp,
    I64Attr:$ofm,
    DenseI64ArrayAttr:$ofm_shape,
    I64Attr:$ofm_zp,
    I64Attr:$weight,
    I64Attr:$bias,
    I64Attr:$rq_table,
    DenseI64ArrayAttr:$kernel,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$pad,
    DenseI64ArrayAttr:$clamp,
    StrAttr:$round
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Knpu_PoolOp : Knpu_Op<"pool"> {
  let summary = "OP_AVGPOOL with its SET state";
  let description = [{
    `ifm` and `ofm` are SRAM offsets and the shapes (H, W, C); the other
    attributes are as in `knpu.avgpool`. OP_AVGPOOL does not use the zero
    points or ROUND_MODE (docs/npu-spec.md 8.4).
  }];
  let arguments = (ins
    I64Attr:$ifm,
    DenseI64ArrayAttr:$ifm_shape,
    I64Attr:$ofm,
    DenseI64ArrayAttr:$ofm_shape,
    DenseI64ArrayAttr:$kernel,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$pad,
    DenseI64ArrayAttr:$clamp
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Knpu_DmaWaitOp : Knpu_Op<"dma_wait"> {
  let summary = "OP_DMA_WAIT: wait until at most `pending` DMAs remain";
  let arguments = (ins I64Attr:$pending);
  let assemblyFormat = "$pending attr-dict";
  let hasVerifier = 1;
}

def Knpu_KernelWaitOp : Knpu_Op<"kernel_wait"> {
  let summary = "OP_KERNEL_WAIT: wait until at most `pending` operations remain";
  let arguments = (ins I64Attr:$pending);
  let assemblyFormat = "$pending attr-dict";
  let hasVerifier = 1;
}

def Knpu_StopOp : Knpu_Op<"stop"> {
  let summary = "OP_STOP";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Model file description
//===----------------------------------------------------------------------===//

def Knpu_RegionOp : Knpu_Op<"region"> {
  let summary = "One entry of the region table of model.knpu";
  let description = [{
    `role` is "const" or "arena" (docs/design.md D40). A constant region
    carries its contents as a 1-D i8 tensor of `size` elements.
  }];
  let arguments = (ins
    I64Attr:$index,
    StrAttr:$role,
    I64Attr:$size,
    OptionalAttr<ElementsAttr>:$content
  );
  let assemblyFormat = [{
    $index $role `size` $size (`content` $content^)? attr-dict
  }];
  let hasVerifier = 1;
}

def Knpu_IoOp : Knpu_Op<"io"> {
  let summary = "One model input or output of model.knpu";
  let arguments = (ins
    StrAttr:$direction,
    I64Attr:$region,
    I64Attr:$offset,
    DenseI64ArrayAttr:$shape,
    F32Attr:$scale,
    I64Attr:$zero_point
  );
  let assemblyFormat = [{
    $direction `region` $region `[` $offset `]` `shape` $shape
    `scale` $scale `zero_point` $zero_point attr-dict
  }];
  let hasVerifier = 1;
}

def Knpu_CpuStepOp : Knpu_Op<"cpu_step"> {
  let summary = "One step the runtime takes after the NPU job";
  let description = [{
    The `knpu.cpu_step` operations of a module, in their order, are what
    the runtime does after the NPU job (docs/design.md 9.5, 10). `kind`
    "copy_in" copies `size` bytes from the arena at `offset` into the array
    `symbol` of the CPU code, "call" calls the function `symbol`, and
    "copy_out" copies `size` bytes of the array `symbol` to the arena at
    `offset`.
  }];
  let arguments = (ins
    StrAttr:$kind,
    FlatSymbolRefAttr:$symbol,
    OptionalAttr<I64Attr>:$offset,
    OptionalAttr<I64Attr>:$size
  );
  let assemblyFormat = [{
    $kind $symbol (`arena` `[` $offset^ `]` `size` $size)? attr-dict
  }];
  let hasVerifier = 1;
}

def Knpu_CpuCodeOp : Knpu_Op<"cpu_code"> {
  let summary = "The CPU code of model.knpu";
  let description = [{
    The bytes of the AArch64 shared object that knpu-compile builds from
    the functions of the cpu layers (docs/design.md 9.4), as a 1-D i8
    tensor. It exports the functions and arrays the `knpu.cpu_step`
    operations name (docs/model-format.md 9).
  }];
  let arguments = (ins ElementsAttr:$code);
  let assemblyFormat = "$code attr-dict";
  let hasVerifier = 1;
}

#endif // KNPU_OPS_TD
