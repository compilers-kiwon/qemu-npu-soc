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

#endif // KNPU_OPS_TD
