/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_GPU_MOSAIC_GPU_TD_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_GPU_MOSAIC_GPU_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def MosaicGPU_Dialect : Dialect {
  let name = "mosaic_gpu";
  let cppNamespace = "::mosaic_gpu";
  let useDefaultTypePrinterParser = 1;
}

class MosaicGPU_Type<string name, string mnemonic_, list<Trait> traits = []>
    : TypeDef<MosaicGPU_Dialect, name, traits> {
  let mnemonic = mnemonic_;
}

def MosaicGPU_Barrier : MosaicGPU_Type<"Barrier", "barrier", [MemRefElementTypeInterface]> {
  let summary = "barrier";
  let description = "A barrier to use for synchronizing threads";
}

def MosaicGPU_InitializeBarrierOp : Op<MosaicGPU_Dialect, "initialize_barrier",
                                      []> {
  let summary = "Initializes a memref of barriers";
  let description = [{
    Initializes a memref of barriers each meant to synchronize exactly
    `arrival_count` threads.
  }];

  let arguments = (ins ConfinedAttr<I64Attr, [IntPositive]>:$arrival_count);
  let results = (outs MemRefOf<[MosaicGPU_Barrier]>:$barriers_ref);

  let assemblyFormat = [{
    $arrival_count attr-dict `:` type($barriers_ref)
  }];
}

def MosaicGPU_FragmentedLayout :
  I32EnumAttr<"FragmentedLayout", "The layout of a FragmentedArray", [

    // FragmentedArrays in this layout are always the result of a splat. Each
    // thread in the warpgroup has a single copy of the value, regardless of
    // the shape of the FragmentedArray. This makes it trivial to broadcast,
    // reshape and do elementwise operations with all other layouts.
    I32EnumAttrCase<"WGSplatFragLayout", 0>,

    // Convert the array to 1D and then shard across threads.
    I32EnumAttrCase<"WGStridedFragLayout", 1>,

    // [m, n] matrix, where m % 64 == 0 == n % 8.
    I32EnumAttrCase<"WGMMAFragLayout", 2>,

    // [m] vector, where m % 64 == 0.
    I32EnumAttrCase<"WGMMARowFragLayout", 3>
  ]> {
  let cppNamespace = "::mosaic_gpu";
  let genSpecializedAttr = 0;
}

def MosaicGPU_FragmentedLayoutAttr : EnumAttr<
  MosaicGPU_Dialect, MosaicGPU_FragmentedLayout, "fragmented_layout"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_GPU_MOSAIC_GPU_TD_
