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

include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinTypeInterfaces.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/CommonTypeConstraints.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/DialectBase.td"
include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OpBase.td"

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

#endif // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_GPU_MOSAIC_GPU_TD_