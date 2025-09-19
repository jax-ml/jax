/* Copyright 2025 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_DIALECT_TPU_TPU_ENUMS_TD_
#define JAXLIB_MOSAIC_DIALECT_TPU_TPU_ENUMS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def TPU_CoreType : I32EnumAttr<"CoreType", "Core type", [
  I32EnumAttrCase<"kTc", 0, "tc">,
  I32EnumAttrCase<"kScScalarSubcore", 1, "sc_scalar_subcore">,
  I32EnumAttrCase<"kScVectorSubcore", 2, "sc_vector_subcore">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_PipelineMode : I32EnumAttr<"PipelineMode", "Pipeline mode", [
  I32EnumAttrCase<"kSynchronous", 1, "synchronous">,
  I32EnumAttrCase<"kDoubleBuffered", 2, "double_buffered">
  ]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_DimensionSemantics : I32EnumAttr<"DimensionSemantics", "Dimension semantics", [
  I32EnumAttrCase<"parallel", 0>,
  I32EnumAttrCase<"arbitrary", 1>,
  I32EnumAttrCase<"core_parallel", 2>,
  I32EnumAttrCase<"subcore_parallel", 3>
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_ContractPrecision : I32EnumAttr<"ContractPrecision", "Contraction precision", [
  I32EnumAttrCase<"kBF16", 0, "bf16">,
  I32EnumAttrCase<"kFP32", 1, "fp32">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_PackFormat : I32EnumAttr<"PackFormat", "Pack format", [
  I32EnumAttrCase<"kCompressed", 0, "compressed">,
  I32EnumAttrCase<"kInterleaved", 1, "interleaved">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_MemorySpace : I32EnumAttr<"MemorySpace", "Memory space", [
  I32EnumAttrCase<"kAny", 4294967295, "any">,
  I32EnumAttrCase<"kVmem", 0, "vmem">,
  I32EnumAttrCase<"kSmem", 1, "smem">,
  I32EnumAttrCase<"kHbm", 2, "hbm">,
  I32EnumAttrCase<"kCmem", 3, "cmem">,
  I32EnumAttrCase<"kSemaphoreMem", 4, "semaphore_mem">,
  I32EnumAttrCase<"kVmemShared", 5, "vmem_shared">,
  I32EnumAttrCase<"kHost", 6, "host">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_ReductionKind : I32EnumAttr<"ReductionKind", "Reduction kind", [
  I32EnumAttrCase<"kSum", 0, "sum">,
  I32EnumAttrCase<"kMax", 1, "max">,
  I32EnumAttrCase<"kMin", 2, "min">,
  I32EnumAttrCase<"kArgMax", 3, "arg_max">,
  I32EnumAttrCase<"kArgMin", 4, "arg_min">,
  I32EnumAttrCase<"kFindFirstSet", 5, "find_first_set">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_RoundingMode : I32EnumAttr<"RoundingMode", "Rounding mode", [
  I32EnumAttrCase<"kTowardsZero", 0, "towards_zero">,
  I32EnumAttrCase<"kToNearestEven", 1, "to_nearest_even">,
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

#endif  // JAXLIB_MOSAIC_DIALECT_TPU_TPU_ENUMS_TD_
