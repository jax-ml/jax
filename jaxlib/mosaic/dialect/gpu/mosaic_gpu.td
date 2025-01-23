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

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
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
  let useDefaultAttributePrinterParser = 1;
}

class MosaicGPU_Type<string name, string mnemonic_, list<Trait> traits = []>
    : TypeDef<MosaicGPU_Dialect, name, traits> {
  let mnemonic = mnemonic_;
}

class MosaicGPU_Attr<string name, string mnemonic_>
    : AttrDef<MosaicGPU_Dialect, name> {
  let mnemonic = mnemonic_;
}

def MosaicGPU_Barrier : MosaicGPU_Type<"Barrier", "barrier", [MemRefElementTypeInterface]> {
  let summary = "barrier";
  let description = "A barrier to use for synchronizing threads";
}

def LLVM_PointerShared : LLVM_PointerInAddressSpace<3>;

def MosaicGPU_InitializeBarrierOp : Op<MosaicGPU_Dialect, "initialize_barrier",
                                      []> {
  let summary = "Initializes a memref of barriers";
  let description = [{
    Initializes a memref of barriers each meant to synchronize exactly
    `arrival_count` threads.

    The base pointer of the result memref corresponds to `base_pointer`, which
    must be a pointer to a shared memory location.
  }];

  let arguments = (ins
    LLVM_PointerShared:$base_pointer,
    ConfinedAttr<I64Attr, [IntPositive]>:$arrival_count);
  let results = (outs MemRefOf<[MosaicGPU_Barrier]>:$barriers_ref);

  let assemblyFormat = [{
    $base_pointer $arrival_count attr-dict `:` type($barriers_ref)
  }];
}


def MosaicGPU_WGStridedFragLayout : AttrDef<MosaicGPU_Dialect, "WGStridedFragLayout", []> {
  let summary = "Annotates an array that can be collapsed to 1D and sharded across threads.";
  let description = [{
    This layout is typically used when working with pointwise operations, or
    other operations with trivial data dependency patterns.

    The layout holds the shape of the nD array it is meant to annotate, and a
    vector size representing the number of contiguous elements sharded to each
    thread.
  }];

  let parameters = (ins "::mlir::ArrayAttr":$shape, "int":$vector_size);
  let mnemonic = "WGStridedFragLayout";
  let assemblyFormat = "`<` $shape`,` $vector_size `>`";
}


def MosaicGPU_WGSplatFragLayout : AttrDef<MosaicGPU_Dialect, "WGSplatFragLayout", []> {
  let summary = "Annotates an array that is the result of a splat.";
  let description = [{
    This layout is used to handle splat values. In this case, each thread in
    the warpgroup gets a single copy of the value.

    The layout holds the shape that the initial scalar is splatted to.
  }];

  let parameters = (ins "::mlir::ArrayAttr":$shape);
  let mnemonic = "WGSplatFragLayout";
  let assemblyFormat = "`<` $shape `>`";
}

def MosaicGPU_WGMMAFragLayout : AttrDef<MosaicGPU_Dialect, "WGMMAFragLayout", []> {
  let summary = "2D array that can be tiled by supported WGMMA shapes.";
  let description = [{
    This layout annotates arrays that are fragmented across all threads in a
    warpgroup that is executing a WGMMA operation. The shape of the array is
    (m, n) where:
      - m % 64 == 0
      - n %  8 == 0
  }];

  let mnemonic = "WGMMAFragLayout";
  let assemblyFormat = "";
}

def MosaicGPU_WGMMARowFragLayout : AttrDef<MosaicGPU_Dialect, "WGMMARowFragLayout", []> {
  let summary = "1D array that is a row that can be tiled by supported WGMMA shapes.";
  let description = [{
    This layout is used to handle rows that are fragmented across all threads
    in a warpgroup that is executing a WGMMA operation. The length of the array
    must be divisible by 64.
  }];

  let mnemonic = "WGMMARowFragLayout";
  let assemblyFormat = "";
}

// Note: This duplicates the Dimension enum in mlir/Dialect/GPU/IR/GPUOps.td
// but it was not possible to reuse that definition. Including that file
// pulls in ops definitions that we don't want and they fail to compile.
def MosaicGPU_Dimension : I32EnumAttr<"Dimension",
    "a dimension, either 'x', 'y', or 'z'",
    [
      I32EnumAttrCase<"x", 0>,
      I32EnumAttrCase<"y", 1>,
      I32EnumAttrCase<"z", 2>
    ]>{
  let cppNamespace = "::mosaic_gpu";
}

def MosaicGPU_SwizzlingMode : I32EnumAttr<"SwizzlingMode",
    "What swizzling to use for a memory access.",
    [
      I32EnumAttrCase<"kNoSwizzle", 16>,
      I32EnumAttrCase<"k32ByteSwizzle", 32>,
      I32EnumAttrCase<"k64ByteSwizzle", 64>,
      I32EnumAttrCase<"k128ByteSwizzle", 128>
    ]>{
  let cppNamespace = "::mosaic_gpu";
}

def TileTransformAttr : MosaicGPU_Attr<"TileTransform", "tile"> {
  let parameters = (ins Variadic<I64>:$tiling);
  let summary = "Tiles a suffix of memref dimensions.";
  let description = [{
    For example, given a memref of shape (5, 128, 128) and a tiling of (64, 32),
    the shape of the result will be (5, 2, 4, 64, 32). The shape always ends
    with the tile shape, and the size of tiled dimensions is divided by the tile
    size. This is especially useful for swizzled WGMMA, which expect tiled
    layouts in shared memory.

    Each tiled dimension must have a size that is either smaller than the
    corresponding tile size or a multiple of the tile size.
  }];
  let assemblyFormat = "`<` $tiling `>`";
}

def TransposeTransformAttr : MosaicGPU_Attr<"TransposeTransform", "transpose"> {
  let parameters = (ins Variadic<I64>:$permutation);
  let summary = "Specifies how to transpose a memref.";
  let assemblyFormat = "`<` $permutation `>`";
}

def GlobalMemory : Resource<"::mosaic_gpu::GlobalMemory">;

def MosaicGPU_AsyncLoadOp : Op<MosaicGPU_Dialect, "async_load",
      [AttrSizedOperandSegments, MemoryEffects<[MemRead<GlobalMemory>]>]> {
  let summary = "Schedules an async load of a MemRef from GMEM to SMEM";
  let description = [{
    Schedules an async copy of the contents of the `source` MemRef in GMEM to
    the `destination` MemRef in SMEM. The `destination` MemRef in SMEM must be
    contiguous.

    If `arrive` is true, the `arrive.expect-tx(expect_count)` operation will be
    executed on the provided `barrier` before the copy is scheduled. Upon
    completion of the copy, the `complete-tx(complete-count)` operation will
    always be executed on the provided `barrier`.

    The `indices` and `slice_lengths` inputs define what slice of the GMEM
    `source` corresponds to the SMEM `destination`. Both `indices` and
    `slice_lengths` must have a length equal to the rank of the `source`. The
    values in `indices` are the starting indices of each dimension and the
    values in `slice_lengths` are the lengths. Providing -1 in `slice_lengths`
    indicates that the slice length is 1 and that the corresponding dimension
    should be collapsed and does not appear in the `destination` MemRef.

    Additional `transforms` may be provided to control how the `source` data is
    mapped to the `destination`. The transformations will be composed in the
    order they are provided. The `swizzle` attribute controls what swizzling
    is applied to the data after it is transformed, before it is finally written
    to SMEM. The transformed data is written in row-major order to the
    contiguous SMEM `destination`. The untransformed `source` data does not need
    to be contiguous, except for the last dimension, which needs to be
    contiguous and the minor-most dimension.

    The `collective` attribute can be provided to use TMA multicast to more
    efficiently load the GMEM data in cases where multiple thread blocks are
    grouped together in a cluster and need to load the same data. Each block in
    a cluster will first load a slice from GMEM to SMEM and then the slices will
    be multicast to all other blocks in the cluster. In this way TMA multicast
    guarnatees L2 cache hits. The `collective` attribute is the list of
    cluster dimensions along which to partition the input data loads.

    The `predicate` input should be set to `true` by a single thread in the
    warpgroup so that it schedules the load operation. All other threads in the
    warpgroup should set the `predicate` to `false`.
  }];

  let arguments = (ins
    MemRefOf<[AnyType]>:$source,
    MemRefOf<[AnyType]>:$destination,
    MemRefRankOf<[MosaicGPU_Barrier], [0]>:$barrier,
    Variadic<I32>:$indices,
    PtxPredicate:$predicate,

    // Attributes
    DenseI64ArrayAttr:$slice_lengths,
    TypedArrayAttrBase<AnyAttrOf<[TileTransformAttr, TransposeTransformAttr]>, "transforms">:$transforms,
    DefaultValuedAttr<MosaicGPU_SwizzlingMode, "SwizzlingMode::kNoSwizzle">:$swizzle,
    DefaultValuedAttr<BoolAttr, "true" >:$arrive,
    TypedArrayAttrBase<MosaicGPU_Dimension, "dimensions">:$collective
  );

  let assemblyFormat = [{
    `source` `(` $source `:` type($source) `)`
    `destination` `(` $destination `:` type($destination) `)`
    `barrier` `(` $barrier `:` type($barrier) `)`
    `indices` `(` $indices `)`
    `predicate` `(` $predicate `)`
    attr-dict
  }];

  let hasVerifier = 1;
}

def MosaicGPU_AsyncStoreOp : Op<MosaicGPU_Dialect, "async_store",
      [AttrSizedOperandSegments, MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Schedules an async store of a MemRef from SMEM to GMEM";
  let description = [{
    Schedules an async store of the contents of the `source` MemRef in SMEM to
    the `destination` MemRef in GMEM. The `source` MemRef in SMEM must be
    contiguous.

    The `indices` and `slice_lengths` inputs define what slice of the GMEM
    `destination` corresponds to the SMEM `source`. Both `indices` and
    `slice_lengths` must have a length equal to the rank of the `destination`.
    The values in `indices` are the starting indices of each dimension and the
    values in `slice_lengths` are the lengths. Providing -1 in `slice_lengths`
    indicates that this dimension is collapsed in the `source` and needs to be
    expanded to a slice of size 1 in the `destination`.

    Additional `transforms` may be provided to control how the `destination`
    data in GMEM is mapped to the `source` data in SMEM. The transformations
    will be composed in the order they are provided. The `swizzle` attribute
    is the swizzling mode of the `source` data in SMEM. The `source` SMEM data
    is contiguous and the transformed data is written to the `destination` GMEM
    which does not need to be contiguous.

    The `predicate` input should be set to `true` by a single thread in the
    warpgroup so that it schedules the store operation. All other threads in the
    warpgroup should set the `predicate` to `false`.
  }];

  let arguments = (ins
    MemRefOf<[AnyType]>:$source,
    MemRefOf<[AnyType]>:$destination,
    Variadic<I32>:$indices,
    PtxPredicate:$predicate,

    // Attributes
    DenseI64ArrayAttr:$slice_lengths,
    TypedArrayAttrBase<AnyAttrOf<[TileTransformAttr, TransposeTransformAttr]>, "transforms">:$transforms,
    DefaultValuedAttr<MosaicGPU_SwizzlingMode, "SwizzlingMode::kNoSwizzle">:$swizzle
  );

  let assemblyFormat = [{
    `source` `(` $source `:` type($source) `)`
    `destination` `(` $destination `:` type($destination) `)`
    `indices` `(` $indices `)`
    `predicate` `(` $predicate `)`
    attr-dict
  }];

  let hasVerifier = 1;
}

def MosaicGPU_WGMMASupportedType : AnyTypeOf<[F16, BF16, F32],
    "A type supported by the WGMMA operation">;

def MosaicGPU_WGMMALayout :
  I32EnumAttr<"WGMMALayout", "The layout of the tiles of a WGMMA operation", [
    I32EnumAttrCase<"RowMajor", 0>,
    I32EnumAttrCase<"ColumnMajor", 1>
  ]> {
  let cppNamespace = "::mosaic_gpu";
}

def MosaicGPU_WGMMAOp : Op<MosaicGPU_Dialect, "wgmma", [InferTypeOpInterface]> {
  let summary = "Multiply two matrices asyncronously using warpgroup level matrix multiply operations.";
  let description = [{
    Schedules WGMMA operations that perform the following matrix multiply and
    accumulate:

      accumulator = a * b + accumulator

    This operation supports larger inputs than the PTX-level WGMMA operation
    and will schedule as many PTX-level WGMMA operations as needed to
    accomplish the calculation. The `b` matrix, and optionally `a`, needs to be
    provided in a 4-dimensional form, where the two minor-most dimensions
    express the tile (group) size and the two major-most dimensions represent
    the total number of tiles in each direction.

    The inputs should have the following shapes:
      - If `a` is in shared memory:
        - a: [groups_m, groups_k, 64,     k]
      - If `a` is in registers:
        - a: [groups_m * 64, groups_k * k]
      - b: [groups_k, groups_n,  k, k]
      - accumulator: [groups_m * 64, groups_n * k]
    Where:
      - `k == swizzle/element_bytediwth` (for `kNoSwizzle`, `swizzle` is 16.)

    The output has an identical shape and type as the input accumulator.

    The `accumulator` is always in registers and `b` is always in shared memory.
    The last two dimensions of any input in shared memory may be physically
    transposed in memory. This is inferred from the strides of the provided
    memrefs. `a` and `b` must have the same element type and when `a` is in
    registers only F16 or BF16 are supported.

    The `accumulator` must be a vector with a FragmentedLayout. The WGMMA
    operation will be executed in the async proxy and any inputs in
    registers need to be synchronized with a memory fence.

    Usually `a` is read from shared memory if it is used directly in the WGMMA
    operation. If `a` needs to be transfromed before it is used in the WGMMA
    operation, it may be more convenient to read it directly form registers.
    This avoids the need to store the data and wait for a fence.
  }];

  let arguments = (ins
    VectorOfAnyRankOf<[MosaicGPU_WGMMASupportedType]>:$accumulator,
    AnyTypeOf<[
      MemRefOf<[MosaicGPU_WGMMASupportedType]>,
      VectorOfAnyRankOf<[MosaicGPU_WGMMASupportedType]>]>:$a,
    MemRefOf<[MosaicGPU_WGMMASupportedType]>:$b,

    // Attributes
    DefaultValuedAttr<MosaicGPU_SwizzlingMode, "SwizzlingMode::k128ByteSwizzle">:$swizzle
  );
  let results = (outs VectorOfAnyRankOf<[MosaicGPU_WGMMASupportedType]>);

  let assemblyFormat = [{
    `accumulator` `(` $accumulator `:` type($accumulator) `)`
    `a` `(` $a `:` type($a) `)`
    `b` `(` $b `:` type($b) `)`
    attr-dict
    `->` type(results)
  }];

  let extraClassDeclaration = [{
    static llvm::LogicalResult inferReturnTypes(
        mlir::MLIRContext *,
        std::optional<mlir::Location> location,
        mlir::ValueRange operands,
        mlir::DictionaryAttr attributes,
        mlir::OpaqueProperties properties,
        mlir::RegionRange regions,
        llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
      inferredReturnTypes.assign({operands[0].getType()});
      return ::mlir::success();
    }
  }];

  let hasVerifier = 1;
}

#endif // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_GPU_MOSAIC_GPU_TD_
