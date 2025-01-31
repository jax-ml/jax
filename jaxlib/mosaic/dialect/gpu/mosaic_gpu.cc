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

#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu.h"

#include <cstdint>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Diagnostics.h"
#include "tsl/platform/statusor.h"

// Generated definitions.
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_dialect.cc.inc"
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_enums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_attrdefs.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_types.cc.inc"
#define GET_OP_CLASSES
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_ops.cc.inc"

namespace mosaic_gpu {
namespace {

using ::mlir::FloatType;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::IntegerType;
using ::mlir::MLIRContext;
using ::mlir::Type;
using ::mlir::TypeRange;
using ::mlir::Value;
using ::mlir::ValueRange;

using Index = ::mlir::TypedValue<::mlir::IndexType>;
using Integer = ::mlir::TypedValue<::mlir::IntegerType>;

Integer ToI64(ImplicitLocOpBuilder& b, Index index) {
  return llvm::cast<Integer>(
      b.create<mlir::arith::IndexCastOp>(b.getI64Type(), index).getResult());
}

template <typename T>
Value Constant(ImplicitLocOpBuilder& b, T scalar, IntegerType type) {
  return b.create<mlir::arith::ConstantOp>(
      type, mlir::IntegerAttr::get(type, scalar));
}

template <typename T>
Value Constant(ImplicitLocOpBuilder& b, T scalar, FloatType type) {
  return b.create<mlir::arith::ConstantOp>(type,
                                           mlir::FloatAttr::get(type, scalar));
}

// Given a range of values of the same type, produces a LLVM array that contains
// all of them in order. Returns a pointer to the start of the newly created
// array.
absl::StatusOr<Pointer> ToLLVMArray(ImplicitLocOpBuilder& b,
                                    ValueRange values) {
  if (values.empty()) {
    return absl::InvalidArgumentError("Can not pack an empty array of values.");
  }

  Type element_type = values.front().getType();

  MLIRContext* ctx = b.getContext();
  mlir::LLVM::LLVMPointerType pointer_type =
      mlir::LLVM::LLVMPointerType::get(ctx);
  Pointer array_pointer = b.create<mlir::LLVM::AllocaOp>(
      pointer_type, element_type, Constant(b, values.size(), b.getI64Type()));

  for (auto [i, value] : llvm::enumerate(values)) {
    if (value.getType() != element_type) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected all values to have the same type, but got ",
          MlirToString(value.getType()), " and ", MlirToString(element_type)));
    }

    auto element_pointer = llvm::cast<Pointer>(
        b.create<mlir::LLVM::GEPOp>(
             pointer_type, element_type, array_pointer,
             mlir::ArrayRef<mlir::LLVM::GEPArg>(mlir::LLVM::GEPArg(i)))
            .getResult());
    b.create<mlir::LLVM::StoreOp>(value, element_pointer);
  }

  return array_pointer;
}

// Extracts a pointer to the start of the parameter memref.
Pointer FromMemref(ImplicitLocOpBuilder& b, Memref memref) {
  Index aligned_pointer_as_index =
      b.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(memref);

  mlir::LLVM::LLVMPointerType pointer_type =
      mlir::LLVM::LLVMPointerType::get(b.getContext());

  Value alloc_pointer = b.create<mlir::LLVM::IntToPtrOp>(
      pointer_type, ToI64(b, aligned_pointer_as_index));

  Type tensor_element_type = memref.getType().getElementType();

  return mlir::cast<Pointer>(
      b.create<mlir::LLVM::GEPOp>(
           pointer_type, tensor_element_type, alloc_pointer,
           mlir::ArrayRef<mlir::LLVM::GEPArg>(
               mlir::LLVM::GEPArg(ToI64(b, aligned_pointer_as_index))))
          .getResult());
}

}  // anonymous namespace

// TODO(bchetioui): add swizzling.
absl::Status InitTmaDescriptor(mlir::OpBuilder& builder,
                               Pointer host_pointer_to_descriptor,
                               Memref gmem_ref,
                               mlir::ArrayRef<int64_t> slice_shape) {
  ImplicitLocOpBuilder b(
      mlir::NameLoc::get(builder.getStringAttr("InitTmaDescriptor")), builder);

  mlir::memref::ExtractStridedMetadataOp extract_strided_metadata_op =
      b.create<mlir::memref::ExtractStridedMetadataOp>(gmem_ref);

  Type tensor_element_type = gmem_ref.getType().getElementType();

  Pointer tensor_base_pointer = FromMemref(b, gmem_ref);

  int64_t tensor_rank = gmem_ref.getType().getRank();
  ValueRange sizes = extract_strided_metadata_op.getSizes();
  ValueRange strides = extract_strided_metadata_op.getStrides();

  if (tensor_rank != slice_shape.size()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Slice shape should have the same rank as the target tensor "
        "but got ",
        slice_shape.size(), " != ", tensor_rank));
  }

  std::vector<Value> sizes_as_i64;
  std::vector<Value> strides_as_i64;
  std::vector<Value> slice_as_i64;
  sizes_as_i64.reserve(tensor_rank);
  strides_as_i64.reserve(tensor_rank);
  slice_as_i64.reserve(tensor_rank);
  for (auto [size, stride, slice_dim] :
       llvm::zip(sizes, strides, slice_shape)) {
    sizes_as_i64.push_back(ToI64(b, llvm::cast<Index>(size)));
    strides_as_i64.push_back(ToI64(b, llvm::cast<Index>(stride)));
    slice_as_i64.push_back(Constant(b, slice_dim, b.getI64Type()));
  }

  TF_ASSIGN_OR_RETURN(Pointer sizes_array, ToLLVMArray(b, sizes_as_i64));
  TF_ASSIGN_OR_RETURN(Pointer strides_array, ToLLVMArray(b, strides_as_i64));
  TF_ASSIGN_OR_RETURN(Pointer slice_array, ToLLVMArray(b, slice_as_i64));

  IntegerType i64 = b.getI64Type();

  int64_t elem_bitwidth = tensor_element_type.getIntOrFloatBitWidth();

  if (elem_bitwidth < 8) {
    return absl::UnimplementedError("Sub-byte types are not yet supported.");
  }

  // TODO(bchetioui): connect this to runtime.
  b.create<mlir::func::CallOp>(
      kRuntimeTmaDescriptorInitializerName, TypeRange{},
      ValueRange{/*tma_desc=*/host_pointer_to_descriptor,
                 /*base_addr=*/tensor_base_pointer,
                 /*elem_bytewidth=*/Constant(b, elem_bitwidth / 8, i64),
                 /*rank=*/Constant(b, tensor_rank, i64),
                 /*sizes=*/sizes_array,
                 /*strides=*/strides_array,
                 // TODO(bchetioui): implement swizzling.
                 /*swizzle_bytes=*/Constant(b, 0, i64),
                 /*window_shape=*/slice_array});

  return absl::OkStatus();
}

void DeclareRuntimeFunctions(mlir::OpBuilder& builder) {
  MLIRContext* ctx = builder.getContext();
  mlir::LLVM::LLVMPointerType ptr = mlir::LLVM::LLVMPointerType::get(ctx);
  IntegerType i64 = builder.getI64Type();

  builder
      .create<mlir::func::FuncOp>(
          builder.getUnknownLoc(), kRuntimeTmaDescriptorInitializerName,
          builder.getFunctionType(
              TypeRange{ptr, ptr, i64, i64, ptr, ptr, i64, ptr}, TypeRange{}))
      .setVisibility(mlir::func::FuncOp::Visibility::Private);
}

bool IsContiguous(mlir::MemRefType type) {
  return type.getLayout().isIdentity() ||
         (type.hasStaticShape() && type.getNumElements() > 0 &&
          mlir::memref::isStaticShapeAndContiguousRowMajor(type));
}

namespace {
llvm::LogicalResult VerifyCommonLoadStoreOp(
    mlir::Location loc, mlir::MemRefType gmem_type, absl::string_view gmem_name,
    mlir::MemRefType smem_type, absl::string_view smem_name,
    mlir::ArrayRef<int64_t> slice_lengths, int num_indices) {
  auto error = [loc](auto... params) {
    return emitError(loc, llvm::formatv(params...));
  };

  if (!IsContiguous(smem_type)) {
    return error("The `{0}` memref must be contiguous.", smem_name);
  }
  if (gmem_type.getElementType() != smem_type.getElementType()) {
    return error(
        "The `source` and `destination` memrefs must have the same element "
        "type.");
  }
  if (absl::c_any_of(slice_lengths, [](int64_t s) { return s < -1; })) {
    return error(
        "The `slice_lengths` attribute must not contain values less than -1.");
  }
  if (gmem_type.getRank() !=
      smem_type.getRank() + absl::c_count(slice_lengths, -1)) {
    return error(
        "The rank of the `{0}` must be equal to the rank of the "
        "`{1}` plus the number of collapsed dimensions as indicated "
        "by -1 values in `slice_lengths`.",
        gmem_name, smem_name);
  }
  if (num_indices != gmem_type.getRank()) {
    return error("The size of `indices` must be equal to the rank of `{0}`.",
                 gmem_name);
  }
  if (slice_lengths.size() != gmem_type.getRank()) {
    return error(
        "The size of `slice_lengths` must be equal to the rank of `{0}`.",
        gmem_name);
  }
  return llvm::success();
}
}  // namespace

llvm::LogicalResult AsyncLoadOp::verify() {
  auto r = VerifyCommonLoadStoreOp(getLoc(), getSource().getType(), "source",
                                   getDestination().getType(), "destination",
                                   getSliceLengths(), getIndices().size());
  if (failed(r)) {
    return r;
  }

  for (int i = 0; i < getCollective().size(); ++i) {
    for (int k = i + 1; k < getCollective().size(); ++k)
      if (getCollective()[i] == getCollective()[k]) {
        return emitError(
            "The `collective` attribute must not contain duplicate "
            "dimensions.");
      }
  }

  return llvm::success();
}

llvm::LogicalResult AsyncStoreOp::verify() {
  return VerifyCommonLoadStoreOp(getLoc(), getDestination().getType(),
                                 "destination", getSource().getType(), "source",
                                 getSliceLengths(), getIndices().size());
}

namespace {
llvm::FailureOr<WGMMALayout> GetWgmmaLayout(mlir::Location loc,
                                            mlir::MemRefType type,
                                            absl::string_view name,
                                            SwizzlingMode swizzling_mode) {
  auto error = [loc](auto... params) {
    return emitError(loc, llvm::formatv(params...));
  };

  auto [strides, offset] = type.getStridesAndOffset();

  WGMMALayout layout = WGMMALayout::RowMajor;
  if (strides[3] == 1) {
    layout = WGMMALayout::RowMajor;
  } else if (strides[2] == 1) {
    layout = WGMMALayout::ColumnMajor;
  } else {
    return error(
        "At least one of the last two dimensions of `{0}` must have a "
        "stride of 1, but they do not: stride(dim 2)={1}, stride(dim 3)={2}",
        name, strides[2], strides[3]);
  }

  auto shape = type.getShape();
  if (layout == WGMMALayout::RowMajor && strides[2] != shape[3]) {
    return error(
        "When `{0}` has row-major layout, the stride of dimension 2 (={1}) "
        "must be equal to size of dimension 3 (={2})",
        shape[3], strides[2], shape[3]);
  }

  if (layout == WGMMALayout::ColumnMajor && strides[3] != shape[2]) {
    return error(
        "When `{0}` has column-major layout, the stride of dimension 3 (={1}) "
        "must be equal to size of dimension 2 (={2})",
        shape[2], strides[3], shape[2]);
  }

  if (strides[1] != shape[2] * shape[3]) {
    return error(
        "Dimension 1 ` of `{0}` must have a stride equal to size of dimension "
        "2 times size of dimension 3 (={1}), but has {2}.",
        name, shape[2] * shape[3], strides[1]);
  }

  return layout;
}

// This is the size of the M dimension in all wgmma instructions. It is fixed,
// unlike the K and N dimensions.
constexpr int kWgmmaSizeM = 64;
}  // namespace

llvm::LogicalResult WGMMAOp::verify() {
  auto error = [this](auto... params) {
    return emitOpError(llvm::formatv(params...));
  };

  auto a_shaped_type = mlir::cast<mlir::ShapedType>(getA().getType());
  mlir::Type element_type = a_shaped_type.getElementType();
  if (element_type != getB().getType().getElementType()) {
    return error("The `a` and `b` inputs must have the same element type.");
  }

  auto b_shape = getB().getType().getShape();
  if (b_shape.size() != 4) {
    return error("The `b` input must have rank 4.");
  }

  int element_bytewidth = element_type.getIntOrFloatBitWidth() / 8;
  int kn_tile = static_cast<int>(getSwizzle()) / element_bytewidth;

  int64_t groups_k = b_shape[0];
  int64_t groups_n = b_shape[1];
  int64_t k_group_size = b_shape[2];
  int64_t n_group_size = b_shape[3];

  // It might be possible to relax that requirement, in particular to allow
  // n_group_size to be smaller than kn_tile and use padding.
  if (n_group_size != kn_tile) {
    return error(
        "The n group size ({0}) must be equal to swizzle/element_bytewidth "
        "({1}).",
        n_group_size, kn_tile);
  }
  if (k_group_size != kn_tile) {
    return error(
        "The k group size ({0}) must be equal to swizzle/element_bytewidth "
        "({1}).",
        k_group_size, kn_tile);
  }

  auto b_layout = GetWgmmaLayout(getLoc(), getB().getType(), "b", getSwizzle());
  if (failed(b_layout)) {
    return b_layout;
  }

  int groups_m = 0;
  auto a_shape = a_shaped_type.getShape();
  if (auto a_memref = dyn_cast<mlir::MemRefType>(getA().getType())) {
    if (a_shape.size() != 4) {
      return error("When `a` is a memref, it must have rank 4.");
    }

    groups_m = a_shape[0];

    if (a_shape[1] != groups_k) {
      return error(
          "When `a` is a memref, dimension 1 ({0}) must be equal to groups_k "
          "which is `b`'s dimension 0 ({1}).",
          a_shape[1], groups_k);
    }

    if (a_shape[2] != kWgmmaSizeM) {
      return error(
          "When `a` is a memref, dimension 2 ({0}) must be equal to {1}.",
          a_shape[2], kWgmmaSizeM);
    }

    if (a_shape[3] != kn_tile) {
      return error(
          "When `a` is a memref, dimension 3 ({0}) must be equal to kn_tile.",
          a_shape[3]);
    }

    auto a_layout = GetWgmmaLayout(getLoc(), a_memref, "a", getSwizzle());
    if (failed(a_layout)) {
      return a_layout;
    }
    if (*a_layout == WGMMALayout::ColumnMajor &&
        getSwizzle() != SwizzlingMode::k128ByteSwizzle) {
      // Not sure what the layout is like, since the tiles aren't square.
      return error(
          "When `a` is a memref and has a column-major layout, only a swizzle "
          "of 128 bytes is currently supported, but got {0}.");
    }
  } else {
    // a is a tensor in registers.
    if (!element_type.isBF16() && !element_type.isF16()) {
      return error(
          "When `a` is a tensor in registers, it must have element type bf16 "
          "or f16.");
    }
    if (a_shape.size() != 2) {
      return error("When `a` is a tensor in registers, it must have rank 2.");
    }
    if (a_shape[0] % kWgmmaSizeM) {
      return error(
          "When `a` is a tensor in registers, dimension 0 must be a multiple "
          "of {0}, but got {1}.",
          kWgmmaSizeM, a_shape[0]);
    }

    groups_m = a_shape[0] / kWgmmaSizeM;

    if (a_shape[1] != kn_tile * groups_k) {
      return error(
          "When `a` is a tensor in registers, dimension 1 must be equal to "
          "kn_tile * groups_k ({0}*{1}), but got {2}.",
          kn_tile, groups_k, a_shape[1]);
    }
  }

  auto accShape = getAccumulator().getType().getShape();
  if (accShape.size() != 2) {
    return error("The accumulator must have rank 2.");
  }
  int expected_acc_0 = groups_m * kWgmmaSizeM;
  int expected_acc_1 = groups_n * n_group_size;
  if (accShape[0] != expected_acc_0 || accShape[1] != expected_acc_1) {
    return error(
        "Incorrect accumulator shape. Expected: [{0},{1}], but got [{2},{3}].",
        expected_acc_0, expected_acc_1, accShape[0], accShape[1]);
  }

  return llvm::success();
}

void MosaicGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_types.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_attrdefs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_ops.cc.inc"
      >();
}

}  // namespace mosaic_gpu
