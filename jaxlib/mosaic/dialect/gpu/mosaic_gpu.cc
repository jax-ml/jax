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
#include <optional>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/tsl/platform/statusor.h"

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
      mlir::arith::IndexCastOp::create(b, b.getI64Type(), index).getResult());
}

template <typename T>
Value Constant(ImplicitLocOpBuilder& b, T scalar, IntegerType type) {
  return mlir::arith::ConstantOp::create(b, type,
                                         mlir::IntegerAttr::get(type, scalar));
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
  Pointer array_pointer =
      mlir::LLVM::AllocaOp::create(b, pointer_type, element_type,
                                   Constant(b, values.size(), b.getI64Type()));

  for (auto [i, value] : llvm::enumerate(values)) {
    if (value.getType() != element_type) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected all values to have the same type, but got ",
          MlirToString(value.getType()), " and ", MlirToString(element_type)));
    }

    auto element_pointer = llvm::cast<Pointer>(
        mlir::LLVM::GEPOp::create(
            b, pointer_type, element_type, array_pointer,
            mlir::ArrayRef<mlir::LLVM::GEPArg>(mlir::LLVM::GEPArg(i)))
            .getResult());
    mlir::LLVM::StoreOp::create(b, value, element_pointer);
  }

  return array_pointer;
}

// Extracts a pointer to the start of the parameter memref.
Pointer FromMemref(ImplicitLocOpBuilder& b, Memref memref) {
  Index aligned_pointer_as_index =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(b, memref);

  mlir::LLVM::LLVMPointerType pointer_type =
      mlir::LLVM::LLVMPointerType::get(b.getContext());

  Value alloc_pointer = mlir::LLVM::IntToPtrOp::create(
      b, pointer_type, ToI64(b, aligned_pointer_as_index));

  Type tensor_element_type = memref.getType().getElementType();

  return mlir::cast<Pointer>(
      mlir::LLVM::GEPOp::create(
          b, pointer_type, tensor_element_type, alloc_pointer,
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
      mlir::memref::ExtractStridedMetadataOp::create(b, gmem_ref);

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
  mlir::func::CallOp::create(
      b, kRuntimeTmaDescriptorInitializerName, TypeRange{},
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

  mlir::func::FuncOp::create(
      builder, builder.getUnknownLoc(), kRuntimeTmaDescriptorInitializerName,
      builder.getFunctionType(TypeRange{ptr, ptr, i64, i64, ptr, ptr, i64, ptr},
                              TypeRange{}))
      .setVisibility(mlir::func::FuncOp::Visibility::Private);
}

namespace {
llvm::LogicalResult VerifyCommonLoadStoreOp(
    mlir::Operation* op, mlir::MemRefType gmem_type, std::string_view gmem_name,
    mlir::MemRefType smem_type, std::string_view smem_name,
    mlir::ArrayRef<int64_t> slice_lengths,
    mlir::Operation::operand_range indices) {
  auto error = [op](auto... params) {
    return op->emitError(llvm::formatv(params...));
  };

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
  if (indices.size() != gmem_type.getRank()) {
    return error("The size of `indices` must be equal to the rank of `{0}`.",
                 gmem_name);
  }
  if (slice_lengths.size() != gmem_type.getRank()) {
    return error(
        "The size of `slice_lengths` must be equal to the rank of `{0}`.",
        gmem_name);
  }
  int first_vector_index_dim = -1;  // -1 means no vector index.
  for (int i = 0; i < indices.size(); ++i) {
    if (auto vec_type =
            mlir::dyn_cast<mlir::VectorType>(indices[i].getType())) {
      if (first_vector_index_dim >= 0) {
        return error(
            "Only one index may be a vector but got multiple vector indices "
            "for dimensions {0} and {1}.", first_vector_index_dim, i);
      }
      first_vector_index_dim = i;
      if (vec_type.getShape()[0] != slice_lengths[i]) {
        return error(
            "The size of the vector index must be equal to the slice length "
            "but got {0} != {1}.",
            vec_type.getShape()[0], slice_lengths[i]);
      }
    }
  }
  return llvm::success();
}
}  // namespace

llvm::LogicalResult AsyncLoadOp::verify() {
  auto r =
      VerifyCommonLoadStoreOp(getOperation(), getSource().getType(), "source",
                              getDestination().getType(), "destination",
                              getSliceLengths(), getIndices());
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

llvm::LogicalResult AsyncPrefetchOp::verify() {
  if (absl::c_any_of(getSliceLengths(), [](int64_t s) { return s < -1; })) {
    return emitOpError(
        "The `slice_lengths` attribute must not contain values less than -1.");
  }
  if (getIndices().size() != getSource().getType().getRank()) {
     return emitOpError(
        "The size of `indices` must be equal to the rank of `source`.");
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
  return VerifyCommonLoadStoreOp(getOperation(), getDestination().getType(),
                                 "destination", getSource().getType(), "source",
                                 getSliceLengths(), getIndices());
}

llvm::LogicalResult WGMMAOp::inferReturnTypes(
    mlir::MLIRContext*, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
  if (operands.empty()) {
    return mlir::emitOptionalError(location, "expected non-empty operands");
  }
  inferredReturnTypes.assign({operands[0].getType()});
  return mlir::success();
}

llvm::LogicalResult WGMMAOp::verify() {
  auto error = [this](auto... params) {
    return getOperation()->emitOpError(llvm::formatv(params...));
  };

  auto a_type = mlir::cast<mlir::ShapedType>(getA().getType());
  auto b_type = getB().getType();
  auto acc_type = getAccumulator().getType();

  if (a_type.getElementType() != b_type.getElementType()) {
    return error("The `a` and `b` inputs must have the same element type.");
  }

  auto a_shape = a_type.getShape();
  auto b_shape = b_type.getShape();
  auto acc_shape = acc_type.getShape();

  int M = acc_shape[0];
  if (M != a_shape[0]) {
    return error(
        "The accumulator's first dimension {0} must be equal to the first "
        "dimensions of `a`: {1}.",
        M, a_shape[0]);
  }
  int K = a_shape[1];  // groups_k * k
  if (K != b_shape[0]) {
    return error(
        "`a`'s contracting dimension {0} must be equal to the first dimension "
        "of `b`: {1}.",
        K, b_shape[0]);
  }
  int N = b_shape[1];  // groups_n * k
  if (N != acc_shape[1]) {
    return error(
        "`b`'s non-contracting dimension {0} must be equal to the "
        "accumulator's second dimension {1}.",
        N, acc_shape[1]);
  }

  // This is the size of the M dimension in all wgmma instructions. It is fixed,
  // unlike the K and N dimensions.
  constexpr int kWgmmaSizeM = 64;
  if (M % kWgmmaSizeM != 0) {
    return error(
        "The accumulator's first dimension must be a multiple of {0}, but got "
        "{1}.",
        kWgmmaSizeM, M);
  }

  return llvm::success();
}

llvm::LogicalResult TcGen05MMAOp::verify() {
  auto error = [this](auto... params) {
    return getOperation()->emitOpError(llvm::formatv(params...));
  };

  auto a_type = getA().getType();
  auto b_type = getB().getType();
  auto acc_type = getAccumulator().getType();

  if (a_type.getElementType() != b_type.getElementType()) {
    return error("The `a` and `b` inputs must have the same element type.");
  }

  auto a_shape = a_type.getShape();
  auto b_shape = b_type.getShape();
  auto acc_shape = acc_type.getShape();

  int M = acc_shape[0];
  if (M != a_shape[0]) {
    return error(
        "The accumulator's first dimension {0} must be equal to the first "
        "dimensions of `a`: {1}.",
        M, a_shape[0]);
  }
  int K = a_shape[1];  // groups_k * k
  if (K != b_shape[0]) {
    return error(
        "`a`'s contracting dimension {0} must be equal to the first dimension "
        "of `b`: {1}.",
        K, b_shape[0]);
  }
  int N = b_shape[1];  // groups_n * k
  if (N != acc_shape[1] && !getCollective()) {
    return error(
        "`b`'s non-contracting dimension {0} must be equal to the "
        "accumulator's second dimension {1}.",
        N, acc_shape[1]);
  }
  if (N * 2 != acc_shape[1] && getCollective()) {
    return error(
        "`b`'s non-contracting dimension {0} must be half the accumulator's "
        "second dimension {1} for collective MMA.",
        N, acc_shape[1]);
  }

  // This is the size of the M dimension in all `tcgen05.mma` instructions. It
  // is fixed, unlike the K and N dimensions.
  constexpr int kTcGen05MmaMinSizeM = 32;
  if (M % kTcGen05MmaMinSizeM != 0) {
    return error(
        "The accumulator's first dimension must be a multiple of {0} but got "
        "{1}.",
        kTcGen05MmaMinSizeM, M);
  }

  mlir::Attribute tmem = TmemAttr::get(getContext());
  mlir::Attribute smem = mlir::gpu::AddressSpaceAttr::get(
      getContext(), mlir::gpu::AddressSpace::Workgroup);

  mlir::Attribute acc_mem_space = getAccumulator().getType().getMemorySpace();
  if (acc_mem_space != tmem) {
    return error("The accumulator must be in TMEM, but got {0}.",
                 acc_mem_space);
  }
  mlir::Attribute a_mem_space = getA().getType().getMemorySpace();
  if (a_mem_space != tmem && a_mem_space != smem) {
    return error("The `a` input must be in TMEM or SMEM, but got {0}.",
                 a_mem_space);
  }
  mlir::Attribute b_mem_space = getB().getType().getMemorySpace();
  if (b_mem_space != smem) {
    return error("The `b` input must be in SMEM, but got {0}.", b_mem_space);
  }

  mlir::TypedValue<mlir::MemRefType> a_scale = getAScale();
  mlir::TypedValue<mlir::MemRefType> b_scale = getBScale();
  if (static_cast<bool>(a_scale) != static_cast<bool>(b_scale)) {
    return error("Either none or both scales should be provided.");
  }

  if (a_scale) {
    mlir::Attribute a_scale_mem_space = a_scale.getType().getMemorySpace();
    if (a_scale_mem_space != tmem) {
      return error("The `a_scale` input must be in TMEM, but got {0}.",
                   a_scale_mem_space);
    }
    mlir::Attribute b_scale_mem_space = b_scale.getType().getMemorySpace();
    if (b_scale_mem_space != tmem) {
      return error("The `b_scale` input must be in TMEM, but got {0}.",
                   b_scale_mem_space);
    }
  }

  return llvm::success();
}

llvm::LogicalResult CustomPrimitiveOp::verify() {
  int num_vector_operands = 0;
  int num_smem_ref_operands = 0;
  mlir::Attribute smem = mlir::gpu::AddressSpaceAttr::get(
      getContext(), mlir::gpu::AddressSpace::Workgroup);
  for (auto operand : getOperands()) {
    if (mlir::isa<mlir::VectorType>(operand.getType())) {
      ++num_vector_operands;
    }

    if (auto ref_ty = mlir::dyn_cast<mlir::MemRefType>(operand.getType())) {
      if (ref_ty.getMemorySpace() == smem) {
        ++num_smem_ref_operands;
      }
    }
  }

  if (num_vector_operands != getInLayouts().size()) {
    return emitOpError(
        "Custom primitive must have a layout for each vector operand.");
  }

  if (num_smem_ref_operands != getInTransforms().size()) {
    return emitOpError(
        "Custom primitive must have transforms for each memref operand in "
        "smem.");
  }

  int num_vector_results = 0;
  for (auto result : getResults()) {
    if (mlir::isa<mlir::VectorType>(result.getType())) {
      ++num_vector_results;
    } else if (mlir::isa<mlir::ShapedType>(result.getType())) {
      return emitOpError(
          "Custom primitive can only return scalars or vectors.");
    }
  }

  if (num_vector_results != getOutLayouts().size()) {
    return emitOpError(
        "Custom primitive must have a layout for each vector result.");
  }

  return llvm::success();
}

llvm::LogicalResult BroadcastInDimOp::verify() {
  auto error = [this](auto... params) {
    return emitOpError(llvm::formatv(params...));
  };

  mlir::VectorType operand_type = getOperand().getType();
  mlir::VectorType result_type = getResult().getType();

  if (operand_type.getRank() == 0) {
    return error("The input vector must have rank > 0.");
  }

  if (operand_type.getRank() > result_type.getRank()) {
    return error(
        "The rank of the input vector must be smaller or equal to the rank "
        "of the result vector.");
  }

  if (operand_type.getRank() != getBroadcastDimensions().size()) {
    return error(
        "The size of the `broadcast_dimensions` attribute must be equal to "
        "the rank of the input vector.");
  }
  auto dims = llvm::to_vector(getBroadcastDimensions());
  for (int i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0 || dims[i] >= result_type.getRank()) {
      return error(
          "The values in the `broadcast_dimensions` attribute must be in the "
          "range [0, result.shape.rank={0}).",
          result_type.getRank());
    }
    if (i > 0 && dims[i] <= dims[i - 1]) {
      return error(
          "The values in the `broadcast_dimensions` attribute must be strictly "
          "increasing.");
    }
  }

  return llvm::success();
}

llvm::LogicalResult ReturnOp::verify() {
  // The operand number and types must match the custom primitive signature.
  const auto& results = getParentOp()->getResultTypes();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing custom_primitive (@"
           << getParentOp()->getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match the result type (" << results[i]
                         << ")"
                         << " in custom_primitive @"
                         << getParentOp()->getName();

  return llvm::success();
}

namespace {
int kTmemMaxColumns = 512;
int kTmemCellBitwidth = 32;

llvm::LogicalResult VerifyTmemRefType(mlir::Operation* op,
                                      mlir::MemRefType tmem_ref_type) {
  mlir::Attribute tmem = TmemAttr::get(op->getContext());
  if (tmem_ref_type.getMemorySpace() != tmem) {
    return op->emitError() << "The tmem memref must have a "
                              "mosaic_gpu.tmem memory space but got: "
                           << tmem_ref_type.getMemorySpace();
  }

  return llvm::success();
}
}  // namespace

llvm::LogicalResult TmemAllocOp::verify() {
  mlir::Attribute smem = mlir::gpu::AddressSpaceAttr::get(
      getContext(), mlir::gpu::AddressSpace::Workgroup);
  mlir::MemRefType smem_ref_type = getSmemPtr().getType();
  if (smem_ref_type.getMemorySpace() != smem) {
    return emitError()
           << "The `smem_ptr` memref must have the Workgroup address "
              "space but got: "
           << smem_ref_type.getMemorySpace();
  }

  mlir::MemRefType tmem_ref_type = getResult().getType();
  llvm::LogicalResult result = VerifyTmemRefType(getOperation(), tmem_ref_type);
  if (result.failed()) {
    return result;
  }

  int num_unpacked_columns = tmem_ref_type.getShape()[1];
  int packing = getPacking();
  if (packing != 1) {
    if (packing * tmem_ref_type.getElementTypeBitWidth() != kTmemCellBitwidth) {
      return emitError() << "Only unpacked, or fully packed allocations "
                            "are supported. Expected packing to be either "
                            "1 or 32 / element_bitwidth, but got: "
                            "packing = "
                         << packing << ", element_bitwidth = "
                         << tmem_ref_type.getElementTypeBitWidth();
    }
    if (num_unpacked_columns % packing != 0) {
      return emitError() << "The number of unpacked columns must be "
                            "divisible by the packing factor, but got: "
                         << num_unpacked_columns << " / " << packing;
    }
  }

  int num_allocated_columns = num_unpacked_columns / packing;
  if (num_allocated_columns > kTmemMaxColumns) {
    return emitError()
           << "The number of allocated columns must be less than or equal to "
           << kTmemMaxColumns << " but got: " << num_allocated_columns;
  }

  return llvm::success();
}

llvm::LogicalResult TmemDeallocOp::verify() {
  return VerifyTmemRefType(getOperation(), getTmemRef().getType());
}

llvm::LogicalResult AsyncLoadTmemOp::inferReturnTypes(
    mlir::MLIRContext*, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
  mlir::MemRefType memref_type =
      mlir::cast<mlir::MemRefType>(operands[0].getType());
  auto vector_type = mlir::VectorType::get(memref_type.getShape(),
                                           memref_type.getElementType());
  inferredReturnTypes.assign({vector_type});
  return mlir::success();
}

llvm::LogicalResult AsyncLoadTmemOp::verify() {
  if (getSource().getType().getElementType() !=
      getResult().getType().getElementType()) {
    return emitError() << "The `source` and `result` must have "
                          "the same element type.";
  }
  if (getSource().getType().getShape() != getResult().getType().getShape()) {
    return emitError() << "The `source` and `result` must have the same shape.";
  }
  return VerifyTmemRefType(getOperation(), getSource().getType());
}

llvm::LogicalResult AsyncStoreTmemOp::verify() {
  if (getSource().getType().getElementType() !=
      getDestination().getType().getElementType()) {
    return emitError() << "The `source` and `destination` must have "
                          "the same element type.";
  }
  if (getSource().getType().getShape() !=
      getDestination().getType().getShape()) {
    return emitError()
           << "The `source` and `destination` must have the same shape.";
  }
  return VerifyTmemRefType(getOperation(), getDestination().getType());
}

llvm::LogicalResult TmemLayoutCastOp::verify() {
  return VerifyTmemRefType(getOperation(), getRef().getType());
}

llvm::LogicalResult SliceTmemOp::verify() {
  if (VerifyTmemRefType(getOperation(), getSource().getType()).failed() ||
      VerifyTmemRefType(getOperation(), getResult().getType()).failed()) {
    return llvm::failure();
  }
  if (getOffset() % 4 != 0) {
    return emitError() << "The offset must be a multiple of 4 but got: "
                       << getOffset();
  }
  // TODO(allanrenucci): We can't precisely compute the number of columns in
  // source/result because we need to know packing. We can however assume
  // packing is either 1 (unpacked) or 32 / element_bitwidth (fully packed) and
  // reject some invalid slices.
  return llvm::success();
}

llvm::LogicalResult VectorLoadOp::inferReturnTypes(
    mlir::MLIRContext*, std::optional<mlir::Location>,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
  mlir::MemRefType memref_type =
      mlir::cast<mlir::MemRefType>(operands[0].getType());
  auto vector_type = mlir::VectorType::get(memref_type.getShape(),
                                           memref_type.getElementType());
  inferredReturnTypes.assign({vector_type});
  return mlir::success();
}

llvm::LogicalResult VectorStoreOp::verify() {
  mlir::VectorType src_type = getValueToStore().getType();
  mlir::MemRefType dst_type = getDestination().getType();
  if (src_type.getShape() != dst_type.getShape()) {
    return emitError()
           << "The source and destination must have the same shape but got "
           << src_type.getShape() << " and " << dst_type.getShape();
  }
  if (src_type.getElementType() != dst_type.getElementType()) {
    return emitError()
           << "The source and destination must have the same element type but "
              "got "
           << src_type.getElementType() << " and " << dst_type.getElementType();
  }
  return llvm::success();
}

llvm::LogicalResult BroadcastedIotaOp::verify() {
  mlir::VectorType result_type = getResult().getType();
  if (getDimension() >= result_type.getRank()) {
    return emitError(llvm::formatv(
        "dimension={0} must be smaller than the rank={1} of the result.",
        getDimension(), result_type.getRank()));
  }
  return llvm::success();
}

llvm::LogicalResult PrintLayoutOp::verify() {
  if (auto ref_ty = mlir::dyn_cast<mlir::MemRefType>(getValue().getType())) {
    if (VerifyTmemRefType(getOperation(), ref_ty).failed()) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

llvm::LogicalResult OptimizationBarrierOp::inferReturnTypes(
    mlir::MLIRContext*, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
  if (operands.empty()) {
    return mlir::emitOptionalError(location, "expected non-empty operands");
  }
  mlir::TypeRange operand_types = operands.getTypes();
  inferredReturnTypes.assign(operand_types.begin(), operand_types.end());
  return mlir::success();
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
