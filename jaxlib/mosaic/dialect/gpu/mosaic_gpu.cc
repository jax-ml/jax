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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/STLExtras.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/Casting.h"
#include "mlir/include/mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/include/mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Dialect.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/Location.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/TypeRange.h"
#include "mlir/include/mlir/IR/Types.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/ValueRange.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "tsl/platform/statusor.h"

// Generated definitions.
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_dialect.cc.inc"

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

  builder
      .create<mlir::func::FuncOp>(
          builder.getUnknownLoc(), kRuntimeMemcpyAsyncH2DName,
          builder.getFunctionType(TypeRange{ptr, ptr, i64, ptr}, TypeRange{}))
      .setVisibility(mlir::func::FuncOp::Visibility::Private);
}

void MosaicGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_types.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "jaxlib/mosaic/dialect/gpu/mosaic_gpu_ops.cc.inc"
      >();
}

}  // namespace mosaic_gpu
