#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/types/span.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "mlir/include/mlir/IR/Value.h"

// TODO: Instead of CHECK_EQs, can we do something like TF_RET_CHECK but with
// MLIR diagnostics?
// e.g.
// #define MLIR_RET_CHECK_EQ(a, b, diagnostic) \
//   do { \
//     const auto a_ = a; \
//     const auto b_ = b; \
//     if (LLVM_UNLIKELY(a_ != b_)) { \
//       return diagnostic << "Check failed: " << #a << " != " << #b << "(" <<
//       a_  << " vs. " << b_ << ")"; \
//     } \
//   } while (false)

#define FAILUREOR_ASSIGN_OR_RETURN_IMPL(failureor, lhs, rhs) \
  auto failureor = rhs;                                      \
  if (failed(failureor)) {                                   \
    return failure();                                        \
  }                                                          \
  lhs = std::move(failureor).value();
#define FAILUREOR_ASSIGN_OR_RETURN(lhs, rhs) \
  FAILUREOR_ASSIGN_OR_RETURN_IMPL(           \
      TF_STATUS_MACROS_CONCAT_NAME(failureor, __COUNTER__), lhs, rhs)

namespace mlir::tpu {

template <bool adjust_bool = false>
FailureOr<int8_t> getTypeBitwidth(Type ty) {
  if (auto integer_ty = dyn_cast<IntegerType>(ty)) {
    const unsigned width = integer_ty.getWidth();
    if constexpr (adjust_bool) {
      // We store only one i1 per vreg element.
      return width == 1 ? 32 : width;
    } else {
      return width;
    }
  }
  if (auto f32_ty = dyn_cast<Float32Type>(ty)) {
    return 32;
  }
  if (auto bf16_ty = dyn_cast<BFloat16Type>(ty)) {
    return 16;
  }
  if (auto f8e5m2_ty = dyn_cast<Float8E5M2Type>(ty)) {
    return 8;
  }
  if (auto f8e4m3fn_ty = dyn_cast<Float8E4M3FNType>(ty)) {
    return 8;
  }
  return emitError(UnknownLoc::get(ty.getContext()), "Unsupported type: ")
         << ty;
}

template <typename T>
ArrayRef<std::remove_const_t<T>> toArrayRef(absl::Span<T> span) {
  return ArrayRef<std::remove_const_t<T>>(span.data(), span.size());
}

inline arith::ConstantOp IdxConst(int64_t idx, OpBuilder &builder,
                                  Location loc) {
  return builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                           builder.getIndexAttr(idx));
}

// Debug only util.
template <typename T>
std::string shapeToString(const T &shape) {
  std::ostringstream os;
  os << "(";
  for (auto it = shape.begin(); it != shape.end(); ++it) {
    if (it != shape.begin()) {
      os << ",";
    }
    os << *it;
  }
  os << ")";
  return os.str();
}

SmallVector<int64_t> ComputeTileStrides(MemRefType memref_ty,
                                        absl::Span<const int64_t> tiling);
// Assuming MKN matmul - This function must only be called after
// canonicalization passes.
//
// Given a set of dimension numbers, Returns a pair of booleans, where the
// first is true if the lhs is transposed
// and the second is true if the rhs is transposed.
std::optional<std::pair<bool, bool>> isTransposedMatmul(
    DotDimensionNumbersAttr dim_numbers);

// Returns true if a >=2D memref has a tiled layout and can be equivalently
// considered as an untiled memref, except for potential padding in the
// minormost dimension up to target_shape[1] (if allow_minormost_padding is
// true).
bool canReinterpretToUntiledMemref(TypedValue<MemRefType> tiled_memref,
                                   const std::array<int64_t, 2> &target_shape,
                                   bool allow_minormost_padding = false);

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_
