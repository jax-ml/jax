#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_

#include <array>
#include <cstdint>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Diagnostics.h"
#include "mlir/include/mlir/IR/Value.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "tsl/platform/statusor.h"

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

// All the macros below here are to handle the case in
// FAILUREOR_ASSIGN_OR_RETURN where the LHS is wrapped in parentheses. See a
// more detailed discussion at https://stackoverflow.com/a/62984543
#define FAILUREOR_ASSIGN_OR_RETURN_UNPARENTHESIZE_IF_PARENTHESIZED(X) \
  FAILUREOR_ASSIGN_OR_RETURN_ESCAPE(FAILUREOR_ASSIGN_OR_RETURN_EMPTY X)
#define FAILUREOR_ASSIGN_OR_RETURN_EMPTY(...) \
  FAILUREOR_ASSIGN_OR_RETURN_EMPTY __VA_ARGS__
#define FAILUREOR_ASSIGN_OR_RETURN_ESCAPE(...) \
  FAILUREOR_ASSIGN_OR_RETURN_ESCAPE_(__VA_ARGS__)
#define FAILUREOR_ASSIGN_OR_RETURN_ESCAPE_(...) \
  FAILUREOR_ASSIGN_OR_RETURN_##__VA_ARGS__
#define FAILUREOR_ASSIGN_OR_RETURN_FAILUREOR_ASSIGN_OR_RETURN_EMPTY

#define FAILUREOR_ASSIGN_OR_RETURN_IMPL(failureor, lhs, rhs)        \
  auto failureor = rhs;                                             \
  if (failed(failureor)) {                                          \
    return failure();                                               \
  }                                                                 \
  FAILUREOR_ASSIGN_OR_RETURN_UNPARENTHESIZE_IF_PARENTHESIZED(lhs) = \
      (std::move(failureor).value());
#define FAILUREOR_ASSIGN_OR_RETURN(lhs, rhs) \
  FAILUREOR_ASSIGN_OR_RETURN_IMPL(           \
      TF_STATUS_MACROS_CONCAT_NAME(failureor, __COUNTER__), lhs, rhs)

#define RETURN_IF_FAILED(...)  \
  do {                         \
    if (failed(__VA_ARGS__)) { \
      return failure();        \
    }                          \
  } while (false)

template <typename Op>
class StatusToDiagnosticAdapter {
 public:
  // Returns an adapter that converts a non-OK absl::Status to an
  // mlir::InFlightDiagnostic.
  explicit StatusToDiagnosticAdapter(Op op) : op_(op) {}

  // Converts a non-OK absl::Status to an mlir::InFlightDiagnostic.
  mlir::InFlightDiagnostic operator()(const absl::Status &status) const {
    return op_->emitOpError(status.ToString());
  }

 private:
  Op op_;
};

// Returns a callable adapter that converts a non-OK absl::Status to an
// mlir::InFlightDiagnostic.
//
// Example usage:
// ASSIGN_OR_RETURN(T result, DoSomething(), _.With(StatusToDiagnostic(&op)));
template <typename Op>
inline StatusToDiagnosticAdapter<Op> StatusToDiagnostic(Op op) {
  return StatusToDiagnosticAdapter<Op>(op);
}

namespace mlir::tpu {

// TPU_ASSERT_* macros should be understood as an assert, i.e. use it to check
// things that should never happen. We prefer returning failure over a CHECK
// because it's easier to debug from Python (particularly from OSS where symbols
// are removed)
#define TPU_ASSERT_IMPL(stream, cond)                    \
  if (LLVM_UNLIKELY(!(cond))) {                          \
    (stream) << "Internal error: assert failed: " #cond; \
  }
#define TPU_ASSERT_CMP_IMPL(stream, lhs, rhs, cmp)                            \
  if (LLVM_UNLIKELY(!((lhs)cmp(rhs)))) {                                      \
    (stream) << "Internal error: assert failed: " #lhs " " #cmp " " #rhs " (" \
             << (lhs) << " vs. " << (rhs) << ")";                             \
    return failure();                                                         \
  }
#define TPU_ASSERT_OP(cond) TPU_ASSERT_IMPL(op.emitOpError(), cond)
#define TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, cmp) \
  TPU_ASSERT_CMP_IMPL(op.emitOpError(), lhs, rhs, cmp)
#define TPU_ASSERT_EQ_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, ==)
#define TPU_ASSERT_GE_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, >=)
#define TPU_ASSERT_GT_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, >)
#define TPU_ASSERT_LE_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, <=)
#define TPU_ASSERT_LT_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, <)
#define TPU_ASSERT_LOC(loc, cond) TPU_ASSERT_IMPL(mlir::emitError(loc), cond)
#define TPU_ASSERT_CMP_LOC_IMPL(loc, lhs, rhs, cmp) \
  TPU_ASSERT_CMP_IMPL(loc, lhs, rhs, cmp)
#define TPU_ASSERT_EQ_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, ==)
#define TPU_ASSERT_GE_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, >=)
#define TPU_ASSERT_GT_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, >)
#define TPU_ASSERT_LT_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, <)
#define TPU_ASSERT_LE_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, <=)

class Print {
 public:
  explicit Print(Operation *t) : payload_(t) {}
  Operation *payload_;

 private:
  friend std::ostream &operator<<(std::ostream &, Print);
};

std::ostream &operator<<(std::ostream &os, Print p);

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
  if (isa<IntegerType, Float32Type, BFloat16Type, Float8E5M2Type,
          Float8E4M3FNType, Float8E4M3B11FNUZType>(ty)) {
    return ty.getIntOrFloatBitWidth();
  }
  return emitError(UnknownLoc::get(ty.getContext()),
                   "Unsupported type in mosaic dialect: ")
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

// Determines whether the given MemRefType has the given memory space.
bool HasMemorySpace(MemRefType ty, tpu::MemorySpace space);

bool layoutIsValidForValue(const Layout &l, const Value v,
                           const std::array<int64_t, 2> target_shape);

// Returns empty vector on null attribute
FailureOr<SmallVector<Layout>> getLayoutArrayFromAttr(const Attribute attr);

FailureOr<SmallVector<Layout>> getOutLayouts(
    Operation &op, const std::array<int64_t, 2> target_shape);

FailureOr<SmallVector<Layout>> getInLayouts(
    Operation &op, const std::array<int64_t, 2> target_shape);

void setInLayout(Operation *op, ArrayRef<Layout> in);
void setOutLayout(Operation *op, Layout out);
void setOutLayout(Operation *op, ArrayRef<Layout> out);
void setLayout(Operation *op, Layout in, Layout out);
void setLayout(Operation *op, ArrayRef<Layout> in, Layout out);
void setLayout(Operation *op, Layout in, ArrayRef<Layout> out);
void setLayout(Operation *op, ArrayRef<Layout> in, ArrayRef<Layout> out);
}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_
