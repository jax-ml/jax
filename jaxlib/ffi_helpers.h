#ifndef JAXLIB_FFI_HELPERS_H_
#define JAXLIB_FFI_HELPERS_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace jax {

// Returns from the function if the argument is an ffi::Error.
#define FFI_RETURN_IF_ERROR(...)             \
  do {                                       \
    ::xla::ffi::Error err = (__VA_ARGS__);   \
    if (ABSL_PREDICT_FALSE(err.failure())) { \
      return err;                            \
    }                                        \
  } while (0)

// Returns from the function with an ffi::Error if the argument is an
// absl::Status.
#define FFI_RETURN_IF_ERROR_STATUS(...)     \
  do {                                      \
    ::absl::Status status = (__VA_ARGS__);  \
    if (ABSL_PREDICT_FALSE(!status.ok())) { \
      return ::jax::AsFfiError(status);     \
    }                                       \
  } while (0)

// Returns from the function with an ffi::Error if the RHS is an absl::Status,
// otherwise assigns to the LHS. Most of the complication here stems from the
// fact that we want to support having the LHS wrapped in parentheses (when
// unpacking a tuple, for example).
#define FFI_ASSIGN_OR_RETURN(lhs, rhs) \
  FFI_ASSIGN_OR_RETURN_IMPL_(          \
      FFI_ASSIGN_OR_RETURN_CONCAT_(_status_or_value, __LINE__), lhs, rhs)

#define FFI_ASSIGN_OR_RETURN_IMPL_(statusor, lhs, rhs)        \
  auto statusor = (rhs);                                      \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                   \
    return ::jax::AsFfiError(statusor.status());              \
  }                                                           \
  FFI_ASSIGN_OR_RETURN_UNPARENTHESIZE_IF_PARENTHESIZED(lhs) = \
      (*std::move(statusor))

#define FFI_ASSIGN_OR_RETURN_CONCAT_INNER_(x, y) x##y
#define FFI_ASSIGN_OR_RETURN_CONCAT_(x, y) \
  FFI_ASSIGN_OR_RETURN_CONCAT_INNER_(x, y)

// All the macros below here are to handle the case in FFI_ASSIGN_OR_RETURN
// where the LHS is wrapped in parentheses.
#define FFI_ASSIGN_OR_RETURN_EAT(...)
#define FFI_ASSIGN_OR_RETURN_REM(...) __VA_ARGS__
#define FFI_ASSIGN_OR_RETURN_EMPTY()

#define FFI_ASSIGN_OR_RETURN_IS_EMPTY_INNER(...) \
  FFI_ASSIGN_OR_RETURN_IS_EMPTY_INNER_HELPER((__VA_ARGS__, 0, 1))
#define FFI_ASSIGN_OR_RETURN_IS_EMPTY_INNER_HELPER(args) \
  FFI_ASSIGN_OR_RETURN_IS_EMPTY_INNER_I args
#define FFI_ASSIGN_OR_RETURN_IS_EMPTY_INNER_I(e0, e1, is_empty, ...) is_empty

#define FFI_ASSIGN_OR_RETURN_IS_EMPTY(...) \
  FFI_ASSIGN_OR_RETURN_IS_EMPTY_I(__VA_ARGS__)
#define FFI_ASSIGN_OR_RETURN_IS_EMPTY_I(...) \
  FFI_ASSIGN_OR_RETURN_IS_EMPTY_INNER(_, ##__VA_ARGS__)

#define FFI_ASSIGN_OR_RETURN_IF_1(_Then, _Else) _Then
#define FFI_ASSIGN_OR_RETURN_IF_0(_Then, _Else) _Else
#define FFI_ASSIGN_OR_RETURN_IF(_Cond, _Then, _Else) \
  FFI_ASSIGN_OR_RETURN_CONCAT_(FFI_ASSIGN_OR_RETURN_IF_, _Cond)(_Then, _Else)

#define FFI_ASSIGN_OR_RETURN_IS_PARENTHESIZED(...) \
  FFI_ASSIGN_OR_RETURN_IS_EMPTY(FFI_ASSIGN_OR_RETURN_EAT __VA_ARGS__)

#define FFI_ASSIGN_OR_RETURN_UNPARENTHESIZE_IF_PARENTHESIZED(...)             \
  FFI_ASSIGN_OR_RETURN_IF(FFI_ASSIGN_OR_RETURN_IS_PARENTHESIZED(__VA_ARGS__), \
                          FFI_ASSIGN_OR_RETURN_REM,                           \
                          FFI_ASSIGN_OR_RETURN_EMPTY())                       \
  __VA_ARGS__

template <typename T>
inline absl::StatusOr<T> MaybeCastNoOverflow(
    std::int64_t value, const std::string& source = __FILE__) {
  if constexpr (sizeof(T) == sizeof(std::int64_t)) {
    return value;
  } else {
    if (value > std::numeric_limits<T>::max()) [[unlikely]] {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s: Value (=%d) exceeds the maximum representable value of the "
          "desired type",
          source, value));
    }
    return static_cast<T>(value);
  }
}

inline ::xla::ffi::Error AsFfiError(const absl::Status& status) {
  if (ABSL_PREDICT_FALSE(!status.ok())) {
    return ::xla::ffi::Error(static_cast<XLA_FFI_Error_Code>(status.code()),
                             std::string(status.message()));
  } else {
    return ::xla::ffi::Error::Success();
  }
}

inline int64_t GetBatchSize(::xla::ffi::Span<const int64_t> dims) {
  return absl::c_accumulate(dims, 1, std::multiplies<int64_t>());
}

inline absl::StatusOr<std::pair<int64_t, int64_t>> SplitBatch1D(
    ::xla::ffi::Span<const int64_t> dims,
    const std::string& source = __FILE__) {
  if (dims.size() < 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: Argument must have at least 1 dimension", source));
  }
  return std::make_pair(GetBatchSize(dims.first(dims.size() - 1)), dims.back());
}

inline absl::StatusOr<std::tuple<int64_t, int64_t, int64_t>> SplitBatch2D(
    ::xla::ffi::Span<const int64_t> dims,
    const std::string& source = __FILE__) {
  if (dims.size() < 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s: Argument must have at least 2 dimensions", source));
  }
  auto trailingDims = dims.last(2);
  return std::make_tuple(GetBatchSize(dims.first(dims.size() - 2)),
                         trailingDims.front(), trailingDims.back());
}

template <::xla::ffi::DataType dtype>
auto AllocateScratchMemory(std::size_t size)
    -> std::unique_ptr<std::remove_extent_t<::xla::ffi::NativeType<dtype>>[]> {
  // TODO(paruzelp): use std::make_unique_for_overwrite when C++20 is available.
  using ValueType = std::remove_extent_t<::xla::ffi::NativeType<dtype>>;
  return std::unique_ptr<ValueType[]>(new ValueType[size]);
}

}  // namespace jax

#endif  // JAXLIB_FFI_HELPERS_H_
