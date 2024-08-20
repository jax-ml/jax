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

#include "jaxlib/gpu/solver_kernels_ffi.h"

#include <algorithm>
#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/blas_handle_pool.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/make_batch_pointers.h"
#include "jaxlib/gpu/solver_handle_pool.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = ::xla::ffi;

namespace {
template <typename T>
inline absl::StatusOr<T*> AllocateWorkspace(ffi::ScratchAllocator& scratch,
                                            int64_t size,
                                            std::string_view name) {
  auto maybe_workspace = scratch.Allocate(sizeof(T) * size);
  if (!maybe_workspace.has_value()) {
    return absl::Status(
        absl::StatusCode::kResourceExhausted,
        absl::StrFormat("Unable to allocate workspace for %s", name));
  }
  return static_cast<T*>(maybe_workspace.value());
}
}  // namespace

#define SOLVER_DISPATCH_IMPL(impl, ...)         \
  if (dataType == ffi::DataType::F32) {         \
    return impl<float>(__VA_ARGS__);            \
  } else if (dataType == ffi::DataType::F64) {  \
    return impl<double>(__VA_ARGS__);           \
  } else if (dataType == ffi::DataType::C64) {  \
    return impl<gpuComplex>(__VA_ARGS__);       \
  } else if (dataType == ffi::DataType::C128) { \
    return impl<gpuDoubleComplex>(__VA_ARGS__); \
  }

// LU decomposition: getrf

namespace {
#define GETRF_KERNEL_IMPL(type, name)                                          \
  template <>                                                                  \
  struct GetrfKernel<type> {                                                   \
    static absl::StatusOr<int> BufferSize(gpusolverDnHandle_t handle, int m,   \
                                          int n) {                             \
      int lwork;                                                               \
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(                                       \
          name##_bufferSize(handle, m, n, /*A=*/nullptr, /*lda=*/m, &lwork))); \
      return lwork;                                                            \
    }                                                                          \
    static absl::Status Run(gpusolverDnHandle_t handle, int m, int n, type* a, \
                            type* workspace, int lwork, int* ipiv,             \
                            int* info) {                                       \
      return JAX_AS_STATUS(                                                    \
          name(handle, m, n, a, m, workspace, lwork, ipiv, info));             \
    }                                                                          \
  }

template <typename T>
struct GetrfKernel;
GETRF_KERNEL_IMPL(float, gpusolverDnSgetrf);
GETRF_KERNEL_IMPL(double, gpusolverDnDgetrf);
GETRF_KERNEL_IMPL(gpuComplex, gpusolverDnCgetrf);
GETRF_KERNEL_IMPL(gpuDoubleComplex, gpusolverDnZgetrf);
#undef GETRF_KERNEL_IMPL

template <typename T>
ffi::Error GetrfImpl(int64_t batch, int64_t rows, int64_t cols,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::Buffer<ffi::DataType::S32>> ipiv,
                     ffi::Result<ffi::Buffer<ffi::DataType::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(rows));
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));

  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));
  FFI_ASSIGN_OR_RETURN(int lwork,
                       GetrfKernel<T>::BufferSize(handle.get(), m, n));
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<T>(scratch, lwork, "getrf"));

  auto a_data = static_cast<T*>(a.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  auto ipiv_data = ipiv->typed_data();
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(
        gpuMemcpyAsync(out_data, a_data, sizeof(T) * batch * rows * cols,
                       gpuMemcpyDeviceToDevice, stream)));
  }

  int ipiv_step = std::min(m, n);
  for (int i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(GetrfKernel<T>::Run(
        handle.get(), m, n, out_data, workspace, lwork, ipiv_data, info_data));
    out_data += m * n;
    ipiv_data += ipiv_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

#define GETRF_BATCHED_KERNEL_IMPL(type, name)                                 \
  template <>                                                                 \
  struct GetrfBatchedKernel<type> {                                           \
    static absl::Status Run(gpublasHandle_t handle, int n, type** a, int lda, \
                            int* ipiv, int* info, int batch) {                \
      return JAX_AS_STATUS(name(handle, n, a, lda, ipiv, info, batch));       \
    }                                                                         \
  }

template <typename T>
struct GetrfBatchedKernel;
GETRF_BATCHED_KERNEL_IMPL(float, gpublasSgetrfBatched);
GETRF_BATCHED_KERNEL_IMPL(double, gpublasDgetrfBatched);
GETRF_BATCHED_KERNEL_IMPL(gpublasComplex, gpublasCgetrfBatched);
GETRF_BATCHED_KERNEL_IMPL(gpublasDoubleComplex, gpublasZgetrfBatched);
#undef GETRF_BATCHED_KERNEL_IMPL

template <typename T>
ffi::Error GetrfBatchedImpl(int64_t batch, int64_t cols, gpuStream_t stream,
                            ffi::ScratchAllocator& scratch, ffi::AnyBuffer a,
                            ffi::Result<ffi::AnyBuffer> out,
                            ffi::Result<ffi::Buffer<ffi::DataType::S32>> ipiv,
                            ffi::Result<ffi::Buffer<ffi::DataType::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));
  FFI_ASSIGN_OR_RETURN(auto handle, BlasHandlePool::Borrow(stream));
  FFI_ASSIGN_OR_RETURN(auto batch_ptrs,
                       AllocateWorkspace<T*>(scratch, batch, "batched getrf"));

  auto a_data = a.untyped_data();
  auto out_data = out->untyped_data();
  auto ipiv_data = ipiv->typed_data();
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(
        gpuMemcpyAsync(out_data, a_data, sizeof(T) * batch * cols * cols,
                       gpuMemcpyDeviceToDevice, stream)));
  }

  MakeBatchPointersAsync(stream, out_data, batch_ptrs, batch,
                         sizeof(T) * n * n);
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuGetLastError()));

  FFI_RETURN_IF_ERROR_STATUS(GetrfBatchedKernel<T>::Run(
      handle.get(), n, batch_ptrs, n, ipiv_data, info_data, batch));

  return ffi::Error::Success();
}

ffi::Error GetrfDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::Buffer<ffi::DataType::S32>> ipiv,
                         ffi::Result<ffi::Buffer<ffi::DataType::S32>> info) {
  auto dataType = a.element_type();
  if (dataType != out->element_type()) {
    return ffi::Error::InvalidArgument(
        "The input and output to getrf must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, rows, cols}, "out", "getrf"));
  FFI_RETURN_IF_ERROR(CheckShape(
      ipiv->dimensions(), {batch, std::min(rows, cols)}, "ipiv", "getrf"));
  FFI_RETURN_IF_ERROR(CheckShape(info->dimensions(), batch, "info", "getrf"));
  if (batch > 1 && rows == cols && rows / batch <= 128) {
    SOLVER_DISPATCH_IMPL(GetrfBatchedImpl, batch, cols, stream, scratch, a, out,
                         ipiv, info);
  } else {
    SOLVER_DISPATCH_IMPL(GetrfImpl, batch, rows, cols, stream, scratch, a, out,
                         ipiv, info);
  }
  return ffi::Error::InvalidArgument("Unsupported element type for getrf");
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GetrfFfi, GetrfDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<gpuStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()                   // a
        .Ret<ffi::AnyBuffer>()                   // out
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // ipiv
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // info
);

// QR decomposition: geqrf

namespace {
#define GEQRF_KERNEL_IMPL(type, name)                                          \
  template <>                                                                  \
  struct GeqrfKernel<type> {                                                   \
    static absl::StatusOr<int> BufferSize(gpusolverDnHandle_t handle, int m,   \
                                          int n) {                             \
      int lwork;                                                               \
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(                                       \
          name##_bufferSize(handle, m, n, /*A=*/nullptr, /*lda=*/m, &lwork))); \
      return lwork;                                                            \
    }                                                                          \
    static absl::Status Run(gpusolverDnHandle_t handle, int m, int n, type* a, \
                            type* tau, type* workspace, int lwork,             \
                            int* info) {                                       \
      return JAX_AS_STATUS(                                                    \
          name(handle, m, n, a, m, tau, workspace, lwork, info));              \
    }                                                                          \
  }

template <typename T>
struct GeqrfKernel;
GEQRF_KERNEL_IMPL(float, gpusolverDnSgeqrf);
GEQRF_KERNEL_IMPL(double, gpusolverDnDgeqrf);
GEQRF_KERNEL_IMPL(gpuComplex, gpusolverDnCgeqrf);
GEQRF_KERNEL_IMPL(gpuDoubleComplex, gpusolverDnZgeqrf);
#undef GEQRF_KERNEL_IMPL

template <typename T>
ffi::Error GeqrfImpl(int64_t batch, int64_t rows, int64_t cols,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::AnyBuffer> tau) {
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(rows));
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));

  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));
  FFI_ASSIGN_OR_RETURN(int lwork,
                       GeqrfKernel<T>::BufferSize(handle.get(), m, n));

  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<T>(scratch, lwork, "geqrf"));
  // Note: We ignore the returned value of info because it is only used for
  // shape checking (which we already do ourselves), but it is expected to be
  // in device memory, so we need to allocate it.
  FFI_ASSIGN_OR_RETURN(auto info, AllocateWorkspace<int>(scratch, 1, "geqrf"));

  auto a_data = static_cast<T*>(a.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  auto tau_data = static_cast<T*>(tau->untyped_data());
  if (a_data != out_data) {
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(
        gpuMemcpyAsync(out_data, a_data, sizeof(T) * batch * rows * cols,
                       gpuMemcpyDeviceToDevice, stream)));
  }

  int out_step = m * n;
  int tau_step = std::min(m, n);
  for (int i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(GeqrfKernel<T>::Run(
        handle.get(), m, n, out_data, tau_data, workspace, lwork, info));
    out_data += out_step;
    tau_data += tau_step;
  }
  return ffi::Error::Success();
}

#define GEQRF_BATCHED_KERNEL_IMPL(type, name)                               \
  template <>                                                               \
  struct GeqrfBatchedKernel<type> {                                         \
    static absl::Status Run(gpublasHandle_t handle, int m, int n, type** a, \
                            type** tau, int* info, int batch) {    \
      return JAX_AS_STATUS(name(handle, m, n, a, m, tau, info, batch));   \
    }                                                                       \
  }

template <typename T>
struct GeqrfBatchedKernel;
GEQRF_BATCHED_KERNEL_IMPL(float, gpublasSgeqrfBatched);
GEQRF_BATCHED_KERNEL_IMPL(double, gpublasDgeqrfBatched);
GEQRF_BATCHED_KERNEL_IMPL(gpublasComplex, gpublasCgeqrfBatched);
GEQRF_BATCHED_KERNEL_IMPL(gpublasDoubleComplex, gpublasZgeqrfBatched);
#undef GEQRF_BATCHED_KERNEL_IMPL

template <typename T>
ffi::Error GeqrfBatchedImpl(int64_t batch, int64_t rows, int64_t cols,
                            gpuStream_t stream, ffi::ScratchAllocator& scratch,
                            ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                            ffi::Result<ffi::AnyBuffer> tau) {
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(rows));
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));
  FFI_ASSIGN_OR_RETURN(auto handle, BlasHandlePool::Borrow(stream));
  FFI_ASSIGN_OR_RETURN(auto out_batch_ptrs,
                       AllocateWorkspace<T*>(scratch, batch, "batched geqrf"));
  FFI_ASSIGN_OR_RETURN(auto tau_batch_ptrs,
                       AllocateWorkspace<T*>(scratch, batch, "batched geqrf"));

  auto a_data = a.untyped_data();
  auto out_data = out->untyped_data();
  auto tau_data = tau->untyped_data();
  if (a_data != out_data) {
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(
        gpuMemcpyAsync(out_data, a_data, sizeof(T) * batch * rows * cols,
                       gpuMemcpyDeviceToDevice, stream)));
  }

  MakeBatchPointersAsync(stream, out_data, out_batch_ptrs, batch,
                         sizeof(T) * m * n);
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuGetLastError()));
  MakeBatchPointersAsync(stream, tau_data, tau_batch_ptrs, batch,
                         sizeof(T) * std::min(m, n));
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuGetLastError()));

  // We ignore the output value of `info` because it is only used for shape
  // checking.
  int info;
  FFI_RETURN_IF_ERROR_STATUS(GeqrfBatchedKernel<T>::Run(
      handle.get(), m, n, out_batch_ptrs, tau_batch_ptrs, &info, batch));

  return ffi::Error::Success();
}

ffi::Error GeqrfDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::AnyBuffer> tau) {
  auto dataType = a.element_type();
  if (dataType != out->element_type() || dataType != tau->element_type()) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to geqrf must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, rows, cols}, "out", "geqrf"));
  FFI_RETURN_IF_ERROR(CheckShape(
      tau->dimensions(), {batch, std::min(rows, cols)}, "tau", "geqrf"));
  if (batch > 1 && rows / batch <= 128 && cols / batch <= 128) {
    SOLVER_DISPATCH_IMPL(GeqrfBatchedImpl, batch, rows, cols, stream, scratch,
                         a, out, tau);
  } else {
    SOLVER_DISPATCH_IMPL(GeqrfImpl, batch, rows, cols, stream, scratch, a, out,
                         tau);
  }
  return ffi::Error::InvalidArgument("Unsupported element type for geqrf");
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(GeqrfFfi, GeqrfDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // a
                                  .Ret<ffi::AnyBuffer>()  // out
                                  .Ret<ffi::AnyBuffer>()  // tau
);

#undef SOLVER_DISPATCH_IMPL

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
