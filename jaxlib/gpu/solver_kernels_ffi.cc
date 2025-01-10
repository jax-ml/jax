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
#include <memory>
#include <optional>
#include <string_view>

#if JAX_GPU_HAVE_64_BIT
#include <cstddef>
#endif

#ifdef JAX_GPU_CUDA
#include <limits>
#endif

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/blas_handle_pool.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/make_batch_pointers.h"
#include "jaxlib/gpu/solver_handle_pool.h"
#include "jaxlib/gpu/solver_interface.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::JAX_GPU_NAMESPACE::SyevdAlgorithm);

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = ::xla::ffi;

#if JAX_GPU_HAVE_64_BIT

// Map an FFI buffer element type to the appropriate GPU solver type.
inline absl::StatusOr<gpuDataType> SolverDataType(ffi::DataType dataType,
                                                  std::string_view func) {
  switch (dataType) {
    case ffi::F32:
      return GPU_R_32F;
    case ffi::F64:
      return GPU_R_64F;
    case ffi::C64:
      return GPU_C_32F;
    case ffi::C128:
      return GPU_C_64F;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported dtype %s in %s", absl::FormatStreamed(dataType), func));
  }
}

#endif

#define SOLVER_DISPATCH_IMPL(impl, ...)           \
  switch (dataType) {                             \
    case ffi::F32:                                \
      return impl<float>(__VA_ARGS__);            \
    case ffi::F64:                                \
      return impl<double>(__VA_ARGS__);           \
    case ffi::C64:                                \
      return impl<gpuComplex>(__VA_ARGS__);       \
    case ffi::C128:                               \
      return impl<gpuDoubleComplex>(__VA_ARGS__); \
    default:                                      \
      break;                                      \
  }

#define SOLVER_BLAS_DISPATCH_IMPL(impl, ...)          \
  switch (dataType) {                                 \
    case ffi::F32:                                    \
      return impl<float>(__VA_ARGS__);                \
    case ffi::F64:                                    \
      return impl<double>(__VA_ARGS__);               \
    case ffi::C64:                                    \
      return impl<gpublasComplex>(__VA_ARGS__);       \
    case ffi::C128:                                   \
      return impl<gpublasDoubleComplex>(__VA_ARGS__); \
    default:                                          \
      break;                                          \
  }

// LU decomposition: getrf

template <typename T>
ffi::Error GetrfImpl(int64_t batch, int64_t rows, int64_t cols,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::Buffer<ffi::S32>> ipiv,
                     ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(rows));
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));

  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));
  FFI_ASSIGN_OR_RETURN(int lwork,
                       solver::GetrfBufferSize<T>(handle.get(), m, n));
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<T>(scratch, lwork, "getrf"));

  auto a_data = static_cast<T*>(a.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  auto ipiv_data = ipiv->typed_data();
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  int ipiv_step = std::min(m, n);
  for (auto i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(solver::Getrf<T>(
        handle.get(), m, n, out_data, workspace, lwork, ipiv_data, info_data));
    out_data += m * n;
    ipiv_data += ipiv_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <typename T>
ffi::Error GetrfBatchedImpl(int64_t batch, int64_t cols, gpuStream_t stream,
                            ffi::ScratchAllocator& scratch, ffi::AnyBuffer a,
                            ffi::Result<ffi::AnyBuffer> out,
                            ffi::Result<ffi::Buffer<ffi::S32>> ipiv,
                            ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));
  FFI_ASSIGN_OR_RETURN(auto handle, BlasHandlePool::Borrow(stream));
  FFI_ASSIGN_OR_RETURN(auto batch_ptrs,
                       AllocateWorkspace<T*>(scratch, batch, "batched getrf"));

  auto a_data = a.untyped_data();
  auto out_data = out->untyped_data();
  auto ipiv_data = ipiv->typed_data();
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  MakeBatchPointersAsync(stream, out_data, batch_ptrs, batch,
                         sizeof(T) * n * n);
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuGetLastError());

  FFI_RETURN_IF_ERROR_STATUS(solver::GetrfBatched(
      handle.get(), n, batch_ptrs, n, ipiv_data, info_data, batch));

  return ffi::Error::Success();
}

ffi::Error GetrfDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::Buffer<ffi::S32>> ipiv,
                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
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
    SOLVER_BLAS_DISPATCH_IMPL(GetrfBatchedImpl, batch, cols, stream, scratch, a,
                              out, ipiv, info);
  } else {
    SOLVER_DISPATCH_IMPL(GetrfImpl, batch, rows, cols, stream, scratch, a, out,
                         ipiv, info);
  }
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in getrf", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GetrfFfi, GetrfDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()         // a
                                  .Ret<ffi::AnyBuffer>()         // out
                                  .Ret<ffi::Buffer<ffi::S32>>()  // ipiv
                                  .Ret<ffi::Buffer<ffi::S32>>()  // info
);

// QR decomposition: geqrf

template <typename T>
ffi::Error GeqrfImpl(int64_t batch, int64_t rows, int64_t cols,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::AnyBuffer> tau) {
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(rows));
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));

  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));
  FFI_ASSIGN_OR_RETURN(int lwork,
                       solver::GeqrfBufferSize<T>(handle.get(), m, n));

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
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  int out_step = m * n;
  int tau_step = std::min(m, n);
  for (auto i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(solver::Geqrf<T>(
        handle.get(), m, n, out_data, tau_data, workspace, lwork, info));
    out_data += out_step;
    tau_data += tau_step;
  }
  return ffi::Error::Success();
}

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
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  MakeBatchPointersAsync(stream, out_data, out_batch_ptrs, batch,
                         sizeof(T) * m * n);
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuGetLastError());
  MakeBatchPointersAsync(stream, tau_data, tau_batch_ptrs, batch,
                         sizeof(T) * std::min(m, n));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuGetLastError());

  // We ignore the output value of `info` because it is only used for shape
  // checking.
  int info;
  FFI_RETURN_IF_ERROR_STATUS(solver::GeqrfBatched<T>(
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
    SOLVER_BLAS_DISPATCH_IMPL(GeqrfBatchedImpl, batch, rows, cols, stream,
                              scratch, a, out, tau);
  } else {
    SOLVER_DISPATCH_IMPL(GeqrfImpl, batch, rows, cols, stream, scratch, a, out,
                         tau);
  }
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in geqrf", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GeqrfFfi, GeqrfDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // a
                                  .Ret<ffi::AnyBuffer>()  // out
                                  .Ret<ffi::AnyBuffer>()  // tau
);

// Householder transformations: orgqr

template <typename T>
ffi::Error OrgqrImpl(int64_t batch, int64_t rows, int64_t cols, int64_t size,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer a, ffi::AnyBuffer tau,
                     ffi::Result<ffi::AnyBuffer> out) {
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(rows));
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));
  FFI_ASSIGN_OR_RETURN(auto k, MaybeCastNoOverflow<int>(size));

  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));
  FFI_ASSIGN_OR_RETURN(int lwork,
                       solver::OrgqrBufferSize<T>(handle.get(), m, n, k));

  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<T>(scratch, lwork, "orgqr"));
  // Note: We ignore the returned value of info because it is only used for
  // shape checking (which we already do ourselves), but it is expected to be
  // in device memory, so we need to allocate it.
  FFI_ASSIGN_OR_RETURN(auto info, AllocateWorkspace<int>(scratch, 1, "orgqr"));

  auto a_data = static_cast<T*>(a.untyped_data());
  auto tau_data = static_cast<T*>(tau.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  int out_step = m * n;
  for (auto i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(solver::Orgqr<T>(
        handle.get(), m, n, k, out_data, tau_data, workspace, lwork, info));
    out_data += out_step;
    tau_data += k;
  }
  return ffi::Error::Success();
}

ffi::Error OrgqrDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         ffi::AnyBuffer a, ffi::AnyBuffer tau,
                         ffi::Result<ffi::AnyBuffer> out) {
  auto dataType = a.element_type();
  if (dataType != tau.element_type() || dataType != out->element_type()) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to orgqr must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  FFI_ASSIGN_OR_RETURN((auto [tau_batch, size]),
                       SplitBatch1D(tau.dimensions()));
  if (tau_batch != batch) {
    return ffi::Error::InvalidArgument(
        "The batch dimensions of the inputs to orgqr must match");
  }
  if (size > cols) {
    return ffi::Error::InvalidArgument(
        "The trailing dimension of the tau input to orgqr must be less than or "
        "equal to the number of columns of the input matrix");
  }
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, rows, cols}, "out", "orgqr"));
  SOLVER_DISPATCH_IMPL(OrgqrImpl, batch, rows, cols, size, stream, scratch, a,
                       tau, out);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in orgqr", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(OrgqrFfi, OrgqrDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // a
                                  .Arg<ffi::AnyBuffer>()  // tau
                                  .Ret<ffi::AnyBuffer>()  // out
);

// Symmetric (Hermitian) eigendecomposition:
// * Jacobi algorithm: syevj/heevj (batches of matrices up to 32)
// * QR algorithm: syevd/heevd
// For historical reasons, the target is called "syevd" even though it
// dispatches dynamically to both syevd and syevj depending on the problem
// size and the algorithm selected by the user via the `algorithm` attribute.

#if JAX_GPU_HAVE_64_BIT

ffi::Error Syevd64Impl(int64_t batch, int64_t n, gpuStream_t stream,
                       ffi::ScratchAllocator& scratch, bool lower,
                       ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                       ffi::Result<ffi::AnyBuffer> w,
                       ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));
  auto dataType = a.element_type();
  FFI_ASSIGN_OR_RETURN(auto aType, SolverDataType(dataType, "syevd"));
  FFI_ASSIGN_OR_RETURN(auto wType, SolverDataType(w->element_type(), "syevd"));

  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_VECTOR;
  gpusolverFillMode_t uplo =
      lower ? GPUSOLVER_FILL_MODE_LOWER : GPUSOLVER_FILL_MODE_UPPER;

  gpusolverDnParams_t params;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusolverDnCreateParams(&params));
  std::unique_ptr<gpusolverDnParams, void (*)(gpusolverDnParams_t)>
      params_cleanup(
          params, [](gpusolverDnParams_t p) { gpusolverDnDestroyParams(p); });

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusolverDnXsyevd_bufferSize(
      handle.get(), params, jobz, uplo, n, aType, /*a=*/nullptr, n, wType,
      /*w=*/nullptr, aType, &workspaceInBytesOnDevice,
      &workspaceInBytesOnHost));

  auto maybe_workspace = scratch.Allocate(workspaceInBytesOnDevice);
  if (!maybe_workspace.has_value()) {
    return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                      "Unable to allocate device workspace for syevd");
  }
  auto workspaceOnDevice = maybe_workspace.value();
  auto workspaceOnHost =
      std::unique_ptr<char[]>(new char[workspaceInBytesOnHost]);

  const char* a_data = static_cast<const char*>(a.untyped_data());
  char* out_data = static_cast<char*>(out->untyped_data());
  char* w_data = static_cast<char*>(w->untyped_data());
  int* info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  size_t out_step = n * n * ffi::ByteWidth(dataType);
  size_t w_step = n * ffi::ByteWidth(ffi::ToReal(dataType));

  for (auto i = 0; i < batch; ++i) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpusolverDnXsyevd(
        handle.get(), params, jobz, uplo, n, aType, out_data, n, wType, w_data,
        aType, workspaceOnDevice, workspaceInBytesOnDevice,
        workspaceOnHost.get(), workspaceInBytesOnHost, info_data));
    out_data += out_step;
    w_data += w_step;
    ++info_data;
  }

  return ffi::Error::Success();
}

#endif

template <typename T>
ffi::Error SyevdImpl(int64_t batch, int64_t size, gpuStream_t stream,
                     ffi::ScratchAllocator& scratch, bool lower,
                     ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::AnyBuffer> w,
                     ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(size));
  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));

  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_VECTOR;
  gpusolverFillMode_t uplo =
      lower ? GPUSOLVER_FILL_MODE_LOWER : GPUSOLVER_FILL_MODE_UPPER;

  auto a_data = static_cast<T*>(a.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  auto w_data =
      static_cast<typename solver::RealType<T>::value*>(w->untyped_data());
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  FFI_ASSIGN_OR_RETURN(int lwork,
                       solver::SyevdBufferSize<T>(handle.get(), jobz, uplo, n));
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<T>(scratch, lwork, "syevd"));
  int out_step = n * n;
  for (auto i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(solver::Syevd<T>(handle.get(), jobz, uplo, n,
                                                out_data, w_data, workspace,
                                                lwork, info_data));
    out_data += out_step;
    w_data += n;
    ++info_data;
  }

  return ffi::Error::Success();
}

template <typename T>
ffi::Error SyevdjImpl(int64_t batch, int64_t size, gpuStream_t stream,
                      ffi::ScratchAllocator& scratch, bool lower,
                      ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                      ffi::Result<ffi::AnyBuffer> w,
                      ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(size));
  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));

  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_VECTOR;
  gpusolverFillMode_t uplo =
      lower ? GPUSOLVER_FILL_MODE_LOWER : GPUSOLVER_FILL_MODE_UPPER;

  auto a_data = static_cast<T*>(a.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  auto w_data =
      static_cast<typename solver::RealType<T>::value*>(w->untyped_data());
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  gpuSyevjInfo_t params;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusolverDnCreateSyevjInfo(&params));
  std::unique_ptr<gpuSyevjInfo, void (*)(gpuSyevjInfo_t)> params_cleanup(
      params, [](gpuSyevjInfo_t p) { gpusolverDnDestroySyevjInfo(p); });

  if (batch == 1) {
    FFI_ASSIGN_OR_RETURN(int lwork, solver::SyevjBufferSize<T>(
                                        handle.get(), jobz, uplo, n, params));
    FFI_ASSIGN_OR_RETURN(auto workspace,
                         AllocateWorkspace<T>(scratch, lwork, "syevj"));
    FFI_RETURN_IF_ERROR_STATUS(solver::Syevj<T>(handle.get(), jobz, uplo, n,
                                                out_data, w_data, workspace,
                                                lwork, info_data, params));
  } else {
    FFI_ASSIGN_OR_RETURN(
        int lwork, solver::SyevjBatchedBufferSize<T>(handle.get(), jobz, uplo,
                                                     n, params, batch));
    FFI_ASSIGN_OR_RETURN(auto workspace,
                         AllocateWorkspace<T>(scratch, lwork, "syevj_batched"));
    FFI_RETURN_IF_ERROR_STATUS(
        solver::SyevjBatched<T>(handle.get(), jobz, uplo, n, out_data, w_data,
                                workspace, lwork, info_data, params, batch));
  }

  return ffi::Error::Success();
}

ffi::Error SyevdDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         SyevdAlgorithm algorithm, bool lower, ffi::AnyBuffer a,
                         ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::AnyBuffer> w,
                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
  auto dataType = a.element_type();
  if (dataType != out->element_type() ||
      ffi::ToReal(dataType) != w->element_type()) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to syevd must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  if (rows != cols) {
    return ffi::Error::InvalidArgument(
        "The input matrix to syevd must be square");
  }
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, rows, cols}, "out", "syevd"));
  FFI_RETURN_IF_ERROR(CheckShape(w->dimensions(), {batch, cols}, "w", "syevd"));
  FFI_RETURN_IF_ERROR(CheckShape(info->dimensions(), batch, "info", "syevd"));
  if (algorithm == SyevdAlgorithm::kJacobi ||
      (algorithm == SyevdAlgorithm::kDefault && cols <= 32)) {
    SOLVER_DISPATCH_IMPL(SyevdjImpl, batch, cols, stream, scratch, lower, a,
                         out, w, info);
  } else {
#if JAX_GPU_HAVE_64_BIT
    return Syevd64Impl(batch, cols, stream, scratch, lower, a, out, w, info);
#else
    SOLVER_DISPATCH_IMPL(SyevdImpl, batch, cols, stream, scratch, lower, a, out,
                         w, info);
#endif
  }
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in syevd", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SyevdFfi, SyevdDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<SyevdAlgorithm>("algorithm")
                                  .Attr<bool>("lower")
                                  .Arg<ffi::AnyBuffer>()         // a
                                  .Ret<ffi::AnyBuffer>()         // out
                                  .Ret<ffi::AnyBuffer>()         // w
                                  .Ret<ffi::Buffer<ffi::S32>>()  // info
);

// Symmetric rank-k update: syrk

template <typename T>
ffi::Error SyrkImpl(gpuStream_t stream, bool transpose, ffi::AnyBuffer a,
                    ffi::AnyBuffer c_in, ffi::AnyBuffer alpha,
                    ffi::AnyBuffer beta, ffi::Result<ffi::AnyBuffer> c_out) {
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  if (alpha.element_count() != 1 || beta.element_count() != 1) {
    return ffi::Error::InvalidArgument(
        "The alpha and beta inputs to syrk must be scalars");
  }
  auto size = transpose ? cols : rows;
  FFI_RETURN_IF_ERROR(
      CheckShape(c_in.dimensions(), {batch, size, size}, "c_in", "syrk"));
  FFI_RETURN_IF_ERROR(
      CheckShape(c_out->dimensions(), {batch, size, size}, "c_out", "syrk"));

  FFI_ASSIGN_OR_RETURN(auto n,
                       MaybeCastNoOverflow<int>(transpose ? cols : rows));
  FFI_ASSIGN_OR_RETURN(auto k,
                       MaybeCastNoOverflow<int>(transpose ? rows : cols));
  gpublasFillMode_t uplo = GPUSOLVER_FILL_MODE_UPPER;
  gpublasOperation_t trans = transpose ? GPUBLAS_OP_N : GPUBLAS_OP_T;

  const T* a_data = static_cast<const T*>(a.untyped_data());
  T* c_data = static_cast<T*>(c_in.untyped_data());
  T* c_out_data = static_cast<T*>(c_out->untyped_data());

  // with alpha or beta provided as device_pointers, cublas<T>syrk will SIGSEGV
  T host_alpha;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(&host_alpha, alpha.untyped_data(),
                                             sizeof(T), gpuMemcpyDeviceToHost,
                                             stream));

  T host_beta;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(&host_beta, beta.untyped_data(),
                                             sizeof(T), gpuMemcpyDeviceToHost,
                                             stream));

  if (c_data != c_out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(
        gpuMemcpyAsync(c_out_data, c_data, c_in.size_bytes(),
                       gpuMemcpyDeviceToDevice, stream));
  }
  FFI_ASSIGN_OR_RETURN(auto handle, BlasHandlePool::Borrow(stream));
  for (int i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(solver::Syrk<T>(handle.get(), uplo, trans, n, k,
                                               &host_alpha, a_data, &host_beta,
                                               c_out_data));
    a_data += k * n;
    c_out_data += n * n;
  }
  return ffi::Error::Success();
}

ffi::Error SyrkDispatch(gpuStream_t stream, bool transpose, ffi::AnyBuffer a,
                        ffi::AnyBuffer c_in, ffi::AnyBuffer alpha,
                        ffi::AnyBuffer beta,
                        ffi::Result<ffi::AnyBuffer> c_out) {
  auto dataType = a.element_type();
  SOLVER_BLAS_DISPATCH_IMPL(SyrkImpl, stream, transpose, a, c_in, alpha, beta,
                            c_out);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in syrk", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SyrkFfi, SyrkDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Attr<bool>("transpose")  // transpose
                                  .Arg<ffi::AnyBuffer>()    // a
                                  .Arg<ffi::AnyBuffer>()    // c_in
                                  .Arg<ffi::AnyBuffer>()    // alpha
                                  .Arg<ffi::AnyBuffer>()    // beta
                                  .Ret<ffi::AnyBuffer>()    // c_out
);

// Singular Value Decomposition: gesvd

#if JAX_GPU_HAVE_64_BIT

ffi::Error Gesvd64Impl(int64_t batch, int64_t m, int64_t n, gpuStream_t stream,
                       ffi::ScratchAllocator& scratch, bool full_matrices,
                       bool compute_uv, ffi::AnyBuffer a,
                       ffi::Result<ffi::AnyBuffer> out,
                       ffi::Result<ffi::AnyBuffer> s,
                       ffi::Result<ffi::AnyBuffer> u,
                       ffi::Result<ffi::AnyBuffer> vt,
                       ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));
  signed char job = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';
  auto dataType = a.element_type();
  FFI_ASSIGN_OR_RETURN(auto aType, SolverDataType(dataType, "syevd"));
  FFI_ASSIGN_OR_RETURN(auto sType, SolverDataType(s->element_type(), "syevd"));

  gpusolverDnParams_t params;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusolverDnCreateParams(&params));
  std::unique_ptr<gpusolverDnParams, void (*)(gpusolverDnParams_t)>
      params_cleanup(
          params, [](gpusolverDnParams_t p) { gpusolverDnDestroyParams(p); });

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusolverDnXgesvd_bufferSize(
      handle.get(), params, job, job, m, n, aType, /*a=*/nullptr, m, sType,
      /*s=*/nullptr, aType, /*u=*/nullptr, m, aType, /*vt=*/nullptr, n, aType,
      &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

  auto maybe_workspace = scratch.Allocate(workspaceInBytesOnDevice);
  if (!maybe_workspace.has_value()) {
    return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                      "Unable to allocate device workspace for gesvd");
  }
  auto workspaceOnDevice = maybe_workspace.value();
  auto workspaceOnHost =
      std::unique_ptr<char[]>(new char[workspaceInBytesOnHost]);

  const char* a_data = static_cast<const char*>(a.untyped_data());
  char* out_data = static_cast<char*>(out->untyped_data());
  char* s_data = static_cast<char*>(s->untyped_data());
  char* u_data = static_cast<char*>(u->untyped_data());
  char* vt_data = static_cast<char*>(vt->untyped_data());
  int* info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  size_t out_step = m * n * ffi::ByteWidth(dataType);
  size_t s_step = n * ffi::ByteWidth(ffi::ToReal(dataType));
  size_t u_step = 0;
  size_t vt_step = 0;
  if (compute_uv) {
    u_step = m * (full_matrices ? m : n) * ffi::ByteWidth(dataType);
    vt_step = n * n * ffi::ByteWidth(dataType);
  }
  for (auto i = 0; i < batch; ++i) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpusolverDnXgesvd(
        handle.get(), params, job, job, m, n, aType, out_data, m, sType, s_data,
        aType, u_data, m, aType, vt_data, n, aType, workspaceOnDevice,
        workspaceInBytesOnDevice, workspaceOnHost.get(), workspaceInBytesOnHost,
        info_data));
    out_data += out_step;
    s_data += s_step;
    u_data += u_step;
    vt_data += vt_step;
    ++info_data;
  }

  return ffi::Error::Success();
}

#else

template <typename T>
ffi::Error GesvdImpl(int64_t batch, int64_t rows, int64_t cols,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     bool full_matrices, bool compute_uv, ffi::AnyBuffer a,
                     ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::AnyBuffer> s,
                     ffi::Result<ffi::AnyBuffer> u,
                     ffi::Result<ffi::AnyBuffer> vt,
                     ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(rows));
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));
  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));
  signed char job = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';

  FFI_ASSIGN_OR_RETURN(int lwork,
                       solver::GesvdBufferSize<T>(handle.get(), job, m, n));
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<T>(scratch, lwork, "gesvd"));
  auto a_data = static_cast<T*>(a.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  auto s_data =
      static_cast<typename solver::RealType<T>::value*>(s->untyped_data());
  auto u_data = compute_uv ? static_cast<T*>(u->untyped_data()) : nullptr;
  auto vt_data = compute_uv ? static_cast<T*>(vt->untyped_data()) : nullptr;
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream)));
  }

  int out_step = m * n;
  int u_step = compute_uv ? m * (full_matrices ? m : n) : 0;
  int vt_step = compute_uv ? n * n : 0;
  for (auto i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(
        solver::Gesvd<T>(handle.get(), job, m, n, out_data, s_data, u_data,
                         vt_data, workspace, lwork, info_data));
    out_data += out_step;
    s_data += n;  // n is always less than m because of the logic in dispatch.
    u_data += u_step;
    vt_data += vt_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

#endif  // JAX_GPU_HAVE_64_BIT

ffi::Error GesvdDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         bool full_matrices, bool compute_uv, bool transposed,
                         ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::AnyBuffer> s,
                         ffi::Result<ffi::AnyBuffer> u,
                         ffi::Result<ffi::AnyBuffer> vt,
                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
  auto dataType = a.element_type();
  if (out->element_type() != dataType ||
      s->element_type() != ffi::ToReal(dataType) ||
      u->element_type() != dataType || vt->element_type() != dataType) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to gesvd must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  int64_t m = transposed ? cols : rows;
  int64_t n = transposed ? rows : cols;
  if (n > m) {
    return ffi::Error::InvalidArgument(
        "The GPU implementation of gesvd requires that the input matrix be m x "
        "n with m >= n");
  }
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, rows, cols}, "out", "gesvd"));
  FFI_RETURN_IF_ERROR(CheckShape(s->dimensions(), {batch, n}, "s", "gesvd"));
  if (compute_uv) {
    if (full_matrices) {
      FFI_RETURN_IF_ERROR(
          CheckShape(u->dimensions(), {batch, m, m}, "u", "gesvd"));
    } else {
      if (transposed) {
        FFI_RETURN_IF_ERROR(
            CheckShape(u->dimensions(), {batch, n, m}, "u", "gesvd"));
      } else {
        FFI_RETURN_IF_ERROR(
            CheckShape(u->dimensions(), {batch, m, n}, "u", "gesvd"));
      }
    }
    FFI_RETURN_IF_ERROR(
        CheckShape(vt->dimensions(), {batch, n, n}, "vt", "gesvd"));
  }
  FFI_RETURN_IF_ERROR(CheckShape(info->dimensions(), batch, "info", "gesvd"));

#if JAX_GPU_HAVE_64_BIT
  return Gesvd64Impl(batch, m, n, stream, scratch, full_matrices, compute_uv, a,
                     out, s, u, vt, info);
#else
  SOLVER_DISPATCH_IMPL(GesvdImpl, batch, m, n, stream, scratch, full_matrices,
                       compute_uv, a, out, s, u, vt, info);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in gesvd", absl::FormatStreamed(dataType)));
#endif
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GesvdFfi, GesvdDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<bool>("full_matrices")
                                  .Attr<bool>("compute_uv")
                                  .Attr<bool>("transposed")
                                  .Arg<ffi::AnyBuffer>()         // a
                                  .Ret<ffi::AnyBuffer>()         // out
                                  .Ret<ffi::AnyBuffer>()         // s
                                  .Ret<ffi::AnyBuffer>()         // u
                                  .Ret<ffi::AnyBuffer>()         // vt
                                  .Ret<ffi::Buffer<ffi::S32>>()  // info
);

#ifdef JAX_GPU_CUDA

template <typename T>
ffi::Error GesvdjImpl(int64_t batch, int64_t rows, int64_t cols,
                      gpuStream_t stream, ffi::ScratchAllocator& scratch,
                      bool full_matrices, bool compute_uv, ffi::AnyBuffer a,
                      ffi::Result<ffi::AnyBuffer> out,
                      ffi::Result<ffi::AnyBuffer> s,
                      ffi::Result<ffi::AnyBuffer> u,
                      ffi::Result<ffi::AnyBuffer> v,
                      ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(rows));
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));
  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));

  gpusolverEigMode_t job =
      compute_uv ? GPUSOLVER_EIG_MODE_VECTOR : GPUSOLVER_EIG_MODE_NOVECTOR;
  int econ = full_matrices ? 0 : 1;

  gpuGesvdjInfo_t params;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpusolverDnCreateGesvdjInfo(&params));
  std::unique_ptr<gpuGesvdjInfo, void (*)(gpuGesvdjInfo_t)> params_cleanup(
      params, [](gpuGesvdjInfo_t p) { gpusolverDnDestroyGesvdjInfo(p); });

  auto a_data = static_cast<T*>(a.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  auto s_data =
      static_cast<typename solver::RealType<T>::value*>(s->untyped_data());
  auto u_data = static_cast<T*>(u->untyped_data());
  auto v_data = static_cast<T*>(v->untyped_data());
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  if (batch <= 1 || batch > std::numeric_limits<int>::max() || m > 32 ||
      n > 32 || econ) {
    FFI_ASSIGN_OR_RETURN(int lwork, solver::GesvdjBufferSize<T>(
                                        handle.get(), job, econ, m, n, params));
    FFI_ASSIGN_OR_RETURN(auto workspace,
                         AllocateWorkspace<T>(scratch, lwork, "gesvdj"));
    int k = std::min(m, n);
    int out_step = m * n;
    int u_step = m * (full_matrices ? m : k);
    int v_step = n * (full_matrices ? n : k);
    for (auto i = 0; i < batch; ++i) {
      FFI_RETURN_IF_ERROR_STATUS(solver::Gesvdj<T>(
          handle.get(), job, econ, m, n, out_data, s_data, u_data, v_data,
          workspace, lwork, info_data, params));
      out_data += out_step;
      s_data += k;
      u_data += u_step;
      v_data += v_step;
      ++info_data;
    }
  } else {
    FFI_ASSIGN_OR_RETURN(int lwork, solver::GesvdjBatchedBufferSize<T>(
                                        handle.get(), job, m, n, params,
                                        static_cast<int>(batch)));
    FFI_ASSIGN_OR_RETURN(
        auto workspace, AllocateWorkspace<T>(scratch, lwork, "gesvdj_batched"));
    FFI_RETURN_IF_ERROR_STATUS(solver::GesvdjBatched<T>(
        handle.get(), job, m, n, out_data, s_data, u_data, v_data, workspace,
        lwork, info_data, params, static_cast<int>(batch)));
  }
  return ffi::Error::Success();
}

ffi::Error GesvdjDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                          bool full_matrices, bool compute_uv, ffi::AnyBuffer a,
                          ffi::Result<ffi::AnyBuffer> out,
                          ffi::Result<ffi::AnyBuffer> s,
                          ffi::Result<ffi::AnyBuffer> u,
                          ffi::Result<ffi::AnyBuffer> v,
                          ffi::Result<ffi::Buffer<ffi::S32>> info) {
  auto dataType = a.element_type();
  if (out->element_type() != dataType ||
      s->element_type() != ffi::ToReal(dataType) ||
      u->element_type() != dataType || v->element_type() != dataType) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to gesvdj must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  int64_t size = std::min(rows, cols);
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, rows, cols}, "out", "gesvdj"));
  FFI_RETURN_IF_ERROR(
      CheckShape(s->dimensions(), {batch, size}, "s", "gesvdj"));
  // U and V must always be allocated even if compute_uv is false.
  if (full_matrices) {
    FFI_RETURN_IF_ERROR(
        CheckShape(u->dimensions(), {batch, rows, rows}, "u", "gesvdj"));
    FFI_RETURN_IF_ERROR(
        CheckShape(v->dimensions(), {batch, cols, cols}, "v", "gesvdj"));
  } else {
    FFI_RETURN_IF_ERROR(
        CheckShape(u->dimensions(), {batch, rows, size}, "u", "gesvdj"));
    FFI_RETURN_IF_ERROR(
        CheckShape(v->dimensions(), {batch, cols, size}, "v", "gesvdj"));
  }
  FFI_RETURN_IF_ERROR(CheckShape(info->dimensions(), batch, "info", "gesvdj"));

  SOLVER_DISPATCH_IMPL(GesvdjImpl, batch, rows, cols, stream, scratch,
                       full_matrices, compute_uv, a, out, s, u, v, info);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in gesvdj", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GesvdjFfi, GesvdjDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<bool>("full_matrices")
                                  .Attr<bool>("compute_uv")
                                  .Arg<ffi::AnyBuffer>()         // a
                                  .Ret<ffi::AnyBuffer>()         // out
                                  .Ret<ffi::AnyBuffer>()         // s
                                  .Ret<ffi::AnyBuffer>()         // u
                                  .Ret<ffi::AnyBuffer>()         // v
                                  .Ret<ffi::Buffer<ffi::S32>>()  // info
);

// csrlsvqr: Linear system solve via Sparse QR

template <typename T>
ffi::Error CsrlsvqrImpl(int64_t n, int64_t nnz, double tol, int reorder,
                        gpuStream_t stream, ffi::AnyBuffer csrValA,
                        ffi::Buffer<ffi::S32> csrColIndA,
                        ffi::Buffer<ffi::S32> csrRowPtrA, ffi::AnyBuffer b,
                        ffi::Result<ffi::AnyBuffer> x) {
  FFI_ASSIGN_OR_RETURN(auto handle, SpSolverHandlePool::Borrow(stream));

  FFI_ASSIGN_OR_RETURN(auto int_n, MaybeCastNoOverflow<int>(n));
  FFI_ASSIGN_OR_RETURN(auto int_nnz, MaybeCastNoOverflow<int>(nnz));

  cusparseMatDescr_t matdesc = nullptr;
  JAX_FFI_RETURN_IF_GPU_ERROR(cusparseCreateMatDescr(&matdesc));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      cusparseSetMatType(matdesc, CUSPARSE_MATRIX_TYPE_GENERAL));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      cusparseSetMatIndexBase(matdesc, CUSPARSE_INDEX_BASE_ZERO));

  auto* csrValA_data = static_cast<T*>(csrValA.untyped_data());
  auto* csrColIndA_data = csrColIndA.typed_data();
  auto* csrRowPtrA_data = csrRowPtrA.typed_data();
  auto* b_data = static_cast<T*>(b.untyped_data());
  auto* x_data = static_cast<T*>(x->untyped_data());

  int singularity = -1;
  auto result = solver::Csrlsvqr<T>(
      handle.get(), int_n, int_nnz, matdesc, csrValA_data, csrRowPtrA_data,
      csrColIndA_data, b_data, tol, reorder, x_data, &singularity);
  cusparseDestroyMatDescr(matdesc);
  FFI_RETURN_IF_ERROR_STATUS(result);

  if (singularity >= 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "Singular matrix in linear solve.");
  }

  return ffi::Error::Success();
}

ffi::Error CsrlsvqrDispatch(gpuStream_t stream, int reorder, double tol,
                            ffi::AnyBuffer csrValA,
                            ffi::Buffer<ffi::S32> csrColIndA,
                            ffi::Buffer<ffi::S32> csrRowPtrA, ffi::AnyBuffer b,
                            ffi::Result<ffi::AnyBuffer> x) {
  auto dataType = csrValA.element_type();
  if (dataType != b.element_type() || dataType != x->element_type()) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to csrlsvqr must have the same element type");
  }
  int64_t n = b.element_count();
  int64_t nnz = csrValA.element_count();
  FFI_RETURN_IF_ERROR(
      CheckShape(csrColIndA.dimensions(), nnz, "csrColIndA", "csrlsvqr"));
  FFI_RETURN_IF_ERROR(
      CheckShape(csrRowPtrA.dimensions(), n + 1, "csrColPtrA", "csrlsvqr"));
  FFI_RETURN_IF_ERROR(CheckShape(x->dimensions(), n, "x", "csrlsvqr"));
  SOLVER_DISPATCH_IMPL(CsrlsvqrImpl, n, nnz, tol, reorder, stream, csrValA,
                       csrColIndA, csrRowPtrA, b, x);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in csrlsvqr", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CsrlsvqrFfi, CsrlsvqrDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Attr<int>("reorder")          // reorder
                                  .Attr<double>("tol")           // tol
                                  .Arg<ffi::AnyBuffer>()         // csrValA
                                  .Arg<ffi::Buffer<ffi::S32>>()  // csrColIndA
                                  .Arg<ffi::Buffer<ffi::S32>>()  // csrRowPtrA
                                  .Arg<ffi::AnyBuffer>()         // b
                                  .Ret<ffi::AnyBuffer>()         // x
);

#endif  // JAX_GPU_CUDA

// Symmetric tridiagonal reduction: sytrd

template <typename T>
ffi::Error SytrdImpl(int64_t batch, int64_t size, gpuStream_t stream,
                     ffi::ScratchAllocator& scratch, bool lower,
                     ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::AnyBuffer> d,
                     ffi::Result<ffi::AnyBuffer> e,
                     ffi::Result<ffi::AnyBuffer> tau,
                     ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(size));
  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));

  gpusolverFillMode_t uplo =
      lower ? GPUSOLVER_FILL_MODE_LOWER : GPUSOLVER_FILL_MODE_UPPER;
  FFI_ASSIGN_OR_RETURN(int lwork,
                       solver::SytrdBufferSize<T>(handle.get(), uplo, n));
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<T>(scratch, lwork, "sytrd"));

  auto* a_data = static_cast<T*>(a.untyped_data());
  auto* out_data = static_cast<T*>(out->untyped_data());
  auto* d_data =
      static_cast<typename solver::RealType<T>::value*>(d->untyped_data());
  auto* e_data =
      static_cast<typename solver::RealType<T>::value*>(e->untyped_data());
  auto* tau_data = static_cast<T*>(tau->untyped_data());
  auto* info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  int out_step = n * n;
  for (int64_t i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(solver::Sytrd<T>(handle.get(), uplo, n, out_data,
                                                d_data, e_data, tau_data,
                                                workspace, lwork, info_data));
    out_data += out_step;
    d_data += n;
    e_data += n - 1;
    tau_data += n - 1;
    ++info_data;
  }
  return ffi::Error::Success();
}

ffi::Error SytrdDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         bool lower, ffi::AnyBuffer a,
                         ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::AnyBuffer> d,
                         ffi::Result<ffi::AnyBuffer> e,
                         ffi::Result<ffi::AnyBuffer> tau,
                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
  auto dataType = a.element_type();
  if (out->element_type() != dataType ||
      d->element_type() != ffi::ToReal(dataType) ||
      e->element_type() != ffi::ToReal(dataType) ||
      tau->element_type() != dataType) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to sytrd must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  if (rows != cols) {
    return ffi::Error::InvalidArgument(
        "The input matrix to sytrd must be square");
  }
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, rows, cols}, "out", "sytrd"));
  FFI_RETURN_IF_ERROR(CheckShape(d->dimensions(), {batch, cols}, "d", "sytrd"));
  FFI_RETURN_IF_ERROR(
      CheckShape(e->dimensions(), {batch, cols - 1}, "e", "sytrd"));
  FFI_RETURN_IF_ERROR(
      CheckShape(tau->dimensions(), {batch, cols - 1}, "tau", "sytrd"));
  FFI_RETURN_IF_ERROR(CheckShape(info->dimensions(), batch, "info", "sytrd"));
  SOLVER_DISPATCH_IMPL(SytrdImpl, batch, rows, stream, scratch, lower, a, out,
                       d, e, tau, info);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in sytrd", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SytrdFfi, SytrdDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<bool>("lower")
                                  .Arg<ffi::AnyBuffer>()         // a
                                  .Ret<ffi::AnyBuffer>()         // out
                                  .Ret<ffi::AnyBuffer>()         // d
                                  .Ret<ffi::AnyBuffer>()         // e
                                  .Ret<ffi::AnyBuffer>()         // tau
                                  .Ret<ffi::Buffer<ffi::S32>>()  // info
);

#undef SOLVER_DISPATCH_IMPL
#undef SOLVER_BLAS_DISPATCH_IMPL

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
