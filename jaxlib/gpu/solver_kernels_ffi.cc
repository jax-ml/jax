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

#define SOLVER_DISPATCH_IMPL(impl, ...)         \
  if (dataType == ffi::F32) {                   \
    return impl<float>(__VA_ARGS__);            \
  } else if (dataType == ffi::F64) {            \
    return impl<double>(__VA_ARGS__);           \
  } else if (dataType == ffi::C64) {            \
    return impl<gpuComplex>(__VA_ARGS__);       \
  } else if (dataType == ffi::C128) {           \
    return impl<gpuDoubleComplex>(__VA_ARGS__); \
  }

#define SOLVER_BLAS_DISPATCH_IMPL(impl, ...)        \
  if (dataType == ffi::F32) {                       \
    return impl<float>(__VA_ARGS__);                \
  } else if (dataType == ffi::F64) {                \
    return impl<double>(__VA_ARGS__);               \
  } else if (dataType == ffi::C64) {                \
    return impl<gpublasComplex>(__VA_ARGS__);       \
  } else if (dataType == ffi::C128) {               \
    return impl<gpublasDoubleComplex>(__VA_ARGS__); \
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

template <typename T>
ffi::Error SyevdImpl(int64_t batch, int64_t size, gpuStream_t stream,
                     ffi::ScratchAllocator& scratch, SyevdAlgorithm algorithm,
                     bool lower, ffi::AnyBuffer a,
                     ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::AnyBuffer> w,
                     ffi::Result<ffi::Buffer<ffi::S32>> info) {
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(size));
  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));

  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_VECTOR;
  gpusolverFillMode_t uplo =
      lower ? GPUSOLVER_FILL_MODE_LOWER : GPUSOLVER_FILL_MODE_UPPER;

  auto a_data = static_cast<T*>(a.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  auto w_data = static_cast<solver::RealType<T>::value*>(w->untyped_data());
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }
  if (algorithm == SyevdAlgorithm::kJacobi ||
      (algorithm == SyevdAlgorithm::kDefault && size <= 32)) {
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
      FFI_ASSIGN_OR_RETURN(
          auto workspace,
          AllocateWorkspace<T>(scratch, lwork, "syevj_batched"));
      FFI_RETURN_IF_ERROR_STATUS(
          solver::SyevjBatched<T>(handle.get(), jobz, uplo, n, out_data, w_data,
                                  workspace, lwork, info_data, params, batch));
    }
  } else {
    FFI_ASSIGN_OR_RETURN(
        int lwork, solver::SyevdBufferSize<T>(handle.get(), jobz, uplo, n));
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
  SOLVER_DISPATCH_IMPL(SyevdImpl, batch, cols, stream, scratch, algorithm,
                       lower, a, out, w, info);
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

#undef SOLVER_DISPATCH_IMPL
#undef SOLVER_BLAS_DISPATCH_IMPL

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
