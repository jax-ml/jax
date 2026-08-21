/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// OneAPI solver FFI kernels.
//
// This file owns all solver dispatch + implementation for the OneAPI backend.
// It should be compiled ONLY for OneAPI (via jaxlib/oneapi/BUILD.bazel) and
// replaces jaxlib/gpu/solver_kernels_ffi.cc in the OneAPI build.

#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/solver_kernels_ffi.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/oneapi/oneapi_gpu_runtime.h"
#include "xla/ffi/api/ffi.h"

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(jax::oneapi::SyevdAlgorithm);

namespace jax {
namespace oneapi {

namespace ffi = ::xla::ffi;

// Picks the real (orgqr/ormqr) vs complex (ungqr/unmqr) oneMKL routine at
// compile time inside one templated Impl.
template <typename T>
inline constexpr bool kIsComplex =
    std::is_same_v<T, gpuComplex> || std::is_same_v<T, gpuDoubleComplex>;

// Maps an element type to its real scalar: complex<T> -> T, real -> itself.
// Used by sytrd/hetrd whose d/e outputs are always real.
template <typename T>
struct RealType {
  using value = T;
};
template <>
struct RealType<gpuComplex> {
  using value = float;
};
template <>
struct RealType<gpuDoubleComplex> {
  using value = double;
};

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

// LU decomposition: getrf (stub)

ffi::Error GetrfDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::Buffer<ffi::S32>> ipiv,
                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "getrf: not yet implemented for OneAPI");
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

// QR Decomposition: geqrf

template <typename T>
ffi::Error GeqrfImpl(int64_t batch, int64_t rows, int64_t cols,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::AnyBuffer> tau) {
  int64_t m = rows;
  int64_t n = cols;
  // JAX lowers linalg FFI calls column-major, so the leading dimension is the
  // row count (see _linalg_ffi_lowering in jax/_src/lax/linalg.py).
  int64_t lda = m;
  int64_t tau_len = std::min(m, n);
  int64_t stride_a = m * n;
  int64_t stride_tau = tau_len;

  int64_t scratchpad_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(TryCatchToStatus([&] {
    scratchpad_size = ::oneapi::mkl::lapack::geqrf_batch_scratchpad_size<T>(
        *stream, m, n, lda, stride_a, stride_tau, batch);
  }));
  FFI_ASSIGN_OR_RETURN(auto scratchpad,
                       AllocateWorkspace<T>(scratch, scratchpad_size, "geqrf"));

  auto* a_data = static_cast<T*>(a.untyped_data());
  auto* out_data = static_cast<T*>(out->untyped_data());
  auto* tau_data = static_cast<T*>(tau->untyped_data());
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  try {
    ::oneapi::mkl::lapack::geqrf_batch(
        *stream, m, n, out_data, lda, stride_a, tau_data, stride_tau, batch,
        scratchpad, scratchpad_size);
  } catch (std::exception const& e) {
    // No info output buffer; any failure is fatal.
    return ffi::Error::Internal(e.what());
  } catch (...) {
    return ffi::Error::Internal("geqrf: unknown exception");
  }
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

  SOLVER_DISPATCH_IMPL(GeqrfImpl, batch, rows, cols, stream, scratch, a, out,
                       tau);
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

// Householder reconstruction: orgqr (real) / ungqr (complex)

template <typename T>
ffi::Error OrgqrImpl(int64_t batch, int64_t rows, int64_t cols, int64_t size,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer a, ffi::AnyBuffer tau,
                     ffi::Result<ffi::AnyBuffer> out) {
  int64_t m = rows;
  int64_t n = cols;
  int64_t k = size;
  int64_t lda = m;
  int64_t stride_a = m * n;
  int64_t stride_tau = k;

  int64_t scratchpad_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(TryCatchToStatus([&] {
    if constexpr (kIsComplex<T>) {
      scratchpad_size = ::oneapi::mkl::lapack::ungqr_batch_scratchpad_size<T>(
          *stream, m, n, k, lda, stride_a, stride_tau, batch);
    } else {
      scratchpad_size = ::oneapi::mkl::lapack::orgqr_batch_scratchpad_size<T>(
          *stream, m, n, k, lda, stride_a, stride_tau, batch);
    }
  }));
  FFI_ASSIGN_OR_RETURN(auto scratchpad,
                       AllocateWorkspace<T>(scratch, scratchpad_size, "orgqr"));

  auto* a_data = static_cast<T*>(a.untyped_data());
  auto* tau_data = static_cast<T*>(tau.untyped_data());
  auto* out_data = static_cast<T*>(out->untyped_data());
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  try {
    if constexpr (kIsComplex<T>) {
      ::oneapi::mkl::lapack::ungqr_batch(
          *stream, m, n, k, out_data, lda, stride_a, tau_data, stride_tau,
          batch, scratchpad, scratchpad_size);
    } else {
      ::oneapi::mkl::lapack::orgqr_batch(
          *stream, m, n, k, out_data, lda, stride_a, tau_data, stride_tau,
          batch, scratchpad, scratchpad_size);
    }
  } catch (std::exception const& e) {
    // No info output buffer; any failure is fatal.
    return ffi::Error::Internal(e.what());
  } catch (...) {
    return ffi::Error::Internal("orgqr: unknown exception");
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

// Householder multiply: ormqr (real) / unmqr (complex)

template <typename T>
ffi::Error OrmqrImpl(int64_t batch, int64_t c_rows, int64_t c_cols, int64_t k,
                     int64_t a_rows, int64_t a_cols, bool left, bool transpose,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer a, ffi::AnyBuffer tau, ffi::AnyBuffer c,
                     ffi::Result<ffi::AnyBuffer> out) {
  int64_t m = c_rows;
  int64_t n = c_cols;
  int64_t lda = a_rows;
  int64_t ldc = m;

  auto side = left ? ::oneapi::mkl::side::left : ::oneapi::mkl::side::right;
  // Real ormqr accepts trans; complex unmqr requires conjtrans.
  ::oneapi::mkl::transpose trans;
  if (!transpose) {
    trans = ::oneapi::mkl::transpose::nontrans;
  } else if constexpr (kIsComplex<T>) {
    trans = ::oneapi::mkl::transpose::conjtrans;
  } else {
    trans = ::oneapi::mkl::transpose::trans;
  }

  int64_t scratchpad_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(TryCatchToStatus([&] {
    if constexpr (kIsComplex<T>) {
      scratchpad_size = ::oneapi::mkl::lapack::unmqr_scratchpad_size<T>(
          *stream, side, trans, m, n, k, lda, ldc);
    } else {
      scratchpad_size = ::oneapi::mkl::lapack::ormqr_scratchpad_size<T>(
          *stream, side, trans, m, n, k, lda, ldc);
    }
  }));
  FFI_ASSIGN_OR_RETURN(auto scratchpad,
                       AllocateWorkspace<T>(scratch, scratchpad_size, "ormqr"));

  auto* a_data = static_cast<T*>(a.untyped_data());
  auto* tau_data = static_cast<T*>(tau.untyped_data());
  auto* c_data = static_cast<T*>(c.untyped_data());
  auto* out_data = static_cast<T*>(out->untyped_data());
  if (c_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, c_data, c.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  int64_t a_step = a_rows * a_cols;
  // oneMKL offers ormqr/unmqr only in the group form (arrays of per-matrix
  // pointers and parameters), so the uniform batch is implemented as a loop.
  for (int64_t i = 0; i < batch; ++i) {
    try {
      if constexpr (kIsComplex<T>) {
        ::oneapi::mkl::lapack::unmqr(*stream, side, trans, m, n, k, a_data,
                                     lda, tau_data, out_data, ldc,
                                     scratchpad, scratchpad_size);
      } else {
        ::oneapi::mkl::lapack::ormqr(*stream, side, trans, m, n, k, a_data,
                                     lda, tau_data, out_data, ldc,
                                     scratchpad, scratchpad_size);
      }
    } catch (std::exception const& e) {
      // No info output buffer; any failure is fatal.
      return ffi::Error::Internal(e.what());
    } catch (...) {
      return ffi::Error::Internal("ormqr: unknown exception");
    }
    out_data += m * n;
    a_data += a_step;
    tau_data += k;
  }
  return ffi::Error::Success();
}

ffi::Error OrmqrDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         bool left, bool transpose, ffi::AnyBuffer a,
                         ffi::AnyBuffer tau, ffi::AnyBuffer c,
                         ffi::Result<ffi::AnyBuffer> out) {
  auto dataType = a.element_type();
  if (dataType != tau.element_type() || dataType != c.element_type() ||
      dataType != out->element_type()) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to ormqr must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, a_rows, a_cols]),
                       SplitBatch2D(a.dimensions()));
  FFI_ASSIGN_OR_RETURN((auto [tau_batch, k]), SplitBatch1D(tau.dimensions()));
  FFI_ASSIGN_OR_RETURN((auto [c_batch, c_rows, c_cols]),
                       SplitBatch2D(c.dimensions()));
  if (tau_batch != batch || c_batch != batch) {
    return ffi::Error::InvalidArgument(
        "The batch dimensions of the inputs to ormqr must match");
  }
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, c_rows, c_cols}, "out", "ormqr"));

  SOLVER_DISPATCH_IMPL(OrmqrImpl, batch, c_rows, c_cols, k, a_rows, a_cols,
                       left, transpose, stream, scratch, a, tau, c, out);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in ormqr", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(OrmqrFfi, OrmqrDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<bool>("left")
                                  .Attr<bool>("transpose")
                                  .Arg<ffi::AnyBuffer>()  // a
                                  .Arg<ffi::AnyBuffer>()  // tau
                                  .Arg<ffi::AnyBuffer>()  // c
                                  .Ret<ffi::AnyBuffer>()  // out
);

// Cholesky Decomposition: potrf

template <typename T>
ffi::Error PotrfImpl(int64_t batch, int64_t n, gpuStream_t stream,
                     ffi::ScratchAllocator& scratch, bool lower,
                     ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::Buffer<ffi::S32>> info) {
  auto uplo = lower ? ::oneapi::mkl::uplo::lower : ::oneapi::mkl::uplo::upper;

  int64_t lda = n;
  int64_t stride_a = n * n;
  int64_t scratchpad_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(TryCatchToStatus([&] {
    scratchpad_size = ::oneapi::mkl::lapack::potrf_batch_scratchpad_size<T>(
        *stream, uplo, n, lda, stride_a, batch);
  }));
  FFI_ASSIGN_OR_RETURN(
      auto scratchpad,
      AllocateWorkspace<T>(scratch, scratchpad_size, "potrf_batch"));

  auto* a_data = static_cast<T*>(a.untyped_data());
  auto* out_data = static_cast<T*>(out->untyped_data());
  auto* info_data = info->typed_data();

  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }
  try {
    JAX_FFI_RETURN_IF_GPU_ERROR(
        SyclMemsetAsync(info_data, 0, batch * sizeof(int32_t), stream));

    try {
      ::oneapi::mkl::lapack::potrf_batch(
          *stream, uplo, n, out_data, lda, stride_a, batch, scratchpad,
          scratchpad_size);
    } catch (::oneapi::mkl::lapack::batch_error const& be) {
      // oneMKL throws on non-positive-definite input. Recover each failing
      // matrix's info code and write it to info_data (info > 0 => JAX
      // NaN-fills).
      auto const& ids = be.ids();
      auto const& exceptions = be.exceptions();
      std::vector<int32_t> host_info(batch, 0);
      for (std::size_t ei = 0; ei < ids.size(); ++ei) {
        try {
          std::rethrow_exception(exceptions[ei]);
        } catch (::oneapi::mkl::lapack::exception const& e) {
          // Only computation_error is packed per matrix; argument errors are a
          // whole-call property, thrown directly, so they never reach here.
          if (e.info() < 0) {
            return ffi::Error::Internal(e.what());
          }
          host_info[ids[ei]] = static_cast<int32_t>(e.info());
        }
      }
      JAX_FFI_RETURN_IF_GPU_ERROR(
          gpuMemcpyAsync(info_data, host_info.data(), batch * sizeof(int32_t),
                         gpuMemcpyHostToDevice, stream));
      // host_info is a stack local, so block until the async copy reads it.
      JAX_FFI_RETURN_IF_GPU_ERROR(gpuStreamSynchronize(stream));
    }
  } catch (std::exception const& e) {
    // Covers both synchronous and asynchronous sycl::exception (both derive
    // from std::exception) and every oneMKL error.
    return ffi::Error::Internal(e.what());
  } catch (...) {
    return ffi::Error::Internal("potrf_batch: unknown exception");
  }
  return ffi::Error::Success();
}

ffi::Error PotrfDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         bool lower, ffi::AnyBuffer a,
                         ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
  auto dataType = a.element_type();
  if (dataType != out->element_type()) {
    return ffi::Error::InvalidArgument(
        "The input and output to potrf must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                       SplitBatch2D(a.dimensions()));
  if (rows != cols) {
    return ffi::Error::InvalidArgument(
        "The input matrix to potrf must be square");
  }
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, rows, cols}, "out", "potrf"));
  FFI_RETURN_IF_ERROR(CheckShape(info->dimensions(), batch, "info", "potrf"));

  SOLVER_DISPATCH_IMPL(PotrfImpl, batch, cols, stream, scratch, lower, a, out,
                       info);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in potrf", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PotrfFfi, PotrfDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Attr<bool>("lower")
                                  .Arg<ffi::AnyBuffer>()         // a
                                  .Ret<ffi::AnyBuffer>()         // out
                                  .Ret<ffi::Buffer<ffi::S32>>()  // info
);

// Symmetric Eigenvalue Decomposition: syevd (real) / heevd (complex) (stub)

ffi::Error SyevdDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         SyevdAlgorithm algorithm, bool lower, ffi::AnyBuffer a,
                         ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::AnyBuffer> w,
                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "syevd: not yet implemented for OneAPI");
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

// Symmetric rank-k update: syrk (BLAS)

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
  int64_t size = transpose ? cols : rows;
  FFI_RETURN_IF_ERROR(
      CheckShape(c_in.dimensions(), {batch, size, size}, "c_in", "syrk"));
  FFI_RETURN_IF_ERROR(
      CheckShape(c_out->dimensions(), {batch, size, size}, "c_out", "syrk"));

  int64_t n = transpose ? cols : rows;
  int64_t k = transpose ? rows : cols;
  auto uplo = ::oneapi::mkl::uplo::upper;
  // Matches the CUDA reference: non-transpose path multiplies A^T, transpose
  // path multiplies A.
  auto trans = transpose ? ::oneapi::mkl::transpose::nontrans
                         : ::oneapi::mkl::transpose::trans;

  const T* a_data = static_cast<const T*>(a.untyped_data());
  T* c_data = static_cast<T*>(c_in.untyped_data());
  T* c_out_data = static_cast<T*>(c_out->untyped_data());

  // The FFI passes alpha/beta as 1-element device buffers. oneMKL's
  // value_or_pointer<T> would accept those pointers, but it silently falls back
  // to a host dereference when they are not recognized as USM in the queue's
  // context, so copy to host explicitly.
  T host_alpha;
  T host_beta;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(&host_alpha, alpha.untyped_data(),
                                             sizeof(T), gpuMemcpyDeviceToHost,
                                             stream));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(&host_beta, beta.untyped_data(),
                                             sizeof(T), gpuMemcpyDeviceToHost,
                                             stream));
  // The copies are async; block before reading the host scalars.
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuStreamSynchronize(stream));

  if (c_data != c_out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(
        gpuMemcpyAsync(c_out_data, c_data, c_in.size_bytes(),
                       gpuMemcpyDeviceToDevice, stream));
  }

  // lda matches the CUDA reference: nontrans (transpose==true) uses n, else k.
  int64_t lda = transpose ? n : k;
  int64_t ldc = n;
  try {
    // Invalid arguments throw synchronously from the call. A failure detected
    // on the device afterwards goes to the queue's async handler, which XLA
    // only logs, so it cannot surface here.
    ::oneapi::mkl::blas::column_major::syrk_batch(
        *stream, uplo, trans, n, k, host_alpha, a_data, lda, k * n, host_beta,
        c_out_data, ldc, n * n, batch);
  } catch (std::exception const& e) {
    return ffi::Error::Internal(e.what());
  } catch (...) {
    return ffi::Error::Internal("syrk: unknown exception");
  }
  return ffi::Error::Success();
}

ffi::Error SyrkDispatch(gpuStream_t stream, bool transpose, ffi::AnyBuffer a,
                        ffi::AnyBuffer c_in, ffi::AnyBuffer alpha,
                        ffi::AnyBuffer beta,
                        ffi::Result<ffi::AnyBuffer> c_out) {
  auto dataType = a.element_type();
  SOLVER_DISPATCH_IMPL(SyrkImpl, stream, transpose, a, c_in, alpha, beta,
                       c_out);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in syrk", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(SyrkFfi, SyrkDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Attr<bool>("transpose")
                                  .Arg<ffi::AnyBuffer>()  // a
                                  .Arg<ffi::AnyBuffer>()  // c_in
                                  .Arg<ffi::AnyBuffer>()  // alpha
                                  .Arg<ffi::AnyBuffer>()  // beta
                                  .Ret<ffi::AnyBuffer>()  // c_out
);

// Singular Value Decomposition: gesvd (stub)

ffi::Error GesvdDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         bool full_matrices, bool compute_uv, bool transposed,
                         ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::AnyBuffer> s,
                         ffi::Result<ffi::AnyBuffer> u,
                         ffi::Result<ffi::AnyBuffer> vt,
                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "gesvd: not yet implemented for OneAPI");
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

// Jacobi SVD: gesvdj (stub, no oneMKL equivalent)

ffi::Error GesvdjDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                          bool full_matrices, bool compute_uv, ffi::AnyBuffer a,
                          ffi::Result<ffi::AnyBuffer> out,
                          ffi::Result<ffi::AnyBuffer> s,
                          ffi::Result<ffi::AnyBuffer> u,
                          ffi::Result<ffi::AnyBuffer> v,
                          ffi::Result<ffi::Buffer<ffi::S32>> info) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "gesvdj: not supported for OneAPI (no oneMKL Jacobi SVD)");
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

// Symmetric/Hermitian Tridiagonal Reduction: sytrd (real) / hetrd (complex)

template <typename T>
ffi::Error SytrdImpl(int64_t batch, int64_t size, gpuStream_t stream,
                     ffi::ScratchAllocator& scratch, bool lower,
                     ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                     ffi::Result<ffi::AnyBuffer> d,
                     ffi::Result<ffi::AnyBuffer> e,
                     ffi::Result<ffi::AnyBuffer> tau,
                     ffi::Result<ffi::Buffer<ffi::S32>> info) {
  using Real = typename RealType<T>::value;
  int64_t n = size;
  int64_t lda = n;
  auto uplo = lower ? ::oneapi::mkl::uplo::lower : ::oneapi::mkl::uplo::upper;

  int64_t scratchpad_size = 0;
  JAX_FFI_RETURN_IF_GPU_ERROR(TryCatchToStatus([&] {
    if constexpr (kIsComplex<T>) {
      scratchpad_size = ::oneapi::mkl::lapack::hetrd_scratchpad_size<T>(
          *stream, uplo, n, lda);
    } else {
      scratchpad_size = ::oneapi::mkl::lapack::sytrd_scratchpad_size<T>(
          *stream, uplo, n, lda);
    }
  }));
  // One scratchpad serves the whole batch: XLA SYCL queues are in-order, so
  // iteration i + 1 cannot start before iteration i has finished with it.
  FFI_ASSIGN_OR_RETURN(auto scratchpad,
                       AllocateWorkspace<T>(scratch, scratchpad_size, "sytrd"));

  auto* a_data = static_cast<T*>(a.untyped_data());
  auto* out_data = static_cast<T*>(out->untyped_data());
  auto* d_data = static_cast<Real*>(d->untyped_data());
  auto* e_data = static_cast<Real*>(e->untyped_data());
  auto* tau_data = static_cast<T*>(tau->untyped_data());
  auto* info_data = info->typed_data();
  if (a_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, a_data, a.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }
  // sytrd/hetrd report failure by throwing, not via an info code, but JAX still
  // reads an info buffer; zero it (0 == success) and treat any throw as fatal
  // (a reduction only fails on an illegal argument).
  JAX_FFI_RETURN_IF_GPU_ERROR(
      SyclMemsetAsync(info_data, 0, batch * sizeof(int32_t), stream));

  int64_t out_step = n * n;
  // oneMKL has no strided sytrd_batch/hetrd_batch, so the uniform batch stays a
  // loop here.
  try {
    for (int64_t i = 0; i < batch; ++i) {
      if constexpr (kIsComplex<T>) {
        ::oneapi::mkl::lapack::hetrd(*stream, uplo, n, out_data, lda, d_data,
                                     e_data, tau_data, scratchpad,
                                     scratchpad_size);
      } else {
        ::oneapi::mkl::lapack::sytrd(*stream, uplo, n, out_data, lda, d_data,
                                     e_data, tau_data, scratchpad,
                                     scratchpad_size);
      }
      out_data += out_step;
      d_data += n;
      e_data += n - 1;
      tau_data += n - 1;
    }
  } catch (std::exception const& ex) {
    return ffi::Error::Internal(ex.what());
  } catch (...) {
    return ffi::Error::Internal("sytrd: unknown exception");
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

}  // namespace oneapi
}  // namespace jax
