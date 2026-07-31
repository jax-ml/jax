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
// It should be compiled ONLY for OneAPI (via jaxlib/oneapi/BUILD) and replaces
// jaxlib/gpu/solver_kernels_ffi.cc in the OneAPI build.

#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <exception>
#include <tuple>

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

// QR Decomposition: geqrf (stub)

ffi::Error GeqrfDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::AnyBuffer> tau) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "geqrf: not yet implemented for OneAPI");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GeqrfFfi, GeqrfDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // a
                                  .Ret<ffi::AnyBuffer>()  // out
                                  .Ret<ffi::AnyBuffer>()  // tau
);

// Householder reconstruction: orgqr (real) / ungqr (complex) (stub)

ffi::Error OrgqrDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         ffi::AnyBuffer a, ffi::AnyBuffer tau,
                         ffi::Result<ffi::AnyBuffer> out) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "orgqr: not yet implemented for OneAPI");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(OrgqrFfi, OrgqrDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // a
                                  .Arg<ffi::AnyBuffer>()  // tau
                                  .Ret<ffi::AnyBuffer>()  // out
);

// Householder multiply: ormqr (real) / unmqr (complex) (stub)

ffi::Error OrmqrDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         bool left, bool transpose, ffi::AnyBuffer a,
                         ffi::AnyBuffer tau, ffi::AnyBuffer c,
                         ffi::Result<ffi::AnyBuffer> out) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "ormqr: not yet implemented for OneAPI");
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
  // oneMKL strided-batch Cholesky: one async submission for the whole batch.
  // Contain any throw from the scratchpad-size query at the FFI boundary.
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
  // Contain any SYCL/oneMKL exception so none escapes the FFI boundary.
  try {
    JAX_FFI_RETURN_IF_GPU_ERROR(
        SyclMemsetAsync(info_data, 0, batch * sizeof(int32_t), stream));

    try {
      sycl::event ev = ::oneapi::mkl::lapack::potrf_batch(
          *stream, uplo, n, out_data, lda, stride_a, batch, scratchpad,
          scratchpad_size);
      ev.wait_and_throw();
    } catch (::oneapi::mkl::lapack::batch_error const& be) {
      // oneMKL throws on non-positive-definite input. Recover each failing
      // matrix's info code and write it to info_data (info > 0 => JAX
      // NaN-fills).
      auto const& ids = be.ids();
      auto const& exceptions = be.exceptions();
      for (std::size_t ei = 0; ei < ids.size(); ++ei) {
        try {
          std::rethrow_exception(exceptions[ei]);
        } catch (::oneapi::mkl::lapack::invalid_argument const& e) {
          // Insufficient scratchpad (detail() reports the required minimum)
          // or an otherwise-invalid argument -- not a numerical failure of
          // the factorization, so surface it as an error, not an info code.
          return ffi::Error::Internal(absl::StrFormat(
              "potrf_batch: scratchpad too small: allocated %d, need at "
              "least %d",
              scratchpad_size, e.detail()));
        } catch (::oneapi::mkl::lapack::exception const& e) {
          if (e.info() < 0) {
            return ffi::Error::Internal(e.what());
          }
          JAX_FFI_RETURN_IF_GPU_ERROR(SyclMemfillAsync(
              &info_data[ids[ei]], static_cast<int32_t>(e.info()), 1, stream));
        }
      }
    }
  } catch (std::exception const& e) {
    // Covers both synchronous and asynchronous sycl::exception (both derive
    // from std::exception) and every oneMKL error surfaced by wait_and_throw().
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

// Symmetric rank-k update: syrk (BLAS) (stub)

ffi::Error SyrkDispatch(gpuStream_t stream, bool transpose, ffi::AnyBuffer a,
                        ffi::AnyBuffer c_in, ffi::AnyBuffer alpha,
                        ffi::AnyBuffer beta,
                        ffi::Result<ffi::AnyBuffer> c_out) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "syrk: not yet implemented for OneAPI");
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

// Jacobi SVD: gesvdj (stub — no oneMKL equivalent)

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

// Symmetric/Hermitian Tridiagonal Reduction:
// sytrd (real) / hetrd (complex) (stub)

ffi::Error SytrdDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         bool lower, ffi::AnyBuffer a,
                         ffi::Result<ffi::AnyBuffer> out,
                         ffi::Result<ffi::AnyBuffer> d,
                         ffi::Result<ffi::AnyBuffer> e,
                         ffi::Result<ffi::AnyBuffer> tau,
                         ffi::Result<ffi::Buffer<ffi::S32>> info) {
  return ffi::Error(ffi::ErrorCode::kUnimplemented,
                    "sytrd: not yet implemented for OneAPI");
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
