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
==============================================================================*/

#include "jaxlib/cpu/tridiagonal_solve_kernels.h"

#include <complex>
#include <cstdint>
#include <vector>

#include "jaxlib/ffi_helpers.h"
#include "jaxlib/tridiagonal_solve_perturbed.h"
#include "xla/ffi/api/ffi.h"

namespace jax {

namespace ffi = xla::ffi;

ffi::Error TridiagonalSolvePerturbedFfiImpl(ffi::AnyBuffer dl, ffi::AnyBuffer d,
                                            ffi::AnyBuffer du, ffi::AnyBuffer b,
                                            ffi::Result<ffi::AnyBuffer> x_out) {
  auto dtype = d.element_type();
  if (dl.element_type() != dtype || du.element_type() != dtype ||
      b.element_type() != dtype || x_out->element_type() != dtype) {
    return ffi::Error::InvalidArgument("All types must match");
  }

  FFI_ASSIGN_OR_RETURN((auto [batch_count, b_rows, b_cols]),
                       SplitBatch2D(b.dimensions()));
  const int64_t b_step{b_rows * b_cols};
  const int64_t d_step{b_rows};

  auto* dl_data = dl.untyped_data();
  auto* d_data = d.untyped_data();
  auto* du_data = du.untyped_data();
  auto* b_data = b.untyped_data();
  auto* x_out_data = x_out->untyped_data();

  switch (dtype) {
    case ffi::DataType::F32: {
      std::vector<float> workspace(b_rows * 3);
      std::vector<float> rhs_row_workspace(b_cols);
      auto* dl_ptr = reinterpret_cast<const float*>(dl_data);
      auto* d_ptr = reinterpret_cast<const float*>(d_data);
      auto* du_ptr = reinterpret_cast<const float*>(du_data);
      auto* b_ptr = reinterpret_cast<const float*>(b_data);
      auto* x_out_ptr = reinterpret_cast<float*>(x_out_data);
      for (int64_t i = 0; i < batch_count; ++i) {
        SolveWithGaussianEliminationWithPivotingAndPerturbSingular<float>(
            b_rows, b_cols, dl_ptr, d_ptr, du_ptr, b_ptr, x_out_ptr,
            workspace.data(), rhs_row_workspace.data());
        dl_ptr += d_step;
        d_ptr += d_step;
        du_ptr += d_step;
        b_ptr += b_step;
        x_out_ptr += b_step;
      }
      break;
    }
    case ffi::DataType::F64: {
      std::vector<double> workspace(b_rows * 3);
      std::vector<double> rhs_row_workspace(b_cols);
      auto* dl_ptr = reinterpret_cast<const double*>(dl_data);
      auto* d_ptr = reinterpret_cast<const double*>(d_data);
      auto* du_ptr = reinterpret_cast<const double*>(du_data);
      auto* b_ptr = reinterpret_cast<const double*>(b_data);
      auto* x_out_ptr = reinterpret_cast<double*>(x_out_data);
      for (int64_t i = 0; i < batch_count; ++i) {
        SolveWithGaussianEliminationWithPivotingAndPerturbSingular<double>(
            b_rows, b_cols, dl_ptr, d_ptr, du_ptr, b_ptr, x_out_ptr,
            workspace.data(), rhs_row_workspace.data());
        dl_ptr += d_step;
        d_ptr += d_step;
        du_ptr += d_step;
        b_ptr += b_step;
        x_out_ptr += b_step;
      }
      break;
    }
    case ffi::DataType::C64: {
      std::vector<std::complex<float>> workspace(b_rows * 3);
      std::vector<std::complex<float>> rhs_row_workspace(b_cols);
      auto* dl_ptr = reinterpret_cast<const std::complex<float>*>(dl_data);
      auto* d_ptr = reinterpret_cast<const std::complex<float>*>(d_data);
      auto* du_ptr = reinterpret_cast<const std::complex<float>*>(du_data);
      auto* b_ptr = reinterpret_cast<const std::complex<float>*>(b_data);
      auto* x_out_ptr = reinterpret_cast<std::complex<float>*>(x_out_data);
      for (int64_t i = 0; i < batch_count; ++i) {
        SolveWithGaussianEliminationWithPivotingAndPerturbSingular<
            std::complex<float>>(b_rows, b_cols, dl_ptr, d_ptr, du_ptr, b_ptr,
                                 x_out_ptr, workspace.data(),
                                 rhs_row_workspace.data());
        dl_ptr += d_step;
        d_ptr += d_step;
        du_ptr += d_step;
        b_ptr += b_step;
        x_out_ptr += b_step;
      }
      break;
    }
    case ffi::DataType::C128: {
      std::vector<std::complex<double>> workspace(b_rows * 3);
      std::vector<std::complex<double>> rhs_row_workspace(b_cols);
      auto* dl_ptr = reinterpret_cast<const std::complex<double>*>(dl_data);
      auto* d_ptr = reinterpret_cast<const std::complex<double>*>(d_data);
      auto* du_ptr = reinterpret_cast<const std::complex<double>*>(du_data);
      auto* b_ptr = reinterpret_cast<const std::complex<double>*>(b_data);
      auto* x_out_ptr = reinterpret_cast<std::complex<double>*>(x_out_data);
      for (int64_t i = 0; i < batch_count; ++i) {
        SolveWithGaussianEliminationWithPivotingAndPerturbSingular<
            std::complex<double>>(b_rows, b_cols, dl_ptr, d_ptr, du_ptr, b_ptr,
                                  x_out_ptr, workspace.data(),
                                  rhs_row_workspace.data());
        dl_ptr += d_step;
        d_ptr += d_step;
        du_ptr += d_step;
        b_ptr += b_step;
        x_out_ptr += b_step;
      }
      break;
    }
    default:
      return ffi::Error::InvalidArgument("Unsupported data type");
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(tridiagonal_solve_perturbed_ffi,
                              TridiagonalSolvePerturbedFfiImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>());

}  // namespace jax
