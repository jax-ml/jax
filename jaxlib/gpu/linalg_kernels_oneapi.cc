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

#include "jaxlib/gpu/linalg_kernels.h"

#include <cstdint>

#include <algorithm>
#include <complex>

#include "absl/status/status.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/tridiagonal_solve_perturbed.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace {

template <typename T>
void drotg(T* da, T* db, T* c, T* s) {
  if (*db == 0) {
    *c = 1.;
    *s = 0.;
    return;
  }
  T denominator = std::max(std::abs(*da), std::abs(*db));
  T a = *da / denominator;
  T b = *db / denominator;
  T rh = T(1) / std::hypot(a, b);
  *c = a * rh;
  *s = -(b * rh);
}

template <typename T>
void CholeskyUpdateKernel(::sycl::nd_item<1> item, T* rMatrix, T* uVector,
                          int nSize) {
  const auto group = item.get_group();
  const int local_id = static_cast<int>(item.get_local_id(0));
  const int local_range = static_cast<int>(item.get_local_range(0));

  // As per current work schedule, each work-item will
  // handle a subset of the columns of the active row.
  // kernel expects row-major, square upper-triangular matrix.
  for (int k = 0; k < nSize; ++k) {
    T c = 0;
    T s = 0;

    // Compute the row's coefficients in work-item 0 (column 0) and
    // broadcast them to other work-items (columns) in the group.
    if (local_id == 0) {
      drotg(rMatrix + k * nSize + k, uVector + k, &c, &s);
    }

    c = ::sycl::group_broadcast(group, c, 0);
    s = ::sycl::group_broadcast(group, s, 0);

    for (int i = k + local_id; i < nSize; i += local_range) {
      const T r = rMatrix[k * nSize + i];
      const T u = uVector[i];
      rMatrix[k * nSize + i] = c * r - s * u;
      uVector[i] = s * r + c * u;
    }

    // The next row depends on this row's updates to uVector.
    ::sycl::group_barrier(group);
  }
}
}  // namespace

template <typename T>
void LaunchCholeskyUpdateFfiKernelBody(::sycl::queue& queue, T* matrix,
                                       T* vector, int local_range, int nSize) {
  // TODO(Intel-tf): Consider a root-group cooperative launch for multiple
  // work-groups.
  queue.submit([&](::sycl::handler& cgh) {
    // Equal global and local ranges create exactly one work-group.
    cgh.parallel_for(::sycl::nd_range<1>(local_range, local_range),
                     [=](::sycl::nd_item<1> item) {
                       CholeskyUpdateKernel(item, matrix, vector, nSize);
                     });
  });
}

absl::Status LaunchCholeskyUpdateFfiKernel(gpuStream_t stream, void* matrix,
                                           void* vector, int size,
                                           bool is_single_precision) {
  ::sycl::queue& queue = *stream;
  ::sycl::device device = queue.get_device();
  const int max_work_group_size = static_cast<int>(
      device.get_info<::sycl::info::device::max_work_group_size>());

  // Use one work-group and distribute the active row's columns cyclically.
  const int local_range = std::min(size, max_work_group_size);

  if (is_single_precision) {
    return JAX_AS_STATUS(TryCatchToStatus([&] {
      LaunchCholeskyUpdateFfiKernelBody<float>(
          *stream, static_cast<float*>(matrix), static_cast<float*>(vector),
          local_range, size);
    }));
  }
  return JAX_AS_STATUS(TryCatchToStatus([&] {
    LaunchCholeskyUpdateFfiKernelBody<double>(
        *stream, static_cast<double*>(matrix), static_cast<double*>(vector),
        local_range, size);
  }));
}

namespace {

void ComputePermutation(const std::int32_t* pivots,
                        std::int32_t* permutation_out,
                        const std::int32_t pivot_size,
                        const std::int32_t permutation_size) {
  for (int i = 0; i < permutation_size; ++i) {
    permutation_out[i] = i;
  }

  for (int i = 0; i < pivot_size; ++i) {
    if ((pivots[i] < 0) || (pivots[i] >= permutation_size)) {
      continue;
    }
    std::int32_t swap_temporary = permutation_out[i];
    permutation_out[i] = permutation_out[pivots[i]];
    permutation_out[pivots[i]] = swap_temporary;
  }
}

void LuPivotsToPermutationKernel(::sycl::nd_item<1> item,
                                 const std::int32_t* pivots,
                                 std::int32_t* permutation_out,
                                 const std::int64_t batch_size,
                                 const std::int32_t pivot_size,
                                 const std::int32_t permutation_size) {
  for (std::int64_t idx = item.get_global_id(0); idx < batch_size;
       idx += item.get_global_range(0)) {
    ComputePermutation(pivots + idx * pivot_size,
                       permutation_out + idx * permutation_size, pivot_size,
                       permutation_size);
  }
}

}  // namespace

void LaunchLuPivotsToPermutationKernel(gpuStream_t stream,
                                       std::int64_t batch_size,
                                       std::int32_t pivot_size,
                                       std::int32_t permutation_size,
                                       const std::int32_t* pivots,
                                       std::int32_t* permutation) {
  const int local_range = 128;  // work-items per work-group
  const std::int64_t num_work_groups = std::min<std::int64_t>(
      1024, (batch_size + local_range - 1) / local_range);

  absl::Status status = TryCatchToStatus([&] {
    stream->submit([&](::sycl::handler& cgh) {
      cgh.parallel_for(
          ::sycl::nd_range<1>(num_work_groups * local_range, local_range),
          [=](::sycl::nd_item<1> item) {
            LuPivotsToPermutationKernel(item, pivots, permutation, batch_size,
                                        pivot_size, permutation_size);
          });
    });
  });
  if (!status.ok()) {
    LOG(ERROR) << "LaunchLuPivotsToPermutationKernel: " << status.message();
  }
}

namespace {

template <typename T>
void TridiagonalSolvePerturbedKernel(::sycl::nd_item<1> item,
                                     std::int64_t batch_size, int n, int k_rhs,
                                     const T* subdiag, const T* diag,
                                     const T* superdiag, const T* rhs, T* x,
                                     T* workspace) {
  for (std::int64_t idx = item.get_global_id(0); idx < batch_size;
       idx += item.get_global_range(0)) {
    T* u_workspace = workspace + idx * (n * 3 + k_rhs);
    T* rhs_row_workspace = u_workspace + n * 3;
    SolveWithGaussianEliminationWithPivotingAndPerturbSingular<T>(
        n, k_rhs, subdiag + idx * n, diag + idx * n, superdiag + idx * n,
        rhs + idx * n * k_rhs, x + idx * n * k_rhs, u_workspace,
        rhs_row_workspace);
  }
}

template <typename T>
void LaunchTridiagonalSolvePerturbedKernelBody(
    ::sycl::queue& queue, std::int64_t batch_size, int n, int k_rhs,
    const void* subdiag, const void* diag, const void* superdiag,
    const void* rhs, void* x, void* workspace) {
  const int local_range = 128;  // work-items per work-group
  const std::int64_t num_work_groups = std::min<std::int64_t>(
      1024, (batch_size + local_range - 1) / local_range);

  queue.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(
        ::sycl::nd_range<1>(num_work_groups * local_range, local_range),
        [=](::sycl::nd_item<1> item) {
          TridiagonalSolvePerturbedKernel<T>(
              item, batch_size, n, k_rhs, static_cast<const T*>(subdiag),
              static_cast<const T*>(diag), static_cast<const T*>(superdiag),
              static_cast<const T*>(rhs), static_cast<T*>(x),
              static_cast<T*>(workspace));
        });
  });
}

}  // namespace

void LaunchTridiagonalSolvePerturbedKernel(
    gpuStream_t stream, std::int64_t batch_size, int n, int k_rhs,
    xla::ffi::DataType dtype, const void* subdiag, const void* diag,
    const void* superdiag, const void* rhs, void* x, void* workspace) {
  absl::Status status = TryCatchToStatus([&] {
    switch (dtype) {
      case xla::ffi::DataType::F32:
        LaunchTridiagonalSolvePerturbedKernelBody<float>(
            *stream, batch_size, n, k_rhs, subdiag, diag, superdiag, rhs, x,
            workspace);
        break;
      case xla::ffi::DataType::F64:
        LaunchTridiagonalSolvePerturbedKernelBody<double>(
            *stream, batch_size, n, k_rhs, subdiag, diag, superdiag, rhs, x,
            workspace);
        break;
      case xla::ffi::DataType::C64:
        LaunchTridiagonalSolvePerturbedKernelBody<std::complex<float>>(
            *stream, batch_size, n, k_rhs, subdiag, diag, superdiag, rhs, x,
            workspace);
        break;
      case xla::ffi::DataType::C128:
        LaunchTridiagonalSolvePerturbedKernelBody<std::complex<double>>(
            *stream, batch_size, n, k_rhs, subdiag, diag, superdiag, rhs, x,
            workspace);
        break;
      default:
        break;
    }
  });
  if (!status.ok()) {
    LOG(ERROR) << "LaunchTridiagonalSolvePerturbedKernel: " << status.message();
  }
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
