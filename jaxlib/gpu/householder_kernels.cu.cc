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

// This file contains kernels that compute the product of elementary Householder
// reflectors, as applied to the identity matrix (LAPACK: orgqr), or to a user
// chosen matrix (LAPACK: orgmr). These kernels should only be used for batches
// of small matrices. Above a certain size we should expect blocked
// implementations of these operations to be faster. cusolver provides a
// suitable unbatched implementation, but lacks a batched implementations, hence
// these kernels.

#include "jaxlib/gpu/householder_kernels.h"

#include <complex>

#define EIGEN_USE_GPU
#include <Eigen/Core>

#include "jaxlib/gpu/vendor.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace {

using Eigen::numext::conj;
#if defined(EIGEN_USING_STD_COMPLEX_OPERATORS)
EIGEN_USING_STD_COMPLEX_OPERATORS
#endif

// Applies the product of elementary reflectors from the left (parallelized over
// columns) to c, which is updated in-place. For orgqr=true, we start by
// initializing c to the identity matrix, whereas for orgqr=false (ormqr) we
// multiply a user-chosen matrix.
// a, tau, and c_data are all batch-major. The matrices must be
// column major within each batch element.
template <typename T, bool IsOrgqr = false>
__global__ void ProductOfElementaryHouseholderReflectorsSmallBatchedLeftKernel(
    int batch, int m, int n, int k, int lda, int64_t a_stride, const T* a_data,
    const T* tau_data, T* c_data, bool transpose) {
#if defined(EIGEN_GPU_COMPILE_PHASE)
  int batch_idx = blockIdx.x;
  if (batch_idx >= batch) return;

  // Each thread computes a single column of the output matrix (col_idx).
  int col_idx = threadIdx.x;
  if (col_idx >= n) return;

  const T* a = a_data + batch_idx * a_stride;
  const T* tau = tau_data + batch_idx * k;
  T* c = c_data + batch_idx * m * n;  // c is m x n, ldc = m

  if constexpr (IsOrgqr) {
    for (int r = 0; r < m; ++r) {
      c[r + col_idx * m] = (r == col_idx) ? T(1) : T(0);
    }
  }

  for (int step = 0; step < k; ++step) {
    // H_1...H_k * C applies H_k first (reverse), H_k^H...H_1^H * C applies
    // H_1^H first (forward)
    int i = transpose ? step : k - 1 - step;

    T t = tau[i];
    if (transpose) {
      t = Eigen::numext::conj(t);
    }

    // From $H C = C - \tau v (v^H C)$, let the row vector $W = v^H C$.
    // This thread's scalar w is the element W[col], i.e., w = v^H C[:, col].
    // Recall the 'v' vectors are packed into the lower triangle of `a`.
    T w = c[i + col_idx * m];  // Implicit leading `1` coefficient.
    for (int r = i + 1; r < m; ++r) {
      // $w \leftarrow w + \overline{A_{r, i}} C_{r, \text{col}}$
      w = w + Eigen::numext::conj(a[r + i * lda]) * c[r + col_idx * m];
    }

    T t_w = t * w;
    // The outer product update is $C \leftarrow C - \tau v W$, where $W = v^H
    // C$. For a specific column 'col', the update for row $r$ is $C_{r, col}
    // \leftarrow C_{r, col} - t_w v_r$. For $r < i$, $v_r = 0$ (no update). For
    // $r = i$, $v_i = 1$:
    c[i + col_idx * m] = c[i + col_idx * m] - t_w;
    // For $r > i$, $v_r = A_{r, i}$:
    for (int r = i + 1; r < m; ++r) {
      c[r + col_idx * m] = c[r + col_idx * m] - a[r + i * lda] * t_w;
    }
  }
#endif
}

// Applies the product of elementary reflectors from the right (parallelized
// over rows) to c, which is updated in-place. a, tau, and c_data are all
// batch-major. The matrices must be column major within each batch element.
// This kernel is similar to the left-sided kernel, except that each thread
// computes a single row of the output.
template <typename T>
__global__ void ProductOfElementaryHouseholderReflectorsSmallBatchedRightKernel(
    int batch, int m, int n, int k, int lda, int64_t a_stride, const T* a_data,
    const T* tau_data, T* c_data, bool transpose) {
#if defined(EIGEN_GPU_COMPILE_PHASE)
  int batch_idx = blockIdx.x;
  if (batch_idx >= batch) return;

  int row_idx = threadIdx.x;
  if (row_idx >= m) return;

  const T* a = a_data + batch_idx * a_stride;

  const T* tau = tau_data + batch_idx * k;
  T* c = c_data + batch_idx * m * n;  // c is m x n, ldc = m

  for (int step = 0; step < k; ++step) {
    // C * H_1...H_k applies H_1 first (forward), C * H_k^H...H_1^H applies
    // H_k^H first (reverse)
    int i = transpose ? k - 1 - step : step;

    T t = tau[i];
    if (transpose) {
      t = Eigen::numext::conj(t);
    }

    // From $C H = C - \tau (C v) v^H$, let the column vector $U = C v$.
    T u = c[row_idx + i * m];
    for (int col = i + 1; col < n; ++col) {
      // $u \leftarrow u + C_{\text{row}, c} A_{c, i}$
      u = u + c[row_idx + col * m] * a[col + i * lda];
    }

    T u_t = u * t;
    // The row update is $c_{row} \leftarrow c_{row} - u_t v^H$. Since $v[i] =
    // 1$ and $v[col > i] = A_{col, i}$: $C_{\text{row}, i} \leftarrow
    // C_{\text{row}, i} - u_t$
    c[row_idx + i * m] = c[row_idx + i * m] - u_t;
    for (int col = i + 1; col < n; ++col) {
      // $C_{\text{row}, col} \leftarrow C_{\text{row}, col} - u_t
      // \overline{A_{col, i}}$
      c[row_idx + col * m] =
          c[row_idx + col * m] - u_t * Eigen::numext::conj(a[col + i * lda]);
    }

    // Each thread works on its own row.
  }
#endif
}

}  // namespace

template <typename T>
gpuError_t LaunchOrmqrSmallBatchedKernel(gpuStream_t stream, int batch, int m,
                                         int n, int k, int lda,
                                         int64_t a_stride, const T* a,
                                         const T* tau, T* out, bool left,
                                         bool transpose) {
  if (batch == 0) {
    return gpuSuccess;
  }

  if (left) {
    int threadsPerBlock = n;  // Parallelize over columns

    ProductOfElementaryHouseholderReflectorsSmallBatchedLeftKernel<T>
        <<<batch, threadsPerBlock, 0, stream>>>(batch, m, n, k, lda, a_stride,
                                                a, tau, out, transpose);
  } else {
    int threadsPerBlock = m;  // Parallelize over rows

    ProductOfElementaryHouseholderReflectorsSmallBatchedRightKernel<T>
        <<<batch, threadsPerBlock, 0, stream>>>(batch, m, n, k, lda, a_stride,
                                                a, tau, out, transpose);
  }
  return gpuGetLastError();
}

template <typename T>
gpuError_t LaunchOrgqrSmallBatchedKernel(gpuStream_t stream, int batch, int m,
                                         int n, int k, int lda,
                                         int64_t a_stride, const T* a,
                                         const T* tau, T* out) {
  if (batch == 0) {
    return gpuSuccess;
  }

  int threadsPerBlock = n;  // Parallelize over columns

  ProductOfElementaryHouseholderReflectorsSmallBatchedLeftKernel<T, true>
      <<<batch, threadsPerBlock, 0, stream>>>(batch, m, n, k, lda, a_stride, a,
                                              tau, out, /*transpose=*/false);
  return gpuGetLastError();
}

/// Explicit instantiations for real types
template gpuError_t LaunchOrmqrSmallBatchedKernel<float>(
    gpuStream_t stream, int batch, int m, int n, int k, int lda,
    int64_t a_stride, const float* a, const float* tau, float* out, bool left,
    bool transpose);

template gpuError_t LaunchOrmqrSmallBatchedKernel<double>(
    gpuStream_t stream, int batch, int m, int n, int k, int lda,
    int64_t a_stride, const double* a, const double* tau, double* out,
    bool left, bool transpose);

template <>
gpuError_t LaunchOrmqrSmallBatchedKernel<gpuComplex>(
    gpuStream_t stream, int batch, int m, int n, int k, int lda,
    int64_t a_stride, const gpuComplex* a, const gpuComplex* tau,
    gpuComplex* out, bool left, bool transpose) {
  if (batch == 0) return gpuSuccess;
  if (left) {
    int threadsPerBlock = n;
    ProductOfElementaryHouseholderReflectorsSmallBatchedLeftKernel<
        std::complex<float>><<<batch, threadsPerBlock, 0, stream>>>(
        batch, m, n, k, lda, a_stride,
        reinterpret_cast<const std::complex<float>*>(a),
        reinterpret_cast<const std::complex<float>*>(tau),
        reinterpret_cast<std::complex<float>*>(out), transpose);
  } else {
    int threadsPerBlock = m;
    ProductOfElementaryHouseholderReflectorsSmallBatchedRightKernel<
        std::complex<float>><<<batch, threadsPerBlock, 0, stream>>>(
        batch, m, n, k, lda, a_stride,
        reinterpret_cast<const std::complex<float>*>(a),
        reinterpret_cast<const std::complex<float>*>(tau),
        reinterpret_cast<std::complex<float>*>(out), transpose);
  }
  return gpuGetLastError();
}

template <>
gpuError_t LaunchOrmqrSmallBatchedKernel<gpuDoubleComplex>(
    gpuStream_t stream, int batch, int m, int n, int k, int lda,
    int64_t a_stride, const gpuDoubleComplex* a, const gpuDoubleComplex* tau,
    gpuDoubleComplex* out, bool left, bool transpose) {
  if (batch == 0) return gpuSuccess;
  if (left) {
    int threadsPerBlock = n;
    ProductOfElementaryHouseholderReflectorsSmallBatchedLeftKernel<
        std::complex<double>><<<batch, threadsPerBlock, 0, stream>>>(
        batch, m, n, k, lda, a_stride,
        reinterpret_cast<const std::complex<double>*>(a),
        reinterpret_cast<const std::complex<double>*>(tau),
        reinterpret_cast<std::complex<double>*>(out), transpose);
  } else {
    int threadsPerBlock = m;
    ProductOfElementaryHouseholderReflectorsSmallBatchedRightKernel<
        std::complex<double>><<<batch, threadsPerBlock, 0, stream>>>(
        batch, m, n, k, lda, a_stride,
        reinterpret_cast<const std::complex<double>*>(a),
        reinterpret_cast<const std::complex<double>*>(tau),
        reinterpret_cast<std::complex<double>*>(out), transpose);
  }
  return gpuGetLastError();
}

template gpuError_t LaunchOrgqrSmallBatchedKernel<float>(
    gpuStream_t stream, int batch, int m, int n, int k, int lda,
    int64_t a_stride, const float* a, const float* tau, float* out);

template gpuError_t LaunchOrgqrSmallBatchedKernel<double>(
    gpuStream_t stream, int batch, int m, int n, int k, int lda,
    int64_t a_stride, const double* a, const double* tau, double* out);

template <>
gpuError_t LaunchOrgqrSmallBatchedKernel<gpuComplex>(
    gpuStream_t stream, int batch, int m, int n, int k, int lda,
    int64_t a_stride, const gpuComplex* a, const gpuComplex* tau,
    gpuComplex* out) {
  if (batch == 0) return gpuSuccess;
  int threadsPerBlock = n;
  ProductOfElementaryHouseholderReflectorsSmallBatchedLeftKernel<
      std::complex<float>, true><<<batch, threadsPerBlock, 0, stream>>>(
      batch, m, n, k, lda, a_stride,
      reinterpret_cast<const std::complex<float>*>(a),
      reinterpret_cast<const std::complex<float>*>(tau),
      reinterpret_cast<std::complex<float>*>(out), /*transpose=*/false);
  return gpuGetLastError();
}

template <>
gpuError_t LaunchOrgqrSmallBatchedKernel<gpuDoubleComplex>(
    gpuStream_t stream, int batch, int m, int n, int k, int lda,
    int64_t a_stride, const gpuDoubleComplex* a, const gpuDoubleComplex* tau,
    gpuDoubleComplex* out) {
  if (batch == 0) return gpuSuccess;
  int threadsPerBlock = n;
  ProductOfElementaryHouseholderReflectorsSmallBatchedLeftKernel<
      std::complex<double>, true><<<batch, threadsPerBlock, 0, stream>>>(
      batch, m, n, k, lda, a_stride,
      reinterpret_cast<const std::complex<double>*>(a),
      reinterpret_cast<const std::complex<double>*>(tau),
      reinterpret_cast<std::complex<double>*>(out), /*transpose=*/false);
  return gpuGetLastError();
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
