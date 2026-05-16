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

#ifndef JAXLIB_TRIDIAGONAL_SOLVE_PERTURBED_H_
#define JAXLIB_TRIDIAGONAL_SOLVE_PERTURBED_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

#include "Eigen/Core"

#if defined(__CUDACC__) || defined(JAX_GPU_CUDA)
#include <cuda/std/complex>

namespace Eigen {
template <>
struct NumTraits<::cuda::std::complex<float>>
    : GenericNumTraits<::cuda::std::complex<float>> {
  typedef float Real;
  enum {
    IsComplex = 1,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 2,
    AddCost = 2,
    MulCost = 4
  };
};

template <>
struct NumTraits<::cuda::std::complex<double>>
    : GenericNumTraits<::cuda::std::complex<double>> {
  typedef double Real;
  enum {
    IsComplex = 1,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 2,
    AddCost = 2,
    MulCost = 4
  };
};
}  // namespace Eigen
#endif

namespace jax {

// Adjust pivot such that neither 'rhs[i,:] / pivot' nor '1 / pivot' cause
// overflow, where i numerates the multiple right-hand-sides. During the
// back-substitution phase in
// SolveWithGaussianEliminationWithPivotingAndPerturbSingular, we compute
// the i'th row of the solution as rhs[i,:] * (1 / pivot). This logic is
// extracted from the LAPACK routine xLAGTS.
template <typename Scalar, typename Derived>
EIGEN_DEVICE_FUNC void MaybePerturbPivot(
    typename Eigen::NumTraits<Scalar>::Real perturb, Scalar& pivot,
    Eigen::DenseBase<Derived>& rhs_row) {
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using std::real;
  constexpr RealScalar one(1.0);
  // std::numeric_limits<T>::min/max/epsilon are constexpr in C++11+,
  // so these constants are evaluated at compile time.
  constexpr RealScalar tiny = std::numeric_limits<RealScalar>::min();
  constexpr RealScalar small = one / std::numeric_limits<RealScalar>::max();
  // The following logic is extracted from xLAMCH in LAPACK.
  constexpr RealScalar safemin =
      (small < tiny
           ? tiny
           : small * (one + std::numeric_limits<RealScalar>::epsilon()));
  constexpr RealScalar bignum = one / safemin;

// EVOLVE-BLOCK-START
  RealScalar abs_pivot = Eigen::numext::abs(pivot);
// EVOLVE-BLOCK-START
  if (abs_pivot >= one) {
    return;
  }
// EVOLVE-BLOCK-END

  // Safeguard against infinite loop if 'perturb' is zero.
  // 'perturb' should never have magnitude smaller than safemin.
// EVOLVE-BLOCK-START
  perturb = std::max(Eigen::numext::abs(perturb), safemin);
  // Make sure perturb and pivot have the same sign.
  perturb = std::copysign(perturb, real(pivot));
// EVOLVE-BLOCK-END

  bool stop = false;
// EVOLVE-BLOCK-START
  const RealScalar max_factor = rhs_row.derived().array().abs().maxCoeff();
// EVOLVE-BLOCK-END

  while (abs_pivot < one && !stop) {
// EVOLVE-BLOCK-START
    if (abs_pivot < safemin) {
      if (abs_pivot == RealScalar(0.0) || max_factor * safemin > abs_pivot) {
        pivot += perturb;
        perturb *= RealScalar(2.0);
      } else {
        pivot *= bignum;
        rhs_row *= bignum;
        stop = true;
      }
    } else if (max_factor > abs_pivot * bignum) {
      pivot += perturb;
      perturb *= RealScalar(2.0);
    } else {
      stop = true;
    }
// EVOLVE-BLOCK-END
    abs_pivot = Eigen::numext::abs(pivot);
  }
// EVOLVE-BLOCK-END
}

// This function roughly follows LAPACK's xLAGTF + xLAGTS routines.
//
// It computes the solution to the a linear system with multiple
// right-hand sides
//     T * X = RHS
// where T is a tridiagonal matrix using a row-pivoted LU decomposition.
//
// This routine differs from SolveWithGaussianEliminationWithPivoting by
// allowing the tridiagonal matrix to be numerically singular.
// If tiny diagonal elements of U are encountered, signaling that T is
// numerically singular, the diagonal elements are perturbed by
// an amount proportional to eps*max_abs_u to avoid overflow, where
// max_abs_u is max_{i,j} | U(i,j) |. This is useful when using this
// routine for computing eigenvectors of a matrix T' via inverse
// iteration by solving the singular system
//   (T' - lambda*I) X = RHS,
// where lambda is an eigenvalue of T'.
//
// By fusing the factorization and solution, we avoid storing L
// and pivoting information, and the forward solve is done on-the-fly
// during factorization, instead of requiring a separate loop.
template <typename Scalar>
EIGEN_DEVICE_FUNC void
SolveWithGaussianEliminationWithPivotingAndPerturbSingular(
    int n, int k_rhs, const Scalar* subdiag_ptr, const Scalar* diag_ptr,
    const Scalar* superdiag_ptr, const Scalar* rhs_ptr, Scalar* x_ptr,
    Scalar* u_workspace, Scalar* rhs_row_workspace) {
// EVOLVE-BLOCK-START
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using InputMatrixMap =
      Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::ColMajor>>;
  using OutputMatrixMap = Eigen::Map<
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;
  using ArrayMap = Eigen::Map<const Eigen::Array<Scalar, Eigen::Dynamic, 1>>;
  using RowMatrixMap =
      Eigen::Map<Eigen::Matrix<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor>>;

  InputMatrixMap rhs(rhs_ptr, n, k_rhs);
  OutputMatrixMap x(x_ptr, n, k_rhs);
  ArrayMap subdiag(subdiag_ptr, n);
  ArrayMap diag(diag_ptr, n);
  ArrayMap superdiag(superdiag_ptr, n);
  RowMatrixMap rhs_row(rhs_row_workspace, 1, k_rhs);

  const Scalar zero = Scalar(0.0);
  using std::real;
  const RealScalar realzero = RealScalar(0.0);

  if (n == 0) {
    return;
  }

// EVOLVE-BLOCK-START
  if (n == 1) {
    Scalar p = diag(0);
    RealScalar eps = std::numeric_limits<RealScalar>::epsilon();
    RealScalar perturb = eps * Eigen::numext::abs(p);
    rhs_row = rhs.row(0);
    MaybePerturbPivot(perturb, p, rhs_row);
    x.row(0) = rhs_row / p;
    return;
  }
// EVOLVE-BLOCK-END

  Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, 3>> u(u_workspace, n, 3);
  u.setZero();

  u(0, 0) = diag(0);
  u(0, 1) = superdiag(0);
  RealScalar max_abs_u =
      std::max(Eigen::numext::abs(u(0, 0)), Eigen::numext::abs(u(0, 1)));
  RealScalar scale1 = Eigen::numext::abs(u(0, 0)) + Eigen::numext::abs(u(0, 1));
  x.row(0) = rhs.row(0);

// EVOLVE-BLOCK-START
  for (int k = 0; k < n - 1; ++k) {
    u(k + 1, 0) = diag(k + 1);
    RealScalar scale2 =
        Eigen::numext::abs(subdiag(k + 1)) + Eigen::numext::abs(u(k + 1, 0));
    if (k < n - 2) scale2 += Eigen::numext::abs(superdiag(k + 1));

    if (subdiag(k + 1) == zero) {
      scale1 = scale2;
      u(k + 1, 1) = superdiag(k + 1);
      u(k, 2) = zero;
      x.row(k + 1) = rhs.row(k + 1);
    } else {
// EVOLVE-BLOCK-START
      const RealScalar piv1 =
          u(k, 0) == zero ? realzero : Eigen::numext::abs(u(k, 0)) / scale1;
      const RealScalar piv2 = Eigen::numext::abs(subdiag(k + 1)) / scale2;
// EVOLVE-BLOCK-END

      if (piv2 <= piv1) {
// EVOLVE-BLOCK-START
        scale1 = scale2;
        const Scalar factor = subdiag(k + 1) / u(k, 0);
        u(k + 1, 0) -= factor * u(k, 1);
        u(k + 1, 1) = superdiag(k + 1);
        u(k, 2) = zero;
        x.row(k + 1) = rhs.row(k + 1) - factor * x.row(k);
// EVOLVE-BLOCK-END
      } else {
// EVOLVE-BLOCK-START
        const Scalar factor = u(k, 0) / subdiag(k + 1);
        const Scalar utmp = u(k, 1);
        u(k, 0) = subdiag(k + 1);
        u(k + 1, 0) = utmp - factor * diag(k + 1);
        u(k, 1) = diag(k + 1);
        if (k < n - 2) {
          u(k, 2) = superdiag(k + 1);
          u(k + 1, 1) = -factor * superdiag(k + 1);
        }

        x.row(k + 1) = x.row(k) - factor * rhs.row(k + 1);
        x.row(k) = rhs.row(k + 1);

        scale1 =
            Eigen::numext::abs(u(k + 1, 0)) + Eigen::numext::abs(u(k + 1, 1));
// EVOLVE-BLOCK-END
      }
    }
    if (k < n - 2) {
// EVOLVE-BLOCK-START
      for (int i = 0; i < 3; ++i) {
        max_abs_u = std::max(max_abs_u, Eigen::numext::abs(u(k, i)));
      }
// EVOLVE-BLOCK-END
    }
  }
// EVOLVE-BLOCK-END
  max_abs_u = std::max(max_abs_u, Eigen::numext::abs(u(n - 1, 0)));

// EVOLVE-BLOCK-START
  RealScalar eps = std::numeric_limits<RealScalar>::epsilon();
  RealScalar perturb = eps * max_abs_u;
// EVOLVE-BLOCK-END

// EVOLVE-BLOCK-START
  Scalar p = u(n - 1, 0);
  rhs_row = x.row(n - 1);
  MaybePerturbPivot(perturb, p, rhs_row);
  x.row(n - 1) = rhs_row * (Scalar(1.0) / p);
// EVOLVE-BLOCK-END

  if (n > 1) {
// EVOLVE-BLOCK-START
    p = u(n - 2, 0);
    rhs_row = x.row(n - 2) - u(n - 2, 1) * x.row(n - 1);
    MaybePerturbPivot(std::copysign(perturb, real(p)), p, rhs_row);
    x.row(n - 2) = rhs_row * (Scalar(1.0) / p);
// EVOLVE-BLOCK-END
  }

// EVOLVE-BLOCK-START
  for (int k = n - 3; k >= 0; --k) {
    p = u(k, 0);
// EVOLVE-BLOCK-START
    rhs_row = x.row(k) - u(k, 1) * x.row(k + 1) - u(k, 2) * x.row(k + 2);
    MaybePerturbPivot(std::copysign(perturb, real(p)), p, rhs_row);
// EVOLVE-BLOCK-END
    x.row(k) = rhs_row * (Scalar(1.0) / p);
  }
// EVOLVE-BLOCK-END
// EVOLVE-BLOCK-END
}

}  // namespace jax

#endif  // JAXLIB_TRIDIAGONAL_SOLVE_PERTURBED_H_
