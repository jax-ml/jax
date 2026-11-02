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

// Shared GPU device natural logarithm helpers using constants from
// `jaxlib/log_coeffs.h`:
//   - `LogF32HiLo`: `(hi, lo)` `float32` `ln(x)` (~31-bit accuracy, matching
//     `_log_f32` in `jax/_src/lax/lgamma.py`).
//   - `LogF64HiLo` / `LogF64`: `(hi, lo)` and single-word `float64` `ln(x)`
//     (~64-bit internal accuracy, matching `_log_f64` in `lgamma.py`).
//   - `LogF2WithRecip`: full double-word `(hi, lo)` `ln(z0 + z1)` in `float`
//     (~49 bits) or `double` (~108 bits), matching `log_f2` in
//     `jax/_src/lax/double_word.py`.

#ifndef JAXLIB_GPU_LOG_CU_H_
#define JAXLIB_GPU_LOG_CU_H_

#include <cstdint>
#include <type_traits>

#include "jaxlib/gpu/double_word.cu.h"
#include "jaxlib/log_coeffs.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace math_internal {

// Computes `ln(x)` for normal `x > 0` as a `(log_hi, log_lo)` `float32` pair,
// matching `_log_f32` in `jax/_src/lax/lgamma.py`.
// Splits `x = 2^k * m` with `m` in `[sqrt(2)/2, sqrt(2))` so `f = m - 1` has
// `|f| <= 0.4142` and loses no bits when subtracting `1.0f`, then evaluates
// `ln(1 + f) = (f - 0.5 * f^2) + f^3 * Q(f)` and adds `k * ln(2)`. Because
// `|k| <= 128` fits in 8 bits and `kLn2HiF32` has 8 trailing zero mantissa
// bits, `k * kLn2HiF32` has zero rounding error.
__device__ inline float LogF32HiLo(float x, float& out_lo) {
  namespace mc = jax::math_coeffs;
  const int32_t bits = __float_as_int(x);
  const int32_t exp_off = (bits - mc::kSqrt2Over2BitsF32) >> 23;
  const float k = static_cast<float>(exp_off);
  const float m = __int_as_float(bits - (exp_off << 23));

  const float f = SubRn(m, 1.0f);
  const float f2 = MulRn(f, f);
  float tl;
  const float th = FastTwoSum(f, MulRn(-0.5f, f2), tl);
  tl = FmaRn(MulRn(f2, f), Horner(f, mc::kLogQF32), tl);
  float e_d;
  const float log_hi = FastTwoSum(MulRn(k, mc::kLn2HiF32), th, e_d);
  out_lo = AddRn(e_d, FmaRn(k, mc::kLn2LoF32, tl));
  return log_hi;
}

// Computes `ln(x) + k_adj * ln(2)` for normal `x > 0` as a `(log_hi, log_lo)`
// `float64` pair (~64-bit accuracy, matching `_log_f64` in `lgamma.py`).
// `k_adj` is an integer exponent offset (used when the caller pre-scaled tiny
// inputs by `2^54` to avoid subnormal underflow and needs `-54 * ln(2)` added).
__device__ inline double LogF64HiLo(double x, double k_adj, double& out_lo) {
  namespace mc = jax::math_coeffs;
  const int64_t bits = __double_as_longlong(x);
  const int64_t exp_off = (bits - mc::kSqrt2Over2BitsF64) >> 52;
  const double k = AddRn(static_cast<double>(exp_off), k_adj);
  const double m = __longlong_as_double(bits - (exp_off << 52));

  // `m` lies in `[0.7071, 1.4142)`, so `f = m - 1.0` loses no low bits.
  // Evaluate `(f - 0.5 * f^2) + f^3 * (1/3 + f * Q(f))` in `(hi, lo)` form.
  const double f = SubRn(m, 1.0);
  double f2_l;
  const double f2_h = TwoProd(f, f, f2_l);

  double tl;
  const double th = FastTwoSum(f, MulRn(-0.5, f2_h), tl);
  tl = FmaRn(-0.5, f2_l, tl);

  // `0.3333333333333333 + 1.850371707708594e-17` is the `(hi, lo)` split of
  // `1/3` in `ln(1 + f) = f - f^2/2 + f^3 * (1/3 + f * Q(f))`.
  const double poly = Horner(f, mc::kLogQF64);
  double ql;
  const double qh = FastTwoSum(0.3333333333333333, MulRn(f, poly), ql);
  ql = AddRn(ql, 1.850371707708594e-17);

  double f3_e;
  const double f3_h = TwoProd(f2_h, f, f3_e);
  const double f3_l = FmaRn(f2_l, f, f3_e);
  double p_e;
  const double ph = TwoProd(f3_h, qh, p_e);
  const double pl = FmaRn(f3_h, ql, FmaRn(f3_l, qh, p_e));

  double sl;
  const double sh = F2AddOrdered(th, tl, ph, pl, sl);
  const double kh = MulRn(k, mc::kLn2HiF64);
  const double kl = MulRn(k, mc::kLn2LoF64);
  return F2AddOrdered(kh, kl, sh, sl, out_lo);
}

__device__ inline double LogF64(double x) {
  double lo;
  const double hi = LogF64HiLo(x, 0.0, lo);
  return AddRn(hi, lo);
}

// Computes double-word `(hi, lo)` `ln(z0 + z1)` in `T` (`float` or `double`)
// given a precomputed reciprocal `(iz0, iz1) ~= 1 / (z0 + z1)`, matching
// `log_f2` in `jax/_src/lax/double_word.py`.
//
// Range reduction writes `z0 = 2^k * m` with `m` in `[sqrt(2)/2, sqrt(2))`,
// scales `m` by a 4-entry step factor `r` in `{1.25, 1.0625, 0.9375, 0.75}` so
// `m * r` lies in `[0.88388, 1.11488]`, sets `f = m * r - 1` and `s = f /
// (2 + f)` (`w = s^2 <= 0.0038`), evaluates `ln(1 + f) = 2 * s * (1 + w *
// P(w))`, and adds the low-word correction `ln(1 + z1 / z0) ~= z1 * (1 / z)`.
template <typename T>
__device__ T LogF2WithRecip(T z0, T z1, T iz0, T iz1, T& out1) {
  namespace mc = jax::math_coeffs;
  constexpr T kZero = T(0);
  constexpr T kOne = T(1);
  constexpr T kTwo = T(2);

  T k;
  T m;
  if constexpr (std::is_same_v<T, float>) {
    const int32_t z_bits = __float_as_int(z0);
    const int32_t exp_off = (z_bits - mc::kSqrt2Over2BitsF32) >> 23;
    k = static_cast<float>(exp_off);
    m = __int_as_float(z_bits - (exp_off << 23));
  } else {
    const int64_t z_bits = __double_as_longlong(z0);
    const int64_t exp_off = (z_bits - mc::kSqrt2Over2BitsF64) >> 52;
    k = static_cast<double>(exp_off);
    m = __longlong_as_double(z_bits - (exp_off << 52));
  }

  // Select the step factor `r` for `m`'s sub-interval (split at `2^(-1/4) ~=
  // 0.8409`, `1.0`, and `2^(1/4) ~= 1.1892`) and its precomputed `-ln(r)`.
  const bool b1 = m >= kOne;
  const T thresh = b1 ? T(1.189207115002721) : T(0.8408964152537145);
  const bool b0 = m >= thresh;
  const int r_idx = (b1 ? 2 : 0) + (b0 ? 1 : 0);

  constexpr T kRTable[4] = {T(1.25), T(1.0625), T(0.9375), T(0.75)};
  const T r = kRTable[r_idx];
  T lr_h;
  T lr_l;
  if constexpr (std::is_same_v<T, float>) {
    lr_h = mc::kNegLnr4HiF32[r_idx];
    lr_l = mc::kNegLnr4LoF32[r_idx];
  } else {
    lr_h = mc::kNegLnr4HiF64[r_idx];
    lr_l = mc::kNegLnr4LoF64[r_idx];
  }

  // `mr_h` lies in `[0.88, 1.12]`, so `mr_h - 1` loses no bits and `FastTwoSum`
  // captures `f = m * r - 1 = fh + fl` without rounding loss.
  T mr_l;
  const T mr_h = TwoProd(m, r, mr_l);
  T fl;
  const T fh = FastTwoSum(SubRn(mr_h, kOne), mr_l, fl);

  // Accumulate `k * ln(2) - ln(r)` using the 3-part split `kLn2Split*`.
  T kln_s;
  T kln_e;
  if constexpr (std::is_same_v<T, float>) {
    const T kln_h = MulRn(k, mc::kLn2SplitF32[0]);
    const T kln_m = MulRn(k, mc::kLn2SplitF32[1]);
    kln_s = FastTwoSum(kln_h, kln_m, kln_e);
    kln_e = FmaRn(k, mc::kLn2SplitF32[2], kln_e);
  } else {
    const T kln_h = MulRn(k, mc::kLn2SplitF64[0]);
    const T kln_m = MulRn(k, mc::kLn2SplitF64[1]);
    kln_s = FastTwoSum(kln_h, kln_m, kln_e);
    kln_e = FmaRn(k, mc::kLn2SplitF64[2], kln_e);
  }
  kln_s = F2Add(kln_s, kln_e, lr_h, lr_l, kln_e);

  T d1;
  const T d0 = FastTwoSum(kTwo, fh, d1);
  d1 = AddRn(d1, fl);
  T s1;
  const T s0 = F2Div(fh, fl, d0, d1, s1);
  T w1;
  const T w0 = F2Square(s0, s1, w1);

  T lp1;
  T lp0;
  if constexpr (std::is_same_v<T, float>) {
    lp0 = HornerHybridF2(w0, w1, mc::kLogPolyLoDegHiF32, mc::kLogPolyLoDegLoF32,
                         mc::kLogPolyHiDegF32, lp1);
  } else {
    lp0 = HornerHybridF2(w0, w1, mc::kLogPolyLoDegHiF64, mc::kLogPolyLoDegLoF64,
                         mc::kLogPolyHiDegF64, lp1);
  }
  lp0 = F2Mul(lp0, lp1, w0, w1, lp1);
  lp0 = F2AddOrdered(kTwo, kZero, lp0, lp1, lp1);
  lp0 = F2Mul(lp0, lp1, s0, s1, lp1);

  T zc_corr1;
  const T zc_corr0 = F2MulScalar(iz0, iz1, z1, zc_corr1);
  lp0 = F2AddOrdered(lp0, lp1, zc_corr0, zc_corr1, lp1);
  return F2Add(kln_s, kln_e, lp0, lp1, out1);
}

}  // namespace math_internal
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_LOG_CU_H_
