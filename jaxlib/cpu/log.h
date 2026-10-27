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

// Shared Highway SIMD natural logarithm helpers for CPU math FFI kernels,
// using constants from `jaxlib/log_coeffs.h`:
//   - `LogF64HiLo`: `(hi, lo)` `ln(x) + k_adj * ln(2)` (~64-bit accuracy,
//     matching `_log_f64` in `jax/_src/lax/lgamma.py`).
//   - `LogF64`: single-word `float64` `ln(x)` (~0.501 ULP), used by the
//     `float32`-via-`float64` kernels.
//   - `LogF2WithRecip`: full double-word `(hi, lo)` `ln(z0 + z1)` (~108-bit
//     accuracy, matching `log_f2` in `jax/_src/lax/double_word.py`).

#ifndef JAXLIB_CPU_LOG_H_
#define JAXLIB_CPU_LOG_H_

#include "jaxlib/log_coeffs.h"

#endif  // JAXLIB_CPU_LOG_H_

// Per-target SIMD definitions (re-included once per Highway target when
// `foreach_target.h` toggles `HWY_TARGET_TOGGLE`).
#if defined(JAXLIB_CPU_LOG_TOGGLE_) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef JAXLIB_CPU_LOG_TOGGLE_
#undef JAXLIB_CPU_LOG_TOGGLE_
#else
#define JAXLIB_CPU_LOG_TOGGLE_
#endif

#include "hwy/highway.h"
#include "jaxlib/cpu/double_word.h"

HWY_BEFORE_NAMESPACE();
namespace jax {
namespace cpu_math {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
namespace mc = jax::math_coeffs;

// Computes `ln(x) + k_adj * ln(2)` for normal `x > 0` as a `(log_hi, log_lo)`
// pair with ~64 bits of accuracy, matching `_log_f64` in `lgamma.py`.
// `k_adj` is an integer exponent offset (used when the caller pre-scaled tiny
// inputs by `2^54` to avoid underflow and needs `-54 * ln(2)` added back).
template <class D, class V>
HWY_INLINE V LogF64HiLo(D d, V x, V k_adj, V& out_lo) {
  const hn::RebindToSigned<D> di64;
  // Split `x = 2^exp_off * m` with `m` centered in `[sqrt(2)/2, sqrt(2))`
  // (`[0.7071, 1.4142)`) rather than `[1, 2)` so `|m - 1| <= 0.4142`.
  const auto bits = hn::BitCast(di64, x);
  const auto exp_off =
      hn::ShiftRight<52>(hn::Sub(bits, hn::Set(di64, mc::kSqrt2Over2BitsF64)));
  const V k = hn::Add(SmallInt64ToDouble(d, exp_off), k_adj);
  const auto mant_bits = hn::Sub(bits, hn::ShiftLeft<52>(exp_off));
  const V m = hn::BitCast(d, mant_bits);

  // Because `m` lies in `[0.7071, 1.4142)`, subtracting `1.0` loses no low
  // bits (`f = m - 1` is exact). Evaluate `ln(1 + f)` as:
  //   (f - 0.5 * f^2) + f^3 * (1/3 + f * Q(f))
  // keeping `(f - 0.5 * f^2)` and `f^3 * (1/3 + f * Q(f))` in `(hi, lo)` form
  // so cancellation near `f = 0` or `k * ln(2) + ln(m) ~= 0` does not lose
  // precision.
  const V f = hn::Sub(m, hn::Set(d, 1.0));
  V f2_l;
  const V f2_h = TwoProd(f, f, f2_l);

  const V neg_half = hn::Set(d, -0.5);
  V tl;
  const V th = FastTwoSum(f, hn::Mul(neg_half, f2_h), tl);
  tl = hn::MulAdd(neg_half, f2_l, tl);

  // `0.3333333333333333 + 1.850371707708594e-17` is the `(hi, lo)` split of
  // `1/3`, the leading `f^3` coefficient of `ln(1 + f)`.
  const V poly = Horner(d, f, mc::kLogQF64);
  V ql;
  const V qh = FastTwoSum(hn::Set(d, 0.3333333333333333), hn::Mul(f, poly), ql);
  ql = hn::Add(ql, hn::Set(d, 1.850371707708594e-17));

  V f3_e;
  const V f3_h = TwoProd(f2_h, f, f3_e);
  const V f3_l = hn::MulAdd(f2_l, f, f3_e);
  V p_e;
  const V ph = TwoProd(f3_h, qh, p_e);
  const V pl = hn::MulAdd(f3_h, ql, hn::MulAdd(f3_l, qh, p_e));

  // Combine `ln(m) = (th + tl) + (ph + pl)` with `k * ln(2)`. Because `|k| <=
  // 1024` has at most 11 significant bits and `kLn2HiF64` has 11 trailing zero
  // mantissa bits, `k * kLn2HiF64` has zero rounding error.
  V sl;
  const V sh = F2AddOrdered(th, tl, ph, pl, sl);
  const V kh = hn::Mul(k, hn::Set(d, mc::kLn2HiF64));
  const V kl = hn::Mul(k, hn::Set(d, mc::kLn2LoF64));
  return F2AddOrdered(kh, kl, sh, sl, out_lo);
}

// Computes `ln(x)` in `float64` to within 0.501 ULP for normal `x > 0`, used by
// the `float32`-via-`float64` kernels.
template <class D, class V>
HWY_INLINE V LogF64(D d, V x) {
  V lo;
  const V hi = LogF64HiLo(d, x, hn::Zero(d), lo);
  return hn::Add(hi, lo);
}

// Computes `ln(z0 + z1)` to ~108-bit `(hi, lo)` accuracy given a precomputed
// reciprocal `(iz0, iz1) ~= 1 / (z0 + z1)`, matching `log_f2` in
// `jax/_src/lax/double_word.py`.
//
// Range reduction writes `z0 = 2^k * m` with `m` in `[sqrt(2)/2, sqrt(2))`,
// then multiplies `m` by a 4-entry step factor `r` in `{1.25, 1.0625, 0.9375,
// 0.75}` so `m * r` lies in `[0.88388, 1.11488]` (within ~12% of `1`). Setting
// `f = m * r - 1` and `s = f / (2 + f)` shrinks `w = s^2` to `<= 0.0038`, where
// the odd series
//   ln(1 + f) = 2 * s * (1 + w/3 + w^2/5 + ... + w^10/21)
// reaches ~108-bit accuracy in 10 polynomial terms (`kLogPoly*F64`). Finally,
// the low word `z1` contributes `ln(1 + z1 / z0) ~= z1 * (1 / z)` via the
// caller's reciprocal `(iz0, iz1)`.
template <class D, class V>
HWY_INLINE V LogF2WithRecip(D d, V z0, V z1, V iz0, V iz1, V& out1) {
  const hn::RebindToSigned<D> di64;
  const V zero = hn::Zero(d);
  const V one = hn::Set(d, 1.0);
  const V two = hn::Set(d, 2.0);

  const auto z_bits = hn::BitCast(di64, z0);
  const auto exp_off = hn::ShiftRight<52>(
      hn::Sub(z_bits, hn::Set(di64, mc::kSqrt2Over2BitsF64)));
  const V k = SmallInt64ToDouble(d, exp_off);
  const V m = hn::BitCast(d, hn::Sub(z_bits, hn::ShiftLeft<52>(exp_off)));

  // Pick the step factor `r` for the sub-interval containing `m`, split at
  // `2^(-1/4) ~= 0.8409`, `1.0`, and `2^(1/4) ~= 1.1892`, along with its
  // precomputed `-ln(r)` in `(lr_h, lr_l)`.
  const auto b1 = hn::Ge(m, one);
  const V thresh = hn::IfThenElse(b1, hn::Set(d, 1.189207115002721),
                                  hn::Set(d, 0.8408964152537145));
  const auto b0 = hn::Ge(m, thresh);

  const V r = hn::IfThenElse(
      b1, hn::IfThenElse(b0, hn::Set(d, 0.75), hn::Set(d, 0.9375)),
      hn::IfThenElse(b0, hn::Set(d, 1.0625), hn::Set(d, 1.25)));
  const V lr_h =
      hn::IfThenElse(b1,
                     hn::IfThenElse(b0, hn::Set(d, mc::kNegLnr4HiF64[3]),
                                    hn::Set(d, mc::kNegLnr4HiF64[2])),
                     hn::IfThenElse(b0, hn::Set(d, mc::kNegLnr4HiF64[1]),
                                    hn::Set(d, mc::kNegLnr4HiF64[0])));
  const V lr_l =
      hn::IfThenElse(b1,
                     hn::IfThenElse(b0, hn::Set(d, mc::kNegLnr4LoF64[3]),
                                    hn::Set(d, mc::kNegLnr4LoF64[2])),
                     hn::IfThenElse(b0, hn::Set(d, mc::kNegLnr4LoF64[1]),
                                    hn::Set(d, mc::kNegLnr4LoF64[0])));

  // Form `f = m * r - 1` as `(fh, fl)`. Because `mr_h` is in `[0.88, 1.12]`,
  // `mr_h - 1.0` loses no bits, and either `mr_h - 1.0 == 0` or `|mr_h - 1.0|
  // >= 2^-53 >= |mr_l|`, so `FastTwoSum` captures `fh + fl` without loss.
  V mr_l;
  const V mr_h = TwoProd(m, r, mr_l);
  V fl;
  const V fh = FastTwoSum(hn::Sub(mr_h, one), mr_l, fl);

  // Accumulate `k * ln(2) - ln(r)` using the 3-part split `kLn2SplitF64` (top
  // two chunks have 12 trailing zero bits so multiplying by `|k| <= 1024` has
  // zero rounding error).
  const V kln_h = hn::Mul(k, hn::Set(d, mc::kLn2SplitF64[0]));
  const V kln_m = hn::Mul(k, hn::Set(d, mc::kLn2SplitF64[1]));
  V kln_e;
  V kln_s = FastTwoSum(kln_h, kln_m, kln_e);
  kln_e = hn::MulAdd(k, hn::Set(d, mc::kLn2SplitF64[2]), kln_e);
  kln_s = F2Add(kln_s, kln_e, lr_h, lr_l, kln_e);

  // Compute `s = f / (2 + f)` and `w = s^2`, then evaluate `2 * s * (1 + w *
  // P(w))`.
  V d1;
  const V d0 = FastTwoSum(two, fh, d1);
  d1 = hn::Add(d1, fl);
  V s1;
  const V s0 = F2Div(d, fh, fl, d0, d1, s1);
  V w1;
  const V w0 = F2Square(s0, s1, w1);

  V lp1;
  V lp0 = HornerHybridF2(d, w0, w1, mc::kLogPolyLoDegHiF64,
                         mc::kLogPolyLoDegLoF64, mc::kLogPolyHiDegF64, lp1);
  lp0 = F2Mul(lp0, lp1, w0, w1, lp1);
  lp0 = F2AddOrdered(two, zero, lp0, lp1, lp1);
  lp0 = F2Mul(lp0, lp1, s0, s1, lp1);

  // Add the low-word correction `z1 / z` and combine with `k * ln(2) - ln(r)`.
  V zc_corr1;
  const V zc_corr0 = F2MulScalar(iz0, iz1, z1, zc_corr1);
  lp0 = F2AddOrdered(lp0, lp1, zc_corr0, zc_corr1, lp1);
  return F2Add(kln_s, kln_e, lp0, lp1, out1);
}

}  // namespace HWY_NAMESPACE
}  // namespace cpu_math
}  // namespace jax
HWY_AFTER_NAMESPACE();

#endif  // JAXLIB_CPU_LOG_TOGGLE_
