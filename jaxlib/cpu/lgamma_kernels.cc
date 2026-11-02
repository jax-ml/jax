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

// Google Highway CPU FFI kernels for `lgamma(x) = ln(|Gamma(x)|)` on
// `bfloat16`, `float16`, `float32`, and `float64`, accurate to within 1 ULP
// across the full input domain.
//
// Algorithm overview (matching `jax/_src/lax/lgamma.py` with hardware FMA):
//
// 1. `float32` (`EvalLgammaF32Batch`, also shared by `bfloat16` and `float16`
//    via packet-level promotion and rounding):
//    - Promotes `float32` SIMD lanes to `float64` (53-bit mantissa vs. 24-bit
//      in `float32`), leaving ~29 guard bits that absorb intermediate rounding
//      without double-word arithmetic.
//    - For `x >= 0.5`, shifts `x < 8` into `[8, inf)` via
//      `Gamma(x) = Gamma(x + m) / (x * (x + 1) * ... * (x + m - 1))`, evaluates
//      Stirling's series at `z = x + m`, and subtracts `ln(prod)`. Within
//      `3/64` of the zeros at `x = 1` and `x = 2` (where `lgamma(z) - ln(prod)`
//      cancels completely), substitutes a degree-9 Taylor expansion around the
//      zero.
//    - For `x < 0.5`, uses Euler's reflection formula with `n = round(x)` and
//      `delta = x - n` in `[-0.5, 0.5]`:
//        lgamma(x) = -lgamma(1 - x) - ln(|delta|) - ln(sin(pi * |delta|) /
//                    (pi * |delta|)).
//      Folding `|delta|` into the shift product `prod` turns the two logarithms
//      `-ln(prod)` and `-ln(|delta|)` into a single `LogF64(|delta| / prod)`.
//
// 2. `float64` (`EvalLgammaF64Vec`):
//    - Evaluates `lgamma` in `(hi, lo)` double-word `float64` arithmetic:
//      * `x in [1, 2.5)`: `lgamma(x) = (x - 1)(x - 2) * P(x - 1.75)` so the
//        zeros at `x = 1` and `x = 2` are factored out directly without
//        subtraction cancellation. Inputs in `(0, 1)` shift into `[1, 2)` via
//        `lgamma(x) = lgamma(x + 1) - ln(x)`.
//      * `x >= 2.5`: shifts `x < 7` into `[7, inf)` in `(hi, lo)` arithmetic,
//        evaluates Stirling's series at `z = x + m`, and subtracts `ln(prod)`.
//      * `x <= 0`: applies Euler's reflection formula outside negative root
//        windows, and switches to precomputed degree-15 expansions from
//        `kLgammaNegTableF64` near the zeros of `lgamma` on `(-17, -2)` where
//        reflection suffers heavy cancellation.
//    - Whole-vector masks (`hn::AllTrue` / `hn::AllFalse`) skip polynomials and
//      table gathers for regions that no lane in the current SIMD packet
//      visits, without changing any output bits.

// Disable legacy SSE2/SSSE3 targets while keeping SSE4 as the 128-bit SIMD
// fallback for builds that do not pass `-march=haswell` (such as baseline
// x86-64 builds). Define `HWY_DISABLE_PCLMUL_AES` because Clang's
// `-march=haswell` omits `__AES__`, allowing `-march=haswell` builds to treat
// `HWY_AVX2` as the static baseline without compiling SSE4 or EMU128.
// TODO(phawkins): Disable `HWY_SSE4` and remove the `!HWY_NATIVE_FMA` fallbacks
// once all supported CPU build configurations require FMA/AVX2.
#undef HWY_DISABLED_TARGETS
#if defined(__clang__) && __clang_major__ < 19
// Work around an LLVM 18.1.8 backend crash (early-machinelicm) when compiling
// `LgammaF16Impl` for `HWY_AVX3_SPR`.
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_AVX3_SPR)
#else
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3)
#endif
#ifndef HWY_DISABLE_PCLMUL_AES
#define HWY_DISABLE_PCLMUL_AES
#endif

#ifndef JAXLIB_CPU_LGAMMA_KERNELS_ONCE_
#define JAXLIB_CPU_LGAMMA_KERNELS_ONCE_

#include "jaxlib/cpu/lgamma_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "jaxlib/cpu/thread_pool.h"
#include "jaxlib/lgamma_coeffs.h"
#include "xla/ffi/api/ffi.h"

#endif  // JAXLIB_CPU_LGAMMA_KERNELS_ONCE_

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jaxlib/cpu/lgamma_kernels.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "jaxlib/cpu/double_word.h"
#include "jaxlib/cpu/log.h"

HWY_BEFORE_NAMESPACE();
namespace jax {
namespace lgamma_internal {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
namespace cm = jax::cpu_math::HWY_NAMESPACE;
namespace mc = jax::math_coeffs;

using DF64 = hn::ScalableTag<double>;
using VF64 = hn::VFromD<DF64>;
using DF32 = hn::Rebind<float, DF64>;
using VF32 = hn::VFromD<DF32>;

// ---------------------------------------------------------------------------
// Float32 `lgamma` via float64 SIMD promotion (shared with BF16 and F16)
// ---------------------------------------------------------------------------

HWY_INLINE VF32 EvalLgammaF32Batch(VF32 x) {
  const DF64 df64;
  const DF32 df32;
  const VF32 zero_f32 = hn::Zero(df32);
  const VF32 neg_half_f32 = hn::Set(df32, -0.5f);
  const VF32 max_finite_f32 = hn::Set(df32, mc::kLgammaMaxFiniteArgF32);

  // Inputs with `|x| < 2^-126` (`0` and subnormals) overflow `lgamma(x)` to
  // `+inf`. For reflection (`x < 0.5`), write `x = n + delta` with integer
  // `n = round(x)` and `|delta| <= 0.5`; both `n` and `delta` are exact in
  // `float32` for `x > -2^23` (below `-2^23`, every `float32` is an integer).
  const auto is_zero = hn::Lt(hn::Abs(x), hn::Set(df32, mc::kLgammaTinyF32));
  const auto is_pos = hn::Ge(x, hn::Set(df32, 0.5f));
  const auto in_neg_non_int =
      hn::And(hn::Not(is_pos),
              hn::Gt(x, hn::Set(df32, mc::kLgammaAllIntegersBelowF32)));
  const VF32 x_neg_safe = hn::IfThenElse(in_neg_non_int, x, neg_half_f32);
  const VF32 n = hn::Round(x_neg_safe);
  const VF32 delta = hn::Sub(x_neg_safe, n);
  const VF32 abs_delta = hn::Abs(delta);

  // Promote to `float64` and form the positive argument `y_d >= 0.5`:
  // `y_d = x` for `x >= 0.5`, and `y_d = 1 - x` for `x < 0.5`.
  const VF64 one_d = hn::Set(df64, 1.0);
  const auto is_pos_d = hn::PromoteMaskTo(df64, df32, is_pos);
  const VF64 x_pos_d = hn::PromoteTo(df64, hn::Min(x, max_finite_f32));
  const VF64 y_d = hn::IfThenElse(
      is_pos_d, x_pos_d, hn::Sub(one_d, hn::PromoteTo(df64, x_neg_safe)));

  // Shift `y_d < 8` into `[8, inf)` using up to 8 factors so Stirling's series
  // converges in 4 terms. Balanced tree multiplication keeps rounding error in
  // `prod` minimal.
  const auto need_shift = hn::Lt(y_d, hn::Set(df64, 8.0));
  VF64 prod = one_d;
  VF64 z_d = y_d;
  if (!hn::AllFalse(df64, need_shift)) {
    const VF64 m = hn::IfThenElseZero(
        need_shift, hn::Ceil(hn::Sub(hn::Set(df64, 8.0), y_d)));
    const VF64 f0 = hn::IfThenElse(need_shift, y_d, one_d);
    const VF64 f1 =
        hn::IfThenElse(hn::Gt(m, one_d), hn::Add(y_d, one_d), one_d);
    const VF64 f2 = hn::IfThenElse(hn::Gt(m, hn::Set(df64, 2.0)),
                                   hn::Add(y_d, hn::Set(df64, 2.0)), one_d);
    const VF64 f3 = hn::IfThenElse(hn::Gt(m, hn::Set(df64, 3.0)),
                                   hn::Add(y_d, hn::Set(df64, 3.0)), one_d);
    const VF64 f4 = hn::IfThenElse(hn::Gt(m, hn::Set(df64, 4.0)),
                                   hn::Add(y_d, hn::Set(df64, 4.0)), one_d);
    const VF64 f5 = hn::IfThenElse(hn::Gt(m, hn::Set(df64, 5.0)),
                                   hn::Add(y_d, hn::Set(df64, 5.0)), one_d);
    const VF64 f6 = hn::IfThenElse(hn::Gt(m, hn::Set(df64, 6.0)),
                                   hn::Add(y_d, hn::Set(df64, 6.0)), one_d);
    const VF64 f7 = hn::IfThenElse(hn::Gt(m, hn::Set(df64, 7.0)),
                                   hn::Add(y_d, hn::Set(df64, 7.0)), one_d);
    prod = hn::Mul(hn::Mul(hn::Mul(f0, f1), hn::Mul(f2, f3)),
                   hn::Mul(hn::Mul(f4, f5), hn::Mul(f6, f7)));
    z_d = hn::Add(y_d, m);
  }

  // Stirling's series on `z_d >= 8`:
  //   lgamma(z) = (z - 0.5) * (ln(z) - 1) + 0.5 * ln(2*pi) + (1/z) * P(1/z^2).
  // Subtracting `1` from `ln(z)` first works because `ln(8) > 2`, avoiding an
  // extra subtraction of `z`.
  const VF64 ln_z = cm::LogF64(df64, z_d);
  const VF64 inv_z = hn::Div(one_d, z_d);
  const VF64 sp =
      cm::Horner(df64, hn::Mul(inv_z, inv_z), mc::kLgammaStirlingF32F64);
  const VF64 lg_z = hn::MulAdd(
      hn::Sub(z_d, hn::Set(df64, 0.5)), hn::Sub(ln_z, one_d),
      hn::MulAdd(inv_z, sp, hn::Set(df64, mc::kLgammaCStirlingHiF64)));

  // Fast path when all lanes satisfy `x >= 0.5`: skip Euler reflection, and
  // skip the second logarithm when `x >= 8` (`prod == 1`).
  VF64 res_d;
  if (HWY_LIKELY(hn::AllTrue(df64, is_pos_d))) {
    const VF64 ln_arg = hn::AllFalse(df64, need_shift) ? hn::Zero(df64)
                                                       : cm::LogF64(df64, prod);
    res_d = hn::Sub(lg_z, ln_arg);
    // Within `3/64` of `x = 1` or `x = 2`, `lg_z` and `ln(prod)` nearly cancel;
    // substitute `t * P(t)` with `t = x - 1` or `t = x - 2`.
    const VF64 near_tol = hn::Set(df64, 3.0 / 64.0);
    const auto near_2 =
        hn::Le(hn::Abs(hn::Sub(x_pos_d, hn::Set(df64, 2.0))), near_tol);
    const VF64 center = hn::IfThenElse(near_2, hn::Set(df64, 2.0), one_d);
    const VF64 t = hn::Sub(x_pos_d, center);
    const auto in_taylor = hn::Le(hn::Abs(t), near_tol);
    if (HWY_UNLIKELY(!hn::AllFalse(df64, in_taylor))) {
      VF64 poly =
          hn::IfThenElse(near_2, hn::Set(df64, mc::kLgammaTaylor2F32F64[8]),
                         hn::Set(df64, mc::kLgammaTaylor1F32F64[8]));
      for (size_t i = 8; i-- > 0;) {
        const VF64 c =
            hn::IfThenElse(near_2, hn::Set(df64, mc::kLgammaTaylor2F32F64[i]),
                           hn::Set(df64, mc::kLgammaTaylor1F32F64[i]));
        poly = hn::MulAdd(poly, t, c);
      }
      res_d = hn::IfThenElse(in_taylor, hn::Mul(t, poly), res_d);
    }
  } else {
    // Mixed/negative packet: share one `LogF64` call across positive lanes
    // (`prod`) and reflection lanes (`|delta| / prod`, since `lgamma(1 - x) +
    // ln(|delta|) = lg_z - ln(prod) + ln(|delta|) = lg_z + ln(|delta|/prod)`).
    const VF64 abs_d = hn::PromoteTo(df64, abs_delta);
    const VF64 abs_delta_safe =
        hn::IfThenElse(hn::Eq(abs_d, hn::Zero(df64)), one_d, abs_d);
    const VF64 log_arg =
        hn::IfThenElse(is_pos_d, prod, hn::Div(abs_delta_safe, prod));
    const VF64 ln_arg = cm::LogF64(df64, log_arg);

    VF64 lg_pos_d = hn::Sub(lg_z, ln_arg);
    const VF64 near_tol = hn::Set(df64, 3.0 / 64.0);
    const auto near_2 =
        hn::Le(hn::Abs(hn::Sub(x_pos_d, hn::Set(df64, 2.0))), near_tol);
    const VF64 center = hn::IfThenElse(near_2, hn::Set(df64, 2.0), one_d);
    const VF64 t = hn::Sub(x_pos_d, center);
    const auto in_taylor = hn::And(is_pos_d, hn::Le(hn::Abs(t), near_tol));
    if (HWY_UNLIKELY(!hn::AllFalse(df64, in_taylor))) {
      VF64 poly =
          hn::IfThenElse(near_2, hn::Set(df64, mc::kLgammaTaylor2F32F64[8]),
                         hn::Set(df64, mc::kLgammaTaylor1F32F64[8]));
      for (size_t i = 8; i-- > 0;) {
        const VF64 c =
            hn::IfThenElse(near_2, hn::Set(df64, mc::kLgammaTaylor2F32F64[i]),
                           hn::Set(df64, mc::kLgammaTaylor1F32F64[i]));
        poly = hn::MulAdd(poly, t, c);
      }
      lg_pos_d = hn::IfThenElse(in_taylor, hn::Mul(t, poly), lg_pos_d);
    }

    // `sinc_d = ln(sin(pi * |delta|) / (pi * |delta|)) = delta^2 * S(delta^2)`.
    const VF64 d_d = hn::PromoteTo(df64, delta);
    const VF64 d2_d = hn::Mul(d_d, d_d);
    const VF64 sinc_d =
        hn::Mul(d2_d, cm::Horner(df64, d2_d, mc::kLgammaSincF32F64));
    const VF64 refl_d = hn::Neg(hn::Add(hn::Add(lg_z, ln_arg), sinc_d));
    res_d = hn::IfThenElse(is_pos_d, lg_pos_d, refl_d);
  }

  VF32 res = hn::DemoteTo(df32, res_d);
  // Canonicalize `-0.0f` at `x = 1, 2` to `+0.0f`, return `+inf` at `0`,
  // non-positive integers, and overflow (`x > kLgammaMaxFiniteArgF32`), and
  // propagate `NaN`.
  res = hn::IfThenElse(hn::Eq(res, zero_f32), zero_f32, res);
  const auto is_neg_int = hn::And(hn::Not(is_pos), hn::Eq(x, hn::Floor(x)));
  const auto is_pole_or_inf =
      hn::Or(hn::Or(is_zero, is_neg_int), hn::Gt(x, max_finite_f32));
  res = hn::IfThenElse(is_pole_or_inf,
                       hn::Set(df32, std::numeric_limits<float>::infinity()),
                       res);
  return hn::IfThenElse(hn::IsNaN(x), hn::NaN(df32), res);
}

void LgammaBF16Impl(const hwy::bfloat16_t* in, hwy::bfloat16_t* out,
                    int64_t n) {
  cm::EvalNarrowViaF32Loop(in, out, n, EvalLgammaF32Batch);
}

void LgammaF16Impl(const hwy::float16_t* in, hwy::float16_t* out, int64_t n) {
  cm::EvalNarrowViaF32Loop(in, out, n, EvalLgammaF32Batch);
}

void LgammaF32Impl(const float* in, float* out, int64_t n) {
  const DF32 df32;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(df32));

  int64_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    const auto x = hn::LoadU(df32, in + i);
    hn::StoreU(EvalLgammaF32Batch(x), df32, out + i);
  }
  if (i < n) {
    const size_t rem = static_cast<size_t>(n - i);
    const auto x = hn::LoadN(df32, in + i, rem);
    hn::StoreN(EvalLgammaF32Batch(x), df32, out + i, rem);
  }
}

// ---------------------------------------------------------------------------
// Float64 `lgamma` via FMA double-word `(hi, lo)` arithmetic
// ---------------------------------------------------------------------------

// Evaluates Stirling's series on `z >= 7` as `(res_h, res_l)`:
//   lgamma(z) = (z - 0.5) * (ln(z) - 1) + 0.5 * ln(2*pi) + (1/z) * P(1/z^2)
// Also sets `psi` to an approximation of `digamma(z) = d/dz lgamma(z)` so the
// caller can apply a first-order correction `psi * z_lo` when `z` was rounded
// during argument shifting (`z = w + m` or `y = 1 - x`).
HWY_INLINE VF64 LgammaStirlingF64(VF64 z, VF64& res_l, VF64& psi) {
  const DF64 d;
  VF64 ll;
  const VF64 lh = cm::LogF64HiLo(d, z, hn::Zero(d), ll);
  VF64 l_m1_l;
  const VF64 l_m1_h = cm::FastTwoSum(lh, hn::Set(d, -1.0), l_m1_l);
  l_m1_l = hn::Add(l_m1_l, ll);

  // Keep the low word `b_lo` of `z - 0.5` (`z >= 7 > 0.5`): for `z > 2^52`,
  // `z - 0.5` rounds back to `z` and drops `-0.5`, which would otherwise lose
  // `-0.5 * (ln(z) - 1)` (~0.55 ULP in `lgamma(z)` on `[2^52, 2^54)`).
  VF64 b_lo;
  const VF64 b = cm::FastTwoSum(z, hn::Set(d, -0.5), b_lo);
  // Use `TwoProdBig` so non-FMA targets do not overflow when `b` approaches
  // `kLgammaMaxFiniteArgF64` (`2.56e305`).
  VF64 pl;
  VF64 ph = cm::TwoProdBig(b, l_m1_h, pl);
  pl = hn::MulAdd(b_lo, l_m1_h, pl);
  ph = cm::FastTwoSum(ph, hn::MulAdd(b, l_m1_l, pl), pl);

  VF64 sl;
  const VF64 sh =
      cm::F2AddOrdered(ph, pl, hn::Set(d, mc::kLgammaCStirlingHiF64),
                       hn::Set(d, mc::kLgammaCStirlingLoF64), sl);

  const VF64 inv_z = hn::Div(hn::Set(d, 1.0), z);
  const VF64 inv_z2 = hn::Mul(inv_z, inv_z);
  const VF64 sp = cm::Horner(d, inv_z2, mc::kLgammaStirlingF64);
  const VF64 res_h = cm::FastTwoSum(sh, hn::MulAdd(inv_z, sp, sl), res_l);

  // 4-term expansion `digamma(z) ~= ln(z) - 1/(2z) - 1/(12z^2) + 1/(120z^4) -
  // 1/(252z^6)` (accurate to ~1e-9 on `z >= 7`, plenty for multiplying a <=0.5
  // ULP rounding residual `z_lo`).
  const VF64 psi_tail = hn::NegMulAdd(
      inv_z2,
      hn::NegMulAdd(inv_z2, hn::Set(d, 1.0 / 252.0), hn::Set(d, 1.0 / 120.0)),
      hn::Set(d, 1.0 / 12.0));
  psi = hn::NegMulAdd(inv_z2, psi_tail,
                      hn::NegMulAdd(hn::Set(d, 0.5), inv_z, lh));
  return res_h;
}

// Shifts `w >= 1` into `z = w + m >= 7` (`m` in `{0, ..., 6}`), returning:
//   - `(zh, zl)`: `lgamma(z)` corrected for any rounding in `z = w + m`,
//   - `(prod_h, prod_l)`: the shift product `w * (w + 1) * ... * (w + m - 1)`,
//   - `psi_y`: `digamma(w)` (used when `w = 1 - x` has a rounding residual
//     `y_lo` during reflection).
HWY_INLINE void LgammaShiftedF64(VF64 w, VF64& zh, VF64& zl, VF64& prod_h,
                                 VF64& prod_l, VF64& psi_y) {
  const DF64 d;
  const VF64 zero = hn::Zero(d);
  const VF64 one = hn::Set(d, 1.0);
  const VF64 seven = hn::Set(d, 7.0);
  const auto need0 = hn::Lt(w, seven);

  // When every lane in the SIMD packet already has `w >= 7`, no shift product
  // is needed.
  if (hn::AllFalse(d, need0)) {
    prod_h = one;
    prod_l = zero;
    zh = LgammaStirlingF64(w, zl, psi_y);
    return;
  }

  const VF64 m = hn::IfThenElseZero(need0, hn::Ceil(hn::Sub(seven, w)));
  // Split `w_shift = min(w, 7)` into a 24-bit high part `wh` and a 29-bit low
  // part `wl`. Because `wh + k <= 12 < 16` uses at most 4 integer bits, `wh +
  // k` fits in 26 mantissa bits with zero rounding error, allowing adjacent
  // pairs of factors to be multiplied via `F2MulExactHighFma`.
  const hn::RebindToSigned<DF64> di64;
  const VF64 w_shift = hn::Min(w, seven);
  const VF64 wh =
      hn::BitCast(d, hn::And(hn::BitCast(di64, w_shift),
                             hn::Set(di64, mc::kLgammaKeep24MantissaMaskF64)));
  const VF64 wl = hn::Sub(w_shift, wh);

  const auto need1 = hn::Gt(m, one);
  const auto need2 = hn::Gt(m, hn::Set(d, 2.0));
  const auto need3 = hn::Gt(m, hn::Set(d, 3.0));
  const auto need4 = hn::Gt(m, hn::Set(d, 4.0));
  const auto need5 = hn::Gt(m, hn::Set(d, 5.0));

  const VF64 fh0 = hn::IfThenElse(need0, wh, one);
  const VF64 fl0 = hn::IfThenElseZero(need0, wl);
  const VF64 fh1 = hn::IfThenElse(need1, hn::Add(wh, one), one);
  const VF64 fl1 = hn::IfThenElseZero(need1, wl);
  const VF64 fh2 = hn::IfThenElse(need2, hn::Add(wh, hn::Set(d, 2.0)), one);
  const VF64 fl2 = hn::IfThenElseZero(need2, wl);
  const VF64 fh3 = hn::IfThenElse(need3, hn::Add(wh, hn::Set(d, 3.0)), one);
  const VF64 fl3 = hn::IfThenElseZero(need3, wl);
  const VF64 fh4 = hn::IfThenElse(need4, hn::Add(wh, hn::Set(d, 4.0)), one);
  const VF64 fl4 = hn::IfThenElseZero(need4, wl);
  const VF64 fh5 = hn::IfThenElse(need5, hn::Add(wh, hn::Set(d, 5.0)), one);
  const VF64 fl5 = hn::IfThenElseZero(need5, wl);

  VF64 p01_l, p23_l, p45_l, p0123_l;
  const VF64 p01_h = cm::F2MulExactHighFma(fh0, fl0, fh1, fl1, p01_l);
  const VF64 p23_h = cm::F2MulExactHighFma(fh2, fl2, fh3, fl3, p23_l);
  const VF64 p45_h = cm::F2MulExactHighFma(fh4, fl4, fh5, fl5, p45_l);
  const VF64 p0123_h = cm::F2Mul(p01_h, p01_l, p23_h, p23_l, p0123_l);
  prod_h = cm::F2Mul(p0123_h, p0123_l, p45_h, p45_l, prod_l);

  // `z = w + m` can lose up to 3 low bits of `w`; capture `z_lo = w - (z - m)`
  // and correct `lgamma(w + m) ~= lgamma(z) + digamma(z) * z_lo`.
  const VF64 z = hn::Add(w, m);
  const VF64 z_lo = hn::Sub(w, hn::Sub(z, m));
  VF64 psi_z;
  zh = LgammaStirlingF64(z, zl, psi_z);
  zl = hn::MulAdd(psi_z, z_lo, zl);
  // On `w < 7`, reflection only ever queries `psi_y = digamma(1 - x)` on
  // `[1.5, 5]` (since `x >= -4` is the only negative region where `1 - x` can
  // have a non-zero `y_lo` while `w < 7`), covered by `kLgammaPsi15F64`.
  psi_y =
      hn::IfThenElse(need0, cm::Horner(d, w_shift, mc::kLgammaPsi15F64), psi_z);
}

// Evaluates `lgamma(x)` near the negative zeros of `lgamma` on `(-17, -2)`
// using the 30 precomputed expansions in `kLgammaNegTableF64` (see
// `jaxlib/lgamma_coeffs.h` for the table layout).
// Two-stage SIMD mask checks avoid all table gathers when no lane lies in
// `(-17, -2)`, and avoid the 15-term polynomial gather loop when no lane falls
// inside a root window `|u| <= u_bound`.
HWY_INLINE hn::Mask<DF64> LgammaNegRootF64(VF64 x_neg_safe, VF64 n,
                                           VF64& rw_out) {
  const DF64 d;
  const VF64 neg_two = hn::Set(d, -2.0);
  // On `(-3, -2)`, `lgamma(x)` has two zeros at `x ~= -2.4570247` (`-2 + r_0`
  // in column 0) and `x ~= -2.7476826` (`-3 + r_3` in column 3), separated by
  // the local minimum at `x ~= -2.6107`. Because `round(x)` switches from
  // `-2` to `-3` at `-2.5` (inside the root window around `-2.4570247`), pin
  // `n_root = -2` across `(-2.6, -2)` so column 0 covers `-2.4570247` without
  // splitting across two integer centers.
  const auto in_m26_m2 = hn::And(hn::Gt(x_neg_safe, hn::Set(d, -2.6)),
                                 hn::Lt(x_neg_safe, neg_two));
  const VF64 n_root = hn::IfThenElse(in_m26_m2, neg_two, n);
  const auto in_k_range =
      hn::And(hn::Lt(x_neg_safe, neg_two), hn::Ge(n_root, hn::Set(d, -16.0)));
  if (HWY_LIKELY(hn::AllFalse(d, in_k_range))) {
    rw_out = hn::Zero(d);
    return in_k_range;
  }

  // Map `(n_root, sign(delta_root))` to table column `idx = 2 * (-n_root) - (3
  // if delta_root > 0 else 4)` in `[0, 29]`.
  const VF64 delta_root = hn::Sub(x_neg_safe, n_root);
  const hn::RebindToSigned<DF64> di64;
  const auto n_i64 =
      cm::SmallDoubleIntToInt64(di64, hn::Max(n_root, hn::Set(d, -17.0)));
  const auto base_idx =
      hn::IfThenElse(hn::RebindMask(di64, hn::Gt(delta_root, hn::Zero(d))),
                     hn::Set(di64, -3), hn::Set(di64, -4));
  const auto idx = hn::Clamp(hn::Sub(base_idx, hn::Add(n_i64, n_i64)),
                             hn::Zero(di64), hn::Set(di64, 29));

  const VF64 r0 = hn::GatherIndex(d, mc::kLgammaNegTableF64[0], idx);
  const VF64 neg_r1_sc = hn::GatherIndex(d, mc::kLgammaNegTableF64[1], idx);
  const VF64 scale = hn::GatherIndex(d, mc::kLgammaNegTableF64[2], idx);
  const VF64 u_bound = hn::GatherIndex(d, mc::kLgammaNegTableF64[3], idx);

  // Rescaled distance from the zero `n_root + r0 + r1`:
  //   u = ((x - n_root) - r0) * scale - r1 * scale.
  // Because `r0` is a nearby `float64` to `delta_root` and `scale` is a power
  // of two, `(delta_root - r0) * scale` is exact in `float64`, and
  // `FastTwoSum` captures `u = u_h + u_l` to ~106 bits.
  VF64 u_l;
  const VF64 u_h =
      cm::FastTwoSum(hn::Mul(hn::Sub(delta_root, r0), scale), neg_r1_sc, u_l);
  const auto in_root_win = hn::And(in_k_range, hn::Le(hn::Abs(u_h), u_bound));
  if (HWY_LIKELY(hn::AllFalse(d, in_root_win))) {
    rw_out = hn::Zero(d);
    return in_root_win;
  }

  // Evaluate `u * (c1_hi + c1_lo + u * (c2 + ... + c15 * u^13))`.
  const VF64 u_s = hn::IfThenElseZero(in_root_win, u_h);
  VF64 p_tail = hn::GatherIndex(d, mc::kLgammaNegTableF64[20], idx);
  for (int row = 19; row >= 6; --row) {
    p_tail = hn::MulAdd(p_tail, u_s,
                        hn::GatherIndex(d, mc::kLgammaNegTableF64[row], idx));
  }
  VF64 pw_l;
  const VF64 pw_h =
      cm::FastTwoSum(hn::GatherIndex(d, mc::kLgammaNegTableF64[4], idx),
                     hn::Mul(u_s, p_tail), pw_l);
  pw_l = hn::Add(pw_l, hn::GatherIndex(d, mc::kLgammaNegTableF64[5], idx));

  VF64 rw_l;
  const VF64 rw_h = cm::TwoProd(u_s, pw_h, rw_l);
  rw_out = hn::Add(rw_h, hn::MulAdd(u_s, pw_l, hn::MulAdd(u_l, pw_h, rw_l)));
  return in_root_win;
}

HWY_INLINE VF64 EvalLgammaF64Vec(VF64 x) {
  const DF64 d;
  const VF64 zero = hn::Zero(d);
  const VF64 one = hn::Set(d, 1.0);
  const VF64 seven = hn::Set(d, 7.0);
  const VF64 max_finite = hn::Set(d, mc::kLgammaMaxFiniteArgF64);

  // Fast path when every lane in the SIMD packet lies in
  // `[7.0, kLgammaMaxFiniteArgF64]`: Stirling's series applies directly with no
  // shift product, small-positive polynomial, reflection, or table lookup.
  if (HWY_LIKELY(
          hn::AllTrue(d, hn::And(hn::Ge(x, seven), hn::Le(x, max_finite))))) {
    VF64 zl, psi_unused;
    const VF64 zh = LgammaStirlingF64(x, zl, psi_unused);
    return hn::Add(zh, zl);
  }

  // Inputs with `|x| < 2^-1022` (`0` and subnormals) overflow `lgamma(x)` to
  // `+inf`. For non-positive `x > -2^52` (below `-2^52`, every `float64` is an
  // integer), write `x = n + delta` with `n = round(x)` and `|delta| <= 0.5`,
  // and form `1 - x = y + y_lo` in `(hi, lo)` form because `1 - n` can exceed
  // `1` while `|delta|` can be much smaller than `ulp(1 - n)`.
  const auto is_zero = hn::Lt(hn::Abs(x), hn::Set(d, mc::kLgammaTinyF64));
  const auto is_pos = hn::Gt(x, zero);
  const VF64 x_safe = hn::IfThenElse(is_zero, one, x);

  const auto in_neg_non_int =
      hn::And(hn::Not(is_pos),
              hn::Gt(x_safe, hn::Set(d, mc::kLgammaAllIntegersBelowF64)));
  const VF64 x_neg_safe =
      hn::IfThenElse(in_neg_non_int, x_safe, hn::Set(d, -0.5));
  const VF64 n = hn::Round(x_neg_safe);
  const VF64 delta = hn::Sub(x_neg_safe, n);
  const VF64 abs_delta = hn::Abs(delta);
  const VF64 one_m_n = hn::Sub(one, n);
  const VF64 y = hn::Sub(one_m_n, delta);
  const VF64 y_lo = hn::Sub(hn::Sub(one_m_n, y), delta);

  const VF64 arg_pos = hn::IfThenElse(is_pos, x_safe, y);
  const VF64 x_bounded = hn::Min(arg_pos, max_finite);

  // Small positive interval `x in (0, 2.5)`:
  // Shift `x in (0, 1)` to `t = x + 1 in (1, 2)` (`lgamma(x) = lgamma(x + 1) -
  // ln(x)`), then evaluate `lgamma(t) = (t - 1)(t - 2) * P(t - 1.75)` on
  // `[1, 2.5)`. Both `u = t - 1` and `v = t - 2` are exact in `float64`, so the
  // zeros at `t = 1` and `t = 2` are exact without cancellation.
  const VF64 two_pt_five = hn::Set(d, 2.5);
  const auto in_small = hn::And(is_pos, hn::Lt(x_bounded, two_pt_five));
  const VF64 x_small_in = hn::Min(x_bounded, two_pt_five);
  const auto is_lt1 = hn::Lt(x_small_in, one);
  VF64 p_h = zero;
  VF64 p_l = zero;
  if (!hn::AllFalse(d, in_small)) {
    const VF64 u = hn::IfThenElse(is_lt1, x_small_in, hn::Sub(x_small_in, one));
    const VF64 v = hn::Sub(u, one);
    const VF64 d_arg = hn::Sub(u, hn::Set(d, 0.75));
    VF64 pl_poly;
    const VF64 ph_poly = cm::FastTwoSum(
        hn::Set(d, mc::kLgammaPos12C0HiF64),
        hn::Mul(d_arg, cm::Horner(d, d_arg, mc::kLgammaPos12TailF64)), pl_poly);
    pl_poly = hn::Add(pl_poly, hn::Set(d, mc::kLgammaPos12C0LoF64));
    VF64 uv_l;
    const VF64 uv_h = cm::TwoProd(u, v, uv_l);
    p_h = cm::F2Mul(uv_h, uv_l, ph_poly, pl_poly, p_l);
  }

  // Fast exit when every lane in the packet lies in `(0, 2.5)`: skip shifted
  // Stirling, reflection, and negative-root lookup completely.
  if (hn::AllTrue(d, in_small) && hn::AllFalse(d, is_zero)) {
    VF64 lg_pos;
    if (hn::AllFalse(d, is_lt1)) {
      lg_pos = hn::Add(p_h, p_l);
    } else {
      VF64 log_l;
      const VF64 log_h = cm::LogF64HiLo(d, x_small_in, zero, log_l);
      lg_pos = hn::IfThenElse(
          is_lt1, cm::F2Sum(p_h, p_l, hn::Neg(log_h), hn::Neg(log_l)),
          hn::Add(p_h, p_l));
    }
    return hn::IfThenElse(hn::Eq(lg_pos, zero), zero, lg_pos);
  }

  // Shifted Stirling on `w_arg >= 1` (used for `x >= 2.5` and for `1 - x >= 1`
  // when `x <= 0`).
  const VF64 w_arg = hn::Max(x_bounded, one);
  VF64 zh, zl, prod_h, prod_l, psi_y;
  LgammaShiftedF64(w_arg, zh, zl, prod_h, prod_l, psi_y);

  // Share a single `LogF64HiLo` call across all three regions that need a
  // logarithm:
  //   - `x in (0, 1)`: `ln(x)`,
  //   - `x >= 2.5`:    `ln(prod_h + prod_l) ~= ln(prod_h) + prod_l / prod_h`,
  //   - `x <= 0`:      `ln(|delta| / (prod_h + prod_l))`.
  // When `x` is very close to `0` from below (`|delta| < 1e-290`), pre-scale
  // `|delta|` by `2^54` so `|delta| / prod_h` cannot underflow to subnormal,
  // and pass `k_adj = -54` to `LogF64HiLo` to subtract `54 * ln(2)`.
  // One hardware division `1 / (prod_h * d_for_rcp)` supplies both `1 / prod_h`
  // and `1 / |delta|`.
  const VF64 abs_delta_safe =
      hn::IfThenElse(hn::Eq(abs_delta, zero), one, abs_delta);
  const auto is_tiny_neg =
      hn::And(hn::Not(is_pos), hn::Lt(abs_delta_safe, hn::Set(d, 1e-290)));
  const VF64 abs_d_s = hn::IfThenElse(
      is_tiny_neg, hn::Mul(abs_delta_safe, hn::Set(d, 18014398509481984.0)),
      abs_delta_safe);

  const VF64 d_for_rcp = hn::IfThenElse(is_pos, one, abs_d_s);
  const VF64 rcp_both = hn::Div(one, hn::Mul(prod_h, d_for_rcp));
  const VF64 rcp_prod = hn::Mul(d_for_rcp, rcp_both);
  const VF64 qh = hn::Mul(abs_d_s, rcp_prod);
  VF64 rem_q;
  if constexpr (HWY_NATIVE_FMA) {
    rem_q = hn::NegMulAdd(qh, prod_l, hn::NegMulAdd(qh, prod_h, abs_d_s));
  } else {
    // TODO(phawkins): Remove the !HWY_NATIVE_FMA fallback once all supported
    // CPU build configurations require FMA/AVX2.
    VF64 pl_q;
    const VF64 ph_q = cm::TwoProd(qh, prod_h, pl_q);
    rem_q = hn::Sub(hn::Sub(hn::Sub(abs_d_s, ph_q), pl_q), hn::Mul(qh, prod_l));
  }

  const VF64 pos_log_h = hn::IfThenElse(in_small, x_small_in, prod_h);
  const VF64 log_arg_h = hn::IfThenElse(is_pos, pos_log_h, qh);
  const VF64 k_adj = hn::IfThenElse(is_tiny_neg, hn::Set(d, -54.0), zero);
  VF64 log_l;
  const VF64 log_h = cm::LogF64HiLo(d, log_arg_h, k_adj, log_l);
  const VF64 pos_log_corr =
      hn::IfThenZeroElse(in_small, hn::Mul(prod_l, rcp_prod));
  const VF64 neg_log_corr = hn::Mul(rem_q, hn::Mul(prod_h, rcp_both));
  log_l = hn::Add(log_l, hn::IfThenElse(is_pos, pos_log_corr, neg_log_corr));

  const VF64 pos_small = hn::IfThenElse(
      is_lt1, cm::F2Sum(p_h, p_l, hn::Neg(log_h), hn::Neg(log_l)),
      hn::Add(p_h, p_l));
  const VF64 lg_pos = hn::IfThenElse(
      in_small, pos_small, cm::F2Sum(zh, zl, hn::Neg(log_h), hn::Neg(log_l)));

  VF64 res = lg_pos;
  if (!hn::AllTrue(d, is_pos)) {
    // Euler's reflection formula on `x <= 0`:
    //   lgamma(x) = -(lgamma(1 - x) + ln(|delta|) + ln(sin(pi*|delta|) /
    //               (pi*|delta|)))
    // where `lgamma(1 - x) + ln(|delta|)` is `(zh + zl) + (log_h + log_l) +
    // psi_y * y_lo`, and `sinc = delta^2 * S(delta^2)`. Near negative zeros on
    // `(-17, -2)`, `LgammaNegRootF64` overrides `refl` with a table expansion.
    VF64 d2_l;
    const VF64 d2_h = cm::TwoProd(delta, delta, d2_l);
    VF64 sl_sinc;
    const VF64 sh_sinc = cm::FastTwoSum(
        hn::Set(d, mc::kLgammaSincC0HiF64),
        hn::Mul(d2_h, cm::Horner(d, d2_h, mc::kLgammaSincTailF64)), sl_sinc);
    sl_sinc = hn::Add(sl_sinc, hn::Set(d, mc::kLgammaSincC0LoF64));
    VF64 sinc_l;
    const VF64 sinc_h = cm::F2Mul(d2_h, d2_l, sh_sinc, sl_sinc, sinc_l);

    VF64 sum_l;
    const VF64 sum_h = cm::F2Add(zh, zl, log_h, log_l, sum_l);
    const VF64 refl = hn::Neg(
        cm::F2Sum(sum_h, sum_l, sinc_h, hn::MulAdd(psi_y, y_lo, sinc_l)));

    VF64 rw;
    const auto in_root_win = LgammaNegRootF64(x_neg_safe, n, rw);
    const VF64 neg_res = hn::IfThenElse(in_root_win, rw, refl);
    res = hn::IfThenElse(is_pos, lg_pos, neg_res);
  }

  // Canonicalize `-0.0` at `x = 1, 2` to `+0.0`, return `+inf` at `0`,
  // non-positive integers, and overflow (`x > kLgammaMaxFiniteArgF64`), and
  // propagate `NaN`.
  res = hn::IfThenElse(hn::Eq(res, zero), zero, res);
  const auto is_neg_int =
      hn::And(hn::Not(is_pos), hn::Eq(x_safe, hn::Floor(x_safe)));
  const auto is_pole_or_inf =
      hn::Or(hn::Or(is_zero, is_neg_int), hn::Gt(x, max_finite));
  res = hn::IfThenElse(
      is_pole_or_inf, hn::Set(d, std::numeric_limits<double>::infinity()), res);
  return hn::IfThenElse(hn::IsNaN(x), hn::NaN(d), res);
}

void LgammaF64Impl(const double* in, double* out, int64_t n) {
  const DF64 d;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(d));

  int64_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    const auto x = hn::LoadU(d, in + i);
    hn::StoreU(EvalLgammaF64Vec(x), d, out + i);
  }
  if (i < n) {
    const size_t rem = static_cast<size_t>(n - i);
    const auto x = hn::LoadN(d, in + i, rem);
    hn::StoreN(EvalLgammaF64Vec(x), d, out + i, rem);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace lgamma_internal
}  // namespace jax
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jax {
namespace lgamma_internal {

HWY_EXPORT(LgammaBF16Impl);
HWY_EXPORT(LgammaF16Impl);
HWY_EXPORT(LgammaF32Impl);
HWY_EXPORT(LgammaF64Impl);

static xla::ffi::Error LgammaDispatch(xla::ffi::AnyBuffer x,
                                      xla::ffi::Result<xla::ffi::AnyBuffer> out,
                                      xla::ffi::ThreadPool thread_pool) {
  if (x.element_type() != out->element_type()) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "Input and output buffer element types must match.");
  }
  const int64_t n = static_cast<int64_t>(x.element_count());
  if (n != static_cast<int64_t>(out->element_count())) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "Input and output buffer element counts must match.");
  }
  static constexpr int64_t kChunkElements = 2048;
  switch (x.element_type()) {
    case xla::ffi::DataType::BF16: {
      const auto* in_ptr = x.reinterpret_data<hwy::bfloat16_t>();
      auto* out_ptr = out->reinterpret_data<hwy::bfloat16_t>();
      const auto fn = HWY_DYNAMIC_POINTER(LgammaBF16Impl);
      return ParallelFor(thread_pool, n, kChunkElements,
                         [&](int64_t start, int64_t count) {
                           fn(in_ptr + start, out_ptr + start, count);
                         });
    }
    case xla::ffi::DataType::F16: {
      const auto* in_ptr = x.reinterpret_data<hwy::float16_t>();
      auto* out_ptr = out->reinterpret_data<hwy::float16_t>();
      const auto fn = HWY_DYNAMIC_POINTER(LgammaF16Impl);
      return ParallelFor(thread_pool, n, kChunkElements,
                         [&](int64_t start, int64_t count) {
                           fn(in_ptr + start, out_ptr + start, count);
                         });
    }
    case xla::ffi::DataType::F32: {
      const auto* in_ptr = x.reinterpret_data<float>();
      auto* out_ptr = out->reinterpret_data<float>();
      const auto fn = HWY_DYNAMIC_POINTER(LgammaF32Impl);
      return ParallelFor(thread_pool, n, kChunkElements,
                         [&](int64_t start, int64_t count) {
                           fn(in_ptr + start, out_ptr + start, count);
                         });
    }
    case xla::ffi::DataType::F64: {
      const auto* in_ptr = x.reinterpret_data<double>();
      auto* out_ptr = out->reinterpret_data<double>();
      const auto fn = HWY_DYNAMIC_POINTER(LgammaF64Impl);
      return ParallelFor(thread_pool, n, kChunkElements,
                         [&](int64_t start, int64_t count) {
                           fn(in_ptr + start, out_ptr + start, count);
                         });
    }
    default:
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInvalidArgument,
          "cpu_lgamma_ffi only supports BF16, F16, F32, and F64 element "
          "types.");
  }
}

}  // namespace lgamma_internal

XLA_FFI_DEFINE_HANDLER_SYMBOL(cpu_lgamma_ffi, lgamma_internal::LgammaDispatch,
                              xla::ffi::Ffi::Bind()
                                  .Arg<xla::ffi::AnyBuffer>()
                                  .Ret<xla::ffi::AnyBuffer>()
                                  .Ctx<xla::ffi::ThreadPool>());

}  // namespace jax
#endif  // HWY_ONCE
