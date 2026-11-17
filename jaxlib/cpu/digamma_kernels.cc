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

// Google Highway CPU FFI kernels for `digamma(x) = d/dx ln(|Gamma(x)|)` on
// `bfloat16`, `float16`, `float32`, and `float64`, accurate to within 1 ULP
// across the full input domain.
//
// Algorithm overview (matching `jax/_src/lax/digamma.py` with hardware FMA):
//
// 1. `float32` (`EvalDigammaF32Batch`, also shared by `bfloat16` and `float16`
//    via packet-level promotion and rounding):
//    - Promotes `float32` SIMD lanes to `float64` (`z = x` for `x >= 0.5`, or
//      `z = 1 - x` for `x < 0.5`).
//    - Shifts `z < 4.5` to `z + 4 >= 4.5` in a single `+4` recurrence step
//      `digamma(z) = digamma(z + 4) - (1/z + 1/(z+1) + 1/(z+2) + 1/(z+3))`,
//      grouping the 4 reciprocals symmetrically around `m = z + 1.5` so only
//      one division is needed.
//    - Evaluates `digamma(z) = ln(z) - 1/(2z) - (1/z^2) * P(1/z^2)` on
//      `z >= 4.5`, adds Euler's reflection term `-pi * cot(pi * delta)` for
//      `x < 0.5` with `delta = x - round(x)`, rounds back to `float32`, and
//      applies the 3 single-float overrides from `kDigammaOverridesInF32` near
//      deep negative zeros where `digamma(z)` and `pi * cot(pi * delta)` cancel
//      past `float64` precision.
//
// 2. `float64` (`EvalDigammaF64Vec`):
//    - Evaluates `digamma` in `(hi, lo)` double-word `float64` arithmetic:
//      * Shifts `z < 12.5` into `[12.5, inf)` using up to three `+4` steps,
//        breaking out of the loop early once all SIMD lanes reach `12.5`.
//      * Evaluates `ln(z) - 1/(2z) - (1/z^2) * P(1/z^2)` using 108-bit
//        `LogF2WithRecip` and the 15-term expansion `kDigammaStir*F64`.
//      * For `x < 0.5`, adds `-pi * cot(pi * delta)` decomposed as `-1/delta`
//        plus the rational pole-removal term `delta * (10 - 4*delta^2) /
//        (4 - 5*delta^2 + delta^4)` plus `delta * P_cot(delta^2)`.
//      * Within `1e-4` of the positive zero `r_+ ~= 1.461632145`, substitutes a
//        degree-4 Taylor expansion (`kDigammaPsi*F64`) to avoid cancellation
//        between `ln(z)` and the recurrence shift sum; near negative zeros,
//        applies the 10 `float64` overrides in `kDigammaOverridesInF64`.

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
// `DigammaF16Impl` for `HWY_AVX3_SPR`.
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_AVX3_SPR)
#else
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3)
#endif
#ifndef HWY_DISABLE_PCLMUL_AES
#define HWY_DISABLE_PCLMUL_AES
#endif

#ifndef JAXLIB_CPU_DIGAMMA_KERNELS_ONCE_
#define JAXLIB_CPU_DIGAMMA_KERNELS_ONCE_

#include "jaxlib/cpu/digamma_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "jaxlib/cpu/thread_pool.h"
#include "jaxlib/digamma_coeffs.h"
#include "xla/ffi/api/ffi.h"

#endif  // JAXLIB_CPU_DIGAMMA_KERNELS_ONCE_

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jaxlib/cpu/digamma_kernels.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "jaxlib/cpu/double_word.h"
#include "jaxlib/cpu/log.h"

HWY_BEFORE_NAMESPACE();
namespace jax {
namespace digamma_internal {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
namespace cm = jax::cpu_math::HWY_NAMESPACE;
namespace mc = jax::math_coeffs;

using DF64 = hn::ScalableTag<double>;
using VF64 = hn::VFromD<DF64>;
using DF32 = hn::Rebind<float, DF64>;
using VF32 = hn::VFromD<DF32>;

// ---------------------------------------------------------------------------
// Float32 `digamma` via float64 SIMD promotion (shared with BF16 and F16)
// ---------------------------------------------------------------------------

HWY_INLINE VF32 EvalDigammaF32Batch(VF32 x) {
  const DF64 df64;
  const DF32 df32;

  const VF32 one_f32 = hn::Set(df32, 1.0f);
  const VF32 inf_f32 = hn::Set(df32, std::numeric_limits<float>::infinity());
  const VF32 half_neg_f32 = hn::Set(df32, -0.5f);

  // Classify special regimes:
  //   - `|x| < 2^-126` (`+-0` and subnormals): `digamma(x) ~= -1/x` overflows
  //     to `-copysign(inf, x)`.
  //   - `x < 0.5`: reflect across `0.5` using `n = round(x)` and `delta = x -
  //     n` in `[-0.5, 0.5]`. Below `-2^24` (`kDigammaAllIntBelowF32`), every
  //     `float32` is an integer where `Gamma` has a pole and `digamma` is
  //     `NaN`.
  const VF32 ax = hn::Abs(x);
  const auto is_zero_or_sub = hn::Lt(ax, hn::Set(df32, mc::kDigammaTinyF32));
  const auto is_pos = hn::Ge(x, hn::Set(df32, 0.5f));
  const auto gt_int_lim = hn::Gt(x, hn::Set(df32, mc::kDigammaAllIntBelowF32));
  const VF32 x_safe = hn::IfThenElse(
      hn::Or(is_zero_or_sub, hn::Not(hn::Lt(ax, inf_f32))), one_f32, x);
  const VF32 x_neg_safe = hn::IfThenElse(
      hn::And(hn::And(hn::Not(is_pos), gt_int_lim), hn::Not(is_zero_or_sub)),
      x_safe, half_neg_f32);

  const VF32 n = hn::Round(x_neg_safe);
  const VF32 delta = hn::Sub(x_neg_safe, n);
  const auto is_delta_zero = hn::Eq(delta, hn::Zero(df32));
  const VF32 delta_nz = hn::IfThenElse(is_delta_zero, half_neg_f32, delta);

  // Promote to `float64` with `z = x` for `x >= 0.5` and `z = (1 - n) - delta =
  // 1 - x >= 0.5` for `x < 0.5`.
  const VF64 one_d = hn::Set(df64, 1.0);
  const auto is_pos_d = hn::PromoteMaskTo(df64, df32, is_pos);
  const VF64 xd = hn::PromoteTo(df64, x_safe);
  const VF64 dd = hn::PromoteTo(df64, delta_nz);
  VF64 z = hn::IfThenElse(is_pos_d, xd,
                          hn::Sub(hn::Sub(one_d, hn::PromoteTo(df64, n)), dd));

  // Shift `z < 4.5` to `z + 4 >= 4.5` using:
  //   1/z + 1/(z+1) + 1/(z+2) + 1/(z+3)
  //     = (1/(m - 1.5) + 1/(m + 1.5)) + (1/(m - 0.5) + 1/(m + 0.5))
  //     = 4 * m * (m^2 - 1.25) / ((m^2 - 2.5) * m^2 + 0.5625)
  // with `m = z + 1.5`, trading 4 divisions for 1.
  const auto need_shift = hn::Lt(z, hn::Set(df64, 4.5));
  VF64 acc = hn::Zero(df64);
  if (!hn::AllFalse(df64, need_shift)) {
    const VF64 zs = hn::IfThenElse(need_shift, z, one_d);
    const VF64 m = hn::Add(zs, hn::Set(df64, 1.5));
    const VF64 q = hn::Mul(m, m);
    const VF64 num = hn::Mul(hn::Mul(hn::Set(df64, 4.0), m),
                             hn::Sub(q, hn::Set(df64, 1.25)));
    const VF64 den =
        hn::MulAdd(hn::Sub(q, hn::Set(df64, 2.5)), q, hn::Set(df64, 0.5625));
    acc = hn::IfThenElseZero(need_shift, hn::Neg(hn::Div(num, den)));
    z = hn::IfThenElse(need_shift, hn::Add(z, hn::Set(df64, 4.0)), z);
  }

  // Evaluate `digamma(z) = acc + ln(z) - 1/(2z) - (1/z^2) * P(1/z^2)` on
  // `z >= 4.5`.
  const VF64 iz = hn::Div(one_d, z);
  const VF64 u = hn::Mul(iz, iz);
  const VF64 st = cm::Horner(df64, u, mc::kDigammaStirF32F64);
  const VF64 psi_z = hn::NegMulAdd(
      u, st,
      hn::NegMulAdd(hn::Set(df64, 0.5), iz, hn::Add(acc, cm::LogF64(df64, z))));

  // Add Euler's reflection term `-pi * cot(pi * delta)` for `x < 0.5`:
  //   -pi * cot(pi * d) = -1/d + d * ((10 - 4*d^2)/(4 - 5*d^2 + d^4) + C(d^2))
  // Removing the poles at `d = 0, +-1, +-2` explicitly makes `C(d^2)` converge
  // in 4 terms on `|d| <= 0.5`.
  VF64 res_d = psi_z;
  if (!hn::AllTrue(df64, is_pos_d)) {
    const VF64 w = hn::Mul(dd, dd);
    const VF64 ct = cm::Horner(df64, w, mc::kDigammaCotF32F64);
    const VF64 rat_num =
        hn::NegMulAdd(hn::Set(df64, 4.0), w, hn::Set(df64, 10.0));
    const VF64 rat_den =
        hn::MulAdd(hn::Sub(w, hn::Set(df64, 5.0)), w, hn::Set(df64, 4.0));
    const VF64 cot = hn::MulAdd(dd, hn::Add(hn::Div(rat_num, rat_den), ct),
                                hn::Neg(hn::Div(one_d, dd)));
    res_d = hn::IfThenElse(is_pos_d, psi_z, hn::Add(psi_z, cot));
  }

  VF32 res = hn::DemoteTo(df32, res_d);

  // Patch the 3 `float32` inputs near deep negative zeros where `psi_z + cot`
  // cancels more than 29 bits (all 3 have `|digamma(x)| <= 5.4e-8f`, so the
  // `|res| < 1e-5f` check skips the loop on normal packets).
  if (!hn::AllTrue(df32, is_pos) &&
      HWY_UNLIKELY(
          !hn::AllFalse(df32, hn::Lt(hn::Abs(res), hn::Set(df32, 1e-5f))))) {
    for (size_t k = 0; k < 3; ++k) {
      res = hn::IfThenElse(
          hn::Eq(x, hn::Set(df32, mc::kDigammaOverridesInF32[k])),
          hn::Set(df32, mc::kDigammaOverridesOutF32[k]), res);
    }
  }

  // Return `NaN` for `NaN` inputs and negative integers (`x <= 0` with `delta
  // == 0` or `x <= -2^24`), `-copysign(inf, x)` at `+-0` / subnormals, and
  // `+inf` at `+inf`.
  const auto is_nan_out = hn::Or(
      hn::IsNaN(x),
      hn::And(hn::Not(is_pos), hn::Or(is_delta_zero, hn::Not(gt_int_lim))));
  res = hn::IfThenElse(is_nan_out, hn::NaN(df32), res);
  const auto zero_inf = hn::Neg(hn::CopySign(inf_f32, x));
  res = hn::IfThenElse(is_zero_or_sub, zero_inf, res);
  return hn::IfThenElse(hn::Eq(x, inf_f32), inf_f32, res);
}

void DigammaBF16Impl(const hwy::bfloat16_t* in, hwy::bfloat16_t* out,
                     int64_t n) {
  cm::EvalNarrowViaF32Loop(in, out, n, EvalDigammaF32Batch);
}

void DigammaF16Impl(const hwy::float16_t* in, hwy::float16_t* out, int64_t n) {
  cm::EvalNarrowViaF32Loop(in, out, n, EvalDigammaF32Batch);
}

void DigammaF32Impl(const float* in, float* out, int64_t n) {
  const DF32 df32;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(df32));

  int64_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    const auto x = hn::LoadU(df32, in + i);
    hn::StoreU(EvalDigammaF32Batch(x), df32, out + i);
  }
  if (i < n) {
    const size_t rem = static_cast<size_t>(n - i);
    const auto x = hn::LoadN(df32, in + i, rem);
    hn::StoreN(EvalDigammaF32Batch(x), df32, out + i, rem);
  }
}

// ---------------------------------------------------------------------------
// Float64 `digamma` via FMA double-word `(hi, lo)` arithmetic
// ---------------------------------------------------------------------------

// Evaluates one `+4` recurrence step `1/c + 1/(c+1) + 1/(c+2) + 1/(c+3)` for
// double-word `c = c0 + c1 >= 0.5` in `(hi, lo)` arithmetic using a single
// double-word division:
//   m = c + 1.5,  u = m^2 - 1.25  =>  sum = 4 * m * u / (u^2 - 1).
// Because `c >= 0.5`, `m >= 2` and `u >= 2.75`, so `u^2 - 1 >= 6.5625` stays
// well away from zero with no subtraction cancellation.
HWY_INLINE VF64 Shift4F2(VF64 c0, VF64 c1, VF64& out1) {
  const DF64 d;
  VF64 e;
  const VF64 s = cm::TwoSum(c0, hn::Set(d, 1.5), e);
  VF64 m1;
  const VF64 m0 = cm::FastTwoSum(s, hn::Add(e, c1), m1);
  VF64 msq1;
  const VF64 msq0 = cm::F2Square(m0, m1, msq1);
  VF64 u1;
  const VF64 u0 = cm::FastTwoSum(msq0, hn::Set(d, -1.25), u1);
  u1 = hn::Add(u1, msq1);
  VF64 usq1;
  const VF64 usq0 = cm::F2Square(u0, u1, usq1);
  VF64 den1;
  const VF64 den0 = cm::FastTwoSum(usq0, hn::Set(d, -1.0), den1);
  den1 = hn::Add(den1, usq1);
  const VF64 four = hn::Set(d, 4.0);
  VF64 num1;
  const VF64 num0 =
      cm::F2Mul(hn::Mul(four, m0), hn::Mul(four, m1), u0, u1, num1);
  return cm::F2Div(d, num0, num1, den0, den1, out1);
}

HWY_INLINE VF64 EvalDigammaF64Vec(VF64 x) {
  const DF64 d;
  const VF64 zero = hn::Zero(d);
  const VF64 one = hn::Set(d, 1.0);
  const VF64 four = hn::Set(d, 4.0);
  const VF64 half_neg = hn::Set(d, -0.5);
  const VF64 inf = hn::Set(d, std::numeric_limits<double>::infinity());

  const VF64 ax = hn::Abs(x);
  const auto is_zero_or_sub = hn::Lt(ax, hn::Set(d, mc::kDigammaTinyF64));
  const auto is_pos = hn::Ge(x, hn::Set(d, 0.5));
  const auto gt_int_lim = hn::Gt(x, hn::Set(d, mc::kDigammaAllIntBelowF64));
  const VF64 x_safe =
      hn::IfThenElse(hn::Or(is_zero_or_sub, hn::Not(hn::Lt(ax, inf))), one, x);
  const VF64 x_neg_safe = hn::IfThenElse(
      hn::And(hn::And(hn::Not(is_pos), gt_int_lim), hn::Not(is_zero_or_sub)),
      x_safe, half_neg);

  const VF64 n = hn::Round(x_neg_safe);
  const VF64 delta = hn::Sub(x_neg_safe, n);
  const auto is_delta_zero = hn::Eq(delta, zero);
  const VF64 delta_nz = hn::IfThenElse(is_delta_zero, half_neg, delta);

  // For `x < 0.5`, form `y = 1 - x = (1 - n) - delta` as a `(yh, yl)` pair.
  // Because `x > -2^53` has `n <= 0` an exact integer and `1 - n >= 1 > 0.5 >=
  // |delta|`, `FastTwoSum(1 - n, -delta)` captures `1 - x` without losing low
  // bits of `delta` when `|n|` is large.
  VF64 yl;
  const VF64 yh = cm::FastTwoSum(hn::Sub(one, n), hn::Neg(delta), yl);
  VF64 z0 = hn::IfThenElse(is_pos, x_safe, yh);
  VF64 z1 = hn::IfThenElse(is_pos, zero, yl);

  // Shift `z < 12.5` into `[12.5, inf)` in up to 3 steps of `+4`, exiting the
  // loop early as soon as every lane in the SIMD packet has `z >= 12.5`.
  VF64 acc0 = zero;
  VF64 acc1 = zero;
  const VF64 z_min = hn::Set(d, mc::kDigammaZMinF64);
  for (int step = 0; step < 3; ++step) {
    const auto mask =
        hn::Or(hn::Lt(z0, z_min), hn::And(hn::Eq(z0, z_min), hn::Lt(z1, zero)));
    if (hn::AllFalse(d, mask)) {
      break;
    }
    const VF64 sc0 = hn::IfThenElse(mask, z0, one);
    const VF64 sc1 = hn::IfThenElse(mask, z1, zero);
    VF64 h1, na1, nz1;
    const VF64 h0 = Shift4F2(sc0, sc1, h1);
    const VF64 na0 = cm::F2Add(acc0, acc1, hn::Neg(h0), hn::Neg(h1), na1);
    const VF64 nz0 = cm::F2Add(sc0, sc1, four, zero, nz1);
    acc0 = hn::IfThenElse(mask, na0, acc0);
    acc1 = hn::IfThenElse(mask, na1, acc1);
    z0 = hn::IfThenElse(mask, nz0, z0);
    z1 = hn::IfThenElse(mask, nz1, z1);
  }

  // Compute `1 / z` and subtract `1 / (2z)`. For `z0 >= 1e80`, `1 / (2z)` is
  // more than 270 bits below `ln(z)` and `1 / z^2` would underflow, so zero out
  // `(iz0, iz1)` there.
  const auto z_big = hn::Ge(z0, hn::Set(d, mc::kDigammaZBigF64));
  const VF64 zc0 = hn::IfThenElse(z_big, one, z0);
  const VF64 zc1 = hn::IfThenElse(z_big, zero, z1);
  VF64 iz1;
  VF64 iz0 = cm::F2Recip(d, zc0, zc1, iz1);
  iz0 = hn::IfThenZeroElse(z_big, iz0);
  iz1 = hn::IfThenZeroElse(z_big, iz1);
  acc0 = cm::F2Add(acc0, acc1, hn::Mul(half_neg, iz0), hn::Mul(half_neg, iz1),
                   acc1);

  // Add 108-bit `ln(z0 + z1)`, reusing the precomputed reciprocal `(iz0, iz1)`.
  VF64 lg1;
  const VF64 lg0 = cm::LogF2WithRecip(d, z0, z1, iz0, iz1, lg1);
  acc0 = cm::F2Add(acc0, acc1, lg0, lg1, acc1);

  // Subtract the 15-term series `(1 / z^2) * P_stir(1 / z^2)`.
  VF64 u1, sp1;
  const VF64 u0 = cm::F2Square(iz0, iz1, u1);
  VF64 sp0 = cm::HornerHybridF2(d, u0, u1, mc::kDigammaStirLoDegHiF64,
                                mc::kDigammaStirLoDegLoF64,
                                mc::kDigammaStirHiDegF64, sp1);
  sp0 = cm::F2Mul(sp0, sp1, hn::Neg(u0), hn::Neg(u1), sp1);
  acc0 = cm::F2Add(acc0, acc1, sp0, sp1, acc1);

  // Euler reflection on `x < 0.5`:
  //   -pi * cot(pi * delta) = -1/delta
  //                         + delta * ((10 - 4*w) / (4 - 5*w + w^2) + C(w))
  // with `w = delta^2`. Pre-scaling `delta` by `2^-54`
  // (`kDigammaRecipScaleF64`) before `F2Recip` prevents `1 / delta` from
  // overflowing when `|delta| < 2^-1022` (subnormal negative `x`).
  VF64 res;
  if (HWY_LIKELY(hn::AllTrue(d, is_pos))) {
    res = hn::Add(acc0, acc1);
  } else {
    const VF64 scale = hn::Set(d, mc::kDigammaRecipScaleF64);
    const VF64 neg_scale = hn::Set(d, -mc::kDigammaRecipScaleF64);
    VF64 ids1;
    const VF64 ids0 = cm::F2Recip(d, hn::Mul(delta_nz, scale), zero, ids1);
    const VF64 id0 = hn::IfThenZeroElse(is_pos, hn::Mul(neg_scale, ids0));
    const VF64 id1 = hn::IfThenZeroElse(is_pos, hn::Mul(neg_scale, ids1));
    acc0 = cm::F2Add(acc0, acc1, id0, id1, acc1);

    VF64 dw1, rn1, wm5_1, rd1, rat1, ct1, cp1;
    const VF64 dw0 = cm::TwoProd(delta_nz, delta_nz, dw1);
    const VF64 neg_four = hn::Set(d, -4.0);
    const VF64 rn0 = cm::F2Add(hn::Set(d, 10.0), zero, hn::Mul(neg_four, dw0),
                               hn::Mul(neg_four, dw1), rn1);
    const VF64 wm5_0 = cm::F2Add(hn::Set(d, -5.0), zero, dw0, dw1, wm5_1);
    VF64 rd0 = cm::F2Mul(wm5_0, wm5_1, dw0, dw1, rd1);
    rd0 = cm::F2Add(four, zero, rd0, rd1, rd1);
    const VF64 rat0 = cm::F2Div(d, rn0, rn1, rd0, rd1, rat1);

    const VF64 ct0 = cm::HornerHybridF2(d, dw0, dw1, mc::kDigammaCotLoDegHiF64,
                                        mc::kDigammaCotLoDegLoF64,
                                        mc::kDigammaCotHiDegF64, ct1);
    VF64 cp0 = cm::F2Add(ct0, ct1, rat0, rat1, cp1);
    const VF64 delta_refl = hn::IfThenZeroElse(is_pos, delta_nz);
    cp0 = cm::F2MulScalar(cp0, cp1, delta_refl, cp1);
    res = cm::F2Sum(acc0, acc1, cp0, cp1);
  }

  // Within `1e-4` of the positive zero `r_+ ~= 1.461632145`, `ln(z)` and the
  // `+4` recurrence sum cancel to up to 53 bits, leaving ~53 bits in `(hi, lo)`
  // arithmetic. Substitute a direct degree-4 Taylor expansion around
  // `kDigammaRposHiF64` (exact `t = x - kDigammaRposHiF64`).
  const VF64 rpos_hi = hn::Set(d, mc::kDigammaRposHiF64);
  const auto in_rpos = hn::Lt(hn::Abs(hn::Sub(x_safe, rpos_hi)),
                              hn::Set(d, mc::kDigammaRposTolF64));
  if (HWY_UNLIKELY(!hn::AllFalse(d, in_rpos))) {
    const VF64 t = hn::IfThenElseZero(in_rpos, hn::Sub(x_safe, rpos_hi));
    VF64 p1, s1;
    const VF64 p0 = cm::TwoProd(t, hn::Set(d, mc::kDigammaPsi1HiF64), p1);
    const VF64 s0 = cm::TwoSum(hn::Set(d, mc::kDigammaPsi0HiF64), p0, s1);
    const VF64 tail_poly =
        hn::MulAdd(t,
                   hn::MulAdd(t,
                              hn::MulAdd(t, hn::Set(d, mc::kDigammaPsi4F64),
                                         hn::Set(d, mc::kDigammaPsi3F64)),
                              hn::Set(d, mc::kDigammaPsi2F64)),
                   hn::Set(d, mc::kDigammaPsi1LoF64));
    const VF64 tail =
        hn::MulAdd(t, tail_poly, hn::Set(d, mc::kDigammaPsi0LoF64));
    const VF64 res_pos = hn::Add(s0, hn::Add(s1, hn::Add(p1, tail)));
    res = hn::IfThenElse(in_rpos, res_pos, res);
  }

  // Patch the 10 `float64` inputs near negative zeros where `digamma(1 - x)`
  // and `pi * cot(pi * delta)` cancel more than 53 bits (all 10 have
  // `|digamma(x)| <= 3.82e-15`, so `|res| < 1e-12` skips the loop on normal
  // packets).
  if (!hn::AllTrue(d, is_pos) &&
      HWY_UNLIKELY(!hn::AllFalse(d, hn::Lt(hn::Abs(res), hn::Set(d, 1e-12))))) {
    for (size_t k = 0; k < 10; ++k) {
      res = hn::IfThenElse(hn::Eq(x, hn::Set(d, mc::kDigammaOverridesInF64[k])),
                           hn::Set(d, mc::kDigammaOverridesOutF64[k]), res);
    }
  }

  // Return `NaN` for `NaN` inputs and negative integers (`x <= 0` with `delta
  // == 0` or `x <= -2^53`), `-copysign(inf, x)` at `+-0` / subnormals, and
  // `+inf` at `+inf`.
  const auto is_nan_out = hn::Or(
      hn::IsNaN(x),
      hn::And(hn::Not(is_pos), hn::Or(is_delta_zero, hn::Not(gt_int_lim))));
  res = hn::IfThenElse(is_nan_out, hn::NaN(d), res);
  const auto zero_inf = hn::Neg(hn::CopySign(inf, x));
  res = hn::IfThenElse(is_zero_or_sub, zero_inf, res);
  return hn::IfThenElse(hn::Eq(x, inf), inf, res);
}

void DigammaF64Impl(const double* in, double* out, int64_t n) {
  const DF64 d;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(d));

  int64_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    const auto x = hn::LoadU(d, in + i);
    hn::StoreU(EvalDigammaF64Vec(x), d, out + i);
  }
  if (i < n) {
    const size_t rem = static_cast<size_t>(n - i);
    const auto x = hn::LoadN(d, in + i, rem);
    hn::StoreN(EvalDigammaF64Vec(x), d, out + i, rem);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace digamma_internal
}  // namespace jax
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jax {
namespace digamma_internal {

HWY_EXPORT(DigammaBF16Impl);
HWY_EXPORT(DigammaF16Impl);
HWY_EXPORT(DigammaF32Impl);
HWY_EXPORT(DigammaF64Impl);

static xla::ffi::Error DigammaDispatch(
    xla::ffi::AnyBuffer x, xla::ffi::Result<xla::ffi::AnyBuffer> out,
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
      const auto fn = HWY_DYNAMIC_POINTER(DigammaBF16Impl);
      return ParallelFor(thread_pool, n, kChunkElements,
                         [&](int64_t start, int64_t count) {
                           fn(in_ptr + start, out_ptr + start, count);
                         });
    }
    case xla::ffi::DataType::F16: {
      const auto* in_ptr = x.reinterpret_data<hwy::float16_t>();
      auto* out_ptr = out->reinterpret_data<hwy::float16_t>();
      const auto fn = HWY_DYNAMIC_POINTER(DigammaF16Impl);
      return ParallelFor(thread_pool, n, kChunkElements,
                         [&](int64_t start, int64_t count) {
                           fn(in_ptr + start, out_ptr + start, count);
                         });
    }
    case xla::ffi::DataType::F32: {
      const auto* in_ptr = x.reinterpret_data<float>();
      auto* out_ptr = out->reinterpret_data<float>();
      const auto fn = HWY_DYNAMIC_POINTER(DigammaF32Impl);
      return ParallelFor(thread_pool, n, kChunkElements,
                         [&](int64_t start, int64_t count) {
                           fn(in_ptr + start, out_ptr + start, count);
                         });
    }
    case xla::ffi::DataType::F64: {
      const auto* in_ptr = x.reinterpret_data<double>();
      auto* out_ptr = out->reinterpret_data<double>();
      const auto fn = HWY_DYNAMIC_POINTER(DigammaF64Impl);
      return ParallelFor(thread_pool, n, kChunkElements,
                         [&](int64_t start, int64_t count) {
                           fn(in_ptr + start, out_ptr + start, count);
                         });
    }
    default:
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInvalidArgument,
          "cpu_digamma_ffi only supports BF16, F16, F32, and F64 element "
          "types.");
  }
}

}  // namespace digamma_internal

XLA_FFI_DEFINE_HANDLER_SYMBOL(cpu_digamma_ffi,
                              digamma_internal::DigammaDispatch,
                              xla::ffi::Ffi::Bind()
                                  .Arg<xla::ffi::AnyBuffer>()
                                  .Ret<xla::ffi::AnyBuffer>()
                                  .Ctx<xla::ffi::ThreadPool>());

}  // namespace jax
#endif  // HWY_ONCE
