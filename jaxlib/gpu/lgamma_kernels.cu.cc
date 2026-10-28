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

// CUDA and ROCm device kernels for `lgamma(x) = ln(|Gamma(x)|)` on `bfloat16`,
// `float16`, `float32`, and `float64`, accurate to within 1 ULP across the full
// input domain (matching the three implementations in `jax/_src/lax/lgamma.py`
// with hardware FMA):
//
// 1. `LgammaF32Kernel` (`float32`-only, also shared by `bfloat16` and
//    `float16` via load/store conversion):
//    Used on GPUs whose `float64` throughput is much lower than `float32`
//    (such as L4, T4, RTX, or RDNA GPUs). Avoids all `float64` instructions by
//    evaluating `(hi, lo)` `float32` polynomials on `[0.5, 4]`, Stirling's
//    series on `(4, inf)`, Euler's reflection formula on `x < 0.5`, and
//    precomputed `float32` expansions from `kLgammaNegTableF32` near negative
//    zeros on `(-10, -2)`.
//
// 2. `LgammaF32WithF64Kernel` (`float32` via native `float64`, also shared by
//    `bfloat16` and `float16`):
//    Used on GPUs with high-throughput `float64` units (`1 <= perf_ratio <= 4`,
//    such as A100, H100, B200, MI250, and MI300), where promoting to `float64`
//    avoids `float32` bit-splitting and negative-root table lookups.
//
// 3. `LgammaF64Kernel` (`float64`):
//    Evaluates `lgamma` in `(hi, lo)` double-word `float64` arithmetic using
//    `(x - 1)(x - 2) * P(x - 1.75)` on `(0, 2.5)`, shifted Stirling (`z >= 7`)
//    on `[2.5, inf)`, Euler's reflection formula on `x <= 0`, and degree-15
//    expansions from `kLgammaNegTableF64` near negative zeros on `(-17, -2)`.

#include "jaxlib/gpu/lgamma_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "jaxlib/gpu/double_word.cu.h"
#include "jaxlib/gpu/log.cu.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/lgamma_coeffs.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace mi = jax::JAX_GPU_NAMESPACE::math_internal;
namespace mc = jax::math_coeffs;

// ---------------------------------------------------------------------------
// `LgammaF32`: `float32`-only 1-ULP evaluation (no `float64` instructions)
// ---------------------------------------------------------------------------

// Evaluates `lgamma(x)` on `[0.5, 4]` as `(hi, out_lo)` using the factored form
//   lgamma(x) = (x - 1) * (x - 2) * P(x - center)
// on `[0.5, 1.5]` (`center = 1.0`) and `(1.5, 4.0]` (`center = 2.75`), matching
// `_lgamma_f32` in `jax/_src/lax/lgamma.py`. Factoring out `(x - 1)` and `(x -
// 2)` preserves full relative accuracy near both positive zeros; splitting
// `u = x - 1`, `v = x - 2`, and `p_sm` to 8 bits lets `(u * v) * p_sm` form its
// leading product without rounding error.
__device__ float LgammaPosSmallF32(float x_sm, float& out_lo) {
  const bool in_15_4 = x_sm > 1.5f;
  const float u = mi::SubRn(x_sm, 1.0f);
  float v_err;
  const float v = mi::FastTwoSum(-2.0f, x_sm, v_err);
  const float d = mi::SubRn(x_sm, in_15_4 ? 2.75f : 1.0f);
  const float c0_hi =
      in_15_4 ? mc::kLgammaPos15_4C0HiF32 : mc::kLgammaPos05_15C0HiF32;
  const float c0_lo =
      in_15_4 ? mc::kLgammaPos15_4C0LoF32 : mc::kLgammaPos05_15C0LoF32;
  const float c1_hh =
      in_15_4 ? mc::kLgammaPos15_4C1HhF32 : mc::kLgammaPos05_15C1HhF32;
  const float poly = in_15_4 ? mi::Horner(d, mc::kLgammaPos15_4Q1RemF32)
                             : mi::Horner(d, mc::kLgammaPos05_15Q1RemF32);
  float d_lo;
  const float d_hi = mi::SplitHalfF32(d, d_lo);
  float p_err;
  float p_sm = mi::FastTwoSum(c0_hi, mi::MulRn(d_hi, c1_hh), p_err);
  const float p_tail = mi::FmaRn(d, poly, mi::FmaRn(d_lo, c1_hh, c0_lo));
  p_sm = mi::FastTwoSum(p_sm, mi::AddRn(p_err, p_tail), p_err);

  float u_lo, v_lo, p_lo_sm;
  const float u_hi = mi::SplitBitsF32<8>(u, u_lo);
  const float v_hi = mi::SplitBitsF32<8>(v, v_lo);
  const float uv_hi = mi::MulRn(u_hi, v_hi);
  const float uv_lo =
      mi::FmaRn(u_lo, v, mi::MulRn(u_hi, mi::AddRn(v_lo, v_err)));
  const float p_hi_sm = mi::SplitBitsF32<8>(p_sm, p_lo_sm);
  return mi::FastTwoSum(
      mi::MulRn(uv_hi, p_hi_sm),
      mi::FmaRn(uv_lo, p_sm, mi::MulRn(uv_hi, mi::AddRn(p_lo_sm, p_err))),
      out_lo);
}

// Evaluates Stirling's series on `x_gt4 > 4` in `(hi, lo)` `float32` form:
//   lgamma(x) = (x - 0.5) * (ln(x) - 1) + 0.5 * ln(2*pi) + (1/x) * P(1/x^2).
// Splitting `ln(x)` and `x - 0.5` to 12 bits makes `b_hi * (h_x - 1)` exact,
// preventing error amplification when `x` approaches `kLgammaMaxFiniteArgF32`
// (`4.085e36f`). Also sets `out_psi ~= digamma(x)` for correcting rounding in
// `y = 1 - x` during reflection.
__device__ float LgammaPosLargeF32(float x_gt4, float& out_s_pos,
                                   float& out_psi) {
  float log_lo;
  const float s = mi::LogF32HiLo(x_gt4, log_lo);
  float s_lo;
  const float h_x = mi::SplitHalfF32(s, s_lo);
  const float l_x = mi::AddRn(s_lo, log_lo);

  float b_err;
  const float b = mi::FastTwoSum(x_gt4, -0.5f, b_err);
  float b_lo;
  const float b_hi = mi::SplitHalfF32(b, b_lo);
  const float hx_m1 = mi::SubRn(h_x, 1.0f);
  float p_hi = mi::MulRn(b_hi, hx_m1);
  const float p_lo =
      mi::FmaRn(mi::AddRn(b_lo, b_err), hx_m1, mi::MulRn(b, l_x));
  const float inv_x = mi::DivRn(1.0f, x_gt4);
  const float sp = mi::Horner(mi::MulRn(inv_x, inv_x), mc::kLgammaStirlingF32);
  float p_rem;
  p_hi = mi::FastTwoSum(p_hi, mc::kLgammaCStirlingHiF32, p_rem);
  out_s_pos = mi::AddRn(
      p_rem, mi::AddRn(p_lo, mi::FmaRn(inv_x, sp, mc::kLgammaCStirlingLoF32)));
  out_psi = mi::FmaRn(-0.5f, inv_x, s);
  return p_hi;
}

// Evaluates `lgamma(x)` inside negative near-zero windows on `(-10, -2)` using
// `mc::kLgammaNegTableF32` (see `jaxlib/lgamma_coeffs.h` for the `[18][16]`
// layout):
//   - Column 1 (`in_basin`, `x` in `(-2.875, -2.5)`) expands around the local
//     minimum `x_min ~= -2.61072` where `lgamma(x_min) ~= -0.05663`
//     (`kLgammaBasinMinF32`) and the first derivative vanishes, so the leading
//     variation is quadratic: `kLgammaBasinMinF32 + u^2 * P(u)`.
//   - All other 15 columns expand linearly `u * P(u)` around a simple zero of
//     `lgamma`.
__device__ bool LgammaNegRootF32(float x, float n, float delta, float& out_rw) {
  const bool in_basin =
      (x > mc::kLgammaBasinLoF32) && (x < mc::kLgammaBasinHiF32);
  const int32_t n_i32 = static_cast<int32_t>(n);
  const int32_t base_idx = delta > 0.0f ? -3 : -4;
  int32_t idx = in_basin ? 1 : (base_idx - (n_i32 + n_i32));
  idx = idx < 0 ? 0 : (idx > 15 ? 15 : idx);

  const float r_hi = mc::kLgammaNegTableF32[0][idx];
  const float neg_r_lo_sc = mc::kLgammaNegTableF32[1][idx];
  const float scale = mc::kLgammaNegTableF32[2][idx];
  const float u_center = mc::kLgammaNegTableF32[3][idx];
  const float u_half = mc::kLgammaNegTableF32[4][idx];
  float u_lo;
  const float u = mi::FastTwoSum(mi::MulRn(mi::SubRn(delta, r_hi), scale),
                                 neg_r_lo_sc, u_lo);
  if (!(mi::AbsVal(mi::SubRn(u, u_center)) <= u_half)) {
    return false;
  }

  // Split `u` to 6 bits (`in_basin`, so `uh^2` fits in 12 bits) or 12 bits
  // (simple root) so `lead_hi * pw_hh` has zero rounding error.
  const int32_t u_mask = in_basin ? -0x40000 : -0x1000;
  const float uh = __int_as_float(__float_as_int(u) & u_mask);
  const float u_tail = mi::AddRn(mi::SubRn(u, uh), u_lo);
  const float lead_hi = in_basin ? mi::MulRn(uh, uh) : uh;
  const float lead_lo = in_basin ? mi::MulRn(u_tail, mi::AddRn(u, uh)) : u_tail;
  const float b_add_hi = in_basin ? mc::kLgammaBasinMinF32 : 0.0f;

  float p_tail = mc::kLgammaNegTableF32[17][idx];
  for (int row = 16; row >= 7; --row) {
    p_tail = mi::FmaRn(p_tail, u, mc::kLgammaNegTableF32[row][idx]);
  }
  float pw_lo;
  const float pw_hi = mi::FastTwoSum(mc::kLgammaNegTableF32[5][idx],
                                     mi::MulRn(u, p_tail), pw_lo);
  const float c0_lo = mc::kLgammaNegTableF32[6][idx];
  float pw_hl;
  const float pw_hh = mi::SplitHalfF32(pw_hi, pw_hl);
  float rw_err;
  const float rw_hi =
      mi::FastTwoSum(b_add_hi, mi::MulRn(lead_hi, pw_hh), rw_err);
  const float rw_lo =
      mi::FmaRn(lead_hi, mi::AddRn(pw_hl, mi::AddRn(pw_lo, c0_lo)),
                mi::MulRn(lead_lo, pw_hi));
  out_rw = mi::AddRn(rw_hi, mi::AddRn(rw_err, rw_lo));
  return true;
}

__device__ float LgammaF32Device(float x) {
  if (isnan(x)) {
    return mi::QuietNan<float>();
  }
  if (mi::AbsVal(x) < mc::kLgammaTinyF32 || x > mc::kLgammaMaxFiniteArgF32) {
    return mi::PosInf<float>();
  }

  if (x >= 0.5f) {
    float lg_lo;
    float lg_hi;
    if (x <= 4.0f) {
      lg_hi = LgammaPosSmallF32(x, lg_lo);
    } else {
      float psi_unused;
      lg_hi = LgammaPosLargeF32(x, lg_lo, psi_unused);
    }
    const float res = mi::AddRn(lg_hi, lg_lo);
    return res == 0.0f ? 0.0f : res;
  }

  // Non-positive integers (`0, -1, -2, ...`, including all `float32` values
  // `<= -2^23`) are poles where `lgamma(x) = +inf`.
  if (x <= mc::kLgammaAllIntegersBelowF32 || x == mi::FloorVal(x)) {
    return mi::PosInf<float>();
  }

  const float n = mi::RintRn(x);
  const float delta = mi::SubRn(x, n);

  if (x < -2.0f && n >= -9.0f) {
    float rw;
    if (LgammaNegRootF32(x, n, delta, rw)) {
      return rw == 0.0f ? 0.0f : rw;
    }
  }

  // Euler's reflection formula on `x < 0.5` outside root windows:
  //   lgamma(x) = -(lgamma(1 - x) + ln(|delta|) + ln(sin(pi*|delta|) /
  //               (pi*|delta|)))
  // Capture `y_lo = (1 - n - y) - delta` so `psi * y_lo` restores low bits of
  // `1 - x` lost when `1 - n` is large compared to `delta`.
  const float one_m_n = mi::SubRn(1.0f, n);
  const float y = mi::SubRn(one_m_n, delta);
  const float y_lo = mi::SubRn(mi::SubRn(one_m_n, y), delta);
  const float y_bounded = mi::MinVal(y, mc::kLgammaMaxFiniteArgF32);

  float log_lo;
  const float log_hi = mi::LogF32HiLo(mi::AbsVal(delta), log_lo);

  float lg_pos_hi;
  float lg_pos_lo;
  float psi;
  if (y_bounded <= 4.0f) {
    lg_pos_hi = LgammaPosSmallF32(y_bounded, lg_pos_lo);
    // Coarse cubic approximation of `digamma(y)` on `[1, 4]` (only multiplied
    // by `y_lo`, which is at most `0.5 * ulp(y)`).
    const float u = mi::SubRn(y_bounded, 1.0f);
    psi = mi::FmaRn(
        u,
        mi::FmaRn(u, mi::FmaRn(u, 1.0f / 18.0f, -5.0f / 12.0f), 49.0f / 36.0f),
        -0.57721566f);
  } else {
    lg_pos_hi = LgammaPosLargeF32(y_bounded, lg_pos_lo, psi);
  }

  // Split `delta` to 6 bits (`dh^2` fits in 12 bits) and multiply by the 12-bit
  // leading constant `kLgammaSincC0HhF32` of `-pi^2 / 6` so `sinc_hi` is exact.
  const float d2 = mi::MulRn(delta, delta);
  float dl;
  const float dh = mi::SplitBitsF32<6>(delta, dl);
  const float sinc_hi = mi::MulRn(mi::MulRn(dh, dh), mc::kLgammaSincC0HhF32);
  const float sinc_lo = mi::FmaRn(
      d2, mi::Horner(d2, mc::kLgammaSincRemF32),
      mi::MulRn(mi::MulRn(dl, mi::AddRn(delta, dh)), mc::kLgammaSincC0HhF32));
  float neg_e1;
  const float neg_hi = mi::FastTwoSum(log_hi, sinc_hi, neg_e1);
  float sum_err;
  const float sum_hi = mi::TwoSum(lg_pos_hi, neg_hi, sum_err);
  const float tail = mi::AddRn(
      mi::AddRn(mi::AddRn(sum_err, neg_e1), mi::AddRn(lg_pos_lo, log_lo)),
      mi::FmaRn(psi, y_lo, sinc_lo));
  const float refl = -mi::AddRn(sum_hi, tail);
  return refl == 0.0f ? 0.0f : refl;
}

// ---------------------------------------------------------------------------
// `LgammaF32WithF64`: `float32` via native `float64` (for fast-FP64 GPUs)
// ---------------------------------------------------------------------------

// Evaluates `float32` `lgamma(x)` in `float64` (~29 guard bits over `float32`),
// matching `_lgamma_f32_with_f64` in `jax/_src/lax/lgamma.py`:
//   - Within `3/64` of `x = 1` or `x = 2`, evaluates a degree-9 Taylor
//     polynomial around the zero (`kLgammaTaylor1F32F64` / `Taylor2`).
//   - Elsewhere on `x >= 0.5`, shifts `x < 8` into `[8, inf)` and evaluates
//     4-term Stirling minus `ln(prod)`.
//   - On `x < 0.5`, reflects via `1 - x` and combines `-ln(prod)` and
//     `ln(|delta|)` into `LogF64(|delta| / prod)`.
__device__ float LgammaF32WithF64Device(float x) {
  if (isnan(x)) {
    return mi::QuietNan<float>();
  }
  if (mi::AbsVal(x) < mc::kLgammaTinyF32 || x > mc::kLgammaMaxFiniteArgF32) {
    return mi::PosInf<float>();
  }

  if (x >= 0.5f) {
    const double x_d = static_cast<double>(x);
    constexpr double kNearTol = 3.0 / 64.0;
    const bool near_2 = mi::AbsVal(mi::SubRn(x_d, 2.0)) <= kNearTol;
    const double t = mi::SubRn(x_d, near_2 ? 2.0 : 1.0);
    if (mi::AbsVal(t) <= kNearTol) {
      const double poly = near_2 ? mi::Horner(t, mc::kLgammaTaylor2F32F64)
                                 : mi::Horner(t, mc::kLgammaTaylor1F32F64);
      const float res = __double2float_rn(mi::MulRn(t, poly));
      return res == 0.0f ? 0.0f : res;
    }

    double prod = 1.0;
    double z_d = x_d;
    double ln_prod = 0.0;
    if (x_d < 8.0) {
      const double m = mi::CeilVal(mi::SubRn(8.0, x_d));
      const int m_int = static_cast<int>(m);
      prod = x_d;
      for (int k = 1; k < m_int; ++k) {
        prod = mi::MulRn(prod, mi::AddRn(x_d, static_cast<double>(k)));
      }
      z_d = mi::AddRn(x_d, m);
      ln_prod = mi::LogF64(prod);
    }

    const double ln_z = mi::LogF64(z_d);
    const double inv_z = mi::DivRn(1.0, z_d);
    const double sp =
        mi::Horner(mi::MulRn(inv_z, inv_z), mc::kLgammaStirlingF32F64);
    const double lg_z =
        mi::FmaRn(mi::SubRn(z_d, 0.5), mi::SubRn(ln_z, 1.0),
                  mi::FmaRn(inv_z, sp, mc::kLgammaCStirlingHiF64));
    const float res = __double2float_rn(mi::SubRn(lg_z, ln_prod));
    return res == 0.0f ? 0.0f : res;
  }

  if (x <= mc::kLgammaAllIntegersBelowF32 || x == mi::FloorVal(x)) {
    return mi::PosInf<float>();
  }

  const float n = mi::RintRn(x);
  const float delta = mi::SubRn(x, n);
  const double y_d = mi::SubRn(1.0, static_cast<double>(x));

  double prod = 1.0;
  double z_d = y_d;
  if (y_d < 8.0) {
    const double m = mi::CeilVal(mi::SubRn(8.0, y_d));
    const int m_int = static_cast<int>(m);
    prod = y_d;
    for (int k = 1; k < m_int; ++k) {
      prod = mi::MulRn(prod, mi::AddRn(y_d, static_cast<double>(k)));
    }
    z_d = mi::AddRn(y_d, m);
  }

  const double ln_z = mi::LogF64(z_d);
  const double inv_z = mi::DivRn(1.0, z_d);
  const double sp =
      mi::Horner(mi::MulRn(inv_z, inv_z), mc::kLgammaStirlingF32F64);
  const double lg_z =
      mi::FmaRn(mi::SubRn(z_d, 0.5), mi::SubRn(ln_z, 1.0),
                mi::FmaRn(inv_z, sp, mc::kLgammaCStirlingHiF64));

  const double abs_delta_d = static_cast<double>(mi::AbsVal(delta));
  const double ln_arg = mi::LogF64(mi::DivRn(abs_delta_d, prod));
  const double d_d = static_cast<double>(delta);
  const double d2_d = mi::MulRn(d_d, d_d);
  const double sinc_d =
      mi::MulRn(d2_d, mi::Horner(d2_d, mc::kLgammaSincF32F64));
  const float res =
      __double2float_rn(-mi::AddRn(mi::AddRn(lg_z, ln_arg), sinc_d));
  return res == 0.0f ? 0.0f : res;
}

// ---------------------------------------------------------------------------
// `LgammaF64`: `float64` double-word `(hi, lo)` evaluation
// ---------------------------------------------------------------------------

// Evaluates Stirling's series on `z >= 7` in `(hi, lo)` `float64` form,
// keeping the low word `b_lo` of `z - 0.5` so `-0.5 * (ln(z) - 1)` is not lost
// when `z > 2^52`. Also returns `psi ~= digamma(z)` for first-order rounding
// corrections.
__device__ double LgammaStirlingF64(double z, double& res_l, double& psi) {
  double ll;
  const double lh = mi::LogF64HiLo(z, 0.0, ll);
  double l_m1_l;
  const double l_m1_h = mi::FastTwoSum(lh, -1.0, l_m1_l);
  l_m1_l = mi::AddRn(l_m1_l, ll);

  double b_lo;
  const double b = mi::FastTwoSum(z, -0.5, b_lo);
  double pl;
  double ph = mi::TwoProd(b, l_m1_h, pl);
  pl = mi::FmaRn(b_lo, l_m1_h, pl);
  ph = mi::FastTwoSum(ph, mi::FmaRn(b, l_m1_l, pl), pl);

  double sl;
  const double sh = mi::F2AddOrdered(ph, pl, mc::kLgammaCStirlingHiF64,
                                     mc::kLgammaCStirlingLoF64, sl);

  const double inv_z = mi::DivRn(1.0, z);
  const double inv_z2 = mi::MulRn(inv_z, inv_z);
  const double sp = mi::Horner(inv_z2, mc::kLgammaStirlingF64);
  const double res_h = mi::FastTwoSum(sh, mi::FmaRn(inv_z, sp, sl), res_l);

  const double psi_tail = mi::NegMulAddRn(
      inv_z2, mi::NegMulAddRn(inv_z2, 1.0 / 252.0, 1.0 / 120.0), 1.0 / 12.0);
  psi = mi::NegMulAddRn(inv_z2, psi_tail, mi::NegMulAddRn(0.5, inv_z, lh));
  return res_h;
}

// Shifts `w >= 1` into `z = w + m >= 7` (`m` in `{0, ..., 6}`), returning
// `(zh, zl) = lgamma(w + m)` (corrected by `digamma(z) * z_lo`), the unlogged
// shift product `(prod_h, prod_l)`, and `psi_y = digamma(w)`.
// Masking `w < 7` to 24 mantissa bits (`wh`) ensures `wh + k` (`k <= 5`) fits
// in 26 bits with zero rounding error so factor pairs can use
// `F2MulExactHighFma`.
__device__ void LgammaShiftedF64(double w, double& zh, double& zl,
                                 double& prod_h, double& prod_l,
                                 double& psi_y) {
  if (w >= 7.0) {
    prod_h = 1.0;
    prod_l = 0.0;
    zh = LgammaStirlingF64(w, zl, psi_y);
    return;
  }

  const double m = mi::CeilVal(mi::SubRn(7.0, w));
  const double wh = __longlong_as_double(__double_as_longlong(w) &
                                         mc::kLgammaKeep24MantissaMaskF64);
  const double wl = mi::SubRn(w, wh);

  const double fh0 = wh;
  const double fl0 = wl;
  const double fh1 = m > 1.0 ? mi::AddRn(wh, 1.0) : 1.0;
  const double fl1 = m > 1.0 ? wl : 0.0;
  const double fh2 = m > 2.0 ? mi::AddRn(wh, 2.0) : 1.0;
  const double fl2 = m > 2.0 ? wl : 0.0;
  const double fh3 = m > 3.0 ? mi::AddRn(wh, 3.0) : 1.0;
  const double fl3 = m > 3.0 ? wl : 0.0;
  const double fh4 = m > 4.0 ? mi::AddRn(wh, 4.0) : 1.0;
  const double fl4 = m > 4.0 ? wl : 0.0;
  const double fh5 = m > 5.0 ? mi::AddRn(wh, 5.0) : 1.0;
  const double fl5 = m > 5.0 ? wl : 0.0;

  double p01_l, p23_l, p45_l, p0123_l;
  const double p01_h = mi::F2MulExactHighFma(fh0, fl0, fh1, fl1, p01_l);
  const double p23_h = mi::F2MulExactHighFma(fh2, fl2, fh3, fl3, p23_l);
  const double p45_h = mi::F2MulExactHighFma(fh4, fl4, fh5, fl5, p45_l);
  const double p0123_h = mi::F2Mul(p01_h, p01_l, p23_h, p23_l, p0123_l);
  prod_h = mi::F2Mul(p0123_h, p0123_l, p45_h, p45_l, prod_l);

  const double z = mi::AddRn(w, m);
  const double z_lo = mi::SubRn(w, mi::SubRn(z, m));
  double psi_z;
  zh = LgammaStirlingF64(z, zl, psi_z);
  zl = mi::FmaRn(psi_z, z_lo, zl);
  psi_y = mi::Horner(w, mc::kLgammaPsi15F64);
}

// Evaluates `lgamma(x)` near the negative zeros on `(-17, -2)` using
// `mc::kLgammaNegTableF64` (see `jaxlib/lgamma_coeffs.h` for the `[21][30]`
// layout). Returns `false` when `x` lies outside all root windows so the caller
// falls back to Euler's reflection formula.
__device__ bool LgammaNegRootF64(double x, double n, double& rw_out) {
  // Pin `n_root = -2` across `(-2.6, -2)` so column 0 covers the zero at
  // `-2.3136...` without splitting across `round(x)`'s jump at `-2.5`.
  const bool in_m26_m2 = (x > -2.6) && (x < -2.0);
  const double n_root = in_m26_m2 ? -2.0 : n;
  if (!(x < -2.0 && n_root >= -16.0)) {
    return false;
  }

  const double delta_root = mi::SubRn(x, n_root);
  const int32_t n_i32 = static_cast<int32_t>(n_root);
  const int32_t base_idx = delta_root > 0.0 ? -3 : -4;
  int32_t idx = base_idx - (n_i32 + n_i32);
  idx = idx < 0 ? 0 : (idx > 29 ? 29 : idx);

  const double r0 = mc::kLgammaNegTableF64[0][idx];
  const double neg_r1_sc = mc::kLgammaNegTableF64[1][idx];
  const double scale = mc::kLgammaNegTableF64[2][idx];
  const double u_bound = mc::kLgammaNegTableF64[3][idx];

  double u_l;
  const double u_h = mi::FastTwoSum(mi::MulRn(mi::SubRn(delta_root, r0), scale),
                                    neg_r1_sc, u_l);
  if (!(mi::AbsVal(u_h) <= u_bound)) {
    return false;
  }

  double p_tail = mc::kLgammaNegTableF64[20][idx];
  for (int row = 19; row >= 6; --row) {
    p_tail = mi::FmaRn(p_tail, u_h, mc::kLgammaNegTableF64[row][idx]);
  }
  double pw_l;
  const double pw_h = mi::FastTwoSum(mc::kLgammaNegTableF64[4][idx],
                                     mi::MulRn(u_h, p_tail), pw_l);
  pw_l = mi::AddRn(pw_l, mc::kLgammaNegTableF64[5][idx]);

  double rw_l;
  const double rw_h = mi::TwoProd(u_h, pw_h, rw_l);
  rw_out = mi::AddRn(rw_h, mi::FmaRn(u_h, pw_l, mi::FmaRn(u_l, pw_h, rw_l)));
  return true;
}

__device__ double LgammaF64Device(double x) {
  if (isnan(x)) {
    return mi::QuietNan<double>();
  }
  if (mi::AbsVal(x) < mc::kLgammaTinyF64 || x > mc::kLgammaMaxFiniteArgF64) {
    return mi::PosInf<double>();
  }

  if (x >= 7.0) {
    double zl, psi_unused;
    const double zh = LgammaStirlingF64(x, zl, psi_unused);
    return mi::AddRn(zh, zl);
  }

  if (x > 0.0) {
    if (x < 2.5) {
      // Shift `(0, 1)` into `(1, 2)` via `lgamma(x) = lgamma(x + 1) - ln(x)`
      // and evaluate `(t - 1)(t - 2) * P(t - 1.75)` on `[1, 2.5)`.
      const bool is_lt1 = x < 1.0;
      const double u = is_lt1 ? x : mi::SubRn(x, 1.0);
      const double v = mi::SubRn(u, 1.0);
      const double d_arg = mi::SubRn(u, 0.75);
      double pl_poly;
      const double ph_poly = mi::FastTwoSum(
          mc::kLgammaPos12C0HiF64,
          mi::MulRn(d_arg, mi::Horner(d_arg, mc::kLgammaPos12TailF64)),
          pl_poly);
      pl_poly = mi::AddRn(pl_poly, mc::kLgammaPos12C0LoF64);
      double uv_l;
      const double uv_h = mi::TwoProd(u, v, uv_l);
      double p_l;
      const double p_h = mi::F2Mul(uv_h, uv_l, ph_poly, pl_poly, p_l);

      double res;
      if (is_lt1) {
        double log_l;
        const double log_h = mi::LogF64HiLo(x, 0.0, log_l);
        res = mi::F2Sum(p_h, p_l, -log_h, -log_l);
      } else {
        res = mi::AddRn(p_h, p_l);
      }
      return res == 0.0 ? 0.0 : res;
    }

    double zh, zl, prod_h, prod_l, psi_unused;
    LgammaShiftedF64(x, zh, zl, prod_h, prod_l, psi_unused);
    double log_l;
    const double log_h = mi::LogF64HiLo(prod_h, 0.0, log_l);
    log_l = mi::AddRn(log_l, mi::DivRn(prod_l, prod_h));
    const double res = mi::F2Sum(zh, zl, -log_h, -log_l);
    return res == 0.0 ? 0.0 : res;
  }

  // Non-positive integers (`0, -1, -2, ...`, including all `float64` values
  // `<= -2^52`) are poles where `lgamma(x) = +inf`.
  if (x <= mc::kLgammaAllIntegersBelowF64 || x == mi::FloorVal(x)) {
    return mi::PosInf<double>();
  }

  const double n = mi::RintRn(x);
  double rw;
  if (LgammaNegRootF64(x, n, rw)) {
    return rw == 0.0 ? 0.0 : rw;
  }

  // Euler's reflection formula on `x < 0` outside root windows. Pre-scaling
  // tiny `|delta| < 1e-290` by `2^54` (`18014398509481984.0`) prevents
  // `|delta| / prod_h` from underflowing to subnormal, compensated by `k_adj =
  // -54.0` in `LogF64HiLo`.
  const double delta = mi::SubRn(x, n);
  const double abs_delta = mi::AbsVal(delta);
  const double one_m_n = mi::SubRn(1.0, n);
  const double y = mi::SubRn(one_m_n, delta);
  const double y_lo = mi::SubRn(mi::SubRn(one_m_n, y), delta);
  const double y_bounded = mi::MinVal(y, mc::kLgammaMaxFiniteArgF64);

  double zh, zl, prod_h, prod_l, psi_y;
  LgammaShiftedF64(y_bounded, zh, zl, prod_h, prod_l, psi_y);

  const bool is_tiny_neg = abs_delta < 1e-290;
  const double abs_d_s =
      is_tiny_neg ? mi::MulRn(abs_delta, 18014398509481984.0) : abs_delta;
  const double rcp_both = mi::DivRn(1.0, mi::MulRn(prod_h, abs_d_s));
  const double rcp_prod = mi::MulRn(abs_d_s, rcp_both);
  const double qh = mi::MulRn(abs_d_s, rcp_prod);
  const double rem_q =
      mi::NegMulAddRn(qh, prod_l, mi::NegMulAddRn(qh, prod_h, abs_d_s));

  const double k_adj = is_tiny_neg ? -54.0 : 0.0;
  double log_l;
  const double log_h = mi::LogF64HiLo(qh, k_adj, log_l);
  log_l = mi::AddRn(log_l, mi::MulRn(rem_q, mi::MulRn(prod_h, rcp_both)));

  double d2_l;
  const double d2_h = mi::TwoProd(delta, delta, d2_l);
  double sl_sinc;
  const double sh_sinc = mi::FastTwoSum(
      mc::kLgammaSincC0HiF64,
      mi::MulRn(d2_h, mi::Horner(d2_h, mc::kLgammaSincTailF64)), sl_sinc);
  sl_sinc = mi::AddRn(sl_sinc, mc::kLgammaSincC0LoF64);
  double sinc_l;
  const double sinc_h = mi::F2Mul(d2_h, d2_l, sh_sinc, sl_sinc, sinc_l);

  double sum_l;
  const double sum_h = mi::F2Add(zh, zl, log_h, log_l, sum_l);
  const double refl =
      -mi::F2Sum(sum_h, sum_l, sinc_h, mi::FmaRn(psi_y, y_lo, sinc_l));
  return refl == 0.0 ? 0.0 : refl;
}

// ---------------------------------------------------------------------------
// Grid-stride `__global__` entry points (`BF16`, `F16`, `F32`, `F64`)
// ---------------------------------------------------------------------------

__global__ void LgammaF32Kernel(const void* in, void* out, int64_t n,
                                xla::ffi::DataType dtype) {
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float x = mi::LoadNarrowOrF32(in, idx, dtype);
    mi::StoreNarrowOrF32(out, idx, dtype, LgammaF32Device(x));
  }
}

__global__ void LgammaF32WithF64Kernel(const void* in, void* out, int64_t n,
                                       xla::ffi::DataType dtype) {
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float x = mi::LoadNarrowOrF32(in, idx, dtype);
    mi::StoreNarrowOrF32(out, idx, dtype, LgammaF32WithF64Device(x));
  }
}

__global__ void LgammaF64Kernel(const double* in, double* out, int64_t n) {
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[idx] = LgammaF64Device(in[idx]);
  }
}

constexpr int kBlockDim = 256;
constexpr int64_t kMaxGridDim = 65535;

}  // namespace

void LaunchLgammaF32Kernel(gpuStream_t stream, int64_t n,
                           xla::ffi::DataType dtype, const void* in,
                           void* out) {
  if (n <= 0) return;
  const int64_t grid_dim =
      std::min<int64_t>(kMaxGridDim, (n + kBlockDim - 1) / kBlockDim);
  LgammaF32Kernel<<<static_cast<unsigned int>(grid_dim), kBlockDim, 0,
                    stream>>>(in, out, n, dtype);
}

void LaunchLgammaF32WithF64Kernel(gpuStream_t stream, int64_t n,
                                  xla::ffi::DataType dtype, const void* in,
                                  void* out) {
  if (n <= 0) return;
  const int64_t grid_dim =
      std::min<int64_t>(kMaxGridDim, (n + kBlockDim - 1) / kBlockDim);
  LgammaF32WithF64Kernel<<<static_cast<unsigned int>(grid_dim), kBlockDim, 0,
                           stream>>>(in, out, n, dtype);
}

void LaunchLgammaF64Kernel(gpuStream_t stream, int64_t n, const double* in,
                           double* out) {
  if (n <= 0) return;
  const int64_t grid_dim =
      std::min<int64_t>(kMaxGridDim, (n + kBlockDim - 1) / kBlockDim);
  LgammaF64Kernel<<<static_cast<unsigned int>(grid_dim), kBlockDim, 0,
                    stream>>>(in, out, n);
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
