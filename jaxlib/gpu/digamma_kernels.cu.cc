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

// CUDA and ROCm device kernels for `digamma(x) = d/dx ln(|Gamma(x)|)` on
// `bfloat16`, `float16`, `float32`, and `float64`, accurate to within 1 ULP
// across the full input domain (matching the three implementations in
// `jax/_src/lax/digamma.py` with hardware FMA):
//
// 1. `DigammaF32Kernel` (`float32`-only, also shared by `bfloat16` and
//    `float16` via load/store conversion):
//    Used on GPUs with low-throughput `float64` units. Evaluates `digamma` in
//    `(hi, lo)` double-word `float32` arithmetic (`DigammaF2Device<float>`),
//    avoiding all `float64` instructions.
//
// 2. `DigammaF32WithF64Kernel` (`float32` via native `float64`, also shared by
//    `bfloat16` and `float16`):
//    Used on GPUs with high-throughput `float64` units (`1 <= perf_ratio <= 4`,
//    such as A100, H100, B200, MI250, and MI300), where promoting to `float64`
//    needs only a single `+4` recurrence shift (`z >= 4.5`) and 4-term
//    polynomials.
//
// 3. `DigammaF64Kernel` (`float64`):
//    Evaluates `digamma` in `(hi, lo)` double-word `float64` arithmetic
//    (`DigammaF2Device<double>`).

#include "jaxlib/gpu/digamma_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "jaxlib/digamma_coeffs.h"
#include "jaxlib/gpu/double_word.cu.h"
#include "jaxlib/gpu/log.cu.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace mi = jax::JAX_GPU_NAMESPACE::math_internal;
namespace mc = jax::math_coeffs;

// Evaluates one `+4` recurrence step `1/c + 1/(c+1) + 1/(c+2) + 1/(c+3)` for
// double-word `c = c0 + c1 >= 0.5` using a single double-word division by
// pairing terms symmetrically around `m = c + 1.5`:
//   u = m^2 - 1.25  =>  sum = 4 * m * u / (u^2 - 1).
// Because `c >= 0.5`, `m >= 2` and `u >= 2.75`, so `u^2 - 1 >= 6.5625` stays
// well away from zero.
template <typename T>
__device__ T Shift4F2(T c0, T c1, T& out1) {
  T e;
  const T s = mi::TwoSum(c0, T(1.5), e);
  T m1;
  const T m0 = mi::FastTwoSum(s, mi::AddRn(e, c1), m1);
  T msq1;
  const T msq0 = mi::F2Square(m0, m1, msq1);
  T u1;
  const T u0 = mi::FastTwoSum(msq0, T(-1.25), u1);
  u1 = mi::AddRn(u1, msq1);
  T usq1;
  const T usq0 = mi::F2Square(u0, u1, usq1);
  T den1;
  const T den0 = mi::FastTwoSum(usq0, T(-1.0), den1);
  den1 = mi::AddRn(den1, usq1);
  T num1;
  const T num0 =
      mi::F2Mul(mi::MulRn(T(4), m0), mi::MulRn(T(4), m1), u0, u1, num1);
  return mi::F2Div(num0, num1, den0, den1, out1);
}

// Evaluates `digamma(x)` in `(hi, lo)` double-word arithmetic in `T` (`float`
// for `DigammaF32Kernel`, `double` for `DigammaF64Kernel`), matching
// `_digamma_f2` in `jax/_src/lax/digamma.py`:
//   - Near the positive zero `r_+ ~= 1.461632145` (within `2^-8` for `float32`,
//     `1e-4` for `float64`), evaluates a short Taylor series directly around
//     `kDigammaRposHi*` so `ln(z)` and the recurrence sum do not cancel.
//   - For `x < 0.5`, checks for negative-integer poles (`NaN`), checks the
//     precomputed near-root override table (`kDigammaOverridesIn*`) where
//     `digamma(1 - x)` and `pi * cot(pi * delta)` cancel past `(hi, lo)`
//     precision, and sets `z = (1 - n) - delta` in `(hi, lo)` form.
//   - Shifts `z < kZMin` (`4.5` for `float32`, `12.5` for `float64`) upward in
//     steps of `+4`, evaluates `ln(z) - 1/(2z) - (1/z^2) * P_stir(1/z^2)`, and
//     adds `-pi * cot(pi * delta)` for `x < 0.5`.
template <typename T>
__device__ T DigammaF2Device(T x) {
  constexpr bool kIsF32 = std::is_same_v<T, float>;
  constexpr T kTiny = kIsF32 ? mc::kDigammaTinyF32 : mc::kDigammaTinyF64;
  constexpr T kAllIntBelow =
      kIsF32 ? mc::kDigammaAllIntBelowF32 : mc::kDigammaAllIntBelowF64;
  constexpr T kZMin = kIsF32 ? mc::kDigammaZMinF32 : mc::kDigammaZMinF64;
  constexpr T kZBig = kIsF32 ? mc::kDigammaZBigF32 : mc::kDigammaZBigF64;
  constexpr T kRecipScale =
      kIsF32 ? mc::kDigammaRecipScaleF32 : mc::kDigammaRecipScaleF64;

  if (isnan(x)) {
    return mi::QuietNan<T>();
  }
  const T ax = mi::AbsVal(x);
  if (ax < kTiny) {
    return -mi::CopySignVal(mi::PosInf<T>(), x);
  }
  if (x == mi::PosInf<T>()) {
    return mi::PosInf<T>();
  }

  // Direct Taylor expansion around the positive zero `r_+ ~= 1.461632145`.
  if constexpr (kIsF32) {
    if (mi::AbsVal(mi::SubRn(x, mc::kDigammaRposHiF32)) <
        mc::kDigammaRposTolF32) {
      const float t = mi::SubRn(x, mc::kDigammaRposHiF32);
      const float tail = mi::FmaRn(
          t, mi::FmaRn(mc::kDigammaHalfPsi2F32, t, mc::kDigammaPsi1Lo16F32),
          mc::kDigammaPsiOffsetF32);
      return mi::AddRn(mi::MulRn(t, mc::kDigammaPsi1Hi16F32), tail);
    }
  } else {
    if (mi::AbsVal(mi::SubRn(x, mc::kDigammaRposHiF64)) <
        mc::kDigammaRposTolF64) {
      const double t = mi::SubRn(x, mc::kDigammaRposHiF64);
      double p1, s1;
      const double p0 = mi::TwoProd(t, mc::kDigammaPsi1HiF64, p1);
      const double s0 = mi::TwoSum(mc::kDigammaPsi0HiF64, p0, s1);
      const double tail_poly = mi::FmaRn(
          t,
          mi::FmaRn(t, mi::FmaRn(t, mc::kDigammaPsi4F64, mc::kDigammaPsi3F64),
                    mc::kDigammaPsi2F64),
          mc::kDigammaPsi1LoF64);
      const double tail = mi::FmaRn(t, tail_poly, mc::kDigammaPsi0LoF64);
      return mi::AddRn(s0, mi::AddRn(s1, mi::AddRn(p1, tail)));
    }
  }

  const bool is_pos = x >= T(0.5);
  T delta = T(0);
  T z0;
  T z1;
  if (is_pos) {
    z0 = x;
    z1 = T(0);
  } else {
    if (!(x > kAllIntBelow)) {
      return mi::QuietNan<T>();
    }
    const T n = mi::RintRn(x);
    delta = mi::SubRn(x, n);
    if (delta == T(0)) {
      return mi::QuietNan<T>();
    }
    if constexpr (kIsF32) {
      for (size_t k = 0; k < 3; ++k) {
        if (x == mc::kDigammaOverridesInF32[k]) {
          return mc::kDigammaOverridesOutF32[k];
        }
      }
    } else {
      for (size_t k = 0; k < 10; ++k) {
        if (x == mc::kDigammaOverridesInF64[k]) {
          return mc::kDigammaOverridesOutF64[k];
        }
      }
    }
    z0 = mi::FastTwoSum(mi::SubRn(T(1), n), -delta, z1);
  }

  T acc0 = T(0);
  T acc1 = T(0);
  while (z0 < kZMin || (z0 == kZMin && z1 < T(0))) {
    T h1;
    const T h0 = Shift4F2(z0, z1, h1);
    acc0 = mi::F2Add(acc0, acc1, -h0, -h1, acc1);
    z0 = mi::F2Add(z0, z1, T(4), T(0), z1);
  }

  // Above `kZBig` (`1e12f` for `float32`, `1e80` for `float64`), `1/(2z)` is
  // negligible compared to `ln(z)` and `1/z^2` would underflow.
  T iz0 = T(0);
  T iz1 = T(0);
  if (z0 < kZBig) {
    iz0 = mi::F2Recip(z0, z1, iz1);
  }
  acc0 = mi::F2Add(acc0, acc1, mi::MulRn(T(-0.5), iz0), mi::MulRn(T(-0.5), iz1),
                   acc1);

  T lg1;
  const T lg0 = mi::LogF2WithRecip(z0, z1, iz0, iz1, lg1);
  acc0 = mi::F2Add(acc0, acc1, lg0, lg1, acc1);

  T u1, sp1;
  const T u0 = mi::F2Square(iz0, iz1, u1);
  T sp0;
  if constexpr (kIsF32) {
    sp0 = mi::HornerHybridF2(u0, u1, mc::kDigammaStirLoDegHiF32,
                             mc::kDigammaStirLoDegLoF32,
                             mc::kDigammaStirHiDegF32, sp1);
  } else {
    sp0 = mi::HornerHybridF2(u0, u1, mc::kDigammaStirLoDegHiF64,
                             mc::kDigammaStirLoDegLoF64,
                             mc::kDigammaStirHiDegF64, sp1);
  }
  sp0 = mi::F2Mul(sp0, sp1, -u0, -u1, sp1);
  acc0 = mi::F2Add(acc0, acc1, sp0, sp1, acc1);

  if (is_pos) {
    return mi::AddRn(acc0, acc1);
  }

  // Reflection term `-pi * cot(pi * delta)` decomposed into `-1/delta` (scaled
  // by `kRecipScale` before `F2Recip` so subnormal negative `delta` does not
  // overflow `1/delta`) plus `delta * ((10 - 4*w)/(4 - 5*w + w^2) + C(w))`.
  T ids1;
  const T ids0 = mi::F2Recip(mi::MulRn(delta, kRecipScale), T(0), ids1);
  acc0 = mi::F2Add(acc0, acc1, mi::MulRn(-kRecipScale, ids0),
                   mi::MulRn(-kRecipScale, ids1), acc1);

  T dw1, rn1, wm5_1, rd1, rat1, ct1, cp1;
  const T dw0 = mi::TwoProd(delta, delta, dw1);
  const T rn0 =
      mi::F2Add(T(10), T(0), mi::MulRn(T(-4), dw0), mi::MulRn(T(-4), dw1), rn1);
  const T wm5_0 = mi::F2Add(T(-5), T(0), dw0, dw1, wm5_1);
  T rd0 = mi::F2Mul(wm5_0, wm5_1, dw0, dw1, rd1);
  rd0 = mi::F2Add(T(4), T(0), rd0, rd1, rd1);
  const T rat0 = mi::F2Div(rn0, rn1, rd0, rd1, rat1);

  T ct0;
  if constexpr (kIsF32) {
    ct0 = mi::HornerHybridF2(dw0, dw1, mc::kDigammaCotLoDegHiF32,
                             mc::kDigammaCotLoDegLoF32, mc::kDigammaCotHiDegF32,
                             ct1);
  } else {
    ct0 = mi::HornerHybridF2(dw0, dw1, mc::kDigammaCotLoDegHiF64,
                             mc::kDigammaCotLoDegLoF64, mc::kDigammaCotHiDegF64,
                             ct1);
  }
  T cp0 = mi::F2Add(ct0, ct1, rat0, rat1, cp1);
  cp0 = mi::F2MulScalar(cp0, cp1, delta, cp1);
  return mi::F2Sum(acc0, acc1, cp0, cp1);
}

// Evaluates `float32` `digamma(x)` via native `float64` (for fast-FP64 GPUs),
// matching `_digamma_f32_with_f64` in `jax/_src/lax/digamma.py`: at most one
// `+4` shift into `[4.5, inf)`, 4-term `kDigammaStirF32F64`, pole-removed
// `kDigammaCotF32F64` on `x < 0.5`, and the 3 near-root overrides in
// `kDigammaOverridesInF32`.
__device__ float DigammaF32WithF64Device(float x) {
  if (isnan(x)) {
    return mi::QuietNan<float>();
  }
  const float ax = mi::AbsVal(x);
  if (ax < mc::kDigammaTinyF32) {
    return -mi::CopySignVal(mi::PosInf<float>(), x);
  }
  if (x == mi::PosInf<float>()) {
    return mi::PosInf<float>();
  }

  const bool is_pos = x >= 0.5f;
  double z;
  double dd = 0.0;
  if (is_pos) {
    z = static_cast<double>(x);
  } else {
    if (!(x > mc::kDigammaAllIntBelowF32)) {
      return mi::QuietNan<float>();
    }
    const float n = mi::RintRn(x);
    const float delta = mi::SubRn(x, n);
    if (delta == 0.0f) {
      return mi::QuietNan<float>();
    }
    for (size_t k = 0; k < 3; ++k) {
      if (x == mc::kDigammaOverridesInF32[k]) {
        return mc::kDigammaOverridesOutF32[k];
      }
    }
    dd = static_cast<double>(delta);
    z = mi::SubRn(mi::SubRn(1.0, static_cast<double>(n)), dd);
  }

  double acc = 0.0;
  if (z < 4.5) {
    const double m = mi::AddRn(z, 1.5);
    const double q = mi::MulRn(m, m);
    const double num = mi::MulRn(mi::MulRn(4.0, m), mi::SubRn(q, 1.25));
    const double den = mi::FmaRn(mi::SubRn(q, 2.5), q, 0.5625);
    acc = -mi::DivRn(num, den);
    z = mi::AddRn(z, 4.0);
  }

  const double iz = mi::DivRn(1.0, z);
  const double u = mi::MulRn(iz, iz);
  const double st = mi::Horner(u, mc::kDigammaStirF32F64);
  const double psi_z = mi::NegMulAddRn(
      u, st, mi::NegMulAddRn(0.5, iz, mi::AddRn(acc, mi::LogF64(z))));

  if (is_pos) {
    return __double2float_rn(psi_z);
  }

  const double w = mi::MulRn(dd, dd);
  const double ct = mi::Horner(w, mc::kDigammaCotF32F64);
  const double rat_num = mi::NegMulAddRn(4.0, w, 10.0);
  const double rat_den = mi::FmaRn(mi::SubRn(w, 5.0), w, 4.0);
  const double cot = mi::FmaRn(dd, mi::AddRn(mi::DivRn(rat_num, rat_den), ct),
                               -mi::DivRn(1.0, dd));
  return __double2float_rn(mi::AddRn(psi_z, cot));
}

// ---------------------------------------------------------------------------
// Grid-stride `__global__` entry points (`BF16`, `F16`, `F32`, `F64`)
// ---------------------------------------------------------------------------

__global__ void DigammaF32Kernel(const void* in, void* out, int64_t n,
                                 xla::ffi::DataType dtype) {
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float x = mi::LoadNarrowOrF32(in, idx, dtype);
    mi::StoreNarrowOrF32(out, idx, dtype, DigammaF2Device<float>(x));
  }
}

__global__ void DigammaF32WithF64Kernel(const void* in, void* out, int64_t n,
                                        xla::ffi::DataType dtype) {
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float x = mi::LoadNarrowOrF32(in, idx, dtype);
    mi::StoreNarrowOrF32(out, idx, dtype, DigammaF32WithF64Device(x));
  }
}

__global__ void DigammaF64Kernel(const double* in, double* out, int64_t n) {
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[idx] = DigammaF2Device<double>(in[idx]);
  }
}

constexpr int kBlockDim = 256;
constexpr int64_t kMaxGridDim = 65535;

}  // namespace

void LaunchDigammaF32Kernel(gpuStream_t stream, int64_t n,
                            xla::ffi::DataType dtype, const void* in,
                            void* out) {
  if (n <= 0) return;
  const int64_t grid_dim =
      std::min<int64_t>(kMaxGridDim, (n + kBlockDim - 1) / kBlockDim);
  DigammaF32Kernel<<<static_cast<unsigned int>(grid_dim), kBlockDim, 0,
                     stream>>>(in, out, n, dtype);
}

void LaunchDigammaF32WithF64Kernel(gpuStream_t stream, int64_t n,
                                   xla::ffi::DataType dtype, const void* in,
                                   void* out) {
  if (n <= 0) return;
  const int64_t grid_dim =
      std::min<int64_t>(kMaxGridDim, (n + kBlockDim - 1) / kBlockDim);
  DigammaF32WithF64Kernel<<<static_cast<unsigned int>(grid_dim), kBlockDim, 0,
                            stream>>>(in, out, n, dtype);
}

void LaunchDigammaF64Kernel(gpuStream_t stream, int64_t n, const double* in,
                            double* out) {
  if (n <= 0) return;
  const int64_t grid_dim =
      std::min<int64_t>(kMaxGridDim, (n + kBlockDim - 1) / kBlockDim);
  DigammaF64Kernel<<<static_cast<unsigned int>(grid_dim), kBlockDim, 0,
                     stream>>>(in, out, n);
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
