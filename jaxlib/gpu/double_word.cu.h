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

// Uncontracted IEEE 754 round-to-nearest-even arithmetic, FMA double-word
// `(hi, lo)` building blocks (matching `jax/_src/lax/double_word.py`), and
// narrow-float (`bfloat16` / `float16`) load/store helpers for CUDA and ROCm
// math FFI kernels.

#ifndef JAXLIB_GPU_DOUBLE_WORD_CU_H_
#define JAXLIB_GPU_DOUBLE_WORD_CU_H_

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

#if defined(JAX_GPU_CUDA)
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#elif defined(JAX_GPU_HIP)
#include "rocm/include/hip/hip_fp16.h"
#endif

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace math_internal {

// ---------------------------------------------------------------------------
// Uncontracted IEEE 754 round-to-nearest-even scalar device operations
// ---------------------------------------------------------------------------
//
// Double-word `(hi, lo)` algorithms (`TwoSum`, `FastTwoSum`, `TwoProd`, etc.)
// rely on individual `+`, `-`, and `*` operations rounding to nearest-even
// without being fused with neighboring multiplies into FMA instructions:
//   - On CUDA (`nvcc` and `clang`), `__fadd_rn`, `__dadd_rn`, `__fsub_rn`,
//     `__dsub_rn`, `__fmul_rn`, `__dmul_rn`, `__fdiv_rn`, and `__ddiv_rn` emit
//     explicit `.rn` PTX instructions that are never fused into FMA, even under
//     `nvcc`'s default `--fmad=true`.
//   - On ROCm (`hipcc` / `clang`), `__clang_hip_math.h` lowers `__fadd_rn(a,
//     b)` to plain `a + b` unless `OCML_BASIC_ROUNDED_OPERATIONS` is set, so we
//     disable floating-point contraction explicitly with
//     `_Pragma("clang fp contract(off)")` inside each helper.

#if defined(JAX_GPU_CUDA)

__device__ inline float AddRn(float a, float b) { return __fadd_rn(a, b); }
__device__ inline double AddRn(double a, double b) { return __dadd_rn(a, b); }

__device__ inline float SubRn(float a, float b) { return __fsub_rn(a, b); }
__device__ inline double SubRn(double a, double b) { return __dsub_rn(a, b); }

__device__ inline float MulRn(float a, float b) { return __fmul_rn(a, b); }
__device__ inline double MulRn(double a, double b) { return __dmul_rn(a, b); }

__device__ inline float DivRn(float a, float b) { return __fdiv_rn(a, b); }
__device__ inline double DivRn(double a, double b) { return __ddiv_rn(a, b); }

#elif defined(JAX_GPU_HIP) && defined(__clang__)

__device__ inline float AddRn(float a, float b) {
  _Pragma("clang fp contract(off)") return a + b;
}
__device__ inline double AddRn(double a, double b) {
  _Pragma("clang fp contract(off)") return a + b;
}

__device__ inline float SubRn(float a, float b) {
  _Pragma("clang fp contract(off)") return a - b;
}
__device__ inline double SubRn(double a, double b) {
  _Pragma("clang fp contract(off)") return a - b;
}

__device__ inline float MulRn(float a, float b) {
  _Pragma("clang fp contract(off)") return a * b;
}
__device__ inline double MulRn(double a, double b) {
  _Pragma("clang fp contract(off)") return a * b;
}

__device__ inline float DivRn(float a, float b) {
  _Pragma("clang fp contract(off)") return a / b;
}
__device__ inline double DivRn(double a, double b) {
  _Pragma("clang fp contract(off)") return a / b;
}

#else
#error "Unsupported GPU compiler: uncontracted IEEE rounding is required."
#endif

__device__ inline float FmaRn(float a, float b, float c) {
  return __fmaf_rn(a, b, c);
}
__device__ inline double FmaRn(double a, double b, double c) {
  return __fma_rn(a, b, c);
}

template <typename T>
__device__ T NegMulAddRn(T a, T b, T c) {
  return FmaRn(-a, b, c);
}

__device__ inline float AbsVal(float x) { return fabsf(x); }
__device__ inline double AbsVal(double x) { return fabs(x); }

__device__ inline float RintRn(float x) { return rintf(x); }
__device__ inline double RintRn(double x) { return rint(x); }

__device__ inline float FloorVal(float x) { return floorf(x); }
__device__ inline double FloorVal(double x) { return floor(x); }

__device__ inline float CeilVal(float x) { return ceilf(x); }
__device__ inline double CeilVal(double x) { return ceil(x); }

__device__ inline float MinVal(float a, float b) { return a < b ? a : b; }
__device__ inline double MinVal(double a, double b) { return a < b ? a : b; }

__device__ inline float MaxVal(float a, float b) { return a > b ? a : b; }
__device__ inline double MaxVal(double a, double b) { return a > b ? a : b; }

__device__ inline float CopySignVal(float mag, float sgn) {
  return copysignf(mag, sgn);
}
__device__ inline double CopySignVal(double mag, double sgn) {
  return copysign(mag, sgn);
}

template <typename T>
__device__ T PosInf();
template <>
__device__ inline float PosInf<float>() {
  return __int_as_float(0x7f800000);
}
template <>
__device__ inline double PosInf<double>() {
  return __longlong_as_double(0x7ff0000000000000LL);
}

template <typename T>
__device__ T QuietNan();
template <>
__device__ inline float QuietNan<float>() {
  return __int_as_float(0x7fc00000);
}
template <>
__device__ inline double QuietNan<double>() {
  return __longlong_as_double(0x7ff8000000000000LL);
}

// Splits a `float32` value `a` into `ah + out_lo == a`, where `ah` keeps the
// top `KeepBits` mantissa bits (including the implicit leading 1 bit, `1 <=
// KeepBits <= 23`) by zeroing the bottom `24 - KeepBits` bits. Products of two
// values split to `<= 12` bits fit in 24 bits with zero rounding error.
template <int KeepBits>
__device__ float SplitBitsF32(float a, float& out_lo) {
  static_assert(KeepBits >= 1 && KeepBits <= 23);
  constexpr int32_t kMask = -(1 << (24 - KeepBits));
  const float ah = __int_as_float(__float_as_int(a) & kMask);
  out_lo = SubRn(a, ah);
  return ah;
}

__device__ inline float SplitHalfF32(float a, float& out_lo) {
  return SplitBitsF32<12>(a, out_lo);
}

// ---------------------------------------------------------------------------
// Double-word `(hi, lo)` arithmetic using hardware FMA
// ---------------------------------------------------------------------------
//
// A double-word value represents `hi + lo` where `hi` is the rounded floating
// point sum and `|lo| <= 0.5 * ulp(hi)` holds the rounding residual (~48 bits
// of precision for `float`, ~106 bits for `double`).

// Adds `a` and `b` without magnitude ordering assumptions: returns the rounded
// sum `s = a + b` and sets `out_err = (a + b) - s`.
template <typename T>
__device__ T TwoSum(T a, T b, T& out_err) {
  const T s = AddRn(a, b);
  const T v = SubRn(s, a);
  out_err = AddRn(SubRn(a, SubRn(s, v)), SubRn(b, v));
  return s;
}

// Faster version of `TwoSum` when `|a| >= |b|` or `a == 0`: returns the rounded
// sum `s = a + b` and sets `out_err = (a + b) - s`.
template <typename T>
__device__ T FastTwoSum(T a, T b, T& out_err) {
  const T s = AddRn(a, b);
  out_err = SubRn(b, SubRn(s, a));
  return s;
}

// Multiplies `a` and `b`: returns the rounded product `p = a * b` and sets
// `out_err = a * b - p` in two instructions via hardware FMA.
template <typename T>
__device__ T TwoProd(T a, T b, T& out_err) {
  const T p = MulRn(a, b);
  out_err = FmaRn(a, b, -p);
  return p;
}

// Adds two double-word numbers `(a0 + a1) + (b0 + b1)` without magnitude
// ordering assumptions.
template <typename T>
__device__ T F2Add(T a0, T a1, T b0, T b1, T& out1) {
  T e;
  const T s = TwoSum(a0, b0, e);
  return FastTwoSum(s, AddRn(e, AddRn(a1, b1)), out1);
}

// Adds two double-word numbers `(a0 + a1) + (b0 + b1)` when `|a0| >= |b0|` or
// `a0 == 0`.
template <typename T>
__device__ T F2AddOrdered(T a0, T a1, T b0, T b1, T& out1) {
  T e;
  const T s = FastTwoSum(a0, b0, e);
  return FastTwoSum(s, AddRn(e, AddRn(a1, b1)), out1);
}

// Adds two double-word numbers `(a0 + a1) + (b0 + b1)` and rounds the sum to a
// single word.
template <typename T>
__device__ T F2Sum(T a0, T a1, T b0, T b1) {
  T e;
  const T s = TwoSum(a0, b0, e);
  return AddRn(s, AddRn(e, AddRn(a1, b1)));
}

// Multiplies two double-word numbers `(a0 + a1) * (b0 + b1)`.
template <typename T>
__device__ T F2Mul(T a0, T a1, T b0, T b1, T& out1) {
  T p1;
  const T p0 = TwoProd(a0, b0, p1);
  p1 = FmaRn(a0, b1, FmaRn(a1, b0, p1));
  return FastTwoSum(p0, p1, out1);
}

// Multiplies double-word `(a0 + a1)` by single-word `b`.
template <typename T>
__device__ T F2MulScalar(T a0, T a1, T b, T& out1) {
  T p1;
  const T p0 = TwoProd(a0, b, p1);
  return FastTwoSum(p0, FmaRn(a1, b, p1), out1);
}

// Multiplies two `(hi, lo)` pairs `(a0, a1)` and `(b0, b1)` whose high parts
// `a0` and `b0` are pre-split so `a0 * b0` has zero rounding error.
template <typename T>
__device__ T F2MulExactHighFma(T a0, T a1, T b0, T b1, T& out1) {
  const T p0 = MulRn(a0, b0);
  const T p1 = FmaRn(a1, b1, FmaRn(a0, b1, MulRn(a1, b0)));
  return FastTwoSum(p0, p1, out1);
}

// Squares a double-word number `(a0 + a1)^2 = a0^2 + 2 * a0 * a1 + a1^2`.
template <typename T>
__device__ T F2Square(T a0, T a1, T& out1) {
  T p1;
  const T p0 = TwoProd(a0, a0, p1);
  p1 = FmaRn(AddRn(a0, a0), a1, p1);
  return FastTwoSum(p0, p1, out1);
}

// Computes the double-word reciprocal `1 / (b0 + b1)` from `y0 = 1 / b0` and
// one Newton residual step `r = 1 - y0 * (b0 + b1)`.
template <typename T>
__device__ T F2Recip(T b0, T b1, T& out1) {
  const T y0 = DivRn(T(1), b0);
  const T r = NegMulAddRn(y0, b1, NegMulAddRn(y0, b0, T(1)));
  return FastTwoSum(y0, MulRn(y0, r), out1);
}

// Computes the double-word quotient `(a0 + a1) / (b0 + b1)` from `q0 = a0 / b0`
// and the residual `(a0 + a1) - q0 * (b0 + b1)`.
template <typename T>
__device__ T F2Div(T a0, T a1, T b0, T b1, T& out1) {
  const T q0 = DivRn(a0, b0);
  T p1;
  const T p0 = TwoProd(q0, b0, p1);
  T rem1;
  const T rem0 = F2Add(a0, a1, -p0, NegMulAddRn(q0, b1, -p1), rem1);
  const T q1 = DivRn(AddRn(rem0, rem1), b0);
  return FastTwoSum(q0, q1, out1);
}

// Evaluates `coeffs[0] + coeffs[1] * x + ... + coeffs[N - 1] * x^(N - 1)` from
// highest degree down to `coeffs[0]` using Horner's rule (`acc = acc * x + c`).
template <typename T, size_t N>
__device__ T Horner(T x, const T (&coeffs)[N]) {
  static_assert(N >= 1);
  T acc = coeffs[N - 1];
  for (size_t i = N - 1; i-- > 0;) {
    acc = FmaRn(acc, x, coeffs[i]);
  }
  return acc;
}

// Evaluates a split-precision polynomial matching `horner_hybrid_f2` in
// `jax/_src/lax/double_word.py`: single-word Horner on the small high-degree
// coefficients `hi_deg` (ascending order), followed by `(hi, lo)` Horner on the
// dominant low-degree coefficients `(lo_deg_hi, lo_deg_lo)` (descending order).
template <typename T, size_t NLo, size_t NHi>
__device__ T HornerHybridF2(T w0, T w1, const T (&lo_deg_hi)[NLo],
                            const T (&lo_deg_lo)[NLo], const T (&hi_deg)[NHi],
                            T& out1) {
  const T p = Horner(w0, hi_deg);
  T acc1;
  T acc0 = TwoProd(p, w0, acc1);
  acc0 = F2AddOrdered(lo_deg_hi[0], lo_deg_lo[0], acc0, acc1, acc1);
  for (size_t i = 1; i < NLo; ++i) {
    acc0 = F2Mul(acc0, acc1, w0, w1, acc1);
    acc0 = F2AddOrdered(lo_deg_hi[i], lo_deg_lo[i], acc0, acc1, acc1);
  }
  out1 = acc1;
  return acc0;
}

// ---------------------------------------------------------------------------
// Narrow-float (`bfloat16` / `float16`) and `float32` buffer load/store helpers
// ---------------------------------------------------------------------------

__device__ inline float Bf16BitsToF32(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

// Rounds `float32` `val` to `bfloat16` using IEEE round-to-nearest-even,
// quieting signaling `NaN` payloads rather than overflowing them to `+-inf`.
__device__ inline uint16_t F32ToBf16BitsRn(float val) {
  uint32_t u = __float_as_uint(val);
  if ((u & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<uint16_t>((u >> 16) | 0x0040U);
  }
  u += 0x00007fffU + ((u >> 16) & 1U);
  return static_cast<uint16_t>(u >> 16);
}

__device__ inline float LoadNarrowOrF32(const void* ptr, int64_t idx,
                                        xla::ffi::DataType dtype) {
  switch (dtype) {
    case xla::ffi::DataType::BF16:
      return Bf16BitsToF32(static_cast<const uint16_t*>(ptr)[idx]);
    case xla::ffi::DataType::F16:
      return __half2float(static_cast<const __half*>(ptr)[idx]);
    default:
      return static_cast<const float*>(ptr)[idx];
  }
}

__device__ inline void StoreNarrowOrF32(void* ptr, int64_t idx,
                                        xla::ffi::DataType dtype, float val) {
  switch (dtype) {
    case xla::ffi::DataType::BF16:
      static_cast<uint16_t*>(ptr)[idx] = F32ToBf16BitsRn(val);
      break;
    case xla::ffi::DataType::F16:
      static_cast<__half*>(ptr)[idx] = __float2half_rn(val);
      break;
    default:
      static_cast<float*>(ptr)[idx] = val;
      break;
  }
}

}  // namespace math_internal
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_DOUBLE_WORD_CU_H_
