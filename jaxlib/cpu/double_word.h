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

// Shared Highway SIMD building blocks for CPU math FFI kernels, matching
// `jax/_src/lax/double_word.py` with hardware fused multiply-add (FMA):
//
//   1. Fast small-integer `int64_t <-> double` SIMD conversions for AVX2 and
//      other targets without native 64-bit integer-to-double instructions.
//   2. Double-word `(hi, lo)` arithmetic, where a high-precision value is
//      stored as the sum `hi + lo` of a leading `double` `hi` and a small
//      rounding-error correction `lo` (`|lo| <= 0.5 * ulp(hi)`), providing
//      roughly twice the precision of `double` (~106 bits).
//   3. Polynomial evaluation helpers (`Horner`, `HornerHybridF2`).
//   4. Narrow-float (`bfloat16` / `float16`) packet loop adapter that promotes
//      inputs to `float32`, calls a shared `float32` SIMD function, and rounds
//      back to the narrow type without duplicating the kernel body.

#ifndef JAXLIB_CPU_DOUBLE_WORD_H_
#define JAXLIB_CPU_DOUBLE_WORD_H_

#include <cstddef>
#include <cstdint>

#include "jaxlib/log_coeffs.h"

#endif  // JAXLIB_CPU_DOUBLE_WORD_H_

// Per-target SIMD definitions (re-included once per Highway target when
// `foreach_target.h` toggles `HWY_TARGET_TOGGLE`).
#if defined(JAXLIB_CPU_DOUBLE_WORD_TOGGLE_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef JAXLIB_CPU_DOUBLE_WORD_TOGGLE_
#undef JAXLIB_CPU_DOUBLE_WORD_TOGGLE_
#else
#define JAXLIB_CPU_DOUBLE_WORD_TOGGLE_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace jax {
namespace cpu_math {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
namespace mc = jax::math_coeffs;

// ---------------------------------------------------------------------------
// Fast small-integer `int64_t <-> double` SIMD conversions
// ---------------------------------------------------------------------------

// Converts signed `int64_t` lanes in `[-2^51, 2^51)` to `double`.
// AVX2 lacks a native `int64_t -> double` instruction (`vcvtqq2pd` requires
// AVX-512DQ). Adding `kInt52MagicBits` (`2^52 + 2^51` as raw `double` bits)
// places `v` directly into the low 52 mantissa bits of `double(v) + (2^52 +
// 2^51)`, after which subtracting `2^52 + 2^51` recovers `double(v)`.
template <class DF64, class VI64>
HWY_INLINE hn::VFromD<DF64> SmallInt64ToDouble(DF64 df64, VI64 v) {
  const hn::RebindToSigned<DF64> di64;
  const VI64 biased = hn::Add(v, hn::Set(di64, mc::kInt52MagicBits));
  return hn::Sub(hn::BitCast(df64, biased),
                 hn::Set(df64, mc::kInt52MagicDouble));
}

// Converts integer-valued `double` lanes in `[-2^51, 2^51)` to `int64_t` by
// adding `2^52 + 2^51` so the integer sits in the low 52 bits of the `double`
// bit pattern, then subtracting `kInt52MagicBits` as an `int64_t`.
template <class DI64, class VF64>
HWY_INLINE hn::VFromD<DI64> SmallDoubleIntToInt64(DI64 di64, VF64 v) {
  const hn::RebindToFloat<DI64> df64;
  const VF64 biased = hn::Add(v, hn::Set(df64, mc::kInt52MagicDouble));
  return hn::Sub(hn::BitCast(di64, biased), hn::Set(di64, mc::kInt52MagicBits));
}

// ---------------------------------------------------------------------------
// Double-word `(hi, lo)` arithmetic
// ---------------------------------------------------------------------------

// Adds `a` and `b` without magnitude ordering assumptions: returns the rounded
// sum `s = a + b` and sets `e` to the exact rounding error `(a + b) - s`.
template <class V>
HWY_INLINE V TwoSum(V a, V b, V& e) {
  const V s = hn::Add(a, b);
  const V v = hn::Sub(s, a);
  e = hn::Add(hn::Sub(a, hn::Sub(s, v)), hn::Sub(b, v));
  return s;
}

// Faster version of `TwoSum` when `|a| >= |b|` or `a == 0`: returns the rounded
// sum `s = a + b` and sets `e` to the exact rounding error `(a + b) - s`.
template <class V>
HWY_INLINE V FastTwoSum(V a, V b, V& e) {
  const V s = hn::Add(a, b);
  e = hn::Sub(b, hn::Sub(s, a));
  return s;
}

// Splits a `float64` value `a` (`|a| < 2^996`) into a 26-bit high part `ah` and
// a low part `al = a - ah` so that products of two split parts fit in 53 bits
// without rounding on targets without hardware FMA.
// TODO(phawkins): Remove the !HWY_NATIVE_FMA fallbacks once all supported CPU
// build configurations require FMA/AVX2.
template <class V>
HWY_INLINE V Split26F64(V a, V& al) {
  const hn::DFromV<V> d;
  const V c = hn::Mul(hn::Set(d, 134217729.0), a);
  const V ah = hn::Sub(c, hn::Sub(c, a));
  al = hn::Sub(a, ah);
  return ah;
}

// Splits a `float64` value `a` into a 26-bit high part `ah` and a 27-bit low
// part `al = a - ah` by masking off the bottom 27 mantissa bits. Unlike
// `Split26F64`, this never multiplies `a` by `2^27 + 1`, so it cannot overflow
// on large inputs near `DBL_MAX`.
// TODO(phawkins): Remove the !HWY_NATIVE_FMA fallbacks once all supported CPU
// build configurations require FMA/AVX2.
template <class V>
HWY_INLINE V SplitHalfMaskF64(V a, V& al) {
  const hn::DFromV<V> d;
  const hn::RebindToSigned<decltype(d)> di64;
  const V ah = hn::BitCast(
      d,
      hn::And(hn::BitCast(di64, a), hn::Set(di64, mc::kKeep26MantissaMaskF64)));
  al = hn::Sub(a, ah);
  return ah;
}

// Multiplies `a` and `b`: returns the rounded product `p = a * b` and sets `e`
// to the exact rounding error `a * b - p`. On FMA targets `MulSub(a, b, p)`
// computes `e` in a single instruction; non-FMA targets fall back to 26-bit
// splitting (`|a|, |b| < 2^996`).
template <class V>
HWY_INLINE V TwoProd(V a, V b, V& e) {
  const V p = hn::Mul(a, b);
  if constexpr (HWY_NATIVE_FMA) {
    e = hn::MulSub(a, b, p);
  } else {
    // TODO(phawkins): Remove the !HWY_NATIVE_FMA fallback once all supported
    // CPU build configurations require FMA/AVX2.
    V al, bl;
    const V ah = Split26F64(a, al);
    const V bh = Split26F64(b, bl);
    e = hn::Add(hn::Add(hn::Add(hn::Sub(hn::Mul(ah, bh), p), hn::Mul(ah, bl)),
                        hn::Mul(al, bh)),
                hn::Mul(al, bl));
  }
  return p;
}

// Overflow-safe `TwoProd` for large inputs up to `DBL_MAX`: identical to
// `TwoProd` on FMA targets, and uses bitmask splitting on non-FMA targets so
// inputs above `2^996` (such as `z - 0.5` near `lgamma`'s overflow limit
// `2.56e305`) do not overflow `134217729.0 * a`.
template <class V>
HWY_INLINE V TwoProdBig(V a, V b, V& e) {
  const V p = hn::Mul(a, b);
  if constexpr (HWY_NATIVE_FMA) {
    e = hn::MulSub(a, b, p);
  } else {
    // TODO(phawkins): Remove the !HWY_NATIVE_FMA fallback once all supported
    // CPU build configurations require FMA/AVX2.
    V al, bl;
    const V ah = SplitHalfMaskF64(a, al);
    const V bh = SplitHalfMaskF64(b, bl);
    e = hn::Add(hn::Add(hn::Add(hn::Sub(hn::Mul(ah, bh), p), hn::Mul(ah, bl)),
                        hn::Mul(al, bh)),
                hn::Mul(al, bl));
  }
  return p;
}

// Adds two double-word numbers `(a0 + a1) + (b0 + b1)` without magnitude
// ordering assumptions.
template <class V>
HWY_INLINE V F2Add(V a0, V a1, V b0, V b1, V& out1) {
  V e;
  const V s = TwoSum(a0, b0, e);
  return FastTwoSum(s, hn::Add(e, hn::Add(a1, b1)), out1);
}

// Adds two double-word numbers `(a0 + a1) + (b0 + b1)` when `|a0| >= |b0|` or
// `a0 == 0`.
template <class V>
HWY_INLINE V F2AddOrdered(V a0, V a1, V b0, V b1, V& out1) {
  V e;
  const V s = FastTwoSum(a0, b0, e);
  return FastTwoSum(s, hn::Add(e, hn::Add(a1, b1)), out1);
}

// Adds two double-word numbers `(a0 + a1) + (b0 + b1)` and rounds the final
// result to a single `double`.
template <class V>
HWY_INLINE V F2Sum(V a0, V a1, V b0, V b1) {
  V e;
  const V s = TwoSum(a0, b0, e);
  return hn::Add(s, hn::Add(e, hn::Add(a1, b1)));
}

// Multiplies two double-word numbers `(a0 + a1) * (b0 + b1)`, keeping the
// first-order cross terms `a0 * b1 + a1 * b0`.
template <class V>
HWY_INLINE V F2Mul(V a0, V a1, V b0, V b1, V& out1) {
  V e;
  const V p = TwoProd(a0, b0, e);
  e = hn::MulAdd(a0, b1, e);
  e = hn::MulAdd(a1, b0, e);
  return FastTwoSum(p, e, out1);
}

// Multiplies double-word `(a0 + a1)` by single-word `b0`.
template <class V>
HWY_INLINE V F2MulScalar(V a0, V a1, V b0, V& out1) {
  V e;
  const V p = TwoProd(a0, b0, e);
  e = hn::MulAdd(a1, b0, e);
  return FastTwoSum(p, e, out1);
}

// Multiplies two `(hi, lo)` pairs `(a0, a1)` and `(b0, b1)` whose high parts
// `a0` and `b0` have at most 26 significant mantissa bits each, so `a0 * b0`
// fits in 52 bits with zero rounding error and needs no `TwoProd`.
template <class V>
HWY_INLINE V F2MulExactHighFma(V a0, V a1, V b0, V b1, V& out1) {
  const V ph = hn::Mul(a0, b0);
  const V pl = hn::MulAdd(a0, b1, hn::MulAdd(a1, b0, hn::Mul(a1, b1)));
  return FastTwoSum(ph, pl, out1);
}

// Squares a double-word number `(a0 + a1)^2 = a0^2 + 2 * a0 * a1 + a1^2`.
template <class V>
HWY_INLINE V F2Square(V a0, V a1, V& out1) {
  V e;
  const V p = hn::Mul(a0, a0);
  if constexpr (HWY_NATIVE_FMA) {
    e = hn::MulSub(a0, a0, p);
  } else {
    // TODO(phawkins): Remove the !HWY_NATIVE_FMA fallback once all supported
    // CPU build configurations require FMA/AVX2.
    V al;
    const V ah = Split26F64(a0, al);
    const V ah_al = hn::Mul(ah, al);
    e = hn::Add(hn::Add(hn::Sub(hn::Mul(ah, ah), p), hn::Add(ah_al, ah_al)),
                hn::Mul(al, al));
  }
  e = hn::MulAdd(hn::Add(a0, a0), a1, e);
  return FastTwoSum(p, e, out1);
}

// Computes the double-word reciprocal `1 / (a0 + a1)`. Starts from the hardware
// reciprocal `r = 1 / a0`, computes the residual `1 - (a0 + a1) * r`, and adds
// one Newton correction `r * rem`.
template <class D, class V>
HWY_INLINE V F2Recip(D d, V a0, V a1, V& out1) {
  const V one = hn::Set(d, 1.0);
  const V r = hn::Div(one, a0);
  V rem;
  if constexpr (HWY_NATIVE_FMA) {
    const V rem0 = hn::NegMulAdd(a0, r, one);
    rem = hn::NegMulAdd(a1, r, rem0);
  } else {
    // TODO(phawkins): Remove the !HWY_NATIVE_FMA fallback once all supported
    // CPU build configurations require FMA/AVX2.
    V e0;
    const V p0 = TwoProd(a0, r, e0);
    rem = hn::Sub(hn::Sub(hn::Sub(one, p0), e0), hn::Mul(a1, r));
  }
  return FastTwoSum(r, hn::Mul(r, rem), out1);
}

// Computes the double-word quotient `(a0 + a1) / (b0 + b1)` using one hardware
// reciprocal `r = 1 / b0`, leading quotient `q0 = a0 * r`, and residual
// `(a0 + a1) - q0 * (b0 + b1)` scaled by `r`.
template <class D, class V>
HWY_INLINE V F2Div(D d, V a0, V a1, V b0, V b1, V& out1) {
  const V r = hn::Div(hn::Set(d, 1.0), b0);
  const V q0 = hn::Mul(a0, r);
  V q1;
  if constexpr (HWY_NATIVE_FMA) {
    const V rem0 = hn::NegMulAdd(q0, b0, a0);
    const V rem1 = hn::NegMulAdd(q0, b1, a1);
    q1 = hn::Mul(hn::Add(rem0, rem1), r);
  } else {
    // TODO(phawkins): Remove the !HWY_NATIVE_FMA fallback once all supported
    // CPU build configurations require FMA/AVX2.
    V e0;
    const V p0 = TwoProd(q0, b0, e0);
    q1 = hn::Mul(
        hn::Add(hn::Sub(hn::Sub(a0, p0), e0), hn::Sub(a1, hn::Mul(q0, b1))), r);
  }
  return FastTwoSum(q0, q1, out1);
}

// ---------------------------------------------------------------------------
// Polynomial evaluation helpers
// ---------------------------------------------------------------------------

// Evaluates `coeffs[0] + coeffs[1] * x + ... + coeffs[N - 1] * x^(N - 1)` from
// highest degree down to `coeffs[0]` using Horner's rule (`acc = acc * x + c`).
template <size_t N, class D, class V>
HWY_INLINE V Horner(D d, V x, const double (&coeffs)[N]) {
  static_assert(N >= 1, "Polynomial must have at least one coefficient.");
  V acc = hn::Set(d, coeffs[N - 1]);
  for (size_t i = N - 1; i-- > 0;) {
    acc = hn::MulAdd(acc, x, hn::Set(d, coeffs[i]));
  }
  return acc;
}

// Evaluates a split-precision polynomial matching `horner_hybrid_f2` in
// `jax/_src/lax/double_word.py`:
//   - High-degree coefficients `hi_deg` (stored in ascending power order) are
//     small enough that standard `float64` Horner evaluation on `x0` already
//     exceeds 106-bit accuracy relative to the constant term.
//   - Low-degree coefficients `(lo_deg_hi, lo_deg_lo)` (stored in descending
//     power order, down to degree 0) are accumulated in `(hi, lo)` double-word
//     arithmetic so the dominant terms retain full ~106-bit precision.
template <size_t NLo, size_t NHi, class D, class V>
HWY_INLINE V HornerHybridF2(D d, V x0, V x1, const double (&lo_deg_hi)[NLo],
                            const double (&lo_deg_lo)[NLo],
                            const double (&hi_deg)[NHi], V& out1) {
  const V pt = Horner(d, x0, hi_deg);
  V p1;
  V p0 = TwoProd(pt, x0, p1);
  p0 = F2AddOrdered(hn::Set(d, lo_deg_hi[0]), hn::Set(d, lo_deg_lo[0]), p0, p1,
                    p1);
  for (size_t i = 1; i < NLo; ++i) {
    V m1;
    const V m0 = F2Mul(p0, p1, x0, x1, m1);
    p0 = F2AddOrdered(hn::Set(d, lo_deg_hi[i]), hn::Set(d, lo_deg_lo[i]), m0,
                      m1, p1);
  }
  out1 = p1;
  return p0;
}

// ---------------------------------------------------------------------------
// Narrow-float (`bfloat16` / `float16`) packet loop adapter
// ---------------------------------------------------------------------------

// Evaluates a unary function on `T16` (`hwy::bfloat16_t` or `hwy::float16_t`)
// by promoting each SIMD packet to `float32`, calling `eval_batch` (marked
// `HWY_NOINLINE` so the `float32` kernel body is compiled once per SIMD target
// rather than duplicated three times), and rounding back to `T16`.
// Loads and stores go through `uint16_t` so masked tail `LoadN` / `StoreN`
// work identically for both `bfloat16_t` and `float16_t` on every target.
template <typename T16, class EvalBatchFn>
HWY_INLINE void EvalNarrowViaF32Loop(const T16* in, T16* out, int64_t n,
                                     EvalBatchFn eval_batch) {
  static_assert(sizeof(T16) == sizeof(uint16_t));
  const hn::ScalableTag<double> df64;
  const hn::Rebind<float, decltype(df64)> df32;
  const hn::Rebind<T16, decltype(df32)> d16;
  const hn::Rebind<uint16_t, decltype(df32)> du16;
  const int64_t lanes = static_cast<int64_t>(hn::Lanes(df32));

  const auto* in_u16 = reinterpret_cast<const uint16_t*>(in);
  auto* out_u16 = reinterpret_cast<uint16_t*>(out);
  int64_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    const auto bits_in = hn::LoadU(du16, in_u16 + i);
    const auto x32 = hn::PromoteTo(df32, hn::BitCast(d16, bits_in));
    const auto y16 = hn::DemoteTo(d16, eval_batch(x32));
    hn::StoreU(hn::BitCast(du16, y16), du16, out_u16 + i);
  }
  if (i < n) {
    const size_t rem = static_cast<size_t>(n - i);
    const auto bits_in = hn::LoadN(du16, in_u16 + i, rem);
    const auto x32 = hn::PromoteTo(df32, hn::BitCast(d16, bits_in));
    const auto y16 = hn::DemoteTo(d16, eval_batch(x32));
    hn::StoreN(hn::BitCast(du16, y16), du16, out_u16 + i, rem);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace cpu_math
}  // namespace jax
HWY_AFTER_NAMESPACE();

#endif  // JAXLIB_CPU_DOUBLE_WORD_TOGGLE_
