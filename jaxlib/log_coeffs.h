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

// Shared `constexpr` constants and polynomial coefficients for natural
// logarithm helpers used by the CPU (Highway SIMD) and GPU (CUDA / ROCm)
// `lgamma` and `digamma` kernels.
//
// All constants match the JAX Python reference implementations in
// `jax/_src/lax/double_word.py` and `jax/_src/lax/lgamma.py`.

#ifndef JAXLIB_LOG_COEFFS_H_
#define JAXLIB_LOG_COEFFS_H_

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define JAXLIB_GPU_DEVICE __device__
#else
#define JAXLIB_GPU_DEVICE inline
#endif

namespace jax {
namespace math_coeffs {

// IEEE 754 bit patterns of `sqrt(2) / 2`. Subtracting these integer bits from
// the bit pattern of a positive normal float `x` before shifting right by the
// fraction width chooses the integer exponent `k` so the reduced significand
// `m = x * 2^-k` lands in `[sqrt(2)/2, sqrt(2))` around `1.0`, keeping `m - 1`
// small (`|m - 1| <= 0.4142`).
JAXLIB_GPU_DEVICE constexpr int32_t kSqrt2Over2BitsF32 = 0x3F3504F3;
JAXLIB_GPU_DEVICE constexpr int64_t kSqrt2Over2BitsF64 = 0x3FE6A09E667F3BCDLL;

// Bitmask clearing the low 27 fraction bits of an IEEE `float64` (keeping the
// sign, exponent, and top 26 significand bits). Used by `TwoProdBig` on targets
// without fused multiply-add so splitting large inputs up to `DBL_MAX` cannot
// overflow.
JAXLIB_GPU_DEVICE constexpr int64_t kKeep26MantissaMaskF64 =
    -134217728LL;  // -(1LL << 27)

// Bias `3 * 2^51` (`0x4338000000000000`) for converting between signed 52-bit
// integers `k in [-2^51, 2^51)` and `double` in two vector instructions on SIMD
// targets without native `int64 <-> double` conversion: adding `3 * 2^51`
// places `k` in the low 52 fraction bits of `[2^52, 2^53)`.
JAXLIB_GPU_DEVICE constexpr int64_t kInt52MagicBits = 0x4338000000000000LL;
JAXLIB_GPU_DEVICE constexpr double kInt52MagicDouble = 6755399441055744.0;

// ---------------------------------------------------------------------------
// `float32` `(hi, lo)` `ln(x)` (`_log_f32` in `jax/_src/lax/lgamma.py`)
// ---------------------------------------------------------------------------

// Split of `ln(2) = kLn2HiF32 + kLn2LoF32` where `kLn2HiF32 = 2839 / 4096` has
// 12 significand bits, so `k * kLn2HiF32` has no rounding error in `float32`
// for all exponent offsets `|k| <= 256`.
JAXLIB_GPU_DEVICE constexpr float kLn2HiF32 = 0.693115234375f;
JAXLIB_GPU_DEVICE constexpr float kLn2LoF32 = 3.19461832987e-05f;

// Minimax degree-7 polynomial for `(ln(1 + f) - (f - 0.5 * f^2)) / f^3` on
// `f = m - 1 in [sqrt(2)/2 - 1, sqrt(2) - 1]`, stored in ascending powers of
// `f` (`_LOG_Q` in `jax/_src/lax/lgamma.py`).
JAXLIB_GPU_DEVICE constexpr float kLogQF32[8] = {
    0.3333333134651184f,   -0.250008225440979f,   0.20001226663589478f,
    -0.16623350977897644f, 0.14201749861240387f,  -0.1316024214029312f,
    0.12761665880680084f,  -0.07634450495243073f,
};

// ---------------------------------------------------------------------------
// `float64` `(hi, lo)` `ln(x)` (`_log_f64` in `jax/_src/lax/lgamma.py`)
// ---------------------------------------------------------------------------

// Split of `ln(2) = kLn2HiF64 + kLn2LoF64` with a 42-bit high part so
// `k * kLn2HiF64` has no rounding error in `float64` for `|k| <= 2048`
// (`_LN2_HI_F64` and `_LN2_LO_F64` in `jax/_src/lax/lgamma.py`).
JAXLIB_GPU_DEVICE constexpr double kLn2HiF64 = 0.6931471805598903;
JAXLIB_GPU_DEVICE constexpr double kLn2LoF64 = 5.497923018708371e-14;

// Degree-20 minimax polynomial for `(ln(1 + f) - (f - f^2/2 + f^3/3)) / f^4`
// on `f = m - 1 in [sqrt(2)/2 - 1, sqrt(2) - 1]`, stored in ascending powers
// of `f` (`_LOG_Q_F64` in `jax/_src/lax/lgamma.py`).
// Generated with Sollya:
//   prec = 300!;
//   p = fpminimax(log1p(x),
//       [|4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
//         22, 23, 24|],
//       [|D...|], [sqrt(2)/2 - 1; sqrt(2) - 1], relative, floating,
//       x - 0.5 * x^2 + (1/3) * x^3);
//   for i from 4 to 24 do print(double(coeff(p, i)));
JAXLIB_GPU_DEVICE constexpr double kLogQF64[21] = {
    -0.25000000000000044,  0.20000000000000684,  -0.1666666666665117,
    0.14285714285524623,   -0.1250000000185486,  0.11111111132038475,
    -0.09999999896240677,  0.09090907860216561,  -0.08333336179500314,
    0.07692350617660347,   -0.07142830929109048, 0.06665745309174184,
    -0.062494384445847165, 0.05894350949521322,  -0.0557491085999632,
    0.05176870915457102,   -0.04765086647690675, 0.04986665866679927,
    -0.05835232191926365,  0.05131732430398771,  -0.020844657565116175,
};

// ---------------------------------------------------------------------------
// Double-word `log_f2` (`jax/_src/lax/double_word.py`)
// ---------------------------------------------------------------------------

// `ln(2)` split into three parts `(hi, mid, lo)` (`LN2_SPLIT_F32` and
// `LN2_SPLIT_F64` in `jax/_src/lax/double_word.py`). `hi` and `mid` have at
// most 15 significand bits in `float32` (since `|k| <= 256`, `9 + 15 = 24`
// bits) and 41 significand bits in `float64` (since `|k| <= 2048`, `12 + 41 =
// 53` bits), so `k * hi` and `k * mid` multiply with no rounding error across
// all normal and subnormal exponent offsets.
JAXLIB_GPU_DEVICE constexpr float kLn2SplitF32[3] = {
    0.693145751953125f,
    1.4285906217992306e-06f,
    1.619850954759361e-11f,
};
JAXLIB_GPU_DEVICE constexpr double kLn2SplitF64[3] = {
    0.6931471805598903,
    5.49792301870721e-14,
    1.1612227229362532e-26,
};

// Precomputed `(hi, lo)` pairs for `-ln(r_i)` at the four exact 4-bit scale
// factors `r_i in {1.25, 1.0625, 0.9375, 0.75}` (`_NEG_LNR4_F32` and
// `_NEG_LNR4_F64` in `jax/_src/lax/double_word.py`). Multiplying `m in
// [sqrt(2)/2, sqrt(2))` by `r_i` narrows `m * r_i` into `[0.88388, 1.11488]`
// without rounding error in half-width parts, shrinking
// `s = (m * r_i - 1) / (m * r_i + 1)` to `w = s^2 <= 0.00380` (~8x smaller than
// unreduced `w <= 0.02944`) so `ln(m * r_i) = 2*s + s*w*P(w)` needs far fewer
// `(hi, lo)` terms.
JAXLIB_GPU_DEVICE constexpr float kNegLnr4HiF32[4] = {
    -0.2231435477733612f,
    -0.0606246218085289f,
    0.06453852355480194f,
    0.28768208622932434f,
};
JAXLIB_GPU_DEVICE constexpr float kNegLnr4LoF32[4] = {
    -3.540848503291727e-09f,
    -7.905942393804377e-12f,
    -2.417230859919073e-09f,
    -1.3777543550475002e-08f,
};
JAXLIB_GPU_DEVICE constexpr double kNegLnr4HiF64[4] = {
    -0.22314355131420976,
    -0.06062462181643484,
    0.06453852113757118,
    0.2876820724517809,
};
JAXLIB_GPU_DEVICE constexpr double kNegLnr4LoF64[4] = {
    9.091270597324799e-18,
    -2.6424025938726934e-18,
    -6.470486661692933e-18,
    2.607160616442564e-17,
};

// Polynomial approximations to `(ln((1 + s) / (1 - s)) / s - 2) / w` in
// `w = s^2` on `[0, 0.00380]` (`_LOG_POLY_F32` and `_LOG_POLY_F64` in
// `jax/_src/lax/double_word.py`), fitted by Chebyshev-node interpolation in
// `mpmath` (`2/(2k + 1)` Taylor series `2/3 + 2/5 w + 2/7 w^2 + ...`).
// The larger low-degree terms are stored in descending order as `(hi, lo)`
// pairs, while the small high-degree terms are stored in ascending order as
// single floats (where powers of `w <= 0.00380` suppress single-word rounding
// error below `(hi, lo)` precision):
// - `float32`: 1 `(hi, lo)` term + 4 `float32` terms (~54-bit accuracy).
// - `float64`: 5 `(hi, lo)` terms + 5 `float64` terms (~108-bit accuracy).
JAXLIB_GPU_DEVICE constexpr float kLogPolyLoDegHiF32[1] = {0.6666666865348816f};
JAXLIB_GPU_DEVICE constexpr float kLogPolyLoDegLoF32[1] = {
    -1.9868215517249155e-08f};
JAXLIB_GPU_DEVICE constexpr float kLogPolyHiDegF32[4] = {
    0.4000000059604645f,
    0.2857142984867096f,
    0.22221730649471283f,
    0.1832914650440216f,
};

JAXLIB_GPU_DEVICE constexpr double kLogPolyLoDegHiF64[5] = {
    0.18181818181818155, 0.2222222222222222, 0.2857142857142857, 0.4,
    0.6666666666666666,
};
JAXLIB_GPU_DEVICE constexpr double kLogPolyLoDegLoF64[5] = {
    -1.0849770252736922e-17, 1.2501947479184836e-17, 1.5860279463478475e-17,
    -2.2204460486792272e-17, 3.700743415417178e-17,
};
JAXLIB_GPU_DEVICE constexpr double kLogPolyHiDegF64[5] = {
    0.15384615384643138, 0.13333333316787233, 0.1176471198512322,
    0.10524956497710565, 0.09691143935120411,
};

}  // namespace math_coeffs
}  // namespace jax

#undef JAXLIB_GPU_DEVICE

#endif  // JAXLIB_LOG_COEFFS_H_
