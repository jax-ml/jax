(numerical-accuracy)=
# Numerical accuracy

<!--* freshness: { reviewed: '2026-09-18' } *-->

This document summarizes the accuracy of JAX mathematical functions.

## Methodology and terminology

- **Units in the Last Place (ULPs)**: Error is measured in ULPs relative to a
  higher-precision reference: for sub-64-bit types (`bfloat16`, `float16`, and
  `float32`), reference values are evaluated in double precision (`float64`); for
  `float64`, reference values are evaluated using {mod}`mpmath` with 100-bit
  precision. A bound of $0.5$ ULP is a correctly rounded result.
- **Exhaustive vs. Sampled Testing**:
  - For 16-bit types (`bfloat16`, `float16`) and 32-bit types (`float32`),
    bounds are verified by exhaustive testing across all bit patterns.
  - For `float64`, bounds are estimated by random sampling across the
    floating-point domain. Because sampling cannot guarantee hitting the worst-case
    input, values in the `float64` table are empirical lower bounds on the true
    maximum error (denoted with $\ge$).
- **Flush-To-Zero (FTZ)**: Subnormal floating-point inputs and outputs are
  typically flushed to zero by default, though some operations on types smaller
  than `float32` do not flush.
- **Hardware Platforms**:
  - **CPU**: x86_64.
  - **NVIDIA GPU**: H100 and B200.
  - **TPU**: TPU v2–v5e, TPU v5p, TPU v6e, and TPU 7x.

---

## `bfloat16` accuracy

Maximum error in units in the last place (ULPs) for `bfloat16`:

| Function | CPU | NVIDIA GPU | TPU v2–v5e | TPU v5p | TPU v6e | TPU 7x |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| {func}`~jax.lax.acos` | 1.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.acosh` | 2.0 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.asin` | 128.0 | 0.5 | 128.0 | 128.0 | 128.0 | 128.0 |
| {func}`~jax.lax.asinh` | 1.5 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.atan` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.atanh` | 1.5 | 0.5 | 1.0 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.bessel_i0e` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.bessel_i1e` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.cbrt` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.cos` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.cosh` | 0.5 | 0.5 | 0.5 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.erf` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.erf_inv` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.erfc` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.exp` | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.exp` (`highest`) | 0.5 | 1.0 | 1.0 | 1.0 | 0.5 | 0.5 |
| {func}`~jax.lax.exp2` | 101.0 | 101.0 | 44.0 | 44.0 | 44.0 | 75.0 |
| {func}`~jax.lax.expm1` | 0.5 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.log` | 0.5 | 0.5 | 1.0 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.numpy.log10` | 2.0 | 2.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.log1p` | 0.5 | 0.5 | 1.0 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.log2` | 2.0 | 2.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.logistic` | 2.5 | 2.5 | 1.0 | 1.0 | 0.5 | 63.0 |
| {func}`~jax.lax.reciprocal` | 0.5 | 0.5 | 1.0 | 1.0 | 0.5 | 0.5 |
| {func}`~jax.lax.rsqrt` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.sin` | 0.5 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.numpy.sinc` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.sinh` | 0.5 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.sqrt` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.square` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.tan` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.tanh` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |

---

## `float16` accuracy

Maximum error in units in the last place (ULPs) for `float16`:

| Function | CPU | NVIDIA GPU | TPU v2–v5e | TPU v5p | TPU v6e | TPU 7x |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| {func}`~jax.lax.acos` | 1.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.acosh` | 2.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.asin` | 1.5 | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.asinh` | 1.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.atan` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.atanh` | 1.5 | 0.5 | 1.0 | 1.0 | 0.5 | 0.5 |
| {func}`~jax.lax.bessel_i0e` | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.bessel_i1e` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.cbrt` | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 0.5 |
| {func}`~jax.lax.cos` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.cosh` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.erf` | 1.0 | 1.0 | 0.5 | 0.5 | 1.0 | 1.0 |
| {func}`~jax.lax.erf_inv` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.erfc` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.exp` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.exp` (`highest`) | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 0.5 |
| {func}`~jax.lax.exp2` | 14.0 | 14.0 | 7.5 | 7.5 | 7.5 | 7.5 |
| {func}`~jax.lax.expm1` | 2.5 | 1.0 | 1.0 | 1.0 | 0.5 | 0.5 |
| {func}`~jax.lax.log` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.numpy.log10` | 1.5 | 1.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.log1p` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.log2` | 2.0 | 2.0 | 1.5 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.logistic` | 2.0 | 2.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.reciprocal` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.rsqrt` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.sin` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.numpy.sinc` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.sinh` | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.sqrt` | 0.5 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.square` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| {func}`~jax.lax.tan` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| {func}`~jax.lax.tanh` | 0.5 | 0.5 | 1.0 | 1.0 | 0.5 | 0.5 |

---

## `float32` accuracy

Maximum error in units in the last place (ULPs) for `float32`:

| Function | CPU | NVIDIA GPU | TPU v2–v5e | TPU v5p | TPU v6e | TPU 7x | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| {func}`~jax.lax.acos` | 1.5 | 1.5 | 5.0 | 5.0 | 4.0 | 5.0 | |
| {func}`~jax.lax.acosh` | 4.5 | 2.5 | 4031.0 | 1003.0 | 984.0 | 984.0 | |
| {func}`~jax.lax.asin` | 8388608.0 | 1.5 | 8388607.0 | 8388607.0 | 8388607.0 | 8388607.0 | |
| {func}`~jax.lax.asinh` | 3.5 | 2.0 | 4034.5 | 2082.5 | 2049.0 | 2049.0 | |
| {func}`~jax.lax.atan` | 4.0–5.5 | 1.5 | 2.5 | 2.5 | 2.5 | 2.5 | CPU bound depends on AMD vs Intel |
| {func}`~jax.lax.atanh` | 3.0 | 3.5 | 2183.5 | 1061.5 | 1025.5 | 1025.5 | |
| {func}`~jax.lax.bessel_i0e` | 7.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | |
| {func}`~jax.lax.bessel_i1e` | 11.0 | 15.5 | 15.5 | 15.5 | 15.5 | 15.5 | |
| {func}`~jax.lax.cbrt` | 0.5 | 1.5 | 4.5 | 4.5 | 1.5 | 1.5 | |
| {func}`~jax.lax.cos` | 0.5 | 2.0 | 3.5 | 3.5 | 3.5 | 3.0 | |
| {func}`~jax.lax.cosh` | 25.0 | 2.5 | 93.5 | 99.0 | 59.0 | 59.5 | |
| {func}`~jax.lax.erf` | 7.0 | 6.5 | 7.5 | 8.5 | 1.5 | 1.5 | |
| {func}`~jax.lax.erf_inv` | 65.0 | 65.0 | 427.0 | 65.5 | 65.0 | 65.5 | |
| {func}`~jax.lax.erfc` | 66.0 | 66.5 | 145.0 | 157.0 | 124.5 | 125.0 | |
| {func}`~jax.lax.exp` | 1.5 | 2.0 | 116.0 | 109.5 | 64.5 | 65.0 | |
| {func}`~jax.lax.exp` (`highest`) | 1.5 | 2.0 | 1.5 | 1.5 | 1.5 | 1.5 | `accuracy=lax.AccuracyMode.HIGHEST` |
| {func}`~jax.lax.exp2` | 68.5 | 69.0 | 141.5 | 133.0 | 90.0 | 90.0 | |
| {func}`~jax.lax.expm1` | 6.5 | 1.5 | 1772.0 | 1357.5 | 64.0 | 63.5 | |
| {func}`~jax.lax.log` | 1.5 | 1.0 | 4030.5 | 62.0 | 2.5 | 2.5 | |
| {func}`~jax.numpy.log10` | 3.0 | 2.5 | 6213.0 | 57.0 | 3.0 | 3.0 | |
| {func}`~jax.lax.log1p` | 3.0 | 1.0 | 4034.0 | 2082.5 | 2049.0 | 2049.0 | |
| {func}`~jax.lax.log2` | 2.5 | 2.0 | 5159.0 | 57.5 | 2.5 | 2.5 | |
| {func}`~jax.lax.logistic` | 2.5 | 4.0 | 243.0 | 124.0 | 65.5 | 64.0 | |
| {func}`~jax.lax.reciprocal` | 0.5 | 1.0 | 198.0 | 40.0 | 1.5 | 1.5 | |
| {func}`~jax.lax.rsqrt` | 1.0–2.0 | 2.0 | 2.5 | 2.5 | 1.5 | 1.0 | CPU bound depends on AMD vs Intel |
| {func}`~jax.lax.sin` | 0.5 | 1.5 | 3.5 | 3.5 | 3.5 | 3.5 | |
| {func}`~jax.numpy.sinc` | 2.5 | 3.5 | 4.0 | 4.0 | 4.0 | 3.5 | |
| {func}`~jax.lax.sinh` | 25.0 | 3.0 | 1794.0 | 1332.5 | 59.0 | 59.5 | |
| {func}`~jax.lax.sqrt` | 0.5 | 1.0 | 3.0 | 3.0 | 2.0 | 2.0 | |
| {func}`~jax.lax.square` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | |
| {func}`~jax.lax.tan` | 0.5 | 3.5 | 6.5 | 7.0 | 5.5 | 5.5 | |
| {func}`~jax.lax.tanh` | 5.0 | 5.5 | 1365.5 | 92.0 | 1.5 | 1.5 | |

---

## `float64` accuracy

Maximum error in units in the last place (ULPs) for `float64`.

```{note}
TPUs have no true float64 hardware. While bounds for smaller types are
exhaustive, for float64 these are lower bounds.
```

| Function | CPU | NVIDIA GPU |
|:---|:---:|:---:|
| {func}`~jax.lax.acos` | $\ge 1.0$ | $\ge 1.5$ |
| {func}`~jax.lax.acosh` | $\ge 3.5$ | $\ge 2.5$ |
| {func}`~jax.lax.asin` | $\ge 4503599627370496.0$ | $\ge 2.5$ |
| {func}`~jax.lax.asinh` | $\ge 2.0$ | $\ge 2.5$ |
| {func}`~jax.lax.atan` | $\ge 3.5$ | $\ge 2.5$ |
| {func}`~jax.lax.atanh` | $\ge 2.5$ | $\ge 3.5$ |
| {func}`~jax.lax.bessel_i0e` | $\ge 7.5$ | $\ge 7.5$ |
| {func}`~jax.lax.bessel_i1e` | $\ge 10.5$ | $\ge 6.0$ |
| {func}`~jax.lax.cbrt` | $\ge 0.5$ | $\ge 1.5$ |
| {func}`~jax.lax.cos` | $\ge 0.5$ | $\ge 1.5$ |
| {func}`~jax.lax.cosh` | $\ge 496.0$ | $\ge 2.5$ |
| {func}`~jax.lax.erf` | $\ge 2.5$ | $\ge 2.5$ |
| {func}`~jax.lax.erf_inv` | $\ge 82.5$ | $\ge 83.5$ |
| {func}`~jax.lax.erfc` | $\ge 350.0$ | $\ge 350.0$ |
| {func}`~jax.lax.exp` | $\ge 1.0$ | $\ge 1.5$ |
| {func}`~jax.lax.exp2` | $\ge 719.0$ | $\ge 719.0$ |
| {func}`~jax.lax.expm1` | $\ge 4.5$ | $\ge 1.5$ |
| {func}`~jax.lax.log` | $\ge 0.5$ | $\ge 1.5$ |
| {func}`~jax.numpy.log10` | $\ge 2.0$ | $\ge 2.5$ |
| {func}`~jax.lax.log1p` | $\ge 2.0$ | $\ge 1.5$ |
| {func}`~jax.lax.log2` | $\ge 1.5$ | $\ge 1.5$ |
| {func}`~jax.lax.logistic` | $\ge 3.5$ | $\ge 4.5$ |
| {func}`~jax.lax.reciprocal` | $\ge 0.5$ | $\ge 0.5$ |
| {func}`~jax.lax.rsqrt` | $\ge 1.5$ | $\ge 1.5$ |
| {func}`~jax.lax.sin` | $\ge 0.5$ | $\ge 2.5$ |
| {func}`~jax.numpy.sinc` | $\ge 2.0$ | $\ge 2.0$ |
| {func}`~jax.lax.sinh` | $\ge 496.0$ | $\ge 2.5$ |
| {func}`~jax.lax.sqrt` | $\ge 0.5$ | $\ge 0.5$ |
| {func}`~jax.lax.square` | $\ge 0.5$ | $\ge 0.5$ |
| {func}`~jax.lax.tan` | $\ge 0.5$ | $\ge 2.5$ |
| {func}`~jax.lax.tanh` | $\ge 6.5$ | $\ge 3.5$ |
