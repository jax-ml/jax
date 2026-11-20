# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precision tests for modified Bessel functions against reference implementations."""

import functools
import math

from absl.testing import parameterized
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np
import scipy.special

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
_IV_ORDERS = (0, 1, 2, 10, 50)
_KN_ORDERS = (0, 1, 2, 10, 50)
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]


def _mpmath_i0e(x):
  if mpmath.isinf(x):
    return mpmath.mpf(0.0)
  return mpmath.besseli(0, x) * mpmath.exp(-abs(x))


def _mpmath_i1e(x):
  if mpmath.isinf(x):
    return mpmath.mpf(0.0)
  return mpmath.besseli(1, x) * mpmath.exp(-abs(x))


def _i1e_reference(x):
  # For tiny |x|, Cephes scipy.special.i1e evaluates a Chebyshev expansion in
  # 0.5 * |x| - 2 whose value at 0 is 0.5 + 2^-53 instead of 0.5.
  ax = np.abs(x)
  small = 0.5 * x * (1.0 - ax + 0.625 * x * x)
  return np.where(ax < 1e-6, small, np.copysign(scipy.special.i1e(x), x))


# Region crossover boundaries at |x| in {0.5, 2.625, 8.0} in bessel_i0e and
# bessel_i1e, plus centers {1.5, 5.25} and nearby transition points.
_BESSEL_INTERESTING_POINTS = [
    *(
        sign * v
        for v in (
            0.375, 0.5, 0.625, 1.5, 2.5, 2.625, 2.75, 5.25, 7.75, 8.0, 8.25,
        )
        for sign in (-1, 1)
    ),
]


@jtu.thread_unsafe_test_class()
class BesselI0eTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_i0e_accuracy(self, dtype):
    """Tests ULP accuracy of lax.bessel_i0e across the floating-point domain."""
    bounds = [
        ("cpu", {bf16: 1.0, f16: 1.0, f32: 2.0, f64: 2.0}),
        ("gpu", {f16: 1.0, f32: 2.5, f64: 2.0}),
        ([*TPU_EUPV1, "tpu_v5p"], {bf16: 1.0, f16: 1.0, f32: 3.0}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 2.0}),
    ]
    util.check_unary_precision(
        self,
        lax.bessel_i0e,
        scipy.special.i0e,
        _mpmath_i0e,
        dtype,
        bounds=bounds,
        interesting_points=_BESSEL_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class BesselI1eTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_i1e_accuracy(self, dtype):
    """Tests ULP accuracy of lax.bessel_i1e across the floating-point domain."""
    bounds = [
        ("cpu", {f16: 1.0, f32: 2.0, f64: 2.0}),
        ("gpu", {f16: 1.0, f32: 2.5, f64: 2.0}),
        ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 3.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 2.0}),
    ]
    util.check_unary_precision(
        self,
        lax.bessel_i1e,
        _i1e_reference,
        _mpmath_i1e,
        dtype,
        bounds=bounds,
        interesting_points=_BESSEL_INTERESTING_POINTS,
    )

  def test_i1e_reference_accuracy(self):
    pos = [2.0**-133, 3 * 2.0**-133, 255 * 2.0**-133, 9.99e-7, 1e-6, 1.01e-6]
    x = np.array([*pos, *(-v for v in pos)], dtype=np.float64)
    with mpmath.workprec(200):
      ref = [_mpmath_i1e(mpmath.mpf(float(v))) for v in x]
      ulp = np.abs(util.ulp_diff(_i1e_reference(x), ref, f64, ftz=False))
    np.testing.assert_array_less(ulp[np.abs(x) < 1e-6], 1.0)
    np.testing.assert_array_less(ulp[np.abs(x) >= 1e-6], 10.0)


def _make_scipy_iv(n: int):
  is_odd = (n & 1) == 1
  fact = float(math.factorial(n))

  def _scipy_iv(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    ax = np.abs(x64)
    is_inf = np.isinf(x64)
    ax_finite = np.where(is_inf, 1.0, ax)
    if n == 0:
      res = scipy.special.i0(ax_finite)
    elif n == 1:
      res = scipy.special.i1(ax_finite)
    else:
      # For tiny |x| < 1e-8, (x/2)^2 / (n + 1) < 2.5e-17 < eps_f64 / 4, so the
      # leading series term (|x|/2)^n / n! is accurate to 0.5 ULP in float64
      # while avoiding scipy.special.iv's premature underflow when K_{n+1}(|x|)
      # overflows.
      small = ax_finite < 1e-8
      under_one = ax_finite < 1.0
      ax_safe = np.where(small, 1.0, ax_finite)
      res = np.where(
          small,
          ((0.5 * ax_finite) ** n) / fact,
          np.where(
              under_one,
              scipy.special.iv(n, ax_safe.astype(np.complex128)).real,
              scipy.special.iv(n, ax_safe),
          ),
      )
    res = np.where(is_inf, np.inf, res)
    if is_odd:
      res = np.copysign(res, x64)
    return res

  return _scipy_iv


def _make_mpmath_iv(n: int):
  is_odd = (n & 1) == 1
  overflow_x = max(1000.0, 2.0 * n)

  def _mpmath_iv(x):
    if mpmath.isnan(x):
      return mpmath.nan
    if x == 0.0:
      return mpmath.mpf(1.0 if n == 0 else 0.0)
    if mpmath.isinf(x) or abs(x) > overflow_x:
      return -mpmath.inf if (x < 0 and is_odd) else mpmath.inf
    return mpmath.besseli(n, x)

  return _mpmath_iv


def _iv_interesting_points(n: int) -> list[float]:
  r_ser = max(12.0, math.sqrt(2.0 * (n + 1)))
  r_asy_f32 = max(r_ser, 10.0 + 0.5 * n * n)
  r_asy_f64 = max(r_ser, 24.0 + 0.5 * n * n)
  r_ser_next_f32 = float(np.nextafter(np.float32(r_ser), np.float32(np.inf)))
  r_ser_next_f64 = float(np.nextafter(np.float64(r_ser), np.float64(np.inf)))
  r_asy_prev_f32 = float(
      np.nextafter(np.float32(r_asy_f32), np.float32(-np.inf))
  )
  r_asy_prev_f64 = float(
      np.nextafter(np.float64(r_asy_f64), np.float64(-np.inf))
  )
  return [
      *_BESSEL_INTERESTING_POINTS,
      *(
          sign * x
          for x in (
              r_ser,
              r_ser_next_f32,
              r_ser_next_f64,
              r_asy_prev_f32,
              r_asy_f32,
              r_asy_prev_f64,
              r_asy_f64,
              91.9,  # near float32 overflow threshold
              713.98,  # near float64 overflow threshold
          )
          for sign in (-1, 1)
      ),
  ]


_IV_BOUNDS = {
    0: [
        ("cpu", {f32: 2.5, f64: 2.0}),
        ("gpu", {f32: 3.0, f64: 2.0}),
        ([*TPU_EUPV1, "tpu_v5p"], {f32: 3.5}),
        (["tpu_v6e", "tpu_7x"], {f32: 2.5}),
    ],
    1: [
        ("cpu", {f16: 1.0, f32: 3.0, f64: 2.5}),
        ("gpu", {bf16: 1.0, f16: 1.0, f32: 3.0, f64: 2.0}),
        ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 3.5}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 3.0}),
    ],
    2: [(["cpu", "gpu", "tpu"], {bf16: 1.0, f16: 1.0, f32: 1.0, f64: 1.0})],
    10: [(["cpu", "gpu", "tpu"], {f32: 1.0, f64: 1.0})],
    50: [(["cpu", "gpu", "tpu"], {f32: 1.0, f64: 1.0})],
}


def _check_iv_accuracy(test_case, v: int, dtype):
  jax_fn = functools.partial(jsp.special.iv, v)
  jax_fn.__name__ = f"bessel_iv_{v}"
  util.check_unary_precision(
      test_case,
      jax_fn,
      _make_scipy_iv(v),
      _make_mpmath_iv(v),
      dtype,
      bounds=_IV_BOUNDS[v],
      interesting_points=_iv_interesting_points(v),
  )


# Separate TestCase classes per order so ClassShardedTestLoader assigns each
# order to its own shard.
@jtu.thread_unsafe_test_class()
class BesselIv0Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_iv_0_accuracy(self, dtype):
    """Tests ULP accuracy of jax.scipy.special.iv(0, x) across real inputs."""
    _check_iv_accuracy(self, 0, dtype)


@jtu.thread_unsafe_test_class()
class BesselIv1Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_iv_1_accuracy(self, dtype):
    """Tests ULP accuracy of jax.scipy.special.iv(1, x) across real inputs."""
    _check_iv_accuracy(self, 1, dtype)


@jtu.thread_unsafe_test_class()
class BesselIv2Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_iv_2_accuracy(self, dtype):
    """Tests ULP accuracy of jax.scipy.special.iv(2, x) across real inputs."""
    _check_iv_accuracy(self, 2, dtype)


@jtu.thread_unsafe_test_class()
class BesselIv10Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_iv_10_accuracy(self, dtype):
    """Tests ULP accuracy of jax.scipy.special.iv(10, x) across real inputs."""
    _check_iv_accuracy(self, 10, dtype)


@jtu.thread_unsafe_test_class()
class BesselIv50Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_iv_50_accuracy(self, dtype):
    """Tests ULP accuracy of jax.scipy.special.iv(50, x) across real inputs."""
    _check_iv_accuracy(self, 50, dtype)


def _make_scipy_kn(n: int):

  def _scipy_kn(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    valid = (x64 > 0.0) & np.isfinite(x64)
    x_pos = np.where(valid, x64, 1.0)
    if n == 0:
      res = scipy.special.k0(x_pos)
    elif n == 1:
      res = scipy.special.k1(x_pos)
    else:
      res = scipy.special.kv(n, x_pos)
    res = np.where(x64 == np.inf, 0.0, res)
    res = np.where(x64 == 0.0, np.inf, res)
    res = np.where((x64 < 0.0) | np.isnan(x64), np.nan, res)
    return res

  return _scipy_kn


def _make_mpmath_kn(n: int):
  underflow_x = max(1100.0, 2.0 * n)

  def _mpmath_kn(x):
    if mpmath.isnan(x) or x < 0.0:
      return mpmath.nan
    if x == 0.0:
      return mpmath.inf
    if mpmath.isinf(x) or x > underflow_x:
      return mpmath.mpf(0.0)
    if n >= 1 and x < 1e-310:
      return mpmath.inf
    return mpmath.besselk(n, x)

  return _mpmath_kn


def _kn_interesting_points(n: int) -> list[float]:
  r_ser = 2.0
  rl_f32 = 10.0
  rl_f64 = 24.0
  r_ser_next_f32 = float(np.nextafter(np.float32(r_ser), np.float32(np.inf)))
  r_ser_next_f64 = float(np.nextafter(np.float64(r_ser), np.float64(np.inf)))
  rl_prev_f32 = float(np.nextafter(np.float32(rl_f32), np.float32(-np.inf)))
  rl_prev_f64 = float(np.nextafter(np.float64(rl_f64), np.float64(-np.inf)))
  return [
      *_BESSEL_INTERESTING_POINTS,
      1.9999999,
      2.0,
      2.0000001,
      r_ser,
      r_ser_next_f32,
      r_ser_next_f64,
      rl_prev_f32,
      rl_f32,
      rl_prev_f64,
      rl_f64,
      85.0,
      103.5,  # near float32 underflow threshold
      705.0,
      744.0,  # near float64 underflow threshold
      float(2 * n + 1),
  ]


_KN_BOUNDS = {
    0: [(["cpu", "gpu", "tpu"], {f16: 1.0, f32: 1.0, f64: 1.0})],
    1: [(["cpu", "gpu", "tpu"], {f32: 1.0, f64: 1.0})],
    2: [(["cpu", "gpu", "tpu"], {f16: 1.0, f32: 1.0, f64: 1.0})],
    10: [(["cpu", "gpu", "tpu"], {f32: 1.0, f64: 1.0})],
    50: [(["cpu", "gpu", "tpu"], {f32: 1.0, f64: 1.0})],
}


def _check_kn_accuracy(test_case, n: int, dtype):
  jax_fn = functools.partial(jsp.special.kn, n)
  jax_fn.__name__ = f"bessel_kn_{n}"
  util.check_unary_precision(
      test_case,
      jax_fn,
      _make_scipy_kn(n),
      _make_mpmath_kn(n),
      dtype,
      bounds=_KN_BOUNDS[n],
      interesting_points=_kn_interesting_points(n),
  )


@jtu.thread_unsafe_test_class()
class BesselKn0Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_kn_0_accuracy(self, dtype):
    """Tests ULP accuracy of jax.scipy.special.kn(0, x) across real inputs."""
    _check_kn_accuracy(self, 0, dtype)


@jtu.thread_unsafe_test_class()
class BesselKn1Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_kn_1_accuracy(self, dtype):
    """Tests ULP accuracy of jax.scipy.special.kn(1, x) across real inputs."""
    _check_kn_accuracy(self, 1, dtype)


@jtu.thread_unsafe_test_class()
class BesselKn2Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_kn_2_accuracy(self, dtype):
    """Tests ULP accuracy of jax.scipy.special.kn(2, x) across real inputs."""
    _check_kn_accuracy(self, 2, dtype)


@jtu.thread_unsafe_test_class()
class BesselKn10Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_kn_10_accuracy(self, dtype):
    """Tests ULP accuracy of jax.scipy.special.kn(10, x) across real inputs."""
    _check_kn_accuracy(self, 10, dtype)


@jtu.thread_unsafe_test_class()
class BesselKn50Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_kn_50_accuracy(self, dtype):
    """Tests ULP accuracy of jax.scipy.special.kn(50, x) across real inputs."""
    _check_kn_accuracy(self, 50, dtype)


util.register_benchmark(lax.bessel_i0e)
util.register_benchmark(lax.bessel_i1e)
for _v in _IV_ORDERS:
  util.register_benchmark(
      functools.partial(jsp.special.iv, _v), name=f"bessel_iv_{_v}"
  )
for _n in _KN_ORDERS:
  util.register_benchmark(
      functools.partial(jsp.special.kn, _n), name=f"bessel_kn_{_n}"
  )


if __name__ == "__main__":
  util.main()
