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

"""Precision tests for hypergeometric functions against reference implementations."""

import itertools
import math

from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
import jax.scipy as jsp

# Under pytest, tests run against an installed wheel that does not
# include `jax.tests`, so skip before importing `jax.tests.numerics`.
if jtu.is_running_under_pytest():
  import pytest

  pytest.skip("Only runs under Bazel", allow_module_level=True)

from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np
import scipy.special

config.parse_flags_with_absl()

f32, f64 = jnp.float32, jnp.float64
# hyp1f1 is not tested in bfloat16 and float16, which are too narrow for its
# series and gamma function ratios.
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [f32, f64]]


def _hyp1f1_ignored(a, b, x):
  """Inputs (a, b, x) excluded from the accuracy checks.

  Only 0.1 <= a, b <= 10 and -700 <= x <= -1e-3 are tested. For x -> -inf,
  1F1(a, b, x) ~ gamma(b) / gamma(b - a) (-x)^-a, so close to the poles of
  gamma(b - a) the result is dominated by the rounding error of b - a, which
  the reference does not have. The poles themselves are kept.
  """
  # b - a == d + d_err exactly (TwoSum).
  d = b - a
  v = d - b
  d_err = (b - (d - v)) - (a + v)
  n = np.round(d)
  pole_distance = (d - n) + d_err
  near_pole = (n <= 0) & (np.abs(pole_distance) < 1e-3) & (pole_distance != 0)
  in_domain = ((a >= 0.1) & (a <= 10) & (b >= 0.1) & (b <= 10)
               & (x >= -700) & (x <= -1e-3))
  return near_pole | ~in_domain


_HYP1F1_INTERESTING_POINTS = list(itertools.product(
    [0.2, 1.0, 2.5, 7.3],
    [0.5, 1.5, 4.0, 9.2],
    # The series is used for x > -100 and the asymptotic expansion below.
    [-1e-3, -5.0, -20.0, -60.0, -99.5, -100.5, -150.0, -400.0, -700.0],
))


@jax.jit
def hyp1f1_grad_a(a, b, x):
  return jax.vmap(jax.grad(jsp.special.hyp1f1, argnums=0))(a, b, x)


@jax.jit
def hyp1f1_grad_b(a, b, x):
  return jax.vmap(jax.grad(jsp.special.hyp1f1, argnums=1))(a, b, x)


@jax.jit
def hyp1f1_grad_x(a, b, x):
  return jax.vmap(jax.grad(jsp.special.hyp1f1, argnums=2))(a, b, x)


def _mpmath_hyp1f1_grad_a(a, b, x):
  return mpmath.diff(lambda t: mpmath.hyp1f1(t, b, x), a)


def _mpmath_hyp1f1_grad_b(a, b, x):
  return mpmath.diff(lambda t: mpmath.hyp1f1(a, t, x), b)


def _mpmath_hyp1f1_grad_x(a, b, x):
  return a / b * mpmath.hyp1f1(a + 1, b + 1, x)


def _hyp1f1_series_reference(a, b, x):
  """Float64 1F1(a, b, x) and its derivatives in a and b, for x <= 0.

  Used as the reference for float32 (float64 tests use mpmath), since scipy
  has no derivatives of hyp1f1 and mpmath is too slow for the float32 sample
  count. Uses Kummer's transform
    1F1(a, b, x) = exp(x) * sum_k (c)_k / (b)_k * z^k / k!,  c = b - a, z = -x,
  whose terms do not alternate in sign once k > -c. The a and b derivatives of
  each term are carried along with the product rule, so integer c <= 0 (where
  the series terminates) needs no special case. In the tested domain this
  agrees with mpmath to within 1e-3 float32 ULP.
  """
  a, b, x = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (a, b, x)))
  f, f_a, f_b = (np.full(a.shape, np.nan) for _ in range(3))
  ok = np.isfinite(a) & (b > 0) & (b < np.inf) & (x <= 0) & (x >= -1000)
  if np.any(ok):
    b_, x_ = b[ok], x[ok]
    c, z = b_ - a[ok], -x_
    u = np.exp(x_)  # exp(x) * (c)_k / (b)_k * z^k / k!
    v = np.zeros_like(u)  # exp(x) * d/dc[(c)_k] / (b)_k * z^k / k!
    h = np.zeros_like(u)  # sum_{j < k} 1 / (b + j)
    s, s_a, s_b = u.copy(), np.zeros_like(u), np.zeros_like(u)
    # The terms peak near k = z and are negligible by k = z + 15 sqrt(z) + 100.
    for k in range(int(z.max() + 15 * np.sqrt(z.max()) + 100)):
      r = z / ((b_ + k) * (k + 1))
      v = (v * (c + k) + u) * r
      h += 1 / (b_ + k)
      u = u * (c + k) * r
      s += u
      s_a -= v
      s_b += v - u * h
    f[ok], f_a[ok], f_b[ok] = s, s_a, s_b
  return f, f_a, f_b


def _hyp1f1_grad_a_reference(a, b, x):
  return _hyp1f1_series_reference(a, b, x)[1]


def _hyp1f1_grad_b_reference(a, b, x):
  return _hyp1f1_series_reference(a, b, x)[2]


def _hyp1f1_grad_x_reference(a, b, x):
  return a / b * _hyp1f1_series_reference(a + 1, b + 1, x)[0]


@jtu.thread_unsafe_test_class()
class Hyp1f1Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  @jax.numpy_dtype_promotion("standard")  # hyp1f1 promotes bool masks to int.
  def test_hyp1f1_accuracy(self, dtype):
    # The errors are largest for b - a < 0, where the first terms of the series
    # of 1F1(b - a, b, -x) alternate in sign.
    bounds = [
        ("cpu", {f32: 800.0, f64: 1100.0}),
        (["gpu", "tpu"], {f32: math.inf, f64: math.inf}),
    ]
    util.check_nary_precision(
        self,
        jsp.special.hyp1f1,
        scipy.special.hyp1f1,
        mpmath.hyp1f1,
        dtype,
        nargs=3,
        bounds=bounds,
        ignore_inputs=_hyp1f1_ignored,
        interesting_points=_HYP1F1_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class Hyp1f1GradTest(jtu.JaxTestCase):

  def _check_grad(self, jax_fn, ref_fn, mpmath_fn, dtype, bounds):
    util.check_nary_precision(
        self,
        jax_fn,
        ref_fn,
        mpmath_fn,
        dtype,
        nargs=3,
        bounds=bounds,
        ignore_inputs=_hyp1f1_ignored,
        interesting_points=_HYP1F1_INTERESTING_POINTS,
        # Almost all full-range random samples fall outside the tested domain
        # and are ignored, so beyond the 32768 samples near the interesting
        # points they only add evaluation time.
        max_samples=2**16 if dtype == f32 else None,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  @jax.numpy_dtype_promotion("standard")
  def test_hyp1f1_grad_a_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 700.0, f64: 650.0}),
        (["gpu", "tpu"], {f32: math.inf, f64: math.inf}),
    ]
    self._check_grad(hyp1f1_grad_a, _hyp1f1_grad_a_reference,
                     _mpmath_hyp1f1_grad_a, dtype, bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  @jax.numpy_dtype_promotion("standard")
  def test_hyp1f1_grad_b_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 1000.0, f64: 1300.0}),
        (["gpu", "tpu"], {f32: math.inf, f64: math.inf}),
    ]
    self._check_grad(hyp1f1_grad_b, _hyp1f1_grad_b_reference,
                     _mpmath_hyp1f1_grad_b, dtype, bounds)

  @parameterized.named_parameters(*DTYPE_PARAMS)
  @jax.numpy_dtype_promotion("standard")
  def test_hyp1f1_grad_x_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 1500.0, f64: 1400.0}),
        (["gpu", "tpu"], {f32: math.inf, f64: math.inf}),
    ]
    self._check_grad(hyp1f1_grad_x, _hyp1f1_grad_x_reference,
                     _mpmath_hyp1f1_grad_x, dtype, bounds)


util.register_benchmark(jsp.special.hyp1f1, nargs=3, dtypes=[f32, f64])


if __name__ == "__main__":
  util.main()
