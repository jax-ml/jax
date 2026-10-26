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


def _in_domain(a, b, x):
  """The inputs that are tested: 0 < a <= 10, 1e-6 <= b <= 10, -1e6 <= x < 100.

  These cover the negative x that hyp1f1 evaluates through Kummer's
  transformation, and the series for positive x. Outside, hyp1f1 is less
  accurate, and only the closed forms and mpmath values in
  lax_scipy_special_functions_test.py are tested:
  - for a < 0 or b close to 0, the terms of the series cancel;
  - for x >= 100, the expansion for x -> +inf diverges for some a and b, e.g.
    hyp1f1(-7.5, 9.2, 100.5) gives 1.6e49 instead of 2.3e19, and similarly for
    x close to -100, e.g. hyp1f1(9.5, -6.5, -100.5).
  The references are only evaluated at these inputs; scipy.special.hyp1f1 can
  take seconds per point for x around -1e10.
  """
  return ((a > 0) & (a <= 10) & (b >= 1e-6) & (b <= 10) & (x >= -1e6)
          & (x < 100))


def _mpmath_hyp1f1(a, b, x):
  try:
    # Tiny a, b or x need a higher working precision than the default maximum.
    return mpmath.hyp1f1(a, b, x, maxprec=30000)
  except mpmath.libmp.NoConvergence:
    return mpmath.nan


def _scipy_hyp1f1(a, b, x):
  """scipy.special.hyp1f1, checked against Kummer's transformation.

  scipy.special.hyp1f1 is wrong at a few points, e.g. it gives 5.9e16 for
  hyp1f1(1e-30, -5.5, 60), which is 6868. Where it disagrees with
  exp(x) * scipy.special.hyp1f1(b - a, b, -x), mpmath is used instead.
  """
  with np.errstate(all="ignore"):
    f = scipy.special.hyp1f1(a, b, x)
    g = np.exp(x) * scipy.special.hyp1f1(b - a, b, -x)
    differ = np.isfinite(f) & np.isfinite(g) & ~np.isclose(f, g, rtol=1e-6)
  f[differ] = [float(_mpmath_hyp1f1(*v))
               for v in zip(a[differ], b[differ], x[differ])]
  return f


def _hyp1f1_ref(a, b, x):
  a, b, x = np.broadcast_arrays(a, b, x)
  out = np.full(a.shape, np.nan)
  m = _in_domain(a, b, x)
  out[m] = _scipy_hyp1f1(a[m], b[m], x[m])
  return out


def _hyp1f1_grad_x_ref(a, b, x):
  # d/dx 1F1(a, b, x) = a / b * 1F1(a + 1, b + 1, x).
  a, b, x = np.broadcast_arrays(a, b, x)
  out = np.full(a.shape, np.nan)
  m = _in_domain(a, b, x)
  out[m] = a[m] / b[m] * _scipy_hyp1f1(a[m] + 1, b[m] + 1, x[m])
  return out


def _mpmath_hyp1f1_in_domain(a, b, x):
  if not _in_domain(a, b, x):
    return mpmath.nan
  return _mpmath_hyp1f1(a, b, x)


def _mpmath_hyp1f1_grad_x(a, b, x):
  if not _in_domain(a, b, x):
    return mpmath.nan
  return a / b * _mpmath_hyp1f1(a + 1, b + 1, x)


def _cancellation(a, b, x):
  """By how much the terms of the series that hyp1f1 sums cancel.

  That is the ratio of the sum of the absolute values of the terms to the
  absolute value of their sum, by which the rounding errors grow. hyp1f1 sums
  the series of 1F1(a, b, x) for 0 <= x < 100, and that of Kummer's
  transformation 1F1(b - a, b, -x) for -100 < x < 0. Returns 1 elsewhere.
  """
  c, z = np.where(x < 0, b - a, a), np.abs(x)
  term = total = total_abs = np.ones_like(z)
  with np.errstate(all="ignore"):
    for k in range(250):
      term = term * (c + k) / (b + k) * z / (k + 1)
      total, total_abs = total + term, total_abs + np.abs(term)
    return np.where(z < 100, total_abs / np.abs(total), 1)


def _two_sum(a, b):
  """a + b rounded to the dtype, and the rounding error (Knuth's TwoSum)."""
  s = a + b
  v = s - a
  return s, (a - (s - v)) + (b - v)


def _hyp1f1_inaccurate(a, b, x, d_err):
  """In-domain inputs where hyp1f1 is not expected to be accurate.

  a, b and x are in the tested dtype, and d_err is the difference between the
  exact b - a and the rounded one that hyp1f1 works with.
  """
  finfo = np.finfo(a.dtype)
  d = (b - a).astype(np.float64)
  a, b, x = (v.astype(np.float64) for v in (a, b, x))
  with np.errstate(all="ignore"):
    f = np.abs(_scipy_hyp1f1(a, b, x))
    # Results within a factor 1 / eps of overflowing or underflowing.
    out = ~(f <= finfo.max * finfo.eps) | (
        (f > 0) & (f < finfo.tiny / finfo.eps))
    # For -100 < x < 0, the terms of the series of Kummer's transformation
    # are scaled by exp(x / 2), and the result by exp(x / 2) once more; they
    # can overflow before the result does.
    out |= f * np.exp(np.maximum(-x, 0) / 2) > finfo.max
  # For x < 0, hyp1f1 works with 1F1(b - a, b, -x). Close to
  # b - a = n = 0, -1, -2, ..., where its series terminates, the part that does
  # not terminate is proportional to b - a - n, so the rounding error of b - a
  # matters when it is large relative to b - a - n.
  n = np.minimum(np.round(d), 0)
  out |= (x < 0) & (np.abs(d_err) > 64 * finfo.eps * np.abs(d + d_err - n))
  out |= _cancellation(a, b, x) > 100
  # For x <= -100, hyp1f1 leaves out the exponentially small part
  # exp(x) gamma(b) / gamma(a) (-x)^(a - b) of the expansion
  # (https://dlmf.nist.gov/13.7.E2 with Kummer's transformation), which is
  # not negligible close to the poles of gamma(b - a).
  with np.errstate(all="ignore"):
    log_dropped = (x + scipy.special.gammaln(b) - scipy.special.gammaln(a)
                   + (a - b) * np.log(-x))
    out |= (x <= -100) & (log_dropped > np.log(finfo.eps * f))
  return out


def _hyp1f1_ignored(a, b, x):
  """Inputs excluded from the accuracy checks."""
  ignored = ~_in_domain(a, b, x)
  # Whether subnormal inputs are flushed to zero depends on the platform.
  for v in (a, b - a):
    ignored |= (v != 0) & (np.abs(v) < np.finfo(a.dtype).tiny)
  m = ~ignored
  _, d_err = _two_sum(b[m], -a[m])
  ignored[m] = _hyp1f1_inaccurate(a[m], b[m], x[m], d_err.astype(np.float64))
  return ignored


def _hyp1f1_grad_x_ignored(a, b, x):
  """The x-derivative is a / b * hyp1f1(a + 1, b + 1, x)."""
  finfo = np.finfo(a.dtype)
  one = a.dtype.type(1)
  ignored = ~_in_domain(a, b, x) | ~_in_domain(a + one, b + one, x)
  for v in (a, b - a):
    ignored |= (v != 0) & (np.abs(v) < finfo.tiny)
  with np.errstate(all="ignore"):
    f = np.abs(_hyp1f1_grad_x_ref(a, b, x))
    # The result within a factor 1 / eps of overflowing or underflowing, and
    # a / b flushing to zero.
    ignored |= ~(f <= finfo.max * finfo.eps) | (
        (f > 0) & (f < finfo.tiny / finfo.eps)) | (
            np.abs(a.astype(np.float64) / b) < finfo.tiny)
  m = ~ignored
  a, b, x = a[m], b[m], x[m]
  # hyp1f1 rounds a + 1 and b + 1, and then their difference, to the dtype.
  a_1, b_1 = a + one, b + one
  d, d_err = _two_sum(b, -a)
  d_err = (d.astype(np.float64) - (b_1 - a_1)) + d_err
  ignored[m] = _hyp1f1_inaccurate(a_1, b_1, x, d_err)
  return ignored


_HYP1F1_INTERESTING_POINTS = list(itertools.product(
    [1e-5, 0.5, 2.5, 9.5],
    [1e-3, 0.5, 4.5, 9.2],
    # hyp1f1 switches from the series to the asymptotic expansion at x = -100.
    [-1e6, -1000.5, -100.5, -99.5, -20.0, -1.0, 1.0, 20.0, 99.5],
))


@jax.jit
def hyp1f1_grad_x(a, b, x):
  return jax.vmap(jax.grad(jsp.special.hyp1f1, argnums=2))(a, b, x)


@jtu.thread_unsafe_test_class()
class Hyp1f1Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hyp1f1_accuracy(self, dtype):
    # The errors are largest close to |x| = 100, where the series have the
    # most terms.
    bounds = [
        ("cpu", {f32: 75.0, f64: 75.0}),
        (["gpu", "tpu"], {f32: math.inf, f64: math.inf}),
    ]
    util.check_nary_precision(
        self,
        jsp.special.hyp1f1,
        _hyp1f1_ref,
        _mpmath_hyp1f1_in_domain,
        dtype,
        nargs=3,
        bounds=bounds,
        ignore_inputs=_hyp1f1_ignored,
        # Results that underflow are +0.0 regardless of their sign, as the
        # exp(x) factor of Kummer's transformation underflows first.
        check_signed_zeros=False,
        interesting_points=_HYP1F1_INTERESTING_POINTS,
    )


# The gradient is tested in a separate class, as the tests are sharded by
# class.
@jtu.thread_unsafe_test_class()
class Hyp1f1GradXTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hyp1f1_grad_x_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 100.0, f64: 75.0}),
        (["gpu", "tpu"], {f32: math.inf, f64: math.inf}),
    ]
    util.check_nary_precision(
        self,
        hyp1f1_grad_x,
        _hyp1f1_grad_x_ref,
        _mpmath_hyp1f1_grad_x,
        dtype,
        nargs=3,
        bounds=bounds,
        ignore_inputs=_hyp1f1_grad_x_ignored,
        check_signed_zeros=False,
        interesting_points=_HYP1F1_INTERESTING_POINTS,
        # The gradient is slower to evaluate than 1F1 itself.
        max_samples=2**18 if dtype == f32 else None,
    )


util.register_benchmark(jsp.special.hyp1f1, nargs=3, dtypes=[f32, f64])


if __name__ == "__main__":
  util.main()
