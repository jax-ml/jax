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

"""Precision tests for Bessel functions against reference implementations."""

from absl.testing import parameterized
from jax import lax
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

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]


def _mpmath_i0e(x):
  if mpmath.isinf(x):
    return mpmath.mpf(0.0)
  return mpmath.besseli(0, x) * mpmath.exp(-abs(x))


def _mpmath_i1e(x):
  if mpmath.isinf(x):
    return mpmath.mpf(0.0)
  return mpmath.besseli(1, x) * mpmath.exp(-abs(x))


# Chebyshev polynomial to asymptotic expansion crossover boundary at |x| = 8.0
# in Cephes/XLA's bessel_i0e and bessel_i1e approximations, plus nearby
# transition points.
_BESSEL_INTERESTING_POINTS = [
    *(sign * v for v in (3.75, 7.75, 8.0, 8.25) for sign in (-1, 1)),
]


@jtu.thread_unsafe_test_class()
class BesselI0eTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_bessel_i0e_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 7.0, f64: 7.5}),
        ("gpu", {f32: 8.0, f64: 7.5}),
        ("tpu", {f32: 8.0}),
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
    bounds = [
        ("cpu", {f16: 1.0, f32: 11.0, f64: 10.5}),
        ("gpu", {f16: 1.0, f32: 15.5, f64: 7.5}),
        ("tpu", {f16: 1.0, f32: 15.5}),
    ]
    ref_fn = lambda x: np.copysign(scipy.special.i1e(x), x)
    util.check_unary_precision(
        self,
        lax.bessel_i1e,
        ref_fn,
        _mpmath_i1e,
        dtype,
        bounds=bounds,
        interesting_points=_BESSEL_INTERESTING_POINTS,
    )


util.register_benchmark(lax.bessel_i0e)
util.register_benchmark(lax.bessel_i1e)


def _mpmath_jn(n, x):
  if mpmath.isnan(x):
    return mpmath.nan
  if mpmath.isinf(x):
    return mpmath.mpf(0.0)
  if x == 0:
    return mpmath.mpf(1.0 if n == 0 else 0.0)
  return mpmath.besselj(n, x)


def _scipy_jn_hankel_f64(n, x_large, use_mpmath_near_zero=True):
  """Vectorized float64 Hankel expansion for |x| > 1e7."""
  n_abs = abs(n)
  n_is_odd = (n_abs & 1) == 1
  neg_n = n < 0 and n_is_odd
  w = np.abs(x_large)
  sx = np.sin(w)
  cx = np.cos(w)
  can_c2x = w <= 8e307
  c2x = np.cos(2.0 * np.minimum(w, 8e307))
  u_sum = cx + sx
  v_diff = sx - cx
  u_stab = np.where(
      (cx * sx >= 0.0) | ~can_c2x,
      u_sum,
      -c2x / np.where(cx * sx < 0.0, v_diff, 1.0),
  )
  v_stab = np.where(
      (cx * sx < 0.0) | ~can_c2x,
      v_diff,
      -c2x / np.where(cx * sx >= 0.0, u_sum, 1.0),
  )
  inv_sqrt2 = np.sqrt(0.5)
  c0 = inv_sqrt2 * u_stab
  s0 = inv_sqrt2 * v_stab
  mod4 = n_abs & 3
  if mod4 == 0:
    c_chi, s_chi = c0, s0
  elif mod4 == 1:
    c_chi, s_chi = s0, -c0
  elif mod4 == 2:
    c_chi, s_chi = -c0, -s0
  else:
    c_chi, s_chi = -s0, c0
  mu = float(4 * n_abs * n_abs)
  inv_8w = 1.0 / (8.0 * w)
  t1 = (mu - 1.0) * inv_8w
  t2 = t1 * ((mu - 9.0) / 2.0) * inv_8w
  t3 = t2 * ((mu - 25.0) / 3.0) * inv_8w
  t4 = t3 * ((mu - 49.0) / 4.0) * inv_8w
  p = 1.0 - t2 + t4
  q = t1 - t3
  val = (np.sqrt(2.0 / np.pi) / np.sqrt(w)) * (p * c_chi - q * s_chi)
  if neg_n:
    val = -val
  if n_is_odd:
    val = np.where(x_large < 0.0, -val, val)
  if use_mpmath_near_zero:
    near_zero = np.abs(c_chi) < 1e-4
    if np.any(near_zero):
      val[near_zero] = [
          float(mpmath.besselj(n, float(v))) for v in x_large[near_zero]
      ]
  return val


def _scipy_jn_ref(n, x, use_mpmath_near_zero=True):
  # Reference in float64 for <=32-bit dtypes: guards domain and large |x| > 1e7
  # where Cephes's float64 `x - (n/2 + 1/4)*pi` loses phase precision.
  x_arr = np.asarray(x, dtype=np.float64)
  out = np.full_like(x_arr, np.nan, dtype=np.float64)
  n_abs = abs(n)
  n_is_odd = (n_abs & 1) == 1
  neg_n = n < 0 and n_is_odd
  out[np.isinf(x_arr)] = 0.0
  in_scipy = np.isfinite(x_arr) & (np.abs(x_arr) <= 1e7)
  if np.any(in_scipy):
    out[in_scipy] = scipy.special.jn(n, x_arr[in_scipy])
  large = (np.abs(x_arr) > 1e7) & np.isfinite(x_arr)
  if np.any(large):
    out[large] = _scipy_jn_hankel_f64(
        n, x_arr[large], use_mpmath_near_zero=use_mpmath_near_zero
    )
  if n_is_odd:
    out = np.where(
        (out == 0) & (np.abs(x_arr) < 1.0),
        np.copysign(0.0, -x_arr if neg_n else x_arr),
        out,
    )
  return out


def _near_jn_zero(n, x, rel_env_tol=0.25):
  """Returns True where |x| > 2 is near a zero of J_n(x) (|J_n(x)| < rel_env_tol * envelope)."""
  x_f64 = np.asarray(x, dtype=np.float64)
  osc = np.isfinite(x_f64) & (np.abs(x_f64) > 2.0)
  if not np.any(osc):
    return np.zeros(x_f64.shape, dtype=bool)
  out = np.zeros(x_f64.shape, dtype=bool)
  x_osc = x_f64[osc]
  env = np.sqrt(2.0 / (np.pi * np.abs(x_osc)))
  y_osc = _scipy_jn_ref(n, x_osc, use_mpmath_near_zero=False)
  out[osc] = np.abs(y_osc) < rel_env_tol * env
  return out


_BESSEL_JN_INTERESTING_POINTS = [
    # Approximation domain boundaries (|x| = 2.0, 10.0, 24.0) and nearby points:
    *(
        sign * v
        for v in (1e-4, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 24.0)
        for sign in (-1, 1)
    ),
]

_JN_BF16_F16_BOUNDS = {
    0: (1.0, (1.0, 1.5)),
    1: ((0.5, 1.0), (1.0, 1.5)),
    -1: ((0.5, 1.0), (1.0, 1.5)),
    2: (1.0, 1.0),
    3: ((0.5, 1.0), (0.5, 1.0)),
}


def _check_jn_order(test_case, n, dtype):
  fn = lambda x: jsp.special.jn(n, x)
  fn.__name__ = f"jn_n{n}"
  bf16_b, f16_b = _JN_BF16_F16_BOUNDS[n]
  # Points close to non-zero roots of J_n(x) (where |J_n(x)| < 0.25 * envelope
  # and the condition number |x * J_n'(x) / J_n(x)| is large) are excluded from
  # the relative-ULP check via `ignore_inputs` so tight ULP bounds can be
  # enforced away from roots, and are checked separately against the envelope
  # ULP in `_check_jn_near_roots_envelope`.
  bounds = [
      ("cpu", {bf16: bf16_b, f16: f16_b, f32: 20.0, f64: 20.0}),
      ("gpu", {bf16: bf16_b, f16: f16_b, f32: 25.0, f64: 25.0}),
      ("tpu", {bf16: bf16_b, f16: f16_b, f32: 30.0}),
  ]
  input_ftz = [
      ("gpu", False),
  ]
  ignore_near_zero = lambda x: _near_jn_zero(n, x)
  ignore_inputs = [
      (["cpu", "gpu", "tpu"], {f32: ignore_near_zero, f64: ignore_near_zero}),
  ]
  util.check_unary_precision(
      test_case,
      fn,
      lambda x: _scipy_jn_ref(n, x),
      lambda x: _mpmath_jn(n, x),
      dtype,
      bounds=bounds,
      input_ftz=input_ftz,
      ignore_inputs=ignore_inputs,
      interesting_points=_BESSEL_JN_INTERESTING_POINTS,
  )


def _check_jn_near_roots_envelope(test_case, n, dtype):
  """Checks that near roots of J_n(x), error is <= 8 ULPs of the envelope."""
  if dtype == f64 and not config.enable_x64.value:
    test_case.skipTest("float64 requires jax_enable_x64")
  np_dtype = np.dtype(dtype)
  eps = float(np.finfo(np_dtype).eps)

  # First 20 positive zeros of J_{|n|}(x) spanning Miller, recurrence, and
  # asymptotic regimes, plus asymptotic zeros at larger arguments.
  zeros = list(scipy.special.jn_zeros(abs(n), 20))
  for k in (50, 200, 1000):
    zeros.append((k + 0.5 * abs(n) - 0.25) * np.pi)
  rel_offsets = [
      0.0,
      -0.5 * eps,
      0.5 * eps,
      -2.0 * eps,
      2.0 * eps,
      -10.0 * eps,
      10.0 * eps,
      -1e-5,
      1e-5,
      -1e-3,
      1e-3,
      -1e-2,
      1e-2,
  ]
  pts = [-8.649218755035513, 8.649218755035513]
  for z0 in zeros:
    for d in rel_offsets:
      pts.append(z0 * (1.0 + d))
      pts.append(-z0 * (1.0 + d))

  x = np.unique(np.asarray(pts, dtype=np_dtype))
  y = np.asarray(jsp.special.jn(n, jnp.asarray(x, dtype=dtype)), dtype=np.float64)
  y_ref = np.array(
      [float(_mpmath_jn(n, mpmath.mpf(float(v)))) for v in x],
      dtype=np.float64,
  )
  env = np.sqrt(2.0 / (np.pi * np.abs(x.astype(np.float64)))).astype(np_dtype)
  env_ulp = np.abs(np.spacing(env)).astype(np.float64)
  env_ulp_err = np.abs(y - y_ref) / env_ulp
  max_env_ulp = float(np.max(env_ulp_err))
  test_case.assertLessEqual(
      max_env_ulp,
      8.0,
      msg=(
          f"Max envelope ULP error near roots for jn(n={n}, {np_dtype.name})"
          f" was {max_env_ulp:.4f} > 8.0"
      ),
  )


@jtu.thread_unsafe_test_class()
class BesselJ0J1Test(jtu.JaxTestCase):

  @parameterized.named_parameters(
      *(
          (f"_n{n}_{d.__name__}", n, d)
          for n in (0, 1, -1)
          for d in (bf16, f16, f32, f64)
      )
  )
  def test_jn_accuracy(self, n, dtype):
    _check_jn_order(self, n, dtype)

  @parameterized.named_parameters(
      *((f"_n{n}_{d.__name__}", n, d) for n in (0, 1, -1) for d in (f32, f64))
  )
  def test_jn_near_roots_envelope_accuracy(self, n, dtype):
    _check_jn_near_roots_envelope(self, n, dtype)

  @parameterized.named_parameters(
      *(
          (f"_{d.__name__}", d)
          for d in (f32, f64)
      )
  )
  def test_jn_vs_bessel_jn(self, dtype):
    """Demonstrates accuracy and domain improvements of jn over bessel_jn."""
    if dtype == f64 and not config.enable_x64.value:
      self.skipTest("float64 requires jax_enable_x64")
    np_dtype = np.dtype(dtype)

    # 1. Small arguments (|x| <= 0.05): bessel_jn's backward recurrence from
    # k=50 overflows and returns NaN, whereas jn uses the ascending series and
    # is accurate to within ~1.5 ULPs.
    x_small = np.array([1e-6, 1e-4, 1e-3, 1e-2, 0.05], dtype=np_dtype)
    y_old_small = np.asarray(jsp.special.bessel_jn(jnp.asarray(x_small), v=3))
    self.assertTrue(np.any(np.isnan(y_old_small)))
    for n in range(4):
      y_new = np.asarray(
          jsp.special.jn(n, jnp.asarray(x_small)), dtype=np.float64
      )
      y_ref = np.array(
          [float(_mpmath_jn(n, mpmath.mpf(float(v)))) for v in x_small],
          dtype=np_dtype,
      )
      ulp_err = np.abs(y_new - y_ref.astype(np.float64)) / np.abs(
          np.spacing(y_ref)
      )
      self.assertLessEqual(float(np.max(ulp_err)), 2.0)

    # 2. Moderate arguments (6.0 <= x <= 35.0) away from roots: both are finite,
    # and jn achieves much smaller max ULP error than bessel_jn.
    x_mod = np.linspace(6.0, 35.0, 200, dtype=np_dtype)
    y_old_mod = np.asarray(
        jsp.special.bessel_jn(jnp.asarray(x_mod), v=3), dtype=np.float64
    )
    for n in range(4):
      mask = ~_near_jn_zero(n, x_mod)
      y_ref = np.array(
          [float(_mpmath_jn(n, mpmath.mpf(float(v)))) for v in x_mod[mask]],
          dtype=np_dtype,
      )
      ulp = np.abs(np.spacing(y_ref)).astype(np.float64)
      y_new = np.asarray(
          jsp.special.jn(n, jnp.asarray(x_mod[mask])), dtype=np.float64
      )
      ulp_new = float(np.max(np.abs(y_new - y_ref.astype(np.float64)) / ulp))
      ulp_old = float(
          np.max(np.abs(y_old_mod[n, mask] - y_ref.astype(np.float64)) / ulp)
      )
      self.assertLessEqual(ulp_new, 25.0)
      self.assertLess(ulp_new, ulp_old)

    # 3. Large arguments (|x| >= 55.0): bessel_jn with default n_iter=50
    # completely breaks down for |x| > n_iter, whereas jn uses Hankel's
    # asymptotic expansion and remains accurate to a few ULPs.
    x_large = np.array([55.0, 75.0, 100.0, 250.0, 1000.0], dtype=np_dtype)
    y_old_large = np.asarray(
        jsp.special.bessel_jn(jnp.asarray(x_large), v=3), dtype=np.float64
    )
    for n in range(4):
      y_ref = np.array(
          [float(_mpmath_jn(n, mpmath.mpf(float(v)))) for v in x_large],
          dtype=np_dtype,
      )
      ulp = np.abs(np.spacing(y_ref)).astype(np.float64)
      y_new = np.asarray(
          jsp.special.jn(n, jnp.asarray(x_large)), dtype=np.float64
      )
      ulp_new = float(np.max(np.abs(y_new - y_ref.astype(np.float64)) / ulp))
      abs_err_old = float(np.max(np.abs(y_old_large[n] - y_ref)))
      self.assertLessEqual(ulp_new, 5.0)
      self.assertGreater(abs_err_old, 0.1)


@jtu.thread_unsafe_test_class()
class BesselJnHigherTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      *(
          (f"_n{n}_{d.__name__}", n, d)
          for n in (2, 3)
          for d in (bf16, f16, f32, f64)
      )
  )
  def test_jn_accuracy(self, n, dtype):
    _check_jn_order(self, n, dtype)

  @parameterized.named_parameters(
      *((f"_n{n}_{d.__name__}", n, d) for n in (2, 3) for d in (f32, f64))
  )
  def test_jn_near_roots_envelope_accuracy(self, n, dtype):
    _check_jn_near_roots_envelope(self, n, dtype)


util.register_benchmark(lambda x: jsp.special.jn(2, x), name="jn_n2")


if __name__ == "__main__":
  util.main()
