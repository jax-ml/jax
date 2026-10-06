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

"""Precision tests for error functions against reference implementations."""

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.scipy import special as jsp_special_impl
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
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]


def _mpmath_erfc(x):
  """Evaluates erfc with magnitude guards to prevent mpmath hanging on |x| > 100."""
  if x > 100.0:
    return mpmath.mpf(0)
  if x < -100.0:
    return mpmath.mpf(2)
  return mpmath.erfc(x)


def _mpmath_erfcx(x):
  if x < -30.0:
    return mpmath.inf
  if x < 1.0:
    return mpmath.exp(x * x) * mpmath.erfc(x)
  return mpmath.hyperu(0.5, 0.5, x * x) / mpmath.sqrt(mpmath.pi)


def erfcx_grad(x):
  return jax.vmap(jax.grad(jsp.special.erfcx))(x)


def _mpmath_erfcx_grad(x):
  if x < -30.0:
    return -mpmath.inf
  if x < 1.0:
    return 2 * x * _mpmath_erfcx(x) - 2 / mpmath.sqrt(mpmath.pi)
  return -mpmath.hyperu(1.0, 0.5, x * x) / mpmath.sqrt(mpmath.pi)


def _erfcx_grad_reference(x: np.ndarray) -> np.ndarray:
  # Used as the float64 reference for <=32-bit types (float64 tests use mpmath).
  #
  # 1. For x in [-27, 512): erfcx'(x) = 2*x*erfcx(x) - 2/sqrt(pi). In float64
  #    (53 bits), this subtraction cancels by at most 2 * 512^2 = 2^19 (19 bits),
  #    leaving >= 34 bits of precision (10 guard bits beyond float32's 24 bits).
  #
  # 2. For x >= 512: substituting u = t^2 - x^2 into
  #    erfcx(x) = (2/sqrt(pi)) * exp(x^2) * \int_x^\infty exp(-t^2) dt gives
  #      erfcx(x) = (1/sqrt(pi)) * \int_0^\infty exp(-u) / sqrt(x^2 + u) du.
  #    Differentiating with respect to x gives
  #      erfcx'(x) = -(1 / (sqrt(pi) * x^2)) * \int_0^\infty exp(-u) * (1 + u/x^2)^(-3/2) du.
  #    By Taylor's theorem on f(s) = (1 + s)^(-3/2) for s >= 0,
  #      1 - (3/2)*s <= (1 + s)^(-3/2) <= 1 - (3/2)*s + (15/8)*s^2.
  #    Integrating against exp(-u) on [0, \infty) using \int_0^\infty u^k exp(-u) du = k!
  #    bounds the exact derivative for all x > 0 between:
  #      1 - 1.5 / x^2 <= -sqrt(pi) * x^2 * erfcx'(x) <= 1 - 1.5 / x^2 + 3.75 / x^4.
  #    For x >= 512 = 2^9, the truncation error 3.75 / x^4 <= 15 / 2^38 < 2^-34.
  out = np.full_like(x, np.nan, dtype=np.float64)
  out[x < -27.0] = -np.inf
  out[x >= 1e75] = -0.0
  direct = (x >= -27.0) & (x < 512.0)
  if np.any(direct):
    xd = x[direct]
    out[direct] = 2.0 * xd * scipy.special.erfcx(xd) - 2.0 / np.sqrt(np.pi)
  asymp = (x >= 512.0) & (x < 1e75)
  if np.any(asymp):
    inv_x2 = 1.0 / np.square(x[asymp])
    out[asymp] = -(1.0 - 1.5 * inv_x2) * inv_x2 / np.sqrt(np.pi)
  return out


def _erfinv_reference(x: np.ndarray) -> np.ndarray:
  """Evaluates erfinv with a domain guard to avoid slow C++ exception handling."""
  return np.where(
      np.abs(x) <= 1.0,
      np.copysign(scipy.special.erfinv(np.clip(x, -1.0, 1.0)), x),
      np.nan,
  )


@jtu.thread_unsafe_test_class()
class ErfTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erf_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 7.0, f64: 2.5}),
        ("gpu", {bf16: 0.5, f16: (0.5, 1.0), f32: 6.5, f64: 2.5}),
        (TPU_EUPV1, {f16: 0.5, f32: 7.5}),
        ("tpu_v5p", {f16: 0.5, f32: 8.5}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 1.5}),
    ]
    input_ftz = [
        ("gpu", False),
    ]
    # Thresholds where |erf(x)| rounds to 1.0 in float32 (1 - 2^-24) and
    # float64 (1 - 2^-53):
    f32_sat = float(scipy.special.erfinv(1.0 - 2.0**-24))
    f64_sat = float(scipy.special.erfinv(1.0 - 2.0**-53))
    interesting_points = [f32_sat, -f32_sat, f64_sat, -f64_sat]
    util.check_unary_precision(
        self,
        lax.erf,
        scipy.special.erf,
        mpmath.erf,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class ErfcTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfc_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 4.0, f64: 2.5}),
        ("gpu", {f16: 1.0, f32: 5.0, f64: 2.5}),
        (TPU_EUPV1, {f16: 1.0, f32: 5.5}),
        (["tpu_v5p", "tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 5.0}),
    ]
    interesting_points = [
        # Negative thresholds where erfc(x) rounds to 2.0:
        -float(scipy.special.erfinv(1.0 - 2.0**-25)),
        -float(scipy.special.erfinv(1.0 - 2.0**-54)),
        # Positive underflow thresholds where erfc(x) reaches normal tiny /
        # min_subnormal, computed via mpmath.findroot(lambda x: mpmath.erfc(x) - target, x0):
        9.194682,  # erfc(x) == 2^-126 (float32 tiny)
        10.054602,  # erfc(x) == 2^-149 (float32 min_subnormal)
        26.54325777920791,  # erfc(x) == 2^-1022 (float64 tiny)
        27.226017025551105,  # erfc(x) == 2^-1074 (float64 min_subnormal)
    ]
    util.check_unary_precision(
        self,
        lax.erfc,
        scipy.special.erfc,
        _mpmath_erfc,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )

  @parameterized.named_parameters(
      {"testcase_name": "_float32", "dtype": f32},
      {"testcase_name": "_float64", "dtype": f64},
  )
  def test_erfc_derivatives(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    grad1_fn = jax.jit(jax.vmap(jax.grad(lax.erfc)))
    grad2_fn = jax.jit(jax.vmap(jax.grad(jax.grad(lax.erfc))))
    max_x = 9.0 if dtype == f32 else 25.0
    x = jnp.linspace(-max_x, max_x, 256, dtype=dtype)
    with jax.debug_nans(True):
      d1 = np.asarray(grad1_fn(x))
      d2 = np.asarray(grad2_fn(x))
      _ = grad1_fn(jnp.array([-jnp.inf, 0.0, jnp.inf], dtype=dtype))
      _ = grad2_fn(jnp.array([-jnp.inf, 0.0, jnp.inf], dtype=dtype))

    def mp_d1(v):
      return -2 * mpmath.exp(-v * v) / mpmath.sqrt(mpmath.pi)

    def mp_d2(v):
      return 4 * v * mpmath.exp(-v * v) / mpmath.sqrt(mpmath.pi)

    x_np = np.asarray(x)
    ref_d1 = np.array(
        [util.eval_mpmath(mp_d1, v.item(), dtype=dtype) for v in x_np],
        dtype=object if dtype == f64 else np.float64,
    )
    ref_d2 = np.array(
        [util.eval_mpmath(mp_d2, v.item(), dtype=dtype) for v in x_np],
        dtype=object if dtype == f64 else np.float64,
    )
    self.assertLessEqual(np.max(util.ulp_diff(d1, ref_d1, dtype)), 3.0)
    self.assertLessEqual(np.max(util.ulp_diff(d2, ref_d2, dtype)), 4.0)


_ERFCX_INTERESTING_POINTS = [
    # Negative overflow thresholds where erfcx(x) or |erfcx'(x)| == finfo.max,
    # computed via mpmath.findroot:
    -9.382414,  # float32 erfcx overflow threshold
    -9.225755,  # float32 erfcx_grad overflow threshold
    -26.62873571375149,  # float64 erfcx overflow threshold
    -26.50644156603363,  # float64 erfcx_grad overflow threshold
    # Piecewise polynomial and clamp thresholds in erfc.py:
    10.0,
    12.0,
    28.0,
    # Underflow thresholds of unscaled erfc(x):
    9.194682,  # erfc(x) == 2^-126 (float32 tiny)
    26.54325777920791,  # erfc(x) == 2^-1022 (float64 tiny)
    512.0,  # direct-to-asymptotic cutoff (2^9) in _erfcx_grad_reference
]


@jtu.thread_unsafe_test_class()
class ErfcxTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfcx_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 3.5, f64: 3.0}),
        ("gpu", {f16: 1.0, f32: 4.0, f64: 3.0}),
        ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 4.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 4.5}),
    ]
    util.check_unary_precision(
        self,
        jsp.special.erfcx,
        scipy.special.erfcx,
        _mpmath_erfcx,
        dtype,
        bounds=bounds,
        interesting_points=_ERFCX_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class ErfcxGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfcx_grad_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 4.5, f64: 2.5}),
        ("gpu", {f16: 1.0, f32: 5.0, f64: 2.5}),
        (TPU_EUPV1, {f16: 1.0, f32: 6.5}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 6.0}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 5.0}),
        ("tpu_7x", {f16: 1.0, f32: 5.0}),
    ]
    util.check_unary_precision(
        self,
        erfcx_grad,
        _erfcx_grad_reference,
        _mpmath_erfcx_grad,
        dtype,
        bounds=bounds,
        interesting_points=_ERFCX_INTERESTING_POINTS,
    )

  @parameterized.named_parameters(
      {"testcase_name": "_float32", "dtype": f32},
      {"testcase_name": "_float64", "dtype": f64},
  )
  def test_erfcx_grad_higher_order(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    grad2_fn = jax.jit(jax.vmap(jax.grad(jax.grad(jsp.special.erfcx))))
    grad3_fn = jax.jit(
        jax.vmap(jax.grad(jax.grad(jax.grad(jsp.special.erfcx))))
    )
    x = jnp.concatenate([
        jnp.linspace(-3.0, 0.0, 32, dtype=dtype),
        jnp.linspace(0.0, 1.0, 32, dtype=dtype),
        jnp.linspace(1.0, 2.0, 32, dtype=dtype),
        jnp.linspace(2.0, 4.0, 32, dtype=dtype),
        jnp.linspace(4.0, 12.0, 32, dtype=dtype),
        jnp.linspace(12.0, 50.0, 32, dtype=dtype),
    ])
    with jax.debug_nans(True):
      d2 = np.asarray(grad2_fn(x))
      d3 = np.asarray(grad3_fn(x))
      _ = grad2_fn(jnp.array([-jnp.inf, 0.0, 1.0, jnp.inf], dtype=dtype))

    def mp_d2(v):
      y0 = _mpmath_erfcx(v)
      y1 = _mpmath_erfcx_grad(v)
      return 2 * y0 + 2 * v * y1

    def mp_d3(v):
      y1 = _mpmath_erfcx_grad(v)
      return 4 * y1 + 2 * v * mp_d2(v)

    x_np = np.asarray(x)
    ref_d2 = np.array(
        [util.eval_mpmath(mp_d2, v.item(), dtype=dtype) for v in x_np],
        dtype=object if dtype == f64 else np.float64,
    )
    ref_d3 = np.array(
        [util.eval_mpmath(mp_d3, v.item(), dtype=dtype) for v in x_np],
        dtype=object if dtype == f64 else np.float64,
    )
    self.assertLessEqual(np.max(util.ulp_diff(d2, ref_d2, dtype)), 35.0)
    self.assertLessEqual(np.max(util.ulp_diff(d3, ref_d3, dtype)), 1500.0)

  @parameterized.named_parameters(
      {"testcase_name": "_float32", "dtype": f32},
      {"testcase_name": "_float64", "dtype": f64},
  )
  @jtu.run_on_devices("cpu")
  def test_erfcx_grad2_method_choice(self, dtype):
    # `_erfcx_grad_jvp` computes erfcx'' for x >= 0 in one of two ways:
    #   formula:  2 * erfcx(x) + 2 * x * erfcx'(x), used for x <= 2.25.
    #   plain AD: AD through the polynomials in `erfcx_grad_impl`, used for
    #             x > 2.25.
    # Check that on every range, our result is at least as accurate as the
    # better of the two, i.e. that 2.25 is a sensible switch point.
    erfcx = jsp.special.erfcx
    d1 = jax.grad(erfcx)
    ours_fn = jax.jit(jax.vmap(jax.grad(d1)))
    formula_fn = jax.jit(jax.vmap(lambda v: 2 * erfcx(v) + 2 * v * d1(v)))
    # Stopping the gradient of `ans` is only valid for x >= 0, where
    # `erfcx_grad_impl` does not use `ans`.
    plain_ad_fn = jax.jit(jax.vmap(jax.grad(
        lambda v: jsp_special_impl._erfcx_grad_impl(
            v, lax.stop_gradient(erfcx(v))))))
    edges = [0.0, 1.0, 2.0, 2.25, 3.0, 4.0, 8.0, 12.0, 50.0, 1e4]
    x = np.concatenate([
        np.linspace(lo, hi, 256, endpoint=False, dtype=dtype)
        for lo, hi in zip(edges[:-1], edges[1:])
    ])
    # Exactly at the polynomial piece boundaries, the `clamp` and `max` calls
    # in `erfcx_grad_impl` drop part of the gradient, so plain AD is badly
    # wrong there. We leave those points out so the comparison measures the
    # error between boundaries.
    x = x[~np.isin(x, np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0], dtype))]

    def mp_d2(v):
      return 2 * _mpmath_erfcx(v) + 2 * v * _mpmath_erfcx_grad(v)

    with mpmath.workdps(60):
      ref = np.array(
          [util.eval_mpmath(mp_d2, v.item(), dtype=dtype) for v in x],
          dtype=object if dtype == f64 else np.float64,
      )
    ours = util.ulp_diff(np.asarray(ours_fn(x)), ref, dtype)
    formula = util.ulp_diff(np.asarray(formula_fn(x)), ref, dtype)
    plain_ad = util.ulp_diff(np.asarray(plain_ad_fn(x)), ref, dtype)
    for lo, hi in zip(edges[:-1], edges[1:]):
      m = (x >= lo) & (x < hi)
      best = min(np.max(formula[m]), np.max(plain_ad[m]))
      self.assertLessEqual(
          np.max(ours[m]), best,
          msg=f"[{lo}, {hi}): ours={np.max(ours[m])}, "
          f"formula={np.max(formula[m])}, plain_ad={np.max(plain_ad[m])}")

  @jtu.run_on_devices("cpu")
  def test_erfcx_grad_reference(self):
    # Verify _erfcx_grad_reference against mpmath across key regimes:
    # - [-26.5, -9.0]: steep negative growth across the f32 overflow boundary
    # - [-2.0, 2.0]: near zero and across the x = 1.0 mpmath branch point
    # - [5.5, 10.0]: f32 cancellation region and x = 8.0 / 9.3 thresholds
    # - [500.0, 525.0]: around the x = 512.0 direct-to-asymptotic cutoff
    # - [525.0, 1e38]: asymptotic regime up to float32 max
    x = np.concatenate([
        np.linspace(-26.5, -9.0, 32, dtype=np.float64),
        np.linspace(-2.0, 2.0, 33, dtype=np.float64),
        np.linspace(5.5, 10.0, 32, dtype=np.float64),
        np.linspace(500.0, 525.0, 32, dtype=np.float64),
        np.geomspace(525.0, 1e38, 32, dtype=np.float64),
    ])
    ref = _erfcx_grad_reference(x)
    mp_ref = np.array(
        [
            float(util.eval_mpmath(_mpmath_erfcx_grad, v.item()))
            for v in x
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(ref, mp_ref, rtol=1e-9, atol=0.0)

    # Check extreme / boundary regimes (-inf overflow, -0.0 underflow, NaN).
    neg_ovf = _erfcx_grad_reference(np.array([-27.5, -100.0, -np.inf]))
    self.assertTrue(np.all(neg_ovf == -np.inf))
    pos_udf = _erfcx_grad_reference(np.array([1e75, 1e100, np.inf]))
    self.assertTrue(np.all(pos_udf == 0.0))
    self.assertTrue(np.all(np.signbit(pos_udf)))
    self.assertTrue(np.isnan(_erfcx_grad_reference(np.array([np.nan]))[0]))


def _mpmath_erfinv(x):
  if not (-1 <= x <= 1):
    return mpmath.nan
  return mpmath.erfinv(x)


@jtu.thread_unsafe_test_class()
class ErfinvTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_erfinv_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 65.0, f64: 500000.0}),
        ("gpu", {f16: 1.0, f32: 65.0, f64: 500000.0}),
        (TPU_EUPV1, {f16: 1.0, f32: 427.0}),
        (["tpu_v5p", "tpu_7x"], {f16: 1.0, f32: 65.5}),
        ("tpu_v6e", {f16: 1.0, f32: 65.0}),
    ]
    interesting_points = [
        # Minimax polynomial piece boundary points in erf_inv implementations:
        *(
            sign * v
            for v in (0.7, 0.85, 0.9, 0.99, 0.9999)
            for sign in (-1, 1)
        ),
    ]
    util.check_unary_precision(
        self,
        lax.erf_inv,
        _erfinv_reference,
        _mpmath_erfinv,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


util.register_benchmark(lax.erf)
util.register_benchmark(lax.erfc)
util.register_benchmark(jsp.special.erfcx)
util.register_benchmark(erfcx_grad)
util.register_benchmark(lax.erf_inv)


if __name__ == "__main__":
  util.main()
