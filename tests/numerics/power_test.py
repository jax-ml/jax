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

"""Precision tests for power, root, and reciprocal functions."""

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp

# Under pytest, tests run against an installed wheel that does not
# include `jax.tests`, so skip before importing `jax.tests.numerics`.
if jtu.is_running_under_pytest():
  import pytest

  pytest.skip("Only runs under Bazel", allow_module_level=True)

from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]


@jtu.thread_unsafe_test_class()
class SqrtTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sqrt_accuracy(self, dtype):
    bounds = [
        ("gpu", {f32: 1.0}),
        ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 3.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 2.0}),
    ]
    input_ftz = [
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.sqrt,
        np.sqrt,
        mpmath.sqrt,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )


@jtu.thread_unsafe_test_class()
class RsqrtTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_rsqrt_accuracy(self, dtype):
    # The max ULP for float32 is either 1.0 and 2.0 depend on CPU vendor (AMD
    # vs Intel). The (min, max) tuple allows tightness checks to pass on both.
    bounds = [
        ("cpu", {f32: (1.0, 2.0), f64: 1.5}),
        ("gpu", {f32: 2.0, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p"], {f32: 2.5}),
        ("tpu_v6e", {f32: 1.5}),
        ("tpu_7x", {f32: 1.0}),
    ]
    input_ftz = [
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self,
        lax.rsqrt,
        lambda x: np.reciprocal(np.sqrt(x)),
        lambda x: mpmath.nan if x < 0 else 1 / mpmath.sqrt(x),
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )


@jtu.thread_unsafe_test_class()
class CbrtTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cbrt_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 4.5}),
        (["tpu_v6e", "tpu_7x"], {f32: 1.5}),
    ]
    input_ftz = [
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.cbrt,
        np.cbrt,
        lambda x: -mpmath.cbrt(-x) if x < 0 else mpmath.cbrt(x),
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )


@jtu.thread_unsafe_test_class()
class SquareTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_square_accuracy(self, dtype):
    bounds = []
    util.check_unary_precision(
        self, jnp.square, np.square, lambda x: x * x, dtype, bounds=bounds
    )


@jtu.thread_unsafe_test_class()
class ReciprocalTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_reciprocal_accuracy(self, dtype):
    bounds = [
        ("gpu", {f32: 1.0}),
        (TPU_EUPV1, {bf16: 1.0, f32: 198.0}),
        ("tpu_v5p", {bf16: 1.0, f32: 40.0}),
        (["tpu_v6e", "tpu_7x"], {f32: 1.5}),
    ]
    input_ftz = [
        ("cpu", {f16: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.reciprocal,
        np.reciprocal,
        lambda x: 1 / x,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )


@jtu.thread_unsafe_test_class()
class HypotTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hypot_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 1.5, f16: 1.5, f32: 1.5, f64: 1.5}),
        ("gpu", {bf16: 1.5, f16: 1.5, f32: 2.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 3.5}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 3.5}),
        ("tpu_v6e", {f16: 1.0, f32: 2.5}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 2.5}),
    ]
    input_ftz = [
        ("cpu", {f16: False, f64: False}),
        ("gpu", {bf16: False, f16: False, f32: False, f64: False}),
        ("tpu", {f16: False}),
    ]

    # _hypot_ref is evaluated in float64 for bf16/f16/f32 (f64 uses mpmath),
    # where x*x and y*y are exact (<= 48 mantissa bits) and cannot overflow or
    # underflow float64, allowing SIMD sqrt(x*x + y*y) instead of scalar libm.
    def _hypot_ref(x, y):
      res = np.hypot(x, y) if dtype == f64 else np.sqrt(x * x + y * y)
      return np.where(np.isinf(x) | np.isinf(y), np.inf, res)

    def _mpmath_hypot(x, y):
      if np.isinf(float(x)) or np.isinf(float(y)):
        return mpmath.inf
      return mpmath.hypot(x, y)

    util.check_nary_precision(
        self,
        jnp.hypot,
        _hypot_ref,
        _mpmath_hypot,
        dtype,
        nargs=2,
        bounds=bounds,
        input_ftz=input_ftz,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hypot_grad(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    finfo = jnp.finfo(dtype)
    small = float(finfo.tiny) * 4.0
    large = float(finfo.max) / 8.0
    big = float(finfo.max)
    # Test normal, axis-aligned, underflow-prone, and overflow-prone regimes,
    # including the top binades where 1 / r is subnormal and where r overflows
    # to inf even though the gradient is finite.
    cases = [
        (3.0, 4.0),
        (-3.0, 4.0),
        (3.0, -4.0),
        (-3.0, -4.0),
        (5.0, 0.0),
        (-5.0, 0.0),
        (0.0, 5.0),
        (0.0, -5.0),
        (3.0 * small, 4.0 * small),
        (-3.0 * small, 4.0 * small),
        (3.0 * large, 4.0 * large),
        (3.0 * large, -4.0 * large),
        (0.6 * big, 0.8 * big),
        (big, big),
        (-big, big),
        (big, 1.0),
    ]
    xs = jnp.array([c[0] for c in cases], dtype=dtype)
    ys = jnp.array([c[1] for c in cases], dtype=dtype)

    # Reference unit vector (x / r, y / r) in float64, normalized by
    # max(|x|, |y|) first so it does not overflow for float64 inputs.
    x64, y64 = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    m64 = np.maximum(np.abs(x64), np.abs(y64))
    a, b = x64 / m64, y64 / m64
    h = np.hypot(a, b)
    expected_gx, expected_gy = a / h, b / h

    grad_fn = jax.jit(jax.vmap(jax.grad(jnp.hypot, argnums=(0, 1))))
    gx, gy = grad_fn(xs, ys)
    tol = 4 * float(finfo.eps)
    self.assertAllClose(
        np.asarray(gx, np.float64), expected_gx, atol=tol, rtol=tol
    )
    self.assertAllClose(
        np.asarray(gy, np.float64), expected_gy, atol=tol, rtol=tol
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hypot_higher_order_grad(self, dtype):
    # Second- and third-order derivatives must stay accurate in the regimes
    # where _hypot rescales its inputs by 2**(+-k); differentiating through the
    # scale would produce overflowing or underflowing scale**n factors.
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    finfo = jnp.finfo(dtype)
    k = (finfo.maxexp - 1) // 2 + 2
    # (3 * s, 4 * s) with s chosen so max(|x|, |y|) = 4 * s lands exactly on
    # the scaled-up (lo) and scaled-down (hi) thresholds in _hypot.
    s_lo = 2.0 ** ((finfo.minexp + finfo.nmant + 1) // 2 - 2)
    s_hi = 2.0 ** (k - 2)

    hessian_fn = jax.jit(jax.hessian(jnp.hypot, argnums=(0, 1)))
    third_fn = jax.jit(jax.grad(jax.grad(jax.grad(jnp.hypot))))
    tol = 8 * float(finfo.eps)
    for s in (1.0, s_lo, s_hi):
      x = jnp.array(3.0 * s, dtype=dtype)
      y = jnp.array(4.0 * s, dtype=dtype)
      r = 5.0 * s
      # d^2 r / dx_i dx_j = (delta_ij - u_i u_j) / r with u = (0.6, 0.8).
      expected_hessian = np.array([[0.64, -0.48], [-0.48, 0.36]]) / r
      hessian = np.array(
          [[float(h) for h in row] for row in hessian_fn(x, y)]
      )
      self.assertAllClose(hessian, expected_hessian, atol=0, rtol=tol)
      # The third derivative ~ 1 / r**2 is subnormal at s_hi, so skip it there.
      if s != s_hi:
        # d^3 r / dx^3 = -3 x y^2 / r^5 = -3 u_1 u_2^2 / r^2.
        expected_third = -3.0 * 0.6 * 0.64 / (r * r)
        self.assertAllClose(
            float(third_fn(x, y)), expected_third, atol=0, rtol=tol
        )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hypot_grad_at_zero(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    grad_fn = jax.jit(jax.grad(jnp.hypot, argnums=(0, 1)))
    for sx in (0.0, -0.0):
      for sy in (0.0, -0.0):
        x = jnp.array(sx, dtype=dtype)
        y = jnp.array(sy, dtype=dtype)
        gx, gy = grad_fn(x, y)
        self.assertTrue(np.isnan(float(gx)))
        self.assertTrue(np.isnan(float(gy)))
        primal, tangent = jax.jvp(
            jnp.hypot,
            (x, y),
            (jnp.ones_like(x), jnp.ones_like(y)),
        )
        self.assertEqual(float(primal), 0.0)
        self.assertTrue(np.isnan(float(tangent)))

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_hypot_jit_constants(self, dtype):
    # Regression test: when one or both operands of jnp.hypot are compile-time
    # constants inside jit, XLA's algebraic simplifier must not reassociate
    # (x * scale)^2 into (scale * scale) * (x * x), which overflows scale^2 to
    # inf in the lo branch (producing inf * 0 = NaN for x = 0) or overflows/
    # underflows x^2 before scaling.
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    finfo = jnp.finfo(dtype)
    k = (finfo.maxexp - 1) // 2 + 2
    s_lo = 2.0 ** ((finfo.minexp + finfo.nmant + 1) // 2 - 2)
    s_hi = 2.0 ** (k - 2)
    tol = 4 * float(finfo.eps)

    for s in (0.0, 1.0, s_lo, s_hi):
      c1 = jnp.array(3.0 * s, dtype=dtype)
      c2 = jnp.array(4.0 * s, dtype=dtype)
      expected = 5.0 * s

      res_both_const = jax.jit(lambda c1=c1, c2=c2: jnp.hypot(c1, c2))()
      self.assertAllClose(float(res_both_const), expected, atol=0, rtol=tol)

      res_lhs_const = jax.jit(lambda y, c1=c1: jnp.hypot(c1, y))(c2)
      self.assertAllClose(float(res_lhs_const), expected, atol=0, rtol=tol)

      res_rhs_const = jax.jit(lambda x, c2=c2: jnp.hypot(x, c2))(c1)
      self.assertAllClose(float(res_rhs_const), expected, atol=0, rtol=tol)

      res_zero_const = jax.jit(
          lambda x: jnp.hypot(jnp.zeros_like(x), x)
      )(c2)
      self.assertAllClose(
          float(res_zero_const), 4.0 * s, atol=0, rtol=tol
      )


util.register_benchmark(jnp.sqrt)
util.register_benchmark(lax.rsqrt)
util.register_benchmark(jnp.cbrt)
util.register_benchmark(jnp.square)
util.register_benchmark(jnp.reciprocal)
util.register_benchmark(jnp.hypot, nargs=2)


if __name__ == "__main__":
  util.main()
