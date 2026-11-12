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

"""Precision tests for exponential and logistic functions."""

import math

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import xla_version
import jax.numpy as jnp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]


@jax.jit
def _exp_highest(x):
  return lax.exp(x, accuracy=lax.AccuracyMode.HIGHEST)


@jax.jit
def _exp2_highest(x):
  return lax.exp2(x, accuracy=lax.AccuracyMode.HIGHEST)


@jax.jit
def _expm1_highest(x):
  return lax.expm1(x, accuracy=lax.AccuracyMode.HIGHEST)


@jax.jit
def _logistic_highest(x):
  return lax.logistic(x, accuracy=lax.AccuracyMode.HIGHEST)


# Additional range-reduction boundaries (k * ln(2)) beyond +-0.5*ln(2) and
# +-ln(2) (which are already in _common_interesting_points), plus worst-case
# inputs across math libraries from Gladman et al., "Accuracy of Mathematical
# Functions in Single, Double, Double Extended, and Quadruple Precision".
_EXP_INTERESTING_POINTS = [
    *(k * math.log(2.0) for k in (-10, -2, 2, 10)),
    *(
        float.fromhex(h)
        for h in (
            "-0x1.49f33ad2c1c58p+9",
            "0x1.fce66609f7428p+5",
            "0x1.b97dc8345c55p+5",
            "0x1.2e8f20cf3cbe7p+8",
            "-0x1.18209ecd19a8cp+6",
            "-0x1.4133f4fd79c1cp-13",
            "0x1p-53",
            "-0x1.ffffffa000001p-27",
            "-0x1.625f1b359729ep+9",
            "-0x1.625f1c27780c8p+9",
        )
    ),
]


def _expm1_interesting_points(dtype):
  # Thresholds where expm1(x) saturates to -1.0 for `dtype`, plus worst-case
  # inputs across math libraries from Gladman et al.
  p = jnp.finfo(dtype).nmant + 1
  return [
      -p * math.log(2.0),
      -(p + 1) * math.log(2.0),
      *(
          float.fromhex(h)
          for h in (
              "0x1.63e063d87938p-2",
              "-0x1.62fe464c64f65p-8",
              "0x1.9a5796c0b73b8p-2",
              "0x1.62ff47a01658fp-2",
              "0x1.e7f93188565ecp-5",
              "0x1p-52",
              "-0x1.62d7c116d2e32p-1",
              "0x1.633f993a730c9p-2",
              "0x1.a0e95d59498e9p-2",
              "0x1.6339ea76a70cap-2",
          )
      ),
  ]


def _logistic_interesting_points(dtype):
  # Saturation to 1.0 thresholds for `dtype` (underflow to 0.0 at
  # -log(fmax) / -log(tiny) is already in _common_interesting_points).
  p = jnp.finfo(dtype).nmant + 1
  return [
      *(sign * k * math.log(2.0) for k in (p, p + 1) for sign in (-1, 1)),
  ]


@jtu.thread_unsafe_test_class()
class ExpTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.5, f64: 2.0}),
        ("gpu", {bf16: 1.0, f16: 1.0, f32: 2.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 116.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 109.5}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 64.5}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 65.0}),
    ]
    util.check_unary_precision(
        self,
        jnp.exp,
        np.exp,
        mpmath.exp,
        dtype,
        bounds=bounds,
        interesting_points=_EXP_INTERESTING_POINTS,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp_highest_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.5, f64: 2.0}),
        ("gpu", {bf16: 1.0, f16: 1.0, f32: 2.0, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p"], {bf16: 1.0, f16: 1.0, f32: 1.5}),
        (["tpu_v6e", "tpu_7x"], {f32: 1.5}),
    ]
    util.check_unary_precision(
        self,
        _exp_highest,
        np.exp,
        mpmath.exp,
        dtype,
        bounds=bounds,
        interesting_points=_EXP_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class Exp2Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp2_accuracy(self, dtype):
    if xla_version >= 80 and jtu.is_libtpu_at_least("0.0.52"):
      bounds = [
          ("cpu", {f16: 1.0}),
          ("gpu", {f16: 1.0, f32: 2.5, f64: 1.0}),
          (TPU_EUPV1, {f16: 1.0, f32: 61.0}),
          ("tpu_v5p", {f16: 1.0, f32: 65.5}),
          (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 1.5}),
      ]
    else:
      bounds = [
          ("cpu", {bf16: 101.0, f16: 14.0, f32: 68.5, f64: 719.0}),
          ("gpu", {bf16: 101.0, f16: 14.0, f32: 69.0, f64: 719.0}),
          (TPU_EUPV1, {bf16: 44.0, f16: 7.5, f32: 141.5}),
          ("tpu_v5p", {bf16: 44.0, f16: 7.5, f32: 133.0}),
          ("tpu_v6e", {bf16: 44.0, f16: 7.5, f32: 90.0}),
          ("tpu_7x", {bf16: 75.0, f16: 7.5, f32: 90.0}),
      ]
    ignore_inputs = [
        (["cpu", "gpu", "tpu"], {bf16: [0x4300]}),
    ]
    # Worst-case inputs across math libraries from Gladman et al.
    interesting_points = [
        float.fromhex(h)
        for h in (
            "-0x1.1a4ce073ea908p-5",
            "-0x1.8002f29666b99p-6",
            "0x1.fffbff4152bafp+9",
            "-0x1.ff95ecb4e6331p-2",
            "-0x1.ff1eb5acee46bp+9",
            "-0x1.b3d9b47ad1b2fp-13",
            "0x1.71547652b82fep-53",
            "-0x1.e383ca1fd4462p+9",
            "-0x1.f7087fb1cf9e8p+9",
            "-0x1.ff40169bd093bp+9",
            "-0x1.ff3e09d141586p+9",
        )
    ]
    util.check_unary_precision(
        self,
        jnp.exp2,
        np.exp2,
        lambda x: mpmath.power(2, x),
        dtype,
        bounds=bounds,
        ignore_inputs=ignore_inputs,
        interesting_points=interesting_points,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp2_highest_accuracy(self, dtype):
    if xla_version < 80 or not jtu.is_libtpu_at_least("0.0.52"):
      self.skipTest("Requires xla_version >= 80 and libtpu >= 0.0.52")
    bounds = [
        ("cpu", {f16: 1.0}),
        ("gpu", {f16: 1.0, f32: 2.5, f64: 1.0}),
        ("tpu", {f16: 1.0, f32: 1.5}),
    ]
    ignore_inputs = [
        (["cpu", "gpu", "tpu"], {bf16: [0x4300]}),
    ]
    util.check_unary_precision(
        self,
        _exp2_highest,
        np.exp2,
        lambda x: mpmath.power(2, x),
        dtype,
        bounds=bounds,
        ignore_inputs=ignore_inputs,
    )


@jtu.thread_unsafe_test_class()
class Expm1Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_expm1_accuracy(self, dtype):
    tpu_bounds = [
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 1772.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 1357.5}),
        (["tpu_v6e", "tpu_7x"], {f32: 3.5}),
    ]
    check_signed_zeros = [
        ([*TPU_EUPV1, "tpu_v5p"], False),
    ]
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.0, f64: 1.0}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 1.5}),
        *tpu_bounds,
    ]
    util.check_unary_precision(
        self,
        jnp.expm1,
        np.expm1,
        mpmath.expm1,
        dtype,
        bounds=bounds,
        check_signed_zeros=check_signed_zeros,
        interesting_points=_expm1_interesting_points(dtype),
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_expm1_highest_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.0, f64: 1.0}),
        ("gpu", {f16: 1.0, f32: 1.5, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 1.0}),
        (["tpu_v6e", "tpu_7x"], {f32: 1.0}),
    ]
    util.check_unary_precision(
        self,
        _expm1_highest,
        np.expm1,
        mpmath.expm1,
        dtype,
        bounds=bounds,
        interesting_points=_expm1_interesting_points(dtype),
    )


def _exp2m1_ref(x: np.ndarray) -> np.ndarray:
  x64 = x.astype(np.float64)
  small = np.abs(x64) <= 0.5
  x_small = np.where(small, x64, 0.0)
  x_large = np.where(small, 0.0, x64)
  return np.where(
      small, np.expm1(x_small * np.log(2.0)), np.exp2(x_large) - 1.0
  ).astype(np.float64)


def _mpmath_exp2m1(x):
  with mpmath.extraprec(40):
    if abs(x) < 0.5:
      return mpmath.expm1(x * mpmath.ln(2))
    return mpmath.power(2, x) - 1


def _exp2m1_interesting_points(dtype):
  p = jnp.finfo(dtype).nmant + 1
  return [
      float(-p),
      float(-(p + 1)),
      float(-(p + 2)),
      float(p),
      float(p + 1),
      -1.5,
      -0.5,
      0.5,
      float.fromhex("0x1.00000000007d7p-1"),
      float.fromhex("0x1.18370b38c9be1p-1"),
      1.5,
  ]


@jtu.thread_unsafe_test_class()
class Exp2m1Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp2m1_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.0, f64: 1.0}),
        ("gpu", {f16: 1.0, f32: 1.0, f64: 1.0}),
        ("tpu", {f16: 1.0, f32: 1.0}),
    ]
    util.check_unary_precision(
        self,
        lax.exp2m1,
        _exp2m1_ref,
        _mpmath_exp2m1,
        dtype,
        bounds=bounds,
        interesting_points=_exp2m1_interesting_points(dtype),
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp2m1_special_values(self, dtype):
    if dtype == f64 and (
        not config.enable_x64.value or jtu.device_under_test() == "tpu"
    ):
      self.skipTest("Requires native float64")
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    finfo = jnp.finfo(dtype)
    p = finfo.nmant + 1
    max_exact_k = min(p - 1, finfo.maxexp - 1)
    cases = [
        (0.0, 0.0),
        (-0.0, -0.0),
        (-np.inf, -1.0),
        (np.inf, np.inf),
        *((float(k), float(2**k - 1)) for k in range(1, max_exact_k + 1)),
        *((float(-k), float(2.0**-k - 1.0)) for k in range(1, p + 1)),
        (float(-(p + 1)), -1.0),
    ]
    xs = np.array([c[0] for c in cases], dtype=dtype)
    expected = np.array([c[1] for c in cases], dtype=dtype)
    jitted = jax.jit(lax.exp2m1)
    actual = np.asarray(jitted(xs))
    self.assertDtypesMatch(actual, expected)
    self.assertArraysEqual(actual.view(udt), expected.view(udt))
    self.assertTrue(np.isnan(float(jitted(np.array(np.nan, dtype=dtype)))))

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_exp2m1_grad_and_hessian(self, dtype):
    if dtype == f64 and (
        not config.enable_x64.value or jtu.device_under_test() == "tpu"
    ):
      self.skipTest("Requires native float64")
    finfo = jnp.finfo(dtype)
    p = finfo.nmant + 1
    jvp_fn = jax.jit(lambda x, dx: jax.jvp(lax.exp2m1, (x,), (dx,))[1])
    grad_fn = jax.jit(jax.grad(lax.exp2m1))
    hess_fn = jax.jit(jax.hessian(lax.exp2m1))
    tol = 8 * float(finfo.eps)
    if jtu.device_under_test() == "tpu" and dtype == f32:
      tol = max(tol, 2e-4)
    e_max = finfo.maxexp - 1
    x_max = jnp.array(float(e_max), dtype=dtype)
    dx = jnp.array(2.5, dtype=dtype)
    self.assertAllClose(
        float(jvp_fn(x_max, dx)),
        2.5 * np.log(2.0) * (2.0**e_max),
        atol=0,
        rtol=tol,
    )
    for s in (-2.0, 0.0, 1.5, float(-(p + 2))):
      x = jnp.array(s, dtype=dtype)
      self.assertAllClose(
          float(grad_fn(x)), np.log(2.0) * (2.0**s), atol=0, rtol=tol
      )
      self.assertAllClose(
          float(hess_fn(x)), (np.log(2.0) ** 2) * (2.0**s), atol=0, rtol=tol
      )


@jtu.thread_unsafe_test_class()
class LogisticTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logistic_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.5, f16: 2.0, f32: 2.5, f64: 3.5}),
        ("gpu", {bf16: 2.0, f16: 2.0, f32: 4.0, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 243.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 124.0}),
        ("tpu_v6e", {f16: 1.0, f32: 65.5}),
        ("tpu_7x", {bf16: 63.0, f16: 1.0, f32: 64.0}),
    ]
    util.check_unary_precision(
        self,
        lax.logistic,
        lambda x: 1.0 / (1.0 + np.exp(-x)),
        lambda x: 1 / (1 + mpmath.exp(-x)),
        dtype,
        bounds=bounds,
        interesting_points=_logistic_interesting_points(dtype),
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logistic_highest_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.0, f16: 2.0, f32: 2.5, f64: 2.5}),
        ("gpu", {bf16: 2.0, f16: 2.0, f32: 4.0, f64: 2.5}),
        ([*TPU_EUPV1, "tpu_v6e"], {bf16: 1.0, f16: 1.0, f32: 4.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 3.5}),
        (
            "tpu_7x",
            {
                bf16: 0.5 if jtu.is_libtpu_at_least("0.0.52") else 63.0,
                f16: 1.0,
                f32: 4.0,
            },
        ),
    ]
    with jtu.global_config_context(jax_accurate_logistic=True):
      util.check_unary_precision(
          self,
          _logistic_highest,
          lambda x: 1.0 / (1.0 + np.exp(-x)),
          lambda x: 1 / (1 + mpmath.exp(-x)),
          dtype,
          bounds=bounds,
          interesting_points=_logistic_interesting_points(dtype),
      )


@jtu.thread_unsafe_test_class()
class LogisticGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logistic_grad_highest_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 3.5, f16: (4.0, 4.5), f32: 6.0, f64: 6.0}),
        ("gpu", {bf16: 3.5, f16: 4.5, f32: 8.5, f64: 4.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 7.5}),
        ("tpu_v5p", {bf16: 0.5, f16: 1.0, f32: 7.5}),
        ("tpu_v6e", {bf16: 0.5, f16: 1.0, f32: 8.0}),
        (
            "tpu_7x",
            {
                bf16: 0.5 if jtu.is_libtpu_at_least("0.0.52") else 63.0,
                f16: 1.0,
                f32: 8.0,
            },
        ),
    ]
    with jtu.global_config_context(jax_tanh_sech_jvp=True):
      util.check_unary_precision(
          self,
          jax.jit(jax.vmap(jax.grad(_logistic_highest))),
          lambda x: (1.0 / (1.0 + np.exp(-x))) * (1.0 / (1.0 + np.exp(x))),
          lambda x: 1 / (4 * mpmath.cosh(x / 2) ** 2),
          dtype,
          bounds=bounds,
          interesting_points=_logistic_interesting_points(dtype),
      )

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_logistic_higher_order_highest_accuracy(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    d2_fn = jax.jit(jax.vmap(jax.grad(jax.grad(_logistic_highest))))
    d3_fn = jax.jit(jax.vmap(jax.grad(jax.grad(jax.grad(_logistic_highest)))))

    # Include tiny non-zero x (where 1 - 2 * logistic(x) previously cancelled
    # 0.5 - 0.5 to 0.0) and large |x| where logistic(x) saturates to 0 or 1.
    x_tiny = 1e-8 if dtype == f32 else 1e-17
    x_tail = 20.0 if dtype == f32 else 50.0
    xs = jnp.array(
        [-x_tail, -2.0, -x_tiny, 0.0, x_tiny, 2.0, x_tail], dtype=dtype
    )

    def mp_d2(v):
      t = mpmath.tanh(v / 2)
      s = 1 / (4 * mpmath.cosh(v / 2) ** 2)
      return -t * s

    def mp_d3(v):
      t = mpmath.tanh(v / 2)
      s = 1 / (4 * mpmath.cosh(v / 2) ** 2)
      return s * (t * t - 2 * s)

    with jtu.global_config_context(jax_tanh_sech_jvp=True):
      util.check_precision_at_points(self, d2_fn, mp_d2, xs, max_ulp=8.0)
      util.check_precision_at_points(self, d3_fn, mp_d3, xs, max_ulp=16.0)
      infs = jnp.array([-jnp.inf, jnp.inf], dtype=dtype)
      self.assertArraysEqual(d2_fn(infs), jnp.zeros_like(infs))
      self.assertArraysEqual(d3_fn(infs), jnp.zeros_like(infs))

  @parameterized.named_parameters(
      ("_complex64", jnp.complex64), ("_complex128", jnp.complex128)
  )
  def test_logistic_complex_grad_highest_accuracy(self, dtype):
    if dtype == jnp.complex128 and jtu.device_under_test() == "tpu":
      self.skipTest("complex128/float64 on TPU is ef57 double-double")
    xs = jnp.array(
        [-2.0 + 0.5j, -0.1 - 0.3j, 0.1 + 0.3j, 2.0 - 0.5j], dtype=dtype
    )
    with jtu.global_config_context(jax_tanh_sech_jvp=True):
      jtu.check_grads(_logistic_highest, (xs,), order=2, modes=["fwd", "rev"])
      # Also verify large |Re(x)| doesn't produce NaNs from overflow.
      x_tail = 20.0 if dtype == jnp.complex64 else 50.0
      xs_tail = jnp.array([-x_tail + 1j, x_tail + 1j], dtype=dtype)
      _, jvps = jax.jvp(_logistic_highest, (xs_tail,), (jnp.ones_like(xs_tail),))
      self.assertAllClose(jvps, jnp.zeros_like(jvps), atol=1e-8)


util.register_benchmark(jnp.exp)
util.register_benchmark(_exp_highest)
util.register_benchmark(jnp.exp2)
util.register_benchmark(_exp2_highest)
util.register_benchmark(lax.exp2m1)
util.register_benchmark(jnp.expm1)
util.register_benchmark(_expm1_highest)
util.register_benchmark(lax.logistic)
util.register_benchmark(_logistic_highest)


if __name__ == "__main__":
  util.main()
