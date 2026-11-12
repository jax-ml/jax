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

"""Precision tests for hyperbolic and inverse hyperbolic functions."""

import math

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version, xla_version
import jax.numpy as jnp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]


def _sinh_cosh_interesting_points(dtype):
  # Overflow threshold where sinh(x)/cosh(x) = 0.5 * exp(|x|) overflows
  # (~ln(max_val) + ln(2); ln(max_val) and +-0.5*ln(2), +-ln(2) are already in
  # _common_interesting_points).
  sinh_overflow = math.log(float(jnp.finfo(dtype).max)) + math.log(2.0)
  return [-sinh_overflow, sinh_overflow]


def _tanh_interesting_points(dtype):
  p = jnp.finfo(dtype).nmant + 1
  return [
      # Saturation thresholds where tanh(x) rounds to +-1.0 (~0.5 * (p + 1) * ln(2)).
      *(
          sign * 0.5 * k * math.log(2.0)
          for k in (p + 1, p + 2)
          for sign in (-1, 1)
      ),
      # Worst-case inputs across math libraries from Gladman et al.,
      # "Accuracy of Mathematical Functions in Single, Double, Double Extended,
      # and Quadruple Precision":
      *(
          float.fromhex(h)
          for h in (
              "0x1.d12ed0af1a27fp-27",
              "0x1.002629fd74484p+0",
              "-0x1.fde5bd2769a01p-1",
              "-0x1.e134557098e37p-3",
              "0x1.00cf9f273d84p+1",
              "0x1.fefc7206bc932p-1",
              "-0x1.c416448380debp-3",
              "-0x1.19398a9a24319p-1",
              "0x1.004a396a065e3p-4",
          )
      ),
  ]


# Note: Separate gradient tests for sinh and cosh are omitted because
# grad(sinh) lowers directly to cosh and grad(cosh) lowers directly to sinh,
# which are already covered by CoshTest and SinhTest.


@jax.jit
def tanh_grad(x):
  return jax.vmap(jax.grad(jnp.tanh))(x)


@jax.jit
def tanh_grad_highest(x):
  return jax.vmap(
      jax.grad(lambda x: lax.tanh(x, accuracy=lax.AccuracyMode.HIGHEST))
  )(x)


@jax.jit
def acosh_grad(x):
  return jax.vmap(jax.grad(jnp.acosh))(x)


@jax.jit
def asinh_grad(x):
  return jax.vmap(jax.grad(jnp.asinh))(x)


@jax.jit
def atanh_grad(x):
  return jax.vmap(jax.grad(jnp.atanh))(x)


@jtu.thread_unsafe_test_class()
class SinhTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_sinh_accuracy(self, dtype):
    if jaxlib_extension_version >= 507:
      bounds = [
          ("cpu", {f16: 1.0, f32: 1.0, f64: 2.0}),
          ("gpu", {f32: 3.0, f64: 2.5}),
          ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 1.0}),
          (["tpu_v6e", "tpu_7x"], {f32: 1.0}),
      ]
      ignore_inputs = []
      check_signed_zeros = True
    else:
      bounds = [
          ("cpu", {f16: 1.0, f32: 25.0, f64: 496.0}),
          ("gpu", {f32: 3.0, f64: 2.5}),
          (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 1794.0}),
          ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 1332.5}),
          ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 59.0}),
          ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 59.5}),
      ]
      # Ignore x = +-89.415985: near the float32 overflow threshold, where the
      # true value is finite (~3.40282e+38) but XLA's lowering on CPU/TPU
      # computes 0.5 * exp(x), where exp(x) overflows float32 to inf.
      ignore_inputs = [
          (["cpu", "tpu"], {f32: [0x42B2D4FC, 0xC2B2D4FC]}),
      ]
      check_signed_zeros = [
          ("tpu", False),
      ]
    interesting_points = [
        *_sinh_cosh_interesting_points(dtype),
        # Worst-case inputs across math libraries from Gladman et al.:
        *(
            float.fromhex(h)
            for h in (
                "0x1.2b705c15abeb9p+5",
                "-0x1.adc135eb544c1p-2",
                "0x1.1feb2a79f307p+3",
                "-0x1.633cae1335f26p+9",
                "-0x1.63324af2fb5b7p-1",
                "0x1.d7131e11fc6b3p-2",
                "0x1.1fd8c752e39f5p+3",
                "0x1.9fa32b1149d35p-2",
                "0x1.bde0cd0efdef8p+0",
                "0x1.ffa5d2e7b5p-5",
            )
        ),
    ]
    util.check_unary_precision(
        self,
        jnp.sinh,
        np.sinh,
        mpmath.sinh,
        dtype,
        bounds=bounds,
        ignore_inputs=ignore_inputs,
        check_signed_zeros=check_signed_zeros,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class CoshTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_cosh_accuracy(self, dtype):
    if jaxlib_extension_version >= 511:
      bounds = [
          ("cpu", {f16: 1.0, f32: 1.0, f64: 2.0}),
          ("gpu", {f16: 1.0, f32: 2.5, f64: 2.5}),
          ("tpu", {f16: 1.0, f32: 1.0}),
      ]
      ignore_inputs = []
    else:
      bounds = [
          ("cpu", {f16: 1.0, f32: 25.0, f64: 496.0}),
          ("gpu", {f16: 1.0, f32: 2.5, f64: 2.5}),
          (TPU_EUPV1, {f16: 1.0, f32: 93.5}),
          ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 99.0}),
          ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 59.0}),
          ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 59.5}),
      ]
      # Ignore x = +-89.415985: near the float32 overflow threshold, where the
      # true value is finite (~3.40282e+38) but XLA's lowering on CPU/TPU
      # computes 0.5 * exp(x), where exp(x) overflows float32 to inf.
      ignore_inputs = [
          (["cpu", "tpu"], {f32: [0x42B2D4FC, 0xC2B2D4FC]}),
      ]
    interesting_points = [
        *_sinh_cosh_interesting_points(dtype),
        # Worst-case inputs across math libraries from Gladman et al.:
        *(
            float.fromhex(h)
            for h in (
                "0x1p-26",
                "-0x1.5a364e6b98134p+9",
                "0x1.ff76fb3f476d5p+0",
                "0x1.633cc2ae1c934p+9",
                "-0x1.6310ab92794a8p+9",
                "-0x1.502bf5ad80729p+0",
                "-0x1.62dabd4848dc4p-2",
                "0x1.ed991002dc978p+0",
                "-0x1.628af341989dap+9",
                "0x1.e7ffe229fe99ep+1",
                "-0x1.e7f9c78a5c332p+1",
            )
        ),
    ]
    util.check_unary_precision(
        self,
        jnp.cosh,
        np.cosh,
        mpmath.cosh,
        dtype,
        bounds=bounds,
        ignore_inputs=ignore_inputs,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class TanhTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tanh_accuracy(self, dtype):
    bounds = [
        ("cpu", {f32: 5.0, f64: 7.0}),
        ("gpu", {f32: 5.5, f64: 3.5}),
        (TPU_EUPV1, {f16: 1.0, f32: 1365.5}),
        ("tpu_v5p", {f16: 1.0, f32: 92.0}),
        (["tpu_v6e", "tpu_7x"], {f32: 1.5}),
    ]
    input_ftz = [
        ("cpu", {bf16: False, f32: False}),
    ]
    util.check_unary_precision(
        self,
        jnp.tanh,
        np.tanh,
        mpmath.tanh,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
        interesting_points=_tanh_interesting_points(dtype),
    )


@jtu.thread_unsafe_test_class()
class TanhGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tanh_grad_accuracy(self, dtype):
    # By default, grad(tanh)(x) is evaluated from y = tanh(x) as
    # (1 + y) * (1 - y), which suffers catastrophic cancellation as |y|
    # approaches 1 and rounds to 0.0 once tanh(x) saturates to +-1.0 (~2^p ULPs).
    bounds = [
        (
            "cpu",
            {
                bf16: 256.0,
                f16: 2045.5,
                f32: 16777206.5,
                f64: 117093590311632992.0,
            },
        ),
        (
            "gpu",
            {
                bf16: 256.0,
                f16: 2045.5,
                f32: 16777206.5,
                f64: 9007199254740964.5,
            },
        ),
        (TPU_EUPV1, {bf16: 383.5, f16: 723.5, f32: 51270903.0}),
        ("tpu_v5p", {bf16: 274.5, f16: 72.0, f32: 18925970.5}),
        ("tpu_v6e", {bf16: 256.0, f16: 3.0, f32: 16777199.0}),
        ("tpu_7x", {bf16: 256.0, f16: 2.5, f32: 16777199.0}),
    ]
    util.check_unary_precision(
        self,
        tanh_grad,
        lambda x: np.square(np.reciprocal(np.cosh(x))),
        lambda x: 1 / mpmath.cosh(x) ** 2,
        dtype,
        bounds=bounds,
        interesting_points=_tanh_interesting_points(dtype),
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_tanh_grad_highest_accuracy(self, dtype):
    # With accuracy=AccuracyMode.HIGHEST and jax_tanh_sech_jvp=True,
    # grad(tanh)(x) is evaluated as (2 * exp(-|x|) / (1 + exp(-2 * |x|)))^2,
    # avoiding both 1 - tanh(x)^2 cancellation for large |x| and premature
    # underflow/overflow when 2 * |x| exceeds the normal exponent range.
    bounds = [
        ("cpu", {bf16: 3.5, f16: (4.0, 4.5), f32: 6.0, f64: 6.0}),
        ("gpu", {bf16: 3.5, f16: 4.5, f32: 8.5, f64: 5.0}),
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
          tanh_grad_highest,
          lambda x: np.square(np.reciprocal(np.cosh(x))),
          lambda x: 1 / mpmath.cosh(x) ** 2,
          dtype,
          bounds=bounds,
          interesting_points=_tanh_interesting_points(dtype),
      )

  @parameterized.named_parameters(("_float32", f32), ("_float64", f64))
  def test_tanh_higher_order_highest_accuracy(self, dtype):
    if dtype == f64 and jtu.device_under_test() == "tpu":
      self.skipTest("float64 on TPU is ef57 double-double")
    f = lambda x: lax.tanh(x, accuracy=lax.AccuracyMode.HIGHEST)
    d2_fn = jax.jit(jax.vmap(jax.grad(jax.grad(f))))
    d3_fn = jax.jit(jax.vmap(jax.grad(jax.grad(jax.grad(f)))))

    # Include tiny non-zero x (where logistic(-2x) - logistic(2x) previously
    # cancelled 0.5 - 0.5 to 0.0) and large |x| where tanh(x) saturates to +-1.
    x_tiny = 1e-8 if dtype == f32 else 1e-17
    x_tail = 15.0 if dtype == f32 else 30.0
    xs = jnp.array(
        [-x_tail, -2.0, -x_tiny, 0.0, x_tiny, 2.0, x_tail], dtype=dtype
    )

    def mp_d2(v):
      t = mpmath.tanh(v)
      s = 1 / mpmath.cosh(v) ** 2
      return -2 * t * s

    def mp_d3(v):
      t = mpmath.tanh(v)
      s = 1 / mpmath.cosh(v) ** 2
      return -2 * s * (s - 2 * t * t)

    with jtu.global_config_context(jax_tanh_sech_jvp=True):
      util.check_precision_at_points(self, d2_fn, mp_d2, xs, max_ulp=8.0)
      util.check_precision_at_points(self, d3_fn, mp_d3, xs, max_ulp=16.0)
      infs = jnp.array([-jnp.inf, jnp.inf], dtype=dtype)
      self.assertArraysEqual(d2_fn(infs), jnp.zeros_like(infs))
      self.assertArraysEqual(d3_fn(infs), jnp.zeros_like(infs))

  @parameterized.named_parameters(
      ("_complex64", jnp.complex64), ("_complex128", jnp.complex128)
  )
  def test_tanh_complex_grad_highest_accuracy(self, dtype):
    if dtype == jnp.complex128 and jtu.device_under_test() == "tpu":
      self.skipTest("complex128/float64 on TPU is ef57 double-double")
    f = lambda x: lax.tanh(x, accuracy=lax.AccuracyMode.HIGHEST)
    xs = jnp.array(
        [-2.0 + 0.5j, -0.1 - 0.3j, 0.1 + 0.3j, 2.0 - 0.5j], dtype=dtype
    )
    with jtu.global_config_context(jax_tanh_sech_jvp=True):
      jtu.check_grads(f, (xs,), order=2, modes=["fwd", "rev"])
      # Also verify large |Re(x)| doesn't produce NaNs from overflow.
      x_tail = 15.0 if dtype == jnp.complex64 else 30.0
      xs_tail = jnp.array([-x_tail + 1j, x_tail + 1j], dtype=dtype)
      _, jvps = jax.jvp(f, (xs_tail,), (jnp.ones_like(xs_tail),))
      self.assertAllClose(jvps, jnp.zeros_like(jvps), atol=1e-10)


@jtu.thread_unsafe_test_class()
class AcoshTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acosh_accuracy(self, dtype):
    if xla_version >= 88:
      cpu_bounds = ("cpu", {bf16: 2.0, f16: 2.0, f32: 3.5, f64: 3.5})
    else:
      cpu_bounds = ("cpu", {bf16: 2.0, f16: 2.0, f32: 4.5, f64: 3.5})
    bounds = [
        cpu_bounds,
        ("gpu", {f16: 1.0, f32: 2.5, f64: 3.0}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4031.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 61.0}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 4.5}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 5.0}),
    ]
    interesting_points = [
        math.cosh(1.0),
        # Worst-case inputs across math libraries from Gladman et al.:
        *(
            float.fromhex(h)
            for h in (
                "0x1.c29124ea23a1cp+950",
                "0x1.01825ca7da7e5p+0",
                "0x1.209fae707a0edp+0",
                "0x1.0001ff6afc4bap+0",
                "-0x1.34e729fd08086p+21",
                "0x1.00007fb3703ddp+0",
                "0x1.0007fd4307b75p+0",
                "0x1.071334daf83adp+0",
                "0x1.1d7bc19163966p+0",
                "0x1.0aaade11e199ap+0",
            )
        ),
    ]
    util.check_unary_precision(
        self,
        jnp.acosh,
        np.arccosh,
        mpmath.acosh,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class AcoshGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_acosh_grad_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.0, f16: 2.0, f32: 5.5, f64: 3.0}),
        ("gpu", {bf16: 2.0, f16: 2.0, f32: 5.0, f64: 2.5}),
        (TPU_EUPV1, {f32: 5.0}),
        ("tpu_v5p", {f32: 5.0}),
        ("tpu_v6e", {f32: 3.5}),
        ("tpu_7x", {f32: 3.0}),
    ]
    util.check_unary_precision(
        self,
        acosh_grad,
        lambda x: np.reciprocal(np.sqrt(x - 1.0) * np.sqrt(x + 1.0)),
        lambda x: (
            mpmath.nan
            if x < 1
            else (mpmath.inf if x == 1 else 1 / mpmath.sqrt(x * x - 1))
        ),
        dtype,
        bounds=bounds,
    )


@jtu.thread_unsafe_test_class()
class AsinhTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asinh_accuracy(self, dtype):
    if xla_version >= 88:
      cpu_bounds = ("cpu", {bf16: 1.5, f16: 1.5, f32: 2.0, f64: 2.5})
    else:
      cpu_bounds = ("cpu", {bf16: 1.5, f16: 1.5, f32: 3.5, f64: 2.5})
    bounds = [
        cpu_bounds,
        ("gpu", {f16: 1.0, f32: 2.0, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4034.5}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 62.5}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 3.5}),
    ]
    interesting_points = [
        -math.sinh(1.0),
        math.sinh(1.0),
        # Worst-case inputs across math libraries from Gladman et al.:
        *(
            float.fromhex(h)
            for h in (
                "-0x1.7137449123ef7p-26",
                "-0x1.000276d9cf31ap-4",
                "0x1.005ae8d126f7ep+0",
                "-0x1.02657ff36d5f3p-2",
                "-0x1.0240f2bdb3f25p-2",
                "-0x1.fdefd03df4cd7p-3",
                "-0x1.00212bb59c31ep-4",
                "-0x1.000eeed78380ap+0",
                "-0x1.8190e51c77d16p-1",
                "0x1.2aab2f0a9d9dp-2",
            )
        ),
    ]
    util.check_unary_precision(
        self,
        jnp.asinh,
        np.arcsinh,
        mpmath.asinh,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class AsinhGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_asinh_grad_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.0, f16: 2.0, f32: 2.5, f64: 2.0}),
        ("gpu", {bf16: 2.0, f16: 2.0, f32: 2.5, f64: 2.0}),
        (TPU_EUPV1, {f16: 1.0, f32: 3.0}),
        ("tpu_v5p", {f16: 1.0, f32: 3.0}),
        ("tpu_v6e", {f16: 1.0, f32: 2.5}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 2.0}),
    ]
    util.check_unary_precision(
        self,
        asinh_grad,
        lambda x: np.reciprocal(np.sqrt(np.square(x) + 1.0)),
        lambda x: 1 / mpmath.sqrt(x * x + 1),
        dtype,
        bounds=bounds,
    )


@jtu.thread_unsafe_test_class()
class AtanhTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atanh_accuracy(self, dtype):
    if xla_version >= 88:
      cpu_bounds = ("cpu", {bf16: 1.5, f16: 1.5, f32: 2.0, f64: 2.5})
    else:
      cpu_bounds = ("cpu", {bf16: 1.5, f16: 1.5, f32: 3.0, f64: 2.5})
    bounds = [
        cpu_bounds,
        ("gpu", {f32: 3.5, f64: 3.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 2183.5}),
        ("tpu_v5p", {f16: 1.0, f32: 43.0}),
        (["tpu_v6e", "tpu_7x"], {f32: 3.5}),
    ]
    interesting_points = [
        -math.tanh(1.0),
        math.tanh(1.0),
        # Worst-case inputs across math libraries from Gladman et al.:
        *(
            float.fromhex(h)
            for h in (
                "0x1.a6a58d55e307bp-26",
                "-0x1.e2cfb2667f17ep-9",
                "-0x1.d8fb311a52173p-2",
                "-0x1.f97fabc0650c4p-4",
                "-0x1.f8a404597baf4p-4",
                "0x1.ffd834a270fp-10",
                "-0x1.ffbe8dd88527fp-9",
                "-0x1.e7c1f36602014p-4",
                "-0x1.f586714622a66p-3",
                "-0x1.2493fecd07e5p-3",
            )
        ),
    ]
    util.check_unary_precision(
        self,
        jnp.atanh,
        np.arctanh,
        mpmath.atanh,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class AtanhGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_atanh_grad_accuracy(self, dtype):
    bounds = [
        (["cpu", "gpu"], {bf16: 2.5, f16: 2.0, f32: 2.5, f64: 3.5}),
        ([*TPU_EUPV1, "tpu_v5p", "tpu_v6e"], {f16: 1.0, f32: 3.0}),
        ("tpu_7x", {bf16: 2.5, f16: 1.0, f32: 3.0}),
    ]
    util.check_unary_precision(
        self,
        atanh_grad,
        lambda x: np.reciprocal((1.0 - x) * (1.0 + x)),
        lambda x: mpmath.inf if abs(x) == 1 else 1 / (1 - x * x),
        dtype,
        bounds=bounds,
        domain=lambda dt: (
            util.Domain.interval(dt, -1.0, 1.0) | util.Domain.infs_and_nans(dt)
        ),
    )


util.register_benchmark(jnp.sinh)
util.register_benchmark(jnp.cosh)
util.register_benchmark(jnp.tanh)
util.register_benchmark(tanh_grad)
util.register_benchmark(tanh_grad_highest)
util.register_benchmark(jnp.acosh)
util.register_benchmark(acosh_grad)
util.register_benchmark(jnp.asinh)
util.register_benchmark(asinh_grad)
util.register_benchmark(jnp.atanh)
util.register_benchmark(atanh_grad)


if __name__ == "__main__":
  util.main()
