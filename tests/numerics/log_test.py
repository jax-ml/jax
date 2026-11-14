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

"""Precision tests for logarithmic functions."""

import math

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import xla_version
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np
import scipy.special

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]

# Mantissa range-reduction breakpoints in [0.5, 2.0] for log implementations.
_LOG_INTERESTING_POINTS = [
    math.sqrt(0.5),
    math.sqrt(2.0),
    2.0 / 3.0,
    4.0 / 3.0,
]


@jax.jit
def _log2_highest(x):
  return lax.log2(x, accuracy=lax.AccuracyMode.HIGHEST)


@jtu.thread_unsafe_test_class()
class LogTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log_accuracy(self, dtype):
    if xla_version >= 88:
      cpu_bounds = ("cpu", {f16: 1.0, f32: 1.5, f64: 1.5})
    else:
      cpu_bounds = ("cpu", {f16: 1.0, f32: 1.5})
    bounds = [
        cpu_bounds,
        ("gpu", {f16: 1.0, f32: 1.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4030.5}),
        ("tpu_v5p", {f16: 1.0, f32: 62.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 2.5}),
    ]
    interesting_points = [
        *_LOG_INTERESTING_POINTS,
        # Worst-case inputs across math libraries from Gladman et al.,
        # "Accuracy of Mathematical Functions in Single, Double, Double
        # Extended, and Quadruple Precision":
        *(
            float.fromhex(h)
            for h in (
                "0x1.1211bef8f68e9p+0",
                "0x1.008000db2e8bep+0",
                "0x1.108058fed54b3p+0",
                "0x1.48ae5a67204f5p+0",
                "0x1.dc0b586f2b26p-1",
                "0x1.490af72a25a81p-1",
                "0x1.5b6e7e4e96f86p+2",
                "0x1.0ffc349469a2fp+0",
                "0x1.69e7aa6da2df5p-1",
                "0x1.5556123e8a2bp-1",
            )
        ),
    ]
    util.check_unary_precision(
        self,
        jnp.log,
        np.log,
        mpmath.log,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class Log2Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log2_accuracy(self, dtype):
    if xla_version >= 80 and jtu.is_libtpu_at_least("0.0.52"):
      bounds = [
          ("gpu", {f32: 1.0, f64: 1.5}),
          (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 5159.0}),
          ("tpu_v5p", {f16: 1.0, f32: 56.5}),
          (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 1.5}),
      ]
    else:
      bounds = [
          ("cpu", {bf16: 2.0, f16: 2.0, f32: 2.5, f64: 1.5}),
          ("gpu", {bf16: 2.0, f16: 2.0, f32: 2.0, f64: 1.5}),
          (TPU_EUPV1, {bf16: 1.0, f16: 1.5, f32: 5159.0}),
          ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 57.5}),
          (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 2.5}),
      ]
    interesting_points = [
        *_LOG_INTERESTING_POINTS,
        # Worst-case inputs across math libraries from Gladman et al.:
        *(
            float.fromhex(h)
            for h in (
                "0x1.1406d79e1b574p+0",
                "0x1.fea11b9451f9bp-1",
                "0x1.0b541b6746bd1p+0",
                "0x1.68d778f076021p+0",
                "0x1.67eaf07ce24d1p+0",
                "0x1.0b53197bd66c8p+0",
                "0x1.6d0092bcf066dp-1",
                "0x1.b4ebe40c95a01p+0",
                "0x1.160732376ad7fp+0",
                "0x1.0b53f741fb8c4p+0",
                "0x1.670c5aa6680abp+0",
                "0x1.55571e90ee956p+0",
            )
        ),
    ]
    util.check_unary_precision(
        self,
        jnp.log2,
        np.log2,
        mpmath.log2,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log2_highest_accuracy(self, dtype):
    if xla_version < 80 or not jtu.is_libtpu_at_least("0.0.52"):
      self.skipTest("Requires xla_version >= 80 and libtpu >= 0.0.52")
    bounds = [
        ("gpu", {f32: 1.0, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p"], {f32: 3.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 1.5}),
    ]
    util.check_unary_precision(
        self,
        _log2_highest,
        np.log2,
        mpmath.log2,
        dtype,
        bounds=bounds,
    )


@jtu.thread_unsafe_test_class()
class Log10Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log10_accuracy(self, dtype):
    if xla_version >= 88:
      cpu_bounds = ("cpu", {bf16: 2.0, f16: 1.5, f32: 2.5, f64: 2.0})
    else:
      cpu_bounds = ("cpu", {bf16: 2.0, f16: 1.5, f32: 3.0, f64: 2.0})
    bounds = [
        cpu_bounds,
        ("gpu", {bf16: 2.0, f16: 1.5, f32: 2.5, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 6213.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 57.0}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 3.0}),
    ]
    util.check_unary_precision(
        self,
        jnp.log10,
        np.log10,
        mpmath.log10,
        dtype,
        bounds=bounds,
        interesting_points=[
            *_LOG_INTERESTING_POINTS,
            # Exact powers of 10 where log10(x) is an integer.
            *(10.0**k for k in (-10, -3, -2, -1, 1, 2, 3, 10)),
            # Worst-case inputs across math libraries from Gladman et al.:
            *(
                float.fromhex(h)
                for h in (
                    "0x1.de02157073b31p-1",
                    "0x1.feda7b62c1033p-1",
                    "0x1.10fdf4211fd45p+0",
                    "0x1.55535a0140a21p+0",
                    "0x1.553e1cb579ee9p+0",
                    "0x1.2501ee5628b08p-1",
                    "0x1.e12d66744ff81p+429",
                    "0x1.e005e1e891807p-1",
                    "0x1.803dea263187fp-1",
                    "0x1.55558bf0b2328p+0",
                )
            ),
        ],
    )


@jtu.thread_unsafe_test_class()
class Log1pTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log1p_accuracy(self, dtype):
    if xla_version >= 88:
      cpu_bounds = ("cpu", {f16: 1.0, f32: 2.0, f64: 2.0})
    else:
      cpu_bounds = ("cpu", {f16: 1.0, f32: 3.0, f64: 2.0})
    bounds = [
        cpu_bounds,
        ("gpu", {f16: 1.0, f32: 1.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4034.0}),
        ("tpu_v5p", {f16: 1.0, f32: 63.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 3.0}),
    ]
    # Points where 1 + x crosses range-reduction thresholds in [0.5, 2.0] or e.
    interesting_points = [
        math.sqrt(0.5) - 1.0,
        math.sqrt(2.0) - 1.0,
        math.e - 1.0,
        # Worst-case inputs across math libraries from Gladman et al.:
        *(
            float.fromhex(h)
            for h in (
                "0x1p-149",
                "-0x1.2bf183e0344b2p-2",
                "0x1.000aee2a2757fp-9",
                "0x1.e0013fd35cbbp-4",
                "-0x1.2bf1de6b04a8ap-2",
                "-0x1.2bf32aaf122e2p-2",
                "-0x1.ffffff3fffffdp-28",
                "0x1p-53",
                "-0x1.8000000000000p-53",
                "-0x1.2e496d25897ecp-2",
                "-0x1.ffffffbaefe27p-2",
                "-0x1.5efad5491a79bp-1022",
            )
        ),
    ]
    util.check_unary_precision(
        self,
        jnp.log1p,
        np.log1p,
        mpmath.log1p,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


def _log2p1_ref(x: np.ndarray) -> np.ndarray:
  x64 = x.astype(np.float64)
  small = np.abs(x64) <= 0.5
  x_small = np.where(small, x64, 0.0)
  x_large = np.where(small, 0.0, x64)
  return np.where(
      small, np.log1p(x_small) / np.log(2.0), np.log2(1.0 + x_large)
  ).astype(np.float64)


def _mpmath_log2p1(x):
  if x < -1:
    return mpmath.nan
  if x == -1:
    return -mpmath.inf
  with mpmath.extraprec(40):
    return mpmath.log1p(x) / mpmath.ln(2)


@jtu.thread_unsafe_test_class()
class Log2p1Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log2p1_accuracy(self, dtype):
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.0, f64: 1.0}),
        ("gpu", {f16: 1.0, f32: 1.0, f64: 1.0}),
        ("tpu", {f16: 1.0, f32: 1.0}),
    ]
    interesting_points = [
        math.sqrt(0.5) - 1.0,
        -0.29289321,
        math.sqrt(2.0) - 1.0,
        0.41421354,
        float.fromhex("0x1.a5d8dd8e5ee0cp-2"),
        -0.5,
        -0.75,
        1.0,
        3.0,
        7.0,
    ]
    util.check_unary_precision(
        self,
        lax.log2p1,
        _log2p1_ref,
        _mpmath_log2p1,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log2p1_special_values(self, dtype):
    if dtype == f64 and (
        not config.enable_x64.value or jtu.device_under_test() == "tpu"
    ):
      self.skipTest("Requires native float64")
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    finfo = jnp.finfo(dtype)
    p = finfo.nmant + 1
    max_exact_k = min(p, finfo.maxexp - 1)
    cases = [
        (0.0, 0.0),
        (-0.0, -0.0),
        (-1.0, -np.inf),
        (np.inf, np.inf),
        *((float(2**k - 1), float(k)) for k in range(1, max_exact_k + 1)),
        *((float(2.0**-k - 1.0), float(-k)) for k in range(1, p + 1)),
    ]
    xs = np.array([c[0] for c in cases], dtype=dtype)
    expected = np.array([c[1] for c in cases], dtype=dtype)
    jitted = jax.jit(lax.log2p1)
    actual = np.asarray(jitted(xs))
    self.assertDtypesMatch(actual, expected)
    self.assertArraysEqual(actual.view(udt), expected.view(udt))
    nan_xs = np.array([-1.5, -2.0, -np.inf, np.nan], dtype=dtype)
    self.assertTrue(np.all(np.isnan(np.asarray(jitted(nan_xs)))))

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log2p1_grad_and_hessian(self, dtype):
    if dtype == f64 and (
        not config.enable_x64.value or jtu.device_under_test() == "tpu"
    ):
      self.skipTest("Requires native float64")
    finfo = jnp.finfo(dtype)
    e_hi = finfo.maxexp - 2
    s_hi = 2.0**e_hi
    jvp_fn = jax.jit(lambda x, dx: jax.jvp(lax.log2p1, (x,), (dx,))[1])
    grad_fn = jax.jit(jax.grad(lax.log2p1))
    hess_fn = jax.jit(jax.hessian(lax.log2p1))
    tol = 8 * float(finfo.eps)
    if jtu.device_under_test() == "tpu" and dtype == f32:
      tol = max(tol, 2e-4)
    x_hi = jnp.array(s_hi, dtype=dtype)
    self.assertAllClose(
        float(jvp_fn(x_hi, x_hi)), 1.0 / np.log(2.0), atol=0, rtol=tol
    )
    self.assertGreater(float(grad_fn(x_hi)), 0.0)
    for s in (-0.75, 0.0, 2.0):
      x = jnp.array(s, dtype=dtype)
      self.assertAllClose(
          float(grad_fn(x)), 1.0 / ((1.0 + s) * np.log(2.0)), atol=0, rtol=tol
      )
      self.assertAllClose(
          float(hess_fn(x)),
          -1.0 / (((1.0 + s) ** 2) * np.log(2.0)),
          atol=0,
          rtol=tol,
      )


def _mpmath_logit(x):
  if not (0 <= x <= 1):
    return mpmath.nan
  if x == 0:
    return -mpmath.inf
  if x == 1:
    return mpmath.inf
  with mpmath.extraprec(40):
    return mpmath.log(x / (1 - x))


@jtu.thread_unsafe_test_class()
class LogitTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_logit_accuracy(self, dtype):
    if xla_version >= 88:
      cpu_bounds = ("cpu", {bf16: 1.5, f16: 1.5, f32: 2.0, f64: 1.5})
    else:
      cpu_bounds = ("cpu", {bf16: 1.5, f16: 1.5, f32: 3.0, f64: 1.5})
    bounds = [
        cpu_bounds,
        ("gpu", {bf16: 1.5, f16: 1.5, f32: 3.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4032.5}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 64.0}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 4.0}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 4.5}),
    ]
    interesting_points = [
        0.25,
        1.0 / 3.0,
        0.5,
        2.0 / 3.0,
        0.75,
        1.0 / (1.0 + math.e),
        math.e / (1.0 + math.e),
    ]
    util.check_unary_precision(
        self,
        jax.jit(jsp.special.logit),
        scipy.special.logit,
        _mpmath_logit,
        dtype,
        bounds=bounds,
        interesting_points=interesting_points,
    )


util.register_benchmark(jnp.log)
util.register_benchmark(jnp.log2)
util.register_benchmark(_log2_highest)
util.register_benchmark(lax.log2p1)
util.register_benchmark(jnp.log10)
util.register_benchmark(jnp.log1p)
util.register_benchmark(jsp.special.logit)


if __name__ == "__main__":
  util.main()
