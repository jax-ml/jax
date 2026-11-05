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
from jax._src.lib import ifrt_version
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
    bounds = [
        ("cpu", {f16: 1.0, f32: 1.5}),
        ("gpu", {f16: 1.0, f32: 1.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4030.5}),
        ("tpu_v5p", {f16: 1.0, f32: 62.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 2.5}),
    ]
    input_ftz = [
        ("gpu", False),
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
        input_ftz=input_ftz,
        interesting_points=interesting_points,
    )


@jtu.thread_unsafe_test_class()
class Log2Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log2_accuracy(self, dtype):
    if ifrt_version >= 80 and jtu.is_libtpu_at_least("0.0.52"):
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
    input_ftz = [
        ("gpu", False),
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
        input_ftz=input_ftz,
        interesting_points=interesting_points,
    )

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log2_highest_accuracy(self, dtype):
    if ifrt_version < 80 or not jtu.is_libtpu_at_least("0.0.52"):
      self.skipTest("Requires ifrt_version >= 80 and libtpu >= 0.0.52")
    bounds = [
        ("gpu", {f32: 1.0, f64: 1.5}),
        ([*TPU_EUPV1, "tpu_v5p"], {f32: 3.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 1.5}),
    ]
    input_ftz = [
        ("gpu", False),
    ]
    util.check_unary_precision(
        self,
        _log2_highest,
        np.log2,
        mpmath.log2,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
    )


@jtu.thread_unsafe_test_class()
class Log10Test(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_log10_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 2.0, f16: 1.5, f32: 3.0, f64: 2.0}),
        ("gpu", {bf16: 2.0, f16: 1.5, f32: 2.5, f64: 2.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 6213.0}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 57.0}),
        (["tpu_v6e", "tpu_7x"], {bf16: 1.0, f16: 1.0, f32: 3.0}),
    ]
    input_ftz = [
        ("gpu", False),
    ]
    util.check_unary_precision(
        self,
        jnp.log10,
        np.log10,
        mpmath.log10,
        dtype,
        bounds=bounds,
        input_ftz=input_ftz,
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
    bounds = [
        ("cpu", {f16: 1.0, f32: 3.0, f64: 2.0}),
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
    bounds = [
        (["cpu", "gpu"], {bf16: 1.5, f16: 1.5, f32: 3.0, f64: 1.5}),
        (TPU_EUPV1, {bf16: 1.0, f16: 1.0, f32: 4032.5}),
        ("tpu_v5p", {bf16: 1.0, f16: 1.0, f32: 64.0}),
        ("tpu_v6e", {bf16: 1.0, f16: 1.0, f32: 4.0}),
        ("tpu_7x", {bf16: 1.0, f16: 1.0, f32: 4.5}),
    ]
    input_ftz = [
        ("gpu", False),
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
        input_ftz=input_ftz,
        interesting_points=interesting_points,
    )


util.register_benchmark(jnp.log)
util.register_benchmark(jnp.log2)
util.register_benchmark(_log2_highest)
util.register_benchmark(jnp.log10)
util.register_benchmark(jnp.log1p)
util.register_benchmark(jsp.special.logit)


if __name__ == "__main__":
  util.main()
