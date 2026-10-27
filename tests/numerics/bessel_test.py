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

from absl.testing import parameterized
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np
import scipy.special

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]


def _mpmath_i0e(x):
  if mpmath.isinf(x):
    return mpmath.mpf(0.0)
  return mpmath.besseli(0, x) * mpmath.exp(-abs(x))


def _mpmath_i1e(x):
  if mpmath.isinf(x):
    return mpmath.mpf(0.0)
  return mpmath.besseli(1, x) * mpmath.exp(-abs(x))


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
    bounds = [
        ("cpu", {f16: 1.0, f32: 2.0, f64: 2.0}),
        ("gpu", {f16: 1.0, f32: 2.5, f64: 2.0}),
        ([*TPU_EUPV1, "tpu_v5p"], {f16: 1.0, f32: 3.0}),
        (["tpu_v6e", "tpu_7x"], {f16: 1.0, f32: 2.0}),
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


if __name__ == "__main__":
  util.main()
