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

"""Precision tests for arithmetic and remainder functions."""

import math

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]


def _mpmath_fmod(x, y):
  fx, fy = float(x), float(y)
  if math.isnan(fx) or math.isnan(fy) or math.isinf(fx) or fy == 0.0:
    return mpmath.nan
  if math.isinf(fy) or fx == 0.0:
    return x
  ax, ay = abs(fx), abs(fy)
  if ax < ay:
    return x
  mx, ex = math.frexp(ax)
  my, ey = math.frexp(ay)
  # Scale mantissas in [0.5, 1.0) by 2^53 to exact integers in [2^52, 2^53).
  # Since ax >= ay and both mantissas lie in [2^52, 2^53), ex >= ey always.
  ix = int(math.ldexp(mx, 53))
  iy = int(math.ldexp(my, 53))
  rem_int = (ix * pow(2, ex - ey, iy)) % iy
  res = mpmath.ldexp(rem_int, ey - 53)
  return -res if fx < 0.0 else res


def _rem_interesting_points(dtype):
  finfo = jnp.finfo(dtype)
  eps = float(finfo.eps)
  tiny = float(finfo.tiny)
  fmax = float(finfo.max)
  max_exp = finfo.maxexp - 1
  min_exp = finfo.minexp

  # Divisor mantissas at binade boundaries (1 + eps, 2 - eps) and repeating
  # binary fractions.
  divisors = [1.0 + eps, 2.0 - eps, 0.1, 0.3, 3.0, 7.0]
  points: list[float | tuple[float, float]] = [
      *(sign * v for v in divisors for sign in (-1, 1)),
      (10.0, 0.1),
      (-10.0, 0.1),
      (10.0, -0.1),
      (-10.0, -0.1),
      (1.5 * tiny, tiny),
      (-1.5 * tiny, tiny),
      (fmax, tiny),
      (-fmax, tiny),
      (fmax, 3.0),
      (fmax, 1.0 + eps),
      (fmax, 2.0 - eps),
  ]

  # Exponent gaps d = ex - ey hitting the radix-16 modular reduction shift
  # transitions in TPU's EmitAccurateRemF32 (m = d // 4 around 20, 24, 32, 52).
  for d in (23, 24, 80, 96, 128, 208, 252):
    ex = min(max_exp, d // 2)
    ey = max(min_exp, ex - d)
    if ex - ey == d:
      x_scale = math.ldexp(1.0, ex)
      y_scale = math.ldexp(1.0, ey)
      for mx in (1.0, 1.5, 2.0 - eps):
        for my in (1.0 + eps, 1.5, 2.0 - eps):
          if mx * x_scale <= fmax and my * y_scale >= tiny:
            points.append((mx * x_scale, my * y_scale))
            points.append((-mx * x_scale, my * y_scale))
  return points


@jtu.thread_unsafe_test_class()
class RemTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_rem_highest_accuracy(self, dtype):
    if jtu.device_under_test() == "tpu":
      jax_fn = jax.jit(
          lax.rem, compiler_options={"xla_tpu_enable_accurate_rem": "true"}
      )
    else:
      jax_fn = lax.rem

    bounds = []
    input_ftz = [
        (["cpu", "gpu"], False),
    ]
    output_ftz = [
        (["cpu", "gpu"], False),
    ]
    util.check_nary_precision(
        self,
        jax_fn,
        np.fmod,
        _mpmath_fmod,
        dtype,
        nargs=2,
        bounds=bounds,
        input_ftz=input_ftz,
        output_ftz=output_ftz,
        interesting_points=_rem_interesting_points(dtype),
    )


util.register_benchmark(lax.rem, nargs=2)


if __name__ == "__main__":
  util.main()
