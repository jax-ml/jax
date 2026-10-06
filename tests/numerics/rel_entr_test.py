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

"""Precision tests for relative entropy."""

from absl.testing import parameterized
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
import scipy.special

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]
TPU_EUPV1 = ["tpu_v2", "tpu_v3", "tpu_v4", "tpu_v4i", "tpu_v5e"]


def _mpmath_rel_entr(p, q):
  if mpmath.isnan(p) or mpmath.isnan(q):
    return mpmath.nan
  pf, qf = float(p), float(q)
  # p == 0 covers signed zero. q >= 0 is true for both +0 and -0.
  if pf == 0.0 and qf >= 0.0:
    return mpmath.mpf(0)
  if pf < 0.0 or qf <= 0.0:
    return mpmath.inf
  if mpmath.isinf(pf) and mpmath.isinf(qf):
    return mpmath.nan
  return p * mpmath.log(p / q)


@jtu.thread_unsafe_test_class()
class RelEntrTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_rel_entr_accuracy(self, dtype):
    # CPU bounds are the measured max ULP, rounded up to a multiple of 0.5.
    # The float32 spike is p * log1p((p - q) / q) flushing to 0 when p - q is
    # subnormal. TPU float32 log1p is thousands of ULPs, so that bound is wider.
    bounds = [
        ("cpu", {bf16: 19.5, f16: 2.5, f32: 78205.5, f64: 54.0}),
        ("gpu", {bf16: 19.5, f16: 2.5, f32: 78205.5, f64: 54.0}),
        (TPU_EUPV1, {bf16: 19.5, f16: 2.5, f32: 1.0e6}),
        ("tpu_v5p", {bf16: 19.5, f16: 2.5, f32: 1.0e6}),
        (["tpu_v6e", "tpu_7x"], {bf16: 19.5, f16: 2.5, f32: 1.0e6}),
    ]
    input_ftz = [
        ("gpu", False),
    ]
    util.check_nary_precision(
        self,
        jsp.special.rel_entr,
        scipy.special.rel_entr,
        _mpmath_rel_entr,
        dtype,
        nargs=2,
        bounds=bounds,
        input_ftz=input_ftz,
    )


util.register_benchmark(jsp.special.rel_entr, nargs=2)


if __name__ == "__main__":
  util.main()
