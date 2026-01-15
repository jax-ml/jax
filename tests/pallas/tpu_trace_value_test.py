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
"""Minimal test for pltpu.trace_value primitive."""

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def simple_kernel_with_trace_value(x_ref, s_ref, o_ref):
  """Simple kernel that emits trace metrics."""
  # Emit a constant to xprof trace
  pltpu.trace_value("constant_value", jnp.float32(42.42))
  scale = s_ref[0]

  z = x_ref[...] + jnp.float32(48.0) + scale.astype(jnp.float32).reshape((1, 1))
  pltpu.trace_value("scale_value", scale)
  o_ref[...] = z


class TraceMetricTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu():
      self.skipTest("trace_value only supported on TPU.")

  def test_simple_trace_metric(self):
    """Test that trace_metric compiles and runs without error."""
    x = jnp.ones((8, 128), dtype=jnp.float32)

    s = jax.random.randint(jax.random.key(0), (1,), minval=0, maxval=100)

    result = pl.pallas_call(
        simple_kernel_with_trace_value,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        in_specs=[
            pl.BlockSpec((8, 128), memory_space=pltpu.VMEM),
            pl.BlockSpec((1,), memory_space=pltpu.SMEM),
        ],
        out_specs=pl.BlockSpec((8, 128), memory_space=pltpu.VMEM),
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
        name="trace_metric_test",
    )(x, s)

    # Just verify the kernel runs and produces correct output
    # TODO(amuzio): Verify the kernel runs and includes the vtrace in an actual
    # xprof.
    self.assertEqual(result.shape, (8, 128))
    self.assertTrue(
        jnp.allclose(result, x + 48.0 + s.astype(jnp.float32).reshape((1, 1)))
    )


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
