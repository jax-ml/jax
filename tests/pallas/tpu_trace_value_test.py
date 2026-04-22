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
from jax.experimental.pallas import tpu_sc as plsc
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
    self.assertEqual(result.shape, (8, 128))
    self.assertTrue(
        jnp.allclose(result, x + 48.0 + s.astype(jnp.float32).reshape((1, 1)))
    )


class SparseCoreTraceValueTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu(5, "p") and not jtu.is_device_tpu_at_least(6):
      self.skipTest("SparseCore only supported on TPU v5p+")

  def test_trace_value(self):
    nl = plsc.get_sparse_core_info().num_lanes
    x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore", num_cores=1
    )

    @pl.kernel(
        out_type=x,
        mesh=mesh,
        scratch_types=(pltpu.VMEM(x.shape, x.dtype),),
        compiler_params=pltpu.CompilerParams(),
    )
    def kernel(x_hbm_ref, o_hbm_ref, tmp_ref):
      pltpu.sync_copy(x_hbm_ref, tmp_ref)
      pltpu.trace_value("sc_trace_value", tmp_ref[1, :nl][1])
      pltpu.sync_copy(tmp_ref, o_hbm_ref)

    result = kernel(x)
    self.assertArraysEqual(result, x)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
