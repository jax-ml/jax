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
"""Tests for Pallas async SC-TC kernels."""

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.pallas import mpmd
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


class TpuAsyncScTcPrefetchVmemTest(jtu.JaxTestCase):

  def setUp(self):
    if jtu.is_cloud_tpu():
      self.skipTest("Not yet supported on Cloud TPU.")
    if not jtu.is_device_tpu(5, "p") and not jtu.is_device_tpu_at_least(6):
      self.skipTest("SparseCore only supported on TPU v5p+")
    super().setUp()

  def test_async_sc_tc_prefetch_vmem(self):
    s_mesh = plsc.ScalarSubcoreMesh(
        axis_name="core",
        num_cores=1,
    )
    tc_mesh = pltpu.create_tensorcore_mesh(axis_name="tc", num_cores=1)

    def scalar_subcore_fn(x_ref, _, out_tc_vmem_ref, tc_sem, sem):
      pltpu.async_remote_copy(
          x_ref, out_tc_vmem_ref, sem, tc_sem, device_id={"tc": 0}
      ).wait_send()

    def tc_fn(x_ref, out_tc_vmem_ref, tc_sem, _):
      pltpu.make_async_copy(x_ref, out_tc_vmem_ref, tc_sem).wait()
      out_tc_vmem_ref[...] += 1

    @jax.jit
    def f(x):
      x, out, sem = mpmd._mpmd_map(
          [(s_mesh, scalar_subcore_fn)],
          out_types=(
              jax.typeof(x),
              pltpu.VMEM(x.shape, x.dtype) @ tc_mesh,
              pltpu.SemaphoreType.DMA @ tc_mesh,
          ),
          scratch_types=[pltpu.SemaphoreType.DMA],
          compiler_params=pltpu.CompilerParams(
              use_tc_tiling_on_sc=True,
          ),
          input_output_aliases={0: 0},
      )(x)
      out = mpmd._mpmd_map(
          [(tc_mesh, tc_fn)],
          out_types=jax.typeof(out),
          input_output_aliases={1: 0},
      )(x, out, sem)
      return out

    x = jnp.arange(8 * 128).reshape(8, 128)
    out = f(x)
    np.testing.assert_array_equal(out, x + 1)


if __name__ == "__main__":
  absltest.main()
