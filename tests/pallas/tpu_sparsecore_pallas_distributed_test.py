# Copyright 2025 The JAX Authors.
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
"""Tests for Pallas on SparseCore with multiple devices."""
import functools
import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


class PallasCallRemoteDMATest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest('Only >=2 devices are supported.')
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('SparseCore only supported on TPU v5+')

  @parameterized.product(direction=['left', 'right'], num_devices=[2, None])
  def test_collective_permute_1d(self, direction, num_devices):
    shape = (8, 128)

    # Implements a very simple collective permute.
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(shape, jnp.int32),
        mesh=plsc.ScalarSubcoreMesh(axis_name='core', num_cores=1),
        scratch_types=(
            pltpu.SemaphoreType.REGULAR,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ),
    )
    def kernel(x_ref, y_ref, ready_sem, send_sem, recv_sem):

      my_id = lax.axis_index('x')
      axis_size = lax.axis_size('x')
      if direction == 'right':
        neighbor = lax.rem(my_id + 1, axis_size)
      else:
        neighbor = lax.rem(my_id + axis_size - 1, axis_size)
      pl.semaphore_signal(ready_sem, device_id=neighbor)
      pl.semaphore_wait(ready_sem)
      pltpu.async_remote_copy(
          x_ref, y_ref, send_sem, recv_sem, device_id=neighbor
      ).wait()

    num_devices = num_devices or jax.device_count()
    x = jnp.arange(num_devices * math.prod(shape), dtype=jnp.int32).reshape(
        (-1, shape[-1])
    )
    device_mesh = mesh_utils.create_device_mesh(
        (num_devices,), jax.devices()[:num_devices]
    )
    mesh = jax.sharding.Mesh(device_mesh, ['x'])
    f = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=jax.P('x'),
            out_specs=jax.P('x'),
            check_vma=False,
        )
    )
    if direction == 'right':
      expected = jnp.concatenate([x[-8:], x[:-8]])
    else:
      expected = jnp.concatenate([x[8:], x[:8]])
    np.testing.assert_allclose(f(x), expected)

  @parameterized.product(direction=['left', 'right'])
  def test_collective_permute_2d(self, direction):
    shape = (8, 128)

    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(shape, jnp.int32),
        mesh=plsc.ScalarSubcoreMesh(axis_name='core', num_cores=1),
        scratch_types=(
            pltpu.SemaphoreType.REGULAR,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ),
    )
    def kernel(x_ref, y_ref, ready_sem, send_sem, recv_sem):
      my_id = lax.axis_index('x')
      my_other_id = lax.axis_index('y')
      axis_size = lax.axis_size('x')
      if direction == 'right':
        neighbor = lax.rem(my_id + 1, axis_size)
      else:
        neighbor = lax.rem(my_id + axis_size - 1, axis_size)
      pl.semaphore_signal(ready_sem, device_id=(my_other_id, neighbor))
      pl.semaphore_wait(ready_sem)
      pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, device_id=(my_other_id, neighbor)
        ).wait()

    axis_size = jax.device_count() // 2
    x = jnp.arange(axis_size * 8 * 128).reshape((axis_size * 8, 128))

    device_mesh = mesh_utils.create_device_mesh((2, axis_size), jax.devices())
    mesh = jax.sharding.Mesh(device_mesh, ['y', 'x'])
    y = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=jax.P('x', None),
            out_specs=jax.P('x', None),
            check_vma=False,
        )
    )(x)
    if direction == 'right':
      expected = jnp.concatenate([x[-8:], x[:-8]])
    else:
      expected = jnp.concatenate([x[8:], x[:8]])
    np.testing.assert_allclose(y, expected)


class DistributedMpmdTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest('Only >=2 devices are supported.')
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('SparseCore only supported on TPU v5+')
    if jtu.is_device_tpu(7, "x") and not jtu.is_cloud_tpu_at_least(2026, 6, 1):
      self.skipTest("Skipping on TPU 7x with libtpu < 0.0.41")

  def test_mpmd_reduce_scatter(self):
    P = jax.P

    mesh = jax.sharding.Mesh(jax.devices(), axis_names='x')
    if mesh.size <= 1:
      self.skipTest('Need at least 2 devices.')

    sc_info = pltpu.get_tpu_info().sparse_core
    num_cores = sc_info.num_cores

    @jax.shard_map(
        mesh=mesh, in_specs=P('x'), out_specs=P('x'), check_vma=False
    )
    @jax.jit
    def f(x):
      axis_size = jax.lax.axis_size('x')
      size_per_device = x.shape[0] // axis_size
      size_per_core = size_per_device // num_cores

      def _barrier(my_sem, scs_sem, tec_sem):
        """Cross-device barrier: each body signals both sems and waits on own."""
        num_subcores = jax.lax.axis_size('subcore')
        for d in range(axis_size):
          for c in range(num_cores):
            pl.semaphore_signal(scs_sem, device_id={'x': d, 'core': c})
            for s in range(num_subcores):
              pl.semaphore_signal(
                  tec_sem, device_id={'x': d, 'core': c, 'subcore': s}
              )
        total_scs = axis_size * num_cores
        total_tec = axis_size * num_cores * num_subcores
        pl.semaphore_wait(my_sem, total_scs + total_tec)

      # SCS handles cross-chip transfers.
      def go_scs(
          x_ref,
          recv_ref,
          *,
          vmem_shd,
          tec_to_scs,
          scs_to_tec,
          barrier_scs_sem,
          barrier_tec_sem,
      ):
        del vmem_shd  # Used only by TEC.
        _barrier(barrier_scs_sem, barrier_scs_sem, barrier_tec_sem)

        core_idx = jax.lax.axis_index('core')

        @functools.partial(
            pl.run_scoped,
            send_sem_ref=pltpu.SemaphoreType.DMA,
            recv_sem_ref=pltpu.SemaphoreType.DMA,
        )
        def _(send_sem_ref, recv_sem_ref):
          my_id = jax.lax.axis_index('x')

          @pl.loop(1, axis_size)
          def loop_body(i):
            pltpu.async_remote_copy(
                pl.select_ref(
                    (i == 1).astype(jnp.uint32),
                    recv_ref.at[core_idx, (i - 1) % 2],
                    x_ref.at[core_idx, (my_id - i) % axis_size],
                ),
                recv_ref.at[core_idx, i % 2],
                send_sem_ref,
                recv_sem_ref,
                device_id={'x': (1 + my_id) % axis_size},
            ).wait()
            pl.semaphore_signal(scs_to_tec, device_id={'subcore': 0})
            # Wait until TEC has added next iteration into recv_ref.
            pl.semaphore_wait(tec_to_scs, value=1)

      # TEC handles within-core stream-add ops.
      def go_tec(
          x_ref,
          recv_ref,
          *,
          vmem_shd,
          tec_to_scs,
          scs_to_tec,
          barrier_scs_sem,
          barrier_tec_sem,
      ):
        _barrier(barrier_tec_sem, barrier_scs_sem, barrier_tec_sem)

        core_idx = jax.lax.axis_index('core')
        my_id = jax.lax.axis_index('x')
        vmem = jax.empty_ref(
            jax.core.ShapedArray((size_per_core,), x.dtype),
            memory_space=pltpu.VMEM,
        )
        iota = jax.empty_ref(
            jax.core.ShapedArray(size_per_core, jnp.int32),
            memory_space=pltpu.VMEM,
        )
        assert (sc := pltpu.get_tpu_info().sparse_core) is not None
        @plsc.parallel_loop(0, size_per_core, step=sc.num_lanes)
        def fill_iota(i):
          iota[pl.ds(i, sc.num_lanes)] = i + jnp.arange(sc.num_lanes)

        @pl.loop(1, axis_size)
        def loop_body(i):
          # While waiting, DMA next iteration's shard-local value into vmem.
          pltpu.sync_copy(x_ref.at[core_idx, (my_id - i - 1) % axis_size], vmem)
          # Block until SCS has received incoming sum part.
          pl.semaphore_wait(scs_to_tec, value=1)
          # TODO(b/514769335): Direct DMA from recv_ref to vmem once supported.
          # Work around the halt by HBM => VMEM_SHARED; VMEM_SHARED +> VMEM
          pltpu.sync_copy(recv_ref.at[core_idx, i % 2], vmem_shd)
          # TODO(bchetioui): Replace this with linear stream add.
          pltpu.sync_copy(vmem_shd.at[iota], vmem, add=True)
          pltpu.sync_copy(vmem, recv_ref.at[core_idx, i % 2])
          pl.semaphore_signal(tec_to_scs)

      scs_mesh = plsc.ScalarSubcoreMesh(axis_name='core', num_cores=num_cores)
      tec_mesh = plsc.VectorSubcoreMesh(
          core_axis_name='core',
          subcore_axis_name='subcore',
          num_cores=num_cores,
          num_subcores=1,
      )

      result = pl.kernel(
          body=[go_scs, go_tec],
          mesh=[scs_mesh, tec_mesh],
          out_type=jax.ShapeDtypeStruct((num_cores, 2, size_per_core), x.dtype),
          scratch_types=dict(
              vmem_shd=pltpu.VMEM_SHARED((size_per_core,), x.dtype),
              tec_to_scs=pltpu.SemaphoreType.REGULAR(()) @ scs_mesh,
              scs_to_tec=pltpu.SemaphoreType.REGULAR(()) @ tec_mesh,
              barrier_scs_sem=pltpu.SemaphoreType.REGULAR(()) @ scs_mesh,
              barrier_tec_sem=pltpu.SemaphoreType.REGULAR(()) @ tec_mesh,
          ),
          compiler_params=pltpu.CompilerParams(
              # TODO(ivyzheng): Remove this when the layout pass flag is gone.
              needs_layout_passes=False,
          ),
          debug=True,
      )(x.reshape(num_cores, mesh.size, size_per_core))
      # result shape: (num_cores, 2, size_per_core)
      # Select buffer index 1 from each core, then flatten.
      actual = result[:, 1].reshape(-1)
      expected = jax.lax.psum_scatter(
          x.reshape(num_cores, mesh.size, size_per_core),
          'x',
          scatter_dimension=1,
      ).reshape(-1)
      return actual, expected

    x = jax.device_put(
        jnp.arange(8 * 8 * 128), jax.sharding.NamedSharding(mesh, P('x'))
    )
    y, y_ = jax.block_until_ready(f(x))
    np.testing.assert_array_equal(y, y_)


if __name__ == '__main__':
  absltest.main()
