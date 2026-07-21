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


import itertools
import math
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as mosaic_interpret
from jax._src.pallas.mosaic_gpu.interpret.params import InterpretGPUParams as InterpretParams
import jax.experimental.pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


def _get_all_subsets(xs):
  return list(
      itertools.chain.from_iterable(
          itertools.combinations(xs, r) for r in range(len(xs) + 1)
      )
  )


@jtu.thread_unsafe_test_class()
class GpuPallasInterpretClusterBarrierTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("CPU-only test")

  @jtu.parameterized.product(with_race=[True, False])
  def test_cluster_barrier_peer_communication_through_gmem(self, with_race):
    x = jnp.arange(32, dtype=jnp.int32).reshape(2, 16)

    def _kernel(in_gmem, scratch_gmem, out_gmem, cluster_barrier):
      cid = jax.lax.axis_index("c")
      # Each block writes its slice of input to a shared GMEM scratch buffer.
      scratch_gmem[cid, :] = in_gmem[cid, :]

      # Arrive at the cluster barrier to signal that GMEM scratch is populated.
      plgpu.barrier_arrive(cluster_barrier)
      if not with_race:
        plgpu.barrier_wait(cluster_barrier)

      # Read from the peer block's slice in GMEM scratch.
      out_gmem[cid, :] = scratch_gmem[1 - cid, :]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            cluster_barrier=plgpu.ClusterBarrier(
                collective_axes=("c",), num_arrivals=1
            ),
        ),
        cluster=(2,),
        cluster_names=("c",),
    )

    scratch_init = jnp.zeros_like(x)
    y = kernel(x, scratch_init)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      expected = jnp.flip(x, axis=0)
      np.testing.assert_array_equal(y, expected)

  @jtu.parameterized.product(
      num_barriers=[1, 2, 3, 4, 5, 6, 7, 8],
      with_race_in_round=[None, 0, 1, 2, 3, 4, 5, 6, 7],
  )
  def test_cluster_barrier_multiple_barriers(
      self, num_barriers, with_race_in_round
  ):
    if with_race_in_round is not None and with_race_in_round >= num_barriers:
      self.skipTest(
          f"with_race_in_round(={with_race_in_round}) >="
          f" num_barriers(={num_barriers})"
      )

    x = jnp.arange(num_barriers * 16, dtype=jnp.int32).reshape(num_barriers, 16)

    def _kernel(in_gmem, scratch_gmem, out_gmem, cluster_barrier):
      cid = jax.lax.axis_index("c")
      scratch_gmem[0, cid, :] = in_gmem[cid, :]

      for r in range(num_barriers):
        plgpu.barrier_arrive(cluster_barrier.at[r])
        if with_race_in_round != r:
          plgpu.barrier_wait(cluster_barrier.at[r])

        # Move to neighbor (cid + 1) % num_barriers and add 1
        neighbor = (cid + 1) % num_barriers
        scratch_gmem[r + 1, cid, :] = scratch_gmem[r, neighbor, :] + 1

      out_gmem[cid, :] = scratch_gmem[num_barriers, cid, :]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            cluster_barrier=plgpu.ClusterBarrier(
                collective_axes=("c",), num_arrivals=1, num_barriers=num_barriers
            ),
        ),
        cluster=(num_barriers,),
        cluster_names=("c",),
    )

    scratch_init = jnp.zeros((num_barriers + 1, num_barriers, 16), dtype=jnp.int32)
    y = kernel(x, scratch_init)
    if with_race_in_round is not None and num_barriers > 1:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      expected = x + num_barriers
      np.testing.assert_array_equal(y, expected)

  def test_cluster_barrier_differ_at_most_one_collective_axis(self):
    x = jnp.arange(64, dtype=jnp.int32).reshape(2, 2, 16)
    s = jnp.zeros((2, 2, 16), dtype=jnp.int32)

    def _kernel(in_gmem, scratch_gmem, out_gmem, cluster_barrier):
      c0 = jax.lax.axis_index("c0")
      c1 = jax.lax.axis_index("c1")

      is_block_0_0 = (c0 == 0) & (c1 == 0)
      @pl.when(is_block_0_0)
      def _():
        scratch_gmem[0, 0, :] = in_gmem[0, 0, :]

      @pl.when(~is_block_0_0)
      def _():
        scratch_gmem[c0, c1, :] = in_gmem[c0, c1, :]

      # Synchronize using the cluster barrier.
      plgpu.barrier_arrive(cluster_barrier)
      plgpu.barrier_wait(cluster_barrier)

      # Block (1, 1) should not be synchronized with block (0, 0), because they
      # differ in more than one collective axis. So reading block (0, 0)'s write
      # from block (1, 1) is a race.
      is_block_1_1 = (c0 == 1) & (c1 == 1)
      @pl.when(is_block_1_1)
      def _():
        out_gmem[c0, c1, :] = scratch_gmem[0, 0, :]

      @pl.when(~is_block_1_1)
      def _():
        out_gmem[c0, c1, :] = scratch_gmem[c0, c1, :]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct.like(x),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            cluster_barrier=plgpu.ClusterBarrier(
                collective_axes=("c0", "c1"), num_arrivals=1
            ),
        ),
        cluster=(2, 2),
        cluster_names=("c0", "c1"),
    )

    _ = kernel(x, s)
    self.assertTrue(mosaic_interpret.get_races().races_found)

  @jtu.parameterized.product(
      with_race=[True, False],
      cluster_dict=[
          dict(),
          dict(c0=2),
          dict(c0=2, c1=3),
          dict(c0=2, c1=3, c2=2),
      ],
      thread_dict=[
          None,
          dict(t=2),
      ],
      collective_axes=_get_all_subsets(("c0", "c1", "c2")),
  )
  def test_cluster_barrier_complex_mesh(
      self, with_race, cluster_dict, thread_dict, collective_axes
  ):
    if with_race and not (collective_axes or thread_dict):
      self.skipTest(
          "Not expecting races in test with no collective axes and only a"
          " single thread"
      )

    cluster_names = tuple(cluster_dict.keys())
    cluster_shape = tuple(cluster_dict.values())
    if any(axis not in cluster_names for axis in collective_axes):
      self.skipTest(
          f"Collective axes {collective_axes} not a subset of cluster axes"
          f" {cluster_names}"
      )

    mesh_kwargs = dict(
        cluster=cluster_shape,
        cluster_names=cluster_names,
    )
    if thread_dict is not None:
      ((thread_name, num_threads),) = thread_dict.items()
      mesh_kwargs["num_threads"] = num_threads
      mesh_kwargs["thread_name"] = thread_name
    else:
      num_threads = 1

    in_shape = cluster_shape + (num_threads,)
    x = jnp.arange(math.prod(in_shape), dtype=jnp.int32).reshape(in_shape)
    s = jnp.zeros(in_shape, dtype=jnp.int32)

    def _kernel(in_gmem, scratch_gmem, out_gmem, cluster_barrier):
      cluster_idx_tuple = tuple(
          jax.lax.axis_index(name) for name in cluster_names
      )
      if thread_dict is not None:
        thread_idx_tuple = (jax.lax.axis_index(thread_name),)
      else:
        thread_idx_tuple = (0,)

      my_idx_tuple = cluster_idx_tuple + thread_idx_tuple
      # Each thread in the cluster copies one entry from the input GMEM into
      # the (shared) scratch GMEM.
      scratch_gmem[my_idx_tuple] = in_gmem[my_idx_tuple]

      plgpu.barrier_arrive(cluster_barrier)
      if not with_race:
        plgpu.barrier_wait(cluster_barrier)

      read_slice = tuple(
          slice(None) if name in collective_axes else jax.lax.axis_index(name)
          for name in cluster_names
      ) + (slice(None),)
      # Each thread in the cluster copies a slice along the full
      # `collective_axes` from the scratch GMEM into the output GMEM. When not
      # using the `cluster_barrier` to synchronize the threads along the
      # `collective_axes`, this read of `scratch_gmem` races with the write
      # above (from `in_gmem`).
      out_gmem[my_idx_tuple] = scratch_gmem[read_slice]

    read_shape = tuple(
        dim for name, dim in cluster_dict.items() if name in collective_axes
    ) + (num_threads,)
    out_shape = in_shape + read_shape

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct(out_shape, jnp.int32),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            cluster_barrier=plgpu.ClusterBarrier(
                collective_axes=collective_axes, num_arrivals=num_threads
            ),
        ),
        **mesh_kwargs,
    )

    y = kernel(x, s)
    expect_race = with_race or (len(collective_axes) > 1)
    if expect_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      expected = np.zeros(out_shape, dtype=np.int32)
      for my_idx in itertools.product(*(range(d) for d in in_shape)):
        read_slice = tuple(
            slice(None) if name in collective_axes else my_idx[k]
            for k, name in enumerate(cluster_names)
        ) + (slice(None),)
        expected[my_idx] = x[read_slice]
      np.testing.assert_array_equal(y, expected)

  @jtu.parameterized.product(with_race=[True, False])
  def test_cluster_barrier_multidimensional_1d(self, with_race):
    shape = (2,)
    x = jnp.arange(32, dtype=jnp.int32).reshape(2, 16)
    out_shape = (2, 2, 16)

    def _kernel(in_gmem, scratch_gmem, out_gmem, cluster_barrier):
      cid = jax.lax.axis_index("c")
      for i in range(2):
        scratch_gmem[i, cid, :] = in_gmem[cid, :]
        plgpu.barrier_arrive(cluster_barrier.at[i])
        if not with_race:
          plgpu.barrier_wait(cluster_barrier.at[i])
        out_gmem[i, cid, :] = scratch_gmem[i, 1 - cid, :]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct(out_shape, jnp.int32),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            cluster_barrier=plgpu.ClusterBarrier(
                collective_axes=("c",), num_arrivals=1, num_barriers=shape
            ),
        ),
        cluster=(2,),
        cluster_names=("c",),
    )

    scratch_init = jnp.zeros((2, 2, 16), dtype=jnp.int32)
    y = kernel(x, scratch_init)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      expected = jnp.broadcast_to(
          jnp.flip(x, axis=0)[jnp.newaxis, :, :], out_shape
      )
      np.testing.assert_array_equal(y, expected)

  @jtu.parameterized.product(with_race=[True, False])
  def test_cluster_barrier_multidimensional_2d(self, with_race):
    shape = (2, 3)
    x = jnp.arange(32, dtype=jnp.int32).reshape(2, 16)
    out_shape = (2, 3, 2, 16)

    def _kernel(in_gmem, scratch_gmem, out_gmem, cluster_barrier):
      cid = jax.lax.axis_index("c")
      for i in range(2):
        for j in range(3):
          scratch_gmem[i, j, cid, :] = in_gmem[cid, :]
          plgpu.barrier_arrive(cluster_barrier.at[i, j])
          if not with_race:
            plgpu.barrier_wait(cluster_barrier.at[i, j])
          out_gmem[i, j, cid, :] = scratch_gmem[i, j, 1 - cid, :]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct(out_shape, jnp.int32),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            cluster_barrier=plgpu.ClusterBarrier(
                collective_axes=("c",), num_arrivals=1, num_barriers=shape
            ),
        ),
        cluster=(2,),
        cluster_names=("c",),
    )

    scratch_init = jnp.zeros((2, 3, 2, 16), dtype=jnp.int32)
    y = kernel(x, scratch_init)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      expected = jnp.broadcast_to(
          jnp.flip(x, axis=0)[jnp.newaxis, jnp.newaxis, :, :], out_shape
      )
      np.testing.assert_array_equal(y, expected)

  @jtu.parameterized.product(with_race=[True, False])
  def test_cluster_barrier_multidimensional_3d(self, with_race):
    shape = (2, 1, 3)
    x = jnp.arange(32, dtype=jnp.int32).reshape(2, 16)
    out_shape = (2, 1, 3, 2, 16)

    def _kernel(in_gmem, scratch_gmem, out_gmem, cluster_barrier):
      cid = jax.lax.axis_index("c")
      for i in range(2):
        for j in range(1):
          for k in range(3):
            scratch_gmem[i, j, k, cid, :] = in_gmem[cid, :]
            plgpu.barrier_arrive(cluster_barrier.at[i, j, k])
            if not with_race:
              plgpu.barrier_wait(cluster_barrier.at[i, j, k])
            out_gmem[i, j, k, cid, :] = scratch_gmem[i, j, k, 1 - cid, :]

    kernel = plgpu.kernel(
        _kernel,
        out_type=jax.ShapeDtypeStruct(out_shape, jnp.int32),
        interpret=InterpretParams(detect_races=True),
        scratch_types=dict(
            cluster_barrier=plgpu.ClusterBarrier(
                collective_axes=("c",), num_arrivals=1, num_barriers=shape
            ),
        ),
        cluster=(2,),
        cluster_names=("c",),
    )

    scratch_init = jnp.zeros((2, 1, 3, 2, 16), dtype=jnp.int32)
    y = kernel(x, scratch_init)
    if with_race:
      self.assertTrue(mosaic_interpret.get_races().races_found)
    else:
      self.assertFalse(mosaic_interpret.get_races().races_found)
      expected = jnp.broadcast_to(
          jnp.flip(x, axis=0)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :],
          out_shape,
      )
      np.testing.assert_array_equal(y, expected)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
