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

"""Tests for distributed pallas GPU operations."""

import functools
import multiprocessing
import os
import portpicker
import tempfile
from absl.testing import absltest
import jax
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec

partial = functools.partial

def launch_multiprocess_test(test_case, test_fn, num_gpus):
    def worker(port, num_gpus, gpu_idx, test_fn):
      jax.distributed.initialize(coordinator_address=f'localhost:{port}',
                                 num_processes=num_gpus,
                                 process_id=gpu_idx,
                                 local_device_ids=[gpu_idx])
      test_fn(test_case)

    port = portpicker.pick_unused_port()
    procs = []
    for gpu_idx in range(num_gpus):
      procs.append(multiprocessing.Process(target=worker,
                                           args=(port,
                                                 num_gpus,
                                                 gpu_idx,
                                                 test_fn)))

    for gpu_idx in range(num_gpus):
      procs[gpu_idx].start()

    for gpu_idx in range(num_gpus):
      procs[gpu_idx].join()

    for gpu_idx in range(num_gpus):
      test_case.assertEqual(procs[gpu_idx].exitcode, 0)


class PallasCallRemoteDMATest(jtu.JaxTestCase):

  def test_basic_remote_dma(self):
    def _test_basic_remote_dma(self):
      # Implements very simple collective permute
      def kernel(x_ref, ready_sem_alias, recv_sem_alias,
                 y_ref, ready_sem, recv_sem):
        del ready_sem_alias, recv_sem_alias
        dev_id = pl.device_id()
        other_dev_id = 1 - dev_id
        pl.semaphore_signal(ready_sem.at[0], device_id=other_dev_id,
                            device_id_type=pl.DeviceIdType.LOGICAL)
        pl.semaphore_wait(ready_sem.at[0])
        copy_done = pl.async_remote_copy(
            x_ref, y_ref, None, recv_sem.at[0], other_dev_id,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )
        copy_done.wait_recv()

      x = jnp.arange(2 * 8 * 128.0).reshape((2 * 8, 128))
      ready_sem = jnp.zeros((1,), dtype=jnp.int32)
      recv_sem = jnp.zeros((1,), dtype=jnp.int32)
      def body(x, ready_sem, recv_sem):
        return pl.pallas_call(
            kernel,
            in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM),
                      pl.BlockSpec(memory_space=plgpu.GMEM),
                      pl.BlockSpec(memory_space=plgpu.GMEM)],
            out_specs=[pl.BlockSpec(memory_space=plgpu.GMEM),
                       pl.BlockSpec(memory_space=plgpu.GMEM),
                       pl.BlockSpec(memory_space=plgpu.GMEM)],
            out_shape=[jax.ShapeDtypeStruct((8, 128), jnp.float32),
                       jax.ShapeDtypeStruct((1,), jnp.int32),
                       jax.ShapeDtypeStruct((1,), jnp.int32)],
            input_output_aliases={1: 1, 2: 2}
        )(x, ready_sem, recv_sem)

      devices = jax.devices()[:2]
      mesh = jax.sharding.Mesh(devices, ['x'])
      y, _, _ = jax.jit(
                shard_map.shard_map(
                  body, mesh,
                  in_specs=(P('x'), P(None), P(None)),
                  out_specs=(P('x'), P(None), P(None)),
                  check_rep=False
                )
      )(x, ready_sem, recv_sem)
      expected = x[8:] if jax.process_index() == 0 else x[:8]
      np.testing.assert_allclose(y.addressable_shards[0].data, expected)

    launch_multiprocess_test(self, _test_basic_remote_dma, 2)

  def test_barrier_semaphore(self):
    def _test_barrier_semaphore(self):
      def kernel(x_ref, barrier_sem_alias, recv_sem_alias,
                 y_ref, barrier_sem, recv_sem):
        del barrier_sem_alias, recv_sem_alias
        my_id = pl.device_id()
        num_devices = lax.psum(1, 'x')
        neighbor = lax.rem(my_id + 1, num_devices)
        pl.semaphore_signal(barrier_sem.at[0], device_id=neighbor,
                            device_id_type=pl.DeviceIdType.LOGICAL)
        pl.semaphore_wait(barrier_sem.at[0])
        pl.async_remote_copy(
            x_ref, y_ref, None, recv_sem.at[0], device_id=neighbor,
            device_id_type=pl.DeviceIdType.LOGICAL
        ).wait()

      num_devices = jax.device_count()
      x = jnp.arange(num_devices * 8 * 128).reshape((num_devices * 8, 128))
      barrier_sem = jnp.zeros((1,), dtype=jnp.int32)
      recv_sem = jnp.zeros((1,), dtype=jnp.int32)

      def body(x, barrier_sem, recv_sem):
        return pl.pallas_call(
            kernel,
            in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM),
                      pl.BlockSpec(memory_space=plgpu.GMEM),
                      pl.BlockSpec(memory_space=plgpu.GMEM)],
            out_specs=[pl.BlockSpec(memory_space=plgpu.GMEM),
                       pl.BlockSpec(memory_space=plgpu.GMEM),
                       pl.BlockSpec(memory_space=plgpu.GMEM)],
            out_shape=[jax.ShapeDtypeStruct((8, 128), jnp.int32),
                       jax.ShapeDtypeStruct((1,), jnp.int32),
                       jax.ShapeDtypeStruct((1,), jnp.int32)],
            input_output_aliases={1: 1, 2: 2},
            compiler_params=plgpu.GPUCompilerParams(use_custom_barrier=True),
        )(x, barrier_sem, recv_sem)

      device_mesh = mesh_utils.create_device_mesh(
          (jax.device_count(),), jax.devices())
      mesh = jax.sharding.Mesh(device_mesh, ['x'])
      y, _, _ = jax.jit(
                shard_map.shard_map(
                    body, mesh,
                    in_specs=(P('x'), P(None), P(None)),
                    out_specs=(P('x'), P(None), P(None)),
                    check_rep=False
                )
      )(x, barrier_sem, recv_sem)
      expected = x[8:] if jax.process_index() == 0 else x[:8]
      np.testing.assert_allclose(y.addressable_shards[0].data, expected)

    launch_multiprocess_test(self, _test_barrier_semaphore, 2)


class VerificationTest(jtu.JaxTestCase):

  # Ring allreduce test from tpu_pallas_distributed_test.py
  def test_verification(self):
    # Currently getting unrealized_conversion_cast failure
    # when "pl.async_remote_copy" is nested under lax.fori_loop
    self.skipTest('Skip test until unrealized_conversion_cast failure is resolved.')
    def _test_verification(self):
      num_devices = jax.device_count()
      def kernel_body(in_ref, recv_sem_alias, capacity_sem_alias,
                      out_ref, recv_sem, capacity_sem, scratch_ref):
        del recv_sem_alias, capacity_sem_alias
        my_id = pl.device_id()
        dst_id = lax.rem(my_id + 1, num_devices)
        src_id = lax.rem(my_id - 1, num_devices)
        pl.semaphore_signal(capacity_sem.at[0], 1, device_id=src_id,
                            device_id_type=pl.DeviceIdType.LOGICAL)
        out_ref[...] = jnp.zeros_like(out_ref)
        scratch_ref[0] = in_ref[0]

        @functools.partial(lax.fori_loop, 0, num_devices - 1, init_val=None)
        def _(i, _):
          slot = i % 2
          next_slot = 1 - slot
          pl.semaphore_wait(capacity_sem.at[0], 1)
          copy = pl.async_remote_copy(
              scratch_ref.at[slot],
              scratch_ref.at[next_slot],
              None,
              recv_sem.at[0],
              device_id=dst_id,
              device_id_type=pl.DeviceIdType.LOGICAL,
          )
          out_ref[...] += scratch_ref[slot]
          copy.wait()
          pl.semaphore_signal(capacity_sem.at[0], 1, device_id=src_id,
                              device_id_type=pl.DeviceIdType.LOGICAL)
        out_ref[...] += scratch_ref[(num_devices - 1) % 2]
        pl.semaphore_wait(capacity_sem.at[0], 1)

      kernel = pl.pallas_call(
          kernel_body,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM),
                    pl.BlockSpec(memory_space=plgpu.GMEM),
                    pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=[pl.BlockSpec(memory_space=plgpu.GMEM),
                     pl.BlockSpec(memory_space=plgpu.GMEM),
                     pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_shape=[jax.ShapeDtypeStruct((128, 128), jnp.float32),
                     jax.ShapeDtypeStruct((1,), jnp.int32),
                     jax.ShapeDtypeStruct((1,), jnp.int32)],
          scratch_shapes=[
              plgpu.SMEM((2, 128, 128), jnp.float32),
          ],
          input_output_aliases={1: 1, 2: 2}
      )
      capacity_sem = jnp.zeros((1,), dtype=jnp.int32)
      recv_sem = jnp.zeros((1,), dtype=jnp.int32)
      devices = mesh_utils.create_device_mesh((num_devices,))
      mesh = jax.sharding.Mesh(devices, ['x'])
      # This is just a smoke test to ensure that the verification does not crash.
      with tempfile.TemporaryDirectory() as tmpdir:
        previous_config = jax.config.read('jax_pallas_dump_promela_to')
        jax.config.update('jax_pallas_dump_promela_to', tmpdir)
        shard_map.shard_map(
            kernel, mesh=mesh,
            in_specs=(P('x'), P(None), P(None)),
            out_specs=(P(None), P(None), P(None)),
            check_rep=False
        )(jnp.ones((num_devices, 128, 128), jnp.float32), recv_sem, capacity_sem)
        jax.config.update('jax_pallas_dump_promela_to', previous_config)
        self.assertNotEmpty(os.listdir(tmpdir))

    launch_multiprocess_test(self, _test_verification, 1)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
