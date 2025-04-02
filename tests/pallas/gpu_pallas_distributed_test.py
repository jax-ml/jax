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
import portpicker
from absl.testing import absltest
import jax
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import mosaic_gpu as plgpu
# from jax.experimental.pallas.ops.gpu.all_gather import all_gather
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
        neighbor_ptr = plgpu.remote_ref(y_ref, other_dev_id)
        neighbor_ptr[...] = x_ref[...]
        pl.semaphore_signal(recv_sem.at[0], device_id=other_dev_id,
                            device_id_type=pl.DeviceIdType.LOGICAL)
        pl.semaphore_wait(recv_sem.at[0])

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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())