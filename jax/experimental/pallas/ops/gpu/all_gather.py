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

# Example usage: mpirun -n 8 python3 gpu_all_gather_test.py

from functools import partial
import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import mosaic_gpu as plgpu
import os

P = jax.sharding.PartitionSpec

@partial(jax.jit, static_argnums=(2, 3))
def all_gather(input, recvs_sems, mesh, axis_name):
  partition = P(axis_name, None)
  num_elts = input.shape[-1]
  num_devices = mesh.shape[axis_name]
  grid_spec = pl.GridSpec(grid=(num_devices,),
                          out_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),
                                     pl.BlockSpec(memory_space=plgpu.GMEM)),
                          in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),
                                    pl.BlockSpec(memory_space=plgpu.GMEM)))
  out_shape = [jax.ShapeDtypeStruct((num_devices, 1, num_elts), jnp.float32),
               jax.ShapeDtypeStruct((num_devices,), jnp.int32)]

  # one shot Allgather
  def one_shot_all_gather_kernel(input_ref,
                                aliased_recv_sems,
                                output_ref,
                                recv_sems):
    del aliased_recv_sems
    outer_step = pl.program_id(0)
    my_id = pl.device_id()
    neighbor_id = lax.rem(my_id + outer_step, num_devices)
    remote_copy_op = pl.make_async_remote_copy(
        src_ref=input_ref,
        dst_ref=output_ref.at[my_id],
        send_sem=None,
        recv_sem=recv_sems.at[outer_step],
        device_id=neighbor_id,
        device_id_type=pl.DeviceIdType.LOGICAL,
    )
    remote_copy_op.start()
    remote_copy_op.wait_recv()

  pallas_all_gather = pl.pallas_call(
    one_shot_all_gather_kernel,
    out_shape=out_shape,
    grid_spec=grid_spec,
    input_output_aliases={1: 1}
  )
  return shard_map.shard_map(
    pallas_all_gather,
    mesh=mesh,
    in_specs=(partition, P(None)),
    out_specs=(partition, P(None)),
    check_rep=False
  )(input, recvs_sems)

if __name__ == "__main__":
  num_elts = int(os.getenv('NUM_ELTS', '1024'))

  jax.distributed.initialize()

  num_devices = jax.device_count()
  if jax.process_index() == 0:
      print(f"Running with {num_devices} {jax.devices()[0].device_kind} devices.")

  partition = P('x', None)
  devices = mesh_utils.create_device_mesh((num_devices,))
  mesh = jax.sharding.Mesh(devices, ('x',))
  sharding = jax.sharding.NamedSharding(mesh, partition)

  # Create an input array that shards the first dimension across
  # all devices.
  input_arr = jnp.arange(num_devices * num_elts, dtype='float32').reshape(num_devices, num_elts)
  input_arr = jax.device_put(input_arr, sharding)

  recvs_sems = jnp.zeros((num_devices,), dtype='int32')

  # Compare Pallas result to lax.all_gather result.
  jitted_allgather = jax.jit(
    shard_map.shard_map(
      lambda x: lax.all_gather(x, 'x'),
      mesh=mesh, in_specs=partition, out_specs=partition,
      check_rep=False
    )
  )

  pallas_result, recvs_sems_out = jax.block_until_ready(all_gather(input_arr, recvs_sems, mesh, 'x'))
  xla_result = jax.block_until_ready(jitted_allgather(input_arr))
  if jnp.allclose(pallas_result, xla_result):
    print('SUCCESS -- Pallas all_gather and lax.all_gather match')
  else:
    print('FAILURE -- Pallas all_gather and lax.all_gather do not match')
    print('Pallas Result: ', pallas_result.shape, pallas_result.addressable_shards)
    print('lax.all_gather Result: ', xla_result.shape, xla_result.addressable_shards)
    exit(1)