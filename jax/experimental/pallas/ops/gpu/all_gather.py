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

P = jax.sharding.PartitionSpec

@partial(jax.jit, static_argnums=(1, 2))
def all_gather(input, mesh, axis_name):
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

  recvs_sems = jnp.zeros((num_devices,), dtype='int32')

  # one shot Allgather
  def one_shot_all_gather_kernel(input_ref,
                                aliased_recv_sems,
                                output_ref,
                                recv_sems):
    del aliased_recv_sems
    outer_step = pl.program_id(0)
    my_id = pl.device_id()
    neighbor_id = lax.rem(my_id + outer_step, num_devices)
    neighbor_ptr = plgpu.remote_ref(output_ref, neighbor_id)
    neighbor_ptr[my_id] = input_ref[...]
    pl.semaphore_signal(recv_sems.at[outer_step], device_id=neighbor_id,
                        device_id_type=pl.DeviceIdType.LOGICAL)
    pl.semaphore_wait(recv_sems.at[outer_step])

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
  )(input, recvs_sems)[0]


def main(unused_argv):
  num_elts = 1024

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

  # Compare Pallas result to lax.all_gather result.
  jitted_allgather = jax.jit(
    shard_map.shard_map(
      lambda x: lax.all_gather(x, 'x'),
      mesh=mesh, in_specs=partition, out_specs=partition,
      check_rep=False
    )
  )

  pallas_result = jax.block_until_ready(all_gather(input_arr, mesh, 'x'))
  xla_result = jax.block_until_ready(jitted_allgather(input_arr))
  if jnp.allclose(pallas_result, xla_result):
    print('SUCCESS -- Pallas all_gather and lax.all_gather match')
  else:
    print('FAILURE -- Pallas all_gather and lax.all_gather do not match')
    print('Pallas Result: ', pallas_result.shape, pallas_result.addressable_shards)
    print('lax.all_gather Result: ', xla_result.shape, xla_result.addressable_shards)
    exit(1)


if __name__ == "__main__":
  from absl import app
  import jax
  jax.config.config_with_absl()
  app.run(main)
