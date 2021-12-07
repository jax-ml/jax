# Copyright 2021 Google LLC
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
"""GlobalDeviceArray serialization and deserialization."""

import asyncio
from typing import Callable

import jax
from jax.experimental import global_device_array as gda
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import numpy as np
import tensorstore as ts


async def create_async_gsda_from_callback(
    global_shape: gda.Shape,
    global_mesh: Mesh,
    mesh_axes: gda.MeshAxes,
    data_callback: Callable[[gda.Index], asyncio.Future],
):
  indices = gda.get_shard_indices(global_shape, global_mesh, mesh_axes)
  future_arrays = [data_callback(indices[d]) for d in global_mesh.local_devices]
  # Pause here and come back to `from_async_callback()` when future_arrays are
  # ready. device_put cannot happen with future_arrays.
  local_arrays = await asyncio.gather(*future_arrays)

  dbs = [jax.device_put(array, device)
         for array, device in zip(local_arrays, global_mesh.local_devices)]
  return gda.GlobalDeviceArray(global_shape, global_mesh, mesh_axes, dbs)


def _get_metadata(gda):
  if gda.dtype == jnp.bfloat16:
    # Tensorstore uses 'bfloat16', not '<V2'.
    dtype = 'bfloat16'
  else:
    dtype = np.dtype(gda.dtype).str

  return {
      'compressor': {
          'id': 'gzip'
      },
      'shape': gda.shape,
      'chunks': np.array(gda.local_data(0).shape),
      'dtype': dtype,
  }


def get_tensorstore_spec(ckpt_path):
  spec = {'driver': 'zarr', 'kvstore': {}}
  # TODO(yashkatariya): Add GCS kvstore too.
  spec['kvstore'] = {
      'driver': 'file',
      'path': ckpt_path,
  }
  return spec


async def async_serialize(ckpt_path: str, gda: gda.GlobalDeviceArray,
                          tensorstore_spec):
  if not tensorstore_spec.get('metadata'):
    tensorstore_spec['metadata'] = _get_metadata(gda)

  async def _write_array(shard):
    if shard.replica_id == 0:
      t = await ts.open(
          ts.Spec(tensorstore_spec),
          create=True,
          open=True,
          context=ts.Context({'file_io_concurrency': {
              'limit': 128
          }}))
      await t[shard.index].write(shard.data)

  async def writer():
    future_write_state = jax.tree_util.tree_map(_write_array,
                                                tuple(gda.local_shards))
    await asyncio.gather(*future_write_state)
  return await writer()


async def async_deserialize(ckpt_path, mesh, mesh_axes, tensorstore_spec):
  t = ts.open(ts.Spec(tensorstore_spec), open=True).result()

  async def cb(index):
    return await t[index].read()

  return await create_async_gsda_from_callback(t.shape, mesh, mesh_axes, cb)
