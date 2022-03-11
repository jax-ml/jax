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
import re
from typing import Callable

import jax
from jax._src.util import prod
from jax.experimental import global_device_array as gda
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import numpy as np
import tensorstore as ts


async def create_async_gda_from_callback(
    global_shape: gda.Shape,
    global_mesh: Mesh,
    mesh_axes: gda.MeshAxes,
    data_callback: Callable[[gda.Index], asyncio.Future],
):
  global_idx_rid = gda.get_shard_indices_replica_ids(
      global_shape, global_mesh, mesh_axes)
  local_devices = global_mesh.local_devices
  future_arrays = [data_callback(global_idx_rid[d][0])
                   for d in local_devices]
  # Pause here and come back to `from_async_callback()` when future_arrays are
  # ready. device_put cannot happen with future_arrays.
  local_arrays = await asyncio.gather(*future_arrays)

  dbs = [jax.device_put(array, device)
         for array, device in zip(local_arrays, local_devices)]
  return gda.GlobalDeviceArray(global_shape, global_mesh, mesh_axes, dbs,
                               gda._GdaFastPathArgs(global_idx_rid, local_devices))


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
      'chunks': np.array(np.maximum(1, gda.local_data(0).shape)),
      'dtype': dtype,
  }


def get_tensorstore_spec(ckpt_path: str):
  spec = {'driver': 'zarr', 'kvstore': {}}

  if ckpt_path.startswith('gs://'):
    m = re.fullmatch('^gs://([^/]*)/(.*)$', ckpt_path, re.DOTALL)
    if m is None:
      raise ValueError('The ckpt_path should contain the bucket name and the '
                       f'file path inside the bucket. Got: {ckpt_path}')
    gcs_bucket = m.group(1)
    path_without_bucket = m.group(2)
    spec['kvstore'] = {'driver': 'gcs', 'bucket': gcs_bucket,
                       'path': path_without_bucket}
  else:
    spec['kvstore'] = {'driver': 'file', 'path': ckpt_path}
  return spec


async def async_serialize(gda_inp: gda.GlobalDeviceArray, tensorstore_spec):
  if not tensorstore_spec.get('metadata'):
    tensorstore_spec['metadata'] = _get_metadata(gda_inp)

  t = await ts.open(
      ts.Spec(tensorstore_spec),
      create=True,
      open=True,
      context=ts.Context({'file_io_concurrency': {
          'limit': 128
      }}))

  async def _write_array(shard):
    if shard.replica_id == 0:
      await t[shard.index].write(shard.data)

  future_write_state = jax.tree_util.tree_map(_write_array,
                                              tuple(gda_inp.local_shards))
  return await asyncio.gather(*future_write_state)


def run_serialization(gdas, tensorstore_specs):
  async def _run_serializer():
    future_writer = jax.tree_map(async_serialize, gdas, tensorstore_specs)
    return await asyncio.gather(*future_writer)
  asyncio.run(_run_serializer())


async def async_deserialize(mesh, mesh_axes, tensorstore_spec, global_shape=None):
  t = ts.open(ts.Spec(tensorstore_spec), open=True).result()
  shape = t.shape if global_shape is None else global_shape
  requires_padding = prod(shape) > prod(t.shape)

  if requires_padding:
    new_shard_shape = gda.get_shard_shape(shape, mesh, mesh_axes)

  async def cb(index):
    if requires_padding:
      # This is needed because the shape the array was saved with is smaller
      # than the requested shape of the array in which it will be reloaded. So
      # the extra values will be filled with 0s.
      out = np.zeros(new_shard_shape, dtype=t.dtype.numpy_dtype)
      requested_domain = ts.IndexTransform(input_shape=shape)[index].domain
      restricted_domain = t.domain.intersect(requested_domain)
      await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]][restricted_domain].write(t[restricted_domain])
      return out
    else:
      return await t[index].read()

  return await create_async_gda_from_callback(shape, mesh, mesh_axes, cb)


def run_deserialization(global_meshes, mesh_axes, tensorstore_specs,
                        global_shapes=None):
  async def _run_deserializer():
    future_gdas = jax.tree_map(
        async_deserialize, global_meshes, mesh_axes, tensorstore_specs,
        [None] * len(tensorstore_specs) if global_shapes is None else global_shapes)
    return await asyncio.gather(*future_gdas)
  return asyncio.run(_run_deserializer())
