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
"""Tests for serialization and deserialization of GDA."""
import asyncio
import pathlib
import unittest

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src import util
from jax.config import config
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.gda_serialization import serialization
from jax.experimental.maps import Mesh
import numpy as np

config.parse_flags_with_absl()


def create_global_mesh(mesh_shape, axis_names):
  size = util.prod(mesh_shape)
  if len(jax.devices()) < size:
    raise unittest.SkipTest(f'Test requires {size} local devices')
  mesh_devices = np.array(jax.devices()[:size]).reshape(mesh_shape)
  global_mesh = Mesh(mesh_devices, axis_names)
  return global_mesh


class CheckpointTest(jtu.JaxTestCase):

  def test_checkpointing(self):
    global_mesh = create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = ['x', 'y']
    num = util.prod(global_input_shape)
    global_input_data1 = np.arange(num).reshape(global_input_shape)
    def cb1(index):
      return global_input_data1[index]

    gsda1 = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                            mesh_axes, cb1)
    ckpt_dir1 = pathlib.Path(self.create_tempdir('first').full_path)

    global_input_data2 = np.arange(num, num + num).reshape(global_input_shape)
    def cb2(index):
      return global_input_data2[index]

    gsda2 = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                            mesh_axes, cb2)
    ckpt_dir2 = pathlib.Path(self.create_tempdir('second').full_path)

    ckpt_paths = [str(ckpt_dir1), str(ckpt_dir2)]

    tspecs = jax.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    # Async Serialization below.
    async def run_serializer():
      future_writer = jax.tree_map(
          serialization.async_serialize,
          ckpt_paths,
          [gsda1, gsda2],
          tspecs
      )
      return await asyncio.gather(*future_writer)
    asyncio.run(run_serializer())

    # Async Deserialization below.
    async def run():
      future_gsdas = jax.tree_map(
          serialization.async_deserialize,
          ckpt_paths,
          [global_mesh, global_mesh],
          [mesh_axes, ['x']],
          tspecs
      )
      return await asyncio.gather(*future_gsdas)

    m1, m2 = asyncio.run(run())

    self.assertArraysEqual(m1.local_shards[0].data.to_py(),
                           np.array([[0], [2]]))
    self.assertArraysEqual(m1.local_shards[1].data.to_py(),
                           np.array([[1], [3]]))
    self.assertEqual(m1.local_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.int32)

    self.assertArraysEqual(m2.local_shards[0].data.to_py(),
                           np.array([[16, 17], [18, 19]]))
    self.assertArraysEqual(m2.local_shards[1].data.to_py(),
                           np.array([[16, 17], [18, 19]]))
    self.assertEqual(m2.local_shards[0].data.shape, (2, 2))
    self.assertEqual(m2.dtype, np.int32)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
