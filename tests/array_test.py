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
"""Tests for GlobalDeviceArray."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
from jax._src import test_util as jtu
from jax._src.lib import xla_client as xc
from jax._src.util import prod
from jax.experimental import PartitionSpec as P
from jax.experimental import sharding
from jax.experimental import array

from jax.config import config
config.parse_flags_with_absl()


def create_array(shape, sharding, global_data=None):
  if global_data is None:
    global_data = np.arange(prod(shape)).reshape(shape)

  return array.make_array_from_callback(
      shape, sharding, lambda idx: global_data[idx]), global_data


class JaxArrayTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      ("mesh_x_y", P("x", "y")),
      ("mesh_x", P("x")),
      ("mesh_y", P("y")),
      ("mesh_none_y", P(None, "y")),
      ("mesh_xy", P(("x", "y"))),
      ("mesh_fully_replicated", P()),
  )
  def test_jax_array_value(self, mesh_axes):
    with jax._src.config.jax_array(True):
      global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
      input_shape = (8, 2)
      arr, global_data = create_array(
          input_shape, sharding.MeshPspecSharding(global_mesh, mesh_axes))
      for s in arr.addressable_shards:
        self.assertLen(s.data._arrays, 1)
        self.assertArraysEqual(s.data._arrays[0], global_data[s.index])
      self.assertArraysEqual(arr._value, global_data)
      self.assertArraysEqual(arr._npy_value, global_data)

  def test_array_delete(self):
    with jax._src.config.jax_array(True):
      global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
      input_shape = (8, 2)
      arr, _ = create_array(
          input_shape, sharding.MeshPspecSharding(global_mesh, P('x', 'y')))
      arr.delete()
      with self.assertRaisesRegex(ValueError, 'Array has been deleted.'):
        arr._check_if_deleted()
      self.assertIsNone(arr._npy_value)
      self.assertIsNone(arr._arrays)

  def test_device_put(self):
    with jax._src.config.jax_array(True):
      numpy_array = np.array([1, 2, 3])
      arr = jax.device_put(numpy_array, jax.devices()[0])
      self.assertIsInstance(arr.sharding, sharding.SingleDeviceSharding)
      self.assertArraysEqual(arr, numpy_array)
      self.assertEqual(arr._committed, True)
      for i in arr.addressable_shards:
        self.assertArraysEqual(i.data, numpy_array)
        self.assertEqual(i.device, jax.devices()[0])
        self.assertEqual(i.index, (slice(None),))

  def test_device_put_array_delete(self):
    with jax._src.config.jax_array(True):
      arr = jax.device_put(np.array([1, 2, 3]), jax.devices()[0])
      arr.delete()
      with self.assertRaisesRegex(ValueError, 'Array has been deleted.'):
        arr._check_if_deleted()
      self.assertIsNone(arr._npy_value)
      self.assertIsNone(arr._arrays)

  def test_array_device_get(self):
    with jax._src.config.jax_array(True):
      global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
      input_shape = (8, 2)
      arr, input_data = create_array(
          input_shape, sharding.MeshPspecSharding(global_mesh, P('x', 'y')))
      self.assertArraysEqual(jax.device_get(arr), input_data)

  def test_repr(self):
    with jax._src.config.jax_array(True):
      global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
      input_shape = (8, 2)
      arr, _ = create_array(
          input_shape, sharding.MeshPspecSharding(global_mesh, P('x', 'y')))
      repr(arr)  # doesn't crash


class ShardingTest(jtu.JaxTestCase):

  def test_mesh_pspec_sharding_interface(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    pspec = P('y', 'x')
    global_shape = (8, 4)
    mp_sharding = sharding.MeshPspecSharding(mesh, pspec)
    di_map = mp_sharding.devices_indices_map(global_shape)
    op_sharding = mp_sharding._to_xla_op_sharding(len(global_shape))
    device_assignment = mp_sharding._device_assignment()

    self.assertEqual(di_map[mesh.devices.flat[0]], (slice(0, 4), slice(0, 1)))
    self.assertArraysEqual(device_assignment, list(mesh.devices.flat))
    self.assertEqual(op_sharding.type, xc.OpSharding.Type.OTHER)
    self.assertListEqual(op_sharding.tile_assignment_dimensions, [2, 4])
    self.assertListEqual(op_sharding.tile_assignment_devices,
                         [0, 2, 4, 6, 1, 3, 5, 7])


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
