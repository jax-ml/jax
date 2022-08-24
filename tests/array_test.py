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
import jax.numpy as jnp
from jax._src import config as jax_config
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
    with jax_config.jax_array(True):
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
    with jax_config.jax_array(True):
      global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
      input_shape = (8, 2)
      arr, _ = create_array(
          input_shape, sharding.MeshPspecSharding(global_mesh, P('x', 'y')))
      arr.delete()
      with self.assertRaisesRegex(RuntimeError, 'Array has been deleted.'):
        arr._check_if_deleted()
      self.assertIsNone(arr._npy_value)
      self.assertIsNone(arr._arrays)

  def test_device_put(self):
    with jax_config.jax_array(True):
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
    with jax_config.jax_array(True):
      arr = jax.device_put(np.array([1, 2, 3]), jax.devices()[0])
      arr.delete()
      with self.assertRaisesRegex(RuntimeError, 'Array has been deleted.'):
        arr._check_if_deleted()
      self.assertIsNone(arr._npy_value)
      self.assertIsNone(arr._arrays)

  def test_array_device_get(self):
    with jax_config.jax_array(True):
      global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
      input_shape = (8, 2)
      arr, input_data = create_array(
          input_shape, sharding.MeshPspecSharding(global_mesh, P('x', 'y')))
      self.assertArraysEqual(jax.device_get(arr), input_data)

  def test_repr(self):
    with jax_config.jax_array(True):
      global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
      input_shape = (8, 2)
      arr, _ = create_array(
          input_shape, sharding.MeshPspecSharding(global_mesh, P('x', 'y')))
      repr(arr)  # doesn't crash

  def test_jnp_array(self):
    with jax_config.jax_array(True):
      arr = jnp.array([1, 2, 3])
      self.assertIsInstance(arr, array.Array)
      self.assertIsInstance(arr.sharding, sharding.SingleDeviceSharding)
      self.assertEqual(arr._committed, False)

  def test_jnp_array_jit_add(self):
    with jax_config.jax_array(True):
      a = jnp.array([1, 2, 3])
      b = jnp.array([4, 5, 6])
      arr = jax.jit(lambda x, y: x + y)(a, b)
      self.assertIsInstance(arr, array.Array)
      self.assertArraysEqual(arr, np.array([5, 7, 9]))
      self.assertIsInstance(arr.sharding, sharding.SingleDeviceSharding)

  def test_jnp_array_jnp_add(self):
    with jax_config.jax_array(True):
      arr = jnp.add(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
      self.assertIsInstance(arr, array.Array)
      self.assertArraysEqual(arr, np.array([5, 7, 9]))
      self.assertIsInstance(arr.sharding, sharding.SingleDeviceSharding)

  def test_jnp_array_normal_add(self):
    with jax_config.jax_array(True):
      a = jnp.array([1, 2, 3])
      b = jnp.array([4, 5, 6])
      arr = a + b
      self.assertIsInstance(arr, array.Array)
      self.assertArraysEqual(arr, np.array([5, 7, 9]))
      self.assertIsInstance(arr.sharding, sharding.SingleDeviceSharding)

  def test_array_sharded_astype(self):
    with jax_config.jax_array(True):
      global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
      input_shape = (8, 2)
      arr, input_data = create_array(
          input_shape, sharding.MeshPspecSharding(global_mesh, P('x', 'y')))
      arr_float32 = arr.astype(jnp.float32)
      self.assertEqual(arr_float32.dtype, np.float32)
      self.assertArraysEqual(arr_float32, input_data.astype(np.float32))
      self.assertLen(arr_float32.addressable_shards, 8)
      for i in arr_float32.addressable_shards:
        self.assertArraysEqual(i.data, input_data[i.index].astype(np.float32))

  def test_jnp_array_astype(self):
    with jax_config.jax_array(True):
      arr = jnp.array([1, 2, 3])
      arr_float32 = arr.astype(jnp.float32)
      self.assertEqual(arr_float32.dtype, np.float32)
      self.assertArraysEqual(arr_float32, arr.astype(np.float32))

  @jax_config.jax_array(True)
  def test_sharded_add(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    a, input_data = create_array(
        input_shape, sharding.MeshPspecSharding(global_mesh, P('x', 'y')))
    b, _ = create_array(
        input_shape, sharding.MeshPspecSharding(global_mesh, P('x')))
    out = a + b
    expected = input_data + input_data
    self.assertArraysEqual(out, expected)
    self.assertLen(out.addressable_shards, 8)
    for i in out.addressable_shards:
      self.assertArraysEqual(i.data, expected[i.index])

  @jax_config.jax_array(True)
  def test_sharded_zeros_like(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    a, input_data = create_array(
        input_shape, sharding.MeshPspecSharding(global_mesh, P('x', 'y')))
    out = jnp.zeros_like(a)
    expected = jnp.zeros(input_data.shape, dtype=np.int32)
    self.assertArraysEqual(out, expected)
    self.assertLen(out.addressable_shards, 8)
    for i in out.addressable_shards:
      self.assertArraysEqual(i.data, expected[i.index])

  @jax_config.jax_array(True)
  def test_zeros_like(self):
    a = jnp.array([1, 2, 3], dtype=np.int32)
    out = jnp.zeros_like(a)
    expected = np.zeros(a.shape, dtype=np.int32)
    self.assertArraysEqual(out, expected)
    self.assertIsInstance(out.sharding, sharding.SingleDeviceSharding)

  @jax_config.jax_array(True)
  def test_wrong_num_arrays(self):
    shape = (8, 2)
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    s = sharding.MeshPspecSharding(mesh, P('x', 'y'))
    inp_data = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    bufs = [jax.device_put(inp_data[s.device_indices(d, shape)], d)
            for d in jax.local_devices()]
    with self.assertRaisesRegex(
        ValueError,
        r'Expected 8 per-device arrays \(this is how many devices are addressable '
        r'by the sharding\), but got 4'):
      array.Array(jax.ShapedArray(shape, np.float32), s, bufs[:4], committed=True)

    with self.assertRaisesRegex(
        ValueError,
        r'Expected 8 per-device arrays \(this is how many devices are addressable '
        r'by the sharding\), but got 16'):
      array.Array(jax.ShapedArray(shape, np.float32), s, bufs + bufs, committed=True)

  @jax_config.jax_array(True)
  def test_arrays_not_in_device_assignment(self):
    if jax.device_count() < 4:
      self.skipTest('Requires more than 4 devices')
    shape = (8, 2)
    mesh = jtu.create_global_mesh((1, 2), ('x', 'y'))
    s = sharding.MeshPspecSharding(mesh, P('x', 'y'))
    inp_data = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    bufs = [jax.device_put(inp_data, d) for d in jax.devices()[2:4]]
    with self.assertRaisesRegex(
        ValueError,
        "Some per-device arrays are placed on devices {2, 3}, which are "
        "not used in the specified sharding"):
      array.Array(jax.ShapedArray(shape, np.float32), s, bufs, committed=True)


class ShardingTest(jtu.JaxTestCase):

  def test_mesh_pspec_sharding_interface(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    pspec = P('y', 'x')
    global_shape = (8, 4)
    mp_sharding = sharding.MeshPspecSharding(mesh, pspec)
    di_map = mp_sharding.devices_indices_map(global_shape)
    op_sharding = mp_sharding._to_xla_op_sharding(len(global_shape))
    device_assignment = mp_sharding._device_assignment

    self.assertEqual(di_map[mesh.devices.flat[0]], (slice(0, 4), slice(0, 1)))
    self.assertArraysEqual(device_assignment, list(mesh.devices.flat))
    self.assertEqual(op_sharding.type, xc.OpSharding.Type.OTHER)
    self.assertListEqual(op_sharding.tile_assignment_dimensions, [2, 4])
    self.assertListEqual(op_sharding.tile_assignment_devices,
                         [0, 2, 4, 6, 1, 3, 5, 7])

  @parameterized.named_parameters(
      ("mesh_x_y", P("x", "y")),
      ("mesh_x", P("x")),
      ("mesh_y", P("y")),
      ("mesh_none_y", P(None, "y")),
      ("mesh_none_x", P(None, "x")),
      ("mesh_xy", P(("x", "y"))),
      ("mesh_fully_replicated", P()),
  )
  def test_op_sharding_indices(self, pspec):
    shape = (8, 4)
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mps = sharding.MeshPspecSharding(mesh, pspec)
    ops = sharding.OpShardingSharding(
        list(mesh.devices.flat), mps._to_xla_op_sharding(len(shape)))
    self.assertDictEqual(
        ops.devices_indices_map(shape), mps.devices_indices_map(shape))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
