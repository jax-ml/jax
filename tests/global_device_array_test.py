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
import unittest
import numpy as np

import jax
from jax import core
from jax._src import test_util as jtu
from jax._src.util import prod, safe_zip

from jax.experimental import PartitionSpec as P
from jax.experimental.maps import Mesh
import jax.experimental.global_device_array as gda_lib
from jax.experimental.global_device_array import GlobalDeviceArray, get_shard_indices

from jax.config import config
config.parse_flags_with_absl()


class GDATest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      ("mesh_x_y", ["x", "y"],
       # There are more slices but for convienient purposes, checking for only
       # 2. The indices + shard_shape + replica_id should be unique enough.
       ((slice(0, 2), slice(0, 1)), (slice(0, 2), slice(1, 2))),
       (2, 1),
       [0, 0, 0, 0, 0, 0, 0, 0], False),
      ("mesh_x_y_pspec", P("x", "y"),
       ((slice(0, 2), slice(0, 1)), (slice(0, 2), slice(1, 2))),
       (2, 1),
       [0, 0, 0, 0, 0, 0, 0, 0], False),
      ("mesh_x", ["x"],
       ((slice(0, 2), slice(None)), (slice(0, 2), slice(None))),
       (2, 2),
       [0, 1, 0, 1, 0, 1, 0, 1], False),
      ("mesh_y", ["y"],
       ((slice(0, 4), slice(None)), (slice(4, 8), slice(None))),
       (4, 2),
       [0, 0, 1, 1, 2, 2, 3, 3], False),
      ("mesh_none_y", [None, "y"],
       ((slice(None), slice(0, 1)), (slice(None), slice(1, 2))),
       (8, 1),
       [0, 0, 1, 1, 2, 2, 3, 3], False),
      ("mesh_xy", [("x", "y")],
       ((slice(0, 1), slice(None)), (slice(1, 2), slice(None))),
       (1, 2),
       [0, 0, 0, 0, 0, 0, 0, 0], False),
      ("mesh_fully_replicated", [],
       ((slice(None), slice(None)), (slice(None), slice(None))),
       (8, 2),
       [0, 1, 2, 3, 4, 5, 6, 7], True),
  )
  def test_gda_2d_shard(self, mesh_axes, expected_index, expected_shard_shape,
                        expected_replica_ids, expected_is_fully_replicated):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    global_input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return global_input_data[index]

    gda = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                          mesh_axes, cb)
    self.assertEqual(gda.ndim, 2)
    self.assertEqual(gda.size, 16)
    self.assertEqual(gda.local_shards[0].index, expected_index[0])
    self.assertArraysEqual(gda.local_data(0),
                           global_input_data[expected_index[0]])
    self.assertEqual(gda.local_shards[1].index, expected_index[1])
    self.assertArraysEqual(gda.local_data(1),
                           global_input_data[expected_index[1]])
    self.assertEqual(gda.local_data(0).shape, expected_shard_shape)
    replica_ids = [i.replica_id for i in gda.local_shards]
    self.assertListEqual(replica_ids, expected_replica_ids)
    self.assertListEqual([i.device.id for i in gda.local_shards],
                         [0, 1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(gda.is_fully_replicated, expected_is_fully_replicated)
    for s in gda.local_shards:
      self.assertEqual(s.data.aval,
                       core.ShapedArray(expected_shard_shape, s.data.dtype))
    for g, l in safe_zip(gda.global_shards, gda.local_shards):
      self.assertEqual(g.device, l.device)
      self.assertEqual(g.index, l.index)
      self.assertEqual(g.replica_id, l.replica_id)
      self.assertEqual(g.data.aval, l.data.aval)
      self.assertArraysEqual(g.data, l.data)


  @parameterized.named_parameters(
      ("mesh_x_y_z", ["x", "y", "z"],
       # There are more slices but for convienient purposes, checking for only
       # 2. The indices + shard_shape + replica_id should be unique enough.
       ((slice(0, 4), slice(0, 2), slice(0, 1)), (slice(0, 4), slice(0, 2), slice(1, 2))),
       (4, 2, 1),
       [0, 0, 0, 0, 0, 0, 0, 0]),
      ("mesh_xy_z", [("x", "y"), "z"],
       ((slice(0, 2), slice(0, 2), slice(None)), (slice(0, 2), slice(2, 4), slice(None))),
       (2, 2, 2),
       [0, 0, 0, 0, 0, 0, 0, 0]),
      ("mesh_z", ["z"],
       ((slice(0, 4), slice(None), slice(None)), (slice(4, 8), slice(None), slice(None))),
       (4, 4, 2),
       [0, 0, 1, 1, 2, 2, 3, 3]),
  )
  def test_gda_3d_shard(self, mesh_axes, expected_index, expected_shard_shape,
                         expected_replica_ids):
    global_mesh = jtu.create_global_mesh((2, 2, 2), ('x', 'y', 'z'))
    global_input_shape = (8, 4, 2)
    global_input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return global_input_data[index]

    gda = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                          mesh_axes, cb)
    self.assertEqual(gda.ndim, 3)
    self.assertEqual(gda.size, 64)
    self.assertEqual(gda.local_shards[0].index, expected_index[0])
    self.assertArraysEqual(gda.local_data(0),
                           global_input_data[expected_index[0]])
    self.assertEqual(gda.local_shards[1].index, expected_index[1])
    self.assertArraysEqual(gda.local_data(1),
                           global_input_data[expected_index[1]])
    self.assertEqual(gda.local_data(0).shape, expected_shard_shape)

    replica_ids = [i.replica_id for i in gda.local_shards]
    self.assertListEqual(replica_ids, expected_replica_ids)

  @parameterized.named_parameters(
      ("mesh_x", ["x"],
       # There are more slices but for convienient purposes, checking for only
       # 2. The indices + shard_shape + replica_id should be unique enough.
       ((slice(0, 2),), (slice(2, 4),)),
       (2,),
       [0, 0, 0, 0, 0, 0, 0, 0]),
      ("mesh_none", [],
       ((slice(None),), (slice(None),)),
       (16,),
       [0, 1, 2, 3, 4, 5, 6, 7]),
  )
  def test_gda_1d_shard(self, mesh_axes, expected_index, expected_shard_shape,
                         expected_replica_ids):
    global_mesh = jtu.create_global_mesh((8,), ('x'))
    global_input_shape = (16,)
    global_input_data = np.arange(prod(global_input_shape)).reshape(-1)
    def cb(index):
      return global_input_data[index]

    gda = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                          mesh_axes, cb)
    self.assertEqual(gda.ndim, 1)
    self.assertEqual(gda.size, 16)
    self.assertEqual(gda.local_shards[0].index, expected_index[0])
    self.assertArraysEqual(gda.local_data(0),
                           global_input_data[expected_index[0]])
    self.assertEqual(gda.local_shards[1].index, expected_index[1])
    self.assertArraysEqual(gda.local_data(1),
                           global_input_data[expected_index[1]])
    self.assertEqual(gda.local_data(0).shape, expected_shard_shape)
    replica_ids = [i.replica_id for i in gda.local_shards]
    self.assertListEqual(replica_ids, expected_replica_ids)

  def test_gda_shape_0_1d_mesh(self):
    global_mesh = jtu.create_global_mesh((8,), ('x'))
    global_input_shape = (0,)
    mesh_axes = [None]
    def cb(index):
      return np.array([])
    gda = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                          mesh_axes, cb)
    self.assertEqual(gda.ndim, 1)
    self.assertEqual(gda.size, 0)
    for i, s in enumerate(gda.local_shards):
      self.assertEqual(s.index, (slice(None),))
      self.assertEqual(s.replica_id, i)
      self.assertArraysEqual(s.data.to_py(), np.array([]))
    self.assertEqual(gda.dtype, np.float32)
    self.assertEqual(
        gda_lib.get_shard_shape(global_input_shape, global_mesh, mesh_axes),
        (0,))


  @parameterized.named_parameters(
      ("mesh_x_y", ["x", "y"],
       # There are more slices but for convienient purposes, checking for only
       # 2. The indices + shard_shape + replica_id should be unique enough.
       ((slice(0, 4), slice(0, 1)), (slice(0, 4), slice(1, 2))),
       (4, 1),
       [0, 0, 0, 0]),
  )
  def test_gda_subset_devices(self, mesh_axes, expected_index,
                               expected_shard_shape, expected_replica_ids):
    global_mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    global_input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return global_input_data[index]

    gda = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                          mesh_axes, cb)
    self.assertEqual(gda.local_shards[0].index, expected_index[0])
    self.assertArraysEqual(gda.local_data(0),
                           global_input_data[expected_index[0]])
    self.assertEqual(gda.local_shards[1].index, expected_index[1])
    self.assertArraysEqual(gda.local_data(1),
                           global_input_data[expected_index[1]])
    self.assertEqual(gda.local_data(0).shape, expected_shard_shape)
    replica_ids = [i.replica_id for i in gda.local_shards]
    self.assertListEqual(replica_ids, expected_replica_ids)
    for g, l in safe_zip(gda.global_shards, gda.local_shards):
      self.assertEqual(g.device, l.device)
      self.assertEqual(g.index, l.index)
      self.assertEqual(g.replica_id, l.replica_id)
      self.assertArraysEqual(g.data, l.data)

  def test_gda_batched_callback(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = [('x', 'y')]
    global_input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)

    def cb(indices):
      self.assertEqual(len(indices), len(global_mesh.local_devices))
      return [global_input_data[index] for index in indices]

    gda = GlobalDeviceArray.from_batched_callback(
        global_input_shape, global_mesh, mesh_axes, cb)
    expected_first_shard_value = np.array([[0, 1]])
    self.assertArraysEqual(gda.local_data(0).to_py(),
                           expected_first_shard_value)
    expected_second_shard_value = np.array([[2, 3]])
    self.assertArraysEqual(gda.local_data(1).to_py(),
                           expected_second_shard_value)

  def test_gda_batched_callback_with_devices(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = ['x']
    global_input_data = np.arange(
        prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    def cb(cb_inp):
      self.assertLen(cb_inp, 4)
      dbs = []
      for inp in cb_inp:
        index, devices = inp
        self.assertLen(devices, 2)
        array = global_input_data[index]
        dbs.extend([jax.device_put(array, device) for device in devices])
      return dbs

    gda = GlobalDeviceArray.from_batched_callback_with_devices(
        global_input_shape, global_mesh, mesh_axes, cb)
    expected_first_shard_value = np.array([[0, 1], [2, 3]], dtype=np.float32)
    self.assertArraysEqual(gda.local_data(0).to_py(),
                           expected_first_shard_value)
    expected_second_shard_value = np.array([[0, 1], [2, 3]], dtype=np.float32)
    self.assertArraysEqual(gda.local_data(1).to_py(),
                           expected_second_shard_value)

  def test_gda_str_repr(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = [('x', 'y')]
    global_input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return global_input_data[index]
    gda = GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes, cb)
    self.assertEqual(str(gda),
                     'GlobalDeviceArray(shape=(8, 2), dtype=int32)')
    self.assertEqual(
        repr(gda),
        ("GlobalDeviceArray(shape=(8, 2), dtype=int32, "
        "global_mesh_shape={'x': 4, 'y': 2}, mesh_axes=[('x', 'y')])"))

  def test_gda_equality_raises_not_implemented(self):
    global_mesh = jtu.create_global_mesh((1, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P(None,)
    global_input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return global_input_data[index]
    input_gda = GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes, cb)
    same_input_gda = GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes, cb)
    with self.assertRaisesRegex(NotImplementedError,
        'GlobalDeviceArray equality is intentionally unimplemented.'):
      input_gda == same_input_gda

  def test_mesh_hash(self):
    global_mesh1 = jtu.create_global_mesh((4, 2), ('x', 'y'))

    global_mesh2 = jtu.create_global_mesh((2, 4), ('x', 'y'))

    global_mesh3 = jtu.create_global_mesh((4, 2), ('x', 'y'))

    self.assertNotEqual(hash(global_mesh1), hash(global_mesh2))
    self.assertEqual(hash(global_mesh1), hash(global_mesh3))

  def test_device_mismatch(self):
    devices = jax.devices()
    if len(devices) < 8:
      raise unittest.SkipTest("Test requires 8 global devices.")
    mesh_devices = np.array([[devices[0], devices[2]],
                             [devices[3], devices[1]],
                             [devices[4], devices[6]],
                             [devices[7], devices[5]]])
    global_mesh = Mesh(mesh_devices, ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = ['x', 'y']
    global_input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    indices = get_shard_indices(global_input_shape, global_mesh, mesh_axes)

    dbs = [
        jax.device_put(global_input_data[indices[d]], d)
        for d in jax.local_devices()
    ]

    with self.assertRaisesRegex(
        ValueError,
        'The `global_mesh.local_devices` and `device_buffers` device order'):
      GlobalDeviceArray(global_input_shape, global_mesh, mesh_axes, dbs)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
