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
"""Tests for GlobalShardedDeviceArray."""

import unittest
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src.util import prod

from jax.experimental import PartitionSpec as P
from jax.experimental.maps import Mesh
from jax.experimental.gsda import GlobalShardedDeviceArray

from jax.config import config
config.parse_flags_with_absl()


def create_global_mesh(mesh_shape, axis_names):
  size = prod(mesh_shape)
  if len(jax.devices()) < size:
    raise unittest.SkipTest(f"Test requires {size} local devices")
  mesh_devices = np.array(jax.devices()[:size]).reshape(mesh_shape)
  global_mesh = Mesh(mesh_devices, axis_names)
  return global_mesh


class GSDATest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      ("mesh_x_y", ["x", "y"],
       # There are more slices but for convienient purposes, checking for only
       # 2. The indices + shard_shape + replica_id should be unique enough.
       ((slice(0, 2), slice(0, 1)), (slice(0, 2), slice(1, 2))),
       (2, 1),
       [0, 0, 0, 0, 0, 0, 0, 0]),
      ("mesh_x_y_pspec", P("x", "y"),
       ((slice(0, 2), slice(0, 1)), (slice(0, 2), slice(1, 2))),
       (2, 1),
       [0, 0, 0, 0, 0, 0, 0, 0]),
      ("mesh_x", ["x"],
       ((slice(0, 2), slice(None)), (slice(0, 2), slice(None))),
       (2, 2),
       [0, 1, 0, 1, 0, 1, 0, 1]),
      ("mesh_y", ["y"],
       ((slice(0, 4), slice(None)), (slice(4, 8), slice(None))),
       (4, 2),
       [0, 0, 1, 1, 2, 2, 3, 3]),
      ("mesh_none_y", [None, "y"],
       ((slice(None), slice(0, 1)), (slice(None), slice(1, 2))),
       (8, 1),
       [0, 0, 1, 1, 2, 2, 3, 3]),
      ("mesh_xy", [("x", "y")],
       ((slice(0, 1), slice(None)), (slice(1, 2), slice(None))),
       (1, 2),
       [0, 0, 0, 0, 0, 0, 0, 0]),
  )
  def test_gsda_2d_shard(self, mesh_axes, expected_index, expected_shard_shape,
                         expected_replica_ids):
    global_mesh = create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    global_input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return global_input_data[index]
    gsda = GlobalShardedDeviceArray.from_callback(global_input_shape,
                                                  jnp.float32, global_mesh,
                                                  mesh_axes, cb)
    self.assertEqual(gsda.local_shards[0].index, expected_index[0])
    self.assertArraysEqual(gsda.local_shards[0].data,
                           global_input_data[expected_index[0]])
    self.assertEqual(gsda.local_shards[1].index, expected_index[1])
    self.assertArraysEqual(gsda.local_shards[1].data,
                           global_input_data[expected_index[1]])
    self.assertEqual(gsda.local_shards[0].data.shape, expected_shard_shape)
    replica_ids = [i.replica_id for i in gsda.local_shards]
    self.assertListEqual(replica_ids, expected_replica_ids)
    self.assertListEqual([i.device.id for i in gsda.local_shards],
                         [0, 1, 2, 3, 4, 5, 6, 7])

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
  def test_gsda_3d_shard(self, mesh_axes, expected_index, expected_shard_shape,
                         expected_replica_ids):
    global_mesh = create_global_mesh((2, 2, 2), ('x', 'y', 'z'))
    global_input_shape = (8, 4, 2)
    global_input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return global_input_data[index]
    gsda = GlobalShardedDeviceArray.from_callback(global_input_shape,
                                                  jnp.float32, global_mesh,
                                                  mesh_axes, cb)
    self.assertEqual(gsda.local_shards[0].index, expected_index[0])
    self.assertArraysEqual(gsda.local_shards[0].data,
                           global_input_data[expected_index[0]])
    self.assertEqual(gsda.local_shards[1].index, expected_index[1])
    self.assertArraysEqual(gsda.local_shards[1].data,
                           global_input_data[expected_index[1]])
    self.assertEqual(gsda.local_shards[0].data.shape, expected_shard_shape)

    replica_ids = [i.replica_id for i in gsda.local_shards]
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
  def test_gsda_1d_shard(self, mesh_axes, expected_index, expected_shard_shape,
                         expected_replica_ids):
    global_mesh = create_global_mesh((8,), ('x'))
    global_input_shape = (16,)
    global_input_data = np.arange(prod(global_input_shape)).reshape(-1)
    def cb(index):
      return global_input_data[index]
    gsda = GlobalShardedDeviceArray.from_callback(global_input_shape,
                                                  jnp.float32, global_mesh,
                                                  mesh_axes, cb)
    self.assertEqual(gsda.local_shards[0].index, expected_index[0])
    self.assertArraysEqual(gsda.local_shards[0].data,
                           global_input_data[expected_index[0]])
    self.assertEqual(gsda.local_shards[1].index, expected_index[1])
    self.assertArraysEqual(gsda.local_shards[1].data,
                           global_input_data[expected_index[1]])
    self.assertEqual(gsda.local_shards[0].data.shape, expected_shard_shape)
    replica_ids = [i.replica_id for i in gsda.local_shards]
    self.assertListEqual(replica_ids, expected_replica_ids)

  @parameterized.named_parameters(
      ("mesh_x_y", ["x", "y"],
       # There are more slices but for convienient purposes, checking for only
       # 2. The indices + shard_shape + replica_id should be unique enough.
       ((slice(0, 4), slice(0, 1)), (slice(0, 4), slice(1, 2))),
       (4, 1),
       [0, 0, 0, 0]),
  )
  def test_gsda_subset_devices(self, mesh_axes, expected_index,
                               expected_shard_shape, expected_replica_ids):
    global_mesh = create_global_mesh((2, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    global_input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return global_input_data[index]
    gsda = GlobalShardedDeviceArray.from_callback(global_input_shape,
                                                  jnp.float32, global_mesh,
                                                  mesh_axes, cb)
    self.assertEqual(gsda.local_shards[0].index, expected_index[0])
    self.assertArraysEqual(gsda.local_shards[0].data,
                           global_input_data[expected_index[0]])
    self.assertEqual(gsda.local_shards[1].index, expected_index[1])
    self.assertArraysEqual(gsda.local_shards[1].data,
                           global_input_data[expected_index[1]])
    self.assertEqual(gsda.local_shards[0].data.shape, expected_shard_shape)
    replica_ids = [i.replica_id for i in gsda.local_shards]
    self.assertListEqual(replica_ids, expected_replica_ids)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
