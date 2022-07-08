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

import pathlib

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src import util
from jax.config import config
from jax.experimental import PartitionSpec as P
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.gda_serialization import serialization
import numpy as np

config.parse_flags_with_absl()


class CheckpointTest(jtu.JaxTestCase):

  def test_checkpointing(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P('x', 'y')
    num = util.prod(global_input_shape)

    # First GDA
    global_input_data1 = np.arange(num).reshape(global_input_shape)
    def cb1(index):
      return global_input_data1[index]
    gda1 = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                           mesh_axes, cb1)
    ckpt_dir1 = pathlib.Path(self.create_tempdir('first').full_path)

    # Second GDA
    global_input_data2 = np.arange(num, num + num).reshape(global_input_shape)
    def cb2(index):
      return global_input_data2[index]
    gda2 = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                           mesh_axes, cb2)
    ckpt_dir2 = pathlib.Path(self.create_tempdir('second').full_path)

    # Third GDA
    def cb3(index):
      return np.array([])
    global_mesh1d = jtu.create_global_mesh((8,), ('x',))
    gda3 = GlobalDeviceArray.from_callback((0,), global_mesh1d, P(None), cb3)
    ckpt_dir3 = pathlib.Path(self.create_tempdir('third').full_path)

    ckpt_paths = [str(ckpt_dir1), str(ckpt_dir2), str(ckpt_dir3)]
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    serialization.run_serialization([gda1, gda2, gda3], tspecs)

    m1, m2, m3 = serialization.run_deserialization(
        [global_mesh, global_mesh, global_mesh1d],
        [mesh_axes, P('x'), P(None)],
        tspecs)

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

    for i, s in enumerate(m3.local_shards):
      self.assertEqual(s.index, (slice(None),))
      self.assertEqual(s.replica_id, i)
      self.assertArraysEqual(s.data.to_py(), np.array([]))
    self.assertEqual(m3.dtype, np.float32)

  def test_checkpointing_with_bigger_shape(self):
    global_mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    num = util.prod(global_input_shape)

    # First GDA
    global_input_data1 = np.arange(num, dtype=np.int32).reshape(global_input_shape)
    def cb1(index):
      return global_input_data1[index]
    gda1 = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                           P('x', 'y'), cb1)
    ckpt_dir1 = pathlib.Path(self.create_tempdir('first').full_path)

    ckpt_paths = [str(ckpt_dir1)]
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    serialization.run_serialization([gda1], tspecs)

    m1, = serialization.run_deserialization(
        [jtu.create_global_mesh((4, 2), ('x', 'y'))],
        [P('x', 'y')],
        tspecs,
        [(12, 2)],
        [np.float32]
    )

    expected_data = {
        0: np.array([[0], [2], [4]], dtype=np.float32),
        1: np.array([[1], [3], [5]], dtype=np.float32),
        2: np.array([[6], [8], [10]], dtype=np.float32),
        3: np.array([[7], [9], [11]], dtype=np.float32),
        4: np.array([[12], [14], [0]], dtype=np.float32),
        5: np.array([[13], [15], [0]], dtype=np.float32),
        6: np.array([[0], [0], [0]], dtype=np.float32),
        7: np.array([[0], [0], [0]], dtype=np.float32),
    }

    for l in m1.local_shards:
      self.assertArraysEqual(l.data.to_py(), expected_data[l.device.id])

  def test_checkpointing_scalar(self):
    global_mesh = jtu.create_global_mesh((2,), ('x'))
    global_input_shape = ()
    data = np.array(4)
    gda1 = GlobalDeviceArray.from_callback(global_input_shape, global_mesh,
                                           P(None), lambda idx: data[idx])
    ckpt_dir1 = pathlib.Path(self.create_tempdir('first').full_path)

    ckpt_paths = [str(ckpt_dir1)]
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    serialization.run_serialization([gda1], tspecs)

    m1, = serialization.run_deserialization(
        [jtu.create_global_mesh((2,), ('x'))],
        [P(None)],
        tspecs,
        [()],
        [np.float32]
    )

    for l in m1.local_shards:
      self.assertArraysEqual(l.data.to_py(), data.astype(np.float32))

  def test_spec_has_metadata(self):
    spec = {
        'a': {
            'b': 1,
            'c': 2,
        },
        'd': 3,
        'e': {
            'a': 2,
            'metadata': 3
        },
        'f': 4
    }
    self.assertTrue(serialization._spec_has_metadata(spec))
    self.assertTrue(
        serialization._spec_has_metadata({
            'driver': 'zarr',
            'kvstore': 'gfile',
            'metadata': {
                'chunks': 4,
                'shape': (32, 64)
            },
            'one_more': 'thing'
        }))

  def test_spec_has_no_metadata(self):
    spec = {
        'a': {
            'b': 1,
            'c': 2,
        },
        'd': 3,
        'e': {
            'a': 2,
        },
        'f': 4
    }
    self.assertFalse(serialization._spec_has_metadata(spec))

  def test_empty_spec_has_no_metadata(self):
    spec = {}
    self.assertFalse(serialization._spec_has_metadata(spec))

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
