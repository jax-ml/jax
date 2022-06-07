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
      self.assertArraysEqual(arr._value(), global_data)


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
