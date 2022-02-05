# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for mesh utils."""

import collections
import dataclasses
from typing import Sequence

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from jax import test_util
from jax.experimental import mesh_utils
from jax.experimental.maps import Mesh


@dataclasses.dataclass
class MockTpuDevice:
  """Mock TPU device for testing."""
  platform: str
  device_kind: str
  process_index: int
  coords: Sequence[int]
  core_on_chip: int


def mock_devices(x, y, z, dev_kind, two_cores_per_chip):
  """Hard-coded reproduction of jax.devices() output on 8x8, 4x4x4."""
  devices = []
  process_index = 0
  for k in range(z):
    for j in range(0, y, 2):
      for i in range(0, x, 2):
        # Local 2x2 subgrid of chips, with 2 cores per chip.
        host_devices = [
            MockTpuDevice('tpu', dev_kind, process_index, (i, j, k), 0),
            MockTpuDevice('tpu', dev_kind, process_index, (i, j, k), 1),
            MockTpuDevice('tpu', dev_kind, process_index, (i + 1, j, k), 0),
            MockTpuDevice('tpu', dev_kind, process_index, (i + 1, j, k), 1),
            MockTpuDevice('tpu', dev_kind, process_index, (i, j + 1, k), 0),
            MockTpuDevice('tpu', dev_kind, process_index, (i, j + 1, k), 1),
            MockTpuDevice('tpu', dev_kind, process_index, (i + 1, j + 1, k), 0),
            MockTpuDevice('tpu', dev_kind, process_index, (i + 1, j + 1, k), 1),
        ]
        if two_cores_per_chip:
          # Only include core_on_chip = 0.
          host_devices = host_devices[::2]
        devices.extend(host_devices)
        # Simulate one process per host (1 host = 2x2x1 slice)
        process_index += 1
  _validate_mocked_process_indices(devices, two_cores_per_chip)
  return devices

# If this function raises, it's a bug in the test code!
def _validate_mocked_process_indices(devices, two_cores_per_chip):
  process_to_devices = collections.defaultdict(lambda: [])
  for d in devices:
    process_to_devices[d.process_index].append(d)

  for local_devices in process_to_devices.values():
    if two_cores_per_chip:
      # 4 devices per process
      assert len(local_devices) == 4, local_devices
    else:
      # 8 devices per process
      assert len(local_devices) == 8, local_devices
    # All devices have same z coord
    assert len(set(d.coords[2] for d in local_devices)) == 1, local_devices
    # All devices in a 2x2 subgrid
    min_coords = min(d.coords for d in local_devices)
    expected = set()
    for x, y in [(0,0), (0,1), (1,0), (1,1)]:
      expected.add((min_coords[0] + x, min_coords[1] + y, min_coords[2]))
    assert set(d.coords for d in local_devices) == expected, local_devices

def mock_8x8_devices():
  """Hard-coded reproduction of jax.devices() output on 8x8."""
  return mock_devices(8, 8, 1, 'TPU v3', False)


def mock_2x2x1_devices(two_cores_per_chip):
  """Hard-coded reproduction of jax.devices() output on 2x2x1."""
  return mock_devices(2, 2, 1, 'TPU v4', two_cores_per_chip)


def mock_2x2x4_devices(two_cores_per_chip):
  """Hard-coded reproduction of jax.devices() output on 2x2x4."""
  return mock_devices(2, 2, 4, 'TPU v4', two_cores_per_chip)


def mock_4x4x4_devices(two_cores_per_chip):
  """Hard-coded reproduction of jax.devices() output on 4x4x4."""
  return mock_devices(4, 4, 4, 'TPU v4', two_cores_per_chip)


def mock_4x4x8_devices(two_cores_per_chip):
  """Hard-coded reproduction of jax.devices() output on 4x4x4."""
  return mock_devices(4, 4, 8, 'TPU v4', two_cores_per_chip)


def mock_8x8x8_devices(two_cores_per_chip):
  """Hard-coded reproduction of jax.devices() output on 8x8x8."""
  return mock_devices(8, 8, 8, 'TPU v4', two_cores_per_chip)


def mock_4x8x8_devices(two_cores_per_chip):
  """Hard-coded reproduction of jax.devices() output on 4x8x8."""
  return mock_devices(4, 8, 8, 'TPU v4', two_cores_per_chip)


def mock_4x8x16_devices(two_cores_per_chip):
  """Hard-coded reproduction of jax.devices() output on 4x8x16."""
  return mock_devices(4, 8, 16, 'TPU v4', two_cores_per_chip)


def mock_8x8x16_devices(two_cores_per_chip):
  """Hard-coded reproduction of jax.devices() output on 8x8x16."""
  return mock_devices(8, 8, 16, 'TPU v4', two_cores_per_chip)


class PartitioningTest(test_util.JaxTestCase):

  @parameterized.named_parameters(
      ('2x2x1_t', mock_2x2x1_devices, True, (2, 2, 1, 1)),
      ('2x2x1_f', mock_2x2x1_devices, False, (2, 2, 1, 2)),
      ('8x8x16_t', mock_8x8x16_devices, True, (8, 8, 16, 1)),
      ('8x8x16_f', mock_8x8x16_devices, False, (8, 8, 16, 2)),
  )
  def test_bounds_from_last_device(self, devices, two_cores_per_chip,
                                   expected_bounds):
    self.assertEqual(
        mesh_utils._bounds_from_last_device(devices(two_cores_per_chip)[-1]),
        expected_bounds)

  @parameterized.named_parameters(
      ('4x4x4', mock_4x4x4_devices, (4, 4, 4)),
      ('4x4x8', mock_4x4x8_devices, (4, 4, 8)),
      ('8x8x8', mock_8x8x8_devices, (8, 8, 8)),
      ('8x8x16', mock_8x8x16_devices, (8, 8, 16)),
  )
  def test_jax_devices_order_normalized(self, devices, expected_shape):
    jax_local_devices_from_process_0 = mock_2x2x1_devices(True)
    jax_devices = devices(True)
    normalized = mesh_utils._jax_devices_order_normalized(
        jax_local_devices_from_process_0, jax_devices)
    self.assertEqual(normalized.shape, expected_shape)
    x, y, z = expected_shape
    # major_to_minor: x, y, z
    for i in range(x):
      for j in range(y):
        for k in range(z):
          self.assertEqual(normalized[i, j, k].coords, (i, j, k))

  @parameterized.named_parameters(
      ('2x2x1', mock_2x2x1_devices, [1, 1, 4], ((), (2,), (0, 1))),
      ('2x2x4', mock_2x2x4_devices, [1, 4, 4], ((), (2,), (0, 1))),
      ('4x4x4', mock_4x4x4_devices, [1, 16, 4], ((), (1, 2), (0,))),
      ('4x4x8a', mock_4x4x8_devices, [1, 16, 8], ((), (0, 1), (2,))),
      ('4x4x8b', mock_4x4x8_devices, [1, 8, 16], ((), (2,), (0, 1))),
      ('4x4x8c', mock_4x4x8_devices, [16, 8, 1], ((0, 1), (2,), ())),
      ('4x8x8', mock_4x8x8_devices, [1, 32, 8], ((), (0, 2), (1,))),
      ('8x8x8', mock_8x8x8_devices, [1, 64, 8], ((), (1, 2), (0,))),
      ('8x8x16', mock_8x8x16_devices, [1, 64, 16], ((), (0, 1), (2,))),
  )
  def test_create_device_mesh_for_tpu_v4(self, devices, mesh_shape,
                                         expected_assignment):
    jax_local_devices_from_process_0 = mock_2x2x1_devices(True)
    jax_devices = devices(True)
    physical_mesh = mesh_utils._jax_devices_order_normalized(
        jax_local_devices_from_process_0, jax_devices)
    _, assignment = mesh_utils._create_device_mesh_for_tpu_v4(
        physical_mesh, mesh_shape)
    self.assertEqual(assignment, expected_assignment)

  def _assert_contiguous_submeshes(self, global_device_mesh):
    global_mesh = Mesh(global_device_mesh, list(range(global_device_mesh.ndim)))
    max_process_index = max(d.process_index
                            for d in global_device_mesh.flatten())
    for p_idx in range(max_process_index + 1):
      # Raises an error if non-contiguous
      global_mesh._local_mesh(p_idx)

  def test_create_contiguous_submeshes_for_tpu_v4(self):
    v4 = mesh_utils._TPU_V4
    process_0_devices = mock_2x2x1_devices(True)
    for topology, mesh_shapes in mesh_utils._TRANSPOSE_TRICKS.items():
      logging.vlog(1, "topology: %s", topology)
      devices = mock_devices(topology[0], topology[1], topology[2], v4,
                             two_cores_per_chip=True)
      for mesh_shape in mesh_shapes:
        logging.vlog(1, "  mesh_shape: %s", mesh_shape)
        mesh = mesh_utils._create_device_mesh(
            process_0_devices, devices, v4, mesh_shape,
            contiguous_submeshes=True)
        self._assert_contiguous_submeshes(mesh)

  def test_create_contiguous_submeshes_errors(self):
    process_0_devices = mock_2x2x1_devices(True)
    v4 = mesh_utils._TPU_V4

    topology = (4, 4, 8)
    mesh_shape = (1, 16, 8)
    devices = mock_devices(topology[0], topology[1], topology[2], v4,
                             two_cores_per_chip=True)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "create_device_mesh cannot create contiguous submeshes for "
        "physical mesh topology (4, 4, 8)"):
      mesh_utils._create_device_mesh(
          process_0_devices, devices, v4, mesh_shape,
          contiguous_submeshes=True)

    topology = (4, 8, 8)
    mesh_shape = (1, 128, 2)
    devices = mock_devices(topology[0], topology[1], topology[2], v4,
                             two_cores_per_chip=True)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "create_device_mesh cannot create contiguous submeshes for mesh_shape "
        "(1, 128, 2) and physical mesh topology (4, 8, 8). "
        "Available mesh_shapes: [(1, 64, 4), (1, 4, 64), (64, 4), (4, 64)]"):
      mesh_utils._create_device_mesh(
          process_0_devices, devices, v4, mesh_shape,
          contiguous_submeshes=True)


if __name__ == '__main__':
  absltest.main()
