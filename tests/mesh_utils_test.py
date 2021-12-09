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

import dataclasses
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
from jax import test_util
from jax.experimental import mesh_utils


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
  device_id = 0
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
        device_id += len(host_devices)
        process_index += 1
  return devices


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


if __name__ == '__main__':
  absltest.main()
