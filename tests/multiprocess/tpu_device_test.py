# Copyright 2025 The JAX Authors.
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

"""Test TpuDevice API on multiprocess setup."""

import unittest

import jax
from jax._src import test_multiprocess as jt_multiprocess


class TpuDeviceTest(jt_multiprocess.MultiProcessTest):

  def test_coords(self):
    for device in jax.local_devices():
      coords = device.coords
      self.assertIsInstance(coords, list)
      self.assertLen(coords, 3)
      for coord in coords:
        self.assertIsInstance(coord, int)

  def test_core(self):
    for device in jax.local_devices():
      core = device.core_on_chip
      self.assertIsInstance(core, int)
      self.assertGreaterEqual(core, 0)
      self.assertLess(core, 2)

  def test_missing_attribute(self):
    for device in jax.local_devices():
      with self.assertRaises(AttributeError):
        device.gpu_type  # pylint: disable=pointless-statement

  def test_memory(self):
    for device in jax.devices():
      device_is_local = device.process_index == jax.process_index()
      self.assertLen(device.addressable_memories(), 3)
      hbm = device.addressable_memories()[0]
      self.assertEqual(
          hbm.process_index == device.process_index, device_is_local)
      self.assertEqual(hbm.platform, device.platform)
      self.assertEqual(hbm.kind, 'device')
      self.assertEqual(hbm, device.memory(hbm.kind))
      self.assertListEqual(hbm.addressable_by_devices(), [device])

      host = device.addressable_memories()[1]
      self.assertEqual(
          host.process_index == device.process_index, device_is_local)
      self.assertEqual(host.platform, device.platform)
      self.assertEqual(host.kind, 'pinned_host')
      self.assertEqual(host, device.memory(host.kind))
      self.assertListEqual(host.addressable_by_devices(), [device])

      with self.assertRaisesRegex(
          jax.errors.JaxRuntimeError,
          'INVALID_ARGUMENT: Could not find memory addressable by device TPU'
          ' v.* Device TPU v.* can address the following memory kinds: device,'
          ' pinned_host, unpinned_host. Got memory kind: gpu_hbm',
      ):
        device.memory('gpu_hbm')

  def test_host_memory_id(self):
    if jax.local_device_count() < 2:
      raise unittest.SkipTest('test requires 2 devices per process')
    self.assertGreaterEqual(len(jax.local_devices()), 2)
    host_0 = jax.local_devices()[0].memory('unpinned_host')
    host_1 = jax.local_devices()[1].memory('unpinned_host')
    self.assertNotEqual(id(host_0), id(host_1))


if __name__ == '__main__':
  jt_multiprocess.main()
