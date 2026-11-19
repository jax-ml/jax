# Copyright 2024 The JAX Authors.
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

import gc

from absl.testing import absltest
import jax
from jax._src.lib import _jax, jaxlib_extension_version
from jax._src import test_util as jtu
from jax._src.lib import xla_client as xc

jax.config.parse_flags_with_absl()


class DeviceTest(jtu.JaxTestCase):

  def test_uninitialized_device_list(self):
    if jaxlib_extension_version < 517:
      self.skipTest('Requires jaxlib_extension_version >= 517')
    dl = _jax.DeviceList.__new__(_jax.DeviceList)
    with self.assertRaises(TypeError):
      _ = dl.addressable_device_list
    with self.assertRaises(TypeError):
      _ = dl.memory_kinds
    with self.assertRaises(TypeError):
      _ = dl.default_memory_kind

  def test_uninitialized_device_gc_clear(self):
    if jaxlib_extension_version < 517:
      self.skipTest('Requires jaxlib_extension_version >= 517')
    dev = _jax.Device.__new__(_jax.Device)
    cycle = [dev]
    cycle.append(cycle)
    del dev, cycle
    gc.collect()

  def test_repr(self):
    device = jax.devices()[0]

    if jtu.is_device_cuda():
      self.assertEqual(device.platform, 'gpu')
      self.assertEqual(repr(device), 'CudaDevice(id=0)')
    elif jtu.is_device_rocm():
      self.assertEqual(device.platform, 'gpu')
      self.assertEqual(repr(device), 'RocmDevice(id=0)')
    elif jtu.test_device_matches(['tpu']):
      self.assertEqual(device.platform, 'tpu')
      self.assertEqual(
          repr(device),
          'TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)',
      )
    elif jtu.is_device_oneapi():
      self.assertEqual(device.platform, 'gpu')
      self.assertEqual(repr(device), 'OneapiDevice(id=0)')
    elif jtu.test_device_matches(['cpu']):
      self.assertEqual(device.platform, 'cpu')
      self.assertEqual(repr(device), 'CpuDevice(id=0)')

  def test_str(self):
    device = jax.devices()[0]

    if jtu.is_device_cuda():
      self.assertEqual(str(device), 'cuda:0')
    elif jtu.is_device_rocm():
      self.assertEqual(str(device), 'rocm:0')
    elif jtu.test_device_matches(['tpu']):
      self.assertEqual(str(device), 'TPU_0(process=0,(0,0,0,0))')
    elif jtu.test_device_matches(['oneapi']):
      self.assertEqual(str(device), 'oneapi:0')
    elif jtu.test_device_matches(['cpu']):
      self.assertEqual(str(device), 'cpu:0')

  def test_empty_device_list_is_fully_addressable(self):
    if jaxlib_extension_version < 518:
      self.skipTest('Requires jaxlib_extension_version >= 518')
    dl = xc.DeviceList(())
    self.assertLen(dl, 0)
    self.assertEqual(dl.process_indices, set())
    self.assertTrue(dl.is_fully_addressable)
    self.assertEqual(dl.addressable_device_list, dl)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
