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

  def test_device_list_equal_tuple(self):
    if jaxlib_extension_version < 526:
      self.skipTest('Requires jaxlib_extension_version >= 526')

    def assert_equal(a, b):
      self.assertTrue(a == b)
      self.assertTrue(b == a)
      self.assertFalse(a != b)
      self.assertFalse(b != a)

    def assert_not_equal(a, b):
      self.assertFalse(a == b)
      self.assertFalse(b == a)
      self.assertTrue(a != b)
      self.assertTrue(b != a)

    devices = tuple(jax.devices())
    assert_equal(xc.DeviceList(devices), devices)

    client = xc.make_cpu_client(num_devices=3)
    d0, d1, d2 = client.devices()
    device_list = xc.DeviceList((d0, d1))

    assert_equal(device_list, (d0, d1))
    assert_equal(xc.DeviceList((d0, d0)), (d0, d0))
    assert_not_equal(device_list, (d1, d0))
    assert_not_equal(device_list, ())
    assert_not_equal(device_list, (d0,))
    assert_not_equal(device_list, (d0, d1, d2))
    assert_not_equal(device_list, (d0, d0))
    assert_not_equal(device_list, (0, 1))
    assert_not_equal(device_list, (d0, None))
    assert_not_equal(device_list, [d0, d1])
    assert_not_equal(device_list, None)

    empty_device_list = xc.DeviceList(())
    assert_equal(empty_device_list, ())
    assert_not_equal(empty_device_list, (d0,))
    assert_not_equal(empty_device_list, [])

    other_client = xc.make_cpu_client(num_devices=2)
    other_d0, other_d1 = other_client.devices()
    assert_not_equal(device_list, (other_d0, other_d1))
    assert_not_equal(device_list, (d0, other_d1))

    multi_client_device_list = xc.DeviceList((d0, other_d0))
    assert_equal(multi_client_device_list, (d0, other_d0))
    assert_not_equal(multi_client_device_list, (d0, d1))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
