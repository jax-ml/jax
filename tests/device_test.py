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
import types

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib import _jax, jaxlib_extension_version
from jax._src.lib import xla_client as xc
import numpy as np

jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(2)


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

  def test_device_assignment_from_device_list(self):
    if jaxlib_extension_version < 526:
      self.skipTest('Requires jaxlib_extension_version >= 526')
    devices = jax.devices()
    dl_1d = xc.DeviceList(tuple(devices))
    da_1d = xc.DeviceAssignment.create(dl_1d, 1, len(devices))
    expected_1d = xc.DeviceAssignment.create(
        np.array([[d.id for d in devices]], dtype=np.int32)
    )
    self.assertEqual(da_1d.serialize(), expected_1d.serialize())

    if len(devices) >= 2:
      d0, d1 = devices[:2]
      six_devices = (d0, d1, d1, d0, d0, d1)
    else:
      six_devices = (devices[0],) * 6
    six_dl = xc.DeviceList(six_devices)
    for num_replicas, num_partitions in ((2, 3), (3, 2)):
      da_2d = xc.DeviceAssignment.create(
          device_list=six_dl,
          num_replicas=num_replicas,
          num_partitions=num_partitions,
      )
      expected_2d = xc.DeviceAssignment.create(
          np.array([d.id for d in six_devices], dtype=np.int32).reshape(
              num_replicas, num_partitions
          )
      )
      self.assertEqual(da_2d.serialize(), expected_2d.serialize())

    fake_devices = tuple(
        types.SimpleNamespace(id=i) for i in (10, 30, 20, 40, 60, 50)
    )
    fake_dl = xc.DeviceList(fake_devices)
    for num_replicas, num_partitions in ((2, 3), (3, 2)):
      da_duck = xc.DeviceAssignment.create(
          fake_dl, num_replicas, num_partitions
      )
      expected_duck = xc.DeviceAssignment.create(
          np.array([d.id for d in fake_devices], dtype=np.int32).reshape(
              num_replicas, num_partitions
          )
      )
      self.assertEqual(da_duck.serialize(), expected_duck.serialize())

  def test_device_assignment_from_device_list_invalid_shapes(self):
    if jaxlib_extension_version < 526:
      self.skipTest('Requires jaxlib_extension_version >= 526')
    devices = jax.devices()
    dl = xc.DeviceList(tuple(devices[i % len(devices)] for i in range(4)))
    for bad_dl, num_replicas, num_partitions in (
        (dl, 0, 4),
        (dl, 4, 0),
        (dl, -1, -4),
        (dl, 2, 3),
        (xc.DeviceList(()), 0, 0),
        (xc.DeviceList(()), 1, 1),
        (xc.DeviceList(()), 1 << 16, 1 << 16),
    ):
      with self.assertRaisesRegex(
          _jax.JaxRuntimeError, 'must be positive|does not match'
      ):
        xc.DeviceAssignment.create(bad_dl, num_replicas, num_partitions)

    with self.assertRaises(AttributeError):
      xc.DeviceAssignment.create(
          xc.DeviceList((types.SimpleNamespace(),)), 1, 1
      )
    with self.assertRaises(RuntimeError):
      xc.DeviceAssignment.create(
          xc.DeviceList((types.SimpleNamespace(id='bad'),)), 1, 1
      )
    with self.assertRaises(TypeError):
      xc.DeviceAssignment.create(_jax.DeviceList.__new__(_jax.DeviceList), 1, 1)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
