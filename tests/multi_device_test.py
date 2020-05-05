# Copyright 2019 Google LLC
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


import os
from unittest import SkipTest

from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax import lax
from jax import test_util as jtu
from jax.lib import xla_bridge
from jax.interpreters import xla

from jax.config import config
config.parse_flags_with_absl()

prev_xla_flags = None


# Run all tests with 8 CPU devices.
def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


class MultiDeviceTest(jtu.JaxTestCase):

  def get_devices(self):
    if len(jax.devices()) < 2:
      raise SkipTest("test requires multiple devices")
    return jax.devices()

  def assert_committed_to_device(self, data, device):
    """Asserts that the data is committed to the device."""
    self.assertIsNotNone(data._device)
    self.assertEqual(data.device_buffer.device(), device)

  def assert_uncommitted_to_device(self, data, device):
    """Asserts that the data is on the device but not committed to it."""
    self.assertIsNone(data._device)
    self.assertEqual(data.device_buffer.device(), device)

  def test_computation_follows_data(self):
    devices = self.get_devices()

    # By default, computation is placed (uncommitted) on device 0
    x = jnp.ones(2)
    self.assert_uncommitted_to_device(x, devices[0])

    # Computing only with uncommitted data, will perform the computation
    # on the same device 0, and the result is uncommitted
    y = jnp.sin(x + x * 1) + 2
    self.assert_uncommitted_to_device(y, devices[0])

    # We can use device_put to both place the data on the requested device,
    # and to commit it there.
    z = jax.device_put(1, devices[1])
    self.assert_committed_to_device(z, devices[1])

    # A computation with some committed inputs will happen on the committed
    # device. Uncommitted data may be moved to the committed device.
    u = z + x  # z is committed, and x is uncommitted
    self.assert_committed_to_device(u, devices[1])

    # A computation with inputs committed to multiple devices will result
    # in an error
    with self.assertRaisesRegex(ValueError,
                                "primitive arguments must be colocated on the same device"):
      jax.device_put(x, devices[2]) + jax.device_put(x, devices[3])

    # A jitted-computation without a device specification behave like any
    # other primitive
    jit_add = jax.jit(lambda a, b: a + b)
    self.assert_uncommitted_to_device(jit_add(1, 2), devices[0])
    self.assert_committed_to_device(jit_add(1, jax.device_put(2, devices[2])),
                                    devices[2])
    with self.assertRaisesRegex(ValueError,
                                "primitive arguments must be colocated on the same device"):
      jit_add(jax.device_put(x, devices[2]), jax.device_put(x, devices[3]))

    # Even jit of trivial computations leaves the result uncommitted
    x_uncommitted = jnp.array([1, 2, 3])
    y = jax.jit(lambda x: x)(x_uncommitted)
    self.assert_uncommitted_to_device(y, devices[0])

    z1, z2 = jax.jit(lambda x: (x, x))(x_uncommitted)
    self.assert_uncommitted_to_device(z1, devices[0])
    self.assert_uncommitted_to_device(z2, devices[0])
    self.assertIs(z1, z2)

    x2_uncommitted = jnp.array([2, 3])
    z1, z2, z3 = jax.jit(lambda x, y: (y, 1, x))(x_uncommitted, x2_uncommitted)
    self.assert_uncommitted_to_device(z1, devices[0])
    self.assertIs(z2, 1)
    self.assert_uncommitted_to_device(z3, devices[0])


    # A jitted computation with an device specification behaves as if the
    # arguments are first device_put to the specified device. The result
    # will be committed on the specified.
    # The `device` parameter is experimental, and subject to change.
    jit_add_on4 = jax.jit(lambda a, b: a + b, device=devices[4])
    self.assert_committed_to_device(jit_add_on4(1, 2), devices[4])
    self.assert_committed_to_device(jit_add_on4(1, jax.device_put(2, devices[2])),
                                    devices[4])
    self.assert_committed_to_device(jit_add_on4(jax.device_put(x_uncommitted, devices[2]),
                                                jax.device_put(x_uncommitted, devices[3])),
                                    devices[4])

  def test_primitive_compilation_cache(self):
    devices = self.get_devices()

    x = jax.device_put(1, devices[1])

    with jtu.count_primitive_compiles() as count:
      y = lax.add(x, x)
      z = lax.add(y, y)

    self.assertEqual(count[0], 1)
    self.assert_committed_to_device(y, devices[1])
    self.assert_committed_to_device(z, devices[1])

  def test_device_put(self):
    devices = self.get_devices()

    # test device_put on regular values
    x = jax.device_put(1, device=devices[0])
    self.assert_committed_to_device(x, devices[0])

    # test device_put on its own output
    y = jax.device_put(x, device=devices[1])
    self.assert_committed_to_device(y, devices[1])

    # test device_put on lazy values
    x = jax.device_put(jnp.zeros(2), device=devices[0])
    self.assert_committed_to_device(x, devices[0])

    y = jax.device_put(x, device=devices[1])
    self.assert_committed_to_device(y, devices[1])

    x = jax.device_put(jnp.zeros(2), device=devices[1])
    self.assert_committed_to_device(x, devices[1])

    # device_put with device=None does not change placement
    x = jax.device_put(jnp.zeros(2))
    self.assert_uncommitted_to_device(x, devices[0])

    x = jax.device_put(jax.device_put(2, device=devices[1]))
    self.assert_committed_to_device(x, devices[1])


  def test_closed_over_values_device_placement(self):
    # see https://github.com/google/jax/issues/1431
    devices = self.get_devices()

    def f(): return lax.add(3., 4.)
    self.assertIsInstance(f(), xla.DeviceArray)
    self.assert_uncommitted_to_device(f(), devices[0])
    self.assert_uncommitted_to_device(jax.jit(f)(), devices[0])
    self.assert_committed_to_device(jax.jit(f, device=devices[1])(),
                                    devices[1])

  def test_reshape(self):
    devices = self.get_devices()
    # computation follows data explicitly placed on device 1
    x = jax.device_put(1, devices[1])
    y = x.reshape((1, 1))
    self.assert_committed_to_device(y, devices[1])
    z = y.reshape((1, 1))
    self.assert_committed_to_device(z, devices[1])

  def test_broadcast(self):
    devices = self.get_devices()

    z = 1 + jnp.ones((2, 3))
    self.assert_uncommitted_to_device(z, devices[0])
    y = jax.device_put(1, devices[2]) + jnp.ones((2, 3))
    self.assert_committed_to_device(y, devices[2])

  def test_transpose(self):
    devices = self.get_devices()

    x = jnp.ones((2, 3))
    self.assert_uncommitted_to_device(x, devices[0])

    y = lax.transpose(x, (1, 0))
    self.assert_uncommitted_to_device(y, devices[0])
    z = lax.transpose(jax.device_put(x, devices[2]), (1, 0))
    self.assert_committed_to_device(z, devices[2])


if __name__ == '__main__':
  absltest.main()
