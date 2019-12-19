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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from unittest import SkipTest

from absl.testing import absltest
import numpy as onp

import jax
import jax.numpy as np
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

  def test_computation_follows_data(self):
    if len(jax.devices()) < 2:
      raise SkipTest("test requires multiple devices")

    x = jax.device_put(1, jax.devices()[0])
    y = jax.device_put(2, jax.devices()[0])
    z = x + y
    self.assertEqual(z, 3)
    self.assertEqual(z.device_buffer.device(), jax.devices()[0])

    x = jax.device_put(1, jax.devices()[1])
    y = jax.device_put(2, jax.devices()[1])
    z = x + y
    self.assertEqual(z, 3)
    self.assertEqual(z.device_buffer.device(), jax.devices()[1])

    x = jax.device_put(1, jax.devices()[1])
    y = 4
    z = x + y
    self.assertEqual(z, 5)
    self.assertEqual(z.device_buffer.device(), jax.devices()[1])

    x = jax.device_put(1, jax.devices()[1])
    y = np.ones(3)
    z = x + y
    self.assertAllClose(z, 1 + onp.ones(3), check_dtypes=False)
    self.assertEqual(z.device_buffer.device(), jax.devices()[1])

    x = jax.device_put(1, jax.devices()[0])
    y = jax.device_put(2, jax.devices()[1])
    self.assertRaisesRegex(
        ValueError,
        "primitive arguments must be colocated on the same device",
        lambda: x + y)

    x = jax.device_put(1, jax.devices()[1])
    y = x.reshape((1, 1))
    self.assertEqual(y.device_buffer.device(), jax.devices()[1])

  def test_primitive_compilation_cache(self):
    if len(jax.devices()) < 2:
      raise SkipTest("test requires multiple devices")

    x = jax.device_put(1, jax.devices()[1])

    with jtu.count_primitive_compiles() as count:
      y = lax.add(x, x)
      z = lax.add(y, y)

    self.assertEqual(count[0], 1)
    self.assertEqual(y.device_buffer.device(), jax.devices()[1])
    self.assertEqual(z.device_buffer.device(), jax.devices()[1])


if __name__ == '__main__':
  absltest.main()
