# Copyright 2020 Google LLC
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

# flake8: noqa

import functools
import itertools
import os
import unittest
from unittest import SkipTest, skip, skipIf

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
from jax import test_util as jtu
from jax import vmap
from jax.experimental.general_map import gmap
from jax.lib import xla_bridge

from jax.config import config
config.parse_flags_with_absl()

ignore_gmap_warning = functools.partial(
  jtu.ignore_warning, message="gmap is an experimental.*")

# TODO(mattjj): de-duplicate setUpModule and tearDownModule with pmap_test.py
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


class GmapTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
    {"testcase_name": "_" + name, "schedule": schedule}
    for name, schedule in [
      ('seq', [('sequential', None)]),
      ('vec', [('vectorized', None)]),
      ('par', [('parallel', None)]),
      ('lim_vmap', [('sequential', None), ('vectorized', 2)]),
      ('soft_pmap', [('parallel', 2), ('vectorized', None)])
    ])
  @ignore_gmap_warning()
  @skipIf(not config.omnistaging_enabled,
          "vmap collectives only supported when omnistaging is enabled")
  def testBasicSchedules(self, schedule):
    def f(x):
      return jnp.dot(jnp.sin(x), x.T) * 4 + x

    x = jnp.arange(800).reshape((8, 10, 10))

    for loop, n in schedule:
      approx_n = x.shape[0] if n is None else n
      if loop == 'parallel' and approx_n > xla_bridge.device_count():
        raise SkipTest("this test requires more XLA devices")

    self.assertAllClose(vmap(f)(x), gmap(f, schedule)(x))

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
