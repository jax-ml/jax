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
from jax import lax
from jax.experimental.general_map import gmap
from jax.lib import xla_bridge
from jax.util import curry

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


@curry
def skip_insufficient_devices(axis_size, fun):
  @functools.wraps(fun)
  def wrapper(*args, schedule, **kwargs):
    for loop, n in schedule:
      approx_n = axis_size if n is None else n
      if loop == 'parallel' and approx_n > xla_bridge.device_count():
        raise SkipTest("this test requires more XLA devices")
    return fun(*args, schedule=schedule, **kwargs)
  return wrapper

@curry
def check_default_schedules(cond, fun):
  schedules = [
    ('seq', [('sequential', None)]),
    ('vec', [('vectorized', None)]),
    ('par', [('parallel', None)]),
    ('lim_vmap', [('sequential', None), ('vectorized', 2)]),
    ('soft_pmap', [('parallel', 2), ('vectorized', None)])
  ]
  schedules = [s for s in schedules if cond(s[1])]
  return parameterized.named_parameters(
    {"testcase_name": "_" + name, "schedule": schedule}
    for name, schedule in schedules)(fun)


class GmapTest(jtu.JaxTestCase):

  @check_default_schedules(lambda _: True)
  @skip_insufficient_devices(8)
  @ignore_gmap_warning()
  @skipIf(not config.omnistaging_enabled,
          "vmap collectives only supported when omnistaging is enabled")
  def testBasicSchedules(self, schedule):
    def f(x):
      return jnp.dot(jnp.sin(x), x.T) * 4 + x

    x = jnp.arange(800).reshape((8, 10, 10))

    self.assertAllClose(gmap(f, schedule)(x), vmap(f)(x))

  @check_default_schedules(lambda s: not any(c[0] == 'sequential' for c in s))
  @skip_insufficient_devices(8)
  @ignore_gmap_warning()
  @skipIf(not config.omnistaging_enabled,
          "vmap collectives only supported when omnistaging is enabled")
  def testAxisName(self, schedule):
    def f(x):
      return x - lax.psum(x, 'i')
    x = jnp.arange(8)
    self.assertAllClose(gmap(f, schedule, axis_name='i')(x),
                        vmap(f, axis_name='i')(x))

  @ignore_gmap_warning()
  @skipIf(not config.omnistaging_enabled,
          "vmap collectives only supported when omnistaging is enabled")
  def testAxisName2d(self):
    def f(x):
      return x - lax.psum(x, 'i') + lax.pmax(x, 'j')
    x = jnp.arange(8 * 8).reshape((8, 8))
    s = [('vectorized', None)]
    self.assertAllClose(gmap(gmap(f, s, axis_name='i'), s, axis_name='j')(x),
                        vmap(vmap(f, axis_name='i'), axis_name='j')(x))

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
