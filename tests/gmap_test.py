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
from functools import partial

import jax
import jax.numpy as jnp
from jax import test_util as jtu
from jax import vmap
from jax import lax
from jax.experimental.general_map import gmap, fake_resources, Mesh, mesh, xmap, A
from jax.lib import xla_bridge
from jax.util import curry, unzip2
from jax.interpreters import pxla

from jax.config import config
config.parse_flags_with_absl()

ignore_gmap_warning = functools.partial(
  jtu.ignore_warning, message=".*is an experimental.*")

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

@curry
def with_mesh(named_shape, f):
  def new_f(*args, **kwargs):
    axis_names, shape = unzip2(named_shape)
    size = np.prod(shape)
    local_devices = list(jax.local_devices())
    if len(local_devices) < size:
      raise SkipTest(f"Test requires {size} local devices")
    mesh_devices = np.array(local_devices[:size]).reshape(shape)
    with mesh(mesh_devices, axis_names):
      return f(*args, **kwargs)
  return new_f


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

  @ignore_gmap_warning()
  @skipIf(not config.omnistaging_enabled,
          "vmap collectives only supported when omnistaging is enabled")
  def testXMap(self):
    def f(a, b):
      return a + 2, b * 4
    with fake_resources(r1=4, r2=2, r3=5):
      fm = xmap(f,
                in_axes=[A({'x': 0, 'z': 1}), A({'y': 1})],
                out_axes=[A({'x': 1, 'z': 0}), A({'y': 0})],
                schedule=[
                  ('x', 'r1'),
                  ('x', 'r2'),
                  ('y', 'r1'),
                  ('z', 'r3'),
                  ('x', 'vectorize'),
                  ('y', 'vectorize'),
                ])
      a = jnp.arange(16*5*2).reshape((16, 5, 2))
      b = jnp.arange(6*16).reshape((6, 16))
      c, d = fm(a, b)
      self.assertAllClose(c, (a + 2).transpose((1, 0, 2)))
      self.assertAllClose(d, (b * 4).T)

  @ignore_gmap_warning()
  @skipIf(not config.omnistaging_enabled,
          "vmap collectives only supported when omnistaging is enabled")
  def testXMapCollectives(self):
    def f(a, b):
      return lax.psum(a + 2, 'x'), b * 4
    with fake_resources(r1=4, r2=2, r3=5):
      fm = xmap(f,
                in_axes=[A({'x': 0, 'z': 1}), A({'y': 1})],
                out_axes=[A({'z': 0}), A({'y': 0})],
                schedule=[
                  ('x', 'r1'),
                  ('x', 'r2'),
                  ('y', 'r1'),
                  ('z', 'r3'),
                  ('x', 'vectorize'),
                  ('y', 'vectorize'),
                ])
      a = jnp.arange(16*5*2).reshape((16, 5, 2))
      b = jnp.arange(6*16).reshape((6, 16))
      c, d = fm(a, b)
      self.assertAllClose(c, (a + 2).sum(0))
      self.assertAllClose(d, (b * 4).T)

  @ignore_gmap_warning()
  def testXMapMeshBasic(self):
    local_devices = list(jax.local_devices())
    if len(local_devices) < 4:
      raise SkipTest("Test requires at least 4 local devices")
    def f(a, b):
      return a * 2, b * 4
    devices = np.array(local_devices[:4]).reshape((2, 2))
    with mesh(devices, ('x', 'y')):
      fm = xmap(f,
                in_axes=[A({'a': 0, 'b': 1}), A({'c': 0})],
                out_axes=[A({'a': 0, 'b': 1}), A({'c': 0})],
                schedule=[
                  ('a', 'x'),
                  ('b', 'y'),
                  ('c', 'x'),
                  ('a', 'vectorize'),
                  ('b', 'vectorize'),
                ])
      ashape = (16, 8, 5)
      a = jnp.arange(np.prod(ashape)).reshape(ashape)
      bshape = (2, 7)
      b = jnp.arange(np.prod(bshape)).reshape(bshape)
      c, d = fm(a, b)
      self.assertAllClose(c, a * 2)
      self.assertAllClose(d, b * 4)

  @ignore_gmap_warning()
  def testXMapMeshCollectives(self):
    local_devices = list(jax.local_devices())
    if len(local_devices) < 4:
      raise SkipTest("Test requires at least 4 local devices")
    def f(a, b):
      return lax.psum(a * 2, 'a'), b * 4
    devices = np.array(local_devices[:4]).reshape((2, 2))
    with mesh(devices, ('x', 'y')):
      fm = xmap(f,
                in_axes=[A({'a': 0, 'b': 1}), A({'c': 0})],
                out_axes=[A({'b': 0}), A({'c': 0})],
                schedule=[
                  ('a', 'x'),
                  ('b', 'y'),
                  ('c', 'x'),
                  ('a', 'vectorize'),
                  ('b', 'vectorize'),
                ])
      ashape = (16, 8, 5)
      a = jnp.arange(np.prod(ashape)).reshape(ashape)
      bshape = (2, 7)
      b = jnp.arange(np.prod(bshape)).reshape(bshape)
      c, d = fm(a, b)
      self.assertAllClose(c, (a * 2).sum(0))
      self.assertAllClose(d, b * 4)

  @ignore_gmap_warning()
  @with_mesh([('x', 2)])
  def testXMapCompilationCache(self):
    def f(x):
      assert python_should_be_executing
      return x * 2
    fm = xmap(f,
              in_axes=[A({'a': 0})],
              out_axes=[A({'a': 0})],
              schedule=[('a', 'x'), ('a', 'vectorize')])
    x = np.arange(8).reshape((2, 2, 2))
    python_should_be_executing = True
    fm(x)
    python_should_be_executing = False
    fm(x)

  @ignore_gmap_warning()
  @with_mesh([('x', 2)])
  def testNestedXMapBasic(self):
    @partial(xmap, in_axes=[A({'a': 1})], out_axes=[A({'a': 0})],
             schedule=[('a', 'x')])
    def f(x):
      y = x * 2
      @partial(xmap, in_axes=[A({'b': 0})], out_axes=[A({'b': 1})],
               schedule=[('b', 'vectorize')])
      def h(y):
        return jnp.sin(y)
      return h(y)
    xshape = (4, 2, 5)
    x = jnp.arange(np.prod(xshape)).reshape(xshape)
    self.assertAllClose(f(x),
                        jnp.sin(x * 2).transpose((1, 2, 0)))

  @ignore_gmap_warning()
  @with_mesh([('x', 2), ('y', 3)])
  def testNestedXMapMesh(self):
    @partial(xmap, in_axes=[A({'a': 1})], out_axes=[A({'a': 0})],
             schedule=[('a', 'y')])
    def f(x):
      y = x * 2
      @partial(xmap, in_axes=[A({'b': 0})], out_axes=[A({'b': 1})],
               schedule=[('b', 'x')])
      def h(y):
        return jnp.sin(y)
      return h(y)
    xshape = (2, 3, 5)
    x = jnp.arange(np.prod(xshape)).reshape(xshape)
    y = f(x)
    self.assertAllClose(y, jnp.sin(x * 2).transpose((1, 2, 0)))
    # Make sure the op really ran accros a 2D mesh.
    self.assertEqual(y.sharding_spec.sharding,
                     (pxla.Chunked(3), None, None))
    self.assertEqual(y.sharding_spec.mesh_mapping,
                     (pxla.Replicated(2), pxla.ShardedAxis(0)))

  @ignore_gmap_warning()
  @with_mesh([('x', 2)])
  def testNestedXMapDifferentResources(self):
    @partial(xmap, in_axes=[A({'a': 0})], out_axes=[A({'a': 0})],
             schedule=[('a', 'x')])
    def f(x):
      with mesh(np.empty((), dtype=np.object), ()):
        @partial(xmap, in_axes=[A({'b': 0})], out_axes=[A({'b': 0})],
                 schedule=[('b', 'vectorize')])
        def h(x):
          return x
        return h(x)
    xshape = (2, 5, 6)
    x = jnp.arange(np.prod(xshape)).reshape(xshape)
    with self.assertRaisesRegex(RuntimeError,
                                "Changing the resource environment.*"):
      f(x)



if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
