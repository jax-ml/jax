# Copyright 2023 The JAX Authors.
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
from functools import partial
import os
import unittest

from absl.testing import absltest
import numpy as np

import jax
from jax import lax
from jax.config import config
from jax.experimental.maps import Mesh
from jax.experimental.pjit import PartitionSpec as P
from jax._src import test_util as jtu
from jax._src.lib import xla_bridge
import jax.numpy as jnp

from jax.experimental.shard_map import shard_map

config.parse_flags_with_absl()

# Helper for some tests.
def create_inputs(a_sharding, b_sharding):
  x, y, z = 2, 2, 2  # pylint: disable=invalid-name
  devices = np.array(jax.devices()[:x * y * z]).reshape((x, y, z))
  mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
  b, e, f = 8, 8, 8  # pylint: disable=invalid-name
  m1 = jax.device_put(
      jnp.arange(b * e).reshape((b, e)),
      jax.sharding.NamedSharding(mesh, a_sharding))
  m2 = jax.device_put(
      jnp.arange(e * f).reshape((e, f)),
      jax.sharding.NamedSharding(mesh, b_sharding))
  return mesh, m1, m2

# Run all tests with 8 CPU devices.
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

  if len(jax.devices()) < 8:
    raise unittest.SkipTest("tests require 8 devices")
  if not jax.config.jax_array:
    raise unittest.SkipTest("test requires jax_array")

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


class ShardMapTest(jtu.JaxTestCase):

  def test_identity(self):
    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))
    assert a.device_buffers[0].shape == (4, 2)

    def identity(x):
      return x

    @jax.jit
    def fwd(a):
      c = shard_map(
          lambda x: x,
          mesh,
          in_specs=(P('z', ('x', 'y')),),
          out_specs=P('z', ('x', 'y')))(a)
      return c

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (4, 2))

  def test_all_gather(self):
    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))
    assert a.device_buffers[0].shape == (4, 2)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', ('x', 'y')),), out_specs=P(None, ('x', 'y')))
    def fwd(a):
      return lax.all_gather(a, 'z', axis=0, tiled=True)

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (8, 2))

  def test_matmul_partial(self):
    raise unittest.SkipTest("invalid replication asserted by out_spec?")

    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))
    assert a.device_buffers[0].shape == (4, 4)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', 'y'), P('y', None)), out_specs=P('z', None))
    def fwd(a):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return c

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (4, 8))

  def test_matmul_reduce_scatter(self):
    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))
    assert a.device_buffers[0].shape == (4, 4)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', 'y'), P('y', None)),
             out_specs=P(('z', 'y'), None))
    def fwd(a, b):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return lax.psum_scatter(c, 'y', scatter_dimension=0, tiled=True)

    c = fwd(a, b)
    self.assertEqual(c.device_buffers[0].shape, (2, 8))

  def test_collective_permute(self):
    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=('x'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(P('x', None),),
             out_specs=P('x', None))
    def fwd(a):
      axis_size = lax.psum(1, 'x')
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(a, 'x', perm=perm)

    c = fwd(a)
    self.assertAllClose(c[1, :], a[0, :])

  def test_all_to_all(self):
    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=('x'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x', None),), out_specs=P(None, 'x'))
    def fwd(a):
      return lax.all_to_all(a, 'x', split_axis=1, concat_axis=1, tiled=True)

    c = fwd(a)
    assert (c == jnp.reshape(a.T, (1, 64))).all()

  def test_eager_repr(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    s = None

    @partial(shard_map, mesh=mesh, in_specs=P('x', 'y'), out_specs=P('x', 'y'))
    def f(x):
      nonlocal s
      s = str(x)
      return x
    _ = f(np.arange(8 * 8.).reshape(8, 8))

    self.assertIsInstance(s, str)
    self.assertIn('at mesh coordinates', s)

  def test_jvp_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))
    args = np.arange(4 * 4.).reshape(4, 4),
    jtu.check_grads(g, args, 2, ['fwd'])
    jtu.check_grads(jax.jit(g), args, 2, ['fwd'])

  def test_linearize_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))
    x = np.arange(4 * 4.).reshape(4, 4)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_linearize_basic_repres(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jax.lax.sin(jax.lax.cos(x)), mesh,
                  in_specs=(P('x',),), out_specs=P('x',))
    x = np.arange(4.)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_linearize_basic_repres_jit(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x',),), out_specs=P('x',))
    x = np.arange(4.)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_replication_checker_eager(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = np.arange(8 * 8.).reshape(8, 8)

    def f(x):
      return 2 * x
    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)

    with self.assertRaisesRegex(ValueError, 'statically inferred'):
      g(x)

    def f2(x):
      return jax.lax.psum(x, 'x')
    def g2(x):
      return shard_map(f2, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
    _ = g2(x)  # doesn't crash

  def test_replication_checker_jit(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = np.arange(8 * 8.).reshape(8, 8)

    def f(x):
      return 2 * x
    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)

    with self.assertRaisesRegex(ValueError, 'statically inferred'):
      jax.jit(g)(x)

    def f2(x):
      return jax.lax.psum(x, 'x')
    def g2(x):
      return shard_map(f2, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
    _ = jax.jit(g2)(x)  # doesn't crash

  def test_process_env_traces(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))
    x = np.arange(8.)

    def g(x):
      y = (3. * x).sum()
      z = shard_map(lambda x: 2 * x * y, mesh,
                    in_specs=(P('x'),), out_specs=P('x'))(np.arange(8.))
      return z

    jtu.check_grads(g, (x,), modes=['fwd'], order=2)

  def test_eager_control_flow(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = jnp.arange(2 * 2.).reshape(2, 2)

    def f(x):
      y = jax.lax.psum(x, ('x', 'y'))
      if y < 0:
        return x
      else:
        return -x

    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))(x)
    y = g(x)
    self.assertAllClose(y, -x, check_dtypes=False)

  def test_outer_jit_detects_shard_map_mesh(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    f = shard_map(lambda x: x.reshape(1, *x.shape), mesh, P(), P('x'))
    _ = jax.jit(f)(jnp.array(2.0))  # doesnt crash

  def test_vmap_basic(self):
    if jax.config.jax_jit_pjit_api_merge:
      raise unittest.SkipTest("pjit batcher error")  # TODO(mattjj)

    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = jnp.arange(8 * 8.).reshape(8, 8)

    def g(x):
      return shard_map(lambda x: 2. * x, mesh,
                       in_specs=P('y'), out_specs=P('y'))(x)
    y = jax.vmap(g, axis_name='x')(x)
    self.assertAllClose(y, 2 * x, check_dtypes=False)

  def test_tree_prefix_error(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    @partial(shard_map, mesh=mesh, in_specs=([P('x', 'y')],), out_specs=P('x', 'y'))
    def f(x):
      return x

    x = jnp.arange(8 * 8.).reshape(8, 8)
    with self.assertRaisesRegex(ValueError, r'shard_map in_specs\[0\]'):
       f([x, x])

  def test_rank_errors(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    def foo():
      return {'hi': [3.]}

    with self.assertRaisesRegex(ValueError, 'which has length 1'):
      shard_map(foo, mesh=mesh, in_specs=(), out_specs={'hi': P('x')})()

    with self.assertRaisesRegex(ValueError, 'which has length 1'):
      jax.jit(lambda: shard_map(foo, mesh=mesh,
                                in_specs=(), out_specs={'hi': P('x')})())()

    with self.assertRaisesRegex(ValueError, 'which has rank 0'):
      shard_map(foo, mesh=mesh, in_specs=({'hi': P('x')},), out_specs=())(
          {'hi': [jnp.array(3.)]})

  def test_reverse_mode_ad(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x',), P(None)), out_specs=P('x',))
    def f(x, y):
      return jnp.sin(x) + 3 + jnp.tan(2.) * jnp.cos(x) + y

    x = jnp.arange(8.) / 10.
    y = jnp.arange(4.) / 10.
    jtu.check_grads(f, (x, y), modes=['fwd', 'rev'], order=2)

  def test_post_process(self):
    # JVPTrace.post_process_shard_map and JaxprTrace.post_process_shard_map
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    def f(x):
      @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
      def g(y):
        return jnp.sin(y) * jnp.sin(x).sum()
      return g(jnp.arange(8.))

    x = jnp.arange(8.)
    _, f_lin = jax.linearize(f, x)
    y_dot = f_lin(x)

    y_dot_expected = jnp.sin(jnp.arange(8.)) * (jnp.cos(x) * x).sum()
    self.assertAllClose(y_dot, y_dot_expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu")
  def test_axis_index(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=P('x'))
    def f():
      return jax.lax.axis_index('x')[None]

    x = f()
    self.assertAllCLose(x, jnp.arange(4), check_dtypes=False)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
