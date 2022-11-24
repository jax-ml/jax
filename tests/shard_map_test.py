# Copyright 2022 The JAX Authors.
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
import functools
import unittest

from absl.testing import absltest
import jax
from jax import lax
from jax.config import config
from jax.experimental.maps import Mesh
from jax.experimental.pjit import PartitionSpec as P
from jax.interpreters import pxla
from jax._src import shard_map
from jax._src import sharding
from jax._src import ad_checkpoint
from jax._src import debugging
from jax._src import dispatch
from jax._src import test_util as jtu
from jax._src.lib import xla_bridge
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()

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


class ShardMapTest(absltest.TestCase):

  def test_identity(self):

    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))

    assert a.device_buffers[0].shape == (4, 2)

    def identity(x):
      return x

    @jax.jit
    def fwd(a):
      c = shard_map.shard_map(
          identity,
          mesh,
          in_pspecs=(P('z', ('x', 'y')),),
          out_pspecs=P('z', ('x', 'y')))(a)
      return c

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (4, 2))

  def test_all_gather(self):

    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))

    assert a.device_buffers[0].shape == (4, 2)

    def all_gather(x):
      return lax.all_gather(x, 'z', axis=0, tiled=True)

    @jax.jit
    def fwd(a):
      c = shard_map.shard_map(
          all_gather,
          mesh,
          in_pspecs=(P('z', ('x', 'y')),),
          out_pspecs=P(None, ('x', 'y')))(
              a)
      return c

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (8, 2))

  def test_matmul_partial(self):

    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))

    assert a.device_buffers[0].shape == (4, 4)

    def matmul_partial(a, b):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return c

    @jax.jit
    def fwd(a):
      c = shard_map.shard_map(
          matmul_partial,
          mesh,
          in_pspecs=(P('z', 'y'), P('y', None)),
          out_pspecs=P('z', None))(a, b)
      return c

    with mesh:
      c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (4, 8))

  def test_matmul_reduce_scatter(self):

    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))

    assert a.device_buffers[0].shape == (4, 4)

    def matmul_reduce_scatter(a, b):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return lax.psum_scatter(c, 'y', scatter_dimension=0, tiled=True)

    @jax.jit
    def fwd(a):
      c = shard_map.shard_map(
          matmul_reduce_scatter,
          mesh,
          in_pspecs=(P('z', 'y'), P('y', None)),
          out_pspecs=P(('z', 'y'), None))(a, b)
      return c

    with mesh:
      c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (2, 8))

  def test_collective_permute(self):

    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=('x'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    def collective_permute(a):
      axis_size = lax.psum(1, 'x')
      return lax.ppermute(
          a, 'x', perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])

    @jax.jit
    def fwd(a):
      c = shard_map.shard_map(
          collective_permute,
          mesh,
          in_pspecs=(P('x', None),),
          out_pspecs=P('x', None))(
              a)
      return c

    with mesh:
      c = fwd(a)
    self.assertTrue((c[1, :] == a[0, :]).all())

  def test_all_to_all(self):

    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=('x'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    def all_to_all(a):
      return lax.all_to_all(a, 'x', split_axis=1, concat_axis=1, tiled=True)

    @jax.jit
    def fwd(a):
      c = shard_map.shard_map(
          all_to_all,
          mesh,
          in_pspecs=(P('x', None),),
          out_pspecs=P(None, 'x'))(
              a)
      return c

    with mesh:
      c = fwd(a)

    assert (c == jnp.reshape(a.T, (1, 64))).all()

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
