# Copyright 2026 The JAX Authors.
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

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import parallel
from jax._src.lib import jaxlib_extension_version
import jax.numpy as jnp


config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


# AsyncCollectivesTest tests that async collectives (e.g., all_gather_start and
# all_gather_done) behave identically to their synchronous counterparts (e.g.,
# all_gather).
class AsyncCollectivesTest(jtu.JaxTestCase):

  def setUp(self):
    if jaxlib_extension_version < 427:
      self.skipTest('Requires jaxlib_extension_version >= 427')

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_all_gather(self, mesh):
    @jax.shard_map(out_specs=jax.P(None, reduced={'i'}))
    def f(x):
      return parallel.all_gather_start(x, 'i', tiled=True, to='reduced').done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.async_start', stablehlo)
    self.assertIn('stablehlo.all_gather', stablehlo)
    self.assertIn('stablehlo.async_done', stablehlo)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_psum(self, mesh):
    @jax.shard_map(out_specs=jax.P('i'))
    def f(x):
      return parallel.psum_start(x, 'i').done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.async_start', stablehlo)
    self.assertIn('stablehlo.all_reduce', stablehlo)
    self.assertIn('stablehlo.async_done', stablehlo)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_psum_scatter(self, mesh):
    @jax.shard_map(out_specs=jax.P('i'))
    def f(x):
      todo = parallel.psum_scatter_start(x, 'i', scatter_dimension=0, tiled=True)
      return todo.done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.async_start', stablehlo)
    self.assertIn('stablehlo.reduce_scatter', stablehlo)
    self.assertIn('stablehlo.async_done', stablehlo)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_all_to_all(self, mesh):
    @jax.shard_map(out_specs=jax.P('i'))
    def f(x):
      todo = parallel.all_to_all_start(x, 'i', split_axis=0, concat_axis=0, tiled=True)
      return todo.done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.async_start', stablehlo)
    self.assertIn('stablehlo.all_to_all', stablehlo)
    self.assertIn('stablehlo.async_done', stablehlo)

  # pbroadcast is only implemented on GPU. If you try to run this on another
  # platform, you'll get an error like this:
  #
  # > NotImplementedError: MLIR translation rule for primitive 'pbroadcast' not
  # > found for platform cpu
  @jtu.run_on_devices('gpu')
  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_pbroadcast(self, mesh):
    @jax.shard_map(out_specs=jax.P('i'))
    def f(x):
      return parallel.pbroadcast_start(x, 'i', source=0).done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.async_start', stablehlo)
    self.assertIn('stablehlo.collective_broadcast', stablehlo)
    self.assertIn('stablehlo.async_done', stablehlo)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_ppermute(self, mesh):
    @jax.jit
    @jax.shard_map(out_specs=jax.P('i'))
    def f(x):
      return parallel.ppermute_start(x, 'i', [(0, 1), (1, 0)]).done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.async_start', stablehlo)
    self.assertIn('stablehlo.collective_permute', stablehlo)
    self.assertIn('stablehlo.async_done', stablehlo)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_all_gather(self, mesh):
    self.skipTest('TODO(mwhittaker): Enable when lowering to HLO works')

    @jax.jit
    @jax.shard_map(
        out_specs=(
            jax.P(None, reduced={'i'}),
            jax.P(None, reduced={'i'}),
        )
    )
    def f(x):
      y_sync = jax.lax.all_gather(x, 'i', tiled=True, to='reduced')
      todo = parallel.all_gather_start(x, 'i', tiled=True, to='reduced')
      y_async = todo.done()
      return y_sync, y_async

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    y_sync, y_async = f(x)
    self.assertArraysEqual(y_sync, y_async)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_psum(self, mesh):
    self.skipTest('TODO(mwhittaker): Enable when lowering to HLO works')

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def f(x):
      y_sync = jax.lax.psum(x, 'i')
      y_async = parallel.psum_start(x, 'i').done()
      return y_sync, y_async

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    y_sync, y_async = f(x)
    self.assertArraysEqual(y_sync, y_async)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_psum_scatter(self, mesh):
    self.skipTest('TODO(mwhittaker): Enable when lowering to HLO works')

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def f(x):
      y_sync = jax.lax.psum_scatter(x, 'i', scatter_dimension=0, tiled=True)
      todo = parallel.psum_scatter_start(x, 'i', scatter_dimension=0, tiled=True)
      y_async = todo.done()
      return y_sync, y_async

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    y_sync, y_async = f(x)
    self.assertArraysEqual(y_sync, y_async)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_all_to_all(self, mesh):
    self.skipTest('TODO(mwhittaker): Enable when lowering to HLO works')

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def f(x):
      y_sync = jax.lax.all_to_all(x, 'i', split_axis=0, concat_axis=0, tiled=True)
      todo = parallel.all_to_all_start(x, 'i', split_axis=0, concat_axis=0, tiled=True)
      y_async = todo.done()
      return y_sync, y_async

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    y_sync, y_async = f(x)
    self.assertArraysEqual(y_sync, y_async)

  # pbroadcast is only implemented on GPU. If you try to run this on another
  # platform, you'll get an error like this:
  #
  # > NotImplementedError: MLIR translation rule for primitive 'pbroadcast' not
  # > found for platform cpu
  @jtu.run_on_devices('gpu')
  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_pbroadcast(self, mesh):
    self.skipTest('TODO(mwhittaker): Enable when lowering to HLO works')

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def f(x):
      y_sync = jax.lax.pbroadcast(x, 'i', source=0)
      todo = parallel.pbroadcast_start(x, 'i', source=0)
      y_async = todo.done()
      return y_sync, y_async

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    y_sync, y_async = f(x)
    self.assertArraysEqual(y_sync, y_async)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_ppermute(self, mesh):
    self.skipTest('TODO(mwhittaker): Enable when lowering to HLO works')

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def f(x):
      y_sync = jax.lax.ppermute(x, 'i', [(0, 1), (1, 0)])
      todo = parallel.ppermute_start(x, 'i', [(0, 1), (1, 0)])
      y_async = todo.done()
      return y_sync, y_async

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    y_sync, y_async = f(x)
    self.assertArraysEqual(y_sync, y_async)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
