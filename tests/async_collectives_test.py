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
import jax.numpy as jnp


config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


# AsyncCollectivesTest tests that async collectives (e.g., all_gather_start and
# all_gather_done) behave identically to their synchronous counterparts (e.g.,
# all_gather).
class AsyncCollectivesTest(jtu.JaxTestCase):
  # DO_NOT_SUBMIT: Add XLA version guard.
  #
  # def setUp(self):
  #   if jaxlib_extension_version < XXX:
  #     self.skipTest('Requires jaxlib_extension_version >= XXX')

  def create_explicit_mesh(self, axes, names):
    axis_types = (jax.sharding.AxisType.Explicit,) * len(axes)
    return jtu.create_mesh(axes, names, iota_order=False, axis_types=axis_types)

  def overlappable_math(self, a):
    # On some backends, async collectives are erased if there isn't any
    # computation to overlap. Hence, we do some math on a.
    return a @ a

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

  def test_async_all_gather(self):
    n = jax.device_count()
    with jax.set_mesh(self.create_explicit_mesh((n,), ('i',))):
      @jax.jit
      @jax.shard_map(out_specs=(jax.P(None, reduced={'i'}), jax.P('i')))
      def all_gather_sync(x, a):
        a = self.overlappable_math(a)
        y_sync = jax.lax.all_gather(x, 'i', tiled=True, to='reduced')
        return y_sync, a

      @jax.jit
      @jax.shard_map(out_specs=(jax.P(None, reduced={'i'}), jax.P('i')))
      def all_gather_async(x, a):
        a = self.overlappable_math(a)
        todo = parallel.all_gather_start(x, 'i', tiled=True, to='reduced')
        y_async = todo.done()
        return y_async, a

      x = jnp.arange(n * 4096.0, out_sharding=jax.P('i'))
      a = jnp.ones((n * 1024, 1024), out_sharding=jax.P('i'))
      y_sync, _ = all_gather_sync(x, a)
      y_async, _ = all_gather_async(x, a)
      self.assertAllClose(y_sync, y_async)

      # On v6e_x8, both collectives should compile to an async collective.
      if jtu.device_kind_match('TPU v6') and len(jax.devices()) == 8:
        hlo_sync = all_gather_sync.lower(x, a).compile().as_text()
        self.assertIn('call-start(', hlo_sync)
        self.assertIn('all-gather(', hlo_sync)
        self.assertIn('call-done(', hlo_sync)

        hlo_async = all_gather_async.lower(x, a).compile().as_text()
        self.assertIn('call-start(', hlo_async)
        self.assertIn('all-gather(', hlo_async)
        self.assertIn('call-done(', hlo_async)

  def test_async_psum(self):
    n = jax.device_count()
    with jax.set_mesh(self.create_explicit_mesh((n,), ('i',))):
      @jax.jit
      @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
      def psum_sync(x, a):
        a = self.overlappable_math(a)
        y_sync = jax.lax.psum(x, 'i')
        return y_sync, a

      @jax.jit
      @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
      def psum_async(x, a):
        a = self.overlappable_math(a)
        y_async = parallel.psum_start(x, 'i').done()
        return y_async, a

      x = jnp.arange(n * 4096.0, out_sharding=jax.P('i'))
      a = jnp.ones((n * 1024, 1024), out_sharding=jax.P('i'))
      y_sync, _ = psum_sync(x, a)
      y_async, _ = psum_async(x, a)
      self.assertAllClose(y_sync, y_async)

      # On v6e_x8, both collectives should compile to an async collective.
      if jtu.device_kind_match('TPU v6') and len(jax.devices()) == 8:
        hlo_sync = psum_sync.lower(x, a).compile().as_text()
        self.assertIn('call-start(', hlo_sync)
        self.assertIn('all-reduce(', hlo_sync)
        self.assertIn('call-done(', hlo_sync)

        hlo_async = psum_async.lower(x, a).compile().as_text()
        self.assertIn('call-start(', hlo_async)
        self.assertIn('all-reduce(', hlo_async)
        self.assertIn('call-done(', hlo_async)

  def test_async_psum_scatter(self):
    n = jax.device_count()
    with jax.set_mesh(self.create_explicit_mesh((n,), ('i',))):
      @jax.jit
      @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
      def psum_scatter_sync(x, a):
        a = self.overlappable_math(a)
        y_sync = jax.lax.psum_scatter(x, 'i', scatter_dimension=0, tiled=True)
        return y_sync, a

      @jax.jit
      @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
      def psum_scatter_async(x, a):
        a = self.overlappable_math(a)
        todo = parallel.psum_scatter_start(x, 'i', scatter_dimension=0, tiled=True)
        y_async = todo.done()
        return y_async, a

      x = jnp.ones((n * 128, 128), dtype=jnp.float32, out_sharding=jax.P('i'))
      a = jnp.ones((n * 1024, 1024), out_sharding=jax.P('i'))
      y_sync, _ = psum_scatter_sync(x, a)
      y_async, _ = psum_scatter_async(x, a)
      self.assertAllClose(y_sync, y_async)

      # On v6e_x8, both collectives should compile to an async collective.
      if jtu.device_kind_match('TPU v6') and len(jax.devices()) == 8:
        hlo_sync = psum_scatter_sync.lower(x, a).compile().as_text()
        self.assertIn('call-start(', hlo_sync)
        self.assertIn('reduce-scatter(', hlo_sync)
        self.assertIn('call-done(', hlo_sync)

        hlo_async = psum_scatter_async.lower(x, a).compile().as_text()
        self.assertIn('call-start(', hlo_async)
        self.assertIn('reduce-scatter(', hlo_async)
        self.assertIn('call-done(', hlo_async)

  def test_async_all_to_all(self):
    n = jax.device_count()
    with jax.set_mesh(self.create_explicit_mesh((n,), ('i',))):
      @jax.jit
      @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
      def all_to_all_sync(x, a):
        a = self.overlappable_math(a)
        y_sync = jax.lax.all_to_all(x, 'i', split_axis=0, concat_axis=0, tiled=True)
        return y_sync, a

      @jax.jit
      @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
      def all_to_all_async(x, a):
        a = self.overlappable_math(a)
        todo = parallel.all_to_all_start(x, 'i', split_axis=0, concat_axis=0, tiled=True)
        y_async = todo.done()
        return y_async, a

      x = jnp.ones((n * 128, 128, 128), dtype=jnp.float32, out_sharding=jax.P('i'))
      a = jnp.ones((n * 1024, 1024), out_sharding=jax.P('i'))
      y_sync, _ = all_to_all_sync(x, a)
      y_async, _ = all_to_all_async(x, a)
      self.assertAllClose(y_sync, y_async)

      # On v6e_x8, both collectives should compile to an async collective.
      if jtu.device_kind_match('TPU v6') and len(jax.devices()) == 8:
        hlo_sync = all_to_all_sync.lower(x, a).compile().as_text()
        self.assertIn('all-to-all-start(', hlo_sync)
        self.assertIn('all-to-all-done(', hlo_sync)

        hlo_async = all_to_all_async.lower(x, a).compile().as_text()
        self.assertIn('all-to-all-start(', hlo_async)
        self.assertIn('all-to-all-done(', hlo_async)

  def test_async_ppermute(self):
    n = jax.device_count()
    permutation = [(i, (i + 1) % n) for i in range(n)]
    with jax.set_mesh(self.create_explicit_mesh((n,), ('i',))):
      @jax.jit
      @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
      def ppermute_sync(x, a):
        a = self.overlappable_math(a)
        y_sync = jax.lax.ppermute(x, 'i', permutation)
        return y_sync, a

      @jax.jit
      @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
      def ppermute_async(x, a):
        a = self.overlappable_math(a)
        todo = parallel.ppermute_start(x, 'i', permutation)
        y_async = todo.done()
        return y_async, a

      x = jnp.arange(n * 4096.0, out_sharding=jax.P('i'))
      a = jnp.ones((n * 1024, 1024), out_sharding=jax.P('i'))
      y_sync, _ = ppermute_sync(x, a)
      y_async, _ = ppermute_async(x, a)
      self.assertAllClose(y_sync, y_async)

      # On v6e_x8, both collectives should compile to an async collective.
      if jtu.device_kind_match('TPU v6') and len(jax.devices()) == 8:
        hlo_sync = ppermute_sync.lower(x, a).compile().as_text()
        self.assertIn('collective-permute-start', hlo_sync)
        self.assertIn('collective-permute-done', hlo_sync)

        hlo_async = ppermute_async.lower(x, a).compile().as_text()
        self.assertIn('collective-permute-start', hlo_async)
        self.assertIn('collective-permute-done', hlo_async)

  # pbroadcast is only implemented on GPU. If you try to run this on another
  # platform, you'll get an error like this:
  #
  # > NotImplementedError: MLIR translation rule for primitive 'pbroadcast' not
  # > found for platform cpu
  @jtu.run_on_devices('gpu')
  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_pbroadcast(self, mesh):
    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def pbroadcast(x):
      y_sync = jax.lax.pbroadcast(x, 'i', source=0)
      todo = parallel.pbroadcast_start(x, 'i', source=0)
      y_async = todo.done()
      return y_sync, y_async

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    y_sync, y_async = pbroadcast(x)
    self.assertAllClose(y_sync, y_async)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
