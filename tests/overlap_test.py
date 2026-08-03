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

import os

os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '')
    + ' --xla_dump_to='
    + os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', '/tmp/hlo_dump')
    + ' --xla_dump_hlo_as_text'
)

from functools import partial
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import parallel
from jax._src.compute_on import compute_on2
from jax.experimental.overlap import program_order
from jax.sharding import PartitionSpec as P
from jax._src.lib import ifrt_version

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class OverlapTest(jtu.JaxTestCase):

  @jtu.with_explicit_mesh((8,), ('x',))
  def test_fsdp_pipeline_grad(self, mesh):
    def ag(x):
      return jax.reshard(x, P(reduced={'x'}))

    if jtu.is_device_tpu_at_least(7):
      ag = compute_on2(ag, compute_type='tpu_sparsecore',
                       out_memory_spaces=jax.memory.Space.Device,
                       compiler_options={'sparse_core_config': {'core_ids': [0]}})

    def rs(x):
      return jax.reshard(x, (P('x', None), P(None, 'x')))

    if jtu.is_device_tpu_at_least(7):
      rs = compute_on2(rs, compute_type='tpu_sparsecore',
                       out_memory_spaces=jax.memory.Space.Device,
                       compiler_options={'sparse_core_config': {'core_ids': [1]}})

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def fsdp_pipe(f, x, ws):
      w = ag(jax.tree.map(lambda x: x[0], ws))
      carry = (x, w)
      def body(carry, w_n_sharded):
        x, w = carry
        w_n = ag(w_n_sharded)
        x = f(x, w)
        return (x, w_n), ()
      (x, w), () = jax.lax.scan(body, carry, jax.tree.map(lambda x: x[1:], ws),
                                unroll=2)  # need for double buffering
      x = f(x, w)
      return x

    def fsdp_pipe_fwd(f, x, ws):
      w = ag(jax.tree.map(lambda x: x[0], ws))
      x, f_vjp_first = jax.vjp(f, x, w)
      f_vjp_first.args_res[1] = None  # could instead use remat

      w = ag(jax.tree.map(lambda x: x[1], ws))
      carry = (x, w)

      def body(carry, w_n_sharded):
        x, w = carry
        w_n = ag(w_n_sharded)

        x, f_vjp = jax.vjp(f, x, w)
        f_vjp.args_res[1] = None

        return (x, w_n), f_vjp
      (x, w_last), f_vjps = jax.lax.scan(
          body, carry, jax.tree.map(lambda x: x[2:], ws), unroll=2)  # need for double buffering

      x, f_vjp_last = jax.vjp(f, x, w_last)
      f_vjp_last.args_res[1] = None
      return x, (f_vjp_first, f_vjps, f_vjp_last, ws)

    def fsdp_pipe_bwd(_, res, x_bar):
      f_vjp_first, f_vjps, f_vjp_last, ws = res

      w_m1 = ag(jax.tree.map(lambda x: x[-1], ws))
      f_vjp_last.args_res[1] = w_m1
      x_bar, w_m1_bar_unreduced = f_vjp_last(x_bar)

      w_m2 = ag(jax.tree.map(lambda x: x[-2], ws))
      carry = (x_bar, w_m2, w_m1_bar_unreduced)

      def body(carry, f_vjp_and_w_m1_sharded):
        y_bar, w, w_p1_bar_unreduced = carry
        f_vjp, w_m1_sharded = f_vjp_and_w_m1_sharded
        w_m1 = ag(w_m1_sharded)
        f_vjp.args_res[1] = w
        x_bar, w_bar_unreduced = f_vjp(y_bar)
        w_p1_bar_sharded = rs(w_p1_bar_unreduced)
        return (x_bar, w_m1, w_bar_unreduced), w_p1_bar_sharded

      (x_bar, w_0, w_1_bar_unreduced), ws_bar = jax.lax.scan(
          body, carry, (f_vjps, jax.tree.map(lambda x: x[:-2], ws)),
          reverse=True, unroll=2)

      f_vjp_first.args_res[1] = w_0
      x_bar, w_0_bar_unreduced = f_vjp_first(x_bar)
      w_1_bar = rs(w_1_bar_unreduced)
      w_0_bar = rs(w_0_bar_unreduced)
      ws_bar = jax.tree.map(
          lambda x, y, z: jnp.concatenate([x[None], y[None], z], axis=0),
          w_0_bar, w_1_bar, ws_bar)
      return x_bar, ws_bar

    fsdp_pipe.defvjp(fsdp_pipe_fwd, fsdp_pipe_bwd)

    def f(x, w):
      w1, w2 = w
      temp = x @ w1
      out = temp @ w2
      return out

    x = jnp.ones((32 * 32, 128), out_sharding=P('x', None))
    w1s = jnp.ones((32, 128, 256), out_sharding=P(None, 'x', None))
    w2s = jnp.ones((32, 256, 128), out_sharding=P(None, None, 'x'))
    ws = (w1s, w2s)

    # primal only
    jax.jit(partial(fsdp_pipe, f))(x, ws)  # doesn't crash

    @jax.jit
    def g(x, ws):
      y, f_vjp = jax.vjp(partial(fsdp_pipe, f), x, ws)
      return f_vjp(jnp.ones_like(y))
    jax.block_until_ready(g(x, ws))  # doesn't crash

  @jtu.with_explicit_mesh((8,), ('x',))
  def test_unrolled_fsdp_pipeline_grad_explicit_mode_program_order(self, mesh):
    def ag(x):
      return jax.reshard(x, P(reduced={'x'}))

    def fsdp_pipe(f, x, w1s, w2s):
      w1 = ag(w1s[0][0])
      w2 = ag(w2s[0][0])
      carry = (x, w1, w2)

      def body(carry, w_n):
        x, w1, w2 = carry
        w1n, w2n = w_n

        @program_order(enforce=True)
        def outer():
          @program_order(enforce=False)
          def inner():
            w1n_ = ag(w1n[0])
            w2n_ = ag(w2n[0])
            temp = f(x, w1, w2)
            return temp, w1n_, w2n_
          temp, w1n_, w2n_ = inner()

          @program_order(enforce=False)
          def inner2():
            _w1n_ = ag(w1n[1])
            _w2n_ = ag(w2n[1])
            out = f(temp, w1n_, w2n_)
            return out, _w1n_, _w2n_
          return inner2()
        x, _w1n_, _w2n_ = outer()

        return (x, _w1n_, _w2n_), ()

      (x, w1, w2), () = jax.lax.scan(body, carry, (w1s[1:], w2s[1:]))
      x = f(x, w1, w2)
      return x

    def f(x, w1, w2):
      temp = x @ w1
      out = temp @ w2
      return out

    x = jnp.ones((32 * 32, 128), out_sharding=P('x', None))
    w1s = jnp.ones((16, 2, 128, 256), out_sharding=P(None, None, 'x', None))
    w2s = jnp.ones((16, 2, 256, 128), out_sharding=P(None, None, None, 'x'))

    if jtu.device_under_test() == 'tpu':
      opts = dict(
          xla_tpu_enable_sparse_core_collective_offload_all_gather='true',
          xla_tpu_enable_sparse_core_collective_offload_2d_all_gather='true',
          xla_tpu_enable_sparse_core_collective_offload_reduce_scatter='true',
          xla_tpu_enable_sparse_core_offload_queuing_in_lhs='true',
          xla_tpu_control_large_2nd_minor_layout_for_x16='true',
          xla_msa_enable='false',
      )
    else:
      opts = {}

    f = jax.jit(partial(fsdp_pipe, f), compiler_options=opts)
    jax.block_until_ready(f(x, w1s, w2s))

  @jtu.with_explicit_mesh((8,), ('x',))
  def test_unrolled_fsdp_pipeline_grad_program_order_shmap(self, mesh):
    def ag(x, axis):
      return jax.lax.all_gather(x, 'x', axis=axis, tiled=True)

    def fsdp_pipe(f, x, w1s, w2s):
      w1 = ag(w1s[0][0], 0)
      w2 = ag(w2s[0][0], 1)
      carry = (x, w1, w2)

      @program_order(enforce=True)
      def body(carry, w_n):
        x, w1, w2 = carry
        w1n, w2n = w_n

        @program_order(enforce=False)
        def inner():
          w1n_ = ag(w1n[0], 0)
          w2n_ = ag(w2n[0], 1)
          temp = f(x, w1, w2)
          return temp, w1n_, w2n_
        temp, w1n_, w2n_ = inner()

        @program_order(enforce=False)
        def inner2():
          _w1n_ = ag(w1n[1], 0)
          _w2n_ = ag(w2n[1], 1)
          out = f(temp, w1n_, w2n_)
          return out, _w1n_, _w2n_
        out, _w1n_, _w2n_ = inner2()

        return (out, _w1n_, _w2n_), ()

      (x, w1, w2), () = jax.lax.scan(body, carry, (w1s[1:], w2s[1:]))
      x = f(x, w1, w2)
      return x

    def f(x, w1, w2):
      temp = x @ w1
      out = temp @ w2
      return out

    x = jnp.ones((32 * 32, 128), out_sharding=P('x', None))
    w1s = jnp.ones((16, 2, 128, 256), out_sharding=P(None, None, 'x', None))
    w2s = jnp.ones((16, 2, 256, 128), out_sharding=P(None, None, None, 'x'))

    if jtu.device_under_test() == 'tpu':
      opts = dict(
          xla_tpu_enable_sparse_core_collective_offload_all_gather='true',
          xla_tpu_enable_sparse_core_collective_offload_2d_all_gather='true',
          xla_tpu_enable_sparse_core_collective_offload_reduce_scatter='true',
          xla_tpu_enable_sparse_core_offload_queuing_in_lhs='true',
          xla_tpu_control_large_2nd_minor_layout_for_x16='true',
          xla_msa_enable='false',
      )
    else:
      opts = {}

    @jax.jit(compiler_options=opts)
    @jax.shard_map(out_specs=P('x', None))
    def g(x, w1s, w2s):
      return fsdp_pipe(f, x, w1s, w2s)

    jax.block_until_ready(g(x, w1s, w2s))

  @parameterized.named_parameters(
      ('full_program_order', True),
      ('partial_program_order', False),
  )
  @jtu.with_explicit_mesh((8,), ('x',))
  def test_unrolled_fsdp_pipeline_grad_program_order_async_decomp(
      self, full_po, mesh
  ):
    if ifrt_version < 63:
      self.skipTest("Requires ifrt_version >= 63")
    if not jtu.is_libtpu_at_least("0.0.45"):
      self.skipTest("Requires libtpu 0.0.45+")

    if not jtu.is_device_tpu_at_least(6):
      self.skipTest("Requires TPU >= 6")

    def ag(x, axis):
      return jax.lax.all_gather(x, 'x', axis=axis, tiled=True)

    def ag_start(x, axis):
      return parallel.all_gather_start(x, 'x', axis=axis, tiled=True)

    def fsdp_pipe(f, x, w1s, w2s):
      w1 = ag(w1s[0][0], 0)
      w2 = ag(w2s[0][0], 1)
      carry = (x, w1, w2)

      if full_po:

        @program_order(enforce=True)
        def body(carry, w_n):
          x, w1, w2 = carry
          w1n, w2n = w_n

          w1n_start = ag_start(w1n[0], 0)
          w2n_start = ag_start(w2n[0], 1)
          temp = f(x, w1, w2)
          w1n_ = w1n_start.done()
          w2n_ = w2n_start.done()

          _w1n_start = ag_start(w1n[1], 0)
          _w2n_start = ag_start(w2n[1], 1)
          out = f(temp, w1n_, w2n_)
          _w1n_ = _w1n_start.done()
          _w2n_ = _w2n_start.done()
          return (out, _w1n_, _w2n_), ()

      else:

        @program_order(enforce=True)
        def body(carry, w_n):
          x, w1, w2 = carry
          w1n, w2n = w_n

          @program_order(enforce=False)
          def inner():
            w1n_start = ag_start(w1n[0], 0)
            w2n_start = ag_start(w2n[0], 1)
            temp = f(x, w1, w2)
            w1n_ = w1n_start.done()
            w2n_ = w2n_start.done()
            return temp, w1n_, w2n_

          temp, w1n_, w2n_ = inner()

          @program_order(enforce=False)
          def inner2():
            _w1n_start = ag_start(w1n[1], 0)
            _w2n_start = ag_start(w2n[1], 1)
            out = f(temp, w1n_, w2n_)
            _w1n_ = _w1n_start.done()
            _w2n_ = _w2n_start.done()
            return out, _w1n_, _w2n_

          out, _w1n_, _w2n_ = inner2()
          return (out, _w1n_, _w2n_), ()

      (x, w1, w2), () = jax.lax.scan(body, carry, (w1s[1:], w2s[1:]))
      x = f(x, w1, w2)
      return x

    def f(x, w1, w2):
      temp = x @ w1
      out = temp @ w2
      return out

    x = jnp.ones((32 * 512 * 2, 1024), dtype=jnp.bfloat16,
                 out_sharding=P('x', None))
    w1s = jnp.ones((16, 2, 1024, 4096), dtype=jnp.bfloat16,
                   out_sharding=P(None, None, 'x', None))
    w2s = jnp.ones((16, 2, 4096, 1024), dtype=jnp.bfloat16,
                   out_sharding=P(None, None, None, 'x'))

    opts = dict(
        xla_tpu_enable_sparse_core_collective_offload_all_gather='true',
        xla_tpu_enable_sparse_core_collective_offload_2d_all_gather='true',
        xla_tpu_enable_sparse_core_collective_offload_reduce_scatter='true',
        xla_tpu_enable_sparse_core_offload_queuing_in_lhs='true',
        xla_tpu_control_large_2nd_minor_layout_for_x16='true',
        xla_msa_enable='false',
    )

    @jax.jit(compiler_options=opts)
    @jax.shard_map(out_specs=P('x', None))
    def g(x, w1s, w2s):
      return fsdp_pipe(f, x, w1s, w2s)

    jax.block_until_ready(g(x, w1s, w2s))

  @jtu.with_explicit_mesh((8,), ('x',))
  def test_fsdp_pipeline_grad_explicit_async_loop_carry_future(self, mesh):
    if ifrt_version < 63:
      self.skipTest('Requires ifrt_version >= 63')
    if not jtu.is_libtpu_at_least('0.0.45'):
      self.skipTest('Requires libtpu 0.0.45+')

    if not jtu.is_device_tpu_at_least(6):
      self.skipTest('Requires TPU >= 6')

    def ag_start(x, axis):
      return parallel.all_gather_start(x, 'x', axis=axis, tiled=True)

    def fsdp_pipe(f, x, w1s, w2s):
      w1_start = ag_start(w1s[0][0], 0)
      w2_start = ag_start(w2s[0][0], 1)
      carry = (x, w1_start, w2_start)

      @program_order(enforce=True)
      def body(carry, w_n):
        x, w1_start, w2_start = carry
        w1n, w2n = w_n

        @program_order(enforce=False)
        def inner():
          w1n_start = ag_start(w1n[0], 0)
          w2n_start = ag_start(w2n[0], 1)
          temp = f(x, w1_start, w2_start)
          return temp, w1n_start, w2n_start

        temp, w1n_start, w2n_start = inner()

        @program_order(enforce=False)
        def inner2():
          _w1n_start = ag_start(w1n[1], 0)
          _w2n_start = ag_start(w2n[1], 1)
          out = f(temp, w1n_start, w2n_start)
          return out, _w1n_start, _w2n_start

        out, _w1n_start, _w2n_start = inner2()
        return (out, _w1n_start, _w2n_start), ()

      (x, w1_last_start, w2_last_start), () = jax.lax.scan(
          body, carry, (w1s[1:], w2s[1:])
      )
      x = f(x, w1_last_start, w2_last_start)
      return x

    def f(x, w1, w2):
      temp = x @ w1.done()
      out = temp @ w2.done()
      return out

    x = jnp.ones(
        (32 * 512 * 2, 1024), dtype=jnp.bfloat16, out_sharding=P('x', None)
    )
    w1s = jnp.ones(
        (16, 2, 1024, 4096),
        dtype=jnp.bfloat16,
        out_sharding=P(None, None, 'x', None),
    )
    w2s = jnp.ones(
        (16, 2, 4096, 1024),
        dtype=jnp.bfloat16,
        out_sharding=P(None, None, None, 'x'),
    )

    opts = dict(
        xla_tpu_enable_sparse_core_collective_offload_all_gather='true',
        xla_tpu_enable_sparse_core_collective_offload_2d_all_gather='true',
        xla_tpu_enable_sparse_core_collective_offload_reduce_scatter='true',
        xla_tpu_enable_sparse_core_offload_queuing_in_lhs='true',
        xla_tpu_control_large_2nd_minor_layout_for_x16='true',
        xla_msa_enable='false',
    )

    @jax.jit(compiler_options=opts)
    @jax.shard_map(out_specs=P('x', None))
    def g(x, w1s, w2s):
      return fsdp_pipe(f, x, w1s, w2s)

    res = g(x, w1s, w2s)
    print('Jetski: Python: g() returned')
    jax.block_until_ready(res)
    print('Jetski: Python: block_until_ready returned')

  @jtu.with_explicit_mesh((8,), ('x',))
  def test_fsdp_pipeline_explicit_mode_async_start_done_in_shmap(self, mesh):
    def ag_start(x, axis):
      @jax.shard_map(out_specs=P(reduced={'x'}))
      def _f(x):
        return parallel.all_gather_start(
            x, 'x', axis=axis, tiled=True, to='reduced'
        )

      return _f(x)

    @jax.shard_map(out_specs=P(reduced={'x'}))
    def ag_done(x):
      return x.done()

    def fsdp_pipe(f, x, w1s, w2s):
      w1_start = ag_start(w1s[0][0], 0)
      w2_start = ag_start(w2s[0][0], 1)
      carry = (x, w1_start, w2_start)

      def body(carry, w_n):
        x, w1_start, w2_start = carry
        w1n, w2n = w_n

        @program_order(enforce=True)
        def outer():
          @program_order(enforce=False)
          def inner():
            w1n_start = ag_start(w1n[0], 0)
            w2n_start = ag_start(w2n[0], 1)
            temp = f(x, w1_start, w2_start)
            return temp, w1n_start, w2n_start

          temp, w1n_start, w2n_start = inner()

          @program_order(enforce=False)
          def inner2():
            _w1n_start = ag_start(w1n[1], 0)
            _w2n_start = ag_start(w2n[1], 1)
            out = f(temp, w1n_start, w2n_start)
            return out, _w1n_start, _w2n_start

          return inner2()

        x, _w1n_start, _w2n_start = outer()
        return (x, _w1n_start, _w2n_start), ()

      (x, w1_start, w2_start), () = jax.lax.scan(
          body, carry, (w1s[1:], w2s[1:])
      )
      x = f(x, w1_start, w2_start)
      return x

    def f(x, w1_start, w2_start):
      temp = x @ ag_done(w1_start)
      out = temp @ ag_done(w2_start)
      return out

    x = jnp.ones((32 * 32, 128), out_sharding=P('x', None))
    w1s = jnp.ones((16, 2, 128, 256), out_sharding=P(None, None, 'x', None))
    w2s = jnp.ones((16, 2, 256, 128), out_sharding=P(None, None, None, 'x'))

    f = jax.jit(partial(fsdp_pipe, f))
    print(f.lower(x, w1s, w2s).as_text())
    jax.block_until_ready(f(x, w1s, w2s))


class AsyncCollectivesTest(jtu.JaxTestCase):

  def setUp(self):
    if ifrt_version < 63:
      self.skipTest("Requires ifrt_version >= 63")
    if not jtu.is_libtpu_at_least("0.0.45"):
      self.skipTest("Requires libtpu 0.0.45+")

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_all_gather(self, mesh):
    @jax.shard_map(out_specs=jax.P(None, reduced={'i'}))
    def f(x):
      return parallel.all_gather_start(x, 'i', tiled=True, to='reduced').done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.custom_call', stablehlo)
    self.assertIn('all-gather-start', stablehlo)
    self.assertIn('all-gather-done', stablehlo)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_psum(self, mesh):
    @jax.shard_map(out_specs=jax.P('i'))
    def f(x):
      return parallel.psum_start(x, 'i').done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.custom_call', stablehlo)
    self.assertIn('all-reduce-start', stablehlo)
    self.assertIn('all-reduce-done', stablehlo)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_psum_scatter(self, mesh):
    @jax.shard_map(out_specs=jax.P('i'))
    def f(x):
      future = parallel.psum_scatter_start(x, 'i', scatter_dimension=0, tiled=True)
      return future.done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.custom_call', stablehlo)
    self.assertIn('reduce-scatter-start', stablehlo)
    self.assertIn('reduce-scatter-done', stablehlo)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_all_to_all(self, mesh):
    @jax.shard_map(out_specs=jax.P('i'))
    def f(x):
      future = parallel.all_to_all_start(x, 'i', split_axis=0, concat_axis=0,
                                         tiled=True)
      return future.done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.custom_call', stablehlo)
    self.assertIn('all-to-all-start', stablehlo)
    self.assertIn('all-to-all-done', stablehlo)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_lower_async_ppermute(self, mesh):
    @jax.jit
    @jax.shard_map(out_specs=jax.P('i'))
    def f(x):
      return parallel.ppermute_start(x, 'i', [(0, 1), (1, 0)]).done()

    x = jnp.arange(64.0, out_sharding=jax.P('i'))
    stablehlo = jax.jit(f).lower(x).as_text()
    self.assertIn('stablehlo.custom_call', stablehlo)
    self.assertIn('collective-permute-start', stablehlo)
    self.assertIn('collective-permute-done', stablehlo)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_all_gather(self, mesh):
    @jax.jit
    @jax.shard_map(out_specs=(jax.P(None, reduced={'i'}), jax.P('i')))
    def all_gather_sync(x, a):
      a = a @ a
      y_sync = jax.lax.all_gather(x, 'i', tiled=True, to='reduced')
      return y_sync, a

    @jax.jit
    @jax.shard_map(out_specs=(jax.P(None, reduced={'i'}), jax.P('i')))
    def all_gather_async(x, a):
      a = a @ a
      future = parallel.all_gather_start(x, 'i', tiled=True, to='reduced')
      y_async = future.done()
      return y_async, a

    x = jnp.arange(2 * 4096.0, out_sharding=jax.P('i'))
    a = jnp.ones((2 * 1024, 1024), out_sharding=jax.P('i'))
    y_sync, _ = all_gather_sync(x, a)
    y_async, _ = all_gather_async(x, a)
    self.assertAllClose(y_sync, y_async)

    # If the synchronous JAX collective lowers to an asynchronous HLO
    # collective, then so should the asynchronous JAX collective.
    hlo_sync = all_gather_sync.lower(x, a).compile().as_text()
    hlo_async = all_gather_async.lower(x, a).compile().as_text()
    for op in ['call-start(', 'all-gather(', 'call-done(']:
      if op in hlo_sync:
        self.assertIn(op, hlo_async)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_psum(self, mesh):
    @jax.jit
    @jax.shard_map(out_specs=(jax.P(), jax.P('i')))
    def psum_sync(x, a):
      a = a @ a
      y_sync = jax.lax.psum(x, 'i')
      return y_sync, a

    @jax.jit
    @jax.shard_map(out_specs=(jax.P(), jax.P('i')))
    def psum_async(x, a):
      a = a @ a
      y_async = parallel.psum_start(x, 'i').done()
      return y_async, a

    x = jnp.arange(2 * 4096.0, out_sharding=jax.P('i'))
    a = jnp.ones((2 * 1024, 1024), out_sharding=jax.P('i'))
    y_sync, _ = psum_sync(x, a)
    y_async, _ = psum_async(x, a)
    self.assertAllClose(y_sync, y_async)

    # If the synchronous JAX collective lowers to an asynchronous HLO
    # collective, then so should the asynchronous JAX collective.
    hlo_sync = psum_sync.lower(x, a).compile().as_text()
    hlo_async = psum_async.lower(x, a).compile().as_text()
    for op in ['call-start(', 'all-reduce(', 'call-done(']:
      if op in hlo_sync:
        self.assertIn(op, hlo_async)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_psum_scatter(self, mesh):
    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def psum_scatter_sync(x, a):
      a = a @ a
      y_sync = jax.lax.psum_scatter(x, 'i', scatter_dimension=0, tiled=True)
      return y_sync, a

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def psum_scatter_async(x, a):
      a = a @ a
      future = parallel.psum_scatter_start(x, 'i', scatter_dimension=0, tiled=True)
      y_async = future.done()
      return y_async, a

    x = jnp.ones((2 * 128, 128), dtype=jnp.float32, out_sharding=jax.P('i'))
    a = jnp.ones((2 * 1024, 1024), out_sharding=jax.P('i'))
    y_sync, _ = psum_scatter_sync(x, a)
    y_async, _ = psum_scatter_async(x, a)
    self.assertAllClose(y_sync, y_async)

    # If the synchronous JAX collective lowers to an asynchronous HLO
    # collective, then so should the asynchronous JAX collective.
    hlo_sync = psum_scatter_sync.lower(x, a).compile().as_text()
    hlo_async = psum_scatter_async.lower(x, a).compile().as_text()
    for op in ['call-start(', 'reduce-scatter(', 'call-done(']:
      if op in hlo_sync:
        self.assertIn(op, hlo_async)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_all_to_all(self, mesh):
    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def all_to_all_sync(x, a):
      a = a @ a
      y_sync = jax.lax.all_to_all(x, 'i', split_axis=0, concat_axis=0, tiled=True)
      return y_sync, a

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def all_to_all_async(x, a):
      a = a @ a
      future = parallel.all_to_all_start(x, 'i', split_axis=0, concat_axis=0,
                                          tiled=True)
      y_async = future.done()
      return y_async, a

    x = jnp.ones((2 * 128, 128, 128), dtype=jnp.float32, out_sharding=jax.P('i'))
    a = jnp.ones((2 * 1024, 1024), out_sharding=jax.P('i'))
    y_sync, _ = all_to_all_sync(x, a)
    y_async, _ = all_to_all_async(x, a)
    self.assertAllClose(y_sync, y_async)

    # If the synchronous JAX collective lowers to an asynchronous HLO
    # collective, then so should the asynchronous JAX collective.
    hlo_sync = all_to_all_sync.lower(x, a).compile().as_text()
    hlo_async = all_to_all_async.lower(x, a).compile().as_text()
    for op in ['all-to-all-start(', 'all-to-all-done(']:
      if op in hlo_sync:
        self.assertIn(op, hlo_async)

  @jtu.with_explicit_mesh((2,), ('i',))
  def test_async_ppermute(self, mesh):
    permutation = [(i, (i + 1) % 2) for i in range(2)]

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def ppermute_sync(x, a):
      a = a @ a
      y_sync = jax.lax.ppermute(x, 'i', permutation)
      return y_sync, a

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def ppermute_async(x, a):
      a = a @ a
      future = parallel.ppermute_start(x, 'i', permutation)
      y_async = future.done()
      return y_async, a

    x = jnp.arange(2 * 4096.0, out_sharding=jax.P('i'))
    a = jnp.ones((2 * 1024, 1024), out_sharding=jax.P('i'))
    y_sync, _ = ppermute_sync(x, a)
    y_async, _ = ppermute_async(x, a)
    self.assertAllClose(y_sync, y_async)

    # If the synchronous JAX collective lowers to an asynchronous HLO
    # collective, then so should the asynchronous JAX collective.
    hlo_sync = ppermute_sync.lower(x, a).compile().as_text()
    hlo_async = ppermute_async.lower(x, a).compile().as_text()
    for op in ['collective-permute-start(', 'collective-permute-done(']:
      if op in hlo_sync:
        self.assertIn(op, hlo_async)


# class ControlDepsTest(jtu.JaxTestCase):

#   def create_explicit_mesh(self, axes, names):
#     axis_types = (jax.sharding.AxisType.Explicit,) * len(axes)
#     return jtu.create_mesh(axes, names, iota_order=False, axis_types=axis_types)

#   @jtu.run_on_devices("tpu", "cpu")
#   def test_math(self):
#     @jax.jit
#     def f_math(x, y, z):
#       a = jnp.sin(x @ x)
#       b = jnp.cos(y @ y)
#       c = jnp.exp(z @ z)
#       schedule([c, b, a])
#       return a + b + c

#     x = jnp.ones((67, 67))
#     hlo = f_math.lower(x, x, x).as_text(dialect="hlo")
#     self.assertIn('custom_call_target="control_dep"', hlo)
#     f_math(x, x, x)  # doesn't crash

#   @jtu.run_on_devices("tpu", "cpu")
#   def test_fsdp(self):
#     k = 4
#     n = jax.device_count()
#     with jax.set_mesh(self.create_explicit_mesh((n,), ("i",))):
#       @jax.jit
#       @jax.shard_map(out_specs=(jax.P("i")), check_vma=False)
#       def f_fsdp(x, ws):
#         starts = []
#         dones = []
#         maths = []

#         # This is a simple version of FSDP where x is like a set of activations
#         # and ws is a list of weights, one per layer. We repeatedly all-gather
#         # the weights for a layer and multiply with x.
#         for w_shard in ws:
#           fut = parallel.all_gather_start(w_shard, "i", tiled=True)
#           w = fut.done()
#           x = x @ w

#           # Note that we pipe out the intermediate values.
#           starts.append(fut)
#           dones.append(w)
#           maths.append(x)

#         # Here we schedule the code to run in a smart FSDP order where the all
#         # gather for the next layer is overlapped with the math for the current
#         # layer.
#         deps = []
#         for i in range(k + 1):
#           if i == 0:
#             deps.append(starts[0])
#             deps.append(dones[0])
#           elif i < k:
#             deps.append(starts[i])
#             deps.append(maths[i - 1])
#             deps.append(dones[i])
#           else:
#             deps.append(maths[i - 1])
#         schedule(deps)

#         return x

#       N = 128 * n
#       x = jnp.ones((n * N, N), out_sharding=jax.P("i", None))
#       ws = [jnp.ones((N, N), out_sharding=jax.P("i", None)) for _ in range(k)]
#       hlo = jax.jit(f_fsdp).lower(x, ws).as_text(dialect="hlo")
#       self.assertIn('custom_call_target="control_dep"', hlo)
#       f_fsdp(x, ws)  # doesn't crash

#   @jtu.run_on_devices("tpu", "cpu")
#   def test_scan_fsdp(self):
#     k = 4
#     n = jax.device_count()
#     with jax.set_mesh(self.create_explicit_mesh((n,), ("i",))):
#       # This test shows FSDP with scan.
#       @jax.jit
#       @jax.shard_map(out_specs=(jax.P("i")), check_vma=False)
#       def f_scan_fsdp(x, ws):
#         # Prologue.
#         w_0 = jax.lax.all_gather(ws[0], "i", tiled=True)

#         # Scan.
#         def f(carry, w_shard):
#           w, x = carry
#           fut = parallel.all_gather_start(w_shard, "i", tiled=True)
#           x = x @ w
#           w_next = fut.done()
#           schedule([fut, x, w_next])
#           return (w_next, x), None
#         (w, x), _ = jax.lax.scan(f, (w_0, x), ws[1:])

#         # Epilogue.
#         x = x @ w
#         return x

#       N = 128 * n
#       x = jnp.ones((n * N, N), out_sharding=jax.P("i", None))
#       ws = jnp.ones((k, N, N), out_sharding=jax.P(None, "i", None))
#       hlo = jax.jit(f_scan_fsdp).lower(x, ws).as_text(dialect="hlo")
#       self.assertIn('custom_call_target="control_dep"', hlo)
#       f_scan_fsdp(x, ws)  # doesn't crash

#   @jtu.run_on_devices("tpu", "cpu")
#   def test_pipeline(self):
#     if jtu.device_under_test() == "tpu" and not jtu.is_device_tpu_at_least(7):
#       self.skipTest("Needs TPU >= 7")
#     k = 4
#     n = jax.device_count()
#     with jax.set_mesh(self.create_explicit_mesh((n,), ("i",))):

#       @jax.jit
#       @jax.shard_map(out_specs=(jax.P("i")), check_vma=False)
#       def f_pipeline(xs, ws):
#         starts = []
#         dones = []
#         maths = []

#         # This shows a form of pipelining across microbatches. xs and ws are the
#         # same size. We need to run xs[i] @ ws[i] for every i.
#         for x, w_shard in zip(xs, ws):
#           f = parallel.all_gather_start(w_shard, "i", tiled=True)
#           w = f.done()
#           y = x @ w

#           starts.append(f)
#           dones.append(w)
#           maths.append(y)

#         # We schedule things to run all the starts, then done and math in the
#         # right order.
#         schedule(starts)
#         schedule([starts[-1], dones[0]])
#         schedule(maths)
#         for i in range(k - 1):
#           control_dep(maths[i], dones[i + 1])

#         return reduce(lambda x, y: x + y, maths)

#       N = 128 * n
#       x = [jnp.ones((N, N), out_sharding=jax.P(None, None)) for _ in range(k)]
#       ws = [jnp.ones((N, N), out_sharding=jax.P("i", None)) for _ in range(k)]
#       hlo = jax.jit(f_pipeline).lower(x, ws).as_text(dialect="hlo")
#       self.assertIn('custom_call_target="control_dep"', hlo)
#       f_pipeline(x, ws)  # doesn't crash


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
