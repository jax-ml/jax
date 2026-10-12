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

from functools import partial
from absl.testing import absltest, parameterized

import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import hijax
from jax._src import test_util as jtu
from jax._src.ad_checkpoint import saved_residuals
from jax._src.lax import parallel
from jax._src.compute_on import compute_on
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.overlap import program_order
from jax.sharding import PartitionSpec as P

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class OverlapTest(jtu.JaxTestCase):

  def get_compiler_opts(self):
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
    return opts

  @jtu.with_explicit_mesh((8,), ('x',))
  def test_fsdp_pipeline_grad(self, mesh):
    def ag(x):
      return jax.reshard(x, P(reduced={'x'}))

    if jtu.is_device_tpu_at_least(7):
      ag = compute_on(ag, compute_type='tpu_sparsecore',
                       out_memory_spaces=jax.memory.Space.Device,
                       compiler_options={'sparse_core_config': {'core_ids': [0]}})

    def rs(x):
      return jax.reshard(x, (P('x', None), P(None, 'x')))

    if jtu.is_device_tpu_at_least(7):
      rs = compute_on(rs, compute_type='tpu_sparsecore',
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

    f = jax.jit(partial(fsdp_pipe, f), compiler_options=self.get_compiler_opts())
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

    @jax.jit(compiler_options=self.get_compiler_opts())
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
      self, full_po, mesh):
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

    @jax.jit(compiler_options=self.get_compiler_opts())
    @jax.shard_map(out_specs=P('x', None))
    def g(x, w1s, w2s):
      return fsdp_pipe(f, x, w1s, w2s)

    jax.block_until_ready(g(x, w1s, w2s))

  @jtu.with_explicit_mesh((2,), 'x')
  def test_simple_program_order(self, mesh):
    x = jax.device_put(jnp.arange(8), P('x'))
    y = jax.device_put(jnp.arange(8), P('x'))

    @jax.jit
    @program_order(enforce=True)
    def f(x, y):
      x1 = jax.reshard(x, P())
      x2 = jnp.sin(x1)
      y1 = jax.reshard(y, P())
      y2 = jnp.cos(y1)
      return x2, y2

    traced = f.trace(x, y)
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)
    self.assertIn('program_order', str(traced.jaxpr))
    lo_jaxpr = traced.lojax.jaxpr
    self.assertEqual(str(lo_jaxpr).count('optimization_barrier'), 5)

    f(x, y)  # doesn't crash

  @config.numpy_dtype_promotion('standard')
  def test_avoid_excess_precision(self):
    @jax.jit(static_argnames=('quant_dtype',))
    def f(x, quant_dtype):
      @program_order(enforce=True)
      def g(x):
        amax = jnp.abs(x).max(axis=-1, keepdims=True).astype(jnp.float32)
        scale = amax / jnp.iinfo(quant_dtype).max
        q = jnp.rint(x / scale).astype(quant_dtype)
        r = q.astype(jnp.float32) * scale
        return jnp.linalg.norm(r.astype(x.dtype) - x, ord=2, axis=-1)
      return g(x)

    x = jax.random.normal(jax.random.key(123), [16, 256], dtype=jnp.bfloat16)
    f(x, quant_dtype=jnp.int2)

  @jtu.with_explicit_mesh((2,), 'x')
  def test_program_order_exclude_args(self, mesh):
    x = jax.device_put(jnp.arange(8.0), P('x'))
    w = jax.device_put(jnp.arange(8.0) * 2.0, P('x'))

    @jax.jit
    @program_order(enforce=True)
    def f(x, w):
      @program_order(enforce=False)
      def op1(x):
        return x + 1.0
      y = op1(x)

      @program_order(enforce=False, exclude_argnames='w')
      def op2(w):
        return y * w
      return op2(w)

    traced = f.trace(x, w)
    self.assertIn('program_order', str(traced.jaxpr))
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)
    self.assertIn('program_order', str(traced.lojax.jaxpr))
    lowered_text = f.lower(x, w).as_text()
    self.assertNotIn('program_order', lowered_text)
    out = f(x, w)
    self.assertArraysEqual(out, (x + 1.0) * w)

    with self.assertRaisesRegex(
        ValueError, 'exclude_argnames cannot be used with enforce=True'):
      @program_order(enforce=True, exclude_argnames=('x',))
      def _(x):
        return x

  def test_program_order_exclude_argnames_none(self):
    x = jnp.arange(8.)
    c = jnp.arange(8.)

    @jax.jit
    @program_order(enforce=True)
    def g(x):
      @program_order(enforce=False)
      def op1(x):
        return x + c

      @program_order(enforce=False)
      def op2(y):
        return y * c

      y = op1(x)
      return op2(y)

    out = g(x)
    self.assertArraysEqual(out, (x + c) * c)

  @jtu.with_explicit_mesh((2,), 'x')
  def test_program_order_true_false_true_nest(self, mesh):
    x = jax.device_put(jnp.arange(8.0), P('x'))
    w = jax.device_put(jnp.arange(8.0) * 2.0, P('x'))

    @jax.jit
    @program_order(enforce=True)
    def f(x, w):
      x = jnp.exp(x)
      w = jnp.exp(w)

      @program_order(enforce=False)
      def op1(x):
        x = jnp.sin(x)
        x = jnp.cos(x)
        @program_order(enforce=True)
        def inner1(x):
          x1 = x + 1.0
          return x1 * 2.0
        return inner1(x)
      y = op1(x)

      @program_order(enforce=False, exclude_argnames='w')
      def op2(w):
        @program_order(enforce=True)
        def inner2(w):
          y1 = y * w
          return y1 + 3.0
        return inner2(w)
      return op2(w)

    traced = f.trace(x, w)
    self.assertIn('program_order', str(traced.jaxpr))
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)
    self.assertIn('program_order', str(traced.lojax.jaxpr))
    lowered_text = f.lower(x, w).as_text()
    self.assertNotIn('program_order', lowered_text)

    f(x, w)  # doesn't crash

  @jtu.with_explicit_mesh((2,), 'x')
  def test_program_order_per_input_barrier(self, mesh):
    x = jax.device_put(jnp.arange(8.0), P('x'))
    a = jax.device_put(jnp.arange(8.0) * 2.0, P('x'))
    b = jax.device_put(jnp.arange(8.0) * 3.0, P('x'))

    @jax.jit
    @program_order(enforce=True)
    def f(x, a, b):
      x = jnp.sin(x)

      @program_order(enforce=False)
      def op1(x):
        return x + 1.0
      y = op1(x)

      @program_order(enforce=False)
      def op2(y, a, b):
        return y * a + b
      z = op2(y, a, b)
      return z

    traced = f.trace(x, a, b)
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)
    jaxpr_str = str(traced.lojax.jaxpr)
    self.assertIn('create_token', jaxpr_str)
    # sin -> op1: 1 token barrier + 0 per-input barrier = 1
    # op1 -> op2: 1 token barrier + 2 per-input barriers = 3
    self.assertEqual(jaxpr_str.count('optimization_barrier'), 4)

    f(x, a, b)  # doesn't crash

  @jtu.with_explicit_mesh((2,), 'x')
  def test_program_order_per_input_barrier_jit(self, mesh):
    x = jax.device_put(jnp.arange(8.0), P('x'))
    a = jax.device_put(jnp.arange(8.0) * 2.0, P('x'))
    b = jax.device_put(jnp.arange(8.0) * 3.0, P('x'))

    @jax.jit
    @program_order(enforce=True)
    def f(x, a, b):
      x = jnp.sin(x)

      @jax.jit(inline=jax.Inline.JAX_LATE)
      def op1(x):
        return x + 1.0
      y = op1(x)

      @jax.jit(inline=jax.Inline.JAX_LATE)
      def op2(y, a, b):
        return y * a + b
      z = op2(y, a, b)
      return z

    traced = f.trace(x, a, b)
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)
    jaxpr_str = str(traced.lojax.jaxpr)
    self.assertIn('create_token', jaxpr_str)
    # sin -> op1: 1 token barrier + 0 per-input barrier = 1
    # op1 -> op2: 1 token barrier + 2 per-input barriers = 3
    self.assertEqual(jaxpr_str.count('optimization_barrier'), 4)

    f(x, a, b)  # doesn't crash

  @jtu.with_explicit_mesh((2,), 'x')
  def test_nullary(self, mesh):

    x = jax.device_put(jnp.arange(8.0), P('x'))
    y = jax.device_put(jnp.arange(8.0), P('x'))

    @jax.jit
    @program_order(enforce=True)
    def f(x, y):
      x1 = jax.reshard(x, P())
      iota = jnp.arange(4.0)
      x2 = jnp.sin(x1)
      y1 = jax.reshard(y, P())
      iota2 = jnp.arange(8.)
      y2 = jnp.cos(y1)
      return iota, iota2, x2, y2

    traced = f.trace(x, y)
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)
    jaxpr = traced.lojax.jaxpr
    self.assertEqual(str(jaxpr).count('optimization_barrier'), 5)
    self.assertEqual(str(jaxpr).count('create_token'), 2)

    f(x, y)  # doesn't crash

  @jtu.with_explicit_mesh((2,), 'x')
  def test_strict_in_out_basic(self, mesh):
    x = jax.device_put(jnp.arange(8.0), P('x'))
    y = jax.device_put(jnp.arange(8.0), P('x'))

    @jax.jit
    @program_order(enforce=True, strict_in_out=True)
    def f(x, y):
      x1 = jax.reshard(x, P())
      x2 = jnp.sin(x1)
      y1 = jax.reshard(y, P())
      y2 = jnp.cos(y1)
      return x2, y2

    traced = f.trace(x, y)
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)
    jaxpr = traced.lojax.jaxpr
    self.assertEqual(jaxpr.eqns[0].primitive.name, 'optimization_barrier')
    self.assertEqual(jaxpr.eqns[-1].primitive.name, 'optimization_barrier')
    self.assertEqual(str(jaxpr).count('optimization_barrier'), 5)

    out_x, out_y = f(x, y)
    self.assertArraysEqual(out_x, jnp.sin(x))
    self.assertArraysEqual(out_y, jnp.cos(y))

    with self.assertRaisesRegex(
        ValueError, 'strict_in_out=True cannot be used with enforce=False'):
      @program_order(enforce=False, strict_in_out=True)
      def _(x):
        return x

  @jtu.with_explicit_mesh((2,), 'x')
  def test_program_order_true_false_true_nest_strict_in_out(self, mesh):
    x = jax.device_put(jnp.arange(8.0), P('x'))
    w = jax.device_put(jnp.arange(8.0) * 2.0, P('x'))

    @jax.jit
    @program_order(enforce=True, strict_in_out=True)
    def f(x, w):
      x = jnp.exp(x)
      w = jnp.exp(w)

      @program_order(enforce=False)
      def op1(x):
        x = jnp.sin(x)
        x = jnp.cos(x)
        @program_order(enforce=True, strict_in_out=True)
        def inner1(x):
          x1 = x + 1.0
          return x1 * 2.0
        return inner1(x)
      y = op1(x)

      @program_order(enforce=False, exclude_argnames='w')
      def op2(w):
        @program_order(enforce=True, strict_in_out=True)
        def inner2(w):
          y1 = y * w
          return y1 + 3.0
        return inner2(w)
      return op2(w)

    traced = f.trace(x, w)
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)
    jaxpr = traced.lojax.jaxpr
    self.assertEqual(jaxpr.eqns[0].primitive.name, 'optimization_barrier')
    self.assertEqual(jaxpr.eqns[-1].primitive.name, 'optimization_barrier')
    lowered_text = f.lower(x, w).as_text()
    self.assertNotIn('program_order', lowered_text)

    f(x, w)  # doesn't crash

  @jtu.with_explicit_mesh((2,), 'x')
  def test_program_order_ad(self, mesh):
    x = jax.device_put(jnp.arange(8.0), P('x'))
    w = jax.device_put(jnp.arange(8.0) * 2.0, P('x'))

    @program_order(enforce=True)
    def f(x, w):
      x1 = jnp.sin(x)
      @program_order(enforce=False)
      def inner(x1, w):
        return x1 * w
      y = inner(x1, w)
      return jnp.cos(y)

    @jax.jit
    def grad_fn(x, w):
      y, f_vjp = jax.vjp(f, x, w)
      return y, f_vjp(jnp.ones_like(y))

    traced = grad_fn.trace(x, w)
    # HiJAX jaxpr has no optimization_barriers during AD
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)
    self.assertEqual(str(traced.jaxpr).count('program_order'), 4)
    # LoJAX jaxpr has optimization_barriers in both primal and backward passes
    self.assertEqual(str(traced.lojax.jaxpr).count('optimization_barrier'), 11)

    y, (dx, dw) = grad_fn(x, w)
    self.assertAllClose(y, jnp.cos(jnp.sin(x) * w))
    expected_dx, expected_dw = jax.grad(
        lambda x, w: jnp.cos(jnp.sin(x) * w).sum(), argnums=(0, 1))(x, w)
    self.assertAllClose(dx, expected_dx)
    self.assertAllClose(dw, expected_dw)

  def test_program_order_opt_barrier_dce(self):
    @jax.jit
    @program_order(enforce=True)
    def f(x, y):
      a = jnp.sin(x)
      _ = jnp.cos(x)
      b = jnp.exp(y)
      return a, b

    traced = f.trace(jnp.arange(8.), jnp.arange(8.))
    self.assertIn('cos', str(traced.jaxpr))
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)

    lo_jaxpr = traced.lojax.jaxpr
    self.assertNotIn('cos', str(lo_jaxpr))
    self.assertEqual([e.primitive.name for e in lo_jaxpr.eqns],
                     ['sin', 'optimization_barrier', 'exp'])
    self.assertLen(lo_jaxpr.eqns[1].invars, 2)
    self.assertEqual(lo_jaxpr.eqns[1].invars[0], lo_jaxpr.eqns[0].outvars[0])
    self.assertEqual(lo_jaxpr.eqns[1].outvars[1], lo_jaxpr.eqns[2].invars[0])

  def test_program_order_unused_output_dce(self):
    @program_order(enforce=True)
    def f(x, y):
      a = jnp.sin(x)
      c = jnp.cos(x)
      b = jnp.exp(y)
      return a, c, b

    @jax.jit
    def g(x, y):
      a, _, b = f(x, y)  # `c` is returned by `f`, but unused in `g`
      return a, b

    traced = g.trace(jnp.arange(8.), jnp.arange(8.))
    self.assertIn('cos', str(traced.jaxpr))
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)

    lo_jaxpr = traced.lojax.jaxpr
    self.assertNotIn('cos', str(lo_jaxpr))
    self.assertEqual([e.primitive.name for e in lo_jaxpr.eqns],
                     ['sin', 'optimization_barrier', 'exp'])
    self.assertLen(lo_jaxpr.eqns[1].invars, 2)
    self.assertEqual(lo_jaxpr.eqns[1].invars[0], lo_jaxpr.eqns[0].outvars[0])
    self.assertEqual(lo_jaxpr.eqns[1].outvars[1], lo_jaxpr.eqns[2].invars[0])

  def test_program_order_hiprim(self):
    class SinCos(hijax.HiPrim):
      def __init__(self, in_aval):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return jnp.cos(jnp.sin(x))

    @jax.jit
    @program_order(enforce=True)
    def f(x, y):
      a = SinCos(jax.typeof(x))(x)
      b = jnp.exp(y)
      return a, b

    x = jnp.arange(8.)
    y = jnp.arange(8.)
    traced = f.trace(x, y)
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)

    lo_jaxpr = traced.lojax.jaxpr
    self.assertEqual([e.primitive.name for e in lo_jaxpr.eqns],
                     ['sin', 'cos', 'optimization_barrier', 'exp'])

    out_a, out_b = f(x, y)
    self.assertAllClose(out_a, jnp.cos(jnp.sin(x)))
    self.assertAllClose(out_b, jnp.exp(y))

  def test_program_order_grad_enforce_false_in_true(self):
    @program_order(enforce=True)
    def f(x):
      @program_order(enforce=False)
      def blk(x):
        return jnp.sin(x)
      return blk(x) * 2.

    x = jnp.arange(8.)
    g = jax.jit(jax.grad(lambda x: f(x).sum()))
    traced = g.trace(x)
    self.assertIn('sin', str(traced.jaxpr))
    self.assertEqual(str(traced.jaxpr).count('optimization_barrier'), 0)
    self.assertNotIn('sin', str(traced.lojax.jaxpr))
    self.assertEqual(str(traced.lojax.jaxpr).count('optimization_barrier'), 2)

    out = g(x)
    self.assertAllClose(out, jnp.cos(x) * 2.)

  def test_program_order_dce_preserves_subfunction_deduplication(self):
    @program_order(enforce=True)
    def hi_step(x):
      return jnp.sin(x) + jnp.cos(x)

    @jax.jit
    def prefill_layer(x):
      r = jnp.remainder(jnp.arange(x.shape[0], dtype=jnp.int32), 2)
      return hi_step(x) + r.astype(x.dtype)

    @jax.jit
    def make_fn(x, tokens):
      x = prefill_layer(x)
      r1 = jnp.remainder(jnp.ones((4,), dtype=jnp.int32), 2)
      rolled = jnp.roll(tokens, r1[0])
      return x.sum() + rolled.sum().astype(x.dtype)

    mlir_text = make_fn.lower(
        jnp.ones((8,), dtype=jnp.float32),
        jnp.arange(16, dtype=jnp.int32),
    ).as_text()
    where_funcs = [
        line.strip()
        for line in mlir_text.splitlines()
        if 'func.func private @_where' in line
    ]
    self.assertLen(where_funcs, 1)

  @jtu.run_on_devices('gpu', 'tpu')
  @jtu.with_explicit_mesh((8,), ('x',))
  def test_simple_fsdp_async_overlap_program_order(self, mesh):
    if jtu.device_under_test() == 'tpu' and not jtu.is_device_tpu_at_least(6):
      self.skipTest('Requires TPU >= 6')

    @jax.shard_map(out_specs=P(reduced={'x'}))
    def ag_start(x):
      return parallel.all_gather_start(x, 'x', axis=0, tiled=True, to='reduced')

    @jax.shard_map(out_specs=P(reduced={'x'}))
    def ag_done(x):
      return x.done()

    @jax.jit
    @program_order(enforce=True)
    def f(x, w1, w2):
      fut_w2 = ag_start(w2)
      x = x @ w1
      w2_done = ag_done(fut_w2)
      return x @ w2_done

    x = jnp.ones((32 * 32, 128), out_sharding=P('x', None))
    w1 = jnp.ones((128, 256), out_sharding=P(None, None))
    w2 = jnp.ones((256, 128), out_sharding=P('x', None))

    out = f(x, w1, w2)
    self.assertEqual(out.sharding, jax.NamedSharding(mesh, P('x', None)))

  @jtu.run_on_devices('gpu', 'tpu')
  @jtu.with_explicit_mesh((8,), ('i',))
  def test_async_psum_program_order(self, mesh):
    if jtu.device_under_test() == 'tpu' and not jtu.is_device_tpu_at_least(5):
      self.skipTest('Requires TPU >= 5')

    @jax.jit
    @jax.shard_map(out_specs=(jax.P(), jax.P('i')))
    def f_sync(x, a):
      a = a @ a
      y = jax.lax.psum(x, 'i')
      return y, a

    @jax.jit
    @jax.shard_map(out_specs=(jax.P(), jax.P('i')))
    def f_async_po(x, a):
      @program_order(enforce=True)
      def body():
        future = parallel.psum_start(x, 'i')
        b = a @ a
        y = future.done()
        return y, b
      return body()

    x = jnp.arange(8 * 128 * 128.0, out_sharding=jax.P('i'))
    a = jnp.ones((8 * 128, 128), out_sharding=jax.P('i', None))
    y_sync, a_sync = f_sync(x, a)
    y_async, a_ = f_async_po(x, a)
    self.assertArraysEqual(y_sync, y_async)
    self.assertArraysEqual(a_sync, a_)

  @jtu.run_on_devices('gpu', 'tpu')
  @jtu.with_explicit_mesh((8,), ('i',))
  def test_async_psum_scatter_opt_barrier(self, mesh):
    if jtu.device_under_test() == 'tpu' and not jtu.is_device_tpu_at_least(7):
      self.skipTest('Requires TPU >= 7')
    if not jtu.is_libtpu_at_least('0.0.50'):
      self.skipTest('Requires libtpu >= 0.0.50')

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def psum_scatter_sync(x, a):
      a = a @ a
      y_sync = jax.lax.psum_scatter(x, 'i', scatter_dimension=0, tiled=True)
      return y_sync, a

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def psum_scatter_async_barrier(x, a):
      @program_order(enforce=True)
      def body():
        future = parallel.psum_scatter_start(x, 'i', scatter_dimension=0, tiled=True)
        b = a @ a
        y_async = future.done()
        return y_async, b
      return body()

    x = jnp.ones((8 * 128, 128), dtype=jnp.float32, out_sharding=jax.P('i'))
    a = jnp.ones((8 * 1024, 1024), out_sharding=jax.P('i'))
    y_sync, a_sync = psum_scatter_sync(x, a)
    y_async, a_ = psum_scatter_async_barrier(x, a)
    self.assertArraysEqual(y_sync, y_async)
    self.assertArraysEqual(a_sync, a_)

  @jtu.run_on_devices('gpu', 'tpu')
  @jtu.with_explicit_mesh((8,), ('i',))
  def test_async_all_to_all_opt_barrier(self, mesh):
    if jtu.device_under_test() == 'tpu' and not jtu.is_device_tpu_at_least(5):
      self.skipTest('Requires TPU >= 5')
    if not jtu.is_libtpu_at_least('0.0.50'):
      self.skipTest('Requires libtpu >= 0.0.50')

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def all_to_all_sync(x, a):
      a = a @ a
      y_sync = jax.lax.all_to_all(x, 'i', split_axis=0, concat_axis=0, tiled=True)
      return y_sync, a

    opts = ({'xla_tpu_enable_async_all_to_all': 'true'}
            if jtu.device_under_test() == 'tpu' else {})

    @jax.jit(compiler_options=opts)
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def all_to_all_async_barrier(x, a):
      @program_order(enforce=True)
      def body():
        future = parallel.all_to_all_start(x, 'i', split_axis=0, concat_axis=0,
                                           tiled=True)
        b = a @ a
        y_async = future.done()
        return y_async, b
      return body()

    x = jnp.ones((8 * 128, 128, 128), dtype=jnp.float32, out_sharding=jax.P('i'))
    a = jnp.ones((8 * 1024, 1024), out_sharding=jax.P('i'))
    y_sync, a_sync = all_to_all_sync(x, a)
    y_async, a_ = all_to_all_async_barrier(x, a)
    self.assertArraysEqual(y_sync, y_async)
    self.assertArraysEqual(a_sync, a_)

  @jtu.run_on_devices('gpu', 'tpu')
  @jtu.with_explicit_mesh((8,), ('i',))
  def test_async_ppermute_opt_barrier(self, mesh):
    if jtu.device_under_test() == 'tpu':
      if not jtu.is_device_tpu_at_least(5):
        self.skipTest('Requires TPU >= 5')
      if not jtu.is_libtpu_at_least('0.0.50'):
        self.skipTest('Requires libtpu >= 0.0.50')

    permutation = [(i, (i + 1) % 8) for i in range(8)]

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def ppermute_sync(x, a):
      a = a @ a
      y_sync = parallel.ppermute(x, 'i', permutation)
      return y_sync, a

    @jax.jit
    @jax.shard_map(out_specs=(jax.P('i'), jax.P('i')))
    def ppermute_async_barrier(x, a):
      @program_order(enforce=True)
      def body():
        future = parallel.ppermute_start(x, 'i', permutation)
        b = a @ a
        y_async = future.done()
        return y_async, b
      return body()

    x = jnp.arange(8 * 4096.0, out_sharding=jax.P('i'))
    a = jnp.ones((8 * 1024, 1024), out_sharding=jax.P('i'))
    y_sync, a_sync = ppermute_sync(x, a)
    y_async, a_ = ppermute_async_barrier(x, a)
    self.assertArraysEqual(y_sync, y_async)
    self.assertArraysEqual(a_sync, a_)

  def test_program_order_grad_exclude_argnames(self):
    @program_order(enforce=True)
    def f(x, w):
      @program_order(enforce=False)
      def op1(x):
        return jnp.sin(x)
      y = op1(x)

      @program_order(enforce=False, exclude_argnames='w')
      def op2(w):
        return y * w
      return op2(w).sum()

    x = jnp.arange(8.0)
    w = jnp.arange(8.0) * 2.0
    dx, dw = jax.jit(jax.grad(f, argnums=(0, 1)))(x, w)
    self.assertAllClose(dx, jnp.cos(x) * w)
    self.assertAllClose(dw, jnp.sin(x))

    dx2, dw2 = jax.jit(
        program_order(enforce=True)(jax.grad(f, argnums=(0, 1)))
    )(x, w)
    self.assertAllClose(dx2, jnp.cos(x) * w)
    self.assertAllClose(dw2, jnp.sin(x))

    dx3, dw3 = jax.jit(jax.grad(jax.remat(f), argnums=(0, 1)))(x, w)
    self.assertAllClose(dx3, jnp.cos(x) * w)
    self.assertAllClose(dw3, jnp.sin(x))

    y, dy = jax.jit(lambda x, w: jax.jvp(f, (x, w), (x, w)))(x, w)
    self.assertAllClose(y, (jnp.sin(x) * w).sum())
    self.assertAllClose(dy, (jnp.cos(x) * x * w + jnp.sin(x) * w).sum())

    # Also test DCE when an excluded or non-excluded argument is unused.
    @jax.jit
    @program_order(enforce=True)
    def f_dce(x, w, unused):
      @program_order(enforce=False, exclude_argnames='w')
      def op(x, w, unused):
        return x * w
      return op(x, w, unused)

    self.assertAllClose(f_dce(x, w, x), x * w)

  def test_remat_no_extra_residuals(self):
    # `b` only feeds a linear op, so the backward pass doesn't need it. When
    # program_order inserted its barriers at trace time, recomputing `a` in the
    # backward pass pulled in a barrier that also took `b`, so `b` was saved.
    def f(x):
      b = checkpoint_name(jnp.cos(x), 'b')
      a = checkpoint_name(jnp.sin(x), 'a')
      return a + b + jnp.sin(a)

    policy = jax.checkpoint_policies.save_only_these_names('a', 'b')
    x = jnp.arange(3.)
    expected = saved_residuals(jax.remat(f, policy=policy), x)
    actual = saved_residuals(
        jax.remat(program_order(enforce=True)(f), policy=policy), x)
    self.assertLen(actual, len(expected))

  @parameterized.named_parameters(
      ('program_order', program_order(enforce=True), program_order(enforce=False)),
      ('no_program_order', lambda f: f, lambda f: f),
  )
  def test_remat_nested_no_extra_residuals(self, po, po_f):
    # The overlap pattern: block 1 computes an activation h and prefetches the
    # next weights p; block 2 uses h nonlinearly and p only linearly.
    @po
    def f(x, w, wn):
      @po_f
      def blk1():
        p = checkpoint_name(jnp.cos(wn), 'b')
        h = checkpoint_name(jnp.sin(x * w), 'a')
        return h, p
      h, p = blk1()
      @po_f
      def blk2():
        return jnp.sin(h) + 2. * p
      return blk2()

    policy = jax.checkpoint_policies.save_only_these_names('a', 'b')
    args = jnp.arange(3.), jnp.arange(3.) + 1., jnp.arange(3.) + 2.
    res = saved_residuals(jax.remat(f, policy=policy), *args)
    self.assertLen(res, 4)  # x, w, wn, and 'a' (not 'b')

  def test_vjp_no_extra_residuals_zero_tangent(self):
    # z is only used through a comparison, so its tangent isn't needed. When
    # program_order inserted its barriers at trace time, z's tangent was tied
    # to x's by a barrier, so cos(x) was saved to compute it.
    def f(x):
      z = jnp.sin(x)
      a = 2. * x
      return a * (a > z).astype(a.dtype)

    # Count array residuals only: without program_order a scalar literal is
    # also saved.
    num_array_residuals = lambda f, x: sum(
        1 for aval, _ in saved_residuals(f, x) if aval.shape)
    x = jnp.arange(3.)
    self.assertEqual(num_array_residuals(f, x), 1)  # the mask
    self.assertEqual(num_array_residuals(program_order(enforce=True)(f), x), 1)

  def test_vjp_no_extra_residuals_multiple_outputs(self):
    # Before a jit equation, all of the previous equation's outputs used to go
    # through one barrier, tying z's (unneeded) tangent to y's.
    @jax.jit
    def g(x):
      return 2. * x, jnp.sin(x)

    @jax.jit
    def h(y):
      return 3. * y

    def f(x):
      y, z = g(x)
      return h(y) + jnp.floor(z)

    x = jnp.arange(3.)
    self.assertEmpty(saved_residuals(f, x))
    self.assertEmpty(saved_residuals(program_order(enforce=True)(f), x))

  def test_enforce_kwargs(self):
    @jax.jit
    def f(x, y):
      return program_order(enforce=True)(
          lambda x, *, y: jnp.sin(x) * y)(x, y=y)

    x = jnp.arange(3.)
    self.assertAllClose(f(x, x), jnp.sin(x) * x)


class AsyncCollectivesTest(jtu.JaxTestCase):

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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
