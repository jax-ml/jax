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
import textwrap
import unittest

from typing import Any, Callable, Sequence

from absl.testing import absltest
import jax
from jax import core
from jax import lax
from jax import tree_util
from jax._src import debugging
from jax._src import dispatch
from jax._src import sharding
from jax._src import test_util as jtu
from jax._src import util
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client
from jax.config import config
from jax.experimental import maps
from jax.experimental import pjit
from jax.interpreters import mlir
from jax.experimental.maps import Mesh
from jax.experimental.maps import xmap
from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np


config.parse_flags_with_absl()

debug_print = debugging.debug_print


def _format_multiline(text):
  return textwrap.dedent(text).lstrip()


prev_xla_flags = None


def setUpModule():
  global prev_xla_flags
  # This will control the CPU devices. On TPU we always have 2 devices
  prev_xla_flags = jtu.set_host_platform_device_count(2)


# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  prev_xla_flags()


callback_p = core.Primitive("callback")
callback_p.multiple_results = True

map, unsafe_map = util.safe_map, map


@callback_p.def_impl
def callback_impl(*args, callback: Callable[..., Any], result_avals,
                  effect: debugging.DebugEffect):
  del result_avals, effect
  return callback(*args)


@callback_p.def_effectful_abstract_eval
def callback_abstract_eval(*flat_avals, callback: Callable[..., Any],
                           effect: debugging.DebugEffect,
                           result_avals: Sequence[core.ShapedArray]):
  del flat_avals, callback
  return result_avals, {effect}


def callback(f, result_shape, *args, ordered: bool = False, **kwargs):
  flat_result_shapes, out_tree = tree_util.tree_flatten(result_shape)
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  flat_result_avals = [
      core.ShapedArray(s.shape, s.dtype) for s in flat_result_shapes
  ]
  effect = (
      debugging.DebugEffect.ORDERED_PRINT
      if ordered else debugging.DebugEffect.PRINT)
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  def _flat_callback(*flat_args):
    args, kwargs = tree_util.tree_unflatten(in_tree, flat_args)
    return tree_util.tree_leaves(f(*args, **kwargs))
  out_flat = callback_p.bind(
      *flat_args,
      callback=_flat_callback,
      effect=effect,
      result_avals=flat_result_avals)
  return tree_util.tree_unflatten(out_tree, out_flat)


def callback_lowering(ctx, *args, effect, callback, **params):

  def _callback(*flat_args):
    return tuple(
        callback_p.impl(*flat_args, effect=effect, callback=callback, **params))

  if effect in core.ordered_effects:
    token = ctx.tokens_in.get(effect)[0]
    result, token, keepalive = mlir.emit_python_callback(
        ctx, _callback, token, list(args), ctx.avals_in, ctx.avals_out, True)
    ctx.set_tokens_out(mlir.TokenSet({effect: (token,)}))
  else:
    result, token, keepalive = mlir.emit_python_callback(
        ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out, True)
  ctx.module_context.add_keepalive(keepalive)
  return result


mlir.register_lowering(callback_p, callback_lowering, platform="cpu")
mlir.register_lowering(callback_p, callback_lowering, platform="gpu")
mlir.register_lowering(callback_p, callback_lowering, platform="tpu")


@jtu.pytest_mark_if_available('pjrt_c_api_unimplemented')  # host callback
class PythonCallbackTest(jtu.JaxTestCase):

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()

  def test_callback_with_scalar_values(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    @jax.jit
    def f(x):
      return callback(lambda x: x + np.float32(1.),
                      core.ShapedArray(x.shape, x.dtype), x)

    out = f(0.)
    self.assertEqual(out, 1.)


  def test_callback_with_wrong_number_of_args(self):

    @jax.jit
    def f():
      # Calling a function that expects `x` with no arguments
      return callback(lambda x: np.ones(4, np.float32),
                      core.ShapedArray((4,), np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  def test_callback_with_wrong_number_of_returned_values(self):

    @jax.jit
    def f():
      # Calling a function with a return value that expects no return values
      return callback(lambda: np.ones(4, np.float32), ())

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

    @jax.jit
    def g():
      # Calling a function with a return value that expects no return values
      return callback(lambda: None, (core.ShapedArray(
          (1,), np.float32), core.ShapedArray((2,), np.float32)))

    with self.assertRaises(RuntimeError):
      g()
      jax.effects_barrier()

  def test_callback_with_wrong_shape_outputs(self):

    @jax.jit
    def f():
      # Calling a function expected a (1,) shaped return value but getting ()
      return callback(lambda: np.float32(1.), core.ShapedArray((1,),
        np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  def test_callback_with_wrong_dtype_outputs(self):

    def _cb():
      return np.array([1], np.float64)

    @jax.jit
    def f():
      # Calling a function expected a f32 return value but getting f64
      return callback(_cb, core.ShapedArray((1,), np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  def test_callback_with_single_return_value(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    @jax.jit
    def f():
      return callback(lambda: np.ones(4, np.float32),
                      core.ShapedArray((4,), np.float32))

    out = f()
    jax.effects_barrier()
    np.testing.assert_allclose(out, np.ones(4, np.float32))

  def test_callback_with_multiple_return_values(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    @jax.jit
    def f():
      return callback(lambda: (np.ones(4, np.float32), np.ones(5, np.int32)),
                      (core.ShapedArray(
                          (4,), np.float32), core.ShapedArray((5,), np.int32)))

    x, y = f()
    jax.effects_barrier()
    np.testing.assert_allclose(x, np.ones(4, np.float32))
    np.testing.assert_allclose(y, np.ones(5, np.int32))

  def test_callback_with_multiple_arguments_and_return_values(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x, y, z):
      return (x, y + z)

    @jax.jit
    def f(x, y, z):
      return callback(_callback, (core.ShapedArray(
          (3,), x.dtype), core.ShapedArray((3,), x.dtype)), x, y, z)

    x, y = f(jnp.ones(3), jnp.arange(3.), jnp.arange(3.) + 1.)
    np.testing.assert_allclose(x, np.ones(3))
    np.testing.assert_allclose(y, np.array([1., 3., 5]))

  def test_send_recv_zero_dim_arrays(self):

    def _callback(x):
      return x

    @jax.jit
    def f(x):
      return callback(_callback, core.ShapedArray((0,), np.float32), x)

    if jax.default_backend() == "tpu":
      with self.assertRaisesRegex(
          NotImplementedError,
          "Callbacks with zero-dimensional values not supported on TPU."):
        f(jnp.zeros(0, jnp.float32))
        jax.effects_barrier()
    else:
      np.testing.assert_allclose(
          f(jnp.zeros(0, jnp.float32)), np.zeros(0, np.float32))

  def test_callback_with_pytree_arguments_and_return_values(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return dict(y=[x])

    @jax.jit
    def f(x):
      return callback(_callback, dict(y=[core.ShapedArray((), np.float32)]),
                      [x])

    out = f(jnp.float32(2.))
    jax.effects_barrier()
    self.assertEqual(out, dict(y=[2.]))

  def test_callback_inside_of_while_loop_of_scalars(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.jit
    def f(x):
      def cond(x):
        return x < 10
      def body(x):
        return callback(_callback, core.ShapedArray((), x.dtype), x)
      return lax.while_loop(cond, body, x)

    out = f(0.)
    jax.effects_barrier()
    self.assertEqual(out, 10.)

  def test_callback_inside_of_while_loop(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.jit
    def f(x):

      def cond(x):
        return jnp.any(x < 10)

      def body(x):
        return callback(_callback, core.ShapedArray(x.shape, x.dtype), x)

      return lax.while_loop(cond, body, x)

    out = f(jnp.arange(5.))
    jax.effects_barrier()
    np.testing.assert_allclose(out, jnp.arange(10., 15.))

  def test_callback_inside_of_cond_of_scalars(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback1(x):
      return (x + 1.).astype(x.dtype)

    def _callback2(x):
      return (x - 1.).astype(x.dtype)

    @jax.jit
    def f(pred, x):

      def true_fun(x):
        return callback(_callback1, core.ShapedArray((), x.dtype), x)

      def false_fun(x):
        return callback(_callback2, core.ShapedArray((), x.dtype), x)

      return lax.cond(pred, true_fun, false_fun, x)

    out = f(True, 1.)
    jax.effects_barrier()
    self.assertEqual(out, 2.)
    out = f(False, 1.)
    jax.effects_barrier()
    self.assertEqual(out, 0.)

  def test_callback_inside_of_cond(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback1(x):
      return x + 1.

    def _callback2(x):
      return x - 1.

    @jax.jit
    def f(pred, x):

      def true_fun(x):
        return callback(_callback1, core.ShapedArray(x.shape, x.dtype), x)

      def false_fun(x):
        return callback(_callback2, core.ShapedArray(x.shape, x.dtype), x)

      return lax.cond(pred, true_fun, false_fun, x)

    out = f(True, jnp.ones(2))
    jax.effects_barrier()
    np.testing.assert_allclose(out, jnp.ones(2) * 2.)
    out = f(False, jnp.ones(2))
    jax.effects_barrier()
    np.testing.assert_allclose(out, jnp.zeros(2))

  def test_callback_inside_of_scan_of_scalars(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.jit
    def f(x):

      def body(x, _):
        x = callback(_callback, core.ShapedArray(x.shape, x.dtype), x)
        return x, ()

      return lax.scan(body, x, jnp.arange(10))[0]

    out = f(0.)
    jax.effects_barrier()
    self.assertEqual(out, 10.)

  def test_callback_inside_of_scan(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return x + 1.

    @jax.jit
    def f(x):

      def body(x, _):
        x = callback(_callback, core.ShapedArray(x.shape, x.dtype), x)
        return x, ()

      return lax.scan(body, x, jnp.arange(10))[0]

    out = f(jnp.arange(2.))
    jax.effects_barrier()
    np.testing.assert_allclose(out, jnp.arange(2.) + 10.)

  def test_callback_inside_of_pmap_of_scalars(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.pmap
    def f(x):
      return callback(_callback, core.ShapedArray(x.shape, x.dtype), x)

    out = f(jnp.arange(jax.local_device_count(), dtype=jnp.float32))
    jax.effects_barrier()
    np.testing.assert_allclose(
        out, np.arange(jax.local_device_count(), dtype=np.float32) + 1.)

  def test_callback_inside_of_pmap(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return x + 1.

    @jax.pmap
    def f(x):
      return callback(_callback, core.ShapedArray(x.shape, x.dtype), x)

    out = f(
        jnp.arange(2 * jax.local_device_count(),
                   dtype=jnp.float32).reshape([-1, 2]))
    jax.effects_barrier()
    np.testing.assert_allclose(
        out,
        np.arange(2 * jax.local_device_count()).reshape([-1, 2]) + 1.)

@jtu.pytest_mark_if_available('pjrt_c_api_unimplemented')  # host callback
class PurePythonCallbackTest(jtu.JaxTestCase):

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()

  def test_pure_callback_passes_ndarrays_without_jit(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def cb(x):
      self.assertIs(type(x), np.ndarray)
      return x

    def f(x):
      return jax.pure_callback(cb, x, x)
    f(jnp.array(2.))

  def test_simple_pure_callback(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    @jax.jit
    def f(x):
      return jax.pure_callback(lambda x: (x * 2.).astype(x.dtype), x, x)
    self.assertEqual(f(2.), 4.)

  def test_can_dce_pure_callback(self):

    if jax.default_backend() == "tpu":
      raise unittest.SkipTest("DCE doesn't currently happen on TPU")

    log = []
    def _callback(x):
      # Should never happen!
      log.append("hello world")
      return (x * 2.).astype(x.dtype)

    @jax.jit
    def f(x):
      _ = jax.pure_callback(_callback, x, x)
      return x * 2.
    _ = f(2.)
    self.assertEmpty(log)

  def test_callback_with_wrong_number_of_args(self):

    @jax.jit
    def f():
      # Calling a function that expects `x` with no arguments
      return jax.pure_callback(lambda x: np.ones(4, np.float32),
                               core.ShapedArray((4,), np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  def test_callback_with_wrong_number_of_returned_values(self):

    @jax.jit
    def f(x):
      # Calling a function with two return values that expects one return value
      return jax.pure_callback(lambda x: (x, np.ones(4, np.float32)), x, x)

    with self.assertRaises(RuntimeError):
      f(2.)
      jax.effects_barrier()

    @jax.jit
    def g():
      return jax.pure_callback(lambda: (), (
        core.ShapedArray((1,), np.float32), core.ShapedArray((2,), np.float32)))

    with self.assertRaises(RuntimeError):
      g()
      jax.effects_barrier()

  def test_callback_with_wrong_shape_outputs(self):

    @jax.jit
    def f():
      # Calling a function expected a (1,) shaped return value but getting ()
      return jax.pure_callback(lambda: np.float32(1.),
                               core.ShapedArray((1,), np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  def test_callback_with_wrong_dtype_outputs(self):

    def _cb():
      return np.array([1], np.float64)

    @jax.jit
    def f():
      # Calling a function expected a f32 return value but getting f64
      return callback(_cb, core.ShapedArray((1,), np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  def test_callback_with_wrongly_specified_64_bit_dtype(self):
    if config.jax_enable_x64:
      raise unittest.SkipTest("Test only needed when 64-bit mode disabled.")

    @jax.jit
    def f():
      return jax.pure_callback(lambda: np.float64(1.),
                               core.ShapedArray((), np.float64))

    with self.assertRaises(ValueError):
      f()
      jax.effects_barrier()

  def test_can_vmap_pure_callback(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    @jax.jit
    @jax.vmap
    def f(x):
      return jax.pure_callback(np.sin, x, x)
    out = f(jnp.arange(4.))
    np.testing.assert_allclose(out, np.sin(np.arange(4.)))

    @jax.jit
    def g(x):
      return jax.pure_callback(np.sin, x, x)
    out = jax.vmap(g, in_axes=1)(jnp.arange(8.).reshape((4, 2)))
    np.testing.assert_allclose(out, np.sin(np.arange(8.).reshape((4, 2))).T)

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(0, None))
    def h(x, y):
      out_shape = jax.ShapeDtypeStruct(x.shape, np.result_type(x.dtype, y.dtype))
      return jax.pure_callback(lambda x, y: np.sin(x) + y, out_shape, x, y)
    out = h(jnp.arange(4.), 4.)
    self.assertArraysAllClose(out, np.sin(np.arange(4.)) + 4.,
                              rtol=1E-7, check_dtypes=False)

    @jax.jit
    @functools.partial(jax.vmap)
    def h(x, y):
      out_shape = jax.ShapeDtypeStruct(x.shape, np.result_type(x.dtype, y.dtype))
      return jax.pure_callback(lambda x, y: np.sin(x) + y, out_shape, x, y)
    out = h(jnp.arange(4.), jnp.arange(10., 14.))
    self.assertArraysAllClose(out, np.sin(np.arange(4.)) + np.arange(10., 14.),
                              rtol=1E-7, check_dtypes=False)

  def test_vmap_vectorized_callback(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def cb(x):
      self.assertTupleEqual(x.shape, ())
      return np.sin(x)

    @jax.jit
    @jax.vmap
    def f(x):
      return jax.pure_callback(cb, x, x)

    np.testing.assert_allclose(f(jnp.arange(4.)), np.sin(np.arange(4.)))

    def cb2(x):
      self.assertTupleEqual(x.shape, (4,))
      return np.sin(x)

    @jax.jit
    @jax.vmap
    def g(x):
      return jax.pure_callback(cb2, x, x, vectorized=True)

    np.testing.assert_allclose(g(jnp.arange(4.)), np.sin(np.arange(4.)))

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(0, None))
    def h(x, y):
      return jax.pure_callback(lambda x, y: np.sin(x) + y, x, x, y,
                               vectorized=True)
    out = h(jnp.arange(4.), 4.)
    np.testing.assert_allclose(out, np.sin(np.arange(4.)) + 4.)

  def test_vmap_vectorized_callback_errors_if_returns_wrong_shape(self):

    def cb(x):
      # Reduces over all dimension when it shouldn't
      return np.sin(x).sum()

    @jax.jit
    @jax.vmap
    def f(x):
      return jax.pure_callback(cb, x, x, vectorized=True)

    with self.assertRaises(RuntimeError):
      f(jnp.arange(4.))
      jax.effects_barrier()

  def test_can_pmap_pure_callback(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    @jax.pmap
    def f(x):
      return jax.pure_callback(np.sin, x, x)
    out = f(jnp.arange(float(jax.local_device_count())))
    np.testing.assert_allclose(out, np.sin(np.arange(jax.local_device_count())))

  def test_can_pjit_pure_callback_under_hard_xmap(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest(
          'Host callback not supported for runtime type: stream_executor.'
      )

    if not hasattr(xla_client.OpSharding.Type, 'MANUAL'):
      raise unittest.SkipTest('Manual partitioning needed for pure_callback')

    jtu.set_spmd_lowering_flag(True)
    jtu.set_spmd_manual_lowering_flag(True)
    try:
      mesh = Mesh(np.array(jax.devices()), axis_names=('x',))

      spec = pjit.PartitionSpec('x')

      def f(x):
        axis_resources = {v: v for v in mesh.axis_names}
        return xmap(
            lambda x: jax.pure_callback(np.sin, x, x),
            in_axes=(('x',),),
            out_axes=('x',),
            axis_resources=axis_resources,
            axis_sizes=mesh.shape,
        )(x)

      def without_xmap_f(x):
        return jax.pure_callback(np.sin, x, x)

      with mesh:
        inp = jnp.arange(float(jax.local_device_count()))
        out = pjit.pjit(f, in_axis_resources=spec, out_axis_resources=spec)(inp)
        np.testing.assert_allclose(
            out, np.sin(np.arange(jax.local_device_count()))
        )

        if jax.local_device_count() > 1:
          with self.assertRaisesRegex(
              NotImplementedError, 'when all mesh axes are partitioned manually'
          ):
            pjit.pjit(
                without_xmap_f, in_axis_resources=spec, out_axis_resources=spec
            )(inp)

    finally:
      jtu.restore_spmd_manual_lowering_flag()
      jtu.restore_spmd_lowering_flag()

  def test_cant_take_grad_of_pure_callback(self):

    def sin(x):
      return np.sin(x)

    @jax.jit
    @jax.grad
    def f(x):
      return jax.pure_callback(sin, x, x)
    with self.assertRaisesRegex(
        ValueError, "Pure callbacks do not support JVP."):
      f(2.)

  def test_can_take_grad_of_pure_callback_with_custom_jvp(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    @jax.custom_jvp
    def sin(x):
      return jax.pure_callback(np.sin, x, x)

    @sin.defjvp
    def sin_jvp(xs, ts):
      (x,), (t,), = xs, ts
      return sin(x), jax.pure_callback(np.cos, x, x) * t

    @jax.jit
    @jax.grad
    def f(x):
      return sin(x)
    out = f(2.)
    np.testing.assert_allclose(out, jnp.cos(2.))

  def test_callback_inside_of_cond(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback1(x):
      return x + 1.

    def _callback2(x):
      return x - 1.

    @jax.jit
    def f(pred, x):

      def true_fun(x):
        return jax.pure_callback(_callback1, x, x)

      def false_fun(x):
        return jax.pure_callback(_callback2, x, x)

      return lax.cond(pred, true_fun, false_fun, x)

    out = f(True, jnp.ones(2))
    np.testing.assert_allclose(out, jnp.ones(2) * 2.)
    out = f(False, jnp.ones(2))
    np.testing.assert_allclose(out, jnp.zeros(2))

  def test_callback_inside_of_scan(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return x + 1.

    @jax.jit
    def f(x):

      def body(x, _):
        x = jax.pure_callback(_callback, x, x)
        return x, ()

      return lax.scan(body, x, jnp.arange(10))[0]

    out = f(jnp.arange(2.))
    np.testing.assert_allclose(out, jnp.arange(2.) + 10.)

  def test_callback_inside_of_while_loop(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _cond_callback(x):
      return np.any(x < 10)

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.jit
    def f(x):

      def cond(x):
        return jax.pure_callback(
            _cond_callback, jax.ShapeDtypeStruct((), np.bool_), x)

      def body(x):
        return jax.pure_callback(_callback, x, x)

      return lax.while_loop(cond, body, x)

    out = f(jnp.arange(5.))
    np.testing.assert_allclose(out, jnp.arange(10., 15.))

  def test_callback_inside_of_pmap(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return x + 1.

    @jax.pmap
    def f(x):
      return jax.pure_callback(_callback, x, x)

    out = f(
        jnp.arange(2 * jax.local_device_count(),
                   dtype=jnp.float32).reshape([-1, 2]))
    np.testing.assert_allclose(
        out,
        np.arange(2 * jax.local_device_count()).reshape([-1, 2]) + 1.)

  def test_callback_inside_xmap(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    def f(x):
      return jax.pure_callback(_callback, x, x)

    f = maps.xmap(f, in_axes=['a'], out_axes=['a'],
                  axis_resources={'a': 'dev'})
    with maps.Mesh(np.array(jax.devices()), ['dev']):
      out = f(np.arange(40.))
    np.testing.assert_allclose(out, jnp.arange(1., 41.))

  def test_vectorized_callback_inside_xmap(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    def f(x):
      return jax.pure_callback(_callback, x, x, vectorized=True)

    f = maps.xmap(f, in_axes=['a'], out_axes=['a'],
                  axis_resources={'a': 'dev'})
    with maps.Mesh(np.array(jax.devices()), ['dev']):
      out = f(np.arange(40.))
    np.testing.assert_allclose(out, jnp.arange(1., 41.))

  def test_array_layout_is_preserved(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    def g(x):
      return jax.pure_callback(lambda x: x, x, x)

    x = np.arange(6, dtype=np.int32).reshape((3, 2))
    np.testing.assert_allclose(g(x), x)

@jtu.pytest_mark_if_available('pjrt_c_api_unimplemented')  # host callback
class IOPythonCallbackTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()

  def test_io_callback_can_mutate_state(self):
    x = 0
    def cb():
      nonlocal x
      x += 1
      return np.array(x, np.int32)

    def f():
      return io_callback(cb, jax.ShapeDtypeStruct((), jnp.int32))
    f()
    jax.effects_barrier()
    self.assertEqual(x, 1)
    f()
    jax.effects_barrier()
    self.assertEqual(x, 2)

  def test_io_callback_can_be_batched_if_unordered(self):
    _mut = 0
    def cb(x):
      nonlocal _mut
      _mut += 1
      return x

    x = jnp.arange(4)
    def f(x):
      return io_callback(cb, jax.ShapeDtypeStruct((), x.dtype), x)
    jax.vmap(f)(x)
    jax.effects_barrier()
    self.assertEqual(_mut, 4)
    jax.vmap(f)(x)
    jax.effects_barrier()
    self.assertEqual(_mut, 8)

  def test_cannot_call_ordered_io_in_pmap(self):
    def f(x):
      return io_callback(
          lambda x: x, jax.ShapeDtypeStruct((), jnp.int32), x, ordered=True)
    with self.assertRaisesRegex(
        ValueError, "Ordered effects not supported in `pmap`"):
      jax.pmap(f)(jnp.arange(jax.local_device_count()))

  def test_cannot_call_ordered_io_in_xmap(self):
    def f(x):
      return io_callback(
          lambda x: x, jax.ShapeDtypeStruct((), jnp.int32), x, ordered=True)
    with self.assertRaisesRegex(
        ValueError, "Cannot `vmap` ordered IO callback"):
      maps.xmap(f, in_axes=([0],), out_axes=[0])(jnp.arange(16))

  def test_cannot_call_ordered_io_in_vmap(self):
    def f(x):
      return io_callback(
          lambda x: x, jax.ShapeDtypeStruct((), jnp.int32), x, ordered=True)
    with self.assertRaisesRegex(
        ValueError, "Cannot `vmap` ordered IO callback"):
      jax.vmap(f)(jnp.arange(4))

  def test_cannot_use_io_callback_in_jvp(self):
    def f(x):
      return io_callback(lambda x: x, jax.ShapeDtypeStruct((), jnp.float32), x)
    with self.assertRaisesRegex(
        ValueError, "IO callbacks do not support JVP."):
      jax.jvp(f, (0.,), (1.,))

  def test_cannot_use_io_callback_in_linearize(self):
    def f(x):
      return io_callback(lambda x: x, jax.ShapeDtypeStruct((), jnp.float32), x)
    with self.assertRaisesRegex(
        ValueError, "IO callbacks do not support JVP."):
      jax.linearize(f, 0.)

  def test_cannot_use_io_callback_in_transpose(self):
    x = jnp.array(1.)

    def f(x):
      return io_callback(lambda x: x, jax.ShapeDtypeStruct((), x.dtype), x)
    with self.assertRaisesRegex(
        ValueError, "IO callbacks do not support transpose."):
      jax.linear_transpose(f, x)(x)

  def test_cannot_vmap_of_cond_io_callback(self):
    def f(pred):
      def true_fun():
        io_callback(lambda: print("true"), None)
      def false_fun():
        io_callback(lambda: print("false"), None)
      return lax.cond(pred, false_fun, true_fun)
    with self.assertRaisesRegex(NotImplementedError,
        "IO effect not supported in vmap-of-cond."):
      jax.vmap(f)(jnp.array([True, True]))

  def test_cannot_vmap_of_while_io_callback(self):
    def check(x):
      assert np.all(x < 5)

    def f(i):
      def cond(i):
        return i < 5
      def body(i):
        io_callback(check, None, i)
        return i + 1
      return lax.while_loop(cond, body, i)
    with self.assertRaisesRegex(NotImplementedError,
        "IO effect not supported in vmap-of-while."):
      jax.vmap(f)(jnp.array([0, 4]))

  def test_cannot_use_io_callback_in_checkpoint(self):
    @jax.grad
    @jax.checkpoint
    def f(x, y):
      io_callback(lambda x: x, y, y)
      return x

    with self.assertRaisesRegex(NotImplementedError,
        "Effects not supported in partial-eval of `checkpoint`"):
      f(2., 3.)

  def test_can_use_io_callback_in_pjit(self):

    _mut = 0
    def _cb(x):
      nonlocal _mut
      _mut = x.sum()

    def f(x):
      io_callback(_cb, None, x)
      return x

    mesh = maps.Mesh(np.array(jax.devices()), ['dev'])
    if config.jax_array:
      spec = sharding.NamedSharding(mesh, pjit.PartitionSpec('dev'))
      out_spec = sharding.NamedSharding(mesh, pjit.PartitionSpec())
    else:
      spec = pjit.PartitionSpec('dev')
      out_spec = pjit.PartitionSpec()
    f = pjit.pjit(f, in_axis_resources=spec, out_axis_resources=out_spec)
    with mesh:
      f(jnp.arange(mesh.size))
      jax.effects_barrier()
    self.assertEqual(_mut, jnp.arange(mesh.size).sum())

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
