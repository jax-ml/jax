# Copyright 2022 Google LLC
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
import contextlib
import functools
import io
import textwrap
import unittest
from unittest import mock

from typing import Any, Callable, Generator, Sequence

from absl.testing import absltest
import jax
from jax import core
from jax import lax
from jax import tree_util
from jax._src import debugging
from jax._src import dispatch
from jax._src import lib as jaxlib
from jax._src import test_util as jtu
from jax._src import util
from jax.config import config
from jax.experimental import maps
from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np


config.parse_flags_with_absl()

debug_print = debugging.debug_print


@contextlib.contextmanager
def capture_stdout() -> Generator[Callable[[], str], None, None]:
  with mock.patch("sys.stdout", new_callable=io.StringIO) as fp:

    def _read() -> str:
      return fp.getvalue()

    yield _read


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


# TODO(sharadmv): remove jaxlib guards for TPU tests when jaxlib minimum
#                 version is >= 0.3.15
disabled_backends = []
if jaxlib.version < (0, 3, 15):
  disabled_backends.append("tpu")

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
if jaxlib.version >= (0, 3, 15):
  mlir.register_lowering(callback_p, callback_lowering, platform="tpu")


class PythonCallbackTest(jtu.JaxTestCase):

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_with_scalar_values(self):

    @jax.jit
    def f(x):
      return callback(lambda x: x + np.float32(1.),
                      core.ShapedArray(x.shape, x.dtype), x)

    out = f(0.)
    self.assertEqual(out, 1.)


  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_with_wrong_number_of_args(self):

    @jax.jit
    def f():
      # Calling a function that expects `x` with no arguments
      return callback(lambda x: np.ones(4, np.float32),
                      core.ShapedArray((4,), np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  @jtu.skip_on_devices(*disabled_backends)
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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_with_wrong_shape_outputs(self):

    @jax.jit
    def f():
      # Calling a function expected a (1,) shaped return value but getting ()
      return callback(lambda: np.float32(1.), core.ShapedArray((1,),
        np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  @jtu.skip_on_devices(*disabled_backends)
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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_with_single_return_value(self):

    @jax.jit
    def f():
      return callback(lambda: np.ones(4, np.float32),
                      core.ShapedArray((4,), np.float32))

    out = f()
    jax.effects_barrier()
    np.testing.assert_allclose(out, np.ones(4, np.float32))

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_with_multiple_return_values(self):

    @jax.jit
    def f():
      return callback(lambda: (np.ones(4, np.float32), np.ones(5, np.int32)),
                      (core.ShapedArray(
                          (4,), np.float32), core.ShapedArray((5,), np.int32)))

    x, y = f()
    jax.effects_barrier()
    np.testing.assert_allclose(x, np.ones(4, np.float32))
    np.testing.assert_allclose(y, np.ones(5, np.int32))

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_with_multiple_arguments_and_return_values(self):

    def _callback(x, y, z):
      return (x, y + z)

    @jax.jit
    def f(x, y, z):
      return callback(_callback, (core.ShapedArray(
          (3,), x.dtype), core.ShapedArray((3,), x.dtype)), x, y, z)

    x, y = f(jnp.ones(3), jnp.arange(3.), jnp.arange(3.) + 1.)
    np.testing.assert_allclose(x, np.ones(3))
    np.testing.assert_allclose(y, np.array([1., 3., 5]))

  @jtu.skip_on_devices(*disabled_backends)
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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_with_pytree_arguments_and_return_values(self):

    def _callback(x):
      return dict(y=[x])

    @jax.jit
    def f(x):
      return callback(_callback, dict(y=[core.ShapedArray((), np.float32)]),
                      [x])

    out = f(jnp.float32(2.))
    jax.effects_barrier()
    self.assertEqual(out, dict(y=[2.]))

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_while_loop_of_scalars(self):

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_while_loop(self):

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_cond_of_scalars(self):

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_cond(self):

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_scan_of_scalars(self):

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_scan(self):

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_pmap_of_scalars(self):

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.pmap
    def f(x):
      return callback(_callback, core.ShapedArray(x.shape, x.dtype), x)

    out = f(jnp.arange(jax.local_device_count(), dtype=jnp.float32))
    jax.effects_barrier()
    np.testing.assert_allclose(
        out, np.arange(jax.local_device_count(), dtype=np.float32) + 1.)

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_pmap(self):

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

class PurePythonCallbackTest(jtu.JaxTestCase):

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()

  @jtu.skip_on_devices(*disabled_backends)
  def test_simple_pure_callback(self):

    @jax.jit
    def f(x):
      return jax.pure_callback(lambda x: (x * 2.).astype(x.dtype), x, x)
    self.assertEqual(f(2.), 4.)

  @jtu.skip_on_devices(*disabled_backends)
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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_with_wrong_number_of_args(self):

    @jax.jit
    def f():
      # Calling a function that expects `x` with no arguments
      return jax.pure_callback(lambda x: np.ones(4, np.float32),
                               core.ShapedArray((4,), np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  @jtu.skip_on_devices(*disabled_backends)
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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_with_wrong_shape_outputs(self):

    @jax.jit
    def f():
      # Calling a function expected a (1,) shaped return value but getting ()
      return jax.pure_callback(lambda: np.float32(1.),
                               core.ShapedArray((1,), np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  @jtu.skip_on_devices(*disabled_backends)
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

  def test_can_vmap_pure_callback(self):

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
      return jax.pure_callback(lambda x, y: np.sin(x) + y, x, x, y)
    out = h(jnp.arange(4.), 4.)
    np.testing.assert_allclose(out, np.sin(np.arange(4.)) + 4.)

  def test_vmap_vectorized_callback(self):

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

    @jax.pmap
    def f(x):
      return jax.pure_callback(np.sin, x, x)
    out = f(jnp.arange(float(jax.local_device_count())))
    np.testing.assert_allclose(out, np.sin(np.arange(jax.local_device_count())))

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_cond(self):

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_scan(self):

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_while_loop(self):

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_of_pmap(self):

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

  @jtu.skip_on_devices(*disabled_backends)
  def test_callback_inside_xmap(self):

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    def f(x):
      return jax.pure_callback(_callback, x, x)

    f = maps.xmap(f, in_axes=['a'], out_axes=['a'],
                  axis_resources={'a': 'dev'})
    with maps.Mesh(np.array(jax.devices()), ['dev']):
      out = f(np.arange(40.))
    np.testing.assert_allclose(out, jnp.arange(1., 41.))

  @jtu.skip_on_devices(*disabled_backends)
  def test_vectorized_callback_inside_xmap(self):

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    def f(x):
      return jax.pure_callback(_callback, x, x, vectorized=True)

    f = maps.xmap(f, in_axes=['a'], out_axes=['a'],
                  axis_resources={'a': 'dev'})
    with maps.Mesh(np.array(jax.devices()), ['dev']):
      out = f(np.arange(40.))
    np.testing.assert_allclose(out, jnp.arange(1., 41.))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
