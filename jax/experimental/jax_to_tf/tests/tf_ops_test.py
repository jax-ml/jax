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
"""Tests for the jax_to_tf transformation."""

from absl.testing import absltest
from absl.testing import parameterized
import functools
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import test_util as jtu
import numpy as np
import tensorflow as tf  # type: ignore[import]

from jax.experimental import jax_to_tf

from jax.config import config
config.parse_flags_with_absl()


# TODO(tomhennigan) Increase coverage here.
LAX_ELEMENTWISE_UNARY = (
    lax.abs,
    lax.acosh,
    lax.asinh,
    lax.atanh,
    lax.bessel_i0e,
    lax.bessel_i1e,
    lax.ceil,
    lax.cos,
    lax.cosh,
    lax.digamma,
    lax.erf,
    lax.erf_inv,
    lax.erfc,
    lax.exp,
    lax.expm1,
    lax.floor,
    lax.is_finite,
    lax.lgamma,
    lax.log,
    lax.log1p,
    lax.neg,
    lax.round,
    lax.rsqrt,
    lax.sign,
    lax.sin,
    lax.sinh,
    lax.sqrt,
    lax.tan,
    lax.tanh,
)

LAX_ELEMENTWISE_BINARY = (
    lax.add,
    lax.atan2,
    lax.div,
    lax.igamma,
    lax.igammac,
    lax.max,
    lax.min,
    lax.nextafter,
    lax.rem,
    lax.sub,
)

LAX_LOGICAL_ELEMENTWISE_BINARY = (
    lax.bitwise_and,
    lax.bitwise_or,
    lax.bitwise_xor,
    lax.shift_left,
    lax.shift_right_arithmetic,
    lax.shift_right_logical,
)

REDUCE = (
    jnp.all,
    jnp.any,
    jnp.max,
    jnp.min,
    jnp.prod,
    jnp.sum,
)

INDEX = (
    jax.ops.index_add,
    jax.ops.index_max,
    jax.ops.index_min,
    jax.ops.index_mul,
    jax.ops.index_update,
)


class TfOpsTest(jtu.JaxTestCase):

  def test_basics(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    f_tf = jax_to_tf.convert(f_jax)
    self.assertIsInstance(f_tf(0.7), tf.Tensor)
    np.testing.assert_allclose(f_jax(0.7), f_tf(0.7))

  def test_variable_input(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    f_tf = jax_to_tf.convert(f_jax)
    v = tf.Variable(0.7)
    self.assertIsInstance(f_tf(v), tf.Tensor)
    np.testing.assert_allclose(f_jax(0.7), f_tf(v))

  def test_jit(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    f_tf = jax_to_tf.convert(f_jax)
    np.testing.assert_allclose(f_jax(0.7), f_tf(0.7))

  def test_converts_jax_arrays(self):
    f_tf = tf.function(lambda x: x)
    self.assertEqual(f_tf(jnp.zeros([])).numpy(), 0.)
    self.assertEqual(f_tf(jnp.ones([])).numpy(), 1.)
    f_tf = tf.function(lambda x: x + x)
    self.assertEqual(f_tf(jnp.ones([])).numpy(), 2.)

    # Test with ShardedDeviceArray.
    n = jax.local_device_count()
    mk_sharded = lambda f: jax.pmap(lambda x: x)(f([n]))
    f_tf = tf.function(lambda x: x)
    np.testing.assert_allclose(f_tf(mk_sharded(jnp.zeros)).numpy(),
                               np.zeros([n]))
    np.testing.assert_allclose(f_tf(mk_sharded(jnp.ones)).numpy(), np.ones([n]))

  def test_function(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    np.testing.assert_allclose(f_jax(0.7), f_tf(0.7))

  @parameterized.parameters(jnp.add, jnp.subtract, jnp.multiply, jnp.divide,
                            jnp.less, jnp.less_equal, jnp.equal, jnp.greater,
                            jnp.greater_equal, jnp.not_equal, jnp.maximum,
                            jnp.minimum)
  def test_type_promotion(self, f_jax):
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    # We only test a few types here, as tensorflow does not support many
    # types like uint* or bool in binary ops.
    types = [np.int32, np.int64, np.float32]
    for x_dtype in types:
      for y_dtype in types:
        x = np.array([1, 2], dtype=x_dtype)
        y = np.array([3, 4], dtype=y_dtype)
        r_jax = f_jax(x, y)
        r_tf = f_tf(x, y)
        np.testing.assert_allclose(r_jax, r_tf)

  def test_concat(self):
    values = [np.array([1, 2], dtype=np.float32),
              np.array([1, 2], dtype=np.int32),
              np.array([1, 2], dtype=np.int8)]
    f_jax = jax.jit(lambda x: jnp.concatenate(x, axis=0))
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    np.testing.assert_allclose(f_jax(values), f_tf(values))

  def test_pad(self):
    values = np.array([1, 2], dtype=np.float32)
    f_jax = jax.jit(lambda x: jax.lax.pad(x, 0.0, [(3, 1, 2)]))
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    np.testing.assert_allclose(f_jax(values), f_tf(values))

  @parameterized.parameters(*LAX_ELEMENTWISE_UNARY)
  def test_unary_elementwise(self, f_jax):
    x = np.array([-1.6, -1.4, -1.0, 0.0, 0.1, 0.2, 1, 1.4, 1.6],
                 dtype=np.float32)
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    r_jax = f_jax(x)
    r_tf = f_tf(x)
    np.testing.assert_allclose(r_jax[np.isfinite(r_jax)],
                               r_tf[np.isfinite(r_tf)], atol=1e-4)

  def test_bitwise_not(self):
    x = np.array([-1, 3, -2, 0, 0, 2, 1, 3], dtype=np.int32)
    f_jax = jax.jit(lax.bitwise_not)
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    r_jax = f_jax(x)
    r_tf = f_tf(x)
    np.testing.assert_allclose(r_jax[np.isfinite(r_jax)],
                               r_tf[np.isfinite(r_tf)], atol=1e-4)

  @parameterized.parameters(*LAX_ELEMENTWISE_BINARY)
  def test_binary_elementwise(self, f_jax):
    a = np.array([-1.6, -1.4, -1.0, 0.0, 0.1, 0.2, 1, 1.4, 1.6],
                 dtype=np.float32)
    b = np.array([-1.6, 1.4, 1.0, 0.0, 0.1, 0.2, 1, 1.4, -1.6],
                 dtype=np.float32)
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    r_jax = f_jax(a, b)
    r_tf = f_tf(a, b)
    # Jax outputs 0 and 1 instead of NaN for values outside the domain.
    # Whereas tensorflow does this for other combinations,
    if f_jax in (lax.igamma, lax.igammac):
      # Make returned array writeable.
      r_jax = np.copy(r_jax)
      r_jax[r_jax == 0] = np.nan
      r_jax[r_jax == 1] = np.nan
      r_tf = np.copy(r_tf)
      r_tf[r_tf == 0] = np.nan
      r_tf[r_tf == 1] = np.nan
    np.testing.assert_allclose(r_jax[np.isfinite(r_jax)],
                               r_tf[np.isfinite(r_tf)], atol=1e-4)

  @parameterized.parameters(*LAX_LOGICAL_ELEMENTWISE_BINARY)
  def test_binary_logical_elementwise(self, f_jax):
    a = np.array([1, 3, 2, 0, 0, 2, 1, 3], dtype=np.uint32)
    b = np.array([1, 2, 3, 0, 1, 0, 2, 3], dtype=np.uint32)
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    r_jax = f_jax(a, b)
    r_tf = f_tf(a, b)
    np.testing.assert_allclose(r_jax[np.isfinite(r_jax)],
                               r_tf[np.isfinite(r_tf)], atol=1e-4)

  @parameterized.parameters((lax.betainc,))
  def test_trinary_elementwise(self, f_jax):
    a = np.array([-1.6, -1.4, -1.0, 0.0, 0.1, 0.3, 1, 1.4, 1.6],
                 dtype=np.float32)
    b = np.array([-1.6, 1.4, 1.0, 0.0, 0.2, 0.1, 1, 1.4, -1.6],
                 dtype=np.float32)
    c = np.array([1.0, -1.0, 2.0, 1.0, 0.3, 0.3, -1.0, 2.4, 1.6],
                 dtype=np.float32)
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    r_jax = f_jax(a, b, c)
    r_tf = f_tf(a, b, c)
    np.testing.assert_allclose(r_jax[np.isfinite(r_jax)],
                               r_tf[np.isfinite(r_tf)], atol=1e-4)

  def test_gather(self):
    values = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    for axis in (0, 1):
      f_jax = jax.jit(lambda v, i: jnp.take(v, i, axis=axis))  # pylint: disable=cell-var-from-loop
      f_tf = tf.function(jax_to_tf.convert(f_jax))
      np.testing.assert_allclose(f_jax(values, indices), f_tf(values, indices))

  def test_boolean_gather(self):
    values = np.array([[True, True], [False, True], [False, False]],
                      dtype=np.bool)
    indices = np.array([0, 1], dtype=np.int32)
    for axis in [0, 1]:
      f_jax = jax.jit(lambda v, i: jnp.take(v, i, axis=axis))  # pylint: disable=cell-var-from-loop
      f_tf = tf.function(jax_to_tf.convert(f_jax))
      np.testing.assert_allclose(f_jax(values, indices), f_tf(values, indices))

  @parameterized.parameters(*REDUCE)
  def test_reduce_ops_with_numerical_input(self, f_jax):
    values = [np.array([1, 2, 3], dtype=np.float32)]
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    np.testing.assert_allclose(f_jax(values), f_tf(values))

  @parameterized.parameters(jnp.cumsum, jnp.cumprod)
  def test_cumulated_ops(self, f_jax):
    values = np.array([1, 2, 3], dtype=np.float32)
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    np.testing.assert_allclose(f_jax(values), f_tf(values))

  @parameterized.parameters(*INDEX)
  def test_scatter_static(self, op):
    values = np.ones((5, 6), dtype=np.float32)
    update = np.float32(6.)
    f_jax = jax.jit(lambda v, u: op(v, jax.ops.index[::2, 3:], u))
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    np.testing.assert_allclose(f_jax(values, update), f_tf(values, update))

  @parameterized.parameters(*REDUCE)
  def test_reduce_ops_with_boolean_input(self, f_jax):
    values = [np.array([True, False, True], dtype=np.bool)]
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    np.testing.assert_allclose(f_jax(values), f_tf(values))

  def test_gather_rank_change(self):
    params = jnp.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [3.0, 3.5, 4.0]])
    indices = jnp.array([[1, 1, 2], [0, 1, 0]])
    f_jax = jax.jit(lambda i: params[i])
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    np.testing.assert_allclose(f_jax(indices), f_tf(indices))

  def test_prngsplit(self):
    f_jax = jax.jit(lambda key: jax.random.split(key, 2))
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    for rng_key in [jax.random.PRNGKey(42),
                    np.array([0, 0], dtype=np.uint32),
                    np.array([0xFFFFFFFF, 0], dtype=np.uint32),
                    np.array([0, 0xFFFFFFFF], dtype=np.uint32),
                    np.array([0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)
                    ]:
      jax_keys = f_jax(rng_key)
      tf_keys = f_tf(rng_key)
      for jax_key, tf_key in zip(jax_keys, tf_keys):
        np.testing.assert_equal(jax_key, tf_key)

  def test_gradients_disabled(self):
    f = jax_to_tf.convert(jnp.tan)
    x = tf.ones([])
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = f(x)
    with self.assertRaisesRegex(ValueError,
                                'jax2tf currently does not support gradients'):
      tape.gradient(y, x)

  def test_zeros_like(self):
    v = np.float32(2.)
    f_jax = jax.ad_util.zeros_like_jaxval
    f_tf = jax_to_tf.convert(f_jax)
    self.assertEqual(f_jax(v), f_tf(v))

  def test_stop_gradient(self):
    f = jax_to_tf.convert(lax.stop_gradient)
    self.assertEqual(f(tf.ones([])), 1.)

  def test_checkpoint_wrapper_types(self):
    m = tf.Module()
    m.a = [tf.Module(), tf.Module()]
    m.b = (tf.Module(), tf.Module())
    m.c = {'a': tf.Module(), 'b': tf.Module()}
    self.assertNotEqual(type(m.a), list)
    self.assertNotEqual(type(m.b), tuple)
    self.assertNotEqual(type(m.c), dict)
    self.assertLen(jax.tree_leaves(m.a), 2)
    self.assertLen(jax.tree_leaves(m.b), 2)
    self.assertLen(jax.tree_leaves(m.c), 2)


class ControlFlowOpsTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_cond(self, with_function=False):
    with jax_to_tf.enable_jit():

      def f_jax(pred, x):
        return lax.cond(pred, lambda t: t + 1., lambda f: f, x)

      f_tf = jax_to_tf.convert(f_jax)
      if with_function:
        f_tf = tf.function(f_tf)
      np.testing.assert_allclose(f_tf(True, 1.), f_jax(True, 1.))
      np.testing.assert_allclose(f_tf(False, 1.), f_jax(False, 1.))

  @parameterized.parameters(False, True)
  def test_cond_multiple_results(self, with_function=False):
    with jax_to_tf.enable_jit():

      def f_jax(pred, x):
        return lax.cond(pred, lambda t: (t + 1., 1.), lambda f: (f + 2., 2.), x)

      f_tf = jax_to_tf.convert(f_jax)
      if with_function:
        f_tf = tf.function(f_tf)
      np.testing.assert_allclose(f_tf(True, 1.), f_jax(True, 1.))
      np.testing.assert_allclose(f_tf(False, 1.), f_jax(False, 1.))

  @parameterized.parameters(False, True)
  def test_scan(self, with_function=False):
    def f_jax(xs, ys):
      # Equivalent to:
      #    res = 0.
      #    for x, y in zip(xs, ys):
      #      res += x * y
      def body(carry, inputs):
        x, y = inputs
        return carry + x * y, carry
      return lax.scan(body, 0., (xs, ys))

    f_tf = jax_to_tf.convert(f_jax)
    if with_function:
      f_tf = tf.function(f_tf)
    arg = np.arange(10, dtype=np.float32)
    res_jax = f_jax(arg, arg)
    res_tf = f_tf(arg, arg)
    for r_jax, r_tf in zip(res_jax, res_tf):
      np.testing.assert_allclose(r_tf, r_jax)


if __name__ == "__main__":
  absltest.main()