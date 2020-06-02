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

import unittest

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import dtypes
import jax.lax as lax
import jax.numpy as jnp
from jax import test_util as jtu
import numpy as np
import tensorflow as tf  # type: ignore[import]

from jax.experimental import jax_to_tf
from . import tf_test_util
from . import primitive_harness


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


class TfOpsTest(tf_test_util.JaxToTfTestCase):

  def test_basics(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    _, res_tf = self.ConvertAndCompare(f_jax, 0.7)
    self.assertIsInstance(res_tf, tf.Tensor)

  def test_variable_input(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    f_tf = jax_to_tf.convert(f_jax)
    v = tf.Variable(0.7)
    self.assertIsInstance(f_tf(v), tf.Tensor)
    self.assertAllClose(f_jax(0.7), f_tf(v))

  def test_jit(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    self.ConvertAndCompare(f_jax, 0.7)

  def test_nested_jit(self):
    f_jax = jax.jit(lambda x: jnp.sin(jax.jit(jnp.cos)(x)))
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
    self.assertAllClose(f_tf(mk_sharded(jnp.zeros)).numpy(),
                        np.zeros([n]))
    self.assertAllClose(f_tf(mk_sharded(jnp.ones)).numpy(),
                        np.ones([n]))

  def test_function(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    self.ConvertAndCompare(f_jax, 0.7, with_function=True)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in [jnp.add, jnp.subtract, jnp.multiply, jnp.divide,
                  jnp.less, jnp.less_equal, jnp.equal, jnp.greater,
                  jnp.greater_equal, jnp.not_equal, jnp.maximum,
                  jnp.minimum]))
  def test_type_promotion(self, f_jax=jnp.add):
    # We only test a few types here, as tensorflow does not support many
    # types like uint* or bool in binary ops.
    types = [np.int32, np.int64, np.float32]
    for x_dtype in types:
      for y_dtype in types:
        x = np.array([1, 2], dtype=x_dtype)
        y = np.array([3, 4], dtype=y_dtype)
        self.ConvertAndCompare(f_jax, x, y, with_function=True)

  def test_concat(self):
    values = [np.array([1, 2], dtype=np.float32),
              np.array([1, 2], dtype=np.int32),
              np.array([1, 2], dtype=np.int8)]
    f_jax = jax.jit(lambda x: jnp.concatenate(x, axis=0))
    self.ConvertAndCompare(f_jax, values, with_function=True)

  @primitive_harness.parameterized(primitive_harness.lax_pad)
  def test_pad(self, harness: primitive_harness.Harness):
    if harness.params["dtype"] is dtypes.bfloat16:
      raise unittest.SkipTest("bfloat16 not implemented")
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           with_function=True)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in LAX_ELEMENTWISE_UNARY))
  def test_unary_elementwise(self, f_jax=lax.abs):
    x = np.array([-1.6, -1.4, -1.0, 0.0, 0.1, 0.2, 1, 1.4, 1.6],
                 dtype=np.float32)
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    r_jax = f_jax(x)
    r_tf = f_tf(x)
    self.assertAllClose(r_jax[np.isfinite(r_jax)],
                        r_tf[np.isfinite(r_tf)], atol=1e-4)

  def test_bitwise_not(self):
    x = np.array([-1, 3, -2, 0, 0, 2, 1, 3], dtype=np.int32)
    f_jax = jax.jit(lax.bitwise_not)
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    r_jax = f_jax(x)
    r_tf = f_tf(x)
    self.assertAllClose(r_jax[np.isfinite(r_jax)],
                        r_tf[np.isfinite(r_tf)], atol=1e-4)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in LAX_ELEMENTWISE_BINARY))
  def test_binary_elementwise(self, f_jax=lax.add):
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
    self.assertAllClose(r_jax[np.isfinite(r_jax)],
                        r_tf[np.isfinite(r_tf)], atol=1e-4)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in LAX_LOGICAL_ELEMENTWISE_BINARY))
  def test_binary_logical_elementwise(self, f_jax):
    a = np.array([1, 3, 2, 0, 0, 2, 1, 3], dtype=np.uint32)
    b = np.array([1, 2, 3, 0, 1, 0, 2, 3], dtype=np.uint32)
    f_tf = tf.function(jax_to_tf.convert(f_jax))
    r_jax = f_jax(a, b)
    r_tf = f_tf(a, b)
    self.assertAllClose(r_jax[np.isfinite(r_jax)],
                        r_tf[np.isfinite(r_tf)], atol=1e-4)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in (lax.betainc,)))
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
    self.assertAllClose(r_jax[np.isfinite(r_jax)],
                        r_tf[np.isfinite(r_tf)], atol=1e-4)

  @primitive_harness.parameterized(primitive_harness.lax_squeeze)
  def test_squeeze(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           with_function=True)

  def test_gather(self):
    values = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    for axis in (0, 1):
      f_jax = jax.jit(lambda v, i: jnp.take(v, i, axis=axis))  # pylint: disable=cell-var-from-loop
      self.ConvertAndCompare(f_jax, values, indices, with_function=True)

  def test_boolean_gather(self):
    values = np.array([[True, True], [False, True], [False, False]],
                      dtype=np.bool)
    indices = np.array([0, 1], dtype=np.int32)
    for axis in [0, 1]:
      f_jax = jax.jit(lambda v, i: jnp.take(v, i, axis=axis))  # pylint: disable=cell-var-from-loop
      self.ConvertAndCompare(f_jax, values, indices, with_function=True)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in REDUCE))
  def test_reduce_ops_with_numerical_input(self, f_jax):
    values = [np.array([1, 2, 3], dtype=np.float32)]
    self.ConvertAndCompare(f_jax, values, with_function=True)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in (jnp.cumsum, jnp.cumprod)))
  def test_cumulated_ops(self, f_jax):
    values = np.array([1, 2, 3], dtype=np.float32)
    self.ConvertAndCompare(f_jax, values, with_function=True)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{op.__name__}",
         op=op)
    for op in INDEX))
  def test_scatter_static(self, op):
    values = np.ones((5, 6), dtype=np.float32)
    update = np.float32(6.)
    f_jax = jax.jit(lambda v, u: op(v, jax.ops.index[::2, 3:], u))
    self.ConvertAndCompare(f_jax, values, update, with_function=True)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in REDUCE))
  def test_reduce_ops_with_boolean_input(self, f_jax):
    values = [np.array([True, False, True], dtype=np.bool)]
    self.ConvertAndCompare(f_jax, values, with_function=True)

  def test_gather_rank_change(self):
    params = jnp.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [3.0, 3.5, 4.0]])
    indices = jnp.array([[1, 1, 2], [0, 1, 0]])
    f_jax = jax.jit(lambda i: params[i])
    self.ConvertAndCompare(f_jax, indices, with_function=True)

  def test_prngsplit(self):
    f_jax = jax.jit(lambda key: jax.random.split(key, 2))
    for rng_key in [jax.random.PRNGKey(42),
                    np.array([0, 0], dtype=np.uint32),
                    np.array([0xFFFFFFFF, 0], dtype=np.uint32),
                    np.array([0, 0xFFFFFFFF], dtype=np.uint32),
                    np.array([0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)
                    ]:
      self.ConvertAndCompare(f_jax, rng_key, with_function=True)

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
    self.ConvertAndCompare(f_jax, v)

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


if __name__ == "__main__":
  absltest.main()
