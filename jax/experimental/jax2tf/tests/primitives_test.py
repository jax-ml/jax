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
"""Tests for JAX primitive coverage."""

import unittest

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import dtypes
from jax import lax
from jax import numpy as jnp
from jax import test_util as jtu
from jax.config import config
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util
from jax.interpreters import xla
import numpy as np
import tensorflow as tf  # type: ignore[import]

config.parse_flags_with_absl()

# Import after parsing flags
from jax.experimental.jax2tf.tests import primitive_harness

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

LAX_LOGICAL_ELEMENTWISE_UNARY = (
    lax.bitwise_not,
)

LAX_LOGICAL_ELEMENTWISE_BINARY = (
    lax.bitwise_and,
    lax.bitwise_or,
    lax.bitwise_xor,
    lax.shift_left,
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


class JaxPrimitiveTest(tf_test_util.JaxToTfTestCase):

  def test_primitive_coverage(self):
    """Fail if there are JAX primitives that are not implemented."""
    # Harvest primitives from XLA translation tables
    all_primitives = (set(xla.translations)
                      | set(xla.backend_specific_translations['cpu'])
                      | set(xla.backend_specific_translations['gpu'])
                      | set(xla.backend_specific_translations['tpu'])
                      | set(xla.initial_style_translations)
                      | set(xla.parallel_translations))

    tf_impl = set(jax.experimental.jax2tf.jax2tf.tf_impl)
    tf_not_yet_impl = set(jax.experimental.jax2tf.jax2tf.tf_not_yet_impl)

    all_primitives = tuple(sorted(all_primitives, key=str))
    for p in all_primitives:
      if p in tf_not_yet_impl:
        self.assertNotIn(p, tf_impl)  # Should not be in both tf_impl and tf_not_yet_impl
      else:
        self.assertIn(p, tf_impl)

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
    # TODO: figure out the bfloat16 story
    if harness.params["dtype"] is dtypes.bfloat16:
      raise unittest.SkipTest("bfloat16 not implemented")
    # TODO: fix pad with negative padding in XLA (fixed on 06/16/2020)
    if any([lo < 0 or hi < 0 for lo, hi, mid in harness.params["pads"]]):
      raise unittest.SkipTest("pad with negative pad not supported")
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           with_function=False)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in LAX_ELEMENTWISE_UNARY))
  def test_unary_elementwise(self, f_jax=lax.abs):
    x = np.array([-1.6, -1.4, -1.0, 0.0, 0.1, 0.2, 1, 1.4, 1.6],
                 dtype=np.float32)
    f_tf = tf.function(jax2tf.convert(f_jax))
    r_jax = f_jax(x)
    r_tf = f_tf(x)
    self.assertAllClose(r_jax[np.isfinite(r_jax)],
                        r_tf[np.isfinite(r_tf)], atol=1e-4)

  def test_bitwise_not(self):
    x = np.array([-1, 3, -2, 0, 0, 2, 1, 3], dtype=np.int32)
    f_jax = jax.jit(lax.bitwise_not)
    f_tf = tf.function(jax2tf.convert(f_jax))
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
    f_tf = tf.function(jax2tf.convert(f_jax))
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
    f_tf = tf.function(jax2tf.convert(f_jax))
    r_jax = f_jax(a, b)
    r_tf = f_tf(a, b)
    self.assertAllClose(r_jax[np.isfinite(r_jax)],
                        r_tf[np.isfinite(r_tf)], atol=1e-4)
    # Checks support for bools.
    if f_jax in (lax.bitwise_and, lax.bitwise_or, lax.bitwise_xor):
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        f_tf = tf.function(jax2tf.convert(f_jax))
        r_jax = f_jax(a, b)
        r_tf = f_tf(a, b)
        self.assertArraysEqual(r_jax, np.asarray(r_tf))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in LAX_LOGICAL_ELEMENTWISE_UNARY))
  def test_unary_logical_elementwise(self, f_jax):
    a = np.array([1, 3, 2, 0, 0, 2, 1, 3], dtype=np.uint32)
    f_tf = tf.function(jax2tf.convert(f_jax))
    r_jax = f_jax(a)
    r_tf = f_tf(a)
    self.assertAllClose(r_jax[np.isfinite(r_jax)],
                        r_tf[np.isfinite(r_tf)], atol=1e-4)
    # Checks support for bools.
    a = np.array([True, False])
    f_tf = tf.function(jax2tf.convert(f_jax))
    r_jax = f_jax(a)
    r_tf = f_tf(a)
    self.assertArraysEqual(r_jax, np.asarray(r_tf))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in LAX_LOGICAL_ELEMENTWISE_BINARY))
  def test_binary_logical_elementwise_bool(self, f_jax):
    if f_jax == lax.shift_left:
      self.skipTest("Shift of bool not supported")
    a = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.bool_)
    b = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool_)
    f_tf = tf.function(jax2tf.convert(f_jax))
    r_jax = f_jax(a, b)
    r_tf = f_tf(a, b)
    self.assertAllClose(r_jax, r_tf)

  # TODO(necula): combine tests that are identical except for the harness
  # wait until we get more experience with using harnesses.
  @primitive_harness.parameterized(primitive_harness.lax_shift_left)
  def test_shift_left(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           with_function=True)

  @primitive_harness.parameterized(primitive_harness.lax_shift_right_logical)
  def test_shift_right_logical(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           with_function=True)

  @primitive_harness.parameterized(primitive_harness.lax_shift_right_arithmetic)
  def test_shift_right_arithmetic(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           with_function=True)

  @primitive_harness.parameterized(primitive_harness.lax_slice)
  def test_slice(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           with_function=True)

  @primitive_harness.parameterized(primitive_harness.lax_dynamic_slice)
  def test_dynamic_slice(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           with_function=True)

  @primitive_harness.parameterized(primitive_harness.lax_dynamic_update_slice)
  def test_dynamic_update_slice(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()),
                           with_function=True)

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
    f_tf = tf.function(jax2tf.convert(f_jax))
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
                      dtype=np.bool_)
    indices = np.array([0, 1], dtype=np.int32)
    for axis in [0, 1]:
      f_jax = jax.jit(lambda v, i: jnp.take(v, i, axis=axis))  # pylint: disable=cell-var-from-loop
      self.ConvertAndCompare(f_jax, values, indices, with_function=True)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in REDUCE))
  def test_reduce_ops_with_numerical_input(self, f_jax):
    values = np.array([1, 2, 3], dtype=np.float32)
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
    values = np.array([True, False, True], dtype=np.bool_)
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


  def test_zeros_like(self):
    v = np.float32(2.)
    f_jax = jax.ad_util.zeros_like_jaxval
    self.ConvertAndCompare(f_jax, v)

  def test_stop_gradient(self):
    f = jax2tf.convert(lax.stop_gradient)
    self.assertEqual(f(tf.ones([])), 1.)

if __name__ == "__main__":
  absltest.main()
