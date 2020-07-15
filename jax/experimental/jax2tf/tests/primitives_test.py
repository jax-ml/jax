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
        self.ConvertAndCompare(f_jax, x, y)

  def test_concat(self):
    values = [np.array([1, 2], dtype=np.float32),
              np.array([1, 2], dtype=np.int32),
              np.array([1, 2], dtype=np.int8)]
    f_jax = jax.jit(lambda x: jnp.concatenate(x, axis=0))
    self.ConvertAndCompare(f_jax, values)

  @primitive_harness.parameterized(primitive_harness.lax_pad)
  def test_pad(self, harness: primitive_harness.Harness):
    # TODO: figure out the bfloat16 story
    if harness.params["dtype"] is dtypes.bfloat16:
      raise unittest.SkipTest("bfloat16 not implemented")
    # TODO: fix pad with negative padding in XLA (fixed on 06/16/2020)
    if any([lo < 0 or hi < 0 for lo, hi, mid in harness.params["pads"]]):
      raise unittest.SkipTest("pad with negative pad not supported")
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_sort)
  def test_sort(self, harness: primitive_harness.Harness):
    if harness.params["dtype"] is dtypes.bfloat16 or harness.params["dtype"] in jtu.dtypes.complex:
      # TODO: implement bfloat16/complex support in XlaSort
      raise unittest.SkipTest("bfloat16/complex support not implemented")
    if harness.params["dtype"] is dtypes.bool_ and len(harness.arg_descriptors) == 4:
      # TODO: _sort uses tfxla.key_value_sort to handle 2 operandes, but the operation is not compatible with boolean keys.
      raise unittest.SkipTest("boolean key key value sort not implemented")
    if harness.params["is_stable"]:
      # TODO: implement stable sort support in XlaSort
      raise unittest.SkipTest("stable sort not implemented")
    if harness.params["dimension"] != len(harness.params["shape"]) - 1:
      # TODO: implement sort on all axes
      raise unittest.SkipTest("conversion not implemented for axis != -1")
    if len(harness.arg_descriptors) > 4:
      # TODO: implement variable number of operands to XlaSort
      raise unittest.SkipTest("conversion not implemented for #operands > 2")
    if (jtu.device_under_test() == "gpu" and
        len(harness.arg_descriptors) == 4 and
        not harness.params["is_stable"]):
      # TODO: fix the TF GPU test
      raise unittest.SkipTest("GPU tests are running TF on CPU")
    # TODO: if we enable this test, we get the error
    #  iterating over `tf.Tensor` is not allowed: AutoGraph is disabled in this function.
    raise unittest.SkipTest("TODO: re-enable the sort test")
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_unary_elementwise)
  def test_unary_elementwise(self, harness: primitive_harness.Harness):
    dtype = harness.params["dtype"]
    if dtype is dtypes.bfloat16:
      raise unittest.SkipTest("bfloat16 not implemented")
    arg, = harness.dyn_args_maker(self.rng())
    custom_assert = None
    if harness.params["lax_name"] == "digamma":
      # digamma is not defined at 0 and -1
      def custom_assert(result_jax, result_tf):
        # lax.digamma returns NaN and tf.math.digamma returns inf
        special_cases = (arg == 0.) | (arg == -1.)
        nr_special_cases = np.count_nonzero(special_cases)
        self.assertAllClose(np.full((nr_special_cases,), dtype(np.nan)),
                            result_jax[special_cases])
        self.assertAllClose(np.full((nr_special_cases,), dtype(np.inf)),
                            result_tf[special_cases])
        # non-special cases are equal
        self.assertAllClose(result_jax[~ special_cases],
                            result_tf[~ special_cases])
    if harness.params["lax_name"] == "erf_inv":
      # TODO(necula): fix bug with erf_inv/f16
      if dtype is np.float16:
        raise unittest.SkipTest("TODO: fix bug")
      # erf_inf is not defined for arg <= -1 or arg >= 1
      def custom_assert(result_jax, result_tf):  # noqa: F811
        # for arg < -1 or arg > 1
        # lax.erf_inf returns NaN; tf.math.erf_inv return +/- inf
        special_cases = (arg < -1.) | (arg > 1.)
        nr_special_cases = np.count_nonzero(special_cases)
        self.assertAllClose(np.full((nr_special_cases,), dtype(np.nan)),
                            result_jax[special_cases])
        signs = np.where(arg[special_cases] < 0., -1., 1.)
        self.assertAllClose(np.full((nr_special_cases,), signs * dtype(np.inf)),
                            result_tf[special_cases])
        # non-special cases are equal
        self.assertAllClose(result_jax[~ special_cases],
                            result_tf[~ special_cases])
    self.ConvertAndCompare(harness.dyn_fun, arg, custom_assert=custom_assert)

  @primitive_harness.parameterized(primitive_harness.lax_bitwise_not)
  def test_bitwise_not(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_binary_elementwise)
  def test_binary_elementwise(self, harness):
    if harness.params["dtype"] is dtypes.bfloat16:
      raise unittest.SkipTest("bfloat16 not implemented")
    # TODO(necula): fix bug with igamma/f16
    if (harness.params["lax_name"] in ("igamma", "igammac") and
        harness.params["dtype"] is np.float16):
      raise unittest.SkipTest("TODO: fix bug")
    # TODO(necula): fix bug with nextafter/f16
    if (harness.params["lax_name"] == "nextafter" and
        harness.params["dtype"] is np.float16):
      raise unittest.SkipTest("TODO: understand unimplemented case")
    arg1, arg2 = harness.dyn_args_maker(self.rng())
    custom_assert = None
    if harness.params["lax_name"] == "igamma":
      # igamma is not defined when the first argument is <=0
      def custom_assert(result_jax, result_tf):
        # lax.igamma returns NaN when arg1 == arg2 == 0; tf.math.igamma returns 0
        special_cases = (arg1 == 0.) & (arg2 == 0.)
        nr_special_cases = np.count_nonzero(special_cases)
        self.assertAllClose(np.full((nr_special_cases,), np.nan),
                            result_jax[special_cases])
        self.assertAllClose(np.full((nr_special_cases,), 0.),
                            result_tf[special_cases])
        # non-special cases are equal
        self.assertAllClose(result_jax[~ special_cases],
                            result_tf[~ special_cases])
    if harness.params["lax_name"] == "igammac":
      # igammac is not defined when the first argument is <=0
      def custom_assert(result_jax, result_tf):  # noqa: F811
        # lax.igammac returns 1. when arg1 <= 0; tf.math.igammac returns NaN
        special_cases = (arg1 <= 0.) | (arg2 <= 0)
        nr_special_cases = np.count_nonzero(special_cases)
        self.assertAllClose(np.full((nr_special_cases,), 1.),
                            result_jax[special_cases])
        self.assertAllClose(np.full((nr_special_cases,), np.nan),
                            result_tf[special_cases])
        # non-special cases are equal
        self.assertAllClose(result_jax[~ special_cases],
                            result_tf[~ special_cases])
    self.ConvertAndCompare(harness.dyn_fun, arg1, arg2,
                           custom_assert=custom_assert)

  @primitive_harness.parameterized(primitive_harness.lax_binary_elementwise_logical)
  def test_binary_elementwise_logical(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))


  @primitive_harness.parameterized(primitive_harness.lax_betainc)
  def test_betainc(self, harness: primitive_harness.Harness):
    if harness.params["dtype"] is dtypes.bfloat16:
      raise unittest.SkipTest("bfloat16 not implemented")
    # TODO(necula): fix bug with betainc/f16
    if harness.params["dtype"] is np.float16:
      raise unittest.SkipTest("TODO: understand betainc/f16 bug")
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  # TODO(necula): combine tests that are identical except for the harness
  # wait until we get more experience with using harnesses.
  @primitive_harness.parameterized(primitive_harness.lax_shift_left)
  def test_shift_left(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_shift_right_logical)
  def test_shift_right_logical(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_shift_right_arithmetic)
  def test_shift_right_arithmetic(self, harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_slice)
  def test_slice(self, harness):
    # JAX.slice rejects negative indices; check, and skip jax2tf
    if any(si < 0 or si >= sh or li < 0 or li > sh
           for sh, si, li in zip(harness.params["shape"],
                                 harness.params["start_indices"],
                                 harness.params["limit_indices"])):
      with self.assertRaisesRegex(TypeError, ""):
        harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
    else:
      self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_dynamic_slice)
  def test_dynamic_slice(self, harness):
    # JAX.dynamic_slice rejects slice sizes too big; check this, and skip jax2tf
    args = harness.dyn_args_maker(self.rng())
    expect_exception = {}
    if any(li - si < 0 or li - si >= sh
           for sh, si, li in zip(harness.params["shape"],
                                 harness.params["start_indices"],
                                 harness.params["limit_indices"])):
      with self.assertRaisesRegex(TypeError, ""):
        harness.dyn_fun(*args)
      return

    # TF compiler gives an error for tf.slice(start_indices < 0)
    if any(si < 0 for si in harness.params["start_indices"]):
      expect_exception["compiled"] = (BaseException, "")

    if any(si == -100 for si in harness.params["start_indices"]):
      # TODO: for this case, TF gives errors except in graph mode
      expect_exception = dict(eager=(ValueError, "Invalid value in tensor used for shape"),
                              compiled=(BaseException, "Expected begin"))
    # TF gives errors for out of bounds access
    if any(si >= sh
          for sh, si, li in zip(harness.params["shape"],
                                harness.params["start_indices"],
                                harness.params["limit_indices"])):
      # TODO: for this case, TF gives errors in compiled mode
      expect_exception = dict(#eager=(ValueError, "Invalid value in tensor used for shape"),
                              compiled=(BaseException, ""))

    self.ConvertAndCompare(harness.dyn_fun, *args,
                           expect_exception=expect_exception)

  @primitive_harness.parameterized(primitive_harness.lax_dynamic_update_slice)
  def test_dynamic_update_slice(self, harness):
    # JAX.dynamic_update_slice rejects update slices too big; check, and skip jax2tf
    if any(ush > sh
           for sh, ush in zip(harness.params["shape"],
                              harness.params["update_shape"])):
      with self.assertRaisesRegex(TypeError, ""):
        harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
    else:
      self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_squeeze)
  def test_squeeze(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  @primitive_harness.parameterized(primitive_harness.lax_gather)
  def test_gather(self, harness: primitive_harness.Harness):
    self.ConvertAndCompare(harness.dyn_fun, *harness.dyn_args_maker(self.rng()))

  def test_boolean_gather(self):
    values = np.array([[True, True], [False, True], [False, False]],
                      dtype=np.bool_)
    indices = np.array([0, 1], dtype=np.int32)
    for axis in [0, 1]:
      f_jax = jax.jit(lambda v, i: jnp.take(v, i, axis=axis))  # pylint: disable=cell-var-from-loop
      # TODO: why can't we compile this code?
      self.ConvertAndCompare(f_jax, values, indices)

  def test_gather_rank_change(self):
    params = jnp.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [3.0, 3.5, 4.0]])
    indices = jnp.array([[1, 1, 2], [0, 1, 0]])
    f_jax = jax.jit(lambda i: params[i])
    self.ConvertAndCompare(f_jax, indices)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in REDUCE))
  def test_reduce_ops_with_numerical_input(self, f_jax):
    values = np.array([1, 2, 3], dtype=np.float32)
    self.ConvertAndCompare(f_jax, values)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in (jnp.cumsum, jnp.cumprod)))
  def test_cumulated_ops(self, f_jax):
    values = np.array([1, 2, 3], dtype=np.float32)
    self.ConvertAndCompare(f_jax, values)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{op.__name__}",
         op=op)
    for op in INDEX))
  def test_scatter_static(self, op):
    values = np.ones((5, 6), dtype=np.float32)
    update = np.float32(6.)
    f_jax = jax.jit(lambda v, u: op(v, jax.ops.index[::2, 3:], u))
    # TODO: compilation fails
    self.ConvertAndCompare(f_jax, values, update)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{f_jax.__name__}",
         f_jax=f_jax)
    for f_jax in REDUCE))
  def test_reduce_ops_with_boolean_input(self, f_jax):
    values = np.array([True, False, True], dtype=np.bool_)
    self.ConvertAndCompare(f_jax, values)

  def test_prngsplit(self):
    f_jax = jax.jit(lambda key: jax.random.split(key, 2))
    for rng_key in [jax.random.PRNGKey(42),
                    np.array([0, 0], dtype=np.uint32),
                    np.array([0xFFFFFFFF, 0], dtype=np.uint32),
                    np.array([0, 0xFFFFFFFF], dtype=np.uint32),
                    np.array([0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)
                    ]:
      self.ConvertAndCompare(f_jax, rng_key)

  def test_zeros_like(self):
    v = np.float32(2.)
    f_jax = jax.ad_util.zeros_like_jaxval
    self.ConvertAndCompare(f_jax, v)

  def test_stop_gradient(self):
    f = jax2tf.convert(lax.stop_gradient)
    self.assertEqual(f(tf.ones([])), 1.)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
