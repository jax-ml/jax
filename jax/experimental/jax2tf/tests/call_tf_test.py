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
"""Tests for call_tf."""
from functools import partial
from typing import Callable, Dict, Tuple
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import dtypes
from jax import lax
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax.config import config
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util

import numpy as np

try:
  import tensorflow as tf  # type: ignore[import]
except ImportError:
  tf = None

config.parse_flags_with_absl()


def _maybe_jit(with_jit: bool, func: Callable) -> Callable:
  if with_jit:
    return jax.jit(func)
  else:
    return func

def _named_test(**kwargs):
  return dict(kwargs,
              testcase_name = "_".join([f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())]))

_parameterized_jit = parameterized.named_parameters(
    _named_test(with_jit=with_jit)
    for with_jit in [True, False])

_call_tf_non_compileable_error = "Error compiling TensorFlow function. call_tf can used in a staged context .* only with compileable functions"
_call_tf_dynamic_shape_error = "Compiled TensorFlow function has dynamic output shape.* call_tf can used in a staged context .* only with compileable functions"


class CallTfTest(tf_test_util.JaxToTfTestCase):

  def setUp(self):
    if tf is None:
      raise unittest.SkipTest("Test requires tensorflow")
    # TODO(b/171320191): this line works around a missing context initialization
    # bug in TensorFlow.
    _ = tf.add(1, 1)
    super().setUp()

  @_parameterized_jit
  def test_eval_scalar_arg(self, with_jit=True):
    def f_tf(x):
      return tf.math.sin(x)
    x = 3.
    res = _maybe_jit(with_jit, jax2tf.call_tf(f_tf))(x)
    self.assertAllClose(jnp.sin(x), res)

  @_parameterized_jit
  def test_eval_scalar_res(self, with_jit=True):
    x = 3.
    res = _maybe_jit(with_jit, jax2tf.call_tf(lambda x: 4.))(x)
    self.assertAllClose(4., res, check_dtypes=False)

  @_parameterized_jit
  def test_eval_numpy_arg(self, with_jit=True):
    x = np.ones((2, 3), dtype=np.float32)
    res = _maybe_jit(with_jit, jax2tf.call_tf(tf.math.sin))(x)
    self.assertAllClose(jnp.sin(x), res)

  @_parameterized_jit
  def test_eval_numpy_res(self, with_jit=False):
    x = np.ones((2, 3))
    res = _maybe_jit(with_jit, jax2tf.call_tf(lambda _: x))(x)
    self.assertAllClose(x, res)

  def test_eval_numpy_no_copy(self):
    if jtu.device_under_test() != "cpu":
      raise unittest.SkipTest("no_copy test works only on CPU")
    # For ndarray, zero-copy only works for sufficiently-aligned arrays.
    x = np.ones((16, 16), dtype=np.float32)
    res = jax2tf.call_tf(lambda x: x)(x)
    self.assertAllClose(x, res)
    self.assertTrue(np.shares_memory(x, res))

  @_parameterized_jit
  def test_eval_devicearray_arg(self, with_jit=False):
    x = jnp.ones((2, 3), dtype=np.float32)
    res = _maybe_jit(with_jit, jax2tf.call_tf(tf.math.sin))(x)
    self.assertAllClose(jnp.sin(x), res)

  def test_eval_devicearray_no_copy(self):
    if jtu.device_under_test() != "cpu":
      # TODO(necula): add tests for GPU and TPU
      raise unittest.SkipTest("no_copy test works only on CPU")
    # For DeviceArray zero-copy works even if not aligned
    x = jnp.ones((3, 3))
    res = jax2tf.call_tf(lambda x: x)(x)
    self.assertAllClose(x, res)
    self.assertTrue(np.shares_memory(x, res))

  @_parameterized_jit
  def test_eval_pytree(self, with_jit=True):

    def fun_tf(x: Dict, y: Tuple) -> Tuple:
      return (x["first"] * x["second"], y[0] + y[1])

    x = dict(first=np.float32(3.), second=np.float32(4.))
    y = (np.float64(5.), np.float64(6.))
    fun_jax = _maybe_jit(with_jit, jax2tf.call_tf(fun_tf))
    res = fun_jax(x, y)
    self.assertAllClose((np.float32(12.), np.float64(11.)), res)

  def test_eval_non_compileable_strings(self):
    # Check that in op-by-op we call a function in eager mode.
    def f_tf_non_compileable(x):
      return tf.strings.length(tf.strings.format("Hello {}!", [x]))

    f_jax = jax2tf.call_tf(f_tf_non_compileable)
    x = np.float32(0.7)
    self.assertAllClose(f_tf_non_compileable(x).numpy(), f_jax(x))
    with self.assertRaisesRegex(ValueError,
                                _call_tf_non_compileable_error):
      jax.jit(f_jax)(x)

    with self.assertRaisesRegex(ValueError,
                                _call_tf_non_compileable_error):
      lax.cond(True, lambda x: f_jax(x), lambda x: f_jax(x), x)

  def test_eval_non_compileable_dynamic_shape(self):
    # Check that in op-by-op we call a function in eager mode.
    def f_tf_non_compileable(x):
      return tf.cond(x[0], lambda: x[1:], lambda: x)

    f_jax = jax2tf.call_tf(f_tf_non_compileable)
    x = np.array([True, False], dtype=np.bool_)
    self.assertAllClose(f_tf_non_compileable(x), f_jax(x))

    with self.assertRaisesRegex(ValueError,
                                _call_tf_dynamic_shape_error):
      jax.jit(f_jax)(x)

  @_parameterized_jit
  def test_control_flow(self, with_jit=True):

    def times_5_tf(x):
      # Multiply x * 5 using a loop
      c = lambda i, acc: tf.less(i, 5)
      b = lambda i, acc: (tf.add(i, 1), tf.add(acc, x))
      _, acc = tf.while_loop(c, b, [tf.constant(0), tf.constant(0.)])
      return acc

    def fun_jax(x):
      # Calls times_5_tf 3 times in a loop
      def body(_, acc):
        return jax2tf.call_tf(times_5_tf)(acc)

      return lax.fori_loop(0, 3, body, x)

    x = np.float32(3.)
    res = _maybe_jit(with_jit, fun_jax)(x)
    self.assertAllClose(np.float32(x * 5 * 5 * 5), res)

  @parameterized.named_parameters(
      dict(
          testcase_name=f"_{dtype.__name__}{'_jit' if with_jit else ''}",
          dtype=dtype,
          with_jit=with_jit)
      for dtype in set(jtu.dtypes.all) - {np.bool_}
      for with_jit in [True, False])
  def test_dtypes(self, dtype=np.int32, with_jit=True):

    def fun_tf(x):
      # AddV2 supports more types
      return tf.raw_ops.AddV2(x=x, y=tf.constant(3, dtype=dtype))

    def fun_jax(x):
      return jax2tf.call_tf(fun_tf)(x) + x

    x = np.ones((3,), dtype=dtype)
    res = _maybe_jit(with_jit, fun_jax)(x)
    self.assertAllClose(dtype(2 * x + 3), res)

  @_parameterized_jit
  def test_bool(self, with_jit=False):

    def fun_tf(x, y):
      return tf.math.logical_and(x, y)

    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([True, True, False, False], dtype=np.bool_)
    res = _maybe_jit(with_jit, jax2tf.call_tf(fun_tf))(x, y)
    self.assertAllClose(
        np.array([True, False, False, False], dtype=np.bool_), res)

  @_parameterized_jit
  def test_x64_input(self, with_jit=True):
    def f_tf(x):
      return tf.math.sin(x)

    x = 5.  # TF interprets this as f64
    res_call_tf = _maybe_jit(with_jit, jax2tf.call_tf(f_tf))(x)
    res_jax = jnp.sin(x)
    self.assertAllClose(res_call_tf, res_jax)

  @_parameterized_jit
  def test_x64_output(self, with_jit=True):
    def f_tf(x):
      return (tf.constant(3., tf.float64), x)

    x = np.float32(5.)
    res_call_tf = _maybe_jit(with_jit, jax2tf.call_tf(f_tf))(x)
    res_jax = (3., x)
    self.assertAllClose(res_call_tf, res_jax)

    res_call_tf_jit = jax.jit(jax2tf.call_tf(f_tf))(x)
    self.assertAllClose(res_call_tf_jit, res_jax)

  @_parameterized_jit
  def test_with_var_read(self, with_jit=True):
    if jtu.device_under_test() == "gpu":
      raise unittest.SkipTest("Test fails on GPU")
    outer_var_array = np.array([3., 4.], dtype=np.float32)
    outer_var = tf.Variable(outer_var_array)

    def fun_tf(x):
      return x * outer_var + 1.

    x = np.array([2., 5.,], dtype=np.float32)
    res = _maybe_jit(with_jit, jax2tf.call_tf(fun_tf))(x)
    self.assertAllClose(x * outer_var_array + 1., res, check_dtypes=False)

  @_parameterized_jit
  def test_with_var_read_x64(self, with_jit=True):
    if jtu.device_under_test() == "gpu":
      raise unittest.SkipTest("Test fails on GPU")
    outer_var_array = np.array([3., 4.], dtype=np.float64)
    outer_var = tf.Variable(outer_var_array)

    def fun_tf(x):
      return x * tf.cast(outer_var, x.dtype) + 1.

    x = np.array([2., 5.,], dtype=np.float32)
    res = _maybe_jit(with_jit, jax2tf.call_tf(fun_tf))(x)
    self.assertAllClose(x * outer_var_array + 1., res, check_dtypes=False)

  def test_with_var_different_shape(self):
    # See https://github.com/google/jax/issues/6050
    if jtu.device_under_test() == "gpu":
      raise unittest.SkipTest("Test fails on GPU")
    v = tf.Variable((4., 2.), dtype=tf.float32)

    def tf_func(x):
      return v + x
    x = np.float32(123.)
    tf_out = tf_func(x)

    jax_func = jax.jit(jax2tf.call_tf(tf_func))
    jax_out = jax_func(x)

    self.assertAllClose(tf_out, jax_out, check_dtypes=False)

  @_parameterized_jit
  def test_with_var_write_error(self, with_jit=True):
    if with_jit:
      raise unittest.SkipTest("variable writes not yet working")
    outer_var = tf.Variable(3., dtype=np.float32)

    def fun_tf(x):
      outer_var.assign(tf.constant(4.))
      return x * outer_var + 1.

    x = np.float32(2.)
    res = _maybe_jit(with_jit, jax2tf.call_tf(fun_tf))(x)
    self.assertAllClose(x * 4. + 1, res, check_dtypes=False)

  @_parameterized_jit
  def test_with_tensor_capture(self, with_jit=True):
    outer_tensor = tf.constant(3., dtype=np.float32)

    def fun_tf(x):
      return x * outer_tensor + 1.

    x = np.float32(2.)
    res = _maybe_jit(with_jit, jax2tf.call_tf(fun_tf))(x)
    self.assertAllClose(x * 3. + 1., res, check_dtypes=False)

  @_parameterized_jit
  def test_with_tensor_capture_x64(self, with_jit=True):
    outer_tensor = tf.constant(3., dtype=np.float64)

    def fun_tf(x):
      return x * tf.cast(outer_tensor * 3.14, tf.float32) + 1.

    x = np.float32(2.)
    res = _maybe_jit(with_jit, jax2tf.call_tf(fun_tf))(x)
    self.assertAllClose(x * 3. * 3.14 + 1., res, check_dtypes=False)

  @_parameterized_jit
  def test_with_value_capture(self, with_jit=True):
    outer_val = np.array(3., dtype=np.float32)

    def fun_tf(x):
      return x * outer_val + 1.

    x = np.float32(2.)
    res = _maybe_jit(with_jit, jax2tf.call_tf(fun_tf))(x)
    self.assertAllClose(x * 3. + 1., res, check_dtypes=False)

  @_parameterized_jit
  def test_with_multiple_capture(self, with_jit=True):
    if jtu.device_under_test() == "gpu":
      raise unittest.SkipTest("Test fails on GPU")
    v2 = tf.Variable(2., dtype=np.float32)
    v3 = tf.Variable(3., dtype=np.float32)
    t4 = tf.constant(4., dtype=np.float32)
    t5 = tf.constant(5., dtype=np.float32)

    def fun_tf(x):
      return (x * v3 + t4 + v2) * v3 + t5

    x = np.float32(2.)
    res = _maybe_jit(with_jit, jax2tf.call_tf(fun_tf))(x)
    self.assertAllClose((x * 3. + 4. + 2.) * 3. + 5., res, check_dtypes=False)

  @_parameterized_jit
  def test_grad(self, with_jit=False):
    x = np.float32(3.)
    res = _maybe_jit(with_jit, jax.grad(jax2tf.call_tf(tf.math.sin)))(x)
    self.assertAllClose(np.cos(x), res)

  @_parameterized_jit
  def test_grad_pytree(self, with_jit=False):

    def fun_tf(x: Dict, y: Tuple) -> Tuple:
      return x["first"] * x["second"] + 3. * y[0] + 4. * y[1]

    x = dict(first=np.float32(3.), second=np.float32(4.))
    y = (np.float32(5.), np.float32(6.))
    grad_x = _maybe_jit(with_jit, jax.grad(jax2tf.call_tf(fun_tf)))(x, y)
    self.assertAllClose(
        dict(first=np.float32(4.), second=np.float32(3.)), grad_x)

  def test_grad_nested(self):
    # We embed the call_tf function in a larger function whose gradient we take
    # It is relevant here that the cotangents flowing through the call_tf
    # function are not scalars.

    b = np.array([[11., 12., 13.], [21., 22., 23.]], dtype=np.float32)  # [2, 3]
    c = np.array([[31., 32.], [41., 42.], [51., 52.], [61., 62.]], dtype=np.float32)  # [4, 2]
    x_dict = dict(b=b, c=c)  # b:[2, 3], c=[4, 2]
    # res: dict(r:[4, 3], s:[4, 2])
    def f_tf(x_dict):
      return dict(r=tf.matmul(x_dict["c"], x_dict["b"]), s=7. * x_dict["c"])

    @jax.jit  # To recognize it in jaxpr
    def f_jax(x_dict):
      return dict(r=jnp.matmul(x_dict["c"], x_dict["b"]), s=7. * x_dict["c"])

    def loss(functional, x_dict):
      prediction = functional(x_dict)  # r:[4, 3], s:[4, 2]
      weights = np.array([1., 2., 3., 4.], dtype=np.float32)  # [4]
      weighted_pred = jnp.matmul(weights, prediction["r"])  # [3]
      return jnp.sum(weighted_pred) + 4. * jnp.sum(prediction["s"])

    g_fun_with_tf = jax.grad(partial(loss, jax2tf.call_tf(f_tf)))
    g_fun_with_jax = jax.grad(partial(loss, f_jax))

    g_tf = g_fun_with_tf(x_dict)
    g_jax = g_fun_with_jax(x_dict)
    self.assertAllClose(g_jax, g_tf)

  def test_grad_int_argument(self):
    # Similar to https://github.com/google/jax/issues/6975
    # state is a pytree that contains an integer and a boolean.
    # The function returns an integer and a boolean.
    def f(param, state, x):
      return param * x, state

    param = np.array([0.7, 0.9], dtype=np.float32)
    state = dict(array=np.float32(1.), counter=7, truth=True)
    x = np.float32(3.)

    # tf.function is important, without it the bug does not appear
    f_call_tf = jax2tf.call_tf(f)
    g_call_tf = jax.grad(lambda *args: jnp.sum(f_call_tf(*args)[0]))(param, state, x)
    g = jax.grad(lambda *args: jnp.sum(f(*args)[0]))(param, state, x)
    self.assertAllClose(g_call_tf, g)

  def test_grad_int_argument_unused(self):
    batch_size = 5
    inputs = np.ones((batch_size, 3), dtype=np.float32)
    rng = np.array([1, 2], dtype=np.uint32)
    params = np.float32(.5)

    # rng is integer, unused
    def jax_model(params, rng, inputs):
      return jnp.ones([batch_size, 2], dtype=jnp.float32)

    tf_model = jax2tf.convert(jax_model, with_gradient=True)

    def _loss_fn(inference_fn, params, rng, inputs):
      prediction = inference_fn(params, rng, inputs)
      return jnp.mean(prediction)

    jax_loss_fn = partial(_loss_fn, jax_model)
    jax_grad = jax.grad(jax_loss_fn)(params, rng, inputs)

    paramsv = tf.Variable(params)
    with tf.GradientTape() as tape:
      tf_prediction = tf_model(paramsv, rng, inputs)
      tf_loss = tf.reduce_mean(tf_prediction)

      tf_grad = tape.gradient(tf_loss, paramsv)
    self.assertAllClose(jax_grad, tf_grad.numpy())

    call_tf_loss_fn = partial(_loss_fn, jax2tf.call_tf(tf_model))
    call_tf_grad = jax.grad(call_tf_loss_fn)(params, rng, inputs)
    self.assertAllClose(jax_grad, call_tf_grad)

  def test_grad_with_float0_result(self):
    # Gradient over integer-argument functions, with float0 result
    def f_jax(x, y):  # x is an int, y is a float; res is a (int, float)
      return (2 * x, 2 * x + y * y)
    def f_tf(x, y):
      # TF needs explicit casts
      return (2 * x, tf.cast(2 * x, dtype=y.dtype) + y * y)

    def wrapper(functional, x, y):  # x: i32
      return jnp.sum(2. * functional(3 * x, 4. * y)[1])

    grad_g = jax.grad(partial(wrapper, f_jax),
                      allow_int=True, argnums=(0, 1))
    grad_g_call_tf = jax.grad(partial(wrapper, jax2tf.call_tf(f_tf)),
                              allow_int=True, argnums=(0, 1))

    x = np.int32(2)
    y = np.float32(3.)
    g_jax = grad_g(x, y)
    g_call_tf = grad_g_call_tf(x, y)
    self.assertEqual(g_jax[0].dtype, dtypes.float0)
    self.assertEqual(g_call_tf[0].dtype, dtypes.float0)
    self.assertAllClose(g_jax[1], g_call_tf[1])

  @_parameterized_jit
  def test_grad_custom(self, with_jit=False):

    @tf.custom_gradient
    def func_square_tf(x):
      # Like x ** 2, but with custom grad 3. * x
      def grad(dy, variables=None):
        # dy, = dys
        return 3. * x * dy,

      return x * x, grad

    x = np.float32(4.)
    grad_x = _maybe_jit(with_jit, jax.grad(jax2tf.call_tf(func_square_tf)))(x)
    self.assertAllClose(np.float32(3.) * x, grad_x)

  @parameterized.named_parameters(
      dict(
          testcase_name=f"_degree={degree}{'_jit' if with_jit else ''}",
          degree=degree,
          with_jit=with_jit)
      for degree in [1, 2, 3, 4]
      for with_jit in [True, False])
  def test_higher_order_grad(self, degree=2, with_jit=False):

    def fun_tf(x):
      return 2. * x * x * x

    def fun_jax(x):
      return 3. * _maybe_jit(with_jit, jax2tf.call_tf(fun_tf))(x)

    def fun_jax_pure(x):
      return 3. * fun_tf(x)

    grad_jax = fun_jax
    grad_jax_pure = fun_jax_pure
    for _ in range(degree):
      grad_jax = jax.grad(grad_jax)
      grad_jax_pure = jax.grad(grad_jax_pure)

    res_jax = grad_jax(np.float32(5.))
    print(f"Grad of {degree} degree is {res_jax}")
    self.assertAllClose(res_jax, grad_jax_pure(np.float32(5.)))

  def test_pmap(self):
    print(f"Running test_pmap on {jax.local_device_count()} devices")

    def plus_2_tf(x):
      return tf.math.add(2., x)

    def fun_jax(x):
      return np.float32(3.) * jax2tf.call_tf(plus_2_tf)(x)

    x = np.arange(jax.local_device_count(), dtype=np.float32)
    res = jax.pmap(fun_jax)(x)
    self.assertAllClose(np.float32(3. * (x + 2)), res)

  def test_function_compile_time_constant_inputs(self):
    # Call a function for which shape inference does not give an output
    # shape.
    x = np.array([1, 2, 3], dtype=np.int32)
    def fun_tf(x):  # x:i32[3]
      # Indexing with a dynamic slice makes the TF shape inference return
      # a partially known shape.
      end_idx = x[1]
      res = x[0:end_idx]
      return res

    # Call in eager mode. Should work!
    res1 = jax2tf.call_tf(fun_tf)(x)
    self.assertAllClose(x[0:x[1]], res1)

    # Now under jit, should fail because the function is not compileable
    with self.assertRaisesRegex(ValueError,
                                "Compiled TensorFlow function has unexpected parameter types"):
      fun_jax = jax.jit(jax2tf.call_tf(fun_tf))
      fun_jax(x)

  def test_experimental_get_compiler_ir_design_doc(self):
    # Not a test of call_tf, but more of how experimental_get_compiler_ir works.
    # Examples are from the design doc.

    # Constant slice. This is the common case.
    x = np.zeros((10,), dtype=np.int32)

    def fun_tf(x):
      begin = 0
      return x[begin:5]  # x must be a compile-time constant

    hlo = tf.function(fun_tf, jit_compile=True).experimental_get_compiler_ir(x)()
    self.assertIn("(arg0.1: s32[10]) -> s32[5]", hlo)

    # Non-constant slice, but compile-time constant depending only on values.
    x = np.zeros((10,), dtype=np.int32)

    def fun_tf(x):
      begin = x[0]
      return x[begin:5]  # x must be a compile-time constant

    hlo = tf.function(fun_tf, jit_compile=True).experimental_get_compiler_ir(x)()
    self.assertIn("() -> s32[5]", hlo)
    x = np.ones((10,), dtype=np.int32)
    hlo = tf.function(fun_tf, jit_compile=True).experimental_get_compiler_ir(x)()
    self.assertIn("() -> s32[4]", hlo)

    # Non-constant slice, but compile-time constant depending only on shapes.
    x = np.zeros((10,), dtype=np.int32)

    def fun_tf(x):
      begin = tf.shape(x)[0] - 2  # begin is a compile-time constant, even if x is not
      return x[begin:]

    hlo = tf.function(fun_tf, jit_compile=True).experimental_get_compiler_ir(x)()
    self.assertIn("(arg0.1: s32[10]) -> s32[2]", hlo)

    # Capture a variable
    outer_var = tf.Variable(np.array([3.], dtype=np.float32))
    x = np.array([2., 3., 4.], dtype=np.float32)

    def fun_tf(x):
      return x * tf.broadcast_to(outer_var, x.shape) + 1.

    hlo = tf.function(fun_tf, jit_compile=True).experimental_get_compiler_ir(x)()
    self.assertIn("(arg0.1: f32[3], arg1.2: f32[1]) -> f32[3]", hlo)

    # Capture a constant
    outer_ct = np.array([3.], dtype=np.float32)
    x = np.array([2., 3., 4.], dtype=np.float32)

    def fun_tf(x):
      return x * tf.broadcast_to(outer_ct, x.shape) + 1.

    hlo = tf.function(fun_tf, jit_compile=True).experimental_get_compiler_ir(x)()
    self.assertIn("(arg0.1: f32[3]) -> f32[3]", hlo)

    # Call get_compiler_ir in a function context
    x = np.array([2., 3., 4.], dtype=np.float32)


    def fun_tf_outer(x):
      x_const = tf.constant(0, shape=x.shape, dtype=x.dtype)
      _ = tf.function(tf.math.sin, jit_compile=True).experimental_get_compiler_ir(x_const)()

    # TODO(b/193754660)
    # with self.assertRaisesRegex(
    #     TypeError, "An op outside of the function building code is being passed"):
    #   tf.function(fun_tf_outer)(x)
    #
    # with self.assertRaisesRegex(
    #     TypeError, "An op outside of the function building code is being passed"):
    #   tf.function(fun_tf_outer, jit_compile=True)(x)

    # Call get_concrete_function in a graph context
    def fun_tf_outer_2(x):
      _ = tf.function(tf.math.sin, jit_compile=True).get_concrete_function(tf.TensorSpec(x.shape, x.dtype))
      return x

    # Outside of a function context, this works.
    _ = tf.function(fun_tf_outer_2)(x)
    _ = tf.function(fun_tf_outer_2, jit_compile=True)(x)

  def test_repro_193754660(self):
    # Try to reproduce b/193754660. I can't.
    # We have to have tf.function(jax2tf.convert(jax2tf.call_tf(f_tf))).
    # The get_compiler_ir will indeed fail for f_tf. Then we try to use
    # shape inference for f_tf.
    # I thought to use a f_tf that uses an op without shape inference, e.g.,
    # tfxla.gather. If we wash it through a saved_model I expect that shape
    # inference would not work on it. Instead, shape inference works!!!
    x = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    def f_jax(x):
      return x[1]
    f_tf = jax2tf.convert(f_jax)
    f_tf_rt, _ = tf_test_util.SaveAndLoadFunction(f_tf, input_args=[x])
    f_jax2 = jax2tf.call_tf(f_tf_rt)
    f_tf2 = jax2tf.convert(f_jax2)
    res = tf.function(f_tf2, autograph=False)(x)
    self.assertAllClose(res.numpy(), f_jax(x))

  def test_module_documentation(self):
    def cos_tf(x):
      return tf.math.cos(x)

    # Compute cos with TF and sin with JAX
    def cos_tf_sin_jax(x):
      return jax.numpy.sin(jax2tf.call_tf(cos_tf)(x))

    # Calls `cos_tf` in TF eager mode
    x = np.float32(1.)
    cos_tf_sin_jax(x)

    # Compiles `cos_tf` using TF and embeds the XLA computation into the JAX
    # XLA computation (containing `sin`). The XLA compiler may even be able to
    # fuse through JAX-TF computations.
    jax.jit(cos_tf_sin_jax)(x)

    # Uses TF gradient for `cos_tf` and JAX gradient for `sin`
    jax.grad(cos_tf_sin_jax)(x)

    print(jax.make_jaxpr(cos_tf_sin_jax)(x))
    print(jax.xla_computation(cos_tf_sin_jax)(x).as_hlo_text())

class RoundTripToJaxTest(tf_test_util.JaxToTfTestCase):
  "Reloading output of jax2tf into JAX with call_tf"
  def setUp(self):
    if tf is None:
      raise unittest.SkipTest("Test requires tensorflow")
    # TODO(b/171320191): this line works around a missing context initialization
    # bug in TensorFlow.
    _ = tf.add(1, 1)
    super().setUp()

  def test_simple(self):
    f_jax = jnp.sin
    f_jax_rt = jax2tf.call_tf(jax2tf.convert(f_jax))
    x = np.float32(0.7)
    self.assertAllClose(f_jax(x), f_jax_rt(x))

  def test_pytree(self):
    def f_jax(x):  # x: dict(a=f32, b=f32)
      return dict(a=x["a"]+1., b=x)
    x = dict(a=0.7, b=0.8)
    f_jax_rt = jax2tf.call_tf(jax2tf.convert(f_jax))
    self.assertAllClose(f_jax(x), f_jax_rt(x))

  def test_custom_grad(self):
    @jax.custom_vjp
    def f(x):
      return x * x

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), np.float32(3.) * x
    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * ct_b,

    f.defvjp(f_fwd, f_bwd)

    f_rt = jax2tf.call_tf(jax2tf.convert(f, with_gradient=True))
    x = np.float32(0.7)
    self.assertAllClose(f(x), f_rt(x))
    self.assertAllClose(jax.grad(f)(x), jax.grad(f_rt)(x))

  def test_shape_poly(self):
    f_jax = jnp.sin
    f_jax_rt = jax2tf.call_tf(jax2tf.convert(f_jax,
                                             polymorphic_shapes=["(b, ...)"]))
    x = np.array([0.7, 0.8], dtype=np.float32)
    self.assertAllClose(f_jax(x), f_jax_rt(x))

  def test_saved_model_simple(self):
    x = np.array([0.7, 0.8], dtype=np.float32)
    def f_jax(x):
      return jnp.sin(x)

    f_tf = jax2tf.convert(f_jax)
    restored_tf, _ = tf_test_util.SaveAndLoadFunction(f_tf, input_args=[x])
    restored_jax = jax2tf.call_tf(restored_tf)
    self.assertAllClose(f_jax(x), restored_jax(x))

  def test_saved_model_variables(self):
    param = np.array([1., 2.], dtype=np.float32)
    x = np.array([0.7, 0.8], dtype=np.float32)
    def f_jax(param, x):
      return jnp.sin(x) + jnp.cos(param)

    param_v = tf.Variable(param)
    f_tf = jax2tf.convert(f_jax)
    _, restored_model = tf_test_util.SaveAndLoadFunction(
        lambda x: f_tf(param_v, x),
        input_args=[x],
        variables=[param_v])
    restored_jax = jax2tf.call_tf(restored_model.f)
    self.assertAllClose(f_jax(param, x), restored_jax(x))
    self.assertAllClose(f_jax(param, x), jax.jit(restored_jax)(x))

  def test_saved_model_shape_poly(self):
    tracing_count = 0
    x = np.array([0.7, 0.8], dtype=np.float32)
    def f_jax(x):
      nonlocal tracing_count
      tracing_count += 1
      return jnp.sin(x)

    f_tf = jax2tf.convert(f_jax, polymorphic_shapes=["(b, ...)"])
    res_jax = f_jax(x)
    self.assertEqual(1, tracing_count)
    # Will trace twice, it seems. Once to get the result signature, and once again
    # for the actual saving.
    restored_f, _ = tf_test_util.SaveAndLoadFunction(
        f_tf, input_signature=[tf.TensorSpec([None], x.dtype)])
    self.assertGreaterEqual(tracing_count, 2)
    tracing_count = 0
    f_jax_rt = jax2tf.call_tf(restored_f)
    self.assertAllClose(res_jax, f_jax_rt(x))
    # Ensure that restored_f works at other batch size as well
    y = np.concatenate([x, x])
    self.assertEqual(0, tracing_count)
    res_jax_y = f_jax(y)
    self.assertEqual(1, tracing_count)
    # No more tracing for f_jax_rt
    self.assertAllClose(res_jax_y, f_jax_rt(y))
    self.assertEqual(1, tracing_count)

  def test_custom_grad_saved_model(self):
    @jax.custom_vjp
    def f(x):
      return x * x

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), np.float32(3.) * x
    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * ct_b,

    f.defvjp(f_fwd, f_bwd)
    def g(x):
      return jnp.sum(f(x))

    g_tf, _ = tf_test_util.SaveAndLoadFunction(
        jax2tf.convert(g, with_gradient=True, polymorphic_shapes=["b, ..."]),
        input_signature=[tf.TensorSpec([None], dtype=tf.float32)])
    g_rt = jax2tf.call_tf(g_tf)
    x = np.array([0.7], dtype=np.float32)
    self.assertAllClose(g(x), g_rt(x))
    self.assertAllClose(jax.grad(g)(x), jax.grad(g_rt)(x))

  def test_without_gradient_saved_model(self):
    # Explicitly with_gradient=False
    f_jax = jnp.sum

    x = np.array([0.7, 0.8], dtype=np.float32)
    f_tf, _ = tf_test_util.SaveAndLoadFunction(
        jax2tf.convert(f_jax, with_gradient=False),
        input_args=[x])
    f_rt = jax2tf.call_tf(f_tf)

    self.assertAllClose(f_jax(x), f_rt(x))
    with self.assertRaisesRegex(Exception,
                                "Gradient explicitly disabled.*jax2tf-converted function does not support gradients. Use `with_gradient` parameter to enable gradients"):
      jax.grad(f_rt)(x)

  def test_saved_model_no_gradients(self):
    # Save without gradients
    f_jax = jnp.sum

    x = np.array([0.7, 0.8], dtype=np.float32)
    f_tf, _ = tf_test_util.SaveAndLoadFunction(
        jax2tf.convert(f_jax, with_gradient=True), input_args=[x],
        save_gradients=False)
    f_rt = jax2tf.call_tf(f_tf)

    self.assertAllClose(f_jax(x), f_rt(x))
    # TODO: clean this up b/191117111: it should fail with a clear error
    # The following results in a confusing error:
    # TypeError: tf.Graph captured an external symbolic tensor.
    with self.assertRaises(TypeError):
      _ = jax.grad(f_rt)(x)


class RoundTripToTfTest(tf_test_util.JaxToTfTestCase):
  "Reloading output of call_tf into TF with jax2tf."

  def setUp(self):
    if tf is None:
      raise unittest.SkipTest("Test requires tensorflow")
    # TODO(b/171320191): this line works around a missing context initialization
    # bug in TensorFlow.
    _ = tf.add(1, 1)
    super().setUp()

  def test_alternate(self):
    # Alternate sin/cos with sin in TF and cos in JAX
    f_tf_inner = tf.math.sin
    def f_jax(x_jax):
      y_jax = jnp.cos(x_jax)
      z_jax = jax2tf.call_tf(f_tf_inner)(y_jax)
      return jnp.cos(z_jax)
    def f_tf_outer(x_tf):
      y_tf = tf.math.sin(x_tf)
      z_tf = jax2tf.convert(f_jax)(y_tf)
      return tf.math.sin(z_tf)

    x = np.float32(0.7)

    self.assertAllClose(np.sin(np.cos(np.sin(np.cos(np.sin(x))))),
                        f_tf_outer(x).numpy())
    xv = tf.Variable(x)
    with tf.GradientTape() as tape:
      res = f_tf_outer(xv)
    g_tf = tape.gradient(res, xv)
    _, gf = tf_test_util.ComputeTfValueAndGrad(f_tf_outer, (x,))
    # Eager
    expected_res = np.sin(np.cos(np.sin(np.cos(np.sin(x)))))
    self.assertAllClose(expected_res, f_tf_outer(x).numpy())

    # Gradient
    expected_grad = (np.cos(np.cos(np.sin(np.cos(np.sin(x))))) *
                     np.sin(np.sin(np.cos(np.sin(x)))) *
                     np.cos(np.cos(np.sin(x))) *
                     np.sin(np.sin(x)) *
                     np.cos(x))
    self.assertAllClose(expected_grad, g_tf.numpy())

    # Graph
    self.assertAllClose(expected_res,
                        tf.function(f_tf_outer, autograph=False)(x).numpy())

    # Compiled
    self.assertAllClose(expected_res,
                        tf.function(f_tf_outer, autograph=False,
                                    jit_compile=True)(x).numpy())

  def test_saved_model(self):
    x = np.array([.7, .8], dtype=np.float32)
    def fun_tf(x):
      return tf.math.sin(x)
    def fun_jax(x):
      return jax2tf.call_tf(fun_tf)(x)

    # Now convert and save to SavedModel
    fun_tf_rt = jax2tf.convert(fun_jax)
    res = fun_tf_rt(x)
    self.assertAllClose(np.sin(x), res.numpy())

    res = tf.function(fun_tf_rt, autograph=False)(x)
    self.assertAllClose(np.sin(x), res.numpy())

    res = tf.function(fun_tf_rt, jit_compile=True, autograph=False)(x)
    self.assertAllClose(np.sin(x), res.numpy())

    reloaded_f, _ = tf_test_util.SaveAndLoadFunction(
        fun_tf_rt, input_args=[x])
    res = reloaded_f(x)
    self.assertAllClose(np.sin(x), res.numpy())

  def test_function_dynamic_shape(self):
    # Call a function for which shape inference does not give an output
    # shape.
    x = np.array([-1, 0, 1], dtype=np.int32)
    def fun_tf(x):  # x:i32[3]
      # The shape depends on the value of x
      return tf.cond(x[0] >= 0, lambda: x, lambda: x[1:])

    # Call in eager mode. Should work!
    res1 = jax2tf.call_tf(fun_tf)(x)
    expected = x[1:]
    self.assertAllClose(expected, res1, check_dtypes=False)

    # Now under jit, should fail because the function is not compileable
    with self.assertRaisesRegex(ValueError,
                                _call_tf_dynamic_shape_error):
      fun_jax = jax.jit(jax2tf.call_tf(fun_tf))
      fun_jax(x)

    # TODO(necula): this should work in op-by-op mode, but it fails because
    # jax2tf.convert does abstract evaluation.
    with self.assertRaisesRegex(ValueError,
                                _call_tf_dynamic_shape_error):
      fun_tf_rt = jax2tf.convert(jax2tf.call_tf(fun_tf))
      fun_tf_rt(x)

  def test_shape_polymorphism_error(self):
    x = np.array([.7, .8], dtype=np.float32)
    def fun_tf(x):
      return tf.math.sin(x)

    fun_jax = jax2tf.call_tf(fun_tf)

    fun_tf_rt = jax2tf.convert(fun_jax,
                               polymorphic_shapes=["b, ..."])
    with self.assertRaisesRegex(
        ValueError,
        "call_tf cannot be applied to shape-polymorphic arguments"):
      fun_tf_rt(x)


  @parameterized.named_parameters(
      _named_test(f2_function=f2_function, f2_saved_model=f2_saved_model,
                  f4_function=f4_function, f4_saved_model=f4_saved_model)
      for f2_function in [True, False]
      for f2_saved_model in [True, False]
      for f4_function in [True, False]
      for f4_saved_model in [True, False])
  def test_several_round_trips(self,
                               f2_function=False, f2_saved_model=False,
                               f4_function=False, f4_saved_model=False):
    x = np.array(.7, dtype=np.float32)
    # f(n)(x) = 2. * x^n
    def f(n):
      def fn(x):
        acc = np.array(2., dtype=x.dtype)
        for i in range(n):
          acc *= x
        return acc
      return fn

    f2_tf = lambda x: x * jax2tf.convert(f(1))(x)
    if f2_function:
      f2_tf = tf.function(f2_tf, autograph=False)
    if f2_saved_model:
      f2_tf, _ = tf_test_util.SaveAndLoadFunction(f2_tf, input_args=[x])

    self.assertAllClose(f(2)(x), f2_tf(x).numpy())
    _, (g_f2_ft,) = tf_test_util.ComputeTfValueAndGrad(f2_tf, [x])
    self.assertAllClose(jax.grad(f(2))(x), g_f2_ft.numpy())

    f3_jax = lambda x: x * jax2tf.call_tf(f2_tf)(x)
    self.assertAllClose(f(3)(x), f3_jax(x))
    self.assertAllClose(f(3)(x), jax.jit(f3_jax)(x))
    self.assertAllClose(jax.grad(f(3))(x), jax.grad(f3_jax)(x))

    f4_tf = lambda x: x * jax2tf.convert(f3_jax)(x)
    self.assertAllClose(f(4)(x), f4_tf(x).numpy())
    _, (g_f4_ft,) = tf_test_util.ComputeTfValueAndGrad(f4_tf, [x])
    self.assertAllClose(jax.grad(f(4))(x), g_f4_ft.numpy())

    if f4_function:
      f4_tf = tf.function(f4_tf, autograph=False)
    if f4_saved_model:
      f4_tf, _ = tf_test_util.SaveAndLoadFunction(f4_tf, input_args=[x])
    self.assertAllClose(f(4)(x), f4_tf(x).numpy())
    _, (g_f4_ft,) = tf_test_util.ComputeTfValueAndGrad(f4_tf, [x])
    self.assertAllClose(jax.grad(f(4))(x), g_f4_ft.numpy())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
