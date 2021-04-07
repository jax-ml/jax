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
"""Tests for JAX2TF converted.

Specific JAX primitive conversion tests are in primitives_test."""

from typing import Dict, Tuple

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import dtypes
from jax import numpy as jnp
from jax import test_util as jtu
from jax.config import config
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util
import numpy as np
import tensorflow as tf  # type: ignore[import]

config.parse_flags_with_absl()


class Jax2TfTest(tf_test_util.JaxToTfTestCase):

  def test_basics(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    _, res_tf = self.ConvertAndCompare(f_jax, jnp.float_(0.7))

  def test_input_output_naming(self):
    @jax2tf.convert
    def f(xs, y):
      return [jnp.add(x, y) for x in xs]

    @tf.function(autograph=False)
    def u(xs, y):
      xs = tf.nest.map_structure(tf.convert_to_tensor, xs)
      with tf.GradientTape() as tape:
        tf.nest.map_structure(tape.watch, xs)
        y = f(xs, y)
        tape.gradient(y, xs)
        return y

    cf = u.get_concrete_function([1., 2., 3.], 4.)
    g = cf.graph
    g.get_operation_by_name("jax2tf_arg_0")
    g.get_operation_by_name("jax2tf_arg_0_1")
    g.get_operation_by_name("jax2tf_arg_0_2")
    g.get_operation_by_name("jax2tf_arg_1")
    g.get_operation_by_name("jax2tf_out")
    g.get_operation_by_name("jax2tf_out_1")
    g.get_operation_by_name("jax2tf_out_2")
    with self.assertRaises(KeyError):
      g.get_operation_by_name("jax2tf_arg_2")
    with self.assertRaises(KeyError):
      g.get_operation_by_name("jax2tf_out_3")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_0")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_1")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_1_1")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_1_2")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out_1")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out_2")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out_3")

  def test_pytrees(self):
    # Take and return pytrees
    def f_jax(x: Tuple[float, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
      x_a, x_dict = x
      return x_a * 2., {k : v * 3. for k, v in x_dict.items()}

    x = (jnp.float_(.7), {"a": jnp.float_(.8), "b": jnp.float_(.9)})
    self.ConvertAndCompare(f_jax, x)

  def test_variable_input(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    f_tf = jax2tf.convert(f_jax)
    v = tf.Variable(0.7, dtype=dtypes.canonicalize_dtype(jnp.float_))
    self.assertIsInstance(f_tf(v), tf.Tensor)
    self.assertAllClose(f_jax(0.7), f_tf(v))

  def test_jit(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    self.ConvertAndCompare(f_jax, jnp.float_(0.7))

  def test_nested_jit(self):
    f_jax = jax.jit(lambda x: jnp.sin(jax.jit(jnp.cos)(x)))
    f_tf = jax2tf.convert(f_jax)
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

  @jtu.skip_on_devices("gpu")
  def test_bfloat16_passed_by_tf(self):
    f_jax = lambda a, b: a + b
    f_tf = tf.function(jax2tf.convert(f_jax),
                       input_signature=[tf.TensorSpec([512, 512], tf.bfloat16),
                                        tf.TensorSpec([512, 512], tf.bfloat16)])
    self.assertIsNotNone(f_tf.get_concrete_function())

  @jtu.skip_on_devices("gpu")
  def test_bfloat16_returned_by_jax(self):
    f_jax = lambda a, b: (a + b).astype(jnp.bfloat16)
    f_tf = jax2tf.convert(f_jax)
    self.assertEqual(f_tf(1., 2.).dtype, tf.bfloat16)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_dtype={dtype.__name__}",
         dtype=dtype)
    for dtype in [np.int64, np.float64]))
  def test_converts_64bit(self, dtype=np.int64, with_function=False):
    if not config.jax_enable_x64:
      self.skipTest("requires x64 mode")
    big_const = np.full((5,), 2 ** 33, dtype=dtype)
    self.ConvertAndCompare(jnp.sin, big_const)
    f_conv = jax2tf.convert(jnp.sin)
    if with_function:
      f_conv = tf.function(f_conv)
    # We check also when we pass tf.Variable or tf.Tensor into the
    # converted function
    self.assertAllClose(jnp.sin(big_const),
                        f_conv(tf.Variable(big_const)))
    self.assertAllClose(jnp.sin(big_const),
                        f_conv(tf.constant(big_const)))

  def test_function(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    self.ConvertAndCompare(f_jax, jnp.float_(0.7))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_gradients_disabled(self, with_function=False):
    f_tf = jax2tf.convert(jnp.tan, with_gradient=False)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    x = tf.ones([])

    # With tf.function the error is raised when we evaluate f_tf(x), in
    # eager mode when we evaluate tape.gradient(y, x)
    with self.assertRaisesRegex(LookupError,
                                "Gradient explicitly disabled.*The jax2tf-converted function does not support gradients"):
      with tf.GradientTape() as tape:
        tape.watch(x)
        y = f_tf(x)
        _ = tape.gradient(y, x)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_gradients(self, with_function=True):
    def f(x, y):
      return x * x, x * y
    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    default_float_type = dtypes.canonicalize_dtype(jnp.float_)
    x = tf.Variable(4., dtype=default_float_type)
    y = tf.Variable(5., dtype=default_float_type)
    with tf.GradientTape(persistent=True) as tape:
      u, v = f_tf(x, y)

    self.assertAllClose(2. * 4., tape.gradient(u, x))
    self.assertAllClose(0., tape.gradient(u, y))
    self.assertAllClose(5., tape.gradient(v, x))
    self.assertAllClose(4., tape.gradient(v, y))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_gradients_pytree(self, with_function=True):
    def f(xy: Tuple[float, float]) -> Dict[str, float]:
      x, y = xy
      return dict(one=x * x, two=x * y)

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    default_float_dtype = dtypes.canonicalize_dtype(jnp.float_)
    x = tf.Variable(4., dtype=default_float_dtype)
    y = tf.Variable(5., dtype=default_float_dtype)
    with tf.GradientTape(persistent=True) as tape:
      uv = f_tf((x, y))

    self.assertAllClose(2. * 4., tape.gradient(uv["one"], x))
    self.assertAllClose(0., tape.gradient(uv["one"], y))
    self.assertAllClose(5., tape.gradient(uv["two"], x))
    self.assertAllClose(4., tape.gradient(uv["two"], y))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_gradients_with_custom_jvp(self, with_function=True):
    """Check gradients, for a function with custom JVP."""
    @jax.custom_jvp
    def f(x):
      return x * x

    @f.defjvp
    def f_jvp(primals, tangents):
      # 3 * x * x_t
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * x_dot
      return primal_out, tangent_out

    self.assertAllClose(4. * 4., f(4.))
    self.assertAllClose(3. * 4., jax.grad(f)(4.))

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    self.assertAllClose(4. * 4., f_tf(jnp.float_(4.)))
    x = tf.Variable(4., dtype=dtypes.canonicalize_dtype(jnp.float_))
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = f_tf(x)

    self.assertAllClose(4. * 4., y)
    self.assertAllClose(3. * 4., tape.gradient(y, x))

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_gradients_with_custom_vjp(self, with_function=True):
    """Check gradients, for a function with custom VJP."""
    @jax.custom_vjp
    def f(x):
      return x * x

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), 3. * x
    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * ct_b,

    f.defvjp(f_fwd, f_bwd)

    self.assertAllClose(4. * 4., f(4.))
    self.assertAllClose(3. * 4., jax.grad(f)(4.))

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    self.assertAllClose(4. * 4., f_tf(jnp.float_(4.)))
    x = tf.Variable(4., dtype=dtypes.canonicalize_dtype(jnp.float_))
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = f_tf(x)

    self.assertAllClose(4. * 4., y)
    self.assertAllClose(3. * 4., tape.gradient(y, x))

  def test_gradient_with_float0_intermediate(self):
    # Gradient over integer-argument functions
    def f(x, y):  # x is an int, y is a float
      return 2 * x + y

    def g(x):  # x: f32
      return 2. * f(3 * x.astype("int32"), x * 4.)

    x = np.float_(2.)
    grad_g = jax.grad(g)
    self.ConvertAndCompare(grad_g, x)


  def test_gradient_with_float0_result(self):
    # Gradient over integer-argument functions, with float0 result
    def f(x, y):  # x is an int, y is a float
      return 2 * x + y

    def g(x):  # x: i32
      return jnp.sum(2. * f(3 * x, 4. * x.astype("float32")))

    grad_g = jax.grad(g, allow_int=True)
    x = 2
    d_dx_jax = grad_g(x)
    d_dx_tf = jax2tf.convert(grad_g)(x)
    self.assertEqual(d_dx_jax.dtype, dtypes.float0)
    self.assertAllClose(jnp.zeros(np.shape(d_dx_jax), dtypes.bfloat16),
                        d_dx_tf.numpy())

    shape = (3, 4)
    x = np.ones(shape, dtype=np.int32)
    d_dx_jax = grad_g(x)
    d_dx_tf = jax2tf.convert(grad_g)(x)
    self.assertEqual(d_dx_jax.dtype, dtypes.float0)
    self.assertAllClose(jnp.zeros(np.shape(d_dx_jax), dtypes.bfloat16),
                        d_dx_tf.numpy())

  def test_convert_argument_non_callable_error(self):
    with self.assertRaisesRegex(TypeError, "Expected a callable value"):
      jax2tf.convert(5.)

  def test_convert_argument_non_tensor_error(self):
    with self.assertRaisesRegex(TypeError,
                                "Argument.*should be NumPy array"):
      jax2tf.convert(lambda x: x)(lambda y: y)

  def test_argument_eager_tensor(self):
    x = jax2tf.convert(jnp.sin)(1.)
    jax2tf.convert(jnp.cos)(x)  # No error

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

  def test_custom_jvp(self):
    """Conversion of function with custom JVP"""
    @jax.custom_jvp
    def f(x):
      return x * x

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * x_dot
      return primal_out, tangent_out

    arg = jnp.float_(0.7)
    self.TransformConvertAndCompare(f, arg, None)
    self.TransformConvertAndCompare(f, arg, "jvp")
    self.TransformConvertAndCompare(f, arg, "vmap")
    self.TransformConvertAndCompare(f, arg, "jvp_vmap")
    self.TransformConvertAndCompare(f, arg, "grad")
    self.TransformConvertAndCompare(f, arg, "grad_vmap")

  def test_custom_vjp(self):
    """Conversion of function with custom VJP"""
    @jax.custom_vjp
    def f(x):
      return x * x

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), 3. * x
    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * ct_b,

    f.defvjp(f_fwd, f_bwd)
    arg = jnp.float_(0.7)
    self.TransformConvertAndCompare(f, arg, None)
    self.TransformConvertAndCompare(f, arg, "vmap")
    self.TransformConvertAndCompare(f, arg, "grad")
    self.TransformConvertAndCompare(f, arg, "grad_vmap")

  def test_remat1(self):
    @jax.remat
    def f(x1):
      x2 = jnp.sin(x1)
      x3 = jnp.sin(x2)
      x4 = jnp.sin(x3)
      return jnp.sum(x4)

    # The computation of grad_f computes "sin" 5 times, 3 for the forward pass
    # and then to rematerialize "x2" and "x3" in the backward pass.
    arg = np.arange(3.)
    self.TransformConvertAndCompare(f, arg, "grad")
    # TODO: check that the TF code also computes "sin" 5 times


  def test_remat_free_var(self):
    def f(x):
      y = 2 * x

      @jax.remat
      def g():
        return y

      return g()
    arg = jnp.float_(3.)
    self.TransformConvertAndCompare(f, arg, None)
    self.TransformConvertAndCompare(f, arg, "grad")

  def test_convert_nullary_func(self):
    # Even nullary functions are converted to TF (as opposed to constant-folded
    # in JAX prior to conversion).
    def f_jax():
      return jnp.sin(1.)
    f_tf = tf.function(jax2tf.convert(f_jax), autograph=False)
    f_tf_graph = f_tf.get_concrete_function().graph.as_graph_def()
    self.assertIn('op: "Sin"', str(f_tf_graph))

  def test_convert_of_nested_independent_jit(self):
    def func(x):
      def inner1(y):
        return x + y
      # The JIT does not have data dependency
      return jax.jit(inner1)(1.)

    jax2tf.convert(func)(2.)

  def test_convert_of_nested_dependent_jit(self):
    def func(x):
      def inner1(y):
        return x + y
      # The JIT does have data dependency
      return jax.jit(inner1)(x)

    jax2tf.convert(func)(2.)  # No error

  def test_nested_convert_error(self):
    def outer(y):
      return jax2tf.convert(jnp.sin)(y)  # Inner convert takes tracer args
    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      jax2tf.convert(outer)(np.ones((4, )))

  def test_nested_convert_error_non_tracer(self):
    """The inner convert takes non-tracer arguments"""
    def outer(y):
      sin_1 = jax2tf.convert(jnp.sin)(1.)  # Inner convert takes non-tracer arg
      return y + sin_1

    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      jax2tf.convert(outer)(2.)


  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{transform}", transform=transform)
    for transform in ["jit", "jvp", "grad", "vmap"]))
  def test_convert_under_transform_error(self, transform="vmap"):
    def outer(y):
      return jax2tf.convert(jnp.sin)(y)  # Inner convert takes tracer args

    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      self.TransformConvertAndCompare(outer, np.ones((4,)), transform)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"_{transform}", transform=transform)
    for transform in ["jit", "jvp", "grad", "vmap"]))
  def test_convert_under_transform_error_non_tracer(self, transform="vmap"):
    def outer(y):
      sin_1 = jax2tf.convert(jnp.sin)(1.)  # Inner convert takes non-tracer arg
      return y + sin_1

    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      self.TransformConvertAndCompare(outer, np.ones((4,)), transform)

  def test_name_scope(self):
    log = []

    @jax.named_call
    def my_test_function(x):
      y = tf.Variable(1., name="foo")
      log.append(y.name)
      return x * x

    jax2tf.convert(my_test_function)(2)
    self.assertIn("my_test_function/foo", log[0])

  def test_bfloat16_constant(self):
    # Re: https://github.com/google/jax/issues/3942
    def jax_fn_scalar(x):
      x = x.astype(jnp.bfloat16)
      x *= 2.
      return x

    def jax_fn_array(x):
      x = x.astype(jnp.bfloat16)
      x *= np.array([1.5, 2.5, 3.5], jnp.bfloat16)
      return x

    tf_fn_scalar = jax2tf.convert(jax_fn_scalar)
    self.assertAllClose(tf_fn_scalar(1.375).numpy(), jnp.bfloat16(2.750))

    tf_fn_array = jax2tf.convert(jax_fn_array)
    self.assertAllClose(
        tf_fn_array(np.array([3, 4, 5])), np.array([4.5, 10, 17.5],
                                                   jnp.bfloat16))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
