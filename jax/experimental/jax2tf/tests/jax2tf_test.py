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
    _, res_tf = self.ConvertAndCompare(f_jax, 0.7)
    self.assertIsInstance(res_tf, tf.Tensor)

  def test_pytrees(self):
    # Take and return pytrees
    def f_jax(x: Tuple[float, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
      x_a, x_dict = x
      return x_a * 2., {k : v * 3. for k, v in x_dict.items()}

    x = (.7, {"a": .8, "b": .9})
    self.ConvertAndCompare(f_jax, x)

  def test_variable_input(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    f_tf = jax2tf.convert(f_jax)
    v = tf.Variable(0.7)
    self.assertIsInstance(f_tf(v), tf.Tensor)
    self.assertAllClose(f_jax(0.7), f_tf(v))

  def test_jit(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    self.ConvertAndCompare(f_jax, 0.7)

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
    dict(testcase_name=f"_dtype={dtype.__name__}_function={with_function}",
         with_function=with_function,
         dtype=dtype)
    for with_function in [False, True]
    for dtype in [np.int64, np.float64]))
  def test_converts_64bit(self, dtype=np.int64, with_function=False):
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
    self.ConvertAndCompare(f_jax, 0.7, with_function=True)

  @parameterized.named_parameters(jtu.cases_from_list(
    dict(testcase_name=f"function={with_function}",
         with_function=with_function)
    for with_function in [False, True]))
  def test_gradients_disabled(self, with_function=False):
    f_tf = jax2tf.convert(jnp.tan)
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
    x = tf.Variable(4.)
    y = tf.Variable(5.)
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
    x = tf.Variable(4.)
    y = tf.Variable(5.)
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
    self.assertAllClose(4. * 4., f_tf(4.))
    x = tf.Variable(4.)
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
    self.assertAllClose(4. * 4., f_tf(4.))
    x = tf.Variable(4.)
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = f_tf(x)

    self.assertAllClose(4. * 4., y)
    self.assertAllClose(3. * 4., tape.gradient(y, x))

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

    arg = 0.7
    self.TransformConvertAndCompare(f, arg, None, with_function=True)
    self.TransformConvertAndCompare(f, arg, "jvp", with_function=True)
    self.TransformConvertAndCompare(f, arg, "vmap", with_function=True)
    self.TransformConvertAndCompare(f, arg, "jvp_vmap", with_function=True)
    self.TransformConvertAndCompare(f, arg, "grad", with_function=True)
    self.TransformConvertAndCompare(f, arg, "grad_vmap", with_function=True)

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
    arg = 0.7
    self.TransformConvertAndCompare(f, arg, None, with_function=True)
    self.TransformConvertAndCompare(f, arg, "vmap", with_function=True)
    self.TransformConvertAndCompare(f, arg, "grad", with_function=True)
    self.TransformConvertAndCompare(f, arg, "grad_vmap", with_function=True)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
