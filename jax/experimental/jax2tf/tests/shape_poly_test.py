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
"""Tests for the jax2tf conversion for control-flow primitives."""

from absl.testing import absltest
from typing import Dict, Sequence

import functools
import operator
import re

import jax
from jax import core
from jax.experimental import jax2tf
from jax import lax
import jax.numpy as jnp
from jax import test_util as jtu
from jax._src import util
import numpy as np
from jax.interpreters import masking


from jax.experimental.jax2tf.tests import tf_test_util

import tensorflow as tf  # type: ignore[import]
import unittest

from jax.config import config
config.parse_flags_with_absl()


class ShapePolyTest(tf_test_util.JaxToTfTestCase):

  def test_simple(self):
    """Test shape polymorphism for a simple case."""
    def f_jax(x):
      return x + jnp.sin(x)

    self.CheckShapePolymorphism(f_jax,
                                input_signature=[tf.TensorSpec([2, 3])],
                                in_shapes=None,
                                expected_output_signature=tf.TensorSpec([2, 3]))

    self.CheckShapePolymorphism(f_jax,
                                input_signature=[tf.TensorSpec([2, None])],
                                in_shapes=["(_, h)"],
                                expected_output_signature=tf.TensorSpec([2, None]))

    self.CheckShapePolymorphism(f_jax,
                                input_signature=[tf.TensorSpec([None, None])],
                                in_shapes=["(h, h)"],
                                expected_output_signature=tf.TensorSpec([None, None]))

  def test_arg_avals(self):
    """Test conversion of actual arguments to abstract values"""
    input_avals = jax2tf.jax2tf._input_avals
    def shaped_array(shape):
      if isinstance(shape, str):
        return core.ShapedArray(masking.parse_spec(shape), np.float32)
      else:
        return core.ShapedArray(shape, np.float32)

    def const(shape):
      return np.ones(shape, dtype=np.float32)
    def tf_const(shape):
      return tf.convert_to_tensor(np.ones(shape, dtype=np.float32))
    def tf_var(init_shape, shape):
      return tf.Variable(np.ones(init_shape, np.float32),
                         dtype=tf.float32, shape=shape)

    # Known shapes for the arguments
    self.assertEqual((shaped_array([2, 3]),),
                     input_avals([const((2, 3))], [None]))
    self.assertEqual((shaped_array([2, 3]),),
                     input_avals([tf_const((2, 3))], [None]))
    self.assertEqual((shaped_array([2, 3]),),
                     input_avals([tf_var((2, 3), (2, 3))], [None]))
    self.assertEqual((shaped_array([2, 3]),),
                     input_avals([const((2, 3))], ["(2, 3)"]))
    self.assertEqual((shaped_array([2, 3]),),
                     input_avals([tf_const((2, 3))], ["(_, 3)"]))
    self.assertEqual((shaped_array([2, 3]),),
                     input_avals([tf_const((2, 3))], ["(_, 3)"]))

    # Partially known shapes for the arguments
    self.assertEqual((shaped_array("(b, 3)"),),
                      input_avals([tf_var((2, 3), [None, 3])], ["(b, 3)"]))

    self.assertEqual((shaped_array("(h, h)"),),
                      input_avals([tf_var((2, 3), [None, None])], [("h, h")]))

    # Some errors
    with self.assertRaisesRegex(TypeError,
                                re.escape("in_shape must be specified when the argument shape (2, None) is partially known")):
      input_avals([tf_var((2, 3), [2, None])], [None])

    with self.assertRaisesRegex(
        TypeError,
        re.escape("in_shape (_) has different rank than actual argument shape (2, 3)")):
      input_avals([const((2, 3))], ["(_)"])

    with self.assertRaisesRegex(
        TypeError,
        re.escape("in_shape (_, _) has `_` placeholders for argument shape dimensions that are unknown: (2, None)")):
      input_avals([tf_var((2, 3), [2, None])], ["(_, _)"])

    with self.assertRaisesRegex(
        TypeError,
        re.escape("in_shape (2, 13) (resolved to (2, 13)) does not match argument shape (2, 3) in dimension 1")):
      input_avals([const((2, 3))], ["(2, 13)"])

    with self.assertRaisesRegex(
        TypeError,
        re.escape("in_shape (2, 3) (resolved to (2, 3)) does not match argument shape (2, None) in dimension 1")):
      input_avals([tf_var((2, 3), [2, None])], ["(2, 3)"])

  def test_solve_shape_vars(self):
    def solve_shape_vars(shape_spec: str, shape: Sequence[int]) -> Dict[str, int]:
      shape_polys = masking.parse_spec(shape_spec)
      return jax2tf.jax2tf._solve_shape_vars(util.safe_zip(shape_polys, shape))

    self.assertAllClose(solve_shape_vars("(a, b, c)", [1, 2, 3]),
                         dict(a=1, b=2, c=3), check_dtypes=False)
    self.assertAllClose(solve_shape_vars("(a + b, b + c, c)", [3, 5, 3]),
                        dict(a=1, b=2, c=3), check_dtypes=False)
    self.assertAllClose(solve_shape_vars("(a + b, 5, b)", [3, 5, 2]),
                        dict(a=1, b=2), check_dtypes=False)
    self.assertAllClose(solve_shape_vars("(2 * a + 1, 3 * a + b + 4, b)", [3, 9, 2]),
                        dict(a=1, b=2), check_dtypes=False)

    self.assertAllClose(jax2tf.jax2tf._solve_shape_vars([(2, 2)]),
                        dict())

    with self.assertRaisesRegex(
        TypeError,
        "only linear polynomials are supported as input shape specifications. Found 'a b'"):
      self.assertAllClose(solve_shape_vars("(a + a * b, 5, b)", [2, 5, 2]),
                          dict(a=1, b=2))

  def test_bad_in_shapes(self):
    def add2(x, y):
      return x + y

    with self.assertRaisesRegex(masking.ShapeSyntaxError, ""):
      self.CheckShapePolymorphism(add2,
                                  input_signature=[tf.TensorSpec([None]), tf.TensorSpec([None])],
                                  in_shapes=[") + (", None],
                                  expected_output_signature=tf.TensorSpec([None]))

    with self.assertRaisesRegex(TypeError,
                                re.escape("in_shapes must be a sequence as long as the argument list (2). "
                                          "Got in_shapes=['(b, 4)']")):
      self.CheckShapePolymorphism(add2,
                                  input_signature=[tf.TensorSpec([None]), tf.TensorSpec([None])],
                                  in_shapes=["(b, 4)"],
                                  expected_output_signature=tf.TensorSpec([None]))

  def test_pytree(self):
    """Arguments and in_shapes are pytrees."""

    # Arguments are of the form [([x00, x01], [x10]), dict(a=ya, b=yb)]
    def add_all_jax(x_pair_of_list, y_dict):
      x_list_0, x_list_1 = x_pair_of_list
      return functools.reduce(operator.add,
                              x_list_0 + x_list_1 + [y_dict["a"], y_dict["b"]])

    self.CheckShapePolymorphism(
      add_all_jax,
      input_signature=[([tf.TensorSpec([None]), tf.TensorSpec([None])],
                        [tf.TensorSpec([None])]),
                       dict(a=tf.TensorSpec([None]), b=tf.TensorSpec([None]))],
      in_shapes=[(["(v,)", "(v,)"], [("v,")]),
                 dict(a="(v,)", b="(v,)")],
      expected_output_signature=tf.TensorSpec([None]))

    # Now partial in_shapes; the parts of the in_shapes that are not specified
    # must have full input_signatures.
    self.CheckShapePolymorphism(
      add_all_jax,
      input_signature=[([tf.TensorSpec([4]), tf.TensorSpec([4])],
                        [tf.TensorSpec([4])]),
                       dict(a=tf.TensorSpec([4]), b=tf.TensorSpec([4]))],
      in_shapes=[(["(4,)", "(_,)"], [("4,")]),
                 dict(a="(_,)", b="(4,)")],
      expected_output_signature=tf.TensorSpec([4]))

  def test_with_custom_vjp(self):
    """Shape-polymorphic custom VJP."""
    raise unittest.SkipTest("Failing after fixing Poly unsoundness #4878")
    @jax.custom_vjp
    def f(x):
      # x: [b1, b2, d1, d2] (a batch of matrices)
      # res: [b1, b2, d1, d1]
      return jnp.matmul(x, jnp.transpose(x, axes=(0, 1, 3, 2)))

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      # x: [b1, b2, d1, d2]
      # b: [b1, b2, d1, d1]
      # residual: [b1, b2, d1, d2]
      return f(x), 3. * x
    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      # residual: [b1, b2, d1, d2]
      # ct_b: [b1, b2, d1, d1]
      # ct_a: [b1, b2, d1, d2]
      return jnp.matmul(ct_b, residual),

    f.defvjp(f_fwd, f_bwd)
    x = np.ones((2, 3, 4, 5), dtype=np.float32)
    res_jax = f(x)
    res_jax_grad = jax.grad(lambda x: jnp.sum(f(x)))(x)

    f_tf = self.CheckShapePolymorphism(
      f,
      input_signature=[tf.TensorSpec([None, None, None, None])],
      in_shapes=["(batch1, batch2, d1, d2)"],
      expected_output_signature=tf.TensorSpec([None, None, None, None]))

    self.assertAllClose(res_jax, f_tf(x))

    xv = tf.Variable(x, dtype=np.float32)
    def tf_value_and_grad(xv):
      with tf.GradientTape() as tape:
        tape.watch(xv)
        res_tf = f_tf(xv)
        res_tf_grad = tape.gradient(res_tf, xv)
        return res_tf, res_tf_grad

    res_tf, res_tf_grad = tf_value_and_grad(xv)
    self.assertAllClose(res_jax, res_tf)
    self.assertAllClose(res_jax_grad, res_tf_grad)

    # Now use TF tracing for the gradient
    tf_grad = tf.function(
      tf_value_and_grad,
      autograph=False).get_concrete_function(tf.TensorSpec([None, None, 8, 9]))

    # The shape of the value
    self.assertEqual((None, None, 8, 8), tuple(tf_grad.output_shapes[0]))
    # The shape of the gradient should match the input
    # TODO: there seems to be a bug here, the output should be (None, None, 8, 9)
    # self.assertEqual((None, None, 8, None), tuple(tf_grad.output_shapes[1]))

  def test_gradients_pytree(self):
    """Shape polymorphism with gradients and pytrees for inputs and outputs."""
    def f(x):
      # x: dict(x=[b, 3, 4])
      # res: dict(res=[b, 3, 4])
      return dict(res=x["x"] * 2.)

    f_tf = self.CheckShapePolymorphism(
      f,
      input_signature=[dict(x=tf.TensorSpec([None, 3, 4]))],
      in_shapes=[dict(x=("b, 3, 4"))],
      expected_output_signature=None)

    x = dict(x=np.ones((2, 3, 4), dtype=np.float32))
    xv = tf.Variable(x["x"], dtype=np.float32)
    def tf_value_and_grad(xv):
      # xv: [b, 3, 4]
      # res_value: dict(res=[b, 3, 4])
      # res_grad: dict(grad=[b, 3, 4])
      with tf.GradientTape() as tape:
        tape.watch(xv)
        res_tf = f_tf(dict(x=xv))
        res_tf_grad = tape.gradient(res_tf, xv)
        return res_tf, dict(grad=res_tf_grad)

    res_tf, res_tf_grad = tf_value_and_grad(xv)
    # Now use TF tracing for the gradient
    tf_grad = tf.function(tf_value_and_grad, autograph=False).get_concrete_function(
      tf.TensorSpec([None, 3, 4]))
    # The shape of the value
    self.assertEqual((None, 3, 4), tuple(tf_grad.output_shapes[0]["res"]))
    # The shape of the gradient should match the input
    self.assertEqual((None, 3, 4), tuple(tf_grad.output_shapes[1]["grad"]))

  def test_cond(self):
    raise unittest.SkipTest("Failing after fixing Poly unsoundness #4878")
    # Test the primitive under conditional
    def f(x, y):
      # x: f32[B, H], y : f32[H]
      return lax.cond(jnp.sum(x) > 0.,
                      lambda _: x + y,
                      lambda _: jnp.zeros_like(x),
                      operand=None)
    x = np.ones((2, 3))
    y = np.ones((3,))
    res_jax = f(x, y)
    self.assertAllClose(res_jax, jax2tf.convert(f, in_shapes=["(b, h)", "h"])(x, y))

  def test_shape_error(self):
    """Some of the examples from the README."""
    raise unittest.SkipTest("Failing after fixing Poly unsoundness #4878")
    with self.assertRaisesRegex(TypeError,
                                re.escape("add got incompatible shapes for broadcasting: (v,), (4,)")):
      self.CheckShapePolymorphism(
        lambda x, y: x + y,
        input_signature=[tf.TensorSpec([None]), tf.TensorSpec([4])],
        in_shapes=["(v,)", "(4,)"],
        expected_output_signature=tf.TensorSpec([None]))

    four_ones = np.ones((4,))
    # We get the error even if we use correct actual arguments
    with self.assertRaisesRegex(TypeError,
                                re.escape("add got incompatible shapes for broadcasting: (v,), (4,)")):
      jax2tf.convert(lambda x, y: x + y,
                     in_shapes=["(v,)", "(4,)"])(four_ones, four_ones)

    with self.assertRaisesRegex(TypeError,
                                re.escape("dot_general requires contracting dimensions to have the same shape, got [4] and [v].")):
      jax2tf.convert(lambda x: jnp.matmul(x, x),
                     in_shapes=["(v, 4)"])(np.ones((4, 4)))

    # TODO: this is an opportunity to improve the translation, should not error
    with self.assertRaisesRegex(TypeError,
                                "Only integers, .* tensors are valid indices, got 0"):
      jax2tf.convert(lambda x: jnp.split(x, 2),
                     in_shapes=["(2*v,)"])(four_ones)


class ShapeAsValueTest(tf_test_util.JaxToTfTestCase):

  def test_concrete_shapes(self):
    # Test shape_as_value with concrete shapes. All transformations work.
    def f(x):
      return jnp.sum(x, axis=0) * jax2tf.shape_as_value(x)[0]

    x = np.arange(3.)
    self.assertAllClose(9., f(x))
    self.assertAllClose(9., jax.jit(f)(x))

    res_primal, res_tangent = jax.jvp(f, (x,), (np.array([0.1, 0.2, 0.3]),))
    self.assertAllClose((9., 1.8), (res_primal, res_tangent))

    self.assertAllClose(np.array([3., 3., 3.]), jax.grad(f)(x))

    xv = np.arange(24.).reshape((2, 3, 4))
    res_vmap = jax.vmap(f, in_axes=1)(xv)
    # Implement by iteration
    res_iter = jnp.stack([f(xv[:, i, :]) for i in range(xv.shape[1])])
    self.assertAllClose(res_iter, res_vmap)

    res_mask2, _ = jax.mask(f, in_shapes=["(b,)"])([x], dict(b=2))
    self.assertAllClose(2., res_mask2)
    res_mask3, _ = jax.mask(f, in_shapes=["(b,)"])([x], dict(b=3))
    self.assertAllClose(9., res_mask3)

  def test_dynamic_shapes(self):
    # Test shape_as_value with dynamic shapes. All transformations work.
    def f(x):
      return jnp.sum(x, axis=0) * jax2tf.shape_as_value(x)[0]

    x = np.arange(3.)
    self.assertAllClose(9., jax2tf.convert(f, in_shapes=["(b,)"])(x))
    self.assertAllClose(9., jax2tf.convert(jax.jit(f), in_shapes=["(b,)"])(x))
    self.assertAllClose(9., tf.function(jax2tf.convert(f, in_shapes=["(b,)"]))(x))

    res_primal, res_tangent = jax2tf.convert(
      lambda x, xt: jax.jvp(f, (x,), (xt,)),
      in_shapes=["b", "b"])(x, np.array([0.1, 0.2, 0.3]))
    self.assertAllClose((9., 1.8), (res_primal, res_tangent))

    self.assertAllClose(np.array([3., 3., 3.]),
                        jax2tf.convert(jax.grad(f),
                                       in_shapes=["b"])(x))

    xv = np.arange(24.).reshape((2, 3, 4))
    res_vmap = jax.vmap(f, in_axes=1)(xv)
    # Implement by iteration
    res_iter = jnp.stack([f(xv[:, i, :]) for i in range(xv.shape[1])])
    self.assertAllClose(res_iter, res_vmap)

    res_mask2, _ = jax.mask(f, in_shapes=["(b,)"])([x], dict(b=2))
    self.assertAllClose(2., res_mask2)
    res_mask3, _ = jax.mask(f, in_shapes=["(b,)"])([x], dict(b=3))
    self.assertAllClose(9., res_mask3)

  def test_cond(self):
    # Test the primitive under conditional
    def f(x):
      return lax.cond(jnp.sum(x) > 0.,
                      lambda _: jnp.sum(x) / functools.reduce(lax.mul,
                                                              jax2tf.shape_as_value(x)),
                      lambda _: 0.,
                      operand=None)
    x = np.ones((2, 3, 4))
    self.assertAllClose(1., f(x))
    self.assertAllClose(1., jax2tf.convert(f, in_shapes=["(a, b, 4)"])(x))

  def test_mean0(self):
    def f_jax(x):
      return jnp.sum(x, axis=0) / jax2tf.shape_as_value(x)[0]

    x = np.arange(12.).reshape((3, 4))
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype)],
      in_shapes=[("batch, _")],
      expected_output_signature=tf.TensorSpec([4]))
    self.assertAllClose(np.array([4., 5., 6., 7.]), f_tf(x))

  def test_mean_all_axes(self):
    def f_jax(x):
      return jnp.sum(x) / np.prod(jax2tf.shape_as_value(x))

    x = np.arange(12.).reshape((3, 4))
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype)],
      in_shapes=[("batch, _")],
      expected_output_signature=tf.TensorSpec([]))

    self.assertAllClose(jnp.mean(x), f_tf(x))


class ShapePolyPrimitivesTest(tf_test_util.JaxToTfTestCase):
  """Tests for primitives that take shape values as parameters."""

  def test_matmul(self):
    raise unittest.SkipTest("Failing after fixing Poly unsoundness #4878")
    def f_jax(x, y):
      return jnp.matmul(x, y)

    self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 8, 4]), tf.TensorSpec([None, 4, None])],
      in_shapes=["(batch, _, 4)", "(batch, 4, w)"],
      expected_output_signature=tf.TensorSpec([None, 8, None]))

  def test_reshape(self):
    raise unittest.SkipTest("Failing after fixing Poly unsoundness #4878")
    def f_jax(x):
      y = jnp.sin(x)
      return y.reshape([2, -1])

    self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, None])],
      in_shapes=["(2 * batch, d)"],
      expected_output_signature=tf.TensorSpec([2, None]))

    self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([4, 3, None])],
      in_shapes=["(4, 3, d)"],
      expected_output_signature=tf.TensorSpec([2, None]))

  def test_reshape_compiled(self):
    raise unittest.SkipTest("Failing after fixing Poly unsoundness #4878")
    # We compile the result of conversion, hence we need to involve the compiler
    # twice, but we trace only once with shape polymorphism
    traced = False
    def f_jax(x):
      nonlocal traced
      traced = True
      y = jnp.sin(x)
      return y.reshape([2, -1])

    x = np.ones((4, 3), dtype=np.float32)
    res_jax = f_jax(x)

    traced = False
    # If we get_concrete_function we trace once
    f_tf = tf.function(jax2tf.convert(f_jax, in_shapes=["(2 * batch, d)"]),
                       autograph=False,
                       experimental_compile=True).get_concrete_function(tf.TensorSpec([None, None], tf.float32))
    self.assertTrue(traced)
    traced = False
    self.assertAllClose(res_jax, f_tf(x))
    self.assertFalse(traced)

    x = np.ones((6, 3), dtype=np.float32)
    res_jax = f_jax(x)
    traced = False

    self.assertAllClose(res_jax, f_tf(x))
    self.assertFalse(traced)  # We are not tracing again


  def test_add(self):
    raise unittest.SkipTest("Failing after fixing Poly unsoundness #4878")
    def f_jax(x, y):
      return jnp.add(x, y)

    x = np.arange(12.).reshape((3, 4))
    y = np.arange(24).reshape((2, 3, 4))
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype),
                       tf.TensorSpec([None, None, 4], dtype=y.dtype)],
      in_shapes=["(d, 4)", "(batch, d, 4)"],
      expected_output_signature=tf.TensorSpec([None, None, 4]))

    self.assertAllClose(f_jax(x, y), f_tf(x, y))

  def test_squeeze(self):
    raise unittest.SkipTest("Failing after fixing Poly unsoundness #4878")
    def f_jax(x):
      return jnp.squeeze(x, axis=1)
    x = np.ones((4, 1))
    res_jax = f_jax(x)

    # Trace with a known dimension to squeeze
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 1], dtype=x.dtype)],
      in_shapes=["(b, _)"],
      expected_output_signature=tf.TensorSpec([None]))

    self.assertAllClose(res_jax, f_tf(x))

    with self.assertRaisesRegex(
        ValueError,
        re.escape("cannot select an axis to squeeze out which has size not equal to one, got shape=(b1, b2) and dimensions=(1,)")):
      # Trace with unknown dimension to squeeze
      self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([None, None])],
        in_shapes=["(b1, b2)"],
        expected_output_signature=tf.TensorSpec([None]))

  def test_broadcast(self):
    raise unittest.SkipTest("Failing after fixing Poly unsoundness #4878")
    def f_jax(x):
      return jnp.broadcast_to(x, [x.shape[0], x.shape[0], x.shape[1]])

    x = np.arange(12.).reshape((3, 4))
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype)],
      in_shapes=[("batch, _")],
      expected_output_signature=tf.TensorSpec([None, None, 4]))

    self.assertAllClose(f_jax(x), f_tf(x))

  def test_iota(self):
    raise unittest.SkipTest("not yet working")
    def f_jax(x):
      x + lax.iota(np.float32, x.shape[0])

    x = np.arange(12.)
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None], dtype=x.dtype)],
      in_shapes=[("d")],
      expected_output_signature=tf.TensorSpec([None]))

    self.assertAllClose(f_jax(x), f_tf(x))

  def test_gather(self):
    def f(a, i):
      return jnp.take(a, i, axis=1)

    x = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))[:2, :3, :4]
    i = np.array([1, 2], np.int32)

    f_tf = self.CheckShapePolymorphism(
      f,
      input_signature=[tf.TensorSpec([None, 3, 4]), tf.TensorSpec([2], np.int32)],
      in_shapes=["batch, _, _", "_"],
      expected_output_signature=tf.TensorSpec([None, 2, 4]))

    self.assertAllClose(f(x, i), f_tf(x, i))

    # Does not yet work
    # f_tf = self.CheckShapePolymorphism(
    #   f,
    #   input_signature=[tf.TensorSpec([None, 3, 4]), tf.TensorSpec([None], np.int32)],
    #   in_shapes=["batch, _, _", "slice_size"],
    #   expected_output_signature=tf.TensorSpec([None, None, 4]))
    # self.assertAllClose(f(x, i), f_tf(x, i))

  def test_gather_vmap(self):
    raise unittest.SkipTest("does not yet work")
    @jax.vmap
    def f(a, i):
      return jnp.take(a, i, axis=0)

    x = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))[:2, :3, :4]
    i = np.array([1, 2], np.int32)

    f_tf = self.CheckShapePolymorphism(
      f,
      input_signature=[tf.TensorSpec([None, 3, 4]), tf.TensorSpec([None], np.int32)],
      in_shapes=["batch, _, _", "batch"],
      expected_output_signature=tf.TensorSpec([None, 2, 4]))

    self.assertAllClose(f(x, i), f_tf(x, i))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
