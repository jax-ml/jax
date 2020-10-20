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

import functools
import operator
import re
import unittest

import jax
from jax import core
from jax.experimental import jax2tf
import jax.numpy as jnp
from jax import test_util as jtu
import numpy as np
from jax.interpreters import masking


from jax.experimental.jax2tf.tests import tf_test_util

import tensorflow as tf  # type: ignore[import]

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
                                in_shapes=None,
                                expected_output_signature=tf.TensorSpec([2, None]))

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

    # Known shapes
    # self.assertEqual((shaped_array([2, 3]),),
    #                  input_avals([const((2, 3))], [None]))
    # self.assertEqual((shaped_array([2, 3]),),
    #                  input_avals([tf_const((2, 3))], [None]))
    # self.assertEqual((shaped_array([2, 3]),),
    #                  input_avals([tf_var((2, 3), (2, 3))], [None]))
    # self.assertEqual((shaped_array([2, 3]),),
    #                  input_avals([const((2, 3))], ["(2, 3)"]))
    # self.assertEqual((shaped_array([2, 3]),),
    #                  input_avals([tf_const((2, 3))], ["(_, 3)"]))
    # self.assertEqual((shaped_array([2, 3]),),
    #                  input_avals([tf_const((2, 3))], ["(_, 3)"]))
    #
    # # Partially known shapes
    # self.assertEqual((shaped_array([2, 3]),),
    #                   input_avals([tf_var((2, 3), [None, 3])], ["(2, 3)"]))
    #
    # self.assertEqual((shaped_array("(h, h)"),),
    #                   input_avals([tf_var((2, 3), [None, None])], [("h, h")]))
    #
    # # Partially known shapes, create shape variables
    # self.assertEqual((shaped_array("(s0, s1)"),),
    #                   input_avals([tf_var((2, 3), [None, None])], [None]))
    # self.assertEqual((shaped_array("(2, s0)"),),
    #                   input_avals([tf_var((2, 3), [2, None])], [None]))

    # Some errors
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
    # TODO: is this test really adding anything???
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


  def test_matmul(self):
    def f_jax(x, y):
      return jnp.matmul(x, y)

    self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 8, 4]), tf.TensorSpec([None, 4, None])],
      in_shapes=["(batch, _, 4)", "(batch, 4, w)"],
      expected_output_signature=tf.TensorSpec([None, 8, None]))

  def test_reshape(self):
    raise unittest.SkipTest("Not implemented")
    def f_jax(x):
      y = jnp.sin(x)
      yshape0, yshape1 = y.shape
      return y.reshape([2, -1])

    self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, None])],
      in_shapes=["(2 * batch, d)"],
      expected_output_signature=tf.TensorSpec([2, None]))


  def test_mean(self):
    raise unittest.SkipTest("Not yet implemented")
    def f_jax(x):
      return jnp.sum(x, axis=0) / jax2tf.shape_as_value(x)[0]

    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 4])],
      in_shapes=[("batch, _")],
      expected_output_signature=tf.TensorSpec([4]))
    x = np.arange(12.).reshape((3, 4))
    self.assertAllClose(np.array([4., 5., 6., 7.]), f_tf(x))


  def test_shape_error(self):
    """Some of the examples from the README."""
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

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
