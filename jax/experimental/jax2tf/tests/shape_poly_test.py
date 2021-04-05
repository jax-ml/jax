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
from typing import Dict, Optional, Sequence

import collections
import functools
import operator
import re

import jax
from jax import core
from jax.experimental import jax2tf
from jax.experimental.jax2tf import shape_poly
from jax import lax
import jax.numpy as jnp
from jax import test_util as jtu
from jax._src.lax import control_flow as lax_control_flow
import numpy as np


from jax.experimental.jax2tf.tests import tf_test_util
from jax.experimental.jax2tf.tests.jax2tf_limitations import Jax2TfLimitation

import tensorflow as tf  # type: ignore[import]
import unittest

from jax.config import config
config.parse_flags_with_absl()

# Import after parsing flags
from jax.experimental.jax2tf.tests import primitive_harness

class ShapePolyTest(tf_test_util.JaxToTfTestCase):

  def setUp(self):
    pass # raise unittest.SkipTest("shape polymorphism not supported anymore. See #6080.")

  def test_simple(self):
    """Test shape polymorphism for a simple case."""
    def f_jax(x):
      return x + jnp.sin(x)

    self.CheckShapePolymorphism(f_jax,
                                input_signature=[tf.TensorSpec([2, 3])],
                                polymorphic_shapes=None,
                                expected_output_signature=tf.TensorSpec([2, 3]))

    self.CheckShapePolymorphism(f_jax,
                                input_signature=[tf.TensorSpec([2, None])],
                                polymorphic_shapes=["(_, h)"],
                                expected_output_signature=tf.TensorSpec([2, None]))

    self.CheckShapePolymorphism(f_jax,
                                input_signature=[tf.TensorSpec([None, None])],
                                polymorphic_shapes=["(h, h)"],
                                expected_output_signature=tf.TensorSpec([None, None]))

  def test_arg_avals(self):
    """Test conversion of actual arguments to abstract values"""
    def check_avals(*, args: Sequence[jax2tf.jax2tf.TfVal],
                    polymorphic_shapes: Sequence[Optional[str]],
                    expected_avals: Sequence[core.ShapedArray]):
      avals, shape_env = jax2tf.jax2tf._args_to_avals_and_env(args,
                                                              polymorphic_shapes)  # The function under test
      self.assertEqual(expected_avals, avals)

    def shaped_array(shape_spec: str, actual_shape: core.Shape):
      return core.ShapedArray(shape_poly.parse_spec(shape_spec, actual_shape), np.float32)

    def const(shape):
      return np.ones(shape, dtype=np.float32)
    def tf_const(shape):
      return tf.convert_to_tensor(np.ones(shape, dtype=np.float32))
    def tf_var(shape, *, initializer_shape=None):
      initializer_shape = initializer_shape or shape
      self.assertEmpty([d for d in initializer_shape if d is None])
      return tf.Variable(np.ones(initializer_shape, np.float32),
                         dtype=tf.float32, shape=shape)

    # Known shapes for the arguments
    check_avals(args=[const((2, 3))],
                polymorphic_shapes=[None],
                expected_avals=(shaped_array("2, 3", [2, 3]),))

    check_avals(args=[tf_const((2, 3))],
                polymorphic_shapes=[None],
                expected_avals=(shaped_array("2, 3", [2, 3]),))

    check_avals(args=[tf_var((2, 3))],
                polymorphic_shapes=[None],
                expected_avals=(shaped_array("2, 3", [2, 3]),))

    check_avals(args=[const((2, 3))],
                polymorphic_shapes=["(2, 3)"],
                expected_avals=(shaped_array("2, 3", [2, 3]),))

    check_avals(args=[tf_const((2, 3))],
                polymorphic_shapes=["(_, 3)"],
                expected_avals=(shaped_array("2, 3", [2, 3]),))

    # Partially known shapes for the arguments
    check_avals(args=[tf_var([None, 3], initializer_shape=(2, 3))],
                polymorphic_shapes=["(b, 3)"],
                expected_avals=(shaped_array("(b, 3)", (2, 3)),))

    check_avals(args=[tf_var([None, None], initializer_shape=(2, 3))],
                polymorphic_shapes=[("h, h")],
                expected_avals=(shaped_array("(h, h)", (2, 2)),))

    check_avals(args=[tf_var([2, None], initializer_shape=(2, 3))],
                polymorphic_shapes=[("h, h")],
                expected_avals=(shaped_array("(h, h)", (2, 2)),))

    check_avals(args=[tf_var([None, 3, 4], initializer_shape=(2, 3, 4))],
                polymorphic_shapes=["(c, b, a)"],
                expected_avals=(shaped_array("(c, b, a)", (2, 3, 4)),),)

    # Some errors
    with self.assertRaisesRegex(ValueError,
                                re.escape("polymorphic_shape must be specified when the argument shape (2, None) is partially known")):
      check_avals(args=[tf_var([2, None], initializer_shape=(2, 3))],
                  polymorphic_shapes=[None],
                  expected_avals=None)

    with self.assertRaisesRegex(
        ValueError,
        re.escape("polymorphic_shape '()' has different rank than argument shape (2, 3)")):
      check_avals(args=[const((2, 3))],
                  polymorphic_shapes=["()"],
                  expected_avals=None)

    with self.assertRaisesRegex(
        ValueError,
        re.escape("polymorphic_shape '(_, _)' has `_` placeholders for argument shape dimensions that are unknown: (2, None)")):
      check_avals(args=[tf_var([2, None], initializer_shape=(2, 3))],
                  polymorphic_shapes=["(_, _)"],
                  expected_avals=None)

    with self.assertRaisesRegex(
        ValueError,
        re.escape("polymorphic_shape '(2, 13)' does not match argument shape (2, 3)")):
      check_avals(args=[const((2, 3))],
                  polymorphic_shapes=["(2, 13)"],
                  expected_avals=None)

    with self.assertRaisesRegex(
        ValueError,
        re.escape("polymorphic_shape '(2, 3)' must contain shape variables for argument shape dimensions that are unknown: (2, None)")):
      check_avals(args=[tf_var([2, None], initializer_shape=(2, 3))],
                  polymorphic_shapes=["(2, 3)"],
                  expected_avals=None)

    with self.assertRaisesRegex(
        ValueError,
        re.escape("polymorphic shape variable 'a' corresponds to multiple values ([2, 3]), in polymorphic_shape '(a, a)' and argument shape (2, 3)")):
      check_avals(args=[tf_var([2, 3], initializer_shape=(2, 3))],
                  polymorphic_shapes=["(a, a)"],
                  expected_avals=None)


  def test_bad_polymorphic_shapes(self):
    def add2(x, y):
      return x + y

    with self.assertRaisesRegex(shape_poly.ShapeSyntaxError, ""):
      self.CheckShapePolymorphism(add2,
                                  input_signature=[tf.TensorSpec([None]), tf.TensorSpec([None])],
                                  polymorphic_shapes=[") + (", None],
                                  expected_output_signature=tf.TensorSpec([None]))

    with self.assertRaisesRegex(TypeError,
                                re.escape("polymorphic_shapes must be a sequence with the same length as the argument list (2). "
                                          "Got polymorphic_shapes_experimental=['(b, 4)']")):
      self.CheckShapePolymorphism(add2,
                                  input_signature=[tf.TensorSpec([None]), tf.TensorSpec([None])],
                                  polymorphic_shapes=["(b, 4)"],
                                  expected_output_signature=tf.TensorSpec([None]))

  def test_pytree(self):
    """Arguments and polymorphic_shapes are pytrees."""

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
      polymorphic_shapes=[(["(v,)", "(v,)"], [("v,")]),
                 dict(a="(v,)", b="(v,)")],
      expected_output_signature=tf.TensorSpec([None]))

    # Now partial polymorphic_shapes; the parts of the polymorphic_shapes that are not specified
    # must have full input_signatures.
    self.CheckShapePolymorphism(
      add_all_jax,
      input_signature=[([tf.TensorSpec([4]), tf.TensorSpec([4])],
                        [tf.TensorSpec([4])]),
                       dict(a=tf.TensorSpec([4]), b=tf.TensorSpec([4]))],
      polymorphic_shapes=[(["(4,)", "(_,)"], [("4,")]),
                          dict(a="(_,)", b="(4,)")],
      expected_output_signature=tf.TensorSpec([4]))

  def test_with_custom_vjp(self):
    """Shape-polymorphic custom VJP."""
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
      polymorphic_shapes=["(batch1, batch2, d1, d2)"],
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
      polymorphic_shapes=[dict(x=("b, 3, 4"))],
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
    self.assertAllClose(res_jax,
                        jax2tf.convert(f, polymorphic_shapes_experimental=["(b, h)", "h"])(x, y))

  def test_shape_error(self):
    """Some of the examples from the README."""
    with self.assertRaisesRegex(TypeError,
                                re.escape("add got incompatible shapes for broadcasting: (v,), (4,)")):
      self.CheckShapePolymorphism(
        lambda x, y: x + y,
        input_signature=[tf.TensorSpec([None]), tf.TensorSpec([4])],
        polymorphic_shapes=["(v,)", "(4,)"],
        expected_output_signature=tf.TensorSpec([None]))

    four_ones = np.ones((4,))
    # We get the error even if we use correct actual arguments
    with self.assertRaisesRegex(TypeError,
                                re.escape("add got incompatible shapes for broadcasting: (v,), (4,)")):
      jax2tf.convert(lambda x, y: x + y,
                     polymorphic_shapes_experimental=["(v,)", "(4,)"])(four_ones, four_ones)

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                re.escape("Shape variable comparison v == 4 is inconclusive")):
      jax2tf.convert(lambda x: jnp.matmul(x, x),
                     polymorphic_shapes_experimental=["(v, 4)"])(np.ones((4, 4)))


  def test_dim_vars(self):
    """Unit tests for DimVar."""
    da, db = shape_poly.parse_spec("a, b", (2, 3))
    self.assertTrue(da == da)
    self.assertFalse(da != da)
    with self.assertRaisesRegex(core.InconclusiveDimensionOperation, ""):
      da == db
    with self.assertRaisesRegex(core.InconclusiveDimensionOperation, ""):
      da != db

    self.assertLen({da, da}, 1)
    self.assertLen({da, db}, 2)
    self.assertTrue(da in {da, db})
    self.assertTrue(db in {da, db})
    self.assertTrue(da in [da, db])
    with self.assertRaisesRegex(core.InconclusiveDimensionOperation, ""):
      db in [da, db]

  def test_dim_vars_symbolic_equal(self):
    da, db = shape_poly.parse_spec("a, b", (2, 3))
    self.assertTrue(core.symbolic_equal_dim(da, da))
    self.assertFalse(core.symbolic_equal_dim(da, 1))
    self.assertFalse(core.symbolic_equal_dim(da, db))

    self.assertTrue(core.symbolic_equal_one_of_dim(da, [2, da]))
    self.assertFalse(core.symbolic_equal_one_of_dim(da, [2, db]))
    self.assertFalse(core.symbolic_equal_one_of_dim(da, []))

    self.assertTrue(core.symbolic_equal_one_of_dim(2, [da, 3, 2]))
    self.assertFalse(core.symbolic_equal_one_of_dim(1, [2, db]))
    self.assertFalse(core.symbolic_equal_one_of_dim(3, []))

  def test_dim_vars_greater_equal(self):
    da, db = shape_poly.parse_spec("a, b", (2, 3))
    self.assertTrue(core.greater_equal_dim(da, da))
    self.assertTrue(core.greater_equal_dim(da, 0))
    self.assertTrue(core.greater_equal_dim(da, 1))

    self.assertTrue(core.greater_equal_shape((da, 2), (1, 1)))

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                "Shape variable comparison .* is inconclusive"):
      core.greater_equal_dim(da, 2)

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                "Shape variable comparison .* is inconclusive"):
      core.greater_equal_dim(da, db)


class ShapeAsValueTest(tf_test_util.JaxToTfTestCase):

  def setUp(self):
    raise unittest.SkipTest("shape_as_value not supported anymore. See #6080.")

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

    res_mask2, _ = jax.mask(f, polymorphic_shapes=["(b,)"])([x], dict(b=2))
    self.assertAllClose(2., res_mask2)
    res_mask3, _ = jax.mask(f, polymorphic_shapes=["(b,)"])([x], dict(b=3))
    self.assertAllClose(9., res_mask3)

  def test_dynamic_shapes(self):
    # Test shape_as_value with dynamic shapes. All transformations work.
    def f(x):
      return jnp.sum(x, axis=0) * jax2tf.shape_as_value(x)[0]

    x = np.arange(3.)
    self.assertAllClose(9., jax2tf.convert(f, polymorphic_shapes_experimental=["(b,)"])(x))
    self.assertAllClose(9., jax2tf.convert(jax.jit(f), polymorphic_shapes_experimental=["(b,)"])(x))
    self.assertAllClose(9., tf.function(jax2tf.convert(f, polymorphic_shapes_experimental=["(b,)"]))(x))

    res_primal, res_tangent = jax2tf.convert(
      lambda x, xt: jax.jvp(f, (x,), (xt,)),
      polymorphic_shapes=["b", "b"])(x, np.array([0.1, 0.2, 0.3]))
    self.assertAllClose((9., 1.8), (res_primal, res_tangent))

    self.assertAllClose(np.array([3., 3., 3.]),
                        jax2tf.convert(jax.grad(f),
                                       polymorphic_shapes_experimental=["b"])(x))

    xv = np.arange(24.).reshape((2, 3, 4))
    res_vmap = jax.vmap(f, in_axes=1)(xv)
    # Implement by iteration
    res_iter = jnp.stack([f(xv[:, i, :]) for i in range(xv.shape[1])])
    self.assertAllClose(res_iter, res_vmap)

    res_mask2, _ = jax.mask(f, polymorphic_shapes=["(b,)"])([x], dict(b=2))
    self.assertAllClose(2., res_mask2)
    res_mask3, _ = jax.mask(f, polymorphic_shapes=["(b,)"])([x], dict(b=3))
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
    self.assertAllClose(1., jax2tf.convert(f, polymorphic_shapes_experimental=["(a, b, 4)"])(x))

  def test_mean0(self):
    def f_jax(x):
      return jnp.sum(x, axis=0) / jax2tf.shape_as_value(x)[0]

    x = np.arange(12.).reshape((3, 4))
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype)],
      polymorphic_shapes=[("batch, _")],
      expected_output_signature=tf.TensorSpec([4]))
    self.assertAllClose(np.array([4., 5., 6., 7.]), f_tf(x))

  def test_mean_all_axes(self):
    def f_jax(x):
      return jnp.sum(x) / np.prod(jax2tf.shape_as_value(x))

    x = np.arange(12.).reshape((3, 4))
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype)],
      polymorphic_shapes=[("batch, _")],
      expected_output_signature=tf.TensorSpec([]))

    self.assertAllClose(jnp.mean(x), f_tf(x))

def _all_shape_poly_harnesses() -> Sequence[primitive_harness.Harness]:
  """For each harness group, pick a single dtype.
  Ignore harnesses that fail in graph mode in jax2tf.
  """
  all_h = primitive_harness.all_harnesses

  # Index by group
  harness_groups: Dict[str, Sequence[primitive_harness.Harness]] = collections.defaultdict(list)
  device = jtu.device_under_test()


  for h in all_h:
    # Drop the the JAX limitations
    if not h.filter(device_under_test=device, include_jax_unimpl=False):
      continue
    # And the jax2tf limitations
    if _get_jax2tf_limitations(device, h):
      continue
    harness_groups[h.group_name].append(h)

  res = []
  for group_name, hlist in harness_groups.items():
    # Pick the dtype with the most harnesses in this group. Some harness
    # groups only test different use cases at a few dtypes.
    c = collections.Counter([h.dtype for h in hlist])
    (dtype, _), = c.most_common(1)
    res.extend([h for h in hlist if h.dtype == dtype])
  return res


def _get_jax2tf_limitations(device, h: primitive_harness.Harness) -> Sequence[Jax2TfLimitation]:
  # And the jax2tf limitations
  def applicable_jax2tf_limitation(l: Jax2TfLimitation) -> bool:
    return (l.filter(device=device,
                     dtype=h.dtype,
                     mode="graph") and l.expect_tf_error)
  limitations = Jax2TfLimitation.limitations_for_harness(h)
  return tuple(filter(applicable_jax2tf_limitation, limitations))

# We do not yet support shape polymorphism for vmap for all primitives
_VMAP_NOT_POLY_YET = {
    # In the random._gamma_impl we do reshape(-1, 2) for the keys
    "random_gamma",

    # In linalg._lu_python we do reshape(-1, ...)
    "lu",
    "custom_linear_solve",

    # We do *= shapes in the batching rule for conv_general_dilated
    "conv_general_dilated",

    # vmap(clamp) fails in JAX
    "clamp",
}

class ShapePolyPrimitivesTest(tf_test_util.JaxToTfTestCase):
  """Tests for primitives that take shape values as parameters."""

  # This test runs for all primitive harnesses, and verifies that the result
  # of vmap over a primitive harness can be converted batch-polymorphically.
  @primitive_harness.parameterized(_all_shape_poly_harnesses(),
                                   include_jax_unimpl=False)
  @jtu.ignore_warning(
      category=UserWarning, message="Using reduced precision for gradient.*")
  def test_prim_vmap(self, harness: primitive_harness.Harness):
    if harness.group_name in _VMAP_NOT_POLY_YET:
      raise unittest.SkipTest(f"TODO: vmap({harness.group_name}) not yet supported")
    func_jax = harness.dyn_fun
    args = harness.dyn_args_maker(self.rng())
    if len(args) == 0:
      # vmap not defined for functions with no args
      return

    res_jax = func_jax(*args)

    # Replicate all arguments
    batch_size = 3
    batched_args = [np.stack([a] * batch_size) for a in args]
    func_jax_vmap = jax.vmap(func_jax, in_axes=0, out_axes=0)
    # Check that batching works
    res_jax_vmap = func_jax_vmap(*batched_args)

    def arr_to_shape_spec(a):
      return "b, " + ", ".join(str(d) for d in a.shape)
    func_jax_vmap_polymorphic_shapes = jax.tree_map(arr_to_shape_spec, tuple(args))
    def arr_to_tf_tensor_spec(a):
      return tf.TensorSpec((None,) + a.shape, a.dtype)
    func_jax_vmap_input_signature = jax.tree_map(arr_to_tf_tensor_spec,
                                                 tuple(args))
    func_jax_vmap_output_signature = jax.tree_map(arr_to_tf_tensor_spec,
                                                  res_jax)
    f_tf = self.CheckShapePolymorphism(func_jax_vmap,
                                input_signature=func_jax_vmap_input_signature,
                                polymorphic_shapes=func_jax_vmap_polymorphic_shapes,
                                expected_output_signature=func_jax_vmap_output_signature)

    limitations = _get_jax2tf_limitations(jtu.device_under_test(), harness)
    if any([l.custom_assert or l.skip_comparison for l in limitations]):
      self.assertAllClose(res_jax_vmap, f_tf(*batched_args))

  def test_add_with_broadcast(self):
    def f_jax(x, y):
      return jnp.add(x, y)

    x = np.arange(12.).reshape((3, 4))
    y = np.arange(24).reshape((2, 3, 4))
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype),
                       tf.TensorSpec([None, None, 4], dtype=y.dtype)],
      polymorphic_shapes=["(d, _)", "(batch, d, _)"],
      expected_output_signature=tf.TensorSpec([None, None, 4]))

    self.assertAllClose(f_jax(x, y), f_tf(x, y))

  def test_broadcast(self):
    def f_jax(x):
      return jnp.broadcast_to(x, [x.shape[0], x.shape[0], x.shape[1]])

    x = np.arange(12.).reshape((3, 4))
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype)],
      polymorphic_shapes=[("batch, _")],
      expected_output_signature=tf.TensorSpec([None, None, 4]))

    self.assertAllClose(f_jax(x), f_tf(x))

  def test_ones(self):
    def f_jax(x):
      return jnp.ones(x.shape, dtype=x.dtype)

    x_shape = (5, 6, 4)
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, None, 4], dtype=x.dtype)],
      polymorphic_shapes=[("width, height, _")],
      expected_output_signature=tf.TensorSpec([None, None, 4]))

    self.assertAllClose(f_jax(x), f_tf(x))

  def test_clamp(self):
    @jax.vmap
    def f_jax(mi, x, ma):
      return lax.clamp(mi, x, ma)

    x = np.ones((7, 2, 3))
    f_tf = self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([None, 2, 3]),
                         tf.TensorSpec([None, 2, 3]),
                         tf.TensorSpec([None, 2, 3]),],
        polymorphic_shapes=["b, _, _", "b, _, _", "b, _, _"],
        expected_output_signature=tf.TensorSpec([None, 2, 3]))
    self.assertAllClose(f_jax(x, x, x), f_tf(x, x, x))

  def test_conv_general_dilated(self):
    batch_size = 7
    lhs_shape = (batch_size, 3, 9, 10)  # 2 spatial dimensions (9, 10)
    rhs_shape = (3, 3, 4, 5)
    window_strides = (2, 3)
    padding = ((0, 0), (0, 0))
    lhs_dilation = (1, 1)
    rhs_dilation = (1, 2)
    dimension_numbers = ('NCHW','OIHW','NCHW')
    feature_group_count = 1
    batch_group_count = 1
    precision = None

    lhs = np.random.rand(*lhs_shape)
    rhs = np.random.rand(*rhs_shape)

    def f_jax(lhs, rhs):
      return lax.conv_general_dilated(lhs, rhs,
          window_strides, padding, lhs_dilation, rhs_dilation,
          dimension_numbers, feature_group_count,
          batch_group_count, precision)

    f_tf = self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec(lhs_shape),
                         tf.TensorSpec(rhs_shape)],
        polymorphic_shapes=["b, _, _, _", "_, _, _, _"],
        expected_output_signature=tf.TensorSpec([None, 3, 3, 1]))
    self.assertAllClose(f_jax(lhs, rhs), f_tf(lhs, rhs))


  def test_conv_general_dilated_vmap(self):
    raise unittest.SkipTest("TODO: vmap(conv_general_dilated) not yet supported")
    lhs_shape = (2, 3, 9, 10)
    rhs_shape = (3, 3, 4, 5)
    window_strides = (2, 3)
    padding = ((0, 0), (0, 0))
    lhs_dilation = (1, 1)
    rhs_dilation = (1, 2)
    dimension_numbers = ('NCHW','OIHW','NCHW')
    feature_group_count = 1
    batch_group_count = 1
    precision = None

    batch_size = 7

    lhs = np.random.rand(batch_size, *lhs_shape)
    rhs = np.random.rand(batch_size, *rhs_shape)
    @jax.vmap
    def f_jax(lhs, rhs):
      return lax.conv_general_dilated(lhs, rhs,
          window_strides, padding, lhs_dilation, rhs_dilation,
          dimension_numbers, feature_group_count,
          batch_group_count, precision)

    f_tf = self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec((None,) + lhs_shape),
                         tf.TensorSpec((None,) + rhs_shape)],
        polymorphic_shapes=["b, _, _, _, _", "b, _, _, _, _"],
        expected_output_signature=tf.TensorSpec([None, 2, 3]))
    self.assertAllClose(f_jax(lhs, rhs), f_tf(lhs, rhs))

  def test_cummax(self):
    @jax.vmap
    def f_jax(x):
      return lax_control_flow.cummax(x, axis=0, reverse=False)

    batch_size = 7
    x = np.random.rand(batch_size, 8, 9)

    f_tf = self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec((None, 8, 9))],
        polymorphic_shapes=["b, _, _"],
        expected_output_signature=tf.TensorSpec([None, 8, 9]))
    self.assertAllClose(f_jax(x), f_tf(x))

  def test_dot_general(self):
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    def f_jax(lhs, rhs):  # lhs: [b, 4, 4], rhs: [b, 4]
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

    batch_size = 7
    lhs = np.random.rand(batch_size, 4, 4)
    rhs = np.random.rand(batch_size, 4)

    f_tf = self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec((None, 4, 4)),
                         tf.TensorSpec((None, 4))],
        polymorphic_shapes=["b, _, _", "b, _"],
        expected_output_signature=tf.TensorSpec([None, 4]))
    self.assertAllClose(f_jax(lhs, rhs), f_tf(lhs, rhs))


  def test_dynamic_slice(self):

    def f_jax(x):  # x:shape: (b, 4)
      return lax.dynamic_slice(x, (0, 1,), (x.shape[0], 2,))
    batch_size = 7
    x = np.arange(100, dtype=np.float32).reshape((10, 10))[:batch_size, :4]

    f_tf = self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec((None, 4))],
        polymorphic_shapes=["b, _"],
        expected_output_signature=tf.TensorSpec([None, 2]))
    self.assertAllClose(f_jax(x), f_tf(x))

  def test_dynamic_slice_vmap(self):
    @jax.vmap
    def f_jax(x, idx):  # x.shape : (4,)
      return lax.dynamic_slice(x, idx, slice_sizes=(2,))
    batch_size = 7
    x = np.arange(100, dtype=np.float32).reshape((10, 10))[:batch_size, :4]
    idx = np.arange(batch_size, dtype=np.int32).reshape((batch_size, 1))
    f_tf = self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec((None, 4)),
                         tf.TensorSpec((None, 1), dtype=idx.dtype)],
        polymorphic_shapes=["b, _", "b, _"],
        expected_output_signature=tf.TensorSpec([None, 2]))
    self.assertAllClose(f_jax(x, idx), f_tf(x, idx))

  def test_gather(self):
    def f(a, i):
      return jnp.take(a, i, axis=1)

    x = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))[:2, :3, :4]
    i = np.array([1, 2], np.int32)

    f_tf = self.CheckShapePolymorphism(
      f,
      input_signature=[tf.TensorSpec([None, 3, 4]), tf.TensorSpec([2], np.int32)],
      polymorphic_shapes=["batch, _, _", "_"],
      expected_output_signature=tf.TensorSpec([None, 2, 4]))

    self.assertAllClose(f(x, i), f_tf(x, i))

    # Does not yet work
    # f_tf = self.CheckShapePolymorphism(
    #   f,
    #   input_signature=[tf.TensorSpec([None, 3, 4]), tf.TensorSpec([None], np.int32)],
    #   polymorphic_shapes=["batch, _, _", "batch"],
    #   expected_output_signature=tf.TensorSpec([None, None, 4]))
    # self.assertAllClose(f(x, i), f_tf(x, i))

  def test_gather_vmap(self):
    @jax.vmap
    def f(a, i):
      return jnp.take(a, i, axis=0)

    x = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))[:2, :3, :4]
    i = np.array([1, 2], np.int32)

    f_tf = self.CheckShapePolymorphism(
      f,
      input_signature=[tf.TensorSpec([None, 3, 4]), tf.TensorSpec([None], np.int32)],
      polymorphic_shapes=["batch, _, _", "batch"],
      expected_output_signature=tf.TensorSpec([None, 4]))

    self.assertAllClose(f(x, i), f_tf(x, i))


  def test_iota(self):
    def f_jax(x):
      x + lax.iota(np.float32, x.shape[0])

    x = np.arange(12.)
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None], dtype=x.dtype)],
      polymorphic_shapes=["d"],
      expected_output_signature=None)

    self.assertAllClose(f_jax(x), f_tf(x))

  def test_matmul(self):
    def f_jax(x, y):
      return jnp.matmul(x, y)

    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 8, 4]), tf.TensorSpec([None, 4, None])],
      polymorphic_shapes=["(batch, _, 4)", "(batch, 4, w)"],
      expected_output_signature=tf.TensorSpec([None, 8, None]))
    x = np.random.rand(7, 8, 4)
    y = np.random.rand(7, 4, 5)
    self.assertAllClose(f_jax(x, y), f_tf(x, y))

  def test_pad(self):
    def f_jax(x):
      return lax.pad(x, 5., ((0, 0, 0), (0, 0, 0), (1, 1, 1)))

    batch_size = 7
    x = np.random.rand(batch_size, 2, 3)
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 2, 3])],
      polymorphic_shapes=["(batch, _, _)"],
      expected_output_signature=tf.TensorSpec([None, 2, 7]))
    self.assertAllClose(f_jax(x), f_tf(x))

  def test_random_gamma(self):
    if "random_gamma" in _VMAP_NOT_POLY_YET:
      raise unittest.SkipTest("TODO: vmap(random_gamma) not yet supported")

    def f_jax(key, a):
      return jax.random.gamma(key, a)
    batch_size = 7
    key = np.random.rand(batch_size, 2)
    a = np.random.rand(batch_size, 3)

    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 2], dtype=key.dtype),
                       tf.TensorSpec([None, 3], dtype=a.dtype)],
      polymorphic_shapes=["(batch, _)", "(batch, _)"],
      expected_output_signature=tf.TensorSpec([None, 3]))
    self.assertAllClose(f_jax(key, a), f_tf(key, a))

  def test_reshape(self):

    self.CheckShapePolymorphism(
      lambda x: x.reshape([x.shape[0], -1]),
      input_signature=[tf.TensorSpec([None, 2, 3])],
      polymorphic_shapes=["(batch, _, _)"],
      expected_output_signature=tf.TensorSpec([None, 6]))

    self.CheckShapePolymorphism(
      lambda x: x.reshape([x.shape[0], -1, x.shape[3], x.shape[2]]),
      input_signature=[tf.TensorSpec([None, 2, None, None, 3])],
      polymorphic_shapes=["(batch, 2, batch, height, 3)"],
      expected_output_signature=tf.TensorSpec([None, 6, None, None]))

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                re.escape("Shapes (batch, 2, batch, height, 3) and (batch, -1, batch) must have the same set of shape variables")):
      self.CheckShapePolymorphism(
        lambda x: x.reshape([x.shape[0], -1, x.shape[2]]),
        input_signature=[tf.TensorSpec([None, 2, None, None, 3])],
        polymorphic_shapes=["(batch, 2, batch, height, 3)"],
        expected_output_signature=tf.TensorSpec([None, 6, None]))

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                re.escape("Cannot divide evenly the sizes of shapes (2, 4) and (-1, 3)")):
      self.CheckShapePolymorphism(
        lambda x: x.reshape([x.shape[0], -1, 3]),
        input_signature=[tf.TensorSpec([None, 2, 4])],
        polymorphic_shapes=["(batch, _, _)"],
        expected_output_signature=tf.TensorSpec([None, 1]))

  def test_reshape_compiled(self):
    # We compile the result of conversion for two shapes, hence we need to
    # involve the TF compiler twice, but we trace only once with shape polymorphism
    traced = False
    def f_jax(x):
      nonlocal traced
      traced = True
      y = jnp.sin(x)
      return y.reshape([x.shape[0], -1])

    x = np.random.rand(4, 2, 3)
    res_jax = f_jax(x)

    traced = False
    # If we get_concrete_function we trace once
    f_tf = tf.function(jax2tf.convert(f_jax, polymorphic_shapes_experimental=["(b, _, _)"]),
                       autograph=False,
                       jit_compile=True).get_concrete_function(tf.TensorSpec([None, 2, 3], x.dtype))
    self.assertTrue(traced)
    traced = False
    self.assertAllClose(res_jax, f_tf(x))
    self.assertFalse(traced)  # We are not tracing again

    x = np.random.rand(6, 2, 3)
    res_jax = f_jax(x)
    traced = False

    self.assertAllClose(res_jax, f_tf(x))
    self.assertFalse(traced)  # We are not tracing again

  def test_scatter(self):
    batch_size = 7
    x = np.arange(100, dtype=np.float32).reshape((10, 10))[:batch_size, :4]
    idx = np.array([[1], [2]], np.int32)
    upd = - np.arange(100, dtype=np.float32).reshape((10, 10))[:batch_size, :2]
    dimension_numbers=((0,), (1,), (1,))
    def f_jax(x, upd):
      return lax.scatter_add(x, scatter_indices=idx,
                             updates=upd,
                             dimension_numbers=lax.ScatterDimensionNumbers(*dimension_numbers),
                             indices_are_sorted=False,
                             unique_indices=True)

    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype),
                       tf.TensorSpec([None, 2], dtype=upd.dtype)],
      polymorphic_shapes=["b, _", "b, _"],
      expected_output_signature=tf.TensorSpec([None, 4]))

    self.assertAllClose(f_jax(x, upd), f_tf(x, upd))

  def test_select(self):
    def f_jax(x):  # x.shape = (b, 3)
      return lax.select(x > 5., x, x)
    x = np.arange(100, dtype=np.float32).reshape((10, 10))[:7, :3]
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 3], dtype=x.dtype)],
      polymorphic_shapes=["(b, _)"],
      expected_output_signature=tf.TensorSpec([None, 3]))

    self.assertAllClose(f_jax(x), f_tf(x))

  def test_select_vmap_0(self):
    @jax.vmap
    def f_jax(x):  # x.shape = (3,)
      return lax.select(x > 5., x, x)
    x = np.arange(100, dtype=np.float32).reshape((10, 10))[:7, :3]
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 3], dtype=x.dtype)],
      polymorphic_shapes=["(b, _)"],
      expected_output_signature=tf.TensorSpec([None, 3]))

    self.assertAllClose(f_jax(x), f_tf(x))

  def test_select_vmap_1(self):
    @functools.partial(jax.vmap, in_axes=(0, None))
    def f_jax(x, y):  # x.shape = (3,)
      return lax.select(x > 5., x, y)
    x = np.arange(100, dtype=np.float32).reshape((10, 10))[:7, :3]
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 3], dtype=x.dtype),
                       tf.TensorSpec([3], dtype=x.dtype)],
      polymorphic_shapes=["(b, _)", "_"],
      expected_output_signature=tf.TensorSpec([None, 3]))

    self.assertAllClose(f_jax(x, x[0]), f_tf(x, x[0]))

  def test_slice(self):
    def f_jax(x):  # x.shape = (b, 3)
      return lax.slice(x, start_indices=(0, 1),
                       limit_indices=(x.shape[0], 3))
    x = np.arange(100, dtype=np.float32).reshape((10, 10))[:7, :3]
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 3], dtype=x.dtype)],
      polymorphic_shapes=["(b, _)"],
      expected_output_signature=tf.TensorSpec([None, 2]))

    self.assertAllClose(f_jax(x), f_tf(x))

  def test_squeeze(self):
    def f_jax(x):
      return jnp.squeeze(x, axis=1)
    x = np.random.rand(4, 1)
    res_jax = f_jax(x)

    # Trace with a known dimension to squeeze
    f_tf = self.CheckShapePolymorphism(
      f_jax,
      input_signature=[tf.TensorSpec([None, 1], dtype=x.dtype)],
      polymorphic_shapes=["(b, _)"],
      expected_output_signature=tf.TensorSpec([None]))

    self.assertAllClose(res_jax, f_tf(x))

    with self.assertRaisesRegex(
        core.InconclusiveDimensionOperation,
        re.escape("Shape variable comparison b2 == 1 is inconclusive")):
      # Trace with unknown dimension to squeeze
      self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([None, None])],
        polymorphic_shapes=["(b1, b2)"],
        expected_output_signature=tf.TensorSpec([None]))



if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
