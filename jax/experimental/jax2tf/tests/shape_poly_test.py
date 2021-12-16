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
"""Tests for the shape-polymorphic jax2tf conversion."""

from absl.testing import absltest, parameterized
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import collections
import functools
from functools import partial
import operator
import re

import jax
from jax import core
from jax.experimental import jax2tf
from jax.experimental.jax2tf import shape_poly
from jax import lax
from jax import linear_util as lu
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src.lax import control_flow as lax_control_flow
from jax._src import util
import numpy as np

from jax.experimental.jax2tf.tests import tf_test_util

import tensorflow as tf  # type: ignore[import]

from jax.config import config

config.parse_flags_with_absl()

# Import after parsing flags
from jax.experimental.jax2tf.tests import primitive_harness
from jax.experimental.jax2tf.tests.primitive_harness import Harness, CustomArg, RandArg, StaticArg
from jax.experimental.jax2tf.tests.jax2tf_limitations import Jax2TfLimitation

PS = jax2tf.PolyShape


class DimPolynomialTest(tf_test_util.JaxToTfTestCase):

  def test_parse_poly_spec(self):
    self.assertEqual((2, 3), shape_poly._parse_spec(None, (2, 3)))
    self.assertEqual((2, 3), shape_poly._parse_spec("2, 3", (2, 3)))
    self.assertEqual((2, 3), shape_poly._parse_spec("2, _", (2, 3)))
    self.assertEqual((2, 3), shape_poly._parse_spec("2, ...", (2, 3)))
    self.assertEqual((2, 3), shape_poly._parse_spec("...", (2, 3)))
    self.assertEqual((2, 3), shape_poly._parse_spec(" ( 2 , 3 ) ", (2, 3)))

    a, b = shape_poly._parse_spec("a, b", (2, 3))
    self.assertEqual((a, 3), shape_poly._parse_spec("(a, ...) ", (None, 3)))
    tshape = tf.TensorShape([None, 3])
    self.assertEqual((a, 3), shape_poly._parse_spec("(a, ...) ", tshape))

  a, b = shape_poly._parse_spec("a, b", (2, 3))
  @parameterized.named_parameters(
      dict(testcase_name=f"_dim_spec={dim_spec}",
           dim_spec=dim_spec, dim_poly=dim_poly)
      for dim_spec, dim_poly in [
          ("2*a*b", 2 * a * b),
          ("-2 * a^2 * b + b^2", -2 * a * a * b + b * b),
          ("-2 * a^2 * b + -1 *b^2*a", -2 * a * a * b - a * b * b),
          ("3 * a * b * a + -2", 3 * a * b * a - 2),
          ("a + 1", a + 1),
          ("a + -1", a - 1),
  ])
  def test_parse_poly_spec_poly(self,
                                dim_spec="3 * a * b * a + -2",
                                dim_poly=3 * a * b * a - 2):
    # For internal usage only (the polymorphic_shapes of VJP) we need to
    # parse polynomials.
    self.assertEqual((dim_poly,), shape_poly._parse_spec(dim_spec, (2,)))
    self.assertEqual((dim_poly,), shape_poly._parse_spec(str(dim_poly), (2,)))

  @parameterized.named_parameters(
      dict(testcase_name=f"_dim_spec={dim_spec}",
           dim_spec=dim_spec, dim_poly=dim_poly)
      for dim_spec, dim_poly in [
          ("2*a*b", 2 * a * b),
          ("-2 * a^2 * b + b^2", -2 * a * a * b + b * b),
          ("-2 * a^2 * b + -1 *b^2*a", -2 * a * a * b - a * b * b),
          ("3 * a * b * a + -2", 3 * a * b * a - 2),
          ("a + 1", a + 1),
          ("a + -1", a - 1),
  ])
  def test_parse_poly_spec_shapeenv(self,
                                dim_spec="3 * a * b * a + -2",
                                dim_poly=3 * a * b * a - 2):
    # For internal usage only (the polymorphic_shapes of VJP) we need to
    # parse polynomials.
    self.assertEqual((dim_poly,), shape_poly._parse_spec(dim_spec, (2,)))
    self.assertEqual((dim_poly,), shape_poly._parse_spec(str(dim_poly), (2,)))

  def test_dim_vars(self):
    a, b, a1 = shape_poly._parse_spec("a, b, a", (2, 3, 2))
    self.assertEqual(True, a == a)
    self.assertEqual(True, a == a1)
    self.assertEqual(False, a != a)
    with self.assertRaisesRegex(
        core.InconclusiveDimensionOperation,
        "Dimension polynomial comparison 'a' == 'b' is inconclusive"):
      a.eq(b)

    with self.assertRaisesRegex(
        core.InconclusiveDimensionOperation,
        "Dimension polynomial comparison 'a' == 'b' is inconclusive"):
      a == b

    with self.assertRaisesRegex(
        core.InconclusiveDimensionOperation,
        "Dimension polynomial comparison 'a' == 'b' is inconclusive"):
      a != b

    self.assertLen({a, a}, 1)
    self.assertLen({a, b}, 2)
    self.assertIn(a, {a, b})
    self.assertIn(b, {a, b})
    self.assertIn(a, [a, b])
    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                "Dimension polynomial comparison .* is inconclusive"):
      b in [a, b]

  def test_get_vars(self):
    a, b = shape_poly._parse_spec("a, b", (2, 3))

    self.assertEqual({"a"}, a.get_vars())
    self.assertEqual({"a", "b"}, (a * b * a).get_vars())

  def test_evaluate(self):
    a, b = shape_poly._parse_spec("a, b", (2, 3))

    self.assertEqual(1, (a * a - b).evaluate(dict(a=2, b=3)))
    self.assertEqual(2, (a * a - b + 1).evaluate(dict(a=-2, b=3)))

  def test_dim_vars_symbolic_equal(self):
    a, b = shape_poly._parse_spec("a, b", (2, 3))
    self.assertTrue(core.symbolic_equal_dim(a, a))
    self.assertFalse(core.symbolic_equal_dim(a, 1))
    self.assertFalse(core.symbolic_equal_dim(a, b))

    self.assertTrue(core.symbolic_equal_one_of_dim(a, [2, a]))
    self.assertFalse(core.symbolic_equal_one_of_dim(a, [2, b]))
    self.assertFalse(core.symbolic_equal_one_of_dim(a, []))

    self.assertTrue(core.symbolic_equal_one_of_dim(2, [a, 3, 2]))
    self.assertFalse(core.symbolic_equal_one_of_dim(1, [2, b]))
    self.assertFalse(core.symbolic_equal_one_of_dim(3, []))

    self.assertTrue(core.symbolic_equal_dim(1, jnp.add(0, 1)))  # A DeviceArray
    with self.assertRaisesRegex(TypeError,
                                re.escape("Shapes must be 1D sequences of concrete values of integer type, got (1, 'a').")):
      self.assertTrue(core.symbolic_equal_dim(1, "a"))

  def test_poly_bounds(self):
    a, b = shape_poly._parse_spec("a, b", (2, 3))
    self.assertEqual(a.bounds(), (1, None))
    self.assertEqual((2 * a).bounds(), (2, None))
    self.assertEqual((2 * a - 3).bounds(), (-1, None))
    self.assertEqual((-2 * a - 3).bounds(), (None, -5))
    self.assertEqual((3 * a * b * b + 5 * a - 7).bounds(), (1, None))
    self.assertEqual((3 * a * b * b - 5 * a - 7).bounds(), (None, None))
    self.assertEqual((a + b - a * b + a * b * a).bounds(), (None, None))
    self.assertEqual((a + 2 * b - a).bounds(), (2, None))

  def test_poly_equal(self):
    a, b = shape_poly._parse_spec("a, b", (2, 3))
    poly3 = a + 3 - a
    self.assertTrue(poly3 == 3)
    self.assertTrue(poly3 == np.array(3, np.int64))
    self.assertTrue(poly3 == np.array(3, np.int64)[()])
    self.assertFalse((poly3 + 1) == 3)
    self.assertFalse(poly3 == poly3 + 1)
    self.assertTrue((2 * a * b * a + 3).eq(1 + b * a * a + a * a * b + 2))
    self.assertFalse((2 * a * b * a + 3).eq(a * b * a + 3))

    self.assertFalse((a * b * a + 3).eq(a * b * a + 4))
    self.assertFalse((2 * a * b * a).eq(a * b * a))
    self.assertFalse((2 * a * b * a + 1).eq(a * b * a))
    self.assertFalse((3 * a * b * a - 1).eq(a * b * a))
    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                re.escape("Dimension polynomial comparison '3*a^2*b + -2' == 'a^2*b' is inconclusive")):
      (3 * a * b * a - 2).eq(a * b * a)

  def test_poly_compare(self):
    a, b = shape_poly._parse_spec("a, b", (2, 3))
    poly = 4 * a + b + 3
    self.assertTrue(poly.ge(0))
    self.assertTrue(poly.ge(8))
    self.assertTrue(poly.ge(poly))
    self.assertTrue(poly.ge(poly - 1))

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation, "inconclusive"):
      poly.ge(9)

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation, "inconclusive"):
      (4 * a - b).ge(0)

  def test_poly_compare_overload(self):
    a, b = shape_poly._parse_spec("a, b", (2, 3))
    poly = 4 * a + b + 3
    self.assertTrue(poly >= 0)
    self.assertTrue(poly >= 8)
    self.assertTrue(poly > 7)
    self.assertTrue(poly >= poly)
    self.assertTrue(poly >= poly - 1)

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation, "inconclusive"):
      poly >= 9

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation, "inconclusive"):
      (4 * a - b) >= 0

  def test_core_greater_equal(self):
    a, b = shape_poly._parse_spec("a, b", (2, 3))
    self.assertTrue(core.greater_equal_dim(a, a))
    self.assertTrue(core.greater_equal_dim(a, 0))
    self.assertTrue(core.greater_equal_dim(a, 1))

    self.assertTrue(core.greater_equal_shape((a, 2), (1, 1)))

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                "Dimension polynomial comparison .* is inconclusive"):
      core.greater_equal_dim(a, 2)

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                "Dimension polynomial comparison .* is inconclusive"):
      core.greater_equal_dim(a, b)

  def test_poly_int_results(self):
    a, b = shape_poly._parse_spec("a, b", (2, 3))
    self.assertEqual(a + 2 - a, 2)
    self.assertIsInstance(a + 2 - a, int)
    self.assertEqual(a + (2 - a), 2)
    self.assertIsInstance(a + (2 - a), int)
    self.assertEqual(a * 2 // a, 2)
    self.assertIsInstance(a * 2 // a, int)

  @parameterized.named_parameters(
      dict(testcase_name=f"_D={dividend}_d={divisor}_q={quotient}_r={remainder}",
           dividend=dividend, divisor=divisor, quotient=quotient,
           remainder=remainder)
      for dividend, divisor, quotient, remainder in [
          (a, 1, a, 0),
          (3 * a, 3, a, 0),
          (3 * a + 3, 3, a + 1, 0),
          (3 * a + 2, 3, a, 2),
          (3 * a + 5, 3, a + 1, 2),
          (3 * a - 2, 3, a - 1, 1),
          (3 * a * a * b + 2 * b * b * a, a * b, 3 * a + 2 * b, 0),
          (a * a - b * b, a + b, a - b, 0),
          (a, b, None, None),
          (3 * a, 2, None, None),
          (2 * a * b + b * b, a + b, None, None),
          (3, a, None, None),
  ])
  def test_poly_divmod(self, *, dividend, quotient, divisor, remainder):
    if quotient is None:
      with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                  "Dimension polynomial .* is not a multiple of .*"):
        divmod(dividend, divisor)
    else:
      self.assertEqual((quotient, remainder), divmod(dividend, divisor))

  @parameterized.named_parameters(
      dict(testcase_name=f"_D={dividend}_d={divisor}_q={quotient}",
           dividend=dividend, divisor=divisor, quotient=quotient)
      for dividend, divisor, quotient in [
          (a, 1, a),
          (3 * a, 3, a),
          (3 * a + 3, 3, a + 1),
          (3 * a + 2, 3, None),
          (3 * a + 5, 3, None),
          (3 * a - 2, 3, None),
          (3 * a * a * b + 2 * b * b * a, a * b, 3 * a + 2 * b),
          (a * a - b * b, a + b, a - b),
          (a, b, None),
          (3 * a, 2, None),
          (2 * a * b + b * b, a + b, None),
  ])
  def test_poly_truediv(self, *, dividend, divisor, quotient):
    if quotient is None:
      with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                  "Dimension polynomial .* is not a multiple of .*"):
        dividend / divisor
    else:
      self.assertEqual(quotient, dividend / divisor)

  def test_poly_truediv_error(self):
    a, = shape_poly._parse_spec("a,", (2,))
    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                "Division of '3' by dimension polynomial .* is not supported"):
      3 / a

  def test_dilate_shape(self):
    """0 if d == 0 else 1 + dilation * (d - 1))"""
    a, = shape_poly._parse_spec("a,", (2,))

    self.assertEqual((4, 7), core.dilate_shape((2, 3), (3, 3)))
    self.assertEqual((0, 7), core.dilate_shape((0, 3), (3, 3)))
    self.assertEqual((a, 7), core.dilate_shape((a, 3), (1, 3)))
    self.assertEqual((2 * a - 1, 7), core.dilate_shape((a, 3), (2, 3)))

  def test_stride_shape(self):
    """(s - window_size) // window_stride + 1"""
    a, stride = shape_poly._parse_spec("a, s", (2, 3))

    self.assertEqual((8, 9), core.stride_shape((10, 20), window_size=(3, 3), window_stride=(1, 2)))
    self.assertEqual((a, 9), core.stride_shape((a, 20), (1, 3), (1, 2)))

    self.assertEqual((a - 1, 9), core.stride_shape((a, 20), (2, 3), (1, 2)))
    self.assertEqual((a + 1, 9), core.stride_shape((a * stride + 2, 20), (2, 3), (stride, 2)))

    with self.assertRaisesRegex(
        core.InconclusiveDimensionOperation,
        re.escape(
          "Cannot compute stride for dimension 'a', window_size '1', stride '2'. Reason: Dimension polynomial 'a + -1' is not a multiple of '2'")):
      core.stride_shape((a, 20), (1, 3), (2, 2))


class ShapePolyTest(tf_test_util.JaxToTfTestCase):

  def test_simple_unary(self):
    """Test shape polymorphism for a simple case, unary function."""

    def f_jax(x):
      return x + jnp.sin(x)

    self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([2, 3])],
        polymorphic_shapes=None,
        expected_output_signature=tf.TensorSpec([2, 3]))

    self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([2, None])],
        polymorphic_shapes=["_, h"],
        expected_output_signature=tf.TensorSpec([2, None]))

    self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([None, None])],
        polymorphic_shapes=["h, h"],
        expected_output_signature=tf.TensorSpec([None, None]))

    self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([None, None])],
        polymorphic_shapes="h, h",
        expected_output_signature=tf.TensorSpec([None, None]))

  def test_simple_binary(self):
    """Test shape polymorphism for a simple case, binary function."""

    def f_jax(x, y):
      return x + jnp.sin(y)

    self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([2, 3]), tf.TensorSpec([2, 3])],
        polymorphic_shapes=None,
        expected_output_signature=tf.TensorSpec([2, 3]))

    self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([2, None]), tf.TensorSpec([2, 3])],
        polymorphic_shapes="_, h",
        expected_output_signature=tf.TensorSpec([2, 3]))

    self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([None, None]), tf.TensorSpec([None, None])],
        polymorphic_shapes=PS("h", "h"),
        expected_output_signature=tf.TensorSpec([None, None]))

  def test_forgot_polymorphic_shapes_error(self):
    msg_re = "polymorphic shape None in axis .* must contain a dimension variable for unknown dimension in argument shape .*. Perhaps you forgot to add the polymorphic_shapes"
    with self.assertRaisesRegex(ValueError, msg_re):
      self.CheckShapePolymorphism(
          jnp.sin,
          input_signature=[tf.TensorSpec([1, None])],
          polymorphic_shapes=None)

  def test_kwargs(self):
    """Test shape polymorphism for a function with kwargs."""

    x = np.ones(3, dtype=np.float32)
    y = np.ones(1, dtype=np.float32)
    def f_jax(x, *, y):
      return x + jnp.sin(y)

    f_tf = jax2tf.convert(f_jax, polymorphic_shapes=["b, ..."])
    f_tf(x, y=y)

  def test_arg_avals(self):
    """Test conversion of actual arguments to abstract values."""

    def check_avals(*, arg_shapes: Sequence[Sequence[Optional[int]]],
                    polymorphic_shapes: Sequence[Optional[Union[str, PS]]],
                    expected_avals: Optional[Sequence[core.ShapedArray]] = None,
                    expected_shapeenv: Optional[Dict[str, int]] = None,
                    eager_mode: bool = False):
      # Use eager mode only for when all arg_shapes are known, in order to
      # check expected_shapeenv.
      arg_dtypes = (_f32,) * len(arg_shapes)
      def f_tf(*tf_args):
        avals = shape_poly.args_avals(
            arg_shapes, arg_dtypes, polymorphic_shapes)  # The function under test
        dim_vars, get_dim_values = shape_poly.prepare_dim_var_env(avals)
        dim_values, _ = util.unzip2(jax2tf.jax2tf._interpret_fun(lu.wrap_init(get_dim_values),
                                                                 tf_args, avals, ""))
        if expected_avals is not None:
          self.assertEqual(expected_avals, avals)
        return dict(zip(dim_vars, dim_values))
      if eager_mode:
        # If we want to check the shape_env then all arg_shapes must be known
        assert all(all(d is not None for d in a_s)
                   for a_s in arg_shapes)
        shape_env = f_tf(*[tf.ones(a_s, dtype=_f32) for a_s in arg_shapes])
        if expected_shapeenv is not None:
          for v, val in expected_shapeenv.items():
            self.assertEqual(val, shape_env.get(v))
      else:
        f_tf = tf.function(autograph=False)(f_tf)
        f_tf.get_concrete_function(*[tf.TensorSpec(a_s, _f32)
                                   for a_s in arg_shapes])
        assert not expected_shapeenv, "Should use eager_mode=True"

    def shaped_array(shape_spec: str, actual_shape: core.Shape):
      return core.ShapedArray(
          shape_poly._parse_spec(shape_spec, actual_shape), np.float32)

    # Known shapes for the arguments
    check_avals(
        arg_shapes=[(2, 3)],
        polymorphic_shapes=[None],
        expected_avals=(shaped_array("2, 3", [2, 3]),))

    check_avals(
        arg_shapes=[(2, 3)],
        polymorphic_shapes=["(2, 3)"],
        expected_avals=(shaped_array("2, 3", [2, 3]),))

    check_avals(
        arg_shapes=[(2, 3)],
        polymorphic_shapes=["(_, 3)"],
        expected_avals=(shaped_array("2, 3", [2, 3]),))

    check_avals(
        arg_shapes=[(2, 3)],
        polymorphic_shapes=[PS("_", 3)],
        expected_avals=(shaped_array("2, 3", [2, 3]),))

    check_avals(
        arg_shapes=[(2, 3)],
        polymorphic_shapes=["..."],
        expected_avals=(shaped_array("2, 3", [2, 3]),))

    check_avals(
        arg_shapes=[(2, 3)],
        polymorphic_shapes=[PS(...)],
        expected_avals=(shaped_array("2, 3", [2, 3]),))

    # Partially known shapes for the arguments
    check_avals(
        arg_shapes=[(None, 3)],
        polymorphic_shapes=[PS("b", ...)],
        expected_avals=(shaped_array("(b, 3)", (2, 3)),))

    check_avals(
        arg_shapes=[(None, None)],
        polymorphic_shapes=["h, h"],
        expected_avals=(shaped_array("(h, h)", (2, 2)),))

    check_avals(
        arg_shapes=[(2, None)],
        polymorphic_shapes=["h, h"],
        expected_avals=(shaped_array("(h, h)", (2, 2)),))

    check_avals(
        arg_shapes=[(None, 3, 4)],
        polymorphic_shapes=["(c, b, a)"],
        expected_avals=(shaped_array("(c, b, a)", (2, 3, 4)),),
    )

    # Check cases when the specifications are polynomials
    check_avals(
        arg_shapes=[(2, 3)],
        polymorphic_shapes=[PS("a + 1", "b + 2")],
        eager_mode=True,
        expected_shapeenv=dict(a=1, b=1))

    check_avals(
        arg_shapes=[(7, 5)],
        polymorphic_shapes=[PS("2 * a + b", "b + 2")],
        eager_mode=True,
        expected_shapeenv=dict(a=2, b=3))

    check_avals(
        arg_shapes=[(7, 11, 4)],
        polymorphic_shapes=[PS("2 * a + b", "b * b + 2", "b + 1")],
        eager_mode=True,
        expected_shapeenv=dict(a=2, b=3))

    check_avals(
        arg_shapes=[(7, 11, 19, 7)],
        polymorphic_shapes=[PS("2 * a + b", "b * b + 2", "b + c * c", "2 * c + -1")],
        eager_mode=True,
        expected_shapeenv=dict(a=2, b=3, c=4))

    with self.assertRaisesRegex(ValueError,
                                "Cannot solve for values of dimension variables {'b'}"):
      check_avals(
          arg_shapes=[(4, 36, 3)],
          polymorphic_shapes=[PS("b * b", "b * d * d", "d")])

    with self.assertRaisesRegex(ValueError,
                                "Dimension variable b must have integer value >= 1"):
      check_avals(
          arg_shapes=[(5, 36)],
          polymorphic_shapes=[PS("3 * b", ...)],
          eager_mode=True)

    with self.assertRaisesRegex(ValueError,
                                "Dimension variable b must have integer value >= 1"):
      check_avals(
          arg_shapes=[(10, 3)],
          polymorphic_shapes=[PS("3 * b + 10", ...)],
          eager_mode=True)

    with self.assertRaisesRegex(ValueError,
                                "Dimension variable b must have integer value >= 1"):
      check_avals(
          arg_shapes=[(7, 3)],
          polymorphic_shapes=[PS("3 * b + 10", ...)],
          eager_mode=True)


    for invalid_syntax in [")(", "2a", "a@", "a - 2", "'a'", "('a', ...)"]:
      with self.assertRaisesRegex(ValueError,
                                  re.escape("has invalid syntax")):
        check_avals(
            arg_shapes=[(2,)], polymorphic_shapes=[invalid_syntax])

    for invalid_syntax in [5.0, ["a list"], ("a tuple",), re.compile(".")]:
      with self.assertRaisesRegex(ValueError,
                                  re.escape("Invalid polymorphic shape element")):
        check_avals(
            arg_shapes=[(2,)], polymorphic_shapes=[PS([invalid_syntax])])

    with self.assertRaisesRegex(
        ValueError,
        re.escape("polymorphic shape '..., 3' can contain Ellipsis only at the end.")):
      check_avals(
          arg_shapes=[(2, 3)],
          polymorphic_shapes=["..., 3"])

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "polymorphic shape '2, 3, 4, ...' of rank 3 must match the rank 2 of argument shape (2, 3).")
    ):
      check_avals(
          arg_shapes=[(2, 3)],
          polymorphic_shapes=["2, 3, 4, ..."])

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "polymorphic shape (Ellipsis, 3) can contain Ellipsis only at the end.")):
      check_avals(
          arg_shapes=[(2, 3)],
          polymorphic_shapes=[PS(..., 3)])

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "polymorphic shape None in axis 1 must contain a dimension variable for unknown dimension in argument shape (2, None)"
        )):
      check_avals(
          arg_shapes=[(2, None)],
          polymorphic_shapes=[None])

    with self.assertRaisesRegex(
        ValueError,
        re.escape("polymorphic shape '()' of rank 0 must match the rank 2 of argument shape (2, 3)")):
      check_avals(
          arg_shapes=[(2, 3)], polymorphic_shapes=["()"])

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "polymorphic shape '(_, _)' in axis 1 must contain a dimension variable "
            "for unknown dimension in argument shape (2, None)"
        )):
      check_avals(
          arg_shapes=[(2, None)],
          polymorphic_shapes=["(_, _)"])

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "polymorphic shape '(2, 13)' in axis 1 must match the known dimension size 3 "
            "for argument shape (2, 3)"
        )):
      check_avals(
          arg_shapes=[(2, 3)],
          polymorphic_shapes=["(2, 13)"])

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "polymorphic shape '(2, 3)' in axis 1 must contain a dimension variable for "
            "unknown dimension in argument shape (2, None)"
        )):
      check_avals(
          arg_shapes=[(2, None)],
          polymorphic_shapes=["(2, 3)"])

    with self.assertRaisesRegex(
        ValueError,
        "Found inconsistency when solving.*"):
      check_avals(
          arg_shapes=[(2, 3)],
          polymorphic_shapes=["(a, a)"],
          eager_mode=True)

    # Same error across multiple arguments
    with self.assertRaisesRegex(
        ValueError,
        "Found inconsistency when solving.*"):
      check_avals(
          arg_shapes=[(2, 3), (5,)],
          polymorphic_shapes=["a, ...", "a"],
          eager_mode=True)

  def test_pytree(self):
    """Arguments and polymorphic_shapes are pytrees."""

    # Arguments are of the form [([x00, x01], [x10]), dict(a=ya, b=yb)]
    def add_all_jax(x_pair_of_list, y_dict):
      x_list_0, x_list_1 = x_pair_of_list
      return functools.reduce(operator.add,
                              x_list_0 + x_list_1 + [y_dict["a"], y_dict["b"]])

    self.CheckShapePolymorphism(
        add_all_jax,
        input_signature=[([tf.TensorSpec([None]),
                           tf.TensorSpec([None])], [tf.TensorSpec([None])]),
                         dict(a=tf.TensorSpec([None]),
                              b=tf.TensorSpec([None]))],
        polymorphic_shapes=[(["v", "v"], [("v")]),
                            dict(a="v", b="v")],
        expected_output_signature=tf.TensorSpec([None]))

    # Now partial polymorphic_shapes; the parts of the polymorphic_shapes that
    # are not specified must have full input_signatures.
    self.CheckShapePolymorphism(
        add_all_jax,
        input_signature=[([tf.TensorSpec([4]),
                           tf.TensorSpec([4])], [tf.TensorSpec([4])]),
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
      # res: [b1, b2, d1, d1]
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
       tf_value_and_grad, autograph=False).get_concrete_function(
           tf.TensorSpec([3, 4, 8, 9]))

    self.assertEqual((3, 4, 8, 8), tuple(tf_grad.output_shapes[0]))
    self.assertEqual((3, 4, 8, 9), tuple(tf_grad.output_shapes[1]))

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
    tf_grad = tf.function(
        tf_value_and_grad,
        autograph=False).get_concrete_function(tf.TensorSpec([None, 3, 4]))
    # The shape of the value
    self.assertEqual((None, 3, 4), tuple(tf_grad.output_shapes[0]["res"]))
    # The shape of the gradient should match the input
    self.assertEqual((None, 3, 4), tuple(tf_grad.output_shapes[1]["grad"]))

  def test_grad_not_var_output(self):
    # Output of the function has poly shapes, non-variable
    def f_jax(x):  # :[b, 3]
      return jnp.reshape(x, (-1,))  # : [3b]
    x = np.arange(12, dtype=np.float32).reshape((4, 3))
    xv = tf.Variable(x)

    f_tf = jax2tf.convert(f_jax, with_gradient=True,
                          polymorphic_shapes=["b, ..."])

    with tf.GradientTape() as tape:
      res_tf = f_tf(xv)
    grad_tf = tape.gradient(res_tf, xv)
    self.assertAllClose(np.ones(x.shape, dtype=np.float32), grad_tf.numpy())


  def test_cond(self):
    # Test the primitive under conditional
    def f(x, y):
      # x: f32[B, H], y : f32[H]
      return lax.cond(
          jnp.sum(x) > 0.,
          lambda _: x + y,
          lambda _: jnp.zeros_like(x),
          operand=None)

    x = np.ones((2, 3))
    y = np.ones((3,))
    res_jax = f(x, y)
    self.assertAllClose(
        res_jax,
        jax2tf.convert(f, polymorphic_shapes=["(b, h)", "h"])(x, y))

  @parameterized.named_parameters(jtu.cases_from_list(
      dict(testcase_name=f"function={with_function}",
           with_function=with_function)
      for with_function in [False, True]))
  def test_grad_int(self, with_function=True):
    # https://github.com/google/jax/issues/7093
    # Also issue #6975.
    x_shape = (2, 3, 4)
    xi = np.arange(np.prod(x_shape), dtype=np.int16).reshape(x_shape)
    yf = xi.astype(np.float32)
    xi_yf = (xi, yf)
    zb = np.array([True, False], dtype=np.bool_)
    def f_jax(xi_yf, zb):  # xi: s16[2, 3, 4], yf: f32[2, 3, 4], zb: bool[2]
      xi, yf = xi_yf
      # Return a tuple:
      #   (1) float constant, with 0 tangent;
      #   (2) a tuple with:
      #     (2.1) the integer input;
      #     (2.2) the boolean input;
      #     (2.3) a float depending on both inputs.
      # TODO: there is a problem if we add a None output
      return (jnp.zeros(xi.shape, dtype=jnp.float32),
              (xi, zb, xi.astype(np.float32) * 2. * yf))

    args = (xi_yf, zb)

    f_tf = jax2tf.convert(f_jax, polymorphic_shapes=[("b1, b2, 4", "b1, b2, 4"), "b1"])
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)

    res_tf, g_tf = tf_test_util.ComputeTfValueAndGrad(f_tf, args)
    self.assertAllClose(g_tf[0][0], np.zeros_like(xi))
    self.assertAllClose(g_tf[0][1], (xi * 2).astype(yf.dtype))
    self.assertAllClose(g_tf[1], np.zeros_like(zb))

  def test_saved_model(self):
    f_jax = jnp.sin
    f_tf = jax2tf.convert(f_jax, polymorphic_shapes=["(b, ...)"])
    x = np.array([0.7, 0.8], dtype=np.float32)
    restored_f, _ = tf_test_util.SaveAndLoadFunction(
        f_tf, input_signature=[tf.TensorSpec([None], x.dtype)])
    self.assertAllClose(f_jax(x), restored_f(x))
    # Ensure that restored_f works at other batch size as well
    y = np.concatenate([x, x])
    self.assertAllClose(f_jax(y), restored_f(y))

  def test_saved_model_int_function(self):
    def f_jax(x):  # x:s32[b, 3, 4]
      return jnp.reshape(x, (-1,))  # : s32[b * 12]
    f_tf = jax2tf.convert(f_jax, polymorphic_shapes=["(b, ...)"])
    f_tf = tf.function(f_tf, autograph=False)
    x_shape = (2, 3, 4)
    x = np.arange(np.prod(x_shape), dtype=np.int32).reshape(x_shape)

    # When saving the model with gradients, we trace the gradient function
    # and we used to get an error when creating zeros_like_aval for a
    # polymorphic shape
    restored_f, _ = tf_test_util.SaveAndLoadFunction(
        f_tf, input_signature=[tf.TensorSpec((None,) + x.shape[1:], x.dtype)])
    f_jax_rt = jax2tf.call_tf(restored_f)
    res_jax_rt = f_jax_rt(x)
    self.assertAllClose(f_jax(x), res_jax_rt)

  def test_saved_model_constant_gradient(self):
    def f_jax(x):  # A function whose gradient is a constant
      return 3.

    f_tf = jax2tf.convert(f_jax, polymorphic_shapes=["(b, ...)"])
    x = np.array([0.7, 0.8], dtype=np.float32)
    restored_f, _ = tf_test_util.SaveAndLoadFunction(
        f_tf, input_signature=[tf.TensorSpec([None], x.dtype)])
    self.assertAllClose(3., restored_f(x))
    self.assertAllClose(np.array([0., 0.], dtype=np.float32), jax.grad(f_jax)(x))

  def test_readme_example(self):
    """Some of the examples from the README."""
    def image_mask_jax(images, mask):
      # images: f32[B, W, W]  and mask: f32[W, W]
      return images * mask

    print(jax.make_jaxpr(image_mask_jax)(np.ones((1024, 28, 28)), np.ones((28, 28))))

    # will invoke broadcast_in_dim with shape=(1, w, w)
    jax2tf.convert(image_mask_jax, polymorphic_shapes=["(b, w, w)", "(w, w)"])

  def test_readme_shape_error(self):
    """Some of the examples from the README."""
    with self.assertRaisesRegex(
        TypeError,
        re.escape("add got incompatible shapes for broadcasting: (v,), (4,)")):
      self.CheckShapePolymorphism(
          lambda x, y: x + y,
          input_signature=[tf.TensorSpec([None]),
                           tf.TensorSpec([4])],
          polymorphic_shapes=["(v,)", "(4,)"],
          expected_output_signature=tf.TensorSpec([None]))

    four_ones = np.ones((4,))
    # We get the error even if we use correct actual arguments
    with self.assertRaisesRegex(
        TypeError,
        re.escape("add got incompatible shapes for broadcasting: (v,), (4,)")):
      jax2tf.convert(
          lambda x, y: x + y, polymorphic_shapes=["(v,)", "(4,)"])(four_ones,
                                                                   four_ones)

    with self.assertRaisesRegex(TypeError,
                                re.escape("dot_general requires contracting dimensions to have the same shape, got [4] and [v]")):
      jax2tf.convert(lambda x: jnp.matmul(x, x),
                     polymorphic_shapes=["(v, 4)"])(np.ones((4, 4)))

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                re.escape("Cannot divide evenly the sizes of shapes (b, 5, 7) and (2, -1)")):
      jax2tf.convert(lambda x: jnp.reshape(x, (2, -1)),
                     polymorphic_shapes=["(b, _, _)"])(np.ones((4, 5, 7)))

    jax2tf.convert(lambda x: jnp.reshape(x, (2, -1)),
                   polymorphic_shapes=["(b, _, _)"])(np.ones((4, 5, 6)))
    jax2tf.convert(lambda x: jnp.reshape(x, (-1, x.shape[0])),
                   polymorphic_shapes=["(b1, b2, ...)"])(np.ones((4, 5, 6)))

    with self.assertRaisesRegex(
        core.InconclusiveDimensionOperation,
        re.compile("Division of .* by dimension polynomial .* is not supported",
                   re.DOTALL)):
      jax2tf.convert(lambda x: jnp.sum(x, axis=0) / x.shape[0],
                     polymorphic_shapes=["(v, _)"])(np.ones((4, 4)))

    with self.assertRaisesRegex(
        core.InconclusiveDimensionOperation,
        re.escape("Dimension polynomial comparison 'a + 1' == 'b' is inconclusive")):
      jax2tf.convert(lambda x: 0 if x.shape[0] + 1 == x.shape[1] else 1,
                     polymorphic_shapes=["(a, b)"])(np.ones((4, 4)))

    # Unsoundness: not checking that the shape is 0
    def f_jax(x):
      return 0 if x.shape[0] == 0 else 1

    x0 = np.array([], np.float32)
    self.assertEqual(0, f_jax(x0))  # JAX sees that the x.shape[0] == 0

    # jax2tf catches the broken assumption b >= 1 if the converted function is executed
    # eagerly.
    # Raises: ValueError: PolyShape 'b' has dimension variable 'b' corresponding to 0, for argument shape (0,)
    with self.assertRaisesRegex(ValueError,
                                "Dimension variable b must have integer value >= 1. Found value 0 when solving .*"):
      jax2tf.convert(f_jax, polymorphic_shapes=["b"])(x0)

    # However, if we first trace to a TensorFlow graph, we may miss the broken assumption:
    f_tf = tf.function(
        jax2tf.convert(f_jax, polymorphic_shapes=["b"])).get_concrete_function(tf.TensorSpec([None], dtype=np.float32))
    self.assertEqual(1, f_tf(x0))

    # Unsoundness: not checking that the actual dimensions denoted by the same
    # dimension variables have equal sizes.
    def f_jax(x):
      return 0 if x.shape[0] != x.shape[1] else 1

    x45 = np.ones((4, 5), dtype=np.float32)
    self.assertEqual(0, f_jax(x45))  # JAX seems that x.shape[0] != x.shape[1]

    # jax2tf catches the broken assumption x.shape[0] == x.shape[1] if the converted
    # function is executed eagerly.
    # Raises: ValueError: PolyShape 'b, b' has dimension variable 'b' corresponding to multiple values ([4, 5]), for argument shape (4, 5)
    with self.assertRaisesRegex(ValueError,
                                "Found inconsistency when solving b == .*"):
      jax2tf.convert(f_jax, polymorphic_shapes=["b, b"])(x45)

    # However, if we first trace to a TensorFlow graph, we may miss the broken assumption.
    f_tf = tf.function(
        jax2tf.convert(f_jax, polymorphic_shapes=["b, b"])).get_concrete_function(tf.TensorSpec([None, None], dtype=np.float32))
    self.assertEqual(1, f_tf(x45))


class DimAsValueTest(tf_test_util.JaxToTfTestCase):

  def test_concrete_shapes(self):
    # Test dim_as_value with concrete shapes.
    def f(x):
      return jnp.sum(x, axis=0) * core.dimension_as_value(x.shape[0])

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


  def test_dynamic_shapes(self):
    # Test dim_as_value with dynamic shapes.
    def f(x):
      return jnp.sum(x, axis=0) * core.dimension_as_value(x.shape[0])

    x = np.arange(3.)
    self.assertAllClose(9., jax2tf.convert(f, polymorphic_shapes=["(b,)"])(x))
    self.assertAllClose(
        9.,
        jax2tf.convert(jax.jit(f), polymorphic_shapes=["(b,)"])(x))
    self.assertAllClose(
        9.,
        tf.function(jax2tf.convert(f, polymorphic_shapes=["(b,)"]))(x))

    res_primal, res_tangent = jax2tf.convert(
        lambda x, xt: jax.jvp(f, (x,), (xt,)),
        polymorphic_shapes=["b", "b"])(x, np.array([0.1, 0.2, 0.3]))
    self.assertAllClose((9., 1.8), (res_primal, res_tangent))

    self.assertAllClose(
        np.array([3., 3., 3.]),
        jax2tf.convert(jax.grad(f), polymorphic_shapes=["b"])(x))

    xv = np.arange(24.).reshape((2, 3, 4))
    res_vmap = jax.vmap(f, in_axes=1)(xv)
    # Implement by iteration
    res_iter = jnp.stack([f(xv[:, i, :]) for i in range(xv.shape[1])])
    self.assertAllClose(res_iter, res_vmap)

    res_vmap_tf = jax2tf.convert(jax.vmap(f, in_axes=1),
                                 polymorphic_shapes=["b1, b2, ..."])(xv)
    self.assertAllClose(res_iter, res_vmap_tf.numpy())

  def test_mean0(self):

    def f_jax(x):
      return jnp.sum(x, axis=0) / core.dimension_as_value(x.shape[0])

    x = np.arange(12.).reshape((3, 4))
    f_tf = self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype)],
        polymorphic_shapes=[("b, _")],
        expected_output_signature=tf.TensorSpec([4]))
    self.assertAllClose(np.array([4., 5., 6., 7.]), f_tf(x))

  def test_mean_all_axes(self):

    def f_jax(x):
      return jnp.sum(x) / core.dimension_as_value(np.prod(x.shape))

    x = np.arange(12.).reshape((3, 4))
    f_tf = self.CheckShapePolymorphism(
        f_jax,
        input_signature=[tf.TensorSpec([None, 4], dtype=x.dtype)],
        polymorphic_shapes=[("b, _")],
        expected_output_signature=tf.TensorSpec([]))

    self.assertAllClose(jnp.mean(x), f_tf(x))

  def test_errors(self):

    with self.assertRaisesRegex(
        TypeError,
        "Shapes must be 1D sequences of concrete values of integer type"):
      core.dimension_as_value(np.array([1, 2], dtype=np.int32))
    with self.assertRaisesRegex(
        TypeError,
        "Shapes must be 1D sequences of concrete values of integer type"):
      core.dimension_as_value(np.float32(1))

###
### We define primitive harnesses for which we will test shape-polymorphic
### conversion.
def _make_harness(group_name: str, name: str,
                  func: Callable,
                  args: primitive_harness.ArgDescriptor,
                  *,
                  poly_axes: Sequence[Optional[Union[int, Sequence[int]]]],
                  check_result=True,
                  skip_jax_run=True,
                  tol=None,
                  enable_and_diable_xla=False,
                  expect_error=(None, None),
                  **params) -> Union[Harness, Sequence[Harness]]:
  """The `poly_axes` must correspond to the non-static arguments, and for each
  one it must specify which axes are: None, or an int (for the index of the
  polymorphic axis), or a tuple of ints (for multiple polymorphic axes).

  For each argument, we use its `poly_axes` entry to generate the polymorphic_shapes
  specification, creating dimension variables `b0`, `b1, ..., for each of its
  polymorphic axes. This means that separate arguments will share the same
  dimension variable names, in the order in which the axes are listed in
  poly_axes.

  The name of the harness within the group will include `poly_axes`.
  You can add an additional `name`.

  `check_result` specifies if we want to check that the result of the shape
  polymorphic conversion produces the same result and the JAX function.

  `expect_error` is a pair of an Exception type and a regular expression to
  match the expected exception string.

  enable_and_diable_xla=True means that we generate two harnesses,
  one with enable_xla=False.
  """
  if enable_and_diable_xla:
    return [
        _make_harness(group_name, name + ("" if enable_xla else "_noxla"),  # type: ignore
                      func, args, poly_axes=poly_axes,
                      check_result=check_result, tol=tol, enable_xla=enable_xla,
                      enable_and_diable_xla=False, skip_jax_run=skip_jax_run,
                      expect_error=expect_error,
                      **params)
        for enable_xla in [True, False]
    ]
  poly_axes_name = f"poly_axes={repr(poly_axes)}"
  assert isinstance(poly_axes, Sequence)
  # Make poly_axes: Sequence[Sequence[int]]
  poly_axes = tuple(map(lambda pa: pa if isinstance(pa, Sequence) or pa is None else (pa,),
                        poly_axes))
  if name:
    name = f"{name}_{poly_axes_name}"
  else:
    name = poly_axes_name
  return Harness(group_name,
                 name,
                 func, args,
                 dtype=np.float32,
                 poly_axes=poly_axes, check_result=check_result,
                 skip_jax_run=skip_jax_run, expect_error=expect_error,
                 tol=tol,
                 **params)


_f32 = np.float32

# List containing either harnesses, or lists of harnesses
_POLY_SHAPE_TEST_HARNESSES = [
    _make_harness("add", "",
                  jnp.add,
                  [RandArg((3, 4), _f32), RandArg((2, 3, 4), _f32)],
                  poly_axes=[0, 1]),
    _make_harness("add_transpose", "",
                  jax.grad(lambda x: jnp.sum(jnp.sum(x, axis=0, keepdims=0) + x)),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("arange", "start",
                  lambda op: jnp.arange(2 * op.shape[0], dtype=_f32),
                  [RandArg((3,), _f32)],
                  poly_axes=[0],
                  enable_and_diable_xla=True),
    _make_harness("arange", "start_no_dtype",
                  lambda op: jnp.arange(op.shape[0]),
                  [RandArg((3,), _f32)],
                  poly_axes=[0]),
    _make_harness("arange", "error1",
                  lambda op: jnp.arange(op.shape[0], 10),
                  [RandArg((3,), _f32)],
                  poly_axes=[0],
                  expect_error=(ValueError, "jax.numpy.arange supports non-constant arguments only in single-argument form")),
    _make_harness("arange", "error2",
                  lambda op: jnp.arange(1, op.shape[0]),
                  [RandArg((3,), _f32)],
                  poly_axes=[0],
                  expect_error=(ValueError, "jax.numpy.arange supports non-constant arguments only in single-argument form")),
    _make_harness("arange", "error3",
                  lambda op: jnp.arange(1, 5, op.shape[0]),
                  [RandArg((3,), _f32)],
                  poly_axes=[0],
                  expect_error=(ValueError, "jax.numpy.arange supports non-constant arguments only in single-argument form")),
    # Reduce the poly dimension
    _make_harness("argmax", "0",
                  lambda op: lax.argmax(op, axis=0, index_dtype=np.int32),
                  [RandArg((3, 4, 5), _f32)],
                  poly_axes=[0],
                  enable_and_diable_xla=True),
    # Reduce the non-poly dimension
    _make_harness("argmax", "1",
                  lambda op: lax.argmax(op, axis=1, index_dtype=np.int32),
                  [RandArg((3, 4, 5), _f32)],
                  poly_axes=[0],
                  enable_and_diable_xla=True),
    [
        _make_harness("average",
                      f"axis={axis}_weights=None",
                      lambda x, axis: jnp.average(x, axis=axis, returned=False, weights=None),
                      [RandArg((7, 8, 4), _f32), StaticArg(axis)],
                      poly_axes=[0])
        for axis in [None, 0, 1]
    ],
    [
        _make_harness("average",
                      f"axis={axis}_weights=Some",
                      lambda x, weights, axis: jnp.average(x, axis=axis, returned=False, weights=weights),
                      [RandArg((7, 8, 4), _f32), RandArg((7, 8, 4), _f32), StaticArg(axis)],
                      poly_axes=[0, 0])
        for axis in [None, 0, 1]
    ],
    _make_harness("broadcast_to", "",
                  lambda x: jnp.broadcast_to(x, [x.shape[0], x.shape[0], 4]),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("broadcast_in_dim", "0",
                  lambda x: lax.broadcast_in_dim(x, [x.shape[0], 4, 5, 6],
                                                 broadcast_dimensions=(0, 2, 3)),
                  [RandArg((3, 1, 6), _f32)],
                  poly_axes=[0]),
    _make_harness("broadcast_in_dim", "poly",
                  lambda x: lax.broadcast_in_dim(x, [x.shape[0], x.shape[0] + x.shape[0], 4],
                                                 broadcast_dimensions=(0, 1, 2)),
                  [RandArg((3, 1, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("broadcast_in_dim", "poly2",
                  lambda x: lax.broadcast_in_dim(x, [x.shape[0], 5, 6, x.shape[2], 4],
                                                 broadcast_dimensions=(0, 2, 3)),
                  [RandArg((3, 1, 4), _f32)],
                  poly_axes=[(0, 2)]),
    _make_harness("broadcast_in_dim", "transpose",
                  jax.grad(lambda x: jnp.sum(lax.broadcast_in_dim(x, [2, x.shape[0], 5, x.shape[2], 4],
                                                 broadcast_dimensions=(1, 2, 3)))),
                  [RandArg((3, 1, 4), _f32)],
                  poly_axes=[(0, 2)]),
    _make_harness("clamp", "",
                  lax.clamp,
                  [RandArg((3, 4, 5), _f32), RandArg((3, 4, 5), _f32),
                   RandArg((3, 4, 5), _f32)],
                  poly_axes=[0, 0, 0]),
    _make_harness("collapse", "",
                  lambda x: lax.collapse(x, 1, 4),
                  [RandArg((3, 4, 5, 6, 7), _f32)],
                  poly_axes=[(0, 1, 3)]),
    _make_harness("conv_general_dilated", "",
                  lambda lhs, rhs: lax.conv_general_dilated(lhs, rhs,
                                                            window_strides=(2, 3),
                                                            padding=((0, 0), (0, 0)),
                                                            lhs_dilation=(1, 1),
                                                            rhs_dilation=(1, 2),
                                                            dimension_numbers=("NCHW", "OIHW", "NCHW"),
                                                            feature_group_count=1,
                                                            batch_group_count=1,
                                                            precision=None),
                  [RandArg((7, 3, 9, 10), _f32), RandArg((3, 3, 4, 5), _f32)],
                  poly_axes=[0, None]),
    _make_harness("cummax", "",
                  lambda x: lax_control_flow.cummax(x, axis=1, reverse=False),
                  [RandArg((3, 4, 5), _f32)],
                  poly_axes=[0]),
    _make_harness("delta", "0",
                  lambda x: lax._delta(_f32, x.shape, axes=(0, 1)),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("dot_general", "",
                  lambda lhs, rhs: lax.dot_general(lhs, rhs,
                                                   dimension_numbers=(((2,), (1,)), ((0,), (0,)))),
                  [RandArg((3, 4, 4), _f32), RandArg((3, 4), _f32)],
                  poly_axes=[0, 0]),
    _make_harness("dynamic_slice", "idx=tuple_int",
                  # x:shape: (b, 4)
                  lambda x: lax.dynamic_slice(x, (0, 1), (x.shape[0], 2)),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0],
                  enable_and_diable_xla=True),
    _make_harness("dynamic_slice", "idx=tuple_arg",
                  # x:shape: (b, 4)
                  lambda x, i0: lax.dynamic_slice(x, (i0, np.int32(1)), (x.shape[0], 2)),
                  [RandArg((3, 4), _f32), np.array(-2, dtype=np.int32)],
                  poly_axes=[0, None],
                  enable_and_diable_xla=True),
    _make_harness("dynamic_slice", "idx=array",
                  # x:shape: (b, 4)
                  lambda x, idx: lax.dynamic_slice(x, idx, (x.shape[0], 2)),
                  [RandArg((3, 4), _f32), np.array([-2, -1], dtype=np.int32)],
                  poly_axes=[0, None],
                  enable_and_diable_xla=True),
    _make_harness("dynamic_slice_in_dim", "idx=0",
                  # x:shape: (b, 4)
                  lambda x: lax.dynamic_slice_in_dim(x, 0, x.shape[0], axis=0),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0],
                  enable_and_diable_xla=True),
    _make_harness("dynamic_update_slice", "idx=tuple_int",
                  # x:shape: (b, 4)
                  lambda x: lax.dynamic_update_slice(x, x, (0, 0)),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0],
                  enable_and_diable_xla=True),
    _make_harness("dynamic_update_slice", "idx=tuple_arg",
                  # x:shape: (b, 4)
                  lambda x, i0: lax.dynamic_update_slice(x, x, (i0, np.int32(0))),
                  [RandArg((3, 4), _f32), np.array(-2, dtype=np.int32)],
                  poly_axes=[0, None],
                  enable_and_diable_xla=True),
    _make_harness("dynamic_update_slice", "idx=array",
                  # x:shape: (b, 4)
                  lambda x, idx: lax.dynamic_update_slice(x, x, idx),
                  [RandArg((3, 4), _f32), np.array([-2, -1], dtype=np.int32)],
                  poly_axes=[0, None],
                  enable_and_diable_xla=True),
    _make_harness("einsum", "0",
                  lambda x: jnp.einsum("...i->...", x),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("einsum", "0_alt",
                  lambda x: jnp.einsum(x, (..., 1), [...]),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("einsum", "1",
                  lambda x, y: jnp.einsum("...ij,...jk->...ik", x, y),
                  [RandArg((3, 4, 5), _f32), RandArg((3, 5, 6), _f32)],
                  poly_axes=[0, 0]),
    _make_harness("einsum", "1_alt",
                  lambda x, y: jnp.einsum(x, [..., 0, 1], y, (..., 1, 2), [..., 0, 2]),
                  [RandArg((3, 4, 5), _f32), RandArg((3, 5, 6), _f32)],
                  poly_axes=[0, 0]),
    _make_harness("einsum", "2",
                  lambda x, y: jnp.einsum("...ij,jk->...ik", x, y),
                  [RandArg((3, 4, 5), _f32), RandArg((5, 6), _f32)],
                  poly_axes=[0, None]),
    _make_harness("einsum", "2_alt",
                  lambda x, y: jnp.einsum(x, [..., 0, 1], y, [1, 2], [..., 0, 2]),
                  [RandArg((3, 4, 5), _f32), RandArg((5, 6), _f32)],
                  poly_axes=[0, None]),
    _make_harness("einsum", "3",
                  # Reduced dimension is polymorphic
                  lambda x, y: jnp.einsum("ij,jk->ik", x, y),
                  [RandArg((3, 4), _f32), RandArg((4, 5), _f32)],
                  poly_axes=[1, 0]),
    _make_harness("einsum", "3_alt",
                  # Reduced dimension is polymorphic
                  lambda x, y: jnp.einsum(x, [0, 1], y, [1, 2], [0, 2]),
                  [RandArg((3, 4), _f32), RandArg((4, 5), _f32)],
                  poly_axes=[1, 0]),
    _make_harness("einsum", "4",
                  # Reduced dimension is polymorphic, and is 2*b
                  lambda x, y: jnp.einsum("ij,jk->ik",
                                          jnp.concatenate([x, x], axis=1),
                                          jnp.concatenate([y, y], axis=0)),
                  [RandArg((3, 4), _f32), RandArg((4, 5), _f32)],
                  poly_axes=[1, 0]),
    _make_harness("einsum", "4_alt",
                  # Reduced dimension is polymorphic, and is 2*b
                  lambda x, y: jnp.einsum(jnp.concatenate([x, x], axis=1), [0, 1],
                                          jnp.concatenate([y, y], axis=0), [1, 2],
                                          [0, 2]),
                  [RandArg((3, 4), _f32), RandArg((4, 5), _f32)],
                  poly_axes=[1, 0]),
    _make_harness("einsum", "multiple_contractions",
                  lambda x, y, z: jnp.einsum("ab,bc,cd->ad", x, y, z),
                  [RandArg((3, 2), _f32), RandArg((2, 3), _f32), RandArg((3, 4), _f32),],
                  poly_axes=[0, None, None]),
    _make_harness("einsum", "incompatible_contractions_error",
                  lambda x, y: jnp.einsum("ab,cb->ac", x, y),
                  [RandArg((2, 3), _f32), RandArg((2, 3), _f32)],
                  poly_axes=[1, (0, 1)],
                  expect_error=(core.InconclusiveDimensionOperation,
                                "Dimension polynomial comparison 'b1' == 'b0' is inconclusive")),
    _make_harness("eye", "N=poly_M=None",
                  lambda x: jnp.eye(x.shape[0]),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("eye", "N=poly_M=poly",
                  lambda x: jnp.eye(x.shape[0], M=x.shape[0] + 2),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("full", "",
                  lambda x: lax.full((x.shape[0], 2), 3.),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    # operand is non-poly, index is poly
    _make_harness("getitem", "op=static_idx=poly",
                  lambda a, i: a[i],
                  [RandArg((3, 4), _f32), np.array([2, 2], np.int32)],
                  poly_axes=[None, 0], enable_and_diable_xla=True),
    # operand is poly, index is integer
    _make_harness("getitem", "op=poly_idx=const",
                  lambda a: a[1],
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0], enable_and_diable_xla=True),
    # operand is poly, index is dim poly
    _make_harness("getitem", "op=poly_idx=dim",
                  lambda a: a[jax.core.dimension_as_value(a.shape[0] - 2)],
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0], enable_and_diable_xla=True),
    # Both the operand and the index are poly
    _make_harness("getitem", "op=poly_idx=poly",
                  lambda a, i: a[i],
                  [RandArg((3, 4), _f32), np.array([1, 2, 0], np.int32)],
                  poly_axes=[0, 0], enable_and_diable_xla=True),
    # op is poly and index is an entire slice
    _make_harness("getitem", "op=poly_idx=slice-all",
                  lambda a: a[:],
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0], enable_and_diable_xla=True),
    # op is poly and index is a partial slice
    _make_harness("getitem", "op=poly_idx=slice-ct-1",
                  lambda a: a[:2],
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0], enable_and_diable_xla=True,
                  expect_error=(IndexError, "Cannot use NumPy slice indexing on an array dimension")),
    _make_harness("getitem", "op=poly_idx=slice-ct-2",
                  lambda a: a[:, :2],
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0], enable_and_diable_xla=True),
    _make_harness("getitem", "op=poly_idx=slice-None-1",
                  lambda a: a[:a.shape[0]],
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0], enable_and_diable_xla=True),
    _make_harness("getitem", "op=poly_idx=slice-poly",
                  lambda a: a[:a.shape[0] - 1],
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0], enable_and_diable_xla=True,
                  expect_error=(IndexError, "Array slice indices must have static")),
    _make_harness("image_resize", "linear_0",
                  lambda x: jax.image.resize(x, (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3]),
                                             method="linear"),
                  [RandArg((3, 16, 32, 3), _f32)],
                  poly_axes=[(1, 2)]),
    _make_harness("image_resize", "linear_to_fixed_dim",
                  lambda x: jax.image.resize(x, (x.shape[0], 64, 64, x.shape[3]),
                                             method="linear"),
                  [RandArg((3, 16, 32, 3), _f32)],
                  poly_axes=[(1, 2)]),
    _make_harness("image_resize", "nearest_0",
                  lambda x: jax.image.resize(x, (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[3]),
                                             method="nearest"),
                  [RandArg((3, 5, 7, 3), _f32)],
                  poly_axes=[(1, 2)]),
    _make_harness("index_in_dim", "0",
                  lambda x: lax.index_in_dim(x, -1, axis=0, keepdims=False),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("index_in_dim", "idx=neg",
                  lambda x: lax.index_in_dim(x, -1, axis=0, keepdims=False),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("index_in_dim", "idx=last",
                  lambda x: lax.index_in_dim(x, x.shape[0] - 1, axis=0, keepdims=False),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("iota", "",
                  lambda x: x + lax.iota(_f32, x.shape[0]),
                  [RandArg((3,), _f32)],
                  poly_axes=[0]),
    _make_harness("matmul", "0",
                  jnp.matmul,
                  [RandArg((7, 8, 4), _f32), RandArg((7, 4, 5), _f32)],
                  poly_axes=[0, 0],
                  tol=1e-5),
    _make_harness("matmul", "1",
                  jnp.matmul,
                  [RandArg((7, 8, 4), _f32), RandArg((4, 5), _f32)],
                  poly_axes=[0, None],
                  tol=1e-5),
    [
        _make_harness("mean",
                      f"axis={axis}_keepdims={keepdims}_where=None",
                      lambda x, axis, keepdims: jnp.mean(x, axis=axis, keepdims=keepdims, where=None),
                      [RandArg((7, 8, 4), _f32), StaticArg(axis), StaticArg(keepdims)],
                      poly_axes=[0])
        for keepdims in [False, True]
        for axis in [None, (0,), (0, 1), (1,)]
    ],
    [
        _make_harness("mean",
                      f"axis={axis}_keepdims={keepdims}_where=Some",
                      lambda x, where, axis, keepdims: jnp.mean(x, axis=axis, keepdims=keepdims, where=where),
                      [RandArg((7, 8, 4), _f32), RandArg((7, 8, 4), np.bool_), StaticArg(axis), StaticArg(keepdims)],
                      poly_axes=[0, 0])
        for keepdims in [False, True]
        for axis in [None, (0,), (0, 1), (1,)]
    ],
    _make_harness("ones", "",
                  lambda x: jnp.ones(x.shape, dtype=_f32),
                  [RandArg((3, 2, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("pad", "",
                  lax.pad,
                  [RandArg((3, 2, 5), _f32), np.float32(5.),
                   StaticArg(((0, 0, 0), (0, 0, 0), (1, 1, 1)))],
                  poly_axes=[0, None]),
    _make_harness("random_gamma", "",
                  lambda key, a: jax.random.gamma(key, a),
                  [RandArg((3, 2), np.uint32), RandArg((3, 3), _f32)],
                  poly_axes=[0, 0]),
    # The known dimensions product must be even.
    _make_harness("random_categorical", "axis=0",
                  lambda key, a: jax.random.categorical(key, a, axis=0),
                  [RandArg((2,), np.uint32), RandArg((3, 8), _f32)],
                  poly_axes=[None, 0]),
    _make_harness("random_categorical", "axis=1",
                  lambda key, a: jax.random.categorical(key, a, axis=1),
                  [RandArg((2,), np.uint32), RandArg((3, 8), _f32)],
                  poly_axes=[None, 0]),
    # Works when the known dimensions are known to be even or odd.
    _make_harness("random_uniform", "even_1",
                  lambda key, a: jax.random.uniform(key, a.shape, dtype=_f32),
                  [RandArg((2,), np.uint32), RandArg((3, 4), _f32)],
                  poly_axes=[None, 0]),
    _make_harness("random_uniform", "even_2",
                  lambda key, a: jax.random.uniform(key, (2 * a.shape[0], a.shape[1]),
                                                    dtype=_f32),
                  [RandArg((2,), np.uint32), RandArg((3, 5), _f32)],
                  poly_axes=[None, 0]),
    _make_harness("random_uniform", "error_not_even",
                  lambda key, a: jax.random.uniform(key, a.shape, dtype=_f32),
                  [RandArg((2,), np.uint32), RandArg((3, 5), _f32)],
                  poly_axes=[None, 0],
                  expect_error=(core.InconclusiveDimensionOperation,
                                "the product of the known dimensions must be even")),
    # TODO(necula): not yet supported, but also unlikely to come up.
    # _make_harness("random_uniform", "odd",
    #               lambda key, a: jax.random.uniform(key, (2 * a.shape[0] + 1, a.shape[1]),
    #                                                 dtype=_f32),
    #               [RandArg((2,), np.uint32), RandArg((3, 5), _f32)],
    #               poly_axes=[None, 0]),
    [
        _make_harness("reduce", reduce_op.__name__,
                      lambda x: reduce_op(x, axis=-1, keepdims=True),
                      [RandArg((3, 5), _f32)],
                      poly_axes=[0])
        for reduce_op in [jnp.all, jnp.any, jnp.max, jnp.min, jnp.prod, jnp.sum]
    ],
    _make_harness("reshape", "0",
                  lambda x: x.reshape([x.shape[0], -1]),
                  [RandArg((3, 2, 3), _f32)],
                  poly_axes=[0]),
    _make_harness("reshape", "1",
                  lambda x: x.reshape([x.shape[0], -1]),
                  [RandArg((3, 2, 3), _f32)],
                  poly_axes=[(0, 1)]),
    _make_harness("reshape", "2",
                  lambda x: x.reshape([x.shape[0], -1, x.shape[3], x.shape[2]]),
                  [RandArg((3, 4, 5, 6, 7), _f32)],
                  poly_axes=[(0, 2, 3)]),
    _make_harness("reshape", "3",
                  lambda x: jnp.reshape(x, [2, -1]),
                  [RandArg((3, 4, 5, 6, 7), _f32)],
                  poly_axes=[(0, 2)]),
    _make_harness("reshape", "error",
                  lambda x: x.reshape([x.shape[0], -1, 3]),
                  [RandArg((3, 2, 4), _f32)],
                  poly_axes=[0],
                  skip_jax_run=True,
                  expect_error=(core.InconclusiveDimensionOperation,
                                re.escape(
                                  "Cannot divide evenly the sizes of shapes (b0, 2, 4) and (b0, -1, 3)"))),

    _make_harness("scatter_add", "",
                  partial(lax.scatter_add, indices_are_sorted=False, unique_indices=True),
                  [RandArg((7, 4), _f32),
                   np.array([[1], [2]], np.int32),  # indices
                   RandArg((7, 2), _f32),  # updates
                   StaticArg(lax.ScatterDimensionNumbers((0,), (1,), (1,)))],
                  poly_axes=[0, None, 0]),
    _make_harness("scatter_add", "clip",
                  partial(lax.scatter_add, indices_are_sorted=False, unique_indices=True, mode=lax.GatherScatterMode.CLIP),
                  [RandArg((7, 4), _f32),
                   np.array([[1], [2]], np.int32),  # indices
                   RandArg((7, 2), _f32),  # updates
                   StaticArg(lax.ScatterDimensionNumbers((0,), (1,), (1,)))],
                  poly_axes=[0, None, 0]),
    _make_harness("select", "0",
                  # x.shape = (b, 3)
                  lambda x: lax.select(x > 5., x, x),
                  [RandArg((7, 3), _f32)],
                  poly_axes=[0]),
    _make_harness("select", "1",
                  # x.shape = (b, 3); y.shape = (3,)
                  jax.vmap(lambda x, y: lax.select(x > 5., x, y), in_axes=[0, None]),
                  [RandArg((7, 3), _f32), RandArg((3,), _f32)],
                  poly_axes=[0, None]),
    _make_harness("slice", "entire_axis",
                  lambda x: lax.slice(x, start_indices=(0, 1), limit_indices=(x.shape[0], 3)),
                  [RandArg((7, 3), _f32)],
                  poly_axes=[0]),
    _make_harness("slice_in_dim", "entire_axis",
                  lambda x: lax.slice_in_dim(x, 0, x.shape[0], stride=1, axis=0),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("slice_in_dim", "start=neg",
                  lambda x: lax.slice_in_dim(x, -1, x.shape[0], stride=1, axis=0),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("slice_in_dim", "limit=neg",
                  lambda x: lax.slice_in_dim(x, 0, -1, stride=1, axis=0),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("squeeze", "axis=None",
                  jnp.squeeze,
                  [RandArg((5,), _f32), StaticArg(())],
                  poly_axes=[0]),
    _make_harness("squeeze", "axis=1",
                  jnp.squeeze,
                  [RandArg((4, 1), _f32), StaticArg((1,))],
                  poly_axes=[0]),
    _make_harness("squeeze", "axis=1_2",
                  jnp.squeeze,
                  [RandArg((4, 1, 1), _f32), StaticArg((1, 2))],
                  poly_axes=[0]),
    _make_harness("squeeze", "error",
                  jnp.squeeze,
                  [RandArg((3, 33), _f32), StaticArg(-1)],
                  poly_axes=[(0, 1)],
                  skip_jax_run=True,
                  expect_error=(ValueError,
                                re.escape(
                                  "cannot select an axis to squeeze out which has size not equal to one, got shape=(b0, b1) and dimensions=(1,)"))
                  ),
    _make_harness("take", "",
                  lambda a, i: jnp.take(a, i, axis=1),
                  [RandArg((3, 4, 5), _f32), np.array([1, 2], np.int32)],
                  poly_axes=[0, None], enable_and_diable_xla=True),
    _make_harness("tile", "0",
                  lambda x: jnp.tile(x, (1, 2)),
                  [RandArg((4, 3), _f32)],
                  poly_axes=[0]),
    _make_harness("tile", "1",
                  # The repetitions are polys
                  lambda x: jnp.tile(x, (1, x.shape[0])),
                  [RandArg((4, 2), _f32)],
                  poly_axes=[0]),
    _make_harness("tri", "N=poly_M=None",
                  lambda x: jnp.tri(x.shape[0]),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    _make_harness("tri", "N=poly_M=poly",
                  lambda x: jnp.tri(x.shape[0], M=x.shape[0] + 2),
                  [RandArg((3, 4), _f32)],
                  poly_axes=[0]),
    [
        _make_harness("var",
                      f"axis={axis}_keepdims={keepdims}_where=None",
                      lambda x, axis, keepdims: jnp.var(x, axis=axis, keepdims=keepdims, where=None),
                      [RandArg((7, 8, 4), _f32), StaticArg(axis), StaticArg(keepdims)],
                      poly_axes=[0])
        for keepdims in [False, True]
        for axis in [None, (0,), (0, 1), (1,)]
    ],
    [
        _make_harness("var",
                      f"axis={axis}_keepdims={keepdims}_where=Some",
                      lambda x, where, axis, keepdims: jnp.var(x, axis=axis, keepdims=keepdims, where=where),
                      [RandArg((7, 8, 4), _f32), RandArg((7, 8, 4), np.bool_), StaticArg(axis), StaticArg(keepdims)],
                      poly_axes=[0, 0])
        for keepdims in [False, True]
        for axis in [None, (0,), (0, 1), (1,)]
    ],
    _make_harness("where", "",
                  jnp.where,
                  [RandArg((2,), np.bool_), RandArg((), _f32), RandArg((2,), _f32)],
                  poly_axes=[0, None, 0]),
]

### We add to the test harnesses some that are obtained from the
### primitive harnesses by applying vmap to the function and then asserting
### that we can convert shape polymorphically the result.

def _add_vmap_primitive_harnesses():
  """For each harness group, pick a single dtype.

  Ignore harnesses that fail in graph mode in jax2tf.
  """
  all_h = primitive_harness.all_harnesses

  # Index by group
  harness_groups: Dict[
    str, Sequence[primitive_harness.Harness]] = collections.defaultdict(list)
  device = jtu.device_under_test()

  for h in all_h:
    # Drop the the JAX limitations
    if not h.filter(device_under_test=device, include_jax_unimpl=False):
      continue
    # And the jax2tf limitations that are known to result in TF error.
    if any(l.expect_tf_error for l in _get_jax2tf_limitations(device, h)):
      continue
    # TODO(marcvanzee): We currently exclude tests with enable_xla=False because
    # this doesn't work with vmap due to a call to lax.gather. We should include
    # them once vmap works with enable_xla=False.
    if not h.params.get("enable_xla", True):
      continue
    harness_groups[h.group_name].append(h)

  selected_harnesses = []
  for group_name, hlist in harness_groups.items():
    # Pick the dtype with the most harnesses in this group. Some harness
    # groups only test different use cases at a few dtypes.
    c = collections.Counter([h.dtype for h in hlist])
    (dtype, _), = c.most_common(1)
    selected_harnesses.extend([h for h in hlist if h.dtype == dtype])

  # We do not yet support shape polymorphism for vmap for some primitives
  _NOT_SUPPORTED_YET = frozenset([
      # In linalg._lu_python we do reshape(-1, ...)
      "lu",
      "custom_linear_solve",

      # We do *= shapes in the batching rule for conv_general_dilated
      "conv_general_dilated",

      "tridiagonal_solve",  # batching not implemented in JAX
      "iota",  # vmap does not make sense for 0-argument functions
      "rng_bit_generator",  # vmap not implemented
  ])

  batch_size = 3
  for h in selected_harnesses:
    if h.group_name in _NOT_SUPPORTED_YET:
      continue

    def make_batched_arg_descriptor(
        ad: primitive_harness.ArgDescriptor) -> Optional[primitive_harness.ArgDescriptor]:
      if isinstance(ad, RandArg):
        return RandArg((batch_size,) + ad.shape, ad.dtype)
      elif isinstance(ad, CustomArg):
        def wrap_custom(rng):
          arg = ad.make(rng)
          return np.stack([arg] * batch_size)

        return CustomArg(wrap_custom)
      else:
        assert isinstance(ad, np.ndarray), ad
        return np.stack([ad] * batch_size)

    new_args = [make_batched_arg_descriptor(ad)
                for ad in h.arg_descriptors
                if not isinstance(ad, StaticArg)]

    # This test does not make sense for nullary functions
    if not new_args:
      continue

    # We do not check the result of harnesses that require custom assertions.
    check_result = all(not l.custom_assert and not l.skip_comparison and l.tol is None
                       for l in _get_jax2tf_limitations(device, h))
    vmap_harness = _make_harness(h.group_name, f"vmap_{h.name}",
                                 jax.vmap(h.dyn_fun, in_axes=0, out_axes=0),
                                 new_args,
                                 poly_axes=[0] * len(new_args),
                                 check_result=check_result,
                                 **h.params)
    _POLY_SHAPE_TEST_HARNESSES.append(vmap_harness)


def _get_jax2tf_limitations(
    device, h: primitive_harness.Harness) -> Sequence[Jax2TfLimitation]:
  # And the jax2tf limitations
  def applicable_jax2tf_limitation(l: Jax2TfLimitation) -> bool:
    # The CheckShapePolymorphism uses tf.function, so we care about "graph"
    return l.filter(device=device, dtype=h.dtype, mode="graph")

  limitations = Jax2TfLimitation.limitations_for_harness(h)
  return tuple(filter(applicable_jax2tf_limitation, limitations))


_add_vmap_primitive_harnesses()

def _flatten_harnesses(harnesses):
  res = []
  for h in harnesses:
    if isinstance(h, Sequence):
      res.extend(h)
    else:
      res.append(h)
  return res

class ShapePolyPrimitivesTest(tf_test_util.JaxToTfTestCase):
  """Tests for primitives that take shape values as parameters."""

  # This test runs for all _POLY_SHAPE_PRIMITIVE_HARNESSES.

  # For each primitive "xxx" the test will be called "test_prim_xxx_...".
  # If you want to run this test for only one harness that includes "foo"
  # in the name (after test_prim), add parameter `one_containing="foo"`
  # to parameterized below.
  @primitive_harness.parameterized(
      _flatten_harnesses(_POLY_SHAPE_TEST_HARNESSES),
      #one_containing="reshape_1_poly_axes=[(0, 1)]"
  )
  def test_prim(self, harness: Harness):
    args = harness.dyn_args_maker(self.rng())
    poly_axes = harness.params["poly_axes"]  # type: Sequence[Sequence[int]]
    assert len(args) == len(poly_axes)
    # Make the polymorphic_shapes and input_signature
    polymorphic_shapes: List[Optional[str]] = []
    input_signature: List[tf.TensorSpec] = []
    for arg, poly_axis in zip(args, poly_axes):
      if poly_axis is None:
        polymorphic_shapes.append(None)
        input_signature.append(tf.TensorSpec(np.shape(arg), arg.dtype))
      else:
        def make_arg_polymorphic_shapes(poly_axis: Sequence[int]) -> Tuple[str, tf.TensorSpec]:
          idx = -1
          dims = []
          tensorspec_dims: List[Optional[int]] = []
          for i, d in enumerate(arg.shape):
            if i in poly_axis:
              idx += 1
              dims.append(f"b{idx}")
              tensorspec_dims.append(None)
            else:
              dims.append(str(d))
              tensorspec_dims.append(d)
          return ", ".join(dims), tf.TensorSpec(tensorspec_dims, arg.dtype)

        arg_polymorphic_shapes, arg_tensorspec = make_arg_polymorphic_shapes(poly_axis)
        polymorphic_shapes.append(arg_polymorphic_shapes)
        input_signature.append(arg_tensorspec)

    skip_jax_run = harness.params["skip_jax_run"]
    if not skip_jax_run:
      res_jax = harness.dyn_fun(*args)

    enable_xla = harness.params.get("enable_xla", True)
    expect_error_type, expect_error_regex = harness.params["expect_error"]
    if expect_error_type is not None:
      with self.assertRaisesRegex(expect_error_type, expect_error_regex):
        f_tf = self.CheckShapePolymorphism(
            harness.dyn_fun,
            input_signature=input_signature,
            polymorphic_shapes=polymorphic_shapes,
            expected_output_signature=None,
            enable_xla=enable_xla)
    else:
      f_tf = self.CheckShapePolymorphism(
          harness.dyn_fun,
          input_signature=input_signature,
          polymorphic_shapes=polymorphic_shapes,
          expected_output_signature=None,
          enable_xla=enable_xla)

    if not skip_jax_run and expect_error_type is None and harness.params["check_result"]:
      tol = harness.params["tol"]
      self.assertAllClose(res_jax, f_tf(*args), atol=tol, rtol=tol)


  def test_vmap_while(self):
    def cond_func(x):  # x: f32[3]
      return jnp.sum(x) >= 0.
    def body_func(x):  # x: f32[3]
      return x - 1.
    def f_jax(x):
      return lax.while_loop(cond_func, body_func, x)

    self.CheckShapePolymorphism(
        jax.vmap(f_jax),
        input_signature=[tf.TensorSpec((None, 3), dtype=tf.float32)],
        polymorphic_shapes=["b, ..."],
        expected_output_signature=tf.TensorSpec((None, 3), dtype=tf.float32)
    )

  def test_reshape_compiled(self):
    # We compile the result of conversion for two shapes, hence we need to
    # involve the TF compiler twice, but we trace only once with shape polymorphism
    traced = False

    def f_jax(x):
      nonlocal traced
      traced = True
      y = jnp.sin(x)
      return y.reshape([x.shape[0], -1])

    x = self.rng().rand(4, 2, 3)
    res_jax = f_jax(x)

    traced = False
    # If we get_concrete_function we trace once
    f_tf = tf.function(
        jax2tf.convert(f_jax, polymorphic_shapes=[PS("b", ...)]),
        autograph=False,
        jit_compile=True).get_concrete_function(
            tf.TensorSpec([None, 2, 3], x.dtype))
    self.assertTrue(traced)
    traced = False
    self.assertAllClose(res_jax, f_tf(x))
    self.assertFalse(traced)  # We are not tracing again

    x = self.rng().rand(6, 2, 3)
    res_jax = f_jax(x)
    traced = False

    self.assertAllClose(res_jax, f_tf(x))
    self.assertFalse(traced)  # We are not tracing again


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
