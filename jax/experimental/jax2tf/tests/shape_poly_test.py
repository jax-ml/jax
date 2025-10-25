# Copyright 2020 The JAX Authors.
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

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
import math
import re
from typing import Any

from absl import logging
from absl.testing import absltest

import jax
from jax.experimental import jax2tf
from jax import export
from jax import lax
import jax.numpy as jnp
from jax import random
from jax._src import config
from jax._src import core
from jax._src import test_util as jtu
from jax._src import util
import numpy as np

from jax.experimental.jax2tf.tests import tf_test_util

import tensorflow as tf

config.parse_flags_with_absl()

# Import after parsing flags
from jax._src.internal_test_util import test_harnesses
from jax._src.internal_test_util.test_harnesses import Harness, RandArg
from jax.experimental.jax2tf.tests.jax2tf_limitations import Jax2TfLimitation

_f32 = np.float32
_i32 = np.int32

expect_error_associative_scan = (
    NotImplementedError,
    "associative scan over axis of non-constant size",
)


class PolyHarness(Harness):
  """Tests a function with shape polymorphism.

  Converts `fun` with shape polymorphism, creates a `tf.ConcreteFunction`
  given `input_signature` and checks the inferred output shapes to match
  `expected_output_shapes`, then checks that the JAX and the TF functions
  produce the same results.
  """
  def __init__(self,
               group_name: str, name: str,
               fun: Callable,
               *,
               arg_descriptors: Sequence[test_harnesses.ArgDescriptor] = (),
               polymorphic_shapes: Sequence[str | None] = (),
               polymorphic_constraints: Sequence[str] = (),
               input_signature: Sequence[tf.TensorSpec] | None = None,
               expected_output_signature: tf.TensorSpec | None = None,
               expect_error: tuple[Any | None, str | None] = (None, None),
               skip_jax_run: bool = False,
               check_result: bool = True,
               tol: float | None = None,
               limitations: Sequence[Jax2TfLimitation] = (),
               override_jax_config_flags: dict[str, Any] = {}):
    """Args:

      group_name, name: The name for the harness. See `Harness.__init__`.
      fun: the function to be converted, possibly after partial application to
        static arguments from `arg_descriptors`. See `Harness.__init__`.
      arg_descriptors: The argument descriptors. See `Harness.__init__`. May
        be missing, in which case `skip_jax_run` should be `True` and
        `input_signature` must be present.
      polymorphic_shapes: For `jax2tf.convert`.
      polymorphic_constraints: For `jax2tf.convert`.
      input_signature: For `tf.function.get_concrete_function`. If missing,
        generated from `polymorphic_shapes`.
      expected_output_signature: the expected inferred output shape.
      expect_error: a pair of an Exception type and a regular expression to
        match the expected exception string.
      skip_jax_run: If True, then neither the JAX nor the TF functions are
        executed.
      check_result: specifies if we want to check that the result of the shape
        polymorphic conversion produces the same result and the JAX function.
      tol: the tolerance to use for checking results.
      limitations: if given, then apply the custom_assert and tolerance from the
        Jax2TfLimitations.
      override_jax_config_flags: jax.config flags to override for the duration
        of the test.
    """
    super().__init__(group_name, name, fun, arg_descriptors,
                     dtype=np.float32)
    self.polymorphic_shapes = polymorphic_shapes
    self.polymorphic_constraints = polymorphic_constraints
    self.input_signature = input_signature
    self.expected_output_signature = expected_output_signature
    self.skip_jax_run = skip_jax_run
    self.expect_error = expect_error
    self.tol = tol
    self.check_result = check_result
    self.limitations = limitations
    self.override_jax_config_flags = override_jax_config_flags

  def run_test(self, tst: tf_test_util.JaxToTfTestCase) -> jax.Array | None:
    def log_message(extra: str):
      return f"[{tst._testMethodName}]: {extra}"

    # Check that we have overridden the jax.config flags
    for fname, fvalue in self.override_jax_config_flags.items():
      tst.assertEqual(getattr(jax.config, fname), fvalue, (
          f"Flag {fname} current value {getattr(jax.config, fname)} != {fvalue}"))

    tst.assertIsNotNone(self.polymorphic_shapes)
    polymorphic_shapes = self.polymorphic_shapes
    if not self.skip_jax_run:
      args = self.dyn_args_maker(tst.rng())
    else:
      tst.assertIsNotNone(self.input_signature)

    if self.input_signature is None:
      tst.assertEqual(
        len(polymorphic_shapes), len(args),
        f"polymorphic_shapes {polymorphic_shapes} of length "
        f"{len(polymorphic_shapes)} must match number of arguments {len(args)}")
      args_specs = export.symbolic_args_specs(args, polymorphic_shapes)
      input_signature = [
        tf.TensorSpec(
            [d if isinstance(d, int) else None for d in a.shape],
            dtype=a.dtype) for a in args_specs]
    else:
      input_signature = self.input_signature  # type: ignore

    expect_error_type, expect_error_regex = self.expect_error
    if self.skip_jax_run and not self.arg_descriptors:
      f_jax = self.fun
    else:
      f_jax = self.dyn_fun

    with contextlib.ExitStack() as stack:
      if expect_error_type is not None:
        stack.enter_context(tst.assertRaisesRegex(expect_error_type, expect_error_regex))

      f_tf = jax2tf.convert(f_jax, polymorphic_shapes=polymorphic_shapes,
                            polymorphic_constraints=self.polymorphic_constraints)
      # Run in tf.Eager mode first, because it is friendlier to debuggers
      res_tf = f_tf(*args) if not self.skip_jax_run else None
      f_tf_func = tf.function(
          f_tf, autograph=False, input_signature=input_signature)
      # Create tf.ConcreteFunction and check inferred output signature
      concrete_f_tf = f_tf_func.get_concrete_function(*input_signature)

    if expect_error_type is not None:
      return None

    if self.expected_output_signature:
      # Strangely, output_shapes can be a single shape for a function with a
      # single result, or a list/tuple of shapes.
      expected_output_signature = self.expected_output_signature
      concrete_output_tf_shape = concrete_f_tf.output_shapes
      if not isinstance(concrete_output_tf_shape, (tuple, list)):  # Single result
        assert not isinstance(self.expected_output_signature, (tuple, list))
        expected_output_signature = [self.expected_output_signature]
        concrete_output_tf_shape = [concrete_output_tf_shape]
      for expected, found in util.safe_zip(expected_output_signature,
                                           concrete_output_tf_shape):
        tst.assertEqual(tuple(expected.shape), tuple(found))

    # Run the JAX and the TF functions and compare the results
    if not self.skip_jax_run:
      res_jax = f_jax(*args)
      if self.check_result:
        res_tf = tf.nest.map_structure(lambda t: t.numpy(), res_tf)
        custom_assert_lims = [
            l for l in self.limitations if l.custom_assert is not None]
        assert len(custom_assert_lims) <= 1, custom_assert_lims
        tol = None
        if self.tol is not None:
          tol = self.tol
        elif self.limitations:
          max_lim = self.limitations[0].get_max_tolerance_limitation(
              self.limitations)
          if max_lim is not None:
            tol = max_lim.tol

        if not custom_assert_lims:
          tst.assertAllClose(res_jax, res_tf, atol=tol, rtol=tol)
        else:
          logging.info(log_message(
              f"Running custom_assert with tol={tol} due "
              f"to {custom_assert_lims[0]}"))
          custom_assert_lims[0].custom_assert(tst, res_jax, res_tf, args=args,  # type: ignore
                                              tol=tol, err_msg=None)
        return res_tf
      else:
        return None
    else:
      return None


def check_shape_poly(tst, f_jax: Callable, *,
                     arg_descriptors: Sequence[test_harnesses.ArgDescriptor] = (),
                     skip_jax_run: bool = False,
                     polymorphic_shapes: Sequence[str | None] = (),
                     polymorphic_constraints: Sequence[str] = (),
                     input_signature: Sequence[tf.TensorSpec] | None = None,
                     expected_output_signature: tf.TensorSpec | None = None,
                     expect_error=(None, None)) -> jax.Array | None:
  # Makes and tests a harness. See PolyHarness documentation.
  h = PolyHarness("", "", f_jax,
                  arg_descriptors=arg_descriptors,
                  skip_jax_run=skip_jax_run,
                  polymorphic_shapes=polymorphic_shapes,
                  polymorphic_constraints=polymorphic_constraints,
                  input_signature=input_signature,
                  expected_output_signature=expected_output_signature,
                  expect_error=expect_error)
  return h.run_test(tst)


@jtu.thread_unsafe_test_class()
class ShapePolyTest(tf_test_util.JaxToTfTestCase):

  def test_simple_unary(self):
    """Test shape polymorphism for a simple case, unary function."""

    def f_jax(x):
      return x + jnp.sin(x)

    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((2, 3), _f32)],
                     polymorphic_shapes=[None],
                     expected_output_signature=tf.TensorSpec([2, 3]))

    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((2, 3), _f32)],
                     polymorphic_shapes=["_, h"],
                     expected_output_signature=tf.TensorSpec([2, None]))

    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((3, 3), _f32)],
                     polymorphic_shapes=["h, h"],
                     expected_output_signature=tf.TensorSpec([None, None]))

    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((3, 3), _f32)],
                     polymorphic_shapes=["h, h"],
                     expected_output_signature=tf.TensorSpec([None, None]))

  def test_simple_binary(self):
    """Test shape polymorphism for a simple case, binary function."""

    def f_jax(x, y):
      return x + jnp.sin(y)

    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((2, 3), _f32), RandArg((2, 3), _f32)],
                     polymorphic_shapes=[None, None],
                     expected_output_signature=tf.TensorSpec([2, 3]))

    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((2, 3), _f32), RandArg((2, 3), _f32)],
                     polymorphic_shapes=["_, h", "_, h"],
                     input_signature=[tf.TensorSpec([2, None]), tf.TensorSpec([2, 3])],
                     expected_output_signature=(
                         # for native serialization we cannot refine the inferred shape of the
                         # output if the input is more specific than polymorphic_shapes.
                         tf.TensorSpec([2, None])))

    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((3, 3), _f32), RandArg((3, 3), _f32)],
                     polymorphic_shapes=["h, h", "h, h"],
                     expected_output_signature=tf.TensorSpec([None, None]))

  def test_static_shape_result(self):
    """The result has static shape."""

    def f_jax(x):
      return jnp.sum(x + jnp.sin(x), axis=0)

    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((2, 3), _f32)],
                     polymorphic_shapes=[None],
                     expected_output_signature=tf.TensorSpec([3]))

    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((2, 3), _f32)],
                     polymorphic_shapes=["b, _"],
                     expected_output_signature=tf.TensorSpec([3]))

  def test_forgot_polymorphic_shapes_error(self):
    msg_re = "syntax error in symbolic shape"
    with self.assertRaisesRegex(ValueError, msg_re):
      check_shape_poly(self,
                       jnp.sin,
                       arg_descriptors=[RandArg((1, 3,), _f32)],
                       input_signature=[tf.TensorSpec([1, None])],
                       polymorphic_shapes=[None])

  def test_with_constraints(self):
    def f_jax(x):  # x: i32[a], with a >= 8
      return lax.dynamic_slice_in_dim(x, 0, 8, 0)
    check_shape_poly(self, f_jax,
                     arg_descriptors=[RandArg((16,), _i32)],
                     polymorphic_shapes=["a"],
                     polymorphic_constraints=["a >= 8"])

  def test_kwargs(self):
    """Test shape polymorphism for a function with kwargs."""

    x = np.ones(3, dtype=np.float32)
    y = np.ones(1, dtype=np.float32)
    def f_jax(x, *, y):
      return x + jnp.sin(y)

    f_tf: Callable[..., Any] = jax2tf.convert(f_jax, polymorphic_shapes=["b, ..."])
    self.assertAllClose(f_jax(x, y=y), f_tf(x, y=y))

  def test_arg_avals_errors(self):
    """Test error reporting for shape polymorphism."""
    def conv_and_run(*, arg_shape: core.Shape,
                     polymorphic_shape: str):
      arg = np.arange(math.prod(arg_shape), dtype=np.float32).reshape(arg_shape)
      check_shape_poly(self, lambda x: x,
                       arg_descriptors=[arg],
                       polymorphic_shapes=[polymorphic_shape])

    with self.assertRaisesRegex(ValueError,
                                re.escape("polymorphic shape spec should be")):
      conv_and_run(arg_shape=(2,), polymorphic_shape=5.)

    with self.assertRaisesRegex(ValueError,
                                re.escape("pytree structure error: different types")):
      conv_and_run(arg_shape=(2,), polymorphic_shape=["a list"])

    with self.assertRaisesRegex(ValueError,
                                re.escape("pytree structure error: different types")):
      conv_and_run(arg_shape=(2,), polymorphic_shape=("a tuple",))

    with self.assertRaisesRegex(ValueError,
                                "Cannot solve for values of dimension variables {'b'}"):
      conv_and_run(arg_shape=(4, 36, 3), polymorphic_shape="b * b, b * d * d, d")

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "Division had remainder 2 when computing the value of 'b'"):
      conv_and_run(arg_shape=(5, 36), polymorphic_shape="3 * b, ...")

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "Expected value >= 1 for dimension variable 'b'"):
      conv_and_run(arg_shape=(10, 3), polymorphic_shape="3 * b + 10, ...")

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "Expected value >= 1 for dimension variable 'b'"):
      conv_and_run(arg_shape=(7, 3), polymorphic_shape="3 * b + 10, ...")

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        re.escape(
          "Found inconsistency between dimension size "
          "args[0].shape[1] (= 3) and the specification 'a' (= 2)")):
      conv_and_run(arg_shape=(2, 3), polymorphic_shape="(a, a)")

  # Tests details of the shape constraints errors.
  # This test exists also in jax_export_test.py.
  @jtu.parameterized_filterable(
    testcase_name=lambda kw: kw["shape"],
    kwargs=[
      dict(shape=(8, 2, 9),  # a = 2, b = 3, c = 4
           poly_spec="(a + 2*b, a, a + b + c)"),
      dict(shape=(2, 2, 6),  # a = 2, b = 0, c = 4
           poly_spec="(a + 2*b, a, a + b + c)",
           expect_error=(
             "Input shapes do not match the polymorphic shapes specification. "
             "Expected value >= 1 for dimension variable 'b'. "
             "Using the following polymorphic shapes specifications: args[0].shape = (2*b + a, a, c + b + a). "
             "Obtained dimension variables: 'a' = 2 from specification 'a' for dimension args[0].shape[1] (= 2), "
             "'b' = 0 from specification '2*b + a' for dimension args[0].shape[0] (= 2), . "
             "Please see https://docs.jax.dev/en/latest/export/shape_poly.html#shape-assertion-errors for more details."
           )),
      dict(shape=(3, 2, 6),  # a = 2, b = 0.5, c = 4 - b is not integer
           poly_spec="(a + 2*b, a, a + b + c)",
           expect_error=(
             "Input shapes do not match the polymorphic shapes specification. "
             "Division had remainder 1 when computing the value of 'b'. "
             "Using the following polymorphic shapes specifications: args[0].shape = (2*b + a, a, c + b + a). "
             "Obtained dimension variables: 'a' = 2 from specification 'a' for dimension args[0].shape[1] (= 2), . "
             "Please see https://docs.jax.dev/en/latest/export/shape_poly.html#shape-assertion-errors for more details."
           )),
      dict(shape=(8, 2, 6),  # a = 2, b = 3 - inconsistency
           poly_spec="(a + 2*b, a, a + b)",
           expect_error=(
             "Input shapes do not match the polymorphic shapes specification. "
             "Found inconsistency between dimension size args[0].shape[0] (= 8) and the specification '2*b + a' (= 10). "
             "Using the following polymorphic shapes specifications: args[0].shape = (2*b + a, a, b + a). "
             "Obtained dimension variables: 'a' = 2 from specification 'a' for dimension args[0].shape[1] (= 2), "
             "'b' = 4 from specification 'b + a' for dimension args[0].shape[2] (= 6), . "
             "Please see https://docs.jax.dev/en/latest/export/shape_poly.html#shape-assertion-errors for more details."
           )),
      dict(shape=(7, 2, 36),  # a = 2, b = 3, c = 6 - cannot solve c
           poly_spec="(2 * a + b, a, c * c)",
           expect_error=(
             "Cannot solve for values of dimension variables {'c'}. "
             "We can only solve linear uni-variate constraints. "
             "Using the following polymorphic shapes specifications: args[0].shape = (b + 2*a, a, c^2). "
             "Unprocessed specifications: 'c^2' for dimension size args[0].shape[2]. "
             "Please see https://docs.jax.dev/en/latest/export/shape_poly.html#dimension-variables-must-be-solvable-from-the-input-shapes for more details."
           )),
  ])
  def test_shape_constraints_errors(self, *,
      shape, poly_spec: str, expect_error: str | None = None):
    def f_jax(x):  # x: f32[a + 2*b, a, a + b + c]
      return 0.

    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    with contextlib.ExitStack() as stack:
      if expect_error is not None:
        stack.push(self.assertRaisesRegex(Exception, re.escape(expect_error)))
      _ = check_shape_poly(self, f_jax,
                           arg_descriptors=[x],
                           polymorphic_shapes=[poly_spec])

  def test_pytree(self):
    """Arguments and polymorphic_shapes are pytrees."""

    # Arguments are of the form [([x00, x01], [x10]), dict(a=ya, b=yb)]
    def add_all_jax(x_pair_of_list, y_dict):
      x_list_0, x_list_1 = x_pair_of_list
      return sum(x_list_0 + x_list_1 + [y_dict["a"], y_dict["b"]])

    input_signature = [([tf.TensorSpec([None]), tf.TensorSpec([None])],
                                       [tf.TensorSpec([None])]),
                                      dict(a=tf.TensorSpec([None]),
                                           b=tf.TensorSpec([None]))]
    check_shape_poly(self,
                     add_all_jax,
                     skip_jax_run=True,
                     input_signature=input_signature,
                     polymorphic_shapes=[(["v", "v"], ["v"]),
                                         dict(a="v", b="v")],
                     expected_output_signature=tf.TensorSpec([None]))

    # Prefix polymorphic shapes
    check_shape_poly(self,
                     add_all_jax,
                     skip_jax_run=True,
                     input_signature=input_signature,
                     polymorphic_shapes="v",
                     expected_output_signature=tf.TensorSpec([None]))

    check_shape_poly(self,
                     add_all_jax,
                     skip_jax_run=True,
                     input_signature=input_signature,
                     polymorphic_shapes=["v", "v"],
                     expected_output_signature=tf.TensorSpec([None]))

    check_shape_poly(self,
                     add_all_jax,
                     skip_jax_run=True,
                     input_signature=input_signature,
                     polymorphic_shapes=[("v", "v"), "v"],
                     expected_output_signature=tf.TensorSpec([None]))

    # Now partial polymorphic_shapes; the parts of the polymorphic_shapes that
    # are not specified must have full input_signatures.
    check_shape_poly(self,
                     add_all_jax,
                     skip_jax_run=True,
                     input_signature=[([tf.TensorSpec([4]), tf.TensorSpec([4])], [tf.TensorSpec([4])]),
                                      dict(a=tf.TensorSpec([4]), b=tf.TensorSpec([4]))],
                     polymorphic_shapes=((["(4,)", "(_,)"], [("4,")]),
                                         dict(a="(_,)", b="(4,)")),
                     expected_output_signature=tf.TensorSpec([4]))

  @jtu.parameterized_filterable(
    kwargs=[
      dict(testcase_name=name, polymorphic_shapes=polymorphic_shapes)
      for name, polymorphic_shapes in [
          ("1", ("b", "b", "b")),
          ("2", dict(a="b")),
          ("3", (dict(a="b"), "b")),
      ]]
  )
  def test_pytree_errors(self, polymorphic_shapes=("b", "b", "b")):
    """Arguments and polymorphic_shapes are not-matching pytrees."""

    # Arguments are of the form [([x00, x01], [x10]), dict(a=ya, b=yb)]
    x = np.arange(4, dtype=_f32)
    args = (([x, x], [x]), dict(a=x, b=x))
    def add_all_jax(x_pair_of_list, y_dict):
      x_list_0, x_list_1 = x_pair_of_list
      return sum(x_list_0 + x_list_1 + [y_dict["a"], y_dict["b"]])

    with self.assertRaisesRegex(ValueError, "pytree structure error"):
      jax2tf.convert(add_all_jax,
                     polymorphic_shapes=polymorphic_shapes)(*args)

  def test_with_nested_jit(self):
    def f_jax(x):  # x: f32[w, h]
      # x + (np.sin(x) + np.broadcast_to(np.arange(x.shape[1]), x.shape))
      return jnp.sin(x) + jnp.arange(x.shape[1], dtype=x.dtype)
    check_shape_poly(self,
                     lambda x: x + jax.jit(f_jax)(x),
                     arg_descriptors=[RandArg((3, 4), _f32)],
                     polymorphic_shapes=["a, b"])

  @jtu.parameterized_filterable(
    kwargs=[
      dict(testcase_name=str(polymorphic_shapes), polymorphic_shapes=polymorphic_shapes)
      # The polymorphic_shapes should have three comma-separated DimExpr matching
      # 16, 24, 32
      for polymorphic_shapes in [
          "b1+6,b1+14,b2",  # b1=10, b2=32
          "2*b1,4*b2,b1+b2+18",  # b1=8,b2=6
          "b1+2*b2,4*b2,b1*b1+16",  # b1=4,b2=6
      ]
  ])
  def test_non_trivial_polynomials_spec(self,
                                        polymorphic_shapes="2*b1,4*b2,b1+b2+18"):
    # We can handle non-trivial polynomials in the input shape,
    # as long as all variables also occur in trivial expressions
    check_shape_poly(self,
        lambda x: 2 * x.shape[0] + 3 * x.shape[1] + 4 * x.shape[2],
        arg_descriptors=[RandArg((16, 24, 32), _f32)],
        polymorphic_shapes=[polymorphic_shapes])

  def test_unused_args(self):
    # Tests with functions that do not use their inputs.

    # First arg unused, not polymorphic
    check_shape_poly(self,
                     lambda x_unused, y: y * 2.0,
                     arg_descriptors=[RandArg((2, 3), _f32), RandArg((3,), _f32)],
                     polymorphic_shapes=[None, "b"])

    # Some args unused, not polymorphic
    check_shape_poly(self,
                     lambda x_unused, y, z_unused, w: jnp.concatenate([y, w]),
                     arg_descriptors=[RandArg((3,), _f32), RandArg((4,), _f32),
                           RandArg((5,), _f32), RandArg((6,), _f32)],
                     polymorphic_shapes=[None, "b1", None, "b2"])

    # A polymorphic arg is not used, but the dimension var appears
    # in a used arg also
    check_shape_poly(self,
                     lambda x_unused, y: y * 2.0,
                     arg_descriptors=[RandArg((3,), _f32), RandArg((3,), _f32)],
                     polymorphic_shapes=["b", "b"])

    # A polymorphic arg is not used, and the dimension var does not appear
    # elsewhere.
    check_shape_poly(self,
        lambda x_unused, y: y * 2.0,
        arg_descriptors=[RandArg((4,), _f32), RandArg((3,), _f32)],
        polymorphic_shapes=["b1", "b2"])

    # A polymorphic arg is not used, and the dimension var does appear
    # elsewhere but not as a trivial monomial.
    check_shape_poly(self,
        lambda x_unused, y: y * 2.0,
        arg_descriptors=[RandArg((3,), _f32), RandArg((9,), _f32)],
        polymorphic_shapes=["b1", "b1 * b1"])

    # It is not sufficient to just use the shape of an input; it is still unused
    check_shape_poly(self,
        lambda x_unused, y: y + x_unused.shape[0],
        arg_descriptors=[RandArg((3,), _f32), RandArg((9,), _f32)],
        polymorphic_shapes=["b1", "b2"])

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

    f_tf = jax2tf.convert(f, polymorphic_shapes=["(batch1, batch2, d1, d2)"])
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

    # for native serialization we cannot refine the inferred shape of the
    # output if the input is more specific than polymorphic_shapes.
    self.assertEqual((None, None, None, None), tuple(tf_grad.output_shapes[0]))
    self.assertEqual((None, None, None, None), tuple(tf_grad.output_shapes[1]))

  def test_gradients_pytree(self):
    """Shape polymorphism with gradients and pytrees for inputs and outputs."""

    def f(x):
      # x: dict(x=[b, 3, 4])
      # res: dict(res=[b, 3, 4])
      return dict(res=x["x"] * 2.)

    check_shape_poly(self,
                     f,
                     skip_jax_run=True,
                     input_signature=[dict(x=tf.TensorSpec([None, 3, 4]))],
                     polymorphic_shapes=[dict(x=("b, 3, 4"))])

    f_tf = jax2tf.convert(f, polymorphic_shapes=[dict(x=("b, 3, 4"))])
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
        check_shape_poly(self, f, arg_descriptors=[x, y],
                         polymorphic_shapes=["(b, h)", "h"]))

  def test_while(self):
    def f(x):
      # x: f32[B], iter: i32
      return lax.while_loop(lambda x_iter: x_iter[1] < 5,
                            lambda x_iter: (x_iter[0] + jnp.arange(x_iter[0].shape[0], dtype=np.float32), x_iter[1] + 1),
                            (x, 0))

    x = np.ones((3,), dtype=np.float32)
    res_tf = check_shape_poly(self, f, arg_descriptors=[x],
                              polymorphic_shapes=["(b,)"])
    self.assertAllClose(f(x), res_tf)

  @jtu.parameterized_filterable(
    kwargs=[dict(with_function=v) for v in [True, False]]
  )
  def test_grad_int(self, with_function=False):
    # https://github.com/jax-ml/jax/issues/7093
    # Also issue #6975.
    x_shape = (2, 3, 4)
    xi = np.arange(math.prod(x_shape), dtype=np.int16).reshape(x_shape)
    yf = xi.astype(np.float32)
    xi_yf = (xi, yf)
    zb = np.array([True, False], dtype=np.bool_)
    def f_jax(xi_yf, zb):  # xi: s16[2, 3, 4], yf: f32[2, 3, 4], zb: bool[2]
      # results: f32[2, 3, 4], s16[2, 3, 4], bool[2], f32[2, 3, 4]
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

  def test_prng(self):
    # The PRNG implementation uses opaque types, test shape polymorphism
    with config.enable_custom_prng(True):

      def f_jax(x):  # x: f32[b1, b2]
        key = random.PRNGKey(123)  #  key: key<fry>[]
        # Exercise key operations that have custom lowering rules
        broadcast_keys = lax.broadcast_in_dim(key, x.shape, ())  # key<fry>[b1, b2]
        gather_keys = lax.broadcast_in_dim(broadcast_keys[0], (1, x.shape[1]), (1,))  # : key[1, b2]
        slice_keys1 = lax.slice(broadcast_keys, (0, 0), (1, x.shape[1]), (1, 1))  # key[1, b2]
        slice_keys2 = lax.dynamic_slice(broadcast_keys, (0, 0), slice_sizes=(1, x.shape[1]))  # key[1, b2]
        upd1 = lax.dynamic_update_slice(slice_keys2, slice_keys1, start_indices=(0, 0))  # key[1, b2]
        _ = lax.dynamic_update_slice(upd1, gather_keys, start_indices=(0, 0))

        # We need to test the special case for vmap(while)
        xs = broadcast_keys
        counts = jnp.arange(broadcast_keys.shape[0], dtype=np.int32)
        def f_vmap_jax(counts, xs):  # counts: i32[b1], xs: key<fry>[b1, b2]
          def inner(count, x):  # count i32, x: key<fry>[b2]
            return lax.fori_loop(0, count, lambda _, acc: acc, x)
          return jax.vmap(inner)(counts, xs)

        _ = f_vmap_jax(counts, xs)
        return x

      check_shape_poly(self, f_jax,
                       arg_descriptors=[RandArg((3, 4), _f32)],
                       polymorphic_shapes=["b1, b2"])

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
    x = np.arange(math.prod(x_shape), dtype=np.int32).reshape(x_shape)

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
      return x

    f_tf = jax2tf.convert(f_jax, polymorphic_shapes=["(b, ...)"])
    x = np.array([0.7, 0.8], dtype=np.float32)
    restored_f, _ = tf_test_util.SaveAndLoadFunction(
        f_tf, input_signature=[tf.TensorSpec([None], x.dtype)])
    self.assertAllClose(f_jax(x), restored_f(x))

  def test_readme_examples(self):
    """Some of the examples from the README."""

    jax2tf.convert(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],)),
                   polymorphic_shapes=["(b, 4)"])(np.ones((3, 4)))

    jax2tf.convert(lambda x: jnp.reshape(x, (math.prod(x.shape),)),
                   polymorphic_shapes=["(b, 4)"])(np.ones((3, 4)))

    jax2tf.convert(lambda x: x + x.shape[0] + jnp.sin(x.shape[0]),
                   polymorphic_shapes=["b"])(np.ones(3))

    jax2tf.convert(lambda x: jnp.sum(x, axis=0) / x.shape[0],
                   polymorphic_shapes=["(v, _)"])(np.ones((3, 4)))

    with self.assertRaisesRegex(TypeError,
                                "prod requires ndarray or scalar arguments"):
      jax2tf.convert(lambda x: jnp.prod(x.shape) + x,
                     polymorphic_shapes=["(b, 4)"])(np.ones((3, 4)))

    jax2tf.convert(lambda x: jnp.prod(jnp.array(x.shape)) + x,
                   polymorphic_shapes=["(b, 4)"])(np.ones((3, 4)))

    four_ones = np.ones((4,))
    with self.assertRaisesRegex(
        TypeError,
        re.escape("add got incompatible shapes for broadcasting: (v,), (4,)")):
      jax2tf.convert(lambda x, y: x + y,
                     polymorphic_shapes=["(v,)", "(4,)"])(four_ones, four_ones)

    # We get the error even if we use correct actual arguments
    with self.assertRaisesRegex(
        TypeError,
        re.escape("add got incompatible shapes for broadcasting: (v,), (4,)")):
      jax2tf.convert(
          lambda x, y: x + y, polymorphic_shapes=["(v,)", "(4,)"])(four_ones,
                                                                   four_ones)

    with self.assertRaisesRegex(TypeError,
                                re.escape("dot_general requires contracting dimensions to have the same shape, got (4,) and (v,)")):
      jax2tf.convert(lambda x: jnp.matmul(x, x),
                     polymorphic_shapes=["(v, 4)"])(np.ones((4, 4)))

    with self.assertRaisesRegex(core.InconclusiveDimensionOperation,
                                re.compile("Cannot divide evenly the sizes of shapes \\(b, 5, 7\\) and \\(2, -1\\)",
                                           re.DOTALL)):
      jax2tf.convert(lambda x: jnp.reshape(x, (2, -1)),
                     polymorphic_shapes=["(b, _, _)"])(np.ones((4, 5, 7)))

    jax2tf.convert(lambda x: jnp.reshape(x, (2, -1)),
                   polymorphic_shapes=["(b, _, _)"])(np.ones((4, 5, 6)))
    jax2tf.convert(lambda x: jnp.reshape(x, (-1, x.shape[0])),
                   polymorphic_shapes=["(b1, b2, ...)"])(np.ones((4, 5, 6)))

    jax2tf.convert(lambda x: jnp.reshape(x, (2, -1)),
                   polymorphic_shapes=["(2*b, ...)"])(np.ones((4, 5, 7)))

    with self.assertRaisesRegex(
        core.InconclusiveDimensionOperation,
        re.escape("Symbolic dimension comparison 'a + 1' >= 'b' is inconclusive")):
      jax2tf.convert(lambda x: 0 if x.shape[0] + 1 >= x.shape[1] else 1,
                     polymorphic_shapes=["(a, b)"])(np.ones((4, 4)))

    # Checking that the dimension variable is >= 1
    def f1_jax(x):  # f32[b]
      # We have to use "x"
      return jnp.concatenate([x, jnp.array([0. if x.shape[0] == 0 else 1.],
                                           dtype=np.float32)])

    x0 = np.array([], np.float32)
    self.assertEqual(jnp.array([0.], dtype=np.float32), f1_jax(x0))

    # We also catch the error with native serialization
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        re.escape(
          "Expected value >= 1 for dimension variable 'b'. "
          "Using the following polymorphic shapes specifications: args[0].shape = (b,). "
          "Obtained dimension variables: 'b' = 0")):
      _ = jax2tf.convert(f1_jax, polymorphic_shapes=["b"])(x0)

    # Checking that the actual dimensions denoted by the same
    # dimension variables have equal sizes.
    def f2_jax(x):  # f32[b, b]
      # We have to use "x"
      return jnp.sum(x) + (0. if x.shape[0] != x.shape[1] else 1.)

    x45 = np.ones((4, 5), dtype=np.float32)
    # JAX with static shapes sees that x.shape[0] != x.shape[1]
    self.assertEqual(jnp.sum(x45), f2_jax(x45))

    # We also catch the error with native serialization
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        re.escape(
          "Found inconsistency between dimension size args[0].shape[1] (= 5) "
          "and the specification 'b' (= 4)")):
      _ = jax2tf.convert(f2_jax, polymorphic_shapes=["b, b"])(x45)

    x = np.ones((5,), dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        "Cannot solve for values of dimension variables"):
      jax2tf.convert(lambda x: jnp.sum(x), polymorphic_shapes=["a + b"])(x)

  def test_dynamic_shapes(self):
    # Test dim_as_value with dynamic shapes.
    def f(x):
      return jnp.sum(x, axis=0) * x.shape[0]

    x = np.arange(3.)
    self.assertAllClose(9.,
                        check_shape_poly(self, f,
                                         arg_descriptors=[x],
                                         polymorphic_shapes=["(b,)"]))
    self.assertAllClose(
        9.,
        check_shape_poly(self, jax.jit(f),
                         arg_descriptors=[x], polymorphic_shapes=["(b,)"]))

    res_primal, res_tangent = check_shape_poly(self,
        lambda x, xt: jax.jvp(f, (x,), (xt,)),
        arg_descriptors=[x, np.array([0.1, 0.2, 0.3])],
        polymorphic_shapes=["b", "b"])
    self.assertAllClose((9., 1.8), (res_primal, res_tangent))

    self.assertAllClose(
        np.array([3., 3., 3.]),
        check_shape_poly(self, jax.grad(f),
                         arg_descriptors=[x],
                         polymorphic_shapes=["b"]))

    xv = np.arange(24.).reshape((2, 3, 4))
    res_vmap = jax.vmap(f, in_axes=1)(xv)
    # Implement by iteration
    res_iter = jnp.stack([f(xv[:, i, :]) for i in range(xv.shape[1])])
    self.assertAllClose(res_iter, res_vmap)

    res_vmap_tf = check_shape_poly(self, jax.vmap(f, in_axes=1),
                                   arg_descriptors=[xv],
                                   polymorphic_shapes=["b1, b2, ..."])
    self.assertAllClose(res_iter, res_vmap_tf)

  def test_mean0(self):
    def f_jax(x):  # x: f32[b, 4]
      return jnp.sum(x, axis=0) / x.shape[0]
    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((3, 4), _f32)],
                     polymorphic_shapes=["b, _"],
                     expected_output_signature=tf.TensorSpec([4]))

  def test_shape_as_array(self):
    def f_jax(x):
      # The entire x.shape is passed to jnp.array
      return x + jnp.sum(jnp.array(x.shape)).astype(np.int32)

    check_shape_poly(self,
                     f_jax,
                     arg_descriptors=[RandArg((3, 4), _f32)],
                     polymorphic_shapes=["b, _"])

  def test_dim_as_value_weak_type(self):
    def f_jax(x):  # x: f32[b]
      d0 = jnp.array(x.shape[0])  # in JAX should have weak_type=True
      if isinstance(d0, core.Tracer):
        self.assertTrue(d0.aval.weak_type), d0

      # And an implicit conversion to array
      d1 = x.shape[0] + jnp.array(4)
      if isinstance(d1, core.Tracer):
        self.assertTrue(d1.aval.weak_type), d1
      return d0 + np.array(5., dtype=np.float32) + d1 + x[0]

    with config.numpy_dtype_promotion("strict"):
      # strict type promotion is sensitive to weak_types
      check_shape_poly(self,
                       f_jax,
                       arg_descriptors=[RandArg((3,), _f32)],
                       polymorphic_shapes=["b"])

  def test_vmap_while(self):
    def cond_func(x):  # x: f32[3]
      return jnp.sum(x) >= 0.
    def body_func(x):  # x: f32[3]
      return x - 1.
    def f_jax(x):
      return lax.while_loop(cond_func, body_func, x)

    check_shape_poly(self,
                     jax.vmap(f_jax),
                     arg_descriptors=[RandArg((5, 3), _f32)],
                     polymorphic_shapes=["b, ..."],
                     expected_output_signature=tf.TensorSpec((None, 3), dtype=tf.float32)
                     )

  def test_vmap_error(self):
    # vmap is careful to give nice error messages when mapped axes have
    # different sizes, but this can be foiled by InconsistentDimensionOperation
    x = y = np.ones((3, 5), dtype=np.float32)
    with self.assertRaisesRegex(ValueError,
                                "vmap got inconsistent sizes for array axes to be mapped"):
      jax2tf.convert(jax.vmap(lambda x, y: x + y),
                     polymorphic_shapes=["b, ...", None])(x, y)

    z = x
    with self.assertRaisesRegex(ValueError,
                                "vmap got inconsistent sizes for array axes to be mapped"):
      jax2tf.convert(jax.vmap(lambda x, y, z: x + y + z),
                     polymorphic_shapes=["b, ...", "c, ...", None])(x, y, z)

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
        jax2tf.convert(f_jax, polymorphic_shapes=["b, ..."]),
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

  def test_eval_poly_shapes(self):
    def f1(x, y):  # x: f32[a, 5] y: f[a, 5] -> f32[a, 10]
      return jnp.concatenate([x, y], axis=1)
    def f2(x, z):  # x: f32[a, 5] z: f32[a, 10]
      return jnp.concatenate([x, jax.lax.slice_in_dim(z, 0, 5, axis=1)],
                             axis=1),

    x = np.arange(np.prod((3, 5)), dtype=np.float32).reshape((3, 5))
    y = x

    x_polymorphic_shape = "a, _"
    y_polymorphic_shape = x_polymorphic_shape
    z_spec, z_polymorphic_shape = jax2tf.eval_polymorphic_shape(
        f1,
        polymorphic_shapes=[x_polymorphic_shape, y_polymorphic_shape])(x, y)
    self.assertEqual(np.float32, z_spec.dtype)
    self.assertEqual("(a, 10)", z_polymorphic_shape)

    # We can use the z_polymorphic_shape for jax2tf.convert
    z = jax2tf.convert(
        f1,
        polymorphic_shapes=[x_polymorphic_shape, y_polymorphic_shape])(x, y)
    res = jax2tf.convert(
        f2,
        polymorphic_shapes=[x_polymorphic_shape, z_polymorphic_shape])(x, z)
    self.assertAllClose(f2(x, f1(x, y)), res)

  def test_eval_poly_shapes_tuple_output(self):
    def f1(x, y):  # x: f32[a, 5] y: f[b, 5] -> (f32[a, 5], f32[a + b, 5])
      return (x, jnp.concatenate([x, y], axis=0))
    def f2(z, w):  # z: f32[a, 5] w: f32[a + b, 5] -> f32[2*a + b, 10]
      return jnp.concatenate([z, w], axis=0)
    x = np.arange(np.prod((3, 5)), dtype=np.float32).reshape((3, 5))
    y = np.arange(np.prod((4, 5)), dtype=np.float32).reshape((4, 5))

    x_polymorphic_shape = "a, _"
    y_polymorphic_shape = "b, _"
    zw_specs, zw_polymorphic_shapes = jax2tf.eval_polymorphic_shape(
        f1,
        polymorphic_shapes=[x_polymorphic_shape, y_polymorphic_shape])(x, y)
    self.assertEqual(np.float32, zw_specs[0].dtype)
    self.assertEqual(np.float32, zw_specs[1].dtype)
    self.assertEqual(("(a, 5)", "(b + a, 5)"), zw_polymorphic_shapes)

    # We can use the zw_polymorphic_shapes for jax2tf.convert
    z, w = jax2tf.convert(
        f1,
        polymorphic_shapes=[x_polymorphic_shape, y_polymorphic_shape])(x, y)
    res = jax2tf.convert(f2, polymorphic_shapes=zw_polymorphic_shapes)(z, w)
    self.assertAllClose(f2(* f1(x, y)), res)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
