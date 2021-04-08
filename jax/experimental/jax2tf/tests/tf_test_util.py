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

import contextlib
import logging

from typing import Any, Callable, List, Optional, Sequence


import jax
from jax import dtypes
from jax import numpy as jnp
from jax import test_util as jtu
from jax import tree_util

from jax.config import config
from jax.experimental import jax2tf
from jax.interpreters import masking
from jax._src import util
import numpy as np
import tensorflow as tf  # type: ignore[import]

DType = Any

def _make_tf_args(args):
  def _convert_to_tensor(v):
    if hasattr(v, "dtype"):
      tf.convert_to_tensor(v)
    return v

  return tf.nest.map_structure(_convert_to_tensor, args)


def _make_tf_input_signature(*tf_args) -> List[tf.TensorSpec]:
  # tf_args can be PyTrees
  def _make_one_arg_signature(tf_arg):
    if np.isscalar(tf_arg):
      tf_arg = np.array(tf_arg)
    return tf.TensorSpec(np.shape(tf_arg), tf_arg.dtype)

  return tf.nest.map_structure(_make_one_arg_signature, list(tf_args))


def _run_tf_function(func_tf: Callable, *tf_args, mode: str):
  if mode == "eager":
    return func_tf(*tf_args)  # EAGER
  elif mode == "graph":
    return tf.function(
        func_tf,
        autograph=False,
        input_signature=_make_tf_input_signature(*tf_args))(*tf_args)  # GRAPH
  elif mode == "compiled":
    # Adding an explicit input_signature prevents TF from constant-folding
    # the computation eagerly before compilation
    return tf.function(
        func_tf,
        autograph=False,
        jit_compile=True,
        input_signature=_make_tf_input_signature(*tf_args))(
            *tf_args)  # COMPILED
  else:
    assert False, (
        f"Expected 'eager', 'graph', or 'compiled' for mode: got '{mode}'")


class JaxToTfTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    # Ensure that all TF ops are created on the proper device (TPU or GPU or CPU)
    tf_preferred_devices = (
        tf.config.list_logical_devices("TPU") +
        tf.config.list_logical_devices("GPU") +
        tf.config.list_logical_devices())
    self.tf_default_device = tf_preferred_devices[0]
    logging.info(f"Running jax2tf converted code on {self.tf_default_device}.")
    # We need --config=cuda build flag for TF to see the GPUs
    self.assertEqual(jtu.device_under_test().upper(),
                     self.tf_default_device.device_type)

    with contextlib.ExitStack() as stack:
      stack.enter_context(tf.device(self.tf_default_device))
      self.addCleanup(stack.pop_all().close)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
    """Compares dtypes across JAX and TF dtypes. Overrides super method."""

    def to_numpy_dtype(dt):
      return dt if isinstance(dt, np.dtype) else dt.as_numpy_dtype

    if not config.x64_enabled and canonicalize_dtypes:
      self.assertEqual(
          dtypes.canonicalize_dtype(to_numpy_dtype(jtu._dtype(x))),
          dtypes.canonicalize_dtype(to_numpy_dtype(jtu._dtype(y))))
    else:
      self.assertEqual(
          to_numpy_dtype(jtu._dtype(x)), to_numpy_dtype(jtu._dtype(y)))

  def ConvertAndCompare(self,
                        func_jax: Callable,
                        *args,
                        enable_xla: bool = True,
                        limitations: Sequence = ()):
    """Compares jax_func(*args) with convert(jax_func)(*args).

    It compares the result of JAX, TF ("eager" mode),
    TF with tf.function ("graph" mode), and TF with
    tf.function(jit_compile=True) ("compiled" mode). In each mode,
    either we expect to encounter a known limitation, or the value should
    match the value from the JAX execution.

    Args:
      func_jax: the function to invoke (``func_jax(*args)``)
      args: the arguments.
      enable_xla: if True, allows the use of XLA ops in jax2tf.convert
        (default: True).
      limitations: the set of limitations for this harness (not yet filtered
        by mode).
    """
    # Run JAX. Should not fail, we assume that the harness has been filtered
    # already by JAX unimplemented primitives.
    result_jax = func_jax(*args)  # JAX
    result_tf = None

    func_tf = jax2tf.convert(func_jax, enable_xla=enable_xla)
    tf_args = _make_tf_args(args)

    unexpected_successes: List[str] = []
    # Run the "compiled" mode first, it is most important
    for mode in ("compiled", "eager", "graph"):
      def log_message(extra):
        return f"[{self._testMethodName}] mode={mode}: {extra}"

      jax2tf_limits = tuple(filter(lambda l: l.filter(mode=mode), limitations))

      skip_tf_run = [l for l in jax2tf_limits if l.skip_tf_run]
      if skip_tf_run:
        logging.info(log_message(f"Skip TF run due to limitations {skip_tf_run}"))
        continue

      try:
        result_tf = _run_tf_function(func_tf, *tf_args, mode=mode)
        tf_exception = None
      except Exception as e:
        tf_exception = e

      expect_tf_error = [l for l in jax2tf_limits if l.expect_tf_error]
      if tf_exception:
        if expect_tf_error:
          logging.info(log_message(
            "Found expected TF error with enabled limitations "
            f"{expect_tf_error}; TF error is {tf_exception}"))
          continue
        else:
          raise tf_exception
      else:
        if expect_tf_error:
          # It is more ergonomic to print all successful modes once
          logging.warning(log_message(
            f"Unexpected success with known limitations {expect_tf_error}"))
          unexpected_successes.append(f"{mode}: {expect_tf_error}")

      skip_comparison = [l for l in jax2tf_limits if l.skip_comparison]
      if skip_comparison:
        logging.warning(log_message(f"Skip result comparison due to {skip_comparison}"))
        continue

      max_tol = None
      max_tol_lim = None if not jax2tf_limits else jax2tf_limits[0].get_max_tolerance_limitation(jax2tf_limits)
      if max_tol_lim is not None:
        max_tol = max_tol_lim.tol
        logging.info(log_message(f"Using tol={max_tol} due to {max_tol_lim}"))

      # Convert results to np.arrays
      result_tf = tf.nest.map_structure(lambda t: t.numpy(), result_tf)  # type: ignore

      custom_assert_lim = [l for l in jax2tf_limits if l.custom_assert]
      assert len(custom_assert_lim) <= 1, f"Expecting at most one applicable limitation with custom_assert, found {custom_assert_lim}"

      if custom_assert_lim:
        logging.info(log_message(f"Running custom_assert with tol={max_tol} due to {custom_assert_lim[0]}"))
        custom_assert_lim[0].custom_assert(self, result_jax, result_tf, args=args, tol=max_tol)
      else:
        logging.info(log_message(f"Running default assert with tol={max_tol}"))
        # In compiled mode we expect the same result as JAX by default
        self.assertAllClose(result_jax, result_tf, atol=max_tol, rtol=max_tol)

    # end "for mode"

    if unexpected_successes:
      msg = (f"[{self._testMethodName}] The following are unexpected "
             "successful modes:\n" + "\n".join(unexpected_successes))
      logging.warning(msg)
      # Uncomment the below if you want to see warnings as failures
      # self.assertEmpty(msg)
    return result_jax, result_tf

  def TransformConvertAndCompare(self, func: Callable, arg,
                                 transform: Optional[str]):
    """Like ConvertAndCompare but first applies a transformation.

    `func` must be a function from one argument to one result. `arg` is
    the argument before the transformation.

    `transform` can be None, "jit", "jvp", "grad", "vmap", "jvp_vmap",
    "grad_vmap"
    """
    if transform is None:
      return self.ConvertAndCompare(func, arg)
    if transform == "jit":
      return self.ConvertAndCompare(jax.jit(func), arg)
    if transform == "jvp":
      t_func = lambda x, xt: jax.jvp(func, (x,), (xt,))
      return self.ConvertAndCompare(t_func, arg, np.full_like(arg, 0.1))
    if transform == "grad":
      return self.ConvertAndCompare(jax.grad(func), arg)
    if transform == "vmap":
      t_arg = np.stack([arg] * 4)
      return self.ConvertAndCompare(jax.vmap(func), t_arg)
    if transform == "jvp_vmap":
      jvp_func = lambda x, xt: jax.jvp(jax.vmap(func), (x,), (xt,))
      t_arg = np.stack([arg] * 4)
      return self.ConvertAndCompare(jvp_func, t_arg, np.full_like(t_arg, 0.1))
    if transform == "grad_vmap":
      grad_func = jax.grad(lambda x: jnp.sum(jax.vmap(func)(x)))
      t_arg = np.stack([arg] * 4)
      return self.ConvertAndCompare(grad_func, t_arg)
    assert False, transform

  def CheckShapePolymorphism(self, f_jax: Callable, *,
                             input_signature: Sequence[tf.TensorSpec],
                             polymorphic_shapes: Optional[Sequence[Any]],
                             expected_output_signature: tf.TensorSpec):
    """Convert a function using polymorphic shapes.

    Args:
      f_jax: a JAX function of `n` arguments
      input_signature: used as the input signature for the tf.function.
      in_shapes: if given, it must be a sequence of `n` shape specifications and
        must match the `input_signature`. (see jax2tf.convert).
    """
    f_tf = tf.function(
        jax2tf.convert(f_jax, polymorphic_shapes=polymorphic_shapes),
        autograph=False,
        input_signature=input_signature)
    concrete_f_tf = f_tf.get_concrete_function(*input_signature)
    if expected_output_signature:
      # Strangely, output_shapes can be a single shape for a function with a
      # single result, or a list/tuple of shapes.
      concrete_output_tf_shape = concrete_f_tf.output_shapes
      if not isinstance(concrete_output_tf_shape, (tuple, list)):  # Single result
        assert not isinstance(expected_output_signature, (tuple, list))
        expected_output_signature = [expected_output_signature]
        concrete_output_tf_shape = [concrete_output_tf_shape]

      for expected, found in util.safe_zip(expected_output_signature,
                                           concrete_output_tf_shape):
        self.assertEqual(tuple(expected.shape), tuple(found))
    return f_tf

  def MakeInputSignature(self, *polymorphic_shapes):
    """From a pytree of in_shape string specification, make a pytree of tf.TensorSpec.

    Dimension variables are replaced with None.
    """

    def polymorphic_shape_to_tensorspec(poly_shape: str) -> tf.TensorSpec:
      in_spec = masking.parse_spec(poly_shape)
      return tf.TensorSpec(
          tuple(
              int(dim_spec) if dim_spec.is_constant else None
              for dim_spec in in_spec),
          dtype=tf.float32)

    return tree_util.tree_multimap(polymorphic_shape_to_tensorspec, polymorphic_shapes)
