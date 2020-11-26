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

import atexit
import contextlib
import logging
import numpy as np
from typing import Any, Callable, List, Optional, Sequence, Tuple
import tensorflow as tf  # type: ignore[import]

import jax
from jax.config import config
from jax import dtypes
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import correctness_stats
from jax.interpreters import masking
from jax import test_util as jtu
from jax import tree_util
from jax import numpy as jnp

import os


if os.getenv('JAX2TF_OUTPUT_LIMITATIONS') is not None:
  output_file = os.path.join(os.path.dirname(__file__),
                             '../g3doc/primitives_with_limited_support.md')
  template_file = os.path.join(os.path.dirname(__file__),
                               '../g3doc/primitives_with_limited_support.md.template')
  atexit.register(correctness_stats.pprint_all_limitations,
                  output_file, template_file)

class JaxToTfTestCase(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    # Ensure that all TF ops are created on the proper device (TPU or GPU or CPU)
    # TODO(necula): why doesn't TF do this automatically?
    tf_preferred_devices = (
        tf.config.list_logical_devices("TPU") +
        tf.config.list_logical_devices("GPU") +
        tf.config.list_logical_devices())
    self.tf_default_device = tf_preferred_devices[0]
    logging.info(f"Running jax2tf converted code on {self.tf_default_device}.")
    if jtu.device_under_test() != "gpu":
      # TODO(necula): Change the build flags to ensure the GPU is seen by TF
      # It seems that we need --config=cuda build flag for this to work?
      self.assertEqual(jtu.device_under_test().upper(),
                       self.tf_default_device.device_type)

    with contextlib.ExitStack() as stack:
      stack.enter_context(tf.device(self.tf_default_device))
      self.addCleanup(stack.pop_all().close)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
    """Compares dtypes across JAX and TF dtypes. Overrides super method."""
    def to_numpy_dtype(dt):
      return dt if isinstance(dt, np.dtype) else dt.as_numpy_dtype

    if not config.FLAGS.jax_enable_x64 and canonicalize_dtypes:
      self.assertEqual(dtypes.canonicalize_dtype(to_numpy_dtype(jtu._dtype(x))),
                       dtypes.canonicalize_dtype(to_numpy_dtype(jtu._dtype(y))))
    else:
      self.assertEqual(to_numpy_dtype(jtu._dtype(x)),
                       to_numpy_dtype(jtu._dtype(y)))

  def ConvertAndCompare(self, func_jax: Callable, *args,
                        custom_assert: Optional[Callable] = None,
                        always_custom_assert: bool = False,
                        expect_tf_exceptions: bool = False,
                        enable_xla: bool = True,
                        atol=None,
                        rtol=None) -> Tuple[Any, Any]:
    """Compares jax_func(*args) with convert(jax_func)(*args).

    It compares the result of JAX, TF ("eager" mode),
    TF with tf.function ("graph" mode), and TF with
    tf.function(experimental_compile=True) ("compiled" mode). In each mode,
    either we expect an exception (see `expect_tf_exceptions`) or the value
    should match the value from the JAX execution.

    Args:
      custom_assert: a function that will be called
        `custom_assert(result_jax, result_tf)` to assert equality of the
        results. Use this function when JAX and TF produce different results.
        This function is only used for "eager" and "graph" modes by default, not
        for the "compiled" mode, because in that case we expect the results to
        be equal (default: None).
      always_custom_assert: if True, custom_assert is also called in "compiled"
        mode. This is useful in cases where JAX and TF produce different but
        equally valid results (default: False).
      expect_tf_exceptions: if True, there may be exceptions in some evaluation
        modes; when there is no exception the result should be the same
        as in JAX (default: False).
      enable_xla: if True, allows the use of XLA ops in jax2tf.convert
        (default: True).
    """
    original_impl = jax2tf.jax2tf.TensorFlowTrace.get_primitive_impl

    # Monkey-patch jax2tf.TensorFlowTrace.get_primitive_impl to wrap the
    # resulting primitive in a categorizer.
    def _new_get_primitive_impl(s, p):
      impl, impl_needs_avals = original_impl(s, p)
      return correctness_stats.collect_limitations(p, impl), impl_needs_avals
    jax2tf.jax2tf.TensorFlowTrace.get_primitive_impl = _new_get_primitive_impl  # type: ignore

    def restore_get_primitive_impl():
      jax2tf.jax2tf.TensorFlowTrace.get_primitive_impl = original_impl

    # Restore the original jax2tf.TensorFlowTrace.get_primitive_impl
    # implementation at the end of the test.
    self.addCleanup(restore_get_primitive_impl)

    # Run JAX
    result_jax = func_jax(*args)
    # Run TF in all execution modes
    func_tf = jax2tf.convert(func_jax, enable_xla=enable_xla)

    def convert_if_bfloat16(v):
      if hasattr(v, "dtype"):
        return tf.convert_to_tensor(np.array(v, jnp.float32) if
                                      v.dtype == jnp.bfloat16 else v,
                                    jax2tf.jax2tf.to_tf_dtype(v.dtype))
      return v

    tf_args = tf.nest.map_structure(convert_if_bfloat16, args)

    def make_input_signature(*tf_args) -> List[tf.TensorSpec]:
      # tf_args can be PyTrees
      def make_one_arg_signature(tf_arg):
        return tf.TensorSpec(np.shape(tf_arg), tf_arg.dtype)
      return tf.nest.map_structure(make_one_arg_signature, list(tf_args))

    def run_tf(mode):
      if mode == "eager":
        return func_tf(*tf_args)
      elif mode == "graph":
        return tf.function(
          func_tf, autograph=False,
          input_signature=make_input_signature(*tf_args))(*tf_args)
      elif mode == "compiled":
        # Adding an explicit input_signature prevents TF from constant-folding
        # the computation eagerly before compilation
        return tf.function(
          func_tf, autograph=False,
          experimental_compile=True,
          input_signature=make_input_signature(*tf_args))(*tf_args)
      else:
        assert False

    def expected_missing_tf_support(lim: correctness_stats.Limitation):
      return (lim.error_type == correctness_stats.CATEGORY_MISSING_TF_SUPPORT and
              self.tf_default_device.device_type in lim.devices)
    def expected_possible_incorrect(lim: correctness_stats.Limitation):
      return (lim.error_type == correctness_stats.CATEGORY_POSSIBLE_INCORRECT_RESULTS and
              self.tf_default_device.device_type in lim.devices)

    result_tf = None
    for mode in ("eager", "graph", "compiled"):
      current_limitations_len = len(correctness_stats.all_limitations)
      try:
        result_tf = run_tf(mode)
        tf_exception = None
      except Exception as e:
        tf_exception = e

      new_limitations = (
        correctness_stats.all_limitations[current_limitations_len:])
      if new_limitations:
        for lim in new_limitations:
          print("Detected limitation: {} for {} devices."
                .format(lim.error_string, ', '.join(lim.devices)))

      if any(map(expected_missing_tf_support, new_limitations)) or expect_tf_exceptions:
        if tf_exception is not None:
          print(f"Encountered expected exception for mode={mode}: {tf_exception}")
          continue
        else:
          print(f"WARNING: did not encounter expected exception for mode={mode}")
      else:
        if tf_exception is not None:
          raise tf_exception

      if custom_assert is not None and (mode in ("eager", "graph") or
                                        always_custom_assert):
        # If we have a custom assert, use it even if we expect incorrect results
        custom_assert(result_jax, result_tf)
      else:
        try:
          # In compiled mode we expect the same result as JAX by default
          self.assertAllClose(result_jax, result_tf, atol=atol, rtol=rtol)
          check_failure = None
        except Exception as e:
          check_failure = e

        if any(map(expected_possible_incorrect, new_limitations)):
          if check_failure is not None:
            print(f"Encountered expected result check failure for mode={mode}: {check_failure}")
            continue
          else:
            print(f"WARNING: did not encounter expected result check failure for mode={mode}")
        else:
          if check_failure is not None:
            raise check_failure

    return (result_jax, result_tf)

  def TransformConvertAndCompare(self, func: Callable,
                                 arg,
                                 transform: Optional[str]):
    """Like ConvertAndCompare but first applies a transformation.

    `func` must be a function from one argument to one result. `arg` is
    the argument before the transformation.

    `transform` can be None, "jit", "jvp", "grad", "vmap", "jvp_vmap", "grad_vmap"
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
      return self.ConvertAndCompare(jvp_func, t_arg,
                                    np.full_like(t_arg, 0.1))
    if transform == "grad_vmap":
      grad_func = jax.grad(lambda x: jnp.sum(jax.vmap(func)(x)))
      t_arg = np.stack([arg] * 4)
      return self.ConvertAndCompare(grad_func, t_arg)
    assert False, transform

  def CheckShapePolymorphism(self, f_jax: Callable, *,
                             input_signature: Sequence[tf.TensorSpec],
                             in_shapes: Optional[Sequence[Any]],
                             expected_output_signature: tf.TensorSpec):
    """Convert a function using polymorphic shapes.

    Args:
      f_jax: a JAX function of `n` arguments
      input_signature: used as the input signature
        for the tf.function.
      in_shapes: if given, it must be a sequence of `n` shape specifications
        and must match the `input_signature`. (see jax2tf.convert).
    """
    f_tf = tf.function(jax2tf.convert(f_jax, in_shapes=in_shapes),
                       autograph=False,
                       input_signature=input_signature)
    concrete_f_tf = f_tf.get_concrete_function(*input_signature)
    if expected_output_signature:
      concrete_output_tf_shape = concrete_f_tf.output_shapes
      assert not isinstance(concrete_output_tf_shape, tuple)  # A single result
      self.assertEqual(tuple(expected_output_signature.shape),
                       tuple(concrete_output_tf_shape))
    return f_tf

  def MakeInputSignature(self, *in_shapes):
    """From a pytree of in_shape string specification, make a pytree of tf.TensorSpec.
    Dimension variables are replaced with None.
    """
    def in_shape_to_tensorspec(in_shape: str) -> tf.TensorSpec:
      in_spec = masking.parse_spec(in_shape)
      return tf.TensorSpec(tuple(int(dim_spec) if dim_spec.is_constant else None
                                 for dim_spec in in_spec), dtype=tf.float32)

    return tree_util.tree_multimap(in_shape_to_tensorspec, in_shapes)
