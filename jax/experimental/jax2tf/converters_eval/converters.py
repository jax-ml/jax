# Copyright 2022 Google LLC
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
"""Converters for jax2tf."""
import functools
import numpy as np
import tempfile
from typing import Any, Callable

from jax._src import dtypes
import jax.numpy as jnp
from jax.experimental import jax2tf
from jax.experimental.jax2tf.converters_eval.models import ModelTestCase
import tensorflow as tf
from tensorflowjs.converters import converter as tfjs_converter

Array = Any
TempDir = tempfile.TemporaryDirectory

DEFAULT_RTOL = 1e-05


class _ReusableSavedModelWrapper(tf.train.Checkpoint):
  """Wraps a function and its parameters for saving to a SavedModel.

  Implements the interface described at
  https://www.tensorflow.org/hub/reusable_saved_models.
  """

  def __init__(self, tf_graph, param_vars):
    """Args:

      tf_graph: a tf.function taking one argument (the inputs), which can be
         be tuples/lists/dictionaries of np.ndarray or tensors. The function
         may have references to the tf.Variables in `param_vars`.
      param_vars: the parameters, as tuples/lists/dictionaries of tf.Variable,
         to be saved as the variables of the SavedModel.
    """
    super().__init__()
    # Implement the interface from https://www.tensorflow.org/hub/reusable_saved_models
    self.variables = tf.nest.flatten(param_vars)
    self.trainable_variables = [v for v in self.variables if v.trainable]
    # If you intend to prescribe regularization terms for users of the model,
    # add them as @tf.functions with no inputs to this list. Else drop this.
    self.regularization_losses = []
    self.__call__ = tf_graph


def _get_signatures(input_specs):

  def _get_signature(tf_arg):
    return tf.TensorSpec(np.shape(tf_arg), jax2tf.dtype_of_val(tf_arg))

  return tf.nest.map_structure(_get_signature, input_specs)


def _get_random_data(x: jnp.ndarray) -> Any:
  dtype = dtypes.canonicalize_dtype(x.dtype)
  if np.issubdtype(dtype, np.integer):
    return np.random.randint(0, 100, size=x.shape, dtype=dtype)
  elif np.issubdtype(dtype, np.floating):
    return np.array(np.random.uniform(size=x.shape), dtype=dtype)
  elif dtype == np.bool:
    return np.random.choice(a=[False, True], size=x.shape)
  else:
    raise ValueError(f"Unsupported dtype for numerical comparison: {dtype}")


def _compare(test_case: ModelTestCase, jax_fn: Callable[..., Any],
             tf_fn: Callable[..., Any], comparison: str):
  xs = [_get_random_data(x) for x in test_case.input_specs]
  # A function may return multiple arrays, which may be of different shapes.
  # We can't just input the tuple into `np.allclose` since it will cast the
  # tuple to an array, which will break if the shapes in the tuple are
  # different. Therefore we iterate over the tuple explicitly.
  wrap_tuple = lambda x: (x,) if not isinstance(x, tuple) else x
  jax_results = wrap_tuple(jax_fn(*xs))
  tf_results = wrap_tuple(tf_fn(*xs))

  if len(tf_results) != len(jax_results):
    raise ValueError(f"For {comparison}: returned output tuples lengths do not"
                     f"match: TF length vs JAX length: {len(tf_results)} != "
                     f"{len(jax_results)}")

  for jax_result, tf_result in zip(jax_results, tf_results):
    np.testing.assert_allclose(jax_result, tf_result, test_case.rtol)
    # TFLite doesn't allow existing references to its data when it is
    # invoked. We therefore delete TF results so we can run multiple inputs on
    # the same interpreter.
    del tf_result
  del tf_results


def jax2tf_xla(test_case: ModelTestCase):
  """Converts the given `module` using the jax2tf emitter with enable_xla=True."""
  jax_fn = functools.partial(test_case.apply, test_case.variables)
  tf_fn = jax2tf.convert(jax_fn, enable_xla=True)
  _compare(test_case, jax_fn, tf_fn, "JAX vs TF (enable_xla=True)")


def jax2tf_to_tfjs(test_case: ModelTestCase):
  """Converts the given `module` using the TFjs converter."""
  # the model must be converted with with_gradient set to True to be able to
  # convert the saved model to TF.js, as "PreventGradient" is not supported.
  tf_fn = jax2tf.convert(test_case.apply, with_gradient=True,
                         enable_xla=False)

  # Create tf.Variables for the parameters. If you want more useful variable
  # names, you can use `tree.map_structure_with_path` from the `dm-tree`
  # package.
  param_vars = tf.nest.map_structure(
    lambda param: tf.Variable(param, trainable=True),
    test_case.variables)

  # This is the function that will be stored in the SavedModel. Note this only
  # supports a single argument, but we'd like to be able to pass more
  # arguments to a Module's __call__ function, so we pass them as a list that
  # we expand when passing it to `tf_fn`.
  @tf.function(autograph=False, jit_compile=False)
  def tf_graph(inputs):
    return tf_fn(param_vars, *inputs)

  s_fn = tf_graph.get_concrete_function(_get_signatures(test_case.input_specs))
  signatures = {tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: s_fn}
  wrapper = _ReusableSavedModelWrapper(tf_graph, param_vars)
  saved_model_options = tf.saved_model.SaveOptions(experimental_custom_gradients=True)

  with TempDir() as saved_model_path, TempDir() as tfjs_model_path:
    tf.saved_model.save(wrapper, saved_model_path, signatures=signatures,
                        options=saved_model_options)
    tfjs_converter.convert([saved_model_path, tfjs_model_path])


def jax2tf_to_tflite(test_case: ModelTestCase):
  """Converts the given `module` using the TFLite converter."""
  apply = functools.partial(test_case.apply, test_case.variables)
  tf_fn = tf.function(
      jax2tf.convert(apply, enable_xla=False),
      input_signature=_get_signatures(test_case.input_specs),
      autograph=False)
  apply_tf = tf_fn.get_concrete_function()

  # First compare JAX output with TF output.
  _compare(test_case, apply, apply_tf, "JAX vs TF (enable_xla=False)")

  # Convert TF function to TF Lite format.
  converter = tf.lite.TFLiteConverter.from_concrete_functions([apply_tf], tf_fn)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
  ]
  # Convert the model.
  tflite_model = converter.convert()

  # Construct an interpreter for doing a numerical comparison.
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  inputs = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  outputs = tuple(interpreter.tensor(out["index"]) for out in output_details)

  def apply_tflite(*xs):
    assert len(xs) == len(inputs)
    for i, x in enumerate(xs):
      interpreter.set_tensor(inputs[i]['index'], x)
    interpreter.invoke()
    if len(outputs) > 1:
      return tuple(o() for o in outputs)
    else:
      return outputs[0]()

  _compare(test_case, apply, apply_tflite, "JAX vs TFLite")
