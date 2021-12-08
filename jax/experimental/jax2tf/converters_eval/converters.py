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
"""Converters for jax2tf."""
import functools
import numpy as np
import tempfile
from typing import Any, Callable, Tuple

from jax._src import dtypes
import jax.numpy as jnp
from jax.experimental import jax2tf
from jax.experimental.jax2tf.examples import saved_model_lib
from jax.experimental.jax2tf.converters_eval import converters_eval_lib as lib
import tensorflow as tf
from tensorflowjs.converters import converter as tfjs_converter

Array = Any
TempDir = tempfile.TemporaryDirectory


def _jax2tf(jax_fn: Callable[..., Any], input_shape: Tuple[int, ...], dtype: Any, *, enable_xla: bool = True):
  """Converts the given `jax_fn` to TF using jax2tf and returns `(tf_fn, concrete_fn)`."""
  tf_fn = tf.function(
        jax2tf.convert(jax_fn, enable_xla=enable_xla),
        input_signature=[
            tf.TensorSpec(
                shape=input_shape,
                dtype=dtype,
                name='input')
        ],
        autograph=False)
  concrete_fn = tf_fn.get_concrete_function()
  return tf_fn, concrete_fn


def _get_random_data(dtype: jnp.dtype, shape: Tuple[int, ...], seed=0) -> Any:
  dtype = dtypes.canonicalize_dtype(dtype)
  np.random.seed(seed)
  # Adjust the max values of the numbers based on the seed, so different seeds
  # result in different ranges.
  max_value = max(1, 100*seed)
  if np.issubdtype(dtype, np.integer):
    return np.random.randint(0, max_value, size=shape, dtype=dtype)
  elif np.issubdtype(dtype, np.floating):
    return np.array(np.random.uniform(size=shape), dtype=dtype) * max_value
  elif dtype == np.bool:
    return np.random.choice(a=[False, True], size=shape)
  else:
    raise ValueError(f"Unsupported dtype for numerical comparison: {dtype}")


def _compare(jax_fn: Callable[..., Any],
             tf_fn: Callable[..., Any],
             module: lib.ModuleToConvert,
             comparison: str,
             nr_runs: int = 5,
             rtol: float = 1e-05):
  for i in range(nr_runs):
    input_data = _get_random_data(module.dtype, module.input_shape, seed=i)
    # A function may return multiple arrays, which may be of different shapes.
    # We can't just input the tuple into `np.allclose` since it will cast the
    # tuple to an array, which will break if the shapes in the tuple are
    # different. Therefore we iterate over the tuple explicitly.
    wrap_tuple = lambda x: (x,) if not isinstance(x, tuple) else x
    jax_results = wrap_tuple(jax_fn(input_data))
    tf_results = wrap_tuple(tf_fn(input_data))

    if len(tf_results) != len(jax_results):
      raise ValueError(f"For {comparison}: returned output tuples lengths do not"
          f"match: TF length vs JAX length: {len(tf_results)} != "
          f"{len(jax_results)}")

    for jax_result, tf_result in zip(jax_results, tf_results):
      if not np.allclose(jax_result, tf_result, rtol):
        raise ValueError(f"For {comparison}: Numerical difference "
                          f"jax_result={jax_result} vs "
                          f"tf_result={tf_result}")
      # TFLite doesn't allow existing references to its data when it is
      # invoked. We therefore delete TF results so we can run multiple inputs on
      # the same interpreter.
      del tf_result
    del tf_results


def jax2tf_xla(module: lib.ModuleToConvert):
  """Converts the given `module` using the jax2tf emitter with enable_xla=True."""
  apply = functools.partial(module.apply, module.variables)
  _, apply_tf = _jax2tf(apply, module.input_shape, module.dtype)
  _compare(apply, apply_tf, module, "JAX vs TF (enable_xla=True)")


def jax2tf_to_tfjs(module: lib.ModuleToConvert):
  """Converts the given `module` using the TFjs converter."""
  with TempDir() as saved_model_path, TempDir() as converted_model_path:
    # the model must be converted with with_gradient set to True to be able to
    # convert the saved model to TF.js, as "PreventGradient" is not supported
    saved_model_lib.convert_and_save_model(
        module.apply,
        module.variables,
        saved_model_path,
        input_signatures=[
            tf.TensorSpec(
                shape=module.input_shape,
                dtype=module.dtype,
                name='input')
        ],
        with_gradient=True,
        compile_model=False,
        enable_xla=False
    )
    tfjs_converter.convert([saved_model_path, converted_model_path])

  # TODO(marcvanzee): Add numerical comparison for TFjs as well.


def jax2tf_to_tflite(module: lib.ModuleToConvert):
  """Converts the given `module` using the TFLite converter."""
  apply = functools.partial(module.apply, module.variables)
  tf_fn, apply_tf = _jax2tf(apply, module.input_shape, module.dtype, enable_xla=False)

  # First compare JAX output with TF output.
  _compare(apply, apply_tf, module, "JAX vs TF (enable_xla=False)")

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

  # We assume a single input, but we allows multiple outputs.
  inputs = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()
  outputs = tuple(interpreter.tensor(out["index"]) for out in output_details)

  def apply_tflite(input_data):
    interpreter.set_tensor(inputs['index'], input_data)
    interpreter.invoke()
    if len(outputs) > 1:
      return tuple(o() for o in outputs)
    else:
      return outputs[0]()

  _compare(apply, apply_tflite, module, "JAX vs TFLite")
