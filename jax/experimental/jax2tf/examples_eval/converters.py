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
from typing import Any, Tuple

from jax._src import dtypes
import jax.numpy as jnp
from jax.experimental import jax2tf
from jax.experimental.jax2tf.examples import saved_model_lib
from jax.experimental.jax2tf.examples_eval import examples_converter
import tensorflow as tf
from tensorflowjs.converters import converter as tfjs_converter

TempDir = tempfile.TemporaryDirectory


def jax2tf_to_tfjs(module: examples_converter.ModuleToConvert):
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


def _get_random_data(dtype: jnp.dtype, shape: Tuple[int, ...]) -> Any:
  dtype = dtypes.canonicalize_dtype(dtype)
  if np.issubdtype(dtype, np.integer):
    return np.random.randint(0, 100, size=shape, dtype=dtype)
  elif np.issubdtype(dtype, np.floating):
    return np.array(np.random.uniform(size=shape), dtype=dtype) * 100
  elif dtype == np.bool:
    return np.random.choice(a=[False, True], size=shape)
  else:
    raise ValueError(f"Unsupported dtype for numerical comparison: {dtype}")


def _all_close(jax_results, tf_results, tf_description, rtol=1e-05):
  if len(tf_results) != len(jax_results):
    raise ValueError(f"Numerical difference: returned output tuples lengths do "
        f"not match: {tf_description} length vs JAX length: "
        f"{len(tf_results)} != {len(jax_results)}")

  for jax_result, tf_result in zip(jax_results, tf_results):
    for jax_array, tf_array in zip(jax_result, tf_result):
      jax_array = np.asarray(jax_array)
      tf_array = np.asarray(tf_array)
      if not np.allclose(jax_array, tf_array, rtol):
        raise ValueError(f"Numerical difference JAX vs {tf_description}: "
                         f"JAX result={jax_result} vs "
                         f"{tf_description} result={tf_result}")


def jax2tf_to_tflite(module: examples_converter.ModuleToConvert):
  """Converts the given `module` using the TFLite converter."""
  apply = functools.partial(module.apply, module.variables)
  tf_predict = tf.function(
      jax2tf.convert(apply, enable_xla=False),
      input_signature=[
          tf.TensorSpec(
              shape=module.input_shape,
              dtype=module.dtype,
              name='input')
      ],
      autograph=False)
  tf_predict_concrete = tf_predict.get_concrete_function()

  input_data = _get_random_data(module.dtype, module.input_shape)

  # First compare JAX output with TF output.
  jax_results = apply(input_data)
  _all_close(jax_results, tf_predict_concrete(input_data), "TF")

  # Convert TF function to TF Lite format.
  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [tf_predict_concrete], tf_predict)
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

  interpreter.set_tensor(inputs['index'], input_data)
  interpreter.invoke()
  tflite_results = tuple(output() for output in outputs)

  if len(tflite_results) == 1:
    # If we only have one result, don't use a tuple since JAX also doesn't.
    tflite_results = tflite_results[0]

  _all_close(jax_results, tflite_results, "TFLite")
