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
    return np.array(np.random.uniform(size=shape), dtype=dtype)
  elif dtype == np.bool:
    return np.random.choice(a=[False, True], size=shape)
  else:
    raise ValueError(f"Unsupported dtype for numerical comparison: {dtype}")


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

  # Convert TF function to TF Lite format.
  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [tf_predict.get_concrete_function()])
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

  # Generate random data and get outputs from TFLite and JAX.
  input_data = _get_random_data(module.dtype, module.input_shape)
  interpreter.set_tensor(inputs['index'], input_data)
  interpreter.invoke()
  tflite_results = tuple(output() for output in outputs)
  jax_results = apply(input_data)

  # If the model returns a single value, put the JAX return value in a tuple so
  # we can compare it with the TFLite output, which is always a tuple.
  if len(tflite_results) == 1:
    jax_results = (jax_results,)

  if len(tflite_results) != len(jax_results):
    raise ValueError(f"Numerical difference: returned output tuples lengths do "
        f"not match: TFLite length vs JAX length: {len(tflite_results)} != "
        f"{len(jax_results)}")

  for jax_result, tflite_result in zip(jax_results, tflite_results):
    for jax_array, tflite_array in zip(jax_result, tflite_result):
      jax_array = np.asarray(jax_array)
      tflite_array = np.asarray(tflite_array)
      if not np.allclose(jax_array, tflite_array, 1e-05):
        raise ValueError(f"Numerical difference: jax_result={jax_result} vs "
                        f"tflite_result={tflite_result}")
