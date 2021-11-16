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
import tempfile

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
      [tf_predict.get_concrete_function()], tf_predict)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
  ]
  converter.convert()
