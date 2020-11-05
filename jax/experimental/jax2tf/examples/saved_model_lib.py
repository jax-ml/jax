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
"""Defines a helper function for creating a SavedModel from the jax2tf trained model.

This has been tested with TensorFlow Hub and TensorFlow model server.
There is very little in this file that is specific to jax2tf. If you
are familiar with how to generate SavedModel, you can most likely use your
own code for this purpose.
"""

from typing import Callable, Sequence, Optional

from jax.experimental import jax2tf # type: ignore[import]
import tensorflow as tf  # type: ignore[import]


def save_model(jax_fn: Callable,
               params,
               model_dir: str,
               *,
               input_signatures: Sequence[tf.TensorSpec],
               shape_polymorphic_input_spec: Optional[str] = None,
               with_gradient: bool = False,
               enable_xla: bool = True,
               compile_model: bool = True):
  """Saves the SavedModel for a function.

  In order to use this wrapper you must first convert your model to a function
  with two arguments: the parameters and the input on which you want to do
  inference. Both arguments may be np.ndarray or
  (nested) tuples/lists/dictionaries thereof.

  If you want to save the model for a function with multiple parameters and
  multiple inputs, you have to collect the parameters and the inputs into
  one argument, e.g., adding a tuple or dictionary at top-level.

  ```
  def jax_fn_multi(param1, param2, input1, input2):
     # JAX model with multiple parameters and multiple inputs. They all can
     # be (nested) tuple/list/dictionaries of np.ndarray.
     ...

  def jax_fn_for_save_model(params, inputs):
     # JAX model with parameters and inputs collected in a tuple each. We can
     # use dictionaries also (in which case the keys would appear as the names
     # of the inputs)
     param1, param2 = params
     input1, input2 = inputs
     return jax_fn_multi(param1, param2, input1, input2)
  save_model(jax_fn_for_save_model, (param1, param2), ...)
  ```

  See examples in mnist_lib.py and saved_model.py.

  Args:
    jax_fn: a JAX function taking two arguments, the parameters and the inputs.
      Both arguments may be (nested) tuples/lists/dictionaries of np.ndarray.
    params: the parameters, to be used as first argument for `jax_fn`. These
      must be (nested) tuples/lists/dictionaries of np.ndarray, and will be
      saved as the variables of the SavedModel.
    model_dir: the directory where the model should be saved.
    input_signatures: the input signatures for the second argument of `jax_fn`
      (the input). A signature must be a `tensorflow.TensorSpec` instance, or a
      (nested) tuple/list/dictionary thereof with a structure matching the
      second argument of `jax_fn`. The first input_signature will be saved as
      the default serving signature. The additional signatures will be used
      only to ensure that the `jax_fn` is traced and converted to TF for the
      corresponding input shapes.
    shape_polymorphic_input_spec: if given then it will be used as the
      `in_shapes` argument to jax2tf.convert for the second parameter of
      `jax_fn`. In this case, a single `input_signatures` is supported, and
      should have `None` in the polymorphic dimensions. Should be a string, or a
      (nesteD) tuple/list/dictionary thereof with a structure matching the
      second argument of `jax_fn`.
    with_gradient: whether the SavedModel should support gradients. If True,
      then a custom gradient is saved. If False, then a
      tf.raw_ops.PreventGradient is saved to error if a gradient is attempted.
      (At the moment due to a bug in SavedModel, custom gradients are not
      supported.)
    enable_xla: whether the jax2tf converter is allowed to use TFXLA ops. If
      False, the conversion tries harder to use purely TF ops and raises an
      exception if it is not possible. (default: True)
    compile_model: use TensorFlow experimental_compiler on the SavedModel. This
      is needed if the SavedModel will be used for TensorFlow serving.
  """
  if not input_signatures:
    raise ValueError("At least one input_signature must be given")
  if shape_polymorphic_input_spec is not None:
    if len(input_signatures) > 1:
      raise ValueError("For shape-polymorphic conversion a single "
                       "input_signature is supported.")
  tf_fn = jax2tf.convert(
      jax_fn,
      with_gradient=with_gradient,
      in_shapes=[None, shape_polymorphic_input_spec],
      enable_xla=enable_xla)

  # Create tf.Variables for the parameters.
  param_vars = tf.nest.map_structure(
      # If with_gradient=False, we mark the variables behind as non-trainable,
      # to ensure that users of the SavedModel will not try to fine tune them.
      lambda param: tf.Variable(param, trainable=with_gradient),
      params)
  tf_graph = tf.function(lambda inputs: tf_fn(param_vars, inputs),
                         autograph=False,
                         experimental_compile=compile_model)

  signatures = {}
  signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
    tf_graph.get_concrete_function(input_signatures[0])
  for input_signature in input_signatures[1:]:
    # If there are more signatures, trace and cache a TF function for each one
    tf_graph.get_concrete_function(input_signature)
  wrapper = _ReusableSavedModelWrapper(tf_graph, param_vars)
  tf.saved_model.save(wrapper, model_dir, signatures=signatures)


class _ReusableSavedModelWrapper(tf.train.Checkpoint):
  """Wraps a function and its parameters for saving to a SavedModel.

  Implements the interface described at
  https://www.tensorflow.org/hub/reusable_saved_models.
  """

  def __init__(self, tf_graph, param_vars):
    """Args:

      tf_fn: a tf.function taking one argument (the inputs), which can be
         be tuples/lists/dictionaries of np.ndarray or tensors.
      params: the parameters, as tuples/lists/dictionaries of tf.Variable,
         and will be saved as the variables of the SavedModel.
    """
    super().__init__()
    # Implement the interface from https://www.tensorflow.org/hub/reusable_saved_models
    self.variables = tf.nest.flatten(param_vars)
    self.trainable_variables = [v for v in self.variables if v.trainable]
    # If you intend to prescribe regularization terms for users of the model,
    # add them as @tf.functions with no inputs to this list. Else drop this.
    self.regularization_losses = []
    self.__call__ = tf_graph
