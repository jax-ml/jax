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
"""Example of converting a Haiku module to a Sonnet module."""

import os
import sys

from absl import app
import jax
from jax.experimental import jax_to_tf

import tensorflow as tf

from jax.config import config
config.config_with_absl()

# Import ../../../../examples/resnet50.py
def from_examples_import_resnet50():
  this_dir = os.path.dirname(os.path.abspath(__file__))
  examples_dir = os.path.abspath(os.path.join(this_dir, "..", "..",
                                              "..", "..", "examples"))
  assert os.path.isfile(os.path.join(examples_dir, "resnet50.py"))
  try:
    sys.path.append(examples_dir)
    import resnet50  # type: ignore[import-error]
    return resnet50
  finally:
    sys.path.pop()

# The next line is rewritten on copybara import.
resnet50 = from_examples_import_resnet50()


class StaxModule(tf.Module):
  """Wraps a JAX function as a tf.Module."""

  def __init__(self, apply_fn, params, name=None):
    super().__init__(name=name)
    self.apply_fn = jax_to_tf.convert(apply_fn)
    self.params = tf.nest.map_structure(tf.Variable, params)

  @tf.function(autograph=False)
  def __call__(self, *a, **k):
    return self.apply_fn(self.params, *a, **k)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  init_fn, apply_fn = resnet50.ResNet50(1000)

  # Initialize the network.
  rng = jax.random.PRNGKey(42)
  input_shape = (224, 224, 3, 1)
  _, params = init_fn(rng, input_shape)

  # Sanity check our JAX model.
  # Note: We expect 161 parameters here but stax does not support optional bias
  # in conv so we have 53 additional biases.
  assert len(jax.tree_leaves(params)) == 214

  # We can use JaxModule to wrap our STAX network in TensorFlow.
  mod = StaxModule(apply_fn, params)
  assert len(mod.trainable_variables) == 214
  assert mod(tf.ones(input_shape)).shape == (1, 1000)

if __name__ == "__main__":
  app.run(main)
