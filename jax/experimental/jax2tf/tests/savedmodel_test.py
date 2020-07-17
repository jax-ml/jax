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

import os

from absl.testing import absltest

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf  # type: ignore[import]

from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util
from jax import test_util as jtu

from jax.config import config
config.parse_flags_with_absl()


class SavedModelTest(tf_test_util.JaxToTfTestCase):

  def save_and_load_model(self, model: tf.Module) -> tf.Module:
    # Roundtrip through saved model on disk.
    model_dir = os.path.join(absltest.get_default_test_tmpdir(), str(id(model)))
    tf.saved_model.save(model, model_dir)
    restored_model = tf.saved_model.load(model_dir)
    return restored_model

  def test_eval(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    model = tf.Module()
    model.f = tf.function(jax2tf.convert(f_jax),
                          autograph=False,
                          input_signature=[tf.TensorSpec([], tf.float32)])
    x = np.array(0.7)
    self.assertAllClose(model.f(x), f_jax(x))
    restored_model = self.save_and_load_model(model)
    self.assertAllClose(restored_model.f(x), f_jax(x))

  def test_gradient_disabled(self):
    f_jax = lambda x: x * x
    model = tf.Module()
    model.f = tf.function(jax2tf.convert(f_jax),
                          autograph=False,
                          input_signature=[tf.TensorSpec([], tf.float32)])
    x = np.array(0.7)
    self.assertAllClose(model.f(x), f_jax(x))
    restored_model = self.save_and_load_model(model)
    xv = tf.Variable(0.7)
    self.assertAllClose(restored_model.f(x), f_jax(x))

    with self.assertRaisesRegex(LookupError,
                                "Gradient explicitly disabled.*The jax2tf-converted function does not support gradients"):
      with tf.GradientTape():
        _ = restored_model.f(xv)

  def test_gradient(self):
    """Save and restore the custom gradient."""
    @jax.custom_jvp
    def f_jax(x):
      return x * x

    @f_jax.defjvp
    def f_jax_jvp(primals, tangents):
      # 3 * x * x_t
      x, = primals
      x_dot, = tangents
      primal_out = f_jax(x)
      tangent_out = 3. * x * x_dot
      return primal_out, tangent_out

    model = tf.Module()
    model.f = tf.function(jax2tf.convert(f_jax, with_gradient=True),
                          autograph=False,
                          input_signature=[tf.TensorSpec([], tf.float32)])
    x = np.array(0.7)
    self.assertAllClose(model.f(x), f_jax(x))
    restored_model = self.save_and_load_model(model)
    xv = tf.Variable(0.7)
    self.assertAllClose(restored_model.f(x), f_jax(x))
    with tf.GradientTape() as tape:
      y = restored_model.f(xv)

    # TODO: at the moment TF does not fully-support custom_gradient in a
    # savedmodel (b/123499169), but at least a warning is printed and an
    # exception is thrown when gradients are taken. The exception, however,
    # is a very strange one, for now.
    with self.assertRaisesRegex(TypeError, "An op outside of the function building code is being passed"):
      _ = tape.gradient(y, xv)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
