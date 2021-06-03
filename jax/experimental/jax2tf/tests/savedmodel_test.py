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
from jax import lax
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
    tf.saved_model.save(
        model, model_dir,
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
    restored_model = tf.saved_model.load(model_dir)
    return restored_model

  def test_eval(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    model = tf.Module()
    model.f = tf.function(jax2tf.convert(f_jax),
                          autograph=False,
                          input_signature=[tf.TensorSpec([], tf.float32)]
                          )
    x = np.array(0.7, dtype=jnp.float32)
    self.assertAllClose(model.f(x), f_jax(x))
    restored_model = self.save_and_load_model(model)
    self.assertAllClose(restored_model.f(x), f_jax(x))

  def test_gradient_disabled(self):
    f_jax = lambda x: x * x

    model = tf.Module()
    model.f = tf.function(jax2tf.convert(f_jax, with_gradient=False),
                          autograph=False,
                          input_signature=[tf.TensorSpec([], tf.float32)])
    x = np.array(0.7, dtype=jnp.float32)
    self.assertAllClose(model.f(x), f_jax(x))
    restored_model = self.save_and_load_model(model)
    xv = tf.Variable(0.7, dtype=jnp.float32)
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
      tangent_out = x * x_dot * 3.
      return primal_out, tangent_out

    model = tf.Module()
    model.f = tf.function(jax2tf.convert(f_jax, with_gradient=True),
                          autograph=False,
                          input_signature=[tf.TensorSpec([], tf.float32)])
    x = np.array(0.7, dtype=jnp.float32)
    self.assertAllClose(model.f(x), f_jax(x))
    restored_model = self.save_and_load_model(model)
    xv = tf.Variable(0.7, dtype=jnp.float32)
    self.assertAllClose(restored_model.f(x), f_jax(x))
    with tf.GradientTape() as tape:
      y = restored_model.f(xv)
    self.assertAllClose(tape.gradient(y, xv).numpy(),
                        jax.grad(f_jax)(x).astype(np.float32))

  def _compare_with_saved_model(self, f_jax, *args):
    # Certain ops are converted to ensure an XLA context, e.g.,
    # tf.gather, so that the index-out-of-bounds behavior matches that of
    # JAX. We check that this information is preserved through a savedmodel
    f_tf = jax2tf.convert(f_jax)
    res = f_tf(*args)

    model = tf.Module()
    input_signature = list(tf.TensorSpec(a.shape, a.dtype) for a in args)
    model.f = tf.function(f_tf,
                          autograph=False,
                          input_signature=input_signature)
    restored_model = self.save_and_load_model(model)
    res_restored = restored_model.f(*args)
    self.assertAllClose(res, res_restored)

  def test_xla_context_preserved_slice(self):
    arr = np.arange(10, dtype=np.float32)
    def f_jax(arr):
      return lax.dynamic_slice(arr, [100], [1])  # out of bounds, should return the last element
    self._compare_with_saved_model(f_jax, arr)

  def test_xla_context_preserved_gather(self):
    def f_jax(arr):
      return arr[100]  # out of bounds, should return the last element
    arr = np.arange(10, dtype=np.float32)
    self._compare_with_saved_model(f_jax, arr)

  # Test does not work on GPU/TPU; would need something like TPU inference
  # converter to separate the model on what needs to run on CPU or accelerator.
  @jtu.skip_on_devices("gpu", "tpu")
  def test_tf_mix_jax_with_uncompileable(self):
    """Show how to combine TF-uncompileable code with compiled JAX-converted code."""
    def tf_fn(x_str, compute_tf_fn=lambda x: x):
      # Some TF preprocessing code that cannot be compiled with XLA because it
      # uses strings.
      numbers_f32 = tf.strings.to_number(x_str, out_type=tf.float32)
      numbers_f16 = tf.cast(numbers_f32, tf.float16)
      return compute_tf_fn(numbers_f16)

    x_str = np.array(["3.14", "2.78"])

    # Test that we get an error if we try to TF-compile `tf_fn`
    with self.assertRaisesRegex(
        Exception,
        "Detected unsupported operations when trying to compile graph"):
      tf.function(tf_fn, jit_compile=True)(x_str)

    # Plug in the TF-compiled JAX-converted `compute_jax_fn`.
    composed_fn = lambda x_str: tf_fn(
        x_str,
        compute_tf_fn=tf.function(jax2tf.convert(jnp.sin),
                                  autograph=True,
                                  jit_compile=True))
    res_tf = composed_fn(x_str)
    self.assertAllClose(res_tf.numpy(),
                        jnp.sin(np.array([3.14, 2.78], dtype=np.float16)))

    # Save and restore SavedModel
    model = tf.Module()
    model.f = tf.function(
        composed_fn,
        input_signature=[tf.TensorSpec((2,), dtype=tf.string)])
    restored_model = self.save_and_load_model(model)
    res_tf_restored = restored_model.f(x_str)
    self.assertAllClose(res_tf_restored.numpy(), res_tf.numpy())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
