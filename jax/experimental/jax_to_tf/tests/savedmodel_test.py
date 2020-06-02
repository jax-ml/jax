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
import unittest

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import test_util as jtu
import jax.numpy as jnp

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from jax.experimental import jax_to_tf

from jax.config import config
config.parse_flags_with_absl()


class SavedModelTest(jtu.JaxTestCase):

  def testSavedModel(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    model = tf.Module()
    model.f = tf.function(jax_to_tf.convert(f_jax),
                          input_signature=[tf.TensorSpec([], tf.float32)])
    x = np.array(0.7)
    np.testing.assert_allclose(model.f(x), f_jax(x))
    # Roundtrip through saved model on disk.
    model_dir = os.path.join(absltest.get_default_test_tmpdir(), str(id(model)))
    tf.saved_model.save(model, model_dir)
    restored_model = tf.saved_model.load(model_dir)
    np.testing.assert_allclose(restored_model.f(x), f_jax(x))

if __name__ == "__main__":
  absltest.main()