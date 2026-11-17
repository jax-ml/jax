# Copyright 2025 The JAX Authors.
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

"""Multihost test for JAX2TF."""

import contextlib
import os
import unittest
import warnings
import jax
from jax import numpy as jnp
from jax._src import pjit
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P

try:
  # TODO(b/470156950): Remove this once a proper fix is in place
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore",
                            category=FutureWarning,
                            message=".*np.object.*")
    import tensorflow as tf
  from jax.experimental import jax2tf
  from jax.experimental.jax2tf.tests import tf_test_util
  JaxToTfTestCase = tf_test_util.JaxToTfTestCase
except ImportError:
  tf = None
  jax2tf = None
  tf_test_util = None
  JaxToTfTestCase = jtu.JaxTestCase


@unittest.skipIf(tf is None, "Test requires tensorflow.")
class Jax2TfMultiProcessTest(JaxToTfTestCase, jt_multiprocess.MultiProcessTest):

  def test_multi_process_pjit_export(self):
    """Pjitted function can be exported."""
    key_w, key_x = jax.random.split(jax.random.PRNGKey(1234), 2)
    w = jax.random.uniform(key_w, [16, 16], dtype=jnp.float32)
    x = jax.random.uniform(key_x, [16, 1], dtype=jnp.float32)

    with jtu.create_mesh((4, 2), ("x", "y")):
      pjit_matmul = pjit.pjit(jnp.matmul, in_shardings=(P("x", "y"), None))
      jax_result = multihost_utils.process_allgather(
          pjit_matmul(w, x), tiled=True)

      # Harnesses that run JAX on a separately loaded libtpu.so set
      # JAX2TF_MULTIPROCESS_TF_ON_CPU=1: the TensorFlow TPU runtime linked into
      # this process cannot share the TPU devices that plugin already opened,
      # so every process then serializes and executes the exported SavedModel
      # on CPU instead.
      use_cpu_tf = os.environ.get("JAX2TF_MULTIPROCESS_TF_ON_CPU") == "1"
      convert_kwargs = (
          {"native_serialization_platforms": ("cpu",)} if use_cpu_tf else {}
      )
      tf_device_ctx = (
          tf.device("/CPU:0") if use_cpu_tf else contextlib.nullcontext()
      )
      with tf_device_ctx:
        tf_model = tf.Module()
        tf_model.w = tf.Variable(w)
        tf_closure = tf.function(
            lambda x: {
                "y": jax2tf.convert(pjit_matmul, **convert_kwargs)(
                    tf_model.w, x
                )
            },
            autograph=False,
        ).get_concrete_function(
            tf.TensorSpec.from_tensor(tf.constant(x), name="x")
        )

        if jax.process_index() == 0:
          export_dir = self.create_tempdir().full_path
          tf.saved_model.save(
              tf_model,
              export_dir,
              signatures={"serving_default": tf_closure},
          )
          loaded = tf.saved_model.load(export_dir)
          tf_result = loaded.signatures["serving_default"](x=x)["y"]

          self.assertAllClose(tf_result.numpy(), jax_result)


if __name__ == "__main__":
  jt_multiprocess.main()
