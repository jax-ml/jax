# Copyright 2023 The JAX Authors.
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
"""Tests for backwards compatibility of custom calls involving TensorFlow.

See the back_compat_test_util module docstring for how to setup and update
these tests.
"""
from typing import Callable, Optional, Sequence

from absl.testing import absltest

from jax import config
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import back_compat_test_util as bctu

from jax.experimental.jax2tf.tests.back_compat_testdata import tf_call_tf_function

import jax.numpy as jnp

from jax._src.lib import xla_extension
from jax._src import test_util as jtu

import tensorflow as tf
from tensorflow.core.framework import graph_pb2  # type: ignore[import]

config.parse_flags_with_absl()


class CompatTensoflowTest(bctu.CompatTestBase):
  """Compatibility tests that use TF.

  Uses tf.Graph to serialize and run the functions; expects that `func`
  contains a `jax2tf.call_tf` and uses `jax2tf.convert` to generate a
  `tf.Graph` containing a XlaCallModule with the actual MLIR module.
  """

  def run_current(self, func: Callable, data: bctu.CompatTestData):
    # Is there a better way to serialize/deserialize TF functions? I thought
    # about using tf.saved_model, but then we have to zip/unzip a whole
    # directory.
    @tf.function(autograph=False, jit_compile=True)
    def tf_func(the_input):  # Use recognizeable names for input and result
      res = jax2tf.convert(func, native_serialization=True)(the_input)
      return tf.identity(res, name="the_result")

    self.tf_func = tf_func
    return tf_func(*data.inputs)  # type: ignore

  def serialize(self, func: Callable, data: bctu.CompatTestData,
                polymorphic_shapes: Optional[Sequence[str]] = None,
                allow_additional_custom_call_targets: Sequence[str] = ()):
    # We serialize as a tf.Graph
    assert len(data.inputs) == 1  # We only support a single input now
    tf_graph = self.tf_func.get_concrete_function(*data.inputs).graph
    for op in tf_graph.get_operations():
      if op.type == "XlaCallModule":
        serialized_module = op.get_attr("module")
        module_str = xla_extension.mlir.deserialize_portable_artifact(
          serialized_module)
        module_version = op.get_attr("version")
        break
    else:
      raise ValueError("Cannot find an XlaCallModule")
    tf_graph_def = tf_graph.as_graph_def()
    # module_str is just for human readability, add both the MLIR module
    # and the tf.Graph
    module_str = ("# First the MLIR module:\n" + module_str +
                  "\n# Then the tf.Graph:\n" + str(tf_graph_def))
    serialized = tf_graph_def.SerializeToString()
    return serialized, module_str, module_version

  def run_serialized(self, data: bctu.CompatTestData,
                     polymorphic_shapes: Optional[Sequence[str]] = None):
    loaded_f_tf_graph = graph_pb2.GraphDef()
    loaded_f_tf_graph.ParseFromString(data.mlir_module_serialized)

    @tf.function(autograph=False)
    def loaded_fun(x):
      result = tf.import_graph_def(loaded_f_tf_graph,
                                   input_map={"the_input": x},
                                   return_elements=["the_result:0"])
      return result[0]

    return (loaded_fun(*data.inputs).numpy(),)

  def test_tf_call_tf_function(self):
    self.skipTest("b/286409830: brittle on function naming.")
    # A custom call tf.call_tf_function is generated when we lower call_tf
    # with the call_tf_graph=True option.
    def func(x):
      def func_tf(x):
        return tf.math.sin(x)
      return jnp.cos(jax2tf.call_tf(func_tf, output_shape_dtype=x,
                                    call_tf_graph=True)(x))

    data = self.load_testdata(tf_call_tf_function.data_2023_06_02)
    self.run_one_test(func, data)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
