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
"""Tests for backwards compatibility of custom calls.


To create a new test, write the JAX function that exercises the custom call you
want, then pick some inputs, and then add this to the new test:

   inputs = (...,)  # Tuple of nd.array
   data = dataclasses.replace(dummy_data, inputs=inputs)
   self.run_test(func, data)

The test will fail, but will print the CustomTestData you need. Copy and paste
it into the test, and you should be set.
"""
import dataclasses
import datetime
import re
from typing import Callable, List, Sequence
import sys

from absl import logging
from absl.testing import absltest

import numpy as np
# Import some NumPy symbols so that we can parse repr(ndarray).
from numpy import array, float32, complex64, complex128

import jax
from jax import lax
import jax.numpy as jnp
from jax._src import core


import tensorflow as tf  # type: ignore[import]
from jax.experimental import jax2tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla  # type: ignore[import]
# pylint: enable=g-direct-tensorflow-import

from jax.config import config
config.parse_flags_with_absl()

# Import after parsing flags
from jax._src import test_util as jtu


@dataclasses.dataclass
class CompatTestData:
    inputs: Sequence[np.ndarray]
    platform: str  # One of "cpu", "tpu", "cuda", "rocm"
    expected_outputs: Sequence[np.ndarray]
    mlir_module_text: str
    mlir_module_serialized: bytes
    custom_call_targets: List[str]
    serialized_date: datetime.date  # e.g., datetime.date(2023, 3, 9)

# The dummy_data is used for getting started for adding a new test and for
# testing the helper functions.

# Pasted from the test output (see module docstring)
dummy_data = CompatTestData(
    platform='cpu',
    custom_call_targets=[],
    serialized_date=datetime.date(2023, 3, 10),
    inputs=(np.array(0., np.float32),),
    expected_outputs=(array(0., dtype=float32),),
    mlir_module_text="""
module @jit_sin {
  func.func public @main(%arg0: tensor<f32> {jax.arg_info = "x", mhlo.sharding = ""}) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = stablehlo.sine %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x01MLIR17.0.0git\x00\x01\x1d\x07\x01\x03\x05\x01\x03\x07\x03\x05\x03\t\x05\x03\x0b\x03E3\x07\x013\x07\x0b\x0b\x13\x0b3\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x01\x07\x0f\x17\x07\x02\xa3\x1f\x05\r\x05\x0f\x03\x03\x03\t\x05\x11\x03\x0b\r\x0f\x19\x1b\x1d\x1f\x03%')\x05\x13\x01\x03\x11\x03\x05\x13\x15\x17\x05\x05\x15\x05\x17\x05\x19\x05\x1b\r\x03\x05\x1d\x01\x03!\x03\x03#\x05\x05\x1f\x05!\x05#\x05%\x1d-/\x05'\x171\x9b\x01\x05)\x1b\x01\x05\x05\x03\x01\x03\x01\x0b\x04A\x05\x01\x11\x01\x07\x07\x03\x01\x05\x03\x11\x01\x0b\x07\x03\x05\x0b\x03\x01\x01\x07\x06+\x03\x01\x03\x01\x05\x04\x01\x03\x03\x06\x03\x01\x05\x01\x00z\x04+\x97/\x0f\x1f\x0b!\x15\x1d\x1d\x05\x1b\x15\x11\x03\x13\x0b\x0f\x0f\x15\x0b\x11builtin\x00func\x00stablehlo\x00module\x00return\x00sine\x00sym_name\x00\x00jit_sin\x00arg_attrs\x00jax.arg_info\x00x\x00mhlo.sharding\x00function_type\x00res_attrs\x00jax.result_info\x00main\x00sym_visibility\x00public\x00jit(sin)/jit(main)/sin\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00",
)  # End paste


class CompatTest(jtu.JaxTestCase):

  def run_one_test(self, func: Callable, data: CompatTestData):
    if jtu.device_under_test() != data.platform:
      self.skipTest(f"Test enabled only for {data.platform}")

    # Check that it runs in JAX native
    res_jax = jax.jit(func)(*data.inputs)
    if not isinstance(res_jax, (list, tuple)):
      res_jax = (res_jax,)
    res_jax = tuple(np.array(a) for a in res_jax)

    # Use the native exporter, to make sure we get the proper serialized module.
    exported = jax2tf.jax2tf.export_native(
        jax.jit(func),
        [core.ShapedArray(a.shape, a.dtype) for a in data.inputs],
        lowering_platform=data.platform,
        # Must turn off strict checks because the custom calls may be unallowed.
        strict_checks=False)

    module_str = str(exported.mlir_module)
    custom_call_targets = list(re.findall(r"stablehlo.custom_call\s*@([^\(]+)\(", module_str))
    np.set_printoptions(threshold=sys.maxsize)
    # Print the test data to simplify updating the test
    updated_testdata = f"""
    # Pasted from the test output (see module docstring)
    harness = CompatTestData(
      platform = {repr(data.platform)},
      custom_call_targets = {repr(custom_call_targets)},
      serialized_date = {repr(datetime.date.today())},
      inputs = {repr(data.inputs)},
      expected_outputs = {repr(res_jax)},
      mlir_module_text = \"\"\"\n{module_str}\"\"\",
      mlir_module_serialized = {repr(exported.mlir_module_serialized)},
    )  # End paste
"""
    print(updated_testdata)
    self.assertAllClose(res_jax, data.expected_outputs)

    res_serialized = self.run_serialized(data)
    self.assertAllClose(res_serialized, data.expected_outputs)
    self.assertListEqual(custom_call_targets, data.custom_call_targets)

  def run_serialized(self, data: CompatTestData):
    # Run the serialized module. For now, use XlaCallModule, eventually
    # come up with a JAX-native way of running serialized modules.
    args_tf = [tf.constant(a) for a in data.inputs]
    res = tfxla.call_module(
        args_tf,
        Tout=[a.dtype for a in args_tf],
        Sout=[a.shape for a in args_tf],
        module=data.mlir_module_serialized)
    return tuple(r.numpy() for r in res)

  def test_dummy(self):
    self.run_one_test(jnp.sin, dummy_data)

  def test_detect_different_output(self):
    data = dataclasses.replace(dummy_data, expected_outputs=(np.array(2.),))
    with self.assertRaisesRegex(AssertionError,
                                "Not equal to tolerance"):
      self.run_one_test(jnp.sin, data)

  def test_detect_different_custom_calls(self):
    data = dataclasses.replace(dummy_data, custom_call_targets=["missing"])
    with self.assertRaisesRegex(AssertionError,
                                "Lists differ"):
      self.run_one_test(jnp.sin, data)

  def test_fft(self):
    def f_jax(x):
      return lax.fft(x, fft_type="fft", fft_lengths=(4,))

    # Pasted from the test output
    harness = CompatTestData(
        platform='cpu',
        custom_call_targets=['ducc_fft'],
        serialized_date=datetime.date(2023, 3, 10),
        inputs=(array([[0., 1., 2., 3.],
                       [4., 5., 6., 7.],
                       [8., 9., 10., 11.]], dtype=float32),),
        expected_outputs=(array([[6. + 0.j, -2. + 2.j, -2. + 0.j, -2. - 2.j],
                                 [22. + 0.j, -2. + 2.j, -2. + 0.j, -2. - 2.j],
                                 [38. + 0.j, -2. + 2.j, -2. + 0.j, -2. - 2.j]], dtype=complex64),),
        mlir_module_text="""module @jit_f_jax {
      func.func public @main(%arg0: tensor<3x4xf32> {jax.arg_info = "x", mhlo.sharding = ""}) -> (tensor<3x4xcomplex<f32>> {jax.result_info = ""}) {
        %0 = call @fft(%arg0) : (tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>>
        return %0 : tensor<3x4xcomplex<f32>>
      }
      func.func private @fft(%arg0: tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>> {
        %0 = stablehlo.convert %arg0 : (tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>>
        %1 = stablehlo.constant dense<"0x18000000140024000000000008000C001000140007001800140000000000000154000000380000001C00000010000000000000000000F03F0000000001000000010000000200000004000000000000000100000000000000000000000200000004000000000000000100000000000000000000000200000003000000000000000400000000000000"> : tensor<136xui8>
        %2 = stablehlo.custom_call @ducc_fft(%1, %0) {api_version = 2 : i32, operand_layouts = [dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<136xui8>, tensor<3x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
        return %2 : tensor<3x4xcomplex<f32>>
      }
    }
    """,
        mlir_module_serialized=b'ML\xefR\x01MLIR17.0.0git\x00\x01#\x07\x01\x03\x05\x01\x03\x07\x03\x07\x03\t\x0b\x05\x07\r\x0f\x11\x03\x99s\x19\x01s\x07\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0bO\x13\x0b3\x0f\x1b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0bf\x04\x0bK\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x13/\x0b\x0b\x0f\x13\x0b\x0b\x01\x19\x17\x17\x07\x17\x07\x17\x0b\x0b\x0f\x0b\x13\x13\x02\xe6\x05\x1f\x05\x13\x05\x15\x1d?\t\x17A\xe5\x01\x05\x17\x05\x19\r\x0b\x05\x1b\x05\x1d\x01\x03;\x05\x1f\x1dM\t\x01\x01%\x17!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03\x03!\x05!\x03\x0b\x0b%\r\x0f\x11/\x035\x137\x01\x03\'\x03\x05)+-\x05\x05#\x05%\x05\'\x01\x031\x03\x033\x05\x05)\x05+\x05-\x03\x0b\x0b\x15\r\x0f\x11\x15\x03\x17\x13=\x03\x01\x05/\x051\x053\x1dE\t\x055\x03\x03IK\x057%\x07"\x02\x18\x00\x00\x00\x14\x00$\x00\x00\x00\x00\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x07\x00\x18\x00\x14\x00\x00\x00\x00\x00\x00\x01T\x00\x00\x008\x00\x00\x00\x1c\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x059\x03\x11QSU\x05WY[\x1b]_acg\x1bik\x05;\x11\x11\t\x05=\x05?\x05A\x05C\x05E\x11\x13\x00\x05G\x01\x05e\x1d%\x15\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05I\x05K\x01\x03\x1d\x03\x03oq\x05M\t\x17\x1b\x05\r\x11\r\x1b\x05\r\x11\x05\x0b\x1b\x03B\x04\x0f\x03\x05\x03\x03\x03\x01\x13\x05\x01E\x01\x02\x02\x01\t\x1b\x03\x05\t\x1b\x03\t\t\x04\x8f\x05\x01\x11\x01\x1f\x07\x03\x01\t\x03\x11\x01#\x07\x03\x05\x0b\x03\x03\x01\x07\x07\x07m\x03\x01\x03\x01\x05\x04\x01\x03\x03\x03\x11\x079\x07\x03\t\x13\x03\x03\x01\t\x06C\x03\x01\x03\x01\x0b\x03\x19G\x03\x07\r\x07\x19O\x03\x01\x05\x05\x03\x05\x04\x07\x03\x07\x06\x03\x01\x05\x01\x00\x8e\x0eO\x0f\x1f/!!)\x13#\x1f\x19\x93\r\xb1\x97\x86\x04\x11\x0f\x0b!\x1d\x05\x1b\x15\t\x1f\x15\x1d\x15\x03\x13\x19\x13\x11\x0b\x0f\x0f\x15\x0b\x11builtin\x00func\x00stablehlo\x00module\x00return\x00call\x00convert\x00constant\x00custom_call\x00sym_name\x00\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00fft\x00jit_f_jax\x00jax.arg_info\x00x\x00mhlo.sharding\x00jax.result_info\x00main\x00public\x00private\x00jit(f_jax)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=fft in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00jit(f_jax)/jit(main)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False]\x00value\x00jit(f_jax)/jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(4,)]\x00api_version\x00backend_config\x00call_target_name\x00ducc_fft\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00',

    )  # End paste
    self.run_one_test(f_jax, harness)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
