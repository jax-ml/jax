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

Since we have to guarantee 6 months of backward compatibility for the
JAX serialized format, we need to guarantee that custom calls continue to
work as before. We test this here.

There is one test for each version of a custom call target, e.g., `test_fft`
tests the FFT custom calls on CPU. Only custom call targets tested here should
be listed in jax2tf._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE. All other custom
call targets will result in an error when encountered during serialization.

Once we stop using a custom call target in JAX, you can remove it from the
_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE and you can add a comment to the
test here to remove it after 6 months.

To create a new test, write the JAX function that exercises the custom call you
want, then pick some inputs, and then add this to the new test:

    def func(...): ...
    inputs = (...,)  # Tuple of nd.array
    data = dataclasses.replace(dummy_data, inputs=inputs,
                               platform=default_jax_backend())
    self.run_one_test(func, data)

The test will fail, but will print the CustomTestData you need. Copy and paste
it into the test, above the `self.run_one_test` above, and then manually
edit the serialization string to remove any pathnames that may be included at
the end, or gxxxxx3 at the beginning.
"""
import dataclasses
import datetime
import re
from typing import Callable, List, Sequence
import sys

# from absl import logging
from absl.testing import absltest
from absl import logging

import numpy as np
# Import some NumPy symbols so that we can parse repr(ndarray).
from numpy import array, float32, complex64, uint32

import jax
from jax import lax
from jax.experimental import jax2tf
import jax.numpy as jnp

from jax._src import core
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb

import tensorflow as tf  # type: ignore[import]

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla  # type: ignore[import]
# pylint: enable=g-direct-tensorflow-import

from jax.config import config
config.parse_flags_with_absl()

def default_jax_backend() -> str:
  # Canonicalize to turn into "cuda" or "rocm"
  return xb.canonicalize_platform(jax.default_backend())

@dataclasses.dataclass
class CompatTestData:
  platform: str  # One of: "cpu", "tpu", "cuda", "rocm"
  custom_call_targets: List[str]
  serialized_date: datetime.date  # e.g., datetime.date(2023, 3, 9)
  inputs: Sequence[np.ndarray]
  expected_outputs: Sequence[np.ndarray]
  mlir_module_text: str
  mlir_module_serialized: bytes
  xla_call_module_version: int  # The version of XlaCallModule to use for testing


# The dummy_data is used for getting started for adding a new test and for
# testing the helper functions.

# Pasted from the test output (see module docstring)
dummy_data = CompatTestData(
    platform='cpu',
    custom_call_targets=[],
    serialized_date=datetime.date(2023, 3, 15),
    inputs =(array(0., dtype=float32),),
    expected_outputs=(array(0., dtype=float32),),
    mlir_module_text="""
  module @jit_sin {
  func.func public @main(%arg0: tensor<f32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = stablehlo.sine %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
""",
    mlir_module_serialized = b"ML\xefR\x03MLIRxxx-trunk\x00\x01\x17\x05\x01\x05\x01\x03\x05\x03\x07\x07\t\x0b\x03K5\x07\x01\x1b\x07\x0b\x13\x0b3\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x03\x1b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x03\x07\x0f\x17\x07\x02\xa7\x1f\x05\r\x03\x03\x03\x07\x05\x0f\x03\x0b\x0b\x1b\r'\x0f)\x031\x113\x05\x11\x05\x13\x05\x15\x05\x17\x1d\x15\x17\x05\x19\x17\x19\xef\x01\x05\x1b\x03\x03\x1d\r\x05\x1f!#%\x1d\x1d\x1d\x1f\x1d!\x1d##\x03\x03\x03+\r\x03-/\x1d%\x1d'\x1d)\x1d+)\x01\x05\x11\x03\x01\x03\x01\t\x04A\x05\x01\x11\x01\x05\x07\x03\x01\x05\x03\x11\x01\t\x05\x03\x05\x0b\x03\x01\x01\x05\x06\x13\x03\x01\x03\x01\x07\x04\x01\x03\x03\x06\x03\x01\x05\x01\x00\x9a\x04-\x0f\x0b\x03!\x1b\x1d\x05\x1b\x83/\x1f\x15\x1d\x15\x11\x13\x15\x11\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00sine_v1\x00return_v1\x00sym_name\x00jit_sin\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(sin)/jit(main)/sin\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00\x00main\x00public\x00",
    xla_call_module_version = 4,
  )  # End paste


class CompatTest(jtu.JaxTestCase):

  def run_one_test(self, func: Callable[..., jax.Array],
                   data: CompatTestData):
    if default_jax_backend() != data.platform:
      self.skipTest(f"Test enabled only for {data.platform}")

    # Check that it runs in JAX native
    res_jax = jax.jit(func)(*data.inputs)
    if not isinstance(res_jax, (list, tuple)):
      res_jax = (res_jax,)
    res_jax = tuple(np.array(a) for a in res_jax)

    # Use the native exporter, to make sure we get the proper serialized module.
    exported = jax2tf.jax2tf.serialize_native(
        jax.jit(func),
        [core.ShapedArray(a.shape, a.dtype) for a in data.inputs],
        lowering_platform=default_jax_backend(),
        # Must turn off strict checks because the custom calls may be unallowed.
        strict_checks=False)

    module_str = str(exported.mlir_module)
    custom_call_re = r"stablehlo.custom_call\s*@([^\(]+)\("
    custom_call_targets = sorted(list(set(re.findall(custom_call_re,
                                                     module_str))))
    np.set_printoptions(threshold=sys.maxsize)
    # Print the test data to simplify updating the test
    updated_testdata = f"""Computed test data for this test (paste this into the test):
    # Pasted from the test output (see module docstring)
    data = CompatTestData(
        platform={repr(default_jax_backend())},
        custom_call_targets={repr(custom_call_targets)},
        serialized_date={repr(datetime.date.today())},
        inputs={repr(data.inputs)},
        expected_outputs={repr(res_jax)},
        mlir_module_text=\"\"\"\n{module_str}\"\"\",
        mlir_module_serialized={repr(exported.mlir_module_serialized)},
        xla_call_module_version={exported.xla_call_module_version},
    )  # End paste
"""
    logging.info("%s", updated_testdata)
    self.assertAllClose(res_jax, data.expected_outputs)

    res_serialized = self.run_serialized(data)
    self.assertAllClose(res_serialized, data.expected_outputs)
    self.assertListEqual(custom_call_targets, data.custom_call_targets)

  def run_serialized(self, data: CompatTestData, run_tf=None):
    # Run the serialized module. For now, use XlaCallModule. This has the
    # disadvantage that it brings TF and jax2tf in the picture, but has the
    # advantage that it is simple (e.g., XlaCallModule already has the
    # machinery to deserialize and run), and also it is the way users actually
    # run serialized modules today.
    # TODO(necula): come up with a JAX-native way of running serialized modules.
    args_tf = [tf.constant(a) for a in data.inputs]
    res_tf = [tf.constant(r) for r in data.expected_outputs]
    res = tfxla.call_module(
          args_tf,
          version=data.xla_call_module_version,
          Tout=[r.dtype for r in res_tf],
          Sout=[r.shape for r in res_tf],
          module=data.mlir_module_serialized)
    return tuple(r.numpy() for r in res)

  def test_dummy(self):
    # Tests the test mechanism. Let this test run on all platforms
    platform_dummy_data = dataclasses.replace(dummy_data,
                                              platform=default_jax_backend())
    self.run_one_test(jnp.sin, platform_dummy_data)

  def test_detect_different_output(self):
    # Test the detection mechanism. Let this test run on all platforms
    platform_dummy_data = dataclasses.replace(
        dummy_data,
        platform=default_jax_backend(),
        expected_outputs=(np.array(2., dtype=np.float32),))
    with self.assertRaisesRegex(AssertionError, "Not equal to tolerance"):
      self.run_one_test(jnp.sin, platform_dummy_data)

  def test_detect_different_custom_calls(self):
    # Test the detection mechanism. Let this test run on all platforms
    platform_dummy_data = dataclasses.replace(
        dummy_data,
        platform=default_jax_backend(),
        custom_call_targets=["missing"])
    with self.assertRaisesRegex(AssertionError, "Lists differ"):
      self.run_one_test(jnp.sin, platform_dummy_data)

  def test_ducc_fft(self):
    def func(x):
      return lax.fft(x, fft_type="fft", fft_lengths=(4,))

    # Pasted from the test output (see module docstring)
    data = CompatTestData(
        platform='cpu',
        custom_call_targets=['ducc_fft'],
        serialized_date=datetime.date(2023, 3, 15),
        inputs=(array([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]], dtype=float32),),
        expected_outputs=(array([[ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j],
        [22.+0.j, -2.+2.j, -2.+0.j, -2.-2.j],
        [38.+0.j, -2.+2.j, -2.+0.j, -2.-2.j]], dtype=complex64),),
        mlir_module_text="""
module @jit_func {
  func.func public @main(%arg0: tensor<3x4xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}) -> (tensor<3x4xcomplex<f32>> {jax.result_info = ""}) {
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
        mlir_module_serialized=b'ML\xefR\x03MLIRxxx-trunk\x00\x01\x1d\x05\x01\x05\x01\x03\x05\x03\r\x07\t\x0b\r\x0f\x11\x03\x99s\x15\x01?\x07\x0b\x0f\x17\x0b\x0b\x0b\x0b\x0f\x13\x0b33\x0b\x0b\x0f\x0b\x13\x0b\x0bK\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x035\x0b\x0b\x0f\x0b\x0bO\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0bf\x04\x0b\x0b\x0b\x13/\x0f\x03\x15\x17\x17\x07\x17\x07\x17\x0b\x07\x13\x13\x02\xca\x05\x1f\x05\x13\x1d\x1b\x07\x17\x1d&\x03\x01\x05\x15\x05\x17\x05\x19\x05\x1b\x1d\'\x07\x03\x03\x03\x15\x05\x1d\x03\x0b\tK\x0b?\rW\x03]\x0f_\x03\x0b\tC\x0b?\rC\x03E\x0fc\x05\x1f\x05!\x1d!\x07\x05#\x03\x03%e\x05%\x05\'\x03\x11+g-A/i1G3k5m7G9q\x05)\x05+\x05-\x05/\x051\x053\x055\x057\x03\x03=E\x059#\x0b\x1d;\x03\x03a\x1d=\x03\x01\x1f\x13!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03M\r\x05OQSU\x1d?\x1dA\x1dC\x1dE\x03\x03Y\r\x03[A\x1dG\x1dI\x1dK\r\x01\x1dM\x1f\x07"\x02\x18\x00\x00\x00\x14\x00$\x00\x00\x00\x00\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x07\x00\x18\x00\x14\x00\x00\x00\x00\x00\x00\x01T\x00\x00\x008\x00\x00\x00\x1c\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x05\x1dO\x05\x01\x03\x05oI\x1f\x11\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03I)\x05\r\x11\r)\x05\r\x11\x05\t)\x03B\x04\x0f\x13\x11\x03\x03\x03\x01\x03\x05!)\x03\x05\t)\x03\t\t\x04\x8f\x05\x01\x11\x01\x13\x07\x03\x01\t\x03\x11\x01\x17\x05\x03\x05\x0b\x03\x03\x01\r\x07\x05;\x03\x01\x03\x01\x05\x04\x01\x03\x03\x03\x11\x05\x19\x05\x03\t\x13\x03\x03\x01\x07\x06\x1f\x03\x01\x03\x01\t\x03\x11#\x03\x07\x0b\x07\x11)\x03\x01\x05\x05\x03\x05\x04\x05\x03\x07\x06\x03\x01\x05\x01\x00\xc2\x0eQ\x13\x11\x0f\x0b!\x1b\x1d\x05\x1b\t\x03\x0f\x1f/!!)#\x1f\x19\x91\r\xaf\x83\x82\x04\x13\x1f\x15\x1d\x15\x13\x11\x1f\x19\x17\x15\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00return_v1\x00convert_v1\x00constant_v1\x00custom_call_v1\x00call_v1\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit_func\x00jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=fft in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00jit(func)/jit(main)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False]\x00value\x00jit(func)/jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(4,)]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00\x00fft\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00main\x00public\x00private\x00ducc_fft\x00',
        xla_call_module_version=4,
    )  # End paste
    self.run_one_test(func, data)


  def test_cu_threefry2x32(self):
    def func(x):
      return jax.random.uniform(x, (2, 4), dtype=np.float32)
    # Pasted from the test output (see module docstring)
    data = CompatTestData(
        platform='cuda',
        custom_call_targets=['cu_threefry2x32'],
        serialized_date=datetime.date(2023, 3, 15),
        inputs=(array([42, 43], dtype=uint32),),
        expected_outputs=(array([[0.42591238, 0.0769949 , 0.44370103, 0.72904015],
        [0.17879379, 0.81439507, 0.00191903, 0.68608475]], dtype=float32),),
        mlir_module_text="""
module @jit_func {
  func.func public @main(%arg0: tensor<2xui32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}) -> (tensor<2x4xf32> {jax.result_info = ""}) {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %4 = stablehlo.iota dim = 0 : tensor<8xui32>
    %5 = "stablehlo.slice"(%arg0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = "stablehlo.slice"(%arg0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = "stablehlo.slice"(%4) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<8xui32>) -> tensor<4xui32>
    %10 = "stablehlo.slice"(%4) {limit_indices = dense<8> : tensor<1xi64>, start_indices = dense<4> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<8xui32>) -> tensor<4xui32>
    %11 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %12 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %13 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<4xui32>) -> tensor<4xui32>
    %14 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<4xui32>) -> tensor<4xui32>
    %15 = stablehlo.custom_call @cu_threefry2x32(%11, %12, %13, %14) {api_version = 2 : i32, backend_config = "\04\00\00\00\00\00\00\00", operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<4xui32>, tensor<4xui32>, tensor<4xui32>, tensor<4xui32>) -> tuple<tensor<4xui32>, tensor<4xui32>>
    %16 = stablehlo.get_tuple_element %15[0] : (tuple<tensor<4xui32>, tensor<4xui32>>) -> tensor<4xui32>
    %17 = stablehlo.get_tuple_element %15[1] : (tuple<tensor<4xui32>, tensor<4xui32>>) -> tensor<4xui32>
    %18 = stablehlo.concatenate %16, %17, dim = 0 : (tensor<4xui32>, tensor<4xui32>) -> tensor<8xui32>
    %19 = stablehlo.reshape %18 : (tensor<8xui32>) -> tensor<2x4xui32>
    %20 = stablehlo.constant dense<9> : tensor<ui32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<ui32>) -> tensor<2x4xui32>
    %22 = stablehlo.shift_right_logical %19, %21 : tensor<2x4xui32>
    %23 = stablehlo.constant dense<1065353216> : tensor<ui32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<ui32>) -> tensor<2x4xui32>
    %25 = stablehlo.or %22, %24 : tensor<2x4xui32>
    %26 = stablehlo.bitcast_convert %25 : (tensor<2x4xui32>) -> tensor<2x4xf32>
    %27 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %29 = stablehlo.subtract %26, %28 : tensor<2x4xf32>
    %30 = stablehlo.subtract %3, %1 : tensor<1x1xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32>
    %32 = stablehlo.multiply %29, %31 : tensor<2x4xf32>
    %33 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32>
    %34 = stablehlo.add %32, %33 : tensor<2x4xf32>
    %35 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32>
    %36 = stablehlo.maximum %35, %34 : tensor<2x4xf32>
    return %36 : tensor<2x4xf32>
  }
}
""",
        mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x013\x05\x01\x05\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%'\x032\x02\xe1)\x01\x9b\x17\x07\x13\x0f\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0f\x13\x0f\x13\x0b\x0f\x0f\x0f\x0f\x0f\x13\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0b\x13\x0b\x0f\x0b#\x0f\x0b\x0b#\x0f\x0b#\x0f\x0b#\x0f\x0b\x0bK\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x03G///\x0f/\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x1f\x0f\x1f//\x0b\x0b\x0b\x0b\x1b\x13\x0f\x0f\x1f\x1fO\x03)\x17\x13\x07\x0f\x0f\x13\x17\x07\x07\x17\x13\x13\x13\x07\x17\x13\x13\x13\x07\x13\x02\xb6\x07\x17?\xb2\x03\x01\x1f\x03\x03\x11\xc3\x1dc\x01\x05)\x05+\x05-\x05/\x051\x1d\x93\x01\x03\x03\x11\xdf\x053\x1d=\x01\x03\x03\t\xc5\x1dO\x01\x03\x03\x11\x9f\x055\x1d\x89\x01\x1d\x8d\x01\x1d\x95\x01\x1d\x97\x01\x1d\x99\x01\x03\x03\x17/\x057\x03\x0b3\xa75\xb37\xb5\x17\xbd9\xbf\x059\x05;\x05=\x05?\x03\x03\t\xc1\x05A\x05C\x03\x03C\xa1\x05E\x1dG\x01\x05G\x03\x07\x0b\x9b\r\x9f\x0f\x9b\x1dM\x01\x05I\x05K\x03\x07\x0b\xc7\r\x9b\x0f\x9b\x1dU\x01\x05M\x03\x07\x0b\xa3\r\x9f\x0f\x9b\x1d[\x01\x05O\x03\x07\x0b\xc9\r\xa3\x0f\x9b\x1da\x01\x05Q\x05S\x03\x11g\xcbi\xcdk\xcfm\xa5o\xd1q\xd3s\xa5u\xd5\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x05c\x03\x03!\xd7\x03\x03!\xd9\x03\x03}\xa1\x05e\x1d\x81\x01\x05g\x1d\x85\x01\x05i\x03\x03\t\xdb\x05k\x03\x03\t\xdd\x05m\x1d\x91\x01\x05o\x05q\x05s\x05u\x05w\x1f\x0b\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x0b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x0f\x01\x1f\x0b\x11\x04\x00\x00\x00\x00\x00\x00\x00\x03\x01\x03\x03\xa9\r\x05\xab\xad\xaf\xb1\x1dy\x1d{\x1d}\x1d\x7f#\x1d\x03\x03\xb7\r\x03\xb9\xbb\x1d\x81\x1d\x83\x1d\x85\x1d\x87\x1f\t\t\x00\x00\x00\x00\x1f\x1f\x01\x1f\t\t\x00\x00\x80?\x1f\x0b\x11\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x0b\x11\x08\x00\x00\x00\x00\x00\x00\x00\x0b\x05\x1d\x89\x1d\x8b\x05\x01\x03\t\x9d\x9d\x9d\x9d\x03\x05\x9d\x9d\x13\x1b\x01\x13\x1b\x05\x1f\x07\t\t\x00\x00\x00\x1f\x07\t\x00\x00\x80?\x1f'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00)\x05\t\x11\x11)\x03\x11\x05%)\x01\x05)\x01\x11)\x03\x05\x0f)\x05\t\x11\x05\x1d\t)\x05\x05\x05\x11)\x03\t\x05)\x03!\x05)\x03\x05\x05\x1b\x11\x03\x15\x03\x01)\x03\x01\x0f/\x05\x03\x03)\x03\x05%\x13)\x03\t\x0f\x04\xd6\x04\x05\x01\x11\x03-\x07\x03\x01\x05\x0f\x11\x031\x05\x03M\x9b\x03\x15\x03\x05\x03\x03;\x03\t\x03\x07\x19\x05\x03\x13\x03\x03\x05\x03\x03\x1b\x03\t\x03\x07\x19\x05\x03\x13\x03\x07\x11\x03EA\x03\x17\x07\x07KI\x03\x19\x03\x01\t\x06\x1d\x03\x07\x03\r\x07\x07SQ\x03\x19\x03\x01\t\x06\x1d\x03\x07\x03\x11\x07\x07YW\x03\x03\x03\x0b\x07\x07_]\x03\x03\x03\x0b\x03\x07\x07\x05\x03\x03\x03\x0f\x03\x07\x07\x05\x03\x03\x03\x13\x03\x07\x07\x1f\x03\x03\x03\x15\x03\x07\x07\x1f\x03\x03\x03\x17\x13\x07\x07e\x03!\t\x19\x1b\x1d\x1f\x0b\x07\x07w\x03\x03\x03!\x0b\x07\x07y\x03\x03\x03!\x15\x07\x7f{\x03\x17\x05#%\t\x06\x83\x03\r\x03'\x05\x03\x03\x87\x03\x07\x03\x07#\x05\x03\r\x03+\x17\x06#\x03\r\x05)-\x05\x03\x03\x8b\x03\x07\x03\x07%\x05\x03\r\x031\x19\x06%\x03\r\x05/3\x1b\x06\x8f\x03\x01\x035\x05\x03\x03\x1b\x03\t\x03\x07\x13\x05\x03\x01\x039\r\x06\x13\x03\x01\x057;\r\x06\x13\x03\x13\x05\t\x05\x03\x07'\x15\x03\x01\x03?\x1d\x06'\x03\x01\x05=A\x03\x07)\x15\x03\x01\x03\x05\x1f\x06)\x03\x01\x05CE\x03\x07+\x15\x03\x01\x03\x05!\x06+\x03\x01\x05IG#\x04\x03\x03K\x06\x03\x01\x05\x01\x00N\x19\x8d!\x13\x0f\x0b\x03!\x1b\x1d\x05\x1b1111y/Q}[\x15\x1f/!!)#\x1f\x19C\x9d\x9d\x9d[\x9d}\x1f\x83\x97\x1f\x15\x1d\x15\x13\r\x13+\x11\x1d\x1d\r\x15\x17\x0f\x19'\r/\x1f\x1f\x11\x11\x19+\x17\x13\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00slice_v1\x00reshape_v1\x00get_tuple_element_v1\x00subtract_v1\x00func_v1\x00iota_v1\x00custom_call_v1\x00concatenate_v1\x00shift_right_logical_v1\x00or_v1\x00bitcast_convert_v1\x00multiply_v1\x00add_v1\x00maximum_v1\x00return_v1\x00value\x00limit_indices\x00start_indices\x00strides\x00broadcast_dimensions\x00sym_name\x00index\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(func)/jit(main)/broadcast_in_dim[shape=(1, 1) broadcast_dimensions=()]\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00iota_dimension\x00jit(func)/jit(main)/iota[dtype=uint32 shape=(8,) dimension=0]\x00jit(func)/jit(main)/slice[start_indices=(0,) limit_indices=(1,) strides=(1,)]\x00jit(func)/jit(main)/squeeze[dimensions=(0,)]\x00jit(func)/jit(main)/slice[start_indices=(1,) limit_indices=(2,) strides=(1,)]\x00jit(func)/jit(main)/slice[start_indices=(0,) limit_indices=(4,) strides=None]\x00jit(func)/jit(main)/slice[start_indices=(4,) limit_indices=(8,) strides=None]\x00jit(func)/jit(main)/threefry2x32\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00dimension\x00jit(func)/jit(main)/concatenate[dimension=0]\x00jit(func)/jit(main)/reshape[new_sizes=(2, 4) dimensions=None]\x00jit(func)/jit(main)/shift_right_logical\x00jit(func)/jit(main)/or\x00jit(func)/jit(main)/bitcast_convert_type[new_dtype=float32]\x00jit(func)/jit(main)/sub\x00jit(func)/jit(main)/mul\x00jit(func)/jit(main)/add\x00jit(func)/jit(main)/max\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00\x00main\x00public\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00cu_threefry2x32\x00",
        xla_call_module_version=4,
    )  # End paste
    self.run_one_test(func, data)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
