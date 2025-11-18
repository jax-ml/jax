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

# ruff: noqa

import datetime
import numpy as np

array = np.array
float32 = np.float32
int32 = np.int32

# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_03_21 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['LuDecomposition'],
    serialized_date=datetime.date(2023, 3, 21),
    inputs=(),
    expected_outputs=(array([[6. , 7. , 8. ],
       [0. , 1. , 2. ],
       [0.5, 0.5, 0. ]], dtype=float32), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "[0]"}, tensor<3xi32> {jax.result_info = "[1]"}, tensor<3xi32> {jax.result_info = "[2]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf32>
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32>
    %2:3 = call @lu(%1) : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>)
    return %2#0, %2#1, %2#2 : tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>
  }
  func.func private @lu(%arg0: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>) {
    %0 = call @xla_fallback_lu(%arg0) : (tensor<3x3xf32>) -> tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>>) -> tensor<3x3xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>>) -> tensor<3xi32>
    %3 = stablehlo.get_tuple_element %0[2] : (tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>>) -> tensor<3xi32>
    return %1, %2, %3 : tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>
  }
  func.func private @xla_fallback_lu(%arg0: tensor<3x3xf32>) -> tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>> {
    %0 = stablehlo.custom_call @LuDecomposition(%arg0) {xla_shape = "(f32[3,3]{1,0}, s32[3]{0}, s32[3]{0})"} : (tensor<3x3xf32>) -> tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>>) -> tensor<3x3xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>>) -> tensor<3xi32>
    %3 = stablehlo.get_tuple_element %0[2] : (tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>>) -> tensor<3xi32>
    %4 = stablehlo.tuple %1, %2, %3 {xla_shape = "(f32[3,3]{1,0}, s32[3]{0}, s32[3]{0})"} : tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>>
    return %4 : tuple<tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32>>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01!\x05\x01\x05\x01\x03\x05\x03\x11\x07\t\x0b\r\x0f\x11\x13\x15\x03\xbd\x99\x15\x01e\x07\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x0b\x17\x13\x0b33\x0b\x173S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0f\x0b\x13\x13\x0b\x0f\x0b\x0f\x0b\x13\x035\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0f\x0f\x03\x15\x13\x17\x07\x17\x07\x1b\x1f\x17\x13\x07\x02f\x04\x1f\x1d')\x05\x17\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x05#\x05%\x03\x03\x0f\x91\x03\x03\x0f\x93\x03\x03\x0f\x95\x05'\x17\x11\xaa\x06\x01\x03\x03\x05!\x05)\x03\x0b\x07e\tq\x0bs\x05\x81\r\x83\x03\x0b\x07e\t\x85\x0be\x05i\rk\x05+\x17\x11\xae\x06\x01\x03\x0b\x07e\t\x87\x0be\x05m\rk\x03\x13/\x891\x8b3\x8d5e7\x8f9e;e=e\x13o\x05-\x05/\x051\x053\x055\x057\x059\x05;\x1dA\x01\x05=\x1dE\x01\x05?\x1dI\x01\x05A\x1dM\x01\x05C\x03\x03\x13o\x1dS\x01\x05E\x03\x03\x1bm\x03\x03Y\x97\x05G\x1d]\x1d\x05I\x1da\x1d\x05K\x03\x03\x1bi\x03\x01\x1dM\x1dO\x1dQ\x1dS\x1dU#\x0b\x03\x07uy}\r\x03gw\x1dW\r\x03g{\x1dY\r\x03g\x7f\x1d[\x1d]\x1d_#\r#\x0f\x0b\x03\x1da\x1dc\x05\x01\x13\x05\x01\x13\x05\x05\x13\x05\t\x13\x13\x01)\x03\r\x05)\x05\r\r\t\x1b/\x07\x03\x01\x01\t\x11\x01\x07\x03\x01\x01\x11\x03\x03\x07\x03\x01\x01\x11\x03\x03\x03\x07)\x03%\t\x1d\x04n\x02\x05\x01\x11\x01\x1f\x07\x03\x01\r\x05\x11\x01#\x05\x03\x0b\x11\x0f\x03[W\x03\x11\x11\x06_\x03\x03\x03\x01\t\x07\x03c\x07\x03\x01\x01\x03\x03\x07\x04\x01\x07\x05\x07\t\x05\x11\x03%\x05\x03\x0b\x17\x03\x03\x01\t\x07\x03U\x03\x07\x03\x01\x03\x07\x03\x15\x03\x03\x03\x03\x03\x07\x03\x17\x03\x01\x03\x03\x03\x07\x03\x19\x03\x01\x03\x03\x07\x04\x03\x07\x05\x07\t\x05\x11\x01+\x05\x03\r\x1b\x03\x03\x01\x0b\x07?-\x03\x07\x03\x01\x03\x07C\x15\x03\x03\x03\x03\x03\x07G\x17\x03\x01\x03\x03\x03\x07K\x19\x03\x01\x03\x03\r\x07QO\x03\x07\x07\x05\x07\t\x07\x04\x01\x03\x0b\x06\x03\x01\x05\x01\x00\x06\re!\x03\x0f\x0b\t\t\tM!\x11\x07!\x85\x87\x1f\x11)))\x1d\x1f/!!)#\x1f\x197\x1b\x0f\x15\x83\r\x1f\x15\x1d\x15\x13\x17\x11\x13\x1f\x11\x15\x11+\x0f\x0b\x11builtin\x00vhlo\x00module\x00get_tuple_element_v1\x00func_v1\x00return_v1\x00call_v1\x00custom_call_v1\x00tuple_v1\x00iota_v1\x00reshape_v1\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00index\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00xla_shape\x00callee\x00jit__lambda_\x00jit(<lambda>)/jit(main)/lu\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00custom-call.2\x00get-tuple-element.3\x00get-tuple-element.4\x00get-tuple-element.5\x00tuple.6\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jax.result_info\x00lu\x00private\x00xla_fallback_lu\x00(f32[3,3]{1,0}, s32[3]{0}, s32[3]{0})\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00LuDecomposition\x00",
    xla_call_module_version=4,
)  # End paste
