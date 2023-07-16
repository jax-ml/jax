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

# flake8: noqa

import datetime
from numpy import array, float32, int32


# Pasted from the test output (see back_compat_test_util.py module docstring)
data_2023_07_16 = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['stablehlo.dynamic_top_k'],
    serialized_date=datetime.date(2023, 7, 16),
    inputs=(array([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.],
       [ 6.,  7.,  8.],
       [ 9., 10., 11.]], dtype=float32),),
    expected_outputs=(array([[ 2.,  1.],
       [ 5.,  4.],
       [ 8.,  7.],
       [11., 10.]], dtype=float32), array([[2, 1],
       [2, 1],
       [2, 1],
       [2, 1]], dtype=int32)),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x?xf32> {jax.arg_info = "a", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<4x?xf32> {jax.result_info = "[0]"}, tensor<4x?xi32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<4x?xf32>) -> tensor<i32> loc(#loc3)
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64> loc(#loc3)
    %2 = stablehlo.constant dense<> : tensor<0xi1> loc(#loc)
    %3 = stablehlo.convert %arg0 : tensor<4x?xf32> loc(#loc)
    %4:2 = call @_wrapped_jax_export_main(%1, %3) : (tensor<i64>, tensor<4x?xf32>) -> (tensor<4x?xf32>, tensor<4x?xi32>) loc(#loc)
    return %4#0, %4#1 : tensor<4x?xf32>, tensor<4x?xi32> loc(#loc)
  } loc(#loc)
  func.func private @_wrapped_jax_export_main(%arg0: tensor<i64> loc(unknown), %arg1: tensor<4x?xf32> {jax.arg_info = "a", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<4x?xf32> {jax.result_info = "[0]"}, tensor<4x?xi32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.convert %arg0 : tensor<i64> loc(#loc4)
    %1 = stablehlo.constant dense<-1> : tensor<i64> loc(#loc5)
    %2 = stablehlo.add %0, %1 : tensor<i64> loc(#loc6)
    %3 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %4:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg1, %3) {api_version = 2 : i32} : (tensor<4x?xf32>, tensor<i32>) -> (tensor<4x?xf32>, tensor<4x?xi32>) loc(#loc5)
    return %4#0, %4#1 : tensor<4x?xf32>, tensor<4x?xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":621:0)
#loc2 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":613:0)
#loc3 = loc("/dimension_size[dimension=1]"(#loc1))
#loc4 = loc("jit(func)/jit(main)/convert_element_type[new_dtype=int64 weak_type=False]"(#loc2))
#loc5 = loc("jit(func)/jit(main)/top_k[k=b + -1]"(#loc2))
#loc6 = loc("jit(func)/jit(main)/add"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01!\x05\x01\x03\x01\x03\x05\x03\x11\x07\t\x0b\r\x0f\x11\x13\x15\x03\xb5\x89\x19\x01Q\x07\x0b\x17\x0f\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f#\x0b\x0b\x0b33\x0f\x0b\x13\x0b\x0f\x0bK\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0b\x17\x13\x13\x0b\x039\x0b\x1b\x13\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x13\x0b\x0b\x0b\x13\x0b\x0b\x0b/\x0b\x0b\x0b\x0b\x0f\x0f\x01\x03\x0f\x03\x177\x0f7\x07\x07\x0f\x13\x1b\x07\x1f\x07\x02R\x04\x1f\x05\x17\x17\x13\x96\t\x01\x1d+\x05\x11\x01\x05\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x05#\x1dGI\x03\x07\x1b\t\x1d\t\x03\x1f\x05%\x05'\x05)\x03\x0b\x0b[\re\x0fU\x03o\x11q\x03\x0b\x0bs\rw\x0fU\x03Y\x11y\x1d'\x05\x05+\x03\x03\x15{\x05-\x1d/\x05\x05/\x03\x113}5\x7f7\x819Q;\x83=Q?QAQ\x051\x053\x055\x057\x059\x05;\x05=\x05?\x03\x03E\x85\x05A\x05C\x17\x13\xb6\t\x01\x03\x03\x15\x87\x03\x03OY\x05E\x03\x01\r\x05]_ac\x03\x05gk\x1dG\x1dI\x03\x03S\x1dK\x1dM\x1dO\x1dQ#\x11\r\x03Wi\x1dS\r\x03Wm\x1dU\x1dW\x1dY\x03\x05uS\r\x01#\x15\x1d[\x1f\x05\x11\xff\xff\xff\xff\xff\xff\xff\xff\x0b\x05\x1d]\x1d_\x05\x01\x13\x0b\x05\x1f\x0f\x01\x01\x02\x02)\x05\x11\x00\xff\xff\xff\xff\xff\xff\xff\xff\x13)\x01\x0b)\x05\x11\x00\xff\xff\xff\xff\xff\xff\xff\xff\t\x1b\x1d)\x01\t)\x03\x01\x17\x11\x03\x03\x05\x03\x07\t\x11\x05\x05\x03\x05\x03\x07\x01\x04\xf3\x05\x01\x11\x01\x19\x07\x03\x01\t\x05\x11\x01!\x05\x03\x0f\x1b\x03\x03\x01\x0f\x07\x17C\x03\r\x03\x01\x03\x06\x17\x03\x05\x03\x03\x07\x03\x01K\x03\x0f\x03\x06\x01\x03\x03\x03\x01\x11\x07\x01M\x05\x03\x07\x05\x05\t\t\x04\x01\x05\x0b\r\x05\x11\x01#\x05\x03\x11\x1b\x05\x05\x01\x03\x01\x03\x06%\x03\x05\x03\x01\x07\x03\x07)\x03\x05\x0b\x06-\x03\x05\x05\x05\x07\x03\x06\x07\x03\r\x03\t\r\x07\x071\x05\x03\x07\x05\x03\x0b\t\x04\x01\x05\r\x0f\x06\x03\x01\x05\x01\x00R\x0ca1\x03\x11\x0f\x0b\t\t\x1b\x1d\x05\x1b3!\x0f;\x15\x1f/!!)#\x1f\x191I\x95\x13%)\r\x83\x1f\x15\x1d\x15\x13\x11-\x1f\x0f\x15\x19\x11\x17\x0f\x0b\x11builtin\x00vhlo\x00module\x00convert_v1\x00func_v1\x00constant_v1\x00return_v1\x00add_v1\x00custom_call_v1\x00get_dimension_size_v1\x00call_v1\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00value\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/convert_element_type[new_dtype=int64 weak_type=False]\x00jit(func)/jit(main)/top_k[k=b + -1]\x00jit(func)/jit(main)/add\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00dimension\x00/dimension_size[dimension=1]\x00callee\x00jax.result_info\x00_wrapped_jax_export_main\x00jax.arg_info\x00a\x00mhlo.sharding\x00{replicated}\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00stablehlo.dynamic_top_k\x00",
    xla_call_module_version=6,
)  # End paste
