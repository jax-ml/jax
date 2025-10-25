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
uint32 = np.uint32


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_17 = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['stablehlo.dynamic_rng_bit_generator'],
    serialized_date=datetime.date(2023, 6, 17),
    inputs=(array([42, 43, 44, 45], dtype=uint32), array([[0., 1., 2.],
       [3., 4., 5.]], dtype=float32)),
    expected_outputs=(array([[2427392780, 1458059130,  393278117, 4094008499],
       [2979149501, 2789479602,  236667834, 2209180022],
       [3272265009,  654898663, 3518128447, 3522413436],
       [3152133794, 2429726816,  183393703, 2087683200],
       [2517633375, 3263052868,  344980918, 3676396031],
       [3594828247, 2571774884, 3751275505, 2435848784]], dtype=uint32),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4xui32> {jax.arg_info = "key", mhlo.sharding = "{replicated}"} loc(unknown), %arg1: tensor<?x?xf32> {jax.arg_info = "a", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<?x4xui32> {jax.result_info = ""}) {
    %0 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x?xf32>) -> tensor<i32> loc(#loc3)
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64> loc(#loc3)
    %2 = stablehlo.get_dimension_size %arg1, dim = 1 : (tensor<?x?xf32>) -> tensor<i32> loc(#loc4)
    %3 = stablehlo.convert %2 : (tensor<i32>) -> tensor<i64> loc(#loc4)
    %4 = stablehlo.constant dense<> : tensor<0xi1> loc(#loc)
    %5 = stablehlo.convert %arg0 : tensor<4xui32> loc(#loc)
    %6 = stablehlo.convert %arg1 : tensor<?x?xf32> loc(#loc)
    %7 = call @_wrapped_jax_export_main(%1, %3, %5, %6) : (tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<?x?xf32>) -> tensor<?x4xui32> loc(#loc)
    return %7 : tensor<?x4xui32> loc(#loc)
  } loc(#loc)
  func.func private @_wrapped_jax_export_main(%arg0: tensor<i64> loc(unknown), %arg1: tensor<i64> loc(unknown), %arg2: tensor<4xui32> {jax.arg_info = "key", mhlo.sharding = "{replicated}"} loc(unknown), %arg3: tensor<?x?xf32> {jax.arg_info = "a", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<?x4xui32> {jax.result_info = ""}) {
    %0 = stablehlo.reshape %arg2 : (tensor<4xui32>) -> tensor<2x2xui32> loc(#loc5)
    %1 = stablehlo.bitcast_convert %0 : (tensor<2x2xui32>) -> tensor<2xui64> loc(#loc5)
    %2 = stablehlo.multiply %arg0, %arg1 : tensor<i64> loc(#loc6)
    %3 = stablehlo.convert %2 : tensor<i64> loc(#loc7)
    %4 = stablehlo.constant dense<10> : tensor<i64> loc(#loc5)
    %5 = stablehlo.multiply %3, %4 : tensor<i64> loc(#loc6)
    %6 = stablehlo.constant dense<4> : tensor<i64> loc(#loc5)
    %7 = stablehlo.convert %5 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %9 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %11 = stablehlo.concatenate %8, %10, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc5)
    %12:2 = stablehlo.custom_call @stablehlo.dynamic_rng_bit_generator(%1, %11) {api_version = 2 : i32, rng_algorithm = #stablehlo<rng_algorithm DEFAULT>} : (tensor<2xui64>, tensor<2xi32>) -> (tensor<2xui64>, tensor<?x4xui32>) loc(#loc5)
    %13 = stablehlo.bitcast_convert %12#0 : (tensor<2xui64>) -> tensor<2x2xui32> loc(#loc5)
    %14 = stablehlo.reshape %13 : (tensor<2x2xui32>) -> tensor<4xui32> loc(#loc5)
    %15 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %16 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %17 = stablehlo.multiply %arg0, %arg1 : tensor<i64> loc(#loc6)
    %18 = stablehlo.convert %17 : tensor<i64> loc(#loc7)
    %19 = stablehlo.constant dense<10> : tensor<i64> loc(#loc8)
    %20 = stablehlo.multiply %18, %19 : tensor<i64> loc(#loc6)
    %21 = stablehlo.constant dense<4> : tensor<i64> loc(#loc8)
    %22 = stablehlo.constant dense<10> : tensor<i64> loc(#loc8)
    %23 = stablehlo.constant dense<1> : tensor<i64> loc(#loc8)
    %24 = stablehlo.convert %15 : (tensor<i64>) -> tensor<i32> loc(#loc8)
    %25 = stablehlo.reshape %24 : (tensor<i32>) -> tensor<1xi32> loc(#loc8)
    %26 = stablehlo.convert %16 : (tensor<i64>) -> tensor<i32> loc(#loc8)
    %27 = stablehlo.reshape %26 : (tensor<i32>) -> tensor<1xi32> loc(#loc8)
    %28 = stablehlo.concatenate %25, %27, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc8)
    %29 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32> loc(#loc8)
    %30 = stablehlo.reshape %29 : (tensor<i32>) -> tensor<1xi32> loc(#loc8)
    %31 = stablehlo.convert %21 : (tensor<i64>) -> tensor<i32> loc(#loc8)
    %32 = stablehlo.reshape %31 : (tensor<i32>) -> tensor<1xi32> loc(#loc8)
    %33 = stablehlo.concatenate %30, %32, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc8)
    %34 = stablehlo.convert %22 : (tensor<i64>) -> tensor<i32> loc(#loc8)
    %35 = stablehlo.reshape %34 : (tensor<i32>) -> tensor<1xi32> loc(#loc8)
    %36 = stablehlo.convert %23 : (tensor<i64>) -> tensor<i32> loc(#loc8)
    %37 = stablehlo.reshape %36 : (tensor<i32>) -> tensor<1xi32> loc(#loc8)
    %38 = stablehlo.concatenate %35, %37, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc8)
    %39 = stablehlo.real_dynamic_slice %12#1, %28, %33, %38 : (tensor<?x4xui32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xui32> loc(#loc8)
    return %39 : tensor<?x4xui32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":260:0)
#loc2 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":697:0)
#loc3 = loc("/dimension_size[dimension=0]"(#loc1))
#loc4 = loc("/dimension_size[dimension=1]"(#loc1))
#loc5 = loc("jit(func)/jit(main)/rng_bit_generator[shape=(10*b0*b1, 4) dtype=uint32 algorithm=RNG_DEFAULT]"(#loc2))
#loc6 = loc("jit(func)/jit(main)/mul"(#loc2))
#loc7 = loc("jit(func)/jit(main)/convert_element_type[new_dtype=int64 weak_type=False]"(#loc2))
#loc8 = loc("jit(func)/jit(main)/slice[start_indices=(0, 0) limit_indices=(10*b0*b1, 4) strides=(10, 1)]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03\xdd\xa3'\x01c\x0f\x07\x0f\x0b\x13\x17\x0f\x0b\x13\x0f\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x13\x0f\x17\x0f#\x0b\x0b\x0b33\x0b\x0b\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x13\x0b\x13\x13\x0b\x03A\x0b\x1b\x0b\x0b\x0b\x1b\x0f\x0b\x0b\x0b\x13\x0b\x0b\x0b\x13\x0b\x0b\x0b\x1b\x0b\x0b//\x0f\x0b\x0b\x0b\x0b//\x0f\x0f\x01\x03\x0f\x03%\x0f\x0f\x13\x13W7\x13\x07\x07\x07\x17\x13\x13\x1b\x07#\x07\x07\x02\xea\x05\x1dS\x0b\x1f\x1d9\x0b\x05\x1f\x03\x03#\x91\x17\x1d\xe6\n\x01\x1d;\x0b\x05!\x03\x03\x07\x8d\x11\x01\x05\x05#\x05%\x05'\x05)\x05+\x1d=\x0b\x03\x03\x07\x8f\x05-\x03\x03\x07\x9b\x1dW)\x17\x1d\x12\x04\x01\x1d[)\x03\x07/\x131\x13\x0f3\x05/\x051\x053\x03\x0b\x15w\x17}\x19o\x0f\x83\x1b\x85\x03\x0b\x15\x87\x17\x89\x19o\x0fu\x1b\x8b\x055\x057\x059\x03\x13A\x93CqE\x95GcI\x97KcMcOcQ\x99\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x05I\x05K\x05M\x03\x03\x07\x9d\x05O\x03\x03#\x9f\x05Q\x03\x03\x07\xa1\x03\x03au\x05S\x03\x01\r\x05gyik\x1dU\x1dW\x1dY\r\x05g{ik\x03\x03\x7f\x1d[\r\x01\x1d]\x03\x05em\x1d_\x1da#\x1d\r\x03\x81q\x1dc\x1de\x1dg\x03\tssem#!\x1di\x1f\x03\x11\n\x00\x00\x00\x00\x00\x00\x00\x1f\x03\x11\x04\x00\x00\x00\x00\x00\x00\x00\x13\x13\x01\x0b\x05\x1dk\x05\x01\x19\x01\x1f\x03\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x03\x11\x01\x00\x00\x00\x00\x00\x00\x00\x13\x13\x05\x1f\x1b\x01\x01\x02\x02)\x01\x13)\x01\x15)\x03\x05\x15)\x03\x11\x11)\x05\x00\xff\xff\xff\xff\xff\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\xff\x1f)\x05\x00\xff\xff\xff\xff\xff\xff\xff\xff\x11\x11)\x03\t\x15%\x1d\x1b)\x05\t\t\x11)\x03\t#)\x03\x01%\x11\x05\t\x0b\x03\r\t\x11\t\x03\x03\t\x0b\x03\r'\x01\x04B\x06\x05\x01\x11\x03-\x07\x03\x01\t\r\x11\x035\x05\x03\x15'\x05\t\x03\x0b\x03\x13\x07'\t\x03\x05\x03\x03\x03\x06'\x03\x03\x03\x05\x13\x07+Y\x03\x05\x03\x03\x03\x06+\x03\x03\x03\t\x07\x03\x03]\x03\x1b\x03\x06\x03\x03\t\x03\x01\x03\x06\x03\x03\x0b\x03\x03\x19\x07\x03_\x03\r\t\x07\x0b\x0f\x11\x11\x04\x03\x03\x13\r\x11\x037\x05\x03[\xa7\t\x03\x03\x03\x03\t\x03\x0b\x03\x05\x06\x05\x03\x17\x03\x05\x0f\x06\x05\x03\x19\x03\t\t\x06\r\x03\x03\x05\x01\x03\x03\x06\x1f\x03\x03\x03\r\x07\x03\x05\x11\x03\x03\t\x06\r\x03\x03\x05\x0f\x11\x07\x03\x05!\x03\x03\x03\x06\x05\x03\x05\x03\x13\x05\x06\x05\x03\x07\x03\x17\x03\x06\x05\x03\x05\x03\x15\x05\x06\x05\x03\x07\x03\x1b\x0b\x07\x05\t\x03\x0f\x05\x19\x1d\x15\x07\x05?\x05\x19\r\x05\x0b\x1f\x0f\x06\x05\x03\x17\x03!\x05\x06\x05\x03\t\x03%\x07\x03\x01%\x03\x03\x07\x03\x01%\x03\x03\t\x06\r\x03\x03\x05\x01\x03\x03\x06\x1f\x03\x03\x03-\x07\x03\x01\x11\x03\x03\t\x06\r\x03\x03\x05/1\x07\x03\x01!\x03\x03\x07\x03\x01\x11\x03\x03\x07\x03\x01U\x03\x03\x03\x06\x01\x03\x05\x03)\x05\x06\x01\x03\x07\x03;\x03\x06\x01\x03\x05\x03+\x05\x06\x01\x03\x07\x03?\x0b\x07\x01\t\x03\x0f\x05=A\x03\x06\x01\x03\x05\x033\x05\x06\x01\x03\x07\x03E\x03\x06\x01\x03\x05\x035\x05\x06\x01\x03\x07\x03I\x0b\x07\x01\t\x03\x0f\x05GK\x03\x06\x01\x03\x05\x037\x05\x06\x01\x03\x07\x03O\x03\x06\x01\x03\x05\x039\x05\x06\x01\x03\x07\x03S\x0b\x07\x01\t\x03\x0f\x05QU\x17\x06\x01\x03\r\t#CMW\x11\x04\x03\x03Y\x06\x03\x01\x05\x01\x00\xae\x10mI\x11\x0f\x0b!\x05\t3\x03\x1b\x1d\x1b\x0f;;\xb9\x1d\x1f/!!)#\x1f\x19\x951\xbd\x13%)\x15\x83\x1f\x15\x1d\x15\x13\r\x11-\x1f-\x15'\x11\x1f\x19\x19\x17\x17\x0f\x0b\x11builtin\x00vhlo\x00module\x00convert_v1\x00reshape_v1\x00constant_v1\x00multiply_v1\x00concatenate_v1\x00func_v1\x00bitcast_convert_v1\x00return_v1\x00get_dimension_size_v1\x00custom_call_v1\x00real_dynamic_slice_v1\x00call_v1\x00value\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00dimension\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/rng_bit_generator[shape=(10*b0*b1, 4) dtype=uint32 algorithm=RNG_DEFAULT]\x00jit(func)/jit(main)/mul\x00jit(func)/jit(main)/convert_element_type[new_dtype=int64 weak_type=False]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00rng_algorithm\x00jit(func)/jit(main)/slice[start_indices=(0, 0) limit_indices=(10*b0*b1, 4) strides=(10, 1)]\x00/dimension_size[dimension=0]\x00/dimension_size[dimension=1]\x00callee\x00jax.arg_info\x00mhlo.sharding\x00{replicated}\x00\x00_wrapped_jax_export_main\x00key\x00a\x00jax.result_info\x00main\x00public\x00private\x00stablehlo.dynamic_rng_bit_generator\x00",
    xla_call_module_version=6,
)  # End paste
