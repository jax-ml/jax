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
from numpy import array, float32, int32


# Pasted from the test output (see back_compat_test.py module docstring)
data_unary_2023_06_17 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['stablehlo.dynamic_reduce_window'],
    serialized_date=datetime.date(2023, 6, 16),
    inputs=(array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.]], dtype=float32),),
    expected_outputs=(array([[ 0.,  1.,  2.,  3.],
       [ 4.,  6.,  8., 10.],
       [12., 15., 18., 21.]], dtype=float32),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<?x4xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<?x4xf32> {jax.result_info = ""}) {
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x4xf32>) -> tensor<i32> loc(#loc3)
    %1 = stablehlo.constant dense<> : tensor<0xi1> loc(#loc)
    %2 = stablehlo.convert %arg0 : tensor<?x4xf32> loc(#loc)
    %3 = call @_wrapped_jax_export_main(%0, %2) : (tensor<i32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(#loc)
    return %3 : tensor<?x4xf32> loc(#loc)
  } loc(#loc)
  func.func @reduce_window_stablehlo.add_float32_reducer(%arg0: tensor<f32> loc(unknown), %arg1: tensor<f32> loc(unknown)) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32> loc(#loc4)
    return %0 : tensor<f32> loc(#loc4)
  } loc(#loc4)
  func.func private @_wrapped_jax_export_main(%arg0: tensor<i32> loc(unknown), %arg1: tensor<?x4xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<?x4xf32> {jax.result_info = ""}) {
    %0 = call @_cumulative_reduction(%arg0, %arg1) : (tensor<i32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(#loc5)
    return %0 : tensor<?x4xf32> loc(#loc)
  } loc(#loc)
  func.func private @_cumulative_reduction(%arg0: tensor<i32> loc(unknown), %arg1: tensor<?x4xf32> loc(unknown)) -> tensor<?x4xf32> {
    %0 = call @cumsum(%arg0, %arg1) : (tensor<i32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(#loc6)
    return %0 : tensor<?x4xf32> loc(#loc5)
  } loc(#loc5)
  func.func private @cumsum(%arg0: tensor<i32> loc(unknown), %arg1: tensor<?x4xf32> loc(unknown)) -> tensor<?x4xf32> {
    %0 = stablehlo.convert %arg0 : tensor<i32> loc(#loc7)
    %1 = stablehlo.constant dense<-1> : tensor<i32> loc(#loc4)
    %2 = stablehlo.add %0, %1 : tensor<i32> loc(#loc8)
    %3 = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %4 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %5 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %6 = stablehlo.concatenate %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc4)
    %7 = stablehlo.reshape %6 : (tensor<2xi32>) -> tensor<1x2xi32> loc(#loc4)
    %8 = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %9 = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %10 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %11 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %12 = stablehlo.concatenate %10, %11, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc4)
    %13 = stablehlo.reshape %12 : (tensor<2xi32>) -> tensor<1x2xi32> loc(#loc4)
    %14 = stablehlo.concatenate %7, %13, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32> loc(#loc4)
    %15 = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc4)
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<f32>) -> tensor<f32> loc(#loc4)
    %17 = stablehlo.constant dense<1> : tensor<i32> loc(#loc4)
    %18 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %19 = stablehlo.reshape %17 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %20 = stablehlo.concatenate %18, %19, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc4)
    %21 = stablehlo.constant dense<1> : tensor<i32> loc(#loc4)
    %22 = stablehlo.constant dense<1> : tensor<i32> loc(#loc4)
    %23 = stablehlo.reshape %21 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %24 = stablehlo.reshape %22 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %25 = stablehlo.concatenate %23, %24, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc4)
    %26 = stablehlo.constant dense<1> : tensor<i32> loc(#loc4)
    %27 = stablehlo.constant dense<1> : tensor<i32> loc(#loc4)
    %28 = stablehlo.reshape %26 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %29 = stablehlo.reshape %27 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %30 = stablehlo.concatenate %28, %29, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc4)
    %31 = stablehlo.constant dense<1> : tensor<i32> loc(#loc4)
    %32 = stablehlo.constant dense<1> : tensor<i32> loc(#loc4)
    %33 = stablehlo.reshape %31 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %34 = stablehlo.reshape %32 : (tensor<i32>) -> tensor<1xi32> loc(#loc4)
    %35 = stablehlo.concatenate %33, %34, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc4)
    %36 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg1, %16, %20, %25, %30, %35, %14) {api_version = 2 : i32, called_computations = [@reduce_window_stablehlo.add_float32_reducer]} : (tensor<?x4xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x4xf32> loc(#loc4)
    return %36 : tensor<?x4xf32> loc(#loc6)
  } loc(#loc6)
} loc(#loc)
#loc1 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":250:0)
#loc2 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":627:0)
#loc3 = loc("/dimension_size[dimension=0]"(#loc1))
#loc4 = loc("jit(func)/jit(main)/jit(_cumulative_reduction)/reduce_window_sum[window_dimensions=(b, 1) window_strides=(1, 1) padding=((b + -1, 0), (0, 0)) base_dilation=(1, 1) window_dilation=(1, 1)]"(#loc2))
#loc5 = loc("jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=_cumulative_reduction keep_unused=False inline=False]"(#loc2))
#loc6 = loc("jit(func)/jit(main)/jit(_cumulative_reduction)/cumsum[axis=0 reverse=False]"(#loc2))
#loc7 = loc("jit(func)/jit(main)/jit(_cumulative_reduction)/convert_element_type[new_dtype=int32 weak_type=False]"(#loc2))
#loc8 = loc("jit(func)/jit(main)/jit(_cumulative_reduction)/add"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01'\x05\x01\x03\x01\x03\x05\x03\x17\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x03\xeb\xb5#\x01m\x0f\x07\x13\x13\x0b\x0b\x0b\x0b\x0b\x17\x0b\x0f\x0f\x13\x0b\x0f\x0b#\x0b\x0b\x0b33\x0b33\x0b3\x0b\x0f\x0b\x13\x0f\x0b\x0b\x13\x13\x0bK\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x0f\x0b\x17\x13\x13\x03I\x0b\x0b\x0b\x0b\x0b\x1b\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x13\x13\x0f\x1f\x1f\x0f\x1f\x0f\x1f\x0b\x0b\x0f\x0b\x0f\x01\x03\x0f\x03!\x0f7\x13\x0f\x13\x07\x07\x07\x17\x13\x17\x1b\x1b\x17\x13\x07\x02\x1a\x06\x1d/\x13\x1f\x03\x03E\xa3\x03\x03\x15\xa9\x05\x1d\x05\x1f\x05!\x05#\x05%\x17!\xce\t\x01\x05'\x1d5\x13\x1d9\x13\x03\x03\x15\xa1\x05)\x11\x01\x05\x05+\x03\x07%\x1f'\x1f\t)\x05-\x05/\x051\x03\x0b\x0b\x83\r\x8d\x0fy\t\x93\x11\x95\x03\x0b\x0bm\r\x97\x0fm\t{\x11q\x053\x03\x0b\x0b\x99\rs\x0fy\t}\x11u\x03\x0b\x0b\x9b\rs\x0f\x9d\t\x7f\x11u\x055\x03\x0b\x0bm\rs\x0fm\t\x81\x11u\x057\x1d=\x13\x059\x03\x03\x15\x9f\x1dC\x13\x05;\x05=\x03\x03\x15\xa5\x03\x03K\xa7\x05?\x03\x11O\xabQqS\xadU\xafW\xb1Ym[m]m\x05A\x05C\x05E\x05G\x05I\x05K\x05M\x05O\x03\x03\x1d\x81\x03\x03\x1d\x7f\x1deg\x05Q\x17!\xea\x03\x01\x03\x03\x15\xb3\x03\x03\x1d}\x03\x01\r\x01\x1dS#\x1b\x1dU\r\x05\x85\x87\x89\x8b\x03\x03\x8f\x1dW\x1dY\x1d[\x1d]\x03\x03w\x1d_\x1da\x1dc\x1de#\x17\r\x03\x91q\x1dg\x1di\x1dk#\x19\x03\x05ow\x03\x05oo\x03\x03o\x1f\x03\t\xff\xff\xff\xff\x1f\x03\t\x00\x00\x00\x00\x13\x11\x01\x1f\t\t\x00\x00\x00\x00\x1f\x1f\x01\x1f\x03\t\x01\x00\x00\x00\x0b\x05\x1dm\x03\x03{\x05\x01\x1f\x15\x01\x01\x02\x02)\x01\r)\x05\x00\xff\xff\xff\xff\xff\xff\xff\xff\x11\x0f)\x03\x05\r)\x01\x0f)\x03\t\r\x1b\t\x1d)\x05\x05\t\r)\x03\x01!\x11\x03\x05\x03\x05\x11\x05\t\t\x03\t\x11\x05\x03\x05\x03\x05)\x05\t\t\r)\x03\x01\x11\x01\x04\x9e\x06\x05\x01\x11\x03#\x07\x03\x01\x15\t\x11\x03+\x05\x03\x0b\x17\x03\x05\x03\x17\x07c\x05\x03\x03\x03\x01\x05\x03\x03i\x03\x15\x0f\x06\x03\x03\x05\x03\x01\r\x07\x03k\x03\x05\x05\x03\x07\x0b\x04\x03\x03\t\t\x11\x01-\x05\x03\x07\x0b\x05\t\x03\t\x03\x11\x06\x01\x03\t\x05\x01\x03\x0b\x04\x01\x03\x05\t\x11\x031\x05\x03\x07\x0b\x05\x03\x03\x05\x03\r\x07\x17a\x03\x05\x05\x01\x03\x0b\x04\x03\x03\x05\t\x11\x173\x05\x03\x07\x0b\x05\x03\x03\x05\x03\r\x07\x19_\x03\x05\x05\x01\x03\x0b\x04\x17\x03\x05\t\x11\x197\x05\x03O\x9b\x05\x03\x03\x05\x03\x0f\x06;\x03\x03\x03\x01\x05\x03\x01?\x03\x03\x11\x06A\x03\x03\x05\x05\x07\x05\x03\x01\x1b\x03\x03\x03\x06\x01\x03\x07\x03\t\x03\x06\x01\x03\x07\x03\x0b\x07\x07\x01\x05\x03\x0b\x05\r\x0f\x03\x06\x01\x03\x13\x03\x11\x05\x03\x01\x1b\x03\x03\x05\x03\x01\x1b\x03\x03\x03\x06\x01\x03\x07\x03\x15\x03\x06\x01\x03\x07\x03\x17\x07\x07\x01\x05\x03\x0b\x05\x19\x1b\x03\x06\x01\x03\x13\x03\x1d\x07\x07\x01\x05\x03\x1d\x05\x13\x1f\x05\x03\x01G\x03\t\x13\x07\x01I\x03\t\x03#\x05\x03\x01\x07\x03\x03\x03\x06\x01\x03\x07\x03\x01\x03\x06\x01\x03\x07\x03'\x07\x07\x01\x05\x03\x0b\x05)+\x05\x03\x01\x07\x03\x03\x05\x03\x01\x07\x03\x03\x03\x06\x01\x03\x07\x03/\x03\x06\x01\x03\x07\x031\x07\x07\x01\x05\x03\x0b\x0535\x05\x03\x01\x07\x03\x03\x05\x03\x01\x07\x03\x03\x03\x06\x01\x03\x07\x039\x03\x06\x01\x03\x07\x03;\x07\x07\x01\x05\x03\x0b\x05=?\x05\x03\x01\x07\x03\x03\x05\x03\x01\x07\x03\x03\x03\x06\x01\x03\x07\x03C\x03\x06\x01\x03\x07\x03E\x07\x07\x01\x05\x03\x0b\x05GI\x15\x07\x01M\x03\x05\x0f\x03%-7AK!\x0b\x04\x19\x03M\x06\x03\x01\x05\x01\x00\x0e\x16oA\x0f\x0b!\x1b\x1d\x05\x1b\x0f-3Y\x11\x03;\x1f/!!)#\x1f\x19+\x15g\xcb\x99\x06\x03\xee\x02\x13%)\x83\x0f\r\x1f\x15\x1d\x15\x13-\x1f)\x0f\x17\x11\x15\x11\x1f\x19\x17\x0f\x0b\x11builtin\x00vhlo\x00module\x00reshape_v1\x00constant_v1\x00concatenate_v1\x00func_v1\x00return_v1\x00call_v1\x00convert_v1\x00add_v1\x00broadcast_in_dim_v1\x00custom_call_v1\x00get_dimension_size_v1\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00value\x00callee\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/jit(_cumulative_reduction)/reduce_window_sum[window_dimensions=(b, 1) window_strides=(1, 1) padding=((b + -1, 0), (0, 0)) base_dilation=(1, 1) window_dilation=(1, 1)]\x00jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=_cumulative_reduction keep_unused=False inline=False]\x00jit(func)/jit(main)/jit(_cumulative_reduction)/cumsum[axis=0 reverse=False]\x00jit(func)/jit(main)/jit(_cumulative_reduction)/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(func)/jit(main)/jit(_cumulative_reduction)/add\x00dimension\x00broadcast_dimensions\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00/dimension_size[dimension=0]\x00\x00private\x00reduce_window_stablehlo.add_float32_reducer\x00_wrapped_jax_export_main\x00_cumulative_reduction\x00cumsum\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00main\x00public\x00stablehlo.dynamic_reduce_window\x00",
    xla_call_module_version=6,
)  # End paste

# Pasted from the test output (see back_compat_test.py module docstring)
data_variadic_2023_06_17 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['stablehlo.dynamic_reduce_window'],
    serialized_date=datetime.date(2023, 6, 17),
    inputs=(array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.]], dtype=float32), array([[100, 101, 102, 103],
       [104, 105, 106, 107],
       [108, 109, 110, 111]], dtype=int32)),
    expected_outputs=(array([[27., 33.],
       [51., 57.]], dtype=float32), array([[-608, -614],
       [-632, -638]], dtype=int32)),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<?x4xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"} loc(unknown), %arg1: tensor<?x4xi32> {jax.arg_info = "y", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<?x?xf32> {jax.result_info = "[0]"}, tensor<?x?xi32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x4xf32>) -> tensor<i32> loc(#loc5)
    %1 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x4xi32>) -> tensor<i32> loc(#loc5)
    %2 = stablehlo.constant dense<> : tensor<0xi1> loc(#loc)
    %3 = stablehlo.convert %arg0 : tensor<?x4xf32> loc(#loc)
    %4 = stablehlo.convert %arg1 : tensor<?x4xi32> loc(#loc)
    %5:2 = call @_wrapped_jax_export_main(%0, %3, %4) : (tensor<i32>, tensor<?x4xf32>, tensor<?x4xi32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) loc(#loc)
    return %5#0, %5#1 : tensor<?x?xf32>, tensor<?x?xi32> loc(#loc)
  } loc(#loc)
  func.func @generic_reduce_window_reducer(%arg0: tensor<f32> loc(unknown), %arg1: tensor<i32> loc(unknown), %arg2: tensor<f32> loc(unknown), %arg3: tensor<i32> loc(unknown)) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.add %arg0, %arg2 : tensor<f32> loc(#loc7)
    %1 = stablehlo.subtract %arg1, %arg3 : tensor<i32> loc(#loc8)
    return %0, %1 : tensor<f32>, tensor<i32> loc(#loc6)
  } loc(#loc6)
  func.func private @_wrapped_jax_export_main(%arg0: tensor<i32> loc(unknown), %arg1: tensor<?x4xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"} loc(unknown), %arg2: tensor<?x4xi32> {jax.arg_info = "y", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<?x?xf32> {jax.result_info = "[0]"}, tensor<?x?xi32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(#loc)
    %1 = stablehlo.constant dense<2> : tensor<i32> loc(#loc)
    %2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %3 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %4 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %5 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %6 = stablehlo.concatenate %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc6)
    %7 = stablehlo.reshape %6 : (tensor<2xi32>) -> tensor<1x2xi32> loc(#loc6)
    %8 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %9 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %10 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %11 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %12 = stablehlo.concatenate %10, %11, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc6)
    %13 = stablehlo.reshape %12 : (tensor<2xi32>) -> tensor<1x2xi32> loc(#loc6)
    %14 = stablehlo.concatenate %7, %13, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32> loc(#loc6)
    %15 = stablehlo.constant dense<2> : tensor<i32> loc(#loc6)
    %16 = stablehlo.reshape %15 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %17 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %18 = stablehlo.concatenate %16, %17, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc6)
    %19 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %20 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %21 = stablehlo.reshape %19 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %22 = stablehlo.reshape %20 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %23 = stablehlo.concatenate %21, %22, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc6)
    %24 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %25 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %26 = stablehlo.reshape %24 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %27 = stablehlo.reshape %25 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %28 = stablehlo.concatenate %26, %27, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc6)
    %29 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %30 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %31 = stablehlo.reshape %29 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %32 = stablehlo.reshape %30 : (tensor<i32>) -> tensor<1xi32> loc(#loc6)
    %33 = stablehlo.concatenate %31, %32, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc6)
    %34:2 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg1, %arg2, %0, %1, %18, %23, %28, %33, %14) {api_version = 2 : i32, called_computations = [@generic_reduce_window_reducer]} : (tensor<?x4xf32>, tensor<?x4xi32>, tensor<f32>, tensor<i32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xi32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) loc(#loc6)
    return %34#0, %34#1 : tensor<?x?xf32>, tensor<?x?xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":250:0)
#loc2 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":648:0)
#loc3 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":650:0)
#loc4 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":651:0)
#loc5 = loc("/dimension_size[dimension=0]"(#loc1))
#loc6 = loc("jit(func)/jit(main)/reduce_window[consts=() window_dimensions=(2, b) window_strides=(1, 1) padding=((0, 0), (0, 0)) base_dilation=(1, 1) window_dilation=(1, 1)]"(#loc2))
#loc7 = loc("jit(func)/jit(main)/add"(#loc3))
#loc8 = loc("jit(func)/jit(main)/sub"(#loc4))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01\'\x05\x01\x03\x01\x03\x05\x03\x17\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x03\xdf\xa5\'\x01]\x0f\x07\x13\x13\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0f\x13\x0f#\x0b\x0b\x0b33\x0b\x173\x13\x0bK\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x17\x0b\x17\x13\x13\x0b\x03I\x0b\x1b\x0b\x0b\x0b\x1b\x13\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x13\x0b\x13\x0b\x0b\x0b\x0b\x17\x0b\x0b\x0b\x1f\x1f\x1f\x0f\x1f\x0b\x0b\x0f\x0b\x0f\x01\x03\x0f\x03%\x0f\x13\x0f\x07\x1377WW\x07\x17\x13\x1f\'#\x07\x17\x07\x02~\x06\x1d+-\x1f\x03\x033\x97\x03\x03\t\x99\x05\x1d\x05\x1f\x05!\x03\x03\t\x95\x05#\x05%\x05\'\x05)\x11\x01\x05\x03\x03\t\x93\x1dSU\x03\x07!\x19#\x19\x0b%\x05+\x05-\x05/\x03\x0b\x11s\x13y\x15i\x0b\x83\x17\x85\x03\x0b\x11]\x13\x87\x15]\x0bm\x17o\x051\x17\r"\n\x01\x03\x0b\x11\x89\x13\x8d\x15i\x0bq\x17\x8f\x03\x03\t\x91\x053\x03\x117\x9b9o;\x9d=\x9f?\xa1A]C]E]\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x1dIK\x05E\x17\r*\n\x01\x1dOQ\x05G\x17\r.\n\x01\x05I\x17\r\xea\x03\x01\x03\x03\t\xa3\x03\x03[q\x05K\x03\x01\r\x05auce\x1dM\x1dO\x1dQ\r\x05awce\x03\x05{\x7f\x1dS\x1dU\x1dW\x1dY\x03\x05_g\x1d[\x1d]#\x1b\r\x03k}\x1d_\r\x03k\x81\x1da\x1dc\x1de#\x1d\x03\x07\x8b_g\r\x01#\x1f\x1dg\x1f\x07\t\x00\x00\x80?\x1f\x03\t\x02\x00\x00\x00\x1f\x03\t\x00\x00\x00\x00\x13!\x01\x1f\x03\t\x01\x00\x00\x00\x0b\x05\x1di\x03\x03m\x05\x01\x1f\x19\x01\x01\x02\x02)\x01\t)\x03\x05\t)\x01\x15\x1b)\x03\t\t)\x05\x00\xff\xff\xff\xff\xff\xff\xff\xff\x11\x15)\x05\x00\xff\xff\xff\xff\xff\xff\xff\xff\x11\t)\x05\x00\xff\xff\xff\xff\xff\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\xff\x15)\x05\x00\xff\xff\xff\xff\xff\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\xff\t\t)\x05\x05\t\t)\x03\x01%\x11\x05\r\x0f\x05\x11\x13\x11\t\x07\x03\x07\x03\x05\x07\x03\x11\x07\x03\r\x0f\x05\x11\x13\x1d)\x05\t\t\t\x01\x04\x1e\x06\x05\x01\x11\x03\x1f\x07\x03\x01\r\t\x11\x03\'\x05\x03\x13\x1f\x05\r\x03\x0f\x03\r\x07\x1d\x05\x03\x03\x03\x01\r\x07\x1d\x05\x03\x03\x03\x03\x03\x03\x03W\x03\x19\x0f\x06\x03\x03\r\x03\x01\x0f\x06\x03\x03\x0f\x03\x03\x17\x07\x03Y\x05\x11\x13\x07\x05\x0b\r\x0b\x04\x03\x05\x0f\x11\t\x11\x01)\x05\x03\r\x0f\t\x07\x03\x03\x03\x07\x03\x03\x03\x13\x06G\x03\x07\x05\x01\x05\x15\x06M\x03\x03\x05\x03\x07\x0b\x04\x01\x05\t\x0b\t\x11\x03/\x05\x03O\x93\x07\x03\x03\r\x03\x0f\x03\x03\x03\x031\x03\x07\x03\x03\x03\x1b\x03\x03\x03\x03\x01\x0f\x03\x03\x03\x03\x01\x0f\x03\x03\x05\x06\x01\x03\x05\x03\x0b\x05\x06\x01\x03\x05\x03\r\x07\x07\x01\x05\x03\x0b\x05\x0f\x11\x05\x06\x01\x03\x17\x03\x13\x03\x03\x01\x0f\x03\x03\x03\x03\x01\x0f\x03\x03\x05\x06\x01\x03\x05\x03\x17\x05\x06\x01\x03\x05\x03\x19\x07\x07\x01\x05\x03\x0b\x05\x1b\x1d\x05\x06\x01\x03\x17\x03\x1f\x07\x07\x01\x05\x03#\x05\x15!\x03\x03\x01\x1b\x03\x03\x05\x06\x01\x03\x05\x03%\x05\x06\x01\x03\x05\x03\x01\x07\x07\x01\x05\x03\x0b\x05\')\x03\x03\x01\x07\x03\x03\x03\x03\x01\x07\x03\x03\x05\x06\x01\x03\x05\x03-\x05\x06\x01\x03\x05\x03/\x07\x07\x01\x05\x03\x0b\x0513\x03\x03\x01\x07\x03\x03\x03\x03\x01\x07\x03\x03\x05\x06\x01\x03\x05\x037\x05\x06\x01\x03\x05\x039\x07\x07\x01\x05\x03\x0b\x05;=\x03\x03\x01\x07\x03\x03\x03\x03\x01\x07\x03\x03\x05\x06\x01\x03\x05\x03A\x05\x06\x01\x03\x05\x03C\x07\x07\x01\x05\x03\x0b\x05EG\x11\x07\x015\x05\x11\x13\x13\x03\x05\x07\t+5?I#\x0b\x04\x03\x05KM\x06\x03\x01\x05\x01\x00\xce\x0ekA\x11\x0f\x0b\t\t\x05\x053\x03=!\x1b\x1d\x1b\x0f;11\x1f/!!)#\x1f\x19\x15\x86\x02\x13%)\x1f\x15\x1d\x15\x83\x13\r\x11\x19\x0f\x1f\x17-\x15\x11\x1f\x17\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00reshape_v1\x00concatenate_v1\x00func_v1\x00return_v1\x00get_dimension_size_v1\x00convert_v1\x00custom_call_v1\x00add_v1\x00subtract_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/reduce_window[consts=() window_dimensions=(2, b) window_strides=(1, 1) padding=((0, 0), (0, 0)) base_dilation=(1, 1) window_dilation=(1, 1)]\x00dimension\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(func)/jit(main)/add\x00jit(func)/jit(main)/sub\x00/dimension_size[dimension=0]\x00callee\x00jax.arg_info\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00generic_reduce_window_reducer\x00\x00_wrapped_jax_export_main\x00x\x00y\x00[0]\x00[1]\x00main\x00public\x00private\x00stablehlo.dynamic_reduce_window\x00',
    xla_call_module_version=6,
)  # End paste
