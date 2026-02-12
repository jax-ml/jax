# Copyright 2026 The JAX Authors.
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

import datetime
import numpy as np
array = np.array
uint32 = np.uint32
float32 = np.float32


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_05 = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hip_threefry2x32_ffi'],
    serialized_date=datetime.date(2026, 2, 5),
    inputs=(array([0, 42], dtype=uint32),),
    expected_outputs=(array([[0.6878003   , 0.599579    , 0.2652017   , 0.24115169  ],
       [0.76292205  , 0.28484797  , 0.040389538 , 0.0032066107]],
      dtype=float32),),
    mlir_module_text=r"""
#loc = loc(unknown)
#loc1 = loc("x")
#loc2 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":808:15)
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]> loc(#loc)
  func.func public @main(%arg0: tensor<2xui32> loc("x")) -> (tensor<2x4xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(#loc)
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc)
    %0 = sdy.sharding_constraint %arg0 <@empty_mesh, [{}]> : tensor<2xui32> loc(#loc)
    %1 = call @_uniform(%0, %cst_0, %cst) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<2x4xf32> loc(#loc4)
    return %1 : tensor<2x4xf32> loc(#loc)
  } loc(#loc)
  func.func private @_uniform(%arg0: tensor<2xui32> loc(unknown), %arg1: tensor<f32> loc(unknown), %arg2: tensor<f32> loc(unknown)) -> tensor<2x4xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(#loc16)
    %c = stablehlo.constant dense<1065353216> : tensor<ui32> loc(#loc16)
    %c_0 = stablehlo.constant dense<9> : tensor<ui32> loc(#loc16)
    %0 = stablehlo.convert %arg1 : tensor<f32> loc(#loc6)
    %1 = stablehlo.convert %arg2 : tensor<f32> loc(#loc6)
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x1xf32> loc(#loc7)
    %3 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<1x1xf32> loc(#loc7)
    %4 = stablehlo.iota dim = 0 : tensor<8xui32> loc(#loc8)
    %5 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32> loc(#loc8)
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32> loc(#loc8)
    %7 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32> loc(#loc8)
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32> loc(#loc8)
    %9 = stablehlo.slice %4 [0:4] : (tensor<8xui32>) -> tensor<4xui32> loc(#loc8)
    %10 = stablehlo.slice %4 [4:8] : (tensor<8xui32>) -> tensor<4xui32> loc(#loc8)
    %11:2 = call @threefry2x32(%6, %8, %9, %10) : (tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<4xui32>, tensor<4xui32>) loc(#loc8)
    %12 = stablehlo.concatenate %11#0, %11#1, dim = 0 : (tensor<4xui32>, tensor<4xui32>) -> tensor<8xui32> loc(#loc8)
    %13 = stablehlo.reshape %12 : (tensor<8xui32>) -> tensor<2x4xui32> loc(#loc8)
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<2x4xui32> loc(#loc9)
    %15 = stablehlo.shift_right_logical %13, %14 : tensor<2x4xui32> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<2x4xui32> loc(#loc10)
    %17 = stablehlo.or %15, %16 : tensor<2x4xui32> loc(#loc10)
    %18 = stablehlo.bitcast_convert %17 : (tensor<2x4xui32>) -> tensor<2x4xf32> loc(#loc11)
    %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc12)
    %20 = stablehlo.subtract %18, %19 : tensor<2x4xf32> loc(#loc12)
    %21 = stablehlo.subtract %3, %2 : tensor<1x1xf32> loc(#loc12)
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32> loc(#loc13)
    %23 = stablehlo.multiply %20, %22 : tensor<2x4xf32> loc(#loc13)
    %24 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32> loc(#loc14)
    %25 = stablehlo.add %23, %24 : tensor<2x4xf32> loc(#loc14)
    %26 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32> loc(#loc15)
    %27 = stablehlo.maximum %26, %25 : tensor<2x4xf32> loc(#loc15)
    return %27 : tensor<2x4xf32> loc(#loc16)
  } loc(#loc16)
  func.func private @threefry2x32(%arg0: tensor<ui32> loc("/rocm-jax/jax/tests/export_back_compat_test.py":808:15), %arg1: tensor<ui32> loc("/rocm-jax/jax/tests/export_back_compat_test.py":808:15), %arg2: tensor<4xui32> loc("/rocm-jax/jax/tests/export_back_compat_test.py":808:15), %arg3: tensor<4xui32> loc("/rocm-jax/jax/tests/export_back_compat_test.py":808:15)) -> (tensor<4xui32>, tensor<4xui32>) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<4xui32> loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<4xui32> loc(#loc3)
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<4xui32>) -> tensor<4xui32> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<4xui32>) -> tensor<4xui32> loc(#loc3)
    %4:2 = stablehlo.custom_call @hip_threefry2x32_ffi(%0, %1, %2, %3) {mhlo.backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<4xui32>, tensor<4xui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<4xui32>, tensor<4xui32>) loc(#loc3)
    return %4#0, %4#1 : tensor<4xui32>, tensor<4xui32> loc(#loc2)
  } loc(#loc2)
} loc(#loc)
#loc3 = loc("threefry2x32")
#loc4 = loc("jit(func)/jit(_uniform)"(#loc2))
#loc5 = loc("jit(func)/jit"(#loc2))
#loc6 = loc("convert_element_type"(#loc2))
#loc7 = loc("broadcast_in_dim"(#loc2))
#loc8 = loc(""(#loc2))
#loc9 = loc("shift_right_logical"(#loc2))
#loc10 = loc("or"(#loc2))
#loc11 = loc("bitcast_convert_type"(#loc2))
#loc12 = loc("sub"(#loc2))
#loc13 = loc("mul"(#loc2))
#loc14 = loc("add"(#loc2))
#loc15 = loc("max"(#loc2))
#loc16 = loc("jit:"(#loc5))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01C\x07\x01\x05\t-\x01\x05\x0f\x13\x03\x05\x17\x1b\x05%\x1f#'+/37;?CGKOSW[_c\x03\xed\xa51\x01Y\x17\x07\x0f\x0f\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03\x07\x0f\x17\x13\x05G\x0f//\x0b\x0b/O\x0f\x0b\x0b\x0b\x1f\x0f/\x0b\x0f\x13\x0b\x0b\x0b\x0b\x17\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x1f\x1f//\x1f\x01\t\x13\x0b\x0f\x0f\x05)\x13\x17\x0f\x0f\x07\x13\x13\x17\x07\x07\x17\x13\x13\x17\x1f'\x13\x13\x07\x13\x02&\x06\x175\xa2\x0c\x1f\x1f\x1dA\x01\x1d/1\x1d7\x03\x1dK\x01\x11\x05\x05\x053\x1d=\x01\x1d?\x01\x1dC\x01\x1dE\x01\x1dM\x01\x1dO\x01\x1dQ\x01\x1dS\x03\x1dW\x01\x03\x07%')\r+\r\x055\x11\x03\x00\x057\x059\x05;\x05=\x1d3\x01\x05?\x05A\x05C\x03\x03;e\x05E\x05G\x05I\x05K\x05M\x05O\x1dI\x01\x05Q\x05S\x05U\x05W\x05Y\x05[\t\x0f\x05]\x05\x01\x01\rU\x03]\x01\x0b\x01\x01\x01\x1f)\x01\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x15\x11\x01\x00\x00\x00\x00\x00\x00\x00\r\x01\x03\x01\x1f\x15\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03e\x1d_\x1da\x1dC\x1f\r\t\x00\x00\x80?\x13\x19\x01\x1f\x15\x11\x04\x00\x00\x00\x00\x00\x00\x00##\x03\x03\x7f\r\x03\x81\x83\x1dc\x1de\x1dg\x1di\x03\x07eee#%#'\x0b\x03\x1dK\x1dk\x05\x01\x03\taaaa\x03\x05aa\x1f\x0f\t\x00\x00\x80?\x1f\x0f\t\t\x00\x00\x00\x1f\x15\x11\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x15\x11\x08\x00\x00\x00\x00\x00\x00\x00\x1f\r\t\x00\x00\x00\x00\x1b\x03\t\x07\x01\t\x01\x02\x02\x01\n\x02)\x03\x11\x11)\x05\t\x11\x1b)\x01\x1b)\x01\x11%)\x03\t\x11)\x03\x05\x19)\x05\t\x11\x11\x1d\t)\x05\x05\x05\x1b)\x03!\x11)\x03\x05\x11\x11\x03\x13\x03\x0b\x11\x07\x13\r\r\x03\x0b\x11\t\x0f\x0f\t\t\x05\t\t)\x03\x01\x19)\x03\x05-\x13)\x03\t\x19\x04n\x06\x05\x03Q\x03#\x01\x07\x04F\x06\x03\x01\x11\x05@\x03\x03\x0fP\x03\x05\x07\x04q\x03\x0f\x1f\x03'\x1f\x00\x01\x06\x1f\x03\x01\x03\x01\x0bB\x03\x07\x03\r\x0bB\x03\t\x03\r\x07F\x03\x0b\x03\x01\x03\x03\x01\x06!\x03\x13\x03\t\x17F!\r\x03\x0b\x07\x0b\x07\x05\x11\x04\x03\x03\r\x0fP\x07\x0f\x07\x04\xf2\x03\x03G\x83\x07%\x19\x19\x00\x0bB\x07\x07\x03\r\x0bB\x07\x11\x03\x0f\x0bB\x07\x13\x03\x0f\x15\x06\x11\x03\r\x03\x03\x15\x06\x11\x03\r\x03\x05\tF\x13\x15\x03\x1d\x03\r\tF\x13\x15\x03\x1d\x03\x0f\x1dB\x05\x17\x03\x1f\rF\x05\x19\x03!\x03\x01\x13\x06\x05\x03\x0f\x03\x17\rF\x05\x1b\x03!\x03\x01\x13\x06\x05\x03\x0f\x03\x1b\rF\x05\x1d\x03\t\x03\x15\rF\x05\x1f\x03\t\x03\x15\x17F\x05!\x05\t\t\t\x19\x1d\x1f!\x1fF\x05\x17\x03\x1f\x05#%\x13\x06\x05\x03\x17\x03'\tF\x15\x15\x03\x17\x03\x0b!\x06\x15\x03\x17\x05)+\tF\x17\x15\x03\x17\x03\t#\x06\x17\x03\x17\x05-/%\x06G\x03\x0b\x031\tF\x0b\x15\x03\x0b\x03\x07\x19\x06\x0b\x03\x0b\x0535\x19\x06\x0b\x03\x1d\x05\x13\x11\tF\x19#\x03\x0b\x039'\x06\x19\x03\x0b\x057;\tF\x1b#\x03\x0b\x03\x11)\x06\x1b\x03\x0b\x05=?\tF\x1d#\x03\x0b\x03\x11+\x06\x1d\x03\x0b\x05CA\x11\x04\x07\x03E\x0fP\x01%\x07\x04\x81\x03\x15\x1b\t\x1f\x01\x1f\x01\x13\x01\x13\x01\x00\tF\t\x15\x03\t\x03\x01\tF\t\x15\x03\t\x03\x03\tF\t'\x03\t\x03\x05\tF\t'\x03\t\x03\x07\x1bG\t9)\x05\t\t\t\t\x0b\r\x0f\x11\x04\x01\x05\x11\x13\x06\x03\x01\x05\x01\x00n\x0bm+\x0f\x0b\x0f!\x11\x131\x05\t\t\t\t+\x07)\x03#+)\x1b_\x1d\x0b\x13%)9\x17\x17\x0f\x19'\r/\x1f\x11\x1f\x19\x11\x17\x17\x15\x11\x13\x19))\x0b\x0f7\x0b\t\x11builtin\x00sdy\x00vhlo\x00unrealized_conversion_cast\x00module\x00mesh\x00sharding_constraint\x00broadcast_in_dim_v1\x00constant_v1\x00slice_v1\x00func_v1\x00return_v1\x00reshape_v1\x00convert_v1\x00call_v1\x00subtract_v1\x00custom_call_v1\x00iota_v1\x00concatenate_v1\x00shift_right_logical_v1\x00or_v1\x00bitcast_convert_v1\x00multiply_v1\x00add_v1\x00maximum_v1\x00empty_mesh\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit:\x00jit(func)/jit\x00/rocm-jax/jax/tests/export_back_compat_test.py\x00threefry2x32\x00mhlo.backend_config\x00convert_element_type\x00broadcast_in_dim\x00\x00shift_right_logical\x00or\x00bitcast_convert_type\x00sub\x00mul\x00add\x00max\x00x\x00jit(func)/jit(_uniform)\x00_uniform\x00private\x00jax.result_info\x00result\x00main\x00public\x00hip_threefry2x32_ffi\x00\x08\x91+\x05[\x01\x05Y\x0f\x0bm{}\x85\x87\x03u\x03\xa3\x03[\x03o\x0b\x89\x8bmoq\x03\x9b\x03\x9d\x03_\x03w\x07cic\x07\x9fcc\x07yic\x07\xa1yc\x03s\x03k\x0bg\x8dgsq\x03i\x11\x8f\x91\x93g\x95\x97g\x99",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
