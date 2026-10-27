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

# ruff: noqa

import datetime
import numpy as np

array = np.array
uint32 = np.uint32
float32 = np.float32

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13 = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapi_threefry2x32_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([42, 43], dtype=uint32),),
    expected_outputs=(array([[0.42591238  , 0.076994896 , 0.44370103  , 0.72904015  ],
       [0.17879379  , 0.81439507  , 0.0019190311, 0.68608475  ]],
      dtype=float32),),
    mlir_module_text=r"""
#loc = loc(unknown)
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2xui32> loc("x")) -> (tensor<2x4xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc20)
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64> loc(#loc20)
    %0 = call @_uniform(%arg0, %cst, %cst_0) : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<2x4xf32> loc(#loc21)
    return %0 : tensor<2x4xf32> loc(#loc20)
  } loc(#loc)
  func.func private @_uniform(%arg0: tensor<2xui32> loc(unknown), %arg1: tensor<f64> loc(unknown), %arg2: tensor<f64> loc(unknown)) -> tensor<2x4xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<1065353216> : tensor<ui32> loc(#loc)
    %c_0 = stablehlo.constant dense<9> : tensor<ui32> loc(#loc)
    %0 = stablehlo.convert %arg1 : (tensor<f64>) -> tensor<f32> loc(#loc4)
    %1 = stablehlo.convert %arg2 : (tensor<f64>) -> tensor<f32> loc(#loc4)
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x1xf32> loc(#loc5)
    %3 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<1x1xf32> loc(#loc5)
    %4 = stablehlo.iota dim = 0 : tensor<8xui32> loc(#loc6)
    %5 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32> loc(#loc7)
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32> loc(#loc7)
    %7 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32> loc(#loc7)
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32> loc(#loc7)
    %9 = stablehlo.slice %4 [0:4] : (tensor<8xui32>) -> tensor<4xui32> loc(#loc8)
    %10 = stablehlo.slice %4 [4:8] : (tensor<8xui32>) -> tensor<4xui32> loc(#loc8)
    %11:2 = call @threefry2x32(%6, %8, %9, %10) : (tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<4xui32>, tensor<4xui32>) loc(#loc9)
    %12 = stablehlo.concatenate %11#0, %11#1, dim = 0 : (tensor<4xui32>, tensor<4xui32>) -> tensor<8xui32> loc(#loc10)
    %13 = stablehlo.reshape %12 : (tensor<8xui32>) -> tensor<2x4xui32> loc(#loc11)
    %14 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<2x4xui32> loc(#loc12)
    %15 = stablehlo.shift_right_logical %13, %14 : tensor<2x4xui32> loc(#loc12)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<2x4xui32> loc(#loc13)
    %17 = stablehlo.or %15, %16 : tensor<2x4xui32> loc(#loc13)
    %18 = stablehlo.bitcast_convert %17 : (tensor<2x4xui32>) -> tensor<2x4xf32> loc(#loc14)
    %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc15)
    %20 = stablehlo.subtract %18, %19 : tensor<2x4xf32> loc(#loc15)
    %21 = stablehlo.subtract %3, %2 : tensor<1x1xf32> loc(#loc15)
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32> loc(#loc16)
    %23 = stablehlo.multiply %20, %22 : tensor<2x4xf32> loc(#loc16)
    %24 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32> loc(#loc17)
    %25 = stablehlo.add %23, %24 : tensor<2x4xf32> loc(#loc17)
    %26 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32> loc(#loc18)
    %27 = stablehlo.maximum %26, %25 : tensor<2x4xf32> loc(#loc18)
    return %27 : tensor<2x4xf32> loc(#loc)
  } loc(#loc24)
  func.func private @threefry2x32(%arg0: tensor<ui32> loc(unknown), %arg1: tensor<ui32> loc(unknown), %arg2: tensor<4xui32> loc(unknown), %arg3: tensor<4xui32> loc(unknown)) -> (tensor<4xui32>, tensor<4xui32>) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<4xui32> loc(#loc23)
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<4xui32> loc(#loc23)
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<4xui32>) -> tensor<4xui32> loc(#loc23)
    %3 = stablehlo.broadcast_in_dim %arg3, dims = [0] : (tensor<4xui32>) -> tensor<4xui32> loc(#loc23)
    %4:2 = stablehlo.custom_call @oneapi_threefry2x32_ffi(%0, %1, %2, %3) {mhlo.backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<4xui32>, tensor<4xui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<4xui32>, tensor<4xui32>) loc(#loc23)
    return %4#0, %4#1 : tensor<4xui32>, tensor<4xui32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":963:6)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":961:15)
#loc4 = loc("convert_element_type")
#loc5 = loc("broadcast_in_dim")
#loc6 = loc("iota")
#loc7 = loc("unstack")
#loc8 = loc("split")
#loc9 = loc("")
#loc10 = loc("concatenate")
#loc11 = loc("reshape")
#loc12 = loc("shift_right_logical")
#loc13 = loc("or")
#loc14 = loc("bitcast_convert_type")
#loc15 = loc("sub")
#loc16 = loc("mul")
#loc17 = loc("add")
#loc18 = loc("max")
#loc19 = loc("threefry2x32")
#loc20 = loc("jit(func)"(#loc2))
#loc21 = loc("jit(func)/jit(_uniform)"(#loc3))
#loc22 = loc("jit(func)/jit"(#loc3))
#loc23 = loc("threefry2x32:"(#loc19))
#loc24 = loc("jit:"(#loc22))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x017\x05\x01\x05'\x01\x03\x0b\x03%\x0f\x13\x17\x1b\x1f#'+/37;?CGKOS\x03\xff\xbb1\x01s\x07\x0f\x0f\x0f\x0f\x0f\x17\x0b\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x17\x0f\x0b\x03I\x0f//\x0b\x0b/O\x0f\x0b\x0b\x0b\x0f/\x0b\x0f\x13\x0b\x0b\x0b\x0b\x17\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x1f\x1f\x1f////\x01\x05\x0b\x0f\x03-\x13\x17\x0f\x0f\x07\x13\x17\x13\x07\x0f\x07\x17\x13\x13\x17\x1f\x07'\x13\x13\x07\x13\x02\xb6\x06\x1f\x1d57\x1dG\x01\x1d_\x01\x1dkm\x11\x03\x05\x17\x0f\x06\x0f\x1f\x05+\x1d?\x01\x1dA\x01\x1dI\x01\x1dW\x01\x1dY\x01\x1da\x01\x1dc\x01\x1de\x01\x03\x07#%'\x0b)\x0b\x05-\x11\x01\x00\x05/\x051\x053\x1d/1\x055\x1d3\r\x057\x059\x1d9\x01\x05;\x03\x03=y\x05=\x05?\x05A\x1dE\x01\x05C\x05E\x05G\x1dM\x01\x05I\x1dQ\x01\x05K\x1dU\x01\x05M\x05O\x05Q\x1d]\x01\x05S\x05U\x05W\x05Y\x05[\x1di\x01\x05]\x05_\x17\x0f\x0e\x0f\r\x1dq\r\x05a\x1f)\x01\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x0f\x11\x01\x00\x00\x00\x00\x00\x00\x00\r\x01\x03\x01\x1f\x0f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03y\x1dc\x1de\x1d;\x13\x15\x01\x1f\x0f\x11\x04\x00\x00\x00\x00\x00\x00\x00#!\x03\x03\x91\r\x03\x93\x95\x1dg\x1di\x1dk\x1dm\x03\x07yyy###'\x0b\x03\x1dI\x1do\x05\x01\x03\tuuuu\x03\x05uu\x1f\x17\t\x00\x00\x80?\x1f\t\t\x00\x00\x80?\x1f\t\t\t\x00\x00\x00\x1f\x0f\x11\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x0f\x11\x08\x00\x00\x00\x00\x00\x00\x00\x1f\x0b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x0b\x11\x00\x00\x00\x00\x00\x00\xf0?\x01\t\x01\x02\x02)\x03\x11\r)\x05\t\x11\x19)\x01\r)\x01%%)\x03\x05\x15)\x05\t\x11\r)\x03\t\r\x1d)\x01\x19\t)\x05\x05\x05\x19)\x03!\r)\x03\x05\r\x11\x03\x13\x03\x07\x11\x07\x13\x0b\x0b\x03\x07\x0b\x11\t\t\t\x05\x05\x05\x05\x05)\x03\x01\x15)\x03\x05-\x13)\x03\t\x15\x04\xf6\x05\x05\x01Q\x01!\x01\x07\x04\xce\x05\x03\x01\r\tP\x01\x03\x07\x04E\x03\t\x13\x03'g\x00\x05B\t\x05\x03\x0b\x05B\t\x07\x03\x0b\x11Fo\t\x03\x07\x07\x01\x03\x05\x0b\x04\t\x03\x07\tP-\x0b\x07\x04\xf2\x03\x03G\x83\x07%\x15\x15\x00\x05B\x01\r\x03\x17\x05B\x01\x0f\x03\t\x05B\x01\x11\x03\t\x0f\x06\x11\x03\x17\x03\x03\x0f\x06\x11\x03\x17\x03\x05\x03F\x13\x13\x03\x1b\x03\r\x03F\x13\x13\x03\x1b\x03\x0f\x17BC\x15\x03\x1d\x07F\x05\x17\x03\x1f\x03\x01\r\x06\x05\x03\t\x03\x17\x07F\x05\x19\x03\x1f\x03\x01\r\x06\x05\x03\t\x03\x1b\x07F\x15\x1b\x03\x05\x03\x15\x07F\x15\x1d\x03\x05\x03\x15\x11FK\x1f\x05\x05\x05\t\x19\x1d\x1f!\x19FO\x15\x03\x1d\x05#%\r\x06S\x03\x11\x03'\x03F\x17\x13\x03\x11\x03\x0b\x1b\x06\x17\x03\x11\x05)+\x03F\x19\x13\x03\x11\x03\t\x1d\x06\x19\x03\x11\x05-/\x1f\x06[\x03\x07\x031\x03F\x07\x13\x03\x07\x03\x07\x13\x06\x07\x03\x07\x0535\x13\x06\x07\x03\x1b\x05\x13\x11\x03F\x1b!\x03\x07\x039!\x06\x1b\x03\x07\x057;\x03F\x1d!\x03\x07\x03\x11#\x06\x1d\x03\x07\x05=?\x03F\x1f!\x03\x07\x03\x11%\x06\x1f\x03\x07\x05CA\x0b\x04\x01\x03E\tP\x01#\x07\x04y\x03\x15\x1b\t\x11\x11\t\t\x00\x03F\x03\x13\x03\x05\x03\x01\x03F\x03\x13\x03\x05\x03\x03\x03F\x03%\x03\x05\x03\x05\x03F\x03%\x03\x05\x03\x07\x15G\x03;'\x05\x05\x05\t\t\x0b\r\x0f\x0b\x04\x01\x05\x11\x13\x06\x03\x01\x05\x01\x00\xa2\x0bq1\x0f\x0b\x0f!\x11\x131\x15\x05\t\t\t\t+\x07)\x11\x19\x03\r\x11\x0b#+)\x1b\x1d\x1d\x0b\x13%)9w\x17\x0f\x19'\r/\x1f\x11\x1f\x19\x11\x17\x17\x15\x11\x13\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00slice_v1\x00func_v1\x00return_v1\x00reshape_v1\x00convert_v1\x00call_v1\x00subtract_v1\x00custom_call_v1\x00iota_v1\x00concatenate_v1\x00shift_right_logical_v1\x00or_v1\x00bitcast_convert_v1\x00multiply_v1\x00add_v1\x00maximum_v1\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit:\x00jit(func)/jit\x00threefry2x32:\x00threefry2x32\x00mhlo.backend_config\x00convert_element_type\x00broadcast_in_dim\x00iota\x00unstack\x00split\x00\x00concatenate\x00reshape\x00shift_right_logical\x00or\x00bitcast_convert_type\x00sub\x00mul\x00add\x00max\x00x\x00jit(func)\x00jit(func)/jit(_uniform)\x00_uniform\x00private\x00jax.result_info\x00result\x00main\x00public\x00oneapi_threefry2x32_ffi\x00\x08\x8b)\x05W\x01\x0b\x81\x8d\x8f\x97\x99\x03\xb7\x03\xb9\x03\x83\x0b\x9b\x9d\x81\x83\x85\x03\xad\x03\xaf\x03\xb1\x03s\x03\x89\x07w}w\x07\xb3ww\x07\x8b}w\x07\xb5\x8bw\x03\x87\x03\x7f\x0b{\x9f{\x87\x85\x03}\x11\xa1\xa3\xa5{\xa7\xa9{\xab",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
