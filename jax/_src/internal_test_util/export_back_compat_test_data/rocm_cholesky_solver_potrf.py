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
float32 = np.float32
float64 = np.float64
complex64 = np.complex64
complex128 = np.complex128

data_2026_02_05 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_05["f32"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_potrf_ffi'],
    serialized_date=datetime.date(2026, 2, 5),
    inputs=(array([[34.812286  , -2.1484861 , 11.958679  , 12.6090975 ],
       [-2.1484861 ,  0.73191243, -2.3005815 ,  0.6536075 ],
       [11.958679  , -2.3005815 , 18.171919  , 12.823325  ],
       [12.6090975 ,  0.6536075 , 12.823325  , 33.237614  ]],
      dtype=float32),),
    expected_outputs=(array([[ 5.9001937 ,  0.        ,  0.        ,  0.        ],
       [-0.36413825,  0.77415484,  0.        ,  0.        ],
       [ 2.0268283 , -2.0183764 ,  3.160703  ,  0.        ],
       [ 2.137065  ,  1.8494937 ,  3.8677585 ,  3.2078629 ]],
      dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf32> loc("x")) -> (tensor<4x4xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc4)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f32> loc(#loc)
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc5)
    %1 = stablehlo.add %arg0, %0 : tensor<4x4xf32> loc(#loc6)
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc7)
    %3 = stablehlo.divide %1, %2 : tensor<4x4xf32> loc(#loc7)
    %4:2 = stablehlo.custom_call @hipsolver_potrf_ffi(%3) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<i32>) loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc4)
    %6 = stablehlo.compare  EQ, %4#1, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %4#0, %8 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc4)
    %11 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc8)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc6)
    %13 = stablehlo.add %11, %12 : tensor<4x4xi32> loc(#loc6)
    %14 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc8)
    %15 = stablehlo.compare  GE, %13, %14,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc10)
    %17 = stablehlo.select %15, %10, %16 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc11)
    return %17 : tensor<4x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":825:6)
#loc3 = loc("jit(cholesky)"(#loc2))
#loc4 = loc("cholesky"(#loc3))
#loc5 = loc("transpose"(#loc3))
#loc6 = loc("add"(#loc3))
#loc7 = loc("div"(#loc3))
#loc8 = loc("iota"(#loc3))
#loc9 = loc("ge"(#loc3))
#loc10 = loc("broadcast_in_dim"(#loc3))
#loc11 = loc("select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03\xdf\xa1'\x01E\x0f\x0f\x07\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x17\x0b\x0f\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03M\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x1f\x1f\x1f\x1fO\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03#\x17\x0f\x0f\x07\x17\x07\x07\x07\x13\x07\x17\x17\x13\x13\x13\x0f\x17\x02\x9a\x05\x1d\x1f\x03\x1d!#\x1f\x1d+\x03\x11\x03\x05\x1d-\x03\x1d7\x03\x03\x07\x11\x13\x15\t\x17\t\x05\x1f\x11\x01\x00\x05!\x05#\x05%\x1d\x1d\x05\x05'\x05)\x05+\x17%\xe6\x0c\r\x05-\x1d)\x03\x05/\x051\x053\x03\x071g3m5\x91\x055\x057\x059\x05;\x1d;\x03\x05=\x1d?\x03\x05?\x1dC\x03\x05A\x1f\x1d\x01\x1f\x1f!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03M\r\x01#\x1b\x03\x03S\r\x03UW\x1dC\x1dE\x1dG\x1dI\x1f\x07\t\x00\x00\x00\x00\x1f\x07\t\x00\x00\xc0\x7f\x1f\t\t\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00@\x1f\x15!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03ik\x1dK\x05\x03\r\x03oq\x1dM\x1dO\x0b\x03\x1dQ\x1dS\x03\x01\x05\x01\x03\x03G\x03\x03\x81\x15\x03\x01\x01\x01\x03\x05G\x85\x1f!\x01\x07\x01\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x01\x13\x0b\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\x93\x05\x99\x9f\x01\x01\x01\x01\x01\x13\x05\x95\x97\x11\x03\x01\x11\x03\x05\x13\x05\x9b\x9d\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x11)\x01\x11)\x01\x13\x1d)\x05\x11\x11\x13\x01\t\x1b)\x03\t\x0b\x13)\x05\x11\x11\x0f\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03\t\x17)\x03\x01\x17)\x01\x0f)\x05\x05\x05\x0f\x04.\x03\x05\x01Q\x05\x0f\x01\x07\x04\x06\x03\x03\x01\x05\x0fP\x05\x03\x07\x04\xda\x02\x031_\x03\x0b\x1b\x00\x05B\x05\x05\x03\x07\x05B\x01\x07\x03\x07\x05B\x05\t\x03\t\x05B\x05\x0b\x03\x07\x11F'\r\x03\x05\x03\x01\x07\x06\x07\x03\x05\x05\x01\x0b\x03F\x0b\x0f\x03\x05\x03\t\x13\x06\x0b\x03\x05\x05\r\x0f\x15G\x01/\x11\x05\x05\t\x03\x11\x03F\x01\x0f\x03\t\x03\x07\tF\x01\x13\x03#\x05\x15\x17\x03F\x01\x0f\x03%\x03\x19\x03F\x01\x0f\x03\x05\x03\x05\x03F\x01\x15\x03\x19\x03\x1b\x0b\x06\x01\x03\x05\x07\x1f\x13\x1d\rB\r\x17\x03\r\x03F\x07\x0f\x03\r\x03\x07\x07\x06\x07\x03\r\x05#%\rB\r\x19\x03\r\tF9\x1b\x03\x19\x05')\x03F=\x0f\x03\x05\x03\x03\x0b\x06A\x03\x05\x07+!-\x17\x04\x05\x03/\x06\x03\x01\x05\x01\x00b\x08U)\x03\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x15_\x1d\x13\x05\x1b%)9\x15\x1f\x15\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00cholesky\x00jit(cholesky)\x00/rocm-jax/jax/tests/export_back_compat_test.py\x00transpose\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00\x00hipsolver_potrf_ffi\x00\x08W\x1d\x053\x01\x0bKOQY[\x03]\x03_\x03a\x03c\x03e\x03E\x11suwy{}\x7f\x83\x05I\x87\x03\x89\x03\x8b\x03\x8d\x05I\x8f",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_05["f64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_potrf_ffi'],
    serialized_date=datetime.date(2026, 2, 5),
    inputs=(array([[ 22.512951744778434 ,  -1.5384549502899474,  -0.7079098803209338,
         -7.513422195751226 ],
       [ -1.5384549502899474,  20.359325409167592 , -11.28730444363775  ,
        -16.410921233814605 ],
       [ -0.7079098803209338, -11.28730444363775  ,   9.98412985373178  ,
         16.396557441992236 ],
       [ -7.513422195751226 , -16.410921233814605 ,  16.396557441992236 ,
         30.554293874957157 ]]),),
    expected_outputs=(array([[ 4.7447815276130925 ,  0.                 ,  0.                 ,
         0.                 ],
       [-0.3242414727288575 ,  4.500465851057001  ,  0.                 ,
         0.                 ],
       [-0.14919757131095024, -2.5187793573024955 ,  1.9020043342940942 ,
         0.                 ],
       [-1.5835127817846915 , -3.7605799733577596 ,  3.5164115306360086 ,
         1.2408341372126808 ]]),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf64> loc("x")) -> (tensor<4x4xf64> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc4)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<f64> loc(#loc)
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc5)
    %1 = stablehlo.add %arg0, %0 : tensor<4x4xf64> loc(#loc6)
    %2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc7)
    %3 = stablehlo.divide %1, %2 : tensor<4x4xf64> loc(#loc7)
    %4:2 = stablehlo.custom_call @hipsolver_potrf_ffi(%3) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<i32>) loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc4)
    %6 = stablehlo.compare  EQ, %4#1, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %4#0, %8 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc4)
    %11 = stablehlo.iota dim = 0 : tensor<4x4xi64> loc(#loc8)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4xi64> loc(#loc6)
    %13 = stablehlo.add %11, %12 : tensor<4x4xi64> loc(#loc6)
    %14 = stablehlo.iota dim = 1 : tensor<4x4xi64> loc(#loc8)
    %15 = stablehlo.compare  GE, %13, %14,  SIGNED : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc10)
    %17 = stablehlo.select %15, %10, %16 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc11)
    return %17 : tensor<4x4xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":825:6)
#loc3 = loc("jit(cholesky)"(#loc2))
#loc4 = loc("cholesky"(#loc3))
#loc5 = loc("transpose"(#loc3))
#loc6 = loc("add"(#loc3))
#loc7 = loc("div"(#loc3))
#loc8 = loc("iota"(#loc3))
#loc9 = loc("ge"(#loc3))
#loc10 = loc("broadcast_in_dim"(#loc3))
#loc11 = loc("select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03\xe3\xa3)\x01E\x0f\x0f\x07\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x17\x0b\x0f\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03O\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b///\x1f/O\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03%\x17\x0f\x07\x0f\x17\x07\x07\x0f\x13\x07\x17\x17\x07\x13\x13\x13\x0f\x17\x02\x02\x06\x1d\x1f\x03\x1d!#\x1f\x1d+\x03\x11\x03\x05\x1d-\x03\x1d7\x03\x03\x07\x11\x13\x15\t\x17\t\x05\x1f\x11\x01\x00\x05!\x05#\x05%\x1d\x1d\x05\x05'\x05)\x05+\x17%\xe6\x0c\r\x05-\x1d)\x03\x05/\x051\x053\x03\x071i3o5\x93\x055\x057\x059\x05;\x1d;\x03\x05=\x1d?\x03\x05?\x1dC\x03\x05A\x1f\x1f\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03M\r\x01#\x1b\x03\x03S\r\x03UW\x1dC\x1dE\x1dG\x1dI\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x13\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x0b\t\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x15!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03km\x1dK\x05\x03\r\x03qs\x1dM\x1dO\x0b\x03\x1dQ\x1dS\x03\x01\x05\x01\x03\x03G\x03\x03\x83\x15\x03\x01\x01\x01\x03\x05G\x87\x1f#\x01\x07\x01\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\t\x01\x13\t\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\x95\x05\x9b\xa1\x01\x01\x01\x01\x01\x13\x05\x97\x99\x11\x03\x01\x11\x03\x05\x13\x05\x9d\x9f\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x11)\x01\x11\x1d)\x01\x1d)\x05\x11\x11\t\x01\x0b)\x01\t)\x03\t\t\x13)\x05\x11\x11\x0f\x11\x03\x05\x03\x05\x1b)\x03\x01\t)\x03\t\x17)\x03\x01\x17)\x01\x0f)\x05\x05\x05\x0f\x04F\x03\x05\x01Q\x05\x0f\x01\x07\x04\x1e\x03\x03\x01\x05\x0fP\x05\x03\x07\x04\xf2\x02\x033c\x03\x0b\x1b\x00\x05B\x05\x05\x03\x07\x05B\x05\x07\x03\x13\x05B\x01\t\x03\x07\x05B\x01\x0b\x03\x0b\x05B\x05\r\x03\x07\x11F'\x0f\x03\x05\x03\x01\x07\x06\x07\x03\x05\x05\x01\r\x03F\x0b\x11\x03\x05\x03\x0b\x13\x06\x0b\x03\x05\x05\x0f\x11\x15G\x01/\x13\x05\x05\x0b\x03\x13\x03F\x01\x11\x03\x0b\x03\t\tF\x01\x15\x03%\x05\x17\x19\x03F\x01\x11\x03'\x03\x1b\x03F\x01\x11\x03\x05\x03\x07\x03F\x01\x17\x03\x19\x03\x1d\x0b\x06\x01\x03\x05\x07!\x15\x1f\rB\r\x19\x03\r\x03F\x07\x11\x03\r\x03\x05\x07\x06\x07\x03\r\x05%'\rB\r\x1b\x03\r\tF9\x1d\x03\x19\x05)+\x03F=\x11\x03\x05\x03\x03\x0b\x06A\x03\x05\x07-#/\x17\x04\x05\x031\x06\x03\x01\x05\x01\x00b\x08U)\x03\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x15_\x1d\x13\x05\x1b%)9\x15\x1f\x15\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00cholesky\x00jit(cholesky)\x00/rocm-jax/jax/tests/export_back_compat_test.py\x00transpose\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00\x00hipsolver_potrf_ffi\x00\x08[\x1f\x053\x01\x0bKOQY[\x03]\x03_\x03a\x03c\x03e\x03g\x03E\x11uwy{}\x7f\x81\x85\x05I\x89\x03\x8b\x03\x8d\x03\x8f\x05I\x91",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_05["c64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_potrf_ffi'],
    serialized_date=datetime.date(2026, 2, 5),
    inputs=(array([[ 62.221153 -6.4815424e-08j, -22.13242  -1.0246023e+01j,
          2.020702 +2.5961372e+01j, -10.631899 -4.9574718e+01j],
       [-22.13242  +1.0246025e+01j,  78.902084 +1.7114758e-07j,
          1.8615259+1.0474155e+01j,  24.668098 +1.4437879e+00j],
       [  2.020702 -2.5961372e+01j,   1.8615259-1.0474156e+01j,
         48.52617  +5.4661250e-08j, -31.667604 -9.3938026e+00j],
       [-10.631899 +4.9574718e+01j,  24.668098 -1.4437873e+00j,
        -31.667604 +9.3938026e+00j,  52.51258  +4.5532943e-08j]],
      dtype=complex64),),
    expected_outputs=(array([[ 7.8880386+0.j       ,  0.       +0.j       ,
         0.       +0.j       ,  0.       +0.j       ],
       [-2.8058202+1.2989317j,  8.327198 +0.j       ,
         0.       +0.j       ,  0.       +0.j       ],
       [ 0.2561729-3.2912328j,  0.8232526-2.3268344j,
         5.6157303+0.j       ,  0.       +0.j       ],
       [-1.3478507+6.284796j ,  1.5278549+1.7340113j,
        -1.3997552+1.2887555j,  1.4952842+0.j       ]], dtype=complex64),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f32>> loc("x")) -> (tensor<4x4xcomplex<f32>> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc4)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %cst_1 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc5)
    %1 = stablehlo.real %0 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc6)
    %2 = stablehlo.imag %0 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc6)
    %3 = stablehlo.negate %2 : tensor<4x4xf32> loc(#loc6)
    %4 = stablehlo.complex %1, %3 : tensor<4x4xcomplex<f32>> loc(#loc6)
    %5 = stablehlo.add %arg0, %4 : tensor<4x4xcomplex<f32>> loc(#loc7)
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc8)
    %7 = stablehlo.divide %5, %6 : tensor<4x4xcomplex<f32>> loc(#loc8)
    %8:2 = stablehlo.custom_call @hipsolver_potrf_ffi(%7) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<i32>) loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc4)
    %10 = stablehlo.compare  EQ, %8#1, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc4)
    %14 = stablehlo.select %13, %8#0, %12 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc4)
    %15 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc7)
    %17 = stablehlo.add %15, %16 : tensor<4x4xi32> loc(#loc7)
    %18 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc9)
    %19 = stablehlo.compare  GE, %17, %18,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc10)
    %20 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc11)
    %21 = stablehlo.select %19, %14, %20 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc12)
    return %21 : tensor<4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":825:6)
#loc3 = loc("jit(cholesky)"(#loc2))
#loc4 = loc("cholesky"(#loc3))
#loc5 = loc("transpose"(#loc3))
#loc6 = loc(""(#loc3))
#loc7 = loc("add"(#loc3))
#loc8 = loc("div"(#loc3))
#loc9 = loc("iota"(#loc3))
#loc10 = loc("ge"(#loc3))
#loc11 = loc("broadcast_in_dim"(#loc3))
#loc12 = loc("select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x013\x07\x01\x05\t!\x01\x03\x0f\x03\x1f\x13\x17\x1b\x1f#'+/37;?CGK\x03\xe7\xa5+\x01I\x0f\x0f\x07\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x17\x0b\x0f\x0b\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03M\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b//\x1f/O\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03'\x17\x0f\x0f\x07\x17\x17\x07\x0b\x07\x07\x13\x07\x17\x17\x13\x13\x13\x0f\x17\x02\xfa\x05\x1d#%\x1d!\x01\x1f\x1d-\x01\x1d/\x01\x11\x03\x05\x1d1\x01\x1d;\x01\x03\x07\x13\x15\x17\x0b\x19\x0b\x05'\x11\x01\x00\x05)\x05+\x05-\x1d\x1f\x05\x05/\x051\x053\x17'\xe6\x0c\r\x055\x1d+\x01\x057\x059\x05;\x05=\x03\x075k7q9\x95\x05?\x05A\x05C\x05E\x1d?\x01\x05G\x1dC\x01\x05I\x1dG\x01\x05K\x1f!\x01\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03Q\r\x01#\x1f\x03\x03W\r\x03Y[\x1dM\x1dO\x1dQ\x1dS\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\t\t\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00@\x00\x00\x00\x00\x1f\x19!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03mo\x1dU\x05\x03\r\x03su\x1dW\x1dY\x0b\x03\x1d9\x1d[\x03\x01\x05\x01\x03\x03K\x03\x03\x85\x15\x03\x01\x01\x01\x03\x05K\x89\x1f%\x01\x07\x01\x1f\x19!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x01\x13\x0b\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\x97\x05\x9d\xa3\x01\x01\x01\x01\x01\x13\x05\x99\x9b\x11\x03\x01\x11\x03\x05\x13\x05\x9f\xa1\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x13)\x01\x13)\x01\x17\x1d)\x05\x11\x11\x17)\x05\x11\x11\x15\x01\x03\x15\t\x1b)\x03\t\x0b\x13)\x05\x11\x11\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03\t\x1b)\x03\x01\x1b)\x01\x11)\x05\x05\x05\x11\x04\xa2\x03\x05\x01Q\x05\x11\x01\x07\x04z\x03\x03\x01\x05\x0fP\x05\x03\x07\x04N\x03\x039o\x03\x0b\x1d\x00\x05B\x05\x05\x03\x07\x05B\x03\x07\x03\x07\x05B\x05\t\x03\t\x05B\x05\x0b\x03\x07\x11F)\r\x03\x05\x03\x01\x13\x06\x07\x03\x0f\x03\x0b\x15\x06\x07\x03\x0f\x03\x0b\x17\x06\x07\x03\x0f\x03\x0f\x19\x06\x07\x03\x05\x05\r\x11\x07\x06\t\x03\x05\x05\x01\x13\x03F\r\x0f\x03\x05\x03\t\x1b\x06\r\x03\x05\x05\x15\x17\x1dG\x033\x11\x05\x05\t\x03\x19\x03F\x03\x0f\x03\t\x03\x07\tF\x03\x13\x03'\x05\x1d\x1f\x03F\x03\x0f\x03)\x03!\x03F\x03\x0f\x03\x05\x03\x05\x03F\x03\x15\x03\x1d\x03#\x0b\x06\x03\x03\x05\x07'\x1b%\rB\x0f\x17\x03\r\x03F\t\x0f\x03\r\x03\x07\x07\x06\t\x03\r\x05+-\rB\x0f\x19\x03\r\tF=\x1b\x03\x1d\x05/1\x03FA\x0f\x03\x05\x03\x03\x0b\x06E\x03\x05\x073)5\x1f\x04\x05\x037\x06\x03\x01\x05\x01\x00\x06\t])\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x03\x15_\x1d\x13\x05\x1b%)9\x15\x1f\x15\x17\x15\x11\x11\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00cholesky\x00jit(cholesky)\x00/rocm-jax/jax/tests/export_back_compat_test.py\x00transpose\x00\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00hipsolver_potrf_ffi\x00\x08W\x1d\x057\x01\x0bOSU]_\x03a\x03c\x03e\x03g\x03i\x03I\x11wy{}\x7f\x81\x83\x87\x05M\x8b\x03\x8d\x03\x8f\x03\x91\x05M\x93",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_05["c128"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_potrf_ffi'],
    serialized_date=datetime.date(2026, 2, 12),
    inputs=(array([[ 89.87952196174669   +0.j                ,
             21.172589387233007  -0.5065005061603287j,
             26.675332736174525 -21.04249820574197j  ,
            -22.454631781241247  -0.2825311262043684j],
           [ 21.172589387233007  +0.5065005061603287j,
            124.54705989631825   +0.j                ,
             19.85813796022367  -18.26763890302077j  ,
              0.8537948497861074+40.151936454090915j ],
           [ 26.675332736174525 +21.04249820574197j  ,
             19.85813796022367  +18.26763890302077j  ,
             80.50543748874053   +0.j                ,
            -27.820142802134686 -26.403643030508185j ],
           [-22.454631781241247  +0.2825311262043684j,
              0.8537948497861074-40.151936454090915j ,
            -27.820142802134686 +26.403643030508185j ,
             68.7846809105664    +0.j                ]]),),
    expected_outputs=(array([[ 9.480481103918022 +0.j                  ,
             0.                +0.j                  ,
             0.                +0.j                  ,
             0.                +0.j                  ],
           [ 2.233282167345174 +0.05342561211909446j ,
            10.934196649105319 +0.j                  ,
             0.                +0.j                  ,
             0.                +0.j                  ],
           [ 2.813710870131933 +2.219560165258458j   ,
             1.2306113307857565+1.2310972103527822j  ,
             8.03940400229041  +0.j                  ,
             0.                +0.j                  ],
           [-2.3685118439781885+0.029801349014619737j,
             0.561701801775981 -3.689802896839719j   ,
            -2.1606964535715916+3.270759732680002j   ,
             5.820421952921915 +0.j                  ]]),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f64>> loc("x")) -> (tensor<4x4xcomplex<f64>> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc4)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %cst_2 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc5)
    %1 = stablehlo.real %0 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc6)
    %2 = stablehlo.imag %0 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc6)
    %3 = stablehlo.negate %2 : tensor<4x4xf64> loc(#loc6)
    %4 = stablehlo.complex %1, %3 : tensor<4x4xcomplex<f64>> loc(#loc6)
    %5 = stablehlo.add %arg0, %4 : tensor<4x4xcomplex<f64>> loc(#loc7)
    %6 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc8)
    %7 = stablehlo.divide %5, %6 : tensor<4x4xcomplex<f64>> loc(#loc8)
    %8:2 = stablehlo.custom_call @hipsolver_potrf_ffi(%7) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xcomplex<f64>>) -> (tensor<4x4xcomplex<f64>>, tensor<i32>) loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc4)
    %10 = stablehlo.compare  EQ, %8#1, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc4)
    %14 = stablehlo.select %13, %8#0, %12 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc4)
    %15 = stablehlo.iota dim = 0 : tensor<4x4xi64> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4xi64> loc(#loc7)
    %17 = stablehlo.add %15, %16 : tensor<4x4xi64> loc(#loc7)
    %18 = stablehlo.iota dim = 1 : tensor<4x4xi64> loc(#loc9)
    %19 = stablehlo.compare  GE, %17, %18,  SIGNED : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1> loc(#loc10)
    %20 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc11)
    %21 = stablehlo.select %19, %14, %20 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc12)
    return %21 : tensor<4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":235:6)
#loc3 = loc("jit(cholesky)"(#loc2))
#loc4 = loc("cholesky"(#loc3))
#loc5 = loc("transpose"(#loc3))
#loc6 = loc(""(#loc3))
#loc7 = loc("add"(#loc3))
#loc8 = loc("div"(#loc3))
#loc9 = loc("iota"(#loc3))
#loc10 = loc("ge"(#loc3))
#loc11 = loc("broadcast_in_dim"(#loc3))
#loc12 = loc("select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x013\x07\x01\x05\t!\x01\x03\x0f\x03\x1f\x13\x17\x1b\x1f#'+/37;?CGK\x03\xeb\xa7-\x01I\x0f\x0f\x07\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x17\x0b\x0f\x0b\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03O\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0bO/O\x1fOO\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03)\x17\x0f\x07\x0f\x17\x17\x07\x0b\x07\x0f\x13\x07\x17\x17\x07\x13\x13\x13\x0f\x17\x02\x92\x06\x1d!\x03\x1d#%\x1f\x1d-\x03\x1d/\x03\x11\x03\x05\x1d1\x03\x1d;\x03\x03\x07\x13\x15\x17\x0b\x19\x0b\x05'\x11\x01\x00\x05)\x05+\x05-\x1d\x1f\x05\x05/\x051\x053\x17'\xae\x03\r\x055\x1d+\x03\x057\x059\x05;\x05=\x03\x075m7s9\x97\x05?\x05A\x05C\x05E\x1d?\x03\x05G\x1dC\x03\x05I\x1dG\x03\x05K\x1f#\x01\x1f%!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03Q\r\x01#\x1f\x03\x03W\r\x03Y[\x1dM\x1dO\x1dQ\x1dS\x1f\x07!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x17\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x0b\t\x00\x00\x00\x00\x1f\x07!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03oq\x1dU\x05\x03\r\x03uw\x1dW\x1dY\x0b\x03\x1d9\x1d[\x03\x01\x05\x01\x03\x03K\x03\x03\x87\x15\x03\x01\x01\x01\x03\x05K\x8b\x1f'\x01\x07\x01\x1f\x19!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\t\x01\x13\t\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\x99\x05\x9f\xa5\x01\x01\x01\x01\x01\x13\x05\x9b\x9d\x11\x03\x01\x11\x03\x05\x13\x05\xa1\xa3\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x13)\x01\x13\x1d)\x01!)\x05\x11\x11\t)\x05\x11\x11\x15\x01\x03\x15\x0b)\x01\t)\x03\t\t\x13)\x05\x11\x11\x11\x11\x03\x05\x03\x05\x1b)\x03\x01\t)\x03\t\x1b)\x03\x01\x1b)\x01\x11)\x05\x05\x05\x11\x04\xba\x03\x05\x01Q\x05\x11\x01\x07\x04\x92\x03\x03\x01\x05\x0fP\x05\x03\x07\x04f\x03\x03;s\x03\x0b\x1d\x00\x05B\x05\x05\x03\x07\x05B\x05\x07\x03\x17\x05B\x01\t\x03\x07\x05B\x01\x0b\x03\x0b\x05B\x05\r\x03\x07\x11F)\x0f\x03\x05\x03\x01\x13\x06\x07\x03\x0f\x03\r\x15\x06\x07\x03\x0f\x03\r\x17\x06\x07\x03\x0f\x03\x11\x19\x06\x07\x03\x05\x05\x0f\x13\x07\x06\t\x03\x05\x05\x01\x15\x03F\r\x11\x03\x05\x03\x0b\x1b\x06\r\x03\x05\x05\x17\x19\x1dG\x013\x13\x05\x05\x0b\x03\x1b\x03F\x01\x11\x03\x0b\x03\t\tF\x01\x15\x03)\x05\x1f!\x03F\x01\x11\x03+\x03#\x03F\x01\x11\x03\x05\x03\x07\x03F\x01\x17\x03\x1d\x03%\x0b\x06\x01\x03\x05\x07)\x1d'\rB\x0f\x19\x03\r\x03F\t\x11\x03\r\x03\x05\x07\x06\t\x03\r\x05-/\rB\x0f\x1b\x03\r\tF=\x1d\x03\x1d\x0513\x03FA\x11\x03\x05\x03\x03\x0b\x06E\x03\x05\x075+7\x1f\x04\x05\x039\x06\x03\x01\x05\x01\x00\x06\t])\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x03\x15_\x1d\x13\x05\x1b%)9\x15\x1f\x15\x17\x15\x11\x11\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00cholesky\x00jit(cholesky)\x00/rocm-jax/jax/tests/export_back_compat_test.py\x00transpose\x00\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00hipsolver_potrf_ffi\x00\x08[\x1f\x057\x01\x0bOSU]_\x03a\x03c\x03e\x03g\x03i\x03k\x03I\x11y{}\x7f\x81\x83\x85\x89\x05M\x8d\x03\x8f\x03\x91\x03\x93\x05M\x95",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
