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
complex64 = np.complex64

data_2026_08_13 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["f32"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_potrf_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[ 10.895977 ,   4.2912526, -18.314928 ,   3.6974142],
       [  4.2912526,  61.31485  ,   4.850662 ,  -2.182202 ],
       [-18.314928 ,   4.850662 ,  36.91168  ,  -4.3174276],
       [  3.6974142,  -2.182202 ,  -4.3174276,  16.82287  ]],
      dtype=float32),),
    expected_outputs=(array([[ 3.3009055 ,  0.        ,  0.        ,  0.        ],
       [ 1.300023  ,  7.7217093 ,  0.        ,  0.        ],
       [-5.548456  ,  1.5623202 ,  1.9197571 ,  0.        ],
       [ 1.1201212 , -0.47118914,  1.3718737 ,  3.6693516 ]],
      dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf32> loc("x")) -> (tensor<4x4xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc3)
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc3)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc4)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<f32> loc(#loc3)
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc5)
    %1 = stablehlo.add %arg0, %0 : tensor<4x4xf32> loc(#loc6)
    %2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc7)
    %3 = stablehlo.divide %1, %2 : tensor<4x4xf32> loc(#loc7)
    %4:2 = stablehlo.custom_call @oneapisolver_potrf_ffi(%3) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<i32>) loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc4)
    %6 = stablehlo.compare EQ, %4#1, %5, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %4#0, %8 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc4)
    %11 = stablehlo.iota dim = 0 : tensor<4x4xi64> loc(#loc8)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4xi64> loc(#loc6)
    %13 = stablehlo.add %11, %12 : tensor<4x4xi64> loc(#loc6)
    %14 = stablehlo.iota dim = 1 : tensor<4x4xi64> loc(#loc8)
    %15 = stablehlo.compare GE, %13, %14, SIGNED : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc10)
    %17 = stablehlo.select %15, %10, %16 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc11)
    return %17 : tensor<4x4xf32> loc(#loc3)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":269:4)
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
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03\xe3\xa3)\x01E\x0f\x0f\x07\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x17\x0b\x0b\x0f\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03O\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x1f/\x1f\x1f\x1fO\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03%\x17\x0f\x07\x0f\x17\x07\x07\x0f\x13\x07\x17\x17\x07\x13\x13\x13\x0f\x17\x02\xd2\x05\x1d\x1f!\x1d%\x01\x1f\x1d+\x01\x11\x03\x05\x1d-\x01\x1d7\x01\x03\x07\x11\x13\x15\t\x17\t\x05\x1f\x11\x01\x00\x05!\x05#\x05%\x1d\x1d\x05\x05'\x05)\x17#6\x04\t\x05+\x05-\x1d)\x01\x05/\x051\x053\x03\x071i3o5\x93\x055\x057\x059\x05;\x1d;\x01\x05=\x1d?\x01\x05?\x1dC\x01\x05A\x1f\x1f\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03M\r\x01#\x1b\x03\x03S\r\x03UW\x1dC\x1dE\x1dG\x1dI\x1f\x07\t\x00\x00\x00\x00\x1f\x13\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\xc0\x7f\x1f\x0b\t\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00@\x1f\x15!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03km\x1dK\x05\x03\r\x03qs\x1dM\x1dO\x0b\x03\x1dQ\x1dS\x03\x01\x05\x01\x03\x03G\x03\x03\x83\x15\x03\x01\x01\x01\x03\x05G\x87\x1f#\x01\x07\x01\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\t\x01\x13\t\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\x95\x05\x9b\xa1\x01\x01\x01\x01\x01\x13\x05\x97\x99\x11\x03\x01\x11\x03\x05\x13\x05\x9d\x9f\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x11)\x01\x11\x1d)\x01\x1d)\x05\x11\x11\t\x01\t)\x01\t)\x03\t\t\x13)\x05\x11\x11\x0f\x11\x03\x05\x03\x05\x1b)\x03\x01\t)\x03\t\x17)\x03\x01\x17)\x01\x0f)\x05\x05\x05\x0f\x04F\x03\x05\x01Q\x05\x0f\x01\x07\x04\x1e\x03\x03\x01\x05\x0fP\x05\x03\x07\x04\xf2\x02\x033c\x03\x0b\x1b\x00\x05B\x01\x05\x03\x07\x05B\x01\x07\x03\x13\x05B\x03\t\x03\x07\x05B\x03\x0b\x03\x0b\x05B\x01\r\x03\x07\x11F'\x0f\x03\x05\x03\x01\x07\x06\x07\x03\x05\x05\x01\r\x03F\x0b\x11\x03\x05\x03\x0b\x13\x06\x0b\x03\x05\x05\x0f\x11\x15G\x03/\x13\x05\x05\x0b\x03\x13\x03F\x03\x11\x03\x0b\x03\t\tF\x03\x15\x03%\x05\x17\x19\x03F\x03\x11\x03'\x03\x1b\x03F\x03\x11\x03\x05\x03\x07\x03F\x03\x17\x03\x19\x03\x1d\x0b\x06\x03\x03\x05\x07!\x15\x1f\rB\r\x19\x03\r\x03F\x07\x11\x03\r\x03\x05\x07\x06\x07\x03\r\x05%'\rB\r\x1b\x03\r\tF9\x1d\x03\x19\x05)+\x03F=\x11\x03\x05\x03\x03\x0b\x06A\x03\x05\x07-#/\x17\x04\x01\x031\x06\x03\x01\x05\x01\x00\x9e\x08U/\x03\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x15\x13w\x1d\x05\x1b%)9\x15\x1f\x15\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00jit(cholesky)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00cholesky\x00transpose\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00\x00oneapisolver_potrf_ffi\x00\x08[\x1f\x053\x01\x0bKOQY[\x03]\x03_\x03a\x03c\x03e\x03g\x03E\x11uwy{}\x7f\x81\x85\x05I\x89\x03\x8b\x03\x8d\x03\x8f\x05I\x91",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["f64"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_potrf_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[ 19.64203917602577  ,  -3.2571292619584797,  -6.113417498905007 ,
        -38.31045157187507  ],
       [ -3.2571292619584797,  30.34754344041282  ,  -0.2111001780310375,
         12.603145919345357 ],
       [ -6.113417498905007 ,  -0.2111001780310375,   8.567277222442657 ,
         15.150956096041329 ],
       [-38.31045157187507  ,  12.603145919345357 ,  15.150956096041329 ,
         77.9454889863391   ]]),),
    expected_outputs=(array([[ 4.431934022074986  ,  0.                 ,  0.                 ,
         0.                 ],
       [-0.7349227776711181 ,  5.4596182972139164 ,  0.                 ,
         0.                 ],
       [-1.3794017393884324 , -0.22434790660948   ,  2.5718079400714915 ,
         0.                 ],
       [-8.644183641059374  ,  1.1448306689037868 ,  1.3546868938685372 ,
         0.27886255592804005]]),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf64> loc("x")) -> (tensor<4x4xf64> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc3)
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc3)
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc4)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<f64> loc(#loc3)
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc5)
    %1 = stablehlo.add %arg0, %0 : tensor<4x4xf64> loc(#loc6)
    %2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc7)
    %3 = stablehlo.divide %1, %2 : tensor<4x4xf64> loc(#loc7)
    %4:2 = stablehlo.custom_call @oneapisolver_potrf_ffi(%3) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<i32>) loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc4)
    %6 = stablehlo.compare EQ, %4#1, %5, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %4#0, %8 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc4)
    %11 = stablehlo.iota dim = 0 : tensor<4x4xi64> loc(#loc8)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4xi64> loc(#loc6)
    %13 = stablehlo.add %11, %12 : tensor<4x4xi64> loc(#loc6)
    %14 = stablehlo.iota dim = 1 : tensor<4x4xi64> loc(#loc8)
    %15 = stablehlo.compare GE, %13, %14, SIGNED : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc10)
    %17 = stablehlo.select %15, %10, %16 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc11)
    return %17 : tensor<4x4xf64> loc(#loc3)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":269:4)
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
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03\xe3\xa3)\x01E\x0f\x0f\x07\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x17\x0b\x0b\x0f\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03O\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b///\x1f/O\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03%\x17\x0f\x07\x0f\x17\x07\x07\x0f\x13\x07\x17\x17\x07\x13\x13\x13\x0f\x17\x02\x02\x06\x1d\x1f!\x1d%\x01\x1f\x1d+\x01\x11\x03\x05\x1d-\x01\x1d7\x01\x03\x07\x11\x13\x15\t\x17\t\x05\x1f\x11\x01\x00\x05!\x05#\x05%\x1d\x1d\x05\x05'\x05)\x17#6\x04\t\x05+\x05-\x1d)\x01\x05/\x051\x053\x03\x071i3o5\x93\x055\x057\x059\x05;\x1d;\x01\x05=\x1d?\x01\x05?\x1dC\x01\x05A\x1f\x1f\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03M\r\x01#\x1b\x03\x03S\r\x03UW\x1dC\x1dE\x1dG\x1dI\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x13\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x0b\t\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x15!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03km\x1dK\x05\x03\r\x03qs\x1dM\x1dO\x0b\x03\x1dQ\x1dS\x03\x01\x05\x01\x03\x03G\x03\x03\x83\x15\x03\x01\x01\x01\x03\x05G\x87\x1f#\x01\x07\x01\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\t\x01\x13\t\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\x95\x05\x9b\xa1\x01\x01\x01\x01\x01\x13\x05\x97\x99\x11\x03\x01\x11\x03\x05\x13\x05\x9d\x9f\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x11)\x01\x11\x1d)\x01\x1d)\x05\x11\x11\t\x01\x0b)\x01\t)\x03\t\t\x13)\x05\x11\x11\x0f\x11\x03\x05\x03\x05\x1b)\x03\x01\t)\x03\t\x17)\x03\x01\x17)\x01\x0f)\x05\x05\x05\x0f\x04F\x03\x05\x01Q\x05\x0f\x01\x07\x04\x1e\x03\x03\x01\x05\x0fP\x05\x03\x07\x04\xf2\x02\x033c\x03\x0b\x1b\x00\x05B\x01\x05\x03\x07\x05B\x01\x07\x03\x13\x05B\x03\t\x03\x07\x05B\x03\x0b\x03\x0b\x05B\x01\r\x03\x07\x11F'\x0f\x03\x05\x03\x01\x07\x06\x07\x03\x05\x05\x01\r\x03F\x0b\x11\x03\x05\x03\x0b\x13\x06\x0b\x03\x05\x05\x0f\x11\x15G\x03/\x13\x05\x05\x0b\x03\x13\x03F\x03\x11\x03\x0b\x03\t\tF\x03\x15\x03%\x05\x17\x19\x03F\x03\x11\x03'\x03\x1b\x03F\x03\x11\x03\x05\x03\x07\x03F\x03\x17\x03\x19\x03\x1d\x0b\x06\x03\x03\x05\x07!\x15\x1f\rB\r\x19\x03\r\x03F\x07\x11\x03\r\x03\x05\x07\x06\x07\x03\r\x05%'\rB\r\x1b\x03\r\tF9\x1d\x03\x19\x05)+\x03F=\x11\x03\x05\x03\x03\x0b\x06A\x03\x05\x07-#/\x17\x04\x01\x031\x06\x03\x01\x05\x01\x00\x9e\x08U/\x03\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x15\x13w\x1d\x05\x1b%)9\x15\x1f\x15\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00jit(cholesky)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00cholesky\x00transpose\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00\x00oneapisolver_potrf_ffi\x00\x08[\x1f\x053\x01\x0bKOQY[\x03]\x03_\x03a\x03c\x03e\x03g\x03E\x11uwy{}\x7f\x81\x85\x05I\x89\x03\x8b\x03\x8d\x03\x8f\x05I\x91",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["c64"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_potrf_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[123.867004 -4.3383392e-07j,  87.94228  +8.6708698e+00j,
        -28.046402 +7.5865143e+01j, -44.464363 -2.5990704e+01j],
       [ 87.94228  -8.6708679e+00j, 122.2006   +2.2517929e-07j,
         -4.3756304+5.4971073e+01j, -21.559742 +1.9333513e+00j],
       [-28.046402 -7.5865135e+01j,  -4.3756304-5.4971073e+01j,
        101.79984  -1.2922367e-06j,  -3.774759 +2.8983820e+01j],
       [-44.464363 +2.5990702e+01j, -21.559742 -1.9333509e+00j,
         -3.774759 -2.8983820e+01j,  83.28284  +3.7324449e-07j]],
      dtype=complex64),),
    expected_outputs=(array([[11.129556   +0.j         ,  0.         +0.j         ,
         0.         +0.j         ,  0.         +0.j         ],
       [ 7.901688   -0.77908486j ,  7.6913557  +0.j         ,
         0.         +0.j         ,  0.         +0.j         ],
       [-2.5199928  -6.816547j   ,  1.3295313  +0.111091286j,
         6.870529   +0.j         ,  0.         +0.j         ],
       [-3.9951606  +2.3352866j  ,  1.5378505  -2.2458315j  ,
         0.040888723+1.0612032j  ,  7.302835   +0.j         ]],
      dtype=complex64),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f32>> loc("x")) -> (tensor<4x4xcomplex<f32>> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc3)
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc3)
    %cst_0 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc4)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %cst_2 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc3)
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc5)
    %1 = stablehlo.real %0 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc6)
    %2 = stablehlo.imag %0 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc7)
    %3 = stablehlo.negate %2 : tensor<4x4xf32> loc(#loc8)
    %4 = stablehlo.complex %1, %3 : tensor<4x4xcomplex<f32>> loc(#loc9)
    %5 = stablehlo.add %arg0, %4 : tensor<4x4xcomplex<f32>> loc(#loc10)
    %6 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc11)
    %7 = stablehlo.divide %5, %6 : tensor<4x4xcomplex<f32>> loc(#loc11)
    %8:2 = stablehlo.custom_call @oneapisolver_potrf_ffi(%7) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<i32>) loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc4)
    %10 = stablehlo.compare EQ, %8#1, %9, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc4)
    %14 = stablehlo.select %13, %8#0, %12 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc4)
    %15 = stablehlo.iota dim = 0 : tensor<4x4xi64> loc(#loc12)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4xi64> loc(#loc10)
    %17 = stablehlo.add %15, %16 : tensor<4x4xi64> loc(#loc10)
    %18 = stablehlo.iota dim = 1 : tensor<4x4xi64> loc(#loc12)
    %19 = stablehlo.compare GE, %17, %18, SIGNED : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1> loc(#loc13)
    %20 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc14)
    %21 = stablehlo.select %19, %14, %20 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc15)
    return %21 : tensor<4x4xcomplex<f32>> loc(#loc3)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":269:4)
#loc3 = loc("jit(cholesky)"(#loc2))
#loc4 = loc("cholesky"(#loc3))
#loc5 = loc("transpose"(#loc3))
#loc6 = loc("real"(#loc3))
#loc7 = loc("imag"(#loc3))
#loc8 = loc("neg"(#loc3))
#loc9 = loc("complex"(#loc3))
#loc10 = loc("add"(#loc3))
#loc11 = loc("div"(#loc3))
#loc12 = loc("iota"(#loc3))
#loc13 = loc("ge"(#loc3))
#loc14 = loc("broadcast_in_dim"(#loc3))
#loc15 = loc("select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x013\x07\x01\x05\t!\x01\x03\x0f\x03\x1f\x13\x17\x1b\x1f#'+/37;?CGK\x03\xf7\xb3-\x01U\x0f\x0f\x07\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x17\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03O\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b///\x1f/O\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03)\x17\x0f\x07\x0f\x17\x17\x07\x0b\x07\x0f\x13\x07\x17\x17\x07\x13\x13\x13\x0f\x17\x02n\x06\x1d\x1f!\x1d%\x01\x1f\x1d;\x01\x11\x03\x05\x1d=\x01\x1dG\x01\x03\x07\x11\x13\x15\t\x17\t\x05'\x11\x01\x00\x05)\x05+\x05-\x1d\x1d\x05\x05/\x051\x17#6\x04\t\x053\x055\x1d)\x01\x057\x1d-\x01\x059\x1d1\x01\x05;\x1d5\x01\x05=\x1d9\x01\x05?\x05A\x05C\x03\x07AyC\x7fE\xa3\x05E\x05G\x05I\x05K\x1dK\x01\x05M\x1dO\x01\x05O\x1dS\x01\x05Q\x1f#\x01\x1f%!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03]\r\x01#\x1f\x03\x03c\r\x03eg\x1dS\x1dU\x1dW\x1dY\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x17\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x0b\t\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00@\x00\x00\x00\x00\x1f\x19!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03{}\x1d[\x05\x03\r\x03\x81\x83\x1d]\x1d_\x0b\x03\x1da\x1dc\x03\x01\x05\x01\x03\x03W\x03\x03\x93\x15\x03\x01\x01\x01\x03\x05W\x97\x1f'\x01\x07\x01\x1f\x19!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\t\x01\x13\t\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\xa5\x05\xab\xb1\x01\x01\x01\x01\x01\x13\x05\xa7\xa9\x11\x03\x01\x11\x03\x05\x13\x05\xad\xaf\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x13)\x01\x13\x1d)\x01!)\x05\x11\x11\t)\x05\x11\x11\x15\x01\x03\x15\t)\x01\t)\x03\t\t\x13)\x05\x11\x11\x11\x11\x03\x05\x03\x05\x1b)\x03\x01\t)\x03\t\x1b)\x03\x01\x1b)\x01\x11)\x05\x05\x05\x11\x04\xba\x03\x05\x01Q\x05\x0f\x01\x07\x04\x92\x03\x03\x01\x05\x0fP\x05\x03\x07\x04f\x03\x03;s\x03\x0b\x1b\x00\x05B\x01\x05\x03\x07\x05B\x01\x07\x03\x17\x05B\x03\t\x03\x07\x05B\x03\x0b\x03\x0b\x05B\x01\r\x03\x07\x11F'\x0f\x03\x05\x03\x01\x13\x06+\x03\x0f\x03\r\x15\x06/\x03\x0f\x03\r\x17\x063\x03\x0f\x03\x11\x19\x067\x03\x05\x05\x0f\x13\x07\x06\x07\x03\x05\x05\x01\x15\x03F\x0b\x11\x03\x05\x03\x0b\x1b\x06\x0b\x03\x05\x05\x17\x19\x1dG\x03?\x13\x05\x05\x0b\x03\x1b\x03F\x03\x11\x03\x0b\x03\t\tF\x03\x15\x03)\x05\x1f!\x03F\x03\x11\x03+\x03#\x03F\x03\x11\x03\x05\x03\x07\x03F\x03\x17\x03\x1d\x03%\x0b\x06\x03\x03\x05\x07)\x1d'\rB\r\x19\x03\r\x03F\x07\x11\x03\r\x03\x05\x07\x06\x07\x03\r\x05-/\rB\r\x1b\x03\r\tFI\x1d\x03\x1d\x0513\x03FM\x11\x03\x05\x03\x03\x0b\x06Q\x03\x05\x075+7\x1f\x04\x01\x039\x06\x03\x01\x05\x01\x00\xaa\te/\x03\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x11\t\x0b\x0b\x15\x13w\x1d\x05\x1b%)9\x15\x1f\x15\x17\x15\x11\x11\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00jit(cholesky)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00cholesky\x00transpose\x00real\x00imag\x00neg\x00complex\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00\x00oneapisolver_potrf_ffi\x00\x08[\x1f\x053\x01\x0b[_aik\x03m\x03o\x03q\x03s\x03u\x03w\x03U\x11\x85\x87\x89\x8b\x8d\x8f\x91\x95\x05Y\x99\x03\x9b\x03\x9d\x03\x9f\x05Y\xa1",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["c128"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_potrf_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[145.98892137813692    +0.j               ,
         40.91401793296874    +4.175781485327597j,
          2.7823411356357504  +9.8851183291892j  ,
         26.733955883991726  +52.65661439791964j ],
       [ 40.91401793296874    -4.175781485327597j,
         33.78265769398051    +0.j               ,
          3.6132624138937786  +5.213542211682853j,
          4.589810669550228  -12.339958092149333j],
       [  2.7823411356357504  -9.8851183291892j  ,
          3.6132624138937786  -5.213542211682853j,
         93.29157057865525    +0.j               ,
         -0.20930676609536647-34.05459375322113j ],
       [ 26.733955883991726  -52.65661439791964j ,
          4.589810669550228  +12.339958092149333j,
         -0.20930676609536647+34.05459375322113j ,
         78.03147614159427    +0.j               ]]),),
    expected_outputs=(array([[12.082587528263014  +0.j                 ,
         0.                 +0.j                 ,
         0.                 +0.j                 ,
         0.                 +0.j                 ],
       [ 3.386196693155718  -0.34560324728124736j,
         4.711357346318622  +0.j                 ,
         0.                 +0.j                 ,
         0.                 +0.j                 ],
       [ 0.2302769277795365 -0.8181292546870778j ,
         0.5414047646829694 -0.535467786342578j  ,
         9.591108526566718  +0.j                 ,
         0.                 +0.j                 ],
       [ 2.2126018803055993 -4.358057764923928j  ,
        -0.9357501654279168 +5.589157126898457j  ,
        -0.08182988296307438+3.2032822133798207j ,
         3.4294580838507396 +0.j                 ]]),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f64>> loc("x")) -> (tensor<4x4xcomplex<f64>> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc3)
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc3)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc4)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %cst_2 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc3)
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc5)
    %1 = stablehlo.real %0 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc6)
    %2 = stablehlo.imag %0 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc7)
    %3 = stablehlo.negate %2 : tensor<4x4xf64> loc(#loc8)
    %4 = stablehlo.complex %1, %3 : tensor<4x4xcomplex<f64>> loc(#loc9)
    %5 = stablehlo.add %arg0, %4 : tensor<4x4xcomplex<f64>> loc(#loc10)
    %6 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc11)
    %7 = stablehlo.divide %5, %6 : tensor<4x4xcomplex<f64>> loc(#loc11)
    %8:2 = stablehlo.custom_call @oneapisolver_potrf_ffi(%7) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xcomplex<f64>>) -> (tensor<4x4xcomplex<f64>>, tensor<i32>) loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc4)
    %10 = stablehlo.compare EQ, %8#1, %9, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc4)
    %14 = stablehlo.select %13, %8#0, %12 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc4)
    %15 = stablehlo.iota dim = 0 : tensor<4x4xi64> loc(#loc12)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4xi64> loc(#loc10)
    %17 = stablehlo.add %15, %16 : tensor<4x4xi64> loc(#loc10)
    %18 = stablehlo.iota dim = 1 : tensor<4x4xi64> loc(#loc12)
    %19 = stablehlo.compare GE, %17, %18, SIGNED : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1> loc(#loc13)
    %20 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc14)
    %21 = stablehlo.select %19, %14, %20 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc15)
    return %21 : tensor<4x4xcomplex<f64>> loc(#loc3)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":269:4)
#loc3 = loc("jit(cholesky)"(#loc2))
#loc4 = loc("cholesky"(#loc3))
#loc5 = loc("transpose"(#loc3))
#loc6 = loc("real"(#loc3))
#loc7 = loc("imag"(#loc3))
#loc8 = loc("neg"(#loc3))
#loc9 = loc("complex"(#loc3))
#loc10 = loc("add"(#loc3))
#loc11 = loc("div"(#loc3))
#loc12 = loc("iota"(#loc3))
#loc13 = loc("ge"(#loc3))
#loc14 = loc("broadcast_in_dim"(#loc3))
#loc15 = loc("select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x013\x07\x01\x05\t!\x01\x03\x0f\x03\x1f\x13\x17\x1b\x1f#'+/37;?CGK\x03\xf7\xb3-\x01U\x0f\x0f\x07\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x17\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03O\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0bO/O\x1fOO\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03)\x17\x0f\x07\x0f\x17\x17\x07\x0b\x07\x0f\x13\x07\x17\x17\x07\x13\x13\x13\x0f\x17\x02\xce\x06\x1d\x1f!\x1d%\x01\x1f\x1d;\x01\x11\x03\x05\x1d=\x01\x1dG\x01\x03\x07\x11\x13\x15\t\x17\t\x05'\x11\x01\x00\x05)\x05+\x05-\x1d\x1d\x05\x05/\x051\x17#6\x04\t\x053\x055\x1d)\x01\x057\x1d-\x01\x059\x1d1\x01\x05;\x1d5\x01\x05=\x1d9\x01\x05?\x05A\x05C\x03\x07AyC\x7fE\xa3\x05E\x05G\x05I\x05K\x1dK\x01\x05M\x1dO\x01\x05O\x1dS\x01\x05Q\x1f#\x01\x1f%!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03]\r\x01#\x1f\x03\x03c\r\x03eg\x1dS\x1dU\x1dW\x1dY\x1f\x07!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x17\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x0b\t\x00\x00\x00\x00\x1f\x07!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03{}\x1d[\x05\x03\r\x03\x81\x83\x1d]\x1d_\x0b\x03\x1da\x1dc\x03\x01\x05\x01\x03\x03W\x03\x03\x93\x15\x03\x01\x01\x01\x03\x05W\x97\x1f'\x01\x07\x01\x1f\x19!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\t\x01\x13\t\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\xa5\x05\xab\xb1\x01\x01\x01\x01\x01\x13\x05\xa7\xa9\x11\x03\x01\x11\x03\x05\x13\x05\xad\xaf\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x13)\x01\x13\x1d)\x01!)\x05\x11\x11\t)\x05\x11\x11\x15\x01\x03\x15\x0b)\x01\t)\x03\t\t\x13)\x05\x11\x11\x11\x11\x03\x05\x03\x05\x1b)\x03\x01\t)\x03\t\x1b)\x03\x01\x1b)\x01\x11)\x05\x05\x05\x11\x04\xba\x03\x05\x01Q\x05\x0f\x01\x07\x04\x92\x03\x03\x01\x05\x0fP\x05\x03\x07\x04f\x03\x03;s\x03\x0b\x1b\x00\x05B\x01\x05\x03\x07\x05B\x01\x07\x03\x17\x05B\x03\t\x03\x07\x05B\x03\x0b\x03\x0b\x05B\x01\r\x03\x07\x11F'\x0f\x03\x05\x03\x01\x13\x06+\x03\x0f\x03\r\x15\x06/\x03\x0f\x03\r\x17\x063\x03\x0f\x03\x11\x19\x067\x03\x05\x05\x0f\x13\x07\x06\x07\x03\x05\x05\x01\x15\x03F\x0b\x11\x03\x05\x03\x0b\x1b\x06\x0b\x03\x05\x05\x17\x19\x1dG\x03?\x13\x05\x05\x0b\x03\x1b\x03F\x03\x11\x03\x0b\x03\t\tF\x03\x15\x03)\x05\x1f!\x03F\x03\x11\x03+\x03#\x03F\x03\x11\x03\x05\x03\x07\x03F\x03\x17\x03\x1d\x03%\x0b\x06\x03\x03\x05\x07)\x1d'\rB\r\x19\x03\r\x03F\x07\x11\x03\r\x03\x05\x07\x06\x07\x03\r\x05-/\rB\r\x1b\x03\r\tFI\x1d\x03\x1d\x0513\x03FM\x11\x03\x05\x03\x03\x0b\x06Q\x03\x05\x075+7\x1f\x04\x01\x039\x06\x03\x01\x05\x01\x00\xaa\te/\x03\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x11\t\x0b\x0b\x15\x13w\x1d\x05\x1b%)9\x15\x1f\x15\x17\x15\x11\x11\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00jit(cholesky)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00cholesky\x00transpose\x00real\x00imag\x00neg\x00complex\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00\x00oneapisolver_potrf_ffi\x00\x08[\x1f\x053\x01\x0b[_aik\x03m\x03o\x03q\x03s\x03u\x03w\x03U\x11\x85\x87\x89\x8b\x8d\x8f\x91\x95\x05Y\x99\x03\x9b\x03\x9d\x03\x9f\x05Y\xa1",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
