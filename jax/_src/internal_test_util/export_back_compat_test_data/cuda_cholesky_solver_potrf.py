# Copyright 2025 The JAX Authors.
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
complex64 = np.complex64

data_2025_10_15 = {}


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_10_15["f32"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_potrf_ffi'],
    serialized_date=datetime.date(2025, 10, 15),
    inputs=(array([[ 10.895978 ,   4.2912526, -18.31493  ,   3.697414 ],
       [  4.2912526,  61.31485  ,   4.850662 ,  -2.1822023],
       [-18.31493  ,   4.850662 ,  36.91168  ,  -4.3174276],
       [  3.697414 ,  -2.1822023,  -4.3174276,  16.82287  ]],
      dtype=float32),),
    expected_outputs=(array([[ 3.3009057,  0.       ,  0.       ,  0.       ],
       [ 1.3000228,  7.721709 ,  0.       ,  0.       ],
       [-5.548456 ,  1.5623201,  1.9197574,  0.       ],
       [ 1.120121 , -0.4711891,  1.3718729,  3.6693518]], dtype=float32),),
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
    %4:2 = stablehlo.custom_call @cusolver_potrf_ffi(%3) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<i32>) loc(#loc4)
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
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":225:6)
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
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03\xdf\xa1'\x01E\x0f\x0f\x07\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x17\x0b\x0f\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03M\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x1f\x1f\x1f\x1fO\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03#\x17\x0f\x0f\x07\x17\x07\x07\x07\x13\x07\x17\x17\x13\x13\x13\x0f\x17\x02\x9a\x05\x1d\x1f\x03\x1d!#\x1f\x1d+\x03\x11\x03\x05\x1d-\x03\x1d7\x03\x03\x07\x11\x13\x15\t\x17\t\x05\x1f\x11\x01\x00\x05!\x05#\x05%\x1d\x1d\x05\x05'\x05)\x05+\x17%\x86\x03\r\x05-\x1d)\x03\x05/\x051\x053\x03\x071g3m5\x91\x055\x057\x059\x05;\x1d;\x03\x05=\x1d?\x03\x05?\x1dC\x03\x05A\x1f\x1d\x01\x1f\x1f!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03M\r\x01#\x1b\x03\x03S\r\x03UW\x1dC\x1dE\x1dG\x1dI\x1f\x07\t\x00\x00\x00\x00\x1f\x07\t\x00\x00\xc0\x7f\x1f\t\t\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00@\x1f\x15!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03ik\x1dK\x05\x03\r\x03oq\x1dM\x1dO\x0b\x03\x1dQ\x1dS\x03\x01\x05\x01\x03\x03G\x03\x03\x81\x15\x03\x01\x01\x01\x03\x05G\x85\x1f!\x01\x07\x01\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x01\x13\x0b\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\x93\x05\x99\x9f\x01\x01\x01\x01\x01\x13\x05\x95\x97\x11\x03\x01\x11\x03\x05\x13\x05\x9b\x9d\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x11)\x01\x11)\x01\x13\x1d)\x05\x11\x11\x13\x01\t\x1b)\x03\t\x0b\x13)\x05\x11\x11\x0f\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03\t\x17)\x03\x01\x17)\x01\x0f)\x05\x05\x05\x0f\x04.\x03\x05\x01Q\x05\x0f\x01\x07\x04\x06\x03\x03\x01\x05\x0fP\x05\x03\x07\x04\xda\x02\x031_\x03\x0b\x1b\x00\x05B\x05\x05\x03\x07\x05B\x01\x07\x03\x07\x05B\x05\t\x03\t\x05B\x05\x0b\x03\x07\x11F'\r\x03\x05\x03\x01\x07\x06\x07\x03\x05\x05\x01\x0b\x03F\x0b\x0f\x03\x05\x03\t\x13\x06\x0b\x03\x05\x05\r\x0f\x15G\x01/\x11\x05\x05\t\x03\x11\x03F\x01\x0f\x03\t\x03\x07\tF\x01\x13\x03#\x05\x15\x17\x03F\x01\x0f\x03%\x03\x19\x03F\x01\x0f\x03\x05\x03\x05\x03F\x01\x15\x03\x19\x03\x1b\x0b\x06\x01\x03\x05\x07\x1f\x13\x1d\rB\r\x17\x03\r\x03F\x07\x0f\x03\r\x03\x07\x07\x06\x07\x03\r\x05#%\rB\r\x19\x03\r\tF9\x1b\x03\x19\x05')\x03F=\x0f\x03\x05\x03\x03\x0b\x06A\x03\x05\x07+!-\x17\x04\x05\x03/\x06\x03\x01\x05\x01\x00r\x08U'\x03\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x15i\x1d\x13\x05\x1b%)9\x15\x1f\x15\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00cholesky\x00jit(cholesky)\x00third_party/py/jax/tests/export_back_compat_test.py\x00transpose\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00\x00cusolver_potrf_ffi\x00\x08W\x1d\x053\x01\x0bKOQY[\x03]\x03_\x03a\x03c\x03e\x03E\x11suwy{}\x7f\x83\x05I\x87\x03\x89\x03\x8b\x03\x8d\x05I\x8f",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_10_15["f64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_potrf_ffi'],
    serialized_date=datetime.date(2025, 10, 15),
    inputs=(array([[ 19.64203917602577   ,  -3.257129261958479  ,
         -6.113417498905007  , -38.31045157187507   ],
       [ -3.257129261958479  ,  30.34754344041282   ,
         -0.21110017803103753,  12.603145919345353  ],
       [ -6.113417498905007  ,  -0.21110017803103753,
          8.567277222442657  ,  15.150956096041329  ],
       [-38.31045157187507   ,  12.603145919345353  ,
         15.150956096041329  ,  77.94548898633911   ]]),),
    expected_outputs=(array([[ 4.431934022074987  ,  0.                 ,  0.                 ,
         0.                 ],
       [-0.7349227776711179 ,  5.459618297213917  ,  0.                 ,
         0.                 ],
       [-1.3794017393884324 , -0.22434790660947998,  2.571807940071491  ,
         0.                 ],
       [-8.644183641059373  ,  1.1448306689037868 ,  1.3546868938685381 ,
         0.27886255592811615]]),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf64> loc("x")) -> (tensor<4x4xf64> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc4)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f64> loc(#loc)
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc5)
    %1 = stablehlo.add %arg0, %0 : tensor<4x4xf64> loc(#loc6)
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc7)
    %3 = stablehlo.divide %1, %2 : tensor<4x4xf64> loc(#loc7)
    %4:2 = stablehlo.custom_call @cusolver_potrf_ffi(%3) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<i32>) loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc4)
    %6 = stablehlo.compare  EQ, %4#1, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %4#0, %8 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc4)
    %11 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc8)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc6)
    %13 = stablehlo.add %11, %12 : tensor<4x4xi32> loc(#loc6)
    %14 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc8)
    %15 = stablehlo.compare  GE, %13, %14,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc10)
    %17 = stablehlo.select %15, %10, %16 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc11)
    return %17 : tensor<4x4xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":225:6)
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
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03\xdf\xa1'\x01E\x0f\x0f\x07\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x17\x0b\x0f\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03M\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b//\x1f/O\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03#\x17\x0f\x0f\x07\x17\x07\x07\x07\x13\x07\x17\x17\x13\x13\x13\x0f\x17\x02\xca\x05\x1d\x1f\x03\x1d!#\x1f\x1d+\x03\x11\x03\x05\x1d-\x03\x1d7\x03\x03\x07\x11\x13\x15\t\x17\t\x05\x1f\x11\x01\x00\x05!\x05#\x05%\x1d\x1d\x05\x05'\x05)\x05+\x17%\x86\x03\r\x05-\x1d)\x03\x05/\x051\x053\x03\x071g3m5\x91\x055\x057\x059\x05;\x1d;\x03\x05=\x1d?\x03\x05?\x1dC\x03\x05A\x1f\x1d\x01\x1f\x1f!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03M\r\x01#\x1b\x03\x03S\r\x03UW\x1dC\x1dE\x1dG\x1dI\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\t\t\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x15!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03ik\x1dK\x05\x03\r\x03oq\x1dM\x1dO\x0b\x03\x1dQ\x1dS\x03\x01\x05\x01\x03\x03G\x03\x03\x81\x15\x03\x01\x01\x01\x03\x05G\x85\x1f!\x01\x07\x01\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x01\x13\x0b\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\x93\x05\x99\x9f\x01\x01\x01\x01\x01\x13\x05\x95\x97\x11\x03\x01\x11\x03\x05\x13\x05\x9b\x9d\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x11)\x01\x11)\x01\x13\x1d)\x05\x11\x11\x13\x01\x0b\x1b)\x03\t\x0b\x13)\x05\x11\x11\x0f\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03\t\x17)\x03\x01\x17)\x01\x0f)\x05\x05\x05\x0f\x04.\x03\x05\x01Q\x05\x0f\x01\x07\x04\x06\x03\x03\x01\x05\x0fP\x05\x03\x07\x04\xda\x02\x031_\x03\x0b\x1b\x00\x05B\x05\x05\x03\x07\x05B\x01\x07\x03\x07\x05B\x05\t\x03\t\x05B\x05\x0b\x03\x07\x11F'\r\x03\x05\x03\x01\x07\x06\x07\x03\x05\x05\x01\x0b\x03F\x0b\x0f\x03\x05\x03\t\x13\x06\x0b\x03\x05\x05\r\x0f\x15G\x01/\x11\x05\x05\t\x03\x11\x03F\x01\x0f\x03\t\x03\x07\tF\x01\x13\x03#\x05\x15\x17\x03F\x01\x0f\x03%\x03\x19\x03F\x01\x0f\x03\x05\x03\x05\x03F\x01\x15\x03\x19\x03\x1b\x0b\x06\x01\x03\x05\x07\x1f\x13\x1d\rB\r\x17\x03\r\x03F\x07\x0f\x03\r\x03\x07\x07\x06\x07\x03\r\x05#%\rB\r\x19\x03\r\tF9\x1b\x03\x19\x05')\x03F=\x0f\x03\x05\x03\x03\x0b\x06A\x03\x05\x07+!-\x17\x04\x05\x03/\x06\x03\x01\x05\x01\x00r\x08U'\x03\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x15i\x1d\x13\x05\x1b%)9\x15\x1f\x15\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00cholesky\x00jit(cholesky)\x00third_party/py/jax/tests/export_back_compat_test.py\x00transpose\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00\x00cusolver_potrf_ffi\x00\x08W\x1d\x053\x01\x0bKOQY[\x03]\x03_\x03a\x03c\x03e\x03E\x11suwy{}\x7f\x83\x05I\x87\x03\x89\x03\x8b\x03\x8d\x05I\x8f",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_10_15["c64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_potrf_ffi'],
    serialized_date=datetime.date(2025, 10, 15),
    inputs=(array([[123.867     +0.j      ,  87.94228   +8.670872j,
        -28.046396 +75.86514j , -44.46436  -25.990704j],
       [ 87.94228   -8.670872j, 122.20061   +0.j      ,
         -4.3756332+54.971073j, -21.559742  +1.933352j],
       [-28.046396 -75.86514j ,  -4.3756332-54.971073j,
        101.799835  +0.j      ,  -3.7747602+28.983824j],
       [-44.46436  +25.990704j, -21.559742  -1.933352j,
         -3.7747602-28.983824j,  83.28284   +0.j      ]], dtype=complex64),),
    expected_outputs=(array([[11.129555  +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [ 7.901689  -0.7790852j ,  7.6913548 +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [-2.5199926 -6.8165474j ,  1.3295308 +0.11109254j,
         6.8705287 +0.j        ,  0.        +0.j        ],
       [-3.9951606 +2.3352869j ,  1.5378516 -2.2458322j ,
         0.04088954+1.0612036j ,  7.3028345 +0.j        ]],
      dtype=complex64),),
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
    %8:2 = stablehlo.custom_call @cusolver_potrf_ffi(%7) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<i32>) loc(#loc4)
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
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":225:6)
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
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x013\x07\x01\x05\t!\x01\x03\x0f\x03\x1f\x13\x17\x1b\x1f#'+/37;?CGK\x03\xe7\xa5+\x01I\x0f\x0f\x07\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x17\x0b\x0f\x0b\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03M\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b//\x1f/O\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03'\x17\x0f\x0f\x07\x17\x17\x07\x0b\x07\x07\x13\x07\x17\x17\x13\x13\x13\x0f\x17\x02\xfa\x05\x1d#%\x1d!\x01\x1f\x1d-\x01\x1d/\x01\x11\x03\x05\x1d1\x01\x1d;\x01\x03\x07\x13\x15\x17\x0b\x19\x0b\x05'\x11\x01\x00\x05)\x05+\x05-\x1d\x1f\x05\x05/\x051\x053\x17'\x86\x03\r\x055\x1d+\x01\x057\x059\x05;\x05=\x03\x075k7q9\x95\x05?\x05A\x05C\x05E\x1d?\x01\x05G\x1dC\x01\x05I\x1dG\x01\x05K\x1f!\x01\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03Q\r\x01#\x1f\x03\x03W\r\x03Y[\x1dM\x1dO\x1dQ\x1dS\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\t\t\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00@\x00\x00\x00\x00\x1f\x19!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03mo\x1dU\x05\x03\r\x03su\x1dW\x1dY\x0b\x03\x1d9\x1d[\x03\x01\x05\x01\x03\x03K\x03\x03\x85\x15\x03\x01\x01\x01\x03\x05K\x89\x1f%\x01\x07\x01\x1f\x19!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x01\x13\x0b\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\x97\x05\x9d\xa3\x01\x01\x01\x01\x01\x13\x05\x99\x9b\x11\x03\x01\x11\x03\x05\x13\x05\x9f\xa1\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x13)\x01\x13)\x01\x17\x1d)\x05\x11\x11\x17)\x05\x11\x11\x15\x01\x03\x15\t\x1b)\x03\t\x0b\x13)\x05\x11\x11\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03\t\x1b)\x03\x01\x1b)\x01\x11)\x05\x05\x05\x11\x04\xa2\x03\x05\x01Q\x05\x11\x01\x07\x04z\x03\x03\x01\x05\x0fP\x05\x03\x07\x04N\x03\x039o\x03\x0b\x1d\x00\x05B\x05\x05\x03\x07\x05B\x03\x07\x03\x07\x05B\x05\t\x03\t\x05B\x05\x0b\x03\x07\x11F)\r\x03\x05\x03\x01\x13\x06\x07\x03\x0f\x03\x0b\x15\x06\x07\x03\x0f\x03\x0b\x17\x06\x07\x03\x0f\x03\x0f\x19\x06\x07\x03\x05\x05\r\x11\x07\x06\t\x03\x05\x05\x01\x13\x03F\r\x0f\x03\x05\x03\t\x1b\x06\r\x03\x05\x05\x15\x17\x1dG\x033\x11\x05\x05\t\x03\x19\x03F\x03\x0f\x03\t\x03\x07\tF\x03\x13\x03'\x05\x1d\x1f\x03F\x03\x0f\x03)\x03!\x03F\x03\x0f\x03\x05\x03\x05\x03F\x03\x15\x03\x1d\x03#\x0b\x06\x03\x03\x05\x07'\x1b%\rB\x0f\x17\x03\r\x03F\t\x0f\x03\r\x03\x07\x07\x06\t\x03\r\x05+-\rB\x0f\x19\x03\r\tF=\x1b\x03\x1d\x05/1\x03FA\x0f\x03\x05\x03\x03\x0b\x06E\x03\x05\x073)5\x1f\x04\x05\x037\x06\x03\x01\x05\x01\x00\x16\t]'\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x03\x15i\x1d\x13\x05\x1b%)9\x15\x1f\x15\x17\x15\x11\x11\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00cholesky\x00jit(cholesky)\x00third_party/py/jax/tests/export_back_compat_test.py\x00transpose\x00\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00cusolver_potrf_ffi\x00\x08W\x1d\x057\x01\x0bOSU]_\x03a\x03c\x03e\x03g\x03i\x03I\x11wy{}\x7f\x81\x83\x87\x05M\x8b\x03\x8d\x03\x8f\x03\x91\x05M\x93",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_10_15["c128"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_potrf_ffi'],
    serialized_date=datetime.date(2025, 10, 15),
    inputs=(array([[145.98892137813692    +0.j               ,
         40.91401793296874    +4.175781485327598j,
          2.782341135635754   +9.885118329189208j,
         26.733955883991726  +52.65661439791964j ],
       [ 40.91401793296874    -4.175781485327598j,
         33.78265769398051    +0.j               ,
          3.6132624138937786  +5.213542211682853j,
          4.589810669550227  -12.339958092149333j],
       [  2.782341135635754   -9.885118329189208j,
          3.6132624138937786  -5.213542211682853j,
         93.29157057865525    +0.j               ,
         -0.20930676609536647-34.05459375322113j ],
       [ 26.733955883991726  -52.65661439791964j ,
          4.589810669550227  +12.339958092149333j,
         -0.20930676609536647+34.05459375322113j ,
         78.03147614159427    +0.j               ]]),),
    expected_outputs=(array([[12.082587528263014  +0.j                ,
         0.                 +0.j                ,
         0.                 +0.j                ,
         0.                 +0.j                ],
       [ 3.386196693155718  -0.3456032472812474j,
         4.711357346318622  +0.j                ,
         0.                 +0.j                ,
         0.                 +0.j                ],
       [ 0.2302769277795368 -0.8181292546870784j,
         0.5414047646829693 -0.5354677863425775j,
         9.59110852656672   +0.j                ,
         0.                 +0.j                ],
       [ 2.2126018803055993 -4.358057764923928j ,
        -0.935750165427917  +5.589157126898457j ,
        -0.08182988296307511+3.2032822133798207j,
         3.4294580838507387 +0.j                ]]),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f64>> loc("x")) -> (tensor<4x4xcomplex<f64>> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc4)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %cst_1 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc5)
    %1 = stablehlo.real %0 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc6)
    %2 = stablehlo.imag %0 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc6)
    %3 = stablehlo.negate %2 : tensor<4x4xf64> loc(#loc6)
    %4 = stablehlo.complex %1, %3 : tensor<4x4xcomplex<f64>> loc(#loc6)
    %5 = stablehlo.add %arg0, %4 : tensor<4x4xcomplex<f64>> loc(#loc7)
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc8)
    %7 = stablehlo.divide %5, %6 : tensor<4x4xcomplex<f64>> loc(#loc8)
    %8:2 = stablehlo.custom_call @cusolver_potrf_ffi(%7) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=4, j=4, k=4, l=4}, custom>} : (tensor<4x4xcomplex<f64>>) -> (tensor<4x4xcomplex<f64>>, tensor<i32>) loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc4)
    %10 = stablehlo.compare  EQ, %8#1, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc4)
    %14 = stablehlo.select %13, %8#0, %12 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc4)
    %15 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc7)
    %17 = stablehlo.add %15, %16 : tensor<4x4xi32> loc(#loc7)
    %18 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc9)
    %19 = stablehlo.compare  GE, %17, %18,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc10)
    %20 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc11)
    %21 = stablehlo.select %19, %14, %20 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc12)
    return %21 : tensor<4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":225:6)
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
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x013\x07\x01\x05\t!\x01\x03\x0f\x03\x1f\x13\x17\x1b\x1f#'+/37;?CGK\x03\xe7\xa5+\x01I\x0f\x0f\x07\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x17\x0b\x0f\x0b\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03M\x0fO\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0bOO\x1fOO\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x13\x0f\x0bO\x0f\x0f\x0b\x05\x11C\x13\x0f\x0f\x13\x0f\x0f\x0b\x01\x05\x0b\x0f\x03'\x17\x0f\x0f\x07\x17\x17\x07\x0b\x07\x07\x13\x07\x17\x17\x13\x13\x13\x0f\x17\x02Z\x06\x1d#%\x1d!\x01\x1f\x1d-\x01\x1d/\x01\x11\x03\x05\x1d1\x01\x1d;\x01\x03\x07\x13\x15\x17\x0b\x19\x0b\x05'\x11\x01\x00\x05)\x05+\x05-\x1d\x1f\x05\x05/\x051\x053\x17'\x86\x03\r\x055\x1d+\x01\x057\x059\x05;\x05=\x03\x075k7q9\x95\x05?\x05A\x05C\x05E\x1d?\x01\x05G\x1dC\x01\x05I\x1dG\x01\x05K\x1f!\x01\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x03\x03Q\r\x01#\x1f\x03\x03W\r\x03Y[\x1dM\x1dO\x1dQ\x1dS\x1f\x07!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\t\t\x00\x00\x00\x00\x1f\x07!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03mo\x1dU\x05\x03\r\x03su\x1dW\x1dY\x0b\x03\x1d9\x1d[\x03\x01\x05\x01\x03\x03K\x03\x03\x85\x15\x03\x01\x01\x01\x03\x05K\x89\x1f%\x01\x07\x01\x1f\x19!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x01\x13\x0b\x05\x07\x05\x15\t\x11\x11\x11\x11\x03\x97\x05\x9d\xa3\x01\x01\x01\x01\x01\x13\x05\x99\x9b\x11\x03\x01\x11\x03\x05\x13\x05\x9f\xa1\x11\x03\t\x11\x03\r\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\x13)\x01\x13)\x01\x17\x1d)\x05\x11\x11\x17)\x05\x11\x11\x15\x01\x03\x15\x0b\x1b)\x03\t\x0b\x13)\x05\x11\x11\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03\t\x1b)\x03\x01\x1b)\x01\x11)\x05\x05\x05\x11\x04\xa2\x03\x05\x01Q\x05\x11\x01\x07\x04z\x03\x03\x01\x05\x0fP\x05\x03\x07\x04N\x03\x039o\x03\x0b\x1d\x00\x05B\x05\x05\x03\x07\x05B\x03\x07\x03\x07\x05B\x05\t\x03\t\x05B\x05\x0b\x03\x07\x11F)\r\x03\x05\x03\x01\x13\x06\x07\x03\x0f\x03\x0b\x15\x06\x07\x03\x0f\x03\x0b\x17\x06\x07\x03\x0f\x03\x0f\x19\x06\x07\x03\x05\x05\r\x11\x07\x06\t\x03\x05\x05\x01\x13\x03F\r\x0f\x03\x05\x03\t\x1b\x06\r\x03\x05\x05\x15\x17\x1dG\x033\x11\x05\x05\t\x03\x19\x03F\x03\x0f\x03\t\x03\x07\tF\x03\x13\x03'\x05\x1d\x1f\x03F\x03\x0f\x03)\x03!\x03F\x03\x0f\x03\x05\x03\x05\x03F\x03\x15\x03\x1d\x03#\x0b\x06\x03\x03\x05\x07'\x1b%\rB\x0f\x17\x03\r\x03F\t\x0f\x03\r\x03\x07\x07\x06\t\x03\r\x05+-\rB\x0f\x19\x03\r\tF=\x1b\x03\x1d\x05/1\x03FA\x0f\x03\x05\x03\x03\x0b\x06E\x03\x05\x073)5\x1f\x04\x05\x037\x06\x03\x01\x05\x01\x00\x16\t]'\x05\x1f\r\x0f\x0b\x0f!\x13#\x07\x0b%3)\t\t\x03\x15i\x1d\x13\x05\x1b%)9\x15\x1f\x15\x17\x15\x11\x11\x1b\x11\x11\x15\x17\x0f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00add_v1\x00compare_v1\x00select_v1\x00iota_v1\x00func_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00x\x00cholesky\x00jit(cholesky)\x00third_party/py/jax/tests/export_back_compat_test.py\x00transpose\x00\x00add\x00div\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00iota\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result\x00main\x00public\x00lower\x00num_batch_dims\x000\x00cusolver_potrf_ffi\x00\x08W\x1d\x057\x01\x0bOSU]_\x03a\x03c\x03e\x03g\x03i\x03I\x11wy{}\x7f\x81\x83\x87\x05M\x8b\x03\x8d\x03\x8f\x03\x91\x05M\x93",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
