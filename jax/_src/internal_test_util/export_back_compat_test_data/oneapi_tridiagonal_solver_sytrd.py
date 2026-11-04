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
    custom_call_targets=['oneapisolver_sytrd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[[ 4.6987123 , -0.6352046 , -3.9608796 , -1.7019314 ],
        [-2.782619  , -2.9592047 ,  1.6707699 , -3.2680469 ],
        [-1.6345627 , -2.5428991 ,  3.3639514 ,  2.17219   ],
        [-0.12571943,  6.198895  ,  1.665511  ,  2.0166407 ]],

       [[ 1.9488412 ,  0.80734503, -1.1175517 , -1.2200787 ],
        [-0.83973795, -2.8905864 , -1.8656269 ,  0.42590132],
        [ 0.8000843 ,  1.0724624 ,  1.3567352 , -2.5237849 ],
        [-3.3680031 , -3.4638197 , -2.2655149 ,  0.129971  ]]],
      dtype=float32),),
    expected_outputs=(array([[[ 4.6987123  , -0.6352046  , -3.9608796  , -1.7019314  ],
        [ 3.229639   , -3.0682716  ,  1.6707699  , -3.2680469  ],
        [ 0.2718717  ,  6.550433   ,  0.8836329  ,  2.17219    ],
        [ 0.020910518,  0.806407   , -1.0597477  ,  4.606025   ]],

       [[ 1.9488412  ,  0.80734503 , -1.1175517  , -1.2200787  ],
        [ 3.5621257  , -0.67145586 , -1.8656269  ,  0.42590132 ],
        [-0.18176036 , -4.6919804  , -0.090105176, -2.5237849  ],
        [ 0.7651312  , -0.45560563 , -0.9679276  , -0.6423193  ]]],
      dtype=float32), array([[ 4.6987123  , -3.0682716  ,  0.8836329  ,  4.606025   ],
       [ 1.9488412  , -0.67145586 , -0.090105176, -0.6423193  ]],
      dtype=float32), array([[ 3.229639 ,  6.550433 , -1.0597477],
       [ 3.5621257, -4.6919804, -0.9679276]], dtype=float32), array([[1.8615882, 1.2119066, 0.       ],
       [1.2357407, 1.6562097, 0.       ]], dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf32> loc("x")) -> (tensor<2x4x4xf32> {jax.result_info = "result[0]"}, tensor<2x4xf32> {jax.result_info = "result[1]"}, tensor<2x3xf32> {jax.result_info = "result[2]"}, tensor<2x3xf32> {jax.result_info = "result[3]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %0:5 = stablehlo.custom_call @oneapisolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o], [i, p], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=3, p=3}, custom>} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare EQ, %0#4, %1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc6)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %14 = stablehlo.select %13, %0#2, %12 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc6)
    %15 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc6)
    %17 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %18 = stablehlo.select %17, %0#3, %16 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc6)
    return %6, %10, %14, %18 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32> loc(#loc5)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":894:13)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":913:4)
#loc4 = loc("jit(func)"(#loc2))
#loc5 = loc("jit(func)"(#loc3))
#loc6 = loc("tridiagonal"(#loc4))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.16.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#\'+\x03\xeb\x9d7\x01-\x0f\x07\x0f\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x17#\x0b\x0b\x0b\x0f\x17\x03S\x0f\x0b/OOo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x0b\x0bo\x05\x1f\x0f_\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x13\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x033\x17\x1b\x07\x07\x17\x07\x07\x17\x0f\x0f\x07\x13\x17#\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x13\x02B\x07\x1d\x1b\x1d\x1f\x11\x03\x05\x05\x17\x05\x19\x03\x07\r\x0f\x11\x05\x13\x05\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x1d\x19\x03\x05#\x05%\x1d\x07\x1f\x17\t\xfa\r\x1b\x03\x07#Y%_\'\x81\x05\'\x05)\x05+\x1d\x07+\x17\tF\x0e\t\x1f\'\x01\x1d-\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f5!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f!1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03;\r\x01#\x1f\x03\tAEIM\r\x03/C\x1d/\r\x03/G\x1d1\r\x03/K\x1d3\r\x03/O\x1d5\x1d7\x1d9\x1f\x15\t\x00\x00\xc0\x7f\x1f\x17\t\x00\x00\x00\x00\r\x03[]\x1d;\x05\x03\r\x03ac\x1d=\x1d?\x0b\x03\x1dA\x1dC\x03\x01\x05\x01\x03\x037\x03\x03s\x15\x03\x01\x01\x01\x03\x0b7333w\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f11\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x11\t\x11\x11\x11\x11\x11\r\r\x03\x83\x0b\x89\x8f\x93\x97\x9b\x01\x01\x01\x01\x01\x13\x07\x7f\x85\x87\x11\x03\x05\x11\x03\t\x13\x07\x7f\x8b\x8d\x11\x03\r\x11\x03\x11\x13\x05\x7f\x91\x11\x03\x15\x13\x05\x7f\x95\x11\x03\x19\x13\x05\x7f\x99\x11\x03\x1d\x13\x03\x7f\x01\t\x01\x02\x02)\x05\t\r\x0b)\x07\t\x11\x11\x0b\x01\t)\x05\t\x11\x0b\x1d\x13)\x05\t\x05\t)\x01\x0b)\x01\x19\x1b)\x03\t\x19)\x05\t\r\t\x11\x03\x07\t\x07\r\x05\x05)\x03\r\x11)\x03\t\x11)\x03\x05\x11)\x03\x01\x0f)\x03\t\t)\x07\t\x05\x05\t)\x03\x05\x0f)\x07\t\x11\x11\t)\x03\r\x0f)\x05\t\x11\t)\x03\t\x0f\x04J\x03\x05\x01Q\x03\x0b\x01\x07\x04"\x03\x03\x01\x05\tP\x03\x03\x07\x04\xf6\x02\x035[\x03\x0f\x17\x00\x07B\x03\x05\x03\x15\x07B\x01\x07\x03\x17\x0bG\x01!\t\x0b\x07\r\x05\x05\x1b\x03\x01\x03F\x01\x0b\x03\x1b\x03\x05\rF\x01\r\x03)\x05\x0f\x11\x03F\x01\x0f\x03+\x03\x13\x03F\x01\x0b\x03\x07\x03\x03\x03F\x01\x11\x03/\x03\x15\x05\x06\x01\x03\x07\x07\x19\x07\x17\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\r\x03\x03\x03F\x01\x13\x033\x03\x1d\x05\x06\x01\x03\r\x07!\t\x1f\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1d\x03%\x05\x06\x01\x03\x05\x07)\x0b\'\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1d\x03-\x05\x06\x01\x03\x05\x071\r/\x0f\x04)\t\x1b#+3\x06\x03\x01\x05\x01\x00\x86\x07E/\x03\x05\x1f\r\x0f\x0b\x15\x15\x15\x15!%3)\x19\x05\x13%)9w\x15\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jit(func)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00tridiagonal\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00result[3]\x00main\x00public\x00lower\x00num_batch_dims\x001\x00\x00oneapisolver_sytrd_ffi\x00\x08E\x15\x05+\x01\x0b9=?QS\x03U\x03W\x11egikmoqu\x03-\x05y{\x031\x03}\x035',
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["f64"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_sytrd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[[-9.8249281653412324e-01,  2.0908479045899564e+00,
          3.5268637263348204e+00,  3.7326527273633188e+00],
        [-4.0828174542291160e+00,  1.9092664585540786e-01,
          1.1115824919649393e+00,  3.1671933387023712e+00],
        [-2.4793478763923451e+00,  6.5078470564206214e-01,
          1.0365617034248609e+00, -4.1776554703541304e-03],
        [ 3.0883501385983619e-01,  2.7115038066190094e+00,
         -4.5777669803915524e+00, -1.4161887310424071e+00]],

       [[ 4.4593512610250201e+00,  5.1282872404660784e+00,
         -4.7508781771470048e+00, -1.7805546419293350e+00],
        [-3.2939221496875728e+00,  5.0717756858004979e+00,
         -8.4469884182758168e+00, -1.2996964892867967e-02],
        [-7.5191148290550225e-01,  6.2263508577159907e+00,
          4.4320197398015422e+00,  6.7301061150728039e+00],
        [-4.7553708768380254e+00, -5.4695834641483820e+00,
          6.3904774984847919e+00, -3.3443673080409262e+00]]]),),
    expected_outputs=(array([[[-9.8249281653412324e-01,  2.0908479045899564e+00,
          3.5268637263348204e+00,  3.7326527273633188e+00],
        [ 4.7866421761517701e+00,  9.9369220935914793e-01,
          1.1115824919649393e+00,  3.1671933387023712e+00],
        [ 2.7953764713012996e-01,  1.0217935040411712e+00,
         -7.8230379985967446e-01, -4.1776554703541304e-03],
        [-3.4820048427975508e-02,  3.8685594666375472e-02,
          5.3259897467262327e+00, -4.0008879126161156e-01]],

       [[ 4.4593512610250201e+00,  5.1282872404660784e+00,
         -4.7508781771470048e+00, -1.7805546419293350e+00],
        [ 5.8334249101712796e+00, -3.3178080744448035e+00,
         -8.4469884182758168e+00, -1.2996964892867967e-02],
        [ 8.2380069254990071e-02,  1.0047709963384454e+01,
          2.7865568175833912e+00,  6.7301061150728039e+00],
        [ 5.2100252632571242e-01, -1.4477866846865198e-01,
          2.2520647542187247e+00,  6.6906793744225270e+00]]]), array([[-0.9824928165341232 ,  0.9936922093591479 , -0.7823037998596745 ,
        -0.40008879126161156],
       [ 4.45935126102502   , -3.3178080744448035 ,  2.7865568175833912 ,
         6.690679374422527  ]]), array([[ 4.78664217615177  ,  1.0217935040411712,  5.325989746726233 ],
       [ 5.83342491017128  , 10.047709963384454 ,  2.2520647542187247]]), array([[1.852960656756571 , 1.9970113223116306, 0.                ],
       [1.564663503929608 , 1.9589389493634926, 0.                ]])),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf64> loc("x")) -> (tensor<2x4x4xf64> {jax.result_info = "result[0]"}, tensor<2x4xf64> {jax.result_info = "result[1]"}, tensor<2x3xf64> {jax.result_info = "result[2]"}, tensor<2x3xf64> {jax.result_info = "result[3]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %0:5 = stablehlo.custom_call @oneapisolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o], [i, p], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=3, p=3}, custom>} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xf64>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare EQ, %0#4, %1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc6)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %14 = stablehlo.select %13, %0#2, %12 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc6)
    %15 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc6)
    %17 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %18 = stablehlo.select %17, %0#3, %16 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc6)
    return %6, %10, %14, %18 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xf64> loc(#loc5)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":894:13)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":913:4)
#loc4 = loc("jit(func)"(#loc2))
#loc5 = loc("jit(func)"(#loc3))
#loc6 = loc("tridiagonal"(#loc4))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.16.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#\'+\x03\xeb\x9d7\x01-\x0f\x07\x0f\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x17#\x0b\x0b\x0b\x0f\x17\x03S\x0f\x0b/OOo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x0b\x0bo\x05\x1f\x0f_\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x13\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x033\x17\x1b\x07\x07\x17\x07\x07\x17\x0f\x0f\x07\x13\x17#\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x13\x02R\x07\x1d\x1b\x1d\x1f\x11\x03\x05\x05\x17\x05\x19\x03\x07\r\x0f\x11\x05\x13\x05\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x1d\x19\x03\x05#\x05%\x1d\x07\x1f\x17\t\xfa\r\x1b\x03\x07#Y%_\'\x81\x05\'\x05)\x05+\x1d\x07+\x17\tF\x0e\t\x1f\'\x01\x1d-\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f5!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f!1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03;\r\x01#\x1f\x03\tAEIM\r\x03/C\x1d/\r\x03/G\x1d1\r\x03/K\x1d3\r\x03/O\x1d5\x1d7\x1d9\x1f\x15\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x17\t\x00\x00\x00\x00\r\x03[]\x1d;\x05\x03\r\x03ac\x1d=\x1d?\x0b\x03\x1dA\x1dC\x03\x01\x05\x01\x03\x037\x03\x03s\x15\x03\x01\x01\x01\x03\x0b7333w\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f11\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x11\t\x11\x11\x11\x11\x11\r\r\x03\x83\x0b\x89\x8f\x93\x97\x9b\x01\x01\x01\x01\x01\x13\x07\x7f\x85\x87\x11\x03\x05\x11\x03\t\x13\x07\x7f\x8b\x8d\x11\x03\r\x11\x03\x11\x13\x05\x7f\x91\x11\x03\x15\x13\x05\x7f\x95\x11\x03\x19\x13\x05\x7f\x99\x11\x03\x1d\x13\x03\x7f\x01\t\x01\x02\x02)\x05\t\r\x0b)\x07\t\x11\x11\x0b\x01\x0b)\x05\t\x11\x0b\x1d\x13)\x05\t\x05\t)\x01\x0b)\x01\x19\x1b)\x03\t\x19)\x05\t\r\t\x11\x03\x07\t\x07\r\x05\x05)\x03\r\x11)\x03\t\x11)\x03\x05\x11)\x03\x01\x0f)\x03\t\t)\x07\t\x05\x05\t)\x03\x05\x0f)\x07\t\x11\x11\t)\x03\r\x0f)\x05\t\x11\t)\x03\t\x0f\x04J\x03\x05\x01Q\x03\x0b\x01\x07\x04"\x03\x03\x01\x05\tP\x03\x03\x07\x04\xf6\x02\x035[\x03\x0f\x17\x00\x07B\x03\x05\x03\x15\x07B\x01\x07\x03\x17\x0bG\x01!\t\x0b\x07\r\x05\x05\x1b\x03\x01\x03F\x01\x0b\x03\x1b\x03\x05\rF\x01\r\x03)\x05\x0f\x11\x03F\x01\x0f\x03+\x03\x13\x03F\x01\x0b\x03\x07\x03\x03\x03F\x01\x11\x03/\x03\x15\x05\x06\x01\x03\x07\x07\x19\x07\x17\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\r\x03\x03\x03F\x01\x13\x033\x03\x1d\x05\x06\x01\x03\r\x07!\t\x1f\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1d\x03%\x05\x06\x01\x03\x05\x07)\x0b\'\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1d\x03-\x05\x06\x01\x03\x05\x071\r/\x0f\x04)\t\x1b#+3\x06\x03\x01\x05\x01\x00\x86\x07E/\x03\x05\x1f\r\x0f\x0b\x15\x15\x15\x15!%3)\x19\x05\x13%)9w\x15\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jit(func)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00tridiagonal\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00result[3]\x00main\x00public\x00lower\x00num_batch_dims\x001\x00\x00oneapisolver_sytrd_ffi\x00\x08E\x15\x05+\x01\x0b9=?QS\x03U\x03W\x11egikmoqu\x03-\x05y{\x031\x03}\x035',
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["c64"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_sytrd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[[-1.1872267 +3.406419j   , -0.31528282-0.3315316j  ,
         -3.4297762 -3.257631j   , -1.139394  -0.019099746j],
        [-2.3842719 +1.3329202j  , -1.5013279 +0.28686357j ,
          1.3731898 -1.0257013j  , -0.8473716 -4.000408j   ],
        [ 0.17507061-0.2298106j  , -3.9399796 +2.8584316j  ,
          1.9746994 +2.2335658j  ,  4.2399807 +0.7693164j  ],
        [-4.315057  +5.2411947j  ,  2.5208282 +4.4563894j  ,
         -1.139267  +0.16494599j ,  2.110906  -2.2521057j  ]],

       [[ 1.0215222 +2.6285636j  , -2.4712446 -1.1273237j  ,
          2.2939186 +1.1271836j  ,  2.1202447 +0.52373666j ],
        [-0.23852621+3.5717723j  , -0.5704699 +0.062138904j,
         -2.2698088 -2.3174975j  , -3.0348637 -0.21346329j ],
        [ 0.7951647 -0.90768296j , -0.14100291+3.9679756j  ,
         -1.1112772 -1.8822902j  ,  1.4533287 -0.21419309j ],
        [ 5.8733315 +5.3228903j  ,  3.1500263 -0.3376814j  ,
         -1.2769403 -2.7346413j  ,  1.7102162 +0.25563255j ]]],
      dtype=complex64),),
    expected_outputs=(array([[[-1.1872267  +0.j         , -0.31528282 -0.3315316j  ,
         -3.4297762  -3.257631j   , -1.139394   -0.019099746j],
        [ 7.3235736  +0.j         ,  2.3338084  +0.j         ,
          1.3731898  -1.0257013j  , -0.8473716  -4.000408j   ],
        [-0.020890437+0.020804338j, -5.786792   +0.j         ,
         -1.9368124  +0.j         ,  4.2399807  +0.7693164j  ],
        [ 0.50902456 -0.47000188j , -0.50863796 -0.391739j   ,
          4.0074725  +0.j         ,  2.1872826  +0.j         ]],

       [[ 1.0215222  +0.j         , -2.4712446  -1.1273237j  ,
          2.2939186  +1.1271836j  ,  2.1202447  +0.52373666j ],
        [ 8.78065    +0.j         ,  1.9812212  +0.j         ,
         -2.2698088  -2.3174975j  , -3.0348637  -0.21346329j ],
        [-0.110663384+0.05681434j ,  5.010742   +0.j         ,
          1.206828   +0.j         ,  1.4533287  -0.21419309j ],
        [-0.36088568 -0.7330926j  ,  0.08949702 -0.25109226j ,
          2.0887785  +0.j         , -3.1595817  +0.j         ]]],
      dtype=complex64), array([[-1.1872267,  2.3338084, -1.9368124,  2.1872826],
       [ 1.0215222,  1.9812212,  1.206828 , -3.1595817]], dtype=float32), array([[ 7.3235736, -5.786792 ,  4.0074725],
       [ 8.78065  ,  5.010742 ,  2.0887785]], dtype=float32), array([[1.3255613-0.18200406j, 1.2902822-0.40316778j,
        1.1139076-0.9934914j ],
       [1.027165 -0.40677765j, 1.8358151-0.24047153j,
        1.9767734-0.21427572j]], dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f32>> loc("x")) -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<2x4xf32> {jax.result_info = "result[1]"}, tensor<2x3xf32> {jax.result_info = "result[2]"}, tensor<2x3xcomplex<f32>> {jax.result_info = "result[3]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %0:5 = stablehlo.custom_call @oneapisolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o], [i, p], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=3, p=3}, custom>} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xcomplex<f32>>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare EQ, %0#4, %1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc6)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %14 = stablehlo.select %13, %0#2, %12 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc6)
    %15 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<2x3xcomplex<f32>> loc(#loc6)
    %17 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %18 = stablehlo.select %17, %0#3, %16 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>> loc(#loc6)
    return %6, %10, %14, %18 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xcomplex<f32>> loc(#loc5)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":894:13)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":913:4)
#loc4 = loc("jit(func)"(#loc2))
#loc5 = loc("jit(func)"(#loc3))
#loc6 = loc("tridiagonal"(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xf3\x9f=\x01-\x0f\x07\x0f\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x17#\x0b\x0b\x0b\x0f\x17\x03U\x0f\x0b/OOo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f/\x1f\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x0b\x0bo\x05\x1f\x0f_\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x13\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x039\x1b\x07\x07\x17\x17\x17\x07\x0b\x07\x17\x0f\x0f\x0f\x07\x13\x17#\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x13\x02\x96\x07\x1d\x1b\x1d\x1f\x11\x03\x05\x05\x17\x05\x19\x03\x07\r\x0f\x11\x05\x13\x05\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x1d\x19\x03\x05#\x05%\x1d\x07\x1f\x17\t\xfa\r\x1b\x03\x07#[%a'\x83\x05'\x05)\x05+\x1d\x07+\x17\tF\x0e\t\x1f-\x01\x1d-\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f;!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03;\r\x01#%\x03\tAEIM\r\x03/C\x1d/\r\x03/G\x1d1\r\x03/K\x1d3\r\x03/O\x1d5\x1d7\x1d9\x1f\x19\t\x00\x00\xc0\x7f\x1f\x1b\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x1d\t\x00\x00\x00\x00\r\x03]_\x1d;\x05\x03\r\x03ce\x1d=\x1d?\x0b\x03\x1dA\x1dC\x03\x01\x05\x01\x03\x037\x03\x03u\x15\x03\x01\x01\x01\x03\x0b7333y\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f71\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x11\t\x11\x11\x11\x11\x11\r\r\x03\x85\x0b\x8b\x91\x95\x99\x9d\x01\x01\x01\x01\x01\x13\x07\x81\x87\x89\x11\x03\x05\x11\x03\t\x13\x07\x81\x8d\x8f\x11\x03\r\x11\x03\x11\x13\x05\x81\x93\x11\x03\x15\x13\x05\x81\x97\x11\x03\x19\x13\x05\x81\x9b\x11\x03\x1d\x13\x03\x81\x01\t\x01\x02\x02)\x07\t\x11\x11\x13\x01\t)\x05\t\x11\t)\x05\t\r\t)\x05\t\r\x13\x1d\x03\t\x13)\x05\t\x05\x07)\x01\t)\x01\x13)\x01\x1f\x1b)\x03\t\x1f)\x05\t\r\x07\x11\x03\x05\t\x05\x0b\r\x0f)\x03\r\x15)\x03\t\x15)\x03\x05\x15)\x03\x01\x11)\x03\t\x07)\x07\t\x05\x05\x07)\x03\x05\x11)\x07\t\x11\x11\x07)\x03\r\x11)\x05\t\x11\x07)\x03\t\x11\x04b\x03\x05\x01Q\x03\x0b\x01\x07\x04:\x03\x03\x01\x05\tP\x03\x03\x07\x04\x0e\x03\x037_\x03\x0b\x17\x00\x07B\x03\x05\x03\x19\x07B\x03\x07\x03\x1b\x07B\x01\t\x03\x1d\x0bG\x01!\x0b\x0b\x05\x0b\r\x0f!\x03\x01\x03F\x01\r\x03!\x03\x07\rF\x01\x0f\x03/\x05\x11\x13\x03F\x01\x11\x031\x03\x15\x03F\x01\r\x03\x05\x03\x05\x03F\x01\x13\x035\x03\x17\x05\x06\x01\x03\x05\x07\x1b\t\x19\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\x0b\x03\x03\x03F\x01\x15\x039\x03\x1f\x05\x06\x01\x03\x0b\x07#\x0b!\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\r\x03\x03\x03F\x01\x15\x03#\x03'\x05\x06\x01\x03\r\x07+\r)\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\x0f\x03\x05\x03F\x01\x15\x03#\x03/\x05\x06\x01\x03\x0f\x073\x0f1\x0f\x04)\t\x1d%-5\x06\x03\x01\x05\x01\x00\x86\x07E/\x03\x05\x1f\r\x0f\x0b\x15\x15\x15\x15!%3)\x19\x05\x13%)9w\x15\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jit(func)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00tridiagonal\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00result[3]\x00main\x00public\x00lower\x00num_batch_dims\x001\x00\x00oneapisolver_sytrd_ffi\x00\x08I\x17\x05+\x01\x0b9=?QS\x03U\x03W\x03Y\x11gikmoqsw\x03-\x05{}\x031\x03\x7f\x035",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["c128"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_sytrd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[[ 0.6018904705299121  -0.5497093029193996j ,
          2.5307622986194938  +1.910047197671941j  ,
          2.324562155640359   +2.7527988011950795j ,
         -1.7110864291883463  +1.8769443140590116j ],
        [ 3.8914252512785916  +2.2850800963692968j ,
          1.779534729604491   +5.067828414693769j  ,
         -0.7728274318557047  -3.019016412243171j  ,
          1.3565994446692629  -0.6211839574000335j ],
        [ 0.6309352618353146  +3.7537886598914074j ,
          0.42837406303951825 -0.4388018251250955j ,
         -0.34495591929905434 +1.7246197578977585j ,
         -2.7153872361619014  +0.7086247161116597j ],
        [-2.38912679620689    -0.7131113874238367j ,
         -5.2550556891357605  +1.4195599236451557j ,
          1.0419122658855056  +0.7742296889704131j ,
         -3.92341035897585    +1.728771291235352j  ]],

       [[ 0.00768216134973277 +1.8249946467261453j ,
         -1.2963676854897774  -7.446944957726921j  ,
         -1.7984687542021258  +1.4904520590710915j ,
          0.37922008330133106 -4.64530411147067j   ],
        [ 1.3331914538399445  +0.8795821990584436j ,
         -3.2910829882045576  -1.637567440624402j  ,
         -1.3802731622857618  -2.005903707345281j  ,
         -0.012647520739661484-0.8912118456313312j ],
        [ 2.070142707714537   +3.946499095342482j  ,
         -0.47099438651553904 -1.4950105016481j    ,
         -3.4840228083763174  +2.986036160547841j  ,
          1.7302362320526177  +2.3088886020996635j ],
        [-0.8030537188026532  -5.230171821667378j  ,
         -1.1996624697009048  +0.14058181398760203j,
         -0.8697322455164035  +1.5099872857618921j ,
          2.473316027643908   +1.0719315528593527j ]]]),),
    expected_outputs=(array([[[ 0.6018904705299121  +0.j                  ,
          2.5307622986194938  +1.910047197671941j   ,
          2.324562155640359   +2.7527988011950795j  ,
         -1.7110864291883463  +1.8769443140590116j  ],
        [-6.4086070904757175  +0.j                  ,
          3.2062715816754666  +0.j                  ,
         -0.7728274318557047  -3.019016412243171j   ,
          1.3565994446692629  -0.6211839574000335j  ],
        [ 0.1354420593245469  +0.33439630009268306j ,
         -2.7802340556215634  +0.j                  ,
         -0.41863499023718154 +0.j                  ,
         -2.7153872361619014  +0.7086247161116597j  ],
        [-0.23571168662488354 -0.016940850091152124j,
          0.13967347984176515 -0.3434946248928184j  ,
         -3.7620862164337416  +0.j                  ,
         -5.276468140108695   +0.j                  ]],

       [[ 0.00768216134973277 +0.j                  ,
         -1.2963676854897774  -7.446944957726921j   ,
         -1.7984687542021258  +1.4904520590710915j  ,
          0.37922008330133106 -4.64530411147067j    ],
        [-7.100070619140439   +0.j                  ,
          0.05988219021905561 +0.j                  ,
         -1.3802731622857618  -2.005903707345281j   ,
         -0.012647520739661484-0.8912118456313312j  ],
        [ 0.29111539683466664 +0.43760518082780786j ,
         -3.9107970653733686  +0.j                  ,
         -2.339528603279976   +0.j                  ,
          1.7302362320526177  +2.3088886020996635j  ],
        [-0.15818838684985748 -0.6036848005480352j  ,
          0.5268809024509943  +0.44065376488721747j ,
         -1.1909881640456055  +0.j                  ,
         -2.022143355876045   +0.j                  ]]]), array([[ 0.6018904705299121 ,  3.2062715816754666 , -0.41863499023718154,
        -5.276468140108695  ],
       [ 0.00768216134973277,  0.05988219021905561, -2.339528603279976  ,
        -2.022143355876045  ]]), array([[-6.4086070904757175, -2.7802340556215634, -3.7620862164337416],
       [-7.100070619140439 , -3.9107970653733686, -1.1909881640456055]]), array([[1.6072185728255854+0.35656423683163774j,
        1.0735149399379793+0.8573616808747281j ,
        1.682728618917841 -0.7306720419658448j ],
       [1.1877715765595225+0.12388358457833608j,
        1.3085165729840078+0.25676223797262654j,
        1.6495715772982202+0.7603004445390666j ]])),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f64>> loc("x")) -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<2x4xf64> {jax.result_info = "result[1]"}, tensor<2x3xf64> {jax.result_info = "result[2]"}, tensor<2x3xcomplex<f64>> {jax.result_info = "result[3]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %0:5 = stablehlo.custom_call @oneapisolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o], [i, p], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=3, p=3}, custom>} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xcomplex<f64>>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare EQ, %0#4, %1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc6)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %14 = stablehlo.select %13, %0#2, %12 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc6)
    %15 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<2x3xcomplex<f64>> loc(#loc6)
    %17 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %18 = stablehlo.select %17, %0#3, %16 : tensor<2x3xi1>, tensor<2x3xcomplex<f64>> loc(#loc6)
    return %6, %10, %14, %18 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xcomplex<f64>> loc(#loc5)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":894:13)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":913:4)
#loc4 = loc("jit(func)"(#loc2))
#loc5 = loc("jit(func)"(#loc3))
#loc6 = loc("tridiagonal"(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xf3\x9f=\x01-\x0f\x07\x0f\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x17#\x0b\x0b\x0b\x0f\x17\x03U\x0f\x0b/OOo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/O\x1f\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x0b\x0bo\x05\x1f\x0f_\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x13\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x039\x1b\x07\x07\x17\x17\x17\x07\x0b\x07\x17\x0f\x0f\x0f\x07\x13\x17#\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x13\x02\xc6\x07\x1d\x1b\x1d\x1f\x11\x03\x05\x05\x17\x05\x19\x03\x07\r\x0f\x11\x05\x13\x05\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x1d\x19\x03\x05#\x05%\x1d\x07\x1f\x17\t\xfa\r\x1b\x03\x07#[%a'\x83\x05'\x05)\x05+\x1d\x07+\x17\tF\x0e\t\x1f-\x01\x1d-\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f;!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03;\r\x01#%\x03\tAEIM\r\x03/C\x1d/\r\x03/G\x1d1\r\x03/K\x1d3\r\x03/O\x1d5\x1d7\x1d9\x1f\x19\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1b!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1d\t\x00\x00\x00\x00\r\x03]_\x1d;\x05\x03\r\x03ce\x1d=\x1d?\x0b\x03\x1dA\x1dC\x03\x01\x05\x01\x03\x037\x03\x03u\x15\x03\x01\x01\x01\x03\x0b7333y\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f71\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x11\t\x11\x11\x11\x11\x11\r\r\x03\x85\x0b\x8b\x91\x95\x99\x9d\x01\x01\x01\x01\x01\x13\x07\x81\x87\x89\x11\x03\x05\x11\x03\t\x13\x07\x81\x8d\x8f\x11\x03\r\x11\x03\x11\x13\x05\x81\x93\x11\x03\x15\x13\x05\x81\x97\x11\x03\x19\x13\x05\x81\x9b\x11\x03\x1d\x13\x03\x81\x01\t\x01\x02\x02)\x07\t\x11\x11\x13\x01\x0b)\x05\t\x11\t)\x05\t\r\t)\x05\t\r\x13\x1d\x03\t\x13)\x05\t\x05\x07)\x01\t)\x01\x13)\x01\x1f\x1b)\x03\t\x1f)\x05\t\r\x07\x11\x03\x05\t\x05\x0b\r\x0f)\x03\r\x15)\x03\t\x15)\x03\x05\x15)\x03\x01\x11)\x03\t\x07)\x07\t\x05\x05\x07)\x03\x05\x11)\x07\t\x11\x11\x07)\x03\r\x11)\x05\t\x11\x07)\x03\t\x11\x04b\x03\x05\x01Q\x03\x0b\x01\x07\x04:\x03\x03\x01\x05\tP\x03\x03\x07\x04\x0e\x03\x037_\x03\x0b\x17\x00\x07B\x03\x05\x03\x19\x07B\x03\x07\x03\x1b\x07B\x01\t\x03\x1d\x0bG\x01!\x0b\x0b\x05\x0b\r\x0f!\x03\x01\x03F\x01\r\x03!\x03\x07\rF\x01\x0f\x03/\x05\x11\x13\x03F\x01\x11\x031\x03\x15\x03F\x01\r\x03\x05\x03\x05\x03F\x01\x13\x035\x03\x17\x05\x06\x01\x03\x05\x07\x1b\t\x19\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\x0b\x03\x03\x03F\x01\x15\x039\x03\x1f\x05\x06\x01\x03\x0b\x07#\x0b!\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\r\x03\x03\x03F\x01\x15\x03#\x03'\x05\x06\x01\x03\r\x07+\r)\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\x0f\x03\x05\x03F\x01\x15\x03#\x03/\x05\x06\x01\x03\x0f\x073\x0f1\x0f\x04)\t\x1d%-5\x06\x03\x01\x05\x01\x00\x86\x07E/\x03\x05\x1f\r\x0f\x0b\x15\x15\x15\x15!%3)\x19\x05\x13%)9w\x15\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jit(func)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00tridiagonal\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00result[3]\x00main\x00public\x00lower\x00num_batch_dims\x001\x00\x00oneapisolver_sytrd_ffi\x00\x08I\x17\x05+\x01\x0b9=?QS\x03U\x03W\x03Y\x11gikmoqsw\x03-\x05{}\x031\x03\x7f\x035",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
