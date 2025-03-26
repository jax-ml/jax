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
from numpy import array, float32, complex64


data_2024_09_26 = {}
data_2024_09_26["f32"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_geqrf_ffi', 'cusolver_orgqr_ffi'],
    serialized_date=datetime.date(2024, 9, 26),
    inputs=(),
    expected_outputs=(array([[[ 0.        ,  0.91287094,  0.40824842],
        [-0.4472136 ,  0.36514843, -0.8164965 ],
        [-0.8944272 , -0.18257415,  0.40824836]],

       [[-0.42426407,  0.80828977,  0.4082495 ],
        [-0.5656854 ,  0.11547142, -0.8164964 ],
        [-0.7071068 , -0.57735085,  0.4082474 ]]], dtype=float32), array([[[-6.7082038e+00, -8.0498447e+00, -9.3914852e+00],
        [ 0.0000000e+00,  1.0954450e+00,  2.1908898e+00],
        [ 0.0000000e+00,  0.0000000e+00,  4.8374091e-08]],

       [[-2.1213203e+01, -2.2910259e+01, -2.4607319e+01],
        [ 0.0000000e+00,  3.4641042e-01,  6.9282258e-01],
        [ 0.0000000e+00,  0.0000000e+00,  1.4548683e-06]]], dtype=float32)),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3x3xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xf32> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<18xf32>) -> tensor<2x3x3xf32> loc(#loc5)
    %2:2 = stablehlo.custom_call @cusolver_geqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x3x3xf32>) -> (tensor<2x3x3xf32>, tensor<2x3xf32>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xf32>, tensor<f32>) -> tensor<2x3x3xf32> loc(#loc7)
    %4 = stablehlo.custom_call @cusolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>]} : (tensor<2x3x3xf32>, tensor<2x3xf32>) -> tensor<2x3x3xf32> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc12)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3x3xf32> loc(#loc12)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xf32> loc(#loc13)
    return %4, %12 : tensor<2x3x3xf32>, tensor<2x3x3xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":391:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/pad"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/iota"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/add"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/ge"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/broadcast_in_dim"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#'+/37\x03\xc7\x89+\x01C\x17\x07\x0b\x0f\x0b\x13\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x03G\x0b/\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x1f\x1f\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x01\x05\x0b\x0f\x03'\x1b\x07\x07\x17\x0f\x07\x0f\x07\x07\x17\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02F\x05\x17\x05\x1e\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x03)q\x1d\t\x01\x1d7\x01\x1d=\x01\x03\x07\x15\x17\x19\x07\x1b\x07\x05!\x11\x01\x00\x05#\x05%\x05'\x1d\t!\x17\x05\x1a\x065\x1d%'\x05)\x17\x05\x1a\x06\x1d\x05+\x1d-\x01\x05-\x1d1\x01\x05/\x1d5\x01\x051\x053\x1d;\x01\x055\x057\x1dA\x01\x059\x03\x01\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d;\x1d=\x1d?\x13\x07\x01\x0b\x03\x1dA\x05\x01\x03\x03W\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Wy\x1f#\x01#\x17\x03\x05ae\r\x05GcIK\x1dC\r\x05GgIK\x1dE\x1dG\x1dI\x1f\r\t\xff\xff\xff\xff\x1f\x11\t\x00\x00\x00\x00\r\x01\x1dK\x03\x03w\x15\x03\x01\x01\x01\x1f\x1f!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dM\x03\x03\x7f\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\t)\x05\r\r\x0f)\x01\x0f\x1b)\x01\t\x13\x01\x11\x01\x05\x05\x05)\x03I\t)\x05\t\r\t)\x03\r\x13)\x03\t\x13)\x03\r\x07)\x03\x01\x07)\x05\r\r\x15)\x07\t\r\r\x15)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x03\x05\x03\r\x07B\x03\x07\x03\x11\x03B\x1f\t\x03\x19\r\x06#\x03\x05\x03\x05\tG+\x0b\x0b\x05\x05\x1b\x03\x07\x0fF/\r\x03\x05\x05\t\x03\tG3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\x05F\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03%\x05\x15\x17\x05F\x11\x17\x03'\x03\x19\x05F\x11\x11\x03\x05\x03\x03\x15\x06?\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00J\x0bO''\x0f\x0b\t\t\x03\x11#!CS79Y9=)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00main\x00public\x00cusolver_geqrf_ffi\x00cusolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0bC]_ik\x03m\x03o\x03M\x11OQsCSUuY\x07EEE\x11OQ{CSY}U\x03[\x03\x81\x05\x83\x85\x03\x87",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_09_26["f64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_geqrf_ffi', 'cusolver_orgqr_ffi'],
    serialized_date=datetime.date(2024, 9, 26),
    inputs=(),
    expected_outputs=(array([[[ 0.                 ,  0.9128709291752773 ,
          0.408248290463862  ],
        [-0.447213595499958  ,  0.36514837167011005,
         -0.8164965809277264 ],
        [-0.894427190999916  , -0.1825741858350547 ,
          0.40824829046386335]],

       [[-0.42426406871192857,  0.8082903768654768 ,
          0.40824829046386124],
        [-0.565685424949238  ,  0.11547005383792364,
         -0.8164965809277263 ],
        [-0.7071067811865476 , -0.577350269189625  ,
          0.4082482904638641 ]]]), array([[[-6.7082039324993694e+00, -8.0498447189992444e+00,
         -9.3914855054991175e+00],
        [ 0.0000000000000000e+00,  1.0954451150103344e+00,
          2.1908902300206661e+00],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
         -1.7577018578317312e-15]],

       [[-2.1213203435596427e+01, -2.2910259710444144e+01,
         -2.4607315985291855e+01],
        [ 0.0000000000000000e+00,  3.4641016151377924e-01,
          6.9282032302755281e-01],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
         -1.8103038069914667e-15]]])),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3x3xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xf64> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<18xf64>) -> tensor<2x3x3xf64> loc(#loc5)
    %2:2 = stablehlo.custom_call @cusolver_geqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x3x3xf64>) -> (tensor<2x3x3xf64>, tensor<2x3xf64>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xf64>, tensor<f64>) -> tensor<2x3x3xf64> loc(#loc7)
    %4 = stablehlo.custom_call @cusolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>]} : (tensor<2x3x3xf64>, tensor<2x3xf64>) -> tensor<2x3x3xf64> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc12)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3x3xf64> loc(#loc12)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xf64> loc(#loc13)
    return %4, %12 : tensor<2x3x3xf64>, tensor<2x3x3xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":391:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/pad"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/iota"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/add"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/ge"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/broadcast_in_dim"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#'+/37\x03\xc7\x89+\x01C\x17\x07\x0b\x0f\x0b\x13\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x03G\x0b/\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x1f/\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x01\x05\x0b\x0f\x03'\x1b\x07\x07\x17\x0f\x07\x0f\x07\x07\x17\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02V\x05\x17\x05\x1e\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x03)q\x1d\t\x01\x1d7\x01\x1d=\x01\x03\x07\x15\x17\x19\x07\x1b\x07\x05!\x11\x01\x00\x05#\x05%\x05'\x1d\t!\x17\x05\x1a\x065\x1d%'\x05)\x17\x05\x1a\x06\x1d\x05+\x1d-\x01\x05-\x1d1\x01\x05/\x1d5\x01\x051\x053\x1d;\x01\x055\x057\x1dA\x01\x059\x03\x01\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d;\x1d=\x1d?\x13\x07\x01\x0b\x03\x1dA\x05\x01\x03\x03W\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Wy\x1f#\x01#\x17\x03\x05ae\r\x05GcIK\x1dC\r\x05GgIK\x1dE\x1dG\x1dI\x1f\r\t\xff\xff\xff\xff\x1f\x11\x11\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\x1dK\x03\x03w\x15\x03\x01\x01\x01\x1f\x1f!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dM\x03\x03\x7f\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x0b)\x05\r\r\x0f)\x01\x0f\x1b)\x01\t\x13\x01\x11\x01\x05\x05\x05)\x03I\t)\x05\t\r\t)\x03\r\x13)\x03\t\x13)\x03\r\x07)\x03\x01\x07)\x05\r\r\x15)\x07\t\r\r\x15)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x03\x05\x03\r\x07B\x03\x07\x03\x11\x03B\x1f\t\x03\x19\r\x06#\x03\x05\x03\x05\tG+\x0b\x0b\x05\x05\x1b\x03\x07\x0fF/\r\x03\x05\x05\t\x03\tG3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\x05F\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03%\x05\x15\x17\x05F\x11\x17\x03'\x03\x19\x05F\x11\x11\x03\x05\x03\x03\x15\x06?\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00J\x0bO''\x0f\x0b\t\t\x03\x11#!CS79Y9=)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00main\x00public\x00cusolver_geqrf_ffi\x00cusolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0bC]_ik\x03m\x03o\x03M\x11OQsCSUuY\x07EEE\x11OQ{CSY}U\x03[\x03\x81\x05\x83\x85\x03\x87",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_09_26["c64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_geqrf_ffi', 'cusolver_orgqr_ffi'],
    serialized_date=datetime.date(2024, 9, 26),
    inputs=(),
    expected_outputs=(array([[[ 0.        +0.j,  0.91287094+0.j,  0.40824836+0.j],
        [-0.4472136 +0.j,  0.36514843+0.j, -0.81649655+0.j],
        [-0.8944272 +0.j, -0.18257417+0.j,  0.4082483 +0.j]],

       [[-0.42426407+0.j,  0.8082899 +0.j,  0.40824962+0.j],
        [-0.5656854 +0.j,  0.11547136+0.j, -0.8164964 +0.j],
        [-0.7071068 +0.j, -0.57735085+0.j,  0.4082474 +0.j]]],
      dtype=complex64), array([[[-6.7082038e+00+0.j, -8.0498447e+00+0.j, -9.3914852e+00+0.j],
        [ 0.0000000e+00+0.j,  1.0954450e+00+0.j,  2.1908898e+00+0.j],
        [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  4.8374091e-08+0.j]],

       [[-2.1213203e+01+0.j, -2.2910259e+01+0.j, -2.4607319e+01+0.j],
        [ 0.0000000e+00+0.j,  3.4641042e-01+0.j,  6.9282258e-01+0.j],
        [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  1.5032538e-06+0.j]]],
      dtype=complex64)),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3x3xcomplex<f32>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xcomplex<f32>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<18xcomplex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc5)
    %2:2 = stablehlo.custom_call @cusolver_geqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x3x3xcomplex<f32>>) -> (tensor<2x3x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc7)
    %4 = stablehlo.custom_call @cusolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>]} : (tensor<2x3x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc12)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc12)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xcomplex<f32>> loc(#loc13)
    return %4, %12 : tensor<2x3x3xcomplex<f32>>, tensor<2x3x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":391:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/pad"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/iota"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/add"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/ge"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/broadcast_in_dim"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#'+/37\x03\xc9\x89-\x01C\x17\x07\x0b\x0f\x0b\x13\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x03G\x0b/\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x1f/\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x01\x05\x0b\x0f\x03)\x1b\x07\x0b\x17\x0f\x07\x0f\x07\x07\x17\x07\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02^\x05\x17\x05\x1e\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x03)q\x1d\t\x01\x1d7\x01\x1d=\x01\x03\x07\x15\x17\x19\x07\x1b\x07\x05!\x11\x01\x00\x05#\x05%\x05'\x1d\t!\x17\x05\x1a\x065\x1d%'\x05)\x17\x05\x1a\x06\x1d\x05+\x1d-\x01\x05-\x1d1\x01\x05/\x1d5\x01\x051\x053\x1d;\x01\x055\x057\x1dA\x01\x059\x03\x01\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d;\x1d=\x1d?\x13\x07\x01\x0b\x03\x1dA\x05\x01\x03\x03W\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Wy\x1f%\x01#\x17\x03\x05ae\r\x05GcIK\x1dC\r\x05GgIK\x1dE\x1dG\x1dI\x1f\r\t\xff\xff\xff\xff\x1f\x11\x11\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\x1dK\x03\x03w\x15\x03\x01\x01\x01\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dM\x03\x03\x7f\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f+!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x03\x19)\x05\r\r\x0f)\x01\x0f\x1b)\x01\t\x13\x01\x11\x01\x05\x05\x05\t)\x03I\t)\x05\t\r\t)\x03\r\x13)\x03\t\x13)\x03\r\x07)\x03\x01\x07)\x05\r\r\x15)\x07\t\r\r\x15)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x03\x05\x03\r\x07B\x03\x07\x03\x11\x03B\x1f\t\x03\x1b\r\x06#\x03\x05\x03\x05\tG+\x0b\x0b\x05\x05\x1d\x03\x07\x0fF/\r\x03\x05\x05\t\x03\tG3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\x05F\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03'\x05\x15\x17\x05F\x11\x17\x03)\x03\x19\x05F\x11\x11\x03\x05\x03\x03\x15\x06?\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00J\x0bO''\x0f\x0b\t\t\x03\x11#!CS79Y9=)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00main\x00public\x00cusolver_geqrf_ffi\x00cusolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0bC]_ik\x03m\x03o\x03M\x11OQsCSUuY\x07EEE\x11OQ{CSY}U\x03[\x03\x81\x05\x83\x85\x03\x87",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_09_26["c128"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_geqrf_ffi', 'cusolver_orgqr_ffi'],
    serialized_date=datetime.date(2024, 9, 26),
    inputs=(),
    expected_outputs=(array([[[ 0.                 +0.j,  0.9128709291752773 +0.j,
          0.4082482904638621 +0.j],
        [-0.447213595499958  +0.j,  0.36514837167011005+0.j,
         -0.8164965809277263 +0.j],
        [-0.894427190999916  +0.j, -0.18257418583505472+0.j,
          0.40824829046386335+0.j]],

       [[-0.42426406871192857+0.j,  0.808290376865477  +0.j,
          0.4082482904638615 +0.j],
        [-0.565685424949238  +0.j,  0.11547005383792353+0.j,
         -0.8164965809277263 +0.j],
        [-0.7071067811865476 +0.j, -0.577350269189625  +0.j,
          0.4082482904638641 +0.j]]]), array([[[-6.7082039324993694e+00+0.j, -8.0498447189992444e+00+0.j,
         -9.3914855054991175e+00+0.j],
        [ 0.0000000000000000e+00+0.j,  1.0954451150103344e+00+0.j,
          2.1908902300206661e+00+0.j],
        [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
         -1.7577018578317312e-15+0.j]],

       [[-2.1213203435596427e+01+0.j, -2.2910259710444144e+01+0.j,
         -2.4607315985291855e+01+0.j],
        [ 0.0000000000000000e+00+0.j,  3.4641016151377924e-01+0.j,
          6.9282032302755292e-01+0.j],
        [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
         -1.7201790115224914e-15+0.j]]])),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3x3xcomplex<f64>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xcomplex<f64>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<18xcomplex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc5)
    %2:2 = stablehlo.custom_call @cusolver_geqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x3x3xcomplex<f64>>) -> (tensor<2x3x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc7)
    %4 = stablehlo.custom_call @cusolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>]} : (tensor<2x3x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc12)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc12)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xcomplex<f64>> loc(#loc13)
    return %4, %12 : tensor<2x3x3xcomplex<f64>>, tensor<2x3x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":391:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/pad"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/iota"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/add"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/ge"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/broadcast_in_dim"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#'+/37\x03\xc9\x89-\x01C\x17\x07\x0b\x0f\x0b\x13\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x03G\x0b/\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x1fO\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x01\x05\x0b\x0f\x03)\x1b\x07\x0b\x17\x0f\x07\x0f\x07\x07\x17\x07\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02~\x05\x17\x05\x1e\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x03)q\x1d\t\x01\x1d7\x01\x1d=\x01\x03\x07\x15\x17\x19\x07\x1b\x07\x05!\x11\x01\x00\x05#\x05%\x05'\x1d\t!\x17\x05\x1a\x065\x1d%'\x05)\x17\x05\x1a\x06\x1d\x05+\x1d-\x01\x05-\x1d1\x01\x05/\x1d5\x01\x051\x053\x1d;\x01\x055\x057\x1dA\x01\x059\x03\x01\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d;\x1d=\x1d?\x13\x07\x01\x0b\x03\x1dA\x05\x01\x03\x03W\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Wy\x1f%\x01#\x17\x03\x05ae\r\x05GcIK\x1dC\r\x05GgIK\x1dE\x1dG\x1dI\x1f\r\t\xff\xff\xff\xff\x1f\x11!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\x1dK\x03\x03w\x15\x03\x01\x01\x01\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dM\x03\x03\x7f\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f+!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x03\x19)\x05\r\r\x0f)\x01\x0f\x1b)\x01\t\x13\x01\x11\x01\x05\x05\x05\x0b)\x03I\t)\x05\t\r\t)\x03\r\x13)\x03\t\x13)\x03\r\x07)\x03\x01\x07)\x05\r\r\x15)\x07\t\r\r\x15)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x03\x05\x03\r\x07B\x03\x07\x03\x11\x03B\x1f\t\x03\x1b\r\x06#\x03\x05\x03\x05\tG+\x0b\x0b\x05\x05\x1d\x03\x07\x0fF/\r\x03\x05\x05\t\x03\tG3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\x05F\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03'\x05\x15\x17\x05F\x11\x17\x03)\x03\x19\x05F\x11\x11\x03\x05\x03\x03\x15\x06?\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00J\x0bO''\x0f\x0b\t\t\x03\x11#!CS79Y9=)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00main\x00public\x00cusolver_geqrf_ffi\x00cusolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0bC]_ik\x03m\x03o\x03M\x11OQsCSUuY\x07EEE\x11OQ{CSY}U\x03[\x03\x81\x05\x83\x85\x03\x87",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
