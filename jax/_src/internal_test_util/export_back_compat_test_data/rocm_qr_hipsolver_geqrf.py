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


data_2026_02_04 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["f32"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_geqrf_ffi', 'hipsolver_orgqr_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(),
    expected_outputs=(array([[[ 0.        ,  0.9128709 ,  0.40824834],
        [-0.4472136 ,  0.3651484 , -0.81649655],
        [-0.8944272 , -0.18257423,  0.40824828]],

       [[-0.42426407,  0.8082909 ,  0.40824693],
        [-0.5656854 ,  0.11546878, -0.81649673],
        [-0.7071068 , -0.5773496 ,  0.40824923]]], dtype=float32), array([[[-6.7082038e+00, -8.0498447e+00, -9.3914852e+00],
        [ 0.0000000e+00,  1.0954450e+00,  2.1908898e+00],
        [ 0.0000000e+00,  0.0000000e+00,  4.8374091e-08]],

       [[-2.1213203e+01, -2.2910263e+01, -2.4607315e+01],
        [ 0.0000000e+00,  3.4641260e-01,  6.9282037e-01],
        [ 0.0000000e+00,  0.0000000e+00, -1.8281829e-06]]], dtype=float32)),
    mlir_module_text=r"""
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xf32> {jax.result_info = "result[0]"}, tensor<2x3x3xf32> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc7)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xf32> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<18xf32>) -> tensor<2x3x3xf32> loc(#loc9)
    %2:2 = stablehlo.custom_call @hipsolver_geqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xf32>) -> (tensor<2x3x3xf32>, tensor<2x3xf32>) loc(#loc10)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xf32>, tensor<f32>) -> tensor<2x3x3xf32> loc(#loc10)
    %4 = stablehlo.custom_call @hipsolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, m, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xf32>, tensor<2x3xf32>) -> tensor<2x3x3xf32> loc(#loc10)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc10)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc10)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc10)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc10)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3x3xf32> loc(#loc10)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xf32> loc(#loc10)
    return %4, %12 : tensor<2x3x3xf32>, tensor<2x3x3xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":403:11)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":402:26)
#loc3 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":402:14)
#loc4 = loc("jit(<lambda>)"(#loc1))
#loc5 = loc("jit(<lambda>)"(#loc2))
#loc6 = loc("jit(<lambda>)"(#loc3))
#loc7 = loc("qr"(#loc4))
#loc8 = loc("iota"(#loc5))
#loc9 = loc("reshape"(#loc6))
#loc10 = loc(""(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03\xdf\x9d+\x01;\x0f\x07\x0b\x0b\x0f\x0f\x0b\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17#\x0b#\x03I\x0b/\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x0b\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x05\x1b\x0f\x17\x0f\x0f\x0fK\x0f\x0f\x17\x13K\x13\x17\x01\x05\x0b\x0f\x03'\x1b\x07\x07\x17\x0f\x07\x0f\x07\x07\x17\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02v\x06\x1d7\x0b\x1f\x05\x1f\x05!\x11\x03\x05\x1d\x05#\x05#\x05%\x05'\x03\x07\x15\x17\x19\t\x1b\t\x05)\x11\x01\x00\x05+\x05-\x05/\x1d!\x0b\x051\x17\x07N\x06\x17\x1d')\x053\x1d\x05+\x17\x07J\x065\x1d/1\x055\x1d\x053\x17\x07J\x06\x1d\x03\x07\rC\x0fE\x11\x8d\x057\x03\x07\rC\x0fE\x11\x97\x03\x01\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d9\x13\x07\x01\r\x01\r\x03ik\x0b\x03\x1d7\x05\x01\x03\x03O\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Os\x1f#\x01#\x17\x03\x05Y]\r\x03?[\x1d;\r\x03?_\x1d=\x1d?\x1dA\x1f\r\t\xff\xff\xff\xff\x1f\x11\t\x00\x00\x00\x00\x1dC\x1dE\x1dG\x03\x03q\x15\x03\x01\x01\x01\x1f\x1f!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dI\x03\x03y\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x13\x07\x83\x8f\x91\x11\x03\r\x11\x03\x11\x11\x03\x15\x15\r\t\r\r\r\r\r\x03\x85\x05\x93\x95\x01\x01\x01\x01\x01\x11\x03\x05\x11\x03\t\x13\x07\x83\x87\x89\x13\x05\x83\x8b\x15\r\t\r\r\r\r\r\x05\x85\x99\x03\x9b\x01\x01\x01\x01\x01\x13\x05\x83\x87\x13\x07\x83\x89\x8b\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\t)\x05\r\r\x0f)\x01\x0f\x1b)\x01\t\x13\x01\x11\x01\x05\x05\x05)\x03I\t)\x05\t\r\t)\x03\r\x13)\x03\t\x13)\x03\r\x07)\x03\x01\x07)\x05\r\r\x15)\x07\t\r\r\x15)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x1f\x05\x03\r\x07B\x03\x07\x03\x11\x03B%\t\x03\x19\r\x06-\x03\x05\x03\x05\tG\x015\x0b\x05\x05\x1b\x03\x07\x0fF\x01\r\x03\x05\x05\t\x03\tG\x019\x0f\x03\x05\x05\r\x0b\x03B\x01\t\x03\x0b\x05F\x01\x11\x03\x0b\x03\x01\x11\x06\x01\x03\x0b\x05\x11\x13\x03B\x01\x13\x03\x0b\x13F\x01\x15\x03%\x05\x15\x17\x05F\x01\x17\x03'\x03\x19\x05F\x01\x11\x03\x05\x03\x03\x15\x06\x01\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00\x0e\x08K))\x05\x1f\x0f\x0b\x15\x15!\x03\x11\x0b\x07\x19%)9%3)s\x1d\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00jit(<lambda>)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00qr\x00iota\x00reshape\x00\x00jax.result_info\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x001\x00hipsolver_geqrf_ffi\x00hipsolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0b;UWac\x03e\x03g\x03A\x11GIm;KMoQ\x07===\x11GIu;KQwM\x03S\x03{\x05}\x7f\x03\x81",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["f64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_geqrf_ffi', 'hipsolver_orgqr_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(),
    expected_outputs=(array([[[ 0.                 ,  0.9128709291752773 ,
          0.408248290463862  ],
        [-0.447213595499958  ,  0.36514837167011   ,
         -0.8164965809277264 ],
        [-0.894427190999916  , -0.18257418583505472,
          0.4082482904638633 ]],

       [[-0.42426406871192857,  0.8082903768654768 ,
          0.4082482904638614 ],
        [-0.565685424949238  ,  0.11547005383792366,
         -0.8164965809277263 ],
        [-0.7071067811865476 , -0.577350269189625  ,
          0.4082482904638642 ]]]), array([[[-6.7082039324993694e+00, -8.0498447189992444e+00,
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
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xf64> {jax.result_info = "result[0]"}, tensor<2x3x3xf64> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i64> loc(#loc7)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xf64> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<18xf64>) -> tensor<2x3x3xf64> loc(#loc9)
    %2:2 = stablehlo.custom_call @hipsolver_geqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xf64>) -> (tensor<2x3x3xf64>, tensor<2x3xf64>) loc(#loc10)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xf64>, tensor<f64>) -> tensor<2x3x3xf64> loc(#loc10)
    %4 = stablehlo.custom_call @hipsolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, m, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xf64>, tensor<2x3xf64>) -> tensor<2x3x3xf64> loc(#loc10)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi64> loc(#loc10)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<3x3xi64> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi64> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi64> loc(#loc10)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1> loc(#loc10)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc10)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3x3xf64> loc(#loc10)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xf64> loc(#loc10)
    return %4, %12 : tensor<2x3x3xf64>, tensor<2x3x3xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":403:11)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":402:26)
#loc3 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":402:14)
#loc4 = loc("jit(<lambda>)"(#loc1))
#loc5 = loc("jit(<lambda>)"(#loc2))
#loc6 = loc("jit(<lambda>)"(#loc3))
#loc7 = loc("qr"(#loc4))
#loc8 = loc("iota"(#loc5))
#loc9 = loc("reshape"(#loc6))
#loc10 = loc(""(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03\xdd\x9d)\x01;\x0f\x07\x0b\x0b\x0f\x0f\x0b\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17#\x0b#\x03I\x0b/\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b//\x0b\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x05\x1b\x0f\x17\x0f\x0f\x0fK\x0f\x0f\x17\x13K\x13\x17\x01\x05\x0b\x0f\x03%\x1b\x07\x07\x17\x0f\x0f\x07\x07\x17\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02\x92\x06\x1d7\x0b\x1f\x05\x1f\x05!\x11\x03\x05\x1d\x05#\x05#\x05%\x05'\x03\x07\x15\x17\x19\t\x1b\t\x05)\x11\x01\x00\x05+\x05-\x05/\x1d!\x0b\x051\x17\x07N\x06\x17\x1d')\x053\x1d\x05+\x17\x07J\x065\x1d/1\x055\x1d\x053\x17\x07J\x06\x1d\x03\x07\rC\x0fE\x11\x8d\x057\x03\x07\rC\x0fE\x11\x97\x03\x01\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d9\x13\x07\x01\r\x01\r\x03ik\x0b\x03\x1d7\x05\x01\x03\x03O\x1f\x1b1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Os\x1f!\x01#\x15\x03\x05Y]\r\x03?[\x1d;\r\x03?_\x1d=\x1d?\x1dA\x1f\r\x11\xff\xff\xff\xff\xff\xff\xff\xff\x1f\x0f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dC\x1dE\x1dG\x03\x03q\x15\x03\x01\x01\x01\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dI\x03\x03y\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f'!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x13\x07\x83\x8f\x91\x11\x03\r\x11\x03\x11\x11\x03\x15\x15\r\t\r\r\r\r\r\x03\x85\x05\x93\x95\x01\x01\x01\x01\x01\x11\x03\x05\x11\x03\t\x13\x07\x83\x87\x89\x13\x05\x83\x8b\x15\r\t\r\r\r\r\r\x05\x85\x99\x03\x9b\x01\x01\x01\x01\x01\x13\x05\x83\x87\x13\x07\x83\x89\x8b\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x0b)\x05\r\r\x07)\x01\x07)\x01\t\x13\x01\x11\x01\x05\x05\x05)\x03I\t)\x05\t\r\t)\x03\r\x11)\x03\t\x11)\x03\r\x07)\x03\x01\x07)\x05\r\r\x13)\x07\t\r\r\x13)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x1f\x05\x03\r\x07B\x03\x07\x03\x0f\x03B%\t\x03\x17\r\x06-\x03\x05\x03\x05\tG\x015\x0b\x05\x05\x19\x03\x07\x0fF\x01\r\x03\x05\x05\t\x03\tG\x019\x0f\x03\x05\x05\r\x0b\x03B\x01\t\x03\x0b\x05F\x01\x11\x03\x0b\x03\x01\x11\x06\x01\x03\x0b\x05\x11\x13\x03B\x01\x13\x03\x0b\x13F\x01\x15\x03#\x05\x15\x17\x05F\x01\x17\x03%\x03\x19\x05F\x01\x11\x03\x05\x03\x03\x15\x06\x01\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00\x0e\x08K))\x05\x1f\x0f\x0b\x15\x15!\x03\x11\x0b\x07\x19%)9%3)s\x1d\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00jit(<lambda>)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00qr\x00iota\x00reshape\x00\x00jax.result_info\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x001\x00hipsolver_geqrf_ffi\x00hipsolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0b;UWac\x03e\x03g\x03A\x11GIm;KMoQ\x07===\x11GIu;KQwM\x03S\x03{\x05}\x7f\x03\x81",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["c64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_geqrf_ffi', 'hipsolver_orgqr_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(),
    expected_outputs=(array([[[ 0.         -0.j,  0.9128708  +0.j,  0.4082482  +0.j],
        [-0.4472136  -0.j,  0.36514837 +0.j, -0.81649655 +0.j],
        [-0.8944272  -0.j, -0.18257421 +0.j,  0.40824828 +0.j]],

       [[-0.42426407 -0.j,  0.8082913  +0.j,  0.4082465  +0.j],
        [-0.5656854  -0.j,  0.115468234+0.j, -0.81649685 +0.j],
        [-0.7071068  -0.j, -0.57734936 +0.j,  0.4082496  +0.j]]],
      dtype=complex64), array([[[-6.7082038e+00+0.j, -8.0498447e+00+0.j, -9.3914852e+00+0.j],
        [ 0.0000000e+00+0.j,  1.0954450e+00+0.j,  2.1908901e+00+0.j],
        [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  0.0000000e+00+0.j]],

       [[-2.1213203e+01+0.j, -2.2910263e+01+0.j, -2.4607315e+01+0.j],
        [ 0.0000000e+00+0.j,  3.4641233e-01+0.j,  6.9282043e-01+0.j],
        [ 0.0000000e+00+0.j,  0.0000000e+00+0.j, -1.9669533e-06+0.j]]],
      dtype=complex64)),
    mlir_module_text=r"""
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<2x3x3xcomplex<f32>> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc7)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xcomplex<f32>> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<18xcomplex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc9)
    %2:2 = stablehlo.custom_call @hipsolver_geqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xcomplex<f32>>) -> (tensor<2x3x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) loc(#loc10)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc10)
    %4 = stablehlo.custom_call @hipsolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, m, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc10)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc10)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc10)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc10)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc10)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc10)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xcomplex<f32>> loc(#loc10)
    return %4, %12 : tensor<2x3x3xcomplex<f32>>, tensor<2x3x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":403:11)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":402:26)
#loc3 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":402:14)
#loc4 = loc("jit(<lambda>)"(#loc1))
#loc5 = loc("jit(<lambda>)"(#loc2))
#loc6 = loc("jit(<lambda>)"(#loc3))
#loc7 = loc("qr"(#loc4))
#loc8 = loc("iota"(#loc5))
#loc9 = loc("reshape"(#loc6))
#loc10 = loc(""(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03\xe1\x9d-\x01;\x0f\x07\x0b\x0b\x0f\x0f\x0b\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17#\x0b#\x03I\x0b/\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x1f/\x0b\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x05\x1b\x0f\x17\x0f\x0f\x0fK\x0f\x0f\x17\x13K\x13\x17\x01\x05\x0b\x0f\x03)\x1b\x07\x0b\x17\x0f\x07\x0f\x07\x07\x17\x07\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02\x8e\x06\x1d7\x0b\x1f\x05\x1f\x05!\x11\x03\x05\x1d\x05#\x05#\x05%\x05'\x03\x07\x15\x17\x19\t\x1b\t\x05)\x11\x01\x00\x05+\x05-\x05/\x1d!\x0b\x051\x17\x07N\x06\x17\x1d')\x053\x1d\x05+\x17\x07J\x065\x1d/1\x055\x1d\x053\x17\x07J\x06\x1d\x03\x07\rC\x0fE\x11\x8d\x057\x03\x07\rC\x0fE\x11\x97\x03\x01\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d9\x13\x07\x01\r\x01\r\x03ik\x0b\x03\x1d7\x05\x01\x03\x03O\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Os\x1f%\x01#\x17\x03\x05Y]\r\x03?[\x1d;\r\x03?_\x1d=\x1d?\x1dA\x1f\r\t\xff\xff\xff\xff\x1f\x11\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dC\x1dE\x1dG\x03\x03q\x15\x03\x01\x01\x01\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dI\x03\x03y\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f+!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x13\x07\x83\x8f\x91\x11\x03\r\x11\x03\x11\x11\x03\x15\x15\r\t\r\r\r\r\r\x03\x85\x05\x93\x95\x01\x01\x01\x01\x01\x11\x03\x05\x11\x03\t\x13\x07\x83\x87\x89\x13\x05\x83\x8b\x15\r\t\r\r\r\r\r\x05\x85\x99\x03\x9b\x01\x01\x01\x01\x01\x13\x05\x83\x87\x13\x07\x83\x89\x8b\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x03\x19)\x05\r\r\x0f)\x01\x0f\x1b)\x01\t\x13\x01\x11\x01\x05\x05\x05\t)\x03I\t)\x05\t\r\t)\x03\r\x13)\x03\t\x13)\x03\r\x07)\x03\x01\x07)\x05\r\r\x15)\x07\t\r\r\x15)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x1f\x05\x03\r\x07B\x03\x07\x03\x11\x03B%\t\x03\x1b\r\x06-\x03\x05\x03\x05\tG\x015\x0b\x05\x05\x1d\x03\x07\x0fF\x01\r\x03\x05\x05\t\x03\tG\x019\x0f\x03\x05\x05\r\x0b\x03B\x01\t\x03\x0b\x05F\x01\x11\x03\x0b\x03\x01\x11\x06\x01\x03\x0b\x05\x11\x13\x03B\x01\x13\x03\x0b\x13F\x01\x15\x03'\x05\x15\x17\x05F\x01\x17\x03)\x03\x19\x05F\x01\x11\x03\x05\x03\x03\x15\x06\x01\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00\x0e\x08K))\x05\x1f\x0f\x0b\x15\x15!\x03\x11\x0b\x07\x19%)9%3)s\x1d\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00jit(<lambda>)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00qr\x00iota\x00reshape\x00\x00jax.result_info\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x001\x00hipsolver_geqrf_ffi\x00hipsolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0b;UWac\x03e\x03g\x03A\x11GIm;KMoQ\x07===\x11GIu;KQwM\x03S\x03{\x05}\x7f\x03\x81",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["c128"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_geqrf_ffi', 'hipsolver_orgqr_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(),
    expected_outputs=(array([[[ 0.                 -0.j,  0.9128709291752763 +0.j,
          0.4082482904638633 +0.j],
        [-0.44721359549995787-0.j,  0.3651483716701112 +0.j,
         -0.8164965809277258 +0.j],
        [-0.8944271909999157 -0.j, -0.1825741858350558 +0.j,
          0.4082482904638628 +0.j]],

       [[-0.42426406871192857-0.j,  0.8082903768654766 +0.j,
          0.4082482904638618 +0.j],
        [-0.565685424949238  -0.j,  0.11547005383792402+0.j,
         -0.8164965809277261 +0.j],
        [-0.7071067811865476 -0.j, -0.5773502691896252 +0.j,
          0.40824829046386385+0.j]]]), array([[[-6.7082039324993694e+00+0.j, -8.0498447189992426e+00+0.j,
         -9.3914855054991158e+00+0.j],
        [ 0.0000000000000000e+00+0.j,  1.0954451150103306e+00+0.j,
          2.1908902300206621e+00+0.j],
        [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
          0.0000000000000000e+00+0.j]],

       [[-2.1213203435596427e+01+0.j, -2.2910259710444144e+01+0.j,
         -2.4607315985291855e+01+0.j],
        [ 0.0000000000000000e+00+0.j,  3.4641016151378073e-01+0.j,
          6.9282032302755370e-01+0.j],
        [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
         -2.1094237467877974e-15+0.j]]])),
    mlir_module_text=r"""
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<2x3x3xcomplex<f64>> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i64> loc(#loc7)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xcomplex<f64>> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<18xcomplex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc9)
    %2:2 = stablehlo.custom_call @hipsolver_geqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xcomplex<f64>>) -> (tensor<2x3x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) loc(#loc10)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc10)
    %4 = stablehlo.custom_call @hipsolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, m, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc10)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi64> loc(#loc10)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<3x3xi64> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi64> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi64> loc(#loc10)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1> loc(#loc10)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc10)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc10)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xcomplex<f64>> loc(#loc10)
    return %4, %12 : tensor<2x3x3xcomplex<f64>>, tensor<2x3x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":403:11)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":402:26)
#loc3 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":402:14)
#loc4 = loc("jit(<lambda>)"(#loc1))
#loc5 = loc("jit(<lambda>)"(#loc2))
#loc6 = loc("jit(<lambda>)"(#loc3))
#loc7 = loc("qr"(#loc4))
#loc8 = loc("iota"(#loc5))
#loc9 = loc("reshape"(#loc6))
#loc10 = loc(""(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03\xdf\x9d+\x01;\x0f\x07\x0b\x0b\x0f\x0f\x0b\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17#\x0b#\x03I\x0b/\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b/O\x0b\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x05\x1b\x0f\x17\x0f\x0f\x0fK\x0f\x0f\x17\x13K\x13\x17\x01\x05\x0b\x0f\x03'\x1b\x07\x0b\x17\x0f\x0f\x07\x07\x17\x07\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02\xba\x06\x1d7\x0b\x1f\x05\x1f\x05!\x11\x03\x05\x1d\x05#\x05#\x05%\x05'\x03\x07\x15\x17\x19\t\x1b\t\x05)\x11\x01\x00\x05+\x05-\x05/\x1d!\x0b\x051\x17\x07N\x06\x17\x1d')\x053\x1d\x05+\x17\x07J\x065\x1d/1\x055\x1d\x053\x17\x07J\x06\x1d\x03\x07\rC\x0fE\x11\x8d\x057\x03\x07\rC\x0fE\x11\x97\x03\x01\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d9\x13\x07\x01\r\x01\r\x03ik\x0b\x03\x1d7\x05\x01\x03\x03O\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Os\x1f#\x01#\x15\x03\x05Y]\r\x03?[\x1d;\r\x03?_\x1d=\x1d?\x1dA\x1f\r\x11\xff\xff\xff\xff\xff\xff\xff\xff\x1f\x0f!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dC\x1dE\x1dG\x03\x03q\x15\x03\x01\x01\x01\x1f\x1f!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dI\x03\x03y\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x13\x07\x83\x8f\x91\x11\x03\r\x11\x03\x11\x11\x03\x15\x15\r\t\r\r\r\r\r\x03\x85\x05\x93\x95\x01\x01\x01\x01\x01\x11\x03\x05\x11\x03\t\x13\x07\x83\x87\x89\x13\x05\x83\x8b\x15\r\t\r\r\r\r\r\x05\x85\x99\x03\x9b\x01\x01\x01\x01\x01\x13\x05\x83\x87\x13\x07\x83\x89\x8b\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x03\x17)\x05\r\r\x07)\x01\x07)\x01\t\x13\x01\x11\x01\x05\x05\x05\x0b)\x03I\t)\x05\t\r\t)\x03\r\x11)\x03\t\x11)\x03\r\x07)\x03\x01\x07)\x05\r\r\x13)\x07\t\r\r\x13)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x1f\x05\x03\r\x07B\x03\x07\x03\x0f\x03B%\t\x03\x19\r\x06-\x03\x05\x03\x05\tG\x015\x0b\x05\x05\x1b\x03\x07\x0fF\x01\r\x03\x05\x05\t\x03\tG\x019\x0f\x03\x05\x05\r\x0b\x03B\x01\t\x03\x0b\x05F\x01\x11\x03\x0b\x03\x01\x11\x06\x01\x03\x0b\x05\x11\x13\x03B\x01\x13\x03\x0b\x13F\x01\x15\x03%\x05\x15\x17\x05F\x01\x17\x03'\x03\x19\x05F\x01\x11\x03\x05\x03\x03\x15\x06\x01\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00\x0e\x08K))\x05\x1f\x0f\x0b\x15\x15!\x03\x11\x0b\x07\x19%)9%3)s\x1d\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00jit(<lambda>)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00qr\x00iota\x00reshape\x00\x00jax.result_info\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x001\x00hipsolver_geqrf_ffi\x00hipsolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0b;UWac\x03e\x03g\x03A\x11GIm;KMoQ\x07===\x11GIu;KQwM\x03S\x03{\x05}\x7f\x03\x81",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
