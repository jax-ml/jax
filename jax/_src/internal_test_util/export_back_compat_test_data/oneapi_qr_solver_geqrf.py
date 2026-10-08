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
    custom_call_targets=['oneapisolver_geqrf_ffi', 'oneapisolver_orgqr_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(),
    expected_outputs=(array([[[ 0.         ,  0.91287076 ,  0.4082487  ],
        [-0.44721356 ,  0.36514866 , -0.8164965  ],
        [-0.8944271  , -0.18257445 ,  0.40824816 ]],

       [[-0.42426407 ,  0.8082888  ,  0.40825114 ],
        [-0.5656854  ,  0.115472935, -0.8164962  ],
        [-0.7071067  , -0.57735175 ,  0.4082462  ]]], dtype=float32), array([[[-6.7082043e+00, -8.0498438e+00, -9.3914852e+00],
        [ 0.0000000e+00,  1.0954441e+00,  2.1908894e+00],
        [ 0.0000000e+00,  0.0000000e+00,  7.1525574e-07]],

       [[-2.1213205e+01, -2.2910259e+01, -2.4607315e+01],
        [ 0.0000000e+00,  3.4640804e-01,  6.9281834e-01],
        [ 0.0000000e+00,  0.0000000e+00,  1.4901161e-06]]], dtype=float32)),
    mlir_module_text=r"""
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xf32> {jax.result_info = "result[0]"}, tensor<2x3x3xf32> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i64> loc(#loc17)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xf32> loc(#loc18)
    %1 = stablehlo.reshape %0 : (tensor<18xf32>) -> tensor<2x3x3xf32> loc(#loc19)
    %2:2 = stablehlo.custom_call @oneapisolver_geqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xf32>) -> (tensor<2x3x3xf32>, tensor<2x3xf32>) loc(#loc28)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xf32>, tensor<f32>) -> tensor<2x3x3xf32> loc(#loc29)
    %4 = stablehlo.custom_call @oneapisolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, m, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xf32>, tensor<2x3xf32>) -> tensor<2x3x3xf32> loc(#loc30)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi64> loc(#loc31)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<3x3xi64> loc(#loc32)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi64> loc(#loc32)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi64> loc(#loc31)
    %9 = stablehlo.compare GE, %7, %8, SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1> loc(#loc33)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc34)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3x3xf32> loc(#loc34)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xf32> loc(#loc35)
    return %4, %12 : tensor<2x3x3xf32>, tensor<2x3x3xf32> loc(#loc16)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":500:11)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":499:26)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":499:14)
#loc4 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":543:4)
#loc5 = loc("jit(<lambda>)"(#loc1))
#loc6 = loc("jit(<lambda>)"(#loc2))
#loc7 = loc("jit(<lambda>)"(#loc3))
#loc8 = loc("geqrf"(#loc1))
#loc9 = loc("pad"(#loc1))
#loc10 = loc("householder_product"(#loc1))
#loc11 = loc("iota"(#loc1))
#loc12 = loc("add"(#loc1))
#loc13 = loc("ge"(#loc1))
#loc14 = loc("broadcast_in_dim"(#loc1))
#loc15 = loc("select_n"(#loc1))
#loc16 = loc("jit(<lambda>)"(#loc4))
#loc17 = loc("qr"(#loc5))
#loc18 = loc("iota"(#loc6))
#loc19 = loc("reshape"(#loc7))
#loc20 = loc(callsite(#loc8 at #loc5))
#loc21 = loc(callsite(#loc9 at #loc5))
#loc22 = loc(callsite(#loc10 at #loc5))
#loc23 = loc(callsite(#loc11 at #loc5))
#loc24 = loc(callsite(#loc12 at #loc5))
#loc25 = loc(callsite(#loc13 at #loc5))
#loc26 = loc(callsite(#loc14 at #loc5))
#loc27 = loc(callsite(#loc15 at #loc5))
#loc28 = loc(""(#loc20))
#loc29 = loc(""(#loc21))
#loc30 = loc(""(#loc22))
#loc31 = loc(""(#loc23))
#loc32 = loc(""(#loc24))
#loc33 = loc(""(#loc25))
#loc34 = loc(""(#loc26))
#loc35 = loc(""(#loc27))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03:\x02\xdd)\x01{\x0f\x17\x0b\x0b\x0b\x07\x0f\x0b\x0b\x0b\x0b\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0f\x17\x0f\x0b\x0f\x17#\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b#\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x17\x03I\x0b/\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b/\x1f\x0b\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x05\x1b\x0f\x17\x0f\x0f\x0fK\x0f\x0f\x17\x13K\x13\x17\x01\x05\x0b\x0f\x03%\x1b\x07\x07\x17\x0f\x0f\x07\x07\x17\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02\xee\x07\x1d\x07\x03\x17\t\xd2\x07\x17\x05\x1f\x05!\x05#\x1f\x11\x03\x05\x05%\x05'\x05)\x05+\x1d\x05W\x1d\x05[\x1d\x05i\x03\x07\x1f!#\r%\r\x05-\x11\x01\x00\x05/\x051\x053\x1d+\x01\x055\x1d\x0f/\x1d\x071\x17\t\xce\x075\x1d57\x057\x1d\x079\x17\t\xce\x07\x1d\x03\x07\x11\x83\x13\x85\x15\xcd\x1d\x05?\x15A\x01\x1dC\x03\x059\x1d\x05G\x15I\x01\x1dK\x03\x05;\x03\x07\x11\x83\x13\x85\x15\xd7\x1d\x05Q\x15S\x01\x1dU\x03\x05=\x15Y\x01\x1d\x0f\x03\x15]\x01\x1d_\x03\x05?\x1d\x05c\x15e\x01\x1dg\x03\x05A\x15k\x01\x1dm\x03\x05C\x1d\x05q\x15s\x01\x1du\x03\x05E\x1d\x07y\x17\t~\x08\t\x03\x01\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dG\x13\x07\x01\r\x01\r\x03\xa9\xab\x0b\x03\x1d\x1f\x05\x01\x03\x03\x8f\x1f\x1b1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05\x8f\xb3\x1f!\x01#\x15\x03\x05\x99\x9d\r\x03\x7f\x9b\x1dI\r\x03\x7f\x9f\x1dK\x1dM\x1dO\x1f\r\x11\xff\xff\xff\xff\xff\xff\xff\xff\x1f\x0f\t\x00\x00\x00\x00\x1dQ\x1dS\x1dU\x03\x03\xb1\x15\x03\x01\x01\x01\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dW\x03\x03\xb9\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f'!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x13\x07\xc3\xcf\xd1\x11\x03\r\x11\x03\x11\x11\x03\x15\x15\r\t\r\r\r\r\r\x03\xc5\x05\xd3\xd5\x01\x01\x01\x01\x01\x11\x03\x05\x11\x03\t\x13\x07\xc3\xc7\xc9\x13\x05\xc3\xcb\x15\r\t\r\r\r\r\r\x05\xc5\xd9\x03\xdb\x01\x01\x01\x01\x01\x13\x05\xc3\xc7\x13\x07\xc3\xc9\xcb\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\t)\x05\r\r\x07)\x01\x07)\x01\t\x13\x01\x11\x01\x05\x05\x05)\x03I\t)\x05\t\r\t)\x03\r\x11)\x03\t\x11)\x03\r\x07)\x03\x01\x07)\x05\r\r\x13)\x07\t\r\r\x13)\x03\t\x07\x04^\x02\x05\x01Q\x0b\x1d\x01\x07\x046\x02\x03\x01\x05\x0bP\x0b\x03\x07\x04\n\x02\x03!A\x07B)\x05\x03\r\x07B\x0b\x07\x03\x0f\x03B-\t\x03\x17\r\x063\x03\x05\x03\x05\tg=;\x0b\x05\x05\x19\x03\x07\x03\x01\t\x03\x01\x0fFE\r\x03\x05\x05\t\x03\tGOM\x0f\x03\x05\x05\r\x0b\x03B\x17\t\x03\x0b\x05F\x19\x11\x03\x0b\x03\x01\x11\x06\x19\x03\x0b\x05\x11\x13\x03B\x17\x13\x03\x0b\x13Fa\x15\x03#\x05\x15\x17\x05F\x1b\x17\x03%\x03\x19\x05F\x1b\x11\x03\x05\x03\x03\x15\x06o\x03\x05\x07\x1b\x1d\t\x17\x04w\x05\x0f\x1f\x06\x03\x01\x05\x01\x00F\tY//\x05\x1f\x0f\x0b\x15\x15!\x13#\x07\t)\t\r\x11\x07\x19%)9%3)\x0bw\x1d\x03\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00\x00jit(<lambda>)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00iota\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00qr\x00reshape\x00geqrf\x00pad\x00householder_product\x00add\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x001\x00oneapisolver_geqrf_ffi\x00oneapisolver_orgqr_ffi\x00\x08_\x19\x05O\x01\x0b{\x95\x97\xa1\xa3\x03\xa5\x03\xa7\x03\x81\x11\x87\x89\xad{\x8b\x8d\xaf\x91\x07}}}\x11\x87\x89\xb5{\x8b\x91\xb7\x8d\x03\x93\x03\xbb\x05\xbd\xbf\x03\xc1",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["f64"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_geqrf_ffi', 'oneapisolver_orgqr_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
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
    %c = stablehlo.constant dense<-1> : tensor<i64> loc(#loc17)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xf64> loc(#loc18)
    %1 = stablehlo.reshape %0 : (tensor<18xf64>) -> tensor<2x3x3xf64> loc(#loc19)
    %2:2 = stablehlo.custom_call @oneapisolver_geqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xf64>) -> (tensor<2x3x3xf64>, tensor<2x3xf64>) loc(#loc28)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xf64>, tensor<f64>) -> tensor<2x3x3xf64> loc(#loc29)
    %4 = stablehlo.custom_call @oneapisolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, m, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xf64>, tensor<2x3xf64>) -> tensor<2x3x3xf64> loc(#loc30)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi64> loc(#loc31)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<3x3xi64> loc(#loc32)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi64> loc(#loc32)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi64> loc(#loc31)
    %9 = stablehlo.compare GE, %7, %8, SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1> loc(#loc33)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc34)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3x3xf64> loc(#loc34)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xf64> loc(#loc35)
    return %4, %12 : tensor<2x3x3xf64>, tensor<2x3x3xf64> loc(#loc16)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":500:11)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":499:26)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":499:14)
#loc4 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":543:4)
#loc5 = loc("jit(<lambda>)"(#loc1))
#loc6 = loc("jit(<lambda>)"(#loc2))
#loc7 = loc("jit(<lambda>)"(#loc3))
#loc8 = loc("geqrf"(#loc1))
#loc9 = loc("pad"(#loc1))
#loc10 = loc("householder_product"(#loc1))
#loc11 = loc("iota"(#loc1))
#loc12 = loc("add"(#loc1))
#loc13 = loc("ge"(#loc1))
#loc14 = loc("broadcast_in_dim"(#loc1))
#loc15 = loc("select_n"(#loc1))
#loc16 = loc("jit(<lambda>)"(#loc4))
#loc17 = loc("qr"(#loc5))
#loc18 = loc("iota"(#loc6))
#loc19 = loc("reshape"(#loc7))
#loc20 = loc(callsite(#loc8 at #loc5))
#loc21 = loc(callsite(#loc9 at #loc5))
#loc22 = loc(callsite(#loc10 at #loc5))
#loc23 = loc(callsite(#loc11 at #loc5))
#loc24 = loc(callsite(#loc12 at #loc5))
#loc25 = loc(callsite(#loc13 at #loc5))
#loc26 = loc(callsite(#loc14 at #loc5))
#loc27 = loc(callsite(#loc15 at #loc5))
#loc28 = loc(""(#loc20))
#loc29 = loc(""(#loc21))
#loc30 = loc(""(#loc22))
#loc31 = loc(""(#loc23))
#loc32 = loc(""(#loc24))
#loc33 = loc(""(#loc25))
#loc34 = loc(""(#loc26))
#loc35 = loc(""(#loc27))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03:\x02\xdd)\x01{\x0f\x17\x0b\x0b\x0b\x07\x0f\x0b\x0b\x0b\x0b\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0f\x17\x0f\x0b\x0f\x17#\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b#\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x17\x03I\x0b/\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b//\x0b\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x05\x1b\x0f\x17\x0f\x0f\x0fK\x0f\x0f\x17\x13K\x13\x17\x01\x05\x0b\x0f\x03%\x1b\x07\x07\x17\x0f\x0f\x07\x07\x17\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02\xfe\x07\x1d\x07\x03\x17\t\xd2\x07\x17\x05\x1f\x05!\x05#\x1f\x11\x03\x05\x05%\x05'\x05)\x05+\x1d\x05W\x1d\x05[\x1d\x05i\x03\x07\x1f!#\r%\r\x05-\x11\x01\x00\x05/\x051\x053\x1d+\x01\x055\x1d\x0f/\x1d\x071\x17\t\xce\x075\x1d57\x057\x1d\x079\x17\t\xce\x07\x1d\x03\x07\x11\x83\x13\x85\x15\xcd\x1d\x05?\x15A\x01\x1dC\x03\x059\x1d\x05G\x15I\x01\x1dK\x03\x05;\x03\x07\x11\x83\x13\x85\x15\xd7\x1d\x05Q\x15S\x01\x1dU\x03\x05=\x15Y\x01\x1d\x0f\x03\x15]\x01\x1d_\x03\x05?\x1d\x05c\x15e\x01\x1dg\x03\x05A\x15k\x01\x1dm\x03\x05C\x1d\x05q\x15s\x01\x1du\x03\x05E\x1d\x07y\x17\t~\x08\t\x03\x01\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dG\x13\x07\x01\r\x01\r\x03\xa9\xab\x0b\x03\x1d\x1f\x05\x01\x03\x03\x8f\x1f\x1b1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05\x8f\xb3\x1f!\x01#\x15\x03\x05\x99\x9d\r\x03\x7f\x9b\x1dI\r\x03\x7f\x9f\x1dK\x1dM\x1dO\x1f\r\x11\xff\xff\xff\xff\xff\xff\xff\xff\x1f\x0f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dQ\x1dS\x1dU\x03\x03\xb1\x15\x03\x01\x01\x01\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dW\x03\x03\xb9\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f'!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x13\x07\xc3\xcf\xd1\x11\x03\r\x11\x03\x11\x11\x03\x15\x15\r\t\r\r\r\r\r\x03\xc5\x05\xd3\xd5\x01\x01\x01\x01\x01\x11\x03\x05\x11\x03\t\x13\x07\xc3\xc7\xc9\x13\x05\xc3\xcb\x15\r\t\r\r\r\r\r\x05\xc5\xd9\x03\xdb\x01\x01\x01\x01\x01\x13\x05\xc3\xc7\x13\x07\xc3\xc9\xcb\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x0b)\x05\r\r\x07)\x01\x07)\x01\t\x13\x01\x11\x01\x05\x05\x05)\x03I\t)\x05\t\r\t)\x03\r\x11)\x03\t\x11)\x03\r\x07)\x03\x01\x07)\x05\r\r\x13)\x07\t\r\r\x13)\x03\t\x07\x04^\x02\x05\x01Q\x0b\x1d\x01\x07\x046\x02\x03\x01\x05\x0bP\x0b\x03\x07\x04\n\x02\x03!A\x07B)\x05\x03\r\x07B\x0b\x07\x03\x0f\x03B-\t\x03\x17\r\x063\x03\x05\x03\x05\tg=;\x0b\x05\x05\x19\x03\x07\x03\x01\t\x03\x01\x0fFE\r\x03\x05\x05\t\x03\tGOM\x0f\x03\x05\x05\r\x0b\x03B\x17\t\x03\x0b\x05F\x19\x11\x03\x0b\x03\x01\x11\x06\x19\x03\x0b\x05\x11\x13\x03B\x17\x13\x03\x0b\x13Fa\x15\x03#\x05\x15\x17\x05F\x1b\x17\x03%\x03\x19\x05F\x1b\x11\x03\x05\x03\x03\x15\x06o\x03\x05\x07\x1b\x1d\t\x17\x04w\x05\x0f\x1f\x06\x03\x01\x05\x01\x00F\tY//\x05\x1f\x0f\x0b\x15\x15!\x13#\x07\t)\t\r\x11\x07\x19%)9%3)\x0bw\x1d\x03\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00\x00jit(<lambda>)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00iota\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00qr\x00reshape\x00geqrf\x00pad\x00householder_product\x00add\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x001\x00oneapisolver_geqrf_ffi\x00oneapisolver_orgqr_ffi\x00\x08_\x19\x05O\x01\x0b{\x95\x97\xa1\xa3\x03\xa5\x03\xa7\x03\x81\x11\x87\x89\xad{\x8b\x8d\xaf\x91\x07}}}\x11\x87\x89\xb5{\x8b\x91\xb7\x8d\x03\x93\x03\xbb\x05\xbd\xbf\x03\xc1",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["c64"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_geqrf_ffi', 'oneapisolver_orgqr_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(),
    expected_outputs=(array([[[ 0.        +0.j,  0.9128705 +0.j,  0.40824863+0.j],
        [-0.44721356-0.j,  0.36514878+0.j, -0.8164964 +0.j],
        [-0.8944271 -0.j, -0.18257454+0.j,  0.4082481 +0.j]],

       [[-0.42426407+0.j,  0.8082889 +0.j,  0.4082506 +0.j],
        [-0.5656854 -0.j,  0.1154726 +0.j, -0.8164962 +0.j],
        [-0.7071067 -0.j, -0.57735157+0.j,  0.4082465 +0.j]]],
      dtype=complex64), array([[[-6.7082043e+00+0.j, -8.0498438e+00+0.j, -9.3914852e+00+0.j],
        [ 0.0000000e+00+0.j,  1.0954436e+00+0.j,  2.1908889e+00+0.j],
        [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  4.8322340e-07+0.j]],

       [[-2.1213205e+01+0.j, -2.2910259e+01+0.j, -2.4607315e+01+0.j],
        [ 0.0000000e+00+0.j,  3.4640834e-01+0.j,  6.9281793e-01+0.j],
        [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  6.9880576e-07+0.j]]],
      dtype=complex64)),
    mlir_module_text=r"""
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<2x3x3xcomplex<f32>> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i64> loc(#loc17)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xcomplex<f32>> loc(#loc18)
    %1 = stablehlo.reshape %0 : (tensor<18xcomplex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc19)
    %2:2 = stablehlo.custom_call @oneapisolver_geqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xcomplex<f32>>) -> (tensor<2x3x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) loc(#loc28)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc29)
    %4 = stablehlo.custom_call @oneapisolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, m, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc30)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi64> loc(#loc31)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<3x3xi64> loc(#loc32)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi64> loc(#loc32)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi64> loc(#loc31)
    %9 = stablehlo.compare GE, %7, %8, SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1> loc(#loc33)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc34)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc34)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xcomplex<f32>> loc(#loc35)
    return %4, %12 : tensor<2x3x3xcomplex<f32>>, tensor<2x3x3xcomplex<f32>> loc(#loc16)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":500:11)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":499:26)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":499:14)
#loc4 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":543:4)
#loc5 = loc("jit(<lambda>)"(#loc1))
#loc6 = loc("jit(<lambda>)"(#loc2))
#loc7 = loc("jit(<lambda>)"(#loc3))
#loc8 = loc("geqrf"(#loc1))
#loc9 = loc("pad"(#loc1))
#loc10 = loc("householder_product"(#loc1))
#loc11 = loc("iota"(#loc1))
#loc12 = loc("add"(#loc1))
#loc13 = loc("ge"(#loc1))
#loc14 = loc("broadcast_in_dim"(#loc1))
#loc15 = loc("select_n"(#loc1))
#loc16 = loc("jit(<lambda>)"(#loc4))
#loc17 = loc("qr"(#loc5))
#loc18 = loc("iota"(#loc6))
#loc19 = loc("reshape"(#loc7))
#loc20 = loc(callsite(#loc8 at #loc5))
#loc21 = loc(callsite(#loc9 at #loc5))
#loc22 = loc(callsite(#loc10 at #loc5))
#loc23 = loc(callsite(#loc11 at #loc5))
#loc24 = loc(callsite(#loc12 at #loc5))
#loc25 = loc(callsite(#loc13 at #loc5))
#loc26 = loc(callsite(#loc14 at #loc5))
#loc27 = loc(callsite(#loc15 at #loc5))
#loc28 = loc(""(#loc20))
#loc29 = loc(""(#loc21))
#loc30 = loc(""(#loc22))
#loc31 = loc(""(#loc23))
#loc32 = loc(""(#loc24))
#loc33 = loc(""(#loc25))
#loc34 = loc(""(#loc26))
#loc35 = loc(""(#loc27))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03>\x02\xdd+\x01{\x0f\x17\x0b\x0b\x0b\x07\x0f\x0b\x0b\x0b\x0b\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0f\x17\x0f\x0b\x0f\x17#\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b#\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x17\x03I\x0b/\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b//\x0b\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x05\x1b\x0f\x17\x0f\x0f\x0fK\x0f\x0f\x17\x13K\x13\x17\x01\x05\x0b\x0f\x03'\x1b\x07\x0b\x17\x0f\x0f\x07\x07\x17\x07\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02\x06\x08\x1d\x07\x03\x17\t\xd2\x07\x17\x05\x1f\x05!\x05#\x1f\x11\x03\x05\x05%\x05'\x05)\x05+\x1d\x05W\x1d\x05[\x1d\x05i\x03\x07\x1f!#\r%\r\x05-\x11\x01\x00\x05/\x051\x053\x1d+\x01\x055\x1d\x0f/\x1d\x071\x17\t\xce\x075\x1d57\x057\x1d\x079\x17\t\xce\x07\x1d\x03\x07\x11\x83\x13\x85\x15\xcd\x1d\x05?\x15A\x01\x1dC\x03\x059\x1d\x05G\x15I\x01\x1dK\x03\x05;\x03\x07\x11\x83\x13\x85\x15\xd7\x1d\x05Q\x15S\x01\x1dU\x03\x05=\x15Y\x01\x1d\x0f\x03\x15]\x01\x1d_\x03\x05?\x1d\x05c\x15e\x01\x1dg\x03\x05A\x15k\x01\x1dm\x03\x05C\x1d\x05q\x15s\x01\x1du\x03\x05E\x1d\x07y\x17\t~\x08\t\x03\x01\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dG\x13\x07\x01\r\x01\r\x03\xa9\xab\x0b\x03\x1d\x1f\x05\x01\x03\x03\x8f\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05\x8f\xb3\x1f#\x01#\x15\x03\x05\x99\x9d\r\x03\x7f\x9b\x1dI\r\x03\x7f\x9f\x1dK\x1dM\x1dO\x1f\r\x11\xff\xff\xff\xff\xff\xff\xff\xff\x1f\x0f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dQ\x1dS\x1dU\x03\x03\xb1\x15\x03\x01\x01\x01\x1f\x1f!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dW\x03\x03\xb9\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x13\x07\xc3\xcf\xd1\x11\x03\r\x11\x03\x11\x11\x03\x15\x15\r\t\r\r\r\r\r\x03\xc5\x05\xd3\xd5\x01\x01\x01\x01\x01\x11\x03\x05\x11\x03\t\x13\x07\xc3\xc7\xc9\x13\x05\xc3\xcb\x15\r\t\r\r\r\r\r\x05\xc5\xd9\x03\xdb\x01\x01\x01\x01\x01\x13\x05\xc3\xc7\x13\x07\xc3\xc9\xcb\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x03\x17)\x05\r\r\x07)\x01\x07)\x01\t\x13\x01\x11\x01\x05\x05\x05\t)\x03I\t)\x05\t\r\t)\x03\r\x11)\x03\t\x11)\x03\r\x07)\x03\x01\x07)\x05\r\r\x13)\x07\t\r\r\x13)\x03\t\x07\x04^\x02\x05\x01Q\x0b\x1d\x01\x07\x046\x02\x03\x01\x05\x0bP\x0b\x03\x07\x04\n\x02\x03!A\x07B)\x05\x03\r\x07B\x0b\x07\x03\x0f\x03B-\t\x03\x19\r\x063\x03\x05\x03\x05\tg=;\x0b\x05\x05\x1b\x03\x07\x03\x01\t\x03\x01\x0fFE\r\x03\x05\x05\t\x03\tGOM\x0f\x03\x05\x05\r\x0b\x03B\x17\t\x03\x0b\x05F\x19\x11\x03\x0b\x03\x01\x11\x06\x19\x03\x0b\x05\x11\x13\x03B\x17\x13\x03\x0b\x13Fa\x15\x03%\x05\x15\x17\x05F\x1b\x17\x03'\x03\x19\x05F\x1b\x11\x03\x05\x03\x03\x15\x06o\x03\x05\x07\x1b\x1d\t\x17\x04w\x05\x0f\x1f\x06\x03\x01\x05\x01\x00F\tY//\x05\x1f\x0f\x0b\x15\x15!\x13#\x07\t)\t\r\x11\x07\x19%)9%3)\x0bw\x1d\x03\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00\x00jit(<lambda>)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00iota\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00qr\x00reshape\x00geqrf\x00pad\x00householder_product\x00add\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x001\x00oneapisolver_geqrf_ffi\x00oneapisolver_orgqr_ffi\x00\x08_\x19\x05O\x01\x0b{\x95\x97\xa1\xa3\x03\xa5\x03\xa7\x03\x81\x11\x87\x89\xad{\x8b\x8d\xaf\x91\x07}}}\x11\x87\x89\xb5{\x8b\x91\xb7\x8d\x03\x93\x03\xbb\x05\xbd\xbf\x03\xc1",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["c128"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_geqrf_ffi', 'oneapisolver_orgqr_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(),
    expected_outputs=(array([[[ 0.                 +0.j,  0.9128709291752773 +0.j,
          0.408248290463862  +0.j],
        [-0.447213595499958  -0.j,  0.36514837167011   +0.j,
         -0.8164965809277264 +0.j],
        [-0.894427190999916  -0.j, -0.18257418583505472+0.j,
          0.4082482904638633 +0.j]],

       [[-0.42426406871192857+0.j,  0.8082903768654771 +0.j,
          0.4082482904638615 +0.j],
        [-0.565685424949238  -0.j,  0.11547005383792358+0.j,
         -0.8164965809277263 +0.j],
        [-0.7071067811865476 -0.j, -0.5773502691896248 +0.j,
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
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<2x3x3xcomplex<f64>> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i64> loc(#loc17)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xcomplex<f64>> loc(#loc18)
    %1 = stablehlo.reshape %0 : (tensor<18xcomplex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc19)
    %2:2 = stablehlo.custom_call @oneapisolver_geqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xcomplex<f64>>) -> (tensor<2x3x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) loc(#loc28)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc29)
    %4 = stablehlo.custom_call @oneapisolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, m, n]) {i=2, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<2x3x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc30)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi64> loc(#loc31)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<3x3xi64> loc(#loc32)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi64> loc(#loc32)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi64> loc(#loc31)
    %9 = stablehlo.compare GE, %7, %8, SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1> loc(#loc33)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc34)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc34)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xcomplex<f64>> loc(#loc35)
    return %4, %12 : tensor<2x3x3xcomplex<f64>>, tensor<2x3x3xcomplex<f64>> loc(#loc16)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":500:11)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":499:26)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":499:14)
#loc4 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":543:4)
#loc5 = loc("jit(<lambda>)"(#loc1))
#loc6 = loc("jit(<lambda>)"(#loc2))
#loc7 = loc("jit(<lambda>)"(#loc3))
#loc8 = loc("geqrf"(#loc1))
#loc9 = loc("pad"(#loc1))
#loc10 = loc("householder_product"(#loc1))
#loc11 = loc("iota"(#loc1))
#loc12 = loc("add"(#loc1))
#loc13 = loc("ge"(#loc1))
#loc14 = loc("broadcast_in_dim"(#loc1))
#loc15 = loc("select_n"(#loc1))
#loc16 = loc("jit(<lambda>)"(#loc4))
#loc17 = loc("qr"(#loc5))
#loc18 = loc("iota"(#loc6))
#loc19 = loc("reshape"(#loc7))
#loc20 = loc(callsite(#loc8 at #loc5))
#loc21 = loc(callsite(#loc9 at #loc5))
#loc22 = loc(callsite(#loc10 at #loc5))
#loc23 = loc(callsite(#loc11 at #loc5))
#loc24 = loc(callsite(#loc12 at #loc5))
#loc25 = loc(callsite(#loc13 at #loc5))
#loc26 = loc(callsite(#loc14 at #loc5))
#loc27 = loc(callsite(#loc15 at #loc5))
#loc28 = loc(""(#loc20))
#loc29 = loc(""(#loc21))
#loc30 = loc(""(#loc22))
#loc31 = loc(""(#loc23))
#loc32 = loc(""(#loc24))
#loc33 = loc(""(#loc25))
#loc34 = loc(""(#loc26))
#loc35 = loc(""(#loc27))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01+\x07\x01\x05\t\x19\x01\x03\x0f\x03\x17\x13\x17\x1b\x1f#'+/37;\x03>\x02\xdd+\x01{\x0f\x17\x0b\x0b\x0b\x07\x0f\x0b\x0b\x0b\x0b\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0f\x17\x0f\x0b\x0f\x17#\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b#\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x17\x03I\x0b/\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b/O\x0b\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x05\x1b\x0f\x17\x0f\x0f\x0fK\x0f\x0f\x17\x13K\x13\x17\x01\x05\x0b\x0f\x03'\x1b\x07\x0b\x17\x0f\x0f\x07\x07\x17\x07\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02&\x08\x1d\x07\x03\x17\t\xd2\x07\x17\x05\x1f\x05!\x05#\x1f\x11\x03\x05\x05%\x05'\x05)\x05+\x1d\x05W\x1d\x05[\x1d\x05i\x03\x07\x1f!#\r%\r\x05-\x11\x01\x00\x05/\x051\x053\x1d+\x01\x055\x1d\x0f/\x1d\x071\x17\t\xce\x075\x1d57\x057\x1d\x079\x17\t\xce\x07\x1d\x03\x07\x11\x83\x13\x85\x15\xcd\x1d\x05?\x15A\x01\x1dC\x03\x059\x1d\x05G\x15I\x01\x1dK\x03\x05;\x03\x07\x11\x83\x13\x85\x15\xd7\x1d\x05Q\x15S\x01\x1dU\x03\x05=\x15Y\x01\x1d\x0f\x03\x15]\x01\x1d_\x03\x05?\x1d\x05c\x15e\x01\x1dg\x03\x05A\x15k\x01\x1dm\x03\x05C\x1d\x05q\x15s\x01\x1du\x03\x05E\x1d\x07y\x17\t~\x08\t\x03\x01\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dG\x13\x07\x01\r\x01\r\x03\xa9\xab\x0b\x03\x1d\x1f\x05\x01\x03\x03\x8f\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05\x8f\xb3\x1f#\x01#\x15\x03\x05\x99\x9d\r\x03\x7f\x9b\x1dI\r\x03\x7f\x9f\x1dK\x1dM\x1dO\x1f\r\x11\xff\xff\xff\xff\xff\xff\xff\xff\x1f\x0f!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dQ\x1dS\x1dU\x03\x03\xb1\x15\x03\x01\x01\x01\x1f\x1f!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dW\x03\x03\xb9\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x13\x07\xc3\xcf\xd1\x11\x03\r\x11\x03\x11\x11\x03\x15\x15\r\t\r\r\r\r\r\x03\xc5\x05\xd3\xd5\x01\x01\x01\x01\x01\x11\x03\x05\x11\x03\t\x13\x07\xc3\xc7\xc9\x13\x05\xc3\xcb\x15\r\t\r\r\r\r\r\x05\xc5\xd9\x03\xdb\x01\x01\x01\x01\x01\x13\x05\xc3\xc7\x13\x07\xc3\xc9\xcb\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x03\x17)\x05\r\r\x07)\x01\x07)\x01\t\x13\x01\x11\x01\x05\x05\x05\x0b)\x03I\t)\x05\t\r\t)\x03\r\x11)\x03\t\x11)\x03\r\x07)\x03\x01\x07)\x05\r\r\x13)\x07\t\r\r\x13)\x03\t\x07\x04^\x02\x05\x01Q\x0b\x1d\x01\x07\x046\x02\x03\x01\x05\x0bP\x0b\x03\x07\x04\n\x02\x03!A\x07B)\x05\x03\r\x07B\x0b\x07\x03\x0f\x03B-\t\x03\x19\r\x063\x03\x05\x03\x05\tg=;\x0b\x05\x05\x1b\x03\x07\x03\x01\t\x03\x01\x0fFE\r\x03\x05\x05\t\x03\tGOM\x0f\x03\x05\x05\r\x0b\x03B\x17\t\x03\x0b\x05F\x19\x11\x03\x0b\x03\x01\x11\x06\x19\x03\x0b\x05\x11\x13\x03B\x17\x13\x03\x0b\x13Fa\x15\x03%\x05\x15\x17\x05F\x1b\x17\x03'\x03\x19\x05F\x1b\x11\x03\x05\x03\x03\x15\x06o\x03\x05\x07\x1b\x1d\t\x17\x04w\x05\x0f\x1f\x06\x03\x01\x05\x01\x00F\tY//\x05\x1f\x0f\x0b\x15\x15!\x13#\x07\t)\t\r\x11\x07\x19%)9%3)\x0bw\x1d\x03\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00\x00jit(<lambda>)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00iota\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00qr\x00reshape\x00geqrf\x00pad\x00householder_product\x00add\x00ge\x00broadcast_in_dim\x00select_n\x00jax.result_info\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x001\x00oneapisolver_geqrf_ffi\x00oneapisolver_orgqr_ffi\x00\x08_\x19\x05O\x01\x0b{\x95\x97\xa1\xa3\x03\xa5\x03\xa7\x03\x81\x11\x87\x89\xad{\x8b\x8d\xaf\x91\x07}}}\x11\x87\x89\xb5{\x8b\x91\xb7\x8d\x03\x93\x03\xbb\x05\xbd\xbf\x03\xc1",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
