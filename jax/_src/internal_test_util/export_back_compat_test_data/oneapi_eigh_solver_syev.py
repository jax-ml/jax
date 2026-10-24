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
    custom_call_targets=['oneapisolver_syevd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(),
    expected_outputs=(array([[-0.7941182  , -0.23413022 ,  0.49515998 ,  0.26339257 ],
       [-0.3696446  , -0.05689677 , -0.8347229  ,  0.40418223 ],
       [ 0.054829847,  0.8161838  ,  0.18396728 ,  0.5449714  ],
       [ 0.47930416 , -0.5251569  ,  0.15559603 ,  0.68576074 ]],
      dtype=float32), array([-3.7082880e+00, -9.3887945e-07,  1.1818995e-06,  3.3708298e+01],
      dtype=float32)),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xf32> {jax.result_info = "result[0]"}, tensor<4xf32> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc23)
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32> loc(#loc15)
    %0 = stablehlo.iota dim = 0 : tensor<16xf32> loc(#loc24)
    %1 = stablehlo.reshape %0 : (tensor<16xf32>) -> tensor<4x4xf32> loc(#loc25)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc26)
    %3 = stablehlo.add %1, %2 : tensor<4x4xf32> loc(#loc27)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc28)
    %5 = stablehlo.divide %3, %4 : tensor<4x4xf32> loc(#loc28)
    %6 = call @tril(%5) : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc21)
    %7:3 = stablehlo.custom_call @oneapisolver_syevd_ffi(%6) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=4, j=4, k=4, l=4, m=4}, custom>} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>, tensor<i32>) loc(#loc23)
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc23)
    %9 = stablehlo.compare EQ, %7#2, %8, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc23)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc23)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc23)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc23)
    %13 = stablehlo.select %12, %7#0, %11 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc23)
    %14 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc23)
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32> loc(#loc23)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc23)
    %17 = stablehlo.select %16, %7#1, %15 : tensor<4xi1>, tensor<4xf32> loc(#loc23)
    return %13, %17 : tensor<4x4xf32>, tensor<4xf32> loc(#loc15)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xf32> loc(unknown)) -> tensor<4x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi64> loc(#loc9)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4xi64> loc(#loc10)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi64> loc(#loc10)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi64> loc(#loc9)
    %4 = stablehlo.compare GE, %2, %3, SIGNED : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1> loc(#loc11)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc12)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc13)
    return %6 : tensor<4x4xf32> loc(#loc)
  } loc(#loc29)
} loc(#loc)
#loc1 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":343:11)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":410:4)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":335:26)
#loc4 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":335:14)
#loc5 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:34)
#loc6 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:15)
#loc7 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:14)
#loc8 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":343:27)
#loc9 = loc("iota")
#loc10 = loc("add")
#loc11 = loc("ge")
#loc12 = loc("broadcast_in_dim")
#loc13 = loc("select_n")
#loc14 = loc("jit(<lambda>)"(#loc1))
#loc15 = loc("jit(<lambda>)"(#loc2))
#loc16 = loc("jit(<lambda>)"(#loc3))
#loc17 = loc("jit(<lambda>)"(#loc4))
#loc18 = loc("jit(<lambda>)"(#loc5))
#loc19 = loc("jit(<lambda>)"(#loc6))
#loc20 = loc("jit(<lambda>)"(#loc7))
#loc21 = loc("jit(<lambda>)/jit(tril)"(#loc8))
#loc22 = loc("jit(<lambda>)/jit"(#loc8))
#loc23 = loc("eigh"(#loc14))
#loc24 = loc("iota"(#loc16))
#loc25 = loc("reshape"(#loc17))
#loc26 = loc("transpose"(#loc18))
#loc27 = loc("add"(#loc19))
#loc28 = loc("div"(#loc20))
#loc29 = loc("jit:"(#loc22))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01/\x07\x01\x05\t\x1d\x01\x03\x0f\x03\x1b\x13\x17\x1b\x1f#'+/37;?C\x03f\x02\xe39\x01o\x07\x0f\x0b\x0b\x0f\x17\x0f\x0b\x0f\x0b\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x17\x17\x0f\x0f\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17\x0f\x0f\x17\x0b\x0f\x17\x0f\x0b#\x0b\x0b\x0b\x03a\x0f\x0b\x0b\x0f\x0b\x0f\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x1f/\x0f\x0b\x1f\x1f\x1fO\x1b\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x0bO/\x05\x15K\x13\x0f\x0f\x13\x0f\x0f\x0f\x0f\x0b\x01\x05\x0b\x0f\x035\x17\x07\x0f\x07\x07\x13\x17\x0f\x07\x0f\x17\x13\x17\x17\x13\x07\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02>\x08\x1f\x1d9;\x05#\x05%\x11\x03\x05\x17\x05^\x057\x1d\x0f\x01\x05'\x1d\x13\x01\x05)\x1d\x07?\x1d]_\x03\x07\x1b\x1d\x1f\t!\t\x05+\x11\x01\x00\x05-\x05/\x051\x1d')\x053\x1d+\x0b\x055\x1d/\x01\x057\x1d3\x01\x059\x1d7\x01\x05;\x05=\x1d\x07=\x17\x05^\x05\x17\x17\x05j\x06\t\x1d\x0fC\x1d\x07E\x17\x05>\x055\x1dIK\x05?\x1d\x07M\x17\x05>\x05\x1d\x1dQS\x05A\x1d\x07U\x17\x05F\x05E\x1d\x13Y\x1d\x07[\x17\x05F\x05\x1f\x05C\x1d\x07a\x17\x05F\x05\x1d\x1de\x0b\x05E\x03\x07i\xa5k\xafm\xcf\x05G\x05I\x05K\x1f!\x01\x03\x01\x1dM\x03\x03\x8f\x1dO\x13\x07\x01\t\x07\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1d\x03\x05\x83\x87\r\x03s\x85\x1dQ\r\x03s\x89\x1dS\x1dU\x1dW\r\x01#\x1f\x1dY\x1f\t\t\x00\x00\x00\x00\x1f\x17\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x07\x05\x07\x05\x1f\t\t\x00\x00\xc0\x7f\x1f\x13\t\x00\x00\x00\x00\x1f\t\t\x00\x00\x00@\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\xa7\xa9\xab\xad\x1d[\x13'\x00\x1d]\x05\x03\r\x03\xb1\xb3\x1d_\x1da\x0b\x03\x1dc\x1de\x05\x01\x03\x03}\x03\x03\xc1\x15\x03\x01\x01\x01\x03\x07}\xc5\xc7\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x01\x07\x01\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x15\x0b\x11\x11\x11\x11\x11\x03\xd1\x07\xd7\xdd\xe1\x01\x01\x01\x01\x01\x13\x05\xd3\xd5\x11\x03\x01\x11\x03\x05\x13\x05\xd9\xdb\x11\x03\t\x11\x03\r\x13\x03\xdf\x11\x03\x11\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\r\x1d)\x01\r\x01\t)\x03\x11\r)\x05\x11\x11\x07)\x01#\x13)\x01\x07)\x05\x11\x11\x0b)\x03\t\x07\x11\x01\x05\x05\x0f\x11\x03\x05\x03\x05)\x03\x01\x07\x1b)\x03A\r!)\x03\t\x15)\x03\x05\x15)\x03\x01\x15)\x01\x0b)\x05\x05\x05\x0b)\x03\x05\x0b)\x03\x11\x0b)\x03\x05\x07\x04^\x04\x05\x01Q\x01\x19\x01\x07\x046\x04\x03\x01\t\x0bP\x01\x03\x07\x04\xba\x02\x03/Y\x05B\x01\x05\x03\t\x05B\x03\x07\x03\x13\x05B\x15\t\x03\t\x07BA\x0b\x03%\x13\x06G\x03\x05\x03\x07\x15FO\r\x03\x05\x03\t\r\x06W\x03\x05\x05\t\x0b\x03F\x17\x0f\x03\x05\x03\x05\x17\x06\x17\x03\x05\x05\r\x0f\x19Fc\x11\x03\x05\x03\x11\x1bG\x03g\x13\x07\x05\x0f\x13\x03\x13\x03F\x03\x0f\x03\x13\x03\x03\x0fF\x03\x15\x03/\x05\x19\x1b\x03F\x03\x0f\x031\x03\x1d\x03F\x03\x0f\x03\x05\x03\x01\x03F\x03\x17\x03\x19\x03\x1f\t\x06\x03\x03\x05\x07#\x15!\x03F\x03\x0f\x033\x03\x1d\x03F\x03\x0f\x03\x0f\x03\x01\x03F\x03\x19\x035\x03'\t\x06\x03\x03\x0f\x07+\x17)\x11\x04\x15\x05%-\x0bP%\x1b\x07\x04\x9b\x03\x15+\x03\t\x00\x05B\x01\x1d\x03\t\x05B\x01\x1f\x03\x17\x07B\r\x0b\x03\x11\x03F\x11\x0f\x03\x11\x03\x05\r\x06\x11\x03\x11\x05\x07\t\x07B\r!\x03\x11\x0fF-#\x03\x19\x05\x0b\r\x03F1\x0f\x03\x05\x03\x03\t\x065\x03\x05\x07\x0f\x01\x11\x11\x04\x01\x03\x13\x06\x03\x01\x05\x01\x00^\ng/\x03\x05\x1f\r\x15\x11\x0f\x0b\x15\x15\x0b!%3)1\t\x15\x11\x0b\x13#\x07%\x0b\x19%)9\t\x0b\x1dw\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jit(<lambda>)\x00iota\x00add\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00jit:\x00jit(<lambda>)/jit\x00ge\x00broadcast_in_dim\x00select_n\x00eigh\x00reshape\x00transpose\x00div\x00jit(<lambda>)/jit(tril)\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00tril\x00result[0]\x00result[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00num_batch_dims\x000\x00\x00oneapisolver_syevd_ffi\x00\x08o%\x05G\x01\x0bq\x7f\x81\x8b\x8d\x03\x9d\x03\x9f\x03\xa1\x03y\x03\xa3\x03o\x03w\x11\xb5\xb7\xb9q\xbb\xbd\xbf\xc3\x05{\xc9\x03\xcb\x03\xcd\x0bu\x91uw\x93\x03\x95\x03\x97\x03\x99\x05{\x9b",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["f64"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_syevd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(),
    expected_outputs=(array([[-0.7941185704969036  , -0.030864422973121777,
        -0.5468522537162451  ,  0.2633926650306618  ],
       [-0.36964433974346067 , -0.36644704035533193 ,
         0.7521413209063954  ,  0.40418196656409766 ],
       [ 0.05482989100998296 ,  0.8254873496300309  ,
         0.13627411933594374 ,  0.5449712680975332  ],
       [ 0.4793041217634258  , -0.4281758863015766  ,
        -0.3415631865260938  ,  0.6857605696309692  ]]), array([-3.7082869338697062e+00, -7.8262818920584397e-15,
       -2.6954043318554572e-16,  3.3708286933869715e+01])),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xf64> {jax.result_info = "result[0]"}, tensor<4xf64> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc23)
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64> loc(#loc15)
    %0 = stablehlo.iota dim = 0 : tensor<16xf64> loc(#loc24)
    %1 = stablehlo.reshape %0 : (tensor<16xf64>) -> tensor<4x4xf64> loc(#loc25)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc26)
    %3 = stablehlo.add %1, %2 : tensor<4x4xf64> loc(#loc27)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc28)
    %5 = stablehlo.divide %3, %4 : tensor<4x4xf64> loc(#loc28)
    %6 = call @tril(%5) : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc21)
    %7:3 = stablehlo.custom_call @oneapisolver_syevd_ffi(%6) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=4, j=4, k=4, l=4, m=4}, custom>} : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i32>) loc(#loc23)
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc23)
    %9 = stablehlo.compare EQ, %7#2, %8, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc23)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc23)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc23)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc23)
    %13 = stablehlo.select %12, %7#0, %11 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc23)
    %14 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc23)
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64> loc(#loc23)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc23)
    %17 = stablehlo.select %16, %7#1, %15 : tensor<4xi1>, tensor<4xf64> loc(#loc23)
    return %13, %17 : tensor<4x4xf64>, tensor<4xf64> loc(#loc15)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xf64> loc(unknown)) -> tensor<4x4xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi64> loc(#loc9)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4xi64> loc(#loc10)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi64> loc(#loc10)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi64> loc(#loc9)
    %4 = stablehlo.compare GE, %2, %3, SIGNED : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1> loc(#loc11)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc12)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc13)
    return %6 : tensor<4x4xf64> loc(#loc)
  } loc(#loc29)
} loc(#loc)
#loc1 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":343:11)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":410:4)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":335:26)
#loc4 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":335:14)
#loc5 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:34)
#loc6 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:15)
#loc7 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:14)
#loc8 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":343:27)
#loc9 = loc("iota")
#loc10 = loc("add")
#loc11 = loc("ge")
#loc12 = loc("broadcast_in_dim")
#loc13 = loc("select_n")
#loc14 = loc("jit(<lambda>)"(#loc1))
#loc15 = loc("jit(<lambda>)"(#loc2))
#loc16 = loc("jit(<lambda>)"(#loc3))
#loc17 = loc("jit(<lambda>)"(#loc4))
#loc18 = loc("jit(<lambda>)"(#loc5))
#loc19 = loc("jit(<lambda>)"(#loc6))
#loc20 = loc("jit(<lambda>)"(#loc7))
#loc21 = loc("jit(<lambda>)/jit(tril)"(#loc8))
#loc22 = loc("jit(<lambda>)/jit"(#loc8))
#loc23 = loc("eigh"(#loc14))
#loc24 = loc("iota"(#loc16))
#loc25 = loc("reshape"(#loc17))
#loc26 = loc("transpose"(#loc18))
#loc27 = loc("add"(#loc19))
#loc28 = loc("div"(#loc20))
#loc29 = loc("jit:"(#loc22))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01/\x07\x01\x05\t\x1d\x01\x03\x0f\x03\x1b\x13\x17\x1b\x1f#'+/37;?C\x03f\x02\xe39\x01o\x07\x0f\x0b\x0b\x0f\x17\x0f\x0b\x0f\x0b\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x17\x17\x0f\x0f\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17\x0f\x0f\x17\x0b\x0f\x17\x0f\x0b#\x0b\x0b\x0b\x03a\x0f\x0b\x0b\x0f\x0b\x0f\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b//\x0f\x0b/\x1f/O\x1b\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x0bO/\x05\x15K\x13\x0f\x0f\x13\x0f\x0f\x0f\x0f\x0b\x01\x05\x0b\x0f\x035\x17\x07\x0f\x07\x07\x13\x17\x0f\x07\x0f\x17\x13\x17\x17\x13\x07\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02n\x08\x1f\x1d9;\x05#\x05%\x11\x03\x05\x17\x05^\x057\x1d\x0f\x01\x05'\x1d\x13\x01\x05)\x1d\x07?\x1d]_\x03\x07\x1b\x1d\x1f\t!\t\x05+\x11\x01\x00\x05-\x05/\x051\x1d')\x053\x1d+\x0b\x055\x1d/\x01\x057\x1d3\x01\x059\x1d7\x01\x05;\x05=\x1d\x07=\x17\x05^\x05\x17\x17\x05j\x06\t\x1d\x0fC\x1d\x07E\x17\x05>\x055\x1dIK\x05?\x1d\x07M\x17\x05>\x05\x1d\x1dQS\x05A\x1d\x07U\x17\x05F\x05E\x1d\x13Y\x1d\x07[\x17\x05F\x05\x1f\x05C\x1d\x07a\x17\x05F\x05\x1d\x1de\x0b\x05E\x03\x07i\xa5k\xafm\xcf\x05G\x05I\x05K\x1f!\x01\x03\x01\x1dM\x03\x03\x8f\x1dO\x13\x07\x01\t\x07\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1d\x03\x05\x83\x87\r\x03s\x85\x1dQ\r\x03s\x89\x1dS\x1dU\x1dW\r\x01#\x1f\x1dY\x1f\t\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x17\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x07\x05\x07\x05\x1f\t\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x13\t\x00\x00\x00\x00\x1f\t\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\xa7\xa9\xab\xad\x1d[\x13'\x00\x1d]\x05\x03\r\x03\xb1\xb3\x1d_\x1da\x0b\x03\x1dc\x1de\x05\x01\x03\x03}\x03\x03\xc1\x15\x03\x01\x01\x01\x03\x07}\xc5\xc7\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x01\x07\x01\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x15\x0b\x11\x11\x11\x11\x11\x03\xd1\x07\xd7\xdd\xe1\x01\x01\x01\x01\x01\x13\x05\xd3\xd5\x11\x03\x01\x11\x03\x05\x13\x05\xd9\xdb\x11\x03\t\x11\x03\r\x13\x03\xdf\x11\x03\x11\x13\x01\x01\t\x01\x02\x02)\x05\x11\x11\r\x1d)\x01\r\x01\x0b)\x03\x11\r)\x05\x11\x11\x07)\x01#\x13)\x01\x07)\x05\x11\x11\x0b)\x03\t\x07\x11\x01\x05\x05\x0f\x11\x03\x05\x03\x05)\x03\x01\x07\x1b)\x03A\r!)\x03\t\x15)\x03\x05\x15)\x03\x01\x15)\x01\x0b)\x05\x05\x05\x0b)\x03\x05\x0b)\x03\x11\x0b)\x03\x05\x07\x04^\x04\x05\x01Q\x01\x19\x01\x07\x046\x04\x03\x01\t\x0bP\x01\x03\x07\x04\xba\x02\x03/Y\x05B\x01\x05\x03\t\x05B\x03\x07\x03\x13\x05B\x15\t\x03\t\x07BA\x0b\x03%\x13\x06G\x03\x05\x03\x07\x15FO\r\x03\x05\x03\t\r\x06W\x03\x05\x05\t\x0b\x03F\x17\x0f\x03\x05\x03\x05\x17\x06\x17\x03\x05\x05\r\x0f\x19Fc\x11\x03\x05\x03\x11\x1bG\x03g\x13\x07\x05\x0f\x13\x03\x13\x03F\x03\x0f\x03\x13\x03\x03\x0fF\x03\x15\x03/\x05\x19\x1b\x03F\x03\x0f\x031\x03\x1d\x03F\x03\x0f\x03\x05\x03\x01\x03F\x03\x17\x03\x19\x03\x1f\t\x06\x03\x03\x05\x07#\x15!\x03F\x03\x0f\x033\x03\x1d\x03F\x03\x0f\x03\x0f\x03\x01\x03F\x03\x19\x035\x03'\t\x06\x03\x03\x0f\x07+\x17)\x11\x04\x15\x05%-\x0bP%\x1b\x07\x04\x9b\x03\x15+\x03\t\x00\x05B\x01\x1d\x03\t\x05B\x01\x1f\x03\x17\x07B\r\x0b\x03\x11\x03F\x11\x0f\x03\x11\x03\x05\r\x06\x11\x03\x11\x05\x07\t\x07B\r!\x03\x11\x0fF-#\x03\x19\x05\x0b\r\x03F1\x0f\x03\x05\x03\x03\t\x065\x03\x05\x07\x0f\x01\x11\x11\x04\x01\x03\x13\x06\x03\x01\x05\x01\x00^\ng/\x03\x05\x1f\r\x15\x11\x0f\x0b\x15\x15\x0b!%3)1\t\x15\x11\x0b\x13#\x07%\x0b\x19%)9\t\x0b\x1dw\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jit(<lambda>)\x00iota\x00add\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00jit:\x00jit(<lambda>)/jit\x00ge\x00broadcast_in_dim\x00select_n\x00eigh\x00reshape\x00transpose\x00div\x00jit(<lambda>)/jit(tril)\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00tril\x00result[0]\x00result[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00num_batch_dims\x000\x00\x00oneapisolver_syevd_ffi\x00\x08o%\x05G\x01\x0bq\x7f\x81\x8b\x8d\x03\x9d\x03\x9f\x03\xa1\x03y\x03\xa3\x03o\x03w\x11\xb5\xb7\xb9q\xbb\xbd\xbf\xc3\x05{\xc9\x03\xcb\x03\xcd\x0bu\x91uw\x93\x03\x95\x03\x97\x03\x99\x05{\x9b",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["c64"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_syevd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(),
    expected_outputs=(array([[-0.79411834 +0.j,  0.14991982 -0.j, -0.5268058  +0.j,
         0.26339254 +0.j],
       [-0.3696446  +0.j,  0.19276462 +0.j,  0.81415087 +0.j,
         0.40418208 +0.j],
       [ 0.05482984 +0.j, -0.8352885  +0.j, -0.047885474+0.j,
         0.5449712  +0.j],
       [ 0.47930416 +0.j,  0.49260414 +0.j, -0.23946011 +0.j,
         0.6857605  +0.j]], dtype=complex64), array([-3.7082870e+00, -3.4622639e-07,  3.5462003e-06,  3.3708294e+01],
      dtype=float32)),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<4xf32> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc29)
    %cst_0 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc29)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc29)
    %cst_1 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc16)
    %0 = stablehlo.iota dim = 0 : tensor<16xcomplex<f32>> loc(#loc30)
    %1 = stablehlo.reshape %0 : (tensor<16xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc31)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc32)
    %3 = stablehlo.real %2 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc40)
    %4 = stablehlo.imag %2 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc41)
    %5 = stablehlo.negate %4 : tensor<4x4xf32> loc(#loc42)
    %6 = stablehlo.complex %3, %5 : tensor<4x4xcomplex<f32>> loc(#loc43)
    %7 = stablehlo.add %1, %6 : tensor<4x4xcomplex<f32>> loc(#loc37)
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc38)
    %9 = stablehlo.divide %7, %8 : tensor<4x4xcomplex<f32>> loc(#loc38)
    %10 = call @tril(%9) : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc27)
    %11:3 = stablehlo.custom_call @oneapisolver_syevd_ffi(%10) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=4, j=4, k=4, l=4, m=4}, custom>} : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4xf32>, tensor<i32>) loc(#loc29)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc29)
    %13 = stablehlo.compare EQ, %11#2, %12, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc29)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc29)
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc29)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc29)
    %17 = stablehlo.select %16, %11#0, %15 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc29)
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc29)
    %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32> loc(#loc29)
    %20 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc29)
    %21 = stablehlo.select %20, %11#1, %19 : tensor<4xi1>, tensor<4xf32> loc(#loc29)
    return %17, %21 : tensor<4x4xcomplex<f32>>, tensor<4xf32> loc(#loc16)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xcomplex<f32>> loc(unknown)) -> tensor<4x4xcomplex<f32>> {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi64> loc(#loc10)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4xi64> loc(#loc11)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi64> loc(#loc11)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi64> loc(#loc10)
    %4 = stablehlo.compare GE, %2, %3, SIGNED : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1> loc(#loc12)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc13)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc14)
    return %6 : tensor<4x4xcomplex<f32>> loc(#loc)
  } loc(#loc39)
} loc(#loc)
#loc1 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":343:11)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":410:4)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":335:26)
#loc4 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":335:14)
#loc5 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:34)
#loc6 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:25)
#loc7 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:15)
#loc8 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:14)
#loc9 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":343:27)
#loc10 = loc("iota")
#loc11 = loc("add")
#loc12 = loc("ge")
#loc13 = loc("broadcast_in_dim")
#loc14 = loc("select_n")
#loc15 = loc("jit(<lambda>)"(#loc1))
#loc16 = loc("jit(<lambda>)"(#loc2))
#loc17 = loc("jit(<lambda>)"(#loc3))
#loc18 = loc("jit(<lambda>)"(#loc4))
#loc19 = loc("jit(<lambda>)"(#loc5))
#loc20 = loc("real"(#loc6))
#loc21 = loc("jit(<lambda>)"(#loc6))
#loc22 = loc("imag"(#loc6))
#loc23 = loc("neg"(#loc6))
#loc24 = loc("complex"(#loc6))
#loc25 = loc("jit(<lambda>)"(#loc7))
#loc26 = loc("jit(<lambda>)"(#loc8))
#loc27 = loc("jit(<lambda>)/jit(tril)"(#loc9))
#loc28 = loc("jit(<lambda>)/jit"(#loc9))
#loc29 = loc("eigh"(#loc15))
#loc30 = loc("iota"(#loc17))
#loc31 = loc("reshape"(#loc18))
#loc32 = loc("transpose"(#loc19))
#loc33 = loc(callsite(#loc20 at #loc21))
#loc34 = loc(callsite(#loc22 at #loc21))
#loc35 = loc(callsite(#loc23 at #loc21))
#loc36 = loc(callsite(#loc24 at #loc21))
#loc37 = loc("add"(#loc25))
#loc38 = loc("div"(#loc26))
#loc39 = loc("jit:"(#loc28))
#loc40 = loc(""(#loc33))
#loc41 = loc(""(#loc34))
#loc42 = loc(""(#loc35))
#loc43 = loc(""(#loc36))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x017\x07\x01\x05\t%\x01\x03\x0f\x03#\x13\x17\x1b\x1f#'+/37;?CGKOS\x03\xce\x02\x16\x02?\x01\x95\x0f\x07\x0b\x0b\x17\x0b\x0f\x0f\x17\x0f\x0b\x0f\x0b\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x17\x17\x0f\x0f\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x0f\x17\x0b\x0f\x17\x0f\x0b#\x0b\x0b\x0b\x03Y\x0f\x0b\x0b\x0f\x0b\x0f\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b//\x0f\x0b\x1f/\x1f/O\x1b\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f\x05\x15K\x13\x0f\x0f\x13\x0f\x0f\x0f\x0f\x0b\x03\x0b/\x0f\x0bO/\x01\x05\x0b\x0f\x03;\x17\x07\x0f\x07\x07\x13\x17\x0f\x0b\x17\x07\x0f\x17\x0f\x13\x17\x17\x13\x07\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02\x92\t\x1d?A\x1f\x05+\x05-\x17\x05F\x053\x05/\x1d\x07\t\x11\x03\x05\x17\x05^\x057\x1d\x15\x03\x051\x1d\x19\x03\x053\x1d\x07E\x1d\x83\x85\x03\x07!#%\x0f'\x0f\x055\x11\x01\x00\x057\x059\x05;\x1d-/\x05=\x1d1\x11\x05?\x1d5\x03\x05A\x1d9\x03\x05C\x1d=\x03\x05E\x05G\x1d\x07C\x17\x05^\x05\x17\x17\x05j\x06\t\x1d\x15I\x1d\x07K\x17\x05>\x055\x1dOQ\x05I\x1d\x07S\x17\x05>\x05\x1d\x1dWY\x05K\x1d\x07[\x17\x05F\x05E\x1d\x0b_\x15a\r\x1dc\t\x05M\x1d\x0bg\x15i\r\x1dk\t\x05O\x1d\x0bo\x15q\r\x1ds\t\x05Q\x1d\x0bw\x15y\r\x1d{\t\x05S\x1d\x19\x7f\x1d\x07\x81\x17\x05F\x05\x1f\x05U\x1d\x07\x87\x17\x05F\x05\x1d\x1d\x8b\x11\x05W\x03\x07\x8f\xcd\x91\xd7\x93\xed\x05Y\x05[\x05]\x1f'\x01\x03\x01\x1d_\x03\x03\xb5\x1da\x13\x07\x01\t\x07\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00##\x03\x05\xa9\xad\r\x03\x99\xab\x1dc\r\x03\x99\xaf\x1de\x1dg\x1di\r\x01#%\x1dk\x1f\t\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x07\x05\x07\x05\x1f\x1f\t\x00\x00\xc0\x7f\x1f\t\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x13\t\x00\x00\x00\x00\x1f\t\x11\x00\x00\x00@\x00\x00\x00\x00\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\xcf\xd1\xd3\xd5\x1dm\x13-\x00\x1do\x05\x03\r\x03\xd9\xdb\x1dq\x1ds\x0b\x03\x1d/\x1du\x05\x01\x03\x03\xa3\x03\x03\xe9\x15\x03\x01\x01\x01\x03\x07\xa3\x02\x02\x06\x02\x15\x0b\x11\x11\x11\x11\x11\x03\xef\x07\xf5\xfb\xff\x01\x01\x01\x01\x01\x13\x05\xf1\xf3\x11\x03\x01\x11\x03\x05\x13\x05\xf7\xf9\x11\x03\t\x11\x03\r\x13\x03\xfd\x11\x03\x11\x13\x01\x1f1\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f3\x01\x07\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x15\x1d)\x01\x15\x01\t)\x03\x11\r)\x05\x11\x11\x07)\x01)\x03\r)\x05\x11\x11\r\x13)\x01\x07)\x05\x11\x11\x0b)\x01\r)\x03\t\x07\x11\x01\x05\x05\x0f\x11\x03\x05\x03\x05)\x03\x01\x07\x1b)\x03A\x15!)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x01\x0b)\x05\x05\x05\x0b)\x03\x05\x0b)\x03\x11\x0b)\x03\x05\x07\x04\xea\x04\x05\x01Q\x03\x1f\x01\x07\x04\xc2\x04\x03\x01\t\x0bP\x03\x03\x07\x04F\x03\x039m\x05B\x01\x05\x03\x1f\x05B\x01\x07\x03\t\x05B\x01\t\x03\x13\x05B\x1b\x0b\x03\t\x07BG\r\x03+\x13\x06M\x03\x05\x03\t\x15FU\x0f\x03\x05\x03\x0b\x17\x06]\x03\x17\x03\r\x19\x06e\x03\x17\x03\r\x1b\x06m\x03\x17\x03\x11\x1d\x06u\x03\x05\x05\x0f\x13\r\x06}\x03\x05\x05\x0b\x15\x03F\x1d\x11\x03\x05\x03\x07\x1f\x06\x1d\x03\x05\x05\x17\x19!F\x89\x13\x03\x05\x03\x1b#G\x01\x8d\x15\x07\x05\x0f\x13\x03\x1d\x03F\x01\x11\x03\x13\x03\x05\x0fF\x01\x17\x035\x05#%\x03F\x01\x11\x037\x03'\x03F\x01\x11\x03\x05\x03\x03\x03F\x01\x19\x03\x1d\x03)\t\x06\x01\x03\x05\x07-\x1f+\x03F\x01\x11\x039\x03'\x03F\x01\x11\x03\x0f\x03\x01\x03F\x01\x1b\x03;\x031\t\x06\x01\x03\x0f\x075!3\x11\x04\x1b\x05/7\x0bP+\x1d\x07\x04\x9b\x03\x15+\x03\t\x00\x05B\x03\x1f\x03\t\x05B\x03!\x03\x1b\x07B\x13\r\x03\x11\x03F\x17\x11\x03\x11\x03\x05\r\x06\x17\x03\x11\x05\x07\t\x07B\x13#\x03\x11\x0fF3%\x03\x1d\x05\x0b\r\x03F7\x11\x03\x05\x03\x03\t\x06;\x03\x05\x07\x0f\x01\x11\x11\x04\x03\x03\x13\x06\x03\x01\x05\x01\x00j\x0bw/\x05\x1f\r\x15\x11\x0f\x0b\x15\x15\x0b!%3)1\t\x11\t\x0b\x0b\x15\x11\x0b\x13#\x07%\x0b\x19%)9\t\x0b\x03\x1dw\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jit(<lambda>)\x00\x00iota\x00add\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00jit:\x00jit(<lambda>)/jit\x00ge\x00broadcast_in_dim\x00select_n\x00eigh\x00reshape\x00transpose\x00real\x00imag\x00neg\x00complex\x00div\x00jit(<lambda>)/jit(tril)\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00tril\x00result[0]\x00result[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00num_batch_dims\x000\x00oneapisolver_syevd_ffi\x00\x08y'\x05S\x01\x0b\x97\xa5\xa7\xb1\xb3\x03\xc3\x03\xc5\x03\xc7\x03\xc9\x03\x9f\x03\xcb\x03\x95\x03\x9d\x11\xdd\xdf\xe1\x97\xe3\xe5\xe7\xeb\x07\xa1\n\x02\x05\x0e\x02\x05\x12\x02\x0b\x9b\xb7\x9b\x9d\xb9\x03\xbb\x03\xbd\x03\xbf\x05\xa1\xc1",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["c128"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_syevd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(),
    expected_outputs=(array([[-0.7941185704969036  +0.j,  0.015700875637630098+0.j,
        -0.5474974726007527  -0.j,  0.26339266503066167 +0.j],
       [-0.36964433974346067 +0.j,  0.38714602123136327 +0.j,
         0.7416993718783406  +0.j,  0.40418196656409777 +0.j],
       [ 0.05482989100998282 +0.j, -0.8213946693756174  +0.j,
         0.15909367404557642 +0.j,  0.5449712680975332  +0.j],
       [ 0.4793041217634259  +0.j,  0.41854777250662395 +0.j,
        -0.35329557332316447 +0.j,  0.6857605696309689  +0.j]]), array([-3.7082869338697062e+00, -3.5727421539663155e-15,
        5.8149603380928531e-15,  3.3708286933869722e+01])),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<4xf64> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc29)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc29)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc29)
    %cst_1 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc16)
    %0 = stablehlo.iota dim = 0 : tensor<16xcomplex<f64>> loc(#loc30)
    %1 = stablehlo.reshape %0 : (tensor<16xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc31)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc32)
    %3 = stablehlo.real %2 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc40)
    %4 = stablehlo.imag %2 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc41)
    %5 = stablehlo.negate %4 : tensor<4x4xf64> loc(#loc42)
    %6 = stablehlo.complex %3, %5 : tensor<4x4xcomplex<f64>> loc(#loc43)
    %7 = stablehlo.add %1, %6 : tensor<4x4xcomplex<f64>> loc(#loc37)
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc38)
    %9 = stablehlo.divide %7, %8 : tensor<4x4xcomplex<f64>> loc(#loc38)
    %10 = call @tril(%9) : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc27)
    %11:3 = stablehlo.custom_call @oneapisolver_syevd_ffi(%10) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=4, j=4, k=4, l=4, m=4}, custom>} : (tensor<4x4xcomplex<f64>>) -> (tensor<4x4xcomplex<f64>>, tensor<4xf64>, tensor<i32>) loc(#loc29)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc29)
    %13 = stablehlo.compare EQ, %11#2, %12, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc29)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc29)
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc29)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc29)
    %17 = stablehlo.select %16, %11#0, %15 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc29)
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc29)
    %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64> loc(#loc29)
    %20 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc29)
    %21 = stablehlo.select %20, %11#1, %19 : tensor<4xi1>, tensor<4xf64> loc(#loc29)
    return %17, %21 : tensor<4x4xcomplex<f64>>, tensor<4xf64> loc(#loc16)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xcomplex<f64>> loc(unknown)) -> tensor<4x4xcomplex<f64>> {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi64> loc(#loc10)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4x4xi64> loc(#loc11)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi64> loc(#loc11)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi64> loc(#loc10)
    %4 = stablehlo.compare GE, %2, %3, SIGNED : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1> loc(#loc12)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc13)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc14)
    return %6 : tensor<4x4xcomplex<f64>> loc(#loc)
  } loc(#loc39)
} loc(#loc)
#loc1 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":343:11)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":410:4)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":335:26)
#loc4 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":335:14)
#loc5 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:34)
#loc6 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:25)
#loc7 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:15)
#loc8 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":337:14)
#loc9 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":343:27)
#loc10 = loc("iota")
#loc11 = loc("add")
#loc12 = loc("ge")
#loc13 = loc("broadcast_in_dim")
#loc14 = loc("select_n")
#loc15 = loc("jit(<lambda>)"(#loc1))
#loc16 = loc("jit(<lambda>)"(#loc2))
#loc17 = loc("jit(<lambda>)"(#loc3))
#loc18 = loc("jit(<lambda>)"(#loc4))
#loc19 = loc("jit(<lambda>)"(#loc5))
#loc20 = loc("real"(#loc6))
#loc21 = loc("jit(<lambda>)"(#loc6))
#loc22 = loc("imag"(#loc6))
#loc23 = loc("neg"(#loc6))
#loc24 = loc("complex"(#loc6))
#loc25 = loc("jit(<lambda>)"(#loc7))
#loc26 = loc("jit(<lambda>)"(#loc8))
#loc27 = loc("jit(<lambda>)/jit(tril)"(#loc9))
#loc28 = loc("jit(<lambda>)/jit"(#loc9))
#loc29 = loc("eigh"(#loc15))
#loc30 = loc("iota"(#loc17))
#loc31 = loc("reshape"(#loc18))
#loc32 = loc("transpose"(#loc19))
#loc33 = loc(callsite(#loc20 at #loc21))
#loc34 = loc(callsite(#loc22 at #loc21))
#loc35 = loc(callsite(#loc23 at #loc21))
#loc36 = loc(callsite(#loc24 at #loc21))
#loc37 = loc("add"(#loc25))
#loc38 = loc("div"(#loc26))
#loc39 = loc("jit:"(#loc28))
#loc40 = loc(""(#loc33))
#loc41 = loc(""(#loc34))
#loc42 = loc(""(#loc35))
#loc43 = loc(""(#loc36))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x017\x07\x01\x05\t%\x01\x03\x0f\x03#\x13\x17\x1b\x1f#'+/37;?CGKOS\x03\xce\x02\x16\x02?\x01\x95\x0f\x07\x0b\x0b\x17\x0b\x0f\x0f\x17\x0f\x0b\x0f\x0b\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x17\x17\x0f\x0f\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0b\x0f\x0f\x17\x0b\x0f\x17\x0f\x0b#\x0b\x0b\x0b\x03Y\x0f\x0b\x0b\x0f\x0b\x0f\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0bO/\x0f\x0b/O\x1fOO\x1b\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f\x05\x15K\x13\x0f\x0f\x13\x0f\x0f\x0f\x0f\x0b\x03\x0b/\x0f\x0bO/\x01\x05\x0b\x0f\x03;\x17\x07\x0f\x07\x07\x13\x17\x0f\x0b\x17\x07\x0f\x17\x0f\x13\x17\x17\x13\x07\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02\x02\n\x1d?A\x1f\x05+\x05-\x17\x05F\x053\x05/\x1d\x07\t\x11\x03\x05\x17\x05^\x057\x1d\x15\x03\x051\x1d\x19\x03\x053\x1d\x07E\x1d\x83\x85\x03\x07!#%\x0f'\x0f\x055\x11\x01\x00\x057\x059\x05;\x1d-/\x05=\x1d1\x11\x05?\x1d5\x03\x05A\x1d9\x03\x05C\x1d=\x03\x05E\x05G\x1d\x07C\x17\x05^\x05\x17\x17\x05j\x06\t\x1d\x15I\x1d\x07K\x17\x05>\x055\x1dOQ\x05I\x1d\x07S\x17\x05>\x05\x1d\x1dWY\x05K\x1d\x07[\x17\x05F\x05E\x1d\x0b_\x15a\r\x1dc\t\x05M\x1d\x0bg\x15i\r\x1dk\t\x05O\x1d\x0bo\x15q\r\x1ds\t\x05Q\x1d\x0bw\x15y\r\x1d{\t\x05S\x1d\x19\x7f\x1d\x07\x81\x17\x05F\x05\x1f\x05U\x1d\x07\x87\x17\x05F\x05\x1d\x1d\x8b\x11\x05W\x03\x07\x8f\xcd\x91\xd7\x93\xed\x05Y\x05[\x05]\x1f'\x01\x03\x01\x1d_\x03\x03\xb5\x1da\x13\x07\x01\t\x07\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00##\x03\x05\xa9\xad\r\x03\x99\xab\x1dc\r\x03\x99\xaf\x1de\x1dg\x1di\r\x01#%\x1dk\x1f\t!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x07\x05\x07\x05\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\t!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x13\t\x00\x00\x00\x00\x1f\t!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\xcf\xd1\xd3\xd5\x1dm\x13-\x00\x1do\x05\x03\r\x03\xd9\xdb\x1dq\x1ds\x0b\x03\x1d/\x1du\x05\x01\x03\x03\xa3\x03\x03\xe9\x15\x03\x01\x01\x01\x03\x07\xa3\x02\x02\x06\x02\x15\x0b\x11\x11\x11\x11\x11\x03\xef\x07\xf5\xfb\xff\x01\x01\x01\x01\x01\x13\x05\xf1\xf3\x11\x03\x01\x11\x03\x05\x13\x05\xf7\xf9\x11\x03\t\x11\x03\r\x13\x03\xfd\x11\x03\x11\x13\x01\x1f1\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f3\x01\x07\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x15\x1d)\x01\x15\x01\x0b)\x03\x11\r)\x05\x11\x11\x07)\x01)\x03\r)\x05\x11\x11\r\x13)\x01\x07)\x05\x11\x11\x0b)\x01\r)\x03\t\x07\x11\x01\x05\x05\x0f\x11\x03\x05\x03\x05)\x03\x01\x07\x1b)\x03A\x15!)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x01\x0b)\x05\x05\x05\x0b)\x03\x05\x0b)\x03\x11\x0b)\x03\x05\x07\x04\xea\x04\x05\x01Q\x03\x1f\x01\x07\x04\xc2\x04\x03\x01\t\x0bP\x03\x03\x07\x04F\x03\x039m\x05B\x01\x05\x03\x1f\x05B\x01\x07\x03\t\x05B\x01\t\x03\x13\x05B\x1b\x0b\x03\t\x07BG\r\x03+\x13\x06M\x03\x05\x03\t\x15FU\x0f\x03\x05\x03\x0b\x17\x06]\x03\x17\x03\r\x19\x06e\x03\x17\x03\r\x1b\x06m\x03\x17\x03\x11\x1d\x06u\x03\x05\x05\x0f\x13\r\x06}\x03\x05\x05\x0b\x15\x03F\x1d\x11\x03\x05\x03\x07\x1f\x06\x1d\x03\x05\x05\x17\x19!F\x89\x13\x03\x05\x03\x1b#G\x01\x8d\x15\x07\x05\x0f\x13\x03\x1d\x03F\x01\x11\x03\x13\x03\x05\x0fF\x01\x17\x035\x05#%\x03F\x01\x11\x037\x03'\x03F\x01\x11\x03\x05\x03\x03\x03F\x01\x19\x03\x1d\x03)\t\x06\x01\x03\x05\x07-\x1f+\x03F\x01\x11\x039\x03'\x03F\x01\x11\x03\x0f\x03\x01\x03F\x01\x1b\x03;\x031\t\x06\x01\x03\x0f\x075!3\x11\x04\x1b\x05/7\x0bP+\x1d\x07\x04\x9b\x03\x15+\x03\t\x00\x05B\x03\x1f\x03\t\x05B\x03!\x03\x1b\x07B\x13\r\x03\x11\x03F\x17\x11\x03\x11\x03\x05\r\x06\x17\x03\x11\x05\x07\t\x07B\x13#\x03\x11\x0fF3%\x03\x1d\x05\x0b\r\x03F7\x11\x03\x05\x03\x03\t\x06;\x03\x05\x07\x0f\x01\x11\x11\x04\x03\x03\x13\x06\x03\x01\x05\x01\x00j\x0bw/\x05\x1f\r\x15\x11\x0f\x0b\x15\x15\x0b!%3)1\t\x11\t\x0b\x0b\x15\x11\x0b\x13#\x07%\x0b\x19%)9\t\x0b\x03\x1dw\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jit(<lambda>)\x00\x00iota\x00add\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00jit:\x00jit(<lambda>)/jit\x00ge\x00broadcast_in_dim\x00select_n\x00eigh\x00reshape\x00transpose\x00real\x00imag\x00neg\x00complex\x00div\x00jit(<lambda>)/jit(tril)\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00tril\x00result[0]\x00result[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00num_batch_dims\x000\x00oneapisolver_syevd_ffi\x00\x08y'\x05S\x01\x0b\x97\xa5\xa7\xb1\xb3\x03\xc3\x03\xc5\x03\xc7\x03\xc9\x03\x9f\x03\xcb\x03\x95\x03\x9d\x11\xdd\xdf\xe1\x97\xe3\xe5\xe7\xeb\x07\xa1\n\x02\x05\x0e\x02\x05\x12\x02\x0b\x9b\xb7\x9b\x9d\xb9\x03\xbb\x03\xbd\x03\xbf\x05\xa1\xc1",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
