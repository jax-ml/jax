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

data_2026_02_16 = {}

# f32 data
data_2026_02_16["f32"] = dict(
  testdata_version=1,
  platform="rocm",
  custom_call_targets=["hipsolver_syevd_ffi"],
  serialized_date=datetime.date(2026, 2, 16),
  inputs=(),
  expected_outputs=(
    array(
      [
        [0.7941185, -0.12958004, -0.53217375, 0.26339266],
        [0.36964434, 0.56943226, 0.6129818, 0.40418196],
        [-0.054829914, -0.75012463, 0.37055779, 0.5449713],
        [-0.4793041, 0.31027225, -0.45136586, 0.68576056],
      ],
      dtype=float32,
    ),
    array(
      [-3.7082865e00, -6.2891360e-07, -1.8522137e-07, 3.3708290e01],
      dtype=float32,
    ),
  ),
  mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xf32> {jax.result_info = "result[0]"}, tensor<4xf32> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc25)
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<16xf32> loc(#loc38)
    %1 = stablehlo.reshape %0 : (tensor<16xf32>) -> tensor<4x4xf32> loc(#loc39)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc40)
    %3 = stablehlo.add %1, %2 : tensor<4x4xf32> loc(#loc41)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc42)
    %5 = stablehlo.divide %3, %4 : tensor<4x4xf32> loc(#loc42)
    %6 = call @tril(%5) : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc31)
    %7:3 = stablehlo.custom_call @hipsolver_syevd_ffi(%6) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=4, j=4, k=4, l=4, m=4}, custom>} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>, tensor<i32>) loc(#loc25)
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc25)
    %9 = stablehlo.compare  EQ, %7#2, %8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc25)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc25)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc25)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc25)
    %13 = stablehlo.select %12, %7#0, %11 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc25)
    %14 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc25)
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32> loc(#loc25)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc25)
    %17 = stablehlo.select %16, %7#1, %15 : tensor<4xi1>, tensor<4xf32> loc(#loc25)
    return %13, %17 : tensor<4x4xf32>, tensor<4xf32> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xf32> loc(unknown)) -> tensor<4x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc43)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc43)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc33)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc34)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc34)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc33)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc35)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc36)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc37)
    return %6 : tensor<4x4xf32> loc(#loc43)
  } loc(#loc43)
} loc(#loc)
#loc1 = loc("<string>":16:15 to :106)
#loc2 = loc("<string>":21:27 to :58)
#loc3 = loc("<string>":21:4 to :86)
#loc4 = loc("<string>":12:30 to :71)
#loc5 = loc("<string>":16:40 to :68)
#loc6 = loc("<string>":12:18 to :79)
#loc7 = loc("<string>":13:35 to :64)
#loc8 = loc("<string>":13:16 to :65)
#loc9 = loc("<string>":13:15 to :71)
#loc10 = loc("<string>":16:31 to :69)
#loc11 = loc("Gen.eigh_harness"(#loc1))
#loc12 = loc("<lambda>"(#loc2))
#loc13 = loc("<module>"(#loc3))
#loc14 = loc("Gen.eigh_input"(#loc4))
#loc15 = loc("Gen.eigh_harness"(#loc5))
#loc16 = loc("Gen.eigh_input"(#loc6))
#loc17 = loc("Gen.eigh_input"(#loc7))
#loc18 = loc("Gen.eigh_input"(#loc8))
#loc19 = loc("Gen.eigh_input"(#loc9))
#loc20 = loc("Gen.eigh_harness"(#loc10))
#loc21 = loc(callsite(#loc12 at #loc13))
#loc22 = loc(callsite(#loc11 at #loc21))
#loc23 = loc(callsite(#loc15 at #loc21))
#loc24 = loc(callsite(#loc20 at #loc21))
#loc25 = loc("jit(<lambda>)/eigh"(#loc22))
#loc26 = loc(callsite(#loc14 at #loc23))
#loc27 = loc(callsite(#loc16 at #loc23))
#loc28 = loc(callsite(#loc17 at #loc23))
#loc29 = loc(callsite(#loc18 at #loc23))
#loc30 = loc(callsite(#loc19 at #loc23))
#loc31 = loc("jit(<lambda>)/jit(tril)"(#loc24))
#loc32 = loc("jit(<lambda>)/jit"(#loc24))
#loc33 = loc("iota"(#loc24))
#loc34 = loc("add"(#loc24))
#loc35 = loc("ge"(#loc24))
#loc36 = loc("broadcast_in_dim"(#loc24))
#loc37 = loc("select_n"(#loc24))
#loc38 = loc("jit(<lambda>)/iota"(#loc26))
#loc39 = loc("jit(<lambda>)/reshape"(#loc27))
#loc40 = loc("jit(<lambda>)/transpose"(#loc28))
#loc41 = loc("jit(<lambda>)/add"(#loc29))
#loc42 = loc("jit(<lambda>)/div"(#loc30))
#loc43 = loc("jit:"(#loc32))
""",
  mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01/\x07\x01\x05\t\x1d\x01\x03\x0f\x03\x1b\x13\x17\x1b\x1f#'+/37;?C\x03\xb6\x02\x0e\x027\x01\x95\x0f\x0b\x0f\x07\x0b\x0f\x0f\x0b\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x1b\x0f\x0b\x1b\x0f\x0b\x1b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0b\x0f\x0f\x1b\x0f\x0b#\x0b\x0b\x0b\x03Y\x0f\x0b\x0b\x0f\x0b\x1f\x0f\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x1f\x0f\x0b\x1f\x1fO\x1b\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x05\x15K\x13\x0f\x0f\x13\x0f\x0f\x0f\x0f\x0b\x03\x07\x0bO/\x01\x05\x0b\x0f\x033\x17\x0f\x0f\x07\x07\x07\x13\x17\x07\x07\x17\x13\x17\x17\x13\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02\x0e\t\x1dMO\x05#\x15-\x11\x1f\x05%\x15_\x11\x1d')\x05'\x1517\x11\x03\x05\x1d=\x05\x1d?\x05\x1d\x81\x83\x03\x07\x1d\x1f!\x13#\x13\x05)\x11\x01\x00\x05+\x05-\x05/\x051\x1d+\x05\x053\x1d\x0f/-\x03\x07!?\x8b\x1d35\x055-\x03\x07)7\x7f\x1d9;\x057-\x03\x07)\t\xb7\x059\x05;\x1dC\x05\x05=\x1dG\x05\x05?\x1dK\x05\x05A\x05C\x15Q\x11\x1d\x0fS-\x03\x07!\x1f\xd5\x1dWY\x05E\x15[\x0b\x1d\t]-\x03\x07\x19=\x8f\x1d\x0fa-\x03\x07!Q\x89\x1deg\x05G\x15i\x0b\x1d\tk-\x03\x07\x19%\x9f\x1doq\x05I\x15s\x0b\x1d\tu-\x03\x07\x1bG\x81\x1dy{\x05K\x15}\x0b\x1d\t\x7f-\x03\x07\x1b!\x83\x05M\x15\x85\x0b\x1d\t\x87-\x03\x07\x1b\x1f\x8f\x1d\x8b\x05\x05O\x03\x07\x8f\xc9\x91\xd3\x93\xed\x05Q\x05S\x05U\x1f!\x01\x03\x01\x1dW\x03\x03\xb7\x1dY\x1f\t\t\x00\x00\x00\x00\x13\x0b\x01\t\x07\x1f'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1d\x03\x05\xab\xaf\r\x03\x99\xad\x1d[\r\x03\x99\xb1\x1d]\x1d_\x1da\r\x01#\x1f\x1dc\x1f\x07\t\x00\x00\x00\x00\x13\x0b\x05\x07\x05\x1f\x07\t\x00\x00\xc0\x7f\x1f\x07\t\x00\x00\x00@\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\xcb\xcd\xcf\xd1\x1de\x13%\x00\x1dg\x05\x03\r\x03\xd5\xd7\x1di\x1dk\x0b\x03\x1dm\x1do\x05\x01\x03\x03\xa5\x03\x03\xe5\x15\x03\x01\x01\x01\x03\x07\xa5\xe9\xeb\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x01\x15\x0b\x11\x11\x11\x11\x11\x03\xef\x07\xf5\xfb\xff\x01\x01\x01\x01\x01\x13\x05\xf1\xf3\x11\x03\x01\x11\x03\x05\x13\x05\xf7\xf9\x11\x03\t\x11\x03\r\x13\x03\xfd\x11\x03\x11\x13\x01\x07\x01\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x0f)\x01\x0f)\x01\x17\x1d\x01\t)\x03\x11\x0f)\x05\x11\x11\x17\x13\x1b)\x05\x11\x11\r)\x03\t\x0b\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03A\x0f!)\x03\t\x15)\x03\x05\x15)\x03\x01\x15)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03\x11\r)\x03\x05\x0b\x04^\x04\x05\x01Q\x07\x1b\x01\x07\x046\x04\x03\x01\t\x0bP\x07\x03\x07\x04\xba\x02\x03/Y\x05B\x07\x05\x03\x07\x05B\x01\x07\x03\t\x05B\x07\t\x03\x07\x07BU\x0b\x03#\x13\x06c\x03\x05\x03\x07\x15Fm\r\x03\x05\x03\t\r\x06w\x03\x05\x05\t\x0b\x03F\x19\x0f\x03\x05\x03\x05\x17\x06\x19\x03\x05\x05\r\x0f\x19F\x89\x11\x03\x05\x03\x11\x1bG\x01\x8d\x13\x07\x05\x11\t\x03\x13\x03F\x01\x0f\x03\t\x03\x03\x0fF\x01\x15\x03-\x05\x19\x1b\x03F\x01\x0f\x03/\x03\x1d\x03F\x01\x0f\x03\x05\x03\x01\x03F\x01\x17\x03\x19\x03\x1f\t\x06\x01\x03\x05\x07#\x15!\x03F\x01\x0f\x031\x03\x1d\x03F\x01\x0f\x03\x11\x03\x01\x03F\x01\x19\x033\x03'\t\x06\x01\x03\x11\x07+\x17)\x11\x04\x07\x05%-\x0bP\r\x1b\x07\x04\x9b\x03\x15+\x03\t\x00\x05B\r\x1d\x03\x07\x05B\r\x07\x03\t\x07B\x15\x0b\x03\x13\x03F\x17\x0f\x03\x13\x03\x05\r\x06\x17\x03\x13\x05\x07\t\x07B\x15\x1f\x03\x13\x0fFA!\x03\x19\x05\x0b\r\x03FE\x0f\x03\x05\x03\x03\t\x06I\x03\x05\x07\x0f\x01\x11\x11\x04\r\x03\x13\x06\x03\x01\x05\x01\x00\xa2\x0bq)\x03\x05\x1f\r\x15\x11\x0f\x0b\x15\x15\x0b!%3)1%%1-''\x13#\x07\t\x0b\x13\x13%\x0b\x19%)9#\x1f\x13\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00<string>\x00Gen.eigh_input\x00Gen.eigh_harness\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00jit:\x00jit(<lambda>)/jit\x00<lambda>\x00<module>\x00iota\x00add\x00ge\x00broadcast_in_dim\x00select_n\x00jit(<lambda>)/eigh\x00jit(<lambda>)/iota\x00jit(<lambda>)/reshape\x00jit(<lambda>)/transpose\x00jit(<lambda>)/add\x00jit(<lambda>)/div\x00jit(<lambda>)/jit(tril)\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00tril\x00result[0]\x00result[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00num_batch_dims\x000\x00\x00hipsolver_syevd_ffi\x00\x08q#\x05K\x01\x0b\x97\xa7\xa9\xb3\xb5\x03\xc3\x03\x9f\x03\xc5\x03\xa1\x03\xc7\x03\x95\x03\x9d\x11\xd9\xdb\xdd\x97\xdf\xe1\xe3\xe7\x07\xa3\x02\x02\x05\x06\x02\x05\n\x02\x0b\x9b\xb9\x9b\x9d\xbb\x03\xbd\x03\xbf\x05\xa3\xc1",
  xla_call_module_version=10,
  nr_devices=1,
)

# f64 data
data_2026_02_16["f64"] = dict(
  testdata_version=1,
  platform="rocm",
  custom_call_targets=["hipsolver_syevd_ffi"],
  serialized_date=datetime.date(2026, 2, 16),
  inputs=(),
  expected_outputs=(
    array([
      [
        0.7941185704969035,
        -0.4585357029041844,
        -0.29957471382305495,
        0.2633926650306618,
      ],
      [
        0.3696443397434602,
        0.3880911290561137,
        0.7412052856988751,
        0.4041819665640975,
      ],
      [
        -0.05482989100998275,
        0.5994248506003262,
        -0.5836864299285845,
        0.5449712680975332,
      ],
      [
        -0.47930412176342585,
        -0.5289802767522555,
        0.1420558580527646,
        0.6857605696309689,
      ],
    ]),
    array([
      -3.7082869338697058e00,
      4.3097085664716422e-16,
      1.0318094698114837e-15,
      3.3708286933869694e01,
    ]),
  ),
  mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xf64> {jax.result_info = "result[0]"}, tensor<4xf64> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc25)
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<16xf64> loc(#loc38)
    %1 = stablehlo.reshape %0 : (tensor<16xf64>) -> tensor<4x4xf64> loc(#loc39)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc40)
    %3 = stablehlo.add %1, %2 : tensor<4x4xf64> loc(#loc41)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc42)
    %5 = stablehlo.divide %3, %4 : tensor<4x4xf64> loc(#loc42)
    %6 = call @tril(%5) : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc31)
    %7:3 = stablehlo.custom_call @hipsolver_syevd_ffi(%6) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=4, j=4, k=4, l=4, m=4}, custom>} : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i32>) loc(#loc25)
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc25)
    %9 = stablehlo.compare  EQ, %7#2, %8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc25)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc25)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc25)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc25)
    %13 = stablehlo.select %12, %7#0, %11 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc25)
    %14 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc25)
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64> loc(#loc25)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc25)
    %17 = stablehlo.select %16, %7#1, %15 : tensor<4xi1>, tensor<4xf64> loc(#loc25)
    return %13, %17 : tensor<4x4xf64>, tensor<4xf64> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xf64> loc(unknown)) -> tensor<4x4xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc43)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc43)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc33)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc34)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc34)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc33)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc35)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc36)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc37)
    return %6 : tensor<4x4xf64> loc(#loc43)
  } loc(#loc43)
} loc(#loc)
#loc1 = loc("<string>":16:15 to :106)
#loc2 = loc("<string>":21:27 to :58)
#loc3 = loc("<string>":21:4 to :86)
#loc4 = loc("<string>":12:30 to :71)
#loc5 = loc("<string>":16:40 to :68)
#loc6 = loc("<string>":12:18 to :79)
#loc7 = loc("<string>":13:35 to :64)
#loc8 = loc("<string>":13:16 to :65)
#loc9 = loc("<string>":13:15 to :71)
#loc10 = loc("<string>":16:31 to :69)
#loc11 = loc("Gen.eigh_harness"(#loc1))
#loc12 = loc("<lambda>"(#loc2))
#loc13 = loc("<module>"(#loc3))
#loc14 = loc("Gen.eigh_input"(#loc4))
#loc15 = loc("Gen.eigh_harness"(#loc5))
#loc16 = loc("Gen.eigh_input"(#loc6))
#loc17 = loc("Gen.eigh_input"(#loc7))
#loc18 = loc("Gen.eigh_input"(#loc8))
#loc19 = loc("Gen.eigh_input"(#loc9))
#loc20 = loc("Gen.eigh_harness"(#loc10))
#loc21 = loc(callsite(#loc12 at #loc13))
#loc22 = loc(callsite(#loc11 at #loc21))
#loc23 = loc(callsite(#loc15 at #loc21))
#loc24 = loc(callsite(#loc20 at #loc21))
#loc25 = loc("jit(<lambda>)/eigh"(#loc22))
#loc26 = loc(callsite(#loc14 at #loc23))
#loc27 = loc(callsite(#loc16 at #loc23))
#loc28 = loc(callsite(#loc17 at #loc23))
#loc29 = loc(callsite(#loc18 at #loc23))
#loc30 = loc(callsite(#loc19 at #loc23))
#loc31 = loc("jit(<lambda>)/jit(tril)"(#loc24))
#loc32 = loc("jit(<lambda>)/jit"(#loc24))
#loc33 = loc("iota"(#loc24))
#loc34 = loc("add"(#loc24))
#loc35 = loc("ge"(#loc24))
#loc36 = loc("broadcast_in_dim"(#loc24))
#loc37 = loc("select_n"(#loc24))
#loc38 = loc("jit(<lambda>)/iota"(#loc26))
#loc39 = loc("jit(<lambda>)/reshape"(#loc27))
#loc40 = loc("jit(<lambda>)/transpose"(#loc28))
#loc41 = loc("jit(<lambda>)/add"(#loc29))
#loc42 = loc("jit(<lambda>)/div"(#loc30))
#loc43 = loc("jit:"(#loc32))
""",
  mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01/\x07\x01\x05\t\x1d\x01\x03\x0f\x03\x1b\x13\x17\x1b\x1f#'+/37;?C\x03\xb6\x02\x0e\x027\x01\x95\x0f\x0b\x0f\x07\x0b\x0f\x0f\x0b\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x1b\x0f\x0b\x1b\x0f\x0b\x1b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0b\x0f\x0f\x1b\x0f\x0b#\x0b\x0b\x0b\x03Y\x0f\x0b\x0b\x0f\x0b\x1f\x0f\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b/\x0f\x0b//O\x1b\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x05\x15K\x13\x0f\x0f\x13\x0f\x0f\x0f\x0f\x0b\x03\x07\x0bO/\x01\x05\x0b\x0f\x033\x17\x0f\x0f\x07\x07\x07\x13\x17\x07\x07\x17\x13\x17\x17\x13\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02>\t\x1dMO\x05#\x15-\x11\x1f\x05%\x15_\x11\x1d')\x05'\x1517\x11\x03\x05\x1d=\x05\x1d?\x05\x1d\x81\x83\x03\x07\x1d\x1f!\x13#\x13\x05)\x11\x01\x00\x05+\x05-\x05/\x051\x1d+\x05\x053\x1d\x0f/-\x03\x07!?\x8b\x1d35\x055-\x03\x07+7u\x1d9;\x057-\x03\x07+\t\xad\x059\x05;\x1dC\x05\x05=\x1dG\x05\x05?\x1dK\x05\x05A\x05C\x15Q\x11\x1d\x0fS-\x03\x07!\x1f\xd5\x1dWY\x05E\x15[\x0b\x1d\t]-\x03\x07\x19=\x8f\x1d\x0fa-\x03\x07!Q\x89\x1deg\x05G\x15i\x0b\x1d\tk-\x03\x07\x19%\x9f\x1doq\x05I\x15s\x0b\x1d\tu-\x03\x07\x1bG\x81\x1dy{\x05K\x15}\x0b\x1d\t\x7f-\x03\x07\x1b!\x83\x05M\x15\x85\x0b\x1d\t\x87-\x03\x07\x1b\x1f\x8f\x1d\x8b\x05\x05O\x03\x07\x8f\xc9\x91\xd3\x93\xed\x05Q\x05S\x05U\x1f!\x01\x03\x01\x1dW\x03\x03\xb7\x1dY\x1f\t\t\x00\x00\x00\x00\x13\x0b\x01\t\x07\x1f'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1d\x03\x05\xab\xaf\r\x03\x99\xad\x1d[\r\x03\x99\xb1\x1d]\x1d_\x1da\r\x01#\x1f\x1dc\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\xcb\xcd\xcf\xd1\x1de\x13%\x00\x1dg\x05\x03\r\x03\xd5\xd7\x1di\x1dk\x0b\x03\x1dm\x1do\x05\x01\x03\x03\xa5\x03\x03\xe5\x15\x03\x01\x01\x01\x03\x07\xa5\xe9\xeb\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x01\x15\x0b\x11\x11\x11\x11\x11\x03\xef\x07\xf5\xfb\xff\x01\x01\x01\x01\x01\x13\x05\xf1\xf3\x11\x03\x01\x11\x03\x05\x13\x05\xf7\xf9\x11\x03\t\x11\x03\r\x13\x03\xfd\x11\x03\x11\x13\x01\x07\x01\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x0f)\x01\x0f)\x01\x17\x1d\x01\x0b)\x03\x11\x0f)\x05\x11\x11\x17\x13\x1b)\x05\x11\x11\r)\x03\t\x0b\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03A\x0f!)\x03\t\x15)\x03\x05\x15)\x03\x01\x15)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03\x11\r)\x03\x05\x0b\x04^\x04\x05\x01Q\x07\x1b\x01\x07\x046\x04\x03\x01\t\x0bP\x07\x03\x07\x04\xba\x02\x03/Y\x05B\x07\x05\x03\x07\x05B\x01\x07\x03\t\x05B\x07\t\x03\x07\x07BU\x0b\x03#\x13\x06c\x03\x05\x03\x07\x15Fm\r\x03\x05\x03\t\r\x06w\x03\x05\x05\t\x0b\x03F\x19\x0f\x03\x05\x03\x05\x17\x06\x19\x03\x05\x05\r\x0f\x19F\x89\x11\x03\x05\x03\x11\x1bG\x01\x8d\x13\x07\x05\x11\t\x03\x13\x03F\x01\x0f\x03\t\x03\x03\x0fF\x01\x15\x03-\x05\x19\x1b\x03F\x01\x0f\x03/\x03\x1d\x03F\x01\x0f\x03\x05\x03\x01\x03F\x01\x17\x03\x19\x03\x1f\t\x06\x01\x03\x05\x07#\x15!\x03F\x01\x0f\x031\x03\x1d\x03F\x01\x0f\x03\x11\x03\x01\x03F\x01\x19\x033\x03'\t\x06\x01\x03\x11\x07+\x17)\x11\x04\x07\x05%-\x0bP\r\x1b\x07\x04\x9b\x03\x15+\x03\t\x00\x05B\r\x1d\x03\x07\x05B\r\x07\x03\t\x07B\x15\x0b\x03\x13\x03F\x17\x0f\x03\x13\x03\x05\r\x06\x17\x03\x13\x05\x07\t\x07B\x15\x1f\x03\x13\x0fFA!\x03\x19\x05\x0b\r\x03FE\x0f\x03\x05\x03\x03\t\x06I\x03\x05\x07\x0f\x01\x11\x11\x04\r\x03\x13\x06\x03\x01\x05\x01\x00\xa2\x0bq)\x03\x05\x1f\r\x15\x11\x0f\x0b\x15\x15\x0b!%3)1%%1-''\x13#\x07\t\x0b\x13\x13%\x0b\x19%)9#\x1f\x13\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00<string>\x00Gen.eigh_input\x00Gen.eigh_harness\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00jit:\x00jit(<lambda>)/jit\x00<lambda>\x00<module>\x00iota\x00add\x00ge\x00broadcast_in_dim\x00select_n\x00jit(<lambda>)/eigh\x00jit(<lambda>)/iota\x00jit(<lambda>)/reshape\x00jit(<lambda>)/transpose\x00jit(<lambda>)/add\x00jit(<lambda>)/div\x00jit(<lambda>)/jit(tril)\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00tril\x00result[0]\x00result[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00num_batch_dims\x000\x00\x00hipsolver_syevd_ffi\x00\x08q#\x05K\x01\x0b\x97\xa7\xa9\xb3\xb5\x03\xc3\x03\x9f\x03\xc5\x03\xa1\x03\xc7\x03\x95\x03\x9d\x11\xd9\xdb\xdd\x97\xdf\xe1\xe3\xe7\x07\xa3\x02\x02\x05\x06\x02\x05\n\x02\x0b\x9b\xb9\x9b\x9d\xbb\x03\xbd\x03\xbf\x05\xa3\xc1",
  xla_call_module_version=10,
  nr_devices=1,
)  # End paste


# c64 data
data_2026_02_16["c64"] = dict(
  testdata_version=1,
  platform="rocm",
  custom_call_targets=["hipsolver_syevd_ffi"],
  serialized_date=datetime.date(2026, 2, 16),
  inputs=(),
  expected_outputs=(
    array(
      [
        [
          0.7941186 + 0.0j,
          -0.09437865 + 0.0j,
          -0.53953 + 0.0j,
          0.26339266 + 0.0j,
        ],
        [0.3696444 + 0.0j, 0.52798 + 0.0j, 0.6490277 + 0.0j, 0.40418193 + 0.0j],
        [
          -0.054829918 + 0.0j,
          -0.7728244 + 0.0j,
          0.32053462 + 0.0j,
          0.5449713 + 0.0j,
        ],
        [
          -0.4793041 + 0.0j,
          0.3392229 + 0.0j,
          -0.43003234 + 0.0j,
          0.68576056 + 0.0j,
        ],
      ],
      dtype=complex64,
    ),
    array(
      [-3.7082870e00, -6.7111625e-07, -1.5134994e-07, 3.3708286e01],
      dtype=float32,
    ),
  ),
  mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<4xf32> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc27)
    %cst_0 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc27)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc27)
    %cst_1 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<16xcomplex<f32>> loc(#loc41)
    %1 = stablehlo.reshape %0 : (tensor<16xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc42)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc43)
    %3 = stablehlo.real %2 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc44)
    %4 = stablehlo.imag %2 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc45)
    %5 = stablehlo.negate %4 : tensor<4x4xf32> loc(#loc46)
    %6 = stablehlo.complex %3, %5 : tensor<4x4xcomplex<f32>> loc(#loc47)
    %7 = stablehlo.add %1, %6 : tensor<4x4xcomplex<f32>> loc(#loc48)
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc49)
    %9 = stablehlo.divide %7, %8 : tensor<4x4xcomplex<f32>> loc(#loc49)
    %10 = call @tril(%9) : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc34)
    %11:3 = stablehlo.custom_call @hipsolver_syevd_ffi(%10) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=4, j=4, k=4, l=4, m=4}, custom>} : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4xf32>, tensor<i32>) loc(#loc27)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc27)
    %13 = stablehlo.compare  EQ, %11#2, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc27)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc27)
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc27)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc27)
    %17 = stablehlo.select %16, %11#0, %15 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc27)
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc27)
    %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32> loc(#loc27)
    %20 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc27)
    %21 = stablehlo.select %20, %11#1, %19 : tensor<4xi1>, tensor<4xf32> loc(#loc27)
    return %17, %21 : tensor<4x4xcomplex<f32>>, tensor<4xf32> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xcomplex<f32>> loc(unknown)) -> tensor<4x4xcomplex<f32>> {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc50)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc50)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc36)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc37)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc37)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc36)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc38)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc39)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc40)
    return %6 : tensor<4x4xcomplex<f32>> loc(#loc50)
  } loc(#loc50)
} loc(#loc)
#loc1 = loc("<string>":16:15 to :106)
#loc2 = loc("<string>":21:27 to :58)
#loc3 = loc("<string>":21:4 to :86)
#loc4 = loc("<string>":12:30 to :71)
#loc5 = loc("<string>":16:40 to :68)
#loc6 = loc("<string>":12:18 to :79)
#loc7 = loc("<string>":13:35 to :64)
#loc8 = loc("<string>":13:26 to :65)
#loc9 = loc("<string>":13:16 to :65)
#loc10 = loc("<string>":13:15 to :71)
#loc11 = loc("<string>":16:31 to :69)
#loc12 = loc("Gen.eigh_harness"(#loc1))
#loc13 = loc("<lambda>"(#loc2))
#loc14 = loc("<module>"(#loc3))
#loc15 = loc("Gen.eigh_input"(#loc4))
#loc16 = loc("Gen.eigh_harness"(#loc5))
#loc17 = loc("Gen.eigh_input"(#loc6))
#loc18 = loc("Gen.eigh_input"(#loc7))
#loc19 = loc("Gen.eigh_input"(#loc8))
#loc20 = loc("Gen.eigh_input"(#loc9))
#loc21 = loc("Gen.eigh_input"(#loc10))
#loc22 = loc("Gen.eigh_harness"(#loc11))
#loc23 = loc(callsite(#loc13 at #loc14))
#loc24 = loc(callsite(#loc12 at #loc23))
#loc25 = loc(callsite(#loc16 at #loc23))
#loc26 = loc(callsite(#loc22 at #loc23))
#loc27 = loc("jit(<lambda>)/eigh"(#loc24))
#loc28 = loc(callsite(#loc15 at #loc25))
#loc29 = loc(callsite(#loc17 at #loc25))
#loc30 = loc(callsite(#loc18 at #loc25))
#loc31 = loc(callsite(#loc19 at #loc25))
#loc32 = loc(callsite(#loc20 at #loc25))
#loc33 = loc(callsite(#loc21 at #loc25))
#loc34 = loc("jit(<lambda>)/jit(tril)"(#loc26))
#loc35 = loc("jit(<lambda>)/jit"(#loc26))
#loc36 = loc("iota"(#loc26))
#loc37 = loc("add"(#loc26))
#loc38 = loc("ge"(#loc26))
#loc39 = loc("broadcast_in_dim"(#loc26))
#loc40 = loc("select_n"(#loc26))
#loc41 = loc("jit(<lambda>)/iota"(#loc28))
#loc42 = loc("jit(<lambda>)/reshape"(#loc29))
#loc43 = loc("jit(<lambda>)/transpose"(#loc30))
#loc44 = loc("jit(<lambda>)/real"(#loc31))
#loc45 = loc("jit(<lambda>)/imag"(#loc31))
#loc46 = loc("jit(<lambda>)/neg"(#loc31))
#loc47 = loc("jit(<lambda>)/complex"(#loc31))
#loc48 = loc("jit(<lambda>)/add"(#loc32))
#loc49 = loc("jit(<lambda>)/div"(#loc33))
#loc50 = loc("jit:"(#loc35))
""",
  mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x017\x07\x01\x05\t%\x01\x03\x0f\x03#\x13\x17\x1b\x1f#'+/37;?CGKOS\x03\xf2\x02>\x02=\x01\xab\x0f\x0b\x0f\x0b\x0f\x07\x0f\x0f\x0b\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x1b\x0f\x0b\x1b\x0f\x0b\x1b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0f\x1b\x0b\x0f\x0f\x1b\x0f\x0b#\x0b\x0b\x0b\x03G\x0f\x0b\x0b\x0f\x0b\x1f\x0f\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b/\x0f\x0b\x1f//O\x1b\x0b\x0f\x0b\x0b\x13\x0b\x0b\x05\x15O\x13\x0f\x0f\x13\x0f\x0f\x13\x0f\x0b\x03\x1b\x0b\x0b\x0b\x0b\x0f\x13\x17\x1f/\x0f\x0bO/\x01\x05\x0b\x0f\x039\x17\x0f\x0f\x07\x07\x07\x13\x17\x0b\x17\x07\x07\x17\x0f\x13\x17\x17\x13\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02\x16\n\x1dOQ\x05+\x15/\x13\x05-\x15a\x13\x1f\x1d)+\x15}\t\x05/\x1539\x11\x03\x05\x1d?\x05\x1dA\x05\x1d\x97\x99\x03\x07\x1f!#\x15%\x15\x051\x11\x01\x00\x053\x055\x057\x059\x1d-\x05\x05;\x1d\x111-\x03\x07!?\x8b\x1d57\x05=-\x03\x07+7u\x1d;=\x05?-\x03\x07+\t\xad\x05A\x05C\x1dE\x05\x05E\x1dI\x05\x05G\x1dM\x05\x05I\x05K\x15S\x13\x1d\x11U-\x03\x07!\x1f\xd5\x1dY[\x05M\x15]\t\x1d\x07_-\x03\x07\x19=\x8f\x1d\x11c-\x03\x07!Q\x89\x1dgi\x05O\x15k\t\x1d\x07m-\x03\x07\x19%\x9f\x1dqs\x05Q\x15u\t\x1d\x07w-\x03\x07\x1bG\x81\x1d{\x0f\x05S\x1d\x07\x7f-\x03\x07\x1b5\x83\x1d\x83\x0f\x05U\x1d\x87\x0f\x05W\x1d\x8b\x0f\x05Y\x1d\x8f\x91\x05[\x15\x93\t\x1d\x07\x95-\x03\x07\x1b!\x83\x05]\x15\x9b\t\x1d\x07\x9d-\x03\x07\x1b\x1f\x8f\x1d\xa1\x05\x05_\x03\x07\xa5\xe1\xa7\xeb\xa9\xf1\x05a\x05c\x05e\x1f'\x01\x03\x01\x1dg\x03\x03\xcd\x1di\x1f\t\t\x00\x00\x00\x00\x13\x0b\x01\t\x07\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00##\x03\x05\xc1\xc5\r\x03\xaf\xc3\x1dk\r\x03\xaf\xc7\x1dm\x1do\x1dq\r\x01#%\x1ds\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x05\x07\x05\x1f\x1f\t\x00\x00\xc0\x7f\x1f\x07\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x07\x11\x00\x00\x00@\x00\x00\x00\x00\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\xe3\xe5\xe7\xe9\x1du\x13+\x00\x1dw\x05\x03\r\x03\xed\xef\x1dy\x1d{\x15\x0b\x11\x11\x11\x11\x11\x03\xf3\x07\xf9\xff\x06\x02\x01\x01\x01\x01\x01\x13\x05\xf5\xf7\x11\x03\x01\x11\x03\x05\x13\x05\xfb\xfd\x11\x03\t\x11\x03\r\x13\x03\x02\x02\x11\x03\x11\x13\x01\x0b\x03\x1d}\x1d\x7f\x05\x01\x03\x03\xbb\x03\x03\"\x02\x15\x03\x01\x01\x01\x03\x07\xbb*\x02.\x02\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f1\x01\x07\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f;\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x15)\x01\x15)\x01\x1b\x1d\x01\t)\x03\x11\x0f)\x05\x11\x11\x1b\x03\x0f)\x05\x11\x11\x0f\x13\x1b)\x05\x11\x11\r)\x01\x0f)\x03\t\x0b\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03A\x15!)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03\x11\r)\x03\x05\x0b\x04\xea\x04\x05\x01Q\x0b\x1d\x01\x07\x04\xc2\x04\x03\x01\t\x0bP\x0b\x03\x07\x04F\x03\x039m\x05B\x01\x05\x03\x1f\x05B\x01\x07\x03\x07\x05B\x01\t\x03\t\x05B\x0b\x0b\x03\x07\x07BW\r\x03)\x13\x06e\x03\x05\x03\t\x15Fo\x0f\x03\x05\x03\x0b\x17\x06y\x03\x17\x03\r\x19\x06\x81\x03\x17\x03\r\x1b\x06\x85\x03\x17\x03\x11\x1d\x06\x89\x03\x05\x05\x0f\x13\r\x06\x8d\x03\x05\x05\x0b\x15\x03F\x1b\x11\x03\x05\x03\x07\x1f\x06\x1b\x03\x05\x05\x17\x19!F\x9f\x13\x03\x05\x03\x1b#G\x01\xa3\x15\x07\x05\x11\t\x03\x1d\x03F\x01\x11\x03\t\x03\x05\x0fF\x01\x17\x033\x05#%\x03F\x01\x11\x035\x03'\x03F\x01\x11\x03\x05\x03\x03\x03F\x01\x19\x03\x1d\x03)\t\x06\x01\x03\x05\x07-\x1f+\x03F\x01\x11\x037\x03'\x03F\x01\x11\x03\x11\x03\x01\x03F\x01\x1b\x039\x031\t\x06\x01\x03\x11\x075!3\x11\x04\x0b\x05/7\x0bP\r\x1d\x07\x04\x9b\x03\x15+\x03\t\x00\x05B\r\x1f\x03\x07\x05B\r\t\x03\t\x07B\x17\r\x03\x13\x03F\x19\x11\x03\x13\x03\x05\r\x06\x19\x03\x13\x05\x07\t\x07B\x17!\x03\x13\x0fFC#\x03\x1d\x05\x0b\r\x03FG\x11\x03\x05\x03\x03\t\x06K\x03\x05\x07\x0f\x01\x11\x11\x04\r\x03\x13\x06\x03\x01\x05\x01\x00\x8e\r\x81)\x03\x05\x1f\r\x15\x11\x0f\x0b\x15\x15\x0b!%3)1%%-%''1-''\x13#\x07\t\x0b\x13\x13%\x0b\x19%)9#\x1f\x13\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00<string>\x00Gen.eigh_input\x00Gen.eigh_harness\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00jit:\x00jit(<lambda>)/jit\x00<lambda>\x00<module>\x00iota\x00add\x00ge\x00broadcast_in_dim\x00select_n\x00jit(<lambda>)/eigh\x00jit(<lambda>)/iota\x00jit(<lambda>)/reshape\x00jit(<lambda>)/transpose\x00jit(<lambda>)/real\x00jit(<lambda>)/imag\x00jit(<lambda>)/neg\x00jit(<lambda>)/complex\x00jit(<lambda>)/add\x00jit(<lambda>)/div\x00jit(<lambda>)/jit(tril)\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00tril\x00result[0]\x00result[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00num_batch_dims\x000\x00\x00hipsolver_syevd_ffi\x00\x08\x83%\x05O\x01\x0b\xad\xbd\xbf\xc9\xcb\x03\xd9\x03\xdb\x03\xb5\x03\xdd\x03\xb7\x03\xdf\x03\xab\x03\xb3\x1f\n\x02\x0e\x02\x12\x02\xad\x16\x02\x1a\x02\x1e\x02&\x02\x07\xb92\x02\x056\x02\x05:\x02\x0b\xb1\xcf\xb1\xb3\xd1\x03\xd3\x03\xd5\x05\xb9\xd7",
  xla_call_module_version=10,
  nr_devices=1,
)  # End paste


# c128 data
data_2026_02_16["c128"] = dict(
  testdata_version=1,
  platform="rocm",
  custom_call_targets=["hipsolver_syevd_ffi"],
  serialized_date=datetime.date(2026, 2, 16),
  inputs=(),
  expected_outputs=(
    array([
      [
        0.7941185704969035 + 0.0j,
        -0.45904919256164606 + 0.0j,
        -0.29878728019863976 + 0.0j,
        0.2633926650306618 + 0.0j,
      ],
      [
        0.3696443397434602 + 0.0j,
        0.389362700303383 + 0.0j,
        0.7405381067929308 + 0.0j,
        0.4041819665640975 + 0.0j,
      ],
      [
        -0.054829891009982694 + 0.0j,
        0.5984221770781727 + 0.0j,
        -0.5847143729899411 + 0.0j,
        0.5449712680975332 + 0.0j,
      ],
      [
        -0.4793041217634258 + 0.0j,
        -0.5287356848199095 + 0.0j,
        0.14296354639565043 + 0.0j,
        0.6857605696309689 + 0.0j,
      ],
    ]),
    array([
      -3.7082869338697053e00,
      4.2939212090511704e-16,
      1.0231783434508128e-15,
      3.3708286933869694e01,
    ]),
  ),
  mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<4xf64> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc27)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc27)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc27)
    %cst_1 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<16xcomplex<f64>> loc(#loc41)
    %1 = stablehlo.reshape %0 : (tensor<16xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc42)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc43)
    %3 = stablehlo.real %2 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc44)
    %4 = stablehlo.imag %2 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc45)
    %5 = stablehlo.negate %4 : tensor<4x4xf64> loc(#loc46)
    %6 = stablehlo.complex %3, %5 : tensor<4x4xcomplex<f64>> loc(#loc47)
    %7 = stablehlo.add %1, %6 : tensor<4x4xcomplex<f64>> loc(#loc48)
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc49)
    %9 = stablehlo.divide %7, %8 : tensor<4x4xcomplex<f64>> loc(#loc49)
    %10 = call @tril(%9) : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc34)
    %11:3 = stablehlo.custom_call @hipsolver_syevd_ffi(%10) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=4, j=4, k=4, l=4, m=4}, custom>} : (tensor<4x4xcomplex<f64>>) -> (tensor<4x4xcomplex<f64>>, tensor<4xf64>, tensor<i32>) loc(#loc27)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc27)
    %13 = stablehlo.compare  EQ, %11#2, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc27)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc27)
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc27)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc27)
    %17 = stablehlo.select %16, %11#0, %15 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc27)
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc27)
    %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64> loc(#loc27)
    %20 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc27)
    %21 = stablehlo.select %20, %11#1, %19 : tensor<4xi1>, tensor<4xf64> loc(#loc27)
    return %17, %21 : tensor<4x4xcomplex<f64>>, tensor<4xf64> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xcomplex<f64>> loc(unknown)) -> tensor<4x4xcomplex<f64>> {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc50)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc50)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc36)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc37)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc37)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc36)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc38)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc39)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc40)
    return %6 : tensor<4x4xcomplex<f64>> loc(#loc50)
  } loc(#loc50)
} loc(#loc)
#loc1 = loc("<string>":16:15 to :106)
#loc2 = loc("<string>":21:27 to :58)
#loc3 = loc("<string>":21:4 to :86)
#loc4 = loc("<string>":12:30 to :71)
#loc5 = loc("<string>":16:40 to :68)
#loc6 = loc("<string>":12:18 to :79)
#loc7 = loc("<string>":13:35 to :64)
#loc8 = loc("<string>":13:26 to :65)
#loc9 = loc("<string>":13:16 to :65)
#loc10 = loc("<string>":13:15 to :71)
#loc11 = loc("<string>":16:31 to :69)
#loc12 = loc("Gen.eigh_harness"(#loc1))
#loc13 = loc("<lambda>"(#loc2))
#loc14 = loc("<module>"(#loc3))
#loc15 = loc("Gen.eigh_input"(#loc4))
#loc16 = loc("Gen.eigh_harness"(#loc5))
#loc17 = loc("Gen.eigh_input"(#loc6))
#loc18 = loc("Gen.eigh_input"(#loc7))
#loc19 = loc("Gen.eigh_input"(#loc8))
#loc20 = loc("Gen.eigh_input"(#loc9))
#loc21 = loc("Gen.eigh_input"(#loc10))
#loc22 = loc("Gen.eigh_harness"(#loc11))
#loc23 = loc(callsite(#loc13 at #loc14))
#loc24 = loc(callsite(#loc12 at #loc23))
#loc25 = loc(callsite(#loc16 at #loc23))
#loc26 = loc(callsite(#loc22 at #loc23))
#loc27 = loc("jit(<lambda>)/eigh"(#loc24))
#loc28 = loc(callsite(#loc15 at #loc25))
#loc29 = loc(callsite(#loc17 at #loc25))
#loc30 = loc(callsite(#loc18 at #loc25))
#loc31 = loc(callsite(#loc19 at #loc25))
#loc32 = loc(callsite(#loc20 at #loc25))
#loc33 = loc(callsite(#loc21 at #loc25))
#loc34 = loc("jit(<lambda>)/jit(tril)"(#loc26))
#loc35 = loc("jit(<lambda>)/jit"(#loc26))
#loc36 = loc("iota"(#loc26))
#loc37 = loc("add"(#loc26))
#loc38 = loc("ge"(#loc26))
#loc39 = loc("broadcast_in_dim"(#loc26))
#loc40 = loc("select_n"(#loc26))
#loc41 = loc("jit(<lambda>)/iota"(#loc28))
#loc42 = loc("jit(<lambda>)/reshape"(#loc29))
#loc43 = loc("jit(<lambda>)/transpose"(#loc30))
#loc44 = loc("jit(<lambda>)/real"(#loc31))
#loc45 = loc("jit(<lambda>)/imag"(#loc31))
#loc46 = loc("jit(<lambda>)/neg"(#loc31))
#loc47 = loc("jit(<lambda>)/complex"(#loc31))
#loc48 = loc("jit(<lambda>)/add"(#loc32))
#loc49 = loc("jit(<lambda>)/div"(#loc33))
#loc50 = loc("jit:"(#loc35))
""",
  mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x017\x07\x01\x05\t%\x01\x03\x0f\x03#\x13\x17\x1b\x1f#'+/37;?CGKOS\x03\xf2\x02>\x02=\x01\xab\x0f\x0b\x0f\x0b\x0f\x07\x0f\x0f\x0b\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x1b\x0f\x0b\x1b\x0f\x0b\x1b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x0f\x1b\x0f\x0b\x0f\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0f\x1b\x0b\x0f\x0f\x1b\x0f\x0b#\x0b\x0b\x0b\x03G\x0f\x0b\x0b\x0f\x0b\x1f\x0f\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0bO\x0f\x0b/OOO\x1b\x0b\x0f\x0b\x0b\x13\x0b\x0b\x05\x15O\x13\x0f\x0f\x13\x0f\x0f\x13\x0f\x0b\x03\x1b\x0b\x0b\x0b\x0b\x0f\x13\x17\x1f/\x0f\x0bO/\x01\x05\x0b\x0f\x039\x17\x0f\x0f\x07\x07\x07\x13\x17\x0b\x17\x07\x07\x17\x0f\x13\x17\x17\x13\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02\x86\n\x1dOQ\x05+\x15/\x13\x05-\x15a\x13\x1f\x1d)+\x15}\t\x05/\x1539\x11\x03\x05\x1d?\x05\x1dA\x05\x1d\x97\x99\x03\x07\x1f!#\x15%\x15\x051\x11\x01\x00\x053\x055\x057\x059\x1d-\x05\x05;\x1d\x111-\x03\x07!?\x8b\x1d57\x05=-\x03\x07+7u\x1d;=\x05?-\x03\x07+\t\xad\x05A\x05C\x1dE\x05\x05E\x1dI\x05\x05G\x1dM\x05\x05I\x05K\x15S\x13\x1d\x11U-\x03\x07!\x1f\xd5\x1dY[\x05M\x15]\t\x1d\x07_-\x03\x07\x19=\x8f\x1d\x11c-\x03\x07!Q\x89\x1dgi\x05O\x15k\t\x1d\x07m-\x03\x07\x19%\x9f\x1dqs\x05Q\x15u\t\x1d\x07w-\x03\x07\x1bG\x81\x1d{\x0f\x05S\x1d\x07\x7f-\x03\x07\x1b5\x83\x1d\x83\x0f\x05U\x1d\x87\x0f\x05W\x1d\x8b\x0f\x05Y\x1d\x8f\x91\x05[\x15\x93\t\x1d\x07\x95-\x03\x07\x1b!\x83\x05]\x15\x9b\t\x1d\x07\x9d-\x03\x07\x1b\x1f\x8f\x1d\xa1\x05\x05_\x03\x07\xa5\xe1\xa7\xeb\xa9\xf1\x05a\x05c\x05e\x1f'\x01\x03\x01\x1dg\x03\x03\xcd\x1di\x1f\t\t\x00\x00\x00\x00\x13\x0b\x01\t\x07\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00##\x03\x05\xc1\xc5\r\x03\xaf\xc3\x1dk\r\x03\xaf\xc7\x1dm\x1do\x1dq\r\x01#%\x1ds\x1f\x07!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x05\x07\x05\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x07!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x07!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\xe3\xe5\xe7\xe9\x1du\x13+\x00\x1dw\x05\x03\r\x03\xed\xef\x1dy\x1d{\x15\x0b\x11\x11\x11\x11\x11\x03\xf3\x07\xf9\xff\x06\x02\x01\x01\x01\x01\x01\x13\x05\xf5\xf7\x11\x03\x01\x11\x03\x05\x13\x05\xfb\xfd\x11\x03\t\x11\x03\r\x13\x03\x02\x02\x11\x03\x11\x13\x01\x0b\x03\x1d}\x1d\x7f\x05\x01\x03\x03\xbb\x03\x03\"\x02\x15\x03\x01\x01\x01\x03\x07\xbb*\x02.\x02\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f1\x01\x07\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f;\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x15)\x01\x15)\x01\x1b\x1d\x01\x0b)\x03\x11\x0f)\x05\x11\x11\x1b\x03\x0f)\x05\x11\x11\x0f\x13\x1b)\x05\x11\x11\r)\x01\x0f)\x03\t\x0b\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03A\x15!)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03\x11\r)\x03\x05\x0b\x04\xea\x04\x05\x01Q\x0b\x1d\x01\x07\x04\xc2\x04\x03\x01\t\x0bP\x0b\x03\x07\x04F\x03\x039m\x05B\x01\x05\x03\x1f\x05B\x01\x07\x03\x07\x05B\x01\t\x03\t\x05B\x0b\x0b\x03\x07\x07BW\r\x03)\x13\x06e\x03\x05\x03\t\x15Fo\x0f\x03\x05\x03\x0b\x17\x06y\x03\x17\x03\r\x19\x06\x81\x03\x17\x03\r\x1b\x06\x85\x03\x17\x03\x11\x1d\x06\x89\x03\x05\x05\x0f\x13\r\x06\x8d\x03\x05\x05\x0b\x15\x03F\x1b\x11\x03\x05\x03\x07\x1f\x06\x1b\x03\x05\x05\x17\x19!F\x9f\x13\x03\x05\x03\x1b#G\x01\xa3\x15\x07\x05\x11\t\x03\x1d\x03F\x01\x11\x03\t\x03\x05\x0fF\x01\x17\x033\x05#%\x03F\x01\x11\x035\x03'\x03F\x01\x11\x03\x05\x03\x03\x03F\x01\x19\x03\x1d\x03)\t\x06\x01\x03\x05\x07-\x1f+\x03F\x01\x11\x037\x03'\x03F\x01\x11\x03\x11\x03\x01\x03F\x01\x1b\x039\x031\t\x06\x01\x03\x11\x075!3\x11\x04\x0b\x05/7\x0bP\r\x1d\x07\x04\x9b\x03\x15+\x03\t\x00\x05B\r\x1f\x03\x07\x05B\r\t\x03\t\x07B\x17\r\x03\x13\x03F\x19\x11\x03\x13\x03\x05\r\x06\x19\x03\x13\x05\x07\t\x07B\x17!\x03\x13\x0fFC#\x03\x1d\x05\x0b\r\x03FG\x11\x03\x05\x03\x03\t\x06K\x03\x05\x07\x0f\x01\x11\x11\x04\r\x03\x13\x06\x03\x01\x05\x01\x00\x8e\r\x81)\x03\x05\x1f\r\x15\x11\x0f\x0b\x15\x15\x0b!%3)1%%-%''1-''\x13#\x07\t\x0b\x13\x13%\x0b\x19%)9#\x1f\x13\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00<string>\x00Gen.eigh_input\x00Gen.eigh_harness\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00jit:\x00jit(<lambda>)/jit\x00<lambda>\x00<module>\x00iota\x00add\x00ge\x00broadcast_in_dim\x00select_n\x00jit(<lambda>)/eigh\x00jit(<lambda>)/iota\x00jit(<lambda>)/reshape\x00jit(<lambda>)/transpose\x00jit(<lambda>)/real\x00jit(<lambda>)/imag\x00jit(<lambda>)/neg\x00jit(<lambda>)/complex\x00jit(<lambda>)/add\x00jit(<lambda>)/div\x00jit(<lambda>)/jit(tril)\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00tril\x00result[0]\x00result[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00num_batch_dims\x000\x00\x00hipsolver_syevd_ffi\x00\x08\x83%\x05O\x01\x0b\xad\xbd\xbf\xc9\xcb\x03\xd9\x03\xdb\x03\xb5\x03\xdd\x03\xb7\x03\xdf\x03\xab\x03\xb3\x1f\n\x02\x0e\x02\x12\x02\xad\x16\x02\x1a\x02\x1e\x02&\x02\x07\xb92\x02\x056\x02\x05:\x02\x0b\xb1\xcf\xb1\xb3\xd1\x03\xd3\x03\xd5\x05\xb9\xd7",
  xla_call_module_version=10,
  nr_devices=1,
)  # End paste
