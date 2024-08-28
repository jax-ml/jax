# Copyright 2024 The JAX Authors.
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
from numpy import array, int32, float32, complex64

data_2024_08_19 = {}

data_2024_08_19["f32"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cu_lu_pivots_to_permutation', 'cusolver_getrf_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([[ 8. ,  9. , 10. , 11. ],
       [ 0. ,  1. ,  2. ,  3. ],
       [ 0.5,  0.5,  0. ,  0. ]], dtype=float32), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<12xf32> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<12xf32>) -> tensor<3x4xf32> loc(#loc5)
    %2:3 = stablehlo.custom_call @cusolver_getrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_0 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x4xf32> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x4xi1>, tensor<3x4xf32> loc(#loc6)
    %11 = stablehlo.custom_call @cu_lu_pivots_to_permutation(%4) {mhlo.backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<3xi32>) -> tensor<3xi32> loc(#loc7)
    return %10, %4, %11 : tensor<3x4xf32>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":441:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":441:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":442:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(12,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 4) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01%\x05\x01\x03\x01\x03\x05\x03\x15\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x03\xe9\xab+\x01c\x0f\x13\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0b+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x17\x0f\x0b\x17S\x0b\x13\x13\x1b\x0b\x0b\x13\x13S\x0f\x0b\x03I\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0bO/\x0f\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0f\x0b\x0f\x0f\x17\x17\x0f\x1f\x0f\x1f\x0b\x0b\x1fO\x0b\x01\x05\x0b\x0f\x03'\x13\x0f\x17\x07\x07\x07\x07\x07\x0f\x1b\x13\x13\x13\x13\x13\x0f\x17\x17\x13\x02^\x06\x1dM!\x03\x03#\x9d\x1f\x05\x1b\x05\x1d\x11\x03\x05\x05\x1f\x05!\x05#\x05%\x05'\x05)\x05+\x05-\x05/\x051\x17\x07\xea\x06\x17\x053\x03\t')+\x0b-\x0b\r/\x055\x11\x01\x00\x057\x059\x05;\x03\x0b3c5y7{\r\x899\x8b\x05=\x05?\x05A\x05C\x03\x03=\x8d\x05E\x1dAC\x05G\x17\x07\xe6\x065\x1dGI\x05I\x17\x07\xe6\x06\x1d\x03\x13\x0fk\x11m\x13\x8f\x15c\x17o\x19q\x1b\x91\x1d\x93\x1f\x97\x05K\x03\x03\t\x9b\x03\x03\t\x9f\x03\x05U\xa1W\xa3\x05M\x05O\x03\x03\t\xa5\x03\x03#\xa7\x03\x13\x0fk\x11m\x13\xa9\x15c\x17o\x19q\x1bw\x1dc\x1fw\x1da!\x05Q\x03\x01\x1dS\x1dU\x1dW\x0b\x03\x1dY\x05\x01\r\x01\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03u#\x17\x03\x07}\x81\x85\r\x05e\x7fgi\x1d[\r\x05e\x83gi\x1d]\r\x05e\x87gi\x1d_\x1da\x1dc\x13\r\x01\x1de\x03\x03s\x03\x03\x95\x15\x03\x01\x01\x01\x03\x07su\x99\x1f\x1f\x01\x1f\x07\t\x01\x00\x00\x00\x1f!\x01\x1f\x07\t\x00\x00\x00\x00\t\x07\x07\x05\x1f\x15\t\x00\x00\xc0\x7f\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1dg\x01\t\x01\x02\x02)\x03\r\x13)\x01\x13)\x05\r\x11\x0b\t\x1d\x13\x01\x1b)\x01\x0b\x11\x01\x07\t\x05\x05)\x031\x0b)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0f)\x03\x01\r)\x01\x11)\x05\x05\x05\x11)\x05\r\x11\x11)\x03\t\r\x04.\x02\x05\x01\x11\x05%\x07\x03\x01\x05\t\x11\x051\x07\x03#A\x0b\x03?;\x03\x19\r\x06E\x03\t\x03\x01\x07\x07\x01K\x07\t\x05\x07\x03\x03\x05\x03\x01O\x03\x07\x03\x07\x01\x03\x03\x05\x03\x0b\x0f\x06\x01\x03\x05\x05\x07\r\x05\x03\x01Q\x03\x07\x03\x07\x01\x03\x03\x07\x03\x11\x11\x07\x01S\x03#\x05\t\x13\x03\x07\x01\x03\x03%\x03\x15\x05\x03\x01Y\x03\x15\x03\x07\x01\x03\x03\t\x03\x19\x03\x07\x01[\x03'\x03\x17\x13\x06\x01\x03\t\x07\x1d\x05\x1b\x07\x07_]\x03\x05\x03\x0f\x15\x04\x05\x07\x1f\x0f!\x06\x03\x01\x05\x01\x00\xe2\x0ei9'\x0f\x0b\t\t\t\x03\x11#!\x8b+\x1b7\x85\x89\x1f\x1f\x15\x1d\x15\x1b%)9+\x1f/!)!)#\x1f\x19\x13\ri\x15\x15\x17\x19\x17\x11\x11\x1f\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00subtract_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00broadcast_dimensions\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(12,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 4) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00compare_type\x00comparison_direction\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00[2]\x00main\x00public\x00cusolver_getrf_ffi\x00cu_lu_pivots_to_permutation\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_08_19["f64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cu_lu_pivots_to_permutation', 'cusolver_getrf_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([[ 8. ,  9. , 10. , 11. ],
       [ 0. ,  1. ,  2. ,  3. ],
       [ 0.5,  0.5,  0. ,  0. ]]), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<12xf64> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<12xf64>) -> tensor<3x4xf64> loc(#loc5)
    %2:3 = stablehlo.custom_call @cusolver_getrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x4xf64>) -> (tensor<3x4xf64>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_0 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x4xf64> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x4xi1>, tensor<3x4xf64> loc(#loc6)
    %11 = stablehlo.custom_call @cu_lu_pivots_to_permutation(%4) {mhlo.backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<3xi32>) -> tensor<3xi32> loc(#loc7)
    return %10, %4, %11 : tensor<3x4xf64>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":441:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":441:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":442:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(12,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 4) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01%\x05\x01\x03\x01\x03\x05\x03\x15\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x03\xe9\xab+\x01c\x0f\x13\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0b+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x17\x0f\x0b\x17S\x0b\x13\x13\x1b\x0b\x0b\x13\x13S\x0f\x0b\x03I\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0bO/\x0f\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0f\x0b\x0f\x0f\x17\x17\x0f\x1f\x0f\x1f\x0b\x0b/O\x0b\x01\x05\x0b\x0f\x03'\x13\x0f\x17\x07\x07\x07\x07\x07\x0f\x1b\x13\x13\x13\x13\x13\x0f\x17\x17\x13\x02n\x06\x1dM!\x03\x03#\x9d\x1f\x05\x1b\x05\x1d\x11\x03\x05\x05\x1f\x05!\x05#\x05%\x05'\x05)\x05+\x05-\x05/\x051\x17\x07\xea\x06\x17\x053\x03\t')+\x0b-\x0b\r/\x055\x11\x01\x00\x057\x059\x05;\x03\x0b3c5y7{\r\x899\x8b\x05=\x05?\x05A\x05C\x03\x03=\x8d\x05E\x1dAC\x05G\x17\x07\xe6\x065\x1dGI\x05I\x17\x07\xe6\x06\x1d\x03\x13\x0fk\x11m\x13\x8f\x15c\x17o\x19q\x1b\x91\x1d\x93\x1f\x97\x05K\x03\x03\t\x9b\x03\x03\t\x9f\x03\x05U\xa1W\xa3\x05M\x05O\x03\x03\t\xa5\x03\x03#\xa7\x03\x13\x0fk\x11m\x13\xa9\x15c\x17o\x19q\x1bw\x1dc\x1fw\x1da!\x05Q\x03\x01\x1dS\x1dU\x1dW\x0b\x03\x1dY\x05\x01\r\x01\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03u#\x17\x03\x07}\x81\x85\r\x05e\x7fgi\x1d[\r\x05e\x83gi\x1d]\r\x05e\x87gi\x1d_\x1da\x1dc\x13\r\x01\x1de\x03\x03s\x03\x03\x95\x15\x03\x01\x01\x01\x03\x07su\x99\x1f\x1f\x01\x1f\x07\t\x01\x00\x00\x00\x1f!\x01\x1f\x07\t\x00\x00\x00\x00\t\x07\x07\x05\x1f\x15\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1dg\x01\t\x01\x02\x02)\x03\r\x13)\x01\x13)\x05\r\x11\x0b\x0b\x1d\x13\x01\x1b)\x01\x0b\x11\x01\x07\t\x05\x05)\x031\x0b)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0f)\x03\x01\r)\x01\x11)\x05\x05\x05\x11)\x05\r\x11\x11)\x03\t\r\x04.\x02\x05\x01\x11\x05%\x07\x03\x01\x05\t\x11\x051\x07\x03#A\x0b\x03?;\x03\x19\r\x06E\x03\t\x03\x01\x07\x07\x01K\x07\t\x05\x07\x03\x03\x05\x03\x01O\x03\x07\x03\x07\x01\x03\x03\x05\x03\x0b\x0f\x06\x01\x03\x05\x05\x07\r\x05\x03\x01Q\x03\x07\x03\x07\x01\x03\x03\x07\x03\x11\x11\x07\x01S\x03#\x05\t\x13\x03\x07\x01\x03\x03%\x03\x15\x05\x03\x01Y\x03\x15\x03\x07\x01\x03\x03\t\x03\x19\x03\x07\x01[\x03'\x03\x17\x13\x06\x01\x03\t\x07\x1d\x05\x1b\x07\x07_]\x03\x05\x03\x0f\x15\x04\x05\x07\x1f\x0f!\x06\x03\x01\x05\x01\x00\xe2\x0ei9'\x0f\x0b\t\t\t\x03\x11#!\x8b+\x1b7\x85\x89\x1f\x1f\x15\x1d\x15\x1b%)9+\x1f/!)!)#\x1f\x19\x13\ri\x15\x15\x17\x19\x17\x11\x11\x1f\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00subtract_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00broadcast_dimensions\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(12,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 4) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00compare_type\x00comparison_direction\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00[2]\x00main\x00public\x00cusolver_getrf_ffi\x00cu_lu_pivots_to_permutation\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_08_19["c64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cu_lu_pivots_to_permutation', 'cusolver_getrf_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([[ 8. +0.j,  9. +0.j, 10. +0.j, 11. +0.j],
       [ 0. +0.j,  1. +0.j,  2. +0.j,  3. +0.j],
       [ 0.5+0.j,  0.5+0.j,  0. +0.j,  0. +0.j]], dtype=complex64), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<12xcomplex<f32>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<12xcomplex<f32>>) -> tensor<3x4xcomplex<f32>> loc(#loc5)
    %2:3 = stablehlo.custom_call @cusolver_getrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x4xcomplex<f32>>) -> (tensor<3x4xcomplex<f32>>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_0 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<3x4xcomplex<f32>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x4xi1>, tensor<3x4xcomplex<f32>> loc(#loc6)
    %11 = stablehlo.custom_call @cu_lu_pivots_to_permutation(%4) {mhlo.backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<3xi32>) -> tensor<3xi32> loc(#loc7)
    return %10, %4, %11 : tensor<3x4xcomplex<f32>>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":441:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":441:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":442:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(12,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 4) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01%\x05\x01\x03\x01\x03\x05\x03\x15\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x03\xeb\xab-\x01c\x0f\x13\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0b+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x17\x0f\x0b\x17S\x0b\x13\x13\x1b\x0b\x0b\x13\x13S\x0f\x0b\x03I\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0bO/\x0f\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0f\x0b\x0f\x0f\x17\x17\x0f\x1f\x0f\x1f\x0b\x0b/O\x0b\x01\x05\x0b\x0f\x03)\x13\x0f\x17\x0b\x07\x07\x07\x07\x0f\x1b\x07\x13\x13\x13\x13\x13\x0f\x17\x17\x13\x02v\x06\x1dM!\x03\x03#\x9d\x1f\x05\x1b\x05\x1d\x11\x03\x05\x05\x1f\x05!\x05#\x05%\x05'\x05)\x05+\x05-\x05/\x051\x17\x07\xea\x06\x17\x053\x03\t')+\x0b-\x0b\r/\x055\x11\x01\x00\x057\x059\x05;\x03\x0b3c5y7{\r\x899\x8b\x05=\x05?\x05A\x05C\x03\x03=\x8d\x05E\x1dAC\x05G\x17\x07\xe6\x065\x1dGI\x05I\x17\x07\xe6\x06\x1d\x03\x13\x0fk\x11m\x13\x8f\x15c\x17o\x19q\x1b\x91\x1d\x93\x1f\x97\x05K\x03\x03\t\x9b\x03\x03\t\x9f\x03\x05U\xa1W\xa3\x05M\x05O\x03\x03\t\xa5\x03\x03#\xa7\x03\x13\x0fk\x11m\x13\xa9\x15c\x17o\x19q\x1bw\x1dc\x1fw\x1da!\x05Q\x03\x01\x1dS\x1dU\x1dW\x0b\x03\x1dY\x05\x01\r\x01\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03u#\x17\x03\x07}\x81\x85\r\x05e\x7fgi\x1d[\r\x05e\x83gi\x1d]\r\x05e\x87gi\x1d_\x1da\x1dc\x13\r\x01\x1de\x03\x03s\x03\x03\x95\x15\x03\x01\x01\x01\x03\x07su\x99\x1f!\x01\x1f\x07\t\x01\x00\x00\x00\x1f#\x01\x1f\x07\t\x00\x00\x00\x00\t\x07\x07\x05\x1f\x15\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1dg\x01\t\x01\x02\x02)\x03\r\x13)\x01\x13)\x05\r\x11\x0b\x03\x19\x1d\x13\x01\x1b)\x01\x0b\x11\x01\x07\t\x05\x05\t)\x031\x0b)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0f)\x03\x01\r)\x01\x11)\x05\x05\x05\x11)\x05\r\x11\x11)\x03\t\r\x04.\x02\x05\x01\x11\x05%\x07\x03\x01\x05\t\x11\x051\x07\x03#A\x0b\x03?;\x03\x1b\r\x06E\x03\t\x03\x01\x07\x07\x01K\x07\t\x05\x07\x03\x03\x05\x03\x01O\x03\x07\x03\x07\x01\x03\x03\x05\x03\x0b\x0f\x06\x01\x03\x05\x05\x07\r\x05\x03\x01Q\x03\x07\x03\x07\x01\x03\x03\x07\x03\x11\x11\x07\x01S\x03%\x05\t\x13\x03\x07\x01\x03\x03'\x03\x15\x05\x03\x01Y\x03\x15\x03\x07\x01\x03\x03\t\x03\x19\x03\x07\x01[\x03)\x03\x17\x13\x06\x01\x03\t\x07\x1d\x05\x1b\x07\x07_]\x03\x05\x03\x0f\x15\x04\x05\x07\x1f\x0f!\x06\x03\x01\x05\x01\x00\xea\x0ei9'\x0f\x0b\t\t\t\x03\x11#!\x8b+\x1b7\x85\x8d\x1f\x1f\x15\x1d\x15\x1b%)9+\x1f/!)!)#\x1f\x19\x13\ri\x15\x15\x17\x19\x17\x11\x11\x1f\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00subtract_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00broadcast_dimensions\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(12,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 4) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00compare_type\x00comparison_direction\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00[2]\x00main\x00public\x00cusolver_getrf_ffi\x00cu_lu_pivots_to_permutation\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_08_19["c128"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cu_lu_pivots_to_permutation', 'cusolver_getrf_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([[ 8. +0.j,  9. +0.j, 10. +0.j, 11. +0.j],
       [ 0. +0.j,  1. +0.j,  2. +0.j,  3. +0.j],
       [ 0.5+0.j,  0.5+0.j,  0. +0.j,  0. +0.j]]), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<12xcomplex<f64>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<12xcomplex<f64>>) -> tensor<3x4xcomplex<f64>> loc(#loc5)
    %2:3 = stablehlo.custom_call @cusolver_getrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x4xcomplex<f64>>) -> (tensor<3x4xcomplex<f64>>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_0 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<3x4xcomplex<f64>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x4xi1>, tensor<3x4xcomplex<f64>> loc(#loc6)
    %11 = stablehlo.custom_call @cu_lu_pivots_to_permutation(%4) {mhlo.backend_config = {}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<3xi32>) -> tensor<3xi32> loc(#loc7)
    return %10, %4, %11 : tensor<3x4xcomplex<f64>>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":441:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":441:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":442:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(12,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 4) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01%\x05\x01\x03\x01\x03\x05\x03\x15\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x03\xeb\xab-\x01c\x0f\x13\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0b+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x17\x0f\x0b\x17S\x0b\x13\x13\x1b\x0b\x0b\x13\x13S\x0f\x0b\x03I\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0bO/\x0f\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0f\x0b\x0f\x0f\x17\x17\x0f\x1f\x0f\x1f\x0b\x0bOO\x0b\x01\x05\x0b\x0f\x03)\x13\x0f\x17\x0b\x07\x07\x07\x07\x0f\x1b\x07\x13\x13\x13\x13\x13\x0f\x17\x17\x13\x02\x96\x06\x1dM!\x03\x03#\x9d\x1f\x05\x1b\x05\x1d\x11\x03\x05\x05\x1f\x05!\x05#\x05%\x05'\x05)\x05+\x05-\x05/\x051\x17\x07\xea\x06\x17\x053\x03\t')+\x0b-\x0b\r/\x055\x11\x01\x00\x057\x059\x05;\x03\x0b3c5y7{\r\x899\x8b\x05=\x05?\x05A\x05C\x03\x03=\x8d\x05E\x1dAC\x05G\x17\x07\xe6\x065\x1dGI\x05I\x17\x07\xe6\x06\x1d\x03\x13\x0fk\x11m\x13\x8f\x15c\x17o\x19q\x1b\x91\x1d\x93\x1f\x97\x05K\x03\x03\t\x9b\x03\x03\t\x9f\x03\x05U\xa1W\xa3\x05M\x05O\x03\x03\t\xa5\x03\x03#\xa7\x03\x13\x0fk\x11m\x13\xa9\x15c\x17o\x19q\x1bw\x1dc\x1fw\x1da!\x05Q\x03\x01\x1dS\x1dU\x1dW\x0b\x03\x1dY\x05\x01\r\x01\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03u#\x17\x03\x07}\x81\x85\r\x05e\x7fgi\x1d[\r\x05e\x83gi\x1d]\r\x05e\x87gi\x1d_\x1da\x1dc\x13\r\x01\x1de\x03\x03s\x03\x03\x95\x15\x03\x01\x01\x01\x03\x07su\x99\x1f!\x01\x1f\x07\t\x01\x00\x00\x00\x1f#\x01\x1f\x07\t\x00\x00\x00\x00\t\x07\x07\x05\x1f\x15!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1dg\x01\t\x01\x02\x02)\x03\r\x13)\x01\x13)\x05\r\x11\x0b\x03\x19\x1d\x13\x01\x1b)\x01\x0b\x11\x01\x07\t\x05\x05\x0b)\x031\x0b)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0f)\x03\x01\r)\x01\x11)\x05\x05\x05\x11)\x05\r\x11\x11)\x03\t\r\x04.\x02\x05\x01\x11\x05%\x07\x03\x01\x05\t\x11\x051\x07\x03#A\x0b\x03?;\x03\x1b\r\x06E\x03\t\x03\x01\x07\x07\x01K\x07\t\x05\x07\x03\x03\x05\x03\x01O\x03\x07\x03\x07\x01\x03\x03\x05\x03\x0b\x0f\x06\x01\x03\x05\x05\x07\r\x05\x03\x01Q\x03\x07\x03\x07\x01\x03\x03\x07\x03\x11\x11\x07\x01S\x03%\x05\t\x13\x03\x07\x01\x03\x03'\x03\x15\x05\x03\x01Y\x03\x15\x03\x07\x01\x03\x03\t\x03\x19\x03\x07\x01[\x03)\x03\x17\x13\x06\x01\x03\t\x07\x1d\x05\x1b\x07\x07_]\x03\x05\x03\x0f\x15\x04\x05\x07\x1f\x0f!\x06\x03\x01\x05\x01\x00\xee\x0ei9'\x0f\x0b\t\t\t\x03\x11#!\x8b+\x1b7\x85\x8f\x1f\x1f\x15\x1d\x15\x1b%)9+\x1f/!)!)#\x1f\x19\x13\ri\x15\x15\x17\x19\x17\x11\x11\x1f\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00subtract_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00broadcast_dimensions\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(12,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 4) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00compare_type\x00comparison_direction\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00[2]\x00main\x00public\x00cusolver_getrf_ffi\x00cu_lu_pivots_to_permutation\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
