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

import datetime
import numpy
array = numpy.array
complex64 = numpy.complex64
float32 = numpy.float32
int32 = numpy.int32


data_2026_02_04 = {}


data_2026_02_04['f32'] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hip_lu_pivots_to_permutation', 'hipsolver_getrf_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(),
    expected_outputs=(array([[ 8. ,  9. , 10. , 11. ],
       [ 0. ,  1. ,  2. ,  3. ],
       [ 0.5,  0.5,  0. ,  0. ]], dtype=float32), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xf32> {jax.result_info = "result[0]"}, tensor<3xi32> {jax.result_info = "result[1]"}, tensor<3xi32> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc7)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc7)
    %c_0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc7)
    %0 = stablehlo.iota dim = 0 : tensor<12xf32> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<12xf32>) -> tensor<3x4xf32> loc(#loc9)
    %2:3 = stablehlo.custom_call @hipsolver_getrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=3, j=4, k=3, l=4, m=3}, custom>} : (tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<3xi32>, tensor<i32>) loc(#loc7)
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc7)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc7)
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc7)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc7)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc7)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x4xf32> loc(#loc7)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x4xi1> loc(#loc7)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x4xi1>, tensor<3x4xf32> loc(#loc7)
    %11 = stablehlo.custom_call @hip_lu_pivots_to_permutation(%4) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([j]) {i=3, j=3}, custom>} : (tensor<3xi32>) -> tensor<3xi32> loc(#loc10)
    return %10, %4, %11 : tensor<3x4xf32>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":463:11)
#loc2 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":462:26)
#loc3 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":462:14)
#loc4 = loc("jit(<lambda>)"(#loc1))
#loc5 = loc("jit(<lambda>)"(#loc2))
#loc6 = loc("jit(<lambda>)"(#loc3))
#loc7 = loc("lu"(#loc4))
#loc8 = loc("iota"(#loc5))
#loc9 = loc("reshape"(#loc6))
#loc10 = loc(""(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01)\x07\x01\x05\t\x17\x01\x03\x0f\x03\x15\x13\x17\x1b\x1f#'+/37\x03\xe1\x9f+\x01;\x0f\x07\x0b\x0b\x0f\x0f\x0b\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0b\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17##\x0f\x0b\x03K\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0bO/\x0f\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x1f\x0f\x0b\x0b\x0b\x0f\x0f\x17\x17\x0f\x0b\x0bO\x0b\x05\x1b\x0f\x0fK\x13\x13\x0f\x0f\x0f\x0f\x0b7\x0f\x0f\x01\x05\x0b\x0f\x03'\x13\x0f\x17\x07\x07\x07\x07\x07\x0f\x1b\x13\x13\x13\x13\x13\x0f\x17\x17\x13\x02\xf6\x05\x1d\x1f\x0b\x1f\x05\x1d\x05\x1f\x11\x03\x05\x1d\x05!\x05!\x05#\x05%\x03\x07\x15\x17\x19\t\x1b\t\x05'\x11\x01\x00\x05)\x05+\x05-\x05/\x17\x07>\x07\x17\x1d%'\x051\x1d\x05)\x17\x07:\x075\x1d-/\x053\x1d\x051\x17\x07:\x07\x1d\x03\x07\rA\x0fC\x11\x89\x03\x07\rA\x0fC\x11\x99\x1d9\x0b\x055\x03\x01\x1f!\x01\x1d7\r\x01\r\x03mo\x0b\x03\x1d5\x05\x01\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03M#\x17\x03\x07UY]\r\x03?W\x1d9\r\x03?[\x1d;\r\x03?_\x1d=\x1d?\x1dA\x1f\x15\t\x00\x00\xc0\x7f\x1f\x07\t\x00\x00\x00\x00\x1f\x07\t\x01\x00\x00\x00\x13\r\x01\x1dC\x1dE\x1dG\x03\x03K\x03\x03w\x15\x03\x01\x01\x01\x03\x07KM{\x1f\x1f\x01\t\x07\x07\x05\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1dI\x11\x03\x01\x11\x03\x05\x15\x0b\r\x11\r\x11\r\x03\x8b\x07\x8d\x93\x97\x01\x01\x01\x01\x01\x13\x05\x85\x87\x13\x05\x8f\x91\x11\x03\t\x11\x03\r\x13\x03\x95\x11\x03\x11\x13\x01\x15\x05\r\r\x03\x9b\x03\x9d\x01\x01\x01\x01\x01\x13\x03\x85\x13\x03\x87\x01\t\x01\x02\x02)\x03\r\x13)\x01\x13)\x05\r\x11\x0b\t\x1d\x13\x01\x1b)\x01\x0b\x11\x01\x07\t\x05\x05)\x031\x0b)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0f)\x03\x01\r)\x01\x11)\x05\x05\x05\x11)\x05\r\x11\x11)\x03\t\r\x04N\x02\x05\x01Q\x03\x13\x01\x07\x04&\x02\x03\x01\x05\tP\x03\x03\x07\x04\xff\x03#A\x05B\x01\x05\x03\x15\x05B\x01\x07\x03\x07\x05B\x01\t\x03\x07\x0bB#\x0b\x03\x19\r\x06+\x03\t\x03\x07\x07G\x013\r\x07\t\x05\x07\x03\t\x03F\x01\x0f\x03\x05\x03\x05\x0f\x06\x01\x03\x05\x05\r\x11\x03F\x01\x0f\x03\x07\x03\x03\x11F\x01\x11\x03#\x05\x0f\x15\x03F\x01\x0f\x03%\x03\x17\x03F\x01\x0f\x03\t\x03\x01\x03F\x01\x13\x03'\x03\x19\x13\x06\x01\x03\t\x07\x1d\x0b\x1b\x07G75\x15\x03\x05\x03\x13\x15\x04\x03\x07\x1f\x13!\x06\x03\x01\x05\x01\x00*\x08K;)\x05\x1f\x0f\x0b\x15\x15\x15!\x03\x11\x0b\x07\x19%)9%3)_\x1d\x15\x15\x17\x19\x17\x11\x11\x1f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00subtract_v1\x00compare_v1\x00select_v1\x00return_v1\x00jit(<lambda>)\x00/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00lu\x00iota\x00reshape\x00\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00num_batch_dims\x000\x00hipsolver_getrf_ffi\x00hip_lu_pivots_to_permutation\x00\x08W\x17\x05;\x01\x0b;QSac\x03e\x03g\x03i\x03k\x11EGq;Isuy\x03=\x05}\x7f\x03\x81\x11EG\x83;IO;O",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste



# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04['f64'] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hip_lu_pivots_to_permutation', 'hipsolver_getrf_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(),
    expected_outputs=(array([[ 8. ,  9. , 10. , 11. ],
       [ 0. ,  1. ,  2. ,  3. ],
       [ 0.5,  0.5,  0. ,  0. ]]), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xf64> {jax.result_info = "result[0]"}, tensor<3xi32> {jax.result_info = "result[1]"}, tensor<3xi32> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc7)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc7)
    %c_0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc7)
    %0 = stablehlo.iota dim = 0 : tensor<12xf64> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<12xf64>) -> tensor<3x4xf64> loc(#loc9)
    %2:3 = stablehlo.custom_call @hipsolver_getrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=3, j=4, k=3, l=4, m=3}, custom>} : (tensor<3x4xf64>) -> (tensor<3x4xf64>, tensor<3xi32>, tensor<i32>) loc(#loc7)
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc7)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc7)
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc7)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc7)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc7)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x4xf64> loc(#loc7)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x4xi1> loc(#loc7)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x4xi1>, tensor<3x4xf64> loc(#loc7)
    %11 = stablehlo.custom_call @hip_lu_pivots_to_permutation(%4) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([j]) {i=3, j=3}, custom>} : (tensor<3xi32>) -> tensor<3xi32> loc(#loc10)
    return %10, %4, %11 : tensor<3x4xf64>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":463:11)
#loc2 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":462:26)
#loc3 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":462:14)
#loc4 = loc("jit(<lambda>)"(#loc1))
#loc5 = loc("jit(<lambda>)"(#loc2))
#loc6 = loc("jit(<lambda>)"(#loc3))
#loc7 = loc("lu"(#loc4))
#loc8 = loc("iota"(#loc5))
#loc9 = loc("reshape"(#loc6))
#loc10 = loc(""(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01)\x07\x01\x05\t\x17\x01\x03\x0f\x03\x15\x13\x17\x1b\x1f#'+/37\x03\xe1\x9f+\x01;\x0f\x07\x0b\x0b\x0f\x0f\x0b\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0b\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17##\x0f\x0b\x03K\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0bO/\x0f\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f\x1f\x0f\x0b\x0b\x0b\x0f\x0f\x17\x17\x0f\x0b\x0bO\x0b\x05\x1b\x0f\x0fK\x13\x13\x0f\x0f\x0f\x0f\x0b7\x0f\x0f\x01\x05\x0b\x0f\x03'\x13\x0f\x17\x07\x07\x07\x07\x07\x0f\x1b\x13\x13\x13\x13\x13\x0f\x17\x17\x13\x02\x06\x06\x1d\x1f\x0b\x1f\x05\x1d\x05\x1f\x11\x03\x05\x1d\x05!\x05!\x05#\x05%\x03\x07\x15\x17\x19\t\x1b\t\x05'\x11\x01\x00\x05)\x05+\x05-\x05/\x17\x07>\x07\x17\x1d%'\x051\x1d\x05)\x17\x07:\x075\x1d-/\x053\x1d\x051\x17\x07:\x07\x1d\x03\x07\rA\x0fC\x11\x89\x03\x07\rA\x0fC\x11\x99\x1d9\x0b\x055\x03\x01\x1f!\x01\x1d7\r\x01\r\x03mo\x0b\x03\x1d5\x05\x01\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03M#\x17\x03\x07UY]\r\x03?W\x1d9\r\x03?[\x1d;\r\x03?_\x1d=\x1d?\x1dA\x1f\x15\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x07\t\x00\x00\x00\x00\x1f\x07\t\x01\x00\x00\x00\x13\r\x01\x1dC\x1dE\x1dG\x03\x03K\x03\x03w\x15\x03\x01\x01\x01\x03\x07KM{\x1f\x1f\x01\t\x07\x07\x05\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1dI\x11\x03\x01\x11\x03\x05\x15\x0b\r\x11\r\x11\r\x03\x8b\x07\x8d\x93\x97\x01\x01\x01\x01\x01\x13\x05\x85\x87\x13\x05\x8f\x91\x11\x03\t\x11\x03\r\x13\x03\x95\x11\x03\x11\x13\x01\x15\x05\r\r\x03\x9b\x03\x9d\x01\x01\x01\x01\x01\x13\x03\x85\x13\x03\x87\x01\t\x01\x02\x02)\x03\r\x13)\x01\x13)\x05\r\x11\x0b\x0b\x1d\x13\x01\x1b)\x01\x0b\x11\x01\x07\t\x05\x05)\x031\x0b)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0f)\x03\x01\r)\x01\x11)\x05\x05\x05\x11)\x05\r\x11\x11)\x03\t\r\x04N\x02\x05\x01Q\x03\x13\x01\x07\x04&\x02\x03\x01\x05\tP\x03\x03\x07\x04\xff\x03#A\x05B\x01\x05\x03\x15\x05B\x01\x07\x03\x07\x05B\x01\t\x03\x07\x0bB#\x0b\x03\x19\r\x06+\x03\t\x03\x07\x07G\x013\r\x07\t\x05\x07\x03\t\x03F\x01\x0f\x03\x05\x03\x05\x0f\x06\x01\x03\x05\x05\r\x11\x03F\x01\x0f\x03\x07\x03\x03\x11F\x01\x11\x03#\x05\x0f\x15\x03F\x01\x0f\x03%\x03\x17\x03F\x01\x0f\x03\t\x03\x01\x03F\x01\x13\x03'\x03\x19\x13\x06\x01\x03\t\x07\x1d\x0b\x1b\x07G75\x15\x03\x05\x03\x13\x15\x04\x03\x07\x1f\x13!\x06\x03\x01\x05\x01\x00*\x08K;)\x05\x1f\x0f\x0b\x15\x15\x15!\x03\x11\x0b\x07\x19%)9%3)_\x1d\x15\x15\x17\x19\x17\x11\x11\x1f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00subtract_v1\x00compare_v1\x00select_v1\x00return_v1\x00jit(<lambda>)\x00/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00lu\x00iota\x00reshape\x00\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00num_batch_dims\x000\x00hipsolver_getrf_ffi\x00hip_lu_pivots_to_permutation\x00\x08W\x17\x05;\x01\x0b;QSac\x03e\x03g\x03i\x03k\x11EGq;Isuy\x03=\x05}\x7f\x03\x81\x11EG\x83;IO;O",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04['c64'] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hip_lu_pivots_to_permutation', 'hipsolver_getrf_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(),
    expected_outputs=(array([[ 8. +0.j,  9. +0.j, 10. +0.j, 11. +0.j],
       [ 0. +0.j,  1. +0.j,  2. +0.j,  3. +0.j],
       [ 0.5+0.j,  0.5+0.j,  0. +0.j,  0. +0.j]], dtype=complex64), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<3xi32> {jax.result_info = "result[1]"}, tensor<3xi32> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc7)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc7)
    %c_0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc7)
    %0 = stablehlo.iota dim = 0 : tensor<12xcomplex<f32>> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<12xcomplex<f32>>) -> tensor<3x4xcomplex<f32>> loc(#loc9)
    %2:3 = stablehlo.custom_call @hipsolver_getrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=3, j=4, k=3, l=4, m=3}, custom>} : (tensor<3x4xcomplex<f32>>) -> (tensor<3x4xcomplex<f32>>, tensor<3xi32>, tensor<i32>) loc(#loc7)
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc7)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc7)
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc7)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc7)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc7)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<3x4xcomplex<f32>> loc(#loc7)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x4xi1> loc(#loc7)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x4xi1>, tensor<3x4xcomplex<f32>> loc(#loc7)
    %11 = stablehlo.custom_call @hip_lu_pivots_to_permutation(%4) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([j]) {i=3, j=3}, custom>} : (tensor<3xi32>) -> tensor<3xi32> loc(#loc10)
    return %10, %4, %11 : tensor<3x4xcomplex<f32>>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":463:11)
#loc2 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":462:26)
#loc3 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":462:14)
#loc4 = loc("jit(<lambda>)"(#loc1))
#loc5 = loc("jit(<lambda>)"(#loc2))
#loc6 = loc("jit(<lambda>)"(#loc3))
#loc7 = loc("lu"(#loc4))
#loc8 = loc("iota"(#loc5))
#loc9 = loc("reshape"(#loc6))
#loc10 = loc(""(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01)\x07\x01\x05\t\x17\x01\x03\x0f\x03\x15\x13\x17\x1b\x1f#'+/37\x03\xe3\x9f-\x01;\x0f\x07\x0b\x0b\x0f\x0f\x0b\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0b\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17##\x0f\x0b\x03K\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0bO/\x0f\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f\x1f\x0f\x0b\x0b\x0b\x0f\x0f\x17\x17\x0f\x0b\x0bO\x0b\x05\x1b\x0f\x0fK\x13\x13\x0f\x0f\x0f\x0f\x0b7\x0f\x0f\x01\x05\x0b\x0f\x03)\x13\x0f\x17\x0b\x07\x07\x07\x07\x0f\x1b\x07\x13\x13\x13\x13\x13\x0f\x17\x17\x13\x02\x0e\x06\x1d\x1f\x0b\x1f\x05\x1d\x05\x1f\x11\x03\x05\x1d\x05!\x05!\x05#\x05%\x03\x07\x15\x17\x19\t\x1b\t\x05'\x11\x01\x00\x05)\x05+\x05-\x05/\x17\x07>\x07\x17\x1d%'\x051\x1d\x05)\x17\x07:\x075\x1d-/\x053\x1d\x051\x17\x07:\x07\x1d\x03\x07\rA\x0fC\x11\x89\x03\x07\rA\x0fC\x11\x99\x1d9\x0b\x055\x03\x01\x1f#\x01\x1d7\r\x01\r\x03mo\x0b\x03\x1d5\x05\x01\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03M#\x17\x03\x07UY]\r\x03?W\x1d9\r\x03?[\x1d;\r\x03?_\x1d=\x1d?\x1dA\x1f\x15\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x07\t\x00\x00\x00\x00\x1f\x07\t\x01\x00\x00\x00\x13\r\x01\x1dC\x1dE\x1dG\x03\x03K\x03\x03w\x15\x03\x01\x01\x01\x03\x07KM{\x1f!\x01\t\x07\x07\x05\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1dI\x11\x03\x01\x11\x03\x05\x15\x0b\r\x11\r\x11\r\x03\x8b\x07\x8d\x93\x97\x01\x01\x01\x01\x01\x13\x05\x85\x87\x13\x05\x8f\x91\x11\x03\t\x11\x03\r\x13\x03\x95\x11\x03\x11\x13\x01\x15\x05\r\r\x03\x9b\x03\x9d\x01\x01\x01\x01\x01\x13\x03\x85\x13\x03\x87\x01\t\x01\x02\x02)\x03\r\x13)\x01\x13)\x05\r\x11\x0b\x03\x19\x1d\x13\x01\x1b)\x01\x0b\x11\x01\x07\t\x05\x05\t)\x031\x0b)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0f)\x03\x01\r)\x01\x11)\x05\x05\x05\x11)\x05\r\x11\x11)\x03\t\r\x04N\x02\x05\x01Q\x03\x13\x01\x07\x04&\x02\x03\x01\x05\tP\x03\x03\x07\x04\xff\x03#A\x05B\x01\x05\x03\x15\x05B\x01\x07\x03\x07\x05B\x01\t\x03\x07\x0bB#\x0b\x03\x1b\r\x06+\x03\t\x03\x07\x07G\x013\r\x07\t\x05\x07\x03\t\x03F\x01\x0f\x03\x05\x03\x05\x0f\x06\x01\x03\x05\x05\r\x11\x03F\x01\x0f\x03\x07\x03\x03\x11F\x01\x11\x03%\x05\x0f\x15\x03F\x01\x0f\x03'\x03\x17\x03F\x01\x0f\x03\t\x03\x01\x03F\x01\x13\x03)\x03\x19\x13\x06\x01\x03\t\x07\x1d\x0b\x1b\x07G75\x15\x03\x05\x03\x13\x15\x04\x03\x07\x1f\x13!\x06\x03\x01\x05\x01\x00*\x08K;)\x05\x1f\x0f\x0b\x15\x15\x15!\x03\x11\x0b\x07\x19%)9%3)_\x1d\x15\x15\x17\x19\x17\x11\x11\x1f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00subtract_v1\x00compare_v1\x00select_v1\x00return_v1\x00jit(<lambda>)\x00/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00lu\x00iota\x00reshape\x00\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00num_batch_dims\x000\x00hipsolver_getrf_ffi\x00hip_lu_pivots_to_permutation\x00\x08W\x17\x05;\x01\x0b;QSac\x03e\x03g\x03i\x03k\x11EGq;Isuy\x03=\x05}\x7f\x03\x81\x11EG\x83;IO;O",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04['c128'] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hip_lu_pivots_to_permutation', 'hipsolver_getrf_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(),
    expected_outputs=(array([[ 8. +0.j,  9. +0.j, 10. +0.j, 11. +0.j],
       [ 0. +0.j,  1. +0.j,  2. +0.j,  3. +0.j],
       [ 0.5+0.j,  0.5+0.j,  0. +0.j,  0. +0.j]]), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<3xi32> {jax.result_info = "result[1]"}, tensor<3xi32> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc7)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc7)
    %c_0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc7)
    %0 = stablehlo.iota dim = 0 : tensor<12xcomplex<f64>> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<12xcomplex<f64>>) -> tensor<3x4xcomplex<f64>> loc(#loc9)
    %2:3 = stablehlo.custom_call @hipsolver_getrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=3, j=4, k=3, l=4, m=3}, custom>} : (tensor<3x4xcomplex<f64>>) -> (tensor<3x4xcomplex<f64>>, tensor<3xi32>, tensor<i32>) loc(#loc7)
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc7)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc7)
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc7)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc7)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc7)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<3x4xcomplex<f64>> loc(#loc7)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x4xi1> loc(#loc7)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x4xi1>, tensor<3x4xcomplex<f64>> loc(#loc7)
    %11 = stablehlo.custom_call @hip_lu_pivots_to_permutation(%4) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([j]) {i=3, j=3}, custom>} : (tensor<3xi32>) -> tensor<3xi32> loc(#loc10)
    return %10, %4, %11 : tensor<3x4xcomplex<f64>>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":463:11)
#loc2 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":462:26)
#loc3 = loc("/rocm-jax/jax/tests/export_back_compat_test.py":462:14)
#loc4 = loc("jit(<lambda>)"(#loc1))
#loc5 = loc("jit(<lambda>)"(#loc2))
#loc6 = loc("jit(<lambda>)"(#loc3))
#loc7 = loc("lu"(#loc4))
#loc8 = loc("iota"(#loc5))
#loc9 = loc("reshape"(#loc6))
#loc10 = loc(""(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01)\x07\x01\x05\t\x17\x01\x03\x0f\x03\x15\x13\x17\x1b\x1f#'+/37\x03\xe3\x9f-\x01;\x0f\x07\x0b\x0b\x0f\x0f\x0b\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0b\x17\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17##\x0f\x0b\x03K\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0bO/\x0f\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0bO\x1f\x1f\x0f\x0b\x0b\x0b\x0f\x0f\x17\x17\x0f\x0b\x0bO\x0b\x05\x1b\x0f\x0fK\x13\x13\x0f\x0f\x0f\x0f\x0b7\x0f\x0f\x01\x05\x0b\x0f\x03)\x13\x0f\x17\x0b\x07\x07\x07\x07\x0f\x1b\x07\x13\x13\x13\x13\x13\x0f\x17\x17\x13\x02.\x06\x1d\x1f\x0b\x1f\x05\x1d\x05\x1f\x11\x03\x05\x1d\x05!\x05!\x05#\x05%\x03\x07\x15\x17\x19\t\x1b\t\x05'\x11\x01\x00\x05)\x05+\x05-\x05/\x17\x07>\x07\x17\x1d%'\x051\x1d\x05)\x17\x07:\x075\x1d-/\x053\x1d\x051\x17\x07:\x07\x1d\x03\x07\rA\x0fC\x11\x89\x03\x07\rA\x0fC\x11\x99\x1d9\x0b\x055\x03\x01\x1f#\x01\x1d7\r\x01\r\x03mo\x0b\x03\x1d5\x05\x01\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03M#\x17\x03\x07UY]\r\x03?W\x1d9\r\x03?[\x1d;\r\x03?_\x1d=\x1d?\x1dA\x1f\x15!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x07\t\x00\x00\x00\x00\x1f\x07\t\x01\x00\x00\x00\x13\r\x01\x1dC\x1dE\x1dG\x03\x03K\x03\x03w\x15\x03\x01\x01\x01\x03\x07KM{\x1f!\x01\t\x07\x07\x05\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1dI\x11\x03\x01\x11\x03\x05\x15\x0b\r\x11\r\x11\r\x03\x8b\x07\x8d\x93\x97\x01\x01\x01\x01\x01\x13\x05\x85\x87\x13\x05\x8f\x91\x11\x03\t\x11\x03\r\x13\x03\x95\x11\x03\x11\x13\x01\x15\x05\r\r\x03\x9b\x03\x9d\x01\x01\x01\x01\x01\x13\x03\x85\x13\x03\x87\x01\t\x01\x02\x02)\x03\r\x13)\x01\x13)\x05\r\x11\x0b\x03\x19\x1d\x13\x01\x1b)\x01\x0b\x11\x01\x07\t\x05\x05\x0b)\x031\x0b)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0f)\x03\x01\r)\x01\x11)\x05\x05\x05\x11)\x05\r\x11\x11)\x03\t\r\x04N\x02\x05\x01Q\x03\x13\x01\x07\x04&\x02\x03\x01\x05\tP\x03\x03\x07\x04\xff\x03#A\x05B\x01\x05\x03\x15\x05B\x01\x07\x03\x07\x05B\x01\t\x03\x07\x0bB#\x0b\x03\x1b\r\x06+\x03\t\x03\x07\x07G\x013\r\x07\t\x05\x07\x03\t\x03F\x01\x0f\x03\x05\x03\x05\x0f\x06\x01\x03\x05\x05\r\x11\x03F\x01\x0f\x03\x07\x03\x03\x11F\x01\x11\x03%\x05\x0f\x15\x03F\x01\x0f\x03'\x03\x17\x03F\x01\x0f\x03\t\x03\x01\x03F\x01\x13\x03)\x03\x19\x13\x06\x01\x03\t\x07\x1d\x0b\x1b\x07G75\x15\x03\x05\x03\x13\x15\x04\x03\x07\x1f\x13!\x06\x03\x01\x05\x01\x00*\x08K;)\x05\x1f\x0f\x0b\x15\x15\x15!\x03\x11\x0b\x07\x19%)9%3)_\x1d\x15\x15\x17\x19\x17\x11\x11\x1f\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00subtract_v1\x00compare_v1\x00select_v1\x00return_v1\x00jit(<lambda>)\x00/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00lu\x00iota\x00reshape\x00\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00num_batch_dims\x000\x00hipsolver_getrf_ffi\x00hip_lu_pivots_to_permutation\x00\x08W\x17\x05;\x01\x0b;QSac\x03e\x03g\x03i\x03k\x11EGq;Isuy\x03=\x05}\x7f\x03\x81\x11EG\x83;IO;O",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
