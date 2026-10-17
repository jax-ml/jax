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
int32 = np.int32

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13 = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapi_lu_pivots_to_permutation'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(),
    expected_outputs=(array([[[0, 1, 2, 3, 4, 5, 6, 7],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7]],

       [[0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7]]], dtype=int32),),
    mlir_module_text=r"""
module @jit__lambda attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x8xi32> {jax.result_info = "result"}) {
    %0 = stablehlo.iota dim = 0 : tensor<24xi32> loc(#loc9)
    %1 = stablehlo.reshape %0 : (tensor<24xi32>) -> tensor<2x3x4xi32> loc(#loc10)
    %2 = stablehlo.custom_call @oneapi_lu_pivots_to_permutation(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "2"}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, l]) {i=2, j=3, k=4, l=8}, custom>} : (tensor<2x3x4xi32>) -> tensor<2x3x8xi32> loc(#loc11)
    return %2 : tensor<2x3x8xi32> loc(#loc8)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":428:26)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":428:14)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":429:11)
#loc4 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":447:4)
#loc5 = loc("jit(<lambda>)"(#loc1))
#loc6 = loc("jit(<lambda>)"(#loc2))
#loc7 = loc("jit(<lambda>)"(#loc3))
#loc8 = loc("jit(<lambda>)"(#loc4))
#loc9 = loc("iota"(#loc5))
#loc10 = loc("reshape"(#loc6))
#loc11 = loc("lu_pivots_to_permutation"(#loc7))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01\x1f\x07\x01\x05\t\r\x01\x03\x0f\x03\x0b\x13\x17\x1b\x1f#\x03\x99m\x15\x019\x0b\x0b\x0f\x07#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x17\x0f\x0b\x0f\x17#\x0b\x0b\x0b\x0f\x0b\x0f\x17\x0f\x17\x03'\x0b\x0f\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0bo\x05\x0f\x0f\x0f?\x17\x0f\x17\x0f\x01\x05\x0b\x0f\x03\x11\x07\x1b\x13\x13\x07\x1b\x13\x07\x02\x8e\x03\x05\x13\x05\x15\x11\x03\x05\x1f\x03\x07\x0b\r\x0f\x05\x11\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x17\x19\x05\x1f\x1d\x01\x1b\x17\x03\xb2\x065\x1d\x1f!\x05!\x1d\x01#\x17\x03\xb2\x06\x1d\x03\x07'M)O+c\x05#\x05%\x05'\x1d/1\x05)\x1d\x013\x17\x03\xb6\x06\x17\x1d\x017\x17\x03\xfe\x06\t\x03\x01\x03\x03]#\t\x03\x03A\r\x03CE\x1d+\x1d-\x1d/\x1d1\x13\r\x01\r\x01\r\x03QS\x1d3\x1d5\x0b\x03\x1d7\x1d9\x05\x01\x1f\x111\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x11\x03\x05\x15\t\t\r\x11!\x03e\x03i\x01\x01\x01\x01\x01\x13\x07_ag\x11\x03\t\x13\x07_ak\x11\x03\r\x01\t\x01\x02\x02\x1b)\x07\t\r!\x05\x11\x01\x03\x07)\x03a\x05\x1d)\x07\t\r\x11\x05)\x03\r\x13\x13\x04c\x05\x01Q\x07\t\x01\x07\x04Q\x03\x01\x05\x03P\x07\x03\x07\x04=\x03\x07\x11\x05B\x15\x05\x03\x0b\x07\x06\x1d\x03\x0f\x03\x01\tG-%\x07\x03\x07\x03\x03\x0b\x045\x03\x05\x06\x03\x01\x05\x01\x00\xee\x06;A\x03\x05\x1f\x0f\x0b\x0f!3%3)\x11\x0b\x19%)9w\x1d\x15\x1f\x17\x11\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00func_v1\x00iota_v1\x00reshape_v1\x00custom_call_v1\x00return_v1\x00jit(<lambda>)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda\x00iota\x00reshape\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00lu_pivots_to_permutation\x00jax.result_info\x00result\x00main\x00public\x00num_batch_dims\x002\x00\x00oneapi_lu_pivots_to_permutation\x00\x08+\t\x05'\x01\x0b9=?GI\x03K\x11UWY9[;9;",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
