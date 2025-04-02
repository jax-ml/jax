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

import datetime
from numpy import array, int32

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_04_01 = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cu_lu_pivots_to_permutation'],
    serialized_date=datetime.date(2025, 4, 1),
    inputs=(),
    expected_outputs=(array([[[0, 1, 2, 3, 4, 5, 6, 7],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7]],

       [[0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7]]], dtype=int32),),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x8xi32> {jax.result_info = "result"}) {
    %0 = stablehlo.iota dim = 0 : tensor<24xi32> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<24xi32>) -> tensor<2x3x4xi32> loc(#loc5)
    %2 = stablehlo.custom_call @cu_lu_pivots_to_permutation(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "2"}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x3x4xi32>) -> tensor<2x3x8xi32> loc(#loc6)
    return %2 : tensor<2x3x8xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":408:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":408:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":409:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.9.3\x00\x01\x1d\x05\x01\x05\r\x01\x03\x0b\x03\x0b\x0f\x13\x17\x1b\x1f\x03yQ\x15\x01+\x07\x0b\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x17\x1b\x0b\x0b\x0f\x0b\x17\x03'\x0b\x0f\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0bo\x01\x05\x0b\x0f\x03\x11\x07\x1b\x13\x13\x07\x1b\x13\x07\x02\x9e\x02\x1f\x05\x11\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x13\x11\x01\x00\x05\x15\x05\x17\x05\x19\x1d\x15\x17\x05\x1b\x17\x03b\x065\x1d\x1b\x1d\x05\x1d\x17\x03b\x06\x1d\x03\x05!?#A\x05\x1f\x05!\x1d')\x05#\x17\x03f\x06\x17\x03\x01\x03\x03O#\t\x03\x033\r\x0357\x1d%\x1d'\x1d)\x1d+\x13\r\x01\r\x01\r\x03CE\x1d-\x1d/\x0b\x03\x1d1\x1d3\x05\x01\x1f\x111\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02\x1b)\x07\t\r!\x05\x11\x01\x03\x07)\x03a\x05\x1d)\x07\t\r\x11\x05)\x03\r\x13\x13\x04c\x05\x01Q\x01\x07\x01\x07\x04Q\x03\x01\x05\x03P\x01\x03\x07\x04=\x03\x07\x11\x05B\x13\x05\x03\x0b\x07\x06\x19\x03\x0f\x03\x01\tG%\x1f\x07\x03\x07\x03\x03\x0b\x04\x01\x03\x05\x06\x03\x01\x05\x01\x00J\x0759\x03\x05\x1f\x0f\x0b\x0f!c3)A;\x1b%)9i\x15\x1f\x17\x11\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00iota_v1\x00reshape_v1\x00custom_call_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/iota\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation\x00jax.result_info\x00result\x00main\x00public\x00num_batch_dims\x002\x00\x00cu_lu_pivots_to_permutation\x00\x08+\t\x05#\x01\x0b+/19;\x03=\x11GIK+M-+-",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
