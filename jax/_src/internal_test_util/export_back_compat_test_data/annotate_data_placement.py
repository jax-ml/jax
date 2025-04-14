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
from numpy import array, float32, int32


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_04_07_tpu = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['annotate_device_placement'],
    serialized_date=datetime.date(2025, 4, 7),
    inputs=(array([0.], dtype=float32), array([0.], dtype=float32)),
    expected_outputs=(array([0.], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc2 = loc("y")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1xf32> {mhlo.memory_kind = "device", mhlo.sharding = "{maximal device=0}"} loc("x"), %arg1: tensor<1xf32> {mhlo.memory_kind = "pinned_host", mhlo.sharding = "{maximal device=0}"} loc("y")) -> (tensor<1xf32> {jax.result_info = "result", mhlo.memory_kind = "pinned_host", mhlo.sharding = "{maximal device=0}"}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1xf32> loc(#loc4)
    %1 = stablehlo.custom_call @annotate_device_placement(%0) {has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<1xf32>) -> tensor<1xf32> loc(#loc)
    return %1 : tensor<1xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":878:13)
#loc4 = loc("jit(func)/jit(main)/add"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.9.3\x00\x01\x1b\x05\x01\x05\x0b\x01\x03\x0b\x03\t\x0f\x13\x17\x1b\x03oQ\x0b\x01%\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0b\x13\x0b\x03-\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x1b\x0b\x0f#\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x01\x05\x0b\x0f\x03\x07\x13\x1b\x07\x02\n\x02\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x13\x01\x05\x17\x1d\x17\x01\x05\x19\x1d\x1b\x1d\x05\x1b\x17\x1f\xba\r\x1b\x05\x1d\x03\x03#E\x05\x1f\x03\x01\x1d!\x1d#\x1d%\x1d'\x03\x0515\r\x05'3)+\x1d)\r\x05'-)+#\x07\x03\x03;\r\x07=?'-)+\x1d+\x1d-\x1d/\x1d1\r\x03G-\x1d3\x0b\x03\x1d5\x1d7\x05\x03\x01\t\x01\x02\x02)\x03\x05\t\x11\x05\x05\x05\x03\x05\t\x04e\x05\x01Q\x01\x05\x01\x07\x04S\x03\x01\x05\x03P\x01\x03\x07\x04?\x03\t\x0f\x05\x0b\x11\x0b\x15\x00\x05\x06\x19\x03\x05\x05\x01\x03\x07G\x01!\x05\x03\x05\x03\x05\t\x04\x01\x03\x07\x06\x03\x01\x05\x01\x00\x9a\x0695\x03-\x0f\x0b\x0f!\x0f\x19'\x1d#3i1\x05\x05\x13%)9\x15\x1f\x0f\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00add_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00y\x00jit(func)/jit(main)/add\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.frontend_attributes\x00mhlo.memory_kind\x00mhlo.sharding\x00{maximal device=0}\x00pinned_host\x00device\x00jax.result_info\x00result\x00main\x00public\x00_xla_buffer_placement\x00\x00annotate_device_placement\x00\x08'\x07\x05\x1f\x01\x0b/79AC\x11IKM%O%%%",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_04_07_cuda = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['annotate_device_placement'],
    serialized_date=datetime.date(2025, 4, 7),
    inputs=(array([0.], dtype=float32), array([0.], dtype=float32)),
    expected_outputs=(array([0.], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc2 = loc("y")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1xf32> {mhlo.memory_kind = "device", mhlo.sharding = "{maximal device=0}"} loc("x"), %arg1: tensor<1xf32> {mhlo.memory_kind = "pinned_host", mhlo.sharding = "{maximal device=0}"} loc("y")) -> (tensor<1xf32> {jax.result_info = "result", mhlo.memory_kind = "pinned_host", mhlo.sharding = "{maximal device=0}"}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1xf32> loc(#loc4)
    %1 = stablehlo.custom_call @annotate_device_placement(%0) {has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<1xf32>) -> tensor<1xf32> loc(#loc)
    return %1 : tensor<1xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":878:13)
#loc4 = loc("jit(func)/jit(main)/add"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.9.3\x00\x01\x1b\x05\x01\x05\x0b\x01\x03\x0b\x03\t\x0f\x13\x17\x1b\x03oQ\x0b\x01%\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0b\x13\x0b\x03-\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x1b\x0b\x0f#\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x01\x05\x0b\x0f\x03\x07\x13\x1b\x07\x02\n\x02\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x13\x01\x05\x17\x1d\x17\x01\x05\x19\x1d\x1b\x1d\x05\x1b\x17\x1f\xba\r\x1b\x05\x1d\x03\x03#E\x05\x1f\x03\x01\x1d!\x1d#\x1d%\x1d'\x03\x0515\r\x05'3)+\x1d)\r\x05'-)+#\x07\x03\x03;\r\x07=?'-)+\x1d+\x1d-\x1d/\x1d1\r\x03G-\x1d3\x0b\x03\x1d5\x1d7\x05\x03\x01\t\x01\x02\x02)\x03\x05\t\x11\x05\x05\x05\x03\x05\t\x04e\x05\x01Q\x01\x05\x01\x07\x04S\x03\x01\x05\x03P\x01\x03\x07\x04?\x03\t\x0f\x05\x0b\x11\x0b\x15\x00\x05\x06\x19\x03\x05\x05\x01\x03\x07G\x01!\x05\x03\x05\x03\x05\t\x04\x01\x03\x07\x06\x03\x01\x05\x01\x00\x9a\x0695\x03-\x0f\x0b\x0f!\x0f\x19'\x1d#3i1\x05\x05\x13%)9\x15\x1f\x0f\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00add_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00y\x00jit(func)/jit(main)/add\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.frontend_attributes\x00mhlo.memory_kind\x00mhlo.sharding\x00{maximal device=0}\x00pinned_host\x00device\x00jax.result_info\x00result\x00main\x00public\x00_xla_buffer_placement\x00\x00annotate_device_placement\x00\x08'\x07\x05\x1f\x01\x0b/79AC\x11IKM%O%%%",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
