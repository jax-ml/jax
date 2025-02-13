# Copyright 2025 The JAX Authors.
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
from numpy import array, float32


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_02_12 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['Sharding', 'xla.sdy.GlobalToLocalShape', 'xla.sdy.LocalToGlobalShape'],
    serialized_date=datetime.date(2025, 2, 12),
    inputs=(array([[0., 1., 2., 3.],
       [4., 5., 6., 7.]], dtype=float32),),
    expected_outputs=(array([[4., 5., 6., 7.],
       [0., 1., 2., 3.]], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":1052:13)
#loc6 = loc("jit(func)/jit(main)/shard_map"(#loc3))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\\\22a\\\22=2]>}"}, mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4xf32> loc("x")) -> (tensor<2x4xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}]>]>"}, mhlo.sharding = "{devices=[2,1]<=[2]}"} : (tensor<2x4xf32>) -> tensor<2x4xf32> loc(#loc5)
    %1 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%0) : (tensor<2x4xf32>) -> tensor<1x4xf32> loc(#loc6)
    %2 = call @xla.sdy.manual_computation_body(%1) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}]>]>"}} : (tensor<1x4xf32>) -> tensor<1x4xf32> loc(#loc6)
    %3 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%2) : (tensor<1x4xf32>) -> tensor<2x4xf32> loc(#loc6)
    return %3 : tensor<2x4xf32> loc(#loc)
  } loc(#loc)
  func.func @xla.sdy.manual_computation_body(%arg0: tensor<1x4xf32> loc("jit(func)/jit(main)/shard_map"(#loc3))) -> tensor<1x4xf32> {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32> loc(#loc7)
    return %0 : tensor<1x4xf32> loc(#loc6)
  } loc(#loc6)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":1051:10)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":1050:15)
#loc5 = loc("jit(func)/jit(main)/sharding_constraint"(#loc2))
#loc7 = loc("jit(func)/jit(main)/ppermute"(#loc4))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.8.8\x00\x01\x1d\x05\x01\x05\r\x01\x03\x0b\x03\x0b\x0f\x13\x17\x1b\x1f\x03\x97q\x13\x019\x0f\x07\x0b\x0b+\x0b\x0f\x13\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x17\x0f\x0b\x17\x0f\x0b\x1b\x0b\x0f\x0b\x17\x13\x039\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0f\x8f\x13\x0b\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0b\x01\x05\x0f\x0b\x03\x0f\x17\x17\x07\x07\x17\x17\x17\x02v\x03\x1d\x1f!\x1f\x05\x11\x05\x13\x03\t\x0b\r\x05\x0f\x15\x17\x19\x1b\x05\x15\x11\x03\x00\x03\x03\x11\x13\x05\x17\x05\x19\x05\x1b\x11\x01\t\x05\x1d\x11\x01\x05\x05\x1f\x05!\x17\x07r\x10\x1b\x1d%\'\x05#\x17\x07j\x10\x1f\x1d+\x03\x05%\x03\x05\x05[/_\x05\'\x1d35\x05)\x17\x07n\x10\x15\x03\x03\x05e\x03\x01\x1d+\x1d-\x0b\x03\x05\x01\x1d/\x03\x03G\r\x01#\r\x03\x03M\r\x03O;\x1d1\x1d3\x1d5#\x0f\x13\x0b\x05\x1f\x11A\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03]=\x1d7\x1d9\x1d;\x1d=\r\x07g=ikm=\x1d?\x1dA\x1dC\x1dE\x1dG\x01\x02\x02\x01\t)\x05\x05\x11\t)\x05\t\x11\t\t\x1d\x11\x03\x07\x03\x07\x11\x03\x05\x03\x05)\x05\t\t\x0b\x04\xb9\x05\x01Q\x03\t\x01\x07\x04\xa7\x03\x01\t\x05P\x03\x03\x07\x04]\x03\x0b\x17\x03\x0f)\x00\x03G1-\x05\x03\x07\x03\x01\x03F\x01\x07\x03\x05\x03\x03\x0bG\x017\t\x03\x05\x03\x05\x03F\x01\x0b\x03\x07\x03\x07\x07\x04\x03\x03\t\x05P\x01\r\x07\x04)\x03\x05\x0b\x03\x0b\x01\x00\tF#\x0f\x03\x05\x03\x01\x07\x04\x01\x03\x03\x06\x03\x01\x05\x01\x00r\x0bI7-3)+7\x13+#\x0f\x0b!Ae\x03Q\x1d\x05;=\x13%)=\x1f9i3\x11-\x15\x11\x1f\x0f\x0b\x11builtin\x00vhlo\x00module\x00custom_call_v1\x00func_v1\x00return_v1\x00collective_permute_v1\x00call_v1\x00mhlo.frontend_attributes\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00xla.sdy.meshes\x00{mesh = #sdy.mesh<[\\"a\\"=2]>}\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/shard_map\x00jit(func)/jit(main)/ppermute\x00x\x00mhlo.sharding\x00jit(func)/jit(main)/sharding_constraint\x00\x00#sdy.sharding_per_value<[<@mesh, [{\\"a\\"}, {}]>]>\x00xla.sdy.manual_computation_body\x00jax.result_info\x00main\x00public\x00xla.sdy.sharding\x00{devices=[2,1]<=[2]}\x00Sharding\x00xla.sdy.GlobalToLocalShape\x00xla.sdy.in_shardings\x00xla.sdy.manual_axes\x00#sdy<manual_axes{\\"a\\"}>\x00xla.sdy.out_shardings\x00xla.sdy.LocalToGlobalShape\x00\x08a\x11\x05;\x01\x0bEIKQS\x11?;a9A999\x11?;c9A999\x03C\x11?;o9A999\x0b9U9C;\x05WY',
    xla_call_module_version=9,
    nr_devices=2,
)  # End paste
