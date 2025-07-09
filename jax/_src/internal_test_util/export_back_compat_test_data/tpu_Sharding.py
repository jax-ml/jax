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

import datetime
import numpy as np

array = np.array
float32 = np.float32


data_2023_03_16 = {}


# Pasted from the test output (see module docstring)
data_2023_03_16['gspmd'] = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['SPMDFullToShardShape', 'SPMDShardToFullShape', 'Sharding'],
    serialized_date=datetime.date(2023, 3, 15),
    inputs=(array([[0., 1., 2., 3.],
                   [4., 5., 6., 7.]], dtype=float32),),
    expected_outputs=(array([[4., 5., 6., 7.],
                             [0., 1., 2., 3.]], dtype=float32),),
    mlir_module_text=r"""
module @jit_wrapped {
  func.func public @main(%arg0: tensor<2x4xf32> {jax.arg_info = "args[0]", mhlo.sharding = "{replicated}"}) -> (tensor<2x4xf32> {jax.result_info = ""}) {
    %0 = call @wrapped(%arg0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  func.func private @wrapped(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = stablehlo.custom_call @Sharding(%0) {mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = stablehlo.custom_call @SPMDFullToShardShape(%1) {mhlo.sharding = "{manual}"} : (tensor<2x4xf32>) -> tensor<1x4xf32>
    %3 = "stablehlo.collective_permute"(%2) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %4 = stablehlo.custom_call @Sharding(%3) {mhlo.sharding = "{manual}"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %5 = stablehlo.custom_call @SPMDShardToFullShape(%4) {mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<1x4xf32>) -> tensor<2x4xf32>
    %6 = stablehlo.custom_call @Sharding(%5) {mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %6 : tensor<2x4xf32>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01\x1b\x05\x01\x05\x01\x03\x05\x03\x0b\x07\t\x0b\r\x0f\x03\x9d\x81\r\x01K\x07\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0bS\x0b\x0b\x0b\x0b\x17\x0b\x13\x0b33\x0b\x0bS\x1b\x0b\x0b\x0f\x0b\x17SS\x13\x0b\x037\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x8f\x0b\x03\r\x17\x17\x07\x07\x17\x17\x02\xb6\x04\x1f\x1d1%\x05\x11\x05\x13\x05\x15\x05\x17\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x1d3%\x05#\x03\x13\x05O\x07M\tY\x0bK\rQ\x0f[\x11K\x13K\x15K\x05%\x05'\x05)\x05+\x17'\x02\x02\x01\x05-\x03\x03\x19+\x05/\x03\x0b\x1d_\x1fS!k\x19q#s\x03\x0b\x1dU\x1fS!U\x19W#w\x051\x053\x03\x13\x05O\x07M\ty\x0bK\rQ\x0f]\x11K\x13K\x15K\x03\x059{;}\x055\x057\x1d?A\x059\x17'\x12\x05\x01\x03\x13\x05O\x07M\tY\x0bK\rQ\x0f]\x11K\x13K\x15K\x03\x13\x05O\x07M\t\x7f\x0bK\rQ\x0f[\x11K\x13K\x15K\x03\x03IW\x05;\x03\x01\x1d=\x0b\x03\x05\x01#\t\x03\x03u\x1d?\x1dA\x1dC\x1dE\x03\x03a\r\x05cegi\x1dG\x1dI\x1d\x1b\x1dK\x03\x03m\r\x03oM\x1dM\x1dO\x1dQ\r\x01\x1dS\x1dU\x13\x07\x05\x1f\x0bA\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dW)\x05\t\x11\x05)\x05\x05\x11\x05\t\x1d\x11\x03\x01\x03\x01)\x05\t\t\x07\x04\xd3\x05\x01\x11\x01)\x07\x03\x01\t\x05\x11\x01-\x05\x03\x05\x0b\x03\x01\x01\x0b\x07\x03G\x03\x01\x03\x01\x07\x04\x01\x03\x03\x05\x11\x03/\x05\x03\x11#\x03\x01\x01\x03\x07\x03\x1b\x03\x01\x03\x01\x03\x07\x17\x1b\x03\x01\x03\x03\x03\x07\x175\x03\x03\x03\x05\t\x07=7\x03\x03\x03\x07\x03\x07\x17C\x03\x03\x03\t\x03\x07\x17E\x03\x01\x03\x0b\x03\x07\x03\x1b\x03\x01\x03\r\x07\x04\x03\x03\x0f\x06\x03\x01\x05\x01\x00\x82\x13Y++\x11\x0f\x0b!\x1b\x11\x1b\x13'\x13\x11\x03\x0f\xa3)\x17\x9e\x02\x1e\x06\x19\x83\x1f\x15\x1d\x15\x13\x1f/!\x1d!)#\x1f\x19\x11-\x15\x11\x1f\x0f\x0b\x11builtin\x00vhlo\x00module\x00custom_call_v1\x00func_v1\x00return_v1\x00collective_permute_v1\x00call_v1\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.sharding\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00jit_wrapped\x00jit(wrapped)/jit(main)/pjit[in_shardings=(GSPMDSharding({devices=[2,1]0,1}),) out_shardings=(GSPMDSharding({devices=[2,1]0,1}),) resource_env=ResourceEnv(Mesh(device_ids=array([0, 1]), axis_names=('a',)), ()) donated_invars=(False,) name=wrapped in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(wrapped)/jit(main)/pjit(wrapped)/shard_map[mesh=Mesh(device_ids=array([0, 1]), axis_names=('a',)) in_names=({0: ('a',)},) out_names=({0: ('a',)},) check_rep=True]\x00channel_id\x00source_target_pairs\x00jit(wrapped)/jit(main)/pjit(wrapped)/ppermute[axis_name=a perm=((0, 1), (1, 0))]\x00callee\x00\x00wrapped\x00Sharding\x00{devices=[2,1]0,1}\x00{manual}\x00jax.arg_info\x00args[0]\x00{replicated}\x00jax.result_info\x00main\x00public\x00private\x00SPMDFullToShardShape\x00SPMDShardToFullShape\x00",
    xla_call_module_version=4,
    nr_devices=2,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2023_03_16['shardy'] = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['xla.sdy.FuncResultSharding', 'xla.sdy.GlobalToLocalShape', 'xla.sdy.LocalToGlobalShape'],
    serialized_date=datetime.date(2025, 5, 28),
    inputs=(array([[0., 1., 2., 3.],
       [4., 5., 6., 7.]], dtype=float32),),
    expected_outputs=(array([[4., 5., 6., 7.],
       [0., 1., 2., 3.]], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":783:6)
#loc4 = loc("jit(func)/jit(main)/shard_map"(#loc2))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22a\22=2]>}"}, mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22a\22}, {}]>"}, mhlo.sharding = "{devices=[2,1]<=[2]}"} loc("x")) -> (tensor<2x4xf32> {jax.result_info = "result", mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
    %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}, {}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>"}} : (tensor<2x4xf32>) -> tensor<1x4xf32> loc(#loc4)
    %1 = call @xla.sdy.manual_computation_body(%0) : (tensor<1x4xf32>) -> tensor<1x4xf32> loc(#loc4)
    %2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}, {}]>]>"}} : (tensor<1x4xf32>) -> tensor<2x4xf32> loc(#loc4)
    %3 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%2) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}, {}]>]>"}} : (tensor<2x4xf32>) -> tensor<2x4xf32> loc(#loc4)
    return %3 : tensor<2x4xf32> loc(#loc)
  } loc(#loc)
  func.func @xla.sdy.manual_computation_body(%arg0: tensor<1x4xf32> loc("jit(func)/jit(main)/shard_map"(#loc2))) -> tensor<1x4xf32> {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32> loc(#loc5)
    return %0 : tensor<1x4xf32> loc(#loc4)
  } loc(#loc4)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":779:13)
#loc5 = loc("jit(func)/jit(main)/ppermute"(#loc3))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.10.3\x00\x01\x1d\x05\x01\x05\r\x01\x03\x0b\x03\x0b\x0f\x13\x17\x1b\x1f\x03\x9fy\x13\x013\x0f\x0b\x07\x0b+\x0b\x0f\x13\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x17\x0f\x0b\x17\x0f\x0b\x13\x13\x13\x03G\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1b\x0b\x13\x0b\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x0f\x8f\x1b\x0b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x01\x05\x0f\x0b\x03\x0f\x17\x17\x07\x07\x17\x17\x17\x02\xae\x03\x1d\x1f!\x05\x11\x1f\x05\x13\x03\t\x0b\r\x03\x0f\x15\x17\x19\x1b\x05\x15\x11\x03\x00\x03\x03\x11\x13\x05\x17\x05\x19\x05\x1b\x11\x01\t\x05\x1d\x11\x01\x05\x05\x1f\x05!\x17\x07>\x0c\r\x1d%\'\x05#\x17\x07.\x0c\x1b\x1d+\x05\x05%\x03\x03\x03g\x03\x03\x03m\x03\x03\x03u\x03\x01\x1d\'\x1d)\x0b\x03\x1d+\x1d-\x1d/\x1d1\x1d3\x1d5\x05\x03\x03\x03K\r\x05MO=?\x1d\x11\r\x03;Q\x1d7#\r\x03\x03W\r\x05Y[=?\x1d9\x1d;\x1d=\x1d?#\x0f\x13\x0b\x05\x1f\x11A\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05i7CE\x1dA\x1dC\r\x05CEo7\x1dE\x1dG\x05\x01\r\x03;7\x1dI\x01\x02\x02\x01\t)\x05\x05\x11\t)\x05\t\x11\t\t\x1d\x11\x03\x07\x03\x07\x11\x03\x05\x03\x05)\x05\t\t\x0b\x04\xbb\x05\x01Q\x05\t\x01\x07\x04\xa9\x03\x01\t\x05P\x05\x03\x07\x04_\x03\x0b\x17\x03\x0f)\x00\x03G\x01-\x05\x03\x05\x03\x01\x0bF\x01\x07\x03\x05\x03\x03\x03G\x01/\t\x03\x07\x03\x05\x03G\x011\x0b\x03\x07\x03\x07\x07\x04\x05\x03\t\x05P\x01\r\x07\x04)\x03\x05\x0b\x03\x0b\x01\x00\tF#\x0f\x03\x05\x03\x01\x07\x04\x01\x03\x03\x06\x03\x01\x05\x01\x00\xaa\x0bK77-7+\x0f\x0b\x0f!E/)A+\x1d#a\x03\x05;=\x13%)9\x1f9i3\x11-\x15\x11\x1f\x0f\x0b\x11builtin\x00vhlo\x00module\x00custom_call_v1\x00func_v1\x00return_v1\x00collective_permute_v1\x00call_v1\x00mhlo.frontend_attributes\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00xla.sdy.meshes\x00{mesh = #sdy.mesh<["a"=2]>}\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/shard_map\x00jit(func)/jit(main)/ppermute\x00x\x00\x00#sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>\x00xla.sdy.sharding\x00mhlo.sharding\x00{devices=[2,1]<=[2]}\x00xla.sdy.manual_computation_body\x00xla.sdy.manual_axes\x00#sdy<manual_axes{"a"}>\x00#sdy.sharding<@mesh, [{"a"}, {}]>\x00jax.result_info\x00result\x00main\x00public\x00xla.sdy.in_shardings\x00xla.sdy.GlobalToLocalShape\x00xla.sdy.out_shardings\x00xla.sdy.LocalToGlobalShape\x00xla.sdy.FuncResultSharding\x00\x08a\x11\x05;\x01\x0bISU]_\x1195k3G333\x03A\x1195q3s333\x1195w3G333\x0b3a3A5\x05ce',
    xla_call_module_version=9,
    nr_devices=2,
)  # End paste


data_2025_06_30 = {}


# Pasted from the test output (see module docstring)
data_2025_06_30['gspmd'] = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['SPMDFullToShardShape', 'SPMDShardToFullShape', 'Sharding'],
    serialized_date=datetime.date(2023, 3, 15),
    inputs=(array([[0., 1., 2., 3.],
                   [4., 5., 6., 7.]], dtype=float32),),
    expected_outputs=(array([[4., 5., 6., 7.],
                             [0., 1., 2., 3.]], dtype=float32),),
    mlir_module_text=r"""
module @jit_wrapped {
  func.func public @main(%arg0: tensor<2x4xf32> {jax.arg_info = "args[0]", mhlo.sharding = "{replicated}"}) -> (tensor<2x4xf32> {jax.result_info = ""}) {
    %0 = call @wrapped(%arg0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  func.func private @wrapped(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = stablehlo.custom_call @Sharding(%0) {mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = stablehlo.custom_call @SPMDFullToShardShape(%1) {mhlo.sharding = "{manual}"} : (tensor<2x4xf32>) -> tensor<1x4xf32>
    %3 = "stablehlo.collective_permute"(%2) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %4 = stablehlo.custom_call @Sharding(%3) {mhlo.sharding = "{manual}"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %5 = stablehlo.custom_call @SPMDShardToFullShape(%4) {mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<1x4xf32>) -> tensor<2x4xf32>
    %6 = stablehlo.custom_call @Sharding(%5) {mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %6 : tensor<2x4xf32>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01\x1b\x05\x01\x05\x01\x03\x05\x03\x0b\x07\t\x0b\r\x0f\x03\x9d\x81\r\x01K\x07\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0bS\x0b\x0b\x0b\x0b\x17\x0b\x13\x0b33\x0b\x0bS\x1b\x0b\x0b\x0f\x0b\x17SS\x13\x0b\x037\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x8f\x0b\x03\r\x17\x17\x07\x07\x17\x17\x02\xb6\x04\x1f\x1d1%\x05\x11\x05\x13\x05\x15\x05\x17\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x1d3%\x05#\x03\x13\x05O\x07M\tY\x0bK\rQ\x0f[\x11K\x13K\x15K\x05%\x05'\x05)\x05+\x17'\x02\x02\x01\x05-\x03\x03\x19+\x05/\x03\x0b\x1d_\x1fS!k\x19q#s\x03\x0b\x1dU\x1fS!U\x19W#w\x051\x053\x03\x13\x05O\x07M\ty\x0bK\rQ\x0f]\x11K\x13K\x15K\x03\x059{;}\x055\x057\x1d?A\x059\x17'\x12\x05\x01\x03\x13\x05O\x07M\tY\x0bK\rQ\x0f]\x11K\x13K\x15K\x03\x13\x05O\x07M\t\x7f\x0bK\rQ\x0f[\x11K\x13K\x15K\x03\x03IW\x05;\x03\x01\x1d=\x0b\x03\x05\x01#\t\x03\x03u\x1d?\x1dA\x1dC\x1dE\x03\x03a\r\x05cegi\x1dG\x1dI\x1d\x1b\x1dK\x03\x03m\r\x03oM\x1dM\x1dO\x1dQ\r\x01\x1dS\x1dU\x13\x07\x05\x1f\x0bA\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dW)\x05\t\x11\x05)\x05\x05\x11\x05\t\x1d\x11\x03\x01\x03\x01)\x05\t\t\x07\x04\xd3\x05\x01\x11\x01)\x07\x03\x01\t\x05\x11\x01-\x05\x03\x05\x0b\x03\x01\x01\x0b\x07\x03G\x03\x01\x03\x01\x07\x04\x01\x03\x03\x05\x11\x03/\x05\x03\x11#\x03\x01\x01\x03\x07\x03\x1b\x03\x01\x03\x01\x03\x07\x17\x1b\x03\x01\x03\x03\x03\x07\x175\x03\x03\x03\x05\t\x07=7\x03\x03\x03\x07\x03\x07\x17C\x03\x03\x03\t\x03\x07\x17E\x03\x01\x03\x0b\x03\x07\x03\x1b\x03\x01\x03\r\x07\x04\x03\x03\x0f\x06\x03\x01\x05\x01\x00\x82\x13Y++\x11\x0f\x0b!\x1b\x11\x1b\x13'\x13\x11\x03\x0f\xa3)\x17\x9e\x02\x1e\x06\x19\x83\x1f\x15\x1d\x15\x13\x1f/!\x1d!)#\x1f\x19\x11-\x15\x11\x1f\x0f\x0b\x11builtin\x00vhlo\x00module\x00custom_call_v1\x00func_v1\x00return_v1\x00collective_permute_v1\x00call_v1\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.sharding\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00jit_wrapped\x00jit(wrapped)/jit(main)/pjit[in_shardings=(GSPMDSharding({devices=[2,1]0,1}),) out_shardings=(GSPMDSharding({devices=[2,1]0,1}),) resource_env=ResourceEnv(Mesh(device_ids=array([0, 1]), axis_names=('a',)), ()) donated_invars=(False,) name=wrapped in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(wrapped)/jit(main)/pjit(wrapped)/shard_map[mesh=Mesh(device_ids=array([0, 1]), axis_names=('a',)) in_names=({0: ('a',)},) out_names=({0: ('a',)},) check_rep=True]\x00channel_id\x00source_target_pairs\x00jit(wrapped)/jit(main)/pjit(wrapped)/ppermute[axis_name=a perm=((0, 1), (1, 0))]\x00callee\x00\x00wrapped\x00Sharding\x00{devices=[2,1]0,1}\x00{manual}\x00jax.arg_info\x00args[0]\x00{replicated}\x00jax.result_info\x00main\x00public\x00private\x00SPMDFullToShardShape\x00SPMDShardToFullShape\x00",
    xla_call_module_version=4,
    nr_devices=2,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_06_30['shardy'] = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=[],
    serialized_date=datetime.date(2025, 6, 30),
    inputs=(array([[0., 1., 2., 3.],
       [4., 5., 6., 7.]], dtype=float32),),
    expected_outputs=(array([[4., 5., 6., 7.],
       [0., 1., 2., 3.]], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":792:6)
#loc4 = loc("jit(func)/shard_map"(#loc2))
#loc6 = loc("shard_map:"(#loc4))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["a"=2]> loc(#loc)
  func.func public @main(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>} loc("x")) -> (tensor<2x4xf32> {jax.result_info = "result", sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<1x4xf32> loc("shard_map:"(#loc4))) {
      %1 = "stablehlo.collective_permute"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32> loc(#loc7)
      sdy.return %1 : tensor<1x4xf32> loc(#loc6)
    } : (tensor<2x4xf32>) -> tensor<2x4xf32> loc(#loc6)
    return %0 : tensor<2x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":783:13)
#loc5 = loc("jit(func)/ppermute"(#loc3))
#loc7 = loc("ppermute:"(#loc5))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.10.9\x00\x01'\x07\x01\x05\t\x11\x01\x05\x0f\x13\x03\x07\x17\x1b\x1f\x05\x07#'+\x03\x89[\x17\x013\x07\x0f\x0f\x0b\x0f\x0b#\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x17\x0b\x0f\x0b\x17\x03\x11\x1b\x0f\x13\x0f\x17\x0f\x13\x0f\x05\x19\x0b\x0f\x13\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x8f\x01\x0b\x0f\x17\x07\x17\x0b\x05\r\x17\x07\x17\x07\x17\x17\x022\x03\x1f\x1d#%\x1d+-\x05\x0b\x1d\x1f\x01\x05\x17\x03\x07\x0f\x11\x13\x15\x17\x19\x05\x19\x11\t\x00\x05\x1b\x11\x01\t\x05\x1d\x11\x01\x05\x05\x1f\t\x07\x05!\x05#\x05%\x1d')\x05'\x17\x0bb\x0c\r\x05)\x1d/1\x05+\x17\x0b>\x0c\x1b\r\x1d\x05;?\x01\x0f\x033\x05\x039\x01\x03#\t\x0b\x03=\x01\x01\t#\x01\x0b\x01\x01\x01\x01\x03!\x1d-\x03\x03G\r\x03C3#\x13\x03\x03M\r\x05OQC3\x1d/\x1d1\x1d3\x1d5\x13\x11\x05\x1f\x15A\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x02\x1b\x05\t\x11\x05\x0b\x1b\x05\x05\x11\x05\x01\t)\x05\t\x11\r\t)\x05\x05\x11\r\x1d\x11\x03\x0b\x03\x0b)\x05\t\t\x11\x04\xbd\x05\x03Q\x01\r\x01\x07\x04\xab\x03\x01\t\x05@\x01\x03\x0bP\x01\x05\x07\x04\x8f\x03\t\x13\x03\x17\t\x00\x01\x06\t\x03\x03\x03\x01\x07V\x03\x07\x03\x03\x03\x03\x07\x04E\x03\t\x13\x03\x0f\x03\x00\x01\x06\x05\x03\x0f\x03\x01\x0fF\x05\t\x03\x0f\x03\x03\x01\x06\x05\x03\x07\x03\x05\t\x04\x03\x03\x07\x01\x06\x01\x03\x0b\x03\x05\r\x04\x01\x03\x07\x06\x03\x01\x05\x01\x00\x16\x067\x0f\x0b\x0f!\x1b'\x15)\x17\x05\x05\x13%)9i-\x15\x11\x0f'\x0b\x0f7\x0b\t\x11builtin\x00sdy\x00vhlo\x00unrealized_conversion_cast\x00module\x00mesh\x00manual_computation\x00return\x00func_v1\x00return_v1\x00collective_permute_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00a\x00shard_map:\x00jit(func)/shard_map\x00ppermute:\x00jit(func)/ppermute\x00sdy.sharding\x00jax.result_info\x00result\x00main\x00public\x00\x08)\x0b\x057\x01\x057\x07\x0bEIKSU\x075A5\x05WY",
    xla_call_module_version=9,
    nr_devices=2,
)  # End paste
