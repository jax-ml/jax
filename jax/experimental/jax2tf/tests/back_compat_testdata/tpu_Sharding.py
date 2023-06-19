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
from numpy import array, float32

# Pasted from the test output (see module docstring)
data_2023_03_16 = dict(
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
)  # End paste
