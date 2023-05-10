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

# flake8: noqa

import datetime
from numpy import array, float32, int32

# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_04_17 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['ApproxTopK'],
    serialized_date=datetime.date(2023, 4, 17),
    inputs=(),
    expected_outputs=(array([7., 6., 5.], dtype=float32), array([6, 5, 4], dtype=int32)),
    mlir_module_text="""
#loc = loc(unknown)
module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @top_k_gt_comparator(%arg0: tensor<f32> loc(unknown), %arg1: tensor<f32> loc(unknown), %arg2: tensor<i32> loc(unknown), %arg3: tensor<i32> loc(unknown)) -> tensor<i1> {
    %0 = stablehlo.compare  GT, %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<i1> loc(#loc2)
    return %0 : tensor<i1> loc(#loc2)
  } loc(#loc2)
  func.func public @main() -> (tensor<3xf32> {jax.result_info = "[0]"}, tensor<3xi32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.constant dense<[3.000000e+00, 1.000000e+00, 4.000000e+00, 2.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<7xf32> loc(#loc)
    %1 = stablehlo.iota dim = 0 : tensor<7xi32> loc(#loc2)
    %2 = stablehlo.constant dense<-1> : tensor<i32> loc(#loc2)
    %3 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64> loc(#loc2)
    %4 = stablehlo.convert %3 : (tensor<f64>) -> tensor<f32> loc(#loc2)
    %5:2 = stablehlo.custom_call @ApproxTopK(%0, %1, %4, %2) {called_computations = [@top_k_gt_comparator], mhlo.backend_config = {aggregate_to_topk = true, recall_target = 0.949999988 : f32, reduction_dim = 0 : i64, reduction_input_size_override = -1 : i64, top_k = 3 : i64}} : (tensor<7xf32>, tensor<7xi32>, tensor<f32>, tensor<i32>) -> (tensor<3xf32>, tensor<3xi32>) loc(#loc2)
    return %5#0, %5#1 : tensor<3xf32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":478:0)
#loc2 = loc("jit(<lambda>)/jit(main)/approx_top_k[k=3 reduction_dimension=-1 recall_target=0.95 is_max_k=True reduction_input_size_override=-1 aggregate_to_topk=True]"(#loc1))
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01\x1f\x05\x01\x05\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xbf\x8b!\x01I\x07\x0f\x0b\x0b\x0f\x0b\x0b\x0b\x0b#\x0b\x0b\x0b3\x0b\x17\x0b3\x13\x13\x0b\x13\x13S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x0b\x0b\x03C\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x7f\x1f/\x0b\x0b\x0f\x0b3\x0b\x0b\x0b\x1f\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x01\x03\x0f\x03\x1f\x0f\x0f\x07\x07\x07\x0f\x13\x13\x13\x0f#\x07\x17\x13\x07\x02\xee\x04\x1f\x1d\x1d\x1f\x05\x15\x05\x17\x11\x01\x05\x05\x19\x05\x1b\x05\x1d\x05\x1f\x03\x07\x15\t\x17\t\x05\x19\x05!\x05#\x05%\x03\x0b\x0bI\rS\x0fI\x05K\x11M\x05'\x17!z\x07\x01\x05)\x03\x0b\x0bI\rU\x0fW\x05a\x11c\x03\x03\x07e\x03\x03)Q\x05+\x03\x03\x07g\x03\x03\x07i\x03\x131k3M5m7o9q;s=I?IAI\x05-\x05/\x051\x053\x055\x057\x059\x05;\x05=\x03\x05E\x87G\x89\x05?\x05A\x03\x01\x1dC\x1dE\x1dG\x13\x0b\x01#\x17#\x1b\x03\x05Y]\r\x03O[\x1dI\r\x03O_\x1dK\x1dM\x1dO\x1f\x139\x00\x00@@\x00\x00\x80?\x00\x00\x80@\x00\x00\x00@\x00\x00\xa0@\x00\x00\xc0@\x00\x00\xe0@\x1f\x03\t\xff\xff\xff\xff\x1f\x15\x11\x00\x00\x00\x00\x00\x00\xf0\xff\x0b\x03\x1dQ\x03\x03K\x05\x01\r\x0buwy{}Q\x7f\x81\x83\x85\x1dS\x05\x03\x1dU\x11\x07\xd0\xcc\xcc\xdc\x0f\x1dW\x1dY\x13\x0b\x03\x1d[\x13\x0b\r\t\x01\x07\x07\x01\x02\x02)\x01\t)\x01\x07\t\x1b\x1d)\x01\x19)\x03\r\x07)\x03\r\t)\x03\x1d\x07)\x01\x1f\x11\t\x05\x05\x03\x03\x03\r\x01\x11\x01\x05\x0f\x11)\x03\x1d\t\x0b\x04\xc3\x05\x01\x11\x01\x13\x07\x03\x01\t\x05\x11\x03\x1b\x05\x03\x0b\x0b\t\x05\x01\x05\x01\x03\x01\x03\x01\x0f\x07\x03C\x03\r\x05\x01\x03\x07\x04\x03\x03\t\x05\x11\x01#\x05\x03\x0f\x1d\x03\x03\x01%\x03\x13\t\x03\x03'\x03\x1d\x03\x03\x03+\x03\x03\x03\x03\x03-\x03\x15\x0b\x06\x03\x03\x05\x03\x07\r\x07\x03/\x05\x0f\x11\t\x01\x03\t\x05\x07\x04\x01\x05\x0b\r\x06\x03\x01\x05\x01\x00\xfa\x0c]\r=\x1d\x1d%\x17\x0f\x0b\t\t!\x03)+\x1b\x1f/!)!)#\x1f\x19\x1f\x83j\x02\x1b%)\x1f\x15\x1d\x15\r\x13\x17\x1f\x17\x11\x15\x11\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00func_v1\x00return_v1\x00iota_v1\x00convert_v1\x00custom_call_v1\x00compare_v1\x00sym_name\x00value\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/approx_top_k[k=3 reduction_dimension=-1 recall_target=0.95 is_max_k=True reduction_input_size_override=-1 aggregate_to_topk=True]\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00iota_dimension\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00top_k_gt_comparator\x00\x00jax.result_info\x00[0]\x00[1]\x00main\x00public\x00ApproxTopK\x00aggregate_to_topk\x00recall_target\x00reduction_dim\x00reduction_input_size_override\x00top_k\x00",
    xla_call_module_version=4,
)  # End paste
