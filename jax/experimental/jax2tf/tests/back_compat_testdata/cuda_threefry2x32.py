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
from numpy import array, float32, uint32

# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_03_15 = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cu_threefry2x32'],
    serialized_date=datetime.date(2023, 3, 15),
    inputs=(array([42, 43], dtype=uint32),),
    expected_outputs=(array([[0.42591238, 0.0769949 , 0.44370103, 0.72904015],
    [0.17879379, 0.81439507, 0.00191903, 0.68608475]], dtype=float32),),
    mlir_module_text=r"""
module @jit_func {
  func.func public @main(%arg0: tensor<2xui32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}) -> (tensor<2x4xf32> {jax.result_info = ""}) {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %4 = stablehlo.iota dim = 0 : tensor<8xui32>
    %5 = "stablehlo.slice"(%arg0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = "stablehlo.slice"(%arg0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = "stablehlo.slice"(%4) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<8xui32>) -> tensor<4xui32>
    %10 = "stablehlo.slice"(%4) {limit_indices = dense<8> : tensor<1xi64>, start_indices = dense<4> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<8xui32>) -> tensor<4xui32>
    %11 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %12 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %13 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<4xui32>) -> tensor<4xui32>
    %14 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<4xui32>) -> tensor<4xui32>
    %15 = stablehlo.custom_call @cu_threefry2x32(%11, %12, %13, %14) {api_version = 2 : i32, backend_config = "\04\00\00\00\00\00\00\00", operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<4xui32>, tensor<4xui32>, tensor<4xui32>, tensor<4xui32>) -> tuple<tensor<4xui32>, tensor<4xui32>>
    %16 = stablehlo.get_tuple_element %15[0] : (tuple<tensor<4xui32>, tensor<4xui32>>) -> tensor<4xui32>
    %17 = stablehlo.get_tuple_element %15[1] : (tuple<tensor<4xui32>, tensor<4xui32>>) -> tensor<4xui32>
    %18 = stablehlo.concatenate %16, %17, dim = 0 : (tensor<4xui32>, tensor<4xui32>) -> tensor<8xui32>
    %19 = stablehlo.reshape %18 : (tensor<8xui32>) -> tensor<2x4xui32>
    %20 = stablehlo.constant dense<9> : tensor<ui32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<ui32>) -> tensor<2x4xui32>
    %22 = stablehlo.shift_right_logical %19, %21 : tensor<2x4xui32>
    %23 = stablehlo.constant dense<1065353216> : tensor<ui32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<ui32>) -> tensor<2x4xui32>
    %25 = stablehlo.or %22, %24 : tensor<2x4xui32>
    %26 = stablehlo.bitcast_convert %25 : (tensor<2x4xui32>) -> tensor<2x4xf32>
    %27 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %27, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %29 = stablehlo.subtract %26, %28 : tensor<2x4xf32>
    %30 = stablehlo.subtract %3, %1 : tensor<1x1xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32>
    %32 = stablehlo.multiply %29, %31 : tensor<2x4xf32>
    %33 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32>
    %34 = stablehlo.add %32, %33 : tensor<2x4xf32>
    %35 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<2x4xf32>
    %36 = stablehlo.maximum %35, %34 : tensor<2x4xf32>
    return %36 : tensor<2x4xf32>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x013\x05\x01\x05\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%'\x032\x02\xe1)\x01\x9b\x17\x07\x13\x0f\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0f\x13\x0f\x13\x0b\x0f\x0f\x0f\x0f\x0f\x13\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0b\x13\x0b\x0f\x0b#\x0f\x0b\x0b#\x0f\x0b#\x0f\x0b#\x0f\x0b\x0bK\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x03G///\x0f/\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x1f\x0f\x1f//\x0b\x0b\x0b\x0b\x1b\x13\x0f\x0f\x1f\x1fO\x03)\x17\x13\x07\x0f\x0f\x13\x17\x07\x07\x17\x13\x13\x13\x07\x17\x13\x13\x13\x07\x13\x02\xb6\x07\x17?\xb2\x03\x01\x1f\x03\x03\x11\xc3\x1dc\x01\x05)\x05+\x05-\x05/\x051\x1d\x93\x01\x03\x03\x11\xdf\x053\x1d=\x01\x03\x03\t\xc5\x1dO\x01\x03\x03\x11\x9f\x055\x1d\x89\x01\x1d\x8d\x01\x1d\x95\x01\x1d\x97\x01\x1d\x99\x01\x03\x03\x17/\x057\x03\x0b3\xa75\xb37\xb5\x17\xbd9\xbf\x059\x05;\x05=\x05?\x03\x03\t\xc1\x05A\x05C\x03\x03C\xa1\x05E\x1dG\x01\x05G\x03\x07\x0b\x9b\r\x9f\x0f\x9b\x1dM\x01\x05I\x05K\x03\x07\x0b\xc7\r\x9b\x0f\x9b\x1dU\x01\x05M\x03\x07\x0b\xa3\r\x9f\x0f\x9b\x1d[\x01\x05O\x03\x07\x0b\xc9\r\xa3\x0f\x9b\x1da\x01\x05Q\x05S\x03\x11g\xcbi\xcdk\xcfm\xa5o\xd1q\xd3s\xa5u\xd5\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x05c\x03\x03!\xd7\x03\x03!\xd9\x03\x03}\xa1\x05e\x1d\x81\x01\x05g\x1d\x85\x01\x05i\x03\x03\t\xdb\x05k\x03\x03\t\xdd\x05m\x1d\x91\x01\x05o\x05q\x05s\x05u\x05w\x1f\x0b\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x0b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x0f\x01\x1f\x0b\x11\x04\x00\x00\x00\x00\x00\x00\x00\x03\x01\x03\x03\xa9\r\x05\xab\xad\xaf\xb1\x1dy\x1d{\x1d}\x1d\x7f#\x1d\x03\x03\xb7\r\x03\xb9\xbb\x1d\x81\x1d\x83\x1d\x85\x1d\x87\x1f\t\t\x00\x00\x00\x00\x1f\x1f\x01\x1f\t\t\x00\x00\x80?\x1f\x0b\x11\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x0b\x11\x08\x00\x00\x00\x00\x00\x00\x00\x0b\x05\x1d\x89\x1d\x8b\x05\x01\x03\t\x9d\x9d\x9d\x9d\x03\x05\x9d\x9d\x13\x1b\x01\x13\x1b\x05\x1f\x07\t\t\x00\x00\x00\x1f\x07\t\x00\x00\x80?\x1f'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00)\x05\t\x11\x11)\x03\x11\x05%)\x01\x05)\x01\x11)\x03\x05\x0f)\x05\t\x11\x05\x1d\t)\x05\x05\x05\x11)\x03\t\x05)\x03!\x05)\x03\x05\x05\x1b\x11\x03\x15\x03\x01)\x03\x01\x0f/\x05\x03\x03)\x03\x05%\x13)\x03\t\x0f\x04\xd6\x04\x05\x01\x11\x03-\x07\x03\x01\x05\x0f\x11\x031\x05\x03M\x9b\x03\x15\x03\x05\x03\x03;\x03\t\x03\x07\x19\x05\x03\x13\x03\x03\x05\x03\x03\x1b\x03\t\x03\x07\x19\x05\x03\x13\x03\x07\x11\x03EA\x03\x17\x07\x07KI\x03\x19\x03\x01\t\x06\x1d\x03\x07\x03\r\x07\x07SQ\x03\x19\x03\x01\t\x06\x1d\x03\x07\x03\x11\x07\x07YW\x03\x03\x03\x0b\x07\x07_]\x03\x03\x03\x0b\x03\x07\x07\x05\x03\x03\x03\x0f\x03\x07\x07\x05\x03\x03\x03\x13\x03\x07\x07\x1f\x03\x03\x03\x15\x03\x07\x07\x1f\x03\x03\x03\x17\x13\x07\x07e\x03!\t\x19\x1b\x1d\x1f\x0b\x07\x07w\x03\x03\x03!\x0b\x07\x07y\x03\x03\x03!\x15\x07\x7f{\x03\x17\x05#%\t\x06\x83\x03\r\x03'\x05\x03\x03\x87\x03\x07\x03\x07#\x05\x03\r\x03+\x17\x06#\x03\r\x05)-\x05\x03\x03\x8b\x03\x07\x03\x07%\x05\x03\r\x031\x19\x06%\x03\r\x05/3\x1b\x06\x8f\x03\x01\x035\x05\x03\x03\x1b\x03\t\x03\x07\x13\x05\x03\x01\x039\r\x06\x13\x03\x01\x057;\r\x06\x13\x03\x13\x05\t\x05\x03\x07'\x15\x03\x01\x03?\x1d\x06'\x03\x01\x05=A\x03\x07)\x15\x03\x01\x03\x05\x1f\x06)\x03\x01\x05CE\x03\x07+\x15\x03\x01\x03\x05!\x06+\x03\x01\x05IG#\x04\x03\x03K\x06\x03\x01\x05\x01\x00N\x19\x8d!\x13\x0f\x0b\x03!\x1b\x1d\x05\x1b1111y/Q}[\x15\x1f/!!)#\x1f\x19C\x9d\x9d\x9d[\x9d}\x1f\x83\x97\x1f\x15\x1d\x15\x13\r\x13+\x11\x1d\x1d\r\x15\x17\x0f\x19'\r/\x1f\x1f\x11\x11\x19+\x17\x13\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00slice_v1\x00reshape_v1\x00get_tuple_element_v1\x00subtract_v1\x00func_v1\x00iota_v1\x00custom_call_v1\x00concatenate_v1\x00shift_right_logical_v1\x00or_v1\x00bitcast_convert_v1\x00multiply_v1\x00add_v1\x00maximum_v1\x00return_v1\x00value\x00limit_indices\x00start_indices\x00strides\x00broadcast_dimensions\x00sym_name\x00index\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(func)/jit(main)/broadcast_in_dim[shape=(1, 1) broadcast_dimensions=()]\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00iota_dimension\x00jit(func)/jit(main)/iota[dtype=uint32 shape=(8,) dimension=0]\x00jit(func)/jit(main)/slice[start_indices=(0,) limit_indices=(1,) strides=(1,)]\x00jit(func)/jit(main)/squeeze[dimensions=(0,)]\x00jit(func)/jit(main)/slice[start_indices=(1,) limit_indices=(2,) strides=(1,)]\x00jit(func)/jit(main)/slice[start_indices=(0,) limit_indices=(4,) strides=None]\x00jit(func)/jit(main)/slice[start_indices=(4,) limit_indices=(8,) strides=None]\x00jit(func)/jit(main)/threefry2x32\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00dimension\x00jit(func)/jit(main)/concatenate[dimension=0]\x00jit(func)/jit(main)/reshape[new_sizes=(2, 4) dimensions=None]\x00jit(func)/jit(main)/shift_right_logical\x00jit(func)/jit(main)/or\x00jit(func)/jit(main)/bitcast_convert_type[new_dtype=float32]\x00jit(func)/jit(main)/sub\x00jit(func)/jit(main)/mul\x00jit(func)/jit(main)/add\x00jit(func)/jit(main)/max\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00\x00main\x00public\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00cu_threefry2x32\x00",
    xla_call_module_version=4,
)  # End paste
