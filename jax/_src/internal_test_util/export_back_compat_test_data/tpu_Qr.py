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
import numpy as np

array = np.array
float32 = np.float32

# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_03_17 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['ProductOfElementaryHouseholderReflectors', 'Qr'],
    serialized_date=datetime.date(2023, 3, 17),
    inputs=(),
    expected_outputs=(array([[ 0.        ,  0.91287076,  0.4082487 ],
       [-0.44721356,  0.36514866, -0.8164965 ],
       [-0.8944271 , -0.18257444,  0.4082482 ]], dtype=float32), array([[-6.7082043, -8.049844 , -9.391484 ],
       [ 0.       ,  1.0954441,  2.1908882],
       [ 0.       ,  0.       ,  0.       ]], dtype=float32)),
    mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "[0]"}, tensor<3x3xf32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf32>
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32>
    %2:2 = call @geqrf(%1) : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>)
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.pad %2#0, %3, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32>
    %5 = call @householder_product(%4, %2#1) : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32>
    %6 = call @triu(%2#0) : (tensor<3x3xf32>) -> tensor<3x3xf32>
    return %5, %6 : tensor<3x3xf32>, tensor<3x3xf32>
  }
  func.func private @geqrf(%arg0: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>) {
    %0 = call @xla_fallback_geqrf(%arg0) : (tensor<3x3xf32>) -> tuple<tensor<3x3xf32>, tensor<3xf32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3x3xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3xf32>
    return %1, %2 : tensor<3x3xf32>, tensor<3xf32>
  }
  func.func private @xla_fallback_geqrf(%arg0: tensor<3x3xf32>) -> tuple<tensor<3x3xf32>, tensor<3xf32>> {
    %0 = stablehlo.custom_call @Qr(%arg0) {xla_shape = "(f32[3,3]{1,0}, f32[3]{0})"} : (tensor<3x3xf32>) -> tuple<tensor<3x3xf32>, tensor<3xf32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3x3xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3xf32>
    %3 = stablehlo.tuple %1, %2 {xla_shape = "(f32[3,3]{1,0}, f32[3]{0})"} : tuple<tensor<3x3xf32>, tensor<3xf32>>
    return %3 : tuple<tensor<3x3xf32>, tensor<3xf32>>
  }
  func.func private @householder_product(%arg0: tensor<3x3xf32>, %arg1: tensor<3xf32>) -> tensor<3x3xf32> {
    %0 = call @xla_fallback_householder_product(%arg0, %arg1) : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
  func.func private @xla_fallback_householder_product(%arg0: tensor<3x3xf32>, %arg1: tensor<3xf32>) -> tensor<3x3xf32> {
    %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1) : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
  func.func private @triu(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32>
    %1 = stablehlo.constant dense<-1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<3x3xi32>
    %3 = stablehlo.add %0, %2 : tensor<3x3xi32>
    %4 = stablehlo.iota dim = 1 : tensor<3x3xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %8 = stablehlo.select %5, %7, %arg0 : tensor<3x3xi1>, tensor<3x3xf32>
    return %8 : tensor<3x3xf32>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01-\x05\x01\x05\x01\x03\x05\x03\x1d\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!\x03j\x02\xff'\x01\xb3\x07\x17\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x0f\x0b\x13\x0b\x0b\x13\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x13\x17\x13\x0b33\x0b33\x0b33\x0b\x0f\x0b\x13\x0b\x0b\x13\x0f\x0b\x1b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0bK\x0f\x0b\x13S\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0f\x0b\x13\x0f\x0b\x0f\x0b\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x13\x13\x03M\x0b\x0b/\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x0b\x1f\x0b\x0b\x0f\x0f\x03'\x17\x13\x07\x13\x17\x07\x07\x0f\x0f\x17\x1b\x17\x1b\x17\x13\x17\x07\x13\x13\x02\x02\x08\x1f\x17\x17\xb2\x05\x01\x05#\x05%\x05'\x05)\x05+\x1dG\x03\x1dS\x03\x05-\x1dM\x03\x05/\x03\x03\x1b\xe9\x051\x053\x03\x03[\xed\x1d]\x03\x03\x03\x1d\xf5\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x05E\x03\x039\xfb\x05G\x03\x039\xfd\x17\x17\xae\x05\x01\x03\x03\x05A\x05I\x03\x0b\x07\xb3\t\xd1\x0b\xd3\x05\xdd\r\xdf\x03\x0b\x07\xb3\t\xe1\x0b\xb3\x05\xbb\r\xb5\x05K\x03\x0b\x07\xb3\t\xe3\x0b\xb3\x05\xbd\r\xb5\x03\x0b\x07\xb3\t\xbf\x0b\xb3\x05\xc1\r\xb5\x05M\x03\x0b\x07\xb3\t\xbf\x0b\xb3\x05\xc3\r\xb5\x03\x0b\x07\xc5\t\xe7\x0b\xc5\x05\xc7\r\xb5\x05O\x1dW\x03\x05Q\x03\x03\x1d\xeb\x05S\x05U\x03\x03\x1b\xef\x1dc\x03\x05W\x03\x05g\xf1i\xf3\x05Y\x05[\x1dm\x03\x05]\x1dq\x03\x05_\x1du\x03\x05a\x03\x11%\xc9'\xcb)\xf7+\xb3-\xcd/\xb31\xb33\xb3\x1d{\x01\x05c\x03\x03\x13\xc3\x03\x13%\xc9'\xcb)\xf9+\xb3-\xcd/\xb31\xb33\xb35\xcf\x1d\x83\x01\x05e\x1d\x87\x01\x05g\x1d\x8b\x01\x05i\x03\x035\xcf\x1d\x91\x01\x05k\x03\x03\x13\xbd\x1d\x97=\x05m\x1d\x9b=\x05o\x03\x03\x13\xbb\x1d\xa1\x03\x05q\x03\x07\xa5\xb7\xa7\xb7\xa9\xb7\x05s\x05u\x05w\x1d\xad\x03\x05y\x03\x03\x13\xc1\x03\x03\x13\xc7\x03\x01\x1d{\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d}\x1d\x7f\x1d\x81#\x19\x1d\x83\x1d\x85\x03\x03\xe5\x1d\x87\x0b\x03\x1d\x89\x05\x01\x1d\x8b#\x13\x03\x05\xd5\xd9\r\x03\xb9\xd7\x1d\x8d\r\x03\xb9\xdb\x1d\x8f\x1d\x91\x1d\x93#\x15#\x17\r\x01#\x1b\x13\r\x01\x1f\x11\t\xff\xff\xff\xff\x1f\x1d\x01\x13\r\x05\t\x07\x07\x05\x1f\x0f\t\x00\x00\x00\x00\x1d\x95\x1d\x97\x13\x0b\x01\x13\x0b\x05)\x05\r\r\x05)\x03\r\x05\t/\x05\x01\x03)\x05\r\r\x0b\x1b\x1d)\x01\x05)\x01\x0b\x11\x01\x05\x01\x01\x11\x03\x01\x05\x01\x03\x11\x03\x01\x03\x07\x11\x05\x01\x03\x03\x01\x11\x03\x01\x03\x01)\x03\x01\r)\x05\r\r!\x01)\x03%\x05)\x03\t\r\x04\xbe\x04\x05\x01\x11\x01?\x07\x03\x01\x19\x03\x11\x01C\x05\x03\x11!\x0b\x03\x95\x19\x03#\x1b\x06\x99\x03\x01\x03\x01\x07\x07\x0f\x9d\x05\x01\x03\x03\x03\r\x03\x9f#\x03\x0f\x1d\x07\xab\xa3\x03\x01\x05\x05\t\x07\x07\x15\xaf\x03\x01\x05\x0b\x07\x07\x07\x11\xb1\x03\x01\x03\x05\x05\x04\x01\x05\r\x0f\x03\x11\x0fE\x05\x03\t\x13\x03\x01\x01\x07\x07\x0f\x93\x03\x07\x03\x01\t\x07\x0f7\x03\x01\x03\x03\t\x07\x0f;\x03\x03\x03\x03\x05\x04\x0f\x05\x05\x07\x03\x11\x01I\x05\x03\x0b\x17\x03\x01\x01\x11\x07\x81\x7f\x03\x07\x03\x01\t\x07\x857\x03\x01\x03\x03\t\x07\x89;\x03\x03\x03\x03\x19\x07\x8f\x8d\x03\x07\x05\x05\x07\x05\x04\x01\x03\t\x03\x11\x15K\x05\x03\x07\x0b\x05\x01\x01\x03\x01\x07\x07\x15}\x03\x01\x05\x01\x03\x05\x04\x15\x03\x05\x03\x11\x01O\x05\x03\x07\x0b\x05\x01\x01\x03\x01\x11\x07yw\x03\x01\x05\x01\x03\x05\x04\x01\x03\x05\x03\x11\x11Q\x05\x03\x15+\x03\x01\x01\x0b\x03U\x19\x03\t\r\x03\x11Y\x03\x11\x0f\x07!\x1f\x03\t\x03\x05\x13\x06!\x03\t\x05\x03\x07\x0b\x03a_\x03\t\x15\x07ke\x03\x1f\x05\t\x0b\r\x03\x11#\x03\x0f\x0f\x07o\x1f\x03\x01\x03\x0f\x17\x06s\x03\x01\x07\r\x11\x01\x05\x04\x11\x03\x13\x06\x03\x01\x05\x01\x00V\x1c\x99\x07S\x0f\x0b\t\t7\x03\x0bC)'\r!\x11\x87##%_\x85\x87\x11))\x1d\x1dW\xb3K+\x1b\x9bM+\x9b\xd2\x02Y=\x1b\r\x15\x1f/!!)#\x1f\x19\r\x1f\x83\x0f\x1f\x15\x1d\x15\x13\x0f\x17\x13\x15\x17\x0f\x1f)\x19\x11+\x11\x15\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00return_v1\x00call_v1\x00get_tuple_element_v1\x00iota_v1\x00constant_v1\x00broadcast_in_dim_v1\x00custom_call_v1\x00add_v1\x00compare_v1\x00select_v1\x00tuple_v1\x00reshape_v1\x00pad_v1\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00callee\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00iota_dimension\x00value\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00xla_shape\x00index\x00jit__lambda_\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00broadcast_dimensions\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00compare_type\x00comparison_direction\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00custom-call.3\x00custom-call.2\x00get-tuple-element.3\x00get-tuple-element.4\x00tuple.5\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00private\x00jax.result_info\x00geqrf\x00xla_fallback_geqrf\x00householder_product\x00xla_fallback_householder_product\x00triu\x00\x00(f32[3,3]{1,0}, f32[3]{0})\x00[0]\x00[1]\x00main\x00public\x00ProductOfElementaryHouseholderReflectors\x00Qr\x00",
    xla_call_module_version=4,
)  # End paste
