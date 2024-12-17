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
from numpy import array, float32, float64, complex64, complex128

data_2023_03_18 = {}

# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_03_18["unbatched"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_geqrf', 'cusolver_orgqr'],
    serialized_date=datetime.date(2023, 3, 18),
    inputs=(),
    expected_outputs=(array([[ 0.        ,  0.9128705 ,  0.40824863],
       [-0.44721356,  0.36514878, -0.8164964 ],
       [-0.8944271 , -0.18257457,  0.40824813]], dtype=float32), array([[-6.7082043e+00, -8.0498438e+00, -9.3914843e+00],
       [ 0.0000000e+00,  1.0954436e+00,  2.1908882e+00],
       [ 0.0000000e+00,  0.0000000e+00,  5.6703755e-08]], dtype=float32)),
    mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "[0]"}, tensor<3x3xf32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf32>
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32>
    %2 = stablehlo.custom_call @cusolver_geqrf(%1) {api_version = 2 : i32, backend_config = "\00\00\00\00\01\00\00\00\03\00\00\00\03\00\00\00\00\00\03\00", operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf32>) -> tuple<tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<196608xf32>>
    %3 = stablehlo.get_tuple_element %2[0] : (tuple<tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<196608xf32>>) -> tensor<3x3xf32>
    %4 = stablehlo.get_tuple_element %2[1] : (tuple<tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<196608xf32>>) -> tensor<3xf32>
    %5 = stablehlo.get_tuple_element %2[2] : (tuple<tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<196608xf32>>) -> tensor<i32>
    %6 = stablehlo.get_tuple_element %2[3] : (tuple<tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<196608xf32>>) -> tensor<196608xf32>
    %7 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i32>) -> tensor<i32>
    %9 = stablehlo.compare  EQ, %5, %8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %11 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %13 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %14 = stablehlo.select %13, %3, %12 : tensor<3x3xi1>, tensor<3x3xf32>
    %15 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %16 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %18 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<1xi1>) -> tensor<3xi1>
    %19 = stablehlo.select %18, %4, %17 : tensor<3xi1>, tensor<3xf32>
    %20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %21 = stablehlo.pad %14, %20, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32>
    %22 = stablehlo.custom_call @cusolver_orgqr(%21, %19) {api_version = 2 : i32, backend_config = "\00\00\00\00\01\00\00\00\03\00\00\00\03\00\00\00\03\00\00\00 \81\00\00", operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf32>, tensor<3xf32>) -> tuple<tensor<3x3xf32>, tensor<i32>, tensor<33056xf32>>
    %23 = stablehlo.get_tuple_element %22[0] : (tuple<tensor<3x3xf32>, tensor<i32>, tensor<33056xf32>>) -> tensor<3x3xf32>
    %24 = stablehlo.get_tuple_element %22[1] : (tuple<tensor<3x3xf32>, tensor<i32>, tensor<33056xf32>>) -> tensor<i32>
    %25 = stablehlo.get_tuple_element %22[2] : (tuple<tensor<3x3xf32>, tensor<i32>, tensor<33056xf32>>) -> tensor<33056xf32>
    %26 = stablehlo.constant dense<0> : tensor<i32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<i32>
    %28 = stablehlo.compare  EQ, %24, %27,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %29 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %30 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %32 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %33 = stablehlo.select %32, %23, %31 : tensor<3x3xi1>, tensor<3x3xf32>
    %34 = call @triu(%14) : (tensor<3x3xf32>) -> tensor<3x3xf32>
    return %33, %34 : tensor<3x3xf32>, tensor<3x3xf32>
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
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01+\x05\x01\x05\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03~\x02\xf79\x01\x99\x0f\x0f\x17\x13\x0f\x07\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x17\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x1b\x13\x13\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0bK\x0b\x13\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0bK\x0b\x13\x0b\x03_O/\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1f\x0b\x0b\x0f\x17\x1b\x0f\x0f\x0f\x0f\x1f\x0b\x1fO/\x0b\x0b\x13\x17\x039\x17\x0f\x0f\x07\x07\x07\x07\x17\x13\x17\x07\x1b\x0f\x17\x13\x1b\x17\x17\x13\x13\x1b\x13\x13\x13\x13\x13\x13\x17\x02\x06\t\x1d{\x05\x1d\x93\x05\x17\x1f\n\x06\x01\x03\x03\x13\xcb\x1dS\x05\x1f\x05!\x05#\x05%\x05'\x03\x03\r\xe9\x05)\x05+\x05-\x05/\x051\x03\x03#\xc7\x053\x1d[\x05\x055\x057\x03\x03\r\xd1\x17\x1f\x06\x06\x01\x059\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x03\x03\x0f\xdd\x03\x03\x0f\xdf\x03\x03\x0f\xe1\x03\x03\r\xe5\x03\x05'\xa7)\xe7\x03\x03\x13\xeb\x03\x03\x11M\x05I\x03\x0b\x17\x9d\x19\xb1\x1b\xb3\x11\xbd\x1d\xbf\x03\x0b\x17\xa3\x19\xc3\x1b\xa3\x11\xa5\x1d\xc5\x05K\x1dW\x05\x05M\x03\x03\r\xc9\x05O\x03\x03#\xcd\x1da\x05\x05Q\x03\x05'\xa7)\xcf\x1dg\x05\x05S\x1dk\x05\x05U\x1do\x05\x05W\x1ds-\x05Y\x1dw-\x05[\x03\x11/\xa91\xd33\xd55\x9d7\xab9\xd7;\xad=\xdb\x05]\x03\x03\x0f\xe3\x03\x03\x13\xed\x1d\x83\x05\x05_\x03\x07\x87\x9f\x89\x9f\x8b\x9f\x05a\x05c\x05e\x1d\x8f\x05\x05g\x03\x11/\xa91\xef3\xf15\x9d7\xab9\xf3;\xad=\xf5\x05i\x03\x03\x97\xa5\x05k\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dm\x03\x03\xc1\x1do\t\x07\x0b\x05\x05\x01\x03\x03\xd9\x1f/\x01#!\x03\x05\xb5\xb9\r\x03\xa1\xb7\x1dq\r\x03\xa1\xbb\x1ds\x1du\x1dw\r\x01##\x1dy\x13\x0b\x01\x1f\x03\t\xff\xff\xff\xff\x1f%\x01\x13\x0b\x05\x07\x05\x1f\x05\t\x00\x00\x00\x00\x1d{\x1d}\x03\x03\x99\x15\x03\x01\x01\x01\x03\t\x99\x9b\xaf\x9b\x13\t\x01\x13\t\x05\x13\t\t\x13\t\r\x1f\x03\t\x00\x00\x00\x00\x07\x01\x1f\x05\t\x00\x00\xc0\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x7f\x1d\x81\x03\x05\x99\x9b\x03\x07\x99\xaf\x9b)\x05\r\r\x07)\x01\t)\x01\x07\t\x1b\x1d\x01)\x05\r\r\t)\x03\r\x07)\x05\r\r\r\x13)\x03\x04\x000\x07)\x01\r)\x05\x05\x05\r)\x03\t\x0b)\x03\x04\x12\x08\x07\x11\x01\x05\x01\x01\x11\x03\x01\x03\x01)\x03\x01\x0b)\x03%\x07/\t\x01\x11\x03\x17)\x03\t\x15)\x03\x05\x15)\x03\x01\x15)\x03\x05\r)\x03\r\r)\x03\x05\x0b/\x07\x01\x03\x1f\x04\xe6\x05\x05\x01\x11\x0bK\x07\x03\x01\t\x0f\x11\x0bO\x05\x03G\x91\x0b\x03q!\x03'\x17\x06u\x03\x01\x03\x01\x13\x07\x01y\x03)\x03\x03\x07\x07\x01?\x03\x01\x03\x05\x07\x07\x01A\x03\x11\x03\x05\x07\x07\x01C\x03\x03\x03\x05\x07\x07\x01}\x03\x17\x03\x05\x05\x03\x01E\x03\x03\x03\x07\x01\x07\x03\x03\x03\x0f\r\x07\x01G\x03\x19\x05\x0b\x11\x03\x07\x01\x07\x03\x1b\x03\x13\x05\x03\x01\x15\x03\x05\x03\x07\x01\x07\x03\x01\x03\x17\x03\x07\x01I\x03\x13\x03\x15\t\x06\x01\x03\x01\x07\x1b\x07\x19\x03\x07\x01\x07\x031\x03\x13\x05\x03\x01\x15\x03\x05\x03\x07\x01\x07\x03\x11\x03!\x03\x07\x01\x7f\x033\x03\x1f\t\x06\x01\x03\x11\x07%\t#\x05\x03\x81+\x03\x05\x19\x07\x8d\x85\x03\x01\x05\x1d)\x13\x07\x03\x91\x037\x05+'\x07\x07\x03?\x03\x01\x03-\x07\x07\x03A\x03\x03\x03-\x07\x07\x03C\x03\x1f\x03-\x05\x03\x03E\x03\x03\x03\x07\x03\x07\x03\x03\x035\r\x07\x03G\x03\x19\x0517\x03\x07\x03\x07\x03\x1b\x039\x05\x03\x03\x15\x03\x05\x03\x07\x03\x07\x03\x01\x03=\x03\x07\x03I\x03\x13\x03;\t\x06\x03\x03\x01\x07A/?\x1b\x07\t\x95\x03\x01\x03\x1d\x11\x04\x0b\x05CE\x0f\x11\tQ\x05\x03\x15+\x03\x01\x0b\x0b\x03U!\x03\x0f\x05\x03\tY\x03\x03\x03\x07%\x07\x03\x0f\x03\x05\x15\x06%\x03\x0f\x05\x03\x07\x0b\x03_]\x03\x0f\r\x07ec\x03\x13\x05\t\x0b\x05\x03\t+\x03\x05\x03\x07i\x07\x03\x01\x03\x0f\t\x06m\x03\x01\x07\r\x11\x01\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\x86\x19\x83\x1f3\x1f+\x11\x0f\x0b\t\t\x0b!\x0fY\x87##%_=\x85\x87W\xb3K\x9bM\x9b\xd2\x02\x1b\x1f/!!)#\x1f\x19+\x1b\x1f\x83\x1f\x15\x1d\x15+\x13\r\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15+\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00get_tuple_element_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00index\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00jax.result_info\x00triu\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x00\x00\x03\x00\x00cusolver_geqrf\x00\x00\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00 \x81\x00\x00\x00cusolver_orgqr\x00",
    xla_call_module_version=4,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_03_18["batched"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cublas_geqrf_batched', 'cusolver_orgqr'],
    serialized_date=datetime.date(2023, 3, 18),
    inputs=(),
    expected_outputs=(array([[[ 0.        ,  0.91287094,  0.40824836],
        [-0.4472136 ,  0.36514843, -0.81649655],
        [-0.8944272 , -0.18257417,  0.4082483 ]],

       [[-0.42426407,  0.80828977,  0.40824953],
        [-0.5656854 ,  0.11547142, -0.8164964 ],
        [-0.7071068 , -0.5773508 ,  0.4082474 ]]], dtype=float32), array([[[-6.7082038e+00, -8.0498447e+00, -9.3914852e+00],
        [ 0.0000000e+00,  1.0954450e+00,  2.1908898e+00],
        [ 0.0000000e+00,  0.0000000e+00,  4.8374091e-08]],

       [[-2.1213203e+01, -2.2910259e+01, -2.4607319e+01],
        [ 0.0000000e+00,  3.4641042e-01,  6.9282258e-01],
        [ 0.0000000e+00,  0.0000000e+00,  1.4548683e-06]]], dtype=float32)),
    mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<2x3x3xf32> {jax.result_info = "[0]"}, tensor<2x3x3xf32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<18xf32>
    %1 = stablehlo.reshape %0 : (tensor<18xf32>) -> tensor<2x3x3xf32>
    %2 = stablehlo.custom_call @cublas_geqrf_batched(%1) {api_version = 2 : i32, backend_config = "\00\00\00\00\02\00\00\00\03\00\00\00\03\00\00\00", operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x3x3xf32>) -> tuple<tensor<2x3x3xf32>, tensor<2x3xf32>, tensor<16xi8>, tensor<16xi8>>
    %3 = stablehlo.get_tuple_element %2[0] : (tuple<tensor<2x3x3xf32>, tensor<2x3xf32>, tensor<16xi8>, tensor<16xi8>>) -> tensor<2x3x3xf32>
    %4 = stablehlo.get_tuple_element %2[1] : (tuple<tensor<2x3x3xf32>, tensor<2x3xf32>, tensor<16xi8>, tensor<16xi8>>) -> tensor<2x3xf32>
    %5 = stablehlo.get_tuple_element %2[2] : (tuple<tensor<2x3x3xf32>, tensor<2x3xf32>, tensor<16xi8>, tensor<16xi8>>) -> tensor<16xi8>
    %6 = stablehlo.get_tuple_element %2[3] : (tuple<tensor<2x3x3xf32>, tensor<2x3xf32>, tensor<16xi8>, tensor<16xi8>>) -> tensor<16xi8>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.pad %3, %7, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xf32>, tensor<f32>) -> tensor<2x3x3xf32>
    %9 = stablehlo.custom_call @cusolver_orgqr(%8, %4) {api_version = 2 : i32, backend_config = "\00\00\00\00\02\00\00\00\03\00\00\00\03\00\00\00\03\00\00\00 \81\00\00", operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x3x3xf32>, tensor<2x3xf32>) -> tuple<tensor<2x3x3xf32>, tensor<2xi32>, tensor<33056xf32>>
    %10 = stablehlo.get_tuple_element %9[0] : (tuple<tensor<2x3x3xf32>, tensor<2xi32>, tensor<33056xf32>>) -> tensor<2x3x3xf32>
    %11 = stablehlo.get_tuple_element %9[1] : (tuple<tensor<2x3x3xf32>, tensor<2xi32>, tensor<33056xf32>>) -> tensor<2xi32>
    %12 = stablehlo.get_tuple_element %9[2] : (tuple<tensor<2x3x3xf32>, tensor<2xi32>, tensor<33056xf32>>) -> tensor<33056xf32>
    %13 = stablehlo.constant dense<0> : tensor<i32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %15 = stablehlo.compare  EQ, %11, %14,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1>
    %17 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f32>) -> tensor<2x3x3xf32>
    %19 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x3x3xi1>
    %20 = stablehlo.select %19, %10, %18 : tensor<2x3x3xi1>, tensor<2x3x3xf32>
    %21 = call @triu(%3) : (tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    return %20, %21 : tensor<2x3x3xf32>, tensor<2x3x3xf32>
  }
  func.func private @triu(%arg0: tensor<2x3x3xf32>) -> tensor<2x3x3xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32>
    %1 = stablehlo.constant dense<-1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<3x3xi32>
    %3 = stablehlo.add %0, %2 : tensor<3x3xi32>
    %4 = stablehlo.iota dim = 1 : tensor<3x3xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
    %6 = stablehlo.broadcast_in_dim %5, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x3x3xf32>
    %9 = stablehlo.select %6, %8, %arg0 : tensor<2x3x3xi1>, tensor<2x3x3xf32>
    return %9 : tensor<2x3x3xf32>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01+\x05\x01\x05\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03\x96\x02\xff=\x01\x9f\x17\x0f\x0f\x0f\x07\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x17\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0bK\x0b\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0bK\x0b\x13\x1b\x13\x13\x13\x13\x0b\x03ao/\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0fO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0bO\x1f\x0b\x0b\x0f\x17\x1b\x0f\x0f\x0f\x0f\x0b\x0b\x13\x17\x1f\x0b/\x1fo\x03=\x1b\x07\x07\x07\x0f\x17\x0f\x07\x13\x07\x13\x1b\x17\x13\x1b\x17\x17\x13\x17\x13\x13\x1b\x07\x13\x13\x13\x17\x13\x1b\x13\x02\x1a\n\x17\x1d\n\x06\x01\x1d\x8f\x01\x1dK\x01\x1dy\x01\x1f\x05!\x03\x03\x0f\xd1\x05#\x05%\x05'\x05)\x05+\x05-\x05/\x051\x03\x03!\xcd\x053\x1dS\x01\x055\x057\x03\x03\x0b\xd9\x17\x1d\x06\x06\x01\x059\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x03\x03\x11\xe5\x03\x03\x11\xe7\x03\x03\x11\xe9\x03\x03\x13E\x05I\x03\x0b\x15\xa3\x17\xb7\x19\xb9\x13\xc3\x1b\xc5\x03\x0b\x15\xa9\x17\xc9\x19\xa9\x13\xab\x1b\xcb\x05K\x1dO\x01\x05M\x03\x03\x0b\xcf\x05O\x03\x03!\xd3\x1dY\x01\x05Q\x03\x05%\xad'\xd5\x1d_\x01\x05S\x03\x03\x0f\xd7\x1de\x01\x05U\x1di\x01\x05W\x1dm\x01\x05Y\x1dq+\x05[\x1du+\x05]\x03\x11-\xaf/\xdb1\xdd3\xa35\xb17\xdf9\xb3;\xe3\x05_\x03\x03\x11\xeb\x1d\x7f\x01\x05a\x03\x07\x83\xa5\x85\xa5\x87\xa5\x05c\x05e\x05g\x1d\x8b\x01\x05i\x03\x11-\xaf/\xed1\xef3\xa35\xb17\xf19\xb3;\xf3\x05k\x03\x03\x0b\xf5\x03\x05%\xad'\xf7\x03\x03\x0f\xf9\x03\x03\x0b\xfb\x03\x03\x0f\xfd\x03\x03\x9d\xab\x05m\x1f/1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1f\x1b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1do\x03\x03\xc7\x1dq\t\x07\x0b\x05\x05\x01\x03\x03\xe1\x1f1!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x05\xbb\xbf\r\x03\xa7\xbd\x1ds\r\x03\xa7\xc1\x1du\x1dw\x1dy\r\x01#!\x1d{\x13\x05\x01\x1f\r\t\xff\xff\xff\xff\x1f#\x01\x13\x05\x05\x07\x05\x1f'!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f\t\t\x00\x00\x00\x00\x1d}\x1d\x7f\x03\x03\x9f\x15\x03\x01\x01\x01\x03\t\x9f\xb5\xa1\xa1\x13\x03\x01\x13\x03\x05\x13\x03\t\x13\x03\r\x1d\x81\x1d\x83\x03\x05\x9f\xb5\x03\x07\x9f\xa1\xa1\x1f\r\t\x00\x00\x00\x00\x07\x01\x1f;\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\t\t\x00\x00\xc0\x7f\x1f\x1b1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00)\x07\t\r\r\x07\x1b\x1d\t)\x01\x07)\x05\r\r\x03)\x01\x03\x01)\x03A-\x13)\x03\t\x03)\x07\t\r\r\x0f)\x05\t\r\x07)\x03\r\x05)\x03\x04\x12\x08\x07\x11\x01\x05\x01\x01\x11\x03\x01\x03\x01)\x03\x01\x05)\x05\r\r\x0f)\x03\t\x05)\x03I\x07/\t\x01\x19\x11\x11\x17)\x03\r\x13)\x03\t\x13)\x03\x05\x13/\x07\x01\x15\x1d)\x03\t\x0f)\x07\t\x05\x05\x0f)\x03\x05\x05\x04r\x04\x05\x01\x11\tC\x07\x03\x01\t\x0b\x11\tG\x05\x03-]\t\x03o\x1f\x03)\x17\x06s\x03\x01\x03\x01\x13\x07\x07w\x03+\x03\x03\x05\x07\x07=\x03\x01\x03\x05\x05\x07\x07?\x03\x19\x03\x05\x05\x07\x07A\x03\x11\x03\x05\x05\x07\x07{\x03\x11\x03\x05\x07\x03})\x03\t\x19\x07\x89\x81\x03\x01\x05\x07\x0f\x13\x07\x03\x8d\x035\x05\x11\t\x05\x07\x03=\x03\x01\x03\x13\x05\x07\x03?\x03\x15\x03\x13\x05\x07\x03A\x03\x1d\x03\x13\x07\x03\x03\x91\x03\r\x03\x07\x03\r\x03\x15\x03\x1b\r\x07\x03\x93\x037\x05\x17\x1d\x03\x07\x03\x95\x039\x03\x1f\x07\x03\x03\x97\x03\t\x03\x07\x03\r\x03\x01\x03#\x03\x07\x03\x99\x03\x17\x03!\x0f\x06\x03\x03\x01\x07'\x15%\x1b\x07\x05\x9b\x03\x01\x03\x07\x11\x04\t\x05)+\x0b\x11\x05I\x05\x03\x17/\x03\x01\t\t\x03M\x1f\x03\x0b\x07\x03\x05Q\x03\r\x03\x07#\r\x03\x0b\x03\x05\x15\x06#\x03\x0b\x05\x03\x07\t\x03WU\x03\x0b\r\x07][\x03%\x05\t\x0b\x03\x07ca\x03\x17\x03\r\x07\x03\x05)\x03\t\x03\x07g\r\x03\x01\x03\x11\x0f\x06k\x03\x01\x07\x0f\x13\x01\x11\x04\x05\x03\x15\x06\x03\x01\x05\x01\x00Z\x1b\x85\x1f3+#\x11\x0f\x0b\t\t\x0b!\x0fY\x9d##%_=\x8b\x89W\xb9\xc1K\x9bM\x9b\xd2\x02\x1b\x1f/!!)#\x1f\x19+\x1b\x1f\x83\x1f\x15\x1d\x15\x13\r+\r\x11\x0f\x17\x0f\x1f\x15\x15\x17\x11\x11\x19+)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00get_tuple_element_v1\x00constant_v1\x00iota_v1\x00func_v1\x00compare_v1\x00select_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00broadcast_dimensions\x00index\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(2, 3, 3) broadcast_dimensions=(1, 2)]\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(2, 3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(18,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(2, 3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00jax.result_info\x00triu\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x00cublas_geqrf_batched\x00\x00\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00 \x81\x00\x00\x00cusolver_orgqr\x00",
    xla_call_module_version=4,
)  # End paste

data_2024_09_26 = {}

data_2024_09_26["f32"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_geqrf_ffi', 'cusolver_orgqr_ffi'],
    serialized_date=datetime.date(2024, 9, 26),
    inputs=(),
    expected_outputs=(array([[[ 0.        ,  0.91287094,  0.40824842],
        [-0.4472136 ,  0.36514843, -0.8164965 ],
        [-0.8944272 , -0.18257415,  0.40824836]],

       [[-0.42426407,  0.80828977,  0.4082495 ],
        [-0.5656854 ,  0.11547142, -0.8164964 ],
        [-0.7071068 , -0.57735085,  0.4082474 ]]], dtype=float32), array([[[-6.7082038e+00, -8.0498447e+00, -9.3914852e+00],
        [ 0.0000000e+00,  1.0954450e+00,  2.1908898e+00],
        [ 0.0000000e+00,  0.0000000e+00,  4.8374091e-08]],

       [[-2.1213203e+01, -2.2910259e+01, -2.4607319e+01],
        [ 0.0000000e+00,  3.4641042e-01,  6.9282258e-01],
        [ 0.0000000e+00,  0.0000000e+00,  1.4548683e-06]]], dtype=float32)),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3x3xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xf32> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<18xf32>) -> tensor<2x3x3xf32> loc(#loc5)
    %2:2 = stablehlo.custom_call @cusolver_geqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x3x3xf32>) -> (tensor<2x3x3xf32>, tensor<2x3xf32>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xf32>, tensor<f32>) -> tensor<2x3x3xf32> loc(#loc7)
    %4 = stablehlo.custom_call @cusolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>]} : (tensor<2x3x3xf32>, tensor<2x3xf32>) -> tensor<2x3x3xf32> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc12)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3x3xf32> loc(#loc12)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xf32> loc(#loc13)
    return %4, %12 : tensor<2x3x3xf32>, tensor<2x3x3xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":391:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/pad"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/iota"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/add"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/ge"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/broadcast_in_dim"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#'+/37\x03\xc7\x89+\x01C\x17\x07\x0b\x0f\x0b\x13\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x03G\x0b/\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x1f\x1f\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x01\x05\x0b\x0f\x03'\x1b\x07\x07\x17\x0f\x07\x0f\x07\x07\x17\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02F\x05\x17\x05\x1e\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x03)q\x1d\t\x01\x1d7\x01\x1d=\x01\x03\x07\x15\x17\x19\x07\x1b\x07\x05!\x11\x01\x00\x05#\x05%\x05'\x1d\t!\x17\x05\x1a\x065\x1d%'\x05)\x17\x05\x1a\x06\x1d\x05+\x1d-\x01\x05-\x1d1\x01\x05/\x1d5\x01\x051\x053\x1d;\x01\x055\x057\x1dA\x01\x059\x03\x01\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d;\x1d=\x1d?\x13\x07\x01\x0b\x03\x1dA\x05\x01\x03\x03W\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Wy\x1f#\x01#\x17\x03\x05ae\r\x05GcIK\x1dC\r\x05GgIK\x1dE\x1dG\x1dI\x1f\r\t\xff\xff\xff\xff\x1f\x11\t\x00\x00\x00\x00\r\x01\x1dK\x03\x03w\x15\x03\x01\x01\x01\x1f\x1f!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dM\x03\x03\x7f\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\t)\x05\r\r\x0f)\x01\x0f\x1b)\x01\t\x13\x01\x11\x01\x05\x05\x05)\x03I\t)\x05\t\r\t)\x03\r\x13)\x03\t\x13)\x03\r\x07)\x03\x01\x07)\x05\r\r\x15)\x07\t\r\r\x15)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x03\x05\x03\r\x07B\x03\x07\x03\x11\x03B\x1f\t\x03\x19\r\x06#\x03\x05\x03\x05\tG+\x0b\x0b\x05\x05\x1b\x03\x07\x0fF/\r\x03\x05\x05\t\x03\tG3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\x05F\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03%\x05\x15\x17\x05F\x11\x17\x03'\x03\x19\x05F\x11\x11\x03\x05\x03\x03\x15\x06?\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00J\x0bO''\x0f\x0b\t\t\x03\x11#!CS79Y9=)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00main\x00public\x00cusolver_geqrf_ffi\x00cusolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0bC]_ik\x03m\x03o\x03M\x11OQsCSUuY\x07EEE\x11OQ{CSY}U\x03[\x03\x81\x05\x83\x85\x03\x87",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_09_26["f64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_geqrf_ffi', 'cusolver_orgqr_ffi'],
    serialized_date=datetime.date(2024, 9, 26),
    inputs=(),
    expected_outputs=(array([[[ 0.                 ,  0.9128709291752773 ,
          0.408248290463862  ],
        [-0.447213595499958  ,  0.36514837167011005,
         -0.8164965809277264 ],
        [-0.894427190999916  , -0.1825741858350547 ,
          0.40824829046386335]],

       [[-0.42426406871192857,  0.8082903768654768 ,
          0.40824829046386124],
        [-0.565685424949238  ,  0.11547005383792364,
         -0.8164965809277263 ],
        [-0.7071067811865476 , -0.577350269189625  ,
          0.4082482904638641 ]]]), array([[[-6.7082039324993694e+00, -8.0498447189992444e+00,
         -9.3914855054991175e+00],
        [ 0.0000000000000000e+00,  1.0954451150103344e+00,
          2.1908902300206661e+00],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
         -1.7577018578317312e-15]],

       [[-2.1213203435596427e+01, -2.2910259710444144e+01,
         -2.4607315985291855e+01],
        [ 0.0000000000000000e+00,  3.4641016151377924e-01,
          6.9282032302755281e-01],
        [ 0.0000000000000000e+00,  0.0000000000000000e+00,
         -1.8103038069914667e-15]]])),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3x3xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xf64> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<18xf64>) -> tensor<2x3x3xf64> loc(#loc5)
    %2:2 = stablehlo.custom_call @cusolver_geqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x3x3xf64>) -> (tensor<2x3x3xf64>, tensor<2x3xf64>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xf64>, tensor<f64>) -> tensor<2x3x3xf64> loc(#loc7)
    %4 = stablehlo.custom_call @cusolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>]} : (tensor<2x3x3xf64>, tensor<2x3xf64>) -> tensor<2x3x3xf64> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc12)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3x3xf64> loc(#loc12)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xf64> loc(#loc13)
    return %4, %12 : tensor<2x3x3xf64>, tensor<2x3x3xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":391:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/pad"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/iota"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/add"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/ge"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/broadcast_in_dim"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#'+/37\x03\xc7\x89+\x01C\x17\x07\x0b\x0f\x0b\x13\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x03G\x0b/\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x1f/\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x01\x05\x0b\x0f\x03'\x1b\x07\x07\x17\x0f\x07\x0f\x07\x07\x17\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02V\x05\x17\x05\x1e\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x03)q\x1d\t\x01\x1d7\x01\x1d=\x01\x03\x07\x15\x17\x19\x07\x1b\x07\x05!\x11\x01\x00\x05#\x05%\x05'\x1d\t!\x17\x05\x1a\x065\x1d%'\x05)\x17\x05\x1a\x06\x1d\x05+\x1d-\x01\x05-\x1d1\x01\x05/\x1d5\x01\x051\x053\x1d;\x01\x055\x057\x1dA\x01\x059\x03\x01\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d;\x1d=\x1d?\x13\x07\x01\x0b\x03\x1dA\x05\x01\x03\x03W\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Wy\x1f#\x01#\x17\x03\x05ae\r\x05GcIK\x1dC\r\x05GgIK\x1dE\x1dG\x1dI\x1f\r\t\xff\xff\xff\xff\x1f\x11\x11\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\x1dK\x03\x03w\x15\x03\x01\x01\x01\x1f\x1f!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dM\x03\x03\x7f\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x0b)\x05\r\r\x0f)\x01\x0f\x1b)\x01\t\x13\x01\x11\x01\x05\x05\x05)\x03I\t)\x05\t\r\t)\x03\r\x13)\x03\t\x13)\x03\r\x07)\x03\x01\x07)\x05\r\r\x15)\x07\t\r\r\x15)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x03\x05\x03\r\x07B\x03\x07\x03\x11\x03B\x1f\t\x03\x19\r\x06#\x03\x05\x03\x05\tG+\x0b\x0b\x05\x05\x1b\x03\x07\x0fF/\r\x03\x05\x05\t\x03\tG3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\x05F\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03%\x05\x15\x17\x05F\x11\x17\x03'\x03\x19\x05F\x11\x11\x03\x05\x03\x03\x15\x06?\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00J\x0bO''\x0f\x0b\t\t\x03\x11#!CS79Y9=)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00main\x00public\x00cusolver_geqrf_ffi\x00cusolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0bC]_ik\x03m\x03o\x03M\x11OQsCSUuY\x07EEE\x11OQ{CSY}U\x03[\x03\x81\x05\x83\x85\x03\x87",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_09_26["c64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_geqrf_ffi', 'cusolver_orgqr_ffi'],
    serialized_date=datetime.date(2024, 9, 26),
    inputs=(),
    expected_outputs=(array([[[ 0.        +0.j,  0.91287094+0.j,  0.40824836+0.j],
        [-0.4472136 +0.j,  0.36514843+0.j, -0.81649655+0.j],
        [-0.8944272 +0.j, -0.18257417+0.j,  0.4082483 +0.j]],

       [[-0.42426407+0.j,  0.8082899 +0.j,  0.40824962+0.j],
        [-0.5656854 +0.j,  0.11547136+0.j, -0.8164964 +0.j],
        [-0.7071068 +0.j, -0.57735085+0.j,  0.4082474 +0.j]]],
      dtype=complex64), array([[[-6.7082038e+00+0.j, -8.0498447e+00+0.j, -9.3914852e+00+0.j],
        [ 0.0000000e+00+0.j,  1.0954450e+00+0.j,  2.1908898e+00+0.j],
        [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  4.8374091e-08+0.j]],

       [[-2.1213203e+01+0.j, -2.2910259e+01+0.j, -2.4607319e+01+0.j],
        [ 0.0000000e+00+0.j,  3.4641042e-01+0.j,  6.9282258e-01+0.j],
        [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  1.5032538e-06+0.j]]],
      dtype=complex64)),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3x3xcomplex<f32>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xcomplex<f32>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<18xcomplex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc5)
    %2:2 = stablehlo.custom_call @cusolver_geqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x3x3xcomplex<f32>>) -> (tensor<2x3x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc7)
    %4 = stablehlo.custom_call @cusolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>]} : (tensor<2x3x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc12)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x3x3xcomplex<f32>> loc(#loc12)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xcomplex<f32>> loc(#loc13)
    return %4, %12 : tensor<2x3x3xcomplex<f32>>, tensor<2x3x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":391:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/pad"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/iota"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/add"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/ge"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/broadcast_in_dim"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#'+/37\x03\xc9\x89-\x01C\x17\x07\x0b\x0f\x0b\x13\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x03G\x0b/\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x1f/\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x01\x05\x0b\x0f\x03)\x1b\x07\x0b\x17\x0f\x07\x0f\x07\x07\x17\x07\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02^\x05\x17\x05\x1e\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x03)q\x1d\t\x01\x1d7\x01\x1d=\x01\x03\x07\x15\x17\x19\x07\x1b\x07\x05!\x11\x01\x00\x05#\x05%\x05'\x1d\t!\x17\x05\x1a\x065\x1d%'\x05)\x17\x05\x1a\x06\x1d\x05+\x1d-\x01\x05-\x1d1\x01\x05/\x1d5\x01\x051\x053\x1d;\x01\x055\x057\x1dA\x01\x059\x03\x01\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d;\x1d=\x1d?\x13\x07\x01\x0b\x03\x1dA\x05\x01\x03\x03W\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Wy\x1f%\x01#\x17\x03\x05ae\r\x05GcIK\x1dC\r\x05GgIK\x1dE\x1dG\x1dI\x1f\r\t\xff\xff\xff\xff\x1f\x11\x11\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\x1dK\x03\x03w\x15\x03\x01\x01\x01\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dM\x03\x03\x7f\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f+!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x03\x19)\x05\r\r\x0f)\x01\x0f\x1b)\x01\t\x13\x01\x11\x01\x05\x05\x05\t)\x03I\t)\x05\t\r\t)\x03\r\x13)\x03\t\x13)\x03\r\x07)\x03\x01\x07)\x05\r\r\x15)\x07\t\r\r\x15)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x03\x05\x03\r\x07B\x03\x07\x03\x11\x03B\x1f\t\x03\x1b\r\x06#\x03\x05\x03\x05\tG+\x0b\x0b\x05\x05\x1d\x03\x07\x0fF/\r\x03\x05\x05\t\x03\tG3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\x05F\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03'\x05\x15\x17\x05F\x11\x17\x03)\x03\x19\x05F\x11\x11\x03\x05\x03\x03\x15\x06?\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00J\x0bO''\x0f\x0b\t\t\x03\x11#!CS79Y9=)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00main\x00public\x00cusolver_geqrf_ffi\x00cusolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0bC]_ik\x03m\x03o\x03M\x11OQsCSUuY\x07EEE\x11OQ{CSY}U\x03[\x03\x81\x05\x83\x85\x03\x87",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_09_26["c128"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_geqrf_ffi', 'cusolver_orgqr_ffi'],
    serialized_date=datetime.date(2024, 9, 26),
    inputs=(),
    expected_outputs=(array([[[ 0.                 +0.j,  0.9128709291752773 +0.j,
          0.4082482904638621 +0.j],
        [-0.447213595499958  +0.j,  0.36514837167011005+0.j,
         -0.8164965809277263 +0.j],
        [-0.894427190999916  +0.j, -0.18257418583505472+0.j,
          0.40824829046386335+0.j]],

       [[-0.42426406871192857+0.j,  0.808290376865477  +0.j,
          0.4082482904638615 +0.j],
        [-0.565685424949238  +0.j,  0.11547005383792353+0.j,
         -0.8164965809277263 +0.j],
        [-0.7071067811865476 +0.j, -0.577350269189625  +0.j,
          0.4082482904638641 +0.j]]]), array([[[-6.7082039324993694e+00+0.j, -8.0498447189992444e+00+0.j,
         -9.3914855054991175e+00+0.j],
        [ 0.0000000000000000e+00+0.j,  1.0954451150103344e+00+0.j,
          2.1908902300206661e+00+0.j],
        [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
         -1.7577018578317312e-15+0.j]],

       [[-2.1213203435596427e+01+0.j, -2.2910259710444144e+01+0.j,
         -2.4607315985291855e+01+0.j],
        [ 0.0000000000000000e+00+0.j,  3.4641016151377924e-01+0.j,
          6.9282032302755292e-01+0.j],
        [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
         -1.7201790115224914e-15+0.j]]])),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3x3xcomplex<f64>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<18xcomplex<f64>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<18xcomplex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc5)
    %2:2 = stablehlo.custom_call @cusolver_geqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x3x3xcomplex<f64>>) -> (tensor<2x3x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc7)
    %4 = stablehlo.custom_call @cusolver_orgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>]} : (tensor<2x3x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %9, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc12)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x3x3xcomplex<f64>> loc(#loc12)
    %12 = stablehlo.select %10, %11, %2#0 : tensor<2x3x3xi1>, tensor<2x3x3xcomplex<f64>> loc(#loc13)
    return %4, %12 : tensor<2x3x3xcomplex<f64>>, tensor<2x3x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":390:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":391:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/pad"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/iota"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/add"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/ge"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/broadcast_in_dim"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#'+/37\x03\xc9\x89-\x01C\x17\x07\x0b\x0f\x0b\x13\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x03G\x0b/\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0fo\x13\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x1fO\x0b\x0b\x0f\x17O\x0b\x0f\x13\x0f\x0b\x0bO\x01\x05\x0b\x0f\x03)\x1b\x07\x0b\x17\x0f\x07\x0f\x07\x07\x17\x07\x13\x17\x13\x13\x13\x13\x17\x1b\x13\x02~\x05\x17\x05\x1e\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x03)q\x1d\t\x01\x1d7\x01\x1d=\x01\x03\x07\x15\x17\x19\x07\x1b\x07\x05!\x11\x01\x00\x05#\x05%\x05'\x1d\t!\x17\x05\x1a\x065\x1d%'\x05)\x17\x05\x1a\x06\x1d\x05+\x1d-\x01\x05-\x1d1\x01\x05/\x1d5\x01\x051\x053\x1d;\x01\x055\x057\x1dA\x01\x059\x03\x01\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d;\x1d=\x1d?\x13\x07\x01\x0b\x03\x1dA\x05\x01\x03\x03W\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05Wy\x1f%\x01#\x17\x03\x05ae\r\x05GcIK\x1dC\r\x05GgIK\x1dE\x1dG\x1dI\x1f\r\t\xff\xff\xff\xff\x1f\x11!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\x1dK\x03\x03w\x15\x03\x01\x01\x01\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dM\x03\x03\x7f\x15\x01\x01\x01\x13\x07\x05\t\x07\x07\x05\x1f+!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\x03\x19)\x05\r\r\x0f)\x01\x0f\x1b)\x01\t\x13\x01\x11\x01\x05\x05\x05\x0b)\x03I\t)\x05\t\r\t)\x03\r\x13)\x03\t\x13)\x03\r\x07)\x03\x01\x07)\x05\r\r\x15)\x07\t\r\r\x15)\x03\t\x07\x04F\x02\x05\x01Q\x03\x13\x01\x07\x04\x1e\x02\x03\x01\x05\x0bP\x03\x03\x07\x04\xfb\x03!A\x07B\x03\x05\x03\r\x07B\x03\x07\x03\x11\x03B\x1f\t\x03\x1b\r\x06#\x03\x05\x03\x05\tG+\x0b\x0b\x05\x05\x1d\x03\x07\x0fF/\r\x03\x05\x05\t\x03\tG3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\x05F\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03'\x05\x15\x17\x05F\x11\x17\x03)\x03\x19\x05F\x11\x11\x03\x05\x03\x03\x15\x06?\x03\x05\x07\x1b\x1d\t\x17\x04\x03\x05\x0f\x1f\x06\x03\x01\x05\x01\x00J\x0bO''\x0f\x0b\t\t\x03\x11#!CS79Y9=)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11\x1f\x19)\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00broadcast_in_dim_v1\x00constant_v1\x00custom_call_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00mhlo.layout_mode\x00default\x00\x00[0]\x00[1]\x00main\x00public\x00cusolver_geqrf_ffi\x00cusolver_orgqr_ffi\x00\x08_\x19\x05;\x01\x0bC]_ik\x03m\x03o\x03M\x11OQsCSUuY\x07EEE\x11OQ{CSY}U\x03[\x03\x81\x05\x83\x85\x03\x87",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
