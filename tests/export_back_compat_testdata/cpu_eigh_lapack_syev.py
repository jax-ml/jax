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
from numpy import array, float32, complex64

data_2023_03_17 = dict(
    # Pasted from the test output (see back_compat_test.py module docstring)
    f32=dict(
        testdata_version=1,
        platform='cpu',
        custom_call_targets=['lapack_ssyevd'],
        serialized_date=datetime.date(2023, 3, 17),
        inputs=(),
        expected_outputs=(array([[-0.6185769    , -0.20142993   , -0.09725195   ,  0.62983674   ,
            -0.07926044   ,  0.3605001    , -0.019093221  , -0.18446997   ],
          [-0.47070873   ,  0.29325768   , -0.19454119   , -0.6394365    ,
            0.0622955    ,  0.33249345   ,  0.28112718   , -0.22856665   ],
          [-0.32284075   , -0.12361939   ,  0.20547704   , -0.18307868   ,
            0.47294614   , -0.3170349    , -0.6373532    , -0.27266347   ],
          [-0.17497246   , -0.079641335  ,  0.15042791   , -0.15416273   ,
            -0.815209     , -0.38054234   , -0.083263926  , -0.31676024   ],
          [-0.027104253  , -0.26490977   ,  0.32271704   ,  0.08653544   ,
            0.30305928   , -0.33998996   ,  0.6926741    , -0.360857     ],
          [ 0.12076397   ,  0.43288827   , -0.64385164   ,  0.2652551    ,
            0.09482376   , -0.37435007   ,  0.00091664493, -0.40495378   ],
          [ 0.26863196   ,  0.51607686   ,  0.53846526   ,  0.16969058   ,
            -0.021670295  ,  0.35755336   , -0.113144726  , -0.4490505    ],
          [ 0.4165004    , -0.57262254   , -0.2814425    , -0.17463988   ,
            -0.01698498   ,  0.3613705    , -0.12186296   , -0.49314725   ]],
          dtype=float32), array([-2.4598808e+01, -3.3105560e-05, -3.1002426e-05, -1.0103593e-05,
          -1.0022322e-05,  4.0141886e-06,  9.5510331e-06,  2.7659882e+02],
          dtype=float32)),
        mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<8x8xf32> {jax.result_info = "[0]"}, tensor<8xf32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xf32>
    %1 = stablehlo.reshape %0 : (tensor<64xf32>) -> tensor<8x8xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %3 = stablehlo.add %1, %2 : tensor<8x8xf32>
    %4 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %6 = stablehlo.divide %3, %5 : tensor<8x8xf32>
    %7 = call @tril(%6) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.constant dense<8> : tensor<i32>
    %11 = stablehlo.custom_call @lapack_ssyevd(%8, %9, %10, %7) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<8x8xf32>) -> tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>
    %12 = stablehlo.get_tuple_element %11[0] : (tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>) -> tensor<8x8xf32>
    %13 = stablehlo.get_tuple_element %11[1] : (tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>) -> tensor<8xf32>
    %14 = stablehlo.get_tuple_element %11[2] : (tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>) -> tensor<i32>
    %15 = stablehlo.get_tuple_element %11[3] : (tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>) -> tensor<177xf32>
    %16 = stablehlo.get_tuple_element %11[4] : (tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>) -> tensor<43xi32>
    %17 = stablehlo.constant dense<0> : tensor<i32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<i32>
    %19 = stablehlo.compare  EQ, %14, %18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %21 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %23 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1>
    %24 = stablehlo.select %23, %12, %22 : tensor<8x8xi1>, tensor<8x8xf32>
    %25 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %26 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0] : (tensor<1xi1>) -> tensor<8xi1>
    %29 = stablehlo.select %28, %13, %27 : tensor<8xi1>, tensor<8xf32>
    return %24, %29 : tensor<8x8xf32>, tensor<8xf32>
  }
  func.func private @tril(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %3 = stablehlo.add %0, %2 : tensor<8x8xi32>
    %4 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %8 = stablehlo.select %5, %arg0, %7 : tensor<8x8xi1>, tensor<8x8xf32>
    return %8 : tensor<8x8xf32>
  }
}
""",
        mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01-\x05\x01\x05\x01\x03\x05\x03\x1d\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!\x03z\x02\xf77\x01\x9b\x0f\x17\x13\x0b\x07\x0f\x0b\x0b\x0b\x0b\x17\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x17\x0f\x13\x13\x13\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x13\x1b\x13\x13\x03]\x0f/\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1fO\x1f\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x1f\x0f\x0f\x0f\x0f\x0f\x0b\x1fO/\x037\x17\x0f\x07\x0f\x07\x13\x07\x07\x17\x07\x17\x13\x17\x13\x17\x17\x13\x17\x1f\x13\x13\x13\x0f\x17\x13\x13\x13\x02\n\t\x1du\x03\x17\x11\xf6\x04\x01\x03\x03\x13\xc5\x05#\x1f\x1d;\x03\x05%\x05'\x05)\x05+\x17\x11\xf2\x04\x01\x05-\x05/\x051\x053\x03\x03!\xc1\x055\x03\x03\x07\xc3\x1dA\x03\x057\x059\x17\x11\xea\x04\x01\x1do\x15\x03\x03\x07\xd1\x03\x03\x07\xf1\x03\x03\x0f5\x05;\x03\x0b\x17\x9f\x19\xab\x1b\xad\x0f\xb7\x1d\xb9\x03\x0b\x17\xa3\x19\xbd\x1b\xa3\x0f\xa5\x1d\xbf\x05=\x1d?\x03\x05?\x05A\x03\x03!\xc7\x1dG\x03\x05C\x03\x05'\xa7)\xc9\x1dM\x03\x05E\x03\x03\x07\xcb\x1dS\x03\x05G\x1dW\x03\x05I\x1d[+\x05K\x1d_+\x05M\x03\x03c\xcd\x05O\x1dg\x15\x05Q\x1dk\x15\x05S\x03\x03\x07\xcf\x05U\x03\x03s\xa5\x05W\x05Y\x03\x03\x07\xd3\x03\x11{\xd5}\xd7\x7f\xd9\x81\x9f\x83\xdb\x85\xdd\x87\xdf\x89\xe3\x05[\x05]\x05_\x05a\x05c\x05e\x05g\x05i\x03\x03\r\xe5\x03\x03\r\xe7\x03\x03\r\xe9\x03\x03\r\xeb\x03\x03\r\xed\x03\x05'\xa7)\xef\x03\x03\x13\xf3\x03\x03\x13\xf5\x1f'\x01\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dk\x03\x03\xbb\x1dm\t\x07\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1d\x03\x05\xaf\xb3\r\x03\xa1\xb1\x1do\r\x03\xa1\xb5\x1dq\x1ds\x1du\r\x01#\x1f\x1dw\x13\r\x01\x1f\x03\t\x00\x00\x00\x00\x1f!\x01\x13\r\x05\x07\x05\x1f\x07\t\x00\x00\x00\x00\x1f\x17!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00@\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x08\x00\x00\x00\x0b\x05\x1dy\x1d{\x05\x01\x03\t\x9b\x9b\x9b\xa9\x03\x03\xe1\x15\x03\x01\r\x01\x03\x0b\xa9\x9d\x9b\x9d\x9d\x13\x05\x01\x13\x05\x05\x13\x05\t\x13\x05\r\x13\x05\x11\x07\x01\x1f\x07\t\x00\x00\xc0\x7f\x1f\x17!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00)\x05!!\t)\x01\x05\x1b)\x01\t\t)\x03!\t\x1d\x01)\x05!!\x05\x13)\x05!!\x0f)\x03\t\r)\x03\x8a\x05\t)\x03\xad\x05\x11\x01\x05\x01\x0b\x11\x03\x01\x03\x01)\x03\x01\r)\x03\x02\x02\t/\x0b\x01\x0b\x03\x19\x1b)\x03\x01\x13)\x03\t\x13)\x03\x05\x13)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\r\x04:\x05\x05\x01\x11\t3\x07\x03\x01\t\r\x11\t7\x05\x03=}\t\x03Y\x1f\x03#\x15\x06]\x03\x01\x03\x01\x17\x07ea\x03\x01\x03\x03\x0f\x06i\x03\x01\x05\x03\x05\x05\x03\tm\x03\x07\x03\x07-\x05\x03\x01\x03\t\x19\x06-\x03\x01\x05\x07\x0b\x1b\x07\x0bq\x03\x01\x03\r\x05\x03\x01/\x03\x03\x05\x03\x01/\x03\x03\x05\x03\x01w\x03\x03\x1d\x07\x01y\x03%\t\x11\x13\x15\x0f\x07\x07\x01\x8b\x03\x01\x03\x17\x07\x07\x01\x8d\x03\x0b\x03\x17\x07\x07\x01\x8f\x03\x03\x03\x17\x07\x07\x01\x91\x03\x19\x03\x17\x07\x07\x01\x93\x03\x1b\x03\x17\x05\x03\x01#\x03\x03\x03\x07\x01\x05\x03\x03\x03#\x11\x07\x01\x95\x03-\x05\x1d%\x03\x07\x01\x05\x03/\x03'\x05\x03\x011\x03\x07\x03\x07\x01\x05\x03\x01\x03+\x03\x07\x01\x97\x03\x15\x03)\x0b\x06\x01\x03\x01\x07/\x19-\x03\x07\x01\x05\x031\x03'\x05\x03\x011\x03\x07\x03\x07\x01\x05\x03\x0b\x035\x03\x07\x01\x99\x033\x033\x0b\x06\x01\x03\x0b\x079\x1b7\x13\x04\t\x051;\r\x11\x0b9\x05\x03\x15+\x03\x01\t\t\x03=\x1f\x03\x11\x05\x03\x0b#\x03\x03\x03\x07%\x05\x03\x11\x03\x05\x0f\x06%\x03\x11\x05\x03\x07\t\x03EC\x03\x11\x11\x07KI\x03\x15\x05\t\x0b\x05\x03\x0bO\x03\x07\x03\x07Q\x05\x03\x01\x03\x0f\x0b\x06U\x03\x01\x07\r\x01\x11\x13\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\xb2\x19}\x1d\x03\x11\x0f\x0b\t\t\x0b!\x1f/!!)#\x1f\x19\x7f\x0f99m\x19\x85\x89W\xb3K\x9bM\x9b\x96\x04\x1b+\x1b\x1f\x1f\x15\x1d\x15+\x83\x13\r\r\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11+\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00get_tuple_element_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00value\x00index\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_ssyevd\x00",
        xla_call_module_version=4,
    ),  # End paste

    # Pasted from the test output (see back_compat_test.py module docstring)
    f64=dict(
        testdata_version=1,
        platform='cpu',
        custom_call_targets=['lapack_dsyevd'],
        serialized_date=datetime.date(2023, 3, 17),
        inputs=(),
        expected_outputs=(array([[-6.1857700048412056e-01,  2.4081403770912022e-01,
            3.5662489253627483e-01, -6.3034019033669797e-01,
            1.0043483479985752e-16, -2.8842036081919542e-02,
            7.7164692943283169e-25, -1.8446994643771725e-01],
          [-4.7070881487314614e-01,  4.7473787464450845e-01,
            -4.8036836210243367e-01,  4.3802686872516400e-01,
            1.7961797619639258e-01,  8.3080980076741355e-03,
            2.1415294457221756e-01, -2.2856669794666584e-01],
          [-3.2284062926217072e-01, -5.4336490915553370e-01,
            2.2181041859724990e-01,  2.9947877954402297e-01,
            -3.6491813600134632e-01,  3.2867679819727436e-01,
            3.8223299448843473e-01, -2.7266344945561438e-01],
          [-1.7497244365119530e-01, -8.9251550609769414e-02,
            -6.3518515114898394e-02,  1.9162997359209971e-01,
            -2.2087281326110139e-01,  5.9957027043505064e-02,
            -8.7632498908241274e-01, -3.1676020096456303e-01],
          [-2.7104258040220038e-02, -3.3772873786627672e-01,
            2.5901386593721748e-01,  1.7032650752287815e-01,
            6.7521217612940332e-01, -4.5036136532965476e-01,
            -1.2279030059078447e-02, -3.6085695247351163e-01],
          [ 1.2076392757075530e-01, -3.3834734096469254e-01,
            -6.5506827461665540e-01, -5.0472498521116749e-01,
            6.9987430903492118e-02,  1.0595648906599275e-01,
            8.3443844143082022e-02, -4.0495370398246017e-01],
          [ 2.6863211318173097e-01,  2.2958613191407318e-01,
            6.3952843755683941e-02,  1.8776775771084137e-02,
            -5.3523731432241317e-01, -5.9199531677602002e-01,
            1.7916671834524248e-01, -4.4905045549140887e-01],
          [ 4.1650029879270661e-01,  3.6355449432857079e-01,
            2.9755313100756142e-01,  1.6826270392615944e-02,
            1.9621068035557282e-01,  5.6830030587314817e-01,
            2.9607517592514246e-02, -4.9314720700035747e-01]]), array([-2.4598804776133626e+01, -4.6567755957874661e-14,
          -1.9932120610662194e-14, -5.7323356091157378e-15,
          -4.5459724251334835e-16,  4.0479851042511616e-14,
            9.2325194924982089e-14,  2.7659880477613365e+02])),
        mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<8x8xf64> {jax.result_info = "[0]"}, tensor<8xf64> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xf64>
    %1 = stablehlo.reshape %0 : (tensor<64xf64>) -> tensor<8x8xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xf64>) -> tensor<8x8xf64>
    %3 = stablehlo.add %1, %2 : tensor<8x8xf64>
    %4 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<8x8xf64>
    %6 = stablehlo.divide %3, %5 : tensor<8x8xf64>
    %7 = call @tril(%6) : (tensor<8x8xf64>) -> tensor<8x8xf64>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.constant dense<8> : tensor<i32>
    %11 = stablehlo.custom_call @lapack_dsyevd(%8, %9, %10, %7) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<8x8xf64>) -> tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>
    %12 = stablehlo.get_tuple_element %11[0] : (tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>) -> tensor<8x8xf64>
    %13 = stablehlo.get_tuple_element %11[1] : (tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>) -> tensor<8xf64>
    %14 = stablehlo.get_tuple_element %11[2] : (tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>) -> tensor<i32>
    %15 = stablehlo.get_tuple_element %11[3] : (tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>) -> tensor<177xf64>
    %16 = stablehlo.get_tuple_element %11[4] : (tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>) -> tensor<43xi32>
    %17 = stablehlo.constant dense<0> : tensor<i32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<i32>
    %19 = stablehlo.compare  EQ, %14, %18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %21 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f64>) -> tensor<8x8xf64>
    %23 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1>
    %24 = stablehlo.select %23, %12, %22 : tensor<8x8xi1>, tensor<8x8xf64>
    %25 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %26 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<f64>) -> tensor<8xf64>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0] : (tensor<1xi1>) -> tensor<8xi1>
    %29 = stablehlo.select %28, %13, %27 : tensor<8xi1>, tensor<8xf64>
    return %24, %29 : tensor<8x8xf64>, tensor<8xf64>
  }
  func.func private @tril(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %3 = stablehlo.add %0, %2 : tensor<8x8xi32>
    %4 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f64>) -> tensor<8x8xf64>
    %8 = stablehlo.select %5, %arg0, %7 : tensor<8x8xi1>, tensor<8x8xf64>
    return %8 : tensor<8x8xf64>
  }
}
""",
        mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01-\x05\x01\x05\x01\x03\x05\x03\x1d\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!\x03z\x02\xf77\x01\x9b\x0f\x17\x13\x0b\x07\x0f\x0b\x0b\x0b\x0b\x17\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x17\x0f\x13\x13\x13\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x13\x1b\x13\x13\x03]\x0f/\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O/\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x1f\x0f\x0f\x0f\x0f\x0f\x0b/O/\x037\x17\x0f\x07\x0f\x07\x13\x07\x07\x17\x07\x17\x13\x17\x13\x17\x17\x13\x17\x1f\x13\x13\x13\x0f\x17\x13\x13\x13\x02:\t\x1du\x03\x17\x11\xf6\x04\x01\x03\x03\x13\xc5\x05#\x1f\x1d;\x03\x05%\x05'\x05)\x05+\x17\x11\xf2\x04\x01\x05-\x05/\x051\x053\x03\x03!\xc1\x055\x03\x03\x07\xc3\x1dA\x03\x057\x059\x17\x11\xea\x04\x01\x1do\x15\x03\x03\x07\xd1\x03\x03\x07\xf1\x03\x03\x0f5\x05;\x03\x0b\x17\x9f\x19\xab\x1b\xad\x0f\xb7\x1d\xb9\x03\x0b\x17\xa3\x19\xbd\x1b\xa3\x0f\xa5\x1d\xbf\x05=\x1d?\x03\x05?\x05A\x03\x03!\xc7\x1dG\x03\x05C\x03\x05'\xa7)\xc9\x1dM\x03\x05E\x03\x03\x07\xcb\x1dS\x03\x05G\x1dW\x03\x05I\x1d[+\x05K\x1d_+\x05M\x03\x03c\xcd\x05O\x1dg\x15\x05Q\x1dk\x15\x05S\x03\x03\x07\xcf\x05U\x03\x03s\xa5\x05W\x05Y\x03\x03\x07\xd3\x03\x11{\xd5}\xd7\x7f\xd9\x81\x9f\x83\xdb\x85\xdd\x87\xdf\x89\xe3\x05[\x05]\x05_\x05a\x05c\x05e\x05g\x05i\x03\x03\r\xe5\x03\x03\r\xe7\x03\x03\r\xe9\x03\x03\r\xeb\x03\x03\r\xed\x03\x05'\xa7)\xef\x03\x03\x13\xf3\x03\x03\x13\xf5\x1f'\x01\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dk\x03\x03\xbb\x1dm\t\x07\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1d\x03\x05\xaf\xb3\r\x03\xa1\xb1\x1do\r\x03\xa1\xb5\x1dq\x1ds\x1du\r\x01#\x1f\x1dw\x13\r\x01\x1f\x03\t\x00\x00\x00\x00\x1f!\x01\x13\r\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x17!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x08\x00\x00\x00\x0b\x05\x1dy\x1d{\x05\x01\x03\t\x9b\x9b\x9b\xa9\x03\x03\xe1\x15\x03\x01\r\x01\x03\x0b\xa9\x9d\x9b\x9d\x9d\x13\x05\x01\x13\x05\x05\x13\x05\t\x13\x05\r\x13\x05\x11\x07\x01\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x17!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00)\x05!!\t)\x01\x05\x1b)\x01\t\x0b)\x03!\t\x1d\x01)\x05!!\x05\x13)\x05!!\x0f)\x03\t\r)\x03\x8a\x05\t)\x03\xad\x05\x11\x01\x05\x01\x0b\x11\x03\x01\x03\x01)\x03\x01\r)\x03\x02\x02\t/\x0b\x01\x0b\x03\x19\x1b)\x03\x01\x13)\x03\t\x13)\x03\x05\x13)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\r\x04:\x05\x05\x01\x11\t3\x07\x03\x01\t\r\x11\t7\x05\x03=}\t\x03Y\x1f\x03#\x15\x06]\x03\x01\x03\x01\x17\x07ea\x03\x01\x03\x03\x0f\x06i\x03\x01\x05\x03\x05\x05\x03\tm\x03\x07\x03\x07-\x05\x03\x01\x03\t\x19\x06-\x03\x01\x05\x07\x0b\x1b\x07\x0bq\x03\x01\x03\r\x05\x03\x01/\x03\x03\x05\x03\x01/\x03\x03\x05\x03\x01w\x03\x03\x1d\x07\x01y\x03%\t\x11\x13\x15\x0f\x07\x07\x01\x8b\x03\x01\x03\x17\x07\x07\x01\x8d\x03\x0b\x03\x17\x07\x07\x01\x8f\x03\x03\x03\x17\x07\x07\x01\x91\x03\x19\x03\x17\x07\x07\x01\x93\x03\x1b\x03\x17\x05\x03\x01#\x03\x03\x03\x07\x01\x05\x03\x03\x03#\x11\x07\x01\x95\x03-\x05\x1d%\x03\x07\x01\x05\x03/\x03'\x05\x03\x011\x03\x07\x03\x07\x01\x05\x03\x01\x03+\x03\x07\x01\x97\x03\x15\x03)\x0b\x06\x01\x03\x01\x07/\x19-\x03\x07\x01\x05\x031\x03'\x05\x03\x011\x03\x07\x03\x07\x01\x05\x03\x0b\x035\x03\x07\x01\x99\x033\x033\x0b\x06\x01\x03\x0b\x079\x1b7\x13\x04\t\x051;\r\x11\x0b9\x05\x03\x15+\x03\x01\t\t\x03=\x1f\x03\x11\x05\x03\x0b#\x03\x03\x03\x07%\x05\x03\x11\x03\x05\x0f\x06%\x03\x11\x05\x03\x07\t\x03EC\x03\x11\x11\x07KI\x03\x15\x05\t\x0b\x05\x03\x0bO\x03\x07\x03\x07Q\x05\x03\x01\x03\x0f\x0b\x06U\x03\x01\x07\r\x01\x11\x13\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\xb2\x19}\x1d\x03\x11\x0f\x0b\t\t\x0b!\x1f/!!)#\x1f\x19\x7f\x0f99m\x19\x85\x89W\xb3K\x9bM\x9b\x96\x04\x1b+\x1b\x1f\x1f\x15\x1d\x15+\x83\x13\r\r\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11+\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00get_tuple_element_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00value\x00index\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_dsyevd\x00",
        xla_call_module_version=4,
    ),  # End paste

    # Pasted from the test output (see back_compat_test.py module docstring)
    c64=dict(
        testdata_version=1,
        platform='cpu',
        custom_call_targets=['lapack_cheevd'],
        serialized_date=datetime.date(2023, 3, 17),
        inputs=(),
        expected_outputs=(array([[-0.6185769    +0.j, -0.20142993   +0.j, -0.09725195   +0.j,
            0.62983674   +0.j, -0.07926044   +0.j,  0.3605001    -0.j,
            -0.019093221  +0.j, -0.18446997   +0.j],
          [-0.47070873   +0.j,  0.29325768   +0.j, -0.19454116   +0.j,
            -0.6394365    +0.j,  0.06229549   +0.j,  0.33249345   +0.j,
            0.28112718   +0.j, -0.22856665   +0.j],
          [-0.32284075   +0.j, -0.12361939   +0.j,  0.20547704   +0.j,
            -0.18307868   +0.j,  0.47294614   +0.j, -0.3170349    +0.j,
            -0.6373532    +0.j, -0.27266347   +0.j],
          [-0.17497246   +0.j, -0.079641335  +0.j,  0.15042792   +0.j,
            -0.15416273   +0.j, -0.815209     +0.j, -0.38054234   +0.j,
            -0.083263926  +0.j, -0.31676024   +0.j],
          [-0.027104257  +0.j, -0.26490977   +0.j,  0.32271704   +0.j,
            0.08653544   +0.j,  0.30305928   +0.j, -0.33998996   +0.j,
            0.6926741    +0.j, -0.360857     +0.j],
          [ 0.120763965  +0.j,  0.43288827   +0.j, -0.64385164   +0.j,
            0.2652551    +0.j,  0.094823755  +0.j, -0.37435007   +0.j,
            0.00091664493+0.j, -0.40495378   +0.j],
          [ 0.26863196   +0.j,  0.51607686   +0.j,  0.53846526   +0.j,
            0.16969058   +0.j, -0.0216703    +0.j,  0.35755336   +0.j,
            -0.113144726  +0.j, -0.4490505    +0.j],
          [ 0.4165004    +0.j, -0.57262254   +0.j, -0.28144246   +0.j,
            -0.17463988   +0.j, -0.016984984  +0.j,  0.3613705    +0.j,
            -0.12186296   +0.j, -0.49314725   +0.j]], dtype=complex64), array([-2.4598808e+01, -3.3105560e-05, -3.1002426e-05, -1.0103593e-05,
          -1.0022322e-05,  4.0141886e-06,  9.5510331e-06,  2.7659882e+02],
          dtype=float32)),
        mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<8x8xcomplex<f32>> {jax.result_info = "[0]"}, tensor<8xf32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xcomplex<f32>>
    %1 = stablehlo.reshape %0 : (tensor<64xcomplex<f32>>) -> tensor<8x8xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>>
    %3 = stablehlo.real %2 : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xf32>
    %4 = stablehlo.imag %2 : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xf32>
    %5 = stablehlo.negate %4 : tensor<8x8xf32>
    %6 = stablehlo.complex %3, %5 : tensor<8x8xcomplex<f32>>
    %7 = stablehlo.add %1, %6 : tensor<8x8xcomplex<f32>>
    %8 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>>
    %10 = stablehlo.divide %7, %9 : tensor<8x8xcomplex<f32>>
    %11 = call @tril(%10) : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>>
    %12 = stablehlo.constant dense<1> : tensor<i32>
    %13 = stablehlo.constant dense<1> : tensor<i32>
    %14 = stablehlo.constant dense<8> : tensor<i32>
    %15 = stablehlo.custom_call @lapack_cheevd(%12, %13, %14, %11) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<8x8xcomplex<f32>>) -> tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>
    %16 = stablehlo.get_tuple_element %15[0] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<8x8xcomplex<f32>>
    %17 = stablehlo.get_tuple_element %15[1] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<8xf32>
    %18 = stablehlo.get_tuple_element %15[2] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<i32>
    %19 = stablehlo.get_tuple_element %15[3] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<81xcomplex<f32>>
    %20 = stablehlo.get_tuple_element %15[4] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<169xf32>
    %21 = stablehlo.get_tuple_element %15[5] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<43xi32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<i32>
    %24 = stablehlo.compare  EQ, %18, %23,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %26 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1>
    %29 = stablehlo.select %28, %16, %27 : tensor<8x8xi1>, tensor<8x8xcomplex<f32>>
    %30 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %31 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %33 = stablehlo.broadcast_in_dim %30, dims = [0] : (tensor<1xi1>) -> tensor<8xi1>
    %34 = stablehlo.select %33, %17, %32 : tensor<8xi1>, tensor<8xf32>
    return %29, %34 : tensor<8x8xcomplex<f32>>, tensor<8xf32>
  }
  func.func private @tril(%arg0: tensor<8x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %3 = stablehlo.add %0, %2 : tensor<8x8xi32>
    %4 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %6 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>>
    %8 = stablehlo.select %5, %arg0, %7 : tensor<8x8xi1>, tensor<8x8xcomplex<f32>>
    return %8 : tensor<8x8xcomplex<f32>>
  }
}
""",
        mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x015\x05\x01\x05\x01\x03\x05\x03%\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%')\x03\xc6\x02\x1e\x02?\x01\xa9\x0f\x17\x13\x0b\x17\x0b\x07\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x17\x0f\x13\x13\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x13\x13\x1b\x17\x03a\x0f/\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O/\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17#\x0f\x0f\x0f\x0f\x0f\x0f\x0b/O\x1f/\x01\x07\x17\x17\x17\x03?\x17\x0f\x07\x0f\x07\x13\x07\x07\x0b\x17\x17\x07\x17\x13\x17\x17\x13\x0f\x17\x17\x13\x17#\x13\x13\x13\x0f\x17\x13\x13\x13\x02&\n\x1d\x83\x03\x17\x13\xf6\x04\x01\x03\x03\x15\xd3\x05+\x17\x13\xf2\x04\x01\x05-\x1f\x1d9\x03\x05/\x051\x053\x055\x057\x059\x05;\x03\x03!\xcf\x05=\x03\x03\x07\xd1\x1d?\x03\x05?\x05A\x17\x13\xea\x04\x01\x1d}\t\x03\x03\x07\xdf\x03\x03\x113\x05C\x03\x0b\x17\xad\x19\xb9\x1b\xbb\x11\xc5\x1d\xc7\x03\x0b\x17\xb1\x19\xcb\x1b\xb1\x11\xb3\x1d\xcd\x05E\x1d=\x03\x05G\x05I\x03\x03!\xd5\x1dE\x03\x05K\x03\x05'\xb5)\xd7\x1dK\x03\x05M\x03\x03\x07\xd9\x1dQ\x03\x05O\x1dU\x03\x05Q\x1dY+\x05S\x1d]+\x05U\x03\x03a\xdb\x05W\x1de\t\x05Y\x1di\t\x05[\x1dm\t\x05]\x1dq\t\x05_\x1du\t\x05a\x1dy\t\x05c\x03\x03\x07\xdd\x05e\x03\x03\x81\xb3\x05g\x05i\x03\x03\x07\xe1\x03\x11\x89\xe3\x8b\xe5\x8d\xe7\x8f\xad\x91\xe9\x93\xeb\x95\xed\x97\xf1\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x03\x03\x0b\xf3\x03\x03\x0b\xf5\x03\x03\x0b\xf7\x03\x03\x0b\xf9\x03\x03\x0b\xfb\x03\x03\x0b\xfd\x03\x05'\xb5)\xff\x03\x03\x07\x02\x02\x1f/\x01\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d{\x03\x03\xc9\x1d}\t\x07\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#%\x03\x05\xbd\xc1\r\x03\xaf\xbf\x1d\x7f\r\x03\xaf\xc3\x1d\x81\x1d\x83\x1d\x85\r\x01#'\x1d\x87\x13\r\x01\x1f\x03\t\x00\x00\x00\x00\x1f)\x01\x13\r\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00@\x00\x00\x00\x00\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x08\x00\x00\x00\x0b\x05\x1d\x89\x1d\x8b\x05\x01\x03\t\xa9\xa9\xa9\xb7\x03\x03\xef\x15\x03\x01\r\x01\x03\r\xb7\xab\xa9\xab\xab\xab\x13\x05\x01\x13\x05\x05\x13\x05\t\x13\x05\r\x13\x05\x11\x13\x05\x15\x07\x01\x1f\x07\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f#\t\x00\x00\xc0\x7f\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03\x15\x06\x02\x03\x03\x07\n\x02\x03\x03\x15\x0e\x02)\x05!!\x11)\x01\x05\x1b)\x01\x11\t)\x03!\t\x1d\x01\x03\t)\x05!!\x05)\x05!!\t\x13)\x05!!\x0f)\x03\t\r)\x03\x8a\x02\x11)\x03J\x05\t)\x03\xad\x05)\x01\t\x11\x01\x05\x01\x0b\x11\x03\x01\x03\x01)\x03\x01\r)\x03\x02\x02\x11/\r\x01\x0b\x03\x1d\x1f!)\x03\x01\x17)\x03\t\x17)\x03\x05\x17)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\r\x04\xda\x05\x05\x01\x11\r1\x07\x03\x01\t\r\x11\r5\x05\x03G\x91\t\x03W\x1f\x03+\x15\x06[\x03\x01\x03\x01\x17\x07c_\x03\x01\x03\x03\x19\x06g\x03\x15\x03\x05\x1b\x06k\x03\x15\x03\x05\x1d\x06o\x03\x15\x03\t\x1f\x06s\x03\x01\x05\x07\x0b\x0f\x06w\x03\x01\x05\x03\r\x05\x03\r{\x03\x07\x03\x07-\x05\x03\x01\x03\x11!\x06-\x03\x01\x05\x0f\x13#\x07\x0f\x7f\x03\x01\x03\x15\x05\x03\x01/\x03\x03\x05\x03\x01/\x03\x03\x05\x03\x01\x85\x03\x03%\x07\x01\x87\x03-\t\x19\x1b\x1d\x17\x07\x07\x01\x99\x03\x01\x03\x1f\x07\x07\x01\x9b\x03\x0b\x03\x1f\x07\x07\x01\x9d\x03\x03\x03\x1f\x07\x07\x01\x9f\x03\x1d\x03\x1f\x07\x07\x01\xa1\x03\x1f\x03\x1f\x07\x07\x01\xa3\x03!\x03\x1f\x05\x03\x01#\x03\x03\x03\x07\x01\x05\x03\x03\x03-\x11\x07\x01\xa5\x035\x05%/\x03\x07\x01\x05\x037\x031\x05\x03\x01\xa7\x03\x07\x03\x07\x01\x05\x03\x01\x035\x03\x07\x01\x12\x02\x03\x19\x033\x0b\x06\x01\x03\x01\x079!7\x03\x07\x01\x05\x039\x031\x05\x03\x01\x16\x02\x03#\x03\x07\x01\x05\x03\x0b\x03?\x03\x07\x01\x1a\x02\x03;\x03=\x0b\x06\x01\x03\x0b\x07C#A\x13\x04\r\x05;E\r\x11\x0f7\x05\x03\x15+\x03\x01\r\t\x03;\x1f\x03\x13\x05\x03\x0f#\x03\x03\x03\x07%\x05\x03\x13\x03\x05\x0f\x06%\x03\x13\x05\x03\x07\t\x03CA\x03\x13\x11\x07IG\x03\x19\x05\t\x0b\x05\x03\x0fM\x03\x07\x03\x07O\x05\x03\x01\x03\x0f\x0b\x06S\x03\x01\x07\r\x01\x11\x13\x04\x0f\x03\x13\x06\x03\x01\x05\x01\x00F\x1c\x8d\x1d\x03\x11\x0f\x0b\t\t\x0b!\x1f/!!)#\x1f\x19\x7f\x0f99A9;;m\x19\x85\x8dW\xb3K\x9bM\x9b\x96\x04\x1b+\x1b\x1f\x1f\x15\x1d\x15+\x83\x13\r\r\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11+\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00get_tuple_element_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00value\x00index\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/real\x00jit(<lambda>)/jit(main)/imag\x00jit(<lambda>)/jit(main)/neg\x00jit(<lambda>)/jit(main)/complex\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_cheevd\x00",
        xla_call_module_version=4,
    ),  # End paste

    # Pasted from the test output (see back_compat_test.py module docstring)
    c128=dict(
        testdata_version=1,
        platform='cpu',
        custom_call_targets=['lapack_zheevd'],
        serialized_date=datetime.date(2023, 3, 17),
        inputs=(),
        expected_outputs=(array([[-6.1857700048412056e-01+0.j,  2.4081403770912022e-01+0.j,
            3.5662489253627483e-01+0.j, -6.3034019033669797e-01+0.j,
            1.0043483479985752e-16+0.j, -2.8842036081919542e-02+0.j,
            7.7164692943283169e-25+0.j, -1.8446994643771725e-01+0.j],
          [-4.7070881487314609e-01+0.j,  4.7473787464450828e-01+0.j,
            -4.8036836210243361e-01+0.j,  4.3802686872516400e-01+0.j,
            1.7961797619639255e-01+0.j,  8.3080980076741355e-03+0.j,
            2.1415294457221759e-01+0.j, -2.2856669794666584e-01+0.j],
          [-3.2284062926217072e-01+0.j, -5.4336490915553370e-01+0.j,
            2.2181041859724987e-01+0.j,  2.9947877954402286e-01+0.j,
            -3.6491813600134637e-01+0.j,  3.2867679819727436e-01+0.j,
            3.8223299448843473e-01+0.j, -2.7266344945561438e-01+0.j],
          [-1.7497244365119527e-01+0.j, -8.9251550609769331e-02+0.j,
            -6.3518515114898352e-02+0.j,  1.9162997359209963e-01+0.j,
            -2.2087281326110142e-01+0.j,  5.9957027043505008e-02+0.j,
            -8.7632498908241274e-01+0.j, -3.1676020096456303e-01+0.j],
          [-2.7104258040220017e-02+0.j, -3.3772873786627688e-01+0.j,
            2.5901386593721754e-01+0.j,  1.7032650752287815e-01+0.j,
            6.7521217612940321e-01+0.j, -4.5036136532965476e-01+0.j,
            -1.2279030059078447e-02+0.j, -3.6085695247351163e-01+0.j],
          [ 1.2076392757075533e-01+0.j, -3.3834734096469249e-01+0.j,
            -6.5506827461665529e-01+0.j, -5.0472498521116760e-01+0.j,
            6.9987430903492132e-02+0.j,  1.0595648906599270e-01+0.j,
            8.3443844143082035e-02+0.j, -4.0495370398246017e-01+0.j],
          [ 2.6863211318173102e-01+0.j,  2.2958613191407312e-01+0.j,
            6.3952843755683969e-02+0.j,  1.8776775771084192e-02+0.j,
            -5.3523731432241317e-01+0.j, -5.9199531677602002e-01+0.j,
            1.7916671834524250e-01+0.j, -4.4905045549140887e-01+0.j],
          [ 4.1650029879270667e-01+0.j,  3.6355449432857068e-01+0.j,
            2.9755313100756148e-01+0.j,  1.6826270392616000e-02+0.j,
            1.9621068035557282e-01+0.j,  5.6830030587314817e-01+0.j,
            2.9607517592514260e-02+0.j, -4.9314720700035747e-01+0.j]]), array([-2.4598804776133626e+01, -4.6567755957874661e-14,
          -1.9932120610662194e-14, -5.7323356091157378e-15,
          -4.5459724251334835e-16,  4.0479851042511616e-14,
            9.2325194924982089e-14,  2.7659880477613365e+02])),
        mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<8x8xcomplex<f64>> {jax.result_info = "[0]"}, tensor<8xf64> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xcomplex<f64>>
    %1 = stablehlo.reshape %0 : (tensor<64xcomplex<f64>>) -> tensor<8x8xcomplex<f64>>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>>
    %3 = stablehlo.real %2 : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xf64>
    %4 = stablehlo.imag %2 : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xf64>
    %5 = stablehlo.negate %4 : tensor<8x8xf64>
    %6 = stablehlo.complex %3, %5 : tensor<8x8xcomplex<f64>>
    %7 = stablehlo.add %1, %6 : tensor<8x8xcomplex<f64>>
    %8 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>>
    %10 = stablehlo.divide %7, %9 : tensor<8x8xcomplex<f64>>
    %11 = call @tril(%10) : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>>
    %12 = stablehlo.constant dense<1> : tensor<i32>
    %13 = stablehlo.constant dense<1> : tensor<i32>
    %14 = stablehlo.constant dense<8> : tensor<i32>
    %15 = stablehlo.custom_call @lapack_zheevd(%12, %13, %14, %11) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<8x8xcomplex<f64>>) -> tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>
    %16 = stablehlo.get_tuple_element %15[0] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<8x8xcomplex<f64>>
    %17 = stablehlo.get_tuple_element %15[1] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<8xf64>
    %18 = stablehlo.get_tuple_element %15[2] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<i32>
    %19 = stablehlo.get_tuple_element %15[3] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<81xcomplex<f64>>
    %20 = stablehlo.get_tuple_element %15[4] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<169xf64>
    %21 = stablehlo.get_tuple_element %15[5] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<43xi32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<i32>
    %24 = stablehlo.compare  EQ, %18, %23,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %26 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1>
    %29 = stablehlo.select %28, %16, %27 : tensor<8x8xi1>, tensor<8x8xcomplex<f64>>
    %30 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %31 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<f64>) -> tensor<8xf64>
    %33 = stablehlo.broadcast_in_dim %30, dims = [0] : (tensor<1xi1>) -> tensor<8xi1>
    %34 = stablehlo.select %33, %17, %32 : tensor<8xi1>, tensor<8xf64>
    return %29, %34 : tensor<8x8xcomplex<f64>>, tensor<8xf64>
  }
  func.func private @tril(%arg0: tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %3 = stablehlo.add %0, %2 : tensor<8x8xi32>
    %4 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %6 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>>
    %8 = stablehlo.select %5, %arg0, %7 : tensor<8x8xi1>, tensor<8x8xcomplex<f64>>
    return %8 : tensor<8x8xcomplex<f64>>
  }
}
""",
        mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x015\x05\x01\x05\x01\x03\x05\x03%\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%')\x03\xc6\x02\x1e\x02?\x01\xa9\x0f\x17\x13\x0b\x17\x0b\x07\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x17\x0f\x13\x13\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x13\x13\x1b\x17\x03a\x0f/\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0bOOO\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17#\x0f\x0f\x0f\x0f\x0f\x0f\x0bOO//\x01\x07\x17\x17\x17\x03?\x17\x0f\x07\x0f\x07\x13\x07\x07\x0b\x17\x17\x07\x17\x13\x17\x17\x13\x0f\x17\x17\x13\x17#\x13\x13\x13\x0f\x17\x13\x13\x13\x02\x96\n\x1d\x83\x03\x17\x13\xf6\x04\x01\x03\x03\x15\xd3\x05+\x17\x13\xf2\x04\x01\x05-\x1f\x1d9\x03\x05/\x051\x053\x055\x057\x059\x05;\x03\x03!\xcf\x05=\x03\x03\x07\xd1\x1d?\x03\x05?\x05A\x17\x13\xea\x04\x01\x1d}\t\x03\x03\x07\xdf\x03\x03\x113\x05C\x03\x0b\x17\xad\x19\xb9\x1b\xbb\x11\xc5\x1d\xc7\x03\x0b\x17\xb1\x19\xcb\x1b\xb1\x11\xb3\x1d\xcd\x05E\x1d=\x03\x05G\x05I\x03\x03!\xd5\x1dE\x03\x05K\x03\x05'\xb5)\xd7\x1dK\x03\x05M\x03\x03\x07\xd9\x1dQ\x03\x05O\x1dU\x03\x05Q\x1dY+\x05S\x1d]+\x05U\x03\x03a\xdb\x05W\x1de\t\x05Y\x1di\t\x05[\x1dm\t\x05]\x1dq\t\x05_\x1du\t\x05a\x1dy\t\x05c\x03\x03\x07\xdd\x05e\x03\x03\x81\xb3\x05g\x05i\x03\x03\x07\xe1\x03\x11\x89\xe3\x8b\xe5\x8d\xe7\x8f\xad\x91\xe9\x93\xeb\x95\xed\x97\xf1\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x03\x03\x0b\xf3\x03\x03\x0b\xf5\x03\x03\x0b\xf7\x03\x03\x0b\xf9\x03\x03\x0b\xfb\x03\x03\x0b\xfd\x03\x05'\xb5)\xff\x03\x03\x07\x02\x02\x1f/\x01\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d{\x03\x03\xc9\x1d}\t\x07\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#%\x03\x05\xbd\xc1\r\x03\xaf\xbf\x1d\x7f\r\x03\xaf\xc3\x1d\x81\x1d\x83\x1d\x85\r\x01#'\x1d\x87\x13\r\x01\x1f\x03\t\x00\x00\x00\x00\x1f)\x01\x13\r\x05\x07\x05\x1f\x07!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x08\x00\x00\x00\x0b\x05\x1d\x89\x1d\x8b\x05\x01\x03\t\xa9\xa9\xa9\xb7\x03\x03\xef\x15\x03\x01\r\x01\x03\r\xb7\xab\xa9\xab\xab\xab\x13\x05\x01\x13\x05\x05\x13\x05\t\x13\x05\r\x13\x05\x11\x13\x05\x15\x07\x01\x1f\x07!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f#\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03\x15\x06\x02\x03\x03\x07\n\x02\x03\x03\x15\x0e\x02)\x05!!\x11)\x01\x05\x1b)\x01\x11\x0b)\x03!\t\x1d\x01\x03\t)\x05!!\x05)\x05!!\t\x13)\x05!!\x0f)\x03\t\r)\x03\x8a\x02\x11)\x03J\x05\t)\x03\xad\x05)\x01\t\x11\x01\x05\x01\x0b\x11\x03\x01\x03\x01)\x03\x01\r)\x03\x02\x02\x11/\r\x01\x0b\x03\x1d\x1f!)\x03\x01\x17)\x03\t\x17)\x03\x05\x17)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\r\x04\xda\x05\x05\x01\x11\r1\x07\x03\x01\t\r\x11\r5\x05\x03G\x91\t\x03W\x1f\x03+\x15\x06[\x03\x01\x03\x01\x17\x07c_\x03\x01\x03\x03\x19\x06g\x03\x15\x03\x05\x1b\x06k\x03\x15\x03\x05\x1d\x06o\x03\x15\x03\t\x1f\x06s\x03\x01\x05\x07\x0b\x0f\x06w\x03\x01\x05\x03\r\x05\x03\r{\x03\x07\x03\x07-\x05\x03\x01\x03\x11!\x06-\x03\x01\x05\x0f\x13#\x07\x0f\x7f\x03\x01\x03\x15\x05\x03\x01/\x03\x03\x05\x03\x01/\x03\x03\x05\x03\x01\x85\x03\x03%\x07\x01\x87\x03-\t\x19\x1b\x1d\x17\x07\x07\x01\x99\x03\x01\x03\x1f\x07\x07\x01\x9b\x03\x0b\x03\x1f\x07\x07\x01\x9d\x03\x03\x03\x1f\x07\x07\x01\x9f\x03\x1d\x03\x1f\x07\x07\x01\xa1\x03\x1f\x03\x1f\x07\x07\x01\xa3\x03!\x03\x1f\x05\x03\x01#\x03\x03\x03\x07\x01\x05\x03\x03\x03-\x11\x07\x01\xa5\x035\x05%/\x03\x07\x01\x05\x037\x031\x05\x03\x01\xa7\x03\x07\x03\x07\x01\x05\x03\x01\x035\x03\x07\x01\x12\x02\x03\x19\x033\x0b\x06\x01\x03\x01\x079!7\x03\x07\x01\x05\x039\x031\x05\x03\x01\x16\x02\x03#\x03\x07\x01\x05\x03\x0b\x03?\x03\x07\x01\x1a\x02\x03;\x03=\x0b\x06\x01\x03\x0b\x07C#A\x13\x04\r\x05;E\r\x11\x0f7\x05\x03\x15+\x03\x01\r\t\x03;\x1f\x03\x13\x05\x03\x0f#\x03\x03\x03\x07%\x05\x03\x13\x03\x05\x0f\x06%\x03\x13\x05\x03\x07\t\x03CA\x03\x13\x11\x07IG\x03\x19\x05\t\x0b\x05\x03\x0fM\x03\x07\x03\x07O\x05\x03\x01\x03\x0f\x0b\x06S\x03\x01\x07\r\x01\x11\x13\x04\x0f\x03\x13\x06\x03\x01\x05\x01\x00J\x1c\x8d\x1d\x03\x11\x0f\x0b\t\t\x0b!\x1f/!!)#\x1f\x19\x7f\x0f99A9;;m\x19\x85\x8fW\xb3K\x9bM\x9b\x96\x04\x1b+\x1b\x1f\x1f\x15\x1d\x15+\x83\x13\r\r\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11+\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00get_tuple_element_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00value\x00index\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/real\x00jit(<lambda>)/jit(main)/imag\x00jit(<lambda>)/jit(main)/neg\x00jit(<lambda>)/jit(main)/complex\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_zheevd\x00",
        xla_call_module_version=4,
    ),  # End paste
)
