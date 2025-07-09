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

# Pasted from the test output (see module docstring)
data = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['Eigh'],
    serialized_date=datetime.date(2023, 3, 16),
    inputs=(),
    expected_outputs=(array([[ 0.6185793 ,  0.00215443,  0.3897468 ,  0.43826708,  0.40065297,
      0.02844685, -0.27930862,  0.18446924],
    [ 0.47071028,  0.10003614,  0.18833847, -0.49787024, -0.61337894,
    -0.102833  , -0.21453871,  0.22856705],
    [ 0.32284275,  0.08050437, -0.678995  ,  0.11350957, -0.0658149 ,
      0.5731847 ,  0.09030591,  0.272664  ],
    [ 0.17496108, -0.7212593 , -0.01523611,  0.01533599, -0.02545237,
    -0.25565413,  0.5313786 ,  0.31675896],
    [ 0.02710732,  0.26626915, -0.16179788, -0.56394535,  0.62556016,
    -0.24231404,  0.06240111,  0.36085674],
    [-0.12076125,  0.39673248, -0.22097541,  0.47754833, -0.24771166,
    -0.5699533 ,  0.03066727,  0.40495545],
    [-0.2686366 ,  0.27486002,  0.5250418 ,  0.02719757, -0.07467701,
      0.44141003,  0.41690692,  0.44905126],
    [-0.41649577, -0.39930964, -0.02611859, -0.01004434,  0.00082341,
      0.12771341, -0.6378056 ,  0.4931458 ]], dtype=float32), array([-2.4598616e+01, -1.1325381e-03, -1.2342700e-04,  2.9237286e-05,
    5.4759425e-05,  3.0579782e-04,  5.1378174e-04,  2.7659894e+02],
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
    %8:2 = call @eigh_jacobi(%7) : (tensor<8x8xf32>) -> (tensor<8xf32>, tensor<8x8xf32>)
    return %8#1, %8#0 : tensor<8x8xf32>, tensor<8xf32>
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
  func.func private @eigh_jacobi(%arg0: tensor<8x8xf32>) -> (tensor<8xf32>, tensor<8x8xf32>) {
    %0 = call @xla_fallback_eigh_jacobi(%arg0) : (tensor<8x8xf32>) -> tuple<tensor<8xf32>, tensor<8x8xf32>>
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<8xf32>, tensor<8x8xf32>>) -> tensor<8xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<8xf32>, tensor<8x8xf32>>) -> tensor<8x8xf32>
    return %1, %2 : tensor<8xf32>, tensor<8x8xf32>
  }
  func.func private @xla_fallback_eigh_jacobi(%arg0: tensor<8x8xf32>) -> tuple<tensor<8xf32>, tensor<8x8xf32>> {
    %0 = stablehlo.custom_call @Eigh(%arg0) {backend_config = "1,1,15,0.000010", xla_shape = "(f32[8,8]{1,0}, f32[8]{0})"} : (tensor<8x8xf32>) -> tuple<tensor<8x8xf32>, tensor<8xf32>>
    %1 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<8x8xf32>, tensor<8xf32>>) -> tensor<8xf32>
    %2 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<8x8xf32>, tensor<8xf32>>) -> tensor<8x8xf32>
    %3 = stablehlo.tuple %1, %2 {xla_shape = "(f32[8]{0}, f32[8,8]{1,0})"} : tuple<tensor<8xf32>, tensor<8x8xf32>>
    return %3 : tuple<tensor<8xf32>, tensor<8x8xf32>>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01/\x05\x01\x05\x01\x03\x05\x03\x1f\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#\x03F\x02\xed'\x01\xa5\x07\x17\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x17\x0b\x13\x0b\x13\x13\x0b\x0f\x17\x0f\x13\x0b33\x0b3\x0b3S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0f\x0b\x13\x0f\x0b\x13\x0b\x0b\x13\x0f\x0b\x1b\x0b\x0b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x13\x03I\x0b\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x0f\x1f\x0f\x0f\x0b\x0b\x1fO\x1f\x03'\x17\x13\x07\x07\x17\x07\x0f\x13\x0f\x17\x17\x1b\x17\x13\x13\x17\x07\x17\x13\x02J\x07\x1f\x17\x13\x8a\x04\x01\x1d;\x03\x05%\x1d7\x03\x05'\x05)\x05+\x05-\x05/\x051\x053\x03\x03k\xdf\x17\x13\x86\x04\x01\x055\x03\x03!\xd5\x057\x03\x03!\xd7\x03\x03'\xdb\x059\x1dm\x03\x17\x13~\x04\x01\x1d\x9f\x1b\x03\x03\x071\x05;\x03\x0b\x0b\xa5\r\xb3\x0f\xb5\x07\xbf\x11\xc1\x03\x0b\x0b\xab\r\xc5\x0f\xab\x07\xad\x11\xa7\x05=\x03\x0b\x0b\xa5\r\xc7\x0f\xa5\x07\xaf\x11\xa7\x05?\x03\x0b\x0b\xa5\r\xc9\x0f\xa5\x07\xb1\x11\xa7\x03\x13A\xcbC\xcdE\xcfG\xa5I\xd1K\xa5M\xa5O\xa5\x1d\xd3\x05A\x05C\x05E\x05G\x05I\x05K\x05M\x05O\x1dS\x01\x05Q\x1dW\x01\x05S\x1d[\x01\x05U\x03\x03\x1d\xd9\x1da\x01\x05W\x03\x03\x15\xb1\x1dg\x03\x05Y\x03\x03\x17\xdd\x05[\x05]\x03\x03'\xe1\x1ds\x03\x05_\x03\x05w\xe3y\xe5\x05a\x05c\x1d}\x03\x05e\x03\x03\x17\xe7\x1d\x83\x03\x05g\x1d\x87\x03\x05i\x1d\x8b+\x05k\x1d\x8f+\x05m\x03\x03\x93\xe9\x05o\x1d\x97\x1b\x05q\x1d\x9b\x1b\x05s\x03\x03\x17\xeb\x05u\x03\x03\x15\xad\x03\x03\x15\xaf\x03\x01\x1dw\x1dy\x03\x03\xc3\x1d{\x1d}\x1d\x7f#\x13\x03\x05\xb7\xbb\r\x03\xa9\xb9\x1d\x81\r\x03\xa9\xbd\x1d\x83\x1d\x85\x1d\x87\r\x01#\x15#\x17#\x19\x0b\x03\x1d\x89\x1d\x8b\x05\x01\x1d\x8d\x13\x07\x05\x13\x07\x01\x1d\x8f\x13\x0b\x01\x1f\x11\t\x00\x00\x00\x00\x1f\x1d\x01\x13\x0b\x05\t\x07\x07\x05\x1f\r\t\x00\x00\x00\x00\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\r\t\x00\x00\x00@)\x05!!\x05)\x03!\x05\t\x1b)\x05!!\x07\x1d)\x01\x05/\x05\x03\x01)\x01\x07\x11\x01\x05\x01\x03\x11\x03\x01\x03\x01\x11\x03\x01\x05\x03\x01\x11\x03\x01\x03\x0f/\x05\x01\x03)\x03\x01\x0b)\x05!!!\x01)\x03\x02\x02\x05)\x03\t\x0b\x04\x1e\x04\x05\x01\x11\x01/\x07\x03\x01\x11\x03\x11\x013\x05\x03\x15)\x0b\x03\x89%\x03#\x1b\x06\x8d\x03\x01\x03\x01\x1d\x07\x95\x91\x03\x01\x03\x03\x11\x06\x99\x03\x01\x05\x03\x05\r\x03\x01\x9d\x03\r\x0f\x07-\x19\x03\x01\x03\t\x1f\x06-\x03\x01\x05\x07\x0b\t\x07\t\xa1\x03\x01\x03\r\t\x07\x05\xa3\x05\x03\x01\x03\x0f\x07\x04\x01\x05\x13\x11\x03\x11\t5\x05\x03\x15+\x03\x01\x01\x0b\x03e%\x03\t\r\x03\ti\x03\x11\x0f\x07)\x19\x03\t\x03\x05\x11\x06)\x03\t\x05\x03\x07\x0b\x03qo\x03\t\x17\x07{u\x03\x1f\x05\t\x0b\r\x03\t\x7f\x03\r\x0f\x07\x81\x19\x03\x01\x03\x0f\x19\x06\x85\x03\x01\x07\r\x01\x11\x07\x04\t\x03\x13\x03\x11\x059\x05\x03\t\x13\x03\x01\x01\t\x07\x05c\x03\x0f\x03\x01\x05\x07\x05#\x03\x03\x03\x03\x05\x07\x05\x1f\x03\x01\x03\x03\x07\x04\x05\x05\x05\x07\x03\x11\x01=\x05\x03\x0b\x17\x03\x01\x01\x13\x07Q?\x03\x1b\x03\x01\x05\x07U\x1f\x03\x03\x03\x03\x05\x07Y#\x03\x01\x03\x03\x15\x07_]\x03\x0f\x05\x05\x07\x07\x04\x01\x03\t\x06\x03\x01\x05\x01\x00\xbe\x1c\x9177\x0b!\x0f\x0b\t\t3\x19\x0b!\x1199m\x19\x85\x89W\xb3K+\x1b\x9bM+\x9b\x11))\x1d\x1f/!!)#\x1f\x19\x8d\x96\x04\x1b\x1f\r\x15\r\x0f\x83\x1f\x15\x1d\x15\x13\x15\x1b\x17\x15\x17\x13\x1f\x0f)\x19\x11\x11\x15+\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00get_tuple_element_v1\x00return_v1\x00call_v1\x00iota_v1\x00constant_v1\x00broadcast_in_dim_v1\x00add_v1\x00custom_call_v1\x00tuple_v1\x00compare_v1\x00select_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00callee\x00value\x00xla_shape\x00index\x00iota_dimension\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/eigh_jacobi[lower=True sort_eigenvalues=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00custom-call.2\x00get-tuple-element.4\x00get-tuple-element.3\x00tuple.5\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00broadcast_dimensions\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00compare_type\x00comparison_direction\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00private\x00jax.result_info\x00tril\x00eigh_jacobi\x00xla_fallback_eigh_jacobi\x00[0]\x00[1]\x00main\x00public\x001,1,15,0.000010\x00Eigh\x00(f32[8,8]{1,0}, f32[8]{0})\x00(f32[8]{0}, f32[8,8]{1,0})\x00",
    xla_call_module_version=4,
)  # End paste
