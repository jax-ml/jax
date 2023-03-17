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
from numpy import array, float32, complex64

# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_03_17 = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['ducc_fft'],
    serialized_date=datetime.date(2023, 3, 17),
    inputs=(array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.]], dtype=float32),),
    expected_outputs=(array([[ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j],
       [22.+0.j, -2.+2.j, -2.+0.j, -2.-2.j],
       [38.+0.j, -2.+2.j, -2.+0.j, -2.-2.j]], dtype=complex64),),
    mlir_module_text="""
module @jit_func {
  func.func public @main(%arg0: tensor<3x4xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}) -> (tensor<3x4xcomplex<f32>> {jax.result_info = ""}) {
    %0 = call @fft(%arg0) : (tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>>
    return %0 : tensor<3x4xcomplex<f32>>
  }
  func.func private @fft(%arg0: tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>> {
    %0 = stablehlo.convert %arg0 : (tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>>
    %1 = stablehlo.constant dense<"0x18000000140024000000000008000C001000140007001800140000000000000154000000380000001C00000010000000000000000000F03F0000000001000000010000000200000004000000000000000100000000000000000000000200000004000000000000000100000000000000000000000200000003000000000000000400000000000000"> : tensor<136xui8>
    %2 = stablehlo.custom_call @ducc_fft(%1, %0) {api_version = 2 : i32, operand_layouts = [dense<0> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<136xui8>, tensor<3x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
    return %2 : tensor<3x4xcomplex<f32>>
  }
}
""",
    mlir_module_serialized=b'ML\xefR\x03MLIRxxx-trunk\x00\x01\x1d\x05\x01\x05\x01\x03\x05\x03\r\x07\t\x0b\r\x0f\x11\x03\x99s\x15\x01?\x07\x0b\x0f\x17\x0b\x0b\x0b\x0b\x0f\x13\x0b33\x0b\x0b\x0f\x0b\x13\x0b\x0bK\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x035\x0b\x0b\x0f\x0b\x0bO\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0bf\x04\x0b\x0b\x0b\x13/\x0f\x03\x15\x17\x17\x07\x17\x07\x17\x0b\x07\x13\x13\x02\xca\x05\x1f\x05\x13\x1d\x1b\x07\x17\x1d^\x03\x01\x05\x15\x05\x17\x05\x19\x05\x1b\x1d\'\x07\x03\x03\x03\x15\x05\x1d\x03\x0b\tK\x0b?\rW\x03]\x0f_\x03\x0b\tC\x0b?\rC\x03E\x0fc\x05\x1f\x05!\x1d!\x07\x05#\x03\x03%e\x05%\x05\'\x03\x11+g-A/i1G3k5m7G9q\x05)\x05+\x05-\x05/\x051\x053\x055\x057\x03\x03=E\x059#\x0b\x1d;\x03\x03a\x1d=\x03\x01\x1f\x13!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03M\r\x05OQSU\x1d?\x1dA\x1dC\x1dE\x03\x03Y\r\x03[A\x1dG\x1dI\x1dK\r\x01\x1dM\x1f\x07"\x02\x18\x00\x00\x00\x14\x00$\x00\x00\x00\x00\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x07\x00\x18\x00\x14\x00\x00\x00\x00\x00\x00\x01T\x00\x00\x008\x00\x00\x00\x1c\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x05\x1dO\x05\x01\x03\x05oI\x1f\x11\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03I)\x05\r\x11\r)\x05\r\x11\x05\t)\x03B\x04\x0f\x13\x11\x03\x03\x03\x01\x03\x05!)\x03\x05\t)\x03\t\t\x04\x8f\x05\x01\x11\x01\x13\x07\x03\x01\t\x03\x11\x01\x17\x05\x03\x05\x0b\x03\x03\x01\r\x07\x05;\x03\x01\x03\x01\x05\x04\x01\x03\x03\x03\x11\x05\x19\x05\x03\t\x13\x03\x03\x01\x07\x06\x1f\x03\x01\x03\x01\t\x03\x11#\x03\x07\x0b\x07\x11)\x03\x01\x05\x05\x03\x05\x04\x05\x03\x07\x06\x03\x01\x05\x01\x00\xc2\x0eQ\x13\x11\x0f\x0b!\x1b\x1d\x05\x1b\t\x03\x0f\x1f/!!)#\x1f\x19\x91\r\xaf\x83\x82\x04\x13\x1f\x15\x1d\x15\x13\x11\x1f\x19\x17\x15\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00return_v1\x00convert_v1\x00constant_v1\x00custom_call_v1\x00call_v1\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit_func\x00jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=fft in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00jit(func)/jit(main)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False]\x00value\x00jit(func)/jit(main)/jit(fft)/fft[fft_type=FftType.FFT fft_lengths=(4,)]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00\x00fft\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00main\x00public\x00private\x00ducc_fft\x00',
    xla_call_module_version=4,
)  # End paste
