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
complex64 = np.complex64

data_2025_04_02 = {}

data_2025_04_02['c128'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgeqrf_ffi', 'lapack_zungqr_ffi'],
    serialized_date=datetime.date(2025, 4, 2),
    inputs=(),
    expected_outputs=(array([[ 0.                 +0.j,  0.9128709291752773 +0.j,
         0.40824829046386235+0.j],
       [-0.447213595499958  -0.j,  0.3651483716701102 +0.j,
        -0.8164965809277263 +0.j],
       [-0.894427190999916  -0.j, -0.1825741858350548 +0.j,
         0.40824829046386324+0.j]]), array([[-6.7082039324993694e+00+0.j, -8.0498447189992444e+00+0.j,
        -9.3914855054991175e+00+0.j],
       [ 0.0000000000000000e+00+0.j,  1.0954451150103341e+00+0.j,
         2.1908902300206665e+00+0.j],
       [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
        -8.8817841970012523e-16+0.j]])),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<3x3xcomplex<f64>> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f64>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc5)
    %2:2 = stablehlo.custom_call @lapack_zgeqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xcomplex<f64>>) -> (tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc7)
    %4 = stablehlo.custom_call @lapack_zungqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc12)
    %11 = stablehlo.select %9, %10, %2#0 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>> loc(#loc13)
    return %4, %11 : tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":411:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":411:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":412:11)
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
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.9.3\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#\'+/37\x03\xc7\x8b)\x01E\x17\x07\x0b\x0f\x0b\x1b\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03G\x0b/\x0b\x0f\x0b\x0b\x0b\x0fO\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x1fO\x0b\x13\x0b\x0b\x0b\x0f\x17/\x0b\x0f\x13\x0f\x0b\x0b\x01\x05\x0b\x0f\x03%\x17\x0b\x07\x17\x0f\x07\x0f\x07\x17\x07\x13\x13\x13\x13\x13\x13\x17\x07\x02\xd2\x04\x17\x05r\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x05\'o)q\x1d\t\x01\x1d7\x01\x03\x07\x13\x15\x17\x07\x19\x07\x05!\x11\x01\x00\x05#\x05%\x05\'\x1d\t\x1f\x17\x05n\x065\x1d#%\x05)\x17\x05n\x06\x1d\x05+\x05-\x1d-\x01\x05/\x1d1\x01\x051\x1d5\x01\x053\x055\x1d;\x01\x057\x1d?\x01\x059\x1dC\x01\x05;\x03\x01\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d=\x13\t\x01\x0b\x03\x1d?\x05\x01\x03\x03U\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x05U}\x1f#\x01#\x15\x03\x05_c\r\x03Ia\x1dA\r\x03Ie\x1dC\x1dE\x1dG\x1f\r\t\xff\xff\xff\xff\x1f\x11!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\r\x03su\x1dI\x1dK\x1dM\x03\x03{\x15\x03\x01\x01\x01\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dO\x03\x03\x83\x15\x01\x01\x01\x13\t\x05\t\x07\x07\x05\x01\t\x01\x02\x02)\x05\r\r\x07\x03\x17\x1d)\x05\r\r\x0f)\x01\x0f\x1b)\x01\x07\x13\x11\x01\x05\x05\x05\x0b)\x03%\x07)\x03\r\x07)\x03\t\x13)\x03\x05\x13)\x03\t\t)\x03\x01\t)\x05\r\r\'\x01\x04"\x02\x05\x01Q\x03\x11\x01\x07\x04\xff\x03\x01\x05\x0bP\x03\x03\x07\x04\xeb\x03\x1f=\x05B\x03\x05\x03\r\x05B\x03\x07\x03\x11\x03B\x1d\t\x03\x19\r\x06!\x03\x05\x03\x05\x07G+\x0b\x0b\x05\x05\x1b\x03\x07\x0fF/\r\x03\x05\x05\t\x03\x07G3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\tF\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03%\x05\x15\x17\tF=\x11\x03\x05\x03\x03\x15\x06A\x03\x05\x07\x19\x1b\t\x17\x04\x03\x05\x0f\x1d\x06\x03\x01\x05\x01\x00\xba\x0bQ%%\x05\x1f\x0f\x0b\x15\x15\x03!CS79Y9=3)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11)\x1f\x19\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00constant_v1\x00custom_call_v1\x00broadcast_in_dim_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x000\x00lapack_zgeqrf_ffi\x00lapack_zungqr_ffi\x00\x08[\x17\x057\x01\x0bE[]gi\x03k\x03m\x03K\x11MOwEQSyW\x07GGG\x11MO\x7fEQW\x81S\x03Y\x03\x85\x05\x87\x89',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2025_04_02['c64'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgeqrf_ffi', 'lapack_cungqr_ffi'],
    serialized_date=datetime.date(2025, 4, 2),
    inputs=(),
    expected_outputs=(array([[ 0.        +0.j,  0.91287076+0.j,  0.4082487 +0.j],
       [-0.44721356-0.j,  0.36514866+0.j, -0.8164965 +0.j],
       [-0.8944271 -0.j, -0.18257445+0.j,  0.40824816+0.j]],
      dtype=complex64), array([[-6.7082043e+00+0.j, -8.0498438e+00+0.j, -9.3914852e+00+0.j],
       [ 0.0000000e+00+0.j,  1.0954441e+00+0.j,  2.1908894e+00+0.j],
       [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  7.1525574e-07+0.j]],
      dtype=complex64)),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<3x3xcomplex<f32>> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f32>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc5)
    %2:2 = stablehlo.custom_call @lapack_cgeqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xcomplex<f32>>) -> (tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc7)
    %4 = stablehlo.custom_call @lapack_cungqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc12)
    %11 = stablehlo.select %9, %10, %2#0 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>> loc(#loc13)
    return %4, %11 : tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":411:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":411:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":412:11)
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
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.9.3\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#\'+/37\x03\xc7\x8b)\x01E\x17\x07\x0b\x0f\x0b\x1b\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03G\x0b/\x0b\x0f\x0b\x0b\x0b\x0fO\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x1f/\x0b\x13\x0b\x0b\x0b\x0f\x17/\x0b\x0f\x13\x0f\x0b\x0b\x01\x05\x0b\x0f\x03%\x17\x0b\x07\x17\x0f\x07\x0f\x07\x17\x07\x13\x13\x13\x13\x13\x13\x17\x07\x02\xb2\x04\x17\x05r\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x05\'o)q\x1d\t\x01\x1d7\x01\x03\x07\x13\x15\x17\x07\x19\x07\x05!\x11\x01\x00\x05#\x05%\x05\'\x1d\t\x1f\x17\x05n\x065\x1d#%\x05)\x17\x05n\x06\x1d\x05+\x05-\x1d-\x01\x05/\x1d1\x01\x051\x1d5\x01\x053\x055\x1d;\x01\x057\x1d?\x01\x059\x1dC\x01\x05;\x03\x01\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d=\x13\t\x01\x0b\x03\x1d?\x05\x01\x03\x03U\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x05U}\x1f#\x01#\x15\x03\x05_c\r\x03Ia\x1dA\r\x03Ie\x1dC\x1dE\x1dG\x1f\r\t\xff\xff\xff\xff\x1f\x11\x11\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\r\x03su\x1dI\x1dK\x1dM\x03\x03{\x15\x03\x01\x01\x01\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dO\x03\x03\x83\x15\x01\x01\x01\x13\t\x05\t\x07\x07\x05\x01\t\x01\x02\x02)\x05\r\r\x07\x03\x17\x1d)\x05\r\r\x0f)\x01\x0f\x1b)\x01\x07\x13\x11\x01\x05\x05\x05\t)\x03%\x07)\x03\r\x07)\x03\t\x13)\x03\x05\x13)\x03\t\t)\x03\x01\t)\x05\r\r\'\x01\x04"\x02\x05\x01Q\x03\x11\x01\x07\x04\xff\x03\x01\x05\x0bP\x03\x03\x07\x04\xeb\x03\x1f=\x05B\x03\x05\x03\r\x05B\x03\x07\x03\x11\x03B\x1d\t\x03\x19\r\x06!\x03\x05\x03\x05\x07G+\x0b\x0b\x05\x05\x1b\x03\x07\x0fF/\r\x03\x05\x05\t\x03\x07G3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\tF\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03%\x05\x15\x17\tF=\x11\x03\x05\x03\x03\x15\x06A\x03\x05\x07\x19\x1b\t\x17\x04\x03\x05\x0f\x1d\x06\x03\x01\x05\x01\x00\xba\x0bQ%%\x05\x1f\x0f\x0b\x15\x15\x03!CS79Y9=3)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11)\x1f\x19\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00constant_v1\x00custom_call_v1\x00broadcast_in_dim_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x000\x00lapack_cgeqrf_ffi\x00lapack_cungqr_ffi\x00\x08[\x17\x057\x01\x0bE[]gi\x03k\x03m\x03K\x11MOwEQSyW\x07GGG\x11MO\x7fEQW\x81S\x03Y\x03\x85\x05\x87\x89',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2025_04_02['f32'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgeqrf_ffi', 'lapack_sorgqr_ffi'],
    serialized_date=datetime.date(2025, 4, 2),
    inputs=(),
    expected_outputs=(array([[ 0.        ,  0.91287076,  0.4082487 ],
       [-0.44721356,  0.36514866, -0.8164965 ],
       [-0.8944271 , -0.18257445,  0.40824816]], dtype=float32), array([[-6.7082043e+00, -8.0498438e+00, -9.3914852e+00],
       [ 0.0000000e+00,  1.0954441e+00,  2.1908894e+00],
       [ 0.0000000e+00,  0.0000000e+00,  7.1525574e-07]], dtype=float32)),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "result[0]"}, tensor<3x3xf32> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<9xf32> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32> loc(#loc5)
    %2:2 = stablehlo.custom_call @lapack_sgeqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32> loc(#loc7)
    %4 = stablehlo.custom_call @lapack_sorgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc12)
    %11 = stablehlo.select %9, %10, %2#0 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc13)
    return %4, %11 : tensor<3x3xf32>, tensor<3x3xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":411:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":411:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":412:11)
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
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.9.3\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#\'+/37\x03\xc5\x8b\'\x01E\x17\x07\x0b\x0f\x0b\x1b\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03G\x0b/\x0b\x0f\x0b\x0b\x0b\x0fO\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x0b\x13\x0b\x0b\x0b\x0f\x17/\x0b\x0f\x13\x0f\x0b\x0b\x01\x05\x0b\x0f\x03#\x17\x07\x07\x17\x0f\x07\x0f\x07\x17\x13\x13\x13\x13\x13\x13\x17\x07\x02\x9a\x04\x17\x05r\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x05\'o)q\x1d\t\x01\x1d7\x01\x03\x07\x13\x15\x17\x07\x19\x07\x05!\x11\x01\x00\x05#\x05%\x05\'\x1d\t\x1f\x17\x05n\x065\x1d#%\x05)\x17\x05n\x06\x1d\x05+\x05-\x1d-\x01\x05/\x1d1\x01\x051\x1d5\x01\x053\x055\x1d;\x01\x057\x1d?\x01\x059\x1dC\x01\x05;\x03\x01\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d=\x13\t\x01\x0b\x03\x1d?\x05\x01\x03\x03U\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x05U}\x1f!\x01#\x15\x03\x05_c\r\x03Ia\x1dA\r\x03Ie\x1dC\x1dE\x1dG\x1f\r\t\xff\xff\xff\xff\x1f\x11\t\x00\x00\x00\x00\r\x01\r\x03su\x1dI\x1dK\x1dM\x03\x03{\x15\x03\x01\x01\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dO\x03\x03\x83\x15\x01\x01\x01\x13\t\x05\t\x07\x07\x05\x01\t\x01\x02\x02)\x05\r\r\x07\t\x1d)\x05\r\r\x0f)\x01\x0f\x1b)\x01\x07\x13\x11\x01\x05\x05\x05)\x03%\x07)\x03\r\x07)\x03\t\x13)\x03\x05\x13)\x03\t\t)\x03\x01\t)\x05\r\r%\x01\x04"\x02\x05\x01Q\x03\x11\x01\x07\x04\xff\x03\x01\x05\x0bP\x03\x03\x07\x04\xeb\x03\x1f=\x05B\x03\x05\x03\r\x05B\x03\x07\x03\x11\x03B\x1d\t\x03\x17\r\x06!\x03\x05\x03\x05\x07G+\x0b\x0b\x05\x05\x19\x03\x07\x0fF/\r\x03\x05\x05\t\x03\x07G3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\tF\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03#\x05\x15\x17\tF=\x11\x03\x05\x03\x03\x15\x06A\x03\x05\x07\x19\x1b\t\x17\x04\x03\x05\x0f\x1d\x06\x03\x01\x05\x01\x00\xba\x0bQ%%\x05\x1f\x0f\x0b\x15\x15\x03!CS79Y9=3)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11)\x1f\x19\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00constant_v1\x00custom_call_v1\x00broadcast_in_dim_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x000\x00lapack_sgeqrf_ffi\x00lapack_sorgqr_ffi\x00\x08[\x17\x057\x01\x0bE[]gi\x03k\x03m\x03K\x11MOwEQSyW\x07GGG\x11MO\x7fEQW\x81S\x03Y\x03\x85\x05\x87\x89',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2025_04_02['f64'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgeqrf_ffi', 'lapack_dorgqr_ffi'],
    serialized_date=datetime.date(2025, 4, 2),
    inputs=(),
    expected_outputs=(array([[ 0.                 ,  0.9128709291752773 ,  0.40824829046386235],
       [-0.447213595499958  ,  0.3651483716701102 , -0.8164965809277263 ],
       [-0.894427190999916  , -0.1825741858350548 ,  0.40824829046386324]]), array([[-6.7082039324993694e+00, -8.0498447189992444e+00,
        -9.3914855054991175e+00],
       [ 0.0000000000000000e+00,  1.0954451150103341e+00,
         2.1908902300206665e+00],
       [ 0.0000000000000000e+00,  0.0000000000000000e+00,
        -8.8817841970012523e-16]])),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf64> {jax.result_info = "result[0]"}, tensor<3x3xf64> {jax.result_info = "result[1]"}) {
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<9xf64> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xf64>) -> tensor<3x3xf64> loc(#loc5)
    %2:2 = stablehlo.custom_call @lapack_dgeqrf_ffi(%1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>) loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf64>, tensor<f64>) -> tensor<3x3xf64> loc(#loc7)
    %4 = stablehlo.custom_call @lapack_dorgqr_ffi(%3, %2#1) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3x3xf64> loc(#loc8)
    %5 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc10)
    %7 = stablehlo.add %5, %6 : tensor<3x3xi32> loc(#loc10)
    %8 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc9)
    %9 = stablehlo.compare  GE, %7, %8,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc11)
    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64> loc(#loc12)
    %11 = stablehlo.select %9, %10, %2#0 : tensor<3x3xi1>, tensor<3x3xf64> loc(#loc13)
    return %4, %11 : tensor<3x3xf64>, tensor<3x3xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":411:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":411:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":412:11)
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
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.9.3\x00\x01)\x05\x01\x05\x19\x01\x03\x0b\x03\x17\x0f\x13\x17\x1b\x1f#\'+/37\x03\xc5\x8b\'\x01E\x17\x07\x0b\x0f\x0b\x1b\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x17\x0f\x0b\x17\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x03G\x0b/\x0b\x0f\x0b\x0b\x0b\x0fO\x13\x0f\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x1f/\x0b\x13\x0b\x0b\x0b\x0f\x17/\x0b\x0f\x13\x0f\x0b\x0b\x01\x05\x0b\x0f\x03#\x17\x07\x07\x17\x0f\x07\x0f\x07\x17\x13\x13\x13\x13\x13\x13\x17\x07\x02\xaa\x04\x17\x05r\x06\x17\x1f\x05\x1d\x11\x03\x05\x05\x1f\x03\x05\'o)q\x1d\t\x01\x1d7\x01\x03\x07\x13\x15\x17\x07\x19\x07\x05!\x11\x01\x00\x05#\x05%\x05\'\x1d\t\x1f\x17\x05n\x065\x1d#%\x05)\x17\x05n\x06\x1d\x05+\x05-\x1d-\x01\x05/\x1d1\x01\x051\x1d5\x01\x053\x055\x1d;\x01\x057\x1d?\x01\x059\x1dC\x01\x05;\x03\x01\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d=\x13\t\x01\x0b\x03\x1d?\x05\x01\x03\x03U\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x05U}\x1f!\x01#\x15\x03\x05_c\r\x03Ia\x1dA\r\x03Ie\x1dC\x1dE\x1dG\x1f\r\t\xff\xff\xff\xff\x1f\x11\x11\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\r\x03su\x1dI\x1dK\x1dM\x03\x03{\x15\x03\x01\x01\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dO\x03\x03\x83\x15\x01\x01\x01\x13\t\x05\t\x07\x07\x05\x01\t\x01\x02\x02)\x05\r\r\x07\x0b\x1d)\x05\r\r\x0f)\x01\x0f\x1b)\x01\x07\x13\x11\x01\x05\x05\x05)\x03%\x07)\x03\r\x07)\x03\t\x13)\x03\x05\x13)\x03\t\t)\x03\x01\t)\x05\r\r%\x01\x04"\x02\x05\x01Q\x03\x11\x01\x07\x04\xff\x03\x01\x05\x0bP\x03\x03\x07\x04\xeb\x03\x1f=\x05B\x03\x05\x03\r\x05B\x03\x07\x03\x11\x03B\x1d\t\x03\x17\r\x06!\x03\x05\x03\x05\x07G+\x0b\x0b\x05\x05\x19\x03\x07\x0fF/\r\x03\x05\x05\t\x03\x07G3\x0b\x0f\x03\x05\x05\r\x0b\x03B\r\t\x03\x0b\tF\x0f\x11\x03\x0b\x03\x01\x11\x06\x0f\x03\x0b\x05\x11\x13\x03B\r\x13\x03\x0b\x13F9\x15\x03#\x05\x15\x17\tF=\x11\x03\x05\x03\x03\x15\x06A\x03\x05\x07\x19\x1b\t\x17\x04\x03\x05\x0f\x1d\x06\x03\x01\x05\x01\x00\xba\x0bQ%%\x05\x1f\x0f\x0b\x15\x15\x03!CS79Y9=3)A\x1b%)9;i\x15\x15\x17\x0f\x0f\x17\x11)\x1f\x19\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00iota_v1\x00constant_v1\x00custom_call_v1\x00broadcast_in_dim_v1\x00func_v1\x00reshape_v1\x00pad_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(<lambda>)/jit(main)/iota\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/reshape\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/pad\x00jit(<lambda>)/jit(main)/householder_product\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/ge\x00jit(<lambda>)/jit(main)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/select_n\x00jax.result_info\x00\x00result[0]\x00result[1]\x00main\x00public\x00num_batch_dims\x000\x00lapack_dgeqrf_ffi\x00lapack_dorgqr_ffi\x00\x08[\x17\x057\x01\x0bE[]gi\x03k\x03m\x03K\x11MOwEQSyW\x07GGG\x11MO\x7fEQW\x81S\x03Y\x03\x85\x05\x87\x89',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
