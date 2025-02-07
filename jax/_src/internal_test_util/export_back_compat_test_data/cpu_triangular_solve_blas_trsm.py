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
from numpy import array, float32, complex64

data_2023_07_16 = {}
# Pasted from the test output (see back_compat_test_util.py module docstring)
data_2023_07_16["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['blas_strsm'],
    serialized_date=datetime.date(2023, 7, 16),
    inputs=(array([[ 5.,  0.,  0.,  0.],
       [ 4., 10.,  0.,  0.],
       [ 8.,  9., 15.,  0.],
       [12., 13., 14., 20.]], dtype=float32), array([[ 0.,  1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.,  9.],
       [10., 11., 12., 13., 14.],
       [15., 16., 17., 18., 19.]], dtype=float32)),
    expected_outputs=(array([[ 0.         ,  0.2        ,  0.4        ,  0.6        ,
         0.8        ],
       [ 0.5        ,  0.52       ,  0.54       ,  0.56       ,
         0.58000004 ],
       [ 0.36666667 ,  0.31466666 ,  0.26266667 ,  0.21066667 ,
         0.15866666 ],
       [ 0.16833334 ,  0.12173338 ,  0.0751333  ,  0.02853328 ,
        -0.018066704]], dtype=float32),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf32> {jax.arg_info = "a", mhlo.sharding = "{replicated}"} loc(unknown), %arg1: tensor<4x5xf32> {jax.arg_info = "b", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<4x5xf32> {jax.result_info = ""}) {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(#loc2)
    %1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %2 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %3 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %5 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %6 = stablehlo.constant dense<5> : tensor<i32> loc(#loc2)
    %7 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %8 = stablehlo.custom_call @blas_strsm(%1, %2, %3, %4, %5, %6, %7, %0, %arg0, %arg1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<4x4xf32>, tensor<4x5xf32>) -> tensor<4x5xf32> loc(#loc2)
    return %8 : tensor<4x5xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":508:0)
#loc2 = loc("jit(func)/jit(main)/triangular_solve[left_side=True lower=True transpose_a=False conjugate_a=False unit_diagonal=False]"(#loc1))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x19\x05\x01\x03\x01\x03\x05\x03\t\x07\t\x0b\r\x03\xa5{\x17\x01?\x0f\x07\x0b\x13\x0f\x0b\x13#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x17\x0b\x13\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03=\x0fO\x0b\x0b\x0b\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x1f\x1f\x1f\x1f\x1f\x0b\x0b\x0b\x0b3\x0f\x13\x0f\x01\x03\x0f\x03\x15\x0f\x17\x07\x17\x0f\x07\x1b\x07\x13\x13\x02J\x04\x1d#%\x1f\x05\x0f\x03\x03\x05c\x11\x01\x05\x05\x11\x03\x03\x05e\x03\x07\x11\t\x13\t\x0b\x15\x05\x13\x05\x15\x05\x17\x03\x0b\x19K\x1bU\x1dW\x0b]\x1f_\x05\x19\x05\x1b\x05\x1d\x05\x1f\x03\x03\x05a\x05!\x17'\xf2\x07\x01\x05#\x03\x03\x05g\x03\x03\x05i\x03\x11/k1I3m5o7q9s;u=y\x05%\x05'\x05)\x05+\x05-\x05/\x051\x053\x1f\x13\x01\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1d5\x1d7\x1d9\x1d;\x03\x05MQ\r\x05COEG\x1d=\r\x05CSEG\x1d?#\x0f\x03\x03Y\r\x03[I\x1dA\x1dC\x1dE\x1f\x0b\t\x00\x00\x80?\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x00\x00\x00\x00\x1f\x03\t\x04\x00\x00\x00\x1f\x03\t\x05\x00\x00\x00\x0b\x05\x1dG\x03\x01\x05\x01\x03\x15????????AA\x03\x03w\x15\x01%\x01\x03\x03A\x01\x02\x02)\x01\x11)\x05\x11\x15\x07\t)\x05\x11\x11\x07)\x01\x07\x13\x11\x05\t\x05\x03\x05\x1b)\x03\x01\r)\x03\t\r\x04\xb9\x05\x01\x11\x03\x0f\x07\x03\x01\x05\x05\x11\x03\x17\x05\x03\x17+\x05\t\x03\x05\x03\x03\x03\x01!\x03\x0b\x03\x03\x01\x07\x03\x03\x03\x03\x01\x07\x03\x03\x03\x03\x01\r\x03\x03\x03\x03\x01\r\x03\x03\x03\x03\x01)\x03\x03\x03\x03\x01+\x03\x03\x03\x03\x01\x07\x03\x03\x07\x07\x01-\x03\x05\x15\x07\t\x0b\r\x0f\x11\x13\x05\x01\x03\t\x04\x03\x03\x15\x06\x03\x01\x05\x01\x00\xca\tI\x17\x0f\x0b!\x05\x05\x03\x1b\x1d\x1b\x1f/!!)#\x1f\x19\x97\xf1\x1f\x15\x1d\x15\x13%)\x13\r\x15\x1f\x11\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00func_v1\x00custom_call_v1\x00return_v1\x00value\x00sym_name\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(func)/jit(main)/triangular_solve[left_side=True lower=True transpose_a=False conjugate_a=False unit_diagonal=False]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.arg_info\x00mhlo.sharding\x00{replicated}\x00\x00a\x00b\x00jax.result_info\x00main\x00public\x00blas_strsm\x00",
    xla_call_module_version=6,
)  # End paste

# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_07_16["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['blas_dtrsm'],
    serialized_date=datetime.date(2023, 7, 16),
    inputs=(array([[ 5.,  0.,  0.,  0.],
       [ 4., 10.,  0.,  0.],
       [ 8.,  9., 15.,  0.],
       [12., 13., 14., 20.]]), array([[ 0.,  1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.,  9.],
       [10., 11., 12., 13., 14.],
       [15., 16., 17., 18., 19.]])),
    expected_outputs=(array([[ 0.                  ,  0.2                 ,
         0.4                 ,  0.6000000000000001  ,
         0.8                 ],
       [ 0.5                 ,  0.52                ,
         0.54                ,  0.5599999999999999  ,
         0.58                ],
       [ 0.36666666666666664 ,  0.3146666666666667  ,
         0.2626666666666667  ,  0.21066666666666667 ,
         0.15866666666666665 ],
       [ 0.16833333333333336 ,  0.1217333333333333  ,
         0.07513333333333323 ,  0.0285333333333333  ,
        -0.018066666666666675]]),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf64> {jax.arg_info = "a", mhlo.sharding = "{replicated}"} loc(unknown), %arg1: tensor<4x5xf64> {jax.arg_info = "b", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<4x5xf64> {jax.result_info = ""}) {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f64> loc(#loc2)
    %1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %2 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %3 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %5 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %6 = stablehlo.constant dense<5> : tensor<i32> loc(#loc2)
    %7 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %8 = stablehlo.custom_call @blas_dtrsm(%1, %2, %3, %4, %5, %6, %7, %0, %arg0, %arg1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<f64>, tensor<4x4xf64>, tensor<4x5xf64>) -> tensor<4x5xf64> loc(#loc2)
    return %8 : tensor<4x5xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":511:0)
#loc2 = loc("jit(func)/jit(main)/triangular_solve[left_side=True lower=True transpose_a=False conjugate_a=False unit_diagonal=False]"(#loc1))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x19\x05\x01\x03\x01\x03\x05\x03\t\x07\t\x0b\r\x03\xa5{\x17\x01?\x0f\x07\x0b\x13\x0f\x0b\x13#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x17\x0b\x13\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03=\x0fO\x0b\x0b\x0b\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0f\x13\x0b\x0b\x0b/\x1f\x1f\x1f\x1f\x0b\x0b\x0b\x0b3\x0f\x13\x0f\x01\x03\x0f\x03\x15\x0f\x17\x07\x17\x0f\x07\x1b\x07\x13\x13\x02Z\x04\x1d#%\x1f\x05\x0f\x03\x03\x05c\x11\x01\x05\x05\x11\x03\x03\x05e\x03\x07\x11\t\x13\t\x0b\x15\x05\x13\x05\x15\x05\x17\x03\x0b\x19K\x1bU\x1dW\x0b]\x1f_\x05\x19\x05\x1b\x05\x1d\x05\x1f\x03\x03\x05a\x05!\x17'\xfe\x07\x01\x05#\x03\x03\x05g\x03\x03\x05i\x03\x11/k1I3m5o7q9s;u=y\x05%\x05'\x05)\x05+\x05-\x05/\x051\x053\x1f\x13\x01\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1d5\x1d7\x1d9\x1d;\x03\x05MQ\r\x05COEG\x1d=\r\x05CSEG\x1d?#\x0f\x03\x03Y\r\x03[I\x1dA\x1dC\x1dE\x1f\x0b\x11\x00\x00\x00\x00\x00\x00\xf0?\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x00\x00\x00\x00\x1f\x03\t\x04\x00\x00\x00\x1f\x03\t\x05\x00\x00\x00\x0b\x05\x1dG\x03\x01\x05\x01\x03\x15????????AA\x03\x03w\x15\x01%\x01\x03\x03A\x01\x02\x02)\x01\x11)\x05\x11\x15\x07\x0b)\x05\x11\x11\x07)\x01\x07\x13\x11\x05\t\x05\x03\x05\x1b)\x03\x01\r)\x03\t\r\x04\xb9\x05\x01\x11\x03\x0f\x07\x03\x01\x05\x05\x11\x03\x17\x05\x03\x17+\x05\t\x03\x05\x03\x03\x03\x01!\x03\x0b\x03\x03\x01\x07\x03\x03\x03\x03\x01\x07\x03\x03\x03\x03\x01\r\x03\x03\x03\x03\x01\r\x03\x03\x03\x03\x01)\x03\x03\x03\x03\x01+\x03\x03\x03\x03\x01\x07\x03\x03\x07\x07\x01-\x03\x05\x15\x07\t\x0b\r\x0f\x11\x13\x05\x01\x03\t\x04\x03\x03\x15\x06\x03\x01\x05\x01\x00\xca\tI\x17\x0f\x0b!\x05\x05\x03\x1b\x1d\x1b\x1f/!!)#\x1f\x19\x97\xf1\x1f\x15\x1d\x15\x13%)\x13\r\x15\x1f\x11\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00func_v1\x00custom_call_v1\x00return_v1\x00value\x00sym_name\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(func)/jit(main)/triangular_solve[left_side=True lower=True transpose_a=False conjugate_a=False unit_diagonal=False]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.arg_info\x00mhlo.sharding\x00{replicated}\x00\x00a\x00b\x00jax.result_info\x00main\x00public\x00blas_dtrsm\x00",
    xla_call_module_version=6,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_07_16["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['blas_ctrsm'],
    serialized_date=datetime.date(2023, 7, 16),
    inputs=(array([[ 5.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
       [ 4.+0.j, 10.+0.j,  0.+0.j,  0.+0.j],
       [ 8.+0.j,  9.+0.j, 15.+0.j,  0.+0.j],
       [12.+0.j, 13.+0.j, 14.+0.j, 20.+0.j]], dtype=complex64), array([[ 0.+0.j,  1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j],
       [ 5.+0.j,  6.+0.j,  7.+0.j,  8.+0.j,  9.+0.j],
       [10.+0.j, 11.+0.j, 12.+0.j, 13.+0.j, 14.+0.j],
       [15.+0.j, 16.+0.j, 17.+0.j, 18.+0.j, 19.+0.j]], dtype=complex64)),
    expected_outputs=(array([[ 0.         +0.j,  0.2        +0.j,  0.4        +0.j,
         0.6        +0.j,  0.8        +0.j],
       [ 0.5        +0.j,  0.52       +0.j,  0.54       +0.j,
         0.56       +0.j,  0.58000004 +0.j],
       [ 0.36666667 +0.j,  0.31466666 +0.j,  0.26266667 +0.j,
         0.21066667 +0.j,  0.15866666 +0.j],
       [ 0.16833334 +0.j,  0.12173338 +0.j,  0.0751333  +0.j,
         0.02853328 +0.j, -0.018066704+0.j]], dtype=complex64),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f32>> {jax.arg_info = "a", mhlo.sharding = "{replicated}"} loc(unknown), %arg1: tensor<4x5xcomplex<f32>> {jax.arg_info = "b", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<4x5xcomplex<f32>> {jax.result_info = ""}) {
    %0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc2)
    %1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %2 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %3 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %5 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %6 = stablehlo.constant dense<5> : tensor<i32> loc(#loc2)
    %7 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %8 = stablehlo.custom_call @blas_ctrsm(%1, %2, %3, %4, %5, %6, %7, %0, %arg0, %arg1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x5xcomplex<f32>>) -> tensor<4x5xcomplex<f32>> loc(#loc2)
    return %8 : tensor<4x5xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":510:0)
#loc2 = loc("jit(func)/jit(main)/triangular_solve[left_side=True lower=True transpose_a=False conjugate_a=False unit_diagonal=False]"(#loc1))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x19\x05\x01\x03\x01\x03\x05\x03\t\x07\t\x0b\r\x03\xa7{\x19\x01?\x0f\x07\x0b\x13\x0f\x0b\x13#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x17\x0b\x13\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03=\x0fO\x0b\x0b\x0b\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0f\x13\x0b\x0b\x0b/\x1f\x1f\x1f\x1f\x0b\x0b\x0b\x0b3\x0f\x13\x0f\x01\x03\x0f\x03\x17\x0f\x17\x0b\x17\x0f\x07\x1b\x07\x07\x13\x13\x02b\x04\x1d#%\x1f\x05\x0f\x03\x03\x05c\x11\x01\x05\x05\x11\x03\x03\x05e\x03\x07\x11\t\x13\t\x0b\x15\x05\x13\x05\x15\x05\x17\x03\x0b\x19K\x1bU\x1dW\x0b]\x1f_\x05\x19\x05\x1b\x05\x1d\x05\x1f\x03\x03\x05a\x05!\x17'\xfa\x07\x01\x05#\x03\x03\x05g\x03\x03\x05i\x03\x11/k1I3m5o7q9s;u=y\x05%\x05'\x05)\x05+\x05-\x05/\x051\x053\x1f\x15\x01\x1f\x17!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1d5\x1d7\x1d9\x1d;\x03\x05MQ\r\x05COEG\x1d=\r\x05CSEG\x1d?#\x0f\x03\x03Y\r\x03[I\x1dA\x1dC\x1dE\x1f\x0b\x11\x00\x00\x80?\x00\x00\x00\x00\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x00\x00\x00\x00\x1f\x03\t\x04\x00\x00\x00\x1f\x03\t\x05\x00\x00\x00\x0b\x05\x1dG\x03\x01\x05\x01\x03\x15????????AA\x03\x03w\x15\x01%\x01\x03\x03A\x01\x02\x02)\x01\x13)\x05\x11\x15\x07\x03\x11)\x05\x11\x11\x07)\x01\x07\x13\x11\x05\t\x05\x03\x05\t\x1b)\x03\x01\r)\x03\t\r\x04\xb9\x05\x01\x11\x03\x0f\x07\x03\x01\x05\x05\x11\x03\x17\x05\x03\x17+\x05\t\x03\x05\x03\x03\x03\x01!\x03\x0b\x03\x03\x01\x07\x03\x03\x03\x03\x01\x07\x03\x03\x03\x03\x01\r\x03\x03\x03\x03\x01\r\x03\x03\x03\x03\x01)\x03\x03\x03\x03\x01+\x03\x03\x03\x03\x01\x07\x03\x03\x07\x07\x01-\x03\x05\x15\x07\t\x0b\r\x0f\x11\x13\x05\x01\x03\t\x04\x03\x03\x15\x06\x03\x01\x05\x01\x00\xca\tI\x17\x0f\x0b!\x05\x05\x03\x1b\x1d\x1b\x1f/!!)#\x1f\x19\x97\xf1\x1f\x15\x1d\x15\x13%)\x13\r\x15\x1f\x11\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00func_v1\x00custom_call_v1\x00return_v1\x00value\x00sym_name\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(func)/jit(main)/triangular_solve[left_side=True lower=True transpose_a=False conjugate_a=False unit_diagonal=False]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.arg_info\x00mhlo.sharding\x00{replicated}\x00\x00a\x00b\x00jax.result_info\x00main\x00public\x00blas_ctrsm\x00",
    xla_call_module_version=6,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_07_16["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['blas_ztrsm'],
    serialized_date=datetime.date(2023, 7, 16),
    inputs=(array([[ 5.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
       [ 4.+0.j, 10.+0.j,  0.+0.j,  0.+0.j],
       [ 8.+0.j,  9.+0.j, 15.+0.j,  0.+0.j],
       [12.+0.j, 13.+0.j, 14.+0.j, 20.+0.j]]), array([[ 0.+0.j,  1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j],
       [ 5.+0.j,  6.+0.j,  7.+0.j,  8.+0.j,  9.+0.j],
       [10.+0.j, 11.+0.j, 12.+0.j, 13.+0.j, 14.+0.j],
       [15.+0.j, 16.+0.j, 17.+0.j, 18.+0.j, 19.+0.j]])),
    expected_outputs=(array([[ 0.                  +0.j,  0.2                 +0.j,
         0.4                 +0.j,  0.6000000000000001  +0.j,
         0.8                 +0.j],
       [ 0.5                 +0.j,  0.52                +0.j,
         0.54                +0.j,  0.5599999999999999  +0.j,
         0.58                +0.j],
       [ 0.36666666666666664 +0.j,  0.3146666666666667  +0.j,
         0.2626666666666667  +0.j,  0.21066666666666667 +0.j,
         0.15866666666666665 +0.j],
       [ 0.16833333333333336 +0.j,  0.1217333333333333  +0.j,
         0.07513333333333323 +0.j,  0.0285333333333333  +0.j,
        -0.018066666666666675+0.j]]),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f64>> {jax.arg_info = "a", mhlo.sharding = "{replicated}"} loc(unknown), %arg1: tensor<4x5xcomplex<f64>> {jax.arg_info = "b", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<4x5xcomplex<f64>> {jax.result_info = ""}) {
    %0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc2)
    %1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %2 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %3 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %5 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %6 = stablehlo.constant dense<5> : tensor<i32> loc(#loc2)
    %7 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %8 = stablehlo.custom_call @blas_ztrsm(%1, %2, %3, %4, %5, %6, %7, %0, %arg0, %arg1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<complex<f64>>, tensor<4x4xcomplex<f64>>, tensor<4x5xcomplex<f64>>) -> tensor<4x5xcomplex<f64>> loc(#loc2)
    return %8 : tensor<4x5xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":510:0)
#loc2 = loc("jit(func)/jit(main)/triangular_solve[left_side=True lower=True transpose_a=False conjugate_a=False unit_diagonal=False]"(#loc1))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x19\x05\x01\x03\x01\x03\x05\x03\t\x07\t\x0b\r\x03\xa7{\x19\x01?\x0f\x07\x0b\x13\x0f\x0b\x13#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x17\x0b\x13\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03=\x0fO\x0b\x0b\x0b\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0f\x13\x0b\x0b\x0bO\x1f\x1f\x1f\x1f\x0b\x0b\x0b\x0b3\x0f\x13\x0f\x01\x03\x0f\x03\x17\x0f\x17\x0b\x17\x0f\x07\x1b\x07\x07\x13\x13\x02\x82\x04\x1d#%\x1f\x05\x0f\x03\x03\x05c\x11\x01\x05\x05\x11\x03\x03\x05e\x03\x07\x11\t\x13\t\x0b\x15\x05\x13\x05\x15\x05\x17\x03\x0b\x19K\x1bU\x1dW\x0b]\x1f_\x05\x19\x05\x1b\x05\x1d\x05\x1f\x03\x03\x05a\x05!\x17'\xfa\x07\x01\x05#\x03\x03\x05g\x03\x03\x05i\x03\x11/k1I3m5o7q9s;u=y\x05%\x05'\x05)\x05+\x05-\x05/\x051\x053\x1f\x15\x01\x1f\x17!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1d5\x1d7\x1d9\x1d;\x03\x05MQ\r\x05COEG\x1d=\r\x05CSEG\x1d?#\x0f\x03\x03Y\r\x03[I\x1dA\x1dC\x1dE\x1f\x0b!\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x00\x00\x00\x00\x1f\x03\t\x04\x00\x00\x00\x1f\x03\t\x05\x00\x00\x00\x0b\x05\x1dG\x03\x01\x05\x01\x03\x15????????AA\x03\x03w\x15\x01%\x01\x03\x03A\x01\x02\x02)\x01\x13)\x05\x11\x15\x07\x03\x11)\x05\x11\x11\x07)\x01\x07\x13\x11\x05\t\x05\x03\x05\x0b\x1b)\x03\x01\r)\x03\t\r\x04\xb9\x05\x01\x11\x03\x0f\x07\x03\x01\x05\x05\x11\x03\x17\x05\x03\x17+\x05\t\x03\x05\x03\x03\x03\x01!\x03\x0b\x03\x03\x01\x07\x03\x03\x03\x03\x01\x07\x03\x03\x03\x03\x01\r\x03\x03\x03\x03\x01\r\x03\x03\x03\x03\x01)\x03\x03\x03\x03\x01+\x03\x03\x03\x03\x01\x07\x03\x03\x07\x07\x01-\x03\x05\x15\x07\t\x0b\r\x0f\x11\x13\x05\x01\x03\t\x04\x03\x03\x15\x06\x03\x01\x05\x01\x00\xca\tI\x17\x0f\x0b!\x05\x05\x03\x1b\x1d\x1b\x1f/!!)#\x1f\x19\x97\xf1\x1f\x15\x1d\x15\x13%)\x13\r\x15\x1f\x11\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00func_v1\x00custom_call_v1\x00return_v1\x00value\x00sym_name\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(func)/jit(main)/triangular_solve[left_side=True lower=True transpose_a=False conjugate_a=False unit_diagonal=False]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.arg_info\x00mhlo.sharding\x00{replicated}\x00\x00a\x00b\x00jax.result_info\x00main\x00public\x00blas_ztrsm\x00",
    xla_call_module_version=6,
)  # End paste

data_2024_12_02 = {}


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_12_02['c128'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_ztrsm_ffi'],
    serialized_date=datetime.date(2024, 12, 2),
    inputs=(
        array([
            [5.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [4.0 + 0.0j, 10.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [8.0 + 0.0j, 9.0 + 0.0j, 15.0 + 0.0j, 0.0 + 0.0j],
            [12.0 + 0.0j, 13.0 + 0.0j, 14.0 + 0.0j, 20.0 + 0.0j],
        ]),
        array([
            [0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j],
            [5.0 + 0.0j, 6.0 + 0.0j, 7.0 + 0.0j, 8.0 + 0.0j, 9.0 + 0.0j],
            [10.0 + 0.0j, 11.0 + 0.0j, 12.0 + 0.0j, 13.0 + 0.0j, 14.0 + 0.0j],
            [15.0 + 0.0j, 16.0 + 0.0j, 17.0 + 0.0j, 18.0 + 0.0j, 19.0 + 0.0j],
        ]),
    ),
    expected_outputs=(
        array([
            [
                0.0 + 0.0j,
                0.2 + 0.0j,
                0.4 + 0.0j,
                0.6000000000000001 + 0.0j,
                0.8 + 0.0j,
            ],
            [
                0.5 + 0.0j,
                0.52 + 0.0j,
                0.54 + 0.0j,
                0.5599999999999999 + 0.0j,
                0.58 + 0.0j,
            ],
            [
                0.36666666666666664 + 0.0j,
                0.3146666666666667 + 0.0j,
                0.2626666666666667 + 0.0j,
                0.21066666666666667 + 0.0j,
                0.15866666666666665 + 0.0j,
            ],
            [
                0.16833333333333336 + 0.0j,
                0.1217333333333333 + 0.0j,
                0.07513333333333323 + 0.0j,
                0.0285333333333333 + 0.0j,
                -0.018066666666666675 + 0.0j,
            ],
        ]),
    ),
    mlir_module_text=r"""
#loc1 = loc("a")
#loc2 = loc("b")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f64>> loc("a"), %arg1: tensor<4x5xcomplex<f64>> loc("b")) -> (tensor<4x5xcomplex<f64>> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc4)
    %0 = stablehlo.custom_call @lapack_ztrsm_ffi(%arg0, %arg1, %cst) {mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x4xcomplex<f64>>, tensor<4x5xcomplex<f64>>, tensor<complex<f64>>) -> tensor<4x5xcomplex<f64>> loc(#loc4)
    return %0 : tensor<4x5xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":715:13)
#loc4 = loc("jit(func)/jit(main)/triangular_solve"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.1\x00\x01\x1b\x05\x01\x05\x0b\x01\x03\x0b\x03\t\x0f\x13\x17\x1b\x03\x87[\x19\x01%\x07\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x17\x0b\x13\x0b\x037O\x0b\x0b\x0f\x0f\x13\x0b\x0f\x13\x0b\x0b\x0bO+\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0f\x0f\x13\x0f\x01\x05\x0b\x0f\x03\x15\x17\x0b\x17\x0f\x07\x07\x1b\x07\x13\x13\x02\x1e\x03\x1f\x11\x03\x05\x1d\x1b\x1d\x03\x07\t\x0b\r\x03\x0f\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x15\x01\x05\x17\x1d\x19\x01\x05\x19\x05\x1b\x17\x1f.\x0b\x1b\x05\x1d\x03\x03#?\x05\x1f\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x01\x1d!\x13\rN\x13\rL\x03\x05''#\x11\x03\x035\r\x037)\x1d#\x1d%\x1d'\x1f\x0b!\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x00\r\tA+C-E+G-\x1d)\x1d+\x1d-\x1d/\x0b\x03\x1d1\x03\x01\x05\x01\x03\x07%%S\x1f\x17\x01\x03\x03W\x15\x01\x05\x01\x03\x03%\x01\t\x01\x02\x02)\x05\x11\x15\x07\x03\x13)\x05\x11\x11\x07)\x01\x07!\x13\x11\x05\t\x05\x03\x05\x0b)\x03\t\x0f)\x03\x01\x0f\x04e\x05\x01Q\x01\x07\x01\x07\x04S\x03\x01\x05\x03P\x01\x03\x07\x04?\x03\t\x0f\x05\x13\x13\x0b\x17\x00\x05B\x05\x05\x03\x0b\x07G\x05!\x07\x03\x05\x07\x01\x03\x05\t\x04\x01\x03\x07\x06\x03\x01\x05\x01\x00r\x053#\x0b\x11\x0b\x0b\x0f\x0b!\x03)iK\x05\x05\x13%)9\x15\x1f\x19\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00constant_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00a\x00b\x00jit(func)/jit(main)/triangular_solve\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00\x00jax.result_info\x00main\x00public\x00diag\x00side\x00trans_x\x00uplo\x00lapack_ztrsm_ffi\x00\x08+\t\x05#\x01\x0b/139;\x03=\x11I)KMOQUY",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_12_02['c64'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_ctrsm_ffi'],
    serialized_date=datetime.date(2024, 12, 2),
    inputs=(
        array(
            [
                [5.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [4.0 + 0.0j, 10.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [8.0 + 0.0j, 9.0 + 0.0j, 15.0 + 0.0j, 0.0 + 0.0j],
                [12.0 + 0.0j, 13.0 + 0.0j, 14.0 + 0.0j, 20.0 + 0.0j],
            ],
            dtype=complex64,
        ),
        array(
            [
                [0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j],
                [5.0 + 0.0j, 6.0 + 0.0j, 7.0 + 0.0j, 8.0 + 0.0j, 9.0 + 0.0j],
                [
                    10.0 + 0.0j,
                    11.0 + 0.0j,
                    12.0 + 0.0j,
                    13.0 + 0.0j,
                    14.0 + 0.0j,
                ],
                [
                    15.0 + 0.0j,
                    16.0 + 0.0j,
                    17.0 + 0.0j,
                    18.0 + 0.0j,
                    19.0 + 0.0j,
                ],
            ],
            dtype=complex64,
        ),
    ),
    expected_outputs=(
        array(
            [
                [0.0 + 0.0j, 0.2 + 0.0j, 0.4 + 0.0j, 0.6 + 0.0j, 0.8 + 0.0j],
                [
                    0.5 + 0.0j,
                    0.52 + 0.0j,
                    0.54 + 0.0j,
                    0.56 + 0.0j,
                    0.58000004 + 0.0j,
                ],
                [
                    0.36666667 + 0.0j,
                    0.31466666 + 0.0j,
                    0.26266667 + 0.0j,
                    0.21066667 + 0.0j,
                    0.15866666 + 0.0j,
                ],
                [
                    0.16833334 + 0.0j,
                    0.12173338 + 0.0j,
                    0.0751333 + 0.0j,
                    0.02853328 + 0.0j,
                    -0.018066704 + 0.0j,
                ],
            ],
            dtype=complex64,
        ),
    ),
    mlir_module_text=r"""
#loc1 = loc("a")
#loc2 = loc("b")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f32>> loc("a"), %arg1: tensor<4x5xcomplex<f32>> loc("b")) -> (tensor<4x5xcomplex<f32>> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc4)
    %0 = stablehlo.custom_call @lapack_ctrsm_ffi(%arg0, %arg1, %cst) {mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x4xcomplex<f32>>, tensor<4x5xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x5xcomplex<f32>> loc(#loc4)
    return %0 : tensor<4x5xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":715:13)
#loc4 = loc("jit(func)/jit(main)/triangular_solve"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.1\x00\x01\x1b\x05\x01\x05\x0b\x01\x03\x0b\x03\t\x0f\x13\x17\x1b\x03\x87[\x19\x01%\x07\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x17\x0b\x13\x0b\x037O\x0b\x0b\x0f\x0f\x13\x0b\x0f\x13\x0b\x0b\x0b/+\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0f\x0f\x13\x0f\x01\x05\x0b\x0f\x03\x15\x17\x0b\x17\x0f\x07\x07\x1b\x07\x13\x13\x02\xfe\x02\x1f\x11\x03\x05\x1d\x1b\x1d\x03\x07\t\x0b\r\x03\x0f\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x15\x01\x05\x17\x1d\x19\x01\x05\x19\x05\x1b\x17\x1f.\x0b\x1b\x05\x1d\x03\x03#?\x05\x1f\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x01\x1d!\x13\rN\x13\rL\x03\x05''#\x11\x03\x035\r\x037)\x1d#\x1d%\x1d'\x1f\x0b\x11\x00\x00\x80?\x00\x00\x00\x00\r\tA+C-E+G-\x1d)\x1d+\x1d-\x1d/\x0b\x03\x1d1\x03\x01\x05\x01\x03\x07%%S\x1f\x17\x01\x03\x03W\x15\x01\x05\x01\x03\x03%\x01\t\x01\x02\x02)\x05\x11\x15\x07\x03\x13)\x05\x11\x11\x07)\x01\x07!\x13\x11\x05\t\x05\x03\x05\t)\x03\t\x0f)\x03\x01\x0f\x04e\x05\x01Q\x01\x07\x01\x07\x04S\x03\x01\x05\x03P\x01\x03\x07\x04?\x03\t\x0f\x05\x13\x13\x0b\x17\x00\x05B\x05\x05\x03\x0b\x07G\x05!\x07\x03\x05\x07\x01\x03\x05\t\x04\x01\x03\x07\x06\x03\x01\x05\x01\x00r\x053#\x0b\x11\x0b\x0b\x0f\x0b!\x03)iK\x05\x05\x13%)9\x15\x1f\x19\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00constant_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00a\x00b\x00jit(func)/jit(main)/triangular_solve\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00\x00jax.result_info\x00main\x00public\x00diag\x00side\x00trans_x\x00uplo\x00lapack_ctrsm_ffi\x00\x08+\t\x05#\x01\x0b/139;\x03=\x11I)KMOQUY",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_12_02['f32'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_strsm_ffi'],
    serialized_date=datetime.date(2024, 12, 2),
    inputs=(
        array(
            [
                [5.0, 0.0, 0.0, 0.0],
                [4.0, 10.0, 0.0, 0.0],
                [8.0, 9.0, 15.0, 0.0],
                [12.0, 13.0, 14.0, 20.0],
            ],
            dtype=float32,
        ),
        array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
            ],
            dtype=float32,
        ),
    ),
    expected_outputs=(
        array(
            [
                [0.0, 0.2, 0.4, 0.6, 0.8],
                [0.5, 0.52, 0.54, 0.56, 0.58000004],
                [0.36666667, 0.31466666, 0.26266667, 0.21066667, 0.15866666],
                [0.16833334, 0.12173338, 0.0751333, 0.02853328, -0.018066704],
            ],
            dtype=float32,
        ),
    ),
    mlir_module_text=r"""
#loc1 = loc("a")
#loc2 = loc("b")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf32> loc("a"), %arg1: tensor<4x5xf32> loc("b")) -> (tensor<4x5xf32> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(#loc4)
    %0 = stablehlo.custom_call @lapack_strsm_ffi(%arg0, %arg1, %cst) {mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x4xf32>, tensor<4x5xf32>, tensor<f32>) -> tensor<4x5xf32> loc(#loc4)
    return %0 : tensor<4x5xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":715:13)
#loc4 = loc("jit(func)/jit(main)/triangular_solve"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.1\x00\x01\x1b\x05\x01\x05\x0b\x01\x03\x0b\x03\t\x0f\x13\x17\x1b\x03\x85[\x17\x01%\x07\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x17\x0b\x13\x0b\x037O\x0b\x0b\x0f\x0f\x13\x0b\x0f\x13\x0b\x0b\x0b\x1f+\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0f\x0f\x13\x0f\x01\x05\x0b\x0f\x03\x13\x17\x07\x17\x0f\x07\x07\x1b\x13\x13\x02\xe6\x02\x1f\x11\x03\x05\x1d\x1b\x1d\x03\x07\t\x0b\r\x03\x0f\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x15\x01\x05\x17\x1d\x19\x01\x05\x19\x05\x1b\x17\x1f.\x0b\x1b\x05\x1d\x03\x03#?\x05\x1f\x1f\x13!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x01\x1d!\x13\rN\x13\rL\x03\x05''#\x11\x03\x035\r\x037)\x1d#\x1d%\x1d'\x1f\x0b\t\x00\x00\x80?\r\tA+C-E+G-\x1d)\x1d+\x1d-\x1d/\x0b\x03\x1d1\x03\x01\x05\x01\x03\x07%%S\x1f\x15\x01\x03\x03W\x15\x01\x05\x01\x03\x03%\x01\t\x01\x02\x02)\x05\x11\x15\x07\t)\x05\x11\x11\x07)\x01\x07!\x13\x11\x05\t\x05\x03\x05)\x03\t\x0f)\x03\x01\x0f\x04e\x05\x01Q\x01\x07\x01\x07\x04S\x03\x01\x05\x03P\x01\x03\x07\x04?\x03\t\x0f\x05\x13\x13\x0b\x17\x00\x05B\x05\x05\x03\x0b\x07G\x05!\x07\x03\x05\x07\x01\x03\x05\t\x04\x01\x03\x07\x06\x03\x01\x05\x01\x00r\x053#\x0b\x11\x0b\x0b\x0f\x0b!\x03)iK\x05\x05\x13%)9\x15\x1f\x19\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00constant_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00a\x00b\x00jit(func)/jit(main)/triangular_solve\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00\x00jax.result_info\x00main\x00public\x00diag\x00side\x00trans_x\x00uplo\x00lapack_strsm_ffi\x00\x08+\t\x05#\x01\x0b/139;\x03=\x11I)KMOQUY",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_12_02['f64'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dtrsm_ffi'],
    serialized_date=datetime.date(2024, 12, 2),
    inputs=(
        array([
            [5.0, 0.0, 0.0, 0.0],
            [4.0, 10.0, 0.0, 0.0],
            [8.0, 9.0, 15.0, 0.0],
            [12.0, 13.0, 14.0, 20.0],
        ]),
        array([
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
        ]),
    ),
    expected_outputs=(
        array([
            [0.0, 0.2, 0.4, 0.6000000000000001, 0.8],
            [0.5, 0.52, 0.54, 0.5599999999999999, 0.58],
            [
                0.36666666666666664,
                0.3146666666666667,
                0.2626666666666667,
                0.21066666666666667,
                0.15866666666666665,
            ],
            [
                0.16833333333333336,
                0.1217333333333333,
                0.07513333333333323,
                0.0285333333333333,
                -0.018066666666666675,
            ],
        ]),
    ),
    mlir_module_text=r"""
#loc1 = loc("a")
#loc2 = loc("b")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf64> loc("a"), %arg1: tensor<4x5xf64> loc("b")) -> (tensor<4x5xf64> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64> loc(#loc4)
    %0 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %arg1, %cst) {mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<4x4xf64>, tensor<4x5xf64>, tensor<f64>) -> tensor<4x5xf64> loc(#loc4)
    return %0 : tensor<4x5xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":715:13)
#loc4 = loc("jit(func)/jit(main)/triangular_solve"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.1\x00\x01\x1b\x05\x01\x05\x0b\x01\x03\x0b\x03\t\x0f\x13\x17\x1b\x03\x85[\x17\x01%\x07\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x17\x0b\x13\x0b\x037O\x0b\x0b\x0f\x0f\x13\x0b\x0f\x13\x0b\x0b\x0b/+\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0f\x0f\x13\x0f\x01\x05\x0b\x0f\x03\x13\x17\x07\x17\x0f\x07\x07\x1b\x13\x13\x02\xf6\x02\x1f\x11\x03\x05\x1d\x1b\x1d\x03\x07\t\x0b\r\x03\x0f\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x15\x01\x05\x17\x1d\x19\x01\x05\x19\x05\x1b\x17\x1f.\x0b\x1b\x05\x1d\x03\x03#?\x05\x1f\x1f\x13!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x01\x1d!\x13\rN\x13\rL\x03\x05''#\x11\x03\x035\r\x037)\x1d#\x1d%\x1d'\x1f\x0b\x11\x00\x00\x00\x00\x00\x00\xf0?\r\tA+C-E+G-\x1d)\x1d+\x1d-\x1d/\x0b\x03\x1d1\x03\x01\x05\x01\x03\x07%%S\x1f\x15\x01\x03\x03W\x15\x01\x05\x01\x03\x03%\x01\t\x01\x02\x02)\x05\x11\x15\x07\x0b)\x05\x11\x11\x07)\x01\x07!\x13\x11\x05\t\x05\x03\x05)\x03\t\x0f)\x03\x01\x0f\x04e\x05\x01Q\x01\x07\x01\x07\x04S\x03\x01\x05\x03P\x01\x03\x07\x04?\x03\t\x0f\x05\x13\x13\x0b\x17\x00\x05B\x05\x05\x03\x0b\x07G\x05!\x07\x03\x05\x07\x01\x03\x05\t\x04\x01\x03\x07\x06\x03\x01\x05\x01\x00r\x053#\x0b\x11\x0b\x0b\x0f\x0b!\x03)iK\x05\x05\x13%)9\x15\x1f\x19\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00constant_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00a\x00b\x00jit(func)/jit(main)/triangular_solve\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00\x00jax.result_info\x00main\x00public\x00diag\x00side\x00trans_x\x00uplo\x00lapack_dtrsm_ffi\x00\x08+\t\x05#\x01\x0b/139;\x03=\x11I)KMOQUY",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
