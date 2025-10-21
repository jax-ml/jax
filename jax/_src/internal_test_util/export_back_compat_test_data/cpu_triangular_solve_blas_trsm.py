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

data_2025_10_20 = {}


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_10_20['f32'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_strsm_ffi'],
    serialized_date=datetime.date(2025, 10, 20),
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
#loc1 = loc("a")
#loc2 = loc("b")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf32> loc("a"), %arg1: tensor<4x5xf32> loc("b")) -> (tensor<4x5xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @lapack_strsm_ffi(%arg0, %arg1) {mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=4, j=4, k=4, l=5, m=4, n=5}, custom>} : (tensor<4x4xf32>, tensor<4x5xf32>) -> tensor<4x5xf32> loc(#loc5)
    return %0 : tensor<4x5xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":654:13)
#loc4 = loc("jit(func)"(#loc3))
#loc5 = loc("triangular_solve"(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01\x1b\x07\x01\x05\t\t\x01\x03\x0f\x03\x07\x13\x17\x1b\x03\xa5{\x13\x01-\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x17\x0b\x03;O\x0b\x0f\x0f\x13\x0b\x0f\x13\x0b\x0b\x0b\x0b+\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x13\x0f\x05\x15K\x13\x0f\x0f\x13\x0f\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x03\x0f\x17\x17\x07\x07\x1b\x13\x07\x02\xba\x03\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x13\x01\x05\x17\x1d\x17\x01\x05\x19\x03\x07\x1bE\x1dO\x1fg\x05\x1b\x05\x1d\x05\x1f\x1d#%\x05!\x1d')\x05#\x17+:\n\x1b\x05%\x1f\x0f!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x01\x13\x0bN\x13\x0bL\x03\x05//#\r\x03\x03;\r\x03=?\x1d'\x1d)\x1d+\x1d-\r\tG1I3K1M3\x1d/\x1d1\x1d3\x1d5\r\x03QS\x1d7\x1d9\x0b\x03\x1d;\x1d=\x03\x01\x05\x01\x03\x05--\x03\x03c\x15\x01\x05\x01\x03\x03-\x15\r\x11\x11\x11\x15\x11\x15\x05io\x03u\x01\x01\x01\x01\x01\x13\x05km\x11\x03\x01\x11\x03\x05\x13\x05qs\x11\x03\t\x11\x03\r\x13\x05wy\x11\x03\x11\x11\x03\x15\x01\t\x01\x02\x02)\x05\x11\x15\t)\x05\x11\x11\t\t!\x11\x05\x07\x05\x03\x05)\x03\t\x11\x13\x04W\x05\x01Q\x01\x05\x01\x07\x04E\x03\x01\x05\x03P\x01\x03\x07\x041\x03\x07\x0b\x05\x0f\x11\x0b\x15\x00\x05G!\x19\x05\x03\x05\x05\x01\x03\x07\x04\x01\x03\x05\x06\x03\x01\x05\x01\x00N\x06?#\x03\x05\x1f\x0b\x11\x0b\x0b\x0f\x0b\x0f!i\x15#%3)\x05\x05\x13%)9\x15\x1f\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00a\x00b\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00triangular_solve\x00jit(func)\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00result\x00main\x00public\x00diag\x00side\x00trans_x\x00uplo\x00num_batch_dims\x000\x00\x00lapack_strsm_ffi\x00\x08'\x07\x05\x1f\x01\x0b579AC\x11UWY[]_ae",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_10_20['f64'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dtrsm_ffi'],
    serialized_date=datetime.date(2025, 10, 20),
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
#loc1 = loc("a")
#loc2 = loc("b")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf64> loc("a"), %arg1: tensor<4x5xf64> loc("b")) -> (tensor<4x5xf64> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %arg1) {mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=4, j=4, k=4, l=5, m=4, n=5}, custom>} : (tensor<4x4xf64>, tensor<4x5xf64>) -> tensor<4x5xf64> loc(#loc5)
    return %0 : tensor<4x5xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":654:13)
#loc4 = loc("jit(func)"(#loc3))
#loc5 = loc("triangular_solve"(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01\x1b\x07\x01\x05\t\t\x01\x03\x0f\x03\x07\x13\x17\x1b\x03\xa5{\x13\x01-\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x17\x0b\x03;O\x0b\x0f\x0f\x13\x0b\x0f\x13\x0b\x0b\x0b\x0b+\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x13\x0f\x05\x15K\x13\x0f\x0f\x13\x0f\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x03\x0f\x17\x17\x07\x07\x1b\x13\x07\x02\xba\x03\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x13\x01\x05\x17\x1d\x17\x01\x05\x19\x03\x07\x1bE\x1dO\x1fg\x05\x1b\x05\x1d\x05\x1f\x1d#%\x05!\x1d')\x05#\x17+:\n\x1b\x05%\x1f\x0f!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x01\x13\x0bN\x13\x0bL\x03\x05//#\r\x03\x03;\r\x03=?\x1d'\x1d)\x1d+\x1d-\r\tG1I3K1M3\x1d/\x1d1\x1d3\x1d5\r\x03QS\x1d7\x1d9\x0b\x03\x1d;\x1d=\x03\x01\x05\x01\x03\x05--\x03\x03c\x15\x01\x05\x01\x03\x03-\x15\r\x11\x11\x11\x15\x11\x15\x05io\x03u\x01\x01\x01\x01\x01\x13\x05km\x11\x03\x01\x11\x03\x05\x13\x05qs\x11\x03\t\x11\x03\r\x13\x05wy\x11\x03\x11\x11\x03\x15\x01\t\x01\x02\x02)\x05\x11\x15\t)\x05\x11\x11\t\x0b!\x11\x05\x07\x05\x03\x05)\x03\t\x11\x13\x04W\x05\x01Q\x01\x05\x01\x07\x04E\x03\x01\x05\x03P\x01\x03\x07\x041\x03\x07\x0b\x05\x0f\x11\x0b\x15\x00\x05G!\x19\x05\x03\x05\x05\x01\x03\x07\x04\x01\x03\x05\x06\x03\x01\x05\x01\x00N\x06?#\x03\x05\x1f\x0b\x11\x0b\x0b\x0f\x0b\x0f!i\x15#%3)\x05\x05\x13%)9\x15\x1f\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00a\x00b\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00triangular_solve\x00jit(func)\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00result\x00main\x00public\x00diag\x00side\x00trans_x\x00uplo\x00num_batch_dims\x000\x00\x00lapack_dtrsm_ffi\x00\x08'\x07\x05\x1f\x01\x0b579AC\x11UWY[]_ae",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_10_20['c64'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_ctrsm_ffi'],
    serialized_date=datetime.date(2025, 10, 20),
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
#loc1 = loc("a")
#loc2 = loc("b")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f32>> loc("a"), %arg1: tensor<4x5xcomplex<f32>> loc("b")) -> (tensor<4x5xcomplex<f32>> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @lapack_ctrsm_ffi(%arg0, %arg1) {mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=4, j=4, k=4, l=5, m=4, n=5}, custom>} : (tensor<4x4xcomplex<f32>>, tensor<4x5xcomplex<f32>>) -> tensor<4x5xcomplex<f32>> loc(#loc5)
    return %0 : tensor<4x5xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":654:13)
#loc4 = loc("jit(func)"(#loc3))
#loc5 = loc("triangular_solve"(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01\x1b\x07\x01\x05\t\t\x01\x03\x0f\x03\x07\x13\x17\x1b\x03\xa7{\x15\x01-\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x17\x0b\x03;O\x0b\x0f\x0f\x13\x0b\x0f\x13\x0b\x0b\x0b\x0b+\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x13\x0f\x05\x15K\x13\x0f\x0f\x13\x0f\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x03\x11\x17\x17\x0b\x07\x1b\x07\x13\x07\x02\xc2\x03\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x13\x01\x05\x17\x1d\x17\x01\x05\x19\x03\x07\x1bE\x1dO\x1fg\x05\x1b\x05\x1d\x05\x1f\x1d#%\x05!\x1d')\x05#\x17+:\n\x1b\x05%\x1f\x11!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x01\x13\x0bN\x13\x0bL\x03\x05//#\r\x03\x03;\r\x03=?\x1d'\x1d)\x1d+\x1d-\r\tG1I3K1M3\x1d/\x1d1\x1d3\x1d5\r\x03QS\x1d7\x1d9\x0b\x03\x1d;\x1d=\x03\x01\x05\x01\x03\x05--\x03\x03c\x15\x01\x05\x01\x03\x03-\x15\r\x11\x11\x11\x15\x11\x15\x05io\x03u\x01\x01\x01\x01\x01\x13\x05km\x11\x03\x01\x11\x03\x05\x13\x05qs\x11\x03\t\x11\x03\r\x13\x05wy\x11\x03\x11\x11\x03\x15\x01\t\x01\x02\x02)\x05\x11\x15\t)\x05\x11\x11\t\x03\x0f!\x11\x05\x07\x05\x03\x05\t)\x03\t\x13\x13\x04W\x05\x01Q\x01\x05\x01\x07\x04E\x03\x01\x05\x03P\x01\x03\x07\x041\x03\x07\x0b\x05\x0f\x11\x0b\x15\x00\x05G!\x19\x05\x03\x05\x05\x01\x03\x07\x04\x01\x03\x05\x06\x03\x01\x05\x01\x00N\x06?#\x03\x05\x1f\x0b\x11\x0b\x0b\x0f\x0b\x0f!i\x15#%3)\x05\x05\x13%)9\x15\x1f\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00a\x00b\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00triangular_solve\x00jit(func)\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00result\x00main\x00public\x00diag\x00side\x00trans_x\x00uplo\x00num_batch_dims\x000\x00\x00lapack_ctrsm_ffi\x00\x08'\x07\x05\x1f\x01\x0b579AC\x11UWY[]_ae",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_10_20['c128'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_ztrsm_ffi'],
    serialized_date=datetime.date(2025, 10, 20),
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
#loc1 = loc("a")
#loc2 = loc("b")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f64>> loc("a"), %arg1: tensor<4x5xcomplex<f64>> loc("b")) -> (tensor<4x5xcomplex<f64>> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @lapack_ztrsm_ffi(%arg0, %arg1) {mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=4, j=4, k=4, l=5, m=4, n=5}, custom>} : (tensor<4x4xcomplex<f64>>, tensor<4x5xcomplex<f64>>) -> tensor<4x5xcomplex<f64>> loc(#loc5)
    return %0 : tensor<4x5xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":654:13)
#loc4 = loc("jit(func)"(#loc3))
#loc5 = loc("triangular_solve"(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01\x1b\x07\x01\x05\t\t\x01\x03\x0f\x03\x07\x13\x17\x1b\x03\xa7{\x15\x01-\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x17\x0b\x03;O\x0b\x0f\x0f\x13\x0b\x0f\x13\x0b\x0b\x0b\x0b+\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x13\x0f\x05\x15K\x13\x0f\x0f\x13\x0f\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x03\x11\x17\x17\x0b\x07\x1b\x07\x13\x07\x02\xc2\x03\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x13\x01\x05\x17\x1d\x17\x01\x05\x19\x03\x07\x1bE\x1dO\x1fg\x05\x1b\x05\x1d\x05\x1f\x1d#%\x05!\x1d')\x05#\x17+:\n\x1b\x05%\x1f\x11!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x01\x13\x0bN\x13\x0bL\x03\x05//#\r\x03\x03;\r\x03=?\x1d'\x1d)\x1d+\x1d-\r\tG1I3K1M3\x1d/\x1d1\x1d3\x1d5\r\x03QS\x1d7\x1d9\x0b\x03\x1d;\x1d=\x03\x01\x05\x01\x03\x05--\x03\x03c\x15\x01\x05\x01\x03\x03-\x15\r\x11\x11\x11\x15\x11\x15\x05io\x03u\x01\x01\x01\x01\x01\x13\x05km\x11\x03\x01\x11\x03\x05\x13\x05qs\x11\x03\t\x11\x03\r\x13\x05wy\x11\x03\x11\x11\x03\x15\x01\t\x01\x02\x02)\x05\x11\x15\t)\x05\x11\x11\t\x03\x0f!\x11\x05\x07\x05\x03\x05\x0b)\x03\t\x13\x13\x04W\x05\x01Q\x01\x05\x01\x07\x04E\x03\x01\x05\x03P\x01\x03\x07\x041\x03\x07\x0b\x05\x0f\x11\x0b\x15\x00\x05G!\x19\x05\x03\x05\x05\x01\x03\x07\x04\x01\x03\x05\x06\x03\x01\x05\x01\x00N\x06?#\x03\x05\x1f\x0b\x11\x0b\x0b\x0f\x0b\x0f!i\x15#%3)\x05\x05\x13%)9\x15\x1f\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00a\x00b\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00triangular_solve\x00jit(func)\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00result\x00main\x00public\x00diag\x00side\x00trans_x\x00uplo\x00num_batch_dims\x000\x00\x00lapack_ztrsm_ffi\x00\x08'\x07\x05\x1f\x01\x0b579AC\x11UWY[]_ae",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
