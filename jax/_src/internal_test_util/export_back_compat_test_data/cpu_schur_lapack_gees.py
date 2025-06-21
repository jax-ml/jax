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

data_2024_11_29 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_11_29["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgees_ffi'],
    serialized_date=datetime.date(2024, 12, 2),
    inputs=(array([[ 0.+0.j,  1.+0.j,  2.+0.j,  3.+0.j],
       [ 4.+0.j,  5.+0.j,  6.+0.j,  7.+0.j],
       [ 8.+0.j,  9.+0.j, 10.+0.j, 11.+0.j],
       [12.+0.j, 13.+0.j, 14.+0.j, 15.+0.j]]),),
    expected_outputs=(array([[ 3.2464249196572972e+01+0.j, -1.3416407864998739e+01+0.j,
        -1.2558842947806125e-14+0.j, -7.3490869705474997e-15+0.j],
       [ 0.0000000000000000e+00+0.j, -2.4642491965729798e+00+0.j,
        -2.5534994473279107e-15+0.j, -1.3671521621839345e-16+0.j],
       [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
        -1.8779126463272594e-15+0.j,  7.2486619604759691e-16+0.j],
       [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
         0.0000000000000000e+00+0.j,  4.8523679991768567e-16+0.j]]), array([[ 0.11417645138733863+0.j, -0.8288327563197511 +0.j,
         0.5401354211381763 +0.j, -0.09085002384085737+0.j],
       [ 0.33000459866554743+0.j, -0.43714638836388686+0.j,
        -0.6524649518290251 +0.j,  0.5237265380279561 +0.j],
       [ 0.545832745943757  +0.j, -0.04546002040802424-0.j,
        -0.31547635975648136+0.j, -0.774903004533341  +0.j],
       [ 0.7616608932219662 +0.j,  0.346226347547838  +0.j,
         0.42780589044732925+0.j,  0.3420264903462419 +0.j]])),
    mlir_module_text=r"""
#loc1 = loc("input")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f64>> loc("input")) -> (tensor<4x4xcomplex<f64>> {jax.result_info = "[0]"}, tensor<4x4xcomplex<f64>> {jax.result_info = "[1]"}) {
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @lapack_zgees_ffi(%arg0) {mhlo.backend_config = {mode = 86 : ui8, sort = 78 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xcomplex<f64>>) -> (tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>>, tensor<4xcomplex<f64>>, tensor<i32>, tensor<i32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc3)
    return %6, %10 : tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":631:13)
#loc3 = loc("jit(func)/jit(main)/schur"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.1\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xa5e-\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03E\x0fO\x0b\x0fO\x0f\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0bO\x1f\x1b\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x0b\x0b\x01\x05\x0b\x0f\x03)\x17\x0f\x0b\x07\x07\x0f\x07\x07\x17\x17\x1b\x07\x07\x13\x13\x13\x13\x13\x0f\x13\x02>\x04\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x03\x05\x1d\x03\x03\x19C\x05\x1f\x05!\x17\x1f\xde\t\x1b\x05#\x1f'\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1d%\x1f%\x01\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03-\r\x01#\x19\x03\x0537\r\x03%5\x1d'\r\x03%9\x1d)\x1d+\x1d-\x1f\x0f!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x07\t\x00\x00\x00\x00\r\x05EGIK\x1d/\x13\x11V\x1d1\x13\x11N\x0b\x03\x1d3\x1d5\x03\x01\x05\x01\x03\x03#\x03\x03[\x15\x03\x01\x01\x01\x03\x0b##_''\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x01\t\x01\x02\x02)\x05\x11\x11\t)\x01\x1d\x03\x1b\x13\x01)\x01\t!\x1d)\x05\x05\x05\r)\x05\x11\x11\r\x11\x03\x05\x05\x05\x05\x0b\x1b)\x03\x11\t)\x03\t\x0b)\x03\x05\x0b)\x03\x01\x0b)\x03\x01\x13)\x01\r)\x03\t\x13\x046\x02\x05\x01Q\x03\x07\x01\x07\x04\x0e\x02\x03\x01\x05\tP\x03\x03\x07\x04\xf3\x03%;\x03\x0b\x13\x00\x05B\x03\x05\x03\x0f\x05B\x03\x07\x03\x07\x0bG\x01\x17\t\x0b\x05\x05\x1f\x07\x07\x03\x01\x03F\x01\x0b\x03\x07\x03\x05\rF\x01\r\x03)\x05\x0f\x11\x03F\x01\x0b\x03\x15\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x0f\x03\x17\x03\x15\x07\x06\x01\x03\x05\x07\x19\x07\x17\x03F\x01\x0b\x03\x15\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x0f\x03\x17\x03\x1d\x07\x06\x01\x03\x05\x07!\t\x1f\x0f\x04\x03\x05\x1b#\x06\x03\x01\x05\x01\x00\xe6\x057#\x03\x0b\x0b\x0f\x0b\t\t!i5)\r\x13%)9\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00input\x00mhlo.backend_config\x00jit(func)/jit(main)/schur\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00[0]\x00[1]\x00main\x00public\x00mode\x00sort\x00\x00lapack_zgees_ffi\x00\x08=\x11\x05#\x01\x0b+/1;=\x03?\x03A\x11MOQSUWY]\x03!\x05ac\x03)",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_11_29["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgees_ffi'],
    serialized_date=datetime.date(2024, 12, 2),
    inputs=(array([[ 0.+0.j,  1.+0.j,  2.+0.j,  3.+0.j],
       [ 4.+0.j,  5.+0.j,  6.+0.j,  7.+0.j],
       [ 8.+0.j,  9.+0.j, 10.+0.j, 11.+0.j],
       [12.+0.j, 13.+0.j, 14.+0.j, 15.+0.j]], dtype=complex64),),
    expected_outputs=(array([[ 3.2464264e+01+0.j, -1.3416414e+01+0.j, -2.1337737e-06+0.j,
         1.8261760e-06+0.j],
       [ 0.0000000e+00+0.j, -2.4642489e+00+0.j, -6.0543999e-07+0.j,
         4.8744488e-07+0.j],
       [ 0.0000000e+00+0.j,  0.0000000e+00+0.j, -6.5878328e-07+0.j,
         3.9895070e-07+0.j],
       [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  0.0000000e+00+0.j,
         3.0199919e-07+0.j]], dtype=complex64), array([[ 0.11417647 +0.j, -0.8288329  +0.j,  0.5404726  +0.j,
        -0.08882082 +0.j],
       [ 0.3300045  +0.j, -0.4371462  +0.j, -0.6544272  +0.j,
         0.52127254 +0.j],
       [ 0.54583293 +0.j, -0.045460045-0.j, -0.312564   +0.j,
        -0.77608234 +0.j],
       [ 0.76166105 +0.j,  0.34622625 +0.j,  0.42651838 +0.j,
         0.34363067 +0.j]], dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("input")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f32>> loc("input")) -> (tensor<4x4xcomplex<f32>> {jax.result_info = "[0]"}, tensor<4x4xcomplex<f32>> {jax.result_info = "[1]"}) {
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @lapack_cgees_ffi(%arg0) {mhlo.backend_config = {mode = 86 : ui8, sort = 78 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>, tensor<i32>, tensor<i32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc3)
    return %6, %10 : tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":631:13)
#loc3 = loc("jit(func)/jit(main)/schur"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.1\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xa5e-\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03E\x0fO\x0b\x0fO\x0f\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0b/\x1f\x1b\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x0b\x0b\x01\x05\x0b\x0f\x03)\x17\x0f\x0b\x07\x07\x0f\x07\x07\x17\x17\x1b\x07\x07\x13\x13\x13\x13\x13\x0f\x13\x02\x1e\x04\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x03\x05\x1d\x03\x03\x19C\x05\x1f\x05!\x17\x1f\xde\t\x1b\x05#\x1f'\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1d%\x1f%\x01\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03-\r\x01#\x19\x03\x0537\r\x03%5\x1d'\r\x03%9\x1d)\x1d+\x1d-\x1f\x0f\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x07\t\x00\x00\x00\x00\r\x05EGIK\x1d/\x13\x11V\x1d1\x13\x11N\x0b\x03\x1d3\x1d5\x03\x01\x05\x01\x03\x03#\x03\x03[\x15\x03\x01\x01\x01\x03\x0b##_''\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x01\t\x01\x02\x02)\x05\x11\x11\t)\x01\x1d\x03\x1b\x13\x01)\x01\t!\x1d)\x05\x05\x05\r)\x05\x11\x11\r\x11\x03\x05\x05\x05\x05\t\x1b)\x03\x11\t)\x03\t\x0b)\x03\x05\x0b)\x03\x01\x0b)\x03\x01\x13)\x01\r)\x03\t\x13\x046\x02\x05\x01Q\x03\x07\x01\x07\x04\x0e\x02\x03\x01\x05\tP\x03\x03\x07\x04\xf3\x03%;\x03\x0b\x13\x00\x05B\x03\x05\x03\x0f\x05B\x03\x07\x03\x07\x0bG\x01\x17\t\x0b\x05\x05\x1f\x07\x07\x03\x01\x03F\x01\x0b\x03\x07\x03\x05\rF\x01\r\x03)\x05\x0f\x11\x03F\x01\x0b\x03\x15\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x0f\x03\x17\x03\x15\x07\x06\x01\x03\x05\x07\x19\x07\x17\x03F\x01\x0b\x03\x15\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x0f\x03\x17\x03\x1d\x07\x06\x01\x03\x05\x07!\t\x1f\x0f\x04\x03\x05\x1b#\x06\x03\x01\x05\x01\x00\xe6\x057#\x03\x0b\x0b\x0f\x0b\t\t!i5)\r\x13%)9\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00input\x00mhlo.backend_config\x00jit(func)/jit(main)/schur\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00[0]\x00[1]\x00main\x00public\x00mode\x00sort\x00\x00lapack_cgees_ffi\x00\x08=\x11\x05#\x01\x0b+/1;=\x03?\x03A\x11MOQSUWY]\x03!\x05ac\x03)",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_11_29["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgees_ffi'],
    serialized_date=datetime.date(2024, 12, 2),
    inputs=(array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.],
       [12., 13., 14., 15.]], dtype=float32),),
    expected_outputs=(array([[ 3.2464233e+01, -1.3416398e+01, -1.6680369e-05,  4.0411728e-06],
       [ 0.0000000e+00, -2.4642496e+00, -1.8640144e-06,  6.7429795e-07],
       [ 0.0000000e+00,  0.0000000e+00, -7.2618576e-07,  3.9895073e-07],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  3.0443638e-07]],
      dtype=float32), array([[-0.11417632 ,  0.8288333  , -0.5413438  ,  0.08334288 ],
       [-0.33000442 ,  0.43714583 ,  0.65967286 , -0.5146185  ],
       [-0.54583275 ,  0.045459934,  0.30468878 ,  0.7792079  ],
       [-0.7616609  , -0.34622616 , -0.4230168  , -0.34793234 ]],
      dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("input")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf32> loc("input")) -> (tensor<4x4xf32> {jax.result_info = "[0]"}, tensor<4x4xf32> {jax.result_info = "[1]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:6 = stablehlo.custom_call @lapack_sgees_ffi(%arg0) {mhlo.backend_config = {mode = 86 : ui8, sort = 78 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<i32>, tensor<i32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#5, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc3)
    return %6, %10 : tensor<4x4xf32>, tensor<4x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":631:13)
#loc3 = loc("jit(func)/jit(main)/schur"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.1\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xa3e+\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03E\x0fO\x0b/\x0fO\x0f\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x1b\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17#\x0b\x0b\x01\x05\x0b\x0f\x03'\x17\x0f\x07\x07\x07\x0f\x13\x07\x07\x17\x17\x1b\x07\x13\x13\x13\x13\x0f\x13\x02\n\x04\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x03\x05\x1d\x03\x03\x19E\x05\x1f\x05!\x17\x1f\xde\t\x1b\x05#\x1f%\x01\x1f\x1f!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1d%\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f#\x01\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03/\r\x01#\x1b\x03\x0559\r\x03%7\x1d'\r\x03%;\x1d)\x1d+\x1d-\x1f\x0f\t\x00\x00\xc0\x7f\x1f\x07\t\x00\x00\x00\x00\r\x05GIKM\x1d/\x13\x13V\x1d1\x13\x13N\x0b\x03\x1d3\x1d5\x03\x01\x05\x01\x03\x03#\x03\x03]\x15\x03\x01\x01\x01\x03\r##''))\t\x07\x07\x01\x01\t\x01\x02\x02)\x05\x11\x11\t)\x01\x1d\t\x13\x01)\x01\t)\x03\x11\t!\x1d)\x05\x05\x05\r)\x05\x11\x11\r\x11\x03\x05\x05\x05\x05\x1b)\x03\t\x0b)\x03\x05\x0b)\x03\x01\x0b)\x03\x01\x15)\x01\r)\x03\t\x15\x04:\x02\x05\x01Q\x03\x07\x01\x07\x04\x12\x02\x03\x01\x05\tP\x03\x03\x07\x04\xf5\x03';\x03\x0b\x13\x00\x05B\x03\x05\x03\x0f\x05B\x03\x07\x03\x07\x0bG\x01\x17\t\r\x05\x05\x11\x11\x07\x07\x03\x01\x03F\x01\x0b\x03\x07\x03\x05\rF\x01\r\x03'\x05\x11\x13\x03F\x01\x0b\x03\x17\x03\x15\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x0f\x03\x19\x03\x17\x07\x06\x01\x03\x05\x07\x1b\x07\x19\x03F\x01\x0b\x03\x17\x03\x15\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x0f\x03\x19\x03\x1f\x07\x06\x01\x03\x05\x07#\t!\x0f\x04\x03\x05\x1d%\x06\x03\x01\x05\x01\x00\xe6\x057#\x03\x0b\x0b\x0f\x0b\t\t!i5)\r\x13%)9\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00input\x00mhlo.backend_config\x00jit(func)/jit(main)/schur\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00[0]\x00[1]\x00main\x00public\x00mode\x00sort\x00\x00lapack_sgees_ffi\x00\x08=\x11\x05#\x01\x0b-13=?\x03A\x03C\x11OQSUWY[_\x03!\x05ac\x03+",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_11_29["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgees_ffi'],
    serialized_date=datetime.date(2024, 12, 2),
    inputs=(array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.],
       [12., 13., 14., 15.]]),),
    expected_outputs=(array([[ 3.2464249196572979e+01, -1.3416407864998748e+01,
         4.7128510442204522e-15, -8.6687960588453852e-15],
       [ 0.0000000000000000e+00, -2.4642491965729767e+00,
         1.8990547895861982e-15, -2.4680570671743780e-16],
       [ 0.0000000000000000e+00,  0.0000000000000000e+00,
        -1.8780225147134376e-15, -7.2486619604759710e-16],
       [ 0.0000000000000000e+00,  0.0000000000000000e+00,
         0.0000000000000000e+00,  4.8523923435746521e-16]]), array([[-0.1141764513873386 ,  0.8288327563197505 ,  0.5401360966805397 ,
         0.09084600741204968],
       [-0.3300045986655475 ,  0.43714638836388714, -0.6524688462214561 ,
        -0.5237216863090944 ],
       [-0.5458327459437569 ,  0.04546002040802441, -0.31547059759870844,
         0.774905350382041  ],
       [-0.7616608932219663 , -0.34622634754783793,  0.4278033471396243 ,
        -0.3420296714849957 ]])),
    mlir_module_text=r"""
#loc1 = loc("input")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf64> loc("input")) -> (tensor<4x4xf64> {jax.result_info = "[0]"}, tensor<4x4xf64> {jax.result_info = "[1]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:6 = stablehlo.custom_call @lapack_dgees_ffi(%arg0) {mhlo.backend_config = {mode = 86 : ui8, sort = 78 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4x4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<i32>, tensor<i32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#5, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc3)
    return %6, %10 : tensor<4x4xf64>, tensor<4x4xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":631:13)
#loc3 = loc("jit(func)/jit(main)/schur"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.1\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xa3e+\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03E\x0fO\x0b/\x0fO\x0f\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0b/\x1f\x1b\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17#\x0b\x0b\x01\x05\x0b\x0f\x03'\x17\x0f\x07\x07\x07\x0f\x13\x07\x07\x17\x17\x1b\x07\x13\x13\x13\x13\x0f\x13\x02\x1a\x04\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x03\x05\x1d\x03\x03\x19E\x05\x1f\x05!\x17\x1f\xde\t\x1b\x05#\x1f%\x01\x1f\x1f!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1d%\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f#\x01\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03/\r\x01#\x1b\x03\x0559\r\x03%7\x1d'\r\x03%;\x1d)\x1d+\x1d-\x1f\x0f\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x07\t\x00\x00\x00\x00\r\x05GIKM\x1d/\x13\x13V\x1d1\x13\x13N\x0b\x03\x1d3\x1d5\x03\x01\x05\x01\x03\x03#\x03\x03]\x15\x03\x01\x01\x01\x03\r##''))\t\x07\x07\x01\x01\t\x01\x02\x02)\x05\x11\x11\t)\x01\x1d\x0b\x13\x01)\x01\t)\x03\x11\t!\x1d)\x05\x05\x05\r)\x05\x11\x11\r\x11\x03\x05\x05\x05\x05\x1b)\x03\t\x0b)\x03\x05\x0b)\x03\x01\x0b)\x03\x01\x15)\x01\r)\x03\t\x15\x04:\x02\x05\x01Q\x03\x07\x01\x07\x04\x12\x02\x03\x01\x05\tP\x03\x03\x07\x04\xf5\x03';\x03\x0b\x13\x00\x05B\x03\x05\x03\x0f\x05B\x03\x07\x03\x07\x0bG\x01\x17\t\r\x05\x05\x11\x11\x07\x07\x03\x01\x03F\x01\x0b\x03\x07\x03\x05\rF\x01\r\x03'\x05\x11\x13\x03F\x01\x0b\x03\x17\x03\x15\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x0f\x03\x19\x03\x17\x07\x06\x01\x03\x05\x07\x1b\x07\x19\x03F\x01\x0b\x03\x17\x03\x15\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x0f\x03\x19\x03\x1f\x07\x06\x01\x03\x05\x07#\t!\x0f\x04\x03\x05\x1d%\x06\x03\x01\x05\x01\x00\xe6\x057#\x03\x0b\x0b\x0f\x0b\t\t!i5)\r\x13%)9\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00input\x00mhlo.backend_config\x00jit(func)/jit(main)/schur\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00[0]\x00[1]\x00main\x00public\x00mode\x00sort\x00\x00lapack_dgees_ffi\x00\x08=\x11\x05#\x01\x0b-13=?\x03A\x03C\x11OQSUWY[_\x03!\x05ac\x03+",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
