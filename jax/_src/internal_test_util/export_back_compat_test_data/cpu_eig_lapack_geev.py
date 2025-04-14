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
from numpy import array, complex64

data_2024_08_19 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_19["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgeev_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([ 3.2464249196572972e+01+0.j, -2.4642491965729794e+00+0.j,
       -1.4596915295025735e-15+0.j,  4.7403016698320490e-16+0.j]), array([[ 0.40377749076862324+0.j,  0.8288327563197503 +0.j,
        -0.5409014947846461 +0.j,  0.10917005482608667-0.j],
       [ 0.4648073711584899 +0.j,  0.43714638836388775-0.j,
         0.7854306338527134 +0.j, -0.5456169434539783 +0.j],
       [ 0.5258372515483575 +0.j,  0.04546002040802463-0.j,
         0.05184321664851461-0.j,  0.7637237224296971 +0.j],
       [ 0.5868671319382249 +0.j, -0.34622634754783843+0.j,
        -0.296372355716581  +0.j, -0.32727683380180517+0.j]]), array([[ 0.11417645138733866+0.j,  0.7327780959803557 +0.j,
        -0.5367326141844461 +0.j, -0.08617176416747369+0.j],
       [ 0.33000459866554754+0.j,  0.28974835239692603-0.j,
         0.6342729310130916 +0.j, -0.28826848493327445+0.j],
       [ 0.5458327459437569 +0.j, -0.15328139118650222+0.j,
         0.34165198052715445-0.j,  0.83505226236897   +0.j],
       [ 0.7616608932219664 +0.j, -0.5963111347699301 +0.j,
        -0.4391922973557999 +0.j, -0.460612013268222  +0.j]])),
    mlir_module_text=r"""
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<4x4xcomplex<f64>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<4x4xcomplex<f64>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<16xcomplex<f64>> loc(#loc3)
    %1 = stablehlo.reshape %0 : (tensor<16xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc4)
    %c = stablehlo.constant dense<4> : tensor<i64> loc(#loc5)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc5)
    %2:4 = stablehlo.custom_call @lapack_zgeev_ffi(%1) {mhlo.backend_config = {compute_left = 86 : ui8, compute_right = 86 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>>, tensor<i32>) loc(#loc5)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %4 = stablehlo.compare  EQ, %2#3, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc5)
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc5)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4xcomplex<f64>> loc(#loc5)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc5)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<4xi1>, tensor<4xcomplex<f64>> loc(#loc5)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %cst_2 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc5)
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc5)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc5)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc5)
    %13 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %cst_3 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc5)
    %14 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc5)
    %15 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc5)
    %16 = stablehlo.select %15, %2#2, %14 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc5)
    return %8, %12, %16 : tensor<4xcomplex<f64>>, tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":210:14)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":211:13)
#loc3 = loc("jit(func)/jit(main)/iota[dtype=complex128 shape=(16,) dimension=0]"(#loc1))
#loc4 = loc("jit(func)/jit(main)/reshape[new_sizes=(4, 4) dimensions=None]"(#loc1))
#loc5 = loc("jit(func)/jit(main)/eig[compute_left_eigenvectors=True compute_right_eigenvectors=True]"(#loc2))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01#\x05\x01\x03\x01\x03\x05\x03\x13\x07\t\x0b\r\x0f\x11\x13\x15\x17\x03\xef\xa57\x01]\x0f\x13\x07\x0b\x0b\x13\x0f\x0b\x17\x0b\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x03I\x0b\x0b\x0b\x0bO\x0f\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0f/\x0b\x0b\x0b\x0b\x1b\x0b\x0b\x0f\x1b/\x0f\x1f\x0f\x0b\x0bO/O\x01\x05\x0b\x0f\x033\x17\x07\x07\x13\x0b\x0f\x0f\x0f\x07\x17\x17\x1b\x07\x13\x07\x07\x13\x13\x13\x13\x0f\x13\x13\x13\x13\x02\xa6\x06\x1d;=\x03\x03\t\x99\x1f\x05\x19\x05\x1b\x03\x03\x07\x9f\x11\x03\x05\x05\x1d\x17\x13J\x03\x1d\x05\x1f\x03\x03\x07\x7f\x03\x03\t\xa3\x03\t\x1b\x1d\x1f\r!\r\x0f#\x05!\x11\x01\x00\x05#\x05%\x05\'\x03\x0b\'])i+k\x0fy-{\x05)\x05+\x05-\x05/\x03\x031}\x051\x1d5\x11\x053\x1d9\x11\x055\x057\x17\x13N\x03\x1b\x03\x13A\x81C\x83E\x85G]I\x87K\x89M\x8fO]Q\x91\x059\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x05I\x03\x03\x07\x97\x03\x05W\x9bY\x9d\x05K\x05M\x03\x03\t\xa1\x03\x01\x1dO\x1dQ\x1dS\x1f%!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13#V#\x1b\x03\x07mqu\r\x05_oac\x1dU\r\x05_sac\x1dW\r\x05_wac\x1dY\x1d[\x1d]\x13\x07\x01\x1f\x13\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d_\x1da\x05\x01\r\x05\x8bg\x8dg\x1dc\x1de\x03\x03e\x03\t\x93ee\x95\x1f\'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f)\x01\x1f\x0f\t\x00\x00\x00\x00\x1f+\x01\t\x07\x07\x01\x1f\x11!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f5!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\r\x1d\x01)\x03\x11\r\x03\x1d)\x01!)\x01\r)\x01\x07\x13)\x05\x05\x05\t)\x05\x11\x11\t\x11\x01\x07\x0b\x05\x05\x0b)\x03A\r\x1b!)\x03\t\x15)\x03\x05\x15)\x03\x01\x15)\x03\x01\x07)\x01\t)\x03\x05\t)\x03\x11\t)\x03\x05\x07)\x03\t\x07\x04"\x03\x05\x01\x11\x05\x19\x07\x03\x01\x05\t\x11\x05%\x07\x035a\x0b\x033/\x03\x1f\r\x067\x03\x05\x03\x01\x05\x03\x01\x15\x03\x13\x05\x03\x01\x15\x03\x13\x0f\x07\x01?\t\x0b\x05\x05\x0f\x03\x03\x05\x03\x01S\x03\x0f\x03\x07\x01\x03\x03\x0f\x03\x11\x11\x07\x01U\x03-\x05\x0f\x13\x03\x07\x01\x03\x03/\x03\x15\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x0b\x03\x19\x03\x07\x01[\x031\x03\x17\x07\x06\x01\x03\x0b\x07\x1d\t\x1b\x03\x07\x01\x03\x03\x17\x03\x15\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x05\x03#\x03\x07\x01\x17\x03\x19\x03!\x07\x06\x01\x03\x05\x07\'\x0b%\x03\x07\x01\x03\x03\x17\x03\x15\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x05\x03-\x03\x07\x01\x17\x03\x19\x03+\x07\x06\x01\x03\x05\x071\r/\x13\x04\x05\x07\x1f)3\x06\x03\x01\x05\x01\x00^\x0eg\x1d\x1b#\x03\x0f\x0b\t\t\t\x11#!+\x1b\x1f/!)!)#\x1f\x19\xb1}\x87\x1f\x1f\x15\x1d\x15\x13%)9i\x13+\r\x15\x17\x1f\x17\x11\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(func)/jit(main)/iota[dtype=complex128 shape=(16,) dimension=0]\x00jit(func)/jit(main)/reshape[new_sizes=(4, 4) dimensions=None]\x00jit(func)/jit(main)/eig[compute_left_eigenvectors=True compute_right_eigenvectors=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_zgeev_ffi\x00compute_left\x00compute_right\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_19["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgeev_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([ 3.2464249e+01+0.j, -2.4642491e+00+0.j, -8.1492220e-07+0.j,
        3.0721142e-07+0.j], dtype=complex64), array([[ 0.40377736 +0.j,  0.8288328  +0.j, -0.53676015 +0.j,
         0.07707452 -0.j],
       [ 0.4648074  +0.j,  0.43714643 -0.j,  0.79694915 +0.j,
        -0.5069523  +0.j],
       [ 0.52583736 +0.j,  0.04545992 -0.j,  0.016383484+0.j,
         0.7826807  +0.j],
       [ 0.5868672  +0.j, -0.34622622 +0.j, -0.2765721  +0.j,
        -0.35280296 +0.j]], dtype=complex64), array([[ 0.114176415+0.j,  0.73277825 +0.j, -0.54227245 +0.j,
        -0.109032825+0.j],
       [ 0.3300045  +0.j,  0.2897482  -0.j,  0.6655821  +0.j,
        -0.25470036 +0.j],
       [ 0.5458329  +0.j, -0.15328139 +0.j,  0.29565343 +0.j,
         0.83649963 +0.j],
       [ 0.7616609  +0.j, -0.59631103 +0.j, -0.4189632  +0.j,
        -0.47276634 +0.j]], dtype=complex64)),
    mlir_module_text=r"""
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<4x4xcomplex<f32>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<4x4xcomplex<f32>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<16xcomplex<f32>> loc(#loc3)
    %1 = stablehlo.reshape %0 : (tensor<16xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc4)
    %c = stablehlo.constant dense<4> : tensor<i64> loc(#loc5)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc5)
    %2:4 = stablehlo.custom_call @lapack_cgeev_ffi(%1) {mhlo.backend_config = {compute_left = 86 : ui8, compute_right = 86 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xcomplex<f32>>) -> (tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<i32>) loc(#loc5)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %4 = stablehlo.compare  EQ, %2#3, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc5)
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc5)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4xcomplex<f32>> loc(#loc5)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc5)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<4xi1>, tensor<4xcomplex<f32>> loc(#loc5)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %cst_2 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc5)
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc5)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc5)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc5)
    %13 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %cst_3 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc5)
    %14 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc5)
    %15 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc5)
    %16 = stablehlo.select %15, %2#2, %14 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc5)
    return %8, %12, %16 : tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":210:14)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":211:13)
#loc3 = loc("jit(func)/jit(main)/iota[dtype=complex64 shape=(16,) dimension=0]"(#loc1))
#loc4 = loc("jit(func)/jit(main)/reshape[new_sizes=(4, 4) dimensions=None]"(#loc1))
#loc5 = loc("jit(func)/jit(main)/eig[compute_left_eigenvectors=True compute_right_eigenvectors=True]"(#loc2))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01#\x05\x01\x03\x01\x03\x05\x03\x13\x07\t\x0b\r\x0f\x11\x13\x15\x17\x03\xef\xa57\x01]\x0f\x13\x07\x0b\x0b\x13\x0f\x0b\x17\x0b\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x03I\x0b\x0b\x0b\x0bO\x0f\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0f/\x0b\x0b\x0b\x0b\x1b\x0b\x0b\x0f\x1b/\x0f\x1f\x0f\x0b\x0b//O\x01\x05\x0b\x0f\x033\x17\x07\x07\x13\x0b\x0f\x0f\x0f\x07\x17\x17\x1b\x07\x13\x07\x07\x13\x13\x13\x13\x0f\x13\x13\x13\x13\x02\x86\x06\x1d;=\x03\x03\t\x99\x1f\x05\x19\x05\x1b\x03\x03\x07\x9f\x11\x03\x05\x05\x1d\x17\x13J\x03\x1d\x05\x1f\x03\x03\x07\x7f\x03\x03\t\xa3\x03\t\x1b\x1d\x1f\r!\r\x0f#\x05!\x11\x01\x00\x05#\x05%\x05\'\x03\x0b\'])i+k\x0fy-{\x05)\x05+\x05-\x05/\x03\x031}\x051\x1d5\x11\x053\x1d9\x11\x055\x057\x17\x13N\x03\x1b\x03\x13A\x81C\x83E\x85G]I\x87K\x89M\x8fO]Q\x91\x059\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x05I\x03\x03\x07\x97\x03\x05W\x9bY\x9d\x05K\x05M\x03\x03\t\xa1\x03\x01\x1dO\x1dQ\x1dS\x1f%!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13#V#\x1b\x03\x07mqu\r\x05_oac\x1dU\r\x05_sac\x1dW\r\x05_wac\x1dY\x1d[\x1d]\x13\x07\x01\x1f\x13\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d_\x1da\x05\x01\r\x05\x8bg\x8dg\x1dc\x1de\x03\x03e\x03\t\x93ee\x95\x1f\'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f)\x01\x1f\x0f\t\x00\x00\x00\x00\x1f+\x01\t\x07\x07\x01\x1f\x11\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f5!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\r\x1d\x01)\x03\x11\r\x03\x1d)\x01!)\x01\r)\x01\x07\x13)\x05\x05\x05\t)\x05\x11\x11\t\x11\x01\x07\x0b\x05\x05\t)\x03A\r\x1b!)\x03\t\x15)\x03\x05\x15)\x03\x01\x15)\x03\x01\x07)\x01\t)\x03\x05\t)\x03\x11\t)\x03\x05\x07)\x03\t\x07\x04"\x03\x05\x01\x11\x05\x19\x07\x03\x01\x05\t\x11\x05%\x07\x035a\x0b\x033/\x03\x1f\r\x067\x03\x05\x03\x01\x05\x03\x01\x15\x03\x13\x05\x03\x01\x15\x03\x13\x0f\x07\x01?\t\x0b\x05\x05\x0f\x03\x03\x05\x03\x01S\x03\x0f\x03\x07\x01\x03\x03\x0f\x03\x11\x11\x07\x01U\x03-\x05\x0f\x13\x03\x07\x01\x03\x03/\x03\x15\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x0b\x03\x19\x03\x07\x01[\x031\x03\x17\x07\x06\x01\x03\x0b\x07\x1d\t\x1b\x03\x07\x01\x03\x03\x17\x03\x15\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x05\x03#\x03\x07\x01\x17\x03\x19\x03!\x07\x06\x01\x03\x05\x07\'\x0b%\x03\x07\x01\x03\x03\x17\x03\x15\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x05\x03-\x03\x07\x01\x17\x03\x19\x03+\x07\x06\x01\x03\x05\x071\r/\x13\x04\x05\x07\x1f)3\x06\x03\x01\x05\x01\x00Z\x0eg\x1d\x1b#\x03\x0f\x0b\t\t\t\x11#!+\x1b\x1f/!)!)#\x1f\x19\xb1}\x85\x1f\x1f\x15\x1d\x15\x13%)9i\x13+\r\x15\x17\x1f\x17\x11\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(func)/jit(main)/iota[dtype=complex64 shape=(16,) dimension=0]\x00jit(func)/jit(main)/reshape[new_sizes=(4, 4) dimensions=None]\x00jit(func)/jit(main)/eig[compute_left_eigenvectors=True compute_right_eigenvectors=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_cgeev_ffi\x00compute_left\x00compute_right\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_19["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgeev_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([ 3.2464241e+01+0.j, -2.4642482e+00+0.j, -4.5555478e-07+0.j,
        2.9215252e-07+0.j], dtype=complex64), array([[-0.40377742+0.j,  0.8288328 +0.j, -0.5253654 +0.j,
        -0.11065983+0.j],
       [-0.46480736+0.j,  0.43714654+0.j,  0.8159359 +0.j,
         0.547376  +0.j],
       [-0.52583736+0.j,  0.04545998+0.j, -0.0557748 +0.j,
        -0.7627722 +0.j],
       [-0.5868672 +0.j, -0.34622627+0.j, -0.23479532+0.j,
         0.32605612+0.j]], dtype=complex64), array([[-0.114176415+0.j,  0.7327782  +0.j, -0.5364275  +0.j,
         0.15489015 +0.j],
       [-0.33000445 +0.j,  0.28974816 +0.j,  0.6327556  +0.j,
         0.18506403 +0.j],
       [-0.54583275 +0.j, -0.15328142 +0.j,  0.34377125 +0.j,
        -0.83479893 +0.j],
       [-0.761661   +0.j, -0.5963111  +0.j, -0.44009918 +0.j,
         0.49484456 +0.j]], dtype=complex64)),
    mlir_module_text=r"""
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<4x4xcomplex<f32>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<4x4xcomplex<f32>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<16xf32> loc(#loc3)
    %1 = stablehlo.reshape %0 : (tensor<16xf32>) -> tensor<4x4xf32> loc(#loc4)
    %c = stablehlo.constant dense<4> : tensor<i64> loc(#loc5)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc5)
    %2:5 = stablehlo.custom_call @lapack_sgeev_ffi(%1) {mhlo.backend_config = {compute_left = 86 : ui8, compute_right = 86 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<i32>) loc(#loc5)
    %3 = stablehlo.complex %2#0, %2#1 : tensor<4xcomplex<f32>> loc(#loc5)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %5 = stablehlo.compare  EQ, %2#4, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc5)
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc5)
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4xcomplex<f32>> loc(#loc5)
    %8 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc5)
    %9 = stablehlo.select %8, %3, %7 : tensor<4xi1>, tensor<4xcomplex<f32>> loc(#loc5)
    %10 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %cst_2 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc5)
    %11 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc5)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc5)
    %13 = stablehlo.select %12, %2#2, %11 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc5)
    %14 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %cst_3 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc5)
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc5)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc5)
    %17 = stablehlo.select %16, %2#3, %15 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc5)
    return %9, %13, %17 : tensor<4xcomplex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":210:14)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":211:13)
#loc3 = loc("jit(func)/jit(main)/iota[dtype=float32 shape=(16,) dimension=0]"(#loc1))
#loc4 = loc("jit(func)/jit(main)/reshape[new_sizes=(4, 4) dimensions=None]"(#loc1))
#loc5 = loc("jit(func)/jit(main)/eig[compute_left_eigenvectors=True compute_right_eigenvectors=True]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01%\x05\x01\x03\x01\x03\x05\x03\x15\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x03\xf3\xa5;\x01]\x0f\x13\x07\x0b\x0b\x13\x0f\x0b\x17\x0b\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x03I\x0b\x0b\x0b\x0bO\x0f/\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0f/\x0b\x0b\x0b\x0b\x1b\x0b\x0b\x0f\x1f\x0f\x1f\x0f\x0b\x0b//O\x01\x05\x0b\x0f\x037\x17\x07\x07\x13\x07\x0f\x0f\x0b\x0f\x07\x13\x17\x17\x1b\x13\x17\x07\x07\x13\x13\x13\x13\x0f\x13\x13\x13\x13\x02\xae\x06\x1d;=\x03\x03\t\x99\x1f\x05\x1b\x05\x1d\x03\x03\x07\x9f\x11\x03\x05\x05\x1f\x17\x13J\x03\x1d\x05!\x03\x03\x07\x81\x03\x03\t\xa3\x03\t\x1b\x1d\x1f\r!\r\x0f#\x05#\x11\x01\x00\x05%\x05'\x05)\x03\x0b'])k+m\x0f{-}\x05+\x05-\x05/\x051\x03\x031\x7f\x053\x1d5\x11\x055\x1d9\x11\x057\x059\x17\x13N\x03\x1b\x03\x13A\x83C\x85E\x87G]I\x89K\x8bM\x91O]Q\x93\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x05I\x05K\x03\x03\x07\x97\x03\x05W\x9bY\x9d\x05M\x05O\x03\x03\t\xa1\x03\x01\x1dQ\x1dS\x1dU\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13'V\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07osw\r\x05_qac\x1dW\r\x05_uac\x1dY\r\x05_yac\x1d[\x1d]\x1d_\x13\x07\x01\x1f\x15\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1da\x1dc\x05\x01\r\x05\x8dg\x8fg\x1de\x1dg\x03\x03e\x03\x0biiee\x95\x1f-\x01\x1f\x0f\t\x00\x00\x00\x00\x1f/\x01\t\x07\x07\x01\x1f\x11\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f9!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x13\x1d\x01)\x03\x11\x13\t)\x01%)\x01\x13\x03\r)\x01\x07\x13)\x03\x11\r)\x05\x05\x05\t)\x05\x11\x11\t\x11\x01\x07\x0b\x05\x05)\x03A\r)\x05\x11\x11\r\x1b!)\x03\t\x17)\x03\x05\x17)\x03\x01\x17)\x03\x01\x07)\x01\t)\x03\x05\t)\x03\x11\t)\x03\x05\x07)\x03\t\x07\x04F\x03\x05\x01\x11\x05\x19\x07\x03\x01\x05\t\x11\x05%\x07\x039e\x0b\x033/\x03!\r\x067\x03#\x03\x01\x05\x03\x01\x15\x03\x15\x05\x03\x01\x15\x03\x15\x0f\x07\x01?\x0b\x19\x19\x05\x05\x0f\x03\x03\x11\x06\x01\x03\x0b\x05\t\x0b\x05\x03\x01S\x03\x0f\x03\x07\x01\x03\x03\x0f\x03\x15\x13\x07\x01U\x031\x05\x11\x17\x03\x07\x01\x03\x033\x03\x19\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x0b\x03\x1d\x03\x07\x01[\x035\x03\x1b\x07\x06\x01\x03\x0b\x07!\x13\x1f\x03\x07\x01\x03\x03\x1b\x03\x19\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x05\x03'\x03\x07\x01\x17\x03\x1d\x03%\x07\x06\x01\x03\x05\x07+\r)\x03\x07\x01\x03\x03\x1b\x03\x19\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x05\x031\x03\x07\x01\x17\x03\x1d\x03/\x07\x06\x01\x03\x05\x075\x0f3\x15\x04\x05\x07#-7\x06\x03\x01\x05\x01\x00\x82\x0ei\x1d\x1b#\x03\x0f\x0b\t\t\t\x11#!+\x1b\x1f/!)!)#\x1f\x19\xb1}\x81\x1f\x1f\x15\x1d\x15\x13%)9i\x13+\r\x15\x17\x17\x1f\x17\x11\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00custom_call_v1\x00complex_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(func)/jit(main)/iota[dtype=float32 shape=(16,) dimension=0]\x00jit(func)/jit(main)/reshape[new_sizes=(4, 4) dimensions=None]\x00jit(func)/jit(main)/eig[compute_left_eigenvectors=True compute_right_eigenvectors=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_sgeev_ffi\x00compute_left\x00compute_right\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_19["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgeev_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([ 3.2464249196572972e+01+0.j, -2.4642491965729789e+00+0.j,
       -1.4885423746029788e-15+0.j,  4.7495173217146935e-16+0.j]), array([[-0.40377749076862246 +0.j, -0.8288327563197503  +0.j,
        -0.541090767303977   +0.j,  0.10767692008040902 +0.j],
       [-0.4648073711584901  +0.j, -0.43714638836388775 +0.j,
         0.7847911174458492  +0.j, -0.5438508504687168  +0.j],
       [-0.5258372515483576  +0.j, -0.045460020408024666+0.j,
         0.05369006702023438 +0.j,  0.7646709406962073  +0.j],
       [-0.5868671319382248  +0.j,  0.34622634754783854 +0.j,
        -0.2973904171621061  +0.j, -0.32849701030789913 +0.j]]), array([[-0.11417645138733848+0.j, -0.7327780959803556 +0.j,
        -0.5370341524353898 +0.j, -0.0849751818967924 +0.j],
       [-0.33000459866554754+0.j, -0.2897483523969262 +0.j,
         0.6357878989446506 +0.j, -0.29000500336734825+0.j],
       [-0.545832745943757  +0.j,  0.15328139118650214+0.j,
         0.33952665941686755+0.j,  0.8349355524250736 +0.j],
       [-0.7616608932219664 +0.j,  0.5963111347699303 +0.j,
        -0.43828040592612855+0.j, -0.45995536716093305+0.j]])),
    mlir_module_text=r"""
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<4x4xcomplex<f64>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<4x4xcomplex<f64>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<16xf64> loc(#loc3)
    %1 = stablehlo.reshape %0 : (tensor<16xf64>) -> tensor<4x4xf64> loc(#loc4)
    %c = stablehlo.constant dense<4> : tensor<i64> loc(#loc5)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc5)
    %2:5 = stablehlo.custom_call @lapack_dgeev_ffi(%1) {mhlo.backend_config = {compute_left = 86 : ui8, compute_right = 86 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], result_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>>, tensor<i32>) loc(#loc5)
    %3 = stablehlo.complex %2#0, %2#1 : tensor<4xcomplex<f64>> loc(#loc5)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %4 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %5 = stablehlo.compare  EQ, %2#4, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc5)
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc5)
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4xcomplex<f64>> loc(#loc5)
    %8 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc5)
    %9 = stablehlo.select %8, %3, %7 : tensor<4xi1>, tensor<4xcomplex<f64>> loc(#loc5)
    %10 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %cst_2 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc5)
    %11 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc5)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc5)
    %13 = stablehlo.select %12, %2#2, %11 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc5)
    %14 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %cst_3 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc5)
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc5)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc5)
    %17 = stablehlo.select %16, %2#3, %15 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc5)
    return %9, %13, %17 : tensor<4xcomplex<f64>>, tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":210:14)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":211:13)
#loc3 = loc("jit(func)/jit(main)/iota[dtype=float64 shape=(16,) dimension=0]"(#loc1))
#loc4 = loc("jit(func)/jit(main)/reshape[new_sizes=(4, 4) dimensions=None]"(#loc1))
#loc5 = loc("jit(func)/jit(main)/eig[compute_left_eigenvectors=True compute_right_eigenvectors=True]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01%\x05\x01\x03\x01\x03\x05\x03\x15\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x03\xf3\xa5;\x01]\x0f\x13\x07\x0b\x0b\x13\x0f\x0b\x17\x0b\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x03I\x0b\x0b\x0b\x0bO\x0f/\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0f/\x0b\x0b\x0b\x0b\x1b\x0b\x0b\x0f\x1f\x0f\x1f\x0f\x0b\x0bO/O\x01\x05\x0b\x0f\x037\x17\x07\x07\x13\x07\x0f\x0f\x0b\x0f\x07\x13\x17\x17\x1b\x13\x17\x07\x07\x13\x13\x13\x13\x0f\x13\x13\x13\x13\x02\xce\x06\x1d;=\x03\x03\t\x99\x1f\x05\x1b\x05\x1d\x03\x03\x07\x9f\x11\x03\x05\x05\x1f\x17\x13J\x03\x1d\x05!\x03\x03\x07\x81\x03\x03\t\xa3\x03\t\x1b\x1d\x1f\r!\r\x0f#\x05#\x11\x01\x00\x05%\x05'\x05)\x03\x0b'])k+m\x0f{-}\x05+\x05-\x05/\x051\x03\x031\x7f\x053\x1d5\x11\x055\x1d9\x11\x057\x059\x17\x13N\x03\x1b\x03\x13A\x83C\x85E\x87G]I\x89K\x8bM\x91O]Q\x93\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x05I\x05K\x03\x03\x07\x97\x03\x05W\x9bY\x9d\x05M\x05O\x03\x03\t\xa1\x03\x01\x1dQ\x1dS\x1dU\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x13'V\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07osw\r\x05_qac\x1dW\r\x05_uac\x1dY\r\x05_yac\x1d[\x1d]\x1d_\x13\x07\x01\x1f\x15\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1da\x1dc\x05\x01\r\x05\x8dg\x8fg\x1de\x1dg\x03\x03e\x03\x0biiee\x95\x1f-\x01\x1f\x0f\t\x00\x00\x00\x00\x1f/\x01\t\x07\x07\x01\x1f\x11!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f9!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x13\x1d\x01)\x03\x11\x13\x0b)\x01%)\x01\x13\x03\r)\x01\x07\x13)\x03\x11\r)\x05\x05\x05\t)\x05\x11\x11\t\x11\x01\x07\x0b\x05\x05)\x03A\r)\x05\x11\x11\r\x1b!)\x03\t\x17)\x03\x05\x17)\x03\x01\x17)\x03\x01\x07)\x01\t)\x03\x05\t)\x03\x11\t)\x03\x05\x07)\x03\t\x07\x04F\x03\x05\x01\x11\x05\x19\x07\x03\x01\x05\t\x11\x05%\x07\x039e\x0b\x033/\x03!\r\x067\x03#\x03\x01\x05\x03\x01\x15\x03\x15\x05\x03\x01\x15\x03\x15\x0f\x07\x01?\x0b\x19\x19\x05\x05\x0f\x03\x03\x11\x06\x01\x03\x0b\x05\t\x0b\x05\x03\x01S\x03\x0f\x03\x07\x01\x03\x03\x0f\x03\x15\x13\x07\x01U\x031\x05\x11\x17\x03\x07\x01\x03\x033\x03\x19\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x0b\x03\x1d\x03\x07\x01[\x035\x03\x1b\x07\x06\x01\x03\x0b\x07!\x13\x1f\x03\x07\x01\x03\x03\x1b\x03\x19\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x05\x03'\x03\x07\x01\x17\x03\x1d\x03%\x07\x06\x01\x03\x05\x07+\r)\x03\x07\x01\x03\x03\x1b\x03\x19\x05\x03\x01\x0b\x03\x11\x03\x07\x01\x03\x03\x05\x031\x03\x07\x01\x17\x03\x1d\x03/\x07\x06\x01\x03\x05\x075\x0f3\x15\x04\x05\x07#-7\x06\x03\x01\x05\x01\x00\x82\x0ei\x1d\x1b#\x03\x0f\x0b\t\t\t\x11#!+\x1b\x1f/!)!)#\x1f\x19\xb1}\x81\x1f\x1f\x15\x1d\x15\x13%)9i\x13+\r\x15\x17\x17\x1f\x17\x11\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00custom_call_v1\x00complex_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(func)/jit(main)/iota[dtype=float64 shape=(16,) dimension=0]\x00jit(func)/jit(main)/reshape[new_sizes=(4, 4) dimensions=None]\x00jit(func)/jit(main)/eig[compute_left_eigenvectors=True compute_right_eigenvectors=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_dgeev_ffi\x00compute_left\x00compute_right\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
