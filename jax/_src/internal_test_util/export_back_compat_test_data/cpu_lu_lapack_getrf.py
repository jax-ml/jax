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
int32 = np.int32
float32 = np.float32
complex64 = np.complex64

data_2024_05_31 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgetrf_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[6. +0.j, 7. +0.j, 8. +0.j],
       [0. +0.j, 1. +0.j, 2. +0.j],
       [0.5+0.j, 0.5+0.j, 0. +0.j]]), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":384:11)
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3))
#loc21 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f64>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:3 = stablehlo.custom_call @lapack_zgetrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x3xcomplex<f64>>) -> (tensor<3x3xcomplex<f64>>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>> loc(#loc6)
    %11 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc7)
    %c_3 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %c_4 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
    %12:4 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_3, %iterArg_6 = %11, %iterArg_7 = %4) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %c_8 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %13 = stablehlo.compare  LT, %iterArg, %c_8,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc11)
      stablehlo.return %13 : tensor<i1> loc(#loc10)
    } do {
      %13:3 = func.call @None(%iterArg_5, %iterArg_6, %iterArg_7) : (tensor<i64>, tensor<3xi32>, tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>, tensor<3xi32>) loc(#loc12)
      %c_8 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %14 = stablehlo.add %iterArg, %c_8 : tensor<i64> loc(#loc13)
      stablehlo.return %14, %13#0, %13#1, %13#2 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc10)
    } loc(#loc10)
    return %10, %4, %12#2 : tensor<3x3xcomplex<f64>>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
  func.func private @None(%arg0: tensor<i64> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3)), %arg1: tensor<3xi32> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3)), %arg2: tensor<3xi32> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3))) -> (tensor<i64>, tensor<3xi32>, tensor<3xi32>) {
    %c = stablehlo.constant dense<1> : tensor<i64> loc(#loc12)
    %0 = stablehlo.add %arg0, %c : tensor<i64> loc(#loc13)
    %c_0 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %1 = stablehlo.compare  LT, %arg0, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_1 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %2 = stablehlo.add %arg0, %c_1 : tensor<i64> loc(#loc13)
    %3 = stablehlo.select %1, %2, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %6 = "stablehlo.gather"(%arg2, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc18)
    %c_2 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %7 = stablehlo.compare  LT, %arg0, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_3 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %8 = stablehlo.add %arg0, %c_3 : tensor<i64> loc(#loc13)
    %9 = stablehlo.select %7, %8, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %10 = stablehlo.convert %9 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %12 = "stablehlo.gather"(%arg1, %11) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc18)
    %c_4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc12)
    %13 = stablehlo.compare  LT, %6, %c_4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %c_5 = stablehlo.constant dense<3> : tensor<i32> loc(#loc12)
    %14 = stablehlo.add %6, %c_5 : tensor<i32> loc(#loc13)
    %15 = stablehlo.select %13, %14, %6 : tensor<i1>, tensor<i32> loc(#loc15)
    %16 = stablehlo.dynamic_slice %arg1, %15, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc19)
    %17 = stablehlo.reshape %16 : (tensor<1xi32>) -> tensor<i32> loc(#loc20)
    %c_6 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %18 = stablehlo.compare  LT, %arg0, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_7 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %19 = stablehlo.add %arg0, %c_7 : tensor<i64> loc(#loc13)
    %20 = stablehlo.select %18, %19, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %21 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %23 = "stablehlo.scatter"(%arg1, %22, %17) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg4: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
      stablehlo.return %arg4 : tensor<i32> loc(#loc21)
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc21)
    %c_8 = stablehlo.constant dense<0> : tensor<i32> loc(#loc12)
    %24 = stablehlo.compare  LT, %6, %c_8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %c_9 = stablehlo.constant dense<3> : tensor<i32> loc(#loc12)
    %25 = stablehlo.add %6, %c_9 : tensor<i32> loc(#loc13)
    %26 = stablehlo.select %24, %25, %6 : tensor<i1>, tensor<i32> loc(#loc15)
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %28 = "stablehlo.scatter"(%23, %27, %12) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg4: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
      stablehlo.return %arg4 : tensor<i32> loc(#loc21)
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc21)
    return %0, %28, %arg2 : tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc12)
  } loc(#loc12)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":383:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":383:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc3))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc3))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc3))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc3))
#loc20 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc3))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\'\x03\xda\x02>\x025\x01\xa9\x0f\x17\x0f\x13\x13\x0f\x0b\x0f\x1b\x13\x13\x0f\x0f\x0f\x0b\x07\x0b\x0f\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b;\x0b\x0b\x0b\x0f\x13;\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x03]\x0b/\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x0b\x0b/O\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x17\x0b//\x0b/\x1f\x1f\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x1f\x0bOO\x01\x1b\x17\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x17\x0b\x13\x0b\x01\x05\x0b\x0f\x031\x0f\x0f\x13\x0f\x17\x07\x13\x0b\x07\x07\x07\x13\x0f\x1b\x07\'\x13\x13\x13\x13\x13\x17\x17\x13\x02^\n\x1dY\x03\x17!\x02\x06\x17\x1d\x8f\x03\x1d*\x02\x03\x03\x037\xb5\x1d\x7f\x03\x05)\x1d[\x03\x03\x053\xbd5\xe1\x03\x03\r\xe3\x03\x03\r\xdf\x1d]\x03\x1d_\x03\x1dc\x03\x05+\x1f\x05-\x1da\x03\x03\x03\r\xe5\x11\x03\x05\x05/\x051\x053\x055\x03\x03\r\xdd\x057\x059\x05;\x03\re\xab;\xb7=\xb9g\xb5?\xbfi\xab\x05=\x05?\x05A\x1dk\x03\x03\x03\r\xe7\x03\r;\xb7=\xb9w\xaby\xab{\xb9}\xb5\x03\x03\x81\xb7\x03\tKMO\'Q\'\x1dS\x05C\x11\x01\x00\x05E\x05G\x05I\x03\x0b)\xad+\xc3-\xc5\x1d\xd3/\xd5\x03\x0b)\xad+\xd7-\xd9\x1d\xbb/\xdb\x05K\x05M\x05O\x05Q\x05S\x05U\x05W\x05Y\x05[\x05]\x03\x03?\xbf\x1dq\x03\x05_\x1du\x03\x05a\x05c\x05e\x05g\x05i\x05k\x05m\x1d\x85\x87\x05o\x17!\xfe\x055\x1d\x8b\x8d\x05q\x17!\xfe\x05\x1d\x05s\x03\x13\x93\xe9\x95\xeb\x97\xed\x99\xad\x9b\xef\x9d\xa9\x9f\xf1\xa1\xf3\xa3\xf7\x05u\x05w\x05y\x05{\x05}\x05\x7f\x05\x81\x05\x83\x05\x85\x03\x03\r\xfd\x03\x053\xbd5\xff\r\x01\x1f\x1b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d\x87\x1d\x89\x1d\x8b\x1f%\x01\x13\x0f\x01\x05\x03\x1d\x8d\t\x07\x1f\x1b\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc7\xcb\xcf\r\x05\xaf\xc9\xb1\xb3\x1d\x8f\r\x05\xaf\xcd\xb1\xb3\x1d\x91\r\x05\xaf\xd1\xb1\xb3\x1d\x93\x1d\x95\x1d\x97##\x03\x07\xa9\xa9\xa9\x1d\x99\x1f\x05\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x00\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x1f\x05\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00\x00\x1f\x07\t\x03\x00\x00\x00\x0b\x03\x1d\x9b\x1d\x9d\x05\x01\x03\x03\xc1\x03\x03\xf5\x15\x03\x01\x01\x01\x03\x07\xc1\xf9\xfb\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x01\x1f\x07\t\x01\x00\x00\x00\x07\x05\x1f\x1d!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03\r\x02\x02\x03\x037\x06\x02\x1d\x16\x02\x03\x05\x9f\x1d\x1e\x02\x03\x05\xa1\x1d&\x02\x03\x05\xa3\x05\xa5\x03\x032\x02\xbb\x05\xa7\x1d:\x02\x03\x05\xa9\x01\t\x01\x02\x02)\x01\x0f)\x01\x15)\x03\r\x15)\x01\x17)\x05\r\r\x13\x1d)\x03\x05\x15\x03!\x1b\x01\x13)\x03\x05\x0f)\x01\x13\x11\x01\x07\r\t\t\x0b\x11\x07\x05\t\t\x07\x05\t\t)\x03\x01\x0f)\x03%\x13)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x05\x05\x05\x17)\x05\r\r\x17)\x03\t\x0f\x04\xb6\t\x05\x01\x11\x1fI\x07\x03\x01\t\x11\x11\x1fU\x07\x033U\x19\x03\x83G\x03\'\x15\x06\x89\x03\r\x03\x01\x03\x03\x05\x13\x03\x05\x03\x03\x05\x13\x03\x05\x1d\x07\x05\x91\x07\r\t\x07\x03\x03\x03\x03\x05\xa5\x03\x07\x05\x07\x05\t\x03\t\x03\x0f\x1f\x06\x05\x03\t\x05\x0b\x11\x03\x03\x05%\x03\x07\x05\x07\x05\t\x03\x07\x03\x15\t\x07\x05\xa7\x03\x0b\x05\r\x17\x05\x07\x05\t\x03/\x03\x19\x03\x03\x05\n\x02\x03\x1d\x05\x07\x05\t\x03\r\x03\x1d\x05\x07\x05\x0e\x02\x031\x03\x1b\x0b\x06\x05\x03\r\x07!\t\x1f\x19\x03\x12\x02G\x03\t\x03\x03\x1a\x02\x15\x03\x05\x03\x03"\x02\x15\x03\x05!\x16\x07\t\x05\x05\t\t\t)\'%\x13\x0b\x03\r\x0f\t\x05\x07\x05\x07\t\x07\t\x07\x03\x03\x07\x13\x03\x05\t\x076\x02\x11\x03\x0b\x05\x01\t\r\x04\x07\x03\x0b\x03\x13\x13\t\x05\x07\x05\x07\t\x07\t\x07#\x07\x01.\x02\x07\x05\t\t\x07\x03\x05\x07\x03\x03\x071\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x0f\r\x04\x07\t\x11\t\x0b\r\r\x04\x1f\x07#\x13/\x11\x11\x01W\x07\x03W\xa7\x07\x05\x01\t\x01\t\x01\x03\x03\x011\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x07\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x01\x0b\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x0f\x0b\x06\x19\x03\x05\x07\r\x11\x01\x0f\x06#\x03\x07\x03\x13\x05\x07\x1b\t\x03\x11\x03\x15\x13\x07A9\x03\x07\x05\x05\x17\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x01\x1b\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x1f\x0b\x06\x19\x03\x05\x07\x1d!\x01\x0f\x06#\x03\x07\x03#\x05\x07\x1b\t\x03\x11\x03%\x13\x07A9\x03\x07\x05\x03\'\x03\x03\x01%\x03\x07\t\x07\x17\x11\x03\x0b\x05\x19+\x03\x03\x01C\x03\x07\x07\x06\x0f\x03\x07\x05\x19/\x0b\x06\x19\x03\x07\x07-1\x19\x1b\x07om\x03\x11\x05\x033\x15\x06s\x03\x07\x035\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x019\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01=\x0b\x06\x19\x03\x05\x07;?\x01\x0f\x06#\x03\x07\x03A\x05\x07\x1b\t\x03\x11\x03C\x17\x17\x0bE\x03\t\x07\x03E7\x07\x03\x05\x07\x05\x07\x0b\x07\x0b\r\x04\x0b\x03\x03\x03\x03\x01%\x03\x07\t\x07\x17\x11\x03\x0b\x05\x19I\x03\x03\x01C\x03\x07\x07\x06\x0f\x03\x07\x05\x19M\x0b\x06\x19\x03\x07\x07KO\x19\x05\x07\x1b\t\x03\x11\x03Q\x17\x17\x0bE\x03\t\x07GS)\x07\x03\x05\x07\x05\x07\x0b\x07\x0b\r\x04\x0b\x03\x03\r\x04\x01\x07\tU\x05\x06\x03\x01\x05\x01\x00\x0e(\xabM\x0f{.\x02\x8b\x83%\x03\x11\x0f\x0b\t\t\t\x0b\x11#!\x1f/!)!)#\x1f\x197\x85\x8d\x1fz\x04\'\x1f;+y\x87.\x04!\x19+\xb1\xb3YMO_\x1b%)9\x19\'#++\x1b\x1f\x15\x1d\x15i\x13\r\x11\x13\x19\x1f#\x11\x17\x17\x15\x11\x17\x15\x15\x17\x0f)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00convert_v1\x00func_v1\x00gather_v1\x00reshape_v1\x00scatter_v1\x00iota_v1\x00dynamic_slice_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00compare_type\x00comparison_direction\x00broadcast_dimensions\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/while/body/closed_call\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00mhlo.layout_mode\x00default\x00None\x00[0]\x00[1]\x00[2]\x00main\x00public\x00private\x00\x00lapack_zgetrf_ffi\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00callee\x00jit(<lambda>)/jit(main)/while/cond/lt\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgetrf_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[6. +0.j, 7. +0.j, 8. +0.j],
       [0. +0.j, 1. +0.j, 2. +0.j],
       [0.5+0.j, 0.5+0.j, 0. +0.j]], dtype=complex64), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":384:11)
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3))
#loc21 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f32>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:3 = stablehlo.custom_call @lapack_cgetrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x3xcomplex<f32>>) -> (tensor<3x3xcomplex<f32>>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>> loc(#loc6)
    %11 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc7)
    %c_3 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %c_4 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
    %12:4 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_3, %iterArg_6 = %11, %iterArg_7 = %4) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %c_8 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %13 = stablehlo.compare  LT, %iterArg, %c_8,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc11)
      stablehlo.return %13 : tensor<i1> loc(#loc10)
    } do {
      %13:3 = func.call @None(%iterArg_5, %iterArg_6, %iterArg_7) : (tensor<i64>, tensor<3xi32>, tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>, tensor<3xi32>) loc(#loc12)
      %c_8 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %14 = stablehlo.add %iterArg, %c_8 : tensor<i64> loc(#loc13)
      stablehlo.return %14, %13#0, %13#1, %13#2 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc10)
    } loc(#loc10)
    return %10, %4, %12#2 : tensor<3x3xcomplex<f32>>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
  func.func private @None(%arg0: tensor<i64> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3)), %arg1: tensor<3xi32> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3)), %arg2: tensor<3xi32> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3))) -> (tensor<i64>, tensor<3xi32>, tensor<3xi32>) {
    %c = stablehlo.constant dense<1> : tensor<i64> loc(#loc12)
    %0 = stablehlo.add %arg0, %c : tensor<i64> loc(#loc13)
    %c_0 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %1 = stablehlo.compare  LT, %arg0, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_1 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %2 = stablehlo.add %arg0, %c_1 : tensor<i64> loc(#loc13)
    %3 = stablehlo.select %1, %2, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %6 = "stablehlo.gather"(%arg2, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc18)
    %c_2 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %7 = stablehlo.compare  LT, %arg0, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_3 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %8 = stablehlo.add %arg0, %c_3 : tensor<i64> loc(#loc13)
    %9 = stablehlo.select %7, %8, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %10 = stablehlo.convert %9 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %12 = "stablehlo.gather"(%arg1, %11) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc18)
    %c_4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc12)
    %13 = stablehlo.compare  LT, %6, %c_4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %c_5 = stablehlo.constant dense<3> : tensor<i32> loc(#loc12)
    %14 = stablehlo.add %6, %c_5 : tensor<i32> loc(#loc13)
    %15 = stablehlo.select %13, %14, %6 : tensor<i1>, tensor<i32> loc(#loc15)
    %16 = stablehlo.dynamic_slice %arg1, %15, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc19)
    %17 = stablehlo.reshape %16 : (tensor<1xi32>) -> tensor<i32> loc(#loc20)
    %c_6 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %18 = stablehlo.compare  LT, %arg0, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_7 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %19 = stablehlo.add %arg0, %c_7 : tensor<i64> loc(#loc13)
    %20 = stablehlo.select %18, %19, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %21 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %23 = "stablehlo.scatter"(%arg1, %22, %17) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg4: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
      stablehlo.return %arg4 : tensor<i32> loc(#loc21)
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc21)
    %c_8 = stablehlo.constant dense<0> : tensor<i32> loc(#loc12)
    %24 = stablehlo.compare  LT, %6, %c_8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %c_9 = stablehlo.constant dense<3> : tensor<i32> loc(#loc12)
    %25 = stablehlo.add %6, %c_9 : tensor<i32> loc(#loc13)
    %26 = stablehlo.select %24, %25, %6 : tensor<i1>, tensor<i32> loc(#loc15)
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %28 = "stablehlo.scatter"(%23, %27, %12) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg4: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
      stablehlo.return %arg4 : tensor<i32> loc(#loc21)
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc21)
    return %0, %28, %arg2 : tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc12)
  } loc(#loc12)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":383:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":383:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc3))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc3))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc3))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc3))
#loc20 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc3))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\'\x03\xda\x02>\x025\x01\xa9\x0f\x17\x0f\x13\x13\x0f\x0b\x0f\x1b\x13\x13\x0f\x0f\x0f\x0b\x07\x0b\x0f\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b;\x0b\x0b\x0b\x0f\x13;\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x03]\x0b/\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x0b\x0b/O\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x17\x0b//\x0b/\x1f\x1f\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x1f\x0b/O\x01\x1b\x17\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x17\x0b\x13\x0b\x01\x05\x0b\x0f\x031\x0f\x0f\x13\x0f\x17\x07\x13\x0b\x07\x07\x07\x13\x0f\x1b\x07\'\x13\x13\x13\x13\x13\x17\x17\x13\x02>\n\x1dY\x03\x17!\x02\x06\x17\x1d\x8f\x03\x1d*\x02\x03\x03\x037\xb5\x1d\x7f\x03\x05)\x1d[\x03\x03\x053\xbd5\xe1\x03\x03\r\xe3\x03\x03\r\xdf\x1d]\x03\x1d_\x03\x1dc\x03\x05+\x1f\x05-\x1da\x03\x03\x03\r\xe5\x11\x03\x05\x05/\x051\x053\x055\x03\x03\r\xdd\x057\x059\x05;\x03\re\xab;\xb7=\xb9g\xb5?\xbfi\xab\x05=\x05?\x05A\x1dk\x03\x03\x03\r\xe7\x03\r;\xb7=\xb9w\xaby\xab{\xb9}\xb5\x03\x03\x81\xb7\x03\tKMO\'Q\'\x1dS\x05C\x11\x01\x00\x05E\x05G\x05I\x03\x0b)\xad+\xc3-\xc5\x1d\xd3/\xd5\x03\x0b)\xad+\xd7-\xd9\x1d\xbb/\xdb\x05K\x05M\x05O\x05Q\x05S\x05U\x05W\x05Y\x05[\x05]\x03\x03?\xbf\x1dq\x03\x05_\x1du\x03\x05a\x05c\x05e\x05g\x05i\x05k\x05m\x1d\x85\x87\x05o\x17!\xfe\x055\x1d\x8b\x8d\x05q\x17!\xfe\x05\x1d\x05s\x03\x13\x93\xe9\x95\xeb\x97\xed\x99\xad\x9b\xef\x9d\xa9\x9f\xf1\xa1\xf3\xa3\xf7\x05u\x05w\x05y\x05{\x05}\x05\x7f\x05\x81\x05\x83\x05\x85\x03\x03\r\xfd\x03\x053\xbd5\xff\r\x01\x1f\x1b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d\x87\x1d\x89\x1d\x8b\x1f%\x01\x13\x0f\x01\x05\x03\x1d\x8d\t\x07\x1f\x1b\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc7\xcb\xcf\r\x05\xaf\xc9\xb1\xb3\x1d\x8f\r\x05\xaf\xcd\xb1\xb3\x1d\x91\r\x05\xaf\xd1\xb1\xb3\x1d\x93\x1d\x95\x1d\x97##\x03\x07\xa9\xa9\xa9\x1d\x99\x1f\x05\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x00\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x1f\x05\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00\x00\x1f\x07\t\x03\x00\x00\x00\x0b\x03\x1d\x9b\x1d\x9d\x05\x01\x03\x03\xc1\x03\x03\xf5\x15\x03\x01\x01\x01\x03\x07\xc1\xf9\xfb\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x01\x1f\x07\t\x01\x00\x00\x00\x07\x05\x1f\x1d\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03\r\x02\x02\x03\x037\x06\x02\x1d\x16\x02\x03\x05\x9f\x1d\x1e\x02\x03\x05\xa1\x1d&\x02\x03\x05\xa3\x05\xa5\x03\x032\x02\xbb\x05\xa7\x1d:\x02\x03\x05\xa9\x01\t\x01\x02\x02)\x01\x0f)\x01\x15)\x03\r\x15)\x01\x17)\x05\r\r\x13\x1d)\x03\x05\x15\x03!\x1b\x01\x13)\x03\x05\x0f)\x01\x13\x11\x01\x07\r\t\t\t\x11\x07\x05\t\t\x07\x05\t\t)\x03\x01\x0f)\x03%\x13)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x05\x05\x05\x17)\x05\r\r\x17)\x03\t\x0f\x04\xb6\t\x05\x01\x11\x1fI\x07\x03\x01\t\x11\x11\x1fU\x07\x033U\x19\x03\x83G\x03\'\x15\x06\x89\x03\r\x03\x01\x03\x03\x05\x13\x03\x05\x03\x03\x05\x13\x03\x05\x1d\x07\x05\x91\x07\r\t\x07\x03\x03\x03\x03\x05\xa5\x03\x07\x05\x07\x05\t\x03\t\x03\x0f\x1f\x06\x05\x03\t\x05\x0b\x11\x03\x03\x05%\x03\x07\x05\x07\x05\t\x03\x07\x03\x15\t\x07\x05\xa7\x03\x0b\x05\r\x17\x05\x07\x05\t\x03/\x03\x19\x03\x03\x05\n\x02\x03\x1d\x05\x07\x05\t\x03\r\x03\x1d\x05\x07\x05\x0e\x02\x031\x03\x1b\x0b\x06\x05\x03\r\x07!\t\x1f\x19\x03\x12\x02G\x03\t\x03\x03\x1a\x02\x15\x03\x05\x03\x03"\x02\x15\x03\x05!\x16\x07\t\x05\x05\t\t\t)\'%\x13\x0b\x03\r\x0f\t\x05\x07\x05\x07\t\x07\t\x07\x03\x03\x07\x13\x03\x05\t\x076\x02\x11\x03\x0b\x05\x01\t\r\x04\x07\x03\x0b\x03\x13\x13\t\x05\x07\x05\x07\t\x07\t\x07#\x07\x01.\x02\x07\x05\t\t\x07\x03\x05\x07\x03\x03\x071\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x0f\r\x04\x07\t\x11\t\x0b\r\r\x04\x1f\x07#\x13/\x11\x11\x01W\x07\x03W\xa7\x07\x05\x01\t\x01\t\x01\x03\x03\x011\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x07\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x01\x0b\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x0f\x0b\x06\x19\x03\x05\x07\r\x11\x01\x0f\x06#\x03\x07\x03\x13\x05\x07\x1b\t\x03\x11\x03\x15\x13\x07A9\x03\x07\x05\x05\x17\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x01\x1b\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x1f\x0b\x06\x19\x03\x05\x07\x1d!\x01\x0f\x06#\x03\x07\x03#\x05\x07\x1b\t\x03\x11\x03%\x13\x07A9\x03\x07\x05\x03\'\x03\x03\x01%\x03\x07\t\x07\x17\x11\x03\x0b\x05\x19+\x03\x03\x01C\x03\x07\x07\x06\x0f\x03\x07\x05\x19/\x0b\x06\x19\x03\x07\x07-1\x19\x1b\x07om\x03\x11\x05\x033\x15\x06s\x03\x07\x035\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x019\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01=\x0b\x06\x19\x03\x05\x07;?\x01\x0f\x06#\x03\x07\x03A\x05\x07\x1b\t\x03\x11\x03C\x17\x17\x0bE\x03\t\x07\x03E7\x07\x03\x05\x07\x05\x07\x0b\x07\x0b\r\x04\x0b\x03\x03\x03\x03\x01%\x03\x07\t\x07\x17\x11\x03\x0b\x05\x19I\x03\x03\x01C\x03\x07\x07\x06\x0f\x03\x07\x05\x19M\x0b\x06\x19\x03\x07\x07KO\x19\x05\x07\x1b\t\x03\x11\x03Q\x17\x17\x0bE\x03\t\x07GS)\x07\x03\x05\x07\x05\x07\x0b\x07\x0b\r\x04\x0b\x03\x03\r\x04\x01\x07\tU\x05\x06\x03\x01\x05\x01\x00\n(\xabM\x0f{.\x02\x8b\x83%\x03\x11\x0f\x0b\t\t\t\x0b\x11#!\x1f/!)!)#\x1f\x197\x85\x8b\x1fz\x04\'\x1f;+y\x87.\x04!\x19+\xb1\xb3YMO_\x1b%)9\x19\'#++\x1b\x1f\x15\x1d\x15i\x13\r\x11\x13\x19\x1f#\x11\x17\x17\x15\x11\x17\x15\x15\x17\x0f)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00convert_v1\x00func_v1\x00gather_v1\x00reshape_v1\x00scatter_v1\x00iota_v1\x00dynamic_slice_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00compare_type\x00comparison_direction\x00broadcast_dimensions\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/while/body/closed_call\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00mhlo.layout_mode\x00default\x00None\x00[0]\x00[1]\x00[2]\x00main\x00public\x00private\x00\x00lapack_cgetrf_ffi\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00callee\x00jit(<lambda>)/jit(main)/while/cond/lt\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgetrf_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[6. , 7. , 8. ],
       [0. , 1. , 2. ],
       [0.5, 0.5, 0. ]], dtype=float32), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":384:11)
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3))
#loc21 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf32> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:3 = stablehlo.custom_call @lapack_sgetrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc6)
    %11 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc7)
    %c_3 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %c_4 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
    %12:4 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_3, %iterArg_6 = %11, %iterArg_7 = %4) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %c_8 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %13 = stablehlo.compare  LT, %iterArg, %c_8,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc11)
      stablehlo.return %13 : tensor<i1> loc(#loc10)
    } do {
      %13:3 = func.call @None(%iterArg_5, %iterArg_6, %iterArg_7) : (tensor<i64>, tensor<3xi32>, tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>, tensor<3xi32>) loc(#loc12)
      %c_8 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %14 = stablehlo.add %iterArg, %c_8 : tensor<i64> loc(#loc13)
      stablehlo.return %14, %13#0, %13#1, %13#2 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc10)
    } loc(#loc10)
    return %10, %4, %12#2 : tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
  func.func private @None(%arg0: tensor<i64> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3)), %arg1: tensor<3xi32> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3)), %arg2: tensor<3xi32> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3))) -> (tensor<i64>, tensor<3xi32>, tensor<3xi32>) {
    %c = stablehlo.constant dense<1> : tensor<i64> loc(#loc12)
    %0 = stablehlo.add %arg0, %c : tensor<i64> loc(#loc13)
    %c_0 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %1 = stablehlo.compare  LT, %arg0, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_1 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %2 = stablehlo.add %arg0, %c_1 : tensor<i64> loc(#loc13)
    %3 = stablehlo.select %1, %2, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %6 = "stablehlo.gather"(%arg2, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc18)
    %c_2 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %7 = stablehlo.compare  LT, %arg0, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_3 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %8 = stablehlo.add %arg0, %c_3 : tensor<i64> loc(#loc13)
    %9 = stablehlo.select %7, %8, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %10 = stablehlo.convert %9 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %12 = "stablehlo.gather"(%arg1, %11) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc18)
    %c_4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc12)
    %13 = stablehlo.compare  LT, %6, %c_4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %c_5 = stablehlo.constant dense<3> : tensor<i32> loc(#loc12)
    %14 = stablehlo.add %6, %c_5 : tensor<i32> loc(#loc13)
    %15 = stablehlo.select %13, %14, %6 : tensor<i1>, tensor<i32> loc(#loc15)
    %16 = stablehlo.dynamic_slice %arg1, %15, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc19)
    %17 = stablehlo.reshape %16 : (tensor<1xi32>) -> tensor<i32> loc(#loc20)
    %c_6 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %18 = stablehlo.compare  LT, %arg0, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_7 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %19 = stablehlo.add %arg0, %c_7 : tensor<i64> loc(#loc13)
    %20 = stablehlo.select %18, %19, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %21 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %23 = "stablehlo.scatter"(%arg1, %22, %17) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg4: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
      stablehlo.return %arg4 : tensor<i32> loc(#loc21)
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc21)
    %c_8 = stablehlo.constant dense<0> : tensor<i32> loc(#loc12)
    %24 = stablehlo.compare  LT, %6, %c_8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %c_9 = stablehlo.constant dense<3> : tensor<i32> loc(#loc12)
    %25 = stablehlo.add %6, %c_9 : tensor<i32> loc(#loc13)
    %26 = stablehlo.select %24, %25, %6 : tensor<i1>, tensor<i32> loc(#loc15)
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %28 = "stablehlo.scatter"(%23, %27, %12) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg4: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
      stablehlo.return %arg4 : tensor<i32> loc(#loc21)
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc21)
    return %0, %28, %arg2 : tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc12)
  } loc(#loc12)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":383:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":383:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc3))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc3))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc3))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc3))
#loc20 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc3))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\'\x03\xd6\x02>\x023\x01\xa9\x0f\x17\x0f\x13\x13\x0f\x0b\x0f\x1b\x13\x13\x0f\x0f\x0f\x0b\x07\x0b\x0f\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b;\x0b\x0b\x0b\x0f\x13;\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x03]\x0b/\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x0b\x0b/O\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x17\x0b//\x0b/\x1f\x1f\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x1f\x0b\x1fO\x01\x1b\x17\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x17\x0b\x13\x0b\x01\x05\x0b\x0f\x03/\x0f\x0f\x13\x0f\x17\x07\x13\x07\x07\x07\x07\x13\x0f\x1b\'\x13\x13\x13\x13\x13\x17\x17\x13\x02&\n\x1dY\x03\x17!\x02\x06\x17\x1d\x8f\x03\x1d*\x02\x03\x03\x037\xb5\x1d\x7f\x03\x05)\x1d[\x03\x03\x053\xbd5\xe1\x03\x03\r\xe3\x03\x03\r\xdf\x1d]\x03\x1d_\x03\x1dc\x03\x05+\x1f\x05-\x1da\x03\x03\x03\r\xe5\x11\x03\x05\x05/\x051\x053\x055\x03\x03\r\xdd\x057\x059\x05;\x03\re\xab;\xb7=\xb9g\xb5?\xbfi\xab\x05=\x05?\x05A\x1dk\x03\x03\x03\r\xe7\x03\r;\xb7=\xb9w\xaby\xab{\xb9}\xb5\x03\x03\x81\xb7\x03\tKMO\'Q\'\x1dS\x05C\x11\x01\x00\x05E\x05G\x05I\x03\x0b)\xad+\xc3-\xc5\x1d\xd3/\xd5\x03\x0b)\xad+\xd7-\xd9\x1d\xbb/\xdb\x05K\x05M\x05O\x05Q\x05S\x05U\x05W\x05Y\x05[\x05]\x03\x03?\xbf\x1dq\x03\x05_\x1du\x03\x05a\x05c\x05e\x05g\x05i\x05k\x05m\x1d\x85\x87\x05o\x17!\xfe\x055\x1d\x8b\x8d\x05q\x17!\xfe\x05\x1d\x05s\x03\x13\x93\xe9\x95\xeb\x97\xed\x99\xad\x9b\xef\x9d\xa9\x9f\xf1\xa1\xf3\xa3\xf7\x05u\x05w\x05y\x05{\x05}\x05\x7f\x05\x81\x05\x83\x05\x85\x03\x03\r\xfd\x03\x053\xbd5\xff\r\x01\x1f\x1b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d\x87\x1d\x89\x1d\x8b\x1f#\x01\x13\x0f\x01\x05\x03\x1d\x8d\t\x07\x1f\x1b\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f\'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc7\xcb\xcf\r\x05\xaf\xc9\xb1\xb3\x1d\x8f\r\x05\xaf\xcd\xb1\xb3\x1d\x91\r\x05\xaf\xd1\xb1\xb3\x1d\x93\x1d\x95\x1d\x97#!\x03\x07\xa9\xa9\xa9\x1d\x99\x1f\x05\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x00\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x1f\x05\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00\x00\x1f\x07\t\x03\x00\x00\x00\x0b\x03\x1d\x9b\x1d\x9d\x05\x01\x03\x03\xc1\x03\x03\xf5\x15\x03\x01\x01\x01\x03\x07\xc1\xf9\xfb\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x01\x1f\x07\t\x01\x00\x00\x00\x07\x05\x1f\x1d\t\x00\x00\xc0\x7f\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03\r\x02\x02\x03\x037\x06\x02\x1d\x16\x02\x03\x05\x9f\x1d\x1e\x02\x03\x05\xa1\x1d&\x02\x03\x05\xa3\x05\xa5\x03\x032\x02\xbb\x05\xa7\x1d:\x02\x03\x05\xa9\x01\t\x01\x02\x02)\x01\x0f)\x01\x15)\x03\r\x15)\x01\x17)\x05\r\r\x13\x1d)\x03\x05\x15\t\x1b\x01\x13)\x03\x05\x0f)\x01\x13\x11\x01\x07\r\t\t\x11\x07\x05\t\t\x07\x05\t\t)\x03\x01\x0f)\x03%\x13)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x05\x05\x05\x17)\x05\r\r\x17)\x03\t\x0f\x04\xb6\t\x05\x01\x11\x1fI\x07\x03\x01\t\x11\x11\x1fU\x07\x033U\x19\x03\x83G\x03%\x15\x06\x89\x03\r\x03\x01\x03\x03\x05\x13\x03\x05\x03\x03\x05\x13\x03\x05\x1d\x07\x05\x91\x07\r\t\x07\x03\x03\x03\x03\x05\xa5\x03\x07\x05\x07\x05\t\x03\t\x03\x0f\x1f\x06\x05\x03\t\x05\x0b\x11\x03\x03\x05%\x03\x07\x05\x07\x05\t\x03\x07\x03\x15\t\x07\x05\xa7\x03\x0b\x05\r\x17\x05\x07\x05\t\x03-\x03\x19\x03\x03\x05\n\x02\x03\x1d\x05\x07\x05\t\x03\r\x03\x1d\x05\x07\x05\x0e\x02\x03/\x03\x1b\x0b\x06\x05\x03\r\x07!\t\x1f\x19\x03\x12\x02G\x03\t\x03\x03\x1a\x02\x15\x03\x05\x03\x03"\x02\x15\x03\x05!\x16\x07\t\x05\x05\t\t\t)\'%\x13\x0b\x03\r\x0f\t\x05\x07\x05\x07\t\x07\t\x07\x03\x03\x07\x13\x03\x05\t\x076\x02\x11\x03\x0b\x05\x01\t\r\x04\x07\x03\x0b\x03\x13\x13\t\x05\x07\x05\x07\t\x07\t\x07#\x07\x01.\x02\x07\x05\t\t\x07\x03\x05\x07\x03\x03\x071\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x0f\r\x04\x07\t\x11\t\x0b\r\r\x04\x1f\x07#\x13/\x11\x11\x01W\x07\x03W\xa7\x07\x05\x01\t\x01\t\x01\x03\x03\x011\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x07\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x01\x0b\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x0f\x0b\x06\x19\x03\x05\x07\r\x11\x01\x0f\x06#\x03\x07\x03\x13\x05\x07\x1b\t\x03\x11\x03\x15\x13\x07A9\x03\x07\x05\x05\x17\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x01\x1b\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x1f\x0b\x06\x19\x03\x05\x07\x1d!\x01\x0f\x06#\x03\x07\x03#\x05\x07\x1b\t\x03\x11\x03%\x13\x07A9\x03\x07\x05\x03\'\x03\x03\x01%\x03\x07\t\x07\x17\x11\x03\x0b\x05\x19+\x03\x03\x01C\x03\x07\x07\x06\x0f\x03\x07\x05\x19/\x0b\x06\x19\x03\x07\x07-1\x19\x1b\x07om\x03\x11\x05\x033\x15\x06s\x03\x07\x035\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x019\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01=\x0b\x06\x19\x03\x05\x07;?\x01\x0f\x06#\x03\x07\x03A\x05\x07\x1b\t\x03\x11\x03C\x17\x17\x0bE\x03\t\x07\x03E7\x07\x03\x05\x07\x05\x07\x0b\x07\x0b\r\x04\x0b\x03\x03\x03\x03\x01%\x03\x07\t\x07\x17\x11\x03\x0b\x05\x19I\x03\x03\x01C\x03\x07\x07\x06\x0f\x03\x07\x05\x19M\x0b\x06\x19\x03\x07\x07KO\x19\x05\x07\x1b\t\x03\x11\x03Q\x17\x17\x0bE\x03\t\x07GS)\x07\x03\x05\x07\x05\x07\x0b\x07\x0b\r\x04\x0b\x03\x03\r\x04\x01\x07\tU\x05\x06\x03\x01\x05\x01\x00\x02(\xabM\x0f{.\x02\x8b\x83%\x03\x11\x0f\x0b\t\t\t\x0b\x11#!\x1f/!)!)#\x1f\x197\x85\x87\x1fz\x04\'\x1f;+y\x87.\x04!\x19+\xb1\xb3YMO_\x1b%)9\x19\'#++\x1b\x1f\x15\x1d\x15i\x13\r\x11\x13\x19\x1f#\x11\x17\x17\x15\x11\x17\x15\x15\x17\x0f)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00convert_v1\x00func_v1\x00gather_v1\x00reshape_v1\x00scatter_v1\x00iota_v1\x00dynamic_slice_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00compare_type\x00comparison_direction\x00broadcast_dimensions\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/while/body/closed_call\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00mhlo.layout_mode\x00default\x00None\x00[0]\x00[1]\x00[2]\x00main\x00public\x00private\x00\x00lapack_sgetrf_ffi\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00callee\x00jit(<lambda>)/jit(main)/while/cond/lt\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgetrf_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[6. , 7. , 8. ],
       [0. , 1. , 2. ],
       [0.5, 0.5, 0. ]]), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":384:11)
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3))
#loc21 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf64> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xf64>) -> tensor<3x3xf64> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xi32>, tensor<i32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc6)
    %4 = stablehlo.subtract %2#1, %3 : tensor<3xi32> loc(#loc6)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  GE, %2#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %10 = stablehlo.select %9, %2#0, %8 : tensor<3x3xi1>, tensor<3x3xf64> loc(#loc6)
    %11 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc7)
    %c_3 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %c_4 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
    %12:4 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_3, %iterArg_6 = %11, %iterArg_7 = %4) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %c_8 = stablehlo.constant dense<3> : tensor<i64> loc(#loc10)
      %13 = stablehlo.compare  LT, %iterArg, %c_8,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc11)
      stablehlo.return %13 : tensor<i1> loc(#loc10)
    } do {
      %13:3 = func.call @None(%iterArg_5, %iterArg_6, %iterArg_7) : (tensor<i64>, tensor<3xi32>, tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>, tensor<3xi32>) loc(#loc12)
      %c_8 = stablehlo.constant dense<1> : tensor<i64> loc(#loc10)
      %14 = stablehlo.add %iterArg, %c_8 : tensor<i64> loc(#loc13)
      stablehlo.return %14, %13#0, %13#1, %13#2 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc10)
    } loc(#loc10)
    return %10, %4, %12#2 : tensor<3x3xf64>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
  func.func private @None(%arg0: tensor<i64> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3)), %arg1: tensor<3xi32> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3)), %arg2: tensor<3xi32> loc("jit(<lambda>)/jit(main)/while/body/closed_call"(#loc3))) -> (tensor<i64>, tensor<3xi32>, tensor<3xi32>) {
    %c = stablehlo.constant dense<1> : tensor<i64> loc(#loc12)
    %0 = stablehlo.add %arg0, %c : tensor<i64> loc(#loc13)
    %c_0 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %1 = stablehlo.compare  LT, %arg0, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_1 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %2 = stablehlo.add %arg0, %c_1 : tensor<i64> loc(#loc13)
    %3 = stablehlo.select %1, %2, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %6 = "stablehlo.gather"(%arg2, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc18)
    %c_2 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %7 = stablehlo.compare  LT, %arg0, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_3 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %8 = stablehlo.add %arg0, %c_3 : tensor<i64> loc(#loc13)
    %9 = stablehlo.select %7, %8, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %10 = stablehlo.convert %9 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %12 = "stablehlo.gather"(%arg1, %11) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc18)
    %c_4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc12)
    %13 = stablehlo.compare  LT, %6, %c_4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %c_5 = stablehlo.constant dense<3> : tensor<i32> loc(#loc12)
    %14 = stablehlo.add %6, %c_5 : tensor<i32> loc(#loc13)
    %15 = stablehlo.select %13, %14, %6 : tensor<i1>, tensor<i32> loc(#loc15)
    %16 = stablehlo.dynamic_slice %arg1, %15, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc19)
    %17 = stablehlo.reshape %16 : (tensor<1xi32>) -> tensor<i32> loc(#loc20)
    %c_6 = stablehlo.constant dense<0> : tensor<i64> loc(#loc12)
    %18 = stablehlo.compare  LT, %arg0, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc14)
    %c_7 = stablehlo.constant dense<3> : tensor<i64> loc(#loc12)
    %19 = stablehlo.add %arg0, %c_7 : tensor<i64> loc(#loc13)
    %20 = stablehlo.select %18, %19, %arg0 : tensor<i1>, tensor<i64> loc(#loc15)
    %21 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32> loc(#loc16)
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %23 = "stablehlo.scatter"(%arg1, %22, %17) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg4: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
      stablehlo.return %arg4 : tensor<i32> loc(#loc21)
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc21)
    %c_8 = stablehlo.constant dense<0> : tensor<i32> loc(#loc12)
    %24 = stablehlo.compare  LT, %6, %c_8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %c_9 = stablehlo.constant dense<3> : tensor<i32> loc(#loc12)
    %25 = stablehlo.add %6, %c_9 : tensor<i32> loc(#loc13)
    %26 = stablehlo.select %24, %25, %6 : tensor<i1>, tensor<i32> loc(#loc15)
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc17)
    %28 = "stablehlo.scatter"(%23, %27, %12) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3)), %arg4: tensor<i32> loc("jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc3))):
      stablehlo.return %arg4 : tensor<i32> loc(#loc21)
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc21)
    return %0, %28, %arg2 : tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc12)
  } loc(#loc12)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":383:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":383:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]"(#loc3))
#loc10 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc3))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc3))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc3))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc3))
#loc20 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc3))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\'\x03\xd6\x02>\x023\x01\xa9\x0f\x17\x0f\x13\x13\x0f\x0b\x0f\x1b\x13\x13\x0f\x0f\x0f\x0b\x07\x0b\x0f\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b;\x0b\x0b\x0b\x0f\x13;\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x03]\x0b/\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x0b\x0b/O\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x17\x0b//\x0b/\x1f\x1f\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x1f\x0b/O\x01\x1b\x17\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x17\x0b\x13\x0b\x01\x05\x0b\x0f\x03/\x0f\x0f\x13\x0f\x17\x07\x13\x07\x07\x07\x07\x13\x0f\x1b\'\x13\x13\x13\x13\x13\x17\x17\x13\x026\n\x1dY\x03\x17!\x02\x06\x17\x1d\x8f\x03\x1d*\x02\x03\x03\x037\xb5\x1d\x7f\x03\x05)\x1d[\x03\x03\x053\xbd5\xe1\x03\x03\r\xe3\x03\x03\r\xdf\x1d]\x03\x1d_\x03\x1dc\x03\x05+\x1f\x05-\x1da\x03\x03\x03\r\xe5\x11\x03\x05\x05/\x051\x053\x055\x03\x03\r\xdd\x057\x059\x05;\x03\re\xab;\xb7=\xb9g\xb5?\xbfi\xab\x05=\x05?\x05A\x1dk\x03\x03\x03\r\xe7\x03\r;\xb7=\xb9w\xaby\xab{\xb9}\xb5\x03\x03\x81\xb7\x03\tKMO\'Q\'\x1dS\x05C\x11\x01\x00\x05E\x05G\x05I\x03\x0b)\xad+\xc3-\xc5\x1d\xd3/\xd5\x03\x0b)\xad+\xd7-\xd9\x1d\xbb/\xdb\x05K\x05M\x05O\x05Q\x05S\x05U\x05W\x05Y\x05[\x05]\x03\x03?\xbf\x1dq\x03\x05_\x1du\x03\x05a\x05c\x05e\x05g\x05i\x05k\x05m\x1d\x85\x87\x05o\x17!\xfe\x055\x1d\x8b\x8d\x05q\x17!\xfe\x05\x1d\x05s\x03\x13\x93\xe9\x95\xeb\x97\xed\x99\xad\x9b\xef\x9d\xa9\x9f\xf1\xa1\xf3\xa3\xf7\x05u\x05w\x05y\x05{\x05}\x05\x7f\x05\x81\x05\x83\x05\x85\x03\x03\r\xfd\x03\x053\xbd5\xff\r\x01\x1f\x1b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d\x87\x1d\x89\x1d\x8b\x1f#\x01\x13\x0f\x01\x05\x03\x1d\x8d\t\x07\x1f\x1b\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f\'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc7\xcb\xcf\r\x05\xaf\xc9\xb1\xb3\x1d\x8f\r\x05\xaf\xcd\xb1\xb3\x1d\x91\r\x05\xaf\xd1\xb1\xb3\x1d\x93\x1d\x95\x1d\x97#!\x03\x07\xa9\xa9\xa9\x1d\x99\x1f\x05\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x00\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x1f\x05\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00\x00\x1f\x07\t\x03\x00\x00\x00\x0b\x03\x1d\x9b\x1d\x9d\x05\x01\x03\x03\xc1\x03\x03\xf5\x15\x03\x01\x01\x01\x03\x07\xc1\xf9\xfb\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x01\x1f\x07\t\x01\x00\x00\x00\x07\x05\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03\r\x02\x02\x03\x037\x06\x02\x1d\x16\x02\x03\x05\x9f\x1d\x1e\x02\x03\x05\xa1\x1d&\x02\x03\x05\xa3\x05\xa5\x03\x032\x02\xbb\x05\xa7\x1d:\x02\x03\x05\xa9\x01\t\x01\x02\x02)\x01\x0f)\x01\x15)\x03\r\x15)\x01\x17)\x05\r\r\x13\x1d)\x03\x05\x15\x0b\x1b\x01\x13)\x03\x05\x0f)\x01\x13\x11\x01\x07\r\t\t\x11\x07\x05\t\t\x07\x05\t\t)\x03\x01\x0f)\x03%\x13)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x05\x05\x05\x17)\x05\r\r\x17)\x03\t\x0f\x04\xb6\t\x05\x01\x11\x1fI\x07\x03\x01\t\x11\x11\x1fU\x07\x033U\x19\x03\x83G\x03%\x15\x06\x89\x03\r\x03\x01\x03\x03\x05\x13\x03\x05\x03\x03\x05\x13\x03\x05\x1d\x07\x05\x91\x07\r\t\x07\x03\x03\x03\x03\x05\xa5\x03\x07\x05\x07\x05\t\x03\t\x03\x0f\x1f\x06\x05\x03\t\x05\x0b\x11\x03\x03\x05%\x03\x07\x05\x07\x05\t\x03\x07\x03\x15\t\x07\x05\xa7\x03\x0b\x05\r\x17\x05\x07\x05\t\x03-\x03\x19\x03\x03\x05\n\x02\x03\x1d\x05\x07\x05\t\x03\r\x03\x1d\x05\x07\x05\x0e\x02\x03/\x03\x1b\x0b\x06\x05\x03\r\x07!\t\x1f\x19\x03\x12\x02G\x03\t\x03\x03\x1a\x02\x15\x03\x05\x03\x03"\x02\x15\x03\x05!\x16\x07\t\x05\x05\t\t\t)\'%\x13\x0b\x03\r\x0f\t\x05\x07\x05\x07\t\x07\t\x07\x03\x03\x07\x13\x03\x05\t\x076\x02\x11\x03\x0b\x05\x01\t\r\x04\x07\x03\x0b\x03\x13\x13\t\x05\x07\x05\x07\t\x07\t\x07#\x07\x01.\x02\x07\x05\t\t\x07\x03\x05\x07\x03\x03\x071\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x0f\r\x04\x07\t\x11\t\x0b\r\r\x04\x1f\x07#\x13/\x11\x11\x01W\x07\x03W\xa7\x07\x05\x01\t\x01\t\x01\x03\x03\x011\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x07\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x01\x0b\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x0f\x0b\x06\x19\x03\x05\x07\r\x11\x01\x0f\x06#\x03\x07\x03\x13\x05\x07\x1b\t\x03\x11\x03\x15\x13\x07A9\x03\x07\x05\x05\x17\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x01\x1b\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01\x1f\x0b\x06\x19\x03\x05\x07\x1d!\x01\x0f\x06#\x03\x07\x03#\x05\x07\x1b\t\x03\x11\x03%\x13\x07A9\x03\x07\x05\x03\'\x03\x03\x01%\x03\x07\t\x07\x17\x11\x03\x0b\x05\x19+\x03\x03\x01C\x03\x07\x07\x06\x0f\x03\x07\x05\x19/\x0b\x06\x19\x03\x07\x07-1\x19\x1b\x07om\x03\x11\x05\x033\x15\x06s\x03\x07\x035\x03\x03\x01\x15\x03\x05\t\x07\x17\x11\x03\x0b\x05\x019\x03\x03\x01\x13\x03\x05\x07\x06\x0f\x03\x05\x05\x01=\x0b\x06\x19\x03\x05\x07;?\x01\x0f\x06#\x03\x07\x03A\x05\x07\x1b\t\x03\x11\x03C\x17\x17\x0bE\x03\t\x07\x03E7\x07\x03\x05\x07\x05\x07\x0b\x07\x0b\r\x04\x0b\x03\x03\x03\x03\x01%\x03\x07\t\x07\x17\x11\x03\x0b\x05\x19I\x03\x03\x01C\x03\x07\x07\x06\x0f\x03\x07\x05\x19M\x0b\x06\x19\x03\x07\x07KO\x19\x05\x07\x1b\t\x03\x11\x03Q\x17\x17\x0bE\x03\t\x07GS)\x07\x03\x05\x07\x05\x07\x0b\x07\x0b\r\x04\x0b\x03\x03\r\x04\x01\x07\tU\x05\x06\x03\x01\x05\x01\x00\x02(\xabM\x0f{.\x02\x8b\x83%\x03\x11\x0f\x0b\t\t\t\x0b\x11#!\x1f/!)!)#\x1f\x197\x85\x87\x1fz\x04\'\x1f;+y\x87.\x04!\x19+\xb1\xb3YMO_\x1b%)9\x19\'#++\x1b\x1f\x15\x1d\x15i\x13\r\x11\x13\x19\x1f#\x11\x17\x17\x15\x11\x17\x15\x15\x17\x0f)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00convert_v1\x00func_v1\x00gather_v1\x00reshape_v1\x00scatter_v1\x00iota_v1\x00dynamic_slice_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00compare_type\x00comparison_direction\x00broadcast_dimensions\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/while/body/closed_call\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jit(<lambda>)/jit(main)/while/body/scatter[update_jaxpr=None update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00mhlo.layout_mode\x00default\x00None\x00[0]\x00[1]\x00[2]\x00main\x00public\x00private\x00\x00lapack_dgetrf_ffi\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1 _split_transpose=False]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00callee\x00jit(<lambda>)/jit(main)/while/cond/lt\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
