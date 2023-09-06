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
from numpy import array, int32, float32, complex64

data_2023_06_14 = {}

# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_14['f32'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgetrf'],
    serialized_date=datetime.date(2023, 6, 14),
    inputs=(),
    expected_outputs=(array([[6. , 7. , 8. ],
       [0. , 1. , 2. ],
       [0.5, 0.5, 0. ]], dtype=float32), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "[0]"}, tensor<3xi32> {jax.result_info = "[1]"}, tensor<3xi32> {jax.result_info = "[2]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf32> loc(#loc3)
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32> loc(#loc4)
    %2 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %3 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %4 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %6 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %8 = stablehlo.concatenate %5, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc5)
    %9 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %10 = stablehlo.convert %9 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %11 = stablehlo.reshape %10 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %12 = stablehlo.constant dense<> : tensor<0xi32> loc(#loc5)
    %13 = stablehlo.constant dense<1> : tensor<i64> loc(#loc5)
    %14 = stablehlo.constant dense<3> : tensor<i32> loc(#loc5)
    %15 = stablehlo.constant dense<3> : tensor<i32> loc(#loc5)
    %16:3 = stablehlo.custom_call @lapack_sgetrf(%13, %14, %15, %1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<i64>, tensor<i32>, tensor<i32>, tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xi32>, tensor<i32>) loc(#loc5)
    %17 = stablehlo.constant dense<1> : tensor<i32> loc(#loc5)
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc5)
    %19 = stablehlo.subtract %16#1, %18 : tensor<3xi32> loc(#loc5)
    %20 = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %22 = stablehlo.compare  GE, %16#2, %21,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %24 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc5)
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc5)
    %26 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc5)
    %27 = stablehlo.select %26, %16#0, %25 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc5)
    %28 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc6)
    %29 = stablehlo.constant dense<0> : tensor<i64> loc(#loc7)
    %30 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %31:4 = stablehlo.while(%iterArg = %30, %iterArg_0 = %29, %iterArg_1 = %28, %iterArg_2 = %19) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %32 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %33 = stablehlo.compare  LT, %iterArg, %32,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc10)
      stablehlo.return %33 : tensor<i1> loc(#loc9)
    } do {
      %32 = stablehlo.constant dense<1> : tensor<i64> loc(#loc9)
      %33 = stablehlo.add %iterArg_0, %32 : tensor<i64> loc(#loc11)
      %34 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %35 = stablehlo.compare  LT, %iterArg_0, %34,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %36 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %37 = stablehlo.add %iterArg_0, %36 : tensor<i64> loc(#loc11)
      %38 = stablehlo.select %35, %37, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %39 = stablehlo.convert %38 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %41 = "stablehlo.gather"(%iterArg_2, %40) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc16)
      %42 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %43 = stablehlo.compare  LT, %iterArg_0, %42,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %44 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %45 = stablehlo.add %iterArg_0, %44 : tensor<i64> loc(#loc11)
      %46 = stablehlo.select %43, %45, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %47 = stablehlo.convert %46 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %48 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %49 = "stablehlo.gather"(%iterArg_1, %48) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc16)
      %50 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
      %51 = stablehlo.compare  LT, %41, %50,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc12)
      %52 = stablehlo.constant dense<3> : tensor<i32> loc(#loc9)
      %53 = stablehlo.add %41, %52 : tensor<i32> loc(#loc11)
      %54 = stablehlo.select %51, %53, %41 : tensor<i1>, tensor<i32> loc(#loc13)
      %55 = stablehlo.dynamic_slice %iterArg_1, %54, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc17)
      %56 = stablehlo.reshape %55 : (tensor<1xi32>) -> tensor<i32> loc(#loc18)
      %57 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %58 = stablehlo.compare  LT, %iterArg_0, %57,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %59 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %60 = stablehlo.add %iterArg_0, %59 : tensor<i64> loc(#loc11)
      %61 = stablehlo.select %58, %60, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %62 = stablehlo.convert %61 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %63 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %64 = "stablehlo.scatter"(%iterArg_1, %63, %56) ({
      ^bb0(%arg0: tensor<i32> loc(unknown), %arg1: tensor<i32> loc(unknown)):
        stablehlo.return %arg1 : tensor<i32> loc(#loc19)
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc19)
      %65 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
      %66 = stablehlo.compare  LT, %41, %65,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc12)
      %67 = stablehlo.constant dense<3> : tensor<i32> loc(#loc9)
      %68 = stablehlo.add %41, %67 : tensor<i32> loc(#loc11)
      %69 = stablehlo.select %66, %68, %41 : tensor<i1>, tensor<i32> loc(#loc13)
      %70 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %71 = "stablehlo.scatter"(%64, %70, %49) ({
      ^bb0(%arg0: tensor<i32> loc(unknown), %arg1: tensor<i32> loc(unknown)):
        stablehlo.return %arg1 : tensor<i32> loc(#loc19)
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc19)
      %72 = stablehlo.constant dense<1> : tensor<i64> loc(#loc9)
      %73 = stablehlo.add %iterArg, %72 : tensor<i64> loc(#loc11)
      stablehlo.return %73, %33, %71, %iterArg_2 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc9)
    } loc(#loc9)
    return %27, %19, %31#2 : tensor<3x3xf32>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":550:0)
#loc2 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":551:0)
#loc3 = loc("jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]"(#loc1))
#loc4 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/lu"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc2))
#loc7 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc2))
#loc8 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1]"(#loc2))
#loc9 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc2))
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc2))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc2))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc2))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc2))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc2))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc2))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc2))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%'\x03\xa6\x02\x0e\x023\x01\xb1\x0f\x0f\x07\x17\x0b\x13\x13\x0f\x1b\x13\x0f\x0f\x13\x0f\x13\x13\x13\x0f\x0f\x0b\x13\x17\x0b\x0b\x0b\x0b;\x0b\x0b\x0b\x0f;#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x13\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x03Q\x0f\x0f/\x0b\x0f\x0b\x0bO\x0b/\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x0f/\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x17/\x1f\x1f\x0b\x1fO/\x0b\x01\x07\x0b\x13\x0b\x01\x03\x0f\x031\x0f\x0f\x13\x13\x0f\x17\x07\x07\x07\x07\x07\x13\x0f\x13\x1b\x13\x13\x13\x13\x13\x13\x17\x17\x13\x02J\t\x1d]\x07\x1d\x8b\x07\x1f\x17-\x9e\x08\x01\x05)\x03\x03/\xb9\x03\x03\t\xd9\x1d\x8d\x07\x03\x051\xc13\xff\x03\x03\t\xfd\x1d\x8f\x07\x1d\x91\x07\x03\x03\t\xdf\x1d\x95\x07\x1d\x02\x02\x07\x03\x03\t\xdd\x03\x03\t\xf5\x1d\x93\x07\x11\x01\x05\x05+\x03\x03S\xb1\x17-\x9a\x08\x01\x05-\x05/\x051\x053\x03\r\x97\xb57\xb19\xbb\x99\xb9;\xc3\x9b\xb5\x055\x057\x059\x1d\x9d\x07\x03\r7\xb19\xbb\xa9\xb5\xab\xb5\xad\xbb\xaf\xb9\x03\x07C%E%'G\x05;\x05=\x05?\x03\x0bK\xbdM\xc5O\xc7'\xd5Q\xd7\x05A\x05C\x05E\x05G\x05I\x1dW+\x05K\x1d[+\x05M\x05O\x03\x03a\xb1\x05Q\x03\x03\t\xdb\x03\x11g\xe1i\xe3k\xe5m\xbdo\xe7q\xe9s\xebu\xef\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x03\x03\t\xf3\x03\x051\xc13\xf7\x03\x03\t\xf9\x03\x03/\xfb\x1d\x81\x07\x05c\x1d\x85\x07\x05e\x1d\x89\x07\x05g\x05i\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x03\x03;\xc3\x1d\xa3\x07\x05}\x1d\xa7\x07\x05\x7f\x05\x81\x05\x83\x05\x85\x05\x87\x13\x11\x01\x1f%\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x89\x1f+\x01\x05\x03\x03\x01\x1f'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x1f\x1d\x11\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc9\xcd\xd1\r\x03\xb7\xcb\x1d\x8b\r\x03\xb7\xcf\x1d\x8d\r\x03\xb7\xd3\x1d\x8f\x1d\x91\x1d\x93\x1f\x03\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1f\x19\x01\x1f\x03\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x03\x00\x00\x00\x0b\x05\x1d\x95\x1d\x97\x05\x01\x03\t\xb3\xb3\xb3\xbf\x03\x03\xed\x15\x03\x01\r\x01\x03\x07\xbf\xf1\xb3\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x00\x00\x00\x00\x07\x05\x1f\x1b\t\x00\x00\xc0\x7f\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x03\x11\x00\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x05\x99\x1d\n\x02\x07\x05\x9b\x01\x02\x02)\x01\x11)\x01\x0f)\x03\r\x0f)\x03\x05\x0f)\x01\x17)\x05\r\r\x13\x1b\x1d\t\x13\x01)\x03\x01\x0f)\x01\x13)\x03\x05\x11\x11\x01\x07\r\x07\x07)\x03%\x13)\x03\t\x0f)\x03\x01\x15)\x03\t\x15)\x03\x05\x15)\x03\x01\x11)\x05\x05\x05\x17)\x05\r\r\x17)\x03\t\x11\x04f\n\x05\x01\x11\x05A\x07\x03\x01\x05\x19\x11\x05I\x05\x03K\x85\x13\x03U)\x03!\x0f\x06Y\x03\r\x03\x01\x03\x03\x01\r\x03\x03\x03\x03\x01\r\x03\x03\x0b\x06\x01\x03\x05\x03\x05\x0f\x06\x01\x03\t\x03\t\x0b\x06\x01\x03\x05\x03\x07\x0f\x06\x01\x03\t\x03\r\x1b\x07\x01_\x03#\x05\x0b\x0f\x03\x03\x01\r\x03\x03\x0b\x06\x01\x03\x05\x03\x13\x0f\x06\x01\x03\t\x03\x15\x03\x03\x01c\x03\x19\x03\x03\x01\x1f\x03\x03\x03\x03\x01\x19\x03\x05\x03\x03\x01\x19\x03\x05\x1d\x07\x01e\x07\r\x07\x05\t\x1b\x1d\x1f\x03\x03\x03\x01w\x03\x05\x05\x07\x01\x0b\x03\x07\x03'\x1f\x06\x01\x03\x07\x05#)\x03\x03\x01!\x03\x05\x05\x07\x01\x0b\x03\x05\x03-\x07\x07\x01y\x03\x0b\x05%/\x05\x07\x01\x0b\x03-\x031\x03\x03\x01{\x03\x1b\x05\x07\x01\x0b\x03\r\x035\x05\x07\x01}\x03/\x033\r\x06\x01\x03\r\x079!7\x13\x03\x7f)\x03\x07\x03\x03\x83\x13\x03\x03\x03\x03\x87\x13\x03\x03!\x16\x03\t\x03\x03\x07\x07\tA?=+\t\x03\r\x0f\t\x03\x05\x03\x05\x07\x05\x07\x05\x03\x03\x03\r\x03\x03\x07\x07\x06\x02\x11\x03\x0b\x05KS\x11\x04\x03\x03U\x03]\xaf\t\x03\x05\x03\x05\x07\x05\x07\x05\x03\x03\x03\x1f\x03\x03\t\x06\x0f\x03\x03\x05MS\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05MW\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05M[\r\x06\x17\x03\x03\x07Y]M\x0b\x06#\x03\x05\x03_\x05\x07\x1b\x0b\x03\t\x03a\x15\x07=5\x03\x05\x05Qc\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05Mg\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05Mk\r\x06\x17\x03\x03\x07imM\x0b\x06#\x03\x05\x03o\x05\x07\x1b\x0b\x03\t\x03q\x15\x07=5\x03\x05\x05Os\x03\x03\x03!\x03\x05\x07\x07\x15\x11\x03\x0b\x05ew\x03\x03\x03\x19\x03\x05\t\x06\x0f\x03\x05\x05e{\r\x06\x17\x03\x05\x07y}e#\x07\xa1\x9f\x03\t\x05O\x7f\x0f\x06\xa5\x03\x05\x03\x81\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05M\x85\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05M\x89\r\x06\x17\x03\x03\x07\x87\x8bM\x0b\x06#\x03\x05\x03\x8d\x05\x07\x1b\x0b\x03\t\x03\x8f\x17\x17\x1d?\x03\x07\x07O\x91\x83\x05\x03\x05\x07\x05\x05\x05\x05\x05\x11\x04\x1d\x03\xa9\x03\x03\x03!\x03\x05\x07\x07\x15\x11\x03\x0b\x05e\x95\x03\x03\x03\x19\x03\x05\t\x06\x0f\x03\x05\x05e\x99\r\x06\x17\x03\x05\x07\x97\x9be\x05\x07\x1b\x0b\x03\t\x03\x9d\x17\x17\x1d?\x03\x07\x07\x93\x9fu\x05\x03\x05\x07\x05\x05\x05\x05\x05\x11\x04\x1d\x03\xa9\x03\x03\x03\x1f\x03\x03\t\x06\x0f\x03\x03\x05K\xa3\x11\x04\x03\t\xa5U\xa1Q\x11\x04\x05\x07;+G\x06\x03\x01\x05\x01\x00v%\x9dM2\x04\x1d\x03\x0f\x0b\t\t\t!'\x1f;+y\x87.\x04!\x19+\xb1\xb3YMO{\xe9\x8b\x83\x1f/!!)#\x1f\x19\x157\x85\x87\x1f\x1f\x15\x1d\x15\x1b%)\x19'#+\x1b+\x83\x13\r#\x13\x19\x1f\x1f\x11\x17\x15\x11\x15\x17\x15\x17\x0f\x17)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00compare_v1\x00add_v1\x00convert_v1\x00select_v1\x00reshape_v1\x00return_v1\x00iota_v1\x00gather_v1\x00scatter_v1\x00func_v1\x00concatenate_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00dynamic_slice_v1\x00value\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00dimension\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_sgetrf\x00jit(<lambda>)/jit(main)/while/body/scatter[update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00jit(<lambda>)/jit(main)/while/cond/lt\x00",
    xla_call_module_version=6,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_14['f64'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgetrf'],
    serialized_date=datetime.date(2023, 6, 14),
    inputs=(),
    expected_outputs=(array([[6. , 7. , 8. ],
       [0. , 1. , 2. ],
       [0.5, 0.5, 0. ]]), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf64> {jax.result_info = "[0]"}, tensor<3xi32> {jax.result_info = "[1]"}, tensor<3xi32> {jax.result_info = "[2]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf64> loc(#loc3)
    %1 = stablehlo.reshape %0 : (tensor<9xf64>) -> tensor<3x3xf64> loc(#loc4)
    %2 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %3 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %4 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %6 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %8 = stablehlo.concatenate %5, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc5)
    %9 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %10 = stablehlo.convert %9 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %11 = stablehlo.reshape %10 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %12 = stablehlo.constant dense<> : tensor<0xi32> loc(#loc5)
    %13 = stablehlo.constant dense<1> : tensor<i64> loc(#loc5)
    %14 = stablehlo.constant dense<3> : tensor<i32> loc(#loc5)
    %15 = stablehlo.constant dense<3> : tensor<i32> loc(#loc5)
    %16:3 = stablehlo.custom_call @lapack_dgetrf(%13, %14, %15, %1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<i64>, tensor<i32>, tensor<i32>, tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xi32>, tensor<i32>) loc(#loc5)
    %17 = stablehlo.constant dense<1> : tensor<i32> loc(#loc5)
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc5)
    %19 = stablehlo.subtract %16#1, %18 : tensor<3xi32> loc(#loc5)
    %20 = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %22 = stablehlo.compare  GE, %16#2, %21,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %24 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc5)
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f64>) -> tensor<3x3xf64> loc(#loc5)
    %26 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc5)
    %27 = stablehlo.select %26, %16#0, %25 : tensor<3x3xi1>, tensor<3x3xf64> loc(#loc5)
    %28 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc6)
    %29 = stablehlo.constant dense<0> : tensor<i64> loc(#loc7)
    %30 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %31:4 = stablehlo.while(%iterArg = %30, %iterArg_0 = %29, %iterArg_1 = %28, %iterArg_2 = %19) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %32 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %33 = stablehlo.compare  LT, %iterArg, %32,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc10)
      stablehlo.return %33 : tensor<i1> loc(#loc9)
    } do {
      %32 = stablehlo.constant dense<1> : tensor<i64> loc(#loc9)
      %33 = stablehlo.add %iterArg_0, %32 : tensor<i64> loc(#loc11)
      %34 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %35 = stablehlo.compare  LT, %iterArg_0, %34,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %36 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %37 = stablehlo.add %iterArg_0, %36 : tensor<i64> loc(#loc11)
      %38 = stablehlo.select %35, %37, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %39 = stablehlo.convert %38 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %41 = "stablehlo.gather"(%iterArg_2, %40) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc16)
      %42 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %43 = stablehlo.compare  LT, %iterArg_0, %42,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %44 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %45 = stablehlo.add %iterArg_0, %44 : tensor<i64> loc(#loc11)
      %46 = stablehlo.select %43, %45, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %47 = stablehlo.convert %46 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %48 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %49 = "stablehlo.gather"(%iterArg_1, %48) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc16)
      %50 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
      %51 = stablehlo.compare  LT, %41, %50,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc12)
      %52 = stablehlo.constant dense<3> : tensor<i32> loc(#loc9)
      %53 = stablehlo.add %41, %52 : tensor<i32> loc(#loc11)
      %54 = stablehlo.select %51, %53, %41 : tensor<i1>, tensor<i32> loc(#loc13)
      %55 = stablehlo.dynamic_slice %iterArg_1, %54, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc17)
      %56 = stablehlo.reshape %55 : (tensor<1xi32>) -> tensor<i32> loc(#loc18)
      %57 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %58 = stablehlo.compare  LT, %iterArg_0, %57,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %59 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %60 = stablehlo.add %iterArg_0, %59 : tensor<i64> loc(#loc11)
      %61 = stablehlo.select %58, %60, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %62 = stablehlo.convert %61 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %63 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %64 = "stablehlo.scatter"(%iterArg_1, %63, %56) ({
      ^bb0(%arg0: tensor<i32> loc(unknown), %arg1: tensor<i32> loc(unknown)):
        stablehlo.return %arg1 : tensor<i32> loc(#loc19)
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc19)
      %65 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
      %66 = stablehlo.compare  LT, %41, %65,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc12)
      %67 = stablehlo.constant dense<3> : tensor<i32> loc(#loc9)
      %68 = stablehlo.add %41, %67 : tensor<i32> loc(#loc11)
      %69 = stablehlo.select %66, %68, %41 : tensor<i1>, tensor<i32> loc(#loc13)
      %70 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %71 = "stablehlo.scatter"(%64, %70, %49) ({
      ^bb0(%arg0: tensor<i32> loc(unknown), %arg1: tensor<i32> loc(unknown)):
        stablehlo.return %arg1 : tensor<i32> loc(#loc19)
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc19)
      %72 = stablehlo.constant dense<1> : tensor<i64> loc(#loc9)
      %73 = stablehlo.add %iterArg, %72 : tensor<i64> loc(#loc11)
      stablehlo.return %73, %33, %71, %iterArg_2 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc9)
    } loc(#loc9)
    return %27, %19, %31#2 : tensor<3x3xf64>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":553:0)
#loc2 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":554:0)
#loc3 = loc("jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]"(#loc1))
#loc4 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/lu"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc2))
#loc7 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc2))
#loc8 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1]"(#loc2))
#loc9 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc2))
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc2))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc2))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc2))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc2))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc2))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc2))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc2))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%'\x03\xa6\x02\x0e\x023\x01\xb1\x0f\x0f\x07\x17\x0b\x13\x13\x0f\x1b\x13\x0f\x0f\x13\x0f\x13\x13\x13\x0f\x0f\x0b\x13\x17\x0b\x0b\x0b\x0b;\x0b\x0b\x0b\x0f;#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x13\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x03Q\x0f\x0f/\x0b\x0f\x0b\x0bO\x0b/\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x0f/\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x17/\x1f\x1f\x0b/O/\x0b\x01\x07\x0b\x13\x0b\x01\x03\x0f\x031\x0f\x0f\x13\x13\x0f\x17\x07\x07\x07\x07\x07\x13\x0f\x13\x1b\x13\x13\x13\x13\x13\x13\x17\x17\x13\x02Z\t\x1d]\x07\x1d\x8b\x07\x1f\x17-\xaa\x08\x01\x05)\x03\x03/\xb9\x03\x03\t\xd9\x1d\x8d\x07\x03\x051\xc13\xff\x03\x03\t\xfd\x1d\x8f\x07\x1d\x91\x07\x03\x03\t\xdf\x1d\x95\x07\x1d\x02\x02\x07\x03\x03\t\xdd\x03\x03\t\xf5\x1d\x93\x07\x11\x01\x05\x05+\x03\x03S\xb1\x17-\xa6\x08\x01\x05-\x05/\x051\x053\x03\r\x97\xb57\xb19\xbb\x99\xb9;\xc3\x9b\xb5\x055\x057\x059\x1d\x9d\x07\x03\r7\xb19\xbb\xa9\xb5\xab\xb5\xad\xbb\xaf\xb9\x03\x07C%E%'G\x05;\x05=\x05?\x03\x0bK\xbdM\xc5O\xc7'\xd5Q\xd7\x05A\x05C\x05E\x05G\x05I\x1dW+\x05K\x1d[+\x05M\x05O\x03\x03a\xb1\x05Q\x03\x03\t\xdb\x03\x11g\xe1i\xe3k\xe5m\xbdo\xe7q\xe9s\xebu\xef\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x03\x03\t\xf3\x03\x051\xc13\xf7\x03\x03\t\xf9\x03\x03/\xfb\x1d\x81\x07\x05c\x1d\x85\x07\x05e\x1d\x89\x07\x05g\x05i\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x03\x03;\xc3\x1d\xa3\x07\x05}\x1d\xa7\x07\x05\x7f\x05\x81\x05\x83\x05\x85\x05\x87\x13\x11\x01\x1f%\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x89\x1f+\x01\x05\x03\x03\x01\x1f'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x1f\x1d\x11\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc9\xcd\xd1\r\x03\xb7\xcb\x1d\x8b\r\x03\xb7\xcf\x1d\x8d\r\x03\xb7\xd3\x1d\x8f\x1d\x91\x1d\x93\x1f\x03\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1f\x19\x01\x1f\x03\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x03\x00\x00\x00\x0b\x05\x1d\x95\x1d\x97\x05\x01\x03\t\xb3\xb3\xb3\xbf\x03\x03\xed\x15\x03\x01\r\x01\x03\x07\xbf\xf1\xb3\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x00\x00\x00\x00\x07\x05\x1f\x1b\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x03\x11\x00\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x05\x99\x1d\n\x02\x07\x05\x9b\x01\x02\x02)\x01\x11)\x01\x0f)\x03\r\x0f)\x03\x05\x0f)\x01\x17)\x05\r\r\x13\x1b\x1d\x0b\x13\x01)\x03\x01\x0f)\x01\x13)\x03\x05\x11\x11\x01\x07\r\x07\x07)\x03%\x13)\x03\t\x0f)\x03\x01\x15)\x03\t\x15)\x03\x05\x15)\x03\x01\x11)\x05\x05\x05\x17)\x05\r\r\x17)\x03\t\x11\x04f\n\x05\x01\x11\x05A\x07\x03\x01\x05\x19\x11\x05I\x05\x03K\x85\x13\x03U)\x03!\x0f\x06Y\x03\r\x03\x01\x03\x03\x01\r\x03\x03\x03\x03\x01\r\x03\x03\x0b\x06\x01\x03\x05\x03\x05\x0f\x06\x01\x03\t\x03\t\x0b\x06\x01\x03\x05\x03\x07\x0f\x06\x01\x03\t\x03\r\x1b\x07\x01_\x03#\x05\x0b\x0f\x03\x03\x01\r\x03\x03\x0b\x06\x01\x03\x05\x03\x13\x0f\x06\x01\x03\t\x03\x15\x03\x03\x01c\x03\x19\x03\x03\x01\x1f\x03\x03\x03\x03\x01\x19\x03\x05\x03\x03\x01\x19\x03\x05\x1d\x07\x01e\x07\r\x07\x05\t\x1b\x1d\x1f\x03\x03\x03\x01w\x03\x05\x05\x07\x01\x0b\x03\x07\x03'\x1f\x06\x01\x03\x07\x05#)\x03\x03\x01!\x03\x05\x05\x07\x01\x0b\x03\x05\x03-\x07\x07\x01y\x03\x0b\x05%/\x05\x07\x01\x0b\x03-\x031\x03\x03\x01{\x03\x1b\x05\x07\x01\x0b\x03\r\x035\x05\x07\x01}\x03/\x033\r\x06\x01\x03\r\x079!7\x13\x03\x7f)\x03\x07\x03\x03\x83\x13\x03\x03\x03\x03\x87\x13\x03\x03!\x16\x03\t\x03\x03\x07\x07\tA?=+\t\x03\r\x0f\t\x03\x05\x03\x05\x07\x05\x07\x05\x03\x03\x03\r\x03\x03\x07\x07\x06\x02\x11\x03\x0b\x05KS\x11\x04\x03\x03U\x03]\xaf\t\x03\x05\x03\x05\x07\x05\x07\x05\x03\x03\x03\x1f\x03\x03\t\x06\x0f\x03\x03\x05MS\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05MW\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05M[\r\x06\x17\x03\x03\x07Y]M\x0b\x06#\x03\x05\x03_\x05\x07\x1b\x0b\x03\t\x03a\x15\x07=5\x03\x05\x05Qc\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05Mg\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05Mk\r\x06\x17\x03\x03\x07imM\x0b\x06#\x03\x05\x03o\x05\x07\x1b\x0b\x03\t\x03q\x15\x07=5\x03\x05\x05Os\x03\x03\x03!\x03\x05\x07\x07\x15\x11\x03\x0b\x05ew\x03\x03\x03\x19\x03\x05\t\x06\x0f\x03\x05\x05e{\r\x06\x17\x03\x05\x07y}e#\x07\xa1\x9f\x03\t\x05O\x7f\x0f\x06\xa5\x03\x05\x03\x81\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05M\x85\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05M\x89\r\x06\x17\x03\x03\x07\x87\x8bM\x0b\x06#\x03\x05\x03\x8d\x05\x07\x1b\x0b\x03\t\x03\x8f\x17\x17\x1d?\x03\x07\x07O\x91\x83\x05\x03\x05\x07\x05\x05\x05\x05\x05\x11\x04\x1d\x03\xa9\x03\x03\x03!\x03\x05\x07\x07\x15\x11\x03\x0b\x05e\x95\x03\x03\x03\x19\x03\x05\t\x06\x0f\x03\x05\x05e\x99\r\x06\x17\x03\x05\x07\x97\x9be\x05\x07\x1b\x0b\x03\t\x03\x9d\x17\x17\x1d?\x03\x07\x07\x93\x9fu\x05\x03\x05\x07\x05\x05\x05\x05\x05\x11\x04\x1d\x03\xa9\x03\x03\x03\x1f\x03\x03\t\x06\x0f\x03\x03\x05K\xa3\x11\x04\x03\t\xa5U\xa1Q\x11\x04\x05\x07;+G\x06\x03\x01\x05\x01\x00v%\x9dM2\x04\x1d\x03\x0f\x0b\t\t\t!'\x1f;+y\x87.\x04!\x19+\xb1\xb3YMO{\xe9\x8b\x83\x1f/!!)#\x1f\x19\x157\x85\x87\x1f\x1f\x15\x1d\x15\x1b%)\x19'#+\x1b+\x83\x13\r#\x13\x19\x1f\x1f\x11\x17\x15\x11\x15\x17\x15\x17\x0f\x17)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00compare_v1\x00add_v1\x00convert_v1\x00select_v1\x00reshape_v1\x00return_v1\x00iota_v1\x00gather_v1\x00scatter_v1\x00func_v1\x00concatenate_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00dynamic_slice_v1\x00value\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00dimension\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_dgetrf\x00jit(<lambda>)/jit(main)/while/body/scatter[update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00jit(<lambda>)/jit(main)/while/cond/lt\x00",
    xla_call_module_version=6,
)  # End paste



# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_14['c64'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgetrf'],
    serialized_date=datetime.date(2023, 6, 14),
    inputs=(),
    expected_outputs=(array([[6. +0.j, 7. +0.j, 8. +0.j],
       [0. +0.j, 1. +0.j, 2. +0.j],
       [0.5+0.j, 0.5+0.j, 0. +0.j]], dtype=complex64), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f32>> {jax.result_info = "[0]"}, tensor<3xi32> {jax.result_info = "[1]"}, tensor<3xi32> {jax.result_info = "[2]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f32>> loc(#loc3)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc4)
    %2 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %3 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %4 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %6 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %8 = stablehlo.concatenate %5, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc5)
    %9 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %10 = stablehlo.convert %9 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %11 = stablehlo.reshape %10 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %12 = stablehlo.constant dense<> : tensor<0xi32> loc(#loc5)
    %13 = stablehlo.constant dense<1> : tensor<i64> loc(#loc5)
    %14 = stablehlo.constant dense<3> : tensor<i32> loc(#loc5)
    %15 = stablehlo.constant dense<3> : tensor<i32> loc(#loc5)
    %16:3 = stablehlo.custom_call @lapack_cgetrf(%13, %14, %15, %1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<i64>, tensor<i32>, tensor<i32>, tensor<3x3xcomplex<f32>>) -> (tensor<3x3xcomplex<f32>>, tensor<3xi32>, tensor<i32>) loc(#loc5)
    %17 = stablehlo.constant dense<1> : tensor<i32> loc(#loc5)
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc5)
    %19 = stablehlo.subtract %16#1, %18 : tensor<3xi32> loc(#loc5)
    %20 = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %22 = stablehlo.compare  GE, %16#2, %21,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %24 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc5)
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc5)
    %26 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc5)
    %27 = stablehlo.select %26, %16#0, %25 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>> loc(#loc5)
    %28 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc6)
    %29 = stablehlo.constant dense<0> : tensor<i64> loc(#loc7)
    %30 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %31:4 = stablehlo.while(%iterArg = %30, %iterArg_0 = %29, %iterArg_1 = %28, %iterArg_2 = %19) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %32 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %33 = stablehlo.compare  LT, %iterArg, %32,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc10)
      stablehlo.return %33 : tensor<i1> loc(#loc9)
    } do {
      %32 = stablehlo.constant dense<1> : tensor<i64> loc(#loc9)
      %33 = stablehlo.add %iterArg_0, %32 : tensor<i64> loc(#loc11)
      %34 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %35 = stablehlo.compare  LT, %iterArg_0, %34,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %36 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %37 = stablehlo.add %iterArg_0, %36 : tensor<i64> loc(#loc11)
      %38 = stablehlo.select %35, %37, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %39 = stablehlo.convert %38 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %41 = "stablehlo.gather"(%iterArg_2, %40) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc16)
      %42 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %43 = stablehlo.compare  LT, %iterArg_0, %42,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %44 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %45 = stablehlo.add %iterArg_0, %44 : tensor<i64> loc(#loc11)
      %46 = stablehlo.select %43, %45, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %47 = stablehlo.convert %46 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %48 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %49 = "stablehlo.gather"(%iterArg_1, %48) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc16)
      %50 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
      %51 = stablehlo.compare  LT, %41, %50,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc12)
      %52 = stablehlo.constant dense<3> : tensor<i32> loc(#loc9)
      %53 = stablehlo.add %41, %52 : tensor<i32> loc(#loc11)
      %54 = stablehlo.select %51, %53, %41 : tensor<i1>, tensor<i32> loc(#loc13)
      %55 = stablehlo.dynamic_slice %iterArg_1, %54, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc17)
      %56 = stablehlo.reshape %55 : (tensor<1xi32>) -> tensor<i32> loc(#loc18)
      %57 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %58 = stablehlo.compare  LT, %iterArg_0, %57,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %59 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %60 = stablehlo.add %iterArg_0, %59 : tensor<i64> loc(#loc11)
      %61 = stablehlo.select %58, %60, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %62 = stablehlo.convert %61 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %63 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %64 = "stablehlo.scatter"(%iterArg_1, %63, %56) ({
      ^bb0(%arg0: tensor<i32> loc(unknown), %arg1: tensor<i32> loc(unknown)):
        stablehlo.return %arg1 : tensor<i32> loc(#loc19)
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc19)
      %65 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
      %66 = stablehlo.compare  LT, %41, %65,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc12)
      %67 = stablehlo.constant dense<3> : tensor<i32> loc(#loc9)
      %68 = stablehlo.add %41, %67 : tensor<i32> loc(#loc11)
      %69 = stablehlo.select %66, %68, %41 : tensor<i1>, tensor<i32> loc(#loc13)
      %70 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %71 = "stablehlo.scatter"(%64, %70, %49) ({
      ^bb0(%arg0: tensor<i32> loc(unknown), %arg1: tensor<i32> loc(unknown)):
        stablehlo.return %arg1 : tensor<i32> loc(#loc19)
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc19)
      %72 = stablehlo.constant dense<1> : tensor<i64> loc(#loc9)
      %73 = stablehlo.add %iterArg, %72 : tensor<i64> loc(#loc11)
      stablehlo.return %73, %33, %71, %iterArg_2 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc9)
    } loc(#loc9)
    return %27, %19, %31#2 : tensor<3x3xcomplex<f32>>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":553:0)
#loc2 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":554:0)
#loc3 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]"(#loc1))
#loc4 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/lu"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc2))
#loc7 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc2))
#loc8 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1]"(#loc2))
#loc9 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc2))
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc2))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc2))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc2))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc2))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc2))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc2))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc2))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%'\x03\xaa\x02\x0e\x025\x01\xb1\x0f\x0f\x07\x17\x0b\x13\x13\x0f\x1b\x13\x0f\x0f\x13\x0f\x13\x13\x13\x0f\x0f\x0b\x13\x17\x0b\x0b\x0b\x0b;\x0b\x0b\x0b\x0f;#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x13\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x03Q\x0f\x0f/\x0b\x0f\x0b\x0bO\x0b/\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x0f/\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x17/\x1f\x1f\x0b/O/\x0b\x01\x07\x0b\x13\x0b\x01\x03\x0f\x033\x0f\x0f\x13\x13\x0f\x17\x07\x07\x0b\x07\x07\x13\x0f\x13\x1b\x07\x13\x13\x13\x13\x13\x13\x17\x17\x13\x02b\t\x1d]\x07\x1d\x8b\x07\x1f\x17-\xaa\x08\x01\x05)\x03\x03/\xb9\x03\x03\t\xd9\x1d\x8d\x07\x03\x051\xc13\xff\x03\x03\t\xfd\x1d\x8f\x07\x1d\x91\x07\x03\x03\t\xdf\x1d\x95\x07\x1d\x02\x02\x07\x03\x03\t\xdd\x03\x03\t\xf5\x1d\x93\x07\x11\x01\x05\x05+\x03\x03S\xb1\x17-\xa6\x08\x01\x05-\x05/\x051\x053\x03\r\x97\xb57\xb19\xbb\x99\xb9;\xc3\x9b\xb5\x055\x057\x059\x1d\x9d\x07\x03\r7\xb19\xbb\xa9\xb5\xab\xb5\xad\xbb\xaf\xb9\x03\x07C%E%'G\x05;\x05=\x05?\x03\x0bK\xbdM\xc5O\xc7'\xd5Q\xd7\x05A\x05C\x05E\x05G\x05I\x1dW+\x05K\x1d[+\x05M\x05O\x03\x03a\xb1\x05Q\x03\x03\t\xdb\x03\x11g\xe1i\xe3k\xe5m\xbdo\xe7q\xe9s\xebu\xef\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x03\x03\t\xf3\x03\x051\xc13\xf7\x03\x03\t\xf9\x03\x03/\xfb\x1d\x81\x07\x05c\x1d\x85\x07\x05e\x1d\x89\x07\x05g\x05i\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x03\x03;\xc3\x1d\xa3\x07\x05}\x1d\xa7\x07\x05\x7f\x05\x81\x05\x83\x05\x85\x05\x87\x13\x11\x01\x1f'\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x89\x1f-\x01\x05\x03\x03\x01\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x1f\x1d\x11\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc9\xcd\xd1\r\x03\xb7\xcb\x1d\x8b\r\x03\xb7\xcf\x1d\x8d\r\x03\xb7\xd3\x1d\x8f\x1d\x91\x1d\x93\x1f\x03\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1f\x19\x01\x1f\x03\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x03\x00\x00\x00\x0b\x05\x1d\x95\x1d\x97\x05\x01\x03\t\xb3\xb3\xb3\xbf\x03\x03\xed\x15\x03\x01\r\x01\x03\x07\xbf\xf1\xb3\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x00\x00\x00\x00\x07\x05\x1f\x1b\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x03\x11\x00\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x05\x99\x1d\n\x02\x07\x05\x9b\x01\x02\x02)\x01\x11)\x01\x0f)\x03\r\x0f)\x03\x05\x0f)\x01\x17)\x05\r\r\x13\x1b\x1d\x03!\x13\x01)\x03\x01\x0f)\x01\x13)\x03\x05\x11\x11\x01\x07\r\x07\x07\t)\x03%\x13)\x03\t\x0f)\x03\x01\x15)\x03\t\x15)\x03\x05\x15)\x03\x01\x11)\x05\x05\x05\x17)\x05\r\r\x17)\x03\t\x11\x04f\n\x05\x01\x11\x05A\x07\x03\x01\x05\x19\x11\x05I\x05\x03K\x85\x13\x03U)\x03#\x0f\x06Y\x03\r\x03\x01\x03\x03\x01\r\x03\x03\x03\x03\x01\r\x03\x03\x0b\x06\x01\x03\x05\x03\x05\x0f\x06\x01\x03\t\x03\t\x0b\x06\x01\x03\x05\x03\x07\x0f\x06\x01\x03\t\x03\r\x1b\x07\x01_\x03%\x05\x0b\x0f\x03\x03\x01\r\x03\x03\x0b\x06\x01\x03\x05\x03\x13\x0f\x06\x01\x03\t\x03\x15\x03\x03\x01c\x03\x19\x03\x03\x01\x1f\x03\x03\x03\x03\x01\x19\x03\x05\x03\x03\x01\x19\x03\x05\x1d\x07\x01e\x07\r\x07\x05\t\x1b\x1d\x1f\x03\x03\x03\x01w\x03\x05\x05\x07\x01\x0b\x03\x07\x03'\x1f\x06\x01\x03\x07\x05#)\x03\x03\x01!\x03\x05\x05\x07\x01\x0b\x03\x05\x03-\x07\x07\x01y\x03\x0b\x05%/\x05\x07\x01\x0b\x03/\x031\x03\x03\x01{\x03\x1b\x05\x07\x01\x0b\x03\r\x035\x05\x07\x01}\x031\x033\r\x06\x01\x03\r\x079!7\x13\x03\x7f)\x03\x07\x03\x03\x83\x13\x03\x03\x03\x03\x87\x13\x03\x03!\x16\x03\t\x03\x03\x07\x07\tA?=+\t\x03\r\x0f\t\x03\x05\x03\x05\x07\x05\x07\x05\x03\x03\x03\r\x03\x03\x07\x07\x06\x02\x11\x03\x0b\x05KS\x11\x04\x03\x03U\x03]\xaf\t\x03\x05\x03\x05\x07\x05\x07\x05\x03\x03\x03\x1f\x03\x03\t\x06\x0f\x03\x03\x05MS\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05MW\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05M[\r\x06\x17\x03\x03\x07Y]M\x0b\x06#\x03\x05\x03_\x05\x07\x1b\x0b\x03\t\x03a\x15\x07=5\x03\x05\x05Qc\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05Mg\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05Mk\r\x06\x17\x03\x03\x07imM\x0b\x06#\x03\x05\x03o\x05\x07\x1b\x0b\x03\t\x03q\x15\x07=5\x03\x05\x05Os\x03\x03\x03!\x03\x05\x07\x07\x15\x11\x03\x0b\x05ew\x03\x03\x03\x19\x03\x05\t\x06\x0f\x03\x05\x05e{\r\x06\x17\x03\x05\x07y}e#\x07\xa1\x9f\x03\t\x05O\x7f\x0f\x06\xa5\x03\x05\x03\x81\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05M\x85\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05M\x89\r\x06\x17\x03\x03\x07\x87\x8bM\x0b\x06#\x03\x05\x03\x8d\x05\x07\x1b\x0b\x03\t\x03\x8f\x17\x17\x1d?\x03\x07\x07O\x91\x83\x05\x03\x05\x07\x05\x05\x05\x05\x05\x11\x04\x1d\x03\xa9\x03\x03\x03!\x03\x05\x07\x07\x15\x11\x03\x0b\x05e\x95\x03\x03\x03\x19\x03\x05\t\x06\x0f\x03\x05\x05e\x99\r\x06\x17\x03\x05\x07\x97\x9be\x05\x07\x1b\x0b\x03\t\x03\x9d\x17\x17\x1d?\x03\x07\x07\x93\x9fu\x05\x03\x05\x07\x05\x05\x05\x05\x05\x11\x04\x1d\x03\xa9\x03\x03\x03\x1f\x03\x03\t\x06\x0f\x03\x03\x05K\xa3\x11\x04\x03\t\xa5U\xa1Q\x11\x04\x05\x07;+G\x06\x03\x01\x05\x01\x00~%\x9dM2\x04\x1d\x03\x0f\x0b\t\t\t!'\x1f;+y\x87.\x04!\x19+\xb1\xb3YMO{\xe9\x8b\x83\x1f/!!)#\x1f\x19\x157\x85\x8b\x1f\x1f\x15\x1d\x15\x1b%)\x19'#+\x1b+\x83\x13\r#\x13\x19\x1f\x1f\x11\x17\x15\x11\x15\x17\x15\x17\x0f\x17)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00compare_v1\x00add_v1\x00convert_v1\x00select_v1\x00reshape_v1\x00return_v1\x00iota_v1\x00gather_v1\x00scatter_v1\x00func_v1\x00concatenate_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00dynamic_slice_v1\x00value\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00dimension\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_cgetrf\x00jit(<lambda>)/jit(main)/while/body/scatter[update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00jit(<lambda>)/jit(main)/while/cond/lt\x00",
    xla_call_module_version=6,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_14['c128'] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgetrf'],
    serialized_date=datetime.date(2023, 6, 14),
    inputs=(),
    expected_outputs=(array([[6. +0.j, 7. +0.j, 8. +0.j],
       [0. +0.j, 1. +0.j, 2. +0.j],
       [0.5+0.j, 0.5+0.j, 0. +0.j]]), array([2, 2, 2], dtype=int32), array([2, 0, 1], dtype=int32)),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f64>> {jax.result_info = "[0]"}, tensor<3xi32> {jax.result_info = "[1]"}, tensor<3xi32> {jax.result_info = "[2]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f64>> loc(#loc3)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc4)
    %2 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %3 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %4 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %6 = stablehlo.convert %3 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %8 = stablehlo.concatenate %5, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32> loc(#loc5)
    %9 = stablehlo.constant dense<3> : tensor<i64> loc(#loc5)
    %10 = stablehlo.convert %9 : (tensor<i64>) -> tensor<i32> loc(#loc5)
    %11 = stablehlo.reshape %10 : (tensor<i32>) -> tensor<1xi32> loc(#loc5)
    %12 = stablehlo.constant dense<> : tensor<0xi32> loc(#loc5)
    %13 = stablehlo.constant dense<1> : tensor<i64> loc(#loc5)
    %14 = stablehlo.constant dense<3> : tensor<i32> loc(#loc5)
    %15 = stablehlo.constant dense<3> : tensor<i32> loc(#loc5)
    %16:3 = stablehlo.custom_call @lapack_zgetrf(%13, %14, %15, %1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<i64>, tensor<i32>, tensor<i32>, tensor<3x3xcomplex<f64>>) -> (tensor<3x3xcomplex<f64>>, tensor<3xi32>, tensor<i32>) loc(#loc5)
    %17 = stablehlo.constant dense<1> : tensor<i32> loc(#loc5)
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<3xi32> loc(#loc5)
    %19 = stablehlo.subtract %16#1, %18 : tensor<3xi32> loc(#loc5)
    %20 = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %22 = stablehlo.compare  GE, %16#2, %21,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %24 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc5)
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc5)
    %26 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc5)
    %27 = stablehlo.select %26, %16#0, %25 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>> loc(#loc5)
    %28 = stablehlo.iota dim = 0 : tensor<3xi32> loc(#loc6)
    %29 = stablehlo.constant dense<0> : tensor<i64> loc(#loc7)
    %30 = stablehlo.constant dense<0> : tensor<i64> loc(#loc8)
    %31:4 = stablehlo.while(%iterArg = %30, %iterArg_0 = %29, %iterArg_1 = %28, %iterArg_2 = %19) : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32>
     cond {
      %32 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %33 = stablehlo.compare  LT, %iterArg, %32,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc10)
      stablehlo.return %33 : tensor<i1> loc(#loc9)
    } do {
      %32 = stablehlo.constant dense<1> : tensor<i64> loc(#loc9)
      %33 = stablehlo.add %iterArg_0, %32 : tensor<i64> loc(#loc11)
      %34 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %35 = stablehlo.compare  LT, %iterArg_0, %34,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %36 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %37 = stablehlo.add %iterArg_0, %36 : tensor<i64> loc(#loc11)
      %38 = stablehlo.select %35, %37, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %39 = stablehlo.convert %38 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %41 = "stablehlo.gather"(%iterArg_2, %40) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc16)
      %42 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %43 = stablehlo.compare  LT, %iterArg_0, %42,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %44 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %45 = stablehlo.add %iterArg_0, %44 : tensor<i64> loc(#loc11)
      %46 = stablehlo.select %43, %45, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %47 = stablehlo.convert %46 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %48 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %49 = "stablehlo.gather"(%iterArg_1, %48) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<3xi32>, tensor<1xi32>) -> tensor<i32> loc(#loc16)
      %50 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
      %51 = stablehlo.compare  LT, %41, %50,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc12)
      %52 = stablehlo.constant dense<3> : tensor<i32> loc(#loc9)
      %53 = stablehlo.add %41, %52 : tensor<i32> loc(#loc11)
      %54 = stablehlo.select %51, %53, %41 : tensor<i1>, tensor<i32> loc(#loc13)
      %55 = stablehlo.dynamic_slice %iterArg_1, %54, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32> loc(#loc17)
      %56 = stablehlo.reshape %55 : (tensor<1xi32>) -> tensor<i32> loc(#loc18)
      %57 = stablehlo.constant dense<0> : tensor<i64> loc(#loc9)
      %58 = stablehlo.compare  LT, %iterArg_0, %57,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc12)
      %59 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
      %60 = stablehlo.add %iterArg_0, %59 : tensor<i64> loc(#loc11)
      %61 = stablehlo.select %58, %60, %iterArg_0 : tensor<i1>, tensor<i64> loc(#loc13)
      %62 = stablehlo.convert %61 : (tensor<i64>) -> tensor<i32> loc(#loc14)
      %63 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %64 = "stablehlo.scatter"(%iterArg_1, %63, %56) ({
      ^bb0(%arg0: tensor<i32> loc(unknown), %arg1: tensor<i32> loc(unknown)):
        stablehlo.return %arg1 : tensor<i32> loc(#loc19)
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc19)
      %65 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
      %66 = stablehlo.compare  LT, %41, %65,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc12)
      %67 = stablehlo.constant dense<3> : tensor<i32> loc(#loc9)
      %68 = stablehlo.add %41, %67 : tensor<i32> loc(#loc11)
      %69 = stablehlo.select %66, %68, %41 : tensor<i1>, tensor<i32> loc(#loc13)
      %70 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<i32>) -> tensor<1xi32> loc(#loc15)
      %71 = "stablehlo.scatter"(%64, %70, %49) ({
      ^bb0(%arg0: tensor<i32> loc(unknown), %arg1: tensor<i32> loc(unknown)):
        stablehlo.return %arg1 : tensor<i32> loc(#loc19)
      }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32> loc(#loc19)
      %72 = stablehlo.constant dense<1> : tensor<i64> loc(#loc9)
      %73 = stablehlo.add %iterArg, %72 : tensor<i64> loc(#loc11)
      stablehlo.return %73, %33, %71, %iterArg_2 : tensor<i64>, tensor<i64>, tensor<3xi32>, tensor<3xi32> loc(#loc9)
    } loc(#loc9)
    return %27, %19, %31#2 : tensor<3x3xcomplex<f64>>, tensor<3xi32>, tensor<3xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":553:0)
#loc2 = loc("third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py":554:0)
#loc3 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]"(#loc1))
#loc4 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/lu"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]"(#loc2))
#loc7 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]"(#loc2))
#loc8 = loc("jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1]"(#loc2))
#loc9 = loc("jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/while/cond/lt"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/while/body/add"(#loc2))
#loc12 = loc("jit(<lambda>)/jit(main)/while/body/lt"(#loc2))
#loc13 = loc("jit(<lambda>)/jit(main)/while/body/select_n"(#loc2))
#loc14 = loc("jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]"(#loc2))
#loc15 = loc("jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]"(#loc2))
#loc16 = loc("jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc2))
#loc17 = loc("jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]"(#loc2))
#loc18 = loc("jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]"(#loc2))
#loc19 = loc("jit(<lambda>)/jit(main)/while/body/scatter[update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%'\x03\xaa\x02\x0e\x025\x01\xb1\x0f\x0f\x07\x17\x0b\x13\x13\x0f\x1b\x13\x0f\x0f\x13\x0f\x13\x13\x13\x0f\x0f\x0b\x13\x17\x0b\x0b\x0b\x0b;\x0b\x0b\x0b\x0f;#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x13\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x03Q\x0f\x0f/\x0b\x0f\x0b\x0bO\x0b/\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x0f/\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x17/\x1f\x1f\x0bOO/\x0b\x01\x07\x0b\x13\x0b\x01\x03\x0f\x033\x0f\x0f\x13\x13\x0f\x17\x07\x07\x0b\x07\x07\x13\x0f\x13\x1b\x07\x13\x13\x13\x13\x13\x13\x17\x17\x13\x02\x82\t\x1d]\x07\x1d\x8b\x07\x1f\x17-\xaa\x08\x01\x05)\x03\x03/\xb9\x03\x03\t\xd9\x1d\x8d\x07\x03\x051\xc13\xff\x03\x03\t\xfd\x1d\x8f\x07\x1d\x91\x07\x03\x03\t\xdf\x1d\x95\x07\x1d\x02\x02\x07\x03\x03\t\xdd\x03\x03\t\xf5\x1d\x93\x07\x11\x01\x05\x05+\x03\x03S\xb1\x17-\xa6\x08\x01\x05-\x05/\x051\x053\x03\r\x97\xb57\xb19\xbb\x99\xb9;\xc3\x9b\xb5\x055\x057\x059\x1d\x9d\x07\x03\r7\xb19\xbb\xa9\xb5\xab\xb5\xad\xbb\xaf\xb9\x03\x07C%E%'G\x05;\x05=\x05?\x03\x0bK\xbdM\xc5O\xc7'\xd5Q\xd7\x05A\x05C\x05E\x05G\x05I\x1dW+\x05K\x1d[+\x05M\x05O\x03\x03a\xb1\x05Q\x03\x03\t\xdb\x03\x11g\xe1i\xe3k\xe5m\xbdo\xe7q\xe9s\xebu\xef\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x03\x03\t\xf3\x03\x051\xc13\xf7\x03\x03\t\xf9\x03\x03/\xfb\x1d\x81\x07\x05c\x1d\x85\x07\x05e\x1d\x89\x07\x05g\x05i\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x03\x03;\xc3\x1d\xa3\x07\x05}\x1d\xa7\x07\x05\x7f\x05\x81\x05\x83\x05\x85\x05\x87\x13\x11\x01\x1f'\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x89\x1f-\x01\x05\x03\x03\x01\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x1f\x1d\x11\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x07\xc9\xcd\xd1\r\x03\xb7\xcb\x1d\x8b\r\x03\xb7\xcf\x1d\x8d\r\x03\xb7\xd3\x1d\x8f\x1d\x91\x1d\x93\x1f\x03\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1f\x19\x01\x1f\x03\x11\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x03\x00\x00\x00\x0b\x05\x1d\x95\x1d\x97\x05\x01\x03\t\xb3\xb3\xb3\xbf\x03\x03\xed\x15\x03\x01\r\x01\x03\x07\xbf\xf1\xb3\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x00\x00\x00\x00\x07\x05\x1f\x1b!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x03\x11\x00\x00\x00\x00\x00\x00\x00\x00\x07\x0b\x05\x99\x1d\n\x02\x07\x05\x9b\x01\x02\x02)\x01\x11)\x01\x0f)\x03\r\x0f)\x03\x05\x0f)\x01\x17)\x05\r\r\x13\x1b\x1d\x03!\x13\x01)\x03\x01\x0f)\x01\x13)\x03\x05\x11\x11\x01\x07\r\x07\x07\x0b)\x03%\x13)\x03\t\x0f)\x03\x01\x15)\x03\t\x15)\x03\x05\x15)\x03\x01\x11)\x05\x05\x05\x17)\x05\r\r\x17)\x03\t\x11\x04f\n\x05\x01\x11\x05A\x07\x03\x01\x05\x19\x11\x05I\x05\x03K\x85\x13\x03U)\x03#\x0f\x06Y\x03\r\x03\x01\x03\x03\x01\r\x03\x03\x03\x03\x01\r\x03\x03\x0b\x06\x01\x03\x05\x03\x05\x0f\x06\x01\x03\t\x03\t\x0b\x06\x01\x03\x05\x03\x07\x0f\x06\x01\x03\t\x03\r\x1b\x07\x01_\x03%\x05\x0b\x0f\x03\x03\x01\r\x03\x03\x0b\x06\x01\x03\x05\x03\x13\x0f\x06\x01\x03\t\x03\x15\x03\x03\x01c\x03\x19\x03\x03\x01\x1f\x03\x03\x03\x03\x01\x19\x03\x05\x03\x03\x01\x19\x03\x05\x1d\x07\x01e\x07\r\x07\x05\t\x1b\x1d\x1f\x03\x03\x03\x01w\x03\x05\x05\x07\x01\x0b\x03\x07\x03'\x1f\x06\x01\x03\x07\x05#)\x03\x03\x01!\x03\x05\x05\x07\x01\x0b\x03\x05\x03-\x07\x07\x01y\x03\x0b\x05%/\x05\x07\x01\x0b\x03/\x031\x03\x03\x01{\x03\x1b\x05\x07\x01\x0b\x03\r\x035\x05\x07\x01}\x031\x033\r\x06\x01\x03\r\x079!7\x13\x03\x7f)\x03\x07\x03\x03\x83\x13\x03\x03\x03\x03\x87\x13\x03\x03!\x16\x03\t\x03\x03\x07\x07\tA?=+\t\x03\r\x0f\t\x03\x05\x03\x05\x07\x05\x07\x05\x03\x03\x03\r\x03\x03\x07\x07\x06\x02\x11\x03\x0b\x05KS\x11\x04\x03\x03U\x03]\xaf\t\x03\x05\x03\x05\x07\x05\x07\x05\x03\x03\x03\x1f\x03\x03\t\x06\x0f\x03\x03\x05MS\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05MW\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05M[\r\x06\x17\x03\x03\x07Y]M\x0b\x06#\x03\x05\x03_\x05\x07\x1b\x0b\x03\t\x03a\x15\x07=5\x03\x05\x05Qc\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05Mg\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05Mk\r\x06\x17\x03\x03\x07imM\x0b\x06#\x03\x05\x03o\x05\x07\x1b\x0b\x03\t\x03q\x15\x07=5\x03\x05\x05Os\x03\x03\x03!\x03\x05\x07\x07\x15\x11\x03\x0b\x05ew\x03\x03\x03\x19\x03\x05\t\x06\x0f\x03\x05\x05e{\r\x06\x17\x03\x05\x07y}e#\x07\xa1\x9f\x03\t\x05O\x7f\x0f\x06\xa5\x03\x05\x03\x81\x03\x03\x03\x13\x03\x03\x07\x07\x15\x11\x03\x0b\x05M\x85\x03\x03\x03\r\x03\x03\t\x06\x0f\x03\x03\x05M\x89\r\x06\x17\x03\x03\x07\x87\x8bM\x0b\x06#\x03\x05\x03\x8d\x05\x07\x1b\x0b\x03\t\x03\x8f\x17\x17\x1d?\x03\x07\x07O\x91\x83\x05\x03\x05\x07\x05\x05\x05\x05\x05\x11\x04\x1d\x03\xa9\x03\x03\x03!\x03\x05\x07\x07\x15\x11\x03\x0b\x05e\x95\x03\x03\x03\x19\x03\x05\t\x06\x0f\x03\x05\x05e\x99\r\x06\x17\x03\x05\x07\x97\x9be\x05\x07\x1b\x0b\x03\t\x03\x9d\x17\x17\x1d?\x03\x07\x07\x93\x9fu\x05\x03\x05\x07\x05\x05\x05\x05\x05\x11\x04\x1d\x03\xa9\x03\x03\x03\x1f\x03\x03\t\x06\x0f\x03\x03\x05K\xa3\x11\x04\x03\t\xa5U\xa1Q\x11\x04\x05\x07;+G\x06\x03\x01\x05\x01\x00\x82%\x9dM2\x04\x1d\x03\x0f\x0b\t\t\t!'\x1f;+y\x87.\x04!\x19+\xb1\xb3YMO{\xe9\x8b\x83\x1f/!!)#\x1f\x19\x157\x85\x8d\x1f\x1f\x15\x1d\x15\x1b%)\x19'#+\x1b+\x83\x13\r#\x13\x19\x1f\x1f\x11\x17\x15\x11\x15\x17\x15\x17\x0f\x17)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00compare_v1\x00add_v1\x00convert_v1\x00select_v1\x00reshape_v1\x00return_v1\x00iota_v1\x00gather_v1\x00scatter_v1\x00func_v1\x00concatenate_v1\x00custom_call_v1\x00subtract_v1\x00while_v1\x00dynamic_slice_v1\x00value\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00index_vector_dim\x00indices_are_sorted\x00slice_sizes\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/lu\x00dimension\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(3,) dimension=0]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=3]\x00jit(<lambda>)/jit(main)/scan[reverse=False length=3 num_consts=0 num_carry=3 linear=(False, False, False) unroll=1]\x00jit(<lambda>)/jit(main)/while[cond_nconsts=0 body_nconsts=0]\x00jit(<lambda>)/jit(main)/while/body/add\x00jit(<lambda>)/jit(main)/while/body/lt\x00jit(<lambda>)/jit(main)/while/body/select_n\x00jit(<lambda>)/jit(main)/while/body/convert_element_type[new_dtype=int32 weak_type=False]\x00jit(<lambda>)/jit(main)/while/body/broadcast_in_dim[shape=(1,) broadcast_dimensions=()]\x00collapsed_slice_dims\x00offset_dims\x00start_index_map\x00jit(<lambda>)/jit(main)/while/body/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)) slice_sizes=(1,) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00jit(<lambda>)/jit(main)/while/body/dynamic_slice[slice_sizes=(1,)]\x00jit(<lambda>)/jit(main)/while/body/squeeze[dimensions=(0,)]\x00inserted_window_dims\x00scatter_dims_to_operand_dims\x00unique_indices\x00update_window_dims\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_zgetrf\x00jit(<lambda>)/jit(main)/while/body/scatter[update_consts=() dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)) indices_are_sorted=True unique_indices=True mode=GatherScatterMode.FILL_OR_DROP]\x00jit(<lambda>)/jit(main)/while/cond/lt\x00",
    xla_call_module_version=6,
)  # End paste
