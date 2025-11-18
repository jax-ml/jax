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

data_2024_09_30 = {}

data_2024_09_30["f32"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_syevd_ffi'],
    serialized_date=datetime.date(2024, 9, 30),
    inputs=(),
    expected_outputs=(array([[ 0.7941186  , -0.3696443  , -0.40418202 ,  0.26339266 ],
       [ 0.3696443  ,  0.7941186  ,  0.26339266 ,  0.4041819  ],
       [-0.054829806, -0.47930413 ,  0.6857606  ,  0.5449713  ],
       [-0.4793042  ,  0.05482992 , -0.5449712  ,  0.68576056 ]],
      dtype=float32), array([-3.7082872e+00, -4.0793765e-07,  4.4458108e-07,  3.3708286e+01],
      dtype=float32)),
    mlir_module_text=r"""
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":274:27)
#loc13 = loc("jit(<lambda>)/jit(main)/pjit"(#loc6))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<16xf32> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<16xf32>) -> tensor<4x4xf32> loc(#loc9)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc10)
    %3 = stablehlo.add %1, %2 : tensor<4x4xf32> loc(#loc11)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc12)
    %5 = stablehlo.divide %3, %4 : tensor<4x4xf32> loc(#loc12)
    %6 = call @tril(%5) : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc13)
    %7:3 = stablehlo.custom_call @cusolver_syevd_ffi(%6) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>, tensor<i32>) loc(#loc14)
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc14)
    %9 = stablehlo.compare  EQ, %7#2, %8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc14)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc14)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc14)
    %13 = stablehlo.select %12, %7#0, %11 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc14)
    %14 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc14)
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32> loc(#loc14)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc14)
    %17 = stablehlo.select %16, %7#1, %15 : tensor<4xi1>, tensor<4xf32> loc(#loc14)
    return %13, %17 : tensor<4x4xf32>, tensor<4xf32> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xf32> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit"(#loc6))) -> (tensor<4x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc15)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc16)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc16)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc15)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc17)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc18)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc19)
    return %6 : tensor<4x4xf32> loc(#loc13)
  } loc(#loc13)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":266:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":266:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:15)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:14)
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":274:11)
#loc8 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc9 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/transpose"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/add"(#loc4))
#loc12 = loc("jit(<lambda>)/jit(main)/div"(#loc5))
#loc14 = loc("jit(<lambda>)/jit(main)/eigh"(#loc7))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota"(#loc6))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc6))
#loc17 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc6))
#loc18 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc6))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x01-\x05\x01\x05\x1d\x01\x03\x0b\x03\x1b\x0f\x13\x17\x1b\x1f#'+/37;?\x03\xfb\xb17\x01U\x0f\x07\x0b\x17\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0f\x0b\x17\x0f\x0b\x17\x0b\x17\x13\x0b\x0b\x17\x03]\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x1f\x0f\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x1f\x0f\x0b\x1f\x1fO\x1b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x0bO/\x01\x05\x0b\x0f\x033\x17\x0f\x0f\x07\x07\x07\x13\x17\x07\x07\x17\x13\x17\x17\x13\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02\xae\x06\x1dQS\x1f\x05!\x17\x05J\x047\x1d\x1f\x07\x11\x03\x05\x1d!\x07\x1d#\x07\x1dIK\x03\x07\x15\x17\x19\x0b\x1b\x0b\x05#\x11\x01\x00\x05%\x05'\x05)\x05+\x05-\x05/\x1d'\x07\x051\x1d+\x07\x053\x1d/\x07\x055\x1d35\x057\x17\x05*\x045\x1d9;\x059\x17\x05*\x04\x1d\x1d?A\x05;\x17\x052\x04E\x1dEG\x05=\x17\x052\x04\x1f\x05?\x17\x052\x04\x1d\x03\x03O\x8d\x05A\x05C\x17\x05J\x04\x17\x1f!\x01\x1dE\x1dG\x03\x01\x1dI\x03\x03{\x1dK\x1f\t\t\x00\x00\x00\x00\x13\x0b\x01\t\x07\x1f'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1d\x03\x05os\r\x05]qWY\x1dM\r\x05]uWY\x1dO\x1dQ\x1dS\r\x03WY#\x1f\x1dU\x1f\x07\t\x00\x00\x00\x00\x13\x0b\x05\x07\x05\x1f\x07\t\x00\x00\xc0\x7f\x1f\x07\t\x00\x00\x00@\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\x8f\x91\x93\x95\x1dW\x13%\x00\x1dY\x05\x03\x0b\x03\x1d[\x1d]\x05\x01\x03\x03i\x03\x03\xa3\x15\x03\x01\x01\x01\x03\x07i\xa7\xa9\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x01\x07\x01\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x0f)\x01\x0f)\x01\x17\x1d\x01\t)\x03\x11\x0f)\x05\x11\x11\x17\x13\x1b)\x05\x11\x11\r)\x03\t\x0b\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03A\x0f!)\x03\t\x15)\x03\x05\x15)\x03\x01\x15)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03\x11\r)\x03\x05\x0b\x04b\x04\x05\x01Q\x03\x13\x01\x07\x04:\x04\x03\x01\t\x0bP\x03\x03\x07\x04\xba\x02\x03/Y\x05B\x03\x05\x03\x07\x05B\x03\x07\x03\t\x05B\x03\t\x03\x07\x07B1\x0b\x03#\x13\x067\x03\x05\x03\x07\x15F=\r\x03\x05\x03\t\r\x06C\x03\x05\x05\t\x0b\x03F\x11\x0f\x03\x05\x03\x05\x17\x06\x11\x03\x05\x05\r\x0f\x19F\t\x11\x03\x05\x03\x11\x1bG\x01M\x13\x07\x05\x11\t\x03\x13\x03F\x01\x0f\x03\t\x03\x03\x0fF\x01\x15\x03-\x05\x19\x1b\x03F\x01\x0f\x03/\x03\x1d\x03F\x01\x0f\x03\x05\x03\x01\x03F\x01\x17\x03\x19\x03\x1f\t\x06\x01\x03\x05\x07#\x15!\x03F\x01\x0f\x031\x03\x1d\x03F\x01\x0f\x03\x11\x03\x01\x03F\x01\x19\x033\x03'\t\x06\x01\x03\x11\x07+\x17)\x11\x04\x03\x05%-\x0bP\t\x1b\x07\x04\x9d\x03\x15+\x03\x0b\t\x00\x05B\x03\x1d\x03\x07\x05B\x03\x07\x03\t\x07B\r\x0b\x03\x13\x03F\x0f\x0f\x03\x13\x03\x05\r\x06\x0f\x03\x13\x05\x07\t\x07B\r\x1f\x03\x13\x0fF%!\x03\x19\x05\x0b\r\x03F)\x0f\x03\x05\x03\x03\t\x06-\x03\x05\x07\x0f\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xe6\r_'\x03\r\x15\x11\x0f\x0b\t\t\x0b!\x11#;)99EA;WgKMO;\x1b%)9i\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit\x00jit(<lambda>)/jit(main)/jit(tril)/iota\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota\x00jit(<lambda>)/jit(main)/reshape\x00jit(<lambda>)/jit(main)/transpose\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/eigh\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00\x00cusolver_syevd_ffi\x00\x08k#\x05;\x01\x0b[kmwy\x03\x87\x03c\x03\x89\x03e\x03\x8b\x03U\x03a\x11\x97\x99\x9b[\x9d\x9f\xa1\xa5\x05g\xab\x03\xad\x03\xaf\x0b_}_a\x7f\x03\x81\x03\x83\x05g\x85",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_09_30["f64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_syevd_ffi'],
    serialized_date=datetime.date(2024, 9, 30),
    inputs=(),
    expected_outputs=(array([[ 0.7941185704969033 , -0.36964433974346045, -0.4041819665640973 ,
         0.2633926650306618 ],
       [ 0.3696443397434605 ,  0.7941185704969035 ,  0.2633926650306616 ,
         0.4041819665640974 ],
       [-0.05482989100998295, -0.47930412176342563,  0.6857605696309688 ,
         0.544971268097533  ],
       [-0.47930412176342574,  0.05482989100998273, -0.544971268097533  ,
         0.6857605696309688 ]]), array([-3.7082869338697053e+00,  7.7329581044653176e-17,
        8.6623770428558249e-16,  3.3708286933869694e+01])),
    mlir_module_text=r"""
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":274:27)
#loc13 = loc("jit(<lambda>)/jit(main)/pjit"(#loc6))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<16xf64> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<16xf64>) -> tensor<4x4xf64> loc(#loc9)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc10)
    %3 = stablehlo.add %1, %2 : tensor<4x4xf64> loc(#loc11)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc12)
    %5 = stablehlo.divide %3, %4 : tensor<4x4xf64> loc(#loc12)
    %6 = call @tril(%5) : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc13)
    %7:3 = stablehlo.custom_call @cusolver_syevd_ffi(%6) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xf64>, tensor<i32>) loc(#loc14)
    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc14)
    %9 = stablehlo.compare  EQ, %7#2, %8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc14)
    %11 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc14)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc14)
    %13 = stablehlo.select %12, %7#0, %11 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc14)
    %14 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc14)
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64> loc(#loc14)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc14)
    %17 = stablehlo.select %16, %7#1, %15 : tensor<4xi1>, tensor<4xf64> loc(#loc14)
    return %13, %17 : tensor<4x4xf64>, tensor<4xf64> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xf64> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit"(#loc6))) -> (tensor<4x4xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc15)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc16)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc16)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc15)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc17)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc18)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc19)
    return %6 : tensor<4x4xf64> loc(#loc13)
  } loc(#loc13)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":266:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":266:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:15)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:14)
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":274:11)
#loc8 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc9 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/transpose"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/add"(#loc4))
#loc12 = loc("jit(<lambda>)/jit(main)/div"(#loc5))
#loc14 = loc("jit(<lambda>)/jit(main)/eigh"(#loc7))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota"(#loc6))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc6))
#loc17 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc6))
#loc18 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc6))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x01-\x05\x01\x05\x1d\x01\x03\x0b\x03\x1b\x0f\x13\x17\x1b\x1f#'+/37;?\x03\xfb\xb17\x01U\x0f\x07\x0b\x17\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0f\x0b\x17\x0f\x0b\x17\x0b\x17\x13\x0b\x0b\x17\x03]\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x1f\x0f\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b/\x0f\x0b//O\x1b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x0bO/\x01\x05\x0b\x0f\x033\x17\x0f\x0f\x07\x07\x07\x13\x17\x07\x07\x17\x13\x17\x17\x13\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02\xde\x06\x1dQS\x1f\x05!\x17\x05J\x047\x1d\x1f\x07\x11\x03\x05\x1d!\x07\x1d#\x07\x1dIK\x03\x07\x15\x17\x19\x0b\x1b\x0b\x05#\x11\x01\x00\x05%\x05'\x05)\x05+\x05-\x05/\x1d'\x07\x051\x1d+\x07\x053\x1d/\x07\x055\x1d35\x057\x17\x05*\x045\x1d9;\x059\x17\x05*\x04\x1d\x1d?A\x05;\x17\x052\x04E\x1dEG\x05=\x17\x052\x04\x1f\x05?\x17\x052\x04\x1d\x03\x03O\x8d\x05A\x05C\x17\x05J\x04\x17\x1f!\x01\x1dE\x1dG\x03\x01\x1dI\x03\x03{\x1dK\x1f\t\t\x00\x00\x00\x00\x13\x0b\x01\t\x07\x1f'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1d\x03\x05os\r\x05]qWY\x1dM\r\x05]uWY\x1dO\x1dQ\x1dS\r\x03WY#\x1f\x1dU\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\x8f\x91\x93\x95\x1dW\x13%\x00\x1dY\x05\x03\x0b\x03\x1d[\x1d]\x05\x01\x03\x03i\x03\x03\xa3\x15\x03\x01\x01\x01\x03\x07i\xa7\xa9\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x01\x07\x01\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x0f)\x01\x0f)\x01\x17\x1d\x01\x0b)\x03\x11\x0f)\x05\x11\x11\x17\x13\x1b)\x05\x11\x11\r)\x03\t\x0b\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03A\x0f!)\x03\t\x15)\x03\x05\x15)\x03\x01\x15)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03\x11\r)\x03\x05\x0b\x04b\x04\x05\x01Q\x03\x13\x01\x07\x04:\x04\x03\x01\t\x0bP\x03\x03\x07\x04\xba\x02\x03/Y\x05B\x03\x05\x03\x07\x05B\x03\x07\x03\t\x05B\x03\t\x03\x07\x07B1\x0b\x03#\x13\x067\x03\x05\x03\x07\x15F=\r\x03\x05\x03\t\r\x06C\x03\x05\x05\t\x0b\x03F\x11\x0f\x03\x05\x03\x05\x17\x06\x11\x03\x05\x05\r\x0f\x19F\t\x11\x03\x05\x03\x11\x1bG\x01M\x13\x07\x05\x11\t\x03\x13\x03F\x01\x0f\x03\t\x03\x03\x0fF\x01\x15\x03-\x05\x19\x1b\x03F\x01\x0f\x03/\x03\x1d\x03F\x01\x0f\x03\x05\x03\x01\x03F\x01\x17\x03\x19\x03\x1f\t\x06\x01\x03\x05\x07#\x15!\x03F\x01\x0f\x031\x03\x1d\x03F\x01\x0f\x03\x11\x03\x01\x03F\x01\x19\x033\x03'\t\x06\x01\x03\x11\x07+\x17)\x11\x04\x03\x05%-\x0bP\t\x1b\x07\x04\x9d\x03\x15+\x03\x0b\t\x00\x05B\x03\x1d\x03\x07\x05B\x03\x07\x03\t\x07B\r\x0b\x03\x13\x03F\x0f\x0f\x03\x13\x03\x05\r\x06\x0f\x03\x13\x05\x07\t\x07B\r\x1f\x03\x13\x0fF%!\x03\x19\x05\x0b\r\x03F)\x0f\x03\x05\x03\x03\t\x06-\x03\x05\x07\x0f\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xe6\r_'\x03\r\x15\x11\x0f\x0b\t\t\x0b!\x11#;)99EA;WgKMO;\x1b%)9i\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit\x00jit(<lambda>)/jit(main)/jit(tril)/iota\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota\x00jit(<lambda>)/jit(main)/reshape\x00jit(<lambda>)/jit(main)/transpose\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/eigh\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00\x00cusolver_syevd_ffi\x00\x08k#\x05;\x01\x0b[kmwy\x03\x87\x03c\x03\x89\x03e\x03\x8b\x03U\x03a\x11\x97\x99\x9b[\x9d\x9f\xa1\xa5\x05g\xab\x03\xad\x03\xaf\x0b_}_a\x7f\x03\x81\x03\x83\x05g\x85",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_09_30["c64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_syevd_ffi'],
    serialized_date=datetime.date(2024, 9, 30),
    inputs=(),
    expected_outputs=(array([[ 0.79411864 +0.j,  0.3696443  +0.j,  0.40418214 +0.j,
        -0.26339263 +0.j],
       [ 0.3696443  +0.j, -0.7941186  +0.j, -0.26339272 +0.j,
        -0.40418193 +0.j],
       [-0.054829765+0.j,  0.47930422 +0.j, -0.6857606  +0.j,
        -0.5449713  +0.j],
       [-0.47930422 +0.j, -0.054829985+0.j,  0.5449712  +0.j,
        -0.6857606  +0.j]], dtype=complex64), array([-3.7082872e+00, -2.9983883e-07,  3.5983098e-07,  3.3708286e+01],
      dtype=float32)),
    mlir_module_text=r"""
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":274:27)
#loc18 = loc("jit(<lambda>)/jit(main)/pjit"(#loc7))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %cst_1 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<16xcomplex<f32>> loc(#loc9)
    %1 = stablehlo.reshape %0 : (tensor<16xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc10)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc11)
    %3 = stablehlo.real %2 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc12)
    %4 = stablehlo.imag %2 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc13)
    %5 = stablehlo.negate %4 : tensor<4x4xf32> loc(#loc14)
    %6 = stablehlo.complex %3, %5 : tensor<4x4xcomplex<f32>> loc(#loc15)
    %7 = stablehlo.add %1, %6 : tensor<4x4xcomplex<f32>> loc(#loc16)
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc17)
    %9 = stablehlo.divide %7, %8 : tensor<4x4xcomplex<f32>> loc(#loc17)
    %10 = call @tril(%9) : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc18)
    %11:3 = stablehlo.custom_call @cusolver_syevd_ffi(%10) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<4xf32>, tensor<i32>) loc(#loc19)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc19)
    %13 = stablehlo.compare  EQ, %11#2, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc19)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc19)
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc19)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc19)
    %17 = stablehlo.select %16, %11#0, %15 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc19)
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc19)
    %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32> loc(#loc19)
    %20 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc19)
    %21 = stablehlo.select %20, %11#1, %19 : tensor<4xi1>, tensor<4xf32> loc(#loc19)
    return %17, %21 : tensor<4x4xcomplex<f32>>, tensor<4xf32> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xcomplex<f32>> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit"(#loc7))) -> (tensor<4x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc20)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc21)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc21)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc20)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc22)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc23)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc24)
    return %6 : tensor<4x4xcomplex<f32>> loc(#loc18)
  } loc(#loc18)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":266:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":266:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:25)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:15)
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:14)
#loc8 = loc("third_party/py/jax/tests/export_back_compat_test.py":274:11)
#loc9 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc10 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/transpose"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/real"(#loc4))
#loc13 = loc("jit(<lambda>)/jit(main)/imag"(#loc4))
#loc14 = loc("jit(<lambda>)/jit(main)/neg"(#loc4))
#loc15 = loc("jit(<lambda>)/jit(main)/complex"(#loc4))
#loc16 = loc("jit(<lambda>)/jit(main)/add"(#loc5))
#loc17 = loc("jit(<lambda>)/jit(main)/div"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/eigh"(#loc8))
#loc20 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota"(#loc7))
#loc21 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc7))
#loc22 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc7))
#loc23 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim"(#loc7))
#loc24 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc7))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x015\x05\x01\x05%\x01\x03\x0b\x03#\x0f\x13\x17\x1b\x1f#'+/37;?CGKO\x03*\x02\xc5=\x01g\x0f\x07\x0b\x17\x0f\x17\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0f\x0b\x17\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0b\x17\x13\x0b\x0b\x17\x03_\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x1f\x0f\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b/\x0f\x0b\x1f//O\x1b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x0bO/\x01\x05\x0b\x0f\x039\x17\x0f\x0f\x07\x07\x07\x13\x17\x0b\x17\x07\x07\x17\x0f\x13\x17\x17\x13\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02\x86\x07\x1dce\x1f\x05)\x17\x05J\x047\x1d!\x07\x17\x052\x043\x11\x03\x05\x1d#\x07\x1d%\x07\x1d[]\x03\x07\x17\x19\x1b\r\x1d\r\x05+\x11\x01\x00\x05-\x05/\x051\x053\x055\x057\x1d)\x07\x059\x1d-\x07\x05;\x1d1\x07\x05=\x1d57\x05?\x17\x05*\x045\x1d;=\x05A\x17\x05*\x04\x1d\x1dAC\x05C\x17\x052\x04E\x1dG\x0b\x05E\x1dK\x0b\x05G\x1dO\x0b\x05I\x1dS\x0b\x05K\x1dWY\x05M\x17\x052\x04\x1f\x05O\x17\x052\x04\x1d\x03\x03a\xa1\x05Q\x05S\x17\x05J\x04\x17\x1f'\x01\x1dU\x1dW\x03\x01\x1dY\x03\x03\x8d\x1d[\x1f\t\t\x00\x00\x00\x00\x13\x0b\x01\t\x07\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00##\x03\x05\x81\x85\r\x05o\x83ik\x1d]\r\x05o\x87ik\x1d_\x1da\x1dc\r\x03ik#%\x1de\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x05\x07\x05\x1f\x1f\t\x00\x00\xc0\x7f\x1f\x07\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x07\x11\x00\x00\x00@\x00\x00\x00\x00\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\xa3\xa5\xa7\xa9\x1dg\x13+\x00\x1di\x05\x03\x0b\x03\x1dk\x1dm\x05\x01\x03\x03{\x03\x03\xb7\x15\x03\x01\x01\x01\x03\x07{\xbb\xbd\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f1\x01\x07\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f;\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x15)\x01\x15)\x01\x1b\x1d\x01\t)\x03\x11\x0f)\x05\x11\x11\x1b\x03\x0f)\x05\x11\x11\x0f\x13\x1b)\x05\x11\x11\r)\x01\x0f)\x03\t\x0b\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03A\x15!)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03\x11\r)\x03\x05\x0b\x04\xee\x04\x05\x01Q\x03\x15\x01\x07\x04\xc6\x04\x03\x01\t\x0bP\x03\x03\x07\x04F\x03\x039m\x05B\x03\x05\x03\x1f\x05B\x03\x07\x03\x07\x05B\x03\t\x03\t\x05B\x03\x0b\x03\x07\x07B3\r\x03)\x13\x069\x03\x05\x03\t\x15F?\x0f\x03\x05\x03\x0b\x17\x06E\x03\x17\x03\r\x19\x06I\x03\x17\x03\r\x1b\x06M\x03\x17\x03\x11\x1d\x06Q\x03\x05\x05\x0f\x13\r\x06U\x03\x05\x05\x0b\x15\x03F\x13\x11\x03\x05\x03\x07\x1f\x06\x13\x03\x05\x05\x17\x19!F\t\x13\x03\x05\x03\x1b#G\x01_\x15\x07\x05\x11\t\x03\x1d\x03F\x01\x11\x03\t\x03\x05\x0fF\x01\x17\x033\x05#%\x03F\x01\x11\x035\x03'\x03F\x01\x11\x03\x05\x03\x03\x03F\x01\x19\x03\x1d\x03)\t\x06\x01\x03\x05\x07-\x1f+\x03F\x01\x11\x037\x03'\x03F\x01\x11\x03\x11\x03\x01\x03F\x01\x1b\x039\x031\t\x06\x01\x03\x11\x075!3\x11\x04\x03\x05/7\x0bP\t\x1d\x07\x04\x9d\x03\x15+\x03\x0b\t\x00\x05B\x03\x1f\x03\x07\x05B\x03\t\x03\t\x07B\x0f\r\x03\x13\x03F\x11\x11\x03\x13\x03\x05\r\x06\x11\x03\x13\x05\x07\t\x07B\x0f!\x03\x13\x0fF'#\x03\x1d\x05\x0b\r\x03F+\x11\x03\x05\x03\x03\t\x06/\x03\x05\x07\x0f\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00r\x10o'\x03\r\x15\x11\x0f\x0b\t\t\x0b!\x11#;)99A9;;EA;WgKMO;\x1b%)9i\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit\x00jit(<lambda>)/jit(main)/jit(tril)/iota\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota\x00jit(<lambda>)/jit(main)/reshape\x00jit(<lambda>)/jit(main)/transpose\x00jit(<lambda>)/jit(main)/real\x00jit(<lambda>)/jit(main)/imag\x00jit(<lambda>)/jit(main)/neg\x00jit(<lambda>)/jit(main)/complex\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/eigh\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00\x00cusolver_syevd_ffi\x00\x08o%\x05?\x01\x0bm}\x7f\x89\x8b\x03\x99\x03\x9b\x03u\x03\x9d\x03w\x03\x9f\x03g\x03s\x11\xab\xad\xafm\xb1\xb3\xb5\xb9\x05y\xbf\x03\xc1\x03\xc3\x0bq\x8fqs\x91\x03\x93\x03\x95\x05y\x97",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_09_30["c128"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_syevd_ffi'],
    serialized_date=datetime.date(2024, 9, 30),
    inputs=(),
    expected_outputs=(array([[ 0.7941185704969035 +0.j,  0.3696443397434604 +0.j,
         0.4041819665640972 +0.j, -0.2633926650306618 +0.j],
       [ 0.3696443397434601 +0.j, -0.7941185704969035 +0.j,
        -0.2633926650306616 +0.j, -0.4041819665640975 +0.j],
       [-0.05482989100998286+0.j,  0.4793041217634256 +0.j,
        -0.6857605696309689 +0.j, -0.5449712680975332 +0.j],
       [-0.47930412176342574+0.j, -0.05482989100998264+0.j,
         0.5449712680975333 +0.j, -0.6857605696309688 +0.j]]), array([-3.7082869338697044e+00,  3.5411017930205070e-16,
        6.5803628062392796e-16,  3.3708286933869694e+01])),
    mlir_module_text=r"""
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":274:27)
#loc18 = loc("jit(<lambda>)/jit(main)/pjit"(#loc7))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x4xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %cst_1 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<16xcomplex<f64>> loc(#loc9)
    %1 = stablehlo.reshape %0 : (tensor<16xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc10)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc11)
    %3 = stablehlo.real %2 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc12)
    %4 = stablehlo.imag %2 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc13)
    %5 = stablehlo.negate %4 : tensor<4x4xf64> loc(#loc14)
    %6 = stablehlo.complex %3, %5 : tensor<4x4xcomplex<f64>> loc(#loc15)
    %7 = stablehlo.add %1, %6 : tensor<4x4xcomplex<f64>> loc(#loc16)
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc17)
    %9 = stablehlo.divide %7, %8 : tensor<4x4xcomplex<f64>> loc(#loc17)
    %10 = call @tril(%9) : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc18)
    %11:3 = stablehlo.custom_call @cusolver_syevd_ffi(%10) {mhlo.backend_config = {algorithm = 0 : ui8, lower = true}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xcomplex<f64>>) -> (tensor<4x4xcomplex<f64>>, tensor<4xf64>, tensor<i32>) loc(#loc19)
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc19)
    %13 = stablehlo.compare  EQ, %11#2, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc19)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc19)
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc19)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc19)
    %17 = stablehlo.select %16, %11#0, %15 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc19)
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc19)
    %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64> loc(#loc19)
    %20 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi1>) -> tensor<4xi1> loc(#loc19)
    %21 = stablehlo.select %20, %11#1, %19 : tensor<4xi1>, tensor<4xf64> loc(#loc19)
    return %17, %21 : tensor<4x4xcomplex<f64>>, tensor<4xf64> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xcomplex<f64>> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit"(#loc7))) -> (tensor<4x4xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc20)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc21)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc21)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc20)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc22)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc23)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc24)
    return %6 : tensor<4x4xcomplex<f64>> loc(#loc18)
  } loc(#loc18)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":266:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":266:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:25)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:15)
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":268:14)
#loc8 = loc("third_party/py/jax/tests/export_back_compat_test.py":274:11)
#loc9 = loc("jit(<lambda>)/jit(main)/iota"(#loc1))
#loc10 = loc("jit(<lambda>)/jit(main)/reshape"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/transpose"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/real"(#loc4))
#loc13 = loc("jit(<lambda>)/jit(main)/imag"(#loc4))
#loc14 = loc("jit(<lambda>)/jit(main)/neg"(#loc4))
#loc15 = loc("jit(<lambda>)/jit(main)/complex"(#loc4))
#loc16 = loc("jit(<lambda>)/jit(main)/add"(#loc5))
#loc17 = loc("jit(<lambda>)/jit(main)/div"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/eigh"(#loc8))
#loc20 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota"(#loc7))
#loc21 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc7))
#loc22 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc7))
#loc23 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim"(#loc7))
#loc24 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc7))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.5.0\x00\x015\x05\x01\x05%\x01\x03\x0b\x03#\x0f\x13\x17\x1b\x1f#'+/37;?CGKO\x03*\x02\xc5=\x01g\x0f\x07\x0b\x17\x0f\x17\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0f\x0b\x17\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0b\x17\x13\x0b\x0b\x17\x03_\x0f\x0b\x0b\x0b\x0b\x0f\x0b\x1f\x0f\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0bO\x0f\x0b/OOO\x1b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x17/\x0f\x0bO/\x01\x05\x0b\x0f\x039\x17\x0f\x0f\x07\x07\x07\x13\x17\x0b\x17\x07\x07\x17\x0f\x13\x17\x17\x13\x13\x07\x13\x13\x13\x0f\x17\x13\x13\x13\x02\xf6\x07\x1dce\x1f\x05)\x17\x05J\x047\x1d!\x07\x17\x052\x043\x11\x03\x05\x1d#\x07\x1d%\x07\x1d[]\x03\x07\x17\x19\x1b\r\x1d\r\x05+\x11\x01\x00\x05-\x05/\x051\x053\x055\x057\x1d)\x07\x059\x1d-\x07\x05;\x1d1\x07\x05=\x1d57\x05?\x17\x05*\x045\x1d;=\x05A\x17\x05*\x04\x1d\x1dAC\x05C\x17\x052\x04E\x1dG\x0b\x05E\x1dK\x0b\x05G\x1dO\x0b\x05I\x1dS\x0b\x05K\x1dWY\x05M\x17\x052\x04\x1f\x05O\x17\x052\x04\x1d\x03\x03a\xa1\x05Q\x05S\x17\x05J\x04\x17\x1f'\x01\x1dU\x1dW\x03\x01\x1dY\x03\x03\x8d\x1d[\x1f\t\t\x00\x00\x00\x00\x13\x0b\x01\t\x07\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00##\x03\x05\x81\x85\r\x05o\x83ik\x1d]\r\x05o\x87ik\x1d_\x1da\x1dc\r\x03ik#%\x1de\x1f\x07!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x13\x0b\x05\x07\x05\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x07!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x07!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x05\xa3\xa5\xa7\xa9\x1dg\x13+\x00\x1di\x05\x03\x0b\x03\x1dk\x1dm\x05\x01\x03\x03{\x03\x03\xb7\x15\x03\x01\x01\x01\x03\x07{\xbb\xbd\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f1\x01\x07\x01\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f;\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x15)\x01\x15)\x01\x1b\x1d\x01\x0b)\x03\x11\x0f)\x05\x11\x11\x1b\x03\x0f)\x05\x11\x11\x0f\x13\x1b)\x05\x11\x11\r)\x01\x0f)\x03\t\x0b\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03A\x15!)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03\x11\r)\x03\x05\x0b\x04\xee\x04\x05\x01Q\x03\x15\x01\x07\x04\xc6\x04\x03\x01\t\x0bP\x03\x03\x07\x04F\x03\x039m\x05B\x03\x05\x03\x1f\x05B\x03\x07\x03\x07\x05B\x03\t\x03\t\x05B\x03\x0b\x03\x07\x07B3\r\x03)\x13\x069\x03\x05\x03\t\x15F?\x0f\x03\x05\x03\x0b\x17\x06E\x03\x17\x03\r\x19\x06I\x03\x17\x03\r\x1b\x06M\x03\x17\x03\x11\x1d\x06Q\x03\x05\x05\x0f\x13\r\x06U\x03\x05\x05\x0b\x15\x03F\x13\x11\x03\x05\x03\x07\x1f\x06\x13\x03\x05\x05\x17\x19!F\t\x13\x03\x05\x03\x1b#G\x01_\x15\x07\x05\x11\t\x03\x1d\x03F\x01\x11\x03\t\x03\x05\x0fF\x01\x17\x033\x05#%\x03F\x01\x11\x035\x03'\x03F\x01\x11\x03\x05\x03\x03\x03F\x01\x19\x03\x1d\x03)\t\x06\x01\x03\x05\x07-\x1f+\x03F\x01\x11\x037\x03'\x03F\x01\x11\x03\x11\x03\x01\x03F\x01\x1b\x039\x031\t\x06\x01\x03\x11\x075!3\x11\x04\x03\x05/7\x0bP\t\x1d\x07\x04\x9d\x03\x15+\x03\x0b\t\x00\x05B\x03\x1f\x03\x07\x05B\x03\t\x03\t\x07B\x0f\r\x03\x13\x03F\x11\x11\x03\x13\x03\x05\r\x06\x11\x03\x13\x05\x07\t\x07B\x0f!\x03\x13\x0fF'#\x03\x1d\x05\x0b\r\x03F+\x11\x03\x05\x03\x03\t\x06/\x03\x05\x07\x0f\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00r\x10o'\x03\r\x15\x11\x0f\x0b\t\t\x0b!\x11#;)99A9;;EA;WgKMO;\x1b%)9i\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit\x00jit(<lambda>)/jit(main)/jit(tril)/iota\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota\x00jit(<lambda>)/jit(main)/reshape\x00jit(<lambda>)/jit(main)/transpose\x00jit(<lambda>)/jit(main)/real\x00jit(<lambda>)/jit(main)/imag\x00jit(<lambda>)/jit(main)/neg\x00jit(<lambda>)/jit(main)/complex\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00mhlo.backend_config\x00jit(<lambda>)/jit(main)/eigh\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00algorithm\x00lower\x00\x00cusolver_syevd_ffi\x00\x08o%\x05?\x01\x0bm}\x7f\x89\x8b\x03\x99\x03\x9b\x03u\x03\x9d\x03w\x03\x9f\x03g\x03s\x11\xab\xad\xafm\xb1\xb3\xb5\xb9\x05y\xbf\x03\xc1\x03\xc3\x0bq\x8fqs\x91\x03\x93\x03\x95\x05y\x97",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
