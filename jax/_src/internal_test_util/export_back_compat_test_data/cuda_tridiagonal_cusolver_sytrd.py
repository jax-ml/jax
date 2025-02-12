# Copyright 2025 The JAX Authors.
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

data_2025_01_09 = {}

data_2025_01_09["f32"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_sytrd_ffi'],
    serialized_date=datetime.date(2025, 1, 9),
    inputs=(array([[[-1.2178137 , -1.4850428 ,  2.5807054 , -2.7725415 ],
        [ 0.7619374 , -4.455563  , -2.7315347 ,  0.73871464],
        [ 1.1837728 , -2.3819368 ,  0.77078664,  0.5918046 ],
        [-4.019883  ,  1.7254075 ,  0.93185544, -0.20608105]],

       [[-1.7672386 , -4.4935865 ,  7.287687  ,  1.945105  ],
        [-0.3699937 ,  3.9466763 , -2.143978  , -0.12787771],
        [-0.02148095,  3.0340018 ,  6.5158176 , -1.6688819 ],
        [ 3.5351803 ,  2.8901145 ,  3.4657938 ,  0.06964709]]],
      dtype=float32),),
    expected_outputs=(array([[[-1.2178137e+00, -1.4850428e+00,  2.5807054e+00, -2.7725415e+00],
        [-4.2592635e+00, -1.5749537e+00, -2.7315347e+00,  7.3871464e-01],
        [ 2.3575491e-01,  2.9705398e+00, -4.0452647e+00,  5.9180462e-01],
        [-8.0058199e-01, -9.9736643e-01, -1.4680552e-01,  1.7293599e+00]],

       [[-1.7672386e+00, -4.4935865e+00,  7.2876868e+00,  1.9451050e+00],
        [ 3.5545545e+00, -5.2433920e-01, -2.1439781e+00, -1.2787771e-01],
        [ 5.4734843e-03, -3.9149244e+00,  9.0688534e+00, -1.6688819e+00],
        [-9.0078670e-01,  3.4653693e-01,  1.6330767e-01,  1.9876275e+00]]],
      dtype=float32), array([[-1.2178137, -1.5749537, -4.0452647,  1.7293599],
       [-1.7672386, -0.5243392,  9.068853 ,  1.9876275]], dtype=float32), array([[-4.2592635 ,  2.9705398 , -0.14680552],
       [ 3.5545545 , -3.9149244 ,  0.16330767]], dtype=float32), array([[1.1788895, 1.002637 , 0.       ],
       [1.10409  , 1.7855742, 0.       ]], dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf32> loc("x")) -> (tensor<2x4x4xf32> {jax.result_info = "[0]"}, tensor<2x4xf32> {jax.result_info = "[1]"}, tensor<2x3xf32> {jax.result_info = "[2]"}, tensor<2x3xf32> {jax.result_info = "[3]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x4x4xi1> loc(#loc5)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc5)
    %5 = stablehlo.select %3, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %7 = stablehlo.compare  EQ, %0#4, %6,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x4xi1> loc(#loc5)
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc5)
    %10 = stablehlo.select %8, %0#1, %9 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %12 = stablehlo.compare  EQ, %0#4, %11,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<2xi1>) -> tensor<2x3xi1> loc(#loc5)
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc5)
    %15 = stablehlo.select %13, %0#2, %14 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc6)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %17 = stablehlo.compare  EQ, %0#4, %16,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %18 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<2xi1>) -> tensor<2x3xi1> loc(#loc5)
    %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc5)
    %20 = stablehlo.select %18, %0#3, %19 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc6)
    return %5, %10, %15, %20 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":831:13)
#loc3 = loc("jit(func)/jit(main)/tridiagonal"(#loc2))
#loc4 = loc("jit(func)/jit(main)/eq"(#loc2))
#loc5 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc2))
#loc6 = loc("jit(func)/jit(main)/select_n"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.3\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xb7u/\x01-\x0f\x0f\x07\x17\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x03I\x0f\x0b\x0b\x0b/Oo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x01\x05\x0b\x0f\x03+\x17\x1b\x13\x07\x17\x13\x07\x07\x0f\x0f\x07\x07\x17#\x13\x13\x13\x13\x1b\x13\x17\x02\xe6\x04\x1d'\x07\x1d)\x07\x1f\x17%\xfe\x0c\x1b\x1d+\x07\x11\x03\x05\x03\x07\x0f\x11\x13\x0b\x15\x0b\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x1b\x05\x05\x1d\x03\x03\x1f[\x05\x1f\x1d#\x07\x05!\x05#\x05%\x05'\x05)\x1f'\x01\x1d+\t\x07\x07\x01\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f!1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03=\r\x01#\x1f\x03\tCGKO\r\x03/E\x1d-\r\x03/I\x1d/\r\x03/M\x1d1\r\x03/Q\x1d3\x1d5\x1d7\x1f\x15\t\x00\x00\xc0\x7f\x1f\x17\t\x00\x00\x00\x00\r\x03]_\x1d9\x05\x03\x0b\x03\x1d;\x1d=\x03\x01\x05\x01\x03\x039\x03\x03o\x15\x03\x01\x01\x01\x03\x0b9777s\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\t\r\x0b)\x07\t\x11\x11\x0b)\x03\t\x19\t)\x05\t\x11\x0b)\x03\t\x11\x01\x13)\x01\x0b)\x01\x19\x1b\x1d)\x05\t\r\x11\x11\x03\x07\t\x07\r\x05\x05)\x03\r\x13)\x03\t\x13)\x03\x05\x13)\x03\x01\x1b)\x07\t\x11\x11\x11)\x03\x05\x1b)\x05\t\x11\x11\x04\x96\x03\x05\x01Q\x05\r\x01\x07\x04n\x03\x03\x01\x05\x0bP\x05\x03\x07\x04B\x03\x039c\x03\x0f\x19\x00\tB\x05\x05\x03\x15\tB\x05\x07\x03\x17\rG!\x1d\t\x0b\x07\r\x05\x05\t\x03\x01\x03F\x01\x0b\x03\t\x03\x05\x05F\x01\r\x03\x0f\x05\x0f\x11\x03F\x03\x0f\x03)\x03\x13\x03F\x03\x0b\x03\x07\x03\x03\x07\x06\t\x03\x07\x07\x15\x07\x17\x03F\x01\x0b\x03\t\x03\x05\x05F\x01\r\x03\x0f\x05\x0f\x1b\x03F\x03\x0f\x03-\x03\x1d\x03F\x03\x0b\x03\r\x03\x03\x07\x06\t\x03\r\x07\x1f\t!\x03F\x01\x0b\x03\t\x03\x05\x05F\x01\r\x03\x0f\x05\x0f%\x03F\x03\x0f\x03\x1d\x03'\x03F\x03\x0b\x03\x05\x03\x03\x07\x06\t\x03\x05\x07)\x0b+\x03F\x01\x0b\x03\t\x03\x05\x05F\x01\r\x03\x0f\x05\x0f/\x03F\x03\x0f\x03\x1d\x031\x03F\x03\x0b\x03\x05\x03\x03\x07\x06\t\x03\x05\x073\r5\x0f\x04\x05\t\x19#-7\x06\x03\x01\x05\x01\x00z\x07?'\x03\r\x0f\x0b\t\t\t\t!;K/iA)\x05\x13%)9\x15\x1f\x11\x19\x15\x17)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00compare_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00mhlo.backend_config\x00jit(func)/jit(main)/tridiagonal\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00jit(func)/jit(main)/select_n\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00lower\x00\x00cusolver_sytrd_ffi\x00\x08=\x11\x05/\x01\x0b;?ASU\x03W\x03Y\x11acegikmq\x03-\x0513\x035",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2025_01_09["f64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_sytrd_ffi'],
    serialized_date=datetime.date(2025, 1, 9),
    inputs=(array([[[-3.1253059079430567e+00,  5.7973737786717416e+00,
          1.2965480927655553e+00,  6.4817268627909046e+00],
        [-1.1936619933620258e+00,  1.2510994308173569e+00,
          1.5304146964685636e-01, -3.9365359484492188e+00],
        [-4.7758906372514920e+00, -8.3938964119717618e-01,
          9.7688102177825908e-01, -1.0925284023457876e-01],
        [ 1.8645139381221081e+00, -1.2926082118180000e+00,
          2.1621899240570671e-03, -3.9399456873538158e-02]],

       [[ 1.8036782176339347e-01, -2.7779317198395299e+00,
          1.3453106407991833e+00,  5.9722843958961107e+00],
        [ 2.8866256663925954e-01, -1.4318659824435502e+00,
          6.4080201583537226e+00,  1.1943604455356474e+00],
        [ 2.6497910527857116e-01, -1.4861093982154459e+00,
          5.4248701360547784e+00, -1.5973136783857038e+00],
        [ 6.3417141892397311e+00, -3.7002383133493355e+00,
         -1.8965912710753372e+00, -2.4871387323298411e+00]]]),),
    expected_outputs=(array([[[-3.1253059079430567 ,  5.797373778671742  ,
          1.2965480927655553 ,  6.481726862790905  ],
        [ 5.264064262415029  ,  0.7243571624190226 ,
          0.15304146964685636, -3.936535948449219  ],
        [ 0.7395622620235721 ,  0.18927242291294558,
          1.2785065803968352 , -0.10925284023457876],
        [-0.28872607234692266, -0.20307594403518822,
         -1.58216916428174   ,  0.18571725290621857]],

       [[ 0.18036782176339347, -2.77793171983953   ,
          1.3453106407991833 ,  5.972284395896111  ],
        [-6.353808217251881  , -2.9702952281045247 ,
          6.408020158353723  ,  1.1943604455356474 ],
        [ 0.03989164783700368, -4.028630349114961  ,
         -0.9432585642079134 , -1.5973136783857038 ],
        [ 0.9547221802795454 , -0.6834393717622312 ,
         -1.599653328439445  ,  5.419419213593815  ]]]), array([[-3.1253059079430567 ,  0.7243571624190226 ,  1.2785065803968352 ,
         0.18571725290621857],
       [ 0.18036782176339347, -2.9702952281045247 , -0.9432585642079134 ,
         5.419419213593815  ]]), array([[ 5.264064262415029  ,  0.18927242291294558, -1.58216916428174   ],
       [-6.353808217251881  , -4.028630349114961  , -1.599653328439445  ]]), array([[1.2267567289944903, 1.9207870511685834, 0.                ],
       [1.0454314258109778, 1.3632434630444663, 0.                ]])),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf64> loc("x")) -> (tensor<2x4x4xf64> {jax.result_info = "[0]"}, tensor<2x4xf64> {jax.result_info = "[1]"}, tensor<2x3xf64> {jax.result_info = "[2]"}, tensor<2x3xf64> {jax.result_info = "[3]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xf64>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x4x4xi1> loc(#loc5)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc5)
    %5 = stablehlo.select %3, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %7 = stablehlo.compare  EQ, %0#4, %6,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x4xi1> loc(#loc5)
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc5)
    %10 = stablehlo.select %8, %0#1, %9 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %12 = stablehlo.compare  EQ, %0#4, %11,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<2xi1>) -> tensor<2x3xi1> loc(#loc5)
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc5)
    %15 = stablehlo.select %13, %0#2, %14 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc6)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %17 = stablehlo.compare  EQ, %0#4, %16,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %18 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<2xi1>) -> tensor<2x3xi1> loc(#loc5)
    %19 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc5)
    %20 = stablehlo.select %18, %0#3, %19 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc6)
    return %5, %10, %15, %20 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":831:13)
#loc3 = loc("jit(func)/jit(main)/tridiagonal"(#loc2))
#loc4 = loc("jit(func)/jit(main)/eq"(#loc2))
#loc5 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc2))
#loc6 = loc("jit(func)/jit(main)/select_n"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.3\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xb7u/\x01-\x0f\x0f\x07\x17\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x03I\x0f\x0b\x0b\x0b/Oo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x01\x05\x0b\x0f\x03+\x17\x1b\x13\x07\x17\x13\x07\x07\x0f\x0f\x07\x07\x17#\x13\x13\x13\x13\x1b\x13\x17\x02\xf6\x04\x1d'\x07\x1d)\x07\x1f\x17%\xfe\x0c\x1b\x1d+\x07\x11\x03\x05\x03\x07\x0f\x11\x13\x0b\x15\x0b\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x1b\x05\x05\x1d\x03\x03\x1f[\x05\x1f\x1d#\x07\x05!\x05#\x05%\x05'\x05)\x1f'\x01\x1d+\t\x07\x07\x01\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f!1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03=\r\x01#\x1f\x03\tCGKO\r\x03/E\x1d-\r\x03/I\x1d/\r\x03/M\x1d1\r\x03/Q\x1d3\x1d5\x1d7\x1f\x15\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x17\t\x00\x00\x00\x00\r\x03]_\x1d9\x05\x03\x0b\x03\x1d;\x1d=\x03\x01\x05\x01\x03\x039\x03\x03o\x15\x03\x01\x01\x01\x03\x0b9777s\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\t\r\x0b)\x07\t\x11\x11\x0b)\x03\t\x19\x0b)\x05\t\x11\x0b)\x03\t\x11\x01\x13)\x01\x0b)\x01\x19\x1b\x1d)\x05\t\r\x11\x11\x03\x07\t\x07\r\x05\x05)\x03\r\x13)\x03\t\x13)\x03\x05\x13)\x03\x01\x1b)\x07\t\x11\x11\x11)\x03\x05\x1b)\x05\t\x11\x11\x04\x96\x03\x05\x01Q\x05\r\x01\x07\x04n\x03\x03\x01\x05\x0bP\x05\x03\x07\x04B\x03\x039c\x03\x0f\x19\x00\tB\x05\x05\x03\x15\tB\x05\x07\x03\x17\rG!\x1d\t\x0b\x07\r\x05\x05\t\x03\x01\x03F\x01\x0b\x03\t\x03\x05\x05F\x01\r\x03\x0f\x05\x0f\x11\x03F\x03\x0f\x03)\x03\x13\x03F\x03\x0b\x03\x07\x03\x03\x07\x06\t\x03\x07\x07\x15\x07\x17\x03F\x01\x0b\x03\t\x03\x05\x05F\x01\r\x03\x0f\x05\x0f\x1b\x03F\x03\x0f\x03-\x03\x1d\x03F\x03\x0b\x03\r\x03\x03\x07\x06\t\x03\r\x07\x1f\t!\x03F\x01\x0b\x03\t\x03\x05\x05F\x01\r\x03\x0f\x05\x0f%\x03F\x03\x0f\x03\x1d\x03'\x03F\x03\x0b\x03\x05\x03\x03\x07\x06\t\x03\x05\x07)\x0b+\x03F\x01\x0b\x03\t\x03\x05\x05F\x01\r\x03\x0f\x05\x0f/\x03F\x03\x0f\x03\x1d\x031\x03F\x03\x0b\x03\x05\x03\x03\x07\x06\t\x03\x05\x073\r5\x0f\x04\x05\t\x19#-7\x06\x03\x01\x05\x01\x00z\x07?'\x03\r\x0f\x0b\t\t\t\t!;K/iA)\x05\x13%)9\x15\x1f\x11\x19\x15\x17)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00compare_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00mhlo.backend_config\x00jit(func)/jit(main)/tridiagonal\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00jit(func)/jit(main)/select_n\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00lower\x00\x00cusolver_sytrd_ffi\x00\x08=\x11\x05/\x01\x0b;?ASU\x03W\x03Y\x11acegikmq\x03-\x0513\x035",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2025_01_09["c64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_sytrd_ffi'],
    serialized_date=datetime.date(2025, 1, 9),
    inputs=(array([[[ 3.1812036 -0.48341316j ,  0.11142776-6.424985j   ,
          1.6697654 -0.7869761j  ,  2.3984313 +0.033103157j],
        [-4.391181  +3.711728j   ,  1.062603  -1.5042974j  ,
          3.2630582 -3.0468545j  , -1.1403834 +4.834375j   ],
        [-3.3597422 +6.9571705j  ,  3.0927958 -3.3640988j  ,
         -0.55644   +0.06318861j , -3.0856206 +2.0522592j  ],
        [-0.12403099-0.7970628j  , -0.56212497-2.0354905j  ,
         -1.429683  +0.91537094j ,  3.375416  +0.4935444j  ]],

       [[-1.8590846 -0.39196187j ,  0.38138282+3.0784519j  ,
          1.966164  -0.5697291j  ,  0.21431345+0.66547275j ],
        [-3.575784  -2.2677388j  , -3.523877  -0.12984382j ,
          1.5619129 +4.8201146j  , -1.321369  -1.2919399j  ],
        [-1.3055397 +5.412672j   , -2.538718  -1.8749187j  ,
          1.4178271 +0.8454019j  ,  1.6860713 +3.5848062j  ],
        [-3.2620752 -5.5018277j  ,  1.5509791 +2.9875515j  ,
         -1.6966074 +1.4490732j  , -2.3220365 +0.102485105j]]],
      dtype=complex64),),
    expected_outputs=(array([[[ 3.1812036e+00+0.0000000e+00j,  1.1142776e-01-6.4249849e+00j,
          1.6697654e+00-7.8697610e-01j,  2.3984313e+00+3.3103157e-02j],
        [ 9.6643763e+00+0.0000000e+00j,  4.1165028e+00+0.0000000e+00j,
          3.2630582e+00-3.0468545e+00j, -1.1403834e+00+4.8343749e+00j],
        [ 3.4564066e-01-4.0370134e-01j, -2.6924424e+00+0.0000000e+00j,
         -3.6695964e+00+0.0000000e+00j, -3.0856206e+00+2.0522592e+00j],
        [-5.7498864e-03+5.5189621e-02j,  1.9120899e-01-4.7203872e-02j,
          2.5072362e+00+0.0000000e+00j,  3.4346726e+00+2.1266517e-09j]],

       [[-1.8590846e+00+0.0000000e+00j,  3.8138282e-01+3.0784519e+00j,
          1.9661640e+00-5.6972909e-01j,  2.1431345e-01+6.6547275e-01j],
        [ 9.4784794e+00+0.0000000e+00j,  3.5050194e+00+0.0000000e+00j,
          1.5619129e+00+4.8201146e+00j, -1.3213691e+00-1.2919399e+00j],
        [ 2.7161252e-02-4.1934705e-01j, -2.4255285e+00+0.0000000e+00j,
         -2.9377868e+00+0.0000000e+00j,  1.6860713e+00+3.5848062e+00j],
        [ 3.1363532e-01+3.6697471e-01j,  5.7014298e-01-8.8611171e-03j,
         -2.7132864e+00+0.0000000e+00j, -4.9953189e+00-4.8905555e-09j]]],
      dtype=complex64), array([[ 3.1812036,  4.116503 , -3.6695964,  3.4346726],
       [-1.8590846,  3.5050194, -2.9377868, -4.995319 ]], dtype=float32), array([[ 9.664376 , -2.6924424,  2.5072362],
       [ 9.478479 , -2.4255285, -2.7132864]], dtype=float32), array([[1.4543678-0.3840629j , 1.8027349-0.47009134j,
        1.9218173+0.38762447j],
       [1.3772529+0.23925133j, 1.2086039+0.6028182j ,
        1.9929025-0.11893065j]], dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f32>> loc("x")) -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "[0]"}, tensor<2x4xf32> {jax.result_info = "[1]"}, tensor<2x3xf32> {jax.result_info = "[2]"}, tensor<2x3xcomplex<f32>> {jax.result_info = "[3]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xcomplex<f32>>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x4x4xi1> loc(#loc5)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc5)
    %5 = stablehlo.select %3, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %7 = stablehlo.compare  EQ, %0#4, %6,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x4xi1> loc(#loc5)
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc5)
    %10 = stablehlo.select %8, %0#1, %9 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %12 = stablehlo.compare  EQ, %0#4, %11,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<2xi1>) -> tensor<2x3xi1> loc(#loc5)
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc5)
    %15 = stablehlo.select %13, %0#2, %14 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc6)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %17 = stablehlo.compare  EQ, %0#4, %16,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %18 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<2xi1>) -> tensor<2x3xi1> loc(#loc5)
    %19 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<2x3xcomplex<f32>> loc(#loc5)
    %20 = stablehlo.select %18, %0#3, %19 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>> loc(#loc6)
    return %5, %10, %15, %20 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":831:13)
#loc3 = loc("jit(func)/jit(main)/tridiagonal"(#loc2))
#loc4 = loc("jit(func)/jit(main)/eq"(#loc2))
#loc5 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc2))
#loc6 = loc("jit(func)/jit(main)/select_n"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.3\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xbfw5\x01-\x0f\x0f\x07\x17\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x03K\x0f\x0b\x0b\x0b/Oo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f/\x1f\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x01\x05\x0b\x0f\x031\x1b\x13\x07\x17\x17\x17\x13\x07\x0b\x07\x0f\x0f\x0f\x07\x07\x17#\x13\x13\x13\x13\x1b\x13\x17\x02:\x05\x1d'\x07\x1d)\x07\x1f\x17%\xfe\x0c\x1b\x1d+\x07\x11\x03\x05\x03\x07\x0f\x11\x13\x0b\x15\x0b\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x1b\x05\x05\x1d\x03\x03\x1f]\x05\x1f\x1d#\x07\x05!\x05#\x05%\x05'\x05)\x1f-\x01\x1d+\t\x07\x07\x01\x1f1\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03=\r\x01#%\x03\tCGKO\r\x03/E\x1d-\r\x03/I\x1d/\r\x03/M\x1d1\r\x03/Q\x1d3\x1d5\x1d7\x1f\x19\t\x00\x00\xc0\x7f\x1f\x1b\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x1d\t\x00\x00\x00\x00\r\x03_a\x1d9\x05\x03\x0b\x03\x1d;\x1d=\x03\x01\x05\x01\x03\x039\x03\x03q\x15\x03\x01\x01\x01\x03\x0b9777u\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x15)\x03\t\x1f\t)\x05\t\x11\t)\x05\t\r\t)\x05\t\r\x15)\x03\t\x13\x01\x03\t\x13)\x01\t)\x01\x15)\x01\x1f\x1b\x1d)\x05\t\r\x13\x11\x03\x05\t\x05\x0b\r\x0f)\x03\r\x17)\x03\t\x17)\x03\x05\x17)\x03\x01!)\x07\t\x11\x11\x13)\x03\x05!)\x05\t\x11\x13\x04\xae\x03\x05\x01Q\x05\r\x01\x07\x04\x86\x03\x03\x01\x05\x0bP\x05\x03\x07\x04Z\x03\x03;g\x03\x0b\x19\x00\tB\x05\x05\x03\x19\tB\x05\x07\x03\x1b\tB\x05\t\x03\x1d\rG!\x1d\x0b\x0b\x05\x0b\r\x0f\x07\x03\x01\x03F\x01\r\x03\x07\x03\x07\x05F\x01\x0f\x03\x11\x05\x11\x13\x03F\x03\x11\x03/\x03\x15\x03F\x03\r\x03\x05\x03\x05\x07\x06\t\x03\x05\x07\x17\t\x19\x03F\x01\r\x03\x07\x03\x07\x05F\x01\x0f\x03\x11\x05\x11\x1d\x03F\x03\x11\x033\x03\x1f\x03F\x03\r\x03\x0b\x03\x03\x07\x06\t\x03\x0b\x07!\x0b#\x03F\x01\r\x03\x07\x03\x07\x05F\x01\x0f\x03\x11\x05\x11'\x03F\x03\x11\x03#\x03)\x03F\x03\r\x03\r\x03\x03\x07\x06\t\x03\r\x07+\r-\x03F\x01\r\x03\x07\x03\x07\x05F\x01\x0f\x03\x11\x05\x111\x03F\x03\x11\x03#\x033\x03F\x03\r\x03\x0f\x03\x05\x07\x06\t\x03\x0f\x075\x0f7\x0f\x04\x05\t\x1b%/9\x06\x03\x01\x05\x01\x00z\x07?'\x03\r\x0f\x0b\t\t\t\t!;K/iA)\x05\x13%)9\x15\x1f\x11\x19\x15\x17)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00compare_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00mhlo.backend_config\x00jit(func)/jit(main)/tridiagonal\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00jit(func)/jit(main)/select_n\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00lower\x00\x00cusolver_sytrd_ffi\x00\x08A\x13\x05/\x01\x0b;?ASU\x03W\x03Y\x03[\x11cegikmos\x03-\x0513\x035",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2025_01_09["c128"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_sytrd_ffi'],
    serialized_date=datetime.date(2025, 1, 9),
    inputs=(array([[[-8.811619206101685  -6.346103055817515j  ,
         -2.0673625472346933 -0.27575832037940934j,
          0.17956803185065526+0.4305341577966682j ,
          1.1602180635956165 +0.9004656831972404j ],
        [ 0.04561867504808473-4.771560375289864j  ,
          4.778928872206952  +1.7033875050911593j ,
         -3.8122399421775315 -5.795050026499773j  ,
          0.5791695386369239 +0.9628332000212088j ],
        [-1.9993689229562661 -4.064915227679414j  ,
         -2.3979548160862456 +0.8988248576397159j ,
         -2.7272138752203796 +0.21666059983248281j,
          2.067134647794104  -0.14136984404543293j],
        [ 5.485533829522301  +1.1676501594179898j ,
         -3.009376295256183  +1.2776833329344761j ,
         -1.0223678823537847 -1.2013281507542595j ,
          0.08734463104455259-2.420057264345883j  ]],

       [[ 2.212505188041518  -3.2177849380091894j ,
         -1.0593623525670381 +5.37426459023754j   ,
         -3.1623822399052592 +0.8649520447907271j ,
         -1.7388123297543543 +3.2870543762142654j ],
        [ 2.793400596418774  -0.739715064363405j  ,
          8.739016966775436  +5.022683652842428j  ,
          0.7559458452282815 +4.110352398353875j  ,
          1.7261562225678564 -3.8393940384377196j ],
        [ 0.37042546632316864-0.1996809256297461j ,
         -1.3181774150972687 -0.32212366971341144j,
          1.3177595427368969 -1.9978296888748126j ,
         -0.6558210664615476 +0.08467474893379115j],
        [-1.7318040703079298 -4.599436443227647j  ,
         -0.20128653725409817-1.115567638745429j  ,
         -1.579624976405968  +7.379126520550004j  ,
          2.0570113545830235 +2.5303646286484853j ]]]),),
    expected_outputs=(array([[[-8.811619206101685   +0.0000000000000000e+00j,
         -2.0673625472346933  -2.7575832037940934e-01j,
          0.17956803185065526 +4.3053415779666820e-01j,
          1.1602180635956165  +9.0046568319724041e-01j],
        [-8.645540449646582   +0.0000000000000000e+00j,
          0.3845716078748591  +0.0000000000000000e+00j,
         -3.8122399421775315  -5.7950500264997729e+00j,
          0.5791695386369239  +9.6283320002120876e-01j],
        [ 0.02053989913865161 -4.5643024157336864e-01j,
          5.036492728413716   +0.0000000000000000e+00j,
          1.84687034311969    +1.1102230246251565e-16j,
          2.067134647794104   -1.4136984404543293e-01j],
        [ 0.4283052472392046  +3.6949438614565994e-01j,
          0.21688243005340535 +5.1643916348653485e-01j,
         -2.7797703440504513  +0.0000000000000000e+00j,
         -0.09238232296342479 -6.5179386557032127e-17j]],

       [[ 2.212505188041518   +0.0000000000000000e+00j,
         -1.0593623525670381  +5.3742645902375399e+00j,
         -3.1623822399052592  +8.6495204479072707e-01j,
         -1.7388123297543543  +3.2870543762142654e+00j],
        [-5.71675727138259    +0.0000000000000000e+00j,
          3.7004622018069355  +0.0000000000000000e+00j,
          0.7559458452282815  +4.1103523983538750e+00j,
          1.7261562225678564  -3.8393940384377196e+00j],
        [ 0.04522526729096931 -1.9532788545989693e-02j,
         -8.147585314007962   +0.0000000000000000e+00j,
          1.0622507332612754  +0.0000000000000000e+00j,
         -0.6558210664615476  +8.4674748933791150e-02j],
        [-0.155346841147821   -5.5396726066186741e-01j,
         -0.036849829076318265-2.2002105614858228e-01j,
         -0.6549744056215347  +0.0000000000000000e+00j,
          7.351074929027147   -9.3780430643058420e-18j]]]), array([[-8.811619206101685  ,  0.3845716078748591 ,  1.84687034311969   ,
        -0.09238232296342479],
       [ 2.212505188041518  ,  3.7004622018069355 ,  1.0622507332612754 ,
         7.351074929027147  ]]), array([[-8.645540449646582 ,  5.036492728413716 , -2.7797703440504513],
       [-5.71675727138259  , -8.147585314007962 , -0.6549744056215347]]), array([[1.0052765556200653-0.5519100168555592j ,
        1.2933847199153174+0.5442027059704965j ,
        1.8134445415378648+0.5816424828382576j ],
       [1.4886337592820684-0.12939417037458123j,
        1.7898082606090093-0.454423909708186j  ,
        1.8735743590190088-0.48669070184720786j]])),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f64>> loc("x")) -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "[0]"}, tensor<2x4xf64> {jax.result_info = "[1]"}, tensor<2x3xf64> {jax.result_info = "[2]"}, tensor<2x3xcomplex<f64>> {jax.result_info = "[3]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xcomplex<f64>>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x4x4xi1> loc(#loc5)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc5)
    %5 = stablehlo.select %3, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %7 = stablehlo.compare  EQ, %0#4, %6,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x4xi1> loc(#loc5)
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc5)
    %10 = stablehlo.select %8, %0#1, %9 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %12 = stablehlo.compare  EQ, %0#4, %11,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<2xi1>) -> tensor<2x3xi1> loc(#loc5)
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc5)
    %15 = stablehlo.select %13, %0#2, %14 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc6)
    %16 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %17 = stablehlo.compare  EQ, %0#4, %16,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %18 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<2xi1>) -> tensor<2x3xi1> loc(#loc5)
    %19 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<2x3xcomplex<f64>> loc(#loc5)
    %20 = stablehlo.select %18, %0#3, %19 : tensor<2x3xi1>, tensor<2x3xcomplex<f64>> loc(#loc6)
    return %5, %10, %15, %20 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":831:13)
#loc3 = loc("jit(func)/jit(main)/tridiagonal"(#loc2))
#loc4 = loc("jit(func)/jit(main)/eq"(#loc2))
#loc5 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc2))
#loc6 = loc("jit(func)/jit(main)/select_n"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.3\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xbfw5\x01-\x0f\x0f\x07\x17\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x03K\x0f\x0b\x0b\x0b/Oo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/O\x1f\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x01\x05\x0b\x0f\x031\x1b\x13\x07\x17\x17\x17\x13\x07\x0b\x07\x0f\x0f\x0f\x07\x07\x17#\x13\x13\x13\x13\x1b\x13\x17\x02j\x05\x1d'\x07\x1d)\x07\x1f\x17%\xfe\x0c\x1b\x1d+\x07\x11\x03\x05\x03\x07\x0f\x11\x13\x0b\x15\x0b\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x1b\x05\x05\x1d\x03\x03\x1f]\x05\x1f\x1d#\x07\x05!\x05#\x05%\x05'\x05)\x1f-\x01\x1d+\t\x07\x07\x01\x1f1\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03=\r\x01#%\x03\tCGKO\r\x03/E\x1d-\r\x03/I\x1d/\r\x03/M\x1d1\r\x03/Q\x1d3\x1d5\x1d7\x1f\x19\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1b!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1d\t\x00\x00\x00\x00\r\x03_a\x1d9\x05\x03\x0b\x03\x1d;\x1d=\x03\x01\x05\x01\x03\x039\x03\x03q\x15\x03\x01\x01\x01\x03\x0b9777u\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x15)\x03\t\x1f\x0b)\x05\t\x11\t)\x05\t\r\t)\x05\t\r\x15)\x03\t\x13\x01\x03\t\x13)\x01\t)\x01\x15)\x01\x1f\x1b\x1d)\x05\t\r\x13\x11\x03\x05\t\x05\x0b\r\x0f)\x03\r\x17)\x03\t\x17)\x03\x05\x17)\x03\x01!)\x07\t\x11\x11\x13)\x03\x05!)\x05\t\x11\x13\x04\xae\x03\x05\x01Q\x05\r\x01\x07\x04\x86\x03\x03\x01\x05\x0bP\x05\x03\x07\x04Z\x03\x03;g\x03\x0b\x19\x00\tB\x05\x05\x03\x19\tB\x05\x07\x03\x1b\tB\x05\t\x03\x1d\rG!\x1d\x0b\x0b\x05\x0b\r\x0f\x07\x03\x01\x03F\x01\r\x03\x07\x03\x07\x05F\x01\x0f\x03\x11\x05\x11\x13\x03F\x03\x11\x03/\x03\x15\x03F\x03\r\x03\x05\x03\x05\x07\x06\t\x03\x05\x07\x17\t\x19\x03F\x01\r\x03\x07\x03\x07\x05F\x01\x0f\x03\x11\x05\x11\x1d\x03F\x03\x11\x033\x03\x1f\x03F\x03\r\x03\x0b\x03\x03\x07\x06\t\x03\x0b\x07!\x0b#\x03F\x01\r\x03\x07\x03\x07\x05F\x01\x0f\x03\x11\x05\x11'\x03F\x03\x11\x03#\x03)\x03F\x03\r\x03\r\x03\x03\x07\x06\t\x03\r\x07+\r-\x03F\x01\r\x03\x07\x03\x07\x05F\x01\x0f\x03\x11\x05\x111\x03F\x03\x11\x03#\x033\x03F\x03\r\x03\x0f\x03\x05\x07\x06\t\x03\x0f\x075\x0f7\x0f\x04\x05\t\x1b%/9\x06\x03\x01\x05\x01\x00z\x07?'\x03\r\x0f\x0b\t\t\t\t!;K/iA)\x05\x13%)9\x15\x1f\x11\x19\x15\x17)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00compare_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00mhlo.backend_config\x00jit(func)/jit(main)/tridiagonal\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00jit(func)/jit(main)/select_n\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00lower\x00\x00cusolver_sytrd_ffi\x00\x08A\x13\x05/\x01\x0b;?ASU\x03W\x03Y\x03[\x11cegikmos\x03-\x0513\x035",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
