# Copyright 2026 The JAX Authors.
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

data_2026_02_04 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["f32"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_sytrd_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[ 3.517052  ,  7.324363  ,  0.17285964,  6.048615  ],
        [ 2.634785  , -1.5823158 ,  5.00702   , -9.524221  ],
        [-9.572195  , -1.6510249 , -7.1607885 ,  0.27307302],
        [-3.616519  ,  9.1224375 ,  8.262698  , -9.876812  ]],

       [[-9.066103  ,  0.91170084, -9.563753  , -2.7974956 ],
        [ 1.9257098 , -3.1175334 ,  3.5549293 , -7.2144737 ],
        [-0.92147785,  5.217206  ,  3.3423822 ,  3.6858516 ],
        [ 4.3601804 ,  6.6677866 , -5.461852  , -6.232803  ]]],
      dtype=float32),),
    expected_outputs=(array([[[  3.517052  ,   7.324363  ,   0.17285964,   6.048615  ],
        [-10.566372  ,  -2.8193984 ,   5.00702   ,  -9.524221  ],
        [ -0.7251028 ,  -3.1670432 ,  -4.499419  ,   0.27307302],
        [ -0.27395472,   0.9202458 ,  11.912114  , -11.301102  ]],

       [[ -9.066103  ,   0.91170084,  -9.563753  ,  -2.7974956 ],
        [ -4.854756  ,   0.4297719 ,   3.5549293 ,  -7.2144737 ],
        [ -0.13590185,  -5.4479847 , -12.038837  ,   3.6858516 ],
        [  0.6430503 ,   0.5523631 ,   3.6679316 ,   5.6011105 ]]],
      dtype=float32), array([[  3.517052 ,  -2.8193984,  -4.499419 , -11.301102 ],
       [ -9.066103 ,   0.4297719, -12.038837 ,   5.6011105]],
      dtype=float32), array([[-10.566372 ,  -3.1670432,  11.912114 ],
       [ -4.854756 ,  -5.4479847,   3.6679316]], dtype=float32), array([[1.2493557, 1.0829235, 0.       ],
       [1.3966646, 1.5324438, 0.       ]], dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf32> loc("x")) -> (tensor<2x4x4xf32> {jax.result_info = "result[0]"}, tensor<2x4xf32> {jax.result_info = "result[1]"}, tensor<2x3xf32> {jax.result_info = "result[2]"}, tensor<2x3xf32> {jax.result_info = "result[3]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o], [i, p], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=3, p=3}, custom>} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc4)
    %14 = stablehlo.select %13, %0#2, %12 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc4)
    %15 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc4)
    %17 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc4)
    %18 = stablehlo.select %17, %0#3, %16 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc4)
    return %6, %10, %14, %18 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":764:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("tridiagonal"(#loc3))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.12.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#\'+\x03\xe7\x997\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03S\x0f\x0b/OOo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x0b\x0bo\x05\x1f\x0f_\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x13\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x033\x17\x1b\x07\x07\x17\x07\x07\x17\x0f\x0f\x07\x13\x17#\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x13\x02"\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x15\x03\x05\x1f\x05!\x1d\x1b\x1d\x05#\x17\x1f\xf2\x0b\x1b\x05%\x03\x07#U%[\'}\x05\'\x05)\x05+\x1f\'\x01\x1d-\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f5!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f!1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x037\r\x01#\x1f\x03\t=AEI\r\x03+?\x1d/\r\x03+C\x1d1\r\x03+G\x1d3\r\x03+K\x1d5\x1d7\x1d9\x1f\x15\t\x00\x00\xc0\x7f\x1f\x17\t\x00\x00\x00\x00\r\x03WY\x1d;\x05\x03\r\x03]_\x1d=\x1d?\x0b\x03\x1dA\x1dC\x03\x01\x05\x01\x03\x033\x03\x03o\x15\x03\x01\x01\x01\x03\x0b3///s\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f11\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x11\t\x11\x11\x11\x11\x11\r\r\x03\x7f\x0b\x85\x8b\x8f\x93\x97\x01\x01\x01\x01\x01\x13\x07{\x81\x83\x11\x03\x05\x11\x03\t\x13\x07{\x87\x89\x11\x03\r\x11\x03\x11\x13\x05{\x8d\x11\x03\x15\x13\x05{\x91\x11\x03\x19\x13\x05{\x95\x11\x03\x1d\x13\x03{\x01\t\x01\x02\x02)\x05\t\r\x0b)\x07\t\x11\x11\x0b\x01\t)\x05\t\x11\x0b\x1d\x13)\x05\t\x05\t)\x01\x0b)\x01\x19\x1b)\x03\t\x19)\x05\t\r\t\x11\x03\x07\t\x07\r\x05\x05)\x03\r\x11)\x03\t\x11)\x03\x05\x11)\x03\x01\x0f)\x03\t\t)\x07\t\x05\x05\t)\x03\x05\x0f)\x07\t\x11\x11\t)\x03\r\x0f)\x05\t\x11\t)\x03\t\x0f\x04J\x03\x05\x01Q\x03\x07\x01\x07\x04"\x03\x03\x01\x05\tP\x03\x03\x07\x04\xf6\x02\x035[\x03\x0f\x13\x00\x07B\x03\x05\x03\x15\x07B\x01\x07\x03\x17\x0bG\x01!\t\x0b\x07\r\x05\x05\x1b\x03\x01\x03F\x01\x0b\x03\x1b\x03\x05\rF\x01\r\x03)\x05\x0f\x11\x03F\x01\x0f\x03+\x03\x13\x03F\x01\x0b\x03\x07\x03\x03\x03F\x01\x11\x03/\x03\x15\x05\x06\x01\x03\x07\x07\x19\x07\x17\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\r\x03\x03\x03F\x01\x13\x033\x03\x1d\x05\x06\x01\x03\r\x07!\t\x1f\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1d\x03%\x05\x06\x01\x03\x05\x07)\x0b\'\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1d\x03-\x05\x06\x01\x03\x05\x071\r/\x0f\x04\x03\t\x1b#+3\x06\x03\x01\x05\x01\x00r\x07E)\x03\x05\x1f\r\x0f\x0b\x15\x15\x15\x15!%3)s\x15\x19\x05\x13%)9\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00tridiagonal\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00result[3]\x00main\x00public\x00lower\x00num_batch_dims\x001\x00\x00hipsolver_sytrd_ffi\x00\x08E\x15\x05#\x01\x0b59;MO\x03Q\x03S\x11acegikmq\x03)\x05uw\x03-\x03y\x031',
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["f64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_sytrd_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[ 7.035349169825569  ,  1.484998381764381  ,
          8.018528146255836  , -4.530334856568022  ],
        [ 9.878251523361389  ,  4.093041770635891  ,
         -9.877581529120809  ,  6.311932164866882  ],
        [-7.49184464571486   , -6.59107026401994   ,
          1.1241396190859394 ,  7.700650796628231  ],
        [ 4.63003758138132   , -0.04006710103875122,
          1.3302998218441715 ,  1.4217252024769529 ]],

       [[ 9.87349262986562   , -0.7610867121172475 ,
         -5.8439127140031255 ,  7.363308757108598  ],
        [-8.602243503020022  ,  1.0149675283651831 ,
          6.189214809971542  ,  3.8755163293613073 ],
        [-7.2550065371548556 , -0.8199129790471194 ,
          3.3263222105883834 ,  9.414314634323905  ],
        [ 9.50720541720979   ,  8.190248152372803  ,
          6.1792809430264235 , -3.395527362127919  ]]]),),
    expected_outputs=(array([[[  7.035349169825569  ,   1.484998381764381  ,
           8.018528146255836  ,  -4.530334856568022  ],
        [-13.234229760712234  ,   7.836821389949192  ,
          -9.877581529120809  ,   6.311932164866882  ],
        [ -0.32414713736847234,  -3.231749037742704  ,
           2.91526168804856   ,   7.700650796628231  ],
        [  0.20032628796856145,   0.8955299566988397 ,
           1.1712939642530218 ,  -4.113176485798968  ]],

       [[  9.87349262986562   ,  -0.7610867121172475 ,
          -5.8439127140031255 ,   7.363308757108598  ],
        [ 14.731621363055496  , -10.83373801371347   ,
           6.189214809971542  ,   3.8755163293613073 ],
        [  0.31092176880233485,   4.1168822837917665 ,
           8.31687010427185   ,   9.414314634323905  ],
        [ -0.4074423792104866 ,   0.5649683724391054 ,
          -1.2386684521243119 ,   3.4626302862672684 ]]]), array([[  7.035349169825569 ,   7.836821389949192 ,   2.91526168804856  ,
         -4.113176485798968 ],
       [  9.87349262986562  , -10.83373801371347  ,   8.31687010427185  ,
          3.4626302862672684]]), array([[-13.234229760712234 ,  -3.231749037742704 ,   1.1712939642530218],
       [ 14.731621363055496 ,   4.1168822837917665,  -1.2386684521243119]]), array([[1.7464168071712372, 1.109893986970275 , 0.                ],
       [1.5839305322218669, 1.51608268641105  , 0.                ]])),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf64> loc("x")) -> (tensor<2x4x4xf64> {jax.result_info = "result[0]"}, tensor<2x4xf64> {jax.result_info = "result[1]"}, tensor<2x3xf64> {jax.result_info = "result[2]"}, tensor<2x3xf64> {jax.result_info = "result[3]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o], [i, p], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=3, p=3}, custom>} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xf64>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc4)
    %14 = stablehlo.select %13, %0#2, %12 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc4)
    %15 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc4)
    %17 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc4)
    %18 = stablehlo.select %17, %0#3, %16 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc4)
    return %6, %10, %14, %18 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":764:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("tridiagonal"(#loc3))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.12.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#\'+\x03\xe7\x997\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03S\x0f\x0b/OOo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x0b\x0bo\x05\x1f\x0f_\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x13\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x033\x17\x1b\x07\x07\x17\x07\x07\x17\x0f\x0f\x07\x13\x17#\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x13\x022\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x15\x03\x05\x1f\x05!\x1d\x1b\x1d\x05#\x17\x1f\xf2\x0b\x1b\x05%\x03\x07#U%[\'}\x05\'\x05)\x05+\x1f\'\x01\x1d-\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f5!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f!1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x037\r\x01#\x1f\x03\t=AEI\r\x03+?\x1d/\r\x03+C\x1d1\r\x03+G\x1d3\r\x03+K\x1d5\x1d7\x1d9\x1f\x15\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x17\t\x00\x00\x00\x00\r\x03WY\x1d;\x05\x03\r\x03]_\x1d=\x1d?\x0b\x03\x1dA\x1dC\x03\x01\x05\x01\x03\x033\x03\x03o\x15\x03\x01\x01\x01\x03\x0b3///s\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f11\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x11\t\x11\x11\x11\x11\x11\r\r\x03\x7f\x0b\x85\x8b\x8f\x93\x97\x01\x01\x01\x01\x01\x13\x07{\x81\x83\x11\x03\x05\x11\x03\t\x13\x07{\x87\x89\x11\x03\r\x11\x03\x11\x13\x05{\x8d\x11\x03\x15\x13\x05{\x91\x11\x03\x19\x13\x05{\x95\x11\x03\x1d\x13\x03{\x01\t\x01\x02\x02)\x05\t\r\x0b)\x07\t\x11\x11\x0b\x01\x0b)\x05\t\x11\x0b\x1d\x13)\x05\t\x05\t)\x01\x0b)\x01\x19\x1b)\x03\t\x19)\x05\t\r\t\x11\x03\x07\t\x07\r\x05\x05)\x03\r\x11)\x03\t\x11)\x03\x05\x11)\x03\x01\x0f)\x03\t\t)\x07\t\x05\x05\t)\x03\x05\x0f)\x07\t\x11\x11\t)\x03\r\x0f)\x05\t\x11\t)\x03\t\x0f\x04J\x03\x05\x01Q\x03\x07\x01\x07\x04"\x03\x03\x01\x05\tP\x03\x03\x07\x04\xf6\x02\x035[\x03\x0f\x13\x00\x07B\x03\x05\x03\x15\x07B\x01\x07\x03\x17\x0bG\x01!\t\x0b\x07\r\x05\x05\x1b\x03\x01\x03F\x01\x0b\x03\x1b\x03\x05\rF\x01\r\x03)\x05\x0f\x11\x03F\x01\x0f\x03+\x03\x13\x03F\x01\x0b\x03\x07\x03\x03\x03F\x01\x11\x03/\x03\x15\x05\x06\x01\x03\x07\x07\x19\x07\x17\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\r\x03\x03\x03F\x01\x13\x033\x03\x1d\x05\x06\x01\x03\r\x07!\t\x1f\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1d\x03%\x05\x06\x01\x03\x05\x07)\x0b\'\x03F\x01\x0f\x03\x13\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1d\x03-\x05\x06\x01\x03\x05\x071\r/\x0f\x04\x03\t\x1b#+3\x06\x03\x01\x05\x01\x00r\x07E)\x03\x05\x1f\r\x0f\x0b\x15\x15\x15\x15!%3)s\x15\x19\x05\x13%)9\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00tridiagonal\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00result[3]\x00main\x00public\x00lower\x00num_batch_dims\x001\x00\x00hipsolver_sytrd_ffi\x00\x08E\x15\x05#\x01\x0b59;MO\x03Q\x03S\x11acegikmq\x03)\x05uw\x03-\x03y\x031',
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["c64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_sytrd_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[-4.489628   -1.9558684j , -8.210964   +7.1605806j ,
         -1.546595   +3.65443j   , -0.42576104 +4.1540704j ],
        [ 1.2732816  +4.58178j   , -4.3097434  +1.9485371j ,
         -7.340492   -4.6551476j ,  0.51207775 -1.5729961j ],
        [-1.6132524  +9.396326j  , -8.6812315  +0.57193804j,
         -4.0404854  +5.580052j  ,  7.198982   -7.8558793j ],
        [ 5.0999346  -5.281331j  , -0.20722196 -8.790846j  ,
         -8.34889    +5.4901276j ,  1.5386024  -7.8014874j ]],

       [[-3.7505336  +4.8419423j , -6.9383445  -6.1508617j ,
         -5.090222   +8.55102j   , -2.8638835  -2.8485649j ],
        [ 0.39724413 -9.112975j  ,  5.8403125  -4.28845j   ,
         -5.876785   +5.7571j    ,  0.81345195 -8.658585j  ],
        [-9.730914   -4.9138327j ,  4.2365274  +0.08631961j,
         -0.020483766+2.811939j  , -3.4501777  -7.5303087j ],
        [-6.0442333  +6.9959636j , -9.286192   +0.29182708j,
          3.5280166  +8.255178j  ,  5.123959   -3.4980805j ]]],
      dtype=complex64),),
    expected_outputs=(array([[[-4.4896278e+00+0.j        , -8.2109642e+00+7.1605806j ,
         -1.5465950e+00+3.65443j   , -4.2576104e-01+4.1540704j ],
        [-1.2938673e+01+0.j        , -1.0068893e-02+0.j        ,
         -7.3404918e+00-4.6551476j ,  5.1207775e-01-1.5729961j ],
        [ 9.0255260e-02+0.63205916j, -1.2945498e+01+0.j        ,
          2.4201365e+00+0.j        ,  7.1989818e+00-7.8558793j ],
        [ 2.1653868e-01-0.44142157j,  5.5885059e-03+0.24181136j,
          7.6457381e+00+0.j        , -9.2216911e+00+0.j        ]],

       [[-3.7505336e+00+0.j        , -6.9383445e+00-6.1508617j ,
         -5.0902219e+00+8.55102j   , -2.8638835e+00-2.8485649j ],
        [-1.6956322e+01+0.j        ,  3.5084348e+00+0.j        ,
         -5.8767848e+00+5.7571j    ,  8.1345195e-01-8.658585j  ],
        [-3.2297978e-01-0.45276803j,  8.2233658e+00+0.j        ,
          1.4219294e+00+0.j        , -3.4501777e+00-7.5303087j ],
        [-4.3895450e-01+0.1726321j , -7.5097442e-02-0.18226749j,
          1.1053572e+01+0.j        ,  6.0134225e+00+0.j        ]]],
      dtype=complex64), array([[-4.489628   , -0.010068893,  2.4201365  , -9.221691   ],
       [-3.7505336  ,  3.5084348  ,  1.4219294  ,  6.0134225  ]],
      dtype=float32), array([[-12.938673, -12.945498,   7.645738],
       [-16.956322,   8.223366,  11.053572]], dtype=float32), array([[1.0984089+0.35411513j, 1.7074363+0.55748767j,
        1.634002 -0.77333146j],
       [1.0234275-0.5374382j , 1.904741 +0.19733457j,
        1.2192558+0.9756674j ]], dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f32>> loc("x")) -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<2x4xf32> {jax.result_info = "result[1]"}, tensor<2x3xf32> {jax.result_info = "result[2]"}, tensor<2x3xcomplex<f32>> {jax.result_info = "result[3]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o], [i, p], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=3, p=3}, custom>} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xcomplex<f32>>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc4)
    %14 = stablehlo.select %13, %0#2, %12 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc4)
    %15 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<2x3xcomplex<f32>> loc(#loc4)
    %17 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc4)
    %18 = stablehlo.select %17, %0#3, %16 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>> loc(#loc4)
    return %6, %10, %14, %18 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":764:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("tridiagonal"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xef\x9b=\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03U\x0f\x0b/OOo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f/\x1f\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x0b\x0bo\x05\x1f\x0f_\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x13\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x039\x1b\x07\x07\x17\x17\x17\x07\x0b\x07\x17\x0f\x0f\x0f\x07\x13\x17#\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x13\x02v\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x15\x03\x05\x1f\x05!\x1d\x1b\x1d\x05#\x17\x1f\xf2\x0b\x1b\x05%\x03\x07#W%]'\x7f\x05'\x05)\x05+\x1f-\x01\x1d-\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f;!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x037\r\x01#%\x03\t=AEI\r\x03+?\x1d/\r\x03+C\x1d1\r\x03+G\x1d3\r\x03+K\x1d5\x1d7\x1d9\x1f\x19\t\x00\x00\xc0\x7f\x1f\x1b\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x1d\t\x00\x00\x00\x00\r\x03Y[\x1d;\x05\x03\r\x03_a\x1d=\x1d?\x0b\x03\x1dA\x1dC\x03\x01\x05\x01\x03\x033\x03\x03q\x15\x03\x01\x01\x01\x03\x0b3///u\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f71\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x11\t\x11\x11\x11\x11\x11\r\r\x03\x81\x0b\x87\x8d\x91\x95\x99\x01\x01\x01\x01\x01\x13\x07}\x83\x85\x11\x03\x05\x11\x03\t\x13\x07}\x89\x8b\x11\x03\r\x11\x03\x11\x13\x05}\x8f\x11\x03\x15\x13\x05}\x93\x11\x03\x19\x13\x05}\x97\x11\x03\x1d\x13\x03}\x01\t\x01\x02\x02)\x07\t\x11\x11\x13\x01\t)\x05\t\x11\t)\x05\t\r\t)\x05\t\r\x13\x1d\x03\t\x13)\x05\t\x05\x07)\x01\t)\x01\x13)\x01\x1f\x1b)\x03\t\x1f)\x05\t\r\x07\x11\x03\x05\t\x05\x0b\r\x0f)\x03\r\x15)\x03\t\x15)\x03\x05\x15)\x03\x01\x11)\x03\t\x07)\x07\t\x05\x05\x07)\x03\x05\x11)\x07\t\x11\x11\x07)\x03\r\x11)\x05\t\x11\x07)\x03\t\x11\x04b\x03\x05\x01Q\x03\x07\x01\x07\x04:\x03\x03\x01\x05\tP\x03\x03\x07\x04\x0e\x03\x037_\x03\x0b\x13\x00\x07B\x03\x05\x03\x19\x07B\x03\x07\x03\x1b\x07B\x01\t\x03\x1d\x0bG\x01!\x0b\x0b\x05\x0b\r\x0f!\x03\x01\x03F\x01\r\x03!\x03\x07\rF\x01\x0f\x03/\x05\x11\x13\x03F\x01\x11\x031\x03\x15\x03F\x01\r\x03\x05\x03\x05\x03F\x01\x13\x035\x03\x17\x05\x06\x01\x03\x05\x07\x1b\t\x19\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\x0b\x03\x03\x03F\x01\x15\x039\x03\x1f\x05\x06\x01\x03\x0b\x07#\x0b!\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\r\x03\x03\x03F\x01\x15\x03#\x03'\x05\x06\x01\x03\r\x07+\r)\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\x0f\x03\x05\x03F\x01\x15\x03#\x03/\x05\x06\x01\x03\x0f\x073\x0f1\x0f\x04\x03\t\x1d%-5\x06\x03\x01\x05\x01\x00r\x07E)\x03\x05\x1f\r\x0f\x0b\x15\x15\x15\x15!%3)s\x15\x19\x05\x13%)9\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00tridiagonal\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00result[3]\x00main\x00public\x00lower\x00num_batch_dims\x001\x00\x00hipsolver_sytrd_ffi\x00\x08I\x17\x05#\x01\x0b59;MO\x03Q\x03S\x03U\x11cegikmos\x03)\x05wy\x03-\x03{\x031",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["c128"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_sytrd_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[ 9.636811089187177  -5.5483943045483075j ,
         -0.7013595675721049 +9.280476944337625j  ,
          6.277881297993044  +9.583766011933022j  ,
          1.5176610958978305 -6.759859673083863j  ],
        [-0.3885087082062455 +0.5895084066496903j ,
          6.964207617750493  +2.527176875608994j  ,
          5.538066653380939  -7.747050813954031j  ,
         -8.175356094532935  -4.1980578398170625j ],
        [ 1.0279072759095182 +3.4596945172274793j ,
          8.950471968823695  -2.304755231947957j  ,
          5.421547479246534  +4.581737009453139j  ,
         -4.897557901418952  +3.5957170627858677j ],
        [ 8.035494168741337  +3.544344260400223j  ,
          9.975999717803504  +6.346230772630385j  ,
          4.740819751509449  +2.756160834913411j  ,
         -3.4198929512643    +0.8932196515922382j ]],

       [[ 9.53448278116884   -2.772437604863242j  ,
         -6.107167742373565  +1.7555826763979088j ,
          0.8426783600121581 -1.4969268724718319j ,
         -7.791881813923409  +2.569663008343241j  ],
        [-0.23022736962356838+5.931740292160859j  ,
          0.19505301380201168+5.199989565595661j  ,
          1.6452867779721974 +7.611177005682343j  ,
          6.421835818993721  +7.956773248634036j  ],
        [ 4.514416045392142  +0.24019209063138547j,
          2.7074137337969297 +7.481936712692281j  ,
         -4.349739517583451  -7.480660445679543j  ,
         -4.16307684864292   -9.500272650463366j  ],
        [-7.187003466047763  -4.305905192850112j  ,
         -2.790206322615103  -1.8113900331362842j ,
          3.92920022932449   +4.618786545064131j  ,
          8.34362192900138   -7.169529907261547j  ]]]),),
    expected_outputs=(array([[[  9.636811089187177  +0.j                 ,
          -0.7013595675721049 +9.280476944337625j  ,
           6.277881297993044  +9.583766011933022j  ,
           1.5176610958978305 -6.759859673083863j  ],
        [  9.52134872118296   +0.j                 ,
          -2.0750820650530635 +0.j                 ,
           5.538066653380939  -7.747050813954031j  ,
          -8.175356094532935  -4.1980578398170625j ],
        [ -0.08266529223728494-0.3540339936208098j ,
         -12.876186833321775  +0.j                 ,
          12.221606003386894  +0.j                 ,
          -4.897557901418952  +3.5957170627858677j ],
        [ -0.7867983987261797 -0.4044627845907229j ,
           0.3731807938459702 +0.29875225015983237j,
          -7.533551178724111  +0.j                 ,
          -1.1806617926011036 +0.j                 ]],

       [[  9.53448278116884   +0.j                 ,
          -6.107167742373565  +1.7555826763979088j ,
           0.8426783600121581 -1.4969268724718319j ,
          -7.791881813923409  +2.569663008343241j  ],
        [ 11.219181358613488  +0.j                 ,
          -2.758824792396279  +0.j                 ,
           1.6452867779721974 +7.611177005682343j  ,
           6.421835818993721  +7.956773248634036j  ],
        [ -0.3022871001148797 -0.17758826769531866j,
          11.759659907134653  +0.j                 ,
           3.455913613847388  +0.j                 ,
          -4.16307684864292   -9.500272650463366j  ],
        [  0.34127558839537164+0.5528899789954966j ,
           0.29825641618428594+0.3179804603998629j ,
          -1.1276010073572176 +0.j                 ,
           3.4918466037688343 +0.j                 ]]]), array([[ 9.636811089187177 , -2.0750820650530635, 12.221606003386894 ,
        -1.1806617926011036],
       [ 9.53448278116884  , -2.758824792396279 ,  3.455913613847388 ,
         3.4918466037688343]]), array([[  9.52134872118296  , -12.876186833321775 ,  -7.533551178724111 ],
       [ 11.219181358613488 ,  11.759659907134653 ,  -1.1276010073572176]]), array([[1.0408039574626542-0.06191438040055822j,
        1.4371627190112481+0.5236740849282041j ,
        1.5581940209289933-0.8297104525068505j ],
       [1.0205208706646687-0.5287141817710955j ,
        1.6435963381260008-0.24653387207157834j,
        1.809618408219541 -0.586956585338351j  ]])),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f64>> loc("x")) -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<2x4xf64> {jax.result_info = "result[1]"}, tensor<2x3xf64> {jax.result_info = "result[2]"}, tensor<2x3xcomplex<f64>> {jax.result_info = "result[3]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_sytrd_ffi(%arg0) {mhlo.backend_config = {lower = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o], [i, p], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=3, p=3}, custom>} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xcomplex<f64>>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc4)
    %14 = stablehlo.select %13, %0#2, %12 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc4)
    %15 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<2x3xcomplex<f64>> loc(#loc4)
    %17 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc4)
    %18 = stablehlo.select %17, %0#3, %16 : tensor<2x3xi1>, tensor<2x3xcomplex<f64>> loc(#loc4)
    return %6, %10, %14, %18 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":764:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("tridiagonal"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xef\x9b=\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03U\x0f\x0b/OOo\x0f\x0b\x0b\x1b\x13\x0b\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/O\x1f\x13\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x0b\x0bo\x05\x1f\x0f_\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x13\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x039\x1b\x07\x07\x17\x17\x17\x07\x0b\x07\x17\x0f\x0f\x0f\x07\x13\x17#\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x13\x02\xa6\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x15\x03\x05\x1f\x05!\x1d\x1b\x1d\x05#\x17\x1f\xf2\x0b\x1b\x05%\x03\x07#W%]'\x7f\x05'\x05)\x05+\x1f-\x01\x1d-\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f;!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x037\r\x01#%\x03\t=AEI\r\x03+?\x1d/\r\x03+C\x1d1\r\x03+G\x1d3\r\x03+K\x1d5\x1d7\x1d9\x1f\x19\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1b!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1d\t\x00\x00\x00\x00\r\x03Y[\x1d;\x05\x03\r\x03_a\x1d=\x1d?\x0b\x03\x1dA\x1dC\x03\x01\x05\x01\x03\x033\x03\x03q\x15\x03\x01\x01\x01\x03\x0b3///u\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f71\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x11\t\x11\x11\x11\x11\x11\r\r\x03\x81\x0b\x87\x8d\x91\x95\x99\x01\x01\x01\x01\x01\x13\x07}\x83\x85\x11\x03\x05\x11\x03\t\x13\x07}\x89\x8b\x11\x03\r\x11\x03\x11\x13\x05}\x8f\x11\x03\x15\x13\x05}\x93\x11\x03\x19\x13\x05}\x97\x11\x03\x1d\x13\x03}\x01\t\x01\x02\x02)\x07\t\x11\x11\x13\x01\x0b)\x05\t\x11\t)\x05\t\r\t)\x05\t\r\x13\x1d\x03\t\x13)\x05\t\x05\x07)\x01\t)\x01\x13)\x01\x1f\x1b)\x03\t\x1f)\x05\t\r\x07\x11\x03\x05\t\x05\x0b\r\x0f)\x03\r\x15)\x03\t\x15)\x03\x05\x15)\x03\x01\x11)\x03\t\x07)\x07\t\x05\x05\x07)\x03\x05\x11)\x07\t\x11\x11\x07)\x03\r\x11)\x05\t\x11\x07)\x03\t\x11\x04b\x03\x05\x01Q\x03\x07\x01\x07\x04:\x03\x03\x01\x05\tP\x03\x03\x07\x04\x0e\x03\x037_\x03\x0b\x13\x00\x07B\x03\x05\x03\x19\x07B\x03\x07\x03\x1b\x07B\x01\t\x03\x1d\x0bG\x01!\x0b\x0b\x05\x0b\r\x0f!\x03\x01\x03F\x01\r\x03!\x03\x07\rF\x01\x0f\x03/\x05\x11\x13\x03F\x01\x11\x031\x03\x15\x03F\x01\r\x03\x05\x03\x05\x03F\x01\x13\x035\x03\x17\x05\x06\x01\x03\x05\x07\x1b\t\x19\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\x0b\x03\x03\x03F\x01\x15\x039\x03\x1f\x05\x06\x01\x03\x0b\x07#\x0b!\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\r\x03\x03\x03F\x01\x15\x03#\x03'\x05\x06\x01\x03\r\x07+\r)\x03F\x01\x11\x03\x17\x03\x15\x03F\x01\r\x03\x0f\x03\x05\x03F\x01\x15\x03#\x03/\x05\x06\x01\x03\x0f\x073\x0f1\x0f\x04\x03\t\x1d%-5\x06\x03\x01\x05\x01\x00r\x07E)\x03\x05\x1f\r\x0f\x0b\x15\x15\x15\x15!%3)s\x15\x19\x05\x13%)9\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00tridiagonal\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00result[3]\x00main\x00public\x00lower\x00num_batch_dims\x001\x00\x00hipsolver_sytrd_ffi\x00\x08I\x17\x05#\x01\x0b59;MO\x03Q\x03S\x03U\x11cegikmos\x03)\x05wy\x03-\x03{\x031",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
