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
import numpy as np

array = np.array
float32 = np.float32
complex64 = np.complex64

data_2025_01_09 = {}

data_2025_01_09["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgtsv_ffi'],
    serialized_date=datetime.date(2025, 1, 9),
    inputs=(array([[2.348589 , 2.3354893, 2.9366405, 1.8607187],
       [1.1200227, 1.6013287, 2.879053 , 1.3576866]], dtype=float32), array([[-1.2610785 ,  1.4220167 ,  2.7938707 , -1.9043953 ],
       [-1.9423647 ,  2.4280026 ,  1.7323711 , -0.51405835]],
      dtype=float32), array([[-0.37088543, -4.999315  ,  1.7937316 , -1.0006919 ],
       [ 3.2661886 ,  0.8512876 , -0.43911585, -0.63183403]],
      dtype=float32), array([[[-1.517145  ,  6.65875   , -0.18931197],
        [ 1.3057536 ,  0.43341616,  0.2829507 ],
        [-0.16516292, -5.546491  , -0.98070467],
        [-0.6278358 , -6.1110263 , -0.9644314 ]],

       [[-4.3102446 , -1.2533706 , -2.7620814 ],
        [ 1.5553199 , -2.982489  , -0.8716973 ],
        [ 0.53110313,  0.9999317 ,  2.074623  ],
        [ 0.70297056,  0.2069549 , -3.355901  ]]], dtype=float32)),
    expected_outputs=(array([[[ 1.3763437e+00, -5.3052106e+00,  3.0924502e-01],
        [-5.8921856e-01,  8.5031107e-02, -5.4105717e-01],
        [ 2.1419016e-01, -2.5409009e+00, -6.6029988e-02],
        [ 5.3895497e-01,  7.2628009e-01,  4.4190830e-01]],

       [[-1.5719855e+00, -9.8367691e+00, -1.9573608e+01],
        [-2.2544973e+00, -6.2335539e+00, -1.2485858e+01],
        [ 1.1214201e+01,  3.2779163e+01,  7.1406868e+01],
        [ 2.8250488e+01,  8.6170906e+01,  1.9512192e+02]]], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("dl")
#loc2 = loc("d")
#loc3 = loc("du")
#loc4 = loc("b")
module @jit_tridiagonal_solve attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4xf32> loc("dl"), %arg1: tensor<2x4xf32> loc("d"), %arg2: tensor<2x4xf32> loc("du"), %arg3: tensor<2x4x3xf32> loc("b")) -> (tensor<2x4x3xf32> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @lapack_sgtsv_ffi(%arg0, %arg1, %arg2, %arg3) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4x3xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4x3xf32>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x3xf32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x3xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#3, %4 : tensor<2x4x3xi1>, tensor<2x4x3xf32> loc(#loc6)
    return %6 : tensor<2x4x3xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":842:4)
#loc6 = loc("jit(tridiagonal_solve)/jit(main)/tridiagonal_solve"(#loc5))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.8.3\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#\'\x03\xa9i-\x01-\x07\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03=O\x0b\x0bo\x0f\x1b\x0b\x0f\x13\x0b\x0b\x0b\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x1b\x17\x17\x17\x17\x1f/\x0b\x0b/o\x01\x05\x0b\x0f\x03)\x17\x1b\x07\x07\x07\x07\x0f\x0f\x07\x13#\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x02"\x05\x1f\x1d\')\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x01\x05\x1d\x1d\x19\x01\x05\x1f\x1d\x1d\x01\x05!\x1d!\x01\x05#\x03\x03%/\x05%\x05\'\x17+*\r\t\x05)\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\x1d+\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f!\x01\x03\t////#\x19\x03\x03=\r\x03?1\x1d-\x1d/\x1d1\x1f\x11\t\x00\x00\xc0\x7f\x1f\x13\t\x00\x00\x00\x00\x0b\x03\x1d3\x03\x01\x05\x01\x03\t---3\x03\tUWY[\x15\x03\x01\x01\x01\x15\x03\x05\x05\x01\x15\x03\t\t\x01\x15\x03\r\r\x01\x03\x0b---3_\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f\'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f+1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\t\x11\t)\x07\t\x11\r\t\t\x13\x1d\x01)\x01\t)\x01\x15\x1b)\x03\t\x15\x11\t\x05\x05\x05\x07\x03\x07)\x03\t\x0b)\x03\r\x0b)\x03\x05\x0b)\x03\x01\r)\x03\t\x0f)\x07\t\x05\x05\x0f)\x03\x05\r)\x07\t\x11\r\x0f)\x03\r\r\x04\xe7\x05\x01Q\x01\x07\x01\x07\x04\xd5\x03\x01\x05\x07P\x01\x03\x07\x04\xc1\x03#+\t\x0b\x13\x0b\x17\x0b\x1b\x0f\x1f\x00\x05B\x01\x05\x03\x11\x05B\x01\x07\x03\x13\tG\x03#\t\x0b\x05\x05\x05\x07\x17\t\x01\x03\x05\x07\x03F\x03\x0b\x03\x17\x03\x0b\x0bF\x03\r\x03#\x05\x15\x17\x03F\x03\x0f\x03%\x03\x19\x03F\x03\x0b\x03\x07\x03\t\x03F\x03\x11\x03)\x03\x1b\r\x06\x03\x03\x07\x07\x1f\x13\x1d\x0f\x04\x01\x03!\x06\x03\x01\x05\x01\x00B\x065#\x0f\x0b!\x03ig)\x05\x07\x05\x07-%)9\x15\x15\x17\x1f\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00select_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_tridiagonal_solve\x00dl\x00d\x00du\x00b\x00mhlo.backend_config\x00jit(tridiagonal_solve)/jit(main)/tridiagonal_solve\x00third_party/py/jax/tests/export_back_compat_test.py\x00\x00jax.result_info\x00main\x00public\x00lapack_sgtsv_ffi\x00\x08A\x13\x05#\x01\x0b79;AC\x03E\x03G\x11I1KMOQS]\x035\x05ac\x03e\x03g',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2025_01_09["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgtsv_ffi'],
    serialized_date=datetime.date(2025, 1, 9),
    inputs=(array([[2.494778962242715 , 2.3722941524275454, 2.6486249672710844,
        1.2569634243247576],
       [2.6178773862569757, 2.911181553781913 , 2.7089262037427817,
        1.3329030294395638]]), array([[ 0.6320525592643097, -7.311247302274209 , -6.653817326066262 ,
        -4.361872952980156 ],
       [-1.0889555364068737,  0.7983983288326992,  5.546873741644092 ,
        -3.8060002459696105]]), array([[-0.7417796495371639 ,  1.4000155025846106 ,  0.12321259967490755,
         0.5718093044751362 ],
       [-3.325582765183565  , -0.4069863648337625 ,  1.2064930847593267 ,
        -0.3963155951154412 ]]), array([[[ 3.829396603784013  , -1.4974483164015677 ,
         -1.0658193119885873 ],
        [ 0.6695346425328166 ,  0.23055605958233888,
          1.003789732344847  ],
        [-4.894645578906023  , -2.2358266737940045 ,
         -1.0629792862124035 ],
        [-2.897049077512793  , -4.130669838646715  ,
          3.815334195848389  ]],

       [[-3.260436578858917  , -4.181931328341748  ,
          0.36468584791257763],
        [ 1.730536552380457  , -0.12735234994740344,
         -2.028097588606966  ],
        [-1.8975815790860078 ,  3.0389728057202934 ,
          2.5403344960758054 ],
        [-1.667256154696997  , -0.3714983575699372 ,
         -1.1148362848933755 ]]])),
    expected_outputs=(array([[[10.424329531691544  , -3.952984535657851  ,
         -3.1070013279331676 ],
        [ 3.719874975078976  , -1.3495189261633447 ,
         -1.210560076067699  ],
        [ 2.2406069947622744 , -0.18461827718521784,
         -0.3401342880320912 ],
        [ 1.3098524830512592 ,  0.8937929781057479 ,
         -0.9727175919611825 ]],

       [[ 0.23222629878926748, -0.449712070255646  ,
         -0.6840492707799803 ],
        [ 0.9043685505529907 ,  1.4047606410354263 ,
          0.11432985420147246],
        [-0.8168266171952064 , -0.14811942239366266,
          0.31447425570814325],
        [ 0.15199827764678878,  0.04573555425460573,
          0.4030477860930123 ]]]),),
    mlir_module_text=r"""
#loc1 = loc("dl")
#loc2 = loc("d")
#loc3 = loc("du")
#loc4 = loc("b")
module @jit_tridiagonal_solve attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4xf64> loc("dl"), %arg1: tensor<2x4xf64> loc("d"), %arg2: tensor<2x4xf64> loc("du"), %arg3: tensor<2x4x3xf64> loc("b")) -> (tensor<2x4x3xf64> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @lapack_dgtsv_ffi(%arg0, %arg1, %arg2, %arg3) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>) -> (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x3xf64> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x3xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#3, %4 : tensor<2x4x3xi1>, tensor<2x4x3xf64> loc(#loc6)
    return %6 : tensor<2x4x3xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":842:4)
#loc6 = loc("jit(tridiagonal_solve)/jit(main)/tridiagonal_solve"(#loc5))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.3\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xa9i-\x01-\x07\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03=O\x0b\x0bo\x0f\x1b\x0b\x0f\x13\x0b\x0b\x0b/\x1f\x0b\x0b\x0b\x0b\x1b\x1b\x17\x17\x17\x17\x1f/\x0b\x0b/o\x01\x05\x0b\x0f\x03)\x17\x1b\x07\x07\x07\x07\x0f\x0f\x07\x13#\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x022\x05\x1f\x1d')\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x01\x05\x1d\x1d\x19\x01\x05\x1f\x1d\x1d\x01\x05!\x1d!\x01\x05#\x03\x03%/\x05%\x05'\x17+*\r\t\x05)\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\x1d+\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f!\x01\x03\t////#\x19\x03\x03=\r\x03?1\x1d-\x1d/\x1d1\x1f\x11\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x13\t\x00\x00\x00\x00\x0b\x03\x1d3\x03\x01\x05\x01\x03\t---3\x03\tUWY[\x15\x03\x01\x01\x01\x15\x03\x05\x05\x01\x15\x03\t\t\x01\x15\x03\r\r\x01\x03\x0b---3_\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f+1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\t\x11\t)\x07\t\x11\r\t\x0b\x13\x1d\x01)\x01\t)\x01\x15\x1b)\x03\t\x15\x11\t\x05\x05\x05\x07\x03\x07)\x03\t\x0b)\x03\r\x0b)\x03\x05\x0b)\x03\x01\r)\x03\t\x0f)\x07\t\x05\x05\x0f)\x03\x05\r)\x07\t\x11\r\x0f)\x03\r\r\x04\xe7\x05\x01Q\x01\x07\x01\x07\x04\xd5\x03\x01\x05\x07P\x01\x03\x07\x04\xc1\x03#+\t\x0b\x13\x0b\x17\x0b\x1b\x0f\x1f\x00\x05B\x01\x05\x03\x11\x05B\x01\x07\x03\x13\tG\x03#\t\x0b\x05\x05\x05\x07\x17\t\x01\x03\x05\x07\x03F\x03\x0b\x03\x17\x03\x0b\x0bF\x03\r\x03#\x05\x15\x17\x03F\x03\x0f\x03%\x03\x19\x03F\x03\x0b\x03\x07\x03\t\x03F\x03\x11\x03)\x03\x1b\r\x06\x03\x03\x07\x07\x1f\x13\x1d\x0f\x04\x01\x03!\x06\x03\x01\x05\x01\x00B\x065#\x0f\x0b!\x03ig)\x05\x07\x05\x07-%)9\x15\x15\x17\x1f\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00select_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_tridiagonal_solve\x00dl\x00d\x00du\x00b\x00mhlo.backend_config\x00jit(tridiagonal_solve)/jit(main)/tridiagonal_solve\x00third_party/py/jax/tests/export_back_compat_test.py\x00\x00jax.result_info\x00main\x00public\x00lapack_dgtsv_ffi\x00\x08A\x13\x05#\x01\x0b79;AC\x03E\x03G\x11I1KMOQS]\x035\x05ac\x03e\x03g",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2025_01_09["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgtsv_ffi'],
    serialized_date=datetime.date(2025, 1, 9),
    inputs=(array([[1.161655 +0.915484j  , 2.2949362+1.3006994j ,
        1.427427 +1.6502609j , 2.241197 +0.8698868j ],
       [1.2044945+0.8410247j , 2.5974963+1.7352734j ,
        1.9650829+0.08904955j, 1.9193184+1.9063966j ]], dtype=complex64), array([[ 4.7784514 -2.4099214j ,  0.18890767+4.239561j  ,
        -4.5527325 -0.53821725j,  0.2307941 +1.3633881j ],
       [ 1.7756704 +4.853113j  ,  1.5267694 +6.1219163j ,
         1.4264932 +0.98921436j,  4.000824  +3.364412j  ]],
      dtype=complex64), array([[-4.857645  +0.9143769j, -0.18497057+1.8075589j,
        -1.5104251 -0.9590714j,  1.8416102 -1.0174615j],
       [-0.06703581+7.4823165j,  0.9713283 +2.3853855j,
         1.7746842 -1.682743j ,  3.7718728 +4.470585j ]], dtype=complex64), array([[[ 3.211505  -2.233342j  , -3.4877088 +1.9213955j ,
         -5.9456267 -3.012781j  ],
        [ 6.849957  -1.7023281j , -2.7873719 -3.729428j  ,
          2.880858  +2.5617855j ],
        [-3.0490608 -0.37559932j, -1.5015787 -1.4306327j ,
         -1.9042435 -1.888194j  ],
        [ 2.5338044 +1.9517052j ,  1.4875154 +2.451393j  ,
         -0.898728  -5.105228j  ]],

       [[ 0.6591945 -3.2614107j ,  0.35148153+5.353837j  ,
         -0.46919197-1.339681j  ],
        [ 0.89471275-2.687814j  , -4.095198  +1.2206303j ,
          0.5824216 +1.4032336j ],
        [-0.2942064 +5.727902j  ,  2.7475152 +1.6112871j ,
          3.7658465 +1.028681j  ],
        [ 6.0720816 +0.53139037j,  1.4968821 -1.9436507j ,
          1.3021148 +2.0795968j ]]], dtype=complex64)),
    expected_outputs=(array([[[ 8.2615507e-01-0.8856711j  , -9.4191217e-01-0.12633294j ,
         -6.5045905e-01-1.3112742j  ],
        [-1.2866397e-01-0.84555596j , -2.5241747e-01-0.100036494j,
         -1.0644866e-03-0.34718406j ],
        [-3.4959085e-02-0.1300348j  , -5.5075741e-01+0.7017009j  ,
          3.4847233e+00-1.0184515j  ],
        [ 1.9227588e+00-1.5074782j  ,  1.3703994e+00-2.21213j    ,
         -5.3321910e+00+6.1347017j  ]],

       [[ 2.9257581e+00-2.2505581j  , -2.0921892e-02-3.328737j   ,
         -8.4216785e-01-0.66157204j ],
        [-1.7808166e+00+2.0819192j  ,  1.5378181e+00+2.0933375j  ,
          5.2676785e-01+0.2872307j  ],
        [ 2.7979846e+00-0.89553785j , -8.7825561e-01-0.63021123j ,
          7.0792496e-01-0.79168034j ],
        [-5.2689123e-01-0.32772654j ,  4.0581912e-01-0.10625661j ,
          4.7700096e-02+0.5221461j  ]]], dtype=complex64),),
    mlir_module_text=r"""
#loc1 = loc("dl")
#loc2 = loc("d")
#loc3 = loc("du")
#loc4 = loc("b")
module @jit_tridiagonal_solve attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4xcomplex<f32>> loc("dl"), %arg1: tensor<2x4xcomplex<f32>> loc("d"), %arg2: tensor<2x4xcomplex<f32>> loc("du"), %arg3: tensor<2x4x3xcomplex<f32>> loc("b")) -> (tensor<2x4x3xcomplex<f32>> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @lapack_cgtsv_ffi(%arg0, %arg1, %arg2, %arg3) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>, tensor<2x4x3xcomplex<f32>>) -> (tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>, tensor<2x4x3xcomplex<f32>>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x3xcomplex<f32>> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x3xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#3, %4 : tensor<2x4x3xi1>, tensor<2x4x3xcomplex<f32>> loc(#loc6)
    return %6 : tensor<2x4x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":842:4)
#loc6 = loc("jit(tridiagonal_solve)/jit(main)/tridiagonal_solve"(#loc5))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.3\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xabi/\x01-\x07\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03=O\x0b\x0bo\x0f\x1b\x0b\x0f\x13\x0b\x0b\x0b/\x1f\x0b\x0b\x0b\x0b\x1b\x1b\x17\x17\x17\x17\x1f/\x0b\x0b/o\x01\x05\x0b\x0f\x03+\x17\x1b\x0b\x07\x07\x07\x0f\x0f\x07\x13#\x07\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x02:\x05\x1f\x1d')\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x01\x05\x1d\x1d\x19\x01\x05\x1f\x1d\x1d\x01\x05!\x1d!\x01\x05#\x03\x03%/\x05%\x05'\x17+*\r\t\x05)\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\x1d+\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f#\x01\x03\t////#\x19\x03\x03=\r\x03?1\x1d-\x1d/\x1d1\x1f\x11\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x13\t\x00\x00\x00\x00\x0b\x03\x1d3\x03\x01\x05\x01\x03\t---3\x03\tUWY[\x15\x03\x01\x01\x01\x15\x03\x05\x05\x01\x15\x03\t\t\x01\x15\x03\r\r\x01\x03\x0b---3_\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f-1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\t\x11\t)\x07\t\x11\r\t\x03\x1b\x13\x1d\x01)\x01\t)\x01\x15\x1b)\x03\t\x15\x11\t\x05\x05\x05\x07\x03\x07\t)\x03\t\x0b)\x03\r\x0b)\x03\x05\x0b)\x03\x01\r)\x03\t\x0f)\x07\t\x05\x05\x0f)\x03\x05\r)\x07\t\x11\r\x0f)\x03\r\r\x04\xe7\x05\x01Q\x01\x07\x01\x07\x04\xd5\x03\x01\x05\x07P\x01\x03\x07\x04\xc1\x03#+\t\x0b\x13\x0b\x17\x0b\x1b\x0f\x1f\x00\x05B\x01\x05\x03\x11\x05B\x01\x07\x03\x13\tG\x03#\t\x0b\x05\x05\x05\x07\x17\t\x01\x03\x05\x07\x03F\x03\x0b\x03\x17\x03\x0b\x0bF\x03\r\x03%\x05\x15\x17\x03F\x03\x0f\x03'\x03\x19\x03F\x03\x0b\x03\x07\x03\t\x03F\x03\x11\x03+\x03\x1b\r\x06\x03\x03\x07\x07\x1f\x13\x1d\x0f\x04\x01\x03!\x06\x03\x01\x05\x01\x00B\x065#\x0f\x0b!\x03ig)\x05\x07\x05\x07-%)9\x15\x15\x17\x1f\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00select_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_tridiagonal_solve\x00dl\x00d\x00du\x00b\x00mhlo.backend_config\x00jit(tridiagonal_solve)/jit(main)/tridiagonal_solve\x00third_party/py/jax/tests/export_back_compat_test.py\x00\x00jax.result_info\x00main\x00public\x00lapack_cgtsv_ffi\x00\x08A\x13\x05#\x01\x0b79;AC\x03E\x03G\x11I1KMOQS]\x035\x05ac\x03e\x03g",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2025_01_09["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgtsv_ffi'],
    serialized_date=datetime.date(2025, 1, 9),
    inputs=(array([[2.1358050226934777+1.2515490287190982j ,
        2.52573287834847  +1.9370104490963989j ,
        1.8593581626523068+1.1078443185091378j ,
        1.268736634636796 +1.607486073436931j  ],
       [2.2685621662863023+1.5181884834757082j ,
        2.7756682505066643+0.15235848558182696j,
        1.881777116052583 +1.2546261732722395j ,
        1.822897170483961 +1.9104296910443275j ]]), array([[-0.463867474174968 -2.4646655634490324j ,
        -6.112774531594041 +0.8460731164954549j ,
        -1.5643409614441883+2.049388595023031j  ,
         2.436301109605215 +2.902903065455858j  ],
       [-1.8156427244728135+0.7451759772314273j ,
        -0.5941423628975409+0.38956823066972107j,
         1.6034433421887833+0.7114576516761909j ,
        -0.3659151794461677-1.596080959640956j  ]]), array([[ 0.4172337598521708+4.018584087388732j ,
        -2.882275572746347 -0.5878730500773758j,
         4.194488499350388 -1.9458424537636554j,
        -3.07082193102224  +1.0579899869475877j],
       [ 4.7275443102126165-2.97548844616448j  ,
        -1.6894986422984135+4.400798167187298j ,
        -0.540395104750296 -2.6952769125097107j,
        -1.0255037966532243+3.9367666972073296j]]), array([[[ 6.819919395019761  +2.1499312673137743j ,
          0.6734700418161376 +3.0891620622227505j ,
         -3.2696988443944903 -0.15034880110654997j],
        [ 2.3841537567230495 -1.0423410617792692j ,
          2.138568567146289  +3.2190765322434145j ,
         -4.417691991309855  -4.16117891222818j   ],
        [-0.2989557117058576 -1.517297418052348j  ,
         -1.6792920797526623 -2.481751753177451j  ,
          3.1325139021416817 -4.102339424469591j  ],
        [ 3.2268423399744837 +3.238467726974734j  ,
         -5.70991752849564   -0.18801265290565045j,
          0.9759993004917924 -1.0249259716224235j ]],

       [[ 2.9235287043852036 -4.044064719273923j  ,
         -0.6909701979214347 -0.05252693063805811j,
          3.5762303539617397 +0.4341894960679442j ],
        [-2.9609754256489698 +4.697696896621638j  ,
         -5.374947585788317  +1.1127752723559232j ,
         -0.2503591039896783 -1.384744852690059j  ],
        [-0.7189881968207082 +2.541200347299558j  ,
         -1.0638470670028264 +5.1135572483425396j ,
         -5.377102891353032  -1.0001590792759882j ],
        [-0.43765500188426254+2.9506272066104526j ,
         -3.537707554193495  +2.8910129954577943j ,
         -7.046242995372967  -4.518320543762175j  ]]])),
    expected_outputs=(array([[[-5.449499845587512  -1.2118267316768148j ,
          1.4775756259003918 -2.440755570388714j  ,
          1.9725747895052064 +2.762364699587123j  ],
        [-2.72967048786596   -2.0947009639011163j ,
          1.5668444849196348 -1.672423658866126j  ,
          1.2411426923890478 +2.409016989432768j  ],
        [ 1.4100549429774594 -1.0089760817348514j ,
         -0.27189383173847337+1.7996055449700372j ,
         -1.7700835879532208 +0.8062991828956373j ],
        [ 0.4239373312040521 +0.4192010500618847j ,
         -0.8304889615232484 +0.15460172924293059j,
          0.9275550478620849 -0.7778697597485181j ]],

       [[-0.510376948136591  -0.0951434045124143j ,
          1.1840492867743164 +2.503692671816512j  ,
          1.9753023583944396 +3.434897560366379j  ],
        [ 0.6576417648364737 -0.39760283818549785j,
          0.15935875034246666+0.8641123638997179j ,
          0.9772572454441909 +1.714760456258439j  ],
        [ 1.001111614081475  -0.08370796768300333j,
         -0.5055136573561947 +1.9775432937953357j ,
         -1.794469937660869  +1.5890107854613884j ],
        [-0.3781536437720755 -1.6044730440025805j ,
         -0.30848414888935966+0.6571535398637388j ,
          2.4739918689902995 +0.10391436482640627j]]]),),
    mlir_module_text=r"""
#loc1 = loc("dl")
#loc2 = loc("d")
#loc3 = loc("du")
#loc4 = loc("b")
module @jit_tridiagonal_solve attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4xcomplex<f64>> loc("dl"), %arg1: tensor<2x4xcomplex<f64>> loc("d"), %arg2: tensor<2x4xcomplex<f64>> loc("du"), %arg3: tensor<2x4x3xcomplex<f64>> loc("b")) -> (tensor<2x4x3xcomplex<f64>> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @lapack_zgtsv_ffi(%arg0, %arg1, %arg2, %arg3) {mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 1, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [3], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4xcomplex<f64>>, tensor<2x4xcomplex<f64>>, tensor<2x4xcomplex<f64>>, tensor<2x4x3xcomplex<f64>>) -> (tensor<2x4xcomplex<f64>>, tensor<2x4xcomplex<f64>>, tensor<2x4xcomplex<f64>>, tensor<2x4x3xcomplex<f64>>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x3xcomplex<f64>> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x3xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#3, %4 : tensor<2x4x3xi1>, tensor<2x4x3xcomplex<f64>> loc(#loc6)
    return %6 : tensor<2x4x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":842:4)
#loc6 = loc("jit(tridiagonal_solve)/jit(main)/tridiagonal_solve"(#loc5))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.8.3\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xabi/\x01-\x07\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03=O\x0b\x0bo\x0f\x1b\x0b\x0f\x13\x0b\x0b\x0bO\x1f\x0b\x0b\x0b\x0b\x1b\x1b\x17\x17\x17\x17\x1f/\x0b\x0b/o\x01\x05\x0b\x0f\x03+\x17\x1b\x0b\x07\x07\x07\x0f\x0f\x07\x13#\x07\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x02Z\x05\x1f\x1d')\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x01\x05\x1d\x1d\x19\x01\x05\x1f\x1d\x1d\x01\x05!\x1d!\x01\x05#\x03\x03%/\x05%\x05'\x17+*\r\t\x05)\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x01\x1d+\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f#\x01\x03\t////#\x19\x03\x03=\r\x03?1\x1d-\x1d/\x1d1\x1f\x11!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x13\t\x00\x00\x00\x00\x0b\x03\x1d3\x03\x01\x05\x01\x03\t---3\x03\tUWY[\x15\x03\x01\x01\x01\x15\x03\x05\x05\x01\x15\x03\t\t\x01\x15\x03\r\r\x01\x03\x0b---3_\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f-1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\t\x11\t)\x07\t\x11\r\t\x03\x1b\x13\x1d\x01)\x01\t)\x01\x15\x1b)\x03\t\x15\x11\t\x05\x05\x05\x07\x03\x07\x0b)\x03\t\x0b)\x03\r\x0b)\x03\x05\x0b)\x03\x01\r)\x03\t\x0f)\x07\t\x05\x05\x0f)\x03\x05\r)\x07\t\x11\r\x0f)\x03\r\r\x04\xe7\x05\x01Q\x01\x07\x01\x07\x04\xd5\x03\x01\x05\x07P\x01\x03\x07\x04\xc1\x03#+\t\x0b\x13\x0b\x17\x0b\x1b\x0f\x1f\x00\x05B\x01\x05\x03\x11\x05B\x01\x07\x03\x13\tG\x03#\t\x0b\x05\x05\x05\x07\x17\t\x01\x03\x05\x07\x03F\x03\x0b\x03\x17\x03\x0b\x0bF\x03\r\x03%\x05\x15\x17\x03F\x03\x0f\x03'\x03\x19\x03F\x03\x0b\x03\x07\x03\t\x03F\x03\x11\x03+\x03\x1b\r\x06\x03\x03\x07\x07\x1f\x13\x1d\x0f\x04\x01\x03!\x06\x03\x01\x05\x01\x00B\x065#\x0f\x0b!\x03ig)\x05\x07\x05\x07-%)9\x15\x15\x17\x1f\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00select_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_tridiagonal_solve\x00dl\x00d\x00du\x00b\x00mhlo.backend_config\x00jit(tridiagonal_solve)/jit(main)/tridiagonal_solve\x00third_party/py/jax/tests/export_back_compat_test.py\x00\x00jax.result_info\x00main\x00public\x00lapack_zgtsv_ffi\x00\x08A\x13\x05#\x01\x0b79;AC\x03E\x03G\x11I1KMOQS]\x035\x05ac\x03e\x03g",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
