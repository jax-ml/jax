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

data_2026_08_13 = {"qr": {}}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["qr"]["f32"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_gesvd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[[ 7.064613  ,  4.4742312 , -0.12700312, -0.71483076],
        [-0.59317935,  4.0224333 ,  0.5515773 ,  6.009665  ],
        [-5.193879  , -1.0297644 , -4.388829  ,  2.3485358 ],
        [-0.8724199 , -1.5610907 ,  0.47096923,  0.10478485]],

       [[ 0.7020009 ,  0.0506321 ,  2.5788887 , -0.44895908],
        [ 2.617715  , -1.5580447 , -0.9952533 ,  1.3444504 ],
        [ 0.7077899 , -0.9494638 ,  2.3607216 , -4.8069396 ],
        [-0.9731158 ,  2.762172  , -3.6125846 , -0.59783787]]],
      dtype=float32),),
    expected_outputs=(array([[[ 0.77634734 ,  0.18322006 , -0.54734755 , -0.25323078 ],
        [ 0.025877645,  0.933256   ,  0.35778686 , -0.018767282],
        [-0.6168493  ,  0.29107252 , -0.7210247  , -0.12205473 ],
        [-0.12693314 , -0.103636496,  0.22917864 , -0.9594922  ]],

       [[-0.35366225 ,  0.12650877 ,  0.30348217 ,  0.875681   ],
        [ 0.08955019 ,  0.562192   , -0.7897132  ,  0.22863586 ],
        [-0.76850456 , -0.45641905 , -0.43880737 , -0.09236207 ],
        [ 0.52564687 , -0.6779514  , -0.30281997 ,  0.4151838  ]]],
      dtype=float32), array([[10.327011  ,  7.6591544 ,  3.7366831 ,  0.61586106],
       [ 6.391762  ,  4.59397   ,  2.8317587 ,  1.2306359 ]],
      dtype=float32), array([[[ 0.85056764,  0.42713335,  0.24819753, -0.18024908],
        [-0.08885985,  0.5791475 , -0.10899168,  0.8030026 ],
        [-0.14292239, -0.16727926,  0.9471644 ,  0.23338945],
        [-0.4982086 ,  0.6739162 ,  0.17146008, -0.51790595]],

       [[-0.16729498,  0.31668338, -0.73756653,  0.5724681 ],
        [ 0.41296428, -0.50256777,  0.2478048 ,  0.7179689 ],
        [-0.66040397,  0.2916794 ,  0.574439  ,  0.38575917],
        [ 0.6044336 ,  0.74970704,  0.2541817 ,  0.08939341]]],
      dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf32> loc("operand")) -> (tensor<2x4x4xf32> {jax.result_info = "result[0]"}, tensor<2x4xf32> {jax.result_info = "result[1]"}, tensor<2x4x4xf32> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %0:5 = stablehlo.custom_call @oneapisolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare EQ, %0#4, %1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc6)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc6)
    return %10, %6, %14 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32> loc(#loc5)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":750:13)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":783:4)
#loc4 = loc("jit(func)"(#loc2))
#loc5 = loc("jit(func)"(#loc3))
#loc6 = loc("svd"(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xeb\xa13\x01-\x0f\x07\x0f\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x17#\x0b\x0b\x0b\x0f\x17\x03So\x0f\x0b/\x0b\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f#\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02B\x07\x1d\x1b\x1d\x1f\x11\x03\x05\x05\x17\x05\x19\x03\x07\r\x0f\x11\x05\x13\x05\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x1d\x19\x03\x05#\x05%\x1d\x07\x1f\x17\t\xba\x0b\x1b\x03\x07#W%_'\x81\x05'\x05)\x05+\x1d\x07+\x17\t>\x0c\t\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f%\x01\x1d-\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f11\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x03=\r\x01#\x1d\x03\x07CGK\r\x031E\x1d/\r\x031I\x1d1\r\x031M\x1d3\x1d5\x1d7\x1f\x11\t\x00\x00\xc0\x7f\x1f\x13\t\x00\x00\x00\x00\r\x07Y5[5]7\x1d9\x1d;\x1d=\r\x03ac\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x03\x03-\x03\x03q\x15\x03\x01\x01\x01\x03\x0b-u--w\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x83\x0b\x89\x8f\x93\x99\x9f\x01\x01\x01\x01\x01\x13\x07\x7f\x85\x87\x11\x03\x05\x11\x03\t\x13\x07\x7f\x8b\x8d\x11\x03\r\x11\x03\x11\x13\x05\x7f\x91\x11\x03\x15\x13\x07\x7f\x95\x97\x11\x03\x19\x11\x03\x1d\x13\x07\x7f\x9b\x9d\x11\x03!\x11\x03%\x13\x03\x7f\x01\t\x01\x02\x02)\x07\t\x11\x11\r\x01)\x05\t\x11\r\x1d\t\x13)\x01\r)\x01\x15\x1b)\x03\t\x15)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xc2\x02\x05\x01Q\x03\x0b\x01\x07\x04\x9a\x02\x03\x01\x05\tP\x03\x03\x07\x04n\x02\x03-K\x03\x0b\x17\x00\x07B\x03\x05\x03\x11\x07B\x01\x07\x03\x13\x0bG\x01!\t\x0b\x05\t\x05\x05\x17\x03\x01\x03F\x01\x0b\x03\x17\x03\x05\rF\x01\r\x03'\x05\x0f\x11\x03F\x01\x0f\x03)\x03\x13\x03F\x01\x0b\x03\t\x03\x03\x03F\x01\x11\x03-\x03\x15\x05\x06\x01\x03\t\x07\x19\t\x17\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03\x1d\x05\x06\x01\x03\x05\x07!\x0b\x1f\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03%\x05\x06\x01\x03\x05\x07)\r'\x0f\x04)\x07#\x1b+\x06\x03\x01\x05\x01\x00\xd2\x07G/\x03\x05\x1f\x17\x1d\x17\x0f\x0b\x15\x15\x15!%3)\t\x11\x13%)9w\x15\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jit(func)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00num_batch_dims\x001\x00\x00oneapisolver_gesvd_ffi\x00\x08E\x15\x05+\x01\x0b;?AOQ\x03S\x03U\x11egik7mos\x03/\x05y{\x033\x03}\x039",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["qr"]["f64"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_gesvd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[[ 1.6666730531514784  , -0.283751211500066   ,
          3.0028872858127387  , -3.539066961520449   ],
        [-3.5009517179281326  , -3.0927716333012025  ,
          0.7807364288380038  , -1.7085927853808216  ],
        [ 1.6481964138894265  , -1.0457448512148775  ,
          6.119638350643893   ,  0.6798789946663015  ],
        [ 0.45137958693199876 , -3.0487560288436093  ,
         -3.5653048640383225  ,  3.1078238891060703  ]],

       [[-0.18274271852634477 , -4.107847311422953   ,
         -6.970910660834766   ,  0.026925818090434366],
        [ 0.609826504319294   , -0.08188345529211022 ,
          0.4988730098060886  ,  0.5371129476436916  ],
        [-1.8478868672212718  ,  5.777430685108351   ,
          3.6156805021156426  , -5.4328316768756855  ],
        [ 2.839181461365762   , -5.931652277530413   ,
         -6.898420189145518   ,  0.33823029148249784 ]]]),),
    expected_outputs=(array([[[ 0.5185598212776659  , -0.08640745162480423 ,
         -0.22604389271595624 , -0.8200814731634891  ],
        [ 0.05878362684886137 , -0.9889125760720033  ,
          0.040834454253227126,  0.1301112963849362  ],
        [ 0.6559139736574807  ,  0.09584764795840565 ,
          0.719753796832644   ,  0.20626332559785526 ],
        [-0.5453595659120882  , -0.07347719082261084 ,
          0.655126841044275   , -0.5176802762713153  ]],

       [[-0.5266925395219694  , -0.49814307967604077 ,
         -0.6468374971059552  ,  0.2367481643444627  ],
        [ 0.004398828045962893,  0.15543570506880658 ,
          0.22846771664609686 ,  0.9610530132891236  ],
        [ 0.5473983592728419  , -0.8074523783002774  ,
          0.20514413534806336 ,  0.07931946025360598 ],
        [-0.6503311890022831  , -0.27516153535307064 ,
          0.6980828307017144  , -0.118472931729151   ]]]), array([[ 8.364583718088925  ,  5.052514677781006  ,  4.631967246026684  ,
         2.5284502274249276 ],
       [14.3186970731309    ,  5.219146075652052  ,  2.1121068338425095 ,
         0.15087576902550018]]), array([[[ 0.1785363115620261 ,  0.07744596914313621,
          0.9039781164136    , -0.3807236167650057 ],
        [ 0.6814293626119453 ,  0.6346900565025612 ,
         -0.03622560141446908,  0.36264343610385835],
        [ 0.20775315131153402, -0.607118314415704  ,
          0.30699755938819173,  0.7028502535752937 ],
        [-0.6786880265219297 ,  0.47178173591321865,
          0.2954044166514324 ,  0.4791041504070956 ]],

       [[-0.1926856006716647 ,  0.6413507353717308 ,
          0.7081088748435267 , -0.22388236844328566],
        [ 0.1718035739905711 , -0.19146225969939268,
          0.48451291001402774,  0.8361058396546506 ],
        [ 0.8808414232069269 , -0.15017073841723122,
          0.25997252143926997, -0.3660347313883318 ],
        [ 0.396830163194108  ,  0.7276401491617723 ,
         -0.4429936224079105 ,  0.3417927521365675 ]]])),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf64> loc("operand")) -> (tensor<2x4x4xf64> {jax.result_info = "result[0]"}, tensor<2x4xf64> {jax.result_info = "result[1]"}, tensor<2x4x4xf64> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %0:5 = stablehlo.custom_call @oneapisolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64>, tensor<2x4x4xf64>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare EQ, %0#4, %1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc6)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc6)
    return %10, %6, %14 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64> loc(#loc5)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":750:13)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":783:4)
#loc4 = loc("jit(func)"(#loc2))
#loc5 = loc("jit(func)"(#loc3))
#loc6 = loc("svd"(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xeb\xa13\x01-\x0f\x07\x0f\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x17#\x0b\x0b\x0b\x0f\x17\x03So\x0f\x0b/\x0b\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f#\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02R\x07\x1d\x1b\x1d\x1f\x11\x03\x05\x05\x17\x05\x19\x03\x07\r\x0f\x11\x05\x13\x05\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x1d\x19\x03\x05#\x05%\x1d\x07\x1f\x17\t\xba\x0b\x1b\x03\x07#W%_'\x81\x05'\x05)\x05+\x1d\x07+\x17\t>\x0c\t\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f%\x01\x1d-\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f11\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x03=\r\x01#\x1d\x03\x07CGK\r\x031E\x1d/\r\x031I\x1d1\r\x031M\x1d3\x1d5\x1d7\x1f\x11\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x13\t\x00\x00\x00\x00\r\x07Y5[5]7\x1d9\x1d;\x1d=\r\x03ac\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x03\x03-\x03\x03q\x15\x03\x01\x01\x01\x03\x0b-u--w\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x83\x0b\x89\x8f\x93\x99\x9f\x01\x01\x01\x01\x01\x13\x07\x7f\x85\x87\x11\x03\x05\x11\x03\t\x13\x07\x7f\x8b\x8d\x11\x03\r\x11\x03\x11\x13\x05\x7f\x91\x11\x03\x15\x13\x07\x7f\x95\x97\x11\x03\x19\x11\x03\x1d\x13\x07\x7f\x9b\x9d\x11\x03!\x11\x03%\x13\x03\x7f\x01\t\x01\x02\x02)\x07\t\x11\x11\r\x01)\x05\t\x11\r\x1d\x0b\x13)\x01\r)\x01\x15\x1b)\x03\t\x15)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xc2\x02\x05\x01Q\x03\x0b\x01\x07\x04\x9a\x02\x03\x01\x05\tP\x03\x03\x07\x04n\x02\x03-K\x03\x0b\x17\x00\x07B\x03\x05\x03\x11\x07B\x01\x07\x03\x13\x0bG\x01!\t\x0b\x05\t\x05\x05\x17\x03\x01\x03F\x01\x0b\x03\x17\x03\x05\rF\x01\r\x03'\x05\x0f\x11\x03F\x01\x0f\x03)\x03\x13\x03F\x01\x0b\x03\t\x03\x03\x03F\x01\x11\x03-\x03\x15\x05\x06\x01\x03\t\x07\x19\t\x17\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03\x1d\x05\x06\x01\x03\x05\x07!\x0b\x1f\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03%\x05\x06\x01\x03\x05\x07)\r'\x0f\x04)\x07#\x1b+\x06\x03\x01\x05\x01\x00\xd2\x07G/\x03\x05\x1f\x17\x1d\x17\x0f\x0b\x15\x15\x15!%3)\t\x11\x13%)9w\x15\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jit(func)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00num_batch_dims\x001\x00\x00oneapisolver_gesvd_ffi\x00\x08E\x15\x05+\x01\x0b;?AOQ\x03S\x03U\x11egik7mos\x03/\x05y{\x033\x03}\x039",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["qr"]["c64"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_gesvd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[[ 1.0732381 +3.5808065j ,  3.9696057 +0.20698753j,
         -0.6425436 +0.5669031j ,  2.2608232 +3.3495777j ],
        [-0.14261106+1.9452835j ,  0.7328605 -3.497075j  ,
         -0.2833068 +2.5005085j , -5.3383408 -0.13752732j],
        [ 0.378204  +0.31973448j, -0.82705253-1.2925721j ,
          4.6363106 -0.6026158j , -4.2700663 +0.3752707j ],
        [-0.5445609 -2.3843932j , -1.5469118 -0.22753051j,
         -1.2669541 -5.0028024j ,  0.03133653-3.4463193j ]],

       [[ 2.5795584 +0.6289093j ,  1.27082   +3.8879561j ,
          1.9604253 +1.1865004j , -2.399359  +1.3273407j ],
        [-1.4925842 +5.878235j  , -5.6121607 -3.4182298j ,
         -1.9998045 +0.10950515j,  1.9120374 +3.3194423j ],
        [ 0.40499273-2.4316337j ,  1.6648822 +2.6184802j ,
         -4.0471325 +3.133329j  , -0.76832575-1.7445682j ],
        [-1.895143  +0.5848787j ,  2.6240315 +5.021989j  ,
          1.449456  +3.472618j  , -2.0201976 -3.5186274j ]]],
      dtype=complex64),),
    expected_outputs=(array([[[-0.25113747  -0.21140064j ,  0.01894782  -0.6437263j  ,
         -0.07004365  -0.54339385j ,  0.21103057  +0.36439025j ],
        [ 0.34577915  -0.55255127j , -0.21863882  +0.16678019j ,
         -0.4815038   -0.24859615j , -0.4531385   +0.022904405j],
        [-0.07727687  -0.37749267j , -0.10726076  +0.562082j   ,
          0.30505404  -0.37139323j ,  0.5355752   -0.07908649j ],
        [ 0.14142954  +0.54670703j ,  0.13847762  +0.40375927j ,
         -0.24584758  -0.33873165j , -0.0011078244+0.5689727j  ]],

       [[-0.15231484  +0.2637356j  , -0.33362192  -0.39039814j ,
         -0.38095388  -0.17617397j , -0.4020869   -0.5528943j  ],
        [ 0.08597733  -0.6945266j  ,  0.21902993  -0.44538897j ,
         -0.33454543  -0.015148772j, -0.2825511   +0.2681601j  ],
        [-0.27464592  +0.24423113j ,  0.34782344  +0.32215017j ,
         -0.7400092   -0.16378608j ,  0.1559686   +0.20345305j ],
        [-0.2525332   +0.46758106j ,  0.37721652  -0.35055062j ,
          0.3376374   -0.15247309j , -0.3850827   +0.4085106j  ]]],
      dtype=complex64), array([[ 9.668089  ,  8.574803  ,  4.5494914 ,  0.57807904],
       [12.876014  ,  7.765103  ,  4.0119534 ,  2.2829204 ]],
      dtype=float32), array([[[-0.38075796  -0.j          ,  0.14001986  +0.06041839j  ,
         -0.46370548  +0.22876573j  , -0.48997858  -0.5695012j   ],
        [-0.32981366  -0.j          , -0.20355025  +0.5129233j   ,
         -0.34164208  -0.42274082j  , -0.19677912  +0.50254226j  ],
        [-0.32920375  +0.j          ,  0.1782829   +0.62404454j  ,
          0.63654786  +0.1484865j   ,  0.07557414  -0.19353434j  ],
        [ 0.79866844  +0.j          ,  0.056182798 +0.49784335j  ,
         -0.09977101  -0.0043059653j, -0.28370285  -0.14375064j  ]],

       [[-0.34102196  -0.j          ,  0.35656646  -0.67877984j  ,
          0.22528777  -0.27213925j  , -0.21556842  +0.35289997j  ],
        [-0.72291887  +0.j          , -0.1283458   -0.110830486j ,
         -0.3442187   +0.47835374j  , -0.14619401  -0.2827561j   ],
        [-0.32744166  +0.j          , -0.1945242   +0.057822432j ,
          0.5366797   -0.43909326j  ,  0.17421928  -0.5834541j   ],
        [ 0.5038595   -0.j          , -0.06922946  -0.5808498j   ,
          0.0073774573+0.21678263j  , -0.2424352   -0.5460057j   ]]],
      dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f32>> loc("operand")) -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<2x4xf32> {jax.result_info = "result[1]"}, tensor<2x4x4xcomplex<f32>> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc6)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %0:5 = stablehlo.custom_call @oneapisolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>>, tensor<2x4x4xcomplex<f32>>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare EQ, %0#4, %1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc6)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc6)
    return %10, %6, %14 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>> loc(#loc5)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":750:13)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":783:4)
#loc4 = loc("jit(func)"(#loc2))
#loc5 = loc("jit(func)"(#loc3))
#loc6 = loc("svd"(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xf1\xa37\x01-\x0f\x07\x0f\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x17#\x0b\x0b\x0b\x0f\x17\x03Uo\x0f\x0b/\x0b\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f\x1f#\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x033\x1b\x07\x17\x07\x07\x07\x0b\x0f\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\x82\x07\x1d\x1b\x1d\x1f\x11\x03\x05\x05\x17\x05\x19\x03\x07\r\x0f\x11\x05\x13\x05\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x1d\x19\x03\x05#\x05%\x1d\x07\x1f\x17\t\xba\x0b\x1b\x03\x07#Y%a'\x83\x05'\x05)\x05+\x1d\x07+\x17\t>\x0c\t\x1f#1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f)\x01\x1d-\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f51\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x03=\r\x01#!\x03\x07CGK\r\x031E\x1d/\r\x031I\x1d1\r\x031M\x1d3\x1d5\x1d7\x1f\x13\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x15\t\x00\x00\xc0\x7f\x1f\x17\t\x00\x00\x00\x00\r\x07[5]5_7\x1d9\x1d;\x1d=\r\x03ce\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x03\x03-\x03\x03s\x15\x03\x01\x01\x01\x03\x0b-w--y\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x85\x0b\x8b\x91\x95\x9b\xa1\x01\x01\x01\x01\x01\x13\x07\x81\x87\x89\x11\x03\x05\x11\x03\t\x13\x07\x81\x8d\x8f\x11\x03\r\x11\x03\x11\x13\x05\x81\x93\x11\x03\x15\x13\x07\x81\x97\x99\x11\x03\x19\x11\x03\x1d\x13\x07\x81\x9d\x9f\x11\x03!\x11\x03%\x13\x03\x81\x01\t\x01\x02\x02)\x07\t\x11\x11\x11\x01)\x05\t\x11\r\x1d\t\x13\x03\r)\x01\x11)\x01\r)\x01\x19\x1b)\x03\t\x19)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xda\x02\x05\x01Q\x03\x0b\x01\x07\x04\xb2\x02\x03\x01\x05\tP\x03\x03\x07\x04\x86\x02\x03/O\x03\x0b\x17\x00\x05B\x03\x05\x03\x13\x05B\x01\x07\x03\x15\x05B\x01\t\x03\x17\x0bG\x01!\x0b\x0b\x05\t\x05\x05\x1b\x03\x01\x03F\x01\r\x03\x1b\x03\x07\rF\x01\x0f\x03+\x05\x11\x13\x03F\x01\x11\x03-\x03\x15\x03F\x01\r\x03\t\x03\x05\x03F\x01\x13\x031\x03\x17\x07\x06\x01\x03\t\x07\x1b\x0b\x19\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03\x1f\x07\x06\x01\x03\x05\x07#\r!\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03'\x07\x06\x01\x03\x05\x07+\x0f)\x0f\x04)\x07%\x1d-\x06\x03\x01\x05\x01\x00\xd2\x07G/\x03\x05\x1f\x17\x1d\x17\x0f\x0b\x15\x15\x15!%3)\t\x11\x13%)9w\x15\x15\x17\x1f\x11\x15\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jit(func)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00num_batch_dims\x001\x00\x00oneapisolver_gesvd_ffi\x00\x08I\x17\x05+\x01\x0b;?AOQ\x03S\x03U\x03W\x11gikm7oqu\x03/\x05{}\x033\x03\x7f\x039",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_08_13["qr"]["c128"] = dict(
    testdata_version=1,
    platform='oneapi',
    custom_call_targets=['oneapisolver_gesvd_ffi'],
    serialized_date=datetime.date(2026, 8, 13),
    inputs=(array([[[-4.699732537674587  -0.18164091805215468j,
         -2.6529672987457267 +2.1873545441571416j ,
         -4.583623180305504  +0.05141217967419283j,
         -1.4684446379730842 +0.5956695859134695j ],
        [ 2.217429580673316  -1.6820541069935535j ,
         -1.489637886109648  -1.1907523648513954j ,
         -5.37070728884717   +0.3011497067658051j ,
         -3.5377553884933244 +1.560799473477663j  ],
        [ 0.4865985561509131 +4.547548126143047j  ,
          1.9744285723487844 +1.579347193702052j  ,
          3.662108610237921  -3.8947365367486944j ,
         -0.46900368026456773+3.897268760016375j  ],
        [-3.9057171822032837 +0.894017787659835j  ,
         -2.665956542656175  -5.446062606216615j  ,
          6.586068520522582  +7.82920032979931j   ,
          0.2438426632437082 -2.5324000439269967j ]],

       [[ 2.3593407739528036 +0.1518669531658939j ,
          0.6163481796609258 +2.2151855304705617j ,
          1.1710769743888314 -6.27345033430341j   ,
          0.9738490103384626 +0.5395897278168652j ],
        [-2.4788654273898656 +0.4265527313512031j ,
         -1.1807578044484868 -0.0496832499163036j ,
         -4.4976038167764765 +1.058853052811918j  ,
         -1.1727797045618331 -5.283007446632174j  ],
        [ 2.1607883932036422 +0.15328185326939148j,
          0.33959787374719413-0.44019437888510504j,
          5.554548585416958  -5.5054723821239575j ,
          3.6501512907075853 +2.5205805340930167j ],
        [ 1.3385284474824868 -5.630140770855095j  ,
         -0.27414990799969   -0.46452124262376304j,
          1.611578799750626  +8.022764935423794j  ,
         -2.616414597337455  +0.02175053549931295j]]]),),
    expected_outputs=(array([[[ 0.010550471640845436-0.17188852445378558j ,
          0.7192461888739362  +0.05423301579908514j ,
         -0.3986393309169211  +0.09925506937224644j ,
          0.27045040059197656 -0.4562657321852802j  ],
        [ 0.1757158215796514  -0.28898168784945594j ,
          0.2705184461915346  +0.2220238094575259j  ,
          0.8080741699984128  +0.31938346887050856j ,
          0.08837920732925568 -0.01838977280512092j ],
        [ 0.1855980910408694  +0.3987655821409668j  ,
         -0.03564991039731414 -0.49144324300371256j ,
          0.1541338867426699  +0.04068867845033524j ,
          0.7321684611544167  +0.04762880211338243j ],
        [-0.802462438531313   +0.13619820370204255j ,
         -0.2338302036229407  +0.2445505198523449j  ,
          0.12332909849753138 +0.18873939840686943j ,
          0.2662520876834143  -0.31827623525067117j ]],

       [[-0.1937381119749961  -0.36780548492823056j ,
          0.26369065783528023 +0.0797383023340004j  ,
         -0.2284250488043049  +0.7133040647598472j  ,
         -0.35394486204017217 +0.25502167015286253j ],
        [-0.24171920617553377 +0.33116818551784577j ,
         -0.5163682164657232  +0.13798196606189456j ,
         -0.4596893341830245  +0.040918917485529595j,
          0.20304338982129225 +0.5403786083964175j  ],
        [ 0.027550609791235487-0.5776868042861197j  ,
          0.2741698942833105  -0.15715672418567253j ,
         -0.20656327152309523 -0.2247313731964578j  ,
          0.6309670599558868  +0.2726894702317326j  ],
        [ 0.40314842667074957 +0.402584641558122j   ,
          0.22887811112799986 -0.6972670403334555j  ,
         -0.29285414368652796 +0.21701276872698252j ,
          0.03733951130463514 +0.050775060769343905j]]]), array([[15.105031148122244 ,  9.104919912640343 ,  5.006211740104107 ,
         3.446376589720919 ],
       [15.343823952995173 ,  7.375371564687318 ,  3.7496815109995794,
         0.8625145657311305]]), array([[[ 0.3983461020993688   +0.j                  ,
          0.13718654285554352  +0.20963212142757814j ,
         -0.40914166828555454  -0.771218371268325j   ,
         -0.017483980342966714 +0.12678422058548103j ],
        [-0.4705168176319192   -0.j                  ,
         -0.4406259159823648   +0.501395999171419j   ,
         -0.27697957785810823  +0.006226163227350235j,
         -0.46230452584693427  +0.20635612960990404j ],
        [ 0.6106772453975172   +0.j                  ,
         -0.25916803915444236  -0.21982460883707208j ,
          0.05682612321797866  +0.27292487464860143j ,
         -0.41495999423821106  +0.5115402022168485j  ],
        [-0.49699860082446     +0.j                  ,
          0.2086557264813752   -0.5767641952667593j  ,
          0.0041166738594976444-0.28867754492885345j ,
         -0.08621603219998744  +0.534802174159026j   ]],

       [[-0.09961676341576108  -0.j                  ,
         -0.0455617064630741   +0.02005499517289743j ,
          0.6993928144861428   +0.5554242311281647j  ,
         -0.2772976000516558   +0.3362411099877399j  ],
        [ 0.9183964339371261   +0.j                  ,
          0.18513592789573782  +0.04864336760149432j ,
         -0.07592192157413644  +0.08808182166509267j ,
          0.02265332870749621  +0.3253779069593686j  ],
        [-0.36489390866852983  -0.j                  ,
          0.5302626885445343   -0.1364692023035999j  ,
         -0.3393839629266361   -0.005000573360891503j,
         -0.01709833277905621  +0.6719756248311357j  ],
        [ 0.11609016321265404  +0.j                  ,
          0.16300036810869822  -0.7965607051693728j  ,
          0.13402108640742136  -0.2359299417410548j  ,
         -0.4709038361949725   -0.1734069924393353j  ]]])),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f64>> loc("operand")) -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<2x4xf64> {jax.result_info = "result[1]"}, tensor<2x4x4xcomplex<f64>> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc6)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %0:5 = stablehlo.custom_call @oneapisolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>>, tensor<2x4x4xcomplex<f64>>, tensor<2xi32>) loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc6)
    %2 = stablehlo.compare EQ, %0#4, %1, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc6)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc6)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc6)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc6)
    return %10, %6, %14 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>> loc(#loc5)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":750:13)
#loc3 = loc("/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py":783:4)
#loc4 = loc("jit(func)"(#loc2))
#loc5 = loc("jit(func)"(#loc3))
#loc6 = loc("svd"(#loc4))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.16.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xf1\xa37\x01-\x0f\x07\x0f\x0b\x0b#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x17#\x0b\x0b\x0b\x0f\x17\x03Uo\x0f\x0b/\x0b\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0bO/\x1f#\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x033\x1b\x07\x17\x07\x07\x07\x0b\x0f\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\xb2\x07\x1d\x1b\x1d\x1f\x11\x03\x05\x05\x17\x05\x19\x03\x07\r\x0f\x11\x05\x13\x05\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x1d\x19\x03\x05#\x05%\x1d\x07\x1f\x17\t\xba\x0b\x1b\x03\x07#Y%a'\x83\x05'\x05)\x05+\x1d\x07+\x17\t>\x0c\t\x1f#1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f)\x01\x1d-\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f51\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x03=\r\x01#!\x03\x07CGK\r\x031E\x1d/\r\x031I\x1d1\r\x031M\x1d3\x1d5\x1d7\x1f\x13!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x15\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x17\t\x00\x00\x00\x00\r\x07[5]5_7\x1d9\x1d;\x1d=\r\x03ce\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x03\x03-\x03\x03s\x15\x03\x01\x01\x01\x03\x0b-w--y\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x85\x0b\x8b\x91\x95\x9b\xa1\x01\x01\x01\x01\x01\x13\x07\x81\x87\x89\x11\x03\x05\x11\x03\t\x13\x07\x81\x8d\x8f\x11\x03\r\x11\x03\x11\x13\x05\x81\x93\x11\x03\x15\x13\x07\x81\x97\x99\x11\x03\x19\x11\x03\x1d\x13\x07\x81\x9d\x9f\x11\x03!\x11\x03%\x13\x03\x81\x01\t\x01\x02\x02)\x07\t\x11\x11\x11\x01)\x05\t\x11\r\x1d\x0b\x13\x03\r)\x01\x11)\x01\r)\x01\x19\x1b)\x03\t\x19)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xda\x02\x05\x01Q\x03\x0b\x01\x07\x04\xb2\x02\x03\x01\x05\tP\x03\x03\x07\x04\x86\x02\x03/O\x03\x0b\x17\x00\x05B\x03\x05\x03\x13\x05B\x01\x07\x03\x15\x05B\x01\t\x03\x17\x0bG\x01!\x0b\x0b\x05\t\x05\x05\x1b\x03\x01\x03F\x01\r\x03\x1b\x03\x07\rF\x01\x0f\x03+\x05\x11\x13\x03F\x01\x11\x03-\x03\x15\x03F\x01\r\x03\t\x03\x05\x03F\x01\x13\x031\x03\x17\x07\x06\x01\x03\t\x07\x1b\x0b\x19\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03\x1f\x07\x06\x01\x03\x05\x07#\r!\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03'\x07\x06\x01\x03\x05\x07+\x0f)\x0f\x04)\x07%\x1d-\x06\x03\x01\x05\x01\x00\xd2\x07G/\x03\x05\x1f\x17\x1d\x17\x0f\x0b\x15\x15\x15!%3)\t\x11\x13%)9w\x15\x15\x17\x1f\x11\x15\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jit(func)\x00/localdisk/oneapi-jax/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00num_batch_dims\x001\x00\x00oneapisolver_gesvd_ffi\x00\x08I\x17\x05+\x01\x0b;?AOQ\x03S\x03U\x03W\x11gikm7oqu\x03/\x05{}\x033\x03\x7f\x039",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
