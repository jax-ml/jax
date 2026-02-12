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

# type: ignore
# ruff: noqa

import datetime
import numpy as np

array = np.array
float32 = np.float32
complex64 = np.complex64

data_2026_02_04 = {"jacobi": {}, "qr": {}}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["jacobi"]["f32"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_gesvdj_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[ 3.7223358 , -1.2193766 ,  1.795808  ,  1.0418197 ],
        [ 4.6901207 ,  0.11149647,  3.8279397 ,  8.07944   ],
        [-3.763075  ,  9.478659  ,  0.14772141,  0.86707467],
        [ 3.7043862 , -5.310245  ,  7.1758327 ,  3.8767374 ]],

       [[-3.2586122 ,  8.368001  ,  3.8854764 , -9.429428  ],
        [ 1.1301196 ,  9.584712  , -0.7165561 , -3.439898  ],
        [ 4.4551563 , -5.166789  ,  1.8632321 ,  4.3077874 ],
        [-1.3375099 , -1.1678746 ,  7.680391  ,  1.983631  ]]],
      dtype=float32),),
    expected_outputs=(array([[[-0.26312542 , -0.02221384 ,  0.19764648 , -0.94403785 ],
        [-0.5306547  , -0.6105559  ,  0.5214799  ,  0.27145126 ],
        [ 0.44124788 , -0.7911463  , -0.3813815  , -0.18421714 ],
        [-0.6741445  , -0.028558023, -0.73725355 ,  0.0342183  ]],

       [[-0.741788   ,  0.3681162  ,  0.2151611  , -0.51763594 ],
        [-0.51899517 , -0.25662977 , -0.7801523  ,  0.2369551  ],
        [ 0.41801134 ,  0.17694637 , -0.549365   , -0.70153725 ],
        [ 0.07524119 ,  0.87596905 , -0.20800538 ,  0.42866164 ]]],
      dtype=float32), array([[14.886929 ,  9.745339 ,  3.74923  ,  2.0411706],
       [17.598677 ,  8.969212 ,  5.1799684,  2.7928188]], dtype=float32), array([[[-0.5122628   ,  0.53899616  , -0.48876467  , -0.4562667   ],
        [-0.007687226 , -0.75814134  , -0.27693856  , -0.59031165  ],
        [ 0.5029314   ,  0.03124599  , -0.79899544  ,  0.32816154  ],
        [-0.69612336  , -0.3656894   , -0.2145239   ,  0.57936424  ]],

       [[ 0.20412575  , -0.76308864  , -0.06554903  ,  0.60969806  ],
        [-0.20881027  , -0.14679018  ,  0.9668265   , -0.0098668765],
        [-0.72434616  , -0.5011017   , -0.2367066   , -0.41010973  ],
        [-0.6245429   ,  0.38084862  , -0.07014099  ,  0.6782189   ]]],
      dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf32> loc("operand")) -> (tensor<2x4x4xf32> {jax.result_info = "result[0]"}, tensor<2x4xf32> {jax.result_info = "result[1]"}, tensor<2x4x4xf32> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_gesvdj_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.transpose %0#3, dims = [0, 2, 1] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32> loc(#loc4)
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %3 = stablehlo.compare  EQ, %0#4, %2,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc4)
    %6 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %7 = stablehlo.select %6, %0#1, %5 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc4)
    %10 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %11 = stablehlo.select %10, %0#2, %9 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc4)
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %15 = stablehlo.select %14, %1, %13 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc4)
    return %11, %7, %15 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":621:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("svd"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01%\x07\x01\x05\t\x13\x01\x03\x0f\x03\x11\x13\x17\x1b\x1f#'+/\x03\xe7\x9d3\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03So\x0f\x0b/\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x1b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/o\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x0f\x0f\x07\x13\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x02~\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x19\x11\x01\x00\x05\x1b\x05\x1d\x05\x1f\x1d\x15\x03\x05!\x05#\x1d\x1b\x1d\x05%\x17\x1f\xb6\t\x1b\x05'\x03\x07#Q%W'}\x05)\x05+\x05-\x1f!1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x01\x1d/\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x1f\x191\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x037\r\x01#\x1f\x03\x07=AE\r\x03-?\x1d1\r\x03-C\x1d3\r\x03-G\x1d5\x1d7\x1d9\x1f\x11\t\x00\x00\xc0\x7f\x1f\x13\t\x00\x00\x00\x00\r\x05S1U1\x1d;\x1d=\r\x03Y[\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x05\x01\x03\x03)\x03\x03k\x15\x03\x01\x01\x01\x03\x0b)o))q\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x191\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x7f\x0b\x85\x8b\x8f\x95\x9b\x01\x01\x01\x01\x01\x13\x07{\x81\x83\x11\x03\x05\x11\x03\t\x13\x07{\x87\x89\x11\x03\r\x11\x03\x11\x13\x05{\x8d\x11\x03\x15\x13\x07{\x91\x93\x11\x03\x19\x11\x03\x1d\x13\x07{\x97\x99\x11\x03!\x11\x03%\x13\x03{\x01\t\x01\x02\x02)\x07\t\x11\x11\r\x01)\x05\t\x11\r\x1d\t\x13)\x01\r)\x01\x15\x1b)\x03\t\x15)\x03\r\x0b)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b\x04\xe2\x02\x05\x01Q\x03\x07\x01\x07\x04\xba\x02\x03\x01\x05\tP\x03\x03\x07\x04\x8e\x02\x03/O\x03\x0b\x13\x00\x07B\x03\x05\x03\x11\x07B\x01\x07\x03\x13\x0bG\x01!\t\x0b\x05\t\x05\x05\x17\x03\x01\rF\x01\x0b\x03\x05\x03\r\x03F\x01\r\x03\x17\x03\x05\x0fF\x01\x0f\x03)\x05\x0f\x13\x03F\x01\x11\x03+\x03\x15\x03F\x01\r\x03\t\x03\x03\x03F\x01\x13\x03/\x03\x17\x05\x06\x01\x03\t\x07\x1b\t\x19\x03F\x01\x11\x03\x1b\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1d\x03\x1f\x05\x06\x01\x03\x05\x07#\x0b!\x03F\x01\x11\x03\x1b\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1d\x03'\x05\x06\x01\x03\x05\x07+\x11)\x11\x04\x03\x07%\x1d-\x06\x03\x01\x05\x01\x00\xca\x07G+\x03\x05\x1f\x1d\x17\x0f\x0b\x15\x15\x15!%3)s\x15\t\x11\x13%)9\x15\x17\x1b\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00transpose_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00num_batch_dims\x001\x00\x00hipsolver_gesvdj_ffi\x00\x08I\x17\x05#\x01\x0b59;IK\x03M\x03O\x11]_acegim\x03s\x03+\x05uw\x03/\x03y\x033",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["jacobi"]["f64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_gesvdj_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[ 6.765010292858648 , -8.41155960620905  , -4.844896569854504 ,
          9.95142460562528  ],
        [-2.3172669439509725, -2.377110651443677 , -6.845706401083533 ,
         -0.2688427639827662],
        [ 5.816161074813861 , -1.7232775955098916,  5.541301732723863 ,
          6.036680333196475 ],
        [-8.991063432111106 , -5.6496730486346936,  6.836412081386392 ,
         -0.5190364760940618]],

       [[-6.141369069758367 ,  5.548388466467886 ,  0.3686736359214944,
         -5.673263508349118 ],
        [-6.44854045236065  , -6.90827339890318  , -3.6114354632331853,
          1.485277184786895 ],
        [ 5.029310351636605 ,  3.1332466505034784, -2.6674713013275557,
         -3.478972614333551 ],
        [-6.013442048617872 , -3.3540895366085755, -2.4965517430649165,
          6.810791397528835 ]]]),),
    expected_outputs=(array([[[-0.8582826839566826  ,  0.1513382685059182  ,
          0.36013035494839635 , -0.3327967703426305  ],
        [-0.06441044954508533 , -0.29407362148116534 ,
          0.6308874214363143  ,  0.7150895472678866  ],
        [-0.3777672731992751  ,  0.4538714107552889  ,
         -0.5233373280534688  ,  0.61433758705165    ],
        [ 0.34131219683611846 ,  0.8274165674758919  ,
          0.4454270198181023  , -0.02196766198937948 ]],

       [[-0.17339573460073665 , -0.9777805849977806  ,
          0.05338819597560617 ,  0.10501784302745319 ],
        [ 0.6262340205527418  , -0.1769136278875054  ,
         -0.7175042887139369  , -0.24843533453825947 ],
        [-0.41056955000343537 ,  0.11049263814893846 ,
         -0.6153961832518882  ,  0.6637104482859019  ],
        [ 0.6396854815724643  , -0.020930188340448217,
          0.32190811398066105 ,  0.697667240190373   ]]]), array([[17.334179944404195 , 12.265833831731298 , 10.472378382078709 ,
         0.3060234958213206],
       [14.908186073428876 ,  9.911591222615328 ,  5.124458445072229 ,
         3.603674967389374 ]]), array([[[-0.6301394090884883  ,  0.35163487969387064 ,
          0.2791741088511297  , -0.6335132622474965  ],
        [-0.25227095254238124 , -0.4916685588357358  ,
          0.7705568653330496  ,  0.3175901636446802  ],
        [-0.5800330711340123  , -0.5866488459963135  ,
         -0.5651544995632785  ,  0.002271454472501866],
        [-0.4503835343435822  ,  0.5389416212599115  ,
         -0.09436273554037346 ,  0.7055439568214439  ]],

       [[-0.5959813910148998  , -0.5849296973169651  ,
         -0.18965121472926696 ,  0.5164260329537445  ],
        [ 0.7897127913564964  , -0.3820311302146965  ,
          0.003626822614449421,  0.47999246751645436 ],
        [-0.14280833301192955 ,  0.438102406219792   ,
          0.6730066572461373  ,  0.5785620977813557  ],
        [ 0.027670720161007956,  0.5656639871938048  ,
         -0.7148995049738235  ,  0.4100942362749819  ]]])),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf64> loc("operand")) -> (tensor<2x4x4xf64> {jax.result_info = "result[0]"}, tensor<2x4xf64> {jax.result_info = "result[1]"}, tensor<2x4x4xf64> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_gesvdj_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64>, tensor<2x4x4xf64>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.transpose %0#3, dims = [0, 2, 1] : (tensor<2x4x4xf64>) -> tensor<2x4x4xf64> loc(#loc4)
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %3 = stablehlo.compare  EQ, %0#4, %2,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc4)
    %6 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %7 = stablehlo.select %6, %0#1, %5 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc4)
    %10 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %11 = stablehlo.select %10, %0#2, %9 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc4)
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %15 = stablehlo.select %14, %1, %13 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc4)
    return %11, %7, %15 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":621:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("svd"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01%\x07\x01\x05\t\x13\x01\x03\x0f\x03\x11\x13\x17\x1b\x1f#'+/\x03\xe7\x9d3\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03So\x0f\x0b/\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f\x1b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/o\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x0f\x0f\x07\x13\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x02\x8e\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x19\x11\x01\x00\x05\x1b\x05\x1d\x05\x1f\x1d\x15\x03\x05!\x05#\x1d\x1b\x1d\x05%\x17\x1f\xb6\t\x1b\x05'\x03\x07#Q%W'}\x05)\x05+\x05-\x1f!1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x01\x1d/\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x1f\x191\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x037\r\x01#\x1f\x03\x07=AE\r\x03-?\x1d1\r\x03-C\x1d3\r\x03-G\x1d5\x1d7\x1d9\x1f\x11\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x13\t\x00\x00\x00\x00\r\x05S1U1\x1d;\x1d=\r\x03Y[\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x05\x01\x03\x03)\x03\x03k\x15\x03\x01\x01\x01\x03\x0b)o))q\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x191\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x7f\x0b\x85\x8b\x8f\x95\x9b\x01\x01\x01\x01\x01\x13\x07{\x81\x83\x11\x03\x05\x11\x03\t\x13\x07{\x87\x89\x11\x03\r\x11\x03\x11\x13\x05{\x8d\x11\x03\x15\x13\x07{\x91\x93\x11\x03\x19\x11\x03\x1d\x13\x07{\x97\x99\x11\x03!\x11\x03%\x13\x03{\x01\t\x01\x02\x02)\x07\t\x11\x11\r\x01)\x05\t\x11\r\x1d\x0b\x13)\x01\r)\x01\x15\x1b)\x03\t\x15)\x03\r\x0b)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b\x04\xe2\x02\x05\x01Q\x03\x07\x01\x07\x04\xba\x02\x03\x01\x05\tP\x03\x03\x07\x04\x8e\x02\x03/O\x03\x0b\x13\x00\x07B\x03\x05\x03\x11\x07B\x01\x07\x03\x13\x0bG\x01!\t\x0b\x05\t\x05\x05\x17\x03\x01\rF\x01\x0b\x03\x05\x03\r\x03F\x01\r\x03\x17\x03\x05\x0fF\x01\x0f\x03)\x05\x0f\x13\x03F\x01\x11\x03+\x03\x15\x03F\x01\r\x03\t\x03\x03\x03F\x01\x13\x03/\x03\x17\x05\x06\x01\x03\t\x07\x1b\t\x19\x03F\x01\x11\x03\x1b\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1d\x03\x1f\x05\x06\x01\x03\x05\x07#\x0b!\x03F\x01\x11\x03\x1b\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1d\x03'\x05\x06\x01\x03\x05\x07+\x11)\x11\x04\x03\x07%\x1d-\x06\x03\x01\x05\x01\x00\xca\x07G+\x03\x05\x1f\x1d\x17\x0f\x0b\x15\x15\x15!%3)s\x15\t\x11\x13%)9\x15\x17\x1b\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00transpose_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00num_batch_dims\x001\x00\x00hipsolver_gesvdj_ffi\x00\x08I\x17\x05#\x01\x0b59;IK\x03M\x03O\x11]_acegim\x03s\x03+\x05uw\x03/\x03y\x033",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["jacobi"]["c64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_gesvdj_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[-6.511656   +3.9382665j ,  6.8709564  +2.5289335j ,
         -9.411133   -9.168636j  , -7.3314643  +7.189627j  ],
        [ 6.2476263  -2.9142983j , -0.060004584+2.8443305j ,
          7.9243956  -7.338787j  ,  7.5318184  -4.295307j  ],
        [-8.464032   -5.655291j  ,  5.0108194  +1.0293938j ,
          1.2175745  +2.8760285j ,  4.345743   +0.604459j  ],
        [ 9.876885   -8.447743j  ,  8.306228   +4.8012953j ,
          7.807553   +3.5845838j ,  3.5843878  -7.8993335j ]],

       [[-9.904369   +8.589717j  ,  6.794971   +2.5694795j ,
          2.2921402  -6.3595896j ,  4.2440815  -4.766851j  ],
        [ 7.0766907  -2.7424748j , -3.7027502  +2.6230326j ,
         -5.010383   +4.308751j  ,  1.4581944  -0.17130353j],
        [ 1.99838    -8.123547j  , -2.6369176  +7.5087905j ,
         -0.5215636  -9.439996j  ,  9.993352   -3.7125206j ],
        [-0.031122532+4.9268885j , -5.443708   -0.5684887j ,
          2.8900268  +7.271322j  , -9.371372   +2.0711803j ]]],
      dtype=complex64),),
    expected_outputs=(array([[[-0.49599195 -0.3477366j   , -0.47453368 +0.21972746j  ,
         -0.35997757 -0.35410944j  , -0.2512339  +0.20374799j  ],
        [ 0.44907737 +0.002405172j , -0.24904558 -0.24744749j  ,
         -0.5188636  -0.1262432j   ,  0.5398943  +0.3137309j   ],
        [ 0.044145714-0.11281507j  ,  0.44468585 +0.40522593j  ,
         -0.076412916-0.63636005j  ,  0.3233734  -0.32864404j  ],
        [ 0.6423971  +0.06360132j  , -0.07980481 +0.484728j    ,
         -0.2172295  +0.043406248j , -0.54105425 -0.011761099j ]],

       [[-0.19801104 +0.5015972j   ,  0.2974156  -0.56987107j  ,
         -0.14941728 +0.2827743j   , -0.044859245-0.43781692j  ],
        [ 0.031362493-0.19080958j  , -0.38242713 +0.34200114j  ,
         -0.05559378 +0.4285319j   ,  0.2569481  -0.66831154j  ],
        [-0.6645802  -0.041444328j , -0.43262926 -0.0030564666j,
         -0.3622315  +0.21935806j  , -0.34698528 +0.26402578j  ],
        [ 0.44450504 -0.17558658j  ,  0.3494469  +0.11952038j  ,
         -0.5150082  +0.5161587j   , -0.24080087 +0.2134198j   ]]],
      dtype=complex64), array([[28.04699  , 15.29802  , 10.812811 ,  8.262912 ],
       [24.471525 , 17.579378 ,  7.0476294,  4.6335263]], dtype=float32), array([[[ 0.3826026   -0.45641175j  ,  0.05130297  +0.1989237j   ,
          0.5836618   +0.0010950209j,  0.2293342   -0.45808792j  ],
        [-0.51104975  -0.089208454j ,  0.05988489  -0.6154523j   ,
          0.33436182  +0.45249024j  ,  0.15088452  -0.09586868j  ],
        [-0.01766008  -0.45967096j  , -0.58548707  +0.16145816j  ,
         -0.0013295817+0.38968956j  , -0.47265458  +0.2103918j   ],
        [-0.14836499  -0.37756056j  , -0.43804413  -0.12117417j  ,
         -0.28388205  -0.33218026j  ,  0.65180457  +0.114404j    ]],

       [[ 0.21023     +0.4984393j   , -0.06341874  -0.4433479j   ,
         -0.15844418  +0.37922654j  , -0.5790327   +0.050856397j ],
        [-0.66821074  +0.044658013j ,  0.11475918  +0.11917833j  ,
          0.55912364  +0.3275715j   , -0.22622015  +0.23027979j  ],
        [ 0.3396066   -0.19580403j  ,  0.8731466   +0.013714722j ,
          0.052092    +0.071940996j , -0.09583993  +0.2582264j   ],
        [-0.31174853  +0.08945723j  , -0.01018406  +0.09680383j  ,
         -0.6347194   +0.020064345j ,  0.13741249  +0.680575j    ]]],
      dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f32>> loc("operand")) -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<2x4xf32> {jax.result_info = "result[1]"}, tensor<2x4x4xcomplex<f32>> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc4)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_gesvdj_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>>, tensor<2x4x4xcomplex<f32>>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.transpose %0#3, dims = [0, 2, 1] : (tensor<2x4x4xcomplex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc4)
    %2 = stablehlo.real %1 : (tensor<2x4x4xcomplex<f32>>) -> tensor<2x4x4xf32> loc(#loc4)
    %3 = stablehlo.imag %1 : (tensor<2x4x4xcomplex<f32>>) -> tensor<2x4x4xf32> loc(#loc4)
    %4 = stablehlo.negate %3 : tensor<2x4x4xf32> loc(#loc4)
    %5 = stablehlo.complex %2, %4 : tensor<2x4x4xcomplex<f32>> loc(#loc4)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %7 = stablehlo.compare  EQ, %0#4, %6,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc4)
    %10 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %11 = stablehlo.select %10, %0#1, %9 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc4)
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %15 = stablehlo.select %14, %0#2, %13 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc4)
    %16 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc4)
    %18 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %19 = stablehlo.select %18, %5, %17 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc4)
    return %15, %11, %19 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":621:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("svd"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01-\x07\x01\x05\t\x1b\x01\x03\x0f\x03\x19\x13\x17\x1b\x1f#'+/37;?\x03\xef\x9f9\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03Uo\x0f\x0b/\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f\x1f\x1b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/o\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x035\x1b\x07\x07\x17\x07\x07\x1b\x0b\x0f\x0f\x0f\x07\x13\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x02\xd6\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05!\x11\x01\x00\x05#\x05%\x05'\x1d\x15\x03\x05)\x05+\x1d\x1b\x1d\x05-\x17\x1f\xb6\t\x1b\x05/\x03\x07#S%Y'\x7f\x051\x053\x055\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x01\x1d7\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x1f\x1f1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x037\r\x01#%\x03\x07=AE\r\x03-?\x1d9\r\x03-C\x1d;\r\x03-G\x1d=\x1d?\x1dA\x1f\x15\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x17\t\x00\x00\xc0\x7f\x1f\x19\t\x00\x00\x00\x00\r\x05U1W1\x1dC\x1dE\r\x03[]\x1dG\x1dI\x0b\x03\x1dK\x1dM\x03\x01\x05\x01\x03\x03)\x03\x03m\x15\x03\x01\x01\x01\x03\x0b)q))s\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1f1\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f7!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x81\x0b\x87\x8d\x91\x97\x9d\x01\x01\x01\x01\x01\x13\x07}\x83\x85\x11\x03\x05\x11\x03\t\x13\x07}\x89\x8b\x11\x03\r\x11\x03\x11\x13\x05}\x8f\x11\x03\x15\x13\x07}\x93\x95\x11\x03\x19\x11\x03\x1d\x13\x07}\x99\x9b\x11\x03!\x11\x03%\x13\x03}\x01\t\x01\x02\x02)\x07\t\x11\x11\x13\x01\t)\x05\t\x11\t\x1d\x13)\x07\t\x11\x11\t\x03\t)\x01\x13)\x01\t)\x01\x1b\x1b)\x03\t\x1b)\x03\r\r)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\x0b\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\r)\x03\t\x07)\x05\t\x05\x07)\x03\x05\r)\x05\t\x11\x07)\x03\t\r\x04n\x03\x05\x01Q\x03\x07\x01\x07\x04F\x03\x03\x01\x05\tP\x03\x03\x07\x04\x1a\x03\x039c\x03\x0b\x13\x00\x05B\x03\x05\x03\x15\x05B\x01\x07\x03\x17\x05B\x01\t\x03\x19\x0bG\x01!\x0b\x0b\x05\x0b\x05\x05\x1d\x03\x01\rF\x01\r\x03\x05\x03\x0f\x0f\x06\x01\x03\x11\x03\x13\x11\x06\x01\x03\x11\x03\x13\x13\x06\x01\x03\x11\x03\x17\x15\x06\x01\x03\x05\x05\x15\x19\x03F\x01\x0f\x03\x1d\x03\x07\x17F\x01\x11\x03/\x05\x11\x1d\x03F\x01\x13\x031\x03\x1f\x03F\x01\x0f\x03\x0b\x03\x05\x03F\x01\x15\x035\x03!\x07\x06\x01\x03\x0b\x07%\x0b#\x03F\x01\x13\x03!\x03\x1f\x03F\x01\x0f\x03\x05\x03\x03\x03F\x01\x17\x03#\x03)\x07\x06\x01\x03\x05\x07-\r+\x03F\x01\x13\x03!\x03\x1f\x03F\x01\x0f\x03\x05\x03\x03\x03F\x01\x17\x03#\x031\x07\x06\x01\x03\x05\x075\x1b3\x19\x04\x03\x07/'7\x06\x03\x01\x05\x01\x00n\x08O+\x03\x05\x1f\x1d\x17\x0f\x0b\x15\x15\x15!%3)s\x15\t\x11\x13%)9\x15\x17\x17\x15\x11\x11\x1b\x1f\x11\x15\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00num_batch_dims\x001\x00\x00hipsolver_gesvdj_ffi\x00\x08M\x19\x05#\x01\x0b59;IK\x03M\x03O\x03Q\x11_acegiko\x03u\x03+\x05wy\x03/\x03{\x033",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["jacobi"]["c128"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_gesvdj_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[-4.305409063295221  +8.562910525232837j  ,
          0.4575391541476215 +0.29707312893051707j,
         -6.438158508718077  +7.685372543598643j  ,
         -4.621486740450309  +2.022385511640895j  ],
        [-3.5167752452788665 +2.745861246229973j  ,
          5.0549083156572685 -2.880319974395058j  ,
          2.5302278647207253 +9.906980376253383j  ,
          1.574596247755391  -2.7961023138146217j ],
        [-1.5784784544179953 +0.0829102322391222j ,
         -9.237548951603042  +9.879555522852257j  ,
          1.8891026618492859 +1.1401945748088593j ,
          5.394099323297649  -3.921517130110881j  ],
        [-2.7752654445789844 -5.990566343812964j  ,
         -8.878414655317396  -4.134572951945792j  ,
          5.958890395293434  -9.029222489105175j  ,
         -6.2944520404181965 +0.45060194198164893j]],

       [[-7.5785161987812755 -0.7677461987355709j ,
         -2.040365479133275  +7.603905932243634j  ,
         -3.691071372147059  +9.89712814444609j   ,
          3.4952539163073126 -7.56289549560567j   ],
        [ 1.3927209035377324 -5.2169992228732625j ,
          0.8154615431862204 -5.117885610665862j  ,
         -9.010010491559825  -4.203757768856371j  ,
          2.9119697324078473 +5.922071817981767j  ],
        [ 0.10728598047305837-2.507471821947698j  ,
         -5.0232104603833605 -5.470368656051647j  ,
         -8.321741577357662  +9.579011995819101j  ,
          9.889795999159439  +1.972451322561211j  ],
        [-7.955739588892468  +6.00000543801551j   ,
         -2.931108343578779  +6.770562873038649j  ,
         -6.313940666870044  +7.8176854249436865j ,
          9.46517759029885   -0.29194868150852216j]]]),),
    expected_outputs=(array([[[-0.3123156550421202  +0.334289656331932j  ,
          0.23030981198221195 +0.4356027470031708j ,
         -0.3877002524330693  +0.45470400435639374j,
         -0.37575438848476694 -0.22284168553736483j],
        [ 0.0787616912430086  +0.4683162862107368j ,
          0.053391106496996105+0.05898993599000597j,
         -0.506388304927506   -0.2242931168984999j ,
          0.3627677994695937  +0.5742900271411976j ],
        [-0.19117032519989163 -0.19935157910756654j,
          0.5228407641137001  -0.6623932430619892j ,
         -0.4086691211938215  -0.1222852659623192j ,
         -0.1177858097416972  -0.12549082728150862j],
        [ 0.3091438755425131  -0.6271592526880687j ,
         -0.08376759919479365 +0.17813739292718545j,
         -0.33697567172992837 +0.2016916577440831j ,
         -0.3364442790211764  +0.45268588836087237j]],

       [[-0.05334865675740175 +0.5161114058374165j ,
          0.17087006804744356 -0.2815593776541724j ,
          0.22716540797510043 +0.6924263330356862j ,
          0.19836035173275507 -0.2278277768093274j ],
        [-0.3731280907418766  -0.23027974587285818j,
         -0.17809521970913889 +0.04907999950574157j,
          0.06027805674221967 +0.36857770774438403j,
         -0.7853175228581392  -0.13195957876015255j],
        [-0.44247472043682406 +0.26036051542030303j,
         -0.35552787026936694 +0.5455376957814335j ,
          0.19677100058992886 -0.19168167356500643j,
          0.24448011783021284 -0.42093450922860026j],
        [-0.2178752955950242  +0.4769575279629936j ,
         -0.3294494231631181  -0.5699584340865812j ,
         -0.34372129423527903 -0.3685833165010564j ,
         -0.18997371175984737 -0.03955164479974399j]]]), array([[23.168604748820908 , 15.485742042115662 , 11.129821974271639 ,
         4.172163032990345 ],
       [31.035300784567507 , 12.500953553782981 ,  6.796077120797179 ,
         1.5199501235275763]]), array([[[ 0.36257669741978926 -0.14221175501714423j ,
         -0.058250697282110166-0.5790785298475211j  ,
          0.7050592229784634  +0.01949879578361566j ,
         -0.06663926262442564 -0.08751780099928909j ],
        [ 0.0644324594822553  +0.27093299962776496j ,
         -0.712394512739288   +0.0252882035037107j  ,
          0.04580605296283848 +0.4195153849647771j  ,
          0.37202557799112873 +0.31273622986546246j ],
        [ 0.6369967429219748  -0.10691103955907996j ,
          0.24877842613377837 +0.02569569786047833j ,
         -0.20245308774453494 -0.2601671253010774j  ,
          0.2720824775068983  +0.5809915184734116j  ],
        [-0.3815416405835403  +0.4560579773766533j  ,
          0.2169579367540845  -0.20858513130029815j ,
          0.20518865758270805 -0.416705369920401j   ,
          0.5823815708987322  +0.030733529055695246j]],

       [[ 0.1477206865428076  +0.315398845668301j   ,
          0.3084825844834702  +0.20608887424285138j ,
          0.6739221002092323  +0.003451102602297634j,
         -0.4061164568347281  -0.34921992229904514j ],
        [-0.30299017724213945 -0.5665496536910978j  ,
         -0.5581728194563791  +0.19040909260866232j ,
          0.3031519199482866  -0.25576049135034773j ,
         -0.2314434986696236  -0.16889438974932564j ],
        [-0.4513340696752332  -0.17983214863943056j ,
          0.226097381405423   -0.42903008355691824j ,
         -0.03866935846260517 +0.4630617127443596j  ,
         -0.5388857439413695  +0.14983925812009435j ],
        [ 0.409305316379769   +0.24969526059794167j ,
         -0.48587930748034175 +0.20805723285421662j ,
         -0.35061304467035664 +0.22282639217479672j ,
         -0.5598471225880682  +0.06888405058400499j ]]])),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f64>> loc("operand")) -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<2x4xf64> {jax.result_info = "result[1]"}, tensor<2x4x4xcomplex<f64>> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc4)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_gesvdj_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>>, tensor<2x4x4xcomplex<f64>>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.transpose %0#3, dims = [0, 2, 1] : (tensor<2x4x4xcomplex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc4)
    %2 = stablehlo.real %1 : (tensor<2x4x4xcomplex<f64>>) -> tensor<2x4x4xf64> loc(#loc4)
    %3 = stablehlo.imag %1 : (tensor<2x4x4xcomplex<f64>>) -> tensor<2x4x4xf64> loc(#loc4)
    %4 = stablehlo.negate %3 : tensor<2x4x4xf64> loc(#loc4)
    %5 = stablehlo.complex %2, %4 : tensor<2x4x4xcomplex<f64>> loc(#loc4)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %7 = stablehlo.compare  EQ, %0#4, %6,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc4)
    %10 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %11 = stablehlo.select %10, %0#1, %9 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc4)
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %15 = stablehlo.select %14, %0#2, %13 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc4)
    %16 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc4)
    %18 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %19 = stablehlo.select %18, %5, %17 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc4)
    return %15, %11, %19 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":621:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("svd"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01-\x07\x01\x05\t\x1b\x01\x03\x0f\x03\x19\x13\x17\x1b\x1f#'+/37;?\x03\xef\x9f9\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03Uo\x0f\x0b/\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0bO/\x1f\x1b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/o\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x035\x1b\x07\x07\x17\x07\x07\x1b\x0b\x0f\x0f\x0f\x07\x13\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x02\x06\x08\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05!\x11\x01\x00\x05#\x05%\x05'\x1d\x15\x03\x05)\x05+\x1d\x1b\x1d\x05-\x17\x1f\xb6\t\x1b\x05/\x03\x07#S%Y'\x7f\x051\x053\x055\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x01\x1d7\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x1f\x1f1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x037\r\x01#%\x03\x07=AE\r\x03-?\x1d9\r\x03-C\x1d;\r\x03-G\x1d=\x1d?\x1dA\x1f\x15!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x17\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x19\t\x00\x00\x00\x00\r\x05U1W1\x1dC\x1dE\r\x03[]\x1dG\x1dI\x0b\x03\x1dK\x1dM\x03\x01\x05\x01\x03\x03)\x03\x03m\x15\x03\x01\x01\x01\x03\x0b)q))s\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1f1\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f7!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x81\x0b\x87\x8d\x91\x97\x9d\x01\x01\x01\x01\x01\x13\x07}\x83\x85\x11\x03\x05\x11\x03\t\x13\x07}\x89\x8b\x11\x03\r\x11\x03\x11\x13\x05}\x8f\x11\x03\x15\x13\x07}\x93\x95\x11\x03\x19\x11\x03\x1d\x13\x07}\x99\x9b\x11\x03!\x11\x03%\x13\x03}\x01\t\x01\x02\x02)\x07\t\x11\x11\x13\x01\x0b)\x05\t\x11\t\x1d\x13)\x07\t\x11\x11\t\x03\t)\x01\x13)\x01\t)\x01\x1b\x1b)\x03\t\x1b)\x03\r\r)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\x0b\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\r)\x03\t\x07)\x05\t\x05\x07)\x03\x05\r)\x05\t\x11\x07)\x03\t\r\x04n\x03\x05\x01Q\x03\x07\x01\x07\x04F\x03\x03\x01\x05\tP\x03\x03\x07\x04\x1a\x03\x039c\x03\x0b\x13\x00\x05B\x03\x05\x03\x15\x05B\x01\x07\x03\x17\x05B\x01\t\x03\x19\x0bG\x01!\x0b\x0b\x05\x0b\x05\x05\x1d\x03\x01\rF\x01\r\x03\x05\x03\x0f\x0f\x06\x01\x03\x11\x03\x13\x11\x06\x01\x03\x11\x03\x13\x13\x06\x01\x03\x11\x03\x17\x15\x06\x01\x03\x05\x05\x15\x19\x03F\x01\x0f\x03\x1d\x03\x07\x17F\x01\x11\x03/\x05\x11\x1d\x03F\x01\x13\x031\x03\x1f\x03F\x01\x0f\x03\x0b\x03\x05\x03F\x01\x15\x035\x03!\x07\x06\x01\x03\x0b\x07%\x0b#\x03F\x01\x13\x03!\x03\x1f\x03F\x01\x0f\x03\x05\x03\x03\x03F\x01\x17\x03#\x03)\x07\x06\x01\x03\x05\x07-\r+\x03F\x01\x13\x03!\x03\x1f\x03F\x01\x0f\x03\x05\x03\x03\x03F\x01\x17\x03#\x031\x07\x06\x01\x03\x05\x075\x1b3\x19\x04\x03\x07/'7\x06\x03\x01\x05\x01\x00n\x08O+\x03\x05\x1f\x1d\x17\x0f\x0b\x15\x15\x15!%3)s\x15\t\x11\x13%)9\x15\x17\x17\x15\x11\x11\x1b\x1f\x11\x15\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00num_batch_dims\x001\x00\x00hipsolver_gesvdj_ffi\x00\x08M\x19\x05#\x01\x0b59;IK\x03M\x03O\x03Q\x11_acegiko\x03u\x03+\x05wy\x03/\x03{\x033",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["qr"]["f32"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_gesvd_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[-8.088836  , -2.6205542 , -7.2633333 ,  5.0895734 ],
        [ 9.784253  ,  8.099774  , -8.470991  ,  3.0590599 ],
        [-6.5720334 , -4.4796433 ,  5.5712957 ,  7.9569902 ],
        [-7.765548  , -8.527966  ,  5.5323663 , -0.61520344]],

       [[-6.685037  ,  5.1507697 , -8.878728  ,  8.7760725 ],
        [ 2.2026582 ,  5.15965   ,  8.254493  ,  5.2088103 ],
        [ 1.649047  , -3.3060083 , -4.6077223 , -0.92500544],
        [-2.111105  ,  2.339333  ,  2.2933412 ,  1.7438462 ]]],
      dtype=float32),),
    expected_outputs=(array([[[-0.19646618 , -0.88533336 ,  0.40994054 , -0.097645864],
        [ 0.6584475  , -0.35089356 , -0.3007529  ,  0.5940271  ],
        [-0.46186477 , -0.27749717 , -0.8389231  , -0.07670852 ],
        [-0.56082886 ,  0.126704   ,  0.19417621 ,  0.7948037  ]],

       [[-0.9831455  , -0.14481494 , -0.070398524, -0.0865914  ],
        [ 0.15959439 , -0.84682167 , -0.50710094 ,  0.016481   ],
        [-0.06209982 ,  0.41913196 , -0.70081264 ,  0.5738761  ],
        [-0.06402065 , -0.29368412 ,  0.49674392 ,  0.8141846  ]]],
      dtype=float32), array([[22.384214 , 12.4938135,  7.4144673,  1.7072145],
       [15.208481 , 12.81391  ,  3.4745562,  0.4607253]], dtype=float32), array([[[ 0.6889739  ,  0.56735736 , -0.4389969  , -0.10345369 ],
        [ 0.3656113  , -0.028776985,  0.68496627 , -0.6295405  ],
        [-0.30387047 , -0.1899195  , -0.5434635  , -0.75910497 ],
        [ 0.54708886 , -0.80075485 , -0.20676109 ,  0.12936543 ]],

       [[ 0.4574187  , -0.2751733  ,  0.6697428  , -0.5162291  ],
        [ 0.03230872 , -0.5609443  , -0.64844155 , -0.51363546 ],
        [-0.8204515  ,  0.14386664 ,  0.23241581 , -0.5021402  ],
        [-0.3414436  , -0.7674136  ,  0.2774007  ,  0.4664135  ]]],
      dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf32> loc("operand")) -> (tensor<2x4x4xf32> {jax.result_info = "result[0]"}, tensor<2x4xf32> {jax.result_info = "result[1]"}, tensor<2x4x4xf32> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc4)
    return %10, %6, %14 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":621:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("svd"(#loc3))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.13.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#\'+\x03\xe7\x9d3\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03So\x0f\x0b/\x0b\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f#\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02"\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x15\x03\x05\x1f\x05!\x1d\x1b\x1d\x05#\x17\x1f\xb6\t\x1b\x05%\x03\x07#S%[\'}\x05\'\x05)\x05+\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f%\x01\x1d-\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f11\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x039\r\x01#\x1d\x03\x07?CG\r\x03-A\x1d/\r\x03-E\x1d1\r\x03-I\x1d3\x1d5\x1d7\x1f\x11\t\x00\x00\xc0\x7f\x1f\x13\t\x00\x00\x00\x00\r\x07U1W1Y3\x1d9\x1d;\x1d=\r\x03]_\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x03\x03)\x03\x03m\x15\x03\x01\x01\x01\x03\x0b)q))s\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x7f\x0b\x85\x8b\x8f\x95\x9b\x01\x01\x01\x01\x01\x13\x07{\x81\x83\x11\x03\x05\x11\x03\t\x13\x07{\x87\x89\x11\x03\r\x11\x03\x11\x13\x05{\x8d\x11\x03\x15\x13\x07{\x91\x93\x11\x03\x19\x11\x03\x1d\x13\x07{\x97\x99\x11\x03!\x11\x03%\x13\x03{\x01\t\x01\x02\x02)\x07\t\x11\x11\r\x01)\x05\t\x11\r\x1d\t\x13)\x01\r)\x01\x15\x1b)\x03\t\x15)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xc2\x02\x05\x01Q\x03\x07\x01\x07\x04\x9a\x02\x03\x01\x05\tP\x03\x03\x07\x04n\x02\x03-K\x03\x0b\x13\x00\x07B\x03\x05\x03\x11\x07B\x01\x07\x03\x13\x0bG\x01!\t\x0b\x05\t\x05\x05\x17\x03\x01\x03F\x01\x0b\x03\x17\x03\x05\rF\x01\r\x03\'\x05\x0f\x11\x03F\x01\x0f\x03)\x03\x13\x03F\x01\x0b\x03\t\x03\x03\x03F\x01\x11\x03-\x03\x15\x05\x06\x01\x03\t\x07\x19\t\x17\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03\x1d\x05\x06\x01\x03\x05\x07!\x0b\x1f\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03%\x05\x06\x01\x03\x05\x07)\r\'\x0f\x04\x03\x07#\x1b+\x06\x03\x01\x05\x01\x00\xbe\x07G)\x03\x05\x1f\x17\x1d\x17\x0f\x0b\x15\x15\x15!%3)s\x15\t\x11\x13%)9\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00num_batch_dims\x001\x00\x00hipsolver_gesvd_ffi\x00\x08E\x15\x05#\x01\x0b7;=KM\x03O\x03Q\x11aceg3iko\x03+\x05uw\x03/\x03y\x035',
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["qr"]["f64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_gesvd_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[ 7.516621433947318  ,  9.495925414094074  ,
          4.361961694107173  ,  5.57838011364683   ],
        [-7.294446875834851  , -9.622183592984825  ,
          4.291499000204173  , -2.5015601805990517 ],
        [-2.40944749022721   ,  6.628440455612534  ,
         -1.6017329763895418 , -8.169012011427746  ],
        [-0.9524103592344364 ,  2.6056644652182754 ,
         -1.7707069509199496 ,  9.079755164737282  ]],

       [[-8.292072309603471  ,  4.387200699980834  ,
         -6.17701480971335   ,  2.3067167339225403 ],
        [-8.572355700016178  ,  9.039479678554585  ,
          4.690587216470503  ,  9.421807731360794  ],
        [-0.40052116228612533, -9.00913849115783   ,
          4.660812984455196  , -0.39685471313335086],
        [ 6.85836794331054   , -0.7762372063612624 ,
         -1.9062402955103508 ,  8.146374381579843  ]]]),),
    expected_outputs=(array([[[ 0.710593156752799   ,  0.04339323803857607 ,
          0.6108799359169268  , -0.3464103006013976  ],
        [-0.6414551962489828  ,  0.2654507519818231  ,
          0.3810239670429581  , -0.6106487255503853  ],
        [ 0.002097973122686177, -0.8301780287591775  ,
         -0.23115175512952152 , -0.5073153902404739  ],
        [ 0.28911623145379906 ,  0.48832096589265755 ,
         -0.6543816214820828  , -0.49973906435868637 ]],

       [[-0.4845794420083184  ,  0.36379198830363746 ,
          0.3585778089804869  ,  0.7101127435384802  ],
        [-0.8117357808096087  , -0.2737011928801392  ,
         -0.48805327524177633 , -0.16726230805289055 ],
        [ 0.32303983104156553 , -0.02153072337219175 ,
         -0.7310502778628764  ,  0.6006223328750929  ],
        [ 0.043738473875528   , -0.8901008224988602  ,
          0.31431106673030207 ,  0.32713303871781874 ]]]), array([[18.683935768333495 , 12.662275558869085 ,  6.8338862408024434,
         4.845264432750695 ],
       [18.74775820344496  , 11.410736777937716 ,  9.946895692662434 ,
         6.137629336027487 ]]), array([[[ 5.2129853559569894e-01,  7.3256436071718900e-01,
         -9.0197653754896436e-03,  4.3762533978891865e-01],
        [-5.9197451292344128e-03, -5.0326988476028811e-01,
          1.4164211376179620e-01,  8.5242119361267688e-01],
        [ 4.3790339491604441e-01, -1.6135574983257261e-01,
          8.5292030374161054e-01, -2.3394848617225703e-01],
        [ 7.3242979876871994e-01, -4.2899091190414557e-01,
         -5.0237745859759764e-01, -1.6471270889728379e-01]],

       [[ 5.9459087097069763e-01, -6.6183264336451042e-01,
          3.2429963316260670e-02, -4.5539822772465766e-01],
        [-5.9298019579717476e-01,  5.9722166419666663e-04,
         -1.6953966020379732e-01, -7.8716607798901594e-01],
        [ 3.6784091434819871e-01,  3.5222631382775416e-01,
         -8.5560743787328120e-01, -9.2550515042935433e-02],
        [-3.9941112313270993e-01, -6.6175057185524333e-01,
         -4.8799642760291367e-01,  4.0548294910380545e-01]]])),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf64> loc("operand")) -> (tensor<2x4x4xf64> {jax.result_info = "result[0]"}, tensor<2x4xf64> {jax.result_info = "result[1]"}, tensor<2x4x4xf64> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64>, tensor<2x4x4xf64>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc4)
    return %10, %6, %14 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":621:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("svd"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xe7\x9d3\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03So\x0f\x0b/\x0b\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f#\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x022\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x15\x03\x05\x1f\x05!\x1d\x1b\x1d\x05#\x17\x1f\xb6\t\x1b\x05%\x03\x07#S%['}\x05'\x05)\x05+\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f%\x01\x1d-\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f11\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x039\r\x01#\x1d\x03\x07?CG\r\x03-A\x1d/\r\x03-E\x1d1\r\x03-I\x1d3\x1d5\x1d7\x1f\x11\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x13\t\x00\x00\x00\x00\r\x07U1W1Y3\x1d9\x1d;\x1d=\r\x03]_\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x03\x03)\x03\x03m\x15\x03\x01\x01\x01\x03\x0b)q))s\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x7f\x0b\x85\x8b\x8f\x95\x9b\x01\x01\x01\x01\x01\x13\x07{\x81\x83\x11\x03\x05\x11\x03\t\x13\x07{\x87\x89\x11\x03\r\x11\x03\x11\x13\x05{\x8d\x11\x03\x15\x13\x07{\x91\x93\x11\x03\x19\x11\x03\x1d\x13\x07{\x97\x99\x11\x03!\x11\x03%\x13\x03{\x01\t\x01\x02\x02)\x07\t\x11\x11\r\x01)\x05\t\x11\r\x1d\x0b\x13)\x01\r)\x01\x15\x1b)\x03\t\x15)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xc2\x02\x05\x01Q\x03\x07\x01\x07\x04\x9a\x02\x03\x01\x05\tP\x03\x03\x07\x04n\x02\x03-K\x03\x0b\x13\x00\x07B\x03\x05\x03\x11\x07B\x01\x07\x03\x13\x0bG\x01!\t\x0b\x05\t\x05\x05\x17\x03\x01\x03F\x01\x0b\x03\x17\x03\x05\rF\x01\r\x03'\x05\x0f\x11\x03F\x01\x0f\x03)\x03\x13\x03F\x01\x0b\x03\t\x03\x03\x03F\x01\x11\x03-\x03\x15\x05\x06\x01\x03\t\x07\x19\t\x17\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03\x1d\x05\x06\x01\x03\x05\x07!\x0b\x1f\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03%\x05\x06\x01\x03\x05\x07)\r'\x0f\x04\x03\x07#\x1b+\x06\x03\x01\x05\x01\x00\xbe\x07G)\x03\x05\x1f\x17\x1d\x17\x0f\x0b\x15\x15\x15!%3)s\x15\t\x11\x13%)9\x15\x17\x1f\x11\x19\x15)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00num_batch_dims\x001\x00\x00hipsolver_gesvd_ffi\x00\x08E\x15\x05#\x01\x0b7;=KM\x03O\x03Q\x11aceg3iko\x03+\x05uw\x03/\x03y\x035",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["qr"]["c64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_gesvd_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[ 5.1529994  -1.2366186j,  3.4151247  -4.7387176j,
          9.5636425  -8.201898j ,  6.536134   -4.6525245j],
        [-8.4091     +6.4209814j, -3.8424058  -0.511163j ,
         -2.1341999  +4.956986j , -0.68947995 +2.1529553j],
        [-0.30219    -9.666059j , -0.044237416+4.2940593j,
         -4.1715403  +5.2976294j, -1.3943245  +1.8789417j],
        [-9.0755005  +7.546829j , -7.630291   -3.62061j  ,
          6.448428   -4.0384274j, -7.065759   -9.805366j ]],

       [[ 4.286493   +0.7896354j, -4.024747   +2.2482893j,
         -8.564833   +5.948362j , -9.7399845  -3.4490995j],
        [ 6.6070595  -8.43279j  ,  1.6180693  +5.6450553j,
         -8.772443   -2.9169078j, -1.2268616  -4.769274j ],
        [ 1.3227078  +1.7313296j,  3.9845204  -0.5684758j,
          9.138408   +6.9373603j, -4.7926993  -4.8423553j],
        [ 3.4307988  -5.6387033j,  5.9531016  -0.8393127j,
         -2.0458186  -6.3643475j, -4.3651037  -8.67327j  ]]],
      dtype=complex64),),
    expected_outputs=(array([[[ 0.30495852 +0.096999586j,  0.72827613 +0.03194175j ,
          0.20767818 +0.042641692j, -0.28966787 -0.4871642j  ],
        [ 0.115591295-0.1449632j  , -0.49115598 +0.23452608j ,
         -0.013264754-0.5263807j  , -0.35325044 -0.5170581j  ],
        [-0.2174027  +0.29466262j , -0.2795165  -0.27255905j ,
         -0.27765262 +0.6150545j  , -0.34620082 -0.37182543j ],
        [ 0.60895675 -0.59857947j , -0.12601145 +0.06371821j ,
         -0.22057684 +0.4168473j  ,  0.15328544 -0.07087361j ]],

       [[-0.022096228+0.5509537j  ,  0.10202758 +0.25620914j ,
          0.46818483 +0.513836j   ,  0.31120205 -0.19959508j ],
        [-0.4309677  +0.4594079j  ,  0.15301313 -0.036881473j,
          0.22284088 -0.48527643j , -0.05199984 +0.53905755j ],
        [ 0.26196426 -0.032041073j,  0.6947033  -0.22732396j ,
         -0.117719345+0.40873566j , -0.19235809 +0.42206088j ],
        [-0.3937812  +0.2728696j  ,  0.23826325 -0.55508226j ,
         -0.22510253 +0.005555059j, -0.16906178 -0.5712348j  ]]],
      dtype=complex64), array([[23.247055 , 20.633308 ,  7.335599 ,  2.013584 ],
       [22.780914 , 15.782362 , 12.775513 ,  1.0651652]], dtype=float32), array([[[-0.57116145 -0.j         , -0.042698324-0.4338231j  ,
          0.4287732  -0.07257515j ,  0.15371813 -0.5205827j  ],
        [ 0.6636313  +0.j         ,  0.17815456 -0.12978375j ,
          0.36668688 -0.52016j    ,  0.271322   -0.17991613j ],
        [-0.40411592 -0.j         ,  0.49817383 -0.045163676j,
          0.049928322-0.5456577j  , -0.1296575  +0.51906395j ],
        [ 0.2646859  +0.j         ,  0.22178493 -0.6796956j  ,
          0.08210006 +0.3144607j  , -0.5465213  +0.12022976j ]],

       [[-0.39417964 -0.j         ,  0.07517068 -0.10199716j ,
          0.3137599  +0.6606039j  , -0.22367497 +0.4936552j  ],
        [ 0.40769133 +0.j         ,  0.31594533 +0.36745775j ,
          0.45824614 +0.397673j   , -0.021774545-0.47993127j ],
        [ 0.6047127  -0.j         , -0.4034314  +0.29415658j ,
          0.054178037-0.06489535j , -0.3735269  +0.48823014j ],
        [-0.55922544 +0.j         , -0.25889853 +0.6578646j  ,
          0.17150019 -0.24589685j , -0.2621226  -0.1699021j  ]]],
      dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f32>> loc("operand")) -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "result[0]"}, tensor<2x4xf32> {jax.result_info = "result[1]"}, tensor<2x4x4xcomplex<f32>> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc4)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>>, tensor<2x4x4xcomplex<f32>>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc4)
    return %10, %6, %14 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":621:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("svd"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xed\x9f7\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03Uo\x0f\x0b/\x0b\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b/\x1f\x1f#\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x033\x1b\x07\x17\x07\x07\x07\x0b\x0f\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02b\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x15\x03\x05\x1f\x05!\x1d\x1b\x1d\x05#\x17\x1f\xb6\t\x1b\x05%\x03\x07#U%]'\x7f\x05'\x05)\x05+\x1f#1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f)\x01\x1d-\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f51\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x039\r\x01#!\x03\x07?CG\r\x03-A\x1d/\r\x03-E\x1d1\r\x03-I\x1d3\x1d5\x1d7\x1f\x13\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x15\t\x00\x00\xc0\x7f\x1f\x17\t\x00\x00\x00\x00\r\x07W1Y1[3\x1d9\x1d;\x1d=\r\x03_a\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x03\x03)\x03\x03o\x15\x03\x01\x01\x01\x03\x0b)s))u\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x81\x0b\x87\x8d\x91\x97\x9d\x01\x01\x01\x01\x01\x13\x07}\x83\x85\x11\x03\x05\x11\x03\t\x13\x07}\x89\x8b\x11\x03\r\x11\x03\x11\x13\x05}\x8f\x11\x03\x15\x13\x07}\x93\x95\x11\x03\x19\x11\x03\x1d\x13\x07}\x99\x9b\x11\x03!\x11\x03%\x13\x03}\x01\t\x01\x02\x02)\x07\t\x11\x11\x11\x01)\x05\t\x11\r\x1d\t\x13\x03\r)\x01\x11)\x01\r)\x01\x19\x1b)\x03\t\x19)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xda\x02\x05\x01Q\x03\x07\x01\x07\x04\xb2\x02\x03\x01\x05\tP\x03\x03\x07\x04\x86\x02\x03/O\x03\x0b\x13\x00\x05B\x03\x05\x03\x13\x05B\x01\x07\x03\x15\x05B\x01\t\x03\x17\x0bG\x01!\x0b\x0b\x05\t\x05\x05\x1b\x03\x01\x03F\x01\r\x03\x1b\x03\x07\rF\x01\x0f\x03+\x05\x11\x13\x03F\x01\x11\x03-\x03\x15\x03F\x01\r\x03\t\x03\x05\x03F\x01\x13\x031\x03\x17\x07\x06\x01\x03\t\x07\x1b\x0b\x19\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03\x1f\x07\x06\x01\x03\x05\x07#\r!\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03'\x07\x06\x01\x03\x05\x07+\x0f)\x0f\x04\x03\x07%\x1d-\x06\x03\x01\x05\x01\x00\xbe\x07G)\x03\x05\x1f\x17\x1d\x17\x0f\x0b\x15\x15\x15!%3)s\x15\t\x11\x13%)9\x15\x17\x1f\x11\x15\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00num_batch_dims\x001\x00\x00hipsolver_gesvd_ffi\x00\x08I\x17\x05#\x01\x0b7;=KM\x03O\x03Q\x03S\x11cegi3kmq\x03+\x05wy\x03/\x03{\x035",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_02_04["qr"]["c128"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_gesvd_ffi'],
    serialized_date=datetime.date(2026, 2, 4),
    inputs=(array([[[-1.9905224803805588-1.8349519681179167j,
          5.735842655793476 +9.800262022922936j ,
          9.930770453807153 -7.660992394369175j ,
         -3.2929936512332603-0.3833378170735138j],
        [ 4.341007948722385 +6.667554829993922j ,
          1.5156378758308087+7.411002120943763j ,
          2.4340125691446453-3.0850645481331496j,
          9.875989691331203 -6.1366179498083735j],
        [ 3.818748621455894 +4.606673418196037j ,
          2.3537469571912517-2.9500306466751454j,
         -7.202066665251294 +9.776480393608004j ,
          1.520165382948818 +3.0178751445902474j],
        [-0.2425449354190654-6.241013740812507j ,
         -4.519117705886435 +3.868789479046006j ,
         -1.1661731919174017-2.2818202948221833j,
         -4.660958033035887 +8.360465211846016j ]],

       [[-6.145471273178405 -1.0978433303755946j,
          9.51601108696423  +5.991155719658801j ,
          1.5805963559425624+8.308317987231838j ,
          3.5484468974281995+9.934912177588416j ],
        [ 7.09882908688272  +9.89156207005951j  ,
          9.567356209650448 -4.229572146087315j ,
         -2.044167232859591 -7.774476424929708j ,
          3.1765005877310397+6.075776012012803j ],
        [ 3.117187801988422 +7.826519832345586j ,
          4.6711728079985235+2.922980100012577j ,
         -9.474622452988285 +9.198398274345259j ,
         -9.629383830755769 -0.6972018625578293j],
        [ 7.526887742226929 +8.892734666231693j ,
          3.629557229748798 -6.978519004131796j ,
         -8.884252532467453 +9.982683952825951j ,
          0.7061661847199137-1.6717995600652547j]]]),),
    expected_outputs=(array([[[-0.35905865386373553  -0.6657106780577163j  ,
          0.10967295665724751  +0.0725422808481802j  ,
         -0.06540584145175349  +0.4126017403819886j  ,
         -0.001892455755736052 -0.48589498312068874j ],
        [-0.019365762327029423 -0.3458402738996379j  ,
         -0.16646314362343423  -0.7163119320774831j  ,
         -0.2931706961839591   -0.30249907055652614j ,
         -0.3813721127443      +0.12769994835294057j ],
        [ 0.23695882564468548  +0.4995905479185137j  ,
         -0.08616434176653853  -0.2306488965643763j  ,
         -0.4047944302869717   +0.45289367548195103j ,
         -0.20089033683194188  -0.4736121330643781j  ],
        [-0.019328042734500266 -0.04256611971555387j ,
          0.40686973660845127  +0.4644679838261983j  ,
         -0.528195531188617    -0.008087255323249769j,
         -0.49750847736278525  +0.29995075218256234j ]],

       [[-0.05849782863570836  +0.1349108572724311j  ,
          0.6915190647040965   -0.00724172047557089j ,
          0.35050509142457625  +0.5154062016013146j  ,
          0.3335086644994931   +0.02001509955933467j ],
        [ 0.12409676403141817  +0.46965841967658595j ,
         -0.0010071875014091454-0.47363472964505565j ,
         -0.6419183280668094   +0.10040875952807082j ,
          0.3407897981969638   +0.03756787161969258j ],
        [ 0.18961104008215124  +0.4852170511373144j  ,
          0.10670334298673206  +0.512049704917858j   ,
         -0.004857315166004077 -0.2903275987339464j  ,
          0.11649485988968188  -0.5976176056439488j  ],
        [ 0.5193617369970899   +0.44863512961725066j ,
         -0.10599623987661626  +0.11226024453798102j ,
          0.3132053628168187   -0.08336823102264292j ,
         -0.10592214730187742  +0.6236064293405494j  ]]]), array([[21.932220876405268 , 18.880687206540177 ,  9.052253933601701 ,
         6.147242993662446 ],
       [27.125418969190424 , 23.667528381493124 , 12.813953745318111 ,
         2.4083691203679507]]), array([[[ 0.1378321823471533  -0.j                  ,
         -0.5548648056348718  -0.0666549568767057j  ,
          0.2667942513133091  +0.7373836529036589j  ,
          0.22664082543819475 +0.049036372646054247j],
        [-0.5423046445376736  +0.j                  ,
         -0.20047141262021817 +0.26380919119352203j ,
         -0.04399203613481576 -0.11619433399764471j ,
          0.18656608276051542 +0.7388357531574219j  ],
        [-0.35321469731224603 -0.j                  ,
          0.11589839419447341 -0.7372405155817185j  ,
          0.48459340421596897 -0.16079096916727068j ,
          0.23904452001415227 -0.021367645785028748j],
        [-0.7497648562215402  -0.j                  ,
         -0.011601857648442198+0.14424794231622187j ,
         -0.14742701295968097 +0.2953481438482363j  ,
         -0.20589294112027193 -0.5153187702614431j  ]],

       [[ 0.6645194179564081  +0.j                  ,
          0.01882587817102112 -0.5020226405475181j  ,
         -0.012734677138399427+0.5459009841135211j  ,
          0.06757793695402495 +0.05741048533823283j ],
        [-0.18562298018354922 +0.j                  ,
          0.3953840883722747  +0.2957581208145406j  ,
          0.4427410359357688  +0.4465462969959478j  ,
         -0.09067453282761491 +0.5640013960056106j  ],
        [-0.5427543454444135  +0.j                  ,
          0.05497243001630322 -0.1241954144322904j  ,
         -0.06802220053499675 +0.5372137093245434j  ,
          0.4327322582872426  -0.45441000743718807j ],
        [ 0.478931908404379   -0.j                  ,
          0.18941886075062406 +0.670441225975174j   ,
          0.11217879809210324 +0.024434632932612357j,
          0.3614901872681235  -0.37602791824264514j ]]])),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f64>> loc("operand")) -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "result[0]"}, tensor<2x4xf64> {jax.result_info = "result[1]"}, tensor<2x4x4xcomplex<f64>> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc4)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc4)
    %0:5 = stablehlo.custom_call @hipsolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, l, m], [i, n], [i, o, p], [i, q, r], [i]) {i=2, j=4, k=4, l=4, m=4, n=4, o=4, p=4, q=4, r=4}, custom>} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>>, tensor<2x4x4xcomplex<f64>>, tensor<2xi32>) loc(#loc4)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc4)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc4)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc4)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc4)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc4)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc4)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc4)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc4)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc4)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc4)
    return %10, %6, %14 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":621:13)
#loc3 = loc("jit(func)"(#loc2))
#loc4 = loc("svd"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.13.1\x00\x01#\x07\x01\x05\t\x11\x01\x03\x0f\x03\x0f\x13\x17\x1b\x1f#'+\x03\xed\x9f7\x01)\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x0b\x17\x0b#\x0b\x0b\x0b\x03Uo\x0f\x0b/\x0b\x0bo\x0f\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0bO/\x1f#\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x05#\x0fg\x17\x0f\x0f\x17\x0f\x0f\x13\x0f\x17\x0f\x0f\x17\x0f\x0f\x0f\x01\x05\x0b\x0f\x033\x1b\x07\x17\x07\x07\x07\x0b\x0f\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\x92\x07\x1d\x17\x19\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x15\x03\x05\x1f\x05!\x1d\x1b\x1d\x05#\x17\x1f\xb6\t\x1b\x05%\x03\x07#U%]'\x7f\x05'\x05)\x05+\x1f#1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f)\x01\x1d-\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f51\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x039\r\x01#!\x03\x07?CG\r\x03-A\x1d/\r\x03-E\x1d1\r\x03-I\x1d3\x1d5\x1d7\x1f\x13!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x15\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x17\t\x00\x00\x00\x00\r\x07W1Y1[3\x1d9\x1d;\x1d=\r\x03_a\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x03\x03)\x03\x03o\x15\x03\x01\x01\x01\x03\x0b)s))u\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x11\x03\x01\x15\x15\t\x11\x11\x11\x11\x11\x11\x11\x11\x11\x03\x81\x0b\x87\x8d\x91\x97\x9d\x01\x01\x01\x01\x01\x13\x07}\x83\x85\x11\x03\x05\x11\x03\t\x13\x07}\x89\x8b\x11\x03\r\x11\x03\x11\x13\x05}\x8f\x11\x03\x15\x13\x07}\x93\x95\x11\x03\x19\x11\x03\x1d\x13\x07}\x99\x9b\x11\x03!\x11\x03%\x13\x03}\x01\t\x01\x02\x02)\x07\t\x11\x11\x11\x01)\x05\t\x11\r\x1d\x0b\x13\x03\r)\x01\x11)\x01\r)\x01\x19\x1b)\x03\t\x19)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xda\x02\x05\x01Q\x03\x07\x01\x07\x04\xb2\x02\x03\x01\x05\tP\x03\x03\x07\x04\x86\x02\x03/O\x03\x0b\x13\x00\x05B\x03\x05\x03\x13\x05B\x01\x07\x03\x15\x05B\x01\t\x03\x17\x0bG\x01!\x0b\x0b\x05\t\x05\x05\x1b\x03\x01\x03F\x01\r\x03\x1b\x03\x07\rF\x01\x0f\x03+\x05\x11\x13\x03F\x01\x11\x03-\x03\x15\x03F\x01\r\x03\t\x03\x05\x03F\x01\x13\x031\x03\x17\x07\x06\x01\x03\t\x07\x1b\x0b\x19\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03\x1f\x07\x06\x01\x03\x05\x07#\r!\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03'\x07\x06\x01\x03\x05\x07+\x0f)\x0f\x04\x03\x07%\x1d-\x06\x03\x01\x05\x01\x00\xbe\x07G)\x03\x05\x1f\x17\x1d\x17\x0f\x0b\x15\x15\x15!%3)s\x15\t\x11\x13%)9\x15\x17\x1f\x11\x15\x19)\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00svd\x00jit(func)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jax.result_info\x00result[0]\x00result[1]\x00result[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00num_batch_dims\x001\x00\x00hipsolver_gesvd_ffi\x00\x08I\x17\x05#\x01\x0b7;=KM\x03O\x03Q\x03S\x11cegi3kmq\x03+\x05wy\x03/\x03{\x035",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
