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
from numpy import array, float32, complex64

data_2023_06_19 = {}



# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_19["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgesdd'],
    serialized_date=datetime.date(2023, 6, 22),
    inputs=(array([[[ 1.5410905 , -2.775912  , -2.374003  ,  4.028736  ],
        [-0.56933475,  1.6115232 ,  0.9041465 , -0.8321383 ],
        [-5.382895  ,  4.734856  ,  2.1972926 ,  1.5553856 ],
        [ 0.5109847 , -1.1969309 ,  3.3766198 , -1.3678027 ]],

       [[ 2.2637439 ,  3.406768  ,  4.809871  ,  2.8010902 ],
        [-1.9981416 , -0.6599986 ,  0.5138156 ,  4.5982494 ],
        [-2.335944  , -9.151717  , -1.0481138 ,  2.272443  ],
        [-8.257684  ,  1.8223318 ,  0.38403794,  5.0769973 ]]],
      dtype=float32),),
    expected_outputs=(array([[[-0.48540133 ,  0.6682397  , -0.48819906 , -0.28196266 ],
        [ 0.2180054  , -0.13631375 ,  0.14819765 , -0.95495003 ],
        [ 0.8457052  ,  0.44643915 , -0.27943406 ,  0.08597418 ],
        [ 0.040523227, -0.57928085 , -0.8133977  , -0.03429017 ]],

       [[-0.21146733 ,  0.46376425 ,  0.786309   ,  0.34917438 ],
        [ 0.3461469  ,  0.21883713 ,  0.3399653  , -0.84659094 ],
        [ 0.6526192  , -0.5834038  ,  0.3972404  ,  0.2755518  ],
        [ 0.6399631  ,  0.6298203  , -0.32915345 ,  0.2922879  ]]],
      dtype=float32), array([[ 8.551608 ,  5.3574076,  2.8073738,  0.5226082],
       [11.457576 , 10.041606 ,  5.6716514,  1.4754109]], dtype=float32), array([[[-0.6319046  ,  0.6612254  ,  0.39110154 , -0.102553196],
        [-0.2971051  ,  0.13673358 , -0.50112    ,  0.80119365 ],
        [ 0.08969147 ,  0.4433047  , -0.73647296 , -0.5030348  ],
        [-0.7101976  , -0.5895471  , -0.23135659 , -0.30745354 ]],

       [[-0.6964344  , -0.5023085  , -0.11150039 ,  0.50023323 ],
        [-0.32121164 ,  0.7889568  ,  0.3183193  ,  0.41598475 ],
        [ 0.5096958  , -0.31399378 ,  0.60193455 ,  0.5284816  ],
        [-0.3898877  , -0.16322286 ,  0.7238198  , -0.5453721  ]]],
      dtype=float32)),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf32> {jax.arg_info = "input", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<2x4x4xf32> {jax.result_info = "[0]"}, tensor<2x4xf32> {jax.result_info = "[1]"}, tensor<2x4x4xf32> {jax.result_info = "[2]"}) {
    %0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %2 = stablehlo.constant dense<2> : tensor<i32> loc(#loc2)
    %3 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %4 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %5 = stablehlo.constant dense<268> : tensor<i32> loc(#loc2)
    %6:7 = stablehlo.custom_call @lapack_sgesdd(%0, %1, %2, %3, %4, %5, %arg0) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 6, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>, tensor<32xi32>, tensor<268xf32>) loc(#loc2)
    %7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc2)
    %9 = stablehlo.compare  EQ, %6#4, %8,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc2)
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc2)
    %11 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc2)
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc2)
    %13 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc2)
    %14 = stablehlo.select %13, %6#1, %12 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc2)
    %15 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %16 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc2)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc2)
    %18 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %19 = stablehlo.select %18, %6#2, %17 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc2)
    %20 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %21 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc2)
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc2)
    %23 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %24 = stablehlo.select %23, %6#3, %22 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc2)
    return %19, %14, %24 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":355:0)
#loc2 = loc("jit(func)/jit(main)/svd[full_matrices=True compute_uv=True]"(#loc1))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xef\xa57\x01Q\x0f\x0b\x07\x13\x0b\x13\x13\x0f\x0b\x13\x13\x13#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x17\x0b\x13\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x03U\x0fo\x0b/\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x1f\x1f\x0b\x0b\x0b\x0b\x0b'\x0f\x17'O\x1f\x0f\x0b\x0b/\x1fOo\x01\x03\x0f\x035\x0f\x1b\x07\x07\x17\x07\x07\x0f\x07\x13\x1b\x1b\x1f\x13\x17\x13\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\xb6\x07\x1d+-\x05\x15\x1f\x03\x03\t\x97\x05\x17\x03\x03\t\x9d\x03\x03\x03\x9f\x11\x01\x05\x05\x19\x03\x03\x03y\x03\x03\x03}\x03\x03\t\xa3\x03\x07\x1b\x0f\x1d\x0f\x11\x1f\x05\x1b\x05\x1d\x05\x1f\x03\x0b#Y%e'g\x11u)w\x05!\x05#\x05%\x05'\x05)\x17/\x8e\x05\x01\x05+\x03\x03\x03{\x03\x03\x03\x7f\x03\x117\x819\x83;\x85=\x87?\x89A\x8bC\x8dE\x91\x05-\x05/\x051\x053\x055\x057\x059\x05;\x03\x03\x03\x95\x03\x05K\x99M\x9b\x05=\x05?\x03\x03\t\xa1\x1f!\x01\x1f#1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dA\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03[\r\x05]_ac\x1dC\x1dE\x1dG\x1dI#\x1b\x03\x07imq\r\x03Uk\x1dK\r\x03Uo\x1dM\r\x03Us\x1dO\x1dQ\x1dS\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x02\x00\x00\x00\x1f\x03\t\x04\x00\x00\x00\x1f\x03\t\x0c\x01\x00\x00\x0b\x05\x1dU\x1dW\x03\x01\x05\x01\x03\x0fQQQQQQS\x03\x03\x8f\x15\x03\x01\x19\x01\x03\x0fS\x93SSWWW\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x03\t\x00\x00\x00\x00\x1f)\x01\t\x07\x07\x01\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x11\t\x00\x00\xc0\x7f\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f51\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x02\x02)\x01\x13)\x07\t\x11\x11\t\x01\t)\x05\t\x11\t\x13\x1d)\x01\t\x1b)\x03\t\x13)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\x0b\x05)\x03\x81\x13)\x03b\x08\t)\x03\x01\r)\x03\r\r)\x03\t\r)\x03\x05\r)\x03\x01\x0f)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0f)\x05\t\x11\x07)\x03\t\x0f)\x03\r\x0f\x04~\x03\x05\x01\x11\x05\x19\x07\x03\x01\x05\t\x11\x05!\x05\x03Ak\x03\x05\x05\x03\x03\x01\x13\x03\x03\x03\x03\x01\x13\x03\x03\x03\x03\x011\x03\x03\x03\x03\x01\x15\x03\x03\x03\x03\x01\x15\x03\x03\x03\x03\x013\x03\x03\x0b\x07\x015\x0f\x05\x0b\x05\x05\x15\x1d\x1f\x0f\x03\x05\x07\t\x0b\r\x01\x03\x03\x01G\x03\x03\x05\x07\x01\x07\x03\x15\x03\x1d\r\x07\x01I\x03+\x05\x17\x1f\x05\x07\x01\x0b\x03-\x03!\x03\x03\x01\r\x03\x11\x05\x07\x01\x07\x03\x0b\x03%\x05\x07\x01O\x031\x03#\x07\x06\x01\x03\x0b\x07)\x11'\x05\x07\x01\x0b\x03\x17\x03!\x03\x03\x01\r\x03\x11\x05\x07\x01\x07\x03\x05\x03/\x05\x07\x01\x17\x03\x19\x03-\x07\x06\x01\x03\x05\x073\x131\x05\x07\x01\x0b\x03\x17\x03!\x03\x03\x01\r\x03\x11\x05\x07\x01\x07\x03\x05\x039\x05\x07\x01\x17\x03\x19\x037\x07\x06\x01\x03\x05\x07=\x15;\x0f\x04\x05\x075+?\x06\x03\x01\x05\x01\x00\xbe\nY\x1d\x03\x0f\x0b\t\t\t\x1b\x1d\r\x1b!+\x1b\x1f/!!)#\x1f\x19\x97y\x1f\x15\x1d\x15\x13%)\x13+\r\x15\x17\x1f\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(func)/jit(main)/svd[full_matrices=True compute_uv=True]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00jax.result_info\x00jax.arg_info\x00input\x00mhlo.sharding\x00{replicated}\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_sgesdd\x00",
    xla_call_module_version=6,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_19["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgesdd'],
    serialized_date=datetime.date(2023, 6, 22),
    inputs=(array([[[ 0.3445689867809981 ,  3.5114993759427104 ,
          4.702602090972179  , -0.2702264758497052 ],
        [ 2.209901632583705  , -2.6286702510632773 ,
          4.591276599385847  ,  3.4465035398844828 ],
        [-1.5083742421154478 ,  3.3225165204269635 ,
          1.2596205557926703 ,  3.524804355848018  ],
        [ 1.5118969169108838 ,  1.838885943509677  ,
          2.818520751293422  ,  3.06002540493494   ]],

       [[-2.4045510943950843 , -1.5657555633438576 ,
         -0.6061472334580296 , -0.23926156407779164],
        [ 4.087879920053448  , -3.2507640936811715 ,
         -2.2556577657517476 ,  6.090369998330348  ],
        [ 1.1165401344486945 ,  2.2134726894037247 ,
          5.225178515435584  ,  1.9794693474107725 ],
        [-4.127878192684534  , -0.37313660200336163,
          0.7893465897510026 , -2.0315217791342848 ]]]),),
    expected_outputs=(array([[[-0.5109626909166218 , -0.41744996156105785,
         -0.731253241567692  ,  0.1729779025790829 ],
        [-0.5623501368035175 ,  0.7608931604238581 ,
          0.03470920608540986,  0.32186828528169453],
        [-0.39585755254587435, -0.4954770291405409 ,
          0.6561880513437818 ,  0.4089212062978684 ],
        [-0.5157288533916834 , -0.03577207859388855,
          0.18297871183094833, -0.8362194085221047 ]],

       [[-0.12124821978030875, -0.30260506534356213,
         -0.5817463045715607 , -0.7451847292758064 ],
        [ 0.8877417367326685 , -0.15794001239879188,
         -0.3761180739267688 ,  0.2133184375808915 ],
        [ 0.03055221675864994,  0.9244545314395409 ,
         -0.3686107533067095 , -0.09260936183071355],
        [-0.44303503260363514, -0.16990864078317836,
         -0.619864940232637  ,  0.624994775612963  ]]]), array([[8.951386926411189  , 5.762891699811626  , 3.839104008889441  ,
        1.2696468971033248 ],
       [9.21500688857692   , 6.477297670883227  , 3.24626945855818   ,
        0.05112101994354587]]), array([[[-0.17890276924244797 , -0.2881812520705063  ,
         -0.7749616998111006  , -0.5332726590950898  ],
        [ 0.38712159387038353 , -0.8985113987184378  ,
          0.1397618670046424  ,  0.15258033445914954 ],
        [-0.23140697924040152 , -0.03708202130554661 ,
         -0.5045854966104308  ,  0.8309447696839614  ],
        [-0.8744034999217865  , -0.32901938548360005 ,
          0.35396957633060866 , -0.043246992182741084]],

       [[ 0.6276106632546885  , -0.26728735347872895 ,
         -0.22995258718774078 ,  0.6941067163520401  ],
        [ 0.2802931697592562  ,  0.4781137804659157  ,
          0.808362569504731   ,  0.19847646746808023 ],
        [ 0.6187014005224262  ,  0.47714095343944474 ,
         -0.3740686697560633  , -0.49961757159793246 ],
        [-0.3804591585793503  ,  0.6872417290515944  ,
         -0.3921025301835001  ,  0.47875384105714014 ]]])),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf64> {jax.arg_info = "input", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<2x4x4xf64> {jax.result_info = "[0]"}, tensor<2x4xf64> {jax.result_info = "[1]"}, tensor<2x4x4xf64> {jax.result_info = "[2]"}) {
    %0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %2 = stablehlo.constant dense<2> : tensor<i32> loc(#loc2)
    %3 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %4 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %5 = stablehlo.constant dense<268> : tensor<i32> loc(#loc2)
    %6:7 = stablehlo.custom_call @lapack_dgesdd(%0, %1, %2, %3, %4, %5, %arg0) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 6, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64>, tensor<2x4x4xf64>, tensor<2xi32>, tensor<32xi32>, tensor<268xf64>) loc(#loc2)
    %7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc2)
    %9 = stablehlo.compare  EQ, %6#4, %8,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc2)
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc2)
    %11 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc2)
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc2)
    %13 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc2)
    %14 = stablehlo.select %13, %6#1, %12 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc2)
    %15 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %16 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc2)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc2)
    %18 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %19 = stablehlo.select %18, %6#2, %17 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc2)
    %20 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %21 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc2)
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc2)
    %23 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %24 = stablehlo.select %23, %6#3, %22 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc2)
    return %19, %14, %24 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":355:0)
#loc2 = loc("jit(func)/jit(main)/svd[full_matrices=True compute_uv=True]"(#loc1))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xef\xa57\x01Q\x0f\x0b\x07\x13\x0b\x13\x13\x0f\x0b\x13\x13\x13#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x17\x0b\x13\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x03U\x0fo\x0b/\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x1f\x1f\x0b\x0b\x0b\x0b\x0b'\x0f\x17'O\x1f\x0f\x0b\x0b//Oo\x01\x03\x0f\x035\x0f\x1b\x07\x07\x17\x07\x07\x0f\x07\x13\x1b\x1b\x1f\x13\x17\x13\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\xc6\x07\x1d+-\x05\x15\x1f\x03\x03\t\x97\x05\x17\x03\x03\t\x9d\x03\x03\x03\x9f\x11\x01\x05\x05\x19\x03\x03\x03y\x03\x03\x03}\x03\x03\t\xa3\x03\x07\x1b\x0f\x1d\x0f\x11\x1f\x05\x1b\x05\x1d\x05\x1f\x03\x0b#Y%e'g\x11u)w\x05!\x05#\x05%\x05'\x05)\x17/\x8e\x05\x01\x05+\x03\x03\x03{\x03\x03\x03\x7f\x03\x117\x819\x83;\x85=\x87?\x89A\x8bC\x8dE\x91\x05-\x05/\x051\x053\x055\x057\x059\x05;\x03\x03\x03\x95\x03\x05K\x99M\x9b\x05=\x05?\x03\x03\t\xa1\x1f!\x01\x1f#1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dA\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03[\r\x05]_ac\x1dC\x1dE\x1dG\x1dI#\x1b\x03\x07imq\r\x03Uk\x1dK\r\x03Uo\x1dM\r\x03Us\x1dO\x1dQ\x1dS\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x02\x00\x00\x00\x1f\x03\t\x04\x00\x00\x00\x1f\x03\t\x0c\x01\x00\x00\x0b\x05\x1dU\x1dW\x03\x01\x05\x01\x03\x0fQQQQQQS\x03\x03\x8f\x15\x03\x01\x19\x01\x03\x0fS\x93SSWWW\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x03\t\x00\x00\x00\x00\x1f)\x01\t\x07\x07\x01\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x11\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f51\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x02\x02)\x01\x13)\x07\t\x11\x11\t\x01\x0b)\x05\t\x11\t\x13\x1d)\x01\t\x1b)\x03\t\x13)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\x0b\x05)\x03\x81\x13)\x03b\x08\t)\x03\x01\r)\x03\r\r)\x03\t\r)\x03\x05\r)\x03\x01\x0f)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0f)\x05\t\x11\x07)\x03\t\x0f)\x03\r\x0f\x04~\x03\x05\x01\x11\x05\x19\x07\x03\x01\x05\t\x11\x05!\x05\x03Ak\x03\x05\x05\x03\x03\x01\x13\x03\x03\x03\x03\x01\x13\x03\x03\x03\x03\x011\x03\x03\x03\x03\x01\x15\x03\x03\x03\x03\x01\x15\x03\x03\x03\x03\x013\x03\x03\x0b\x07\x015\x0f\x05\x0b\x05\x05\x15\x1d\x1f\x0f\x03\x05\x07\t\x0b\r\x01\x03\x03\x01G\x03\x03\x05\x07\x01\x07\x03\x15\x03\x1d\r\x07\x01I\x03+\x05\x17\x1f\x05\x07\x01\x0b\x03-\x03!\x03\x03\x01\r\x03\x11\x05\x07\x01\x07\x03\x0b\x03%\x05\x07\x01O\x031\x03#\x07\x06\x01\x03\x0b\x07)\x11'\x05\x07\x01\x0b\x03\x17\x03!\x03\x03\x01\r\x03\x11\x05\x07\x01\x07\x03\x05\x03/\x05\x07\x01\x17\x03\x19\x03-\x07\x06\x01\x03\x05\x073\x131\x05\x07\x01\x0b\x03\x17\x03!\x03\x03\x01\r\x03\x11\x05\x07\x01\x07\x03\x05\x039\x05\x07\x01\x17\x03\x19\x037\x07\x06\x01\x03\x05\x07=\x15;\x0f\x04\x05\x075+?\x06\x03\x01\x05\x01\x00\xbe\nY\x1d\x03\x0f\x0b\t\t\t\x1b\x1d\r\x1b!+\x1b\x1f/!!)#\x1f\x19\x97y\x1f\x15\x1d\x15\x13%)\x13+\r\x15\x17\x1f\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(func)/jit(main)/svd[full_matrices=True compute_uv=True]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00jax.result_info\x00jax.arg_info\x00input\x00mhlo.sharding\x00{replicated}\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_dgesdd\x00",
    xla_call_module_version=6,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_19["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgesdd'],
    serialized_date=datetime.date(2023, 6, 22),
    inputs=(array([[[ 1.6052934 +0.45878917j,  4.587192  -4.5177283j ,
          0.4177733 -1.9419309j , -2.2248359 -4.5042715j ],
        [-7.083374  -8.127356j  ,  2.7596245 -4.991001j  ,
         -0.52622825+5.033981j  , -0.35441273-1.8215327j ],
        [-0.7996552 -2.4052901j , -0.8506142 -3.164714j  ,
         -0.3090829 +2.2020447j ,  1.2367196 +2.8830793j ],
        [ 1.4633094 -0.5451007j , -3.7833478 +6.6770763j ,
         -3.1279542 -2.2322626j , -2.1099617 -2.9661314j ]],

       [[ 1.2560439 -5.4743752j , -2.0085676 +2.0063214j ,
         -0.8132642 -3.4407883j , -0.17360081+0.6419895j ],
        [ 2.3756726 +6.3315964j , -0.31447247-1.9387872j ,
          4.6732006 -4.286903j  ,  1.7702469 -1.4957623j ],
        [ 1.6918924 -0.52161306j,  0.49963537+4.7751374j ,
         -1.9243752 -4.5870543j ,  2.8829405 +1.7382988j ],
        [ 1.4884951 -0.44194785j, -1.3645276 -2.8733373j ,
         -0.39430943+2.4366508j , -0.76268387+5.2014065j ]]],
      dtype=complex64),),
    expected_outputs=(array([[[ 0.016725361+0.19210356j ,  0.5452691  +0.5572638j  ,
          0.41363996 +0.18964858j , -0.26152334 -0.28195143j ],
        [ 0.53678626 +0.64057267j , -0.21783225 -0.21288812j ,
          0.28426644 +0.30535883j ,  0.15201284 +0.10768581j ],
        [ 0.21286921 +0.154735j   ,  0.066471666-0.25652882j ,
         -0.4074613  -0.10356682j , -0.11794163 -0.81844836j ],
        [-0.39079374 -0.20583564j , -0.18335931 -0.4421772j  ,
          0.63489586 +0.19758748j ,  0.038680226-0.36351213j ]],

       [[-0.3178596  +0.39032036j , -0.1273337  -0.30841744j ,
          0.26394194 +0.26815224j , -0.21332254 -0.66947937j ],
        [-0.39241245 -0.60790956j , -0.14006221 +0.41040683j ,
         -0.0830612  -0.10184447j , -0.45091942 -0.2603987j  ],
        [-0.36103728 +0.2876153j  , -0.4965461  +0.10084368j ,
         -0.13752826 -0.6203828j  ,  0.35439825 -0.028546419j],
        [ 0.062335093-0.078214265j,  0.35014474 -0.5668197j  ,
         -0.42214075 -0.5090833j  , -0.2889288  -0.15894148j ]]],
      dtype=complex64), array([[15.135655  ,  9.373035  ,  7.444931  ,  0.41523397],
       [12.316969  ,  8.661011  ,  5.005059  ,  2.115905  ]],
      dtype=float32), array([[[-0.6537865  +0.j         , -0.20306697 -0.6166746j  ,
          0.29948467 +0.24257992j , -0.007604365+0.04945353j ],
        [ 0.52712685 +0.j         , -0.11291563 -0.7116954j  ,
         -0.089219   -0.36348897j , -0.23654723 -0.08269388j ],
        [-0.31538543 +0.j         , -0.014410622+0.15958191j ,
         -0.17958623 -0.13690898j , -0.6930434  -0.58613425j ],
        [-0.44185135 +0.j         ,  0.17604677 -0.050492246j,
         -0.4213856  -0.69485146j ,  0.22373371 +0.2465445j  ]],

       [[-0.64551586 +0.j         ,  0.32932255 -0.11672116j ,
         -0.093527466+0.6710145j  , -0.038554154+0.02716677j ],
        [ 0.4241116  +0.j         ,  0.031135002-0.539813j   ,
         -0.26271763 +0.22760014j , -0.63609654 -0.04817467j ],
        [-0.4577485  +0.j         , -0.15202768 +0.2734652j  ,
          0.18931003 -0.3297506j  , -0.7331101  -0.10269702j ],
        [ 0.44034657 +0.j         ,  0.29474002 +0.63307834j ,
          0.31271848 +0.4216674j  , -0.20595454 -0.020532424j]]],
      dtype=complex64)),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f32>> {jax.arg_info = "input", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "[0]"}, tensor<2x4xf32> {jax.result_info = "[1]"}, tensor<2x4x4xcomplex<f32>> {jax.result_info = "[2]"}) {
    %0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %2 = stablehlo.constant dense<2> : tensor<i32> loc(#loc2)
    %3 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %4 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %5 = stablehlo.constant dense<264> : tensor<i32> loc(#loc2)
    %6:8 = stablehlo.custom_call @lapack_cgesdd(%0, %1, %2, %3, %4, %5, %arg0) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 6, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>>, tensor<2x4x4xcomplex<f32>>, tensor<2xi32>, tensor<32xi32>, tensor<100xf32>, tensor<264xcomplex<f32>>) loc(#loc2)
    %7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc2)
    %9 = stablehlo.compare  EQ, %6#4, %8,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc2)
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc2)
    %11 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc2)
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc2)
    %13 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc2)
    %14 = stablehlo.select %13, %6#1, %12 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc2)
    %15 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %16 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc2)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc2)
    %18 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %19 = stablehlo.select %18, %6#2, %17 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc2)
    %20 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %21 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc2)
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc2)
    %23 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %24 = stablehlo.select %23, %6#3, %22 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc2)
    return %19, %14, %24 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":355:0)
#loc2 = loc("jit(func)/jit(main)/svd[full_matrices=True compute_uv=True]"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xf9\xa9=\x01S\x0f\x0b\x07\x13\x0b\x13\x0f\x0b\x13\x13\x13\x13#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x17\x0b\x13\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x13\x03W\x0fo/\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x1f\x1f\x0b\x0b\x0b\x0b\x0b\'\x0f\x17+O\x1f\x0f\x0b\x0b/\x1fO/o\x01\x03\x0f\x03;\x0f\x1b\x07\x07\x17\x07\x07\x0b\x07\x0f\x13\x0f\x1b\x1b\x1f\x13\x17\x17\x13\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\x1e\x08\x1d+-\x05\x15\x1f\x03\x03\t\x99\x05\x17\x03\x03\t\x9f\x11\x01\x05\x05\x19\x03\x03\x03{\x03\x03\x03\x7f\x03\x03\x03\xa5\x03\x03\t\xa7\x03\x07\x1b\r\x1d\r\x0f\x1f\x05\x1b\x05\x1d\x05\x1f\x03\x0b#[%g\'i\x0fw)y\x05!\x05#\x05%\x05\'\x05)\x17/\x8e\x05\x01\x05+\x03\x03\x03}\x03\x03\x03\x81\x03\x117\x839\x85;\x87=\x89?\x8bA\x8dC\x8fE\x93\x05-\x05/\x051\x053\x055\x057\x059\x05;\x03\x03\x03\x97\x03\x05K\x9bM\x9d\x05=\x05?\x03\x03\x03\xa1\x03\x03\t\xa3\x1f\'\x01\x1f)1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dA\x03\x03]\r\x05_ace\x1dC\x1dE\x1dG\x1dI#\x1f\x03\x07kos\r\x03Ym\x1dK\r\x03Yq\x1dM\r\x03Yu\x1dO\x1dQ\x1dS\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x02\x00\x00\x00\x1f\x03\t\x04\x00\x00\x00\x1f\x03\t\x08\x01\x00\x00\x0b\x05\x1dU\x1dW\x03\x01\x05\x01\x03\x0fSSSSSSU\x03\x03\x91\x15\x03\x01\x19\x01\x03\x11U\x95UUWWWW\x1f+!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x03\t\x00\x00\x00\x00\x1f/\x01\t\x07\x07\x01\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19\t\x00\x00\xc0\x7f\x1f9!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x15\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f;1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x02\x02)\x01\x13)\x07\t\x11\x11\x11\x01\t)\x05\t\x11\t\x13\x1d\x03\t\x1b)\x01\x11)\x03\t\x13)\x01\t)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\x0b\x05)\x03\x81\x13)\x03"\x03\t)\x03B\x08\x11)\x03\x01\r)\x03\r\r)\x03\t\r)\x03\x05\r)\x03\x01\x0f)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0f)\x05\t\x11\x07)\x03\t\x0f)\x03\r\x0f\x04\x82\x03\x05\x01\x11\x05\x19\x07\x03\x01\x05\t\x11\x05!\x05\x03Ck\x03\x05\x05\x03\x03\x01\x11\x03\x03\x03\x03\x01\x11\x03\x03\x03\x03\x011\x03\x03\x03\x03\x01\x13\x03\x03\x03\x03\x01\x13\x03\x03\x03\x03\x013\x03\x03\x0b\x07\x015\x11\x05\x0b\x05\x05\x17!#%\x0f\x03\x05\x07\t\x0b\r\x01\x03\x03\x01G\x03\x03\x05\x07\x01\x07\x03\x17\x03\x1f\r\x07\x01I\x031\x05\x17!\x05\x07\x01\x0b\x033\x03#\x03\x03\x01O\x03\x19\x05\x07\x01\x07\x03\x0b\x03\'\x05\x07\x01Q\x037\x03%\x07\x06\x01\x03\x0b\x07+\x11)\x05\x07\x01\x0b\x03\x1b\x03#\x03\x03\x01\x15\x03\x15\x05\x07\x01\x07\x03\x05\x031\x05\x07\x01\x17\x03\x1d\x03/\x07\x06\x01\x03\x05\x075\x133\x05\x07\x01\x0b\x03\x1b\x03#\x03\x03\x01\x15\x03\x15\x05\x07\x01\x07\x03\x05\x03;\x05\x07\x01\x17\x03\x1d\x039\x07\x06\x01\x03\x05\x07?\x15=\x0f\x04\x05\x077-A\x06\x03\x01\x05\x01\x00\xbe\nY\x1d\x03\x0f\x0b\t\t\t\x1b\x1d\r\x1b!+\x1b\x1f/!!)#\x1f\x19\x97y\x1f\x15\x1d\x15\x13%)\x13+\r\x15\x17\x1f\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(func)/jit(main)/svd[full_matrices=True compute_uv=True]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00jax.result_info\x00jax.arg_info\x00input\x00mhlo.sharding\x00{replicated}\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_cgesdd\x00',
    xla_call_module_version=6,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_19["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgesdd'],
    serialized_date=datetime.date(2023, 6, 22),
    inputs=(array([[[-0.9247611722912019-1.3615157109291343j ,
         -1.0663457975211892+4.73170030936092j   ,
         -1.4918732811689488-2.880861991859318j  ,
         -1.111356346434667 -2.869701609083459j  ],
        [-4.71291623424314  -1.5444012898828912j ,
         -5.232967549101415 -0.41287816948482003j,
          0.8905737109262459+9.50245186328329j   ,
          4.397722119094926 -6.842005210371916j  ],
        [ 1.9369405063276903+2.3496014107398917j ,
         -1.5609345742256133+4.2102103739897805j ,
          0.6596030248996742+5.195353435247212j  ,
          0.6315014498240328-1.2778849649354402j ],
        [ 5.115159214503849 -0.8856276268773485j ,
          1.3719934567460779-2.236070491368575j  ,
          0.4974504006612811-3.0462081956756637j ,
         -0.2620346712025989+4.424682727912594j  ]],

       [[-1.8242711798401063-0.8543252170262536j ,
         -2.724527211360488 +2.256038331706666j  ,
         -1.2777487543905157+0.976556823566376j  ,
          3.7438974536713223-0.4994301527847589j ],
        [-0.6359051102028691+2.730662301129662j  ,
         -1.2877728943263032+3.9124921723649053j ,
         -3.4618573226579894+1.7835551986994034j ,
         -1.4710491660152465+2.144967500163963j  ],
        [-3.6013691182532828+2.8182351980619034j ,
          2.0045935428878803+1.1146211993017152j ,
         -2.332213857689336 -0.874915651404938j  ,
         -1.5393862406530452+0.6852883119580928j ],
        [-2.674897392856801 +2.0724239502976984j ,
         -3.349108041292141 -1.0215359152295307j ,
          0.2603515088197114-1.9093411474619364j ,
          5.41252457188561  +8.634368042893094j  ]]]),),
    expected_outputs=(array([[[-0.04173678258633362+0.10796693731538423j ,
          0.6813428383170976 +0.34327979589293334j ,
         -0.41770229002865755+0.20028957850808823j ,
         -0.43443513665085287+0.034743251442636465j],
        [-0.8408468609573512 -0.1326064604464803j  ,
         -0.21674151028481228+0.015170556885426551j,
          0.17147327711152338+0.1531041615298256j  ,
         -0.3568765623609291 +0.21904384306708768j ],
        [-0.2673618144044136 +0.1379833616281103j  ,
         -0.17534278352558025-0.378992615769627j   ,
         -0.8179957069096054 -0.037506032257391624j,
          0.25392637883428526-0.009771014463849802j],
        [ 0.40569239968065934-0.08297706578106905j ,
         -0.4321527034953765 +0.09791545663574397j ,
         -0.23439193826962654-0.08427130532228161j ,
         -0.42348296145608866+0.6251448114949291j  ]],

       [[ 0.0272684373986653 +0.36312055550335454j ,
          0.270297713559288  +0.1304616587162563j  ,
          0.04286867013923673-0.4765859417602139j  ,
          0.7242702256119968 +0.15420620503522459j ],
        [-0.08593436615104483+0.1189990183325552j  ,
          0.37050286109355285-0.6240865462984536j  ,
          0.46902056878806025-0.34747949920770266j ,
         -0.31667671459632074-0.10340064369932994j ],
        [-0.07914843440873574-0.033487314943774035j,
          0.4110353453489128 -0.455090805566563j   ,
         -0.431131803930273  +0.40910871949632j    ,
          0.13782730102420274+0.49428280062680086j ],
        [-0.7478497242333215 +0.5283836938016964j  ,
         -0.08345894989956631+0.011807690067190268j,
         -0.27178304569905287+0.056526279406748176j,
         -0.09911954913441999-0.2598859654000683j  ]]]), array([[16.80132997488892  ,  7.744755614558116 ,  5.831221808032041 ,
         1.1195288361137765],
       [12.39537594694893  ,  8.218551160453814 ,  4.683634850274079 ,
         1.8820915363839188]]), array([[[ 0.35796251040556704 +0.j                  ,
          0.40179383774178046 -0.1269359716702074j  ,
         -0.0751486661300563  -0.6109813931761136j  ,
         -0.23049271148274278 +0.51209309438597j    ],
        [-0.4682861415308549  +0.j                  ,
         -0.013958972669495105+0.4210606476774211j  ,
         -0.6006888466394119  -0.3766516564723718j  ,
         -0.24264518623237025 -0.20408557153193485j ],
        [-0.6392945524816095  +0.j                  ,
          0.2432388607602898  -0.6679928485374246j  ,
          0.18168178910997038 -0.08126854868489754j ,
         -0.2030612067046724  -0.07124733621915219j ],
        [-0.49383540371426055 +0.j                  ,
         -0.010402968929686592+0.3734624991410737j  ,
          0.27994282704104956 +0.01949406216762731j ,
          0.32588905219319236 +0.6569569657140543j  ]],

       [[ 0.2666920370516844  +0.j                  ,
          0.24929033811571413 +0.27271089049933883j ,
         -0.012922512768026735+0.16383354123801513j ,
          0.07388201893235022 -0.8717175469187741j  ],
        [-0.6156140469162428  +0.j                  ,
         -0.33787077397020143 +0.37797154650923376j ,
         -0.3916043058726119  -0.2839601305776179j  ,
         -0.2714888604157674  -0.23729034093304682j ],
        [ 0.5618758038857617  +0.j                  ,
         -0.5788776267734554  -0.13833058883452312j ,
         -0.48995086206819644 +0.19259594116096765j ,
         -0.22967101640965012 -0.012926826751577613j],
        [-0.48393210641613593 +0.j                  ,
         -0.1049229605428438  -0.4911419972025977j  ,
         -0.07782239226461217 +0.6751317817750165j  ,
          0.11941657609231515 -0.19354808489959852j ]]])),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_func attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f64>> {jax.arg_info = "input", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "[0]"}, tensor<2x4xf64> {jax.result_info = "[1]"}, tensor<2x4x4xcomplex<f64>> {jax.result_info = "[2]"}) {
    %0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %1 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %2 = stablehlo.constant dense<2> : tensor<i32> loc(#loc2)
    %3 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %4 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %5 = stablehlo.constant dense<264> : tensor<i32> loc(#loc2)
    %6:8 = stablehlo.custom_call @lapack_zgesdd(%0, %1, %2, %3, %4, %5, %arg0) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 6, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>>, tensor<2x4x4xcomplex<f64>>, tensor<2xi32>, tensor<32xi32>, tensor<100xf64>, tensor<264xcomplex<f64>>) loc(#loc2)
    %7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc2)
    %9 = stablehlo.compare  EQ, %6#4, %8,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc2)
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc2)
    %11 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc2)
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc2)
    %13 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc2)
    %14 = stablehlo.select %13, %6#1, %12 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc2)
    %15 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %16 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc2)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc2)
    %18 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %19 = stablehlo.select %18, %6#2, %17 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc2)
    %20 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %21 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc2)
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc2)
    %23 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %24 = stablehlo.select %23, %6#3, %22 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc2)
    return %19, %14, %24 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":355:0)
#loc2 = loc("jit(func)/jit(main)/svd[full_matrices=True compute_uv=True]"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xf9\xa9=\x01S\x0f\x0b\x07\x13\x0b\x13\x0f\x0b\x13\x13\x13\x13#\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0b\x17\x0b\x13\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x13\x03W\x0fo/\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x17\x13\x0b\x13\x0b\x13\x0b\x0b\x0b\x1f\x1f\x1f\x1f\x0b\x0b\x0b\x0b\x0b\'\x0f\x17+O\x1f\x0f\x0b\x0b//OOo\x01\x03\x0f\x03;\x0f\x1b\x07\x07\x17\x07\x07\x0b\x07\x0f\x13\x0f\x1b\x1b\x1f\x13\x17\x17\x13\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02N\x08\x1d+-\x05\x15\x1f\x03\x03\t\x99\x05\x17\x03\x03\t\x9f\x11\x01\x05\x05\x19\x03\x03\x03{\x03\x03\x03\x7f\x03\x03\x03\xa5\x03\x03\t\xa7\x03\x07\x1b\r\x1d\r\x0f\x1f\x05\x1b\x05\x1d\x05\x1f\x03\x0b#[%g\'i\x0fw)y\x05!\x05#\x05%\x05\'\x05)\x17/\x8e\x05\x01\x05+\x03\x03\x03}\x03\x03\x03\x81\x03\x117\x839\x85;\x87=\x89?\x8bA\x8dC\x8fE\x93\x05-\x05/\x051\x053\x055\x057\x059\x05;\x03\x03\x03\x97\x03\x05K\x9bM\x9d\x05=\x05?\x03\x03\x03\xa1\x03\x03\t\xa3\x1f\'\x01\x1f)1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dA\x03\x03]\r\x05_ace\x1dC\x1dE\x1dG\x1dI#\x1f\x03\x07kos\r\x03Ym\x1dK\r\x03Yq\x1dM\r\x03Yu\x1dO\x1dQ\x1dS\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x02\x00\x00\x00\x1f\x03\t\x04\x00\x00\x00\x1f\x03\t\x08\x01\x00\x00\x0b\x05\x1dU\x1dW\x03\x01\x05\x01\x03\x0fSSSSSSU\x03\x03\x91\x15\x03\x01\x19\x01\x03\x11U\x95UUWWWW\x1f+!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x03\t\x00\x00\x00\x00\x1f/\x01\t\x07\x07\x01\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f9!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x15!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f;1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x02\x02)\x01\x13)\x07\t\x11\x11\x11\x01\x0b)\x05\t\x11\t\x13\x1d\x03\t\x1b)\x01\x11)\x03\t\x13)\x01\t)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\x0b\x05)\x03\x81\x13)\x03"\x03\t)\x03B\x08\x11)\x03\x01\r)\x03\r\r)\x03\t\r)\x03\x05\r)\x03\x01\x0f)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0f)\x05\t\x11\x07)\x03\t\x0f)\x03\r\x0f\x04\x82\x03\x05\x01\x11\x05\x19\x07\x03\x01\x05\t\x11\x05!\x05\x03Ck\x03\x05\x05\x03\x03\x01\x11\x03\x03\x03\x03\x01\x11\x03\x03\x03\x03\x011\x03\x03\x03\x03\x01\x13\x03\x03\x03\x03\x01\x13\x03\x03\x03\x03\x013\x03\x03\x0b\x07\x015\x11\x05\x0b\x05\x05\x17!#%\x0f\x03\x05\x07\t\x0b\r\x01\x03\x03\x01G\x03\x03\x05\x07\x01\x07\x03\x17\x03\x1f\r\x07\x01I\x031\x05\x17!\x05\x07\x01\x0b\x033\x03#\x03\x03\x01O\x03\x19\x05\x07\x01\x07\x03\x0b\x03\'\x05\x07\x01Q\x037\x03%\x07\x06\x01\x03\x0b\x07+\x11)\x05\x07\x01\x0b\x03\x1b\x03#\x03\x03\x01\x15\x03\x15\x05\x07\x01\x07\x03\x05\x031\x05\x07\x01\x17\x03\x1d\x03/\x07\x06\x01\x03\x05\x075\x133\x05\x07\x01\x0b\x03\x1b\x03#\x03\x03\x01\x15\x03\x15\x05\x07\x01\x07\x03\x05\x03;\x05\x07\x01\x17\x03\x1d\x039\x07\x06\x01\x03\x05\x07?\x15=\x0f\x04\x05\x077-A\x06\x03\x01\x05\x01\x00\xbe\nY\x1d\x03\x0f\x0b\t\t\t\x1b\x1d\r\x1b!+\x1b\x1f/!!)#\x1f\x19\x97y\x1f\x15\x1d\x15\x13%)\x13+\r\x15\x17\x1f\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(func)/jit(main)/svd[full_matrices=True compute_uv=True]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00jax.result_info\x00jax.arg_info\x00input\x00mhlo.sharding\x00{replicated}\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_zgesdd\x00',
    xla_call_module_version=6,
)  # End paste
