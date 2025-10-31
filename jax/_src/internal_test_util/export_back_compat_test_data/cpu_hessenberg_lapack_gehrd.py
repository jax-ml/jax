# Copyright 2024 The JAX Authors.
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

data_2024_08_31 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_31["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgehrd_ffi'],
    serialized_date=datetime.date(2024, 8, 30),
    inputs=(),
    expected_outputs=(array([[[ 0.7137638961069523  +2.4533812415320035e+00j,
         -0.3272236912989258  -3.2003874808591863e+00j,
         -3.065817294924296   +1.6978219378771007e+00j,
         -3.3971558164664     +2.6931967836060400e-01j],
        [ 6.346214936866542   +0.0000000000000000e+00j,
          2.083218259144673   -1.2191838498692813e+00j,
          1.9552582313969427  -3.3216313521481879e+00j,
          2.7451664155727293  +2.5460553490974451e+00j],
        [-0.16133388943502391 +3.6906265775683444e-01j,
         -4.698636849217318   +0.0000000000000000e+00j,
          2.5396292124414077  -3.3038474840573420e+00j,
          2.5410992366186456  +4.1958389320867528e-01j],
        [ 0.47396123039280513 +3.9524384493417053e-03j,
          0.058880409351504966-7.8934332132630333e-02j,
          0.9469634796174572  +0.0000000000000000e+00j,
         -3.130422531669044   -8.8070401977461810e-01j]],

       [[-6.7065483048969465  -4.1981401054281309e-01j,
         -0.21813268822330256 -3.8602920478381799e+00j,
         -0.8248337528620167  -2.9073223456990824e+00j,
         -3.597231249446879   +2.7626541679004930e+00j],
        [-6.812126638479044   +0.0000000000000000e+00j,
         -0.20651586628458585 -1.0948249928988512e+00j,
         -1.6675586608354327  +4.2553627621795744e+00j,
         -2.410110723267707   +3.6065122124698634e-01j],
        [ 0.038235817369200516-3.7823713529009173e-01j,
         -8.508141062606947   +0.0000000000000000e+00j,
          4.260708077719245   -6.8052584397204630e-02j,
          5.345997177836541   -1.1955161503390279e+00j],
        [-0.18541509608158574 -1.2016051097247168e-01j,
         -0.02698777746917469 -4.4847463691672246e-01j,
          6.149305574585603   +0.0000000000000000e+00j,
         -2.483131585236393   +2.8524912589603817e+00j]]]), array([[1.2286220194325557+0.5121060656500841j ,
        1.9529937219183482-0.23299856112387676j,
        1.5940499664125072-0.8044281430962614j ],
       [1.6682114302246909-0.11372755955977935j,
        1.4075913155446236-0.6008708461880701j ,
        1.5086928152468893-0.8609480935086589j ]])),
    mlir_module_text=r"""
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f64>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[(0.71376389610695234,2.4533812415320035), (-1.0686093138739379,-1.885041510645256), (3.2629529488994033,-0.87160041258342402), (2.4332168907311504,3.4960248990882183)], [(-1.450884474619478,-3.249935163088522), (0.53920035905924757,-5.0056840575116066), (0.13157186736298554,2.5015499854549939), (-1.2451270607408882,0.24345856951924827)], [(2.457366083193417,-2.3532935513245605), (-0.37595429769485644,1.5729223427874068), (3.5877693970448052,-0.30904304334212157), (-1.685615117470264,2.6148811836470265)], [(-3.6826776618664727,-1.5711608241015744), (-0.12407609317204518,-4.7137561145212281), (1.3298255603911306,-1.6739172003954141), (-2.6345448161870149,-0.089008252847513236)]], [[(-6.7065483048969465,-0.41981401054281309), (-2.1586544949255457,0.34815132010709054), (-5.1462488701272413,3.440817752555807), (1.0301804086076078,-0.6994760434270566)], [(4.551940883969797,-0.77472653800638502), (4.4485186470774796,-0.0024458890677252756), (0.66610302132250898,2.5976571401862039), (-5.0693248202533674,-5.7405538897950699)], [(0.14148406399087146,-4.3279346473525058), (-2.353557113110897,2.0880432773400326), (-3.2524452107293618,-0.42398740171508631), (3.7200566224095519,-0.56951559566037058)], [(-2.2001612082232613,-1.2218661647417151), (0.72437359623190833,8.6381970213061301), (0.72314820631775734,0.058458198280771749), (0.37498718985014962,2.1160469724471378)]]]> : tensor<2x4x4xcomplex<f64>> loc(#loc)
    %0:3 = stablehlo.custom_call @lapack_zgehrd_ffi(%cst) {mhlo.backend_config = {high = 4 : i32, low = 1 : i32}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x3xcomplex<f64>>, tensor<2xi32>) loc(#loc2)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc2)
    %2 = stablehlo.compare  EQ, %0#2, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc2)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc2)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc2)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc2)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc2)
    %cst_1 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc2)
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<complex<f64>>) -> tensor<2x3xcomplex<f64>> loc(#loc2)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc2)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x3xi1>, tensor<2x3xcomplex<f64>> loc(#loc2)
    return %6, %10 : tensor<2x4x4xcomplex<f64>>, tensor<2x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":697:13)
#loc2 = loc("jit(func)/jit(main)/hessenberg"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xe5\x9b5\x01Q\x0f\x07\x0b\x0b\x13\x0f\x0b\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0b\x13\x1b\x0b\x0b\x13\x13\x03K\x0b\x0b\x0b\x0bo\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b&\x10\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17\x17O/\x1f\x0f\x0b\x0b/OoO\x01\x05\x0b\x0f\x031\x1b\x07\x17\x07\x07\x0b\x07\x0f\x13\x0f\x17\x07\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x17\x13\x02"\x0f\x1d?A\x1f\x05\x15\x05\x17\x03\x03\x05\x8d\x11\x03\x05\x05\x19\x03\x03\x05\x93\x03\x03\x07\x95\x03\t\x15\x17\x19\x0b\x1b\x0b\r\x1d\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x03\x0b!Q#[%]\rg\'i\x05#\x05%\x05\'\x05)\x03\x03\x07k\x03\x13-m/o1q3Q5s7u9\x7f;\x81=\x85\x05+\x05-\x05/\x051\x053\x055\x057\x059\x05;\x05=\x17C\xe6\n\x1b\x05?\x03\x03\x07\x8b\x03\x05I\x8fK\x91\x05A\x05C\x03\x03\x05\x97\x03\x03\x05\x99\x03\x01\x1dE\x1dG\x1dI\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00#\x19\x03\x05_c\r\x05SaUW\x1dK\r\x05SeUW\x1dM\x1dO\x1dQ\x1f\x05\x02\x08p\t\xdba\'\xd7\xe6?\xa8\xff\'X\x86\xa0\x03@\x0c\xa2t\x14\x06\x19\xf1\xbfT.}I!)\xfe\xbf\x0fG_\x13\x87\x1a\n@\xae:g\x8c&\xe4\xeb\xbf\xeb\x1e\xcej:w\x03@N\xaf\xfc\xe6\xdb\xf7\x0b@\x9f<\x8c\xa3\xd26\xf7\xbf^\xaf\xbc\x01\xde\xff\t\xc0b\xd4\x84\x1c!A\xe1?\xd6{\xa4\n\xd2\x05\x14\xc0\xf0\xe6\xb2\xd1X\xd7\xc0?2\xb5\x86\xa3,\x03\x04@\x91\xf2SZ\n\xec\xf3\xbf\x04\x10\x02\x81\xa6)\xcf?8\xec\x8c\x8c\xaf\xa8\x03@\r\x9d\xc6\x91\x8b\xd3\x02\xc0\xb0\xf6X\x9d\xa2\x0f\xd8\xbf\xbd\xb6V\x9e\xb0*\xf9?7-\x0fq\xc0\xb3\x0c@{|\ry\\\xc7\xd3\xbf\x04\xd9\xb2\x8eG\xf8\xfa\xbf\x9b\x84u\xd3F\xeb\x04@\xf4h\xbb\xb4\x1fv\r\xc0\xdc\\D\x88y#\xf9\xbf\x9a\xaecjs\xc3\xbf\xbf<\xc1\x04\xe2\xe2\xda\x12\xc0\x89<\xb4*\xf7F\xf5?\x1b\x90\xfef]\xc8\xfa\xbf\xdc\xf4\x8a;\x8c\x13\x05\xc0\xf8\xdd\r\xaf>\xc9\xb6\xbfvN\x1af\x81\xd3\x1a\xc0Z\xc6k\x95;\xde\xda\xbf\x87\x8c\xd8\xa5\xecD\x01\xc0\xdd\xd3zy\x1cH\xd6?\x04\x18\x89C\xc2\x95\x14\xc0\x8c\xc95u\xcb\x86\x0b@\x881\xbfs\x9e{\xf0?\x92Y[\x95\x1bb\xe6\xbf\x06\xe7\xb7\xfd/5\x12@L\x95\x02O\x8f\xca\xe8\xbf2`\xe3xH\xcb\x11@>\xda\xc6\xb1f\td\xbfZ\x1a\x8bH\xb7P\xe5?\xa8\x90zw\x00\xc8\x04@<(\xef\x15\xfdF\x14\xc0\xb4aF\xc2S\xf6\x16\xc0\xc1{\xdfY&\x1c\xc2?\xcfj\xa6\x19\xceO\x11\xc0\xc4\xa2p\xc0\x15\xd4\x02\xc0\xfcv\xa6\x08P\xb4\x00@^\xea\xa0\xfe\x01\x05\n\xc0^\x11\x12\x0e\x9c"\xdb\xbfR#\xe4\x0b\xad\xc2\r@F\x8b=\xc5x9\xe2\xbfZ\xf9\x99\x1e\xee\x99\x01\xc0My\x1a\x89\xc3\x8c\xf3\xbf\xd1\xdc<\x89\x11.\xe7?2\xd4\x8d\xc2\xc1F!@mw\t\xb5\x07$\xe7?G\x16\x99\xa3;\xee\xad?M\xd24E\xca\xff\xd7?\xa2\xae\xfb\x08\xaa\xed\x00@\x0b\x03\x1dS\x1dU\x05\x01\r\x05wy{}\x1dW\x13\x0b\x11\x1dY\x13\x0b\x05\x03\x03Y\x03\x03\x83\x15\x03\x01\x01\x01\x03\x07Y\x87\x89\x1f\x1f!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x17\t\x00\x00\x00\x00\x1f#\x01\t\x07\x07\x01\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x13!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f-1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x0f\x01)\x05\t\r\x0f\x1b\x1d\x03\x1b\x13)\x01\x0f)\x03\t\x0b)\x01\x0b\x11\x01\x05\x05\t\x0b)\x03\r\x11)\x03\t\x11)\x03\x05\x11)\x03\x01\r)\x03\t\x07)\x07\t\x05\x05\x07)\x03\x05\r)\x07\t\x11\x11\x07)\x03\r\r)\x05\t\x05\x07)\x05\t\r\x07)\x03\t\r\x042\x02\x05\x01\x11\x03\x13\x07\x03\x01\x05\t\x11\x03\x1f\x07\x03#A\x05\x03\x03)\x03\x05\x0b\x07\x01+\x07\x05\t\x15\x03\x01\x05\x03\x01E\x03\x17\x03\x07\x01\t\x03\x15\x03\t\r\x07\x01G\x03%\x05\x07\x0b\x03\x07\x01\x0f\x03\'\x03\r\x05\x03\x01\x11\x03\x13\x03\x07\x01\t\x03\x05\x03\x11\x03\x07\x01M\x03+\x03\x0f\x07\x06\x01\x03\x05\x07\x15\x03\x13\x03\x07\x01\x0f\x03/\x03\r\x05\x03\x01\x11\x03\x13\x03\x07\x01\t\x03\t\x03\x1b\x03\x07\x01O\x031\x03\x19\x07\x06\x01\x03\t\x07\x1f\x05\x1d\x0f\x04\x03\x05\x17!\x06\x03\x01\x05\x01\x00\x82\n[\t\x0b%\x03\x0f\x0b\t\t\x11#!+\x1bi?\x1f/!)!)#\x1f\x19\x1f\x15\x1d\x15\x13%)9\x13\r+\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00broadcast_dimensions\x00value\x00sym_name\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(func)/jit(main)/hessenberg\x00third_party/py/jax/tests/export_back_compat_test.py\x00compare_type\x00comparison_direction\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00main\x00public\x00\x00lapack_zgehrd_ffi\x00high\x00low\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_31["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgehrd_ffi'],
    serialized_date=datetime.date(2024, 8, 30),
    inputs=(),
    expected_outputs=(array([[[ 5.2023945 -0.878671j   , -2.8841915 -0.47488597j ,
          1.3024182 +0.6651789j  ,  4.9291854 -1.9147056j  ],
        [ 6.3457894 +0.j         ,  1.6869383 -4.6557646j  ,
          0.88955224-1.7617276j  ,  2.9149916 +4.342665j   ],
        [-0.2465725 -0.5776757j  , -5.3007755 +0.j         ,
         -0.9786545 -0.0633831j  , -1.3690261 -1.5921416j  ],
        [ 0.35462287+0.35993803j , -0.38403815-0.46558398j ,
          2.8020499 +0.j         ,  0.5636822 -6.218306j   ]],

       [[ 1.0687767 -3.88293j    , -4.0144    -2.5885587j  ,
          5.3900986 -0.8850739j  ,  2.079677  +3.5515747j  ],
        [ 7.5675693 +0.j         ,  0.5971966 -3.6699948j  ,
          2.246994  -1.0858283j  , -0.8870981 -0.022960603j],
        [-0.2183232 +0.10552277j ,  5.860886  +0.j         ,
         -5.091036  +6.2841997j  ,  5.008773  +1.8765848j  ],
        [ 0.1378771 +0.427895j   ,  0.63263524-0.3470098j  ,
          6.4528017 +0.j         , -4.233642  -0.84165764j ]]],
      dtype=complex64), array([[1.0933675-0.3605358j  , 1.1987956+0.5659744j  ,
        1.9999101-0.013409062j],
       [1.4504763-0.44363326j , 1.3110259-0.07426627j ,
        1.227255 +0.97383535j ]], dtype=complex64)),
    mlir_module_text=r"""
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f32>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[(5.20239449,-0.87867099), (-0.211780012,-0.923053801), (-5.25181627,1.90887547), (-1.61342144,-1.98000157)], [(-5.924900e-01,2.28788424), (-1.74142945,-3.25563216), (3.08765078,-3.25260139), (-3.35189271,-0.571629047)], [(3.032444,3.44394636), (1.22205484,0.808871626), (2.58686161,-7.47011566), (1.9139297,-2.57945323)], [(-3.28396916,-1.68601465), (2.62759161,-0.953538239), (-2.78763294,-0.0429570749), (0.426534384,-0.211706176)]], [[(1.06877673,-3.882930e+00), (-0.0192247611,5.96663713), (1.15329504,-5.0599103), (-1.76508892,-1.98541296)], [(-3.40901089,3.35722542), (-6.13531398,2.55851483), (-4.8095789,0.164206699), (-0.247624069,-3.13545418)], [(2.04217815,-1.89123917), (-1.18974173,-1.69466627), (-2.28673625,-0.487834573), (3.01541853,-1.85637176)], [(-2.9499588,-4.23393869), (8.44624137,5.57274485), (-1.09048736,2.4864223), (-0.305431545,-0.298133373)]]]> : tensor<2x4x4xcomplex<f32>> loc(#loc)
    %0:3 = stablehlo.custom_call @lapack_cgehrd_ffi(%cst) {mhlo.backend_config = {high = 4 : i32, low = 1 : i32}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x3xcomplex<f32>>, tensor<2xi32>) loc(#loc2)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc2)
    %2 = stablehlo.compare  EQ, %0#2, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc2)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %cst_0 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc2)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc2)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc2)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc2)
    %cst_1 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc2)
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<complex<f32>>) -> tensor<2x3xcomplex<f32>> loc(#loc2)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc2)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>> loc(#loc2)
    return %6, %10 : tensor<2x4x4xcomplex<f32>>, tensor<2x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":697:13)
#loc2 = loc("jit(func)/jit(main)/hessenberg"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xe5\x9b5\x01Q\x0f\x07\x0b\x0b\x13\x0f\x0b\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0b\x13\x1b\x0b\x0b\x13\x13\x03K\x0b\x0b\x0b\x0bo\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b&\x08\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17\x17O/\x1f\x0f\x0b\x0b//oO\x01\x05\x0b\x0f\x031\x1b\x07\x17\x07\x07\x0b\x07\x0f\x13\x0f\x17\x07\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x17\x13\x02\x02\x0b\x1d?A\x1f\x05\x15\x05\x17\x03\x03\x05\x8d\x11\x03\x05\x05\x19\x03\x03\x05\x93\x03\x03\x07\x95\x03\t\x15\x17\x19\x0b\x1b\x0b\r\x1d\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x03\x0b!Q#[%]\rg\'i\x05#\x05%\x05\'\x05)\x03\x03\x07k\x03\x13-m/o1q3Q5s7u9\x7f;\x81=\x85\x05+\x05-\x05/\x051\x053\x055\x057\x059\x05;\x05=\x17C\xe6\n\x1b\x05?\x03\x03\x07\x8b\x03\x05I\x8fK\x91\x05A\x05C\x03\x03\x05\x97\x03\x03\x05\x99\x03\x01\x1dE\x1dG\x1dI\x1f\x1d1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00#\x19\x03\x05_c\r\x05SaUW\x1dK\r\x05SeUW\x1dM\x1dO\x1dQ\x1f\x05\x02\x04\x04z\xa6@\x95\xf0`\xbf\xdc\xdcX\xbeAMl\xbf\xe1\x0e\xa8\xc0\x08V\xf4?\x98\x84\xce\xbf\xb1p\xfd\xbfm\xad\x17\xbf\xb2l\x12@)\xe7\xde\xbfG\\P\xc0\x12\x9cE@\x9f*P\xc0i\x85V\xc0HV\x12\xbf\x90\x13B@\x9ei\\@Kl\x9c?6\x12O?$\x8f%@0\x0b\xef\xc0\xa6\xfb\xf4?\xc3\x15%\xc0\x8d,R\xc0T\xcf\xd7\xbfv*(@\x15\x1bt\xbf\x94h2\xc0\xc2\xf3/\xbd\xb7b\xda>\x81\xc9X\xbe\xad\xcd\x88?\xed\x81x\xc0?}\x9d\xbc\xb1\xee\xbe@,\x9f\x93?\xc9\xea\xa1\xc0o\xee\xe1\xbf\x03"\xfe\xbf<-Z\xc0\xc8\xdcV@~T\xc4\xc0\xb5\xbe#@\x12\xe8\x99\xc0\xcd%(>*\x91}\xbeH\xabH\xc0\x0c\xb3\x02@ \x14\xf2\xbfuI\x98\xbf\xd3\xea\xd8\xbf\xe3Y\x12\xc0t\xc5\xf9\xbe\x9e\xfc@@\x97\x9d\xed\xbf \xcc<\xc0m|\x87\xc0\xce#\x07A\xedS\xb2@\x17\x95\x8b\xbf\x8b!\x1f@\x86a\x9c\xbe\xf0\xa4\x98\xbe\x0b\x03\x1dS\x1dU\x05\x01\r\x05wy{}\x1dW\x13\x0b\x11\x1dY\x13\x0b\x05\x03\x03Y\x03\x03\x83\x15\x03\x01\x01\x01\x03\x07Y\x87\x89\x1f\x1f!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f!\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x17\t\x00\x00\x00\x00\x1f#\x01\t\x07\x07\x01\x1f)\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x13\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f-1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x0f\x01)\x05\t\r\x0f\x1b\x1d\x03\x1b\x13)\x01\x0f)\x03\t\x0b)\x01\x0b\x11\x01\x05\x05\t\t)\x03\r\x11)\x03\t\x11)\x03\x05\x11)\x03\x01\r)\x03\t\x07)\x07\t\x05\x05\x07)\x03\x05\r)\x07\t\x11\x11\x07)\x03\r\r)\x05\t\x05\x07)\x05\t\r\x07)\x03\t\r\x042\x02\x05\x01\x11\x03\x13\x07\x03\x01\x05\t\x11\x03\x1f\x07\x03#A\x05\x03\x03)\x03\x05\x0b\x07\x01+\x07\x05\t\x15\x03\x01\x05\x03\x01E\x03\x17\x03\x07\x01\t\x03\x15\x03\t\r\x07\x01G\x03%\x05\x07\x0b\x03\x07\x01\x0f\x03\'\x03\r\x05\x03\x01\x11\x03\x13\x03\x07\x01\t\x03\x05\x03\x11\x03\x07\x01M\x03+\x03\x0f\x07\x06\x01\x03\x05\x07\x15\x03\x13\x03\x07\x01\x0f\x03/\x03\r\x05\x03\x01\x11\x03\x13\x03\x07\x01\t\x03\t\x03\x1b\x03\x07\x01O\x031\x03\x19\x07\x06\x01\x03\t\x07\x1f\x05\x1d\x0f\x04\x03\x05\x17!\x06\x03\x01\x05\x01\x00\x82\n[\t\x0b%\x03\x0f\x0b\t\t\x11#!+\x1bi?\x1f/!)!)#\x1f\x19\x1f\x15\x1d\x15\x13%)9\x13\r+\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00broadcast_dimensions\x00value\x00sym_name\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(func)/jit(main)/hessenberg\x00third_party/py/jax/tests/export_back_compat_test.py\x00compare_type\x00comparison_direction\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00main\x00public\x00\x00lapack_cgehrd_ffi\x00high\x00low\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_31["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgehrd_ffi'],
    serialized_date=datetime.date(2024, 8, 30),
    inputs=(),
    expected_outputs=(array([[[-3.5237675  , -6.1161256  , -0.549011   , -4.7706876  ],
        [ 5.8401766  ,  3.424213   ,  0.3059119  ,  2.3492367  ],
        [ 0.63135445 ,  2.7238827  , -0.106214404, -0.82470125 ],
        [-0.27146497 ,  0.09917235 ,  0.2545611  , -0.5113605  ]],

       [[ 4.297168   , -1.8758869  ,  0.33528137 ,  5.867136   ],
        [-7.129698   , -3.3118155  , -1.3492918  , -2.8959117  ],
        [-0.7266852  , -3.506432   ,  4.77164    , -4.0780373  ],
        [ 0.14084078 ,  0.3389384  ,  2.3910007  , -0.79807365 ]]],
      dtype=float32), array([[1.3584172, 1.9805213, 0.       ],
       [1.2920669, 1.7939165, 0.       ]], dtype=float32)),
    mlir_module_text=r"""
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-3.52376747, -0.758410036, 4.85795927, -6.0243597], [-2.09321976, -1.27957773, -0.956288218, -1.11928439], [-5.00878525, 0.51314038, 3.53047514, -2.91282868], [2.15363932, 0.635739565, -0.21264787, 0.555740714]], [[4.29716778, -3.86209464, -2.39021468, 4.17441607], [2.08234859, -1.03958249, 4.09025383, 5.22586823], [-6.69425774, 3.43749118, -0.691099107, 1.59547663], [1.29743183, -2.00156212, 3.08750296, 2.39243269]]]> : tensor<2x4x4xf32> loc(#loc)
    %0:3 = stablehlo.custom_call @lapack_sgehrd_ffi(%cst) {mhlo.backend_config = {high = 4 : i32, low = 1 : i32}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x3xf32>, tensor<2xi32>) loc(#loc2)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc2)
    %2 = stablehlo.compare  EQ, %0#2, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc2)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc2)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc2)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc2)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc2)
    %cst_1 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc2)
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc2)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc2)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc2)
    return %6, %10 : tensor<2x4x4xf32>, tensor<2x3xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":697:13)
#loc2 = loc("jit(func)/jit(main)/hessenberg"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xe3\x9b3\x01Q\x0f\x07\x0b\x0b\x13\x0f\x0b\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0b\x13\x1b\x0b\x0b\x13\x13\x03K\x0b\x0b\x0b\x0bo\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b&\x04\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17\x17O/\x1f\x0f\x0b\x0b/\x1foO\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x07\x0f\x13\x0f\x17\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x17\x13\x02\xea\x08\x1d?A\x1f\x05\x15\x05\x17\x03\x03\x05\x8d\x11\x03\x05\x05\x19\x03\x03\x05\x93\x03\x03\x07\x95\x03\t\x15\x17\x19\x0b\x1b\x0b\r\x1d\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x03\x0b!Q#[%]\rg\'i\x05#\x05%\x05\'\x05)\x03\x03\x07k\x03\x13-m/o1q3Q5s7u9\x7f;\x81=\x85\x05+\x05-\x05/\x051\x053\x055\x057\x059\x05;\x05=\x17C\xe6\n\x1b\x05?\x03\x03\x07\x8b\x03\x05I\x8fK\x91\x05A\x05C\x03\x03\x05\x97\x03\x03\x05\x99\x03\x01\x1dE\x1dG\x1dI\x1f\x1b1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00#\x19\x03\x05_c\r\x05SaUW\x1dK\r\x05SeUW\x1dM\x1dO\x1dQ\x1f\x05\x02\x02h\x85a\xc0)\'B\xbfgt\x9b@\x8e\xc7\xc0\xc0P\xf7\x05\xc04\xc9\xa3\xbfN\xcft\xbf\xb6D\x8f\xbf\xf8G\xa0\xc0+]\x03?N\xf3a@\xc9k:\xc0:\xd5\t@\xd4\xbf"?]\xc0Y\xbe\x06E\x0e?f\x82\x89@\x8f,w\xc0G\xf9\x18\xc0\xd1\x94\x85@3E\x05@\n\x11\x85\xbf\\\xe3\x82@P:\xa7@\\7\xd6\xc0\xdb\xff[@\xdf\xeb0\xbf\x948\xcc??\x12\xa6?\x98\x19\x00\xc0\xa6\x99E@\x9e\x1d\x19@\x0b\x03\x1dS\x1dU\x05\x01\r\x05wy{}\x1dW\x13\x0b\x11\x1dY\x13\x0b\x05\x03\x03Y\x03\x03\x83\x15\x03\x01\x01\x01\x03\x07Y\x87\x89\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x17\t\x00\x00\x00\x00\x1f!\x01\t\x07\x07\x01\x1f\'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x13\t\x00\x00\xc0\x7f\x1f+1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x0f\x01)\x05\t\r\x0f\x1b\x1d\t\x13)\x01\x0f)\x03\t\x0b)\x01\x0b\x11\x01\x05\x05\t)\x03\r\x11)\x03\t\x11)\x03\x05\x11)\x03\x01\r)\x03\t\x07)\x07\t\x05\x05\x07)\x03\x05\r)\x07\t\x11\x11\x07)\x03\r\r)\x05\t\x05\x07)\x05\t\r\x07)\x03\t\r\x042\x02\x05\x01\x11\x03\x13\x07\x03\x01\x05\t\x11\x03\x1f\x07\x03#A\x05\x03\x03)\x03\x05\x0b\x07\x01+\x07\x05\t\x15\x03\x01\x05\x03\x01E\x03\x17\x03\x07\x01\t\x03\x15\x03\t\r\x07\x01G\x03#\x05\x07\x0b\x03\x07\x01\x0f\x03%\x03\r\x05\x03\x01\x11\x03\x13\x03\x07\x01\t\x03\x05\x03\x11\x03\x07\x01M\x03)\x03\x0f\x07\x06\x01\x03\x05\x07\x15\x03\x13\x03\x07\x01\x0f\x03-\x03\r\x05\x03\x01\x11\x03\x13\x03\x07\x01\t\x03\t\x03\x1b\x03\x07\x01O\x03/\x03\x19\x07\x06\x01\x03\t\x07\x1f\x05\x1d\x0f\x04\x03\x05\x17!\x06\x03\x01\x05\x01\x00\x82\n[\t\x0b%\x03\x0f\x0b\t\t\x11#!+\x1bi?\x1f/!)!)#\x1f\x19\x1f\x15\x1d\x15\x13%)9\x13\r+\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00broadcast_dimensions\x00value\x00sym_name\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(func)/jit(main)/hessenberg\x00third_party/py/jax/tests/export_back_compat_test.py\x00compare_type\x00comparison_direction\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00main\x00public\x00\x00lapack_sgehrd_ffi\x00high\x00low\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_31["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgehrd_ffi'],
    serialized_date=datetime.date(2024, 8, 30),
    inputs=(),
    expected_outputs=(array([[[ 0.9307390587491866  , -0.35692982324474015 ,
         -0.1271353200176119  , -0.43952156917870067 ],
        [ 2.2633695323673964  ,  0.9965090965971986  ,
         -1.3244131008423046  ,  1.7324542351344163  ],
        [ 0.24558316247256504 ,  2.922776762811796   ,
          3.630059093036474   ,  1.4330664619737252  ],
        [-0.2856727718012896  , -0.4601276537179077  ,
         -2.8602148466873802  ,  1.9928744545245372  ]],

       [[-0.5351339571818844  ,  5.753313169426148   ,
          0.1385440281649789  ,  2.8445493054193807  ],
        [ 4.676815781213274   ,  2.920688567170204   ,
         -2.610159425457712   ,  4.0359806870679655  ],
        [-0.16963242599901043 , -2.342935131066633   ,
          4.179999589709703   , -0.6810604472011716  ],
        [ 0.030645999613174775, -0.2271804227402005  ,
         -2.2755242550977153  ,  0.7136684502626782  ]]]), array([[1.751436143556826 , 1.6505497938190505, 0.                ],
       [1.9422862513069978, 1.9018440331997255, 0.                ]])),
    mlir_module_text=r"""
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.93073905874918661, 0.18483901505653183, -0.11804347408930886, -0.53725392025434981], [-1.700777672846173, 1.3531570270421245, -2.4375034855727518, 2.2945174202226699], [-0.97352780716312858, -0.8319788592736328, 2.4986640885328582, -2.8118637941861766], [1.1324489199416958, -1.9301638714393787, 1.5523821278819048, 2.7676215285832253]], [[-0.53513395718188439, -5.2137633671981938, 2.9644475919777618, 2.2891023676266191], [-4.4068992105328642, 1.2751848926168665, -2.8947257279736456, -2.6817410994805888], [1.5408926111334784, -0.85423691880254915, 6.4217874587762065, -0.43997818045540715], [-0.27837952612324207, 1.1509460853774549, -0.21686805683301608, 0.11738425574951133]]]> : tensor<2x4x4xf64> loc(#loc)
    %0:3 = stablehlo.custom_call @lapack_dgehrd_ffi(%cst) {mhlo.backend_config = {high = 4 : i32, low = 1 : i32}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x3xf64>, tensor<2xi32>) loc(#loc2)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc2)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc2)
    %2 = stablehlo.compare  EQ, %0#2, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc2)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc2)
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc2)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc2)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc2)
    %6 = stablehlo.select %5, %0#0, %4 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc2)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc2)
    %cst_1 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc2)
    %8 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc2)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc2)
    %10 = stablehlo.select %9, %0#1, %8 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc2)
    return %6, %10 : tensor<2x4x4xf64>, tensor<2x3xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":697:13)
#loc2 = loc("jit(func)/jit(main)/hessenberg"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xe3\x9b3\x01Q\x0f\x07\x0b\x0b\x13\x0f\x0b\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x17\x0b\x13\x1b\x0b\x0b\x13\x13\x03K\x0b\x0b\x0b\x0bo\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b&\x08\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17\x17O/\x1f\x0f\x0b\x0b//oO\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x07\x0f\x13\x0f\x17\x13\x13\x13\x13\x13\x1b\x13\x1b\x13\x17\x17\x13\x02\xfa\n\x1d?A\x1f\x05\x15\x05\x17\x03\x03\x05\x8d\x11\x03\x05\x05\x19\x03\x03\x05\x93\x03\x03\x07\x95\x03\t\x15\x17\x19\x0b\x1b\x0b\r\x1d\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x03\x0b!Q#[%]\rg\'i\x05#\x05%\x05\'\x05)\x03\x03\x07k\x03\x13-m/o1q3Q5s7u9\x7f;\x81=\x85\x05+\x05-\x05/\x051\x053\x055\x057\x059\x05;\x05=\x17C\xe6\n\x1b\x05?\x03\x03\x07\x8b\x03\x05I\x8fK\x91\x05A\x05C\x03\x03\x05\x97\x03\x03\x05\x99\x03\x01\x1dE\x1dG\x1dI\x1f\x1b1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00#\x19\x03\x05_c\r\x05SaUW\x1dK\r\x05SeUW\x1dM\x1dO\x1dQ\x1f\x05\x02\x04\xa6\x00NG\x9d\xc8\xed?\xf2\xa8X\n\xce\xa8\xc7?#E\xb8\xdc\x188\xbe\xbf\xb8|$"/1\xe1\xbf\xc4B*\xa6b6\xfb\xbf\xe8\xf9\x97\xfb\x87\xa6\xf5?)^\xd3\xd3\x01\x80\x03\xc0T\xab\xff\xf2+[\x02@4d\xb0\xc9#\'\xef\xbf~e\xf1 \x92\x9f\xea\xbf\x96\x81\xff\x98C\xfd\x03@W\xb0\xe6q\xb2~\x06\xc0F\xa48\xc2\x82\x1e\xf2?\xcc\x0b\xfc\x82\xf3\xe1\xfe\xbf\xdc\\b\xa4\x8e\xd6\xf8?\x8c\xc3\x87\xc1\x16$\x06@\x83h\xa2?\xd1\x1f\xe1\xbf\xdc\xcb\xbc\xc8\xe4\xda\x14\xc0\xe6\x00\x92L0\xb7\x07@Q8\xf1\xe6\x14P\x02@\t\x07\xc8/\xaa\xa0\x11\xc0\x8eH"F(g\xf4?\xf5Jd\xf6e(\x07\xc0\x9e\xddt\xad4t\x05\xc0\x1cv\xb7\x02\x7f\xa7\xf8?B^\xa9\xa9\xe8U\xeb\xbf\x1e:5\r\xe9\xaf\x19@\xa2\x9c\x00>\x9a(\xdc\xbf\xc1\xd1$\\\xf8\xd0\xd1\xbf}|BqFj\xf2?6\x8b\xd2\x1dU\xc2\xcb\xbfdk\x82\x03\xe5\x0c\xbe?\x0b\x03\x1dS\x1dU\x05\x01\r\x05wy{}\x1dW\x13\x0b\x11\x1dY\x13\x0b\x05\x03\x03Y\x03\x03\x83\x15\x03\x01\x01\x01\x03\x07Y\x87\x89\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x17\t\x00\x00\x00\x00\x1f!\x01\t\x07\x07\x01\x1f\'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x13\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f+1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x0f\x01)\x05\t\r\x0f\x1b\x1d\x0b\x13)\x01\x0f)\x03\t\x0b)\x01\x0b\x11\x01\x05\x05\t)\x03\r\x11)\x03\t\x11)\x03\x05\x11)\x03\x01\r)\x03\t\x07)\x07\t\x05\x05\x07)\x03\x05\r)\x07\t\x11\x11\x07)\x03\r\r)\x05\t\x05\x07)\x05\t\r\x07)\x03\t\r\x042\x02\x05\x01\x11\x03\x13\x07\x03\x01\x05\t\x11\x03\x1f\x07\x03#A\x05\x03\x03)\x03\x05\x0b\x07\x01+\x07\x05\t\x15\x03\x01\x05\x03\x01E\x03\x17\x03\x07\x01\t\x03\x15\x03\t\r\x07\x01G\x03#\x05\x07\x0b\x03\x07\x01\x0f\x03%\x03\r\x05\x03\x01\x11\x03\x13\x03\x07\x01\t\x03\x05\x03\x11\x03\x07\x01M\x03)\x03\x0f\x07\x06\x01\x03\x05\x07\x15\x03\x13\x03\x07\x01\x0f\x03-\x03\r\x05\x03\x01\x11\x03\x13\x03\x07\x01\t\x03\t\x03\x1b\x03\x07\x01O\x03/\x03\x19\x07\x06\x01\x03\t\x07\x1f\x05\x1d\x0f\x04\x03\x05\x17!\x06\x03\x01\x05\x01\x00\x82\n[\t\x0b%\x03\x0f\x0b\t\t\x11#!+\x1bi?\x1f/!)!)#\x1f\x19\x1f\x15\x1d\x15\x13%)9\x13\r+\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00broadcast_dimensions\x00value\x00sym_name\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(func)/jit(main)/hessenberg\x00third_party/py/jax/tests/export_back_compat_test.py\x00compare_type\x00comparison_direction\x00jax.result_info\x00mhlo.layout_mode\x00default\x00[0]\x00[1]\x00main\x00public\x00\x00lapack_dgehrd_ffi\x00high\x00low\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
