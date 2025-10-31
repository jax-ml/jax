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

data_2024_08_13 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_13["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgesdd_ffi'],
    serialized_date=datetime.date(2024, 8, 13),
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
    expected_outputs=(array([[[-0.0417367825863334  +0.10796693731538422j ,
          0.6813428383170979  +0.3432797958929331j  ,
         -0.4177022900286576  +0.20028957850808846j ,
         -0.4344351366508529  +0.034743251442636236j],
        [-0.8408468609573512  -0.13260646044648036j ,
         -0.21674151028481226 +0.015170556885426567j,
          0.17147327711152344 +0.15310416152982537j ,
         -0.3568765623609291  +0.2190438430670875j  ],
        [-0.26736181440441353 +0.1379833616281102j  ,
         -0.1753427835255798  -0.3789926157696272j  ,
         -0.8179957069096053  -0.037506032257391686j,
          0.25392637883428515 -0.009771014463849592j],
        [ 0.4056923996806594  -0.08297706578106906j ,
         -0.4321527034953763  +0.097915456635744j   ,
         -0.23439193826962634 -0.0842713053222817j  ,
         -0.423482961456089   +0.625144811494929j   ]],

       [[ 0.027268437398665468+0.3631205555033544j  ,
          0.2702977135592881  +0.13046165871625626j ,
          0.042868670139236786-0.47658594176021335j ,
          0.7242702256119966  +0.15420620503522459j ],
        [-0.08593436615104452 +0.11899901833255505j ,
          0.370502861093553   -0.6240865462984537j  ,
          0.46902056878805953 -0.3474794992077024j  ,
         -0.31667671459632085 -0.1034006436993295j  ],
        [-0.07914843440873574 -0.033487314943774216j,
          0.4110353453489126  -0.4550908055665629j  ,
         -0.43113180393027273 +0.40910871949631994j ,
          0.137827301024203   +0.49428280062680047j ],
        [-0.7478497242333215  +0.5283836938016965j  ,
         -0.08345894989956637 +0.011807690067190318j,
         -0.27178304569905287 +0.05652627940674812j ,
         -0.0991195491344199  -0.25988596540006825j ]]]), array([[16.80132997488892  ,  7.74475561455812  ,  5.831221808032042 ,
         1.1195288361137763],
       [12.395375946948931 ,  8.218551160453815 ,  4.68363485027408  ,
         1.882091536383919 ]]), array([[[ 0.3579625104055671  +0.j                  ,
          0.40179383774178024 -0.12693597167020743j ,
         -0.0751486661300563  -0.6109813931761134j  ,
         -0.23049271148274275 +0.51209309438597j    ],
        [-0.46828614153085474 +0.j                  ,
         -0.013958972669495653+0.4210606476774212j  ,
         -0.6006888466394118  -0.3766516564723723j  ,
         -0.24264518623236989 -0.20408557153193463j ],
        [-0.6392945524816099  +0.j                  ,
          0.24323886076029005 -0.6679928485374246j  ,
          0.18168178910997027 -0.08126854868489738j ,
         -0.2030612067046727  -0.07124733621915219j ],
        [-0.49383540371426055 +0.j                  ,
         -0.010402968929686451+0.37346249914107377j ,
          0.2799428270410499  +0.019494062167627474j,
          0.32588905219319264 +0.6569569657140542j  ]],

       [[ 0.26669203705168437 +0.j                  ,
          0.24929033811571388 +0.27271089049933883j ,
         -0.012922512768026959+0.16383354123801502j ,
          0.07388201893235019 -0.8717175469187742j  ],
        [-0.6156140469162427  +0.j                  ,
         -0.33787077397020177 +0.3779715465092333j  ,
         -0.39160430587261197 -0.2839601305776179j  ,
         -0.27148886041576736 -0.23729034093304668j ],
        [ 0.5618758038857614  +0.j                  ,
         -0.5788776267734558  -0.13833058883452376j ,
         -0.48995086206819655 +0.19259594116096806j ,
         -0.22967101640965004 -0.012926826751577636j],
        [-0.48393210641613604 +0.j                  ,
         -0.10492296054284367 -0.4911419972025976j  ,
         -0.07782239226461207 +0.6751317817750168j  ,
          0.11941657609231512 -0.19354808489959857j ]]])),
    mlir_module_text=r"""
#loc1 = loc("input")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f64>> {mhlo.layout_mode = "default"} loc("input")) -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xcomplex<f64>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<i64> loc(#loc3)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc3)
    %c_1 = stablehlo.constant dense<4> : tensor<i64> loc(#loc3)
    %0:5 = stablehlo.custom_call @lapack_zgesdd_ffi(%arg0) {mhlo.backend_config = {mode = 65 : ui8}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>>, tensor<2x4x4xcomplex<f64>>, tensor<2xi32>) loc(#loc3)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %cst_3 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %cst_4 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc3)
    return %10, %6, %14 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":574:13)
#loc3 = loc("jit(func)/jit(main)/svd[full_matrices=True compute_uv=True subset_by_index=None]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xf9\xab;\x01Y\x0f\x0b\x07\x13\x0b\x13\x0f\x0b\x13\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x13\x03S\x0b\x0bo\x0b\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b//\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0f\x0f\x17\x1fO/\x1f\x0f\x0b\x0b//OOo\x01\x05\x0b\x0f\x037\x1b\x0f\x07\x07\x17\x07\x07\x0f\x0b\x13\x07\x0f\x0f\x1b\x1b\x1f\x07\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\x1a\x08\x1d35\x05\x15\x1f\x03\x03\t\x9b\x05\x17\x03\x03\t\xa1\x11\x03\x05\x05\x19\x03\x03\x03{\x03\x03\x03\xa7\x03\x03\t\xa9\x03\t\x19\x1b\x1d\r\x1f\r\x0f!\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x03\x0b%a'e)g\x0fu+w\x05#\x05%\x05'\x05)\x1d/\x05\x05+\x03\x03\x03y\x05-\x177\xfa\x08\x1b\x05/\x03\x13;}=\x7f?\x81A\x83C\x85E\x87G\x8dI\x8fK\x93\x051\x053\x055\x057\x059\x05;\x05=\x05?\x05A\x03\x03\x03\x99\x03\x05Q\x9dS\x9f\x05C\x05E\x03\x03\x03\xa3\x03\x03\t\xa5\x1dG\x1dI\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dK\x03\x03c\r\x03Y[##\x03\x07imq\r\x05_kY[\x1dM\r\x05_oY[\x1dO\r\x05_sY[\x1dQ\x1dS\x1dU\x1f\x07\x11\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1dW\x1dY\x03\x01\x05\x01\r\x03\x89\x8b\x1d[\x13%A\x03\x03]\x03\x03\x91\x15\x03\x01\x01\x01\x03\x0b]\x95]]\x97\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1b\t\x00\x00\x00\x00\x1f-\x01\t\x07\x07\x01\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f7!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x13!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f91\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x15)\x01\t\x1d\x01)\x05\t\x11\x0f\x0b\x13)\x01\x15\x03\x0f)\x03\t\x19\x1b)\x01\x19)\x01\x0f)\x07\t\x05\x05\x0b)\x07\t\x11\x11\x0b\x11\x03\x05\x07\x05\r\x05!)\x03\r\x11)\x03\t\x11)\x03\x05\x11)\x03\x01\t)\x03\t\x0b)\x05\t\x05\x0b)\x03\x05\t)\x05\t\x11\x0b)\x03\t\t)\x03\r\t\x04\x16\x03\x05\x01\x11\x05\x17\x07\x03\x01\x05\t\x11\x05#\x07\x037_\x03\x05-\x05\x03\x011\x03\x07\x05\x03\x01\x11\x03\x07\x05\x03\x01\x11\x03\x07\x0b\x07\x019\x0b\x05\r\x05\x05\x17\x03\x01\x05\x03\x01M\x03\x1b\x03\x07\x01\x07\x03\x17\x03\x13\r\x07\x01O\x03/\x05\x11\x15\x03\x07\x01\x0b\x031\x03\x17\x05\x03\x01U\x03\x1d\x03\x07\x01\x07\x03\r\x03\x1b\x03\x07\x01W\x035\x03\x19\x07\x06\x01\x03\r\x07\x1f\x0b\x1d\x03\x07\x01\x0b\x03\x1f\x03\x17\x05\x03\x01\x13\x03\x13\x03\x07\x01\x07\x03\x05\x03%\x03\x07\x01\x15\x03!\x03#\x07\x06\x01\x03\x05\x07)\r'\x03\x07\x01\x0b\x03\x1f\x03\x17\x05\x03\x01\x13\x03\x13\x03\x07\x01\x07\x03\x05\x03/\x03\x07\x01\x15\x03!\x03-\x07\x06\x01\x03\x05\x073\x0f1\x0f\x04\x05\x07+!5\x06\x03\x01\x05\x01\x00f\x0b]\x0b%\x03\x0f\x0b\t\t\t!\x11#+\x1b\x1f/!)!)#\x1f\x19i\xa3\r\x1f\x15\x1d\x15\x13%)9\x13+\r\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00input\x00jit(func)/jit(main)/svd[full_matrices=True compute_uv=True subset_by_index=None]\x00third_party/py/jax/tests/export_back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_zgesdd_ffi\x00mode\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_13["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgesdd_ffi'],
    serialized_date=datetime.date(2024, 8, 13),
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
    expected_outputs=(array([[[ 0.016725361+0.19210356j ,  0.545269   +0.5572638j  ,
          0.41363978 +0.18964852j , -0.26152337 -0.28195122j ],
        [ 0.53678614 +0.6405725j  , -0.21783227 -0.21288806j ,
          0.28426635 +0.30535886j ,  0.15201291 +0.1076857j  ],
        [ 0.21286921 +0.15473497j ,  0.06647172 -0.25652882j ,
         -0.4074609  -0.10356678j , -0.11794218 -0.8184482j  ],
        [-0.39079374 -0.20583557j , -0.18335938 -0.44217706j ,
          0.63489586 +0.19758745j ,  0.038679928-0.363512j   ]],

       [[-0.31785947 +0.39032045j , -0.12733367 -0.30841753j ,
          0.2639419  +0.26815215j , -0.21332225 -0.6694792j  ],
        [-0.39241248 -0.60790956j , -0.14006217 +0.4104069j  ,
         -0.08306134 -0.101844534j, -0.45091915 -0.26039878j ],
        [-0.36103737 +0.28761536j , -0.49654633 +0.100843735j,
         -0.13752809 -0.6203827j  ,  0.35439843 -0.028546259j],
        [ 0.062335134-0.07821423j ,  0.35014486 -0.5668197j  ,
         -0.42214072 -0.5090834j  , -0.2889286  -0.15894136j ]]],
      dtype=complex64), array([[15.135656 ,  9.3730345,  7.44493  ,  0.4152342],
       [12.316968 ,  8.661011 ,  5.005059 ,  2.1159043]], dtype=float32), array([[[-0.65378654 +0.j         , -0.20306695 -0.6166746j  ,
          0.29948464 +0.24257994j , -0.00760437 +0.049453575j],
        [ 0.5271269  +0.j         , -0.112915546-0.7116953j  ,
         -0.08921899 -0.36348897j , -0.23654734 -0.08269382j ],
        [-0.31538552 +0.j         , -0.014410704+0.15958196j ,
         -0.17958632 -0.136909j   , -0.6930434  -0.58613425j ],
        [-0.44185144 +0.j         ,  0.17604697 -0.05049205j ,
         -0.42138547 -0.6948516j  ,  0.22373372 +0.24654455j ]],

       [[-0.64551586 +0.j         ,  0.3293224  -0.1167212j  ,
         -0.09352748 +0.6710144j  , -0.038554132+0.02716675j ],
        [ 0.4241116  +0.j         ,  0.031135   -0.539813j   ,
         -0.26271757 +0.22760022j , -0.6360964  -0.04817466j ],
        [-0.45774835 +0.j         , -0.15202752 +0.2734652j  ,
          0.18930997 -0.32975054j , -0.73310995 -0.10269694j ],
        [ 0.4403465  +0.j         ,  0.29474002 +0.6330784j  ,
          0.31271845 +0.42166728j , -0.20595443 -0.02053237j ]]],
      dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("input")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f32>> {mhlo.layout_mode = "default"} loc("input")) -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xcomplex<f32>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<i64> loc(#loc3)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc3)
    %c_1 = stablehlo.constant dense<4> : tensor<i64> loc(#loc3)
    %0:5 = stablehlo.custom_call @lapack_cgesdd_ffi(%arg0) {mhlo.backend_config = {mode = 65 : ui8}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>>, tensor<2x4x4xcomplex<f32>>, tensor<2xi32>) loc(#loc3)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %cst_3 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %cst_4 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc3)
    return %10, %6, %14 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":574:13)
#loc3 = loc("jit(func)/jit(main)/svd[full_matrices=True compute_uv=True subset_by_index=None]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xf9\xab;\x01Y\x0f\x0b\x07\x13\x0b\x13\x0f\x0b\x13\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x13\x03S\x0b\x0bo\x0b\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b//\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0f\x0f\x17\x1fO/\x1f\x0f\x0b\x0b/\x1fO/o\x01\x05\x0b\x0f\x037\x1b\x0f\x07\x07\x17\x07\x07\x0f\x0b\x13\x07\x0f\x0f\x1b\x1b\x1f\x07\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\xea\x07\x1d35\x05\x15\x1f\x03\x03\t\x9b\x05\x17\x03\x03\t\xa1\x11\x03\x05\x05\x19\x03\x03\x03{\x03\x03\x03\xa7\x03\x03\t\xa9\x03\t\x19\x1b\x1d\r\x1f\r\x0f!\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x03\x0b%a'e)g\x0fu+w\x05#\x05%\x05'\x05)\x1d/\x05\x05+\x03\x03\x03y\x05-\x177\xfa\x08\x1b\x05/\x03\x13;}=\x7f?\x81A\x83C\x85E\x87G\x8dI\x8fK\x93\x051\x053\x055\x057\x059\x05;\x05=\x05?\x05A\x03\x03\x03\x99\x03\x05Q\x9dS\x9f\x05C\x05E\x03\x03\x03\xa3\x03\x03\t\xa5\x1dG\x1dI\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dK\x03\x03c\r\x03Y[##\x03\x07imq\r\x05_kY[\x1dM\r\x05_oY[\x1dO\r\x05_sY[\x1dQ\x1dS\x1dU\x1f\x07\x11\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1dW\x1dY\x03\x01\x05\x01\r\x03\x89\x8b\x1d[\x13%A\x03\x03]\x03\x03\x91\x15\x03\x01\x01\x01\x03\x0b]\x95]]\x97\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1b\t\x00\x00\x00\x00\x1f-\x01\t\x07\x07\x01\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1d\t\x00\x00\xc0\x7f\x1f7!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x13\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f91\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x15)\x01\t\x1d\x01)\x05\t\x11\x0f\t\x13)\x01\x15\x03\x0f)\x03\t\x19\x1b)\x01\x19)\x01\x0f)\x07\t\x05\x05\x0b)\x07\t\x11\x11\x0b\x11\x03\x05\x07\x05\r\x05!)\x03\r\x11)\x03\t\x11)\x03\x05\x11)\x03\x01\t)\x03\t\x0b)\x05\t\x05\x0b)\x03\x05\t)\x05\t\x11\x0b)\x03\t\t)\x03\r\t\x04\x16\x03\x05\x01\x11\x05\x17\x07\x03\x01\x05\t\x11\x05#\x07\x037_\x03\x05-\x05\x03\x011\x03\x07\x05\x03\x01\x11\x03\x07\x05\x03\x01\x11\x03\x07\x0b\x07\x019\x0b\x05\r\x05\x05\x17\x03\x01\x05\x03\x01M\x03\x1b\x03\x07\x01\x07\x03\x17\x03\x13\r\x07\x01O\x03/\x05\x11\x15\x03\x07\x01\x0b\x031\x03\x17\x05\x03\x01U\x03\x1d\x03\x07\x01\x07\x03\r\x03\x1b\x03\x07\x01W\x035\x03\x19\x07\x06\x01\x03\r\x07\x1f\x0b\x1d\x03\x07\x01\x0b\x03\x1f\x03\x17\x05\x03\x01\x13\x03\x13\x03\x07\x01\x07\x03\x05\x03%\x03\x07\x01\x15\x03!\x03#\x07\x06\x01\x03\x05\x07)\r'\x03\x07\x01\x0b\x03\x1f\x03\x17\x05\x03\x01\x13\x03\x13\x03\x07\x01\x07\x03\x05\x03/\x03\x07\x01\x15\x03!\x03-\x07\x06\x01\x03\x05\x073\x0f1\x0f\x04\x05\x07+!5\x06\x03\x01\x05\x01\x00f\x0b]\x0b%\x03\x0f\x0b\t\t\t!\x11#+\x1b\x1f/!)!)#\x1f\x19i\xa3\r\x1f\x15\x1d\x15\x13%)9\x13+\r\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00input\x00jit(func)/jit(main)/svd[full_matrices=True compute_uv=True subset_by_index=None]\x00third_party/py/jax/tests/export_back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_cgesdd_ffi\x00mode\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_13["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgesdd_ffi'],
    serialized_date=datetime.date(2024, 8, 13),
    inputs=(array([[[ 1.5410905 , -2.775912  , -2.374003  ,  4.028736  ],
        [-0.56933475,  1.6115232 ,  0.9041465 , -0.8321383 ],
        [-5.382895  ,  4.734856  ,  2.1972926 ,  1.5553856 ],
        [ 0.5109847 , -1.1969309 ,  3.3766198 , -1.3678027 ]],

       [[ 2.2637439 ,  3.406768  ,  4.809871  ,  2.8010902 ],
        [-1.9981416 , -0.6599986 ,  0.5138156 ,  4.5982494 ],
        [-2.335944  , -9.151717  , -1.0481138 ,  2.272443  ],
        [-8.257684  ,  1.8223318 ,  0.38403794,  5.0769973 ]]],
      dtype=float32),),
    expected_outputs=(array([[[-0.48540133 ,  0.6682398  , -0.48819908 , -0.28196266 ],
        [ 0.21800542 , -0.13631387 ,  0.14819776 , -0.9549501  ],
        [ 0.84570533 ,  0.44643924 , -0.27943408 ,  0.08597416 ],
        [ 0.04052323 , -0.57928103 , -0.8133976  , -0.034290295]],

       [[-0.21146727 ,  0.46376404 ,  0.7863092  ,  0.34917426 ],
        [ 0.3461469  ,  0.21883708 ,  0.3399651  , -0.846591   ],
        [ 0.6526193  , -0.58340365 ,  0.39724028 ,  0.27555162 ],
        [ 0.6399629  ,  0.6298205  , -0.32915345 ,  0.29228795 ]]],
      dtype=float32), array([[ 8.551605  ,  5.3574076 ,  2.8073733 ,  0.52260846],
       [11.457574  , 10.041604  ,  5.671653  ,  1.4754113 ]],
      dtype=float32), array([[[-0.6319044 ,  0.66122514,  0.39110142, -0.10255312],
        [-0.29710513,  0.13673344, -0.50112027,  0.8011937 ],
        [ 0.08969161,  0.4433049 , -0.736473  , -0.5030347 ],
        [-0.7101976 , -0.5895469 , -0.23135659, -0.30745378]],

       [[-0.69643414, -0.50230867, -0.11150038,  0.50023323],
        [-0.32121184,  0.7889567 ,  0.31831914,  0.4159848 ],
        [ 0.5096959 , -0.31399366,  0.60193473,  0.5284817 ],
        [-0.3898877 , -0.16322279,  0.72382   , -0.5453722 ]]],
      dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("input")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf32> {mhlo.layout_mode = "default"} loc("input")) -> (tensor<2x4x4xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xf32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<i64> loc(#loc3)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc3)
    %c_1 = stablehlo.constant dense<4> : tensor<i64> loc(#loc3)
    %0:5 = stablehlo.custom_call @lapack_sgesdd_ffi(%arg0) {mhlo.backend_config = {mode = 65 : ui8}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>) loc(#loc3)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %cst_3 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc3)
    return %10, %6, %14 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":574:13)
#loc3 = loc("jit(func)/jit(main)/svd[full_matrices=True compute_uv=True subset_by_index=None]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xf1\xa77\x01W\x0f\x07\x0b\x13\x0b\x13\x13\x0f\x0b\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x03Q\x0b\x0bo\x0b\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b//\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0f\x0f\x17\x1fO/\x1f\x0f\x0b\x0b/\x1fOo\x01\x05\x0b\x0f\x033\x1b\x0f\x07\x07\x17\x0f\x07\x07\x13\x07\x0f\x1b\x1b\x1f\x07\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\x9a\x07\x1d35\x1f\x05\x15\x03\x03\t\x99\x05\x17\x03\x03\t\x9f\x03\x03\x05\xa1\x11\x03\x05\x05\x19\x03\x03\x05y\x03\x03\t\xa5\x03\t\x19\x1b\x1d\x0f\x1f\x0f\x11!\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x03\x0b%_'c)e\x11s+u\x05#\x05%\x05'\x05)\x1d/\x03\x05+\x03\x03\x05w\x05-\x177\xfa\x08\x1b\x05/\x03\x13;{=}?\x7fA\x81C\x83E\x85G\x8bI\x8dK\x91\x051\x053\x055\x057\x059\x05;\x05=\x05?\x05A\x03\x03\x05\x97\x03\x05Q\x9bS\x9d\x05C\x05E\x03\x03\t\xa3\x1dG\x1dI\x1f#1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dK\x03\x03a\r\x03WY#\x1f\x03\x07gko\r\x05]iWY\x1dM\r\x05]mWY\x1dO\r\x05]qWY\x1dQ\x1dS\x1dU\x1f\x07\x11\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1dW\x1dY\x03\x01\x05\x01\r\x03\x87\x89\x1d[\x13!A\x03\x03[\x03\x03\x8f\x15\x03\x01\x01\x01\x03\x0b[\x93[[\x95\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19\t\x00\x00\x00\x00\x1f)\x01\t\x07\x07\x01\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x0f\t\x00\x00\xc0\x7f\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f51\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x11)\x01\t\x1d\x01)\x05\t\x11\x11)\x01\x11\t\x13)\x03\t\x17\x1b)\x01\x17)\x07\t\x05\x05\x0b)\x07\t\x11\x11\x0b\x11\x03\x05\x07\x05\r\x05!)\x03\r\x13)\x03\t\x13)\x03\x05\x13)\x03\x01\t)\x03\t\x0b)\x05\t\x05\x0b)\x03\x05\t)\x05\t\x11\x0b)\x03\t\t)\x03\r\t\x04\x16\x03\x05\x01\x11\x03\x17\x07\x03\x01\x05\t\x11\x03#\x07\x037_\x03\x05-\x05\x03\x011\x03\x07\x05\x03\x01\x13\x03\x07\x05\x03\x01\x13\x03\x07\x0b\x07\x019\x0b\x05\r\x05\x05\x15\x03\x01\x05\x03\x01M\x03\x19\x03\x07\x01\x07\x03\x15\x03\x13\r\x07\x01O\x03+\x05\x11\x15\x03\x07\x01\x0b\x03-\x03\x17\x05\x03\x01\r\x03\x0f\x03\x07\x01\x07\x03\r\x03\x1b\x03\x07\x01U\x031\x03\x19\x07\x06\x01\x03\r\x07\x1f\x0b\x1d\x03\x07\x01\x0b\x03\x1b\x03\x17\x05\x03\x01\r\x03\x0f\x03\x07\x01\x07\x03\x05\x03%\x03\x07\x01\x15\x03\x1d\x03#\x07\x06\x01\x03\x05\x07)\r'\x03\x07\x01\x0b\x03\x1b\x03\x17\x05\x03\x01\r\x03\x0f\x03\x07\x01\x07\x03\x05\x03/\x03\x07\x01\x15\x03\x1d\x03-\x07\x06\x01\x03\x05\x073\x0f1\x0f\x04\x03\x07+!5\x06\x03\x01\x05\x01\x00f\x0b]\x0b%\x03\x0f\x0b\t\t\t!\x11#+\x1b\x1f/!)!)#\x1f\x19i\xa3\r\x1f\x15\x1d\x15\x13%)9\x13+\r\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00input\x00jit(func)/jit(main)/svd[full_matrices=True compute_uv=True subset_by_index=None]\x00third_party/py/jax/tests/export_back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_sgesdd_ffi\x00mode\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_13["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgesdd_ffi'],
    serialized_date=datetime.date(2024, 8, 13),
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
    expected_outputs=(array([[[-0.5109626909166218  , -0.41744996156105796 ,
         -0.731253241567692   ,  0.17297790257908272 ],
        [-0.5623501368035175  ,  0.7608931604238581  ,
          0.03470920608540995 ,  0.32186828528169453 ],
        [-0.39585755254587396 , -0.49547702914054115 ,
          0.6561880513437817  ,  0.4089212062978682  ],
        [-0.5157288533916832  , -0.035772078593888285,
          0.18297871183094855 , -0.8362194085221047  ]],

       [[-0.12124821978030864 , -0.30260506534356224 ,
         -0.5817463045715605  , -0.7451847292758066  ],
        [ 0.8877417367326683  , -0.1579400123987918  ,
         -0.37611807392676866 ,  0.21331843758089156 ],
        [ 0.030552216758649886,  0.9244545314395404  ,
         -0.36861075330670934 , -0.09260936183071362 ],
        [-0.443035032603635   , -0.1699086407831784  ,
         -0.6198649402326368  ,  0.624994775612963   ]]]), array([[8.951386926411187 , 5.762891699811625 , 3.8391040088894437,
        1.269646897103325 ],
       [9.215006888576916 , 6.4772976708832255, 3.246269458558178 ,
        0.0511210199435459]]), array([[[-0.1789027692424481 , -0.28818125207050604,
         -0.7749616998111009 , -0.5332726590950896 ],
        [ 0.3871215938703837 , -0.8985113987184387 ,
          0.13976186700464233,  0.1525803344591491 ],
        [-0.2314069792404015 , -0.03708202130554682,
         -0.5045854966104311 ,  0.8309447696839618 ],
        [-0.8744034999217863 , -0.32901938548360005,
          0.35396957633060844, -0.04324699218274111]],

       [[ 0.6276106632546885 , -0.267287353478729  ,
         -0.2299525871877408 ,  0.69410671635204   ],
        [ 0.28029316975925644,  0.47811378046591546,
          0.8083625695047307 ,  0.1984764674680803 ],
        [ 0.6187014005224261 ,  0.4771409534394446 ,
         -0.37406866975606345, -0.4996175715979325 ],
        [-0.38045915857935025,  0.6872417290515942 ,
         -0.3921025301835002 ,  0.4787538410571401 ]]])),
    mlir_module_text=r"""
#loc1 = loc("input")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf64> {mhlo.layout_mode = "default"} loc("input")) -> (tensor<2x4x4xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xf64> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<i64> loc(#loc3)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc3)
    %c_1 = stablehlo.constant dense<4> : tensor<i64> loc(#loc3)
    %0:5 = stablehlo.custom_call @lapack_dgesdd_ffi(%arg0) {mhlo.backend_config = {mode = 65 : ui8}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64>, tensor<2x4x4xf64>, tensor<2xi32>) loc(#loc3)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %cst_3 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %cst_4 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc3)
    return %10, %6, %14 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":574:13)
#loc3 = loc("jit(func)/jit(main)/svd[full_matrices=True compute_uv=True subset_by_index=None]"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1f\x05\x01\x03\x01\x03\x05\x03\x0f\x07\t\x0b\r\x0f\x11\x13\x03\xf1\xa77\x01W\x0f\x07\x0b\x13\x0b\x13\x13\x0f\x0b\x13\x13+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x17\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x13\x03Q\x0b\x0bo\x0b\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b//\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0f\x0f\x17\x1fO/\x1f\x0f\x0b\x0b//Oo\x01\x05\x0b\x0f\x033\x1b\x0f\x07\x07\x17\x0f\x07\x07\x13\x07\x0f\x1b\x1b\x1f\x07\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\xaa\x07\x1d35\x1f\x05\x15\x03\x03\t\x99\x05\x17\x03\x03\t\x9f\x03\x03\x05\xa1\x11\x03\x05\x05\x19\x03\x03\x05y\x03\x03\t\xa5\x03\t\x19\x1b\x1d\x0f\x1f\x0f\x11!\x05\x1b\x11\x01\x00\x05\x1d\x05\x1f\x05!\x03\x0b%_'c)e\x11s+u\x05#\x05%\x05'\x05)\x1d/\x03\x05+\x03\x03\x05w\x05-\x177\xfa\x08\x1b\x05/\x03\x13;{=}?\x7fA\x81C\x83E\x85G\x8bI\x8dK\x91\x051\x053\x055\x057\x059\x05;\x05=\x05?\x05A\x03\x03\x05\x97\x03\x05Q\x9bS\x9d\x05C\x05E\x03\x03\t\xa3\x1dG\x1dI\x1f#1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1dK\x03\x03a\r\x03WY#\x1f\x03\x07gko\r\x05]iWY\x1dM\r\x05]mWY\x1dO\r\x05]qWY\x1dQ\x1dS\x1dU\x1f\x07\x11\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1dW\x1dY\x03\x01\x05\x01\r\x03\x87\x89\x1d[\x13!A\x03\x03[\x03\x03\x8f\x15\x03\x01\x01\x01\x03\x0b[\x93[[\x95\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19\t\x00\x00\x00\x00\x1f)\x01\t\x07\x07\x01\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x0f\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f51\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x11)\x01\t\x1d\x01)\x05\t\x11\x11)\x01\x11\x0b\x13)\x03\t\x17\x1b)\x01\x17)\x07\t\x05\x05\x0b)\x07\t\x11\x11\x0b\x11\x03\x05\x07\x05\r\x05!)\x03\r\x13)\x03\t\x13)\x03\x05\x13)\x03\x01\t)\x03\t\x0b)\x05\t\x05\x0b)\x03\x05\t)\x05\t\x11\x0b)\x03\t\t)\x03\r\t\x04\x16\x03\x05\x01\x11\x03\x17\x07\x03\x01\x05\t\x11\x03#\x07\x037_\x03\x05-\x05\x03\x011\x03\x07\x05\x03\x01\x13\x03\x07\x05\x03\x01\x13\x03\x07\x0b\x07\x019\x0b\x05\r\x05\x05\x15\x03\x01\x05\x03\x01M\x03\x19\x03\x07\x01\x07\x03\x15\x03\x13\r\x07\x01O\x03+\x05\x11\x15\x03\x07\x01\x0b\x03-\x03\x17\x05\x03\x01\r\x03\x0f\x03\x07\x01\x07\x03\r\x03\x1b\x03\x07\x01U\x031\x03\x19\x07\x06\x01\x03\r\x07\x1f\x0b\x1d\x03\x07\x01\x0b\x03\x1b\x03\x17\x05\x03\x01\r\x03\x0f\x03\x07\x01\x07\x03\x05\x03%\x03\x07\x01\x15\x03\x1d\x03#\x07\x06\x01\x03\x05\x07)\r'\x03\x07\x01\x0b\x03\x1b\x03\x17\x05\x03\x01\r\x03\x0f\x03\x07\x01\x07\x03\x05\x03/\x03\x07\x01\x15\x03\x1d\x03-\x07\x06\x01\x03\x05\x073\x0f1\x0f\x04\x03\x07+!5\x06\x03\x01\x05\x01\x00f\x0b]\x0b%\x03\x0f\x0b\t\t\t!\x11#+\x1b\x1f/!)!)#\x1f\x19i\xa3\r\x1f\x15\x1d\x15\x13%)9\x13+\r\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00value\x00broadcast_dimensions\x00sym_name\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00input\x00jit(func)/jit(main)/svd[full_matrices=True compute_uv=True subset_by_index=None]\x00third_party/py/jax/tests/export_back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00\x00lapack_dgesdd_ffi\x00mode\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
