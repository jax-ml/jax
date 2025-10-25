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

data_2024_08_19 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_19["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zheevd_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([[-6.1857700048412056e-01+0.j,  2.4081403770912022e-01+0.j,
         3.5662489253627483e-01+0.j, -6.3034019033669797e-01+0.j,
         1.0043483479985752e-16+0.j, -2.8842036081919542e-02+0.j,
         7.7164692943283169e-25+0.j, -1.8446994643771725e-01+0.j],
       [-4.7070881487314609e-01+0.j,  4.7473787464450828e-01+0.j,
        -4.8036836210243361e-01+0.j,  4.3802686872516400e-01+0.j,
         1.7961797619639255e-01+0.j,  8.3080980076741355e-03+0.j,
         2.1415294457221759e-01+0.j, -2.2856669794666584e-01+0.j],
       [-3.2284062926217072e-01+0.j, -5.4336490915553370e-01+0.j,
         2.2181041859724987e-01+0.j,  2.9947877954402286e-01+0.j,
        -3.6491813600134637e-01+0.j,  3.2867679819727436e-01+0.j,
         3.8223299448843473e-01+0.j, -2.7266344945561438e-01+0.j],
       [-1.7497244365119527e-01+0.j, -8.9251550609769331e-02+0.j,
        -6.3518515114898352e-02+0.j,  1.9162997359209963e-01+0.j,
        -2.2087281326110142e-01+0.j,  5.9957027043505008e-02+0.j,
        -8.7632498908241274e-01+0.j, -3.1676020096456303e-01+0.j],
       [-2.7104258040220017e-02+0.j, -3.3772873786627688e-01+0.j,
         2.5901386593721754e-01+0.j,  1.7032650752287815e-01+0.j,
         6.7521217612940321e-01+0.j, -4.5036136532965476e-01+0.j,
        -1.2279030059078447e-02+0.j, -3.6085695247351163e-01+0.j],
       [ 1.2076392757075533e-01+0.j, -3.3834734096469249e-01+0.j,
        -6.5506827461665529e-01+0.j, -5.0472498521116760e-01+0.j,
         6.9987430903492132e-02+0.j,  1.0595648906599270e-01+0.j,
         8.3443844143082035e-02+0.j, -4.0495370398246017e-01+0.j],
       [ 2.6863211318173102e-01+0.j,  2.2958613191407312e-01+0.j,
         6.3952843755683969e-02+0.j,  1.8776775771084192e-02+0.j,
        -5.3523731432241317e-01+0.j, -5.9199531677602002e-01+0.j,
         1.7916671834524250e-01+0.j, -4.4905045549140887e-01+0.j],
       [ 4.1650029879270667e-01+0.j,  3.6355449432857068e-01+0.j,
         2.9755313100756148e-01+0.j,  1.6826270392616000e-02+0.j,
         1.9621068035557282e-01+0.j,  5.6830030587314817e-01+0.j,
         2.9607517592514260e-02+0.j, -4.9314720700035747e-01+0.j]]), array([-2.4598804776133626e+01, -4.6567755957874661e-14,
       -1.9932120610662194e-14, -5.7323356091157378e-15,
       -4.5459724251334835e-16,  4.0479851042511616e-14,
        9.2325194924982089e-14,  2.7659880477613365e+02])),
    mlir_module_text=r"""
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":277:27)
#loc18 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc7))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x8xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<8xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xcomplex<f64>> loc(#loc9)
    %1 = stablehlo.reshape %0 : (tensor<64xcomplex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc10)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc11)
    %3 = stablehlo.real %2 : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xf64> loc(#loc12)
    %4 = stablehlo.imag %2 : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xf64> loc(#loc13)
    %5 = stablehlo.negate %4 : tensor<8x8xf64> loc(#loc14)
    %6 = stablehlo.complex %3, %5 : tensor<8x8xcomplex<f64>> loc(#loc15)
    %7 = stablehlo.add %1, %6 : tensor<8x8xcomplex<f64>> loc(#loc16)
    %cst = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc17)
    %9 = stablehlo.divide %7, %8 : tensor<8x8xcomplex<f64>> loc(#loc17)
    %10 = call @tril(%9) : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc18)
    %c = stablehlo.constant dense<8> : tensor<i64> loc(#loc19)
    %c_0 = stablehlo.constant dense<8> : tensor<i64> loc(#loc19)
    %11:3 = stablehlo.custom_call @lapack_zheevd_ffi(%10) {mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<8x8xcomplex<f64>>) -> (tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>) loc(#loc19)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc19)
    %12 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc19)
    %13 = stablehlo.compare  EQ, %11#2, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc19)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc19)
    %cst_2 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc19)
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc19)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1> loc(#loc19)
    %17 = stablehlo.select %16, %11#0, %15 : tensor<8x8xi1>, tensor<8x8xcomplex<f64>> loc(#loc19)
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc19)
    %cst_3 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc19)
    %19 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<8xf64> loc(#loc19)
    %20 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi1>) -> tensor<8xi1> loc(#loc19)
    %21 = stablehlo.select %20, %11#1, %19 : tensor<8xi1>, tensor<8xf64> loc(#loc19)
    return %17, %21 : tensor<8x8xcomplex<f64>>, tensor<8xf64> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<8x8xcomplex<f64>> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc7))) -> (tensor<8x8xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32> loc(#loc20)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc18)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32> loc(#loc21)
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32> loc(#loc21)
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32> loc(#loc22)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1> loc(#loc23)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc18)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc24)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xcomplex<f64>> loc(#loc25)
    return %6 : tensor<8x8xcomplex<f64>> loc(#loc18)
  } loc(#loc18)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":269:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":269:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:25)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:15)
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:14)
#loc8 = loc("third_party/py/jax/tests/export_back_compat_test.py":277:11)
#loc9 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(64,) dimension=0]"(#loc1))
#loc10 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/real"(#loc4))
#loc13 = loc("jit(<lambda>)/jit(main)/imag"(#loc4))
#loc14 = loc("jit(<lambda>)/jit(main)/neg"(#loc4))
#loc15 = loc("jit(<lambda>)/jit(main)/complex"(#loc4))
#loc16 = loc("jit(<lambda>)/jit(main)/add"(#loc5))
#loc17 = loc("jit(<lambda>)/jit(main)/div"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]"(#loc8))
#loc20 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]"(#loc7))
#loc21 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc7))
#loc22 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]"(#loc7))
#loc23 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc7))
#loc24 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]"(#loc7))
#loc25 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc7))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\'\x03\xda\x02*\x02?\x01\xab\x0f\x0b\x13\x17\x0f\x0b\x07\x17\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x0f\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x0f\x0b\x17\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x13\x0b\x17\x13\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03a\x0b\x0b\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0bOOO/\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17\x17/\x0f\x0bOO//\x01\x0b\x1f\x17\x17\x17\x17\x01\x05\x0b\x0f\x03;\x17\x07\x0f\x0f\x07\x07\x13\x17\x0b\x17\x0f\x07\x07\x17\x13\x07\x0f\x17\x17\x13\x17\x13\x13\x13\x0f\x17\x13\x13\x13\x02\xa6\n\x1d\x93\x95\x05)\x03\x03\x13\xd5\x17\x03V\x047\x1d?\x07\x05+\x1f\x17\x03>\x043\x05-\x05/\x11\x03\x05\x051\x053\x055\x057\x03\x03!\xd1\x059\x03\x03\x0b\xd3\x1dE\x07\x05;\x05=\x1d\x8b\x8d\x03\x03\x0b\xe1\x03\t135\x157\x15\x119\x05?\x11\x01\x00\x05A\x05C\x05E\x03\x0b\x17\xaf\x19\xbb\x1b\xbd\x11\xc7\x1d\xc9\x03\x0b\x17\xb3\x19\xcd\x1b\xb3\x11\xb5\x1d\xcf\x05G\x1dC\x07\x05I\x05K\x03\x03!\xd7\x1dK\x07\x05M\x03\x05\'\xb7)\xd9\x1dQ\x07\x05O\x03\x03\x0b\xdb\x1dW\x07\x05Q\x1d[\x07\x05S\x1d_a\x05U\x17\x036\x045\x1deg\x05W\x17\x036\x04\x1d\x03\x03k\xdd\x05Y\x1doq\x05[\x17\x03>\x04E\x1du\x0f\x05]\x1dy\x0f\x05_\x1d}\x0f\x05a\x1d\x81\x0f\x05c\x1d\x85\x87\x05e\x17\x03>\x04\x1f\x03\x03\x0b\xdf\x05g\x17\x03>\x04\x1d\x03\x03\x91\xb5\x05i\x05k\x17\x03V\x04\x17\x03\x13\x99\xe3\x9b\xe5\x9d\xe7\x9f\xaf\xa1\xe9\xa3\xeb\xa5\xf5\xa7\xf7\xa9\xfb\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x05}\x1d\x7f\x1d\x81\x03\x01\x1d\x83\x03\x03\xcb\x1d\x85\t\x07\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\'\x03\x05\xbf\xc3\r\x05\xb1\xc1\xab\xad\x1d\x87\r\x05\xb1\xc5\xab\xad\x1d\x89\x1d\x8b\x1d\x8d\r\x03\xab\xad#)\x1d\x8f\x13\x07\x01\x1f\x0b\t\x00\x00\x00\x00\x1f+\x01\x13\x07\x05\x07\x05\x1f\t!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\t!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19\x11\x08\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x91\x1d\x93\x05\x01\r\x05\xed\xef\xf1\xf3\x1d\x95\x13#V\x1d\x97\x13#L\x03\x03\xb9\x03\x03\xf9\x15\x03\x01\x01\x01\x03\x07\xb9\xfd\xff\x1f1\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f3\x01\x07\x01\x1f\t!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f%\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05\'\xb7)\x02\x02\x03\x03\x0b\x06\x02\x03\x03\x13\n\x02\x03\x03\x0b\x0e\x02\x03\x03\x13\x12\x02\x01\t\x01\x02\x02)\x05!!\x15\x1d)\x01\x15)\x01\x1d\x01\x0b)\x03!\x0f)\x05!!\x1d\x03\x0f)\x05!!\x0f)\x01\x07\x13\x1b)\x05!!\r)\x03\t\x07!)\x01\x0f\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\x07)\x03\x02\x02\x15)\x03\t\x1b)\x03\x05\x1b)\x03\x01\x1b)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03!\r)\x03\x05\x07\x04\x06\x05\x05\x01\x11\r/\x07\x03\x01\t\x0b\x11\r;\x07\x03=u\x07\x03]\x1f\x03-\x13\x06c\x03\x05\x03\x01\x15\x07mi\x03\x05\x03\x03\x17\x06s\x03\x17\x03\x05\x19\x06w\x03\x17\x03\x05\x1b\x06{\x03\x17\x03\t\x1d\x06\x7f\x03\x05\x05\x07\x0b\r\x06\x83\x03\x05\x05\x03\r\x05\x03\r\x89\x03\t\x03\x07+\x05\x03\x05\x03\x11\x1f\x06+\x03\x05\x05\x0f\x13!\x07\t\x8f\x03\x05\x03\x15\x05\x03\x01-\x03\x19\x05\x03\x01-\x03\x19#\x07\x01\x97\x07\x05\x11\x0b\x03\x17\x05\x03\x01#\x03\x0b\x03\x07\x01\x05\x03\x0b\x03#\x0f\x07\x01\x16\x02\x035\x05!%\x03\x07\x01\x05\x037\x03\'\x05\x03\x01\x1a\x02\x03\t\x03\x07\x01\x05\x03\x05\x03+\x03\x07\x01\x1e\x02\x03\x1f\x03)\t\x06\x01\x03\x05\x07/\x1d-\x03\x07\x01\x05\x039\x03\'\x05\x03\x01"\x02\x03%\x03\x07\x01\x05\x03\x11\x035\x03\x07\x01&\x02\x03;\x033\t\x06\x01\x03\x11\x079\x1f7\x11\x04\r\x051;\x0b\x11\t=\x07\x03\x15+\x03\x05\t\x07\x03A\x1f\x03\x13\x05\x03\t#\x03\x0b\x03\x07%\x05\x03\x13\x03\x05\r\x06%\x03\x13\x05\x03\x07\x07\x03IG\x03\x13\x0f\x07OM\x03\x1f\x05\t\x0b\x05\x03\tS\x03\t\x03\x07U\x05\x03\x05\x03\x0f\t\x06Y\x03\x05\x07\r\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xe2\x1c\x99\x0b\x0b%\x03\x11\x0f\x0b\t\t\x0b!\x11#\x1f/!)!)#\x1f\x19\xa9\x0f99A9;;m\x19\x85\x8fW\xb3K\x9bM\x9bn\x03\x1b%)9+\x1b\x1f\x1f\x15\x1d\x15+\x13\ri\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/real\x00jit(<lambda>)/jit(main)/imag\x00jit(<lambda>)/jit(main)/neg\x00jit(<lambda>)/jit(main)/complex\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_zheevd_ffi\x00mode\x00uplo\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_19["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cheevd_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([[-0.6185769    +0.j, -0.20142993   +0.j, -0.09725195   +0.j,
         0.62983674   +0.j, -0.07926044   +0.j,  0.3605001    -0.j,
        -0.019093221  +0.j, -0.18446997   +0.j],
       [-0.47070873   +0.j,  0.29325768   +0.j, -0.19454116   +0.j,
        -0.6394365    +0.j,  0.06229549   +0.j,  0.33249345   +0.j,
         0.28112718   +0.j, -0.22856665   +0.j],
       [-0.32284075   +0.j, -0.12361939   +0.j,  0.20547704   +0.j,
        -0.18307868   +0.j,  0.47294614   +0.j, -0.3170349    +0.j,
        -0.6373532    +0.j, -0.27266347   +0.j],
       [-0.17497246   +0.j, -0.079641335  +0.j,  0.15042792   +0.j,
        -0.15416273   +0.j, -0.815209     +0.j, -0.38054234   +0.j,
        -0.083263926  +0.j, -0.31676024   +0.j],
       [-0.027104257  +0.j, -0.26490977   +0.j,  0.32271704   +0.j,
         0.08653544   +0.j,  0.30305928   +0.j, -0.33998996   +0.j,
         0.6926741    +0.j, -0.360857     +0.j],
       [ 0.120763965  +0.j,  0.43288827   +0.j, -0.64385164   +0.j,
         0.2652551    +0.j,  0.094823755  +0.j, -0.37435007   +0.j,
         0.00091664493+0.j, -0.40495378   +0.j],
       [ 0.26863196   +0.j,  0.51607686   +0.j,  0.53846526   +0.j,
         0.16969058   +0.j, -0.0216703    +0.j,  0.35755336   +0.j,
        -0.113144726  +0.j, -0.4490505    +0.j],
       [ 0.4165004    +0.j, -0.57262254   +0.j, -0.28144246   +0.j,
        -0.17463988   +0.j, -0.016984984  +0.j,  0.3613705    +0.j,
        -0.12186296   +0.j, -0.49314725   +0.j]], dtype=complex64), array([-2.4598808e+01, -3.3105560e-05, -3.1002426e-05, -1.0103593e-05,
       -1.0022322e-05,  4.0141886e-06,  9.5510331e-06,  2.7659882e+02],
      dtype=float32)),
    mlir_module_text=r"""
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":277:27)
#loc18 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc7))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x8xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<8xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xcomplex<f32>> loc(#loc9)
    %1 = stablehlo.reshape %0 : (tensor<64xcomplex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc10)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc11)
    %3 = stablehlo.real %2 : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xf32> loc(#loc12)
    %4 = stablehlo.imag %2 : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xf32> loc(#loc13)
    %5 = stablehlo.negate %4 : tensor<8x8xf32> loc(#loc14)
    %6 = stablehlo.complex %3, %5 : tensor<8x8xcomplex<f32>> loc(#loc15)
    %7 = stablehlo.add %1, %6 : tensor<8x8xcomplex<f32>> loc(#loc16)
    %cst = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc17)
    %9 = stablehlo.divide %7, %8 : tensor<8x8xcomplex<f32>> loc(#loc17)
    %10 = call @tril(%9) : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc18)
    %c = stablehlo.constant dense<8> : tensor<i64> loc(#loc19)
    %c_0 = stablehlo.constant dense<8> : tensor<i64> loc(#loc19)
    %11:3 = stablehlo.custom_call @lapack_cheevd_ffi(%10) {mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<8x8xcomplex<f32>>) -> (tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>) loc(#loc19)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc19)
    %12 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc19)
    %13 = stablehlo.compare  EQ, %11#2, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc19)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc19)
    %cst_2 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc19)
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc19)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1> loc(#loc19)
    %17 = stablehlo.select %16, %11#0, %15 : tensor<8x8xi1>, tensor<8x8xcomplex<f32>> loc(#loc19)
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc19)
    %cst_3 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc19)
    %19 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8xf32> loc(#loc19)
    %20 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi1>) -> tensor<8xi1> loc(#loc19)
    %21 = stablehlo.select %20, %11#1, %19 : tensor<8xi1>, tensor<8xf32> loc(#loc19)
    return %17, %21 : tensor<8x8xcomplex<f32>>, tensor<8xf32> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<8x8xcomplex<f32>> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc7))) -> (tensor<8x8xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32> loc(#loc20)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc18)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32> loc(#loc21)
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32> loc(#loc21)
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32> loc(#loc22)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1> loc(#loc23)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc18)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc24)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xcomplex<f32>> loc(#loc25)
    return %6 : tensor<8x8xcomplex<f32>> loc(#loc18)
  } loc(#loc18)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":269:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":269:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:25)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:15)
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:14)
#loc8 = loc("third_party/py/jax/tests/export_back_compat_test.py":277:11)
#loc9 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(64,) dimension=0]"(#loc1))
#loc10 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/real"(#loc4))
#loc13 = loc("jit(<lambda>)/jit(main)/imag"(#loc4))
#loc14 = loc("jit(<lambda>)/jit(main)/neg"(#loc4))
#loc15 = loc("jit(<lambda>)/jit(main)/complex"(#loc4))
#loc16 = loc("jit(<lambda>)/jit(main)/add"(#loc5))
#loc17 = loc("jit(<lambda>)/jit(main)/div"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]"(#loc8))
#loc20 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]"(#loc7))
#loc21 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc7))
#loc22 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]"(#loc7))
#loc23 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc7))
#loc24 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]"(#loc7))
#loc25 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc7))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\'\x03\xda\x02*\x02?\x01\xab\x0f\x0b\x13\x17\x0f\x0b\x07\x17\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x0f\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x0f\x0b\x17\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x13\x0b\x17\x13\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03a\x0b\x0b\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O//\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17\x17/\x0f\x0b/O\x1f/\x01\x0b\x1f\x17\x17\x17\x17\x01\x05\x0b\x0f\x03;\x17\x07\x0f\x0f\x07\x07\x13\x17\x0b\x17\x0f\x07\x07\x17\x13\x07\x0f\x17\x17\x13\x17\x13\x13\x13\x0f\x17\x13\x13\x13\x026\n\x1d\x93\x95\x05)\x03\x03\x13\xd5\x17\x03V\x047\x1d?\x07\x05+\x1f\x17\x03>\x043\x05-\x05/\x11\x03\x05\x051\x053\x055\x057\x03\x03!\xd1\x059\x03\x03\x0b\xd3\x1dE\x07\x05;\x05=\x1d\x8b\x8d\x03\x03\x0b\xe1\x03\t135\x157\x15\x119\x05?\x11\x01\x00\x05A\x05C\x05E\x03\x0b\x17\xaf\x19\xbb\x1b\xbd\x11\xc7\x1d\xc9\x03\x0b\x17\xb3\x19\xcd\x1b\xb3\x11\xb5\x1d\xcf\x05G\x1dC\x07\x05I\x05K\x03\x03!\xd7\x1dK\x07\x05M\x03\x05\'\xb7)\xd9\x1dQ\x07\x05O\x03\x03\x0b\xdb\x1dW\x07\x05Q\x1d[\x07\x05S\x1d_a\x05U\x17\x036\x045\x1deg\x05W\x17\x036\x04\x1d\x03\x03k\xdd\x05Y\x1doq\x05[\x17\x03>\x04E\x1du\x0f\x05]\x1dy\x0f\x05_\x1d}\x0f\x05a\x1d\x81\x0f\x05c\x1d\x85\x87\x05e\x17\x03>\x04\x1f\x03\x03\x0b\xdf\x05g\x17\x03>\x04\x1d\x03\x03\x91\xb5\x05i\x05k\x17\x03V\x04\x17\x03\x13\x99\xe3\x9b\xe5\x9d\xe7\x9f\xaf\xa1\xe9\xa3\xeb\xa5\xf5\xa7\xf7\xa9\xfb\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x05}\x1d\x7f\x1d\x81\x03\x01\x1d\x83\x03\x03\xcb\x1d\x85\t\x07\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\'\x03\x05\xbf\xc3\r\x05\xb1\xc1\xab\xad\x1d\x87\r\x05\xb1\xc5\xab\xad\x1d\x89\x1d\x8b\x1d\x8d\r\x03\xab\xad#)\x1d\x8f\x13\x07\x01\x1f\x0b\t\x00\x00\x00\x00\x1f+\x01\x13\x07\x05\x07\x05\x1f\t\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\t\x11\x00\x00\x00@\x00\x00\x00\x00\x1f\x19\x11\x08\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x91\x1d\x93\x05\x01\r\x05\xed\xef\xf1\xf3\x1d\x95\x13#V\x1d\x97\x13#L\x03\x03\xb9\x03\x03\xf9\x15\x03\x01\x01\x01\x03\x07\xb9\xfd\xff\x1f1\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f3\x01\x07\x01\x1f\t\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f%\t\x00\x00\xc0\x7f\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05\'\xb7)\x02\x02\x03\x03\x0b\x06\x02\x03\x03\x13\n\x02\x03\x03\x0b\x0e\x02\x03\x03\x13\x12\x02\x01\t\x01\x02\x02)\x05!!\x15\x1d)\x01\x15)\x01\x1d\x01\t)\x03!\x0f)\x05!!\x1d\x03\x0f)\x05!!\x0f)\x01\x07\x13\x1b)\x05!!\r)\x03\t\x07!)\x01\x0f\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\x07)\x03\x02\x02\x15)\x03\t\x1b)\x03\x05\x1b)\x03\x01\x1b)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03!\r)\x03\x05\x07\x04\x06\x05\x05\x01\x11\r/\x07\x03\x01\t\x0b\x11\r;\x07\x03=u\x07\x03]\x1f\x03-\x13\x06c\x03\x05\x03\x01\x15\x07mi\x03\x05\x03\x03\x17\x06s\x03\x17\x03\x05\x19\x06w\x03\x17\x03\x05\x1b\x06{\x03\x17\x03\t\x1d\x06\x7f\x03\x05\x05\x07\x0b\r\x06\x83\x03\x05\x05\x03\r\x05\x03\r\x89\x03\t\x03\x07+\x05\x03\x05\x03\x11\x1f\x06+\x03\x05\x05\x0f\x13!\x07\t\x8f\x03\x05\x03\x15\x05\x03\x01-\x03\x19\x05\x03\x01-\x03\x19#\x07\x01\x97\x07\x05\x11\x0b\x03\x17\x05\x03\x01#\x03\x0b\x03\x07\x01\x05\x03\x0b\x03#\x0f\x07\x01\x16\x02\x035\x05!%\x03\x07\x01\x05\x037\x03\'\x05\x03\x01\x1a\x02\x03\t\x03\x07\x01\x05\x03\x05\x03+\x03\x07\x01\x1e\x02\x03\x1f\x03)\t\x06\x01\x03\x05\x07/\x1d-\x03\x07\x01\x05\x039\x03\'\x05\x03\x01"\x02\x03%\x03\x07\x01\x05\x03\x11\x035\x03\x07\x01&\x02\x03;\x033\t\x06\x01\x03\x11\x079\x1f7\x11\x04\r\x051;\x0b\x11\t=\x07\x03\x15+\x03\x05\t\x07\x03A\x1f\x03\x13\x05\x03\t#\x03\x0b\x03\x07%\x05\x03\x13\x03\x05\r\x06%\x03\x13\x05\x03\x07\x07\x03IG\x03\x13\x0f\x07OM\x03\x1f\x05\t\x0b\x05\x03\tS\x03\t\x03\x07U\x05\x03\x05\x03\x0f\t\x06Y\x03\x05\x07\r\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xde\x1c\x99\x0b\x0b%\x03\x11\x0f\x0b\t\t\x0b!\x11#\x1f/!)!)#\x1f\x19\xa9\x0f99A9;;m\x19\x85\x8dW\xb3K\x9bM\x9bn\x03\x1b%)9+\x1b\x1f\x1f\x15\x1d\x15+\x13\ri\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/real\x00jit(<lambda>)/jit(main)/imag\x00jit(<lambda>)/jit(main)/neg\x00jit(<lambda>)/jit(main)/complex\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_cheevd_ffi\x00mode\x00uplo\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_19["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_ssyevd_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([[-0.6185769    , -0.20142993   , -0.09725195   ,  0.62983674   ,
        -0.07926044   ,  0.3605001    , -0.019093221  , -0.18446997   ],
       [-0.47070873   ,  0.29325768   , -0.19454119   , -0.6394365    ,
         0.0622955    ,  0.33249345   ,  0.28112718   , -0.22856665   ],
       [-0.32284075   , -0.12361939   ,  0.20547704   , -0.18307868   ,
         0.47294614   , -0.3170349    , -0.6373532    , -0.27266347   ],
       [-0.17497246   , -0.079641335  ,  0.15042791   , -0.15416273   ,
        -0.815209     , -0.38054234   , -0.083263926  , -0.31676024   ],
       [-0.027104253  , -0.26490977   ,  0.32271704   ,  0.08653544   ,
         0.30305928   , -0.33998996   ,  0.6926741    , -0.360857     ],
       [ 0.12076397   ,  0.43288827   , -0.64385164   ,  0.2652551    ,
         0.09482376   , -0.37435007   ,  0.00091664493, -0.40495378   ],
       [ 0.26863196   ,  0.51607686   ,  0.53846526   ,  0.16969058   ,
        -0.021670295  ,  0.35755336   , -0.113144726  , -0.4490505    ],
       [ 0.4165004    , -0.57262254   , -0.2814425    , -0.17463988   ,
        -0.01698498   ,  0.3613705    , -0.12186296   , -0.49314725   ]],
      dtype=float32), array([-2.4598808e+01, -3.3105560e-05, -3.1002426e-05, -1.0103593e-05,
       -1.0022322e-05,  4.0141886e-06,  9.5510331e-06,  2.7659882e+02],
      dtype=float32)),
    mlir_module_text=r"""
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":277:27)
#loc13 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc6))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x8xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<8xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xf32> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<64xf32>) -> tensor<8x8xf32> loc(#loc9)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32> loc(#loc10)
    %3 = stablehlo.add %1, %2 : tensor<8x8xf32> loc(#loc11)
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32> loc(#loc)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x8xf32> loc(#loc12)
    %5 = stablehlo.divide %3, %4 : tensor<8x8xf32> loc(#loc12)
    %6 = call @tril(%5) : (tensor<8x8xf32>) -> tensor<8x8xf32> loc(#loc13)
    %c = stablehlo.constant dense<8> : tensor<i64> loc(#loc14)
    %c_0 = stablehlo.constant dense<8> : tensor<i64> loc(#loc14)
    %7:3 = stablehlo.custom_call @lapack_ssyevd_ffi(%6) {mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8xf32>, tensor<i32>) loc(#loc14)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc14)
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc14)
    %9 = stablehlo.compare  EQ, %7#2, %8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc14)
    %cst_2 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc14)
    %11 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x8xf32> loc(#loc14)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1> loc(#loc14)
    %13 = stablehlo.select %12, %7#0, %11 : tensor<8x8xi1>, tensor<8x8xf32> loc(#loc14)
    %14 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc14)
    %cst_3 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc14)
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8xf32> loc(#loc14)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xi1>) -> tensor<8xi1> loc(#loc14)
    %17 = stablehlo.select %16, %7#1, %15 : tensor<8xi1>, tensor<8xf32> loc(#loc14)
    return %13, %17 : tensor<8x8xf32>, tensor<8xf32> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<8x8xf32> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc6))) -> (tensor<8x8xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32> loc(#loc15)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc13)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32> loc(#loc16)
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32> loc(#loc16)
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32> loc(#loc17)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1> loc(#loc18)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc13)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x8xf32> loc(#loc19)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xf32> loc(#loc20)
    return %6 : tensor<8x8xf32> loc(#loc13)
  } loc(#loc13)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":269:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":269:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:15)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:14)
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":277:11)
#loc8 = loc("jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(64,) dimension=0]"(#loc1))
#loc9 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/add"(#loc4))
#loc12 = loc("jit(<lambda>)/jit(main)/div"(#loc5))
#loc14 = loc("jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]"(#loc7))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]"(#loc6))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc6))
#loc17 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]"(#loc6))
#loc18 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]"(#loc6))
#loc20 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc6))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01+\x05\x01\x03\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03\x96\x02\xff9\x01\xa1\x0f\x13\x17\x0b\x0f\x0b\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x0f\x13\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x17\x13\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x03_\x0b\x0b\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1fO\x1f/\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17\x17/\x0f\x0b\x1fO/\x01\x05\x0b\x0f\x035\x17\x0f\x07\x0f\x07\x07\x13\x17\x0f\x07\x07\x17\x13\x07\x17\x17\x13\x17\x13\x13\x13\x0f\x17\x13\x13\x13\x02:\t\x1d\x83\x85\x03\x03\x11\xcb\x17\x07V\x047\x05!\x1d?\x05\x05#\x1f\x05%\x05'\x11\x03\x05\x05)\x05+\x05-\x05/\x03\x03\x1f\xc7\x051\x03\x03\x0b\xc9\x1dE\x05\x053\x055\x1d{}\x03\x03\x0b\xd7\x03\x03\x0b\xf9\x03\t135\x137\x13\x0f9\x057\x11\x01\x00\x059\x05;\x05=\x03\x0b\x15\xa5\x17\xb1\x19\xb3\x0f\xbd\x1b\xbf\x03\x0b\x15\xa9\x17\xc3\x19\xa9\x0f\xab\x1b\xc5\x05?\x1dC\x05\x05A\x05C\x03\x03\x1f\xcd\x1dK\x05\x05E\x03\x05%\xad'\xcf\x1dQ\x05\x05G\x03\x03\x0b\xd1\x1dW\x05\x05I\x1d[\x05\x05K\x1d_a\x05M\x17\x076\x045\x1deg\x05O\x17\x076\x04\x1d\x03\x03k\xd3\x05Q\x1doq\x05S\x17\x07>\x04E\x1duw\x05U\x17\x07>\x04\x1f\x03\x03\x0b\xd5\x05W\x17\x07>\x04\x1d\x03\x03\x81\xab\x05Y\x05[\x17\x07V\x04\x17\x03\x13\x89\xd9\x8b\xdb\x8d\xdd\x8f\xa5\x91\xdf\x93\xe1\x95\xeb\x97\xed\x99\xf1\x05]\x05_\x05a\x05c\x05e\x05g\x05i\x05k\x05m\x03\x05%\xad'\xf7\x03\x03\x11\xfb\x03\x03\x11\xfd\x1do\x1dq\x03\x01\x1ds\x03\x03\xc1\x1du\t\x07\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#!\x03\x05\xb5\xb9\r\x05\xa7\xb7\xa1\xa3\x1dw\r\x05\xa7\xbb\xa1\xa3\x1dy\x1d{\x1d}\r\x03\xa1\xa3##\x1d\x7f\x13\t\x01\x1f\x0b\t\x00\x00\x00\x00\x1f%\x01\x13\t\x05\x07\x05\x1f\x07\t\x00\x00\x00\x00\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00@\x1f\x15\x11\x08\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x81\x1d\x83\x05\x01\r\x05\xe3\xe5\xe7\xe9\x1d\x85\x13\x1fV\x1d\x87\x13\x1fL\x03\x03\xaf\x03\x03\xef\x15\x03\x01\x01\x01\x03\x07\xaf\xf3\xf5\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x01\x07\x01\x1f\x07\t\x00\x00\xc0\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05!!\x0f)\x01\x0f\x1d)\x01\x19\x01\t)\x03!\x0f)\x05!!\x19)\x01\t\x13\x1b)\x05!!\r)\x03\t\t!\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\t)\x03\x02\x02\x0f)\x03\t\x17)\x03\x05\x17)\x03\x01\x17)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03!\r)\x03\x05\t\x04~\x04\x05\x01\x11\r/\x07\x03\x01\t\x0b\x11\r;\x07\x035e\x07\x03]\x1d\x03'\x13\x06c\x03\x05\x03\x01\x15\x07mi\x03\x05\x03\x03\r\x06s\x03\x05\x05\x03\x05\x05\x03\ry\x03\x07\x03\x07)\x03\x03\x05\x03\t\x17\x06)\x03\x05\x05\x07\x0b\x19\x07\t\x7f\x03\x05\x03\r\x05\x03\x01+\x03\x15\x05\x03\x01+\x03\x15\x1b\x07\x01\x87\x07\x05\x11\x0b\x03\x0f\x05\x03\x01!\x03\x0b\x03\x07\x01\x03\x03\x0b\x03\x1b\x0f\x07\x01\x9b\x03/\x05\x19\x1d\x03\x07\x01\x03\x031\x03\x1f\x05\x03\x01-\x03\x07\x03\x07\x01\x03\x03\x05\x03#\x03\x07\x01\x9d\x03\x1b\x03!\t\x06\x01\x03\x05\x07'\x15%\x03\x07\x01\x03\x033\x03\x1f\x05\x03\x01-\x03\x07\x03\x07\x01\x03\x03\x11\x03-\x03\x07\x01\x9f\x035\x03+\t\x06\x01\x03\x11\x071\x17/\x11\x04\r\x05)3\x0b\x11\t=\x07\x03\x15+\x03\x05\t\x07\x03A\x1d\x03\x13\x05\x03\t!\x03\x0b\x03\x07#\x03\x03\x13\x03\x05\r\x06#\x03\x13\x05\x03\x07\x07\x03IG\x03\x13\x0f\x07OM\x03\x1b\x05\t\x0b\x05\x03\tS\x03\x07\x03\x07U\x03\x03\x05\x03\x0f\t\x06Y\x03\x05\x07\r\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00J\x1a\x89\x0b\x0b%\x03\x11\x0f\x0b\t\t\x0b!\x11#\x1f/!)!)#\x1f\x19\xa9\x0f99m\x19\x85\x89W\xb3K\x9bM\x9bn\x03\x1b%)9+\x1b\x1f\x1f\x15\x1d\x15+\x13\ri\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_ssyevd_ffi\x00mode\x00uplo\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_19["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dsyevd_ffi'],
    serialized_date=datetime.date(2024, 8, 19),
    inputs=(),
    expected_outputs=(array([[-6.1857700048412056e-01,  2.4081403770912022e-01,
         3.5662489253627483e-01, -6.3034019033669797e-01,
         1.0043483479985752e-16, -2.8842036081919542e-02,
         7.7164692943283169e-25, -1.8446994643771725e-01],
       [-4.7070881487314614e-01,  4.7473787464450845e-01,
        -4.8036836210243367e-01,  4.3802686872516400e-01,
         1.7961797619639258e-01,  8.3080980076741355e-03,
         2.1415294457221756e-01, -2.2856669794666584e-01],
       [-3.2284062926217072e-01, -5.4336490915553370e-01,
         2.2181041859724990e-01,  2.9947877954402297e-01,
        -3.6491813600134632e-01,  3.2867679819727436e-01,
         3.8223299448843473e-01, -2.7266344945561438e-01],
       [-1.7497244365119530e-01, -8.9251550609769414e-02,
        -6.3518515114898394e-02,  1.9162997359209971e-01,
        -2.2087281326110139e-01,  5.9957027043505064e-02,
        -8.7632498908241274e-01, -3.1676020096456303e-01],
       [-2.7104258040220038e-02, -3.3772873786627672e-01,
         2.5901386593721748e-01,  1.7032650752287815e-01,
         6.7521217612940332e-01, -4.5036136532965476e-01,
        -1.2279030059078447e-02, -3.6085695247351163e-01],
       [ 1.2076392757075530e-01, -3.3834734096469254e-01,
        -6.5506827461665540e-01, -5.0472498521116749e-01,
         6.9987430903492118e-02,  1.0595648906599275e-01,
         8.3443844143082022e-02, -4.0495370398246017e-01],
       [ 2.6863211318173097e-01,  2.2958613191407318e-01,
         6.3952843755683941e-02,  1.8776775771084137e-02,
        -5.3523731432241317e-01, -5.9199531677602002e-01,
         1.7916671834524248e-01, -4.4905045549140887e-01],
       [ 4.1650029879270661e-01,  3.6355449432857079e-01,
         2.9755313100756142e-01,  1.6826270392615944e-02,
         1.9621068035557282e-01,  5.6830030587314817e-01,
         2.9607517592514246e-02, -4.9314720700035747e-01]]), array([-2.4598804776133626e+01, -4.6567755957874661e-14,
       -1.9932120610662194e-14, -5.7323356091157378e-15,
       -4.5459724251334835e-16,  4.0479851042511616e-14,
        9.2325194924982089e-14,  2.7659880477613365e+02])),
    mlir_module_text=r"""
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":277:27)
#loc13 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc6))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x8xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<8xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xf64> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<64xf64>) -> tensor<8x8xf64> loc(#loc9)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xf64>) -> tensor<8x8xf64> loc(#loc10)
    %3 = stablehlo.add %1, %2 : tensor<8x8xf64> loc(#loc11)
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64> loc(#loc)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<8x8xf64> loc(#loc12)
    %5 = stablehlo.divide %3, %4 : tensor<8x8xf64> loc(#loc12)
    %6 = call @tril(%5) : (tensor<8x8xf64>) -> tensor<8x8xf64> loc(#loc13)
    %c = stablehlo.constant dense<8> : tensor<i64> loc(#loc14)
    %c_0 = stablehlo.constant dense<8> : tensor<i64> loc(#loc14)
    %7:3 = stablehlo.custom_call @lapack_dsyevd_ffi(%6) {mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<8xf64>, tensor<i32>) loc(#loc14)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc14)
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc14)
    %9 = stablehlo.compare  EQ, %7#2, %8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc14)
    %cst_2 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc14)
    %11 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<8x8xf64> loc(#loc14)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1> loc(#loc14)
    %13 = stablehlo.select %12, %7#0, %11 : tensor<8x8xi1>, tensor<8x8xf64> loc(#loc14)
    %14 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc14)
    %cst_3 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc14)
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<8xf64> loc(#loc14)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xi1>) -> tensor<8xi1> loc(#loc14)
    %17 = stablehlo.select %16, %7#1, %15 : tensor<8xi1>, tensor<8xf64> loc(#loc14)
    return %13, %17 : tensor<8x8xf64>, tensor<8xf64> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<8x8xf64> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc6))) -> (tensor<8x8xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32> loc(#loc15)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc13)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32> loc(#loc16)
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32> loc(#loc16)
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32> loc(#loc17)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1> loc(#loc18)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc13)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<8x8xf64> loc(#loc19)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xf64> loc(#loc20)
    return %6 : tensor<8x8xf64> loc(#loc13)
  } loc(#loc13)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":269:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":269:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:15)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":271:14)
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":277:11)
#loc8 = loc("jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(64,) dimension=0]"(#loc1))
#loc9 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/add"(#loc4))
#loc12 = loc("jit(<lambda>)/jit(main)/div"(#loc5))
#loc14 = loc("jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]"(#loc7))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]"(#loc6))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc6))
#loc17 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]"(#loc6))
#loc18 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]"(#loc6))
#loc20 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc6))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01+\x05\x01\x03\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03\x96\x02\xff9\x01\xa1\x0f\x13\x17\x0b\x0f\x0b\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x0f\x13\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x17\x13\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x03_\x0b\x0b\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O//\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17\x17/\x0f\x0b/O/\x01\x05\x0b\x0f\x035\x17\x0f\x07\x0f\x07\x07\x13\x17\x0f\x07\x07\x17\x13\x07\x17\x17\x13\x17\x13\x13\x13\x0f\x17\x13\x13\x13\x02j\t\x1d\x83\x85\x03\x03\x11\xcb\x17\x07V\x047\x05!\x1d?\x05\x05#\x1f\x05%\x05'\x11\x03\x05\x05)\x05+\x05-\x05/\x03\x03\x1f\xc7\x051\x03\x03\x0b\xc9\x1dE\x05\x053\x055\x1d{}\x03\x03\x0b\xd7\x03\x03\x0b\xf9\x03\t135\x137\x13\x0f9\x057\x11\x01\x00\x059\x05;\x05=\x03\x0b\x15\xa5\x17\xb1\x19\xb3\x0f\xbd\x1b\xbf\x03\x0b\x15\xa9\x17\xc3\x19\xa9\x0f\xab\x1b\xc5\x05?\x1dC\x05\x05A\x05C\x03\x03\x1f\xcd\x1dK\x05\x05E\x03\x05%\xad'\xcf\x1dQ\x05\x05G\x03\x03\x0b\xd1\x1dW\x05\x05I\x1d[\x05\x05K\x1d_a\x05M\x17\x076\x045\x1deg\x05O\x17\x076\x04\x1d\x03\x03k\xd3\x05Q\x1doq\x05S\x17\x07>\x04E\x1duw\x05U\x17\x07>\x04\x1f\x03\x03\x0b\xd5\x05W\x17\x07>\x04\x1d\x03\x03\x81\xab\x05Y\x05[\x17\x07V\x04\x17\x03\x13\x89\xd9\x8b\xdb\x8d\xdd\x8f\xa5\x91\xdf\x93\xe1\x95\xeb\x97\xed\x99\xf1\x05]\x05_\x05a\x05c\x05e\x05g\x05i\x05k\x05m\x03\x05%\xad'\xf7\x03\x03\x11\xfb\x03\x03\x11\xfd\x1do\x1dq\x03\x01\x1ds\x03\x03\xc1\x1du\t\x07\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#!\x03\x05\xb5\xb9\r\x05\xa7\xb7\xa1\xa3\x1dw\r\x05\xa7\xbb\xa1\xa3\x1dy\x1d{\x1d}\r\x03\xa1\xa3##\x1d\x7f\x13\t\x01\x1f\x0b\t\x00\x00\x00\x00\x1f%\x01\x13\t\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x15\x11\x08\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x81\x1d\x83\x05\x01\r\x05\xe3\xe5\xe7\xe9\x1d\x85\x13\x1fV\x1d\x87\x13\x1fL\x03\x03\xaf\x03\x03\xef\x15\x03\x01\x01\x01\x03\x07\xaf\xf3\xf5\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x01\x07\x01\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05!!\x0f)\x01\x0f\x1d)\x01\x19\x01\x0b)\x03!\x0f)\x05!!\x19)\x01\t\x13\x1b)\x05!!\r)\x03\t\t!\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\t)\x03\x02\x02\x0f)\x03\t\x17)\x03\x05\x17)\x03\x01\x17)\x01\r)\x05\x05\x05\r)\x03\x05\r)\x03!\r)\x03\x05\t\x04~\x04\x05\x01\x11\r/\x07\x03\x01\t\x0b\x11\r;\x07\x035e\x07\x03]\x1d\x03'\x13\x06c\x03\x05\x03\x01\x15\x07mi\x03\x05\x03\x03\r\x06s\x03\x05\x05\x03\x05\x05\x03\ry\x03\x07\x03\x07)\x03\x03\x05\x03\t\x17\x06)\x03\x05\x05\x07\x0b\x19\x07\t\x7f\x03\x05\x03\r\x05\x03\x01+\x03\x15\x05\x03\x01+\x03\x15\x1b\x07\x01\x87\x07\x05\x11\x0b\x03\x0f\x05\x03\x01!\x03\x0b\x03\x07\x01\x03\x03\x0b\x03\x1b\x0f\x07\x01\x9b\x03/\x05\x19\x1d\x03\x07\x01\x03\x031\x03\x1f\x05\x03\x01-\x03\x07\x03\x07\x01\x03\x03\x05\x03#\x03\x07\x01\x9d\x03\x1b\x03!\t\x06\x01\x03\x05\x07'\x15%\x03\x07\x01\x03\x033\x03\x1f\x05\x03\x01-\x03\x07\x03\x07\x01\x03\x03\x11\x03-\x03\x07\x01\x9f\x035\x03+\t\x06\x01\x03\x11\x071\x17/\x11\x04\r\x05)3\x0b\x11\t=\x07\x03\x15+\x03\x05\t\x07\x03A\x1d\x03\x13\x05\x03\t!\x03\x0b\x03\x07#\x03\x03\x13\x03\x05\r\x06#\x03\x13\x05\x03\x07\x07\x03IG\x03\x13\x0f\x07OM\x03\x1b\x05\t\x0b\x05\x03\tS\x03\x07\x03\x07U\x03\x03\x05\x03\x0f\t\x06Y\x03\x05\x07\r\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00J\x1a\x89\x0b\x0b%\x03\x11\x0f\x0b\t\t\x0b!\x11#\x1f/!)!)#\x1f\x19\xa9\x0f99m\x19\x85\x89W\xb3K\x9bM\x9bn\x03\x1b%)9+\x1b\x1f\x1f\x15\x1d\x15+\x13\ri\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_dsyevd_ffi\x00mode\x00uplo\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
