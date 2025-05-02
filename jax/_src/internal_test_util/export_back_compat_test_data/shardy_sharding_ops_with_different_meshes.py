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
from numpy import array, float32


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_02_12 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['Sharding', 'xla.sdy.GlobalToLocalShape', 'xla.sdy.LocalToGlobalShape'],
    serialized_date=datetime.date(2025, 2, 12),
    inputs=(array([[0., 1., 2., 3.],
       [4., 5., 6., 7.]], dtype=float32),),
    expected_outputs=(array([[4., 5., 6., 7.],
       [0., 1., 2., 3.]], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":1052:13)
#loc6 = loc("jit(func)/jit(main)/shard_map"(#loc3))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\\\22a\\\22=2]>}"}, mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4xf32> loc("x")) -> (tensor<2x4xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}]>]>"}, mhlo.sharding = "{devices=[2,1]<=[2]}"} : (tensor<2x4xf32>) -> tensor<2x4xf32> loc(#loc5)
    %1 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%0) : (tensor<2x4xf32>) -> tensor<1x4xf32> loc(#loc6)
    %2 = call @xla.sdy.manual_computation_body(%1) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}]>]>"}} : (tensor<1x4xf32>) -> tensor<1x4xf32> loc(#loc6)
    %3 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%2) : (tensor<1x4xf32>) -> tensor<2x4xf32> loc(#loc6)
    return %3 : tensor<2x4xf32> loc(#loc)
  } loc(#loc)
  func.func @xla.sdy.manual_computation_body(%arg0: tensor<1x4xf32> loc("jit(func)/jit(main)/shard_map"(#loc3))) -> tensor<1x4xf32> {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32> loc(#loc7)
    return %0 : tensor<1x4xf32> loc(#loc6)
  } loc(#loc6)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":1051:10)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":1050:15)
#loc5 = loc("jit(func)/jit(main)/sharding_constraint"(#loc2))
#loc7 = loc("jit(func)/jit(main)/ppermute"(#loc4))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.8.8\x00\x01\x1d\x05\x01\x05\r\x01\x03\x0b\x03\x0b\x0f\x13\x17\x1b\x1f\x03\x97q\x13\x019\x0f\x07\x0b\x0b+\x0b\x0f\x13\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x17\x0f\x0b\x17\x0f\x0b\x1b\x0b\x0f\x0b\x17\x13\x039\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0f\x8f\x13\x0b\x0b\x0b\x0b#\x0b\x0b\x0b\x0b\x0b\x01\x05\x0f\x0b\x03\x0f\x17\x17\x07\x07\x17\x17\x17\x02v\x03\x1d\x1f!\x1f\x05\x11\x05\x13\x03\t\x0b\r\x05\x0f\x15\x17\x19\x1b\x05\x15\x11\x03\x00\x03\x03\x11\x13\x05\x17\x05\x19\x05\x1b\x11\x01\t\x05\x1d\x11\x01\x05\x05\x1f\x05!\x17\x07r\x10\x1b\x1d%\'\x05#\x17\x07j\x10\x1f\x1d+\x03\x05%\x03\x05\x05[/_\x05\'\x1d35\x05)\x17\x07n\x10\x15\x03\x03\x05e\x03\x01\x1d+\x1d-\x0b\x03\x05\x01\x1d/\x03\x03G\r\x01#\r\x03\x03M\r\x03O;\x1d1\x1d3\x1d5#\x0f\x13\x0b\x05\x1f\x11A\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03]=\x1d7\x1d9\x1d;\x1d=\r\x07g=ikm=\x1d?\x1dA\x1dC\x1dE\x1dG\x01\x02\x02\x01\t)\x05\x05\x11\t)\x05\t\x11\t\t\x1d\x11\x03\x07\x03\x07\x11\x03\x05\x03\x05)\x05\t\t\x0b\x04\xb9\x05\x01Q\x03\t\x01\x07\x04\xa7\x03\x01\t\x05P\x03\x03\x07\x04]\x03\x0b\x17\x03\x0f)\x00\x03G1-\x05\x03\x07\x03\x01\x03F\x01\x07\x03\x05\x03\x03\x0bG\x017\t\x03\x05\x03\x05\x03F\x01\x0b\x03\x07\x03\x07\x07\x04\x03\x03\t\x05P\x01\r\x07\x04)\x03\x05\x0b\x03\x0b\x01\x00\tF#\x0f\x03\x05\x03\x01\x07\x04\x01\x03\x03\x06\x03\x01\x05\x01\x00r\x0bI7-3)+7\x13+#\x0f\x0b!Ae\x03Q\x1d\x05;=\x13%)=\x1f9i3\x11-\x15\x11\x1f\x0f\x0b\x11builtin\x00vhlo\x00module\x00custom_call_v1\x00func_v1\x00return_v1\x00collective_permute_v1\x00call_v1\x00mhlo.frontend_attributes\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.uses_shape_polymorphism\x00xla.sdy.meshes\x00{mesh = #sdy.mesh<[\\"a\\"=2]>}\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/shard_map\x00jit(func)/jit(main)/ppermute\x00x\x00mhlo.sharding\x00jit(func)/jit(main)/sharding_constraint\x00\x00#sdy.sharding_per_value<[<@mesh, [{\\"a\\"}, {}]>]>\x00xla.sdy.manual_computation_body\x00jax.result_info\x00main\x00public\x00xla.sdy.sharding\x00{devices=[2,1]<=[2]}\x00Sharding\x00xla.sdy.GlobalToLocalShape\x00xla.sdy.in_shardings\x00xla.sdy.manual_axes\x00#sdy<manual_axes{\\"a\\"}>\x00xla.sdy.out_shardings\x00xla.sdy.LocalToGlobalShape\x00\x08a\x11\x05;\x01\x0bEIKQS\x11?;a9A999\x11?;c9A999\x03C\x11?;o9A999\x0b9U9C;\x05WY',
    xla_call_module_version=9,
    nr_devices=2,
)


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_04_14 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['Sharding', 'xla.sdy.GlobalToLocalShape', 'xla.sdy.LocalToGlobalShape'],
    serialized_date=datetime.date(2025, 4, 14),
    inputs=(array([[0., 1., 2., 3.],
       [4., 5., 6., 7.]], dtype=float32),),
    expected_outputs=(array([[4., 5., 6., 7.],
       [0., 1., 2., 3.]], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":1017:8 to :54)
#loc4 = loc("third_party/py/absl/testing/absltest.py":2872:19 to :56)
#loc5 = loc("third_party/py/absl/testing/absltest.py":2908:35 to 2910:3)
#loc6 = loc("third_party/py/absl/testing/absltest.py":2449:6 to :34)
#loc7 = loc("third_party/py/absl/app.py":404:13 to :23)
#loc8 = loc("third_party/py/absl/app.py":484:6 to :27)
#loc9 = loc("third_party/py/absl/testing/absltest.py":2451:4 to :31)
#loc10 = loc("third_party/py/absl/testing/absltest.py":2333:2 to :38)
#loc11 = loc("third_party/py/jax/tests/export_back_compat_test.py":1021:2 to :47)
#loc12 = loc("third_party/py/jax/tests/export_back_compat_test.py":1008:13 to :30)
#loc15 = loc("ShardyCompatTest.test_shardy_sharding_ops_with_different_meshes"(#loc3))
#loc16 = loc("_run_and_get_tests_result"(#loc4))
#loc17 = loc("run_tests"(#loc5))
#loc18 = loc("_run_in_app.<locals>.main_function"(#loc6))
#loc19 = loc("_run_main"(#loc7))
#loc20 = loc("run"(#loc8))
#loc21 = loc("_run_in_app"(#loc9))
#loc22 = loc("main"(#loc10))
#loc23 = loc("<module>"(#loc11))
#loc24 = loc("ShardyCompatTest.test_shardy_sharding_ops_with_different_meshes.<locals>.func"(#loc12))
#loc26 = loc(callsite(#loc22 at #loc23))
#loc28 = loc(callsite(#loc21 at #loc26))
#loc30 = loc(callsite(#loc20 at #loc28))
#loc32 = loc(callsite(#loc19 at #loc30))
#loc34 = loc(callsite(#loc18 at #loc32))
#loc36 = loc(callsite(#loc17 at #loc34))
#loc38 = loc(callsite(#loc16 at #loc36))
#loc40 = loc(callsite(#loc15 at #loc38))
#loc43 = loc(callsite(#loc24 at #loc40))
#loc46 = loc("jit(func)/jit(main)/shard_map"(#loc43))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22a\22=2]>}"}, mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4xf32> loc("x")) -> (tensor<2x4xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}, {}]>]>"}, mhlo.sharding = "{devices=[2,1]<=[2]}"} : (tensor<2x4xf32>) -> tensor<2x4xf32> loc(#loc45)
    %1 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}, {}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>"}} : (tensor<2x4xf32>) -> tensor<1x4xf32> loc(#loc46)
    %2 = call @xla.sdy.manual_computation_body(%1) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}, {}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}, {}]>]>"}} : (tensor<1x4xf32>) -> tensor<1x4xf32> loc(#loc46)
    %3 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%2) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}, {}]>]>"}} : (tensor<1x4xf32>) -> tensor<2x4xf32> loc(#loc46)
    return %3 : tensor<2x4xf32> loc(#loc)
  } loc(#loc)
  func.func @xla.sdy.manual_computation_body(%arg0: tensor<1x4xf32> loc("jit(func)/jit(main)/shard_map"(#loc43))) -> tensor<1x4xf32> {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32> loc(#loc47)
    return %0 : tensor<1x4xf32> loc(#loc46)
  } loc(#loc46)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":1007:10 to :73)
#loc13 = loc("third_party/py/jax/tests/export_back_compat_test.py":1006:15 to :46)
#loc14 = loc("ShardyCompatTest.test_shardy_sharding_ops_with_different_meshes.<locals>.func"(#loc2))
#loc25 = loc("ShardyCompatTest.test_shardy_sharding_ops_with_different_meshes.<locals>.func.<locals>.shard_map_func"(#loc13))
#loc27 = loc(callsite(#loc21 at #loc22))
#loc29 = loc(callsite(#loc20 at #loc27))
#loc31 = loc(callsite(#loc19 at #loc29))
#loc33 = loc(callsite(#loc18 at #loc31))
#loc35 = loc(callsite(#loc17 at #loc33))
#loc37 = loc(callsite(#loc16 at #loc35))
#loc39 = loc(callsite(#loc15 at #loc37))
#loc41 = loc(callsite(#loc24 at #loc39))
#loc42 = loc(callsite(#loc14 at #loc40))
#loc44 = loc(callsite(#loc25 at #loc41))
#loc45 = loc("jit(func)/jit(main)/sharding_constraint"(#loc42))
#loc47 = loc("jit(func)/jit(main)/ppermute"(#loc44))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.9.5\x00\x01\x1d\x05\x01\x05\r\x01\x03\x0b\x03\x0b\x0f\x13\x17\x1b\x1f\x03\x1a\x02\xe7\x13\x01\xa7\x0f\x0b\x0b\x0b\x07\x0f\x0b\x0f\x0f\x0f\x0f\x0f\x0f\x0b\x0f\x0f\x0f+\x0b\x0f\x13\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0b\x0f\x1f\x0b\x1f\x0f\x0b\x1f\x0f\x0b\'\x0f\x0b\x1f\x0f\x0b\x1f\x0f\x0b\x1f\x0f\x0b\x1f\x0f\x0b\x1f\x0f\x0b\x1f\x0f\x0b\x0f\x0f\x0b\x1f\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x1f\x13\x13\x13\x03A\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0f\x8f\x13\x0b\x0b\x0b\x0b\x1b\x0b#\x1b\x0b\x01\x05\x0f\x0b\x03\x0f\x17\x17\x07\x07\x17\x17\x17\x02\xce\x06\x1d9;\x05\x11\x05\x13\x05\x15\x1f\x1d\r=\x05\x17\x15\x11C\x1d?A\x1dEG\x1dKM\x1dQS\x1dWY\x05\x19\x1d]_\x1dce\x1dik\x03\t%\'\x03)/135\x05\x1b\x11\x03\x00\x03\x03+-\x05\x1d\x05\x1f\x05!\x11\x01\t\x05#\x11\x01\x05\x05%\x05\'\x15\x0b\x0f-\x05\x07\xc2\x0f\x1b=\x05)-\x05\x07\xe6\x0f\x11m\x15\x13I\x05+-\x07\x07\xe2,\'q\x15\x15O\x05--\x07\tr-Gz-\x07\x15\x17U\x05/-\x07\x07F&\rE\x15\x19[\x051-\x1b\x07R\x06\x1b/\x15\x1da\x053-\x1b\x07\x92\x07\r7\x15\x1fg\x055-\x07\x07N&\t?\x15!m\x057-\x07\x07v$\x05M\x1doq\x059-\x05\x07\xf6\x0f\x05_\x1duw\x05;\x15y\x7f\x1d{}\x05=-\x05\x07\xba\x0f\x1f]\x15\x0b\x81\x15\x11\x83\x15\x13\x85\x15\x15\x87\x15\x17\x89\x15\x19\x8b\x15\x1d\x8d\x15\x1f!\x1d\x91\t\x05?\x03\x05\x03\xd3\x95\xd7\x05A\x1d\x99\x9b\x05C\x15\x9d\x0f\x1d\r\x9f-\x05\x07\xbe\x0f\x15\x93\x03\x03\x03\xdd\x03\x03\x03\xe1\x03\x03\x03\xe3\x03\x01\x1dE\x1dG\x0b\x03\x1dI\x1dK\x1dM\x1dO\x05\x03\x1dQ\x03\x03\xbd\r\x01#\r\x03\x03\xc3\r\x03\xc5\xc7\x1dS\x1dU\x1d7\x1dW#\x0f\x13\x0b\x05\x1f\x11A\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\r\x03\xd5\xa9\x1dY\x1d[\x1d]\x05\x01\r\x05\xb5\xa9\xaf\xb1\x1d_\r\x07\xb5\xa9\xaf\xb1\xb9\xa9\r\x05\xaf\xb1\xb9\xa9\x1da\x01\x02\x02\x01\t)\x05\x05\x11\t)\x05\t\x11\t\t\x1d\x11\x03\x07\x03\x07\x11\x03\x05\x03\x05)\x05\t\t\x0b\x04\xbd\x05\x01Q\t#\x01\x07\x04\xab\x03\x01\t\x05P\t\x03\x07\x04a\x03\x0b\x17\x03\x0f\x8f\x00\x03G\x97\x93\x05\x03\x07\x03\x01\x03G\x01\xa1\x07\x03\x05\x03\x03\x0bG\x01\xa3\t\x03\x05\x03\x05\x03G\x01\xa5\x0b\x03\x07\x03\x07\x07\x04\t\x03\t\x05P\x01\r\x07\x04)\x03\x05\x0b\x03\x0b\x01\x00\tFs\x0f\x03\x05\x03\x01\x07\x04\x01\x03\x03\x06\x03\x01\x05\x01\x00.\x12c77\x13+#\x0f\x0f!-+A/)\x03aQ\x1d\x05\xcd;\x13\x0b\x19\t\x15G\x155\x81=\x13%)9\x1f97\x9dQi3\x11-\x15\x11\x1f\x0f\x0b\x11builtin\x00vhlo\x00module\x00custom_call_v1\x00func_v1\x00return_v1\x00collective_permute_v1\x00call_v1\x00mhlo.frontend_attributes\x00third_party/py/jax/tests/export_back_compat_test.py\x00third_party/py/absl/testing/absltest.py\x00ShardyCompatTest.test_shardy_sharding_ops_with_different_meshes.<locals>.func\x00third_party/py/absl/app.py\x00jax.uses_shape_polymorphism\x00xla.sdy.meshes\x00{mesh = #sdy.mesh<["a"=2]>}\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/shard_map\x00ShardyCompatTest.test_shardy_sharding_ops_with_different_meshes\x00_run_and_get_tests_result\x00run_tests\x00_run_in_app.<locals>.main_function\x00_run_main\x00run\x00_run_in_app\x00main\x00<module>\x00jit(func)/jit(main)/ppermute\x00ShardyCompatTest.test_shardy_sharding_ops_with_different_meshes.<locals>.func.<locals>.shard_map_func\x00x\x00mhlo.sharding\x00jit(func)/jit(main)/sharding_constraint\x00#sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>\x00\x00xla.sdy.manual_axes\x00#sdy<manual_axes{"a"}>\x00xla.sdy.manual_computation_body\x00xla.sdy.in_shardings\x00xla.sdy.out_shardings\x00jax.result_info\x00result\x00public\x00xla.sdy.sharding\x00{devices=[2,1]<=[2]}\x00Sharding\x00xla.sdy.GlobalToLocalShape\x00xla.sdy.LocalToGlobalShape\x00\x08a\x11\x05o\x01\x0b\xbb\xbf\xc1\xc9\xcb\x11\xad\xab\xd9\xa7\xdb\xa7\xa7\xa7\x11\xad\xab\xdf\xa7\xb7\xa7\xa7\xa7\x03\xb3\x11\xad\xab\xe5\xa7\xb7\xa7\xa7\xa7\x0b\xa7\xcd\xa7\xb3\xab\x05\xcf\xd1',
    xla_call_module_version=9,
    nr_devices=2,
)  # End paste
