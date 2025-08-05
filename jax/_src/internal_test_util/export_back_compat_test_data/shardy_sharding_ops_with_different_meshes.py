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
int32 = np.int32

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_06_30 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=[],
    serialized_date=datetime.date(2025, 6, 30),
    inputs=(array([[0., 1., 2., 3.],
       [4., 5., 6., 7.]], dtype=float32),),
    expected_outputs=(array([[4., 5., 6., 7.],
       [0., 1., 2., 3.]], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":1001:8 to :54)
#loc4 = loc("third_party/py/absl/testing/absltest.py":2900:19 to :56)
#loc5 = loc("third_party/py/absl/testing/absltest.py":2936:35 to 2938:3)
#loc6 = loc("third_party/py/absl/testing/absltest.py":2446:6 to :34)
#loc7 = loc("third_party/py/absl/app.py":404:13 to :23)
#loc8 = loc("third_party/py/absl/app.py":484:6 to :27)
#loc9 = loc("third_party/py/absl/testing/absltest.py":2448:4 to :31)
#loc10 = loc("third_party/py/absl/testing/absltest.py":2330:2 to :38)
#loc11 = loc("third_party/py/jax/tests/export_back_compat_test.py":1005:2 to :47)
#loc12 = loc("third_party/py/jax/tests/export_back_compat_test.py":992:13 to :30)
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
#loc46 = loc("jit(func)/shard_map"(#loc43))
#loc49 = loc("shard_map:"(#loc46))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["a"=2]> loc(#loc)
  func.func public @main(%arg0: tensor<2x4xf32> loc("x")) -> (tensor<2x4xf32> {jax.result_info = "result"}) {
    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {}]> : tensor<2x4xf32> loc(#loc48)
    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<1x4xf32> loc("shard_map:"(#loc46))) {
      %2 = "stablehlo.collective_permute"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<1x4xf32>) -> tensor<1x4xf32> loc(#loc50)
      sdy.return %2 : tensor<1x4xf32> loc(#loc49)
    } : (tensor<2x4xf32>) -> tensor<2x4xf32> loc(#loc49)
    return %1 : tensor<2x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":991:10 to :73)
#loc13 = loc("third_party/py/jax/tests/export_back_compat_test.py":990:15 to :46)
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
#loc45 = loc("jit(func)/sharding_constraint"(#loc42))
#loc47 = loc("jit(func)/ppermute"(#loc44))
#loc48 = loc("sharding_constraint:"(#loc45))
#loc50 = loc("ppermute:"(#loc47))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.10.9\x00\x01)\x07\x01\x05\t\x13\x01\x05\x0f\x13\x03\t\x17\x1b\x1f#\x05\x07'+/\x03\xfb\xcd\x17\x01\xa7\x07\x0b\x0b\x0f\x0f\x0b\x0f\x0b\x0f\x0f\x0f\x0f\x0f\x0f\x0b\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0f\x1f\x0b\x1f\x0f\x0b\x1f\x0f\x0b'\x0f\x0b\x1f\x0f\x0b\x1f\x0f\x0b\x1f\x0f\x0b\x1f\x0f\x0b\x1f\x0f\x0b\x1f\x0b\x0b\x0f\x0b\x0f\x1f\x0b\x0f\x0b\x0f\x0f\x0b\x1f\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x03\x11\x1b\x0f\x13\x0f\x17\x0f\x13\x0f\x05\x17\x0f\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0f\x8f\x01\x0b\x17\x0f\x07\x17\x0b\x05\r\x17\x07\x17\x07\x17\x17\x02N\x06\x1f\x05\x19\x05\x1b\x1d\x7f\x81\x1d\x89\x8b\x05\x0b\x1d7\x01\x05\x1d\x15\x13M\x1dIK\x1dOQ\x1dUW\x1d[]\x1dac\x05\x1f\x1dgi\x1dmo\x1dsu\x1d\x0f\x87\x03\x07)+-/13\x05!\x11\t\x00\x05#\x11\x03\t\x05%\x11\x03\x05\x05'\x05)\t\x0b\x1d=?\x05+\x1dAC\x05-\x15E\x11\x1d\x0fG-\x03\x07~\x0f\x15\x93\x05/-\x03\x07\xa6\x0f\x11m\x15\x15S\x051-\x05\x07R-'q\x15\x17Y\x053-\x05\t\xe2-G\xea-\x07\x15\x19_\x055-\x05\x07:&\rE\x15\x1be\x057-\x1d\x07R\x06\x1b/\x15\x1fk\x059-\x1d\x07\x92\x07\r7\x15!q\x05;-\x05\x07B&\t?\x15#w\x05=-\x05\x07j$\x05M\x1dy{\x05?-\x03\x07\xb6\x0f\x05_\x05A\x05C\x1d\x83\x85\x05E\x15%\x11-\x03\x07\x82\x0f\x1b=\x05G\x1d\x8d\x8f\x05I\x15\x91\x97\x1d\x93\x95\x05K-\x03\x07z\x0f\x1f]\x15%\x99\x15\x13\x9b\x15\x15\x9d\x15\x17\x9f\x15\x19\xa1\x15\x1b\xa3\x15\x1f\xa5\x15!#\r9\x05\xaf\xb3\x01\x0f\x03\xa7\x05\x03\xad\x01\x03A\t\x0b\x03\xb1\x01\x01\tA\x01\x0b\x01\x01\x01\x01\x03}\x03\x03\xb9\r\x01#\x13\x03\x03\xbf\r\x03\xc1\xc3\x1dM\x1dO\x1d=\x1dQ\x13\x11\x05\x1f\x15A\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1b\x05\t\x11\x05\x01\x02\x02\x0b\x1b\x05\x05\x11\x05\x01\t)\x05\t\x11\r\t)\x05\x05\x11\r\x1d\x11\x03\x0b\x03\x0b)\x05\t\t\x11\x04\xcd\x05\x03Q\x01'\x01\x07\x04\xbb\x03\x01\t\x05@\x01\x03\rP\x01\x05\x07\x04\x9f\x03\x0b\x17\x03\x17\r\x00\x01\x06\r\x03\x01\x03\x01\x07F;\x07\x03\x01\x03\x03\tV\x07\t\x03\x01\x03\x05\x07\x04E\x03\t\x13\x03\x0f\x07\x00\x01\x06\t\x03\x0f\x03\x01\x11F\t\x0b\x03\x0f\x03\x03\x01\x06\t\x03\x07\x03\x05\x0b\x04\x07\x03\x07\x01\x06\x01\x03\x0b\x03\x07\x0f\x04\x01\x03\t\x06\x03\x01\x05\x01\x00\xba\rS\x0f\x0f!\xcd'\x15)\x17\x05\x13\x0b\x19\t\x15G\x155\x81=+\x05\x13%)97\x9dQi-\x15\x11\x0f')\x0b\x0f7\x0b\t\x11builtin\x00sdy\x00vhlo\x00unrealized_conversion_cast\x00module\x00mesh\x00sharding_constraint\x00manual_computation\x00return\x00func_v1\x00return_v1\x00collective_permute_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00third_party/py/absl/testing/absltest.py\x00ShardyCompatTest.test_shardy_sharding_ops_with_different_meshes.<locals>.func\x00third_party/py/absl/app.py\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00x\x00sharding_constraint:\x00jit(func)/sharding_constraint\x00ShardyCompatTest.test_shardy_sharding_ops_with_different_meshes\x00_run_and_get_tests_result\x00run_tests\x00_run_in_app.<locals>.main_function\x00_run_main\x00run\x00_run_in_app\x00main\x00<module>\x00a\x00shard_map:\x00jit(func)/shard_map\x00ppermute:\x00jit(func)/ppermute\x00ShardyCompatTest.test_shardy_sharding_ops_with_different_meshes.<locals>.func.<locals>.shard_map_func\x00jax.result_info\x00result\x00public\x00\x08-\r\x05k\x01\x05\xab\x0b\x0b\xb7\xbb\xbd\xc5\xc7\x03\xa7\x07\xa9\xb5\xa9\x05\xc9\xcb",
    xla_call_module_version=9,
    nr_devices=2,
)  # End paste
