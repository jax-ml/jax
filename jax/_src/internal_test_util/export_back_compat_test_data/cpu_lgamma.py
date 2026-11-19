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

data_2026_10_26 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_10_26["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['cpu_lgamma_ffi'],
    serialized_date=datetime.date(2026, 10, 26),
    inputs=(array([-2.5, -0.5,  0.5,  1.5,  5. ], dtype=float32),),
    expected_outputs=(array([-0.056243718,  1.2655121  ,  0.5723649  , -0.12078224 ,
        3.1780539  ], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_lgamma attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<5xf32> loc("x")) -> (tensor<5xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @cpu_lgamma_ffi(%arg0) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i]) {i=5}, custom>} : (tensor<5xf32>) -> tensor<5xf32> loc(#loc2)
    return %0 : tensor<5xf32> loc(#loc3)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("jit(lgamma)/lgamma")
#loc3 = loc("jit(lgamma)")
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.18.0\x00\x01\x1b\x07\x01\x05\t\t\x01\x03\x0f\x03\x07\x13\x17\x1b\x03wQ\x0f\x01%\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x03'\x0b\x0b\x0f\x0f\x0b\x0f\x13\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b/\x05\x07\x0f3\x0f\x01\x05\x0b\x0f\x03\x0b\x13\x17\x07\x13\x07\x02B\x02\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x13\x01\x05\x17\x03\x07\x17'\x19;\x1bM\x05\x19\x05\x1b\x05\x1d\x1d\x1f\x01\x05\x1f\x1d#\x01\x05!\x03\x01\r\x01\x03\x03I\x03\x03'#\x07\x03\x031\r\x0335\x1d#\x1d%\x1d'\x1d)\r\x03=?\x1d+\x1d-\x0b\x03\x1d/\x1d1\x05\x01\x1f\x0b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x03O\x15\x03\x15\x03K\x03K\x01\x01\x01\x01\x01\x11\x03\x01\x01\t\x01\x02\x02)\x03\x15\t\x11\x03\x05\x03\x05\t)\x03\x05\r\x13\x04Q\x05\x01Q\x01\x05\x01\x07\x04?\x03\x01\x05\x03P\x01\x03\x07\x04+\x03\x05\x0b\x03\x0b\x11\x00\x05G\x1d\x15\x05\x03\x05\x03\x01\x07\x04!\x03\x03\x06\x03\x01\x05\x01\x00\x12\x053\x1f\x03\x05\x1f\x0f\x0b\x0f!\x19'%3)\x05\x17%)9\x15\x1f\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00func_v1\x00custom_call_v2\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_lgamma\x00x\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jit(lgamma)/lgamma\x00jit(lgamma)\x00jax.result_info\x00result\x00main\x00public\x00num_batch_dims\x001\x00\x00cpu_lgamma_ffi\x00\x08)\x07\x05\x1f\x01\x0b+-/79\x13ACE%G)%)%",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_10_26["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['cpu_lgamma_ffi'],
    serialized_date=datetime.date(2026, 10, 26),
    inputs=(array([-2.5, -0.5,  0.5,  1.5,  5. ]),),
    expected_outputs=(array([-0.056243716497674054,  1.2655121234846454  ,
        0.5723649429247001  , -0.12078223763524522 ,
        3.1780538303479458  ]),),
    mlir_module_text=r"""
#loc1 = loc("x")
module @jit_lgamma attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<5xf64> loc("x")) -> (tensor<5xf64> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @cpu_lgamma_ffi(%arg0) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "1"}, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i]) {i=5}, custom>} : (tensor<5xf64>) -> tensor<5xf64> loc(#loc2)
    return %0 : tensor<5xf64> loc(#loc3)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("jit(lgamma)/lgamma")
#loc3 = loc("jit(lgamma)")
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.18.0\x00\x01\x1b\x07\x01\x05\t\t\x01\x03\x0f\x03\x07\x13\x17\x1b\x03wQ\x0f\x01%\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x03'\x0b\x0b\x0f\x0f\x0b\x0f\x13\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b/\x05\x07\x0f3\x0f\x01\x05\x0b\x0f\x03\x0b\x13\x17\x07\x13\x07\x02B\x02\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x13\x01\x05\x17\x03\x07\x17'\x19;\x1bM\x05\x19\x05\x1b\x05\x1d\x1d\x1f\x01\x05\x1f\x1d#\x01\x05!\x03\x01\r\x01\x03\x03I\x03\x03'#\x07\x03\x031\r\x0335\x1d#\x1d%\x1d'\x1d)\r\x03=?\x1d+\x1d-\x0b\x03\x1d/\x1d1\x05\x01\x1f\x0b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x13\x03O\x15\x03\x15\x03K\x03K\x01\x01\x01\x01\x01\x11\x03\x01\x01\t\x01\x02\x02)\x03\x15\t\x11\x03\x05\x03\x05\x0b)\x03\x05\r\x13\x04Q\x05\x01Q\x01\x05\x01\x07\x04?\x03\x01\x05\x03P\x01\x03\x07\x04+\x03\x05\x0b\x03\x0b\x11\x00\x05G\x1d\x15\x05\x03\x05\x03\x01\x07\x04!\x03\x03\x06\x03\x01\x05\x01\x00\x12\x053\x1f\x03\x05\x1f\x0f\x0b\x0f!\x19'%3)\x05\x17%)9\x15\x1f\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00func_v1\x00custom_call_v2\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_lgamma\x00x\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00jit(lgamma)/lgamma\x00jit(lgamma)\x00jax.result_info\x00result\x00main\x00public\x00num_batch_dims\x001\x00\x00cpu_lgamma_ffi\x00\x08)\x07\x05\x1f\x01\x0b+-/79\x13ACE%G)%)%",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
