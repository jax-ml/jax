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

import datetime
import numpy as np

array = np.array
float32 = np.float32
int32 = np.int32

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_08_19 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['AllocateBuffer'],
    serialized_date=datetime.date(2025, 8, 19),
    inputs=(),
    expected_outputs=(array([[0., 0.],
       [0., 0.]], dtype=float32),),
    mlir_module_text=r"""
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x2xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @AllocateBuffer() : () -> tensor<2x2xf32> loc(#loc3)
    return %0 : tensor<2x2xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":843:13)
#loc2 = loc("jit(func)"(#loc1))
#loc3 = loc("empty"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01\x19\x05\x01\x05\t\x01\x03\x0b\x03\x07\x0f\x13\x17\x03S5\x0b\x01\x1d\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x17\x0b\x03\x19\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x01\x05\x0b\x0f\x03\x07\x17\x13\x07\x02\xa5\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\r\x11\x01\x00\x05\x0f\x05\x11\x05\x13\x1d\x13\x15\x05\x15\x1d\x17\x19\x05\x17\x17\x1b.\r\x1b\x05\x19\x03\x01#\x07\x03\x03#\r\x03%'\x1d\x1b\x1d\x1d\x1d\x1f\x1d!\x0b\x03\x1d#\x1d%\x05\x01\x01\t\x01\x02\x02)\x05\t\t\t\x11\x01\x03\x05\t\x04C\x05\x01Q\x01\x05\x01\x07\x041\x03\x01\x05\x03P\x01\x03\x07\x04\x1d\x03\x03\t\x05B\x11\x05\x03\x05\x07\x04\x01\x03\x01\x06\x03\x01\x05\x01\x00.\x04'\x1f\x03\x0f\x0b\x0f!i\x15\r\x13%)9\x15\x1f\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00empty\x00jit(func)\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00result\x00main\x00public\x00\x00AllocateBuffer\x00\x08'\x07\x05\x1f\x01\x0b\x1d\x1f!)+\x11-/1\x1d3\x1d\x1d\x1d",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste
