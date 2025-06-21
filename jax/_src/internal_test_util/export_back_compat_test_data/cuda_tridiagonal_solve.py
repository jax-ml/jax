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

data_2025_06_16 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_06_16["f32"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusparse_gtsv2_ffi'],
    serialized_date=datetime.date(2025, 6, 16),
    inputs=(array([0., 2., 3.], dtype=float32), array([1., 1., 1.], dtype=float32), array([1., 2., 0.], dtype=float32), array([[1.],
       [1.],
       [1.]], dtype=float32)),
    expected_outputs=(array([[ 0.57142854],
       [ 0.42857146],
       [-0.2857143 ]], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("dl")
#loc2 = loc("d")
#loc3 = loc("du")
#loc4 = loc("b")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3xf32> loc("dl"), %arg1: tensor<3xf32> loc("d"), %arg2: tensor<3xf32> loc("du"), %arg3: tensor<3x1xf32> loc("b")) -> (tensor<3x1xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @cusparse_gtsv2_ffi(%arg0, %arg1, %arg2, %arg3) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32> loc(#loc6)
    return %0 : tensor<3x1xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":760:13)
#loc6 = loc("jit(func)/jit(main)/tridiagonal_solve"(#loc5))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.10.4\x00\x01\x19\x05\x01\x05\t\x01\x03\x0b\x03\x07\x0f\x13\x17\x03\x83]\x13\x01/\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x1b\x0b\x0b\x0f\x0b\x17\x0b\x03/\x0b/O\x1b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x0f\x13\x0f\x01\x05\x0b\x0f\x03\x0f\x13\x17\x07\x07#\x13\x13\x02\xea\x02\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\r\x11\x01\x00\x05\x0f\x05\x11\x05\x13\x1d\x13\x01\x05\x15\x1d\x17\x01\x05\x17\x1d\x1b\x01\x05\x19\x1d\x1f\x01\x05\x1b\x03\x05#/%E\x05\x1d\x05\x1f\x1d)+\x05!\x17-\xe2\x0b\x1b\x05#\r\x01\x1f\x0f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x11!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\t////#\r\x03\x03;\r\x03=?\x1d%\x1d'\x1d)\x1d+\r\x03GI\x1d-\x1d/\x0b\x03\x1d1\x1d3\x03\x01\x05\x01\x03\t1113\x03\x03Y\x15\x01\r\x01\x03\x033\x01\t\x01\x02\x02)\x03\r\t)\x05\r\x05\t\t\x13\x11\t\x05\x05\x05\x07\x03\x07)\x03\x05\x0b)\x03\t\x0b\x04c\x05\x01Q\x01\x05\x01\x07\x04Q\x03\x01\x05\x03P\x01\x03\x07\x04=\x03\x0b\x0b\t\x0b\x11\x0b\x15\x0b\x19\x0f\x1d\x00\x05G'!\x05\x03\x07\t\x01\x03\x05\x07\x07\x04\x01\x03\t\x06\x03\x01\x05\x01\x00\xd2\x055'\x03\x05\x1f\x0f\x0b\x0f!iM3)\x05\x07\x05\x07\x13%)9\x15\x1f\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00dl\x00d\x00du\x00b\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00jit(func)/jit(main)/tridiagonal_solve\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00result\x00main\x00public\x00num_batch_dims\x000\x00\x00cusparse_gtsv2_ffi\x00\x08'\x07\x05\x1f\x01\x0b579AC\x11KMOQSUW[",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2025_06_16["f64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusparse_gtsv2_ffi'],
    serialized_date=datetime.date(2025, 6, 16),
    inputs=(array([0., 2., 3.]), array([1., 1., 1.]), array([1., 2., 0.]), array([[1.],
       [1.],
       [1.]])),
    expected_outputs=(array([[ 0.5714285714285714 ],
       [ 0.42857142857142855],
       [-0.2857142857142857 ]]),),
    mlir_module_text=r"""
#loc1 = loc("dl")
#loc2 = loc("d")
#loc3 = loc("du")
#loc4 = loc("b")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<3xf64> loc("dl"), %arg1: tensor<3xf64> loc("d"), %arg2: tensor<3xf64> loc("du"), %arg3: tensor<3x1xf64> loc("b")) -> (tensor<3x1xf64> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @cusparse_gtsv2_ffi(%arg0, %arg1, %arg2, %arg3) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64> loc(#loc6)
    return %0 : tensor<3x1xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":760:13)
#loc6 = loc("jit(func)/jit(main)/tridiagonal_solve"(#loc5))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.10.4\x00\x01\x19\x05\x01\x05\t\x01\x03\x0b\x03\x07\x0f\x13\x17\x03\x83]\x13\x01/\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x1b\x0b\x0b\x0f\x0b\x17\x0b\x03/\x0b/O\x1b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x0f\x13\x0f\x01\x05\x0b\x0f\x03\x0f\x13\x17\x07\x07#\x13\x13\x02\xea\x02\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\r\x11\x01\x00\x05\x0f\x05\x11\x05\x13\x1d\x13\x01\x05\x15\x1d\x17\x01\x05\x17\x1d\x1b\x01\x05\x19\x1d\x1f\x01\x05\x1b\x03\x05#/%E\x05\x1d\x05\x1f\x1d)+\x05!\x17-\xe2\x0b\x1b\x05#\r\x01\x1f\x0f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x11!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\t////#\r\x03\x03;\r\x03=?\x1d%\x1d'\x1d)\x1d+\r\x03GI\x1d-\x1d/\x0b\x03\x1d1\x1d3\x03\x01\x05\x01\x03\t1113\x03\x03Y\x15\x01\r\x01\x03\x033\x01\t\x01\x02\x02)\x03\r\t)\x05\r\x05\t\x0b\x13\x11\t\x05\x05\x05\x07\x03\x07)\x03\x05\x0b)\x03\t\x0b\x04c\x05\x01Q\x01\x05\x01\x07\x04Q\x03\x01\x05\x03P\x01\x03\x07\x04=\x03\x0b\x0b\t\x0b\x11\x0b\x15\x0b\x19\x0f\x1d\x00\x05G'!\x05\x03\x07\t\x01\x03\x05\x07\x07\x04\x01\x03\t\x06\x03\x01\x05\x01\x00\xd2\x055'\x03\x05\x1f\x0f\x0b\x0f!iM3)\x05\x07\x05\x07\x13%)9\x15\x1f\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00dl\x00d\x00du\x00b\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00jit(func)/jit(main)/tridiagonal_solve\x00third_party/py/jax/tests/export_back_compat_test.py\x00jax.result_info\x00result\x00main\x00public\x00num_batch_dims\x000\x00\x00cusparse_gtsv2_ffi\x00\x08'\x07\x05\x1f\x01\x0b579AC\x11KMOQSUW[",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste
