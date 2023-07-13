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

# Verifies the strip-locations pass strips all locations.

# RUN: %PYTHON %s | FileCheck %s

from absl import app
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager
from jax._src.lib.mlir.dialects import hlo


def main(_):
  # We should have only one location left (unknown).
  # CHECK: jit__lambda_
  # CHECK-NOT: #loc1
  with ir.Context() as c:
    hlo.register_dialect(c)
    m = ir.Module.parse("""
  module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
    func.func public @main(%arg0: tensor<i32> loc(#loc2)) -> (tensor<i32>) {
      %0 = stablehlo.constant dense<2> : tensor<i32> loc(#loc1)
      %1 = stablehlo.multiply %arg0, %0 : tensor<i32> loc(#loc2)
      return %1 : tensor<i32> loc(#loc1)
    } loc(#loc1)
  } loc(#loc2)
  #loc1 = loc("<ipython-input-9-5f2529662a56>":1:18)
  #loc2 = loc("jit(<lambda>)/jit(main)/mul"(#loc1))
    """)
    passes = passmanager.PassManager.parse(
        "builtin.module(jax-strip-locations)"
    )
    passes.run(m.operation)
    m.operation.print(enable_debug_info=True)


if __name__ == "__main__":
  app.run(main)
