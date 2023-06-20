# Copyright 2021 The JAX Authors.
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

# Tests for MLIR helpers.

# RUN: %PYTHON %s | FileCheck %s

from absl import app

import jax
from jax import numpy as jnp
from jax.interpreters import mlir
from jax._src.lib.mlir import ir
import numpy as np

from jax.tests.filecheck.jax_filecheck_helpers import print_ir

jax.config.update("jax_enable_x64", True)


def main(_):

  # The lowering of cumsum is annotated with @cache_lowering, which means we
  # should lower it as an out-of-line function once for any given shape.

  # CHECK-LABEL: TEST: cumsum_only_once int32[2,7] int32[2,7]
  # CHECK: func private @cumsum
  # CHECK-NOT: func private @cumsum
  @print_ir(np.empty([2, 7], np.int32), np.empty([2, 7], np.int32))
  def cumsum_only_once(x, y):
    return jnp.cumsum(x) + jnp.cumsum(y)

  # Test merging modules
  # CHECK-LABEL: TEST: merge_modules
  # CHECK: module @jit_g
  # CHECK: func public @main(
  # CHECK: func private @f(
  # CHECK: func private @m2_main_renamed(
  # CHECK: func private @f_0(
  def make_module(c):
    @jax.jit
    def f(x):
      return x + c

    @jax.jit
    def g(x):
      return f(x * 2)

    return g.lower(7).compiler_ir()

  m1 = make_module(10)
  m2 = str(make_module(20))

  with m1.context:
    # Reparse m2 in m1's context.
    m2_copy = ir.Module.parse(m2)
    mlir.merge_mlir_modules(m1, "m2_main_renamed", m2_copy)
  print("\nTEST: merge_modules")
  print(str(m1))


  # Test symbol renaming when merging modules
  # CHECK-LABEL: TEST: merge_modules_2
  # CHECK: module @jit_f
  # CHECK: func public @main(
  # CHECK: call @f(
  # CHECK: func private @f(
  # CHECK: func private @f_0(
  # CHECK: call @f_1(
  # CHECK: func private @f_1(

  with mlir.make_ir_context():
    m_str = """
module @jit_f {
  func.func public @main(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = call @f(%arg0) : (tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
  }
  func.func private @f(%arg0: tensor<i64>) -> tensor<i64> {
    return %arg0 : tensor<i64>
  }
}"""
    m1 = ir.Module.parse(m_str)
    m2 = ir.Module.parse(m_str)
    mlir.merge_mlir_modules(m1, "f", m2)
  print("\nTEST: merge_modules_2")
  print(str(m1))


if __name__ == "__main__":
  app.run(main)
