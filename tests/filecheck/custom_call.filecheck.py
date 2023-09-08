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

# Tests for mlir.custom_call().

# RUN: %PYTHON %s | FileCheck %s

from absl import app

import jax
from jax.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
import numpy as np

ShapedArray = jax.core.ShapedArray

def print_custom_call(name, arg_avals, result_avals, **kw):
  print(f"TEST: {name}")
  ctx = mlir.make_ir_context()
  loc = ir.Location.unknown(context=ctx)
  with ctx, loc:
    module = ir.Module.create(loc=ir.Location.unknown())
    ip = ir.InsertionPoint(module.body)
    arg_types = [mlir.aval_to_ir_type(aval) for aval in arg_avals]
    result_types = [mlir.aval_to_ir_type(aval) for aval in result_avals]
    ftype = ir.FunctionType.get(arg_types, result_types)
    func = func_dialect.FuncOp("func", ftype, ip=ip)
    entry_block = func.add_entry_block()
    with ir.InsertionPoint(entry_block):
      outs = mlir.custom_call(
          name, result_types=result_types, operands=entry_block.arguments, **kw
      )
      func_dialect.ReturnOp(outs)
  module.operation.verify()
  print(str(module))

def main(_):
  aval1 = ShapedArray((2, 3), np.dtype(np.float32))
  aval2 = ShapedArray((3, 4), np.dtype(np.int64))
  # CHECK-LABEL: TEST: simple
  # CHECK: stablehlo.custom_call @simple(%arg0) {api_version = 2 : i32} : (tensor<2x3xf32>) -> tensor<3x4xi64>
  print_custom_call("simple", [aval1], [aval2])

  # CHECK-LABEL: TEST: sideeffect
  # CHECK: stablehlo.custom_call @sideeffect(%arg0) {has_side_effect = true} : (tensor<2x3xf32>) -> tensor<3x4xi64>
  print_custom_call("sideeffect", [aval1], [aval2], api_version=1,
                    has_side_effect=True)

  # CHECK-LABEL: TEST: backendconfig
  # CHECK: stablehlo.custom_call @backendconfig(%arg0) {backend_config = "hello"} : (tensor<2x3xf32>) -> tensor<3x4xi64>
  print_custom_call("backendconfig", [aval1], [aval2], api_version=1,
                    backend_config=b"hello")

  # CHECK-LABEL: TEST: calledcomputations
  # CHECK: stablehlo.custom_call @calledcomputations(%arg0) {called_computations = [@a, @b]} : (tensor<2x3xf32>) -> tensor<3x4xi64>
  print_custom_call("calledcomputations", [aval1], [aval2], api_version=1,
                    called_computations=["a", "b"])

  # CHECK-LABEL: TEST: aliases
  # CHECK: stablehlo.custom_call @aliases(%arg0, %arg1) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = []>]} : (tensor<2x3xf32>, tensor<3x4xi64>) -> (tensor<3x4xi64>, tensor<2x3xf32>)
  print_custom_call("aliases", [aval1, aval2], [aval2, aval1], api_version=1,
                    operand_output_aliases={1: 0})

  # CHECK-LABEL: TEST: layouts
  # CHECK: stablehlo.custom_call @layouts(%arg0, %arg1) {operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>]} : (tensor<2x3xf32>, tensor<3x4xi64>) -> (tensor<3x4xi64>, tensor<2x3xf32>)
  print_custom_call("layouts", [aval1, aval2], [aval2, aval1], api_version=1,
                    operand_layouts=[[0, 1], [1, 0]],
                    result_layouts=[[1, 0], [0, 1]])

if __name__ == "__main__":
  app.run(main)
