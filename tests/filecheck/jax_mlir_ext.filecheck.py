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

# RUN: %PYTHON %s | FileCheck %s -dump-input=always

from absl import app
import jax
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo
from jaxlib.mlir._mlir_libs import _jax_mlir_ext
import numpy as np


jax.config.parse_flags_with_absl()


def test_inlined_func_call():
  # CHECK: #loc = loc(unknown)
  # CHECK: module {
  # CHECK-NEXT:   func.func public @caller(%arg0: tensor<2x3xf32> loc(unknown), %arg1: tensor<2x3xf32> loc(unknown)) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  # CHECK-NEXT:     %0 = stablehlo.add %arg1, %arg0 : tensor<2x3xf32> loc(#loc8)
  # CHECK-NEXT:     %1 = stablehlo.multiply %0, %arg1 : tensor<2x3xf32> loc(#loc9)
  # CHECK-NEXT:     return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32> loc(#loc)
  # CHECK-NEXT:   } loc(#loc)
  # CHECK-NEXT: } loc(#loc)
  # CHECK-NEXT: #loc1 = loc("callee_file":1:2)
  # CHECK-NEXT: #loc2 = loc("caller_file":3:4)
  # CHECK-NEXT: #loc3 = loc("callee_stack"(#loc1))
  # CHECK-NEXT: #loc4 = loc("caller_stack"(#loc2))
  # CHECK-NEXT: #loc5 = loc(callsite(#loc3 at #loc4))
  # CHECK-NEXT: #loc6 = loc("caller_name/callee_name1"(#loc5))
  # CHECK-NEXT: #loc7 = loc("caller_name/callee_name2"(#loc5))
  # CHECK-NEXT: #loc8 = loc("caller_type:"(#loc6))
  # CHECK-NEXT: #loc9 = loc("caller_type:"(#loc7))
  ctx = mlir.make_ir_context()
  loc = ir.Location.unknown(context=ctx)
  aval = jax.core.ShapedArray((2, 3), np.dtype(np.float32))
  arg_avals = [aval, aval]
  result_avals = [aval, aval]
  with ctx, loc:
    callee_stack_loc = ir.Location.name(
        "callee_stack", ir.Location.file("callee_file", 1, 2)
    )
    callee_loc1 = ir.Location.name(
        "callee_name1", ir.Location.name("callee_type1:", callee_stack_loc)
    )
    callee_loc2 = ir.Location.name(
        "callee_name2", ir.Location.name("callee_type2:", callee_stack_loc)
    )
    caller_stack_loc = ir.Location.name(
        "caller_stack", ir.Location.file("caller_file", 3, 4)
    )
    caller_loc = ir.Location.name(
        "caller_name", ir.Location.name("caller_type:", caller_stack_loc)
    )

    module = ir.Module.create(loc=ir.Location.unknown())
    ip = ir.InsertionPoint(module.body)
    arg_types = [mlir.aval_to_ir_type(aval) for aval in arg_avals]
    result_types = [mlir.aval_to_ir_type(aval) for aval in result_avals]
    ftype = ir.FunctionType.get(arg_types, result_types)
    callee = func_dialect.FuncOp("callee", ftype, ip=ip)
    callee.attributes["sym_visibility"] = ir.StringAttr.get("private")
    entry_block = callee.add_entry_block()
    with ir.InsertionPoint(entry_block):
      with callee_loc1:
        x = hlo.add(entry_block.arguments[0], entry_block.arguments[1])
      with callee_loc2:
        y = hlo.multiply(x, entry_block.arguments[0])
      func_dialect.ReturnOp([x, y])

    caller = func_dialect.FuncOp("caller", ftype, ip=ip)
    caller.attributes["sym_visibility"] = ir.StringAttr.get("public")
    entry_block = caller.add_entry_block()
    with ir.InsertionPoint(entry_block):
      x, y = entry_block.arguments
      with caller_loc:
        x, y = _jax_mlir_ext.inlined_func_call(callee, [y, x], entry_block)
      func_dialect.ReturnOp([x, y])
    module.operation.verify()
    pipeline = passmanager.PassManager.parse("builtin.module(symbol-dce)")
    pipeline.run(module.operation)
    print(module.operation.print(enable_debug_info=True))


def test_traceback_to_location():
  def f():
    return g()

  def g():
    return h()

  def h():
    return jax._src.lib._jax.Traceback.get_traceback()

  tb = f()

  def _code_to_filename(code):
    return (
        "jax_mlir_ext_test.filecheck.py"
        if "jax_mlir_ext.filecheck" in code.co_filename
        else None
    )

  # CHECK: --- test_traceback_to_location
  print("--- test_traceback_to_location")
  ctx = mlir.make_ir_context()
  with ctx:
    # CHECK: loc(callsite("test_traceback_to_location.<locals>.h"("jax_mlir_ext_test.filecheck.py":{{[0-9]+}}:{{[0-9]+}} to :{{[0-9]+}}) at callsite("test_traceback_to_location.<locals>.g"("jax_mlir_ext_test.filecheck.py":{{[0-9]+}}:{{[0-9]+}} to :{{[0-9]+}}) at callsite("test_traceback_to_location.<locals>.f"("jax_mlir_ext_test.filecheck.py":{{[0-9]+}}:{{[0-9]+}} to :{{[0-9]+}}) at callsite("test_traceback_to_location"("jax_mlir_ext_test.filecheck.py":{{[0-9]+}}:{{[0-9]+}} to :{{[0-9]+}}) at callsite("main"("jax_mlir_ext_test.filecheck.py":{{[0-9]+}}:{{[0-9]+}} to :{{[0-9]+}}) at "<module>"("jax_mlir_ext_test.filecheck.py":{{[0-9]+}}:{{[0-9]+}} to :{{[0-9]+}})))))))
    cache = _jax_mlir_ext.TracebackToLocationCache(
        code_to_filename=_code_to_filename, frame_limit=1000)
    loc = cache.get(tb)
    print(loc)

    # CHECK: loc(callsite("test_traceback_to_location.<locals>.h"("jax_mlir_ext_test.filecheck.py":{{[0-9]+}}:{{[0-9]+}} to :{{[0-9]+}}) at "test_traceback_to_location.<locals>.g"("jax_mlir_ext_test.filecheck.py":{{[0-9]+}}:{{[0-9]+}} to :{{[0-9]+}})))
    cache = _jax_mlir_ext.TracebackToLocationCache(
        code_to_filename=_code_to_filename, frame_limit=2)
    loc = cache.get(tb)
    print(loc)


def main(_):
  test_inlined_func_call()
  test_traceback_to_location()


if __name__ == "__main__":
  app.run(main)
