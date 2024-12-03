# Copyright 2024 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test utils for the Mosaic GPU MLIR dialect."""

from typing import Callable

from absl.testing import parameterized
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import gpu
from jax.experimental.mosaic.gpu import dialect as mgpu  # pylint: disable=g-importing-member
from jax.experimental.mosaic.gpu import gpu_address_space_to_nvptx  # pylint: disable=g-importing-member,g-multiple-import

_cext = mgpu._cext if mgpu is not None else None


def _make_ir_context():
  context = ir.Context()
  context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  context.load_all_available_dialects()
  mgpu.register_dialect(context)
  return context


def walk_operations(op: ir.OpView, callback):
  for region in op.operation.regions:
    for block in region:
      for block_op in block:
        walk_operations(block_op, callback)
  callback(op)


def find_if(
    module: ir.Module, predicate: Callable[[ir.OpView], bool]
) -> list[ir.OpView]:
  result = []

  def callback(op: ir.OpView):
    if predicate(op):
      result.append(op)

  for op in module.body.operations:
    walk_operations(op, callback)
  return result


def is_mosaic_gpu_op(op: ir.OpView) -> bool:
  return op.name.startswith("mosaic_gpu.")


def workgroup_ptr_ty() -> ir.Type:
  workgroup_nvptx_address_space = gpu_address_space_to_nvptx(
      gpu.AddressSpace.Workgroup)
  return ir.Type.parse(f"!llvm.ptr<{workgroup_nvptx_address_space}>")


class DialectTest(parameterized.TestCase):

  def setUp(self):
    if mgpu is None:
      raise self.skipTest("Test requires Mosaic GPU dialect")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()

  def test_dialect_module_is_loaded(self):
    self.assertTrue(_cext.globals._check_dialect_module_loaded("mosaic_gpu"))
