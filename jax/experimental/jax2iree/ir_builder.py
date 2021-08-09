# Copyright 2021 Google LLC
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

from typing import Optional, Sequence, Union

import io
from iree.runtime.system_api import load_vm_module

import numpy as np

from jax import core

from .iree_imports import *

# TODO: This is hoaky and needs to go in a real runtime layer.
_cached_global_config: Optional[iree_runtime.system_api.Config] = None
def _get_global_config() -> iree_runtime.system_api.Config:
  global _cached_global_config
  if not _cached_global_config:
    _cached_global_config = iree_runtime.system_api.Config("dylib")
  return _cached_global_config


class Builder:
  """An input module under construction."""

  def __init__(self, context: Optional[ir.Context] = None):
    self.context = context if context else ir.Context()
    self.current_loc: Optional[ir.Location] = None
    self.input_module = ir.Module.create(loc=self.loc)
    self.module_op = self.input_module.operation.opview
    self.ip = ir.InsertionPoint(self.module_op.body)

    # Dialect registration.
    self.context.allow_unregistered_dialects = True
    mhlo.register_mhlo_dialect(self.context)
    chlo.register_chlo_dialect(self.context)

    # Compiler API.
    self.compiler_options = compiler_driver.CompilerOptions()
    self.compiler_options.set_input_dialect_mhlo()
    self.compiler_options.add_target_backend("cpu")

  def compile_module_to_binary(self) -> memoryview:
    """Compiles an input MLIR module to a binary blob."""
    # TODO: Wire in diagnostics/error handling (just dumps to stderr as is).
    with self.context:
      pm = passmanager.PassManager()
      compiler_driver.build_iree_vm_pass_pipeline(self.compiler_options, pm)
      pm.run(self.input_module)
      bytecode_io = io.BytesIO()
      compiler_driver.translate_module_to_vm_bytecode(self.compiler_options,
        self.input_module, bytecode_io)
      return bytecode_io.getbuffer()

  @staticmethod
  def load_compiled_binary(blob: memoryview):
    """Loads a compiled binary into the runtime.

    This should be part of a dedicated runtime layer but is here for demo
    purposes.
    """
    # TODO: This API in IREE needs substantial ergonomic work for loading
    # a module from a memory image.
    config = _get_global_config()
    vm_module = iree_runtime.binding.VmModule.from_flatbuffer(blob)
    return load_vm_module(vm_module, config)

  @property
  def loc(self) -> ir.Location:
    if self.current_loc is None:
      # TODO: Extract from traceback in some way.
      return ir.Location.unknown(self.context)
    return self.current_loc

  def create_function(self, name: str, input_types: Sequence[ir.Type],
                      return_types: Sequence[ir.Type]) -> "FunctionBuilder":
    with self.context:
      ftype = ir.FunctionType.get(input_types, return_types)
      func_op = builtin.FuncOp(name, ftype, loc=self.loc, ip=self.ip)
    return FunctionBuilder(self, func_op)

  def convert_aval_to_ir_type(self, aval: core.AbstractValue) -> ir.Type:
    with self.loc:
      if isinstance(aval, core.ShapedArray):
        element_type = self.convert_dtype_to_ir_type(aval.dtype)
        shape = aval.shape  # TODO: Handle symbols, etc
        return ir.RankedTensorType.get(shape, element_type)
      elif isinstance(aval, core.UnshapedArray):
        element_type = self.convert_dtype_to_ir_type(aval.dtype)
        return ir.UnrankedTensorType.get(element_type)

      raise NotImplementedError(f"Unsupported AbstractValue conversion: {aval}")

  def convert_dtype_to_ir_type(self, dtype) -> ir.Type:
    """Convert a dtype to an ir type."""
    # TODO: Actually switch on types, produce errors, etc.
    return ir.F32Type.get(self.context)

  def convert_ir_type_to_dtype(self, element_type: ir.Type):
    """Convert an IR type to a dtype."""
    # TODO: Actually switch on types, produce errors, etc.
    return np.float32

  def get_shaped_type_dims_list(self,
                                t: ir.ShapedType) -> Sequence[Union[None, int]]:
    # TODO: Ugh. Has anyone tried to use this before? Add a simple .shape
    # property.
    if not t.has_rank:
      raise ValueError(f"Cannot get dims from unranked type")

    def get_dim(index):
      return None if t.is_dynamic_dim(index) else t.get_dim_size(index)

    return [get_dim(i) for i in range(t.rank)]


class FunctionBuilder:
  """Manages state for constructing a global function."""

  def __init__(self, builder: Builder, func_op: ir.Operation):
    self.b = builder
    self.context = self.b.context
    self.func_op = func_op
    self.entry_block = self.func_op.add_entry_block()
    self.ip = ir.InsertionPoint(self.entry_block)

  def emit_return(self, values: Sequence[ir.Value]):
    """Emits a return op, also updating the containing function type."""
    std.ReturnOp(values, loc=self.b.loc, ip=self.ip)
    # Update function type to match.
    ftype = self.func_op.type
    return_types = [v.type for v in values]
    with self.context:
      ftype_attr = ir.TypeAttr.get(
          ir.FunctionType.get(ftype.inputs, return_types))
      # TODO: Provide a FuncOp.type setter upstream.
      self.func_op.attributes["type"] = ftype_attr
