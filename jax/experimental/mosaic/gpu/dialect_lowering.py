# Copyright 2024 The JAX Authors.
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

"""Lowering rules and pass for the MLIR Mosaic GPU dialect."""

from collections.abc import Callable
import functools
import operator
from typing import Sequence, Type, cast

from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib import mosaic_gpu_dialect as mgpu
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import builtin
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import nvvm
from jax._src.lib.mlir.dialects import vector
import numpy as np

from .fragmented_array import FragmentedArray, WGStridedFragLayout
from .layout_inference import has_any_layout_set
from .layout_inference import should_have_layout
from .layout_inference import strided_fragmented_layout
from .utils import c, ptr_as_memref, single_thread_predicate

# mypy: ignore-errors


MlirLoweringRule = Callable[[ir.Operation | ir.OpView], Sequence[ir.Value]]


_lowerings: dict[str, MlirLoweringRule] = {}


def _fragmented_array_to_ir(
    fragmented_array: FragmentedArray, ty: ir.Type
) -> ir.Value:
  if not isinstance(fragmented_array.layout, WGStridedFragLayout):
    raise NotImplementedError(fragmented_array.layout)

  conversion_cast = builtin.UnrealizedConversionCastOp(
      [ty], fragmented_array.registers.tolist()
  )

  conversion_cast.attributes["layout"] = strided_fragmented_layout()

  # TODO(bchetioui): make sub-attributes part of the layout attribute.
  i32 = ir.IntegerType.get_signless(32)
  conversion_cast.attributes["layout_shape"] = ir.ArrayAttr.get(
      [ir.IntegerAttr.get(i32, s) for s in fragmented_array.shape]
  )
  conversion_cast.attributes["vec_size"] = ir.IntegerAttr.get(
      i32, fragmented_array.layout.vec_size
  )

  if fragmented_array.is_signed is not None:
    conversion_cast.attributes["is_signed"] = ir.BoolAttr.get(
        fragmented_array.is_signed
    )
  return conversion_cast.result


# TODO(bchetioui): add code that verifies the layout is as inferred.
def _fragmented_array_from_ir(
    fragmented_array_as_ir: ir.Value,
) -> FragmentedArray:

  conversion_cast = cast(
      builtin.UnrealizedConversionCastOp, fragmented_array_as_ir.owner.opview  # pytype: disable=attribute-error
  )

  if not isinstance(conversion_cast, builtin.UnrealizedConversionCastOp):
    raise ValueError(f"{conversion_cast} is not a conversion_cast")

  conversion_cast_layout = conversion_cast.attributes["layout"]

  if conversion_cast_layout != strided_fragmented_layout():
    raise NotImplementedError(
        f"Converting conversion_casts with layout {conversion_cast_layout} "
        "back to FragmentedArrays is not supported."
    )

  converted_outputs = builtin.unrealized_conversion_cast(
      [operand.type for operand in conversion_cast.operands],
      conversion_cast.results,
  )

  reverse_conversion_cast = converted_outputs[0].owner.opview
  for attribute in conversion_cast.attributes:
    attribute = cast(ir.NamedAttribute, attribute)
    reverse_conversion_cast.attributes[attribute.name] = attribute.attr

  registers = np.array(list(converted_outputs))

  shape = tuple(int(s) for s in conversion_cast.attributes["layout_shape"])
  vec_size = int(conversion_cast.attributes["vec_size"])
  if ir.IntegerType.isinstance(conversion_cast.outputs[0].type):
    is_signed = bool(conversion_cast.attributes["is_signed"])
  else:
    is_signed = None

  layout = WGStridedFragLayout(shape, vec_size)

  return FragmentedArray(
      _registers=registers, _layout=layout, _is_signed=is_signed
  )


# TODO(bchetioui): Remove this when minimum jaxlib version >= 0.4.36.
# Jaxlib doesn't contain Mosaic GPU dialect bindings.
InitializeBarrierOp = mgpu.InitializeBarrierOp if mgpu is not None else None

def _register_lowering(
    op: str | Type[ir.OpView]
) -> Callable[[MlirLoweringRule], MlirLoweringRule]:
  def wrapper(f):
    op_name = op if isinstance(op, str) else op.OPERATION_NAME  # pytype: disable=attribute-error
    _lowerings[op_name] = f
    return f

  return wrapper


def _lowered_barrier_type() -> ir.Type:
  return ir.IntegerType.get_signless(64)


def gpu_address_space_to_nvptx(address_space: gpu.AddressSpace) -> int:
  match address_space:
    case gpu.AddressSpace.Global:
      return 1
    case gpu.AddressSpace.Workgroup:
      return 3
    case _:
      raise NotImplementedError(f"address_space not supported: {address_space}")


@_register_lowering(InitializeBarrierOp)
def _initialize_barrier_op_lowering_rule(
    initialize_barrier_op: InitializeBarrierOp,
) -> Sequence[ir.Value]:

  shape = initialize_barrier_op.barriers_ref.type.shape
  num_barriers = functools.reduce(operator.mul, shape, 1)

  i32 = ir.IntegerType.get_signless(32)
  workgroup_nvptx_address_space = gpu_address_space_to_nvptx(
      gpu.AddressSpace.Workgroup)
  ptr_ty = ir.Type.parse(f"!llvm.ptr<{workgroup_nvptx_address_space}>")

  lowered_barrier_type = _lowered_barrier_type()

  predicate = single_thread_predicate(per_block=True)
  for i in range(num_barriers):
    nvvm.mbarrier_init_shared(
        llvm.getelementptr(ptr_ty, initialize_barrier_op.base_pointer, [], [i],
                           lowered_barrier_type),
        c(initialize_barrier_op.arrival_count.value, i32),
        predicate=predicate
    )

  barrier_base_ptr = llvm.getelementptr(
      ir.Type.parse("!llvm.ptr"),
      initialize_barrier_op.base_pointer, [], [0], lowered_barrier_type)

  return ptr_as_memref(
      barrier_base_ptr, initialize_barrier_op.barriers_ref.type),


@_register_lowering(vector.LoadOp)
def _vector_load_op_lowering_rule(
    vector_load_op: vector.LoadOp,
) -> Sequence[ir.Value]:
  (out_layout,) = cast(ir.ArrayAttr, vector_load_op.attributes["out_layouts"])

  if out_layout != strided_fragmented_layout():
    raise ValueError(
        f"{vector_load_op} has an unsupported layout: {out_layout}"
    )

  for i in vector_load_op.indices:
    index_defining_op = i.owner.opview
    if (
        not isinstance(index_defining_op, arith.ConstantOp)
        or index_defining_op.literal_value != 0
    ):
      # TODO(bchetioui,dasenov): support non-zero indices.
      raise NotImplementedError(
          "Only constants with value 0 are supported as indices "
          f"for {vector_load_op}"
      )

  fragmented_array = FragmentedArray.load_strided(vector_load_op.base)
  return [_fragmented_array_to_ir(fragmented_array, vector_load_op.result.type)]


@_register_lowering(vector.StoreOp)
def _vector_store_op_lowering_rule(
    vector_store_op: vector.StoreOp,
) -> Sequence[ir.Value]:

  in_layout, *_ = cast(ir.ArrayAttr, vector_store_op.attributes["in_layouts"])

  if in_layout != strided_fragmented_layout():
    raise ValueError(
        f"{vector_store_op} has an unsupported layout: {in_layout}"
    )

  for i in vector_store_op.indices:
    index_defining_op = i.owner.opview
    if (
        not isinstance(index_defining_op, arith.ConstantOp)
        or index_defining_op.literal_value != 0
    ):
      # TODO(bchetioui,dasenov): support non-zero indices.
      raise NotImplementedError(
          "Only constants with value 0 are supported as indices "
          f"for {vector_store_op}"
      )

  fragmented_array = _fragmented_array_from_ir(vector_store_op.valueToStore)
  fragmented_array.store_untiled(vector_store_op.base)

  return []


@_register_lowering(arith.AddFOp)
def _arith_addf_op_lowering_rule(add: arith.AddFOp) -> Sequence[ir.Value]:

  fragmented_array_lhs = _fragmented_array_from_ir(add.lhs)
  fragmented_array_rhs = _fragmented_array_from_ir(add.rhs)

  return [
      _fragmented_array_to_ir(
          fragmented_array_lhs + fragmented_array_rhs, add.result.type
      )
  ]


def lower_mgpu_dialect(module: ir.Module):
  module.context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  module.context.load_all_available_dialects()

  lowered_operations: set[ir.Operation | ir.OpView] = set()

  def _lower_op(op: ir.OpView):
    if op.name not in _lowerings:
      return
    lowering_rule = _lowerings[op.name]

    # TODO(bchetioui): make sure all layouts are set here.
    if should_have_layout(op) and not has_any_layout_set(op):
      raise ValueError(f"{op} is missing a layout and can not be lowered.")

    new_results = lowering_rule(op)

    for old, new in zip(op.results, new_results):
      old.replace_all_uses_with(new)
    lowered_operations.add(op)

  def _traverse_and_lower_op(op: ir.OpView):
    for region in op.operation.regions:
      for block in region:
        for block_op in list(block):
          with ir.InsertionPoint(block_op):
            _traverse_and_lower_op(block_op)
    _lower_op(op)

  with ir.InsertionPoint(module.body):
    for op in list(module.body):
      _traverse_and_lower_op(op)

  for lowered_op in lowered_operations:
    lowered_op.erase()
