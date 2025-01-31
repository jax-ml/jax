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
import dataclasses
import functools
import operator
from typing import Sequence, Type, cast

import jax
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

from . import fragmented_array as fa
from . import launch_context
from . import layouts
from . import utils
from . import wgmma

# mypy: ignore-errors


@dataclasses.dataclass()
class LoweringContext:
  launch_context: launch_context.LaunchContext | None
  single_thread_per_block_predicate: ir.Value | None
  single_thread_per_warpgroup_predicate: ir.Value | None


MlirLoweringRule = Callable[
    [LoweringContext, ir.Operation | ir.OpView], Sequence[ir.Value]
]


_lowerings: dict[str, MlirLoweringRule] = {}


def _fragmented_array_to_ir(
    fragmented_array: fa.FragmentedArray, ty: ir.Type
) -> ir.Value:
  conversion_cast = builtin.UnrealizedConversionCastOp(
      [ty], fragmented_array.registers.flatten().tolist()
  )

  conversion_cast.attributes["registers_shape"] = ir.ArrayAttr.get([
      ir.IntegerAttr.get(ir.IntegerType.get_signless(64), s)
      for s in fragmented_array.registers.shape
  ])

  conversion_cast.attributes["layout"] = layouts.to_layout_attr(
      fragmented_array.layout
  )

  if fragmented_array.is_signed is not None:
    conversion_cast.attributes["is_signed"] = ir.BoolAttr.get(
        fragmented_array.is_signed
    )
  return conversion_cast.result


# TODO(bchetioui): add code that verifies the layout is as inferred.
def _fragmented_array_from_ir(
    fragmented_array_as_ir: ir.Value,
) -> fa.FragmentedArray:

  conversion_cast = cast(
      builtin.UnrealizedConversionCastOp, fragmented_array_as_ir.owner.opview  # pytype: disable=attribute-error
  )

  if not isinstance(conversion_cast, builtin.UnrealizedConversionCastOp):
    raise ValueError(f"{conversion_cast} is not a conversion_cast")

  converted_outputs = builtin.unrealized_conversion_cast(
      [operand.type for operand in conversion_cast.operands],
      conversion_cast.results,
  )
  if not isinstance(converted_outputs, list):
    converted_outputs = [converted_outputs]


  reverse_conversion_cast = converted_outputs[0].owner.opview
  for attribute in conversion_cast.attributes:
    attribute = cast(ir.NamedAttribute, attribute)
    reverse_conversion_cast.attributes[attribute.name] = attribute.attr

  registers = np.array(list(converted_outputs)).reshape(
    [attr.value for attr in conversion_cast.attributes["registers_shape"]]
  )
  layout = layouts.from_layout_attr(conversion_cast.attributes["layout"])

  if ir.IntegerType.isinstance(conversion_cast.outputs[0].type):
    is_signed = bool(conversion_cast.attributes["is_signed"])
  else:
    is_signed = None

  return fa.FragmentedArray(
      _registers=registers, _layout=layout, _is_signed=is_signed
  )


# TODO(dasenov): Remove this when minimum jaxlib version >= 0.5.1.
# Jaxlib doesn't contain the latest Mosaic GPU dialect bindings.
WaitOp = mgpu.WaitOp if jax.version._version == jax.lib.__version__ else None
ArriveExpectTxOp = mgpu.ArriveExpectTxOp if jax.version._version == jax.lib.__version__ else None

def _register_lowering(
    op: str | Type[ir.OpView] | None
) -> Callable[[MlirLoweringRule], MlirLoweringRule]:
  def wrapper(f):
    if op is not None:
      op_name = op if isinstance(op, str) else op.OPERATION_NAME  # pytype: disable=attribute-error
      _lowerings[op_name] = f
    return f

  return wrapper


def _lowered_barrier_type() -> ir.Type:
  return ir.IntegerType.get_signless(64)


@_register_lowering(mgpu.InitializeBarrierOp)
def _initialize_barrier_op_lowering_rule(
    ctx: LoweringContext,
    initialize_barrier_op: mgpu.InitializeBarrierOp,
) -> Sequence[ir.Value]:

  shape = initialize_barrier_op.barriers_ref.type.shape
  num_barriers = functools.reduce(operator.mul, shape, 1)

  i32 = ir.IntegerType.get_signless(32)
  workgroup_nvptx_address_space = utils.gpu_address_space_to_nvptx(
      gpu.AddressSpace.Workgroup)
  ptr_ty = ir.Type.parse(f"!llvm.ptr<{workgroup_nvptx_address_space}>")

  lowered_barrier_type = _lowered_barrier_type()

  for i in range(num_barriers):
    nvvm.mbarrier_init_shared(
        llvm.getelementptr(ptr_ty, initialize_barrier_op.base_pointer, [], [i],
                           lowered_barrier_type),
        utils.c(initialize_barrier_op.arrival_count.value, i32),
        predicate=ctx.single_thread_per_block_predicate
    )

  gpu.barrier()

  barrier_base_ptr = llvm.getelementptr(
      ir.Type.parse("!llvm.ptr"),
      initialize_barrier_op.base_pointer, [], [0], lowered_barrier_type)

  return utils.ptr_as_memref(
      barrier_base_ptr, initialize_barrier_op.barriers_ref.type),


@_register_lowering(vector.LoadOp)
def _vector_load_op_lowering_rule(
    _: LoweringContext, vector_load_op: vector.LoadOp
) -> Sequence[ir.Value]:
  (out_layout_attr,) = cast(
      ir.ArrayAttr, vector_load_op.attributes["out_layouts"]
  )

  if not layouts.is_strided_fragmented_layout(out_layout_attr):
    raise ValueError(
        f"{vector_load_op} has an unsupported layout: {out_layout_attr}"
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

  fragmented_array = fa.FragmentedArray.load_strided(vector_load_op.base)
  return [_fragmented_array_to_ir(fragmented_array, vector_load_op.result.type)]


@_register_lowering(vector.StoreOp)
def _vector_store_op_lowering_rule(
     _: LoweringContext, vector_store_op: vector.StoreOp
) -> Sequence[ir.Value]:
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

  # TODO(dasenov): This is not efficient for WGMMA layouts
  fragmented_array.store_untiled(vector_store_op.base)

  return []

@_register_lowering(vector.SplatOp)
def _vector_splat_op_lowering_rule(
    _: LoweringContext, vector_splat_op: vector.SplatOp
) -> Sequence[ir.Value]:

  out_vec_ty = ir.VectorType(vector_splat_op.aggregate.type)
  fragmented_array = fa.FragmentedArray.splat(
      vector_splat_op.input,
      tuple(out_vec_ty.shape),
      layouts.from_layout_attr(vector_splat_op.attributes["out_layouts"][0]),
  )
  return [_fragmented_array_to_ir(fragmented_array, out_vec_ty)]


@_register_lowering(mgpu.AsyncLoadOp)
def _mgpu_async_load_op_lowering_rule(
    ctx: LoweringContext, load_op: mgpu.AsyncLoadOp
) -> Sequence[ir.Value]:
  assert ctx.launch_context is not None
  barrier = utils.BarrierRef.from_dialect_barrier_memref(load_op.barrier)
  # TODO(dasenov): Add support for the remaining op properties.
  ctx.launch_context.async_copy(
      src_ref=load_op.source,
      dst_ref=load_op.destination,
      barrier=barrier,
      arrive=False,
      uniform=True,
      swizzle=load_op.swizzle.value,
      predicate=ctx.single_thread_per_warpgroup_predicate,
  )
  return []


@_register_lowering(mgpu.AsyncStoreOp)
def _mgpu_async_store_op_lowering_rule(
    ctx: LoweringContext, store_op: mgpu.AsyncStoreOp
) -> Sequence[ir.Value]:
  assert ctx.launch_context is not None
  # TODO(dasenov): Add support for the remaining op properties.
  ctx.launch_context.async_copy(
      src_ref=store_op.source,
      dst_ref=store_op.destination,
      swizzle=store_op.swizzle.value,
      uniform=True,
      predicate=ctx.single_thread_per_warpgroup_predicate,
  )
  return []


@_register_lowering(arith.AddFOp)
def _arith_addf_op_lowering_rule(
    _: LoweringContext, add: arith.AddFOp
) -> Sequence[ir.Value]:

  fragmented_array_lhs = _fragmented_array_from_ir(add.lhs)
  fragmented_array_rhs = _fragmented_array_from_ir(add.rhs)

  return [
      _fragmented_array_to_ir(
          fragmented_array_lhs + fragmented_array_rhs, add.result.type
      )
  ]


@_register_lowering(mgpu.WGMMAOp)
def _mgpu_wgmma_op_lowering_rule(
    _: LoweringContext, wgmma_op: mgpu.WGMMAOp
) -> Sequence[ir.Value]:

  # TODO(dasenov): Move the value -> accumulator conversion outisde of wgmma.
  # The associated fence could be a little expensive and is not needed if the
  # result a wgmma feeds into another wgmma (even in another loop step).
  acc_in = _fragmented_array_from_ir(wgmma_op.accumulator)
  regs = acc_in.to_layout(fa.WGMMA_LAYOUT)
  acc = wgmma.WGMMAAccumulator.from_registers(regs)

  a_operand = wgmma_op.a
  if ir.VectorType.isinstance(a_operand.type):
    a_operand = _fragmented_array_from_ir(a_operand)

  new_acc = wgmma.wgmma(
      acc,
      a_operand,
      wgmma_op.b,
      swizzle=wgmma_op.swizzle.value,
  )

  return [_fragmented_array_to_ir(new_acc.value, wgmma_op.accumulator.type)]


@_register_lowering(ArriveExpectTxOp)
def _mgpu_arrive_expect_tx_op_lowering_rule(
    ctx: LoweringContext, arrive_expect_tx_op: ArriveExpectTxOp
) -> Sequence[ir.Value]:

  barrier = utils.BarrierRef.from_dialect_barrier_memref(arrive_expect_tx_op.barrier)
  barrier.arrive_expect_tx(
      arrive_expect_tx_op.expect_tx.value,
      ctx.single_thread_per_warpgroup_predicate,
  )

  return []


@_register_lowering(WaitOp)
def _mgpu_wait_op_lowering_rule(
    _: LoweringContext, wait_op: WaitOp
) -> Sequence[ir.Value]:

  barrier = utils.BarrierRef.from_dialect_barrier_memref(wait_op.barrier)
  barrier.wait_parity(wait_op.parity)

  return []


def single_thread_predicates(module: ir.Module) -> tuple[ir.Value, ir.Value]:
  """Returns a single thread predicate per block and one per warpgroup."""
  block_predicate = warpgroup_predicate = None
  for op in module.body.operations:
    for region in op.operation.regions:
      for block in region.blocks:
        for sub_op in block.operations:
          if sub_op.operation.name == "gpu.launch":
            with ir.InsertionPoint.at_block_begin(
                sub_op.operation.regions[0].blocks[0]
            ):
              assert block_predicate is None
              block_predicate = utils.single_thread_predicate(per_block=True)
              warpgroup_predicate = utils.single_thread_predicate(
                  per_block=False
              )

  if block_predicate is None:
    raise ValueError(
        "No suitable function found to instantiate the single thread"
        " predicates."
    )

  return block_predicate, warpgroup_predicate


def lower_mgpu_dialect(
    module: ir.Module, launch_context: launch_context.LaunchContext | None
):
  # TODO(bchetioui): rethink this API. It doesn't make sense to pass in a full
  # module and to traverse all `gpu.LaunchOp`s if we have a `LaunchContext` that
  # references a single `gpu.LaunchOp`.
  #
  # A `LaunchContext` should have all the information needed to lower a single
  # kernel.
  module.context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  module.context.load_all_available_dialects()

  lowered_operations: set[ir.Operation | ir.OpView] = set()

  # TODO(bchetioui): fix tests to not have a test-only path polluting the API.
  if launch_context is None:  # this case is used in some tests
    block_predicate = warpgroup_predicate = None
  else:
    block_predicate, warpgroup_predicate = single_thread_predicates(module)

  ctx = LoweringContext(launch_context, block_predicate, warpgroup_predicate)

  def _lower_op(op: ir.OpView):
    if op.name not in _lowerings:
      return
    lowering_rule = _lowerings[op.name]

    # TODO(bchetioui): make sure all layouts are set here.
    if layouts.should_have_layout(op) and not layouts.has_any_layout_set(op):
      raise ValueError(f"{op} is missing a layout and can not be lowered.")

    new_results = lowering_rule(ctx, op)

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
