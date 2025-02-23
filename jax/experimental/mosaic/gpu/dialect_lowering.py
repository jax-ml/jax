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
from typing import Any, Sequence, Type, cast

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
  """Converts a FragmentedArray to an IR value.

  The fragmented array's signedness is omitted from the IR representation.
  """
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

  return conversion_cast.result


# TODO(bchetioui): add code that verifies the layout is as inferred.
def _fragmented_array_from_ir(
    fragmented_array_as_ir: ir.Value,
    is_signed: bool | None = None,
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

  if ir.IntegerType.isinstance(conversion_cast.outputs[0].type.element_type):
    is_signed = False if is_signed is None else is_signed

  return fa.FragmentedArray(
      _registers=registers, _layout=layout, _is_signed=is_signed
  )


# TODO(dasenov): Remove this when minimum jaxlib version >= 0.5.1.
# Jaxlib doesn't contain the latest Mosaic GPU dialect bindings.
WaitOp = getattr(mgpu, "WaitOp", None)
ArriveExpectTxOp = getattr(mgpu, "ArriveExpectTxOp", None)

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

  element_type = vector_load_op.result.type.element_type
  is_signed = False if ir.IntegerType.isinstance(element_type) else None

  fragmented_array = fa.FragmentedArray.load_strided(
      vector_load_op.base, is_signed=is_signed
  )
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
  is_signed = (
      False if ir.IntegerType.isinstance(out_vec_ty.element_type) else None
  )
  fragmented_array = fa.FragmentedArray.splat(
      vector_splat_op.input,
      tuple(out_vec_ty.shape),
      layouts.from_layout_attr(vector_splat_op.attributes["out_layouts"][0]),
      is_signed=is_signed,
  )
  return [_fragmented_array_to_ir(fragmented_array, out_vec_ty)]


def memref_layout_to_swizzle_and_transforms(
    layout: ir.Attribute,
) -> tuple[mgpu.SwizzlingMode, tuple[launch_context.MemRefTransform, ...]]:
  """Returns the swizzle and transforms that are encoded in the given layout.

    If the layout is not a LayoutAttr, the swizzle is kNoSwizzle and the
    transforms are empty. Otherwise, the layout may have at most one swizzle
    transform and any combination of tiling and transpose transforms.
  """
  swizzle = None
  gmem_transforms: list[launch_context.MemRefTransform] = []

  if mgpu.LayoutAttr.isinstance(layout):
    transforms_attr = mgpu.LayoutAttr(layout).transforms
    for transform in transforms_attr:
      if swizzle is not None:
        raise ValueError(f"{layout} contains more transforms after the initial swizzle.")
      if mgpu.SwizzleTransformAttr.isinstance(transform):
        # TODO(dasenov): Swizzling can change if the ref is sliced in certain
        # ways. We might want to enforce some restrictions here.
        swizzle = mgpu.SwizzleTransformAttr(transform).swizzle
      elif mgpu.TileTransformAttr.isinstance(transform):
        tiling = mgpu.TileTransformAttr(transform).tiling
        tiling_transform = launch_context.TileTransform(tuple(tiling))
        gmem_transforms.append(tiling_transform)
      elif mgpu.TransposeTransformAttr.isinstance(transform):
        permutation = mgpu.TransposeTransformAttr(transform).permutation
        transpose_transform = launch_context.TransposeTransform(
            tuple(permutation)
        )
        gmem_transforms.append(transpose_transform)
      else:
        raise ValueError(f"{layout} has an unsupported transform: {transform}")

  return swizzle or mgpu.SwizzlingMode.kNoSwizzle, tuple(gmem_transforms)


def transform_memref(
    mem_ref: ir.Value, transforms: tuple[launch_context.MemRefTransform, ...]
) -> ir.Value:
  """Reinterprets the memref to one where the shape is transformed as given."""
  if not transforms:
    return mem_ref

  mem_ref_type = ir.MemRefType(mem_ref.type)
  if mem_ref_type.memory_space != ir.Attribute.parse(
      "#gpu.address_space<workgroup>"
  ):
    raise ValueError(f"Only workgroup memory is supported but got {mem_ref}.")

  shape = mem_ref_type.shape
  for t in transforms:
    shape = t.transform_shape(shape)

  memref_new_type = ir.MemRefType.get(
      shape,
      mem_ref_type.element_type,
      memory_space=mem_ref_type.memory_space,
  )

  ms = utils.WORKGROUP_NVPTX_ADDRESS_SPACE
  ptr = utils.memref_ptr(mem_ref, memory_space=ms)
  return utils.ptr_as_memref(ptr, memref_new_type, ptr_memory_space=ms)


@_register_lowering(mgpu.AsyncLoadOp)
def _mgpu_async_load_op_lowering_rule(
    ctx: LoweringContext, load_op: mgpu.AsyncLoadOp
) -> Sequence[ir.Value]:
  assert ctx.launch_context is not None
  barrier = utils.BarrierRef.from_dialect_barrier_memref(load_op.barrier)

  dst_layout = ir.MemRefType(load_op.destination.type).layout
  swizzle, transforms = memref_layout_to_swizzle_and_transforms(dst_layout)
  # TODO(dasenov): Add support for the remaining op properties.
  ctx.launch_context.async_copy(
      src_ref=load_op.source,
      dst_ref=transform_memref(load_op.destination, transforms),
      barrier=barrier,
      arrive=False,
      uniform=True,
      swizzle=swizzle,
      gmem_transform=transforms,
      predicate=ctx.single_thread_per_warpgroup_predicate,
  )
  return []


@_register_lowering(mgpu.AsyncStoreOp)
def _mgpu_async_store_op_lowering_rule(
    ctx: LoweringContext, store_op: mgpu.AsyncStoreOp
) -> Sequence[ir.Value]:
  assert ctx.launch_context is not None

  src_layout = ir.MemRefType(store_op.source.type).layout
  swizzle, transforms = memref_layout_to_swizzle_and_transforms(src_layout)
  # TODO(dasenov): Add support for the remaining op properties.
  ctx.launch_context.async_copy(
      src_ref=transform_memref(store_op.source, transforms),
      dst_ref=store_op.destination,
      swizzle=swizzle,
      gmem_transform=transforms,
      uniform=True,
      predicate=ctx.single_thread_per_warpgroup_predicate,
  )
  return []


def _binary_op_lowering_rule(
    _: LoweringContext,
    op: Any,
    is_signed: bool | None,
    impl: Callable[
        [fa.FragmentedArray, fa.FragmentedArray], fa.FragmentedArray
    ],
) -> Sequence[ir.Value]:
  lhs = _fragmented_array_from_ir(op.lhs, is_signed)
  rhs = _fragmented_array_from_ir(op.rhs, is_signed)
  return [_fragmented_array_to_ir(impl(lhs, rhs), op.result.type)]


for op, impl, is_signed in [
    (arith.AddIOp, operator.add, False),
    (arith.AddFOp, operator.add, None),
    (arith.SubIOp, operator.sub, False),
    (arith.SubFOp, operator.sub, None),
    (arith.MulIOp, operator.mul, False),
    (arith.MulFOp, operator.mul, None),
    (arith.FloorDivSIOp, operator.floordiv, True),
    (arith.DivUIOp, operator.floordiv, False),
    (arith.DivFOp, operator.truediv, None),
    (arith.RemSIOp, operator.mod, True),
    (arith.RemUIOp, operator.mod, False),
    (arith.RemFOp, operator.mod, None),
    (arith.AndIOp, operator.and_, False),
    (arith.OrIOp, operator.or_, False),
    (arith.XOrIOp, operator.xor, False),
    (arith.MaxSIOp, fa.FragmentedArray.max, True),
    (arith.MaxUIOp, fa.FragmentedArray.max, False),
    (arith.MaximumFOp, fa.FragmentedArray.max, None),
    (arith.MinSIOp, fa.FragmentedArray.min, True),
    (arith.MinUIOp, fa.FragmentedArray.min, False),
    (arith.MinimumFOp, fa.FragmentedArray.min, None),
]:
  _lowerings[op.OPERATION_NAME] = functools.partial(
      _binary_op_lowering_rule, impl=impl, is_signed=is_signed
  )


CMPI_IMPLS = {
    arith.CmpIPredicate.eq: (operator.eq, False),
    arith.CmpIPredicate.ne: (operator.ne, False),
    arith.CmpIPredicate.slt: (operator.lt, True),
    arith.CmpIPredicate.sle: (operator.le, True),
    arith.CmpIPredicate.sgt: (operator.gt, True),
    arith.CmpIPredicate.sge: (operator.ge, True),
    arith.CmpIPredicate.ult: (operator.lt, False),
    arith.CmpIPredicate.ule: (operator.le, False),
    arith.CmpIPredicate.ugt: (operator.gt, False),
    arith.CmpIPredicate.uge: (operator.ge, False),
}


@_register_lowering(arith.CmpIOp)
def _cmpi_op_lowering_rule(
    _: LoweringContext, op: arith.CmpIOp
) -> Sequence[ir.Value]:
  impl, is_signed = CMPI_IMPLS[op.predicate.value]
  lhs = _fragmented_array_from_ir(op.lhs, is_signed)
  rhs = _fragmented_array_from_ir(op.rhs, is_signed)
  return [_fragmented_array_to_ir(impl(lhs, rhs), op.result.type)]


CMPF_IMPLS = {
    arith.CmpFPredicate.OEQ: operator.eq,
    arith.CmpFPredicate.UNE: operator.ne,
    arith.CmpFPredicate.OLT: operator.lt,
    arith.CmpFPredicate.OLE: operator.le,
    arith.CmpFPredicate.OGT: operator.gt,
    arith.CmpFPredicate.OGE: operator.ge,
}


@_register_lowering(arith.CmpFOp)
def _cmpf_op_lowering_rule(
    _: LoweringContext, op: arith.CmpFOp
) -> Sequence[ir.Value]:
  impl = CMPF_IMPLS[op.predicate.value]
  lhs = _fragmented_array_from_ir(op.lhs)
  rhs = _fragmented_array_from_ir(op.rhs)
  return [_fragmented_array_to_ir(impl(lhs, rhs), op.result.type)]


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

  b_layout = ir.MemRefType(wgmma_op.b.type).layout
  b_swizzle, b_transforms = memref_layout_to_swizzle_and_transforms(b_layout)

  if ir.VectorType.isinstance(wgmma_op.a.type):
    a_operand = _fragmented_array_from_ir(wgmma_op.a)
  else:
    a_layout = ir.MemRefType(wgmma_op.a.type).layout
    a_swizzle, a_transforms = memref_layout_to_swizzle_and_transforms(a_layout)
    if a_swizzle != b_swizzle:
      raise ValueError(
          f"Non-matching swizzles of operands a and b in WGMMA: {a_swizzle} !="
          f" {b_swizzle}"
      )
    a_operand = transform_memref(wgmma_op.a, a_transforms)

  new_acc = wgmma.wgmma(
      acc,
      a_operand,
      transform_memref(wgmma_op.b, b_transforms),
      swizzle=b_swizzle,
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


def _should_lower(op: ir.OpView) -> bool:
  """Returns 'true' if the operation should be lowered."""
  if isinstance(op.name, ir.StringAttr):
    name = op.name.value
  else:
    name = op.name
  return name.startswith("mosaic_gpu.") or layouts.should_have_layout(op)


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
    if not _should_lower(op):
      return

    if op.name not in _lowerings:
      raise NotImplementedError(f"Missing lowering rule for {op.name}")

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
