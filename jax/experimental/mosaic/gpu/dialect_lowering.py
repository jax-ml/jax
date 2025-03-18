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
import itertools
import operator
from typing import Any, Sequence, Type, cast

from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib import mosaic_gpu_dialect as mgpu
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import builtin
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import llvm
from jax._src.lib.mlir.dialects import math as mlir_math
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import nvvm
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
import numpy as np

from . import fragmented_array as fa
from . import inference_utils
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
  lowered_operations: set[ir.Operation | ir.OpView] = dataclasses.field(
      default_factory=set
  )

  def lower_op(self, op: ir.OpView):
    if not _should_lower(op):
      return

    if (name := op.OPERATION_NAME) not in _lowerings:  # pytype: disable=attribute-error
      raise NotImplementedError(f"Missing lowering rule for {op}")

    lowering_rule = _lowerings[name]

    # TODO(bchetioui): make sure all layouts are set here.
    if inference_utils.should_have_layout(
        op
    ) and not inference_utils.has_any_layout_set(op):
      raise ValueError(f"{op} is missing a layout and can not be lowered.")

    new_results = lowering_rule(self, op)
    if new_results is not RECURSED:
      for old, new in zip(op.results, new_results):
        old.replace_all_uses_with(new)
      self.lowered_operations.add(op)


class Recursed:
  pass
RECURSED = Recursed()

MlirLoweringRuleResult = Sequence[ir.Value] | Recursed
MlirLoweringRule = Callable[
    [LoweringContext, ir.Operation | ir.OpView], MlirLoweringRuleResult
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


def _fragmented_array_from_ir(
    fragmented_array_as_ir: ir.Value,
    layout: ir.Attribute,
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
  producer_layout = layouts.from_layout_attr(conversion_cast.attributes["layout"])

  if ir.IntegerType.isinstance(conversion_cast.outputs[0].type.element_type):
    is_signed = False if is_signed is None else is_signed

  return fa.FragmentedArray(
      _registers=registers, _layout=producer_layout, _is_signed=is_signed
  ).to_layout(layouts.from_layout_attr(layout))


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


@_register_lowering(arith.ConstantOp)
def _arith_constant_op_lowering_rule(
    _: LoweringContext, op: arith.ConstantOp
) -> Sequence[ir.Value]:
  if not ir.DenseElementsAttr.isinstance(op.value):
    raise NotImplementedError(f"Unsupported constant op: {op}")

  value = ir.DenseElementsAttr(op.value)
  if not value.is_splat:
    raise NotImplementedError(f"Unsupported constant op: {op}")

  ty = ir.VectorType(op.result.type)
  is_signed = False if ir.IntegerType.isinstance(ty.element_type) else None

  return [
      _fragmented_array_to_ir(
          fa.FragmentedArray.splat(
              arith.constant(ty.element_type, value.get_splat_value()),
              tuple(ty.shape),
              layouts.from_layout_attr(op.attributes["out_layouts"][0]),
              is_signed=is_signed,
          ),
          op.result.type,
      )
  ]


def _check_transforms_and_swizzle_are_supported(
    ref_ty: ir.MemRefType,
    transforms: Sequence[launch_context.MemRefTransform],
    swizzle: mgpu.SwizzlingMode,
    minimum_swizzle: mgpu.SwizzlingMode = mgpu.SwizzlingMode.kNoSwizzle,
):
  """Checks that the list of provided transforms and swizzle are supported.

  Currently, we allow the following:
    - any swizzle that is larger than or equal to `minimum_swizzle`;
    - optionally, a single tile transform (with rank equal to the rank of the
      memref being annotated);
    - optionally, a single transpose transform.
  """
  if swizzle < minimum_swizzle:
    raise NotImplementedError(
        f"Unsupported swizzle {swizzle} smaller than {minimum_swizzle}."
    )

  partitioned_transforms = {
      k: list(v)
      for k, v in itertools.groupby(
          transforms, lambda t: isinstance(t, launch_context.TileTransform)
      )
  }

  tile_transforms = partitioned_transforms.get(True, [])
  other_transforms = partitioned_transforms.get(False, [])

  if len(tile_transforms) > 1:
    raise NotImplementedError(
        f"{tile_transforms} contains more than one tile transform."
    )

  if len(tile_transforms) == 1:
    if len(tile_transforms[0].tiling) != len(ref_ty.shape):
      raise NotImplementedError(
          f"Only tile transforms with rank equal to the rank of the memref "
          f"being annotated are supported but got {tile_transforms[0]} for "
          f"{ref_ty}."
      )

  if len(other_transforms) > 1:
    raise NotImplementedError(
        f"{other_transforms} contains more than one transform."
    )

  if len(other_transforms) == 1:
    if not isinstance(other_transforms[0], launch_context.TransposeTransform):
      raise NotImplementedError(
          f"{other_transforms[0]} is not a transpose transform."
      )


@_register_lowering(vector.LoadOp)
def _vector_load_op_lowering_rule(
    _: LoweringContext, vector_load_op: vector.LoadOp
) -> Sequence[ir.Value]:
  (out_layout_attr,) = cast(
      ir.ArrayAttr, vector_load_op.attributes["out_layouts"]
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

  if layouts.is_strided_fragmented_layout(out_layout_attr):
    strided_layout = layouts.from_strided_fragmented_layout_attr(
        out_layout_attr
    )
    fragmented_array = fa.FragmentedArray.load_strided(
        vector_load_op.base,
        is_signed=is_signed,
        vec_size=strided_layout.vec_size,
    )
  elif layouts.from_layout_attr(out_layout_attr) == fa.WGMMA_LAYOUT:
    swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
        inference_utils.in_transforms(vector_load_op)[0]
    )
    ref_ty = ir.MemRefType(vector_load_op.base.type)
    _check_transforms_and_swizzle_are_supported(ref_ty, transforms, swizzle)
    transformed_ref = transform_memref(vector_load_op.base, transforms)
    fragmented_array = fa.FragmentedArray.load_tiled(
        transformed_ref,
        swizzle=swizzle,
        is_signed=is_signed,
        layout=fa.WGMMA_LAYOUT,
    )
  else:
    raise ValueError(
        f"{vector_load_op} has an unsupported layout: {out_layout_attr}"
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

  [to_store_layout] = inference_utils.in_layouts(vector_store_op)
  fragmented_array = _fragmented_array_from_ir(
      vector_store_op.valueToStore, to_store_layout
  )

  if fragmented_array.layout == fa.WGMMA_LAYOUT:
    swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
        inference_utils.in_transforms(vector_store_op)[0]
    )
    ref_ty = ir.MemRefType(vector_store_op.base.type)
    _check_transforms_and_swizzle_are_supported(ref_ty, transforms, swizzle)
    fragmented_array.store_tiled(
        transform_memref(vector_store_op.base, transforms), swizzle
    )
  elif (isinstance(fragmented_array.layout, fa.WGStridedFragLayout) or
        isinstance(fragmented_array.layout, fa.WGSplatFragLayout)):
    fragmented_array.store_untiled(vector_store_op.base)
  else:
    raise ValueError(
        f"{vector_store_op} has an unsupported layout: {to_store_layout}"
    )

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


@_register_lowering(vector.ShapeCastOp)
def _vector_shape_cast_op_lowering_rule(
    _: LoweringContext, op: vector.ShapeCastOp
) -> Sequence[ir.Value]:
  [layout] = inference_utils.in_layouts(op)
  out_vec_ty = ir.VectorType(op.result.type)
  assert out_vec_ty.has_static_shape
  is_signed = (
      False if ir.IntegerType.isinstance(out_vec_ty.element_type) else None
  )
  a = _fragmented_array_from_ir(op.source, layout, is_signed)
  return [_fragmented_array_to_ir(a.reshape(out_vec_ty.shape), out_vec_ty)]


@_register_lowering(vector.ReductionOp)
def _vector_reduction_op_lowering_rule(
    ctx: LoweringContext, op: vector.ReductionOp
) -> Sequence[ir.Value]:
  del ctx  # Unused.
  [layout] = inference_utils.in_layouts(op)
  () = inference_utils.out_layouts(op)
  element_type = ir.VectorType(op.vector.type).element_type
  is_signed = False if ir.IntegerType.isinstance(element_type) else None
  a = _fragmented_array_from_ir(op.vector, layout, is_signed)
  match str(op.kind):
    case "#vector.kind<add>":
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      scratch = _slice_smem(
          ir.MemRefType.get([4], element_type, memory_space=smem),
          arith.constant(None, op.attributes["offset"]),
      )
      result = a.reduce_sum(scratch)
    case (
        "#vector.kind<maxsi>" | "#vector.kind<maxui>" | "#vector.kind<maximumf>"
    ):
      # TODO(slebedev): Implement this and remove the raise below.
      raise NotImplementedError(f"Unsupported reduction kind: {op.kind}")
    case _:
      raise NotImplementedError(f"Unsupported reduction kind: {op.kind}")
  return [_fragmented_array_to_ir(result, op.result.type)]


def swizzle_and_transforms_from_transforms_attr(
    transforms: ir.ArrayAttr,
) -> tuple[mgpu.SwizzlingMode, tuple[launch_context.MemRefTransform, ...]]:
  """Returns the swizzle and MemrefTransforms for the given transforms.

  Args:
    transforms: a list of transform attributes.

  Returns:
    A tuple containing the swizzle mode and MemRefTransforms corresponding to
    the parameter transforms. If `transforms` is empty, or does not contain
    any swizzling transform, the swizzle mode is assumed to be kNoSwizzle.
  Raises:
    ValueError: if a swizzling transform is followed by any transform.
  """
  swizzle = None
  gmem_transforms: list[launch_context.MemRefTransform] = []

  for transform in transforms:
    if swizzle is not None:
      raise ValueError(f"{transforms} contain more transforms after swizzle.")
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
      raise ValueError("Unknown transform: {transform}")

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

  if inference_utils.has_in_transforms_set(load_op):
    [transforms] = inference_utils.in_transforms(load_op)
    swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
        transforms
    )
  else:
    swizzle = mgpu.SwizzlingMode.kNoSwizzle
    transforms = ()

  gmem_slice = []
  for idx_i32, size in zip(load_op.indices, load_op.slice_lengths):
    idx = arith.index_cast(ir.IndexType.get(), idx_i32)
    v = idx if size < 0 else utils.DynamicSlice(idx, size)
    gmem_slice.append(v)

  # TODO(dasenov): Add support for the remaining op properties.
  ctx.launch_context.async_copy(
      src_ref=load_op.source,
      dst_ref=transform_memref(load_op.destination, transforms),
      gmem_slice=tuple(gmem_slice),
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

  if inference_utils.has_in_transforms_set(store_op):
    [transforms] = inference_utils.in_transforms(store_op)
    swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
        transforms
    )
  else:
    swizzle = mgpu.SwizzlingMode.kNoSwizzle
    transforms = ()

  gmem_slice = []
  for idx_i32, size in zip(store_op.indices, store_op.slice_lengths):
    idx = arith.index_cast(ir.IndexType.get(), idx_i32)
    v = idx if size < 0 else utils.DynamicSlice(idx, size)
    gmem_slice.append(v)

  # TODO(dasenov): Add support for the remaining op properties.
  ctx.launch_context.async_copy(
      src_ref=transform_memref(store_op.source, transforms),
      dst_ref=store_op.destination,
      gmem_slice=tuple(gmem_slice),
      swizzle=swizzle,
      gmem_transform=transforms,
      uniform=True,
      predicate=ctx.single_thread_per_warpgroup_predicate,
      arrive=store_op.commit_group,
  )
  return []


def _conversion_op_lowering_rule(
    _: LoweringContext,
    op: ir.OpView,
    source_is_signed: bool | None,
    target_is_signed: bool | None,
) -> Sequence[ir.Value]:
  [in_layout] = inference_utils.in_layouts(op)
  [layout] = inference_utils.out_layouts(op)
  if in_layout != layout:
    raise ValueError("Layout mismatch")

  target_ty = op.result.type.element_type  # pytype: disable=attribute-error
  operand = _fragmented_array_from_ir(op.operands[0], layout, source_is_signed)
  converted = operand.astype(target_ty, is_signed=target_is_signed)
  return [_fragmented_array_to_ir(converted, op.result.type)]


for op, source_is_signed, target_is_signed in [
    (arith.ExtFOp, None, None),
    (arith.ExtSIOp, True, True),
    (arith.ExtUIOp, False, False),
    (arith.FPToSIOp, None, True),
    (arith.FPToUIOp, None, False),
    (arith.SIToFPOp, True, None),
    (arith.TruncFOp, None, None),
    (arith.TruncIOp, False, False),
    (arith.UIToFPOp, False, None),
]:
  _lowerings[op.OPERATION_NAME] = functools.partial(
      _conversion_op_lowering_rule,
      source_is_signed=source_is_signed,
      target_is_signed=target_is_signed,
  )


def _unary_op_lowering_rule(
    _: LoweringContext,
    op: Any,
    impl: Callable[[fa.FragmentedArray], fa.FragmentedArray],
    is_signed: bool | None = None,
) -> Sequence[ir.Value]:
  in_layouts = inference_utils.in_layouts(op)
  [layout] = inference_utils.out_layouts(op)
  if any(in_layout != layout for in_layout in in_layouts):
    raise ValueError("Layout mismatch")
  kwargs = {}
  if hasattr(op, "fastmath"):
    kwargs = dict(
        approx=op.fastmath == ir.Attribute.parse("#arith.fastmath<afn>")
    )
  a = _fragmented_array_from_ir(op.operand, layout, is_signed)
  return [_fragmented_array_to_ir(impl(a, **kwargs), op.result.type)]


for op, impl, is_signed in [
    (mlir_math.RsqrtOp, fa.FragmentedArray.rsqrt, None),
    (mlir_math.ExpOp, fa.FragmentedArray.exp, None),
    (mlir_math.Exp2Op, fa.FragmentedArray.exp2, None),
    (mlir_math.LogOp, fa.FragmentedArray.log, None),
    (mlir_math.TanhOp, fa.FragmentedArray.tanh, None),
]:
  _lowerings[op.OPERATION_NAME] = functools.partial(
      _unary_op_lowering_rule, impl=impl, is_signed=is_signed
  )


def _binary_op_lowering_rule(
    _: LoweringContext,
    op: Any,
    is_signed: bool | None,
    impl: Callable[
        [fa.FragmentedArray, fa.FragmentedArray], fa.FragmentedArray
    ],
) -> Sequence[ir.Value]:
  in_layouts = inference_utils.in_layouts(op)
  [layout] = inference_utils.out_layouts(op)
  if any(in_layout != layout for in_layout in in_layouts):
    raise ValueError("Layout mismatch")
  lhs = _fragmented_array_from_ir(op.lhs, layout, is_signed)
  rhs = _fragmented_array_from_ir(op.rhs, layout, is_signed)
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
  in_layouts = inference_utils.in_layouts(op)
  [layout] = inference_utils.out_layouts(op)
  if any(in_layout != layout for in_layout in in_layouts):
    raise ValueError("Layout mismatch")
  impl, is_signed = CMPI_IMPLS[op.predicate.value]
  lhs = _fragmented_array_from_ir(op.lhs, layout, is_signed)
  rhs = _fragmented_array_from_ir(op.rhs, layout, is_signed)
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
  in_layouts = inference_utils.in_layouts(op)
  [layout] = inference_utils.out_layouts(op)
  if any(in_layout != layout for in_layout in in_layouts):
    raise ValueError("Layout mismatch")
  impl = CMPF_IMPLS[op.predicate.value]
  lhs = _fragmented_array_from_ir(op.lhs, layout)
  rhs = _fragmented_array_from_ir(op.rhs, layout)
  return [_fragmented_array_to_ir(impl(lhs, rhs), op.result.type)]


@_register_lowering(arith.BitcastOp)
def _bitcast_op_lowering_rule(
    _: LoweringContext, op: arith.BitcastOp
) -> Sequence[ir.Value]:
  in_layouts = inference_utils.in_layouts(op)
  [layout] = inference_utils.out_layouts(op)
  if any(in_layout != layout for in_layout in in_layouts):
    raise ValueError("Layout mismatch")
  in_ = _fragmented_array_from_ir(op.in_, layout)
  out_element_type = ir.VectorType(op.result.type).element_type
  out = in_.bitcast(
      out_element_type,
      output_is_signed=False
      if ir.IntegerType.isinstance(out_element_type)
      else None,
  )
  return [_fragmented_array_to_ir(out, op.result.type)]


@_register_lowering(mgpu.WGMMAOp)
def _mgpu_wgmma_op_lowering_rule(
    _: LoweringContext, wgmma_op: mgpu.WGMMAOp
) -> Sequence[ir.Value]:
  if wgmma_op.transpose_a or wgmma_op.transpose_b:
    raise ValueError("Transpose arguments are to be deleted.")

  fa_layouts = (
      *inference_utils.in_layouts(wgmma_op),
      *inference_utils.out_layouts(wgmma_op),
  )
  is_supported_layout = (
      lambda l: layouts.from_tiled_layout_attr(l) == fa.WGMMA_LAYOUT
  )
  if not all(map(is_supported_layout, fa_layouts)):
    raise ValueError("Layout mismatch")
  wgmma_layout = fa_layouts[0]

  # TODO(dasenov): Move the value -> accumulator conversion outisde of wgmma.
  # The associated fence could be a little expensive and is not needed if the
  # result a wgmma feeds into another wgmma (even in another loop step).
  acc_in = _fragmented_array_from_ir(wgmma_op.accumulator, wgmma_layout)
  regs = acc_in.to_layout(fa.WGMMA_LAYOUT)
  acc = wgmma.WGMMAAccumulator.from_registers(regs)

  if ir.VectorType.isinstance(wgmma_op.a.type):
    a_transforms = None
    b_transforms = inference_utils.in_transforms(wgmma_op)[0]
  else:
    a_transforms, b_transforms = inference_utils.in_transforms(wgmma_op)

  b_swizzle, b_transforms = swizzle_and_transforms_from_transforms_attr(
      b_transforms
  )
  minimum_swizzle = mgpu.SwizzlingMode.k32ByteSwizzle
  ref_ty = ir.MemRefType(wgmma_op.b.type)
  _check_transforms_and_swizzle_are_supported(
      ref_ty, b_transforms, b_swizzle, minimum_swizzle
  )
  b_operand = transform_memref(wgmma_op.b, b_transforms)

  if ir.VectorType.isinstance(wgmma_op.a.type):
    a_operand = _fragmented_array_from_ir(wgmma_op.a, wgmma_layout)
  else:
    a_swizzle, a_transforms = swizzle_and_transforms_from_transforms_attr(
        a_transforms
    )
    ref_ty = ir.MemRefType(wgmma_op.a.type)
    _check_transforms_and_swizzle_are_supported(
        ref_ty, a_transforms, a_swizzle, minimum_swizzle
    )
    if a_swizzle != b_swizzle:
      raise ValueError(
          f"Non-matching swizzles of operands a and b in WGMMA: {a_swizzle} !="
          f" {b_swizzle}"
      )
    a_operand = transform_memref(wgmma_op.a, a_transforms)

  new_acc = wgmma.wgmma(acc, a_operand, b_operand, swizzle=b_swizzle)

  return [
      _fragmented_array_to_ir(
          new_acc.value.to_layout(fa.WGMMA_LAYOUT),
          wgmma_op.accumulator.type,
      )
  ]


@_register_lowering(mgpu.ArriveExpectTxOp)
def _mgpu_arrive_expect_tx_op_lowering_rule(
    ctx: LoweringContext, arrive_expect_tx_op: mgpu.ArriveExpectTxOp
) -> Sequence[ir.Value]:

  barrier = utils.BarrierRef.from_dialect_barrier_memref(arrive_expect_tx_op.barrier)
  barrier.arrive_expect_tx(
      arrive_expect_tx_op.expect_tx.value,
      ctx.single_thread_per_warpgroup_predicate,
  )

  return []


@_register_lowering(mgpu.WaitOp)
def _mgpu_wait_op_lowering_rule(
    _: LoweringContext, wait_op: mgpu.WaitOp
) -> Sequence[ir.Value]:

  barrier = utils.BarrierRef.from_dialect_barrier_memref(wait_op.barrier)
  barrier.wait_parity(wait_op.parity)

  return []


# TODO(bchetioui): remove this once jaxlib minimum version >= 0.5.2.
SliceSMEMOp = getattr(mgpu, "SliceSMEMOp", None)


@_register_lowering(SliceSMEMOp)
def _mgpu_slice_smem_op_lowering_rule(
    ctx: LoweringContext, op: SliceSMEMOp
) -> Sequence[ir.Value]:
  del ctx
  return [_slice_smem(op.result.type, op.offset)]


def _slice_smem(result: ir.Type, offset: ir.Value):
  i8 = ir.IntegerType.get_signless(8)
  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
  smem_base = gpu.dynamic_shared_memory(
      ir.MemRefType.get((utils.DYNAMIC,), i8, memory_space=smem)
  )
  offset = arith.index_cast(ir.IndexType.get(), offset)
  return memref.view(result, smem_base, offset, [])


@_register_lowering(scf.ForOp)
def _for_op_lowering_rule(
    ctx: LoweringContext, for_op: scf.ForOp
) -> MlirLoweringRuleResult:
  if not inference_utils.should_have_layout(for_op):
    return _traverse_op_lowering_rule(ctx, for_op)
  in_layouts = inference_utils.in_layouts(for_op)
  out_layouts = inference_utils.out_layouts(for_op)
  yield_op = for_op.body.operations[len(for_op.body.operations) - 1]
  yield_layouts = inference_utils.in_layouts(yield_op)
  if in_layouts != out_layouts or in_layouts != yield_layouts:
    raise ValueError("Layout mismatch")
  fa_layouts = in_layouts

  fa_layouts_it = iter(fa_layouts)
  arg_template = [
      (_fragmented_array_from_ir(arg, next(fa_layouts_it)), arg.type)
      if ir.VectorType.isinstance(arg.type)
      else (arg, arg.type)
      for arg in for_op.initArgs
  ]
  def lower_carry(carry):
    fa_layouts_it = iter(fa_layouts)
    carry_with_fas = [
        _fragmented_array_from_ir(arg, next(fa_layouts_it))
        if ir.VectorType.isinstance(arg.type)
        else arg
        for arg in carry
    ]
    lowered_carry = []
    for c in carry_with_fas:
      if isinstance(c, fa.FragmentedArray):
        lowered_carry.extend(c.registers.flat)
      else:
        lowered_carry.append(c)
    return lowered_carry

  def recreate_carry(lowered_carry):
    recreated_carry = []
    arg_it = iter(lowered_carry)
    for arg_value, arg_type in arg_template:
      if isinstance(arg_value, fa.FragmentedArray):
        carry_registers = np.asarray(
            [next(arg_it) for _ in arg_value.registers.flat], dtype=object
        )
        carry_registers = carry_registers.reshape(arg_value.registers.shape)
        carry = fa.FragmentedArray(
            _registers=carry_registers,
            _layout=arg_value.layout,
            _is_signed=arg_value.is_signed,
        )
        recreated_carry.append(_fragmented_array_to_ir(carry, arg_type))
      else:
        recreated_carry.append(next(arg_it))
    return recreated_carry

  new_for_op = scf.ForOp(
      for_op.lowerBound,
      for_op.upperBound,
      for_op.step,
      lower_carry(for_op.initArgs),
  )
  with ir.InsertionPoint(new_for_op.body):
    recreated_carry = recreate_carry(new_for_op.body.arguments[1:])
    ops_to_lower = []
    for op in for_op.body:
      if op == yield_op:
        continue
      mgpu.private_operation_remove_from_parent(op)
      mgpu.private_block_append_owned_operation(new_for_op.body, op)
      ops_to_lower.append(op)
    new_args = (new_for_op.induction_variable, *recreated_carry)
    for old_carry, new_carry in zip(for_op.body.arguments, new_args, strict=True):
      old_carry.replace_all_uses_with(new_carry)

  for op in ops_to_lower:
    with ir.InsertionPoint(op):
      ctx.lower_op(op)

  with ir.InsertionPoint(new_for_op.body):
    new_yield_operands = lower_carry(yield_op.operands)
    yield_op.erase()
    scf.yield_(new_yield_operands)
  return recreate_carry(new_for_op.results)


@_register_lowering(func.FuncOp)
@_register_lowering(gpu.LaunchOp)
@_register_lowering(scf.IfOp)  # TODO(apaszke,bchetioui): Add a proper rule.
@_register_lowering(scf.IndexSwitchOp)  # TODO(apaszke,bchetioui): Add a proper rule.
def _traverse_op_lowering_rule(
    ctx: LoweringContext, op: ir.OpView
) -> MlirLoweringRuleResult:
  if inference_utils.should_have_layout(op):
    raise ValueError(
        f"Rule cannot handle an op with vector operands or results: {op}"
    )
  for region in op.operation.regions:
    for block in region:
      for block_op in list(block):
        with ir.InsertionPoint(block_op):
          ctx.lower_op(block_op)
  return RECURSED


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
  return (
      op.OPERATION_NAME.startswith("mosaic_gpu.")  # pytype: disable=attribute-error
      or inference_utils.should_have_layout(op)
      or any(bool(b) for r in op.regions for b in r)  # Does it have subblocks?
  )


def lower_mgpu_dialect(
    module: ir.Module,
    launch_context: launch_context.LaunchContext | None,
):
  # TODO(apaszke,bchetioui): Make sure the layouts match.
  # TODO(bchetioui): rethink this API. It doesn't make sense to pass in a full
  # module and to traverse all `gpu.LaunchOp`s if we have a `LaunchContext` that
  # references a single `gpu.LaunchOp`.
  #
  # A `LaunchContext` should have all the information needed to lower a single
  # kernel.
  module.context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  module.context.load_all_available_dialects()

  # TODO(bchetioui): fix tests to not have a test-only path polluting the API.
  if launch_context is None:  # this case is used in some tests
    block_predicate = warpgroup_predicate = None
  else:
    block_predicate, warpgroup_predicate = single_thread_predicates(module)

  ctx = LoweringContext(launch_context, block_predicate, warpgroup_predicate)
  with ir.InsertionPoint(module.body):
    for op in list(module.body):
      ctx.lower_op(op)

  for lowered_op in ctx.lowered_operations:
    lowered_op.erase()
