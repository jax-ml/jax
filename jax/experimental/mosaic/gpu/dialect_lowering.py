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

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import functools
import itertools
import math
import operator
from typing import Any, cast

from jax._src import lib as jaxlib
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
from jax._src.util import safe_zip
from jax.experimental.mosaic.gpu import layouts as layouts_lib
from jax.experimental.mosaic.gpu import utils as mgpu_utils
import numpy as np

from . import fragmented_array as fa
from . import inference_utils
from . import launch_context
from . import layouts
from . import tcgen05
from . import utils
from . import wgmma


@dataclasses.dataclass()
class LoweringContext:
  launch_context: launch_context.LaunchContext | None
  single_thread_per_block_predicate: ir.Value | None
  single_thread_per_warpgroup_predicate: ir.Value | None
  single_warp_per_block_predicate: ir.Value | None
  auto_barriers: bool
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
    if not isinstance(new_results, Recursed):
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


def _undo_conversion_cast(
    ir_value: ir.Value,
    expected_types: Sequence[ir.Type],
) -> tuple[builtin.UnrealizedConversionCastOp, Sequence[ir.Value]]:
  """Undoes the provided unrealized conversion cast.

  The `ir_value` must be an unrealized conversion cast. This function will
  create a new conversion cast that undoes the original one. The returned tuple
  contains:
  - The original unrealzied conversion cast (useful for extract attributes).
  - The list of operands of the original conversion cast (which are the result
    values of the undone conversion cast).

  The function will verify that the returned values have types that match
  `expected_types`.
  """
  conversion_cast = cast(
      builtin.UnrealizedConversionCastOp, ir_value.owner.opview  # pytype: disable=attribute-error
  )

  if not isinstance(conversion_cast, builtin.UnrealizedConversionCastOp):
    raise ValueError(f"{conversion_cast} is not a conversion_cast")

  converted_outputs = builtin.unrealized_conversion_cast(
      [operand.type for operand in conversion_cast.operands],
      conversion_cast.results,
  )
  if isinstance(converted_outputs, ir.OpResultList):
    converted_outputs = list(converted_outputs)
  elif not isinstance(converted_outputs, list):
    converted_outputs = [converted_outputs]

  for v, t in zip(converted_outputs, expected_types, strict=True):
    if v.type != t:
      raise ValueError(f"Expected type {t} for value {v}")

  return conversion_cast, converted_outputs


def fragmented_array_to_ir(
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
  producer_layout_attr = fragmented_array_as_ir.owner.attributes["layout"]
  producer_layout = layouts.from_layout_attr(producer_layout_attr)
  vector_ty = ir.VectorType(fragmented_array_as_ir.type)
  reg_shape = producer_layout.registers_shape(tuple(vector_ty.shape))
  reg_ty = producer_layout.registers_element_type(vector_ty.element_type)

  conversion_cast, converted_outputs = _undo_conversion_cast(
      fragmented_array_as_ir, [reg_ty] * math.prod(reg_shape)
  )

  reverse_conversion_cast = converted_outputs[0].owner.opview
  for attribute in conversion_cast.attributes:
    attribute = cast(ir.NamedAttribute, attribute)
    reverse_conversion_cast.attributes[attribute.name] = attribute.attr

  registers = np.array(list(converted_outputs)).reshape(
    [attr.value for attr in conversion_cast.attributes["registers_shape"]]
  )

  if ir.IntegerType.isinstance(conversion_cast.outputs[0].type.element_type):
    is_signed = False if is_signed is None else is_signed

  return fa.FragmentedArray(
      _registers=registers, _layout=producer_layout, _is_signed=is_signed
  ).to_layout(layouts.from_layout_attr(layout))


def wrap_transformed_memref(
    transformed_memref: ir.Value,
    logical_type: ir.Type,
    transforms: ir.ArrayAttr,
) -> ir.Value:
  """Wraps a transformed memref to an unrealized cast with transforms.

  The return type of the cast is the untransformed logical type.
  """
  conversion_cast = builtin.UnrealizedConversionCastOp(
      [logical_type], [transformed_memref]
  )
  conversion_cast.attributes["transforms"] = transforms
  return conversion_cast.result


def unwrap_transformed_memref(
    ref: ir.Value, expected_transforms: ir.ArrayAttr
) -> ir.Value:
  """Uwraps a memref from an unrealized cast and verifies its transforms."""

  _, transforms = swizzle_and_transforms_from_transforms_attr(expected_transforms)
  transformed_type = transformed_smem_ref_type(ref.type, transforms)
  conversion_cast, [result] = _undo_conversion_cast(ref, [transformed_type])

  # Check that the actual transforms match the expected ones.
  if expected_transforms != conversion_cast.attributes["transforms"]:
    raise ValueError(
        f"Expected transforms {expected_transforms} do not match actual"
        f" transforms {conversion_cast.attributes['transforms']}"
    )

  return result


def _register_lowering(
    op: str | type[ir.OpView] | None
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
  num_barriers = math.prod(shape)

  i32 = ir.IntegerType.get_signless(32)
  workgroup_nvptx_address_space = utils.gpu_address_space_to_nvptx(
      gpu.AddressSpace.Workgroup)
  ptr_ty = ir.Type.parse(f"!llvm.ptr<{workgroup_nvptx_address_space}>")

  lowered_barrier_type = _lowered_barrier_type()

  for i in range(num_barriers):
    nvvm.mbarrier_init_shared(
        llvm.getelementptr(
            ptr_ty,
            initialize_barrier_op.base_pointer,
            [],
            [i],
            lowered_barrier_type,
            llvm.GEPNoWrapFlags.none,
        ),
        utils.c(
            initialize_barrier_op.arrival_count.value * utils.WARPGROUP_SIZE,
            i32,
        ),
        predicate=ctx.single_thread_per_block_predicate,
    )

  gpu.barrier()

  barrier_base_ptr = llvm.getelementptr(
      ir.Type.parse("!llvm.ptr"),
      initialize_barrier_op.base_pointer, [], [0], lowered_barrier_type, llvm.GEPNoWrapFlags.none)

  return utils.ptr_as_memref(
      barrier_base_ptr, initialize_barrier_op.barriers_ref.type),


@_register_lowering(mgpu.OptimizationBarrierOp)
def _optimization_barrier_op_lowering_rule(
    _: LoweringContext,
    op: mgpu.OptimizationBarrierOp,
) -> Sequence[ir.Value]:
  if not all(ir.VectorType.isinstance(operand.type) for operand in op.operands):
    raise NotImplementedError(
        f"Optimization barrier op {op} has non-vector operands."
    )

  fragmented_arrays = []
  for operand, layout in safe_zip(op.operands, inference_utils.in_layouts(op)):
    ty = ir.VectorType(operand.type)
    is_signed = False if ir.IntegerType.isinstance(ty.element_type) else None
    fragmented_arrays.append(
        _fragmented_array_from_ir(operand, layout, is_signed=is_signed)
    )

  lowered_fragmented_arrays = fa.optimization_barrier(*fragmented_arrays)
  if isinstance(lowered_fragmented_arrays, fa.FragmentedArray):
    lowered_fragmented_arrays = [lowered_fragmented_arrays]

  return [
      fragmented_array_to_ir(arr, result.type)
      for arr, result in safe_zip(lowered_fragmented_arrays, op.results)
  ]


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
      fragmented_array_to_ir(
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

  tile_transforms = cast(
      list[launch_context.TileTransform],
      partitioned_transforms.get(True, []),
  )
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

  def _fragmented_array_to_ir(fragmented_array: fa.FragmentedArray) -> ir.Value:
    return fragmented_array_to_ir(fragmented_array, vector_load_op.result.type)

  if layouts.is_strided_fragmented_layout(out_layout_attr):
    strided_layout = layouts.from_strided_fragmented_layout_attr(
        out_layout_attr
    )
    # TODO(bchetioui): Process transforms.
    fragmented_array = fa.FragmentedArray.load_strided(
        vector_load_op.base,
        is_signed=is_signed,
        vec_size=strided_layout.vec_size,
    )
    return [_fragmented_array_to_ir(fragmented_array)]

  if not layouts.is_tiled_layout(out_layout_attr):
    raise ValueError(
        f"{vector_load_op} has an unsupported layout: {out_layout_attr}"
    )

  layout = layouts.from_tiled_layout_attr(out_layout_attr)
  ref_ty = ir.MemRefType(vector_load_op.base.type)
  if ref_ty.memory_space is None:  # GMEM
    fragmented_array = fa.FragmentedArray.load_untiled(
        vector_load_op.base,
        layout=layout,
        optimized=False,
    )
    return [_fragmented_array_to_ir(fragmented_array)]

  if ref_ty.memory_space != utils.smem():
    raise ValueError(f"Unsupported memory space: {ref_ty.memory_space}")

  transforms_attr = inference_utils.in_transforms(vector_load_op)[0]
  swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
      transforms_attr
  )
  has_transforms = swizzle != mgpu.SwizzlingMode.kNoSwizzle or transforms
  if has_transforms:
    _check_transforms_and_swizzle_are_supported(ref_ty, transforms, swizzle)
    transformed_ref = unwrap_transformed_memref(
        vector_load_op.base, transforms_attr
    )
    fragmented_array = fa.FragmentedArray.load_tiled(
        transformed_ref,
        swizzle=swizzle,
        is_signed=is_signed,
        layout=layout,
    )
  else:
    is_tmem_native = layout == tcgen05.TMEM_NATIVE_LAYOUT
    fragmented_array = fa.FragmentedArray.load_untiled(
        vector_load_op.base,
        layout=layout,
        optimized=not is_tmem_native,
    )

  return [_fragmented_array_to_ir(fragmented_array)]


@_register_lowering(vector.StoreOp)
def _vector_store_op_lowering_rule(
     ctx: LoweringContext, vector_store_op: vector.StoreOp
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

  if ctx.auto_barriers:
    mgpu_utils.warpgroup_barrier()  # Make sure the reads have completed.

  ref = vector_store_op.base
  ref_type = ir.MemRefType(ref.type)

  if ref_type.memory_space is None:  # GMEM
    fragmented_array.store_untiled(ref, optimized=False)
  elif ref_type.memory_space == utils.smem():
    transforms_attr = inference_utils.in_transforms(vector_store_op)[0]
    swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
        transforms_attr
    )
    has_transforms = swizzle != mgpu.SwizzlingMode.kNoSwizzle or transforms
    if has_transforms:
      _check_transforms_and_swizzle_are_supported(ref_type, transforms, swizzle)
      unwrapped_ref = unwrap_transformed_memref(ref, transforms_attr)
      fragmented_array.store_tiled(unwrapped_ref, swizzle)
    else:
      is_tmem_native = fragmented_array.layout == tcgen05.TMEM_NATIVE_LAYOUT
      fragmented_array.store_untiled(ref, optimized=not is_tmem_native)
  else:
    raise ValueError(f"Unsupported memory space: {ref_type.memory_space}")

  if ctx.auto_barriers:
    mgpu_utils.warpgroup_barrier()  # Make sure the writes have completed.

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
  return [fragmented_array_to_ir(fragmented_array, out_vec_ty)]


@_register_lowering(vector.BroadcastOp)
def _vector_broadcast_op_lowering_rule(
    _: LoweringContext, vector_broadcast_op: vector.BroadcastOp
) -> Sequence[ir.Value]:

  out_vec_ty = ir.VectorType(vector_broadcast_op.vector.type)
  is_signed = (
      False if ir.IntegerType.isinstance(out_vec_ty.element_type) else None
  )
  fragmented_array = fa.FragmentedArray.splat(
      vector_broadcast_op.source,
      tuple(out_vec_ty.shape),
      layouts.from_layout_attr(
          vector_broadcast_op.attributes["out_layouts"][0]
      ),
      is_signed=is_signed,
  )
  return [fragmented_array_to_ir(fragmented_array, out_vec_ty)]


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
  return [fragmented_array_to_ir(a.reshape(out_vec_ty.shape), out_vec_ty)]


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
      scratch = _slice_smem(
          ir.MemRefType.get([4], element_type, memory_space=utils.smem()),
          arith.constant(None, op.attributes["offset"]),
      )
      result = a.reduce("add", range(len(a.shape)), scratch)
    case (
        "#vector.kind<maxsi>" | "#vector.kind<maxui>" | "#vector.kind<maximumf>"
    ):
      # TODO(slebedev): Implement this and remove the raise below.
      raise NotImplementedError(f"Unsupported reduction kind: {op.kind}")
    case _:
      raise NotImplementedError(f"Unsupported reduction kind: {op.kind}")
  return [fragmented_array_to_ir(result, op.result.type)]

@_register_lowering(vector.MultiDimReductionOp)
def _vector_multi_dim_reduction_op_lowering_rule(
    ctx: LoweringContext, op: vector.MultiDimReductionOp
) -> Sequence[ir.Value]:
  del ctx

  [in_layout, acc_layout] = inference_utils.in_layouts(op)
  [out_layout] = inference_utils.out_layouts(op)
  if layouts.from_layout_attr(in_layout) != fa.WGMMA_LAYOUT:
    raise NotImplementedError(f"Unsupported input layout: {in_layout}")
  if layouts.from_layout_attr(out_layout) not in {
      fa.WGMMA_ROW_LAYOUT,
      fa.WGMMA_COL_LAYOUT,
  }:
    raise NotImplementedError(f"Unsupported output layout: {out_layout}")
  if out_layout != acc_layout:
    raise ValueError(
        f"Output layout {out_layout} must match the accumulator layout"
        f" {acc_layout}"
    )

  element_type = ir.VectorType(op.source.type).element_type

  is_signed = False if ir.IntegerType.isinstance(element_type) else None
  source_fa = _fragmented_array_from_ir(op.source, in_layout, is_signed)
  acc_fa = _fragmented_array_from_ir(op.acc, acc_layout, is_signed)
  match vector.CombiningKind[
      str(op.kind).removeprefix("#vector.kind<").removesuffix(">").upper()
  ]:
    case vector.CombiningKind.ADD:
      result = source_fa.reduce("add", op.reduction_dims[0])
      result += acc_fa
    case (
        vector.CombiningKind.MAXIMUMF
        | vector.CombiningKind.MAXSI
        | vector.CombiningKind.MAXUI
    ):
      result = source_fa.reduce("max", op.reduction_dims[0])
      result = result.max(acc_fa)
    case _:
      raise NotImplementedError(f"Unsupported reduction kind: {op.kind}")
  return [fragmented_array_to_ir(result, op.result.type)]


@_register_lowering(mgpu.LayoutCastOp)
def _mgpu_layout_cast_op_lowering_rule(
    _: LoweringContext, op: mgpu.LayoutCastOp
) -> Sequence[ir.Value]:
  [in_layout] = inference_utils.in_layouts(op)
  [out_layout] = inference_utils.out_layouts(op)
  in_array = _fragmented_array_from_ir(op.x, in_layout)
  out_array = in_array.to_layout(layouts.from_layout_attr(out_layout))
  return [fragmented_array_to_ir(out_array, op.result.type)]


@_register_lowering(mgpu.BroadcastInDimOp)
def _mgpu_broadcast_in_dim_op_lowering_rule(
    _: LoweringContext, op: mgpu.BroadcastInDimOp
) -> Sequence[ir.Value]:
  in_ty = ir.VectorType(op.operand.type)
  out_ty = ir.VectorType(op.result.type)
  if len(in_ty.shape) != 1 or len(out_ty.shape) != 2:
    raise NotImplementedError(
        "Broadcast in dim with non-trivial broadcast dimensions is not"
        f" supported: {op}"
    )

  broadcast_dims = list(op.broadcast_dimensions)
  in_layout = inference_utils.in_layouts(op)[0]
  operand_fa = _fragmented_array_from_ir(op.operand, in_layout)

  if operand_fa.layout == fa.WGMMA_ROW_LAYOUT and broadcast_dims == [0]:
    out = operand_fa.broadcast_minor(out_ty.shape[1])
  elif operand_fa.layout == fa.WGMMA_COL_LAYOUT and broadcast_dims == [1]:
    out = operand_fa.broadcast_in_dim(out_ty.shape, (1,), fa.WGMMA_LAYOUT)
  else:
    raise NotImplementedError(
        "Broadcast in dim with non-trivial broadcast dimensions is not"
        f" supported: {op}"
    )
  return [fragmented_array_to_ir(out, out_ty)]


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


def _is_memref_transposed(mem_ref_type: ir.MemRefType) -> bool:
  strides, _ = mem_ref_type.get_strides_and_offset()
  prev_stride = math.inf
  for stride in strides:
    if stride > prev_stride:
      return True
    prev_stride = stride
  return False


def transformed_smem_ref_type(
    ref_ty: ir.MemRefType,
    transforms: tuple[launch_context.MemRefTransform, ...],
) -> ir.MemRefType:
  """Returns the transformed ref type for the given logical ref and transforms.
  """
  transposed = _is_memref_transposed(ref_ty)
  if not transforms and not transposed:
    return ref_ty

  if not utils.is_smem_ref(ref_ty):
    raise ValueError(f"Only workgroup memory is supported but got {ref_ty}.")

  shape = ref_ty.shape
  strides, offset = ref_ty.get_strides_and_offset()
  if transposed:
    if len(shape) != 2:
      raise NotImplementedError(
          f"Only 2D shapes can be transposed, but got {shape}"
      )
    if strides[0] != 1 or strides[1] != shape[0]:
      raise NotImplementedError(
          f"Only contiguous 2D memrefs can be transposed, but got {ref_ty}"
      )

  for t in transforms:
    shape = list(t.transform_shape(shape))

  minor_to_major_stride_order: tuple[int, ...]
  if transposed:
    # The expected output is a transposed ref and `shape` is already transposed.
    # We need to compute the correct strides to match the shape.
    if len(shape) == 2:
      minor_to_major_stride_order = (1, 0)
    elif len(shape) == 4:
      minor_to_major_stride_order = (2, 3, 0, 1)
    else:
      raise NotImplementedError(
          f"Expected a 2D or 4D shape after transforms, but got {shape}"
      )
  else:
    minor_to_major_stride_order = tuple(reversed(range(len(shape))))

  new_strides = [1] * len(shape)
  for i in range(1, len(shape)):
    dim = minor_to_major_stride_order[i]
    prev_dim = minor_to_major_stride_order[i-1]
    new_strides[dim] = new_strides[prev_dim] * shape[prev_dim]

  new_ref_ty = ir.MemRefType.get(
      shape,
      ref_ty.element_type,
      memory_space=ref_ty.memory_space,
      layout=ir.StridedLayoutAttr.get(offset, new_strides),
  )
  return new_ref_ty


def reinterpret_smem_ref(
    ref: ir.Value,
    transforms: tuple[launch_context.MemRefTransform, ...],
) -> ir.Value:
  """Applies transforms on the ref, and makes sure that their effect is
  propagated appropriately on the strides.

  This function is used any time we lower from a dialect SMEM ref (2D for wgmma)
  with given transforms to a "physical" SMEM ref (4D for wgmma) that is fully
  transformed and transposed as needed.
  """
  ref_ty = ir.MemRefType(ref.type)
  new_ref_ty = transformed_smem_ref_type(ref_ty, transforms)
  if ref_ty == new_ref_ty:
    return ref
  ms = utils.WORKGROUP_NVPTX_ADDRESS_SPACE
  ptr = utils.memref_ptr(ref, memory_space=ms)
  new_ref = utils.ptr_as_memref(ptr, new_ref_ty, ptr_memory_space=ms)
  return new_ref


@_register_lowering(mgpu.AsyncLoadOp)
def _mgpu_async_load_op_lowering_rule(
    ctx: LoweringContext, load_op: mgpu.AsyncLoadOp
) -> Sequence[ir.Value]:
  assert ctx.launch_context is not None
  barrier = utils.DialectBarrierRef.from_barrier_memref(load_op.barrier)

  [transforms_attr] = inference_utils.in_transforms(load_op)
  swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
      transforms_attr
  )
  unwrapped_destination = unwrap_transformed_memref(
      load_op.destination, transforms_attr
  )

  gmem_slice = []
  for idx_i32, size in zip(load_op.indices, load_op.slice_lengths):
    idx = arith.index_cast(ir.IndexType.get(), idx_i32)
    v = idx if size < 0 else utils.DynamicSlice(idx, size)
    gmem_slice.append(v)

  # TODO(dasenov): async_copy requires all GMEM strides except the last one
  # to be a multiple of 16 bytes. This restriction could be loosned with
  # strided layouts when they are contiguous in GMEM. In that case, we could do:
  # flatten -> async_copy -> unflatted here, as long as flattened size is a
  # multiple of 16.

  # TODO(dasenov): Add support for the remaining op properties.
  if ctx.auto_barriers:
    mgpu_utils.warpgroup_barrier()  # Make sure the writes have completed.
  ctx.launch_context.async_copy(
      src_ref=load_op.source,
      dst_ref=unwrapped_destination,
      gmem_slice=tuple(gmem_slice),
      barrier=barrier.barrier_ref,
      arrive=False,
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

  [transforms_attr] = inference_utils.in_transforms(store_op)
  swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
      transforms_attr
  )
  unwrapped_source = unwrap_transformed_memref(store_op.source, transforms_attr)

  gmem_slice = []
  for idx_i32, size in zip(store_op.indices, store_op.slice_lengths):
    idx = arith.index_cast(ir.IndexType.get(), idx_i32)
    v = idx if size < 0 else utils.DynamicSlice(idx, size)
    gmem_slice.append(v)

  # TODO(dasenov): async_copy requires all GMEM strides except the last one
  # to be a multiple of 16 bytes. This restriction could be loosned with
  # strided layouts when they are contiguous in GMEM. In that case, we could do:
  # flatten -> async_copy -> unflatted here, as long as flattened size is a
  # multiple of 16.

  # TODO(dasenov): Add support for the remaining op properties.
  ctx.launch_context.async_copy(
      src_ref=unwrapped_source,
      dst_ref=store_op.destination,
      gmem_slice=tuple(gmem_slice),
      swizzle=swizzle,
      gmem_transform=transforms,
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
  return [fragmented_array_to_ir(converted, op.result.type)]


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
    impl: Callable[..., fa.FragmentedArray],
    is_signed: bool | None = None,
) -> Sequence[ir.Value]:
  in_layouts = inference_utils.in_layouts(op)
  [layout] = inference_utils.out_layouts(op)
  if any(in_layout != layout for in_layout in in_layouts):
    raise ValueError("Layout mismatch")
  a = _fragmented_array_from_ir(op.operand, layout, is_signed)
  if hasattr(op, "fastmath"):
    approx = op.fastmath == ir.Attribute.parse("#arith.fastmath<afn>")
    result_fa = impl(a, approx=approx)
  else:
    result_fa = impl(a)

  return [fragmented_array_to_ir(result_fa, op.result.type)]


for op, unary_impl, is_signed in [
    (mlir_math.RsqrtOp, fa.FragmentedArray.rsqrt, None),
    (mlir_math.ExpOp, fa.FragmentedArray.exp, None),
    (mlir_math.Exp2Op, fa.FragmentedArray.exp2, None),
    (mlir_math.LogOp, fa.FragmentedArray.log, None),
    (mlir_math.TanhOp, fa.FragmentedArray.tanh, None),
]:
  _lowerings[op.OPERATION_NAME] = functools.partial(
      _unary_op_lowering_rule, impl=unary_impl, is_signed=is_signed
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
  return [fragmented_array_to_ir(impl(lhs, rhs), op.result.type)]


for op, binary_impl, is_signed in [
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
      _binary_op_lowering_rule, impl=binary_impl, is_signed=is_signed
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
  return [fragmented_array_to_ir(impl(lhs, rhs), op.result.type)]


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
  return [fragmented_array_to_ir(impl(lhs, rhs), op.result.type)]


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
  return [fragmented_array_to_ir(out, op.result.type)]


@_register_lowering(mgpu.WGMMAOp)
def _mgpu_wgmma_op_lowering_rule(
    _: LoweringContext, wgmma_op: mgpu.WGMMAOp
) -> Sequence[ir.Value]:
  fa_layouts = (
      *inference_utils.in_layouts(wgmma_op),
      *inference_utils.out_layouts(wgmma_op),
  )
  wgmma_layout = layouts.to_layout_attr(fa.WGMMA_LAYOUT)
  for layout in fa_layouts:
    if layout != wgmma_layout:
      raise ValueError("Layout mismatch")

  # TODO(dasenov): Move the value -> accumulator conversion outside of wgmma.
  # The associated fence could be a little expensive and is not needed if the
  # result a wgmma feeds into another wgmma (even in another loop step).
  regs = _fragmented_array_from_ir(wgmma_op.accumulator, wgmma_layout)
  acc = wgmma.WGMMAAccumulator.from_registers(regs)

  if ir.VectorType.isinstance(wgmma_op.a.type):
    a_transforms = None
    b_transforms = inference_utils.in_transforms(wgmma_op)[0]
    unwrapped_a_ref = None
    unwrapped_b_ref = unwrap_transformed_memref(wgmma_op.b, b_transforms)
  else:
    a_transforms, b_transforms = inference_utils.in_transforms(wgmma_op)
    unwrapped_a_ref = unwrap_transformed_memref(wgmma_op.a, a_transforms)
    unwrapped_b_ref = unwrap_transformed_memref(wgmma_op.b, b_transforms)

  b_swizzle, b_transforms = swizzle_and_transforms_from_transforms_attr(
      b_transforms
  )
  minimum_swizzle = mgpu.SwizzlingMode.k32ByteSwizzle
  _check_transforms_and_swizzle_are_supported(
      ir.MemRefType(wgmma_op.b.type), b_transforms, b_swizzle, minimum_swizzle
  )

  if ir.VectorType.isinstance(wgmma_op.a.type):
    a_operand = _fragmented_array_from_ir(wgmma_op.a, wgmma_layout)
  else:
    a_swizzle, a_transforms = swizzle_and_transforms_from_transforms_attr(
        a_transforms
    )
    _check_transforms_and_swizzle_are_supported(
        ir.MemRefType(wgmma_op.a.type), a_transforms, a_swizzle, minimum_swizzle
    )
    if a_swizzle != b_swizzle:
      raise ValueError(
          f"Non-matching swizzles of operands a and b in WGMMA: {a_swizzle} !="
          f" {b_swizzle}"
      )
    assert unwrapped_a_ref is not None
    a_operand = unwrapped_a_ref

  new_acc = wgmma.wgmma(acc, a_operand, unwrapped_b_ref, swizzle=b_swizzle)

  return [
      fragmented_array_to_ir(
          new_acc.value.to_layout(fa.WGMMA_LAYOUT),
          wgmma_op.accumulator.type,
      )
  ]


@_register_lowering(mgpu.ArriveExpectTxOp)
def _mgpu_arrive_expect_tx_op_lowering_rule(
    _: LoweringContext, arrive_expect_tx_op: mgpu.ArriveExpectTxOp
) -> Sequence[ir.Value]:
  bytes = arrive_expect_tx_op.expect_tx.value
  if bytes % utils.WARPGROUP_SIZE:
    raise NotImplementedError(
        "Only copies of a multiple of 128 bytes are supported"
    )
  # We arrive uniformly from each thread in the WG, so we need to divide the
  # number of bytes by the number of threads in the WG.
  # TODO: dasenov - Relax this. We can just select the WG leader and have it
  # arrive with the whole transfer size, while everyone else arrives with 0.
  # But we should continue using this scheme as it's likely to be faster.
  bytes //= utils.WARPGROUP_SIZE
  bytes = utils.c(bytes, ir.IntegerType.get_signless(32))

  barrier = utils.DialectBarrierRef.from_barrier_memref(
      arrive_expect_tx_op.barrier
  )
  nvvm.mbarrier_arrive_expect_tx_shared(barrier.get_ptr(), bytes)

  return []


@_register_lowering(mgpu.WaitOp)
def _mgpu_wait_op_lowering_rule(
    _: LoweringContext, wait_op: mgpu.WaitOp
) -> Sequence[ir.Value]:

  barrier = utils.DialectBarrierRef.from_barrier_memref(wait_op.barrier)
  barrier.wait_parity(wait_op.parity)

  return []


@_register_lowering(mgpu.SliceSMEMOp)
def _mgpu_slice_smem_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.SliceSMEMOp
) -> Sequence[ir.Value]:
  del ctx
  sliced_ref = _slice_smem(op.result.type, op.offset)

  memref_ty = ir.MemRefType(sliced_ref.type)
  if (
      memref_ty.element_type == ir.Type.parse("!mosaic_gpu.barrier")
  ):
    # Barrier memrefs are not transformed and must not be wrapped.
    assert not inference_utils.has_out_transforms_set(op)
    return [sliced_ref]

  out_transforms = inference_utils.out_transforms(op)[0]
  _, transforms = swizzle_and_transforms_from_transforms_attr(out_transforms)
  transformed_ref = reinterpret_smem_ref(sliced_ref, transforms)
  wrapped_ref = wrap_transformed_memref(transformed_ref, op.result.type, out_transforms)
  return [wrapped_ref]


def _slice_smem(result: ir.Type, offset: ir.Value):
  i8 = ir.IntegerType.get_signless(8)
  smem_base = gpu.dynamic_shared_memory(
      ir.MemRefType.get((utils.DYNAMIC,), i8, memory_space=utils.smem())
  )
  offset = arith.index_cast(ir.IndexType.get(), offset)
  lowered_result_type = result
  if ir.MemRefType.isinstance(result):
    memref_ty = ir.MemRefType(result)
    if memref_ty.element_type == ir.Type.parse("!mosaic_gpu.barrier"):
      lowered_result_type = ir.MemRefType.get(
          memref_ty.shape, _lowered_barrier_type(), memory_space=utils.smem()
      )
  view = memref.view(lowered_result_type, smem_base, offset, [])
  if result == lowered_result_type:
    return view
  return builtin.unrealized_conversion_cast([result], [view])


@_register_lowering(mgpu.WithTransformsOp)
def _mgpu_with_transforms_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.WithTransformsOp
) -> Sequence[ir.Value]:
  """Lowering rule for mgpu.WithTransformsOp.
  This is a noop that simply returns its input.
  """
  del ctx

  [in_transforms] = inference_utils.in_transforms(op)
  unwrapped_source_ref = unwrap_transformed_memref(op.ref, in_transforms)
  out_transforms = inference_utils.out_transforms(op)[0]
  wrapped_ref = wrap_transformed_memref(
      unwrapped_source_ref, op.result.type, out_transforms
  )
  return [wrapped_ref]


def _tile_transform_offsets(
    tiling: Sequence[int],
    static_offsets: Sequence[int],
    dynamic_offsets: Sequence[ir.Value],
) -> tuple[Sequence[int], Sequence[ir.Value]]:
  """Computes the static and dynamic offsets after the given tiling is applied.

  Conceptually, this function is analogous to
  tile.transform_shape(static_offsets), except that it also handles dynamic offsets.
  """
  dynamic_offset_index = 0
  new_static_offsets = []
  new_dynamic_offsets = []

  # Preserve all offsets in non-tiled dimensions.
  for offset in static_offsets[: -len(tiling)]:
    new_static_offsets.append(offset)
    if offset == ir.ShapedType.get_dynamic_stride_or_offset():
      new_dynamic_offsets.append(dynamic_offsets[dynamic_offset_index])
      dynamic_offset_index += 1

  # Compute static and dynamic offsets of tiled dimensions.
  for tile_size, offset in zip(
      tiling, static_offsets[-len(tiling) :], strict=True
  ):
    if offset == ir.ShapedType.get_dynamic_stride_or_offset():
      # Here we assume that the offset is divisble by the tile size, but we
      # don't check it. This has been established at the time the tiling was
      # inferred.
      dyn_offset = arith.divui(
          dynamic_offsets[dynamic_offset_index],
          utils.c(tile_size, ir.IndexType.get()),
      )
      new_dynamic_offsets.append(dyn_offset)
      new_static_offsets.append(ir.ShapedType.get_dynamic_stride_or_offset())
      dynamic_offset_index += 1
    else:
      assert offset % tile_size == 0
      new_static_offsets.append(offset // tile_size)

  # Add 0 offsets for the newly created dimension of the tile.
  new_static_offsets += [0] * len(tiling)

  return new_static_offsets, new_dynamic_offsets


@_register_lowering(memref.SubViewOp)
def _memref_subview_op_lowering_rule(
    ctx: LoweringContext, op: memref.SubViewOp
) -> Sequence[ir.Value]:
  del ctx

  in_transforms = inference_utils.in_transforms(op)[0]
  out_transforms = inference_utils.out_transforms(op)[0]

  if in_transforms != out_transforms:
    raise NotImplementedError(
        "SubViewOp transforms for the input and output refs must be identical."
    )

  if any(s != 1 for s in op.static_strides):
    raise NotImplementedError(
        "SubViewOp only supports static strides of 1."
    )

  if _is_memref_transposed(op.source.type):
    raise NotImplementedError(
        "SubViewOp does not support transposed memrefs."
    )

  unwrapped_source_ref = unwrap_transformed_memref(op.source, in_transforms)
  swizzle, transforms = swizzle_and_transforms_from_transforms_attr(out_transforms)
  if swizzle != mgpu.SwizzlingMode.kNoSwizzle:
    source_ty = ir.MemRefType(op.source.type)
    source_strides, _ = source_ty.get_strides_and_offset()
    for stride, slice, size in zip(source_strides, op.static_sizes, source_ty.shape, strict=True):
      if stride != 1:
        continue
      # A dimension with stride 1 is a minor dimension and is swizzled.
      if slice != size:
        raise NotImplementedError("Slicing a swizzled dimension is unsupported.")

  match transforms:
    case ():
      new_subview_op = memref.SubViewOp(
          op.result.type,
          unwrapped_source_ref,
          op.offsets,
          None,
          None,
          static_offsets=op.static_offsets,
          static_sizes=op.static_sizes,
          static_strides=op.static_strides,
      )
    case (tile_transform, ) if isinstance(tile_transform, launch_context.TileTransform):
      in_transformed_ty = ir.MemRefType(unwrapped_source_ref.type)
      tiling = tile_transform.tiling
      if any(
          ir.ShapedType.is_dynamic_size(s)
          for s in list(op.static_sizes)[-len(tiling) :]
      ):
        raise NotImplementedError(
            "SubViewOp only supports static sizes for the tiled dimensions."
        )
      new_sizes = tile_transform.transform_shape(list(op.static_sizes))
      new_static_offsets, new_dynamic_offsets = _tile_transform_offsets(
          tiling, list(op.static_offsets), list(op.offsets)
      )

      new_subview_op = memref.SubViewOp(
          transformed_smem_ref_type(op.result.type, transforms),
          unwrapped_source_ref,
          new_dynamic_offsets,
          None,
          None,
          static_offsets=new_static_offsets,
          static_sizes=new_sizes,
          static_strides=[1] * len(in_transformed_ty.shape),
      )
    case _:
      raise NotImplementedError(
          "SubViewOp only supports a single tile transform."
      )

  wrapped_ref = wrap_transformed_memref(
      new_subview_op.result, op.result.type, out_transforms
  )
  return [wrapped_ref]


@_register_lowering(memref.CastOp)
def _memref_cast_op_lowering_rule(
    ctx: LoweringContext, op: memref.CastOp
) -> Sequence[ir.Value]:
  """Lowering rule for memref.CastOp.
  Only casts that add a dynamic offset are supported.
  """
  del ctx

  in_transforms = inference_utils.in_transforms(op)[0]
  out_transforms = inference_utils.out_transforms(op)[0]
  if in_transforms != out_transforms:
    raise NotImplementedError(
        "CastOp transforms for the input and output refs must be identical."
    )

  in_ty = ir.MemRefType(op.source.type)
  out_ty = ir.MemRefType(op.result.type)
  if in_ty.element_type != out_ty.element_type:
    raise NotImplementedError(
        "CastOp only supports casts between memrefs with the same element type."
    )
  if in_ty.shape != out_ty.shape:
    raise NotImplementedError(
        "CastOp only supports casts between memrefs with the same shape."
    )
  in_strides, _ = in_ty.get_strides_and_offset()
  out_strides, out_offset = out_ty.get_strides_and_offset()
  if in_strides != out_strides:
    raise NotImplementedError(
        "CastOp only supports casts between memrefs with the same strides."
    )

  unwrapped_source_ref = unwrap_transformed_memref(op.source, in_transforms)
  in_transformed_ty = ir.MemRefType(unwrapped_source_ref.type)
  transformed_strides, _ = in_transformed_ty.get_strides_and_offset()
  out_layout = ir.StridedLayoutAttr.get(out_offset, transformed_strides)
  out_transformed_ty = ir.MemRefType.get(
      in_transformed_ty.shape,
      in_transformed_ty.element_type,
      memory_space=in_transformed_ty.memory_space,
      layout=out_layout,
  )
  new_cast_op = memref.CastOp(out_transformed_ty, unwrapped_source_ref)
  wrapped_ref = wrap_transformed_memref(
      new_cast_op.result, op.result.type, out_transforms
  )
  return [wrapped_ref]


def _permutation_to_affine_map_attr(
    permutation: Sequence[int],
) -> ir.AffineMapAttr:
  return ir.AffineMapAttr.get(ir.AffineMap.get_permutation(permutation))


@_register_lowering(memref.TransposeOp)
def _memref_transpose_op_lowering_rule(
    ctx: LoweringContext, op: memref.TransposeOp
) -> Sequence[ir.Value]:
  del ctx

  in_transforms = inference_utils.in_transforms(op)[0]
  unwrapped_in_ref = unwrap_transformed_memref(op.in_, in_transforms)
  in_transformed_ty = ir.MemRefType(unwrapped_in_ref.type)
  if len(in_transformed_ty.shape) == 2:
    new_permutation = op.permutation
  elif len(in_transformed_ty.shape) == 4:
    if op.permutation == _permutation_to_affine_map_attr([0, 1]):
      new_permutation = _permutation_to_affine_map_attr([0, 1, 2, 3])
    elif op.permutation == _permutation_to_affine_map_attr([1, 0]):
      new_permutation = _permutation_to_affine_map_attr([1, 0, 3, 2])
    else:
      raise NotImplementedError("Unsupported permutation.")
  else:
    raise NotImplementedError(
        "TransposeOp only supports transposing 2D and 4D memrefs."
    )

  out_transforms = inference_utils.out_transforms(op)[0]
  _, transforms = swizzle_and_transforms_from_transforms_attr(out_transforms)
  new_transpose_op = memref.TransposeOp(
      transformed_smem_ref_type(op.result.type, transforms),
      unwrapped_in_ref,
      new_permutation,
  )

  wrapped_ref = wrap_transformed_memref(
      new_transpose_op.result, op.result.type, out_transforms
  )
  return [wrapped_ref]


@_register_lowering(memref.LoadOp)
def _memref_load_op_lowering_rule(
    ctx: LoweringContext, op: memref.LoadOp
) -> Sequence[ir.Value]:
  """Lowering rule for memref.LoadOp.

  Loads are never transformed so this rule is mostly just a pass-through.
  """
  del ctx

  in_transforms = inference_utils.in_transforms(op)[0]
  if in_transforms:
    raise NotImplementedError(f"memref.LoadOp does not support transforms: {op}")

  new_load_op = memref.LoadOp(
      memref=unwrap_transformed_memref(op.memref, in_transforms),
      indices=op.indices,
      nontemporal=op.nontemporal,
  )
  return [new_load_op.result]


@_register_lowering(memref.StoreOp)
def _memref_store_op_lowering_rule(
    ctx: LoweringContext, op: memref.StoreOp
) -> Sequence[ir.Value]:
  """Lowering rule for memref.StoreOp.

  Stores are never transformed so this rule is mostly just a pass-through.
  """
  del ctx

  in_transforms = inference_utils.in_transforms(op)[0]
  if in_transforms:
    raise NotImplementedError(f"memref.StoreOp does not support transforms: {op}")

  memref.StoreOp(
      value=op.value,
      memref=unwrap_transformed_memref(op.memref, in_transforms),
      indices=op.indices,
      nontemporal=op.nontemporal,
  )
  return []


@_register_lowering(mgpu.TmemAllocOp)
def _tmem_alloc_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.TmemAllocOp
) -> Sequence[ir.Value]:
  """Lowering rule for mgpu.TmemAllocOp."""
  output_shape = ir.MemRefType(op.result.type).shape
  ncols = output_shape[1] // op.packing.value

  with mgpu_utils.when(ctx.single_warp_per_block_predicate):
    tcgen05.tmem_alloc(op.smem_ptr, ncols, op.collective, op.exact)
  gpu.barrier()
  tmem_addr = memref.load(op.smem_ptr, [])

  cast_op = builtin.UnrealizedConversionCastOp(
      [op.result.type], [tmem_addr]
  )
  cast_op.attributes["collective"] = op.collective
  cast_op.attributes["exact"] = op.exact
  cast_op.attributes["packing"] = op.packing

  return [cast_op.result]

# TODO(allanrenucci): Remove this after the minimal jaxlib version is 0.7.1.
if jaxlib.version >= (0, 7, 1):
  @_register_lowering(mgpu.TmemRelinquishAllocPermitOp)
  def _tmem_relinquish_alloc_permit_op_lowering_rule(
      ctx: LoweringContext, op: mgpu.TmemRelinquishAllocPermitOp
  ) -> Sequence[ir.Value]:
    """Lowering rule for mgpu.TmemRelinquishAllocPermitOp."""
    with mgpu_utils.when(ctx.single_warp_per_block_predicate):
      tcgen05.tmem_relinquish_alloc_permit(op.collective)
    return []


@_register_lowering(mgpu.TmemDeallocOp)
def _tmem_dealloc_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.TmemDeallocOp
) -> Sequence[ir.Value]:
  """Lowering rule for mgpu.TmemDeallocOp."""
  i32 = ir.IntegerType.get_signless(32)
  conversion_cast, [tmem_addr] = _undo_conversion_cast(op.tmem_ref, [i32])
  collective = ir.BoolAttr(conversion_cast.attributes["collective"]).value
  exact = ir.BoolAttr(conversion_cast.attributes["exact"]).value
  packing = ir.IntegerAttr(conversion_cast.attributes["packing"]).value

  output_shape = ir.MemRefType(op.tmem_ref.type).shape
  ncols = output_shape[1] // packing

  with mgpu_utils.when(ctx.single_warp_per_block_predicate):
    tcgen05.tmem_dealloc(tmem_addr, ncols, collective, exact)

  return []


def _swizzle(attrs: Sequence[ir.Attribute]) -> mgpu.SwizzlingMode:
  """Returns the swizzle transform from the given attributes."""
  swizzle = None
  for attr in attrs:
    if mgpu.SwizzleTransformAttr.isinstance(attr):
      if swizzle is not None:
        raise ValueError("Multiple swizzle transforms are not supported.")
      swizzle = mgpu.SwizzleTransformAttr(attr).swizzle
  return swizzle if swizzle is not None else mgpu.SwizzlingMode.kNoSwizzle


def _tmem_ref_from_ir(ref: ir.Value, layout: ir.Attribute) -> tcgen05.TMEMRef:
  """Returns a TMEMRef from an IR value."""
  if not ir.MemRefType.isinstance(ref.type):
    raise ValueError(f"{ref} is not a memref.")
  mem_ref_ty = ir.MemRefType(ref.type)

  if mem_ref_ty.memory_space != mgpu_utils.tmem():
    raise ValueError(
        f"{ref} has a memory space {mem_ref_ty.memory_space} that is not TMEM."
    )

  i32 = ir.IntegerType.get_signless(32)
  _, [tmem_addr] = _undo_conversion_cast(ref, [i32])

  shape = tuple(mem_ref_ty.shape)
  el_ty = mem_ref_ty.element_type
  layout = layouts_lib.from_layout_attr(layout)
  assert isinstance(layout, fa.TiledLayout)
  in_tmem_layout = tcgen05.TMEMLayout(
      layout.tiling, layout.warp_dims, layout.lane_dims, layout.vector_dim
  )
  return tcgen05.TMEMRef(tmem_addr, shape, el_ty, in_tmem_layout)


@_register_lowering(mgpu.TcGen05MMAOp)
def _tcgen05_mma_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.TcGen05MMAOp
) -> Sequence[ir.Value]:
  # TODO(allanrenucci): Add support for `a` in TMEM.
  if op.a.type.memory_space == mgpu_utils.tmem():
    raise NotImplementedError(f"{op.a} is not in TMEM. Only SMEM is supported.")

  a_transforms, b_transforms = inference_utils.in_transforms(op)
  unwrapped_a_ref = unwrap_transformed_memref(op.a, a_transforms)
  unwrapped_b_ref = unwrap_transformed_memref(op.b, b_transforms)

  acc_layout = inference_utils.in_tmem_layouts(op)[0]
  acc_ref = _tmem_ref_from_ir(op.accumulator, acc_layout)

  with mgpu_utils.when(ctx.single_thread_per_block_predicate):
    tcgen05.mma(
        acc_ref,
        unwrapped_a_ref,
        unwrapped_b_ref,
        a_swizzle=_swizzle(a_transforms),
        b_swizzle=_swizzle(b_transforms),
        a_scale=op.a_scale,
        b_scale=op.b_scale,
        accumulate=op.accumulate,
        collective=op.collective.value,
    )

  return []


# TODO(dasenov): Remove this after the minimal jaxlib version is 0.7.1.
if jaxlib.version >= (0, 7, 1):
  @_register_lowering(mgpu.AsyncLoadTmemOp)
  def _async_load_tmem_op_lowering_rule(
      ctx: LoweringContext, op: mgpu.AsyncLoadTmemOp
  ) -> Sequence[ir.Value]:
    """Lowering rule for mgpu.AsyncLoadTmemOp."""
    del ctx

    tmem = _tmem_ref_from_ir(op.source, inference_utils.in_tmem_layouts(op)[0])

    out_layout_attr = inference_utils.out_layouts(op)[0]
    out_layout = layouts_lib.from_tiled_layout_attr(out_layout_attr)
    el_type = ir.MemRefType(op.source.type).element_type
    is_signed = False if ir.IntegerType.isinstance(el_type) else None
    fa = tmem.load(out_layout, is_signed)
    return [fragmented_array_to_ir(fa, op.result.type)]


# TODO(dasenov): Remove this after the minimal jaxlib version is 0.7.1.
if jaxlib.version >= (0, 7, 1):
  @_register_lowering(mgpu.AsyncStoreTmemOp)
  def _async_store_tmem_op_lowering_rule(
      ctx: LoweringContext, op: mgpu.AsyncStoreTmemOp
  ) -> Sequence[ir.Value]:
    """Lowering rule for mgpu.AsyncStoreTmemOp."""
    del ctx

    tmem = _tmem_ref_from_ir(
        op.destination, inference_utils.in_tmem_layouts(op)[0]
    )

    in_layout_attr = inference_utils.in_layouts(op)[0]
    el_type = ir.VectorType(op.source.type).element_type
    is_signed = False if ir.IntegerType.isinstance(el_type) else None
    fa = _fragmented_array_from_ir(op.source, in_layout_attr, is_signed)
    tmem.store(fa)

    return []

def inline_block(
    block: ir.Block, args: Sequence[ir.Value], mapper: dict[ir.Value, ir.Value],
    clone_terminator: bool, terminator_type: type[ir.OpView],
) -> list[ir.Value]:
  """
  Inlines the given block at the current insertion point.

  The block args are replaced with the provided `args`. If the input mapper is
  not empty, it could further be used to replace captured values with an
  alternative.

  If `clone_terminator` is False, the terminator of the block is not cloned. If
  `clone_terminator` is True, the terminator is cloned. This is useful when
  inlining the block into another block. In both cases the operands of the
  terminator are returned as results.
  """
  for arg, val in zip(block.arguments, args, strict=True):
    mapper[arg] =  val
  return_op = None
  for op in block.operations:
    if isinstance(op.opview, terminator_type):
      assert return_op is None
      return_op = op.opview
      if not clone_terminator:
        continue
    # Operands not in the mapper are captured from the context.
    new_operands = [mapper[o] if o in mapper else o for o in op.operands]
    new_attributes = {
        named_attr.name: named_attr.attr
        for named_attr in op.attributes
    }
    new_op = ir.Operation.create(
        name=op.name,
        results=[res.type for res in op.results],
        operands=new_operands,
        attributes=new_attributes,
    )
    for old_result, new_result in zip(op.results, new_op.results):
      mapper[old_result] = new_result

  if return_op is None:
    raise ValueError("A custom return op must terminate the block.")

  inlined_return_values = [mapper[o] for o in return_op.operands]
  return inlined_return_values


@_register_lowering(mgpu.CustomPrimitiveOp)
def _mgpu_custom_primitive_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.CustomPrimitiveOp
) -> Sequence[ir.Value]:
  """Lowering rule for mgpu.CustomPrimitiveOp."""
  # The block already contains unwrapping and wrapping conversion casts.
  return inline_block(
      op.body.blocks[0],
      op.operands,
      mapper={},
      clone_terminator=False,
      terminator_type=mgpu.ReturnOp,
  )


# The metadata needed to recostruct a vector from its flattened representation.
_VectorTemplate = tuple[Sequence[int], fa.FragmentedLayout, ir.VectorType]


def _flatten_ir_values(
    values: Sequence[ir.Value], fa_layouts: Iterable[ir.Attribute]
) -> tuple[Sequence[ir.Value], Sequence[_VectorTemplate | None]]:
  """Flattens a sequence of values.

  Non-vector values are preserved as is. Vectors are mapped to fragmented
  arrays and then flattened into per-register values.

  Args:
    values: The sequence of values to flatten.
    fa_layouts: The layouts of vectors in ``values``.

  Returns:
    A tuple of (flattened values, templates). The templates are used to
    reconstruct the vectors from the per-register  values.
  """
  fa_layouts_it = iter(fa_layouts)
  result: list[ir.Value] = []
  templates: list[_VectorTemplate | None] = []
  for v in values:
    if ir.VectorType.isinstance(v.type):
      fa = _fragmented_array_from_ir(v, next(fa_layouts_it))
      result.extend(fa.registers.flat)
      templates.append((fa.registers.shape, fa.layout, ir.VectorType(v.type)))
    else:
      result.append(v)
      templates.append(None)
  return result, templates


def _unflatten_ir_values(
    flat_values: Sequence[ir.Value], templates: Sequence[_VectorTemplate | None]
) -> Sequence[ir.Value]:
  """The inverse of ``_flatten_ir_values``."""
  result = []
  flat_values_it = iter(flat_values)
  for template in templates:
    if template is None:
      result.append(next(flat_values_it))
      continue
    registers_shape, layout, vec_type = template
    value_registers = np.asarray(
        [next(flat_values_it) for _ in range(math.prod(registers_shape))],
        dtype=object,
    )
    value = fa.FragmentedArray(
        _registers=value_registers.reshape(registers_shape),
        _layout=layout,
        _is_signed=False
        if ir.IntegerType.isinstance(vec_type.element_type)
        else None,
    )
    result.append(fragmented_array_to_ir(value, vec_type))
  return result


def _move_scf_block_to_block_with_flattened_arguments(
    ctx: LoweringContext,
    old_block: ir.Block,
    new_block: ir.Block,
    last_op_type: type[ir.OpView],
    args_template: Sequence[_VectorTemplate | None],
    *new_leading_args: Sequence[ir.Value],
) -> Sequence[_VectorTemplate | None]:
  """Moves the operations from `old_block` to `new_block`.

  The input arguments to the block, if any, are flattened using the provided
  `args_template`, except for any new_leading_args which are simply prepended
  to the flattened arguments and must be part of the template.

  The last operation of the old block must be of type `last_op_type` which
  is expected to be either a `scf.YieldOp` or a `scf.ConditionOp`. This
  operation is recreated with flattened output arguments.
  """
  out_template = None
  with ir.InsertionPoint(new_block):
    new_carry = _unflatten_ir_values(new_block.arguments[len(new_leading_args):], args_template)
    new_args = new_leading_args + tuple(new_carry)
    for old_arg, new_arg in zip(old_block.arguments, new_args, strict=True):
      old_arg.replace_all_uses_with(new_arg)
    for op in [*old_block]:
      if not isinstance(op, last_op_type):
        mgpu.private_operation_remove_from_parent(op)
        mgpu.private_block_append_owned_operation(new_block, op)
        ctx.lower_op(op)
      else:
        assert out_template is None
        layouts = (
            inference_utils.in_layouts(op)
            if inference_utils.has_in_layouts_set(op)
            else []
        )
        if isinstance(op, scf.YieldOp):
          flat_operands, out_template = _flatten_ir_values(op.operands, layouts)
          scf.yield_(flat_operands)
        elif isinstance(op, scf.ConditionOp):
          flat_carry, out_template = _flatten_ir_values(op.args, layouts)
          scf.condition(op.condition, flat_carry)
        else:
          raise NotImplementedError(f"Unsupported op type: {op}")
        op.erase()
  assert out_template is not None
  return out_template

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

  flat_init_args, args_template = _flatten_ir_values(
      for_op.initArgs, in_layouts
  )
  new_for_op = scf.ForOp(
      for_op.lowerBound,
      for_op.upperBound,
      for_op.step,
      flat_init_args,
  )

  _move_scf_block_to_block_with_flattened_arguments(
      ctx,
      for_op.body,
      new_for_op.body,
      scf.YieldOp,
      args_template,
      new_for_op.induction_variable,
  )

  return _unflatten_ir_values(new_for_op.results, args_template)


@_register_lowering(scf.WhileOp)
def _while_op_lowering_rule(
    ctx: LoweringContext, while_op: scf.WhileOp
) -> MlirLoweringRuleResult:
  if not inference_utils.should_have_layout(while_op):
    return _traverse_op_lowering_rule(ctx, while_op)

  before_block = while_op.before.blocks[0]
  after_block = while_op.after.blocks[0]
  condition_op = before_block.operations[len(before_block.operations) - 1]
  yield_op = after_block.operations[len(after_block.operations) - 1]

  in_layouts = inference_utils.in_layouts(while_op)
  out_layouts = inference_utils.out_layouts(while_op)

  if in_layouts:
    yield_layouts = inference_utils.in_layouts(yield_op)
    if in_layouts != yield_layouts:
      raise ValueError(
          f"Input layouts {in_layouts} do not match yield layouts"
          f" {yield_layouts}"
      )

  if out_layouts:
    condition_layouts = inference_utils.in_layouts(condition_op)
    if out_layouts != condition_layouts:
      raise ValueError(
          f"Output layouts {out_layouts} do not match condition layouts"
          f" {condition_layouts}"
      )

  flat_inits, inits_template = _flatten_ir_values(while_op.inits, in_layouts)
  result_types = _infer_flat_result_types(while_op, out_layouts)
  new_while_op = scf.WhileOp(result_types, flat_inits)

  # Before block
  init_types = [v.type for v in flat_inits]
  new_before_block = new_while_op.before.blocks.append(*init_types)
  results_template = _move_scf_block_to_block_with_flattened_arguments(
      ctx,
      before_block,
      new_before_block,
      scf.ConditionOp,
      inits_template,
  )

  # After block
  new_after_block = new_while_op.after.blocks.append(*result_types)
  _move_scf_block_to_block_with_flattened_arguments(
      ctx,
      after_block,
      new_after_block,
      scf.YieldOp,
      results_template,
  )

  return _unflatten_ir_values(new_while_op.results, results_template)


def _infer_flat_result_types(
    op: ir.OpView, out_layouts: Sequence[ir.Attribute]
) -> Sequence[ir.Type]:
  result_types: list[ir.Type] = []
  out_layouts_it = iter(out_layouts)
  for r in op.results:
    if not ir.VectorType.isinstance(r.type):
      result_types.append(r.type)
      continue
    vec_type = ir.VectorType(r.type)
    layout = layouts_lib.from_layout_attr(next(out_layouts_it))
    result_types.extend(
        [layout.registers_element_type(vec_type.element_type)]
        * math.prod(layout.registers_shape(tuple(vec_type.shape)))
    )
  return result_types


@_register_lowering(scf.IfOp)
def _if_op_lowering_rule(
    ctx: LoweringContext, if_op: scf.IfOp
) -> MlirLoweringRuleResult:
  if not inference_utils.should_have_layout(if_op):
    return _traverse_op_lowering_rule(ctx, if_op)

  raise NotImplementedError


@_register_lowering(scf.IndexSwitchOp)
def _index_switch_op_lowering_rule(
    ctx: LoweringContext, switch_op: scf.IndexSwitchOp
) -> MlirLoweringRuleResult:
  if not inference_utils.should_have_layout(switch_op):
    return _traverse_op_lowering_rule(ctx, switch_op)

  out_layouts = inference_utils.out_layouts(switch_op)
  new_switch_op = scf.IndexSwitchOp(
      _infer_flat_result_types(switch_op, out_layouts),
      switch_op.arg,
      switch_op.cases,
      len(switch_op.regions) - 1,
  )

  results_template: Sequence[_VectorTemplate | None] = []
  for region, new_region in zip(
      switch_op.regions, new_switch_op.regions, strict=True
  ):
    [block] = region.blocks
    new_block = new_region.blocks.append()
    results_template = _move_scf_block_to_block_with_flattened_arguments(
        ctx, block, new_block, scf.YieldOp, []
    )
  return _unflatten_ir_values(new_switch_op.results, results_template)


@_register_lowering(func.FuncOp)
@_register_lowering(gpu.LaunchOp)
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


def _should_lower(op: ir.OpView) -> bool:
  """Returns 'true' if the operation should be lowered."""
  return (
      op.OPERATION_NAME.startswith("mosaic_gpu.")  # pytype: disable=attribute-error
      or inference_utils.should_have_layout(op)
      or inference_utils.should_have_transforms(op)
      or any(bool(b) for r in op.regions for b in r)  # Does it have subblocks?
  )


def _gpu_launch_op(module: ir.Module) -> ir.Operation:
  for op in module.body.operations:
    for region in op.operation.regions:
      for block in region.blocks:
        for sub_op in block.operations:
          if sub_op.operation.name == "gpu.launch":
            return sub_op.operation
  raise ValueError("gpu.launch op not found.")


def _lowering_context(
    module: ir.Module,
    launch_context: launch_context.LaunchContext | None,
    auto_barriers: bool,
) -> LoweringContext:
  """Returns a `LoweringContext` for the given `LaunchContext`."""
  # TODO(bchetioui): fix tests to not have a test-only path polluting the API.
  if launch_context is None:  # this case is used in some tests
    return LoweringContext(None, None, None, None, auto_barriers)

  gpu_launch_op = _gpu_launch_op(module)
  with ir.InsertionPoint.at_block_begin(gpu_launch_op.regions[0].blocks[0]):
    block_predicate = utils.single_thread_predicate(
        scope=utils.ThreadSubset.BLOCK
    )
    warpgroup_predicate = utils.single_thread_predicate(
        scope=utils.ThreadSubset.WARPGROUP
    )
    eq = arith.CmpIPredicate.eq
    i32 = ir.IntegerType.get_signless(32)
    warp_predicate = arith.cmpi(eq, utils.warp_idx(sync=False), utils.c(0, i32))
    return LoweringContext(
        launch_context,
        block_predicate,
        warpgroup_predicate,
        warp_predicate,
        auto_barriers,
    )


def lower_mgpu_dialect(
    module: ir.Module,
    launch_context: launch_context.LaunchContext | None,
    auto_barriers: bool = True,
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
  ctx = _lowering_context(module, launch_context, auto_barriers)
  with ir.InsertionPoint(module.body):
    for op in list(module.body):
      ctx.lower_op(op)

  for lowered_op in ctx.lowered_operations:
    lowered_op.erase()
