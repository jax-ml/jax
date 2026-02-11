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

# mypy has been causing more problems than it solves here. Disable it for these
# files. We have pytype checks anyway.
# mypy: ignore-errors

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import functools
import itertools
import math
import operator
from typing import Any, Protocol, cast

from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib import mosaic_gpu_dialect as mgpu
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import builtin
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import math as mlir_math
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import nvvm
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
import numpy as np

from . import fragmented_array as fa
from . import inference_utils
from . import launch_context as lc
from . import layouts as layouts_lib
from . import tcgen05
from . import utils
from . import wgmma


@dataclasses.dataclass()
class LoweringContext:
  launch_context: lc.LaunchContext | None
  single_thread_per_block_predicate: ir.Value | None
  single_thread_per_warpgroup_predicate: ir.Value | None
  single_warp_per_block_predicate: ir.Value | None
  auto_barriers: bool
  smem_requested_bytes: int
  lowered_operations: set[ir.Operation | ir.OpView] = dataclasses.field(
      default_factory=set
  )
  is_collective_kernel: bool | None = dataclasses.field(
      init=False, default=None
  )

  def check_collective(self, op: ir.OpView) -> None:
    """Checks that the collective attribute is consistent across operations.

    It is an error to mix collective and non-collective operations in the same
    kernel.
    """
    if "collective" not in op.attributes:
      return
    if self.is_collective_kernel is None:
      self.is_collective_kernel = op.attributes["collective"]
    elif self.is_collective_kernel != op.attributes["collective"]:
      raise ValueError(
          "Collective attributes are inconsistent across operations in the"
          " kernel."
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
  conversion_cast = ir_value.owner

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

  conversion_cast.attributes["layout"] = layouts_lib.to_layout_attr(
      fragmented_array.layout
  )

  return conversion_cast.result


def _default_is_signed(dtype: ir.Type) -> bool | None:
  """Returns `False` for Integer types, `None` otherwise.

  When converting from Pallas dtype to IR type, we lose the `is_signed`
  information. We can default to `False` for most use cases.
  """
  return False if isinstance(dtype, ir.IntegerType) else None


def _fragmented_array_from_ir(
    fragmented_array_as_ir: ir.Value,
    layout: ir.Attribute,
    is_signed: bool | None = None,
) -> fa.FragmentedArray:
  producer_layout_attr = fragmented_array_as_ir.owner.attributes["layout"]
  producer_layout = layouts_lib.from_layout_attr(producer_layout_attr)
  vector_ty = ir.VectorType(fragmented_array_as_ir.type)
  reg_shape = producer_layout.registers_shape(tuple(vector_ty.shape))
  reg_ty = producer_layout.registers_element_type(vector_ty.element_type)

  conversion_cast, converted_outputs = _undo_conversion_cast(
      fragmented_array_as_ir, [reg_ty] * math.prod(reg_shape)
  )

  reverse_conversion_cast = converted_outputs[0].owner.opview
  for attribute in conversion_cast.attributes:
    reverse_conversion_cast.attributes[attribute] = conversion_cast.attributes[attribute]

  registers = np.array(list(converted_outputs)).reshape(
    [attr.value for attr in conversion_cast.attributes["registers_shape"]]
  )

  if isinstance(conversion_cast.outputs[0].type.element_type, ir.IntegerType):
    is_signed = False if is_signed is None else is_signed

  return fa.FragmentedArray(
      _registers=registers, _layout=producer_layout, _is_signed=is_signed
  ).to_layout(layouts_lib.from_layout_attr(layout))


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
  transformed_type = transform_type(ref.type, transforms)
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
    op: mgpu.InitializeBarrierOp,
) -> Sequence[ir.Value]:
  i32 = ir.IntegerType.get_signless(32)
  lowered_barrier_type = _lowered_barrier_type()

  for i in range(op.num_barriers.value):
    nvvm.mbarrier_init(
        utils.getelementptr(op.base_pointer, [i], lowered_barrier_type),
        utils.c(
            op.arrival_count.value * utils.WARPGROUP_SIZE,
            i32,
        ),
        predicate=ctx.single_thread_per_block_predicate,
    )

  gpu.barrier()
  return []


@_register_lowering(mgpu.OptimizationBarrierOp)
def _optimization_barrier_op_lowering_rule(
    _: LoweringContext,
    op: mgpu.OptimizationBarrierOp,
) -> Sequence[ir.Value]:
  if not all(
      isinstance(operand.type, ir.VectorType) for operand in op.operands
  ):
    raise NotImplementedError(
        f"Optimization barrier op {op} has non-vector operands."
    )

  fragmented_arrays = []
  for operand, layout in zip(op.operands, inference_utils.in_layouts(op), strict=True):
    fragmented_arrays.append(_fragmented_array_from_ir(operand, layout))

  lowered_fragmented_arrays = fa.optimization_barrier(*fragmented_arrays)
  if isinstance(lowered_fragmented_arrays, fa.FragmentedArray):
    lowered_fragmented_arrays = [lowered_fragmented_arrays]

  return [
      fragmented_array_to_ir(arr, result.type)
      for arr, result in zip(lowered_fragmented_arrays, op.results, strict=True)
  ]


@_register_lowering(arith.ConstantOp)
def _arith_constant_op_lowering_rule(
    _: LoweringContext, op: arith.ConstantOp
) -> Sequence[ir.Value]:
  if not isinstance(op.value, ir.DenseElementsAttr):
    raise NotImplementedError(f"Unsupported constant op: {op}")

  value = ir.DenseElementsAttr(op.value)
  if not value.is_splat:
    raise NotImplementedError(f"Unsupported constant op: {op}")

  ty = ir.VectorType(op.result.type)
  is_signed = _default_is_signed(ty.element_type)

  return [
      fragmented_array_to_ir(
          fa.FragmentedArray.splat(
              arith.constant(ty.element_type, value.get_splat_value()),
              tuple(ty.shape),
              layouts_lib.from_layout_attr(op.attributes["out_layouts"][0]),
              is_signed=is_signed,
          ),
          op.result.type,
      )
  ]


def _check_transforms_and_swizzle_are_supported(
    ref_ty: ir.MemRefType,
    transforms: Sequence[lc.MemRefTransform],
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
          transforms, lambda t: isinstance(t, lc.TileTransform)
      )
  }

  tile_transforms = cast(
      list[lc.TileTransform],
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
    if not isinstance(other_transforms[0], lc.TransposeTransform):
      raise NotImplementedError(
          f"{other_transforms[0]} is not a transpose transform."
      )


class _Transfer(Protocol):
  def __call__(self, optimized: bool) -> Any:
    ...


def _retry_on_failure(transfer: _Transfer, optimized: bool | None) -> Any:
  """If `optimized` is `None`, retry `transfer` with `optimized=False` on failure."""
  if optimized is not None:
    return transfer(optimized)

  # TODO(allanrenucci): Ideally we would have a way to know if we can emit an
  # optimzed transfer. This relies on DCE to delete instructions generated by
  # a failed call to `transfer`.
  try:
    return transfer(optimized=True)
  except ValueError:
    return transfer(optimized=False)


@_register_lowering(mgpu.VectorLoadOp)
def _vector_load_op_lowering_rule(
    _: LoweringContext, op: mgpu.VectorLoadOp
) -> Sequence[ir.Value]:
  (out_layout_attr,) = inference_utils.out_layouts(op)

  element_type = ir.VectorType(op.result.type).element_type
  is_signed = _default_is_signed(element_type)

  def _fragmented_array_to_ir(
      fragmented_array: fa.FragmentedArray,
  ) -> ir.Value:
    return fragmented_array_to_ir(fragmented_array, op.result.type)

  if layouts_lib.is_strided_fragmented_layout(out_layout_attr):
    strided_layout = layouts_lib.from_strided_fragmented_layout_attr(
        out_layout_attr
    )
    # TODO(bchetioui): Process transforms.
    fragmented_array = fa.FragmentedArray.load_strided(
        op.source,
        is_signed=is_signed,
        vec_size=strided_layout.vec_size,
    )
    return [_fragmented_array_to_ir(fragmented_array)]

  if not layouts_lib.is_tiled_layout(out_layout_attr):
    raise ValueError(f"{op} has an unsupported layout: {out_layout_attr}")

  optimized = op.optimized.value if op.optimized is not None else None
  layout = layouts_lib.from_tiled_layout_attr(out_layout_attr)
  ref_ty = ir.MemRefType(op.source.type)
  if ref_ty.memory_space is None:  # GMEM
    fragmented_array = fa.FragmentedArray.load_untiled(
        op.source,
        layout=layout,
        is_signed=is_signed,
        optimized=bool(optimized),
    )
    return [_fragmented_array_to_ir(fragmented_array)]

  if ref_ty.memory_space != utils.smem():
    raise ValueError(f"Unsupported memory space: {ref_ty.memory_space}")

  transforms_attr = inference_utils.in_transforms(op)[0]
  swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
      transforms_attr
  )
  has_transforms = swizzle != mgpu.SwizzlingMode.kNoSwizzle or transforms
  if has_transforms:
    _check_transforms_and_swizzle_are_supported(ref_ty, transforms, swizzle)
    transformed_ref = unwrap_transformed_memref(op.source, transforms_attr)

    def load_tiled(optimized: bool) -> fa.FragmentedArray:
      return fa.FragmentedArray.load_tiled(
          transformed_ref,
          swizzle,
          is_signed=is_signed,
          layout=layout,
          optimized=optimized,
      )

    fragmented_array = _retry_on_failure(load_tiled, optimized)
  else:

    def load_untiled(optimized: bool) -> fa.FragmentedArray:
      return fa.FragmentedArray.load_untiled(
          op.source,
          layout=layout,
          is_signed=is_signed,
          optimized=optimized,
      )

    fragmented_array = _retry_on_failure(load_untiled, optimized)

  return [_fragmented_array_to_ir(fragmented_array)]


@_register_lowering(mgpu.VectorStoreOp)
def _vector_store_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.VectorStoreOp
) -> Sequence[ir.Value]:
  [to_store_layout] = inference_utils.in_layouts(op)
  fragmented_array = _fragmented_array_from_ir(op.valueToStore, to_store_layout)

  if ctx.auto_barriers:
    utils.warpgroup_barrier()  # Make sure the reads have completed.

  ref = op.destination
  ref_type = ir.MemRefType(ref.type)
  optimized = op.optimized.value if op.optimized is not None else None

  if ref_type.memory_space is None:  # GMEM
    fragmented_array.store_untiled(ref, optimized=bool(optimized))
  elif ref_type.memory_space == utils.smem():
    transforms_attr = inference_utils.in_transforms(op)[0]
    swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
        transforms_attr
    )
    has_transforms = swizzle != mgpu.SwizzlingMode.kNoSwizzle or transforms
    if has_transforms:
      _check_transforms_and_swizzle_are_supported(ref_type, transforms, swizzle)
      unwrapped_ref = unwrap_transformed_memref(ref, transforms_attr)

      def store_tiled(optimized: bool):
        fragmented_array.store_tiled(unwrapped_ref, swizzle, optimized)

      _retry_on_failure(store_tiled, optimized)
    else:

      def store_untiled(optimized: bool):
        fragmented_array.store_untiled(ref, optimized=optimized)

      _retry_on_failure(store_untiled, optimized)
  else:
    raise ValueError(f"Unsupported memory space: {ref_type.memory_space}")

  if ctx.auto_barriers:
    utils.warpgroup_barrier()  # Make sure the writes have completed.

  return []


@_register_lowering(mgpu.DebugPrintOp)
def _debug_print_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.DebugPrintOp
) -> Sequence[ir.Value]:
  del ctx
  [layout] = inference_utils.in_layouts(op)
  a = _fragmented_array_from_ir(op.value, layout)
  a.debug_print(op.format.value)
  return []


def pprint_layout(v: fa.FragmentedArray | tcgen05.TMEMRef) -> str:
  if isinstance(v, fa.FragmentedArray):
    match v.layout:
      case fa.WGMMA_LAYOUT:
        return "WGMMA"
      case fa.WGMMA_ROW_LAYOUT:
        return "WGMMA_ROW"
      case fa.WGMMA_TRANSPOSED_LAYOUT:
        return "WGMMA_TRANSPOSED"
      case fa.TCGEN05_LAYOUT:
        return "TCGEN05"
      case fa.TCGEN05_TRANSPOSED_LAYOUT:
        return "TCGEN05_TRANSPOSED"
      case fa.TMEM_NATIVE_LAYOUT:
        return "TCGEN05_TMEM_NATIVE"
      case _:
        return str(v.layout)
  else:
    assert isinstance(v, tcgen05.TMEMRef), v
    if v.layout == tcgen05.tmem_default_layout(packing=v.packing):
      return f"TMEM_DEFAULT(packing={v.packing})"
    return str(v.layout)


@_register_lowering(mgpu.PrintLayoutOp)
def _print_layout_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.PrintLayoutOp
) -> Sequence[ir.Value]:
  del ctx
  if isinstance(op.value.type, ir.VectorType):
    (layout,) = inference_utils.in_layouts(op)
    a = _fragmented_array_from_ir(op.value, layout)
    print(op.format.value.format(pprint_layout(a)))
  else:
    (layout,) = inference_utils.in_tmem_layouts(op)
    ref = _tmem_ref_from_ir(op.value, layout)
    print(op.format.value.format(pprint_layout(ref)))
  return []


@_register_lowering(mgpu.BroadcastedIotaOp)
def _broadcasted_iota_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.BroadcastedIotaOp
) -> Sequence[ir.Value]:
  del ctx
  [layout] = inference_utils.out_layouts(op)
  result_type = ir.VectorType(op.result.type)
  a = fa.FragmentedArray.broadcasted_iota(
      result_type.element_type,
      tuple(result_type.shape),
      op.dimension.value,
      layouts_lib.from_layout_attr(layout),
      is_signed=_default_is_signed(result_type.element_type),
  )
  return [fragmented_array_to_ir(a, result_type)]


@_register_lowering(vector.BroadcastOp)
def _vector_broadcast_op_lowering_rule(
    _: LoweringContext, op: vector.BroadcastOp
) -> Sequence[ir.Value]:
  out_vec_ty = ir.VectorType(op.vector.type)
  fragmented_array = fa.FragmentedArray.splat(
      op.source,
      tuple(out_vec_ty.shape),
      layouts_lib.from_layout_attr(
          op.attributes["out_layouts"][0]
      ),
      is_signed=_default_is_signed(out_vec_ty.element_type),
  )
  return [fragmented_array_to_ir(fragmented_array, out_vec_ty)]


@_register_lowering(vector.ShapeCastOp)
def _vector_shape_cast_op_lowering_rule(
    _: LoweringContext, op: vector.ShapeCastOp
) -> Sequence[ir.Value]:
  [layout] = inference_utils.in_layouts(op)
  out_vec_ty = ir.VectorType(op.result.type)
  assert out_vec_ty.has_static_shape
  a = _fragmented_array_from_ir(op.source, layout)
  return [
      fragmented_array_to_ir(a.reshape(tuple(out_vec_ty.shape)), out_vec_ty)
  ]


@_register_lowering(vector.ExtractStridedSliceOp)
def _vector_extract_strided_slice_op_lowering_rule(
    ctx: LoweringContext, op: vector.ExtractStridedSliceOp
) -> Sequence[ir.Value]:
  del ctx
  if any(ir.IntegerAttr(s).value != 1 for s in op.strides):
    raise NotImplementedError("`strides` must contain only 1s.")
  [in_layout] = inference_utils.in_layouts(op)
  [out_layout] = inference_utils.out_layouts(op)
  assert in_layout == out_layout
  out_vec_ty = ir.VectorType(op.result.type)
  assert out_vec_ty.has_static_shape
  a = _fragmented_array_from_ir(op.source, in_layout)
  indices = tuple(
      utils.DynamicSlice(
          ir.IntegerAttr(offset).value, ir.IntegerAttr(length).value
      )
      for offset, length in zip(op.offsets, op.sizes, strict=True)
  )
  result = a[indices]
  assert result.layout == layouts_lib.from_layout_attr(out_layout)
  return [fragmented_array_to_ir(result, out_vec_ty)]


@_register_lowering(vector.ExtractOp)
def _vector_extract_op_lowering_rule(
    ctx: LoweringContext, op: vector.ExtractOp
) -> Sequence[ir.Value]:
  del ctx
  if op.dynamic_position:
    raise NotImplementedError("Only slicing with static indices allowed.")

  [in_layout] = inference_utils.in_layouts(op)
  a = _fragmented_array_from_ir(op.source, in_layout)

  if not isinstance(op.result.type, ir.VectorType):  # scalar result
    result = a[tuple(op.static_position)]
    assert isinstance(result.layout, fa.WGSplatFragLayout)
    return [result.registers.item()]

  [out_layout] = inference_utils.out_layouts(op)
  assert in_layout == out_layout
  a = _fragmented_array_from_ir(op.source, in_layout)
  result_type = ir.VectorType(op.result.type)
  slices = tuple(slice(i, i + 1) for i in op.static_position)
  # TODO(allanrenucci): Add direct support for indexing to FragmentedArray.
  result = a[slices].reshape(tuple(result_type.shape))
  assert result.layout == layouts_lib.from_layout_attr(out_layout)
  return [fragmented_array_to_ir(result, result_type)]


def _combining_kind(attr: ir.Attribute) -> vector.CombiningKind:
  return vector.CombiningKind[
      str(attr).removeprefix("#vector.kind<").removesuffix(">").upper()
  ]


def _is_reduction_signed(kind: vector.CombiningKind) -> bool | None:
  if kind in (vector.CombiningKind.MAXSI, vector.CombiningKind.MINSI):
    return True
  if kind in (vector.CombiningKind.MAXUI, vector.CombiningKind.MINUI):
    return False
  return None


@_register_lowering(vector.ReductionOp)
def _vector_reduction_op_lowering_rule(
    ctx: LoweringContext, op: vector.ReductionOp
) -> Sequence[ir.Value]:
  [layout] = inference_utils.in_layouts(op)
  element_type = op.vector.type.element_type
  scratch = _slice_smem(
      ir.MemRefType.get([4], element_type, memory_space=utils.smem()),
      arith.constant(None, op.attributes["offset"]),
      ctx.smem_requested_bytes,
  )
  axes = range(op.vector.type.rank)
  op_kind = _combining_kind(op.kind)
  is_signed = _is_reduction_signed(op_kind)
  a = _fragmented_array_from_ir(op.vector, layout, is_signed)
  match op_kind:
    case vector.CombiningKind.ADD:
      result = a.reduce("add", axes, scratch)
    case vector.CombiningKind.MAXSI | vector.CombiningKind.MAXUI | vector.CombiningKind.MAXIMUMF:
      result = a.reduce("max", axes, scratch)
    case vector.CombiningKind.MINUI | vector.CombiningKind.MINSI | vector.CombiningKind.MINIMUMF:
      result = a.reduce("min", axes, scratch)
    case _:
      raise NotImplementedError(f"Unsupported reduction kind: {op.kind}")
  assert isinstance(result.layout, fa.WGSplatFragLayout)
  return [result.registers.item()]


@_register_lowering(vector.MultiDimReductionOp)
def _vector_multi_dim_reduction_op_lowering_rule(
    ctx: LoweringContext, op: vector.MultiDimReductionOp
) -> Sequence[ir.Value]:
  [in_layout, acc_layout] = inference_utils.in_layouts(op)
  [out_layout] = inference_utils.out_layouts(op)
  if out_layout != acc_layout:
    raise ValueError(
        f"Output layout {out_layout} must match the accumulator layout"
        f" {acc_layout}"
    )

  if len(op.reduction_dims) != 1:
    raise NotImplementedError("Only 1 reduction dimension is supported.")

  op_kind = _combining_kind(op.kind)
  is_signed = _is_reduction_signed(op_kind)
  src = _fragmented_array_from_ir(op.source, in_layout, is_signed)
  acc = _fragmented_array_from_ir(op.acc, acc_layout, is_signed)

  if not isinstance(src.layout, fa.TiledLayout):
    raise NotImplementedError(f"Unsupported layout: {src.layout}")
  reduced_dim = src.layout.tiling.tile_dimension(op.reduction_dims[0])
  if any(reduced_dim[d] for d in src.layout.partitioned_warp_dims):
    # cross-warp reductions require scratch space.
    dtype = op.source.type.element_type
    allocation_size = ir.IntegerAttr(op.attributes["scratch_size"]).value * 8 // utils.bitwidth(dtype)
    scratch = _slice_smem(
        ir.MemRefType.get([allocation_size], dtype, memory_space=utils.smem()),
        arith.constant(None, op.attributes["offset"]),
        ctx.smem_requested_bytes,
    )
  else:
    scratch = None

  match op_kind:
    case vector.CombiningKind.ADD:
      result = src.reduce("add", op.reduction_dims[0], scratch)
      result += acc
    case vector.CombiningKind.MAXSI | vector.CombiningKind.MAXUI | vector.CombiningKind.MAXIMUMF:
      result = src.reduce("max", op.reduction_dims[0], scratch)
      result = result.max(acc)
    case vector.CombiningKind.MINUI | vector.CombiningKind.MINSI | vector.CombiningKind.MINIMUMF:
      result = src.reduce("min", op.reduction_dims[0], scratch)
      result = result.min(acc)
    case _:
      raise NotImplementedError(f"Unsupported reduction kind: {op.kind}")
  assert result.layout == layouts_lib.from_layout_attr(out_layout)  # pytype: disable=attribute-error
  return [fragmented_array_to_ir(result, op.result.type)]


@_register_lowering(mgpu.LayoutCastOp)
def _mgpu_layout_cast_op_lowering_rule(
    _: LoweringContext, op: mgpu.LayoutCastOp
) -> Sequence[ir.Value]:
  [in_layout] = inference_utils.in_layouts(op)
  [out_layout] = inference_utils.out_layouts(op)
  in_array = _fragmented_array_from_ir(op.x, in_layout)
  out_array = in_array.to_layout(layouts_lib.from_layout_attr(out_layout))
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

  broadcast_dims = tuple(op.broadcast_dimensions)
  in_layout_attr = inference_utils.in_layouts(op)[0]
  operand_fa = _fragmented_array_from_ir(op.operand, in_layout_attr)
  out_layout = layouts_lib.from_layout_attr(inference_utils.out_layouts(op)[0])
  out = operand_fa.broadcast_in_dim(out_ty.shape, broadcast_dims, out_layout)
  return [fragmented_array_to_ir(out, out_ty)]


def swizzle_and_transforms_from_transforms_attr(
    transforms: ir.ArrayAttr,
) -> tuple[mgpu.SwizzlingMode, tuple[lc.MemRefTransform, ...]]:
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
  gmem_transforms: list[lc.MemRefTransform] = []

  for transform in transforms:
    if swizzle is not None:
      raise ValueError(f"{transforms} contain more transforms after swizzle.")
    if mgpu.SwizzleTransformAttr.isinstance(transform):
      # TODO(dasenov): Swizzling can change if the ref is sliced in certain
      # ways. We might want to enforce some restrictions here.
      swizzle = mgpu.SwizzleTransformAttr(transform).swizzle
    elif mgpu.TileTransformAttr.isinstance(transform):
      tiling = mgpu.TileTransformAttr(transform).tiling
      tiling_transform = lc.TileTransform(tuple(tiling))
      gmem_transforms.append(tiling_transform)
    elif mgpu.TransposeTransformAttr.isinstance(transform):
      permutation = mgpu.TransposeTransformAttr(transform).permutation
      transpose_transform = lc.TransposeTransform(
          tuple(permutation)
      )
      gmem_transforms.append(transpose_transform)
    else:
      raise ValueError("Unknown transform: {transform}")

  return swizzle or mgpu.SwizzlingMode.kNoSwizzle, tuple(gmem_transforms)


def tile_offset(
    offsets: tuple[int, ...], tiling: tuple[int, ...]
) -> tuple[int, ...]:
  """Tiles the trailing offsets in `offsets` according to `tiling`.

  Raises if the offsets are not aligned with the start of a tile.
  """
  if len(offsets) < len(tiling):
    raise ValueError(f"Offsets {offsets} have lower rank than tiling {tiling}")
  untiled_offsets, tiled_offsets = (
      offsets[: -len(tiling)],
      offsets[-len(tiling) :],
  )
  for i, t in zip(tiled_offsets, tiling, strict=True):
    if i % t != 0:
      raise ValueError(f"Offset {i} is not divisible by tile size {t}")
  return (
      *untiled_offsets,
      *[i // t for i, t in zip(tiled_offsets, tiling, strict=True)],
      *[0] * len(tiling),
  )


def tile_strides(
    strides: tuple[int, ...], tiling: tuple[int, ...]
) -> tuple[int, ...]:
  """Tiles the trailing strides in `strides` according to `tiling`.

  The `len(tiling)` trailing strides in `strides` must be the `len(tiling)`
  smallest strides in `strides`. The same property holds in the result, i.e.,
  given two tiles with indices i and j (i < j) with strides tiled according to
  this function, then all the elements in tile i are physically ordered before
  all the elements in tile j.

  E.g., tile_strides((2048, 32, 1), (8, 4)) = (2048, 256, 32, 4, 1)
  """
  if len(strides) < len(tiling):
    raise ValueError(f"Strides {strides} have lower rank than tiling {tiling}")
  ordered_strides = sorted(strides, reverse=True)
  if set(ordered_strides[-len(tiling):]) != set(strides[-len(tiling):]):
    raise ValueError(
        "Can not tile strides when tiled dimensions have been transposed with "
        f"untiled dimensions. Strides: {strides}, tiling: {tiling}"
    )
  tiled_ordered_strides = ordered_strides[-len(tiling):]
  untiled_strides, tiled_strides = strides[:-len(tiling)], strides[-len(tiling):]

  to_ordered = lambda i: tiled_ordered_strides.index(tiled_strides[i])
  from_ordered = lambda i: tiled_strides.index(tiled_ordered_strides[i])

  ordered_tiling = [tiling[from_ordered(i)] for i in range(len(tiling))]
  ordered_tiled_strides = [tiled_strides[from_ordered(i)] for i in range(len(tiling))]

  ordered_tiled_tiling_strides = [1]
  for t in reversed(ordered_tiling):
    ordered_tiled_tiling_strides.append(ordered_tiled_tiling_strides[-1] * t)

  prev_s = ordered_tiled_strides[-1]
  for s, t in zip(ordered_tiled_strides[:-1][::-1], ordered_tiling[1:][::-1], strict=True):
    d = prev_s * t
    prev_s = s
    if s % d != 0:
      raise ValueError(
          f"Stride {s} is not divisible by {d} (tile size = {t}). "
          f"Strides: {strides}, tiling: {tiling}"
      )
    ordered_tiled_tiling_strides.append(s // d * ordered_tiled_tiling_strides[-1])

  ordered_tiled_tiling_strides.reverse()

  return (
      *untiled_strides,
      *[ordered_tiled_tiling_strides[to_ordered(i)] for i in range(len(tiling))],
      *[ordered_tiled_tiling_strides[len(tiling) + to_ordered(i)] for i in range(len(tiling))]
  )


def transform_type(
    ref_ty: ir.MemRefType,
    transforms: tuple[lc.MemRefTransform, ...],
) -> ir.MemRefType:
  if not utils.is_smem_ref(ref_ty):
    raise ValueError(f"Only workgroup memory is supported but got {ref_ty}.")

  if not transforms:
    return ref_ty

  # TODO(bchetioui): this should be trivial to relax if ever necessary.
  if len(transforms) > 1 or not isinstance(transforms[0], lc.TileTransform):
    raise NotImplementedError(f"Unsupported transforms: {transforms}")
  tile_transform: lc.TileTransform = transforms[0]  # pytype: disable=attribute-error

  strides, offset = ref_ty.get_strides_and_offset()
  tiled_shape = tile_transform.transform_shape(ref_ty.shape)
  tiled_strides = tile_strides(strides, tile_transform.tiling)

  if offset == ir.ShapedType.get_dynamic_stride_or_offset():
    tiled_offset = offset
  else:
    delinearized_offset = [0] * len(strides)
    for i, stride in sorted(enumerate(strides), key=lambda es: es[1], reverse=True):
      delinearized_offset[i] = offset // stride
      offset %= stride
    tiled_delinearized_offset = tile_offset(
        tuple(delinearized_offset), tile_transform.tiling
    )
    tiled_offset = sum(o * s for o, s in zip(tiled_delinearized_offset, tiled_strides, strict=True))

  if isinstance(ref_ty.layout, ir.StridedLayoutAttr):
    layout = ir.StridedLayoutAttr.get(tiled_offset, tiled_strides)
  else:
    layout = None

  return ir.MemRefType.get(
      tiled_shape,
      ref_ty.element_type,
      memory_space=ref_ty.memory_space,
      layout=layout
  )


def _gmem_slice_and_predicate(
    ctx: LoweringContext,
    op: mgpu.AsyncLoadOp | mgpu.AsyncPrefetchOp | mgpu.AsyncStoreOp,
) -> tuple[
    tuple[ir.Value | fa.FragmentedArray | utils.DynamicSlice, ...],
    dict[str, ir.Value],
]:
  """Returns the GMEM slice and predicate for the given async op."""
  gmem_slice = []
  predicate = dict(predicate=ctx.single_thread_per_warpgroup_predicate)
  for idx, size in zip(op.indices, op.slice_lengths, strict=True):
    if isinstance(idx.type, ir.IntegerType):
      idx_int = arith.index_cast(ir.IndexType.get(), idx)
      v = idx_int if size < 0 else utils.DynamicSlice(idx_int, size)
      gmem_slice.append(v)
    elif isinstance(idx.type, ir.VectorType):
      layout = inference_utils.in_layouts(op)[0]
      assert layouts_lib.from_layout_attr(layout) == fa.TMA_GATHER_INDICES_LAYOUT
      idx_fa = _fragmented_array_from_ir(idx, layout)
      gmem_slice.append(idx_fa)
      predicate = dict()
    else:
      raise TypeError(f"Unsupported index type: {idx.type}")
  return tuple(gmem_slice), predicate


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

  unwrapped_dst = unwrap_transformed_memref(
      load_op.destination, transforms_attr
  )

  if utils.is_memref_transposed(unwrapped_dst.type):
    strides, _ = ir.MemRefType(unwrapped_dst.type).get_strides_and_offset()
    permutation = tuple(
        sorted(range(len(strides)), key=lambda i: strides[i], reverse=True)
    )
    # We undo the tranpose and apply it as a transform.
    unwrapped_dst = utils.memref_transpose(
        unwrapped_dst, permutation
    )
    if transforms:
      raise NotImplementedError("Can't transpose transformed refs.")
    transforms = (lc.TransposeTransform(permutation),)

  gmem_slice, predicate = _gmem_slice_and_predicate(ctx, load_op)

  collective = [
      gpu.Dimension(ir.IntegerAttr(axis).value)
      for axis in load_op.collective or []
  ]

  # TODO(dasenov): async_copy requires all GMEM strides except the last one
  # to be a multiple of 16 bytes. This restriction could be loosned with
  # strided layouts when they are contiguous in GMEM. In that case, we could do:
  # flatten -> async_copy -> unflatted here, as long as flattened size is a
  # multiple of 16.

  # TODO(dasenov): Add support for the remaining op properties.
  if ctx.auto_barriers:
    utils.warpgroup_barrier()  # Make sure the writes have completed.
  ctx.launch_context.async_copy(
      src_ref=load_op.source,
      dst_ref=unwrapped_dst,
      gmem_slice=gmem_slice,
      barrier=barrier.barrier_ref,
      collective=collective,
      arrive=False,
      swizzle=swizzle,
      gmem_transform=transforms,
      **predicate,
  )
  return []


@_register_lowering(mgpu.AsyncPrefetchOp)
def _mgpu_async_prefetch_op_lowering_rule(
    ctx: LoweringContext, load_op: mgpu.AsyncPrefetchOp
) -> Sequence[ir.Value]:
  assert ctx.launch_context is not None

  gmem_slice, predicate = _gmem_slice_and_predicate(ctx, load_op)

  if load_op.collective:
    raise NotImplementedError("Collective prefetches are not supported yet.")

  ctx.launch_context.async_prefetch(
      gmem_ref=load_op.source,
      gmem_slice=gmem_slice,
      swizzle=None,
      gmem_transform=(),
      **predicate,
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
  if utils.is_memref_transposed(unwrapped_source.type):
    strides, _ = ir.MemRefType(unwrapped_source.type).get_strides_and_offset()
    permutation = tuple(
        sorted(range(len(strides)), key=lambda i: strides[i], reverse=True)
    )
    # We undo the tranpose and apply it as a transform.
    unwrapped_source = utils.memref_transpose(
        unwrapped_source, permutation
    )
    if transforms:
      raise NotImplementedError("Can't transpose transformed refs.")
    transforms = (lc.TransposeTransform(permutation),)

  gmem_slice, predicate = _gmem_slice_and_predicate(ctx, store_op)

  # TODO(dasenov): async_copy requires all GMEM strides except the last one
  # to be a multiple of 16 bytes. This restriction could be loosned with
  # strided layouts when they are contiguous in GMEM. In that case, we could do:
  # flatten -> async_copy -> unflatted here, as long as flattened size is a
  # multiple of 16.
  if store_op.reduction_op is not None:
    reduction_op = mgpu.TMAReduction(store_op.reduction_op.value).name.lower()
  else:
    reduction_op = None

  # TODO(dasenov): Add support for the remaining op properties.
  ctx.launch_context.async_copy(
      src_ref=unwrapped_source,
      dst_ref=store_op.destination,
      gmem_slice=gmem_slice,
      swizzle=swizzle,
      gmem_transform=transforms,
      **predicate,
      arrive=store_op.commit_group,
      reduction_op=reduction_op
  )
  return []


@_register_lowering(mgpu.TmemLayoutCastOp)
def _tmem_layout_cast_lowering_rule(
    ctx: LoweringContext,
    op: mgpu.TmemLayoutCastOp,
) -> Sequence[ir.Value]:
  del ctx
  in_layout = inference_utils.in_tmem_layouts(op)[0]
  tmem_ref = _tmem_ref_from_ir(op.ref, in_layout)
  # We can't relayout TMEM.
  assert layouts_lib.to_layout_attr(tmem_ref.layout) == op.new_layout
  return [op.ref]


@_register_lowering(mgpu.SliceTmemOp)
def _slice_tmem_lowering_rule(
    ctx: LoweringContext, op: mgpu.SliceTmemOp
) -> Sequence[ir.Value]:
  del ctx
  in_layout_attr = inference_utils.in_tmem_layouts(op)[0]
  out_layout_attr = inference_utils.out_tmem_layouts(op)[0]
  source = _tmem_ref_from_ir(op.source, in_layout_attr)
  i32 = ir.IntegerType.get_signless(32)
  offset = arith.constant(i32, op.offset)
  dest_addr = arith.addi(source.address, offset)
  conversion_cast = builtin.UnrealizedConversionCastOp([op.result.type], [dest_addr])
  conversion_cast.attributes["layout"] = out_layout_attr
  return [conversion_cast.result]


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


for _op, _source_is_signed, _target_is_signed in [
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
  _lowerings[_op.OPERATION_NAME] = functools.partial(
      _conversion_op_lowering_rule,
      source_is_signed=_source_is_signed,
      target_is_signed=_target_is_signed,
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
    if op.fastmath == ir.Attribute.parse("#arith.fastmath<afn>"):
      result_fa = impl(a, approx=True)
    else:
      result_fa = impl(a)
  else:
    result_fa = impl(a)

  return [fragmented_array_to_ir(result_fa, op.result.type)]


for _op, _unary_impl, _is_signed in [
    (mlir_math.RsqrtOp, fa.FragmentedArray.rsqrt, None),
    (mlir_math.ExpOp, fa.FragmentedArray.exp, None),
    (mlir_math.Exp2Op, fa.FragmentedArray.exp2, None),
    (mlir_math.SinOp, fa.FragmentedArray.sin, None),
    (mlir_math.CosOp, fa.FragmentedArray.cos, None),
    (mlir_math.LogOp, fa.FragmentedArray.log, None),
    (mlir_math.TanhOp, fa.FragmentedArray.tanh, None),
    (mlir_math.AbsFOp, fa.FragmentedArray.abs, None),
    (mlir_math.AbsIOp, fa.FragmentedArray.abs, True),
    (mlir_math.RoundOp, fa.FragmentedArray.round, None),
    (mlir_math.RoundEvenOp, fa.FragmentedArray.round_even, None),
    (mlir_math.ErfOp, fa.FragmentedArray.erf, None),
]:
  _lowerings[_op.OPERATION_NAME] = functools.partial(
      _unary_op_lowering_rule, impl=_unary_impl, is_signed=_is_signed
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


for _op, _binary_impl, _is_signed in [
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
    (mlir_math.Atan2Op, fa.FragmentedArray.atan2, None),
    (mlir_math.CopySignOp, fa.FragmentedArray.copysign, None),
]:
  _lowerings[_op.OPERATION_NAME] = functools.partial(
      _binary_op_lowering_rule, impl=_binary_impl, is_signed=_is_signed
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
  impl, is_signed = CMPI_IMPLS[op.predicate.value]  # pytype: disable=attribute-error
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
  impl = CMPF_IMPLS[op.predicate.value]  # pytype: disable=attribute-error
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
      output_is_signed=_default_is_signed(out_element_type),
  )
  return [fragmented_array_to_ir(out, op.result.type)]


@_register_lowering(arith.SelectOp)
def _select_op_lowering_rule(
    ctx: LoweringContext, op: arith.SelectOp
) -> Sequence[ir.Value]:
  del ctx
  in_layouts = inference_utils.in_layouts(op)
  [layout] = inference_utils.out_layouts(op)
  if any(in_layout != layout for in_layout in in_layouts):
    raise ValueError("Layout mismatch")
  pred = _fragmented_array_from_ir(op.condition, layout)
  true_value = _fragmented_array_from_ir(op.true_value, layout)
  false_value = _fragmented_array_from_ir(op.false_value, layout)
  result = pred.select(true_value, false_value)
  return [fragmented_array_to_ir(result, op.result.type)]


@_register_lowering(mgpu.WGMMAOp)
def _mgpu_wgmma_op_lowering_rule(
    _: LoweringContext, wgmma_op: mgpu.WGMMAOp
) -> Sequence[ir.Value]:
  in_layouts = inference_utils.in_layouts(wgmma_op)
  assert in_layouts[0] == layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)
  [out_layout] = inference_utils.out_layouts(wgmma_op)
  assert out_layout == layouts_lib.to_layout_attr(fa.WGMMA_LAYOUT)

  # s8/i8 WGMMA expects signed integer accumulator.
  element_type = wgmma_op.a.type.element_type
  is_signed = True if isinstance(element_type, ir.IntegerType) else None
  # TODO(dasenov): Move the value -> accumulator conversion outside of wgmma.
  # The associated fence could be a little expensive and is not needed if the
  # result a wgmma feeds into another wgmma (even in another loop step).
  regs = _fragmented_array_from_ir(
      wgmma_op.accumulator, in_layouts[0], is_signed
  )
  acc = wgmma.WGMMAAccumulator.from_registers(regs)

  if isinstance(wgmma_op.a.type, ir.VectorType):
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

  if isinstance(wgmma_op.a.type, ir.VectorType):
    expected_a_layout = (
        fa.WGMMA_LAYOUT_8BIT
        if utils.bitwidth(element_type) == 8
        else fa.WGMMA_LAYOUT
    )
    assert in_layouts[1] == layouts_lib.to_layout_attr(expected_a_layout)
    a_operand = _fragmented_array_from_ir(wgmma_op.a, in_layouts[1], is_signed)
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


@_register_lowering(mgpu.ArriveOp)
def _mgpu_arrive_op_lowering_rule(
    ctx: LoweringContext, arrive_op: mgpu.ArriveOp
) -> Sequence[ir.Value]:
  barrier = utils.DialectBarrierRef.from_barrier_memref(arrive_op.barrier)
  orders_tc = arrive_op.orders_tensor_core.value
  if orders_tc:
    # Only one thread arrives, so make sure it ups the arrival count for the
    # whole warpgroup.
    #
    # TODO(b/415721295): At the moment we assume that there is a single arrival
    # per warpgroup. If we need to support also Warp-level semantics we will
    # need to use a warp-level predicate.
    predicate = ctx.single_thread_per_warpgroup_predicate
    arrival_count = utils.WARPGROUP_SIZE
  else:
    # Each thread arrives once.
    arrival_count = 1
    predicate = None

  barrier.barrier_ref.arrive(
      arrival_count=arrival_count,
      orders_tensor_core=orders_tc,
      predicate=predicate,
  )
  return []


@_register_lowering(mgpu.ArriveExpectTxOp)
def _mgpu_arrive_expect_tx_op_lowering_rule(
    _: LoweringContext, arrive_expect_tx_op: mgpu.ArriveExpectTxOp
) -> Sequence[ir.Value]:
  num_bytes = arrive_expect_tx_op.expect_tx.value
  if num_bytes % utils.WARPGROUP_SIZE:
    raise NotImplementedError(
        "Only copies of a multiple of 128 bytes are supported"
    )
  # We arrive uniformly from each thread in the WG, so we need to divide the
  # number of bytes by the number of threads in the WG.
  # TODO(dasenov): Relax this. We can just select the WG leader and have it
  # arrive with the whole transfer size, while everyone else arrives with 0.
  # But we should continue using this scheme as it's likely to be faster.
  num_bytes //= utils.WARPGROUP_SIZE
  num_bytes = utils.c(num_bytes, ir.IntegerType.get_signless(32))

  barrier = utils.DialectBarrierRef.from_barrier_memref(
      arrive_expect_tx_op.barrier
  )
  utils.nvvm_mbarrier_arrive_expect_tx(barrier.get_ptr(), num_bytes)

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
  ref_ty = ir.MemRefType(op.result.type)
  if ref_ty.element_type == ir.Type.parse("!mosaic_gpu.barrier"):
    # Barrier memrefs are not transformed and must not be wrapped.
    assert not inference_utils.has_out_transforms_set(op)
    return [_slice_smem(ref_ty, op.offset, ctx.smem_requested_bytes)]

  [out_transforms] = inference_utils.out_transforms(op)
  _, transforms = swizzle_and_transforms_from_transforms_attr(out_transforms)
  transformed_ref_ty = transform_type(ref_ty, transforms)
  transformed_ref = _slice_smem(transformed_ref_ty, op.offset, ctx.smem_requested_bytes)
  return [wrap_transformed_memref(transformed_ref, op.result.type, out_transforms)]


def _slice_smem(result: ir.MemRefType, offset: ir.Value, smem_size: int):
  if isinstance(offset.owner, arith.ConstantOp):
    cst_offset = ir.IntegerAttr(offset.owner.value).value
    size = math.prod(result.shape) * utils.bitwidth(result.element_type) // 8
    if cst_offset + size > smem_size:
      raise ValueError("Ran out of shared memory.")

  i8 = ir.IntegerType.get_signless(8)
  smem_base = gpu.dynamic_shared_memory(
      ir.MemRefType.get((utils.DYNAMIC,), i8, memory_space=utils.smem())
  )
  offset = arith.index_cast(ir.IndexType.get(), offset)
  lowered_result_type = result
  if result.element_type == ir.Type.parse("!mosaic_gpu.barrier"):
    lowered_result_type = ir.MemRefType.get(
        result.shape, _lowered_barrier_type(), memory_space=utils.smem()
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

  if any(s != 1 for s in op.static_strides):
    raise NotImplementedError("SubViewOp only supports static strides of 1.")
  if op.sizes:
    raise NotImplementedError("SubViewOp only supports static sizes.")
  src_ty = ir.MemRefType(op.source.type)

  if utils.is_memref_transposed(src_ty):
    raise NotImplementedError("SubViewOp does not support transposed memrefs.")

  if utils.is_tmem_ref(src_ty):
    [in_tmem_layout] = inference_utils.in_tmem_layouts(op)
    [out_tmem_layout] = inference_utils.out_tmem_layouts(op)
    assert in_tmem_layout == out_tmem_layout
    ref = _tmem_ref_from_ir(op.source, in_tmem_layout)
    indices = []
    dynamic_offset_index = 0
    for offset, size in zip(op.static_offsets, op.static_sizes, strict=True):
      if ir.ShapedType.is_dynamic_size(offset):
        offset = op.offsets[dynamic_offset_index]
        dynamic_offset_index += 1
      indices.append(utils.DynamicSlice(offset, size))
    return [_tmem_ref_to_ir(ref.slice(*indices))]

  in_transforms = inference_utils.in_transforms(op)[0]
  out_transforms = inference_utils.out_transforms(op)[0]

  if in_transforms != out_transforms:
    raise NotImplementedError(
        "SubViewOp transforms for the input and output refs must be identical."
    )

  unwrapped_source_ref = unwrap_transformed_memref(op.source, in_transforms)
  swizzle, transforms = swizzle_and_transforms_from_transforms_attr(out_transforms)
  if swizzle != mgpu.SwizzlingMode.kNoSwizzle:
    swizzle_elems = swizzle * 8 // utils.bitwidth(src_ty.element_type)
    source_strides, _ = src_ty.get_strides_and_offset()
    for stride, offset, size in zip(
        source_strides, op.static_offsets, op.static_sizes, strict=True
    ):
      if stride != 1:
        continue
      # A dimension with stride 1 is a minor dimension and is swizzled.
      if size % swizzle_elems != 0:
        raise ValueError(
            f"Swizzled dimension of {size=} is not a multiple of"
            f" {swizzle_elems=}."
        )
      # TODO(allanrenucci): Support dynamic offsets that are divisible by
      # `swizzle_elems`. E.g. using `utils.is_known_divisible`.
      if ir.ShapedType.is_dynamic_size(offset):
        raise NotImplementedError(
            "Slicing a swizzled dynamic dimension is not supported."
        )
      if offset % swizzle_elems != 0:
        raise ValueError(
            f"subview {offset=} is not a multiple of {swizzle_elems=}."
        )

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
    case (tile_transform, ) if isinstance(tile_transform, lc.TileTransform):
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
      # TODO(bchetioui): support transposed offsets.
      new_static_offsets, new_dynamic_offsets = _tile_transform_offsets(
          tiling, list(op.static_offsets), list(op.offsets)
      )

      new_subview_op = memref.SubViewOp(
          transform_type(ir.MemRefType(op.result.type), transforms),
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
  if in_transformed_ty.rank == op.in_.type.rank:
    new_permutation = op.permutation
  elif in_transformed_ty.rank == 4:
    if op.permutation == _permutation_to_affine_map_attr([0, 1]):
      new_permutation = _permutation_to_affine_map_attr([0, 1, 2, 3])
    elif op.permutation == _permutation_to_affine_map_attr([1, 0]):
      new_permutation = _permutation_to_affine_map_attr([1, 0, 3, 2])
    else:
      raise NotImplementedError(f"Unsupported permutation={op.permutation}.")
  else:
    raise NotImplementedError(
        "TransposeOp only supports transposing 4D tiled memrefs and untiled"
        " memrefs."
    )

  out_transforms = inference_utils.out_transforms(op)[0]
  _, transforms = swizzle_and_transforms_from_transforms_attr(out_transforms)
  new_transpose_op = memref.TransposeOp(
      transform_type(ir.MemRefType(op.result.type), transforms),
      unwrapped_in_ref,
      new_permutation,
  )

  wrapped_ref = wrap_transformed_memref(
      new_transpose_op.result, op.result.type, out_transforms
  )
  return [wrapped_ref]


@_register_lowering(memref.ExpandShapeOp)
def _memref_expand_shape_op_lowering_rule(
    ctx: LoweringContext, op: memref.ExpandShapeOp
) -> Sequence[ir.Value]:
  del ctx

  in_transforms = inference_utils.in_transforms(op)[0]
  unwrapped_in_ref = unwrap_transformed_memref(op.src, in_transforms)
  in_transformed_ty = ir.MemRefType(unwrapped_in_ref.type)

  out_transforms = inference_utils.out_transforms(op)[0]
  _, transforms = swizzle_and_transforms_from_transforms_attr(out_transforms)
  out_transformed_ty = transform_type(ir.MemRefType(op.result.type), transforms)

  reassociation = list(op.reassociation)
  num_tiling_dims = len(in_transformed_ty.shape) - len(op.src.type.shape)

  # We don't currently allow expanding tiled dimensions. So to compute the
  # reassociation on the lowered types, we just need to backfill the original
  # one with the number of missing dimensions.
  if num_tiling_dims > 0 and any(
      len(x) > 1 for x in reassociation[-num_tiling_dims:]
  ):
    # If we ever remove this restriction, we will need to ensure this is
    # compatible with `transform_type`.
    raise NotImplementedError("Expanding tiled dimensions is not supported.")

  start_index = len(op.static_output_shape)
  for i in range(start_index, start_index + num_tiling_dims):
    reassociation.append([i])

  new_expand_shape_op = memref.ExpandShapeOp(
      out_transformed_ty,
      unwrapped_in_ref,
      reassociation,
      output_shape=op.output_shape,
      static_output_shape=out_transformed_ty.shape,
  )

  wrapped_ref = wrap_transformed_memref(
      new_expand_shape_op.result, op.result.type, out_transforms
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
  ctx.check_collective(op)

  output_shape = ir.MemRefType(op.result.type).shape
  ncols = output_shape[1] // op.packing.value

  with utils.when(ctx.single_warp_per_block_predicate):
    tcgen05.tmem_alloc(op.smem_ptr, ncols, op.collective, exact=False)
  gpu.barrier()
  tmem_addr = memref.load(op.smem_ptr, [])

  cast_op = builtin.UnrealizedConversionCastOp(
      [op.result.type], [tmem_addr]
  )
  cast_op.attributes["collective"] = op.collective
  cast_op.attributes["packing"] = op.packing
  cast_op.attributes["layout"] = inference_utils.out_tmem_layouts(op)[0]

  return [cast_op.result]


@_register_lowering(mgpu.TmemRelinquishAllocPermitOp)
def _tmem_relinquish_alloc_permit_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.TmemRelinquishAllocPermitOp
) -> Sequence[ir.Value]:
  """Lowering rule for mgpu.TmemRelinquishAllocPermitOp."""
  ctx.check_collective(op)
  with utils.when(ctx.single_warp_per_block_predicate):
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
  packing = ir.IntegerAttr(conversion_cast.attributes["packing"]).value

  output_shape = ir.MemRefType(op.tmem_ref.type).shape
  ncols = output_shape[1] // packing

  with utils.when(ctx.single_warp_per_block_predicate):
    tcgen05.tmem_dealloc(tmem_addr, ncols, collective, exact=False)

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


def _tmem_ref_from_ir(
    ref: ir.Value, expected_layout: ir.Attribute
) -> tcgen05.TMEMRef:
  """Returns a TMEMRef from an IR value.

  Throws an error if the annotated layout does not match the expected layout.
  """
  if not isinstance(ref.type, ir.MemRefType):
    raise ValueError(f"{ref} is not a memref.")
  mem_ref_ty = ir.MemRefType(ref.type)

  if mem_ref_ty.memory_space != utils.tmem():
    raise ValueError(
        f"{ref} has a memory space {mem_ref_ty.memory_space} that is not TMEM."
    )

  i32 = ir.IntegerType.get_signless(32)
  conversion_cast, [tmem_addr] = _undo_conversion_cast(ref, [i32])

  shape = tuple(mem_ref_ty.shape)
  el_ty = mem_ref_ty.element_type
  layout_attr = conversion_cast.attributes["layout"]
  if layout_attr != expected_layout:
    raise ValueError(
        f"{ref} has a layout {layout_attr} that does not match the expected"
        f" layout {expected_layout}."
    )
  layout = layouts_lib.from_layout_attr(layout_attr)
  assert isinstance(layout, fa.TiledLayout)
  tmem_layout = tcgen05.TMEMLayout(
      layout.tiling, layout.warp_dims, layout.lane_dims, layout.vector_dim
  )
  return tcgen05.TMEMRef(tmem_addr, shape, el_ty, tmem_layout)


def _tmem_ref_to_ir(ref: tcgen05.TMEMRef) -> ir.Value:
  """Returns an IR value from a TMEMRef."""
  ty = ir.MemRefType.get(ref.shape, ref.dtype, memory_space=utils.tmem())
  conversion_cast = builtin.UnrealizedConversionCastOp([ty], [ref.address])
  conversion_cast.attributes["layout"] = layouts_lib.to_layout_attr(ref.layout)
  return conversion_cast.result


@_register_lowering(mgpu.TcGen05MMAOp)
def _tcgen05_mma_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.TcGen05MMAOp
) -> Sequence[ir.Value]:
  ctx.check_collective(op)

  in_tmem_layouts = inference_utils.in_tmem_layouts(op)
  acc_layout = in_tmem_layouts[0]
  acc_ref = _tmem_ref_from_ir(op.accumulator, acc_layout)

  if utils.is_smem_ref(op.a):
    a_transforms, b_transforms = inference_utils.in_transforms(op)
    a_swizzle = _swizzle(a_transforms)
    b_swizzle = _swizzle(b_transforms)
    a_ref = unwrap_transformed_memref(op.a, a_transforms)
    b_ref = unwrap_transformed_memref(op.b, b_transforms)
  else:
    a_ref = _tmem_ref_from_ir(op.a, in_tmem_layouts[1])
    [b_transforms] = inference_utils.in_transforms(op)
    b_swizzle = _swizzle(b_transforms)
    a_swizzle = b_swizzle
    b_ref = unwrap_transformed_memref(op.b, b_transforms)

  with utils.when(ctx.single_thread_per_block_predicate):
    tcgen05.mma(
        acc_ref,
        a_ref,
        b_ref,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        a_scale=op.a_scale,
        b_scale=op.b_scale,
        accumulate=op.accumulate,
        collective=op.collective.value,
    )

  return []


@_register_lowering(mgpu.AsyncLoadTmemOp)
def _async_load_tmem_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.AsyncLoadTmemOp
) -> Sequence[ir.Value]:
  """Lowering rule for mgpu.AsyncLoadTmemOp."""
  del ctx
  in_layout_attr = inference_utils.in_tmem_layouts(op)[0]
  tmem_ref = _tmem_ref_from_ir(op.source, in_layout_attr)
  out_layout_attr = inference_utils.out_layouts(op)[0]
  out_layout = layouts_lib.from_tiled_layout_attr(out_layout_attr)
  is_signed = _default_is_signed(ir.MemRefType(op.source.type).element_type)
  arr = tmem_ref.load(out_layout, is_signed)
  return [fragmented_array_to_ir(arr, op.result.type)]


@_register_lowering(mgpu.AsyncStoreTmemOp)
def _async_store_tmem_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.AsyncStoreTmemOp
) -> Sequence[ir.Value]:
  """Lowering rule for mgpu.AsyncStoreTmemOp."""
  del ctx
  in_layout_attr = inference_utils.in_tmem_layouts(op)[0]
  tmem_ref = _tmem_ref_from_ir(op.destination, in_layout_attr)
  in_layout_attr = inference_utils.in_layouts(op)[0]
  arr = _fragmented_array_from_ir(op.source, in_layout_attr)
  tmem_ref.store(arr)

  return []


@_register_lowering(mgpu.CustomPrimitiveOp)
def _mgpu_custom_primitive_op_lowering_rule(
    ctx: LoweringContext, op: mgpu.CustomPrimitiveOp
) -> Sequence[ir.Value]:
  """Lowering rule for mgpu.CustomPrimitiveOp."""
  del ctx
  block = op.body.blocks[0]
  for arg, op in zip(block.arguments, op.operands, strict=True):
    arg.replace_all_uses_with(op)

  return_op = None
  ip = ir.InsertionPoint.current
  for op in block.operations:
    if isinstance(op.opview, mgpu.ReturnOp):
      assert return_op is None
      return_op = op.opview
      continue
    op.detach_from_parent()
    ip.insert(op)

  if return_op is None:
    raise ValueError("A custom return op must terminate the block.")

  return return_op.operands


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
    if isinstance(v.type, ir.VectorType):
      arr = _fragmented_array_from_ir(v, next(fa_layouts_it))
      result.extend(arr.registers.flat)
      templates.append((arr.registers.shape, arr.layout, ir.VectorType(v.type)))
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
        _is_signed=_default_is_signed(vec_type.element_type),
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
        # `append` moves the operation.
        new_block.append(op)
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

  in_layouts = (
      inference_utils.in_layouts(while_op)
      if inference_utils.should_have_in_layout(while_op)
      else []
  )
  out_layouts = (
      inference_utils.out_layouts(while_op)
      if inference_utils.should_have_out_layout(while_op)
      else []
  )

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
    if not isinstance(r.type, ir.VectorType):
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
  )

  results_template: Sequence[_VectorTemplate | None] = []
  for region, new_region in zip(
      switch_op.regions, new_switch_op.regions, strict=True
  ):
    [block] = region.blocks
    new_block = new_region.blocks[0]
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
      or inference_utils.should_have_tmem_layout(op)
      # Does it have subblocks?
      or any(bool(b) for r in op.regions for b in r)  # pylint: disable=g-complex-comprehension
  )


def _gpu_launch_op(module: ir.Module) -> gpu.LaunchOp:
  for op in module.body.operations:
    for region in op.operation.regions:
      for block in region.blocks:
        for sub_op in block.operations:
          if isinstance(sub_op, gpu.LaunchOp):
            return sub_op
  raise ValueError("gpu.launch op not found.")


def _lowering_context(
    module: ir.Module,
    launch_context: lc.LaunchContext | None,
    auto_barriers: bool,
) -> LoweringContext:
  """Returns a `LoweringContext` for the given `LaunchContext`."""
  # TODO(bchetioui): fix tests to not have a test-only path polluting the API.
  if launch_context is None:  # this case is used in some tests
    return LoweringContext(None, None, None, None, auto_barriers, 10**9)

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
    smem_size = gpu_launch_op.dynamicSharedMemorySize
    assert isinstance(smem_size.owner, arith.ConstantOp)
    smem_size = ir.IntegerAttr(smem_size.owner.value).value
    return LoweringContext(
        launch_context,
        block_predicate,
        warpgroup_predicate,
        warp_predicate,
        auto_barriers,
        smem_size,
    )


def lower_mgpu_dialect(
    module: ir.Module,
    launch_context: lc.LaunchContext | None,
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
