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

from collections.abc import Callable, Iterable
import dataclasses
import functools
import itertools
import math
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
from jax._src.util import safe_zip
from jax.experimental.mosaic.gpu import layouts as layouts_lib
from jax.experimental.mosaic.gpu import utils as mgpu_utils
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


# TODO(bchetioui): remove once minimum jaxlib >= 0.5.3.
OptimizationBarrierOp = getattr(mgpu, "OptimizationBarrierOp", None)


@_register_lowering(OptimizationBarrierOp)
def _optimization_barrier_op_lowering_rule(
    _: LoweringContext,
    op: OptimizationBarrierOp,
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
      _fragmented_array_to_ir(arr, result.type)
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
    transformed_ref = reinterpret_smem_ref(vector_load_op.base, transforms)
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

  mgpu_utils.warpgroup_barrier()  # Make sure the reads have completed.
  if fragmented_array.layout == fa.WGMMA_LAYOUT:
    swizzle, transforms = swizzle_and_transforms_from_transforms_attr(
        inference_utils.in_transforms(vector_store_op)[0]
    )
    ref_ty = ir.MemRefType(vector_store_op.base.type)
    _check_transforms_and_swizzle_are_supported(ref_ty, transforms, swizzle)
    fragmented_array.store_tiled(
        reinterpret_smem_ref(vector_store_op.base, transforms), swizzle
    )
  elif (fragmented_array.layout == fa.WGMMA_ROW_LAYOUT or
        fragmented_array.layout == fa.WGMMA_COL_LAYOUT or
        isinstance(fragmented_array.layout, fa.WGStridedFragLayout) or
        isinstance(fragmented_array.layout, fa.WGSplatFragLayout)):
    fragmented_array.store_untiled(vector_store_op.base)
  else:
    raise ValueError(
        f"{vector_store_op} has an unsupported layout: {to_store_layout}"
    )
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
      result = a.reduce("add", range(len(a.shape)), scratch)
    case (
        "#vector.kind<maxsi>" | "#vector.kind<maxui>" | "#vector.kind<maximumf>"
    ):
      # TODO(slebedev): Implement this and remove the raise below.
      raise NotImplementedError(f"Unsupported reduction kind: {op.kind}")
    case _:
      raise NotImplementedError(f"Unsupported reduction kind: {op.kind}")
  return [_fragmented_array_to_ir(result, op.result.type)]

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
  return [_fragmented_array_to_ir(result, op.result.type)]


@_register_lowering(mgpu.LayoutCastOp)
def _mgpu_layout_cast_op_lowering_rule(
    _: LoweringContext, layout_cast_op: mgpu.LayoutCastOp
) -> Sequence[ir.Value]:
  return [layout_cast_op.x]


# TODO(dasenov): Remove this after the minimal jaxlib version is 0.6.1.
if hasattr(mgpu, "BroadcastInDimOp"):
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

    if (operand_fa.layout == fa.WGMMA_ROW_LAYOUT and broadcast_dims == [0]):
      out = operand_fa.broadcast_minor(out_ty.shape[1])
    elif (operand_fa.layout == fa.WGMMA_COL_LAYOUT and broadcast_dims == [1]):
      out = operand_fa.broadcast_major(out_ty.shape[0])
    else:
      raise NotImplementedError(
          "Broadcast in dim with non-trivial broadcast dimensions is not"
          f" supported: {op}"
      )
    return [_fragmented_array_to_ir(out, out_ty)]


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


def _transformed_smem_ref_type(
    ref_ty: ir.MemRefType,
    transforms: tuple[launch_context.MemRefTransform, ...],
) -> ir.MemRefType:
  """Returns the transformed ref type for the given logical ref and transforms.
  """
  transposed = _is_memref_transposed(ref_ty)
  if not transforms and not transposed:
    return ref_ty

  if ref_ty.memory_space != ir.Attribute.parse("#gpu.address_space<workgroup>"):
    raise ValueError(f"Only workgroup memory is supported but got {ref_ty}.")

  shape = ref_ty.shape
  if transposed:
    if len(shape) != 2:
      raise NotImplementedError(
          f"Only 2D shapes can be transposed, but got {shape}"
      )
    strides, _ = ref_ty.get_strides_and_offset()
    if strides[0] != 1 or strides[1] != shape[0]:
      raise NotImplementedError(
          f"Only contiguous 2D memrefs can be transposed, but got {ref_ty}"
      )

  for t in transforms:
    shape = list(t.transform_shape(shape))

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
      layout=ir.StridedLayoutAttr.get(0, new_strides),
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
  new_ref_ty = _transformed_smem_ref_type(ref_ty, transforms)
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

  # TODO(dasenov): async_copy requires all GMEM strides except the last one
  # to be a multiple of 16 bytes. This restriction could be loosned with
  # strided layouts when they are contiguous in GMEM. In that case, we could do:
  # flatten -> async_copy -> unflatted here, as long as flattened size is a
  # multiple of 16.

  # TODO(dasenov): Add support for the remaining op properties.
  ctx.launch_context.async_copy(
      src_ref=load_op.source,
      dst_ref=reinterpret_smem_ref(load_op.destination, transforms),
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

  # TODO(dasenov): async_copy requires all GMEM strides except the last one
  # to be a multiple of 16 bytes. This restriction could be loosned with
  # strided layouts when they are contiguous in GMEM. In that case, we could do:
  # flatten -> async_copy -> unflatted here, as long as flattened size is a
  # multiple of 16.

  # TODO(dasenov): Add support for the remaining op properties.
  ctx.launch_context.async_copy(
      src_ref=reinterpret_smem_ref(store_op.source, transforms),
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

  # TODO(dasenov): Move the value -> accumulator conversion outside of wgmma.
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
  b_operand = reinterpret_smem_ref(wgmma_op.b, b_transforms)

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
    a_operand = reinterpret_smem_ref(wgmma_op.a, a_transforms)

  new_acc = wgmma.wgmma(acc, a_operand, b_operand, swizzle=b_swizzle)

  return [
      _fragmented_array_to_ir(
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
  return [_slice_smem(op.result.type, op.offset)]


def _slice_smem(result: ir.Type, offset: ir.Value):
  i8 = ir.IntegerType.get_signless(8)
  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
  smem_base = gpu.dynamic_shared_memory(
      ir.MemRefType.get((utils.DYNAMIC,), i8, memory_space=smem)
  )
  offset = arith.index_cast(ir.IndexType.get(), offset)
  return memref.view(result, smem_base, offset, [])


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
  result = []
  templates = []
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
    result.append(_fragmented_array_to_ir(value, vec_type))
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
              block_predicate = utils.single_thread_predicate(
                  scope=utils.ThreadSubset.BLOCK
              )
              warpgroup_predicate = utils.single_thread_predicate(
                  scope=utils.ThreadSubset.WARPGROUP
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
