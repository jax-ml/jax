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
"""Lowering for Pallas TPU SparseCore."""

from collections.abc import Sequence
import functools
from typing import Any, NoReturn, cast

from jax._src import core as jax_core
from jax._src import debugging
from jax._src import lax
from jax._src import numpy as jnp
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import vector
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering as tc_lowering
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax._src.pallas.mosaic import sc_core
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax.experimental.mosaic.dialects import tpu


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


MemorySpace = tpu_core.MemorySpace
CoreMemorySpace = pallas_core.CoreMemorySpace

ShapedAbstractValue = tc_lowering.ShapedAbstractValue

LoweringContext = tc_lowering.LoweringContext
LoweringRuleContext = tc_lowering.LoweringRuleContext
MosaicGridMapping = tc_lowering.MosaicGridMapping

_dtype_to_ir_type = tc_lowering._dtype_to_ir_type
_transform_ref = tc_lowering._transform_ref
_dma_unflatten = tpu_primitives._dma_unflatten
_get_ref_and_transforms = tpu_primitives._get_ref_and_transforms


register_lowering_rule = functools.partial(
    tc_lowering.register_lowering_rule,
    kernel_types=(
        tpu_core.CoreType.SC_SCALAR_SUBCORE,
        tpu_core.CoreType.SC_VECTOR_SUBCORE,
    ),
)


@register_lowering_rule(state_primitives.get_p)
def _get_lowering_rule(ctx: LoweringRuleContext, ref, *flat_transforms, tree):
  return _load_lowering_rule(ctx, ref, None, *flat_transforms, tree=tree)


def _load_lowering_rule(
    ctx: LoweringRuleContext, ref, mask, *flat_transforms, tree
):
  ref_aval, *_flat_index_avals = ctx.avals_in
  assert isinstance(ref_aval, state.AbstractRef)
  [out_aval] = ctx.avals_out
  assert isinstance(out_aval, jax_core.ShapedArray)

  ref_memory_space = tpu_core.memory_space_to_tpu_memory_space(
      ref_aval.memory_space, ctx.lowering_context.kernel_type
  )
  if (
      ref_memory_space is MemorySpace.HBM
      or ref_memory_space is MemorySpace.VMEM_SHARED
  ):
    raise NotImplementedError(
        f"Get does not support loading from {ref_memory_space!r}."
        " Copy the data to a core-local memory space, e.g. VMEM,"
        " via `pltpu.async_copy`."
    )

  transforms = list(tree_util.tree_unflatten(tree, flat_transforms))
  if not transforms or not isinstance(transforms[-1], indexing.NDIndexer):
    tref_aval = state.transform_type(transforms, ref_aval)
    assert isinstance(tref_aval, state.AbstractRef)
    transforms.append(indexing.NDIndexer.make_trivial_indexer(tref_aval.shape))
  *prev_transforms, indexer = transforms
  ref_block_shape, *_ = ctx.block_shapes
  ref, ref_block_shape = _transform_ref(
      ref, ref_aval, ref_block_shape, prev_transforms
  )
  starts, sizes, strides, squeeze_dims, _ = tc_lowering._indexer_to_start_size_stride(
      indexer, ref_block_shape, cast_to_index=True
  )
  for first_nontrivial_dim, s in enumerate(sizes):
    if s != 1:
      break
  else:
    first_nontrivial_dim = len(sizes)
  if any(squeeze_dims[first_nontrivial_dim:]):
    raise NotImplementedError(
        "Integer indexing of refs that follows a non-trivial slice is not"
        " supported on SC"
    )
  if not all(s == 1 for s in strides):
    raise NotImplementedError(
        "Get only supports slices with stride 1, got {strides}"
    )

  if (out_aval.ndim == 0) != (ref_memory_space is MemorySpace.SMEM):
    message = "Get only supports loading scalars from SMEM."
    if ref_memory_space is MemorySpace.SMEM:
      message += " Trying to load an array of shape {out_aval.shape}."
    elif ref_memory_space is MemorySpace.VMEM:
      message += (
          " To load a scalar from VMEM, load an array first and then extract a"
          " particular element, e.g. ``v = ref[pl.ds(idx, ...)]; v[0]``."
      )
    else:
      message += f" Trying to load a scalar from {ref_memory_space!r}."
    raise NotImplementedError(message)
  if out_aval.ndim == 0:
    if mask is not None:
      raise NotImplementedError("Get does not support masked scalar loads")
    return memref.load(ref, starts)

  if not ctx.lowering_context.needs_layout_passes:
    _check_aval_is_supported("Get", out_aval)
  out_vec_type = ir.VectorType.get(
      out_aval.shape, _dtype_to_ir_type(out_aval.dtype)
  )
  if not ctx.lowering_context.needs_layout_passes:
    return tpu.vector_load(
        out_vec_type, ref, indices=starts, strides=[], mask=mask
    )
  # Load at the full memref rank, keeping integer-indexed dims as size 1,
  # because apply-vector-layout requires the vector rank to match the memref.
  memref_vec_shape = cast(
      Sequence[int],
      [1 if squeeze else s for s, squeeze in zip(sizes, squeeze_dims)],
  )
  memref_vec_type = ir.VectorType.get(
      memref_vec_shape, _dtype_to_ir_type(out_aval.dtype)
  )
  load_val = tpu.vector_load(
      memref_vec_type, ref, indices=starts, strides=[], mask=mask
  )
  return vector.shape_cast(out_vec_type, load_val)


@register_lowering_rule(state_primitives.swap_p)
def _swap_lowering_rule(
    ctx: LoweringRuleContext, ref, val, *flat_transforms, tree
):
  return _store_lowering_rule(
      ctx, ref, val, None, *flat_transforms, tree=tree, add=False
  )


@register_lowering_rule(state_primitives.addupdate_p)
def _addupdate_lowering_rule(
    ctx: LoweringRuleContext, ref, val, *flat_transforms, tree
):
  ctx = ctx.replace(avals_out=[ctx.avals_in[1]])
  _store_lowering_rule(
      ctx, ref, val, None, *flat_transforms, tree=tree, add=True
  )
  return ()


def _store_lowering_rule(
    ctx: LoweringRuleContext, ref, val, mask, *flat_transforms, tree, add
):
  ref_aval, _, *_flat_index_avals = ctx.avals_in
  assert isinstance(ref_aval, state.AbstractRef)
  [out_aval] = ctx.avals_out
  assert isinstance(out_aval, jax_core.ShapedArray)

  ref_memory_space = tpu_core.memory_space_to_tpu_memory_space(
      ref_aval.memory_space, ctx.lowering_context.kernel_type
  )
  if (
      ref_memory_space is MemorySpace.HBM
      or ref_memory_space is MemorySpace.VMEM_SHARED
  ):
    raise NotImplementedError(
        f"Swap does not support storing to {ref_memory_space!r}."
        " Copy the data to a core-local memory space, e.g. VMEM,"
        " via `pltpu.async_copy`."
    )

  transforms = list(tree_util.tree_unflatten(tree, flat_transforms))
  if not transforms or not isinstance(transforms[-1], indexing.NDIndexer):
    tref_aval = state.transform_type(transforms, ref_aval)
    assert isinstance(tref_aval, state.AbstractRef)
    transforms.append(indexing.NDIndexer.make_trivial_indexer(tref_aval.shape))
  *prev_transforms, indexer = transforms
  ref_block_shape, *_ = ctx.block_shapes
  ref, ref_block_shape = _transform_ref(
      ref, ref_aval, ref_block_shape, prev_transforms
  )
  starts, sizes, strides, squeeze_dims, _ = tc_lowering._indexer_to_start_size_stride(
      indexer, ref_block_shape, cast_to_index=True
  )
  for first_nontrivial_dim, s in enumerate(sizes):
    if s != 1:
      break
  else:
    first_nontrivial_dim = len(sizes)
  if any(squeeze_dims[first_nontrivial_dim:]):
    raise NotImplementedError(
        "Integer indexing of refs that follows a non-trivial slice is not"
        " supported on SC"
    )
  if not all(s == 1 for s in strides):
    raise NotImplementedError(
        "Swap only supports slices with stride 1, got {strides}"
    )

  if (out_aval.ndim == 0) != (ref_memory_space is MemorySpace.SMEM):
    message = "Swap only supports scalars in SMEM."
    if ref_memory_space is MemorySpace.SMEM:
      message += " Trying to swap an array of shape {out_aval.shape}."
    else:
      message += f" Trying to swap a scalar in {ref_memory_space!r}."
    raise NotImplementedError(message)

  if out_aval.ndim == 0:
    if mask is not None:
      raise NotImplementedError("Swap does not support masked scalar stores")
    if add:
      # TODO(slebedev): We can use memref.atomic_rmw here, but the SC compiler
      # doesn't support it yet.
      raise NotImplementedError("Swap does not support atomic scalar adds")
    old_val = memref.load(ref, starts)
    memref.store(val, ref, starts)
    return old_val

  if not ctx.lowering_context.needs_layout_passes:
    _check_aval_is_supported("Swap", out_aval)
  out_vec_type = ir.VectorType.get(
      out_aval.shape, _dtype_to_ir_type(out_aval.dtype)
  )
  if not ctx.lowering_context.needs_layout_passes:
    old_val = tpu.vector_load(out_vec_type, ref, starts, strides=[], mask=mask)
    _ = tpu.vector_store(
        val, ref, indices=starts, strides=[], mask=mask, add=add
    )
    return old_val
  # Load and store at the full memref rank, keeping integer-indexed dims as
  # size 1, because apply-vector-layout requires the vector rank to match
  # the memref.
  memref_vec_shape = cast(
      Sequence[int],
      [1 if squeeze else s for s, squeeze in zip(sizes, squeeze_dims)],
  )
  memref_vec_type = ir.VectorType.get(
      memref_vec_shape, _dtype_to_ir_type(out_aval.dtype)
  )
  old_val = tpu.vector_load(memref_vec_type, ref, starts, strides=[], mask=mask)
  old_val = vector.shape_cast(out_vec_type, old_val)
  val_memref_rank = vector.shape_cast(memref_vec_type, val)
  tpu.vector_store(val_memref_rank, ref, starts, strides=[], mask=mask, add=add)
  return old_val


@register_lowering_rule(lax.iota_p,
                        kernel_types=[tpu_core.CoreType.SC_VECTOR_SUBCORE])
def _iota_lowering_rule_sc(ctx: LoweringRuleContext, dtype, shape, dimension,
                           sharding):
  sc_info = sc_core.get_sparse_core_info()
  if shape != (sc_info.num_lanes,):
    raise ValueError(
        f"Unsupported iota shape for SC vector subcore. Got {shape}, supported "
        f"shape is {(sc_info.num_lanes,)}."
    )
  [out_aval] = ctx.avals_out
  out_type = ir.VectorType.get(
      [sc_info.num_lanes], _dtype_to_ir_type(out_aval.dtype)
  )
  return tpu.iota(out_type, dimensions=[dimension])


def _check_aval_is_supported(caller: str, aval: jax_core.ShapedArray) -> None:
  supported_shapes = sc_core.supported_shapes(aval.dtype)
  if aval.shape in supported_shapes:
    return
  if not supported_shapes:
    raise NotImplementedError(f"{caller} does not support {aval.dtype} arrays")
  else:
    raise NotImplementedError(
        f"{caller} only supports {aval.dtype} arrays of shapes"
        f" [{', '.join(map(repr, supported_shapes))}], got {aval.shape}"
    )


@register_lowering_rule(debugging.debug_print_p)
def _debug_print_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    fmt: str,
    ordered,
    partitioned,
    in_tree,
    static_args,
    np_printoptions,
    has_placeholders,
    logging_record,
):
  del partitioned, np_printoptions, in_tree, static_args
  def fail(reason: str) -> NoReturn:
    raise NotImplementedError(
        f"pl.debug_print() {reason} when lowering to SparseCore"
    )

  if ordered:
    fail("does not support ordered print")
  if has_placeholders:
    fail("does not support placeholders")

  match args:
    case []:
      tpu.log(inputs=[], tag=fmt)
    case [arg] if isinstance(arg.type, ir.MemRefType):
      tpu.log_buffer(arg, ctx.avals_in[0].shape, fmt)
    case [arg]:
      tpu.log(inputs=[arg], tag=fmt)
    case _:
      fail("does not support multiple inputs")
  return []


def _prepare_dma_refs(
    src_ref,
    dst_ref,
    src_aval,
    dst_aval,
    core_type: tpu_core.CoreType,
    is_add: bool = False,
):
  """Prepares the DMA source and destination references."""
  src_ref_orig, dst_ref_orig = src_ref, dst_ref
  src_memory_space = tpu_core.memory_space_to_tpu_memory_space(
      src_aval.memory_space, core_type
  )
  dst_memory_space = tpu_core.memory_space_to_tpu_memory_space(
      dst_aval.memory_space, core_type
  )
  src_ref, src_transforms = _get_ref_and_transforms(src_ref)
  dst_ref, dst_transforms = _get_ref_and_transforms(dst_ref)
  src_aval, src_transforms_aval = _get_ref_and_transforms(src_aval)
  dst_aval, dst_transforms_aval = _get_ref_and_transforms(dst_aval)
  match src_memory_space, dst_memory_space:
    case MemorySpace.HBM | MemorySpace.VMEM_SHARED, MemorySpace.VMEM:
      if _has_indirect_offsets(dst_transforms, dst_transforms_aval, core_type):
        raise ValueError(
            "Only the source ref can be indexed when doing a gather via"
            " `pltpu.async_copy`"
        )
      dst_ref, _ = _transform_ref(
          dst_ref, dst_aval, dst_aval.shape, dst_transforms
      )
      dst_ref_shape = tuple(ir.MemRefType(dst_ref.type).shape)
      indirect_offsets, src_transforms = _extract_indirect_offsets(
          src_transforms, dst_ref_shape, src_transforms_aval, core_type
      )
      src_ref, _ = _transform_ref(
          src_ref, src_aval, src_aval.shape, src_transforms
      )
      indirect_offsets_ref_str = "src_ref"
    case MemorySpace.VMEM, MemorySpace.HBM | MemorySpace.VMEM_SHARED:
      if _has_indirect_offsets(src_transforms, src_transforms_aval, core_type):
        raise ValueError(
            "Only the destination ref can be indexed when doing a scatter via"
            " `pltpu.async_copy`"
        )
      src_ref, _ = _transform_ref(
          src_ref, src_aval, src_aval.shape, src_transforms
      )
      src_ref_shape = tuple(ir.MemRefType(src_ref.type).shape)
      indirect_offsets, dst_transforms = _extract_indirect_offsets(
          dst_transforms, src_ref_shape, dst_transforms_aval, core_type
      )
      dst_ref, _ = _transform_ref(
          dst_ref, dst_aval, dst_aval.shape, dst_transforms
      )
      indirect_offsets_ref_str = "dst_ref"
    case _:  # Indirect DMA is not supported.
      if (
          # fmt: off
          _has_indirect_offsets(src_transforms, src_transforms_aval, core_type) or
          _has_indirect_offsets(dst_transforms, dst_transforms_aval, core_type)
          # fmt: on
      ):
        raise NotImplementedError(
            "Scatter/gather via `pltpu.async_copy` from"
            f" {src_memory_space!r} to {dst_memory_space!r} is not"
            " supported"
        )
      if is_add:
        raise ValueError(
            "DMAs with `add=True` are only supported between VMEM and "
            f"HBM/VMEM_SHARED."
            f"Got (src, dst)={(src_aval.memory_space, dst_aval.memory_space)}"
        )
      indirect_offsets = None
      indirect_offsets_ref_str = ""
  if is_add and indirect_offsets is None:
    raise NotImplementedError(
        "DMAs with `add=True` must (for now) specify offsets of the"
        " majormost dimension. You can do this by writing"
        " `pltpu.async_copy(..., {ref}={ref}.at[jnp.arange(vec_dim)], ...)`"
        " or `pltpu.async_copy(..., {ref}={ref}.at[indices_ref],"
        " ...)`.".format(ref=indirect_offsets_ref_str)
    )
  if indirect_offsets is None:
    # If typical DMA path, don't alter the refs.
    return src_ref_orig, dst_ref_orig, None
  return src_ref, dst_ref, indirect_offsets


# TODO(slebedev): Use the TC rule once we align the ``LoweringRuleContext``
# with the TC lowering.
@register_lowering_rule(tpu_primitives.dma_start_p)
def _dma_start_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    tree,
    device_id_type: pallas_primitives.DeviceIdType,
    priority: int,
    add: bool,
):
  src_ref, dst_ref, sem, src_sem, device_id = _dma_unflatten(
      tree, args
  )
  src_aval, dst_aval, sem_aval, src_sem_aval, device_id_aval = _dma_unflatten(
      tree, ctx.avals_in
  )

  src_ref, dst_ref, indirect_offsets = _prepare_dma_refs(
      src_ref,
      dst_ref,
      src_aval,
      dst_aval,
      ctx.lowering_context.kernel_type,
      add,
  )
  if add and indirect_offsets is None:
    # TODO: Support regular DMA with add=True.
    raise NotImplementedError(
        "DMAs with `add=True` must (for now) specify offsets of the majormost "
        "dimension. You can do this by writing "
        "`pltpu.async_copy(..., dst_ref=ref.at[jnp.arange(vec_dim)], ...)` or "
        "`pltpu.async_copy(..., dst_ref=ref.at[iota_ref], ...)`."
    )
  core_index = None
  subcore_index = None
  if device_id is not None:
    if isinstance(sem_aval.memory_space, pallas_core.CoreMemorySpace):
      dest_mesh = sem_aval.memory_space.mesh
    else:
      dest_mesh = None
    device_id, core_index, subcore_index = tc_lowering._device_id_to_logical(
        ctx, device_id, device_id_type, device_id_aval, dest_mesh
    )

  # If not ``None``, we lower to an indirect DMA instead.
  if indirect_offsets is None:
    def _dma_start(src_ref, dst_ref, sem, src_sem):
      tpu.enqueue_dma(
          source=src_ref,
          target=dst_ref,
          target_semaphore=sem,
          source_semaphore=src_sem,
          device_id=device_id,
          priority=priority,
          core_id=core_index,
          subcore_id=subcore_index,  # pyrefly: ignore[unexpected-keyword]
      )
      return []

    return tc_lowering.lower_with_transformed_refs(
        _dma_start,
        [src_ref, dst_ref, sem, src_sem],
        [src_aval, dst_aval, sem_aval, src_sem_aval],
    )

  if device_id is not None:
    raise NotImplementedError(
        "Scatter/gather to or from a remote device via `pltpu.async_copy` is"
        " not supported"
    )

  offset_filter = None
  if indirect_offsets.ignored_value is not None:
    offset_filter = tc_lowering._ensure_mlir_value(
        indirect_offsets.ignored_value, jax_core.ShapedArray((), jnp.int32)
    )

  sem_aval, _ = _get_ref_and_transforms(sem_aval)
  sem, _ = _transform_ref(sem, sem_aval, sem_aval.shape)
  tpu.enqueue_indirect_dma(
      src_ref,
      dst_ref,
      indirect_offsets.values,
      sem,
      add=add,
      offset_filter=offset_filter,
  )
  return []


# TODO(slebedev): Use the TC rule once we align the ``LoweringRuleContext``
# with the TC lowering.
@register_lowering_rule(tpu_primitives.dma_wait_p)
def _dma_wait_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    tree,
    device_id_type: pallas_primitives.DeviceIdType,
    insert_dummy_device: bool,
):
  src_ref, dst_ref, sem, _, device_id = _dma_unflatten(
      tree, args
  )
  src_aval, dst_aval, sem_aval, _, device_id_aval = _dma_unflatten(
      tree, ctx.avals_in
  )

  src_ref, dst_ref, indirect_offsets = _prepare_dma_refs(
      src_ref,
      dst_ref,
      src_aval,
      dst_aval,
      ctx.lowering_context.kernel_type,
  )
  core_id = None
  subcore_id = None
  if insert_dummy_device:
    i32 = ir.IntegerType.get_signless(32)
    core_id = device_id = arith.constant(i32, ir.IntegerAttr.get(i32, 0))
  elif device_id is not None:
    if isinstance(sem_aval.memory_space, pallas_core.CoreMemorySpace):
      dest_mesh = sem_aval.memory_space.mesh
    else:
      dest_mesh = None
    device_id, core_id, subcore_id = tc_lowering._device_id_to_logical(
        ctx, device_id, device_id_type, device_id_aval, dest_mesh
    )
    if core_id:
      raise NotImplementedError(
          "Core index must be None when waiting on a local DMA."
      )
    if subcore_id:
      raise NotImplementedError(
          "Subcore index must be None when waiting on a local DMA."
      )

  # If not ``None``, we lower to an indirect DMA instead of a regular DMA.
  if indirect_offsets is None:
    def _dma_wait(src_ref, dst_ref, sem):
      # `wait_dma2` does not support `subcore_id`, so it is ignored until
      # we migrate to `wait_dma`.
      tpu.wait_dma2(
        sem, src_ref, dst_ref, device_id=device_id, core_id=core_id
      )
      return []
    return tc_lowering.lower_with_transformed_refs(
        _dma_wait,
        [src_ref, dst_ref, sem],
        [src_aval, dst_aval, sem_aval],
    )

  if device_id is not None:
    raise NotImplementedError(
        "Scatter/gather to or from a remote device via `pltpu.async_copy` is"
        " not supported"
    )
  sem_aval, _ = _get_ref_and_transforms(sem_aval)
  sem, _ = _transform_ref(sem, sem_aval, sem_aval.shape)
  tpu.wait_indirect_dma(sem, src_ref, dst_ref)
  return []


def _extract_indirect_offsets_from_indices(
    indices: Sequence[Any],
    indices_aval: Sequence[Any],
    core_type: tpu_core.CoreType,
    indexer_shape: tuple[int | Any, ...],
    expected_shape: tuple[int, ...] | None = None,
) -> sc_core.Indices | None:
  """Extracts the indirect offsets from the indices, if there are any.

  Note that it ignores any dimensions other than major. The indices might
  need to be split further to deal with slicing of minor dimensions.
  """
  match indices_aval:
    case [sc_core.Indices(offsets_aval), *_]:
      offsets = indices[0]
      assert isinstance(offsets, sc_core.Indices)
      extracted = _extract_indirect_offsets_from_indices(
          [offsets.values, *indices[1:]],
          [offsets_aval, *indices_aval[1:]],
          core_type,
          indexer_shape,
          expected_shape,
      )
      if extracted is None:
        return None
      return sc_core.Indices(
          extracted.values, ignored_value=offsets.ignored_value
      )

    case [jax_core.AbstractValue() as offsets_aval, *_] if (
        # fmt: off
        isinstance(offsets_aval, state.AbstractRef) or
        (isinstance(offsets_aval, jax_core.ShapedArray) and offsets_aval.shape)
    ):  # fmt: on
      if len(offsets_aval.shape) != 1:
        raise NotImplementedError(
            "Only 1D indices are supported by scatter/gather via"
            " `pltpu.async_copy` on SparseCore, got rank"
            f" {len(offsets_aval.shape)}"
        )
      shape = (*offsets_aval.shape, *indexer_shape[len(offsets_aval.shape) :])
      if expected_shape is not None and shape != expected_shape:
        raise NotImplementedError(
            "The indexer shape in scatter/gather via `pltpu.async_copy` does"
            f" not match the expected shape. Want: {expected_shape}, got:"
            f" {shape}."
        )
      offsets = indices[0]
      assert isinstance(offsets, ir.Value)
    case [state.TransformedRef() as offsets_aval, *_]:
      if offsets_aval.dtype != jnp.dtype("int32"):
        raise NotImplementedError(
            "Only int32 indices are supported by scatter/gather via"
            " `pltpu.async_copy` with a dynamically-shaped indexer"
        )
      offsets_ref = indices[0]
      assert isinstance(offsets_ref, state.TransformedRef)
      offsets, _ = _transform_ref(
          offsets_ref.ref,
          offsets_aval.ref,
          offsets_aval.ref.shape,  # The shape before the indexing.
          offsets_ref.transforms,
      )
      assert isinstance(offsets, ir.Value)
    case _:
      return None

  if isinstance(offsets_aval, (state.AbstractRef, state.TransformedRef)):
    offsets_memory_space = tpu_core.memory_space_to_tpu_memory_space(
        offsets_aval.memory_space, core_type=core_type
    )
    if isinstance(offsets_memory_space, CoreMemorySpace):
      offsets_memory_space = offsets_memory_space.memory_space
    if offsets_memory_space is not MemorySpace.VMEM:
      raise NotImplementedError(
          "Indices for scatter/gather via `pltpu.async_copy` must be in VMEM,"
          f" got {offsets_memory_space!r}"
      )
  return sc_core.Indices(offsets)


def _extract_indirect_offsets(
    transforms: Sequence[state.Transform],
    expected_shape: tuple[int, ...],
    transforms_aval: Sequence[state.Transform],
    core_type: tpu_core.CoreType,
) -> tuple[sc_core.Indices | None, Sequence[state.Transform]]:
  for i, (indexer, indexer_aval) in enumerate(zip(transforms, transforms_aval)):
    if not isinstance(indexer, indexing.NDIndexer):
      continue
    assert isinstance(indexer_aval, indexing.NDIndexer)
    offsets = _extract_indirect_offsets_from_indices(
        indexer.indices,
        indexer_aval.indices,
        core_type,
        indexer.get_indexer_shape(),
        expected_shape,
    )
    if offsets is None:
      continue
    # The slices applied to other dimensions are processed independently of
    # indirect offsets.
    split_indices = (indexing.Slice(0, indexer.shape[0]), *indexer.indices[1:])
    split_indexer = indexing.NDIndexer(split_indices, indexer.shape, ())
    if i != len(transforms) - 1:
      raise NotImplementedError(
          "The indexed ref in scatter/gather via `pltpu.async_copy` cannot have"
          " any transforms following the indexer"
      )
    return offsets, [*transforms[:i], split_indexer]

  return None, transforms


def _has_indirect_offsets(
    transforms: Sequence[state.Transform],
    transforms_aval: Sequence[state.Transform],
    core_type: tpu_core.CoreType,
) -> bool:
  return any(
      _extract_indirect_offsets_from_indices(
          indexer.indices,
          indexer_aval.indices,  # pyrefly: ignore[missing-attribute]
          core_type,
          indexer.get_indexer_shape(),
      )
      is not None
      for indexer, indexer_aval in zip(transforms, transforms_aval)
      if isinstance(indexer, indexing.NDIndexer)
  )


@register_lowering_rule(jax_core.empty_ref_p)
def _empty_ref_lowering_rule(ctx: LoweringRuleContext, ty, memory_space):
  del ty, memory_space
  [aval_out] = ctx.avals_out
  return tc_lowering._alloc_value(aval_out, ctx=ctx)


@register_lowering_rule(
    lax.sort_p, kernel_types=[tpu_core.CoreType.SC_VECTOR_SUBCORE]
)
def _sort_lowering_rule(
    ctx: LoweringRuleContext, *xs, dimension, is_stable, num_keys
):
  del is_stable  # Unused, always stable.
  if dimension not in (0, -1):
    raise ValueError(f"Unsupported dimension: {dimension}")
  if num_keys != 1:
    raise NotImplementedError("Multiple sort keys not supported")
  sc_info = sc_core.get_sparse_core_info()
  supported_shape = (sc_info.num_lanes,)
  for i, aval in enumerate(ctx.avals_in):
    if aval.shape != supported_shape:
      raise NotImplementedError(
          f"Unsupported shape for operand {i} of SC sort: Got {aval.shape}, "
          f"expected {supported_shape}"
      )
  keys = xs[0]
  values = xs[1:]
  mask_type = ir.VectorType.get(
      [sc_info.num_lanes], ir.IntegerType.get_signless(1))
  mask = arith.constant(mask_type, ir.DenseElementsAttr.get_splat(
      mask_type, ir.BoolAttr.get(True)))
  if not values:
    _, sorted_keys, _ = tpu.sort(
        mask_type, keys.type, keys.type, keys, keys, mask=mask
    )
    return (sorted_keys,)
  results: list[ir.Value] = []
  for value in values:
    _, sorted_keys, sorted_value = tpu.sort(
        mask_type, keys.type, value.type, keys, value, mask=mask
    )
    if not results:
      results.append(sorted_keys)
    results.append(sorted_value)
  return tuple(results)


@register_lowering_rule(
    lax.rev_p, kernel_types=[tpu_core.CoreType.SC_VECTOR_SUBCORE]
)
def _rev_lowering_rule(ctx: LoweringRuleContext, x, dimensions):
  del ctx  # Unused.
  if dimensions != (0,):
    raise NotImplementedError(f"Invalid dimensions for SC lax.rev: {dimensions}")
  i32 = ir.IntegerType.get_signless(32)
  vec_dim = sc_core.get_sparse_core_info().num_lanes
  cdim = arith.constant(i32, ir.IntegerAttr.get(i32, vec_dim - 1))
  cdim_vec = vector.broadcast(ir.VectorType.get((vec_dim,), cdim.type), cdim)
  return tpu.dynamic_gather(
      x,
      arith.subi(cdim_vec, tpu.iota(cdim_vec.type, dimensions=[0])),
      dimensions=[0],
  )
