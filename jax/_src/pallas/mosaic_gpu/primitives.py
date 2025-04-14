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

"""GPU-specific Pallas primitives."""

from __future__ import annotations

from collections.abc import Sequence, Callable
import dataclasses
import enum
import itertools
import math
from typing import Any, Literal

import jax
from jax._src import core as jax_core
from jax._src import pretty_printer as pp
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.lib.mlir.dialects import llvm as llvm_dialect
from jax._src.lib.mlir.dialects import memref as memref_dialect
from jax._src.lib.mlir.dialects import nvvm as nvvm_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.pallas.mosaic_gpu.core import state_types
from jax._src.state import discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import utils as mgpu_utils
import jax.numpy as jnp


WARPGROUP_SIZE = 128


_Ref = pallas_core.AbstractMemoryRef | state_types.TransformedRef


def _check_ref(
    aval: object, name: str, memory_space: gpu_core.GPUMemorySpace
) -> None:
  if not isinstance(aval, state_types.AbstractRef):
    raise TypeError(f"{name} must be a reference, got {aval}")
  aval_memory_space = getattr(aval, "memory_space", None) or gpu_core.GMEM
  if aval_memory_space is not memory_space:
    raise ValueError(
        f"{name} must be a {memory_space.name.upper()} reference, got {aval}"
    )


load_p = jax_core.Primitive("load")

@load_p.def_effectful_abstract_eval
def _load_abstract_eval(src, *avals_flat, args_tree, layout, optimized):
  del layout, optimized  # Unused.
  transforms = args_tree.unflatten(avals_flat)
  return (
      jax_core.ShapedArray(transforms[-1].get_indexer_shape(), src.dtype),
      {state.ReadEffect(0)},
  )

@lowering.register_lowering_rule(load_p, mgpu.LoweringSemantics.Lane)
def _load_p_lowering_rule(
    ctx: lowering.LoweringRuleContext, x_ref, *leaves, args_tree, layout, optimized
):
  if not isinstance(x_ref, ir.Value) or not ir.MemRefType.isinstance(x_ref.type):
    raise TypeError(f"Can only load from references (got {x_ref}).")

  x_aval = ctx.avals_in[0]

  transforms = jax.tree.unflatten(args_tree, leaves)
  x_ref, transforms = lowering._handle_transforms(x_ref, transforms)

  if layout is not None:
    layout = layout.to_mgpu()

  is_signed = mgpu_utils.is_signed(x_aval.dtype)
  match transforms:
    case (gpu_core.UnswizzleRef(swizzle), gpu_core.UntileRef(tiling)):
      if tiling != (8, swizzle // x_aval.dtype.itemsize):
        raise NotImplementedError("Tiling does not fit swizzle")
      return mgpu.FragmentedArray.load_tiled(
          x_ref,
          is_signed=is_signed,
          swizzle=swizzle,
          layout=layout,
      )
    case ():
      # Handle scalar indexing.
      if not ctx.avals_out[0].shape:
        is_signed = mgpu_utils.is_signed(x_aval.dtype)
        val = memref_dialect.load(x_ref, [])
        return mgpu.FragmentedArray.splat(
            val, shape=(), layout=layout, is_signed=is_signed
        )
      match layout:
        case mgpu.WGMMA_ROW_LAYOUT | mgpu.WGMMA_COL_LAYOUT:
          return mgpu.FragmentedArray.load_untiled(
              x_ref,
              is_signed=is_signed,
              layout=layout,
              swizzle=16,
              optimized=optimized,
          )
        case mgpu.WGStridedFragLayout(shape=shape, vec_size=vec_size):
          ref_ty = ir.MemRefType(x_ref.type)
          if shape != tuple(ref_ty.shape):
            raise ValueError(
                f"Unsupported shape {shape}, (expected {tuple(ref_ty.shape)})"
            )
          return mgpu.FragmentedArray.load_strided(
              x_ref, is_signed=is_signed, vec_size=vec_size,
          )
        case None:
          return mgpu.FragmentedArray.load_strided(x_ref, is_signed=is_signed)
        case _:
          raise NotImplementedError(f"Unsupported layout: {layout}")
    case _:
      raise NotImplementedError(f"Unsupported transforms: {transforms}")


def load(
    src: _Ref,
    idx,
    *,
    layout: Layout | ParameterizedLayout | None = None,
    optimized: bool = True,
) -> jax.Array:
  """Loads from a reference into an array with the specified layout.

  Args:
    src: The reference to load from. Can be either in SMEM or GMEM.
    idx: The index to load from.
    layout: The optional layout to use for the resulting array.
    optimized: If True, a compilation error will be raised if no optimized
      implementation for the load is available.

  Returns:
    The loaded array.
  """
  src, src_transforms = state_primitives.get_ref_and_transforms(
      src, idx, "load", force_trailing_indexer=True,
  )
  flat_src_transforms, src_transforms_treedef = tree_util.tree_flatten(
      src_transforms
  )
  return load_p.bind(
      src,
      *flat_src_transforms,
      args_tree=src_transforms_treedef,
      layout=layout,
      optimized=optimized,
  )


copy_smem_to_gmem_p = jax_core.Primitive("copy_smem_to_gmem")
copy_smem_to_gmem_p.multiple_results = True


@copy_smem_to_gmem_p.def_effectful_abstract_eval
def _copy_smem_to_gmem_abstract_eval(src, dst, *args, **params):
  _check_ref(src, "src", gpu_core.SMEM)
  _check_ref(dst, "dst", gpu_core.GMEM)
  del args, params  # Unused.
  return (), {state.ReadEffect(0), state.WriteEffect(1)}


def _copy_smem_to_gmem_pp_eqn(
    eqn: jax_core.JaxprEqn,
    context: jax_core.JaxprPpContext,
    settings: jax_core.JaxprPpSettings,
):
  src, dst, *flat_args = eqn.invars
  src_transforms_treedef = eqn.params["src_transforms_treedef"]
  dst_transforms_treedef = eqn.params["dst_transforms_treedef"]
  pp_params = {}
  if not (commit_group := eqn.params["commit_group"]):
    pp_params["commit_group"] = commit_group
  if eqn.params["has_user_predicate"]:
    flat_args, user_predicate = flat_args[:-1], flat_args[-1]
    pp_params["user_predicate"] = jax_core.pp_var(user_predicate, context)
  if reduction_op := eqn.params["reduction_op"]:
    pp_params["reduction_op"] = reduction_op
  flat_src_transforms, flat_dst_transforms = util.split_list(
      flat_args,
      [src_transforms_treedef.num_leaves],
  )
  src_transforms = src_transforms_treedef.unflatten(flat_src_transforms)
  dst_transforms = dst_transforms_treedef.unflatten(flat_dst_transforms)
  return pp.concat([
      pp.text("copy_smem_to_gmem"),
      jax_core.pp_kv_pairs(pp_params.items(), context, settings),
      pp.text(" "),
      state_primitives.pp_ref_transforms(context, src, src_transforms),
      pp.text(" -> "),
      state_primitives.pp_ref_transforms(context, dst, dst_transforms),
  ])


jax_core.pp_eqn_rules[copy_smem_to_gmem_p] = _copy_smem_to_gmem_pp_eqn


@lowering.register_lowering_rule(
    copy_smem_to_gmem_p, mgpu.LoweringSemantics.Lane)
@lowering.register_lowering_rule(
    copy_smem_to_gmem_p, mgpu.LoweringSemantics.Lane,
    primitive_semantics=gpu_core.PrimitiveSemantics.Warp)
@lowering.register_lowering_rule(
    copy_smem_to_gmem_p, mgpu.LoweringSemantics.Warpgroup
)
def _copy_smem_to_gmem_lowering(
    ctx: lowering.LoweringRuleContext,
    src,
    dst,
    *flat_args,
    src_transforms_treedef,
    dst_transforms_treedef,
    has_user_predicate,
    commit_group,
    reduction_op,
):
  if has_user_predicate:
    flat_args, user_predicate = flat_args[:-1], flat_args[-1]
    predicate = lowering._ensure_ir_value(user_predicate, jnp.bool)
  else:
    predicate = None

  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    if predicate is not None:
      assert ctx.module_ctx.single_lane_predicate is not None
      predicate = arith_dialect.andi(
          predicate, ctx.module_ctx.single_lane_predicate
      )
    else:
      predicate = ctx.module_ctx.single_lane_predicate

  flat_src_transforms, flat_dst_transforms = util.split_list(
      flat_args,
      [src_transforms_treedef.num_leaves],
  )
  src_transforms = src_transforms_treedef.unflatten(flat_src_transforms)
  dst_transforms = dst_transforms_treedef.unflatten(flat_dst_transforms)
  src, src_transforms = lowering._handle_transforms(src, src_transforms, handle_transposes=False)
  copy_params = _extract_gmem_copy_params(dst_transforms) | _extract_smem_copy_params(src_transforms)
  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    ctx.launch_ctx.async_copy(
        src_ref=src,
        dst_ref=dst,
        predicate=predicate,
        arrive=commit_group,
        reduction_op=reduction_op,
        **copy_params,
    )
    return ()

  if "gmem_slice" not in copy_params:
    i32 = ir.IntegerType.get_signless(32)
    slice_lengths = ir.MemRefType(src.type).shape
    indices = [mgpu.utils.c(0, i32)] * len(slice_lengths)
  else:
    indices, slice_lengths = _split_gmem_slice(copy_params["gmem_slice"])
  assert copy_params.get("swizzle") is None
  assert not copy_params.get("gmem_transform")
  mgpu.dialect.async_store(
      src,
      dst,
      indices,
      slice_lengths,
      predicate=predicate,
      commit_group=commit_group,  # type: ignore[call-arg]
  )
  return ()


def _split_gmem_slice(gmem_slice):
  i32 = ir.IntegerType.get_signless(32)
  indices = []
  slice_lengths = []
  for idx in gmem_slice:
    match idx:
      case slice():
        indices.append(mgpu_utils.c(idx.start, i32))
        slice_lengths.append(idx.stop - idx.start)
      case mgpu.DynamicSlice():
        indices.append(arith_dialect.index_cast(i32, idx.base))
        slice_lengths.append(idx.length)
      case ir.Value():
        indices.append(arith_dialect.index_cast(i32, idx))
        slice_lengths.append(-1)
      case _:
        raise NotImplementedError(f"Unsupported GMEM slice: {idx}")
  return indices, slice_lengths


def _extract_gmem_copy_params(transforms):
  if not transforms:
    return {}
  for transform in transforms:
    if not isinstance(transform, indexing.NDIndexer):
      raise NotImplementedError(
          "Non-indexing transforms on GMEM refs are not implemented.")
  indexer = lowering.merge_indexers(transforms)
  return dict(
      gmem_slice=lowering._ndindexer_indices(indexer),
  )


def _extract_smem_copy_params(transforms):
  if not transforms:
    return {}
  # Split off swizzling, if present
  match transforms:
    case [gpu_core.UnswizzleRef(swizzle), *transforms]:
      pass
    case _:
      swizzle = None
  gpu_transforms = tuple(t.undo_to_gpu_transform() for t in transforms[::-1])
  return dict(
      gmem_transform=gpu_transforms,
      swizzle=swizzle,
  )


def copy_smem_to_gmem(
    src: _Ref,
    dst: _Ref,
    predicate: jax.Array | None = None,
    *,
    commit_group: bool = True,
    reduction_op: mgpu.ReductionOp | None = None,
) -> None:
  """Asynchronously copies a SMEM reference to a GMEM reference.

  Args:
    src: The SMEM reference to copy from.
    dst: The GMEM reference to copy to.
    predicate: A boolean indicating whether the copy should be performed. If
      ``None``, the copy is always performed.
    commit_group: If ``True``, this and any previously uncommitted copies are
      committed to a group and can be awaited jointly via
      :func:`jax.experimental.mosaic.gpu.wait_smem_to_gmem`.
    reduction_op: If set, perform the specified reduction operation when storing
      to GMEM. For example, using ``"add"`` is conceptually equivalent to
      doing ``src += dst``.

  See also:
    :func:`jax.experimental.mosaic.gpu.wait_smem_to_gmem`
    :func:`jax.experimental.mosaic.gpu.commit_smem`
  """
  src, src_transforms = state_primitives.get_ref_and_transforms(
      src, None, "copy_smem_to_gmem", force_trailing_indexer=False,
  )
  dst, dst_transforms = state_primitives.get_ref_and_transforms(
      dst, None, "copy_smem_to_gmem", force_trailing_indexer=False,
  )
  flat_src_transforms, src_transforms_treedef = tree_util.tree_flatten(
      src_transforms
  )
  flat_dst_transforms, dst_transforms_treedef = tree_util.tree_flatten(
      dst_transforms
  )
  copy_smem_to_gmem_p.bind(
      src,
      dst,
      *flat_src_transforms,
      *flat_dst_transforms,
      *[] if predicate is None else [predicate],
      src_transforms_treedef=src_transforms_treedef,
      dst_transforms_treedef=dst_transforms_treedef,
      has_user_predicate=predicate is not None,
      commit_group=commit_group,
      reduction_op=reduction_op,
  )
  return None


copy_gmem_to_smem_p = jax_core.Primitive("copy_gmem_to_smem")
copy_gmem_to_smem_p.multiple_results = True


@copy_gmem_to_smem_p.def_effectful_abstract_eval
def _copy_gmem_to_smem_abstract_eval(src, dst, barrier, *args, **params):
  del args, params  # Unused.
  _check_ref(src, "src", gpu_core.GMEM)
  _check_ref(dst, "dst", gpu_core.SMEM)
  _check_ref(barrier, "barrier", gpu_core.SMEM)
  return (), {state.ReadEffect(0), state.WriteEffect(1)}


def _copy_gmem_to_smem_pp_eqn(
    eqn: jax_core.JaxprEqn,
    context: jax_core.JaxprPpContext,
    settings: jax_core.JaxprPpSettings,
):
  src, dst, barrier, *flat_args = eqn.invars
  src_transforms_treedef = eqn.params["src_transforms_treedef"]
  dst_transforms_treedef = eqn.params["dst_transforms_treedef"]
  barrier_transforms_treedef = eqn.params["barrier_transforms_treedef"]
  pp_params = {}
  if collective_axes := eqn.params["collective_axes"]:
    pp_params["collective_axes"] = collective_axes
  flat_src_transforms, flat_dst_transforms, flat_barrier_transforms = (
      util.split_list(
          flat_args,
          [
              src_transforms_treedef.num_leaves,
              dst_transforms_treedef.num_leaves,
          ],
      )
  )
  src_transforms = src_transforms_treedef.unflatten(flat_src_transforms)
  dst_transforms = dst_transforms_treedef.unflatten(flat_dst_transforms)
  barrier_transforms = barrier_transforms_treedef.unflatten(
      flat_barrier_transforms
  )
  return pp.concat([
      pp.text("copy_gmem_to_smem"),
      jax_core.pp_kv_pairs(pp_params.items(), context, settings),
      pp.text(" "),
      state_primitives.pp_ref_transforms(context, src, src_transforms),
      pp.text(" -> "),
      state_primitives.pp_ref_transforms(context, dst, dst_transforms),
      pp.text(" using "),
      state_primitives.pp_ref_transforms(context, barrier, barrier_transforms),
  ])


jax_core.pp_eqn_rules[copy_gmem_to_smem_p] = _copy_gmem_to_smem_pp_eqn


@lowering.register_lowering_rule(
    copy_gmem_to_smem_p, mgpu.LoweringSemantics.Lane)
@lowering.register_lowering_rule(
    copy_gmem_to_smem_p, mgpu.LoweringSemantics.Lane,
    primitive_semantics=gpu_core.PrimitiveSemantics.Warp)
@lowering.register_lowering_rule(
    copy_gmem_to_smem_p, mgpu.LoweringSemantics.Warpgroup
)
def _copy_gmem_to_smem_lowering(
    ctx: lowering.LoweringRuleContext,
    src,
    dst,
    barrier,
    *flat_transforms,
    src_transforms_treedef,
    dst_transforms_treedef,
    barrier_transforms_treedef,
    collective_axes,
):
  flat_src_transforms, flat_dst_transforms, flat_barrier_transforms = (
      util.split_list(
          flat_transforms,
          [
              src_transforms_treedef.num_leaves,
              dst_transforms_treedef.num_leaves,
          ],
      )
  )
  src_transforms = src_transforms_treedef.unflatten(flat_src_transforms)
  dst_transforms = dst_transforms_treedef.unflatten(flat_dst_transforms)
  dst, dst_transforms = lowering._handle_transforms(dst, dst_transforms, handle_transposes=False)
  copy_params = _extract_smem_copy_params(dst_transforms) | _extract_gmem_copy_params(src_transforms)
  barrier_indexer = _extract_barrier_indexer(
      barrier_transforms_treedef.unflatten(flat_barrier_transforms)
  )
  if barrier_indexer is not None:
    barrier = barrier.__getitem__(
        *map(lowering._as_index, barrier_indexer.indices)
    )
  collective = None
  if collective_axes is not None:
    collective = tuple(
        lowering._resolve_cluster_axis(ctx.module_ctx.axis_names, axis)
        for axis in collective_axes
    )
  dst_ty = ir.MemRefType(dst.type)
  bits = math.prod(dst_ty.shape) * mgpu.bitwidth(dst_ty.element_type)
  if bits % 8:
    raise ValueError(
        f"Can only transfer integer bytes (shape={dst_ty.shape},"
        f" dtype={dst_ty.element_type})"
    )
  bytes = bits // 8
  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    if bytes % WARPGROUP_SIZE:
      raise NotImplementedError("Only aligned copies are supported")
    # We arrive uniformly from each thread in the WG, so we need to divide the
    # number of bytes by the number of threads in the WG.
    # TODO: apaszke - Relax this. We can just select the WG leader and have it
    # arrive with the whole transfer size, while everyone else arrives with 0.
    # But we should continue using this scheme as it's likely to be faster.
    bytes //= WARPGROUP_SIZE
    barrier.arrive_expect_tx(bytes)
    ctx.launch_ctx.async_copy(
        src_ref=src,
        dst_ref=dst,
        barrier=barrier,
        arrive=False,
        predicate=ctx.module_ctx.single_lane_predicate,
        collective=collective,
        **copy_params,
    )
    return ()

  if "gmem_slice" not in copy_params:
    i32 = ir.IntegerType.get_signless(32)
    slice_lengths = ir.MemRefType(src.type).shape
    indices = [mgpu.utils.c(0, i32)] * len(slice_lengths)
  else:
    indices, slice_lengths = _split_gmem_slice(copy_params["gmem_slice"])
  assert copy_params.get("swizzle") is None
  assert not copy_params.get("gmem_transform")
  barrier_ref = barrier.as_dialect_barrier_memref()
  mgpu.dialect.arrive_expect_tx(barrier_ref, bytes)
  mgpu.dialect.async_load(
      src,
      dst,
      barrier_ref,
      indices,
      slice_lengths,
      collective=ir.ArrayAttr.get([]),
  )
  return ()


def copy_gmem_to_smem(
    src: _Ref,
    dst: _Ref,
    barrier: _Ref,
    *,
    collective_axes: str | tuple[str, ...] | None = None,
) -> None:
  """Asynchronously copies a GMEM reference to a SMEM reference.

  See also:
    :func:`jax.experimental.mosaic.gpu.barrier_arrive`
    :func:`jax.experimental.mosaic.gpu.barrier_wait`
  """
  src, src_transforms = state_primitives.get_ref_and_transforms(
      src, None, "copy_gmem_to_smem", force_trailing_indexer=False,
  )
  dst, dst_transforms = state_primitives.get_ref_and_transforms(
      dst, None, "copy_gmem_to_smem", force_trailing_indexer=False,
  )
  flat_src_transforms, src_transforms_treedef = tree_util.tree_flatten(
      src_transforms
  )
  flat_dst_transforms, dst_transforms_treedef = tree_util.tree_flatten(
      dst_transforms
  )
  barrier, barrier_transforms = state_primitives.get_ref_and_transforms(
      barrier, None, "copy_gmem_to_smem", force_trailing_indexer=False,
  )
  flat_barrier_transforms, barrier_transforms_treedef = tree_util.tree_flatten(
      barrier_transforms
  )
  if isinstance(collective_axes, str):
    collective_axes = (collective_axes,)
  copy_gmem_to_smem_p.bind(
      src,
      dst,
      barrier,
      *flat_src_transforms,
      *flat_dst_transforms,
      *flat_barrier_transforms,
      src_transforms_treedef=src_transforms_treedef,
      dst_transforms_treedef=dst_transforms_treedef,
      barrier_transforms_treedef=barrier_transforms_treedef,
      collective_axes=collective_axes,
  )
  return None


def _extract_barrier_indexer(transforms) -> indexing.NDIndexer | None:
  if not transforms:
    return None
  match transforms:
    case [indexing.NDIndexer(indices=[idx]) as indexer]:
      if not isinstance(idx, indexing.Slice):
        return indexer
      if indexing.Slice.from_slice(slice(None), *indexer.shape) == idx:
        # Special-case: the whole slice.
        return None
      else:
        raise ValueError(
            f"Barrier can only be indexed with an integer, got {idx}"
        )
    case [indexing.NDIndexer()]:
      raise NotImplementedError("Barrier does not support multiple indices")
    case []:
      return None
    case _:
      raise ValueError("Barrier does not support arbirary transforms")


barrier_arrive_p = jax_core.Primitive("barrier_arrive")
barrier_arrive_p.multiple_results = True


@barrier_arrive_p.def_effectful_abstract_eval
def _barrier_arrive_abstract_eval(barrier, *args, **params):
  del args, params  # Unused.
  _check_ref(barrier, "barrier", gpu_core.SMEM)
  return (), {gpu_core._memory_effect}


def _barrier_arrive_pp_eqn(
    eqn: jax_core.JaxprEqn,
    context: jax_core.JaxprPpContext,
    settings: jax_core.JaxprPpSettings,
):
  del settings
  barrier, *flat_transforms = eqn.invars
  transforms_treedef = eqn.params["transforms_treedef"]
  transforms = transforms_treedef.unflatten(flat_transforms)
  return pp.concat([
      pp.text("barrier_arrive"),
      pp.text(" "),
      state_primitives.pp_ref_transforms(context, barrier, transforms),
  ])


jax_core.pp_eqn_rules[barrier_arrive_p] = _barrier_arrive_pp_eqn


@lowering.register_lowering_rule(barrier_arrive_p, mgpu.LoweringSemantics.Lane)
@lowering.register_lowering_rule(barrier_arrive_p, mgpu.LoweringSemantics.Warpgroup)
def _barrier_arrive_lowering(
    ctx: lowering.LoweringRuleContext,
    barrier,
    *flat_transforms,
    transforms_treedef,
):
  del ctx  # Unused.
  transforms = transforms_treedef.unflatten(flat_transforms)
  indexer = _extract_barrier_indexer(transforms)
  if indexer is not None:
    barrier = barrier.__getitem__(*map(lowering._as_index, indexer.indices))
  barrier.arrive()
  return ()


def barrier_arrive(barrier: pallas_core.AbstractMemoryRef) -> None:
  """Arrives at the given barrier."""
  barrier, transforms = state_primitives.get_ref_and_transforms(
      barrier, None, "barrier_arrive", force_trailing_indexer=False,
  )
  flat_transforms, transforms_treedef = tree_util.tree_flatten(transforms)
  barrier_arrive_p.bind(
      barrier, *flat_transforms, transforms_treedef=transforms_treedef
  )


barrier_wait_p = jax_core.Primitive("barrier_wait")
barrier_wait_p.multiple_results = True


@barrier_wait_p.def_effectful_abstract_eval
def _barrier_wait_abstract_eval(barrier, *args, **params):
  _check_ref(barrier, "barrier", gpu_core.SMEM)
  del args, params  # Unused.
  return (), {gpu_core._memory_effect}


def _barrier_wait_pp_eqn(
    eqn: jax_core.JaxprEqn,
    context: jax_core.JaxprPpContext,
    settings: jax_core.JaxprPpSettings,
):
  del settings
  barrier, *flat_transforms = eqn.invars
  transforms_treedef = eqn.params["transforms_treedef"]
  transforms = transforms_treedef.unflatten(flat_transforms)
  return pp.concat([
      pp.text("barrier_wait"),
      pp.text(" "),
      state_primitives.pp_ref_transforms(context, barrier, transforms),
  ])


jax_core.pp_eqn_rules[barrier_wait_p] = _barrier_wait_pp_eqn


@lowering.register_lowering_rule(barrier_wait_p, mgpu.LoweringSemantics.Lane)
@lowering.register_lowering_rule(barrier_wait_p, mgpu.LoweringSemantics.Warpgroup)
def _barrier_wait_lowering(
    ctx: lowering.LoweringRuleContext,
    barrier,
    *flat_transforms,
    transforms_treedef,
):
  del ctx  # Unused.
  transforms = transforms_treedef.unflatten(flat_transforms)
  indexer = _extract_barrier_indexer(transforms)
  if indexer is not None:
    barrier = barrier.__getitem__(*map(lowering._as_index, indexer.indices))
  barrier.wait()
  return ()


def barrier_wait(barrier: pallas_core.AbstractMemoryRef) -> None:
  """Waits on the given barrier."""
  barrier, transforms = state_primitives.get_ref_and_transforms(
      barrier, None, "barrier_wait", force_trailing_indexer=False,
  )
  flat_transforms, transforms_treedef = tree_util.tree_flatten(transforms)
  barrier_wait_p.bind(
      barrier, *flat_transforms, transforms_treedef=transforms_treedef
  )


wait_smem_to_gmem_p = jax_core.Primitive("wait_smem_to_gmem")
wait_smem_to_gmem_p.multiple_results = True


@wait_smem_to_gmem_p.def_effectful_abstract_eval
def _wait_smem_to_gmem_abstract_eval(n, *, wait_read_only):
  del n, wait_read_only  # Unused.
  return (), {gpu_core._memory_effect}


@lowering.register_lowering_rule(
    wait_smem_to_gmem_p, mgpu.LoweringSemantics.Lane)
@lowering.register_lowering_rule(
    wait_smem_to_gmem_p, mgpu.LoweringSemantics.Warpgroup
)
def _wait_smem_to_gmem_lowering(
    ctx: lowering.LoweringRuleContext, n, *, wait_read_only
):
  ctx.launch_ctx.await_async_copy(
      allow_groups=n, await_read_only=wait_read_only
  )
  return ()


def wait_smem_to_gmem(n: int, wait_read_only: bool = False) -> None:
  """Waits until there are no more than ``n`` SMEM->GMEM copies in flight.

  Args:
    n: The maximum number of copies in flight to wait for.
    wait_read_only: If ``True``, wait for the in flight copies to finish
      reading from SMEM. The writes to GMEM are not waited for.
  """
  wait_smem_to_gmem_p.bind(n, wait_read_only=wait_read_only)


commit_group_p = jax_core.Primitive("commit_group")
commit_group_p.multiple_results = True


@commit_group_p.def_effectful_abstract_eval
def _commit_group_abstract_eval():
  return (), {gpu_core._memory_effect}


@lowering.register_lowering_rule(commit_group_p, mgpu.LoweringSemantics.Lane)
@lowering.register_lowering_rule(
    commit_group_p, mgpu.LoweringSemantics.Warpgroup)
def _commit_group_lowering(ctx: lowering.LoweringRuleContext):
  del ctx  # Unused.
  nvvm_dialect.cp_async_bulk_commit_group()
  return ()


def commit_smem_to_gmem_group() -> None:
  """Commits all issued but uncommited SMEM->GMEM copies to a group."""
  commit_group_p.bind()


# WGMMA on an accumulator reference
wgmma_ref_p = jax_core.Primitive("wgmma_ref")
wgmma_ref_p.multiple_results = True


def wgmma(acc: gpu_core.WGMMAAbstractAccumulatorRef, a, b) -> None:
  """Performs an asynchronous warp group matmul-accumulate on the given references.

  Conceptually, this is equivalent to doing ``acc[...] += a[...] @ b[...]``,
  except that the computation is performed asynchronously.

  Args:
    acc: The accumulator reference. Needs to be allocated via
      :func:`jax.experimental.pallas.run_scoped` called with a
      :func:`jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef`.
    a: The left hand side operand reference.
    b: The right hand side operand reference.

  See also:
    :func:`jax.experimental.pallas.mosaic_gpu.wgmma_wait`
  """
  m, n = acc.shape
  m2, k = a.shape
  k2, n2 = b.shape

  if m != m2 or n != n2 or k != k2:
    raise ValueError(
        f"Incompatible shapes for matrix multiplication: lhs={a.shape},"
        f" rhs={b.shape=}, acc={acc.shape}"
    )

  if a.dtype != b.dtype:
    raise ValueError(f"Mixed input dtypes for matrix multiplication unsupported: lhs={a.dtype}, rhs={b.dtype}")

  if isinstance(a, pallas_core.TransformedRef):
    a_transforms_leaves, a_transforms_tree = jax.tree.flatten(a.transforms)
    a = a.ref
  else:
    a_transforms_leaves, a_transforms_tree = [], None

  if isinstance(b, pallas_core.TransformedRef):
    b_transforms_leaves, b_transforms_tree = jax.tree.flatten(b.transforms)
    b = b.ref
  else:
    b_transforms_leaves, b_transforms_tree = [], None

  wgmma_ref_p.bind(
      acc,
      a,
      b,
      *a_transforms_leaves,
      *b_transforms_leaves,
      a_transforms_tree=a_transforms_tree,
      b_transforms_tree=b_transforms_tree,
  )


@wgmma_ref_p.def_effectful_abstract_eval
def _wgmma_ref_effectful_abstract_eval(acc_aval, a_aval, b_aval, *_, **params):
  del b_aval, params
  if not isinstance(acc_aval, gpu_core.WGMMAAbstractAccumulatorRef):
    raise TypeError(f"Expected WGMMAAbstractAccumulatorRef got {acc_aval}")
  return (), {
      gpu_core._wgmma_pipeline_effect,
      state.WriteEffect(0),
      state.ReadEffect(0),
      state.ReadEffect(2),
      *([state.ReadEffect(1)] if isinstance(a_aval, state.AbstractRef) else [])
  }


def _wgmma_ref_pp_eqn(
    eqn: jax_core.JaxprEqn,
    context: jax_core.JaxprPpContext,
    settings: jax_core.JaxprPpSettings,
):
  del settings
  acc, a, b, *leaves = eqn.invars
  a_transforms_treedef = eqn.params["a_transforms_tree"]
  b_transforms_treedef = eqn.params["b_transforms_tree"]
  split = getattr(a_transforms_treedef, "num_leaves", 0)
  a_transforms = (
      a_transforms_treedef.unflatten(leaves[:split])
      if a_transforms_treedef is not None
      else []
  )
  b_transforms = (
      b_transforms_treedef.unflatten(leaves[split:])
      if b_transforms_treedef is not None
      else []
  )
  return pp.concat([
      pp.text("wgmma_ref"),
      pp.text(" "),
      pp.text(jax_core.pp_var(acc, context)),
      pp.text(" <- "),
      state_primitives.pp_ref_transforms(context, a, a_transforms),
      pp.text(" @ "),
      state_primitives.pp_ref_transforms(context, b, b_transforms),
  ])


jax_core.pp_eqn_rules[wgmma_ref_p] = _wgmma_ref_pp_eqn


@discharge.register_discharge_rule(wgmma_ref_p)
def _wgmma_ref_discharge(in_avals, out_avals, *args, **kwargs):
  del in_avals, out_avals
  return (wgmma_p.bind(*args, **kwargs), *([None] * (len(args) - 1))), []


# Functional WGMMA, returns a shaped array. Internal.
wgmma_p = jax_core.Primitive("wgmma")


@lowering.register_lowering_rule(wgmma_p, mgpu.LoweringSemantics.Lane)
def _wgmma_lowering(
    ctx: lowering.LoweringRuleContext,
    acc,
    a,
    b,
    *transforms_leaves,
    a_transforms_tree,
    b_transforms_tree,
):
  _, a_aval, *_ = ctx.avals_in
  lhs_swizzle: int | None = None
  if a_transforms_tree is not None:
    a_transforms_leaves, b_transforms_leaves = util.split_list(
        transforms_leaves, [a_transforms_tree.num_leaves]
    )
    a_transforms = a_transforms_tree.unflatten(a_transforms_leaves)
    a, a_transforms = lowering._handle_transforms(
        a, a_transforms, handle_transposes=False, handle_reshapes=False
    )
    match a_transforms:
      case (gpu_core.UnswizzleRef(lhs_swizzle), gpu_core.UntileRef(tiling)):
        lhs_transpose = False
      case (
          gpu_core.UnswizzleRef(lhs_swizzle),
          gpu_core.UntileRef(tiling),
          gpu_core.TransposeRef((1, 0)),
      ):
        lhs_transpose = True
      case _:
        raise ValueError(f"WGMMA lhs has unsupported transforms: {a_transforms}.")
    swizzle_elems = lhs_swizzle // a_aval.dtype.itemsize
    if tiling != (8, swizzle_elems):
      raise NotImplementedError("WGMMA lhs tiling does not fit swizzle")
  else:
    lhs_transpose = False
    b_transforms_leaves = transforms_leaves  # type: ignore
    if not isinstance(a, mgpu.FragmentedArray):
      raise ValueError(
          "When WGMMA lhs is passed in as a ref, it must be transformed by"
          " swizzling and tiling appropriately."
      )

  b_transforms = b_transforms_tree.unflatten(b_transforms_leaves)
  b, b_transforms = lowering._handle_transforms(
      b, b_transforms, handle_transposes=False, handle_reshapes=False
  )

  match b_transforms:
    case (gpu_core.UnswizzleRef(rhs_swizzle), gpu_core.UntileRef(rhs_tiling)):
      rhs_transpose = False
    case (
        gpu_core.UnswizzleRef(rhs_swizzle),
        gpu_core.UntileRef(rhs_tiling),
        gpu_core.TransposeRef((1, 0)),
    ):
      rhs_transpose = True
    case (
        gpu_core.UnswizzleRef(rhs_swizzle),
        gpu_core.TransposeRef((1, 0, 2, 3, 4)),
        gpu_core.UntileRef(rhs_tiling),
        gpu_core.TransposeRef(permutation=(1, 0, 2)),
        state.types.RefReshaper(shape=new_shape),
    ):
      if len(rhs_tiling) != 2 or len(new_shape) != 2:
        raise ValueError("WGMMA expects shapes 2D tiled into 2D tiles.")

      if any(d % t != 0 for d, t in util.safe_zip(new_shape, rhs_tiling)):
        raise ValueError(
            f"The last reshape {new_shape} is not divisible by the tiling"
            f" {rhs_tiling}."
        )

      high_dims = [d // t for d, t in util.safe_zip(new_shape, rhs_tiling)]
      b = mgpu.memref_reshape(b, (*high_dims, *rhs_tiling))
      rhs_transpose = False
    case _:
      raise ValueError(f"WGMMA rhs has unsupported transforms: {b_transforms}.")

  if lhs_swizzle is not None:
    swizzle_elems = rhs_swizzle // a_aval.dtype.itemsize
    if rhs_swizzle != lhs_swizzle:
      raise NotImplementedError("WGMMA rhs swizzle must match lhs swizzle")
    if rhs_tiling != (8, swizzle_elems):
      raise NotImplementedError("WGMMA rhs tiling does not fit swizzle")

  if lhs_transpose:
    a = mgpu.memref_transpose(a, (1, 0, 3, 2))
  if rhs_transpose:
    b = mgpu.memref_transpose(b, (1, 0, 3, 2))
  new_acc = mgpu.wgmma(acc, a, b, swizzle=rhs_swizzle)
  nvvm_dialect.wgmma_commit_group_sync_aligned()
  return new_acc


@lowering.register_lowering_rule(wgmma_p, mgpu.LoweringSemantics.Warpgroup)
def _wgmma_warpgroup_lowering(
    ctx: lowering.LoweringRuleContext,
    acc,
    a,
    b,
    *transforms_leaves,
    a_transforms_tree,
    b_transforms_tree,
):
  del ctx  # Unused.

  if a_transforms_tree is not None:
    a_transforms_leaves, b_transforms_leaves = util.split_list(
        transforms_leaves, [a_transforms_tree.num_leaves]
    )
    a_transforms = a_transforms_tree.unflatten(a_transforms_leaves)
    a, a_transforms = lowering._handle_transforms(a, a_transforms)
    match a_transforms:
      case (gpu_core.TransposeRef((1, 0)),):
        a = mgpu.memref_transpose(a, (1, 0))
      case ():
        pass
      case _:
        raise ValueError(
            f"WGMMA lhs has unsupported transforms: {a_transforms}."
        )
  else:
    b_transforms_leaves = transforms_leaves  # type: ignore

  if b_transforms_tree is not None:
    b_transforms = b_transforms_tree.unflatten(b_transforms_leaves)
    b, b_transforms = lowering._handle_transforms(b, b_transforms)
    match b_transforms:
      case (gpu_core.TransposeRef((1, 0)),):
        b = mgpu.memref_transpose(b, (1, 0))
      case ():
        pass
      case _:
        raise ValueError(
            f"WGMMA rhs has unsupported transforms: {b_transforms}."
        )

  new_acc = mgpu.dialect.wgmma(acc, a, b)
  nvvm_dialect.wgmma_commit_group_sync_aligned()
  return new_acc


@wgmma_p.def_effectful_abstract_eval
def _wgmma_effectful_abstract_eval(acc, lhs_ref, *args, **kwargs):
  del args, kwargs
  return acc, {
      gpu_core._wgmma_pipeline_effect,
      state.ReadEffect(2),
      *([state.ReadEffect(1)] if isinstance(lhs_ref, state.AbstractRef) else [])
  }

wgmma_wait_p = jax_core.Primitive("wgmma_wait")
wgmma_wait_p.multiple_results = True


def wgmma_wait(n: int):
  """Waits until there is no more than ``n`` WGMMA operations in flight."""
  return wgmma_wait_p.bind(n)


@wgmma_wait_p.def_effectful_abstract_eval
def wgmma_wait_effectful_abstract_eval(_):
  return [], {gpu_core._wgmma_pipeline_effect}


@lowering.register_lowering_rule(wgmma_wait_p, mgpu.LoweringSemantics.Lane)
@lowering.register_lowering_rule(wgmma_wait_p, mgpu.LoweringSemantics.Warpgroup)
def _wgmma_wait_lowering(ctx: lowering.LoweringRuleContext, allow_groups):
  del ctx
  nvvm_dialect.wgmma_wait_group_sync_aligned(allow_groups)
  return ()


wgmma_accumulator_deref_p = jax_core.Primitive("wgmma_accumulator_deref_p")

def wgmma_accumulator_deref(acc):
  """Dereferences an accumulator register."""

  if not isinstance(acc.aval, gpu_core.WGMMAAbstractAccumulatorRef):
    raise TypeError(f"acc must be a WGMMAAccumulatorAbstractRef, got {acc.aval=}")

  return wgmma_accumulator_deref_p.bind(acc)

@wgmma_accumulator_deref_p.def_effectful_abstract_eval
def _wgmma_accumulator_deref_abstract_eval(acc):
  # Dereferencing implies flushing so we have a wgmma pipeline effect.
  ret = acc.inner_aval if isinstance(acc, state.AbstractRef) else acc
  assert isinstance(ret, jax_core.ShapedArray), acc
  return ret, {gpu_core._wgmma_pipeline_effect}


@discharge.register_discharge_rule(wgmma_accumulator_deref_p)
def _wgmma_accumulator_deref_discharge(in_avals, out_avals, acc):
  del in_avals, out_avals
  return (None,), wgmma_accumulator_deref_p.bind(acc)


@lowering.register_lowering_rule(
    wgmma_accumulator_deref_p, mgpu.LoweringSemantics.Lane
)
@lowering.register_lowering_rule(
    wgmma_accumulator_deref_p, mgpu.LoweringSemantics.Warpgroup
)
def _wgmma_accumulator_deref_lowering(ctx: lowering.LoweringRuleContext, acc):
  nvvm_dialect.wgmma_wait_group_sync_aligned(0)
  return (
      acc.value
      if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane
      else acc
  )


class Layout(enum.Enum):
  #: [m, n] matrix, where m % 64 == 0 == n % 8.
  WGMMA = enum.auto()
  #: [m] matrix, where m % 64 == 0.
  WGMMA_ROW = enum.auto()
  #: [n] matrix, where n % 8 == 0.
  WGMMA_COL = enum.auto()
  WGMMA_TRANSPOSED = enum.auto()

  WG_SPLAT = enum.auto()
  WG_STRIDED = enum.auto()

  def __call__(self, *args, **kwargs) -> ParameterizedLayout:
    return ParameterizedLayout(self, args, kwargs)

  def to_mgpu(self, *args, **kwargs) -> mgpu.FragmentedLayout:
    def check_no_args():
      if args or kwargs:
        raise ValueError(f"Can't instantiate {self} with arguments.")

    match self:
      case Layout.WGMMA_TRANSPOSED:
        check_no_args()
        return mgpu.WGMMA_TRANSPOSED_LAYOUT
      case Layout.WGMMA:
        check_no_args()
        return mgpu.WGMMA_LAYOUT
      case Layout.WGMMA_ROW:
        check_no_args()
        return mgpu.WGMMA_ROW_LAYOUT
      case Layout.WGMMA_COL:
        check_no_args()
        return mgpu.WGMMA_COL_LAYOUT
      case Layout.WG_SPLAT:
        return mgpu.WGSplatFragLayout(*args, **kwargs)  # pytype: disable=missing-parameter
      case Layout.WG_STRIDED:
        return mgpu.WGStridedFragLayout(*args, **kwargs)  # pytype: disable=missing-parameter

@dataclasses.dataclass(frozen=True)
class ParameterizedLayout:
  layout_cls: Layout
  args: Sequence[Any]
  kwargs: Any

  def to_mgpu(self) -> mgpu.FragmentedLayout:
    return self.layout_cls.to_mgpu(*self.args, **self.kwargs)


layout_cast_p = jax_core.Primitive("layout_cast")


@layout_cast_p.def_abstract_eval
def _layout_cast_abstract_eval(x, new_layout):
  del new_layout  # Unused.
  return x


@lowering.register_lowering_rule(layout_cast_p, mgpu.LoweringSemantics.Lane)
def _layout_cast_lowering(ctx: lowering.LoweringRuleContext, x, *, new_layout):
  del ctx  # Unused.
  return x.to_layout(new_layout.to_mgpu())


@lowering.register_lowering_rule(layout_cast_p, mgpu.LoweringSemantics.Warpgroup)
def _layout_cast_lowering_wg(
    ctx: lowering.LoweringRuleContext, x, *, new_layout
):
  del ctx  # Unused.
  return mgpu.dialect.layout_cast(x, mgpu.to_layout_attr(new_layout.to_mgpu()))


def layout_cast(x: Any, new_layout: Layout | ParameterizedLayout):
  """Casts the layout of the given array."""
  return layout_cast_p.bind(x, new_layout=new_layout)


set_max_registers_p = jax_core.Primitive("set_max_registers_p")
set_max_registers_p.multiple_results = True


@set_max_registers_p.def_effectful_abstract_eval
def _set_max_registers_abstract_eval(n, *, action):
  del n, action  # Unused.
  return (), {gpu_core._memory_effect}


@lowering.register_lowering_rule(
    set_max_registers_p, mgpu.LoweringSemantics.Lane)
@lowering.register_lowering_rule(
    set_max_registers_p, mgpu.LoweringSemantics.Warpgroup)
def _set_max_registers_lowering(
    ctx: lowering.LoweringRuleContext, n, *, action
):
  del ctx
  nvvm_dialect.setmaxregister(
      n,
      nvvm_dialect.SetMaxRegisterAction.increase
      if action == "increase"
      else nvvm_dialect.SetMaxRegisterAction.decrease,
  )
  return ()


def set_max_registers(n: int, *, action: Literal["increase", "decrease"]):
  """Sets the maximum number of registers owned by a warp."""
  set_max_registers_p.bind(n, action=action)


commit_smem_p = jax_core.Primitive("commit_smem")
commit_smem_p.multiple_results = True


@commit_smem_p.def_effectful_abstract_eval
def _commit_smem_abstract_eval():
  return (), {gpu_core._memory_effect}


@lowering.register_lowering_rule(commit_smem_p, mgpu.LoweringSemantics.Lane)
@lowering.register_lowering_rule(
    commit_smem_p, mgpu.LoweringSemantics.Warpgroup)
def _commit_smem_lowering(ctx: lowering.LoweringRuleContext):
  # TODO(bchetioui): add primitive for commit smem to mosaic_gpu dialect.
  mgpu.commit_shared()
  return ()


def commit_smem():
  """Commits all writes to SMEM, making them visible to loads, TMA and WGMMA."""
  commit_smem_p.bind()


broadcasted_iota_p = jax_core.Primitive("broadcasted_iota")

@broadcasted_iota_p.def_abstract_eval
def _broadcasted_iota_abstract_eval(dtype, shape, dimension, layout):
  del layout, dimension
  return jax_core.ShapedArray(shape, dtype)


@lowering.register_lowering_rule(
    broadcasted_iota_p, mgpu.LoweringSemantics.Lane)
def _broadcasted_iota_lowering(
    ctx: lowering.LoweringRuleContext, dtype, shape, dimension, layout
):
  del ctx  # Unused.
  mlir_dtype = mgpu_utils.dtype_to_ir_type(dtype)
  if ir.FloatType.isinstance(mlir_dtype):
    i32 = ir.IntegerType.get_signless(32)
    cast = lambda x: arith_dialect.uitofp(
        mlir_dtype, arith_dialect.index_cast(i32, x)
    )
  else:
    cast = lambda x: arith_dialect.index_cast(mlir_dtype, x)
  is_signed = mgpu_utils.is_signed(dtype)
  return mgpu.FragmentedArray.splat(
      llvm_dialect.mlir_undef(mlir_dtype),
      shape,
      layout.to_mgpu(),
      is_signed=is_signed,
  ).foreach(
      lambda _, idx: cast(idx[dimension]),
      create_array=True,
      is_signed=is_signed,
  )


def broadcasted_iota(
    dtype: jax.typing.DTypeLike,
    shape: Sequence[int],
    dimension: int,
    *,
    layout: Layout | None = None,
) -> jax.Array:
  return broadcasted_iota_p.bind(
      dtype=jnp.dtype(dtype), shape=shape, dimension=dimension, layout=layout
  )


jaxpr_call_p = jax_core.Primitive("jaxpr_call")
jaxpr_call_p.multiple_results = True


@jaxpr_call_p.def_abstract_eval
def _jaxpr_call_abstract_eval(*args, jaxpr: jax_core.Jaxpr, **params):
  del args, params  # Unused.
  return [v.aval for v in jaxpr.outvars]


def _jaxpr_call_pp_eqn(
    eqn: jax_core.JaxprEqn,
    context: jax_core.JaxprPpContext,
    settings: jax_core.JaxprPpSettings,
):
  flat_args = eqn.invars
  ref_treedefs = eqn.params["ref_treedefs"]
  flat_refs, _ = util.split_list(
      flat_args, [sum(treedef.num_leaves for treedef in ref_treedefs)]
  )
  flat_refs = util.split_list(
      flat_refs,
      [treedef.num_leaves for treedef in ref_treedefs[: len(ref_treedefs) - 1]],
  )
  trailer = []
  for treedef, flat_ref in zip(ref_treedefs, flat_refs):
    ref = treedef.unflatten(flat_ref)
    transforms = []
    if isinstance(ref, tuple):
      ref, transforms = ref
    trailer.append(pp.text(" "))
    trailer.append(state_primitives.pp_ref_transforms(context, ref, transforms))
  return pp.concat([
      pp.text("jaxpr_call"),
      pp.text("["),
      jax_core.pp_kv_pair("jaxpr", eqn.params["jaxpr"], context, settings),
      pp.text("]"),
      pp.concat(trailer),
  ])


jax_core.pp_eqn_rules[jaxpr_call_p] = _jaxpr_call_pp_eqn


@lowering.register_lowering_rule(jaxpr_call_p, mgpu.LoweringSemantics.Lane)
@lowering.register_lowering_rule(jaxpr_call_p, mgpu.LoweringSemantics.Warpgroup)
def _jaxpr_call_lowering_rule(
    ctx: lowering.LoweringRuleContext,
    *flat_args,
    jaxpr: jax_core.Jaxpr,
    ref_treedefs,
    program_ids_treedef,
):
  args = []
  flat_refs, flat_program_ids = util.split_list(
      flat_args, [sum(treedef.num_leaves for treedef in ref_treedefs)]
  )
  flat_refs = util.split_list(
      flat_refs,
      [treedef.num_leaves for treedef in ref_treedefs[: len(ref_treedefs) - 1]],
  )
  for treedef, flat_ref in zip(ref_treedefs, flat_refs):
    ref = treedef.unflatten(flat_ref)
    if isinstance(ref, tuple):
      ref, transforms = ref
      # We ignore other transforms here, because they are already embedded
      # in the jaxpr.
      ref, _ = lowering._handle_transforms(
          ref, transforms, handle_reshapes=False, handle_transposes=False
      )
    args.append(ref)
  program_ids = program_ids_treedef.unflatten(flat_program_ids)
  for axis, pid in enumerate(program_ids):
    if pid is not None:
      continue
    program_ids[axis] = lowering._program_id(axis, ctx.module_ctx.squashed_dims)
  new_module_ctx = dataclasses.replace(ctx.module_ctx, program_ids=program_ids)
  return lowering.lower_jaxpr_to_mosaic_gpu(
      new_module_ctx, ctx.launch_ctx, jaxpr, args
  )


@lowering._register_resource_estimator(jaxpr_call_p)
def _jaxpr_call_resource_estimator(
    ctx: lowering.ResourceEstimatorContext,
    *args,
    jaxpr: jax_core.Jaxpr,
    **params,
):
  del args, params  # Unused.
  return lowering._estimate_resources(ctx, jaxpr)


@discharge.register_partial_discharge_rule(jaxpr_call_p)
def _jaxpr_call_discharge(
    flat_should_discharge,
    in_avals,
    out_avals,
    *flat_args,
    jaxpr,
    ref_treedefs,
    program_ids_treedef,
):
  del in_avals, out_avals  # Unused.
  flat_should_discharge = util.split_list(
      flat_should_discharge,
      [treedef.num_leaves for treedef in ref_treedefs[: len(ref_treedefs) - 1]],
  )
  should_discharge = [*map(any, flat_should_discharge)]
  discharged_jaxpr, discharged_consts = discharge.discharge_state(
      jaxpr, (), should_discharge=should_discharge
  )
  assert not discharged_consts
  outs = jaxpr_call_p.bind(
      *flat_args,
      jaxpr=discharged_jaxpr,
      ref_treedefs=ref_treedefs,
      program_ids_treedef=program_ids_treedef,
  )
  discharged_outs_it = iter(outs[len(jaxpr.outvars) :])
  new_in_vals = tuple(
      itertools.chain.from_iterable(
          [next(discharged_outs_it) if discharged else None]
          * ref_treedefs[idx].num_leaves
          for idx, discharged in enumerate(should_discharge)
      )
  ) + (None,) * program_ids_treedef.num_leaves
  return new_in_vals, outs[: len(jaxpr.outvars)]


def jaxpr_call(
    jaxpr: jax_core.Jaxpr,
    *refs: pallas_core.AbstractMemoryRef | state_types.TransformedRef,
    program_ids: Sequence[jax.Array | None],
) -> Sequence[jax.Array]:
  """Internal primitive for calling a kernel jaxpr inside ``emit_pipeline``.

  This is *not* a general purpose primitive. In particular, it assumes that
  the transformed references have been indexed.

  Args:
    jaxpr: The jaxpr to call.
    *refs: The references to pass into the jaxpr.
    program_ids: The loop-bound program IDs to pass into the jaxpr, or None
      if the program ID corresponds to a parallel dimension.

  Returns:
    The outputs of the jaxpr.
  """
  assert not jaxpr.outvars
  flat_refs = []
  ref_treedefs = []
  ref: Any
  for ref in refs:
    if isinstance(ref, state_types.TransformedRef):
      if not isinstance(ref.transforms[-1], indexing.NDIndexer):
        raise ValueError(
            "TransformedRef must have been indexed before passing into"
            f" jaxpr_call. Got {ref}."
        )
      ref = (ref.ref, ref.transforms)
    flat_ref, treedef = jax.tree.flatten(ref)
    flat_refs.extend(flat_ref)
    ref_treedefs.append(treedef)
  flat_program_ids, program_ids_treedef = jax.tree.flatten(program_ids)
  return jaxpr_call_p.bind(
      *flat_refs,
      *flat_program_ids,
      jaxpr=jaxpr,
      ref_treedefs=ref_treedefs,
      program_ids_treedef=program_ids_treedef,
  )


@dataclasses.dataclass(frozen=True)
class GPUShapeDtypeStruct:
  shape: tuple[int, ...]
  dtype: jnp.dtype
  layout: ParameterizedLayout | Layout


inline_mgpu_p = jax_core.Primitive("inline_mgpu_p")
inline_mgpu_p.multiple_results = True


@dataclasses.dataclass(frozen=True)
class RefType:
  ...


def inline_mgpu(arg_types=(), return_type=None):
  """Decorate a function that inlines mgpu code.

  Arguments provided to the decorated function may be Pallas
  references or array values. The body will accept the corresponding
  mgpu values.

  The decorated function may return a tree of `FragmentedArray`s.

  ```
  layout = plgpu.Layout.WG_STRIDED(x_ref.shape, vec_size=4)
  @plgpu.inline_mgpu(
      arg_types=(plgpu.RefType(),),
      return_type=plgpu.GPUShapeDtypeStruct(
          (128, 128), dtype, layout=layout
      ),
  )
  def foo(ctx, smem_ref):
    del ctx
    x = mgpu.FragmentedArray.load_tiled(smem_ref, )
    y = mgpu.FragmentedArray.splat(
        mgpu.c(1, x.mlir_dtype), shape=x.shape, layout=x.layout
    )
    return (x + y)

  arr = foo(smem_ref)
  ```

  Args:

    arg_types: a sequence of pytrees where the leaves are `RefType` or
      `Layout` for references or arrays respectively as the return
      type.

    return_type: A pytree where the leaves are `GPUShapeDtypeStruct`
      represeinting the arrays returned by the decorated function.

  Returns:
    A decorator that creates a function that inlines mgpu code.

  """
  flat_arg_types, treedef_ty = jax.tree.flatten(tuple(arg_types))
  flat_ret_ty, pytree_ret_ty = jax.tree.flatten(return_type)
  if return_type and not all(isinstance(r, GPUShapeDtypeStruct) for r in flat_ret_ty):
    raise ValueError(
        "inline_mgpu_p only supports GPUShapeDtypeStructx return types."
    )
  if not all(isinstance(r, (Layout, ParameterizedLayout, RefType)) for r in flat_arg_types):
    raise ValueError(
        "inline_mgpu_p only supports only Layout, ParameterizedLayout and"
        " RefType arg types."
    )

  def inner(f):
    def wrapper(*args):
      flat_args, treedef = jax.tree.flatten(tuple(args))
      if treedef != treedef_ty:
        raise ValueError(f"Mismatched type shape: {treedef} != {treedef_ty}")

      # Strip the transforms from the refs since they will be recorded in
      # the types.
      raw_refs_flat_args = []
      for a, t in zip(flat_args, flat_arg_types):
        def traced_ty(ty):
          return isinstance(a, jax_core.Tracer) and isinstance(a.aval, ty)

        if isinstance(t, ParameterizedLayout) and traced_ty(jax_core.ShapedArray):
          raw_refs_flat_args.append(a)
        elif isinstance(t, RefType) and traced_ty(_Ref):
          ref, transforms = a, ()
          if isinstance(a, state_types.TransformedRef):
            ref, transforms = ref.ref, ref.transforms

          raw_refs_flat_args.append(ref)
          if transforms:
            raise NotImplementedError("Transformed refs (or types) are not supported.")
        else:
          raise ValueError(f"Mismatched type: {a, t}")

      flat_ret = inline_mgpu_p.bind(
          *flat_args,
          args_treedef=treedef,
          flat_ret_ty=flat_ret_ty,
          pytree_ret_ty=pytree_ret_ty,
          flat_arg_types=flat_arg_types,
          mgpu_fn=f,
      )
      return jax.tree.unflatten(pytree_ret_ty, flat_ret)
    return wrapper

  return inner


@inline_mgpu_p.def_effectful_abstract_eval
def _inline_mgpu_abstract_eval(
    *flat_args,
    args_treedef,
    flat_arg_types,
    flat_ret_ty,
    pytree_ret_ty,
    mgpu_fn,
):
  del args_treedef, flat_arg_types, pytree_ret_ty, mgpu_fn  # Unused.
  aval_return = tuple(
      jax_core.ShapedArray(x.shape, x.dtype) for x in flat_ret_ty
  )
  # TODO(cperivol): Let the user set the effects.
  return aval_return, {
      gpu_core._wgmma_pipeline_effect,
      gpu_core._memory_effect,
      *itertools.chain.from_iterable(
          (state.ReadEffect(i), state.WriteEffect(i))
          for i, r in enumerate(flat_args)
          if isinstance(r, pallas_core.AbstractMemoryRef)
      ),
  }


@discharge.register_partial_discharge_rule(inline_mgpu_p)
def _inline_mgpu_discharge(*args, **kwargs):
  del args, kwargs
  raise NotImplementedError("inline_mgpu_p does not support discharge.")


def _type_check_mgpu(v, ty):
  match (ty, v):
    case (RefType(), ir.Value()) if ir.MemRefType.isinstance(v.type):
      pass
    case (GPUShapeDtypeStruct(), mgpu.FragmentedArray()):
      mlir_dtype = mgpu_utils.dtype_to_ir_type(ty.dtype)
      if v.mlir_dtype != mlir_dtype or ty.shape != v.shape or v.layout != ty.layout.to_mgpu():
        raise ValueError(f"Array type mismatch at {v} != {ty}.")
    case (Layout() , mgpu.FragmentedArray()) | (ParameterizedLayout(), mgpu.FragmentedArray()):
      if ty.to_mgpu() != v.layout:
        raise ValueError(f"Unexpected layout for {v} (expected: {ty})")
    case _:
      raise ValueError(f"Unexpected type {ty} for value {v}")


@lowering.register_lowering_rule(inline_mgpu_p, mgpu.LoweringSemantics.Lane)
def _inline_mgpu_lowering_rule(
    ctx: lowering.LoweringRuleContext,
    *flat_args,
    mgpu_fn: Callable[..., Any],
    flat_arg_types,
    flat_ret_ty,
    pytree_ret_ty,
    args_treedef,
):
  for a, t in zip(flat_args, flat_arg_types):
    _type_check_mgpu(a, t)

  args = jax.tree.unflatten(args_treedef, flat_args)
  ret = mgpu_fn(ctx.launch_ctx, *args)
  ret_leaves, ret_tree = jax.tree.flatten(
      ret, is_leaf=lambda x: isinstance(x, mgpu.FragmentedArray)
  )

  if ret_tree != pytree_ret_ty:
    return_type = jax.tree.unflatten(pytree_ret_ty, flat_ret_ty)
    raise ValueError(
        f"inline_mgpu_p return type tree mismatch: {ret} != {return_type}"
    )

  for ty, r in zip(flat_ret_ty, ret_leaves):
    _type_check_mgpu(r, ty)

  return ret_leaves
