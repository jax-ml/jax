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
import functools
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
from jax._src.lib.mlir.dialects import gpu as gpu_dialect
from jax._src.lib.mlir.dialects import nvvm as nvvm_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.pallas.mosaic_gpu.core import state_types
from jax._src.state import discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.mosaic.gpu import tcgen05
import jax.numpy as jnp


WARP_SIZE = 32
WARPGROUP_SIZE = 128


_Ref = state.AbstractRef | state_types.TransformedRef
Layout = gpu_core.Layout
ParameterizedLayout = gpu_core.ParameterizedLayout


def _check_ref(
    aval: object, name: str, memory_space: gpu_core.MemorySpace
) -> None:
  if not isinstance(aval, state_types.AbstractRef):
    raise TypeError(f"{name} must be a reference, got {aval}")
  aval_memory_space = getattr(aval, "memory_space", None) or gpu_core.GMEM
  if aval_memory_space is not memory_space:
    raise ValueError(
        f"{name} must be a {memory_space.name.upper()} reference, got {aval}"
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
  src, src_transforms = lowering._handle_transforms(
      ctx, src, src_transforms, handle_transposes=False
  )
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
  if copy_params.get("gmem_peer_id", None) is not None:
    raise NotImplementedError(
        "GMEM refs with peer ids are not supported in warpgroup lowering."
    )
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
  peer_id = None
  indexers = []
  for transform in transforms:
    if isinstance(transform, gpu_core.PeerMemRef):
      if transform.device_id_type != pallas_primitives.DeviceIdType.LOGICAL:
        raise NotImplementedError(
            "Only logical device ids are supported for GMEM refs."
        )
      peer_id = lowering._ensure_ir_value(transform.device_id, jnp.int32)
      continue
    elif isinstance(transform, indexing.NDIndexer):
      indexers.append(transform)
    else:
      raise NotImplementedError(
          "Non-indexing transforms on GMEM refs are not implemented.")
  indexer = lowering.merge_indexers(indexers)
  return dict(
      gmem_slice=lowering._ndindexer_indices(indexer),
      gmem_peer_id=peer_id,
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
    partitioned_axis,
    for_warpgroup: bool = True,
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
  dst, dst_transforms = lowering._handle_transforms(
      ctx, dst, dst_transforms, handle_transposes=False
  )
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
    if for_warpgroup:
      # We arrive uniformly from each thread in the WG, so we need to divide the
      # number of bytes by the number of threads in the WG.
      # TODO: apaszke - Relax this. We can just select the WG leader and have it
      # arrive with the whole transfer size, while everyone else arrives with 0.
      # But we should continue using this scheme as it's likely to be faster.
      bytes //= WARPGROUP_SIZE
      if collective and partitioned_axis is not None:
        raise NotImplementedError(
            "Collective partitioned copies not implemented."
        )
      if ctx.module_ctx.auto_barriers:
        mgpu.warpgroup_barrier()  # Make sure all reads have completed.
      barrier.arrive_expect_tx(bytes)
    else:
      # In Warp-level lowering, we arrive on each CUDA thread in a warp, but
      # the barrier still expects a full 128 arrivals so we arrive 4 times
      # on each CUDA thread instead.
      # TODO(justinfu): The arrival counts are wrong if called outside of a
      # single warp. Figure out how to guard against this in user code.
      bytes = bytes // WARP_SIZE
      if collective and partitioned_axis is not None:
        if len(collective) != 1:
          raise ValueError(
              f"Expected exactly one collective axis, got {collective_axes=}"
          )
        if math.prod(ctx.launch_ctx.cluster_size) != 2:
          raise NotImplementedError(
              "Partitioned loads only supported for clusters of size 2"
          )
        # Bytes is the destination size, which is only half of the total
        # size of the partitioned transfer so we need to double it.
        bytes *= 2
        first_block = arith_dialect.cmpi(
            arith_dialect.CmpIPredicate.eq,
            ctx.launch_ctx.cluster_idx(collective[0]),
            mgpu.c(0, ir.IndexType.get()),
        )
        with mgpu.when(first_block):
          barrier.arrive(arrival_count=3, can_complete=False)
          barrier.arrive_expect_tx(bytes)
      else:
        barrier.arrive(arrival_count=3, can_complete=False)
        barrier.arrive_expect_tx(bytes)

    ctx.launch_ctx.async_copy(
        src_ref=src,
        dst_ref=dst,
        barrier=barrier,
        arrive=False,
        predicate=ctx.module_ctx.single_lane_predicate,
        collective=collective,
        partitioned=partitioned_axis,
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
  if copy_params.get("gmem_peer_id", None) is not None:
    raise NotImplementedError(
        "GMEM refs with peer ids are not supported in warpgroup lowering."
    )
  barrier_ref = barrier.as_barrier_memref()
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


lowering.register_lowering_rule(
    copy_gmem_to_smem_p,
    mgpu.LoweringSemantics.Lane,
    primitive_semantics=gpu_core.PrimitiveSemantics.Warp,
)(functools.partial(_copy_gmem_to_smem_lowering, for_warpgroup=False))


def copy_gmem_to_smem(
    src: _Ref,
    dst: _Ref,
    barrier: _Ref,
    *,
    collective_axes: str | tuple[str, ...] | None = None,
    partitioned_axis: int | None = None,
) -> None:
  """Asynchronously copies a GMEM reference to a SMEM reference.

  If collective_axes is specified, this performs a multicast copy where
  all CUDA blocks that share the same index along the collective axis
  receive a copy of the same block of data loaded from `dst` to `src`.

  If both collective_axes and partitioned_axis are specified, this will perform
  a partitioned collective copy where each block in the cluster will receive
  a tile of `transfer_size // cluster_size` data from the `src` Ref.
  For example, if `src` has a shape of (256, 256) and a partitioned
  copy is performed along axis 0 with cluster size 2, then the first block will
  receive `src[0:128, :]` and the second will receive `src[128:256, :]`.
  NOTE: Only the first block in the cluster will arrive on the barrier,
  and an additional cluster barrier is necessary to ensure that all blocks in
  the cluster have finished the copy.

  Args:
    src: The source Ref. Must be in GMEM.
    dst: The destination Ref. Must be in SMEM.
    barrier: The barrier to use for tracking completion of the copy.
    collective_axes: The collective axes to use for the copy.
    partitioned_axis: Indicates which array axis along the src/dst Refs to
     partition across during a partitioned collective copy. Requires
     collective_axes to also be specified.

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
      partitioned_axis=partitioned_axis,
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
      raise ValueError("Barrier does not support arbitrary transforms")


barrier_arrive_p = jax_core.Primitive("barrier_arrive")
barrier_arrive_p.multiple_results = True


@barrier_arrive_p.def_effectful_abstract_eval
def _barrier_arrive_abstract_eval(barrier, *args, **params):
  del args, params  # Unused.
  _check_ref(barrier, "barrier", gpu_core.SMEM)
  if getattr(barrier.inner_aval.dtype, "orders_tensor_core", False):
    raise ValueError("Cannot arrive on a tensor core barrier.")
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


def barrier_arrive(barrier: state.AbstractRef) -> None:
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
@lowering.register_lowering_rule(
    barrier_wait_p,
    mgpu.LoweringSemantics.Lane,
    gpu_core.PrimitiveSemantics.Warp,
)
@lowering.register_lowering_rule(
    barrier_wait_p, mgpu.LoweringSemantics.Warpgroup
)
def _barrier_wait_lowering(
    ctx: lowering.LoweringRuleContext,
    barrier,
    *flat_transforms,
    transforms_treedef,
):
  barrier_aval = ctx.avals_in[0]
  transforms = transforms_treedef.unflatten(flat_transforms)
  indexer = _extract_barrier_indexer(transforms)
  orders_tensor_core = getattr(
      barrier_aval.inner_aval.dtype, "orders_tensor_core", False)
  if indexer is not None:
    barrier = barrier.__getitem__(*map(lowering._as_index, indexer.indices))
  barrier.wait(orders_tensor_core=orders_tensor_core)
  return ()


def barrier_wait(barrier: state.AbstractRef) -> None:
  """Waits on the given barrier."""
  barrier, transforms = state_primitives.get_ref_and_transforms(
      barrier, None, "barrier_wait", force_trailing_indexer=False,
  )
  flat_transforms, transforms_treedef = tree_util.tree_flatten(transforms)
  barrier_wait_p.bind(
      barrier, *flat_transforms, transforms_treedef=transforms_treedef,
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
  """Commits all issued but uncommitted SMEM->GMEM copies to a group."""
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

  acc_transforms_leaves: list
  if isinstance(acc, pallas_core.TransformedRef):
    acc_transforms_leaves, acc_transforms_tree = jax.tree.flatten(acc.transforms)
    acc = acc.ref
  else:
    acc_transforms_leaves, acc_transforms_tree = [], None

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
      *acc_transforms_leaves,
      *a_transforms_leaves,
      *b_transforms_leaves,
      acc_transforms_tree=acc_transforms_tree,
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
  transform_treedefs = [
      eqn.params["acc_transforms_tree"],
      eqn.params["a_transforms_tree"],
      eqn.params["b_transforms_tree"],
  ]
  transform_leaves = util.split_list(
      leaves, [getattr(tree, "num_leaves", 0) for tree in transform_treedefs]
  )
  acc_transforms, a_transforms, b_transforms = (
      () if treedef is None else treedef.unflatten(leaves)
      for treedef, leaves in zip(transform_treedefs, transform_leaves)
  )
  return pp.concat([
      pp.text("wgmma_ref"),
      pp.text(" "),
      state_primitives.pp_ref_transforms(context, acc, acc_transforms),
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
    acc_transforms_tree,
    a_transforms_tree,
    b_transforms_tree,
):
  lhs_swizzle: int | None = None
  transform_treedefs = [
      acc_transforms_tree, a_transforms_tree, b_transforms_tree
  ]
  transform_leaves = util.split_list(
      transforms_leaves, [getattr(tree, "num_leaves", 0) for tree in transform_treedefs]
  )
  acc_transforms, a_transforms, b_transforms = (
      None if treedef is None else treedef.unflatten(leaves)
      for treedef, leaves in zip(transform_treedefs, transform_leaves)
  )

  acc_indices = None
  if acc_transforms is not None:
    if not all(isinstance(t, indexing.NDIndexer) for t in acc_transforms):
      raise ValueError("WGMMA accumulator only supports indexing transforms")
    acc_indexer = lowering.merge_indexers(acc_transforms)
    if acc_indexer.int_indexer_shape:
      raise NotImplementedError("int_indexer_shape non-empty")
    acc_indices = lowering._ndindexer_indices(acc_indexer)

  if a_transforms is not None:
    a, a_transforms = lowering._handle_transforms(
        ctx, a, a_transforms, handle_transposes=False, handle_reshapes=False
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
    a_mlir_dtype = ir.MemRefType(a.type).element_type
    swizzle_elems = lhs_swizzle // mgpu_utils.bytewidth(a_mlir_dtype)
    if tiling != (8, swizzle_elems):
      raise NotImplementedError("WGMMA lhs tiling does not fit swizzle")
  else:
    lhs_transpose = False
    if not isinstance(a, mgpu.FragmentedArray):
      raise ValueError(
          "When WGMMA lhs is passed in as a ref, it must be transformed by"
          " swizzling and tiling appropriately."
      )

  assert b_transforms is not None
  b, b_transforms = lowering._handle_transforms(
      ctx, b, b_transforms, handle_transposes=False, handle_reshapes=False
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
    b_mlir_dtype = ir.MemRefType(b.type).element_type
    swizzle_elems = rhs_swizzle // mgpu_utils.bytewidth(b_mlir_dtype)
    if rhs_swizzle != lhs_swizzle:
      raise NotImplementedError("WGMMA rhs swizzle must match lhs swizzle")
    if rhs_tiling != (8, swizzle_elems):
      raise NotImplementedError("WGMMA rhs tiling does not fit swizzle")

  if lhs_transpose:
    a = mgpu.memref_transpose(a, (1, 0, 3, 2))
  if rhs_transpose:
    b = mgpu.memref_transpose(b, (1, 0, 3, 2))
  acc_in = acc
  if acc_indices is not None:
    acc_in = mgpu.WGMMAAccumulator(_value=acc.value[acc_indices], _sync=False)
  acc_out = mgpu.wgmma(acc_in, a, b, swizzle=rhs_swizzle)
  if acc_indices is not None:
    acc_value = acc.value.copy()
    acc_value[acc_indices] = acc_out.value
    acc_out = mgpu.WGMMAAccumulator(_value=acc_value, _sync=False)
  nvvm_dialect.wgmma_commit_group_sync_aligned()
  return acc_out


@lowering.register_lowering_rule(wgmma_p, mgpu.LoweringSemantics.Warpgroup)
def _wgmma_warpgroup_lowering(
    ctx: lowering.LoweringRuleContext,
    acc,
    a,
    b,
    *transforms_leaves,
    acc_transforms_tree,
    a_transforms_tree,
    b_transforms_tree,
):
  if acc_transforms_tree is not None:
    raise NotImplementedError
  if a_transforms_tree is not None:
    a_transforms_leaves, b_transforms_leaves = util.split_list(
        transforms_leaves, [a_transforms_tree.num_leaves]
    )
    a_transforms = a_transforms_tree.unflatten(a_transforms_leaves)
    a, a_transforms = lowering._handle_transforms(ctx, a, a_transforms)
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
    b, b_transforms = lowering._handle_transforms(ctx, b, b_transforms)
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


# MMA for TensorCore gen 5.
tcgen05_mma_p = jax_core.Primitive("tcgen05_mma")
tcgen05_mma_p.multiple_results = True

def tcgen05_mma(acc: _Ref,
                a: _Ref,
                b: _Ref,
                barrier: _Ref | None = None,
                accumulate: bool | jax.Array = True,
                collective_axis: str | None = None):
  """Asynchronous matrix-multiply accumulate for TensorCore gen 5 (Blackwell).

  If run in collective mode, `acc`, `a` (LHS), and `b` (RHS) should correspond
  to half of the total inputs to the MMA, where `acc` and `a` (LHS) are split
  in half along the rows and `b` (RHS) is split along the columns like so:

   -----------    -----------   -----------
   |  ACC1   |    |  LHS1   |   |    |    |
   ----------- += ----------- @ |RHS1|RHS2|
   |  ACC2   |    |  LHS2   |   |    |    |
   -----------    -----------   -----------

  Args:
    acc: The accumulator. Must be a TMEM Ref.
    a: The left-hand side. Must be a TMEM/SMEM Ref.
    b: The right-hand side. Must be an SMEM Ref.
    barrier: Optional barrier Ref for synchronizing with the tensor core.
      Must have orders_tensor_core set to True. If not specified, the MMA
      completion should be explicitly observed by calling
      `tcgen05_commit_arrive`
    accumulate: Whether to accumulate into acc or overwrite it.
    collective_axis: The name of the cluster axis along which to perform
      a collective MMA. The cluster axis should have a size of exactly 2,
      and must be on the minormost cluster axis.
  """
  acc_m, acc_n = acc.shape
  lhs_m, lhs_k = a.shape
  rhs_k, rhs_n = b.shape
  if collective_axis is not None:
    acc_n /= 2
  if acc_m != lhs_m:
    raise ValueError(
        f"Accumulator and LHS have incompatible shapes. Accumulator: {acc.shape}. LHS: {a.shape}.")
  if acc_n != rhs_n:
    raise ValueError(
        f"Accumulator and RHS have incompatible shapes. Accumulator: {acc.shape}. RHS: {b.shape}.")
  if lhs_k != rhs_k:
    raise ValueError(
        f"LHS and RHS have incompatible shapes. LHS: {a.shape}. RHS: {b.shape}.")

  if isinstance(acc, pallas_core.TransformedRef):
    acc_transforms_leaves, acc_transforms_tree = jax.tree.flatten(
        acc.transforms)
    acc = acc.ref
  else:
    acc_transforms_leaves, acc_transforms_tree = [], None

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

  if isinstance(barrier, pallas_core.TransformedRef):
    barrier_transforms_leaves, barrier_transforms_tree = jax.tree.flatten(
        barrier.transforms
    )
    barrier = barrier.ref
  else:
    barrier_transforms_leaves, barrier_transforms_tree = [], None

  if barrier is not None:
    barrier_ref = [barrier]
    arrive = True
  else:
    barrier_ref = []
    arrive = False

  tcgen05_mma_p.bind(acc, a, b, accumulate, *barrier_ref,
                     *acc_transforms_leaves, *a_transforms_leaves,
                     *b_transforms_leaves,
                     *barrier_transforms_leaves,
                     acc_transforms_tree=acc_transforms_tree,
                     a_transforms_tree=a_transforms_tree,
                     b_transforms_tree=b_transforms_tree,
                     barrier_transforms_tree=barrier_transforms_tree,
                     collective_axis=collective_axis,
                     arrive=arrive)


@tcgen05_mma_p.def_abstract_eval
def _tcgen05_mma_abstract_eval(acc, a, b, accumulate,
                               *barrier_and_transforms_leaves,
                               acc_transforms_tree, a_transforms_tree,
                               b_transforms_tree,
                               barrier_transforms_tree,
                               collective_axis,
                               arrive):
  del (accumulate, acc_transforms_tree,
       a_transforms_tree, b_transforms_tree, barrier_transforms_tree)

  if acc.memory_space != gpu_core.TMEM:
    raise ValueError("Accumulator must be a TMEM Ref.")
  if a.memory_space not in (gpu_core.SMEM, gpu_core.TMEM):
    raise ValueError("LHS must be a TMEM/SMEM Ref.")
  if b.memory_space != gpu_core.SMEM:
    raise ValueError("RHS must be an SMEM Ref.")

  if collective_axis is not None:
    # TODO(justinfu): If under a core_map, the avals for acc/a
    # become normal MemRefs so we cannot check if they are collective.
    # Figure out a way to fix this.
    if isinstance(acc, gpu_core.AbstractTMEMRef) and not acc.collective:
      raise ValueError(
          "Accumulator Ref must be collective if collective_axis is set.")
    if isinstance(a, gpu_core.AbstractTMEMRef) and not a.collective:
      raise ValueError(
          "LHS Ref must be collective if collective_axis is set.")

  if arrive:
    barrier = barrier_and_transforms_leaves[0]
    orders_tensor_core = getattr(
        barrier.inner_aval.dtype, "orders_tensor_core", False)
    if not orders_tensor_core:
      raise ValueError("MMA barrier must have orders_tensor_core set to True.")

  return []


@lowering.register_lowering_rule(tcgen05_mma_p, *gpu_core.LANExWG_SEMANTICS)
@lowering.register_lowering_rule(tcgen05_mma_p, *gpu_core.LANExWARP_SEMANTICS)
def _tcgen05_mma_lowering(
    ctx: lowering.LoweringRuleContext,
    acc: tcgen05.TMEMRef,
    a_ref,
    b_ref,
    accumulate: bool | ir.Value,
    *barrier_and_transforms_leaves,
    acc_transforms_tree,
    a_transforms_tree,
    b_transforms_tree,
    barrier_transforms_tree,
    collective_axis,
    arrive,
):
  _, a_aval, b_aval, *_ = ctx.avals_in
  lhs_swizzle: int | None = None
  lhs_transpose: bool = False
  if arrive:
    barrier_ref, *transforms_leaves = barrier_and_transforms_leaves
  else:
    barrier_ref = None
    transforms_leaves = barrier_and_transforms_leaves  # type: ignore[assignment]

  transforms_trees = (
      acc_transforms_tree,
      a_transforms_tree,
      b_transforms_tree,
      barrier_transforms_tree,
  )
  (acc_transforms_leaves, a_transforms_leaves, b_transforms_leaves, barrier_transforms_leaves, _) = (
      util.split_list(
          transforms_leaves,
          [getattr(tree, "num_leaves", 0) for tree in transforms_trees],
      )
  )

  if acc_transforms_tree is not None:
    acc_transforms = acc_transforms_tree.unflatten(acc_transforms_leaves)
    acc, acc_transforms = lowering._handle_transforms(ctx, acc, acc_transforms)
    if acc_transforms:
      raise NotImplementedError(
          f"Unsupported transforms: {acc_transforms}."
      )

  if a_transforms_tree is not None:
    a_transforms = a_transforms_tree.unflatten(a_transforms_leaves)
    a_dtype = lowering._transform_dtype(a_aval.dtype, a_transforms)
    a_ref, a_transforms = lowering._handle_transforms(
        ctx, a_ref, a_transforms, handle_transposes=False, handle_reshapes=True
    )
    match a_transforms:
      case (gpu_core.UnswizzleRef(lhs_swizzle), gpu_core.UntileRef(lhs_tiling)):
        lhs_transpose = False
      case (
          gpu_core.UnswizzleRef(lhs_swizzle),
          gpu_core.UntileRef(lhs_tiling),
          gpu_core.TransposeRef((1, 0)),
      ):
        lhs_transpose = True
      case () if isinstance(a_ref, tcgen05.TMEMRef):
        lhs_tiling = None  # type: ignore
      case _:
        raise NotImplementedError(
            f"Unsupported transforms: {a_transforms}."
        )
    if not isinstance(a_ref, tcgen05.TMEMRef):
      swizzle_elems = lhs_swizzle // a_dtype.itemsize  # type: ignore
      if lhs_tiling != (8, swizzle_elems):
        raise ValueError("MMA lhs tiling does not fit swizzle. "
                        f"{lhs_tiling=} expected={(8, swizzle_elems)}")

  assert b_transforms_tree is not None
  b_transforms = b_transforms_tree.unflatten(b_transforms_leaves)
  b_dtype = lowering._transform_dtype(b_aval.dtype, b_transforms)
  b_ref, b_transforms = lowering._handle_transforms(
      ctx, b_ref, b_transforms, handle_transposes=False, handle_reshapes=True
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
    case _:
      raise NotImplementedError(
          f"Unsupported transforms: {b_transforms}."
      )
  swizzle_elems = rhs_swizzle // b_dtype.itemsize
  if rhs_tiling != (8, swizzle_elems):
    raise ValueError(
        "MMA rhs tiling does not fit swizzle"
        f" {rhs_tiling=} expected={(8, swizzle_elems)}"
    )

  if barrier_transforms_tree is not None and barrier_ref is not None:
    barrier_transforms = barrier_transforms_tree.unflatten(
        barrier_transforms_leaves
    )
    indexer = _extract_barrier_indexer(barrier_transforms)
    if indexer is not None:
      barrier_ref = barrier_ref.__getitem__(
          *map(lowering._as_index, indexer.indices)
      )

  if lhs_swizzle is None:
    lhs_swizzle = rhs_swizzle
  elif rhs_swizzle != lhs_swizzle:
    raise ValueError("MMA rhs swizzle must match lhs swizzle."
                      f" {lhs_swizzle=} {rhs_swizzle=}")
  if lhs_transpose:
    if isinstance(a_ref, tcgen05.TMEMRef):
      raise ValueError("TMEM transpose not allowed.")
    a_ref = mgpu.memref_transpose(a_ref, (1, 0, 3, 2))
  if rhs_transpose:
    b_ref = mgpu.memref_transpose(b_ref, (1, 0, 3, 2))
  if isinstance(accumulate, bool):
    accumulate = mgpu.c(accumulate, ir.IntegerType.get_signless(1))
  elif isinstance(accumulate, mgpu.FragmentedArray):
    accumulate = accumulate.registers.item()
    assert isinstance(accumulate, ir.Value)

  predicate = ctx.module_ctx.single_lane_predicate
  if collective_axis is not None:
    is_leader_block = _collective_mma_predicate(ctx, collective_axis)
    predicate = arith_dialect.andi(predicate, is_leader_block)
    collective = True
  else:
    collective = False

  with mgpu.when(predicate):
    tcgen05.mma(
              acc,
              a_ref,
              b_ref,
              a_swizzle=int(lhs_swizzle),
              b_swizzle=int(rhs_swizzle),
              accumulate=accumulate,
              collective=collective,
          )
    if arrive:
      tcgen05.commit_arrive(barrier_ref,
                            collective=collective,
                            ctx=ctx.launch_ctx)
  return []


tcgen05_commit_arrive_p = jax_core.Primitive("tcgen05_commit_arrive")
tcgen05_commit_arrive_p.multiple_results = True


def tcgen05_commit_arrive(barrier: _Ref,
                          collective_axis: str | None = None):
  """Arrive on a Barrier to track completion of a preceding `tcgen05_mma` call.

  Args:
    barrier: Barrier Ref for synchronizing with the tensor core. Must have
      orders_tensor_core set to True.
    collective_axis: The name of the cluster axis along which the
      MMA was performed if it was collective. The cluster axis should have a
      size of exactly 2, and must be on the minormost cluster axis.
  """
  if isinstance(barrier, pallas_core.TransformedRef):
    barrier_transforms_leaves, barrier_transforms_tree = jax.tree.flatten(
        barrier.transforms
    )
    barrier = barrier.ref
  else:
    barrier_transforms_leaves, barrier_transforms_tree = [], None

  tcgen05_commit_arrive_p.bind(
      barrier, *barrier_transforms_leaves,
      barrier_transforms_tree=barrier_transforms_tree,
      collective_axis=collective_axis)


@tcgen05_commit_arrive_p.def_abstract_eval
def _tcgen05_commit_arrive_abstract_eval(barrier,
                               *barrier_transforms_leaves,
                               barrier_transforms_tree,
                               collective_axis):
  del (barrier_transforms_leaves, barrier_transforms_tree, collective_axis)
  orders_tensor_core = getattr(
      barrier.inner_aval.dtype, "orders_tensor_core", False)
  if not orders_tensor_core:
    raise ValueError("MMA barrier must have orders_tensor_core set to True.")
  return []


@lowering.register_lowering_rule(
    tcgen05_commit_arrive_p, *gpu_core.LANExWG_SEMANTICS)
@lowering.register_lowering_rule(
    tcgen05_commit_arrive_p, *gpu_core.LANExWARP_SEMANTICS)
def _tcgen05_commit_arrive_lowering(
    ctx: lowering.LoweringRuleContext,
    barrier_ref: mgpu.BarrierRef,
    *barrier_transforms_leaves,
    barrier_transforms_tree,
    collective_axis,
):
  if barrier_transforms_tree is not None:
    barrier_transforms = barrier_transforms_tree.unflatten(
        barrier_transforms_leaves
    )
    indexer = _extract_barrier_indexer(barrier_transforms)
    if indexer is not None:
      barrier_ref = barrier_ref.__getitem__(
          *map(lowering._as_index, indexer.indices)
      )

  predicate = ctx.module_ctx.single_lane_predicate
  if collective_axis is not None:
    is_leader_block = _collective_mma_predicate(ctx, collective_axis)
    predicate = arith_dialect.andi(predicate, is_leader_block)
    collective = True
  else:
    collective = False

  with mgpu.when(predicate):
    tcgen05.commit_arrive(barrier_ref,
                          collective=collective,
                          ctx=ctx.launch_ctx)
  return []


def _collective_mma_predicate(ctx: lowering.LoweringRuleContext,
                              collective_axis: str) -> ir.Value:
  """Computes a predicate to run only on the leader block."""
  cluster_axis = lowering._resolve_cluster_axis(
      ctx.module_ctx.axis_names, collective_axis)
  if cluster_axis != gpu_dialect.Dimension(0):
    # Note: resolve_cluster_axis checks if axis_names exists.
    assert ctx.module_ctx.axis_names is not None
    if len(ctx.module_ctx.axis_names.cluster) <= 1:
      raise ValueError("No cluster axes found.")
    minormost_cluster_axis = ctx.module_ctx.axis_names.cluster[0]
    raise ValueError(
        "Can only perform collective MMA along minormost cluster axis. "
        f"Got {collective_axis}, expected {minormost_cluster_axis}.")
  index = ir.IndexType.get()
  is_leader_block = arith_dialect.cmpi(
      arith_dialect.CmpIPredicate.eq,
      ctx.launch_ctx.cluster_idx(cluster_axis), mgpu.c(0, index))
  return is_leader_block


commit_tmem_p = jax_core.Primitive("commit_tmem")
commit_tmem_p.multiple_results = True


@commit_tmem_p.def_effectful_abstract_eval
def _commit_tmem_abstract_eval():
  return (), {gpu_core._memory_effect}


@lowering.register_lowering_rule(commit_tmem_p, mgpu.LoweringSemantics.Lane)
def _commit_tmem_lowering(_):
  tcgen05.commit_tmem()
  return ()


def commit_tmem():
  """Commits all writes to TMEM issued by the current thread.

  Once this function returns, the effects of calling ``async_store_tmem`` from
  the current thread are visible to TMEM loads, MMA and barrier operations of
  ``Barrier``s with ``orders_tensor_core=True``.
  """
  commit_tmem_p.bind()


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
  """Commits all writes to SMEM, making them visible to TMA and MMA operations."""
  commit_smem_p.bind()


def broadcasted_iota(
    dtype: jax.typing.DTypeLike,
    shape: Sequence[int],
    dimension: int,
    *,
    layout: Layout | None = None,
) -> jax.Array:
  result = jax.lax.broadcasted_iota(dtype, shape, dimension)
  if layout is not None:
    result = gpu_core.layout_cast(result, layout)
  return result


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
          ctx, ref, transforms, handle_reshapes=False, handle_transposes=False
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
      ref_treedefs=tuple(ref_treedefs),
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
    *refs: state.AbstractRef | state_types.TransformedRef,
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
      ref_treedefs=tuple(ref_treedefs),
      program_ids_treedef=program_ids_treedef,
  )


@dataclasses.dataclass(frozen=True)
class ShapeDtypeStruct:
  shape: tuple[int, ...]
  dtype: jnp.dtype
  layout: ParameterizedLayout | Layout


inline_mgpu_p = jax_core.Primitive("inline_mgpu_p")
inline_mgpu_p.multiple_results = True


@dataclasses.dataclass(frozen=True)
class RefType:
  transforms: tuple[gpu_core.MemoryRefTransform, ...] = ()


def _undo_transforms(
    raw_ref: state.AbstractRef,
    memory_transforms: Sequence[gpu_core.MemoryRefTransform],
):
  """Extract the `Transform`s that reverse the `MemoryRefTransform`s"""
  tmp_ref = state_types.TransformedRef(raw_ref, transforms=())
  tmp_ref = functools.reduce(lambda r, t: t.undo(r), reversed(memory_transforms), tmp_ref)
  return tmp_ref.transforms


def inline_mgpu(*, arg_types=(), return_type=None):
  r"""Returns a decorator that inlines Mosaic GPU code.

  This allows using lower-level Mosaic GPU abstractions and operations, which
  are otherwise not directly exposed in Pallas.

  Example::

      layout = plgpu.Layout.WG_STRIDED(x_ref.shape, vec_size=4)

      @plgpu.inline_mgpu(
          arg_types=(plgpu.RefType(),),
          return_type=plgpu.ShapeDtypeStruct(
              (128, 128), dtype, layout=layout
          ),
      )
      def add_one(ctx, smem_ref):
        x = mgpu.FragmentedArray.load_tiled(smem_ref)
        y = mgpu.FragmentedArray.splat(
            mgpu.c(1, x.mlir_dtype), shape=x.shape, layout=x.layout
        )
        return x + y

  Args:
    arg_types: A sequence of pytrees where the leaves are
      {class}`~jax.experimental.pallas.mosaic_gpu.RefType`\s or
      {class}`~jax.experimental.pallas.mosaic_gpu.Layout`\s for reference or
      array arguments respectively.
    return_type: A pytree where the leaves are
      {class}`~jax.experimental.pallas.mosaic_gpu.ShapeDtypeStruct`\s
      representing the arrays returned by the decorated function.
  """
  flat_arg_types, treedef_ty = jax.tree.flatten(tuple(arg_types))
  flat_ret_ty, pytree_ret_ty = jax.tree.flatten(return_type)
  if return_type and not all(isinstance(r, ShapeDtypeStruct) for r in flat_ret_ty):
    raise ValueError(
        "inline_mgpu_p only supports plgpu.ShapeDtypeStruct return types."
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
      ref_transforms = []
      raw_flat_args = []
      for a, t in zip(flat_args, flat_arg_types):
        if isinstance(a, state_types.TransformedRef) and isinstance(t, RefType):
          raw_flat_args.append(a.ref)
          ref_transforms.append(a.transforms)
        elif isinstance(aval := jax_core.get_aval(a), jax_core.ShapedArray) and isinstance(t, (ParameterizedLayout, Layout)):
          raw_flat_args.append(a)
          ref_transforms.append(None)
        elif isinstance(aval, state.AbstractRef) and isinstance(t, RefType):
          raw_flat_args.append(a)
          ref_transforms.append(())
        else:
          raise ValueError(f"Mismatched type: {a, t}")

      flat_ref_transforms, pytree_ref_transforms = jax.tree.flatten(ref_transforms)
      flat_ret = inline_mgpu_p.bind(
          *raw_flat_args,
          *flat_ref_transforms,
          flat_arg_types=tuple(flat_arg_types),
          flat_ret_ty=tuple(flat_ret_ty),
          pytree_ret_ty=pytree_ret_ty,
          pytree_args=treedef,
          pytree_ref_transforms=pytree_ref_transforms,
          mgpu_fn=f,
      )
      return jax.tree.unflatten(pytree_ret_ty, flat_ret)
    return wrapper

  return inner


@inline_mgpu_p.def_effectful_abstract_eval
def _inline_mgpu_abstract_eval(
    *flat_args_and_transforms,
    flat_arg_types,
    flat_ret_ty,
    pytree_args,
    pytree_ref_transforms,
    pytree_ret_ty,
    mgpu_fn,
):
  del flat_arg_types, pytree_ret_ty, pytree_ref_transforms, mgpu_fn  # Unused.
  aval_return = tuple(
      jax_core.ShapedArray(x.shape, x.dtype) for x in flat_ret_ty
  )
  # TODO(cperivol): Let the user set the effects.
  flat_args = flat_args_and_transforms[:pytree_args.num_leaves]
  return aval_return, {
      gpu_core._wgmma_pipeline_effect,
      gpu_core._memory_effect,
      *itertools.chain.from_iterable(
          (state.ReadEffect(i), state.WriteEffect(i))
          for i, r in enumerate(flat_args)
          if isinstance(r, state.AbstractRef)
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
    case (ShapeDtypeStruct(), mgpu.FragmentedArray()):
      mlir_dtype = mgpu_utils.dtype_to_ir_type(ty.dtype)
      if v.mlir_dtype != mlir_dtype:
        raise ValueError(
            f"Array dtype mismatch: expected {v.mlir_dtype} got {mlir_dtype}."
        )
      if ty.shape != v.shape:
        raise ValueError(
            f"Array shape mismatch: expected {ty.shape} got {v.shape}."
        )
      if v.layout != ty.layout.to_mgpu():
        raise ValueError(
            f"Array layout mismatch: expected {v.layout} got {ty.layout.to_mgpu()}."
        )
    case (Layout() , mgpu.FragmentedArray()) | (ParameterizedLayout(), mgpu.FragmentedArray()):
      if ty.to_mgpu() != v.layout:
        raise ValueError(f"Unexpected layout for {v} (expected: {ty})")
    case _:
      raise ValueError(f"Unexpected type {ty} for value {v}")


@lowering.register_lowering_rule(inline_mgpu_p, mgpu.LoweringSemantics.Lane)
def _inline_mgpu_lowering_rule(
    ctx: lowering.LoweringRuleContext,
    *flat_args_and_transforms,
    mgpu_fn: Callable[..., Any],
    flat_arg_types,
    flat_ret_ty,
    pytree_args,
    pytree_ref_transforms,
    pytree_ret_ty,
):
  flat_args = flat_args_and_transforms[:pytree_args.num_leaves]
  flat_arg_avals = ctx.avals_in[:pytree_args.num_leaves]
  ref_transforms = pytree_ref_transforms.unflatten(flat_args_and_transforms[pytree_args.num_leaves:])
  for a, t in zip(flat_args, flat_arg_types):
    _type_check_mgpu(a, t)

  flat_transformed = []
  for a, aval, t, transforms in zip(
      flat_args, flat_arg_avals, flat_arg_types, ref_transforms, strict=True
  ):
    if not isinstance(t, RefType):
      flat_transformed.append(a)
      assert transforms is None
      continue
    assert isinstance(aval, state.AbstractRef)
    a, user_transforms = lowering._handle_transforms(
        ctx, a, transforms, handle_transposes=False
    )
    # Transforms that do not originate from a MemoryRefTransform are
    # applied implicitly (eg by emit-pipeline) and therefore we do not
    # expect the user to pass them to the type. The transforms not
    # passed by the user here will be discharged.
    ty_transforms = _undo_transforms(aval, t.transforms)
    if ty_transforms != tuple(user_transforms):
      raise ValueError(f"Transform mismatch: got {user_transforms}, expected {ty_transforms}")
    flat_transformed.append(a)

  args = jax.tree.unflatten(pytree_args, flat_transformed)
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

load_p = jax_core.Primitive("load")


@load_p.def_effectful_abstract_eval
def _load_abstract_eval(src, *avals_flat, tree, optimized):
  del optimized  # Unused.
  transforms = tree.unflatten(avals_flat)
  dtype = lowering._transform_dtype(src.dtype, transforms)
  return (
      jax_core.ShapedArray(transforms[-1].get_indexer_shape(), dtype),
      {state.ReadEffect(0)},
  )


lowering.register_lowering_rule(load_p, mgpu.LoweringSemantics.Lane)(
    lowering._get_lowering_rule
)


def load(
    src: _Ref,
    idx,
    *,
    layout: Layout | None = None,
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
  result = load_p.bind(
      src,
      *flat_src_transforms,
      tree=src_transforms_treedef,
      optimized=optimized,
  )
  if layout is not None:
    result = gpu_core.layout_cast(result, layout)
  return result


async_load_tmem_p = jax_core.Primitive("async_load")

def async_load_tmem(src: _Ref, idx = (), *, layout: Layout | None = None) -> jax.Array:
  """Performs an asynchronous load from the TMEM array.

  The load operation is only partly asynchronous. The returned array can be used
  immediately, without any additional synchronization. However, it cannot be
  assumed that the read from TMEM has completed when the function returns. If
  you ever attempt to overwrite the read region, you should ensure that
  ``wait_load_tmem`` has been called before that happens. Failure to do so
  can result in nondeterministic data races.

  For example, the following sequence of operations at the end of the kernel is
  valid, even though the TMEM load is never awaited::

    smem_ref[...] = plgpu.async_load_tmem(tmem_ref)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(smem_ref, gmem_ref)
    plgpu.wait_smem_to_gmem(0)

  However, if the kernel was persistent and might reuse the TMEM again, the
  sequence should be extended with a call to ``wait_load_tmem``.

  Args:
    src: The TMEM reference to load from.
    idx: An optional slice that specifies a subregion that will be loaded. All
      data will be loaded if not specified.
    layout: The optional layout hint to use for the resulting array.
  """
  src, src_transforms = state_primitives.get_ref_and_transforms(
      src, idx, "async_load_tmem", force_trailing_indexer=True,
  )
  flat_src_transforms, src_transforms_treedef = tree_util.tree_flatten(
      src_transforms
  )
  result = async_load_tmem_p.bind(
      src, *flat_src_transforms, tree=src_transforms_treedef
  )
  if layout is not None:
    result = gpu_core.layout_cast(result, layout)
  return result

@async_load_tmem_p.def_effectful_abstract_eval
def _async_load_tmem_abstract_eval(src, *avals_flat, tree):
  if src.memory_space != gpu_core.MemorySpace.TMEM:
    raise ValueError("Async load only supports TMEM refs")
  return state_primitives._get_abstract_eval(src, *avals_flat, tree=tree)

@lowering.register_lowering_rule(async_load_tmem_p, mgpu.LoweringSemantics.Lane)
def _async_load_tmem_lowering_rule(
    ctx: lowering.LoweringRuleContext, x_ref, *leaves, tree
):
  assert isinstance(x_ref, tcgen05.TMEMRef)
  transforms = jax.tree.unflatten(tree, leaves)
  x_tmem, transforms = lowering._handle_transforms(
      ctx, x_ref, transforms, handle_transposes=False, handle_reshapes=False,
  )
  if transforms:
    raise NotImplementedError(
        f"Unimplemented transforms for TMEM refs. {transforms=}"
    )
  layout_hint = None
  if isinstance(ctx.out_layout_hint, mgpu.TiledLayout):
    layout_hint = ctx.out_layout_hint
  return x_tmem.load(layout=layout_hint)


wait_load_tmem_p = jax_core.Primitive("wait_load_tmem")
wait_load_tmem_p.multiple_results = True

def wait_load_tmem():
  """Awaits all previously asynchronous TMEM loads issued by the calling thread.

  Once this function returns, the TMEM loads issued by the calling thread are
  guaranteed to have completed. The read TMEM regions can be safely overwritten
  by the calling thread, or any threads signalled through ``Barrier``s with
  ``orders_tensor_core=True``.
  """
  wait_load_tmem_p.bind()


@wait_load_tmem_p.def_effectful_abstract_eval
def _wait_load_tmem_abstract_eval():
  return (), {gpu_core._memory_effect}


@lowering.register_lowering_rule(wait_load_tmem_p, mgpu.LoweringSemantics.Lane)
def _wait_load_tmem_lowering(_):
  tcgen05.wait_load_tmem()
  return ()


async_store_tmem_p = jax_core.Primitive("async_store_tmem")
async_store_tmem_p.multiple_results = True

def async_store_tmem(ref: _Ref, value):
  """Stores the value to TMEM.

  The store is asynchronous and is not guaranteed to be visible (e.g. by reads
  or MMA operations) until ``commit_tmem`` has been called.

  Args:
    ref: The TMEM reference to store to.
    value: The value to store.
  """
  ref, ref_transforms = state_primitives.get_ref_and_transforms(
      ref, None, "async_store_tmem", force_trailing_indexer=True,
  )
  flat_ref_transforms, ref_transforms_treedef = tree_util.tree_flatten(
      ref_transforms
  )
  async_store_tmem_p.bind(
      ref, value, *flat_ref_transforms, tree=ref_transforms_treedef
  )

@async_store_tmem_p.def_effectful_abstract_eval
def _async_store_tmem_abstract_eval(ref, val, *avals_flat, tree):
  if ref.memory_space != gpu_core.MemorySpace.TMEM:
    raise ValueError("Async store only supports TMEM refs")
  _, effects = state_primitives._swap_abstract_eval(
      ref, val, *avals_flat, tree=tree
  )
  return (), effects

@lowering.register_lowering_rule(async_store_tmem_p, mgpu.LoweringSemantics.Lane)
def _async_store_tmem_lowering_rule(
    ctx: lowering.LoweringRuleContext, x_ref, value, *leaves, tree
):
  assert isinstance(x_ref, tcgen05.TMEMRef)
  transforms = jax.tree.unflatten(tree, leaves)
  x_tmem, transforms = lowering._handle_transforms(
      ctx, x_ref, transforms, handle_transposes=False, handle_reshapes=False,
  )
  if transforms:
    raise NotImplementedError(
        f"Unimplemented transforms for TMEM refs. {transforms=}"
    )
  x_tmem.store(value)
  return ()
