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

import enum
from typing import Any, Literal

import jax
from jax._src import core as jax_core
from jax._src import effects
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.lib.mlir.dialects import nvvm as nvvm_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.state import discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
import jax.experimental.mosaic.gpu as mgpu


copy_smem_to_gmem_p = jax_core.Primitive("copy_smem_to_gmem")
copy_smem_to_gmem_p.multiple_results = True


@copy_smem_to_gmem_p.def_effectful_abstract_eval
def _copy_smem_to_gmem_abstract_eval(*avals, **params):
  del avals, params  # Unused.
  return (), {state.ReadEffect(0), state.WriteEffect(1)}


@lowering.register_lowering_rule(copy_smem_to_gmem_p)
def _copy_smem_to_gmem_lowering(
    ctx: lowering.LoweringRuleContext,
    src,
    dst,
    *flat_transforms,
    src_transforms_treedef,
    dst_transforms_treedef,
):
  flat_src_transforms, flat_dst_transforms = util.split_list(
      flat_transforms,
      [src_transforms_treedef.num_leaves],
  )
  src_transforms = src_transforms_treedef.unflatten(flat_src_transforms)
  dst_transforms = dst_transforms_treedef.unflatten(flat_dst_transforms)
  src, src_transforms = lowering._handle_indexing(src, src_transforms)
  copy_params = _extract_gmem_copy_params(dst_transforms) | _extract_smem_copy_params(src_transforms)
  mgpu.commit_shared()
  ctx.launch_ctx.async_copy(src_ref=src, dst_ref=dst, **copy_params)
  return ()


def _extract_gmem_copy_params(transforms):
  if not transforms:
    return {}
  if len(transforms) > 1:
    raise NotImplementedError("Only one level of indexing on GMEM refs supported")
  if not isinstance(indexer := transforms[0], indexing.NDIndexer):
    raise NotImplementedError("Only indexing on GMEM refs supported")
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
    src: pallas_core.AbstractMemoryRef, dst: pallas_core.AbstractMemoryRef
) -> None:
  """Asynchronously copies a SMEM reference to a GMEM reference.

  See also:
    :func:`jax.experimental.mosaic.gpu.wait_smem_to_gmem`
  """
  if src.memory_space is not gpu_core.SMEM:
    raise TypeError(f"src must be a SMEM reference, got {src.memory_space}")
  if dst.memory_space is not gpu_core.GMEM:
    raise ValueError(f"dst must be a GMEM reference, got {dst.memory_space}")
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
      src_transforms_treedef=src_transforms_treedef,
      dst_transforms_treedef=dst_transforms_treedef,
  )
  return None


copy_gmem_to_smem_p = jax_core.Primitive("copy_gmem_to_smem")
copy_gmem_to_smem_p.multiple_results = True


@copy_gmem_to_smem_p.def_effectful_abstract_eval
def _copy_gmem_to_smem_abstract_eval(*avals, **params):
  del avals, params  # Unused.
  return (), {state.ReadEffect(0), state.WriteEffect(1)}


@lowering.register_lowering_rule(copy_gmem_to_smem_p)
def _copy_gmem_to_smem_lowering(
    ctx: lowering.LoweringRuleContext,
    src,
    dst,
    barrier,
    *flat_transforms,
    src_transforms_treedef,
    dst_transforms_treedef,
    barrier_transforms_treedef,
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
  dst, dst_transforms = lowering._handle_indexing(dst, dst_transforms)
  copy_params = _extract_smem_copy_params(dst_transforms) | _extract_gmem_copy_params(src_transforms)
  barrier_indexer = _extract_barrier_indexer(
      barrier_transforms_treedef.unflatten(flat_barrier_transforms)
  )
  if barrier_indexer is not None:
    barrier = barrier.__getitem__(
        *map(lowering._as_index, barrier_indexer.indices)
    )
  ctx.launch_ctx.async_copy(
      src_ref=src, dst_ref=dst, barrier=barrier, **copy_params
  )
  return ()


def copy_gmem_to_smem(
    src: pallas_core.AbstractMemoryRef,
    dst: pallas_core.AbstractMemoryRef,
    *,
    barrier: pallas_core.AbstractMemoryRef,
) -> None:
  """Asynchronously copies a GMEM reference to a SMEM reference.

  See also:
    :func:`jax.experimental.mosaic.gpu.barrier_arrive`
    :func:`jax.experimental.mosaic.gpu.barrier_wait`
  """
  if src.memory_space is not gpu_core.GMEM:
    raise TypeError(f"src must be a GMEM reference, got {src.memory_space}")
  if dst.memory_space is not gpu_core.SMEM:
    raise ValueError(f"dst must be a SMEM reference, got {dst.memory_space}")
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


class MemoryEffect(jax_core.Effect):
  ...


effects.control_flow_allowed_effects.add_type(MemoryEffect)

_memory_effect = MemoryEffect()


barrier_arrive_p = jax_core.Primitive("barrier_arrive")
barrier_arrive_p.multiple_results = True


@barrier_arrive_p.def_effectful_abstract_eval
def _barrier_arrive_abstract_eval(*avals, **params):
  del avals, params  # Unused.
  return (), {_memory_effect}


@lowering.register_lowering_rule(barrier_arrive_p)
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
def _barrier_wait_abstract_eval(*avals, **params):
  del avals, params  # Unused.
  return (), {_memory_effect}


@lowering.register_lowering_rule(barrier_wait_p)
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
def _wait_smem_to_gmem_abstract_eval(n):
  del n  # Unused.
  return (), {_memory_effect}


@lowering.register_lowering_rule(wait_smem_to_gmem_p)
def _wait_smem_to_gmem_lowering(ctx: lowering.LoweringRuleContext, n):
  ctx.launch_ctx.await_async_copy(allow_groups=n)
  return ()


def wait_smem_to_gmem(n: int) -> None:
  """Waits until there are no more than ``n`` SMEM->GMEM copies in flight."""
  wait_smem_to_gmem_p.bind(n)


class _WGMMAPipelineEffect(effects.Effect):
  pass


_wgmma_pipeline_effect = _WGMMAPipelineEffect()
effects.control_flow_allowed_effects.add_type(_WGMMAPipelineEffect)

# WGMMA on an accumulator reference
wgmma_ref_p = jax_core.Primitive("wgmma_ref")
wgmma_ref_p.multiple_results = True


def wgmma(
    acc: gpu_core.WGMMAAbstractAccumulatorRef,
    a,
    b: pallas_core.TransformedRef,
) -> None:
  """Performs and asynchronous warp group matmul-accumulate on the given references.

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
  b_transforms_leaves, b_transforms_tree = jax.tree.flatten(b.transforms)

  wgmma_ref_p.bind(
      acc,
      a,
      b.ref,
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
      _wgmma_pipeline_effect,
      state.WriteEffect(0),
      state.ReadEffect(0),
      state.ReadEffect(2),
      *([state.ReadEffect(1)] if isinstance(a_aval, state.AbstractRef) else [])
  }


@discharge.register_discharge_rule(wgmma_ref_p)
def _wgmma_ref_discharge(in_avals, out_avals, *args, **kwargs):
  del in_avals, out_avals
  return (wgmma_p.bind(*args, **kwargs), *([None] * (len(args) - 1))), []


# Functional WGMMA, returns a shaped array. Internal.
wgmma_p = jax_core.Primitive("wgmma")


@lowering.register_lowering_rule(wgmma_p)
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
  lhs_swizzle = None
  if a_transforms_tree is not None:
    a_transforms_leaves, b_transforms_leaves = util.split_list(
        transforms_leaves, [a_transforms_tree.num_leaves]
    )
    a_transforms = a_transforms_tree.unflatten(a_transforms_leaves)
    a, a_transforms = lowering._handle_indexing(a, a_transforms)
    match a_transforms:
      case (gpu_core.UnswizzleRef(lhs_swizzle), gpu_core.UntileRef(tiling)):
        swizzle_elems = lhs_swizzle // a_aval.dtype.itemsize
        if tiling != (64, swizzle_elems):
          raise NotImplementedError("WGMMA lhs tiling does not fit swizzle")
      case _:
        raise ValueError(f"WGMMA lhs has unsupported transforms: {a_transforms}.")
  else:
    b_transforms_leaves = transforms_leaves  # type: ignore
    if not isinstance(a, mgpu.FragmentedArray):
      raise ValueError(
          "When WGMMA lhs is passed in as a ref, it must be transformed by"
          " swizzling and tiling appropriately."
      )

  b_transforms = b_transforms_tree.unflatten(b_transforms_leaves)
  b, b_transforms = lowering._handle_indexing(b, b_transforms)

  match b_transforms:
    case (gpu_core.UnswizzleRef(rhs_swizzle), gpu_core.UntileRef(rhs_tiling)):
      rhs_transpose = False
    case (
        gpu_core.UnswizzleRef(rhs_swizzle),
        gpu_core.TransposeRef((1, 0, 2, 3)),  # Only transpose between tiles
        gpu_core.UntileRef(rhs_tiling),
        gpu_core.TransposeRef((1, 0)),  # Transpose the two logical dims
    ):
      rhs_transpose = True
    case _:
      raise ValueError(f"WGMMA rhs has unsupported transforms: {b_transforms}.")

  if lhs_swizzle is not None:
    swizzle_elems = rhs_swizzle // a_aval.dtype.itemsize
    if rhs_swizzle != lhs_swizzle:
      raise NotImplementedError("WGMMA rhs swizzle must match lhs swizzle")
    if rhs_tiling != (swizzle_elems, swizzle_elems):
      raise NotImplementedError("WGMMA rhs tiling does not fit swizzle")

  new_acc = mgpu.wgmma(
      acc,
      a,
      b,
      swizzle=rhs_swizzle,
      b_order=mgpu.WGMMALayout.COL_MAJOR
      if rhs_transpose
      else mgpu.WGMMALayout.ROW_MAJOR,
  )
  nvvm_dialect.wgmma_commit_group_sync_aligned()
  return new_acc


@wgmma_p.def_effectful_abstract_eval
def _wgmma_effectful_abstract_eval(acc, lhs_ref, *args, **kwargs):
  del args, kwargs
  return acc, {
      _wgmma_pipeline_effect,
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
  return [], {_wgmma_pipeline_effect}


@lowering.register_lowering_rule(wgmma_wait_p)
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
  ret = acc.inner_aval if isinstance(acc, gpu_core.WGMMAAbstractAccumulatorRef) else acc
  assert isinstance(ret, jax_core.ShapedArray), acc
  return ret, {_wgmma_pipeline_effect}


@discharge.register_discharge_rule(wgmma_accumulator_deref_p)
def _wgmma_accumulator_deref_discharge(in_avals, out_avals, acc):
  del in_avals, out_avals
  return (None,), wgmma_accumulator_deref_p.bind(acc)


@lowering.register_lowering_rule(wgmma_accumulator_deref_p)
def _wgmma_accumulator_deref_lowering(ctx: lowering.LoweringRuleContext, acc):
  del ctx
  nvvm_dialect.wgmma_wait_group_sync_aligned(0)
  return acc.value


class Layout(enum.Enum):
  #: [m, n] matrix, where m % 64 == 0 == n % 8.
  WGMMA = mgpu.WGMMA_LAYOUT
  #: [m] matrix, where m % 64 == 0.
  WGMMA_ROW = mgpu.WGMMA_ROW_LAYOUT


layout_cast_p = jax_core.Primitive("layout_cast")


@layout_cast_p.def_abstract_eval
def _layout_cast_abstract_eval(x, new_layout):
  del new_layout  # Unused.
  return x


@lowering.register_lowering_rule(layout_cast_p)
def _layout_cast_lowering(ctx: lowering.LoweringRuleContext, x, *, new_layout):
  del ctx  # Unused.
  return x.to_layout(new_layout.value)


def layout_cast(x: Any, new_layout: Layout):
  """Casts the layout of the given array."""
  return layout_cast_p.bind(x, new_layout=new_layout)


set_max_registers_p = jax_core.Primitive("set_max_registers_p")
set_max_registers_p.multiple_results = True


@set_max_registers_p.def_effectful_abstract_eval
def _set_max_registers_abstract_eval(n, *, action):
  del n, action  # Unused.
  return (), {_memory_effect}


@lowering.register_lowering_rule(set_max_registers_p)
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
  return (), {_memory_effect}


@lowering.register_lowering_rule(commit_smem_p)
def _commit_smem_lowering(ctx: lowering.LoweringRuleContext):
  mgpu.commit_shared()
  return ()


def commit_smem():
  """Commits all writes to SMEM, making them visible to loads, TMA and WGMMA."""
  commit_smem_p.bind()
