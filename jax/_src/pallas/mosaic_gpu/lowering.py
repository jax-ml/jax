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

"""Module for lowering JAX primitives to Mosaic GPU."""

from __future__ import annotations

import collections
from collections.abc import Callable, Hashable, Iterator, MutableMapping, MutableSequence, Sequence
import contextlib
import dataclasses
import functools
import itertools
import math
import operator
from typing import Any, Protocol, Self, TypeVar, cast

import jax
from jax import api_util
from jax import lax
from jax._src import checkify
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import pjit
from jax._src import source_info_util
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.lib.mlir.dialects import cf as cf_dialect
from jax._src.lib.mlir.dialects import gpu as gpu_dialect
from jax._src.lib.mlir.dialects import llvm as llvm_dialect
from jax._src.lib.mlir.dialects import math as math_dialect
from jax._src.lib.mlir.dialects import memref as memref_dialect
from jax._src.lib.mlir.dialects import nvvm as nvvm_dialect
from jax._src.lib.mlir.dialects import scf as scf_dialect
from jax._src.lib.mlir.dialects import vector as vector_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas import helpers as pallas_helpers
from jax._src.pallas import pallas_call
from jax._src.pallas import primitives
from jax._src.pallas import utils as pallas_utils
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.state import discharge
from jax._src.state import indexing
from jax._src.state import primitives as sp
from jax._src.state import types as state_types
from jax._src.state.types import RefReshaper
from jax._src.util import foreach
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.experimental.mosaic.gpu import profiler as mgpu_profiler
from jax.experimental.mosaic.gpu import tcgen05
from jax.experimental.mosaic.gpu import utils as mgpu_utils
import jax.numpy as jnp
import numpy as np


# TODO(slebedev): Enable type checking.
# mypy: ignore-errors

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

partial = functools.partial
SMEM = gpu_core.SMEM
WARPGROUP_SIZE = 128
RefOrTmemType = TypeVar("RefOrTmemType", ir.Value, tcgen05.TMEMRef)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResourceEstimatorContext:
  axis_names: _AxisNames
  lowering_semantics: mgpu.LoweringSemantics

  @property
  def arrival_multiplier(self) -> int:
    return (
        WARPGROUP_SIZE
        if self.lowering_semantics == mgpu.LoweringSemantics.Lane
        else 1
    )


AnyBarrier = mgpu.Barrier | mgpu.ClusterBarrier


@dataclasses.dataclass(kw_only=True, frozen=True)
class Resources:
  smem_scratch_bytes: int = 0
  tmem_scratch_cols: int = 0
  tmem_collective_scratch_cols: int = 0
  barrier_counts: collections.Counter[AnyBarrier] = dataclasses.field(
      default_factory=collections.Counter
  )
  gmem_semaphores: int = 0

  def __post_init__(self):
    object.__setattr__(
        self,
        "smem_scratch_bytes",
        gpu_core.align_to(self.smem_scratch_bytes, gpu_core.SMEM_ALIGNMENT),
    )

    # TMEM must be allocated in 128x8 chunks.
    object.__setattr__(
        self,
        "tmem_scratch_cols",
        gpu_core.align_to(self.tmem_scratch_cols, 8),
    )
    object.__setattr__(
        self,
        "tmem_collective_scratch_cols",
        gpu_core.align_to(self.tmem_collective_scratch_cols, 8),
    )

  @property
  def barriers(self) -> Sequence[AnyBarrier]:
    return list(self.barrier_counts.elements())

  def __add__(self, other: Resources) -> Resources:
    # TODO(slebedev): Optimize this.
    #
    # At the moment, if we have run_scoped(b1) followed by run_scoped(b2)
    # we will allocate two barriers, even though one would be enough.
    return Resources(
        smem_scratch_bytes=self.smem_scratch_bytes + other.smem_scratch_bytes,
        tmem_scratch_cols=self.tmem_scratch_cols + other.tmem_scratch_cols,
        tmem_collective_scratch_cols=self.tmem_collective_scratch_cols
        + other.tmem_collective_scratch_cols,
        barrier_counts=self.barrier_counts + other.barrier_counts,
        gmem_semaphores=self.gmem_semaphores + other.gmem_semaphores,
    )

  def __or__(self, other: Resources) -> Resources:
    return Resources(
        smem_scratch_bytes=max(
            self.smem_scratch_bytes, other.smem_scratch_bytes
        ),
        tmem_scratch_cols=max(self.tmem_scratch_cols, other.tmem_scratch_cols),
        tmem_collective_scratch_cols=max(
            self.tmem_collective_scratch_cols,
            other.tmem_collective_scratch_cols,
        ),
        barrier_counts=self.barrier_counts | other.barrier_counts,
        gmem_semaphores=max(self.gmem_semaphores, other.gmem_semaphores),
    )


class ResourceEstimator(Protocol):

  def __call__(
      self, ctx: ResourceEstimatorContext, *args: Any, **params: Any
  ) -> Resources:
    ...


_resource_estimators: dict[jax_core.Primitive, ResourceEstimator] = {}


def _register_resource_estimator(primitive: jax_core.Primitive):
  def deco(fn):
    _resource_estimators[primitive] = fn
    return fn

  return deco


def _estimate_resources(
    ctx: ResourceEstimatorContext, jaxpr: jax_core.Jaxpr
) -> Resources:
  """Estimates the resources required by the kernel."""
  rs = Resources(smem_scratch_bytes=0)
  for eqn in jaxpr.eqns:
    # TODO(slebedev): Add support for other primitives, notably control flow.
    if rule := _resource_estimators.get(eqn.primitive):
      rs |= rule(ctx, *(invar.aval for invar in eqn.invars), **eqn.params)
      continue
    # Assume that unsupported primitives are neutral wrt resource usage,
    # unless they have a jaxpr in their params.
    if any(
        isinstance(v, (jax_core.Jaxpr, jax_core.ClosedJaxpr))
        for v in eqn.params.values()
    ):
      raise NotImplementedError(
          f"Resource estimation does not support {eqn.primitive}"
      )

  return rs


@_register_resource_estimator(lax.cond_p)
def _cond_resource_estimator(
    ctx: ResourceEstimatorContext, *args, branches
) -> Resources:
  del args  # Unused.
  return functools.reduce(
      lambda a, b: a | b,
      (_estimate_resources(ctx, branch.jaxpr) for branch in branches),
  )


@_register_resource_estimator(lax.scan_p)
def _scan_resource_estimator(
    ctx: ResourceEstimatorContext, *args, jaxpr: jax_core.ClosedJaxpr, **params
) -> Resources:
  del args, params  # Unused.
  return _estimate_resources(ctx, jaxpr.jaxpr)


@_register_resource_estimator(lax.while_p)
def _while_resource_estimator(
    ctx: ResourceEstimatorContext,
    *args,
    cond_jaxpr: jax_core.ClosedJaxpr,
    body_jaxpr: jax_core.ClosedJaxpr,
    **params,
) -> Resources:
  del args, params  # Unused.
  return _estimate_resources(ctx, cond_jaxpr.jaxpr) | _estimate_resources(
      ctx, body_jaxpr.jaxpr
  )


@_register_resource_estimator(pjit.jit_p)
def _pjit_resource_estimator(
    ctx: ResourceEstimatorContext,
    *args,
    jaxpr: jax_core.ClosedJaxpr,
    **params,
) -> Resources:
  del args, params  # Unused.
  return _estimate_resources(ctx, jaxpr.jaxpr)


@_register_resource_estimator(pallas_core.core_map_p)
def _core_map_resource_estimator(
    ctx: ResourceEstimatorContext, *args, jaxpr: jax_core.Jaxpr, **params
) -> Resources:
  del args, params  # Unused.
  return _estimate_resources(ctx, jaxpr)


@_register_resource_estimator(discharge.run_state_p)
def _run_state_resource_estimator(
    ctx: ResourceEstimatorContext, *args, jaxpr: jax_core.Jaxpr, **params
) -> Resources:
  del args, params  # Unused.
  return _estimate_resources(ctx, jaxpr)


@_register_resource_estimator(primitives.run_scoped_p)
def _run_scoped_resource_estimator(
    ctx: ResourceEstimatorContext,
    *consts,
    jaxpr: jax_core.Jaxpr,
    collective_axes,
) -> Resources:
  del collective_axes  # Unused.

  # NOTE: This rule assumes that the allocation happens collectively, although
  # it can't be checked here due to limited context. We check this in the actual
  # lowering rule.
  del consts  # Unused.
  rs = Resources()
  for v in jaxpr.invars:
    aval = cast(ShapedAbstractValue, v.aval)
    if isinstance(aval.dtype, gpu_core.BarrierType):
      multiplier = 1 if aval.dtype.orders_tensor_core else ctx.arrival_multiplier
      rs += Resources(
          barrier_counts=collections.Counter([
              mgpu.Barrier(
                  aval.dtype.num_arrivals * multiplier, *aval.shape
              )
          ])
      )
      continue
    if isinstance(aval.dtype, gpu_core.ClusterBarrierType):
      collective_dims = jax.tree.map(
          lambda axis: _resolve_cluster_axis(ctx.axis_names, axis),
          aval.dtype.collective_axes,
      )
      rs += Resources(
          barrier_counts=collections.Counter(
              [mgpu.ClusterBarrier(collective_dims, *aval.shape)]
          )
      )
      continue
    assert isinstance(aval, state_types.AbstractRef)
    if aval.memory_space == gpu_core.TMEM:
      if len(aval.shape) != 2:
        raise ValueError(f"TMEM allocations must be 2D. Got {aval.shape}")
      # Estimate columns used.
      if isinstance(aval, gpu_core.AbstractRefUnion):
        assert aval.shape[0] == 128
        cols_used = aval.shape[1]
      else:
        cols_used = aval.layout.cols_in_shape(
            aval.shape, dtypes.bit_width(aval.dtype)
        )
      if aval.collective:
        rs += Resources(tmem_collective_scratch_cols=cols_used)
      else:
        rs += Resources(tmem_scratch_cols=cols_used)
    elif aval.memory_space == gpu_core.SMEM:
      rs += Resources(
          smem_scratch_bytes=aval.size * dtypes.bit_width(aval.dtype) // 8
      )
    elif aval.memory_space == gpu_core.REGS:
      # Don't need to allocate anything.
      pass
    elif aval.memory_space == gpu_core.GMEM and jnp.issubdtype(aval.dtype, pallas_core.semaphore):
      rs += Resources(gmem_semaphores=aval.size)
    else:
      raise NotImplementedError(
          f"Unsupported memory space: {aval.memory_space}")
  return rs + _estimate_resources(ctx, jaxpr)

REDUCE_SCRATCH_ELEMS = 128 * 2  # vector of 2 elements per lane in each WG

@_register_resource_estimator(lax.reduce_sum_p)
@_register_resource_estimator(lax.reduce_max_p)
def _reduce_resource_estimator(
    ctx: ResourceEstimatorContext, x_aval: jax_core.ShapedArray, *, axes
) -> Resources:
  del ctx, axes  # Unused.
  # We don't need SMEM for some reductions, but it depends on the layout, so we
  # conservatively request the maximum scratch space we might need.
  return Resources(smem_scratch_bytes=REDUCE_SCRATCH_ELEMS * x_aval.dtype.itemsize)


@dataclasses.dataclass(frozen=True)
class _AxisNames:
  grid: Sequence[Hashable]
  cluster: Sequence[Hashable] = ()
  wg: Hashable | None = None

  def __iter__(self) -> Iterator[Hashable]:
    return itertools.chain(
        self.grid, self.cluster, [self.wg] if self.wg is not None else []
    )

  def reverse(self) -> "_AxisNames":
    return _AxisNames(self.grid[::-1], self.cluster[::-1], self.wg)


AnyBarrierRef = (
    mgpu.BarrierRef | mgpu.DialectBarrierRef | mgpu.CollectiveBarrierRef
)


@dataclasses.dataclass
class ModuleContext:
  name: str
  axis_names: _AxisNames
  program_ids: Sequence[ir.Value] | None
  approx_math: bool
  single_wg_lane_predicate: ir.Value | None
  single_warp_lane_predicate: ir.Value | None
  smem_requested_bytes: int
  smem_used_bytes: int
  tmem_requested_cols: int
  tmem_used_cols: int
  tmem_base_ptr: ir.Value
  tmem_collective_requested_cols: int
  tmem_collective_used_cols: int
  tmem_collective_base_ptr: ir.Value
  gmem_used_semaphores: int
  gmem_semaphore_base_ptr: ir.Value | None
  runtime_barriers: MutableMapping[AnyBarrier, MutableSequence[AnyBarrierRef]]
  name_stack: source_info_util.NameStack
  traceback_caches: mlir.TracebackCaches
  squashed_dims: tuple[int, ...]
  lowering_semantics: mgpu.LoweringSemantics
  primitive_semantics: gpu_core.PrimitiveSemantics
  mesh_info: pallas_utils.MeshInfo | None
  # See the documentation of unsafe_no_auto_barriers in CompilerParams.
  auto_barriers: bool
  warp_axis_name: str | None = None

  @property
  def single_lane_predicate(self) -> ir.Value:
    """Returns a predicate that is True for a single lane within the current
    thread semantics.
    """
    assert self.lowering_semantics == mgpu.LoweringSemantics.Lane
    match self.primitive_semantics:
      case gpu_core.PrimitiveSemantics.Warpgroup:
        return self.single_wg_lane_predicate
      case gpu_core.PrimitiveSemantics.Warp:
        return self.single_warp_lane_predicate
      case _:
        raise ValueError(f"Unknown semantics: {self.primitive_semantics}")

  @contextlib.contextmanager
  def reserve_barrier(
      self, barrier: mgpu.Barrier
  ) -> Iterator[
      mgpu.BarrierRef | mgpu.DialectBarrierRef | mgpu.CollectiveBarrierRef
  ]:
    """Reserves a barrier.

    Raises:
      RuntimeError: If the barrier is already reserved.
    """
    available = self.runtime_barriers.get(barrier, [])
    if not available:
      raise RuntimeError(f"Barrier {barrier} is already reserved")
    barrier = available.pop()
    yield barrier
    available.append(barrier)

  @contextlib.contextmanager
  def reserve_semaphores(self, shape: tuple[int, ...]) -> Iterator[ir.Value]:
    allocated_sems = math.prod(shape)
    ref = mgpu.memref_slice(
        self.gmem_semaphore_base_ptr,
        mgpu.ds(self.gmem_used_semaphores, allocated_sems),
    )
    ref = mgpu.memref_reshape(ref, shape)
    self.gmem_used_semaphores += allocated_sems
    yield ref
    # TODO: In debug mode verify the values of all semaphores are again 0
    self.gmem_used_semaphores -= allocated_sems

  @contextlib.contextmanager
  def alloc_tmem(
      self,
      struct: jax.ShapeDtypeStruct,
      *,
      layout: tcgen05.TMEMLayout,
      collective: bool,
  ) -> Iterator[ir.Value]:
    if collective:
      off = arith_dialect.addi(
          self.tmem_collective_base_ptr,
          _i32_constant(self.tmem_collective_used_cols),
      )
    else:
      off = arith_dialect.addi(
          self.tmem_base_ptr, _i32_constant(self.tmem_used_cols)
      )
    tmem_ref = tcgen05.TMEMRef(
        address=off,
        shape=struct.shape,
        dtype=mgpu_utils.dtype_to_ir_type(struct.dtype),
        layout=layout)
    cols_used = layout.cols_in_shape(
        struct.shape, dtypes.bit_width(struct.dtype)
    )
    cols_used = gpu_core.align_to(cols_used, gpu_core.TMEM_COL_ALIGNMENT)
    if collective:
      self.tmem_collective_used_cols += cols_used
      yield tmem_ref
      self.tmem_collective_used_cols -= cols_used
    else:
      self.tmem_used_cols += cols_used
      yield tmem_ref
      self.tmem_used_cols -= cols_used

  # TODO(cperivol): Only return the shapes and figure out the sizes when freeing.
  @contextlib.contextmanager
  def scratch_view(
      self, structs: Sequence[jax.ShapeDtypeStruct]
  ) -> Iterator[Sequence[ir.Value]]:
    """Creates a view into the runtime scratch buffer for each struct.

    This is a low-level API. Use it only if you know what you are doing.

    The function allocates bytes at the top of a stack, which need to be
    deallocated in a FIFO fashion with :meth:`ModuleContext.stack_free_smem`.
    After deallocation, the view is invalid and cannot be used.

    Args:
      structus: The shapes and dtypes of the views to create.

    Returns:
      A tuple, where the first element is the number of bytes allocated,
      and the second element is a sequence of memref views into the
      runtime scratch buffer.
    """
    smem_base = None
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    if self.lowering_semantics == mgpu.LoweringSemantics.Lane:
      smem_base = gpu_dialect.dynamic_shared_memory(
          ir.MemRefType.get(
              (mgpu_utils.DYNAMIC,), i8, memory_space=mgpu_utils.smem()
          )
      )
    views = []
    off = initial_used_bytes = self.smem_used_bytes
    assert off % gpu_core.SMEM_ALIGNMENT == 0
    for s in structs:
      scratch_ty = ir.MemRefType.get(
          s.shape,
          mgpu_utils.dtype_to_ir_type(s.dtype),
          memory_space=mgpu_utils.smem(),
      )
      # The below code emission relies on the assumption that the first scratch
      # operand provided by Mosaic GPU always begins at the beginning of
      # dynamic SMEM. Mosaic GPU is expected to uphold that invariant.
      if self.lowering_semantics == mgpu.LoweringSemantics.Lane:
        view = memref_dialect.view(
            scratch_ty, smem_base, _as_index(off), []
        )
      else:
        view = mgpu.dialect.slice_smem(scratch_ty, mgpu_utils.c(off, i32))
      views.append(view)

      off += gpu_core.align_to(
          math.prod(s.shape) * dtypes.bit_width(jnp.dtype(s.dtype)) // 8,
          gpu_core.SMEM_ALIGNMENT,
      )
    assert off <= self.smem_requested_bytes, "Ran out of scoped SMEM"
    assert off % gpu_core.SMEM_ALIGNMENT == 0

    self.smem_used_bytes = off
    yield views
    self.smem_used_bytes = initial_used_bytes


# This is morally ``ShapedArray | state.AbstractRef``, but pytype does not
# allow calling methods on a union type, making ``update`` non-callable, so
# we use a protocol instead of a union.
class ShapedAbstractValue(Protocol):
  shape: tuple[jax_core.DimSize, ...]
  dtype: jnp.dtype
  weak_type: bool

  @property
  def ndim(self) -> int:
    ...

  @property
  def size(self) -> int:
    ...

  def update(self, **kwargs: Any) -> Self:
    raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class LoweringRuleContext:
  module_ctx: ModuleContext
  launch_ctx: mgpu.LaunchContext
  prim: jax_core.Primitive
  avals_in: Sequence[ShapedAbstractValue]
  avals_out: Sequence[ShapedAbstractValue]
  out_layout_hint: mgpu.FragmentedLayout | None

  def replace(self, **changes: Any) -> LoweringRuleContext:
    # The wrapper is necessary to convince pytype that this is a method.
    return dataclasses.replace(self, **changes)

  @property
  def estimator_ctx(self) -> ResourceEstimatorContext:
    return ResourceEstimatorContext(
        axis_names=self.module_ctx.axis_names,
        lowering_semantics=self.module_ctx.lowering_semantics,
    )


@dataclasses.dataclass(frozen=True)
class LoweringResult:
  module: ir.Module
  grid: tuple[int, ...]
  block: tuple[int, ...]
  new_out_shapes: tuple[jax.ShapeDtypeStruct, ...]  # Does not include gmem scratch!
  profiler_context: ProfilerContext | None
  gmem_scratch_shapes: tuple[jax.ShapeDtypeStruct, ...]


@dataclasses.dataclass(frozen=True)
class ProfilerContext:
  dump_path: str
  spec: mgpu_profiler.ProfilerSpec


class LoweringError(Exception):  # pylint: disable=g-bad-exception-name
  pass


def _eval_index_map(
    module_ctx: ModuleContext,
    launch_ctx: mgpu.LaunchContext,
    idx: Sequence[ir.Value],
    block_mapping: pallas_core.BlockMapping,
) -> Sequence[ir.Value]:
  block_indices = lower_jaxpr_to_mosaic_gpu(
      module_ctx, launch_ctx, block_mapping.index_map_jaxpr.jaxpr, idx
  )
  result = []
  for i, b in zip(block_indices, block_mapping.block_shape):
    match b:
      case pallas_core.Squeezed() | pallas_core.Element():
        result.append(i)
      case pallas_core.Blocked():
        result.append(arith_dialect.muli(_as_index(i), _as_index(b)))
      case _:
        raise ValueError(f"Unsupported block dim type: {b}")
  return tuple(result)


def _check_block_mappings(
    block_mappings: Sequence[pallas_core.BlockMapping],
    debug_info: jax_core.DebugInfo,
) -> None:
  def err_details(bm: pallas_core.BlockMapping) -> str:
    return (
        f"Block spec for {bm.origin} in pallas_call {debug_info.func_src_info}"
        f" has block shape {bm.block_shape}, array shape"
        f" {bm.array_shape_dtype.shape},"
        # TODO(necula): add index_map source location info
        f" and index_map {bm.index_map_jaxpr.jaxpr} in"
        f" memory space {bm.transformed_block_aval.memory_space}."
        " See details at"
        " https://docs.jax.dev/en/latest/pallas/grid_blockspec.html#pallas-blockspec."
    )

  for bm in block_mappings:
    if (
        bm.transformed_block_aval.memory_space == gpu_core.GMEM
        and not bm.has_trivial_window()
    ):
      raise NotImplementedError(
          "Mosaic GPU lowering currently requires blocks in GMEM memory space "
          "to have same block shape as the array shape "
          "and a trivial index_map (returning all 0s).\n\n"
          + err_details(bm)
      )

    if any(isinstance(b, pallas_core.Element) for b in bm.block_shape):
      raise NotImplementedError(
          "Only Blocked indexing mode is supported in Mosaic GPU lowering.\n\n"
          + err_details(bm)
      )

    if bm.pipeline_mode is not None:
      raise NotImplementedError(
          "Pipeline mode is not supported in Mosaic GPU lowering.\n\n"
          + err_details(bm)
      )


def _block_spec_from_block_mapping(
    bm: pallas_core.BlockMapping,
    which_parallel: Sequence[bool],
) -> pallas_core.BlockSpec:
  eval_index_map = functools.partial(
      jax.core.eval_jaxpr,
      bm.index_map_jaxpr.jaxpr,
      bm.index_map_jaxpr.consts,
  )

  def index_map(*indices):
    # Inject the parallel indices into the sequential ones coming from
    # `emit_pipeline`.
    new_indices = util.merge_lists(
        which_parallel,
        indices,
        [
            primitives.program_id(axis - 1)
            for axis, is_parallel in zip(
                itertools.accumulate(which_parallel), which_parallel
            )
            if is_parallel
        ],
    )
    return eval_index_map(*new_indices)

  return gpu_core.BlockSpec(
      bm.block_shape,
      index_map,
      memory_space=bm.transformed_block_aval.memory_space,
      transforms=cast(Sequence[gpu_core.MemoryRefTransform], bm.transforms),
  )


def lower_pipelined_jaxpr_to_module(
    grid_mapping: pallas_core.GridMapping,
    gpu_mesh: pallas_core.Mesh | None,
    jax_mesh: mesh_lib.Mesh | None,
    jaxpr: jax_core.Jaxpr,
    params: gpu_core.CompilerParams,
    cost_estimate: pallas_core.CostEstimate | None,
) -> LoweringResult:
  del cost_estimate  # Unused.

  assert len(jaxpr.outvars) == 0
  assert not grid_mapping.vmapped_dims
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "Dynamic grid bounds not supported in the Mosaic GPU lowering."
    )
  if grid_mapping.num_index_operands:
    raise NotImplementedError(
        "Scalar prefetch not supported in Mosaic GPU lowering."
    )

  block_mappings = grid_mapping.block_mappings
  _check_block_mappings(block_mappings, jaxpr.debug_info)
  in_block_mappings, out_block_mappings = util.split_list(
      block_mappings, [grid_mapping.num_inputs]
  )

  if gpu_mesh:
    assert isinstance(gpu_mesh, gpu_core.Mesh)
    block = (128 * (gpu_mesh.num_threads or 1), 1, 1)
    grid = gpu_mesh.grid
    thread_axis = (
        gpu_mesh.thread_name if gpu_mesh.thread_name is not None else ()
    )
  else:
    block = (128, 1, 1)
    grid = grid_mapping.grid
    thread_axis = ()

  if params.dimension_semantics is None:
    which_parallel = [True] * len(grid)
  else:
    assert len(params.dimension_semantics) == len(grid)
    which_parallel = [ds == "parallel" for ds in params.dimension_semantics]

  sequential_grid = tuple(
      d for axis, d in enumerate(grid) if not which_parallel[axis]
  )
  parallel_grid = tuple(
      d for axis, d in enumerate(grid) if which_parallel[axis]
  )

  from jax._src.pallas.mosaic_gpu import pipeline  # pytype: disable=import-error
  from jax._src.pallas.mosaic_gpu import primitives as gpu_primitives  # pytype: disable=import-error

  def ref_for_aval(aval: ShapedAbstractValue):
    if isinstance(aval, gpu_core.WGMMAAbstractAccumulatorRef):
      return gpu_core.WGMMAAccumulatorRef(aval.shape, aval.dtype)
    elif isinstance(aval, gpu_core.AbstractTMEMRef):
      return gpu_core.GPUMemoryRef(
          aval.shape, aval.dtype, gpu_core.TMEM,
          transforms=(), layout=aval.layout, collective=aval.collective,
      )
    elif isinstance(aval, state_types.AbstractRef):
      return pallas_core.MemoryRef(aval.shape, aval.dtype, aval.memory_space)
    else:
      return gpu_core.SMEM(aval.shape, aval.dtype)

  def pipeline_fn(*refs):
    primitives.run_scoped(
        functools.partial(scoped_pipeline_fn, *refs),
        scratch_refs=[
            ref_for_aval(cast(ShapedAbstractValue, v.aval))
            for v in jaxpr.invars[grid_mapping.slice_scratch_ops]
        ],
        collective_axes=thread_axis,  # scratch_refs are shared across threads
    )
    return ()  # ``wrap_init`` does not support functions returning None.

  def scoped_pipeline_fn(*refs, scratch_refs):
    def body_fn(indices, *refs):
      program_ids_template = util.merge_lists(
          which_parallel, indices, [None] * sum(which_parallel)
      )
      assert len(refs) + len(scratch_refs) == len(jaxpr.invars)
      return gpu_primitives.jaxpr_call(
          jaxpr, *refs, *scratch_refs, program_ids=program_ids_template
      )

    return pipeline.emit_pipeline(
        body_fn,
        grid=sequential_grid,
        in_specs=[
            _block_spec_from_block_mapping(bm, which_parallel)
            for bm in in_block_mappings
        ],
        out_specs=[
            _block_spec_from_block_mapping(bm, which_parallel)
            for bm in out_block_mappings
        ],
        max_concurrent_steps=params.max_concurrent_steps,
        delay_release=params.delay_release,
    )(*refs)

  with grid_mapping.trace_env():
    new_jaxpr, _, new_consts = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(pipeline_fn, debug_info=jaxpr.debug_info),
        [
            gpu_core.GMEM(
                bm.array_shape_dtype.shape, bm.array_shape_dtype.dtype
            ).get_ref_aval()
            for bm in block_mappings
        ],
    )
    assert not new_consts

  axis_names = (
      _AxisNames(gpu_mesh.grid_names, gpu_mesh.cluster_names, gpu_mesh.thread_name)
      if gpu_mesh is not None
      else _AxisNames(grid_mapping.grid_names or ())
  )
  with grid_mapping.trace_env():
    return lower_jaxpr_to_module(
        jax_mesh,
        axis_names,
        parallel_grid,
        block,
        gpu_mesh.cluster if gpu_mesh is not None else (),
        [bm.array_shape_dtype for bm in in_block_mappings],
        [bm.array_shape_dtype for bm in out_block_mappings],
        new_jaxpr,
        params,
        new_consts,
    )


def lower_jaxpr_to_module(
    jax_mesh: mesh_lib.Mesh | None,
    axis_names: _AxisNames,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    cluster: tuple[int, ...],
    in_shapes: Sequence[jax.ShapeDtypeStruct],
    out_shapes: Sequence[jax.ShapeDtypeStruct],
    jaxpr: jax_core.Jaxpr,
    params: gpu_core.CompilerParams,
    consts=(),
) -> LoweringResult:
  debug_info = jaxpr.debug_info
  approx_math = params.approx_math
  lowering_semantics = params.lowering_semantics

  if len(cluster) < 3:
    cluster = (1,) * (3 - len(cluster)) + cluster
  else:
    assert len(cluster) == 3

  if len(grid) <= 3:
    squashed_dims = ()
    parallel_grid = (1,) * (3 - len(grid)) + grid
  else:
    # If we have >3 parallel dimensions, we flatten all but the minormost 2 dims.
    # Ex: (2, 3, 4, 5) -> (6, 4, 5)
    squashed_dims = grid[:-2]
    parallel_grid = (math.prod(grid[:-2]), *grid[-2:])

  # We reverse the order because Pallas prefers row-major iteration while the
  # CUDA runtime prefers column-major iteration.
  parallel_grid = parallel_grid[::-1]
  cluster = cluster[::-1]
  squashed_dims = squashed_dims[::-1]
  axis_names = axis_names.reverse()

  rs = _estimate_resources(
      ResourceEstimatorContext(
          axis_names=axis_names, lowering_semantics=lowering_semantics
      ),
      jaxpr,
  )

  def body(launch_ctx: mgpu.LaunchContext, *buffers: ir.Value):
    *buffers_gmem, (
        runtime_smem,
        runtime_barriers,
        runtime_tmem,
        runtime_tmem_collective,
    ) = buffers
    gmem_semaphores = None
    if rs.gmem_semaphores:
      # Extract the semaphores local to the current block.
      index = ir.IndexType.get()
      block_idx = arith_dialect.index_castui(index, mgpu_utils.block_idx())
      gmem_semaphores = mgpu.memref_slice(
          buffers_gmem[-1],
          mgpu.ds(
              arith_dialect.muli(
                  block_idx, arith_dialect.constant(index, rs.gmem_semaphores)
              ),
              rs.gmem_semaphores,
          ),
      )
      # The semaphore buffer is an aliased input/output, so we need to skip it twice.
      buffers_gmem = buffers_gmem[:len(in_shapes)] + buffers_gmem[-len(out_shapes) - 1:-1]

    grouped_barriers = collections.defaultdict(list)
    for barrier, barrier_ref in zip(rs.barriers, runtime_barriers):
      grouped_barriers[barrier].append(barrier_ref)
    if runtime_tmem is not None:
      tmem_cols = math.prod(runtime_tmem.shape) // tcgen05.TMEM_ROWS
    else:
      tmem_cols = 0
    if runtime_tmem_collective is not None:
      tmem_collective_cols = (
          math.prod(runtime_tmem_collective.shape) // tcgen05.TMEM_ROWS
      )
    else:
      tmem_collective_cols = 0

    if lowering_semantics == mgpu.LoweringSemantics.Lane:
      single_wg_lane_predicate = mgpu.single_thread_predicate(
          scope=mgpu.ThreadSubset.WARPGROUP)
      single_warp_lane_predicate = mgpu.single_thread_predicate(
          scope=mgpu.ThreadSubset.WARP)
    else:  # Warpgroup semantics do not have a single lane predicate.
      single_wg_lane_predicate = None
      single_warp_lane_predicate = None

    module_ctx = ModuleContext(
        mlir.sanitize_name(debug_info.func_name),
        axis_names,
        [_program_id(axis, squashed_dims, len(grid)) for axis in range(len(grid))],
        approx_math,
        single_wg_lane_predicate,
        single_warp_lane_predicate,
        smem_requested_bytes=math.prod(ir.MemRefType(runtime_smem.type).shape),
        smem_used_bytes=0,
        tmem_requested_cols=tmem_cols,
        tmem_used_cols=0,
        tmem_base_ptr=runtime_tmem.address if runtime_tmem else None,
        tmem_collective_requested_cols=tmem_collective_cols,
        tmem_collective_used_cols=0,
        tmem_collective_base_ptr=runtime_tmem_collective.address
        if runtime_tmem_collective
        else None,
        gmem_used_semaphores=0,
        gmem_semaphore_base_ptr=gmem_semaphores,
        runtime_barriers=grouped_barriers,
        name_stack=source_info_util.NameStack(),
        traceback_caches=mlir.TracebackCaches(),
        squashed_dims=squashed_dims,
        lowering_semantics=lowering_semantics,
        primitive_semantics=gpu_core.PrimitiveSemantics.Warpgroup,
        mesh_info=pallas_utils.MeshInfo.from_mesh(jax_mesh) if jax_mesh is not None else None,
        auto_barriers=not params.unsafe_no_auto_barriers,
    )
    del runtime_smem, grouped_barriers, runtime_barriers
    _ = lower_jaxpr_to_mosaic_gpu(
        module_ctx, launch_ctx, jaxpr, buffers_gmem, consts
    )

  scratch_buffers = [
      jax.ShapeDtypeStruct(shape=[rs.smem_scratch_bytes], dtype=np.int8),
      rs.barriers,
  ]
  if rs.tmem_scratch_cols > 0:
    scratch_buffers.append(
        mgpu.TMEM(
            shape=(tcgen05.TMEM_ROWS, rs.tmem_scratch_cols),
            dtype=np.int32,
            collective=False,
        ),
    )
  else:
    scratch_buffers.append(None)
  if rs.tmem_collective_scratch_cols > 0:
    scratch_buffers.append(
        mgpu.TMEM(
            shape=(tcgen05.TMEM_ROWS, rs.tmem_collective_scratch_cols),
            dtype=np.int32,
            collective=True,
        ),
    )
  else:
    scratch_buffers.append(None)

  prof_ctx = prof_spec = None
  if params.profile_space:
    # Each range is 2 events, each event is 4 bytes.
    prof_spec = mgpu_profiler.ProfilerSpec(params.profile_space * 2 * 4)
    prof_ctx = ProfilerContext(params.profile_dir, prof_spec)
  cuda_grid = tuple(map(operator.mul, parallel_grid, cluster))
  semaphores_shape = ()
  if rs.gmem_semaphores:
    semaphores_shape = (
        jax.ShapeDtypeStruct(
            shape=(math.prod(cuda_grid) * rs.gmem_semaphores,), dtype=np.int32
        ),
    )
  # NOTE: new_out_shapes has out_shapes, then semaphores_shape and
  # optionally the profiler buffer.
  module, new_out_shapes, _, launch_ctx = (
      mgpu_core._lower_as_gpu_kernel(
          body,
          grid=cuda_grid,
          cluster=cluster,
          block=block,
          in_shapes=(*in_shapes, *semaphores_shape),
          out_shape=(*out_shapes, *semaphores_shape),
          inout_shape=(),
          smem_scratch_shape=scratch_buffers,
          lowering_semantics=lowering_semantics,
          module_name=mlir.sanitize_name(debug_info.func_name),
          kernel_name=mlir.sanitize_name(debug_info.func_name),
          prof_spec=prof_spec,
      )
  )

  if lowering_semantics == mgpu.LoweringSemantics.Warpgroup:
    # We need to run a pass that removes dead-code for which layout inference
    # does not work.
    pm = mlir.passmanager.PassManager.parse("builtin.module(canonicalize)", module.context)
    pm.run(module.operation)

    # Run Python lowering passes. The remaining passes will be run in C++ in
    # jax/jaxlib/mosaic/gpu/custom_call.cc
    mgpu.infer_layout(module)  # pytype: disable=attribute-error
    mgpu.infer_transforms(module)  # pytype: disable=attribute-error
    mgpu.lower_mgpu_dialect(
        module, launch_ctx, auto_barriers=not params.unsafe_no_auto_barriers
    )

  launch_ctx.scratch.finalize_size()

  return LoweringResult(
      module, cuda_grid, block, new_out_shapes, prof_ctx, semaphores_shape
  )


mosaic_lowering_rules = {
    # Lowering rules when using Mosaic GPU lane semantics.
    (mgpu.LoweringSemantics.Lane, gpu_core.PrimitiveSemantics.Warpgroup): {} ,
    gpu_core.LANExWARP_SEMANTICS: {} ,
    # Lowering rules when using Mosaic GPU warpgroup semantics.
    (mgpu.LoweringSemantics.Warpgroup,
     gpu_core.PrimitiveSemantics.Warpgroup): {},
}


def register_lowering_rule(
    primitive: jax_core.Primitive,
    lowering_semantics: mgpu.LoweringSemantics,
    primitive_semantics: gpu_core.PrimitiveSemantics = gpu_core.PrimitiveSemantics.Warpgroup,
):
  def deco(fn):
    mosaic_lowering_rules[
        (lowering_semantics, primitive_semantics)][primitive] = fn
    return fn

  return deco


def _compute_name_stack_updates(
    old_name_stack: list[str],
    new_name_stack: list[str]
) -> tuple[list[str], list[str]]:
  common_prefix_idx = 0
  for i, (old, new) in enumerate(unsafe_zip(old_name_stack, new_name_stack)):
    if old == new:
      common_prefix_idx = i+1
    else:
      break
  return old_name_stack[common_prefix_idx:], new_name_stack[common_prefix_idx:]


def lower_jaxpr_to_mosaic_gpu(
    module_ctx: ModuleContext,
    launch_ctx: mgpu.LaunchContext,
    jaxpr: jax_core.Jaxpr,
    args: Sequence[ir.Value],
    consts=(),
) -> Sequence[ir.Value]:
  env = {}

  def read_env(atom: jax_core.Atom):
    return atom.val if isinstance(atom, jax_core.Literal) else env[atom]

  def write_env(var: jax_core.Var, val, require_value: bool = True):
    env[var] = val
    # TODO(apaszke): Handle other avals (refs, etc.).
    if isinstance(aval := var.aval, jax_core.ShapedArray):
      # TODO(apaszke): Clarify the type invariants for lane semantics?
      if module_ctx.lowering_semantics == mgpu.LoweringSemantics.Warpgroup:
        # Shaped arrays must be vectors if and only if their shape is non-empty.
        # Those with empty shapes should be represented by their scalar type.
        mlir_dtype = mgpu_utils.dtype_to_ir_type(aval.dtype)
        if not isinstance(val, ir.Value):
          if require_value:
            raise AssertionError(f"Shaped arrays must be represented by ir.Values, got: {val}")
          else:
            if aval.shape:
              raise AssertionError("Only scalars can be represented by non-ir.Values")
            return  # Skip following checks.
        if aval.shape:
          if not ir.VectorType.isinstance(val.type):
            raise AssertionError(f"Non-scalar arrays must be represented by vectors, got: {val.type}")
          vty = ir.VectorType(val.type)
          if vty.element_type != mlir_dtype:
            raise AssertionError(f"Vector element type must match ShapedArray dtype, got: {val.type} != {mlir_dtype}")
          if tuple(vty.shape) != aval.shape:
            raise AssertionError(f"Vector shape must match ShapedArray shape, got: {vty.shape} != {aval.shape}")
        else:
          if ir.VectorType.isinstance(val.type):
            raise AssertionError(f"Scalars must be represented by non-vector types, got: {val.type}")
          if val.type != mlir_dtype:
            raise AssertionError(f"Scalar type must match ShapedArray dtype, got: {val.type} != {mlir_dtype}")

  foreach(
      functools.partial(write_env, require_value=False), jaxpr.constvars, consts
  )
  foreach(functools.partial(write_env, require_value=False), jaxpr.invars, args)

  # TODO(justinfu): Handle transform scopes.
  last_local_name_stack: list[str] = []
  named_regions = []
  for i, eqn in enumerate(jaxpr.eqns):
    invals = map(read_env, eqn.invars)
    eqn_name_stack = module_ctx.name_stack + eqn.source_info.name_stack
    loc = mlir.source_info_to_location(  # pytype: disable=wrong-arg-types
        module_ctx, eqn.primitive, eqn_name_stack, eqn.source_info.traceback
    )
    with source_info_util.user_context(eqn.source_info.traceback), loc:
      if eqn.primitive not in mosaic_lowering_rules[
          (module_ctx.lowering_semantics, module_ctx.primitive_semantics)]:
        raise NotImplementedError(
            "Unimplemented primitive in Pallas Mosaic GPU lowering: "
            f"{eqn.primitive.name} for lowering semantics "
            f"{module_ctx.lowering_semantics} and user thread semantics "
            f"{module_ctx.primitive_semantics}. "
            "Please file an issue on https://github.com/jax-ml/jax/issues."
        )
      new_local_name_stack = [scope.name for scope in eqn.source_info.name_stack.stack]
      popped, pushed = _compute_name_stack_updates(last_local_name_stack, new_local_name_stack)
      last_local_name_stack = new_local_name_stack
      for _ in popped:
        named_regions.pop().close()
      for name in pushed:
        wrapper_stack = contextlib.ExitStack()
        wrapper_stack.enter_context(launch_ctx.named_region(name))
        named_regions.append(wrapper_stack)
      rule = mosaic_lowering_rules[
          (module_ctx.lowering_semantics, module_ctx.primitive_semantics)
          ][eqn.primitive]
      # If the equation is immediately followed by a layout cast on its output,
      # we provide the layout as a hint to the rule.
      out_layout_hint = None
      if i + 1 < len(jaxpr.eqns):
        lookahead_eqn = jaxpr.eqns[i + 1]
        is_layout_cast = lookahead_eqn.primitive == gpu_core.layout_cast_p
        uses_eqn_output = lookahead_eqn.invars == eqn.outvars
        if is_layout_cast and uses_eqn_output:
          out_layout_hint = lookahead_eqn.params["new_layout"].to_mgpu()
      rule_ctx = LoweringRuleContext(
          module_ctx,
          launch_ctx,
          avals_in=[cast(jax_core.ShapedArray, v.aval) for v in eqn.invars],
          avals_out=[cast(jax_core.ShapedArray, v.aval) for v in eqn.outvars],
          prim=eqn.primitive,
          out_layout_hint=out_layout_hint,
      )
      try:
        outvals = rule(rule_ctx, *invals, **eqn.params)
      except LoweringError:
        raise  # We only add the extra info to the innermost exception.
      except Exception as e:
        if not pallas_call._verbose_errors_enabled():
          raise
        inval_types = map(lambda t: getattr(t, "type", None), invals)
        raise LoweringError(
            f"Exception while lowering eqn:\n  {eqn}\nWith context:\n "
            f" {rule_ctx}\nWith inval types={inval_types}\nIn jaxpr:\n{jaxpr}"
        ) from e
      if eqn.primitive.multiple_results:
        foreach(write_env, eqn.outvars, outvals)
      else:
        write_env(eqn.outvars[0], outvals)
  while named_regions:  # Drain the name stack.
    named_regions.pop().close()
  return map(read_env, jaxpr.outvars)


@register_lowering_rule(primitives.program_id_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(
    primitives.program_id_p, mgpu.LoweringSemantics.Warpgroup)
def _program_id_lowering_rule(ctx: LoweringRuleContext, axis):
  if ctx.module_ctx.program_ids is None:
    raise NotImplementedError("pl.program_id() is not supported in this context")
  return ctx.module_ctx.program_ids[axis]

def _unravel_program_id(
    block_id: ir.Value,
    axis: int,
    dimensions: tuple[int, ...],
    row_major: bool = False
) -> ir.Value:
  """Computes the program ID for axes compressed into one block dimension."""
  if row_major:
    div_value = math.prod(dimensions[axis+1:])
  else:
    div_value = math.prod(dimensions[:axis])
  div_value = _as_index(_i32_constant(div_value))
  pid = arith_dialect.divui(block_id, div_value)
  axis_size = _as_index(_i32_constant(dimensions[axis]))
  pid = arith_dialect.remui(pid, axis_size)
  return arith_dialect.index_cast(ir.IntegerType.get_signless(32), pid)


def _program_id(
    parallel_axis: int, squashed_dims: tuple[int, ...], grid_size: int
) -> ir.Value:
  """Returns the id of the current kernel instance along the given axis in the original Pallas grid."""
  if parallel_axis < len(squashed_dims):
    # All squashed dimensions are mapped to Dimension.z.
    block_id = gpu_dialect.block_id(gpu_dialect.Dimension.z)
    idx = len(squashed_dims) - 1 - parallel_axis
    return _unravel_program_id(block_id, idx, squashed_dims)
  else:
    idx = grid_size - 1 - parallel_axis
    assert idx in (0, 1, 2)
    return arith_dialect.index_cast(
        ir.IntegerType.get_signless(32),
        gpu_dialect.block_id(gpu_dialect.Dimension(idx)))


def _lower_fun(
    fun: Callable[..., Any], *, multiple_results: bool
) -> Callable[..., Any]:

  def lowering_rule(ctx: LoweringRuleContext, *args, **params):
    wrapped_fun = lu.wrap_init(
        fun
        if multiple_results
        else lambda *args, **params: (fun(*args, **params),),
        params,
        debug_info=api_util.debug_info(
            "Pallas Mosaic GPU lower_fun", fun, args, params
        ),
    )
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, ctx.avals_in)
    out = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, jaxpr, args, consts
    )
    return out if multiple_results else out[0]

  return lowering_rule

def _handle_dtype_bitcast(
    ref: ir.Value, src_dtype: ir.Type, dst_dtype: ir.Type
) -> ir.Value:
  """Allows bitcasting a SMEM ref from one element type to another.

  Args:
    ref: the reference to bitcast.
    src_dtype: the source element type.
    dst_dtype: the destination element type.

  Returns:
    A bitcasted version of `ref` with element type `dst_dtype`.

  Raises:
    ValueError: if the source ref is not in SMEM.
  """
  if src_dtype == dst_dtype:
    return ref
  if src_dtype != ir.IntegerType.get_signless(8):
    raise NotImplementedError(
        "Data type bitcast is only supported from i8 to other types."
    )
  ref_ty = ir.MemRefType(ref.type)
  if not mgpu_utils.is_smem_ref(ref_ty):
    raise ValueError(f"Only workgroup memory is supported but got {ref}.")
  if len(ref_ty.shape) != 1:
    raise NotImplementedError(
        "Data type bitcast is only supported for 1D arrays."
    )
  [stride], _ = ref_ty.get_strides_and_offset()
  if stride != 1:
    raise ValueError(
        "Data type bitcast is only supported for contiguous 1D arrays, but got "
        f"stride={stride}."
    )
  [shape_bytes] = ref_ty.shape
  shape_bitwidth = shape_bytes * 8
  target_bitwidth = mgpu_utils.bitwidth(dst_dtype)

  if shape_bitwidth % target_bitwidth:
    raise ValueError(
        f"Can not bitcast memory region of size {shape_bitwidth} bits to dtype "
        f"with {target_bitwidth} bits."
    )

  result_type = ir.MemRefType.get(
      shape=(shape_bitwidth // target_bitwidth,),
      element_type=dst_dtype,
      memory_space=ref_ty.memory_space,
  )

  # Do a memref_ptr/ptr_as_memref roundtrip instead of using `memref.view`,
  # which refuses to take in our source ref. This is because `memref.view` only
  # works on a super restricted set of `memref`s. E.g., it does not work if an
  # offset is specified, which can be the case for our SMEM refs.
  smem = mgpu_utils.WORKGROUP_NVPTX_ADDRESS_SPACE
  ref = mgpu_utils.memref_ptr(ref, memory_space=smem)
  return mgpu_utils.ptr_as_memref(ref, result_type, ptr_memory_space=smem)


def _extract_aliased_ref(
    ref: RefOrTmemType, transforms: Sequence[state_types.Transform]
) -> tuple[RefOrTmemType, Sequence[state_types.Transform]]:
  match transforms:
    case (
        gpu_core.ExtractAliasedRef(
            dtype, transformed_shape, offset, layout
        ),
        *other_transforms,
    ):
      mlir_dtype = mgpu_utils.dtype_to_ir_type(dtype)
      if isinstance(ref, tcgen05.TMEMRef):
        assert layout is not None
        if ref.shape[0] != transformed_shape[0]:
          raise ValueError(
              "TMEM aliasing only supported for Refs with the same first"
              f" dimension, got {ref.shape[0]} != {transformed_shape[0]}."
          )
        address = arith_dialect.addi(ref.address, _i32_constant(offset))
        ref = tcgen05.TMEMRef(
          address=address,
          shape=transformed_shape,
          dtype=mgpu_utils.dtype_to_ir_type(dtype),
          layout=layout)
      else:
        assert layout is None
        ref_bits = math.prod(transformed_shape) * mgpu_utils.bitwidth(mlir_dtype)
        if ref_bits % 8:
          raise NotImplementedError("Only byte-aligned bitcasts are supported.")
        assert offset % gpu_core.SMEM_ALIGNMENT == 0
        ref_bytes = ref_bits // 8
        ref = mgpu.memref_slice(ref, slice(offset, offset + ref_bytes))
        ref = _handle_dtype_bitcast(
            ref,
            ir.MemRefType(ref.type).element_type,
            mgpu_utils.dtype_to_ir_type(dtype),
        )
        ref = mgpu.memref_reshape(ref, transformed_shape)
      return ref, tuple(other_transforms)
    case _:
      return ref, transforms


def _transform_dtype(
    dtype: dtypes.DType,
    transforms: Sequence[state_types.Transform],
) -> dtypes.DType:
  """Applies `t.transform_dtype` for `t` in `transforms` sequentially on `dtype`."""
  for transform in transforms:
    dtype = transform.transform_dtype(dtype)
  assert dtype is not None
  return dtype  # pytype: disable=bad-return-type


def _handle_transforms(
    ctx: LoweringRuleContext,
    ref: RefOrTmemType,
    transforms: Sequence[state_types.Transform],
    *,
    handle_transposes=True,
    handle_reshapes=True,
    allow_peer_refs=False,
) -> tuple[RefOrTmemType, Sequence[state_types.Transform]]:
  # Before we handle other transforms, we resolve any possible leading
  # aliasing transform.
  ref, transforms = _extract_aliased_ref(ref, transforms)
  if isinstance(ref, tcgen05.TMEMRef):
    mlir_dtype = ref.dtype
  else:
    mlir_dtype = ir.MemRefType(ref.type).element_type
  transformed_ref = ref
  new_transforms = []
  def _bubble_up(untransform_fn, data):
    nonlocal new_transforms
    new_transforms_rev = []
    for t in reversed(new_transforms):
      data, new_t = untransform_fn(t, data)
      new_transforms_rev.append(new_t)

    new_transforms = list(reversed(new_transforms_rev))
    return data

  peer_device_id = None
  for t in transforms:
    match t:
      case indexing.NDIndexer():
        indexer = cast(indexing.NDIndexer, t)
        if indexer.int_indexer_shape:
          raise NotImplementedError("int_indexer_shape non-empty")
        indices = _ndindexer_indices(indexer)
        indices = _bubble_up(
            lambda t, idxs: t.untransform_index(mlir_dtype, idxs), indices
        )
        if isinstance(transformed_ref, tcgen05.TMEMRef):
          transformed_ref = transformed_ref.slice(*indices)
        else:
          transformed_ref = mgpu.memref_slice(transformed_ref, indices)
      case gpu_core.TransposeRef(perm) if handle_transposes:
        perm = _bubble_up(lambda t, p: t.untransform_transpose(p),
                                          perm)
        if isinstance(transformed_ref, tcgen05.TMEMRef):
          raise ValueError("TMEM transpose not allowed.")
        transformed_ref = mgpu.memref_transpose(transformed_ref, perm)
      case RefReshaper(dtype=dtype, shape=shape) if handle_reshapes:
        shape = _bubble_up(
            lambda t, p: t.untransform_reshape(dtype, p),  # pylint: disable=cell-var-from-loop
            shape)
        if isinstance(transformed_ref, tcgen05.TMEMRef):
          raise ValueError("TMEM reshape not allowed.")
        transformed_ref = mgpu.memref_reshape(transformed_ref, shape)
      case gpu_core.PeerMemRef(device_id, device_id_type):
        peer_device_id, other_axes = primitives.device_id_to_logical(
            ctx.module_ctx.mesh_info,
            device_id,
            device_id_type,
            lambda name: _axis_index_rule(ctx, axis_name=name),
        )
        if other_axes:
          raise ValueError(
              "Only JAX mesh axes can be used to obtain peer references, but"
              f" got {other_axes}"
          )
      case _:
        new_transforms.append(t)
  if peer_device_id is not None:
    if not allow_peer_refs:
      raise NotImplementedError(
          "Peer device references are not allowed in the lowering of this"
          " primitive."
      )
    transformed_ref = ctx.launch_ctx.to_remote(
        transformed_ref, _ensure_ir_value(peer_device_id, jnp.int32)
    )
  return transformed_ref, new_transforms


def _ndindexer_indices(
    indexer: indexing.NDIndexer, allow_arrays: bool = False
) -> tuple[gpu_core.Index | mgpu.FragmentedArray, ...]:
  indices = []
  for idx in indexer.indices:
    if isinstance(idx, mgpu.FragmentedArray) and idx.shape:
      if not allow_arrays:
        raise ValueError("Arrays are not supported as indices.")
      indices.append(idx)
    elif not isinstance(idx, indexing.Slice):
      indices.append(_as_index(idx))
    elif not idx.is_dynamic_start and not idx.is_dynamic_size:
      indices.append(slice(idx.start, idx.start + idx.size, idx.stride))
    elif idx.stride == 1:
      indices.append(
          mgpu.DynamicSlice(
              _as_index(idx.start) if idx.is_dynamic_start else idx.start,
              _as_index(idx.size) if idx.is_dynamic_size else idx.size,
          )
      )
    else:
      raise NotImplementedError(f"Unsupported slice: {idx}")
  return tuple(indices)


@register_lowering_rule(sp.get_p, mgpu.LoweringSemantics.Lane)
def _get_lowering_rule(
    ctx: LoweringRuleContext, x_ref, *leaves, tree, optimized=True
):
  if isinstance(x_ref, tcgen05.TMEMRef):
    raise RuntimeError(
        "Loads from TMEM are asynchronous operations and cannot be performed"
        " using the usual syntax. Please use plgpu.async_load_tmem instead."
    )
  if not isinstance(x_ref, ir.Value) and ir.MemRefType.isinstance(x_ref):
    raise TypeError(f"Can only load from references (got {x_ref}).")
  dtype = ctx.avals_out[0].dtype

  transforms = jax.tree.unflatten(tree, leaves)
  x_smem, transforms = _handle_transforms(
      ctx, x_ref, transforms, allow_peer_refs=True
  )
  del x_ref  # Don't use x_ref anymore. Use x_smem instead!

  is_signed = mgpu_utils.is_signed(dtype)

  if not ctx.avals_out[0].shape:  # The scalar case is simple.
    val = memref_dialect.load(x_smem, [])
    return mgpu.FragmentedArray.splat(val, shape=(), is_signed=is_signed)

  match transforms:
    case (gpu_core.UnswizzleRef(swizzle), gpu_core.UntileRef(tiling)):
      if len(tiling) != 2:
        raise NotImplementedError(f"Only 2D tiling is supported, got: {tiling}")
      expected_minor_tiling = swizzle * 8 // dtypes.bit_width(dtype)
      if tiling[-1] != expected_minor_tiling:
        raise NotImplementedError(
            "Minor tiling dimension does not fit swizzle: "
            f" expected {expected_minor_tiling}, got {tiling[-1]}"
        )
      layout = ctx.out_layout_hint or mgpu.WGMMA_LAYOUT
      return mgpu.FragmentedArray.load_tiled(
          x_smem, is_signed=is_signed, swizzle=swizzle, layout=layout, optimized=optimized
      )
    case ():
      match ctx.out_layout_hint:
        case mgpu.WGStridedFragLayout(shape=shape, vec_size=vec_size):
          ref_ty = ir.MemRefType(x_smem.type)
          if shape != tuple(ref_ty.shape):
            raise ValueError(
                f"Unsupported shape {shape}, (expected {tuple(ref_ty.shape)})"
            )
          return mgpu.FragmentedArray.load_strided(
              x_smem,
              is_signed=is_signed,
              vec_size=vec_size,
          )
        case None:
          return mgpu.FragmentedArray.load_strided(x_smem, is_signed=is_signed)
        case _:
          return mgpu.FragmentedArray.load_untiled(
              x_smem,
              is_signed=is_signed,
              layout=ctx.out_layout_hint,
              swizzle=16,
              optimized=optimized,
          )
    case _:
      raise NotImplementedError(f"Unsupported transforms: {transforms}")


@register_lowering_rule(sp.get_p, mgpu.LoweringSemantics.Warpgroup)
def _get_lowering_rule_wg(ctx: LoweringRuleContext, x_smem, *leaves, tree):
  if not isinstance(x_smem, ir.Value) and ir.MemRefType.isinstance(x_smem):
    raise TypeError(f"Can only load from references (got {x_smem}).")

  transforms = jax.tree.unflatten(tree, leaves)
  x_smem, transforms = _handle_transforms(
      ctx, x_smem, transforms, allow_peer_refs=True
  )
  assert isinstance(x_smem, ir.Value)
  mlir_dtype = ir.MemRefType(x_smem.type).element_type

  if transforms:
    raise NotImplementedError(
        "Transforms are not yet implemented for warpgroup semantics"
    )

  shape = ctx.avals_out[0].shape
  ty = ir.VectorType.get(shape, mlir_dtype)
  if shape:
    zero_index = arith_dialect.constant(ir.IndexType.get(), 0)
    indices = [zero_index for _ in range(len(shape))]
    return vector_dialect.load(ty, x_smem, indices)
  else:
    return memref_dialect.load(x_smem, [])


@register_lowering_rule(sp.swap_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(
    sp.swap_p, mgpu.LoweringSemantics.Lane, gpu_core.PrimitiveSemantics.Warp
)
def _swap_lowering_rule(
    ctx: LoweringRuleContext, x_ref, value, *leaves, tree
):
  if isinstance(x_ref, tcgen05.TMEMRef):
    raise RuntimeError(
        "Stores to TMEM are asynchronous operations and cannot be performed"
        " using the usual syntax. Please use plgpu.async_store_tmem instead."
    )
  barrier = mgpu.warpgroup_barrier
  if ctx.module_ctx.primitive_semantics == gpu_core.PrimitiveSemantics.Warp:
    if ctx.avals_out[0].shape:
      raise NotImplementedError("Can only store scalars in warp-level lowering.")
    i32 = ir.IntegerType.get_signless(32)
    barrier = functools.partial(
        nvvm_dialect.bar_warp_sync, arith_dialect.constant(i32, -1)
    )
  if not isinstance(value, mgpu.FragmentedArray):
    raise TypeError(f"Can only store arrays (got {value}).")

  if not isinstance(x_ref, ir.Value) and ir.MemRefType.isinstance(x_ref):
    raise TypeError(f"Can only store to references (got {x_ref}).")
  v_aval = ctx.avals_in[1]
  transforms = jax.tree.unflatten(tree, leaves)
  transposed_value = value.layout in (
      mgpu.WGMMA_TRANSPOSED_LAYOUT,
      mgpu.TCGEN05_TRANSPOSED_LAYOUT,
  )
  x_smem, transforms = _handle_transforms(
      ctx, x_ref, transforms, handle_transposes=not transposed_value,
      allow_peer_refs=True
  )
  del x_ref  # Don't use x_ref anymore. Use x_smem instead!

  if ctx.module_ctx.auto_barriers:
    barrier()  # Make sure reads have completed before we write.
  match transforms:
    case _ if not ctx.avals_out[0].shape:  # Scalar case.
      old_value = mgpu.FragmentedArray.splat(
          memref_dialect.load(x_smem, []),
          shape=(),
          is_signed=mgpu_utils.is_signed(v_aval.dtype),
      )
      memref_dialect.store(
          _ensure_ir_value(value, ctx.avals_out[0].dtype), x_smem, []
      )
    case (
        gpu_core.UnswizzleRef(swizzle),
        gpu_core.UntileRef(tiling),
        *maybe_transpose,
    ):
      if len(tiling) != 2:
        raise NotImplementedError(f"Only 2D tiling is supported, got: {tiling}")
      bw = dtypes.bit_width(v_aval.dtype)
      expected_minor_tiling = swizzle * 8 // bw
      if tiling[-1] != expected_minor_tiling:
        raise NotImplementedError(
            "Minor tiling dimension does not fit swizzle: "
            f" expected {expected_minor_tiling}, got {tiling[-1]}"
        )

      if transposed_value != bool(maybe_transpose):
        raise ValueError(
            "Either both the ref and the value are transposed or neither is."
        )

      if maybe_transpose:
        if maybe_transpose != [gpu_core.TransposeRef((1, 0))]:
          raise NotImplementedError(
              f"Unsupported transforms: {transforms} ({maybe_transpose})"
          )

        x_smem = mgpu.memref_transpose(x_smem, (1, 0, 3, 2))

      old_value = mgpu.FragmentedArray.load_tiled(
          x_smem,
          is_signed=mgpu_utils.is_signed(v_aval.dtype),
          swizzle=swizzle,
          layout=value.layout,
      )
      value.store_tiled(x_smem, swizzle=swizzle)
    case ():
      match value.layout:
        case mgpu.TiledLayout():
          old_value = mgpu.FragmentedArray.load_untiled(
              x_smem,
              layout=value.layout,
              is_signed=mgpu_utils.is_signed(v_aval.dtype),
              optimized=False,
          )
          value.store_untiled(x_smem, optimized=False)
        case _:
          old_value = mgpu.FragmentedArray.load_strided(
              x_smem, is_signed=mgpu_utils.is_signed(v_aval.dtype)
          )
          value.store_untiled(x_smem)
    case _:
      raise NotImplementedError(f"Unsupported transforms: {transforms}")
  if ctx.module_ctx.auto_barriers:
    barrier()  # Make sure the writes have completed.
  return old_value


@register_lowering_rule(sp.swap_p, mgpu.LoweringSemantics.Warpgroup)
def _swap_lowering_rule_wg(
    ctx: LoweringRuleContext, x_smem, value, *leaves, tree
):
  shape = ctx.avals_out[0].shape
  if shape and not ir.VectorType.isinstance(value.type):
    raise TypeError(f"Can only store scalars or vectors (got {value}).")
  if not (
      isinstance(x_smem, ir.Value) and ir.MemRefType.isinstance(x_smem.type)
  ):
    raise TypeError(f"Can only store to references (got {x_smem}).")
  transforms = jax.tree.unflatten(tree, leaves)
  x_smem, transforms = _handle_transforms(
      ctx, x_smem, transforms, allow_peer_refs=True)
  if transforms:
    raise NotImplementedError(
        "Transforms are not yet implemented for warpgroup semantics"
    )
  assert isinstance(x_smem, ir.Value)
  x_mlir_dtype = ir.MemRefType(x_smem.type).element_type
  ty = ir.VectorType.get(shape, x_mlir_dtype)
  if shape:
    zero_index = arith_dialect.constant(ir.IndexType.get(), 0)
    indices = [zero_index for _ in range(len(shape))]
    old_value = vector_dialect.load(ty, x_smem, indices)
    vector_dialect.store(value, x_smem, indices)
  else:
    old_value = memref_dialect.load(x_smem, [])
    memref_dialect.store(value, x_smem, [])
  return old_value


@register_lowering_rule(pjit.jit_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(pjit.jit_p, mgpu.LoweringSemantics.Warpgroup)
def _pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **kwargs):
  if jaxpr.consts:
    raise NotImplementedError
  return lower_jaxpr_to_mosaic_gpu(
      ctx.module_ctx, ctx.launch_ctx, jaxpr.jaxpr, args,
  )


@register_lowering_rule(lax.slice_p, mgpu.LoweringSemantics.Lane)
def _slice_lowering_rule(
    ctx: LoweringRuleContext, x, limit_indices, start_indices, strides
):
  if strides is not None:
    raise NotImplementedError("Strides are not supported.")

  return x[tuple(slice(b, e) for b, e in zip(start_indices, limit_indices))]


@register_lowering_rule(lax.select_n_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.select_n_p, mgpu.LoweringSemantics.Lane,
                        gpu_core.PrimitiveSemantics.Warp)
@register_lowering_rule(lax.select_n_p, mgpu.LoweringSemantics.Warpgroup)
def _select_n_lowering_rule(ctx: LoweringRuleContext, pred, *cases):
  if len(cases) != 2:
    raise NotImplementedError(
        "Mosaic GPU lowering only supports select_n with 2 cases, got"
        f" {len(cases)}"
    )
  pred_aval, *cases_avals = ctx.avals_in
  if ctx.module_ctx.primitive_semantics == gpu_core.PrimitiveSemantics.Warp:
    if not all(aval.shape == () for aval in ctx.avals_in):
      raise NotImplementedError(
          "Can only select on scalars in warp-level lowering.")
  [out_aval] = ctx.avals_out
  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    pred = _ensure_fa(pred, pred_aval.dtype)
    cases = _bcast(*cases, *cases_avals, out_aval)
    # ``select`` expects the first case to be the true branch, but ``select_n``
    # orders the cases in reverse.
    return pred.select(*reversed(cases))
  else:
    pred = _ensure_ir_value(pred, pred_aval.dtype)
    cases = [_ensure_ir_value(c, c_aval.dtype) for c, c_aval in zip(cases, cases_avals)]
    # TODO(bchetioui): support implicit broadcast.
    if any(a.shape != out_aval.shape for a in ctx.avals_in):
      raise NotImplementedError(
          "Implicit broadcast not implemented with warpgroup semantics")
    # ``select`` expects the first case to be the true branch, but ``select_n``
    # orders the cases in reverse.
    return arith_dialect.select(pred, *reversed(cases))


@register_lowering_rule(lax.broadcast_in_dim_p, mgpu.LoweringSemantics.Lane)
def _broadcast_in_dim_lowering_rule(
    ctx: LoweringRuleContext,
    x: mgpu.FragmentedArray,
    *,
    broadcast_dimensions,
    shape,
    sharding,
):
  del sharding
  [x_aval] = ctx.avals_in
  [y_aval] = ctx.avals_out
  x = _ensure_fa(x, x_aval.dtype)
  rank_diff = y_aval.ndim - x_aval.ndim
  if (isinstance(x.layout, mgpu.WGSplatFragLayout) and
      broadcast_dimensions == tuple(range(rank_diff, rank_diff + x_aval.ndim))):
    return x.broadcast(shape)
  if not isinstance(layout := x.layout, mgpu.TiledLayout):
    raise NotImplementedError(f"Unsupported layout: {x.layout}")
  if any(d1 >= d2 for d1, d2 in zip(broadcast_dimensions[:-1], broadcast_dimensions[1:])):
    raise NotImplementedError("broadcast_dimensions must be strictly increasing")
  new_dims = [d for d in range(y_aval.ndim) if d not in broadcast_dimensions]
  if (new_layout := ctx.out_layout_hint) is None:
    candidates = (
      mgpu.WGMMA_LAYOUT,
      mgpu.WGMMA_TRANSPOSED_LAYOUT,
      mgpu.TCGEN05_LAYOUT,
      mgpu.TCGEN05_TRANSPOSED_LAYOUT,
      tcgen05.TMEM_NATIVE_LAYOUT,
      tcgen05.fa_m64_collective_layout(y_aval.shape[-1]),
    )
    for candidate in candidates:
      if len(candidate.base_tile_shape) != len(shape):
        continue
      if candidate.reduce(new_dims) == layout:
        if new_layout is None:
          new_layout = candidate
        elif candidate == mgpu.TCGEN05_LAYOUT and new_layout == mgpu.WGMMA_LAYOUT:
          continue  # Choosing WGMMA_LAYOUT for backwards compatibility.
        else:
          raise NotImplementedError(
              "Multiple options for the layout of the broadcast result (found"
              f" at least {new_layout} and {candidate}). Use plgpu.layout_cast"
              " on the output to suggest the desired output layout."
          )
  if new_layout is None:
    raise NotImplementedError(
        "No compatible layout found for the broadcast result. Use"
        " plgpu.layout_cast on the output to suggest the desired output layout."
    )
  return x.broadcast_in_dim(y_aval.shape, broadcast_dimensions, new_layout)


@register_lowering_rule(
    lax.broadcast_in_dim_p, mgpu.LoweringSemantics.Warpgroup)
def _broadcast_in_dim_lowering_rule_wg(
    ctx: LoweringRuleContext,
    x,
    *,
    broadcast_dimensions,
    shape,
    sharding,
):
  del sharding

  [x_aval] = ctx.avals_in

  if not broadcast_dimensions:
    # Even though we could implement this case by passing a 0D vector as input
    # to mgpu.dialect.BroadcastInDimOp we don't want that. 0D vectors are
    # generally problematic and so we avoid them by specializing that case
    # directly here.
    x = _ensure_ir_value(x, x_aval.dtype)
    return vector_dialect.broadcast(
        ir.VectorType.get(shape, mgpu_utils.dtype_to_ir_type(x_aval.dtype)),
        x,
    )
  mlir_type = mgpu_utils.dtype_to_ir_type(x_aval.dtype)
  result_ty = ir.VectorType.get(shape, mlir_type)
  return mgpu.dialect.broadcast_in_dim(result_ty, x, broadcast_dimensions)


@register_lowering_rule(lax.convert_element_type_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.convert_element_type_p,
  mgpu.LoweringSemantics.Lane, gpu_core.PrimitiveSemantics.Warp)
def _convert_element_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding
):
  del weak_type, sharding
  [x_aval] = ctx.avals_in
  if ctx.module_ctx.primitive_semantics == gpu_core.PrimitiveSemantics.Warp:
    if x_aval.shape != ():
      raise NotImplementedError(
          "Non-scalar arithmetic is not supported in warp-level lowering.")
  return _ensure_fa(x, x_aval.dtype).astype(
      mgpu_utils.dtype_to_ir_type(new_dtype), is_signed=mgpu_utils.is_signed(new_dtype)
  )


@register_lowering_rule(
    lax.convert_element_type_p, mgpu.LoweringSemantics.Warpgroup)
def _convert_element_type_lowering_rule_wg(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding
):
  del weak_type, sharding
  [x_aval] = ctx.avals_in
  [y_aval] = ctx.avals_out
  x = _ensure_ir_value(x, x_aval.dtype)

  cur_dtype = mgpu_utils.dtype_to_ir_type(x_aval.dtype)
  new_dtype = mgpu_utils.dtype_to_ir_type(new_dtype)

  if cur_dtype == new_dtype:
    return x

  if 1 < mgpu_utils.bitwidth(cur_dtype) < 8 or 1 < mgpu_utils.bitwidth(new_dtype) < 8:
    raise NotImplementedError("Conversion involving sub-byte types unsupported")

  from_float = ir.FloatType.isinstance(cur_dtype)
  to_float = ir.FloatType.isinstance(new_dtype)
  from_integer = ir.IntegerType.isinstance(cur_dtype)
  to_integer = ir.IntegerType.isinstance(new_dtype)
  if from_float and to_float:
    cur_ty_width = ir.FloatType(cur_dtype).width
    new_ty_width = ir.FloatType(new_dtype).width
    if cur_ty_width == new_ty_width:
      # There is no instruction to perform conversions between two float types
      # of the same width. Go through the next-larger standard type.
      # TODO(bchetioui): support conversions between float types of width 8.
      # Which larger type to pick will depend on the number of bits in the
      # smallest exponent.
      if cur_ty_width != 16:
        raise NotImplementedError(
            "Conversion between float types of width other than 16 not"
            " supported"
        )
      larger_ty = ir.F32Type.get()
      if x_aval.shape:
        upcast_ty = ir.VectorType.get(x_aval.shape, larger_ty)
      else:
        upcast_ty = larger_ty

      def convert(ty, x):
        return arith_dialect.truncf(ty, arith_dialect.extf(upcast_ty, x))

    elif ir.FloatType(cur_dtype).width > ir.FloatType(new_dtype).width:
      convert = arith_dialect.truncf
    else:
      convert = arith_dialect.extf
  elif from_integer and to_integer:
    if ir.IntegerType(cur_dtype).width > ir.IntegerType(new_dtype).width:
      convert = arith_dialect.trunci
    elif ir.IntegerType(cur_dtype).width < ir.IntegerType(new_dtype).width:
      if mgpu_utils.is_signed(x_aval.dtype):
        convert = arith_dialect.extsi
      else:
        convert = arith_dialect.extui
    else:
      convert = lambda _, x: x  # signed <-> unsigned conversions
  elif from_integer and to_float:
    if mgpu_utils.is_signed(x_aval.dtype):
      convert = arith_dialect.sitofp
    else:
      convert = arith_dialect.uitofp
  elif from_float and to_integer:
    dst_width = mgpu_utils.bitwidth(new_dtype)
    # We clamp the float value to the min/max integer destination value
    # in order to match JAX/XLA casting behavior. Note that this differs
    # from numpy casting behavior.
    if mgpu_utils.is_signed(y_aval.dtype):
      maxint = 2 ** (dst_width - 1) - 1
      minint = -(2 ** (dst_width - 1))
      convert = arith_dialect.fptosi
    else:
      maxint = 2**dst_width - 1
      minint = 0
      convert = arith_dialect.fptoui

    maxint = _ir_constant(maxint, cur_dtype)
    minint = _ir_constant(minint, cur_dtype)
    if x_aval.shape:
      maxint = vector_dialect.broadcast(x.type, maxint)
      minint = vector_dialect.broadcast(x.type, minint)
    x = arith_dialect.minimumf(x, maxint)
    x = arith_dialect.maximumf(x, minint)
  else:
    raise NotImplementedError(f"Unsupported conversion {cur_dtype} -> {new_dtype}")

  ty = ir.VectorType.get(x_aval.shape, new_dtype) if x_aval.shape else new_dtype
  return convert(ty, x)


mosaic_lowering_rules[gpu_core.LANExWG_SEMANTICS].update({
    lax.neg_p: lambda ctx, x: -x,
    lax.not_p: lambda ctx, x: ~x,
})

def _unary_warp_lowering_rule(impl):
  def _lowering_rule(ctx: LoweringRuleContext, x):
    if not all(aval_in.shape == () for aval_in in ctx.avals_in):
      raise NotImplementedError(
          "Non-scalar arithmetic is not supported in warp-level lowering.")
    return impl(x)
  return _lowering_rule

mosaic_lowering_rules[gpu_core.LANExWARP_SEMANTICS].update({
    lax.neg_p: _unary_warp_lowering_rule(lambda x: -x),
    lax.not_p: _unary_warp_lowering_rule(lambda x: ~x)
})

mosaic_lowering_rules[gpu_core.WGxWG_SEMANTICS].update({
    lax.neg_p: _lower_fun(lambda x: jnp.subtract(0, x), multiple_results=False),
    lax.not_p: _lower_fun(
        lambda x: jnp.astype(jnp.bitwise_xor(jnp.astype(x, int), -1), jnp.dtype(x)), multiple_results=False,
    ),
})


def _binary_op_lowering_rule(ctx: LoweringRuleContext, x, y, *, impl):
  if ctx.module_ctx.primitive_semantics == gpu_core.PrimitiveSemantics.Warp:
    if not all(aval_in.shape == () for aval_in in ctx.avals_in):
      raise NotImplementedError(
          "Non-scalar arithmetic is not supported in warp-level lowering.")
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return impl(x, y)


def _div(x, y):
  return x / y if ir.FloatType.isinstance(x.mlir_dtype) else x // y


for semantics in [gpu_core.LANExWG_SEMANTICS, gpu_core.LANExWARP_SEMANTICS]:
  mosaic_lowering_rules[semantics].update({
    lax.add_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x + y),
    lax.sub_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x - y),
    lax.mul_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x * y),
    lax.div_p: partial(_binary_op_lowering_rule, impl=_div),
    lax.rem_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x % y),
    lax.and_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x & y),
    lax.or_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x | y),
    lax.xor_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x ^ y),
    lax.gt_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x > y),
    lax.lt_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x < y),
    lax.ge_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x >= y),
    lax.le_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x <= y),
    lax.eq_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x == y),
    lax.ne_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x != y),
    lax.max_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x.max(y)),
    lax.min_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x.min(y)),
  })

def _binary_op_lowering_rule_wg(
    ctx: LoweringRuleContext, x, y, *, ui_impl, si_impl, f_impl=None
):
  x_aval, y_aval = ctx.avals_in
  [out_aval] = ctx.avals_out
  x, y = _bcast_wg(x, y, *ctx.avals_in, *ctx.avals_out)
  if jnp.issubdtype(out_aval, jnp.signedinteger):
    return si_impl(x, y)
  elif jnp.issubdtype(out_aval, jnp.integer):
    return ui_impl(x, y)
  elif f_impl is not None and jnp.issubdtype(out_aval, jnp.floating):
    return f_impl(x, y)
  else:
    raise NotImplementedError(
        f"{ctx.prim} does not support {x_aval.dtype} and {y_aval.dtype}"
    )


for op, si_impl, ui_impl, f_impl in [
    (lax.add_p, arith_dialect.addi, arith_dialect.addi, arith_dialect.addf),
    (lax.sub_p, arith_dialect.subi, arith_dialect.subi, arith_dialect.subf),
    (lax.mul_p, arith_dialect.muli, arith_dialect.muli, arith_dialect.mulf),
    (
        lax.div_p,
        arith_dialect.floordivsi,
        arith_dialect.divui,
        arith_dialect.divf,
    ),
    (lax.rem_p, arith_dialect.remsi, arith_dialect.remui, arith_dialect.remf),
    (
        lax.max_p,
        arith_dialect.maxsi,
        arith_dialect.maxui,
        arith_dialect.maximumf,
    ),
    (
        lax.min_p,
        arith_dialect.minsi,
        arith_dialect.minui,
        arith_dialect.minimumf,
    ),
]:
  mosaic_lowering_rules[gpu_core.WGxWG_SEMANTICS][op] = partial(
      _binary_op_lowering_rule_wg,
      si_impl=si_impl,
      ui_impl=ui_impl,
      f_impl=f_impl,
  )


def _binary_boolean_op_lowering_rule_wg(
    ctx: LoweringRuleContext, x, y, *, impl
):
  x, y = _bcast_wg(x, y, *ctx.avals_in, *ctx.avals_out)
  return impl(x, y)

for op, impl in [
    (lax.and_p, arith_dialect.andi),
    (lax.or_p, arith_dialect.ori),
    (lax.xor_p, arith_dialect.xori),
]:
  mosaic_lowering_rules[gpu_core.WGxWG_SEMANTICS][op] = partial(
      _binary_boolean_op_lowering_rule_wg,
      impl=impl,
  )

CmpIPred = arith_dialect.CmpIPredicate
CmpFPred = arith_dialect.CmpFPredicate

def _comparison_lowering_rule_wg(
    ctx: LoweringRuleContext, x, y, *, si_pred, ui_pred, f_pred
):
  x_aval, y_aval = ctx.avals_in
  x, y = _bcast_wg(x, y, *ctx.avals_in, *ctx.avals_out)
  if jnp.issubdtype(x_aval, jnp.signedinteger):
    return arith_dialect.cmpi(si_pred, x, y)
  elif jnp.issubdtype(x_aval, jnp.unsignedinteger) or jnp.issubdtype(x_aval, jnp.bool):
    return arith_dialect.cmpi(ui_pred, x, y)
  elif jnp.issubdtype(x_aval, jnp.floating):
    return arith_dialect.cmpf(f_pred, x, y)
  else:
    raise NotImplementedError(
        f"{ctx.prim} does not support {x_aval.dtype} and {y_aval.dtype}"
    )


for op, si_pred, ui_pred, f_pred in [
    (lax.eq_p, CmpIPred.eq, CmpIPred.eq, CmpFPred.OEQ),
    (lax.ne_p, CmpIPred.ne, CmpIPred.ne, CmpFPred.UNE),
    (lax.lt_p, CmpIPred.slt, CmpIPred.ult, CmpFPred.OLT),
    (lax.le_p, CmpIPred.sle, CmpIPred.ule, CmpFPred.OLE),
    (lax.gt_p, CmpIPred.sgt, CmpIPred.ugt, CmpFPred.OGT),
    (lax.ge_p, CmpIPred.sge, CmpIPred.uge, CmpFPred.OGE),
]:
  mosaic_lowering_rules[gpu_core.WGxWG_SEMANTICS][op] = partial(
      _comparison_lowering_rule_wg,
      si_pred=si_pred,
      ui_pred=ui_pred,
      f_pred=f_pred,
  )

@register_lowering_rule(lax.integer_pow_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.integer_pow_p, mgpu.LoweringSemantics.Warpgroup)
def _integer_pow_lowering_rule(ctx: LoweringRuleContext, x, y):
  [x_aval] = ctx.avals_in
  if y <= 1:
    raise NotImplementedError

  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    mul_op = operator.mul
  elif jnp.issubdtype(x_aval.dtype, jnp.integer):
    mul_op = arith_dialect.muli
  elif jnp.issubdtype(x_aval.dtype, jnp.floating):
    mul_op = arith_dialect.mulf
  else:
    raise NotImplementedError(f"Unsupported dtype {x_aval.dtype}")

  # Y is an integer. Here we start with res = x so the range is y-1
  res = x
  # Repeated doubling algorithm.
  for i in reversed(range(y.bit_length() - 1)):
    res = mul_op(res, res)
    if (y >> i) & 1:
      res = mul_op(res, x)
  return res


@register_lowering_rule(lax.square_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.square_p, mgpu.LoweringSemantics.Warpgroup)
def _square_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    x = _ensure_fa(x, x_aval.dtype)
    return x * x
  if jnp.issubdtype(x_aval.dtype, jnp.integer):
    return arith_dialect.muli(x, x)
  if jnp.issubdtype(x_aval.dtype, jnp.floating):
    return arith_dialect.mulf(x, x)
  raise NotImplementedError(f"Unsupported dtype {x_aval.dtype}")


@register_lowering_rule(lax.rsqrt_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.rsqrt_p, mgpu.LoweringSemantics.Warpgroup)
def _rsqrt_lowering_rule(ctx: LoweringRuleContext, x, accuracy):
  if accuracy is not None:
    raise NotImplementedError("Not implemented: accuracy")
  [x_aval] = ctx.avals_in
  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    return _ensure_fa(x, x_aval.dtype).rsqrt(approx=ctx.module_ctx.approx_math)
  fastmath = (
      arith_dialect.FastMathFlags.afn if ctx.module_ctx.approx_math else None
  )
  return math_dialect.rsqrt(
      _ensure_ir_value(x, x_aval.dtype), fastmath=fastmath
  )


@register_lowering_rule(lax.tanh_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.tanh_p, mgpu.LoweringSemantics.Warpgroup)
def _tanh_lowering_rule(ctx: LoweringRuleContext, x, accuracy):
  if accuracy is not None:
    raise NotImplementedError("Not implemented: accuracy")
  [x_aval] = ctx.avals_in
  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    return _ensure_fa(x, x_aval.dtype).tanh(approx=ctx.module_ctx.approx_math)
  fastmath = (
      arith_dialect.FastMathFlags.afn if ctx.module_ctx.approx_math else None
  )
  return math_dialect.tanh(_ensure_ir_value(x, x_aval.dtype), fastmath=fastmath)


def _logistic(x, accuracy):
  if accuracy is not None:
    raise NotImplementedError("Not implemented: accuracy")
  return 1.0 / (1 + lax.exp(-x))


mosaic_lowering_rules[gpu_core.LANExWG_SEMANTICS][lax.logistic_p] = _lower_fun(
    _logistic, multiple_results=False
)
mosaic_lowering_rules[gpu_core.WGxWG_SEMANTICS][lax.logistic_p] = (
    _lower_fun(_logistic, multiple_results=False)
)


@register_lowering_rule(lax.exp_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.exp_p, mgpu.LoweringSemantics.Warpgroup)
def _exp_lowering_rule(ctx: LoweringRuleContext, x, accuracy):
  if accuracy is not None:
    raise NotImplementedError("Not implemented: accuracy")
  [x_aval] = ctx.avals_in
  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    return _ensure_fa(x, x_aval.dtype).exp(approx=ctx.module_ctx.approx_math)
  fastmath = (
      arith_dialect.FastMathFlags.afn if ctx.module_ctx.approx_math else None
  )
  return math_dialect.exp(_ensure_ir_value(x, x_aval.dtype), fastmath=fastmath)


@register_lowering_rule(lax.exp2_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.exp2_p, mgpu.LoweringSemantics.Warpgroup)
def _exp2_lowering_rule(ctx: LoweringRuleContext, x, accuracy):
  if accuracy is not None:
    raise NotImplementedError("Not implemented: accuracy")
  [x_aval] = ctx.avals_in
  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    return _ensure_fa(x, x_aval.dtype).exp2(approx=ctx.module_ctx.approx_math)
  fastmath = (
      arith_dialect.FastMathFlags.afn if ctx.module_ctx.approx_math else None
  )
  return math_dialect.exp2(_ensure_ir_value(x, x_aval.dtype), fastmath=fastmath)


@register_lowering_rule(lax.log_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.log_p, mgpu.LoweringSemantics.Warpgroup)
def _log_lowering_rule(ctx: LoweringRuleContext, x, accuracy):
  if accuracy is not None:
    raise NotImplementedError("Not implemented: accuracy")
  [x_aval] = ctx.avals_in
  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    return _ensure_fa(x, x_aval.dtype).log(approx=ctx.module_ctx.approx_math)
  fastmath = (
      arith_dialect.FastMathFlags.afn if ctx.module_ctx.approx_math else None
  )
  return math_dialect.log(_ensure_ir_value(x, x_aval.dtype), fastmath=fastmath)


def _reduce_lowering_rule(op, ctx: LoweringRuleContext, x, *, axes):
  [x_aval] = ctx.avals_in
  match x.layout:
    case mgpu.WGStridedFragLayout():
      if set(axes) != set(range(x_aval.ndim)):
        raise NotImplementedError("No support for axes yet")
      # To relax the restriction below, you need to ensure sufficient
      # synchronization with other places that use `scratch_view` (which at the
      # time of writing is only `run_scoped`).
      if ctx.module_ctx.axis_names.wg is not None:
        raise NotImplementedError(
            "No support for reduce_sum over all axes and multiple Pallas"
            " threads"
        )
      scratch_ty = jax.ShapeDtypeStruct(shape=(4,), dtype=x_aval.dtype)
      with ctx.module_ctx.scratch_view([scratch_ty]) as [scratch]:
        return x.reduce(op, axes, scratch)
    case mgpu.TiledLayout():
      if len(axes) != 1:
        raise NotImplementedError("Multi-axis reductions not supported")
      reduced_dim = x.layout.tiling.tile_dimension(axes[0])
      if any(reduced_dim[d] for d in x.layout.partitioned_warp_dims):
        scratch_ty = jax.ShapeDtypeStruct(shape=(REDUCE_SCRATCH_ELEMS,), dtype=x_aval.dtype)
        ctx = ctx.module_ctx.scratch_view([scratch_ty])
      else:
        ctx = contextlib.nullcontext([None])
      with ctx as [scratch]:
        return x.reduce(op, axes[0], scratch=scratch)
    case _:
      raise NotImplementedError(f"Unsupported layout {x.layout}")

register_lowering_rule(lax.reduce_sum_p, mgpu.LoweringSemantics.Lane)(
    functools.partial(_reduce_lowering_rule, "add")
)
register_lowering_rule(lax.reduce_max_p, mgpu.LoweringSemantics.Lane)(
    functools.partial(_reduce_lowering_rule, "max")
)

def _reduce_lowering_rule_wg(
    kind: vector_dialect.CombiningKind,
    acc: object,
    ctx: LoweringRuleContext,
    x,
    *,
    axes,
) -> ir.OpView:
  [x_aval] = ctx.avals_in
  [out_aval] = ctx.avals_out
  x = _ensure_ir_value(x, x_aval.dtype)
  out_type = mgpu_utils.dtype_to_ir_type(out_aval.dtype)
  if not out_aval.shape:
    # Special-case: reducing to a scalar.
    if x_aval.ndim != 1:
      # Flatten to 1D, since vector.reduction only supports 1D inputs.
      x = vector_dialect.shape_cast(
          ir.VectorType.get([x_aval.size], out_type), x
      )
    return vector_dialect.ReductionOp(out_type, kind, x)
  acc = vector_dialect.broadcast(
      ir.VectorType.get(out_aval.shape, out_type),
      _ensure_ir_value(acc, out_aval.dtype),
  )
  return vector_dialect.MultiDimReductionOp(kind, x, acc, axes)


@register_lowering_rule(lax.reduce_sum_p, mgpu.LoweringSemantics.Warpgroup)
def _reduce_sum_lowering_rule_wg(ctx: LoweringRuleContext, x, *, axes):
  op = _reduce_lowering_rule_wg(
      vector_dialect.CombiningKind.ADD, 0, ctx, x, axes=axes
  )
  op.attributes["offset"] = ir.IntegerAttr.get(
      ir.IntegerType.get_signless(32), ctx.module_ctx.smem_used_bytes
  )
  return op.result


@register_lowering_rule(lax.reduce_max_p, mgpu.LoweringSemantics.Warpgroup)
def _reduce_max_lowering_rule_wg(ctx: LoweringRuleContext, x, *, axes):
  [x_aval] = ctx.avals_in
  if jnp.issubdtype(x_aval.dtype, jnp.floating):
    kind = vector_dialect.CombiningKind.MAXIMUMF
    acc = float("-inf")
  elif jnp.issubdtype(x_aval.dtype, jnp.signedinteger):
    kind = vector_dialect.CombiningKind.MAXSI
    acc = np.iinfo(x_aval.dtype).max
  elif jnp.issubdtype(x_aval.dtype, jnp.unsignedinteger):
    kind = vector_dialect.CombiningKind.MAXUI
    acc = np.iinfo(x_aval.dtype).max
  else:
    raise NotImplementedError(f"Unsupported dtype {x_aval.dtype}")
  return _reduce_lowering_rule_wg(kind, acc, ctx, x, axes=axes).result


def _block_id(ctx: LoweringRuleContext, dim: gpu_dialect.Dimension) -> ir.Value:
  result = gpu_dialect.block_id(dim)
  cluster_size = ctx.launch_ctx.cluster_size
  if math.prod(cluster_size) == 1 or cluster_size[dim.value] == 1:
    return result
  # We scale the grid in the presence of clusters, so we need to scale the
  # block ID back here.
  return arith_dialect.divui(result, _as_index(cluster_size[dim.value]))


def _resolve_cluster_axis(axis_names: _AxisNames | None, axis_name: str):
  if not axis_names:
    raise LookupError(
        "No axis names are available. Make sure you are using `pl.core_map`"
        " with a `plgpu.Mesh`."
    )
  if not axis_names or axis_name not in axis_names.cluster:
    raise LookupError(
        f"Unknown cluster axis {axis_name}, available axes:"
        f" {[*axis_names.cluster]}"
    )
  return gpu_dialect.Dimension(axis_names.cluster.index(axis_name))


@register_lowering_rule(lax.axis_index_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.axis_index_p, mgpu.LoweringSemantics.Lane, gpu_core.PrimitiveSemantics.Warp)
@register_lowering_rule(lax.axis_index_p, mgpu.LoweringSemantics.Warpgroup)
def _axis_index_rule(ctx: LoweringRuleContext, *, axis_name: Hashable):
  if ctx.module_ctx.primitive_semantics == gpu_core.PrimitiveSemantics.Warp:
    if axis_name == ctx.module_ctx.warp_axis_name:
      return mgpu.warp_idx(sync=True)
    raise ValueError(
        "Named axes can only refer to the warp axis name inside of core_map."
    )
  gpu_axis_names = ctx.module_ctx.axis_names
  jax_axis_names = getattr(ctx.module_ctx.mesh_info, "axis_names", ())
  if gpu_axis_names is None and not jax_axis_names:
    raise LookupError(
        "No axis names are available. Make sure you are using `pl.core_map`"
        " with a `plgpu.Mesh` or an appropriate JAX device mesh."
    )
  if axis_name not in itertools.chain(gpu_axis_names or (), jax_axis_names):
    raise LookupError(
        f"Axis {axis_name} does not refer to a GPU mesh axis (available axes:"
        f" {[*gpu_axis_names]}) or a JAX mesh axis (available axes:"
        f" {[*jax_axis_names]})"
    )
  if axis_name in jax_axis_names:
    jax_mesh = ctx.module_ctx.mesh_info
    assert jax_mesh is not None
    device_id = ctx.launch_ctx.device_id()
    jax_mesh_shape = jax_mesh.mesh_shape
    axis_index = jax_axis_names.index(axis_name)
    i32 = ir.IntegerType.get_signless(32)
    axis_size = _ir_constant(jax_mesh_shape[axis_index], i32)
    minor_divisor = _ir_constant(
        np.prod(jax_mesh_shape[axis_index + 1 :], dtype=np.int32), i32
    )
    return arith_dialect.remsi(arith_dialect.divsi(device_id, minor_divisor), axis_size)

  # We already checked that the axis is in scope and it wasn't a JAX mesh axis.
  assert gpu_axis_names is not None

  # We only deal with GPU axes from now on.
  axis_names = gpu_axis_names
  if axis_names.wg is not None and axis_name == axis_names.wg:
    return mgpu.warpgroup_idx(sync=True)

  if axis_name in axis_names.cluster:
    return arith_dialect.index_cast(
        ir.IntegerType.get_signless(32),
        gpu_dialect.cluster_block_id(
            gpu_dialect.Dimension(axis_names.cluster.index(axis_name))
        ),
    )

  squashed_dims = ctx.module_ctx.squashed_dims
  if squashed_dims:
    unsquashed_names = axis_names.grid[:2]
    squashed_names = axis_names.grid[2:]
  else:
    # These are unused but initialized for type checkers.
    unsquashed_names = squashed_names = ()

  if squashed_dims:
    if axis_name in unsquashed_names:
      # We reversed the grid and cluster axes.
      # e.g. for the grid (a, b, c, d, wg)
      # squashed = (a, b)  Mapped to Dimension.z (2)
      # unsquashed = (c, d)  Mapped to Dimension.y (1) and Dimension.x (0)
      idx = unsquashed_names.index(axis_name)
      return arith_dialect.index_cast(
          ir.IntegerType.get_signless(32),
          _block_id(ctx, gpu_dialect.Dimension(idx)),
      )
    else:
      assert axis_name in squashed_names
      # All squashed dimensions are mapped to Dimension.z.
      axis = squashed_names.index(axis_name)
      return _unravel_program_id(
          _block_id(ctx, gpu_dialect.Dimension.z), axis, squashed_dims
      )
  else:
    assert axis_name in axis_names.grid
    idx = axis_names.grid.index(axis_name)
    return arith_dialect.index_cast(
        ir.IntegerType.get_signless(32),
        _block_id(ctx, gpu_dialect.Dimension(idx)),
    )

@register_lowering_rule(primitives.debug_print_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(primitives.debug_print_p, mgpu.LoweringSemantics.Lane,
                        gpu_core.PrimitiveSemantics.Warp)
def _debug_print_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    fmt,
    has_placeholders: bool,
):
  del has_placeholders  # Unused.
  primitives.check_debug_print_format(fmt, *args)
  scope = mgpu.ThreadSubset.WARPGROUP
  if ctx.module_ctx.primitive_semantics == gpu_core.PrimitiveSemantics.Warp:
    scope = mgpu.ThreadSubset.WARP
  if not any(aval.shape for aval in ctx.avals_in):
    mgpu.debug_print(
        fmt,
        *(
            _ensure_ir_value(arg, aval.dtype)
            for arg, aval in zip(args, ctx.avals_in)
        ),
        scope=scope
    )
  elif len(ctx.avals_in) == 1:
    [arg] = args
    arg.debug_print(fmt)
  else:
    raise NotImplementedError(
        "debug_print only supports printing of scalar values, or a single array"
        " value when using the Mosaic GPU backend."
    )

  return ()

@register_lowering_rule(primitives.debug_print_p, mgpu.LoweringSemantics.Warpgroup)
def _debug_print_lowering_rule_wg(
    ctx: LoweringRuleContext,
    *args,
    fmt,
    has_placeholders: bool,
):
  del ctx, has_placeholders  # Unused.
  if args:
    raise NotImplementedError("debug_print only supports string messages in warpgroup semantics")
  mgpu.debug_print(fmt)
  return ()


@register_lowering_rule(primitives.run_scoped_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(primitives.run_scoped_p, mgpu.LoweringSemantics.Warpgroup)
def _run_scoped_lowering_rule(
    ctx: LoweringRuleContext, *consts, jaxpr: jax_core.Jaxpr, collective_axes
):
  input_refs = []
  should_discharge = []
  wg_axis = ctx.module_ctx.axis_names.wg
  is_multithreaded = wg_axis is not None
  is_thread_collective = is_multithreaded and collective_axes == (wg_axis,)
  # Make sure everyone has exited previous scoped allocations. Note that we
  # don't synchronize when we exit the allocation, but only when we might want
  # to reuse its memory again.
  if is_multithreaded and is_thread_collective:
    gpu_dialect.barrier()
  with contextlib.ExitStack() as alloc_stack:
    for v in jaxpr.invars:
      aval = cast(ShapedAbstractValue, v.aval)
      if isinstance(aval, gpu_core.WGMMAAbstractAccumulatorRef):
        if collective_axes:
          raise ValueError(
              "WGMMA accumulators can only be allocated non-collectively. Hint:"
              " remove collective_axes from run_scoped. If other allocations"
              " are performed as well, split the run_scoped into two."
          )
        dtype = mlir.dtype_to_ir_type(aval.dtype)
        if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
          input_refs.append(mgpu.WGMMAAccumulator.zero(*aval.shape, dtype))
        else:
          zero = arith_dialect.constant(dtype, ir.FloatAttr.get(dtype, 0.0))
          acc = vector_dialect.broadcast(
              ir.VectorType.get(aval.shape, dtype), zero
          )
          acc = mgpu.dialect.optimization_barrier([acc])
          nvvm_dialect.wgmma_fence_aligned()
          input_refs.append(acc)
        should_discharge.append(True)
        continue
      # All other allocations must be made collectively across all threads.
      if is_multithreaded and not is_thread_collective:
        raise NotImplementedError(
            "Only thread-collective allocations are supported in multithreaded"
            " kernels. Hint: add"
            f" collective_axes={ctx.module_ctx.axis_names.wg} to your"
            " run_scoped if you intend all threads to share the same"
            f" allocation (currently collective_axes={collective_axes})."
        )
      if isinstance(aval.dtype, gpu_core.BarrierType):
        multiplier = (1 if aval.dtype.orders_tensor_core else
                      ctx.estimator_ctx.arrival_multiplier)
        barrier_ref = alloc_stack.enter_context(
            ctx.module_ctx.reserve_barrier(
                mgpu.Barrier(
                    aval.dtype.num_arrivals * multiplier,
                    *aval.shape,
                )
            )
        )
        input_refs.append(barrier_ref)
        should_discharge.append(False)
        continue
      if isinstance(aval.dtype, gpu_core.ClusterBarrierType):
        collective_dims = jax.tree.map(
            lambda axis: _resolve_cluster_axis(ctx.module_ctx.axis_names, axis),
            aval.dtype.collective_axes,
        )
        barrier_ref = alloc_stack.enter_context(
            ctx.module_ctx.reserve_barrier(
                mgpu.ClusterBarrier(collective_dims, *aval.shape)
            )
        )
        input_refs.append(barrier_ref)
        should_discharge.append(False)
        continue

      if not isinstance(aval, state_types.AbstractRef):
        raise ValueError(f"Can't convert to ref: {aval}")
      if aval.memory_space == gpu_core.SMEM:
        [input_ref] = alloc_stack.enter_context(
            ctx.module_ctx.scratch_view(
                [jax.ShapeDtypeStruct(shape=aval.shape, dtype=aval.dtype)]
            )
        )
        input_refs.append(input_ref)
        should_discharge.append(False)
      elif aval.memory_space == gpu_core.TMEM:
        input_ref = alloc_stack.enter_context(
            ctx.module_ctx.alloc_tmem(
                jax.ShapeDtypeStruct(shape=aval.shape, dtype=aval.dtype),
                layout=aval.layout,
                collective=aval.collective,
            )
        )
        input_refs.append(input_ref)
        should_discharge.append(False)
      elif aval.memory_space == gpu_core.GMEM and jnp.issubdtype(aval.dtype, pallas_core.semaphore):
        input_ref = alloc_stack.enter_context(
            ctx.module_ctx.reserve_semaphores(aval.shape)
        )
        input_refs.append(input_ref)
        should_discharge.append(False)

    if any(should_discharge):
      # We convert consts to args, because we only have ir.Values and
      # not JAX values during lowering. discharge_state() produces JAX
      # valiues for the arguments but expects them to be provided for the
      # consts. We also don't want to wrap the values in refs.
      no_const_jaxpr = pe.convert_constvars_jaxpr(jaxpr)
      should_discharge = [False] * len(consts) + should_discharge
      discharged_jaxpr, _ = discharge.discharge_state(no_const_jaxpr, (), should_discharge=should_discharge)
      new_input_vals = (*consts, *input_refs)
      outs = lower_jaxpr_to_mosaic_gpu(
          ctx.module_ctx,
          ctx.launch_ctx,
          discharged_jaxpr,
          new_input_vals,
          (),
      )
      # Discharge appends to the output the refs that got discharged.
      outs = outs[:-sum(should_discharge)]
    else:
      outs = lower_jaxpr_to_mosaic_gpu(
          ctx.module_ctx,
          ctx.launch_ctx,
          jaxpr,
          input_refs,
          consts,
      )

  assert len(outs) == len(jaxpr.outvars), (jaxpr, outs)
  return outs


@register_lowering_rule(discharge.run_state_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(discharge.run_state_p, mgpu.LoweringSemantics.Warpgroup)
def _run_state_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr: jax_core.Jaxpr,
    which_linear: tuple[bool, ...],
    is_initialized: tuple[bool, ...],
):
  del which_linear
  # TODO(apaszke): This should be unified with run_scoped.
  if not all(is_initialized):
    raise NotImplementedError("Uninitialized Refs are not supported in lowering of run_state.")

  should_discharge = []
  new_input_vals = []
  for arg, v, out_aval in zip(args, jaxpr.invars, ctx.avals_out):
    aval = v.aval
    if isinstance(aval, gpu_core.WGMMAAbstractAccumulatorRef):
      if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Warpgroup:
        arg = mgpu.dialect.optimization_barrier([arg])
        nvvm_dialect.wgmma_fence_aligned()
        new_input_vals.append(arg)
      else:
        new_input_vals.append(mgpu.WGMMAAccumulator.from_registers(arg))
      should_discharge.append(True)
      assert isinstance(out_aval, jax_core.ShapedArray)
    else:
      new_input_vals.append(arg)
      should_discharge.append(not isinstance(out_aval, state_types.AbstractRef))
  if not any(should_discharge):
    raise NotImplementedError(
        "Expected at least one accumulator to in run_state."
    )

  discharged_jaxpr, new_consts = discharge.discharge_state(
      jaxpr, (), should_discharge=should_discharge
  )
  assert not new_consts
  outs = lower_jaxpr_to_mosaic_gpu(
      ctx.module_ctx, ctx.launch_ctx, discharged_jaxpr, new_input_vals, ()
  )
  # Await the accumulators and extract their final values.
  nvvm_dialect.wgmma_wait_group_sync_aligned(0)
  outs = [
      out.value if isinstance(out, mgpu.WGMMAAccumulator) else out
      for out in outs
  ]
  # Blend the discharge results with refs we closed over. I don't fully
  # understand the reasons behind this calling convention, but sharadmv@ has
  # assured me that this is ok.
  outs_it = iter(outs)
  return [next(outs_it) if d else a for d, a in zip(should_discharge, args)]


def _lower_jaxpr_to_for_loop(
    ctx: LoweringRuleContext,
    jaxpr: jax_core.Jaxpr,
    start: ir.Value,
    length: int | ir.Value,
    consts,
    *args,
    has_loop_index: bool,
    unroll: int | None = None,
):
  _consts_avals, arg_avals = util.split_list(ctx.avals_in, [len(consts)])
  arg_avals = arg_avals[has_loop_index:]
  out_avals = []
  if arg_avals:
    out_avals = ctx.avals_out[-len(arg_avals):]

  is_acc = [isinstance(v, mgpu.WGMMAAccumulator) for v in args]
  def as_values(vals, avals):
    if is_acc != [isinstance(v, mgpu.WGMMAAccumulator) for v in vals]:
      raise ValueError("Unexpected loop carry w.r.t. accumulators.")

    _ensure = (
        _ensure_fa
        if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane
        else _ensure_ir_value
    )
    return [
        v if a else _ensure(v, av.dtype)
        for a, v, av in zip(is_acc, vals, avals)
    ]

  def loop(base_loop_index, body_args):
    outs = body_args
    if unroll is not None:
      base_loop_index = arith_dialect.muli(
          base_loop_index, _ir_constant(unroll, start.type)
      )
    base_loop_index = arith_dialect.addi(base_loop_index, start)
    for step in range(unroll or 1):
      if has_loop_index:
        loop_index = arith_dialect.addi(
            base_loop_index, _ir_constant(step, start.type)
        )
        jaxpr_args = [*consts, loop_index, *outs]
      else:
        jaxpr_args = [*consts, *outs]
      outs = lower_jaxpr_to_mosaic_gpu(
          ctx.module_ctx, ctx.launch_ctx, jaxpr, jaxpr_args
      )
    return as_values(outs, out_avals)

  if unroll is not None:
    if not isinstance(length, int):
      raise NotImplementedError(
          "``length`` must be an integer when ``unroll` is specified, got"
          f" {length}"
      )
    if length % unroll:
      # TODO(slebedev): Emit an epilogue taking care of the remaining steps.
      raise NotImplementedError(
          f"``unroll`` must divide ``length``, got {unroll=} and {length=}"
      )
    if unroll == length:
      # Special-case: the loop is fully unrolled.
      return loop(_ir_constant(0, start.type), as_values(args, arg_avals))
    return mgpu.fori(
        _ir_constant(length // unroll, start.type), as_values(args, arg_avals)
    )(loop).results
  else:
    if not isinstance(length, ir.Value):
      length = _ir_constant(length, start.type)
    return mgpu.fori(length, as_values(args, arg_avals))(loop).results


@register_lowering_rule(lax.scan_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.scan_p, mgpu.LoweringSemantics.Warpgroup)
@register_lowering_rule(lax.scan_p, mgpu.LoweringSemantics.Lane,
                        gpu_core.PrimitiveSemantics.Warp)
def _scan_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr: jax_core.ClosedJaxpr,
    linear: tuple[bool, ...],
    length: int,
    reverse: bool,
    unroll: bool | int,
    num_consts: int,
    num_carry: int,
    _split_transpose: bool,
):
  # Can only handle fori_loop-like scans.
  if (num_extensive := len(args) - num_consts - num_carry) or reverse:
    raise NotImplementedError
  del linear, num_extensive, reverse

  jaxpr, jaxpr_consts = jaxpr.jaxpr, jaxpr.consts
  if jaxpr_consts:
    raise NotImplementedError
  del jaxpr_consts

  jaxpr, has_loop_index = pallas_utils.pattern_match_scan_to_fori_loop(
      jaxpr, num_consts, num_carry
  )
  consts, args = util.split_list(args, [num_consts])
  _consts_avals, arg_avals = util.split_list(ctx.avals_in, [num_consts])
  if has_loop_index:
    start, *args = args
    index_aval, *_ = arg_avals
    start: ir.Value = _ensure_ir_value(start, index_aval.dtype)
  else:
    start = _i32_constant(0)

  for_out = _lower_jaxpr_to_for_loop(
      ctx,
      jaxpr,
      start,
      length,
      consts,
      *args,
      has_loop_index=has_loop_index,
      unroll=unroll,
  )
  if has_loop_index:
    # Need to return the final loop index value if the outer scan expects
    # it as an output.
    loop_index = arith_dialect.addi(start, _ir_constant(length, start.type))
    return [loop_index, *for_out]
  return for_out


def _lower_while_via_fori(
    ctx: LoweringRuleContext,
    *args,
    fori_jaxpr,
    cond_nconsts,
    body_nconsts,
):
  assert not fori_jaxpr.constvars
  # The pattern matcher looks for conditions with no constants.
  assert cond_nconsts == 0

  # Reflect the changes of the pattern matcher to the context.
  lb_aval, ub_aval, *_ = ctx.avals_in[cond_nconsts + body_nconsts:]
  ctx = ctx.replace(
      avals_in=(
          *ctx.avals_in[cond_nconsts:body_nconsts],
          ctx.avals_in[body_nconsts],  # the index
          *ctx.avals_in[body_nconsts + 2 :],
      ),
      avals_out=tuple(ctx.avals_out[2:]),
  )
  _, consts, (lb, ub, *args) = util.split_list(
      args, [cond_nconsts, body_nconsts]
  )
  lb = _ensure_ir_value(lb, lb_aval.dtype)
  ub = _ensure_ir_value(ub, ub_aval.dtype)
  for_out = _lower_jaxpr_to_for_loop(
      ctx,
      fori_jaxpr,
      lb,
      arith_dialect.subi(ub, lb),
      consts,
      *args,
      has_loop_index=True,
  )
  return ub, ub, *for_out


@register_lowering_rule(lax.while_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.while_p, *gpu_core.LANExWARP_SEMANTICS)
@register_lowering_rule(lax.while_p, mgpu.LoweringSemantics.Warpgroup)
def _while_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    cond_jaxpr,
    body_jaxpr,
    cond_nconsts,
    body_nconsts,
):
  # First try to lower via a simpler fori loop, which may optimize better.
  fori_jaxpr, _ = pallas_utils.pattern_match_while_to_fori_loop(
      cond_jaxpr, cond_nconsts, body_jaxpr, body_nconsts
  )
  if fori_jaxpr is not None:
    return _lower_while_via_fori(
        ctx,
        *args,
        fori_jaxpr=fori_jaxpr,
        cond_nconsts=cond_nconsts,
        body_nconsts=body_nconsts,
    )

  _is_acc = lambda x: isinstance(x, mgpu.WGMMAAccumulator)
  _ensure = _ensure_ir_value
  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
    _ensure = lambda v, aval: v if _is_acc(v) else _ensure_fa(v, aval.dtype)

  # If we fail conversion to fori, fallback to an ordinary while loop.
  cond_consts, body_consts, carry = util.split_list(
      args, [cond_nconsts, body_nconsts]
  )
  _cond_avals, _body_avals, carry_avals = util.split_list(
      ctx.avals_in, [cond_nconsts, body_nconsts]
  )
  carry = [*map(_ensure, carry, carry_avals)]
  # Flatten the carry to get a concatenated list of registers from each FA.
  # Note that the treedef is also used below to unflatten the body results.
  flat_carry, carry_treedef = jax.tree.flatten(carry)
  flat_carry_types = [a.type for a in flat_carry]
  while_op = scf_dialect.WhileOp(flat_carry_types, flat_carry)

  before_block = while_op.before.blocks.append(*flat_carry_types)
  with ir.InsertionPoint.at_block_begin(before_block):
    cond_args = [*cond_consts, *carry_treedef.unflatten(before_block.arguments)]
    [cond] = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, cond_jaxpr.jaxpr, cond_args
    )
    scf_dialect.condition(
        _ensure_ir_value(cond, *cond_jaxpr.out_avals), before_block.arguments
    )

  after_block = while_op.after.blocks.append(*flat_carry_types)
  with ir.InsertionPoint.at_block_begin(after_block):
    body_args = [*body_consts, *carry_treedef.unflatten(after_block.arguments)]
    loop_out = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, body_jaxpr.jaxpr, body_args
    )
    loop_out = [*map(_ensure, loop_out, carry_avals)]
    if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Lane:
      for idx, (carry_fa, out_fa) in enumerate(zip(carry, loop_out)):
        if _is_acc(carry_fa) != _is_acc(out_fa):
          raise ValueError(
              f"The loop body output has unexpected accumulator type:"
              f" output[{idx}] is {out_fa}, when it should be {carry_fa}."
          )

        if not _is_acc(out_fa) and carry_fa.layout != out_fa.layout:
          raise ValueError(
              f"The loop body output has unexpected layout: output[{idx}] has"
              f" layout {out_fa.layout}, when it should be {carry_fa.layout}."
          )
    scf_dialect.yield_(
        carry_treedef.flatten_up_to(loop_out) if loop_out else []
    )
  return carry_treedef.unflatten(list(while_op.results))


@register_lowering_rule(lax.cond_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(lax.cond_p,
  mgpu.LoweringSemantics.Lane, gpu_core.PrimitiveSemantics.Warp)
@register_lowering_rule(lax.cond_p, mgpu.LoweringSemantics.Warpgroup)
def _cond_lowering_rule(ctx: LoweringRuleContext, index, *args, branches,
                        **params):
  if params:
    raise NotImplementedError("platform_dependent cond")
  index_aval, *_arg_avals = ctx.avals_in

  def _yielded_values(outs, avals):
    ret = []
    for out, aval in zip(outs, avals):
      if isinstance(out, (mgpu.WGMMAAccumulator, mgpu.FragmentedArray)):
        ret.append(out)
      else:
        ret.append(_ensure_ir_value(out, aval.dtype))
    return ret

  # We need to know the result types ahead of time to construct the switch
  # operation. Below we lower the first branch in a throw-away module to
  # extract them.
  with ir.InsertionPoint(ir.Module.create().body):
    outs = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, branches[0].jaxpr, args
    )
    yielded_types = [
        v.type for v in jax.tree.leaves(_yielded_values(outs, ctx.avals_out))
    ]
    del outs

  switch_op = scf_dialect.IndexSwitchOp(
      yielded_types,
      _as_index(_ensure_ir_value(index, index_aval.dtype)),
      ir.DenseI64ArrayAttr.get(range(len(branches) - 1)),
      num_caseRegions=len(branches) - 1,
  )

  # ``RegionSequence`` in MLIR does not support slicing, so the
  # auto-generated Python bindings for ``caseRegions`` fail at runtime!
  # We convert it to a list to work around that.
  regions = list(switch_op.regions)
  # Move the default region to the back.
  regions = regions[1:] + regions[:1]
  treedef = None
  for branch, region in zip(branches, regions):
    with ir.InsertionPoint(region.blocks.append()):
      outs = lower_jaxpr_to_mosaic_gpu(
          ctx.module_ctx, ctx.launch_ctx, branch.jaxpr, args, consts=branch.consts
      )

      yielded_leaves, yielded_treedef = jax.tree.flatten(_yielded_values(outs, ctx.avals_out))
      if treedef is None:
        treedef = yielded_treedef
      else:
        assert treedef == yielded_treedef

      scf_dialect.yield_(yielded_leaves)

  assert treedef is not None
  return treedef.unflatten(list(switch_op.results))


@register_lowering_rule(lax.bitcast_convert_type_p, mgpu.LoweringSemantics.Lane)
@register_lowering_rule(
    lax.bitcast_convert_type_p, mgpu.LoweringSemantics.Warpgroup
)
def _bitcast_convert_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype
):
  [x_aval] = ctx.avals_in
  src_elem_type = mgpu_utils.dtype_to_ir_type(x_aval.dtype)
  dst_elem_type = mgpu_utils.dtype_to_ir_type(new_dtype)
  assert isinstance(src_elem_type, (ir.IntegerType, ir.FloatType))
  assert isinstance(dst_elem_type, (ir.IntegerType, ir.FloatType))
  if src_elem_type.width != dst_elem_type.width:
    raise NotImplementedError(
        f"Cannot bitcast from {x_aval.dtype} to {new_dtype} because they"
        " have different widths"
    )

  if ctx.module_ctx.lowering_semantics == mgpu.LoweringSemantics.Warpgroup:
    x = _ensure_ir_value(x, x_aval.dtype)
    return arith_dialect.bitcast(
        ir.VectorType.get(x_aval.shape, dst_elem_type), x
    )

  x = _ensure_fa(x, x_aval.dtype)
  if ir.IntegerType.isinstance(dst_elem_type):
    output_is_signed = mgpu_utils.is_signed(new_dtype)
  else:
    output_is_signed = None
  return mgpu.FragmentedArray.bitcast(
      x, dst_elem_type, output_is_signed=output_is_signed
  )


@register_lowering_rule(lax.optimization_barrier_p, mgpu.LoweringSemantics.Lane)
def _optimization_barrier_lowering(ctx: LoweringRuleContext, *args):
  result = mgpu.optimization_barrier(
      *(_ensure_fa(arg, aval.dtype) for arg, aval in zip(args, ctx.avals_in))
  )
  return (result,) if len(ctx.avals_in) == 1 else result


@register_lowering_rule(
    lax.optimization_barrier_p, mgpu.LoweringSemantics.Warpgroup
)
def _optimization_barrier_lowering_wg(ctx: LoweringRuleContext, *args):
  result = mgpu.dialect.optimization_barrier([
      _ensure_ir_value(arg, aval.dtype) for arg, aval in zip(args, ctx.avals_in)
  ])
  return (result,) if len(ctx.avals_in) == 1 else result


@register_lowering_rule(pallas_core.core_map_p, mgpu.LoweringSemantics.Lane)
def _core_map_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr,
    mesh,
    **_,
):
  if isinstance(mesh, gpu_core.WarpMesh):
    # A core_map over a WarpMesh represents a fork/join over individual
    # warps in a warpgroup.
    if (ctx.module_ctx.warp_axis_name or
        ctx.module_ctx.primitive_semantics == gpu_core.PrimitiveSemantics.Warp):
      raise LoweringError(
          "Cannot nest core_maps. Already under core_map with warp_axis_name "
          f"{ctx.module_ctx.warp_axis_name}.")
    module_ctx = dataclasses.replace(
        ctx.module_ctx,
        warp_axis_name=mesh.axis_name,
        primitive_semantics=gpu_core.PrimitiveSemantics.Warp,
    )
    for aval_in in ctx.avals_in:
      if isinstance(aval_in, jax_core.ShapedArray) and aval_in.shape:
        raise LoweringError(
          "Can only close over scalars and Refs when using core_map with "
          f"WarpMesh. Found array of shape {aval_in}."
        )
    # We allow the warps to schedule async copies without synchronizing with
    # other warps, so we need to add a barrier here to make sure all reads and
    # writes have completed.
    if ctx.module_ctx.auto_barriers:
      mgpu.warpgroup_barrier()
    _ = lower_jaxpr_to_mosaic_gpu(
        module_ctx,
        ctx.launch_ctx,
        jaxpr,
        args=(),
        consts=args,
    )
    if ctx.module_ctx.auto_barriers:
      # We need to ensure that any effects produced by one warp
      # (e.g. async copies) are observable by all other warps.
      mgpu.warpgroup_barrier()
    return []
  raise ValueError(f"Unsupported mesh: {mesh}")


def _bcast(
    x: Any,
    y: Any,
    x_aval: ShapedAbstractValue,
    y_aval: ShapedAbstractValue,
    out_aval: ShapedAbstractValue,
) -> tuple[mgpu.FragmentedArray, mgpu.FragmentedArray]:
  if not isinstance(x, mgpu.FragmentedArray):
    x_dtype = x_aval.dtype
    if x_aval.weak_type:
      x_dtype = y_aval.dtype
    x = _ensure_fa(x, x_dtype)
  if not isinstance(y, mgpu.FragmentedArray):
    y_dtype = y_aval.dtype
    if y_aval.weak_type:
      y_dtype = x_aval.dtype
    y = _ensure_fa(y, y_dtype)
  if x_aval.shape != out_aval.shape:
    x = x.broadcast(out_aval.shape)
  if y_aval.shape != out_aval.shape:
    y = y.broadcast(out_aval.shape)
  return x, y


def _ensure_fa(x: object, dtype: jnp.dtype) -> mgpu.FragmentedArray:
  if isinstance(x, mgpu.FragmentedArray):
    assert x.mlir_dtype == mgpu_utils.dtype_to_ir_type(dtype)
    return x
  return mgpu.FragmentedArray.splat(
      _ensure_ir_value(x, dtype), (), is_signed=mgpu_utils.is_signed(dtype)
  )


def _bcast_wg(
    x: Any,
    y: Any,
    x_aval: ShapedAbstractValue,
    y_aval: ShapedAbstractValue,
    out_aval: ShapedAbstractValue,
) -> tuple[ir.Value, ir.Value]:
  """Ensures that ``x`` and ``y`` have the expected shapes and dtypes.

  More specifically, the inputs are converted to vectors of the same dtype
  as ``x_aval`` and ``y_aval``, and broadcasted to the output shape
  if necessary.
  """
  if not out_aval.shape:
    return _ensure_ir_value(x, x_aval.dtype), _ensure_ir_value(y, y_aval.dtype)
  x_dtype = x_aval.dtype
  if not isinstance(x, ir.Value):
    if x_aval.weak_type:
      x_dtype = y_aval.dtype
    x = _ensure_ir_value(x, x_dtype)
  y_dtype = y_aval.dtype
  if not isinstance(y, ir.Value):
    if y_aval.weak_type:
      y_dtype = x_aval.dtype
    y = _ensure_ir_value(y, y_dtype)
  if not ir.VectorType.isinstance(x.type):
    assert not x_aval.shape
    x = vector_dialect.broadcast(
        ir.VectorType.get(out_aval.shape, mgpu_utils.dtype_to_ir_type(x_dtype)),
        x,
    )
  elif x_aval.shape != out_aval.shape:
    raise NotImplementedError("Unsupported broadcast")
  if not ir.VectorType.isinstance(y.type):
    assert not y_aval.shape
    y = vector_dialect.broadcast(
        ir.VectorType.get(out_aval.shape, mgpu_utils.dtype_to_ir_type(y_dtype)),
        y,
    )
  elif y_aval.shape != out_aval.shape:
    raise NotImplementedError("Unsupported broadcast")
  return x, y


def _ensure_ir_value(x: Any, dtype: jnp.dtype) -> ir.Value:
  if isinstance(x, ir.Value):
    mlir_dtype = mgpu_utils.dtype_to_ir_type(dtype)
    if ir.VectorType.isinstance(x.type):
      assert ir.VectorType(x.type).element_type == mlir_dtype
    else:
      assert x.type == mlir_dtype, (x.type, mlir_dtype)
    return x
  elif isinstance(x, mgpu.FragmentedArray):
    assert x.mlir_dtype == mgpu_utils.dtype_to_ir_type(dtype)
    if isinstance(x.layout, mgpu.WGSplatFragLayout):
      return x.registers.item()
    raise NotImplementedError(f"Unsupported layout: {x.layout}")
  return _ir_constant(x, mgpu_utils.dtype_to_ir_type(dtype))


def _ir_constant(v: object, t: ir.Type) -> ir.Value:
  if isinstance(v, (np.number, np.ndarray, int, float)):
    if isinstance(t, (ir.IntegerType, ir.IndexType)):
      v = int(v)
    else:
      assert isinstance(t, ir.FloatType)
      v = float(v)
    return arith_dialect.constant(t, v)
  raise NotImplementedError(f"Unsupported constant: {v!r}")


def _i32_constant(v: int) -> ir.Value:
  if v < jnp.iinfo(jnp.int32).min or v > jnp.iinfo(jnp.int32).max:
    raise ValueError(f"Integer constant out of range for i32: {v}")
  return arith_dialect.constant(ir.IntegerType.get_signless(32), v)


def _i64_constant(v: int) -> ir.Value:
  if v < jnp.iinfo(jnp.int64).min or v > jnp.iinfo(jnp.int64).max:
    raise ValueError(f"Integer constant out of range for i64: {v}")
  return arith_dialect.constant(ir.IntegerType.get_signless(64), v)


def _as_index(v: object) -> ir.Value:
  match v:
    case int():
      return arith_dialect.constant(ir.IndexType.get(), v)
    case ir.Value() if ir.IndexType.isinstance(v.type):
      return v
    case ir.Value() if ir.IntegerType.isinstance(v.type):
      return arith_dialect.index_cast(ir.IndexType.get(), v)
    case mgpu.FragmentedArray(layout=mgpu.WGSplatFragLayout()):
      return _as_index(v.registers.item())
    case _:
      raise ValueError(f"Unsupported index: {v} of type {type(v)}")


def merge_indexers(
    indexers: Sequence[indexing.NDIndexer]) -> indexing.NDIndexer:
  """Merges multiple indexers into a single indexer.

  This function computes a new indexer such that applying the
  new indexer produces the same result as applying the sequence
  of input indexers in order from first-to-last.
  """
  if len(indexers) == 0:
    raise ValueError("Cannot merge empty list of indexers")
  if len(indexers) == 1:
    return indexers[0]
  root_shape = indexers[0].shape
  current_indices = [indexing.Slice(0, size, 1) for size in root_shape]
  removed_dimensions = set()
  for indexer in indexers:
    if indexer.int_indexer_shape:
      raise NotImplementedError()

    def _ensure_idx_fa(x: Any) -> mgpu.FragmentedArray:
      i32 = ir.IntegerType.get_signless(32)
      if isinstance(x, ir.Value):
        # TODO(cperivol): We assume all indices are signed. We should
        # look at the JAX avals to see if the integers are signed or
        # not to figure out is_signed.
        is_signed = False if ir.IntegerType.isinstance(x.type) else None
        return mgpu.FragmentedArray.splat(x, (), is_signed=is_signed).astype(
            i32, is_signed=False
        )
      if isinstance(x, mgpu.FragmentedArray):
        return x.astype(i32, is_signed=False)
      if isinstance(x, int):
        return mgpu.FragmentedArray.splat(mgpu.c(x, i32), (), is_signed=False)
      raise NotImplementedError(x)

    num_skipped = 0
    for i in range(len(current_indices)):
      # Integer indexers remove dimensions which should be
      # skipped by following indexers.
      if i in removed_dimensions:
        num_skipped += 1
        continue
      dim_indexer = indexer.indices[i - num_skipped]
      current_index = current_indices[i]
      assert isinstance(current_index, indexing.Slice)

      current_start_index = _ensure_idx_fa(current_index.start)
      if isinstance(dim_indexer, indexing.Slice):
        if dim_indexer.stride != 1:
          raise NotImplementedError("Non-unit strides not implemented.")
        current_indices[i] = indexing.Slice(
            current_start_index + _ensure_idx_fa(dim_indexer.start),
            dim_indexer.size,
            1,
        )
      else:
        current_indices[i] = current_start_index + _ensure_idx_fa(dim_indexer)
        removed_dimensions.add(i)
  return indexing.NDIndexer(
      indices=tuple(current_indices),
      shape=root_shape,
      int_indexer_shape=(),
  )


@register_lowering_rule(primitives.semaphore_read_p, mgpu.LoweringSemantics.Lane)
def _semaphore_read_lowering_rule(ctx: LoweringRuleContext, *args, args_tree):
  sem, transforms = tree_util.tree_unflatten(args_tree, args)
  sem, transforms = _handle_transforms(ctx, sem, transforms)
  if transforms:
    raise NotImplementedError(f"Unhandled transforms for semaphore_read: {transforms}")
  sem_ptr = mgpu.utils.memref_ptr(sem)
  i32_ty = ir.IntegerType.get_signless(32)
  return llvm_dialect.inline_asm(
    i32_ty, [sem_ptr], "ld.acquire.sys.u32 $0,[$1];", "=r,l", has_side_effects=True,
  )


@register_lowering_rule(primitives.semaphore_signal_p, mgpu.LoweringSemantics.Lane)
def _semaphore_signal_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    args_tree,
    device_id_type,
):
  i32 = ir.IntegerType.get_signless(32)
  sem, transforms, value, device_id, core_index = tree_util.tree_unflatten(
      args_tree, args
  )
  if core_index is not None:
    raise NotImplementedError(
        "Mosaic GPU backend does not support the concept of cores, but"
        " core_index is specified"
    )
  sem, transforms = _handle_transforms(ctx, sem, transforms)
  if transforms:
    raise NotImplementedError(f"Unhandled transforms for semaphore_signal: {transforms}")
  sem_ptr = mgpu.utils.memref_ptr(sem)
  if device_id is not None:
    device_id, other_axes = primitives.device_id_to_logical(
        ctx.module_ctx.mesh_info,
        device_id,
        device_id_type,
        lambda name: _axis_index_rule(ctx, axis_name=name),
    )
    if other_axes:
      raise NotImplementedError(
          f"Only JAX mesh axes can be used in device_id, but found {other_axes}"
      )
    sem_ptr = ctx.launch_ctx.to_remote(
        sem_ptr, _ensure_ir_value(device_id, jnp.int32)
    )
  # TODO(apaszke): Narrow the scope from .sys to .gpu when the semaphore is local.
  val = _ir_constant(value, i32)
  # We only signal the semaphore from a single lane, which does not guarantee
  # anything about the state of the other three warps in the warpgroup (they
  # might still be e.g. reading memory that someone will overwrite once they
  # receive a signal).
  if ctx.module_ctx.auto_barriers:
    mgpu.utils.warpgroup_barrier()
  mgpu_utils.SemaphoreRef(sem_ptr).signal(
      val, predicate=ctx.module_ctx.single_wg_lane_predicate
  )
  return ()


@register_lowering_rule(primitives.semaphore_wait_p, mgpu.LoweringSemantics.Lane)
def _semaphore_wait_lowering_rule(ctx: LoweringRuleContext, *args, args_tree):
  sem, transforms, value = tree_util.tree_unflatten(args_tree, args)
  sem, transforms = _handle_transforms(ctx, sem, transforms)
  if transforms:
    raise NotImplementedError(
        f"Unhandled transforms for semaphore_wait: {transforms}"
    )
  i32 = ir.IntegerType.get_signless(32)
  val = _ir_constant(value, i32)
  mgpu_utils.SemaphoreRef(mgpu.utils.memref_ptr(sem)).wait(val)
  return ()


@register_lowering_rule(checkify.check_p, mgpu.LoweringSemantics.Lane)
def _check_lowering_rule(ctx: LoweringRuleContext, *err_args, err_tree, debug):
  del ctx  # Unused.

  if not debug:
    raise NotImplementedError(
        "Non-debug checks are not supported by the Mosaic GPU backend."
        " Functionalize them via `jax.experimental.checkify`."
    )
  if not pallas_helpers.debug_checks_enabled():
    return []

  error = jax.tree.unflatten(err_tree, err_args)
  [pred] = error._pred.values()
  [exception_tree] = error._metadata.values()
  [payload] = error._payload.values()
  exception = jax.tree.unflatten(exception_tree, payload)
  assert isinstance(exception, checkify.FailedCheckError)

  # check_p has an inverted predicate compared to assert, so we need to compute
  # ``not pred`` here.
  minus_one = _ir_constant(-1, mgpu_utils.dtype_to_ir_type(jnp.bool))
  not_pred = arith_dialect.xori(pred.registers.item(), minus_one)
  cf_dialect.assert_(not_pred, exception.fmt_string)
  return []

@register_lowering_rule(gpu_core.layout_cast_p, mgpu.LoweringSemantics.Lane)
def _layout_cast_lowering(ctx: LoweringRuleContext, x, *, new_layout):
  del ctx  # Unused.
  return x.to_layout(new_layout.to_mgpu())


@register_lowering_rule(gpu_core.layout_cast_p, mgpu.LoweringSemantics.Warpgroup)
def _layout_cast_lowering_wg(
    ctx: LoweringRuleContext, x, *, new_layout
):
  del ctx  # Unused.
  return mgpu.dialect.layout_cast(x, mgpu.to_layout_attr(new_layout.to_mgpu()))


@register_lowering_rule(lax.iota_p, mgpu.LoweringSemantics.Lane)
def _iota_lowering(
    ctx: LoweringRuleContext, dtype, shape, dimension, sharding
):
  del sharding  # Unused.
  if ctx.out_layout_hint is None:
    raise RuntimeError(
        "Failed to infer the output layout of the iota. Please apply"
        " plgpu.layout_cast to its output right after its creation."
    )
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
      ctx.out_layout_hint,
      is_signed=is_signed,
  ).foreach(
      lambda _, idx: cast(idx[dimension]),
      create_array=True,
      is_signed=is_signed,
  )
