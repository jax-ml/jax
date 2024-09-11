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

from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import Any, cast

import jax
from jax._src import core as jax_core
from jax._src import pjit
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.lib.mlir.dialects import gpu as gpu_dialect
from jax._src.lib.mlir.dialects import memref as memref_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.state import primitives as sp
from jax.experimental.mosaic import gpu as mosaic_gpu
from jax.experimental.mosaic.gpu import dsl as mgpu
import jax.numpy as jnp
import numpy as np


# TODO(slebedev): Enable type checking.
# mypy: ignore-errors
# pytype: skip-file

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

partial = functools.partial

_smem_estimators = {}


def _regiter_smem_estimator(primitive: jax_core.Primitive):
  def deco(fn):
    _smem_estimators[primitive] = fn
    return fn

  return deco


def _estimate_smem_scratch_bytes(jaxpr: jax_core.Jaxpr) -> int:
  """Estimates the amount of SMEM scratch bytes required by the kernel."""
  max_used = 0
  for eqn in jaxpr.eqns:
    # TODO(slebedev): Add support for other primitives, notably control flow.
    rule = _smem_estimators.get(eqn.primitive)
    if rule is None:
      # Assume that unsupported primitives are neutral wrt SMEM usage.
      continue
    max_used = max(
        max_used, rule(*(invar.aval for invar in eqn.invars), **eqn.params)
    )
  return max_used


@_regiter_smem_estimator(primitives.run_scoped_p)
def _run_scoped_smem_estimator(*consts, jaxpr: jax_core.Jaxpr) -> int:
  del consts  # Unused.
  in_avals = (v.aval.inner_aval for v in jaxpr.invars)
  return sum(math.prod(aval.shape) * aval.dtype.itemsize for aval in in_avals)


@_regiter_smem_estimator(lax.reduce_sum_p)
def _reduce_sum_smem_estimator(x_aval: jax_core.ShapedArray, *, axes) -> int:
  if axes != (0,):
    raise NotImplementedError("No support for axes other than 0 yet")
  return 4 * x_aval.dtype.itemsize


@dataclasses.dataclass
class ModuleContext:
  name: str
  grid_mapping: pallas_core.GridMapping
  runtime_smem: ir.Value  # ir.MemRefType
  smem_used_bytes: int

  # TODO(cperivol): Only return the shapes and figure out the sizes when freeing.
  def scratch_view(
      self, structs: Sequence[jax.ShapeDtypeStruct]
  ) -> tuple[int, Sequence[ir.Value]]:
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
    smem_scratch_bytes = math.prod(ir.MemRefType(self.runtime_smem.type).shape)
    required_scratch_bytes = sum(
        math.prod(sh.shape) * jnp.dtype(sh.dtype).itemsize for sh in structs
    )
    if smem_scratch_bytes < required_scratch_bytes:
      raise ValueError(
          f"Too few {smem_scratch_bytes=} provided (pass via compiler_params),"
          f" we need {required_scratch_bytes} ({structs=})"
      )

    views = []
    off = self.smem_used_bytes
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    for s in structs:
      scratch_ty = ir.MemRefType.get(
          s.shape,
          mlir.dtype_to_ir_type(s.dtype),
          memory_space=smem,
      )
      views.append(
          memref_dialect.view(scratch_ty, self.runtime_smem, _as_index(off), [])
      )
      off += math.prod(s.shape) * jnp.dtype(s.dtype).itemsize

    total_bytes = off - self.smem_used_bytes
    self.smem_used_bytes = off
    return total_bytes, views

  def stack_free_smem(self, bytes: int):
    """Frees the ``bytes`` last allocated."""
    if bytes > self.smem_used_bytes:
      raise ValueError("Tried to free more bytes than was allocated")
    self.smem_used_bytes -= bytes


@dataclasses.dataclass(frozen=True)
class LoweringRuleContext:
  module_context: ModuleContext
  avals_in: Sequence[jax_core.ShapedArray]
  avals_out: Sequence[jax_core.ShapedArray]

  replace = dataclasses.replace


@dataclasses.dataclass(frozen=True)
class LoweringResult:
  module: ir.Module
  grid: tuple[int, ...]
  out_structs: tuple[jax.ShapeDtypeStruct, ...]


class LoweringError(Exception):  # pylint: disable=g-bad-exception-name
  pass


def _eval_index_map(
    ctx: ModuleContext, idx: ir.Value, block_mapping: pallas_core.BlockMapping
) -> Sequence[ir.Value]:
  block_indices = lower_jaxpr_to_mosaic_gpu(
      ctx, block_mapping.index_map_jaxpr.jaxpr, idx
  )
  result = []
  for i, b in zip(block_indices, block_mapping.block_shape):
    if b is pallas_core.mapped:
      result.append(i)
    else:
      # TODO(slebedev): Use a type-agnostic multiplication wrapper.
      result.append(arith_dialect.muli(_as_index(i), _as_index(b)))
  return tuple(result)


def lower_jaxpr_to_module(
    grid_mapping: pallas_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    name_and_src_info: pallas_core.NameAndSrcInfo,
    compiler_params: dict[str, Any],
    cost_estimate: pallas_core.CostEstimate | None,
) -> LoweringResult:
  del cost_estimate  # Unused.

  block_mappings = grid_mapping.block_mappings

  assert len(jaxpr.outvars) == 0
  assert not grid_mapping.vmapped_dims
  if len(grid_mapping.grid) > 3:
    raise NotImplementedError(
        "Only <=3D grids are supported in Mosaic GPU lowering."
    )
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "Dynamic grid bounds not supported in the Mosaic GPU lowering."
    )
  if grid_mapping.num_index_operands:
    raise NotImplementedError(
        "Scalar prefetch not supported in Mosaic GPU lowering."
    )
  if not all(
      isinstance(bm.indexing_mode, pallas_core.Blocked) for bm in block_mappings
  ):
    raise NotImplementedError(
        "Only Blocked indexing mode is supported in Mosaic GPU lowering."
    )

  with grid_mapping.trace_env():
    jaxpr, _ = pe.dce_jaxpr(
        jaxpr, [True] * len(jaxpr.outvars), instantiate=True
    )

  grid = grid_mapping.grid
  if len(grid) < 3:
    grid += (1,) * (3 - len(grid))
  block = (128,) + (1,) * (len(grid) - 1)

  num_inputs = grid_mapping.num_inputs
  params = compiler_params.get("mosaic_gpu", {})
  num_stages = params.get("num_stages", 1)
  dimension_semantics = params.get(
      "dimension_semantics", ["parallel"] * len(grid_mapping.grid)
  )
  if len(dimension_semantics) != len(grid_mapping.grid):
    raise ValueError(
        "dimension_semantics must have an entrey for each grid dimension:"
        f" {len(dimension_semantics)=}, but len(grid={grid_mapping.grid})."
    )
  sequential_axes = tuple(
      i for i, s in enumerate(dimension_semantics) if s == "sequential"
  )
  assert all(grid[axis] for axis in sequential_axes)
  assert all(block[axis] == 1 for axis in sequential_axes)

  in_structs_gmem = [*grid_mapping.in_shapes]
  in_block_shapes = [
      bm.block_shape
      for bm in grid_mapping.block_mappings[:num_inputs]
  ]
  in_structs_smem = [
      jax.ShapeDtypeStruct(
          [num_stages,
           *bm.ref_aval.inner_aval.shape],  # pytype: disable=attribute-error
          bm.ref_aval.inner_aval.dtype)  # pytype: disable=attribute-error
      for bm in block_mappings[:num_inputs]
  ]
  in_gmem_transforms = [
    bm.transforms for bm in grid_mapping.block_mappings[:num_inputs]
  ]
  _get_swizzle = (
      lambda bm: bm.swizzle
      if isinstance(bm, gpu_core.GPUBlockMapping)
      else None
  )
  in_swizzles = map(_get_swizzle, grid_mapping.block_mappings[:num_inputs])
  out_structs_gmem = [*grid_mapping.out_shapes]
  # TODO(justinfu): Implement output Memref transforms
  out_structs_smem = [
      jax.ShapeDtypeStruct([num_stages, *bm.block_shape], s.dtype)
      for bm, s in zip(
          block_mappings[grid_mapping.num_inputs :],
          grid_mapping.out_shapes,
      )
  ]

  def body(launch_ctx: mosaic_gpu.LaunchContext, *buffers: ir.Value):
    *buffers_gmem, (*buffers_smem, runtime_smem, barriers) = buffers
    assert len(buffers_gmem) == len(buffers_smem)
    in_buffers_gmem, out_buffers_gmem = util.split_list(
        buffers_gmem, [grid_mapping.num_inputs]
    )
    in_buffers_smem, out_buffers_smem = util.split_list(
        buffers_smem, [grid_mapping.num_inputs]
    )

    module_ctx = ModuleContext(
        name_and_src_info.name, grid_mapping, runtime_smem, smem_used_bytes=0
    )
    program_ids = map(_program_id, range(len(grid_mapping.grid)))
    start_indices = map(
        functools.partial(_eval_index_map, module_ctx, program_ids),
        block_mappings,
    )
    in_start_indices, out_start_indices = util.split_list(
        start_indices, [grid_mapping.num_inputs]
    )

    def gmem_slice(
        start_indices: Sequence[ir.Value],
        step: ir.Value,
        shape: Sequence[int],
    ) -> Sequence[mgpu.DynamicSlice]:
      return tuple(
          mgpu.ds(
              arith_dialect.addi(
                  start_index, arith_dialect.muli(step, _as_index(dim))
              )
              if axis in sequential_axes
              else start_index,
              dim,
          )
          for axis, (start_index, dim) in enumerate(zip(start_indices, shape))
      )

    def fetch(idx: int, step: ir.Value, slot: ir.Value) -> None:
      # TODO(slebedev): Support 128-byte swizzling, once we can lower matmuls.
      gmem_transforms = (x.to_gpu_transform() for x in in_gmem_transforms[idx])
      launch_ctx.async_copy(
          src_ref=in_buffers_gmem[idx],
          dst_ref=mgpu.memref_slice(in_buffers_smem[idx], slot),
          gmem_slice=gmem_slice(
              in_start_indices[idx],
              step,
              in_block_shapes[idx],
          ),
          barrier=barriers[slot],
          gmem_transform=tuple(gmem_transforms),
          swizzle=in_swizzles[idx],
          arrive=True,
          uniform=False,
      )

    def store(idx: int, step: ir.Value, slot: ir.Value) -> None:
      # TODO(slebedev): Support 128-byte swizzling, once we can lower matmuls.
      launch_ctx.async_copy(
          src_ref=mgpu.memref_slice(out_buffers_smem[idx], slot),
          dst_ref=out_buffers_gmem[idx],
          gmem_slice=gmem_slice(
              out_start_indices[idx],
              step,
              ir.MemRefType(out_buffers_smem[idx].type).shape[1:],
          ),
          swizzle=None,
          uniform=False,
      )

    # Compute the number of steps along each sequential axis.
    if sequential_axes:
      # TODO(slebedev): Support multiple sequential axes.
      if len(sequential_axes) > 1:
        raise NotImplementedError(
            "Multiple sequential axes are not supported in Mosaic GPU lowering."
        )
      [sequential_axis] = sequential_axes
      if any(
          b_gmem.shape[sequential_axis] % b_smem.shape[1 + sequential_axis]
          for b_gmem, b_smem in zip(in_structs_gmem, in_structs_smem)
      ):
        raise ValueError(
            "Array dimensions along the sequential axis must be divisible by"
            " the corresponding block dimensions."
        )
      num_steps, *rest = {
          b_gmem.shape[sequential_axis] // b_smem.shape[1 + sequential_axis]
          for b_gmem, b_smem in zip(in_structs_gmem, in_structs_smem)
      }
      if rest:
        raise ValueError(
            "Array dimensions along the sequential axis must produce the same"
            " number of steps when devided by the corresponding block"
            " dimensions."
        )
    else:
      num_steps = 1

    with mgpu.single_thread():
      for slot in range(min(num_stages, num_steps)):
        for idx in range(grid_mapping.num_inputs):
          fetch(idx, _as_index(slot), _as_index(slot))

    @mgpu.fori(_as_index(num_steps), ())
    def _(step, _):
      slot = arith_dialect.remui(step, _as_index(num_stages))
      if grid_mapping.num_inputs:
        # Only wait if async copies were issued.
        barriers[slot].wait()

      _ = lower_jaxpr_to_mosaic_gpu(
          module_ctx,
          jaxpr,
          [mgpu.memref_slice(b_smem, slot) for b_smem in buffers_smem],
      )
      mgpu.commit_shared()

      with mgpu.single_thread():
        for idx in range(grid_mapping.num_outputs):
          store(idx, step, slot)

      next_step = arith_dialect.addi(step, _as_index(num_stages))
      next_step_in_bounds = arith_dialect.cmpi(
          arith_dialect.CmpIPredicate.ult, next_step, _as_index(num_steps)
      )
      with mgpu.when(next_step_in_bounds), mgpu.single_thread():
        for idx in range(grid_mapping.num_inputs):
          fetch(idx, next_step, slot)

      return ()

    launch_ctx.await_async_copy(0)

  smem_scratch_bytes = compiler_params.get("smem_scratch_bytes")
  if smem_scratch_bytes is None:
    smem_scratch_bytes = _estimate_smem_scratch_bytes(jaxpr)
  extra_smem_scratch = [
      jax.ShapeDtypeStruct(shape=[smem_scratch_bytes], dtype=np.int8)
  ]
  module, out_structs_smem, _ = mosaic_gpu._lower_as_gpu_kernel(
      body,
      grid=grid,
      cluster=(),
      block=block,
      in_shapes=in_structs_gmem,
      out_shape=out_structs_gmem,
      smem_scratch_shape=(
          *in_structs_smem,
          *out_structs_smem,
          *extra_smem_scratch,
          mgpu.Barrier(
              arrival_count=len(in_structs_gmem),
              num_barriers=num_stages,
          ),
      ),
      module_name=name_and_src_info.name,
  )

  return LoweringResult(module, grid, out_structs_smem)


mosaic_lowering_rules = {}


def register_lowering_rule(primitive: jax_core.Primitive):
  def deco(fn):
    mosaic_lowering_rules[primitive] = fn
    return fn

  return deco


def lower_jaxpr_to_mosaic_gpu(
    ctx: ModuleContext,
    jaxpr: jax_core.Jaxpr,
    args: Sequence[ir.Value],
    consts=(),
) -> Sequence[ir.Value]:
  env = {}

  def read_env(atom: jax_core.Atom):
    return atom.val if isinstance(atom, jax_core.Literal) else env[atom]

  def write_env(var: jax_core.Var, val):
    env[var] = val

  map(write_env, jaxpr.constvars, consts)
  map(write_env, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    invals = map(read_env, eqn.invars)
    if eqn.primitive not in mosaic_lowering_rules:
      raise NotImplementedError(
          "Unimplemented primitive in Pallas Mosaic GPU lowering: "
          f"{eqn.primitive.name}. "
          "Please file an issue on https://github.com/google/jax/issues."
      )
    rule = mosaic_lowering_rules[eqn.primitive]
    rule_ctx = LoweringRuleContext(
        ctx,
        avals_in=[cast(jax_core.ShapedArray, v.aval) for v in eqn.invars],
        avals_out=[cast(jax_core.ShapedArray, v.aval) for v in eqn.outvars],
    )
    try:
      outvals = rule(rule_ctx, *invals, **eqn.params)
    except LoweringError:
      raise  # We only add the extra info to the innermost exception.
    except Exception as e:
      inval_types = map(lambda t: getattr(t, "type", None), invals)
      raise LoweringError(
          f"Exception while lowering eqn:\n  {eqn}\nWith context:\n "
          f" {rule_ctx}\nWith inval types={inval_types}\nIn jaxpr:\n{jaxpr}"
      ) from e
    if eqn.primitive.multiple_results:
      map(write_env, eqn.outvars, outvals)
    else:
      write_env(eqn.outvars[0], outvals)
  return map(read_env, jaxpr.outvars)


@register_lowering_rule(primitives.program_id_p)
def _program_id_lowering_rule(ctx: LoweringRuleContext, axis):
  del ctx  # Unused.
  return _program_id(axis)


def _program_id(axis: int) -> ir.Value:
  return arith_dialect.index_cast(
      ir.IntegerType.get_signless(32),
      gpu_dialect.block_id(gpu_dialect.Dimension(axis)),
  )


@register_lowering_rule(primitives.num_programs_p)
def _num_programs_lowering_rule(ctx: LoweringRuleContext, axis):
  del ctx  # Unused.
  return arith_dialect.index_cast(
      ir.IntegerType.get_signless(32),
      gpu_dialect.block_dim(gpu_dialect.Dimension(axis)),
  )


@register_lowering_rule(sp.get_p)
def _get_lowering_rule(ctx: LoweringRuleContext, x_smem, *indexers, tree):
  del ctx, tree  # Unused.
  if indexers:
    raise NotImplementedError("No support for indexers yet")
  return mgpu.FragmentedArray.load_strided(x_smem)


@register_lowering_rule(sp.swap_p)
def _swap_lowering_rule(
    ctx: LoweringRuleContext, x_smem, value, *indexers, tree
):
  del ctx, tree  # Unused.
  if indexers:
    raise NotImplementedError("No support for indexers yet")
  old_value = mgpu.FragmentedArray.load_strided(x_smem)
  value.store_untiled(x_smem)
  return old_value


@register_lowering_rule(pjit.pjit_p)
def _pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_):
  if jaxpr.consts:
    raise NotImplementedError
  return lower_jaxpr_to_mosaic_gpu(ctx.module_context, jaxpr.jaxpr, args)


@register_lowering_rule(lax.broadcast_in_dim_p)
def _broadcast_in_dim_lowering_rule(
    ctx: LoweringRuleContext,
    x: mgpu.FragmentedArray,
    *,
    broadcast_dimensions,
    shape,
):
  if broadcast_dimensions:
    raise NotImplementedError
  return _ensure_fa(x, ctx.avals_in[0]).broadcast(shape)


@register_lowering_rule(lax.convert_element_type_p)
def _convert_element_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding
):
  del weak_type, sharding
  return _ensure_fa(x, *ctx.avals_in).astype(mlir.dtype_to_ir_type(new_dtype))


def _binary_op_lowering_rule(ctx: LoweringRuleContext, x, y, *, impl):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return impl(x, y)


mosaic_lowering_rules.update({
    lax.add_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x + y),
    lax.sub_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x - y),
    lax.mul_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x * y),
    lax.div_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x / y),
})


@register_lowering_rule(lax.integer_pow_p)
def _integer_pow_lowering_rule(ctx: LoweringRuleContext, x, y):
  x = _ensure_fa(x, *ctx.avals_in)
  if y == 2:
    return x * x
  return NotImplementedError


@register_lowering_rule(lax.rsqrt_p)
def _rsqrt_lowering_rule(ctx: LoweringRuleContext, x):
  return _ensure_fa(x, *ctx.avals_in).rsqrt()


@register_lowering_rule(lax.reduce_sum_p)
def _reduce_sum_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  if axes != (0,):
    raise NotImplementedError("No support for axes other than 0 yet")
  [x_aval] = ctx.avals_in
  _, [scratch] = ctx.module_context.scratch_view(
      [jax.ShapeDtypeStruct(shape=(4,), dtype=x_aval.dtype)]
  )
  return mgpu.FragmentedArray.splat(x.reduce_sum(scratch), ())


@register_lowering_rule(primitives.debug_print_p)
def _debug_print_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    fmt,
    has_placeholders: bool,
):
  del ctx
  del has_placeholders
  primitives.check_debug_print_format(fmt, *args)
  mgpu.debug_print(fmt, *args)
  return ()


@register_lowering_rule(primitives.run_scoped_p)
def _run_scoped_lowering_rule(
    ctx: LoweringRuleContext, *consts, jaxpr: jax_core.Jaxpr
):
  in_avals = [v.aval.inner_aval for v in jaxpr.invars]
  bytes_allocated, input_refs = ctx.module_context.scratch_view(
      [jax.ShapeDtypeStruct(shape=aval.shape, dtype=aval.dtype) for aval in in_avals]
  )
  outs = lower_jaxpr_to_mosaic_gpu(
      ctx.module_context, jaxpr, input_refs, consts
  )
  ctx.module_context.stack_free_smem(bytes_allocated)
  return outs


def _bcast(
    x: ir.Value,
    y: ir.Value,
    x_aval: jax_core.ShapedArray,
    y_aval: jax_core.ShapedArray,
    out_aval: jax_core.ShapedArray,
) -> ir.Value:
  if isinstance(x, (np.ndarray, np.number, int, float)):
    x_dtype = x_aval.dtype
    if x_aval.weak_type:
      x_dtype = y_aval.dtype
    x = mgpu.FragmentedArray.splat(
        _ir_constant(x, mlir.dtype_to_ir_type(x_dtype)), ()
    )
  if isinstance(y, (np.ndarray, np.number, int, float)):
    y_dtype = y_aval.dtype
    if y_aval.weak_type:
      y_dtype = x_aval.dtype
    y = mgpu.FragmentedArray.splat(
        _ir_constant(y, mlir.dtype_to_ir_type(y_dtype)), ()
    )
  assert isinstance(x, mgpu.FragmentedArray)
  assert isinstance(y, mgpu.FragmentedArray)
  if x_aval.shape != out_aval.shape:
    x = x.broadcast(out_aval.shape)
  if y_aval.shape != out_aval.shape:
    y = y.broadcast(out_aval.shape)
  return x, y


def _ensure_fa(x: object, aval: jax_core.ShapedArray) -> mgpu.FragmentedArray:
  if isinstance(x, mgpu.FragmentedArray):
    return x
  elif isinstance(x, (np.number, np.ndarray, int, float)):
    return mgpu.FragmentedArray.splat(
        _ir_constant(x, mlir.dtype_to_ir_type(aval.dtype)), ()
    )
  elif isinstance(x, ir.Value):
    if isinstance(x.type, (ir.IntegerType, ir.FloatType)):
      return mgpu.FragmentedArray.splat(x, ())
  raise NotImplementedError


def _ir_constant(v: object, t: ir.Type) -> ir.Value:
  if isinstance(v, (np.number, np.ndarray, int, float)):
    if isinstance(t, (ir.IntegerType, ir.IndexType)):
      v = int(v)
    else:
      assert isinstance(t, ir.FloatType)
      v = float(v)
    return arith_dialect.constant(t, v)
  raise NotImplementedError(f"Unsupported constant: {v!r}")


def _i32_constant(v: object) -> ir.Value:
  return _ir_constant(v, ir.IntegerType.get_signless(32))


def _as_index(v: int | ir.Value) -> ir.Value:
  if isinstance(v, int):
    return arith_dialect.constant(ir.IndexType.get(), v)
  if ir.IndexType.isinstance(v.type):
    return v
  return arith_dialect.index_cast(ir.IndexType.get(), v)
