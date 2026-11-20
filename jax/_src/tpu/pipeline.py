# Copyright 2026 The JAX Authors.
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

"""TPU lowerings for `lax.pipeline`, `block_slice`, and `block_update`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import contextlib
import dataclasses
import math
import threading
from typing import Any, Protocol

from jax._src import core
from jax._src.interpreters import mlir
from jax._src.lax import compiler_params as cp
from jax._src.lax import slicing
from jax._src.lax.control_flow import loops
from jax._src.lax.control_flow import pipeline
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.tpu import frontend_attrs as tpu_attrs
from jax._src.util import safe_zip, split_list  # pylint: disable=g-multiple-import
import numpy as np

zip, unsafe_zip = safe_zip, zip  # pylint: disable=redefined-builtin

# XLA:TPU frontend attribute keys written by the TPU lowerings below.
DISJOINT_READ_WRITE_REGIONS = "xla_disjoint_read_write_regions"
IMMEDIATE_REVISITING = "xla_while_loop_async_immediate_revisiting"
COLOR_INPUT_TO_HBM = "ColorInputToHBM"


@contextlib.contextmanager
def _temporarily_set(
    state: threading.local, attr: str, value: Any
) -> Iterator[None]:
  """Sets `state.attr` to `value` within the context."""
  prev = getattr(state, attr)
  setattr(state, attr, value)
  try:
    yield
  finally:
    setattr(state, attr, prev)


@dataclasses.dataclass(frozen=True)
class LoopFacts:
  """Structural facts about a `lax.pipeline` body for TPU lowering.

  Attributes:
    trip_count: Static number of grid steps.
    has_nested_loop: Whether the body contains a `while`, `scan`, or
      `lax.pipeline`.
    has_block_write: Whether the body contains a registered block write or
      `dynamic_update_slice`.
    pipeline_depth: Number of enclosing `lax.pipeline` bodies.
    num_consts: Number of leading loop invariant body inputs.
    num_coords: Number of grid coordinate inputs that follow.
    num_carries: Number of carry inputs that follow, matching the outputs.
  """

  trip_count: int
  has_nested_loop: bool
  has_block_write: bool
  pipeline_depth: int
  num_consts: int = 0
  num_coords: int = 0
  num_carries: int = 0


class TpuPipelineHandler(Protocol):
  """TPU lowering callbacks for a `PipelineParams` subclass.

  An implementation may also define
  `verify_pipeline_body(self, params, facts, jaxpr) -> None` to reject an
  unsupported body before lowering emits the loop.
  """

  def pipeline_loop_attributes(
      self, params: Any, facts: LoopFacts
  ) -> Mapping[str, str]:
    """Returns frontend attributes for the pipeline `while` op."""
    ...

  def pipeline_body_attributes(
      self, params: Any, facts: LoopFacts
  ) -> Mapping[str, str]:
    """Returns frontend attributes inherited by ops in the pipeline body."""
    ...

  def block_read_attributes(
      self, params: Any, facts: LoopFacts
  ) -> Mapping[str, str]:
    """Returns frontend attributes for a block read directly in the body."""
    ...

  def block_write_attributes(
      self, params: Any, facts: LoopFacts
  ) -> Mapping[str, str]:
    """Returns frontend attributes for a block write directly in the body."""
    ...


_PIPELINE_HANDLERS: dict[type[Any], TpuPipelineHandler] = {}


def register_pipeline_handler(
    params_cls: type[Any], handler: TpuPipelineHandler
) -> None:
  """Registers `handler` as the TPU lowering handler for `params_cls`."""
  _PIPELINE_HANDLERS[params_cls] = handler


def unregister_pipeline_handler(params_cls: type[Any]) -> None:
  """Removes any TPU lowering handler registered for `params_cls`."""
  _PIPELINE_HANDLERS.pop(params_cls, None)


def _pipeline_handler(params: Any) -> TpuPipelineHandler | None:
  """Returns the TPU handler for `params`, or `None` when `params` is `None`."""
  if params is None:
    return None
  for cls in type(params).__mro__:
    if (handler := _PIPELINE_HANDLERS.get(cls)) is not None:
      return handler
  raise NotImplementedError(
      "No TPU lax.pipeline handler registered for compiler_params of type"
      f" {type(params).__name__}."
  )


_BLOCK_WRITE_PRIMITIVES: set[core.Primitive] = {
    pipeline.block_update_p,
    slicing.dynamic_update_slice_p,
}
_NESTED_LOOP_PRIMITIVES = frozenset(
    {loops.while_p, loops.scan_p, pipeline.pipeline_p}
)
# Kernel call primitives whose body jaxprs lower inside a custom call rather
# than XLA HLO, so loops and writes inside them do not affect `LoopFacts`.
_OPAQUE_KERNEL_PRIMITIVE_NAMES = frozenset({"pallas_call", "mpmd_map"})


def register_block_write_primitive(prim: core.Primitive) -> None:
  """Registers `prim` as a block write for `LoopFacts.has_block_write`."""
  _BLOCK_WRITE_PRIMITIVES.add(prim)


def _walk_jaxpr(
    jaxpr: core.Jaxpr, *, into_kernels: bool = True
) -> Iterator[core.JaxprEqn]:
  """Yields the equations of `jaxpr` and of the jaxprs nested in it."""
  for eqn in jaxpr.eqns:
    yield eqn
    if (
        not into_kernels
        and eqn.primitive.name in _OPAQUE_KERNEL_PRIMITIVE_NAMES
    ):
      continue
    for sub in core.jaxprs_in_params(eqn.params):
      yield from _walk_jaxpr(sub, into_kernels=into_kernels)


@dataclasses.dataclass(frozen=True)
class EnclosingPipeline:
  """A `lax.pipeline` whose body is being lowered on TPU."""

  params: Any
  facts: LoopFacts


class _LoweringState(threading.local):
  pipelines: tuple[EnclosingPipeline, ...] = ()


_lowering_state = _LoweringState()


def lowering_pipeline_body(
    frame: EnclosingPipeline,
) -> contextlib.AbstractContextManager[None]:
  """Enters `frame` as the enclosing pipeline body during TPU lowering."""
  return _temporarily_set(
      _lowering_state, "pipelines", (*_lowering_state.pipelines, frame)
  )


def enclosing_pipeline() -> EnclosingPipeline | None:
  """Returns the innermost pipeline whose body is being lowered, or `None`.

  Counts pipelines at any depth, including inside calls within the body.
  Register lowerings that read it with `cacheable=False`.

  Returns:
    The innermost `EnclosingPipeline`, or `None` outside a pipeline body.
  """
  pipelines = _lowering_state.pipelines
  return pipelines[-1] if pipelines else None


def enclosing_body_attributes(frame: EnclosingPipeline) -> dict[str, str]:
  """Returns the TPU body attributes of `frame`."""
  handler = _pipeline_handler(frame.params)
  if handler is None:
    return {}
  return tpu_attrs.attributes(
      frame.params, lambda p: handler.pipeline_body_attributes(p, frame.facts)
  )


def loop_facts(
    jaxpr: core.Jaxpr, grid: tuple[int, ...], num_consts: int
) -> LoopFacts:
  """Returns structural facts about a pipeline body and its nested jaxprs.

  `while`, `scan`, and nested pipelines count as loops. Loops and block writes
  inside opaque kernel primitives (`pallas_call`, `mpmd_map`) lower inside the
  custom call rather than XLA HLO and are not counted.

  Args:
    jaxpr: Pipeline body jaxpr.
    grid: Static grid dimensions of the pipeline.
    num_consts: Number of loop invariant operands preceding coordinates.

  Returns:
    `LoopFacts` for `jaxpr`.
  """
  prims = {eqn.primitive for eqn in _walk_jaxpr(jaxpr, into_kernels=False)}
  return LoopFacts(
      trip_count=math.prod(grid),
      has_nested_loop=bool(prims & _NESTED_LOOP_PRIMITIVES),
      has_block_write=bool(prims & _BLOCK_WRITE_PRIMITIVES),
      pipeline_depth=len(_lowering_state.pipelines),
      num_consts=num_consts,
      num_coords=len(grid),
      num_carries=len(jaxpr.outvars),
  )


def _owner(value: ir.Value) -> ir.Operation | None:
  """Returns the op that defines `value`, or `None` for a block argument."""
  owner = value.owner
  if isinstance(owner, ir.OpView):
    return owner.operation
  return owner if isinstance(owner, ir.Operation) else None


def _attach_to_owner(value: ir.Value, attrs: Mapping[str, str]) -> None:
  owner = _owner(value)
  assert owner is not None, value
  tpu_attrs.attach_frontend_attributes(owner, attrs)


def emit_block_read(
    ctx: mlir.LoweringRuleContext,
    src: ir.Value,
    index: Sequence[ir.Value],
    attrs: Mapping[str, str],
) -> ir.Value:
  """Emits the `dynamic_slice` of a block read with `attrs` on it."""
  (aval_out,) = ctx.avals_out
  out = mlir.dynamic_slice(ctx, aval_out, src, start_indices=index)
  _attach_to_owner(out, attrs)
  return mlir.lower_with_explicit_types(ctx, out, aval_out)


def emit_block_write(
    ctx: mlir.LoweringRuleContext,
    dst: ir.Value,
    tile: ir.Value,
    index: Sequence[ir.Value],
    attrs: Mapping[str, str],
) -> ir.Value:
  """Emits the `dynamic_update_slice` of a block write with `attrs` on it."""
  (aval_out,) = ctx.avals_out
  out = mlir.dynamic_update_slice(ctx, aval_out, dst, tile, start_indices=index)
  _attach_to_owner(out, attrs)
  return mlir.lower_with_explicit_types(ctx, out, aval_out)


def _require_enclosing_pipeline(name: str) -> EnclosingPipeline:
  # `_pipeline_tpu_lowering` checks placement in its body, so a block op with
  # no enclosing pipeline is at the top level.
  if (frame := enclosing_pipeline()) is None:
    raise ValueError(pipeline.body_only_placement_message(name))
  return frame


def tpu_block_read_attributes(*, revisits_tile: bool) -> dict[str, str]:
  """Returns TPU frontend attributes for a block read directly in a body.

  The read's own attributes override those of the enclosing pipeline params.

  Args:
    revisits_tile: Whether consecutive grid steps may revisit the same tile.

  Returns:
    Frontend attributes for the emitted `dynamic_slice` op.

  Raises:
    ValueError: Outside a pipeline body.
  """
  frame = _require_enclosing_pipeline("A block read")
  handler = _pipeline_handler(frame.params)
  attrs: dict[str, str] = {}
  if handler is not None:
    attrs = tpu_attrs.attributes(
        frame.params, lambda p: handler.block_read_attributes(p, frame.facts)
    )
  if not revisits_tile:
    # Read by WhileLoopAsyncPipeliner: the prefetch slot may be reused.
    attrs[IMMEDIATE_REVISITING] = "false"
  return attrs


def tpu_block_write_attributes(*, disjoint: bool) -> dict[str, str]:
  """Returns TPU frontend attributes for a block write directly in a body.

  Args:
    disjoint: Whether pending writes are disjoint from other steps' reads.

  Returns:
    Frontend attributes for the emitted `dynamic_update_slice` op.

  Raises:
    ValueError: Outside a pipeline body.
  """
  frame = _require_enclosing_pipeline("A block write")
  handler = _pipeline_handler(frame.params)
  attrs: dict[str, str] = {}
  if handler is not None:
    attrs = tpu_attrs.attributes(
        frame.params, lambda p: handler.block_write_attributes(p, frame.facts)
    )
  if disjoint:
    # WhileLoopAsyncPipeliner keeps a prefetch from a buffer the loop updates
    # only if every update into it carries this key.
    attrs[DISJOINT_READ_WRITE_REGIONS] = "true"
  if frame.params is None:
    # A loop without params has no loop keys and runs unpipelined, so the
    # write's operand stays in HBM. With params, `block_write_attributes`
    # decides.
    attrs[COLOR_INPUT_TO_HBM] = "false"
  return attrs


def _block_slice_tpu_lowering(ctx, src, *index, block_shape, revisits_tile):
  del block_shape
  attrs = tpu_block_read_attributes(revisits_tile=revisits_tile)
  return [emit_block_read(ctx, src, index, attrs)]


def _block_update_tpu_lowering(ctx, dst, tile, *index, disjoint):
  attrs = tpu_block_write_attributes(disjoint=disjoint)
  return [emit_block_write(ctx, dst, tile, index, attrs)]


def _blocks(op: ir.Operation | ir.OpView) -> Iterator[ir.Block]:
  if isinstance(op, ir.OpView):
    op = op.operation
  for region in op.regions:
    yield from region.blocks


def _walk_block(block: ir.Block) -> Iterator[ir.Operation]:
  for op in block.operations:
    op = op.operation
    yield op
    for b in _blocks(op):
      yield from _walk_block(b)


def _pipeline_tpu_lowering(
    ctx: mlir.LoweringRuleContext,
    *args,
    grid,
    num_consts,
    jaxpr,
    compiler_params,
):
  """Lowers `pipeline_p` on TPU to a tagged `stablehlo.while`."""
  params: pipeline.PipelineParams | None = compiler_params
  pipeline.check_body_only_placement(jaxpr)
  handler = _pipeline_handler(params)
  facts = loop_facts(jaxpr, grid, num_consts)
  if (verify := getattr(handler, "verify_pipeline_body", None)) is not None:
    verify(params, facts, jaxpr)
  if handler is None:
    loop_attrs: dict[str, str] = {}
    body_attrs: dict[str, str] = {}
  else:
    loop_attrs = tpu_attrs.attributes(
        params, lambda p: handler.pipeline_loop_attributes(p, facts)
    )
    body_attrs = tpu_attrs.attributes(
        params, lambda p: handler.pipeline_body_attributes(p, facts)
    )

  # The loop state is (*constants, flat grid step, *carries): an ND grid runs
  # as one `stablehlo.while` and the body derives its coordinates by divmod.
  state_avals = [
      *ctx.avals_in[:num_consts],
      pipeline.COORD_AVAL,
      *ctx.avals_in[num_consts:],
  ]
  state_types = [
      mlir._aval_to_ir_types(ctx.module_context, a)  # pylint: disable=protected-access
      for a in state_avals
  ]
  flat_types, state_tree = mlir.ir_tree_registry.flatten(state_types)
  consts, init = split_list(args, [num_consts])
  zero = mlir.ir_constant(np.int32(0))
  flat_args, _ = mlir.ir_tree_registry.flatten([*consts, zero, *init])
  while_op = hlo.WhileOp(flat_types, flat_args)
  name_stack = ctx.name_stack.extend("pipeline")

  cond_block = while_op.regions[0].blocks.append(*flat_types)
  with ir.InsertionPoint(cond_block):
    state = state_tree.unflatten(list(cond_block.arguments))
    trip_count = mlir.ir_constant(np.int32(math.prod(grid)))
    pred = mlir.compare_hlo(state[num_consts], trip_count, "LT", "SIGNED")
    hlo.return_([pred])

  body_block = while_op.regions[1].blocks.append(*flat_types)
  with ir.InsertionPoint(body_block):
    state = state_tree.unflatten(list(body_block.arguments))
    body_consts, (flat,), carries = split_list(state, [num_consts, 1])
    coords = pipeline.unravel_grid_index(
        flat,
        grid,
        hlo.remainder,
        hlo.divide,
        lambda n: mlir.ir_constant(np.int32(n)),
    )
    jaxpr_consts = mlir.ir_consts(
        jaxpr.consts,
        [v.aval for v in jaxpr.constvars],
        const_lowering=ctx.const_lowering,
    )
    # Empty lowering caches: they key on the jaxpr, and a function lowers
    # differently in this body than outside it.
    body_ctx = cp.isolated_lowering_context(ctx.module_context)
    with lowering_pipeline_body(EnclosingPipeline(params, facts)):
      outs, _ = mlir.jaxpr_subcomp(
          body_ctx,
          jaxpr,
          name_stack.extend("body"),
          mlir.TokenSet(),
          jaxpr_consts,
          *body_consts,
          *coords,
          *carries,
          dim_var_values=ctx.dim_var_values,
          const_lowering=ctx.const_lowering,
          outer_traceback=ctx.traceback,
      )
    next_flat = hlo.add(flat, mlir.ir_constant(np.int32(1)))
    flat_outs, _ = mlir.ir_tree_registry.flatten(
        [*body_consts, next_flat, *outs]
    )
    hlo.return_(flat_outs)

  # The body attributes go on every op of the body, nested regions included,
  # since XLA reads keys such as `scoped_vmem_limit_kib` from the ops it fuses
  # later. An op's own keys win, and callees keep their own attributes.
  for op in _walk_block(body_block):
    if op.name not in ("stablehlo.return", "stablehlo.constant"):
      tpu_attrs.attach_missing_frontend_attributes(op, body_attrs)
  tpu_attrs.attach_frontend_attributes(while_op.operation, loop_attrs)

  results = state_tree.unflatten(list(while_op.results))
  _, _, carries_out = split_list(results, [num_consts, 1])
  return [
      mlir.lower_with_sharding_in_types(ctx, op, aval)
      for op, aval in zip(carries_out, ctx.avals_out)
  ]


mlir.register_lowering(
    pipeline.block_slice_p,
    _block_slice_tpu_lowering,
    platform="tpu",
    cacheable=False,
)
mlir.register_lowering(
    pipeline.block_update_p,
    _block_update_tpu_lowering,
    platform="tpu",
    cacheable=False,
)
mlir.register_lowering(
    pipeline.pipeline_p,
    _pipeline_tpu_lowering,
    platform="tpu",
    cacheable=False,
)
