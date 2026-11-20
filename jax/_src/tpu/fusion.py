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

"""TPU lowering for `lax.fusion`.

XLA:TPU's `UserGuidedFusionIdAssigner` turns any group key on a `func.call` op
into a fresh `MUST_FUSE` id before inlining. Lowering keeps `lax.fusion` call
ops free of group keys, writes the combined directive onto fusible ops in an
isolated callee, and keeps `MAXIMAL_FUSE` on inlined leaf helper calls so each
helper callsite forms its own `MUST_FUSE` group.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
import dataclasses
import threading
from typing import Any, Protocol

from jax._src import core
from jax._src.interpreters import mlir
from jax._src.lax import compiler_params as cp
from jax._src.lax import fusion
from jax._src.lib.mlir import ir
from jax._src.tpu import frontend_attrs as tpu_attrs
from jax._src.tpu import pipeline as tpu_pipeline


@dataclasses.dataclass(frozen=True)
class FusionContext:
  """Context of a lowered `lax.fusion` call passed to `TpuFusionHandler`.

  Attributes:
    fusion_id: Id unique per lowered fusion call in the module.
    num_consts: Number of leading operands closed over from an enclosing trace.
    num_args: Number of explicit flattened arguments after the closed over
      operands.
  """

  fusion_id: int
  num_consts: int
  num_args: int


class TpuFusionHandler(Protocol):
  """TPU lowering callbacks for a `FusionParams` subclass.

  An implementation may also define
  `verify_fusion_body(self, params, ctx, jaxpr) -> None` and
  `verify_fusion_operands(self, params, ctx, operands) -> None` to validate a
  fusion call before lowering emits it.
  """

  def fusion_call_attributes(
      self, params: Any, ctx: FusionContext
  ) -> Mapping[str, str]:
    """Returns frontend attributes for the fusion call op."""
    ...

  def fusion_body_attributes(
      self, params: Any, ctx: FusionContext
  ) -> Mapping[str, str]:
    """Returns frontend attributes for each fusible op in the fusion body."""
    ...


_FUSION_HANDLERS: dict[type[Any], TpuFusionHandler] = {}


def register_fusion_handler(
    params_cls: type[Any], handler: TpuFusionHandler
) -> None:
  """Registers `handler` as the TPU lowering handler for `params_cls`."""
  _FUSION_HANDLERS[params_cls] = handler


def unregister_fusion_handler(params_cls: type[Any]) -> None:
  """Removes any TPU lowering handler registered for `params_cls`."""
  _FUSION_HANDLERS.pop(params_cls, None)


def _fusion_handler(params: Any) -> TpuFusionHandler | None:
  """Returns the TPU handler for `params`, or `None` when `params` is `None`."""
  if params is None:
    return None
  for cls in type(params).__mro__:
    if (handler := _FUSION_HANDLERS.get(cls)) is not None:
      return handler
  raise NotImplementedError(
      "No TPU lax.fusion handler registered for compiler_params of type"
      f" {type(params).__name__}."
  )


class _LoweringState(threading.local):
  depth: int = 0


_lowering_state = _LoweringState()


@contextlib.contextmanager
def lowering_fusion_body() -> Iterator[None]:
  """Marks the dynamic extent of lowering a `lax.fusion` body on TPU."""
  _lowering_state.depth += 1
  try:
    yield
  finally:
    _lowering_state.depth -= 1


def in_fusion_body() -> bool:
  """Returns whether a `lax.fusion` body is being lowered, at any depth.

  Register lowerings that call this with `cacheable=False`.

  Returns:
    `True` inside `lowering_fusion_body()`, `False` otherwise.
  """
  return _lowering_state.depth > 0


class _ModuleFusionState:
  """Per module TPU lowering state, shared across isolated lowering contexts."""

  def __init__(self):
    self.last_id = 0
    # Effective group directive of each lowered `lax.fusion` body function.
    self.callees: dict[str, _Directive | None] = {}


def _fusion_state(mod_ctx: mlir.ModuleContext) -> _ModuleFusionState:
  return cp.module_state(mod_ctx).setdefault("lax.fusion", _ModuleFusionState())


# Returns, constants, barriers, and sharding ops skipped by body tagging.
_UNTAGGED_OPS = frozenset({
    "func.return",
    "stablehlo.return",
    "stablehlo.constant",
    "stablehlo.optimization_barrier",
    "sdy.sharding_constraint",
    "sdy.return",
})
# Compiler directive custom calls skipped by body tagging.
_DIRECTIVE_CUSTOM_CALLS = frozenset({
    "LayoutConstraint",
    "Sharding",
    "annotate_device_placement",
})
_DIRECTIVE_CUSTOM_CALL_PREFIXES = ("xla_color_",)
# Control flow ops whose regions carry fusible subcomputations of the body.
_TAGGED_REGION_OPS = frozenset({
    "stablehlo.while",
    "stablehlo.case",
    "stablehlo.if",
})
_MUST_FUSE = "MUST_FUSE"
_MAXIMAL_FUSE = "MAXIMAL_FUSE"
_GROUP_KEYS = (_MUST_FUSE, _MAXIMAL_FUSE)


@dataclasses.dataclass(frozen=True)
class _Directive:
  """A fusion group directive (`MUST_FUSE` or `MAXIMAL_FUSE` and group id)."""

  key: str
  group: str


def _read_directive(attrs: Mapping[str, str]) -> _Directive | None:
  """Reads the group directive from `attrs`; `MUST_FUSE` wins if both exist."""
  for key in _GROUP_KEYS:
    if key in attrs:
      return _Directive(key, attrs[key])
  return None


def _write_directive(op: ir.Operation, directive: _Directive | None) -> None:
  """Replaces any group keys on `op` with at most one key from `directive`."""
  attrs = tpu_attrs.frontend_attributes(op)
  kept = {k: v for k, v in attrs.items() if k not in _GROUP_KEYS}
  if directive is not None:
    kept[directive.key] = directive.group
  if kept == attrs:
    return
  if "mhlo.frontend_attributes" in op.attributes:
    del op.attributes["mhlo.frontend_attributes"]
  tpu_attrs.attach_frontend_attributes(op, kept)


def _combine(
    outer: _Directive | None, inner: _Directive | None
) -> _Directive | None:
  """Returns the outermost `MUST_FUSE`, else the outermost `MAXIMAL_FUSE`."""
  if outer is None:
    return inner
  if inner is None:
    return outer
  if outer.key == _MUST_FUSE:
    return outer
  if inner.key == _MUST_FUSE:
    return inner
  return outer


def _is_tagged(op: ir.Operation) -> bool:
  if op.name in _UNTAGGED_OPS:
    return False
  if op.name == "stablehlo.custom_call":
    target = ir.StringAttr(op.attributes["call_target_name"]).value
    return not (
        target in _DIRECTIVE_CUSTOM_CALLS
        or target.startswith(_DIRECTIVE_CUSTOM_CALL_PREFIXES)
    )
  return True


def _body_ops(op: ir.Operation) -> Iterator[ir.Operation]:
  for region in op.regions:
    for block in region.blocks:
      for inner in block.operations:
        inner = inner.operation
        yield inner
        if inner.name in _TAGGED_REGION_OPS:
          yield from _body_ops(inner)


def _inlined_early(call: ir.Operation) -> bool:
  """Returns whether XLA inlines `call` before fusion."""
  attrs = tpu_attrs.frontend_attributes(call)
  return (
      attrs.get("inlineable") not in ("false", "xla_late")
      and "_xla_compute_type" not in attrs
  )


def _callee_name(call: ir.Operation) -> str:
  return ir.FlatSymbolRefAttr(call.attributes["callee"]).value


def _callee(mod_ctx: mlir.ModuleContext, call: ir.Operation) -> ir.Operation:
  return mod_ctx.symbol_table[_callee_name(call)].operation


def _calls_fusion(
    mod_ctx: mlir.ModuleContext,
    state: _ModuleFusionState,
    func: ir.Operation,
    seen: set[str] | None = None,
) -> bool:
  """Returns whether `func` transitively calls a lowered `lax.fusion`."""
  if seen is None:
    seen = set()
  for op in _body_ops(func):
    if op.name != "func.call":
      continue
    name = _callee_name(op)
    if name in state.callees:
      return True
    if _inlined_early(op) and name not in seen:
      seen.add(name)
      if _calls_fusion(mod_ctx, state, _callee(mod_ctx, op), seen):
        return True
  return False


class _BodyWalker:
  """Applies body attributes and a group directive across a fusion body.

  Group keys follow XLA's override semantics via `_combine` (outer `MUST_FUSE`
  wins, else outer `MAXIMAL_FUSE`). Other keys follow `attach_missing` so the
  innermost setting wins, and stay on nested fusion `func.call` ops rather than
  entering their bodies.

  Helper callees come from `cp.isolated_lowering_context` because their ops are
  tagged in place. Fusion calls drop group keys; inlined leaf helper calls keep
  `MAXIMAL_FUSE` so XLA assigns each callsite its own `MUST_FUSE` group.
  """

  def __init__(
      self,
      mod_ctx: mlir.ModuleContext,
      state: _ModuleFusionState,
      directive: _Directive | None,
      non_group_attrs: Mapping[str, str],
  ):
    self.mod_ctx = mod_ctx
    self.state = state
    self.directive = directive
    self.non_group_attrs = non_group_attrs
    self.visited: set[str] = set()

  def tag(self, func: ir.Operation) -> None:
    """Tags fusible ops and callees under `func`."""
    for op in _body_ops(func):
      if op.name != "func.call":
        if _is_tagged(op):
          tpu_attrs.attach_missing_frontend_attributes(op, self.non_group_attrs)
          inner = _read_directive(tpu_attrs.frontend_attributes(op))
          _write_directive(op, _combine(self.directive, inner))
      elif (name := _callee_name(op)) in self.state.callees:
        call_dir = _read_directive(tpu_attrs.frontend_attributes(op))
        _write_directive(op, None)
        tpu_attrs.attach_missing_frontend_attributes(op, self.non_group_attrs)
        prev_dir = self.state.callees[name]
        nested_dir = _combine(self.directive, _combine(call_dir, prev_dir))
        self.state.callees[name] = nested_dir
        if nested_dir is not None and nested_dir != prev_dir:
          _BodyWalker(self.mod_ctx, self.state, nested_dir, {}).tag(
              _callee(self.mod_ctx, op)
          )
      elif _inlined_early(op):
        attrs = tpu_attrs.frontend_attributes(op)
        call_dir = _read_directive(attrs)
        helper_dir = _combine(self.directive, call_dir)
        callee = _callee(self.mod_ctx, op)
        call_tag: _Directive | None = None
        if (
            helper_dir is not None
            and helper_dir.key == _MAXIMAL_FUSE
            and "scheduling_group" not in attrs
            and not _calls_fusion(self.mod_ctx, self.state, callee)
        ):
          call_tag = helper_dir
        _write_directive(op, call_tag)
        if name not in self.visited:
          self.visited.add(name)
          if helper_dir == self.directive:
            self.tag(callee)
          else:
            _BodyWalker(
                self.mod_ctx, self.state, helper_dir, self.non_group_attrs
            ).tag(callee)


def _fusion_tpu_lowering(
    ctx: mlir.LoweringRuleContext,
    *args,
    jaxpr: core.Jaxpr,
    num_consts: int,
    compiler_params,
):
  """Lowers `fusion_p` on TPU to an isolated callee and tagged `func.call`."""
  # Assigns a module unique `fusion_id`, builds `FusionContext`, and queries
  # the registered `TpuFusionHandler` (or emits `MAXIMAL_FUSE` when `params`
  # is `None`). Top level fusions in a `lax.pipeline` inherit
  # `pipeline_body_attributes` under `fusion_call_attributes`, including
  # inside `jit` callees. Group keys combine via `_combine` onto fusible ops in
  # an isolated callee.
  params: fusion.FusionParams | None = compiler_params
  handler = _fusion_handler(params)
  mod_ctx = (
      ctx.module_context
      if ctx.platforms is None
      else ctx.module_context.replace(platforms=ctx.platforms)
  )
  state = _fusion_state(mod_ctx)
  state.last_id += 1
  num_args = len(ctx.avals_in) - num_consts
  fusion_ctx = FusionContext(
      fusion_id=state.last_id, num_consts=num_consts, num_args=num_args
  )
  if (verify := getattr(handler, "verify_fusion_body", None)) is not None:
    verify(params, fusion_ctx, jaxpr)
  pipe_attrs: dict[str, str] = {}
  if (
      not in_fusion_body()
      and (frame := tpu_pipeline.enclosing_pipeline()) is not None
  ):
    pipe_attrs = tpu_pipeline.enclosing_body_attributes(frame)
  if handler is None:
    own_call_attrs: dict[str, str] = {}
    body_attrs: dict[str, str] = {_MAXIMAL_FUSE: str(fusion_ctx.fusion_id)}
  else:
    own_call_attrs = tpu_attrs.attributes(
        params, lambda p: handler.fusion_call_attributes(p, fusion_ctx)
    )
    body_attrs = tpu_attrs.attributes(
        params, lambda p: handler.fusion_body_attributes(p, fusion_ctx)
    )
  operands, _ = mlir.ir_tree_registry.flatten(args)
  # Validate in the enclosing lowering context before lowering the body.
  if (verify := getattr(handler, "verify_fusion_operands", None)) is not None:
    verify(params, fusion_ctx, operands)
  body_ctx = cp.isolated_lowering_context(mod_ctx)
  with lowering_fusion_body():
    call, outs = fusion.emit_fusion_call(
        ctx, body_ctx, args, jaxpr, share_callee=False
    )
  callee = _callee(mod_ctx, call.operation)
  directive = _combine(
      _combine(_read_directive(pipe_attrs), _read_directive(own_call_attrs)),
      _read_directive(body_attrs),
  )
  state.callees[_callee_name(call.operation)] = directive
  call_attrs = {
      k: v
      for k, v in (pipe_attrs | own_call_attrs).items()
      if k not in _GROUP_KEYS
  }
  non_group_body_attrs = {
      k: v for k, v in body_attrs.items() if k not in _GROUP_KEYS
  }
  # Always walk `callee` so inner ops and calls normalize their group keys.
  _BodyWalker(body_ctx, state, directive, non_group_body_attrs).tag(callee)
  tpu_attrs.attach_frontend_attributes(call.operation, call_attrs)
  return outs


mlir.register_lowering(
    fusion.fusion_p,
    _fusion_tpu_lowering,
    platform="tpu",
    cacheable=False,
    shard_map_collective_rule=lambda *_, **__: None,
)
