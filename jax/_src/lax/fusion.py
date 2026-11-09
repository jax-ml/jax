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

"""`lax.fusion`: a call primitive with platform compiler params.

Platform lowerings may read `compiler_params` and track fusion body state, as
TPU lowering does in `jax._src.tpu.fusion`.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any, overload

from jax._src import api_util
from jax._src import core
from jax._src import dispatch
from jax._src import effects as effects_lib
from jax._src import flattree as ft
from jax._src import util
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import compiler_params as cp
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.tree_util import tree_unflatten

map, unsafe_map = util.safe_map, map  # pylint: disable=redefined-builtin
zip, unsafe_zip = util.safe_zip, zip  # pylint: disable=redefined-builtin


class FusionParams:
  """Base class for `lax.fusion` compiler parameters.

  Platform parameter types subclass `FusionParams` as frozen dataclasses and
  register a platform lowering handler where needed.
  """


class _NotSpecified:
  pass


_NOT_SPECIFIED = _NotSpecified()


@overload
def fusion(
    f: Callable[..., Any], *, compiler_params: Any = None
) -> Callable[..., Any]:
  ...


@overload
def fusion(
    f: _NotSpecified = ..., *, compiler_params: Any = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  ...


@util.set_module("jax.lax")
def fusion(
    f: Callable[..., Any] | _NotSpecified = _NOT_SPECIFIED,
    *,
    compiler_params: Any = None,
) -> Callable[..., Any]:
  """Returns a function or decorator that calls ``f`` as one fusion unit.

  ``fusion(f)(*args)`` or ``@fusion(...)`` computes ``f(*args)`` as one
  ``fusion`` equation whose body is ``f``'s jaxpr, so a backend can fuse the
  body's operations into one kernel according to ``compiler_params``.

  Args:
    f: the function to call. Its arguments and results may be pytrees of arrays.
      When omitted, returns a decorator binding ``compiler_params``.
    compiler_params: ``None``, or a frozen ``FusionParams`` dataclass
      configuring the target platform's lowering. On TPU, ``None`` requests
      maximal fusion of the body; pass ``jax.experimental.nori.FusionOptions``
      for other policy, such as ``must_fuse=True``. Other backends lower to a
      plain function call.

  Returns:
    A wrapped function with the same signature as ``f``, or a decorator when
    ``f`` is omitted.

  ``fusion`` supports ``jit`` and ``vmap``. ``grad`` and ``jvp`` work when no
  tangent reaches its inputs and raise ``NotImplementedError`` otherwise; wrap
  the call in ``jax.custom_vjp`` to differentiate through the body.
  """
  params = cp.check_compiler_params(compiler_params, "lax.fusion", FusionParams)
  if isinstance(f, _NotSpecified):
    return lambda fn: fusion(fn, compiler_params=params)
  if not callable(f):
    raise TypeError(f"lax.fusion: f must be callable, got {f!r}.")

  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    dbg = api_util.debug_info("lax.fusion", f, args, kwargs)
    in_avals = ft.flatten((args, kwargs)).map(core.shaped_abstractify)
    jaxpr, out_avals = pe.trace_to_jaxpr(f, in_avals, dbg)
    jaxpr, consts = _lift_tracer_consts(jaxpr)
    args_flat = ft.flatten((args, kwargs)).vals
    outs = fusion_p.bind(
        *consts,
        *args_flat,
        jaxpr=jaxpr,
        num_consts=len(consts),
        compiler_params=params,
    )
    return tree_unflatten(out_avals.tree, outs)

  return wrapped


def _lift_tracer_consts(jaxpr: core.Jaxpr) -> tuple[core.Jaxpr, list[Any]]:
  """Lifts closed over tracers into leading invars; keeps concrete consts."""
  is_tracer = [isinstance(c, core.Tracer) for c in jaxpr.consts]
  if not any(is_tracer):
    return jaxpr, []
  kept, lifted = util.partition_list(is_tracer, jaxpr.consts)
  # Expose consts as invars, move concrete ones first, and reattach them.
  open_jaxpr = pe.convert_constvars_jaxpr(jaxpr)
  to_front = [not t for t in is_tracer] + [False] * len(jaxpr.invars)
  return (
      pe.move_binders_to_front(open_jaxpr, to_front).with_consts(kept),
      lifted,
  )


fusion_p = core.Primitive("fusion")
fusion_p.multiple_results = True
fusion_p.is_effectful = lambda p: bool(p["jaxpr"].effects)  # type: ignore[assignment]
dispatch.simple_impl(fusion_p)


# Leading `num_consts` operands are closed over tracers, followed by args.
def _fusion_abstract_eval(
    *in_avals, jaxpr: core.Jaxpr, num_consts: int, compiler_params
):
  del in_avals, num_consts, compiler_params
  return jaxpr.out_avals, core.positional_effects(jaxpr)


fusion_p.def_effectful_abstract_eval(_fusion_abstract_eval)


def _fusion_typecheck(
    _, *in_atoms, jaxpr: core.Jaxpr, num_consts: int, compiler_params
):
  """Typechecks a `fusion_p` equation and its body jaxpr."""
  try:
    cp.check_compiler_params(compiler_params, "fusion", FusionParams)
  except TypeError as e:
    raise core.JaxprTypeError(str(e)) from None
  if not 0 <= num_consts <= len(in_atoms):
    raise core.JaxprTypeError(
        f"fusion has num_consts={num_consts} but {len(in_atoms)} operands"
    )
  core.check_distinct_refs(fusion_p, in_atoms)
  in_avals = [x.aval for x in in_atoms]
  if len(in_avals) != len(jaxpr.in_avals) or not all(
      map(core.typecompat, jaxpr.in_avals, in_avals)
  ):
    raise core.JaxprTypeError(
        f"fusion passes operands of types {in_avals} to a body expecting"
        f" {jaxpr.in_avals}"
    )
  core.check_jaxpr(jaxpr)
  return jaxpr.out_avals, core.positional_effects(jaxpr)


core.custom_typechecks[fusion_p] = _fusion_typecheck


def _fusion_batcher(
    axis_data, vals_in, dims_in, *, jaxpr, num_consts, compiler_params
):
  batched_jaxpr, dims_out = batching.batch_jaxpr2(jaxpr, axis_data, dims_in)
  outs = fusion_p.bind(
      *vals_in,
      jaxpr=batched_jaxpr,
      num_consts=num_consts,
      compiler_params=compiler_params,
  )
  return outs, dims_out


batching.fancy_primitive_batchers[fusion_p] = _fusion_batcher


def emit_fusion_call(
    ctx: mlir.LoweringRuleContext,
    mod_ctx: mlir.ModuleContext,
    args,
    jaxpr: core.Jaxpr,
    *,
    share_callee: bool,
) -> tuple[func_dialect.CallOp, list[Any]]:
  """Lowers `jaxpr` to a private function in `mod_ctx` and emits a call to it.

  With `share_callee=True`, equal bodies share one function. With
  `share_callee=False`, each call gets a fresh function whose ops a platform
  lowering such as `jax._src.tpu.fusion` can tag independently.

  Args:
    ctx: Rule context for the enclosing lowering.
    mod_ctx: Module context in which to emit the private callee.
    args: Lowered MLIR operands passed to `jaxpr`.
    jaxpr: Fusion body jaxpr to lower.
    share_callee: Whether equal bodies may share one private function.

  Returns:
    A `(call_op, outputs)` pair with the emitted `func.call` and result values.
  """
  ordered_effects = list(effects_lib.ordered_effects.filter_in(jaxpr.effects))
  tokens_in = ctx.tokens_in.subset(ordered_effects)
  const_args_and_avals = (
      core.jaxpr_const_args(jaxpr)
      if mod_ctx.lowering_parameters.hoist_constants_as_args
      else []
  )
  const_args, const_avals = util.unzip2(const_args_and_avals)
  in_avals = (*const_avals, *jaxpr.in_avals)
  if share_callee:
    func_op, output_types, effects = mlir.lower_called_computation(
        "fusion",
        jaxpr,
        mod_ctx,
        len(const_args),
        in_avals,
        ctx.avals_out,
        tokens_in,
    )
  else:
    effects = list(tokens_in.effects())
    func_op = mlir.lower_jaxpr_to_fun(
        mod_ctx,
        "fusion",
        jaxpr,
        effects,
        num_const_args=len(const_args),
        in_avals=in_avals,
        out_avals=ctx.avals_out,
    )
    # Match `mlir.lower_called_computation` output structure.
    output_types = [mlir.token_type()] * len(effects) + [
        mlir._aval_to_ir_types(mod_ctx, a)  # pylint: disable=protected-access
        for a in ctx.avals_out
    ]
  flat_output_types, treedef = mlir.ir_tree_registry.flatten(output_types)
  tokens = [tokens_in.get(eff) for eff in effects]
  hoisted, _ = mlir.ir_tree_registry.flatten([
      mlir.ir_constants(c, const_lowering=ctx.const_lowering, aval=aval)
      for c, aval in const_args_and_avals
  ])
  flat_args, _ = mlir.ir_tree_registry.flatten(
      (*ctx.dim_var_values, *tokens, *hoisted, *args)
  )
  call = func_dialect.CallOp(
      flat_output_types, ir.FlatSymbolRefAttr.get(func_op.name.value), flat_args
  )
  out_nodes = treedef.unflatten(call.results)
  tokens_out, out_nodes = util.split_list(out_nodes, [len(effects)])
  ctx.set_tokens_out(
      ctx.tokens_in.update_tokens(mlir.TokenSet(dict(zip(effects, tokens_out))))
  )
  return call, [
      mlir.lower_with_sharding_in_types(ctx, o, a)
      for o, a in zip(out_nodes, ctx.avals_out)
  ]


def _rule_module_context(ctx: mlir.LoweringRuleContext) -> mlir.ModuleContext:
  """Returns `ctx.module_context` narrowed to `ctx.platforms` when set."""
  if ctx.platforms is None:
    return ctx.module_context
  return ctx.module_context.replace(platforms=ctx.platforms)


# `cacheable=False`: platform lowerings such as `jax._src.tpu.fusion` allocate
# per call ids and callees and track fusion and pipeline nesting state.
def _fusion_lowering(
    ctx: mlir.LoweringRuleContext,
    *args,
    jaxpr: core.Jaxpr,
    num_consts: int,
    compiler_params,
):
  """Lowers `fusion_p` to a plain `func.call`."""
  del num_consts, compiler_params
  mod_ctx = _rule_module_context(ctx)
  if ctx.platforms is not None:
    # Callee caches key only on the jaxpr, so isolate them per platform set.
    caches = cp.module_state(mod_ctx).setdefault(
        "lax.fusion.branch_caches:" + ",".join(ctx.platforms),
        cp.isolated_lowering_context(mod_ctx),
    )
    mod_ctx = mod_ctx.replace(
        lowering_cache=caches.lowering_cache,
        cached_primitive_lowerings=caches.cached_primitive_lowerings,
    )
  _, outs = emit_fusion_call(ctx, mod_ctx, args, jaxpr, share_callee=True)
  return outs


mlir.register_lowering(
    fusion_p,
    _fusion_lowering,
    cacheable=False,
    shard_map_collective_rule=lambda *_, **__: None,
)
