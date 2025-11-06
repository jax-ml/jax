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

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
from typing import Any, Protocol

from jax._src import api_util
from jax._src import core
from jax._src import custom_api_util
from jax._src import linear_util as lu
from jax._src.traceback_util import api_boundary
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas.mosaic import lowering as mosaic_lowering
from jax._src.pallas import core as pallas_core
from jax._src.pallas.fuser import block_spec as block_spec_lib


custom_fusion_p = core.Primitive('custom_fusion')
custom_fusion_p.multiple_results = True

CustomPullBlockSpecRuleFn = Callable[[tuple[pallas_core.BlockSpec, ...]],
                                     Sequence[pallas_core.BlockSpec]]

CustomPushBlockSpecRuleFn = Callable[[tuple[pallas_core.BlockSpec, ...]],
                                     tuple[pallas_core.BlockSpec, ...]]

@dataclasses.dataclass(frozen=True)
class CustomEvalContext:
  out_block_specs: tuple[pallas_core.BlockSpec, ...]
  out_block_indices: tuple[Any, ...]

class CustomEvalRuleFn(Protocol):

  def __call__(
      self,
      ctx: CustomEvalContext,
      *args: Any,
  ) -> Sequence[Any]:
    ...


@custom_api_util.register_custom_decorator_type
class custom_fusion:
  fun: Callable[..., Any]

  eval_rule: CustomEvalRuleFn | None = None

  pull_block_spec_rule: CustomPullBlockSpecRuleFn | None = None

  # Optional if this custom_fusion is only used as an input fusion.
  push_block_spec_rule: CustomPushBlockSpecRuleFn | None = None

  # Optional alternative implementation to use instead of `fun` for when this
  # custom fusion is run inside a Pallas kernel.
  pallas_impl: Callable[..., Any] | None = None

  def __init__(self, fun: Callable[..., Any]):
    functools.update_wrapper(self, fun)
    self.fun = fun

  def def_pallas_impl(self, pallas_impl):
    self.pallas_impl = pallas_impl
    return pallas_impl

  def def_pull_block_spec(
      self, pull_block_spec_rule: CustomPullBlockSpecRuleFn):
    self.pull_block_spec_rule = pull_block_spec_rule
    return pull_block_spec_rule

  def def_push_block_spec(
      self, push_block_spec_rule: CustomPushBlockSpecRuleFn):
    self.push_block_spec_rule = push_block_spec_rule
    return push_block_spec_rule

  def def_eval_rule(self, eval_rule: CustomEvalRuleFn):
    self.eval_rule = eval_rule
    return eval_rule

  @functools.partial(api_boundary,
                     repro_api_name="jax.pallas.custom_fusion.__call__")
  def __call__(self, *args, **kwargs):
    debug_fun = api_util.debug_info("custom_fusion fun", self.fun, args, kwargs)

    # TODO(jburnim): Better error messages here.
    assert self.eval_rule is not None
    assert self.pull_block_spec_rule is not None

    try:
      args = api_util.resolve_kwargs(self.fun, args, kwargs)
    except TypeError as e:
      raise TypeError(
          "The input arguments to the custom_fusion-decorated function "
          f"{debug_fun.func_name} could not be resolved to positional-only "
          f"arguments. Binding failed with the error:\n{e}"
      ) from e

    # flatten and get jaxpr
    args_flat, in_tree = tree_util.tree_flatten(args)
    in_avals = [core.get_aval(x) for x in args_flat]
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(
        lu.wrap_init(self.fun, debug_info=debug_fun.with_unknown_names()),
        in_tree)
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)

    # if a Pallas implementation was provided, get its jaxpr
    if self.pallas_impl is not None:
      debug_pallas_impl = api_util.debug_info(
          "custom_fusion pallas_impl", self.pallas_impl, args, kwargs)

      flat_pallas_impl, pallas_out_tree = api_util.flatten_fun_nokwargs(
          lu.wrap_init(self.pallas_impl, debug_info=debug_pallas_impl),
          in_tree)
      # TODO(jburnim): Error if out_tree() and kernel_out_tree() are different?
      del pallas_out_tree
      pallas_jaxpr, _, pallas_consts = (
          pe.trace_to_jaxpr_dynamic(flat_pallas_impl, in_avals))
    else:
      pallas_jaxpr = None
      pallas_consts = []

    # debug_info for rules
    out_flat = custom_fusion_p.bind(
        *consts,
        *pallas_consts,
        *args_flat,
        jaxpr=jaxpr,
        num_consts=len(consts),
        eval_rule=self.eval_rule,
        pull_block_spec_rule=self.pull_block_spec_rule,
        push_block_spec_rule=self.push_block_spec_rule,
        pallas_jaxpr=pallas_jaxpr,
        pallas_num_consts=len(pallas_consts),
        in_tree=in_tree,
        out_tree=out_tree(),
        kernel_out_tree=out_tree())

    return tree_util.tree_unflatten(out_tree(), out_flat)


@custom_fusion_p.def_impl
def _custom_fusion_impl(
    *args,
    jaxpr: core.Jaxpr,
    num_consts: int,
    pallas_num_consts: int,
    **_):
  consts, _, args = util.split_list(args, [num_consts, pallas_num_consts])  # type: ignore[assignment]
  return core.eval_jaxpr(jaxpr, consts, *args)

mlir.register_lowering(custom_fusion_p, mlir.lower_fun(
    _custom_fusion_impl, multiple_results=True))


@custom_fusion_p.def_effectful_abstract_eval
def _custom_fusion_effectful_abstract_eval(
    *args,
    jaxpr: core.Jaxpr,
    pallas_jaxpr: core.Jaxpr | None,
    **_):
  del args
  # TODO(jburnim): Error if pallas_jaxpr has different number of outputs, or
  # different shapes and types of outputs?
  if jaxpr.effects:
    raise NotImplementedError(
        "custom_fusion-decorated function {jaxpr.debug_info.func_src_info} "
        "has effects, which is not yet supported: {jaxpr.effects}")
  if pallas_jaxpr is not None and pallas_jaxpr.effects:
    raise NotImplementedError(
        "custom_fusion-decorated function {jaxpr.debug_info.func_src_info} "
        "has a pallas_impl with effects, which is not yet supported: "
        f"{pallas_jaxpr.effects}")
  return jaxpr.out_avals, jaxpr.effects


@block_spec_lib.register_eval_rule(custom_fusion_p)
def _custom_fusion_eval_rule(
    ctx: block_spec_lib.KernelEvalContext,
    *args,
    eval_rule: CustomEvalRuleFn,
    num_consts: int,
    pallas_num_consts: int,
    **_):
  args = args[num_consts + pallas_num_consts:]
  return eval_rule(CustomEvalContext(
      out_block_specs=ctx.out_block_specs,
      out_block_indices=ctx.get_out_block_indices(),
  ), *args)


# TODO(jburnim): Lowering rules for SC and Mosaic GPU.

@mosaic_lowering.register_lowering_rule(custom_fusion_p)
def _custom_fusion_mosaic_lowering_rule(
    ctx: mosaic_lowering.LoweringRuleContext,
    *args,
    jaxpr: core.Jaxpr,
    num_consts: int,
    pallas_jaxpr: core.Jaxpr | None,
    pallas_num_consts: int,
    **_):
  consts, pallas_consts, args = util.split_list(
      args, [num_consts, pallas_num_consts])
  if pallas_jaxpr is None:
    pallas_jaxpr = jaxpr
    pallas_consts = consts
  lowering_context = ctx.lowering_context.replace(block_shapes=ctx.block_shapes)
  return mosaic_lowering.jaxpr_subcomp(
      lowering_context, pallas_jaxpr, *pallas_consts, *args)


@block_spec_lib.register_pull_block_spec_rule(custom_fusion_p)  # type: ignore[arg-type]
def _custom_fusion_pull_block_spec_rule(
    ctx : block_spec_lib.PullRuleContext,
    out_block_specs : tuple[pallas_core.BlockSpec, ...],
    *,
    pull_block_spec_rule : CustomPullBlockSpecRuleFn,
    **_,
) -> Sequence[pallas_core.BlockSpec]:
  del ctx
  return pull_block_spec_rule(out_block_specs)


@block_spec_lib.register_push_block_spec_rule(custom_fusion_p)  # type: ignore[arg-type]
def _custom_fusion_push_block_spec_rule(
    ctx : block_spec_lib.PushRuleContext,
    *block_specs : pallas_core.BlockSpec,
    push_block_spec_rule : CustomPushBlockSpecRuleFn,
    **_
) -> tuple[pallas_core.BlockSpec, ...]:
  del ctx
  # TODO(jburnim): Better error message if push_block_spec_rule is None.
  return push_block_spec_rule(block_specs)


@block_spec_lib.register_usage_rule(custom_fusion_p)  # type: ignore[arg-type]
def _custom_fusion_usage_rule(
    ctx : block_spec_lib.UsageRuleContext,
    used_out: Sequence[set[block_spec_lib.Usage]],
    *,
    jaxpr: core.Jaxpr,
    **_
) -> Sequence[set[block_spec_lib.Usage]]:
  del ctx
  # TODO(jburnim): Error if jaxpr.jaxpr gives different usage than pallas_jaxpr?
  read_usage_env = block_spec_lib.compute_usage(jaxpr, used_out)
  return util.safe_map(read_usage_env, jaxpr.invars)
