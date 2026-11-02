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

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import inspect
from typing import Any

from jax._src import ad_util
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import effects
from jax._src import source_info_util
from jax._src.api_util import donation_vector, resolve_argnums
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import remat
from jax._src.lax import eval_jaxpr as eval_jaxpr_rules
from jax._src.lax.lax import create_token, optimization_barrier
from jax._src.pjit import jit_p
from jax._src.state import discharge
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten
from jax._src.util import (
    foreach, merge_lists, partition_list, safe_map, safe_zip, split_list,
    subs_list, wraps)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


program_order_p = core.Primitive("program_order")
program_order_p.multiple_results = True
program_order_p.skip_canonicalization = True


def program_order(f=None, *, enforce: bool,
                  strict_in_out: bool | tuple[bool, bool] = False,
                  exclude_argnames: str | Sequence[str] | None = None):
  if enforce and exclude_argnames is not None:
    raise ValueError("exclude_argnames cannot be used with enforce=True.")
  if not enforce and strict_in_out:
    raise ValueError("strict_in_out=True cannot be used with enforce=False")
  if isinstance(strict_in_out, bool):
    strict_in_out = (strict_in_out,) * 2
  strict_in, strict_out = strict_in_out
  kwargs = dict(enforce=enforce, strict_in=strict_in, strict_out=strict_out,
                exclude_argnames=exclude_argnames)
  if f is None:
    return lambda g: _program_order(g, **kwargs)
  return _program_order(f, **kwargs)


def _program_order(fun, *, enforce, strict_in, strict_out, exclude_argnames):
  @wraps(fun)
  def wrapped(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    if exclude_argnames is None:
      arg_exclude_mask = (False,) * len(args_flat)
    else:
      fun_signature = inspect.signature(fun)
      ex_argnums, ex_argnames, _, _ = resolve_argnums(
          fun, fun_signature, None, exclude_argnames, None, None)
      arg_exclude_mask = donation_vector(ex_argnums, ex_argnames, in_tree)
    assert len(args_flat) == len(arg_exclude_mask)
    traced = api.jit(fun).trace(*args, **kwargs)
    assert in_tree == traced.in_tree
    exclude_mask = (False,) * len(traced._consts) + arg_exclude_mask
    out_flat = program_order_p.bind(
        *traced._consts, *args_flat, call_jaxpr=traced.jaxpr,
        enforce=enforce, strict_in=strict_in, strict_out=strict_out,
        exclude_mask=exclude_mask)
    return tree_unflatten(traced.out_tree, out_flat)
  return wrapped


def opt_barrier_per_input(prev_outvars, prev_outs, cur_invars, cur_inps):
  token = create_token()
  token, prev_outs = optimization_barrier((token, prev_outs))
  prev_out_map = dict(zip(prev_outvars, prev_outs))
  # Tokens are not DCEd by opt_barrier even if they are unused.
  cur_inps = [prev_out_map[v] if v in prev_out_map
              else optimization_barrier((token, cinp))[1]
              for v, cinp in safe_zip(cur_invars, cur_inps)]
  return prev_outs, cur_inps


def insert_opt_barrier(prev_outvars, prev_outs, cur_invars, cur_inps):
  in_cur_invars = [v in cur_invars for v in prev_outvars]
  barrier_pouts, excluded_outs = partition_list(in_cur_invars, prev_outs)
  barrier_pouts, cur_inps = optimization_barrier((barrier_pouts, cur_inps))
  prev_outs = merge_lists(in_cur_invars, barrier_pouts, excluded_outs)
  return prev_outs, cur_inps


def eval_jaxpr_program_order(strict_in, strict_out, jaxpr, consts, *args):
  def read(v) -> Any:
    return v.val if isinstance(v, core.Literal) else env[v]

  def write(v, val: Any) -> None:
    if config.enable_checks.value:
      assert core.typecheck(v.aval, val), (v.aval, core.typeof(val), val)
    env[v] = val

  def eqn_write(eqn, ans):
    if eqn.primitive.multiple_results:
      foreach(write, eqn.outvars, ans)
    else:
      ans = ans[0] if isinstance(ans, list) else ans
      write(eqn.outvars[0], ans)

  env = {}
  foreach(write, jaxpr.constvars, consts)
  if strict_in:
    args = optimization_barrier(args)
  foreach(write, jaxpr.invars, args)
  last_used = core.last_used(jaxpr)
  prev_eqn = None
  for cur_eqn in jaxpr.eqns:
    bind_params = cur_eqn.primitive.get_bind_params(cur_eqn.params)
    name_stack = source_info_util.current_name_stack() + cur_eqn.source_info.name_stack
    traceback = cur_eqn.source_info.traceback
    with (source_info_util.user_context(traceback, name_stack=name_stack),
          cur_eqn.ctx.manager):
      cur_inps = map(read, cur_eqn.invars)
      if not cur_inps:  # nullary
        if prev_eqn is not None:
          token = create_token()
          prev_outs = map(read, prev_eqn.outvars)
          prev_outs, token = optimization_barrier((prev_outs, token))
          eqn_write(prev_eqn, prev_outs)
          ans = api.jit(lambda token: cur_eqn.primitive.bind(**bind_params),
                        inline=api.Inline.XLA_LATE)(token)
        else:
          ans = cur_eqn.primitive.bind(*cur_inps, **bind_params)
      else:
        if prev_eqn is not None:
          is_literal = [isinstance(i, core.Literal) for i in cur_eqn.invars]
          cur_invars, _ = partition_list(is_literal, cur_eqn.invars)
          cur_inps, literal_inps = partition_list(is_literal, cur_inps)
          prev_outs = map(read, prev_eqn.outvars)
          if (cur_eqn.primitive is program_order_p and
              not cur_eqn.params["enforce"]):
            exclude_mask = cur_eqn.params["exclude_mask"]
            barrier_inps, excluded_inps = partition_list(exclude_mask, cur_inps)
            if barrier_inps:
              barrier_invars, _ = partition_list(exclude_mask, cur_invars)
              prev_outs, barrier_inps = opt_barrier_per_input(
                  prev_eqn.outvars, prev_outs, barrier_invars, barrier_inps)
              cur_inps = merge_lists(exclude_mask, barrier_inps, excluded_inps)
          elif cur_eqn.primitive is jit_p:
            prev_outs, cur_inps = opt_barrier_per_input(
                prev_eqn.outvars, prev_outs, cur_invars, cur_inps)
          else:
            prev_outs, cur_inps = insert_opt_barrier(
                prev_eqn.outvars, prev_outs, cur_invars, cur_inps)
          eqn_write(prev_eqn, prev_outs)
          cur_inps = merge_lists(is_literal, cur_inps, literal_inps)
        ans = cur_eqn.primitive.bind(*cur_inps, **bind_params)
    eqn_write(cur_eqn, ans)
    prev_eqn = cur_eqn
    core.clean_up_dead_vars(cur_eqn, env, last_used)
  outvals = map(read, jaxpr.outvars)
  if strict_out:
    outvals = optimization_barrier(outvals)
  return outvals


# ----------------------------- rules ------------------------------------------

program_order_p.def_impl(core.eval_jaxpr_p.impl)
program_order_p.def_effectful_abstract_eval(core.eval_jaxpr_p.abstract_eval)

mlir.register_lowering(
    program_order_p,
    partial(mlir.core_call_lowering, name="program_order", inline_jax_late=True),
    cacheable=False)

batching.fancy_primitive_batchers[program_order_p] = partial(
    eval_jaxpr_rules.eval_jaxpr_batch, program_order_p)


def _program_order_is_high(*avals, call_jaxpr, enforce, **_) -> bool:
  return enforce or call_jaxpr.is_high
program_order_p.is_high = _program_order_is_high


def _program_order_to_lojax(*hi_args, call_jaxpr, enforce, strict_in,
                            strict_out, exclude_mask, **params):
  if enforce:
    call_jaxpr, _ = pe.dce_jaxpr(call_jaxpr, True, instantiate=True)
    return eval_jaxpr_program_order(
        strict_in, strict_out, call_jaxpr, call_jaxpr.consts, *hi_args)
  else:
    lo_jaxpr = pe.lower_jaxpr2(call_jaxpr)
    lo_args, lo_mask = [], []
    for aval, x, m in zip(call_jaxpr.in_avals, hi_args, exclude_mask):
      lo_vals = aval.lower_val(x)
      lo_args.extend(lo_vals)
      lo_mask.extend([m] * len(lo_vals))
    lo_outs = program_order_p.bind(
        *lo_args, call_jaxpr=lo_jaxpr, enforce=enforce, strict_in=strict_in,
        strict_out=strict_out, exclude_mask=tuple(lo_mask), **params)
    return pe.raise_lo_outs(call_jaxpr.out_avals, lo_outs)
program_order_p.to_lojax = _program_order_to_lojax


def _program_order_typecheck(ctx_factory, *in_atoms, call_jaxpr, exclude_mask,
                             **_):
  if len(exclude_mask) != len(in_atoms):
    raise core.JaxprTypeError(
        f"program_order has {len(in_atoms)} inputs but an exclude_mask of "
        f"length {len(exclude_mask)}")
  return core.custom_typechecks[core.eval_jaxpr_p](
      ctx_factory, *in_atoms, call_jaxpr=call_jaxpr)
core.custom_typechecks[program_order_p] = _program_order_typecheck


def _program_order_jvp(primals, tangents, *, call_jaxpr, exclude_mask,
                       **params):
  nzs = [type(t) is not ad_util.Zero for t in tangents]
  jaxpr_jvp, nzs_out = ad.jvp_jaxpr(call_jaxpr, nzs, False)
  nz_tangents = [t for t, nz in zip(tangents, nzs) if nz]
  tangents_mask = tuple(m for m, nz in zip(exclude_mask, nzs) if nz)
  outs = program_order_p.bind(
      *primals, *nz_tangents, call_jaxpr=jaxpr_jvp,
      exclude_mask=(*exclude_mask, *tangents_mask), **params)
  primals_out, nz_tangents_out = split_list(outs, [len(call_jaxpr.out_avals)])
  nz_tangents_out_ = iter(nz_tangents_out)
  tangents_out = [next(nz_tangents_out_) if nz else ad_util.Zero(a.to_tangent_aval())
                  for a, nz in zip(call_jaxpr.out_avals, nzs_out)]
  return primals_out, tangents_out
ad.primitive_jvps[program_order_p] = _program_order_jvp


def _program_order_linearize(is_vjp, nzs, *primals_in, call_jaxpr,
                             exclude_mask, **params):
  primal_jaxpr, out_tree, nzs_out, in_fwd_res, tangent_jaxpr = (
      ad.linearize_jaxpr(call_jaxpr, nzs, is_vjp=is_vjp))
  _, ures_avals, sres_avals = out_tree.unpack()
  num_res_out = len(ures_avals) + len(sres_avals)
  primals_and_res = program_order_p.bind(
      *primals_in, call_jaxpr=primal_jaxpr, exclude_mask=exclude_mask,
      **params)
  primals_out, non_fwd_res = split_list(
      primals_and_res, [len(primals_and_res) - num_res_out])
  ures, sres_flat = split_list(non_fwd_res, [len(ures_avals)])
  res = subs_list(in_fwd_res, [*call_jaxpr.consts, *primals_in], ures)
  sres = sres_avals.update(sres_flat).unflatten()
  tangents_mask = tuple(m for m, nz in zip(exclude_mask, nzs) if nz)

  def tangent_fun(res, sres, *tangents):
    sres_flat = tree_leaves(sres)
    nz_tangents = [ad.instantiate_zeros(x) for nz, x in zip(nzs, tangents) if nz]
    prim = ad.vjp_node(program_order_p) if is_vjp else program_order_p
    nz_tangents_out = prim.bind(
        *res, *nz_tangents, *sres_flat, call_jaxpr=tangent_jaxpr,
        exclude_mask=((False,) * len(res) + tangents_mask
                      + (False,) * len(sres_flat)),
        **params)
    tangent_avals_out = [v.aval.to_tangent_aval() for v in call_jaxpr.outvars]
    nz_tangents_out_ = iter(nz_tangents_out)
    tangents_out = [next(nz_tangents_out_) if nz else ad_util.Zero(aval)
                    for aval, nz in zip(tangent_avals_out, nzs_out)]
    assert next(nz_tangents_out_, None) is None
    return tangents_out
  return primals_out, nzs_out, res, sres, tangent_fun
ad.primitive_linearizations[program_order_p] = _program_order_linearize


def _program_order_transpose(ct, *args, call_jaxpr, exclude_mask, **params):
  primals_ctrefs, specs = ad.project_accums(args)
  # project_accums keeps the non-linear inputs, which keep their exclude_mask
  # entries, and the cotangent Refs, which aren't excluded.
  primals_ctrefs_mask = [m if kind is None else False
                         for (kind, _), m in zip(specs, exclude_mask)
                         if kind is None or kind is ad.RefAccum]
  assert len(tree_leaves(primals_ctrefs)) == len(primals_ctrefs_mask)
  in_flat, in_tree = tree_flatten((primals_ctrefs, ct))
  num_cts = len(in_flat) - len(primals_ctrefs_mask)
  in_avals = [core.typeof(x) for x in in_flat]
  trans_jaxpr, out_tree = eval_jaxpr_rules._transpose_jaxpr(
      call_jaxpr, in_tree, (*in_avals,), specs)
  outs = program_order_p.bind(
      *in_flat, call_jaxpr=trans_jaxpr,
      exclude_mask=(*primals_ctrefs_mask, *(False,) * num_cts), **params)
  cts_out, logs = tree_unflatten(out_tree, outs)
  for x, ct in zip(args, cts_out):
    if isinstance(x, ad.ValAccum):
      x.accum(ct)
  return logs
ad.fancy_transposes[program_order_p] = _program_order_transpose

def _program_order_partial_eval(trace, *in_tracers, call_jaxpr, exclude_mask,
                                **params):
  in_pvals = [t.pval for t in in_tracers]
  unknown_ins = [not pv.is_known() for pv in in_pvals]
  known_jaxpr, unknown_jaxpr, unknown_outs, res_avals, in_fwd_res = (
      pe.partial_eval_jaxpr_nounits_fwd(call_jaxpr, unknown_ins,
                                        instantiate=False))
  known_mask, unknown_mask = partition_list(unknown_ins, exclude_mask)
  consts = [pv.get_known() for pv in in_pvals if pv.is_known()]
  all_known_outs = program_order_p.bind(
      *consts, call_jaxpr=known_jaxpr, exclude_mask=tuple(known_mask),
      **params)
  known_outs, res = split_list(all_known_outs,
                               [len(all_known_outs) - len(res_avals)])
  res_ = iter(res)
  res = [next(res_) if f is None else [*call_jaxpr.consts, *consts][f]
         for f in in_fwd_res]
  assert next(res_, sentinel := object()) is sentinel
  res_tracers = map(trace.new_instantiated_const, res)
  unk_tracers_in = [t for t in in_tracers if not t.pval.is_known()]
  unk_tracers_out = [pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
                     for aval in unknown_jaxpr.out_avals]
  staged_params = dict(
      params, call_jaxpr=unknown_jaxpr,
      exclude_mask=(False,) * len(res_tracers) + tuple(unknown_mask))
  eqn = pe.new_eqn_recipe(trace, [*res_tracers, *unk_tracers_in],
                          unk_tracers_out, program_order_p, staged_params,
                          core.positional_effects(unknown_jaxpr),
                          source_info_util.current())
  for t in unk_tracers_out:
    t.recipe = eqn
  if effects.partial_eval_kept_effects.filter_in(unknown_jaxpr.effects):
    trace.effect_handles.append(
        pe.EffectHandle([*unk_tracers_in, *res_tracers], eqn))
  return merge_lists(unknown_outs, known_outs, unk_tracers_out)
pe.custom_partial_eval_rules[program_order_p] = _program_order_partial_eval


def _program_order_partial_eval_custom_params_updater(
    unks_in, inst_in, kept_outs_known, kept_outs_staged, num_res_out,
    num_res_in, params_known, params_staged):
  del kept_outs_known, kept_outs_staged, num_res_out  # unused
  known_mask, _ = partition_list(unks_in, params_known["exclude_mask"])
  # The known block can also take residual Refs as trailing inputs.
  num_res_refs = len(params_known["call_jaxpr"].invars) - len(known_mask)
  params_known = dict(params_known,
                      exclude_mask=(*known_mask, *(False,) * num_res_refs))
  _, staged_mask = partition_list(inst_in, params_staged["exclude_mask"])
  params_staged = dict(params_staged,
                       exclude_mask=(*(False,) * num_res_in, *staged_mask))
  return params_known, params_staged
pe.partial_eval_jaxpr_custom_rules[program_order_p] = partial(
    pe.closed_call_partial_eval_custom_rule, "call_jaxpr",
    _program_order_partial_eval_custom_params_updater)


def _program_order_dce(used_outputs, live_ins, eqn):
  used_inputs, new_eqn = pe.dce_jaxpr_closed_call_rule(
      used_outputs, live_ins, eqn)
  if new_eqn is eqn:
    return used_inputs, eqn
  if new_eqn is not None:
    exclude_mask = tuple(m for m, used in
                         zip(eqn.params["exclude_mask"], used_inputs) if used)
    new_eqn = new_eqn.replace(params=dict(new_eqn.params,
                                          exclude_mask=exclude_mask))
  return used_inputs, new_eqn
pe.dce_rules[program_order_p] = _program_order_dce


def _program_order_remat(trace, *args, call_jaxpr, exclude_mask, **params):
  jaxpr_fwd, jaxpr_rem, fwds = remat.remat_jaxpr(
      call_jaxpr, trace.policy, trace.custom_vjp_rules, allow_fwds=True)
  primals_res_out = program_order_p.bind(
      *args, call_jaxpr=jaxpr_fwd, exclude_mask=exclude_mask, **params)
  primals_out, res = split_list(primals_res_out, [len(call_jaxpr.outvars)])
  res_ = iter(res)
  res_full = [primals_out[f] if f is not None else next(res_) for f in fwds]
  assert next(res_, None) is None
  rem_mask = (False,) * len(res_full) + exclude_mask
  rem = lambda res_full, *args: program_order_p.bind(
      *res_full, *args, call_jaxpr=jaxpr_rem, exclude_mask=rem_mask, **params)
  return primals_out, res_full, rem
remat.rules[program_order_p] = _program_order_remat

discharge.register_discharge_rule(program_order_p)(
    partial(discharge._eval_jaxpr_discharge_rule, program_order_p))
