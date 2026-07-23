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

from __future__ import annotations
from contextlib import contextmanager
from functools import partial
from collections.abc import Sequence
import json

from jax._src import config
from jax._src.lib import _jax
from jax._src import dispatch
from jax._src import core
from jax._src import effects as effects_lib
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src.interpreters import ad, batching, mlir, partial_eval as pe
from jax._src import flattree as ft
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten
from jax._src.xla_metadata import _check_no_qdd
from jax._src.util import (safe_map, safe_zip, weakref_lru_cache, unzip2,
                           split_list, subs_list, merge_lists)
from jax._src.api_util import debug_info, flatten_fun_nokwargs, flatten_axes
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir import ir

config_ext = _jax.config
map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


@contextmanager
def extend_compute_type(c_type: str | None):
  if c_type is None:
    yield
    return

  prev = config.compute_on_context_manager.swap_local(c_type)
  try:
    yield c_type
  finally:
    config.compute_on_context_manager.set_local(prev)


@contextmanager
def compute_on(compute_type: str):
  if not isinstance(compute_type, str):
    raise TypeError("`compute_on`'s compute_type argument must be a string.")
  _check_valid(compute_type)

  with extend_compute_type(compute_type):
    yield


def _check_valid(c_type: str):
  if (c_type not in {'device_host', 'device', 'tpu_sparsecore'}
      and not c_type.startswith("gpu_stream:")):
    raise ValueError(
        f'Invalid compute type {c_type}. Current supported values '
        'are `device_host`, `device`, `tpu_sparsecore`, and `gpu_stream:#`.')


def compute_on2(f=None, *, compute_type, out_memory_spaces,
                compiler_options=None):
  kwargs = dict(compute_type=compute_type, out_memory_spaces=out_memory_spaces,
                compiler_options=compiler_options)
  if f is None:
    return lambda g: _compute_on2(g, **kwargs)
  return _compute_on2(f, **kwargs)

def _compute_on2(f, *, compute_type, out_memory_spaces, compiler_options):
  if not isinstance(compute_type, str):
    raise TypeError("`compute_on`'s compute_type argument must be a string.")
  _check_valid(compute_type)

  def wrapped(*args):
    nonlocal compiler_options
    dbg = debug_info('compute_on', f, args, {})
    args_flat, in_tree = tree_flatten(args)
    in_avals = tuple(core.shaped_abstractify(x) for x in args_flat)
    with extend_compute_type(compute_type):
      jaxpr, out_tree = _trace_to_jaxpr(f, in_avals, in_tree, dbg)
      if any(isinstance(c, core.Tracer) for c in jaxpr.consts):
        jaxpr, consts = pe.separate_consts(jaxpr)
      else:
        consts = []
    out_memory_spaces_flat = flatten_axes(
        "compute_on out_memory_spaces", out_tree, out_memory_spaces)
    if compute_type == 'tpu_sparsecore' and compiler_options is not None:
      sc_config = compiler_options.get('sparse_core_config')
      if isinstance(sc_config, dict) and 'core_ids' in sc_config:
        compiler_options = {
            **compiler_options,
            'sparse_core_config': {**sc_config, 'core_id_mutability': False},
        }

    compiler_options_json = (
        None if compiler_options is None else json.dumps(compiler_options)
    )
    outs_flat = compute_on_p.bind(
        *consts, *args_flat, jaxpr=jaxpr, compute_type=compute_type,
        out_memory_spaces=tuple(out_memory_spaces_flat),
        compiler_options_json=compiler_options_json)
    return tree_unflatten(out_tree, outs_flat)
  return wrapped

@weakref_lru_cache
def _trace_to_jaxpr(fun, in_avals, in_tree, dbg):
  f = lu.wrap_init(fun, debug_info=dbg)
  f, out_tree = flatten_fun_nokwargs(f, in_tree)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(f, in_avals)
  return jaxpr.with_consts(consts), out_tree()

compute_on_p = core.Primitive('compute_on')
compute_on_p.multiple_results = True
dispatch.simple_impl(compute_on_p)


def _compute_on_abstract_eval(*in_avals, jaxpr, compute_type, out_memory_spaces,
                              compiler_options_json):
  out_avals = [a.update(memory_space=s)
               for a, s in zip(jaxpr.out_avals, out_memory_spaces)]
  return out_avals, core.positional_effects(jaxpr)
compute_on_p.def_effectful_abstract_eval(_compute_on_abstract_eval)


def _compute_on_lowering(ctx, *args, jaxpr, compute_type, out_memory_spaces,
                         compiler_options_json):
  if dispatch.jaxpr_has_primitive(jaxpr, 'compute_on'):
    raise ValueError("Nesting `compute_on` with different compute types is "
                     "not allowed.")
  const_args_and_avals = core.jaxpr_const_args(jaxpr)
  const_args, const_avals = unzip2(const_args_and_avals)
  const_arg_values = [
      mlir.ir_constants(c, const_lowering=ctx.const_lowering, aval=aval)
      for c, aval in const_args_and_avals]
  in_avals = (*const_avals, *ctx.avals_in)
  func_op, output_types, effects = mlir.lower_called_computation(
      "compute_on", jaxpr, ctx.module_context, len(const_args), in_avals,
      ctx.avals_out, ctx.tokens_in)

  symbol_name = func_op.name.value
  flat_output_types, treedef = mlir.ir_tree_registry.flatten(output_types)
  tokens = [ctx.tokens_in.get(eff) for eff in effects]
  args = (*ctx.dim_var_values, *tokens, *const_arg_values, *args)
  flat_args, _ = mlir.ir_tree_registry.flatten(args)
  call = func_dialect.CallOp(
      flat_output_types, ir.FlatSymbolRefAttr.get(symbol_name),
      flat_args)

  if compute_type.startswith("gpu_stream:"):
    dict_attr = {
        "_xla_stream_annotation": ir.StringAttr.get(compute_type.split(":")[1]),
        "inlineable": ir.StringAttr.get("false"),
    }
  else:
    ctype = mlir.map_compute_type(compute_type)
    dict_attr = {"_xla_compute_type": ir.StringAttr.get(ctype)}

  if compiler_options_json is not None:
    dict_attr |= {'backend_config': ir.StringAttr.get(compiler_options_json)}
  elif compute_type == 'device':
    dict_attr |= {'inlineable': ir.StringAttr.get('false')}

  call.operation.attributes['mhlo.frontend_attributes'] = ir.DictAttr.get(dict_attr)  # type: ignore

  out_nodes = treedef.unflatten(call.results)
  tokens, out_nodes = split_list(out_nodes, [len(effects)])
  tokens_out = ctx.tokens_in.update_tokens(mlir.TokenSet(dict(zip(effects, tokens))))
  ctx.set_tokens_out(tokens_out)
  return [
      mlir.wrap_with_memory_kind(ctx.module_context, on, core.mem_space_to_kind(oms), out_aval)
      for on, out_aval, oms in zip(out_nodes, ctx.avals_out, out_memory_spaces)
  ]

mlir.register_lowering(compute_on_p, _compute_on_lowering)


def _compute_on_batcher(axis_data, vals_in, dims_in, *, jaxpr, compute_type,
                        out_memory_spaces, compiler_options_json):
  batched_jaxpr, dims_out = batching.batch_jaxpr2(jaxpr, axis_data, dims_in)
  outs = compute_on_p.bind(*vals_in, jaxpr=batched_jaxpr,
                           compute_type=compute_type,
                           out_memory_spaces=out_memory_spaces,
                           compiler_options_json=compiler_options_json)
  return outs, dims_out
batching.fancy_primitive_batchers[compute_on_p] = _compute_on_batcher


def _compute_on_jvp(primals, tangents, *, jaxpr, compute_type,
                    out_memory_spaces, compiler_options_json):
  nzs = [not isinstance(t, ad.Zero) for t in tangents]
  jaxpr_jvp, out_nzs = ad.jvp_jaxpr(jaxpr, nzs, False)
  nz_tangents = [t for t in tangents if not isinstance(t, ad.Zero)]
  spaces_jvp = (*out_memory_spaces,
                *[s for s, nz in zip(out_memory_spaces, out_nzs) if nz])
  outs = compute_on_p.bind(*primals, *nz_tangents, jaxpr=jaxpr_jvp,
                           compute_type=compute_type,
                           out_memory_spaces=spaces_jvp,
                           compiler_options_json=compiler_options_json)
  primals_out, nz_tangents_out = outs[:len(out_nzs)], outs[len(out_nzs):]
  nz_outs = iter(nz_tangents_out)
  tangents_out = [next(nz_outs) if nz else ad.Zero(aval.to_tangent_aval())
                  for aval, nz in zip(jaxpr.out_avals, out_nzs)]
  assert next(nz_outs, None) is None
  return primals_out, tangents_out
ad.primitive_jvps[compute_on_p] = _compute_on_jvp


def _compute_on_lin(is_vjp, nzs, *primals, jaxpr, compute_type,
                    out_memory_spaces, compiler_options_json):
  primal_jaxpr, out_tree, nzs_out, in_fwd_res, tangent_jaxpr = \
      ad.linearize_jaxpr(jaxpr, nzs, is_vjp=is_vjp)
  _, ures_avals, sres_avals = out_tree.unpack()
  num_res_out = len(ures_avals) + len(sres_avals)
  num_primals_out = len(primal_jaxpr.out_avals) - num_res_out

  _, in_fwd_ures, in_fwd_sres = split_list(
      pe._jaxpr_forwarding(primal_jaxpr), [num_primals_out, len(ures_avals)])
  assert all(f is None for f in in_fwd_ures)
  in_fwd = [None] * (num_primals_out + len(ures_avals)) + in_fwd_sres
  primal_jaxpr = pe.prune_closed_jaxpr_outputs(
      primal_jaxpr, [f is None for f in in_fwd])
  primal_jaxpr, out_fwd = pe.dedup_jaxpr_outputs(primal_jaxpr, num_primals_out)
  num_kept_res = sum(f is None for f in out_fwd) - num_primals_out

  tangent_avals_out = [a.to_tangent_aval() for a in jaxpr.out_avals]
  def _filter_zeros(is_nz_l, l):
    return tuple(x for nz, x in zip(is_nz_l, l) if nz)

  def tangent_fun(residuals, structured_residuals, *tangents):
    tangents_nz = _filter_zeros(nzs, tangents)
    sres_flat = tree_leaves(structured_residuals)
    assert (len(residuals) + len(tangents_nz) + len(sres_flat)
            == len(tangent_jaxpr.invars)), (
        len(residuals), len(tangents_nz), len(sres_flat),
        len(tangent_jaxpr.invars))
    tangent_out_mem_spaces = _filter_zeros(nzs_out, out_memory_spaces)
    nz_outs = compute_on_p.bind(*residuals, *tangents_nz, *sres_flat,
                                jaxpr=tangent_jaxpr, compute_type=compute_type,
                                out_memory_spaces=tangent_out_mem_spaces,
                                compiler_options_json=compiler_options_json)
    nz_outs_ = iter(nz_outs)
    outs = [next(nz_outs_) if nz else ad.Zero(a)
            for nz, a in zip(nzs_out, tangent_avals_out)]
    assert next(nz_outs_, None) is None
    return outs

  primal_out_mem_spaces = out_memory_spaces + (core.MemorySpace.Device,) * num_kept_res
  ans = compute_on_p.bind(*primals, jaxpr=primal_jaxpr, compute_type=compute_type,
                          out_memory_spaces=primal_out_mem_spaces,
                          compiler_options_json=compiler_options_json)
  ans = subs_list(out_fwd, ans, ans)
  ans = subs_list(in_fwd, primals, ans)
  primal_ans, residuals_ans = split_list(ans, [len(ans) - num_res_out])
  ures, sres_flat = split_list(residuals_ans, [len(ures_avals)])
  ures = subs_list(in_fwd_res, [*jaxpr.consts, *primals], ures)
  sres = sres_avals.update(sres_flat).unflatten()
  return primal_ans, nzs_out, ures, sres, tangent_fun
ad.primitive_linearizations[compute_on_p] = _compute_on_lin

def _compute_on_partial_eval(trace: pe.JaxprTrace, *in_tracers, jaxpr,
                             compute_type, out_memory_spaces,
                             compiler_options_json):
  in_pvals = [t.pval for t in in_tracers]
  known_ins = tuple(pv.is_known() for pv in in_pvals)
  unknown_ins = tuple(not k for k in known_ins)

  (known_jaxpr, unknown_jaxpr, unknown_outs, res_out_avals,
   in_fwd_res) = pe.partial_eval_jaxpr_nounits_fwd(
       jaxpr, unknown_ins, instantiate=False)
  unknown_outs = tuple(unknown_outs)
  known_outs = tuple(not uk for uk in unknown_outs)

  def keep_where(l, should_keep):
    return tuple(x for x, keep in zip(l, should_keep) if keep)

  known_out_memory_spaces = (keep_where(out_memory_spaces, known_outs)
                             + (core.MemorySpace.Device,) * len(res_out_avals))
  known_params = dict(jaxpr=known_jaxpr, compute_type=compute_type,
                      out_memory_spaces=known_out_memory_spaces,
                      compiler_options_json=compiler_options_json)

  known_inputs = [pv.get_known() for pv in in_pvals if pv.is_known()]
  all_known_outs = compute_on_p.bind(*known_inputs, **known_params)

  known_out_vals, residual_vals = split_list(
      all_known_outs, [len(all_known_outs) - len(res_out_avals)])
  residual_vals_ = iter(residual_vals)
  residual_vals = [next(residual_vals_) if f is None
                   else [*jaxpr.consts, *known_inputs][f] for f in in_fwd_res]
  assert next(residual_vals_, None) is None
  residual_tracers = map(trace.new_instantiated_const, residual_vals)

  unknown_params = dict(
      jaxpr=unknown_jaxpr, compute_type=compute_type,
      out_memory_spaces=keep_where(out_memory_spaces, unknown_outs),
      compiler_options_json=compiler_options_json)

  unknown_tracers_in = [*residual_tracers,
                        *(t for t in in_tracers if not t.pval.is_known())]
  unknown_out_avals = unknown_jaxpr.out_avals
  unknown_tracers_out = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
      for aval in unknown_out_avals
  ]
  eqn = pe.new_eqn_recipe(trace, unknown_tracers_in,
                          unknown_tracers_out,
                          compute_on_p,
                          unknown_params,
                          core.positional_effects(unknown_jaxpr),
                          source_info_util.current())
  for t in unknown_tracers_out: t.recipe = eqn
  if effects_lib.partial_eval_kept_effects.filter_in(unknown_jaxpr.effects):
    trace.effect_handles.append(pe.EffectHandle(unknown_tracers_in, eqn))
  return merge_lists(unknown_outs, known_out_vals, unknown_tracers_out)
pe.custom_partial_eval_rules[compute_on_p] = _compute_on_partial_eval

def _compute_on_partial_eval_custom_params_updater(
    unks_in: Sequence[bool], inst_in: Sequence[bool],
    kept_outs_known: Sequence[bool], kept_outs_staged: Sequence[bool],
    num_res_out: int, num_res_in: int, params_known, params_staged):
  # prune inputs to jaxpr_known according to unks_in
  _, out_memory_spaces_known = pe.partition_list(
      kept_outs_known, params_known['out_memory_spaces'])
  new_params_known = dict(
      params_known,
      out_memory_spaces=(*out_memory_spaces_known,
                         *[core.MemorySpace.Device] * num_res_out),
  )
  assert (len(new_params_known['out_memory_spaces']) ==
          len(params_known['jaxpr'].out_avals))

  # added num_res new inputs to jaxpr_staged, and pruning according to inst_in
  _, out_memory_spaces_staged = pe.partition_list(
      kept_outs_staged, params_staged['out_memory_spaces'])
  new_params_staged = dict(
      params_staged,
      out_memory_spaces=tuple(out_memory_spaces_staged),
  )
  assert (len(new_params_staged['out_memory_spaces']) ==
          len(params_staged['jaxpr'].out_avals))
  return new_params_known, new_params_staged

pe.partial_eval_jaxpr_custom_rules[compute_on_p] = \
    partial(pe.closed_call_partial_eval_custom_rule, 'jaxpr',
            _compute_on_partial_eval_custom_params_updater)

@weakref_lru_cache
@weakref_lru_cache
def _transpose_jaxpr(jaxpr, in_tree, in_avals, specs):
  cell = lambda: None
  def transposed(*in_flat):
    primals_ctrefs, cts_in = tree_unflatten(in_tree, in_flat)
    args = ad.unproject_accums(specs, primals_ctrefs)
    logs = ad.backward_pass3(jaxpr, False, jaxpr.consts, args, cts_in)
    cts_out = [x.freeze() if isinstance(x, ad.ValAccum) else None for x in args]
    outs, cell.out_tree = tree_flatten((cts_out, logs))  # pyrefly: ignore[missing-attribute]
    return outs
  dbg = jaxpr.debug_info.with_unknown_names()
  trans_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(transposed, debug_info=dbg), in_avals)
  return trans_jaxpr.with_consts(consts), cell.out_tree  # pyrefly: ignore[missing-attribute]

def _compute_on_transpose(cts_in, *args, jaxpr, compute_type,
                          out_memory_spaces, compiler_options_json):
  primals_ctrefs, specs = ad.project_accums(args)
  in_flat, in_tree = tree_flatten((primals_ctrefs, cts_in))
  in_avals = tuple(core.typeof(x) for x in in_flat)
  trans_jaxpr, out_tree = _transpose_jaxpr(jaxpr, in_tree, in_avals, specs)
  cts_out_, logs_ = tree_unflatten(out_tree, trans_jaxpr.out_avals)
  arg_spaces = [x.aval.memory_space if isinstance(x, ad.GradAccum)  # type: ignore
                else core.typeof(x).memory_space for x in args]
  trans_spaces = tuple(s for x, s in zip(cts_out_, arg_spaces)
                       if isinstance(x, core.AbstractValue))
  trans_spaces += tuple(a.memory_space for a in tree_leaves(logs_))
  outs = compute_on_p.bind(*in_flat, jaxpr=trans_jaxpr,
                           compute_type=compute_type,
                           out_memory_spaces=trans_spaces,
                           compiler_options_json=compiler_options_json)
  cts_out, logs = tree_unflatten(out_tree, outs)
  for x, ct in zip(args, cts_out):
    if isinstance(x, ad.ValAccum):
      x.accum(ct)
  return logs
ad.fancy_transposes[compute_on_p] = _compute_on_transpose


def _compute_on_to_lojax(*hi_args, jaxpr, compute_type, out_memory_spaces,
                         compiler_options_json):
  _check_no_qdd(jaxpr, 'compute_on')
  lo_args_lol = [a.lower_val(x) for a, x in zip(jaxpr.in_avals, hi_args)]
  lo_args = [x for xs in lo_args_lol for x in xs]
  in_avals = ft.flatten(([[core.typeof(x) for x in xs] for xs in lo_args_lol],
                         {}))
  lo_jaxpr, out_avals = pe.lower_jaxpr(jaxpr, in_avals)
  _, out_lol = out_avals.unpack()
  lo_spaces = tuple(s for l, s in zip(out_lol.unpack(), out_memory_spaces)
                    for _ in l)
  all_outs = compute_on_p.bind(*lo_args, jaxpr=lo_jaxpr,
                               compute_type=compute_type,
                               out_memory_spaces=lo_spaces,
                               compiler_options_json=compiler_options_json)
  _, lo_outs = out_avals.update(all_outs).unpack()
  return [a.raise_val2(y) for a, y in zip(jaxpr.out_avals, lo_outs.unpack())]
compute_on_p.to_lojax = _compute_on_to_lojax

def dce_jaxpr_compute_on_rule(used_outputs: list[bool], eqn: pe.JaxprEqn
                              ) -> tuple[list[bool], pe.JaxprEqn | None]:
  if not any(used_outputs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None

  dced_jaxpr, used_inputs = pe._cached_closed_call_dce(
      eqn.params['jaxpr'], tuple(used_outputs))

  def keep_where(xs, keeps):
    return tuple(x for x, keep in zip(xs, keeps) if keep)

  new_params = dict(eqn.params, jaxpr=dced_jaxpr,
                    out_memory_spaces=keep_where(eqn.params["out_memory_spaces"],
                                                 used_outputs))
  if not any(used_inputs) and not any(used_outputs) and not dced_jaxpr.effects:
    return used_inputs, None
  else:
    new_invars = [v for v, used in zip(eqn.invars, used_inputs) if used]
    new_effs = core.eqn_effects(dced_jaxpr, new_invars)
    new_eqn = pe.new_jaxpr_eqn(
        new_invars,
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, new_effs, eqn.source_info, eqn.ctx)
    return used_inputs, new_eqn
pe.dce_rules[compute_on_p] = dce_jaxpr_compute_on_rule
