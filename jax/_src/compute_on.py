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
from typing import Sequence

from jax._src import config
from jax._src.lib import xla_client
from jax._src import dispatch
from jax._src import core
from jax._src import linear_util as lu
from jax._src.interpreters import ad, batching, mlir, partial_eval as pe
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import (safe_map, safe_zip, weakref_lru_cache, unzip2,
                           split_list)
from jax._src.api_util import debug_info, flatten_fun_nokwargs, flatten_axes
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir import ir

config_ext = xla_client._xla.config
map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


@contextmanager
def extend_compute_type(c_type: str | None):
  if c_type is None:
    yield
    return

  prev = config.compute_on_context_manager.swap_local(c_type)
  try:
    if prev is not None and prev is not config_ext.unset and c_type != prev:
      raise NotImplementedError(
          'Nesting `compute_on` with different compute types is not supported'
          f' yet. Current compute_on type: {prev}')
    yield c_type
  finally:
    config.compute_on_context_manager.set_local(prev)


def _check_valid(c_type: str):
  if (c_type not in {'device_host', 'device', 'tpu_sparsecore'}
      and not c_type.startswith("gpu_stream:")):
    raise ValueError(
        f'Invalid compute type {c_type}. Current supported values '
        'are `device_host`, `device`, `tpu_sparsecore`, and `gpu_stream:#`.')

@contextmanager
def compute_on(compute_type: str):
  if not isinstance(compute_type, str):
    raise TypeError("`compute_on`'s compute_type argument must be a string.")
  _check_valid(compute_type)

  with extend_compute_type(compute_type):
    yield

def compute_on2(f=None, *, compute_type, out_memory_spaces):
  kwargs = dict(compute_type=compute_type, out_memory_spaces=out_memory_spaces)
  if f is None:
    return lambda g: _compute_on2(g, **kwargs)
  return _compute_on2(f, **kwargs)

def _compute_on2(f, *, compute_type, out_memory_spaces):
  def wrapped(*args):
    dbg = debug_info('compute_on', f, args, {})
    args_flat, in_tree = tree_flatten(args)
    in_avals = tuple(core.shaped_abstractify(x) for x in args_flat)
    jaxpr, out_tree = _trace_to_jaxpr(f, in_avals, in_tree, dbg)
    out_memory_spaces_flat = flatten_axes(
        "compute_on out_memory_spaces", out_tree, out_memory_spaces)
    outs_flat = compute_on_p.bind(
        *args_flat, jaxpr=jaxpr, compute_type=compute_type,
        out_memory_spaces=tuple(out_memory_spaces_flat))
    return tree_unflatten(out_tree, outs_flat)
  return wrapped

@weakref_lru_cache
def _trace_to_jaxpr(fun, in_avals, in_tree, dbg):
  f = lu.wrap_init(fun, debug_info=dbg)
  f, out_tree = flatten_fun_nokwargs(f, in_tree)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(f, in_avals)
  return core.ClosedJaxpr(jaxpr, consts), out_tree()

compute_on_p = core.Primitive('compute_on')
compute_on_p.multiple_results = True
dispatch.simple_impl(compute_on_p)


def _compute_on_abstract_eval(*in_avals, jaxpr, compute_type, out_memory_spaces):
  return [a.update(memory_space=s)
          for a, s in zip(jaxpr.out_avals, out_memory_spaces)]
compute_on_p.def_abstract_eval(_compute_on_abstract_eval)


def _compute_on_lowering(ctx, *args, jaxpr, compute_type, out_memory_spaces):
  const_args_and_avals = core.jaxpr_const_args(jaxpr.jaxpr)
  const_args, const_avals = unzip2(const_args_and_avals)
  const_arg_values = [
      mlir.ir_constant(c, const_lowering=ctx.const_lowering, aval=aval)
      for c, aval in const_args_and_avals]
  in_avals = (*const_avals, *ctx.avals_in)
  func_op, output_types, effects = mlir.lower_called_computation(
      "compute_on", jaxpr, ctx.module_context, len(const_args), in_avals,
      ctx.avals_out, ctx.tokens_in)

  symbol_name = func_op.name.value
  flat_output_types = mlir.flatten_ir_types(output_types)
  tokens = [ctx.tokens_in.get(eff) for eff in effects]
  args = (*ctx.dim_var_values, *tokens, *const_arg_values, *args)
  call = func_dialect.CallOp(
      flat_output_types, ir.FlatSymbolRefAttr.get(symbol_name),
      mlir.flatten_ir_values(args))

  if compute_type.startswith("gpu_stream:"):
    dict_attr = {
      "_xla_stream_annotation": ir.StringAttr.get(compute_type.split(":")[1]),
      "inlineable": ir.StringAttr.get("false"),
    }
  else:
    dict_attr = {
        "_xla_compute_type": ir.StringAttr.get(mlir.map_compute_type(compute_type))
    }
  call.operation.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(dict_attr)

  out_nodes = mlir.unflatten_ir_values_like_types(call.results, output_types)
  tokens, out_nodes = split_list(out_nodes, [len(effects)])
  tokens_out = ctx.tokens_in.update_tokens(mlir.TokenSet(zip(effects, tokens)))
  ctx.set_tokens_out(tokens_out)
  return [mlir.wrap_with_memory_kind(on, core.mem_space_to_kind(oms), out_aval)
          for on, out_aval, oms in zip(out_nodes, ctx.avals_out, out_memory_spaces)]

mlir.register_lowering(compute_on_p, _compute_on_lowering)


def _compute_on_batcher(axis_data, vals_in, dims_in, *, jaxpr, compute_type,
                        out_memory_spaces):
  batched_jaxpr, dims_out = batching.batch_jaxpr2(jaxpr, axis_data, dims_in)
  outs = compute_on_p.bind(*vals_in, jaxpr=batched_jaxpr,
                           compute_type=compute_type,
                           out_memory_spaces=out_memory_spaces)
  return outs, dims_out
batching.fancy_primitive_batchers[compute_on_p] = _compute_on_batcher


def _compute_on_jvp(primals, tangents, *, jaxpr, compute_type,
                    out_memory_spaces):
  nzs = [not isinstance(t, ad.Zero) for t in tangents]
  jaxpr_jvp, out_nzs = ad.jvp_jaxpr(jaxpr, nzs, False)
  nz_tangents = [t for t in tangents if not isinstance(t, ad.Zero)]
  spaces_jvp = (*out_memory_spaces,
                *[s for s, nz in zip(out_memory_spaces, out_nzs) if nz])
  outs = compute_on_p.bind(*primals, *nz_tangents, jaxpr=jaxpr_jvp,
                           compute_type=compute_type,
                           out_memory_spaces=spaces_jvp)
  primals_out, nz_tangents_out = outs[:len(out_nzs)], outs[len(out_nzs):]
  nz_outs = iter(nz_tangents_out)
  tangents_out = [next(nz_outs) if nz else ad.Zero(aval.to_tangent_aval())
                  for aval, nz in zip(jaxpr.out_avals, out_nzs)]
  assert next(nz_outs, None) is None
  return primals_out, tangents_out
ad.primitive_jvps[compute_on_p] = _compute_on_jvp


def _compute_on_lin(nzs, *primals, jaxpr, compute_type, out_memory_spaces):
  jaxpr_jvp, out_nzs = ad.jvp_jaxpr(jaxpr, nzs, False)
  lin_outs = [False] * len(out_nzs) + [True] * sum(out_nzs)
  jaxpr_lin_, used_inputs = pe.dce_jaxpr(jaxpr_jvp.jaxpr, lin_outs, False)
  jaxpr_lin = pe.close_jaxpr(jaxpr_lin_)
  spaces_lin = tuple(s for s, nz in zip(out_memory_spaces, out_nzs) if nz)
  primals_out = compute_on_p.bind(*primals, jaxpr=jaxpr,
                                  compute_type=compute_type,
                                  out_memory_spaces=out_memory_spaces)
  tangent_avals_out = [a.to_tangent_aval() for a in jaxpr.out_avals]

  def compute_on_lin(primals, *tangents):
    nz_tangents = [t for t in tangents if not isinstance(t, ad.Zero)]
    inputs = [x for x, u in zip([*primals, *nz_tangents], used_inputs) if u]
    nz_outs = compute_on_p.bind(*inputs, jaxpr=jaxpr_lin,
                                compute_type=compute_type,
                                out_memory_spaces=spaces_lin)
    nz_outs_ = iter(nz_outs)
    outs = [next(nz_outs_) if nz else ad.Zero(a)
            for nz, a in zip(out_nzs, tangent_avals_out)]
    assert next(nz_outs_, None) is None
    return outs
  return primals_out, out_nzs, primals, compute_on_lin
ad.primitive_linearizations[compute_on_p] = _compute_on_lin

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
def _transpose_jaxpr(jaxpr, in_avals, in_tree):
  cell = lambda: None
  def transposed(*in_flat):
    primals_in, cts_in = tree_unflatten(in_tree, in_flat)
    out = ad.backward_pass(jaxpr.jaxpr, False, jaxpr.consts, primals_in, cts_in)
    out = [ct if not isinstance(ct, ad.Zero) else None for ct in out]
    cts_out, cell.out_tree = tree_flatten(out)  # type: ignore
    return cts_out
  dbg = jaxpr.jaxpr.debug_info.with_unknown_names()
  trans_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(transposed, debug_info=dbg), in_avals)
  return core.ClosedJaxpr(trans_jaxpr, consts), cell.out_tree  # type: ignore

def _compute_on_transpose(cts_in, *primals_in, jaxpr, compute_type,
                          out_memory_spaces):
  in_flat, in_tree = tree_flatten((primals_in, cts_in))
  in_avals = tuple(core.typeof(x) for x in in_flat)
  trans_jaxpr, out_tree = _transpose_jaxpr(jaxpr, in_avals, in_tree)
  in_spaces = [x.aval.memory_space if isinstance(x, ad.UndefinedPrimal)
               else core.typeof(x).memory_space for x in primals_in]
  cts_out_ = tree_unflatten(out_tree, trans_jaxpr.out_avals)
  trans_spaces = tuple(s for x, s in zip(cts_out_, in_spaces) if x)
  cts_out = compute_on_p.bind(*in_flat, jaxpr=trans_jaxpr,
                              compute_type=compute_type,
                              out_memory_spaces=trans_spaces)
  return tree_unflatten(out_tree, cts_out)
ad.primitive_transposes[compute_on_p] = _compute_on_transpose
