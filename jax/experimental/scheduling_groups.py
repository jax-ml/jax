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

from typing import Any
from functools import partial

from jax._src import core
from jax._src import linear_util as lu
from jax._src import api_util
from jax._src.util import safe_map, safe_zip
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.interpreters import ad, mlir, partial_eval as pe
from jax._src.lib.mlir import ir

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

def scheduling_group(name):
  return xla_metadata_call(scheduling_group=name)

def xla_metadata_call(f=None, **meta):
  if f is None:
    return lambda g: _xla_metadata_call(g, **meta)
  return _xla_metadata_call(f, **meta)

def _xla_metadata_call(f, **meta):
  def wrapped(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    dbg = api_util.debug_info('xla_metadata_call', f, args, kwargs)
    f_, out_tree = api_util.flatten_fun(lu.wrap_init(f, debug_info=dbg), in_tree)
    out_flat = xla_metadata_call_p.bind(f_, *args_flat, meta=tuple(meta.items()))
    return tree_unflatten(out_tree(), out_flat)
  return wrapped

xla_metadata_call_p = core.ClosedCallPrimitive('xla_metadata_call')
xla_metadata_call_p.def_impl(core.call_impl)
xla_metadata_call_p.def_effectful_abstract_eval(
    lambda *_, call_jaxpr: (call_jaxpr.out_avals, call_jaxpr.effects))

def _xla_metadata_call_lowering(
    ctx: mlir.LoweringRuleContext, *args, meta: tuple[tuple[str, Any], ...],
    call_jaxpr: core.ClosedJaxpr):
  out_nodes, tokens = mlir.call_lowering(
      "xla_metadata_call", call_jaxpr, None, ctx.module_context,
      ctx.avals_in, ctx.avals_out, ctx.tokens_in, *args,
      dim_var_values=ctx.dim_var_values, const_lowering=ctx.const_lowering,
      attributes={k: attr_get(v) for k, v in meta})
  ctx.set_tokens_out(tokens)
  return out_nodes
mlir.register_lowering(xla_metadata_call_p, _xla_metadata_call_lowering)

def attr_get(x):
  if isinstance(x, str):
    return ir.StringAttr.get(x)
  else:
    raise NotImplementedError(f'mlir attr handler for {type(x)=}')

pe.partial_eval_jaxpr_custom_rules[xla_metadata_call_p] = \
    partial(pe.closed_call_partial_eval_custom_rule, 'call_jaxpr',
            lambda _, __, ___, ____, _____, ______, x, y: (x, y))

def _xla_metadata_call_transpose(params, jaxpr, args, ct, cts_in_avals):
  jaxpr_, consts = jaxpr.jaxpr, jaxpr.consts
  jaxpr_ = pe.convert_constvars_jaxpr(jaxpr_)
  return ad.call_transpose(
      xla_metadata_call_p, params, jaxpr_, (*consts, *args),
      ct, cts_in_avals)
ad.primitive_transposes[xla_metadata_call_p] = _xla_metadata_call_transpose

pe.dce_rules[xla_metadata_call_p] = pe.dce_jaxpr_closed_call_rule
