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

from jax._src import core
from jax._src import linear_util as lu
from jax._src import api_util
from jax._src.util import safe_map, safe_zip
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

def scheduling_group(name):
  def wrap1(f):
    def wrap2(*args, **kwargs):
      args_flat, in_tree = tree_flatten((args, kwargs))
      dbg = api_util.debug_info('scheduling_group', f, args, kwargs)
      f_, out_tree = api_util.flatten_fun(lu.wrap_init(f, debug_info=dbg), in_tree)
      out_flat = scheduling_group_p.bind(f_, *args_flat, name=name)
      return tree_unflatten(out_tree(), out_flat)
    return wrap2
  return wrap1

scheduling_group_p = core.ClosedCallPrimitive('scheduling_group')
scheduling_group_p.def_impl(core.call_impl)
scheduling_group_p.def_effectful_abstract_eval(
    lambda *_, call_jaxpr: (call_jaxpr.out_avals, call_jaxpr.effects))

def _scheduling_group_lowering(
    ctx: mlir.LoweringRuleContext, *args, name: str, call_jaxpr: core.ClosedJaxpr):
  out_nodes, tokens = mlir.call_lowering(
      "scheduling_group_call", call_jaxpr, None, ctx.module_context,
      ctx.avals_in, ctx.avals_out, ctx.tokens_in, *args,
      dim_var_values=ctx.dim_var_values, const_lowering=ctx.const_lowering,
      attributes=dict(scheduling_group=ir.StringAttr.get(name)))
  ctx.set_tokens_out(tokens)
  return out_nodes
mlir.register_lowering(scheduling_group_p, _scheduling_group_lowering)

ad.primitive_transposes[scheduling_group_p] = ad.primitive_transposes[core.closed_call_p]
