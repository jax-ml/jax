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
"""Transformation rules for the eval_jaxpr primitive.

The primitive itself is defined in partial_eval.py; here we register the rules
that depend on the ad and batching machinery.
"""

from functools import partial

from jax._src import ad_util
from jax._src import core
from jax._src import flattree as ft
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters.partial_eval import eval_jaxpr_p
from jax._src.state import discharge
from jax._src.tree_util import tree_leaves, tree_flatten, tree_unflatten
from jax._src.util import safe_map, safe_zip, split_list, subs_list, weakref_lru_cache

_map = safe_map
zip = safe_zip


def _eval_jaxpr_jvp(prim, primals, tangents, *, call_jaxpr, **params):
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  jaxpr_jvp, nonzeros_out = ad.jvp_jaxpr(call_jaxpr, nonzeros, False)
  nz_tangents = [t for t, nz in zip(tangents, nonzeros) if nz]
  outs = prim.bind(*primals, *nz_tangents, call_jaxpr=jaxpr_jvp, **params)
  primals_out, tangents_out = split_list(outs, [len(call_jaxpr.out_avals)])
  nz_tangents_out = iter(tangents_out)
  tangents_out = [next(nz_tangents_out) if nz else ad_util.Zero(aval.to_tangent_aval())
                  for aval, nz in zip(call_jaxpr.out_avals, nonzeros_out)]
  return primals_out, tangents_out

def _eval_jaxpr_batch(prim, axis_data, args, dims, *, call_jaxpr, **params):
  batched = [d is not None for d in dims]
  new_jaxpr, out_batched = batching.batch_jaxpr(call_jaxpr, axis_data, batched, False)
  new_args = [batching.moveaxis(x, d, 0) if d is not None and d != 0 else x
              for x, d in zip(args, dims)]
  outs = prim.bind(*new_args, call_jaxpr=new_jaxpr, **params)
  out_dims = [0 if b else None for b in out_batched]
  return outs, out_dims

def _eval_jaxpr_linearize(prim, is_vjp, nzs, *primals_in, call_jaxpr, **params):
  primal_jaxpr, out_tree, nzs_out, in_fwd_res, tangent_jaxpr = \
      ad.linearize_jaxpr(call_jaxpr, nzs, is_vjp=is_vjp)
  _, ures_avals, sres_avals = out_tree.unpack()
  num_res_out = len(ures_avals) + len(sres_avals)
  primals_and_res = prim.bind(*primals_in, call_jaxpr=primal_jaxpr, **params)
  primals_out, non_fwd_res = split_list(
      primals_and_res, [len(primals_and_res) - num_res_out])
  ures, sres_flat = split_list(non_fwd_res, [len(ures_avals)])
  res = subs_list(in_fwd_res, [*call_jaxpr.consts, *primals_in], ures)
  sres = sres_avals.update(sres_flat).unflatten()

  def tangent_fun(res, sres, *tangents):
    sres_flat = tree_leaves(sres)
    nz_tangents = [ad.instantiate_zeros(x) for nz, x in zip(nzs, tangents) if nz]
    nz_tangents_out = prim.bind(*res, *nz_tangents, *sres_flat,
                                call_jaxpr=tangent_jaxpr, **params)
    tangent_avals_out = [v.aval.to_tangent_aval() for v in call_jaxpr.outvars]
    nz_tangents_out_ = iter(nz_tangents_out)
    tangents_out = [next(nz_tangents_out_) if nz else ad_util.Zero(aval)
                    for aval, nz in zip(tangent_avals_out, nzs_out)]
    assert next(nz_tangents_out_, None) is None
    return tangents_out

  return primals_out, nzs_out, res, sres, tangent_fun

def _eval_jaxpr_transpose(prim, ct, *args, call_jaxpr, **params):
  primals_ctrefs, specs = ad.project_accums(args)
  in_flat, in_tree = tree_flatten((primals_ctrefs, ct))
  in_avals = [core.typeof(x) for x in in_flat]
  trans_jaxpr, out_tree = _transpose_jaxpr(call_jaxpr, in_tree, (*in_avals,), specs)
  outs = prim.bind(*in_flat, call_jaxpr=trans_jaxpr, **params)
  cts_out, logs = tree_unflatten(out_tree, outs)
  for x, ct in zip(args, cts_out):
    if isinstance(x, ad.ValAccum):
      x.accum(ct)
  return logs

# TODO(mattjj): this is a copy of xla_metadata.py's _transpose_jaxpr, dedupe!
# also dedupe with fused.py, compute_on.py... we have several call prims!
@weakref_lru_cache
def _transpose_jaxpr(jaxpr, in_tree, in_avals, specs):
  out_tree = None
  def transposed(*in_flat):
    nonlocal out_tree
    primals_ctrefs, cts_in = tree_unflatten(in_tree, in_flat)
    args = ad.unproject_accums(specs, primals_ctrefs)
    logs = ad.backward_pass3(jaxpr, False, jaxpr.consts, args, cts_in)
    cts_out = [x.freeze() if isinstance(x, ad.ValAccum) else None for x in args]
    outs, out_tree = tree_flatten((cts_out, logs))
    return outs
  dbg = jaxpr.debug_info.with_unknown_names()
  trans_jaxpr, _ = pe.trace_to_jaxpr(transposed, ft.flatten_args(*in_avals), dbg)
  return trans_jaxpr, out_tree


eval_jaxpr_jvp = _eval_jaxpr_jvp
eval_jaxpr_batch = _eval_jaxpr_batch
eval_jaxpr_linearize = _eval_jaxpr_linearize
eval_jaxpr_transpose = _eval_jaxpr_transpose


def register_call_primitive_rules(
    prim: core.Primitive,
    name: str | None = None,
    transpose_rule=None,
) -> None:
  """Registers standard call transformation rules onto a Primitive."""
  prim.multiple_results = True
  prim.def_impl(eval_jaxpr_p.impl)
  prim.def_effectful_abstract_eval(eval_jaxpr_p.abstract_eval)
  core.custom_typechecks[prim] = core.custom_typechecks[eval_jaxpr_p]
  prim.to_lojax = partial(pe._eval_jaxpr_to_lojax, prim)
  ad.primitive_jvps[prim] = partial(_eval_jaxpr_jvp, prim)
  ad.primitive_linearizations[prim] = partial(_eval_jaxpr_linearize, prim)
  ad.fancy_transposes[prim] = transpose_rule or partial(_eval_jaxpr_transpose, prim)
  batching.fancy_primitive_batchers[prim] = partial(_eval_jaxpr_batch, prim)
  pe.custom_partial_eval_rules[prim] = partial(pe._eval_jaxpr_partial_eval, prim)
  pe.partial_eval_jaxpr_custom_rules[prim] = (
      pe.partial_eval_jaxpr_custom_rules[eval_jaxpr_p]
  )
  pe.dce_rules[prim] = pe.dce_rules[eval_jaxpr_p]
  discharge.register_discharge_rule(prim)(
      partial(discharge._eval_jaxpr_discharge_rule, prim)
  )
  if name is not None:
    mlir.register_lowering(
        prim,
        partial(mlir.core_call_lowering, name=name),
        cacheable=False,
    )


def create_call_primitive(name: str) -> core.Primitive:
  """Creates a JAX Primitive with standard call rules registered."""
  prim = core.Primitive(name)
  register_call_primitive_rules(prim, name=name)
  return prim


register_call_primitive_rules(eval_jaxpr_p, name="eval_jaxpr")
