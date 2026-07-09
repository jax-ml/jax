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
"""Module for eval_jaxpr primitive."""

from jax._src import ad_util
from jax._src import core
from jax._src.core import Jaxpr
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.util import safe_map, safe_zip, split_list, subs_list

_map = safe_map
zip = safe_zip


eval_jaxpr_p = core.Primitive('eval_jaxpr')
eval_jaxpr_p.multiple_results = True

def _stage_jaxpr(trace: pe.DynamicJaxprTrace, source_info, *tracers,
                 jaxpr: Jaxpr):
  params = dict(call_jaxpr=jaxpr)
  return trace.default_process_primitive(core.closed_call_p, tracers, params,
                                         source_info=source_info)
pe.custom_staging_rules[eval_jaxpr_p] = _stage_jaxpr

@eval_jaxpr_p.def_effectful_abstract_eval  # abstract eval only used for jax2tf
def _stage_jaxpr_abstract_eval(*_, jaxpr):
  return jaxpr.out_avals, core.positional_effects(jaxpr)

@eval_jaxpr_p.def_impl
def _eval_jaxpr_impl(*args, jaxpr):
  return core.jaxpr_as_fun(jaxpr)(*args)

def _eval_jaxpr_jvp(primals, tangents, *, jaxpr):
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  jaxpr_jvp, nonzeros_out = ad.jvp_jaxpr(jaxpr, nonzeros, False)
  nz_tangents = [t for t, nz in zip(tangents, nonzeros) if nz]
  outs = eval_jaxpr_p.bind(*primals, *nz_tangents, jaxpr=jaxpr_jvp)
  primals_out, tangents_out = split_list(outs, [len(jaxpr.out_avals)])
  nz_tangents_out = iter(tangents_out)
  tangents_out = [next(nz_tangents_out) if nz else ad_util.Zero(aval.to_tangent_aval())
                  for aval, nz in zip(jaxpr.out_avals, nonzeros_out)]
  return primals_out, tangents_out
ad.primitive_jvps[eval_jaxpr_p] = _eval_jaxpr_jvp

def _eval_jaxpr_batching_rule(axis_data, args, dims, *, jaxpr):
  batched = [d is not None for d in dims]
  new_jaxpr, out_batched = batching.batch_jaxpr(jaxpr, axis_data, batched, False)
  new_args = [batching.moveaxis(x, d, 0)
              if d is not None and d != 0 else x
              for x, d in zip(args, dims)]
  outs = eval_jaxpr_p.bind(*new_args, jaxpr=new_jaxpr)
  out_dims = [0 if b else None for b in out_batched]
  return outs, out_dims
batching.fancy_primitive_batchers[eval_jaxpr_p] = _eval_jaxpr_batching_rule

def _eval_jaxpr_linearize(is_vjp, nzs, *primals_in, jaxpr):
  lin_out = ad.linearize_jaxpr(jaxpr, nzs, is_vjp=is_vjp)
  primal_jaxpr, num_res_out, nzs_out, in_fwd_res, tangent_jaxpr = lin_out
  primals_and_res = eval_jaxpr_p.bind(*primals_in, jaxpr=primal_jaxpr)
  primals_out, non_fwd_res = split_list(
      primals_and_res, [len(primals_and_res) - num_res_out])
  res = subs_list(in_fwd_res, [*jaxpr.consts, *primals_in], non_fwd_res)

  def tangent_fun(res, *tangents):
    nz_tangents = [ad.instantiate_zeros(x) for nz, x in zip(nzs, tangents) if nz]
    nz_tangents_out = eval_jaxpr_p.bind(*res, *nz_tangents, jaxpr=tangent_jaxpr)
    tangent_avals_out = [v.aval.to_tangent_aval() for v in jaxpr.outvars]
    nz_tangents_out_ = iter(nz_tangents_out)
    tangents_out = [next(nz_tangents_out_) if nz else ad_util.Zero(aval)
                    for aval, nz in zip(tangent_avals_out, nzs_out)]
    assert next(nz_tangents_out_, None) is None
    return tangents_out

  return primals_out, nzs_out, res, tangent_fun
ad.primitive_linearizations[eval_jaxpr_p] = _eval_jaxpr_linearize

def _eval_jaxpr_transpose(ct, *args, jaxpr):
  jaxpr_, consts = jaxpr, jaxpr.consts
  jaxpr_ = pe.convert_constvars_jaxpr(jaxpr_)
  ad.call_transpose_fancy(core.closed_call_p, ct, *consts, *args,
                          call_jaxpr=jaxpr_)
ad.fancy_transposes[eval_jaxpr_p] = _eval_jaxpr_transpose
