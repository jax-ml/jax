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
from jax._src import dispatch
from jax._src.core import typeof
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_map, safe_zip, weakref_lru_cache, unzip2
from jax._src.api_util import debug_info, flatten_fun_nokwargs
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib.mlir import ir

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

def fused(*, out_spaces):
  def wrap(f):
    def wrapped(*args):
      dbg = debug_info('fused', f, args, {})
      args_flat, in_tree = tree_flatten(args)
      in_avals = [typeof(x).update(memory_space=core.MemorySpace.Any)
                  for x in args_flat]
      jaxpr, out_tree = _trace_to_jaxpr(f, in_tree, tuple(in_avals), dbg)
      outs_flat = fused_p.bind(*args_flat, jaxpr=jaxpr, out_spaces=out_spaces)
      return tree_unflatten(out_tree, outs_flat)
    return wrapped
  return wrap

@weakref_lru_cache
def _trace_to_jaxpr(fun, in_tree, in_avals, dbg):
  f = lu.wrap_init(fun, debug_info=dbg)
  f, out_tree = flatten_fun_nokwargs(f, in_tree)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(f, in_avals)
  return core.ClosedJaxpr(jaxpr, consts), out_tree()

fused_p = core.Primitive('fused_call')
fused_p.multiple_results = True

@fused_p.def_abstract_eval
def _fused_abstract_eval(*in_avals, out_spaces, jaxpr):
  return [a.update(memory_space=s)
          for a, s in zip(jaxpr.out_avals, out_spaces)]

dispatch.simple_impl(fused_p)

def _fused_lowering(ctx, *args, out_spaces, jaxpr):
  const_args_and_avals = core.jaxpr_const_args(jaxpr.jaxpr)
  const_args, const_arg_avals = unzip2(const_args_and_avals)
  const_arg_values = [
      mlir.ir_constant(c, const_lowering=ctx.const_lowering, aval=aval)
      for c, aval in const_args_and_avals]
  in_avals = [*const_arg_avals, *ctx.avals_in]
  func_op, _, _ = mlir.lower_called_computation(
      "fused", jaxpr, ctx.module_context, len(const_args), in_avals,
      ctx.avals_out, ctx.tokens_in)
  out_spaces_ = [ir.StringAttr.get(str(s)) for s in out_spaces]
  fused = mlir.custom_call(
      "fused",
      result_types=func_op.type.results,
      operands=mlir.flatten_ir_values([*const_arg_values, *args]),
      called_computations=[func_op.name.value],
      backend_config=dict(out_spaces=ir.ArrayAttr.get(out_spaces_),
                          inlineable=ir.BoolAttr.get(False),
                          MUST_FUSE=ir.BoolAttr.get(True)),
  )
  return fused.results
mlir.register_lowering(fused_p, _fused_lowering, platform="cuda")

def _fused_batcher(axis_data, vals_in, dims_in, *, jaxpr, out_spaces):
  batched_jaxpr, dims_out = batching.batch_jaxpr2(jaxpr, axis_data, dims_in)
  outs = fused_p.bind(*vals_in, jaxpr=batched_jaxpr, out_spaces=out_spaces)
  return outs, dims_out
batching.fancy_primitive_batchers[fused_p] = _fused_batcher

def _fused_jvp(primals, tangents, *, jaxpr, out_spaces):
  nzs = [not isinstance(t, ad.Zero) for t in tangents]
  jaxpr_jvp, out_nzs = ad.jvp_jaxpr(jaxpr, nzs, False)
  nz_tangents = [t for t in tangents if not isinstance(t, ad.Zero)]
  spaces_jvp = (*out_spaces, *[s for s, nz in zip(out_spaces, out_nzs) if nz])
  outs = fused_p.bind(*primals, *nz_tangents, jaxpr=jaxpr_jvp,
                      out_spaces=spaces_jvp)
  primals_out, nz_tangents_out = outs[:len(out_nzs)], outs[len(out_nzs):]
  nz_outs = iter(nz_tangents_out)
  tangents_out = [next(nz_outs) if nz else ad.Zero(aval.to_tangent_aval())
                  for aval, nz in zip(jaxpr.out_avals, out_nzs)]
  assert next(nz_outs, None) is None
  return primals_out, tangents_out
ad.primitive_jvps[fused_p] = _fused_jvp

def _fused_lin(nzs, *primals, jaxpr, out_spaces):
  jaxpr_jvp, out_nzs = ad.jvp_jaxpr(jaxpr, nzs, False)
  lin_outs = [False] * len(out_nzs) + [True] * sum(out_nzs)
  jaxpr_lin_, used_inputs = pe.dce_jaxpr(jaxpr_jvp.jaxpr, lin_outs, False)
  jaxpr_lin = pe.close_jaxpr(jaxpr_lin_)
  spaces_lin = tuple(s for s, nz in zip(out_spaces, out_nzs) if nz)
  primals_out = fused_p.bind(*primals, jaxpr=jaxpr, out_spaces=out_spaces)
  tangent_avals_out = [a.to_tangent_aval() for a in jaxpr.out_avals]

  def fused_lin(primals, *tangents):
    nz_tangents = [t for t in tangents if not isinstance(t, ad.Zero)]
    inputs = [x for x, u in zip([*primals, *nz_tangents], used_inputs) if u]
    nz_outs = fused_p.bind(*inputs, jaxpr=jaxpr_lin, out_spaces=spaces_lin)
    nz_outs_ = iter(nz_outs)
    outs = [next(nz_outs_) if nz else ad.Zero(a)
            for nz, a in zip(out_nzs, tangent_avals_out)]
    assert next(nz_outs_, None) is None
    return outs

  return primals_out, out_nzs, primals, fused_lin
ad.primitive_linearizations[fused_p] = _fused_lin

def _fused_transpose(cts_in, *primals_in, jaxpr, out_spaces):
  in_flat, in_tree = tree_flatten((primals_in, cts_in))
  in_avals = [typeof(x).update(memory_space=core.MemorySpace.Any)
              for x in in_flat]
  trans_jaxpr, out_tree = _transpose_jaxpr(jaxpr, in_tree, (*in_avals,))
  in_spaces = [x.aval.memory_space if isinstance(x, ad.UndefinedPrimal)
               else typeof(x).memory_space for x in primals_in]
  cts_out_ = tree_unflatten(out_tree, trans_jaxpr.out_avals)
  trans_spaces = tuple(s for x, s in zip(cts_out_, in_spaces) if x)
  cts_out = fused_p.bind(*in_flat, jaxpr=trans_jaxpr, out_spaces=trans_spaces)
  return tree_unflatten(out_tree, cts_out)

@weakref_lru_cache
def _transpose_jaxpr(jaxpr, in_tree, in_avals):
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
ad.primitive_transposes[fused_p] = _fused_transpose
