# Copyright 2018 Google LLC
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


import functools
import itertools as it
from typing import Any, Callable, Dict, Set, List, Sequence, Optional

from . import partial_eval as pe
from .. import core as core
from ..core import Trace, Tracer, new_master, get_aval, call_p, Primitive, Literal
from ..ad_util import (add_jaxvals, add_jaxvals_p, zeros_like_jaxval, zeros_like_aval,
                       zeros_like_p, Zero)
from ..abstract_arrays import raise_to_shaped
from ..util import unzip2, safe_map, safe_zip, partial, split_list, wrap_name
from ..tree_util import register_pytree_node
from .. import linear_util as lu
from ..api_util import flatten_fun, flatten_fun_nokwargs
from ..tree_util import tree_flatten, tree_unflatten

zip = safe_zip
map = safe_map
def identity(x): return x


def jvp(fun: lu.WrappedFun, has_aux=False, instantiate=True) -> Any:
  if not has_aux:
    return jvpfun(jvp_subtrace(fun), instantiate)
  else:
    fun, aux = jvp_subtrace_aux(fun)
    return jvpfun(fun, instantiate), aux


@lu.transformation
def jvpfun(instantiate, primals, tangents):
  with new_master(JVPTrace) as master:
    out_primals, out_tangents = yield (master, primals, tangents), {}
    del master
  if type(instantiate) is bool:
    instantiate = [instantiate] * len(out_tangents)
  out_tangents = [instantiate_zeros(x, t) if inst else t for x, t, inst
                  in zip(out_primals, out_tangents, instantiate)]
  yield out_primals, out_tangents

@lu.transformation
def jvp_subtrace(master, primals, tangents):
  trace = JVPTrace(master, core.cur_sublevel())
  for x in list(primals) + list(tangents):
    if isinstance(x, Tracer):
      assert x._trace.level < trace.level
  in_tracers = [JVPTracer(trace, x, t) if type(t) is not Zero else x
                for x, t in zip(primals, tangents)]
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  yield unzip2([(out_tracer.primal, out_tracer.tangent)
                for out_tracer in out_tracers])

@lu.transformation_with_aux
def jvp_subtrace_aux(master, primals, tangents):
  trace = JVPTrace(master, core.cur_sublevel())
  for x in list(primals) + list(tangents):
    if isinstance(x, Tracer):
      assert x._trace.level < trace.level
  ans, aux = yield map(partial(JVPTracer, trace), primals, tangents), {}
  ans_tracers = map(trace.full_raise, ans)
  aux_tracers = map(trace.full_raise, aux)
  out_primals, out_tangents = unzip2((t.primal, t.tangent) for t in ans_tracers)
  aux_primals, _            = unzip2((t.primal, t.tangent) for t in aux_tracers)
  aux_primals = map(core.full_lower, aux_primals)
  yield (out_primals, out_tangents), aux_primals

def linearize(traceable, *primals, **kwargs):
  has_aux = kwargs.pop('has_aux', False)
  if not has_aux:
    jvpfun = jvp(traceable)
  else:
    jvpfun, aux = jvp(traceable, has_aux=True)

  in_pvals = (tuple(pe.PartialVal.known(p) for p in primals)
            + tuple(pe.PartialVal.unknown(get_aval(p).at_least_vspace())
                    for p in primals))
  _, in_tree = tree_flatten(((primals, primals), {}))
  jvpfun_flat, out_tree = flatten_fun(jvpfun, in_tree)
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr(jvpfun_flat, in_pvals)
  out_primals_pvals, out_tangents_pvals = tree_unflatten(out_tree(), out_pvals)
  assert all(out_primal_pval.is_known() for out_primal_pval in out_primals_pvals)
  _, out_primals_consts = unzip2(out_primals_pvals)
  jaxpr.invars = jaxpr.invars[len(primals):]
  jaxpr.outvars = jaxpr.outvars[len(out_primals_pvals):]
  if not has_aux:
    return out_primals_consts, out_tangents_pvals, jaxpr, consts
  else:
    return out_primals_consts, out_tangents_pvals, jaxpr, consts, aux()

def vjp(traceable, primals, has_aux=False):
  if not has_aux:
    out_primals, pvals, jaxpr, consts = linearize(traceable, *primals)
  else:
    out_primals, pvals, jaxpr, consts, aux = linearize(traceable, *primals, has_aux=True)
  def vjp_(*cts):
    cts = tuple(map(ignore_consts, cts, pvals))
    dummy_args = [UndefinedPrimal(v.aval) for v in jaxpr.invars]
    arg_cts = backward_pass(jaxpr, consts, dummy_args, cts)
    return map(instantiate_zeros, primals, arg_cts)

  if not has_aux:
    return out_primals, vjp_
  else:
    return out_primals, vjp_, aux

def ignore_consts(ct, pval):
  aval, const = pval
  if isinstance(aval, core.AbstractValue):
    return ct
  elif aval is None:
    return core.unit
  else:
    raise TypeError(aval)

def unpair_pval(pval):
  aval, const = pval
  const_1, const_2 = const
  if aval is None:
    return (None, const_1), (None, const_2)
  else:
    aval_1, aval_2 = aval
    return (aval_1, const_1), (aval_2, const_2)

# NOTE: The FIXMEs below are caused by primal/tangent mixups (type errors if you will)
def backward_pass(jaxpr: core.Jaxpr, consts, primals_in, cotangents_in):
  if all(type(ct) is Zero for ct in cotangents_in):
    return map(lambda v: Zero(v.aval), jaxpr.invars)

  def write_cotangent(v, ct):
    # assert v not in primal_env
    if ct is not None and type(v) is not Literal and type(ct) is not Zero:
      ct_env[v] = add_tangents(ct_env[v], ct) if v in ct_env else ct
      if not core.skip_checks:
        ct_aval = core.get_aval(ct_env[v])
        assert v.aval == core.lattice_join(v.aval, ct_aval), (v.aval, ct_aval)

  def read_cotangent(v):
    return ct_env.get(v, Zero(v.aval))

  def read_primal(v):
    if type(v) is Literal:
      return v.val
    else:
      return primal_env.get(v, UndefinedPrimal(v.aval))

  def write_primal(v, val):
    if not is_undefined_primal(val):
      primal_env[v] = val

  primal_env: Dict[Any, Any] = {}
  write_primal(core.unitvar, core.unit)
  map(write_primal, jaxpr.constvars, consts)
  # FIXME: invars can contain both primal and tangent values, and this line
  #        forces primal_in to contain UndefinedPrimals for tangent values!
  map(write_primal, jaxpr.invars, primals_in)

  # Find the last use of each cotangent so that they can be removed
  # as soon as possible.
  drop_cts: List[Set[Any]] = []
  seen_vars: Set[Any] = set(jaxpr.invars)
  for eqn in jaxpr.eqns:
    read_set = set(eqn.outvars)  # NOTE: eqn is not transposed yet!
    drop_cts.append(read_set - seen_vars)
    seen_vars |= read_set

  ct_env: Dict[Any, Any] = {}
  map(write_cotangent, jaxpr.outvars, cotangents_in)
  for eqn, to_drop in zip(jaxpr.eqns[::-1], drop_cts[::-1]):
    # FIXME: Some invars correspond to tangents
    invals = map(read_primal, eqn.invars)
    if eqn.primitive.multiple_results:
      cts_in = map(read_cotangent, eqn.outvars)
    else:
      cts_in, = map(read_cotangent, eqn.outvars)
    if eqn.primitive.call_primitive or eqn.primitive.map_primitive:
      cts_in_avals = [v.aval for v in eqn.outvars]
      call_jaxpr, params = core.extract_call_jaxpr(eqn.primitive, eqn.params)
      cts_out = get_primitive_transpose(eqn.primitive)(
          params, call_jaxpr, invals, cts_in, cts_in_avals)
    else:
      cts_out = get_primitive_transpose(eqn.primitive)(cts_in, *invals, **eqn.params)
    cts_out = map(lambda v: Zero(v.aval), eqn.invars) if cts_out is Zero else cts_out
    # FIXME: Some invars correspond to primals!
    map(write_cotangent, eqn.invars, cts_out)
    for var in to_drop:
      ct_env.pop(var, None)  # NB: Constant cotangents might be missing

  cotangents_out = map(read_cotangent, jaxpr.invars)
  return cotangents_out

class UndefinedPrimal:
  __slots__ = ['aval']
  def __init__(self, aval):
    self.aval = aval
  def __repr__(self):
    return 'UndefinedPrimal({})'.format(self.aval)

def is_undefined_primal(x):
  return type(x) is UndefinedPrimal

register_pytree_node(UndefinedPrimal,
                     lambda z: ((), z.aval),
                     lambda aval, _: UndefinedPrimal(aval))

def get_primitive_transpose(p):
  try:
    return primitive_transposes[p]
  except KeyError as err:
    raise NotImplementedError(
        "Transpose rule (for reverse-mode differentiation) for '{}' "
        "not implemented".format(p)) from err

class JVPTrace(Trace):

  def pure(self, val):
    return JVPTracer(self, val, Zero.from_value(val))

  def lift(self, val):
    return JVPTracer(self, val, Zero.from_value(val))

  def sublift(self, val):
    return JVPTracer(self, val.primal, val.tangent)

  def process_primitive(self, primitive, tracers, params):
    primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
    try:
      jvp = primitive_jvps[primitive]
    except KeyError as err:
      raise NotImplementedError(
          "Forward-mode differentiation rule for '{}' not implemented"
          .format(primitive)) from err
    primal_out, tangent_out = jvp(primals_in, tangents_in, **params)
    if primitive.multiple_results:
      return [JVPTracer(self, x, t) for x, t in zip(primal_out, tangent_out)]
    else:
      return JVPTracer(self, primal_out, tangent_out)

  def process_call(self, call_primitive, f: lu.WrappedFun, tracers, params):
    assert call_primitive.multiple_results
    primals, tangents = unzip2((t.primal, t.tangent) for t in tracers)
    nonzero_tangents, in_tree_def = tree_flatten(tangents)
    f_jvp, out_tree_def = traceable(jvp_subtrace(f, self.master),
                                    len(primals), in_tree_def)
    name = params.get('name', f.__name__)
    new_params = dict(params, name=wrap_name(name, 'jvp'))
    if 'donated_invars' in new_params:
      new_donated_invars = (*params['donated_invars'],
                            *[m for m, t in zip(params['donated_invars'], tangents)
                              if t is not zero])
      new_params['donated_invars'] = tuple(new_donated_invars)
    result = call_primitive.bind(f_jvp, *primals, *nonzero_tangents, **new_params)
    primal_out, tangent_out = tree_unflatten(out_tree_def(), result)
    return [JVPTracer(self, p, t) for p, t in zip(primal_out, tangent_out)]

  def post_process_call(self, call_primitive, out_tracers, params):
    primals, tangents = unzip2((t.primal, t.tangent) for t in out_tracers)
    out = primals + tangents
    del primals, tangents
    master = self.master
    def todo(x):
      n = len(x) // 2
      primals, tangents = x[:n], x[n:]
      trace = JVPTrace(master, core.cur_sublevel())
      return map(partial(JVPTracer, trace), primals, tangents)
    return out, todo

  def process_map(self, map_primitive, f: lu.WrappedFun, tracers, params):
    # only differs from process_call in that it must update mapped_invars
    # TODO de-duplicate code
    assert map_primitive.multiple_results
    primals, tangents = unzip2((t.primal, t.tangent) for t in tracers)
    nonzero_tangents, in_tree_def = tree_flatten(tangents)
    f_jvp, out_tree_def = traceable(jvp_subtrace(f, self.master),
                                    len(primals), in_tree_def)
    new_name = wrap_name(params.get('name', f.__name__), 'jvp')
    new_mapped_invars = (*params['mapped_invars'],
                         *[m for m, t in zip(params['mapped_invars'], tangents)
                           if type(t) is not Zero])
    new_donated_invars = (*params['donated_invars'],
                          *[m for m, t in zip(params['donated_invars'], tangents)
                            if type(t) is not Zero])
    new_params = dict(params, name=new_name, mapped_invars=new_mapped_invars,
                      donated_invars=new_donated_invars)
    result = map_primitive.bind(f_jvp, *primals, *nonzero_tangents, **new_params)
    primal_out, tangent_out = tree_unflatten(out_tree_def(), result)
    return [JVPTracer(self, p, t) for p, t in zip(primal_out, tangent_out)]
  post_process_map = post_process_call

  def process_custom_jvp_call(self, _, __, f_jvp, tracers):
    primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
    primals_in = map(core.full_lower, primals_in)
    tangents_in = map(instantiate_zeros, primals_in, tangents_in)
    outs = f_jvp.call_wrapped(*it.chain(primals_in, tangents_in))
    primals_out, tangents_out = split_list(outs, [len(outs) // 2])
    return map(partial(JVPTracer, self), primals_out, tangents_out)

  def process_custom_vjp_call(self, _, __, fwd, bwd, tracers, *, out_trees):
    primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
    tangents_in = map(instantiate_zeros, primals_in, tangents_in)
    res_and_primals_out = fwd.call_wrapped(*map(core.full_lower, primals_in))
    out_tree, res_tree = out_trees()
    res, primals_out = split_list(res_and_primals_out, [res_tree.num_leaves])
    avals_out = [raise_to_shaped(core.get_aval(x)) for x in primals_out]
    tangents_out = custom_lin_p.bind(
        *res, *tangents_in, num_res=res_tree.num_leaves, bwd=bwd,
        avals_out=avals_out)
    return map(partial(JVPTracer, self), primals_out, tangents_out)

  def join(self, xt, yt):
    xz, yz = type(xt) is Zero, type(yt) is Zero
    if xz == yz:
      return xt, yt
    elif yz and not xz:
      return xt, zeros_like_jaxval(xt)
    elif xz and not yz:
      return zeros_like_jaxval(yt), yt
    else:
      raise TypeError((xt, yt))


class JVPTracer(Tracer):
  __slots__ = ['primal', 'tangent']

  def __init__(self, trace, primal, tangent):
    if not core.skip_checks:
      _primal_tangent_shapes_match(primal, tangent)
    self._trace = trace
    self.primal = primal
    self.tangent = tangent

  @property
  def aval(self):
    # TODO(dougalm): add epsilon ball
    return get_aval(self.primal)

  def full_lower(self):
    if type(self.tangent) is Zero:
      return core.full_lower(self.primal)
    else:
      return self

def _primal_tangent_shapes_match(primal, tangent):
  if type(tangent) is not Zero:
    primal_aval = raise_to_shaped(get_aval(primal))
    tangent_aval = raise_to_shaped(get_aval(tangent))
    assert primal_aval == tangent_aval

# -------------------- Primitives --------------------


primitive_jvps : Dict[core.Primitive, Callable] = {}

primitive_transposes: Dict[core.Primitive, Callable] = {}


def deflinear(primitive, transpose_rule):
  primitive_jvps[primitive] = partial(linear_jvp, primitive)
  primitive_transposes[primitive] = partial(linear_transpose, transpose_rule)

def linear_jvp(primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  if all(type(tangent) is Zero for tangent in tangents):
    return val_out, Zero.from_value(val_out)
  else:
    tangents = map(instantiate_zeros, primals, tangents)
    return val_out, primitive.bind(*tangents, **params)

def linear_transpose(transpose_rule, cotangent, *args, **kwargs):
  return Zero if type(cotangent) is Zero else transpose_rule(cotangent, **kwargs)


def deflinear2(primitive, transpose_rule):
  primitive_jvps[primitive] = partial(linear_jvp, primitive)
  primitive_transposes[primitive] = partial(linear_transpose2, transpose_rule)

def linear_transpose2(transpose_rule, cotangent, *args, **kwargs):
  return Zero if type(cotangent) is Zero else transpose_rule(cotangent, *args, **kwargs)


def defjvp(primitive, *jvprules):
  assert isinstance(primitive, Primitive)
  assert not primitive.multiple_results
  primitive_jvps[primitive] = partial(standard_jvp, jvprules, primitive)


def standard_jvp(jvprules, primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  tangents_out = [rule(t, *primals, **params) for rule, t in zip(jvprules, tangents)
                  if rule is not None and type(t) is not Zero]
  return val_out, functools.reduce(add_tangents, tangents_out, Zero.from_value(val_out))

def defjvp2(primitive, *jvprules):
  assert isinstance(primitive, Primitive)
  assert not primitive.multiple_results
  primitive_jvps[primitive] = partial(standard_jvp2, jvprules, primitive)

def standard_jvp2(jvprules, primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  tangents_out = (rule(t, val_out, *primals, **params) for rule, t in zip(jvprules, tangents)
                  if rule is not None and type(t) is not Zero)
  tangents_out = list(tangents_out)
  return val_out, functools.reduce(add_tangents, tangents_out, Zero.from_value(val_out))

def add_tangents(x, y):
  if type(x) is Zero:
    return y
  elif type(y) is Zero:
    return x
  else:
    return add_jaxvals(x, y)


def defbilinear_broadcasting(bcast, prim, lhs_rule, rhs_rule):
  assert isinstance(prim, Primitive)
  lhs_jvp = lambda g, x, y, **kwargs: prim.bind(bcast(g, y), y, **kwargs)
  rhs_jvp = lambda g, x, y, **kwargs: prim.bind(x, bcast(g, x), **kwargs)
  defjvp(prim, lhs_jvp, rhs_jvp)
  primitive_transposes[prim] = partial(bilinear_transpose, lhs_rule, rhs_rule)
defbilinear = partial(defbilinear_broadcasting, lambda g, x: g)

def bilinear_transpose(lhs_rule, rhs_rule, cotangent, x, y, **kwargs):
  assert is_undefined_primal(x) ^ is_undefined_primal(y)
  if type(cotangent) is Zero:
    return Zero
  if is_undefined_primal(x):
    out = lhs_rule(cotangent, y, **kwargs)
    return Zero if out is Zero else (out, None)
  else:
    out = rhs_rule(cotangent, x, **kwargs)
    return Zero if out is Zero else (None, out)


def defjvp_zero(primitive):
  assert isinstance(primitive, Primitive)
  primitive_jvps[primitive] = partial(zero_jvp, primitive)

def zero_jvp(primitive, primals, tangents, **params):
  r = primitive.bind(*primals, **params)
  return r, Zero.from_value(r)


deflinear(zeros_like_p, lambda t: [Zero.from_value(t)])
deflinear(core.identity_p, lambda t: (t,))
deflinear(add_jaxvals_p, lambda t: (t, t))

def instantiate_zeros(example, tangent):
  if type(tangent) is Zero:
    return zeros_like_jaxval(example)
  else:
    return tangent

def instantiate_zeros_aval(aval, tangent):
  if type(tangent) is Zero:
    return zeros_like_aval(aval)
  else:
    return tangent

@lu.transformation_with_aux
def traceable(num_primals, in_tree_def, *primals_and_tangents):
  new_primals  = primals_and_tangents[:num_primals]
  new_tangents = primals_and_tangents[num_primals:]
  new_tangents = tree_unflatten(in_tree_def, new_tangents)
  primal_out, tangent_out = yield (new_primals, new_tangents), {}
  out_flat, tree_def = tree_flatten((primal_out, tangent_out))
  yield out_flat, tree_def


def call_transpose(primitive, params, call_jaxpr, args, ct, _):
  all_args, in_tree_def = tree_flatten(((), args, ct))  # empty consts
  fun = lu.hashable_partial(lu.wrap_init(backward_pass), call_jaxpr)
  fun, out_tree = flatten_fun_nokwargs(fun, in_tree_def)
  params = dict(params, name=wrap_name(params['name'], 'transpose'))
  if 'donated_invars' in params:
    new_donated_invars = (*[d for d, x in zip(params['donated_invars'], args)
                            if not is_undefined_primal(x)],
                          *[False for x in ct if x is not zero])
    params['donated_invars'] = tuple(new_donated_invars)
  out_flat = primitive.bind(fun, *all_args, **params)
  return tree_unflatten(out_tree(), out_flat)
primitive_transposes[core.call_p] = partial(call_transpose, call_p)


def remat_transpose(params, call_jaxpr, primals_in, cotangents_in, cotangent_in_avals):
  # backward_pass can only transpose linear computations, but the call_jaxpr embedded in
  # remat contains primal (non-linear) equations too. Hence, we have to eliminate those
  # (in this case via partial_eval) before we call into backward_pass again.
  typed_call_jaxpr = core.TypedJaxpr(
      call_jaxpr, [],
      [raise_to_shaped(p.aval if is_undefined_primal(p) else get_aval(p)) for p in primals_in],
      cotangent_in_avals)
  primal_jaxpr, tangent_jaxpr, out_unknowns = \
    pe.partial_eval_jaxpr(typed_call_jaxpr,
                          unknowns=map(is_undefined_primal, primals_in),
                          instantiate=True,
                          trace_type=None)

  def do_transpose(primals_in, cotangents_in):
    # NOTE: This is passing in undefined primals in place of tangent arguments, but it
    #       should all work out, because we're only computing the primal part here.
    residuals = core.jaxpr_as_fun(primal_jaxpr)(*primals_in)[len(cotangents_in):]
    # Now that we have a purely linear jaxpr, we can transpose it
    cotangents_out = backward_pass(tangent_jaxpr.jaxpr, (), primals_in + residuals, cotangents_in)
    # backward_pass will return cotangents computed for all invars, but some of them
    # are residuals appended by partial eval, so we need to skip those before we return.
    return cotangents_out[:len(primals_in)]

  flat_args, in_tree_def = tree_flatten((primals_in, cotangents_in))
  flat_do_transpose, out_tree = flatten_fun_nokwargs(lu.wrap_init(do_transpose), in_tree_def)
  flat_cotangents_out = pe.remat_call_p.bind(flat_do_transpose, *flat_args, **params)
  return tree_unflatten(out_tree(), flat_cotangents_out)
primitive_transposes[pe.remat_call_p] = remat_transpose


def map_transpose(primitive, params, call_jaxpr, args, ct, _):
  all_args, in_tree_def = tree_flatten(((), args, ct))  # empty consts
  fun = lu.hashable_partial(lu.wrap_init(backward_pass), call_jaxpr)
  fun, out_tree = flatten_fun_nokwargs(fun, in_tree_def)
  new_mapped_invars = (*[m for m, x in zip(params['mapped_invars'], args)
                         if not is_undefined_primal(x)],
                       *[True for x in ct if type(x) is not zero])
  new_donated_invars = (*[d for d, x in zip(params['donated_invars'], args)
                          if not is_undefined_primal(x)],
                        *[False for x in ct if type(x) is not zero])
  new_params = dict(params, name=wrap_name(params['name'], 'transpose'),
                    mapped_invars=tuple(new_mapped_invars),
                    donated_invars=tuple(new_donated_invars))
  out_flat = primitive.bind(fun, *all_args, **new_params)
  arg_cts = tree_unflatten(out_tree(), out_flat)

  mapped_invars = params['mapped_invars']  # True for each mapped invar
  # The freevars are being fanned out (not mapped). During transpose the
  # dual of fan-out is fan-in-sum. We apply it to the unmapped invars.
  assert len(mapped_invars) == len(arg_cts)
  arg_cts = (arg_ct if arg_mapped or type(arg_ct) is Zero else arg_ct.sum(0)
             for arg_ct, arg_mapped in zip(arg_cts, mapped_invars))

  return arg_cts


def jvp_jaxpr(jaxpr, nonzeros, instantiate):
  assert len(jaxpr.in_avals) == len(nonzeros)
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
  f_jvp, out_nonzeros = f_jvp_traceable(jvp(f, instantiate=instantiate), nonzeros)
  tangent_avals = [aval for aval, nz in zip(jaxpr.in_avals, nonzeros) if nz]
  avals_in = list(it.chain(jaxpr.in_avals, tangent_avals))
  pvals = [pe.PartialVal.unknown(aval) for aval in avals_in]
  jaxpr_out, pvals_out, literals_out = pe.trace_to_jaxpr(f_jvp, pvals, instantiate=True)
  avals_out, _ = unzip2(pvals_out)
  jaxpr_out = core.TypedJaxpr(jaxpr_out, literals_out, avals_in, avals_out)
  return jaxpr_out, out_nonzeros()

@lu.transformation_with_aux
def f_jvp_traceable(nonzeros, *primals_and_nztangents):
  num_primals = len(nonzeros)
  primals = list(primals_and_nztangents[:num_primals])
  nonzero_tangents = iter(primals_and_nztangents[num_primals:])
  tangents = [next(nonzero_tangents) if nz else Zero.from_value(p)
	      for p, nz in zip(primals, nonzeros)]
  primals_out, tangents_out = yield (primals, tangents), {}
  out_nonzeros = [type(t) is not Zero for t in tangents_out]
  nonzero_tangents_out = [t for t in tangents_out if type(t) is not Zero]
  yield list(primals_out) + nonzero_tangents_out, out_nonzeros

def rearrange_binders(jaxpr: core.TypedJaxpr, primals_in, tangents_in, primals_out, tangents_out):
  new_invars = _perm(primals_in, tangents_in, jaxpr.jaxpr.invars)
  new_outvars = _perm(primals_out, tangents_out, jaxpr.jaxpr.outvars)
  new_jaxpr = core.Jaxpr(jaxpr.jaxpr.constvars,
                         new_invars, new_outvars, jaxpr.jaxpr.eqns)
  new_in_avals = _perm(primals_in, tangents_in, jaxpr.in_avals)
  new_out_avals = _perm(primals_out, tangents_out, jaxpr.out_avals)
  new_typed_jaxpr = core.TypedJaxpr(new_jaxpr, jaxpr.literals, new_in_avals,
                                    new_out_avals)
  return new_typed_jaxpr

def _perm(primal_counts, tangent_counts, lst):
  n = sum(primal_counts)
  primals, tangents = lst[:n], lst[n:]
  primal_groups = split_list(primals, primal_counts[:-1])
  tangent_groups = split_list(tangents, tangent_counts[:-1])
  return _interleave(primal_groups, tangent_groups)

def _interleave(xs, ys):
  assert len(xs) == len(ys)
  return [e for pair in zip(xs, ys) for l in pair for e in l]


custom_lin_p = core.Primitive('custom_lin')
custom_lin_p.def_abstract_eval(lambda *_, avals_out, **__: avals_out)
custom_lin_p.multiple_results = True

def _raise_custom_vjp_error_on_jvp(*_, **__):
  raise TypeError("can't apply forward-mode autodiff (jvp) to a custom_vjp "
                  "function.")
custom_lin_p.def_impl(_raise_custom_vjp_error_on_jvp)

def _custom_lin_transpose(cts_out, *invals, num_res, bwd, avals_out):
  res, _ = split_list(invals, [num_res])
  cts_out = map(instantiate_zeros_aval, avals_out, cts_out)
  cts_in = bwd.call_wrapped(*res, *cts_out)
  cts_in_flat, _ = tree_flatten(cts_in)  # already checked tree structure
  return [None] * num_res + cts_in_flat
primitive_transposes[custom_lin_p] = _custom_lin_transpose


# TODO(mattjj): delete everything below here (deprecated custom_transforms)

def defvjp_all(prim, custom_vjp):
  # see https://github.com/google/jax/pull/636
  name = prim.name

  def fun_jvp(xs, ts, **params):
    ts = map(instantiate_zeros, xs, ts)
    primals_and_tangents = fun_jvp_p.bind(*it.chain(xs, ts), **params)
    primals, tangents = split_list(primals_and_tangents, [len(primals_and_tangents) // 2])
    if prim.multiple_results:
      return primals, tangents
    else:
      primal, = primals
      tangent, = tangents
      return primal, tangent
  primitive_jvps[prim] = fun_jvp

  fun_jvp_p = core.Primitive('{name}_jvp'.format(name=name))
  fun_jvp_p.multiple_results = True
  def fun_jvp_partial_eval(trace, *tracers, **params):
    primals, tangents = split_list(tracers, [len(tracers) // 2])
    primals_out, vjp_py = custom_vjp(*primals, **params)
    if not prim.multiple_results:
      primals_out = [primals_out]
    out_avals = [raise_to_shaped(get_aval(x)) for x in primals_out]
    ct_pvals = [pe.PartialVal.unknown(aval) for aval in out_avals]
    with core.initial_style_staging():
      jaxpr, _, res = pe.trace_to_jaxpr(lu.wrap_init(vjp_py), ct_pvals,
                                        instantiate=True)
    tangents_out = fun_lin_p.bind(*it.chain(res, tangents), trans_jaxpr=jaxpr,
                                  num_res=len(res), out_avals=out_avals)
    return primals_out + tangents_out
  pe.custom_partial_eval_rules[fun_jvp_p] = fun_jvp_partial_eval

  fun_lin_p = core.Primitive('{name}_lin'.format(name=name))
  fun_lin_p.multiple_results = True
  fun_lin_p.def_abstract_eval(lambda *_, **kwargs: kwargs['out_avals'])
  def fun_lin_transpose(cts, *args, **kwargs):
    num_res, trans_jaxpr = kwargs['num_res'], kwargs['trans_jaxpr']
    res, _ = split_list(args, [num_res])
    cts = map(instantiate_zeros_aval, kwargs['out_avals'], cts)
    outs = core.eval_jaxpr(trans_jaxpr, res, *cts)
    return [None] * num_res + outs
  primitive_transposes[fun_lin_p] = fun_lin_transpose

def defvjp(prim, *vjps):
  def vjpmaker(*primals):
    ans = prim.bind(*primals)
    vjpfun = lambda ct: [vjp(ct, *primals) if vjp else zeros_like_jaxval(x)
                         for x, vjp in zip(primals, vjps)]
    return ans, vjpfun
  defvjp_all(prim, vjpmaker)

def defvjp2(prim, *vjps):
  def vjpmaker(*primals):
    ans = prim.bind(*primals)
    vjpfun = lambda ct: [vjp(ct, ans, *primals) if vjp else zeros_like_jaxval(x)
                         for x, vjp in zip(primals, vjps)]
    return ans, vjpfun
  defvjp_all(prim, vjpmaker)
