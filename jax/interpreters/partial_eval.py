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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
from collections import namedtuple, Counter, defaultdict

from .. import core
from .. import linear_util as lu
from ..linear_util import thunk, transformation, transformation_with_aux
from ..util import unzip2, safe_zip, safe_map, toposort, partial
from ..core import (Trace, Tracer, new_master, Jaxpr, JaxprEqn, get_aval, pack,
                    AbstractValue, AbstractTuple, unit, unitvar, Primitive,
                    call_p)

map = safe_map
zip = safe_zip

class JaxprTrace(Trace):
  def pure(self, val):
    return self.new_const(val)

  def lift(self, val):
    return self.new_const(val)

  def sublift(self, val):
    return JaxprTracer(self, val.pval, FreeVar(val))

  def new_const(self, val):
    if isinstance(val, Tracer) and val.trace.level == self.level:
      raise Exception
    return JaxprTracer(self, PartialVal((None, val)), unit)

  def new_instantiated_const(self, val):
    return JaxprTracer(self, PartialVal((get_aval(val), unit)), ConstVar(val))

  def new_arg(self, pval):
    _, const = pval
    return JaxprTracer(self, pval, LambdaBinding())

  def instantiate_const(self, tracer):
    pv, const = tracer.pval
    if isinstance(pv, AbstractValue):
      return tracer
    elif isinstance(pv, JaxprTracerTuple):
      return pack(map(lambda t: self.instantiate_const(self.full_raise(t)), tracer))
    elif pv is None:
      return self.new_instantiated_const(const)
    else:
      raise TypeError(pv)

  def process_primitive(self, primitive, tracers, params):
    tracers = map(self.instantiate_const, tracers)
    avals = [t.aval for t in tracers]
    out_aval = primitive.abstract_eval(*avals, **params)
    eqn = JaxprEqn(tracers, None, primitive, (), False, params)
    return JaxprTracer(self, PartialVal((out_aval, unit)), eqn)

  def pack(self, tracers):
    eqn = JaxprEqn(tracers, None, core.pack_p, (), False, {})
    pval = pack_pvals([t.pval for t in tracers])
    return JaxprTracer(self, pval, eqn)

  def process_call(self, call_primitive, f, tracers, params):
    in_pvs, in_consts = unzip2([t.pval for t in tracers])
    fun, aux = partial_eval(f, self, in_pvs)
    out_pv_const, consts = call_primitive.bind(fun, *in_consts, **params)
    out_pv, jaxpr, env = aux()
    const_tracers = map(self.new_instantiated_const, consts)
    env_tracers = map(self.full_raise, env)
    bound_subjaxpr = (jaxpr, const_tracers, env_tracers)
    eqn = JaxprEqn(tracers, None, call_primitive, (bound_subjaxpr,), False, params)
    return JaxprTracer(self, PartialVal((out_pv, out_pv_const)), eqn)

  def post_process_call(self, call_primitive, out_tracer):
    jaxpr, consts, env = tracers_to_jaxpr([], out_tracer)
    out_pv, out_pv_const = out_tracer.pval
    out = pack((out_pv_const, pack(consts)))
    master = self.master
    def todo(x):
      out_pv_const, consts = x
      trace = JaxprTrace(master, core.cur_sublevel())
      const_tracers = map(trace.new_instantiated_const, consts)
      env_tracers = map(trace.full_raise, env)
      bound_subjaxpr = (jaxpr, const_tracers, env_tracers)
      eqn = JaxprEqn([], None, call_primitive, (bound_subjaxpr,), False, {})
      return JaxprTracer(trace, PartialVal((out_pv, out_pv_const)), eqn)

    return out, todo


def partial_eval(f, trace, pvs):
  f = trace_to_subjaxpr(f, trace.master)
  return partial_eval_wrapper(f, tuple(pvs))


@transformation_with_aux
def partial_eval_wrapper(avals, *consts, **kwargs):
  jaxpr, (out_pval, consts, env) = yield (map(PartialVal, zip(avals, consts)),)
  out_pv, out_pv_const = out_pval
  out = pack((out_pv_const, pack(consts)))
  yield out, (out_pv, jaxpr, env)


class JaxprTracer(Tracer):

  def __init__(self, trace, pval, recipe):
    assert isinstance(pval, PartialVal)
    pv, const = pval
    if isinstance(const, Tracer):
      assert const.trace.level < trace.level
    self.trace = trace
    self.pval = pval
    self.recipe = recipe

  def __repr__(self):
    return 'Traced<{}:{}>'.format(self.aval, self.trace)

  @property
  def aval(self):
    pv, const = self.pval
    return partial_val_aval(pv, const)

  @property
  def parents(self):
    if isinstance(self.recipe, JaxprEqn):
      return eqn_parents(self.recipe)
    elif isinstance(self.recipe, Destructuring):
      return eqn_parents(self.recipe.eqn)
    else:
      return []

  def ispure(self):
    pv, _ = self.pval
    return pv is None

  def full_lower(self):
    if self.ispure():
      _, const = self.pval
      return core.full_lower(const)
    else:
      return self

  def unpack(self):
    pv, const = self.pval
    if isinstance(pv, (AbstractValue, JaxprTracerTuple)):
      n = len(pv)
      if isinstance(pv, AbstractValue):
        const = [unit for _ in range(n)]
      key = object()
      eqn = JaxprEqn([self], [None]*n, core.identity_p, (), True, {})
      def child_tracer(i, pval, c):
        d = Destructuring(i, eqn, key)
        return JaxprTracer(self.trace, PartialVal((pval, c)), d).full_lower()
      return map(child_tracer, range(n), pv, const)
    elif pv is None:
      return const
    else:
      raise TypeError(pv)

class JaxprTracerTuple(tuple): pass

Destructuring = namedtuple('Destructuring', ['i', 'eqn', 'key'])

class PartialVal(tuple):
  def __new__(cls, xs):
    assert core.skip_checks or (
        isinstance(xs[0], valid_pv_types)
        and isinstance(xs[1], core.Tracer) or core.valid_jaxtype(xs[1])
    ), xs
    return tuple.__new__(cls, xs)

valid_pv_types = (AbstractValue, JaxprTracerTuple, type(None))


def def_abstract_eval(primitive, abstract_eval):
  primitive.abstract_eval = abstract_eval

def abstract_eval_unimplemented(primitive, *args, **kwargs):
  raise NotImplementedError("Abstract evaluation for '{}' not implemented"
                            .format(primitive.name))

Primitive.def_abstract_eval = def_abstract_eval
Primitive.abstract_eval = abstract_eval_unimplemented

abstract_unit = core.AbstractTuple()

def merge_pvals(val, pval):
  pv, const = pval
  if isinstance(pv, AbstractValue):
    return val
  elif isinstance(pv, JaxprTracerTuple):
    return pack(map(merge_pvals, val, zip(pv, const)))
  elif pv is None:
    return const
  else:
    raise TypeError(pv)

def as_abstract_val(pv):
  if isinstance(pv, AbstractValue):
    return pv
  elif isinstance(pv, JaxprTracerTuple):
    return AbstractTuple(map(as_abstract_val, pv))
  elif pv is None:
    raise TypeError("{} is not abstract".format(pv))


def partial_val_aval(pv, const):
  if isinstance(pv, AbstractValue):
    return pv
  elif isinstance(pv, JaxprTracerTuple):
    return AbstractTuple(map(partial_val_aval, pv, const))
  elif pv is None:
    return get_aval(const)
  else:
    raise TypeError(pv)


def pack_pvals(pvals):
  pvs, consts = unzip2(pvals)
  if all(pv is None for pv in pvs):
    pv_out = None
  elif all(isinstance(pv, AbstractValue) for pv in pvs):
    pv_out = AbstractTuple(pvs)
  else:
    pv_out = JaxprTracerTuple(pvs)
  return PartialVal((pv_out, pack(consts)))



def abstractify(x):
  return PartialVal((core.concrete_aval(x), unit))

def trace_unwrapped_to_jaxpr(fun, pvals, **kwargs):
  return trace_to_jaxpr(lu.wrap_init(fun), pvals, **kwargs)

def trace_to_jaxpr(fun, pvals, **kwargs):
  """Traces a function, given abstract inputs, to a jaxpr."""
  with new_master(JaxprTrace) as master:
    fun = trace_to_subjaxpr(fun, master)
    jaxpr, (out_pval, consts, env) = fun.call_wrapped(pvals, **kwargs)
    assert not env
    del master

  return jaxpr, out_pval, consts

@transformation
def trace_to_subjaxpr(master, pvals, **kwargs):
  assert all([isinstance(pv, PartialVal) for pv in pvals]), pvals
  trace = JaxprTrace(master, core.cur_sublevel())
  in_tracers = map(trace.new_arg, pvals)
  out_tracer = yield in_tracers
  out_tracer = trace.full_raise(out_tracer)
  jaxpr, consts, env = tracers_to_jaxpr(in_tracers, out_tracer)
  out_pval = out_tracer.pval
  del trace, in_tracers, out_tracer
  yield jaxpr, (out_pval, consts, env)


FreeVar = namedtuple('FreeVar', ['val'])
ConstVar = namedtuple('ConstVar', ['val'])
LambdaBinding = namedtuple('LambdaBinding', [])

def eqn_tracer_to_var(var, outvars, eqn):
  invars, _, primitive, bound_subjaxprs, destructure, params = eqn
  invars = map(var, invars)
  new_bound_subjaxprs = [(j, map(var, c), map(var, f))
                         for j, c, f in bound_subjaxprs]
  return JaxprEqn(invars, outvars, primitive,
                  new_bound_subjaxprs, destructure, params)


def tracers_to_jaxpr(in_tracers, out_tracer):
  newvar = gensym('')
  t_to_var = defaultdict(newvar)
  var = lambda t: t_to_var[id(t)]
  sorted_tracers = toposort(out_tracer)
  invars = map(var, in_tracers)
  eqns = []
  env = {}
  consts = {}
  destructuring_vars = {}
  for t in sorted_tracers:
    recipe = t.recipe
    if isinstance(recipe, JaxprEqn):
      eqns.append(eqn_tracer_to_var(var, [var(t)], recipe))
    elif isinstance(recipe, LambdaBinding):
      assert in_tracers, "Lambda binding with no args"
    elif isinstance(recipe, FreeVar):
      env[var(t)] = recipe.val
    elif isinstance(recipe, ConstVar):
      consts[var(t)] = recipe.val
    elif isinstance(recipe, Destructuring):
      i, eqn, key = recipe
      if key not in destructuring_vars:
        outvars = [newvar() for _ in eqn.outvars]
        eqns.append(eqn_tracer_to_var(var, outvars, eqn))
        destructuring_vars[key] = outvars
      else:
        outvars = destructuring_vars[key]
      t_to_var[id(t)] = outvars[i]
    elif recipe is unit:
      t_to_var[id(t)] = unitvar
    else:
      raise TypeError(recipe)

  env_vars, env_vals = unzip2(env.items())
  const_vars, const_vals = unzip2(consts.items())
  jaxpr = Jaxpr(const_vars, env_vars, invars, var(out_tracer), eqns)
  core.skip_checks or core.check_jaxpr(jaxpr)
  return jaxpr, const_vals, env_vals


def gensym(suffix):
  counter = it.count()
  return lambda: Var(next(counter), suffix)

class Var(object):
  def __init__(self, count, suffix):
    self.count = count
    self.suffix = suffix

  def __repr__(self):
    rem = self.count
    s = ''
    while True:
      rem, i = rem // 26, rem % 26
      s = chr(97 + i % 26) + s
      if not rem:
        break
    return s + self.suffix

def eqn_parents(eqn):
  subjaxpr_tracers = [it.chain(c, f) for _, c, f in eqn.bound_subjaxprs]
  return list(it.chain(eqn.invars,  *subjaxpr_tracers))


def eval_jaxpr_raw(jaxpr, consts, freevar_vals, *args):
  assert all(map(core.valid_jaxtype, consts))
  assert all(map(core.valid_jaxtype, freevar_vals))
  assert all(map(core.valid_jaxtype, args))

  def read(v):
    return env[v]

  def write(v, val):
    env[v] = val

  env = {}
  write(unitvar, unit)
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  map(write, jaxpr.freevars, freevar_vals)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    subfuns = [partial(core.eval_jaxpr, subjaxpr, map(read, const_bindings),
                                                  map(read, freevar_bindings))
               for subjaxpr, const_bindings, freevar_bindings
               in eqn.bound_subjaxprs]
    ans = eqn.primitive.impl(*(subfuns + in_vals), **eqn.params)  # not bind!
    outvals = list(ans) if eqn.destructure else [ans]
    map(write, eqn.outvars, outvals)
  return read(jaxpr.outvar)

def compiled_call_impl(fun, *args, **kwargs):
  with new_master(JaxprTrace, True) as master:
    pvals = map(abstractify, args)
    jaxpr, (pval, consts, env) = trace_to_subjaxpr(fun, master).call_wrapped(pvals)
    jaxpr_ans = eval_jaxpr_raw(jaxpr, consts, env, *args)
    ans = merge_pvals(jaxpr_ans, pval)
    del master, pvals, pval, consts, env, jaxpr_ans, jaxpr
    return ans

compiled_call_p = Primitive('compiled_call')
compiled_call = partial(core.call_bind, compiled_call_p)
compiled_call_p.def_custom_bind(compiled_call)
compiled_call_p.def_impl(compiled_call_impl)
