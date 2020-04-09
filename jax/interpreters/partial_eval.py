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


import itertools as it
from collections import namedtuple
import contextlib
import threading
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union
from weakref import ref

import numpy as onp

from .. import core
from .. import linear_util as lu
from ..abstract_arrays import ShapedArray, ConcreteArray, raise_to_shaped
from ..util import (unzip2, safe_zip, safe_map, toposort, partial, split_list,
                    wrap_name, cache)
from ..core import (Trace, Tracer, new_master, Jaxpr, Literal, get_aval,
                    AbstractValue, unit, unitvar, abstract_unit,
                    TypedJaxpr, new_jaxpr_eqn)

map = safe_map
zip = safe_zip
def identity(x): return x

class PartialVal(tuple):
  """Partial value: either a known value or an unknown (abstract) value.

  Represented as a pair `(aval_opt, const)` of one of two kinds:
  * `(None, <Constant>)` indicates a known value, either a Python regular
    value, or a Tracer.
  * `(<AbstractValue>, *)` indicates an unknown value characterized by an
    abstract value.
  """
  def __new__(cls, xs: Tuple[Optional[AbstractValue], core.Value]):
    pv, const = xs
    if not core.skip_checks:
      # type checks
      assert isinstance(pv, (AbstractValue, type(None))), xs
      assert isinstance(const, core.Tracer) or core.valid_jaxtype(const), xs
      # invariant checks
      if isinstance(pv, AbstractValue):
        assert const == core.unit, xs
    return tuple.__new__(cls, xs)

  @classmethod
  def known(cls, const: core.Value) -> 'PartialVal':
    return PartialVal((None, const))

  @classmethod
  def unknown(cls, aval: AbstractValue) -> 'PartialVal':
    return PartialVal((aval, core.unit))

  def is_known(self):
    return self[0] is None

  def get_known(self) -> Optional[core.Value]:
    """Get the known value, if known, else None."""
    return self[1] if self[0] is None else None

  def get_aval(self) -> AbstractValue:
    """Get the AbstractValue either directly for unknown values, or from the known constant."""
    known = self.get_known()
    if known is not None:
      return get_aval(known)
    else:
      return self[0]

  def merge_with_known(self, val: core.Value) -> core.Value:
    """Either the stored known value, or the given 'val'."""
    known = self.get_known()
    return known if known is not None else val


class JaxprTrace(Trace):
  def pure(self, val):
    return self.new_const(val)

  def lift(self, val):
    return self.new_const(val)

  def sublift(self, val):
    return JaxprTracer(self, val.pval, FreeVar(val))

  def new_const(self, val):
    if isinstance(val, Tracer) and val._trace.level == self.level:
      raise Exception
    return JaxprTracer(self, PartialVal.known(val), unit)

  def new_instantiated_literal(self, val):
    return JaxprTracer(self, PartialVal.unknown(get_aval(val)), Literal(val))

  def new_instantiated_const(self, val):
    return JaxprTracer(self, PartialVal.unknown(get_aval(val)), ConstVar(val))

  def new_arg(self, pval: PartialVal):
    return JaxprTracer(self, pval, LambdaBinding())

  def instantiate_const(self, tracer):
    const = tracer.pval.get_known()
    if const is None:
      return tracer
    else:
      if type(const) in core.literalable_types and onp.shape(const) == ():
        return self.new_instantiated_literal(const)
      else:
        return self.new_instantiated_const(const)

  def instantiate_const_abstracted(self, tracer):
    const = tracer.pval.get_known()
    if const is None:
      return tracer
    else:
      aval = raise_to_shaped(get_aval(const), onp.isscalar(const))
      return JaxprTracer(self, PartialVal.unknown(aval), ConstVar(const))

  def process_primitive(self, primitive, tracers, params):
    if primitive in custom_partial_eval_rules:
      return custom_partial_eval_rules[primitive](self, *tracers, **params)
    else:
      return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(self, primitive, tracers, params):
    consts = tuple(t.pval.get_known() for t in tracers)
    if all(c is not None for c in consts):
      return primitive.bind(*consts, **params)
    tracers = map(self.instantiate_const, tracers)
    avals = [t.aval for t in tracers]
    out_aval = primitive.abstract_eval(*avals, **params)
    if primitive.multiple_results:
      out_tracers = [JaxprTracer(self, PartialVal.unknown(aval), None)
                     for aval in out_aval]
      eqn = new_eqn_recipe(tracers, out_tracers, primitive, params)
      for t in out_tracers: t.recipe = eqn
      return out_tracers
    else:
      out_tracer = JaxprTracer(self, PartialVal.unknown(out_aval), None)
      out_tracer.recipe = new_eqn_recipe(tracers, [out_tracer], primitive, params)
      return out_tracer

  def process_call(self, call_primitive, f: lu.WrappedFun, tracers, params):
    name = params.get('name', f.__name__)
    if self.master.trace_type is StagingJaxprTrace:
      tracers = map(self.instantiate_const_abstracted, tracers)
    else:
      name = wrap_name(name, 'pe')
    params = dict(params, name=name)
    if call_primitive in call_partial_eval_rules:
      return call_partial_eval_rules[call_primitive](self, call_primitive, f, tracers, params)
    if call_primitive in map_primitives:
      return self.process_map(call_primitive, f, tracers, params)
    in_pvs, in_consts = unzip2([t.pval for t in tracers])
    fun, aux = partial_eval(f, self, in_pvs)
    out_flat = call_primitive.bind(fun, *in_consts, **params)
    out_pvs, jaxpr, env = aux()
    env_tracers = map(self.full_raise, env)
    out_pv_consts, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
    const_tracers = map(self.new_instantiated_const, consts)
    lifted_jaxpr = convert_constvars_jaxpr(jaxpr)
    out_tracers = [JaxprTracer(self, PartialVal((out_pv, out_pv_const)), None)
                   for out_pv, out_pv_const in zip(out_pvs, out_pv_consts)]
    new_params = dict(params, call_jaxpr=lifted_jaxpr)
    # The `jaxpr` already contains the env_vars at start of invars
    eqn = new_eqn_recipe(tuple(it.chain(const_tracers, env_tracers, tracers)),
                         out_tracers, call_primitive, new_params)
    for t in out_tracers:
      t.recipe = eqn
    return out_tracers

  def process_map(self, map_primitive, f: lu.WrappedFun, tracers, params):
    in_pvs, in_consts = unzip2([t.pval for t in tracers])
    reduced_pvs = [None if pv is None else _mapped_aval(pv) for pv in in_pvs]
    fun, aux = partial_eval(f, self, reduced_pvs)
    out_flat = map_primitive.bind(fun, *in_consts, **params)
    out_pvs_reduced, jaxpr, env = aux()
    out_pv_consts, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
    out_pvs = [None if pv is None else _unmapped_aval(params['axis_size'], pv)
               for pv in out_pvs_reduced]
    const_tracers = map(self.new_instantiated_const, consts)
    env_tracers = map(self.full_raise, env)
    lifted_jaxpr = convert_constvars_jaxpr(jaxpr)
    out_tracers = [JaxprTracer(self, PartialVal((out_pv, out_pv_const)), None)
                   for out_pv, out_pv_const in zip(out_pvs, out_pv_consts)]
    # The `jaxpr` already contains the env_vars at start of invars
    new_params = dict(params,
                      mapped_invars=tuple([True] * len(const_tracers) +
                                          [False] * len(env_tracers) +
                                          [True] * len(tracers)),
                      call_jaxpr=lifted_jaxpr)
    eqn = new_eqn_recipe(tuple(it.chain(const_tracers, env_tracers, tracers)),
                         out_tracers, map_primitive, new_params)
    for t in out_tracers:
      t.recipe = eqn
    return out_tracers

  def post_process_call(self, call_primitive, out_tracers, params):
    if call_primitive in map_primitives:
      return self.post_process_map(call_primitive, out_tracers, params)
    jaxpr, consts, env = tracers_to_jaxpr([], out_tracers)
    out_pvs, out_pv_consts = unzip2(t.pval for t in out_tracers)
    out = out_pv_consts + consts
    del consts, out_pv_consts
    master = self.master
    def todo(x):
      n = len(jaxpr.outvars)
      out_pv_consts, consts = x[:n], x[n:]
      trace = JaxprTrace(master, core.cur_sublevel())
      const_tracers = map(trace.new_instantiated_const, consts)
      env_tracers = map(trace.full_raise, env)
      lifted_jaxpr = convert_constvars_jaxpr(jaxpr)
      out_tracers = [JaxprTracer(trace, PartialVal((out_pv, out_pv_const)), None)
                     for out_pv, out_pv_const in zip(out_pvs, out_pv_consts)]
      new_params = dict(params, call_jaxpr=lifted_jaxpr)
      # The `jaxpr` already contains the env_vars at start of invars
      eqn = new_eqn_recipe(tuple(it.chain(const_tracers, env_tracers)),
                           out_tracers, call_primitive, new_params)
      for t in out_tracers:
        t.recipe = eqn
      return out_tracers
    return out, todo

  def post_process_map(self, map_primitive, out_tracers, params):
    jaxpr, consts, env = tracers_to_jaxpr([], out_tracers)
    out_pvs_reduced, out_pv_consts = unzip2(t.pval for t in out_tracers)
    out_pvs = [None if pv is None else _unmapped_aval(params['axis_size'], pv)
               for pv in out_pvs_reduced]
    out = out_pv_consts + consts
    del consts, out_pv_consts
    master = self.master
    def todo(x):
      n = len(jaxpr.outvars)
      out_pv_consts, consts = x[:n], x[n:]
      trace = JaxprTrace(master, core.cur_sublevel())
      const_tracers = map(trace.new_instantiated_const, consts)
      # The `jaxpr` already contains the env_vars at start of invars
      lifted_jaxpr = convert_constvars_jaxpr(jaxpr)
      out_tracers = [JaxprTracer(trace, PartialVal((out_pv, out_pv_const)), None)
                     for out_pv, out_pv_const in zip(out_pvs, out_pv_consts)]
      new_params = dict(params,
                        mapped_invars=tuple([True] * len(const_tracers) +
                                            [False] * len(env)),
                        call_jaxpr=lifted_jaxpr)
      env_tracers = map(trace.full_raise, env)
      eqn = new_eqn_recipe(it.chain(const_tracers, env_tracers),
                           out_tracers, map_primitive, new_params)
      for t in out_tracers:
        t.recipe = eqn
      return out_tracers
    return out, todo

  def process_custom_jvp_call(self, prim, fun, jvp, tracers):
    # We form jaxprs using JaxprTraces for two distinct purposes: to stage
    # program representations completely out of the JAX system (e.g. for XLA
    # using jit or pmap), and to build a representation of a function that may
    # require further JAX transformations (e.g. in "initial-style" higher-order
    # primitives, like for control flow). In particular, in the latter case we
    # need custom differentiation rules to stick around, but in the former we do
    # not. This method call should only be reachable in the former case, and so
    # we check that the former case is indicated (with a StagingJaxprTrace) and
    # then drop the differentiation rules.
    assert self.master.trace_type is StagingJaxprTrace
    return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees):
    # See comment in the above process_custom_jvp_call method.
    assert self.master.trace_type is StagingJaxprTrace
    return fun.call_wrapped(*tracers)

# This subclass is used just for its type tag, which switches the behavior of
# process_call to stage out into the jaxpr any call primitives encountered
# (rather than doing partial evaluation into the call).
class StagingJaxprTrace(JaxprTrace):
  pass

def _mapped_aval(aval):
  if aval is core.abstract_unit:
    return aval
  elif isinstance(aval, ShapedArray):
    # might be raising abstraction level from Concrete here
    return ShapedArray(aval.shape[1:], aval.dtype)
  else:
    raise TypeError(aval)

def _unmapped_aval(size, aval):
  if aval is core.abstract_unit:
    return aval
  elif isinstance(aval, ShapedArray):
    return ShapedArray((size,) + aval.shape, aval.dtype)
  else:
    raise TypeError(aval)

map_primitives: Set[core.Primitive] = set()
custom_partial_eval_rules: Dict[core.Primitive, Callable] = {}
call_partial_eval_rules: Dict[core.Primitive, Callable] = {}


def partial_eval(f, trace, pvs: Sequence[Optional[AbstractValue]], instantiate=False):
  f = trace_to_subjaxpr(f, trace.master, instantiate)
  return partial_eval_wrapper(f, tuple(pvs))


@lu.transformation_with_aux
def partial_eval_wrapper(avals: Sequence[Optional[AbstractValue]], *consts):
  py_args = (map(PartialVal, zip(avals, consts)),)
  jaxpr, (out_pvals, consts, env) = yield py_args, {}
  out_pvs, out_consts = unzip2(out_pvals)
  out = tuple(out_consts) + tuple(consts)  # TODO: can consts be traced?
  yield out, (out_pvs, jaxpr, env)


def abstract_eval_fun(fun, *avals, **params):
  pvals_in = [PartialVal.unknown(a) for a in avals]
  _, pvals_out, _ = trace_to_jaxpr(lu.wrap_init(fun, params), pvals_in,
                                  instantiate=True, stage_out=True)
  avals_out, _ = unzip2(pvals_out)
  for aval_out in avals_out:
    assert isinstance(aval_out, AbstractValue)  # instantiate=True
  return avals_out


class JaxprTracer(Tracer):
  __slots__ = ['pval', 'recipe']

  def __init__(self, trace, pval: PartialVal, recipe):
    assert isinstance(pval, PartialVal)
    pv, const = pval
    if isinstance(const, Tracer) and const._trace.level >= trace.level:
      raise core.escaped_tracer_error(
          "Tracer from a higher level: {} in trace {}".format(const, trace))
    self._trace = trace
    self.pval = pval
    self.recipe = recipe

  def __repr__(self):
    return 'Traced<{}:{}>'.format(self.aval, self._trace)

  @property
  def aval(self):
    return self.pval.get_aval()

  @property
  def parents(self):
    if isinstance(self.recipe, JaxprEqnRecipe):
      return self.recipe.invars
    else:
      return []

  def ispure(self):
    return self.pval.is_known()

  def full_lower(self):
    known = self.pval.get_known()
    if known is not None:
      return core.full_lower(known)
    else:
      return self


# TODO(necula): this should return a TypedJaxpr
def trace_to_jaxpr(fun: lu.WrappedFun, pvals: Sequence[PartialVal],
                   instantiate=False, stage_out=False, bottom=False) \
    -> Tuple[Jaxpr, Sequence[PartialVal], Sequence[core.Value]]:
  """Traces a function into a Jaxpr, given PartialVals for inputs.

  Returns (`jaxpr`, `out_pvals`, `consts`).
  The `jaxpr` contains only the computation that depends on unknown inputs.
  The `out_pvals` are the PartialVal for the outputs. The intermediate
  values that depend only on known inputs and are needed to compute the output
  of `jaxpr` are in `consts` and are passed in as the constvars of
  the `jaxpr`. The handling of the known outputs depends on `instantiate`.

  For example, given `fun` defined as follows::

     def fun(ki, ui):  # ki will be a known input in this example
       ka = ki + 2
       kb = ka + 3
       return (kb, ui + ka)

  with `ki` the known PartialVal `1.`, and `ui` an unknown PartialVal. The only
  computation that depends on unknown inputs is `ui + ka` and will be the only
  computation in the body of the `jaxpr`. This computation depends on the
  known intermediate value `ka`, which will be computed statically. Currently,
  such constants are either embedded in the Jaxpr if they are scalars, or
  passed as a constvar to `jaxpr`, and then the value of the actual constant
  will be in `consts`:

  When `instantiate=False` we get::

     jaxpr =
      { lambda ka ; ki ui.
        let c = add ui ka
        in (*, c) }   # known outputs are `*`
     out_pvals = [known(6), unknown(ShapedArray)]  # the known outputs are known PartialVal
     consts = [3]  # the constant for `ka`

  When `instantiate=True` we get::

     jaxpr =
      { lambda ka kb ; ki ui.
        let c = add ui ka
        in (kb, c) }   # known output are explicit
     out_pvals = [abstract(ConcreteArray(6)), abstract(ShapedArray)]  # all are unknown PartialVal
     consts = [3, 6]  # values for `ka` and `kb` constvars
  """
  trace_type = StagingJaxprTrace if stage_out else JaxprTrace
  with new_master(trace_type, bottom=bottom) as master:
    fun = trace_to_subjaxpr(fun, master, instantiate)
    jaxpr, (out_pvals, consts, env) = fun.call_wrapped(pvals)
    assert not env
    del master

  return jaxpr, out_pvals, consts

@lu.transformation
def trace_to_subjaxpr(master: core.MasterTrace, instantiate: Union[bool, Sequence[bool]],
                      pvals: Sequence[PartialVal]):
  assert all([isinstance(pv, PartialVal) for pv in pvals]), pvals
  trace = JaxprTrace(master, core.cur_sublevel())
  in_tracers = map(trace.new_arg, pvals)
  ans = yield in_tracers, {}
  instantiate = [instantiate] * len(ans) if isinstance(instantiate, bool) else instantiate
  out_tracers = map(trace.full_raise, map(core.full_lower, ans))
  out_tracers = map(partial(instantiate_const_at, trace), instantiate, out_tracers)
  jaxpr, consts, env = tracers_to_jaxpr(in_tracers, out_tracers)
  out_pvals = [t.pval for t in out_tracers]
  del trace, in_tracers, out_tracers
  yield jaxpr, (out_pvals, consts, env)

def instantiate_const_at(trace, instantiate: bool, tracer):
  if instantiate:
    return trace.instantiate_const(trace.full_raise(tracer))
  else:
    return tracer


FreeVar = namedtuple('FreeVar', ['val'])
ConstVar = namedtuple('ConstVar', ['val'])
LambdaBinding = namedtuple('LambdaBinding', [])
JaxprEqnRecipe = namedtuple('JaxprEqnRecipe',
                            ['eqn_id', 'invars', 'outvars', 'primitive', 'params'])

def new_eqn_recipe(invars, outvars, primitive, params):
  """Constructs a new JaxEqnRecipe.

  Params:
    invars: the tracers for the primitive inputs.
    outvars: the tracers for the primitive outputs.
    primitive: the primitive.
    params: the primitive params
  """
  if primitive.call_primitive:
    # TODO(necula): move these checks to core.check_jaxpr, and call it
    # in more places.
    assert "call_jaxpr" in params
  return JaxprEqnRecipe(object(), tuple(invars), map(ref, outvars), primitive,
                        params)


def recipe_to_eqn(unused_var, getvar, recipe):
  _, in_tracers, out_tracer_refs, primitive, params = recipe
  out_tracers = [t_ref() for t_ref in out_tracer_refs]
  invars  = [getvar(t) for t in in_tracers]
  outvars = [unused_var() if t is None else getvar(t) for t in out_tracers]
  return new_jaxpr_eqn(invars, outvars, primitive, params)

def tracers_to_jaxpr(in_tracers, out_tracers):
  """Constructs Jaxpr given tracers for inputs and outputs.

  Params:
    in_tracers: the tracers that were created for the function inputs
    out_tracers: the tracers that were output by the function.

  Returns: a triple of a `Jaxpr`, a list of constant values corresponding to
    the `constvars` in the returned Jaxps, and a list of environment values.
    The vars for the environment values have been prepended to the Jaxpr's
    `invars`.
  """
  newvar = core.gensym('')
  t_to_var = {}
  def getvar(t):
    var = t_to_var.get(id(t))
    if var is None:
      aval = t.pval.get_aval() if not t.pval.is_known() else abstract_unit
      var = t_to_var[id(t)] = newvar(aval)
    return var
  sorted_tracers = toposort(out_tracers)
  invars = map(getvar, in_tracers)
  eqns = []
  env = {}
  consts = {}
  const_to_var = {}
  def getconstvar(c):
    var = const_to_var.get(id(c))
    if var is None:
      var = const_to_var[id(c)] = newvar(get_aval(c))
    return var
  processed_eqn_ids = set()
  for t in sorted_tracers:
    recipe = t.recipe
    if isinstance(recipe, JaxprEqnRecipe):
      if recipe.eqn_id not in processed_eqn_ids:
        eqns.append(recipe_to_eqn(lambda: newvar(core.abstract_unit), getvar, recipe))
        processed_eqn_ids.add(recipe.eqn_id)
    elif isinstance(recipe, LambdaBinding):
      if not any(t is in_tracer for in_tracer in in_tracers):
        raise core.escaped_tracer_error(
            "Tracer not among input tracers {}".format(t))
      assert in_tracers, "Lambda binding with no args"
    elif isinstance(recipe, FreeVar):
      env[getvar(t)] = recipe.val
    elif isinstance(recipe, ConstVar):
      v = t_to_var[id(t)] = getconstvar(recipe.val)
      consts[v] = recipe.val
    elif isinstance(recipe, Literal):
      t_to_var[id(t)] = recipe
    elif recipe is unit:
      t_to_var[id(t)] = unitvar
    else:
      raise TypeError(recipe)

  env_vars, env_vals = unzip2(env.items())
  const_vars, const_vals = unzip2(consts.items())
  # The env_vars are pre-pended to the invars
  jaxpr = Jaxpr(const_vars, list(it.chain(env_vars, invars)), list(map(getvar, out_tracers)), eqns)
  core.skip_checks or core.check_jaxpr(jaxpr)
  return jaxpr, const_vals, env_vals

@cache()
def convert_constvars_jaxpr(jaxpr):
  """Moves the constvars to the start of invars."""
  core.skip_checks or core.check_jaxpr(jaxpr)
  lifted_jaxpr = Jaxpr(constvars=(),
                       invars=jaxpr.constvars + jaxpr.invars,
                       outvars=jaxpr.outvars, eqns=jaxpr.eqns)
  core.skip_checks or core.check_jaxpr(lifted_jaxpr)
  return lifted_jaxpr

def partial_eval_jaxpr(jaxpr: TypedJaxpr, unknowns: Sequence[bool],
                       instantiate: bool) -> Tuple[TypedJaxpr, TypedJaxpr, Sequence[bool]]:
  """Specializes a Jaxpr given an indication of which inputs are known.

  Returns: (jaxpr_known, jaxpr_unknown, out_unknowns).

  `out_unknowns` specifies which outputs are unknown (depend on some unknown inputs).
  `jaxpr_known` takes the same inputs as `jaxpr`, ignores the unknown inputs,
  and performs *all* the computation in `jaxpr` that depends only on the known inputs.
  Outputs correspond to those of `jaxpr`, with the unknown ones replaced with `*`,
  appended with the known residuals (the intermediate computations in `jaxpr`
  that depend only on known inputs and that are needed to compute the unknown outputs).

  `jaxpr_unknown` takes the same inputs as `jaxpr` along with the known residuals
  computed by `jaxpr_known` and returns the same outputs as `jaxpr` with the known
  outputs replaced by `*`.

  Roughly, `jaxpr(ki, ui)` is decomposed assuming `ki` and `ui` are the known and respectively
  unknown inputs into:

     jaxpr(ki, ui) = let kout, _, kresidual = jaxpr_known(kin, *)
                     let _, uout = jaxpr_unknown(ki, ui, kresidual)
                     in (kout, uout)

  For example, if `jaxpr` is lambda ki, ui: let ka = ki + 2
                                            in (ki + 3, ui + ka)"
  then
    `jaxpr_known` = lambda ki, ui: let ka = ki + 2
                                    in (ki + 3, *, ka)
    'jaxpr_unknown` = lambda ki, ui, ka: (*, ui + ka)
  """
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))

  cell = []
  def fun(*vals):
    pvals = [PartialVal.unknown(aval) if uk else PartialVal.known(val)
             for aval, val, uk in zip(jaxpr.in_avals, vals, unknowns)]
    jaxpr_2, out_pvals_2, consts_2 = trace_to_jaxpr(f, pvals, instantiate=instantiate)
    out_pvs_2, out_consts_2 = unzip2(out_pvals_2)
    cell.append((out_pvs_2, jaxpr_2, len(consts_2)))
    return out_consts_2 + consts_2

  # For jaxpr_known we pass core.unit for the unknown inputs, and known PartialVal for the
  # known inputs.
  pvals = [PartialVal.unknown(abstract_unit) if uk else PartialVal.unknown(aval)
           for aval, uk in zip(jaxpr.in_avals, unknowns)]
  jaxpr_1, out_pvals, consts_1 = trace_to_jaxpr(lu.wrap_init(fun), pvals, instantiate=True)
  (out_pvs_2, jaxpr_2, num_res), = cell
  assert len(jaxpr_2.constvars) == num_res

  #   jaxpr :: a -> b
  # jaxpr_1 :: a1 -> [b1, res]
  # jaxpr_2 :: res | a2 -> b2
  # jaxpr_2 :: [a2, res] -> b2
  jaxpr_2 = convert_constvars_jaxpr(jaxpr_2)
  jaxpr_2.invars = jaxpr_2.invars[num_res:] + jaxpr_2.invars[:num_res]
  for var, unknown in zip(jaxpr_2.invars[:len(unknowns)], unknowns):
    if not unknown:
      var.aval = abstract_unit

  uk_out = [pv is not None for pv in out_pvs_2]

  in_avals_1, in_avals_2 = unzip2(map(_split_aval, unknowns, jaxpr.in_avals))
  out_avals_1, out_avals_2 = unzip2(map(_split_aval, uk_out, jaxpr.out_avals))
  # out_avals_1 and in_avals_2 need the residuals added
  out_pvs, _ = unzip2(out_pvals)
  res_avals = out_pvs[len(jaxpr.out_avals):]
  assert len(res_avals) == num_res
  out_avals_1 = out_avals_1 + res_avals
  in_avals_2 = in_avals_2 + res_avals

  typed_jaxpr_1 = TypedJaxpr(jaxpr_1, consts_1, in_avals_1, out_avals_1)
  typed_jaxpr_2 = TypedJaxpr(jaxpr_2, (), in_avals_2, out_avals_2)
  return typed_jaxpr_1, typed_jaxpr_2, uk_out

def _split_aval(unknown, aval):
  return (abstract_unit, aval) if unknown else (aval, abstract_unit)


remat_call_p = core.Primitive('remat_call')
remat_call_p.call_primitive = True
remat_call = partial(core.call_bind, remat_call_p)
remat_call_p.def_custom_bind(remat_call)
remat_call_p.def_impl(core.call_impl)
remat_call_p.multiple_results = True

def _remat_partial_eval(trace, _, f, tracers, params):
  concrete = params['concrete']

  # Unlike JaxprTrace.process_call, we want to form a jaxpr for the entirety of
  # the function being called, not just for the unknown parts. To do that, we
  # instantiate all the input tracers as constants in the jaxpr being formed.
  # Those tracers might have concrete avals, and doing abstract interpretation
  # on concrete avals engenders a tradeoff: it allows data-dependent Python
  # control flow to work, but it can in some cases lead to redundant FLOPs (done
  # both in the `bind` call below and the `core.jaxpr_as_fun` call). We use the
  # `concrete` parameter to switch this behavior, and if `concrete` is False
  # then we raise the avals to the Shaped level.
  if concrete:
    instantiated_tracers = map(trace.instantiate_const, tracers)
  else:
    instantiated_tracers = map(trace.instantiate_const_abstracted, tracers)

  # Using the instantiated tracers, run call_bind like JaxprTrace.process_call.
  in_pvs, in_consts = unzip2(t.pval for t in instantiated_tracers)
  fun, aux = partial_eval(f, trace, in_pvs)
  with core.initial_style_staging():
    out_flat = remat_call_p.bind(fun, *in_consts, **params)
  out_pvs, jaxpr, env = aux()
  env = map(trace.full_raise, env)
  out_pval_consts1, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
  out_pvals1 = [PartialVal((pv, const)) for pv, const in zip(out_pvs, out_pval_consts1)]

  # Since we traced with everything marked as unknown, but we need to know which
  # outputs are known/unknown, we use partial_eval_jaxpr to get out_unknowns.

  in_avals = ([raise_to_shaped(t.pval.get_aval()) for t in env]
              + [raise_to_shaped(pv) for pv in in_pvs])
  out_avals = [raise_to_shaped(pv if pv is not None
                               else abstract_unit if var is unitvar
                               else get_aval(var.val) if type(var) is Literal
                               else get_aval(const))
               for var, pv, const in zip(jaxpr.outvars, out_pvs, out_pval_consts1)]
  typed_jaxpr = core.TypedJaxpr(jaxpr, consts, in_avals, out_avals)
  in_unknowns = [t.pval[0] is not None for t in it.chain(env, tracers)]
  jaxpr_1, jaxpr_2, out_unknowns = partial_eval_jaxpr(typed_jaxpr, in_unknowns, False)
  num_res = len(jaxpr_1.out_avals) - len(jaxpr_2.out_avals)

  # First, we prune the jaxpr to be staged out not to have too many outputs.
  typed_jaxpr = _dce_jaxpr(typed_jaxpr, out_unknowns)

  # Next, we need values for the outputs that should be known. Since consts
  # weren't passed through Python for evaluation, we need to evaluate jaxpr_1,
  # minus the residual outputs that we don't need. When `concrete=True`, as an
  # optimization we can avoid redoing *some* redundant FLOPs, namely those that
  # produced concrete avals at the output, simply by using those as computed
  # values. For the use case of reverse-mode ad in op-by-op ("eager mode")
  # evaluation, all the primal outputs should be concrete (thus not recomputed).
  to_compute = [not uk and type(pv) is not ConcreteArray
                for uk, pv in zip(out_unknowns, out_pvs)]
  jaxpr_1_primals = _dce_jaxpr(jaxpr_1, to_compute + [False] * num_res)
  _, in_consts = unzip2(t.pval for t in it.chain(env, tracers))
  out_pval_consts2 = core.jaxpr_as_fun(jaxpr_1_primals)(*in_consts)[:-num_res or None]
  out_pvals = map(_reconstruct_pval, out_pvals1, out_pval_consts2, out_unknowns)

  # Now that we have out_pvals, the rest is just like JaxprTrace.process_call.
  instantiated_tracers = env + instantiated_tracers
  const_tracers = map(trace.new_instantiated_const, consts)
  lifted_jaxpr = convert_constvars_jaxpr(typed_jaxpr.jaxpr)
  out_tracers = [JaxprTracer(trace, out_pval, None) for out_pval in out_pvals]
  new_params = dict(params, call_jaxpr=lifted_jaxpr)
  eqn = new_eqn_recipe(tuple(it.chain(const_tracers, instantiated_tracers)),
                       out_tracers, remat_call_p, new_params)
  for t in out_tracers: t.recipe = eqn
  return out_tracers
call_partial_eval_rules[remat_call_p] = _remat_partial_eval

def _dce_jaxpr(typed_jaxpr, outputs):
  # This dead-code elimination is pretty rudimentary, and in particular doesn't
  # nontrivially DCE through scan, call, or other higher-order primitives.
  # TODO(mattjj): better DCE
  jaxpr = typed_jaxpr.jaxpr
  outvars, out_avals = jaxpr.outvars, typed_jaxpr.out_avals
  out_pairs = [(var, aval) if output else (unitvar, core.abstract_unit)
              for var, aval, output in zip(outvars, out_avals, outputs)]
  new_outvars, new_out_avals = unzip2(out_pairs)

  needed_vars = {v for v in new_outvars if type(v) is not Literal}
  new_eqns = []
  for eqn in jaxpr.eqns[::-1]:
    if set(eqn.outvars) & needed_vars:
      new_eqns.append(eqn)
      needed_vars.update(v for v in eqn.invars if type(v) is not Literal)
  new_eqns = new_eqns[::-1]
  new_jaxpr = core.Jaxpr(jaxpr.constvars, jaxpr.invars,
                         new_outvars, new_eqns)
  return core.TypedJaxpr(new_jaxpr, typed_jaxpr.literals, typed_jaxpr.in_avals,
                         new_out_avals)

def _reconstruct_pval(pval1: PartialVal, const2: core.Value, unknown: bool):
  pv1, _ = pval1
  if unknown or pval1.is_known():
    return pval1
  else:
    if type(pv1) is ConcreteArray:
      return PartialVal.known(pv1.val)
    else:
      return PartialVal.known(const2)


def move_binders_to_front(typed_jaxpr: TypedJaxpr, to_move: Sequence[bool]) -> TypedJaxpr:
  """Reorder the `invars` to move to front the ones for which `to_move` is True."""
  assert not typed_jaxpr.jaxpr.constvars
  assert len(typed_jaxpr.in_avals) == len(to_move)
  new_invars = _move_to_front(typed_jaxpr.jaxpr.invars, to_move)
  new_jaxpr = core.Jaxpr((), new_invars, typed_jaxpr.jaxpr.outvars,
                         typed_jaxpr.jaxpr.eqns)
  new_in_avals = _move_to_front(typed_jaxpr.in_avals, to_move)
  new_typed_jaxpr = core.TypedJaxpr(new_jaxpr, typed_jaxpr.literals,
                                    new_in_avals, typed_jaxpr.out_avals)
  return new_typed_jaxpr

def _move_to_front(lst: Sequence, to_move: Sequence[bool]) -> Sequence:
  return ([elt for elt, move in zip(lst, to_move) if move] +
          [elt for elt, move in zip(lst, to_move) if not move])
