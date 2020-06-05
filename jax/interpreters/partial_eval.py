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

from typing import Any, List

import itertools as it
from collections import namedtuple
from typing import (Callable, Dict, NamedTuple, Optional, Sequence,
                    Set, Tuple, Type, Union, cast)
from weakref import ref

import numpy as onp

from .. import core
from .. import linear_util as lu
from ..abstract_arrays import ConcreteArray, raise_to_shaped
from ..ad_util import Zero
from ..util import (unzip2, safe_zip, safe_map, toposort, partial, split_list,
                    wrap_name, cache, curry)
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
      assert isinstance(const, core.Tracer) or type(const) is Zero or core.valid_jaxtype(const), xs
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

  def is_known(self) -> bool:
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


# We form Jaxprs using `JaxprTrace` for three distinct purposes:
#  (1) to stage program representations completely out of the JAX system
#      (e.g. for XLA using jit or pmap). In this case we are using the
#      `StagingJaxprTrace` subclass.
#  (3) to linearize a function for reverse-mode AD. In this case we are
#      using the `JaxprTrace` subclass.
#  (2) to build a representation of a function that may require further JAX
#     transformations (e.g. in "initial-style" higher-order primitives, like
#     for control flow). In this case we use the `JaxprTrace` class.
class JaxprTrace(Trace):
  def pure(self, val) -> 'JaxprTracer':
    return self.new_const(val)

  def lift(self, val) -> 'JaxprTracer':
    return self.new_const(val)

  def sublift(self, val) -> 'JaxprTracer':
    return JaxprTracer(self, val.pval, FreeVar(val))

  def new_const(self, val) -> 'JaxprTracer':
    if isinstance(val, Tracer) and val._trace.level == self.level:
      raise Exception
    return JaxprTracer(self, PartialVal.known(val), unit)

  def new_instantiated_literal(self, val) -> 'JaxprTracer':
    return JaxprTracer(self, PartialVal.unknown(get_aval(val)), Literal(val))

  def new_instantiated_const(self, val) -> 'JaxprTracer':
    return JaxprTracer(self, PartialVal.unknown(get_aval(val)), ConstVar(val))

  def new_arg(self, pval: PartialVal) -> 'JaxprTracer':
    const = pval.get_known()
    if const is None:
      return JaxprTracer(self, pval, LambdaBinding())
    else:
      return self.new_const(const)

  def instantiate_const(self, tracer) -> Tracer:
    const = tracer.pval.get_known()
    if const is None:
      return tracer
    else:
      if type(const) in core.literalable_types and onp.shape(const) == ():
        return self.new_instantiated_literal(const)
      else:
        return self.new_instantiated_const(const)

  def instantiate_const_abstracted(self, tracer) -> 'JaxprTracer':
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
    """By default, if all the input tracers are known, then execute the primitive
    and all the ouputs are known. Otherwise, all the outputs are unknown."""
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

  def process_call(self, primitive, f: lu.WrappedFun, tracers, params):
    name = params.get('name', f.__name__)
    if (self.master.trace_type is StagingJaxprTrace
        and primitive in staged_out_calls):
      tracers = map(self.instantiate_const_abstracted, tracers)
    params = dict(params, name=name)

    if primitive in call_partial_eval_rules:
      return call_partial_eval_rules[primitive](self, primitive, f, tracers, params)

    @curry
    def modify_aval(modify, args):
      pval, is_mapped = args
      if pval.is_known() or not is_mapped:
        return pval
      return PartialVal((modify(params['axis_size'], pval[0]), pval[1]))

    in_pvals = [t.pval for t in tracers]
    if primitive.map_primitive:
      in_pvals = map(modify_aval(core.mapped_aval), zip(in_pvals, params['mapped_invars']))
    jaxpr, out_pvals, consts, env_tracers = self.partial_eval(
        f, in_pvals, partial(primitive.bind, **params))
    if primitive.map_primitive:
      out_pvals = map(modify_aval(core.unmapped_aval),
                      [(pval, True) for pval in out_pvals])

    # Don't bother if the traced jaxpr is trivial. Simply evaluate it in here.
    # XXX: We don't allow this fast path for map primitives, because this simplification might
    #      e.g. reduce the number of required devices if someone pmaps an identity function.
    if not primitive.map_primitive and not jaxpr.eqns:
      env = {core.unitvar: core.unit}
      map(env.setdefault, jaxpr.invars, (*env_tracers, *tracers))
      map(env.setdefault, jaxpr.constvars, consts)
      return [v.val if type(v) is Literal
              else pval.get_known() if pval.is_known()
              else env[v]
              for v, pval in zip(jaxpr.outvars, out_pvals)]

    # Skip known invars and outvars, and lift constants as regular invars
    in_knowns = tuple(t.pval.is_known() for t in it.chain(env_tracers, tracers))
    out_unknowns = tuple(not pval.is_known() for pval in out_pvals)
    jaxpr = _drop_invars(jaxpr, in_knowns)
    jaxpr = _dce_untyped_jaxpr(jaxpr, out_unknowns, drop_outputs=True)
    lifted_jaxpr = convert_constvars_jaxpr(jaxpr)

    # Known tracers get propagated as if they were constants
    known_tracers_out = [self.new_const(pval.get_known()) for pval in out_pvals if pval.is_known()]

    # Unknown tracers need to have the jaxpr set up as their recipe
    unknown_tracers_out = [JaxprTracer(self, pval, None) for pval in out_pvals if not pval.is_known()]
    unknown_tracers_in = [t for t in tracers if not t.pval.is_known()]
    const_tracers = map(self.new_instantiated_const, consts)
    new_params = dict(params, call_jaxpr=lifted_jaxpr)
    if 'donated_invars' in params:
      new_donated_invars = ((False,) * len(const_tracers) +
                            (False,) * len(env_tracers) +
                            tuple(v for v, t in zip(params['donated_invars'], tracers) if not t.pval.is_known()))
      new_params['donated_invars'] = new_donated_invars
    if primitive.map_primitive:
      new_mapped_invars = ((True,) * len(const_tracers) +
                           (False,) * len(env_tracers) +
                           tuple(v for v, t in zip(params['mapped_invars'], tracers) if not t.pval.is_known()))
      new_params['mapped_invars'] = new_mapped_invars
    eqn = new_eqn_recipe(tuple(it.chain(const_tracers, env_tracers, unknown_tracers_in)),
                         unknown_tracers_out, primitive, new_params)
    for t in unknown_tracers_out:
      t.recipe = eqn

    return _zip_knowns(known_tracers_out, unknown_tracers_out, out_unknowns)

  def post_process_call(self, call_primitive, out_tracers, params):
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
      invars = tuple(it.chain(const_tracers, env_tracers))
      new_params = dict(params, call_jaxpr=lifted_jaxpr)
      if 'donated_invars' in params:
        new_params['donated_invars'] = (False,) * len(invars)
      # The `jaxpr` already contains the env_vars at start of invars
      eqn = new_eqn_recipe(invars, out_tracers, call_primitive, new_params)
      for t in out_tracers:
        t.recipe = eqn
      return out_tracers
    return out, todo

  def process_map(self, map_primitive, f: lu.WrappedFun, tracers, params):
    return self.process_call(map_primitive, f, tracers, params)

  def post_process_map(self, map_primitive, out_tracers, params):
    jaxpr, consts, env = tracers_to_jaxpr([], out_tracers)
    out_pvs_reduced, out_pv_consts = unzip2(t.pval for t in out_tracers)
    out_pvs = [None if pv is None
               else core.unmapped_aval(params['axis_size'], pv)
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
      new_donated_invars = (False,) * (len(const_tracers) + len(env))
      new_mapped_invars = (True,) * len(const_tracers) + (False,) * len(env)
      new_params = dict(params, donated_invars=tuple(new_donated_invars),
                        mapped_invars=tuple(new_mapped_invars),
                        call_jaxpr=lifted_jaxpr)
      env_tracers = map(trace.full_raise, env)
      eqn = new_eqn_recipe(tuple(it.chain(const_tracers, env_tracers)),
                           out_tracers, map_primitive, new_params)
      for t in out_tracers:
        t.recipe = eqn
      return out_tracers
    return out, todo

  def process_custom_jvp_call(self, prim, fun, jvp, tracers):
    # See comment at top of `JaxprTrace`. This method should be reachable
    # only when we stage out, and in that case we drop the custom differentiation
    # rules, because we do not need them.
    assert self.master.trace_type is StagingJaxprTrace
    return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees):
    # See comment in the above process_custom_jvp_call method.
    assert self.master.trace_type is StagingJaxprTrace
    return fun.call_wrapped(*tracers)

  def partial_eval(self, f: lu.WrappedFun, pvals: Sequence[PartialVal],
                   app: Callable[[lu.WrappedFun, Tuple[core.Value, ...]], Tuple[core.Value]]):
    """Partially evaluate f on a sequence of PartialVals."""
    in_avals, in_consts = unzip2(pvals)
    f = trace_to_subjaxpr(f, self.master, False)
    f, aux = partial_eval_wrapper(f, tuple(in_avals))
    out_flat, (out_avals, jaxpr, env) = app(f, *in_consts), aux()
    out_consts, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
    out_pvs = map(PartialVal, zip(out_avals, out_consts))
    env_tracers = map(self.full_raise, env)
    return jaxpr, out_pvs, consts, env_tracers

# This subclass is used just for its type tag (see comment for `JaxprTrace`)
# This switches the behavior of process_call to stage out into the jaxpr any
# call primitives encountered (rather than doing partial evaluation into the call).
class StagingJaxprTrace(JaxprTrace):
  pass


@lu.transformation_with_aux
def partial_eval_wrapper(avals: Sequence[Optional[AbstractValue]], *consts):
  py_args = (map(PartialVal, zip(avals, consts)),)
  jaxpr, (out_pvals, consts, env) = yield py_args, {}
  out_pvs, out_consts = unzip2(out_pvals)
  out = tuple(out_consts) + tuple(consts)  # TODO: can consts be traced?
  yield out, (out_pvs, jaxpr, env)


custom_partial_eval_rules: Dict[core.Primitive, Callable] = {}
call_partial_eval_rules: Dict[core.Primitive, Callable] = {}
staged_out_calls: Set[core.Primitive] = set()


def abstract_eval_fun(fun, *avals, **params):
  pvals_in = [PartialVal.unknown(a) for a in avals]
  _, pvals_out, _ = trace_to_jaxpr(lu.wrap_init(fun, params), pvals_in,
                                  instantiate=True, stage_out=True)
  avals_out, _ = unzip2(pvals_out)
  for aval_out in avals_out:
    assert isinstance(aval_out, AbstractValue)  # instantiate=True
  return avals_out


JaxprTracerRecipe = Union['JaxprEqnRecipe', 'LambdaBinding', 'FreeVar',
                          'ConstVar', Literal, core.Unit]

class JaxprTracer(Tracer):
  __slots__ = ['pval', 'recipe']

  def __init__(self, trace: JaxprTrace, pval: PartialVal,
               recipe: Optional[JaxprTracerRecipe]):
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
  def aval(self) -> AbstractValue:
    return self.pval.get_aval()

  @property
  def parents(self) -> Sequence['JaxprTracer']:
    if isinstance(self.recipe, JaxprEqnRecipe):
      return self.recipe.invars
    else:
      return []

  def full_lower(self):
    known = self.pval.get_known()
    if known is not None:
      return core.full_lower(known)
    else:
      return self

  def is_known(self):
      return self.pval.is_known()

# TODO(necula): this should return a TypedJaxpr
# TODO(necula): remove stage_out, replace trace_type=pe.StagingJaxprTrace
def trace_to_jaxpr(fun: lu.WrappedFun, pvals: Sequence[PartialVal],
                   instantiate: Union[bool, Sequence[bool]] = False,
                   stage_out=False, bottom=False,
                   trace_type: Optional[Type[Trace]] = None) \
    -> Tuple[Jaxpr, Tuple[PartialVal, ...], Tuple[core.Value, ...]]:
  """Traces a function into a Jaxpr, given PartialVals for inputs.

  `trace_type` can be one of `StagingJaxprTrace` or `JaxprTrace` (see
  comments for that class).

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
  trace_type = trace_type or (StagingJaxprTrace if stage_out else JaxprTrace)
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

def instantiate_const_at(trace: JaxprTrace, instantiate: bool, tracer):
  if instantiate:
    return trace.instantiate_const(trace.full_raise(tracer))
  else:
    return tracer


FreeVar = namedtuple('FreeVar', ['val'])
ConstVar = namedtuple('ConstVar', ['val'])
LambdaBinding = namedtuple('LambdaBinding', [])
class JaxprEqnRecipe(NamedTuple):
  eqn_id: object
  invars: Sequence[JaxprTracer]
  outvars: 'Sequence[ref[JaxprTracer]]'
  primitive: core.Primitive
  params: Dict[str, Any]

def new_eqn_recipe(invars: Sequence[JaxprTracer],
                   outvars: Sequence[JaxprTracer],
                   primitive: core.Primitive,
                   params: Dict[str, Any]) -> JaxprEqnRecipe:
  """Constructs a new JaxEqnRecipe.

  Params:
    invars: the tracers for the primitive inputs.
    outvars: the tracers for the primitive outputs.
    primitive: the primitive.
    params: the primitive params
  """
  # TODO(necula): move these checks to core.check_jaxpr, and call in more places
  if primitive.call_primitive or primitive.map_primitive:
    assert "call_jaxpr" in params
  if primitive.map_primitive:
    assert "mapped_invars" in params
    assert "donated_invars" in params
  return JaxprEqnRecipe(object(), tuple(invars), map(ref, outvars), primitive,
                        params)


def recipe_to_eqn(unused_var: Callable[[], core.Var],
                  getvar: Callable[[JaxprTracer], core.Atom],
                  recipe: JaxprEqnRecipe) -> core.JaxprEqn:
  _, in_tracers, out_tracer_refs, primitive, params = recipe
  out_tracers = [t_ref() for t_ref in out_tracer_refs]
  invars  = [getvar(t) for t in in_tracers]
  outvars = [unused_var() if t is None else cast(core.Var, getvar(t))
             for t in out_tracers]
  return new_jaxpr_eqn(invars, outvars, primitive, params)

def tracers_to_jaxpr(
  in_tracers: List[JaxprTracer],
  out_tracers: List[JaxprTracer]
  ) -> Tuple[Jaxpr, Tuple[Any, ...], Tuple[Any, ...]]:
  """Constructs Jaxpr given tracers for inputs and outputs.

  Params:
    in_tracers: the tracers that were created for the function inputs
    out_tracers: the tracers that were output by the function.

  Returns: a triple of a `Jaxpr`, a list of constant values corresponding to
    the `constvars` in the returned Jaxps, and a list of environment values.
    The vars for the environment values have been prepended to the Jaxpr's
    `invars`.
  """
  newvar = core.gensym()
  t_to_var: Dict[int, core.Atom] = {}
  def getvar(t: JaxprTracer) -> core.Atom:
    var = t_to_var.get(id(t))
    if var is None:
      aval = t.pval.get_aval() if not t.pval.is_known() else abstract_unit
      var = t_to_var[id(t)] = newvar(aval)
    return var
  sorted_tracers = toposort(out_tracers)
  invars = map(getvar, in_tracers)
  eqns: List[core.JaxprEqn] = []
  env: Dict[core.Var, Any] = {}
  consts: Dict[core.Var, Any] = {}
  const_to_var: Dict[int, core.Var] = {}
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
        eqns.append(recipe_to_eqn(lambda: core.dropvar, getvar, recipe))
        processed_eqn_ids.add(recipe.eqn_id)
    elif isinstance(recipe, LambdaBinding):
      if not any(t is in_tracer for in_tracer in in_tracers):
        raise core.escaped_tracer_error(
            "Tracer not among input tracers {}".format(t))
      assert in_tracers, "Lambda binding with no args"
    elif isinstance(recipe, FreeVar):
      env[cast(core.Var, getvar(t))] = recipe.val
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
def convert_constvars_jaxpr(jaxpr: Jaxpr):
  """Moves the constvars to the start of invars."""
  core.skip_checks or core.check_jaxpr(jaxpr)
  lifted_jaxpr = Jaxpr(constvars=(),
                       invars=jaxpr.constvars + jaxpr.invars,
                       outvars=jaxpr.outvars, eqns=jaxpr.eqns)
  core.skip_checks or core.check_jaxpr(lifted_jaxpr)
  return lifted_jaxpr

def partial_eval_jaxpr(jaxpr: TypedJaxpr, unknowns: Sequence[bool],
                       instantiate: Union[bool, Sequence[bool]],
                       trace_type: Optional[Type[core.Trace]]
                       ) -> Tuple[TypedJaxpr, TypedJaxpr, Sequence[bool]]:
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
    jaxpr_2, out_pvals_2, consts_2 = trace_to_jaxpr(f, pvals, instantiate=instantiate,
                                                    trace_type=trace_type)
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

def _split_aval(unknown: bool, aval: AbstractValue) -> Tuple[AbstractValue, AbstractValue]:
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
  in_pvals = [t.pval for t in instantiated_tracers]
  with core.initial_style_staging():
    jaxpr, eval_out_pvals, consts, env_tracers = trace.partial_eval(
      f, in_pvals, partial(remat_call_p.bind, **params))

  # Since we traced with everything marked as unknown, but we need to know which
  # outputs are known/unknown, we use partial_eval_jaxpr to get out_unknowns.

  in_avals = ([raise_to_shaped(t.pval.get_aval()) for t in env_tracers]
              + [raise_to_shaped(pval.get_aval()) for pval in in_pvals])
  out_avals = [raise_to_shaped(abstract_unit if var is unitvar
                               else get_aval(var.val) if type(var) is Literal
                               else pval.get_aval())
               for var, pval in zip(jaxpr.outvars, eval_out_pvals)]
  typed_jaxpr = core.TypedJaxpr(jaxpr, consts, in_avals, out_avals)
  in_unknowns = [not t.is_known() for t in it.chain(env_tracers, tracers)]
  jaxpr_known, jaxpr_unknown, out_unknowns = partial_eval_jaxpr(typed_jaxpr, in_unknowns,
                                                                instantiate=False,
                                                                trace_type=trace.master.trace_type)
  out_knowns = [not b for b in out_unknowns]

  # First, we prune the jaxpr to be staged out not to have too many outputs.
  typed_jaxpr = _dce_jaxpr(typed_jaxpr, out_unknowns, drop_outputs=True)

  out_known_pvals, out_unknown_pvals = _partition_knowns(eval_out_pvals, out_unknowns)

  # Next, we need values for the outputs that should be known. Since consts
  # weren't passed through Python for evaluation, we need to evaluate jaxpr_known,
  # minus the residual outputs that we don't need. When `concrete=True`, as an
  # optimization we can avoid redoing *some* redundant FLOPs, namely those that
  # produced concrete avals at the output, simply by using those as computed
  # values. For the use case of inverse-mode ad in op-by-op ("eager mode")
  # evaluation, all the primal outputs should be concrete (thus not recomputed).
  to_compute = [type(pval[0]) is not ConcreteArray
                for uk, pval in zip(out_unknowns, eval_out_pvals)
                if not uk]
  num_outputs = len(jaxpr_unknown.out_avals)
  num_res = len(jaxpr_known.out_avals) - num_outputs
  jaxpr_known_nores = _dce_jaxpr(jaxpr_known, out_knowns + [False] * num_res, drop_outputs=True)
  jaxpr_known_comp = _dce_jaxpr(jaxpr_known_nores, to_compute)
  _, in_consts = unzip2(t.pval for t in it.chain(env_tracers, tracers))
  reconstructed_consts = core.jaxpr_as_fun(jaxpr_known_comp)(*in_consts)
  out_known_pvals = map(_reconstruct_pval, out_known_pvals, reconstructed_consts)

  # Now that we have out_pvals, the rest is similar to JaxprTrace.process_call.
  # Known outputs should keep propagating as constants
  assert all(pv.is_known() for pv in out_known_pvals)
  known_output_tracers = [trace.new_const(pval.get_known()) for pval in  out_known_pvals]

  # Unknown outputs get wrapped in tracers with the appropriate recipe, as in JaxprTrace.process_call
  const_tracers = map(trace.new_instantiated_const, consts)
  unknown_output_tracers = [JaxprTracer(trace, out_pval, None) for out_pval in out_unknown_pvals]
  lifted_jaxpr = convert_constvars_jaxpr(typed_jaxpr.jaxpr)
  new_params = dict(params, call_jaxpr=lifted_jaxpr)
  eqn = new_eqn_recipe(tuple(it.chain(const_tracers, env_tracers, instantiated_tracers)),
                       unknown_output_tracers,
                       remat_call_p,
                       new_params)
  for t in unknown_output_tracers: t.recipe = eqn

  return _zip_knowns(known_output_tracers, unknown_output_tracers, out_unknowns)
call_partial_eval_rules[remat_call_p] = _remat_partial_eval

def _partition_knowns(l, unknowns):
  return ([e for e, unknown in zip(l, unknowns) if not unknown],
          [e for e, unknown in zip(l, unknowns) if unknown])

def _zip_knowns(kl, ul, unknowns):
  ul_it = iter(ul)
  kl_it = iter(kl)
  return [next(ul_it) if unknown else next(kl_it) for unknown in unknowns]


def _dce_jaxpr(typed_jaxpr: TypedJaxpr, outputs: Sequence[bool], drop_outputs=False) -> TypedJaxpr:
  if drop_outputs:
    new_out_avals = [aval for aval, output in zip(typed_jaxpr.out_avals, outputs) if output]
  else:
    new_out_avals = [aval if output else core.abstract_unit
                     for aval, output in zip(typed_jaxpr.out_avals, outputs)]
  new_jaxpr = _dce_untyped_jaxpr(typed_jaxpr.jaxpr, tuple(outputs), drop_outputs)
  return core.TypedJaxpr(new_jaxpr, typed_jaxpr.literals, typed_jaxpr.in_avals,
                         new_out_avals)

@cache()
def _dce_untyped_jaxpr(jaxpr: Jaxpr, outputs: Tuple[bool, ...], drop_outputs=False) -> Jaxpr:
  # This dead-code elimination is pretty rudimentary, and in particular doesn't
  # nontrivially DCE through scan, call, or other higher-order primitives.
  # TODO(mattjj): better DCE
  if drop_outputs:
    new_outvars = [var for var, output in zip(jaxpr.outvars, outputs) if output]
  else:
    new_outvars = [var if output else unitvar
                   for var, output in zip(jaxpr.outvars, outputs)]

  needed_vars = {v for v in new_outvars if type(v) is not Literal}
  new_eqns = []
  for eqn in jaxpr.eqns[::-1]:
    if set(eqn.outvars) & needed_vars:
      new_eqns.append(eqn)
      needed_vars.update(v for v in eqn.invars if type(v) is not Literal)
  new_eqns = new_eqns[::-1]
  return core.Jaxpr(jaxpr.constvars, jaxpr.invars,
                    new_outvars, new_eqns)

@cache()
def _drop_invars(jaxpr: Jaxpr, drop: Tuple[bool, ...]):
  return core.Jaxpr(jaxpr.constvars, [v for v, d in zip(jaxpr.invars, drop) if not d],
                    jaxpr.outvars, jaxpr.eqns)


def _reconstruct_pval(pval1: PartialVal, const2: core.Value):
  pv1, _ = pval1
  if pval1.is_known():
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
