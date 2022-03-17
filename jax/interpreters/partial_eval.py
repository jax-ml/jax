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

from collections import namedtuple
import contextlib
import functools
from functools import partial
import inspect
import itertools as it
import operator as op
from typing import (Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple,
                    List, Union, Set, cast)
from weakref import ref

import numpy as np

from jax import core
from jax._src import dtypes
from jax import linear_util as lu
from jax._src import profiler
from jax._src.ad_util import Zero
from jax._src.api_util import flattened_fun_in_tree, flatten_fun_nokwargs
from jax._src.tree_util import (PyTreeDef, treedef_tuple, tree_unflatten,
                                tree_leaves)
from jax._src.util import (unzip2, safe_zip, safe_map, toposort, split_list,
                           merge_lists, partition_list, cache, OrderedSet,
                           as_hashable_function)
from jax.core import (Trace, Tracer, Jaxpr, Literal, get_aval, AbstractValue,
                      unit, unitvar, abstract_unit, ClosedJaxpr, new_jaxpr_eqn,
                      ConcreteArray, raise_to_shaped, Var, Atom, JaxprEqn,
                      Primitive, DShapedArray, mapped_aval, unmapped_aval)
from jax._src import source_info_util
from jax.config import config

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
    if config.jax_enable_checks:
      # type checks
      assert isinstance(pv, (AbstractValue, type(None))), xs
      assert isinstance(const, core.Tracer) or type(const) is Zero or core.valid_jaxtype(const), xs
      # invariant checks
      if isinstance(pv, AbstractValue):
        assert get_aval(const) == core.abstract_unit, xs
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
    """Get AbstractValue directly (if unknown) or from the constant (known)."""
    known = self.get_known()
    if known is not None:
      return get_aval(known)
    else:
      return self[0]

  def merge_with_known(self, val: core.Value) -> core.Value:
    """Either the stored known value, or the given 'val'."""
    known = self.get_known()
    return known if known is not None else val
    breakpoint()


class JaxprTrace(Trace):

  def __init__(self, *args, name_stack: source_info_util.NameStack):
    super().__init__(*args)
    self.name_stack = name_stack

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
    aval = get_aval(val)
    return JaxprTracer(self, PartialVal.unknown(aval),
                       Literal(val, raise_to_shaped(aval)))

  def new_instantiated_const(self, val) -> 'JaxprTracer':
    return JaxprTracer(self, PartialVal.unknown(get_aval(val)), ConstVar(val))

  def new_arg(self, pval: PartialVal) -> 'JaxprTracer':
    const = pval.get_known()
    # XXX: Think twice before changing this constant argument pruning!
    # This has really important consequences for partial_eval_jaxpr.
    # Most importantly, this guarantees that the unknown jaxpr never uses
    # known inputs (if it needs them, then they get passed through residuals).
    if const is None:
      return JaxprTracer(self, pval, LambdaBinding())
    else:
      return self.new_const(const)

  def instantiate_const(self, tracer) -> Tracer:
    const = tracer.pval.get_known()
    if const is None:
      return tracer
    else:
      if type(const) in core.literalable_types and np.shape(const) == ():
        return self.new_instantiated_literal(const)
      else:
        return self.new_instantiated_const(const)

  def instantiate_const_abstracted(self, tracer) -> 'JaxprTracer':
    const = tracer.pval.get_known()
    if const is None:
      return tracer
    else:
      aval = raise_to_shaped(get_aval(const), np.isscalar(const))
      return JaxprTracer(self, PartialVal.unknown(aval), ConstVar(const))

  def process_primitive(self, primitive, tracers, params):
    if primitive in custom_partial_eval_rules:
      return custom_partial_eval_rules[primitive](self, *tracers, **params)
    else:
      return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(self, primitive, tracers, params):
    """By default, if all the input tracers are known, then execute the primitive
    and all the outputs are known. Otherwise, all the outputs are unknown."""
    consts = [t.pval.get_known() for t in tracers]
    if all(c is not None for c in consts):
      return primitive.bind(*consts, **params)
    tracers = map(self.instantiate_const, tracers)
    avals = [t.aval for t in tracers]
    out_aval = primitive.abstract_eval(*avals, **params)
    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    if primitive.multiple_results:
      out_tracers = [JaxprTracer(self, PartialVal.unknown(aval), None)
                     for aval in out_aval]
      eqn = new_eqn_recipe(tracers, out_tracers, primitive, params, source)
      for t in out_tracers: t.recipe = eqn
      return out_tracers
    else:
      out_tracer = JaxprTracer(self, PartialVal.unknown(out_aval), None)
      out_tracer.recipe = new_eqn_recipe(tracers, [out_tracer], primitive,
                                         params, source)
      return out_tracer

  def process_call(self, primitive, f: lu.WrappedFun, tracers, params):
    if primitive in call_partial_eval_rules:
      return call_partial_eval_rules[primitive](self, primitive, f, tracers, params)

    update_params = call_param_updaters.get(primitive) or (lambda p, _, __: p)
    in_knowns, in_avals, in_consts = partition_pvals([t.pval for t in tracers])

    # We want to partially evaluate this call into two calls: one evaluated now
    # taking known values (in_consts) as inputs and producing known values
    # (out_consts) as outputs, and the other staged out as an eqn into the jaxpr
    # being built. The latter takes as input residuals (res) produced as outputs
    # of the first call, shared closed-over values (env), and explicit arguments
    # which were unknown to the first call (corresponding to in_avals).

    # Wrap f to perform the partial evaluation and plumb out aux data.
    f = trace_to_subjaxpr_nounits(f, self.main, False)
    f, aux = partial_eval_wrapper_nounits(f, tuple(in_knowns), tuple(in_avals))
    # Adjust parameters (e.g. donated_invars) for the call to be evaluated now.
    const_params = update_params(params, in_knowns, 0)

    # Run the call, getting known out vals and aux data used for staged-out call
    out = primitive.bind(f, *in_consts, **const_params)
    out_knowns, out_avals, jaxpr, env = aux()
    # Split apart known outputs from the original call and residuals.
    out_consts, res = split_list(out, [len(out) - len(jaxpr.constvars)])

    # Create the input tracers for the staged-out (unknown-value) call.
    const_tracers = map(self.new_instantiated_const, res)
    env_tracers = map(self.full_raise, env)
    unknown_arg_tracers = [t for t in tracers if not t.is_known()]
    # Adjust parameters (e.g. donated_invars) for the staged-out call's args.
    num_new_args = len(const_tracers) + len(env_tracers)
    staged_params = update_params(params, map(op.not_, in_knowns), num_new_args)
    staged_params = dict(staged_params, call_jaxpr=convert_constvars_jaxpr(jaxpr))
    # The outputs of the staged-out call are Tracers with the new eqn as recipe.
    out_tracers = [JaxprTracer(self, PartialVal.unknown(a), None)
                   for a in out_avals]
    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    eqn = new_eqn_recipe((*const_tracers, *env_tracers, *unknown_arg_tracers),
                         out_tracers, primitive, staged_params, source)
    for t in out_tracers: t.recipe = eqn
    return merge_lists(out_knowns, out_tracers, out_consts)

  def process_map(self, primitive, f: lu.WrappedFun, tracers, params):
    update_params = call_param_updaters.get(primitive) or (lambda p, _, __: p)
    in_knowns, in_avals, in_consts = partition_pvals([t.pval for t in tracers])

    # This method is like process_call above, except:
    #   1. we delete an axis from mapped-over input avals' shapes, and
    #      analogously add an axis to mapped-over output avals' shapes;
    #   2. we update the in_axes and out_axes/out_axes_thunk parameters to
    #      reflect the inputs and outputs pruned from the unknown/known sides.

    # Map (delete an axis from) unknown inputs' avals as dictated by in_axes.
    unk_in_axes, const_in_axes = partition_list(in_knowns, params['in_axes'])
    in_avals_mapped = [mapped_aval(params['axis_size'], ax, aval)
                       for ax, aval in zip(unk_in_axes, in_avals)]

    # Wrap f to perform partial evaluation and plumb out aux data.
    f = trace_to_subjaxpr_nounits(f, self.main, False)
    f, aux = partial_eval_wrapper_nounits(f, tuple(in_knowns),
                                          tuple(in_avals_mapped))
    # Adjust params for knowns (e.g. donated_invars, in_axes, out_axes_thunk)
    const_params = update_params(params, in_knowns, 0)  # handles donated_invars
    out_axes_thunk = params['out_axes_thunk']
    @as_hashable_function(closure=out_axes_thunk)
    def const_out_axes_thunk():
      out_knowns, _, jaxpr, _ = aux()
      _, out_axes = partition_list(out_knowns, out_axes_thunk())
      return tuple(out_axes) + (0,) * len(jaxpr.constvars)  # res mapped axis 0
    const_params = dict(const_params, in_axes=tuple(const_in_axes),
                        out_axes_thunk=const_out_axes_thunk)

    # Run the map, getting known out vals and aux data used for staged-out map.
    with core.extend_axis_env(params['axis_name'], params['axis_size'], None):
      out = primitive.bind(f, *in_consts, **const_params)
    out_knowns, out_avals_mapped, jaxpr, env = aux()
    # Split apart known outputs from the original call and residuals.
    out_consts, res = split_list(out, [len(out) - len(jaxpr.constvars)])

    # We can only check_jaxpr with the dynamic axis environment extended:
    with core.extend_axis_env(params['axis_name'], params['axis_size'], None):
      call_jaxpr = convert_constvars_jaxpr(jaxpr)

    # Compute staged and const out_axes, taking into account residuals.
    out_axes = params['out_axes_thunk']()
    staged_out_axes, _ = partition_list(out_knowns, out_axes)
    staged_in_axes = (0,) * len(res) + (None,) * len(env) + (*unk_in_axes,)

    # Create the input tracers for the staged-out (unkonwn-value) call.
    const_tracers = map(self.new_instantiated_const, res)
    env_tracers = map(self.full_raise, env)
    unknown_arg_tracers = [t for t in tracers if not t.is_known()]
    # Adjust params for staged-out call on unknown values.
    num_new_args = len(const_tracers) + len(env_tracers)
    staged_params = update_params(params, map(op.not_, in_knowns), num_new_args)
    staged_params = dict(staged_params, in_axes=staged_in_axes,
                         out_axes=tuple(staged_out_axes), call_jaxpr=call_jaxpr)
    # The outputs of the staged-out call are Tracers with the new eqn as recipe.
    out_avals = [unmapped_aval(params['axis_size'], params['axis_name'], ax, a)
                 for ax, a in zip(staged_out_axes, out_avals_mapped)]
    out_tracers = [JaxprTracer(self, PartialVal.unknown(a), None)
                   for a in out_avals]
    eqn = new_eqn_recipe((*const_tracers, *env_tracers, *unknown_arg_tracers),
                         out_tracers, primitive, staged_params,
                         source_info_util.current())
    for t in out_tracers: t.recipe = eqn

    return merge_lists(out_knowns, out_tracers, out_consts)

  def post_process_call(self, primitive, out_tracers, params):
    unknown_out_tracers = [t for t in out_tracers if not t.is_known()]
    jaxpr, res, env = tracers_to_jaxpr([], unknown_out_tracers)
    out_pvals = [t.pval for t in out_tracers]
    out_knowns, out_avals, out_consts = partition_pvals(out_pvals)
    out = [*out_consts, *res]
    main = self.main

    def todo(out):
      trace = main.with_cur_sublevel()
      out_consts, res = split_list(out, [len(out) - len(jaxpr.constvars)])
      const_tracers = map(trace.new_instantiated_const, res)
      in_tracers = (*const_tracers, *map(trace.full_raise, env))
      out_tracers = [JaxprTracer(trace, PartialVal.unknown(a), None)
                     for a in out_avals]
      update_params = call_param_updaters.get(primitive) or (lambda p, _, __: p)
      new_params = update_params(params, [], len(in_tracers))
      new_params = dict(new_params, call_jaxpr=convert_constvars_jaxpr(jaxpr))
      name_stack = self._current_truncated_name_stack()
      source = source_info_util.current().replace(name_stack=name_stack)
      eqn = new_eqn_recipe(in_tracers, out_tracers, primitive, new_params, source)
      for t in out_tracers: t.recipe = eqn
      return merge_lists(out_knowns, out_tracers, out_consts)

    return out, todo

  def post_process_map(self, primitive, out_tracers, params):
    unknown_out_tracers = [t for t in out_tracers if not t.is_known()]
    jaxpr, res, env = tracers_to_jaxpr([], unknown_out_tracers)
    out_pvals = [t.pval for t in out_tracers]
    out_knowns, out_avals_mapped, out_consts = partition_pvals(out_pvals)
    out = [*out_consts, *res]
    main = self.main

    with core.extend_axis_env(params['axis_name'], params['axis_size'], None):
      call_jaxpr = convert_constvars_jaxpr(jaxpr)

    def todo(out):
      trace = main.with_cur_sublevel()
      out_consts, res = split_list(out, [len(out) - len(jaxpr.constvars)])
      const_tracers = map(trace.new_instantiated_const, res)
      env_tracers = map(trace.full_raise, env)

      staged_out_axes = tuple(out_axes_unknown)  # set by out_axes_transform
      staged_in_axes = (0,) * len(res) + (None,) * len(env)

      update_params = call_param_updaters.get(primitive) or (lambda p, _, __: p)
      staged_params = update_params(params, [], len(res) + len(env))
      staged_params = dict(staged_params, in_axes=staged_in_axes,
                           out_axes=tuple(staged_out_axes),
                           call_jaxpr=call_jaxpr)

      out_avals = [unmapped_aval(params['axis_size'], params['axis_name'], d, a)
                   for d, a in zip(staged_out_axes, out_avals_mapped)]
      out_tracers = [JaxprTracer(trace, PartialVal.unknown(a), None)
                     for a in out_avals]
      name_stack = self._current_truncated_name_stack()
      source = source_info_util.current().replace(name_stack=name_stack)
      eqn = new_eqn_recipe((*const_tracers, *env_tracers), out_tracers,
                           primitive, staged_params, source)
      for t in out_tracers: t.recipe = eqn
      return merge_lists(out_knowns, out_tracers, out_consts)

    def out_axes_transform(out_axes):
      nonlocal out_axes_unknown
      out_axes_unknown, out_axes_known = partition_list(out_knowns, out_axes)
      return tuple(out_axes_known) + (0,) * len(jaxpr.constvars)
    out_axes_unknown: Optional[list] = None

    return out, (todo, out_axes_transform)

  def _current_truncated_name_stack(self):
    return source_info_util.current_name_stack()[len(self.name_stack):]

  def partial_eval(self, f: lu.WrappedFun, pvals: Sequence[PartialVal],
                   app: Callable[[lu.WrappedFun, Tuple[core.Value, ...]], Tuple[core.Value]],
                   instantiate: bool):
    """Partially evaluate f on a sequence of PartialVals."""
    in_avals, in_consts = unzip2(pvals)
    f = trace_to_subjaxpr(f, self.main, instantiate)
    f, aux = partial_eval_wrapper(f, tuple(in_avals))
    out_flat, (out_avals, jaxpr, env) = app(f, *in_consts), aux()
    out_consts, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
    out_pvs = map(PartialVal, zip(out_avals, out_consts))
    env_tracers = map(self.full_raise, env)
    return jaxpr, out_pvs, consts, env_tracers

  def process_custom_jvp_call(self, prim, fun, jvp, tracers):
    tracers = map(self.instantiate_const_abstracted, tracers)
    in_avals, in_consts = unzip2(t.pval for t in tracers)  # in_consts are units
    fun = trace_to_subjaxpr(fun, self.main, True)
    fun, aux = partial_eval_wrapper(fun, tuple(in_avals))
    out_flat = prim.bind(fun, jvp, *in_consts)
    out_avals, jaxpr, env = aux()
    out_consts, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
    out_pvals = map(PartialVal, zip(out_avals, out_consts))  # out_consts are units
    env_tracers = map(self.full_raise, env)
    out_tracers = [JaxprTracer(self, pval, None) for pval in out_pvals]
    const_tracers = map(self.new_instantiated_const, consts)
    in_tracers = (*const_tracers, *env_tracers, *tracers)
    closed_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ())

    @_memoize
    def jvp_jaxpr_thunk():
      jvp_ = trace_to_subjaxpr(jvp, self.main, True)
      jvp_, aux = partial_eval_wrapper(jvp_, tuple(in_avals) * 2)
      with core.new_sublevel():
        out_flat = jvp_.call_wrapped(*(in_consts * 2))  # in_consts are units
      out_avals, jaxpr, env = aux()
      _, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
      converted_jaxpr = convert_envvars_to_constvars(jaxpr, len(env))
      return converted_jaxpr, (*consts, *env)

    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    eqn = new_eqn_recipe(in_tracers, out_tracers, prim.initial_style,
                         dict(fun_jaxpr=closed_jaxpr,
                              jvp_jaxpr_thunk=jvp_jaxpr_thunk,
                              num_consts=len(consts) + len(env)),
                         source)
    for t in out_tracers: t.recipe = eqn
    return out_tracers

  def post_process_custom_jvp_call(self, out_tracers, _):
    # This path should only be reachable if we expose a partial eval API
    # unrelated to autodiff, since we raise an error when differentiation with
    # respect to values over which a custom_jvp function closes is detected.
    raise NotImplementedError  # TODO(mattjj)

  def process_custom_transpose(self, prim, call, tracers, **params):
    res_ts, lin_ts = split_list(tracers, [params['res_tree'].num_leaves])
    assert all(t.is_known()     for t in res_ts)
    lin_all_known   = all(t.is_known()     for t in lin_ts)
    if lin_all_known:
      res_cvals = [t.pval[1] for t in res_ts]
      lin_cvals = [t.pval[1] for t in lin_ts]
      return prim.bind(call, *res_cvals, *lin_cvals, **params)
    else:
      out_tracers = [JaxprTracer(self, PartialVal.unknown(aval), None)
                     for aval in params['out_types']]
      in_tracers = map(self.instantiate_const, tracers)
      new_params = dict(params, call=call)
      eqn = new_eqn_recipe(in_tracers, out_tracers, prim, new_params,
                           source_info_util.current())
      for t in out_tracers: t.recipe = eqn
      return out_tracers

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees):
    tracers = map(self.instantiate_const_abstracted, tracers)
    in_avals, in_consts = unzip2(t.pval for t in tracers)  # in_consts are units
    fun = trace_to_subjaxpr(fun, self.main, True)
    fun, aux = partial_eval_wrapper(fun, tuple(in_avals))
    out_flat = prim.bind(fun, fwd, bwd, *in_consts, out_trees=out_trees)
    out_avals, jaxpr, env = aux()
    out_consts, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
    out_pvals = map(PartialVal, zip(out_avals, out_consts))  # out_consts are units
    env_tracers = map(self.full_raise, env)
    out_tracers = [JaxprTracer(self, pval, None) for pval in out_pvals]
    const_tracers = map(self.new_instantiated_const, consts)
    in_tracers = (*const_tracers, *env_tracers, *tracers)
    closed_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ())

    @_memoize
    def fwd_jaxpr_thunk():
      fwd_ = trace_to_subjaxpr(fwd, self.main, True)
      fwd_, aux = partial_eval_wrapper(fwd_, tuple(in_avals))
      with core.new_sublevel():
        out_flat = fwd_.call_wrapped(*in_consts)  # in_consts are units
      out_avals, jaxpr, env = aux()
      _, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
      converted_jaxpr = convert_envvars_to_constvars(jaxpr, len(env))
      return converted_jaxpr, (*consts, *env)

    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    eqn = new_eqn_recipe(in_tracers, out_tracers, prim.initial_style,
                         dict(fun_jaxpr=closed_jaxpr,
                              fwd_jaxpr_thunk=fwd_jaxpr_thunk,
                              num_consts=len(consts) + len(env),
                              bwd=bwd, out_trees=out_trees),
                         source)
    for t in out_tracers: t.recipe = eqn
    return out_tracers

  def post_process_custom_vjp_call(self, out_tracers, _):
    # This path should only be reachable if we expose a partial eval API
    # unrelated to autodiff, since we raise an error when differentiation with
    # respect to values over which a custom_vjp function closes is detected.
    raise NotImplementedError  # TODO(mattjj)

def partition_pvals(
    pvals: List[PartialVal]
  ) -> Tuple[List[bool], List[AbstractValue], List[Any]]:
  knowns = [pval.is_known()  for pval in pvals                       ]
  avals  = [pval.get_aval()  for pval in pvals if not pval.is_known()]
  consts = [pval.get_known() for pval in pvals if     pval.is_known()]
  return knowns, avals, consts

@lu.transformation_with_aux
def partial_eval_wrapper_nounits(
    in_knowns: Sequence[bool], in_avals: Sequence[AbstractValue],
    *in_consts: Any):
  in_avals_, in_consts_ = iter(in_avals), iter(in_consts)
  in_pvals = [PartialVal.known(next(in_consts_)) if known else
              PartialVal.unknown(next(in_avals_)) for known in in_knowns]
  sentinel = object()
  assert next(in_avals_, sentinel) is next(in_consts_, sentinel) is sentinel
  jaxpr, (out_pvals, res, env) = yield (in_pvals,), {}
  out_knowns, out_avals, out_consts = partition_pvals(out_pvals)
  yield (*out_consts, *res), (out_knowns, out_avals, jaxpr, env)

custom_partial_eval_rules: Dict[Primitive, Callable] = {}
call_partial_eval_rules: Dict[Primitive, Callable] = {}
call_param_updaters: Dict[Primitive, Callable] = {}


def abstract_eval_fun(fun, *avals, debug_info=None, **params):
  _, avals_out, _ = trace_to_jaxpr_dynamic(
      lu.wrap_init(fun, params), avals, debug_info)
  assert all(isinstance(aval, AbstractValue) for aval in avals_out)
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
          const, "Tracer from a higher level: {} in trace {}".format(const, trace))
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

@profiler.annotate_function
def trace_to_jaxpr(
    fun: lu.WrappedFun, pvals: Sequence[PartialVal],
    instantiate: Union[bool, Sequence[bool]] = False,
  ) -> Tuple[Jaxpr, List[PartialVal], List[core.Value]]:
  """
  Partially evaluate a function, building a jaxpr for un-evaluated computation.

  Args:
    fun: lu.WrappedFun representing the function to be partially evaluated. The
      function must be flattened, in the sense of accepting jaxpr type arguments
      and returning a flat list of jaxpr type outputs.
    pvals: sequence of PartialVals of length equal to the number of inputs to
      `fun` indicating which inputs are known or unknown.
    instantiate: optional bool or sequence of bools of length equal to the
      number of outputs of `fun` indicating which outputs should be forced to be
      treated as unknown and hence instantiated in the jaxpr. If a single bool,
      the value is applied to all outputs. Default False.

  Returns:
    A triple where the first element is a jaxpr representing the computation
    which depends on unknown inputs; the second element is a list of PartialVals
    of length equal to the length of the output of `fun` representing which
    outputs are known and unknown (along with their values and abstract values,
    respectively); the third element is a list of known residual values. The
    returned jaxpr takes as inputs the known residual values followed by values
    of the originally unknown inputs.
  """
  current_name_stack = source_info_util.current_name_stack()
  with core.new_main(JaxprTrace, name_stack=current_name_stack) as main:
    fun = trace_to_subjaxpr(fun, main, instantiate)
    jaxpr, (out_pvals, consts, env) = fun.call_wrapped(pvals)
    assert not env
    del main, fun, env

  return jaxpr, out_pvals, consts


@lu.transformation
def trace_to_subjaxpr_nounits(
    main: core.MainTrace, instantiate: Union[bool, Sequence[bool]],
    in_pvals: Sequence[PartialVal]):
  assert all([isinstance(pv, PartialVal) for pv in in_pvals]), in_pvals
  trace = main.with_cur_sublevel()
  in_knowns  = [pval.is_known()     for pval in in_pvals]
  in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
  in_tracers = [trace.new_arg(pval) for pval in in_pvals if not pval.is_known()]
  in_args = merge_lists(in_knowns, in_tracers, in_consts)
  ans = yield in_args, {}
  assert isinstance(ans, (list, tuple)), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  assert all(isinstance(x, core.Tracer) or core.valid_jaxtype(x) for x in ans), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  if isinstance(instantiate, bool):
    instantiate = [instantiate] * len(ans)
  out_tracers = map(trace.full_raise, map(core.full_lower, ans))
  out_tracers = map(partial(instantiate_const_at, trace), instantiate, out_tracers)
  out_pvals = [t.pval for t in out_tracers]
  out_tracers_ = [t for t in out_tracers if not t.is_known()]
  jaxpr, consts, env = tracers_to_jaxpr(in_tracers, out_tracers_)
  del trace, in_tracers, out_tracers, out_tracers_
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
  primitive: Primitive
  params: Dict[str, Any]
  source_info: source_info_util.SourceInfo

def new_eqn_recipe(invars: Sequence[JaxprTracer],
                   outvars: Sequence[JaxprTracer],
                   primitive: Primitive,
                   params: Dict[str, Any],
                   source_info: source_info_util.SourceInfo
                  ) -> JaxprEqnRecipe:
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
    # assert len(invars) == len(params["call_jaxpr"].invars)  # TODO constvars?
    assert len(outvars) == len(params["call_jaxpr"].outvars)
    assert ("donated_invars" not in params or
            len(params["donated_invars"]) == len(params["call_jaxpr"].invars))
  if primitive.map_primitive:
    assert ("in_axes" in params and
            len(params["in_axes"]) == len(params["call_jaxpr"].invars))
    assert ("donated_invars" in params and
            len(params["donated_invars"]) == len(params["call_jaxpr"].invars))
  return JaxprEqnRecipe(object(), tuple(invars), map(ref, outvars), primitive,
                        params, source_info)


def recipe_to_eqn(getvar: Callable[[JaxprTracer], Atom],
                  recipe: JaxprEqnRecipe) -> core.JaxprEqn:
  _, in_tracers, out_tracer_refs, primitive, params, source_info = recipe
  out_tracers = [t_ref() for t_ref in out_tracer_refs]
  invars  = [getvar(t) for t in in_tracers]
  outvars = [core.DropVar(core.abstract_unit) if t is None
             else cast(Var, getvar(t)) for t in out_tracers]
  return new_jaxpr_eqn(invars, outvars, primitive, params, source_info)

def tracers_to_jaxpr(
  in_tracers: Sequence[JaxprTracer],
  out_tracers: Sequence[JaxprTracer]
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
  t_to_var: Dict[int, Atom] = {}
  def getvar(t: JaxprTracer) -> Atom:
    var = t_to_var.get(id(t))
    if var is None:
      aval = t.pval.get_aval() if not t.pval.is_known() else abstract_unit
      var = t_to_var[id(t)] = newvar(aval)
    return var
  sorted_tracers = toposort(out_tracers)
  invars = map(getvar, in_tracers)
  eqns: List[core.JaxprEqn] = []
  env: Dict[Var, Any] = {}
  consts: Dict[Var, Any] = {}
  const_to_var: Dict[int, Var] = {}
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
        eqns.append(recipe_to_eqn(getvar, recipe))
        processed_eqn_ids.add(recipe.eqn_id)
    elif isinstance(recipe, LambdaBinding):
      if not any(t is in_tracer for in_tracer in in_tracers):
        raise core.escaped_tracer_error(
            t, "Tracer not among input tracers {}".format(t))
      assert in_tracers, "Lambda binding with no args"
    elif isinstance(recipe, FreeVar):
      env[cast(Var, getvar(t))] = recipe.val
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
  jaxpr = Jaxpr(const_vars, [*env_vars, *invars], map(getvar, out_tracers), eqns)
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  return jaxpr, const_vals, env_vals

@cache()
def convert_constvars_jaxpr(jaxpr: Jaxpr) -> Jaxpr:
  """Moves the constvars to the start of invars."""
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  lifted_jaxpr = Jaxpr(constvars=(),
                       invars=jaxpr.constvars + jaxpr.invars,
                       outvars=jaxpr.outvars, eqns=jaxpr.eqns)
  config.jax_enable_checks and core.check_jaxpr(lifted_jaxpr)
  return lifted_jaxpr

def convert_envvars_to_constvars(jaxpr: Jaxpr, num_env_vars: int) -> Jaxpr:
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  env_vars, invars = split_list(jaxpr.invars, [num_env_vars])
  converted_jaxpr = Jaxpr(constvars=jaxpr.constvars + env_vars,
                          invars=invars, outvars=jaxpr.outvars, eqns=jaxpr.eqns)
  config.jax_enable_checks and core.check_jaxpr(converted_jaxpr)
  return converted_jaxpr


def _split_aval(unknown: bool, aval: AbstractValue) -> Tuple[AbstractValue, AbstractValue]:
  return (abstract_unit, aval) if unknown else (aval, abstract_unit)

def partial_eval_jaxpr(jaxpr: ClosedJaxpr, unknowns: Sequence[bool],
                       instantiate: Union[bool, Sequence[bool]],
                       ) -> Tuple[ClosedJaxpr, ClosedJaxpr, Sequence[bool]]:
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

  Note that if instantiate is True for a given output, then jaxpr_known always returns a
  unit in its place. So when instantiate is True, the expectation is the one doesn't
  run `jaxpr_known` for any of its outputs, but only to generate residuals that will allow
  to obtain the full outputs once `jaxpr_unknown` is ran. Outputs known ahead of time will
  simply get passed as residual constants and returned immediately.
  """
  instantiate = tuple(instantiate) if isinstance(instantiate, list) else instantiate
  return _partial_eval_jaxpr(jaxpr, tuple(unknowns), instantiate)

@cache()
def _partial_eval_jaxpr(jaxpr, unknowns, instantiate):
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))

  cell = []
  def fun(*vals):
    pvals = [PartialVal.unknown(aval) if uk else PartialVal.known(val)
            for aval, val, uk in zip(jaxpr.in_avals, vals, unknowns)]
    jaxpr_2, out_pvals_2, consts_2 = trace_to_jaxpr(f, pvals, instantiate=instantiate)
    out_pvs_2, out_consts_2 = unzip2(out_pvals_2)
    cell.append((out_pvs_2, jaxpr_2, len(consts_2)))
    return out_consts_2 + consts_2

  # For jaxpr_known we pass core.unit for the unknown inputs, and known
  # PartialVal for the known inputs.
  in_avals = [abstract_unit if uk else a for a, uk in zip(jaxpr.in_avals, unknowns)]
  jaxpr_1, out_avals, consts_1 = trace_to_jaxpr_dynamic(lu.wrap_init(fun), in_avals)
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
  res_avals = out_avals[len(jaxpr.out_avals):]
  assert len(res_avals) == num_res
  out_avals_1 = [*out_avals_1, *res_avals]
  in_avals_2 = [*in_avals_2, *res_avals]

  return ClosedJaxpr(jaxpr_1, consts_1), ClosedJaxpr(jaxpr_2, ()), uk_out


remat_call_p: Primitive = core.CallPrimitive('remat_call')
remat_call = remat_call_p.bind
remat_call_p.def_impl(core.call_impl)

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
  in_pvals = [PartialVal.unknown(t.pval[0]) for t in instantiated_tracers]
  jaxpr, eval_out_pvals, consts, env_tracers = trace.partial_eval(
    f, in_pvals, partial(remat_call_p.bind, **params), instantiate=False)
  jaxpr = convert_constvars_jaxpr(jaxpr)

  # Since we traced with everything marked as unknown, but we need to know which
  # outputs are known/unknown, we use partial_eval_jaxpr to get out_unknowns.
  in_unknowns = ([False] * len(consts) +
                 [not t.is_known() for t in it.chain(env_tracers, tracers)])
  if params['policy']:
    # unzip into jaxpr_known and jaxpr_unknown
    jaxpr_known, jaxpr_unknown, out_unknowns, out_inst, _ = _partial_eval_jaxpr_custom(
        jaxpr, in_unknowns, params['policy'])
    jaxpr_known, in_used_known = dce_jaxpr(jaxpr_known, [True] * len(jaxpr_known.outvars))
    _, used_outs_unknown = partition_list(out_inst, out_unknowns)
    jaxpr_unknown, in_used_unknown = dce_jaxpr(jaxpr_unknown, used_outs_unknown)

    # compute known outputs and residuals (hoisted out of a remat_call)
    if concrete: raise NotImplementedError  # TODO(mattjj)
    _, in_consts_ = unzip2(t.pval for t in it.chain(env_tracers, tracers)
                           if t.pval.is_known())
    _, in_consts = partition_list(in_used_known, [*consts, *in_consts_])
    out_consts = core.eval_jaxpr(jaxpr_known, (), *in_consts)
    out_consts_ = iter(out_consts)
    # reconstruct known outs, inserting units
    outs1 = [pval.get_known() if x.aval is abstract_unit else next(out_consts_)
             for uk, pval, x in zip(out_unknowns, eval_out_pvals, jaxpr.outvars)
             if not uk]
    # form known outputs and collect residual tracers
    out_known_tracers = [JaxprTracer(trace, PartialVal.known(c), None)
                         for c in outs1]
    residuals = list(out_consts_)

    # set up unknown outputs with a recipe to call remat
    res_tracers = map(trace.new_instantiated_const, residuals)
    const_tracers = map(trace.new_instantiated_const, consts)
    in_jaxpr_tracers = [*res_tracers, *const_tracers, *env_tracers,
                        *instantiated_tracers]
    _, in_jaxpr_tracers = partition_list(in_used_unknown, in_jaxpr_tracers)
    out_jaxpr_tracers = [JaxprTracer(trace, PartialVal.unknown(x.aval), None)
                         for x in jaxpr_unknown.outvars]
    new_params = dict(params, call_jaxpr=jaxpr_unknown, differentiated=True)
    recipe = new_eqn_recipe(in_jaxpr_tracers, out_jaxpr_tracers, remat_call_p,
                            new_params, source_info_util.current())
    for t in out_jaxpr_tracers: t.recipe = recipe
    return _zip_knowns(out_known_tracers, out_jaxpr_tracers, out_unknowns)
  else:
    # TODO(mattjj): this is an old parallel code path, to be deleted once the
    # new path is fully functional
    closed_jaxpr = core.ClosedJaxpr(jaxpr, ())
    jaxpr_known, jaxpr_unknown, out_unknowns = partial_eval_jaxpr(
        closed_jaxpr, in_unknowns, instantiate=False)  # type: ignore
    out_knowns = [not b for b in out_unknowns]
    out_known_pvals, out_unknown_pvals = _partition_knowns(eval_out_pvals, out_unknowns)

    # Next, we need values for the outputs that should be known. Since consts
    # weren't passed through Python for evaluation, we need to evaluate
    # jaxpr_known, minus the residual outputs that we don't need. When
    # `concrete=True`, as an optimization we can avoid redoing *some* redundant
    # FLOPs, namely those that produced concrete avals at the output, simply by
    # using those as computed values. For the use case of inverse-mode ad in
    # op-by-op ("eager mode") evaluation, all the primal outputs should be
    # concrete (thus not recomputed).
    to_compute = [type(pval[0]) is not ConcreteArray
                  for uk, pval in zip(out_unknowns, eval_out_pvals) if not uk]
    num_outputs = len(jaxpr_unknown.out_avals)
    num_res = len(jaxpr_known.out_avals) - num_outputs
    jaxpr_known_nores = _dce_jaxpr(jaxpr_known, out_knowns + [False] * num_res,
                                  drop_outputs=True)
    jaxpr_known_comp = _dce_jaxpr(jaxpr_known_nores, to_compute)
    _, in_consts = unzip2(t.pval for t in it.chain(env_tracers, tracers))
    reconstructed_consts = core.jaxpr_as_fun(jaxpr_known_comp)(*consts, *in_consts)
    out_known_pvals = map(_reconstruct_pval, out_known_pvals, reconstructed_consts)

    # Known outputs should keep propagating as constants
    assert all(pv.is_known() for pv in out_known_pvals)
    known_output_tracers = [trace.new_const(pval.get_known())
                            for pval in out_known_pvals]
    # Unknown outputs get wrapped in tracers with the appropriate recipe
    unknown_output_tracers = [JaxprTracer(trace, out_pval, None)
                              for out_pval in out_unknown_pvals]

    # dce jaxpr outputs
    new_jaxpr = _dce_jaxpr(closed_jaxpr, out_unknowns, drop_outputs=True).jaxpr
    new_params = dict(params, call_jaxpr=new_jaxpr, differentiated=True)

    # set up eqn for unknown outputs
    const_tracers = map(trace.new_instantiated_const, consts)
    in_tracers = (*const_tracers, *env_tracers, *instantiated_tracers)
    eqn = new_eqn_recipe(in_tracers, unknown_output_tracers, remat_call_p,
                         new_params, source_info_util.current())
    for t in unknown_output_tracers: t.recipe = eqn
    return _zip_knowns(known_output_tracers, unknown_output_tracers, out_unknowns)
call_partial_eval_rules[remat_call_p] = _remat_partial_eval

def _partition_knowns(pvals, unknowns: Sequence[bool]):
  return ([e for e, unknown in zip(pvals, unknowns) if not unknown],
          [e for e, unknown in zip(pvals, unknowns) if unknown])

def _zip_knowns(known_list, unknown_list, which_unknown: Sequence[bool]):
  assert len(known_list) + len(unknown_list) == len(which_unknown)
  known_iter, unknown_iter = iter(known_list), iter(unknown_list)
  return [next(unknown_iter) if uk else next(known_iter) for uk in which_unknown]


def _partial_eval_jaxpr_custom(
    jaxpr: Jaxpr, in_unknowns: Sequence[bool], saveable: Callable[..., bool],
  ) -> Tuple[Jaxpr, Jaxpr, Sequence[bool], Sequence[bool], int]:
  if jaxpr.constvars: raise NotImplementedError  # TODO(mattjj)
  env: Dict[Var, Tuple[bool, bool]] = {}
  residuals: OrderedSet[Var] = OrderedSet()

  def read(x: Atom) -> Tuple[bool, bool]:
    if type(x) is Var:
      return env[x]
    return (False, True)

  def write(unk: bool, inst: bool, v: Var) -> None:
    assert (unk, inst) != (True, False)
    env[v] = (unk, inst)

  def ensure_instantiated(inst: bool, x: Atom) -> Atom:
    if type(x) is Var and not inst:
      residuals.add(x)
    return x

  known_eqns, staged_eqns = [], []
  write(False, True, unitvar)
  map(write, in_unknowns, [True] * len(in_unknowns), jaxpr.invars)
  for eqn in jaxpr.eqns:
    unks_in, inst_in = unzip2(map(read, eqn.invars))
    rule = partial_eval_jaxpr_custom_rules.get(eqn.primitive)
    if rule:
      eqn1, eqn2, unks_out, inst_out, res = rule(saveable, unks_in, inst_in, eqn)
      eqn1 and known_eqns.append(eqn1); eqn2 and staged_eqns.append(eqn2)  # type: ignore
      residuals.update(res)
      map(write, unks_out, inst_out, eqn.outvars)
    elif any(unks_in):
      inputs = map(ensure_instantiated, inst_in, eqn.invars)
      staged_eqns.append(new_jaxpr_eqn(inputs, eqn.outvars, eqn.primitive,
                                       eqn.params, eqn.source_info))
      map(partial(write, True, True), eqn.outvars)
    else:
      known_eqns.append(eqn)
      if saveable(eqn.primitive, *[x.aval for x in eqn.invars], **eqn.params):
        map(partial(write, False, False), eqn.outvars)
      else:
        inputs = map(ensure_instantiated, inst_in, eqn.invars)
        staged_eqns.append(new_jaxpr_eqn(inputs, eqn.outvars, eqn.primitive,
                                         eqn.params, eqn.source_info))
        map(partial(write, False, True), eqn.outvars)
  out_unknowns, out_inst = unzip2(map(read, jaxpr.outvars))
  assert all(type(v) is Var for v in residuals), residuals

  ins_known, _ = partition_list(in_unknowns, jaxpr.invars)
  outs_known_, _ = partition_list(out_unknowns, jaxpr.outvars)
  outs_known = [x for x in outs_known_ if x.aval is not abstract_unit]
  jaxpr_known = Jaxpr((), ins_known, [*outs_known, *residuals], known_eqns)
  config.jax_enable_checks and core.check_jaxpr(jaxpr_known)

  _, outs_staged = partition_list(out_inst, jaxpr.outvars)
  jaxpr_staged = Jaxpr((), [*residuals, *jaxpr.invars], outs_staged, staged_eqns)
  config.jax_enable_checks and core.check_jaxpr(jaxpr_staged)

  return jaxpr_known, jaxpr_staged, out_unknowns, out_inst, len(residuals)

# A primitive rule for policy-driven partial evaluation returns a 5-tuple
# with the components representing, respectively:
#  * the JaxprEqn for the 'known' side (or None if there is no known component),
#  * the JaxprEqn for the 'unknown' side (or None),
#  * a list of booleans indicating which of the original outputs are unknown,
#  * a list of booleans indicating which of the original outputs are
#    instantiated (i.e. available) in the 'unknown' side,
#  * a list of Var instances representing residuals to be added (i.e. to be
#    plumbed as outputs of the 'known' side jaxpr and added as input binders to
#    the 'unknown' jaxpr).
PartialEvalCustomResult = Tuple[Optional[JaxprEqn], Optional[JaxprEqn],
                                Sequence[bool], Sequence[bool], List[Var]]
PartialEvalCustomRule = Callable[
    [Callable[..., bool], Sequence[bool], Sequence[bool], JaxprEqn],
    PartialEvalCustomResult]
partial_eval_jaxpr_custom_rules: Dict[Primitive, PartialEvalCustomRule] = {}

def partial_eval_jaxpr_custom_rule_not_implemented(
    name: str, saveable: Callable[..., bool], unks_in: Sequence[bool],
    inst_in: Sequence[bool], eqn: JaxprEqn) -> PartialEvalCustomResult:
  msg = (f'custom-policy remat rule not implemented for {name}, '
         'open a feature request at https://github.com/google/jax/issues!')
  raise NotImplementedError(msg)


ParamsUpdater = Callable[[Sequence[bool], Sequence[bool], Sequence[bool],
                          int, dict, dict], Tuple[dict, dict]]

def call_partial_eval_custom_rule(
    jaxpr_param_name: str, params_updater: ParamsUpdater,
    saveable: Callable[..., bool], unks_in: List[bool], inst_in: List[bool],
    eqn: JaxprEqn
  ) -> Tuple[JaxprEqn, JaxprEqn, Sequence[bool], Sequence[bool], List[Var]]:
  jaxpr = eqn.params[jaxpr_param_name]
  jaxpr_known, jaxpr_staged, unks_out, inst_out, num_res = \
      _partial_eval_jaxpr_custom(jaxpr, unks_in, saveable)
  ins_known, _ = partition_list(unks_in, eqn.invars)
  # by convention, _partial_eval_jaxpr_custom drops units on known outputs
  known_units_out = [v.aval is core.abstract_unit for v in jaxpr.outvars]
  dropped_outs_known = map(op.or_, unks_out, known_units_out)
  kept_outs_known = [not d for d in dropped_outs_known]
  out_binders_known, _ = partition_list(dropped_outs_known, eqn.outvars)
  _, out_binders_staged = partition_list(inst_out, eqn.outvars)
  kept_outs_staged = inst_out
  newvar = core.gensym([jaxpr_known, jaxpr_staged])
  residuals = [newvar(v.aval) for v in jaxpr_staged.invars[:num_res]]
  params_known = {**eqn.params, jaxpr_param_name: jaxpr_known}
  params_staged = {**eqn.params, jaxpr_param_name: jaxpr_staged}
  params_known, params_staged = params_updater(
      unks_in, kept_outs_known, kept_outs_staged, num_res, params_known, params_staged)
  eqn_known = new_jaxpr_eqn(ins_known, [*out_binders_known, *residuals],
                            eqn.primitive, params_known, eqn.source_info)
  eqn_staged = new_jaxpr_eqn([*residuals, *eqn.invars], out_binders_staged,
                             eqn.primitive, params_staged, eqn.source_info)
  assert len(eqn_staged.invars) == len(jaxpr_staged.invars)
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is Var and not inst]
  return eqn_known, eqn_staged, unks_out, inst_out, new_inst + residuals
partial_eval_jaxpr_custom_rules[core.call_p] = \
    partial(call_partial_eval_custom_rule, 'call_jaxpr',
            lambda _, __, ___, ____, x, y: (x, y))
partial_eval_jaxpr_custom_rules[core.named_call_p] = \
    partial(call_partial_eval_custom_rule, 'call_jaxpr',
            lambda _, __, ___, ____, x, y: (x, y))
partial_eval_jaxpr_custom_rules[remat_call_p] = \
    partial(call_partial_eval_custom_rule, 'call_jaxpr',
            lambda _, __, ___, ____, p1, p2: (p1, dict(p2, differentiated=True)))


# TODO(mattjj): unify with dce code below
def dce_jaxpr(jaxpr: Jaxpr, used_outputs: List[bool]
              ) -> Tuple[Jaxpr, List[bool]]:
  if jaxpr.constvars: raise NotImplementedError  # TODO(mattjj)
  env: Dict[Var, bool] = {}

  def read(v: Var) -> bool:
    return env.get(v, False)

  def write(x: Atom, b: bool) -> None:
    if type(x) is Var:
      env[x] = read(x) or b

  new_eqns = []
  map(write, jaxpr.outvars, used_outputs)
  for eqn in jaxpr.eqns[::-1]:
    used_outs = map(read, eqn.outvars)
    # If any outputs are used, then we need to keep a version of the eqn and
    # potentially mark some inputs as used. Otherwise mark all inputs as unused.
    if any(used_outs) or core.primitive_uses_outfeed(eqn.primitive, eqn.params):
      # If there's a rule for modifying the eqn and computing used inputs, apply
      # it. Otherwise, keep the eqn unmodified and mark all inputs as used.
      rule = dce_rules.get(eqn.primitive)
      if rule:
        used_ins, new_eqn = rule(used_outs, eqn)
      else:
        used_ins = [True] * len(eqn.invars)
        new_eqn = eqn
      new_eqns.append(new_eqn)
    else:
      used_ins = [False] * len(eqn.invars)
    map(write, eqn.invars, used_ins)
  used_inputs = map(read, jaxpr.invars)

  new_jaxpr = Jaxpr((),
                    [v for v, b in zip(jaxpr.invars, used_inputs)   if b],
                    [v for v, b in zip(jaxpr.outvars, used_outputs) if b],
                    new_eqns[::-1])
  config.jax_enable_checks and core.check_jaxpr(new_jaxpr)

  return new_jaxpr, used_inputs

DCERule = Callable[[List[bool], JaxprEqn], Tuple[List[bool], JaxprEqn]]
dce_rules: Dict[Primitive, DCERule] = {}


def dce_jaxpr_call_rule(used_outputs: List[bool], eqn: JaxprEqn
                        ) -> Tuple[List[bool], JaxprEqn]:
  new_jaxpr, used_inputs = dce_jaxpr(eqn.params['call_jaxpr'], used_outputs)
  new_params = dict(eqn.params, call_jaxpr=new_jaxpr)
  update_params = call_param_updaters.get(eqn.primitive)
  if update_params:
    new_params = update_params(new_params, used_inputs, 0)
  new_eqn = new_jaxpr_eqn([v for v, used in zip(eqn.invars, used_inputs) if used],
                          [v for v, used in zip(eqn.outvars, used_outputs) if used],
                          eqn.primitive, new_params, eqn.source_info)
  return used_inputs, new_eqn
dce_rules[core.call_p] = dce_jaxpr_call_rule
dce_rules[core.named_call_p] = dce_jaxpr_call_rule
dce_rules[remat_call_p] = dce_jaxpr_call_rule


def _dce_jaxpr(closed_jaxpr: ClosedJaxpr, outputs: Sequence[bool], drop_outputs=False) -> ClosedJaxpr:
  new_jaxpr = _dce_open_jaxpr(closed_jaxpr.jaxpr, tuple(outputs), drop_outputs)
  return core.ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)

@cache()
def _dce_open_jaxpr(jaxpr: Jaxpr, outputs: Tuple[bool, ...], drop_outputs=False) -> Jaxpr:
  # This dead-code elimination is pretty rudimentary, and in particular doesn't
  # nontrivially DCE through scan, call, or other higher-order primitives.
  # TODO(mattjj): better DCE (i.e. use above dce_jaxpr)
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
  return Jaxpr(jaxpr.constvars, jaxpr.invars, new_outvars, new_eqns)

@cache()
def _drop_vars(jaxpr: Jaxpr, drop_ins: Tuple[bool, ...], drop_outs: Tuple[bool, ...]):
  return Jaxpr(jaxpr.constvars,
               [v for v, d in zip(jaxpr.invars, drop_ins) if not d],
               [v for v, d in zip(jaxpr.outvars, drop_outs) if not d],
               jaxpr.eqns)


def _reconstruct_pval(pval1: PartialVal, const2: core.Value):
  pv1, _ = pval1
  if pval1.is_known():
    return pval1
  else:
    if type(pv1) is ConcreteArray:
      return PartialVal.known(pv1.val)  # pytype: disable=attribute-error
    else:
      return PartialVal.known(const2)


def move_binders_to_front(closed_jaxpr: ClosedJaxpr, to_move: Sequence[bool]) -> ClosedJaxpr:
  """Reorder the `invars` to move to front the ones for which `to_move` is True."""
  assert len(closed_jaxpr.in_avals) == len(to_move)
  new_invars = _move_to_front(closed_jaxpr.jaxpr.invars, to_move)
  new_jaxpr = Jaxpr(closed_jaxpr.jaxpr.constvars, new_invars,
                    closed_jaxpr.jaxpr.outvars, closed_jaxpr.jaxpr.eqns)
  new_closed_jaxpr = core.ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)
  return new_closed_jaxpr

def _move_to_front(lst: Sequence, to_move: Sequence[bool]) -> Sequence:
  return ([elt for elt, move in zip(lst, to_move) if move] +
          [elt for elt, move in zip(lst, to_move) if not move])


class DynamicJaxprTracer(core.Tracer):
  __slots__ = ['aval']

  def __init__(self, trace, aval, line_info=None):
    self._trace = trace
    self._line_info = line_info
    self.aval = aval

  def full_lower(self):
    return self

  def _contents(self):
    return ()

  def _origin_msg(self):
    if not self._trace.main.jaxpr_stack:  # type: ignore
      # If this Tracer has been leaked the jaxpr stack may no longer be
      # available. So we can't print as much origin information.
      return ("\nThis Tracer was created on line "
              f"{source_info_util.summarize(self._line_info)}")
    else:
      invar_pos, progenitor_eqns = self._trace.frame.find_progenitors(self)
    dbg = self._trace.main.debug_info
    if dbg is None:
      return ""
    if invar_pos:
      origin = (f"While tracing the function {dbg.func_src_info} "
                f"for {dbg.traced_for}, "
                "this concrete value was not available in Python because it "
                f"depends on the value{'s' if len(invar_pos) > 1 else ''} "
                f"of {dbg.arg_info(invar_pos)}.")
    elif progenitor_eqns:
      msts = ["  operation "
              f"{core.pp_eqn(eqn, core.JaxprPpContext(), core.JaxprPpSettings(print_shapes=True))}\n"
              f"    from line {source_info_util.summarize(eqn.source_info)}"
              for eqn in progenitor_eqns[:5]]  # show at most 5
      origin = (f"While tracing the function {dbg.func_src_info} "
                f"for {dbg.traced_for}, "
                "this value became a tracer due to JAX operations on these lines:"
                "\n\n" + "\n\n".join(msts))
      if len(progenitor_eqns) > 5:
        origin += "\n\n(Additional originating lines are not shown.)"
    else:
      origin = (f"The error occured while tracing the function {dbg.func_src_info} "
                f"for {dbg.traced_for}.")
    return "\n" + origin

  def _assert_live(self) -> None:
    if not self._trace.main.jaxpr_stack:  # type: ignore
      raise core.escaped_tracer_error(self, None)

TracerId = int
AvalId = int
ConstId = int
class JaxprStackFrame:
  gensym: Callable[[AbstractValue], Var]
  tracer_to_var: Dict[TracerId, Var]
  constid_to_tracer: Dict[ConstId, Tracer]
  constvar_to_val: Dict[Var, Any]
  tracers: List[DynamicJaxprTracer]  # hold onto strong refs for all tracers
  eqns: List[JaxprEqn]
  invars: List[Var]

  def __init__(self):
    self.gensym = core.gensym()
    self.tracer_to_var = {}
    self.constid_to_tracer = {}
    self.constvar_to_val = {}
    self.tracers = []   # circ refs, frame->tracer->trace->main->frame,
    self.eqns = []      # cleared when we pop frame from main
    self.invars = []

  def to_jaxpr(self, out_tracers):
    # It's not necessary, but we keep the tracer-to-var mapping injective:
    assert len(self.tracer_to_var) == len(set(self.tracer_to_var.values()))
    outvars = [self.tracer_to_var[id(t)] for t in out_tracers]
    constvars, constvals = unzip2(self.constvar_to_val.items())
    jaxpr = Jaxpr(constvars, self.invars, outvars, self.eqns)
    jaxpr, constvals = _const_folding_and_forwarding(jaxpr, constvals)
    jaxpr, constvals = _inline_literals(jaxpr, constvals)
    return jaxpr, constvals

  def newvar(self, aval):
    if isinstance(aval, DShapedArray):
      # this aval may have tracers in it, so we replace those with variables
      new_shape = [self.tracer_to_var[id(d)] if isinstance(d, Tracer) else d
                   for d in aval.shape]
      aval = aval.update(shape=tuple(new_shape))
    return self.gensym(aval)

  def find_progenitors(self, tracer):
    var = self.tracer_to_var.get(id(tracer))
    if not var:
      return None, None
    active_vars = {var}
    for eqn in self.eqns[::-1]:
      produced = set(eqn.outvars) & active_vars
      if produced:
        active_vars.difference_update(produced)
        active_vars.update(eqn.invars)
    invar_positions = [i for i, v in enumerate(self.invars) if v in active_vars]
    constvars = active_vars & set(self.constvar_to_val)
    const_eqns = [eqn for eqn in self.eqns if set(eqn.invars) & constvars]
    return invar_positions, const_eqns

def _const_folding_and_forwarding(jaxpr, constvals):
  consts: Dict[Var, Any] = dict(zip(jaxpr.constvars, constvals))
  var_subs: Dict[Var, Var] = {}  # not Dict[Var, Atom] b/c literals not inlined
  new_eqns = []
  for eqn in jaxpr.eqns:
    # always apply invar substitutions
    eqn = JaxprEqn([var_subs.get(v, v) for v in eqn.invars], eqn.outvars,
                    eqn.primitive, eqn.params, eqn.source_info)
    # if any inputs are constants and we have a constant-folding rule, apply it
    if eqn.primitive in const_fold_rules and any(v in consts for v in eqn.invars):
      consts_in = [consts.get(v) for v in eqn.invars]
      consts_out, new_eqn = const_fold_rules[eqn.primitive](consts_in, eqn)
      assert (new_eqn is None) == all(c is not None for c in consts_out)
      for v, c in zip(eqn.outvars, consts_out):
        if c is not None: consts[v] = c
      if new_eqn is None: continue
      else: eqn = new_eqn
    # if the application trivially maps some inputs to outputs, simplify
    if eqn.primitive in forwarding_rules:
      fwd_vars, new_eqn = forwarding_rules[eqn.primitive](eqn)
      assert (new_eqn is None) == all(v is not None for v in fwd_vars)
      for v_orig, v_new in zip(eqn.outvars, fwd_vars):
        if v_new is not None: var_subs[v_orig] = v_new
      if new_eqn is None: continue
      else: eqn = new_eqn
    new_eqns.append(eqn)
  new_constvars, new_constvals = unzip2(consts.items())
  new_outvars = [var_subs.get(v, v) for v in jaxpr.outvars]
  new_jaxpr = Jaxpr(new_constvars, jaxpr.invars, new_outvars, new_eqns)
  return new_jaxpr, new_constvals

ConstFoldRule = Callable[[List[Optional[Any]], JaxprEqn],
                         Tuple[List[Optional[Any]], Optional[JaxprEqn]]]
const_fold_rules: Dict[Primitive, ConstFoldRule] = {}

ForwardingRule = Callable[[JaxprEqn],
                          Tuple[List[Optional[Var]], Optional[JaxprEqn]]]
forwarding_rules: Dict[Primitive, ForwardingRule] = {}

def _inline_literals(jaxpr, constvals):
  # This function also ensures variables are labeled in a canonical ordering,
  # prunes unused constants, and inserts `dropvar` symbols.
  consts = dict(zip(jaxpr.constvars, constvals))
  newname: Callable[[AbstractValue], Var] = core.gensym()
  newvars: Dict[Var, Var] = {}
  newvar = lambda aval: newname(_substitute_vars_in_type(newvars, aval))
  var = lambda v: newvars.get(v) or newvars.setdefault(v, newvar(v.aval))

  def lit(v: Var) -> Optional[Literal]:
    val = consts.get(v)
    if type(val) in core.literalable_types and not np.shape(val):
      return Literal(val, v.aval)
    else:
      return None

  def vars_in_shape(aval: AbstractValue) -> Sequence[Var]:
    if isinstance(aval, DShapedArray):
      return [d for d in aval.shape if isinstance(d, Var)]
    return []

  used = {v for eqn in jaxpr.eqns for invar in eqn.invars
          for v in it.chain([invar], vars_in_shape(invar.aval))}
  used |= {v for outvar in jaxpr.outvars
           for v in it.chain([outvar], vars_in_shape(outvar.aval))}
  new_constvars = [var(v) for v in jaxpr.constvars if v in used and not lit(v)]
  new_constvals = [c for v, c in zip(jaxpr.constvars, constvals)
                   if v in used and not lit(v)]
  new_invars = [var(v) for v in jaxpr.invars]
  new_eqns = []
  for eqn in jaxpr.eqns:
    invars = [lit(v) or var(v) for v in eqn.invars]
    outvars = [var(v) if v in used else core.DropVar(v.aval)
               for v in eqn.outvars]
    new_eqns.append(new_jaxpr_eqn(invars, outvars, eqn.primitive, eqn.params,
                                  eqn.source_info))
  new_outvars = [lit(v) or var(v) for v in jaxpr.outvars]
  new_jaxpr = Jaxpr(new_constvars, new_invars, new_outvars, new_eqns)
  return new_jaxpr, new_constvals

class DynamicJaxprTrace(core.Trace):
  __slots__ = []  # type: ignore

  @property
  def frame(self):
    return self.main.jaxpr_stack[-1]  # pytype: disable=attribute-error

  def new_arg(self, aval):
    tracer = DynamicJaxprTracer(self, aval, source_info_util.current())
    self.frame.tracers.append(tracer)
    self.frame.tracer_to_var[id(tracer)] = var = self.frame.newvar(aval)
    self.frame.invars.append(var)
    return tracer

  def new_const(self, c):
    tracer = self.frame.constid_to_tracer.get(id(c))
    if tracer is None:
      aval = raise_to_shaped(get_aval(c), weak_type=dtypes.is_weakly_typed(c))
      tracer = self._new_const(aval, c)
    return tracer

  pure = lift = new_const

  def _new_const(self, aval, c):
    tracer = DynamicJaxprTracer(self, aval, source_info_util.current())
    self.frame.tracers.append(tracer)
    self.frame.tracer_to_var[id(tracer)] = var = self.frame.newvar(aval)
    self.frame.constid_to_tracer[id(c)] = tracer
    self.frame.constvar_to_val[var] = c
    return tracer

  def sublift(self, t):
    # When lifting closed-over tracers corresponding to this same trace, the
    # variable to lift could have tracers (representing axis size variables) in
    # its shape. We must lift those too!
    tracer = self.frame.constid_to_tracer.get(id(t))
    if tracer is None:
      aval = raise_to_shaped(get_aval(t), weak_type=dtypes.is_weakly_typed(t))
      aval = self._lift_tracers_in_aval(aval)
      tracer = self._new_const(aval, t)
    return tracer

  def _lift_tracers_in_aval(self, aval):
    if (not isinstance(aval, DShapedArray) or
        not any(isinstance(d, Tracer) for d in aval.shape)):
      return aval
    shape = [self.full_raise(d) if isinstance(d, Tracer) else d
             for d in aval.shape]
    return aval.update(shape=tuple(shape))

  def getvar(self, tracer):
    var = self.frame.tracer_to_var.get(id(tracer))
    if var is None:
      raise core.escaped_tracer_error(tracer)
    return var

  def makevar(self, tracer):
    var = self.frame.tracer_to_var.get(id(tracer))
    assert var is None, "a jaxpr variable must be created only once per tracer"
    self.frame.tracers.append(tracer)
    var = self.frame.tracer_to_var[id(tracer)] = self.frame.newvar(tracer.aval)
    return var

  def instantiate_const(self, val):
    if (isinstance(val, Tracer) and val._trace.main is self.main
        and val._trace.sublevel == self.sublevel):
      return val
    else:
      return self.new_const(val)

  def process_primitive(self, primitive, tracers, params):
    if primitive in custom_staging_rules:
      return custom_staging_rules[primitive](self, *tracers, **params)
    return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(self, primitive, tracers, params):
    avals = [t.aval for t in tracers]
    out_avals = primitive.abstract_eval(*avals, **params)
    out_avals = [out_avals] if not primitive.multiple_results else out_avals
    source_info = source_info_util.current()
    out_tracers = [DynamicJaxprTracer(self, a, source_info) for a in out_avals]
    invars = map(self.getvar, tracers)
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn(invars, outvars, primitive, params, source_info)
    self.frame.eqns.append(eqn)
    return out_tracers if primitive.multiple_results else out_tracers.pop()

  def process_call(self, call_primitive, f, tracers, params):
    dim_tracers = _get_tracers_only_in_shapes(tracers)
    in_avals = _tracers_to_avals(dim_tracers + tracers)
    keep_inputs = [False] * len(dim_tracers) + [True] * len(tracers)
    name_stack = source_info_util.current_name_stack()
    with core.new_sublevel():
      jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(
        f, self.main, in_avals, keep_inputs=keep_inputs)
    if params.get('inline', False):
      with source_info_util.set_name_stack(name_stack):
        return core.eval_jaxpr(jaxpr, consts, *dim_tracers, *tracers)
    source_info = source_info_util.current()
    env = {v: t for v, t in zip((*jaxpr.constvars, *jaxpr.invars),
                                (*consts, *dim_tracers, *tracers))
           if isinstance(t, Tracer)}
    subs = partial(_substitute_tracers_in_type, env)
    out_tracers = [DynamicJaxprTracer(self, subs(a), source_info)
                   for a in out_avals]
    invars = map(self.getvar, dim_tracers + tracers)
    constvars = map(self.getvar, map(self.instantiate_const, consts))
    outvars = map(self.makevar, out_tracers)
    new_params = dict(params, call_jaxpr=convert_constvars_jaxpr(jaxpr))
    update_params = call_param_updaters.get(call_primitive)
    if update_params:
      new_params = update_params(new_params, [True] * len(tracers),
                                 len(consts) + len(dim_tracers))
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars,
                        call_primitive, new_params, source_info)
    self.frame.eqns.append(eqn)
    return out_tracers

  def post_process_call(self, call_primitive, out_tracers, params):
    assert False  # unreachable

  def process_map(self, map_primitive, f, tracers, params):
    in_avals = [t.aval for t in tracers]
    axis_name, axis_size = params['axis_name'], params['axis_size']
    reduced_in_avals = [core.mapped_aval(axis_size, in_axis, a)
                        if in_axis is not None else a
                        for a, in_axis in zip(in_avals, params['in_axes'])]
    with core.extend_axis_env(axis_name, axis_size, None):  # type: ignore
      with core.new_sublevel():
        jaxpr, reduced_out_avals, consts = trace_to_subjaxpr_dynamic(
            f, self.main, reduced_in_avals)
      out_axes = params['out_axes_thunk']()
      out_avals = [core.unmapped_aval(axis_size, axis_name, out_axis, a)
                  if out_axis is not None else a
                  for a, out_axis in zip(reduced_out_avals, out_axes)]
      source_info = source_info_util.current()
      out_tracers = [DynamicJaxprTracer(self, a, source_info) for a in out_avals]
      invars = map(self.getvar, tracers)
      constvars = map(self.getvar, map(self.instantiate_const, consts))
      outvars = map(self.makevar, out_tracers)
      new_in_axes = (None,) * len(consts) + params['in_axes']
      new_params = dict(params, in_axes=new_in_axes, out_axes=out_axes,
                        call_jaxpr=convert_constvars_jaxpr(jaxpr))
      del new_params['out_axes_thunk']
      update_params = call_param_updaters.get(map_primitive)
      if update_params:
        new_params = update_params(new_params, [True] * len(tracers), len(consts))
      eqn = new_jaxpr_eqn([*constvars, *invars], outvars, map_primitive,
                          new_params, source_info)
      self.frame.eqns.append(eqn)
    return out_tracers

  def post_process_map(self, map_primitive, out_tracers, params):
    assert False  # unreachable

  def process_custom_jvp_call(self, prim, fun, jvp, tracers):
    in_avals = [t.aval for t in tracers]
    with core.new_sublevel():
      fun_jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, self.main, in_avals)
    closed_fun_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(fun_jaxpr), ())
    main_ = ref(self.main)
    jvp_jaxpr_thunk = _memoize(
        lambda: trace_to_subjaxpr_dynamic(jvp, main_(), 2 * in_avals)[::2])
    out_tracers = [DynamicJaxprTracer(self, a) for a in out_avals]
    invars = map(self.getvar, tracers)
    constvars = map(self.getvar, map(self.instantiate_const, consts))
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, prim.initial_style,
                        dict(fun_jaxpr=closed_fun_jaxpr,
                             jvp_jaxpr_thunk=jvp_jaxpr_thunk,
                             num_consts=len(consts)),
                        source_info_util.current())
    self.frame.eqns.append(eqn)
    return out_tracers

  def post_process_custom_jvp_call(self, out_tracers, _):
    assert False  # unreachable

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees):
    in_avals = [t.aval for t in tracers]
    with core.new_sublevel():
      fun_jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, self.main, in_avals)
    closed_fun_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(fun_jaxpr), ())
    main_ = ref(self.main)
    fwd_jaxpr_thunk = _memoize(
        lambda: trace_to_subjaxpr_dynamic(fwd, main_(), in_avals)[::2])
    out_tracers = [DynamicJaxprTracer(self, a) for a in out_avals]
    invars = map(self.getvar, tracers)
    constvars = map(self.getvar, map(self.instantiate_const, consts))
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, prim.initial_style,
                        dict(fun_jaxpr=closed_fun_jaxpr,
                             fwd_jaxpr_thunk=fwd_jaxpr_thunk,
                             num_consts=len(consts),
                             bwd=bwd, out_trees=out_trees),
                        source_info_util.current())
    self.frame.eqns.append(eqn)
    return out_tracers

  def post_process_custom_vjp_call(self, out_tracers, _):
    assert False  # unreachable

  def process_custom_transpose(self, prim, call, tracers,
                               transpose, out_types,
                               lin_tree, res_tree, out_tree):
    tracers_res, tracers_lin = split_list(tracers, [res_tree.num_leaves])

    in_avals_p = [t.aval for t in tracers]
    in_avals_t = [*[t.aval for t in tracers_res], *out_types]

    with core.new_sublevel():
      call_jaxpr, out_avals, call_consts = trace_to_subjaxpr_dynamic(
          call, self.main, in_avals_p)
    closed_call_jaxpr = core.ClosedJaxpr(
        convert_constvars_jaxpr(call_jaxpr), ())

    transpose_flat, in_tree2 = flatten_fun_nokwargs(
        lu.wrap_init(transpose), treedef_tuple((res_tree, out_tree)))
    transpose_jaxpr, in_avals2, transpose_consts = trace_to_subjaxpr_dynamic(
        transpose_flat, self.main, in_avals_t)
    closed_transpose_jaxpr = core.ClosedJaxpr(
        convert_constvars_jaxpr(transpose_jaxpr), ())

    out_tracers = [DynamicJaxprTracer(self, a) for a in out_avals]
    invars = map(self.getvar, tracers)
    constvars = map(self.getvar, map(self.instantiate_const, call_consts))
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, prim,
                        dict(call_jaxpr=closed_call_jaxpr,
                             transpose_jaxpr=(closed_transpose_jaxpr,
                                              transpose_consts),
                             num_consts=len(call_consts)),
                        source_info_util.current())
    self.frame.eqns.append(eqn)
    return out_tracers


custom_staging_rules: Dict[Primitive, Callable] = {}

def _memoize(thunk):
  if config.jax_check_tracer_leaks:
    return thunk

  cell = []
  saved_state = core.thread_local_state.trace_state.copy()
  def memoized():
    if not cell:
      prev_state = core.thread_local_state.trace_state
      core.thread_local_state.trace_state = saved_state
      try:
        cell.append(thunk())
      finally:
        core.thread_local_state.trace_state = prev_state
    return cell[0]
  return memoized


class DebugInfo(NamedTuple):
  func_src_info: str
  traced_for: str
  arg_info: Callable[[int], str]


def debug_info_final(fn: lu.WrappedFun, traced_for: str) -> DebugInfo:
  in_tree, has_kwargs = flattened_fun_in_tree(fn) or (None, False)
  return debug_info(fn.f, in_tree, has_kwargs, traced_for)

def debug_info(fn: Callable, in_tree: Optional[PyTreeDef], has_kwargs: bool,
               traced_for: str) -> DebugInfo:
  func_src_info = fun_sourceinfo(fn)
  if in_tree is not None:
    arg_info = partial(arg_info_pytree, fn, in_tree, has_kwargs)
  else:
    arg_info = arg_info_flattened  # type: ignore
  return DebugInfo(func_src_info, traced_for, arg_info)

def fun_sourceinfo(fun: Callable):
  while isinstance(fun, functools.partial):
    fun = fun.func
  fun = inspect.unwrap(fun)
  try:
    filename = fun.__code__.co_filename
    lineno = fun.__code__.co_firstlineno
    line_info = f"{fun.__name__} at {filename}:{lineno}"
    return line_info
  except AttributeError:
    return "<unknown>"

def arg_info_pytree(fn: Callable, in_tree: PyTreeDef, has_kwargs: bool,
                    flat_pos: List[int]) -> str:
  dummy_args = [False] * in_tree.num_leaves
  for i in flat_pos: dummy_args[i] = True
  if has_kwargs:
    args, kwargs = tree_unflatten(in_tree, dummy_args)
  else:
    args, kwargs = tree_unflatten(in_tree, dummy_args), {}
  try:
    ba = inspect.signature(fn).bind(*args, **kwargs)
  except (TypeError, ValueError):
    return arg_info_flattened(flat_pos)
  arg_names = [f"'{name}'" for name, x in ba.arguments.items()
               if any(tree_leaves(x))]
  if len(arg_names) == 1:
    return f"the argument {arg_names[0]}"
  elif len(arg_names) == 2:
    return f"the arguments {arg_names[0]} and {arg_names[1]}"
  else:
    *rest, last = arg_names
    return f"the arguments {', '.join(rest)}, and {last}"

def arg_info_flattened(flat_pos: List[int]) -> str:
  if len(flat_pos) > 1:
    return f"the argument passed at flattened positions {flat_pos}"
  else:
    return f"the argument passed at flattened position {flat_pos[0]}"


@profiler.annotate_function
def trace_to_jaxpr_dynamic(fun: lu.WrappedFun,
                           in_avals: Sequence[AbstractValue],
                           debug_info: Optional[DebugInfo] = None,
                           *,
                           keep_inputs: Optional[List[bool]] = None):
  with core.new_main(DynamicJaxprTrace, dynamic=True) as main:  # type: ignore
    main.debug_info = debug_info  # type: ignore
    main.jaxpr_stack = ()  # type: ignore
    jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(
      fun, main, in_avals, keep_inputs=keep_inputs)
    del main, fun
  return jaxpr, out_avals, consts

def trace_to_subjaxpr_dynamic(fun: lu.WrappedFun, main: core.MainTrace,
                              in_avals: Sequence[AbstractValue], *,
                              keep_inputs: Optional[List[bool]] = None):
  # In general, the Tracers passed to ther Python callable underlying `fun` may
  # correspond to a subset of `in_avals` (i.e. a subset of the input binders in
  # the jaxpr). For example:
  #
  #   n = core.DShapedArray((), jnp.dtype('int32'), weak_type=False)
  #   a = core.DShapedArray((n,), jnp.dtype('float32'), weak_type=False)
  #   b = core.DShapedArray((n,), jnp.dtype('float32'), weak_type=False)
  #
  #   @lu.wrap_init
  #   def f(x, y):
  #     return x, y
  #
  #   jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(f, [n, a, b],
  #                                           keep_inputs=[False, True, True])
  #   print(jaxpr)
  #   # { lambda ; a:i32[] b:f32[a] c:f32[a]. let  in (b, c) }
  #
  # The abstract values passed to trace_to_jaxpr_dynamic are in direct
  # correspondence to the input binders (invars) of the jaxpr it returns. But in
  # general the Tracers passed to the function f correspond only to a subset of
  # those abstract values. That's because axis size variables may not be
  # explicit arguments to f, while we make everything explicit in the jaxpr.
  keep_inputs = [True] * len(in_avals) if keep_inputs is None else keep_inputs

  frame = JaxprStackFrame()
  with extend_jaxpr_stack(main, frame), source_info_util.reset_name_stack():
    trace = DynamicJaxprTrace(main, core.cur_sublevel())
    in_tracers = _avals_to_tracers(trace, in_avals)
    in_tracers_ = [t for t, keep in zip(in_tracers, keep_inputs) if keep]
    ans = fun.call_wrapped(*in_tracers_)
    out_tracers = map(trace.full_raise, ans)
    jaxpr, consts = frame.to_jaxpr(out_tracers)
    del fun, main, trace, frame, in_tracers, out_tracers, ans
  return jaxpr, [v.aval for v in jaxpr.outvars], consts

@contextlib.contextmanager
def extend_jaxpr_stack(main, frame):
  main.jaxpr_stack = main.jaxpr_stack + (frame,)
  try:
    yield
  finally:
    assert frame is main.jaxpr_stack[-1]
    main.jaxpr_stack = main.jaxpr_stack[:-1]

@profiler.annotate_function
def trace_to_jaxpr_final(fun: lu.WrappedFun,
                         in_avals: Sequence[AbstractValue],
                         debug_info: Optional[DebugInfo] = None):
  with core.new_base_main(DynamicJaxprTrace) as main:  # type: ignore
    main.debug_info = debug_info  # type: ignore
    main.jaxpr_stack = ()  # type: ignore
    with core.new_sublevel():
      jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, main, in_avals)
    del fun, main
  return jaxpr, out_avals, consts

def partial_eval_to_jaxpr_dynamic(fun: lu.WrappedFun, in_pvals: Sequence[PartialVal]):
  # This function provides a partial evaluation behavior used by Flax. We can't
  # use trace_to_jaxpr directly because of an interaction with the curent
  # custom_derivatives.py, which we work around by adding the EvalTrace.
  # TODO(mattjj): alias to trace_to_jaxpr after revising custom_derivatives.py
  with core.new_main(core.EvalTrace, dynamic=True) as _:  # type: ignore
    return trace_to_jaxpr(fun, in_pvals)


def _avals_to_tracers(
    trace: DynamicJaxprTrace, in_avals: Sequence[AbstractValue]
  )  -> Sequence[Tracer]:
  # Create input Tracers given input AbstractValues, each of which can contain
  # other AbstractValues. That is, each element `a` of `in_avals` can have
  # abstract values in its shape, which must occur to the left of `a`.
  env: Dict[AvalId, Tracer] = {}
  in_tracers: List[Tracer] = []
  for a in in_avals:
    t = env[id(a)] = trace.new_arg(_substitute_tracers_in_aval(env, a))
    in_tracers.append(t)
  return in_tracers

def _substitute_tracers_in_aval(
    env: Dict[AvalId, Tracer], a: AbstractValue
  ) -> AbstractValue:
  # Substitute Tracers into a given AbstractValue using the given environment.
  # That is, the input is an AbstractValue possibly containing AbstractValues,
  # and the output is an AbstractValue possibly containing Tracers.
  if (isinstance(a, DShapedArray) and
      any(isinstance(d, AbstractValue) for d in a.shape)):
    shape = [env[id(d)] if isinstance(d, AbstractValue) else d for d in a.shape]
    return a.update(shape=tuple(shape))
  return a

def _tracers_to_avals(tracers: Sequence[Tracer]) -> List[AbstractValue]:
  # Replace Tracers with corresponding abstract values, handling Tracers in
  # shapes and ensuring each Tracer object is mapped to a single AbstractValue.
  env: Dict[TracerId, AbstractValue] = {}
  avals: List[AbstractValue] = []
  for t in tracers:
    aval = env.get(id(t))
    if aval is None:
      aval = env[id(t)] = _substitute_avals_in_aval(env, t.aval)
    avals.append(aval)
  return avals

def _substitute_avals_in_aval(
    env: Dict[TracerId, AbstractValue], a: AbstractValue
  ) -> AbstractValue:
  # Substitute AbstractValues into given AbstractValue using given environment.
  # That is, the input is an AbstractValue possibly containing Tracers and the
  # output is an AbstractValue possibly containing AbstractValues.
  if (isinstance(a, DShapedArray) and
      any(isinstance(d, Tracer) for d in a.shape)):
    shape = [env.setdefault(id(d), d.aval) if isinstance(d, Tracer) else d
             for d in a.shape]
    return a.update(shape=tuple(shape))
  else:
    return a

def _substitute_vars_in_type(
    env: Dict[Var, Var], a: AbstractValue
  ) -> AbstractValue:
  # Substitutes variables into a given AbstractValue using given environment.
  # That is, the input is an AbstractValue possibly containing Vars, and the
  # output is an aval possibly containing Vars.
  if isinstance(a, DShapedArray) and any(isinstance(d, Var) for d in a.shape):
    shape = [env[d] if isinstance(d, Var) else d for d in a.shape]
    return a.update(shape=tuple(shape))
  else:
    return a

def _substitute_tracers_in_type(
    env: Dict[Var, Tracer], a: AbstractValue
  ) -> AbstractValue:
  # Substitutes variables into a given AbstractValue using given environment.
  # That is, the input is an AbstractValue possibly containing Vars, and the
  # output is an aval possibly containing Tracers.
  if isinstance(a, DShapedArray) and any(isinstance(d, Var) for d in a.shape):
    shape = [env[d] if isinstance(d, Var) else d for d in a.shape]
    return a.update(shape=tuple(shape))
  else:
    return a

def _get_tracers_only_in_shapes(in_tracers: Sequence[Tracer]) -> List[Tracer]:
  # In DynamicJaxprTrace.process_call (e.g. for handling jit-of-jit) we want to
  # extract Tracers from the shapes of arguments so as to elaborate them into
  # explicit inputs to the call that appears in the jaxpr. Some of these Tracers
  # may already appear as explicit inputs, so we only need to get those present
  # exclusively in shapes.
  return _get_tracers_in_shapes({id(t) for t in in_tracers}, in_tracers)

def _get_tracers_in_shapes(seen: Set[TracerId], in_tracers: Sequence[Tracer]
                           ) -> List[Tracer]:
  dim_tracers: List[Tracer] = []
  for t in in_tracers:
    if isinstance(t.aval, core.DShapedArray):
      for d in t.aval.shape:
        if isinstance(d, Tracer) and id(d) not in seen:
          seen.add(id(d))
          dim_tracers.append(d)
  return dim_tracers

# TODO(mattjj): the following are deprecated; update callers to _nounits version
# See https://github.com/google/jax/pull/9498
@lu.transformation
def trace_to_subjaxpr(main: core.MainTrace, instantiate: Union[bool, Sequence[bool]],
                      pvals: Sequence[PartialVal]):
  assert all([isinstance(pv, PartialVal) for pv in pvals]), pvals
  trace = main.with_cur_sublevel()
  in_tracers = map(trace.new_arg, pvals)
  ans = yield in_tracers, {}
  assert isinstance(ans, (list, tuple)), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  assert all(isinstance(x, core.Tracer) or core.valid_jaxtype(x) for x in ans), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  instantiate = [instantiate] * len(ans) if isinstance(instantiate, bool) else instantiate
  out_tracers = map(trace.full_raise, map(core.full_lower, ans))
  out_tracers = map(partial(instantiate_const_at, trace), instantiate, out_tracers)
  jaxpr, consts, env = tracers_to_jaxpr(in_tracers, out_tracers)
  out_pvals = [t.pval for t in out_tracers]
  del trace, in_tracers, out_tracers
  yield jaxpr, (out_pvals, consts, env)

@lu.transformation_with_aux
def partial_eval_wrapper(pvs: Sequence[Optional[AbstractValue]], *consts):
  py_args = map(PartialVal, zip(pvs, consts))
  jaxpr, (out_pvals, consts, env) = yield (py_args,), {}
  out_pvs, out_consts = unzip2(out_pvals)
  out = tuple(out_consts) + tuple(consts)
  yield out, (out_pvs, jaxpr, env)
