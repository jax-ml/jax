# Copyright 2020 Google LLC
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

from typing import Any, Callable, Dict, Sequence, Union

import jax.numpy as jnp

from jax import core
from jax.core import Trace, Tracer, jaxpr_as_fun
from jax import lax
from jax import custom_derivatives as cd
from jax.interpreters import partial_eval as pe
from jax import linear_util as lu
from jax._src.util import partial, safe_map, wraps, split_list
from jax._src.lax import control_flow as lcf

import inspect
from jax.api_util import flatten_fun_nokwargs
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure, tree_map

map = safe_map

### Public

def callback_transform(
    fun: Callable, callback: Callable, strip_calls: bool=False) -> Callable:
  _check_callable(fun)

  @wraps(fun)
  def wrapped_fun(*args):
    args_flat, in_tree  = tree_flatten(args)
    f = lu.wrap_init(fun)
    flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
    out_flat = callback_fun(flat_fun, args_flat, callback, strip_calls)
    return tree_unflatten(out_tree(), out_flat)

  return wrapped_fun

### Example Transform

def find_by_value(fun: Callable, queries) -> Callable:
  def find_callback(
          prim: core.Primitive,
          vals: Sequence[core.Tracer],
          params: Dict[str, Any]) -> Union[core.Tracer, Sequence[core.Tracer]]:
    vals = prim.bind(*vals, **params)
    _contains_query(vals, queries)
    return vals
  return callback_transform(fun, find_callback, True)

def rewrite(fun: Callable, rules) -> Callable:
  assert isinstance(rules, dict)
  def rewrite_callback(
          prim: core.Primitive,
          vals: Sequence[core.Tracer],
          params: Dict[str, Any]) -> Union[core.Tracer, Sequence[core.Tracer]]:
    if prim in rules:
      return rules[prim](*vals, **params)
    return prim.bind(*vals, **params)
  return callback_transform(fun, rewrite_callback)

class FoundValue(Exception):
  pass

def _contains_query(vals, query):
  if isinstance(query, tuple):
    return map(partial(_contains_query, vals), query)

  if jnp.isnan(query):
    if jnp.any(jnp.isnan(vals)):
      raise FoundValue('NaN')
  elif jnp.isinf(query):
    if jnp.any(jnp.isinf(vals)):
      raise FoundValue('Found Inf')
  elif jnp.isscalar(query):
    if jnp.any(vals == query):
      raise FoundValue(str(query))
  else:
    raise ValueError('Malformed Query: {}'.format(query))

### Helper Functions

def callback_fun(fun : lu.WrappedFun, in_vals, callback, strip_calls):
  fun = callback_subtrace(fun)
  fun = _callback_fun(fun, callback, strip_calls)
  return fun.call_wrapped(*in_vals)

@lu.transformation
def callback_subtrace(main, *in_vals, **params):
  trace = CallbackTrace(main, core.cur_sublevel())
  in_tracers = [CallbackTracer(trace, val) for val in in_vals]
  outs = yield in_tracers, params
  out_tracers = map(trace.full_raise, outs)
  out_vals = [t.val for t in out_tracers]
  yield out_vals

@lu.transformation
def _callback_fun(callback, strip_calls, *in_vals, **params):
  with core.new_main(CallbackTrace) as main:
    main.callback = callback # NOTE: Is this OK?
    main.strip_calls = strip_calls
    out_vals = yield (main,) + in_vals, params
    del main
  yield out_vals

def callback_jaxpr(closed_jaxpr, callback, strip_calls):
  fun = lu.wrap_init(jaxpr_as_fun(closed_jaxpr))
  fun = callback_subtrace(fun)
  fun = _callback_fun(fun, callback, strip_calls)
  avals_in = closed_jaxpr.in_avals
  jaxpr_out, consts = cd._initial_style_jaxpr(fun, avals_in)
  return core.ClosedJaxpr(jaxpr_out, consts)

def _check_callable(fun):
  if not callable(fun):
    raise TypeError(f"Expected a callable value, got {fun}")
  if inspect.isgeneratorfunction(fun):
    raise TypeError(f"Expected a function, got a generator function: {fun}")

### Tracer

class CallbackTracer(Tracer):
  __slots__ = ['val']

  def __init__(self, trace, val):
    self._trace = trace
    self.val = val

  @property
  def aval(self):
    return core.get_aval(self.val)

  def full_lower(self):
    return self

class CallbackTrace(Trace):
  def pure(self, val):
    return CallbackTracer(self, val)

  def lift(self, val):
    return CallbackTracer(self, val)

  def sublift(self, val):
    return CallbackTracer(self, val.val)

  def process_primitive(self, primitive, tracers, params):
    if primitive in custom_callback_rules:
      return custom_callback_rules[primitive](self, *tracers, **params)
    vals_in = [t.val for t in tracers]
    vals_out = self.main.callback(primitive, vals_in, params)  # type: ignore
    if primitive.multiple_results:
      return [CallbackTracer(self, val) for val in vals_out]
    return CallbackTracer(self, vals_out)

  def process_call(self, call_primitive, f: lu.WrappedFun, tracers, params):
    if self.main.strip_calls: # type: ignore
      return f.call_wrapped(*tracers)
    vals_in = [t.val for t in tracers]
    f = callback_subtrace(f, self.main)
    vals_out = call_primitive.bind(f, *vals_in, **params)
    return [CallbackTracer(self, val) for val in vals_out]

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers):
    vals_in = [t.val for t in tracers]
    fun = callback_subtrace(fun, self.main)
    jvp = callback_subtrace(jvp, self.main)
    out = primitive.bind(fun, jvp, *vals_in)
    return safe_map(self.pure, out)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers,
                              out_trees):
    vals_in = [t.val for t in tracers]
    fun = callback_subtrace(fun, self.main)
    fwd = callback_subtrace(fwd, self.main)
    bwd = callback_subtrace(bwd, self.main)
    out = primitive.bind(fun, fwd, bwd, *vals_in, out_trees=out_trees)
    return safe_map(self.pure, out)

custom_callback_rules: Dict[Any, Any] = {}

def _scan_callback_rule(trace, *tracers, reverse, length, num_consts, num_carry,
                        jaxpr, linear, unroll):
  const_tracers, carry_tracers, xs_tracers = split_list(tracers, [num_consts, num_carry])
  carry_avals, xs_avals = tree_map(lambda x: x.aval, (carry_tracers, xs_tracers))
  const_vals, carry_vals, xs_vals = tree_map(lambda x: x.val, (const_tracers, carry_tracers, xs_tracers))

  x_tracers = [t[0] for t in xs_tracers]
  x_avals = [t.aval for t in x_tracers]

  body_fun = jaxpr_as_fun(jaxpr)

  def new_body(*vals):
    out = body_fun(*vals)
    out_carry, y = split_list(out, [num_carry])
    return out_carry, y
  main = trace.main
  new_body = callback_transform(new_body, main.callback, strip_calls=main.strip_calls)  # type: ignore
  in_tree = tree_structure(carry_avals + xs_avals)
  new_jaxpr, new_consts, _ = lcf._initial_style_jaxpr(
      new_body, in_tree, tuple(carry_avals + x_avals))
  vals = tuple(it.chain(new_consts, carry_vals, xs_vals))
  out_vals = lax.scan_p.bind(*vals, reverse=reverse, length=length,
                             num_consts=len(new_consts), num_carry=num_carry,
                             jaxpr=new_jaxpr, linear=linear, unroll=unroll)
  return safe_map(trace.pure, out_vals)

custom_callback_rules[lax.scan_p] = _scan_callback_rule


def _while_callback_rule(trace, *tracers, cond_jaxpr, body_jaxpr,
                         cond_nconsts, body_nconsts):
  cond_const_tracers, body_const_tracers, init_tracers = split_list(
            tracers, [cond_nconsts, body_nconsts])
  init_avals = safe_map(lambda x: x.aval, init_tracers)
  cond_const_vals, body_const_vals, init_vals = tree_map(
      lambda x: x.val, (cond_const_tracers, body_const_tracers, init_tracers))

  body_fun = jaxpr_as_fun(body_jaxpr)
  cond_fun = jaxpr_as_fun(cond_jaxpr)

  def cond(*carry):
    return cond_fun(*it.chain(cond_const_vals, carry))

  def body(*carry):
    return body_fun(*it.chain(body_const_vals, carry))

  main = trace.main
  new_cond = callback_transform(cond, main.callback, strip_calls=main.strip_calls)  # type: ignore
  new_body = callback_transform(body, main.callback, strip_calls=main.strip_calls)  # type: ignore
  in_tree = tree_structure(init_avals)

  new_cond_jaxpr, new_cond_consts, _ = lcf._initial_style_jaxpr(new_cond, in_tree, tuple(init_avals))
  new_body_jaxpr, new_body_consts, _ = lcf._initial_style_jaxpr(new_body, in_tree, tuple(init_avals))
  out = lcf.while_p.bind(
      *it.chain(new_cond_consts, new_body_consts, init_vals),
      cond_nconsts=len(new_cond_consts),
      body_nconsts=len(new_body_consts),
      cond_jaxpr=new_cond_jaxpr,
      body_jaxpr=new_body_jaxpr)
  return safe_map(trace.pure, out)

custom_callback_rules[lax.while_p] = _while_callback_rule

def _custom_derivative_call_jaxpr_callback_rule(primitive, trace, *tracers,
                                                fun_jaxpr, num_consts, **params):
  main = trace.main
  vals = [t.val for t in tracers]

  new_closed_jaxpr = callback_jaxpr(fun_jaxpr, main.callback, strip_calls=main.strip_calls)
  if primitive == cd.custom_jvp_call_jaxpr_p:
    thunk_name = 'jvp_jaxpr_thunk'
  elif primitive == cd.custom_vjp_call_jaxpr_p:
    thunk_name = 'fwd_jaxpr_thunk'
    params['bwd'] = callback_subtrace(params['bwd'], main)
  else:
    raise NotImplementedError(primitive)

  thunk = params.pop(thunk_name)
  @pe._memoize
  def new_thunk():
    thunk_jaxpr = core.ClosedJaxpr(*thunk())
    closed_jaxpr = callback_jaxpr(thunk_jaxpr, main.callback, main.strip_calls)
    return closed_jaxpr.jaxpr, closed_jaxpr.literals

  params[thunk_name] = new_thunk
  new_fun_jaxpr, new_consts = new_closed_jaxpr.jaxpr, new_closed_jaxpr.literals
  closed_fun_jaxpr = core.ClosedJaxpr(pe.convert_constvars_jaxpr(new_fun_jaxpr), ())
  new_num_consts = len(new_consts) + num_consts
  out = primitive.bind(*it.chain(new_consts, vals), fun_jaxpr=closed_fun_jaxpr,
                       num_consts=new_num_consts, **params)
  return safe_map(trace.pure, out)

custom_callback_rules[cd.custom_jvp_call_jaxpr_p] = partial(
    _custom_derivative_call_jaxpr_callback_rule, cd.custom_jvp_call_jaxpr_p)
custom_callback_rules[cd.custom_vjp_call_jaxpr_p] = partial(
    _custom_derivative_call_jaxpr_callback_rule, cd.custom_vjp_call_jaxpr_p)
