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

import numpy as onp
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax.numpy as np

from jax import core
from jax.core import Trace, Tracer, new_master
from jax import linear_util as lu
from jax.util import partial, safe_map, wrap_name
from jax.interpreters import xla
from jax.interpreters import partial_eval as pe

import inspect
from jax.api_util import (wraps, flatten_fun_nokwargs)
from jax.tree_util import (tree_flatten, tree_unflatten)

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

  if np.isnan(query):
    if np.any(np.isnan(vals)):
      raise FoundValue('NaN')
  elif np.isinf(query):
    if np.any(np.isinf(vals)):
      raise FoundValue('Found Inf')
  elif np.isscalar(query):
    if np.any(vals == query):
      raise FoundValue(str(query))
  else:
    raise ValueError('Malformed Query: {}'.format(query))

### Helper Functions

def callback_fun(fun : lu.WrappedFun, in_vals, callback, strip_calls):
  fun = callback_subtrace(fun)
  fun = _callback_fun(fun, callback, strip_calls)
  return fun.call_wrapped(*in_vals)

@lu.transformation
def callback_subtrace(master, *in_vals, **params):
  trace = CallbackTrace(master, core.cur_sublevel())
  in_tracers = [CallbackTracer(trace, val) for val in in_vals]
  outs = yield in_tracers, params
  out_tracers = map(trace.full_raise, outs)
  out_vals = [t.val for t in out_tracers]
  yield out_vals

@lu.transformation
def _callback_fun(callback, strip_calls, *in_vals, **params):
  with new_master(CallbackTrace) as master:
    master.callback = callback # NOTE: Is this OK?
    master.strip_calls = strip_calls
    out_vals = yield (master,) + in_vals, params
    del master
  yield out_vals

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
    vals_in = [t.val for t in tracers]
    vals_out = self.master.callback(primitive, vals_in, params)  # type: ignore
    if primitive.multiple_results:
      return [CallbackTracer(self, val) for val in vals_out]
    return CallbackTracer(self, vals_out)

  def process_call(self, call_primitive, f: lu.WrappedFun, tracers, params):
    if self.master.strip_calls: # type: ignore
      return f.call_wrapped(*tracers)
    vals_in = [t.val for t in tracers]
    f = callback_subtrace(f, self.master)
    vals_out = call_primitive.bind(f, *vals_in, **params)
    return [CallbackTracer(self, val) for val in vals_out]
