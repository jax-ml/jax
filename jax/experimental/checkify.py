# Copyright 2021 Google LLC
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

from dataclasses import dataclass
from functools import partial
import itertools as it
from typing import Union, Optional, Callable, Dict

import numpy as np

import jax.numpy as jnp

from jax import core
from jax import linear_util as lu
from jax.api_util import flatten_fun
from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node
from jax._src import source_info_util, traceback_util
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax import control_flow
from jax._src.util import as_hashable_function, unzip2, split_list

source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


## Utils

def popattr(obj, attrname):
  val = getattr(obj, attrname)
  delattr(obj, attrname)
  return val

def setnewattr(obj, name, val):
  sentinel = object()
  assert getattr(obj, name, sentinel) is sentinel
  setattr(obj, name, val)


## Error value data type and functional assert.

@dataclass(frozen=True)
class Error:
  err: Union[bool, core.Tracer]
  code: Union[int, core.Tracer]
  msgs: Dict[int, str]

  def get(self) -> Optional[str]:
    assert np.shape(self.err) == np.shape(self.code)
    if np.size(self.err) == 1:
      if self.err:
        return self.msgs[int(self.code)]
    else:
      return '\n'.join(f'at mapped index {", ".join(map(str, idx))}: '  # type: ignore
                       f'{self.msgs[int(self.code[idx])]}'              # type: ignore
                       for idx, e in np.ndenumerate(self.err) if e) or None
    return None

register_pytree_node(Error,
                     lambda e: ((e.err, e.code), tuple(sorted(e.msgs.items()))),
                     lambda msgs, data: Error(*data, dict(msgs)))  # type: ignore

init_error = Error(False, 0, {})
next_code = it.count(1).__next__  # globally unique ids, could be uuid4


Bool = Union[bool, core.Tracer]
Int = Union[int, core.Tracer]

def assert_func(error: Error, pred: Bool, msg: str) -> Error:
  code = next_code()
  out_err = error.err | jnp.logical_not(pred)
  out_code = lax.select(error.err, error.code, code)
  return Error(out_err, out_code, {code: msg, **error.msgs})


## Checkify transformation for plumbing functional error values.

class ErrorTracer(core.Tracer):
  def __init__(self, trace, val):
    self._trace = trace
    self.val = val
    core.get_aval(val), val
  aval = property(lambda self: core.get_aval(self.val))
  full_lower = lambda self: self

class ErrorTrace(core.Trace):
  pure = lift = lambda self, val: ErrorTracer(self, val)

  def sublift(self, tracer):
    return ErrorTracer(self, tracer.val)

  def process_primitive(self, primitive, tracers, params):
    in_vals = [t.val for t in tracers]
    rule = error_checks.get(primitive)
    if rule:
      out, self.main.error = rule(self.main.error, *in_vals, **params)  # type: ignore
    else:
      out = primitive.bind(*in_vals, **params)
    if primitive.multiple_results:
      return [ErrorTracer(self, x) for x in out]
    else:
      return ErrorTracer(self, out)

  def process_call(self, primitive, f, tracers, params):
    in_vals = [t.val for t in tracers]
    e = popattr(self.main, 'error')
    f, msgs = check_errors_subtrace(f, self.main, tuple(e.msgs.items()))
    params_ = dict(params, donated_invars=(False, False, *params['donated_invars']))
    err, code, *out_vals = primitive.bind(f, e.err, e.code, *in_vals, **params_)
    setnewattr(self.main, 'error', Error(err, code, msgs()))
    return [ErrorTracer(self, x) for x in out_vals]

  def process_map(self, primitive, f, tracers, params):
    in_vals = [t.val for t in tracers]
    e = popattr(self.main, 'error')
    f, msgs = check_errors_subtrace(f, self.main, tuple(e.msgs.items()))

    @as_hashable_function(closure=params['out_axes_thunk'])
    def new_out_axes_thunk():
      return (0, 0, *params['out_axes_thunk']())

    params_ = dict(params, in_axes=(None, None, *params['in_axes']),
                   out_axes_thunk=new_out_axes_thunk,
                   donated_invars=(False, False, *params['donated_invars']))
    errs, codes, *outs = primitive.bind(f, e.err, e.code, *in_vals, **params_)
    err, code = _reduce_any_error(errs, codes)
    setnewattr(self.main, 'error', Error(err, code, msgs()))
    return [ErrorTracer(self, x) for x in outs]

  def post_process_call(self, primitive, tracers, params):
    vals = [t.val for t in tracers]
    main = self.main
    e = popattr(self.main, 'error')
    err, code, main.msgs = e.err, e.code, e.msgs
    def todo(vals):
      trace = main.with_cur_sublevel()
      err, code, *vals = vals
      return [ErrorTracer(trace, x) for x in vals]
    return (err, code, *vals), todo

  def post_process_map(self, primitive, tracers, params):
    vals = [t.val for t in tracers]
    main = self.main
    e = popattr(self.main, 'error')
    err, code, main.msgs = e.err, e.code, e.msgs
    def todo(vals):
      trace = main.with_cur_sublevel()
      err, code, *vals = vals
      return [ErrorTracer(trace, x) for x in vals]
    def out_axes_transform(out_axes):
      return (0, 0, *out_axes)
    return (err, code, *vals), (todo, out_axes_transform)

def _reduce_any_error(errs, codes):
  errs_, codes_ = lax.sort_key_val(errs, codes, dimension=0)
  return errs_[-1], codes_[-1]

ErrorCheckRule = Callable
error_checks: Dict[core.Primitive, ErrorCheckRule] = {}

def check_errors_flat(fun: lu.WrappedFun, *args):
  fun, msgs = check_errors_subtrace(fun)
  fun = check_errors_toplevel(fun)
  err, code, *out_vals = fun.call_wrapped(*args)
  return (err, code, out_vals), msgs()

@lu.transformation
def check_errors_toplevel(*args):
  error = init_error
  with core.new_main(ErrorTrace) as main:
    msgs = tuple(error.msgs.items())
    outs = yield (main, msgs, error.err, error.code, *args), {}
    del main
  yield outs

@lu.transformation_with_aux
def check_errors_subtrace(main, msgs, err, code, *args):
  setnewattr(main, 'error', Error(err, code, dict(msgs)))
  trace = main.with_cur_sublevel()
  in_tracers = [ErrorTracer(trace, x) for x in args]
  out = yield in_tracers, {}
  out_tracers = map(trace.full_raise, out)
  out_vals = [t.val for t in out_tracers]
  err, code, msgs = main.error.err, main.error.code, main.error.msgs
  del main.error
  yield (err, code, *out_vals), msgs

# TODO take (error_aval, code_aval) instead of error here?
def checkify_jaxpr(jaxpr, error):
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
  f, msgs = check_errors_subtrace(f)
  f = check_errors_traceable(f, tuple(error.msgs.items()))
  err_aval = core.raise_to_shaped(core.get_aval(error.err))
  code_aval = core.raise_to_shaped(core.get_aval(error.code))
  avals_in = [err_aval, code_aval, *jaxpr.in_avals]
  jaxpr_out, _, literals_out = pe.trace_to_jaxpr_dynamic(f, avals_in)
  return core.ClosedJaxpr(jaxpr_out, literals_out), msgs()

# TODO dedup with check_errors_toplevel
@lu.transformation
def check_errors_traceable(msgs, err, code, *args):
  with core.new_main(ErrorTrace) as main:
    outs = yield (main, msgs, err, code, *args), {}
    del main
  yield outs


## assert primitive

def assert_(pred: Bool, msg: str) -> None:
  code = next_code()
  return assert2_(pred, code, {code: msg})

def assert2_(pred: Bool, code: Int, msgs: Dict[int, str]) -> None:
  return assert_p.bind(pred, code, msgs=msgs)

assert_p = core.Primitive('assert')
assert_p.multiple_results = True  # zero results

@assert_p.def_impl
def assert_impl(pred, code, *, msgs):
  assert pred, msgs[int(code)]
  return []

@assert_p.def_abstract_eval
def assert_abstract_eval(pred, code, *, msgs):
  raise Exception("can't be staged!")


## checkify rules

def summary() -> str:
  return str(source_info_util.summarize(source_info_util.current()))

def nan_error_check(prim, error, *in_vals, **params):
  out = prim.bind(*in_vals, **params)
  no_nans = jnp.logical_not(jnp.any(jnp.isnan(out)))
  msg = f"nan generated by primitive {prim.name} at {summary()}"
  return out, assert_func(error, no_nans, msg)

error_checks[lax.sin_p] = partial(nan_error_check, lax.sin_p)
error_checks[lax.cos_p] = partial(nan_error_check, lax.cos_p)

def gather_error_check(error, operand, start_indices, *,
                       dimension_numbers, slice_sizes, unique_indices,
                       indices_are_sorted, mode, fill_value):
  out = slicing.gather_p.bind(
      operand, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=slice_sizes, unique_indices=unique_indices,
      indices_are_sorted=indices_are_sorted, mode=mode, fill_value=fill_value)

  # compare to OOB masking logic in lax._gather_translation_rule
  dnums = dimension_numbers
  operand_dims = np.array(operand.shape)

  upper_bound = operand_dims[np.array(dnums.start_index_map)]
  upper_bound -= np.array(slice_sizes)[np.array(dnums.start_index_map)]
  all_inbounds = jnp.all((start_indices >= 0) & (start_indices <= upper_bound))

  msg = f"out-of-bounds indexing at {summary()}"
  return out, assert_func(error, all_inbounds, msg)
error_checks[slicing.gather_p] = gather_error_check

def cond_error_check(error, index, *ops, branches, linear):
  new_branches, msgs_ = unzip2(checkify_jaxpr(jxpr, error) for jxpr in branches)
  new_linear = (False, False, *linear)
  err, code, *outs = control_flow.cond_p.bind(
      index, error.err, error.code, *ops,
      branches=tuple(new_branches), linear=new_linear)
  new_msgs = {k:v for d in it.chain([error.msgs], msgs_) for k, v in d.items()}
  return outs, Error(err, code, new_msgs)
error_checks[control_flow.cond_p] = cond_error_check

def scan_error_check(error, *in_flat, reverse, length, jaxpr, num_consts, num_carry, linear, unroll):
  consts, carry, xs = split_list(in_flat, [num_consts, num_carry])
  checked_jaxpr, msgs_ = checkify_jaxpr(jaxpr, error)
  new_linear = (False, False, *linear)
  new_in_flat = [*consts, error.err, error.code, *carry, *xs]
  err, code, *outs = control_flow.scan_p.bind(
      *consts, *new_in_flat,
      reverse=reverse, length=length, jaxpr=checked_jaxpr,
      num_consts=len(consts), num_carry=len(carry)+2,
      linear=new_linear, unroll=unroll)
  new_msgs = {**error.msgs, **msgs_}
  return outs, Error(err, code, new_msgs)
error_checks[control_flow.scan_p] = scan_error_check

# TODO(mattjj,lenamartens): currently we bundle effectful-assert-discharging
# with the error-check-adding transformation (checkify), but they could be
# separated into two orthogonal transformations.
def assert_discharge_rule(error, pred, code, *, msgs):
  out_err = error.err | jnp.logical_not(pred)
  out_code = lax.select(error.err, error.code, code)
  return [], Error(out_err, out_code, {**error.msgs, **msgs})
error_checks[assert_p] = assert_discharge_rule


## checkify api

def checkify(fun: Callable) -> Callable:
  @traceback_util.api_boundary
  def checked_fun(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    f, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
    (err, code, out_flat), msgs = check_errors_flat(f, *args_flat)
    out = tree_unflatten(out_tree(), out_flat)
    return Error(err, code, msgs), out
  return checked_fun
