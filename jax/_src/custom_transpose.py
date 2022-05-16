# Copyright 2022 Google LLC
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
from typing import Any, Callable, Optional, Tuple

from jax import core
from jax import linear_util as lu
from jax import tree_util
from jax.interpreters import ad
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.tree_util import (tree_flatten, tree_leaves, tree_map,
                           tree_structure, treedef_children,
                           treedef_tuple, tree_unflatten)
from jax._src import ad_util
from jax._src import api_util
from jax._src import custom_api_util
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


### bespoke linear_util and api_util deviations

class StoreEqual(lu.Store):
  """Stores an unchanging value. Checks empty reads and unequal overwrites."""
  def store(self, val):
    if self._val is not lu._EMPTY_STORE_VALUE and val != self._val:
      raise lu.StoreException(
          f"Store assignment mismatch, from {self._val} to {val}")
    self._val = val

@util.curry
def transformation_with_aux(
    gen, fun: lu.WrappedFun, *gen_static_args) -> Tuple[lu.WrappedFun, Any]:
  out_store = StoreEqual()
  out_thunk = lambda: out_store.val
  return fun.wrap(gen, gen_static_args, out_store), out_thunk

@transformation_with_aux
def flatten_fun_producing_avals_nokwargs(in_tree, *args_flat):
  py_args = tree_unflatten(in_tree, args_flat)
  ans = yield py_args, {}
  outs, out_tree = tree_flatten(ans)
  out_avals = map(core.get_aval, outs)
  yield outs, (out_tree, out_avals)


### api

@custom_api_util.register_custom_decorator_type
class custom_transpose:
  fun: Callable
  transpose: Optional[Callable] = None
  with_outs: bool

  def __init__(self, fun: Callable, with_outs: bool = False):
    functools.update_wrapper(self, fun)
    self.fun = fun  # type: ignore[assignment]
    self.with_outs = with_outs

  __getattr__ = custom_api_util.forward_attr

  def def_transpose(self, transpose: Callable):
    self.transpose = transpose
    return transpose

  @traceback_util.api_boundary
  def __call__(self, res_arg, lin_arg):
    res_args_flat, res_tree = tree_flatten(res_arg)
    lin_args_flat, lin_tree = tree_flatten(lin_arg)
    args_flat, in_tree = tree_flatten((res_arg, lin_arg))

    flat_fun, out_tree_and_avals = flatten_fun_producing_avals_nokwargs(
        lu.wrap_init(self.fun), in_tree)
    out_tree = lambda: out_tree_and_avals()[0]
    out_avals = lambda: out_tree_and_avals()[1]

    if self.with_outs:
      _, out_tree = treedef_children(res_tree)
      out_avals = None

    lin_avals = [core.raise_to_shaped(core.get_aval(x)) for x in lin_args_flat]
    flat_transpose = flatten_transpose(
        lu.wrap_init(self.transpose), res_tree, lin_tree, out_tree, lin_avals)
    out_flat = custom_transpose_p.bind(flat_fun, *args_flat,
                                       transpose=flat_transpose,
                                       lin_tree=lin_tree,
                                       res_tree=res_tree,
                                       out_tree=out_tree,
                                       out_avals=out_avals)
    if not self.with_outs:
      out_tree = out_tree()
    return tree_unflatten(out_tree, out_flat)


### utils

def tree_fill(x, treedef):
  return tree_unflatten(treedef, [x] * treedef.num_leaves)

def tree_fill_like(x, tree):
  return tree_fill(x, tree_structure(tree))

def tree_broadcast(full_treedef, tree, is_leaf=None):
  full_tree = tree_fill(0, full_treedef)
  return tree_map(tree_fill_like, tree, full_tree, is_leaf=is_leaf)

def is_treedef_prefix(entire, prefix):
  entire = tree_fill(0, entire)
  prefix = tree_fill(0, prefix)
  try:
    tree_map(lambda x, y: x, prefix, entire)
  except ValueError:
    return False
  return True

def rule_name(rule):
  return getattr(rule, '__name__', '<unnamed transpose rule>')

def check_transpose_rule_trees(rule, lin_tree, rule_out_tree):
  if not is_treedef_prefix(lin_tree, rule_out_tree):
    raise TypeError(
        'structure of custom transpose rule\'s output does not prefix-match '
        'structure of primal function\'s linear inputs under '
        f'custom transpose rule ({rule_name(rule)}).\n'
        f'Transpose rule output: {rule_out_tree}\n'
        f'Linear primal inputs: {lin_tree}')

def make_transpose_from_thunk(thunk, lin_tree):
  transpose_jaxpr, transpose_consts = thunk()
  transpose_jaxpr = core.ClosedJaxpr(
      pe.convert_constvars_jaxpr(transpose_jaxpr), ())
  return lu.wrap_init(core.jaxpr_as_fun(transpose_jaxpr))

def tree_mismatch_err_msg(ref_treedef, prefix_tree):
  # TODO(frostig,mattjj): error message should not be specific to VJPs
  prefix_treedef = tree_structure(prefix_tree)
  err_msg = (
      'Custom VJP rule must produce an output with the same container '
      '(pytree) structure as the args tuple of the primal function, '
      'and in particular must produce a tuple of length equal to the '
      'number of arguments to the primal function, but got VJP output '
      f'structure {prefix_treedef} for primal input structure {ref_treedef}.')
  return err_msg

def replace_nones(treedef, prefix_with_nones, sentinel):
  # TODO(frostig,mattjj): check if prefix_with_nones is even a tuple?
  num_leaves = lambda x: tree_structure(x).num_leaves
  out = []
  ref_fill_val = object()

  def replace_none(x, ref) -> None:
    if x is None and ref is None:
      pass  # matching tree structure Nones
    elif x is None and ref is not None:
      out.extend([sentinel] * num_leaves(ref)) # None to be replaced by sentinel
    elif x is not None and ref is not None:
      # x must be a non-None leaf, so ref better match it
      assert tree_util.treedef_is_leaf(tree_structure(x))
      if ref is not ref_fill_val:
        raise _InternalException(f'tree mismatch: {x} vs {ref}')
      out.append(x)
    elif x is not None and ref is None:
      raise _InternalException(
          f'value {x} where the primal function had a None input.')
    return x

  dummy_tree = tree_fill(ref_fill_val, treedef)
  try:
    tree_map(replace_none, prefix_with_nones, dummy_tree,
             is_leaf=lambda x: x is None)
  except _InternalException:
    raise TypeError(tree_mismatch_err_msg(treedef, prefix_with_nones)) from None
  except ValueError:
    raise TypeError(tree_mismatch_err_msg(treedef, prefix_with_nones)) from None
  return out

class _InternalException(ValueError): pass


def check_aval_and_make_zero(zero, x, aval):
  if x is zero:
    return ad_util.Zero(aval)
  if not core.typecheck(aval, x):
    raise TypeError(f'custom VJP/transpose rule returned {x}, '
                    f'incompatible with corresponding input type {aval}')
  return x

@lu.transformation
def flatten_transpose(res_tree, lin_tree, out_tree, lin_avals_flat, *args_flat):
  out_tree = out_tree() if callable(out_tree) else out_tree
  assert len(args_flat) == res_tree.num_leaves + out_tree.num_leaves
  res_args_flat, out_args_flat = util.split_list(
      args_flat, [res_tree.num_leaves])
  res_args = tree_unflatten(res_tree, res_args_flat)
  out_args = tree_unflatten(out_tree, out_args_flat)
  lin_outs = yield (res_args, out_args), {}
  zero = object()   # TODO(frostig, mattjj): revisit, maybe keep symbolic
  lin_outs_flat = replace_nones(lin_tree, lin_outs, zero)
  lin_outs_flat = [check_aval_and_make_zero(zero, x, aval)
                   for x, aval in zip(lin_outs_flat, lin_avals_flat)]
  yield lin_outs_flat


### custom_transpose primitive and rules

class CustomTransposePrimitive(core.Primitive):
  call_primitive = False
  map_primitive = False
  multiple_results = True

  def bind(self, call, *args, **params):
    # TODO(frostig,mattjj): This doesn't handle closures yet, which is
    # a bit involved. Closures are complicated by us binding `call`
    # twice in the JVP rule for custom transpose. The `env_trace_todo`
    # output by `process_env_traces` due to one of those two bindings
    # should be passable to the other, and need to be passed onward
    # since the second bind is deferred by partial eval (since it
    # typically receives unknowns)
    top_trace = core.find_top_trace(args)
    tracers = map(top_trace.full_raise, args)
    outs = top_trace.process_custom_transpose(self, call, tracers, **params)
    return outs

  # TODO(frostig,mattjj): consider keeping `call` as a named parameter
  # instead of following this "call primitive" convention.
  def get_bind_params(self, params):
    assert 'call_jaxpr' in params
    assert 'transpose_jaxpr_thunk' in params
    new_params = dict(params)
    new_params['transpose'] = make_transpose_from_thunk(
        new_params.pop('transpose_jaxpr_thunk'),
        new_params['lin_tree'])
    call = lu.wrap_init(core.jaxpr_as_fun(new_params.pop('call_jaxpr')))
    return [call], new_params


# TODO(frostig,mattjj): reinstate checks
def custom_transpose_typecheck(*avals, **params):
  return None, core.no_effects


def custom_transpose_transpose_rule(cts, *args, res_tree, lin_tree, **params):
  if 'transpose_jaxpr_thunk' in params:
    assert 'call_jaxpr' in params
    transpose = make_transpose_from_thunk(
        params['transpose_jaxpr_thunk'], lin_tree)
  else:
    assert 'call' in params
    transpose = params['transpose']

  # TODO(frostig,mattjj): `lin_arg` indicates the inputs with respect
  # to which we are transposing (via `ad.is_undefined_primal`).
  # Consider passing this information to the custom transpose rule?
  res_args, lin_args = util.split_list(args, [res_tree.num_leaves])
  del lin_args
  assert all(not ad.is_undefined_primal(x) for x in res_args)

  cts_out = [
      ad_util.zeros_like_aval(ct.aval) if type(ct) is ad_util.Zero else ct
      for ct in cts]
  cts_lin = transpose.call_wrapped(*res_args, *cts_out)
  return [None] * len(res_args) + cts_lin


def custom_transpose_lowering(*args, call_jaxpr, **params):
  return core.jaxpr_as_fun(call_jaxpr)(*args)


custom_transpose_p = CustomTransposePrimitive('custom_transpose_call')
core.custom_typechecks[custom_transpose_p] = custom_transpose_typecheck
ad.primitive_transposes[custom_transpose_p] = custom_transpose_transpose_rule
mlir.register_lowering(
    custom_transpose_p,
    mlir.lower_fun(custom_transpose_lowering, multiple_results=True))
xla.register_initial_style_primitive(custom_transpose_p)
