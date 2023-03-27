# Copyright 2022 The JAX Authors.
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
"""Module for the common control flow utilities."""
import os
from functools import partial

from typing import Callable, Optional, Sequence

from jax._src import core
from jax._src import linear_util as lu
from jax._src.lax import lax
from jax._src.effects import control_flow_allowed_effects as allowed_effects
from jax._src import ad_util
from jax._src import util
from jax._src.util import cache, weakref_lru_cache, safe_map, unzip3
from jax.api_util import flatten_fun_nokwargs
from jax._src.interpreters import partial_eval as pe
from jax.tree_util import tree_map, tree_unflatten

map, unsafe_map = safe_map, map

allowed_effects.add_type(lax.InOutFeedEffect)


def _abstractify(x):
  return core.raise_to_shaped(core.get_aval(x))

def _typecheck_param(prim, param, name, msg_required, pred):
  if not pred:
    msg = (f'invalid {prim} param {name} of type {type(param).__name__}, '
           f'{msg_required} required:')
    param_str = str(param)
    sep = os.linesep if os.linesep in param_str else ' '
    msg = sep.join([msg, param_str])
    raise core.JaxprTypeError(msg)

@weakref_lru_cache
def _initial_style_open_jaxpr(fun: Callable, in_tree, in_avals,
                              primitive_name: Optional[str] = None):
  wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  debug = pe.debug_info(fun, in_tree, out_tree, False,
                        primitive_name or "<unknown>")
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals, debug)
  return jaxpr, consts, out_tree()

@weakref_lru_cache
def _initial_style_jaxpr(fun: Callable, in_tree, in_avals,
                         primitive_name: Optional[str] = None):
  jaxpr, consts, out_tree = _initial_style_open_jaxpr(
      fun, in_tree, in_avals, primitive_name)
  closed_jaxpr = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
  return closed_jaxpr, consts, out_tree

@cache()
def _initial_style_jaxprs_with_common_consts(
    funs: Sequence[Callable], in_tree, in_avals, primitive_name: str):
  # When staging the branches of a conditional into jaxprs, constants are
  # extracted from each branch and converted to jaxpr arguments. To use the
  # staged jaxprs as the branches to a conditional *primitive*, we need for
  # their (input) signatures to match. This function "joins" the staged jaxprs:
  # for each one, it makes another that accepts *all* constants, but only uses
  # those that it needs (dropping the rest).

  jaxprs, all_consts, all_out_trees = \
      unzip3(_initial_style_open_jaxpr(fun, in_tree, in_avals, primitive_name)
             for fun in funs)

  newvar = core.gensym(jaxprs, suffix='_')
  all_const_avals = [map(_abstractify, consts) for consts in all_consts]
  unused_const_vars = [map(newvar, const_avals)
                       for const_avals in all_const_avals]
  def pad_jaxpr_constvars(i, jaxpr):
    prefix = util.concatenate(unused_const_vars[:i])
    suffix = util.concatenate(unused_const_vars[i + 1:])
    constvars = [*prefix, *jaxpr.constvars, *suffix]
    return jaxpr.replace(constvars=constvars)

  consts = util.concatenate(all_consts)
  jaxprs = tuple(pad_jaxpr_constvars(i, jaxpr) for i, jaxpr in enumerate(jaxprs))
  closed_jaxprs = [core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
                   for jaxpr in jaxprs]
  return closed_jaxprs, consts, all_out_trees

def _check_tree_and_avals(what, tree1, avals1, tree2, avals2):
  """Raises TypeError if (tree1, avals1) does not match (tree2, avals2).

  Corresponding `tree` and `avals` must match in the sense that the number of
  leaves in `tree` must be equal to the length of `avals`. `what` will be
  prepended to details of the mismatch in TypeError.
  """
  if tree1 != tree2:
    raise TypeError(
        f"{what} must have same type structure, got {tree1} and {tree2}.")
  if not all(map(core.typematch, avals1, avals2)):
    diff = tree_map(_show_diff, tree_unflatten(tree1, avals1),
                    tree_unflatten(tree2, avals2))
    raise TypeError(f"{what} must have identical types, got\n{diff}.")

def _check_tree(func_name, expected_name, actual_tree, expected_tree, has_aux=False):
  if has_aux:
    actual_tree_children = actual_tree.children()

    if len(actual_tree_children) == 2:
      # select first child as result tree
      actual_tree = actual_tree_children[0]
    else:
      raise ValueError(
        f"{func_name}() produced a pytree with structure "
        f"{actual_tree}, but a pytree tuple with auxiliary "
        f"output was expected because has_aux was set to True.")

  if actual_tree != expected_tree:
    raise TypeError(
        f"{func_name}() output pytree structure must match {expected_name}, "
        f"got {actual_tree} and {expected_tree}.")

def _prune_zeros(ts):
  return [t for t in ts if type(t) is not ad_util.Zero]

def _make_closed_jaxpr(traceable: lu.WrappedFun, in_avals: Sequence[core.AbstractValue]):
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(traceable, in_avals)
  return core.ClosedJaxpr(jaxpr, consts)

def _show_diff(array1, array2):
  if core.typematch(array1, array2):
    return f"{array1}"
  return f"DIFFERENT {array1} vs. {array2}"

def _avals_short(avals):
  to_str = lambda aval: getattr(aval, 'str_short', partial(str, aval))()
  return ' '.join(map(to_str, avals))
