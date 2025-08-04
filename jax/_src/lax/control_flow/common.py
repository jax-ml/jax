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

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from functools import partial
from typing import Any

from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import config
from jax._src import linear_util as lu
from jax._src.util import weakref_lru_cache, safe_map
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (equality_errors_pytreedef, tree_map,
                                tree_unflatten, keystr, PyTreeDef)

map, unsafe_map = safe_map, map


def _typecheck_param(prim, param, name, msg_required, pred):
  if not pred:
    msg = (f'invalid {prim} param {name} of type {type(param).__name__}, '
           f'{msg_required} required:')
    param_str = str(param)
    # Avoid using os.linesep here to have the same multi-line error message
    # format on different platforms.
    sep = os.linesep if '\n' in param_str or '\r' in param_str else ' '
    msg = sep.join([msg, param_str])
    raise core.JaxprTypeError(msg)

@weakref_lru_cache
def _initial_style_open_jaxpr(fun: Callable,
                              in_tree: PyTreeDef,
                              in_avals: Sequence[core.AbstractValue],
                              debug_info: core.DebugInfo):
  wrapped_fun, out_tree = api_util.flatten_fun_nokwargs(
      lu.wrap_init(fun, debug_info=debug_info),
      in_tree)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals)
  return jaxpr, consts, out_tree()

@weakref_lru_cache
def _initial_style_jaxpr(fun: Callable,
                         in_tree: PyTreeDef,
                         in_avals: Sequence[core.AbstractValue],
                         debug_info: core.DebugInfo) -> tuple[core.ClosedJaxpr, Sequence[Any], PyTreeDef]:
  jaxpr, consts, out_tree = _initial_style_open_jaxpr(
      fun, in_tree, in_avals, debug_info)
  closed_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))
  return closed_jaxpr, consts, out_tree

def _initial_style_jaxprs_with_common_consts(
    funs: Sequence[Callable],
    in_tree: PyTreeDef, in_avals: Sequence[core.AbstractValue],
    debug_infos: Sequence[core.DebugInfo]):
  jaxpr_data = [_initial_style_open_jaxpr(fn, in_tree, in_avals, debug_info)
                for fn, debug_info in zip(funs, debug_infos)]
  if not jaxpr_data: return [], [], []
  jaxprs, all_consts, all_out_trees = zip(*jaxpr_data)

  # Jaxprs must share consts, so we concat consts and pad the jaxprs' constvars.
  lens = map(len, all_consts)
  consts = [c for cs in all_consts for c in cs]
  avals = tuple(map(core.typeof, consts))
  jaxprs = [_pad_constvars(jaxpr, avals[:sum(lens[:i])], avals[sum(lens[:i+1]):])
            for i, jaxpr in enumerate(jaxprs)]
  # De-duplicate shared constants.
  const_ids = tuple(id(c) for c in consts)
  seen = set()
  consts = [c for c in consts if id(c) not in seen and not seen.add(id(c))]  # type: ignore
  jaxprs = [_dedup_consts(jaxpr, const_ids) for jaxpr in jaxprs]

  closed_jaxprs = [pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))
                   for jaxpr in jaxprs]
  return closed_jaxprs, consts, all_out_trees

@weakref_lru_cache
def _pad_constvars(jaxpr: core.Jaxpr, left: tuple[core.AbstractValue, ...],
                   right: tuple[core.AbstractValue, ...]) -> core.Jaxpr:
  constvars = [*map(core.Var, left), *jaxpr.constvars, *map(core.Var, right)]
  effs = pe._renumber_effects([*constvars, *jaxpr.invars],
                              [*jaxpr.constvars, *jaxpr.invars], jaxpr.effects)
  jaxpr = jaxpr.replace(constvars=constvars, effects=effs)
  config.enable_checks.value and core.check_jaxpr(jaxpr)
  return jaxpr

@weakref_lru_cache
def _dedup_consts(jaxpr, const_ids):
  newvars = {}
  canonicalize = {v: newvars.setdefault(constid, v)
                  for constid, v in zip(const_ids, jaxpr.constvars)}
  eqns = [e.replace(invars=[canonicalize.get(x, x) if isinstance(x, core.Var)
                            else x for x in e.invars]) for e in jaxpr.eqns]
  outvars = [canonicalize.get(x, x) if isinstance(x, core.Var) else x
             for x in jaxpr.outvars]
  constvars = list(newvars.values())
  effs = pe._renumber_effects(
      [*constvars, *jaxpr.invars],
      [*map(canonicalize.get, jaxpr.constvars), *jaxpr.invars], jaxpr.effects)
  jaxpr = jaxpr.replace(constvars=constvars, eqns=eqns, outvars=outvars,
                        effects=effs)
  config.enable_checks.value and core.check_jaxpr(jaxpr)
  return jaxpr

def _check_tree_and_avals(what1, tree1, avals1, what2, tree2, avals2):
  """Raises TypeError if (tree1, avals1) does not match (tree2, avals2).

  Corresponding `tree` and `avals` must match in the sense that the number of
  leaves in `tree` must be equal to the length of `avals`. `what1` and
  `what2` describe what the `tree1` and `tree2` represent.
  """
  if tree1 != tree2:
    errs = list(equality_errors_pytreedef(tree1, tree2))
    msg = []
    msg.append(
        f"{what1} must have same type structure as {what2}, but there are differences: ")
    for path, thing1, thing2, explanation in errs:
      msg.append(
          f"    * at output{keystr(tuple(path))}, {what1} has {thing1} and "
          f"{what2} has {thing2}, so {explanation}")
    raise TypeError('\n'.join(msg))

  if not all(map(core.typematch, avals1, avals2)):
    diff = tree_map(_show_diff, tree_unflatten(tree1, avals1),
                    tree_unflatten(tree2, avals2))
    raise TypeError(f"{what1} and {what2} must have identical types, got\n{diff}.")

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

def _make_closed_jaxpr(traceable: lu.WrappedFun,
                       in_avals: Sequence[core.AbstractValue]):
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(traceable, in_avals)
  return core.ClosedJaxpr(jaxpr, consts)

def _show_diff(array1, array2):
  if core.typematch(array1, array2):
    return f"{array1}"
  return f"DIFFERENT {array1} vs. {array2}"

def _avals_short(avals):
  to_str = lambda aval: getattr(aval, 'str_short', partial(str, aval))()
  return ' '.join(map(to_str, avals))
