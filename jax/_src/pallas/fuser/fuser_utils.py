# Copyright 2025 The JAX Authors.
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

"""Basic utils for fuser internals."""

import functools
import itertools
from typing import Any

from jax._src import api_util
from jax._src import core
from jax._src import flattree as ft
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.state import discharge as state_discharge
from jax._src.state import types as state_types
import numpy as np


def make_jaxpr(f, *args, **kwargs):
  args_avals = tree_util.tree_map(core.shaped_abstractify, args)
  kwargs_avals = tree_util.tree_map(core.shaped_abstractify, kwargs)
  in_avals_ft = ft.flatten((args_avals, kwargs_avals))
  debug_info = api_util.debug_info('make_jaxpr', f, args, kwargs)
  closed_jaxpr, out_avals_ft = pe.trace_to_jaxpr_nocache(
      f, in_avals_ft, debug_info
  )
  return closed_jaxpr, closed_jaxpr.consts, in_avals_ft.tree, out_avals_ft.tree


def compare_jaxprs(jaxpr1: core.Jaxpr, jaxpr2: core.Jaxpr) -> bool:
  """Compares two jaxprs for symbolic equivalence up to renaming and DCE.

  Walks both jaxprs back from their outputs in lockstep, making the comparison
  insensitive to variable names, topologically equivalent linearizations of
  pure equations, and dead code. Effectful equations must match 1-to-1 in
  program order.

  The implementation is conservative: it may return False for equivalent jaxprs,
  but never True for different ones.
  """
  if jaxpr1 is jaxpr2:
    return True
  if jaxpr1.in_avals != jaxpr2.in_avals or jaxpr1.out_avals != jaxpr2.out_avals:
    return False
  effectful1 = [eqn for eqn in jaxpr1.eqns if eqn.effects]
  effectful2 = [eqn for eqn in jaxpr2.eqns if eqn.effects]
  if len(effectful1) != len(effectful2):
    return False
  effectful_pairs = set(zip(effectful1, effectful2))
  origins1 = _origins(jaxpr1)
  origins2 = _origins(jaxpr2)

  def same_atom(a: core.Atom, b: core.Atom) -> bool:
    if a.aval != b.aval:
      return False
    match a, b:
      case core.Literal(), core.Literal():
        return _same_value(a.val, b.val)
      case core.Var(), core.Var():
        match origins1[a], origins2[b]:
          case ('invar', i), ('invar', j):
            return i == j
          case ('const', c1), ('const', c2):
            return _same_value(c1, c2)
          case ('eqn', eqn1, i), ('eqn', eqn2, j):
            return i == j and same_eqn(eqn1, eqn2)
    return False

  @functools.cache
  def same_eqn(eqn1: core.JaxprEqn, eqn2: core.JaxprEqn) -> bool:
    # Effectful equations cannot be reordered, so they must correspond in
    # program order.
    if (eqn1.effects or eqn2.effects) and (eqn1, eqn2) not in effectful_pairs:
      return False
    return (
        eqn1.primitive is eqn2.primitive
        and len(eqn1.invars) == len(eqn2.invars)
        and _same_params(eqn1.params, eqn2.params)
        and all(map(same_atom, eqn1.invars, eqn2.invars))
    )

  # Effectful equations without outvars are not reachable from outvars, so walk
  # effectful equations explicitly as well.
  return (
      all(map(same_atom, jaxpr1.outvars, jaxpr2.outvars))
      and all(map(same_eqn, effectful1, effectful2))
  )


def _origins(jaxpr: core.Jaxpr) -> dict[core.Var, tuple[Any, ...]]:
  """Maps each variable in ``jaxpr`` to where its value comes from."""
  origins: dict[core.Var, tuple[Any, ...]] = {}
  for i, v in enumerate(jaxpr.invars):
    origins[v] = ('invar', i)
  for v, c in zip(jaxpr.constvars, jaxpr.consts, strict=True):
    origins[v] = ('const', c)
  for eqn in jaxpr.eqns:
    for i, v in enumerate(eqn.outvars):
      origins[v] = ('eqn', eqn, i)
  return origins


def _same_params(p1, p2) -> bool:
  leaves1, tree1 = tree_util.tree_flatten(p1)
  leaves2, tree2 = tree_util.tree_flatten(p2)
  return tree1 == tree2 and all(map(_same_value, leaves1, leaves2))


def _same_value(x, y) -> bool:
  if type(x) is not type(y):
    return False
  if isinstance(x, core.Jaxpr):
    return compare_jaxprs(x, y)
  if isinstance(x, np.ndarray):
    return (x.dtype, x.shape, x.tobytes()) == (y.dtype, y.shape, y.tobytes())
  try:
    return hash(x) == hash(y) and x == y
  except TypeError:  # Unhashable, e.g. a closed-over jax.Array.
    return x is y


def get_write_indices(jaxpr):
  effects = set()
  all_vars = {v: i for i, v in enumerate(jaxpr.constvars + jaxpr.invars)}
  for eqn in jaxpr.eqns:
    for e in eqn.effects:
      if isinstance(e, (state_types.WriteEffect, state_types.AccumEffect)):
        if (idx := all_vars.get(e.input, None)) is not None:
          effects.add(idx)
  return effects


def discharge_state(
    jaxpr : core.Jaxpr,
    *,
    allow_additional_outputs: bool = True,
    dce: bool = False
) -> tuple[core.Jaxpr, list[bool], dict[int, int]]:
  """Converts a stateful fusion jaxpr into a pure one.

  Discharging replace ``Ref`` inputs with regular values and threads updates
  through the computation.

  Args:
    jaxpr: The fusion jaxpr to discharge.
    allow_additional_outputs: If True, the returned jaxpr will have an
      output for each modified Ref, containing the final, updated value for
      that Ref.
    dce: If True, perform dead code elimination on the discharged jaxpr,
      assuming all outputs are used.

  Returns:
    A tuple of ``(discharged_jaxpr, used_consts, output_input_aliases)`` where
    ``discharged_jaxpr`` is the discharged jaxpr, ``used_consts`` is a boolean
    list indicating which consts of the pre-DCE discharged jaxpr are
    used/included in the returned ``discharged_jaxpr`` (or all ``True`` if DCE
    was not requested), and ``output_input_aliases`` is a dict
    mapping ``outvars`` indices to ``constvars + inputvars`` indices in
    ``discharged_jaxpr` -- an entry ``(o, i)`` indicates that additional output
    ``o`` is the updated value for const/input ``i`` (which was a Ref in the
    original jaxpr).
  """
  should_discharge = [isinstance(v.aval, state_types.AbstractRef)
                      for v in itertools.chain(jaxpr.constvars, jaxpr.invars)]
  if not any(should_discharge):
    return jaxpr, [True] * len(jaxpr.constvars), {}

  jaxpr_no_consts = pe.convert_constvars_jaxpr(jaxpr)
  closed_discharged_jaxpr = state_discharge.discharge_state(
      jaxpr_no_consts,
      should_discharge=should_discharge,
      lower=False,
      strip_memory_space=True,
  )
  assert not closed_discharged_jaxpr.consts, (
      closed_discharged_jaxpr, closed_discharged_jaxpr.consts)
  discharged_jaxpr = closed_discharged_jaxpr

  # ref_input_idxs[i] is the index, for the i-th new output, of the input Ref
  # that it corresponds to.
  ref_input_idxs = [i for i, v in enumerate(jaxpr_no_consts.invars)
              if isinstance(v.aval, state_types.AbstractRef)]
  num_new_outvars = len(discharged_jaxpr.outvars) - len(jaxpr.outvars)
  assert len(ref_input_idxs) == num_new_outvars, (
      len(ref_input_idxs), len(discharged_jaxpr.outvars), len(jaxpr.outvars),
  )

  # discharged_jaxpr has N new outputs, where N is the number of Ref inputs in
  # jaxpr_no_consts.  If allow_additional_outputs is True, we only want to keep
  # a new output if the original jaxpr actually writes to it.  If
  # allow_additional_outputs is False, we drop all of these new outputs.
  # (Callers will use this to discharge Ref updates to be outputs only for
  # output fusions.)
  write_idxs = get_write_indices(jaxpr) if allow_additional_outputs else set()
  keep_outvar = (
      [True] * len(jaxpr.outvars) + [i in write_idxs for i in ref_input_idxs])
  instantiate = ([i in write_idxs for i in range(len(jaxpr.constvars))]
                 + [True] * len(jaxpr.invars))
  if dce:
    discharged_jaxpr, used_inputs = pe.dce_jaxpr(
        discharged_jaxpr, used_outputs=keep_outvar, instantiate=instantiate)
    assert all(used_inputs[i] for i in write_idxs)
    used_consts = used_inputs[:len(jaxpr.constvars)]
  else:
    discharged_jaxpr = discharged_jaxpr.replace(
        outvars=[v for keep, v in zip(keep_outvar, discharged_jaxpr.outvars)
                 if keep])
    used_consts = [True] * len(jaxpr.constvars)
    used_inputs = used_consts + [True] * len(jaxpr.invars)
  discharged_jaxpr = discharged_jaxpr.with_consts(
      [c for c, used in zip(jaxpr.consts, used_consts) if used])

  # adjust indices given used_inputs, so we can compute output_input_aliases
  new_input_idx = list(itertools.accumulate(used_inputs, initial=-1))[1:]
  write_idxs  = {new_input_idx[i] for i in write_idxs}
  ref_input_idxs = [new_input_idx[i] for i in ref_input_idxs if used_inputs[i]]
  written_ref_input_idxs = [i for i in ref_input_idxs if i in write_idxs]
  output_input_aliases = {(i + len(jaxpr.outvars)): j
                           for i, j in enumerate(written_ref_input_idxs)}
  assert len(output_input_aliases) == len(write_idxs), (
      ref_input_idxs, write_idxs, output_input_aliases
  )
  return discharged_jaxpr, used_consts, output_input_aliases


@util.weakref_lru_cache
def jaxpr_all_prims(jaxpr: core.Jaxpr) -> set[core.Primitive]:
  prims = {eqn.primitive for eqn in jaxpr.eqns}
  for subjaxpr in core.subjaxprs(jaxpr):
    prims |= jaxpr_all_prims(subjaxpr)
  return prims


def filter_no_block_specs(values, block_specs):
  """Filters out values and block specs that have no_block_spec by replacing them with None."""
  filtered_values = tree_util.tree_map(
      lambda v, bs: v if bs is not pallas_core.no_block_spec else None,
      values,
      block_specs,
  )
  filtered_specs = tree_util.tree_map(
      lambda bs: bs if bs is not pallas_core.no_block_spec else None,
      block_specs,
  )
  return filtered_values, filtered_specs


def block_spec_with_prefetch(block_specs, scalar_prefetch_smem_refs):
  """Appends scalar prefetch refs to index maps of block specs in a PyTree."""
  def _wrap_one(bs):
    if bs is pallas_core.no_block_spec or bs is None or bs.index_map is None:
      return bs
    return bs.replace(
        index_map=lambda *args: bs.index_map(*args, *scalar_prefetch_smem_refs)
    )
  return tree_util.tree_map(_wrap_one, block_specs)
