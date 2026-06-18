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

import hashlib
import itertools

from jax._src import api_util
from jax._src import core
from jax._src import effects as effects_lib
from jax._src import tree_util
from jax._src.interpreters import partial_eval as pe
from jax._src.state import discharge as state_discharge
from jax._src.state import types as state_types

import numpy as np

def make_jaxpr(f, *args, **kwargs):
  args_avals = tree_util.tree_map(core.shaped_abstractify, args)
  kwargs_avals = tree_util.tree_map(core.shaped_abstractify, kwargs)
  in_avals_ft = tree_util.FlatTree.flatten((args_avals, kwargs_avals))
  debug_info = api_util.debug_info('make_jaxpr', f, args, kwargs)
  closed_jaxpr, out_avals_ft = pe.trace_to_jaxpr(
      f, in_avals_ft, debug_info
  )
  return closed_jaxpr.jaxpr, closed_jaxpr.consts, in_avals_ft.tree, out_avals_ft.tree


# symbolic jaxpr comparison (for index map comparisons)

# Use a value numbering approach to hash the jaxpr, this makes comparison
# insensitive to variable name choices and topologically equivalent
# linearizations.


def _make_hashable(val):
  if isinstance(val, (core.Jaxpr, core.ClosedJaxpr)):
    return _jaxpr_signature(val)
  elif isinstance(val, (list, tuple)):
    return tuple(_make_hashable(v) for v in val)
  elif isinstance(val, dict):
    return tuple(sorted((k, _make_hashable(v)) for k, v in val.items()))
  elif isinstance(val, (set, frozenset)):
    return frozenset(_make_hashable(v) for v in val)
  elif hasattr(val, 'shape') and hasattr(val, 'dtype'):
    try:
      b = (
          val.tobytes()
          if hasattr(val, 'tobytes')
          else np.asarray(val).tobytes()
      )
      arr_hash = hashlib.sha256(b).hexdigest()
      return ('array', tuple(val.shape), str(val.dtype), arr_hash)
    except Exception:
      return ('array_fallback', tuple(val.shape), str(val.dtype))
  else:
    try:
      hash(val)
      return type(val), val
    except TypeError:
      return type(val), str(val)


def _jaxpr_signature(jaxpr_obj):
  env = {}
  if isinstance(jaxpr_obj, core.ClosedJaxpr):
    jaxpr = jaxpr_obj.jaxpr
    for v, c in zip(jaxpr.constvars, jaxpr_obj.consts):
      env[v] = ('constval', _make_hashable(c))
  else:
    jaxpr = jaxpr_obj
    for i, v in enumerate(jaxpr.constvars):
      env[v] = ('constvar_idx', i)
  for i, v in enumerate(jaxpr.invars):
    env[v] = ('invar', i)

  def get_var_sig(v):
    if isinstance(v, core.Literal):
      return ('literal', _make_hashable(v.val))
    elif type(v).__name__ == 'DropVar':
      return ('dropvar',)
    elif v in env:
      return env[v]
    else:
      return ('unknown_var', str(v))

  def get_effect_sig(e):
    if isinstance(e, effects_lib.JaxprInputEffect) and isinstance(
        e.input, core.Var):
      return f'{type(e).__name__}<{get_var_sig(e.input)}>'
    return str(e)

  eqn_sigs = []
  for eqn in jaxpr.eqns:
    in_sigs = tuple(hash(get_var_sig(v)) for v in eqn.invars)
    params_sig = hash(_make_hashable(eqn.params))
    effects = tuple(sorted(get_effect_sig(e) for e in getattr(eqn, 'effects', [])))
    op_sig = ('eqn', eqn.primitive.name, in_sigs, params_sig, effects)
    eqn_sigs.append(op_sig)
    for i, outvar in enumerate(eqn.outvars):
      if type(outvar).__name__ != 'DropVar':
        env[outvar] = ('out', op_sig, i)  # pyrefly: ignore[unsupported-operation]
  out_sigs = tuple(get_var_sig(v) for v in jaxpr.outvars)
  jaxpr_effects = tuple(
      sorted(get_effect_sig(e) for e in getattr(jaxpr, 'effects', [])))
  eqn_sigs_sorted = tuple(sorted(eqn_sigs, key=lambda x: str(x)))
  return ('jaxpr_dag', out_sigs, eqn_sigs_sorted, jaxpr_effects)


def compare_jaxprs(jaxpr1, jaxpr2):
  """Compares two JAXPRs for symbolic equivalence."""
  sig1 = _jaxpr_signature(jaxpr1)
  sig2 = _jaxpr_signature(jaxpr2)
  return sig1 == sig2


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
      core.ClosedJaxpr(jaxpr_no_consts, ()),
      should_discharge=should_discharge,
      lower=False,
  )
  assert not closed_discharged_jaxpr.consts, (
      closed_discharged_jaxpr.jaxpr, closed_discharged_jaxpr.consts)
  discharged_jaxpr = closed_discharged_jaxpr.jaxpr

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
  discharged_jaxpr = pe.convert_invars_to_constvars(
      discharged_jaxpr, sum(used_consts))

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
