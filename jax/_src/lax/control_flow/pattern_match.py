# Copyright 2021 The JAX Authors.
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
from typing import Dict, List, Optional, Set, Tuple, Union

import jax
from jax._src.lax.slicing import scatter_to_gather_dnums, dynamic_update_slice_p, scatter_p


def pattern_match_inplace_select(jaxpr: jax.core.ClosedJaxpr) -> jax.core.ClosedJaxpr:
  """Pattern matches an inplace-select into a select-inplace"""
  changed = True
  new_jaxpr = jaxpr.jaxpr
  while changed:
    new_jaxpr, changed = _pattern_match_inplace_select(new_jaxpr)
  if jax.config.jax_enable_checks:
    jax.core.check_jaxpr(new_jaxpr)
  return jaxpr.replace(jaxpr=new_jaxpr)


@dataclass
class _RewrittenEqns:
  old_eqn: jax.core.JaxprEqn
  new_eqns: Optional[List[jax.core.JaxprEqn]]  # mutable field


@dataclass
class _DeferredEqn:
  old_eqn: jax.core.JaxprEqn
  rewritten_eqns: Optional[_RewrittenEqns]  # mutable field

  def __hash__(self):
    return hash(id(self))

  def __eq__(self, other):
    return self is other


def _default_no_rewrite(eqn, new_eqns, deferred_eqns):
  if eqn.primitive in inplace_select_rules.keys():
    # Deferred processing scatters or dynamic_update_slices until later; they might
    # need reordering downstream.
    deferred_eqn = _DeferredEqn(old_eqn=eqn, rewritten_eqns=None)
    for outvar in eqn.outvars:
      if type(outvar) is jax.core.Var:
        deferred_eqns[outvar] = deferred_eqn
    new_eqns.append(deferred_eqn)
  else:
    # Default behaviour: save the equation normally.
    new_eqns.append(eqn)


def _pattern_match_inplace_select(jaxpr: jax.core.Jaxpr) -> Tuple[jax.core.Jaxpr, bool]:
  possible_eqns: List[Union[jax.core.JaxprEqn, _RewrittenEqns, _DeferredEqn]] = []
  deferred_eqns: Dict[jax.core.Var, _DeferredEqn] = {}
  gensym = jax.core.gensym([jaxpr])

  for eqn in jaxpr.eqns:
    # Check to see if `eqn` is a direct dependent of any equations that admit rewrite
    # rules. (By default, scatters and dynamic_update_slices.)
    # If `eqn` is the only direct dependent, then we may be able to rewrite it.
    # (By default, if `eqn` is a select.)
    # The original upstream equation is said to be "deferred", in case of a rewrite
    # later.
    upstream_deferred_eqns: Set[_DeferredEqn] = set()
    for invar in eqn.invars:
      if type(invar) is jax.core.Var:
        try:
          deferred_eqn = deferred_eqns[invar]
        except KeyError:
          pass
        else:
          upstream_deferred_eqns.add(deferred_eqn)
    # If any of those deferred equations have already been rewritten,
    # then undo the rewrite: `eqn` is now a second downstream dependent of those
    # operations, and so we need the intermediate value that the rewrite would remove.
    for deferred_eqn in upstream_deferred_eqns:
      if deferred_eqn.rewritten_eqns is not None:
        deferred_eqn.rewritten_eqns.new_eqns = None
    # Try to rewrite.
    if len(upstream_deferred_eqns) == 1:
      # Downstream of a single deferred equation; let's try to rewrite it.
      [deferred_eqn] = upstream_deferred_eqns
      if deferred_eqn.rewritten_eqns is None:
        # And this deferred equation hasn't already been rewritten.
        rewrite_rule = inplace_select_rules[deferred_eqn.old_eqn.primitive]
        rewritten_eqns = rewrite_rule(deferred_eqn.old_eqn, eqn, gensym)
        if rewritten_eqns is None:
          # We couldn't rewrite, so fall back to default behaviour.
          _default_no_rewrite(eqn, possible_eqns, deferred_eqns)
        else:
          rewritten_eqns = _RewrittenEqns(old_eqn=eqn, new_eqns=rewritten_eqns)
          deferred_eqn.rewritten_eqns = rewritten_eqns
          possible_eqns.append(rewritten_eqns)
      else:
        # This deferred equation has already been rewritten. No rewrite possible, so
        # default processing.
        _default_no_rewrite(eqn, possible_eqns, deferred_eqns)
    else:
      # Not downstream, or downstream of multiple. So default processing.
      _default_no_rewrite(eqn, possible_eqns, deferred_eqns)

  changed = False
  out_eqns: List[jax.core.JaxprEqn] = []
  for possible_eqn in possible_eqns:
    if type(possible_eqn) is jax.core.JaxprEqn:
      out_eqns.append(possible_eqn)
    elif type(possible_eqn) is _DeferredEqn:
      if possible_eqn.rewritten_eqns is None:  # never got rewritten
        out_eqns.append(possible_eqn.old_eqn)
      elif possible_eqn.rewritten_eqns.new_eqns is None:  # rewrite was undone
        out_eqns.append(possible_eqn.old_eqn)
      # else don't do anything: we'll use the rewritten equation instead.
    elif type(possible_eqn) is _RewrittenEqns:
      if possible_eqn.new_eqns is None:  # rewrite was undone; use original equation
        out_eqns.append(possible_eqn.old_eqn)
      else:  # rewrite success!
        changed = True
        out_eqns.extend(possible_eqn.new_eqns)
    else:
      assert False
  return jaxpr.replace(eqns=out_eqns), changed


def _scatter_to_gather_params(eqn):
  params = eqn.params
  operand, _, updates = eqn.invars
  dimension_numbers, slice_sizes = scatter_to_gather_dnums(params['dimension_numbers'],
                                                           operand.aval.ndim,
                                                           updates.aval.shape)
  new_params = dict(
    dimension_numbers=dimension_numbers,
    slice_sizes=slice_sizes,
    unique_indices=params['unique_indices'],
    indices_are_sorted=params['indices_are_sorted'],
    mode=params['mode'],
    fill_value=0,
  )
  return new_params


def _make_new_select_eqn(select_eqn, xs, update_x, ys, gensym):
  pred, a, b = select_eqn.invars
  [updated_xs] = select_eqn.outvars

  shaped_pred = pred.aval.shape == xs.aval.shape
  if not shaped_pred:
    assert pred.aval.shape == ()

  if shaped_pred:
    new_pred = gensym(jax.core.ShapedArray(update_x.aval.shape, pred.aval.dtype))
  else:
    new_pred = pred
  new_x = gensym(update_x.aval)
  new_update_x = gensym(update_x.aval)

  if ys is a:
    if xs is not b:
      return
    select_invars = [new_pred, update_x, new_x]
  elif ys is b:
    if xs is not a:
      return
    select_invars = [new_pred, new_x, update_x]
  else:
    assert ys is pred
    return

  new_select_eqn = jax.core.new_jaxpr_eqn(
    invars=select_invars, outvars=[new_update_x],
    primitive=jax.lax.select_n_p, params={}, effects=jax.core.no_effects)

  return new_select_eqn, pred, updated_xs, new_pred, new_x, new_update_x, shaped_pred


def _scatter_rule(scatter_eqn, select_eqn, gensym):
  if select_eqn.primitive is not jax.lax.select_n_p:
    return
  if len(select_eqn.invars) != 3:
    return
  assert scatter_eqn.primitive is jax.lax.scatter_p
  assert scatter_eqn.effects == jax.core.no_effects
  assert select_eqn.effects == jax.core.no_effects
  assert select_eqn.params == {}

  xs, i, update_x = scatter_eqn.invars
  [ys] = scatter_eqn.outvars
  info = _make_new_select_eqn(select_eqn, xs, update_x, ys, gensym)
  if info is None:
    return
  new_select_eqn, pred, updated_xs, new_pred, new_x, new_update_x, shaped_pred = info
  gather_params = _scatter_to_gather_params(scatter_eqn)

  out = []
  if shaped_pred:
    new_pred_eqn = jax.core.new_jaxpr_eqn(
      invars=[pred, i], outvars=[new_pred],
      primitive=jax.lax.gather_p, params=gather_params, effects=jax.core.no_effects
    )
    out.append(new_pred_eqn)
  new_index_eqn = jax.core.new_jaxpr_eqn(
    invars=[xs, i], outvars=[new_x],
    primitive=jax.lax.gather_p, params=gather_params, effects=jax.core.no_effects)
  new_scatter_eqn = jax.core.new_jaxpr_eqn(
    invars=[xs, i, new_update_x], outvars=[updated_xs],
    primitive=jax.lax.scatter_p, params=scatter_eqn.params, effects=jax.core.no_effects)
  out.append(new_index_eqn)
  out.append(new_select_eqn)
  out.append(new_scatter_eqn)

  return out


def _dynamic_update_slice_rule(slice_eqn, select_eqn, gensym):
  if select_eqn.primitive is not jax.lax.select_n_p:
    return
  if len(select_eqn.invars) != 3:
    return
  assert slice_eqn.primitive is jax.lax.dynamic_update_slice_p
  assert slice_eqn.effects == jax.core.no_effects
  assert slice_eqn.params == {}
  assert select_eqn.effects == jax.core.no_effects
  assert select_eqn.params == {}

  xs, update_x, *indices = slice_eqn.invars
  [ys] = slice_eqn.outvars
  new_select_eqn, pred, updated_xs, new_pred, new_x, new_update_x, shaped_pred  = _make_new_select_eqn(select_eqn, xs, update_x, ys, gensym)
  if new_select_eqn is None:
    return
  slice_params = dict(slice_sizes=update_x.aval.shape)

  out = []
  if shaped_pred:
    new_pred_eqn = jax.core.new_jaxpr_eqn(
      invars=[pred, *indices], outvars=[new_pred],
      primitive=jax.lax.dynamic_slice_p, params=slice_params, effects=jax.core.no_effects
    )
    out.append(new_pred_eqn)
  new_index_eqn = jax.core.new_jaxpr_eqn(
    invars=[xs, *indices], outvars=[new_x],
    primitive=jax.lax.dynamic_slice_p, params=slice_params, effects=jax.core.no_effects)
  new_update_eqn = jax.core.new_jaxpr_eqn(
    invars=[xs, new_update_x, *indices], outvars=[updated_xs],
    primitive=jax.lax.dynamic_update_slice_p, params={}, effects=jax.core.no_effects)
  out.append(new_index_eqn)
  out.append(new_select_eqn)
  out.append(new_update_eqn)

  return out


inplace_select_rules = {}
inplace_select_rules[scatter_p] = _scatter_rule
inplace_select_rules[dynamic_update_slice_p] = _dynamic_update_slice_rule
