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
"""Module for discharging state primitives."""
from __future__ import annotations
import dataclasses
from functools import partial

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from typing_extensions import Protocol

import numpy as np

from jax import core
from jax import lax
from jax import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax._src.util import safe_map, safe_zip, split_list

from jax._src.state.types import ShapedArrayRef
from jax._src.state.primitives import get_p, swap_p, addupdate_p

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

## Discharging state

# Let's say we have a jaxpr that takes in `Ref`s and outputs regular JAX values
# (`Ref`s should never be outputs from jaxprs). We'd like to convert that jaxpr
# into a "pure" jaxpr that takes in and outputs values and no longer has the
# `Read/Write/Accum` effects.

def discharge_state(jaxpr: core.Jaxpr, consts: Sequence[Any], * ,
                    should_discharge: Union[bool, Sequence[bool]] = True
                    ) -> Tuple[core.Jaxpr, List[Any]]:
  """Converts a jaxpr that takes in `Ref`s into one that doesn't."""
  if isinstance(should_discharge, bool):
    should_discharge = [should_discharge] * len(jaxpr.invars)
  in_avals = [core.ShapedArray(v.aval.shape, v.aval.dtype)
              if type(v.aval) is ShapedArrayRef and d
              else v.aval for v, d in zip(jaxpr.invars, should_discharge)]
  eval_jaxpr = lu.wrap_init(partial(_eval_jaxpr_discharge_state, jaxpr,
                                    should_discharge, consts))
  new_jaxpr, _ , new_consts = pe.trace_to_jaxpr_dynamic(eval_jaxpr, in_avals)
  return new_jaxpr, new_consts

@dataclasses.dataclass
class Environment:
  env: Dict[core.Var, Any]

  def read(self, v: core.Atom) -> Any:
    if type(v) is core.Literal:
      return v.val
    assert isinstance(v, core.Var)
    return self.env[v]

  def write(self, v: core.Var, val: Any) -> None:
    self.env[v] = val

class DischargeRule(Protocol):

  def __call__(self, in_avals: Sequence[core.AbstractValue],
      out_avals: Sequence[core.AbstractValue], *args: Any,
      **params: Any) -> Tuple[Sequence[Optional[Any]], Sequence[Any]]:
    ...

_discharge_rules: dict[core.Primitive, DischargeRule] = {}

def register_discharge_rule(prim: core.Primitive):
  def register(f: DischargeRule):
    _discharge_rules[prim] = f
  return register

def _has_refs(eqn: core.JaxprEqn):
  return any(isinstance(v.aval, ShapedArrayRef) for v in eqn.invars)

def _eval_jaxpr_discharge_state(
    jaxpr: core.Jaxpr, should_discharge: Sequence[bool], consts: Sequence[Any],
    *args: Any):
  env = Environment({})

  map(env.write, jaxpr.constvars, consts)
  # Here some args may correspond to `Ref` avals but they'll be treated like
  # regular values in this interpreter.
  map(env.write, jaxpr.invars, args)

  refs_to_discharge = set(id(v.aval) for v, d
                          in zip(jaxpr.invars, should_discharge) if d
                          and isinstance(v.aval, ShapedArrayRef))

  for eqn in jaxpr.eqns:
    if _has_refs(eqn) and any(id(v.aval) in refs_to_discharge
                              for v in eqn.invars):
      if eqn.primitive not in _discharge_rules:
        raise NotImplementedError("No state discharge rule implemented for "
            f"primitive: {eqn.primitive}")
      invals = map(env.read, eqn.invars)
      in_avals = [v.aval for v in eqn.invars]
      out_avals = [v.aval for v in eqn.outvars]
      new_invals, ans = _discharge_rules[eqn.primitive](
          in_avals, out_avals, *invals, **eqn.params)
      for new_inval, invar in zip(new_invals, eqn.invars):
        if new_inval is not None:
          env.write(invar, new_inval)  # type: ignore[arg-type]
    else:
      # Default primitive rule, similar to `core.eval_jaxpr`. Note that here
      # we assume any higher-order primitives inside of the jaxpr are *not*
      # stateful.
      subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
      ans = eqn.primitive.bind(*subfuns, *map(env.read, eqn.invars),
                               **bind_params)
    if eqn.primitive.multiple_results:
      map(env.write, eqn.outvars, ans)
    else:
      env.write(eqn.outvars[0], ans)
  # By convention, we return the outputs of the jaxpr first and then the final
  # values of the `Ref`s. Callers to this function should be able to split
  # them up by looking at `len(jaxpr.outvars)`.
  out_vals = map(env.read, jaxpr.outvars)
  ref_vals = map(
      env.read, [v for v in jaxpr.invars if id(v.aval) in refs_to_discharge])
  return out_vals + ref_vals

@register_discharge_rule(get_p)
def _get_discharge_rule(
    in_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue], x, *non_slice_idx,
    indexed_dims: Sequence[bool]):
  del in_avals, out_avals
  y = _get_discharge(x, non_slice_idx, indexed_dims)
  return (None,) * (len(non_slice_idx) + 1), y

def _get_discharge(x, idx, indexed_dims):
  if not any(indexed_dims):
    return x
  if all(not i.shape for i in idx):
    return _dynamic_index(x, idx, indexed_dims)
  else:
    return _prepend_gather(x, idx, indexed_dims)

def _prepend_gather(x, idx, indexed_dims):
  indexer = _indexer(idx, indexed_dims)
  # NumPy advanced int indexing won't prepend w/ only one dim, so add dummy.
  return x[None][(np.array(0, 'int32'), *indexer)]

def _prepend_scatter(x, idx, indexed_dims, val, *, add=False):
  indexer = _indexer(idx, indexed_dims)
  if add:
    return x[None].at[(0, *indexer)].add(val)[0]
  return x[None].at[(0, *indexer)].set(val)[0]

def _indexer(idx, indexed_dims):
  idx_ = iter(idx)
  indexer = tuple([next(idx_) if b else slice(None) for b in indexed_dims])
  assert next(idx_, None) is None
  return indexer

@register_discharge_rule(swap_p)
def _swap_discharge_rule(
    in_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue], x, val, *non_slice_idx,
    indexed_dims: Sequence[bool]):
  del in_avals, out_avals
  if not any(indexed_dims):
    z, x_new = x, val
  z, x_new = _swap_discharge(x, val, non_slice_idx, indexed_dims)
  return (x_new, None) + (None,) * len(non_slice_idx), z

def _swap_discharge(x, val, idx, indexed_dims):
  if not any(indexed_dims):
    z, x_new = x, val
  elif all(not i.shape for i in idx):
    z = _dynamic_index(x, idx, indexed_dims)
    x_new = _dynamic_update_index(x, idx, val, indexed_dims)
  else:
    z = _prepend_gather(x, idx, indexed_dims)
    x_new = _prepend_scatter(x, idx, indexed_dims, val)
  return z, x_new

@register_discharge_rule(addupdate_p)
def _addupdate_discharge_rule(
    in_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue], x, val, *non_slice_idx,
    indexed_dims: Sequence[bool]):
  del in_avals, out_avals
  ans = _addupdate_discharge(x, val, non_slice_idx, indexed_dims)
  return (ans, None) + (None,) * len(non_slice_idx), []

def _addupdate_discharge(x, val, idx, indexed_dims):
  if not any(indexed_dims):
    return x + val
  if all(not i.shape for i in idx):
    y = val + _dynamic_index(x, idx, indexed_dims)
    return _dynamic_update_index(x, idx, y, indexed_dims)
  else:
    return _prepend_scatter(x, idx, indexed_dims, val, add=True)

def _dynamic_index(x, idx, indexed_dims):
  assert isinstance(idx, (list, tuple)) and idx
  idx_ = iter(idx)
  starts = [next(idx_) if b else np.int32(0) for b in indexed_dims]
  assert next(idx_, None) is None
  sizes = [1 if b else size for b, size in zip(indexed_dims, x.shape)]
  out = lax.dynamic_slice(x, starts, sizes)
  return lax.squeeze(out, [i for i, b in enumerate(indexed_dims) if b])

def _dynamic_update_index(x, idx, val, indexed_dims):
  assert isinstance(idx, (list, tuple)) and idx
  idx_ = iter(idx)
  starts = [next(idx_) if b else np.int32(0) for b in indexed_dims]
  assert next(idx_, None) is None
  sizes = [1 if b else size for b, size in zip(indexed_dims, x.shape)]
  return lax.dynamic_update_slice(x, val.reshape(sizes), starts)

@register_discharge_rule(core.closed_call_p)
def _closed_call_discharge_rule(
    in_avals: Sequence[core.AbstractValue], _,*args,
    call_jaxpr: core.ClosedJaxpr):
  jaxpr, consts = call_jaxpr.jaxpr, call_jaxpr.consts
  num_outs = len(jaxpr.outvars)
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, consts)
  discharged_closed_jaxpr = core.ClosedJaxpr(discharged_jaxpr,
                                             discharged_consts)
  fun = lu.wrap_init(core.jaxpr_as_fun(discharged_closed_jaxpr))
  out_and_ref_vals = core.closed_call_p.bind(fun, *args,
                                             call_jaxpr=discharged_closed_jaxpr)
  out_vals, ref_vals = split_list(out_and_ref_vals, [num_outs])
  ref_vals_iter = iter(ref_vals)
  new_invals = tuple(next(ref_vals_iter) if isinstance(aval, ShapedArrayRef)
                     else None for aval in in_avals)
  assert next(ref_vals_iter, None) is None
  return new_invals, out_vals
