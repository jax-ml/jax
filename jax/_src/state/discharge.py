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
"""Module for discharging state primitives."""
from functools import partial
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from jax import core
from jax import lax
from jax import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax._src.util import safe_map, safe_zip

from jax._src.state.types import ShapedArrayRef
from jax._src.state.primitives import get_p, swap_p, addupdate_p

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

## Discharging state

# Let's say we have a jaxpr that takes in `Ref`s and outputs regular JAX values
# (`Ref`s should never be outputs from jaxprs). We'd like to convert that jaxpr
# into a "pure" jaxpr that takes in and outputs values and no longer has the
# `StateEffect` effect.

def discharge_state(jaxpr: core.Jaxpr, consts: Sequence[Any]
                    ) -> Tuple[core.Jaxpr, List[Any]]:
  """Converts a jaxpr that takes in `Ref`s into one that doesn't."""
  in_avals = [core.ShapedArray(v.aval.shape, v.aval.dtype)
              if type(v.aval) is ShapedArrayRef
              else v.aval for v in jaxpr.invars]
  eval_jaxpr = lu.wrap_init(partial(_eval_jaxpr_discharge_state, jaxpr, consts))
  new_jaxpr, _ , new_consts = pe.trace_to_jaxpr_dynamic(eval_jaxpr, in_avals)
  return new_jaxpr, new_consts

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

def _eval_jaxpr_discharge_state(jaxpr: core.Jaxpr, consts: Sequence[Any],
                                *args: Any):
  env: Dict[core.Var, Any] = {}

  def read(v: core.Atom) -> Any:
    if type(v) is core.Literal:
      return v.val
    assert isinstance(v, core.Var)
    return env[v]

  def write(v: core.Var, val: Any) -> None:
    env[v] = val

  map(write, jaxpr.constvars, consts)
  # Here some args may correspond to `Ref` avals but they'll be treated like
  # regular values in this interpreter.
  map(write, jaxpr.invars, args)

  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    if eqn.primitive is get_p:
       # `y <- x[i]` becomes `y = ds x i`
      x, *idx = in_vals
      y = _get_discharge(x, idx, eqn.params['indexed_dims'])
      write(eqn.outvars[0], y)
    elif eqn.primitive is swap_p:
      # `z, x[i] <- x[i], val` becomes:
      #    z = ds x i
      #    x = dus x i val
      assert isinstance(eqn.invars[0], core.Var)
      x, val, *idx = in_vals
      z, x_new = _swap_discharge(x, val, idx, eqn.params['indexed_dims'])
      write(eqn.outvars[0], z)
      write(eqn.invars[0] , x_new)
    elif eqn.primitive is addupdate_p:
      # `x[i] += val` becomes:
      #    y = ds x i
      #    z = y + val
      #    x = dus x i z
      assert isinstance(eqn.invars[0], core.Var)
      x, val, *idx = in_vals
      ans = _addupdate_discharge(x, val, idx, eqn.params['indexed_dims'])
      write(eqn.invars[0], ans)
    else:
      # Default primitive rule, similar to `core.eval_jaxpr`. Note that here
      # we assume any higher-order primitives inside of the jaxpr are *not*
      # stateful.
      subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
      ans = eqn.primitive.bind(*subfuns, *map(read, eqn.invars), **bind_params)
      if eqn.primitive.multiple_results:
        map(write, eqn.outvars, ans)
      else:
        write(eqn.outvars[0], ans)
  # By convention, we return the outputs of the jaxpr first and then the final
  # values of the `Ref`s. Callers to this function should be able to split
  # them up by looking at `len(jaxpr.outvars)`.
  out_vals = map(read, jaxpr.outvars)
  ref_vals = map(
      read, [v for v in jaxpr.invars if type(v.aval) is ShapedArrayRef])
  return out_vals + ref_vals
