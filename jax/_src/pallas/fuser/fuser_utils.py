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

from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src.interpreters import partial_eval as pe
import numpy as np


def make_jaxpr(f, *args, **kwargs):
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  flat_avals = [core.shaped_abstractify(x) for x in flat_args]
  debug_info = api_util.debug_info('make_jaxpr', f, args, kwargs)
  flat_fun, out_tree_thunk = api_util.flatten_fun(
      lu.wrap_init(f, debug_info=debug_info), in_tree
  )
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
  out_tree = out_tree_thunk()
  return jaxpr, consts, in_tree, out_tree


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

  eqn_sigs = []
  for eqn in jaxpr.eqns:
    in_sigs = tuple(hash(get_var_sig(v)) for v in eqn.invars)
    params_sig = hash(_make_hashable(eqn.params))
    effects = tuple(sorted(str(e) for e in getattr(eqn, 'effects', [])))
    op_sig = ('eqn', eqn.primitive.name, in_sigs, params_sig, effects)
    eqn_sigs.append(op_sig)
    for i, outvar in enumerate(eqn.outvars):
      if type(outvar).__name__ != 'DropVar':
        env[outvar] = ('out', op_sig, i)  # pyrefly: ignore[unsupported-operation]
  out_sigs = tuple(get_var_sig(v) for v in jaxpr.outvars)
  jaxpr_effects = tuple(sorted(str(e) for e in getattr(jaxpr, 'effects', [])))
  eqn_sigs_sorted = tuple(sorted(eqn_sigs, key=lambda x: str(x)))
  return ('jaxpr_dag', out_sigs, eqn_sigs_sorted, jaxpr_effects)


def compare_jaxprs(jaxpr1, jaxpr2):
  """Compares two JAXPRs for symbolic equivalence."""
  sig1 = _jaxpr_signature(jaxpr1)
  sig2 = _jaxpr_signature(jaxpr2)
  return sig1 == sig2
