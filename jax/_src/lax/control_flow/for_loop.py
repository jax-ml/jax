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
from __future__ import annotations

from functools import partial

import numpy as np

from jax._src import api
from jax._src import core
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src.util import weakref_lru_cache
from jax._src.state import discharge, types
from jax._src.api_util import flatten_fun_nokwargs, debug_info
from jax._src.tree_util import tree_map, tree_flatten, tree_unflatten
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow import loops

# These are backward-compatibility shims, and probably shouldn't be used.

def for_loop_unrolled(nsteps, body, init_state, *, unroll=1):
  dbg = debug_info("for_loop", body, (0, init_state), {})
  (_, *state_flat), in_tree = tree_flatten((0, init_state))
  in_avals = [types.AbstractRef(core.typeof(x)) for x in state_flat]
  jaxpr, consts = _trace_to_jaxpr(body, in_tree, tuple(in_avals), dbg)
  for i in range(nsteps):
    state_flat = core.eval_jaxpr(jaxpr, consts, i, *state_flat)
  return tree_unflatten(in_tree.children()[1], state_flat)

@weakref_lru_cache
def _trace_to_jaxpr(fn, in_tree, in_avals, debug_info):
  f, _ = flatten_fun_nokwargs(lu.wrap_init(fn, debug_info=debug_info), in_tree)
  idx_aval = core.ShapedArray((), dtypes.canonicalize_dtype(np.int64))
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(f, [idx_aval, *in_avals])
  dis = [False] + [True] * len(in_avals)
  jaxpr, consts = discharge.discharge_state(jaxpr, consts, should_discharge=dis)
  return jaxpr, consts

for_loop = for_loop_unrolled


def for_loop_rolled(nsteps, body, init_state, *, unroll=1):
  return _for_loop_rolled(nsteps, body, init_state, unroll)

@partial(api.jit, static_argnums=(0, 1, 3))
def _for_loop_rolled(n, body, init_val, unroll):
  refs = tree_map(core.array_ref, init_val)
  def body_(i, _):
    body(i, refs)
    return i + 1, ()
  _, () = loops.scan(body_, 0, (), length=n, unroll=unroll)
  return tree_map(core.freeze, refs)
