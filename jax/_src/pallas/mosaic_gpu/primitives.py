# Copyright 2024 The JAX Authors.
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

"""Module for Pallas:MosaicGPU-specific JAX primitives and functions."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jax._src import api_util
from jax._src import core as jax_core
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src.interpreters import partial_eval as pe

run_scoped_p = jax_core.Primitive("run_scoped")
run_scoped_p.multiple_results = True


# TODO(cperivol): consolidate run_scoped with the pallas TPU version
# of the same op.
def run_scoped(f: Callable[..., Any], *types, **kw_types) -> Any:
  """Call the function with allocated shared mem references.

  Args:
    f: The function that generatest the jaxpr.
    *types: The types of the function's positional arguments.
    **kw_types: The types of the function's keyword arguments.
  """

  flat_types, in_tree = tree_util.tree_flatten((types, kw_types))
  flat_fun, out_tree_thunk = api_util.flatten_fun(lu.wrap_init(f), in_tree)
  avals = [t.get_aval() for t in flat_types]
  # Turn the function into a jaxpr. The body of run_scoped may have
  # effects (IO) on constvars (i.e. variables inherited from the
  # parent scope). Jax can't reason about effects to references that
  # are not in the invars of an operation so we just put them all
  # there.
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fun, avals)
  out = run_scoped_p.bind(*consts, jaxpr=jaxpr)
  return tree_util.tree_unflatten(out_tree_thunk(), out)


@run_scoped_p.def_effectful_abstract_eval
def _run_scoped_abstract_eval(*args, jaxpr):
  del args
  # jaxpr will have effects for its inputs (Refs that are allocated) and for
  # constvars (closed over Refs). The effects for the allocated Refs are local
  # to the jaxpr and shouldn't propagate out.
  nonlocal_effects = {
      eff
      for eff in jaxpr.effects
      if not (
          isinstance(eff, effects.JaxprInputEffect)
          and eff.input_index >= len(jaxpr.constvars)
      )
  }
  return [v.aval for v in jaxpr.outvars], nonlocal_effects
