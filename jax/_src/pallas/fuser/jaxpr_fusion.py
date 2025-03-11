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

"""Fuses a function."""

from typing import Any

import jax
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src.interpreters import partial_eval as pe

from jax._src.pallas.fuser import fusable_dtype
from jax._src.pallas.fuser import fusion as fusion_lib
from jax._src.pallas.fuser.fusable import fusable_p


def _get_aval(x):
  return jax_core.raise_to_shaped(jax_core.get_aval(x))


def fuse(f=None, *, physicalize: bool = False, debug: bool = False):
  """Fuses a function into a single fusable.

  Args:
    f: The function to fuse.
    physicalize: (experimental) whether to physicalize the function.
    debug: Whether to print debug information.

  There should be a single call to a `fusable` inside the body of `f`. `fuse`
  returns a transformed function that will fuse the surrounding computation into
  the fusable and invoke it.
  """

  def decorator(f):
    def wrapper(*args, **kwargs):
      flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
      debug_info = api_util.debug_info("fuse", f, args, kwargs)
      flat_fun, out_tree_thunk = api_util.flatten_fun(
          lu.wrap_init(f, debug_info=debug_info), in_tree
      )
      flat_avals = [_get_aval(x) for x in flat_args]
      jaxpr, _, consts, _ = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
      if debug:
        print("Jaxpr before fusion:")
        print(jaxpr)
      out_tree = out_tree_thunk()
      out_flat = fuse_jaxpr(jaxpr, out_tree, consts, *flat_args)
      return tree_util.tree_unflatten(out_tree, out_flat)

    if physicalize:
      wrapper = fusable_dtype.physicalize(wrapper)
    return wrapper

  if f is not None:
    return decorator(f)
  return decorator


_fusable: dict[jax_core.Primitive, Any] = {}


def construct_fusion(
    candidate_values, jaxpr: jax_core.Jaxpr, outvars, *invars, **kwargs
) -> fusion_lib.Fusion:
  flat_outvars, out_tree = tree_util.tree_flatten(outvars)
  flat_invars, in_tree = tree_util.tree_flatten((invars, kwargs))
  new_jaxpr_no_dce = jaxpr.replace(
      outvars=flat_outvars,
      constvars=jaxpr.constvars + jaxpr.invars,
      invars=flat_invars,
  )
  new_jaxpr, used_consts, used_invars = pe.dce_jaxpr_consts(
      new_jaxpr_no_dce,
      [True] * len(new_jaxpr_no_dce.outvars),
      instantiate=[False] * len(new_jaxpr_no_dce.constvars)
      + [True] * len(new_jaxpr_no_dce.invars),
  )
  assert all(used_invars), new_jaxpr_no_dce
  new_values = tuple(
      c for used, c in zip(used_consts, candidate_values, strict=True) if used
  )
  kernel_in_tree = tree_util.tree_structure((invars, kwargs))

  def _fn(*args, **kwargs):
    flat_args, _ = tree_util.tree_flatten((args, kwargs))
    out_flat = jax_core.eval_jaxpr(new_jaxpr, new_values, *flat_args)
    return tree_util.tree_unflatten(out_tree, out_flat)

  flat_in_type = [
      jax.ShapeDtypeStruct(x.aval.shape, x.aval.dtype) for x in flat_invars
  ]
  in_type = tree_util.tree_unflatten(kernel_in_tree, flat_in_type)
  out_type = tree_util.tree_unflatten(
      out_tree,
      [jax.ShapeDtypeStruct(x.aval.shape, x.aval.dtype) for x in flat_outvars],
  )
  return fusion_lib.Fusion(_fn, in_type, out_type)


def fuse_jaxpr(
    jaxpr: jax_core.Jaxpr, out_tree: tree_util.PyTreeDef, consts, *args
):
  fusion_eqn_index = None

  # Collect input fusions
  for i, eqn in enumerate(jaxpr.eqns):
    if eqn.primitive is fusable_p:
      fusion_eqn_index = i
      break
  if fusion_eqn_index is None:
    raise ValueError("No fusable eqn found")
  fusion_eqn = jaxpr.eqns[fusion_eqn_index]

  candidate_values = [*consts, *args]

  # Construct fusions for non-constant inputs to the fusable.
  in_fusions_flat = [
      construct_fusion(
          candidate_values,
          jaxpr.replace(
              eqns=jaxpr.eqns[:fusion_eqn_index],
          ),
          var,
      )
      for var in fusion_eqn.invars[fusion_eqn.params["num_consts"] :]
  ]
  in_fusions = tree_util.tree_unflatten(
      fusion_eqn.params["in_tree"], in_fusions_flat
  )
  out_fusion = construct_fusion(
      candidate_values,
      jaxpr.replace(
          eqns=jaxpr.eqns[:fusion_eqn_index]
          + jaxpr.eqns[fusion_eqn_index + 1 :]
      ),
      tree_util.tree_unflatten(out_tree, jaxpr.outvars),
      tree_util.tree_unflatten(
          fusion_eqn.params["out_tree"], fusion_eqn.outvars
      ),
  )
  # Run the fusable.
  out = fusion_eqn.params["func"](*in_fusions, out_fusion)

  # Now return the flattened output (the fuse_jaxpr caller should unflatten).
  out_flat = tree_util.tree_leaves(out)
  assert len(out_flat) == len(jaxpr.outvars)
  return out_flat
