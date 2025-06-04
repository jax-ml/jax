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

from collections.abc import Sequence
import functools
from typing import Any
import jax
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas.fuser import fusible_dtype
from jax._src.pallas.fuser import fusion as fusion_lib
from jax._src.pallas.fuser.fusible import fusible_p


def fuse(f=None, *, resolve_fusion_dtypes: bool = True, debug: bool = False):
  """Fuses a function into a single fusible.

  Args:
    f: The function to fuse.
    resolve_fusion_dtypes: (experimental) whether or not to resolve fusion
      dtypes (which don't correspond to physical dtypes)
    debug: Whether to print debug information.

  There should be a single call to a `fusible` inside the body of `f`. `fuse`
  returns a transformed function that will fuse the surrounding computation into
  the fusible and invoke it.
  """

  def decorator(f):
    def wrapper(*args, **kwargs):
      flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
      debug_info = api_util.debug_info("fuse", f, args, kwargs)
      flat_fun, out_tree_thunk = api_util.flatten_fun(
          lu.wrap_init(f, debug_info=debug_info), in_tree
      )
      flat_avals = [jax_core.get_aval(x) for x in flat_args]
      jaxpr, _, consts, _ = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
      if debug:
        print("Jaxpr before fusion:")
        print(jaxpr)
      out_tree = out_tree_thunk()
      out_flat = fuse_jaxpr(jaxpr, out_tree, consts, *flat_args)
      return tree_util.tree_unflatten(out_tree, out_flat)

    if resolve_fusion_dtypes:
      wrapper = fusible_dtype.physicalize(wrapper)
    return wrapper

  if f is not None:
    return decorator(f)
  return decorator


_fusible: dict[jax_core.Primitive, Any] = {}


def _construct_fusion_jaxpr(
    candidate_values, jaxpr: jax_core.Jaxpr, outvars, *invars, **kwargs
):
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
  flat_in_type = [
      jax.ShapeDtypeStruct(x.aval.shape, x.aval.dtype) for x in flat_invars
  ]
  in_type = tree_util.tree_unflatten(kernel_in_tree, flat_in_type)
  out_type = tree_util.tree_unflatten(
      out_tree,
      [jax.ShapeDtypeStruct(x.aval.shape, x.aval.dtype) for x in flat_outvars],
  )
  return new_jaxpr, new_values, in_type, out_type, out_tree


def construct_fusion(
    candidate_values, jaxpr: jax_core.Jaxpr, outvars, *invars, **kwargs
) -> fusion_lib.Fusion:
  new_jaxpr, new_values, in_type, out_type, out_tree = _construct_fusion_jaxpr(
      candidate_values, jaxpr, outvars, *invars, **kwargs
  )

  def _fn(*args, **kwargs):
    flat_args, _ = tree_util.tree_flatten((args, kwargs))
    out_flat = jax_core.eval_jaxpr(new_jaxpr, new_values, *flat_args)
    return tree_util.tree_unflatten(out_tree, out_flat)

  return fusion_lib.Fusion(_fn, in_type, out_type)


def _find_downstream(
    jaxpr: jax_core.Jaxpr, in_used: Sequence[bool]
) -> tuple[bool, ...]:
  # TODO(sharadmv): We use partial_eval to query downstream dependencies which
  # is not an officially sanctioned way to do so, since PE is really used for
  # AD. In the future, we should have a special Jaxpr API that queries this.
  _, _, out_used, *_ = pe.partial_eval_jaxpr_custom(
      jaxpr,
      in_unknowns=in_used,
      in_inst=in_used,
      ensure_out_unknowns=False,
      ensure_out_inst=False,
      saveable=lambda *_, **__: False,
  )
  return tuple(out_used)


def _construct_output_permutation(
    used: list[tuple[bool, ...]],
) -> list[int]:
  order = []
  for u in used:
    true_vals = [i for i in range(len(u)) if u[i]]
    order.extend(true_vals)
  return [order.index(i) for i in range(len(order))]


def _construct_output_fusions(
    candidate_values,
    jaxpr,
    out_tree,
    fusion_eqn_index,
    fusion_eqn_outvars,  # Flat list of vars output by the fusible eqn
    fusion_eqn_out_tree,  # Tree structure of the fusible eqn outputs
    output_fusion_prefix,  # Pytree defining output groups
):
  # 1. Create jaxpr_out: represents computation *after* the fusible
  #    Inputs: fusion_eqn_outvars
  #    Outputs: jaxpr.outvars
  jaxpr_out, all_values, _, _, _ = _construct_fusion_jaxpr(
      candidate_values,
      jaxpr.replace(
          eqns=jaxpr.eqns[:fusion_eqn_index]
          + jaxpr.eqns[fusion_eqn_index + 1 :]
      ),
      tree_util.tree_unflatten(out_tree, jaxpr.outvars),  # Original outputs
      tree_util.tree_unflatten(
          fusion_eqn_out_tree, fusion_eqn_outvars
      ),  # Fusible outputs as inputs
  )

  # 2. Group fusible outputs based on the mask
  unflat_fusible_outvars = jax.tree.unflatten(
      fusion_eqn_out_tree, fusion_eqn_outvars
  )
  partial_flat = jax.tree.structure(output_fusion_prefix).flatten_up_to(
      unflat_fusible_outvars
  )

  # 3. Calculate dependencies and check disjointedness
  downstream_outputs_used_masks = []  # List of bool tuples, one per group
  already_used_final_outputs = set()  # Indices of final outputs already claimed
  for outvars_group in partial_flat:
    # Identify vars in this group
    used_fusible_outvars = set(jax.tree.leaves(outvars_group))
    # Create mask for jaxpr_out inputs corresponding to this group
    in_used_mask = [
        True if v in used_fusible_outvars else False for v in jaxpr_out.invars
    ]
    # Trace dependencies through jaxpr_out to find which final outputs are affected
    downstream_used_mask = _find_downstream(
        jaxpr_out, in_used_mask
    )  # Mask for jaxpr_out.outvars (== jaxpr.outvars)

    # Check for overlap in final output usage across groups
    for i, used in enumerate(downstream_used_mask):
      if used:
        if i in already_used_final_outputs:
          raise ValueError(
              "Outputs must be disjoint in order to use separate output fusions"
          )
        already_used_final_outputs.add(i)
    downstream_outputs_used_masks.append(downstream_used_mask)

  # 4. Construct output permutation needed to restore original output order
  output_permutation = _construct_output_permutation(
      downstream_outputs_used_masks
  )

  # Construct fusions for each group by DCEing the jaxpr_out
  output_fusions = []
  for i, outvars_group in enumerate(partial_flat):
    flat_group_vars, _ = tree_util.tree_flatten(outvars_group)
    downstream_used_mask = downstream_outputs_used_masks[i]

    used_jaxpr_invars = [False] * len(all_values) + [
        v in flat_group_vars for v in jaxpr_out.invars
    ]
    jaxpr_out_for_group, used_consts, _ = pe.dce_jaxpr_consts(
        jaxpr_out, downstream_used_mask, instantiate=used_jaxpr_invars
    )
    values_for_jaxpr = tuple(
        c for used, c in zip(used_consts, all_values, strict=True) if used
    )

    def _fn(jaxpr, vals, *args, **kwargs):
      flat_args, _ = tree_util.tree_flatten((args, kwargs))
      out_flat = jax_core.eval_jaxpr(jaxpr, vals, *flat_args)
      return tuple(out_flat)

    fn = functools.partial(_fn, jaxpr_out_for_group, values_for_jaxpr)
    in_type = jax.tree.map(
        lambda v: jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype),  # pytype: disable=attribute-error
        outvars_group,
    )
    out_type = tuple(
        jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype)  # pytype: disable=attribute-error
        for v in jaxpr_out_for_group.outvars
    )
    fusion = fusion_lib.Fusion(
        fn,
        (in_type, {}),
        out_type,
    )
    output_fusions.append(fusion)

  return (
      tree_util.tree_unflatten(
          tree_util.tree_structure(output_fusion_prefix), output_fusions
      ),
      output_permutation,
  )


def fuse_jaxpr(
    jaxpr: jax_core.Jaxpr, out_tree: tree_util.PyTreeDef, consts, *args
):
  fusion_eqn_index = None

  # Collect input fusions
  for i, eqn in enumerate(jaxpr.eqns):
    if eqn.primitive is fusible_p:
      fusion_eqn_index = i
      break
  if fusion_eqn_index is None:
    raise ValueError("No fusible eqn found")
  fusion_eqn = jaxpr.eqns[fusion_eqn_index]

  # Now let's check if we need to do any fusion at all, e.g. do the outputs of
  # the jaxpr have any dependence on the fusion at all? We can DCE the jaxpr
  # with all the inputs and outputs to check if there is a dependence.
  dced_jaxpr, _ = pe.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars),
                               instantiate=True)
  if not any(eqn.primitive is fusible_p for eqn in dced_jaxpr.eqns):
    # Short circuit if there is nothing to fuse.
    return jax_core.eval_jaxpr(dced_jaxpr, consts, *args)

  candidate_values = [*consts, *args]

  # Construct fusions for non-constant inputs to the fusible.
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
  output_fusions, output_permutation = _construct_output_fusions(
      candidate_values,
      jaxpr,
      out_tree,
      fusion_eqn_index,
      fusion_eqn.outvars,
      fusion_eqn.params["out_tree"],
      fusion_eqn.params["output_fusion_prefix"],
  )
  out = fusion_eqn.params["func"](*in_fusions, output_fusions)
  flat_out = jax.tree.leaves(out)
  permuted_out = [flat_out[i] for i in output_permutation]
  assert len(permuted_out) == len(jaxpr.outvars), (
      len(permuted_out),
      len(jaxpr.outvars),
  )
  return permuted_out
