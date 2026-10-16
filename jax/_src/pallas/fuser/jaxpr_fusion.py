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

from collections.abc import Iterable, Sequence
import contextlib
import functools
import math
from typing import Any

import jax
from jax import lax
from jax._src import api_util
from jax._src import config
from jax._src import core as jax_core
from jax._src import flattree as ft
from jax._src import hijax
from jax._src import shard_map
from jax._src import tree_util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas.fuser import block_spec
from jax._src.pallas.fuser import fuser_utils
from jax._src.pallas.fuser import fusible_dtype
from jax._src.pallas.fuser import fusion as fusion_lib
from jax._src.pallas.fuser.fusible import Fusible
from jax._src.state import types as state_types
from jax._src.traceback_util import api_boundary
import jax.numpy as jnp


_disable_nested_fuse = config.config_ext.Config[bool](
    "pallas_fuser_disable_nested_fuse",
    False,
    include_in_jit_key=True,
    include_in_trace_context=True,
)


@contextlib.contextmanager
def disable_nested_fuse(value: bool = True):
  # A way to prevent nested fuse calls from fusing.
  old_value = _disable_nested_fuse.swap_local(value)
  try:
    yield
  finally:
    _disable_nested_fuse.set_local(old_value)


@functools.partial(api_boundary, repro_api_name="fuser.fuse")
def fuse(
    f=None,
    *,
    resolve_fusion_dtypes: bool = True,
    debug: bool = False,
    strict_mode: bool = True,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
):
  """Fuses a function into a single fusible.

  There should be a single call to a `fusible` inside the body of `f`. `fuse`
  returns a transformed function that will fuse the surrounding computation into
  the fusible and invoke it. The fused function cannot take `jax.ref.Ref`s as
  arguments (it must close over `Ref`s instead).

  Args:
    f: The function to fuse.
    resolve_fusion_dtypes: (experimental) whether or not to resolve fusion
      dtypes (which don't correspond to physical dtypes)
    debug: Whether to print debug information.
    strict_mode: Whether to verify block index map equality in collisions during
      block spec propagations in output fusions.
    static_argnums: An optional int or collection of ints that specify which
      positional arguments to treat as static (compile-time constant).
    static_argnames: An optional string or collection of strings specifying
      which named arguments to treat as static (compile-time constant).

  Returns:
    A fused version of `f`, or a decorator if `f` is `None`.
  """

  def decorator(f):
    sig = api_util.fun_signature(f)
    _, _, static_argnums_, static_argnames_ = api_util.resolve_argnums(
        f,
        sig,
        donate_argnums=None,
        donate_argnames=None,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      in_ft = ft.flatten_static_argnums_argnames(
          args, kwargs, static_argnums_, static_argnames_
      )
      flat_args = in_ft.vals
      debug_info = api_util.debug_info(
          "fuse",
          f,
          args,
          kwargs,
          static_argnums=static_argnums_,
          static_argnames=static_argnames_,
      )
      ref_arg = next((v for v in flat_args if isinstance(v, jax.ref.Ref)), None)
      if ref_arg is not None:
        raise NotImplementedError(
            f"Fused function {debug_info.func_src_info} was passed an argument "
            f"of type {ref_arg}.  Fused functions cannot take Refs as "
            "arguments -- they must close over such Refs, instead.")
      in_avals_ft = in_ft.map(jax_core.typeof)
      # Disable nested fuse so the outer fuse captures the full (unfused) jaxpr.
      with disable_nested_fuse(True):
        closed_jaxpr, out_avals_ft = pe.trace_to_jaxpr(
            f, in_avals_ft, debug_info
        )
      jaxpr = closed_jaxpr
      consts = closed_jaxpr.consts
      if debug:
        print("Jaxpr before fusion:")
        print(jaxpr)
      out_tree = out_avals_ft.tree
      fuse_fn = functools.partial(
          fuse_jaxpr, jaxpr, out_tree, consts, strict_mode=strict_mode
      )
      if resolve_fusion_dtypes and fusible_dtype.has_fusion_dtypes(jaxpr):
        fuse_fn = fusible_dtype.physicalize(fuse_fn)
      out_flat = fuse_fn(*flat_args)
      return tree_util.tree_unflatten(out_tree, out_flat)

    @functools.wraps(f)
    def outer_wrapper(*args, **kwargs):
      if _disable_nested_fuse.value:
        return f(*args, **kwargs)
      return wrapper(*args, **kwargs)

    return outer_wrapper

  if f is not None:
    return decorator(f)
  return decorator


def _construct_fusion_jaxpr(
    candidate_values, jaxpr: jax_core.Jaxpr, outvars, *invars, **kwargs
):
  flat_outvars, out_tree = tree_util.tree_flatten(outvars)
  flat_invars, in_tree = tree_util.tree_flatten((invars, kwargs))
  new_jaxpr_no_dce = jaxpr.replace(
      outvars=flat_outvars,
      constvars=jaxpr.constvars + jaxpr.invars,
      invars=flat_invars,
      debug_info=jaxpr.debug_info.with_unknown_names(),
      consts=tuple(candidate_values),
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
  flat_in_type = [x.aval for x in flat_invars]
  in_type = tree_util.tree_unflatten(in_tree, flat_in_type)
  out_type = tree_util.tree_unflatten(
      out_tree,
      [x.aval for x in flat_outvars],
  )
  return new_jaxpr, new_values, in_type, out_type, out_tree


def construct_input_fusion(
    candidate_values, jaxpr: jax_core.Jaxpr, outvars
) -> fusion_lib.Fusion:
  new_jaxpr, new_values, in_type, out_type, out_tree = _construct_fusion_jaxpr(
      candidate_values, jaxpr, outvars,
  )

  def _fn():
    out_flat = jax_core.eval_jaxpr(new_jaxpr, new_values)
    return tree_util.tree_unflatten(out_tree, out_flat)

  return fusion_lib.Fusion(_fn, in_type, out_type)


def _find_downstream(
    jaxpr: jax_core.Jaxpr, in_used: Sequence[bool]
) -> tuple[bool, ...]:
  # TODO(sharadmv): We use partial_eval to query downstream dependencies which
  # is not an officially sanctioned way to do so, since PE is really used for
  # AD. In the future, we should have a special Jaxpr API that queries this.
  discharged_jaxpr, *_ = fuser_utils.discharge_state(jaxpr)
  _, _, out_used, *_ = pe.partial_eval_jaxpr_custom(
      discharged_jaxpr,
      in_unknowns=in_used,
      in_inst=in_used,
      ensure_out_unknowns=False,
      ensure_out_inst=False,
      saveable=lambda *_, **__: False,
  )
  # NOTE: out_used[:len(jaxpr.outvars)] reports whether or not the original
  # outputs depend on the inputs for which `in_used` is True.
  # out_used[len(jaxpr.outvars):] reports whether or not the new outputs
  # (updates for discharged Refs) depend on thse inputs.
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
    *,
    strict_mode: bool = True,
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
  if len([x for x in partial_flat if jax.tree.leaves(x)]) > 1:
    if any(isinstance(e, (state_types.WriteEffect, state_types.AccumEffect))
           for e in jaxpr_out.effects):
      raise ValueError("Multiple output fusions are not currently supported "
                       "for fusions that write to Refs.")

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

  for u in list(zip(*downstream_outputs_used_masks))[len(jaxpr_out.outvars):]:
    if sum(u) == 0:
      raise ValueError("A write to a Ref in a fusion must depend on "
                       "an output of the fusible")
  downstream_outputs_used_masks = [
      used[:len(jaxpr_out.outvars)] for used in downstream_outputs_used_masks]

  # 4. Construct output permutation needed to restore original output order
  output_permutation = _construct_output_permutation(
      downstream_outputs_used_masks
  )

  # Construct fusions for each group by DCEing the jaxpr_out
  output_fusions: list[fusion_lib.Fusion | None] = []
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

    if (
        not jaxpr_out_for_group.eqns
        and jaxpr_out_for_group.outvars == jaxpr_out_for_group.invars
    ):
      output_fusions.append(None)
      continue

    def _fn(jaxpr, vals, *args, **kwargs):
      flat_args, _ = tree_util.tree_flatten((args, kwargs))
      if config._check_vma.value:
        flat_args = [
            jax_core.pvary(a, tuple(v.aval.mat.varying - a_aval.mat.varying))
            if isinstance(v.aval, jax_core.ShapedArray)
            and isinstance(a_aval := jax_core.typeof(a), jax_core.ShapedArray)
            and (v.aval.mat.varying - a_aval.mat.varying)
            else a
            for v, a in zip(jaxpr.invars, flat_args, strict=True)
        ]
      out_flat = jax_core.eval_jaxpr(jaxpr, vals, *flat_args)
      return tuple(out_flat)

    fn = functools.partial(_fn, jaxpr_out_for_group, values_for_jaxpr)
    in_type = jax.tree.map(lambda x: x.aval, outvars_group)
    out_type = tuple(v.aval for v in jaxpr_out_for_group.outvars)
    fusion = fusion_lib.Fusion(
        fn,
        (in_type, {}),
        out_type,
        strict_mode=strict_mode,
    )
    output_fusions.append(fusion)

  return (
      tree_util.tree_unflatten(
          tree_util.tree_structure(output_fusion_prefix), output_fusions
      ),
      output_permutation,
  )


def _is_fusible_eqn(eqn: jax_core.JaxprEqn) -> bool:
  return eqn.primitive is hijax.call_hi_primitive_p and isinstance(
      eqn.params.get("_prim"), Fusible
  )


def _count_fusible_eqns(jaxpr: jax_core.Jaxpr) -> int:
  return sum(
      _count_fusible_eqns(eqn.params["jaxpr"])
      if eqn.primitive is shard_map.shard_map_p
      else int(_is_fusible_eqn(eqn))
      for eqn in jaxpr.eqns
  )


def _block_spec_from_sharding(
    shape: tuple[int, ...],
    spec: jax.sharding.PartitionSpec,
    mesh: jax.sharding.Mesh,
    manual_axes: tuple[Any, ...],
    axis_idx_cache: dict[tuple[Any, ...], Any],
) -> pallas_core.BlockSpec:
  names = shard_map._spec_to_names(spec)
  block_shape = tuple(
      sz
      // math.prod(mesh.shape[n] for n in names.get(i, ()) if n in manual_axes)
      for i, sz in enumerate(shape)
  )

  def index_map(*axis_indices):
    axis_map = dict(zip(manual_axes, axis_indices, strict=True))
    out = []
    for i in range(len(shape)):
      axes = tuple(n for n in names.get(i, ()) if n in manual_axes)
      if not axes:
        out.append(0)
      else:
        key = (axes, tuple(map(id, axis_indices)))
        if key not in axis_idx_cache:
          idx = axis_map[axes[0]]
          for n in axes[1:]:
            idx = idx * mesh.shape[n] + axis_map[n]
          axis_idx_cache[key] = idx
        out.append(axis_idx_cache[key])
    return tuple(out)

  return pallas_core.BlockSpec(block_shape=block_shape, index_map=index_map)


def _extract_local_block(
    v_local: jax.Array,
    shard_bs: pallas_core.BlockSpec,
    pulled_bs: pallas_core.BlockSpec | pallas_core.NoBlockSpec,
    axis_indices: tuple[jax.Array, ...],
    sp_vals: tuple[Any, ...],
) -> jax.Array:
  if not isinstance(pulled_bs, pallas_core.BlockSpec) or v_local.ndim == 0:
    return v_local
  assert shard_bs.index_map is not None and pulled_bs.index_map is not None
  assert pulled_bs.block_shape is not None
  shard_idx = shard_bs.index_map(*axis_indices)
  pulled_idx = pulled_bs.index_map(*axis_indices, *sp_vals)
  starts, sizes, squeezed = [], [], []
  for d, (s_d, bd) in enumerate(zip(v_local.shape, pulled_bs.block_shape)):
    b_d = pallas_core.get_block_size(bd)
    sizes.append(b_d)
    if bd is None or isinstance(bd, pallas_core.Squeezed):
      squeezed.append(d)
    if pulled_idx[d] is shard_idx[d] and b_d == s_d:
      starts.append(0)
    else:
      p_off = (
          pulled_idx[d]
          if bd is None
          or isinstance(bd, (pallas_core.Squeezed, pallas_core.Element))
          else pulled_idx[d].start
          if isinstance(bd, pallas_core.BoundedSlice)
          else pulled_idx[d] * b_d
      )
      s_off = shard_idx[d] * s_d
      starts.append(
          p_off if isinstance(s_off, int) and s_off == 0 else p_off - s_off
      )
  out = v_local
  if all(isinstance(s, int) for s in starts):
    if any(
        s != 0 or sz != d_sz
        for s, sz, d_sz in zip(starts, sizes, v_local.shape)
    ):
      out = lax.slice(out, starts, [s + sz for s, sz in zip(starts, sizes)])
  else:
    dtypes = [s.dtype for s in starts if hasattr(s, "dtype")]
    dtype = jax.dtypes.result_type(*dtypes) if dtypes else jnp.int32
    idx_arrays = [jnp.asarray(s, dtype=dtype) for s in starts]
    if config._check_vma.value:
      vma = frozenset().union(*(
          a.mat.varying
          for v in (out, *idx_arrays)
          if isinstance(a := jax_core.typeof(v), jax_core.ShapedArray)
      ))
      if vma:
        out, *idx_arrays = [
            jax_core.pvary(v, tuple(vma - a.mat.varying))
            if isinstance(a := jax_core.typeof(v), jax_core.ShapedArray)
            and (vma - a.mat.varying)
            else v
            for v in (out, *idx_arrays)
        ]
    out = lax.dynamic_slice(out, idx_arrays, sizes)
  return lax.squeeze(out, squeezed) if squeezed else out


def _fuse_shard_map_jaxpr(
    jaxpr: jax_core.Jaxpr,
    consts: Sequence[Any],
    args: Sequence[Any],
    shmap_eqn_index: int,
    *,
    strict_mode: bool = True,
):
  shmap_eqn = jaxpr.eqns[shmap_eqn_index]
  inner_jaxpr: jax_core.Jaxpr = shmap_eqn.params["jaxpr"]
  mesh: jax.sharding.Mesh = shmap_eqn.params["mesh"]
  shmap_in_specs = shmap_eqn.params["in_specs"]
  shmap_out_specs = shmap_eqn.params["out_specs"]
  check_vma: bool = shmap_eqn.params["check_vma"]
  newly_manual: frozenset[Any] = shmap_eqn.params["newly_manual_axes"]
  manual_axes = tuple(a for a in mesh.axis_names if a in newly_manual)
  num_axes = len(manual_axes)
  axis_idx_cache: dict[tuple[Any, ...], Any] = {}
  outer_vars = [*jaxpr.constvars, *jaxpr.invars]
  outer_vals = [*consts, *args]
  outer_val_map = dict(zip(outer_vars, outer_vals, strict=True))
  cand_vars = [*outer_vars, *shmap_eqn.outvars]
  cand_vals = [*outer_vals, *(v.aval for v in shmap_eqn.outvars)]
  non_shmap_jaxpr = jaxpr.replace(
      eqns=jaxpr.eqns[:shmap_eqn_index] + jaxpr.eqns[shmap_eqn_index + 1 :],
      constvars=cand_vars,
      invars=[],
      consts=tuple(cand_vals),
      debug_info=jaxpr.debug_info.with_unknown_names(),
  )

  def _manual(aval):
    if not isinstance(aval, jax_core.ShapedArray):
      return jax.P()
    return shard_map._manual_spec(newly_manual, aval.sharding.spec, mesh)

  def _bs(aval, spec):
    return _block_spec_from_sharding(
        aval.shape, spec, mesh, manual_axes, axis_idx_cache
    )

  has_auto = any(
      mesh._name_to_type[a] == jax_core.AxisType.Auto for a in newly_manual
  )
  var_specs = {
      **{v: _manual(v.aval) for v in outer_vars},
      **dict(zip(shmap_eqn.outvars, shmap_out_specs, strict=True)),
  }
  if has_auto:
    var_specs.update(
        (v, s)
        for v, s in zip(shmap_eqn.invars, shmap_in_specs, strict=True)
        if isinstance(v, jax_core.Var)
    )
    for eqn in reversed(jaxpr.eqns[:shmap_eqn_index]):
      specs = [var_specs[v] for v in eqn.outvars if v in var_specs]
      for v in eqn.invars:
        if isinstance(v, jax_core.Var) and not any(var_specs.get(v, ())):
          if specs and getattr(v.aval, "ndim", -1) == getattr(
              eqn.outvars[0].aval, "ndim", -2
          ):
            var_specs[v] = specs[0]
    for eqn in jaxpr.eqns[shmap_eqn_index + 1 :]:
      specs = [
          var_specs[v]
          for v in eqn.invars
          if isinstance(v, jax_core.Var) and v in var_specs
      ]
      for v in eqn.outvars:
        if specs and getattr(v.aval, "ndim", -1) == getattr(
            eqn.invars[0].aval, "ndim", -2
        ):
          var_specs[v] = specs[0]
  var_shard_bs = {
      v: _bs(v.aval, s)
      for v, s in var_specs.items()
      if isinstance(v.aval, jax_core.ShapedArray)
  }
  used_outer_vars: list[jax_core.Var] = []

  def _pull(v: jax_core.Var, spec: jax.sharding.PartitionSpec):
    sub_jaxpr, used, _ = pe.dce_jaxpr_consts(
        non_shmap_jaxpr.replace(outvars=[v]),
        [True],
        instantiate=[False] * len(cand_vars),
    )
    used_vars = [u for keep, u in zip(used, cand_vars, strict=True) if keep]
    used_vals = [c for keep, c in zip(used, cand_vals, strict=True) if keep]
    usage = block_spec.compute_usage(sub_jaxpr, ({block_spec.Usage.REGULAR},))
    sp_mask = block_spec._scalar_prefetch_mask(sub_jaxpr, usage)
    sp_vars = [u for u, sp in zip(used_vars, sp_mask, strict=True) if sp]
    sp_vals = [c for c, sp in zip(used_vals, sp_mask, strict=True) if sp]
    reg_vars = [u for u, sp in zip(used_vars, sp_mask, strict=True) if not sp]
    for u in (*reg_vars, *sp_vars):
      if u in outer_val_map and u not in used_outer_vars:
        used_outer_vars.append(u)
    split_jaxpr = sub_jaxpr.replace(
        constvars=sp_vars, invars=reg_vars, consts=tuple(sp_vals)
    )
    usage = block_spec.compute_usage(split_jaxpr, ({block_spec.Usage.REGULAR},))
    in_bs, env = block_spec._pull_block_spec(
        split_jaxpr,
        (_bs(v.aval, spec),),
        scalar_prefetch_handler=None,
        read_usage_env=usage,
        grid_len=num_axes,
        strict_mode=strict_mode,
    )
    kfn = block_spec.make_kernel_function(
        split_jaxpr,
        tuple(sp_vals),
        tree_util.tree_structure((tuple(reg_vars), {})),
        tree_util.tree_structure((v,)),
        usage,
        env,
        None,
        num_axes,
    )
    in_bs = tuple(
        block_spec._wrap_block_spec_scalar_prefetch(bs, num_grid_args=num_axes)
        if isinstance(bs, pallas_core.BlockSpec)
        else bs
        for bs in in_bs
    )

    def _eval(val_env, axis_indices):
      sp = tuple(val_env[u] for u in sp_vars)
      reg = tuple(
          _extract_local_block(
              val_env[u], var_shard_bs[u], bs, axis_indices, sp
          )
          for u, bs in zip(reg_vars, in_bs, strict=True)
      )
      return kfn(axis_indices, sp, *reg)[0]

    return _eval

  pre_fns = [
      (lambda _env, _idx, val=v.val: val)
      if isinstance(v, jax_core.Literal)
      else _pull(v, s)
      for v, s in zip(shmap_eqn.invars, shmap_in_specs, strict=True)
  ]
  post_fns, final_out_specs, out_resolvers = [], [], []
  for v in jaxpr.outvars:
    if isinstance(v, jax_core.Literal):
      out_resolvers.append(lambda _outs, val=v.val: val)
    elif v in outer_val_map:
      out_resolvers.append(lambda _outs, val=outer_val_map[v]: val)
    else:
      sub_jaxpr, used, _ = pe.dce_jaxpr_consts(
          non_shmap_jaxpr.replace(outvars=[v]),
          [True],
          instantiate=[False] * len(cand_vars),
      )
      if not any(used[len(outer_vars) :]):
        consts_v = [c for u, c in zip(used[: len(outer_vars)], outer_vals) if u]
        val = jax_core.eval_jaxpr(sub_jaxpr, consts_v)[0]
        out_resolvers.append(lambda _outs, val=val: val)
      else:
        spec = var_specs.get(v, _manual(v.aval))
        post_fns.append(_pull(v, spec))
        final_out_specs.append(spec)
        out_resolvers.append(lambda outs, idx=len(post_fns) - 1: outs[idx])

  def combined_inner_fn(*local_outer_vals):
    env = dict(zip(used_outer_vars, local_outer_vals, strict=True))
    axis_indices = tuple(lax.axis_index(a) for a in manual_axes)
    in_locals = [fn(env, axis_indices) for fn in pre_fns]
    out_locals = jax_core.eval_jaxpr(
        inner_jaxpr, inner_jaxpr.consts, *in_locals
    )
    env.update(zip(shmap_eqn.outvars, out_locals, strict=True))
    return tuple(fn(env, axis_indices) for fn in post_fns)

  def fused_shmap_body(*local_outer_vals):
    in_ft = ft.flatten((local_outer_vals, {}))
    dbg = api_util.debug_info(
        "fuse_shard_map", combined_inner_fn, local_outer_vals, {}
    )
    with disable_nested_fuse(True):
      combined_jaxpr, out_avals_ft = pe.trace_to_jaxpr(
          combined_inner_fn, in_ft.map(jax_core.typeof), dbg
      )
    return tuple(
        fuse_jaxpr(
            combined_jaxpr,
            out_avals_ft.tree,
            combined_jaxpr.consts,
            *in_ft.vals,
            strict_mode=strict_mode,
        )
    )

  shmap_outs = (
      shard_map.shard_map(
          fused_shmap_body,
          mesh=mesh,
          in_specs=tuple(var_specs[v] for v in used_outer_vars),
          out_specs=tuple(final_out_specs),
          axis_names=newly_manual,
          check_vma=check_vma,
      )(*(outer_val_map[v] for v in used_outer_vars))
      if post_fns
      else ()
  )
  return [resolve(shmap_outs) for resolve in out_resolvers]


def fuse_jaxpr(
    jaxpr: jax_core.Jaxpr, out_tree: tree_util.PyTreeDef, consts, *args,
    strict_mode: bool = True,
):
  # Collect input fusions
  top_fusibles = [i for i, eqn in enumerate(jaxpr.eqns) if _is_fusible_eqn(eqn)]
  shmap_fusibles = [
      i
      for i, eqn in enumerate(jaxpr.eqns)
      if eqn.primitive is shard_map.shard_map_p
      and _count_fusible_eqns(eqn.params["jaxpr"]) > 0
  ]
  total_fusibles = len(top_fusibles) + sum(
      _count_fusible_eqns(jaxpr.eqns[i].params["jaxpr"]) for i in shmap_fusibles
  )
  if total_fusibles == 0:
    raise ValueError("No fusible eqn found")
  if total_fusibles > 1:
    raise ValueError("Multiple fusible equations found")
  if shmap_fusibles:
    return _fuse_shard_map_jaxpr(
        jaxpr, consts, args, shmap_fusibles[0], strict_mode=strict_mode
    )
  fusion_eqn_index = top_fusibles[0]
  fusion_eqn = jaxpr.eqns[fusion_eqn_index]
  fusible: Fusible = fusion_eqn.params["_prim"]

  # Now let's check if we need to do any fusion at all, e.g. do the outputs of
  # the jaxpr have any dependence on the fusion at all?
  candidate_values = [*consts, *args]
  jaxpr_without_fusible = jaxpr.replace(
      eqns=(jaxpr.eqns[:fusion_eqn_index]
                + jaxpr.eqns[fusion_eqn_index + 1 :]),
      constvars=jaxpr.constvars + jaxpr.invars,
      invars=fusion_eqn.outvars,
      debug_info=jaxpr.debug_info.with_unknown_names(),
      consts=tuple(candidate_values))
  discharged_jaxpr_without_fusible, *_ = (
      fuser_utils.discharge_state(jaxpr_without_fusible))
  independent_jaxpr, _, out_used, *_ = pe.partial_eval_jaxpr_custom(
      discharged_jaxpr_without_fusible,
      in_unknowns=[True] * len(fusion_eqn.outvars),
      in_inst=[True] * len(fusion_eqn.outvars),
      ensure_out_unknowns=False,
      ensure_out_inst=False,
      saveable=lambda *_, **__: False)
  if not any(out_used):
    # Short circuit if there is no need to run the fusible at all.
    if discharged_jaxpr_without_fusible is not jaxpr_without_fusible:
      independent_jaxpr, _, out_used, *_ = pe.partial_eval_jaxpr_custom(
          jaxpr_without_fusible,
          in_unknowns=[True] * len(fusion_eqn.outvars),
          in_inst=[True] * len(fusion_eqn.outvars),
          ensure_out_unknowns=False,
          ensure_out_inst=False,
          saveable=lambda *_, **__: False)
      assert not any(out_used)
    return jax_core.eval_jaxpr(independent_jaxpr, candidate_values)

  # Construct fusions for non-constant inputs to the fusible.
  in_fusions_flat = [
      construct_input_fusion(
          candidate_values,
          jaxpr.replace(
              eqns=jaxpr.eqns[:fusion_eqn_index],
          ),
          var,
      )
      for var in fusion_eqn.invars[fusible.num_consts :]
  ]
  in_fusions = tree_util.tree_unflatten(fusible.args_tree, in_fusions_flat)
  output_fusions, output_permutation = _construct_output_fusions(
      candidate_values,
      jaxpr,
      out_tree,
      fusion_eqn_index,
      fusion_eqn.outvars,
      fusible.out_tree,
      fusible.output_fusion_prefix,
      strict_mode=strict_mode,
  )
  out = fusible.func(*in_fusions, output_fusions)
  flat_out = jax.tree.leaves(out)
  permuted_out = [flat_out[i] for i in output_permutation]
  assert len(permuted_out) == len(jaxpr.outvars), (
      len(permuted_out),
      len(jaxpr.outvars),
  )
  return permuted_out
