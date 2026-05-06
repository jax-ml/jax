# Copyright 2026 The JAX Authors.
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

"""APIs for defining MPMD kernels in Pallas."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import contextlib
import functools
import itertools as it
from typing import Any, TypeVar, cast

from jax._src import api
from jax._src import api_util
from jax._src import config
from jax._src import core as jax_core
from jax._src import effects
from jax._src import hijax
from jax._src import linear_util as lu
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.frozen_dict import FrozenDict
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas import pallas_call
from jax._src.state import discharge as state_discharge
from jax._src.typing import Array


_T = TypeVar("_T")


mpmd_map_p = hijax.HiPrimitive("mpmd_map")
mpmd_map_p.multiple_results = True


@mpmd_map_p.def_impl
def _mpmd_map_impl(*args, **params):
  jit_impl = api.jit(functools.partial(mpmd_map_p.bind, **params))
  with config.disable_jit(False):
    return jit_impl(*args)


@mpmd_map_p.def_effectful_abstract_eval
def _mpmd_map_abstract_eval(
    *in_avals,
    jaxprs,
    out_avals,
    input_output_aliases,
    interpret,
    compiler_params,
    meshes,
    **params,
):
  del params  # Unused.

  effs = {*pallas_core.get_interpret_effects(interpret)}
  all_mesh_axis_names = {
      eff.name
      for jaxpr in jaxprs
      for eff in jaxpr.effects
      if isinstance(eff, jax_core.NamedAxisEffect)
  }
  for mesh, jaxpr in zip(meshes, jaxprs):
    for eff in jaxpr.effects:
      if mesh.discharges_effect(eff):
        continue
      if pallas_core.kernel_local_effects.contains(eff):
        continue
      if isinstance(eff, effects.JaxprInputEffect):
        # We emit an effect if we have a Ref input that has been written to in
        # the kernel.
        assert not jaxpr.constvars
        if eff.input_index < len(in_avals) and isinstance(
            in_avals[eff.input_index], state.AbstractRef
        ):
          effs.add(eff)
        continue
      if not isinstance(eff, jax_core.NamedAxisEffect):
        effs.add(eff)
        continue
      if eff.name not in all_mesh_axis_names:
        effs.add(eff)
  if getattr(compiler_params, "has_side_effects", False):
    # TODO(slebedev): Fix internal breakages and add
    # ``jax_core.GenericEffect(pallas_call_p)`` here.
    effs = jax_core.no_effects

  # TODO(slebedev): Handle pinned buffers as in ``pallas_call``.
  outin_aliases = {
      out_idx: in_idx for in_idx, out_idx in input_output_aliases.items()
  }
  out_avals = [
      in_avals[outin_aliases[out_idx]] if out_idx in outin_aliases else a
      for out_idx, a in enumerate(out_avals)
  ]
  return out_avals, effs


def _mpmd_map_typecheck_rule(ctx_factory, *in_atoms, meshes, **params):
  del ctx_factory  # Unused.
  ctx = contextlib.ExitStack()
  for mesh in meshes:
    ctx.enter_context(jax_core.extend_axis_env_nd(mesh.shape.items()))
  with ctx:
    return _mpmd_map_abstract_eval(
        *(x.aval for x in in_atoms), meshes=meshes, **params
    )


jax_core.custom_typechecks[mpmd_map_p] = _mpmd_map_typecheck_rule


def _mpmd_map_discharge_rule(
    avals_in: Sequence[jax_core.AbstractValue],
    avals_out: Sequence[jax_core.AbstractValue],
    *args: Any,
    jaxprs,
    meshes,
    input_output_aliases,
    debug,
    interpret,
    compiler_params,
    cost_estimate,
    metadata,
    name,
    external_meshes,
    **_,
):
  write_indices = set()
  for jaxpr in jaxprs:
    for eff in jaxpr.effects:
      if isinstance(eff, (state.WriteEffect, state.AccumEffect)):
        write_index = eff.input_index
        if write_index < len(avals_in):
          write_indices.add(write_index)

  write_indices = sorted(write_indices)
  num_in = len(avals_in)
  num_out_orig = len(avals_out)
  num_out_new = len(write_indices)

  new_jaxprs = []
  super_mesh_shape = get_super_mesh_shape(it.chain(meshes, external_meshes))

  def _rewrite_to_include_new_outputs(jaxpr):

    def new_body(*args):
      in_refs, orig_out_refs, new_out_refs, scratch_refs = util.split_list(
          args, [num_in, num_out_orig, num_out_new]
      )
      del new_out_refs
      jax_core.eval_jaxpr(
          jaxpr, (), *(in_refs + orig_out_refs + scratch_refs)
      )
      return ()

    all_in_avals = [v.aval for v in jaxpr.invars]
    in_avals_trace, orig_out_avals_trace, scratch_avals_trace = util.split_list(
        all_in_avals, [num_in, num_out_orig]
    )
    new_out_avals_trace = [avals_in[i] for i in write_indices]
    tracing_avals = (
        in_avals_trace
        + orig_out_avals_trace
        + new_out_avals_trace
        + scratch_avals_trace
    )

    debug_info = api_util.debug_info(
        "mpmd_map_discharge", new_body, tracing_avals, {}
    )
    wrapped_fun = lu.wrap_init(new_body, debug_info=debug_info)
    new_jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(wrapped_fun, tracing_avals)
    return new_jaxpr

  with (
      jax_core.extend_axis_env_nd(super_mesh_shape.items()),
      config._check_vma(False),
  ):
    for jaxpr in jaxprs:
      new_jaxprs.append(_rewrite_to_include_new_outputs(jaxpr))

  new_out_avals = [avals_in[i].inner_aval for i in write_indices]
  updated_out_avals = list(avals_out) + new_out_avals

  new_aliases = dict(input_output_aliases)
  for out_idx, in_idx in enumerate(write_indices):
    new_aliases[in_idx] = num_out_orig + out_idx
  if debug:
    print("discharged mpmd_map")
    for mesh, jaxpr in zip(meshes, new_jaxprs):
      print(f"mesh: {mesh}")
      print(f"new_jaxpr: {jaxpr}")
    print(f"new_aliases: {new_aliases}")

  res = mpmd_map_p.bind(
      *args,
      jaxprs=tuple(new_jaxprs),
      meshes=meshes,
      input_output_aliases=FrozenDict(new_aliases),
      out_avals=tuple(updated_out_avals),
      debug=debug,
      interpret=interpret,
      compiler_params=compiler_params,
      cost_estimate=cost_estimate,
      metadata=metadata,
      name=name,
      external_meshes=external_meshes,
  )

  # Split the results into original outputs and updated refs.
  ans, updated_refs = util.split_list(res, [num_out_orig])
  new_invals = [None] * len(avals_in)
  for out_idx, in_idx in enumerate(write_indices):
    new_invals[in_idx] = updated_refs[out_idx]

  return new_invals, ans


state_discharge.register_discharge_rule(mpmd_map_p)(_mpmd_map_discharge_rule)


def _mpmd_map_is_high(*args, jaxprs, **params):
  del args, params
  return any(jaxpr.is_high for jaxpr in jaxprs)
mpmd_map_p.is_high = _mpmd_map_is_high


def _mpmd_map_to_lojax(
    *hi_args,
    meshes,
    jaxprs,
    external_meshes,
    out_avals,
    input_output_aliases,
    compiler_params,
    interpret,
    debug,
    cost_estimate,
    metadata,
    name,
    **params,
):
  in_avals = [jax_core.typeof(a) for a in hi_args]
  if any(aval.has_qdd for aval in in_avals):
    raise NotImplementedError("mpmd_map does not support QDD for inputs")
  if any(aval.has_qdd for aval in out_avals):
    raise NotImplementedError("mpmd_map does not support QDD for outputs")

  lo_args = [
      lo_val
      for aval, x in zip(in_avals, hi_args)
      for lo_val in (aval.read_loval(x) if aval.has_qdd else aval.lower_val(x))
  ]

  lo_out_avals = [lo_aval for aval in out_avals for lo_aval in aval.lo_ty()]

  super_mesh_shape = get_super_mesh_shape(it.chain(meshes, external_meshes))
  lo_jaxprs = []
  with (
      jax_core.extend_axis_env_nd(super_mesh_shape.items()),
      config._check_vma(False),
  ):
    for jaxpr in jaxprs:
      closed_jaxpr = jax_core.ClosedJaxpr(jaxpr, ())
      closed_lo_jaxpr = pe.lower_jaxpr2(closed_jaxpr)
      assert not closed_lo_jaxpr.consts
      lo_jaxprs.append(closed_lo_jaxpr.jaxpr)

  input_index_mapping = pallas_call._get_index_mapping(in_avals)
  output_index_mapping = pallas_call._get_index_mapping(out_avals)
  new_input_output_aliases = {}
  for i, o in input_output_aliases.items():
    assert i in input_index_mapping
    assert o in output_index_mapping
    for i_lo, o_lo in zip(input_index_mapping[i], output_index_mapping[o]):
      new_input_output_aliases[i_lo] = o_lo

  lo_outs = mpmd_map_p.bind(
      *lo_args,
      meshes=meshes,
      jaxprs=tuple(lo_jaxprs),
      external_meshes=external_meshes,
      out_avals=tuple(lo_out_avals),
      input_output_aliases=FrozenDict(new_input_output_aliases),
      compiler_params=compiler_params,
      interpret=interpret,
      debug=debug,
      cost_estimate=cost_estimate,
      metadata=metadata,
      name=name,
      **params,
  )
  return pe.raise_lo_outs(out_avals, lo_outs)


mpmd_map_p.to_lojax = _mpmd_map_to_lojax


def _mpmd_map_tpu_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxprs,
    meshes,
    input_output_aliases,
    debug,
    interpret,
    compiler_params,
    cost_estimate,
    out_avals,
    metadata,
    name,
    external_meshes,
):
  try:
    from jax._src.pallas.mosaic import pallas_call_registration
  except ImportError:
    raise pallas_call._unsupported_lowering_error("tpu")
  return pallas_call_registration.mpmd_map_tpu_lowering_rule(
      ctx,
      *in_nodes,
      jaxprs=jaxprs,
      meshes=meshes,
      input_output_aliases=input_output_aliases,
      debug=debug,
      interpret=interpret,
      compiler_params=compiler_params,
      cost_estimate=cost_estimate,
      out_avals=out_avals,
      metadata=metadata,
      name=name,
      external_meshes=external_meshes,
  )


def _mpmd_map_fallback_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    meshes,
    jaxprs,
    out_avals,
    input_output_aliases,
    compiler_params,
    interpret,
    debug,
    cost_estimate,
    metadata,
    name,
    external_meshes,
):
  if len(jaxprs) != 1:
    raise NotImplementedError(
        "Lowering multiple mesh/function pairs is not currently supported"
    )
  if external_meshes:
    raise NotImplementedError(
        "External meshes are not currently supported in fallback lowering"
    )
  [jaxpr] = jaxprs
  [mesh] = meshes

  if hasattr(mesh, "dimension_semantics"):
    compiler_params = compiler_params.replace(
        dimension_semantics=mesh.dimension_semantics
    )
  if hasattr(mesh, "core_type"):
    compiler_params = compiler_params.replace(kernel_type=mesh.core_type)

  num_scratch = len(jaxpr.invars) - len(in_nodes) - len(out_avals)
  scratch_avals = (
      [v.aval for v in jaxpr.invars[-num_scratch:]] if num_scratch > 0 else []
  )
  scratch_types = tuple(
      pallas_core.MemoryRef(v.inner_aval, v.memory_space) for v in scratch_avals
  )
  grid_spec = pallas_core.GridSpec(
      grid=tuple(mesh.shape.items()),
      in_specs=tuple(
          pallas_core.BlockSpec(
              memory_space=aval.memory_space
              if isinstance(aval, jax_core.ShapedArray)
              and not isinstance(aval.memory_space, jax_core.MemorySpace)
              else mesh.default_memory_space,
          )
          for aval in ctx.avals_in
      ),
      out_specs=tuple(
          pallas_core.BlockSpec(
              memory_space=aval.memory_space
              if isinstance(aval, jax_core.ShapedArray)
              and not isinstance(aval.memory_space, jax_core.MemorySpace)
              else mesh.default_memory_space,
          )
          for aval in out_avals
      ),
      scratch_shapes=scratch_types,
  )

  in_tree = tree_util.tree_structure(in_nodes)
  out_tree = tree_util.tree_structure(out_avals)

  in_origins = [f"arg{i}" for i in range(len(in_nodes))]
  out_origins = [f"out{i}" for i in range(len(out_avals))]

  _, grid_mapping = pallas_core.get_grid_mapping(
      grid_spec,
      [
          v.aval.inner_aval if isinstance(v.aval, state.AbstractRef) else v.aval
          for v in jaxpr.invars[: len(in_nodes)]
      ],
      in_tree,
      in_origins,
      [
          v.aval.inner_aval if isinstance(v.aval, state.AbstractRef) else v.aval
          for v in jaxpr.invars[len(in_nodes) : len(in_nodes) + len(out_avals)]
      ],
      out_tree,
      out_origins,
      debug=debug,
  )

  return pallas_call._pallas_call_lowering(
      ctx,
      *in_nodes,
      jaxpr=jaxpr,
      grid_mapping=grid_mapping,
      mesh=mesh,
      input_output_aliases=tuple(input_output_aliases.items()),
      debug=debug,
      interpret=interpret,
      compiler_params=compiler_params,
      cost_estimate=cost_estimate,
      out_avals=out_avals,
      metadata=metadata,
      name=name,
  )


@functools.partial(mlir.register_lowering, mpmd_map_p)
def _mpmd_map_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, **params):
  return mlir.lower_per_platform(
      ctx,
      "mpmd_map",
      dict(
          cpu=_mpmd_map_fallback_lowering,
          tpu=_mpmd_map_tpu_lowering,
          cuda=_mpmd_map_fallback_lowering,
          rocm=_mpmd_map_fallback_lowering,
      ),
      None,  # default_rule
      jax_core.no_effects,
      *in_nodes,
      **params,
  )


def mpmd_map(
    meshes_and_fns: Sequence[tuple[pallas_core.Mesh, Callable[..., None]]],
    /,
    out_types: tree_util.PyTree = (),
    *,
    scratch_types: pallas_core.ScratchShapeTree = (),
    compiler_params: Any | None = None,
    interpret: bool | Any = False,
    debug: bool = False,
    cost_estimate: pallas_core.CostEstimate | None = None,
    name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[..., _T]:
  interpret = (
      config.pallas_tpu_interpret_mode_context_manager.value or interpret)
  return _mpmd_map(
      meshes_and_fns,
      out_types,
      input_output_aliases={},
      scratch_types=scratch_types,
      compiler_params=compiler_params,
      interpret=interpret,
      debug=debug,
      cost_estimate=cost_estimate,
      name=name,
      metadata=metadata,
  )


def _aval_to_ref_aval(
    aval: Any,
    meshes: Sequence[pallas_core.Mesh],
) -> state.AbstractRef:
  match aval:
    case state.AbstractRef():
      return aval
    case jax_core.ShapedArray(memory_space=memory_space):
      if memory_space == jax_core.MemorySpace.Device:
        defaults = {mesh.default_memory_space for mesh in meshes}
        if len(defaults) != 1:
          raise ValueError(
              "Multiple meshes with different default memory spaces are not"
              " supported."
          )
        memory_space = list(defaults)[0]
      return state.AbstractRef(aval, memory_space=memory_space)
    case jax_core.AbstractValue():
      return state.AbstractRef(aval, memory_space=None)
    case _ if hasattr(aval, "get_ref_aval"):
      ref_aval = aval.get_ref_aval()
      assert isinstance(ref_aval, state.AbstractRef)
      return ref_aval
    case _:
      raise ValueError(f"Unsupported abstract value type: {type(aval), aval}")


def _error_if_non_ref_consts(consts, debug_info):
  consts_avals = [
      aval
      for c in consts
      if not isinstance(aval := jax_core.typeof(c), state.AbstractRef)
  ]
  non_scalar_consts_avals = [
      aval
      for aval in consts_avals
      if not (isinstance(aval, jax_core.ShapedArray) and not aval.shape)
  ]
  if non_scalar_consts_avals:
    ctx = jax_core.JaxprPpContext()
    pp_consts_avals = ", ".join(
        jax_core.pp_aval(aval, ctx) for aval in non_scalar_consts_avals
    )
    raise ValueError(
        "The kernel function in the mpmd_map"
        f" {debug_info.func_src_info} captures non-Ref constants"
        f" [{pp_consts_avals}]. You should pass them as inputs."
    )


def _get_unique_consts(
    consts: Sequence[Sequence[Any]],
) -> tuple[list[Array], set[int]]:
  unique_consts = []
  unique_const_ids = set()
  for cs in consts:
    for c in cs:
      if id(c) not in unique_const_ids:
        unique_consts.append(c)
        unique_const_ids.add(id(c))
  return unique_consts, unique_const_ids


def _dedup_consts_and_unify_jaxpr_signatures(
    jaxprs: Sequence[jax_core.Jaxpr],
    consts_per_fn: Sequence[Sequence[Any]],
    flat_args: Sequence[Any],
    unflat_in_avals: Sequence[jax_core.AbstractValue],
    unflat_out_avals: Sequence[jax_core.AbstractValue],
    flat_kernel_avals: Sequence[jax_core.AbstractValue],
    super_mesh_shape: Mapping[str, int],
) -> tuple[list[jax_core.Jaxpr], list[Array]]:
  # Example:
  #   c1, c2, c3 are closed-over refs.
  #   fn1 closes over [c1, c2] -> traced jaxpr1 has constvars for [c1, c2]
  #   fn2 closes over [c2, c3] -> traced jaxpr2 has constvars for [c2, c3]
  #
  #   `_dedup_consts_and_unify_jaxpr_signatures` will:
  #     1. Deduplicate constants to `unique_consts` = [c1, c2, c3].
  #     2. Rewrite jaxprs to take all `unique_consts` as explicit inputs instead
  #        of constvars:
  #        new_jaxpr1: (in_args, c1, c2, c3, out_args, scratch_args) -> ()
  #        new_jaxpr2: (in_args, c1, c2, c3, out_args, scratch_args) -> ()
  #     3. Return the new jaxprs (with empty constvars) and `unique_consts`.

  unique_consts, const_ids = _get_unique_consts(consts_per_fn)

  arg_ids = {id(arg) for arg in flat_args}
  if any(const_id in arg_ids for const_id in const_ids):
    raise NotImplementedError(
        "Closed-over ref aliases with a passed-in ref is not supported."
    )

  unique_const_avals = [jax_core.typeof(c) for c in unique_consts]

  num_inputs = len(tree_util.tree_leaves(unflat_in_avals))
  num_outputs = len(tree_util.tree_leaves(unflat_out_avals))

  in_avals_flat, out_avals_flat, scratch_avals_flat = util.split_list(
      flat_kernel_avals, [num_inputs, num_outputs]
  )

  def make_rewritten_body(original_jaxpr, original_consts):
    def _rewritten_body(*args):
      in_args, unique_const_args, out_args, scratch_args = util.split_list(
          args, [num_inputs, len(unique_consts), num_outputs]
      )

      # Extract only the consts used by this jaxpr.
      c_map = {
          id(uc): arg for uc, arg in zip(unique_consts, unique_const_args)
      }
      mapped_consts = [c_map[id(c)] for c in original_consts]

      eval_args = in_args + out_args + scratch_args
      jax_core.eval_jaxpr(original_jaxpr, mapped_consts, *eval_args)
      return []

    return _rewritten_body

  new_jaxprs = []
  tracing_avals = (
      in_avals_flat
      + unique_const_avals
      + out_avals_flat
      + scratch_avals_flat
  )
  for jaxpr, consts in zip(jaxprs, consts_per_fn):
    debug_info = api_util.debug_info(
        "mpmd_map_closed_over",
        make_rewritten_body(jaxpr, consts),
        tracing_avals,
        {},
    )
    wrapped_fun = lu.wrap_init(
        make_rewritten_body(jaxpr, consts), debug_info=debug_info
    )
    with (jax_core.extend_axis_env_nd(super_mesh_shape.items()),
          config._check_vma(False)):
      new_jaxpr, _, new_consts = pe.trace_to_jaxpr_dynamic(
          wrapped_fun, tracing_avals
      )
    assert not new_consts
    new_jaxprs.append(new_jaxpr)
  return new_jaxprs, unique_consts


def get_super_mesh_shape(
    meshes: Iterable[pallas_core.Mesh],
) -> Mapping[str, int]:
  super_mesh_shape = {}
  for mesh in meshes:
    for k, v in mesh.shape.items():
      # An extra check since `check_is_compatible_with` should catch it.
      assert (
          k not in super_mesh_shape or super_mesh_shape[k] == v
      ), f"Conflicting size for axis {k}"
      super_mesh_shape[k] = v
  return super_mesh_shape


def _mpmd_map(
    meshes_and_fns: Sequence[tuple[pallas_core.Mesh, Callable[..., None]]],
    /,
    out_types: tree_util.PyTree = (),
    *,
    input_output_aliases: Mapping[int, int] = {},
    scratch_types: pallas_core.ScratchShapeTree = (),
    compiler_params: Any | None = None,
    interpret: bool | Any = False,
    debug: bool = False,
    cost_estimate: pallas_core.CostEstimate | None = None,
    name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[..., _T]:
  """Like ``pallas_call``, but MPMD and without pipelining."""
  if not meshes_and_fns:
    raise ValueError("At least one mesh/function pair is required")

  is_output_sequence = isinstance(out_types, Sequence)

  flat_out_types_with_paths, out_tree = tree_util.tree_flatten_with_path(
      out_types
  )
  out_paths, flat_out_types = util.unzip2(flat_out_types_with_paths)
  # TODO(sharadmv): Use out_paths for debugging info.
  del out_paths
  flat_out_avals = tuple(
      map(pallas_core._convert_out_shape_to_aval, flat_out_types)
  )

  def wrapper(*args):
    flat_args_with_paths, in_tree = tree_util.tree_flatten_with_path(args)
    in_paths, flat_args = util.unzip2(flat_args_with_paths)
    del in_paths

    seen_ref_ids = set()
    for arg in flat_args:
      if isinstance(arg, jax_core.Ref):
        if id(arg) in seen_ref_ids:
          raise NotImplementedError(
              "Cannot pass the same ref into a mpmd map multiple times"
          )
        seen_ref_ids.add(id(arg))
    # TODO(sharadmv): Use in_paths for debugging info.
    flat_avals = tuple(map(jax_core.typeof, flat_args))

    external_meshes = []
    meshes = tuple(mesh for mesh, _ in meshes_and_fns)

    flat_scratch_types, scratch_tree = tree_util.tree_flatten(scratch_types)
    if len(meshes_and_fns) > 1:
      # TODO(rdyro): For MPMD with more than one mesh, come up with a better
      # solution for how to enforce core_type presence in scratch_shape.
      # TODO(rdyro): Check if we need to have a similar check for in-kernel
      # allocations (e.g., run_scoped, empty_ref) or can we assume the
      # core_type is inherited from the caller (we then need the core_type in
      # the caller context during tracing).
      # TODO(rdyro): Also check inputs and outputs for core type.
      for scratch_type in flat_scratch_types:
        from jax._src.pallas.mosaic import core as tpu_core

        if not isinstance(
            scratch_type.memory_space, pallas_core.CoreMemorySpace
        ) and scratch_type.memory_space not in (
            tpu_core.MemorySpace.HBM,
            tpu_core.MemorySpace.VMEM_SHARED,
        ):
          raise NotImplementedError(
              "MPMD map with more than one mesh requires scratch_type to have"
              f" a `core_type` specified, but {scratch_type=} is missing it."
          )

    # Kernels may have Refs that belong to external meshes (usually for
    # async kernels). For example, the SC ScalarSubcore may have a Reference
    # to a TC semaphore that it is signaling. There is no explicit TC mesh as
    # part of the user-provided meshes, and are instead snuck in via the aval.
    for aval in [*flat_avals, *flat_out_avals, *flat_scratch_types]:
      if (
          isinstance(aval, jax_core.ShapedArray)
          and isinstance(aval.memory_space, pallas_core.CoreMemorySpace)
          and aval.memory_space.mesh not in it.chain(meshes, external_meshes)
      ):
        external_meshes.append(aval.memory_space.mesh)

    all_meshes = [*meshes, *external_meshes]
    # Check that meshes are compatible with each other (e.g, have a consistent
    # core axis name in the sparsecore).
    for i, mesh in enumerate(all_meshes):
      for other_mesh in list(all_meshes)[i + 1 :]:
        mesh.check_is_compatible_with(other_mesh)

    super_mesh_shape = get_super_mesh_shape(all_meshes)
    unflat_in_avals = in_tree.unflatten(flat_avals)
    unflat_out_avals = out_tree.unflatten(flat_out_avals)
    unflat_scratch_types = scratch_tree.unflatten(flat_scratch_types)
    kernel_arg_avals = list(unflat_in_avals)
    if is_output_sequence:
      kernel_arg_avals.extend(unflat_out_avals)
    else:
      kernel_arg_avals.append(unflat_out_avals)
    if isinstance(unflat_scratch_types, Mapping):
      kernel_kwarg_avals = unflat_scratch_types
    else:
      kernel_arg_avals.extend(unflat_scratch_types)
      kernel_kwarg_avals = {}

    unflat_kernel_avals = tree_util.tree_map(
        functools.partial(_aval_to_ref_aval, meshes=meshes),
        (kernel_arg_avals, kernel_kwarg_avals),
    )
    flat_kernel_avals, kernel_aval_tree = tree_util.tree_flatten(
        unflat_kernel_avals
    )

    jaxprs: list[jax_core.Jaxpr] = []
    consts_per_fn = []
    for _, fn in meshes_and_fns:
      debug_info = api_util.debug_info("mpmd_map", fn, flat_kernel_avals, {})
      if name is not None:
        debug_info = debug_info.replace_func_name(name)
      flat_fun, out_tree_thunk = api_util.flatten_fun(
          lu.wrap_init(fn, debug_info=debug_info), kernel_aval_tree
      )
      with (jax_core.extend_axis_env_nd(super_mesh_shape.items()),
            config._check_vma(False)):
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
            flat_fun, flat_kernel_avals
        )
      fun_out_tree = out_tree_thunk()
      if fun_out_tree != tree_util.tree_structure(None):
        raise ValueError(
            f"The kernel function in mpmd_map {debug_info.func_src_info}"
            f" should return None. It returns a PyTree: {fun_out_tree}."
        )
      if consts:
        _error_if_non_ref_consts(consts, debug_info)
      jaxprs.append(jaxpr)
      consts_per_fn.append(consts)

    if any(consts_per_fn):
      # If we close over any constants in the kernel functions, we need to
      # deduplicate them and then unify the jaxpr signatures.
      jaxprs, consts = _dedup_consts_and_unify_jaxpr_signatures(
          jaxprs, consts_per_fn, flat_args, unflat_in_avals, unflat_out_avals,
          flat_kernel_avals, super_mesh_shape
      )
    else:
      consts: list[Array] = []

    if debug:
      for mesh, jaxpr in zip(meshes, jaxprs):
        print(f"jaxpr for {mesh.core_type}")
        print(jaxpr)

    # TODO(slebedev): The named scope should not be necessary here.
    ctx = (
        api.named_scope(name) if name is not None else contextlib.nullcontext()
    )
    with ctx:
      flat_outs = mpmd_map_p.bind(
          *flat_args,
          *consts,
          meshes=tuple(meshes),
          jaxprs=tuple(jaxprs),
          external_meshes=tuple(external_meshes),
          out_avals=flat_out_avals,
          input_output_aliases=FrozenDict(input_output_aliases),
          compiler_params=compiler_params,
          interpret=interpret,
          debug=debug,
          cost_estimate=cost_estimate,
          metadata=FrozenDict(metadata) if metadata is not None else None,
          name=name,
      )
    return out_tree.unflatten(flat_outs)

  return cast(Callable[..., _T], wrapper)
