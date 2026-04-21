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

from collections.abc import Callable, Mapping, Sequence
import contextlib
import functools
from typing import cast, Any, ParamSpec, TypeVar

from jax._src import api
from jax._src import api_util
from jax._src import config
from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.frozen_dict import FrozenDict
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas import pallas_call


_P = ParamSpec("_P")
_T = TypeVar("_T")


mpmd_map_p = jax_core.Primitive("mpmd_map")
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
    **params,
):
  del params  # Unused.

  effs = {*pallas_core.get_interpret_effects(interpret)}
  if getattr(compiler_params, "has_side_effects", False):
    # TODO(slebedev): Fix internal breakages and add
    # ``jax_core.GenericEffect(pallas_call_p)`` here.
    effs = jax_core.no_effects

  for jaxpr in jaxprs:
    if not all(isinstance(aval, state.AbstractRef) for aval in jaxpr.in_avals):
      raise TypeError("MPMD kernels must only have Ref inputs")

  # TODO(slebedev): Handle pinned buffers as in ``pallas_call``.
  outin_aliases = {
      out_idx: in_idx for in_idx, out_idx in input_output_aliases.items()
  }
  out_avals = [
      in_avals[outin_aliases[out_idx]] if out_idx in outin_aliases else a
      for out_idx, a in enumerate(out_avals)
  ]
  # Make sure we don't return ShapedArray with pallas memory space to the
  # outside world.
  out_avals = tuple(a.update(memory_space=jax_core.MemorySpace.Device)
                    if isinstance(a, jax_core.ShapedArray) else a
                    for a in out_avals)
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


def _mpmd_map_tpu_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxprs,
    grid_mappings,
    meshes,
    input_output_aliases,
    debug,
    interpret,
    compiler_params,
    cost_estimate,
    out_avals,
    metadata,
    name,
):
  try:
    from jax._src.pallas.mosaic import pallas_call_registration
  except ImportError:
    raise pallas_call._unsupported_lowering_error("tpu")

  return pallas_call_registration.mpmd_map_tpu_lowering_rule(
      ctx,
      *in_nodes,
      jaxprs=jaxprs,
      grid_mappings=grid_mappings,
      meshes=meshes,
      input_output_aliases=input_output_aliases,
      debug=debug,
      interpret=interpret,
      compiler_params=compiler_params,
      cost_estimate=cost_estimate,
      out_avals=out_avals,
      metadata=metadata,
      name=name,
  )


def _mpmd_map_fallback_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    meshes,
    jaxprs,
    grid_mappings,
    out_avals,
    input_output_aliases,
    compiler_params,
    interpret,
    debug,
    cost_estimate,
    metadata,
    name,
):
  if len(jaxprs) != 1:
    raise NotImplementedError(
        "Lowering multiple mesh/function pairs is not currently supported"
    )
  [jaxpr] = jaxprs
  [mesh] = meshes
  [grid_mapping] = grid_mappings

  if hasattr(mesh, "dimension_semantics"):
    compiler_params = compiler_params.replace(
        dimension_semantics=mesh.dimension_semantics
    )
  if hasattr(mesh, "core_type"):
    compiler_params = compiler_params.replace(kernel_type=mesh.core_type)

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
    meshes_and_fns: Sequence[tuple[pallas_core.Mesh, Callable[_P, _T]]],
    /,
    out_types: tree_util.PyTree,
    *,
    scratch_types: pallas_core.ScratchShapeTree = (),
    compiler_params: Any | None = None,
    interpret: bool | Any = False,
    debug: bool = False,
    cost_estimate: pallas_core.CostEstimate | None = None,
    name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[_P, _T]:
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


def _mpmd_map(
    meshes_and_fns: Sequence[tuple[pallas_core.Mesh, Callable[_P, _T]]],
    /,
    out_types: tree_util.PyTree,
    *,
    input_output_aliases: Mapping[int, int] = {},
    scratch_types: pallas_core.ScratchShapeTree = (),
    compiler_params: Any | None = None,
    interpret: bool | Any = False,
    debug: bool = False,
    cost_estimate: pallas_core.CostEstimate | None = None,
    name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[_P, _T]:
  """Like ``pallas_call``, but MPMD and without pipelining."""
  if not meshes_and_fns:
    raise ValueError("At least one mesh/function pair is required")

  flat_out_types_with_paths, out_tree = tree_util.tree_flatten_with_path(
      out_types
  )
  out_paths, flat_out_types = util.unzip2(flat_out_types_with_paths)
  flat_out_avals = tuple(
      map(pallas_core._convert_out_shape_to_aval, flat_out_types)
  )
  out_origins = tuple(f"outputs{tree_util.keystr(p)}" for p in out_paths)

  @functools.partial(api.jit, inline=True)
  def wrapper(*args):
    flat_args_with_paths, in_tree = tree_util.tree_flatten_with_path(args)
    in_paths, flat_args = util.unzip2(flat_args_with_paths)
    flat_avals = tuple(map(jax_core.typeof, flat_args))
    in_origins = tuple(f"args{tree_util.keystr(p)}" for p in in_paths)

    # NOTE: ``grid_mapping`` are only needed for us to reuse the ``pallas_call``
    # lowering machinery.
    meshes = []
    jaxprs = []
    grid_mappings = []

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

    # Check that meshes are compatible with each other (e.g, have a consistent
    # core axis name in the sparsecore).
    for i, (mesh, _) in enumerate(meshes_and_fns):
      for other_mesh, _ in list(meshes_and_fns)[i+1:]:
        mesh.check_is_compatible_with(other_mesh)

    super_mesh_shape = {}
    for mesh, _ in meshes_and_fns:
      for k, v in mesh.shape.items():
        # An extra check since `check_is_compatible_with` should catch it.
        assert k not in super_mesh_shape or super_mesh_shape[k] == v, (
            f"Conflicting size for axis {k}"
        )
        super_mesh_shape[k] = v

    for mesh, fn in meshes_and_fns:
      grid_spec = pallas_core.GridSpec(
          grid=tuple(mesh.shape.items()),  # pyrefly: ignore[bad-argument-type]
          in_specs=in_tree.unflatten(
              pallas_core.BlockSpec(
                  memory_space=aval.memory_space
                  if isinstance(aval, jax_core.ShapedArray)
                  and not isinstance(aval.memory_space, jax_core.MemorySpace)
                  else mesh.default_memory_space,
              )
              for aval in flat_avals
          ),
          out_specs=out_tree.unflatten(
              pallas_core.BlockSpec(
                  memory_space=aval.memory_space
                  if isinstance(aval, jax_core.ShapedArray)
                  and not isinstance(aval.memory_space, jax_core.MemorySpace)
                  else mesh.default_memory_space,
              )
              for aval in flat_out_avals
          ),
          scratch_shapes=flat_scratch_types,
      )
      kernel_args, grid_mapping = pallas_core.get_grid_mapping(
          grid_spec,
          flat_avals,
          in_tree,
          in_origins,
          flat_out_avals,
          out_tree,
          out_origins,
      )
      kernel_args, scratch_args = util.split_list(
          kernel_args, [len(kernel_args) - scratch_tree.num_leaves])
      scratch_args = scratch_tree.unflatten(scratch_args)
      if isinstance(scratch_args, dict):
        kernel_args_kwargs = (kernel_args, scratch_args)
      else:
        kernel_args_kwargs = (kernel_args + list(scratch_args), {})
      flat_kernel_avals, kernel_in_tree = tree_util.tree_flatten(
          kernel_args_kwargs)
      debug_info = api_util.debug_info(
          "mpmd_map",
          fn,
          *kernel_args_kwargs,
      )
      if name is not None:
        debug_info = debug_info.replace_func_name(name)
      flat_fun, out_tree_thunk = api_util.flatten_fun(
          lu.wrap_init(fn, debug_info=debug_info), kernel_in_tree
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
        raise NotImplementedError("MPMD kernels cannot close over constants")

      meshes.append(mesh)
      jaxprs.append(jaxpr)
      grid_mappings.append(grid_mapping)

    # TODO(slebedev): The named scope should not be necessary here.
    ctx = (
        api.named_scope(name) if name is not None else contextlib.nullcontext()
    )
    with ctx:
      flat_outs = mpmd_map_p.bind(
          *flat_args,
          meshes=tuple(meshes),
          jaxprs=tuple(jaxprs),
          grid_mappings=tuple(grid_mappings),
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

  return cast(Callable[_P, _T], wrapper)
