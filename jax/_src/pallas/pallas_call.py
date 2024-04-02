# Copyright 2023 The JAX Authors.
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

"""Module for calling pallas functions from JAX."""
from __future__ import annotations

import itertools
from functools import partial
from functools import reduce

from typing import Any, Callable
from collections.abc import Sequence

import jax
from jax import api_util
from jax import tree_util
from jax import lax
from jax._src import state
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import mlir
from jax._src.interpreters import xla
from jax._src import ad_util
from jax._src import core as jax_core
from jax._src.state import primitives as sp
from jax._src import linear_util as lu
from jax._src.state import discharge as state_discharge
from jax._src.util import (
    split_list, safe_map, safe_zip, weakref_lru_cache,
    tuple_insert, partition_list, merge_lists)
import jax.numpy as jnp
import numpy as np

from jax._src.pallas import core as pallas_core

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Grid = pallas_core.Grid
BlockSpec = pallas_core.BlockSpec
GridSpec = pallas_core.GridSpec
BlockMapping = pallas_core.BlockMapping
GridMapping = pallas_core.GridMapping
NoBlockSpec = pallas_core.NoBlockSpec
no_block_spec = pallas_core.no_block_spec

pallas_call_p = jax_core.Primitive('pallas_call')
pallas_call_p.multiple_results = True

def _maybe_dynamic_slice(start_idx, block_shape, value, is_indexing):
  if start_idx is None:
    assert is_indexing is None
    return value
  assert is_indexing is not None
  start_idx = tuple(jnp.asarray(s, dtype=jnp.int32) for s in start_idx)
  output = lax.dynamic_slice(value, start_idx, slice_sizes=block_shape)
  squeeze_dims = tuple(np.arange(len(is_indexing))[np.array(is_indexing,
                                                            dtype=np.bool_)])
  return lax.squeeze(output, squeeze_dims)

def _maybe_dynamic_update_slice(start_idx, block_shape, value, update,
                                is_indexing):
  if start_idx is None:
    assert is_indexing is None
    return update
  assert is_indexing is not None
  start_idx = tuple(jnp.asarray(s, dtype=jnp.int32) for s in start_idx)
  broadcast_dims = tuple(i for i, b in enumerate(is_indexing)
                         if not b)
  update = lax.broadcast_in_dim(update, block_shape, broadcast_dims)
  assert update.shape == block_shape
  return lax.dynamic_update_slice(value, update, start_idx)

def _uninitialized_value(shape, dtype):
  if jnp.issubdtype(dtype, jnp.floating):
    return jnp.full(shape, jnp.nan, dtype)
  elif jnp.issubdtype(dtype, jnp.integer):
    return jnp.full(shape, jnp.iinfo(dtype).min, dtype)
  raise NotImplementedError(dtype)

def _get_next_indices(grid, indices):
  next_indices = []
  carry = True
  for dim_size, index in reversed(list(zip(grid, indices))):
    i = jnp.where(carry, index + 1, index)
    carry = dim_size == i
    next_indices.append(jnp.where(carry, 0, i))
  return tuple(reversed(next_indices))

def _pallas_call_impl(*args, jaxpr, name, out_shapes, which_linear,
                      interpret, debug: bool,
                      in_shapes,
                      input_output_aliases: tuple[tuple[int, int], ...],
                      grid_mapping: GridMapping,
                      compiler_params: Any):
  dynamic_grid_args, args = split_list(  # type: ignore
      args, [grid_mapping.num_dynamic_grid_bounds]
  )
  if interpret:
    # If we're in interpreter mode, we *scan* over the grid and eval the
    # discharged jaxpr. This should reproduce exactly what compiling to Triton
    # will do.
    dynamic_grid_args_iter = iter(dynamic_grid_args)
    grid = tuple(
        a if a is not None else next(dynamic_grid_args_iter)
        for a in grid_mapping.grid
    )
    assert next(dynamic_grid_args_iter, None) is None
    discharged_jaxpr, consts = state_discharge.discharge_state(jaxpr, ())
    if debug:
      print(discharged_jaxpr)
    oi_map = {v: k for k, v in input_output_aliases}
    out = []
    for i, out_shape in enumerate(out_shapes):
      if i in oi_map:
        out.append(args[oi_map[i]])
      else:
        # TODO(sharadmv): use unitialized values for outputs
        out.append(jnp.zeros(out_shape.shape, out_shape.dtype))
    scalars, args = split_list(args, [grid_mapping.num_index_operands])  # type: ignore
    # invars: [*scalar_prefetch, *inputs, *outputs, *scratch]
    num_invars = len(jaxpr.invars)
    num_inputs_outputs = (
        num_invars
        - grid_mapping.num_index_operands
        - grid_mapping.num_scratch_operands
    )
    _, _, scratch_invars = split_list(
        jaxpr.invars, [grid_mapping.num_index_operands, num_inputs_outputs]
    )
    scratch_avals = [v.aval for v in scratch_invars]
    if not all(
        hasattr(a, "shape") and hasattr(a, "dtype") for a in scratch_avals
    ):
      raise NotImplementedError(f"Cannot initialize scratch: {scratch_avals}")
    scratch_values = [_uninitialized_value(a.shape, a.dtype)
                      for a in scratch_avals]

    carry = []
    for x, bm in zip(itertools.chain(args, out), grid_mapping.block_mappings):
      if bm is not None and isinstance(bm.indexing_mode, pallas_core.Unblocked):
        padding = bm.indexing_mode.padding
        if padding is not None and any(p != (0, 0) for p in padding):
          if input_output_aliases:
            raise NotImplementedError("Padding with aliasing not supported.")
          x = lax.pad(x, jnp.zeros((), x.dtype), [(*p, 0) for p in padding])
      carry.append(x)
    carry.extend(scratch_values)
    num_inout = len(args) + len(out)
    grid_start_indices = (jnp.int32(0),) * len(grid)
    if grid:
      num_iterations = reduce(jnp.multiply, grid)
    else:
      # Base case is always one iteration when grid is ()
      num_iterations = 1
    def cond(carry):
      i, *_ = carry
      return i < num_iterations
    def body(carry):
      i, loop_idx, *carry = carry
      local_grid_env = tuple(
          (idx, b)
          for dim, (idx, b) in enumerate(zip(loop_idx, grid))
          if dim not in grid_mapping.mapped_dims
      )
      carry, scratch = split_list(carry, [num_inout])
      with pallas_core.grid_env(local_grid_env):
        start_indices = [
            None if bm is None else bm.compute_start_indices(loop_idx, *scalars)
            for bm in grid_mapping.block_mappings]
      block_shapes_without_mapped_dims = [
          None if block_mapping is None else block_mapping.block_shape
          for block_mapping in grid_mapping.block_mappings
      ]
      is_indexing_dim = [
          None if bm is None else tuple(b is pallas_core.mapped for b in bm)
          for bm in block_shapes_without_mapped_dims
      ]
      block_shapes = [
          None if bm is None else tuple(1 if i else b for i, b in zip(iid, bm))
          for iid, bm in zip(is_indexing_dim, block_shapes_without_mapped_dims)
      ]
      blocks = map(_maybe_dynamic_slice, start_indices, block_shapes, carry,
                   is_indexing_dim)
      with pallas_core.grid_env(local_grid_env):
        assert len(discharged_jaxpr.invars) == len(scalars) + len(blocks) + len(
            scratch_values
        ), (
            len(discharged_jaxpr.invars),
            len(scalars),
            len(blocks),
            len(scratch_values),
        )
        blocks = jax.core.eval_jaxpr(discharged_jaxpr, consts, *scalars,
                                     *blocks, *scratch)
      blocks = blocks[grid_mapping.num_index_operands:]
      blocks, out_scratch = split_list(blocks, [num_inout])
      carry = map(_maybe_dynamic_update_slice, start_indices, block_shapes,
                  carry, blocks, is_indexing_dim)
      return (i + 1, _get_next_indices(grid, loop_idx), *carry, *out_scratch)
    (_, _, *carry) = lax.while_loop(
        cond, body, (jnp.int32(0), grid_start_indices, *carry)
    )
    _, out, _ = split_list(carry, [len(args), len(out)])
    assert len(grid_mapping.block_mappings) == len(args) + len(out)
    out_block_mappings = grid_mapping.block_mappings[len(args):]
    out_nopad = []
    for o, bm in zip(out, out_block_mappings):
      if bm is not None and isinstance(bm.indexing_mode, pallas_core.Unblocked):
        padding = bm.indexing_mode.padding
        if padding is not None and any(p != (0, 0) for p in padding):
          if input_output_aliases:
            raise NotImplementedError("Padding with aliasing not supported.")
          pad_low, pad_high = zip(*padding)
          limit_indices = [s - p for s, p in zip(out.shape, pad_high)]
          o = lax.slice(o, pad_low, limit_indices)
      out_nopad.append(o)
    return out_nopad
  return xla.apply_primitive(pallas_call_p, *args, jaxpr=jaxpr, name=name,
                             in_shapes=in_shapes,
                             out_shapes=out_shapes, which_linear=which_linear,
                             grid_mapping=grid_mapping, interpret=interpret,
                             debug=debug,
                             input_output_aliases=input_output_aliases,
                             compiler_params=compiler_params)
pallas_call_p.def_impl(_pallas_call_impl)

def _pallas_call_abstract_eval(*avals, out_shapes, **_):
  return map(lambda x: jax_core.ShapedArray(x.shape, x.dtype), out_shapes)
pallas_call_p.def_abstract_eval(_pallas_call_abstract_eval)

def _pallas_call_jvp_rule(primals, tangents, *, jaxpr, name, which_linear,
    input_output_aliases: tuple[tuple[int, int], ...],
    in_shapes, out_shapes, grid_mapping, debug, interpret, compiler_params: Any):
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError("interpret with dynamic grid bounds unsupported")
  if grid_mapping.num_index_operands:
    raise NotImplementedError
  if input_output_aliases:
    raise NotImplementedError("JVP with aliasing not supported.")
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  nonzero_tangents_with_outputs = nonzero_tangents + [True] * len(out_shapes)
  closed_jaxpr = jax_core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, nonzero_tangents_with_outputs, [])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  # `pallas_call` takes in inputs and returns outputs but its jaxpr *does not*.
  # `pallas_call` takes in a stateful jaxpr, meaning the jaxpr accepts input
  # `Ref`s that are read from followed by output `Ref`s that are written to.
  # This means that when we do `jvp_jaxpr` on the `jaxpr`, we get out a new
  # jaxpr that has tangents following primals. In order for this jaxpr to be
  # compatible w/ `pallas_call` (inputs then outputs), we need to shuffle around
  # the jaxpr's invars.
  primal_refs, primal_out_refs, tangent_refs, tangent_out_refs = split_list(
      jvp_jaxpr.invars, [len(primals), len(out_shapes), len(tangents)]
  )
  invars = (*primal_refs, *tangent_refs, *primal_out_refs, *tangent_out_refs)
  # TODO(sharadmv): Fix state effect tracking after invar switch.
  jvp_jaxpr = jvp_jaxpr.replace(invars=invars)
  if debug:
    print(jvp_jaxpr)
  in_bms, out_bms = split_list(grid_mapping.block_mappings, [len(primals)])
  jvp_bms = (*in_bms, *in_bms, *out_bms, *out_bms)
  out_flat = pallas_call_p.bind(
      *primals,
      *tangents,
      jaxpr=jvp_jaxpr,
      name=f"{name}_jvp",
      in_shapes=(*in_shapes, *in_shapes),
      out_shapes=(*out_shapes, *out_shapes),
      grid_mapping=grid_mapping.replace(block_mappings=jvp_bms),
      which_linear=which_linear + (True,) * len(tangents),
      interpret=interpret,
      debug=debug,
      input_output_aliases=(),
      compiler_params=compiler_params,
  )
  out_primals, out_tangents = split_list(out_flat, [len(out_flat) // 2])
  return out_primals, out_tangents
ad.primitive_jvps[pallas_call_p] = _pallas_call_jvp_rule

def _batch_block_mapping(grid: tuple[int, ...], aval: jax_core.ShapedArray,
                         dim: int | batching.NotMapped,
                         block_mapping: BlockMapping | None) -> BlockMapping:
  def _block_map_function(new_idx, *args):
    if block_mapping is None:
      indices = [0] * len(aval.shape)
    else:
      indices = jax_core.eval_jaxpr(block_mapping.index_map_jaxpr.jaxpr,
                                    block_mapping.index_map_jaxpr.consts,
                                    *args)
    if dim is not batching.not_mapped:
      indices.insert(dim, new_idx)
    return tuple(indices)
  i32_aval = jax_core.ShapedArray((), jnp.int32)
  if block_mapping is None:
    idx_avals = [i32_aval] * (len(grid) + 1)
  else:
    idx_avals = [i32_aval, *block_mapping.index_map_jaxpr.in_avals]
  block_mapping_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(_block_map_function), idx_avals)
  shape = aval.shape if block_mapping is None else block_mapping.block_shape
  if dim is batching.not_mapped:
    new_block_shape = shape
  else:
    new_block_shape = tuple_insert(shape, dim, pallas_core.mapped)
  jaxpr = jax_core.ClosedJaxpr(block_mapping_jaxpr, consts)
  if block_mapping is None:
    return BlockMapping(
        block_shape=new_block_shape,
        index_map_jaxpr=jaxpr,
        indexing_mode=pallas_core.blocked,
    )
  return block_mapping.replace(block_shape=new_block_shape,
                               index_map_jaxpr=jaxpr)

def _pallas_call_batching_rule(args, dims, *,
                               jaxpr: jax_core.Jaxpr,
                               name: str,
                               in_shapes: tuple[jax.ShapeDtypeStruct, ...],
                               out_shapes: tuple[jax.ShapeDtypeStruct, ...],
                               grid_mapping: GridMapping,
                               input_output_aliases: tuple[tuple[int, int], ...],
                               debug: bool,
                               interpret: bool,
                               which_linear: tuple[bool, ...],
                               compiler_params: Any):

  def _maybe_squeeze_out_bdim(
      x: jax.Array, bdim: int | batching.NotMapped
  ) -> jax.Array:
    if bdim is batching.not_mapped:
      return x
    return jnp.squeeze(x, axis=bdim)

  dynamic_grid_args, args = split_list(
      args, [grid_mapping.num_dynamic_grid_bounds]
  )
  dynamic_grid_dims, dims = split_list(
      dims, [grid_mapping.num_dynamic_grid_bounds]
  )
  if all(
      bdim is batching.not_mapped or arg.shape[bdim] == 1
      for arg, bdim in zip(dynamic_grid_args, dynamic_grid_dims)
  ):
    dynamic_grid_args = safe_map(
        _maybe_squeeze_out_bdim, dynamic_grid_args, dynamic_grid_dims)
  elif any(bdim is not batching.not_mapped for bdim in dynamic_grid_dims):
    raise NotImplementedError(
        f"Batched dynamic grid bounds unsupported: {dynamic_grid_dims}"
    )
  else:
    pass  # No dynamic grid dimensions
  del dynamic_grid_dims
  if grid_mapping.num_index_operands:
    scalar_args, args = split_list(args, [grid_mapping.num_index_operands])
    scalar_bdims, bdims = split_list(dims, [grid_mapping.num_index_operands])
    # Ordinarily, adding support for scalar prefetch in vmap would involve
    # modifying the block specs in a nontrivial way. However, if we are only
    # vmapping over 1-sized dimensions, we can just get rid of the dimensions
    # and pretend we were never vmapping over them at all.
    if all(
        bdim is batching.not_mapped or arg.shape[bdim] == 1
        for arg, bdim in zip(scalar_args, scalar_bdims)
    ):
      scalar_args = safe_map(_maybe_squeeze_out_bdim, scalar_args, scalar_bdims)
      scalar_bdims = [None] * len(scalar_args)
      args = (*scalar_args, *args)
      dims = (*scalar_bdims, *bdims)
    else:
      # TODO(sharadmv,apaszke): enable batching over prefetched scalar args
      raise NotImplementedError
  if not dims:
    raise NotImplementedError("vmapping pallas_call with no arguments.")
  axis_size, = {x.shape[d] for x, d in zip(args, dims)
                if d is not batching.not_mapped}
  block_mappings = grid_mapping.block_mappings
  avals = [v.aval for v in jaxpr.invars]
  # How should we pick output dimensions? This actually matters because XLA
  # can't optimize our pallas kernels, and this layout impacts performance. For
  # now, because `vmap` doesn't really offer a way of inferring good output
  # dimensions. For now, we just use 0.
  # TODO(sharadmv): explore inferring better output dimensions via a heuristic
  # TODO(sharadmv): explore a long term solution to output dim inference

  # When we have input/output aliasing, since the output will be mapped, we need
  # to make sure to broadcast the input across that dimension if it is not
  # mapped.
  dims_ = list(dims)
  args_ = list(args)
  for input_index, _ in input_output_aliases:
    dim = dims_[input_index]
    if dim is batching.not_mapped:
      dims_[input_index] = 0
      args_[input_index] = batching.broadcast(args_[input_index], axis_size, 0)
  args = tuple(args_)
  dims = tuple(dims_)

  all_dims = list(dims) + [0] * len(out_shapes)

  num_index_operands = grid_mapping.num_index_operands
  num_scratch_operands = grid_mapping.num_scratch_operands

  # Only add a batch dimension for the avals that actually have a grid mapping.
  # This excludes scalar prefetch inputs (the first in the list) and scratch
  # operands (the last in the list).
  avals_to_batch = avals[num_index_operands:(len(avals) - num_scratch_operands)]
  batched_block_mappings = map(
      partial(_batch_block_mapping, grid_mapping.grid),
      avals_to_batch,
      all_dims[num_index_operands:],
      block_mappings,
  )

  batched_in_shapes = tuple(
      jax.ShapeDtypeStruct(x.shape if dim is batching.not_mapped else
                           tuple_insert(x.shape, dim, axis_size),
                           x.dtype)
      for x, dim in zip(in_shapes, dims))
  batched_out_shapes = tuple(
      jax.ShapeDtypeStruct(tuple_insert(x.shape, 0, axis_size), x.dtype)
      for x in out_shapes)

  batched_grid_mapping = grid_mapping.replace(
      grid=(axis_size, *grid_mapping.grid),
      block_mappings=tuple(batched_block_mappings),
      mapped_dims=(0,) + tuple(a + 1 for a in grid_mapping.mapped_dims))
  out = pallas_call_p.bind(
      *dynamic_grid_args,
      *args,
      jaxpr=jaxpr,
      name=f"batched_{name}",
      in_shapes=batched_in_shapes,
      out_shapes=batched_out_shapes,
      which_linear=which_linear,
      grid_mapping=batched_grid_mapping,
      input_output_aliases=input_output_aliases,
      debug=debug,
      interpret=interpret,
      compiler_params=compiler_params,
  )
  return out, (0,) * len(out)
batching.primitive_batchers[pallas_call_p] = _pallas_call_batching_rule

def _hoist_consts_to_refs(jaxpr: jax_core.Jaxpr) -> jax_core.Jaxpr:
  all_const_avals = [var.aval for var in jaxpr.constvars]
  is_const_ref = [isinstance(var.aval, state.AbstractRef) for var in
                  jaxpr.constvars]
  const_avals, const_ref_avals = partition_list(is_const_ref, all_const_avals)
  const_avals = map(state.AbstractRef, const_avals)
  merged_const_avals = merge_lists(is_const_ref, const_avals, const_ref_avals)
  arg_avals = [var.aval for var in jaxpr.invars]
  in_avals = [*merged_const_avals, *arg_avals]
  num_consts = len(merged_const_avals)

  def _hoist(*consts_args):
    all_consts, args = split_list(consts_args, [num_consts])
    consts, const_refs = partition_list(is_const_ref, all_consts)
    # We immediately read the const values out of the `Ref`s.
    consts = map(lambda x: sp.ref_get(x, ()), consts)
    all_consts = merge_lists(is_const_ref, consts, const_refs)
    return jax_core.eval_jaxpr(jaxpr, all_consts, *args)
  hoisted_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(_hoist), in_avals)
  assert not consts, "All consts should have been converted to refs"
  return hoisted_jaxpr

@weakref_lru_cache
def _trace_to_jaxpr(fun: Callable, grid_spec: GridSpec, flat_in_avals,
                    flat_out_avals, in_tree, out_tree):
  avals, grid_mapping = grid_spec.get_grid_mapping(flat_in_avals, in_tree,
                                                   flat_out_avals, out_tree)
  jaxpr_flat_avals, jaxpr_in_tree = tree_util.tree_flatten(avals)
  wrapped_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(fun), jaxpr_in_tree)
  debug = pe.debug_info(fun, jaxpr_in_tree, out_tree_thunk, False, "pallas_call")
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(wrapped_fun, jaxpr_flat_avals, debug)
  jaxpr = _hoist_consts_to_refs(jaxpr)
  return grid_mapping, jaxpr, consts, out_tree_thunk()

def _extract_function_name(f: Callable, name: str | None) -> str:
  if name is None:
    name = f.__name__ if hasattr(f, "__name__") and f.__name__ else "func"
  return name


def _unsupported_lowering_error(platform: str) -> Exception:
  return ValueError(
      f"Cannot lower pallas_call on platform: {platform}. To use Pallas on GPU,"
      " install jaxlib GPU 0.4.24 or newer. To use Pallas on TPU, install"
      " jaxlib TPU and libtpu. See"
      " https://jax.readthedocs.io/en/latest/installation.html."
  )


def _pallas_call_lowering(
    ctx: mlir.LoweringRuleContext, *in_nodes, interpret: bool, **params
):
  if interpret:
    # If we are in interpret mode, we don't care what platform we are on.
    impl = partial(_pallas_call_impl, **params, interpret=True)
    return mlir.lower_fun(impl, multiple_results=True)(ctx, *in_nodes)

  try:
    [platform] = ctx.module_context.platforms
  except ValueError:
    raise ValueError(
        "Can only lower pallas_call on a single platform."
    ) from None

  if platform == "cpu":
    raise ValueError("Only interpret mode is supported on CPU backend.")
  elif platform == "cuda" or platform == "rocm":
    try:
      from jax._src.pallas.triton import pallas_call_registration  # type: ignore
    except ImportError:
      pass
    else:
      return pallas_call_registration.pallas_call_lowering(
          ctx, *in_nodes, interpret=interpret, **params
      )
  elif platform == "tpu":
    try:
      from jax._src.pallas.mosaic import pallas_call_registration  # type: ignore
    except ImportError:
      pass
    else:
      return pallas_call_registration.pallas_call_tpu_lowering_rule(
          ctx, *in_nodes, interpret=interpret, **params
      )

  raise _unsupported_lowering_error(platform)


mlir.register_lowering(pallas_call_p, _pallas_call_lowering)


def pallas_call(
    f: Callable[..., None],
    out_shape: Any,
    *,
    grid_spec: GridSpec | None = None,
    debug: bool = False,
    grid: Grid | None = None,
    in_specs: Sequence[BlockSpec | NoBlockSpec] | NoBlockSpec = no_block_spec,
    out_specs: BlockSpec | NoBlockSpec
    | Sequence[BlockSpec | NoBlockSpec] = no_block_spec,
    input_output_aliases: dict[int, int] = {},
    interpret: bool = False,
    name: str | None = None,
    compiler_params: dict[str, Any] | None = None,
    **compiler_params_: Any,
):
  name = _extract_function_name(f, name)
  if compiler_params is None:
    compiler_params = {}
  assert not (compiler_params and compiler_params_)
  if compiler_params_:
    compiler_params = compiler_params_
  if grid is not None and grid_spec is not None:
    raise ValueError("Cannot specify both grid and grid_spec at the same time.")
  if grid_spec is None:
    grid_spec = GridSpec(grid, in_specs, out_specs)
  grid_spec, dynamic_grid_bounds = grid_spec.unzip_dynamic_grid_bounds()
  if isinstance(out_shape, list):
    out_shape = tuple(out_shape)
  flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)
  flat_out_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype)
                     for x in flat_out_shapes]
  @jax.jit
  def wrapped(*args):
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_in_avals = tuple(jax_core.raise_to_shaped(jax_core.get_aval(a))
                          for a in flat_args)
    flat_out_avals = tuple(jax_core.ShapedArray(v.shape, v.dtype)
                           for v in flat_out_shapes)
    grid_mapping, jaxpr, consts, _ = _trace_to_jaxpr(
        f, grid_spec, flat_in_avals, flat_out_avals, in_tree,
        out_tree)
    which_linear = (False,) * len(flat_args)
    out_flat = pallas_call_p.bind(
        *dynamic_grid_bounds, *consts, *flat_args,
        jaxpr=jaxpr, name=name, which_linear=which_linear,
        in_shapes=tuple(jax.ShapeDtypeStruct(a.shape, a.dtype)
                        for a in flat_args),
        out_shapes=tuple(flat_out_shapes), debug=debug,
        interpret=interpret,
        grid_mapping=grid_mapping,
        input_output_aliases=tuple(input_output_aliases.items()),
        compiler_params=compiler_params)
    out = tree_util.tree_unflatten(out_tree, out_flat)
    return out
  return wrapped
