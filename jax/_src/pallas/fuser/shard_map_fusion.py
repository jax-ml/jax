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

"""Fuses a function whose fusible is inside a ``shard_map``."""

from collections.abc import Sequence
import functools
import math
from typing import Any

import jax
from jax import lax
from jax._src import core as jax_core
from jax._src import shard_map
from jax._src import util
from jax._src.export import shape_poly
from jax._src.export import shape_poly_decision
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas.fuser import block_spec
from jax._src.pallas.fuser import jaxpr_fusion
import jax.numpy as jnp

del shape_poly_decision  # Imported to initialize shape_poly._bounds_decision.


def _block_spec_from_sharding(
    shape: tuple[int, ...],
    spec: jax.sharding.PartitionSpec,
    mesh: jax.sharding.Mesh,
    manual_axes: tuple[Any, ...],
) -> pallas_core.BlockSpec:
  """Returns the BlockSpec of one device's shard of an array.

  The "grid" is the ``manual_axes`` of ``mesh``: the index map takes one
  ``lax.axis_index`` per manual axis (followed by any scalar prefetch values,
  which it ignores) and returns the index of that device's shard of an array of
  shape ``shape`` sharded as ``spec``.
  """
  names = shard_map._spec_to_names(spec)
  dim_axes = [names.get(d, ()) for d in range(len(shape))]
  block_shape = jax.NamedSharding(mesh, spec).shard_shape(shape)

  def index_map(*axis_indices_and_scalar_prefetch):
    axis_indices = axis_indices_and_scalar_prefetch[: len(manual_axes)]
    axis_map = dict(zip(manual_axes, axis_indices, strict=True))

    def dim_index(axes):
      return functools.reduce(
          lambda idx, n: idx * mesh.shape[n] + axis_map[n], axes, 0
      )

    return tuple(dim_index(axes) for axes in dim_axes)

  return pallas_core.BlockSpec(block_shape=block_shape, index_map=index_map)


def _make_block_extractor(
    aval: jax_core.ShapedArray,
    shard_bs: pallas_core.BlockSpec,
    pulled_bs: pallas_core.BlockSpec | pallas_core.NoBlockSpec,
    mesh: jax.sharding.Mesh,
    manual_axes: tuple[Any, ...],
):
  """Verifies per-shard block containment and returns a local-shard slicer."""
  if not isinstance(pulled_bs, pallas_core.BlockSpec):
    return lambda x, *_: x
  assert shard_bs.block_shape is not None and shard_bs.index_map is not None
  shard_sizes = pallas_core._get_block_shape(
      pallas_core._canonicalize_block_shape(shard_bs.block_shape)
  )
  block_shape = pallas_core._canonicalize_block_shape(
      aval.shape if pulled_bs.block_shape is None else pulled_bs.block_shape
  )
  sizes = pallas_core._get_block_shape(block_shape)
  for d, (s_size, b, bd) in enumerate(
      zip(shard_sizes, sizes, block_shape, strict=True)
  ):
    if isinstance(bd, pallas_core.Element) and bd.padding != (0, 0):
      raise NotImplementedError(
          "Element block dim with non-zero padding is not supported in"
          f" shard_map fusion, got {bd}."
      )
    if isinstance(bd, (pallas_core.BoundedSlice, pallas_core.Indirect)):
      raise NotImplementedError(
          f"{type(bd).__name__} block dim is not supported in shard_map fusion,"
          f" got {bd}."
      )
    if b > s_size:
      raise NotImplementedError(
          f"Cannot fuse outer operation across shard_map: dimension {d} of"
          f" value with shape {aval.shape} requires block of size {b}, which"
          f" exceeds local shard size {s_size}."
      )
  if aval.ndim == 0:
    return lambda x, *_: x
  squeezed = tuple(
      d for d, bd in enumerate(block_shape) if block_spec._is_squeezed(bd)
  )
  pulled_index_map = (
      pallas_core.default_index_map(aval.ndim)
      if pulled_bs.index_map is None
      else pulled_bs.index_map
  )

  def _local_starts(axis_indices):
    assert shard_bs.index_map is not None
    return [
        pallas_core._get_start_index(p_idx, bd) - s_idx * s_size
        for s_size, bd, s_idx, p_idx in zip(
            shard_sizes,
            block_shape,
            shard_bs.index_map(*axis_indices),
            pulled_index_map(*axis_indices),
            strict=True,
        )
    ]

  active_axes = [
      (i, m) for i, a in enumerate(manual_axes) if (m := mesh.shape[a]) > 1
  ]
  sym_vars = (
      shape_poly.symbolic_shape(
          ", ".join(f"v{i}" for i, _ in active_axes),
          constraints=[f"v{i} <= {m}" for i, m in active_axes],
      )
      if active_axes
      else ()
  )
  sym_map = {
      i: v - 1 for (i, _), v in zip(active_axes, sym_vars, strict=True)
  }
  sym_axes = [sym_map.get(i, 0) for i in range(len(manual_axes))]
  with (
      jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), ())),
      jax_core.eval_context(),
  ):
    try:
      sym_starts, sym_err = _local_starts(sym_axes), None
    except (
        shape_poly.InconclusiveDimensionOperation,
        NotImplementedError,
        TypeError,
    ) as e:
      sym_starts, sym_err = [None] * aval.ndim, e

  static_starts: list[int | None] = [None] * aval.ndim
  for d, (sym, s_size, b) in enumerate(
      zip(sym_starts, shard_sizes, sizes, strict=True)
  ):
    err = sym_err
    if sym is None:
      lb, ub = -math.inf, math.inf
    else:
      try:
        lb, ub = shape_poly.symbolic_dim_bounds(sym)
      except (shape_poly.InconclusiveDimensionOperation, TypeError) as e:
        lb, ub, err = -math.inf, math.inf, e
    if not (math.isfinite(lb) and math.isfinite(ub)):
      if s_size < aval.shape[d]:
        raise NotImplementedError(
            f"Cannot fuse outer operation across shard_map: dimension {d} of"
            f" value with shape {aval.shape} has a block index map that cannot"
            f" be statically verified to lie within local shard [0:{s_size}]."
        ) from err
      continue
    lo, hi = int(lb), int(ub)
    if lo < 0 or hi + b > s_size:
      bad = lo if lo < 0 else hi
      raise NotImplementedError(
          f"Cannot fuse outer operation across shard_map: dimension {d} of"
          f" value with shape {aval.shape} requires local shard slice"
          f" [{bad}:{bad + b}], which lies outside local shard [0:{s_size}]."
      )
    if lo == hi:
      static_starts[d] = lo

  if all(s is not None for s in static_starts):
    int_starts = [s for s in static_starts if s is not None]
    if not any(int_starts) and sizes == shard_sizes and not squeezed:
      return lambda x, *_: x
    limits = [s + b for s, b in zip(int_starts, sizes)]
    need_slice = any(int_starts) or sizes != shard_sizes

    def _slice_static(x, *_):
      if need_slice:
        x = lax.slice(x, int_starts, limits)
      return lax.squeeze(x, squeezed) if squeezed else x

    return _slice_static

  def _slice_dynamic(x, axis_indices):
    starts = [
        d if s is None else s
        for s, d in zip(static_starts, _local_starts(axis_indices))
    ]
    dyn_dtypes = [s.dtype for s in starts if hasattr(s, "dtype")]
    dtype = jax.dtypes.result_type(*(dyn_dtypes or (jnp.int32,)))
    x, *idx_arrays = block_spec.pvary_operands(
        (x, *(jnp.asarray(s, dtype) for s in starts))
    )
    x = lax.dynamic_slice(x, idx_arrays, sizes)
    return lax.squeeze(x, squeezed) if squeezed else x

  return _slice_dynamic


def fuse_shard_map_jaxpr(
    jaxpr: jax_core.Jaxpr,
    consts: Sequence[Any],
    args: Sequence[Any],
    *,
    strict_mode: bool = True,
):
  """Fuses outer prologue and epilogue ops into a ``shard_map`` wrapping a ``fusible``.

  The ``shard_map`` is treated like a kernel whose grid is its manual mesh axes
  and whose blocks are the per-device shards. The outer ops feeding (and
  consuming) it become input (and output) fusions: we pull each boundary
  value's shard BlockSpec through them, which gives the block of each outer
  value this device needs. A new ``shard_map`` slices those blocks out of the
  local shards and recomputes the outer ops per shard around the original body.
  The new body has a top-level fusible and is fused as usual.
  """
  shmap_eqn_index, shmap_eqn = next(
      (i, eqn)
      for i, eqn in enumerate(jaxpr.eqns)
      if eqn.primitive is shard_map.shard_map_p
      and jaxpr_fusion._fusible_eqns(eqn.params["jaxpr"])
  )
  inner_jaxpr: jax_core.Jaxpr = shmap_eqn.params["jaxpr"]
  mesh: jax.sharding.Mesh = shmap_eqn.params["mesh"]
  check_vma: bool = shmap_eqn.params["check_vma"]
  newly_manual: frozenset[Any] = shmap_eqn.params["newly_manual_axes"]
  if any(
      mesh._name_to_type[a] != jax_core.AxisType.Explicit for a in newly_manual
  ):
    raise NotImplementedError(
        "fuse only supports shard_map over Explicit mesh axes."
    )
  if any(
      not isinstance(v.aval, jax_core.ShapedArray)
      for v in (*jaxpr.all_invars, *inner_jaxpr.constvars)
  ):
    raise NotImplementedError(
        "fuse with shard_map only supports array values, not Refs or tokens."
    )
  outer_eqns = jaxpr.eqns[:shmap_eqn_index] + jaxpr.eqns[shmap_eqn_index + 1 :]
  jaxpr_without_shmap = jaxpr.replace(
      eqns=outer_eqns,
      effects=jax_core.join_effects(*(eqn.effects for eqn in outer_eqns)),
  )
  inner_non_fusible_effects = frozenset().union(*(
      eqn.effects
      for eqn in inner_jaxpr.eqns
      if not jaxpr_fusion._is_fusible_eqn(eqn)
  ))
  if jaxpr_without_shmap.effects or jax_core.filter_named_axis_effects(
      inner_non_fusible_effects, mesh.axis_names
  ):
    raise NotImplementedError(
        "fuse with shard_map does not support side effects outside the fusible."
    )
  manual_axes = tuple(a for a in mesh.axis_names if a in newly_manual)
  candidate_values = [*consts, *args]

  def _manual(aval, spec=None):
    if not isinstance(aval, jax_core.ShapedArray):
      raise NotImplementedError(
          f"fuse with shard_map only supports array values, got {aval}."
      )
    return shard_map._manual_spec(
        newly_manual, aval.sharding.spec if spec is None else spec, mesh
    )

  def _shard_bs(aval, spec=None):
    return _block_spec_from_sharding(
        aval.shape, _manual(aval, spec), mesh, manual_axes
    )

  def _extractor(aval, bs):
    return _make_block_extractor(aval, _shard_bs(aval), bs, mesh, manual_axes)

  def _pull(fusion, out_bs, *in_avals):
    # Same steps as a fusible kernel: split out the fusion's values, pull the
    # output's shard block spec back to them, verify each pulled block lies
    # within the device's local shard, and in the body evaluate the fusion on
    # the blocks of the local shards.
    kfn, vals, sp = block_spec.get_fusion_values(fusion, *in_avals)
    if sp:
      raise NotImplementedError(
          "fuse with shard_map does not support scalar prefetch in outer"
          " operations."
      )
    kfn, (vals_bs, *in_bs), _ = block_spec.pull_block_spec(
        kfn, out_bs, grid_len=len(manual_axes), strict_mode=strict_mode
    )(vals, *in_avals)
    val_fns = [
        _extractor(jax_core.typeof(v), bs)
        for v, bs in zip(vals, vals_bs, strict=True)
    ]
    in_fns = [_extractor(a, bs) for a, bs in zip(in_avals, in_bs, strict=True)]

    def _eval(local_vals, axis_indices, *local_ins):
      return kfn(
          axis_indices,
          (),
          tuple(
              f(x, axis_indices) for f, x in zip(val_fns, local_vals, strict=True)
          ),
          *(f(x, axis_indices) for f, x in zip(in_fns, local_ins, strict=True)),
      )

    return _eval, vals

  # Input fusions: as in `jaxpr_fusion.fuse_jaxpr`, each `shard_map` operand is
  # computed by the eqns before the `shard_map`. A Python scalar operand is
  # inlined as a `Literal`, which `pull_block_spec` can't pull through, so it's
  # passed as is.
  prologue = jaxpr.replace(
      eqns=jaxpr.eqns[:shmap_eqn_index], effects=jax_core.no_effects
  )
  pre_fns, pre_operands = [], []
  for v, spec in zip(
      shmap_eqn.invars, shmap_eqn.params["in_specs"], strict=True
  ):
    if isinstance(v, jax_core.Literal):
      pre_fns.append(lambda *_, val=v.val: val)
      pre_operands.append(())
      continue
    fusion = jaxpr_fusion.construct_input_fusion(candidate_values, prologue, v)
    pre_fn, operands = _pull(fusion, _shard_bs(v.aval, spec))
    pre_fns.append(pre_fn)
    pre_operands.append(operands)

  # Output fusions, one per output (`pull_block_spec` can't pull through a
  # value read by several outputs): the eqns after the `shard_map`, which may
  # also read values computed before it, as in
  # `jaxpr_fusion._construct_output_fusions`. We pull through these too (no
  # push is needed): each output's sharding gives its shard block spec, which
  # we pull back to the `shard_map` outputs.
  # Outputs that don't depend on the `shard_map` are computed outside of it.
  shmap_out_avals = [v.aval for v in shmap_eqn.outvars]
  uses_shmap = []
  independent_outs = []
  post_fns = []
  post_operands = []
  for v in jaxpr.outvars:
    fn_jaxpr, fn_consts, *_ = jaxpr_fusion._construct_fusion_jaxpr(
        candidate_values, jaxpr_without_shmap, v, *shmap_eqn.outvars
    )
    independent_jaxpr, used_shmap_outs = pe.dce_jaxpr(fn_jaxpr, [True])
    uses_shmap.append(any(used_shmap_outs))
    if not uses_shmap[-1]:
      independent_outs.extend(
          jax_core.eval_jaxpr(independent_jaxpr, fn_consts)
      )
      continue

    def fusion(*outs, j=fn_jaxpr, c=fn_consts):
      return jax_core.eval_jaxpr(j, c, *outs)[0]

    post_fn, operands = _pull(fusion, _shard_bs(v.aval), *shmap_out_avals)
    post_fns.append(post_fn)
    post_operands.append(operands)
  if not post_fns:
    if shmap_eqn.effects:
      raise NotImplementedError(
          "fuse with shard_map does not support effectful fusibles with no"
          " dependent outputs."
      )
    return independent_outs

  def body(pre_operands, post_operands):
    axis_indices = tuple(lax.axis_index(a) for a in manual_axes)
    ins = [
        fn(operands, axis_indices)
        for fn, operands in zip(pre_fns, pre_operands, strict=True)
    ]
    outs = jax_core.eval_jaxpr(inner_jaxpr, inner_jaxpr.consts, *ins)
    return tuple(
        fn(operands, axis_indices, *outs)
        for fn, operands in zip(post_fns, post_operands, strict=True)
    )

  shmap_outs = shard_map.shard_map(
      jaxpr_fusion.fuse(
          body, resolve_fusion_dtypes=False, strict_mode=strict_mode
      ),
      mesh=mesh,
      out_specs=tuple(
          _manual(v.aval) for v, u in zip(jaxpr.outvars, uses_shmap) if u
      ),
      axis_names=newly_manual,
      check_vma=check_vma,
  )(pre_operands, post_operands)
  return util.merge_lists(uses_shmap, independent_outs, shmap_outs)
