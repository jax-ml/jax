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
from jax._src import config
from jax._src import core as jax_core
from jax._src import shard_map
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas.fuser import block_spec
from jax._src.pallas.fuser import jaxpr_fusion
import jax.numpy as jnp


def _count_fusible_eqns(jaxpr: jax_core.Jaxpr) -> int:
  return sum(
      _count_fusible_eqns(eqn.params["jaxpr"])
      if eqn.primitive is shard_map.shard_map_p
      else int(jaxpr_fusion._is_fusible_eqn(eqn))
      for eqn in jaxpr.eqns
  )


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
  dim_axes = [
      tuple(n for n in names.get(d, ()) if n in manual_axes)
      for d in range(len(shape))
  ]
  block_shape = tuple(
      size // math.prod(mesh.shape[n] for n in axes)
      for size, axes in zip(shape, dim_axes, strict=True)
  )

  def index_map(*axis_indices_and_scalar_prefetch):
    axis_indices = axis_indices_and_scalar_prefetch[: len(manual_axes)]
    axis_map = dict(zip(manual_axes, axis_indices, strict=True))

    def dim_index(axes):
      return functools.reduce(
          lambda idx, n: idx * mesh.shape[n] + axis_map[n], axes, 0
      )

    return tuple(dim_index(axes) for axes in dim_axes)

  return pallas_core.BlockSpec(block_shape=block_shape, index_map=index_map)


def _extract_local_block(
    x: jax.Array,
    shard_bs: pallas_core.BlockSpec,
    pulled_bs: pallas_core.BlockSpec | pallas_core.NoBlockSpec,
    axis_indices: tuple[jax.Array, ...],
    sp_vals: tuple[Any, ...],
) -> jax.Array:
  """Slices the block selected by ``pulled_bs`` out of this device's shard.

  ``x`` is the local shard (described by ``shard_bs``) of an array from outside
  the ``shard_map``, and ``pulled_bs`` is the block of that array that this
  device needs to recompute the outer ops per shard. Both are indexed by
  ``axis_indices`` (one per manual mesh axis). The pulled block lies inside the
  shard, so we convert its global offset into an offset within the shard, slice
  it out, and squeeze the dimensions that ``pulled_bs`` squeezes.
  """
  if not isinstance(pulled_bs, pallas_core.BlockSpec) or x.ndim == 0:
    return x
  assert shard_bs.index_map is not None and pulled_bs.index_map is not None
  assert pulled_bs.block_shape is not None
  shard_idx = shard_bs.index_map(*axis_indices)
  pulled_idx = pulled_bs.index_map(*axis_indices, *sp_vals)
  starts, sizes, squeezed = [], [], []
  for d, (size, bd, s_idx, p_idx) in enumerate(
      zip(x.shape, pulled_bs.block_shape, shard_idx, pulled_idx, strict=True)
  ):
    b = pallas_core.get_block_size(bd)
    sizes.append(b)
    if bd is None or isinstance(bd, pallas_core.Squeezed):
      squeezed.append(d)
    if b == size:
      # The block spans the whole shard dimension, so it must start at 0.
      starts.append(0)
      continue
    if bd is None or isinstance(
        bd, (pallas_core.Squeezed, pallas_core.Element)
    ):
      start = p_idx
    elif isinstance(bd, pallas_core.BoundedSlice):
      start = p_idx.start
    else:
      start = p_idx * b
    offset = s_idx * size
    starts.append(
        start if isinstance(offset, int) and offset == 0 else start - offset
    )
  if all(isinstance(s, int) for s in starts):
    if any(starts) or tuple(sizes) != x.shape:
      x = lax.slice(x, starts, [s + b for s, b in zip(starts, sizes)])
  else:
    dtypes = [s.dtype for s in starts if hasattr(s, "dtype")]
    dtype = jax.dtypes.result_type(*dtypes)
    idx_arrays = [jnp.asarray(s, dtype) for s in starts]
    if config._check_vma.value:
      vma = frozenset().union(*(
          a.mat.varying
          for v in (x, *idx_arrays)
          if isinstance(a := jax_core.typeof(v), jax_core.ShapedArray)
      ))
      if vma:
        x, *idx_arrays = [
            jax_core.pvary(v, tuple(vma - a.mat.varying))
            if isinstance(a := jax_core.typeof(v), jax_core.ShapedArray)
            and (vma - a.mat.varying)
            else v
            for v in (x, *idx_arrays)
        ]
    x = lax.dynamic_slice(x, idx_arrays, sizes)
  return lax.squeeze(x, squeezed) if squeezed else x


def fuse_shard_map_jaxpr(
    jaxpr: jax_core.Jaxpr,
    consts: Sequence[Any],
    args: Sequence[Any],
    *,
    strict_mode: bool = True,
):
  """Fuses ``jaxpr``, which has a ``shard_map`` with a ``fusible`` inside.

  The ``shard_map`` is treated like a kernel whose grid is its manual mesh axes
  and whose blocks are the per-device shards. The outer ops feeding (and
  consuming) it become input (and output) fusions: we pull each boundary
  value's shard BlockSpec through them, which gives the block of each outer
  value this device needs. A new ``shard_map`` slices those blocks out of the
  local shards and recomputes the outer ops per shard around the original body.
  The new body has a top-level fusible and is fused as usual.
  """
  shmap_fusible_counts = [
      (i, count)
      for i, eqn in enumerate(jaxpr.eqns)
      if eqn.primitive is shard_map.shard_map_p
      and (count := _count_fusible_eqns(eqn.params["jaxpr"]))
  ]
  match shmap_fusible_counts:
    case []:
      raise ValueError("No fusible found")
    case [(shmap_eqn_index, 1)]:
      pass
    case _:
      num_fusibles = sum(count for _, count in shmap_fusible_counts)
      raise ValueError(
          "Only a single fusible is allowed in a fused function, but found"
          f" {num_fusibles}"
      )
  shmap_eqn = jaxpr.eqns[shmap_eqn_index]
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
  manual_axes = tuple(a for a in mesh.axis_names if a in newly_manual)
  candidate_values = [*consts, *args]

  def _manual(aval):
    if not isinstance(aval, jax_core.ShapedArray):
      raise NotImplementedError(
          f"fuse with shard_map only supports array values, got {aval}."
      )
    return shard_map._manual_spec(newly_manual, aval.sharding.spec, mesh)

  def _shard_bs(aval, spec=None):
    return _block_spec_from_sharding(
        aval.shape, _manual(aval) if spec is None else spec, mesh, manual_axes
    )

  def _pull(fusion, out_bs, *in_avals):
    # Same steps as a fusible kernel: split out the fusion's values, pull the
    # output's shard block spec back to them, and in the body evaluate the
    # fusion on the blocks of the local shards.
    kfn, vals, sp = block_spec.get_fusion_values(fusion, *in_avals)
    kfn, (vals_bs, *in_bs), _ = block_spec.pull_block_spec(
        kfn, out_bs, grid_len=len(manual_axes), strict_mode=strict_mode
    )(vals, *in_avals)
    val_avals = [jax_core.typeof(v) for v in vals]

    def _eval(local_vals, local_sp, axis_indices, *local_ins):
      def blocks(xs, avals, bss):
        return tuple(
            _extract_local_block(x, _shard_bs(a), bs, axis_indices, local_sp)
            for x, a, bs in zip(xs, avals, bss, strict=True)
        )

      return kfn(
          axis_indices,
          local_sp,
          blocks(local_vals, val_avals, vals_bs),
          *blocks(local_ins, in_avals, in_bs),
      )

    return _eval, (vals, sp)

  # Input fusions: as in `jaxpr_fusion.fuse_jaxpr`, each `shard_map` operand is
  # computed by the eqns before the `shard_map`. A Python scalar operand is
  # inlined as a `Literal`, which `pull_block_spec` can't pull through, so it's
  # passed as is.
  prologue = jaxpr.replace(eqns=jaxpr.eqns[:shmap_eqn_index])
  pre_fns, pre_operands = [], []
  for v, spec in zip(
      shmap_eqn.invars, shmap_eqn.params["in_specs"], strict=True
  ):
    if isinstance(v, jax_core.Literal):
      pre_fns.append(lambda *_, val=v.val: val)
      pre_operands.append(((), ()))
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
  jaxpr_without_shmap = jaxpr.replace(
      eqns=jaxpr.eqns[:shmap_eqn_index] + jaxpr.eqns[shmap_eqn_index + 1 :]
  )
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
    return independent_outs

  def body(pre_operands, post_operands):
    axis_indices = tuple(lax.axis_index(a) for a in manual_axes)
    ins = [
        fn(*operands, axis_indices)
        for fn, operands in zip(pre_fns, pre_operands, strict=True)
    ]
    outs = jax_core.eval_jaxpr(inner_jaxpr, inner_jaxpr.consts, *ins)
    return tuple(
        fn(*operands, axis_indices, *outs)
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
