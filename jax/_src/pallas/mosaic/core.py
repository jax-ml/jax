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

"""Contains TPU-specific Pallas abstractions."""
from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import enum
import functools

from jax._src import core as jax_core
from jax._src import state
from jax._src import tree_util
from jax._src import util
import jax.numpy as jnp
from jax._src.pallas import core as pallas_core

# TODO(sharadmv): enable type checking
# mypy: ignore-errors

partial = functools.partial
Grid = pallas_core.Grid
BlockSpec = pallas_core.BlockSpec
GridMapping = pallas_core.GridMapping
_preprocess_grid = pallas_core._preprocess_grid
_compute_shape_from_block_spec = pallas_core._compute_shape_from_block_spec
_convert_block_spec_to_block_mapping = pallas_core._convert_block_spec_to_block_mapping
split_list = util.split_list


class TPUMemorySpace(enum.Enum):
  VMEM = "vmem"
  SMEM = "smem"
  CMEM = "cmem"

  def __str__(self) -> str:
    return self.value


@dataclasses.dataclass(init=False)
class PrefetchScalarGridSpec(pallas_core.GridSpec):
  grid: Grid
  num_scalar_prefetch: int
  in_specs: Sequence[BlockSpec | None] | None
  out_specs: tuple[BlockSpec | None, ...] | None

  def __init__(
      self,
      num_scalar_prefetch: int,
      grid: Grid | None = None,
      in_specs: Sequence[BlockSpec | None] | None = None,
      out_specs: BlockSpec | Sequence[BlockSpec | None] | None = None,
  ):
    if grid is None:
      raise NotImplementedError("Should pass in non-`None` grid.")
    self.grid = _preprocess_grid(grid)
    if out_specs is not None and not isinstance(out_specs, (tuple, list)):
      out_specs = (out_specs,)
    if out_specs is not None and not isinstance(out_specs, tuple):
      out_specs = tuple(out_specs)
    self.num_scalar_prefetch = num_scalar_prefetch
    self.in_specs = in_specs
    self.out_specs = out_specs

  def get_grid_mapping(
      self, in_avals, in_tree, out_avals, out_tree
  ) -> tuple[tuple[jax_core.AbstractValue, ...], GridMapping]:
    scalar_avals, in_avals = split_list(in_avals, [self.num_scalar_prefetch])
    flat_in_specs = tree_util.tree_leaves(self.in_specs)
    flat_out_specs = tree_util.tree_leaves(self.out_specs)
    in_specs, in_ref_avals, out_specs, out_ref_avals = (
        pallas_core._get_ref_avals(
            self.grid, in_avals, flat_in_specs,
            out_avals, flat_out_specs))
    scalar_ref_avals = [
        state.shaped_array_ref(aval.shape, aval.dtype)
        for aval in scalar_avals]
    grid_avals = [jax_core.ShapedArray((), jnp.dtype("int32"))] * len(self.grid)
    in_block_mappings = map(
        partial(_convert_block_spec_to_block_mapping,
                (*grid_avals, *scalar_ref_avals)), in_specs)
    out_block_mappings = map(
        partial(_convert_block_spec_to_block_mapping,
                (*grid_avals, *scalar_ref_avals)), out_specs)
    grid_mapping = GridMapping(
        grid=self.grid,
        block_mappings=(*in_block_mappings, *out_block_mappings),
        mapped_dims=(),
        num_index_operands=self.num_scalar_prefetch,
    )
    jaxpr_in_avals = tree_util.tree_unflatten(
        in_tree, [*scalar_ref_avals, *in_ref_avals])
    jaxpr_out_avals = tree_util.tree_unflatten(out_tree, out_ref_avals)
    return (*jaxpr_in_avals, *jaxpr_out_avals), grid_mapping
