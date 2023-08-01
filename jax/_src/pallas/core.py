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

"""Module for pallas-core functionality."""
from __future__ import annotations

from collections.abc import Sequence
import contextlib
import dataclasses
import functools
from typing import Any, Callable, Iterator

from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src.state import discharge as state_discharge
import jax.numpy as jnp

# TODO(sharadmv): enable type checking
# mypy: ignore-errors

partial = functools.partial
Grid = tuple[int, ...]
split_list = util.split_list

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


@dataclasses.dataclass
class GridEnv:
  axis_index: Any
  axis_size: int

_grid_env_stack: list[tuple[GridEnv, ...]] = []


@contextlib.contextmanager
def grid_env(env: tuple[tuple[Any, int], ...]) -> Iterator[None]:
  _grid_env_stack.append(tuple(GridEnv(axis_index, axis_size)
                               for axis_index, axis_size in env))
  try:
    yield
  finally:
    _grid_env_stack.pop()


def current_grid_env() -> tuple[GridEnv, ...] | None:
  if not _grid_env_stack:
    return None
  return _grid_env_stack[-1]


class Mapped:
  pass
mapped = Mapped()


@dataclasses.dataclass(frozen=True)
class BlockSpec:
  index_map: Callable[..., Any]
  block_shape: tuple[int | None, ...]

  def compute_index(self, *args):
    out = self.index_map(*args)
    if not isinstance(out, tuple):
      out = (out,)
    return out


@dataclasses.dataclass(frozen=True)
class BlockMapping:
  block_shape: tuple[Mapped | int, ...]
  index_map_jaxpr: jax_core.ClosedJaxpr

  def compute_start_indices(self, loop_idx, *args):
    discharged_jaxpr, discharged_consts = state_discharge.discharge_state(
        self.index_map_jaxpr.jaxpr, self.index_map_jaxpr.consts
    )
    jaxpr = jax_core.ClosedJaxpr(discharged_jaxpr, discharged_consts)
    block_indices_and_rest = jax_core.jaxpr_as_fun(jaxpr)(*loop_idx, *args)
    # Since we're passing in `Ref`s potentially, we need to split out their
    # updated values since we only care about the return values.
    block_indices, _ = split_list(block_indices_and_rest,
                                  [len(self.block_shape)])
    return tuple(i if b is mapped else b * i
                 for b, i in zip(self.block_shape, block_indices))

  replace = dataclasses.replace


@dataclasses.dataclass(frozen=True)
class GridMapping:
  grid: tuple[int, ...]
  block_mappings: tuple[BlockMapping | None, ...]
  mapped_dims: tuple[int, ...]
  num_index_operands: int

  replace = dataclasses.replace


def _preprocess_grid(grid: Grid | int | None) -> Grid:
  if grid is None:
    return ()
  if isinstance(grid, int):
    return (grid,)
  return grid


def _convert_block_spec_to_block_mapping(
    in_avals: list[jax_core.ShapedArray], block_spec: BlockSpec | None,
    ) -> BlockSpec | None:
  if block_spec is _no_block_spec:
    return None
  block_shape = tuple(
      mapped if s is None else s for s in block_spec.block_shape)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(block_spec.compute_index), in_avals)
  return BlockMapping(block_shape, jax_core.ClosedJaxpr(jaxpr, consts))


def _compute_shape_from_block_spec(block_spec: BlockSpec | None,
                                   arg_shape: tuple[int, ...]
                                   ) -> tuple[int, ...]:
  if block_spec is _no_block_spec:
    return arg_shape
  return tuple(s for s in block_spec.block_shape if s is not None)


def _get_ref_avals(grid, in_avals, in_specs, out_avals, out_specs):
  if grid is None:
    in_specs = [None] * len(in_avals)
    out_specs = [None] * len(out_avals)
    in_ref_avals = [state.shaped_array_ref(arg.shape, arg.dtype)
                    for arg in in_avals]
    out_ref_avals = [state.shaped_array_ref(arg.shape, arg.dtype)
                     for arg in out_avals]
  else:
    in_ref_avals = [
        state.shaped_array_ref(
            _compute_shape_from_block_spec(
                block_spec, arg.shape), arg.dtype)
        for block_spec, arg in zip(in_specs, in_avals)]
    out_ref_avals = [
        state.shaped_array_ref(
            _compute_shape_from_block_spec(
                block_spec, arg.shape), arg.dtype)
        for block_spec, arg in zip(out_specs, out_avals)]
  return in_specs, in_ref_avals, out_specs, out_ref_avals


_no_block_spec = object()

@dataclasses.dataclass(init=False)
class GridSpec:
  grid: Grid
  in_specs: Sequence[BlockSpec | None] | None
  out_specs: tuple[BlockSpec | None, ...] | None

  def __init__(
      self,
      grid: Grid | None = None,
      in_specs: Sequence[BlockSpec | None] | None = None,
      out_specs: BlockSpec | Sequence[BlockSpec | None] | None = None,
  ):
    if grid is None:
      if in_specs is not None:
        raise ValueError("Cannot specify `in_specs` with a `None` grid.")
      if out_specs is not None:
        raise ValueError("Cannot specify `out_specs` with a `None` grid.")
    self.grid = _preprocess_grid(grid)
    self.in_specs = in_specs
    if out_specs is not None and not isinstance(out_specs, (tuple, list)):
      out_specs = (out_specs,)
    if out_specs is not None and not isinstance(out_specs, tuple):
      out_specs = tuple(out_specs)
    self.out_specs = out_specs

  def get_grid_mapping(
      self, in_avals, in_tree, out_avals, out_tree
  ) -> tuple[tuple[jax_core.AbstractValue, ...], GridMapping]:
    if self.in_specs is not None:
      in_specs = self.in_specs
      in_spec_tree = tree_util.tree_structure(tuple(in_specs))
      if in_spec_tree != in_tree:
        raise ValueError(
            "Pytree specs for arguments and `in_specs` must match: "
            f"{in_tree} vs. {in_spec_tree}")
    else:
      in_specs = [_no_block_spec] * len(in_avals)
    if self.out_specs is not None:
      out_specs = self.out_specs
      out_spec_tree = tree_util.tree_structure(out_specs)
      if out_spec_tree != out_tree:
        raise ValueError(
            "Pytree specs for `out_shape` and `out_specs` must match: "
            f"{out_tree} vs. {out_spec_tree}")
    else:
      out_specs = [_no_block_spec] * len(out_avals)
    flat_in_specs = tree_util.tree_leaves(in_specs)
    flat_out_specs = tree_util.tree_leaves(out_specs)
    in_specs, in_ref_avals, out_specs, out_ref_avals = _get_ref_avals(
        self.grid, in_avals, flat_in_specs, out_avals,
        flat_out_specs)
    grid_avals = [jax_core.ShapedArray((), jnp.dtype("int32"))] * len(self.grid)
    in_block_mappings = map(
        partial(_convert_block_spec_to_block_mapping, grid_avals), in_specs)
    out_block_mappings = map(
        partial(_convert_block_spec_to_block_mapping, grid_avals), out_specs)
    grid_mapping = GridMapping(
        self.grid, (*in_block_mappings, *out_block_mappings), (),
        num_index_operands=0)
    jaxpr_in_avals = tree_util.tree_unflatten(in_tree, in_ref_avals)
    jaxpr_out_avals = tree_util.tree_unflatten(out_tree, out_ref_avals)
    return (*jaxpr_in_avals, *jaxpr_out_avals), grid_mapping
