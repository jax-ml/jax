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

import copy
from collections.abc import Sequence
import contextlib
import dataclasses
import functools
from typing import Any, Callable, Union
from collections.abc import Iterator

from jax._src import api_util
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
Grid = tuple[Union[int, None], ...]  # None indicates that the bound is dynamic.
StaticGrid = tuple[int, ...]  # None indicates that the bound is dynamic.
split_list = util.split_list

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


class AbstractMemoryRef(state.AbstractRef):
  __slots__ = ["inner_aval", "memory_space"]

  def __init__(self, inner_aval: jax_core.AbstractValue,
               memory_space: Any):
    assert isinstance(inner_aval, jax_core.ShapedArray)
    self.inner_aval = inner_aval
    self.memory_space = memory_space

  def __repr__(self) -> str:
    return f'MemRef<{self.memory_space}>{{{self.inner_aval.str_short()}}}'

  def join(self, other):
    assert isinstance(other, AbstractMemoryRef)
    return AbstractMemoryRef(self.inner_aval.join(other.inner_aval),
                             self.memory_space)

  def update(self, inner_aval=None, memory_space=None):
    inner_aval = self.inner_aval if inner_aval is None else inner_aval
    memory_space = self.memory_space if memory_space is None else memory_space
    return AbstractMemoryRef(inner_aval, memory_space)

  def at_least_vspace(self):
    return AbstractMemoryRef(
        self.inner_aval.at_least_vspace(), self.memory_space)

  def __eq__(self, other):
    return (type(self) is type(other) and self.inner_aval == other.inner_aval
            and self.memory_space == other.memory_space)

  def __hash__(self):
    return hash((self.__class__, self.inner_aval, self.memory_space))


def _ref_raise_to_shaped(ref_aval: AbstractMemoryRef, weak_type):
  return AbstractMemoryRef(
      jax_core.raise_to_shaped(ref_aval.inner_aval, weak_type),
      ref_aval.memory_space)
jax_core.raise_to_shaped_mappings[AbstractMemoryRef] = _ref_raise_to_shaped


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
class Unblocked:
  padding: tuple[tuple[int, int], ...] | None = None
unblocked = Unblocked()


class Blocked:
  pass
blocked = Blocked()


IndexingMode = Union[Blocked, Unblocked]


@dataclasses.dataclass(init=False, unsafe_hash=True)
class BlockSpec:
  index_map: Callable[..., Any] | None
  block_shape: tuple[int | None, ...] | None
  memory_space: Any
  indexing_mode: IndexingMode

  def __init__(self, index_map: Callable[..., Any] | None = None,
               block_shape: tuple[int | None, ...] | None = None,
               memory_space: Any = None, indexing_mode: IndexingMode = blocked):
    self.index_map = index_map
    if block_shape is not None and not isinstance(block_shape, tuple):
      block_shape = tuple(block_shape)
    self.block_shape = block_shape
    self.memory_space = memory_space
    self.indexing_mode = indexing_mode

  def compute_index(self, *args):
    assert self.index_map is not None
    assert self.block_shape is not None
    out = self.index_map(*args)
    if not isinstance(out, tuple):
      out = (out,)
    return out


@dataclasses.dataclass(frozen=True)
class BlockMapping:
  block_shape: tuple[Mapped | int, ...]
  index_map_jaxpr: jax_core.ClosedJaxpr
  indexing_mode: IndexingMode

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
    if isinstance(self.indexing_mode, Blocked):
      return tuple(i if b is mapped else b * i
                  for b, i in zip(self.block_shape, block_indices))
    elif isinstance(self.indexing_mode, Unblocked):
      return block_indices
    else:
      raise RuntimeError(f"Unknown indexing mode: {self.indexing_mode}")

  replace = dataclasses.replace


@dataclasses.dataclass(frozen=True)
class GridMapping:
  grid: Grid
  block_mappings: tuple[BlockMapping | None, ...]
  mapped_dims: tuple[int, ...]
  num_index_operands: int
  num_scratch_operands: int

  replace = dataclasses.replace

  @property
  def num_dynamic_grid_bounds(self):
    return sum(b is None for b in self.grid)

  @property
  def static_grid(self) -> StaticGrid:
    if self.num_dynamic_grid_bounds:
      raise ValueError("Expected a grid with fully static bounds")
    return self.grid  # typing: ignore


def _preprocess_grid(grid: Grid | int | None) -> Grid:
  if grid is None:
    return ()
  if isinstance(grid, int):
    return (grid,)
  return grid


def _convert_block_spec_to_block_mapping(
    in_avals: list[jax_core.ShapedArray], block_spec: BlockSpec | None,
    aval: jax_core.ShapedArray, in_tree: Any,
    ) -> BlockSpec | None:
  if block_spec is no_block_spec:
    return None
  if block_spec.index_map is None:
    compute_index = lambda *args, **kwargs: (0,) * len(aval.shape)
    block_shape = aval.shape
  else:
    compute_index = block_spec.compute_index
    block_shape = block_spec.block_shape
  block_shape = tuple(
      mapped if s is None else s for s in block_shape)
  flat_fun, _ = api_util.flatten_fun(lu.wrap_init(compute_index), in_tree)
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
  return BlockMapping(
      block_shape, jax_core.ClosedJaxpr(jaxpr, consts), block_spec.indexing_mode
  )

def _tile_ref(ref: state.AbstractRef, block_shape: tuple[int, ...] | None
             ) -> state.AbstractRef:
  if block_shape is None:
    return ref
  shape = tuple(s for s in block_shape if s is not None)
  return ref.update(inner_aval=ref.inner_aval.update(shape=shape))


def _get_ref_avals(grid, in_avals, in_specs, out_avals, out_specs):
  def _get_memory_space(spec):
    if spec is no_block_spec:
      return None
    return spec.memory_space
  in_ref_avals = [
      AbstractMemoryRef(aval, _get_memory_space(in_spec))
      for aval, in_spec in zip(in_avals, in_specs)
  ]
  out_ref_avals = [
      AbstractMemoryRef(aval, _get_memory_space(out_spec))
      for aval, out_spec in zip(out_avals, out_specs)
  ]
  if grid is None:
    in_specs = [None] * len(in_avals)
    out_specs = [None] * len(out_avals)
  tiled_in_ref_avals = [
      aval if in_spec is no_block_spec
      else _tile_ref(aval, in_spec.block_shape)
      for aval, in_spec in zip(in_ref_avals, in_specs)
  ]
  tiled_out_ref_avals = [
      aval if out_spec is no_block_spec
      else _tile_ref(aval, out_spec.block_shape)
      for aval, out_spec in zip(out_ref_avals, out_specs)
  ]
  return in_specs, tiled_in_ref_avals, out_specs, tiled_out_ref_avals

class NoBlockSpec:
  pass
no_block_spec = NoBlockSpec()

@dataclasses.dataclass(init=False, unsafe_hash=True)
class GridSpec:
  grid: Grid
  in_specs: tuple[BlockSpec | NoBlockSpec, ...]
  out_specs: tuple[BlockSpec | NoBlockSpec, ...]
  in_specs_tree: Any
  out_specs_tree: Any

  def __init__(
      self,
      grid: Grid | None = None,
      in_specs: BlockSpec
      | Sequence[BlockSpec | NoBlockSpec]
      | NoBlockSpec = no_block_spec,
      out_specs: BlockSpec
      | Sequence[BlockSpec | NoBlockSpec]
      | NoBlockSpec = no_block_spec,
  ):
    # Be more lenient for in/out_specs
    if isinstance(in_specs, list):
      in_specs = tuple(in_specs)
    if isinstance(out_specs, list):
      out_specs = tuple(out_specs)

    self.grid = _preprocess_grid(grid)
    if in_specs is not no_block_spec:
      flat_in_specs, self.in_specs_tree = tree_util.tree_flatten(in_specs)
      self.in_specs = tuple(flat_in_specs)
    else:
      self.in_specs = in_specs
      self.in_specs_tree = None
    if out_specs is not no_block_spec:
      flat_out_specs, self.out_specs_tree = tree_util.tree_flatten(out_specs)
      self.out_specs = tuple(flat_out_specs)
    else:
      self.out_specs = out_specs
      self.out_specs_tree = None

  def _get_in_out_specs(self, in_avals, in_tree, out_avals, out_tree):
    if self.in_specs is no_block_spec:
      flat_in_specs = [no_block_spec] * len(in_avals)
    else:
      flat_in_specs = self.in_specs
      if self.in_specs_tree != in_tree:
        raise ValueError(
            "Pytree specs for arguments and `in_specs` must match: "
            f"{in_tree} vs. {self.in_specs_tree}")
    if self.out_specs is no_block_spec:
      flat_out_specs = [no_block_spec] * len(out_avals)
    else:
      flat_out_specs = self.out_specs
      if self.out_specs_tree != out_tree:
        raise ValueError(
            "Pytree specs for `out_shape` and `out_specs` must match: "
            f"{out_tree} vs. {self.out_specs_tree}")
    return flat_in_specs, flat_out_specs

  def get_grid_mapping(
      self, in_avals, in_tree, out_avals, out_tree
  ) -> tuple[tuple[jax_core.AbstractValue, ...], GridMapping]:
    flat_in_specs, flat_out_specs = self._get_in_out_specs(
        in_avals, in_tree, out_avals, out_tree)
    in_specs, in_ref_avals, out_specs, out_ref_avals = _get_ref_avals(
        self.grid, in_avals, flat_in_specs, out_avals,
        flat_out_specs)
    grid_avals = [jax_core.ShapedArray((), jnp.dtype("int32"))] * len(self.grid)
    # Create args, kwargs pytree def
    grid_tree = tree_util.tree_structure((tuple(grid_avals), {}))
    in_block_mappings = map(
        partial(_convert_block_spec_to_block_mapping, grid_avals,
                in_tree=grid_tree), in_specs, in_ref_avals)
    out_block_mappings = map(
        partial(_convert_block_spec_to_block_mapping, grid_avals,
                in_tree=grid_tree), out_specs, out_ref_avals)
    grid_mapping = GridMapping(
        self.grid, (*in_block_mappings, *out_block_mappings), (),
        num_index_operands=0, num_scratch_operands=0)
    jaxpr_in_avals = tree_util.tree_unflatten(in_tree, in_ref_avals)
    jaxpr_out_avals = tree_util.tree_unflatten(out_tree, out_ref_avals)
    if not isinstance(jaxpr_out_avals, (tuple, list)):
      jaxpr_out_avals = (jaxpr_out_avals,)
    return (*jaxpr_in_avals, *jaxpr_out_avals), grid_mapping

  def unzip_dynamic_grid_bounds(self) -> tuple[GridSpec, tuple[Any, ...]]:
    static_grid = tuple(d if isinstance(d, int) else None for d in self.grid)
    dynamic_bounds = tuple(d for d in self.grid if not isinstance(d, int))
    # We can't use dataclasses.replace, because our fields are incompatible
    # with __init__'s signature.
    static_self = copy.copy(self)
    static_self.grid = static_grid
    return static_self, dynamic_bounds
