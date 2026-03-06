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
"""Contains SparseCore-specific Pallas abstractions."""

from __future__ import annotations

import collections
from collections.abc import Sequence
import dataclasses
from typing import Any, TypeAlias

import jax
from jax._src import core as jax_core
from jax._src import state
from jax._src import tree_util
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import tpu_info
import jax.numpy as jnp


Tiling: TypeAlias = Sequence[Sequence[int]]


@dataclasses.dataclass(frozen=True)
class MemoryRef(pallas_core.MemoryRef):
  """A MemoryRef for SparseCore."""

  tiling: Tiling | None = None

  def __init__(
      self,
      shape: Sequence[int],
      dtype: jax.typing.DTypeLike,
      memory_space: tpu_core.MemorySpace,
      tiling: Tiling | None = None,
  ):
    super().__init__(jax_core.ShapedArray(shape, dtype), memory_space)

    for tile in tiling or ():
      if len(tile) > len(shape):
        raise ValueError(
            f"Tile rank must not exceed shape rank: {tile=} vs {shape=}"
        )

    object.__setattr__(self, "tiling", tiling)

  def get_ref_aval(self) -> state.TransformedRef | state.AbstractRef:
    # TODO(sharadmv): Clean this up. ShapedArrayWithMemorySpace fails when we
    # try to apply JAX ops to it.
    return AbstractRef(self.inner_aval, self.memory_space, tiling=self.tiling)


class AbstractRef(state.AbstractRef):
  """An AbstractRef for SparseCore."""

  tiling: Tiling | None

  def __init__(
      self,
      aval: jax_core.AbstractValue,
      memory_space: tpu_core.MemorySpace,
      *,
      kind: Any | None = None,
      tiling: Tiling | None = None,
  ):
    super().__init__(aval, memory_space, kind)

    self.tiling = tiling

  def update(
      self,
      inner_aval: Any | None = None,
      memory_space: Any | None = None,
      kind: Any | None = None,
      tiling: Tiling | None = None,
  ) -> AbstractRef:
    return AbstractRef(
        inner_aval if inner_aval is not None else self.inner_aval,
        memory_space if memory_space is not None else self.memory_space,
        kind=kind if kind is not None else self.kind,
        tiling=tiling if tiling is not None else self.tiling,
    )


@dataclasses.dataclass
class BlockSpec(pallas_core.BlockSpec):
  """A BlockSpec for SparseCore.

  Attributes:
    indexed_by: The optional index of a parameter to use as the indexer. If set,
      the pipeline emitter will issue and indirect stream indexing into the
      value of this parameter as part of the pipeline.
    indexed_dim: The dimension to index into. Optional unless ``indexed_by`` is
      set.

  See also:
    :class:`jax.experimental.pallas.BlockSpec`
  """

  indexed_by: int | None = None
  indexed_dim: int | None = None

  def __post_init__(self):
    if (self.indexed_by is None) != (self.indexed_dim is None):
      raise ValueError(
          "indexed_by and indexed_dim must both be set or both unset"
      )

  def to_block_mapping(
      self,
      origin: pallas_core.OriginStr,
      array_aval: jax_core.ShapedArray,
      *,
      index_map_avals: Sequence[jax_core.AbstractValue],
      index_map_tree: tree_util.PyTreeDef,
      grid: pallas_core.GridMappingGrid,
      vmapped_dims: tuple[int, ...],
      debug: bool = False,
  ) -> BlockMapping:
    bm = super().to_block_mapping(
        origin,
        array_aval,
        index_map_avals=index_map_avals,
        index_map_tree=index_map_tree,
        grid=grid,
        vmapped_dims=vmapped_dims,
        debug=debug,
    )
    return BlockMapping(
        **{f.name: getattr(bm, f.name) for f in dataclasses.fields(bm)},
        indexed_by=self.indexed_by,
        indexed_dim=self.indexed_dim,
    )


@dataclasses.dataclass(frozen=True)
class BlockMapping(pallas_core.BlockMapping):
  indexed_by: int | None = None
  indexed_dim: int | None = None


def get_sparse_core_info() -> tpu_info.SparseCoreInfo:
  """Returns the SparseCore information for the current device."""
  return tpu_info.get_tpu_info().sparse_core or tpu_info.SparseCoreInfo(
      num_cores=0, num_subcores=0, num_lanes=0, dma_granule_size_bytes=0,
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScalarSubcoreMesh:
  axis_name: str
  num_cores: int

  def __post_init__(self):
    sc_info = get_sparse_core_info()
    if self.num_cores > sc_info.num_cores:
      raise ValueError(
          f"Mesh has {self.num_cores} cores, but the current TPU chip has only"
          f" {sc_info.num_cores} SparseCores"
      )

  @property
  def kernel_type(self) -> tpu_core.CoreType:
    return tpu_core.CoreType.SC_SCALAR_SUBCORE

  @property
  def default_memory_space(self) -> tpu_core.MemorySpace:
    return tpu_core.MemorySpace.HBM

  @property
  def shape(self):
    return collections.OrderedDict({self.axis_name: self.num_cores})

  @property
  def dimension_semantics(self) -> Sequence[str]:
    return ["core_parallel"]

  def discharges_effect(self, effect):
    del effect  # Unused.
    return False


def _scalar_subcore_mesh_discharge_rule(
    in_avals,
    out_avals,
    *args,
    mesh,
    jaxpr,
    compiler_params,
    interpret,
    debug,
    cost_estimate,
    name,
    metadata,
):
  if not isinstance(mesh, ScalarSubcoreMesh):
    raise TypeError(f"Mesh must be a ScalarSubcoreMesh, got {type(mesh)}")
  assert len(mesh.shape) == 1
  if compiler_params is None:
    compiler_params = tpu_core.CompilerParams()
  if compiler_params.dimension_semantics is not None:
    raise ValueError("ScalarSubcoreMesh does not support dimension_semantics=")
  jaxpr, in_avals, out_avals, args, is_scalar_const = tpu_core.pass_scalars_as_refs(
      jaxpr, args, in_avals, out_avals, mesh,
      # TODO(sharadmv): Delete this once we can pass into SMEM directly on
      # SparseCore.
      copy_to_smem=True,
  )
  refs_out, out = pallas_core.default_mesh_discharge_rule(
      in_avals,
      out_avals,
      *args,
      mesh=mesh,
      jaxpr=jaxpr,
      compiler_params=compiler_params,
      interpret=interpret,
      debug=debug,
      cost_estimate=cost_estimate,
      name=name,
      metadata=metadata,
  )
  refs_out = [
      a if not is_scalar else None
      for is_scalar, a in zip(is_scalar_const, refs_out)
  ]
  return refs_out, out


pallas_core._core_map_mesh_rules[ScalarSubcoreMesh] = (
    _scalar_subcore_mesh_discharge_rule
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class VectorSubcoreMesh:
  core_axis_name: str
  subcore_axis_name: str
  num_cores: int
  num_subcores: int = dataclasses.field(
      default_factory=lambda: get_sparse_core_info().num_subcores
  )

  def __post_init__(self):
    sc_info = get_sparse_core_info()
    if self.num_cores > sc_info.num_cores:
      raise ValueError(
          f"Mesh has {self.num_cores} cores, but the current TPU chip has only"
          f" {sc_info.num_cores} SparseCores"
      )
    if self.num_subcores > sc_info.num_subcores:
      raise ValueError(
          f"Mesh has {self.num_subcores} subcores, but the current TPU chip has"
          f" only {sc_info.num_subcores} subcores"
      )

  @property
  def kernel_type(self) -> tpu_core.CoreType:
    return tpu_core.CoreType.SC_VECTOR_SUBCORE

  @property
  def default_memory_space(self) -> tpu_core.MemorySpace:
    return tpu_core.MemorySpace.HBM

  @property
  def shape(self):
    return collections.OrderedDict({
        self.core_axis_name: self.num_cores,
        self.subcore_axis_name: self.num_subcores,
    })

  @property
  def dimension_semantics(self) -> Sequence[str]:
    return ["core_parallel", "subcore_parallel"]

  def discharges_effect(self, effect):
    del effect  # Unused.
    return False


def _vector_subcore_mesh_discharge_rule(
    in_avals,
    out_avals,
    *args,
    mesh,
    jaxpr,
    compiler_params,
    interpret,
    debug,
    cost_estimate,
    name,
    metadata,
):
  if not isinstance(mesh, VectorSubcoreMesh):
    raise TypeError(f"Mesh must be a VectorSubcoreMesh, got {type(mesh)}")
  assert len(mesh.shape) == 2
  if compiler_params is None:
    compiler_params = tpu_core.CompilerParams()
  if compiler_params.dimension_semantics is not None:
    raise ValueError("VectorSubcoreMesh does not support dimension_semantics=")
  return pallas_core.default_mesh_discharge_rule(
      in_avals,
      out_avals,
      *args,
      mesh=mesh,
      jaxpr=jaxpr,
      compiler_params=compiler_params,
      interpret=interpret,
      debug=debug,
      cost_estimate=cost_estimate,
      name=name,
      metadata=metadata,
  )


pallas_core._core_map_mesh_rules[VectorSubcoreMesh] = (
    _vector_subcore_mesh_discharge_rule
)


def supported_shapes(dtype: jax.typing.DTypeLike) -> Sequence[tuple[int, ...]]:
  """Returns all supported array shapes for the given dtype on SparseCore."""
  sc_info = get_sparse_core_info()
  num_lanes = sc_info.num_lanes
  itemsize = jnp.dtype(dtype).itemsize
  if itemsize > 4:
    raise ValueError(f"Unsupported dtype: {dtype}")
  packing_factor = 4 // itemsize
  if packing_factor == 1:
    return [(num_lanes,)]
  return [(num_lanes * packing_factor,), (packing_factor, num_lanes)]
