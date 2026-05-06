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


def get_sparse_core_info() -> tpu_info.SparseCoreInfo:
  """Returns the SparseCore information for the current device.

  Raises:
    RuntimeError: If the current TPU does not have SparseCores.
  """
  sc_info = tpu_info.get_tpu_info().sparse_core
  if sc_info is None:
    raise RuntimeError("The current TPU does not have SparseCores")
  return sc_info


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScalarSubcoreMesh(pallas_core.Mesh):
  axis_name: str
  num_cores: int = dataclasses.field(
      default_factory=lambda: get_sparse_core_info().num_cores
  )

  def __post_init__(self):
    sc_info = get_sparse_core_info()
    if self.num_cores > sc_info.num_cores:
      raise ValueError(
          f"Mesh has {self.num_cores} cores, but the current TPU chip has only"
          f" {sc_info.num_cores} SparseCores"
      )

  @property
  def core_type(self) -> tpu_core.CoreType:
    return tpu_core.CoreType.SC_SCALAR_SUBCORE

  @property
  def default_memory_space(self) -> tpu_core.MemorySpace:
    return tpu_core.MemorySpace.HBM

  @property
  def shape(self):
    return collections.OrderedDict({self.axis_name: self.num_cores})

  @property
  def size(self) -> int:
    return self.num_cores

  @property
  def dimension_semantics(self) -> Sequence[tpu_core.DimensionSemantics]:
    return [tpu_core.GridDimensionSemantics.CORE_PARALLEL]

  def discharges_effect(self, effect):
    del effect  # Unused.
    return False

  def check_is_compatible_with(self, other_mesh):
    if isinstance(other_mesh, ScalarSubcoreMesh):
      raise ValueError("You can't use two different ScalarSubcoreMeshes.")
    elif isinstance(other_mesh, VectorSubcoreMesh):
      if (self.axis_name == other_mesh.core_axis_name
          and self.num_cores == other_mesh.num_cores):
        return True
      raise ValueError(f"{self} should have the same core axis name and number"
                       f" of cores as the VectorSubcoreMesh {other_mesh}.")
    elif isinstance(other_mesh, tpu_core.TensorCoreMesh):
      assert len(other_mesh.axis_names) == 1
      axis_name = other_mesh.axis_names[0]
      if self.axis_name == axis_name:
        raise ValueError(
            f"{self} should have a different axis name from the TensorCoreMesh"
            f" {other_mesh}."
        )
      return True
    return super().check_is_compatible_with(other_mesh)

  @property
  def supported_memory_spaces(self) -> Sequence[Any]:
    return [
        tpu_core.MemorySpace.VMEM_SHARED,
        tpu_core.MemorySpace.SMEM,
        tpu_core.MemorySpace.SEMAPHORE,
    ]


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
class VectorSubcoreMesh(pallas_core.Mesh):
  core_axis_name: str
  subcore_axis_name: str
  num_cores: int = dataclasses.field(
      default_factory=lambda: get_sparse_core_info().num_cores
  )
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
  def core_type(self) -> tpu_core.CoreType:
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
  def size(self) -> int:
    return self.num_cores * self.num_subcores

  @property
  def dimension_semantics(self) -> Sequence[tpu_core.DimensionSemantics]:
    return [
        tpu_core.GridDimensionSemantics.CORE_PARALLEL,
        tpu_core.GridDimensionSemantics.SUBCORE_PARALLEL,
    ]

  def discharges_effect(self, effect):
    del effect  # Unused.
    return False

  def check_is_compatible_with(self, other_mesh):
    if isinstance(other_mesh, VectorSubcoreMesh):
      raise ValueError("You can't use two different VectorSubcoreMeshes.")
    elif isinstance(other_mesh, ScalarSubcoreMesh):
      if (other_mesh.axis_name == self.core_axis_name
          and other_mesh.num_cores == self.num_cores):
        return True
      raise ValueError(f"{self} should have the same core axis name and number"
                       f" of cores as the ScalarSubcoreMesh {other_mesh}.")
    elif isinstance(other_mesh, tpu_core.TensorCoreMesh):
      assert len(other_mesh.axis_names) == 1
      axis_name = other_mesh.axis_names[0]
      if self.core_axis_name == axis_name:
        raise ValueError(
            f"{self} should have a different core axis name from the"
            f" TensorCoreMesh {other_mesh}."
        )
      if self.subcore_axis_name == axis_name:
        raise ValueError(
            f"{self} should have a different subcore axis name from the"
            f" TensorCoreMesh {other_mesh}."
        )
      return True
    return super().check_is_compatible_with(other_mesh)

  @property
  def supported_memory_spaces(self) -> Sequence[Any]:
    return [
        tpu_core.MemorySpace.VMEM,
        tpu_core.MemorySpace.VMEM_SHARED,
        tpu_core.MemorySpace.SMEM,
        tpu_core.MemorySpace.SEMAPHORE,
    ]


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


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Indices:
  """Indices for a gather or a scatter on SparseCore.

  Attributes:
    values: The values of the indices. Can be an array or a ref.
    ignored_value: If not None, the indices with this value will be ignored.
  """

  values: Any
  ignored_value: int | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )

  def pretty_print(
      self, context: jax_core.JaxprPpContext, *, print_dtype: bool = True
  ) -> str:
    values = jax_core.pp_var(
        self.values, context, print_literal_dtype=print_dtype
    )
    if self.ignored_value is None:
      return values
    return f"{values}~{self.ignored_value}"
