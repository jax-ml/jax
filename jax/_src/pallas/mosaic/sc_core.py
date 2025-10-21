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
import math
from typing import Any, TypeAlias

import jax
from jax._src import core as jax_core
from jax._src import state
from jax._src import tree_util
from jax._src.lax import lax
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas.mosaic import core as tpu_core
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
    return AbstractRef(self.inner_aval, self.memory_space, self.tiling)


class AbstractRef(state.AbstractRef):
  """An AbstractRef for SparseCore."""

  tiling: Tiling | None = None

  def __init__(
      self,
      aval: jax_core.AbstractValue,
      memory_space: tpu_core.MemorySpace,
      tiling: Tiling | None,
  ):
    super().__init__(aval, memory_space)

    self.tiling = tiling

  def update(  # type: ignore[override]
      self,
      inner_aval: Any | None = None,
      memory_space: Any | None = None,
      tiling: Tiling | None = None,
  ) -> AbstractRef:
    return AbstractRef(
        inner_aval if inner_aval is not None else self.inner_aval,
        memory_space if memory_space is not None else self.memory_space,
        tiling if tiling is not None else self.tiling,
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScalarSubcoreMesh:
  axis_name: str
  num_cores: int

  @property
  def backend(self) -> str:
    return "mosaic_tpu"

  @property
  def shape(self):
    return collections.OrderedDict(core=self.num_cores)

  def discharges_effect(self, effect):
    del effect  # Unused.
    return False


def _num_available_cores():
  """Returns the number of SparseCores on the current device."""
  device_kind = tpu_core.get_device_kind()
  match device_kind:
    case "TPU v5" | "TPU v5p":
      return 4
    case "TPU v6 lite" | "TPU v6" | "TPU7x":
      return 2
    case _:
      raise NotImplementedError(
          f"Unsupported device kind: {device_kind}"
      )


def _vector_dimension():
  """Returns the supported vector dimension for the current device."""
  device_kind = tpu_core.get_device_kind()
  match device_kind:
    case "TPU v5" | "TPU v5p" | "TPU v6" | "TPU v6 lite":
      return 8
    case "TPU7x":
      return 16
    case _:
      raise NotImplementedError(
          f"Unsupported device kind: {device_kind}"
      )


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
  if mesh.num_cores > (num_expected := _num_available_cores()):
    raise ValueError(
        f"Mesh has {mesh.num_cores} cores, but the current TPU chip has only"
        f" {num_expected} SparseCores"
    )
  if compiler_params is None:
    compiler_params = tpu_core.CompilerParams()
  if compiler_params.dimension_semantics is not None:
    raise ValueError("ScalarSubcoreMesh does not support dimension_semantics=")
  return pallas_core.default_mesh_discharge_rule(
      in_avals,
      out_avals,
      *args,
      mesh=mesh,
      jaxpr=jaxpr,
      compiler_params=dataclasses.replace(
          compiler_params,
          dimension_semantics=["core_parallel"],
          kernel_type=tpu_core.KernelType.SC_SCALAR_SUBCORE,
      ),
      interpret=interpret,
      debug=debug,
      cost_estimate=cost_estimate,
      name=name,
      memory_space=tpu_core.MemorySpace.HBM,
      metadata=metadata,
  )


pallas_core._core_map_mesh_rules[ScalarSubcoreMesh] = (
    _scalar_subcore_mesh_discharge_rule
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class VectorSubcoreMesh:
  core_axis_name: str
  subcore_axis_name: str
  num_cores: int
  num_subcores: int = dataclasses.field(default=16, init=False)

  @property
  def backend(self) -> str:
    return "mosaic_tpu"

  @property
  def shape(self):
    return collections.OrderedDict(
        core=self.num_cores, subcore=self.num_subcores)

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
  if mesh.num_cores > (num_expected := _num_available_cores()):
    raise ValueError(
        f"Mesh has {mesh.num_cores} cores, but the current TPU chip has only"
        f" {num_expected} SparseCores"
    )
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
      compiler_params=dataclasses.replace(
          compiler_params,
          dimension_semantics=["core_parallel", "subcore_parallel"],
          kernel_type=tpu_core.KernelType.SC_VECTOR_SUBCORE,
      ),
      interpret=interpret,
      debug=debug,
      cost_estimate=cost_estimate,
      name=name,
      memory_space=tpu_core.MemorySpace.HBM,
      metadata=metadata,
  )


pallas_core._core_map_mesh_rules[VectorSubcoreMesh] = (
    _vector_subcore_mesh_discharge_rule
)

def kernel(
    out_shape: object,
    *,
    mesh: pallas_core.Mesh,
    scratch_shapes: pallas_core.ScratchShapeTree = (),
    **kwargs: object,
):
  if unwrap_out := not isinstance(out_shape, (tuple, list)):
    out_shape = (out_shape,)

  def decorator(body):
    @jax.jit
    def wrapper(*args):
      arg_refs = jax.tree.map(jax_core.new_ref, args)
      out_refs = jax.tree.map(
          lambda out: jax_core.new_ref(
              lax.empty(out.shape, out.dtype),
              memory_space=getattr(out, "memory_space", None),
          ),
          out_shape,
      )

      @pallas_core.core_map(mesh, **kwargs)
      def _():
        return pallas_primitives.run_scoped(
            lambda *scratch_refs: body(*arg_refs, *out_refs, *scratch_refs),
            *scratch_shapes,
        )

      outs = jax.tree.map(lambda ref: ref[...], out_refs)
      return outs[0] if unwrap_out else outs

    return wrapper

  return decorator


# TODO(slebedev): Add more dtypes and vector shapes.
SUPPORTED_VECTOR_SHAPES = collections.defaultdict(list)
for dtype in [jnp.int32, jnp.uint32, jnp.float32]:
  SUPPORTED_VECTOR_SHAPES[jnp.dtype(dtype)].extend([
      # fmt: off
      (8,), (16,), (32,), (64,),
      (1, 8), (1, 16),
      (2, 8), (2, 16),
      (4, 8), (4, 16),
      # fmt: on
  ])
for dtype in [jnp.int16, jnp.uint16, jnp.float16, jnp.bfloat16]:
  SUPPORTED_VECTOR_SHAPES[jnp.dtype(dtype)].extend([
      # fmt: off
      (16,), (32,), (64,),
      (2, 8), (2, 16),
      # fmt: on
  ])
for dtype in [jnp.float16, jnp.bfloat16]:
  SUPPORTED_VECTOR_SHAPES[jnp.dtype(dtype)].extend([
      # fmt: off
      (4, 8), (4, 16),
      # fmt: on
  ])
for dtype in [jnp.int8, jnp.uint8]:
  SUPPORTED_VECTOR_SHAPES[jnp.dtype(dtype)].extend([
      # fmt: off
      (32,), (64,),
      (4, 8), (4, 16),
      # fmt: on
  ])


# Make sure all combinations are divisible by the vector register size.
supported_shapes: list[Any] = []
for dtype, supported_shapes in SUPPORTED_VECTOR_SHAPES.items():
  for shape in supported_shapes:
    assert (math.prod(shape) * dtype.itemsize) % 32 == 0
del dtype, supported_shapes
