# Copyright 2024 The JAX Authors.
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

"""Contains GPU-specific Pallas abstractions."""

from collections.abc import Sequence
import dataclasses
import enum
from typing import ClassVar, Literal
from jax import core as jax_core
from jax._src import core
from jax._src import tree_util
from jax._src.pallas import core as pallas_core
from jax.experimental.mosaic import gpu as mosaic_gpu
import jax.numpy as jnp


AbstractMemoryRef = pallas_core.AbstractMemoryRef


@dataclasses.dataclass(frozen=True)
class GPUCompilerParams(pallas_core.CompilerParams):
  """Mosaic GPU compiler parameters.

  Attributes:
    dimension_semantics: A list of dimension semantics for each grid
      dimension of the kernel. Either "parallel" for dimensions that can
      execute in any order, or "sequential" for dimensions that must be
      executed sequentially.
    num_stages: The number of pipline stages in the kernel. Defaults to 1,
      meaning no pipelining is done.
  """
  PLATFORM: ClassVar[str] = "mosaic_gpu"
  dimension_semantics: Sequence[Literal["parallel", "sequential"]] | None = None
  num_stages: int = 1


class GPUMemorySpace(enum.Enum):
  GMEM = "gmem"
  SMEM = "smem"
  REGS = "regs"

  def __str__(self) -> str:
    return self.value

  def __call__(self, shape: tuple[int, ...], dtype: jnp.dtype):
    # A convenience function for constructing MemoryRef types.
    return MemoryRef(shape, dtype, self)


class TilingTransform(pallas_core.MemrefTransform):
  """Represents a tiling transformation for Memrefs.

  A tiling of (X, Y) on an array of shape (M, N) will result in a transformed
  shape of (M // X, N // Y, X, Y). Ex. A (256, 256) block that is tiled with a
  tiling of (64, 32) will be tiled as (4, 8, 64, 32).
  """

  def __init__(self, tiling: tuple[int, ...]):
    self.tiling = tiling

  def __call__(
      self, block_aval: pallas_core.AbstractMemoryRef
  ) -> pallas_core.AbstractMemoryRef:
    block_shape = block_aval.inner_aval.shape  # pytype: disable=attribute-error
    old_tiled_dims = block_shape[-len(self.tiling) :]
    num_tiles = tuple(
        block_dim // tiling_dim
        for block_dim, tiling_dim in zip(old_tiled_dims, self.tiling)
    )
    rem = (
        block_dim % tiling_dim
        for block_dim, tiling_dim in zip(old_tiled_dims, self.tiling)
    )
    if any(rem):
      raise ValueError(
          f"Block shape {block_shape} is not divisible by tiling {self.tiling}"
      )
    new_block_shape = block_shape[: -len(self.tiling)] + num_tiles + self.tiling
    return block_aval.update(
        inner_aval=block_aval.inner_aval.update(shape=new_block_shape)
    )

  def to_gpu_transform(self) -> mosaic_gpu.MemRefTransform:
    return mosaic_gpu.TileTransform(self.tiling)


@dataclasses.dataclass(frozen=True)
class GPUBlockMapping(pallas_core.BlockMapping):
  swizzle: int | None = None


@dataclasses.dataclass
class GPUBlockSpec(pallas_core.BlockSpec):
  # TODO(justinfu): Replace tiling a list of transforms.
  tiling: tuple[int, ...] | None = None
  swizzle: int | None = None

  def to_block_mapping(
      self,
      origin: pallas_core.OriginStr,
      array_aval: core.ShapedArray,
      *,
      index_map_avals: Sequence[core.AbstractValue],
      index_map_tree: tree_util.PyTreeDef,
      grid: pallas_core.GridMappingGrid,
      mapped_dims: tuple[int, ...],
  ) -> GPUBlockMapping:
    bm = super().to_block_mapping(
        origin,
        array_aval,
        index_map_avals=index_map_avals,
        index_map_tree=index_map_tree,
        grid=grid,
        mapped_dims=mapped_dims,
    )
    transforms: tuple[pallas_core.MemrefTransform, ...] = ()
    if self.tiling is not None:
      transforms += (TilingTransform(self.tiling),)
    return GPUBlockMapping(
        block_shape=bm.block_shape,
        block_aval=bm.block_aval,
        origin=bm.origin,
        index_map_jaxpr=bm.index_map_jaxpr,
        index_map_src_info=bm.index_map_src_info,
        indexing_mode=bm.indexing_mode,
        array_shape_dtype=bm.array_shape_dtype,
        transforms=transforms,
        swizzle=self.swizzle,
    )


# TODO(b/354568887): Cosolidate this with TPU's MemoryRef.
@dataclasses.dataclass(frozen=True)
class MemoryRef:
  """Like jax.ShapeDtypeStruct but with memory spaces."""

  shape: tuple[int, ...]
  dtype: jnp.dtype
  memory_space: GPUMemorySpace

  def get_aval(self) -> AbstractMemoryRef:
    return AbstractMemoryRef(
        jax_core.ShapedArray(self.shape, self.dtype), self.memory_space
    )

GMEM = GPUMemorySpace.GMEM
SMEM = GPUMemorySpace.SMEM
REGS = GPUMemorySpace.REGS
