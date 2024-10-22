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

from __future__ import annotations

import abc
import collections
from collections.abc import Sequence
import dataclasses
import enum
from typing import Any, ClassVar, Literal

from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import tree_util
from jax._src.pallas import core as pallas_core
from jax._src.pallas import pallas_call
from jax._src.state.types import Transform
import jax.experimental.mosaic.gpu as mgpu
import jax.numpy as jnp
from jaxlib.mlir import ir


AbstractMemoryRef = pallas_core.AbstractMemoryRef

DimensionSemantics = Literal["parallel", "sequential"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class GPUCompilerParams(pallas_core.CompilerParams):
  """Mosaic GPU compiler parameters.

  Attributes:
    approx_math: If True, the compiler is allowed to use approximate
      implementations of some math operations, e.g. ``exp``. Defaults to False.
    dimension_semantics: A list of dimension semantics for each grid
      dimension of the kernel. Either "parallel" for dimensions that can
      execute in any order, or "sequential" for dimensions that must be
      executed sequentially.
    max_concurrent_steps: The maximum number of sequential stages that are
      active concurrently. Defaults to 1.
    delay_release: The number of steps to wait before reusing the input/output
      references. Defaults to 0, and must be strictly smaller than
      max_concurrent_steps. Generally, you'll want to set it to 1 if you don't
      await the WGMMA in the body.
  """
  PLATFORM: ClassVar[str] = "mosaic_gpu"
  approx_math: bool = False
  dimension_semantics: Sequence[DimensionSemantics] | None = None
  max_concurrent_steps: int = 1
  delay_release: int = 0


class GPUMemorySpace(enum.Enum):
  #: Global memory.
  GMEM = "gmem"
  #: Shared memory.
  SMEM = "smem"
  #: Registers.
  REGS = "regs"

  def __str__(self) -> str:
    return self.value

  def __call__(
      self,
      shape: tuple[int, ...],
      dtype: jnp.dtype,
      transforms: Sequence[MemoryRefTransform] = (),
  ):
    # A convenience function for constructing MemoryRef types.
    return GPUMemoryRef(shape, dtype, memory_space=self, transforms=transforms)


@dataclasses.dataclass(frozen=True)
class GPUMemoryRef(pallas_core.MemoryRef):
  transforms: Sequence[MemoryRefTransform] = ()

  def get_ref_aval(self) -> pallas_core.TransformedRef | AbstractMemoryRef:
    aval = jax_core.ShapedArray(self.shape, self.dtype)
    for t in self.transforms:
      aval = t(aval)
    ref = pallas_core.TransformedRef(
        AbstractMemoryRef(aval, memory_space=self.memory_space), ()
    )
    for t in reversed(self.transforms):
      ref = t.undo(ref)
    if not ref.transforms:
      return ref.ref
    return ref


class MemoryRefTransform(pallas_core.MemoryRefTransform, abc.ABC):
  @abc.abstractmethod
  def to_gpu_transform(self) -> mgpu.MemRefTransform:
    pass

  def __call__(self, aval: jax_core.ShapedArray) -> jax_core.ShapedArray:
    return aval.update(
        shape=self.to_gpu_transform().transform_shape(aval.shape)
    )

Index = slice | int | ir.Value

@dataclasses.dataclass(frozen=True)
class TilingTransform(MemoryRefTransform):
  """Represents a tiling transformation for memory refs.

  A tiling of (X, Y) on an array of shape (M, N) will result in a transformed
  shape of (M // X, N // Y, X, Y). Ex. A (256, 256) block that is tiled with a
  tiling of (64, 32) will be tiled as (4, 8, 64, 32).
  """

  tiling: tuple[int, ...]

  def undo(self, ref: pallas_core.TransformedRef) -> pallas_core.TransformedRef:
    return dataclasses.replace(
        ref, transforms=(*ref.transforms, UntileRef(self.tiling))
    )

  def to_gpu_transform(self) -> mgpu.MemRefTransform:
    return mgpu.TileTransform(self.tiling)


@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class UntileRef(Transform):
  tiling: tuple[int, ...]

  def transform_shape(self, shape):
    if shape is None:
      return None
    assert shape[-len(self.tiling) :] == self.tiling
    shape = shape[: -len(self.tiling)]  # Drop tiling
    return shape[: -len(self.tiling)] + tuple(
        block_dim * tiling_dim
        for block_dim, tiling_dim in zip(shape[-len(self.tiling) :], self.tiling)
    )

  def transform_dtype(self, dtype):
    return dtype

  def untransform_index(
      self, idxs: tuple[Index, ...]
  ) -> tuple[tuple[Index, ...], Transform]:
    untiled_idxs = idxs[: -len(self.tiling)]
    tiled_idxs = idxs[-len(self.tiling) :]
    idxs_after_tiling = []
    for idx, tile in zip(tiled_idxs, self.tiling):
      if not isinstance(idx, slice):
        raise NotImplementedError("Non-slice indices are not supported")
      assert isinstance(idx, slice)
      if idx.step is not None and idx.step != 1:
        raise NotImplementedError("Strided slices unsupported")
      if (idx.start is not None and idx.start % tile) or (idx.stop is not None and idx.stop % tile):
        raise ValueError("Non-empty slices must be tile aligned")
      idxs_after_tiling.append(slice(idx.start // tile, idx.stop // tile))
    return (*untiled_idxs, *idxs_after_tiling, *(slice(None) for _ in self.tiling)), self

  def undo_to_gpu_transform(self) -> mgpu.MemRefTransform:
    return mgpu.TileTransform(self.tiling)

  def tree_flatten(self):
    return (), (self.tiling,)

  @classmethod
  def tree_unflatten(cls, metadata, arrays):
    assert not arrays
    return cls(*metadata)


def _perm_inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
  inverse = [-1] * len(permutation)
  for i, p in enumerate(permutation):
    inverse[p] = i
  return tuple(inverse)


@dataclasses.dataclass(frozen=True)
class TransposeTransform(MemoryRefTransform):
  """Transpose a tiled memref."""
  permutation: tuple[int, ...]

  def __post_init__(self):
    if set(self.permutation) != set(range(len(self.permutation))):
      raise ValueError(f"Permutation {self.permutation} is not a permutation.")

  def undo(self, ref: pallas_core.TransformedRef) -> pallas_core.TransformedRef:
    return dataclasses.replace(
        ref,
        transforms=(
            *ref.transforms,
            TransposeRef(_perm_inverse(self.permutation)),
        ),
    )

  def to_gpu_transform(self) -> mgpu.MemRefTransform:
    return mgpu.TransposeTransform(self.permutation)


@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class TransposeRef(Transform):
  permutation: tuple[int, ...]

  def transform_shape(self, shape):
    if shape is None:
      return None
    return tuple(shape[i] for i in self.permutation)

  def transform_dtype(self, dtype):
    return dtype

  def untransform_index(
      self, idxs: tuple[Index, ...]
  ) -> tuple[tuple[Index, ...], Transform]:
    removed_dims = [
        i for i, idx in enumerate(idxs) if not isinstance(idx, slice)
    ]
    new_perm = tuple(
        p - sum(d < p for d in removed_dims)
        for p in self.permutation
        if p not in removed_dims
    )
    new_idxs = tuple(idxs[i] for i in _perm_inverse(self.permutation))
    return new_idxs, TransposeRef(new_perm)

  def undo_to_gpu_transform(self) -> mgpu.MemRefTransform:
    return mgpu.TransposeTransform(_perm_inverse(self.permutation))

  def tree_flatten(self):
    return (), (self.permutation,)

  @classmethod
  def tree_unflatten(cls, metadata, arrays):
    assert not arrays
    return cls(*metadata)


def transpose_ref(
    ref: pallas_core.TransformedRef | Any,
    permutation: tuple[int, ...],
) -> pallas_core.TransformedRef:
  if not isinstance(ref, pallas_core.TransformedRef):
    if not isinstance(jax_core.get_aval(ref), pallas_core.AbstractMemoryRef):
      raise TypeError("ref must be a reference")
    ref = pallas_core.TransformedRef(ref, transforms=())
  return pallas_core.TransformedRef(
      ref.ref, (*ref.transforms, TransposeRef(permutation)),
  )


@dataclasses.dataclass(frozen=True)
class SwizzleTransform(MemoryRefTransform):
  swizzle: int

  def __post_init__(self):
    if self.swizzle not in {32, 64, 128}:
      raise ValueError(
          f"Swizzle {self.swizzle} is not supported. Only 32, 64 and 128 are"
          " accepted."
      )

  def undo(self, ref: pallas_core.TransformedRef) -> pallas_core.TransformedRef:
    return dataclasses.replace(
        ref, transforms=(*ref.transforms, UnswizzleRef(self.swizzle))
    )

  def to_gpu_transform(self) -> mgpu.MemRefTransform:
    raise RuntimeError("SwizzleTransform does not have a GPU transform.")

  def undo_to_gpu_transform(self) -> mgpu.MemRefTransform:
    # There's no swizzle transform in mgpu right now. It's a separate arg.
    raise NotImplementedError

  def __call__(self, aval: jax_core.ShapedArray) -> jax_core.ShapedArray:
    swizzle_elems = self.swizzle // aval.dtype.itemsize
    if swizzle_elems != aval.shape[-1]:
      raise ValueError(
          f"Swizzle {self.swizzle} requires the trailing dimension to be of"
          f" size {swizzle_elems}, but got shape: {aval.shape}"
      )
    return aval


@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class UnswizzleRef(Transform):
  swizzle: int

  def untransform_index(
      self, idxs: tuple[Index, ...]
  ) -> tuple[tuple[Index, ...], Transform]:
    if not idxs:
      return idxs, self
    if not all(isinstance(idx, slice) for idx in idxs[-2:]):
      raise NotImplementedError(
          "Non-slice indices are not supported in 2 minormost dims"
      )
    last_idx = idxs[-1]
    assert isinstance(last_idx, slice)
    if last_idx.step is not None and last_idx.step != 1:
      raise NotImplementedError("Swizzled dims cannot be sliced")
    if (last_idx.start is not None and last_idx.start != 0) or (
        last_idx.stop is not None and last_idx.stop != self.swizzle
    ):
      raise ValueError("Swizzled dims cannot be sliced")
    return idxs, self

  def tree_flatten(self):
    return (), (self.swizzle,)

  @classmethod
  def tree_unflatten(cls, metadata, arrays):
    assert not arrays
    return cls(*metadata)


@dataclasses.dataclass
class GPUBlockSpec(pallas_core.BlockSpec):
  transforms: Sequence[MemoryRefTransform] = ()

  def to_block_mapping(
      self,
      origin: pallas_core.OriginStr,
      array_aval: jax_core.ShapedArray,
      *,
      index_map_avals: Sequence[jax_core.AbstractValue],
      index_map_tree: tree_util.PyTreeDef,
      grid: pallas_core.GridMappingGrid,
      mapped_dims: tuple[int, ...],
  ) -> pallas_core.BlockMapping:
    bm = super().to_block_mapping(
        origin,
        array_aval,
        index_map_avals=index_map_avals,
        index_map_tree=index_map_tree,
        grid=grid,
        mapped_dims=mapped_dims,
    )
    block_inner_aval = bm.block_aval.inner_aval
    for t in self.transforms:
      block_inner_aval = t(block_inner_aval)
    return bm.replace(
        transformed_block_aval=bm.block_aval.update(
            inner_aval=block_inner_aval
        ),
        transforms=self.transforms,
    )


GMEM = GPUMemorySpace.GMEM
SMEM = GPUMemorySpace.SMEM
REGS = GPUMemorySpace.REGS


class barrier_dtype(dtypes.extended):
  pass


@dataclasses.dataclass(frozen=True)
class BarrierType(dtypes.ExtendedDType):
  type: ClassVar[Any] = barrier_dtype
  name: ClassVar[str] = "barrier"

  num_arrivals: int

  def __str__(self):
    return self.name


@dataclasses.dataclass(frozen=True)
class Barrier:
  num_arrivals: int
  num_barriers: int = 1

  def get_ref_aval(self) -> AbstractMemoryRef:
    aval = jax_core.ShapedArray(
        [self.num_barriers], BarrierType(self.num_arrivals)
    )
    return AbstractMemoryRef(aval, SMEM)


@dataclasses.dataclass(frozen=True)
class WGMMAAccumulatorRef:
  shape: tuple[int, int]
  dtype: jnp.dtype = jnp.float32

  def get_ref_aval(self) -> AbstractMemoryRef:
    return WGMMAAbstractAccumulatorRef(
        jax_core.ShapedArray(shape=self.shape, dtype=self.dtype), GPUMemorySpace.REGS
    )


def _is_trivial_index(idx):
  _is_deref1 = lambda i: i is Ellipsis or i == slice(None)
  if isinstance(idx, tuple):
    return all(_is_deref1(i) for i in idx)

  return _is_deref1(idx)

class WGMMAAbstractAccumulatorRef(AbstractMemoryRef):
  __slots__ = ["inner_aval", "memory_space"]

  def __repr__(self) -> str:
    return f'Accumulator{{{self.inner_aval.str_short()}}}'

  def join(self, other):
    return _as_accum(super().join(other))

  def update(self, inner_aval=None, memory_space=None):
    return _as_accum(super().update(inner_aval=None, memory_space=None))

  def at_least_vspace(self):
    return _as_accum(super().at_least_vspace())

  def _getitem(self, tracer, idx):
    from jax._src.pallas.mosaic_gpu.primitives import wgmma_accumulator_deref  # pytype: disable=import-error
    arr = wgmma_accumulator_deref(tracer)

    if not _is_trivial_index(idx):
      arr = arr[idx]

    return arr


def _as_accum(ref) -> WGMMAAbstractAccumulatorRef:
  return WGMMAAbstractAccumulatorRef(
      inner_aval=ref.inner_aval,
      memory_space=ref.memory_space,  # pytype: disable=attribute-error
  )

def _ref_raise_to_shaped(ref_aval, weak_type):
  return _as_accum(jax_core.raise_to_shaped_mappings[AbstractMemoryRef](ref_aval, weak_type))
jax_core.raise_to_shaped_mappings[WGMMAAbstractAccumulatorRef] = _ref_raise_to_shaped


_WARPGROUP_AXIS_NAME = object()

@dataclasses.dataclass(frozen=True, kw_only=True)
class GPUMesh:
  grid: tuple[int, ...] = ()
  cluster: tuple[int, ...] = ()
  # Those are NOT CUDA threads. On Hopper they correspond to warpgroups.
  num_threads: int | None = None
  axis_names: tuple[str, ...] = ()

  def __post_init__(self):
    if len(self.axis_names) != len(self.grid) + (self.num_threads is not None):
      raise ValueError("Need as many axis names as grid dimensions + warp groups")
    if self.num_threads > 2048 // 128:
      raise ValueError(
          "Requested too many CUDA threads per block. Each Mosaic thread"
          " corresponds to 128 CUDA threads."
      )
    if self.cluster:
      raise NotImplementedError(
          "Pallas/MosaicGPU does not support clusters yet."
      )

  @property
  def shape(self):
    if self.num_threads is not None:
      pairs = zip(self.axis_names, (*self.grid, *self.cluster, self.num_threads))
    else:
      pairs = tuple(
          zip(
              (*self.axis_names, _WARPGROUP_AXIS_NAME),
              (*self.grid, *self.cluster, 1),
          )
      )
    return collections.OrderedDict(pairs)


def _gpu_mesh_discharge_rule(
    in_avals,
    out_avals,
    *args,
    mesh,
    jaxpr,
):
  del out_avals
  assert isinstance(mesh, GPUMesh)
  if mesh.cluster:
    raise NotImplementedError
  if mesh.num_threads is None:
    raise NotImplementedError
  def body(*args):
    # Due to aliasing, args contains aliased inputs and outputs so we remove
    # outputs.
    in_refs = args[:len(in_avals)]
    jax_core.eval_jaxpr(jaxpr, in_refs)
  assert len(jaxpr.outvars) == 0
  any_spec = pallas_core.BlockSpec(memory_space=pallas_core.MemorySpace.ANY)
  out = pallas_call.pallas_call(
      body,
      out_shape=in_avals,
      in_specs=[any_spec] * len(in_avals),
      out_specs=[any_spec] * len(in_avals),
      input_output_aliases={i: i for i in range(len(in_avals))},
      grid=tuple(mesh.shape.items()),
      backend="mosaic_gpu",
  )(*args)
  return out, ()

pallas_core._core_map_mesh_rules[GPUMesh] = _gpu_mesh_discharge_rule
