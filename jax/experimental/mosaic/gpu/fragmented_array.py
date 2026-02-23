# Copyright 2024 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for code generator."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import functools
import itertools
import math
from typing import Any, Protocol, TypeAlias, TypeVar, cast, overload, runtime_checkable

import jax
import jax.experimental.mosaic.gpu as mgpu
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import math as mlir_math
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import vector
import numpy as np

from . import utils


T = TypeVar("T")
WARPGROUP_SIZE = utils.WARPGROUP_SIZE
WARP_SIZE = 32
WARPS_IN_WARPGROUP = WARPGROUP_SIZE // WARP_SIZE
SMEM_BANKS = 32
SMEM_BANK_BYTES = 4
c = utils.c


# TODO(bchetioui): Clean this up once minimum jaxlib version is at least 0.9.1.
if hasattr(nvvm, "ReductionKind"):
  ReductionKind = nvvm.ReductionKind
else:
  assert hasattr(nvvm, "ReduxKind")
  ReductionKind = nvvm.ReduxKind

Tiling: Any = mgpu.dialect.Tiling


def enumerate_negative(elems: Sequence[T]) -> Iterable[tuple[int, T]]:
  """Like built-in enumerate, but returns negative indices into the sequence."""
  offset = len(elems)
  for i, e in enumerate(elems):
    yield i - offset, e


Replicated: Any = mgpu.dialect.Replicated

@dataclasses.dataclass(frozen=True)
class TiledLayoutImpl:
  """A FragmentedArray layout derived from a tiling expression.

  A logical array is transformed according to the tiling expression, and then
  split across warps (within a warpgroup), lanes, and vectorized according to
  the dimension indices. All dimension indices must be negative and should refer
  to the dimensions after tiling is applied.

  To better understand this layout, consider the example of WGMMA-related tiling
  from https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-d as
  applied to a 128x128 array. The corresponding TiledLayout has a tiling of:

      (64, 8)(16, 8)(8, 8)(1, 2)

  and warp_dims=(-8,), lane_dims=(-4, -3), vector_dim=-1.

  We begin by applying the tiling (note that it always applies to a suffix):

          Tiled shape                       Remaining tiling actions
  ===========================================================================
  128 128                                  (64, 8)(16, 8)(8, 8)(1, 2)
    2  16  64  8                           (16, 8)(8, 8)(1, 2)
    2  16   4  1  16  8                    (8, 8)(1, 2)
    2  16   4  1   2  1  8  8              (1, 2)
    2  16   4  1   2  1  8  4  1  2

  The last expression is our final shape. At this stage, we're ready to partition
  the dimensions: warp_dims=(-8,) means that the 8-th dimension from the
  end is partitioned over 4 warps in a warpgroup (and so it must be of size 4).
  lane_dims=(-4, -3) indicate that those two dimensions are partitioned over
  the lanes within a warp (their product must be equal to 32, i.e. warp size).
  Finally, vector_dim=-1 indicates that each (logical) register is a vector
  containing 2 elements (there are no shape restrictions here).

  Given the above, the shape of the (logical) register array used to represent
  the array in each thread is: (2, 16, 1, 1, 2, 1, 1, 1, 1, 1). We have set all
  the dimensions above to 1, since each thread is a member of a single warp,
  a single lane, and the elements along the vectorized dimension are represented
  by a single (logical) register.
  """
  tiling: Tiling
  warp_dims: tuple[int | Replicated, ...]  # major-to-minor
  lane_dims: tuple[int | Replicated, ...]  # major-to-minor
  vector_dim: int
  # Whether to enforce that the layout is canonical. Users of `TiledLayout`
  # should not set this to `False`, but it is helpful to be able to construct
  # non-canonical layouts as an intermediate state when implementing layout
  # transformations.
  _check_canonical: dataclasses.InitVar[bool] = True

  def __post_init__(self, _check_canonical: bool):
    if not self.tiling.tiles:
      raise ValueError("Tiling must have at least one tile")
    min_shape = self.tiling.tiles[0]
    min_tiled_shape = self.tiling.tile_shape(min_shape)
    dims_set = {
        *self.partitioned_warp_dims, *self.partitioned_lane_dims, self.vector_dim,
    }
    if len(dims_set) != len(self.partitioned_warp_dims) + len(self.partitioned_lane_dims) + 1:
      raise ValueError("Duplicate partitioning dimensions")
    for d in dims_set:
      if d >= 0:
        raise ValueError("All dimensions must be negative")
      if d < -(len(min_tiled_shape) - len(min_shape)):
        raise ValueError("Dimension out of range")
    warp_dims_prod = math.prod(
        d.times if isinstance(d, Replicated) else min_tiled_shape[d]
        for d in self.warp_dims
    )
    if warp_dims_prod != WARPS_IN_WARPGROUP:
      raise ValueError(
          "The product of warp dims does not equal the number of warps in a"
          " warpgroup"
      )
    lane_dims_prod = math.prod(
        d.times if isinstance(d, Replicated) else min_tiled_shape[d]
        for d in self.lane_dims
    )
    if lane_dims_prod != WARP_SIZE:
      raise ValueError("The product of lane dims does not equal the warp size")
    if _check_canonical:
      canonical_layout = self.canonicalize()
      if self != canonical_layout:
        raise ValueError(f"{self} is not canonical.")

  @functools.cached_property
  def partitioned_warp_dims(self) -> tuple[int, ...]:
    return tuple(
      d for d in self.warp_dims if not isinstance(d, Replicated)
    )

  @functools.cached_property
  def partitioned_lane_dims(self) -> tuple[int, ...]:
    return tuple(
      d for d in self.lane_dims if not isinstance(d, Replicated)
    )

  def thread_idxs(self, shape: tuple[int, ...]) -> Iterable[tuple[ir.Value, ...]]:
    # We first find the linear index and then divide by the shape to
    # get the index.
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    contig_strides = tuple(utils.get_contiguous_strides(shape))
    tile_strides = self.tiling.tile_strides(contig_strides)
    dyn_tile_strides = [c(s, i32) for s in tile_strides[-self.tiled_tiling_rank:]]
    warp_offset = utils.dyn_dot(self.warp_indices(), dyn_tile_strides)
    lane_offset = utils.dyn_dot(self.lane_indices(), dyn_tile_strides)
    dyn_offset = arith.addi(warp_offset, lane_offset)
    register_shape = self.registers_shape(shape)
    for tile_idx in np.ndindex(register_shape):
      tile_lin_idx = sum(i * s for i, s in zip(tile_idx, tile_strides))
      dyn_lin_idx = arith.addi(dyn_offset, c(tile_lin_idx, i32))
      idx = []
      for stride in contig_strides:
        idx.append(arith.index_castui(index, arith.divui(dyn_lin_idx, c(stride, i32))))
        dyn_lin_idx = arith.remui(dyn_lin_idx, c(stride, i32))
      yield tuple(idx)

  @property
  def base_tile_shape(self) -> tuple[int, ...]:
    """The shape of the first tile in the tiling expression.

    This tile acts as the divisibility constraint for a suffix of arrays to
    which this layout applies.
    """
    return self.tiling.tiles[0]

  @functools.cached_property
  def tiled_tiling_shape(self) -> tuple[int, ...]:
    """The shape of the suffix of the array after tiling.

    We only allow our repeated tiling actions to further subdivide the
    dimensions created by previous tiling actions (except for the first one),
    so the tiled shape always ends with this suffix, no matter what array shape
    it's applied to.
    """
    base_tile_shape = self.base_tile_shape
    return self.tiling.tile_shape(base_tile_shape)[len(base_tile_shape):]

  @functools.cached_property
  def tiled_tiling_rank(self) -> int:
    return len(self.tiled_tiling_shape)

  @property
  def vector_length(self) -> int:
    return self.tiled_tiling_shape[self.vector_dim]

  def registers_element_type(self, t: ir.Type) -> ir.Type:
    return ir.VectorType.get((self.vector_length,), t)

  def registers_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Returns the shape of the register array needed to represent an array of the given logical shape."""
    tiled_shape = list(self.tiling.tile_shape(shape))
    for d in self.partitioned_warp_dims:
      tiled_shape[d] = 1
    for d in self.partitioned_lane_dims:
      tiled_shape[d] = 1
    tiled_shape[self.vector_dim] = 1
    return tuple(tiled_shape)

  def shape_from_registers_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Returns the logical shape of an array given its register array shape.

    Inverse to `registers_shape`.
    """
    tiled_tiling = self.tiled_tiling_shape
    shape = list(shape)
    for d in self.partitioned_warp_dims:
      shape[d] = tiled_tiling[d]
    for d in self.partitioned_lane_dims:
      shape[d] = tiled_tiling[d]
    shape[self.vector_dim] = tiled_tiling[self.vector_dim]
    return self.tiling.untile_shape(tuple(shape))

  def _delinearize_index(
      self, idx: ir.Value, dims: tuple[int | Replicated, ...]
  ) -> tuple[ir.Value, ...]:
    i32 = ir.IntegerType.get_signless(32)
    tiled_shape = self.tiled_tiling_shape
    dims_shape = tuple(
        d.times if isinstance(d, Replicated) else tiled_shape[d]
        for d in dims
    )
    dims_strides = utils.get_contiguous_strides(dims_shape)
    dims_indices = tuple(
        arith.remui(arith.divui(idx, c(stride, i32)), c(size, i32))
        for stride, size in zip(dims_strides, dims_shape)
    )
    full_indices = [arith.constant(i32, 0)] * len(tiled_shape)
    for d, i in zip(dims, dims_indices):
      if isinstance(d, Replicated):
        continue
      full_indices[d] = i
    return tuple(full_indices)

  def lane_indices(self) -> tuple[ir.Value, ...]:
    i32 = ir.IntegerType.get_signless(32)
    lane_idx = arith.remui(utils.thread_idx(), c(WARP_SIZE, i32))
    return self._delinearize_index(lane_idx, self.lane_dims)

  def warp_indices(self) -> tuple[ir.Value, ...]:
    i32 = ir.IntegerType.get_signless(32)
    warp_idx = arith.remui(
        arith.divui(utils.thread_idx(), c(WARP_SIZE, i32)),
        c(WARPS_IN_WARPGROUP, i32),
    )
    return self._delinearize_index(warp_idx, self.warp_dims)

  def remove_dimension(self, dim: int) -> TiledLayoutImpl:
    if dim < 0 or dim >= len(self.tiling.tiles[0]):
      raise ValueError(f"Dimension {dim} is out of range for {self.tiling}")
    new_tiling = self.tiling.remove_dimension(dim)
    tiled_shape = self.tiled_tiling_shape
    removed_dim = self.tiling.tile_dimension(dim)
    dim_offsets = np.cumsum(removed_dim[::-1])[::-1].tolist()
    if removed_dim[self.vector_dim]:
      new_tiling = Tiling((*new_tiling.tiles, (1,)))
      new_vector_dim = -1
      dim_offsets = [o - 1 for o in dim_offsets]  # We inserted an extra dim.
    else:
      new_vector_dim = self.vector_dim + dim_offsets[self.vector_dim]
    def replace_tiled_dim(d: int | Replicated, size: int):
      if isinstance(d, Replicated):
        return d
      elif removed_dim[d]:
        return Replicated(size)
      else:
        return d + dim_offsets[d]
    return TiledLayoutImpl(
        new_tiling,
        tuple(
            d if isinstance(d, Replicated) else replace_tiled_dim(d, tiled_shape[d])
            for d in self.warp_dims
        ),
        tuple(
            d if isinstance(d, Replicated) else replace_tiled_dim(d, tiled_shape[d])
            for d in self.lane_dims
        ),
        new_vector_dim,
        _check_canonical=False,
    ).canonicalize()

  def reduce(self, axes: Sequence[int]) -> TiledLayoutImpl:
    reduced_layout = self
    for a in sorted(axes, reverse=True):
      reduced_layout = reduced_layout.remove_dimension(a)
    return reduced_layout

  def canonicalize(self) -> TiledLayoutImpl:
    """Returns a version of this layout where tiling is canonical."""
    canonical_tiling = self.tiling.canonicalize()

    s = self.base_tile_shape
    tiled_tiling_shape = self.tiled_tiling_shape
    canonical_tiled_tiling_shape = canonical_tiling.tile_shape(s)[len(s):]
    offset = len(canonical_tiled_tiling_shape) - 1

    rev_removed_dims = []
    # Iterate starting from the end in order to eliminate leading dimensions,
    # whenever possible. For instance, say we have
    #
    #   shape=(4, 32, 1, 1, 1, 1, 1)
    #   warp_dims=(-7,),
    #   lane_dims=(-6,)
    #   vector_dim=-1
    #
    # and we want to canonicalize this to
    #
    #   shape=(4, 32, 1)
    #   warp_dims=(-3,),
    #   lane_dims=(-2,)
    #   vector_dim=-1.
    #
    # After the loop below, we end up with
    #
    #   rev_removed_dims=[False, True, True, True, True, False, False]
    #
    # which will yield offsets `4` for `warp_dims[0]`, `4` for `lane_dims[0]`,
    # and `0` for `vector_dim`.
    for d in reversed(tiled_tiling_shape):
      if offset >= 0 and d == canonical_tiled_tiling_shape[offset]:
        rev_removed_dims.append(False)
        offset -= 1
      else:
        rev_removed_dims.append(True)
    assert offset == -1

    dim_offsets = np.cumsum(rev_removed_dims)[::-1].tolist()

    def replace_tiled_dim(d: int | Replicated):
      return d if isinstance(d, Replicated) else d + dim_offsets[d]

    def is_nontrivial(d: int | Replicated):
      return isinstance(d, Replicated) or tiled_tiling_shape[d] != 1

    return TiledLayoutImpl(
        canonical_tiling,
        tuple(replace_tiled_dim(d) for d in self.warp_dims if is_nontrivial(d)),
        tuple(replace_tiled_dim(d) for d in self.lane_dims if is_nontrivial(d)),
        replace_tiled_dim(self.vector_dim),
        _check_canonical=False,
    )

# TODO(olechwierowicz): Clean this up once C++ TiledLayout and init_cc_mlir are always available in JAX build (min ver 0.9.1).
TiledLayout: Any
if (
    hasattr(mgpu.dialect, "TiledLayout")
    and (
        all_attrs_implemented := all(
            hasattr(mgpu.dialect.TiledLayout, attr)
            for attr in dir(TiledLayoutImpl)
            if not attr.startswith("_")
        )
    )
):
  TiledLayout = mgpu.dialect.TiledLayout
else:
  TiledLayout = TiledLayoutImpl


def _tiled_wgmma_layout(shape: tuple[int, ...]):
  """Returns the tiled layout relevant for WGMMA operations.

  The tiled layout is equivalent to one described here in PTX documentation:
  https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-d
  """
  if len(shape) != 2:
    raise ValueError(f"Shape {shape} is not 2D")
  if shape[0] % 64 != 0 or shape[1] % 8 != 0:
    raise ValueError(f"Shape {shape} is not a multiple of 64x8")
  return WGMMA_LAYOUT


@dataclasses.dataclass(frozen=True)
class WGSplatFragLayout:
  """A fragmented array where all the values are equal represented as a register per thread.

  FragmentedArrays in this layout can be are always the result of a
  splat, each thread in the warpgroup has a single copy of the value,
  while the FragmentedArray pretends it has whatever shape the user
  wants. This means we can trivially broadcast, reshape and do
  elementwise operations with all other layouts.

  Examples:

  To load a value in
  ```
  FragmentedArray.splat(memref.load(ref_1d, [1]), (10,20,2))
  ```

  A shape is always provided for sanity check reasons.

  """

  shape: tuple[int, ...] = ()

  def can_broadcast_to(self, shape) -> bool:
    """Check that the shape can be broadcast.

    All source dimensions must match the target's trailing dimensions by
    equality or being set to 1 (i.e. we can broadcast 1-sized dimensions or
    create new leading dimensions).
    """
    return len(self.shape) <= len(shape) and all(
        dim1 == dim2 or dim1 == 1
        for dim1, dim2 in zip(self.shape[::-1], shape[::-1])
    )

  def registers_element_type(self, t: ir.Type) -> ir.Type:
    return t

  def registers_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Returns the shape of the register array needed to represent an array of the given logical shape."""
    del shape  # Unused.
    return ()

  def shape_from_registers_shape(
      self, shape: tuple[int, ...]
  ) -> tuple[int, ...]:
    del shape  # Unused.
    return self.shape

  def thread_idxs(self, shape):
    assert shape == self.shape
    raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class WGStridedFragLayout:
  """Convert the array to 1D and then shard across threads."""

  shape: tuple[int, ...]
  vec_size: int

  def __post_init__(self):
    if np.prod(self.shape) % (self.vec_size * WARPGROUP_SIZE) != 0:
      raise ValueError((self, WARPGROUP_SIZE))

  @classmethod
  def from_shaped_type(cls, shaped_ty: ir.Type) -> WGStridedFragLayout | None:
    """Returns a WGStridedFragLayout for the given shaped type.

    Return None if the shaped type cannot have a strided layout.
    """
    if not isinstance(shaped_ty, ir.ShapedType):
      raise TypeError(shaped_ty)

    shaped_ty = ir.ShapedType(shaped_ty)
    if (bitwidth := mgpu.bitwidth(shaped_ty.element_type)) % 8:
      return None
    bw = bitwidth // 8
    assert 8 % bw == 0 and 8 // bw != 0, bw
    if math.prod(shaped_ty.shape) % WARPGROUP_SIZE != 0:
      return None
    max_vec_size = np.prod(shaped_ty.shape) // WARPGROUP_SIZE
    return cls(
        shape=tuple(shaped_ty.shape), vec_size=min(8 // bw, max_vec_size)
    )

  def registers_element_type(self, t: ir.Type) -> ir.Type:
    return ir.VectorType.get((self.vec_size,), t)

  def registers_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Returns the shape of the register array needed to represent an array of the given logical shape."""
    if shape != self.shape:
      raise ValueError(f"Shape {shape} is not compatible with {self}")
    return (math.prod(self.shape) // (WARPGROUP_SIZE * self.vec_size),)

  def shape_from_registers_shape(
      self, shape: tuple[int, ...]
  ) -> tuple[int, ...]:
    del shape  # Unused.
    return self.shape

  def thread_idxs(self, shape):
    assert shape == self.shape
    index = ir.IndexType.get()
    for v in self.linear_thread_idxs():
      res = []
      for dim in reversed(self.shape):
        dim = c(dim, index)
        res.append(arith.remui(v, dim))
        v = arith.divui(v, dim)
      res.reverse()
      yield res

  def linear_thread_idxs(self):
    """The indexes to be used for vector load/store WGStridedFragLayout.

    Yields:
      The indices of the vector that correspond to the current thread.
    """
    index = ir.IndexType.get()
    cardinality = np.prod(self.shape)
    assert cardinality % (WARPGROUP_SIZE * self.vec_size) == 0
    reg_num = cardinality // (WARPGROUP_SIZE * self.vec_size)
    tidx = arith.remui(gpu.thread_id(gpu.Dimension.x), c(WARPGROUP_SIZE, index))
    off = arith.muli(tidx, c(self.vec_size, tidx.type))
    for i in range(reg_num):
      yield arith.addi(off, c(i * WARPGROUP_SIZE * self.vec_size, tidx.type))

FragmentedLayout: TypeAlias = (
    WGSplatFragLayout | WGStridedFragLayout | TiledLayout
)


WGMMA_COL_LAYOUT = TiledLayout(
    Tiling(((8,), (2,))),
    warp_dims=(Replicated(4),),
    lane_dims=(Replicated(8), -2),
    vector_dim=-1,
)
WGMMA_ROW_LAYOUT = TiledLayout(
    Tiling(((64,), (16,), (8,), (1,))),
    warp_dims=(-4,),
    lane_dims=(-2, Replicated(4)),
    vector_dim=-1,
)

# The tiled layout is equivalent to one described here in PTX documentation:
# https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-d
# In this layout, we partition the 64x8 tiles over 4 warps into 16x8 tiles.
# Then, we further split the 16x8 tiles into 8x8 submatrices which are the unit
# of data that is split across a warp. Since 8*8 = 64, but a warp has only 32
# threads, we vectorize pairs of elements along columns.
# The assignment of elements to warp lanes is as follows:
#
#   0  0  1  1  2  2  3  3
#   4  4  5  5  6  6  7  7
#   8  8  9  9 10 10 11 11
#  12 12 13 13 14 14 15 15
#          ...
WGMMA_LAYOUT = TiledLayout(
    Tiling(((64, 8), (16, 8), (8, 8), (2,))),
    warp_dims=(-7,),
    lane_dims=(-3, -2),
    vector_dim=-1,
)
# This is the same as WGMMA_LAYOUT, only with a vector length of 1. LLVM now
# treats <2 x float> as a native PTX type and uses 64-bit registers to store
# them. This, in turn, means that we have to explode them into 32-bit registers
# right before WGMMA, which makes ptxas very unhappy and causes it to insert
# lots of WGMMA waits that absolutely tank the performance. As a workaround,
# we use this layout when 32-bit data with WGMMA_LAYOUT is used to initialize
# a WGMMAAccumulator, to ensure that the LLVM accumulator registers will always
# be represented as 32-bit PTX registers.
WGMMA_LAYOUT_ACC_32BIT = TiledLayout(
    Tiling(((64, 8), (16, 8), (8, 8), (2,), (1,))),
    warp_dims=(-8,),
    lane_dims=(-4, -3),
    vector_dim=-1,
)
# The tiled layout is equivalent to one described here in PTX documentation:
# https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n32-a
# In this layout, we partition the 64x16 tiles over 4 warps into 16x16 tiles.
# Then, we further split the 16x16 tiles into 8x16 submatrices which are the unit
# of data that is split across a warp. Since 8*16 = 128, but a warp has only 32
# threads, we vectorize quadruplets of elements along columns.
# The assignment of elements to warp lanes is as follows:
#
#   0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  3
#   4  4  4  4  5  5  5  5  6  6  6  6  7  7  7  7
#   8  8  8  8  9  9  9  9 10 10 10 10 11 11 11 11
#  12 12 12 12 13 13 13 13 14 14 14 14 15 15 15 15
#                     ...
WGMMA_LAYOUT_8BIT = TiledLayout(
    Tiling(((64, 16), (16, 16), (8, 16), (4,))),
    warp_dims=(-7,),
    lane_dims=(-3, -2),
    vector_dim=-1,
)
# This tiled layout is similar to the WGMMA layout, only the unit at which we
# assign submatrices to warps grows from 8x8 to 8x16. The elements within each
# submatrix are assigned to threads in the following way:
#
#   0  0  0  0  2  2  2  2  1  1  1  1  3  3  3  3
#   4  4  4  4  6  6  6  6  5  5  5  5  7  7  7  7
#                        ...
#
# Our vector length is twice the size of that of WGMMA_LAYOUT, which lets us use
# 32-bit SMEM loads/stores when dealing with 8-bit values. The conversion
# to the WGMMA layout only requires communication between with index differing
# in their 2 bit (i.e. 0 and 1, 2 and 4), so the conversion to WGMMA_LAYOUT
# only requires a single warp shuffle (plus permutes local to each thread).
WGMMA_LAYOUT_UPCAST_2X = TiledLayout(
    Tiling(((64, 16), (16, 16), (8, 16), (8,), (4,))),
    warp_dims=(-8,),
    lane_dims=(-4, -2, -3),
    vector_dim=-1,
)
# This layout should be used when upcasting 4-bit elements to 16-bit, for the
# purpose of passing them into WGMMA later. The core matrices stored by a warp
# are 8x32, because each of the 4 threads in a row holds 8 elements in a single
# vector. Note that unlike WGMMA_LAYOUT_UPCAST_2X, we assign columns to each
# group of 4 threads in order (as opposed to the swapping between 1 and 2,
# 5 and 6, etc. that WGMMA_LAYOUT_UPCAST_2X does).
WGMMA_LAYOUT_UPCAST_4X = TiledLayout(
    Tiling(((64, 32), (16, 32), (8, 32), (8,))),
    warp_dims=(-7,),
    lane_dims=(-3, -2),
    vector_dim=-1,
)
# This tiled layout is similar to WGMMA_LAYOUT. There, each warp stores a 8x8
# submatrix in the following way (we only show the first 4 rows for brevity):
#
#   0  0  1  1  2  2  3  3
#   4  4  5  5  6  6  7  7
#   8  8  9  9 10 10 11 11
#  12 12 13 13 14 14 15 15
#          ...
#
# This tiled layout stores the same 8x8 submatrix in the following way:
#
#   0  4  1  5  2  6  3  7
#   0  4  1  5  2  6  3  7
#   8 12  9 13 10 14 11 15
#   8 12  9 13 10 14 11 15
#          ...
#
# You can see that we have taken 2x2 submatrices from the above layout and
# transposed them. The assignment of lanes to elements is such that in both
# layouts the same two lanes map to a single 2x2 submatrix, making the transpose
# very cheap (one shuffle and permute suffices to change between those layouts).
WGMMA_TRANSPOSED_LAYOUT = TiledLayout(
    Tiling(((64, 8), (16, 8), (8, 8), (2, 2), (2, 1))),
    warp_dims=(-10,),
    lane_dims=(-6, -3, -5),
    vector_dim=-2,
)

# Like WGMMA_LAYOUT, only each warp holds a 32xN strip instead of 16xN.
TCGEN05_LAYOUT = TiledLayout(
    Tiling(((128, 8), (32, 8), (8, 8), (2,))),
    warp_dims=(-7,),
    lane_dims=(-3, -2),
    vector_dim=-1,
)
# Like WGMMA_TRANSPOSED_LAYOUT, only each warp holds a 32xN strip instead of 16xN.
TCGEN05_TRANSPOSED_LAYOUT = TiledLayout(
    Tiling(((128, 8), (32, 8), (8, 8), (2, 2), (2, 1))),
    warp_dims=(-10,),
    lane_dims=(-6, -3, -5),
    vector_dim=-2,
)
# TCGEN05_ROW_LAYOUT is to TCGEN05_LAYOUT as WGMMA_ROW_LAYOUT is to
# WGMMA_LAYOUT.
TCGEN05_ROW_LAYOUT = TiledLayout(
    Tiling(tiles=((128,), (32,), (8,), (1,))),
    warp_dims=(-4,),
    lane_dims=(-2, Replicated(times=4)),
    vector_dim=-1,
)
# TCGEN05_COL_LAYOUT is to TCGEN05_LAYOUT as WGMMA_COL_LAYOUT is to
# WGMMA_LAYOUT.
TCGEN05_COL_LAYOUT = TiledLayout(
    Tiling(tiles=((8,), (2,))),
    warp_dims=(Replicated(times=4),),
    lane_dims=(Replicated(times=8), -2),
    vector_dim=-1,
)


def tmem_native_layout(vector_length: int):
  """A layout resembling the logical organization of TMEM.

  The 128 rows in a tile are assigned to 128 lanes in the warpgroup. Useful when
  the result needs to be processed in registers and then stored back into TMEM.
  Usually shouldn't be used if the result is to be written back to SMEM, as
  there is no good way to store it without bank conflicts, but it still
  sometimes pays off.
  """
  return TiledLayout(
      Tiling(((128, vector_length), (32, vector_length))),
      warp_dims=(-4,),
      lane_dims=(-2,),
      vector_dim=-1,
  )

# We use a vector_dim of 2, to be able to make sure that the vectors are always
# a multiple of 32-bits, even when the data is 16-bits.
TMEM_NATIVE_LAYOUT = tmem_native_layout(2)

# A layout for the row indices used by TMA gather4/scatter4 instructions.
# Index 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 ...
# Warp  <--- 0 ---> <--- 1 ---> <--- 2 ---> <--- 3 ---> <--- 0 --
TMA_GATHER_INDICES_LAYOUT = TiledLayout(
    Tiling(((16,), (4,))),
    warp_dims=(-2,),
    lane_dims=(Replicated(32),),
    vector_dim=-1,
)


def can_relayout_wgmma_4x_to_wgmma_2x(bitwidth: int) -> bool:
  return bitwidth == 4


def can_relayout_wgmma_2x_to_wgmma(bitwidth: int) -> bool:
  return bitwidth <= 16


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(init=False, frozen=True, slots=True)
class FragmentedArray:
  # An array of ir.Value, see checks in init for shapes.
  registers: np.ndarray = dataclasses.field(repr=False)
  layout: FragmentedLayout
  is_signed: bool | None

  def __init__(
      self,
      *,
      _registers: np.ndarray,
      _layout: FragmentedLayout,
      _is_signed: bool | None,
  ):
    """Initializes a fragmented array.

    This is a low-level API. Prefer using classmethods to construct fragmented
    arrays instead.
    """
    # We need to use ``object.__setattr__`` here because of ``frozen=True``.
    object.__setattr__(self, "registers", _registers)
    object.__setattr__(self, "layout", _layout)
    object.__setattr__(self, "is_signed", _is_signed)

    if (_is_signed is not None) != isinstance(self.mlir_dtype, ir.IntegerType):
      raise TypeError(
          "is_signed must be non-None if and only if the MLIR type is an"
          f" integer type, got {_is_signed=} for {self.mlir_dtype}"
      )

    match self.layout:
      # Registers are flat
      case WGStridedFragLayout(shape):
        [reg_size] = ir.VectorType(_registers.flat[0].type).shape
        if (
            math.prod(shape)
            != math.prod(_registers.shape) * WARPGROUP_SIZE * reg_size
        ):
          raise ValueError(
              f"Invalid register array shape: math.prod({_registers.shape}) *"
              f" {WARPGROUP_SIZE} * {reg_size}, want: math.prod({shape})"
          )

      # Just a single register
      case WGSplatFragLayout():
        if _registers.size != 1:
          raise ValueError(f"Invalid register array shape: {_registers.shape}")

      case TiledLayout():
        try:
          self.layout.shape_from_registers_shape(_registers.shape)
        except ValueError:
          raise ValueError(
              "Register array shape does not match the tiled layout"
          ) from None
        [vec_len] = self.registers.flat[0].type.shape
        assert vec_len == self.layout.vector_length

      case _:
        raise NotImplementedError

  @classmethod
  def load_strided(
      cls,
      ref: ir.Value,
      *,
      is_signed: bool | None = None,
      vec_size: int | None = None,
  ) -> FragmentedArray:
    if not isinstance(ref.type, ir.MemRefType):
      raise TypeError(ref.type)

    ref_ty = ir.MemRefType(ref.type)
    shape = tuple(ref_ty.shape)
    if vec_size is None:
      layout = WGStridedFragLayout.from_shaped_type(ref_ty)
      if layout is None:
        raise ValueError(
            f"{ref_ty} must have a number of elements that is a multiple of"
            f" {WARPGROUP_SIZE} (got {math.prod(shape)})"
        )
    else:
      layout = WGStridedFragLayout(shape=shape, vec_size=vec_size)
    registers = np.empty(layout.registers_shape(shape), dtype=object)
    vec_ty = ir.VectorType.get((layout.vec_size,), ref_ty.element_type)
    for _get, update, ref, idx in cls.transfer_strided(ref, layout.vec_size):
      update(registers, vector.load(vec_ty, ref, idx))
    return cls(_registers=registers, _layout=layout, _is_signed=is_signed)

  @classmethod
  def splat(
      cls, value, shape, layout=None, *, is_signed: bool | None = None
  ) -> FragmentedArray:
    layout = layout or WGSplatFragLayout(shape)
    match layout:
      case WGSplatFragLayout():
        pass
      case WGStridedFragLayout() | TiledLayout():
        value = vector.broadcast(
            layout.registers_element_type(value.type), value
        )
      case _:
        raise NotImplementedError(layout)

    return cls(
        _registers=np.full(layout.registers_shape(shape), value, dtype=object),
        _layout=layout,
        _is_signed=is_signed,
    )

  @staticmethod
  def broadcasted_iota(
      dtype: ir.Type,
      shape: tuple[int, ...],
      dimension: int,
      layout: FragmentedLayout | None = None,
      *,
      is_signed: bool | None = None,
  ) -> FragmentedArray:
    """Creates a broadcasted iota array along the specified dimension."""
    if dimension >= len(shape):
      raise ValueError(
          "`dimension` must be smaller than the rank of the array."
      )

    def cast(idx: ir.Value) -> ir.Value:
      if isinstance(dtype, ir.FloatType):
        i32 = ir.IntegerType.get_signless(32)
        return arith.uitofp(dtype, arith.index_cast(i32, idx))
      return arith.index_cast(dtype, idx)

    return mgpu.FragmentedArray.splat(
        llvm.mlir_undef(dtype),
        shape,
        layout,
        is_signed=is_signed,
    ).foreach(
        lambda _, idx: cast(idx[dimension]),
        create_array=True,
        is_signed=is_signed,
    )

  @property
  def shape(self) -> tuple[int, ...]:
    match self.layout:
      case WGStridedFragLayout(shape):
        return shape
      case WGSplatFragLayout(shape=shape):
        return shape
      case TiledLayout():
        return self.layout.shape_from_registers_shape(self.registers.shape)
      case _:
        raise NotImplementedError

  @property
  def mlir_dtype(self) -> ir.Type:
    reg_ty = self.registers.flat[0].type
    match self.layout:
      case WGStridedFragLayout() | TiledLayout():
        return ir.VectorType(reg_ty).element_type
      case WGSplatFragLayout():
        return reg_ty
      case _:
        raise NotImplementedError

  def to_layout(self, new_layout: FragmentedLayout) -> FragmentedArray:
    """Converts the fragmented array to the given layout."""
    i32 = ir.IntegerType.get_signless(32)
    c = lambda x: arith.constant(i32, x)
    if self.layout == new_layout:
      return self
    shape = self.shape
    bitwidth = utils.bitwidth(self.mlir_dtype)
    transpose_pairs = (
        (WGMMA_LAYOUT, WGMMA_TRANSPOSED_LAYOUT),
        (TCGEN05_LAYOUT, TCGEN05_TRANSPOSED_LAYOUT),
    )
    if bitwidth in {16, 32} and (
        (self.layout, new_layout) in transpose_pairs
        or (new_layout, self.layout) in transpose_pairs
    ):
      is_even_row = arith.cmpi(
          arith.CmpIPredicate.eq,
          arith.remui(arith.divui(utils.thread_idx(), c(4)), c(2)),
          c(0),
      )
      perm = arith.select(is_even_row, c(0x5410), c(0x3276))
      tmp_new_regs = []
      for reg in self.registers.flat:
        reg_ty = reg.type
        if bitwidth == 16:
          reg = utils.bitcast(reg, i32)
          reg_shfl = utils.shfl_bfly(reg, 4)
          new_reg = utils.prmt(reg, reg_shfl, perm)
        elif bitwidth == 32:
          i32_vec = ir.VectorType.get((1,), i32)
          regs = [
              utils.bitcast(utils.vector_slice(reg, slice(i, i + 1)), i32)
              for i in range(2)
          ]
          reg_to_shfl = arith.select(is_even_row, regs[1], regs[0])
          reg_shfl = utils.shfl_bfly(reg_to_shfl, 4)
          new_reg_low = arith.select(is_even_row, regs[0], reg_shfl)
          new_reg_high = arith.select(is_even_row, reg_shfl, regs[1])
          new_reg_i32 = utils.vector_concat([
              utils.bitcast(new_reg_low, i32_vec),
              utils.bitcast(new_reg_high, i32_vec),
          ])
          new_reg = utils.bitcast(new_reg_i32, reg_ty)
        else:
          raise ValueError(f"Unsupported bitwidth: {bitwidth}")
        tmp_new_regs.append(utils.bitcast(new_reg, reg_ty))
      new_regs = np.asarray(
          tmp_new_regs, dtype=object
      ).reshape(new_layout.registers_shape(shape))
      return FragmentedArray(
          _registers=new_regs, _layout=new_layout, _is_signed=self.is_signed
      )
    if (
        isinstance(self.layout, TiledLayout)
        and isinstance(new_layout, TiledLayout)
        and self.layout == tmem_native_layout(self.layout.vector_length)
        and new_layout == tmem_native_layout(new_layout.vector_length)
    ):
      new_registers = np.empty(new_layout.registers_shape(shape), dtype=object)
      if self.layout.vector_length > new_layout.vector_length:
        ratio = self.layout.vector_length // new_layout.vector_length
        new_length = new_layout.vector_length
        for idx, reg in np.ndenumerate(self.registers):
          for i in range(ratio):
            new_reg = utils.vector_slice(
                reg, slice(i * new_length, (i + 1) * new_length)
            )
            new_registers[(idx[0], idx[1] * ratio + i, *idx[2:])] = new_reg
      elif self.layout.vector_length < new_layout.vector_length:
        ratio = new_layout.vector_length // self.layout.vector_length
        for idx in np.ndindex(new_registers.shape):
          new_reg = utils.vector_concat([
              self.registers[idx[0], idx[1] * ratio + i, *idx[2:]]
              for i in range(ratio)
          ])
          new_registers[idx] = new_reg
      return FragmentedArray(
          _registers=new_registers, _layout=new_layout, _is_signed=self.is_signed,
      )
    if self.layout == WGMMA_LAYOUT_ACC_32BIT and new_layout == WGMMA_LAYOUT:
      new_regs_shape = new_layout.registers_shape(shape)
      assert new_regs_shape[-1] == 1
      assert self.registers.shape == (*new_regs_shape[:-1], 2, 1)
      new_regs = np.empty(new_regs_shape, dtype=object)
      for idx in np.ndindex(new_regs_shape[:-1]):
        new_regs[(*idx, 0)] = utils.vector_concat([
            self.registers[*idx, i, 0] for i in range(2)
        ])
      return FragmentedArray(
          _registers=new_regs, _layout=new_layout, _is_signed=self.is_signed,
      )
    if self.layout == WGMMA_LAYOUT and new_layout == WGMMA_LAYOUT_ACC_32BIT:
      new_regs_shape = new_layout.registers_shape(shape)
      assert self.registers.shape[-1] == 1
      assert new_regs_shape == (*self.registers.shape[:-1], 2, 1)
      new_regs = np.empty(new_regs_shape, dtype=object)
      for idx, reg in np.ndenumerate(self.registers):
        for i in range(2):
          new_regs[(*idx[:-1], i, 0)] = utils.vector_slice(reg, slice(i, i + 1))
      return FragmentedArray(
          _registers=new_regs, _layout=new_layout, _is_signed=self.is_signed,
      )
    dtype_bitwidth = utils.bitwidth(self.mlir_dtype)
    if (
        self.layout == WGMMA_LAYOUT_UPCAST_2X
        and new_layout == WGMMA_LAYOUT
        and can_relayout_wgmma_2x_to_wgmma(dtype_bitwidth)
    ):
      assert shape[1] % 16 == 0  # Should be implied by the layout
      new_registers = np.empty(new_layout.registers_shape(shape), dtype=object)
      is_even = arith.cmpi(
          arith.CmpIPredicate.eq, arith.remui(utils.thread_idx(), c(2)), c(0)
      )
      registers = self.registers
      if dtype_bitwidth == 4:
        if registers.shape[1] % 2:
          raise NotImplementedError(
              "This relayout implementation requires an even number of column"
              " tiles (to pack pairs of them for efficiency)"
          )
        # We pair up the consecutive column tiles, so each register is 32-bit.
        # If this layout originated from a WGMMA_LAYOUT_UPCAST_4X layout,
        # LLVM will realize that the paired up vectors actually came from the
        # same 32-bit register and it will become a no-op.
        col_minor_registers = np.moveaxis(registers, 1, -1)
        flat_registers = [
            utils.vector_concat((l, h))
            for l, h in zip(
                col_minor_registers.flat[::2], col_minor_registers.flat[1::2]
            )
        ]
        registers = np.asarray(flat_registers, dtype=object).reshape(
            *col_minor_registers.shape[:-1], col_minor_registers.shape[-1] // 2
        )
        registers = np.moveaxis(registers, -1, 1)
      for idx, reg in np.ndenumerate(registers):
        if dtype_bitwidth == 16:
          assert reg.type.shape == [4]
          # A single vector is 64-bits, but shuffles are only 32-bit wide.
          # We only shuffle the half that needs to go to other thread.
          low = utils.vector_slice(reg, slice(0, 2))
          high = utils.vector_slice(reg, slice(2, 4))
          to_exchange = arith.select(is_even, high, low)
          # Exchange values between even and odd threads.
          exchanged = utils.shfl_bfly(to_exchange, 1)
          low = arith.select(is_even, low, exchanged)
          high = arith.select(is_even, exchanged, high)
          new_registers[(idx[0], idx[1] * 2, *idx[2:-1])] = low
          new_registers[(idx[0], idx[1] * 2 + 1, *idx[2:-1])] = high
        elif dtype_bitwidth == 8:
          assert reg.type.shape == [4]
          # The vector is 32-bits, so we just shuffle the whole thing and
          # use prmt to blend it with the local register.
          exchanged = utils.shfl_bfly(reg, 1)
          # Consider lanes 0 and 1, because the situation is symmetric for
          # each pair. If we feed reg[lane] and exchanged[lane] (which is
          # really the same as reg of the other lane) to prmt, we can index
          # the elements of the result using the following indices:
          #     reg[0]:   0 1 2 3       reg[1]:  8 9 10 11
          #     prmt[0]:  0 1 2 3                4 5  6  7
          #     prmt[1]:  4 5 6 7                0 1  2  3
          # The expected outputs and their respective permutations are:
          #     out[0]:   0 1 8 9       out[1]:  2 3 10 11
          #     prmt[0]:  0 1 4 5       prmt[1]: 6 7  2  3
          # Note that the patterns still need to be flipped, since we listed
          # bytes with LSB on the left, which is the opposite of how the
          # numeric constants are spelled in Python (LSB on the right).
          perm = arith.select(is_even, c(0x5410), c(0x3276))
          blend = utils.prmt(reg, exchanged, perm)
          for i in range(2):
            reg = utils.vector_slice(blend, slice(i * 2, i * 2 + 2))
            new_registers[(idx[0], idx[1] * 2 + i, *idx[2:-1])] = reg
        else:
          assert dtype_bitwidth == 4
          assert reg.type.shape == [8]  # We paired up the registers above.
          exchanged = utils.shfl_bfly(reg, 1)
          # See comment above for a more complete explanation.
          #     reg[0]:   0 1 2 3 16 17 18 19   reg[1]:  8 9 10 11 24 25 26 27
          #     prmt[0]:  -0- -1- --2-- --3--            -4- --5-- --6-- --7--
          #     prmt[1]:  -4- -5- --6-- --7--            -0- --1-- --2-- --3--
          # The expected outputs and their respective permutations are:
          #     out[0]:   0 1 8 9 16 17 24 25   out[1]:  2 3 10 11 18 19 26 27
          #     prmt[0]:  -0- -4- --2-- --6--  prmt[1]:  -5- --1-- --7-- --3--
          perm = arith.select(is_even, c(0x6240), c(0x3715))
          blend = utils.prmt(reg, exchanged, perm)
          for i in range(4):
            reg = utils.vector_slice(blend, slice(i * 2, i * 2 + 2))
            new_registers[(idx[0], idx[1] * 4 + i, *idx[2:-1])] = reg
      assert all(r is not None for r in new_registers)
      return FragmentedArray(
          _registers=new_registers, _layout=new_layout, _is_signed=self.is_signed,
      )
    if (
        self.layout == WGMMA_LAYOUT_UPCAST_4X
        and new_layout == WGMMA_LAYOUT_UPCAST_2X
        and can_relayout_wgmma_4x_to_wgmma_2x(dtype_bitwidth)
    ):
      assert shape[0] % 64 == 0  # Should be implied by the layout
      assert shape[1] % 32 == 0  # Should be implied by the layout
      new_registers = np.empty(new_layout.registers_shape(shape), dtype=object)
      i32 = ir.IntegerType.get_signless(32)
      c = lambda x: arith.constant(i32, x)
      is_01 = arith.cmpi(
          arith.CmpIPredicate.ult, arith.remui(utils.thread_idx(), c(4)), c(2)
      )
      for idx, reg in np.ndenumerate(self.registers):
        assert ir.VectorType(reg.type).shape == [8]
        # The vector is 32-bits, so we just shuffle the whole thing and
        # use prmt to blend it with the local register.
        exchanged = utils.shfl_bfly(reg, 2)
        # See comments above for conventions. Here we exchange data between
        # threads with lane index related by flipping 2nd bit (e.g. 0 and 2).
        #     reg[0]:   0 1 2 3 4 5 6 7       reg[2]:  16 17 18 19 20 21 22 23
        #     prmt[0]:  -0- -1- -2- -3-                --4-- --5-- --6-- --7--
        #     prmt[1]:  -4- -5- -6- -7-                --0-- --1-- --2-- --3--
        # The expected outputs and their respective permutations are:
        #     out[0]:   0 1 2 3 16 17 18 19   out[2]:  4 5 6 7 20 21 22 23
        #     prmt[0]:  -0- -1- --4-- --5--  prmt[2]:  -6- -7- --2-- --3--
        perm = arith.select(is_01, c(0x5410), c(0x3276))
        blend = utils.prmt(reg, exchanged, perm)
        for i in range(2):
          reg = utils.vector_slice(blend, slice(i * 4, i * 4 + 4))
          new_registers[(idx[0], idx[1] * 2 + i, *idx[2:-1])] = reg
      assert all(r is not None for r in new_registers)
      return FragmentedArray(
          _registers=new_registers, _layout=new_layout, _is_signed=self.is_signed,
      )
    if self.layout == WGMMA_LAYOUT_UPCAST_4X and new_layout == WGMMA_LAYOUT:
      return self.to_layout(WGMMA_LAYOUT_UPCAST_2X).to_layout(new_layout)
    if not isinstance(self.layout, WGSplatFragLayout):
      raise NotImplementedError(
          f"Cannot convert from {self.layout} to {new_layout}"
      )
    return type(self).splat(
        self.registers.item(), self.shape, new_layout, is_signed=self.is_signed
    )

  def _pointwise(
      self,
      op,
      *other,
      output_is_signed: bool | None = None,
      restrict_bitwidth: bool = True,
  ) -> FragmentedArray:
    if restrict_bitwidth:
      if (bitwidth := utils.bitwidth(self.mlir_dtype)) <= 8 and bitwidth != 1:
        raise NotImplementedError(
            f"Pointwise operations on {bitwidth}-bit types are unsupported"
            " (except bitwise operations). Upcast to a 16- or 32-bit type"
            " before performing the operation."
        )
    # If our layout is a splat, then we should either dispatch to a non-splat
    # layout, or broadcast ourselves to the output shape first.
    if isinstance(self.layout, WGSplatFragLayout):
      output_shape = self.shape
      for i, o in enumerate(other):
        if not isinstance(o, FragmentedArray):
          continue
        elif not isinstance(o.layout, WGSplatFragLayout):
          return o._pointwise(
              lambda o, this, *args: op(this, *args[:i], o, *args[i:]),
              self,
              *other[:i],
              *other[i + 1 :],
              output_is_signed=output_is_signed,
          )
        else:
          output_shape = np.broadcast_shapes(output_shape, o.shape)
      # If we get here then we haven't found any non-splat layout.
      if self.shape != output_shape:
        return self.broadcast(output_shape)._pointwise(
            op, *other, output_is_signed=output_is_signed
        )

    other_arrs = []
    for o in other:
      if not isinstance(o, FragmentedArray):
        if isinstance(o, (float, int)):
          o = utils.c(o, self.mlir_dtype)
        elif not isinstance(o, ir.Value):
          raise NotImplementedError(o)

        o = FragmentedArray.splat(
            o, shape=self.shape, layout=self.layout, is_signed=self.is_signed
        )

      if isinstance(o.layout, WGSplatFragLayout):
        if not o.layout.can_broadcast_to(self.shape):
          raise ValueError(
              f"Cannot broadcast shape {self.shape} to layout {o.layout}")
        o = FragmentedArray.splat(
            o.registers.flat[0],
            shape=self.shape,
            layout=self.layout,
            is_signed=o.is_signed,
        )
      else:
        if self.layout != o.layout:
          raise ValueError("Incompatible FragmentedArray layouts")
        if self.registers.shape != o.registers.shape:
          raise ValueError("Incompatible FragmentedArray shapes")

      other_arrs.append(o)
    new_regs = np.empty_like(self.registers)

    for idx, reg in np.ndenumerate(self.registers):
      new_regs[idx] = op(reg, *(o.registers[idx] for o in other_arrs))
    reg_ty = new_regs.flat[0].type
    if isinstance(reg_ty, ir.VectorType):
      reg_ty = ir.VectorType(reg_ty).element_type
    if output_is_signed is None and isinstance(reg_ty, ir.IntegerType):
      output_is_signed = self.is_signed
    return FragmentedArray(
        _registers=new_regs, _layout=self.layout, _is_signed=output_is_signed
    )

  def __pos__(self):
    return self

  def __neg__(self):
    if isinstance(self.mlir_dtype, ir.FloatType):
      return self._pointwise(arith.negf)
    elif isinstance(self.mlir_dtype, ir.IntegerType):
      return 0 - self
    else:
      return NotImplemented

  def __add__(self, other):
    if isinstance(self.mlir_dtype, ir.FloatType):
      return self._pointwise(addf, other)
    elif isinstance(self.mlir_dtype, ir.IntegerType):
      return self._pointwise(arith.addi, other)
    else:
      return NotImplemented

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    if isinstance(self.mlir_dtype, ir.FloatType):
      return self._pointwise(mulf, other)
    elif isinstance(self.mlir_dtype, ir.IntegerType):
      return self._pointwise(arith.muli, other)
    else:
      return NotImplemented

  def __rmul__(self, other):
    return self * other

  def __sub__(self, other):
    if isinstance(self.mlir_dtype, ir.FloatType):
      return self._pointwise(subf, other)
    elif isinstance(self.mlir_dtype, ir.IntegerType):
      return self._pointwise(arith.subi, other)
    else:
      return NotImplemented

  def __rsub__(self, other):
    if isinstance(self.mlir_dtype, ir.FloatType):
      return self._pointwise(lambda s, o: subf(o, s), other)
    elif isinstance(self.mlir_dtype, ir.IntegerType):
      return self._pointwise(lambda s, o: arith.subi(o, s), other)
    else:
      return NotImplemented

  def __truediv__(self, other):
    if not isinstance(self.mlir_dtype, ir.FloatType):
      return NotImplemented
    return self._pointwise(arith.divf, other)

  def __rtruediv__(self, other):
    if not isinstance(self.mlir_dtype, ir.FloatType):
      return NotImplemented
    if isinstance(self.mlir_dtype, ir.Float8E8M0FNUType) and other == 1:
      def e8m0_inv(x, _):
        if not isinstance(x.type, ir.VectorType):
          raise NotImplementedError(x.type)
        [vec_len] = ir.VectorType(x.type).shape
        i8 = ir.IntegerType.get_signless(8)
        i8_vec = ir.VectorType.get((vec_len,), i8)
        c254 = vector.broadcast(i8_vec, arith.constant(i8, 254))
        return utils.bitcast(arith.subi(c254, utils.bitcast(x, i8_vec)), x.type)
      return self._pointwise(e8m0_inv, other, restrict_bitwidth=False)
    return self._pointwise(lambda s, o: arith.divf(o, s), other)

  def __floordiv__(self, other):
    if isinstance(self.mlir_dtype, ir.FloatType):
      return self._pointwise(
          lambda s, o: mlir_math.floor(arith.divf(s, o)), other
      )
    elif isinstance(self.mlir_dtype, ir.IntegerType):
      if self.is_signed:
        return self._pointwise(arith.floordivsi, other)
      else:
        return self._pointwise(arith.divui, other)
    else:
      return NotImplemented

  def __rfloordiv__(self, other):
    if isinstance(self.mlir_dtype, ir.FloatType):
      return self._pointwise(
          lambda s, o: mlir_math.floor(arith.divf(o, s)), other
      )
    elif isinstance(self.mlir_dtype, ir.IntegerType):
      if self.is_signed:
        return self._pointwise(lambda s, o: arith.floordivsi(o, s), other)
      else:
        return self._pointwise(lambda s, o: arith.divui(o, s), other)
    else:
      return NotImplemented

  def __mod__(self, other):
    if not isinstance(self.mlir_dtype, ir.IntegerType):
      return NotImplemented
    if self.is_signed:
      return self._pointwise(arith.remsi, other)
    else:
      return self._pointwise(arith.remui, other)

  def __rmod__(self, other):
    if not isinstance(self.mlir_dtype, ir.IntegerType):
      return NotImplemented
    if self.is_signed:
      return self._pointwise(lambda s, o: arith.remsi(o, s), other)
    else:
      return self._pointwise(lambda s, o: arith.remui(o, s), other)

  def __invert__(self):
    if not isinstance(self.mlir_dtype, ir.IntegerType):
      return NotImplemented
    return self ^ ~0

  def __or__(self, other):
    if not isinstance(self.mlir_dtype, ir.IntegerType):
      return NotImplemented
    return self._pointwise(arith.ori, other, restrict_bitwidth=False)

  def __ror__(self, other):
    return self | other

  def __and__(self, other):
    if not isinstance(self.mlir_dtype, ir.IntegerType):
      return NotImplemented
    return self._pointwise(arith.andi, other, restrict_bitwidth=False)

  def __rand__(self, other):
    return self & other

  def __xor__(self, other):
    if not isinstance(self.mlir_dtype, ir.IntegerType):
      return NotImplemented
    return self._pointwise(arith.xori, other, restrict_bitwidth=False)

  def __rxor__(self, other):
    return self ^ other

  def __lshift__(self, other):
    if not isinstance(self.mlir_dtype, ir.IntegerType):
      return NotImplemented
    return self._pointwise(arith.shli, other, restrict_bitwidth=False)

  def __rshift__(self, other):
    if not isinstance(self.mlir_dtype, ir.IntegerType):
      return NotImplemented
    return self._pointwise(
        arith.shrsi if self.is_signed else arith.shrui,
        other,
        restrict_bitwidth=False,
    )

  def __eq__(self, other):
    return self._compare(
        other,
        f_pred=arith.CmpFPredicate.OEQ,
        si_pred=arith.CmpIPredicate.eq,
        ui_pred=arith.CmpIPredicate.eq,
        restrict_bitwidth=False,
    )

  def __ne__(self, other):
    return self._compare(
        other,
        f_pred=arith.CmpFPredicate.UNE,
        si_pred=arith.CmpIPredicate.ne,
        ui_pred=arith.CmpIPredicate.ne,
        restrict_bitwidth=False,
    )

  def __lt__(self, other):
    return self._compare(
        other,
        f_pred=arith.CmpFPredicate.OLT,
        si_pred=arith.CmpIPredicate.slt,
        ui_pred=arith.CmpIPredicate.ult,
    )

  def __le__(self, other):
    return self._compare(
        other,
        f_pred=arith.CmpFPredicate.OLE,
        si_pred=arith.CmpIPredicate.sle,
        ui_pred=arith.CmpIPredicate.ule,
    )

  def __gt__(self, other):
    return self._compare(
        other,
        f_pred=arith.CmpFPredicate.OGT,
        si_pred=arith.CmpIPredicate.sgt,
        ui_pred=arith.CmpIPredicate.ugt,
    )

  def __ge__(self, other):
    return self._compare(
        other,
        f_pred=arith.CmpFPredicate.OGE,
        si_pred=arith.CmpIPredicate.sge,
        ui_pred=arith.CmpIPredicate.uge,
    )

  def _compare(
      self, other, *, f_pred, si_pred, ui_pred, restrict_bitwidth=True
  ):
    if isinstance(self.mlir_dtype, ir.FloatType):
      pred = functools.partial(arith.cmpf, f_pred)
    elif isinstance(self.mlir_dtype, ir.IntegerType):
      if self.is_signed:
        pred = functools.partial(arith.cmpi, si_pred)
      else:
        pred = functools.partial(arith.cmpi, ui_pred)
    else:
      return NotImplemented
    return self._pointwise(
        pred, other, output_is_signed=False, restrict_bitwidth=restrict_bitwidth
    )

  def max(self, other) -> FragmentedArray:
    if isinstance(self.mlir_dtype, ir.FloatType):
      maximumf = arith.maximumf
      if isinstance(self.mlir_dtype, ir.F32Type):
        maximumf = self._lift_fast_instr("max.NaN.f32")
      elif isinstance(self.mlir_dtype, ir.F16Type):
        maximumf = self._lift_fast_packed_instr("max.NaN.f16x2", "max.NaN.f16")
      elif isinstance(self.mlir_dtype, ir.BF16Type):
        maximumf = self._lift_fast_packed_instr("max.NaN.bf16x2", "max.NaN.bf16")
      return self._pointwise(maximumf, other)
    elif isinstance(self.mlir_dtype, ir.IntegerType):
      width = utils.bitwidth(self.mlir_dtype)
      if width == 16:
        sign = "s" if self.is_signed else "u"
        instr = self._lift_fast_packed_instr(f"max.{sign}16x2", f"max.{sign}16")
        return self._pointwise(instr, other)
      return self._pointwise(
          arith.maxsi if self.is_signed else arith.maxui, other
      )
    else:
      raise NotImplementedError

  def min(self, other) -> FragmentedArray:
    if isinstance(self.mlir_dtype, ir.FloatType):
      minimumf = arith.minimumf
      if isinstance(self.mlir_dtype, ir.F32Type):
        minimumf = self._lift_fast_instr("min.NaN.f32")
      elif isinstance(self.mlir_dtype, ir.F16Type):
        minimumf = self._lift_fast_packed_instr("min.NaN.f16x2", "min.NaN.f16")
      elif isinstance(self.mlir_dtype, ir.BF16Type):
        minimumf = self._lift_fast_packed_instr("min.NaN.bf16x2", "min.NaN.bf16")
      return self._pointwise(minimumf, other)
    elif isinstance(self.mlir_dtype, ir.IntegerType):
      width = utils.bitwidth(self.mlir_dtype)
      if width == 16:
        sign = "s" if self.is_signed else "u"
        instr = self._lift_fast_packed_instr(f"min.{sign}16x2", f"min.{sign}16")
        return self._pointwise(instr, other)
      return self._pointwise(
          arith.minsi if self.is_signed else arith.minui, other
      )
    else:
      raise NotImplementedError

  def copysign(self, other: FragmentedArray) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError
    return self._pointwise(mlir_math.copysign, other)

  def exp(self, *, approx: bool = False) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError
    if approx:
      dtype = self.mlir_dtype
      log2e = arith.constant(dtype, ir.FloatAttr.get(dtype, 1.4426950408889634))
      return cast(FragmentedArray, self * log2e).exp2()
    return self._pointwise(mlir_math.exp)

  def exp2(self, *, approx: bool = False) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError
    if approx:
      if not isinstance(self.mlir_dtype, ir.F32Type):
        raise NotImplementedError(self.mlir_dtype)
      return self._pointwise(self._lift_fast_instr("ex2.approx.ftz.f32"))
    return self._pointwise(mlir_math.exp2)

  def log(self, *, approx: bool = False) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError
    if approx:
      dtype = self.mlir_dtype
      ln2 = arith.constant(dtype, ir.FloatAttr.get(dtype, 0.6931471805599453))
      return self.log2(approx=True) * ln2
    return self._pointwise(mlir_math.log)

  def log2(self, *, approx: bool = False) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError(self.mlir_dtype)
    if approx:
      if not isinstance(self.mlir_dtype, ir.F32Type):
        raise NotImplementedError(self.mlir_dtype)
      return self._pointwise(self._lift_fast_instr("lg2.approx.ftz.f32"))
    return self._pointwise(mlir_math.log2)

  def sin(self, *, approx: bool = False) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError
    if approx and self.mlir_dtype != ir.F32Type.get():
      raise NotImplementedError
    return self._pointwise(
        self._lift_fast_instr("sin.approx.f32") if approx else mlir_math.sin
    )

  def cos(self, *, approx: bool = False) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError
    if approx and self.mlir_dtype != ir.F32Type.get():
      raise NotImplementedError
    return self._pointwise(
        self._lift_fast_instr("cos.approx.f32") if approx else mlir_math.cos
    )

  def tanh(self, *, approx: bool = False) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError
    if approx and self.mlir_dtype != ir.F32Type.get():
      raise NotImplementedError
    return self._pointwise(
        self._lift_fast_instr("tanh.approx.f32") if approx else mlir_math.tanh
    )

  def rsqrt(self, *, approx: bool = False) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError
    if approx and self.mlir_dtype != ir.F32Type.get():
      raise NotImplementedError
    return self._pointwise(
        self._lift_fast_instr("rsqrt.approx.f32") if approx else mlir_math.rsqrt
    )

  def abs(self) -> FragmentedArray:
    if isinstance(self.mlir_dtype, ir.FloatType):
      return self._pointwise(mlir_math.absf)
    if isinstance(self.mlir_dtype, ir.IntegerType):
      return self._pointwise(mlir_math.absi)
    raise NotImplementedError

  def round(self) -> FragmentedArray:
    """Same as `lax.round(..., AWAY_FROM_ZERO)`."""
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError
    return self._pointwise(mlir_math.round)

  def round_even(self) -> FragmentedArray:
    """Same as `lax.round(..., TO_NEAREST_EVEN)`."""
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError
    return self._pointwise(mlir_math.roundeven)

  def erf(self) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError(self.mlir_dtype)
    return self._pointwise(mlir_math.erf)

  def atan2(self, other: FragmentedArray) -> FragmentedArray:
    if not isinstance(self.mlir_dtype, ir.FloatType):
      raise NotImplementedError(self.mlir_dtype)
    return self._pointwise(mlir_math.atan2, other)

  @staticmethod
  def _lift_fast_instr(
      instr: str | Callable[[ir.Value], ir.Value],
  ) -> Callable[[ir.Value, ir.Value], ir.Value]:
    def fast_instr(*args):
      f32 = ir.F32Type.get()
      arg_ty = args[0].type
      assert all(a.type == arg_ty for a in args)
      if arg_ty == f32:
        if isinstance(instr, str):
          args_ptx = ", ".join(f"${i}" for i in range(len(args) + 1))
          return llvm.inline_asm(
              f32, args, f"{instr} {args_ptx};", "=f" + ",f" * len(args)
          )
        else:
          return instr(*args)
      elif isinstance(arg_ty, ir.VectorType):
        result = llvm.mlir_undef(arg_ty)
        [vec_len] = ir.VectorType(arg_ty).shape
        for i in range(vec_len):
          vs = [
              vector.extract(
                  a,
                  dynamic_position=[],
                  static_position=ir.DenseI64ArrayAttr.get([i]),
              )
              for a in args
          ]
          vr = fast_instr(*vs)
          result = vector.insert(
              vr,
              result,
              dynamic_position=[],
              static_position=ir.DenseI64ArrayAttr.get([i]),
          )
        return result
      else:
        raise NotImplementedError(arg_ty)
    return fast_instr

  @staticmethod
  def _lift_fast_packed_instr(
      packed_instr: str, single_instr: str,
  ) -> Callable[[ir.Value, ir.Value], ir.Value]:
    def fast_instr(*args):
      arg_ty = original_arg_ty = args[0].type
      assert all(a.type == arg_ty for a in args)
      if not isinstance(arg_ty, ir.VectorType):
        args = [vector.broadcast(ir.VectorType.get((1,), arg_ty), a) for a in args]
      arg_ty = ir.VectorType(args[0].type)
      [vec_len] = arg_ty.shape
      vec_bitwidth = vec_len * utils.bitwidth(arg_ty.element_type)
      if vec_len == 1 or vec_bitwidth == 32:
        assert vec_bitwidth.bit_count() == 1
        if vec_bitwidth == 32:
          cstr = "r"
        elif vec_bitwidth == 16:
          cstr = "h"
        else:
          raise NotImplementedError(vec_bitwidth)
        int_ty = ir.IntegerType.get_signless(vec_bitwidth)
        args_ptx = ", ".join(f"${i}" for i in range(len(args) + 1))
        args_int = [utils.bitcast(a, int_ty) for a in args]
        result_int = llvm.inline_asm(
            int_ty,
            args_int,
            f"{single_instr if vec_len == 1 else packed_instr} {args_ptx};",
            f"={cstr}" + f",{cstr}" * len(args)
        )
        return utils.bitcast(result_int, original_arg_ty)
      else:
        assert vec_bitwidth > 32
        slice_len = 32 // utils.bitwidth(arg_ty.element_type)
        offset = 0
        slices = []
        while offset < vec_len:
          slice_end = min(offset + slice_len, vec_len)
          args_slice = [utils.vector_slice(a, slice(offset, slice_end)) for a in args]
          slices.append(fast_instr(*args_slice))
          offset = slice_end
        return utils.vector_concat(slices)
    return fast_instr

  def bitcast(
      self, elt: ir.Type, *, output_is_signed: bool | None = None
  ) -> FragmentedArray:
    if (output_is_signed is not None) != isinstance(elt, ir.IntegerType):
      raise TypeError(
          "output_is_signed must be non-None if and only if the MLIR type is an"
          f" integer type, got {output_is_signed=} for {elt}"
      )

    if elt == self.mlir_dtype:
      return self
    reg_type = self.registers.flat[0].type
    if isinstance(reg_type, ir.VectorType):
      reg_shape = ir.VectorType(reg_type).shape
      ty = ir.VectorType.get(reg_shape, elt)
    else:
      ty = elt

    return self._pointwise(
        lambda x: arith.bitcast(ty, x), output_is_signed=output_is_signed
    )

  def __getitem__(self, idx) -> FragmentedArray:
    base_idx, slice_shape, is_squeezed = utils.parse_indices(idx, self.shape)
    if isinstance(self.layout, WGSplatFragLayout):
      shape = tuple(d for d, s in zip(slice_shape, is_squeezed) if not s)
      return self.splat(self.registers.item(), shape, is_signed=self.is_signed)
    if not isinstance(self.layout, TiledLayout):
      raise NotImplementedError("Only arrays with tiled layouts can be sliced")
    if any(isinstance(idx, ir.Value) for idx in base_idx):
      raise ValueError("Only slicing with static indices allowed")
    if any(is_squeezed):
      raise NotImplementedError("Integer indexing not implemented (only slicing allowed)")
    base_tile_shape = self.layout.base_tile_shape
    if untiled_rank := len(self.shape) - len(base_tile_shape):
      base_tile_shape = (1,) * untiled_rank + base_tile_shape
    if any(b % t for b, t in zip(base_idx, base_tile_shape, strict=True)):
      raise ValueError(
          "Base indices of array slices must be aligned to the beginning of a"
          f" tile. The array uses a tiling of {base_tile_shape}, but your base"
          f" indices are {base_idx}. Consider using a different array layout."
      )
    if any(l % t for l, t in zip(slice_shape, base_tile_shape, strict=True)):
      raise ValueError(
          "The slice shape must be a multiple of the tile shape. The array"
          f" uses a tiling of {base_tile_shape}, but your slice shape is"
          f" {slice_shape}. Consider using a different array layout."
      )
    register_slices = tuple(
        slice(b // t, (b + l) // t)
        for b, l, t in zip(base_idx, slice_shape, base_tile_shape, strict=True)
    )
    new_regs = self.registers[register_slices]
    return FragmentedArray(
        _registers=new_regs, _layout=self.layout, _is_signed=self.is_signed
    )

  def __setitem__(self, idx: object, value: FragmentedArray) -> None:
    if not isinstance(value, FragmentedArray):
      raise ValueError(f"Expected a FragmentedArray, got: {value}")
    if not isinstance(self.layout, TiledLayout):
      raise NotImplementedError("Only arrays with tiled layouts can be sliced")
    base_idx, slice_shape, is_squeezed = utils.parse_indices(idx, self.shape)
    if any(isinstance(idx, ir.Value) for idx in base_idx):
      raise ValueError("Only slicing with static indices allowed")
    if any(is_squeezed):
      raise NotImplementedError("Integer indexing not implemented (only slicing allowed)")
    if value.shape != tuple(slice_shape):
      raise ValueError(
          f"Slice has shape {tuple(slice_shape)}, but assigned array has shape"
          f" {value.shape}"
      )
    if value.mlir_dtype != self.mlir_dtype:
      raise ValueError(
          f"Array has dtype {value.mlir_dtype}, but assigned array has dtype"
          f" {self.mlir_dtype}"
      )
    if value.layout != self.layout:
      raise ValueError(
          f"Array has layout {value.layout}, but assigned array has layout"
          f" {self.layout}"
      )
    base_tile_shape = self.layout.base_tile_shape
    if len(base_tile_shape) != len(self.shape):
      raise NotImplementedError("Tiling has different rank than array")
    if any(
        b % t or l % t
        for b, l, t in zip(base_idx, slice_shape, base_tile_shape, strict=True)
    ):
      raise NotImplementedError("Only tile aligned slicing supported")
    register_slices = tuple(
        slice(b // t, (b + l) // t)
        for b, l, t in zip(base_idx, slice_shape, base_tile_shape, strict=True)
    )
    assert self.registers[register_slices].shape == value.registers.shape
    self.registers[register_slices] = value.registers

  def copy(self) -> FragmentedArray:
    return FragmentedArray(
        _registers=np.copy(self.registers),
        _layout=self.layout,
        _is_signed=self.is_signed,
    )

  # TODO(apaszke): Support JAX dtypes here as well?
  def astype(
      self, new_dtype: ir.Type, *, is_signed: bool | None = None
  ) -> FragmentedArray:
    i4 = ir.IntegerType.get_signless(4)
    i8 = ir.IntegerType.get_signless(8)
    i16 = ir.IntegerType.get_signless(16)
    i32 = ir.IntegerType.get_signless(32)
    f32 = ir.F32Type.get()
    f16 = ir.F16Type.get()
    bf16 = ir.BF16Type.get()
    f8e4m3fn = ir.Float8E4M3FNType.get()
    f8e5m2 = ir.Float8E5M2Type.get()
    f8e8m0fnu = ir.Float8E8M0FNUType.get()

    cur_dtype = self.mlir_dtype
    if cur_dtype == new_dtype:
      if self.is_signed == is_signed:
        return self
      return FragmentedArray(
          _registers=self.registers, _layout=self.layout, _is_signed=is_signed
      )
    # Otherwise, mypy is unhappy with using ``idx`` for both range and
    # np.ndenumerate.
    idx: Any
    any_reg = self.registers.flat[0]
    reg_type = any_reg.type
    is_vector_reg = isinstance(reg_type, ir.VectorType)
    reg_shape = tuple(ir.VectorType(reg_type).shape) if is_vector_reg else (1,)
    [vector_len] = reg_shape  # This is meant to be a 1D assertion.
    if (new_reg_bitwidth := utils.bitwidth(new_dtype) * vector_len) % 8:
      raise ValueError(
          "Register bitwidth in target type must be divisible by 8, got"
          f" {new_reg_bitwidth}"
      )
    # If the vector originates from a slice (common after relayouts), we
    # can fuse the slicing into the conversion and reuse many
    # preprocessing ops (shifts, prmts) accross different vectors.
    regs_from_32bit_slice = (
        isinstance(
            _slice_op := getattr(any_reg.owner, "opview", None),
            vector.ExtractStridedSliceOp,
        )
        and utils.bitwidth(_slice_op.source.type) == 32
        and _slice_op.strides[0].value == 1
    )
    def packed_registers(
        dst_vector_len: int, *, if_not_sliced: bool
    ) -> Iterable[tuple[Sequence[tuple[int, ...]], ir.Value]]:
      """Tries to pack registers up to destination vector length."""
      if regs_from_32bit_slice and if_not_sliced:
        for idx, reg in np.ndenumerate(self.registers):
          yield [idx], reg
        return
      generator = np.ndenumerate(self.registers)
      indices = []
      regs = []
      while True:
        try:
          for _ in range(max(dst_vector_len // vector_len, 1)):
            idx, reg = next(generator)
            indices.append(idx)
            regs.append(reg)
          yield indices, utils.vector_concat(regs)
          regs.clear()
          indices.clear()
        except StopIteration:
          break
      if regs:
        yield indices, utils.vector_concat(regs)

    if cur_dtype == i4 and new_dtype == f8e4m3fn:
      # The algorithm here is taken from CUTLASS's `NumericArrayConverter`
      # specialization for int4 -> f8e4m3, available at
      # https://github.com/NVIDIA/cutlass/blob/5c6bca04414e06ce74458ab0a2018e2b8272701c/include/cutlass/numeric_conversion.h#L4982.
      # Each call to the function below will upcast 4 contiguous nibbles of
      # the input 32-bit register, and whether to select the 4 low nibbles or
      # the 4 high nibbles is determined by the `part` argument.
      def upcast_to_f8e4m3fn(reg: ir.Value, part: int):
        lut = [
            0x44403800,  # [0, 1, 2, 3] encoded as f8e4m3fn
            0x4E4C4A48,  # [4, 5, 6, 7] encoded as f8e4m3fn
            0xCACCCED0,  # [-8, -7, -6, -5] encoded as f8e4m3fn
            0xB8C0C4C8,  # [-4, -3, -2, -1] encoded as f8e4m3fn
        ]

        sign = arith.shrui(arith.andi(reg, c(0x88888888, i32)), c(1, i32))
        # Ignore the sign when indexing into the LUT.
        lut_idx = arith.andi(reg, c(0x77777777, i32))

        assert 0 <= part < 2
        if part == 1:
          lut_idx = arith.shrui(lut_idx, c(16, i32))
          sign = arith.shrui(sign, c(16, i32))

        prmt_sign_pattern = arith.ori(sign, c(0x32103210, i32))
        return llvm.inline_asm(
            i32,
            [lut_idx, prmt_sign_pattern],
            f"""
            {{
            .reg .b32 pos_f8s, neg_f8s;
            prmt.b32 pos_f8s, {lut[0]}, {lut[1]}, $1;
            prmt.b32 neg_f8s, {lut[2]}, {lut[3]}, $1;
            prmt.b32 $0, pos_f8s, neg_f8s, $2;
            }}
            """,
            "=r,r,r",
        )
      new_registers = np.empty_like(self.registers)

      # TODO(apaszke,bchetioui): Using 8 helps some (but not all) cases.
      # TODO(apaszke,bchetioui): Add the slice optimization here.
      packing_width = 8 if vector_len == 2 else 4
      for indices, reg in packed_registers(packing_width, if_not_sliced=False):
        [group_size] = ir.VectorType(reg.type).shape
        assert group_size % vector_len == 0
        int_ty = ir.IntegerType.get_signless(group_size * 4)
        reg_as_i32 = utils.bitcast(reg, int_ty)
        if int_ty != i32:
          reg_as_i32 = arith.extsi(i32, reg_as_i32)
        out_i32_regs = [
            upcast_to_f8e4m3fn(reg_as_i32, part=part)
            for part in range(max(group_size // 4, 1))
        ]
        out_vec_int = utils.vector_concat([
            vector.broadcast(ir.VectorType.get((1,), i32), out_i32_reg)
            for out_i32_reg in out_i32_regs
        ])
        out_vector_len = len(out_i32_regs) * 4
        # Bitcast to i8 first to allow slicing as necessary, since LLVM chokes
        # on f8 types.
        out_vec = utils.bitcast(
            out_vec_int, ir.VectorType.get((out_vector_len,), i8)
        )
        offset = 0
        for idx in indices:
          sliced_out_vec = utils.vector_slice(
              out_vec, slice(offset, offset + vector_len)
          )
          new_registers[idx] = utils.bitcast(
              sliced_out_vec, ir.VectorType.get((vector_len,), f8e4m3fn)
          )
          offset += vector_len
      return FragmentedArray(
          _registers=new_registers, _layout=self.layout, _is_signed=None
      )
    if cur_dtype == i4 and self.is_signed and new_dtype == bf16 and vector_len % 2 == 0:
      new_registers = np.empty_like(self.registers)
      out_vec_ty = ir.VectorType.get((vector_len,), new_dtype)
      # We use packed_registers for consistency, even though the packing is not
      # really profitable here: the PTX below begins by an op dependent on the
      # extracted part and so there are no ops that can be shared across packed
      # parts.
      for indices, reg in packed_registers(2, if_not_sliced=True):
        # The algorithm here is largely the same as CUTLASS's
        # NumericArrayConverter specialization for int4 -> bf16 casts.
        # We modify it slightly, because we only extract 2 values.
        # We first shift the value by 4 bits, to put the high int4 in low bits.
        # The prmt then blends the two values together, by putting them into the
        # low bits of each 16-bit subword of our register. Then, we use the lop3
        # to zero any bits that don't belong to our int4s, and finally use the
        # XOR to: (1) set the exponent bits to 0x43 (at which point the mantissa
        # represents integer increments) and (2) flip the sign bit. If we
        # interpret the 4 bits as uint4 after the flip, then we'll see that
        # positive int4s will end up larger than negative int4s, with a bias of
        # 8. Use use the sub to subtract the base (our initial exponent) and the
        # bias coming from flipping the sign bit which is 136 (0x4308 as bits).
        def upcast_i4_to_bf16(reg: ir.Value, reg_shr: ir.Value, part: int):
          assert 0 <= part < 4
          int_reg = llvm.inline_asm(
              i32,
              [reg, reg_shr],
              f"""
              {{
              .reg .b32 s<4>;
              prmt.b32 s1, $1, $2, 0xF{part + 4}F{part};
              lop3.b32 s2, s1, 0x000F000F, 0x43084308, (0xf0 & 0xcc) ^ 0xaa;
              mov.b32 s3, 0x43084308;
              sub.bf16x2 $0, s2, s3;
              }}
              """,
              "=r,r,r",
          )
          return utils.bitcast(int_reg, ir.VectorType.get((2,), bf16))
        [group_size] = ir.VectorType(reg.type).shape
        assert group_size % vector_len == 0
        assert group_size * 4 <= 32
        int_ty = ir.IntegerType.get_signless(group_size * 4)
        # If the vector originates from a slice (common after relayouts), we
        # can fuse the slicing into the conversion and prevent LLVM from
        # generating a bunch of shifts to align the vector data to the LSB.
        # This also lets us share the right shift among more vectors.
        out_int_regs: list[ir.Value] = []
        if regs_from_32bit_slice:
          slice_op = reg.owner
          slice_offset = slice_op.offsets[0].value
          reg_int = utils.bitcast(slice_op.source, i32)
          reg_int_shr = arith.shrui(reg_int, c(4, i32))
          assert slice_offset % 2 == 0
          out_int_regs.extend(
              upcast_i4_to_bf16(reg_int, reg_int_shr, part=slice_offset // 2 + part)
              for part in range(group_size // 2)
          )
        else:
          reg_slice_int = utils.bitcast(reg, int_ty)
          if int_ty != i32:
            reg_slice_int = arith.extsi(i32, reg_slice_int)
          reg_slice_int_shr = arith.shrui(reg_slice_int, c(4, i32))
          out_int_regs.extend(
              upcast_i4_to_bf16(reg_slice_int, reg_slice_int_shr, part=part)
              for part in range(group_size // 2)
          )
        out_reg = utils.vector_concat(out_int_regs)
        offset = 0
        for idx in indices:
          new_registers[idx] = new_reg = utils.vector_slice(
              out_reg, slice(offset, offset + vector_len)
          )
          offset += vector_len
          assert new_reg.type == out_vec_ty
      return FragmentedArray(
          _registers=new_registers, _layout=self.layout, _is_signed=None
      )
    if cur_dtype == i4 and self.is_signed and new_dtype == i8 and is_signed:
      new_registers = np.empty_like(self.registers)
      out_vec_ty = ir.VectorType.get((vector_len,), new_dtype)
      for indices, reg in packed_registers(8, if_not_sliced=True):
        def upcast_i4_to_i8(reg: ir.Value, first_valid_nibble: int = 0):
          # When first_valid_nibble is >0, then only the nibbles in the range
          # [first_valid_nibble, 8) will be upcast and placed in the low
          # elements of the output vector. All high entries are undefined.
          assert first_valid_nibble % 2 == 0
          low_prmt = "".join(str(min(first_valid_nibble // 2 + i, 7)) for i in [5, 1, 4, 0])
          high_prmt = "".join(str(min(first_valid_nibble // 2 + i, 7)) for i in [7, 3, 6, 2])
          # Note: (0xf0 & 0xaa) | (0xcc & ~0xaa) = 0xe4. lop3 acts as a blend.
          # Below xN means the value of nibble N, sN means that all 4 bits are
          # equal to the sign bit of nibble N, and 00 means an all 0 nibble.
          out_struct = llvm.inline_asm(
              ir.Type.parse("!llvm.struct<(i32, i32)>"),
              [reg],
              f"""
              {{
              .reg .b32 high_even;  // $2 is high_odd
              .reg .b32 low_odd;    // $2 is low_even
              .reg .b32 sign_even, sign_odd;
              .reg .b32 i8_odd, i8_even;
              shl.b32 high_even, $2, 4;                              // x6x5x4x3x2x1x000
              prmt.b32 sign_even, high_even, high_even, 0xba98;      // s6s6s4s4s2s2s0s0
              prmt.b32 sign_odd, $2, $2, 0xba98;                     // s7s7s5s5s3s3s1s1
              shr.u32 low_odd, $2, 4;                                // 00x7x6x5x4x3x2x1
              lop3.b32 i8_odd, sign_odd, low_odd, 0xf0f0f0f0, 0xe4;  // s7x7s5x5s3x3s1x1
              lop3.b32 i8_even, sign_even, $2, 0xf0f0f0f0, 0xe4;     // s6x6s4x4s2x2s0x0
              prmt.b32 $0, i8_even, i8_odd, 0x{low_prmt};            // s3x3s2x2s1x2s0x0
              prmt.b32 $1, i8_even, i8_odd, 0x{high_prmt};           // s7x7s6x5s4x4s3x3
              }}
              """,
              "=r,=r,r",
          )
          i8_vec = ir.VectorType.get((4,), i8)
          return utils.vector_concat([
              utils.bitcast(llvm.extractvalue(i32, out_struct, (i,)), i8_vec)
              for i in range(2)
          ])
        [group_size] = ir.VectorType(reg.type).shape
        assert group_size % vector_len == 0
        assert group_size * 4 <= 32
        int_ty = ir.IntegerType.get_signless(group_size * 4)
        if regs_from_32bit_slice:
          slice_op = reg.owner
          slice_offset = slice_op.offsets[0].value
          reg_int = utils.bitcast(slice_op.source, i32)
          reg_i8 = upcast_i4_to_i8(reg_int, first_valid_nibble=slice_offset)
        else:
          reg_slice_int = utils.bitcast(reg, int_ty)
          if int_ty != i32:
            reg_slice_int = arith.extsi(i32, reg_slice_int)
          reg_i8 = upcast_i4_to_i8(reg_slice_int)

        # distribute packed registers to original indices
        offset = 0
        for idx in indices:
          new_registers[idx] = new_reg = utils.vector_slice(
              reg_i8, slice(offset, offset + vector_len)
          )
          offset += vector_len
          assert new_reg.type == out_vec_ty
      return FragmentedArray(
          _registers=new_registers, _layout=self.layout, _is_signed=is_signed
      )
    if (
        cur_dtype == i8
        and self.is_signed
        and new_dtype == bf16
        and (vector_len == 2 or vector_len % 4 == 0)
    ):
      new_registers = np.empty_like(self.registers)
      def upcast_i8_to_bf16(reg, high):
        # We first embed the s8 into a bf16 with the exponent equal to
        # bias + mantissa bits. Then, we zero the msb that didn't fit into the
        # mantissa, zero out all bits other than msb, and subtract the last
        # two values from each other. This takes advantage of the fact that the
        # lsb of the exponent (msb of the second byte) is zero, which allows us
        # to losslesly pack the msb there. When 1, it doubles the value of s2,
        # making the result negative.
        return llvm.inline_asm(
            i32,
            [reg],
            f"""
            {{
            .reg .b32 s<3>;
            prmt.b32 s0, $1, 0x43, {0x4342 if high else 0x4140};
            and.b32 s1, s0, 0xff7fff7f;
            and.b32 s2, s0, 0xff80ff80;
            sub.bf16x2 $0, s1, s2;
            }}
            """,
            "=r,r",
        )
      empty_vec_32 = llvm.mlir_undef(ir.VectorType.get((vector_len // 2,), i32))
      pad_vec_16 = llvm.mlir_undef(ir.VectorType.get((1,), i16))
      for idx, reg in np.ndenumerate(self.registers):
        if vector_len == 2:
          reg_16 = vector.bitcast(ir.VectorType.get((1,), i16), reg)
          reg_32 = utils.vector_concat([reg_16, pad_vec_16])
          new_reg_32 = upcast_i8_to_bf16(reg_32, high=False)
          new_vec_32 = llvm.insertelement(empty_vec_32, new_reg_32, c(0, i32))
        else:
          assert vector_len % 4 == 0
          vec_32 = vector.bitcast(ir.VectorType.get((vector_len // 4,), i32), reg)
          new_vec_32 = empty_vec_32
          for i in range(vector_len // 4):
            reg_32 = vector.extract(vec_32, [], [i])
            low = upcast_i8_to_bf16(reg_32, high=False)
            high = upcast_i8_to_bf16(reg_32, high=True)
            new_vec_32 = llvm.insertelement(new_vec_32, low, c(2 * i, i32))
            new_vec_32 = llvm.insertelement(new_vec_32, high, c(2 * i + 1, i32))
        new_registers[idx] = vector.bitcast(
            ir.VectorType.get((vector_len,), new_dtype), new_vec_32
        )
      return FragmentedArray(
          _registers=new_registers, _layout=self.layout, _is_signed=is_signed
      )

    # Most f8 casts are done by converting two elements at a time.
    def pairwise_convert(do_convert):
      src_bitwidth = utils.bitwidth(cur_dtype)
      tgt_bitwidth = utils.bitwidth(new_dtype)
      assert tgt_bitwidth <= 16
      src_int_ty = ir.IntegerType.get_signless(src_bitwidth)
      tgt_int_ty = ir.IntegerType.get_signless(tgt_bitwidth)
      tgt_pair_int_ty = ir.IntegerType.get_signless(tgt_bitwidth * 2)
      even_vector_len = vector_len + (vector_len % 2)
      new_registers = np.empty_like(self.registers)
      empty_pair_vec = llvm.mlir_undef(
          ir.VectorType.get((even_vector_len // 2,), tgt_pair_int_ty)
      )
      for idx, reg in np.ndenumerate(self.registers):
        reg = utils.bitcast(reg, ir.VectorType.get((vector_len,), src_int_ty))
        if vector_len % 2:
          reg = utils.vector_concat([reg, llvm.mlir_undef(ir.VectorType.get((1,), src_int_ty))])
        carry_pair_vec = empty_pair_vec
        for base_idx in range(0, even_vector_len, 2):
          pair_vec = utils.vector_slice(reg, slice(base_idx, base_idx + 2))
          new_pair_vec = do_convert(pair_vec)
          carry_pair_vec = llvm.insertelement(carry_pair_vec, new_pair_vec, c(base_idx // 2, i32))
        if vector_len % 2:
          new_reg = vector.bitcast(ir.VectorType.get((even_vector_len,), tgt_int_ty), carry_pair_vec)
          new_reg = utils.vector_slice(new_reg, slice(0, vector_len))
          new_reg = vector.bitcast(ir.VectorType.get((vector_len,), new_dtype), new_reg)
        else:
          new_reg = vector.bitcast(ir.VectorType.get((vector_len,), new_dtype), carry_pair_vec)
        new_registers[idx] = new_reg
      return FragmentedArray(
          _registers=new_registers, _layout=self.layout, _is_signed=is_signed
      )

    # Here we handle all conversions involving f8 types.
    # TODO(apaszke): Figure out proper satfinite and rounding modes.
    supported_f8_f16 = {f8e4m3fn: f16, f8e5m2: f16, f8e8m0fnu: bf16}
    f8_ptx_names = {f8e4m3fn: "e4m3", f8e5m2: "e5m2", f8e8m0fnu: "ue8m0"}
    f16_ptx_names = {f16: "f16", bf16: "bf16"}
    f8_types = f8_ptx_names.keys()
    f16_types = f16_ptx_names.keys()
    if f8e8m0fnu in {cur_dtype, new_dtype} and utils.get_arch().major < 10:
      raise ValueError(
          "f8e8m0fnu type only supported on Blackwell and newer GPUs"
      )
    # f8 <-> f32
    if cur_dtype == f32 and new_dtype in f8_types:
      name_8 = f8_ptx_names[new_dtype]
      rounding = "rz" if new_dtype == f8e8m0fnu else "rn"
      def do_convert(pair_vec):
        e0, e1 = (
            vector.extract(pair_vec, dynamic_position=[], static_position=[i])
            for i in range(2)
        )
        return llvm.inline_asm(
            i16,
            [e1, e0],
            f"cvt.{rounding}.satfinite.{name_8}x2.f32 $0, $1, $2;",
            "=h,r,r",
        )
      return pairwise_convert(do_convert)
    # No f8 type supports direct conversion to f32, so we go via 16-bit floats.
    if cur_dtype in f8_types and new_dtype == f32:
      return self.astype(supported_f8_f16[cur_dtype]).astype(f32)
    # f8 <-> f16
    if new_dtype in f8_types and cur_dtype == supported_f8_f16[new_dtype]:
      name_16 = f16_ptx_names[cur_dtype]
      name_8 = f8_ptx_names[new_dtype]
      rounding = "rz" if new_dtype == f8e8m0fnu else "rn"
      ptx = f"cvt.{rounding}.satfinite.{name_8}x2.{name_16}x2 $0, $1;"
      def do_convert(pair_vec):
        return llvm.inline_asm(i16, [utils.bitcast(pair_vec, i32)], ptx, "=h,r")
      return pairwise_convert(do_convert)
    if cur_dtype in f8_types and new_dtype == supported_f8_f16[cur_dtype]:
      name_8 = f8_ptx_names[cur_dtype]
      name_16 = f16_ptx_names[new_dtype]
      ptx = f"cvt.rn.{name_16}x2.{name_8}x2 $0, $1;"
      def do_convert(pair_vec):
        return llvm.inline_asm(i32, [utils.bitcast(pair_vec, i16)], ptx, "=r,h")
      return pairwise_convert(do_convert)
    # We don't emulate the unsupported f8 <-> f16 conversions, but rather force
    # the user to go via f32 to let them know it's expensive.
    if (new_dtype in f8_types and cur_dtype in f16_types) or (
        new_dtype in f16_types and cur_dtype in f8_types
    ):
      # Remap the 16-bit type to the supported one.
      ok_cur_dtype = supported_f8_f16.get(new_dtype, cur_dtype)
      ok_new_dtype = supported_f8_f16.get(cur_dtype, new_dtype)
      raise NotImplementedError(
          f"Hardware has no support for converting from {cur_dtype} to"
          f" {new_dtype} (only cast from {ok_cur_dtype} to {ok_new_dtype} is"
          " supported). Cast to f32 first and then to the target type"
          " (expensive, but sufficient)."
      )
    # Repack through a shared 16-bit type.
    if cur_dtype in f8_types and new_dtype in f8_types:
      if supported_f8_f16[cur_dtype] == supported_f8_f16[new_dtype]:
        return self.astype(supported_f8_f16[cur_dtype]).astype(new_dtype)
      raise NotImplementedError(
          f"Conversion from {cur_dtype} to {new_dtype} must go through f32,"
          " which is expensive. Cast to f32 explicitly if you really want it."
      )

    # Generic path.
    from_float = isinstance(cur_dtype, ir.FloatType)
    to_float = isinstance(new_dtype, ir.FloatType)
    from_integer = isinstance(cur_dtype, ir.IntegerType)
    to_integer = isinstance(new_dtype, ir.IntegerType)
    from_narrow_float = from_float and utils.bitwidth(cur_dtype) <= 8
    to_narrow_float = to_float and utils.bitwidth(new_dtype) <= 8
    if from_narrow_float or to_narrow_float:
      raise NotImplementedError(
          f"Unsupported conversion involving narrow float types: {cur_dtype}"
          f" to {new_dtype}"
      )
    if from_float and to_float:
      cur_ty_width = ir.FloatType(cur_dtype).width
      new_ty_width = ir.FloatType(new_dtype).width
      if cur_ty_width == new_ty_width:
        # There is no instruction to perform conversions between two float types
        # of the same width. Go through the next-larger standard type.
        # TODO(bchetioui): support conversions between float types of width 8.
        # Which larger type to pick will depend on the number of bits in the
        # smallest exponent.
        if cur_ty_width != 16:
          raise NotImplementedError(
              "Conversion between float types of width other than 16 not"
              " supported"
          )
        larger_ty = ir.F32Type.get()
        match self.layout:
          case WGStridedFragLayout() | TiledLayout():
            shape = ir.VectorType(self.registers.flat[0].type).shape
            upcast_ty = ir.VectorType.get(shape, larger_ty)
          case WGSplatFragLayout():
            upcast_ty = larger_ty
          case _:
            raise NotImplementedError(f"Unsupported layout {self.layout}")
        convert = lambda ty, x: arith.truncf(ty, arith.extf(upcast_ty, x))
      elif cur_ty_width > new_ty_width:
        convert = arith.truncf
      else:
        convert = arith.extf
    elif from_integer and to_integer:
      if ir.IntegerType(cur_dtype).width > ir.IntegerType(new_dtype).width:
        convert = arith.trunci
      else:
        convert = arith.extsi if self.is_signed else arith.extui
    elif from_integer and to_float:
      convert = arith.sitofp if self.is_signed else arith.uitofp
    elif from_float and to_integer:
      convert = arith.fptosi if is_signed else arith.fptoui
    else:
      raise NotImplementedError(f"Unsupported conversion {cur_dtype} -> {new_dtype}")
    new_registers = np.empty_like(self.registers)
    match self.layout:
      case WGStridedFragLayout() | TiledLayout():
        shape = ir.VectorType(self.registers.flat[0].type).shape
        new_reg_ty = ir.VectorType.get(shape, new_dtype)
      case WGSplatFragLayout():
        new_reg_ty = new_dtype
      case _:
        raise NotImplementedError(f"Unsupported layout {self.layout}")
    for idx, reg in np.ndenumerate(self.registers):
      new_registers[idx] = convert(new_reg_ty, reg)
    return FragmentedArray(
        _registers=new_registers, _layout=self.layout, _is_signed=is_signed
    )

  def reduce(
      self,
      op: str | Callable[[ir.Value, ir.Value], ir.Value],
      axis: int | Sequence[int],
      scratch: ir.Value | None = None,
  ) -> FragmentedArray:
    i32 = ir.IntegerType.get_signless(32)
    if isinstance(axis, int):
      axis = (axis,)
    splat_op = None
    redux_op = None
    # TODO(apaszke): For associative reductions that reduce both inside and
    # across warps, we could just have everyone use SMEM atomics instead of
    # performing an explicit warp reduction in registers.
    if isinstance(op, str):
      match op:
        case "add":
          reduced_elems = math.prod(self.shape[a] for a in axis)
          if isinstance(self.mlir_dtype, ir.FloatType):
            op = addf
            splat_op = lambda x: arith.mulf(x, c(reduced_elems, x.type))
            # TODO(apaszke): Use redux.sync on Blackwell for f32.
          elif isinstance(self.mlir_dtype, ir.IntegerType):
            op = arith.addi
            splat_op = lambda x: arith.muli(x, c(reduced_elems, x.type))
            if utils.bitwidth(self.mlir_dtype) == 32:
              redux_op = functools.partial(utils.redux, kind=ReductionKind.ADD)
          else:
            raise NotImplementedError(self.mlir_dtype)
        case "max":
          if isinstance(self.mlir_dtype, ir.F32Type):
            op = self._lift_fast_instr("max.NaN.f32")
            if utils.get_arch().major == 10:
              redux_op = functools.partial(utils.redux, kind=ReductionKind.FMAX)
          elif isinstance(self.mlir_dtype, ir.F16Type):
            op = self._lift_fast_packed_instr("max.NaN.f16x2", "max.NaN.f16")
          elif isinstance(self.mlir_dtype, ir.BF16Type):
            op = self._lift_fast_packed_instr("max.NaN.bf16x2", "max.NaN.bf16")
          elif isinstance(self.mlir_dtype, ir.FloatType):
            op = arith.maximumf
          elif isinstance(self.mlir_dtype, ir.IntegerType):
            op = arith.maxsi if self.is_signed else arith.maxui
            if utils.bitwidth(self.mlir_dtype) == 32:
              kind = ReductionKind.MAX if self.is_signed else ReductionKind.UMAX
              redux_op = functools.partial(utils.redux, kind=kind)
          else:
            raise NotImplementedError(self.mlir_dtype)
          splat_op = lambda x: x
        case "min":
          if isinstance(self.mlir_dtype, ir.F32Type):
            op = self._lift_fast_instr("min.NaN.f32")
            if utils.get_arch().major == 10:
              redux_op = functools.partial(utils.redux, kind=ReductionKind.FMIN)
          elif isinstance(self.mlir_dtype, ir.F16Type):
            op = self._lift_fast_packed_instr("min.NaN.f16x2", "min.NaN.f16")
          elif isinstance(self.mlir_dtype, ir.BF16Type):
            op = self._lift_fast_packed_instr("min.NaN.bf16x2", "min.NaN.bf16")
          elif isinstance(self.mlir_dtype, ir.FloatType):
            op = arith.minimumf
          elif isinstance(self.mlir_dtype, ir.IntegerType):
            op = arith.minsi if self.is_signed else arith.minui
          else:
            raise NotImplementedError(self.mlir_dtype)
          splat_op = lambda x: x
        case "prod":
          reduced_elems = math.prod(self.shape[a] for a in axis)
          if isinstance(self.mlir_dtype, ir.FloatType):
            op = arith.mulf
            # For splat, prod(x, x, ..., x) = x^n
            splat_op = lambda x: mlir_math.powf(
                x, c(float(reduced_elems), x.type)
            )
          elif isinstance(self.mlir_dtype, ir.IntegerType):
            op = arith.muli
            # For splat, use repeated squaring to compute x^n
            def int_pow(x, n=reduced_elems):
              result = c(1, x.type)
              base = x
              while n > 0:
                if n % 2 == 1:
                  result = arith.muli(result, base)
                base = arith.muli(base, base)
                n //= 2
              return result
            splat_op = int_pow
          else:
            raise NotImplementedError(self.mlir_dtype)
        case _:
          raise ValueError(f"Unrecognized reduction operator: {op}")
    assert not isinstance(op, str)
    match self.layout:
      case WGStridedFragLayout(shape=_, vec_size=vec_size):
        if set(axis) != set(range(len(self.shape))):
          raise NotImplementedError(
              "Warpgroup strided layout only support reductions along all axes"
          )
        # We reinterpret the data as a tiled layout. We're reducing it all anyway.
        layout = TiledLayout(
            tiling=Tiling(((128 * vec_size,), (32 * vec_size,), (vec_size,))),
            warp_dims=(-3,),
            lane_dims=(-2,),
            vector_dim=-1,
        )
        return FragmentedArray(
            _registers=self.registers.reshape(
                layout.registers_shape((math.prod(self.shape),))
            ),
            _layout=layout,
            _is_signed=self.is_signed,
        ).reduce(op, 0, scratch)
      case WGSplatFragLayout():
        if splat_op is None:
          raise NotImplementedError(
              "Splat reductions only supported when the operator is a string"
          )
        assert not self.registers.shape
        return FragmentedArray(
            _registers=np.asarray(
                splat_op(self.registers.item()), dtype=object
            ),
            _layout=WGSplatFragLayout(
                tuple(d for a, d in enumerate(self.shape) if a not in axis)
            ),
            _is_signed=self.is_signed,
        )
      case TiledLayout():
        pass
      case _:
        raise NotImplementedError(self.layout)
    # Silence type checker complaints.
    assert isinstance(self.layout, TiledLayout)
    if isinstance(axis, int):
      axis = (axis,)
    layout = self.layout
    untiled_rank = len(self.shape) - len(layout.base_tile_shape)
    tiled_tiling_shape = layout.tiled_tiling_shape
    tiled_axes = tuple(a - untiled_rank for a in axis if a >= untiled_rank)
    tiled_reduced_dims = (False,) * (len(layout.base_tile_shape) + len(tiled_tiling_shape))
    for a in tiled_axes:
      tiled_reduced_dims = tuple(
          r or d for r, d in zip(tiled_reduced_dims, layout.tiling.tile_dimension(a), strict=True)
      )
    reduced_dims = (*(a in axis for a in range(untiled_rank)), *tiled_reduced_dims)
    regs_shape = self.registers.shape
    reduced_shape = tuple(
        d if r else 1 for r, d in zip(reduced_dims, regs_shape, strict=True)
    )
    remaining_shape = tuple(
        1 if r else d for r, d in zip(reduced_dims, regs_shape)
    )
    out_regs = np.empty(remaining_shape, dtype=object)
    index = ir.IndexType.get()

    def reduce_within_warp(out_idx):
      out_reg = None
      for red_idx in np.ndindex(reduced_shape):
        src_idx = tuple(o + r for o, r in zip(out_idx, red_idx))
        if out_reg is None:
          out_reg = self.registers[src_idx]
        else:
          out_reg = op(out_reg, self.registers[src_idx])
      assert out_reg is not None
      # Reduce within the vector dimension, if necessary.
      if reduced_dims[layout.vector_dim]:
        [vec_len] = ir.VectorType(out_reg.type).shape
        scalar_out_reg = None
        for i in range(vec_len):
          scalar = vector.extract(
              out_reg,
              dynamic_position=[],
              static_position=ir.DenseI64ArrayAttr.get([i]),
          )
          scalar_out_reg = (
              scalar if scalar_out_reg is None else op(scalar_out_reg, scalar)
          )
        out_reg = vector.broadcast(
            ir.VectorType.get((1,), out_reg.type.element_type), scalar_out_reg
        )
      # Reduce across warp lanes, if necessary (using warp shuffles).
      if any(reduced_dims[d] for d in layout.partitioned_lane_dims):
        all_lanes = (
            layout.partitioned_lane_dims == layout.lane_dims and
            all(reduced_dims[d] for d in layout.lane_dims)
        )
        # It doesn't make sense to use redux unless we reduce across all lanes.
        # The instruction seems to have a uniform register output.
        if redux_op is not None and all_lanes:
          out_reg = redux_op(out_reg, arith.constant(i32, 0xffffffff))
        else:
          lane_stride = 1
          for d in layout.lane_dims[::-1]:  # Iterate minor-to-major
            if isinstance(d, Replicated):
              lane_stride *= d.times
            elif not reduced_dims[d]:
              lane_stride *= tiled_tiling_shape[d]
            else:
              assert lane_stride.bit_count() == 1
              reduction_size = tiled_tiling_shape[d]
              while reduction_size > 1:
                other_out_reg = utils.shfl_bfly(out_reg, lane_stride)
                out_reg = op(out_reg, other_out_reg)
                lane_stride *= 2
                reduction_size //= 2
          assert lane_stride == WARP_SIZE, lane_stride
      return out_reg

    def swizzle_warp_idx_fn(lane_idx: ir.Value, vec_len: int):
      bitwidth = utils.bitwidth(self.mlir_dtype)
      num_banks = bank_bitwidth = 32
      bitwidth_per_store = vec_len * bitwidth
      num_banks_per_output = WARPS_IN_WARPGROUP * bitwidth_per_store // bank_bitwidth
      # This range supports vector types from vector<1xi8> to
      # {vector<32xi8>, vector<16xi16>, vector<8xi32>}, which should be plenty
      # for realistic use cases. Other cases are not guaranteed to work, but
      # even if they do, their performance hasn't been evaluated. As a result,
      # we prefer failing explicitly, to make sure that we don't end up
      # emitting reductions with poor performance.
      #
      # Note: this implementation is batch invariant, because we use a XOR
      # swizzle, and a binary tree to perform the final reduction. The XOR
      # swizzle only permutes the left and right subtrees, which allows us to
      # always recover the same result (since floating-point operations are
      # commutative).
      if (
          WARPS_IN_WARPGROUP * bitwidth_per_store < bank_bitwidth or
          bitwidth_per_store > 128 or
          num_banks % num_banks_per_output != 0
      ):
        raise NotImplementedError(
            "Unoptimized configuration for cross-warp reduction: "
            f"{self.mlir_dtype} with {vec_len=}"
        )
      # Define one row to be 128 bytes (32 banks of 4 bytes). For a given lane
      # index, we want to store the data coming from all 4 warps
      # contiguously in order to enable vectorized loads later on. If we
      # simply store the data in order of thread_idx naively, this will
      # result in bank conflicts, since each warp will hit only a quarter of
      # the shared memory banks.
      #
      # In order to avoid these bank conflicts, we swizzle the data
      # manually, such that across every four rows of 128 bytes, each warp
      # will hit all the shared memory banks.
      lanes_per_row = num_banks // num_banks_per_output
      num_rows = WARP_SIZE // lanes_per_row
      row_idx = arith.divui(lane_idx, c(lanes_per_row, i32))
      match num_banks_per_output:
        case 1:
          assert num_rows == 1, num_rows
          # Here, each lane stores to a different bank, so we don't need to
          # swizzle the warp index at all.
          swizzle_warp_idx = lambda widx: widx
        case 2:
          assert num_rows == 2, num_rows
          # In this case, each lane stores 16-bit in a single bank, and the
          # stores look like:
          #
          #     |            Bank 0           |            Bank 1           | ...
          #     |    16-bit    |    16-bit    |    16-bit    |    16-bit    | ...
          # r0: | warp0 lane0  | warp1 lane0  | warp2 lane0  | warp3 lane0  | ...
          # r1: | warp2 lane16 | warp3 lane16 | warp0 lane16 | warp1 lane16 | ...
          #
          # such that the first 16 lanes of each warp are mapped to row 0, and
          # the last 16 lanes of each warp are mapped to row 1, and the
          # relative ordering of elements coming from different warps is
          # always the same in each row.
          #
          # We avoid bank conflicts on the two rows by xoring the warp index
          # with 2 on row 1, such that lanes in warp0 and warp1 hit even banks
          # on row 0 and odd banks on row 1 (and the opposite for warp2 and
          # warp3).
          lane_xor = arith.shli(row_idx, c(1, i32))
          swizzle_warp_idx = lambda widx: arith.xori(widx, lane_xor)
        # As long as we use a multiple of 4 banks, we can use the same
        # formulation to swizzle the order of the 4 warps.
        case x if x % 4 == 0:
          # In that case, each lane stores a multiple of 32-bit. The following
          # shows how the stores look like in the 32-bit case:
          #
          # r0: | warp0 lane0  | warp1 lane0  | warp2 lane0  | warp3 lane0  | ...
          # r1: | warp1 lane8  | warp0 lane8  | warp3 lane8  | warp2 lane8  | ...
          # r2: | warp2 lane16 | warp3 lane16 | warp0 lane16 | warp1 lane16 | ...
          # r3: | warp3 lane24 | warp2 lane24 | warp1 lane24 | warp0 lane24 | ...
          #
          # Lanes 0-8 are mapped to row 0, lanes 8-16 to row 1, lanes 16-24 to
          # row 2, and lanes 24-32 to row 3. In each row, the index of the
          # warp is swizzled by xoring it with the index of the row.
          rhs = row_idx if x == 4 else arith.andi(row_idx, c(3, i32))
          swizzle_warp_idx = lambda widx: arith.xori(widx, rhs)
        case _:
          raise NotImplementedError(num_banks_per_output)
      return swizzle_warp_idx

    def store_swizzled(
        reg: ir.Value,
        step_idx: int,
        lane_idx: ir.Value,
        scratch: ir.Value,
        swizzle_warp_idx: Callable[[ir.Value], ir.Value]
    ):
      [vec_len] = ir.VectorType(reg.type).shape
      warp_idx = arith.divui(
          arith.remui(thread_idx, c(WARPGROUP_SIZE, i32)), c(WARP_SIZE, i32)
      )

      step_base_scratch_idx = c(step_idx * WARPGROUP_SIZE, i32)
      lane_base_scratch_idx = arith.addi(
          step_base_scratch_idx, arith.muli(lane_idx, c(WARPS_IN_WARPGROUP, i32))
      )
      store_idx = arith.addi(lane_base_scratch_idx, swizzle_warp_idx(warp_idx))
      as_index = lambda x: arith.index_cast(index, x)
      # TODO(bchetioui): investigate whether adding predication here can yield
      # additional performance improvements. In the case where we have a
      # `Replicated` dimension in there, we will repeat the same store
      # multiple times. Maybe the memory controller resolves this conflict
      # automatically, but we should investigate.
      vector.store(
          reg, scratch, [as_index(arith.muli(store_idx, c(vec_len, i32)))])

    def reduce_stored(
        reg_ty: ir.VectorType,
        step_idx: int,
        lane_idx: ir.Value,
        swizzle_warp_idx: Callable[[ir.Value], ir.Value]
    ):
      [vec_len] = ir.VectorType(reg_ty).shape
      out_reg = None
      step_base_scratch_idx = c(step_idx * WARPGROUP_SIZE, i32)
      lane_base_scratch_idx = arith.addi(
          step_base_scratch_idx, arith.muli(lane_idx, c(WARPS_IN_WARPGROUP, i32))
      )
      # warp_idx & warp_group_mask gives you the reduction group of the current warp.
      if all(isinstance(d, int) and reduced_dims[d] for d in layout.warp_dims):
        # When we load all the data that we have stored, we can omit swizzling
        # the warp index without any loss of correctness or determinism. By
        # relying on the properties of XOR and using a "tree reduction"
        # to reduce the data, we also maintain batch invariance.
        #
        # Without this manual optimization, LLVM can fail to recognize that
        # it can use wider load instructions, leading to worse performance
        # (presumably due to scheduling pressure) and sometimes also due to
        # unnecessary bank conflicts.
        #
        # TODO(bchetioui): there are still load conflicts in the case where
        # `vec_len * WARPS_IN_WARPGROUP * bitwidth` exceeds 128 bits. To
        # avoid bank conflicts in that case, we need to swizzle the loads as
        # well, using a similar pattern as above for groups of 128 bits. A
        # little more care will have to be taken to uphold batch invariance.
        load_ty = ir.VectorType.get((vec_len * WARPS_IN_WARPGROUP,),
                                    reg_ty.element_type)
        load_idx = arith.muli(lane_base_scratch_idx, c(vec_len, i32))
        parts = vector.load(
            load_ty, scratch, [arith.index_cast(index, load_idx)]
        )
        parts = [utils.vector_slice(parts, slice(i * vec_len, (i + 1) * vec_len))
                 for i in range(WARPS_IN_WARPGROUP)]
        out_reg = op(op(parts[0], parts[1]), op(parts[2], parts[3]))
      else:
        # 4 has only two non-trivial prime factors: 2 and 2.
        assert len(layout.warp_dims) == 2
        wd0, wd1 = layout.warp_dims
        # TODO(bchetioui): these paths are optimizable. The above logic is
        # well-suited for loads of values stored by all 4 warps, but we should
        # adapt the store logic to also account for these cases where we only
        # load the value stored by every other warp. In this case, we should
        # use a different swizzle function, in order to make sure we can
        # always get vectorized loads!
        if isinstance(wd0, int) and reduced_dims[wd0]:
          warp_offsets, warp_group_mask = [0, 2], 1
        else:
          assert isinstance(wd1, int) and reduced_dims[wd1]
          warp_offsets, warp_group_mask = [0, 1], 2
        thread_idx = utils.thread_idx()
        warp_idx = arith.divui(
            arith.remui(thread_idx, c(WARPGROUP_SIZE, i32)), c(WARP_SIZE, i32)
        )
        warp_reduction_group = arith.andi(warp_idx, arith.constant(i32, warp_group_mask))
        for warp_offset in warp_offsets:
          reduced_warp = arith.addi(warp_reduction_group, c(warp_offset, i32))
          load_idx = arith.addi(lane_base_scratch_idx, swizzle_warp_idx(reduced_warp))
          part = vector.load(
              reg_ty, scratch,
              [arith.index_cast(index, arith.muli(load_idx, c(vec_len, i32)))]
          )
          out_reg = part if out_reg is None else op(out_reg, part)
      return out_reg

    if reduced_shape:
      vec_len = layout.reduce(tiled_axes).vector_length
    else:
      vec_len = 1

    thread_idx = utils.thread_idx()
    lane_idx = arith.remui(thread_idx, c(WARP_SIZE, i32))

    reduce_across_warps = any(reduced_dims[d] for d in layout.partitioned_warp_dims)
    if reduce_across_warps:
      if scratch is None:
        raise ValueError(
            "scratch must be provided when cross-warp reduction is required"
        )
      scratch_ty = ir.MemRefType(scratch.type)
      if scratch_ty.rank != 1:
        raise ValueError(f"Expected rank 1 for scratch, got {scratch_ty.rank}")
      if scratch_ty.element_type != self.mlir_dtype:
        raise ValueError(
            f"Expected element type {self.mlir_dtype} for scratch, got"
            f" {scratch_ty.element_type}"
        )
      # TODO(apaszke): All lanes that replicate data can share the same scratch.
      # For now we treat the complete reduction as a special case.
      reduces_all_dims = set(axis) == set(range(len(self.shape)))
      unique_lanes = 1 if reduces_all_dims else 32
      scratch_ty = ir.MemRefType(scratch.type)
      scratch_elems_per_register = WARPS_IN_WARPGROUP * unique_lanes * vec_len
      if scratch_ty.shape[0] < scratch_elems_per_register:
        available_bytes = scratch_ty.shape[0] * utils.bitwidth(scratch_ty.element_type) // 8
        required_bytes = scratch_elems_per_register * utils.bitwidth(scratch_ty.element_type) // 8
        raise ValueError(
            f"Required reduction scratch size ({required_bytes} bytes) is "
            f"larger than the available scratch size ({available_bytes} bytes)"
        )
      if scratch_ty.get_strides_and_offset()[0] != [1]:
        raise ValueError("Expected scratch to be contiguous")
      num_concurrent_cross_warp_reductions = scratch_ty.shape[0] // scratch_elems_per_register
      if reduces_all_dims:
        lane_idx = c(0, i32)
      swizzle_warp_idx = swizzle_warp_idx_fn(lane_idx, vec_len)
    else:
      lane_idx = num_concurrent_cross_warp_reductions = swizzle_warp_idx = None

    unreduced_indices: list[tuple[int, ...]] = []
    for out_idx in np.ndindex(remaining_shape):
      out_reg = reduce_within_warp(out_idx)
      reg_ty = ir.VectorType(out_reg.type)
      if reduce_across_warps:
        # TODO(bchetioui): explore pipelining computer+store and loads+reduce
        # by double buffering the scratch. This could offer more
        # instruction-level parallelism.
        step = len(unreduced_indices)
        store_swizzled(out_reg, step, lane_idx, scratch, swizzle_warp_idx)
        unreduced_indices.append(out_idx)
        if len(unreduced_indices) == num_concurrent_cross_warp_reductions:
          utils.warpgroup_barrier()
          for i, unreduced_index in enumerate(unreduced_indices):
            out_regs[unreduced_index] = reduce_stored(
                reg_ty, i, lane_idx, swizzle_warp_idx
            )
          unreduced_indices = []
          utils.warpgroup_barrier()
      else:
        out_regs[out_idx] = out_reg
    if unreduced_indices:
      utils.warpgroup_barrier()
      for i, unreduced_index in enumerate(unreduced_indices):
        out_regs[unreduced_index] = reduce_stored(
            reg_ty, i, lane_idx, swizzle_warp_idx  # pytype: disable=undefined-variable
        )
      utils.warpgroup_barrier()
    del unreduced_indices
    # Infer the output layout and reshape the registers accordingly.
    reduced_logical_shape = list(self.shape)
    for a in sorted(axis, reverse=True):
      del reduced_logical_shape[a]
    if not reduced_logical_shape:  # Complete reduction results in a splat.
      reduced_layout: FragmentedLayout = WGSplatFragLayout(())
      assert out_regs.size == 1
      out_reg = out_regs.flat[0]
      assert ir.VectorType(out_reg.type).shape == [1]
      out_reg = vector.extract(
          out_reg,
          dynamic_position=[],
          static_position=ir.DenseI64ArrayAttr.get([0]),
      )
      out_regs = np.asarray(out_reg, dtype=object)
    else:
      reduced_layout = layout.reduce(tiled_axes)
      out_regs = out_regs.reshape(
          reduced_layout.registers_shape(tuple(reduced_logical_shape))
      )
    return FragmentedArray(
        _registers=out_regs, _layout=reduced_layout, _is_signed=self.is_signed
    )

  def broadcast(self, shape) -> FragmentedArray:
    if isinstance(self.layout, WGStridedFragLayout):
      src_shape, dst_shape = self.layout.shape, shape
      if len(src_shape) > len(dst_shape):
        raise ValueError(
            f"Shape length mismatch. Expected len({src_shape}) <= len({dst_shape})"
        )
      if not all(s == 1 or s == d for s, d in zip(src_shape[::-1], dst_shape[::-1])):
        raise ValueError(
            "Can broadcast if all source dimensions match trailing target"
            " dimensions by being equal or set to 1. Broadcasting from"
            f" {src_shape} to {dst_shape}"
        )
      rank_diff = len(dst_shape) - len(src_shape)
      src_shape = tuple([1] * rank_diff + list(src_shape))

      assert len(src_shape) == len(dst_shape), (src_shape, dst_shape)
      len_suffix = next(
          (i for i in range(len(src_shape)) if src_shape[~i] != dst_shape[~i]),
          len(src_shape)
      )
      if len_suffix > 0 and all(x == 1 for x in src_shape[:-len_suffix]):
        return FragmentedArray(
            _registers=np.tile(self.registers, np.prod(dst_shape[:-len_suffix])),
            _layout=WGStridedFragLayout(shape, self.layout.vec_size),
            _is_signed=self.is_signed,
        )

      raise NotImplementedError(
          "Only major-most broadcast for WGStridedFragLayout is implemented."
          f" Broadcasting from: {src_shape}, to: {dst_shape}."
      )

    if not isinstance(self.layout, WGSplatFragLayout):
      raise NotImplementedError(self.layout)

    if self.shape == shape:
      return self

    if not self.layout.can_broadcast_to(shape):
      raise ValueError(f"Can't broadcast {self.shape} to {shape}")

    return FragmentedArray(
        _registers=self.registers,
        _layout=WGSplatFragLayout(shape),
        _is_signed=self.is_signed,
    )

  def reshape(self, shape: tuple[int, ...]) -> FragmentedArray:
    if self.shape == shape:
      return self
    if math.prod(shape) != math.prod(self.shape):
      raise ValueError(f"Can't reshape {self.shape} to {shape}")

    match self.layout:
      case WGSplatFragLayout() | WGStridedFragLayout():
        new_layout = dataclasses.replace(self.layout, shape=shape)
        return FragmentedArray(
            _registers=self.registers,
            _layout=new_layout,
            _is_signed=self.is_signed,
        )
      case TiledLayout():
        base_tile_shape = self.layout.base_tile_shape
        assert base_tile_shape
        old_shape_suffix = self.shape[-len(base_tile_shape):]
        new_shape_suffix = shape[-len(base_tile_shape):]
        # We already know that old_shape_suffix[0] is divisible by
        # base_tile_shape[0].
        if (
            old_shape_suffix[1:] != new_shape_suffix[1:]
            or new_shape_suffix[0] % base_tile_shape[0]
        ):
          raise ValueError(
              f"Can't reshape {self.shape} to {shape} with a tiled layout with"
              f" base tile of {base_tile_shape}"
          )
        new_registers_shape = self.layout.registers_shape(shape)
        return FragmentedArray(
            _registers=self.registers.reshape(new_registers_shape),
            _layout=self.layout,
            _is_signed=self.is_signed,
        )
      case _:
        raise NotImplementedError(self.layout)

  def broadcast_minor(self, n) -> FragmentedArray:
    if len(self.shape) != 1:
      raise ValueError("Broadcast minor is only supported for 1D arrays")
    if n % 8:
      raise ValueError(f"The broadcast dimension must be a multiple of 8, got {n}")
    if self.layout == WGMMA_ROW_LAYOUT:
      new_layout = WGMMA_LAYOUT
    elif self.layout == TCGEN05_ROW_LAYOUT:
      new_layout = TCGEN05_LAYOUT
    else:
      raise NotImplementedError(self.layout)
    return self.broadcast_in_dim((self.shape[0], n), (0,), new_layout)

  def broadcast_in_dim(
      self, shape, source_dimensions, layout: FragmentedLayout
  ) -> FragmentedArray:
    for i, target_dim in enumerate(source_dimensions):
      if self.shape[i] != shape[target_dim]:
        raise ValueError(
            f"Dimension {i} has size {self.shape[i]} in source shape and"
            f" {shape[target_dim]} in shape after broadcast"
        )
    if isinstance(self.layout, WGSplatFragLayout):
      return type(self).splat(
        self.registers.item(), shape, layout, is_signed=self.is_signed
      )
    if isinstance(self.layout, WGStridedFragLayout) and isinstance(layout, WGStridedFragLayout):
      new_dims = set(range(len(shape))) - set(source_dimensions)
      vec_match = self.layout.vec_size == layout.vec_size
      broadcast_dim_match = new_dims == set(range(len(new_dims)))
      assert layout.shape == shape, (layout.shape, shape)
      if vec_match and broadcast_dim_match:
        return FragmentedArray(
            _registers=np.tile(
                self.registers,
                np.prod(shape[:len(new_dims)]),
            ),
            _layout=layout,
            _is_signed=self.is_signed,
        )
    if not isinstance(self.layout, TiledLayout) or not isinstance(layout, TiledLayout):
      raise NotImplementedError(self.layout, layout)
    if any(d1 >= d2 for d1, d2 in zip(source_dimensions, source_dimensions[1:])):
      raise NotImplementedError("source_dimensions must be strictly increasing")
    if len(layout.base_tile_shape) != len(shape):
      raise NotImplementedError("Tiling rank different than broadcast result rank")
    new_dimensions = sorted(set(range(len(shape))) - set(source_dimensions))
    expected_layout = layout.reduce(new_dimensions)
    if expected_layout != self.layout:
      raise ValueError(
          "Source and destination layouts aren't compatible for a broadcast"
      )
    new_registers_shape = layout.registers_shape(shape)
    pre_broadcast_registers_shape = list(new_registers_shape)
    for new_dim in new_dimensions:
      for i, is_new in enumerate(layout.tiling.tile_dimension(new_dim)):
        if is_new:
          pre_broadcast_registers_shape[i] = 1
    # The broadcast for all dims but the vector_dim amounts to repeating the
    # registers along the new dimensions. Along the vector_dim, we actually need
    # to extend the vector length to change the type of the registers.
    if layout.vector_length != self.layout.vector_length:
      assert self.layout.vector_length == 1
      registers = np.empty_like(self.registers)
      for idx, reg in np.ndenumerate(self.registers):
        registers[idx] = utils.vector_concat([reg] * layout.vector_length)
    else:
      registers = self.registers
    new_registers = np.broadcast_to(
        registers.reshape(pre_broadcast_registers_shape), new_registers_shape,
    )
    return FragmentedArray(
        _registers=new_registers, _layout=layout, _is_signed=self.is_signed,
    )

  def select(self, on_true, on_false):
    if (
        not isinstance(self.mlir_dtype, ir.IntegerType)
        or ir.IntegerType(self.mlir_dtype).width != 1
    ):
      raise NotImplementedError
    # We change the receiver here, because the return type is defined by
    # `on_true` and `on_false` and not the predicate `self`.
    return on_true._pointwise(
        lambda t, p, f: arith.select(p, t, f), self, on_false,
    )

  @classmethod
  def build(
      cls,
      shape: tuple[int, ...],
      layout: FragmentedLayout,
      fn: Callable[..., ir.Value],  # ir.Value varargs, one for each dim
      *,
      is_signed: bool | None = None,
  ) -> FragmentedArray:
    undef = llvm.mlir_undef(ir.IntegerType.get_signless(32))
    dummy = cls.splat(undef, shape, layout, is_signed=False)
    return dummy.foreach(
        lambda _, idx: fn(*idx), create_array=True, is_signed=is_signed
    )

  def foreach(
      self,
      fn: Callable[[ir.Value, tuple[ir.Value, ...]], ir.Value | None],
      *,
      create_array=False,
      is_signed=None,
  ):
    """Call a function for each value and index."""
    index = ir.IndexType.get()
    new_regs = None
    orig_fn = fn
    del fn
    def wrapped_fn(*args):
      nonlocal new_regs
      result = orig_fn(*args)
      old_reg_type = self.registers.flat[0].type
      # Lazily create new_regs once we know the desired output type.
      if create_array and new_regs is None:
        assert result is not None
        if isinstance(old_reg_type, ir.VectorType):
          new_reg_type = ir.VectorType.get(old_reg_type.shape, result.type)
        else:
          new_reg_type = result.type
        new_regs = np.full_like(self.registers, llvm.mlir_undef(new_reg_type))
      return result
    for mlir_idx, reg_idx in zip(self.layout.thread_idxs(self.shape), np.ndindex(self.registers.shape), strict=True):
      reg = self.registers[reg_idx]
      assert len(mlir_idx) == len(self.shape), (mlir_idx, self.shape)
      if isinstance(reg.type, ir.VectorType):
        [elems] = ir.VectorType(reg.type).shape
        for i in range(elems):
          c_i = c(i, index)
          val = wrapped_fn(
              vector.extract(
                  reg,
                  dynamic_position=[],
                  static_position=ir.DenseI64ArrayAttr.get([i]),
              ),
              (*mlir_idx[:-1], arith.addi(mlir_idx[-1], c_i)),
          )
          if create_array:
            assert new_regs is not None
            new_regs[reg_idx] = vector.insert(
                val,
                new_regs[reg_idx],
                dynamic_position=[],
                static_position=ir.DenseI64ArrayAttr.get([i]),
            )
      else:
        val = wrapped_fn(reg, mlir_idx)
        if create_array:
          assert new_regs is not None
          new_regs[reg_idx] = val

    if create_array:
      assert new_regs is not None
      return FragmentedArray(_registers=new_regs, _layout=self.layout, _is_signed=is_signed)

  def debug_print(self, fmt: str) -> None:
    idx_fmt = ", ".join(["{}"] * len(self.shape))
    @self.foreach
    def _(val, idx):
      fmt_str = fmt.format(f"[{idx_fmt}]: {{}}")
      utils.debug_print(fmt_str, *idx, val, uniform=False)

  def store_untiled(
      self, ref: ir.Value | utils.MultimemRef, *, swizzle: int = 16, optimized: bool = True
  ) -> None:
    if not isinstance(ref.type, ir.MemRefType):
      raise ValueError(ref)
    match self.layout:
      case WGSplatFragLayout():
        if isinstance(ref, utils.MultimemRef):
          raise NotImplementedError("Splat layout does not support multimem")
        # All values are the same so swizzle does not affect anything here.
        self._store_untiled_splat(ref)
      case WGStridedFragLayout():
        if swizzle != 16:
          raise ValueError("Only TiledLayouts support swizzling")
        assert isinstance(self.layout, WGStridedFragLayout)
        for get, _update, ref, idx in self.transfer_strided(ref, self.layout.vec_size):
          if isinstance(ref, utils.MultimemRef):
            ref.store(get(self.registers), idx)
          else:
            vector.store(get(self.registers), ref, idx)
      case TiledLayout():
        ref_shape = ir.MemRefType(ref.type).shape
        ref = utils.memref_reshape(ref, (*(1 for _ in ref_shape), *ref_shape))
        self.store_tiled(ref, swizzle=swizzle, optimized=optimized)
      case _:
        raise NotImplementedError(self.layout)

  @classmethod
  def load_reduce_untiled(
      cls,
      ref: utils.MultimemRef,
      layout: TiledLayout | WGStridedFragLayout,
      reduction: utils.MultimemReductionOp,
      swizzle: int = 16,
      is_signed: bool | None = None,
  ):
    ref_ty = ir.MemRefType(ref.type)
    shape = tuple(ref_ty.shape)
    if isinstance(layout, WGStridedFragLayout):
      if swizzle != 16:
        raise ValueError("Only TiledLayouts support swizzling")
      registers = np.empty(layout.registers_shape(shape), dtype=object)
      vec_ty = ir.VectorType.get((layout.vec_size,), ref_ty.element_type)
      for _get, update, ref, idx in cls.transfer_strided(ref, layout.vec_size):
        ptr = utils.memref_ptr(utils.memref_slice(ref.ref, tuple(idx)))
        update(registers, utils.multimem_load_reduce(vec_ty, ptr, reduction, is_signed))
      return cls(_registers=registers, _layout=layout, _is_signed=is_signed)
    ref = utils.memref_reshape(ref, (*(1 for _ in shape), *shape))
    return cls.load_tiled(
        ref.ref,
        swizzle=swizzle,
        is_signed=is_signed,
        layout=layout,
        optimized=False,  # multimem refs are always GMEM refs.
        _load_fun=functools.partial(
            utils.multimem_load_reduce, reduction=reduction, is_signed=is_signed
        ),
        # multimem_load_reduce supports vectors of narrow floats, so we don't
        # need to do any casting.
        _narrow_float_as_int=False,
    )

  @classmethod
  def load_untiled(
      cls,
      ref: ir.Value,
      *,
      layout: TiledLayout,
      swizzle: int = 16,
      is_signed: bool | None = None,
      optimized: bool = True,
  ) -> FragmentedArray:
    ref_ty = ir.MemRefType(ref.type)
    ref = utils.memref_reshape(ref, (*(1 for _ in ref_ty.shape), *ref_ty.shape))
    return cls.load_tiled(
        ref, swizzle=swizzle, is_signed=is_signed, layout=layout, optimized=optimized
    )

  def _store_untiled_splat(self, ref: ir.Value):
    if math.prod(self.shape) == 1:
      c0 = c(0, ir.IndexType.get())
      memref.store(
          self.registers.flat[0], ref, [c0] * len(ir.MemRefType(ref.type).shape)
      )
      return

    vec_size = 64 // mgpu.bitwidth(self.mlir_dtype)
    if np.prod(self.shape) < vec_size * WARPGROUP_SIZE:
      vec_size = 1

    if np.prod(self.shape) % WARPGROUP_SIZE * vec_size:
      raise NotImplementedError(
          "Arrays with the splat layout can only be stored when they have a"
          f" single element or a multiple of {WARPGROUP_SIZE} elements"
      )

    fa = FragmentedArray.splat(
        self.registers.flat[0],
        self.shape,
        layout=WGStridedFragLayout(shape=self.shape, vec_size=vec_size),
        is_signed=self.is_signed,
    )
    fa.store_untiled(ref)

  def store_tiled_async(
      self,
      ref: ir.Value,
      barrier: utils.BarrierRef,
      cluster_dim: gpu.Dimension,
      cluster_idx: ir.Value,
      swizzle: int | None,
      optimized: bool = True,
      tiling_rank: int | None = None,
  ):
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    if isinstance(ref, utils.MultimemRef):
      raise ValueError("Multimem refs are not supported in store_tiled_async")
    layout, shape = self.layout, self.shape
    if not isinstance(layout, TiledLayout):
      raise NotImplementedError(self.layout)
    if any(
        isinstance(d, Replicated)
        for d in itertools.chain(layout.warp_dims, layout.lane_dims)
    ):
      raise NotImplementedError("Replicated dimensions are not supported")
    full_cluster_idx = [gpu.cluster_block_id(d) for d in gpu.Dimension]
    full_cluster_idx[cluster_dim] = cluster_idx
    lin_cluster_idx = arith.index_cast(i32, utils.cluster_idx(gpu.Dimension, full_cluster_idx))
    cluster_barrier_ptr = utils.get_cluster_ptr(
        barrier.get_ptr(), lin_cluster_idx, generic=False
    )
    cluster_ref = utils.get_cluster_ref(
        ref, cluster_dim, cluster_idx, generic=False
    )
    stores = self.transfer_tiled(
        cluster_ref, swizzle, layout, shape, optimized, ref_tiling_rank=tiling_rank
    )
    for get, _update, _idx, cluster_ptr in stores:
      reg = get(self.registers)
      reg_ty = ir.VectorType(reg.type)
      element_bitwidth = utils.bitwidth(reg_ty.element_type)
      if (
          isinstance(reg_ty.element_type, ir.FloatType)
          and element_bitwidth <= 8
      ):
        narrow_int = ir.IntegerType.get_signless(element_bitwidth)
        reg = vector.bitcast(ir.VectorType.get(reg_ty.shape, narrow_int), reg)
      reg_bitwidth = utils.bitwidth(reg_ty)
      if reg_bitwidth == 32:
        ptx_constraint = "r"
        ptx_type = "b32"
        reg = utils.bitcast(reg, i32)
      elif reg_bitwidth == 64:
        ptx_constraint = "l"
        ptx_type = "b64"
        reg = utils.bitcast(reg, i64)
      else:
        raise NotImplementedError(f"Unsupported register bitwidth: {reg_bitwidth}")
      llvm.inline_asm(
          ir.Type.parse("!llvm.void"),
          [cluster_ptr, reg, cluster_barrier_ptr],
          f"st.async.cluster.shared::cluster.mbarrier::complete_tx::bytes.{ptx_type} [$0], $1, [$2];",
          f"l,{ptx_constraint},l",
          has_side_effects=True,
      )

  def store_tiled(
      self,
      ref: ir.Value | utils.MultimemRef,
      swizzle: int | None,
      optimized: bool = True,
      tiling_rank: int | None = None,
  ):
    if not isinstance(self.layout, TiledLayout):
      raise NotImplementedError(self.layout)
    layout, shape = self.layout, self.shape
    # Note that the loop below will "race" for layouts that replicate data.
    # However, in that case all of the racing writes store the same data, which
    # is ok in the CUDA memory model.
    if isinstance(ref, utils.MultimemRef):
      stores = self.transfer_tiled(
          ref.ref, swizzle, layout, shape, optimized, ref_tiling_rank=tiling_rank
      )
      for get, _update, _idx, ptr in stores:
        utils.multimem_store(ptr, get(self.registers))
    else:
      stores = self.transfer_tiled(
          ref, swizzle, layout, shape, optimized, ref_tiling_rank=tiling_rank
      )
      for get, _update, _idx, ptr in stores:
        reg = get(self.registers)
        reg_ty = ir.VectorType(reg.type)
        element_bitwidth = utils.bitwidth(reg_ty.element_type)
        if (
            isinstance(reg_ty.element_type, ir.FloatType)
            and element_bitwidth <= 8
        ):
          narrow_int = ir.IntegerType.get_signless(element_bitwidth)
          reg = vector.bitcast(ir.VectorType.get(reg_ty.shape, narrow_int), reg)
        llvm.store(reg, ptr)

  @classmethod
  def load_tiled(
      cls,
      ref,
      swizzle: int | None,
      *,
      is_signed: bool | None = None,
      layout: FragmentedLayout = WGMMA_LAYOUT,
      optimized: bool = True,
      tiling_rank: int | None = None,
      _load_fun: Callable[[ir.VectorType, ir.Value], ir.Value] = llvm.load,
      _narrow_float_as_int: bool = True,
  ) -> FragmentedArray:
    if not isinstance(layout, TiledLayout):
      raise NotImplementedError(layout)
    ref_ty = ir.MemRefType(ref.type)
    dtype = ref_ty.element_type
    tiled_shape = ref_ty.shape
    if tiling_rank is None:
      if len(tiled_shape) % 2:
        raise ValueError("Tiled reference must have even rank")
      if len(tiled_shape) < 2:
        raise ValueError("Tiled reference must have at least two dimensions")
      tiling_rank = len(tiled_shape) // 2
    else:
      if tiling_rank > len(tiled_shape) // 2:
        raise ValueError(
            f"Tiling rank for reference of shape {tiled_shape} must be at most"
            f" {len(tiled_shape) // 2}"
        )
      if not tiling_rank:
        raise ValueError(
            "Tiling rank for reference of shape {tiled_shape} must be at least"
            " 1"
        )
    tiling = Tiling((tiled_shape[-tiling_rank:],))
    shape = tiling.untile_shape(tiled_shape)
    reg_ty = ir.VectorType.get((layout.vector_length,), dtype)
    zero = vector.broadcast(reg_ty, c(0, dtype))
    registers = np.full(layout.registers_shape(shape), zero, dtype=object)
    is_narrow_float = (
        isinstance(dtype, ir.FloatType) and utils.bitwidth(dtype) <= 8
    )
    narrow_int = ir.IntegerType.get_signless(utils.bitwidth(dtype))
    # Narrow floats are not supported by LLVM, so we need to transfer them as
    # narrow ints and bitcast back to the desired type.
    transfer_ty = ir.VectorType.get(
        (layout.vector_length,),
        narrow_int if is_narrow_float and _narrow_float_as_int else dtype
    )
    loads = cls.transfer_tiled(
        ref, swizzle, layout, shape, optimized, ref_tiling_rank=tiling_rank
    )
    for _get, update, _idx, ptr in loads:
      loaded_reg = _load_fun(transfer_ty, ptr)
      if is_narrow_float and _narrow_float_as_int:
        loaded_reg = vector.bitcast(reg_ty, loaded_reg)
      update(registers, loaded_reg)
    return cls(_registers=registers, _layout=layout, _is_signed=is_signed)

  @classmethod
  def transfer_strided(self, ref: ir.Value, vec_size: int):
    ref_ty = ir.MemRefType(ref.type)
    layout = WGStridedFragLayout(shape=tuple(ref_ty.shape), vec_size=vec_size)
    try:
      # Flattening the reference potentially produces simpler PTX but
      # if the ref is not already 1D and has strided dimensions
      # flattening won't work.
      ref = mgpu.memref_fold(ref, 0, len(ref_ty.shape))
    except ValueError:
      if vec_size > 1:
        ref_ty = ir.MemRefType(ref.type)
        shape = ref_ty.shape
        strides, _ = ref_ty.get_strides_and_offset()
        # Try to fold contiguous dimension pairs.
        for i in reversed(range(len(shape) - 1)):
          if strides[i] == shape[i+1] * strides[i+1]:
            ref = mgpu.memref_fold(ref, i, 2)
        ref_ty = ir.MemRefType(ref.type)
        shape = ref_ty.shape
        strides, _ = ref_ty.get_strides_and_offset()
        has_contiguous_dim = False
        for size, stride in zip(shape, strides):
          if stride == 1:
            has_contiguous_dim = True
            if size % vec_size != 0:
              raise ValueError(
                  "The contiguous dimension of the reference must be a"
                  f" multiple of the layout's vector size (got {size} and"
                  f" vector size {vec_size})"
              ) from None
          elif size > 1:
            if stride % vec_size != 0:
              raise ValueError(
                  "Non-contiguous dimension of the reference must have strides"
                  " that are multiples of the layout's vector size (got"
                  f" {stride} and vector size {vec_size})"
              ) from None
        if not has_contiguous_dim:
          raise ValueError(
              "The reference must have a contiguous dimension when vec_size > 1"
          )
      layout = WGStridedFragLayout(shape=tuple(ref_ty.shape), vec_size=vec_size)
      idx_gen = layout.thread_idxs(tuple(ref_ty.shape))
    else:
      idx_gen = map(lambda x: [x], layout.linear_thread_idxs())
    for i, vec_idx in enumerate(idx_gen):
      def update(registers, reg, _i=i):
        registers[_i] = reg
      def get(registers, _i=i):
        return registers[_i]
      yield get, update, ref, vec_idx

  @staticmethod
  def transfer_tiled(
      ref: ir.Value,
      swizzle: int | None,
      layout: TiledLayout,
      shape: tuple[int, ...],
      optimized: bool = True,
      ref_tiling_rank: int | None = None,
  ):
    """Generate a transfer schedule for a tiled layout.

    Given a ref with one level tiling applied to it (we assume all dimensions
    have been tiled), this function generates an iterable describing a good
    schedule for swizzled SMEM loads/stores.

    At each step, the iterable yields a tuple of three values:
    * a function that takes a register array and returns the register to be
      stored at the current address
    * a function that takes a register array and a register loaded from the
      current address, and updates the register array with that register
    * the current address for load/store instructions
    """
    # TODO(apaszke): Use ldmatrix/stmatrix when possible.
    c = lambda x: arith.constant(ir.IntegerType.get_signless(32), x)
    i32 = ir.IntegerType.get_signless(32)
    tiling = layout.tiling

    ref_ty = ir.MemRefType(ref.type)
    dtype = ref_ty.element_type
    if ref_tiling_rank is None:
      if len(ref_ty.shape) % 2:
        raise ValueError("Tiled reference must have an even rank when its tiling rank is not specified")
      ref_tiling_rank = ref_ty.rank // 2
    if ref_tiling_rank > len(ref_ty.shape) // 2:
      raise ValueError(
          f"Tiling rank for reference of shape {ref_ty.shape} must be at most"
          f" {len(ref_ty.shape) // 2}"
      )
    assert ref_tiling_rank and ref_ty.rank > ref_tiling_rank
    ref_logical_rank = ref_ty.rank - ref_tiling_rank
    ref_tiling_shape = tuple(ref_ty.shape[ref_logical_rank:])
    ref_tiling = Tiling((ref_tiling_shape,))
    ref_strides, _ = ref_ty.get_strides_and_offset()
    if (ref_logical_shape := ref_tiling.untile_shape(tuple(ref_ty.shape))) != shape:
      raise ValueError(
          f"The reference has untiled shape of {ref_logical_shape} while the"
          f" register array has shape {shape}"
      )
    first_tiled_dim = ref_logical_rank - ref_tiling_rank
    nested_ref_shape = tuple(
        (ref_ty.shape[i], ref_ty.shape[i + ref_tiling_rank])
        if i >= first_tiled_dim and ref_ty.shape[i + ref_tiling_rank] != 1
        else (ref_ty.shape[i],)
        for i in range(ref_logical_rank)
    )
    nested_ref_strides = tuple(
        (ref_strides[i], ref_strides[i + ref_tiling_rank])
        if i >= first_tiled_dim and ref_ty.shape[i + ref_tiling_rank] != 1
        else (ref_strides[i],)
        for i in range(ref_logical_rank)
    )
    tiled_nested_shape, tiled_nested_strides = tiling.tile_nested_shape_strides(
        nested_ref_shape, nested_ref_strides
    )
    # Not sure if this is strictly required for all data types, but it certainly
    # is for sub-byte types (else we might not increment the pointer by whole bytes).
    if any(
        any(s % layout.vector_length and d != 1 for s, d in zip(ss, ds))
        for i, (ss, ds) in enumerate_negative(list(zip(tiled_nested_strides, tiled_nested_shape)))
        if i != layout.vector_dim
    ):
      raise ValueError(
          "Tiled strides must be a multiple of the vector length, except for the"
          " vector dimension"
      )
    if tiled_nested_strides[layout.vector_dim] != (1,):
      raise ValueError(
          "Vectorized dimension should not require further tiling and have a"
          " stride of 1"
      )

    tiles_shape = list(tiled_nested_shape)
    tiles_strides = list(tiled_nested_strides)
    for d in (*layout.partitioned_warp_dims, *layout.partitioned_lane_dims, layout.vector_dim):
      # We could avoid repeating the singleton dimensions, but it simplifies the
      # code below that computes the register index for a given tile.
      tiles_shape[d] = (1,) * len(tiles_shape[d])
      tiles_strides[d] = (0,) * len(tiles_strides[d])
    tiles_shape = list(itertools.chain.from_iterable(tiles_shape))
    tiles_strides = list(itertools.chain.from_iterable(tiles_strides))
    warp_shape = list(itertools.chain.from_iterable(
        (d.times,) if isinstance(d, Replicated) else tiled_nested_shape[d] for d in layout.warp_dims
    ))
    warp_strides = list(itertools.chain.from_iterable(
        (0,) if isinstance(d, Replicated) else tiled_nested_strides[d] for d in layout.warp_dims
    ))
    lane_shape = list(itertools.chain.from_iterable(
        (d.times,) if isinstance(d, Replicated) else tiled_nested_shape[d] for d in layout.lane_dims
    ))
    lane_strides = list(itertools.chain.from_iterable(
        (0,) if isinstance(d, Replicated) else tiled_nested_strides[d] for d in layout.lane_dims
    ))
    vector_length = layout.vector_length

    element_bits = mgpu.bitwidth(dtype)
    if (vector_length * element_bits) % 8 != 0:
      raise ValueError(
          f"Vector length ({vector_length}) must be a multiple of bytes,"
          f" but has {vector_length * element_bits} bits"
      )
    transfer_bytes = (vector_length * element_bits) // 8

    if swizzle not in {16, 32, 64, 128}:
      raise ValueError("Only swizzled transfers supported")
    # We will be computing the offsets in units of vectors, not elements,
    # to better support sub-byte types.
    swizzle_tile_transfers = 16 // transfer_bytes
    swizzle_group_transfers = 128 // transfer_bytes
    swizzle_groups_per_block = swizzle // 16
    swizzle_block_transfers = swizzle_groups_per_block * swizzle_group_transfers
    if isinstance(dtype, ir.FloatType) and element_bits <= 8:
      narrow_int = ir.IntegerType.get_signless(element_bits)
      transfer_dtype = ir.VectorType.get((vector_length,), narrow_int)
    else:
      transfer_dtype = ir.VectorType.get((vector_length,), dtype)

    if ref_ty.memory_space is None:
      llvm_memory_space = None
    elif utils.is_smem_ref(ref_ty):
      llvm_memory_space = 3
    elif ref_ty.memory_space == ir.IntegerAttr.get(i32, 7):  # Cluster SMEM
      llvm_memory_space = 7
    else:
      raise ValueError(f"Unsupported memory space: {ref_ty.memory_space}")

    if optimized:
      if llvm_memory_space != 3 and llvm_memory_space != 7:
        raise NotImplementedError("Only optimized transfers to SMEM supported")
      plan = plan_tiled_transfer(
          tiles_shape, tiles_strides,
          warp_shape, warp_strides,
          lane_shape, lane_strides,
          vector_length, element_bits, swizzle
      )
    else:
      plan = TrivialTransferPlan()

    tiles_strides_transfer = [s // vector_length for s in tiles_strides]
    # Technically we should keep the vector_dim stride set to 1, but its shape
    # is 1 so it does not matter.
    dyn_tiled_strides = [
        c(s // vector_length)
        for s in itertools.chain.from_iterable(
            tiled_nested_strides[-layout.tiled_tiling_rank :]
        )
    ]
    # This expands a tiled index into a finer-grained index that accounts for
    # the fact that some tiled dims are tiled further in the nested shape.
    def expand_nested_dims(idxs: Sequence[ir.Value]) -> list[ir.Value]:
      assert len(idxs) == layout.tiled_tiling_rank
      new_idxs = []
      for idx, dim_shape in zip(idxs, tiled_nested_shape[-layout.tiled_tiling_rank :]):
        if dim_shape == (1,):
          new_idxs.append(idx)
          continue
        dim_strides = utils.get_contiguous_strides(dim_shape)
        for i, (size, stride) in enumerate(zip(dim_shape, dim_strides)):
          new_idx = arith.divui(idx, c(stride))
          if i != 0:  # No need to apply rem to the first dim.
            new_idx = arith.remui(new_idx, c(size))
          new_idxs.append(new_idx)
      assert len(new_idxs) == sum(map(len, tiled_nested_shape[-layout.tiled_tiling_rank :]))
      return new_idxs
    # All offsets are in units of transfer_dtype.
    lane_offset = utils.dyn_dot(expand_nested_dims(layout.lane_indices()), dyn_tiled_strides)
    warp_offset = utils.dyn_dot(expand_nested_dims(layout.warp_indices()), dyn_tiled_strides)
    dyn_offset = arith.addi(lane_offset, warp_offset)
    ptr = utils.memref_ptr(ref, memory_space=llvm_memory_space)
    _as_consts = lambda consts: [c(const) for const in consts.tolist()]
    # This has bits set only for the offset bits that influence swizzling.
    swizzle_mask = swizzle_block_transfers - swizzle_tile_transfers
    for tile_idx in np.ndindex(*tiles_shape):
      indices = np.asarray([f(tile_idx) for f in plan.tile_index_transforms])
      const_offset = np.dot(indices, tiles_strides_transfer)
      # We split the offset into a part that interacts with swizzling and a
      # part that doesn't. This lets us generate better code because constant
      # offsets can be fused into load and store instructions.
      const_offset_swizzle = const_offset & swizzle_mask
      const_offset_no_swizzle = const_offset - const_offset_swizzle
      offset_pre_swizzle = arith.addi(
          dyn_offset, plan.select(_as_consts(const_offset_swizzle))
      )
      swizzle_group = arith.remui(
          arith.divui(offset_pre_swizzle, c(swizzle_group_transfers)),
          c(swizzle_groups_per_block),
      )
      swizzle_bits = arith.muli(swizzle_group, c(swizzle_tile_transfers))
      offset = arith.xori(offset_pre_swizzle, swizzle_bits)
      reg_ptr = utils.getelementptr(ptr, [offset], transfer_dtype)
      offset_no_swizzle = plan.select(_as_consts(const_offset_no_swizzle))
      reg_ptr = utils.getelementptr(reg_ptr, [offset_no_swizzle], transfer_dtype)
      # Here, registers are organized in an array with shape obtained by tiling
      # the logical data bounds. But, the reference was tiled and so each
      # logical tiled dimension can map to multiple dims in tiled_shape.
      # The transform below maps this potentially higher-rank representation
      # back to the lower-rank representation used by the register arrays.
      def mem_idx_to_reg_idx(idx):
        reg_tiled_idx = []
        base_idx = 0
        for dim_shape in tiled_nested_shape:
          dim_strides = utils.get_contiguous_strides(dim_shape)
          dim_idxs = idx[base_idx:base_idx + len(dim_shape)]
          base_idx += len(dim_shape)
          reg_tiled_idx.append(sum(i * s for i, s in zip(dim_idxs, dim_strides)))
        return tuple(reg_tiled_idx)
      reg_idxs = [mem_idx_to_reg_idx(idx) for idx in indices.tolist()]
      def get_register(regs, reg_idxs=reg_idxs):
        # f8 data types are not handled by the LLVM dialect, so we need to
        # transfer them as i8 and bitcast them back to f8.
        return plan.select([regs[reg_idx] for reg_idx in reg_idxs])
      def update_registers(regs, new, reg_idxs=reg_idxs):
        # TODO(apaszke): If the staggering forms a permutation with a small
        # cycle length, then instead of blending at each step we could construct
        # a small routing network (kind of like a sorting network) to fix up
        # each cycle separately after all the loads are performed.
        # This would be especially useful for dims that are powers of two and
        # staggered by another power of 2, since all cycles are of length 2 (and
        # we could save half the selects).
        for i, reg_idx in enumerate(reg_idxs):
          regs[reg_idx] = plan.select_if_group(i, regs[reg_idx], new)
      def get_base_index():
        if not isinstance(plan, TrivialTransferPlan):
          raise NotImplementedError(
              "Base index computation only supported for trivial transfer plans"
          )
        if any(len(t) != 1 for t in tiled_nested_shape):
          raise NotImplementedError("Tiling too complicated")
        return tiling.untile_indices(indices.tolist()[0])
      yield get_register, update_registers, get_base_index, reg_ptr

  def tree_flatten(self):
    aux = self.layout, self.registers.shape, self.is_signed
    return list(self.registers.flat), aux

  @classmethod
  def tree_unflatten(cls, aux, flat_registers):
    layout, reg_shape, is_signed = aux
    registers = np.asarray(flat_registers, dtype=object).reshape(reg_shape)
    return cls(_registers=registers, _layout=layout, _is_signed=is_signed)


IndexTransform: TypeAlias = Callable[[tuple[int, ...]], tuple[int, ...]]


@runtime_checkable
class TransferPlan(Protocol):
  tile_index_transforms: tuple[IndexTransform, ...]

  def select(self, group_elems: Sequence[ir.Value]) -> ir.Value:
    """Selects the value corresponding to the group of the current thread.

    The argument must be of the same length as tile_index_transforms.
    """
    raise NotImplementedError

  def select_if_group(self, group_idx: int, old: ir.Value, new: ir.Value) -> ir.Value:
    """Returns `new` if the current thread belongs to the given group and `old` otherwise.

    group_idx must be between 0 and len(tile_index_transforms) - 1.
    """
    raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class TrivialTransferPlanImpl(TransferPlan):
  @property
  def tile_index_transforms(self):
    return (lambda x: x,)

  def select(self, group_elems: Sequence[ir.Value]) -> ir.Value:
    assert len(group_elems) == 1
    return group_elems[0]

  def select_if_group(self, group_idx: int, old: ir.Value, new: ir.Value) -> ir.Value:
    assert group_idx == 0
    return new


@dataclasses.dataclass(frozen=True)
class StaggeredTransferPlanImpl(TransferPlan):
  stagger: int
  dim: int
  size: int
  group_pred: ir.Value

  @property
  def tile_index_transforms(self):
    dim = self.dim  # pytype: disable=attribute-error
    def rotate(idx: tuple[int, ...]) -> tuple[int, ...]:
      return (
          *idx[:dim], (idx[dim] + self.stagger) % self.size, *idx[dim + 1 :],
      )
    return (lambda x: x, rotate)

  def select(self, group_elems: Sequence[ir.Value]) -> ir.Value:
    assert len(group_elems) == 2
    return arith.select(self.group_pred, group_elems[1], group_elems[0])

  def select_if_group(self, group_idx: int, old: ir.Value, new: ir.Value) -> ir.Value:
    assert 0 <= group_idx <= 1
    sides = [old, new] if group_idx == 0 else [new, old]
    return arith.select(self.group_pred, *sides)

TrivialTransferPlan: type[TransferPlan]
StaggeredTransferPlan: type[TransferPlan]

# TODO(olechwierowicz): Remove this once the C++ impl is always included in jaxlib (min ver 0.9.1).
if (
    hasattr(mgpu.dialect, "TrivialTransferPlan")
    and hasattr(mgpu.dialect, "StaggeredTransferPlan")
    and hasattr(mgpu.dialect, "init_cc_mlir")
    and mgpu.dialect.init_cc_mlir(ir)
):
  TrivialTransferPlan = mgpu.dialect.TrivialTransferPlan
  StaggeredTransferPlan = mgpu.dialect.StaggeredTransferPlan
else:
  TrivialTransferPlan = TrivialTransferPlanImpl
  StaggeredTransferPlan = StaggeredTransferPlanImpl


def plan_tiled_transfer(
    tiles_shape: Sequence[int],
    tiles_strides: Sequence[int],
    warp_shape: Sequence[int],
    warp_strides: Sequence[int],
    lane_shape: Sequence[int],
    lane_strides: Sequence[int],
    vector_length: int,
    element_bits: int,
    swizzle: int,
) -> TransferPlan:
  """Plans the tiled transfer in a way that avoids SMEM bank conflicts.

  Note that while xyz_shape length should always match the length of
  xyz_strides, we do not require the iteration spaces of tiles/warps/lanes to
  have the same rank.

  Arguments:
    tiles_shape: The nd-iteration space over tiles.
    tiles_strides: The memory strides (in elements) for each tile dimension.
    warp_shape: The nd-iteration space over warps in warpgroup.
    warp_strides: The memory strides (in elements) for each warp dimension.
    lane_shape: The nd-iteration space over lanes in a warp.
    lane_strides: The memory strides (in elements) for each lane dimension.
    vector_length: The length of a single transfer.
    element_bits: Element bitwidth.
    swizzle: The swizzle pattern length.
  """
  i32 = ir.IntegerType.get_signless(32)
  c = lambda x: arith.constant(i32, x)
  # TODO(apaszke): Rewrite this function in terms of transfer_bytes (that we get
  # from the caller).
  swizzle_tile_elems = (16 * 8) // element_bits
  swizzle_group_elems = (128 * 8) // element_bits
  # Should be checked at the call site.
  assert vector_length * element_bits % 8 == 0
  transfer_bytes = (vector_length * element_bits) // 8
  # Below, all calculations are in elements, not in bytes, since it should
  # generalize better to sub-byte types.
  # Here, we verify two conditions:
  # 1. Each vector transfer only accesses addresses that fall within a single
  # swizzle tile (if not we'd need to split it and swizzle parts differently).
  chain = itertools.chain
  transfer_alignment = math.gcd(*(
      s
      for (s, d) in zip(
          chain(tiles_strides, warp_strides, lane_strides),
          chain(tiles_shape, warp_shape, lane_shape),
      )
      if d > 1
  ))
  if (
      swizzle_tile_elems % transfer_alignment
      and vector_length <= transfer_alignment
  ):
    raise ValueError(
        "Failed to prove that vector transfers don't cross swizzle tile"
        " boundaries. This check is incomplete, and does not guarantee that"
        f" this is a user error, but it might be. {transfer_alignment=}"
    )

  # 2. The transfer pattern does not cause bank conflicts.
  # TODO(apaszke): For now, when performing transfers narrower than a bank,
  # we simply narrow each bank to the transfer width. The truth is more likely
  # that bank conflicts only don't occur if the addresses mapping to the same
  # bank are contiguous, but that's a more complicated check to perform.
  if transfer_bytes > SMEM_BANK_BYTES * 4:
    raise NotImplementedError
  if element_bits > SMEM_BANK_BYTES * 8:
    raise NotImplementedError
  smem_bank_bytes = min(SMEM_BANK_BYTES, transfer_bytes)
  num_banks = SMEM_BANKS * (SMEM_BANK_BYTES // smem_bank_bytes)
  elems_per_bank = (smem_bank_bytes * 8) // element_bits
  num_wavefronts = max(transfer_bytes // smem_bank_bytes, 1)
  wavefront_lanes = WARP_SIZE // num_wavefronts

  lane_mask = np.full(lane_shape, False)
  lane_mask[tuple(slice(0, 1) if s == 0 else slice(None) for s in lane_strides)] = True
  wavefront_mask = lane_mask.reshape(num_wavefronts, wavefront_lanes)

  lane_offsets_in_tile = np.dot(list(np.ndindex(*lane_shape)), lane_strides)
  def has_bank_conflicts(tile_idx_transform):
    num_tiles = math.prod(tiles_shape)
    tile_idxs = np.unravel_index(np.arange(num_tiles), tiles_shape)
    tile_idxs = np.expand_dims(np.stack(tile_idxs, 1), 1)  # [#tiles, 1, #dims]
    lane_tile_idx = tile_idx_transform(tile_idxs)  # [#tiles, #lanes/1, #dims]
    assert lane_tile_idx.shape[1] in {1, WARP_SIZE}
    lane_tile_offsets = np.dot(lane_tile_idx, tiles_strides)
    offsets = lane_tile_offsets + lane_offsets_in_tile  # [#tiles, #lanes]
    assert offsets.shape[-1] == WARP_SIZE
    swizzle_groups = (offsets // swizzle_group_elems) % (swizzle // 16)
    swizzle_bits = swizzle_groups * swizzle_tile_elems
    lane_banks = ((offsets ^ swizzle_bits) // elems_per_bank) % num_banks
    wavefront_banks = lane_banks.reshape(-1, num_wavefronts, wavefront_lanes)
    # We step over wavefronts since they might have a different number of lanes.
    wavefront_banks = wavefront_banks.swapaxes(0, 1)
    for banks, mask in zip(wavefront_banks, wavefront_mask):
      banks = banks[:, mask]
      # Order of threads within the wavefront is unimportant.
      banks = np.sort(banks, axis=-1)
      # There are no conflicts if each wavefront only contains unique banks.
      repeats = np.any(banks[..., 1:] == banks[..., :-1])
      if repeats:
        return True
    return False

  # We don't need any special treatment if there are no conflicts when each lane
  # transfers the same tile at a time.
  if not has_bank_conflicts(lambda tile_idx: tile_idx):
    return TrivialTransferPlan()

  # Otherwise, we will try to partition the lanes into two groups and have
  # each group store to different tile. The only tile dimensions that can help
  # us with bank conflicts are those that have multiple elements and a stride
  # that's not a multiple of the number of banks.
  #
  # Note that the code is set up so that we could also consider partitioning
  # the lanes into more groups, but the selects will become more expensive if
  # we do that. It's a possibility we have if we need it.
  candidate_dims = (
      i for i, (s, d) in enumerate(zip(tiles_strides, tiles_shape))
      if d > 1 and s % (SMEM_BANKS * elems_per_bank)
  )
  for dim in candidate_dims:
    for group_stride in (1, 2, 4, 8, 16):
      # We change the group assignment each group_stride lanes.
      lane_id = np.arange(WARP_SIZE)[:, None]
      lane_group = (lane_id // group_stride) % 2
      # We only consider a transformation where the second group stores to a
      # tile that's a constant offset (modulo dim size) from the first one.
      for stagger in range(1, tiles_shape[dim]):
        offset = np.zeros(len(tiles_shape), np.int64)
        offset[dim] = stagger
        transform = lambda idx: (idx + offset * lane_group) % tiles_shape
        if not has_bank_conflicts(transform):
          # We've found a strategy that avoids bank conflicts!
          lane_idx = arith.remui(utils.thread_idx(), c(WARP_SIZE))
          group_idx = arith.remui(arith.divui(lane_idx, c(group_stride)), c(2))
          group_pred = arith.cmpi(arith.CmpIPredicate.ne, group_idx, c(0))
          return StaggeredTransferPlan(  # type: ignore[call-arg]
              stagger, dim, tiles_shape[dim], group_pred  # pylint: disable=too-many-function-args
          )
  raise ValueError(
      "Failed to synthesize a transfer pattern that avoids bank conflicts"
  )

# We allow contractions, to potentially take advantage of FMA instructions.
# They can change the results, but the precision should only increase.
def addf(a: ir.Value, b: ir.Value):
  return arith.addf(a, b, fastmath=arith.FastMathFlags.contract)

def subf(a: ir.Value, b: ir.Value):
  return arith.subf(a, b, fastmath=arith.FastMathFlags.contract)

def mulf(a: ir.Value, b: ir.Value):
  return arith.mulf(a, b, fastmath=arith.FastMathFlags.contract)


@overload
def optimization_barrier(
    a: mgpu.FragmentedArray,
    b: mgpu.FragmentedArray,
    /,
    *arrays: mgpu.FragmentedArray,
) -> Sequence[mgpu.FragmentedArray]:
  ...


@overload
def optimization_barrier(a: mgpu.FragmentedArray) -> mgpu.FragmentedArray:
  ...


def optimization_barrier(*arrays):
  """Acts as an optimization barrier for LLVM.

  Passing arrays through this function will make sure that they are computed
  before any side-effecting operations that follow this barrier.
  """
  i32 = ir.IntegerType.get_signless(32)

  def _repack(regs_it, reg_ty):
    if not isinstance(reg_ty, ir.VectorType):
      result_reg = next(regs_it)
      assert result_reg.type == reg_ty
      return result_reg

    num_i32_regs = utils.bitwidth(reg_ty) // 32
    i32_reg_ty = ir.VectorType.get((num_i32_regs,), i32)
    reg = llvm.mlir_undef(i32_reg_ty)
    for i_elem in range(num_i32_regs):
      val = llvm.bitcast(i32, next(regs_it))
      reg = llvm.insertelement(reg, val, arith.constant(i32, i_elem))
    return vector.bitcast(reg_ty, reg)

  regs = []
  reg_dtypes = []
  reg_constraints = []
  # We unpack each array into a flat list of registers, and prepare the
  # functions that invert the transform in repack_fns.
  for array in arrays:
    reg_ty = array.registers.flat[0].type
    dtype = array.mlir_dtype
    if isinstance(dtype, ir.F32Type) or dtype == i32:
      if isinstance(reg_ty, ir.VectorType):
        [vec_len] = ir.VectorType(reg_ty).shape
        array_regs = [  # pylint: disable=g-complex-comprehension
            vector.extract(
                reg,
                dynamic_position=[],
                static_position=ir.DenseI64ArrayAttr.get([pos]),
            )
            for reg in array.registers.flat
            for pos in range(vec_len)
        ]
      else:
        array_regs = list(array.registers.flat)
      reg_constraint = "r" if dtype == i32 else "f"
    elif utils.bitwidth(dtype) < 32:
      reg_packing = 4 // utils.bytewidth(dtype)
      if not isinstance(reg_ty, ir.VectorType):
        raise NotImplementedError(array.mlir_dtype)
      [vec_len] = ir.VectorType(reg_ty).shape
      if vec_len % reg_packing:
        raise NotImplementedError(vec_len)
      num_i32_regs = vec_len // reg_packing
      i32_reg_ty = ir.VectorType.get((num_i32_regs,), i32)
      array_regs = [
          vector.extract(
              vector.bitcast(i32_reg_ty, reg),
              dynamic_position=[],
              static_position=ir.DenseI64ArrayAttr.get([i]),
          )
          for i in range(num_i32_regs)
          for reg in array.registers.flat
      ]
      reg_constraint = "r"
    else:
      raise NotImplementedError(array.mlir_dtype)
    regs += array_regs
    reg_dtypes += [array_regs[0].type] * len(array_regs)
    reg_constraints += [reg_constraint] * len(array_regs)
  ptx = ""
  all_reg_constraints = ",".join(
      [*("=" + c for c in reg_constraints), *map(str, range(len(reg_constraints)))]
  )

  if len(reg_dtypes) == 1:
    # The InlineAsm::verify() function doesn't allow a struct output when there
    # is only one element (even though that seems to work for the case below).
    result_elem = llvm.inline_asm(
        reg_dtypes[0], regs, ptx, all_reg_constraints,
        asm_dialect=0, has_side_effects=True,
    )
    regs = [result_elem]
  else:
    struct_ty = ir.Type.parse(
        f"!llvm.struct<({','.join(map(str, reg_dtypes))})>"
    )
    result_struct = llvm.inline_asm(
        struct_ty, regs, ptx, all_reg_constraints,
        asm_dialect=0, has_side_effects=True,
    )
    regs = [
        llvm.extractvalue(dtype, result_struct, [i])
        for i, dtype in enumerate(reg_dtypes)
    ]

  i32 = ir.IntegerType.get_signless(32)
  results = []
  regs_it = iter(regs)
  for array in arrays:
    num_regs = array.registers.size
    reg_ty = array.registers.flat[0].type
    if isinstance(reg_ty, ir.VectorType):
      reg_ty = ir.VectorType(reg_ty)
    new_registers = np.empty((num_regs,), dtype=object)
    for i_vreg in range(num_regs):
      reg = _repack(regs_it, reg_ty)
      assert reg.type == reg_ty, (reg.type, reg_ty)
      new_registers[i_vreg] = reg
    results.append(
        FragmentedArray(
            _registers=new_registers.reshape(array.registers.shape),
                        _layout=array.layout,
            _is_signed=array.is_signed,
        )
    )
  # pytype cannot type check the return type of an overloaded function.
  return results[0] if len(arrays) == 1 else results  # pytype: disable=bad-return-type


def tiled_copy_smem_gmem_layout(
    row_tiles: int, col_tiles: int, swizzle: int, bitwidth: int
) -> TiledLayout:
  swizzle_elems = 8 * swizzle // bitwidth
  if row_tiles % 4 == 0:
    warp_row_tiles, warp_col_tiles = 4, 1
  elif row_tiles % 2 == 0:
    if col_tiles % 2:
      raise NotImplementedError("Number of tiles is not a multiple of 4")
    warp_row_tiles, warp_col_tiles = 2, 2
  else:
    if col_tiles % 4:
      raise NotImplementedError("Number of tiles is not a multiple of 4")
    warp_row_tiles, warp_col_tiles = 1, 4
  row_tiles //= warp_row_tiles
  col_tiles //= warp_col_tiles
  bytes_per_thread = min(16, 8 * swizzle // WARP_SIZE)
  lane_row_tiles = lane_col_tiles = 1
  if bytes_per_thread < 16:  # Try to splread multiple tiles over a warp.
    max_scale_up = 16 // bytes_per_thread
    while max_scale_up > 1 and col_tiles % 2 == 0:
      max_scale_up //= 2
      lane_col_tiles *= 2
      col_tiles //= 2
    while max_scale_up > 1 and row_tiles % 2 == 0:
      max_scale_up //= 2
      lane_row_tiles *= 2
      row_tiles //= 2
    bytes_per_thread *= lane_row_tiles * lane_col_tiles
  if 8 * bytes_per_thread < bitwidth:
    raise NotImplementedError("Element types with bitwidth so large aren't supported")
  vector_length = bytes_per_thread * 8 // bitwidth
  assert swizzle_elems % vector_length == 0
  # How many steps of vector transfers are needed to transfer a single tile?
  if vector_length * WARP_SIZE > 8 * swizzle_elems:
    steps_per_tile = 1
  else:
    steps_per_tile = 8 * swizzle_elems // (vector_length * WARP_SIZE)
  tile_rows_per_step = 8 // steps_per_tile
  # There are two cases to consider here: either a single transfer fits within
  # a single tile (lane_row_tiles == lane_col_tiles == 1), which is the case
  # for large swizzles, or it spans multiple tiles. The layout below ensures
  # that consecutive lanes first traverse the columns within a tile, followed
  # by rows within a tile, columns across tiles, and then rows across tiles.
  # This ensures we never end up with bank conflicts, and yields well
  # coalesced GMEM accesses.
  return TiledLayout(
      Tiling(
          (
              (warp_row_tiles * lane_row_tiles * 8, warp_col_tiles * lane_col_tiles * swizzle_elems),
              (lane_row_tiles * 8, lane_col_tiles * swizzle_elems),
              (8, swizzle_elems),
              (tile_rows_per_step, swizzle_elems),
              (vector_length,)
          )
      ),
      warp_dims=(-9, -8),
      lane_dims=(-7, -6, -3, -2),
      vector_dim=-1,
      _check_canonical=False,
  ).canonicalize()


def copy_tiled(src: ir.Value, dst: ir.Value, swizzle: int = 16):
  """Copy the data from the src reference to the dst reference.

  Exactly one of src/dst should be in SMEM, while the other should be in GMEM.
  The SMEM reference is expected to be tiled into (8, swizzle_elems) (as it
  would for MMA), and so should have a rank larger by 2 than the GMEM ref.
  """
  src_ty = ir.MemRefType(src.type)
  dst_ty = ir.MemRefType(dst.type)
  if math.prod(src_ty.shape) != math.prod(dst_ty.shape):
    raise ValueError(
        "Source and destination must have the same number of elements, but got"
        f" source shape {src_ty.shape} and destination shape {dst_ty.shape}"
    )
  if src_ty.element_type != dst_ty.element_type:
    raise ValueError(
        "Source and destination must have the same element type, but got"
        f" source type {src_ty.element_type} and destination type"
        f" {dst_ty.element_type}"
    )
  bitwidth = utils.bitwidth(src_ty.element_type)
  # Signedness doesn't matter, but we need to specify something for the
  # intermediate arrays.
  is_signed = False if isinstance(src_ty.element_type, ir.IntegerType) else None
  if utils.is_smem_ref(src_ty) != utils.is_smem_ref(dst_ty):
    if utils.is_smem_ref(src_ty):
      smem_ty, gmem_ty = src_ty, dst_ty
    else:
      smem_ty, gmem_ty = dst_ty, src_ty
    if smem_ty.rank != gmem_ty.rank + 2:
      raise ValueError(
          "SMEM reference must have a rank larger by 2 than the destination"
          f" reference (due to 2D tiling), but got SMEM rank {smem_ty.rank} and"
          f" destination rank {gmem_ty.rank}."
      )
    swizzle_elems = 8 * swizzle // bitwidth
    if smem_ty.shape[-2:] != [8, swizzle_elems]:
      raise NotImplementedError(
          f"For {swizzle=}, expected SMEM tiling to be (8, {swizzle_elems})"
      )
    expected_src_shape = utils.tile_shape(gmem_ty.shape, (8, swizzle_elems))
    if tuple(smem_ty.shape) != expected_src_shape:
      raise ValueError(
          f"Expected SMEM reference to have shape {expected_src_shape} (tiling"
          f" {gmem_ty.shape} by (8, {swizzle_elems})), but got {smem_ty.shape}"
      )
    layout = tiled_copy_smem_gmem_layout(
        *smem_ty.shape[-4:-2], swizzle, bitwidth  # type: ignore[call-arg]
    )
    if utils.is_smem_ref(src_ty):
      regs = FragmentedArray.load_tiled(src, swizzle, is_signed=is_signed, layout=layout)
      regs.store_untiled(dst, optimized=False)
    else:
      regs = FragmentedArray.load_untiled(src, is_signed=is_signed, layout=layout, optimized=False)
      regs.store_tiled(dst, swizzle)
    return
  raise NotImplementedError(f"Unsupported copy: {src.type} -> {dst.type}")
