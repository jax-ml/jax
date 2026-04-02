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

"""Module for emitting custom TPU pipelines within a Pallas call."""
from __future__ import annotations

from collections.abc import Sequence
import contextlib
from contextlib import contextmanager
import dataclasses
import enum
import functools
from typing import Any, Literal, Union

import jax
from jax import core as jax_core
from jax import lax
from jax import tree_util
from jax._src import state
from jax._src import util as jax_util
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import helpers as tpu_helpers
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax._src.pallas.mosaic import tpu_info
from jax.experimental import pallas as pl
import jax.numpy as jnp


SMEM = tpu_core.MemorySpace.SMEM
VMEM = tpu_core.MemorySpace.VMEM
HBM = tpu_core.MemorySpace.HBM
ANY = pallas_core.MemorySpace.ANY
REF = jax.Ref
GridDimensionSemantics = tpu_core.GridDimensionSemantics
PARALLEL = tpu_core.PARALLEL
ARBITRARY = tpu_core.ARBITRARY
SemaphoreType = tpu_core.SemaphoreType
SemaphoreTuple = jax.Array
ArrayRef = Union[REF, jax.Array]
Tiling = tpu_info.Tiling

GridIndices = tuple[jax.Array, ...]
CondVal = Union[jax.Array, bool]
PipelineBlockSpecs = Union[Sequence[pallas_core.BlockSpec], Any]
PipelineRefs = Union[Sequence[REF], Any]


def _round_up_to_nearest_multiple(
    s: int | jax.Array, multiple: int
) -> int | jax.Array:
  if isinstance(s, int) and s % multiple == 0:
    return s
  # Subtract off the remainder, then add multiple
  return s - s % multiple + multiple


def _make_block_ds(
    idx: jax.Array | int, size: jax.Array | int
) -> pl.Slice:
  """Make a DMA slice with mosaic size hints."""
  out = pl.ds(idx * size, size)
  assert isinstance(out, pl.Slice)
  return out


def _create_blocked_slice(
    block_index: jax.Array | int,
    block_size: int,
    dim_size: int,
    tiling: int | None,
):
  block_start = block_size * block_index
  if (dim_rem := dim_size % block_size) == 0:
    return pl.ds(block_start, block_size)
  if tiling is None:
    raise ValueError("If tiling is None, block_size must divide dim_size.")
  if block_size % tiling != 0:
    raise ValueError(f"Block size must divide tiling: {block_size=}, {tiling=}")
  num_blocks = pl.cdiv(dim_size, block_size)
  is_last = block_index == num_blocks - 1
  rounded_size = jnp.where(
      is_last,
      _round_up_to_nearest_multiple(dim_rem % block_size, tiling),
      block_size,
  )
  rounded_size = pl.multiple_of(rounded_size, tiling)
  return pl.ds(block_index * block_size, rounded_size)


def _create_bounded_slice(slice_start: jax.Array | int,
                          slice_size: jax.Array | int,
                          block_size: int,
                          dim_size: int,
                          tiling: int | None):
  if tiling is not None and block_size % tiling != 0:
    raise ValueError(f"Block size must divide tiling: {block_size=}, {tiling=}")
  # We assume by construction that slice_size <= block_size. We also assume
  # that the slice_start is already aligned to the tiling.

  if tiling is None:
    return pl.ds(slice_start, slice_size)

  # If we are out of bound, we need to round the slice size down to the nearest
  # multiple of the tiling.
  is_oob = slice_start + slice_size > dim_size
  remaining = dim_size - slice_start
  rounded_size = jnp.where(
      is_oob,
      _round_up_to_nearest_multiple(remaining, tiling),
      slice_size,
  )
  rounded_size = pl.multiple_of(rounded_size, tiling)
  return pl.ds(slice_start, rounded_size)


def _make_block_slice(
    block_index: jax.Array, block_size: pl.BlockDim | int | None, size: int,
    tiling: int | None
) -> pl.Slice | slice | int | jax.Array:
  # Computes a slice given a block index and block size. In the default case,
  # we return slice(block_index * block_size, (block_index + 1) * block_size).
  # However, if the total size of the ref does not divide block size and we are
  # selecting the last block, we need to pick the lowest tiling size multiple
  # that contains the block.
  match block_size:
    case pl.Blocked():
      return _create_blocked_slice(block_index, block_size.block_size, size, tiling)
    case int():
      return _create_blocked_slice(block_index, block_size, size, tiling)
    case pl.Element():
      block_start = block_index
      block_size = block_size.block_size
      return _create_bounded_slice(
          block_start, block_size, block_size, size, tiling
      )
    case pl.BoundedSlice(block_size):
      if not isinstance(block_index, pl.Slice):
        raise ValueError(
            "Must return a pl.ds from the index_map for a BoundedSlice"
            " dimension."
        )
      slice_start = block_index.start
      slice_size = block_index.size
      return _create_bounded_slice(
          slice_start, slice_size, block_size, size, tiling
      )
    case None | pl.Squeezed():
      return block_index
    case _:
      raise ValueError(f"Unsupported block dimension type: {block_size}")


def _tuples_differ(xs, ys):
  """Dynamic index-tuple comparison calculation."""
  differences = jax.tree.leaves(jax.tree.map(lambda x, y: x != y, xs, ys))
  return functools.reduce(lambda x, y: x | y, differences, False)

def _tuple_all_binop(binop, xs, ys):
  """Dynamic reduce_all calculation with a user-provided comparison op."""
  differences = jax.tree.leaves(jax.tree.map(lambda x, y: binop(x, y), xs, ys))
  return functools.reduce(lambda x, y: x & y, differences, True)

_tuple_lt = functools.partial(_tuple_all_binop, lambda x, y: x < y)


def _grid_size(grid):
  """Dynamic grid size calculation."""
  size = jnp.array(1, jnp.int32)
  for dim in grid:
    size *= dim
  return size


def _spec_has_trivial_windowing(spec, grid, full_shape):
  if spec is None:
    return True
  if spec.block_shape is None:
    return True
  for bs, fs in zip(spec.block_shape, full_shape):
    if bs is None:
      return False
    if isinstance(bs, (pallas_core.BoundedSlice, pl.Squeezed, pl.Element)):
      return False
    if pallas_core.get_block_size(bs) != fs:
      return False
  if spec.index_map is None:
    return True
  nontrivial_dims = {
      i for i, d in enumerate(grid) if not isinstance(d, int) or d != 1
  }
  if not nontrivial_dims:
    return True
  static_dummy_grid = tuple(d if isinstance(d, int) else 2 for d in grid)
  with pallas_core.tracing_grid_env(static_dummy_grid, mapped_dims=()):
    closed_jaxpr = jax.make_jaxpr(spec.index_map)(*[0] * len(grid))
  jaxpr = closed_jaxpr.jaxpr
  # Refs can be mutated while the pipeline is running so we should not assume
  # that they are constant.
  if any(isinstance(v.aval, state.AbstractRef) for v in jaxpr.constvars):
    return False
  nontrivial_invar_ids = {id(jaxpr.invars[i]) for i in nontrivial_dims}
  for v in jaxpr.outvars:
    if id(v) in nontrivial_invar_ids:
      return False
  for eqn in jaxpr.eqns:
    for v in eqn.invars:
      if id(v) in nontrivial_invar_ids:
        return False
  return True


class BufferType(enum.Enum):
  """Buffer type for the arguments to an emitted pipeline."""
  INPUT = 1
  OUTPUT = 2
  INPUT_OUTPUT = 3
  MANUAL = 4

  @property
  def is_input(self):
    return self in [
        BufferType.INPUT,
        BufferType.INPUT_OUTPUT,
    ]

  @property
  def is_output(self):
    return self in [
        BufferType.OUTPUT,
        BufferType.INPUT_OUTPUT,
    ]


def _get_block_shape(spec: pl.BlockSpec) -> tuple[int, ...]:
  """Get the block shape for a given block spec."""
  def _get_dim_size(bd):
    match bd:
      case pl.Blocked(block_size):
        return block_size
      case pl.Element(block_size):
        return block_size
      case pl.BoundedSlice(block_size):
        return block_size
      case int():
        return bd
      case None | pl.Squeezed():
        return None
      case _:
        raise ValueError(f"Unsupported block dimension type: {bd}")
  if spec.block_shape is None:
    raise ValueError("Block shape must be specified.")
  block_shape_nones = tuple(_get_dim_size(x) for x in spec.block_shape)
  return tuple(x for x in block_shape_nones if x is not None)


class BufferedRefBase:
  """Abstract interface for BufferedRefs."""

  @property
  def spec(self) -> pl.BlockSpec:
    raise NotImplementedError()

  @property
  def buffer_type(self) -> BufferType:
    raise NotImplementedError()

  @property
  def is_buffered(self) -> bool:
    return False

  @property
  def is_input(self):
    return self.buffer_type.is_input

  @property
  def is_output(self):
    return self.buffer_type.is_output

  @property
  def is_input_output(self):
    return self.buffer_type == BufferType.INPUT_OUTPUT

  @property
  def is_manual(self):
    return self.buffer_type == BufferType.MANUAL

  @property
  def is_trivial_windowing(self) -> bool:
    """Whether the reference uses trivial windowing.

    Returns:
      True if the reference uses trivial windowing, False otherwise.
      Trivial windowing means that the BlockSpec just uses the full array,
      meaning there are no real opportunities for pipelining. Instead, we can
      just issue sync copies in/out before/after the pipeline for an
      input/output reference.
    """
    return False

  def initialize_slots(self):
    """Initializes slots to 0."""
    raise NotImplementedError()

  def advance_copy_in_slot(self, predicate: bool = True) -> BufferedRefBase:
    """Advance the copy in slot."""
    raise NotImplementedError()

  def advance_wait_in_slot(self, predicate: bool = True) -> BufferedRefBase:
    """Advance the wait in slot."""
    raise NotImplementedError()

  def advance_copy_out_slot(self, predicate: bool = True) -> BufferedRefBase:
    """Advance the copy out slot."""
    raise NotImplementedError()

  def advance_wait_out_slot(self, predicate: bool = True) -> BufferedRefBase:
    """Advance the wait out slot."""
    raise NotImplementedError()

  @property
  def block_shape(self) -> Sequence[pl.BlockDim | int | None] | None:
    return self.spec.block_shape

  @property
  def has_allocated_buffer(self) -> bool:
    """Returns True if the reference has an allocated buffer outside loop."""
    raise NotImplementedError()

  @property
  def compute_index(self):
    return self.spec.index_map

  def get_dma_slice(self, src_ty, grid_indices):
    # We need to handle blocks that might go OOB in the src array. An in bounds
    # block looks like this (for array shape (600, 600) and block shape
    # (256, 256)):
    #
    # +--------------+------------------|
    # | Block (0,0)  |                  |
    # | (256, 256)   |                  |
    # +--------------+                  |
    # |    A (600, 600)                 |
    # |                                 |
    # +---------------------------------+
    #
    # For in-bounds blocks, we don't need to do anything special.
    # An out-of-bounds block looks like this:
    #
    # +--------------+------------------|
    # |                                 |
    # |                                 |
    # +                                 |
    # |    A (600, 600)                 |
    # +--------------+                  |
    # | Block (2,0)  |                  |
    # + --------------------------------|
    # | XXXXXXXXXX   |
    # +--------------+
    # where the X's indicate where the block is out of bounds.
    #
    # When we have an out of bounds block like this, we need to truncate it to
    # a tile boundary (tiles are (8, 128) along the two minormost dimensions).
    # In this case, we'll have a block that is indexing the
    # 512:768 elements of A along the first dimension. We need to convert 768
    # into 600 (600 % 8 == 0), so our indexing will look like this:

    # +--------------+------------------|
    # |                                 |
    # |                                 |
    # +                                 |
    # |    A (600, 600)                 |
    # +--------------+                  |
    # | Block (2,0)  |                  |
    # + --------------------------------|
    # where it is now a (88, 256) sized block.
    #
    # Suppose A is now (601, 600), instead of picking a (88, 256)-sized block
    # for the last iteration on that dimension, we will pick the next highest
    # tile multiple, i.e. (96, 256).

    if (src_shape := getattr(src_ty, "shape", None)) is None:
      raise ValueError(f"Type {src_ty} does not have a shape")

    if not src_shape:
      return ()

    tiling = tpu_info.infer_tiling(src_ty, getattr(self, "tiling", None))
    block_indices = self.compute_index(*grid_indices)
    return tuple(
        _make_block_slice(bi, bs, ss, t)
        for bi, bs, ss, t in zip(
            # pyrefly: ignore[bad-argument-type]  # pyrefly#2385
            block_indices, self.block_shape, src_shape, tiling, strict=True
        )
    )

  def _to_window_slice(self, dma_slice):
    return tuple(
        pl.ds(0, s.size)
        for s, bd in zip(dma_slice, self.block_shape)  # pyrefly: ignore[bad-argument-type]
        if not (bd is None or isinstance(bd, pl.Squeezed))
    )

  def bind_existing_ref(self, window_ref, indices):
    """For handling VMEM references, the pipeline aliases the existing ref."""
    del window_ref, indices
    return self

  def unbind_refs(self):
    return self

  def with_spec(self, spec: pl.BlockSpec) -> BufferedRefBase:
    """Returns a new BufferedRefBase with the given block spec."""
    raise NotImplementedError()


def _ref_to_value_aval(ref):
  """Return the inner of a ref, or a ShapedArray for TransformedRefs."""
  return (
      jax_core.ShapedArray(shape=ref.shape, dtype=ref.dtype)
      if isinstance(ref, state.TransformedRef)
      else jax.typeof(ref).inner_aval
  )


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferedRef(BufferedRefBase):
  """A helper class to automate VMEM double buffering in pallas pipelines.

  Attributes:
    spec: pallas blockspec.
    buffer_type: enum indicating whether this is an input, output, or in/out
      buffered reference.
    window_ref: a multiple-buffer to hold the working and dirty buffers used
      to copy into and out of.  In the case of a BufferedRef targeting a VMEM
      reference, this simply points to the existing ref.
    copy_in_slot: current slot to copy in for the working buffer.
    copy_out_slot: current slot to copy out for the working buffer.
    wait_in_slot: current slot to wait in for the working buffer.
    wait_out_slot: current slot to wait out for the working buffer.
    next_fetch: Holds the next grid indices to fetch for lookahead. This
      is the register state used to track the indices within the pipeline loop.
    sem_recvs: Multiple buffered semaphores for input DMAs.
    sem_sends: Multiple buffered semaphores for output DMAs.
    tiling: The tiling to assume for the buffers.
    is_trivial_windowing: Whether the reference uses trivial windowing.
    has_allocated_buffer: Whether the reference has an allocated buffer
      due to being in a different memory space than the source ref.
  """
  _spec: pl.BlockSpec = dataclasses.field(metadata=dict(static=True))
  _buffer_type: BufferType = dataclasses.field(metadata=dict(static=True))
  _buffer_count: int = dataclasses.field(metadata=dict(static=True))
  _grid_rank: int | None = dataclasses.field(metadata=dict(static=True))
  window_ref: ArrayRef | None
  copy_in_slot: int | jax.Array | None
  wait_in_slot: int | jax.Array | None
  copy_out_slot: int | jax.Array | None
  wait_out_slot: int | jax.Array | None
  next_fetch: Sequence[jax.Array] | None
  sem_recvs: SemaphoreTuple | None
  sem_sends: SemaphoreTuple | None
  tiling: Tiling | None = dataclasses.field(metadata=dict(static=True))
  is_trivial_windowing: bool = dataclasses.field(
      default=False, metadata=dict(static=True)
  )
  has_allocated_buffer: bool = dataclasses.field(
      default=False, metadata=dict(static=True)
  )

  def __post_init__(self):
    if self.is_buffered and self.buffer_count < 1:
      raise ValueError(
          f"buffer_count must be at least 1, got {self.buffer_count}"
      )
    if self.is_output:
      if self.is_buffered and self.buffer_count > 2:
        raise NotImplementedError(
            "Buffer count >2 not supported for output buffered refs."
        )

  @property
  def spec(self):
    return self._spec

  @property
  def buffer_type(self):
    return self._buffer_type

  @property
  def is_buffered(self) -> bool:
    """Whether this buffer is multiple-buffered."""
    return self._buffer_count > 0

  @property
  def use_lookahead(self) -> bool:
    """Whether this buffer allows lookahead for fetching blocks."""
    return self._grid_rank is not None

  @property
  def buffer_count(self) -> int:
    """Returns the number of buffers used for multiple buffering."""
    if not self.is_buffered:
      raise ValueError("buffer count is undefined")
    return self._buffer_count

  @classmethod
  def create(
      cls,
      spec: pl.BlockSpec,
      dtype_or_type,
      buffer_type,
      buffer_count,
      grid_rank=None,
      use_lookahead=False,
      source_memory_space: tpu_core.MemorySpace | Literal[ANY] = ANY,  # type: ignore[valid-type]
      tiling: Tiling | None = None,
      is_trivial_windowing: bool = False,
  ) -> BufferedRef:
    """Create a BufferedRef.

    Args:
      spec: pallas blockspec.
      dtype_or_type: dtype or aval for buffers. If an aval, the shape is
        ignored.
      buffer_type: enum indicating whether this is an input, output, or in/out
        buffered reference.
      grid_rank: rank of the pipeline grid.
      use_lookahead: whether to enable pipeline lookahead.
      source_memory_space: The memory space of the backing source Ref.
      tiling: The tiling to assume for the buffers.

    Returns:
      Initialized BufferedRef
    """

    # (123, 456) is a dummy shape since we never use ty without
    # calling .update(shape=...) first.
    ty = (
        dtype_or_type
        if isinstance(dtype_or_type, jax_core.AbstractValue)
        else jax_core.ShapedArray((123, 456), dtype_or_type)
    )

    buffer_memory_space = (
          VMEM if spec.memory_space is None else spec.memory_space)
    if buffer_memory_space not in (SMEM, VMEM, HBM):
      raise ValueError(
          f"Unsupported buffer memory space: {buffer_memory_space}"
      )
    if source_memory_space is buffer_memory_space:
      return cls(
          _spec=spec,
          _buffer_type=buffer_type,
          _buffer_count=0,
          _grid_rank=None,
          window_ref=None,  # to be bound to existing ref by the pipeline routine
          copy_in_slot=None,
          wait_in_slot=None,
          copy_out_slot=None,
          wait_out_slot=None,
          next_fetch=None,
          sem_recvs=None,
          sem_sends=None,
          tiling=None,
          is_trivial_windowing=is_trivial_windowing,
          has_allocated_buffer=False,
      )
    else:
      if use_lookahead and grid_rank is None:
        raise ValueError(
            "grid_rank must be specified when use_lookahead is True."
        )

      if is_trivial_windowing:
        buffer_ty = ty
      else:
        block_shape = _get_block_shape(spec)
        if len(block_shape) == 1 and tiling is not Tiling.SPARSE_CORE:
          buffer_ty = ty.update(shape=(buffer_count * block_shape[0],))
        else:
          buffer_ty = ty.update(shape=(buffer_count, *block_shape))
      return cls(
          _spec=spec,
          _buffer_type=buffer_type,
          _buffer_count=buffer_count,
          _grid_rank=grid_rank if use_lookahead else None,
          window_ref=buffer_memory_space.from_type(buffer_ty),
          copy_in_slot=None,
          wait_in_slot=None,
          copy_out_slot=None,
          wait_out_slot=None,
          next_fetch=None,
          sem_recvs=(
              None
              if buffer_type is BufferType.OUTPUT or is_trivial_windowing
              else SemaphoreType.DMA((buffer_count,))
          ),
          sem_sends=(
              None
              if buffer_type is BufferType.INPUT or is_trivial_windowing
              else SemaphoreType.DMA((buffer_count,))
          ),
          tiling=tiling,
          is_trivial_windowing=is_trivial_windowing,
          has_allocated_buffer=True,
      )

  @classmethod
  def input(cls, spec, dtype_or_type, buffer_count=2, **kwargs):
    return cls.create(
        spec, dtype_or_type, BufferType.INPUT, buffer_count, **kwargs
    )

  @classmethod
  def output(cls, spec, dtype_or_type, buffer_count=2, **kwargs):
    return cls.create(
        spec, dtype_or_type, BufferType.OUTPUT, buffer_count, **kwargs
    )

  @classmethod
  def input_output(cls, spec, dtype_or_type, buffer_count=2, **kwargs):
    return cls.create(
        spec, dtype_or_type, BufferType.INPUT_OUTPUT, buffer_count, **kwargs
    )

  def with_spec(self, spec: pl.BlockSpec) -> BufferedRef:
    """Returns a new BufferedRef with the given block spec."""
    return dataclasses.replace(self, _spec=spec)

  def with_next_fetch(
      self,
      next_fetch: Sequence[jax.Array] | None = None,
  ):
    return dataclasses.replace(self, next_fetch=next_fetch)

  def with_slot_index(
      self,
      copy_in_slot: int | jax.Array | None = None,
      copy_out_slot: int | jax.Array | None = None,
      wait_in_slot: int | jax.Array | None = None,
      wait_out_slot: int | jax.Array | None = None,
  ) -> "BufferedRef":
    """Returns a new BufferedRef with the given slot index."""
    new_buf = self
    if copy_in_slot is not None:
      new_buf = dataclasses.replace(new_buf, copy_in_slot=copy_in_slot)
    if copy_out_slot is not None:
      new_buf = dataclasses.replace(new_buf, copy_out_slot=copy_out_slot)
    if wait_in_slot is not None:
      new_buf = dataclasses.replace(new_buf, wait_in_slot=wait_in_slot)
    if wait_out_slot is not None:
      new_buf = dataclasses.replace(new_buf, wait_out_slot=wait_out_slot)
    return new_buf

  @property
  def current_ref(self):
    """Returns the current working slice of the double-buffer."""
    assert not (
        self.window_ref is None
        or isinstance(self.window_ref, state.AbstractRef)
    )
    if not self.is_buffered or self.is_trivial_windowing:
      return self.window_ref
    else:
      if self.is_output:
        slot = self.current_copy_out_slot
      else:
        slot = self.current_wait_in_slot
      return self._window_ref_at(slot)

  @property
  def cumulative_copy_in(self):
    """The cumulative number of copy_ins issued on this buffer."""
    assert self.copy_in_slot is not None
    return self.copy_in_slot

  @property
  def current_copy_in_slot(self):
    """Index in multiple buffer corresponding to the current slot."""
    return lax.rem(self.cumulative_copy_in, jnp.uint32(self.buffer_count))

  @property
  def cumulative_copy_out(self):
    """The cumulative number of copy_outs issued on this buffer."""
    assert self.copy_out_slot is not None
    return self.copy_out_slot

  @property
  def current_copy_out_slot(self):
    """Index in multiple buffer corresponding to the current copy slot."""
    return lax.rem(self.cumulative_copy_out, jnp.uint32(self.buffer_count))

  @property
  def cumulative_wait_in(self):
    """The cumulative number of wait_ins issued on this buffer."""
    assert self.wait_in_slot is not None
    return self.wait_in_slot

  @property
  def current_wait_in_slot(self):
    """Index in multiple buffer corresponding to the current wait slot."""
    return lax.rem(self.cumulative_wait_in, jnp.uint32(self.buffer_count))

  @property
  def cumulative_wait_out(self):
    """The cumulative number of wait_outs issued on this buffer."""
    assert self.wait_out_slot is not None
    return self.wait_out_slot

  @property
  def current_wait_out_slot(self):
    """Index in multiple buffer corresponding to the current wait slot."""
    return lax.rem(self.cumulative_wait_out, jnp.uint32(self.buffer_count))

  @property
  def next_fetch_indices(self):
    """Returns the next grid indices to fetch from if using lookahead."""
    if not self.use_lookahead:
      raise ValueError("Can only get fetch indices if using lookahead.")
    assert self.next_fetch is not None
    return self.next_fetch

  def bind_existing_ref(self, window_ref, indices):
    """For handling VMEM references, the pipeline aliases the existing ref."""
    if not self.is_buffered and not self.has_allocated_buffer:
      if self.is_trivial_windowing:
        return dataclasses.replace(self, window_ref=window_ref)
      return dataclasses.replace(
          self, window_ref=window_ref.at[self.compute_slice(indices)]
      )
    return self

  def unbind_refs(self):
    if not self.is_buffered and not self.has_allocated_buffer:
      return dataclasses.replace(self, window_ref=None)
    return self

  def compute_slice(self, grid_indices):
    """Compute DMA slice from grid indices."""
    indices = self.compute_index(*grid_indices)
    assert self.block_shape is not None
    assert len(self.block_shape) == len(indices)
    indexer = []
    for bd, idx in zip(self.block_shape, indices, strict=True):
      match bd:
        case None | pl.Squeezed():
          # Dimension is squeezed out so we don't do anything.
          indexer.append(idx)
        case pl.Element():
          raise ValueError(
              "Element block dimensions are not supported."
          )
        case pl.BoundedSlice():
          raise ValueError(
              "BoundedSlice block dimensions are not supported."
          )
        case pl.Blocked(block_size):
          indexer.append(_make_block_ds(idx, block_size))
        case int():
          indexer.append(_make_block_ds(idx, bd))
        case _:
          raise ValueError(f"Unsupported block dimension type: {type(bd)}")
    return tuple(indexer)

  def initialize_slots(self) -> "BufferedRef":
    return dataclasses.replace(
        self,
        copy_in_slot=jnp.uint32(0) if self.buffer_type.is_input else None,
        wait_in_slot=jnp.uint32(0) if self.buffer_type.is_input else None,
        copy_out_slot=jnp.uint32(0) if self.buffer_type.is_output else None,
        wait_out_slot=jnp.uint32(0) if self.buffer_type.is_output else None,
        next_fetch=(
            tuple(jnp.int32(0) for _ in range(self._grid_rank))
            if self._grid_rank is not None
            else None
        ),
    )

  def _advance_slot(self, reg_slot, slot_kwarg, predicate) -> "BufferedRef":
    assert reg_slot is not None
    new_current_slot = lax.select(predicate, reg_slot + 1, reg_slot)
    return self.with_slot_index(**{slot_kwarg: new_current_slot})

  def advance_copy_in_slot(
      self, predicate: bool | jax.Array = True
  ) -> "BufferedRef":
    """Switch to the next copy slot."""
    if not self.is_buffered or not self.is_input:
      return self
    return self._advance_slot(self.copy_in_slot, "copy_in_slot", predicate)

  def advance_wait_in_slot(
      self, predicate: bool | jax.Array = True
  ) -> "BufferedRef":
    """Switch to the next wait slot."""
    if not self.is_buffered or not self.is_input:
      return self
    return self._advance_slot(self.wait_in_slot, "wait_in_slot", predicate)

  def advance_copy_out_slot(
      self, predicate: bool | jax.Array = True
  ) -> "BufferedRef":
    """Switch to the next copy slot."""
    if not self.is_buffered or not self.is_output:
      return self
    return self._advance_slot(self.copy_out_slot, "copy_out_slot", predicate)

  def advance_wait_out_slot(
      self, predicate: bool | jax.Array = True
  ) -> "BufferedRef":
    """Switch to the next wait slot."""
    if not self.is_buffered or not self.is_output:
      return self
    return self._advance_slot(self.wait_out_slot, "wait_out_slot", predicate)

  def _window_ref_at(self, slot, window_slice=None):
    assert self.window_ref is not None
    if self.window_ref.ndim > 1:
      return self.window_ref.at[(slot, *(window_slice or ()))]

    # 1D ``window_ref`` stores all slots contiguously.
    n = self.window_ref.shape[0] // self.buffer_count
    if window_slice is None:
      return self.window_ref.at[pl.ds(slot * n, n)]
    assert len(window_slice) == 1
    return self.window_ref.at[
        pl.ds(slot * n + window_slice[0].start, window_slice[0].size)
    ]

  def copy_in(self, src_ref, grid_indices):
    """Starts copy of HBM dma slice into the current slot."""
    assert self.is_input
    if not self.is_buffered: return
    assert not (self.window_ref is None or isinstance(self.window_ref, state.AbstractRef))
    assert self.sem_recvs is not None
    slot = self.current_copy_in_slot
    src_slice = self.get_dma_slice(_ref_to_value_aval(src_ref), grid_indices)
    dst_slice = self._to_window_slice(src_slice)
    tpu_primitives.make_async_copy(
        src_ref.at[src_slice],
        self._window_ref_at(slot, dst_slice),
        self.sem_recvs.at[slot],
    ).start()

  def copy_out(self, dst_ref, grid_indices):
    """Starts copy of HBM dma slice from the current slot."""
    assert self.is_output
    if not self.is_buffered: return
    assert not (self.window_ref is None or isinstance(self.window_ref, state.AbstractRef))
    assert self.sem_sends is not None
    slot = self.current_copy_out_slot
    dst_slice = self.get_dma_slice(_ref_to_value_aval(dst_ref), grid_indices)
    src_slice = self._to_window_slice(dst_slice)
    tpu_primitives.make_async_copy(
        self._window_ref_at(slot, src_slice),
        dst_ref.at[dst_slice],
        self.sem_sends.at[slot],
    ).start()

  def wait_in(self, src_ref, grid_indices):
    """Waits for input copy to finish."""
    assert self.is_input
    if not self.is_buffered: return
    assert not (self.window_ref is None or isinstance(self.window_ref, state.AbstractRef))
    assert self.sem_recvs is not None
    src_slice = self.get_dma_slice(_ref_to_value_aval(src_ref), grid_indices)
    dst_slice = self._to_window_slice(src_slice)
    wait_slot = self.current_wait_in_slot
    tpu_primitives.make_async_copy(
        src_ref.at[src_slice],  # nb: doesn't matter
        self._window_ref_at(
            wait_slot, dst_slice
        ),  # only dst shape is important
        self.sem_recvs.at[wait_slot],
    ).wait()

  def wait_out(self, dst_ref, grid_indices):
    """Waits for output copy to finish."""
    assert self.is_output
    if not self.is_buffered: return
    assert not (self.window_ref is None or isinstance(self.window_ref, state.AbstractRef))
    assert self.sem_sends is not None
    wait_slot = self.current_wait_out_slot
    dst_slice = self.get_dma_slice(_ref_to_value_aval(dst_ref), grid_indices)
    src_slice = self._to_window_slice(dst_slice)
    tpu_primitives.make_async_copy(
        self._window_ref_at(wait_slot, src_slice),  # nb: doesn't matter
        dst_ref.at[dst_slice],  # only dst shape is important
        self.sem_sends.at[wait_slot],
    ).wait()


def fetch_with_lookahead(buffered_ref, src_ref,
                         grid,
                         grid_offsets,
                         predicate: jax.Array | bool = True,
                         max_num_fetches: int | None = None,
                         update_slots: bool = True):
  """Fetch future blocks using unbounded lookahead.

  Args:
    buffered_ref: the BufferedRef to fetch for.
    src_ref: the source Ref.
    grid: the grid bounds.
    grid_offsets: the grid offsets (used for megacore).
    predicate: a boolean predicate for whether to perform the fetch.
    max_num_fetches: the maximum number of fetches to perform. If None,
      this will continually fetch until all copy_in slots are full.
    update_slots: whether to update the register slot indices.
  """
  assert buffered_ref.use_lookahead
  add_offset = lambda x: tuple(
      i + j for i, j in zip(x, grid_offsets, strict=True))
  index_inbound = lambda x: _tuple_lt(x, grid)
  increment_indices = lambda x: _next_index(x, grid, allow_overflow=True)
  def as_uint32(x):
    if isinstance(x, bool):
      return jnp.uint32(x)
    else:
      return x.astype(jnp.uint32)

  fetch_limit = buffered_ref.cumulative_wait_in + buffered_ref.buffer_count
  if max_num_fetches is not None:
    fetch_once_limit = buffered_ref.cumulative_copy_in + max_num_fetches
    # We would like to write jnp.minimum(fetch_limit, fetch_once_limit)
    # but this does not compile in Mosaic.
    fetch_limit = lax.select(fetch_limit < fetch_once_limit,
                             fetch_limit, fetch_once_limit)

  def _loop_cond(carry):
    _, next_indices, cumulative_copy_in = carry
    # Don't fetch more blocks than we have buffers.
    within_limit = cumulative_copy_in < fetch_limit
    # Don't fetch past the end of the grid.
    in_bounds = index_inbound(next_indices)
    return predicate & within_limit & in_bounds

  def _loop_body(carry):
    current_indices, next_indices, cumulative_copy_in = carry
    cur_indices_offset = add_offset(current_indices)
    next_indices_offset = add_offset(next_indices)
    block_indices = buffered_ref.compute_index(*cur_indices_offset)
    next_block_indices = buffered_ref.compute_index(*next_indices_offset)
    will_change = _tuples_differ(block_indices, next_block_indices)
    pred = will_change
    bref = buffered_ref.with_slot_index(copy_in_slot=cumulative_copy_in)
    @pl.when(pred)
    def _start():
      bref.copy_in(src_ref, next_indices_offset)  # pylint: disable=cell-var-from-loop
    next_copy_in = cumulative_copy_in + as_uint32(pred)
    next_next_indices = increment_indices(next_indices)
    return next_indices, next_next_indices, next_copy_in
  current_indices = buffered_ref.next_fetch_indices
  next_fetch = increment_indices(current_indices)
  final_indices, _, final_copy_in_slot = lax.while_loop(
      _loop_cond, _loop_body,
      (current_indices, next_fetch, buffered_ref.cumulative_copy_in))

  buffered_ref = buffered_ref.with_next_fetch(final_indices)
  if update_slots:
    buffered_ref = buffered_ref.with_slot_index(copy_in_slot=final_copy_in_slot)
  return buffered_ref, final_copy_in_slot


# Helper to tree map over BufferedRefs as leaves.
map_brefs = functools.partial(
    jax.tree.map,
    is_leaf=lambda x: isinstance(x, BufferedRefBase)
)


def map_inputs(f, *args):
  """Maps over all input BufferedRefs."""
  def fmap(bref, *f_args):
    if bref.is_input:
      return f(bref, *f_args)
    return bref
  return map_brefs(fmap, *args)


def map_outputs(f, *args):
  """Maps over all output BufferedRefs."""
  def fmap(bref, *f_args):
    if bref.is_output:
      return f(bref, *f_args)
    return bref
  return map_brefs(fmap, *args)


def _filter_indices(
    indices: tuple[int | jax.Array, ...], grid: tuple[int | jax.Array, ...]
) -> tuple[int | jax.Array, ...]:
  return tuple(
      0 if isinstance(g, int) and g == 1 else i
      for i, g in zip(indices, grid, strict=True)
  )


def _next_index(
    indices: tuple[int | jax.Array, ...], grid: tuple[int | jax.Array, ...],
    allow_overflow: bool = False,
) -> tuple[int | jax.Array, ...]:
  """Increments the grid indices by one.

  Args:
    indices: the current grid indices.
    grid: the pallas grid.
    allow_overflow: whether to allow the indices to overflow the grid.
      If False (default), indices will wrap around to zero after reaching the
      maximum grid size. If True, the bounds on the first grid position
      will be ignored.

  Returns:
    The next grid indices.
  """
  out = []
  carry: bool | jax.Array = True
  for position, (i, g) in enumerate(
      reversed(list(zip(indices, grid, strict=True)))):
    inc = jax.lax.select(carry, i + 1, i)
    if allow_overflow and (position == len(grid) - 1):
      carry = False
    else:
      carry = inc == g
    out.append(jax.lax.select(carry, 0, inc))
  if allow_overflow:
    return tuple(reversed(out))
  else:
    return _filter_indices(tuple(reversed(out)), grid)


def _prev_index(
    indices: tuple[int | jax.Array, ...], grid: tuple[int | jax.Array, ...]
) -> tuple[int | jax.Array, ...]:
  out = []
  borrow: bool | jax.Array = True
  for i, g in reversed(list(zip(indices, grid, strict=True))):
    dec = jax.lax.select(borrow, i - 1, i)
    borrow = dec == -1
    out.append(jax.lax.select(borrow, g - 1, dec))
  return _filter_indices(tuple(reversed(out)), grid)


class Scheduler:
  """Sequences input and output copies and waits for a pipeline."""

  def __init__(
      self,
      step: jax.Array,
      indices: tuple[int | jax.Array, ...],
      grid: tuple[int | jax.Array, ...],
      grid_offsets: tuple[int | jax.Array, ...],
      num_stages: int,
      trace_scopes=True,
      _explicit_indices: bool = False,
  ):
    """Initializes scheduler.

    Args:
      step: inner step number.
      indices: current grid indices.
      grid: pallas grid for BufferedRefs.
      grid_offsets: offsets for grid indices (used for megacore).
      num_stages: number of stages in the pipeline.
      trace_scopes: whether to use named_scope to trace blocks in the pipeline.
      _explicit_indices: whether the pipeline uses explicit indices.
    """
    self.step = step
    self.grid = grid
    self.grid_offsets = grid_offsets
    self.num_stages = num_stages
    self.trace_scopes = trace_scopes
    self._explicit_indices = _explicit_indices

    # Total number of linear steps.
    self.num_steps = _grid_size(grid)

    # First and last inner step conditionals.
    self.first_step = step == 0
    self.last_step = step == self.num_steps - 1

    # Derived grid indices for present, previous, and next steps.
    self.indices = tuple(
        i + j for i, j in zip(indices, grid_offsets, strict=True)
    )

    self.prev_indices = tuple(
        i + j
        for i, j in zip(_prev_index(indices, grid), grid_offsets, strict=True)
    )
    next_indices = _next_index(indices, grid)
    self.next_indices = tuple(
        i + j
        for i, j in zip(next_indices, grid_offsets, strict=True)
    )
    self.add_offset = lambda x: tuple(i + j for i, j in zip(x, grid_offsets,
                                                            strict=True))
    # TODO(justinfu): Don't recompute these on each iteration.
    # fetch_indices stores the grid indices indexed by the amount of lookahead.
    # i.e. fetch_indices[2] contains the grid indices 2 iterations
    # ahead.
    self.fetch_indices = [self.indices, self.next_indices]
    fetch_indices = next_indices
    for _ in range(self.num_stages-1):
      fetch_indices = _next_index(fetch_indices, grid)
      self.fetch_indices.append(tuple(
            i + j
            for i, j in zip(fetch_indices, grid_offsets, strict=True)
      ))

  @contextmanager
  def _named_scope(self, name):
    if self.trace_scopes:
      with jax.named_scope(name):
        yield
    else:
      yield

  def grid_env(self):
    if self._explicit_indices:
      return contextlib.nullcontext()
    return pallas_core.grid_env(
        list(map(pallas_core.GridAxis, self.indices, self.grid)))  # pyrefly: ignore[bad-argument-type]  # pyrefly#2385

  def out_of_fetch(self, buffered_ref):
    """Returns whether there are no more blocks to fetch."""
    # Currently this is based on the iteration, but if we want to support
    # lookahead this will depend on whether the lookahead reached the end.
    if not buffered_ref.is_buffered:
      return jnp.bool(False)
    return self.step >= (self.num_steps - buffered_ref.buffer_count + 1)

  def has_changed(self, buffered_ref):
    if not buffered_ref.is_buffered or buffered_ref.is_trivial_windowing:
      return False
    indices = buffered_ref.compute_index(*self.indices)
    prev_indices = buffered_ref.compute_index(*self.prev_indices)
    return _tuples_differ(indices, prev_indices)

  def will_change_current(self, buffered_ref):
    if not buffered_ref.is_buffered or buffered_ref.is_trivial_windowing:
      return False
    indices = buffered_ref.compute_index(*self.indices)
    next_indices = buffered_ref.compute_index(*self.next_indices)
    return _tuples_differ(indices, next_indices)

  def will_change_fetch(self, buffered_ref):
    if not buffered_ref.is_buffered or buffered_ref.is_trivial_windowing:
      return False
    if buffered_ref.buffer_count < 2:
      return self.has_changed(buffered_ref)
    indices = buffered_ref.compute_index(
        *self.fetch_indices[buffered_ref.buffer_count-2])
    next_indices = buffered_ref.compute_index(
        *self.fetch_indices[buffered_ref.buffer_count-1])
    return _tuples_differ(indices, next_indices)

  def alias_local_refs(self, buffered_ref, ref):
    return buffered_ref.bind_existing_ref(ref, self.indices)

  def unalias_local_refs(self, buffered_ref):
    return buffered_ref.unbind_refs()

  # SCHEDULE ----------------------------------------------------------------

  # Below is the sequence of conditional waits and copies used for inputs,
  # outputs, and in-outs.

  def initialize_step(self, buffered_ref, src_ref, step=0):

    with self._named_scope(f"ep_initialize_{step}"):

      if not buffered_ref.is_input or not buffered_ref.is_buffered:
        return buffered_ref

      if (step + 1) >= buffered_ref.buffer_count:
        return buffered_ref

      if buffered_ref.use_lookahead:
        if step == 0:
          # We always fetch the first block.
          @pl.when(self.first_step)
          def _start():
            buffered_ref.copy_in(src_ref,
              self.add_offset(buffered_ref.next_fetch_indices))  # pylint: disable=cell-var-from-loop
          buffered_ref = buffered_ref.advance_copy_in_slot(self.first_step)
        else:
          buffered_ref, _ = fetch_with_lookahead(
              buffered_ref,
              src_ref,
              self.grid,
              self.grid_offsets,
              predicate=self.first_step,
              max_num_fetches=1,
          )
      else:
        if step == 0:
          predicate = self.first_step
          fetch_indices = self.fetch_indices[step]
        else:
          fetch_indices = self.fetch_indices[step]
          prev_grid_indices = self.fetch_indices[step - 1]
          block_indices = buffered_ref.compute_index(*fetch_indices)
          prev_block_indices = buffered_ref.compute_index(*prev_grid_indices)
          block_changed = _tuples_differ(block_indices, prev_block_indices)
          predicate = self.first_step & block_changed
        @pl.when(predicate)  # pylint: disable=cell-var-from-loop
        def _start():
          buffered_ref.copy_in(src_ref, fetch_indices)  # pylint: disable=cell-var-from-loop
        buffered_ref = buffered_ref.advance_copy_in_slot(predicate)
    return buffered_ref

  def wait_in(self, buffered_ref, src_ref) -> "BufferedRef":
    if buffered_ref.is_trivial_windowing:
      return buffered_ref
    pred = self.has_changed(buffered_ref) | self.first_step

    @pl.when(pred)
    @self._named_scope("ep_wait_in")
    def _wait():
      if buffered_ref.is_input:
        buffered_ref.wait_in(src_ref, self.indices)
    return buffered_ref

  def copy_in(self, buffered_ref, src_ref) -> "BufferedRef":
    if buffered_ref.is_trivial_windowing:
      return buffered_ref
    pred = (self.will_change_fetch(buffered_ref) &
            ~self.out_of_fetch(buffered_ref))

    # Single-buffered refs skip the prologue, so the first copy_in in the
    # loop must always fire to populate the buffer before wait_in.
    if buffered_ref.is_buffered and buffered_ref.buffer_count < 2:
      pred = pred | self.first_step
    if not buffered_ref.is_input:
      return buffered_ref

    if buffered_ref.use_lookahead:
      buffered_ref, _ = fetch_with_lookahead(
          buffered_ref, src_ref, self.grid, self.grid_offsets, predicate=True
      )
    else:
      @pl.when(pred)
      @self._named_scope("ep_copy_in")
      def _send():
        if buffered_ref.is_input and buffered_ref.is_buffered:
          buffered_ref.copy_in(src_ref,
            self.fetch_indices[buffered_ref.buffer_count-1])
      buffered_ref = buffered_ref.advance_copy_in_slot(
          pred & buffered_ref.is_input)
    return buffered_ref

  def wait_out(self, buffered_ref, dst_ref) -> "BufferedRef":
    if buffered_ref.is_trivial_windowing:
      return buffered_ref
    pred = self.has_changed(buffered_ref) & ~self.first_step
    @pl.when(pred)
    @self._named_scope("ep_wait_out")
    def _wait():
      if buffered_ref.is_output:
        # Note: As implemented, the current scheduler cannot support multiple
        # buffering on outputs. In order to do so properly, we need to save
        # the indices for which the copy_out was issued, and wait on them
        # here. In the current schedule we always immediately wait_out
        # on the iteration after the copy_out, so the prev_indices is always
        # the correct grid index to wait on.
        buffered_ref.wait_out(dst_ref, self.prev_indices)
    return buffered_ref.advance_wait_out_slot(pred & buffered_ref.is_output)

  def copy_out(self, buffered_ref, dst_ref) -> "BufferedRef":
    if buffered_ref.is_trivial_windowing:
      return buffered_ref
    pred = self.will_change_current(buffered_ref) | self.last_step

    @pl.when(pred)
    @self._named_scope("ep_copy_out")
    def _copy_out():
      if buffered_ref.is_output:
        buffered_ref.copy_out(dst_ref, self.indices)

    return buffered_ref.advance_copy_out_slot(pred & buffered_ref.is_output)

  def finalize(self, buffered_ref, dst_ref):
    if buffered_ref.is_trivial_windowing:
      return
    pred = self.last_step

    @pl.when(pred)
    @self._named_scope("ep_finalize")
    def _end():
      if buffered_ref.is_output:
        buffered_ref.wait_out(dst_ref, self.indices)

  def advance_slots(self, buffered_ref):
    if buffered_ref.is_input:
      pred = self.will_change_current(buffered_ref) | self.last_step
      buffered_ref = buffered_ref.advance_wait_in_slot(pred)
    # Currently we advance copy_in and output slots after their respective
    # operation.
    return buffered_ref

  # END SCHEDULE --------------------------------------------------------------


# Main pipeline methods


def _normalize_specs(specs: Any) -> tuple[pl.BlockSpec, ...]:
  if not isinstance(specs, (list, tuple)):
    specs = (specs,)
  if isinstance(specs, list):
    specs = tuple(specs)
  return specs


def _make_pipeline_allocations(
    *refs,
    in_specs=(),
    out_specs=(),
    tiling: Tiling | None = None,
    grid=(),
):
  """Create BufferedRefs for the pipeline.

  This function creates buffered refs for an inner pipeline that can be
  created at the top-level of a pallas call such that they may be reused across
  multiple invocations of the inner pipeline.

  Args:
    in_specs: input pallas block specs
    out_specs: output pallas block specs
    grid: grid to use for the pipeline.

  Returns:
    A list of BufferedRefs, one corresponding to each ref specified in the
    in_specs and out_specs.
  """
  # TODO(levskaya): generalize argument tree handling here and in emit_pipeline.
  num_in_specs = len(in_specs)
  in_specs = _normalize_specs(in_specs)
  out_specs = _normalize_specs(out_specs)
  in_refs = refs[:num_in_specs]
  out_refs = refs[num_in_specs:]
  def make_input_bref(in_spec, in_ref):
    in_aval = _ref_to_value_aval(in_ref)
    buffer_count = 2
    use_lookahead = False
    if has_buffering := in_spec.pipeline_mode is not None:
      buffer_count = in_spec.pipeline_mode.buffer_count
      use_lookahead = in_spec.pipeline_mode.use_lookahead
    if use_lookahead and grid is None:
      raise ValueError("Grid must be specified when using lookahead.")
    is_trivial = _spec_has_trivial_windowing(in_spec, grid, in_aval.shape)
    if not has_buffering and is_trivial:
      buffer_count = 1

    return BufferedRef.input(
        in_spec,
        in_aval,
        buffer_count,
        grid_rank=len(grid),
        use_lookahead=use_lookahead,
        source_memory_space=in_ref.memory_space,
        tiling=tiling,
        is_trivial_windowing=is_trivial,
    )
  in_brefs = jax.tree.map(make_input_bref, in_specs, in_refs)
  def make_output_bref(out_spec, out_ref):
    out_aval = _ref_to_value_aval(out_ref)
    buffer_count = 2
    if has_buffering := out_spec.pipeline_mode is not None:
      buffer_count = out_spec.pipeline_mode.buffer_count
      if out_spec.pipeline_mode.use_lookahead:
        raise ValueError("Output buffering does not support lookahead.")
    is_trivial = _spec_has_trivial_windowing(out_spec, grid, out_aval.shape)
    if not has_buffering and is_trivial:
      buffer_count = 1

    return BufferedRef.output(
        out_spec,
        out_aval,
        buffer_count,
        source_memory_space=out_ref.memory_space,
        tiling=tiling,
        is_trivial_windowing=is_trivial,
    )
  out_brefs = jax.tree.map(make_output_bref, out_specs, out_refs)
  return (*in_brefs, *out_brefs)


def _partition_grid(
    grid: tuple[int | jax.Array, ...],
    core_axis: tuple[int | str, ...] | int | str | None,
    dimension_semantics: tuple[GridDimensionSemantics, ...] | None,
) -> tuple[tuple[int | jax.Array, ...], tuple[int | jax.Array, ...]]:
  if core_axis is None:
    # We aren't partitioning the grid
    return grid, (0,) * len(grid)
  if isinstance(core_axis, int):
    num_cores = pl.num_programs(core_axis)
    core_id = pl.program_id(core_axis)
  else:
    num_cores = jax.lax.axis_size(core_axis)
    core_id = jax.lax.axis_index(core_axis)
  # Check that num_cores is statically known
  if not isinstance(num_cores, int):
    raise NotImplementedError(
        f"Cannot partition grid over dynamic number of cores: {core_axis=}"
    )
  if num_cores == 1:
    # We aren't partitioning the grid
    return grid, (0,) * len(grid)

  # If dimension_semantics aren't provided, we assume it is all arbitrary.
  if dimension_semantics is None:
    dimension_semantics = (ARBITRARY,) * len(grid)
  if len(dimension_semantics) != len(grid):
    raise ValueError("dimension_semantics must be the same length as grid.")

  parallel_dimensions = {
      i for i, d in enumerate(dimension_semantics) if d == PARALLEL
  }
  # If there are no parallel dimensions, we can't partition the grid
  if not parallel_dimensions:
    # TODO(sharadmv): enable running kernel on just one core
    raise NotImplementedError(
        "Cannot partition over cores without parallel grid dimensions:"
        f" {dimension_semantics=}"
    )

  # Try to find a divisible dimension to partition the grid on
  divisible_dimensions = {
      i
      for i in parallel_dimensions
      if isinstance(grid[i], int) and grid[i] % num_cores == 0
  }
  if divisible_dimensions:
    first_divisible_dimension, *_ = (
        i for i in range(len(dimension_semantics)) if i in divisible_dimensions
    )
    partitioned_dim_size = grid[first_divisible_dimension] // num_cores
    partitioned_dim_offset = core_id * partitioned_dim_size
    new_grid = jax_util.tuple_update(
        grid, first_divisible_dimension, partitioned_dim_size
    )
    offsets = jax_util.tuple_update(
        (0,) * len(grid),
        first_divisible_dimension,
        partitioned_dim_offset,
    )
    return new_grid, offsets

  # Separate the remaining dimensions into dynamic and static.
  dynamic_dims = [
      i
      for i in range(len(grid))
      if i in parallel_dimensions and not isinstance(grid[i], int)
  ]
  static_dims = [
      i
      for i in range(len(grid))
      if i in parallel_dimensions and isinstance(grid[i], int)
  ]

  if len(dynamic_dims) > 1:
    raise NotImplementedError(
        f"Cannot partition over multiple dynamic parallel dimensions: {grid=}"
    )

  if dynamic_dims and not static_dims:
    # Exactly one dynamic dimension and no static non-divisible dimensions
    partition_dimension = dynamic_dims[0]
  else:
    # No divisible static dimensions, so we can't evenly partition the grid.
    # Let's pick the largest dimension and try to divide it as evenly as
    # possible.
    # TODO(sharadmv): take the product of many nondivisible dimensions to
    # potentially divide it more evenly
    largest_parallel_dimension = max(grid[i] for i in static_dims)
    partition_dimension, *_ = (
        i for i in static_dims if grid[i] == largest_parallel_dimension
    )

  base_num_iters, rem = divmod(grid[partition_dimension], num_cores)
  # We have some remainder iterations that we need to assign somewhere. We
  # know that rem < num_cores, so we can assign one extra iteration to each
  # core except for the last (num_cores - rem).
  num_iters = jnp.where(core_id < rem, base_num_iters + 1, base_num_iters)
  new_grid = jax_util.tuple_update(grid, partition_dimension, num_iters)
  # Ordinarily, we would compute the offset as:
  #   grid_offset = pl.program_id(core_axis) * num_iters
  # However, since we have some cores that don't have an extra iteration, we
  # need to adjust the offset by `rem`.
  grid_offset = jnp.where(
      core_id < rem,
      core_id * num_iters,
      core_id * base_num_iters + rem,
  )
  offsets = jax_util.tuple_update(
      (0,) * len(grid),
      partition_dimension,
      grid_offset,
  )
  return new_grid, offsets  # type: ignore[return-value]


def sync_copy(src: REF | BufferedRef, dst: REF | BufferedRef, indices):
  """Perform a synchronous copy from src to dst."""
  bref: BufferedRef
  hbm_ref: REF
  if isinstance(src, BufferedRef):
    bref = src
    if isinstance(dst, BufferedRef):
      raise ValueError("Only one of src or dst can be a BufferedRef.")
    hbm_ref = dst
    copy_in = False
  else:
    if not isinstance(dst, BufferedRef):
      raise ValueError("One of src or dst must be a BufferedRef.")
    bref = dst
    hbm_ref = src
    copy_in = True
  window_ref = bref.current_ref
  if not bref.is_trivial_windowing:
    hbm_slice = bref.get_dma_slice(_ref_to_value_aval(hbm_ref), indices)
    bref_slice = bref._to_window_slice(hbm_slice)
    hbm_ref = hbm_ref.at[hbm_slice]
    window_ref = window_ref.at[bref_slice]
  if copy_in:
    tpu_helpers.sync_copy(hbm_ref, window_ref)
  else:
    tpu_helpers.sync_copy(window_ref, hbm_ref)


def emit_pipeline(
    body,
    *,
    grid: tuple[int | jax.Array, ...],
    in_specs=(),
    out_specs=(),
    tiling: Tiling | None = None,
    core_axis: tuple[int, ...] | int | None = None,
    core_axis_name: tuple[str, ...] | str | None = None,
    dimension_semantics: tuple[GridDimensionSemantics, ...] | None = None,
    trace_scopes: bool = True,
    no_pipelining: bool = False,
    _explicit_indices: bool = False,
):
  """Creates a function to emit a manual pallas pipeline.

  This has the same semantics as pallas_call but is meant to be called inside
  pallas_call for nesting grids. This is useful when you need to have separate
  windowing strategies for communication and computation.

  Args:
    body: pallas kernel to set up pipeline for.
    grid: a pallas grid definition.
    in_specs: input pallas block specs
    out_specs: output pallas block specs
    tiling: optional tiling to assume for the refs.
    core_axis: optional int or tuple of int, indicates whether or not to
      partition the grid along the core axis.
    core_axis_name: optional str or tuple of str, indicates whether or not to
      partition the grid along the core axis.
    dimension_semantics: optional tuple of GridDimensionSemantics (e.g. PARALLEL
      or ARBITRARY).
    trace_scopes: optional bool, indicates whether to annotate each region in
      the pipeline using named_scope.
    no_pipelining: If True, turns off pipelining and all copies will be made
      synchronous. This is useful for debugging multiple-buffering related bugs.
    _explicit_indices: If True, the body will receive the iteration indices as
      its first argument. This parameter is meant for internal use only.
  """
  if any(not isinstance(d, (int, jax.Array)) for d in grid):
    grid_types = tuple(type(d) for d in grid)
    raise ValueError(
        f"Grid must consist of Python integers and JAX Arrays: {grid_types}"
    )
  if not (core_axis is None or core_axis_name is None):
    raise ValueError("core_axis and core_axis_name cannot both be provided.")
  core_axis_ = core_axis_name if core_axis is None else core_axis
  grid, grid_offsets = _partition_grid(grid, core_axis_, dimension_semantics)

  num_steps = _grid_size(grid)
  in_specs = _normalize_specs(in_specs)
  out_specs = _normalize_specs(out_specs)
  get_buffer_count = lambda spec: (spec.pipeline_mode.buffer_count if
    (spec is not None and spec.pipeline_mode is not None) else 2)
  flattened_specs = jax.tree.leaves((in_specs, out_specs))
  max_buffer_count = max((2, *map(get_buffer_count, flattened_specs)))

  def pipeline(
      *refs: Any,
      scratches=None,
      allocations=None,
      body_prologue=None,
  ):
    """
    Run the pipeline.

    Args:
      *ref_args: a list of pallas refs (or more generally a list of pytrees of
        pallas refs)
      scratches: scratch buffers for the inner kernel
      allocations: a list of BufferedRefs, one corresponding to each ref
      body_prologue: For running code within the grid environment before the
        body is run. Useful for updating manual refs.
    """
    if scratches is None:
      scratches = ()
    if allocations is None:
      # run with inline scoped allocations
      return primitives.run_scoped(
          lambda allocations: pipeline(
              *refs,
              scratches=scratches,
              allocations=allocations,
          ),
          _make_pipeline_allocations(
              *refs,
              in_specs=in_specs,
              out_specs=out_specs,
              grid=grid,
              tiling=tiling,
          ),
      )
    if isinstance(allocations, list):
      allocations = tuple(allocations)

    def make_scheduler(step, indices):
      return Scheduler(
          step,
          indices,
          grid,
          grid_offsets=grid_offsets,
          num_stages=max_buffer_count,
          trace_scopes=trace_scopes,
          _explicit_indices=_explicit_indices,
      )

    def loop_body(step, carry):
      unaliased_brefs, indices = carry
      indices = _filter_indices(indices, grid)
      scheduler = make_scheduler(step, indices)
      with scheduler.grid_env():
        # prepare any local VMEM aliases
        brefs = map_brefs(scheduler.alias_local_refs, unaliased_brefs, refs)
        # loop input handling phase
        brefs = map_brefs(scheduler.copy_in, brefs, refs)
        brefs = map_brefs(scheduler.wait_in, brefs, refs)

        # run the kernel!
        if body_prologue is not None:
          body_prologue()
        current_refs = map_brefs(lambda x: x.current_ref, brefs)
        with scheduler._named_scope("ep_run_kernel"):
          if _explicit_indices:
            body(scheduler.indices, *current_refs, *scratches)
          else:
            body(*current_refs, *scratches)

        # loop output handling phase
        brefs = map_brefs(scheduler.copy_out, brefs, refs)
        brefs = map_brefs(scheduler.wait_out, brefs, refs)

        brefs = map_brefs(scheduler.advance_slots, brefs)
        # Unbind window_refs for VMEM-backed buffers. Without this
        # we will be returning TransformedRefs which are not valid
        # JAX types.
        brefs = map_brefs(scheduler.unalias_local_refs, brefs)
      return brefs, _next_index(indices, grid)

    if no_pipelining:
      # Debugging mode where all copies are synchronous.
      initial_indices = (0,) * len(grid)
      brefs = map_brefs(lambda bref: bref.initialize_slots(), allocations)

      @functools.partial(
          jax.lax.fori_loop,
          0,
          num_steps,
          init_val=(brefs, initial_indices),
      )
      def _loop_body(step, carry):
        brefs, indices = carry
        indices = _filter_indices(indices, grid)
        scheduler = make_scheduler(step, indices)
        with scheduler.grid_env():
          # prepare any local VMEM aliases
          brefs = map_brefs(scheduler.alias_local_refs, brefs, refs)
          # loop input handling phase
          copy_in = lambda bref, ref: sync_copy(ref, bref, indices)
          map_inputs(copy_in, brefs, refs)
          # run the kernel!
          if body_prologue is not None:
            body_prologue()
          current_refs = map_brefs(lambda x: x.current_ref, brefs)
          with scheduler._named_scope("ep_run_kernel"):
            if _explicit_indices:
              body(scheduler.indices, *current_refs, *scratches)
            else:
              body(*current_refs, *scratches)
          # loop output handling phase
          copy_out = lambda bref, ref: sync_copy(bref, ref, indices)
          map_outputs(copy_out, brefs, refs)
        brefs = map_brefs(scheduler.unalias_local_refs, brefs)
        return brefs, _next_index(indices, grid)
    else:
      @pl.when(num_steps > 0)
      def _():
        # pipeline prologue
        initial_indices = (0,) * len(grid)
        scheduler = make_scheduler(0, initial_indices)
        brefs = map_brefs(lambda bref: bref.initialize_slots(), allocations)
        def _sync_copy_in(bref, ref):
          if bref.is_trivial_windowing and bref.window_ref is not None:
            sync_copy(ref, bref, initial_indices)

        map_inputs(_sync_copy_in, brefs, refs)
        with scheduler.grid_env():
          # We issue num_stages-1 prefetch copies per buffer.
          # We iterate over steps in the outer loop because we want to
          # queue all iteration 0 prefetches before iteration 1, and so on.
          for step in range(scheduler.num_stages - 1):
            brefs = map_brefs(functools.partial(
                scheduler.initialize_step, step=step),
                brefs, refs)

        # pipeline loop
        brefs, next_indices = lax.fori_loop(
            0, num_steps, loop_body, (brefs, initial_indices)
        )

        # pipeline epilogue
        final_indices = _prev_index(next_indices, grid)
        scheduler = make_scheduler(num_steps - 1, final_indices)
        with scheduler.grid_env():
          map_brefs(scheduler.finalize, brefs, refs)

        def _sync_copy_out(bref, ref):
          if bref.is_trivial_windowing and bref.window_ref is not None:
            sync_copy(bref, ref, initial_indices)

        map_outputs(_sync_copy_out, brefs, refs)

  return pipeline


def emit_pipeline_with_allocations(
    body,
    *,
    grid,
    in_specs=(),
    out_specs=(),
):
  """Creates pallas pipeline and top-level allocation preparation functions.

  Args:
    body: pallas kernel to set up pipeline for.
    grid: a pallas grid definition.
    in_specs: input pallas block specs
    out_specs: output pallas block specs

  Returns:
    (emit_pipeline, make_allocations) function pair, where
      - emit_pipeline is the pallas pipeline function.
      - make_allocations is a function to create buffered refs for the inner
        pipeline that can be created at the top-level of a pallas call to be
        reused across multiple invocations of the inner pipeline.
  """
  make_allocations = functools.partial(_make_pipeline_allocations,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid)
  pipeline = emit_pipeline(
      body,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs)
  return pipeline, make_allocations
