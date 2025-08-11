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
from contextlib import contextmanager
import dataclasses
import enum
import functools
from typing import Any, Union

import jax
from jax import lax
from jax import tree_util
from jax._src import util as jax_util
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives as primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import helpers as tpu_helpers
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax.experimental import pallas as pl
from jax.extend.backend import get_default_device
import jax.numpy as jnp
import numpy as np


SMEM = tpu_core.MemorySpace.SMEM
VMEM = tpu_core.MemorySpace.VMEM
ANY = tpu_core.MemorySpace.ANY
REF = pallas_core.MemoryRef
GridDimensionSemantics = tpu_core.GridDimensionSemantics
PARALLEL = tpu_core.PARALLEL
ARBITRARY = tpu_core.ARBITRARY
SemaphoreType = tpu_core.SemaphoreType
SemaphoreTuple = jax.Array
ArrayRef = Union[REF, jax.Array]

GridIndices = tuple[jax.Array, ...]
CondVal = Union[jax.Array, bool]
PipelineBlockSpecs = Union[Sequence[pallas_core.BlockSpec], Any]
PipelineRefs = Union[Sequence[REF], Any]


# TODO(sharadmv): make this a parameter and make it queryable from the Device.
_TILING = (8, 128)

def _broadcast_pytree_to(from_pytree, to_pytree):
  """Broadcast a prefix pytree to a given full tree."""
  proxy = object()
  treedef = tree_util.tree_structure(to_pytree)
  broadcast_leaves = []
  def add_leaves(i, x):
    broadcast_leaves.extend(
        [i] * tree_util.tree_structure(x).num_leaves)
  try:
    tree_util.tree_map(add_leaves, from_pytree, to_pytree,
                       is_leaf=lambda x: x is None)
  except ValueError:
    raise ValueError(f"Cannot broadcast tree {from_pytree} "
                     f"to full tree structure {treedef}.") from None
  broadcast_leaves = [None if a is proxy else a for a in broadcast_leaves]
  assert len(broadcast_leaves) == treedef.num_leaves
  return tree_util.tree_unflatten(treedef, broadcast_leaves)


@jax_util.cache(trace_context_in_key=False)
def _get_tpu_generation() -> int:
  kind = get_default_device().device_kind
  if kind.endswith(' lite'):
    kind = kind[:-len(' lite')]
  if kind.startswith("TPU v"):
    return int(kind[5])
  else:
    assert "TPU7x" in kind
    return 7

def _make_tiling(shape: tuple[int, ...], dtype: np.dtype) -> tuple[int, ...]:
  # For a n-dimensional shape, returns (8, 128) for the last 2 dimensions
  # and 1 for the leading n - 2. For example, (256, 256) -> (8, 128) and
  # (2, 3, 128, 128) -> (1, 1, 8, 128).
  if len(shape) < 2:
    raise ValueError(f"Shape must have at least 2 dimensions: {shape=}")
  leading_dims, final_dims = shape[:-2], shape[-2:]
  # We want to find the minimum power of 2 that fits the second-minor dimension
  # of shape, with maximum value 8.
  second_minor, _ = final_dims
  packing = 4 // dtype.itemsize
  max_tiling = _TILING[0]
  second_minor_tiling = (1 + int(_get_tpu_generation() < 4)) * packing
  while second_minor_tiling < min(second_minor, max_tiling):
    second_minor_tiling *= 2
  return (*(1,) * len(leading_dims), second_minor_tiling, _TILING[1])


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

def _create_blocked_slice(block_index: jax.Array | int,
                          block_size: int,
                          dim_size: int,
                          tiling: int):
  block_start = block_size * block_index
  if (dim_rem := dim_size % block_size) == 0:
    return pl.ds(block_start, block_size)
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
                          tiling: int):
  if block_size % tiling != 0:
    raise ValueError(f"Block size must divide tiling: {block_size=}, {tiling=}")
  # We assume by construction that slice_size <= block_size. We also assume
  # that the slice_start is already aligned to the tiling.

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
    tiling: int
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


class BufferType(enum.Enum):
  """Buffer type for the arguments to an emitted pipeline."""
  INPUT = 1
  OUTPUT = 2
  ACCUMULATOR = 3
  INPUT_OUTPUT = 4

  MANUAL = 5

  @property
  def is_input(self):
    return self in [
        BufferType.INPUT,
        BufferType.ACCUMULATOR,
        BufferType.INPUT_OUTPUT,
    ]

  @property
  def is_output(self):
    return self in [
        BufferType.OUTPUT,
        BufferType.ACCUMULATOR,
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
  def is_accumulator(self):
    return self.buffer_type == BufferType.ACCUMULATOR

  @property
  def is_input_output(self):
    return self.buffer_type == BufferType.INPUT_OUTPUT

  @property
  def is_manual(self):
    return self.buffer_type == BufferType.MANUAL

  def init_slots(self):
    """Initialize slot indices."""
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

  def load_slots(self, predicate: bool | jax.Array = True) -> BufferedRefBase:
    """Load slot information into registers."""
    raise NotImplementedError()

  def save_slots(self, predicate: bool | jax.Array = True):
    """Save slot information from registers."""
    raise NotImplementedError()

  @property
  def block_shape(self) -> Sequence[pl.BlockDim | int | None] | None:
    return self.spec.block_shape

  @property
  def compute_index(self):
    return self.spec.index_map

  def get_dma_slice(self, src_shape, src_dtype, grid_indices):
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
    if len(src_shape) < 2:
      raise NotImplementedError("Must use >1D values.")

    tiling = _make_tiling(src_shape, src_dtype)
    block_indices = self.compute_index(*grid_indices)
    return tuple(
        _make_block_slice(bi, bs, ss, t)
        for bi, bs, ss, t in zip(
            block_indices, self.block_shape, src_shape, tiling, strict=True
        )
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


# TODO(justinfu): Refactor and rename slot fields to reflect cumulative values
# instead of slot index.
@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class BufferedRef(BufferedRefBase):
  """A helper class to automate VMEM double buffering in pallas pipelines.

  Attributes:
    spec: pallas blockspec.
    dtype: dtype for buffers.
    buffer_type: enum indicating whether this is an input, output, or in/out
      accumulator buffered reference.
    window_ref: a multiple-buffer to hold the working and dirty buffers used
      to copy into and out of.  In the case of a BufferedRef targeting a VMEM
      reference, this simply points to the existing ref.
    accum_ref: accumulating buffer used by accumulator BufferedRefs.
    copy_in_slot: current slot to copy in for the working buffer.
    copy_out_slot: current slot to copy out for the working buffer.
    wait_in_slot: current slot to wait in for the working buffer.
    wait_out_slot: current slot to wait out for the working buffer.
    next_fetch_smem: Holds the next grid indices to fetch for lookahead. This
      is the SMEM backing buffer used to persist state between pipeline
      invocations.
    next_fetch_sreg: Holds the next grid indices to fetch for lookahead. This
      is the register state used to track the indices within the pipeline loop.
    sem_recvs: Multiple buffered semaphores for input DMAs.
    sem_sends: Multiple buffered semaphores for output DMAs.
    block_shape: passthrough property for the BlockSpec's block_shape.
    compute_index: passthrough property for the BlockSpec's compute_index.
    memory_space: passthrough property for the BlockSpec's memory_space.
    current_ref: points to the current working slice of the double-buffer.
    is_input: whether this BufferedRef acts as a pipeline input.
    is_output: whether this BufferedRef acts as a pipeline output.
    is_accumulator: whether this BufferedRef is an accumulator.
    is_input_output: whether this BufferedRef is an input/output without
      automatic accumulation.
    swap: Tracks whether the BufferedRef slots need to be swapped before next
      copy.
  """
  _spec: pl.BlockSpec       # static metadata
  dtype: Any                # static metadata
  _buffer_type: BufferType  # static metadata
  window_ref: ArrayRef | None
  accum_ref: ArrayRef | None
  copy_in_slot: ArrayRef | None
  wait_in_slot: ArrayRef | None
  copy_out_slot: ArrayRef | None
  wait_out_slot: ArrayRef | None
  _copy_in_slot_reg: int | jax.Array | None
  _wait_in_slot_reg: int | jax.Array | None
  _copy_out_slot_reg: int | jax.Array | None
  _wait_out_slot_reg: int | jax.Array | None
  next_fetch_smem: Sequence[jax.Array] | None
  next_fetch_sreg: Sequence[jax.Array] | None
  sem_recvs: SemaphoreTuple | None
  sem_sends: SemaphoreTuple | None
  # TODO(ramiroleal): Improve prefetch/postyeet interface to avoid
  # using this ref.
  swap: ArrayRef | None

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
    slots = [self.copy_in_slot, self.wait_in_slot,
             self.copy_out_slot, self.wait_out_slot]
    return any(x is not None for x in slots)

  @property
  def use_lookahead(self) -> bool:
    """Whether this buffer allows lookahead for fetching blocks."""
    return self.next_fetch_smem is not None

  @property
  def buffer_count(self) -> int:
    """Returns the number of buffers used for multiple buffering."""
    if not self.is_buffered:
      raise ValueError("buffer count is undefined")
    return self.window_ref.shape[0]  # type: ignore[union-attr]

  def tree_flatten(self):
    return (
        (
            self.window_ref,
            self.accum_ref,
            self.copy_in_slot,
            self.wait_in_slot,
            self.copy_out_slot,
            self.wait_out_slot,
            self._copy_in_slot_reg,
            self._wait_in_slot_reg,
            self._copy_out_slot_reg,
            self._wait_out_slot_reg,
            self.next_fetch_smem,
            self.next_fetch_sreg,
            self.sem_recvs,
            self.sem_sends,
            self.swap,
        ),
        (self._spec, self.dtype, self._buffer_type),
    )

  @classmethod
  def tree_unflatten(cls, meta, data):
    return cls(*meta, *data)

  @staticmethod
  def buffer_types() -> type[BufferType]:
    return BufferType

  @classmethod
  def create(cls, spec: pl.BlockSpec, dtype, buffer_type, buffer_count,
             needs_swap_ref=True,
             grid_rank=None,
             use_lookahead=False,
             source_memory_space: tpu_core.MemorySpace = ANY) -> BufferedRef:
    """Create a BufferedRef.

    Args:
      spec: pallas blockspec.
      dtype: dtype for buffers.
      buffer_type: enum indicating whether this is an input, output, or in/out
        accumulator buffered reference.
      needs_swap_ref: whether a swap slots tracker needs to be allocated.
      grid_rank: rank of the pipeline grid.
      use_lookahead: whether to enable pipeline lookahead.
      source_memory_space: The memory space of the backing source Ref.

    Returns:
      Initialized BufferedRef
    """
    block_shape = _get_block_shape(spec)
    if buffer_type is BufferType.ACCUMULATOR:
      accum_ref = VMEM(block_shape, dtype)
    else:
      accum_ref = None
    if source_memory_space == VMEM:
      # We don't need to do any double-buffering in the case that our pipeline
      # reference is already in VMEM, we just need allocate the accumulation
      # buffer and we will refer to the original reference slices directly.
      if spec.memory_space not in (VMEM, None):
        raise ValueError(
            f"Cannot hold a non-buffered ref in {spec.memory_space=}")
      return cls(
          _spec=spec,
          dtype=dtype,
          _buffer_type=buffer_type,
          window_ref=None,  # to be bound to existing ref by the pipeline routine
          accum_ref=accum_ref,
          copy_in_slot=None,
          wait_in_slot=None,
          copy_out_slot=None,
          wait_out_slot=None,
          _copy_in_slot_reg=None,
          _wait_in_slot_reg=None,
          _copy_out_slot_reg=None,
          _wait_out_slot_reg=None,
          next_fetch_smem=None,
          next_fetch_sreg=None,
          sem_recvs=None,
          sem_sends=None,
          swap=None,
      )
    else:
      buffer_memory_space = (
          VMEM if spec.memory_space is None else spec.memory_space)
      if buffer_memory_space not in (SMEM, VMEM):
        raise ValueError(
            f"Unsupported buffer memory space: {buffer_memory_space}"
        )
      if use_lookahead and grid_rank is None:
        raise ValueError(
            "grid_rank must be specified when use_lookahead is True."
        )
      return cls(
          _spec=spec,
          dtype=dtype,
          _buffer_type=buffer_type,
          window_ref=buffer_memory_space((buffer_count,) + block_shape, dtype),
          accum_ref=accum_ref,
          copy_in_slot=SMEM((1,), jnp.uint32) if buffer_type.is_input else None,
          wait_in_slot=SMEM((1,), jnp.uint32) if buffer_type.is_input else None,
          copy_out_slot=SMEM((1,), jnp.uint32) if buffer_type.is_output else None,
          wait_out_slot=SMEM((1,), jnp.uint32) if buffer_type.is_output else None,
          _copy_in_slot_reg=None,
          _wait_in_slot_reg=None,
          _copy_out_slot_reg=None,
          _wait_out_slot_reg=None,
          next_fetch_smem=[SMEM((1,), jnp.int32) for _ in range(
              grid_rank)] if use_lookahead else None,
          next_fetch_sreg=None,
          sem_recvs=(
              None
              if buffer_type is BufferType.OUTPUT
              else SemaphoreType.DMA((buffer_count,))
          ),
          sem_sends=(
              None
              if buffer_type is BufferType.INPUT
              else SemaphoreType.DMA((buffer_count,))
          ),
          swap=SMEM((1,), jnp.bool) if needs_swap_ref else None,
      )

  @classmethod
  def input(cls, spec, dtype, buffer_count=2, **kwargs):
    return cls.create(spec, dtype, BufferType.INPUT, buffer_count, **kwargs)

  @classmethod
  def output(cls, spec, dtype, buffer_count=2, **kwargs):
    return cls.create(spec, dtype, BufferType.OUTPUT, buffer_count, **kwargs)

  @classmethod
  def accumulator(cls, spec, dtype, buffer_count=2, **kwargs):
    return cls.create(spec, dtype, BufferType.ACCUMULATOR, buffer_count,
                      **kwargs)

  @classmethod
  def input_output(cls, spec, dtype, buffer_count=2, **kwargs):
    return cls.create(spec, dtype, BufferType.INPUT_OUTPUT, buffer_count,
                      **kwargs)

  @property
  def block_shape(self):
    return self.spec.block_shape

  @property
  def compute_index(self):
    return self.spec.index_map

  def with_spec(self, spec: pl.BlockSpec) -> BufferedRef:
    """Returns a new BufferedRef with the given block spec."""
    return dataclasses.replace(self, _spec=spec)

  def with_next_fetch(
    self, next_fetch: Sequence[jax.Array] | None = None,
  ):
    return dataclasses.replace(self, next_fetch_sreg=next_fetch)

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
      new_buf = dataclasses.replace(new_buf, _copy_in_slot_reg=copy_in_slot)
    if copy_out_slot is not None:
      new_buf = dataclasses.replace(new_buf, _copy_out_slot_reg=copy_out_slot)
    if wait_in_slot is not None:
      new_buf = dataclasses.replace(new_buf, _wait_in_slot_reg=wait_in_slot)
    if wait_out_slot is not None:
      new_buf = dataclasses.replace(new_buf, _wait_out_slot_reg=wait_out_slot)
    return new_buf

  @property
  def current_ref(self):
    buffer_slice = tuple(
        slice(None)
        for x in self.block_shape
        if not (x is None or isinstance(x, pl.Squeezed))
    )
    assert not (self.window_ref is None or isinstance(self.window_ref, REF))
    if not self.is_buffered:
      return self.window_ref.at[buffer_slice]
    else:
      if self.is_output:
        slot = self.current_copy_out_slot
      else:
        slot = self.current_wait_in_slot
      return self.window_ref.at[(slot, *buffer_slice)]

  @property
  def cumulative_copy_in(self):
    """The cumulative number of copy_ins issued on this buffer."""
    if self._copy_in_slot_reg is not None:
      val = self._copy_in_slot_reg
    else:
      val = self.copy_in_slot[0]
    return val

  @property
  def current_copy_in_slot(self):
    """Index in multiple buffer corresponding to the current slot."""
    return lax.rem(self.cumulative_copy_in, jnp.uint32(self.buffer_count))

  @property
  def cumulative_copy_out(self):
    """The cumulative number of copy_outs issued on this buffer."""
    if self._copy_out_slot_reg is not None:
      val = self._copy_out_slot_reg
    else:
      val = self.copy_out_slot[0]
    return val

  @property
  def current_copy_out_slot(self):
    """Index in multiple buffer corresponding to the current copy slot."""
    return lax.rem(self.cumulative_copy_out, jnp.uint32(self.buffer_count))

  @property
  def cumulative_wait_in(self):
    """The cumulative number of wait_ins issued on this buffer."""
    if self._wait_in_slot_reg is not None:
      val = self._wait_in_slot_reg
    else:
      val = self.wait_in_slot[0]
    return val

  @property
  def current_wait_in_slot(self):
    """Index in multiple buffer corresponding to the current wait slot."""
    return lax.rem(self.cumulative_wait_in, jnp.uint32(self.buffer_count))

  @property
  def cumulative_wait_out(self):
    """The cumulative number of wait_outs issued on this buffer."""
    if self._wait_out_slot_reg is not None:
      val = self._wait_out_slot_reg
    else:
      val = self.wait_out_slot[0]
    return val

  @property
  def current_wait_out_slot(self):
    """Index in multiple buffer corresponding to the current wait slot."""
    return lax.rem(self.cumulative_wait_out, jnp.uint32(self.buffer_count))

  @property
  def next_fetch_indices(self):
    """Returns the next grid indices to fetch from if using lookahead."""
    if not self.use_lookahead:
      raise ValueError("Can only get fetch indices if using lookahead.")
    if self.next_fetch_sreg is not None:
      return self.next_fetch_sreg
    return tuple(smem[0] for smem in self.next_fetch_smem)

  def bind_existing_ref(self, window_ref, indices):
    """For handling VMEM references, the pipeline aliases the existing ref."""
    if not self.is_buffered:
      return dataclasses.replace(
          self, window_ref=window_ref.at[self.compute_slice(indices)]
      )
    return self

  def unbind_refs(self):
    if not self.is_buffered:
      return dataclasses.replace(self, window_ref=None)
    return self

  def compute_slice(self, grid_indices):
    """Compute DMA slice from grid indices."""
    indices = self.compute_index(*grid_indices)
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

  def init_slots(self):
    """Initialize slot indices."""
    if not self.is_buffered: return
    if self.is_input:
      self.copy_in_slot[0] = 0
      self.wait_in_slot[0] = 0
      if self.use_lookahead:
        for i in range(len(self.next_fetch_smem)):
          self.next_fetch_smem[i][0] = 0
    if self.is_output:
      self.copy_out_slot[0] = 0
      self.wait_out_slot[0] = 0
    if self.swap is not None:
      self.swap[0] = False

  def advance_copy_in_slot(self, predicate: bool | jax.Array = True) -> "BufferedRef":
    """Switch to the next copy slot."""
    if not self.is_buffered: return self
    if not self.is_input:
      return self
    current_slot = (self.copy_in_slot[0] if  # type: ignore[index]
                    self._copy_in_slot_reg is None else self._copy_in_slot_reg)
    new_current_slot = lax.select(predicate, current_slot + 1, current_slot)
    if self._copy_in_slot_reg is not None:
      return self.with_slot_index(copy_in_slot=new_current_slot)
    assert isinstance(self.copy_in_slot, jax.Array)
    self.copy_in_slot[0] = new_current_slot
    return self

  def advance_wait_in_slot(self, predicate: bool | jax.Array = True) -> "BufferedRef":
    """Switch to the next wait slot."""
    if not self.is_buffered: return self
    if not self.is_input:
      return self
    current_slot = (self.wait_in_slot[0] if  # type: ignore[index]
                    self._wait_in_slot_reg is None else self._wait_in_slot_reg)
    new_current_slot = lax.select(predicate, current_slot + 1, current_slot)
    if self._wait_in_slot_reg is not None:
      return self.with_slot_index(wait_in_slot=new_current_slot)
    assert isinstance(self.wait_in_slot, jax.Array)
    self.wait_in_slot[0] = new_current_slot
    return self

  def advance_copy_out_slot(self, predicate: bool | jax.Array = True) -> "BufferedRef":
    """Switch to the next copy slot."""
    if not self.is_buffered: return self
    if not self.is_output:
      return self
    current_slot = (self.copy_out_slot[0] if self._copy_out_slot_reg  # type: ignore[index]
                    is None else self._copy_out_slot_reg)
    new_current_slot = lax.select(predicate, current_slot + 1, current_slot)
    if self._copy_out_slot_reg is not None:
      return self.with_slot_index(copy_out_slot=new_current_slot)
    assert isinstance(self.copy_out_slot, jax.Array)
    self.copy_out_slot[0] = new_current_slot
    return self

  def advance_wait_out_slot(self, predicate: bool | jax.Array = True) -> "BufferedRef":
    """Switch to the next wait slot."""
    if not self.is_buffered: return self
    if not self.is_output:
      return self
    current_slot = (self.wait_out_slot[0] if self._wait_out_slot_reg  # type: ignore[index]
                    is None else self._wait_out_slot_reg)
    new_current_slot = lax.select(predicate, current_slot + 1, current_slot)
    if self._wait_out_slot_reg is not None:
      return self.with_slot_index(wait_out_slot=new_current_slot)
    assert isinstance(self.wait_out_slot, jax.Array)
    self.wait_out_slot[0] = new_current_slot
    return self

  def load_slots(self, predicate: bool | jax.Array = True) -> BufferedRef:
    """Load slot information into registers."""
    if not self.is_buffered:
      return self
    def _do_load():
      copy_in = self.copy_in_slot[0] if self.is_input else None
      copy_out = self.copy_out_slot[0] if self.is_output else None
      wait_in = self.wait_in_slot[0] if self.is_input else None
      wait_out = self.wait_out_slot[0] if self.is_output else None
      if self.use_lookahead:
        next_fetch = tuple(self.next_fetch_smem[i][0] for i in range(
            len(self.next_fetch_smem)))
      else:
        next_fetch = None
      return (copy_in, copy_out, wait_in, wait_out, next_fetch)
    def _no_load():
      copy_in = copy_out = wait_in = wait_out = None
      # Need to make sure that we return a non-none value to make sure
      # the pytrees for both branches match.
      _ensure_not_none = lambda x: x if x is not None else jnp.uint32(0)
      if self.is_input:
        copy_in = _ensure_not_none(self._copy_in_slot_reg)
        wait_in = _ensure_not_none(self._wait_in_slot_reg)
      if self.is_output:
        copy_out = _ensure_not_none(self._copy_out_slot_reg)
        wait_out = _ensure_not_none(self._wait_out_slot_reg)
      if self.use_lookahead:
        if self.next_fetch_sreg is None:
          next_fetch = tuple(jnp.int32(0) for _ in range(
              len(self.next_fetch_smem)))
        else:
          next_fetch = self.next_fetch_sreg
      else:
        next_fetch = None
      return (copy_in, copy_out, wait_in, wait_out, next_fetch)
    (copy_in_slot, copy_out_slot, wait_in_slot, wait_out_slot,
     next_fetch) = lax.cond(predicate, _do_load, _no_load)
    bref = self.with_slot_index(
        copy_in_slot=copy_in_slot,
        copy_out_slot=copy_out_slot,
        wait_in_slot=wait_in_slot,
        wait_out_slot=wait_out_slot,
    )
    if bref.next_fetch_smem is not None:
      bref = bref.with_next_fetch(next_fetch=next_fetch)
    return bref

  def save_slots(self, predicate: bool | jax.Array = True):
    """Save slot information from registers."""
    if not self.is_buffered:
      return
    @pl.when(predicate)
    def _():
      if self.is_input:
        assert self._copy_in_slot_reg is not None
        self.copy_in_slot[0] = self._copy_in_slot_reg
        assert self._wait_in_slot_reg is not None
        self.wait_in_slot[0] = self._wait_in_slot_reg
        if self.use_lookahead:
          assert self.next_fetch_sreg is not None
          for i in range(len(self.next_fetch_smem)):
            self.next_fetch_smem[i][0] = self.next_fetch_sreg[i]
      if self.is_output:
        assert self._copy_out_slot_reg is not None
        self.copy_out_slot[0] = self._copy_out_slot_reg
        assert self._wait_out_slot_reg is not None
        self.wait_out_slot[0] = self._wait_out_slot_reg

  def copy_in(self, src_ref, grid_indices):
    """Starts copy of HBM dma slice into the current slot."""
    assert self.is_input
    if not self.is_buffered: return
    assert not (self.window_ref is None or isinstance(self.window_ref, REF))
    assert self.sem_recvs is not None
    if self.swap is not None:
      self.swap[0] = True
    slot = self.current_copy_in_slot
    src_slice = self.get_dma_slice(src_ref.shape, src_ref.dtype, grid_indices)
    dst_slice = tuple(
        pl.ds(0, s.size)
        for s, bd in zip(src_slice, self.block_shape)
        if not (bd is None or isinstance(bd, pl.Squeezed))
    )
    tpu_primitives.make_async_copy(
        src_ref.at[src_slice],
        self.window_ref.at[(slot, *dst_slice)],
        self.sem_recvs.at[slot],
    ).start()

  def copy_out(self, dst_ref, grid_indices):
    """Starts copy of HBM dma slice from the current slot."""
    assert self.is_output
    if not self.is_buffered: return
    assert not (self.window_ref is None or isinstance(self.window_ref, REF))
    assert self.sem_sends is not None
    if self.swap is not None:
      self.swap[0] = True
    slot = self.current_copy_out_slot
    dst_slice = self.get_dma_slice(dst_ref.shape, dst_ref.dtype, grid_indices)
    src_slice = tuple(
        pl.ds(0, s.size)
        for s, bd in zip(dst_slice, self.block_shape)
        if not (bd is None or isinstance(bd, pl.Squeezed))
    )
    tpu_primitives.make_async_copy(
        self.window_ref.at[(slot, *src_slice)],
        dst_ref.at[dst_slice],
        self.sem_sends.at[slot],
    ).start()

  def wait_in(self, src_ref, grid_indices):
    """Waits for input copy to finish."""
    assert self.is_input
    if not self.is_buffered: return
    assert not (self.window_ref is None or isinstance(self.window_ref, REF))
    assert self.sem_recvs is not None
    src_slice = self.get_dma_slice(src_ref.shape, src_ref.dtype, grid_indices)
    dst_slice = tuple(
        pl.ds(0, s.size)
        for s, bd in zip(src_slice, self.block_shape)
        if not (bd is None or isinstance(bd, pl.Squeezed))
    )
    wait_slot = self.current_wait_in_slot
    tpu_primitives.make_async_copy(
        src_ref.at[src_slice],  # nb: doesn't matter
        self.window_ref.at[
            (wait_slot, *dst_slice)
        ],  # only dst shape is important
        self.sem_recvs.at[wait_slot],
    ).wait()

  def wait_out(self, dst_ref, grid_indices):
    """Waits for output copy to finish."""
    assert self.is_output
    if not self.is_buffered: return
    assert not (self.window_ref is None or isinstance(self.window_ref, REF))
    assert self.sem_sends is not None
    wait_slot = self.current_wait_out_slot
    dst_slice = self.get_dma_slice(dst_ref.shape, dst_ref.dtype, grid_indices)
    src_slice = tuple(
        pl.ds(0, s.size)
        for s, bd in zip(dst_slice, self.block_shape)
        if not (bd is None or isinstance(bd, pl.Squeezed))
    )
    tpu_primitives.make_async_copy(
        self.window_ref.at[(wait_slot, *src_slice)],  # nb: doesn't matter
        dst_ref.at[dst_slice],  # only dst shape is important
        self.sem_sends.at[wait_slot],
    ).wait()

  # Accumulator methods
  #
  # Accumulating inline in VMEM saves half the HBM<->VMEM bandwidth cost of
  # doing another full loop around HBM to do a reduction, at the current cost
  # of allocating another VMEM buffer.
  #
  # NB: there's no actual need to have an additional accumulation buffer, if
  # we just rewrote inner kernels to handle the initial-zero-init and output
  # reduction, we don't need to waste VMEM.  Consider removing this magic
  # init and reduce support.

  def set_accumulator(self, init=False):
    """Set accumulator or zero it out to initialize."""
    assert self.is_accumulator
    if self.accum_ref is not None:
      accum_dtype = self.accum_ref.dtype
      def _init():
        self.accum_ref[...] = jnp.zeros_like(self.accum_ref[...])
      def _set():
        self.accum_ref[...] = self.current_ref[...].astype(accum_dtype)
      lax.cond(init, _init, _set)

  def accumulate(self):
    """Add into the current slot."""
    assert self.is_accumulator
    if self.accum_ref is not None:
      assert self.window_ref is not None
      accum_dtype = jnp.float32
      if self.window_ref.dtype == jnp.int32:
        accum_dtype = jnp.int32
      # TODO(levskaya): we could generalize init and reduction functions,
      # could it ever be useful to support more generic monoids?
      self.current_ref[...] = (
          self.current_ref[...].astype(accum_dtype)
          + self.accum_ref[...].astype(accum_dtype)
      ).astype(self.window_ref.dtype)


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
      first_cycle=None,
      last_cycle=None,
      init_accumulators=None,
      trace_scopes=True,
  ):
    """Initializes scheduler.

    Args:
      step: inner step number.
      indices: current grid indices.
      grid: pallas grid for BufferedRefs.
      grid_offsets: offsets for grid indices (used for megacore).
      num_stages: number of stages in the pipeline.
      first_cycle: whether this is the first invocation of the pipeline.
      last_cycle: whether this is the last invocation of the pipeline.
      init_accumulators: do we zero-initialize accumulator state for this
        invocation of the pipeline.
      trace_scopes: whether to use named_scope to trace blocks in the pipeline.
    """
    self.step = step
    self.grid = grid
    self.grid_offsets = grid_offsets
    self.num_stages = num_stages
    self.first_cycle = first_cycle
    self.last_cycle = last_cycle
    self.init_accumulators = init_accumulators
    self.trace_scopes = trace_scopes

    # Total number of linear steps.
    self.num_steps = _grid_size(grid)

    # First and last inner step conditionals.
    self.first_step = step == 0
    self.last_step = step == self.num_steps - 1

    # First and last total step conditionals.
    self.first_step_ever = first_cycle & self.first_step
    self.last_step_ever = last_cycle & self.last_step

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
    return pallas_core.grid_env(
        list(map(pallas_core.GridAxis, self.indices, self.grid)))

  def out_of_fetch(self, buffered_ref):
    """Returns whether there are no more blocks to fetch."""
    # Currently this is based on the iteration, but if we want to support
    # lookahead this will depend on whether the lookahead reached the end.
    if not buffered_ref.is_buffered:
      return False
    return self.step >= (self.num_steps - buffered_ref.buffer_count + 1)

  def has_changed(self, buffered_ref):
    if not buffered_ref.is_buffered:
      return False
    indices = buffered_ref.compute_index(*self.indices)
    prev_indices = buffered_ref.compute_index(*self.prev_indices)
    return _tuples_differ(indices, prev_indices)

  def will_change_current(self, buffered_ref):
    if not buffered_ref.is_buffered:
      return False
    indices = buffered_ref.compute_index(*self.indices)
    next_indices = buffered_ref.compute_index(*self.next_indices)
    return _tuples_differ(indices, next_indices)

  def will_change_fetch(self, buffered_ref):
    if not buffered_ref.is_buffered:
      return False
    if buffered_ref.buffer_count < 2:
      raise NotImplementedError()
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
  # outputs, and in-out accumulators.

  def initialize_step(self, buffered_ref, src_ref, schedule=None, step=0):
    if schedule is None:
      schedule = _default_schedule
    # TODO(justinfu): Should cache this, but it doesn't actually do computation
    # in both default & fixed schedules right now so it doesn't increase
    # the Jaxpr size.
    do_copy = schedule["prologue_copy_in"](self, buffered_ref, src_ref)

    with self._named_scope(f"ep_initialize_{step}"):
      if step == 0:
        @pl.when(self.first_step_ever)
        def _init_slots():
          buffered_ref.init_slots()
        buffered_ref = buffered_ref.load_slots()

      if not buffered_ref.is_input or not buffered_ref.is_buffered:
        return buffered_ref

      if (step + 1) >= buffered_ref.buffer_count:
        return buffered_ref

      if buffered_ref.use_lookahead:
        if step == 0:
          # We always fetch the first block.
          @pl.when(do_copy)
          def _start():
            buffered_ref.copy_in(src_ref,
              self.add_offset(buffered_ref.next_fetch_indices))  # pylint: disable=cell-var-from-loop
          buffered_ref = buffered_ref.advance_copy_in_slot(do_copy)
        else:
          buffered_ref, _ = fetch_with_lookahead(
              buffered_ref,
              src_ref,
              self.grid,
              self.grid_offsets,
              predicate=self.first_step_ever & do_copy,
              max_num_fetches=1,
          )
      else:
        if step == 0:
          predicate = do_copy
          fetch_indices = self.fetch_indices[step]
        else:
          fetch_indices = self.fetch_indices[step]
          prev_grid_indices = self.fetch_indices[step - 1]
          block_indices = buffered_ref.compute_index(*fetch_indices)
          prev_block_indices = buffered_ref.compute_index(*prev_grid_indices)
          block_changed = _tuples_differ(block_indices, prev_block_indices)
          predicate = do_copy & block_changed
        @pl.when(predicate)  # pylint: disable=cell-var-from-loop
        def _start():
          buffered_ref.copy_in(src_ref, fetch_indices)  # pylint: disable=cell-var-from-loop
        buffered_ref = buffered_ref.advance_copy_in_slot(predicate)
    return buffered_ref

  def wait_in(self, buffered_ref, src_ref, schedule=None) -> "BufferedRef":
    if schedule is None:
      schedule = _default_schedule
    pred = schedule["wait_in"](self, buffered_ref, src_ref)

    @self._named_scope("ep_wait_in")
    def _wait():
      if buffered_ref.is_input:
        buffered_ref.wait_in(src_ref, self.indices)
      if buffered_ref.is_accumulator:
        # In most cases we won't be waiting when init_accumulators is True,
        # so this is usually just setting what we just copied.
        buffered_ref.set_accumulator(self.init_accumulators)

    @self._named_scope("ep_set_accum")
    def _no_wait():
      if buffered_ref.is_accumulator:

        @pl.when(self.first_step | self.has_changed(buffered_ref))
        def _set_accumulator():
          # In most cases we will skip waiting when init_accumulators is True,
          # so this is usually just setting the accumulator to 0.
          buffered_ref.set_accumulator(self.init_accumulators)
    lax.cond(pred, _wait, _no_wait)
    return buffered_ref

  def copy_in(self, buffered_ref, src_ref, schedule=None) -> "BufferedRef":
    if schedule is None:
      schedule = _default_schedule
    pred = schedule['copy_in'](self, buffered_ref, src_ref)
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

  # --> Call prefetch here to grab the first inputs of next cycle.

  # convenience method for prefetch callbacks.
  def prefetch(self, buffered_ref, src_ref, schedule=None):
    if schedule is None:
      schedule = _default_schedule
    pred = schedule['prefetch'](self, buffered_ref, src_ref)

    if not buffered_ref.is_input or not buffered_ref.is_buffered:
      return

    if buffered_ref.use_lookahead:
      buffered_ref = buffered_ref.with_next_fetch(
          jax.tree.map(jnp.zeros_like, buffered_ref.next_fetch_sreg))
      @pl.when(pred)
      def _start():
        buffered_ref.copy_in(
            src_ref, self.add_offset(buffered_ref.next_fetch_sreg))  # pylint: disable=cell-var-from-loop
      buffered_ref = buffered_ref.advance_copy_in_slot(pred)

      buffered_ref, final_copy_in_slot = fetch_with_lookahead(
          buffered_ref,
          src_ref,
          self.grid,
          self.grid_offsets,
          predicate=pred,
          update_slots=False,
      )
      @pl.when(pred)
      def _():
        bref = buffered_ref.with_slot_index(copy_in_slot=final_copy_in_slot)
        bref.save_slots()
    else:
      pred = pred & self.last_step
      grid_indices = self.indices
      for i in range(buffered_ref.buffer_count - 1):
        next_grid_indices = self.fetch_indices[i+1]
        block_indices = buffered_ref.compute_index(*grid_indices)
        next_block_indices = buffered_ref.compute_index(*next_grid_indices)
        if i == 0:
          # If the prefetch predicate triggers, we already know that the
          # first block needs to be copied.
          should_prefetch = True
        else:
          should_prefetch = _tuples_differ(block_indices, next_block_indices)

        @pl.when(pred & should_prefetch)
        def _():
          buffered_ref.copy_in(src_ref, next_grid_indices)  # pylint: disable=cell-var-from-loop
        buffered_ref = buffered_ref.advance_copy_in_slot(pred & should_prefetch)
        grid_indices = next_grid_indices
      buffered_ref.save_slots()
    return

  def wait_out(self, buffered_ref, dst_ref, schedule=None) -> "BufferedRef":
    if schedule is None:
      schedule = _default_schedule
    pred = schedule['wait_out'](self, buffered_ref, dst_ref)

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

  # --> Call "postyeet" here, after last output copy is finished from previous
  #     cycle

  def copy_out(self, buffered_ref, dst_ref, schedule=None) -> "BufferedRef":
    if schedule is None:
      schedule = _default_schedule
    pred = schedule['copy_out'](self, buffered_ref, dst_ref)

    @self._named_scope("ep_copy_out")
    def _copy_out_and_accumulate():
      if buffered_ref.is_accumulator:
        buffered_ref.accumulate()
      if buffered_ref.is_output:
        buffered_ref.copy_out(dst_ref, self.indices)

    @self._named_scope("ep_accum")
    def _just_accumulate():
      if buffered_ref.is_accumulator:
        # We accumulate on the last step because we will set the accumulator
        # on the next first step. We can optimize this away if it becomes
        # a problem, but it is probably not worth the complexity to support
        # chains of different pipelines that want to reuse the accumulator with
        # slightly different schedules.
        @pl.when(self.last_step)
        def _accumulate():
          buffered_ref.accumulate()
    lax.cond(pred, _copy_out_and_accumulate, _just_accumulate)
    return buffered_ref.advance_copy_out_slot(pred & buffered_ref.is_output)

  def finalize(self, buffered_ref, dst_ref, schedule=None):
    if schedule is None:
      schedule = _default_schedule
    pred = schedule['epilogue_wait_out'](self, buffered_ref, dst_ref)

    @pl.when(pred)
    @self._named_scope("ep_finalize")
    def _end():
      if buffered_ref.is_output:
        buffered_ref.wait_out(dst_ref, self.indices)

    buffered_ref.save_slots()

  def advance_slots(self, buffered_ref, schedule=None):
    if schedule is None:
      schedule = _default_schedule

    if buffered_ref.is_input:
      pred = schedule['advance_wait_in'](self, buffered_ref, schedule)
      buffered_ref = buffered_ref.advance_wait_in_slot(pred)
    # Currently we advance copy_in and output slots after their respective
    # operation.
    return buffered_ref

  # END SCHEDULE --------------------------------------------------------------


# Scheduling overrides.

# When trying to fuse across pipelines that use accumulator arguments, we
# sometimes need to mess with the default scheduling above to avoid data-races
# or to maximize performance.  A schedule is simply a set of functions that
# calculate predicates for whether or not the pipeline input and output
# BufferedRefs should do copies and waits.


# Copy of the default pipeline schedule.  The default schedule tacitly assumes
# that the source and target HBM Refs change with each cycle.
_default_schedule = dict(
    prologue_copy_in=lambda s, bref, _: s.first_step_ever,
    # We assume that the source ref changed for prefetch.
    wait_in=lambda s, bref, _: s.has_changed(bref) | s.first_step,
    advance_wait_in=lambda s, bref, _: (
        s.will_change_current(bref) | s.last_step),
    copy_in=lambda s, bref, _: s.will_change_fetch(bref) & ~s.out_of_fetch(
        bref),
    # We assume that the source ref changed. E.g. because of a CM DMA.
    prefetch=lambda s, bref, _: (
        (s.will_change_fetch(bref) | s.last_step) & ~s.last_step_ever
    ),
    # We assume that the target ref changed. E.g. because of a CM DMA.
    wait_out=lambda s, bref, _: (s.has_changed(bref) | s.first_step) & ~s.first_step_ever,
    # We assume that the target ref is changing. E.g. because of a CM DMA.
    copy_out=lambda s, bref, _: s.will_change_current(bref) | s.last_step,
    epilogue_wait_out=lambda s, bref, _: s.last_step_ever,
)


# Alternative schedule needed for accumulators reading and writing to a fixed
# HBM reference to avoid HBM data races for trivially small grids: only
# read/write when tiles change or at the very beginning or end of a fused
# pipeline schedule.
_fixed_schedule = dict(
    prologue_copy_in=lambda s, bref, _: s.first_step_ever,
    # We don't assume that the source ref changed for prefetch.
    wait_in=lambda s, bref, _: s.has_changed(bref) | s.first_step_ever,
    advance_wait_in=lambda s, bref, _: s.will_change_current(bref),
    copy_in=lambda s, bref, _: s.will_change_fetch(bref) & ~s.out_of_fetch(
        bref),
    # We don't assume that the source ref changed.
    prefetch=lambda s, bref, _: s.will_change_fetch(bref) & ~s.last_step_ever,
    # We don't assume that the target ref changed.
    wait_out=lambda s, bref, _: (s.has_changed(bref) & ~s.first_step_ever),
    # We don't assume that the target ref is changing.
    copy_out=lambda s, bref, _: s.will_change_current(bref) | s.last_step_ever,
    epilogue_wait_out=lambda s, bref, _: s.last_step_ever,
)


def skip_input_copies_when_init_accumulators(schedule) -> Any:
  """Skip input copies in schedule when init_accumulators is True."""
  new_schedule = {**schedule}
  for k in ["prologue_copy_in", "wait_in", "copy_in"]:

    def new_pred(original_pred_fn, *a):
      pred = original_pred_fn(*a)
      if a[1].is_accumulator or a[1].is_input_output:
        pred &= jnp.logical_not(a[0].init_accumulators)
      return pred

    new_schedule[k] = functools.partial(
        new_pred,
        schedule[k],
    )
  return new_schedule


_default_schedule = skip_input_copies_when_init_accumulators(_default_schedule)
_fixed_schedule = skip_input_copies_when_init_accumulators(_fixed_schedule)

def get_pipeline_schedule(schedule) -> Any:
  """Retrieve a named pipeline schedule or pass through fully specified one."""
  predefined_schedules = {
      'default': _default_schedule,
      'fixed': _fixed_schedule
  }
  if isinstance(schedule, str):
    return predefined_schedules[schedule].copy()
  return schedule


# Main pipeline methods


def make_pipeline_allocations(
    *refs,
    in_specs=None,
    out_specs=None,
    should_accumulate_out=False,
    needs_swap_ref=True,
    grid=None,
):
  """Create BufferedRefs for the pipeline.

  This function creates buffered refs for an inner pipeline that can be
  created at the top-level of a pallas call such that they may be reused across
  multiple invocations of the inner pipeline.

  Args:
    in_specs: input pallas block specs
    out_specs: output pallas block specs
    should_accumulate_out: booleans to indicate which outputs should be treated
      as accumulators.
    needs_swap_ref: whether a swap slots tracker needs to be allocated.
    grid: grid to use for the pipeline.

  Returns:
    A list of BufferedRefs, one corresponding to each ref specified in the
    in_specs and out_specs.
  """
  # TODO(levskaya): generalize argument tree handling here and in emit_pipeline.
  num_in_specs = len(in_specs)
  if not isinstance(in_specs, (list, tuple)):
    in_specs = (in_specs,)
  if not isinstance(out_specs, (list, tuple)):
    out_specs = (out_specs,)
  if isinstance(in_specs, list):
    in_specs = tuple(in_specs)
  if isinstance(out_specs, list):
    out_specs = tuple(out_specs)
  in_refs = refs[:num_in_specs]
  out_refs = refs[num_in_specs:]
  def make_input_bref(in_spec, in_ref):
    buffer_count = 2
    use_lookahead = False
    if in_spec.pipeline_mode is not None:
      buffer_count = in_spec.pipeline_mode.buffer_count
      use_lookahead = in_spec.pipeline_mode.use_lookahead
    if use_lookahead and grid is None:
      raise ValueError("Grid must be specified when using lookahead.")
    return BufferedRef.input(in_spec, in_ref.dtype, buffer_count,
                             needs_swap_ref=needs_swap_ref,
                             grid_rank=len(grid),
                             use_lookahead=use_lookahead,
                             source_memory_space=in_ref.memory_space)
  in_brefs = jax.tree.map(make_input_bref, in_specs, in_refs)
  def make_output_bref(out_spec, out_ref, accumulate):
    buffer_count = 2
    if out_spec.pipeline_mode is not None:
      buffer_count = out_spec.pipeline_mode.buffer_count
      if out_spec.pipeline_mode.use_lookahead:
        raise ValueError("Output buffering does not support lookahead.")

    if accumulate:
      return BufferedRef.accumulator(out_spec, out_ref.dtype, buffer_count,
                                     needs_swap_ref=needs_swap_ref,
                                     source_memory_space=out_ref.memory_space)
    return BufferedRef.output(out_spec, out_ref.dtype, buffer_count,
                              needs_swap_ref=needs_swap_ref,
                              source_memory_space=out_ref.memory_space)
  out_brefs = jax.tree.map(
      make_output_bref, out_specs, out_refs, should_accumulate_out)
  return (*in_brefs, *out_brefs)


def _partition_grid(
    grid: tuple[int | jax.Array, ...],
    core_axis: int | str | None,
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

  parallel_dimensions = {i for i, d in enumerate(dimension_semantics)
                         if d == PARALLEL}
  # If there are no parallel dimensions, we can't partition the grid
  if not parallel_dimensions:
    # TODO(sharadmv): enable running kernel on just one core
    raise NotImplementedError(
        "Cannot partition over cores without parallel grid dimensions:"
        f" {dimension_semantics=}"
    )
  if all(not isinstance(grid[i], int) for i in parallel_dimensions):
    raise NotImplementedError(
        f"Cannot partition cores over only dynamic grid dimensions: {grid=}"
    )
  # Try to find a divisible dimension to partition the grid on
  divisible_dimensions = {
      i for i in parallel_dimensions
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
        (0,) * len(grid), first_divisible_dimension, partitioned_dim_offset
    )
  else:
    # No divisible dimensions, so we can't evenly partition the grid. Let's pick
    # the largest dimension and try to divide it as evenly as possible.
    # TODO(sharadmv): take the product of many nondivisible dimensions to
    # potentially divide it more evenly
    largest_parallel_dimension = max(grid[i] for i in parallel_dimensions
                                     if isinstance(grid[i], int))  # type: ignore
    partition_dimension, *_ = (
        i
        for i, d in enumerate(grid)
        if isinstance(d, int) and d == largest_parallel_dimension
    )
    base_num_iters, rem = divmod(grid[partition_dimension], num_cores)
    assert rem > 0, rem
    # We have some remainder iterations that we need to assign somewhere. We
    # know that rem < num_cores, so we can assign one extra iteration to each
    # core except for the last (num_cores - rem).
    num_iters = jnp.where(core_id < rem, base_num_iters + 1,
                          base_num_iters)
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
        (0,) * len(grid), partition_dimension, grid_offset
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
  hbm_slice = bref.get_dma_slice(hbm_ref.shape, hbm_ref.dtype, indices)
  bref_slice = tuple(
      pl.ds(0, s.size)
      for s, bd in zip(hbm_slice, bref.block_shape)
      if not (bd is None or isinstance(bd, pl.Squeezed))
  )
  if copy_in:
    tpu_helpers.sync_copy(hbm_ref.at[hbm_slice],
                          bref.current_ref.at[bref_slice])  # type: ignore[union-attr]
  else:
    tpu_helpers.sync_copy(bref.current_ref.at[bref_slice],  # type: ignore[union-attr]
                          hbm_ref.at[hbm_slice])


def emit_pipeline(
    body,
    *,
    grid: tuple[int | jax.Array, ...],
    in_specs=None,
    out_specs=None,
    should_accumulate_out: bool = False,
    core_axis: int | None = None,
    core_axis_name: str | None = None,
    dimension_semantics: tuple[GridDimensionSemantics, ...] | None = None,
    trace_scopes: bool = True,
    no_pipelining: bool = False,
):
  """Creates a function to emit a manual pallas pipeline.

  This has the same semantics as pallas_call but is meant to be called inside
  pallas_call for nesting grids. This is useful when you need to have separate
  windowing strategies for communication and computation.

  The new argument `should_accumulate_out` can be used to specify which outputs
  we should accumulate into automatically within and across pipeline
  invocations.

  Args:
    body: pallas kernel to set up pipeline for.
    grid: a pallas grid definition.
    in_specs: input pallas block specs
    out_specs: output pallas block specs
    should_accumulate_out: booleans to indicate which outputs should be treated
      as accumulators.
    core_axis: optional int, indicates whether or not to partition the grid
      along the core axis.
    core_axis_name: optional str, indicates whether or not to partition the grid
      along the core axis.
    dimension_semantics: optional tuple of GridDimensionSemantics (e.g. PARALLEL
      or ARBITRARY).
    trace_scopes: optional bool, indicates whether to annotate each region in
      the pipeline using named_scope.
    no_pipelining: If True, turns off pipelining and all copies will be made
      synchronous. This is useful for debugging multiple-buffering related bugs.
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
  if not isinstance(in_specs, (list, tuple)):
    in_specs = (in_specs,)
  if not isinstance(out_specs, (list, tuple)):
    out_specs = (out_specs,)
  if isinstance(in_specs, list):
    in_specs = tuple(in_specs)
  if isinstance(out_specs, list):
    out_specs = tuple(out_specs)
  should_accumulate_out = _broadcast_pytree_to(should_accumulate_out, out_specs)
  get_buffer_count = lambda spec: (spec.pipeline_mode.buffer_count if
    (spec is not None and spec.pipeline_mode is not None) else 2)
  flattened_specs = jax.tree.leaves((in_specs, out_specs))
  max_buffer_count = max((2, *map(get_buffer_count, flattened_specs)))

  def pipeline(
    *refs: Any,
    scratches=None,
    allocations=None,
    first_cycle: CondVal = True,
    last_cycle: CondVal = True,
    init_accumulators: CondVal = False,
    prefetch=None,
    postyeet=None,
    schedule=None,
    body_prologue=None,
  ):
    """
    Run the pipeline.

    Args:
      *ref_args: a list of pallas refs (or more generally a list of pytrees of
        pallas refs)
      scratches: scratch buffers for the inner kernel
      allocations: a list of BufferedRefs, one corresponding to each ref
      first_cycle: boolean indicating if this is the first invocation of the
        inner pipeline cycle.
      last_cycle: boolean indicating if this is the last invocation of the
        inner pipeline cycle.
      init_accumulators: whether to zero-init accumulators during this cycle.
      prefetch: callback called as fn(*brefs, scheduler) that is used to fetch
        the next cycle invocations first inputs.  Called during the inputs phase
        in the final inner step.
      postyeet: callback called as fn(*brefs, scheduler) that is used to finish
        any writes or transfers from the last output of the previous cycle.
        Called during the outputs phase in the first inner step.
      schedule: manually specified pipeline schedules for brefs, None indicates
        default schedule.
      body_prologue: For running code within the grid environment before the
        body is run. Useful for updating manual refs.
    """
    if scratches is None:
      scratches = ()
    if allocations is None:
      # run with inline scoped allocations

      # Prefetch and postyeet are arbitrary functions that can copy
      # into or out of any of the BufferedRefs. Thus, we need a ref
      # for the scheduler to mark when the prefetch or postyeet
      # functions perform a copy and the slots need to be
      # swapped. Without prefetch and postyeet, the swapping logic can
      # be performed without the need for state.
      needs_swap_ref = prefetch is not None or postyeet is not None
      return primitives.run_scoped(
          lambda allocations: pipeline(
              *refs,
              scratches=scratches,
              allocations=allocations,
              first_cycle=first_cycle,
              last_cycle=last_cycle,
              init_accumulators=init_accumulators,
              prefetch=prefetch,
              postyeet=postyeet,
              schedule=schedule,
          ),
          make_pipeline_allocations(
              *refs,
              in_specs=in_specs,
              out_specs=out_specs,
              should_accumulate_out=should_accumulate_out,
              needs_swap_ref=needs_swap_ref,
              grid=grid,
          ),
      )
    if isinstance(allocations, list):
      allocations = tuple(allocations)
    # Normalize custom schedule arguments.
    if schedule is None:
      schedule = map_brefs(lambda x: None, allocations)
    if not isinstance(schedule, (list, tuple)):
      schedule = map_brefs(lambda x: schedule, allocations)
    if isinstance(schedule, list):
      schedule = tuple(schedule)
    schedule = map_brefs(
        lambda _, x: get_pipeline_schedule(x), allocations, schedule)

    def make_scheduler(step, indices):
      return Scheduler(
          step,
          indices,
          grid,
          grid_offsets=grid_offsets,
          num_stages=max_buffer_count,
          first_cycle=first_cycle,
          last_cycle=last_cycle,
          init_accumulators=init_accumulators,
          trace_scopes=trace_scopes,
      )

    def loop_body(step, carry):
      unaliased_brefs, indices = carry
      scheduler = make_scheduler(step, indices)
      with scheduler.grid_env():

        # prepare any local VMEM aliases
        brefs = map_brefs(scheduler.alias_local_refs, unaliased_brefs, refs)
        # loop input handling phase
        brefs = map_brefs(scheduler.copy_in, brefs, refs, schedule)
        brefs = map_brefs(scheduler.wait_in, brefs, refs, schedule)

        # prefetch inputs for the *next* invocation of this pipeline
        with scheduler._named_scope("ep_prefetch"):
          if prefetch is not None:
            do_prefetch = step == num_steps - 1
            map_brefs(lambda x: x.save_slots(do_prefetch), brefs)
            lax.cond(do_prefetch,
                    lambda: prefetch(*brefs, scheduler),
                    lambda: None)
            brefs = map_brefs(lambda x: x.load_slots(do_prefetch), brefs)

        # run the kernel!
        if body_prologue is not None:
          body_prologue()
        current_refs = map_brefs(lambda x: x.current_ref, brefs)
        with scheduler._named_scope("ep_run_kernel"):
          body(*current_refs, *scratches)

        # loop output handling phase
        brefs = map_brefs(scheduler.copy_out, brefs, refs, schedule)
        brefs = map_brefs(scheduler.wait_out, brefs, refs, schedule)
        # handle writes for the *last* invocation of this pipeline's outputs
        with scheduler._named_scope("ep_postyeet"):
          if postyeet is not None:
            do_postyeet = step == 0
            map_brefs(lambda x: x.save_slots(do_postyeet), brefs)
            lax.cond(do_postyeet,
                    lambda: postyeet(*brefs, scheduler),
                    lambda: None)
            brefs = map_brefs(lambda x: x.load_slots(do_postyeet), brefs)

        brefs = map_brefs(scheduler.advance_slots, brefs, schedule)
        # Unbind window_refs for VMEM-backed buffers. Without this
        # we will be returning TransformedRefs which are not valid
        # JAX types.
        brefs = map_brefs(scheduler.unalias_local_refs, brefs)
      return brefs, _next_index(indices, grid)

    if no_pipelining:
      # Debugging mode where all copies are synchronous.
      initial_indices = (0,) * len(grid)
      scheduler = make_scheduler(0, initial_indices)
      brefs = map_brefs(scheduler.alias_local_refs, allocations, refs)
      map_brefs(lambda bref: bref.init_slots(), brefs)
      if postyeet is not None or prefetch is not None:
        raise NotImplementedError("Prefetch/Postyeet not supported")
      if any(bref.is_accumulator for bref in brefs):
        raise NotImplementedError("Accumulators not supported")
      @functools.partial(jax.lax.fori_loop, 0, num_steps,
                         init_val=(brefs, initial_indices))
      def _loop_body(step, carry):
        brefs, indices = carry
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
        brefs = allocations
        with scheduler.grid_env():
          # We issue num_stages-1 prefetch copies per buffer.
          # We iterate over steps in the outer loop because we want to
          # queue all iteration 0 prefetches before iteration 1, and so on.
          for step in range(scheduler.num_stages - 1):
            brefs = map_brefs(functools.partial(
                scheduler.initialize_step, step=step),
                brefs, refs, schedule)

        # pipeline loop
        brefs, next_indices = lax.fori_loop(
            0, num_steps, loop_body, (brefs, initial_indices)
        )

        # pipeline epilogue
        final_indices = _prev_index(next_indices, grid)
        scheduler = make_scheduler(num_steps - 1, final_indices)
        with scheduler.grid_env():
          map_brefs(scheduler.finalize, brefs, refs, schedule)

  return pipeline


def emit_pipeline_with_allocations(
    body,
    *,
    grid,
    in_specs=None,
    out_specs=None,
    should_accumulate_out=False,
):
  """Creates pallas pipeline and top-level allocation preparation functions.

  Args:
    body: pallas kernel to set up pipeline for.
    grid: a pallas grid definition.
    in_specs: input pallas block specs
    out_specs: output pallas block specs
    should_accumulate_out: booleans to indicate which outputs should be treated
      as accumulators.

  Returns:
    (emit_pipeline, make_allocations) function pair, where:
    emit_pipeline is the pallas pipeline function.
    make_allocations is a function to create buffered refs for the inner
      pipeline that can be created at the top-level of a pallas call to be
      reused across multiple invocations of the inner pipeline.
  """
  make_allocations = functools.partial(make_pipeline_allocations,
                    in_specs=in_specs,
                    out_specs=out_specs,
                    should_accumulate_out=should_accumulate_out,
                    grid=grid)
  pipeline = emit_pipeline(
      body,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      should_accumulate_out=should_accumulate_out)

  return pipeline, make_allocations
