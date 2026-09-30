# Copyright 2026 The JAX Authors.
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
import dataclasses
import enum
import functools
import math

import jax
import numpy as np
from jax import numpy as jnp


class MaskType(enum.Enum):
  ZEROS = 0
  PARTIAL = 1
  ONES = 2


@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=["data_next",
                                "mask_next",
                                "block_mask",
                                "num_nonzero_blocks",
                                "partial_mask_blocks"],
                   meta_fields=["data_tiling", "is_dkv"])
@dataclasses.dataclass(frozen=True)
class MaskInfo:
  """Contains runtime masking information for the attention kernel.

  The ``data_next`` and ``num_nonzero_blocks`` fields are tiled according
  to the number of compute warpgroups used by the kernel. This is because when using
  warp specialization, the data blocks are shared between all of the compute
  warpgroups, so each warpgroup will receive the same data block and then decide
  whether to keep or skip the block based on ``block_mask``.
  However, the mask data is not tiled since the mask typically varies
  between q_blocks, and therefore cannot be shared between the compute warpgroups.

  The ``data_next``, ``mask_next``, and ``block_mask`` tensors are compressed such that
  all non-zero blocks are sorted to the front (for fwd/dq kernels) or to the end (for dkv).
  The ``num_nonzero_blocks`` tensor indicates how many non-zero blocks there are, so it
  is sufficient to loop from 0 to ``num_nonzero_blocks`` to visit all non-zero blocks
  in the mask.

  Attributes:
    data_next: An integer[batch_size, heads=1, num_q_blocks // q_tiling, num_kv_blocks // kv_tiling]
      array where each entry contains the next ``kv`` block index to
      prefetch. This tensor is either tiled along the q or kv dimension, depending
      on which kernel this mask is for. For forward/dq kernels, the tiling is done
      along the ``num_q_blocks`` dimension. For the dkv kernel, the tiling is done
      along the ``num_kv_blocks`` dimension.
    mask_next: An integer[batch_size, heads=1, num_q_blocks, num_kv_blocks]
      array where each entry contains the next mask block index in
      `partial_mask_blocks` to prefetch.
    block_mask: An integer[batch_size, heads=1, num_q_blocks, num_kv_blocks]
      array whose entries can be 0, 1 or 2. An entry of 0 indicates that
      the corresponding block in the full mask was all zeros. An entry of 1
      indicates that the corresponding block in the full mask contained both
      zeros and ones. An entry of 2 indicates the corresponding block was
      entirely ones.
    num_nonzero_blocks: An integer[batch_size, heads=1, num_q_or_kv_blocks // q_or_kv_tiling]
      array containing the number of non-zero blocks along each q-block dimension (or kv dimension
      if computing the dkv mask).
    partial_mask_blocks: A bool[batch_size, 1, num_partial_blocks, block_q, block_kv]
      array that contains the blocks of the original mask that contained both
      zeros and ones. The entries in `mask_next` point to indices in the first
      axis of this array.
    data_tiling: The tiling factor (q_tiling or kv_tiling)
    is_dkv: Whether this mask is for the dkv kernel. If False, this mask is constructed
      for the forward/dq kernels.
  """

  data_next: jax.Array
  mask_next: jax.Array
  block_mask: jax.Array
  num_nonzero_blocks: jax.Array
  partial_mask_blocks: jax.Array
  data_tiling: int
  is_dkv: bool

  @property
  def q_block_size(self) -> int:
    return self.partial_mask_blocks.shape[3]

  @property
  def kv_block_size(self) -> int:
    return self.partial_mask_blocks.shape[4]


@jax.jit(static_argnames=("axis", "placeholder", "flip"))
def compress_array(x: jax.Array, axis=-1, placeholder=-1, flip: bool = False) -> tuple[jax.Array, jax.Array]:
  """Compress array by shifting non-placeholder entries to the left along the specified axis.

  Placeholder entries are pushed to the right while preserving relative order of non-placeholder elements.

  Args:
      x: Input JAX array
      axis: Axis along which to compress (default: -1, last axis)
      placeholder: Value to treat as placeholder (default: -1)
      flip: If True, will compress to the right instead of to the left.

  Returns:
      A tuple containing (compressed_array, sort_indices).
  """
  # Mask of valid (non-placeholder) elements
  mask = x != placeholder

  # Size along compression axis
  n = x.shape[axis]

  # Create indices along the compression axis, broadcast to input shape
  shape = [1] * x.ndim
  shape[axis] = n
  indices = jnp.arange(n).reshape(shape)
  indices = jnp.broadcast_to(indices, x.shape)

  # Sort key: valid elements keep original index, invalid get large values
  # This ensures valid elements sort first while preserving relative order
  if flip:
    key = jnp.where(mask, n + indices, indices)
  else:
    key = jnp.where(mask, indices, n + indices)

  # Get sorted indices and apply
  sorted_indices = jnp.argsort(key, axis=axis)
  return jnp.take_along_axis(x, sorted_indices, axis=axis), sorted_indices


def tiled_max(x: jax.Array, tile_size: int, axis: int = -2) -> jax.Array:
  """Computes a max over tiles along an axis.

  For example, for an array [0, 1, 2, 3, 4, 5]:
    with tile_size=2, the result would be [1, 3, 5]
    with tile_size=3, the result would be [2, 5]
    with tile_size=6, the result would be [5] (equivalent to a non-tiled max)
  """
  if tile_size == 1:
    return x
  if axis < 0:
    axis += x.ndim
  axis_size = x.shape[axis]
  new_axis_size = axis_size // tile_size

  # Tile to group rows: (X, Y) -> (X // num_compute_wgs, num_compute_wgs, Y)
  new_shape = x.shape[:axis] + (new_axis_size, tile_size) + x.shape[axis + 1:]
  x_reshaped = x.reshape(new_shape)

  # Take max along the grouping axis
  return jnp.max(x_reshaped, axis=axis + 1)


def process_dynamic_mask(
    mask: jax.Array,
    block_shape: tuple[int, int],
    is_dkv: bool,
    *,
    data_tiling: int = 1,
    downcast_block_mask_data: bool = True,
) -> MaskInfo:
  """Processes a dense attention mask into a sparse ``MaskInfo`` representation.

  Since the mask is dynamic, we can't know the exact number of partial mask
  blocks at trace time. Therefore, the entire mask is materialized in
  ``partial_mask_blocks``.

  Args:
    mask: A [batch_size, heads=1, q_seq_len, kv_seq_len] jax.Array representing the dense
      mask to process.
    block_shape: A tuple of (q_block_size, kv_block_size).
    is_dkv: True if we are processing the dKV mask
    data_tiling: The tiling factor to apply to the data blocks. This should be set to
      the number of compute warpgroups used by the kernel.
    downcast_block_mask_data: If True, downcast the scalar-memory data of MaskInfo to
      a data type smaller than np.int32 (if possible).

  Returns:
    `MaskInfo`, a sparse representation of the dense mask.

  Raises:
    ValueError: if the input mask is invalid or the block sizes are not
    compatible with the mask sizes.
  """
  if len(mask.shape) != 4:
    raise ValueError(f"Expected a 4-dim mask, instead got: {mask.shape}.")

  if mask.dtype != jnp.bool:
    raise ValueError(f"Expected a bool mask, instead got: {mask.dtype}.")

  batch_size, head_count, q_seq_len, kv_seq_len = mask.shape
  if head_count != 1:
    raise ValueError(
        f"Expected head dimension to be 1, instead got: {head_count}.")
  q_block_size, kv_block_size = block_shape
  q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
  kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)

  if q_mod != 0:
    raise ValueError(f"{q_block_size=} should divide {q_seq_len=}.")
  if kv_mod != 0:
    raise ValueError(f"{kv_block_size=} should divide {kv_seq_len=}.")

  block_mask_shape = (
      batch_size,
      head_count,
      q_blocks_count,
      kv_blocks_count,
  )

  # Tile the last 2 dimensions of the mask into 2D tiles of size `block_shape`.
  partial_mask_blocks = (
      mask.reshape(
          batch_size,
          head_count,
          q_blocks_count,
          q_block_size,
          kv_blocks_count,
          kv_block_size,
      )
      .swapaxes(-2, -3)
      .astype(np.bool_)
  )

  # The block mask is 2 for all blocks with all entries set to True and 1 for
  # blocks with a mix of True and False entries.
  is_full_mask = jnp.all(partial_mask_blocks, axis=(-1, -2))
  is_empty_mask = jnp.logical_not(jnp.any(partial_mask_blocks, axis=(-1, -2)))

  block_mask = jnp.ones(block_mask_shape, dtype=np.int32) * \
      MaskType.PARTIAL.value
  block_mask = jnp.where(is_full_mask, MaskType.ONES.value, block_mask)
  block_mask = jnp.where(is_empty_mask, MaskType.ZEROS.value, block_mask)

  mask_info_slice_shape = (batch_size, 1, q_blocks_count, kv_blocks_count)
  mask_info_slice_shape_per_batch = (1, *mask_info_slice_shape[1:])
  mask_next_slice = jnp.arange(math.prod(mask_info_slice_shape_per_batch), dtype=np.int32).reshape(
      mask_info_slice_shape_per_batch
  )
  mask_next_slice = jnp.tile(mask_next_slice, (batch_size, 1, 1, 1))
  mask_next_slice = jnp.where(block_mask == 1, mask_next_slice, -1)

  # data_next stores the index of the next non-empty data block in the sequence.
  # The indices of empty blocks are set to 0 to avoid copying extra data when
  # pipeling.
  if is_dkv:
    data_next = jnp.arange(q_blocks_count, dtype=np.int32)[None, None, :, None]
  else:
    data_next = jnp.arange(kv_blocks_count, dtype=np.int32)[
        None, None, None, :]
  data_next = jnp.broadcast_to(data_next, mask_info_slice_shape)
  data_next = jnp.where(block_mask == 0, -1, data_next)

  # Compress and tile mask data.
  # The mask_info is structured such that data blocks are shared between compute warpgroups.
  # This is due to the structure of a warp-specialized kernel where there is a single memory
  # warpgroup loading data that can be shared between multiple compute warpgroups, reducing
  # the overall amount of data copied from HBM.
  # We do not share the mask between compute warpgroups since the masking pattern typically
  # differs between different Q-blocks and therefore cannot be shared.
  #
  # Consider the simple case with a single compute warpgroup. Before compression, we could have:
  #   block_mask = [ 0  0  2  2  1  0  0]
  #   data_next  = [-1 -1  2  3  4 -1 -1]
  # Compression will move all non-zero blocks to the beginning of the array, resulting in:
  #   block_mask = [ 2  2  1  0  0  0  0]
  #   data_next  = [ 2  3  4 -1 -1 -1 -1]
  # Additionally, we will compute `num_nonzero_blocks=3` to indicate to the kernel that it only
  # needs to loop from 0 to 3 to visit all non-masked blocks. This operation prevents the kernel
  # from wasting time while iterating through empty blocks.
  #
  # Now consider the case where we have 2 compute warpgroups and one memory warpgroup. We
  # have 2 `block_mask` rows (handled by the compute warpgroup) per `data_next` row (handled by
  # the memory warpgroup).
  #   block_mask = [ 0  0  2  2  1  0  0]
  #                [ 0  0  0  2  2  1  0]
  #   data_next  = [-1 -1  2  3  4  5 -1]
  # In this case, we cannot compress each row of block_mask individually and can only ignore
  # a block if both warpgroups have that block masked. Therefore, the result of compression should
  # yield:
  #   block_mask = [ 2  2  1  0  0  0  0]
  #                [ 0  2  2  1  0  0  0]
  #   data_next  = [ 2  3  4  5 -1 -1 -1]
  # And we have `num_nonzero_blocks=4`. Now, it is up to the compute warpgroup 0 to zero-out its
  # mask on iteration 3 and compute warpgroup 1 to zero-out its mask on iteration 0.
  # For the dkv mask we apply the same logic but on the q-dimension instead of the kv-dimension.

  if is_dkv:
    tile_axis = -1  # kv dim
    compress_axis = -2  # q dim
    # For dkv we compress entires to the end of the array instead of the beginning.
    # This matches the behavior of the TPU splash attention masking logic.
    flip_compress = True
  else:
    tile_axis = -2  # q dim
    compress_axis = -1  # kv dim
    flip_compress = False
  if data_next.shape[tile_axis] % data_tiling != 0:
    raise ValueError(
        f"data_tiling={data_tiling} must divide tile axis={data_next.shape[tile_axis]}")
  data_next = tiled_max(data_next, data_tiling, axis=tile_axis)
  data_next, permutation = compress_array(
      data_next, axis=compress_axis, placeholder=-1, flip=flip_compress)
  if data_tiling > 1:
    permutation = jnp.repeat(permutation, data_tiling, axis=tile_axis)
  block_mask = jnp.take_along_axis(block_mask, permutation, axis=compress_axis)
  mask_next = jnp.take_along_axis(
      mask_next_slice, permutation, axis=compress_axis)
  num_nonzero_blocks = jnp.sum(data_next >= 0, axis=compress_axis)

  if is_dkv:
    partial_mask_blocks = partial_mask_blocks.swapaxes(-1, -2)

  def _downcast(array: jax.Array, max_value: int) -> jax.Array:
    if array.size == 0:
      return array

    if array.dtype != np.int32:
      raise ValueError(f"Expected int32 input, but got {array.dtype}.")

    if max_value <= np.iinfo(np.int8).max:
      return array.astype(np.int8)
    elif max_value <= np.iinfo(np.int16).max:
      return array.astype(np.int16)
    else:
      return array.astype(np.int32)

  if downcast_block_mask_data:
    # values are in the range [0, 1, 2]
    block_mask = block_mask.astype(np.int8)
    data_next = _downcast(
        data_next, q_blocks_count if is_dkv else kv_blocks_count)
    mask_next = _downcast(mask_next, q_blocks_count * kv_blocks_count)
  partial_mask_blocks = partial_mask_blocks.reshape(
      [batch_size, 1, -1, q_block_size, kv_block_size])

  if is_dkv:
    assert data_next.shape == (
        batch_size, 1, q_blocks_count, kv_blocks_count // data_tiling), data_next.shape
    assert mask_next.shape == (
        batch_size, 1, q_blocks_count, kv_blocks_count), mask_next.shape
    assert block_mask.shape == (
        batch_size, 1, q_blocks_count, kv_blocks_count), block_mask.shape
    assert num_nonzero_blocks.shape == (
        batch_size, 1, kv_blocks_count // data_tiling), num_nonzero_blocks.shape
    assert partial_mask_blocks.shape == (
        batch_size,
        1,
        q_blocks_count * kv_blocks_count,
        q_block_size,
        kv_block_size,
    ), partial_mask_blocks.shape
  else:
    assert data_next.shape == (
        batch_size, 1, q_blocks_count // data_tiling, kv_blocks_count), data_next.shape
    assert mask_next.shape == (
        batch_size, 1, q_blocks_count, kv_blocks_count), mask_next.shape
    assert block_mask.shape == (
        batch_size, 1, q_blocks_count, kv_blocks_count), block_mask.shape
    assert num_nonzero_blocks.shape == (
        batch_size, 1, q_blocks_count // data_tiling), num_nonzero_blocks.shape
    assert partial_mask_blocks.shape == (
        batch_size,
        1,
        q_blocks_count * kv_blocks_count,
        q_block_size,
        kv_block_size,
    ), partial_mask_blocks.shape

  return MaskInfo(
      data_next=data_next,
      mask_next=mask_next,
      block_mask=block_mask,
      num_nonzero_blocks=num_nonzero_blocks,
      partial_mask_blocks=partial_mask_blocks,
      data_tiling=data_tiling,
      is_dkv=is_dkv,
  )
