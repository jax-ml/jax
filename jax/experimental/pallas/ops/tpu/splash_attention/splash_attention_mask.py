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

"""Mini-mask creation library."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any
import numpy as np

# mypy: ignore-errors

class Mask:
  """A base class for splash attention masks."""

  @property
  def shape(self) -> tuple[int, ...]:
    raise NotImplementedError

  def __getitem__(self, idx) -> np.ndarray:
    raise NotImplementedError

  def __bool__(self) -> bool:
    raise NotImplementedError(
        'Conversion to bool is unsupported. Could be caused by using logical'
        ' instead of bitwise operations on masks.'
    )

  def __or__(self, other: Mask) -> Mask:
    if self.shape != other.shape:
      raise ValueError(
          f'Invalid shape for other: {other.shape}, expected: {self.shape}'
      )
    return LogicalOr(self, other)

  def __and__(self, other: Mask) -> Mask:
    if self.shape != other.shape:
      raise ValueError(
          f'Invalid shape for other: {other.shape}, expected: {self.shape}'
      )
    return LogicalAnd(self, other)


def make_causal_mask(shape: tuple[int, int], offset: int = 0) -> np.ndarray:
  """Makes a causal attention mask.

  Args:
    shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.

  Returns:
    The causal mask.
  """
  q_seq_len, kv_seq_len = shape
  q_idx = np.arange(q_seq_len, dtype=np.int32)
  kv_idx = np.arange(kv_seq_len, dtype=np.int32)
  return (q_idx[:, None] + offset >= kv_idx[None, :]).astype(np.bool_)


def make_local_attention_mask(
    shape: tuple[int, int],
    window_size: tuple[int | None, int | None],
    *,
    offset: int = 0,
) -> np.ndarray:
  """Makes a local attention mask."""
  q_seq_len, kv_seq_len = shape
  q_idx = np.arange(q_seq_len, dtype=np.int32)
  kv_idx = np.arange(kv_seq_len, dtype=np.int32)
  mask = np.ones((q_seq_len, kv_seq_len), dtype=np.bool_)
  left, right = window_size
  if left is not None:
    mask = mask & (q_idx[:, None] - left + offset <= kv_idx[None, :])
  if right is not None:
    mask = mask & (q_idx[:, None] + right + offset >= kv_idx[None, :])
  return mask.astype(np.bool_)


def make_chunk_attention_mask(
    shape: tuple[int, int], chunk_size: int
) -> np.ndarray:
  """Makes a chunked causal attention mask.

  Args:
    shape: The desired shape of the mask (q_seq_len, kv_seq_len).
    chunk_size: The size of the attention chunks.

  Returns:
    A boolean mask of shape `mask_shape` where True indicates attention is
    allowed according to chunked causal rules, and False otherwise.

  Raises:
    ValueError: If chunk_window_size is None or not positive.
  """
  if chunk_size <= 0:
    raise ValueError('chunk_size must be positive')

  q_seq_len, kv_seq_len = shape
  q_idx = np.arange(q_seq_len, dtype=np.int32)
  kv_idx = np.arange(kv_seq_len, dtype=np.int32)

  # chunk mask calculation
  same_chunk = (q_idx[:, None] // chunk_size) == (kv_idx[None, :] // chunk_size)
  mask = same_chunk & (q_idx[:, None] >= kv_idx[None, :])
  return mask


def make_random_mask(
    shape: tuple[int, int], sparsity: float, seed: int
) -> np.ndarray:
  """Makes a random attention mask."""
  np.random.seed(seed)
  return np.random.binomial(n=1, p=1.0 - sparsity, size=shape).astype(np.bool_)


@dataclasses.dataclass
class LogicalOr(Mask):
  left: Mask
  right: Mask

  def __init__(self, left: Mask, right: Mask):
    if left.shape != right.shape:
      raise ValueError('Masks must have the same shape')
    self.left = left
    self.right = right

  @property
  def shape(self) -> tuple[int, ...]:
    return self.left.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.left[idx] | self.right[idx]

  def __hash__(self):
    return hash((type(self),) + (self.left, self.right))


@dataclasses.dataclass
class LogicalAnd(Mask):
  left: Mask
  right: Mask

  def __init__(self, left: Mask, right: Mask):
    if left.shape != right.shape:
      raise ValueError('Masks must have the same shape')
    self.left = left
    self.right = right

  @property
  def shape(self) -> tuple[int, ...]:
    return self.left.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.left[idx] & self.right[idx]

  def __hash__(self):
    return hash((type(self),) + (self.left, self.right))


@dataclasses.dataclass
class MultiHeadMask(Mask):
  """Lazy multihead mask, combines multiple lazy masks one per head."""

  masks: Sequence[Mask]

  def __post_init__(self):
    if not self.masks:
      raise ValueError('Unsupported empty tuple of masks')

    shape = self.masks[0].shape
    for mask in self.masks[1:]:
      if shape != mask.shape:
        raise ValueError(
            f'Unexpected mask shape, got: {mask.shape}, expected: {shape}'
        )

    if not all(isinstance(mask, Mask) for mask in self.masks):
      raise ValueError('masks should be of type Mask')

    if any(isinstance(mask, MultiHeadMask) for mask in self.masks):
      raise ValueError('Nesting MultiHeadMasks is not supported')

  @property
  def shape(self) -> tuple[int, ...]:
    return (len(self.masks),) + self.masks[0].shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 3:
      raise NotImplementedError(f'Unsupported slice: {idx}')

    head_slice = idx[0]
    if isinstance(head_slice, int):
      assert head_slice >= 0 and head_slice <= len(self.masks)
      return self.masks[head_slice][idx[1:]]
    else:
      slice_masks = [mask[idx[1:]] for mask in self.masks[head_slice]]
      return np.stack(slice_masks)

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return self.masks == other.masks

  def __hash__(self):
    return hash((type(self),) + tuple(hash(mask) for mask in self.masks))


class _ComputableMask(Mask):
  """Superclass for all masks that can be computed inside the kernel using a callable object.

  This subclass is designed to be used with Splash Attention.
  It allows the mask logic to be computed on-the-fly or fused into the attention
  kernel, avoiding the memory cost of materializing the full
  (sequence_length, sequence_length) boolean mask array, which can be excessive
  for long sequences.

  Attributes:
    _shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
    q_sequence: Indices of Q sequence. q_sequence is reused across __getitem__
      calls which is important for compile-time performance.
    mask_function: Function used by the SplashAttention kernel to compute the
      mask rather than loading it.
  """

  _shape: tuple[int, int]
  q_sequence: np.ndarray
  mask_function: Callable[..., Any]

  def __init__(
      self,
      shape: tuple[int, int],
      mask_function: Callable[..., Any],
      shard_count: int = 1,
  ):
    self._shape = shape
    self.mask_function = mask_function
    q_seq_len = self.shape[0]

    if q_seq_len % (shard_count * shard_count) != 0:
      raise ValueError(
          f'Shard count squared ({shard_count * shard_count}) must'
          f' divide Q seq_len ({self.shape[0]}) evenly.'
      )

    self.q_sequence = np.arange(q_seq_len, dtype=np.int32)

  @property
  def shape(self) -> tuple[int, ...]:
    return self._shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f'Unsupported slice: {idx}')

    q_slice, kv_slice = idx
    if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
      raise NotImplementedError(f'Unsupported slice: {idx}')

    q_slice = _fill_slice(q_slice, self.shape[0])
    kv_slice = _fill_slice(kv_slice, self.shape[1])

    rows = self.q_sequence[q_slice]
    cols = np.arange(kv_slice.start, kv_slice.stop)

    return self.mask_function(rows[:, None], cols[None, :])

  def __eq__(self, other: object):
    raise NotImplementedError()

  def __hash__(self):
    raise NotImplementedError()


class CausalMask(_ComputableMask):
  """Lazy causal mask, prevents the model from attending to future tokens.

  Attributes:
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  offset: int

  def __init__(
      self,
      shape: tuple[int, int],
      offset: int = 0,
      shard_count: int = 1,
  ):
    self.offset = offset

    def causal_mask_function(q_ids, kv_ids):
      # When evaluating the mask in _process_mask we typically work with numpy
      # array views.
      # Avoid the addition when possible to avoid instantiating an actual array.
      if self.offset == 0:
        return q_ids >= kv_ids
      else:
        return q_ids + self.offset >= kv_ids

    mask_function = causal_mask_function

    super().__init__(
        shape=shape,
        mask_function=mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return (
        self.shape == other.shape
        and self.offset == other.offset
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.offset,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


class ChunkedCausalMask(_ComputableMask):
  """Lazy chunked causal mask.

  Attention is causal within each chunk (0, K), (K, 2K), (2K, 3K), ... tokens
  attend to each other but not across chunks.
  Llama4 models use interleaved chunk attention along with global attention.


  Attributes:
    chunk_size: The size of each attention chunk.
  """

  chunk_size: int

  def __init__(
      self,
      shape: tuple[int, int],
      chunk_size: int,
      shard_count: int = 1,
  ):
    if chunk_size <= 0:
      raise ValueError('chunk_size must be positive')
    self.chunk_size = chunk_size

    # Define the mask function for chunk attention
    def chunked_causal_mask_function(q_ids, kv_ids):
      """Computes the mask logic for the given slice indices."""
      # Condition 1: Same chunk
      same_chunk = (q_ids // self.chunk_size) == (kv_ids // self.chunk_size)

      # Condition 2: Causal
      causal = q_ids >= kv_ids

      return same_chunk & causal

    super().__init__(
        shape=shape,
        mask_function=chunked_causal_mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return (
        self.shape == other.shape
        and self.chunk_size == other.chunk_size
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.chunk_size,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


class LocalMask(_ComputableMask):
  """Lazy local mask, prevents model from attending to tokens outside window.

  Attributes:
    window_size: Size of the two sides of the local window (None identifies no
      limit for the given side).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  window_size: tuple[int | None, int | None]
  offset: int

  def __init__(
      self,
      shape: tuple[int, int],
      window_size: tuple[int | None, int | None],
      offset: int,
      shard_count: int = 1,
  ):
    self.window_size = window_size
    self.offset = offset

    def local_mask_function(q_ids, kv_ids):
      """Computes the local attention mask for the given slice indices."""
      left_size, right_size = self.window_size

      assert q_ids.ndim == 2
      assert kv_ids.ndim == 2

      if left_size is None and right_size is None:
        return np.ones((q_ids.shape[0], kv_ids.shape[1]), dtype=np.bool_)

      # Avoid the addition when possible to avoid instantiating an actual array.
      if offset != 0:
        shifted_q_ids = q_ids + self.offset
      else:
        shifted_q_ids = q_ids

      mask = None
      if left_size is not None:
        mask = shifted_q_ids - left_size <= kv_ids
      if right_size is not None:
        if mask is None:
          mask = shifted_q_ids + right_size >= kv_ids
        else:
          mask &= shifted_q_ids + right_size >= kv_ids
      return mask

    super().__init__(
        shape=shape,
        mask_function=local_mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return False

    return (
        self.shape == other.shape
        and self.window_size == other.window_size
        and self.offset == other.offset
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.window_size,
        self.offset,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


@dataclasses.dataclass
class NumpyMask(Mask):
  """A mask backed by a dense numpy array."""

  array: np.ndarray

  def __post_init__(self):
    if self.array.ndim != 2:
      raise ValueError('Expected a 2-dim array')

    if self.array.dtype != np.bool_:
      raise ValueError('Mask must be a boolean array')

  @property
  def shape(self) -> tuple[int, ...]:
    return self.array.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.array[idx]

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return np.array_equal(self.array, other.array, equal_nan=True)

  def __hash__(self):
    return hash((type(self), self.array.tobytes()))


def _fill_slice(inp_slice: slice, size: int) -> slice:
  assert inp_slice.step is None or inp_slice.step == 1
  start = 0 if inp_slice.start is None else inp_slice.start
  stop = size if inp_slice.stop is None else inp_slice.stop
  assert start >= 0
  assert stop <= size
  return slice(start, stop, None)


@dataclasses.dataclass(frozen=True)
class FullMask(Mask):
  """Lazy full mask, allows all tokens to attend to all other tokens."""

  # TODO(amagni): Transform FullMask into a _ComputableMask.

  _shape: tuple[int, int]

  def __post_init__(self):
    if not isinstance(self.shape, tuple):
      raise ValueError(f'Unsupported shape type: {type(self.shape)}')

  @property
  def shape(self) -> tuple[int, ...]:
    return self._shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f'Unsupported slice: {idx}')
    i, j = idx
    if not isinstance(i, slice) or not isinstance(j, slice):
      raise NotImplementedError(f'Unsupported slice: {idx}')
    i = _fill_slice(i, self.shape[0])
    j = _fill_slice(j, self.shape[1])
    return np.ones((i.stop - i.start, j.stop - j.start), dtype=np.bool_)

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return self.shape == other.shape

  def __hash__(self):
    return hash((type(self), self.shape))
