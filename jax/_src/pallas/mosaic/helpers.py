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

"""Helpers for Pallas TPU kernels."""

import functools
import jax
from jax._src.pallas import helpers as pl_helpers
from jax._src.pallas import primitives as pl_primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import primitives as plm_primitives
import jax.numpy as jnp


def sync_copy(src_ref, dst_ref, *, add: bool = False) -> None:
  """Synchronously copies a PyTree of refs to another PyTree of refs."""
  if not jax.tree.leaves(src_ref):
    # No buffers to copy so skip the function.
    return

  @functools.partial(
      pl_primitives.run_scoped, sem=tpu_core.SemaphoreType.DMA(())
  )
  def _(sem):
    def _copy_start_or_wait(action, src_ref, dst_ref):
      descriptor = plm_primitives.make_async_copy(src_ref, dst_ref, sem)
      if action == "start":
        descriptor.start(add=add)
      elif action == "wait":
        descriptor.wait()
      else:
        raise ValueError(f"Unknown action: {action}")

    jax.tree.map(
        functools.partial(_copy_start_or_wait, "start"),
        src_ref,
        dst_ref,
    )
    jax.tree.map(
        functools.partial(_copy_start_or_wait, "wait"),
        src_ref,
        dst_ref,
    )


def run_on_first_core(core_axis_name: str):
  """Runs a function on the first core in a given axis."""
  num_cores = jax.lax.axis_size(core_axis_name)
  if num_cores == 1:
    return lambda f: f()

  def wrapped(f):
    core_id = jax.lax.axis_index(core_axis_name)

    @pl_helpers.when(core_id == 0)
    @functools.wraps(f)
    def _():
      return f()

  return wrapped


def core_barrier(sem, *, core_axis_name: str):
  """Synchronizes all cores in a given axis."""
  num_cores = jax.lax.axis_size(core_axis_name)
  core_id = jax.lax.axis_index(core_axis_name)

  @pl_helpers.when(num_cores > 1)
  def _():
    with jax.named_scope("sync_cores"):

      def signal_core(i):
        # Don't signal ourself
        @pl_helpers.when(core_id != i)
        def _():
          pl_primitives.semaphore_signal(sem, 1, core_index=i)

      for i in range(num_cores):
        signal_core(i)
      pl_primitives.semaphore_wait(sem, num_cores - 1)


def join_bits(
    *,
    two_bits: jax.Array | None = None,
    four_bits: jax.Array | None = None,
) -> jax.Array:
  """Joins sub-byte bit segments into an 8-bit container array (`int8`).

  When both `two_bits` and `four_bits` are provided, `four_bits` is placed into
  the least significant bits [3:0] and `two_bits` is placed into bits [5:4],
  with upper bits [7:6] zero-filled. When only one segment is provided, it is
  placed into the least significant bits of each byte with upper bits zeroed.

  Args:
    two_bits: An optional 2-bit array (`int2` or `uint2`).
    four_bits: An optional 4-bit array (`int4` or `uint4`).

  Returns:
    An 8-bit container array (`int8`) of the same shape.
  """
  if two_bits is None and four_bits is None:
    raise ValueError("At least one bit segment must be specified.")
  if (
      two_bits is not None
      and four_bits is not None
      and two_bits.shape != four_bits.shape
  ):
    raise ValueError(
        f"Shapes of two_bits {two_bits.shape} and four_bits {four_bits.shape}"
        " must match."
    )

  two_b_i4 = None
  if two_bits is not None:
    if jax.dtypes.itemsize_bits(two_bits.dtype) != 2:
      raise ValueError(f"Expected 2-bit type, got {two_bits.dtype}")
    # Zero-extend 2b -> 4b so bits [3:2] are 00.
    two_b_u4 = plm_primitives.bitcast(two_bits, ty=jnp.uint2).astype(jnp.uint4)
    two_b_i4 = plm_primitives.bitcast(two_b_u4, ty=jnp.int4)

  if four_bits is not None:
    if jax.dtypes.itemsize_bits(four_bits.dtype) != 4:
      raise ValueError(f"Expected 4-bit type, got {four_bits.dtype}")
    low_4b = plm_primitives.bitcast(four_bits, ty=jnp.int4)
    high_8b = (
        two_b_i4.astype(jnp.int8)
        if two_b_i4 is not None
        else jnp.zeros(low_4b.shape, dtype=jnp.int8)
    )
  else:
    assert two_b_i4 is not None
    low_4b = two_b_i4
    high_8b = jnp.zeros(low_4b.shape, dtype=jnp.int8)

  low_8b = low_4b.astype(jnp.int8)
  out = plm_primitives.pack_elementwise(
      (low_8b, high_8b), packed_dtype=jnp.int4
  )
  return plm_primitives.bitcast(out, ty=jnp.int8)


def split_bits(
    x: jax.Array,
    bitwidths: tuple[int, ...],
) -> tuple[jax.Array, ...]:
  """Splits an 8-bit array into sub-byte bit segments.

  Bitwidths are specified from least significant bits (LSB) to most significant
  bits (MSB).

  Args:
    x: An 8-bit array (`int8` or `uint8`).
    bitwidths: A tuple of sub-byte bitwidths to extract (e.g. `(4, 2)`, `(2,
      4)`, `(4,)`, `(2,)`).

  Returns:
    A tuple of sub-byte arrays (`int2` or `int4`) corresponding to each
    bitwidth.
  """
  if jax.dtypes.itemsize_bits(x.dtype) != 8:
    raise ValueError(f"Expected 8-bit array, got {x.dtype}")
  if sum(bitwidths) > 8:
    raise ValueError(f"Total bitwidth {sum(bitwidths)} exceeds 8 bits.")

  x_i8 = plm_primitives.bitcast(x, ty=jnp.int8)
  results = []
  shift = 0
  for bw in bitwidths:
    if bw not in (2, 4):
      raise ValueError(
          f"Unsupported bitwidth: {bw}. Only 2 and 4 are supported."
      )
    target_dtype = jnp.int4 if bw == 4 else jnp.int2
    if shift == 0:
      shifted = x_i8
    elif shift % bw == 0:
      shifted = plm_primitives.unpack_elementwise(
          x_i8,
          index=shift // bw,
          packed_dtype=target_dtype,
          unpacked_dtype=jnp.int8,
      )
    else:
      x_u8 = plm_primitives.bitcast(x_i8, ty=jnp.uint8)
      shifted = plm_primitives.bitcast(x_u8 >> shift, ty=jnp.int8)
    results.append(shifted.astype(target_dtype))
    shift += bw

  return tuple(results)


def unpack_fp6(
    two_bits: jax.Array,
    four_bits: jax.Array,
) -> jax.Array:
  """Unpacks 2b and 4b split layout arrays into a 6b-in-8b container array.

  Places `four_bits` into bits [3:0] (LSB) and `two_bits` into bits [5:4] (MSB)
  of each byte in the 8-bit container, with upper bits [7:6] zero-filled.

  Args:
    two_bits: A 2-bit array (`int2` or `uint2`).
    four_bits: A 4-bit array (`int4` or `uint4`).

  Returns:
    An 8-bit container array (`int8`) of the same shape.
  """
  return join_bits(two_bits=two_bits, four_bits=four_bits)


def pack_fp6(
    fp6_in_8b: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Packs a 6b-in-8b container array into 2b and 4b split layout arrays.

  Extracts bits [5:4] into a 2-bit array (`int2`) and bits [3:0] into a 4-bit
  array (`int4`).

  Args:
    fp6_in_8b: An 8-bit array containing 6b elements in the lower 6 bits.

  Returns:
    A tuple of `(two_bits, four_bits)` where `two_bits` is `int2` and
    `four_bits` is `int4`.
  """
  four_bits, two_bits = split_bits(fp6_in_8b, bitwidths=(4, 2))
  return two_bits, four_bits
