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
"""Implementation of the Philox PRNG as a Pallas kernel."""
from collections.abc import Sequence
import jax
from jax import typing
from jax._src import prng
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.random import prng_utils

Shape = Sequence[int]

BLOCK_SIZE = (256, 256)

# Philox constants. See original paper at:
# "Parallel Random Numbers: As Easy as 1, 2, 3", Salmon et. al. 2011
K_HI_32 = 0x9E3779B9
K_LO_32 = 0xBB67AE85
MUL_A = 0xCD9E8D57
MUL_B = 0xD2511F53


def mul32_hi_lo(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Multiplies 2 32-bit values and returns the hi+low bits of the result."""
  xhi = x >> 16
  yhi = y >> 16
  xlo = x & 0xffff
  ylo = y & 0xffff

  xy_hi = xhi * yhi
  xy_lo = xlo * ylo
  cross_xy = xhi * ylo
  cross_yx = xlo * yhi
  carry = (cross_xy & 0xffff) + (cross_yx & 0xffff) + (xy_lo >> 16)
  return xy_hi + (cross_xy >> 16) + (cross_yx >> 16) + (carry >> 16), xy_lo


def philox_4x32(hi0, lo0, hi1, lo1, k_hi, k_lo, rounds = 10):
  """Philox 4x32 keyed hash function."""
  k_hi_const = jnp.array(K_HI_32, dtype=jnp.uint32)
  k_lo_const = jnp.array(K_LO_32, dtype=jnp.uint32)
  mul_a = jnp.array(MUL_A, dtype=jnp.uint32)
  mul_b = jnp.array(MUL_B, dtype=jnp.uint32)

  for i in range(rounds):
    # Compute the round.
    new_hi0, new_lo0 = mul32_hi_lo(mul_a, hi1)
    new_hi0 = new_hi0 ^ lo0 ^ k_hi
    new_hi1, new_lo1 = mul32_hi_lo(mul_b, hi0)
    new_hi1 = new_hi1 ^ lo1 ^ k_lo
    hi0, lo0, hi1, lo1 = new_hi0, new_lo0, new_hi1, new_lo1

    # Raise the key on all iterations except for the last round.
    if i != rounds - 1:
      k_hi = k_hi + k_hi_const
      k_lo = k_lo + k_lo_const
  return hi0, lo0, hi1, lo1


def philox_4x32_kernel(key,
                      shape: Shape,
                      unpadded_shape: Shape,
                      block_size: tuple[int, int],
                      offset: typing.ArrayLike = 0,
                      fuse_output: bool = True):
  """Generates random bits using the Philox keyed hash function.

  Args:
    key: A Philox key of shape (2,).
    shape: The shape of the output. Must be divisible by `block_size`.
    unpadded_shape: If `shape` is padded, then this is the shape of the
      output tensor if it were not padded. This is important for indexing
      calculations within the kernel. If `shape` is not padded, then this
      should be equal to `shape`.
    block_size: The block size of the kernel.
    offset: An optional offset to the counts.
    fuse_output: Whether to fuse the output bits into a single value.

  Returns:
    A tensor of random bits of shape `shape` if fuse_output=True. Otherwise,
    this will return a tensor of shape (2, *shape) with the first channel being
    the high bits and the second channel being the low bits.
  """
  shape = tuple(shape)
  if np.prod(shape) > jnp.iinfo(jnp.uint32).max:
    raise ValueError(
        f"Shape too large: {np.prod(shape)} > {np.iinfo(jnp.uint32).max}")

  if (shape[-2] % block_size[-2] != 0) or (shape[-1] % block_size[-1] != 0):
    raise ValueError(
        f"Shape dimension {shape[-2:]} must be divisible by {block_size}")
  grid_dims = shape[:-2] + (
      shape[-2] // block_size[-2], shape[-1] // block_size[1],)
  offset = jnp.array(offset, dtype=jnp.uint32)
  if offset.ndim != 0:
    raise ValueError(f"Offset must be scalar, got {offset.shape}")
  offset = jnp.reshape(offset, (1,))

  def kernel(offset_ref, key_ref, out_ref):
    counts_idx = tuple(pl.program_id(i) for i in range(len(grid_dims)))
    offset = prng_utils.compute_scalar_offset(
        counts_idx, unpadded_shape, block_shape)
    counts_lo = prng_utils.blocked_iota(block_size, unpadded_shape)
    counts_lo = counts_lo + offset + offset_ref[0]
    counts_lo = counts_lo.astype(jnp.uint32)
    # TODO(justinfu): Support hi bits on count.
    _zeros = jnp.zeros_like(counts_lo)
    k1 = jnp.reshape(key_ref[0, 0], (1, 1))
    k2 = jnp.reshape(key_ref[0, 1], (1, 1))
    o1, o2, _, _ = philox_4x32(_zeros, counts_lo, _zeros, _zeros, k1, k2)
    if fuse_output:
      out_bits = o1 ^ o2
      out_ref[...] = out_bits.reshape(out_ref.shape)
    else:
      out_ref[0, ...] = o1.reshape(out_ref[0].shape)
      out_ref[1, ...] = o2.reshape(out_ref[0].shape)

  key = key.reshape((1, 2))
  block_shape = (1,) * (len(shape)-2) + block_size
  if fuse_output:
    out = jax.ShapeDtypeStruct(shape, dtype=jnp.uint32)
    out_spec = pl.BlockSpec(block_shape, lambda *idxs: idxs)
  else:
    out = jax.ShapeDtypeStruct((2,) + shape, dtype=jnp.uint32)
    out_spec = pl.BlockSpec((2,) + block_shape, lambda *idxs: (0, *idxs))
  return pl.pallas_call(
      kernel,
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.SMEM),
      ],
      out_specs=out_spec,
      grid=grid_dims,
      out_shape=out,
  )(offset, key)


def philox_4x32_count(key,
                      shape: Shape,
                      offset: typing.ArrayLike = 0,
                      fuse_output: bool = True):
  """Convenience function to call philox_4x32_kernel with padded shapes."""
  if len(shape) == 0:
    return philox_4x32_count(
        key, (1, 1), offset=offset, fuse_output=fuse_output)[..., 0, 0]
  elif len(shape) == 1:
    return philox_4x32_count(
        key, (1, *shape), offset=offset, fuse_output=fuse_output)[..., 0, :]

  requires_pad = (
      shape[-2] % BLOCK_SIZE[-2] != 0) or (shape[-1] % BLOCK_SIZE[-1] != 0)
  if requires_pad:
    padded_shape = tuple(shape[:-2]) + (
        prng_utils.round_up(shape[-2], BLOCK_SIZE[-2]),
        prng_utils.round_up(shape[-1], BLOCK_SIZE[-1]),
    )
    padded_result = philox_4x32_kernel(
        key, padded_shape, shape,
        block_size=BLOCK_SIZE, offset=offset,
        fuse_output=fuse_output)
    return padded_result[..., :shape[-2], :shape[-1]]
  else:
    return philox_4x32_kernel(key, shape, shape,
                              block_size=BLOCK_SIZE, offset=offset,
                              fuse_output=fuse_output)


def philox_split(key, shape: Shape):
  """Splits the key into two keys of the same shape."""
  bits1, bits2 = philox_4x32_count(key, shape, fuse_output=False)
  return jnp.stack([bits1, bits2], axis=bits1.ndim)


def philox_random_bits(key, bit_width: int, shape: Shape):
  if bit_width != 32:
    raise ValueError("Only 32-bit PRNG supported.")
  return philox_4x32_count(key, shape, fuse_output=True)


def philox_fold_in(key, data):
  assert data.ndim == 0
  return philox_4x32_count(key, (), offset=data, fuse_output=False)


plphilox_prng_impl = prng.PRNGImpl(
    key_shape=(2,),
    seed=prng.threefry_seed,
    split=philox_split,
    random_bits=philox_random_bits,
    fold_in=philox_fold_in,
    name="pallas_philox4x32",
    tag="pllox")

prng.register_prng(plphilox_prng_impl)
