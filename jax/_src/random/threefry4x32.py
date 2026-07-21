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

"""Threefry 4x32 PRNG implementation for JAX.

This implements the Threefry 4x32 counter-based PRNG from the Random123 library
(D.E. Shaw Research) as a JAX PRNGImpl. The key shape is (4,) with uint32 dtype,
and the hash function applies 20 rounds of the Threefry 4-word mix.

Usage:
  key = jax.random.key(0, impl='threefry4x32')
  x = jax.random.uniform(key, shape=(10,))
"""

from __future__ import annotations

import functools
import math

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import numpy as jnp
from jax._src import typing
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import lax
from jax._src.random import prng

import numpy as np

# -- Threefry 4x32 rotation constants from Random123 --
# These are the R_32x4 constants from the Threefish/Skein reference.
# Each group of 4 rounds uses a pair of rotation constants (rot0, rot1).
# The pattern repeats every 8 rounds.
_ROTATIONS_32X4 = np.array(
    [
        [10, 26],
        [11, 21],
        [13, 27],
        [23, 5],
        [6, 20],
        [17, 11],
        [25, 10],
        [18, 20],
    ],
    dtype=np.uint32,
)

# Skein key schedule parity constant.
_SKEIN_KS_PARITY32 = np.uint32(0x1BD11BDA)

_DEFAULT_ROUNDS = 20


# -- rotate_left helper --


def _rotate_left_u32(x, d):
  """Rotate 32-bit unsigned integer x left by d bits."""
  d = np.uint32(d)
  return lax.shift_left(x, d) | lax.shift_right_logical(x, np.uint32(32) - d)


# -- Core Threefry 4x32 hash --


def _threefry4x32_lowering(k0, k1, k2, k3, x0, x1, x2, x3):
  """Apply the Threefry 4x32 hash with 20 rounds.

  Args:
    k0: uint32 array forming the first part of the key.
    k1: uint32 array forming the second part of the key.
    k2: uint32 array forming the third part of the key.
    k3: uint32 array forming the fourth part of the key.
    x0: uint32 array forming the first part of the counter/input.
    x1: uint32 array forming the second part of the counter/input.
    x2: uint32 array forming the third part of the counter/input.
    x3: uint32 array forming the fourth part of the counter/input.

  Returns:
    A tuple of four uint32 arrays (out0, out1, out2, out3).
  """
  # Key schedule: ks[i] = k[i], ks[4] = k0 ^ k1 ^ k2 ^ k3 ^ parity
  ks = [k0, k1, k2, k3, k0 ^ k1 ^ k2 ^ k3 ^ _SKEIN_KS_PARITY32]

  # Initial key injection
  x0 = x0 + ks[0]
  x1 = x1 + ks[1]
  x2 = x2 + ks[2]
  x3 = x3 + ks[3]

  for rnd in range(_DEFAULT_ROUNDS):
    rot0 = _ROTATIONS_32X4[rnd % 8, 0]
    rot1 = _ROTATIONS_32X4[rnd % 8, 1]

    if (rnd % 2) == 0:
      # Even sub-round: mix (0,1) and (2,3)
      x0 = x0 + x1
      x1 = _rotate_left_u32(x1, rot0)
      x1 = x0 ^ x1
      x2 = x2 + x3
      x3 = _rotate_left_u32(x3, rot1)
      x3 = x2 ^ x3
    else:
      # Odd sub-round: mix (0,3) and (2,1) — the 4-word permutation
      x0 = x0 + x3
      x3 = _rotate_left_u32(x3, rot0)
      x3 = x0 ^ x3
      x2 = x2 + x1
      x1 = _rotate_left_u32(x1, rot1)
      x1 = x2 ^ x1

    # Key injection every 4 rounds
    if (rnd & 3) == 3:
      inject_idx = rnd // 4
      x0 = x0 + ks[(1 + inject_idx) % 5]
      x1 = x1 + ks[(2 + inject_idx) % 5]
      x2 = x2 + ks[(3 + inject_idx) % 5]
      x3 = x3 + ks[(4 + inject_idx) % 5]
      x3 = x3 + np.uint32(1 + inject_idx)

  return (x0, x1, x2, x3)


# -- Primitive definition --


def _threefry4x32_abstract_eval(*args):
  """Abstract evaluation for the threefry4x32 primitive."""
  if len(args) != 8:
    raise TypeError(f"threefry4x32_p expects 8 arguments, got {len(args)}.")
  if all(isinstance(arg, core.ShapedArray) for arg in args):
    shape = lax.broadcasting_shape_rule("threefry4x32", *args)
    sharding = lax.broadcasting_sharding_rule("threefry4x32", *args)
    aval = core.ShapedArray(shape, np.dtype("uint32"), sharding=sharding)
  else:
    raise TypeError(f"Arguments to threefry4x32 must all be arrays, got {args}")
  if any(a.dtype != np.uint32 for a in args):
    raise TypeError(
        f"Arguments to threefry4x32 must have uint32 type, got {args}"
    )
  return (aval,) * 4


threefry4x32_p = core.Primitive("threefry4x32")
threefry4x32_p.multiple_results = True
threefry4x32_p.def_impl(
    functools.partial(dispatch.apply_primitive, threefry4x32_p)
)
threefry4x32_p.def_abstract_eval(_threefry4x32_abstract_eval)
batching.defbroadcasting(threefry4x32_p)

_threefry4x32_lowering_rule = mlir.lower_fun(
    _threefry4x32_lowering, multiple_results=True
)

mlir.register_lowering(
    threefry4x32_p, _threefry4x32_lowering_rule, inline=False
)


# -- PRNGImpl functions --


def _is_threefry4x32_key(key: typing.Array) -> bool:
  """Check if the key is a valid Threefry 4x32 PRNG key."""
  try:
    return key.shape == (4,) and key.dtype == np.uint32
  except AttributeError:
    return False


def threefry4x32_seed(seed: typing.Array) -> typing.Array:
  """Create a single Threefry 4x32 PRNG key from an integer seed.

  The 4-word key is constructed by splitting the seed into two uint32 values
  and hashing with a fixed counter to fill all 4 words.

  Args:
    seed: A scalar integer array.

  Returns:
    A 4-word uint32 PRNG key with shape (4,).
  """
  return _threefry4x32_seed(seed)


@api.jit(inline=True)
def _threefry4x32_seed(seed: typing.Array) -> typing.Array:
  """Create a single Threefry 4x32 PRNG key from an integer seed."""
  if seed.shape:
    raise TypeError(f"PRNG key seed must be a scalar; got {seed!r}.")
  if not np.issubdtype(seed.dtype, np.integer):
    raise TypeError(f"PRNG key seed must be an integer; got {seed!r}")
  convert = lambda k: lax.convert_element_type(k, np.uint32)
  k0 = convert(
      lax.shift_right_logical(seed, lax.convert_element_type(32, seed.dtype))
  )
  with config.numpy_dtype_promotion("standard"):
    k1 = convert(jnp.bitwise_and(seed, np.uint32(0xFFFFFFFF)))
  # Hash through threefry4x32 to fill all 4 words from the 2-word seed.
  out = threefry4x32_p.bind(
      k0,
      k1,
      np.uint32(0),
      np.uint32(0),
      np.uint32(0),
      np.uint32(0),
      np.uint32(0),
      np.uint32(0),
  )
  return jnp.stack([lax.expand_dims(x, [0]) for x in out], axis=0).reshape(4)


def threefry4x32_split(key: typing.Array, shape: prng.Shape) -> typing.Array:
  """Split a Threefry 4x32 key into multiple sub-keys.

  Args:
    key: A 4-word uint32 PRNG key with shape (4,).
    shape: The shape of the output array of sub-keys.

  Returns:
    A batched array of sub-keys with shape (*shape, 4).
  """
  shape = tuple(map(core.concrete_dim_or_error, shape))
  return _threefry4x32_split(key, shape)


@api.jit(static_argnums=(1,), inline=True)
def _threefry4x32_split(key: typing.Array, shape: prng.Shape) -> typing.Array:
  """Split a Threefry 4x32 key into multiple sub-keys."""
  k0, k1, k2, k3 = key[0], key[1], key[2], key[3]

  # Generate counters for each sub-key.
  counts1, counts2 = prng.iota_2x32_shape(shape)
  zeros = jnp.zeros(shape, dtype=np.uint32)

  out0, out1, out2, out3 = threefry4x32_p.bind(
      k0, k1, k2, k3, zeros, zeros, counts1, counts2
  )

  return jnp.stack([out0, out1, out2, out3], axis=len(shape))


def threefry4x32_fold_in(key: typing.Array, data: typing.Array) -> typing.Array:
  """Fold data into a Threefry 4x32 key.

  Args:
    key: A 4-word uint32 PRNG key with shape (4,).
    data: A scalar integer array to fold into the key.

  Returns:
    A 4-word uint32 PRNG key with shape (*data.shape, 4,).
  """
  return _threefry4x32_fold_in(key, data)


@api.jit
def _threefry4x32_fold_in(key: typing.Array, data: typing.Array) -> typing.Array:
  """Fold data into a Threefry 4x32 key."""
  # Hash the key with the data used as part of the counter.
  k0, k1, k2, k3 = key[0], key[1], key[2], key[3]
  out0, out1, out2, out3 = threefry4x32_p.bind(
      k0, k1, k2, k3, np.uint32(0), np.uint32(0), np.uint32(0), data
  )
  return jnp.array([out0, out1, out2, out3], dtype=np.uint32)


def threefry4x32_random_bits(
    key: typing.Array, bit_width: int, shape: tuple[int, ...]
) -> typing.Array:
  """Sample uniform random bits using a Threefry 4x32 key.

  Args:
    key: A 4-word uint32 PRNG key with shape (4,).
    bit_width: The bit width of the output random bits (8, 16, 32, or 64).
    shape: The shape of the output array of random bits.

  Returns:
    An array of uniform random bits with shape (*shape,) and dtype corresponding
    to the bit width.
  """
  if not _is_threefry4x32_key(key):
    raise TypeError("threefry4x32_random_bits got invalid prng key.")
  if bit_width not in (8, 16, 32, 64):
    raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")
  return _threefry4x32_random_bits(key, bit_width, shape)


@api.jit(static_argnums=(1, 2), inline=True)
def _threefry4x32_random_bits(
    key: typing.Array, bit_width: int, shape: tuple[int, ...]
) -> typing.Array:
  """Sample uniform random bits using a Threefry 4x32 key."""
  if all(core.is_constant_dim(d) for d in shape) and math.prod(shape) > 2**64:
    raise NotImplementedError("random bits array of size exceeding 2 ** 64")

  k0, k1, k2, k3 = key[0], key[1], key[2], key[3]
  counts1, counts2 = prng.iota_2x32_shape(shape)
  zeros = jnp.zeros(shape, dtype=np.uint32)

  out0, out1, out2, out3 = threefry4x32_p.bind(
      k0, k1, k2, k3, counts1, counts2, zeros, zeros
  )

  dtype = prng.UINT_DTYPES[bit_width]
  if bit_width == 64:
    # Combine four 32-bit outputs into one 64-bit value.
    bits_hi = lax.convert_element_type(out0 ^ out2, dtype)
    bits_lo = lax.convert_element_type(out1 ^ out3, dtype)
    return lax.shift_left(bits_hi, jnp.asarray(32, dtype=dtype)) | bits_lo
  elif bit_width == 32:
    # XOR all four outputs for maximum mixing.
    return out0 ^ out1 ^ out2 ^ out3
  else:
    return lax.convert_element_type(out0 ^ out1 ^ out2 ^ out3, dtype)


# -- PRNGImpl registration --

threefry4x32_prng_impl = prng.PRNGImpl(
    key_shape=(4,),
    seed=threefry4x32_seed,
    split=threefry4x32_split,
    random_bits=threefry4x32_random_bits,
    fold_in=threefry4x32_fold_in,
    name="threefry4x32",
    tag="fry4",
)

prng.register_prng(threefry4x32_prng_impl)
