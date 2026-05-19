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

"""Philox 4x32 PRNG implementation for JAX.

This implements the Philox 4x32 counter-based PRNG from the Random123 library
(D.E. Shaw Research) as a JAX PRNGImpl. The key shape is (2,) with uint32 dtype,
matching the Philox 4x32 algorithm's native 2-word (64-bit) key.
Philox uses integer multiplication (mulhi/mullo) and XOR for mixing, with
10 rounds by default.

Usage:
  key = jax.random.key(0, impl='philox4x32')
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

# -- Philox 4x32 constants from Random123 --
# Multiplier constants for the mulhilo step.
_PHILOX_M4x32_0 = np.uint32(0xD2511F53)
_PHILOX_M4x32_1 = np.uint32(0xCD9E8D57)

# Weyl sequence constants for key bumping.
_PHILOX_W32_0 = np.uint32(0x9E3779B9)
_PHILOX_W32_1 = np.uint32(0xBB67AE85)

_DEFAULT_ROUNDS = 10


# -- Helper functions --


# TODO(jakevdp) some platforms have single-instruction 32x32->64bit integer
# multiply. We should use that here – either via a new primitive, or perhaps
# an out_dtype argument on mul.
def _mulhilo32(a, b):
  """Compute full 64-bit product of two uint32 values, return (lo, hi).

  Uses only 32-bit operations (half-width multiply), following the
  _mulhilo_c99_tpl pattern from Random123. This avoids requiring
  jax_enable_x64=True.

  Args:
    a: uint32 array
    b: uint32 array

  Returns:
    Tuple of (low, high) 64-bit result split into two uint32 arrays.
  """
  whalf = np.uint32(16)
  lomask = np.uint32(0xFFFF)

  # Split inputs into 16-bit halves.
  ahi = lax.shift_right_logical(a, whalf)
  alo = a & lomask
  bhi = lax.shift_right_logical(b, whalf)
  blo = b & lomask

  # Low 32 bits of the product (modular uint32 multiply gives this directly).
  lo = a * b

  # Cross products (each fits in uint32 since both operands are ≤ 0xFFFF).
  ahbl = ahi * blo
  albh = alo * bhi

  # Sum of the lower halves of the cross products.
  ahbl_albh = (ahbl & lomask) + (albh & lomask)

  # High 32 bits: ahi*bhi plus upper halves of cross products plus carries.
  hi = (
      ahi * bhi
      + lax.shift_right_logical(ahbl, whalf)
      + lax.shift_right_logical(albh, whalf)
      + lax.shift_right_logical(ahbl_albh, whalf)
  )
  # Carry from the addition of lo's upper half with ahbl_albh's lower half.
  lo_upper = lax.shift_right_logical(lo, whalf)
  carry = lax.convert_element_type(lo_upper < (ahbl_albh & lomask), np.uint32)
  hi = hi + carry

  return lo, hi


# -- Core Philox 4x32 hash --


def _philox4x32_lowering(k0, k1, x0, x1, x2, x3):
  """Apply the Philox 4x32 hash with 10 rounds.

  Args:
    k0: uint32 array, the upper 32 bits of the 64-bit Philox key.
    k1: uint32 array, the lower 32 bits of the 64-bit Philox key.
    x0: uint32 array, the first word of the counter.
    x1: uint32 array, the second word of the counter.
    x2: uint32 array, the third word of the counter.
    x3: uint32 array, the fourth word of the counter.

  Returns:
    A tuple of four uint32 arrays (out0, out1, out2, out3).
  """
  for rnd in range(_DEFAULT_ROUNDS):
    # Bump key before each round except the first.
    if rnd > 0:
      k0 = k0 + _PHILOX_W32_0
      k1 = k1 + _PHILOX_W32_1

    # Philox round function:
    #   lo0, hi0 = mulhilo(M0, x0)
    #   lo1, hi1 = mulhilo(M1, x2)
    #   out = [hi1 ^ x1 ^ k0, lo1, hi0 ^ x3 ^ k1, lo0]
    lo0, hi0 = _mulhilo32(_PHILOX_M4x32_0, x0)
    lo1, hi1 = _mulhilo32(_PHILOX_M4x32_1, x2)

    x0_new = hi1 ^ x1 ^ k0
    x1_new = lo1
    x2_new = hi0 ^ x3 ^ k1
    x3_new = lo0

    x0, x1, x2, x3 = x0_new, x1_new, x2_new, x3_new

  return (x0, x1, x2, x3)


# -- Primitive definition --


def _philox4x32_abstract_eval(*args):
  """Abstract evaluation rule for philox4x32_p."""
  if len(args) != 6:
    raise TypeError(f"philox4x32_p expects 6 arguments, got {len(args)}.")
  if all(isinstance(arg, core.ShapedArray) for arg in args):
    shape = lax.broadcasting_shape_rule("philox4x32", *args)
    sharding = lax.broadcasting_sharding_rule("philox4x32", *args)
    aval = core.ShapedArray(shape, np.dtype("uint32"), sharding=sharding)
  else:
    raise TypeError(f"Arguments to philox4x32 must all be arrays, got {args}")
  if any(a.dtype != np.uint32 for a in args):
    raise TypeError(
        f"Arguments to philox4x32 must have uint32 type, got {args}"
    )
  return (aval,) * 4


philox4x32_p = core.Primitive("philox4x32")
philox4x32_p.multiple_results = True
philox4x32_p.def_impl(functools.partial(dispatch.apply_primitive, philox4x32_p))
philox4x32_p.def_abstract_eval(_philox4x32_abstract_eval)
batching.defbroadcasting(philox4x32_p)

_philox4x32_lowering_rule = mlir.lower_fun(
    _philox4x32_lowering, multiple_results=True
)

# TODO(jakevdp): we could potentially speed this up with device-specific
#   lowerings that take advantage of native 64-bit instructions.
mlir.register_lowering(philox4x32_p, _philox4x32_lowering_rule, inline=False)


# -- PRNGImpl functions --


def _is_philox4x32_key(key: typing.Array) -> bool:
  """Return True if key is a Philox 4x32 key."""
  try:
    return key.shape == (2,) and key.dtype == np.uint32
  except AttributeError:
    return False


def philox4x32_seed(seed: typing.Array) -> typing.Array:
  """Create a single Philox 4x32 PRNG key from an integer seed."""
  return _philox4x32_seed(seed)


@api.jit(inline=True)
def _philox4x32_seed(seed: typing.Array) -> typing.Array:
  """Internal implementation of philox4x32_seed."""
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
  # Hash through philox4x32 to mix the seed bits into a 2-word key.
  # Use the seed halves as counter words so both influence the output.
  out = philox4x32_p.bind(
      np.uint32(0),
      np.uint32(0),
      k0,
      k1,
      np.uint32(0),
      np.uint32(0),
  )
  return jnp.array([out[0], out[1]], dtype=np.uint32)


def philox4x32_split(key: typing.Array, shape: prng.Shape) -> typing.Array:
  """Split a Philox 4x32 key into multiple sub-keys."""
  shape = tuple(map(core.concrete_dim_or_error, shape))
  return _philox4x32_split(key, shape)


@api.jit(static_argnums=(1,), inline=True)
def _philox4x32_split(key: typing.Array, shape: prng.Shape) -> typing.Array:
  """Internal implementation of philox4x32_split."""
  k0, k1 = key[0], key[1]

  # Generate counters for each sub-key.
  counts1, counts2 = prng.iota_2x32_shape(shape)
  zeros = jnp.zeros(shape, dtype=np.uint32)

  out0, out1, _, _ = philox4x32_p.bind(k0, k1, zeros, zeros, counts1, counts2)

  return jnp.stack([out0, out1], axis=len(shape))


def philox4x32_fold_in(key: typing.Array, data: typing.Array) -> typing.Array:
  """Fold-in an integer value to create a new Philox4x32 key."""
  assert not data.shape
  return _philox4x32_fold_in(key, jnp.asarray(data, dtype="uint32"))


@api.jit
def _philox4x32_fold_in(key: typing.Array, data: typing.Array) -> typing.Array:
  """Internal implementation of philox4x32_fold_in."""
  # Hash the key with the data used as part of the counter.
  k0, k1 = key[0], key[1]
  out0, out1, _, _ = philox4x32_p.bind(
      k0, k1, np.uint32(0), np.uint32(0), np.uint32(0), data
  )
  return jnp.array([out0, out1], dtype=np.uint32)


def philox4x32_random_bits(
    key: typing.Array, bit_width: int, shape: tuple[int, ...]
) -> typing.Array:
  """Sample uniform random bits using a Philox 4x32 key."""
  if not _is_philox4x32_key(key):
    raise TypeError("philox4x32_random_bits got invalid prng key.")
  if bit_width not in (8, 16, 32, 64):
    raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")
  return _philox4x32_random_bits(key, bit_width, shape)


@api.jit(static_argnums=(1, 2), inline=True)
def _philox4x32_random_bits(
    key: typing.Array, bit_width: int, shape: tuple[int, ...]
) -> typing.Array:
  """Internal implementation of philox4x32_random_bits."""
  if all(core.is_constant_dim(d) for d in shape) and math.prod(shape) > 2**64:
    raise NotImplementedError("random bits array of size exceeding 2 ** 64")

  k0, k1 = key[0], key[1]
  counts1, counts2 = prng.iota_2x32_shape(shape)
  zeros = jnp.zeros(shape, dtype=np.uint32)

  out0, out1, out2, out3 = philox4x32_p.bind(
      k0, k1, counts1, counts2, zeros, zeros
  )

  dtype = prng.UINT_DTYPES[bit_width]
  if bit_width == 64:
    # Combine two 32-bit outputs into one 64-bit value.
    bits_hi = lax.convert_element_type(out0, dtype)
    bits_lo = lax.convert_element_type(out1, dtype)
    return lax.shift_left(bits_hi, jnp.asarray(32, dtype=dtype)) | bits_lo
  elif bit_width == 32:
    # XOR all four outputs for maximum mixing.
    return out0 ^ out1 ^ out2 ^ out3
  else:
    return lax.convert_element_type(out0 ^ out1 ^ out2 ^ out3, dtype)


# -- PRNGImpl registration --

philox4x32_prng_impl = prng.PRNGImpl(
    key_shape=(2,),
    seed=philox4x32_seed,
    split=philox4x32_split,
    random_bits=philox4x32_random_bits,
    fold_in=philox4x32_fold_in,
    name="philox4x32",
    tag="phx4",
)

prng.register_prng(philox4x32_prng_impl)
