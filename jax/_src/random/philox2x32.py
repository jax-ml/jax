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

"""Philox 2x32 PRNG implementation for JAX.

This implements the Philox 2x32 counter-based PRNG from the Random123 library
(D.E. Shaw Research) as a JAX PRNGImpl. The key shape is (1,) with uint32 dtype,
matching the Philox 2x32 algorithm's native 1-word (32-bit) key.
Philox 2x32 uses a single mulhilo32 per round with a 1-word key and 2-word
counter, applying 10 rounds by default.

Usage:
  key = jax.random.key(0, impl='philox2x32')
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

# -- Philox 2x32 constants from Random123 --
# Multiplier constant for the mulhilo step.
_PHILOX_M2x32_0 = np.uint32(0xD256D193)

# Weyl sequence constant for key bumping.
_PHILOX_W32_0 = np.uint32(0x9E3779B9)

_DEFAULT_ROUNDS = 10


# -- Core Philox 2x32 hash --


def _philox2x32_lowering(k0, x0, x1):
  """Apply the Philox 2x32 hash with 10 rounds.

  Args:
    k0: uint32 array, the Philox 2x32 key word.
    x0: uint32 array containing the upper 32 bits of the counter.
    x1: uint32 array containing the lower 32 bits of the counter.

  Returns:
    A tuple of two uint32 arrays (out0, out1).
  """
  for rnd in range(_DEFAULT_ROUNDS):
    # Bump key before each round except the first.
    if rnd > 0:
      k0 = k0 + _PHILOX_W32_0

    # Philox 2x32 round function:
    #   lo, hi = mulhilo(M2x32_0, x0)
    #   out = [hi ^ x1 ^ k0, lo]
    lo, hi = lax.mul(_PHILOX_M2x32_0, x0), lax.mulhi(_PHILOX_M2x32_0, x0)

    x0 = hi ^ x1 ^ k0
    x1 = lo

  return (x0, x1)


# -- Primitive definition --


def _philox2x32_abstract_eval(*args):
  """Abstract evaluation rule for philox2x32_p."""
  if len(args) != 3:
    raise TypeError(f"philox2x32_p expects 3 arguments, got {len(args)}.")
  if all(isinstance(arg, core.ShapedArray) for arg in args):
    shape = lax.broadcasting_shape_rule("philox2x32", *args)
    sharding = lax.broadcasting_sharding_rule("philox2x32", *args)
    aval = core.ShapedArray(shape, np.dtype("uint32"), sharding=sharding)
  else:
    raise TypeError(f"Arguments to philox2x32 must all be arrays, got {args}")
  if any(a.dtype != np.uint32 for a in args):
    raise TypeError(
        f"Arguments to philox2x32 must have uint32 type, got {args}"
    )
  return (aval,) * 2


philox2x32_p = core.Primitive("philox2x32")
philox2x32_p.multiple_results = True
philox2x32_p.def_impl(functools.partial(dispatch.apply_primitive, philox2x32_p))
philox2x32_p.def_abstract_eval(_philox2x32_abstract_eval)
batching.defbroadcasting(philox2x32_p)

_philox2x32_lowering_rule = mlir.lower_fun(
    _philox2x32_lowering, multiple_results=True
)

# TODO(jakevdp): we could potentially speed this up with device-specific
#   lowerings that take advantage of native 64-bit instructions.
mlir.register_lowering(philox2x32_p, _philox2x32_lowering_rule, inline=False)


# -- PRNGImpl functions --


def _is_philox2x32_key(key: typing.Array) -> bool:
  """Return True if the input is a valid Philox 2x32 PRNG key."""
  try:
    return key.shape == (1,) and key.dtype == np.uint32
  except AttributeError:
    return False


def philox2x32_seed(seed: typing.Array) -> typing.Array:
  """Create a single Philox 2x32 PRNG key from an integer seed."""
  return _philox2x32_seed(seed)


@api.jit(inline=True)
def _philox2x32_seed(seed: typing.Array) -> typing.Array:
  """Internal implementation of philox2x32_seed."""
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
  # Hash through philox2x32 to mix the seed bits into a 1-word key.
  # Use both seed halves as counter words so they both influence the output.
  out0, _ = philox2x32_p.bind(np.uint32(0), k0, k1)
  return jnp.array([out0], dtype=np.uint32)


def philox2x32_split(key: typing.Array, shape: prng.Shape) -> typing.Array:
  """Split a Philox 2x32 PRNG key into multiple sub-keys."""
  shape = tuple(map(core.concrete_dim_or_error, shape))
  return _philox2x32_split(key, shape)


@api.jit(static_argnums=(1,), inline=True)
def _philox2x32_split(key: typing.Array, shape: prng.Shape) -> typing.Array:
  """Internal implementation of philox2x32_split."""
  counts1, counts2 = prng.iota_2x32_shape(shape)
  out0, _ = philox2x32_p.bind(key[0], counts1, counts2)
  return lax.expand_dims(out0, [len(shape)])


def philox2x32_fold_in(key: typing.Array, data: typing.Array) -> typing.Array:
  """Fold-in an integer value to create a new Philox2x32 key."""
  assert not data.shape
  return _philox2x32_fold_in(key, jnp.asarray(data, dtype="uint32"))


@api.jit
def _philox2x32_fold_in(key: typing.Array, data: typing.Array) -> typing.Array:
  """Internal implementation of philox2x32_fold_in."""
  out0, _ = philox2x32_p.bind(key[0], np.uint32(0), data)
  return jnp.array([out0], dtype=np.uint32)


def philox2x32_random_bits(
    key: typing.Array, bit_width: int, shape: tuple[int, ...]
) -> typing.Array:
  """Sample uniform random bits using a Philox 2x32 key."""
  if not _is_philox2x32_key(key):
    raise TypeError("philox2x32_random_bits got invalid prng key.")
  if bit_width not in (8, 16, 32, 64):
    raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")
  return _philox2x32_random_bits(key, bit_width, shape)


@api.jit(static_argnums=(1, 2), inline=True)
def _philox2x32_random_bits(
    key: typing.Array, bit_width: int, shape: tuple[int, ...]
) -> typing.Array:
  """Internal implementation of philox2x32_random_bits."""
  if all(core.is_constant_dim(d) for d in shape) and math.prod(shape) > 2**64:
    raise NotImplementedError("random bits array of size exceeding 2 ** 64")

  counts1, counts2 = prng.iota_2x32_shape(shape)
  out0, out1 = philox2x32_p.bind(key[0], counts1, counts2)

  dtype = prng.UINT_DTYPES[bit_width]
  if bit_width == 64:
    bits_hi = lax.convert_element_type(out0, dtype)
    bits_lo = lax.convert_element_type(out1, dtype)
    return lax.shift_left(bits_hi, jnp.asarray(32, dtype=dtype)) | bits_lo
  elif bit_width == 32:
    return out0 ^ out1
  else:
    return lax.convert_element_type(out0 ^ out1, dtype)


# -- PRNGImpl registration --

philox2x32_prng_impl = prng.PRNGImpl(
    key_shape=(1,),
    seed=philox2x32_seed,
    split=philox2x32_split,
    random_bits=philox2x32_random_bits,
    fold_in=philox2x32_fold_in,
    name="philox2x32",
    tag="phx2",
)

prng.register_prng(philox2x32_prng_impl)
