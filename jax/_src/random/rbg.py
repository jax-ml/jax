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

from __future__ import annotations

from collections.abc import Sequence
import math
from jax._src import typing

import numpy as np

from jax._src import api
from jax._src import core
from jax._src import numpy as jnp
from jax._src.lax import lax
from jax._src.lax import slicing as lax_slicing
from jax._src.random import prng
from jax._src.random import threefry2x32

# -- RngBitGenerator PRNG implementation

# This code is experimental!
# https://www.openxla.org/xla/operation_semantics#rngbitgenerator
# Notice that the RngBitGenerator operations are not guaranteed to be
# stable/deterministic across backends or compiler versions. Correspondingly, we
# reserve the right to change any of these implementations at any time!

def _rbg_seed(seed: typing.Array) -> typing.Array:
  assert not seed.shape
  halfkey = threefry2x32.threefry_seed(seed)
  return jnp.concatenate([halfkey, halfkey])

def _rbg_split(key: typing.Array, shape: prng.Shape) -> typing.Array:
  if core.current_jaxpr_eqn_ctx().threefry_partitionable:
    _threefry_split = threefry2x32._threefry_split_foldlike
  else:
    _threefry_split = threefry2x32._threefry_split_original
  halfkeys = key.reshape(2, 2)
  return api.vmap(
      _threefry_split, (0, None), len(shape))(halfkeys, shape).reshape(
          *shape, 4)

def _rbg_fold_in(key: typing.Array, data: typing.Array) -> typing.Array:
  assert not data.shape
  return api.vmap(threefry2x32._threefry_fold_in, (0, None), 0)(key.reshape(2, 2), data).reshape(4)

def _rbg_random_bits(key: typing.Array, bit_width: int, shape: Sequence[int]
                     ) -> typing.Array:
  if not key.shape == (4,) and key.dtype == np.dtype('uint32'):
    raise TypeError("_rbg_random_bits got invalid prng key.")
  if bit_width not in (8, 16, 32, 64):
    raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")
  _, bits = lax.rng_bit_generator(key, shape, dtype=prng.UINT_DTYPES[bit_width])
  return bits

rbg_prng_impl = prng.PRNGImpl(
    key_shape=(4,),
    seed=_rbg_seed,
    split=_rbg_split,
    random_bits=_rbg_random_bits,
    fold_in=_rbg_fold_in,
    name='rbg',
    tag='rbg')

prng.register_prng(rbg_prng_impl)


def _unsafe_rbg_split(key: typing.Array, shape: prng.Shape) -> typing.Array:
  # treat 10 iterations of random bits as a 'hash function'
  num = math.prod(shape)
  _, keys = lax.rng_bit_generator(key, (10 * num, 4), dtype='uint32')
  return lax_slicing.slice_in_dim(
      keys, start_index=None, limit_index=None, stride=10).reshape(*shape, 4)

def _unsafe_rbg_fold_in(key: typing.Array, data: typing.Array) -> typing.Array:
  assert not data.shape
  _, random_bits = lax.rng_bit_generator(_rbg_seed(data), (10, 4), dtype='uint32')
  return key ^ random_bits[-1]

unsafe_rbg_prng_impl = prng.PRNGImpl(
    key_shape=(4,),
    seed=_rbg_seed,
    split=_unsafe_rbg_split,
    random_bits=_rbg_random_bits,
    fold_in=_unsafe_rbg_fold_in,
    name='unsafe_rbg',
    tag='urbg')

prng.register_prng(unsafe_rbg_prng_impl)


# Register export serialization for PRNG key types.
try:
  from jax._src.export import serialization
  from jax._src.export import serialization_generated as ser_flatbuf
except ImportError:
  # This can happen if flatbuffers is not installed, in which case export
  # serialization is not supported and it is safe to skip the registration.
  pass
else:
  serialization.register_dtype_kind(
      prng.KeyTy(prng.prngs["rbg"]), ser_flatbuf.DType.key_rbg)
  serialization.register_dtype_kind(
      prng.KeyTy(prng.prngs["unsafe_rbg"]), ser_flatbuf.DType.key_unsafe_rbg)
