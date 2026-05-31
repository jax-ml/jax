# Copyright 2021 The JAX Authors.
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

from functools import partial
import math

import numpy as np

from jax._src import api
from jax._src import config as config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import ffi
from jax._src import numpy as jnp
from jax._src import typing
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lax import lax
from jax._src.lax import slicing as lax_slicing
from jax._src.random import prng


# -- threefry2x32 PRNG implementation


def _is_threefry_prng_key(key: typing.Array) -> bool:
  try:
    return key.shape == (2,) and key.dtype == np.uint32
  except AttributeError:
    return False


def threefry_seed(seed: typing.Array) -> typing.Array:
  """Create a single raw threefry PRNG key from an integer seed.

  Args:
    seed: a 64- or 32-bit integer used as the value of the key.

  Returns:
    The PRNG key contents, modeled as an array of shape (2,) and dtype
    uint32. The key is constructed from a 64-bit seed by effectively
    bit-casting to a pair of uint32 values (or from a 32-bit seed by
    first padding out with zeros).
  """
  return _threefry_seed(seed)

@api.jit(inline=True)
def _threefry_seed(seed: typing.Array) -> typing.Array:
  if seed.shape:
    raise TypeError(f"PRNG key seed must be a scalar; got {seed!r}.")
  if not np.issubdtype(seed.dtype, np.integer):
    raise TypeError(f"PRNG key seed must be an integer; got {seed!r}")
  convert = lambda k: lax.expand_dims(lax.convert_element_type(k, np.uint32), [0])
  k1 = convert(
      lax.shift_right_logical(seed, lax._const(seed, 32)))
  with config.numpy_dtype_promotion('standard'):
    # TODO(jakevdp): in X64 mode, this can generate 64-bit computations for 32-bit
    # inputs. We should avoid this.
    k2 = convert(jnp.bitwise_and(seed, np.uint32(0xFFFFFFFF)))
  return lax.concatenate([k1, k2], 0)


def _make_rotate_left(dtype):
  if not dtypes.issubdtype(dtype, np.integer):
    raise TypeError("_rotate_left only accepts integer dtypes.")
  nbits = np.array(dtypes.iinfo(dtype).bits, dtype)

  def _rotate_left(x, d):
    if lax.dtype(d) != dtype:
      d = lax.convert_element_type(d, dtype)
    if lax.dtype(x) != dtype:
      x = lax.convert_element_type(x, dtype)
    return lax.shift_left(x, d) | lax.shift_right_logical(x, nbits - d)
  return _rotate_left


### hash function and split

def _threefry2x32_abstract_eval(*args):
  if any(a.dtype != np.uint32 for a in args):
    raise TypeError("Arguments to threefry2x32 must have uint32 type, got {}"
                    .format(args))
  if all(isinstance(arg, core.ShapedArray) for arg in args):
    shape = lax.broadcasting_shape_rule("threefry2x32", *args)
    sharding = lax.broadcasting_sharding_rule("threefry2x32", *args)
    aval = core.ShapedArray(shape, np.dtype('uint32'), sharding=sharding)
  else:
    raise TypeError(f"Arguments to threefry2x32 must all be arrays, got {args}")
  return (aval,) * 2


rotate_left = _make_rotate_left(np.uint32)


def apply_round(v, rot):
  v = v[:]
  v[0] = v[0] + v[1]
  v[1] = rotate_left(v[1], rot)
  v[1] = v[0] ^ v[1]
  return v


def rotate_list(xs):
  return xs[1:] + xs[:1]


def rolled_loop_step(i, state):
  x, ks, rotations = state
  for r in rotations[0]:
    x = apply_round(x, r)
  new_x = [x[0] + ks[0], x[1] + ks[1] + jnp.asarray(i + 1, dtype=np.uint32)]
  return new_x, rotate_list(ks), rotate_list(rotations)


def _threefry2x32_lowering(key1, key2, x1, x2, use_rolled_loops=True):
  """Apply the Threefry 2x32 hash.

  Args:
    keypair: a pair of 32bit unsigned integers used for the key.
    count: an array of dtype uint32 used for the counts.

  Returns:
    An array of dtype uint32 with the same shape as `count`.
  """
  x = [x1, x2]

  rotations = [np.array([13, 15, 26, 6], dtype=np.uint32),
               np.array([17, 29, 16, 24], dtype=np.uint32)]
  ks = [key1, key2, key1 ^ key2 ^ np.uint32(0x1BD11BDA)]

  x[0] = x[0] + ks[0]
  x[1] = x[1] + ks[1]

  if use_rolled_loops:
    x, _, _ = lax_control_flow.fori_loop(
        0, 5, rolled_loop_step, (x, rotate_list(ks), rotations)
    )

  else:
    for r in rotations[0]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[1]
    x[1] = x[1] + ks[2] + np.uint32(1)

    for r in rotations[1]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[2]
    x[1] = x[1] + ks[0] + np.uint32(2)

    for r in rotations[0]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[0]
    x[1] = x[1] + ks[1] + np.uint32(3)

    for r in rotations[1]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[1]
    x[1] = x[1] + ks[2] + np.uint32(4)

    for r in rotations[0]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[2]
    x[1] = x[1] + ks[0] + np.uint32(5)

  return tuple(x)


# Since the unrolled lowering is large, emit it as an out-of-line function.
_threefry2x32_lowering_rule = mlir.lower_fun(
    partial(_threefry2x32_lowering, use_rolled_loops=False),
    multiple_results=True)


def _threefry2x32_gpu_lowering_rule(ctx, k1, k2, x1, x2, *, target_name_prefix):
  if not config.threefry_gpu_kernel_lowering.value:  # back to default lowering
    return _threefry2x32_lowering_rule(ctx, k1, k2, x1, x2)

  aval_out, aval_out_2 = ctx.avals_out
  assert aval_out == aval_out_2
  k1_aval, k2_aval, x1_aval, x2_aval = ctx.avals_in
  rank = len(aval_out.shape)
  if 0 in aval_out.shape:
    zeros = mlir.full_like_aval(ctx, 0, aval_out)
    return [zeros, zeros]
  def _broadcast(x, aval):
    return mlir.broadcast_in_dim(ctx, x, aval_out,
                                 broadcast_dimensions=range(rank - len(aval.shape), rank))

  sub_ctx = ctx.replace(avals_in=(aval_out,) * 4)
  rule = ffi.ffi_lowering(
      f"{target_name_prefix}_threefry2x32_ffi")
  return rule(sub_ctx, _broadcast(k1, k1_aval), _broadcast(k2, k2_aval),
              _broadcast(x1, x1_aval), _broadcast(x2, x2_aval))


threefry2x32_p = core.Primitive("threefry2x32")
threefry2x32_p.multiple_results = True
threefry2x32_p.def_impl(partial(dispatch.apply_primitive, threefry2x32_p))
threefry2x32_p.def_abstract_eval(_threefry2x32_abstract_eval)
batching.defbroadcasting(threefry2x32_p)
mlir.register_lowering(
    threefry2x32_p, _threefry2x32_lowering_rule, inline=False)
mlir.register_lowering(
    threefry2x32_p,
    partial(_threefry2x32_gpu_lowering_rule, target_name_prefix='cu'),
    platform='cuda',
    inline=False)
mlir.register_lowering(
    threefry2x32_p,
    partial(_threefry2x32_gpu_lowering_rule, target_name_prefix='hip'),
    platform='rocm',
    inline=False)


@api.jit(inline=True)
def threefry_2x32(keypair, count):
  """Apply the Threefry 2x32 hash.

  Args:
    keypair: a pair of 32bit unsigned integers used for the key.
    count: an array of dtype uint32 used for the counts.

  Returns:
    An array of dtype uint32 with the same shape as `count`.
  """
  key1, key2 = keypair
  if not lax.dtype(key1) == lax.dtype(key2) == lax.dtype(count) == np.uint32:
    msg = "threefry_2x32 requires uint32 arguments, got {}"
    raise TypeError(msg.format([lax.dtype(x) for x in [key1, key2, count]]))

  flat_count = count.ravel()
  odd_size = flat_count.shape[0] % 2
  if core.is_constant_dim(odd_size):
    if odd_size:
      x = list(jnp.split(jnp.concatenate([flat_count, jnp.uint32([0])]), 2))
    else:
      x = list(jnp.split(flat_count, 2))
  else:
    # With symbolic shapes we cannot always tell statically if odd_size is true
    # or false, so we rewrite this without a conditional.
    flat_count_padded = jnp.concatenate([flat_count, jnp.uint32([0])])
    flat_count_padded_half_size = flat_count_padded.shape[0] // 2
    x = [
      lax_slicing.dynamic_slice(flat_count_padded, (0,),
                                (flat_count_padded_half_size,)),
      lax_slicing.dynamic_slice(flat_count_padded,
                                (flat_count_padded_half_size,),
                                (flat_count_padded_half_size,))
    ]
  assert x[0].shape == x[1].shape, (x[0].shape, x[1].shape)

  x = threefry2x32_p.bind(key1, key2, x[0], x[1])
  out = jnp.concatenate(x)
  assert out.dtype == np.uint32
  if core.is_constant_dim(odd_size):
    return lax.reshape(out[:-1] if odd_size else out, count.shape)
  else:
    out_no_padding = lax_slicing.dynamic_slice(out, (0,), (flat_count.shape[0],))
  return lax.reshape(out_no_padding, count.shape)


def threefry_split(key: typing.Array, shape: prng.Shape) -> typing.Array:
  shape = tuple(map(core.concrete_dim_or_error, shape))
  return _threefry_split(key, shape)

@api.jit(static_argnums=(1,))
def _threefry_split(key, shape) -> typing.Array:
  if core.current_jaxpr_eqn_ctx().threefry_partitionable:
    return _threefry_split_foldlike(key, shape)
  else:
    return _threefry_split_original(key, shape)

@api.jit(static_argnums=(1,), inline=True)
def _threefry_split_original(key, shape) -> typing.Array:
  num = math.prod(shape)
  counts = lax.iota(np.uint32, num * 2)
  return lax.reshape(threefry_2x32(key, counts), (*shape, 2))

@api.jit(static_argnums=(1,), inline=True)
def _threefry_split_foldlike(key, shape) -> typing.Array:
  k1, k2 = key
  counts1, counts2 = prng.iota_2x32_shape(shape)
  bits1, bits2 = threefry2x32_p.bind(k1, k2, counts1, counts2)
  return jnp.stack([bits1, bits2], axis=bits1.ndim)


def threefry_fold_in(key: typing.Array, data: typing.Array) -> typing.Array:
  assert not data.shape
  return _threefry_fold_in(key, jnp.asarray(data, dtype='uint32'))

@api.jit
def _threefry_fold_in(key, data):
  return threefry_2x32(key, threefry_seed(data))


def threefry_random_bits(key: typing.Array, bit_width, shape):
  """Sample uniform random bits of given width and shape using PRNG key."""
  if not _is_threefry_prng_key(key):
    raise TypeError("threefry_random_bits got invalid prng key.")
  if bit_width not in (8, 16, 32, 64):
    raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")

  if core.current_jaxpr_eqn_ctx().threefry_partitionable:
    return _threefry_random_bits_partitionable(key, bit_width, shape)
  else:
    return _threefry_random_bits_original(key, bit_width, shape)

def _threefry_random_bits_partitionable(key: typing.Array, bit_width, shape):
  if all(core.is_constant_dim(d) for d in shape) and math.prod(shape) > 2 ** 64:
    raise NotImplementedError('random bits array of size exceeding 2 ** 64')

  k1, k2 = key
  counts1, counts2 = prng.iota_2x32_shape(shape)
  bits1, bits2 = threefry2x32_p.bind(k1, k2, counts1, counts2)

  dtype = prng.UINT_DTYPES[bit_width]
  if bit_width == 64:
    bits_hi = lax.convert_element_type(bits1, dtype)
    bits_lo = lax.convert_element_type(bits2, dtype)
    return lax.shift_left(bits_hi, jnp.asarray(32, dtype=dtype)) | bits_lo
  elif bit_width == 32:
    return bits1 ^ bits2
  else:
    return lax.convert_element_type(bits1 ^ bits2, dtype)

@api.jit(static_argnums=(1, 2), inline=True)
def _threefry_random_bits_original(key: typing.Array, bit_width, shape):
  size = math.prod(shape)
  # Compute ceil(bit_width * size / 32) in a way that is friendly to shape
  # polymorphism
  max_count, r = divmod(bit_width * size, 32)
  if r > 0:
    max_count += 1

  if core.is_constant_dim(max_count):
    nblocks, rem = divmod(max_count, dtypes.iinfo(np.uint32).max)
  else:
    nblocks, rem = 0, max_count

  if not nblocks:
    bits = threefry_2x32(key, lax.iota(np.uint32, rem))
  else:
    keys = threefry_split(key, (nblocks + 1,))
    subkeys, last_key = keys[:-1], keys[-1]
    blocks = api.vmap(threefry_2x32, in_axes=(0, None))(subkeys, lax.iota(np.uint32, dtypes.iinfo(np.uint32).max))
    last = threefry_2x32(last_key, lax.iota(np.uint32, rem))
    bits = lax.concatenate([blocks.ravel(), last], 0)

  dtype = prng.UINT_DTYPES[bit_width]
  if bit_width == 64:
    bits = [lax.convert_element_type(x, dtype) for x in jnp.split(bits, 2)]
    bits = lax.shift_left(bits[0], jnp.asarray(32, dtype=dtype)) | bits[1]
  elif bit_width in [8, 16]:
    # this is essentially bits.view(dtype)[:size]
    bits = lax.bitwise_and(
      jnp.asarray(np.iinfo(dtype).max, dtype='uint32'),
      lax.shift_right_logical(
        lax.broadcast(bits, (1,)),
        lax.mul(
          np.uint32(bit_width),
          lax.broadcasted_iota(np.uint32, (32 // bit_width, 1), 0)
        )
      )
    )
    bits = lax.reshape(bits, ((max_count * 32 // bit_width),), (1, 0))
    bits = lax.convert_element_type(bits, dtype)[:size]
  return lax.reshape(bits, shape)


threefry_prng_impl = prng.PRNGImpl(
    key_shape=(2,),
    seed=threefry_seed,
    split=threefry_split,
    random_bits=threefry_random_bits,
    fold_in=threefry_fold_in,
    name='threefry2x32',
    tag='fry')

prng.register_prng(threefry_prng_impl)


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
      prng.KeyTy(prng.prngs["threefry2x32"]), ser_flatbuf.DType.key_fry)
