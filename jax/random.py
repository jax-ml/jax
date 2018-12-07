# Copyright 2018 Google LLC
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

"""LAX-based pseudo-random number generators (PRNGs)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as onp

from . import lax
from . import numpy as np
from . import tree_util
from .api import jit
from jax.lib import xla_bridge


class PRNGKey(object):
  """A pseudo-random number generator (PRNG) key for use with lax.random."""
  __slots__ = ["keypair"]

  def __init__(self, seed):
    """Create a new PRNG key.

    Args:
      seed: a scalar integer value used to initialize the PRNG key.

    Returns:
      A new PRNGKey object.
    """
    convert = lambda key: lax.convert_element_type(key, onp.uint32)
    if onp.shape(seed):
      raise TypeError("PRNGKey seed must be a scalar.")
    if isinstance(seed, (int, onp.ndarray)):
      # Special handling of raw integer values, which may have be 64bit even
      # when jax_enable_x64=False and we don't want to drop the top 32 bits
      k1 = convert(onp.bitwise_and(onp.right_shift(seed, 32), 0xFFFFFFFF))
    else:
      k1 = convert(lax.shift_right_logical(seed, 32))
    k2 = convert(lax.bitwise_and(seed, 0xFFFFFFFF))
    self.keypair = (k1, k2)

  @classmethod
  def from_keypair(cls, keypair):
    """Internal method to create a PRNGKey instance from a raw key pair."""
    new = cls.__new__(cls)
    new.keypair = tuple(keypair)
    return new


tree_util.register_pytree_node(PRNGKey, lambda k: (k.keypair, None),
                               lambda _, xs: PRNGKey.from_keypair(xs))


### utilities


def _make_rotate_left(dtype):
  if not onp.issubdtype(dtype, onp.integer):
    raise TypeError("_rotate_left only accepts integer dtypes.")
  nbits = onp.array(onp.iinfo(dtype).bits, dtype)

  def _rotate_left(x, d):
    if lax._dtype(d) != lax._dtype(x):
      d = lax.convert_element_type(d, x.dtype)
    return (x << d) | lax.shift_right_logical(x, nbits - d)
  return _rotate_left


def _bit_stats(bits):
  """This is a debugging function to compute the statistics of bit fields."""
  return onp.array([list(map(int, onp.binary_repr(x, 64))) for x in bits]).mean(0)


### hash function and split


@jit
def threefry_2x32(keypair, count):
  """Apply the Threefry 2x32 hash.

  Args:
    keypair: a pair of 32bit unsigned integers used for the key.
    count: an array of dtype uint32 used for the counts.

  Returns:
    An array of dtype uint32 with the same shape as `count`.
  """
  # Based on ThreeFry2x32 by phawkins@ in //.../xla/client/lib/prng.cc
  key1, key2 = keypair[0], keypair[1]
  if not lax._dtype(key1) == lax._dtype(key2) == lax._dtype(count) == onp.uint32:
    msg = "threefry_2x32 requires uint32 arguments, got {}"
    raise TypeError(msg.format([lax._dtype(x) for x in [key1, key2, count]]))

  rotate_left = _make_rotate_left(lax._dtype(count))

  def apply_round(v, rot):
    v = v[:]
    v[0] = v[0] + v[1]
    v[1] = rotate_left(v[1], rot)
    v[1] = v[0] ^ v[1]
    return v

  odd_size = count.size % 2
  if odd_size:
    x = list(np.split(np.concatenate([count.ravel(), onp.uint32([0])]), 2))
  else:
    x = list(np.split(count.ravel(), 2))

  rotations = [13, 15, 26, 6, 17, 29, 16, 24]
  ks = [key1, key2, key1 ^ key2 ^ onp.uint32(0x1BD11BDA)]

  x[0] = x[0] + ks[0]
  x[1] = x[1] + ks[1]

  for r in rotations[:4]:
    x = apply_round(x, r)
  x[0] = x[0] + ks[1]
  x[1] = x[1] + ks[2] + onp.uint32(1)

  for r in rotations[4:]:
    x = apply_round(x, r)
  x[0] = x[0] + ks[2]
  x[1] = x[1] + ks[0] + onp.uint32(2)

  for r in rotations[:4]:
    x = apply_round(x, r)
  x[0] = x[0] + ks[0]
  x[1] = x[1] + ks[1] + onp.uint32(3)

  for r in rotations[4:]:
    x = apply_round(x, r)
  x[0] = x[0] + ks[1]
  x[1] = x[1] + ks[2] + onp.uint32(4)

  for r in rotations[:4]:
    x = apply_round(x, r)
  x[0] = x[0] + ks[2]
  x[1] = x[1] + ks[0] + onp.uint32(5)

  out = np.concatenate(x)
  assert out.dtype == onp.uint32
  return lax.reshape(out[:-1] if odd_size else out, count.shape)


@partial(jit, static_argnums=(1,))
def split(key, num=2):
  """Splits a PRNG key pair of 32bit unsigned integers into `num` new key pairs.

  Args:
    key: a PRNGKey used as the random key.
    num: optional, a positive integer indicating the number of keys to produce
      (default 2).

  Returns:
    A tuple of length `num` of new PRNGKey instances.
  """
  counts = onp.arange(num * 2, dtype=onp.uint32)
  bits = lax.reshape(threefry_2x32(key.keypair, counts), (num, 2))
  keypairs = (lax.index_in_dim(bits, i, keepdims=False) for i in range(num))
  return tuple(PRNGKey.from_keypair((kp[0], kp[1])) for kp in keypairs)


def _random_bits(key, bit_width, shape):
  """Sample uniform random bits of given width and shape using PRNG key."""
  if bit_width not in (32, 64):
    raise TypeError("requires 32- or 64-bit field width.")
  max_count = (bit_width // 32) * onp.prod(shape)
  if max_count >= onp.iinfo(onp.uint32).max:
    # TODO(mattjj): just split the key here
    raise TypeError("requesting more random bits than a single call provides.")

  bits = threefry_2x32(key.keypair, onp.arange(max_count, dtype=onp.uint32))
  if bit_width == 64:
    bits = [lax.convert_element_type(x, onp.uint64) for x in np.split(bits, 2)]
    bits = (bits[0] << onp.uint64(32)) | bits[1]
  return lax.reshape(bits, shape)


### random samplers


@partial(jit, static_argnums=(1, 2))
def uniform(key, shape, dtype=onp.float32, minval=0., maxval=1.):
  """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: a tuple of nonnegative integers representing the shape.
    dtype: optional, a float dtype for the returned values (default float32).
    minval: optional, a minimum (inclusive) value for the range (default 0).
    maxval: optional, a maximum (exclusive) value for the range (default 1).

  Returns:
    A random array with the specified shape and dtype.
  """
  if not onp.issubdtype(dtype, onp.floating):
    raise TypeError("uniform only accepts floating point dtypes.")

  dtype = xla_bridge.canonicalize_dtype(dtype)
  minval = lax.convert_element_type(minval, dtype)
  maxval = lax.convert_element_type(maxval, dtype)
  finfo = onp.finfo(dtype)
  nbits, nmant = finfo.bits, finfo.nmant

  if nbits not in (32, 64):
    raise TypeError("uniform only accepts 32- or 64-bit dtypes.")

  bits = _random_bits(key, nbits, shape)

  # The strategy here is to randomize only the mantissa bits with an exponent of
  # 1 (after applying the bias), then shift and scale to the desired range. The
  # bit-level transformation we use relies on Numpy and XLA having bit-for-bit
  # equivalent float representations, which might not be true on all platforms.
  float_bits = lax.bitwise_or(
      lax.shift_right_logical(bits, onp.array(nbits - nmant, lax._dtype(bits))),
      onp.array(1., dtype).view(onp.uint32 if nbits == 32 else onp.uint64))
  floats = lax.bitcast_convert_type(float_bits, dtype) - onp.array(1., dtype)
  return lax.max(
      minval,
      lax.reshape(floats * (maxval - minval) + minval, shape))


@partial(jit, static_argnums=(1, 4))
def randint(key, shape, minval, maxval, dtype=onp.int32):
  """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: a tuple of nonnegative integers representing the shape.
    minval: optional, a minimum (inclusive) value for the range (default 0).
    maxval: optional, a maximum (exclusive) value for the range (default 1).
    dtype: optional, an int dtype for the returned values (default int32).

  Returns:
    A random array with the specified shape and dtype.
  """
  if not onp.issubdtype(dtype, onp.integer):
    raise TypeError("randint only accepts integer dtypes.")

  dtype = xla_bridge.canonicalize_dtype(dtype)
  minval = lax.convert_element_type(minval, dtype)
  maxval = lax.convert_element_type(maxval, dtype)
  nbits = onp.iinfo(dtype).bits

  if nbits not in (32, 64):
    raise TypeError("randint only accepts 32- or 64-bit dtypes.")

  # This algorithm is biased whenever (maxval - minval) is not a power of 2.
  # We generate double the number of random bits required by the dtype so as to
  # reduce that bias.
  k1, k2 = split(key)
  rbits = lambda key: _random_bits(key, nbits, shape)
  higher_bits, lower_bits = rbits(k1), rbits(k2)

  unsigned_dtype = onp.uint32 if nbits == 32 else onp.uint64
  span = lax.convert_element_type(maxval - minval, unsigned_dtype)

  # To compute a remainder operation on an integer that might have twice as many
  # bits as we can represent in the native unsigned dtype, we compute a
  # multiplier equal to 2**nbits % span (using that nbits is 32 or 64).
  multiplier = lax.rem(onp.array(2**16, unsigned_dtype), span)
  multiplier = lax.rem(lax.mul(multiplier, multiplier), span)
  if nbits == 64:
    multiplier = lax.rem(lax.mul(multiplier, multiplier), span)

  random_offset = lax.add(lax.mul(lax.rem(higher_bits, span), multiplier),
                          lax.rem(lower_bits, span))
  random_offset = lax.rem(random_offset, span)
  return lax.add(minval, lax.convert_element_type(random_offset, dtype))


@partial(jit, static_argnums=(2,))
def shuffle(key, x, axis=0):
  """Shuffle the elements of an array uniformly at random along an axis.

  Args:
    key: a PRNGKey used as the random key.
    x: the array to be shuffled.
    axis: optional, an int axis along which to shuffle (default 0).

  Returns:
    A shuffled version of x.
  """
  # On parallel architectures, Fisher-Yates is more expensive than doing
  # multiple sorts. This algorithm is based on one developed and analyzed by
  # tjablin@. We sort according to randomly-generated 32bit keys, but those keys
  # may have collisions. If we repeat the process, using fresh 32bit keys for
  # each sort, then whenever all pairs of elements have been assigned distinct
  # keys at some iteration (or equivalently when the strings formed by
  # concatenating the successive keys for each element are all distinct) then we
  # are guaranteed to have a perfect sample (assuming that either the sort is
  # stable or that any bias is not value-dependent). Since checking uniqueness
  # at runtime may be expensive, we use a heuristic static stop criterion
  # developed by tjablin@. See tensorflow/compiler/tf2xla/random_ops.cc for more
  # info, and for the original implementation of this algorithm. See also
  # Section 2 of http://people.csail.mit.edu/costis/6896sp11/lec5s.pdf for
  # another analysis (where the keys are generated one bit at a time).
  exponent = 3  # see tjablin@'s analysis for explanation of this parameter
  num_rounds = int(onp.ceil(exponent * onp.log(len(x)) / 32))

  for _ in range(num_rounds):
    key, subkey = split(key)
    sort_keys = _random_bits(subkey, 32, x.shape)
    _, x = lax.sort_key_val(sort_keys, x, axis)

  return x


@partial(jit, static_argnums=(1, 2))
def normal(key, shape, dtype=onp.float32):
  """Sample standard normal random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: a tuple of nonnegative integers representing the shape.
    dtype: optional, a float dtype for the returned values (default float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  lo = onp.nextafter(onp.array(-1., dtype), 0., dtype=dtype)
  hi = onp.array(1., dtype)
  u = uniform(key, shape, dtype, lo, hi)
  return onp.array(onp.sqrt(2), dtype) * lax.erf_inv(u)


@partial(jit, static_argnums=(2,))
def bernoulli(key, mean=onp.float32(0.5), shape=()):
  """Sample Bernoulli random values with given shape and mean.

  Args:
    key: a PRNGKey used as the random key.
    mean: optional, an array-like broadcastable to `shape` for the mean of the
      random variables (default 0.5).
    shape: optional, a tuple of nonnegative integers representing the shape
      (default scalar).

  Returns:
    A random array with the specified shape and boolean dtype.
  """
  shape = shape or onp.shape(mean)
  if not onp.issubdtype(lax._dtype(mean), onp.float32):
    mean = lax.convert_element_type(mean, onp.float32)
  if onp.shape(mean) != shape:
    mean = lax.broadcast(mean, shape)
  return lax.lt(uniform(key, shape), mean)
