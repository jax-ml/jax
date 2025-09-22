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

"""
Experimental implicitly-updated PRNG, based on Ref.
"""

import dataclasses
import operator
from typing import Sequence

from jax._src import api_util
from jax._src import core
from jax._src import dtypes
from jax._src import numpy as jnp
from jax._src import random
from jax._src import ref
from jax._src import tree_util
from jax._src import typing
from jax._src.typing import Array, ArrayLike, DTypeLike

import numpy as np


def _canonicalize_size(size: int | Sequence[int] | None, *args: ArrayLike) -> tuple[int, ...]:
  if size is None:
    return np.broadcast_shapes(*(np.shape(arg) for arg in args))
  elif isinstance(size, int):
    return (size,)
  else:
    return tuple(map(operator.index, size))


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StatefulPRNG:
  """Implicit PRNG backed by Ref.

  This should be instantiated using the :func:`default_rng` function.

  Attributes:
    base_key: a typed JAX PRNG key object (see :func:`jax.random.key`).
    counter: a scalar integer wrapped in a :class:`Ref`

  >>> from jax.experimental.stateful_rng import default_rng
  >>> rng = default_rng(42)
  >>> rng
  StatefulPRNG(base_key=Array((), dtype=key<fry>) overlaying:
  [ 0 42], counter=Ref(0, dtype=int32, weak_type=True))
  """
  base_key: Array
  counter: core.Ref

  def __post_init__(self):
    if self.base_key is api_util.SENTINEL:
      return
    if not (isinstance(self.base_key, Array)
            and dtypes.issubdtype(self.base_key.dtype, dtypes.prng_key)):
      raise ValueError(f"Expected base_key to be a typed PRNG key; got {self.base_key}")
    # TODO(jakevdp): how to validate a traced mutable array?
    if not (isinstance(self.counter, (core.Ref, core.Tracer))
            and self.counter.shape == ()
            and dtypes.issubdtype(self.counter.dtype, np.integer)):
      raise ValueError(f"Expected counter to be a mutable scalar integer; got {self.counter}")

  def key(self, shape: int | Sequence[int] = ()) -> Array:
    """Generate a new JAX PRNGKey, updating the internal state.

    Args:
      shape: an optional shape if returning multiple keys.

    Returns:
      A new, independent PRNG key with the same impl/dtype as
      ``self.base_key``.

    Examples:
      >>> from jax.experimental.stateful_rng import default_rng
      >>> rng = default_rng(0)
      >>> rng.key()
      Array((), dtype=key<fry>) overlaying:
      [1797259609 2579123966]
      >>> rng.key()
      Array((), dtype=key<fry>) overlaying:
      [ 928981903 3453687069]
    """
    key = random.fold_in(self.base_key, self.counter[...])
    self.counter[...] += 1
    shape_tuple = _canonicalize_size(shape)
    return random.split(key, shape_tuple) if shape_tuple else key

  def random(
      self,
      size: int | Sequence[int] | None = None,
      dtype: DTypeLike = float,
  ):
    """Return random floats in the half-open interval [0.0, 1.0)."""
    # TODO(jakevdp): write docstring
    return random.uniform(self.key(), shape=_canonicalize_size(size), dtype=dtype)


  def uniform(
      self,
      low: ArrayLike = 0,
      high: ArrayLike = 1,
      size: int | Sequence[int] | None = None,
      *,
      dtype: DTypeLike = float,
  ) -> Array:
    """Draw uniformly distributed pseudorandom values."""
    # TODO(jakevdp): write docstring
    return random.uniform(self.key(), _canonicalize_size(size, low, high),
                          minval=low, maxval=high, dtype=dtype)

  def normal(
      self,
      loc: ArrayLike = 0,
      scale: ArrayLike = 1,
      size: int | Sequence[int] | None = None,
      *,
      dtype: DTypeLike = float,
  ) -> Array:
    """Draw normally-distributed pseudorandom values."""
    # TODO(jakevdp): write docstring
    norm = random.normal(self.key(), _canonicalize_size(size, loc, scale), dtype)
    return (jnp.asarray(loc) + jnp.asarray(scale) * norm).astype(dtype)

  def integers(
      self,
      low: ArrayLike,
      high: ArrayLike | None = None,
      size: int | Sequence[int] | None = None,
      *,
      dtype: DTypeLike = int,
  ) -> Array:
    """Draw pseudorandom integers."""
    # TODO(jakevdp): write docstring
    if high is None:
      low, high = 0, low
    return random.randint(self.key(), _canonicalize_size(size, low, high),
                          minval=low, maxval=high, dtype=dtype)

  def spawn(self, n_children: int) -> list['StatefulPRNG']:
    """Create new independent child generators.

    Args:
      n_children: non-negative integer.

    Returns:
      A list of length ``n_children`` containing new independent ``StatefulPRNG`` instances
      spawned from the original instance.

    Examples:
      >>> from jax.experimental.stateful_rng import default_rng
      >>> rng = default_rng(123)
      >>> child_rngs = rng.spawn(2)
      >>> [crng.integers(0, 10, 2) for crng in child_rngs]
      [Array([1, 3], dtype=int32), Array([9, 9], dtype=int32)]
    """
    return [self.__class__(self.key(), ref.new_ref(0)) for _ in range(n_children)]


def default_rng(seed: typing.ArrayLike, *,
                impl: random.PRNGSpecDesc | None = None) -> StatefulPRNG:
  """
  Implicitly updated PRNG API.

  This implements a stateful PRNG API similar to :func:`numpy.random.default_rng`.
  It is compatible with JAX transformations like :func:`~jax.jit`, :func:`~jax.vmap`,
  and others, with a few exceptions mentioned in the Notes below.

  Args:
    seed: a 64- or 32-bit integer used as the value of the key.
    impl: optional string specifying the PRNG implementation (e.g.
      ``'threefry2x32'``)

  Returns:
    A StatefulPRNG object, with methods for generating random values.

  Notes:
    The StatefulPRNG object created by this method uses

  Examples:
    >>> from jax.experimental.stateful_rng import default_rng
    >>> rng = default_rng(42)

    Repeated draws implicitly update the key:

    >>> rng.uniform()
    Array(0.5302608, dtype=float32)
    >>> rng.uniform()
    Array(0.72766423, dtype=float32)

    This also works under transformations like :func:`jax.jit`:

    >>> import jax
    >>> jit_uniform = jax.jit(rng.uniform)
    >>> jit_uniform()
    Array(0.6672406, dtype=float32)
    >>> jit_uniform()
    Array(0.3890121, dtype=float32)

    Keys can be generated directly if desired:

    >>> rng.key()
    Array((), dtype=key<fry>) overlaying:
    [2954079971 3276725750]
    >>> rng.key()
    Array((), dtype=key<fry>) overlaying:
    [2765691542  824333390]
  """
  return StatefulPRNG(
    base_key=random.key(seed, impl=impl),
    counter=ref.new_ref(0)
  )
