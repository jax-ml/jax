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

"""
Stateful, implicitly-updated PRNG implementation based on mutable refs.
"""
from __future__ import annotations

import dataclasses
import operator
from collections.abc import Sequence

from jax._src import api_util
from jax._src import core
from jax._src import dtypes
from jax._src import numpy as jnp
from jax._src import random
from jax._src import ref
from jax._src import tree_util
from jax._src import typing
from jax._src.state import primitives as ref_primitives
from jax._src.state import types as state_types
from jax._src.typing import Array, ArrayLike, DTypeLike

import numpy as np


def _canonicalize_size(size: int | Sequence[int] | None, *args: ArrayLike) -> tuple[int, ...]:
  if size is None:
    return np.broadcast_shapes(*(np.shape(arg) for arg in args))
  elif isinstance(size, (int, np.number)):
    return (operator.index(size),)
  else:
    return tuple(map(operator.index, size))


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StatefulPRNG:
  """Stateful JAX random generator.

  This should be instantiated using the :func:`jax.experimental.random.stateful_rng` function.

  Attributes:
    _base_key: a typed JAX PRNG key object (see :func:`jax.random.key`).
    _counter: a scalar integer wrapped in a :class:`jax.Ref`.

  Examples:

  >>> from jax.experimental import random
  >>> rng = random.stateful_rng(42)
  >>> rng
  StatefulPRNG(_base_key=Array((), dtype=key<fry>) overlaying:
  [ 0 42], _counter=Ref(0, dtype=int32, weak_type=True))
  """
  _base_key: Array
  _counter: core.Ref

  def __post_init__(self):
    if self._base_key is api_util.SENTINEL:
      return
    if not (isinstance(self._base_key, Array)
            and dtypes.issubdtype(self._base_key.dtype, dtypes.prng_key)):
      raise ValueError(f"Expected base_key to be a typed PRNG key; got {self._base_key}")

    # TODO(jakevdp): how to validate a traced mutable array?
    if not (isinstance(self._counter, core.Ref) or
            (isinstance(self._counter, core.Tracer)
             and isinstance(self._counter.aval, state_types.AbstractRef))):
      raise ValueError(f"Expected counter to be a scalar integer ref; got {self._counter}")

  def key(self, shape: int | Sequence[int] = ()) -> Array:
    """Generate a new JAX PRNGKey, updating the internal state.

    Args:
      shape: an optional shape if returning multiple keys.

    Returns:
      A new, independent PRNG key with the same impl/dtype as
      ``self._base_key``.

    Examples:
      >>> from jax.experimental import random
      >>> rng = random.stateful_rng(0)
      >>> rng.key()
      Array((), dtype=key<fry>) overlaying:
      [1797259609 2579123966]
      >>> rng.key()
      Array((), dtype=key<fry>) overlaying:
      [ 928981903 3453687069]
    """
    if self._base_key.shape:
      # TODO(jakevdp): better error message.
      raise ValueError("cannot operate on split stateful generator")

    key = random.fold_in(self._base_key, ref_primitives.ref_get(self._counter))
    ref_primitives.ref_addupdate(self._counter, ..., 1)  # pytype: disable=wrong-arg-types  # pytype bug?
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

  def split(self, num: int | Sequence[int]) -> StatefulPRNG:
    """Create independent child generators suitable for use in :func:`jax.vmap`.

    Args:
      num: integer or sequence of integers specifying the split shape

    Returns:
      a single StatefulPRNG object with split contents, suitable for use
      with :func:`jax.vmap`

    Examples:
      >>> import jax
      >>> from jax.experimental import random
      >>> rng = random.stateful_rng(123)
      >>> x = jax.numpy.zeros(3)
      >>> def f(rng, x):
      ...   return x + rng.uniform()
      >>> jax.vmap(f)(rng.split(3), x)
      Array([0.35525954, 0.21937883, 0.5336956 ], dtype=float32)

    See also:
      - :meth:`jax.experimental.random.StatefulPRNG.spawn`: This is similar to ``split``, but
        returns a Python list of :class:`StatefulPRNG`` objects.
    """
    return StatefulPRNG(
      _base_key=self.key(num),
      _counter=ref.new_ref(jnp.zeros(num, dtype=int))
    )

  def spawn(self, n_children: int) -> list['StatefulPRNG']:
    """Create a list of independent child generators.

    Args:
      n_children: non-negative integer.

    Returns:
      A list of length ``n_children`` containing new independent ``StatefulPRNG`` instances
      spawned from the original instance.

    Examples:
      >>> from jax.experimental import random
      >>> rng = random.stateful_rng(123)
      >>> child_rngs = rng.spawn(2)
      >>> [r.integers(0, 10, 2) for r in child_rngs]
      [Array([4, 5], dtype=int32), Array([2, 1], dtype=int32)]

    See also:
      - :meth:`jax.experimental.random.StatefulPRNG.split`: this is similar to spawn, but returns
        a single mapped :class:`jax.experimental.random.StatefulPRNG`` which can be passed to
        :func:`jax.vmap`.
    """
    return [self.__class__(key, ref.new_ref(0)) for key in self.key(n_children)]


def stateful_rng(seed: typing.ArrayLike | None = None, *,
                 impl: random.PRNGSpecDesc | None = None) -> StatefulPRNG:
  """
  Experimental stateful RNG with implicitly-updated state.

  This implements a stateful PRNG API similar to :func:`numpy.random.default_rng`.
  It is compatible with JAX transformations like :func:`~jax.jit` and others,
  with a few exceptions mentioned in the Notes below.

  .. note::

    This stateful PRNG API is a convenience wrapper around JAX's classic
    stateless, explicitly updated PRNG, described in :mod:`jax.random`.
    For performance-critical applications, it is recommended to use
    :func:`jax.random.key` with explicit random state semantics.

  For a discussion of design considerations for this API, refer to
  :ref:`stateful-randomness-jep`.

  Args:
    seed: an optional 64- or 32-bit integer used as the value of the key.
      This must be specified if the generator is instantiated within transformed
      code; when used at the top level of the program, it may be omitted in
      which case the RNG will be seeded using the default NumPy seeding.
    impl: optional string specifying the PRNG implementation (e.g.
      ``'threefry2x32'``)

  Returns:
    A :class:`~jax.experimental.random.StatefulPRNG` object, with methods for generating
    random values.

  Notes:
    The :class:`~jax.experimental.random.StatefulPRNG` object created by this method uses
    :func:`~jax.Ref` objects to allow implicit updates of state, and thus
    inherits some of its limitiations. For example:

    - :class:`StatefulPRNG` objects cannot be among the return values of functions
      wrapped in JIT or other JAX transformations. This means in particular
      they cannot be used as `carry` values for :func:`jax.lax.scan`,
      :func:`jax.lax.while_loop`, and other JAX control flow.
    - :class:`StatefulPRNG` objects cannot be used together with
      :func:`jax.checkpoint` or :func:`jax.remat`; in these cases it's best to
      use the :meth:`StatefulPRNG.key` method to produce a standard JAX PRNG key.

  Examples:
    >>> from jax.experimental import random
    >>> rng = random.stateful_rng(42)

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
  if seed is None:
    if not core.trace_ctx.is_top_level():
      raise TypeError(
        "When used within transformed code, jax.experimental.random.stateful_rng()"
        " requires an explicit seed to be set.")
    entropy = np.random.SeedSequence().entropy
    assert isinstance(entropy, int)
    seed = np.int64(entropy & np.iinfo(np.int64).max)  # pyrefly: ignore[no-matching-overload]  # pyrefly#2398
  assert seed is not None
  return StatefulPRNG(
    _base_key=random.key(seed, impl=impl),
    _counter=ref.new_ref(0)
  )
