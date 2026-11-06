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

from jax._src import core
from jax._src import dtypes
from jax._src import lax
from jax._src.numpy import lax_numpy
from jax._src.numpy import ufuncs
from jax._src.typing import Array, ArrayLike
from jax._src.util import set_module

export = set_module('jax.numpy')


def _mirrored_iota(dtype, M: int) -> Array:
  """Returns each index's distance from the nearer end of a length-`M` array.

  For example, `M = 5` gives `[0, 1, 2, 1, 0]` and `M = 4` gives `[0, 1, 1, 0]`.

  The cosine windows below are symmetric: the value at index `n` equals the
  value at index `M - 1 - n`. Computing them from `lax.iota` directly breaks
  that symmetry slightly, because `n / (M - 1)` and `(M - 1 - n) / (M - 1)`
  round differently. Using this mirrored index instead gives both halves of the
  window identical inputs, so the output is exactly symmetric.
  """
  n = lax.iota(dtype, M)
  return lax.min(n, (M - 1) - n)


@export
def blackman(M: int) -> Array:
  """Return a Blackman window of size M.

  JAX implementation of :func:`numpy.blackman`.

  Args:
    M: The window size.

  Returns:
    An array of size M containing the Blackman window.

  Examples:
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.blackman(4))
    [0.   0.63 0.63 0.  ]

  See also:
    - :func:`jax.numpy.bartlett`: return a Bartlett window of size M.
    - :func:`jax.numpy.hamming`: return a Hamming window of size M.
    - :func:`jax.numpy.hanning`: return a Hanning window of size M.
    - :func:`jax.numpy.kaiser`: return a Kaiser window of size M.
  """
  M = core.concrete_or_error(int, M, "M argument of jnp.blackman")
  dtype = dtypes.default_float_dtype()
  if M <= 1:
    return lax.full((M,), 1, dtype)
  n = _mirrored_iota(dtype, M)
  # Adding 0.42 and 0.08 first makes their rounding errors cancel: in both
  # float32 and float64 the sum rounds to exactly 0.5, so the endpoints are
  # exactly 0 and odd-size midpoints are exactly 1.
  return (0.42 + 0.08 * lax.cospi(4 * n / (M - 1))) - 0.5 * lax.cospi(2 * n / (M - 1))


@export
def bartlett(M: int) -> Array:
  """Return a Bartlett window of size M.

  JAX implementation of :func:`numpy.bartlett`.

  Args:
    M: The window size.

  Returns:
    An array of size M containing the Bartlett window.

  Examples:
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.bartlett(4))
    [0.   0.67 0.67 0.  ]

  See also:
    - :func:`jax.numpy.blackman`: return a Blackman window of size M.
    - :func:`jax.numpy.hamming`: return a Hamming window of size M.
    - :func:`jax.numpy.hanning`: return a Hanning window of size M.
    - :func:`jax.numpy.kaiser`: return a Kaiser window of size M.
  """
  M = core.concrete_or_error(int, M, "M argument of jnp.bartlett")
  dtype = dtypes.default_float_dtype()
  if M <= 1:
    return lax.full((M,), 1, dtype)
  n = lax.iota(dtype, M)
  return 1 - ufuncs.abs(2 * n + 1 - M) / (M - 1)


@export
def hamming(M: int) -> Array:
  """Return a Hamming window of size M.

  JAX implementation of :func:`numpy.hamming`.

  Args:
    M: The window size.

  Returns:
    An array of size M containing the Hamming window.

  Examples:
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.hamming(4))
    [0.08 0.77 0.77 0.08]

  See also:
    - :func:`jax.numpy.bartlett`: return a Bartlett window of size M.
    - :func:`jax.numpy.blackman`: return a Blackman window of size M.
    - :func:`jax.numpy.hanning`: return a Hanning window of size M.
    - :func:`jax.numpy.kaiser`: return a Kaiser window of size M.
  """
  M = core.concrete_or_error(int, M, "M argument of jnp.hamming")
  dtype = dtypes.default_float_dtype()
  if M <= 1:
    return lax.full((M,), 1, dtype)
  n = _mirrored_iota(dtype, M)
  return 0.54 - 0.46 * lax.cospi(2 * n / (M - 1))


@export
def hanning(M: int) -> Array:
  """Return a Hanning window of size M.

  JAX implementation of :func:`numpy.hanning`.

  Args:
    M: The window size.

  Returns:
    An array of size M containing the Hanning window.

  Examples:
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.hanning(4))
    [0.   0.75 0.75 0.  ]

  See also:
    - :func:`jax.numpy.bartlett`: return a Bartlett window of size M.
    - :func:`jax.numpy.blackman`: return a Blackman window of size M.
    - :func:`jax.numpy.hamming`: return a Hamming window of size M.
    - :func:`jax.numpy.kaiser`: return a Kaiser window of size M.
  """
  M = core.concrete_or_error(int, M, "M argument of jnp.hanning")
  dtype = dtypes.default_float_dtype()
  if M <= 1:
    return lax.full((M,), 1, dtype)
  n = _mirrored_iota(dtype, M)
  return 0.5 * (1 - lax.cospi(2 * n / (M - 1)))


@export
def kaiser(M: int, beta: ArrayLike) -> Array:
  """Return a Kaiser window of size M.

  JAX implementation of :func:`numpy.kaiser`.

  Args:
    M: The window size.
    beta: The Kaiser window parameter.

  Returns:
    An array of size M containing the Kaiser window.

  Examples:
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.kaiser(4, 1.5))
    [0.61 0.95 0.95 0.61]

  See also:
    - :func:`jax.numpy.bartlett`: return a Bartlett window of size M.
    - :func:`jax.numpy.blackman`: return a Blackman window of size M.
    - :func:`jax.numpy.hamming`: return a Hamming window of size M.
    - :func:`jax.numpy.hanning`: return a Hanning window of size M.
  """
  M = core.concrete_or_error(int, M, "M argument of jnp.kaiser")
  dtype = dtypes.default_float_dtype()
  if M <= 1:
    return lax.full((M,), 1, dtype)
  n = lax.iota(dtype, M)
  alpha = 0.5 * (M - 1)
  return lax_numpy.i0(beta * ufuncs.sqrt(1 - ((n - alpha) / alpha) ** 2)) / lax_numpy.i0(beta)
