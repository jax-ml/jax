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

import numpy as np

from jax._src import core
from jax._src import dtypes
from jax._src.numpy import lax_numpy
from jax._src.numpy import ufuncs
from jax._src.typing import Array, ArrayLike
from jax._src.util import set_module
from jax import lax

export = set_module('jax.numpy')


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
    [-0.    0.63  0.63 -0.  ]

  See also:
    - :func:`jax.numpy.bartlett`: return a Bartlett window of size M.
    - :func:`jax.numpy.hamming`: return a Hamming window of size M.
    - :func:`jax.numpy.hanning`: return a Hanning window of size M.
    - :func:`jax.numpy.kaiser`: return a Kaiser window of size M.
  """
  M = core.concrete_or_error(int, M, "M argument of jnp.blackman")
  dtype = dtypes.canonicalize_dtype(dtypes.float_)
  if M <= 1:
    return lax.full((M,), 1, dtype)
  n = lax.iota(dtype, M)
  return 0.42 - 0.5 * ufuncs.cos(2 * np.pi * n / (M - 1)) + 0.08 * ufuncs.cos(4 * np.pi * n / (M - 1))


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
  dtype = dtypes.canonicalize_dtype(dtypes.float_)
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
  dtype = dtypes.canonicalize_dtype(dtypes.float_)
  if M <= 1:
    return lax.full((M,), 1, dtype)
  n = lax.iota(dtype, M)
  return 0.54 - 0.46 * ufuncs.cos(2 * np.pi * n / (M - 1))


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
  dtype = dtypes.canonicalize_dtype(dtypes.float_)
  if M <= 1:
    return lax.full((M,), 1, dtype)
  n = lax.iota(dtype, M)
  return 0.5 * (1 - ufuncs.cos(2 * np.pi * n / (M - 1)))


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
  dtype = dtypes.canonicalize_dtype(dtypes.float_)
  if M <= 1:
    return lax.full((M,), 1, dtype)
  n = lax.iota(dtype, M)
  alpha = 0.5 * (M - 1)
  return lax_numpy.i0(beta * ufuncs.sqrt(1 - ((n - alpha) / alpha) ** 2)) / lax_numpy.i0(beta)
