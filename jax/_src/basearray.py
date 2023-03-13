# Copyright 2022 The JAX Authors.
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

# Note that type annotations for this file are defined in basearray.pyi

import abc
import numpy as np
from typing import Union

# Array is a type annotation for standard JAX arrays and tracers produced by
# core functions in jax.lax and jax.numpy; it is not meant to include
# future non-standard array types like KeyArray and BInt.
class Array(abc.ABC):
  """Array base class for JAX

  ``jax.Array`` is the public interface for instance checks and type annotation
  of JAX arrays and tracers. Its main applications are in instance checks and
  type annotations; for example::

    x = jnp.arange(5)
    isinstance(x, jax.Array)  # returns True both inside and outside traced functions.

    def f(x: Array) -> Array:  # type annotations are valid for traced and non-traced types.
      return x

  ``jax.Array`` should not be used directly for creation of arrays; instead you
  should use array creation routines offered in :mod:`jax.numpy`, such as
  :func:`jax.numpy.array`, :func:`jax.numpy.zeros`, :func:`jax.numpy.ones`,
  :func:`jax.numpy.full`, :func:`jax.numpy.arange`, etc.
  """
  # Note: abstract methods for this class are defined dynamically in
  # lax_numpy.py
  # For the sake of static type analysis, these definitions are mirrored in the
  # associated basearray.pyi file.

  __slots__ = ['__weakref__']

  # at property must be defined because we overwrite its docstring in
  # lax_numpy.py
  @property
  def at(self):
    raise NotImplementedError("property must be defined in subclasses")


Array.__module__ = "jax"


# ArrayLike is a Union of all objects that can be implicitly converted to a
# standard JAX array (i.e. not including future non-standard array types like
# KeyArray and BInt). It's different than np.typing.ArrayLike in that it doesn't
# accept arbitrary sequences, nor does it accept string data.
ArrayLike = Union[
  Array,  # JAX array type
  np.ndarray,  # NumPy array type
  np.bool_, np.number,  # NumPy scalar types
  bool, int, float, complex,  # Python scalar types
]
ArrayLike.__doc__ = "Type annotation for JAX array-like objects."
