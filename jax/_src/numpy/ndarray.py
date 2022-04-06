# Copyright 2022 Google LLC
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

import abc
from typing import Any, Tuple, Optional, Union
from typing_extensions import Protocol

from jax import core
from jax.interpreters import pxla
from jax._src import device_array
import numpy as np

class Indexed(Protocol):
  """Helper object to call indexed update functions for an (advanced) index.

  This object references a source array and a specific indexer into that array.
  Methods on this object return copies of the source array that have been
  modified at the positions specified by the indexer.
  """

  def get(self, indices_are_sorted=False, unique_indices=False,
          mode=None, fill_value=None):
    """Equivalent to ``x[idx]``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexing <numpy.doc.indexing>` ``x[idx]``. This function differs from
    the usual array indexing syntax in that it allows additional keyword
    arguments ``indices_are_sorted`` and ``unique_indices`` to be passed.

    See :mod:`jax.ops` for details.
    """

  def set(self, values, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:`indexed assignment <numpy.doc.indexing>` ``x[idx] = y``.

    See :mod:`jax.ops` for details.
    """

  def apply(self, func, indices_are_sorted=False, unique_indices=False,
            mode=None):
    """Pure equivalent of ``func.at(x, idx)`` for a unary ufunc ``func``.

    Returns the value of ``x`` that would result from applying the unary
    function ``func`` to ``x`` at the given indices. This is similar to
    ``x.at[idx].set(func(x[idx]))``, but differs in the case of repeated indices:
    in ``x.at[idx].apply(func)``, repeated indices result in the function being
    applied multiple times.

    Note that in the current implementation, ``scatter_apply`` is not compatible
    with automatic differentiation.

    See :mod:`jax.ops` for details.
    """

  def add(self, values, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] += y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] += y``.

    See :mod:`jax.ops` for details.
    """

  def multiply(self, values, indices_are_sorted=False, unique_indices=False,
               mode=None):
    """Pure equivalent of ``x[idx] *= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] *= y``.

    See :mod:`jax.ops` for details.
    """

  def mul(self, values, indices_are_sorted=False, unique_indices=False,
               mode=None):
    """Pure equivalent of ``x[idx] *= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] *= y``.

    See :mod:`jax.ops` for details.
    """

  def divide(self, values, indices_are_sorted=False, unique_indices=False,
             mode=None):
    """Pure equivalent of ``x[idx] /= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] /= y``.

    See :mod:`jax.ops` for details.
    """

  def power(self, values, indices_are_sorted=False, unique_indices=False,
            mode=None):
    """Pure equivalent of ``x[idx] **= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] **= y``.

    See :mod:`jax.ops` for details.
    """

  def min(self, values, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = minimum(x[idx], y)``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>`
    ``x[idx] = minimum(x[idx], y)``.

    See :mod:`jax.ops` for details.
    """

  def max(self, values, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = maximum(x[idx], y)``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>`
    ``x[idx] = maximum(x[idx], y)``.

    See :mod:`jax.ops` for details.
    """

class At(Protocol):
  """Helper property for index update functionality.

  The ``at`` property provides a functionally pure equivalent of in-place
  array modificatons.

  In particular:

  ==============================  ================================
  Alternate syntax                Equivalent In-place expression
  ==============================  ================================
  ``x = x.at[idx].set(y)``        ``x[idx] = y``
  ``x = x.at[idx].add(y)``        ``x[idx] += y``
  ``x = x.at[idx].multiply(y)``   ``x[idx] *= y``
  ``x = x.at[idx].divide(y)``     ``x[idx] /= y``
  ``x = x.at[idx].power(y)``      ``x[idx] **= y``
  ``x = x.at[idx].min(y)``        ``x[idx] = minimum(x[idx], y)``
  ``x = x.at[idx].max(y)``        ``x[idx] = maximum(x[idx], y)``
  ``x = x.at[idx].apply(ufunc)``  ``ufunc.at(x, idx)``
  ``x = x.at[idx].get()``         ``x = x[idx]``
  ==============================  ================================

  None of the ``x.at`` expressions modify the original ``x``; instead they return
  a modified copy of ``x``. However, inside a :py:func:`~jax.jit` compiled function,
  expressions like :code:`x = x.at[idx].set(y)` are guaranteed to be applied in-place.

  Unlike NumPy in-place operations such as :code:`x[idx] += y`, if multiple
  indices refer to the same location, all updates will be applied (NumPy would
  only apply the last update, rather than applying all updates.) The order
  in which conflicting updates are applied is implementation-defined and may be
  nondeterministic (e.g., due to concurrency on some hardware platforms).

  By default, JAX assumes that all indices are in-bounds. There is experimental
  support for giving more precise semantics to out-of-bounds indexed accesses,
  via the ``mode`` parameter (see below).

  Arguments
  ---------
  mode : str
      Specify out-of-bound indexing mode. Options are:

      - ``"promise_in_bounds"``: (default) The user promises that indices are in bounds.
        No additional checking will be performed. In practice, this means that
        out-of-bounds indices in ``get()`` will be clipped, and out-of-bounds indices
        in ``set()``, ``add()``, etc. will be dropped.
      - ``"clip"``: clamp out of bounds indices into valid range.
      - ``"drop"``: ignore out-of-bound indices.
      - ``"fill"``: alias for ``"drop"``.  For `get()`, the optional ``fill_value``
        argument specifies the value that will be returned.

  indices_are_sorted : bool
      If True, the implementation will assume that the indices passed to ``at[]``
      are sorted in ascending order, which can lead to more efficient execution
      on some backends.
  unique_indices : bool
      If True, the implementation will assume that the indices passed to ``at[]``
      are unique, which can result in more efficient execution on some backends.
  fill_value : Any
      Only applies to the ``get()`` method: the fill value to return for out-of-bounds
      slices when `mode` is ``'fill'``. Ignored otherwise. Defaults to ``NaN`` for
      inexact types, the largest negative value for signed types, the largest positive
      value for unsigned types, and ``True`` for booleans.

  Examples
  --------
  >>> x = jnp.arange(5.0)
  >>> x
  DeviceArray([0., 1., 2., 3., 4.], dtype=float32)
  >>> x.at[2].add(10)
  DeviceArray([ 0.,  1., 12.,  3.,  4.], dtype=float32)
  >>> x.at[10].add(10)  # out-of-bounds indices are ignored
  DeviceArray([0., 1., 2., 3., 4.], dtype=float32)
  >>> x.at[20].add(10, mode='clip')
  DeviceArray([ 0.,  1.,  2.,  3., 14.], dtype=float32)
  >>> x.at[2].get()
  DeviceArray(2., dtype=float32)
  >>> x.at[20].get()  # out-of-bounds indices clipped
  DeviceArray(4., dtype=float32)
  >>> x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
  DeviceArray(nan, dtype=float32)
  >>> x.at[20].get(mode='fill', fill_value=-1)  # custom fill value
  DeviceArray(-1., dtype=float32)
  """
  def __getitem__(self, index) -> Indexed:
    pass

class ArrayMeta(abc.ABCMeta):
  """Metaclass for overriding ndarray isinstance checks."""

  def __instancecheck__(self, instance):
    # Allow tracer instances with avals that are instances of UnshapedArray.
    # We could instead just declare Tracer an instance of the ndarray type, but
    # there can be traced values that are not arrays. The main downside here is
    # that isinstance(x, ndarray) might return true but
    # issubclass(type(x), ndarray) might return false for an array tracer.
    try:
      return (hasattr(instance, "aval") and
              isinstance(instance.aval, core.UnshapedArray))
    except AttributeError:
      super().__instancecheck__(instance)


class ndarray(metaclass=ArrayMeta):
  dtype: np.dtype
  ndim: int
  shape: Tuple[int, ...]
  size: int

  def __init__(self, shape, dtype=None, buffer=None, offset=0, strides=None,
               order=None):
    raise TypeError("jax.numpy.ndarray() should not be instantiated explicitly."
                    " Use jax.numpy.array, or jax.numpy.zeros instead.")

  @abc.abstractmethod
  def __getitem__(self, key, indices_are_sorted=False,
                  unique_indices=False) -> Any: ...
  @abc.abstractmethod
  def __setitem__(self, key, value) -> Any: ...
  @abc.abstractmethod
  def __len__(self) -> Any: ...
  @abc.abstractmethod
  def __iter__(self) -> Any: ...
  @abc.abstractmethod
  def __reversed__(self) -> Any: ...

  # Comparisons
  @abc.abstractmethod
  def __lt__(self, other) -> Any: ...
  @abc.abstractmethod
  def __le__(self, other) -> Any: ...
  @abc.abstractmethod
  def __eq__(self, other) -> Any: ...
  @abc.abstractmethod
  def __ne__(self, other) -> Any: ...
  @abc.abstractmethod
  def __gt__(self, other) -> Any: ...
  @abc.abstractmethod
  def __ge__(self, other) -> Any: ...

  # Unary arithmetic

  @abc.abstractmethod
  def __neg__(self) -> Any: ...
  @abc.abstractmethod
  def __pos__(self) -> Any: ...
  @abc.abstractmethod
  def __abs__(self) -> Any: ...
  @abc.abstractmethod
  def __invert__(self) -> Any: ...

  # Binary arithmetic

  @abc.abstractmethod
  def __add__(self, other) -> Any: ...
  @abc.abstractmethod
  def __sub__(self, other) -> Any: ...
  @abc.abstractmethod
  def __mul__(self, other) -> Any: ...
  @abc.abstractmethod
  def __matmul__(self, other) -> Any: ...
  @abc.abstractmethod
  def __truediv__(self, other) -> Any: ...
  @abc.abstractmethod
  def __floordiv__(self, other) -> Any: ...
  @abc.abstractmethod
  def __mod__(self, other) -> Any: ...
  @abc.abstractmethod
  def __divmod__(self, other) -> Any: ...
  @abc.abstractmethod
  def __pow__(self, other) -> Any: ...
  @abc.abstractmethod
  def __lshift__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rshift__(self, other) -> Any: ...
  @abc.abstractmethod
  def __and__(self, other) -> Any: ...
  @abc.abstractmethod
  def __xor__(self, other) -> Any: ...
  @abc.abstractmethod
  def __or__(self, other) -> Any: ...

  @abc.abstractmethod
  def __radd__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rsub__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rmul__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rmatmul__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rtruediv__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rfloordiv__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rmod__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rdivmod__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rpow__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rlshift__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rrshift__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rand__(self, other) -> Any: ...
  @abc.abstractmethod
  def __rxor__(self, other) -> Any: ...
  @abc.abstractmethod
  def __ror__(self, other) -> Any: ...

  @abc.abstractmethod
  def __bool__(self) -> Any: ...
  @abc.abstractmethod
  def __complex__(self) -> Any: ...
  @abc.abstractmethod
  def __int__(self) -> Any: ...
  @abc.abstractmethod
  def __float__(self) -> Any: ...
  @abc.abstractmethod
  def __round__(self, ndigits=None) -> Any: ...

  @abc.abstractmethod
  def __index__(self) -> Any: ...

  # np.ndarray methods:
  @abc.abstractmethod
  def all(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
          keepdims=None) -> Any: ...
  @abc.abstractmethod
  def any(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
          keepdims=None) -> Any: ...
  @abc.abstractmethod
  def argmax(self, axis: Optional[int] = None, out=None, keepdims=None) -> Any: ...
  @abc.abstractmethod
  def argmin(self, axis: Optional[int] = None, out=None, keepdims=None) -> Any: ...
  @abc.abstractmethod
  def argpartition(self, kth, axis=-1, kind='introselect', order=None) -> Any: ...
  @abc.abstractmethod
  def argsort(self, axis: Optional[int] = -1, kind='quicksort', order=None) -> Any: ...
  @abc.abstractmethod
  def astype(self, dtype) -> Any: ...
  @abc.abstractmethod
  def choose(self, choices, out=None, mode='raise') -> Any: ...
  @abc.abstractmethod
  def clip(self, a_min=None, a_max=None, out=None) -> Any: ...
  @abc.abstractmethod
  def compress(self, condition, axis: Optional[int] = None, out=None) -> Any: ...
  @abc.abstractmethod
  def conj(self) -> Any: ...
  @abc.abstractmethod
  def conjugate(self) -> Any: ...
  @abc.abstractmethod
  def copy(self) -> Any: ...
  @abc.abstractmethod
  def cumprod(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
              dtype=None, out=None) -> Any: ...
  @abc.abstractmethod
  def cumsum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             dtype=None, out=None) -> Any: ...
  @abc.abstractmethod
  def diagonal(self, offset=0, axis1: int = 0, axis2: int = 1) -> Any: ...
  @abc.abstractmethod
  def dot(self, b, *, precision=None) -> Any: ...
  @abc.abstractmethod
  def flatten(self) -> Any: ...
  @property
  @abc.abstractmethod
  def imag(self) -> Any: ...
  @abc.abstractmethod
  def item(self, *args) -> Any: ...
  @abc.abstractmethod
  def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
          keepdims=None, initial=None, where=None) -> Any: ...
  @abc.abstractmethod
  def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
           out=None, keepdims=False, *, where=None,) -> Any: ...
  @abc.abstractmethod
  def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
          keepdims=None, initial=None, where=None) -> Any: ...
  @property
  @abc.abstractmethod
  def nbytes(self) -> Any: ...
  @abc.abstractmethod
  def nonzero(self, *, size=None, fill_value=None) -> Any: ...
  @abc.abstractmethod
  def prod(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
           out=None, keepdims=None, initial=None, where=None) -> Any: ...
  @abc.abstractmethod
  def ptp(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
          keepdims=False,) -> Any: ...
  @abc.abstractmethod
  def ravel(self, order='C') -> Any: ...
  @property
  @abc.abstractmethod
  def real(self) -> Any: ...
  @abc.abstractmethod
  def repeat(self, repeats, axis: Optional[int] = None, *,
             total_repeat_length=None) -> Any: ...
  @abc.abstractmethod
  def reshape(self, *args, order='C') -> Any: ...
  @abc.abstractmethod
  def round(self, decimals=0, out=None) -> Any: ...
  @abc.abstractmethod
  def searchsorted(self, v, side='left', sorter=None) -> Any: ...
  @abc.abstractmethod
  def sort(self, axis: Optional[int] = -1, kind='quicksort', order=None) -> Any: ...
  @abc.abstractmethod
  def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any: ...
  @abc.abstractmethod
  def std(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
          dtype=None, out=None, ddof=0, keepdims=False, *, where=None) -> Any: ...
  @abc.abstractmethod
  def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
          out=None, keepdims=None, initial=None, where=None) -> Any: ...
  @abc.abstractmethod
  def swapaxes(self, axis1: int, axis2: int) -> Any: ...
  @abc.abstractmethod
  def take(self, indices, axis: Optional[int] = None, out=None,
           mode=None) -> Any: ...
  @abc.abstractmethod
  def tobytes(self, order='C') -> Any: ...
  @abc.abstractmethod
  def tolist(self) -> Any: ...
  @abc.abstractmethod
  def trace(self, offset=0, axis1: int = 0, axis2: int = 1, dtype=None,
            out=None) -> Any: ...
  @abc.abstractmethod
  def transpose(self, *args) -> Any: ...
  @abc.abstractmethod
  def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
          dtype=None, out=None, ddof=0, keepdims=False, *, where=None) -> Any: ...
  @abc.abstractmethod
  def view(self, dtype=None, type=None) -> Any: ...

  # Even though we don't always support the NumPy array protocol, e.g., for
  # tracer types, for type checking purposes we must declare support so we
  # implement the NumPy ArrayLike protocol.
  def __array__(self) -> Any: ...

  # JAX extensions
  @property
  @abc.abstractmethod
  def at(self) -> At: ...
  @property
  @abc.abstractmethod
  def aval(self) -> Any: ...
  @property
  @abc.abstractmethod
  def weak_type(self) -> bool: ...


ndarray.register(device_array.DeviceArray)
for t in device_array.device_array_types:
  ndarray.register(t)
ndarray.register(pxla._SDA_BASE_CLASS)
