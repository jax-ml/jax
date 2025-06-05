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

import types
from functools import partial
import operator
from typing import Any, Literal, overload

import numpy as np

from jax._src.api import device_put, jit
from jax._src import core
from jax._src import dtypes
from jax._src.lax import lax
from jax._src.lib import xla_client as xc
from jax._src.numpy.array import asarray
from jax._src.numpy import ufuncs
from jax._src.numpy import util
from jax._src.sharding import Sharding
from jax._src.typing import Array, ArrayLike, DuckTypedArray, DTypeLike
from jax._src.util import canonicalize_axis, set_module


export = set_module('jax.numpy')


# Like core.canonicalize_shape, but also accept int-like (non-sequence)
# arguments for `shape`.
def canonicalize_shape(shape: Any, context: str="") -> core.Shape:
  if (not isinstance(shape, (tuple, list)) and
      (getattr(shape, 'ndim', None) == 0 or np.ndim(shape) == 0)):
    return core.canonicalize_shape((shape,), context)
  else:
    return core.canonicalize_shape(shape, context)


@export
def zeros(shape: Any, dtype: DTypeLike | None = None, *,
          device: xc.Device | Sharding | None = None) -> Array:
  """Create an array full of zeros.

  JAX implementation of :func:`numpy.zeros`.

  Args:
    shape: int or sequence of ints specifying the shape of the created array.
    dtype: optional dtype for the created array; defaults to float32 or float64
      depending on the X64 configuration (see :ref:`default-dtypes`).
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.zeros_like`
    - :func:`jax.numpy.empty`
    - :func:`jax.numpy.ones`
    - :func:`jax.numpy.full`

  Examples:
    >>> jnp.zeros(4)
    Array([0., 0., 0., 0.], dtype=float32)
    >>> jnp.zeros((2, 3), dtype=bool)
    Array([[False, False, False],
           [False, False, False]], dtype=bool)
  """
  if isinstance(shape, types.GeneratorType):
    raise TypeError("expected sequence object with len >= 0 or a single integer")
  if (m := _check_forgot_shape_tuple("zeros", shape, dtype)): raise TypeError(m)
  dtypes.check_user_dtype_supported(dtype, "zeros")
  shape = canonicalize_shape(shape)
  return lax.full(shape, 0, dtypes.jax_dtype(dtype), sharding=util.normalize_device_to_sharding(device))


@export
def ones(shape: Any, dtype: DTypeLike | None = None, *,
         device: xc.Device | Sharding | None = None) -> Array:
  """Create an array full of ones.

  JAX implementation of :func:`numpy.ones`.

  Args:
    shape: int or sequence of ints specifying the shape of the created array.
    dtype: optional dtype for the created array; defaults to float32 or float64
      depending on the X64 configuration (see :ref:`default-dtypes`).
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.ones_like`
    - :func:`jax.numpy.empty`
    - :func:`jax.numpy.zeros`
    - :func:`jax.numpy.full`

  Examples:
    >>> jnp.ones(4)
    Array([1., 1., 1., 1.], dtype=float32)
    >>> jnp.ones((2, 3), dtype=bool)
    Array([[ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
  """
  if isinstance(shape, types.GeneratorType):
    raise TypeError("expected sequence object with len >= 0 or a single integer")
  if (m := _check_forgot_shape_tuple("ones", shape, dtype)): raise TypeError(m)
  shape = canonicalize_shape(shape)
  dtypes.check_user_dtype_supported(dtype, "ones")
  return lax.full(shape, 1, dtypes.jax_dtype(dtype), sharding=util.normalize_device_to_sharding(device))


@export
def empty(shape: Any, dtype: DTypeLike | None = None, *,
          device: xc.Device | Sharding | None = None) -> Array:
  """Create an empty array.

  JAX implementation of :func:`numpy.empty`. Because XLA cannot create an
  un-initialized array, :func:`jax.numpy.empty` will always return an array
  full of zeros.

  Args:
    shape: int or sequence of ints specifying the shape of the created array.
    dtype: optional dtype for the created array; defaults to float32 or float64
      depending on the X64 configuration (see :ref:`default-dtypes`).
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.empty_like`
    - :func:`jax.numpy.zeros`
    - :func:`jax.numpy.ones`
    - :func:`jax.numpy.full`

  Examples:
    >>> jnp.empty(4)
    Array([0., 0., 0., 0.], dtype=float32)
    >>> jnp.empty((2, 3), dtype=bool)
    Array([[False, False, False],
           [False, False, False]], dtype=bool)
  """
  if (m := _check_forgot_shape_tuple("empty", shape, dtype)): raise TypeError(m)
  dtypes.check_user_dtype_supported(dtype, "empty")
  return zeros(shape, dtype, device=device)


def _check_forgot_shape_tuple(name, shape, dtype) -> str | None:  # type: ignore
  if isinstance(dtype, int) and isinstance(shape, int):
    return (f"Cannot interpret '{dtype}' as a data type."
            f"\n\nDid you accidentally write "
            f"`jax.numpy.{name}({shape}, {dtype})` "
            f"when you meant `jax.numpy.{name}(({shape}, {dtype}))`, i.e. "
            "with a single tuple argument for the shape?")

@export
def full(shape: Any, fill_value: ArrayLike,
         dtype: DTypeLike | None = None, *,
         device: xc.Device | Sharding | None = None) -> Array:
  """Create an array full of a specified value.

  JAX implementation of :func:`numpy.full`.

  Args:
    shape: int or sequence of ints specifying the shape of the created array.
    fill_value: scalar or array with which to fill the created array.
    dtype: optional dtype for the created array; defaults to the dtype of the
      fill value.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.full_like`
    - :func:`jax.numpy.empty`
    - :func:`jax.numpy.zeros`
    - :func:`jax.numpy.ones`

  Examples:
    >>> jnp.full(4, 2, dtype=float)
    Array([2., 2., 2., 2.], dtype=float32)
    >>> jnp.full((2, 3), 0, dtype=bool)
    Array([[False, False, False],
           [False, False, False]], dtype=bool)

    `fill_value` may also be an array that is broadcast to the specified shape:

    >>> jnp.full((2, 3), fill_value=jnp.arange(3))
    Array([[0, 1, 2],
           [0, 1, 2]], dtype=int32)
  """
  dtypes.check_user_dtype_supported(dtype, "full")
  util.check_arraylike("full", fill_value)

  if np.ndim(fill_value) == 0:
    shape = canonicalize_shape(shape)
    return lax.full(shape, fill_value, dtype, sharding=util.normalize_device_to_sharding(device))
  else:
    return device_put(
        util._broadcast_to(asarray(fill_value, dtype=dtype), shape), device)


@export
def zeros_like(a: ArrayLike | DuckTypedArray,
               dtype: DTypeLike | None = None,
               shape: Any = None, *,
               device: xc.Device | Sharding | None = None) -> Array:
  """Create an array full of zeros with the same shape and dtype as an array.

  JAX implementation of :func:`numpy.zeros_like`.

  Args:
    a: Array-like object with ``shape`` and ``dtype`` attributes.
    shape: optionally override the shape of the created array.
    dtype: optionally override the dtype of the created array.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.zeros`
    - :func:`jax.numpy.empty_like`
    - :func:`jax.numpy.ones_like`
    - :func:`jax.numpy.full_like`

  Examples:
    >>> x = jnp.arange(4)
    >>> jnp.zeros_like(x)
    Array([0, 0, 0, 0], dtype=int32)
    >>> jnp.zeros_like(x, dtype=bool)
    Array([False, False, False, False], dtype=bool)
    >>> jnp.zeros_like(x, shape=(2, 3))
    Array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)
  """
  if not (hasattr(a, 'dtype') and hasattr(a, 'shape')):  # support duck typing
    if hasattr(a, '__jax_array__'):
      a = a.__jax_array__()
    util.check_arraylike("zeros_like", a)
  dtypes.check_user_dtype_supported(dtype, "zeros_like")
  if shape is not None:
    shape = canonicalize_shape(shape)
  return lax.full_like(a, 0, dtype, shape, sharding=util.normalize_device_to_sharding(device))


@export
def ones_like(a: ArrayLike | DuckTypedArray,
              dtype: DTypeLike | None = None,
              shape: Any = None, *,
              device: xc.Device | Sharding | None = None) -> Array:
  """Create an array of ones with the same shape and dtype as an array.

  JAX implementation of :func:`numpy.ones_like`.

  Args:
    a: Array-like object with ``shape`` and ``dtype`` attributes.
    shape: optionally override the shape of the created array.
    dtype: optionally override the dtype of the created array.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.empty`
    - :func:`jax.numpy.zeros_like`
    - :func:`jax.numpy.ones_like`
    - :func:`jax.numpy.full_like`

  Examples:
    >>> x = jnp.arange(4)
    >>> jnp.ones_like(x)
    Array([1, 1, 1, 1], dtype=int32)
    >>> jnp.ones_like(x, dtype=bool)
    Array([ True,  True,  True,  True], dtype=bool)
    >>> jnp.ones_like(x, shape=(2, 3))
    Array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)
  """
  if not (hasattr(a, 'dtype') and hasattr(a, 'shape')):  # support duck typing
    if hasattr(a, '__jax_array__'):
      a = a.__jax_array__()
    util.check_arraylike("ones_like", a)
  dtypes.check_user_dtype_supported(dtype, "ones_like")
  if shape is not None:
    shape = canonicalize_shape(shape)
  return lax.full_like(a, 1, dtype, shape, sharding=util.normalize_device_to_sharding(device))


@export
def empty_like(prototype: ArrayLike | DuckTypedArray,
               dtype: DTypeLike | None = None,
               shape: Any = None, *,
               device: xc.Device | Sharding | None = None) -> Array:
  """Create an empty array with the same shape and dtype as an array.

  JAX implementation of :func:`numpy.empty_like`. Because XLA cannot create
  an un-initialized array, :func:`jax.numpy.empty` will always return an
  array full of zeros.

  Args:
    a: Array-like object with ``shape`` and ``dtype`` attributes.
    shape: optionally override the shape of the created array.
    dtype: optionally override the dtype of the created array.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.empty`
    - :func:`jax.numpy.zeros_like`
    - :func:`jax.numpy.ones_like`
    - :func:`jax.numpy.full_like`

  Examples:
    >>> x = jnp.arange(4)
    >>> jnp.empty_like(x)
    Array([0, 0, 0, 0], dtype=int32)
    >>> jnp.empty_like(x, dtype=bool)
    Array([False, False, False, False], dtype=bool)
    >>> jnp.empty_like(x, shape=(2, 3))
    Array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)
  """
  if not (hasattr(prototype, 'dtype') and hasattr(prototype, 'shape')):  # support duck typing
    if hasattr(prototype, '__jax_array__'):
      prototype = prototype.__jax_array__()
    util.check_arraylike("ones_like", prototype)
  dtypes.check_user_dtype_supported(dtype, "ones_like")
  if shape is not None:
    shape = canonicalize_shape(shape)
  return lax.full_like(prototype, 0, dtype, shape, sharding=util.normalize_device_to_sharding(device))


@export
def full_like(a: ArrayLike | DuckTypedArray,
              fill_value: ArrayLike, dtype: DTypeLike | None = None,
              shape: Any = None, *,
              device: xc.Device | Sharding | None = None) -> Array:
  """Create an array full of a specified value with the same shape and dtype as an array.

  JAX implementation of :func:`numpy.full_like`.

  Args:
    a: Array-like object with ``shape`` and ``dtype`` attributes.
    fill_value: scalar or array with which to fill the created array.
    shape: optionally override the shape of the created array.
    dtype: optionally override the dtype of the created array.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.full`
    - :func:`jax.numpy.empty_like`
    - :func:`jax.numpy.zeros_like`
    - :func:`jax.numpy.ones_like`

  Examples:
    >>> x = jnp.arange(4.0)
    >>> jnp.full_like(x, 2)
    Array([2., 2., 2., 2.], dtype=float32)
    >>> jnp.full_like(x, 0, shape=(2, 3))
    Array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

    `fill_value` may also be an array that is broadcast to the specified shape:

    >>> x = jnp.arange(6).reshape(2, 3)
    >>> jnp.full_like(x, fill_value=jnp.array([[1], [2]]))
    Array([[1, 1, 1],
           [2, 2, 2]], dtype=int32)
  """
  if hasattr(a, 'dtype') and hasattr(a, 'shape'):  # support duck typing
    util.check_arraylike("full_like", 0, fill_value)
  else:
    util.check_arraylike("full_like", a, fill_value)
    if hasattr(a, '__jax_array__'):
      a = a.__jax_array__()
  dtypes.check_user_dtype_supported(dtype, "full_like")
  if shape is not None:
    shape = canonicalize_shape(shape)
  if np.ndim(fill_value) == 0:
    return lax.full_like(a, fill_value, dtype, shape, sharding=util.normalize_device_to_sharding(device))
  else:
    shape = np.shape(a) if shape is None else shape  # type: ignore[arg-type]
    dtype = dtypes.result_type(a) if dtype is None else dtype
    return device_put(
        util._broadcast_to(asarray(fill_value, dtype=dtype), shape), device)

@overload
def linspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
             endpoint: bool = True, retstep: Literal[False] = False,
             dtype: DTypeLike | None = None,
             axis: int = 0,
             *, device: xc.Device | Sharding | None = None) -> Array: ...
@overload
def linspace(start: ArrayLike, stop: ArrayLike, num: int,
             endpoint: bool, retstep: Literal[True],
             dtype: DTypeLike | None = None,
             axis: int = 0,
             *, device: xc.Device | Sharding | None = None) -> tuple[Array, Array]: ...
@overload
def linspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
             endpoint: bool = True, *, retstep: Literal[True],
             dtype: DTypeLike | None = None,
             axis: int = 0,
             device: xc.Device | Sharding | None = None) -> tuple[Array, Array]: ...
@overload
def linspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
             endpoint: bool = True, retstep: bool = False,
             dtype: DTypeLike | None = None,
             axis: int = 0,
             *, device: xc.Device | Sharding | None = None) -> Array | tuple[Array, Array]: ...
@export
def linspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
             endpoint: bool = True, retstep: bool = False,
             dtype: DTypeLike | None = None,
             axis: int = 0,
             *, device: xc.Device | Sharding | None = None) -> Array | tuple[Array, Array]:
  """Return evenly-spaced numbers within an interval.

  JAX implementation of :func:`numpy.linspace`.

  Args:
    start: scalar or array of starting values.
    stop: scalar or array of stop values.
    num: number of values to generate. Default: 50.
    endpoint: if True (default) then include the ``stop`` value in the result.
      If False, then exclude the ``stop`` value.
    retstep: If True, then return a ``(result, step)`` tuple, where ``step`` is the
      interval between adjacent values in ``result``.
    axis: integer axis along which to generate the linspace. Defaults to zero.
    device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    An array ``values``, or a tuple ``(values, step)`` if ``retstep`` is True, where:

    - ``values`` is an array of evenly-spaced values from ``start`` to ``stop``
    - ``step`` is the interval between adjacent values.

  See also:
    - :func:`jax.numpy.arange`: Generate ``N`` evenly-spaced values given a starting
      point and a step
    - :func:`jax.numpy.logspace`: Generate logarithmically-spaced values.
    - :func:`jax.numpy.geomspace`: Generate geometrically-spaced values.

  Examples:
    List of 5 values between 0 and 10:

    >>> jnp.linspace(0, 10, 5)
    Array([ 0. ,  2.5,  5. ,  7.5, 10. ], dtype=float32)

    List of 8 values between 0 and 10, excluding the endpoint:

    >>> jnp.linspace(0, 10, 8, endpoint=False)
    Array([0.  , 1.25, 2.5 , 3.75, 5.  , 6.25, 7.5 , 8.75], dtype=float32)

    List of values and the step size between them

    >>> vals, step = jnp.linspace(0, 10, 9, retstep=True)
    >>> vals
    Array([ 0.  ,  1.25,  2.5 ,  3.75,  5.  ,  6.25,  7.5 ,  8.75, 10.  ],      dtype=float32)
    >>> step
    Array(1.25, dtype=float32)

    Multi-dimensional linspace:

    >>> start = jnp.array([0, 5])
    >>> stop = jnp.array([5, 10])
    >>> jnp.linspace(start, stop, 5)
    Array([[ 0.  ,  5.  ],
           [ 1.25,  6.25],
           [ 2.5 ,  7.5 ],
           [ 3.75,  8.75],
           [ 5.  , 10.  ]], dtype=float32)
  """
  num = core.concrete_dim_or_error(num, "'num' argument of jnp.linspace")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.linspace")
  return _linspace(start, stop, num, endpoint, retstep, dtype, axis, device=device)

@partial(jit, static_argnames=('num', 'endpoint', 'retstep', 'dtype', 'axis', 'device'))
def _linspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
              endpoint: bool = True, retstep: bool = False,
              dtype: DTypeLike | None = None,
              axis: int = 0,
              *, device: xc.Device | Sharding | None = None) -> Array | tuple[Array, Array]:
  """Implementation of linspace differentiable in start and stop args."""
  dtypes.check_user_dtype_supported(dtype, "linspace")
  if num < 0:
    raise ValueError(f"Number of samples, {num}, must be non-negative.")
  start, stop = util.ensure_arraylike("linspace", start, stop)

  if dtype is None:
    dtype = dtypes.to_inexact_dtype(dtypes.result_type(start, stop))
  dtype = dtypes.jax_dtype(dtype)
  computation_dtype = dtypes.to_inexact_dtype(dtype)
  start = start.astype(computation_dtype)
  stop = stop.astype(computation_dtype)

  bounds_shape = list(lax.broadcast_shapes(np.shape(start), np.shape(stop)))
  broadcast_start = util._broadcast_to(start, bounds_shape)
  broadcast_stop = util._broadcast_to(stop, bounds_shape)
  axis = len(bounds_shape) + axis + 1 if axis < 0 else axis
  bounds_shape.insert(axis, 1)
  div = (num - 1) if endpoint else num
  if num > 1:
    delta: Array = lax.convert_element_type(stop - start, computation_dtype) / asarray(div, dtype=computation_dtype)
    iota_shape = [1,] * len(bounds_shape)
    iota_shape[axis] = div
    # This approach recovers the endpoints with float32 arithmetic,
    # but can lead to rounding errors for integer outputs.
    real_dtype = dtypes.finfo(computation_dtype).dtype
    step = lax.iota(real_dtype, div).reshape(iota_shape) / asarray(div, real_dtype)
    step = step.astype(computation_dtype)
    out = (broadcast_start.reshape(bounds_shape) * (1 - step) +
      broadcast_stop.reshape(bounds_shape) * step)

    if endpoint:
      out = lax.concatenate([out, lax.expand_dims(broadcast_stop, (axis,))],
                            canonicalize_axis(axis, out.ndim))

  elif num == 1:
    delta = asarray(np.nan if endpoint else stop - start, dtype=computation_dtype)
    out = broadcast_start.reshape(bounds_shape)
  else:  # num == 0 degenerate case, match numpy behavior
    empty_shape = list(lax.broadcast_shapes(np.shape(start), np.shape(stop)))
    empty_shape.insert(axis, 0)
    delta = full((), np.nan, computation_dtype)
    out = empty(empty_shape, dtype)

  if dtypes.issubdtype(dtype, np.integer) and not dtypes.issubdtype(out.dtype, np.integer):
    out = lax.floor(out)

  sharding = util.canonicalize_device_to_sharding(device)
  result = lax._convert_element_type(out, dtype, sharding=sharding)
  return (result, delta) if retstep else result


@export
def logspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
             endpoint: bool = True, base: ArrayLike = 10.0,
             dtype: DTypeLike | None = None, axis: int = 0) -> Array:
  """Generate logarithmically-spaced values.

  JAX implementation of :func:`numpy.logspace`.

  Args:
    start: scalar or array. Used to specify the start value. The start value is
      ``base ** start``.
    stop: scalar or array. Used to specify the stop value. The end value is
      ``base ** stop``.
    num: int, optional, default=50. Number of values to generate.
    endpoint: bool, optional, default=True. If True, then include the ``stop`` value
      in the result. If False, then exclude the ``stop`` value.
    base: scalar or array, optional, default=10. Specifies the base of the logarithm.
    dtype: optional. Specifies the dtype of the output.
    axis: int, optional, default=0. Axis along which to generate the logspace.

  Returns:
    An array of logarithm.

  See also:
    - :func:`jax.numpy.arange`: Generate ``N`` evenly-spaced values given a starting
      point and a step value.
    - :func:`jax.numpy.linspace`: Generate evenly-spaced values.
    - :func:`jax.numpy.geomspace`: Generate geometrically-spaced values.

  Examples:
    List 5 logarithmically spaced values between 1 (``10 ** 0``) and 100
    (``10 ** 2``):

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.logspace(0, 2, 5)
    Array([  1.   ,   3.162,  10.   ,  31.623, 100.   ], dtype=float32)

    List 5 logarithmically-spaced values between 1(``10 ** 0``) and 100
    (``10 ** 2``), excluding endpoint:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.logspace(0, 2, 5, endpoint=False)
    Array([ 1.   ,  2.512,  6.31 , 15.849, 39.811], dtype=float32)

    List 7 logarithmically-spaced values between 1 (``2 ** 0``) and 4 (``2 ** 2``)
    with base 2:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.logspace(0, 2, 7, base=2)
    Array([1.   , 1.26 , 1.587, 2.   , 2.52 , 3.175, 4.   ], dtype=float32)

    Multi-dimensional logspace:

    >>> start = jnp.array([0, 5])
    >>> stop = jnp.array([5, 0])
    >>> base = jnp.array([2, 3])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.logspace(start, stop, 5, base=base)
    Array([[  1.   , 243.   ],
           [  2.378,  61.547],
           [  5.657,  15.588],
           [ 13.454,   3.948],
           [ 32.   ,   1.   ]], dtype=float32)
  """
  num = core.concrete_or_error(operator.index, num, "'num' argument of jnp.logspace")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.logspace")
  return _logspace(start, stop, num, endpoint, base, dtype, axis)

@partial(jit, static_argnames=('num', 'endpoint', 'dtype', 'axis'))
def _logspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
              endpoint: bool = True, base: ArrayLike = 10.0,
              dtype: DTypeLike | None = None, axis: int = 0) -> Array:
  """Implementation of logspace differentiable in start and stop args."""
  dtypes.check_user_dtype_supported(dtype, "logspace")
  if dtype is None:
    dtype = dtypes.to_inexact_dtype(dtypes.result_type(start, stop))
  dtype = dtypes.jax_dtype(dtype)
  computation_dtype = dtypes.to_inexact_dtype(dtype)
  start, stop = util.ensure_arraylike("logspace", start, stop)
  start = start.astype(computation_dtype)
  stop = stop.astype(computation_dtype)
  lin = linspace(start, stop, num,
                 endpoint=endpoint, retstep=False, dtype=None, axis=axis)
  return lax.convert_element_type(ufuncs.power(base, lin), dtype)


@export
def geomspace(start: ArrayLike, stop: ArrayLike, num: int = 50, endpoint: bool = True,
              dtype: DTypeLike | None = None, axis: int = 0) -> Array:
  """Generate geometrically-spaced values.

  JAX implementation of :func:`numpy.geomspace`.

  Args:
    start: scalar or array. Specifies the starting values.
    stop: scalar or array. Specifies the stop values.
    num: int, optional, default=50. Number of values to generate.
    endpoint: bool, optional, default=True. If True, then include the ``stop`` value
      in the result. If False, then exclude the ``stop`` value.
    dtype: optional. Specifies the dtype of the output.
    axis: int, optional, default=0. Axis along which to generate the geomspace.

  Returns:
    An array containing the geometrically-spaced values.

  See also:
    - :func:`jax.numpy.arange`: Generate ``N`` evenly-spaced values given a starting
      point and a step value.
    - :func:`jax.numpy.linspace`: Generate evenly-spaced values.
    - :func:`jax.numpy.logspace`: Generate logarithmically-spaced values.

  Examples:
    List 5 geometrically-spaced values between 1 and 16:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.geomspace(1, 16, 5)
    Array([ 1.,  2.,  4.,  8., 16.], dtype=float32)

    List 4 geomtrically-spaced values between 1 and 16, with ``endpoint=False``:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.geomspace(1, 16, 4, endpoint=False)
    Array([1., 2., 4., 8.], dtype=float32)

    Multi-dimensional geomspace:

    >>> start = jnp.array([1, 1000])
    >>> stop = jnp.array([27, 1])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.geomspace(start, stop, 4)
    Array([[   1., 1000.],
           [   3.,  100.],
           [   9.,   10.],
           [  27.,    1.]], dtype=float32)
  """
  num = core.concrete_or_error(operator.index, num, "'num' argument of jnp.geomspace")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.geomspace")
  return _geomspace(start, stop, num, endpoint, dtype, axis)

@partial(jit, static_argnames=('num', 'endpoint', 'dtype', 'axis'))
def _geomspace(start: ArrayLike, stop: ArrayLike, num: int = 50, endpoint: bool = True,
               dtype: DTypeLike | None = None, axis: int = 0) -> Array:
  """Implementation of geomspace differentiable in start and stop args."""
  dtypes.check_user_dtype_supported(dtype, "geomspace")
  if dtype is None:
    dtype = dtypes.to_inexact_dtype(dtypes.result_type(start, stop))
  dtype = dtypes.jax_dtype(dtype)
  computation_dtype = dtypes.to_inexact_dtype(dtype)
  start, stop = util.ensure_arraylike("geomspace", start, stop)
  start = start.astype(computation_dtype)
  stop = stop.astype(computation_dtype)

  sign = ufuncs.sign(start)
  res = sign * logspace(ufuncs.log10(start / sign), ufuncs.log10(stop / sign),
                        num, endpoint=endpoint, base=10.0,
                        dtype=computation_dtype, axis=0)
  axis = canonicalize_axis(axis, res.ndim)
  if axis != 0:
    # res = moveaxis(res, 0, axis)
    res = lax.transpose(res, permutation=(*range(1, axis + 1), 0, *range(axis + 1, res.ndim)))
  return lax.convert_element_type(res, dtype)
