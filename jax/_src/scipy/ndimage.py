# Copyright 2019 The JAX Authors.
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

from collections.abc import Callable, Sequence
import functools
import itertools
import operator

from jax._src import api
from jax._src import util
from jax import lax
import jax
import jax.numpy as jnp
from jax._src.typing import ArrayLike, Array
from jax._src.util import safe_zip as zip

from scipy.ndimage import _ni_support

def _nonempty_prod(arrs: Sequence[Array]) -> Array:
  return functools.reduce(operator.mul, arrs)

def _nonempty_sum(arrs: Sequence[Array]) -> Array:
  return functools.reduce(operator.add, arrs)

def _mirror_index_fixer(index: Array, size: int) -> Array:
    s = size - 1 # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return jnp.abs((index + s) % (2 * s) - s)

def _reflect_index_fixer(index: Array, size: int) -> Array:
    return jnp.floor_divide(_mirror_index_fixer(2*index+1, 2*size+1) - 1, 2)

_INDEX_FIXERS: dict[str, Callable[[Array, int], Array]] = {
    'constant': lambda index, size: index,
    'nearest': lambda index, size: jnp.clip(index, 0, size - 1),
    'wrap': lambda index, size: index % size,
    'mirror': _mirror_index_fixer,
    'reflect': _reflect_index_fixer,
}


def _round_half_away_from_zero(a: Array) -> Array:
  return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)


def _nearest_indices_and_weights(coordinate: Array) -> list[tuple[Array, ArrayLike]]:
  index = _round_half_away_from_zero(coordinate).astype(jnp.int32)
  weight = coordinate.dtype.type(1)
  return [(index, weight)]


def _linear_indices_and_weights(coordinate: Array) -> list[tuple[Array, ArrayLike]]:
  lower = jnp.floor(coordinate)
  upper_weight = coordinate - lower
  lower_weight = 1 - upper_weight
  index = lower.astype(jnp.int32)
  return [(index, lower_weight), (index + 1, upper_weight)]


@functools.partial(api.jit, static_argnums=(2, 3, 4))
def _map_coordinates(input: ArrayLike, coordinates: Sequence[ArrayLike],
                     order: int, mode: str, cval: ArrayLike) -> Array:
  input_arr = jnp.asarray(input)
  coordinate_arrs = [jnp.asarray(c) for c in coordinates]
  cval = jnp.asarray(cval, input_arr.dtype)

  if len(coordinates) != input_arr.ndim:
    raise ValueError('coordinates must be a sequence of length input.ndim, but '
                     '{} != {}'.format(len(coordinates), input_arr.ndim))

  index_fixer = _INDEX_FIXERS.get(mode)
  if index_fixer is None:
    raise NotImplementedError(
        'jax.scipy.ndimage.map_coordinates does not yet support mode {}. '
        'Currently supported modes are {}.'.format(mode, set(_INDEX_FIXERS)))

  if mode == 'constant':
    is_valid = lambda index, size: (0 <= index) & (index < size)
  else:
    is_valid = lambda index, size: True

  if order == 0:
    interp_fun = _nearest_indices_and_weights
  elif order == 1:
    interp_fun = _linear_indices_and_weights
  else:
    raise NotImplementedError(
        'jax.scipy.ndimage.map_coordinates currently requires order<=1')

  valid_1d_interpolations = []
  for coordinate, size in zip(coordinate_arrs, input_arr.shape):
    interp_nodes = interp_fun(coordinate)
    valid_interp = []
    for index, weight in interp_nodes:
      fixed_index = index_fixer(index, size)
      valid = is_valid(index, size)
      valid_interp.append((fixed_index, valid, weight))
    valid_1d_interpolations.append(valid_interp)

  outputs = []
  for items in itertools.product(*valid_1d_interpolations):
    indices, validities, weights = util.unzip3(items)
    if all(valid is True for valid in validities):
      # fast path
      contribution = input_arr[indices]
    else:
      all_valid = functools.reduce(operator.and_, validities)
      contribution = jnp.where(all_valid, input_arr[indices], cval)
    outputs.append(_nonempty_prod(weights) * contribution)  # type: ignore
  result = _nonempty_sum(outputs)
  if jnp.issubdtype(input_arr.dtype, jnp.integer):
    result = _round_half_away_from_zero(result)
  return result.astype(input_arr.dtype)


"""
    Only nearest neighbor (``order=0``), linear interpolation (``order=1``) and
    modes ``'constant'``, ``'nearest'``, ``'wrap'`` ``'mirror'`` and ``'reflect'`` are currently supported.

    """

def map_coordinates(
    input: ArrayLike, coordinates: Sequence[ArrayLike], order: int,
    mode: str = 'constant', cval: ArrayLike = 0.0,
):
  """
  Map the input array to new coordinates using interpolation.

  JAX implementation of :func:`scipy.ndimage.map_coordinates`

  Given an input array and a set of coordinates, this function returns the
  interpolated values of the input array at those coordinates.

  Args:
    input: N-dimensional input array from which values are interpolated.
    coordinates: length-N sequence of arrays specifying the coordinates
      at which to evaluate the interpolated values
    order: The order of interpolation. JAX supports the following:

      * 0: Nearest-neighbor
      * 1: Linear

    mode: Points outside the boundaries of the input are filled according to the given mode.
      JAX supports one of ``('constant', 'nearest', 'mirror', 'wrap', 'reflect')``. Note the
      ``'wrap'`` mode in JAX behaves as ``'grid-wrap'`` mode in SciPy, and ``'constant'``
      mode in JAX behaves as ``'grid-constant'`` mode in SciPy. This discrepancy was caused
      by a former bug in those modes in SciPy (https://github.com/scipy/scipy/issues/2640),
      which was first fixed in JAX by changing the behavior of the existing modes, and later
      on fixed in SciPy, by adding modes with new names, rather than fixing the existing
      ones, for backwards compatibility reasons. Default is 'constant'.
    cval: Value used for points outside the boundaries of the input if ``mode='constant'``
      Default is 0.0.

  Returns:
    The interpolated values at the specified coordinates.

  Examples:
    >>> input = jnp.arange(12.0).reshape(3, 4)
    >>> input
    Array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]], dtype=float32)
    >>> coordinates = [jnp.array([0.5, 1.5]),
    ...                jnp.array([1.5, 2.5])]
    >>> jax.scipy.ndimage.map_coordinates(input, coordinates, order=1)
    Array([3.5, 8.5], dtype=float32)

  Note:
    Interpolation near boundaries differs from the scipy function, because JAX
    fixed an outstanding bug; see https://github.com/jax-ml/jax/issues/11097.
    This function interprets the ``mode`` argument as documented by SciPy, but
    not as implemented by SciPy.
  """
  return _map_coordinates(input, coordinates, order, mode, cval)


def _gaussian(x, sigma):
    return jnp.exp(-0.5 / sigma ** 2 * x ** 2) / jnp.sqrt(2 * jnp.pi * sigma ** 2)

def _grad_order(func, order):
    """Compute higher order grads recursively"""
    if order == 0:
      return func

    return jax.grad(_grad_order(func, order - 1))


def _gaussian_kernel1d(sigma, order, radius):
    """Computes a 1-D Gaussian convolution kernel"""
    if order < 0:
        raise ValueError(f'Order must be non-negative, got {order}')
    
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    func = _grad_order(functools.partial(_gaussian, sigma=sigma), order)
    kernel = jax.vmap(func)(x)
    
    if order == 0:
       return kernel / jnp.sum(kernel)
    
    return kernel


def gaussian_filter1d(
      input: ArrayLike,
      sigma: float,
      axis: int = -1,
      order: int = 0,
      mode: str = 'reflect',
      cval: float = 0.0,
      truncate: float = 4.0,
      *,
      radius: int | None = None,
      method: str = "auto"):
    """Compute a 1D Gaussian filter on the input array along the specified axis.

    Args:
        input: N-dimensional input array to filter.
        sigma: The standard deviation of the Gaussian filter.
        axis: The axis along which to apply the filter.
        order: The order of the Gaussian filter.
        mode: The mode to use for padding the input array. See :func:`jax.numpy.pad` for more details.
        cval: The value to use for padding the input array.
        truncate: The number of standard deviations to include in the filter.
        radius: The radius of the filter. Overrides `truncate` if provided.
        method: The method to use for the convolution.

    Returns:
        The filtered array.

    Examples:
        >>> from jax import numpy as jnp
        >>> import jax
        >>> input = jnp.arange(12.0).reshape(3, 4)
        >>> input
        Array([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.]], dtype=float32)
        >>> jax.scipy.ndimage.gaussian_filter1d(input, sigma=1.0, axis=0, order=0)
       Array([[2.8350844, 3.8350847, 4.8350844, 5.8350844],
              [4.0000005, 5.       , 6.       , 7.0000005],
              [5.1649156, 6.1649156, 7.164916 , 8.164916 ]], dtype=float32)
    """
    if radius is None:
        radius = int(truncate * sigma + 0.5)

    if radius < 0:
        raise ValueError(f'Radius must be non-negative, got {radius}')

    if sigma <= 0:
        raise ValueError(f'Sigma must be positive, got {sigma}')
    
    pad_width = [(0, 0)] * input.ndim
    pad_width[axis] = (int(radius), int(radius))

    pad_kwargs = {'mode': mode}
    
    if mode == 'constant':
       # jnp.pad errors if constant_values is provided and mode is not 'constant'
       pad_kwargs['constant_values'] = cval

    input_pad = jnp.pad(input, pad_width=pad_width, **pad_kwargs)

    kernel = _gaussian_kernel1d(sigma, order=order, radius=radius)
    
    axes = list(range(input.ndim))
    axes.pop(axis)
    kernel = jnp.expand_dims(kernel, axes)
    
    # boundary handling is done by jnp.pad, so we use the fixed valid mode
    return jax.scipy.signal.convolve(input_pad, kernel, mode="valid", method=method)

  
def gaussian_filter(
      input: ArrayLike,
      sigma: float | Sequence[float],
      order: int | Sequence[int] = 0,
      mode: str = 'reflect',
      cval: float | Sequence[float] = 0.0,
      truncate: float | Sequence[float] = 4.0,
      *,
      radius: None | Sequence[int] = None,
      axes: Sequence[int] = None,
      method="auto",
    ):
    """Gaussian filter for N-dimensional input
    
     Args:
        input: N-dimensional input array to filter.
        sigma: The standard deviation of the Gaussian filter.
        order: The order of the Gaussian filter.
        mode: The mode to use for padding the input array. See :func:`jax.numpy.pad` for more details.
        cval: The value to use for padding the input array.
        truncate: The number of standard deviations to include in the filter.
        radius: The radius of the filter. Overrides `truncate` if provided.
        method: The method to use for the convolution.
    """
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    orders = _ni_support._normalize_sequence(order, num_axes)
    sigmas = _ni_support._normalize_sequence(sigma, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    radii = _ni_support._normalize_sequence(radius, num_axes)

    # the loop goes over the input axes, so it is always low-dimensional and
    # keeping a Python loop is ok
    for idx in range(input.ndim):
       input = gaussian_filter1d(
          input,
          sigmas[idx],
          axis=idx,
          order=orders[idx],
          mode=modes[idx],
          cval=cval,
          truncate=truncate,
          radius=radii[idx],
          method=method,
       )

    return input

