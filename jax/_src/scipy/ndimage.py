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

import numpy as np

from jax._src import api
from jax._src import dtypes
from jax._src import numpy as jnp
from jax._src import util
from jax._src.lax import lax
from jax._src.typing import ArrayLike, Array
from jax._src.util import safe_zip as zip


def _nonempty_prod(arrs: Sequence[Array]) -> Array:
  return functools.reduce(operator.mul, arrs)

def _nonempty_sum(arrs: Sequence[Array]) -> Array:
  return sum(arrs[1:], arrs[0])

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
  return a if dtypes.issubdtype(a.dtype, np.integer) else lax.round(a)


def _round_half_to_posinf(a: Array) -> Array:
  return a if dtypes.issubdtype(a.dtype, np.integer) else lax.floor(a + 0.5)


def _filter_index_and_weight(coordinate: Array, even: bool = False) -> tuple[Array, Array]:
  lower = jnp.floor(coordinate + 0.5 if even else coordinate)
  lower_dist = coordinate - lower
  # (index, dist to lower knot)
  return (lower.astype(np.int32), lower_dist)


def _nearest_indices_and_weights(coordinate: Array) -> list[tuple[Array, Array]]:
  index = _round_half_to_posinf(coordinate).astype(np.int32)
  weight = coordinate.dtype.type(1)
  return [(index, weight)]


def _linear_indices_and_weights(coordinate: Array) -> list[tuple[Array, Array]]:
  (index, lower_dist) = _filter_index_and_weight(coordinate)
  return [(index, 1 - lower_dist), (index + 1, lower_dist)]


def _quadratic_indices_and_weights(coordinate: Array) -> list[tuple[Array, Array]]:
  (index, t) = _filter_index_and_weight(coordinate, even=True)
  # t from -0.5 to 0.5
  return [
    (index - 1, 0.5 * (0.5 - t)**2),
    (index,     0.75 - t * t),
    (index + 1, 0.5 * (t + 0.5)**2),
  ]


def _cubic_indices_and_weights(coordinate: Array) -> list[tuple[Array, Array]]:
  (index, t) = _filter_index_and_weight(coordinate)
  t1 = 1 - t
  return [
    (index - 1, t1 * t1 * t1 / 6.),
    (index,     (4. + 3. * t * t * (t - 2.0)) / 6.),
    (index + 1, (4. + 3. * t1 * t1 * (t1 - 2.0)) / 6.),
    (index + 2, t * t * t / 6.),
  ]


def _quartic_indices_and_weights(coordinate: Array) -> list[tuple[Array, Array]]:
  (index, t) = _filter_index_and_weight(coordinate, even=True)
  t_sq = t**2
  y = t + 1
  t1 = 1 - t
  return [
    (index - 2, (0.5 - t)**4 / 24.0),
    (index - 1, y * (y * (y * (5.0 - y) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0),
    (index,     t_sq * (t_sq * 0.25 - 0.625) + 115.0 / 192.0),
    (index + 1, t1 * (t1 * (t1 * (5.0 - t1) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0),
    (index + 2, (t + 0.5)**4 / 24.0),
  ]


def _quintic_indices_and_weights(coordinate: Array) -> list[tuple[Array, Array]]:
  (index, t) = _filter_index_and_weight(coordinate)
  t1 = 1 - t
  t_sq = t * t
  t1_sq = t1 * t1
  y = t + 1
  y1 = t1 + 1
  return [
    (index - 2, t1 * t1_sq * t1_sq / 120.0),
    (index - 1, y * (y * (y * (y * (y / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425),
    (index,     t_sq * (t_sq * (0.25 - t / 12.0) - 0.5) + 0.55),
    (index + 1, t1_sq * (t1_sq * (0.25 - t1 / 12.0) - 0.5) + 0.55),
    (index + 2, y1 * (y1 * (y1 * (y1 * (y1 / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425),
    (index + 3, t * t_sq * t_sq / 120.0),
  ]


_INTERP_FNS: dict[int, Callable[[Array], list[tuple[Array, Array]]]] = {
  0: _nearest_indices_and_weights,
  1: _linear_indices_and_weights,
  2: _quadratic_indices_and_weights,
  3: _cubic_indices_and_weights,
  4: _quartic_indices_and_weights,
  5: _quintic_indices_and_weights,
}


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

  interp_fun = _INTERP_FNS.get(int(order))
  if interp_fun is None:
    raise NotImplementedError(
        'jax.scipy.ndimage.map_coordinates does not yet support order {}. '
        'Currently supported orders are {}.'.format(int(order), set(_INTERP_FNS)))

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
  if dtypes.issubdtype(input_arr.dtype, np.integer):
    result = _round_half_away_from_zero(result)
  return result.astype(input_arr.dtype)


def map_coordinates(
    input: ArrayLike, coordinates: Sequence[ArrayLike], order: int,
    mode: str = 'constant', cval: ArrayLike = 0.0, prefilter: bool = True,
) -> Array:
  """
  Map the input array to new coordinates using interpolation.

  JAX implementation of :func:`scipy.ndimage.map_coordinates`

  Given an input array and a set of coordinates, this function returns the
  interpolated values of the input array at those coordinates.

  Args:
    input: N-dimensional input array from which values are interpolated.
    coordinates: length-N sequence of arrays specifying the coordinates
      at which to evaluate the interpolated values
    order: The order of interpolation. JAX supports orders 0-5, where 0 is nearest-neighbor
      interpolation, 1 is linear interpolation, 3 is cubic interpolation, etc.
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
    prefilter: Determines if the array is prefiltered with :func:`spline_prefilter` before
      use. The default is `True`. Only has an effect for ``order > 1``.

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
  if order > 1 and prefilter:
    if mode in ('nearest', 'constant'):
      raise NotImplementedError("requires prepadding")
    input = spline_filter(jnp.asarray(input).astype(float), order, mode)

  return _map_coordinates(input, coordinates, order, mode, cval)


def _init_mirror_causal(arr: Array, z: float) -> Array:
  idx = jnp.arange(0, arr.size - 1, dtype=arr.dtype)
  z_n = z**(arr.dtype.type(arr.size) - 1)
  return (
    jnp.sum(z**idx * (arr[:-1] + z_n * arr[:0:-1]))
  ) / (1 - z_n**2)

def _init_mirror_anticausal(arr: Array, z: float) -> Array:
  return z / (z**2 - 1) * (z * arr[-2] + arr[-1])

def _init_wrap_causal(arr: Array, z: float) -> Array:
  idx = jnp.arange(1, arr.size, dtype=arr.dtype)
  return (
    arr[0] + jnp.sum(z**idx * arr[:0:-1])
  ) / (1 - z**arr.size)

def _init_wrap_anticausal(arr: Array, z: float) -> Array:
  idx = jnp.arange(1, arr.size, dtype=arr.dtype)
  return (
    arr[-1] + jnp.sum(z**idx * arr[:-1])
  ) * z / (z**arr.size - 1)

def _init_reflect_causal(arr: Array, z: float) -> Array:
  idx = jnp.arange(arr.size, dtype=arr.dtype)
  z_n = z**arr.dtype.type(arr.size)
  return arr[0] + z / (1 - z_n**2) * jnp.sum(z**idx * (arr + z_n * arr[::-1]))

def _init_reflect_anticausal(arr: Array, z: float) -> Array:
  return z / (z - 1) * arr[-1]

_SPLINE_BOUNDARY_FNS: dict[str, tuple[Callable[[Array, float], Array], Callable[[Array, float], Array]]] = {
  'reflect': (_init_reflect_causal, _init_reflect_anticausal),
  'wrap': (_init_wrap_causal, _init_wrap_anticausal),
  'mirror': (_init_mirror_causal, _init_mirror_anticausal),
  # closest b.c. to nearest
  'nearest': (_init_reflect_causal, _init_reflect_anticausal),
  # default to mirror boundary
  'constant': (_init_mirror_causal, _init_mirror_anticausal),
}

_SPLINE_FILTER_POLES: dict[int, list[float]] = {
  2: [-0.171572875253809902396622551580603843],
  3: [-0.267949192431122706472553658494127633],
  4: [-0.361341225900220177092212841325675255, -0.013725429297339121360331226939128204],
  5: [-0.430575347099973791851434783493520110, -0.043096288203264653822712376822550182],
}


@functools.partial(api.jit, static_argnums=(1, 2, 3))
def _spline_filter1d(
    input: Array, order: int, axis: int, mode: str = 'mirror',
) -> Array:
  from jax._src.lax.control_flow.loops import associative_scan

  poles = _SPLINE_FILTER_POLES.get(order)
  if poles is None:
    raise ValueError("Spline order '{}' not supported for pre-filtering".format(order))

  (causal_fn, anticausal_fn) = _SPLINE_BOUNDARY_FNS.get(mode, (None, None))
  if causal_fn is None or anticausal_fn is None:
    raise ValueError("Boundary mode '{}' not supported for pre-filtering".format(mode))

  gain = functools.reduce(operator.mul, (
    (1.0 - z) * (1.0 - 1.0 / z) for z in poles
  ))
  arr = input.astype(float) * gain

  # compose an affine transform (y = k*x + b)
  # t1 @ t0 => y = (k0*k1)*x + (b0 + k0*b1)
  def compose_affine(t1: tuple[Array, Array], t0: tuple[Array, Array]) -> tuple[Array, Array]:
    return (t0[0] * t1[0], t0[1] + t0[0]*t1[1])

  #import jax

  for z in poles:
    #jax.debug.print("pole: {}", z)
    # causal
    init = jnp.apply_along_axis(lambda arr: jnp.array([causal_fn(arr, z)]), axis, arr)
    #jax.debug.print("causal init: {}", init)
    arr_rest = lax.slicing.slice_in_dim(arr, 1, None, axis=axis)
    K, B = associative_scan(compose_affine, (jnp.full_like(arr_rest, z), arr_rest), axis=axis)
    arr = lax.concatenate([init, K * init + B], axis)
    #jax.debug.print("after causal: {}", arr)

    # anticausal
    init = jnp.apply_along_axis(lambda arr: jnp.array([anticausal_fn(arr, z)]), axis, arr)
    #jax.debug.print("anticausal init: {}", init)
    arr_rest = lax.slicing.slice_in_dim(arr, None, -1, axis=axis)
    K, B = associative_scan(compose_affine, (jnp.full_like(arr_rest, z), -z * arr_rest), axis=axis, reverse=True)
    arr = lax.concatenate([K * init + B, init], axis)
    #jax.debug.print("after anticausal: {}", arr)

  if dtypes.issubdtype(input.dtype, np.integer):
    arr = _round_half_away_from_zero(arr)
  return arr.astype(input.dtype)


def spline_filter(
    input: ArrayLike,
    order: int = 3,
    mode: str = 'mirror',
) -> Array:
  """
  Applies a multidimensional spline pre-filter.

  JAX implementation of :func:`scipy.ndimage.spline_filter`.

  Given an input array, this function pre-calculates the B-spline coefficients
  for an interpolation with the given order and boundary conditions. These
  coefficients can then be consumed by interpolation functions with ``prefilter=False``.

  Args:
    input: N-dimensional input array for which prefiltering is performed
    order: The order of the spline. Supported orders are 2-5.
    mode: Boundary mode to use. See :func:`map_coordinates` for more details.
      Modes 'nearest' and 'constant' cannot be used, as they have no analytic
      solution for the prefilter. Instead, pad the array by the filter size
      prior to pre-filtering.

  Returns:
    An array of B-spline coefficients with the same shape and dtype as ``input``.
  """
  arr = jnp.asarray(input)

  for ax in range(arr.ndim):
    arr = spline_filter1d(arr, order, ax, mode)
  return arr


def spline_filter1d(
    input: ArrayLike,
    order: int = 3,
    axis: int = -1,
    mode: str = 'mirror',
) -> Array:
  """
  Applies a one-dimensional spline pre-filter.

  JAX implementation of :func:`scipy.ndimage.spline_filter1d`.

  Given an input array, this function pre-calculates the B-spline coefficients
  for an interpolation with the given order and boundary conditions along the given axis.
  These coefficients can then be consumed by interpolation functions with ``prefilter=False``.

  Args:
    input: N-dimensional input array for which prefiltering is performed
    order: The order of the spline. Supported orders are 2-5.
    axis: Axis to apply the spline filter along.
    mode: Boundary mode to use. See :func:`map_coordinates` for more details.
      Modes 'nearest' and 'constant' cannot be used, as they have no analytic
      solution for the prefilter. Instead, pad the array by the filter size
      prior to pre-filtering.

  Returns:
    An array of B-spline coefficients with the same shape and dtype as ``input``.
  """
  if mode in ('nearest', 'constant'):
    raise ValueError("Boundary mode '{}' has no exact filter. "
                     "Instead, pad the array by the filter size "
                     "and use mode 'mirror'".format(mode))
  input = jnp.asarray(input)
  axis = util.canonicalize_axis(axis, input.ndim)
  return _spline_filter1d(input, order, axis, mode)
