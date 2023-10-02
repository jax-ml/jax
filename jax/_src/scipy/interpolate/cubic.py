# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import typing

import jax
import jax.numpy as jnp
import scipy.interpolate
from jax._src.numpy.util import _wraps

from jax._src.scipy.interpolate.ppoly import PPoly


@_wraps(scipy.interpolate.CubicHermiteSpline)
class CubicHermiteSpline(PPoly):
  """Piecewise-cubic interpolator matching values and first derivatives."""

  def __init__(self, x: jax.Array, y: jax.Array, dydx: jax.Array, axis: int = 0, extrapolate: typing.Optional[bool] = None):
    if extrapolate is None:
      extrapolate = True
    x, dx, y, axis, dydx = _prepare_input(x, y, axis, dydx)
    dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
    slope = jnp.diff(y, axis=0) / dxr
    t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr
    c = jnp.empty((4, len(x) - 1) + y.shape[1:], dtype=t.dtype)
    c = c.at[0].set(t / dxr)
    c = c.at[1].set((slope - dydx[:-1]) / dxr - t)
    c = c.at[2].set(dydx[:-1])
    c = c.at[3].set(y[:-1])
    super().__init__(c, x, extrapolate=extrapolate, axis=axis)


def _prepare_input(x: jax.Array, y: jax.Array, axis: int, dydx: typing.Optional[jax.Array] = None):
  if jnp.issubdtype(x.dtype, jnp.complexfloating):
    raise ValueError("`x` must contain real values.")
  if dydx is not None:
    if y.shape != dydx.shape:
      raise ValueError("The shapes of `y` and `dydx` must be identical.")
  axis = axis % y.ndim
  if x.ndim != 1:
    raise ValueError("`x` must be 1-dimensional.")
  if x.shape[0] < 2:
    raise ValueError("`x` must contain at least 2 elements.")
  if x.shape[0] != y.shape[axis]:
    raise ValueError(f"The length of `y` along `axis`={axis} doesn't match the length of `x`")
  dx = jnp.diff(x)
  # if jnp.any(dx <= 0):  # TracerBoolConversionError
  #   raise ValueError("`x` must be strictly increasing sequence.")
  y = jnp.moveaxis(y, axis, 0)
  if dydx is not None:
    dydx = jnp.moveaxis(dydx, axis, 0)
  return x, dx, y, axis, dydx
