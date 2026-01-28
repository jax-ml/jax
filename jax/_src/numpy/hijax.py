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

"""NumPy function implementations as hijax primitives."""
from typing import Any

from jax._src import ad_util
from jax._src import core
from jax._src import dtypes
from jax._src import numpy as jnp
from jax._src.hijax import VJPHiPrimitive
from jax._src.lax import lax
from jax._src.typing import Array, ArrayLike


class BinaryUfuncMixin:
  def __init__(
      self,
      x_aval: core.ShapedArray,
      y_aval: core.ShapedArray,
  ):
    if x_aval.dtype != y_aval.dtype:
      raise ValueError(f"{self.__class__.__name__}: inputs must have the same dtype."
                       f" Got x.dtype={x_aval.dtype} y.dtype={y_aval.dtype}")
    out_shape = lax.broadcasting_shape_rule(
      self.__class__.__name__, x_aval, y_aval)
    self.in_avals = (x_aval, y_aval)
    self.out_aval = x_aval.update(shape=out_shape)
    self.params: dict[str, Any] = {}
    super().__init__()

  def batch(
      self,
      _axis_data: Any,
      args: tuple[Array, Array],
      bdims: tuple[int | None, int | None]
  ) -> tuple[Array, int | None]:
    del _axis_data  # unused
    out_bdim: int | None
    x, y = args
    if bdims[0] is None and bdims[1] is None:
      out_bdim = None
    elif bdims[0] is not None and bdims[1] is None:
      y = lax.expand_dims(y, (bdims[0],))
      out_bdim = bdims[0]
    elif bdims[0] is None and bdims[1] is not None:
      x = lax.expand_dims(x, (bdims[1],))
      out_bdim = bdims[1]
    else:
      assert bdims[0] is not None
      assert bdims[1] is not None
      y = jnp.moveaxis(y, bdims[1], bdims[0])
      out_bdim = bdims[0]
    batched_caller = self.__class__(core.typeof(x), core.typeof(y))
    return batched_caller(x, y), out_bdim  # type: ignore[operator]


class NumpyMultiply(BinaryUfuncMixin, VJPHiPrimitive):
  """Hijax primitive for numpy.multiply."""
  def expand(self, x: ArrayLike, y: ArrayLike) -> Array:
    if dtypes.dtype(x) == bool:
      return lax.bitwise_and(x, y)
    return lax.mul(x, y)

  def jvp(self, primals: tuple[Array, Array], tangents: tuple[Array, Array]) -> tuple[Array | ad_util.Zero, Array | ad_util.Zero]:
    x, y = primals
    x_dot, y_dot = tangents
    return multiply(x, y), add(multiply(x, y_dot), multiply(x_dot, y))

  def vjp_fwd(self, nzs_in: Any, x: Array, y: Array) -> tuple[Array, tuple[Array, Array]]:
    del nzs_in  # unused
    return self(x, y), (x, y)

  def vjp_bwd_retval(self, res, g):
    x, y = res
    return multiply(g, y), multiply(x, g)


def multiply(x: ArrayLike, y: ArrayLike) -> Array:
  """np.multiply via hijax primitive."""
  return NumpyMultiply(core.typeof(x), core.typeof(y))(x, y)


class NumpyAdd(BinaryUfuncMixin, VJPHiPrimitive):
  """Hijax primitive for numpy.add."""
  def expand(self, x: ArrayLike, y: ArrayLike) -> Array:
    if dtypes.dtype(x) == bool:
      return lax.bitwise_or(x, y)
    return lax.add(x, y)

  def jvp(self, primals: tuple[Array, Array], tangents: tuple[Array, Array]) -> tuple[Array | ad_util.Zero, Array | ad_util.Zero]:
    return add(*primals), add(*tangents)

  def vjp_fwd(self, nzs_in: Any, x: Array, y: Array) -> tuple[Array, tuple[Array, Array]]:
    del nzs_in  # unused
    return self(x, y), (x, y)

  def vjp_bwd_retval(self, res, g):
    return g, g


def add(x: ArrayLike, y: ArrayLike) -> Array:
  """np.add via hijax primitive."""
  return NumpyAdd(core.typeof(x), core.typeof(y))(x, y)


class NumpySubtract(BinaryUfuncMixin, VJPHiPrimitive):
  """Hijax primitive for numpy.subtract."""
  def expand(self, x: ArrayLike, y: ArrayLike) -> Array:
    return lax.sub(x, y)

  def jvp(self, primals: tuple[Array, Array], tangents: tuple[Array, Array]) -> tuple[Array | ad_util.Zero, Array | ad_util.Zero]:
    return subtract(*primals), subtract(*tangents)

  def vjp_fwd(self, nzs_in: Any, x: Array, y: Array) -> tuple[Array, tuple[Array, Array]]:
    del nzs_in  # unused
    return self(x, y), (x, y)

  def vjp_bwd_retval(self, res, g):
    return g, -g


def subtract(x: ArrayLike, y: ArrayLike) -> Array:
  """np.subtract via hijax primitive."""
  return NumpySubtract(core.typeof(x), core.typeof(y))(x, y)
