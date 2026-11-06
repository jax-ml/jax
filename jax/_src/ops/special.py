# Copyright 2018 The JAX Authors.
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

from __future__ import annotations

from functools import partial
from typing import overload, Literal

from jax._src import api
from jax._src import config
from jax._src import custom_derivatives
from jax._src.lax import lax
from jax._src import numpy as jnp
from jax._src.numpy import reductions
from jax._src.numpy import ufuncs
from jax._src.numpy.reductions import _reduction_dims, Axis
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike

import numpy as np

# The definitions of logsumexp and softmax are shared between jax.nn and
# jax.scipy, and although they match scipy's definition, we put them here to
# avoid unnecessary dependencies.

# TODO(phawkins): this jit was found to change numerics in a test. Debug this.
# @api.jit(static_argnames=("axis",))
def softmax(x: ArrayLike,
            axis: Axis = -1,
            where: ArrayLike | None = None) -> Array:
  r"""Softmax function.

  Computes the function which rescales elements to the range :math:`[0, 1]`
  such that the elements along :code:`axis` sum to :math:`1`.

  .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  Args:
    x : input array
    axis: the axis or axes along which the softmax should be computed. The
      softmax output summed across these dimensions should sum to :math:`1`.
      Either an integer, tuple of integers, or ``None`` (all axes).
    where: Elements to include in the :code:`softmax`. The output for any
      masked-out element is zero.

  Returns:
    An array.

  Note:
    If any input values are ``+inf``, the result will be all ``NaN``: this reflects the
    fact that ``inf / inf`` is not well-defined in the context of floating-point math.

  See also:
    :func:`log_softmax`
  """
  if config.softmax_custom_jvp.value:
    return _softmax(x, axis, where)
  else:
    return _softmax_deprecated(x, axis, where)

# TODO(mattjj): replace softmax with _softmax when deprecation flag is removed
@partial(custom_derivatives.custom_jvp, nondiff_argnums=(1,))
def _softmax(
    x: ArrayLike,
    axis: Axis = -1,
    where: ArrayLike | None = None) -> Array:
  x_max = jnp.max(x, axis, where=where, initial=-np.inf, keepdims=True)
  x_safe = x if where is None else jnp.where(where, x, -np.inf)
  unnormalized = jnp.exp(x_safe - x_max)
  result = unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)
  if where is not None:
    result = jnp.where(where, result, 0)
  return result

@_softmax.defjvp
def _softmax_jvp(axis, primals, tangents):
  (x, where), (x_dot, _) = primals, tangents
  y = _softmax(x, axis, where)
  if where is not None:
    x_dot = jnp.where(where, x_dot, 0)
  y_dot = y * (x_dot - (y * x_dot).sum(axis, where=where, keepdims=True))
  if where is not None:
    y_dot = jnp.where(where, y_dot, 0)
  return y, y_dot

def _softmax_deprecated(
    x: ArrayLike,
    axis: Axis = -1,
    where: ArrayLike | None = None) -> Array:
  x_max = jnp.max(x, axis, where=where, initial=-np.inf, keepdims=True)
  x_safe = x if where is None else jnp.where(where, x, -np.inf)
  unnormalized = jnp.exp(x_safe - lax.stop_gradient(x_max))
  result = unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)
  if where is not None:
    result = jnp.where(where, result, 0)
  return result


@partial(custom_derivatives.custom_jvp, nondiff_argnums=(1, 3, 4))
def _logsumexp_impl(
    a: ArrayLike,
    axis: Axis = None,
    b: ArrayLike | None = None,
    keepdims: bool = False,
    return_sign: bool = False,
    where: ArrayLike | None = None,
) -> Array | tuple[Array, Array]:
  if where is not None:
    a = jnp.where(where, a, 0)
  if b is not None:
    a_arr, b_arr = promote_args_inexact("logsumexp", a, b)
    a_arr = jnp.where(b_arr != 0, a_arr, -np.inf)
  else:
    a_arr, = promote_args_inexact("logsumexp", a)
    b_arr = a_arr  # for type checking
  pos_dims, dims = _reduction_dims(a_arr, axis)
  amax = reductions.max(a_arr.real, axis=dims, keepdims=keepdims, where=where, initial=-np.inf)
  amax = lax.stop_gradient(lax.select(ufuncs.isfinite(amax), amax, lax.full_like(amax, 0)))
  amax_with_dims = amax if keepdims else lax.expand_dims(amax, pos_dims)

  exp_a = lax.exp(lax.sub(a_arr, amax_with_dims.astype(a_arr.dtype)))
  if b is not None:
    exp_a = lax.mul(exp_a, b_arr)
  sumexp = exp_a.sum(axis=dims, keepdims=keepdims, where=where)
  sign = lax.sign(sumexp)
  if return_sign or not np.issubdtype(a_arr.dtype, np.complexfloating):
    sumexp = abs(sumexp)
  out = lax.add(lax.log(sumexp), amax.astype(sumexp.dtype))

  if return_sign:
    return (out, sign)
  if b is not None and not np.issubdtype(out.dtype, np.complexfloating):
    with config.debug_nans(False):
      out = jnp.where(sign < 0, jnp.array(np.nan, dtype=out.dtype), out)
  return out


@_logsumexp_impl.defjvp
def _logsumexp_impl_jvp(axis, keepdims, return_sign, primals, tangents):
  a, b, where = primals
  a_dot, b_dot, _ = tangents
  if b is None and not np.issubdtype(jnp.asarray(a).dtype, np.complexfloating):
    primal_out = _logsumexp_impl(a, axis, b, keepdims, return_sign, where)
    s = softmax(a, axis=axis, where=where)
    tangent_out = reductions.sum(s * a_dot, axis=axis, keepdims=keepdims)
    if return_sign:
      _, sign = primal_out
      return primal_out, (tangent_out, lax.full_like(sign, 0))
    return primal_out, tangent_out
  elif b is None:
    f = lambda a_: _logsumexp_impl.fun(
        a_, axis, None, keepdims, return_sign, where
    )
    return api.jvp(f, (a,), (a_dot,))
  else:
    f = lambda a_, b_: _logsumexp_impl.fun(
        a_, axis, b_, keepdims, return_sign, where
    )
    return api.jvp(f, (a, b), (a_dot, b_dot))


@overload
def logsumexp(a: ArrayLike, axis: Axis = None, b: ArrayLike | None = None,
              keepdims: bool = False, return_sign: Literal[False] = False, where: ArrayLike | None = None) -> Array: ...

@overload
def logsumexp(a: ArrayLike, axis: Axis = None, b: ArrayLike | None = None,
              keepdims: bool = False, *, return_sign: Literal[True], where: ArrayLike | None = None) -> tuple[Array, Array]: ...

@overload
def logsumexp(a: ArrayLike, axis: Axis = None, b: ArrayLike | None = None,
              keepdims: bool = False, return_sign: bool = False, where: ArrayLike | None = None) -> Array | tuple[Array, Array]: ...

def logsumexp(a: ArrayLike, axis: Axis = None, b: ArrayLike | None = None,
              keepdims: bool = False, return_sign: bool = False, where: ArrayLike | None = None) -> Array | tuple[Array, Array]:
  r"""Log-sum-exp reduction.

  JAX implementation of :func:`scipy.special.logsumexp`.

  .. math::
    \operatorname{logsumexp} a = \log \sum_i b_i \exp a_i

  where the :math:`i` indices range over one or more dimensions to be reduced.

  Args:
    a: the input array
    axis: int or sequence of ints, default=None. Axis along which the sum to be
      computed. If None, the sum is computed along all the axes.
    b: scaling factors for the exponentials. Must be broadcastable to the shape of `a`.
    keepdims: If ``True``, the axes that are reduced are left in the output as
      dimensions of size 1.
    return_sign: If ``True``, the output will be a ``(result, sign)`` pair,
      where ``sign`` is the sign of the sums and ``result`` contains the
      logarithms of their absolute values. If ``False`` only ``result`` is
      returned and it will contain NaN values if the sums are negative.
    where: Elements to include in the reduction.

  Returns:
    Either an array ``result`` or a pair of arrays ``(result, sign)``, depending
    on the value of the ``return_sign`` argument.

  See also:
    :func:`jax.nn.logmeanexp`
  """
  return _logsumexp_impl(a, axis, b, keepdims, return_sign, where)
