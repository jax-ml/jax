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

import jax
from jax import lax
from jax import numpy as jnp
from jax._src.numpy.lax_numpy import _reduction_dims, _promote_args_inexact

# The definition of logsumexp is shared between jax.nn and jax.scipy, and
# although it matches scipy's definition, we put it here to avoid having
# unnecessary scipy dependencies.

def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
  r"""Log-sum-exp reduction.

  Computes

  .. math::
    \mathrm{logsumexp}(a) = \mathrm{log} \sum_j b \cdot \mathrm{exp}(a_{ij})

  where the :math:`j` indices range over one or more dimensions to be reduced.

  Args:
    a: the input array
    axis: the axis or axes over which to reduce. May be either ``None``, an
      int, or a tuple of ints.
    b: scaling factors for :math:`\mathrm{exp}(a)`. Must be broadcastable to the
      shape of `a`.
    keepdims: If ``True``, the axes that are reduced are left in the output as
      dimensions of size 1.
    return_sign: If ``True``, the output will be a ``(result, sign)`` pair,
      where ``sign`` is the sign of the sums and ``result`` contains the
      logarithms of their absolute values. If ``False`` only ``result`` is
      returned and it will contain NaN values if the sums are negative.

  Returns:
    Either an array ``result`` or a pair of arrays ``(result, sign)``, depending
    on the value of the ``return_sign`` argument.
  """
  if b is not None:
    a, b = _promote_args_inexact("logsumexp", a, b)
    a = jnp.where(b != 0, a, -jnp.inf)
  else:
    a, = _promote_args_inexact("logsumexp", a)
  pos_dims, dims = _reduction_dims(a, axis)
  amax = jnp.max(a, axis=dims, keepdims=keepdims)
  amax = lax.stop_gradient(lax.select(jnp.isfinite(amax), amax, lax.full_like(amax, 0)))
  amax_with_dims = amax if keepdims else lax.expand_dims(amax, pos_dims)
  # fast path if the result cannot be negative.
  if b is None and not jnp.issubdtype(a.dtype, jnp.complexfloating):
    out = lax.add(lax.log(jnp.sum(lax.exp(lax.sub(a, amax_with_dims)),
                                  axis=dims, keepdims=keepdims)),
                  amax)
    sign = jnp.where(jnp.isnan(out), out, 1.0)
    sign = jnp.where(jnp.isneginf(out), 0.0, sign).astype(out.dtype)
  else:
    expsub = lax.exp(lax.sub(a, amax_with_dims))
    if b is not None:
      expsub = lax.mul(expsub, b)
    sumexp = jnp.sum(expsub, axis=dims, keepdims=keepdims)

    sign = lax.stop_gradient(jnp.sign(sumexp))
    if jnp.issubdtype(sumexp.dtype, jnp.complexfloating):
      if return_sign:
        sumexp = sign*sumexp
      out = lax.add(lax.log(sumexp), amax)
    else:
      out = lax.add(lax.log(lax.abs(sumexp)), amax)
  if return_sign:
    return (out, sign)
  if b is not None:
    if not jnp.issubdtype(out.dtype, jnp.complexfloating):
      with jax.debug_nans(False):
        out = jnp.where(sign < 0, jnp.array(jnp.nan, dtype=out.dtype), out)
  return out
