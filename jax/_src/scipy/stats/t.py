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


from jax._src import lax
from jax._src import numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact
from jax._src.scipy.special import betaln
from jax._src.typing import Array, ArrayLike


def logpdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Student's T log probability distribution function.

  JAX implementation of :obj:`scipy.stats.t` ``logpdf``.

  The Student's T probability distribution function is given by

  .. math::

     f(x, \nu) = \frac{\Gamma((\nu + 1)/2)}{\sqrt{\pi\nu}\Gamma(\nu/2)}(1 + x^2/\nu)^{(\nu+1)/2}

  Where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function, and :math:`\nu > 0`
  is the degrees of freedom (JAX follows the scipy convention of naming this ``df``).

  Args:
    x: arraylike, value at which to evaluate the PDF
    df: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array of logpdf values.

  See Also:
    :func:`jax.scipy.stats.t.pdf`
  """
  x, df, loc, scale = promote_args_inexact("t.logpdf", x, df, loc, scale)
  half = _lax_const(x, 0.5)
  delta = lax.sub(x, loc)
  df_over_two = lax.mul(df, half)
  df_plus_one_over_two = lax.add(df_over_two, half)
  # Avoid under/overflow of scale**2 * df before taking its logarithm.
  normalize_term_tmp = lax.add(lax.log(lax.abs(scale)), lax.mul(lax.log(df), half))
  normalize_term = lax.add(betaln(df_over_two, half), normalize_term_tmp)

  # Even when the normalization is stable, ((x - loc) / scale)**2
  # can overflow while the log density remains finite. Preserve the fast
  # ordinary-input path; use a log-domain formula only if the batch contains
  # an extreme standardized input or a scale/df that risks overflow.
  def regular_quadratic(_):
    scaled_x = lax.div(delta, scale)
    return lax.log1p(lax.div(lax.mul(scaled_x, scaled_x), df))

  def stable_quadratic(_):
    abs_delta = lax.abs(delta)
    safe_abs_delta = jnp.where(abs_delta == 0, 1, abs_delta)
    log_abs_scaled = lax.sub(lax.log(safe_abs_delta), lax.log(lax.abs(scale)))
    log_quadratic = lax.sub(
        lax.mul(_lax_const(x, 2), log_abs_scaled), lax.log(df)
    )
    stable_tail = lax.add(
        jnp.maximum(log_quadratic, 0),
        lax.log1p(lax.exp(lax.neg(lax.abs(log_quadratic)))),
    )
    return jnp.where(abs_delta == 0, 0, stable_tail)

  use_stable = jnp.any(
      (lax.abs(scale) < _lax_const(x, 1e-16))
      | (lax.abs(delta) > lax.mul(lax.abs(scale), _lax_const(x, 1e12)))
      | (df < _lax_const(x, 1e-5))
  )
  log_one_plus_quadratic = lax.cond(
      use_stable, stable_quadratic, regular_quadratic, operand=None
  )
  return lax.neg(lax.add(
      normalize_term, lax.mul(df_plus_one_over_two, log_one_plus_quadratic)
  ))


def pdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  r"""Student's T probability distribution function.

  JAX implementation of :obj:`scipy.stats.t` ``pdf``.

  The Student's T probability distribution function is given by

  .. math::

     f(x, \nu) = \frac{\Gamma((\nu + 1)/2)}{\sqrt{\pi\nu}\Gamma(\nu/2)}(1 + x^2/\nu)^{(\nu+1)/2}

  Where :math:`\Gamma` is the :func:`~jax.scipy.special.gamma` function, and :math:`\nu > 0`
  is the degrees of freedom (JAX follows the scipy convention of naming this ``df``).

  Args:
    x: arraylike, value at which to evaluate the PDF
    df: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter
    scale: arraylike, distribution scale parameter

  Returns:
    array

  See Also:
    :func:`jax.scipy.stats.t.logpdf`
  """
  return lax.exp(logpdf(x, df, loc, scale))
