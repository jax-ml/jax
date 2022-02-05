# Copyright 2018 Google LLC
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

from functools import partial

import numpy as np
import scipy.special as osp_special

import jax
from jax._src import api
from jax import jit
from jax import lax, core
from jax.interpreters import ad
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy.lax_numpy import (asarray, _reduction_dims, _constant_like,
                                      _promote_args_inexact)
from jax._src.numpy.util import _wraps

from typing import Optional, Tuple


@_wraps(osp_special.gammaln)
def gammaln(x):
  x, = _promote_args_inexact("gammaln", x)
  return lax.lgamma(x)


@_wraps(osp_special.betaln)
def betaln(x, y):
  x, y = _promote_args_inexact("betaln", x, y)
  return lax.lgamma(x) + lax.lgamma(y) - lax.lgamma(x + y)


@_wraps(osp_special.betainc)
def betainc(a, b, x):
  a, b, x = _promote_args_inexact("betainc", a, b, x)
  return lax.betainc(a, b, x)


@_wraps(osp_special.digamma, lax_description="""\
The JAX version only accepts real-valued inputs.""")
def digamma(x):
  x, = _promote_args_inexact("digamma", x)
  return lax.digamma(x)
ad.defjvp(lax.digamma_p, lambda g, x: lax.mul(g, polygamma(1, x)))


@_wraps(osp_special.gammainc, update_doc=False)
def gammainc(a, x):
  a, x = _promote_args_inexact("gammainc", a, x)
  return lax.igamma(a, x)


@_wraps(osp_special.gammaincc, update_doc=False)
def gammaincc(a, x):
  a, x = _promote_args_inexact("gammaincc", a, x)
  return lax.igammac(a, x)


@_wraps(osp_special.erf)
def erf(x):
  x, = _promote_args_inexact("erf", x)
  return lax.erf(x)


@_wraps(osp_special.erfc, update_doc=False)
def erfc(x):
  x, = _promote_args_inexact("erfc", x)
  return lax.erfc(x)


@_wraps(osp_special.erfinv)
def erfinv(x):
  x, = _promote_args_inexact("erfinv", x)
  return lax.erf_inv(x)


@api.custom_jvp
@_wraps(osp_special.logit, update_doc=False)
def logit(x):
  x = asarray(x)
  return lax.log(lax.div(x, lax.sub(lax._const(x, 1), x)))
logit.defjvps(
    lambda g, ans, x: lax.div(g, lax.mul(x, lax.sub(lax._const(x, 1), x))))


@api.custom_jvp
@_wraps(osp_special.expit, update_doc=False)
def expit(x):
  x = asarray(x)
  one = lax._const(x, 1)
  return lax.div(one, lax.add(one, lax.exp(lax.neg(x))))
expit.defjvps(lambda g, ans, x: g * ans * (lax._const(ans, 1) - ans))


@_wraps(osp_special.logsumexp)
def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
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
  if b is None and not np.issubdtype(a.dtype, np.complexfloating):
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
    if np.issubdtype(sumexp.dtype, np.complexfloating):
      if return_sign:
        sumexp = sign*sumexp
      out = lax.add(lax.log(sumexp), amax)
    else:
      out = lax.add(lax.log(lax.abs(sumexp)), amax)
  if return_sign:
    return (out, sign)
  if b is not None:
    if not np.issubdtype(out.dtype, np.complexfloating):
      with jax.debug_nans(False):
        out = jnp.where(sign < 0, jnp.array(np.nan, dtype=out.dtype), out)
  return out


@_wraps(osp_special.xlogy)
def xlogy(x, y):
  x, y = _promote_args_inexact("xlogy", x, y)
  x_ok = x != 0.
  safe_x = jnp.where(x_ok, x, 1.)
  safe_y = jnp.where(x_ok, y, 1.)
  return jnp.where(x_ok, lax.mul(safe_x, lax.log(safe_y)), jnp.zeros_like(x))


@_wraps(osp_special.xlog1py, update_doc=False)
def xlog1py(x, y):
  x, y = _promote_args_inexact("xlog1py", x, y)
  x_ok = x != 0.
  safe_x = jnp.where(x_ok, x, 1.)
  safe_y = jnp.where(x_ok, y, 1.)
  return jnp.where(x_ok, lax.mul(safe_x, lax.log1p(safe_y)), jnp.zeros_like(x))


@_wraps(osp_special.entr)
def entr(x):
  x, = _promote_args_inexact("entr", x)
  return lax.select(lax.lt(x, _constant_like(x, 0)),
                    lax.full_like(x, -np.inf),
                    lax.neg(xlogy(x, x)))


@_wraps(osp_special.multigammaln, update_doc=False)
def multigammaln(a, d):
  d = core.concrete_or_error(int, d, "d argument of multigammaln")
  a, d_ = _promote_args_inexact("multigammaln", a, d)

  constant = lax.mul(lax.mul(lax.mul(_constant_like(a, 0.25), d_),
                             lax.sub(d_, _constant_like(a, 1))),
                     lax.log(_constant_like(a, np.pi)))
  b = lax.div(jnp.arange(d, dtype=d_.dtype), _constant_like(a, 2))
  res = jnp.sum(gammaln(jnp.expand_dims(a, axis=-1) -
                        jnp.expand_dims(b, axis=tuple(range(a.ndim)))),
                axis=-1)
  return res + constant


# coefs of (2k)! / B_{2k} where B are bernoulli numbers
# those numbers are obtained using https://www.wolframalpha.com
_BERNOULLI_COEFS = [
    12,
    -720,
    30240,
    -1209600,
    47900160,
    -1307674368000 / 691,
    74724249600,
    -10670622842880000 / 3617,
    5109094217170944000 / 43867,
    -802857662698291200000 / 174611,
    14101100039391805440000 / 77683,
    -1693824136731743669452800000 / 236364091,
    186134520519971831808000000 / 657931,
    -37893265687455865519472640000000 / 3392780147,
    759790291646040068357842010112000000 / 1723168255201,
    -134196726836183700385281186201600000000 / 7709321041217,
]


@_wraps(osp_special.zeta)
def zeta(x, q=None):
  assert q is not None, "Riemann zeta function is not implemented yet."
  # Reference: Johansson, Fredrik.
  # "Rigorous high-precision computation of the Hurwitz zeta function and its derivatives."
  # Numerical Algorithms 69.2 (2015): 253-270.
  # https://arxiv.org/abs/1309.2877 - formula (5)
  # here we keep the same notation as in reference
  s, a = _promote_args_inexact("zeta", x, q)
  dtype = lax.dtype(a).type
  s_, a_ = jnp.expand_dims(s, -1), jnp.expand_dims(a, -1)
  # precision ~ N, M
  N = M = dtype(8) if lax.dtype(a) == jnp.float32 else dtype(16)
  assert M <= len(_BERNOULLI_COEFS)
  k = jnp.expand_dims(np.arange(N, dtype=N.dtype), tuple(range(a.ndim)))
  S = jnp.sum((a_ + k) ** -s_, -1)
  I = lax.div((a + N) ** (dtype(1) - s), s - dtype(1))
  T0 = (a + N) ** -s
  m = jnp.expand_dims(np.arange(2 * M, dtype=M.dtype), tuple(range(s.ndim)))
  s_over_a = (s_ + m) / (a_ + N)
  T1 = jnp.cumprod(s_over_a, -1)[..., ::2]
  T1 = jnp.clip(T1, a_max=jnp.finfo(dtype).max)
  coefs = np.expand_dims(np.array(_BERNOULLI_COEFS[:T1.shape[-1]], dtype=dtype),
                         tuple(range(a.ndim)))
  T1 = T1 / coefs
  T = T0 * (dtype(0.5) + T1.sum(-1))
  return S + I + T


@_wraps(osp_special.polygamma, update_doc=False)
def polygamma(n, x):
  assert jnp.issubdtype(lax.dtype(n), jnp.integer)
  n, x = _promote_args_inexact("polygamma", n, x)
  shape = lax.broadcast_shapes(n.shape, x.shape)
  return _polygamma(jnp.broadcast_to(n, shape), jnp.broadcast_to(x, shape))


@api.custom_jvp
def _polygamma(n, x):
  dtype = lax.dtype(n).type
  n_plus = n + dtype(1)
  sign = dtype(1) - (n_plus % dtype(2)) * dtype(2)
  return jnp.where(n == 0, digamma(x), sign * jnp.exp(gammaln(n_plus)) * zeta(n_plus, x))
_polygamma.defjvps(None, lambda g, ans, n, x: lax.mul(g, _polygamma(n + 1, x)))


# Normal distributions

# Functions "ndtr" and "ndtri" are derived from calculations made in:
# https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
# In the following email exchange, the author gives his consent to redistribute
# derived works under an Apache 2.0 license.
#
# From: Stephen Moshier <steve@moshier.net>
# Date: Sat, Jun 9, 2018 at 2:36 PM
# Subject: Re: Licensing cephes under Apache (BSD-like) license.
# To: rif <rif@google.com>
#
#
#
# Hello Rif,
#
# Yes, Google may distribute Cephes files under the Apache 2 license.
#
# If clarification is needed, I do not favor BSD over other free licenses.
# I would agree that Apache 2 seems to cover the concern you mentioned
# about sublicensees.
#
# Best wishes for good luck with your projects!
# Steve Moshier
#
#
#
# On Thu, 31 May 2018, rif wrote:
#
# > Hello Steve.
# > My name is Rif. I work on machine learning software at Google.
# >
# > Your cephes software continues to be incredibly useful and widely used. I
# > was wondering whether it would be permissible for us to use the Cephes code
# > under the Apache 2.0 license, which is extremely similar in permissions to
# > the BSD license (Wikipedia comparisons). This would be quite helpful to us
# > in terms of avoiding multiple licenses on software.
# >
# > I'm sorry to bother you with this (I can imagine you're sick of hearing
# > about this by now), but I want to be absolutely clear we're on the level and
# > not misusing your important software. In former conversation with Eugene
# > Brevdo (ebrevdo@google.com), you wrote "If your licensing is similar to BSD,
# > the formal way that has been handled is simply to add a statement to the
# > effect that you are incorporating the Cephes software by permission of the
# > author." I wanted to confirm that (a) we could use the Apache license, (b)
# > that we don't need to (and probably you don't want to) keep getting
# > contacted about individual uses, because your intent is generally to allow
# > this software to be reused under "BSD-like" license, and (c) you're OK
# > letting incorporators decide whether a license is sufficiently BSD-like?
# >
# > Best,
# >
# > rif
# >
# >
# >

# log_ndtr uses different functions over the ranges
# (-infty, lower](lower, upper](upper, infty)
# Lower bound values were chosen by examining where the support of ndtr
# appears to be zero, relative to scipy's (which is always 64bit). They were
# then made more conservative just to be safe. (Conservative means use the
# expansion more than we probably need to.)
_LOGNDTR_FLOAT64_LOWER = np.array(-20, np.float64)
_LOGNDTR_FLOAT32_LOWER = np.array(-10, np.float32)

# Upper bound values were chosen by examining for which values of 'x'
# Log[cdf(x)] is 0, after which point we need to use the approximation
# Log[cdf(x)] = Log[1 - cdf(-x)] approx -cdf(-x). We chose a value slightly
# conservative, meaning we use the approximation earlier than needed.
_LOGNDTR_FLOAT64_UPPER = np.array(8, np.float64)
_LOGNDTR_FLOAT32_UPPER = np.array(5, np.float32)


def ndtr(x):
  r"""Normal distribution function.

  Returns the area under the Gaussian probability density function, integrated
  from minus infinity to x:

  .. math::
    \begin{align}
    \mathrm{ndtr}(x) =&
      \ \frac{1}{\sqrt{2 \pi}}\int_{-\infty}^{x} e^{-\frac{1}{2}t^2} dt \\
    =&\ \frac{1}{2} (1 + \mathrm{erf}(\frac{x}{\sqrt{2}})) \\
    =&\ \frac{1}{2} \mathrm{erfc}(\frac{x}{\sqrt{2}})
    \end{align}

  Args:
    x: An array of type `float32`, `float64`.

  Returns:
    An array with `dtype=x.dtype`.

  Raises:
    TypeError: if `x` is not floating-type.
  """
  x = jnp.asarray(x)
  dtype = lax.dtype(x)
  if dtype not in (jnp.float32, jnp.float64):
    raise TypeError(
        "x.dtype={} is not supported, see docstring for supported types."
        .format(dtype))
  return _ndtr(x)


def _ndtr(x):
  """Implements ndtr core logic."""
  dtype = lax.dtype(x).type
  half_sqrt_2 = dtype(0.5) * np.sqrt(2., dtype=dtype)
  w = x * half_sqrt_2
  z = lax.abs(w)
  y = lax.select(lax.lt(z, half_sqrt_2),
                      dtype(1.) + lax.erf(w),
                      lax.select(lax.gt(w, dtype(0.)),
                                      dtype(2.) - lax.erfc(z),
                                      lax.erfc(z)))
  return dtype(0.5) * y


def ndtri(p):
  r"""The inverse of the CDF of the Normal distribution function.

  Returns `x` such that the area under the PDF from :math:`-\infty` to `x` is equal
  to `p`.

  A piece-wise rational approximation is done for the function.
  This is a based on the implementation in netlib.

  Args:
    p: an array of type `float32`, `float64`.

  Returns:
    an array with `dtype=p.dtype`.

  Raises:
    TypeError: if `p` is not floating-type.
  """
  dtype = lax.dtype(p)
  if dtype not in (jnp.float32, jnp.float64):
    raise TypeError(
        "x.dtype={} is not supported, see docstring for supported types."
        .format(dtype))
  return _ndtri(p)


def _ndtri(p):
  """Implements ndtri core logic."""

  # Constants used in piece-wise rational approximations. Taken from the cephes
  # library:
  # https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
  p0 = list(reversed([-5.99633501014107895267E1,
                      9.80010754185999661536E1,
                      -5.66762857469070293439E1,
                      1.39312609387279679503E1,
                      -1.23916583867381258016E0]))
  q0 = list(reversed([1.0,
                      1.95448858338141759834E0,
                      4.67627912898881538453E0,
                      8.63602421390890590575E1,
                      -2.25462687854119370527E2,
                      2.00260212380060660359E2,
                      -8.20372256168333339912E1,
                      1.59056225126211695515E1,
                      -1.18331621121330003142E0]))
  p1 = list(reversed([4.05544892305962419923E0,
                      3.15251094599893866154E1,
                      5.71628192246421288162E1,
                      4.40805073893200834700E1,
                      1.46849561928858024014E1,
                      2.18663306850790267539E0,
                      -1.40256079171354495875E-1,
                      -3.50424626827848203418E-2,
                      -8.57456785154685413611E-4]))
  q1 = list(reversed([1.0,
                      1.57799883256466749731E1,
                      4.53907635128879210584E1,
                      4.13172038254672030440E1,
                      1.50425385692907503408E1,
                      2.50464946208309415979E0,
                      -1.42182922854787788574E-1,
                      -3.80806407691578277194E-2,
                      -9.33259480895457427372E-4]))
  p2 = list(reversed([3.23774891776946035970E0,
                      6.91522889068984211695E0,
                      3.93881025292474443415E0,
                      1.33303460815807542389E0,
                      2.01485389549179081538E-1,
                      1.23716634817820021358E-2,
                      3.01581553508235416007E-4,
                      2.65806974686737550832E-6,
                      6.23974539184983293730E-9]))
  q2 = list(reversed([1.0,
                      6.02427039364742014255E0,
                      3.67983563856160859403E0,
                      1.37702099489081330271E0,
                      2.16236993594496635890E-1,
                      1.34204006088543189037E-2,
                      3.28014464682127739104E-4,
                      2.89247864745380683936E-6,
                      6.79019408009981274425E-9]))

  dtype = lax.dtype(p).type
  shape = jnp.shape(p)

  def _create_polynomial(var, coeffs):
    """Compute n_th order polynomial via Horner's method."""
    coeffs = np.array(coeffs, dtype)
    if not coeffs.size:
      return jnp.zeros_like(var)
    return coeffs[0] + _create_polynomial(var, coeffs[1:]) * var


  maybe_complement_p = jnp.where(p > dtype(-np.expm1(-2.)), dtype(1.) - p, p)
  # Write in an arbitrary value in place of 0 for p since 0 will cause NaNs
  # later on. The result from the computation when p == 0 is not used so any
  # number that doesn't result in NaNs is fine.
  sanitized_mcp = jnp.where(
      maybe_complement_p <= dtype(0.),
      jnp.full(shape, dtype(0.5)),
      maybe_complement_p)

  # Compute x for p > exp(-2): x/sqrt(2pi) = w + w**3 P0(w**2)/Q0(w**2).
  w = sanitized_mcp - dtype(0.5)
  ww = lax.square(w)
  x_for_big_p = w + w * ww * (_create_polynomial(ww, p0)
                              / _create_polynomial(ww, q0))
  x_for_big_p *= -dtype(np.sqrt(2. * np.pi))

  # Compute x for p <= exp(-2): x = z - log(z)/z - (1/z) P(1/z) / Q(1/z),
  # where z = sqrt(-2. * log(p)), and P/Q are chosen between two different
  # arrays based on whether p < exp(-32).
  z = lax.sqrt(dtype(-2.) * lax.log(sanitized_mcp))
  first_term = z - lax.log(z) / z
  second_term_small_p = (
      _create_polynomial(dtype(1.) / z, p2) /
      _create_polynomial(dtype(1.) / z, q2) / z)
  second_term_otherwise = (
      _create_polynomial(dtype(1.) / z, p1) /
      _create_polynomial(dtype(1.) / z, q1) / z)
  x_for_small_p = first_term - second_term_small_p
  x_otherwise = first_term - second_term_otherwise

  x = jnp.where(sanitized_mcp > dtype(np.exp(-2.)),
                x_for_big_p,
                jnp.where(z >= dtype(8.0), x_for_small_p, x_otherwise))

  x = jnp.where(p > dtype(1. - np.exp(-2.)), x, -x)
  infinity = jnp.full(shape, dtype(np.inf))
  x_nan_replaced = jnp.where(
      p <= dtype(0.0), -infinity, jnp.where(p >= dtype(1.0), infinity, x))
  return x_nan_replaced


@partial(api.custom_jvp, nondiff_argnums=(1,))
def log_ndtr(x, series_order=3):
  r"""Log Normal distribution function.

  For details of the Normal distribution function see `ndtr`.

  This function calculates :math:`\log(\mathrm{ndtr}(x))` by either calling
  :math:`\log(\mathrm{ndtr}(x))` or using an asymptotic series. Specifically:

  - For `x > upper_segment`, use the approximation `-ndtr(-x)` based on
    :math:`\log(1-x) \approx -x, x \ll 1`.
  - For `lower_segment < x <= upper_segment`, use the existing `ndtr` technique
    and take a log.
  - For `x <= lower_segment`, we use the series approximation of `erf` to compute
    the log CDF directly.

  The `lower_segment` is set based on the precision of the input:

  .. math::
    \begin{align}
    \mathit{lower\_segment} =&
      \ \begin{cases}
        -20 &  x.\mathrm{dtype}=\mathit{float64} \\
        -10 &  x.\mathrm{dtype}=\mathit{float32} \\
        \end{cases} \\
    \mathit{upper\_segment} =&
      \ \begin{cases}
        8&  x.\mathrm{dtype}=\mathit{float64} \\
        5&  x.\mathrm{dtype}=\mathit{float32} \\
        \end{cases}
    \end{align}


  When `x < lower_segment`, the `ndtr` asymptotic series approximation is:

  .. math::
    \begin{align}
     \mathrm{ndtr}(x) =&\  \mathit{scale} * (1 + \mathit{sum}) + R_N \\
     \mathit{scale}   =&\  \frac{e^{-0.5 x^2}}{-x \sqrt{2 \pi}} \\
     \mathit{sum}     =&\  \sum_{n=1}^N {-1}^n (2n-1)!! / (x^2)^n \\
     R_N     =&\  O(e^{-0.5 x^2} (2N+1)!! / |x|^{2N+3})
    \end{align}

  where :math:`(2n-1)!! = (2n-1) (2n-3) (2n-5) ...  (3) (1)` is a
  `double-factorial
  <https://en.wikipedia.org/wiki/Double_factorial>`_ operator.


  Args:
    x: an array of type `float32`, `float64`.
    series_order: Positive Python integer. Maximum depth to
      evaluate the asymptotic expansion. This is the `N` above.

  Returns:
    an array with `dtype=x.dtype`.

  Raises:
    TypeError: if `x.dtype` is not handled.
    TypeError: if `series_order` is a not Python `integer.`
    ValueError:  if `series_order` is not in `[0, 30]`.
  """
  if not isinstance(series_order, int):
    raise TypeError("series_order must be a Python integer.")
  if series_order < 0:
    raise ValueError("series_order must be non-negative.")
  if series_order > 30:
    raise ValueError("series_order must be <= 30.")

  x = jnp.asarray(x)
  dtype = lax.dtype(x)

  if dtype == jnp.float64:
    lower_segment = _LOGNDTR_FLOAT64_LOWER
    upper_segment = _LOGNDTR_FLOAT64_UPPER
  elif dtype == jnp.float32:
    lower_segment = _LOGNDTR_FLOAT32_LOWER
    upper_segment = _LOGNDTR_FLOAT32_UPPER
  else:
    raise TypeError("x.dtype={} is not supported.".format(np.dtype(dtype)))

  # The basic idea here was ported from:
  #   https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
  # We copy the main idea, with a few changes
  # * For x >> 1, and X ~ Normal(0, 1),
  #     Log[P[X < x]] = Log[1 - P[X < -x]] approx -P[X < -x],
  #     which extends the range of validity of this function.
  # * We use one fixed series_order for all of 'x', rather than adaptive.
  # * Our docstring properly reflects that this is an asymptotic series, not a
  #   Taylor series. We also provided a correct bound on the remainder.
  # * We need to use the max/min in the _log_ndtr_lower arg to avoid nan when
  #   x=0. This happens even though the branch is unchosen because when x=0
  #   the gradient of a select involves the calculation 1*dy+0*(-inf)=nan
  #   regardless of whether dy is finite. Note that the minimum is a NOP if
  #   the branch is chosen.
  return jnp.where(
      lax.gt(x, upper_segment),
      -_ndtr(-x),  # log(1-x) ~= -x, x << 1
      jnp.where(lax.gt(x, lower_segment),
                       lax.log(_ndtr(lax.max(x, lower_segment))),
                       _log_ndtr_lower(lax.min(x, lower_segment),
                                       series_order)))
def _log_ndtr_jvp(series_order, primals, tangents):
  (x,), (t,) = primals, tangents
  ans = log_ndtr(x, series_order=series_order)
  t_out = lax.mul(t, lax.exp(lax.sub(_norm_logpdf(x), ans)))
  return ans, t_out
log_ndtr.defjvp(_log_ndtr_jvp)

def _log_ndtr_lower(x, series_order):
  """Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`."""
  dtype = lax.dtype(x).type
  x_2 = lax.square(x)
  # Log of the term multiplying (1 + sum)
  log_scale = -dtype(0.5) * x_2 - lax.log(-x) - dtype(0.5 * np.log(2. * np.pi))
  return log_scale + lax.log(_log_ndtr_asymptotic_series(x, series_order))


def _log_ndtr_asymptotic_series(x, series_order):
  """Calculates the asymptotic series used in log_ndtr."""
  dtype = lax.dtype(x).type
  if series_order <= 0:
    return np.array(1, dtype)
  x_2 = lax.square(x)
  even_sum = jnp.zeros_like(x)
  odd_sum = jnp.zeros_like(x)
  x_2n = x_2  # Start with x^{2*1} = x^{2*n} with n = 1.
  for n in range(1, series_order + 1):
    y = np.array(_double_factorial(2 * n - 1), dtype) / x_2n
    if n % 2:
      odd_sum += y
    else:
      even_sum += y
    x_2n *= x_2
  return dtype(1.) + even_sum - odd_sum


def _double_factorial(n):
  """The double factorial function for small Python integer `n`."""
  return np.prod(np.arange(n, 1, -2))


_norm_logpdf_constant = np.log(np.sqrt(2 * np.pi))

def _norm_logpdf(x):
  neg_half = _constant_like(x, -0.5)
  log_normalizer = _constant_like(x, _norm_logpdf_constant)
  return lax.sub(lax.mul(neg_half, lax.square(x)), log_normalizer)

@_wraps(osp_special.i0e)
def i0e(x):
  x, = _promote_args_inexact("i0e", x)
  return lax.bessel_i0e(x)

@_wraps(osp_special.i0)
def i0(x):
  x, = _promote_args_inexact("i0", x)
  return lax.mul(lax.exp(lax.abs(x)), lax.bessel_i0e(x))

@_wraps(osp_special.i1e)
def i1e(x):
  x, = _promote_args_inexact("i1e", x)
  return lax.bessel_i1e(x)

@_wraps(osp_special.i1)
def i1(x):
  x, = _promote_args_inexact("i1", x)
  return lax.mul(lax.exp(lax.abs(x)), lax.bessel_i1e(x))


def _gen_recurrence_mask(
    l_max: int, is_normalized: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Generates mask for recurrence relation on the remaining entries.

  The remaining entries are with respect to the diagonal and offdiagonal
  entries.

  Args:
    l_max: see `gen_normalized_legendre`.
    is_normalized: True if the recurrence mask is used by normalized associated
      Legendre functions.

  Returns:
    Arrays representing the mask used by the recurrence relations.
  """

  # Computes all coefficients.
  m_mat, l_mat = jnp.mgrid[:l_max + 1, :l_max + 1]
  if is_normalized:
    c0 = l_mat * l_mat
    c1 = m_mat * m_mat
    c2 = 2.0 * l_mat
    c3 = (l_mat - 1.0) * (l_mat - 1.0)
    d0 = jnp.sqrt((4.0 * c0 - 1.0) / (c0 - c1))
    d1 = jnp.sqrt(((c2 + 1.0) * (c3 - c1)) / ((c2 - 3.0) * (c0 - c1)))
  else:
    d0 = (2.0 * l_mat - 1.0) / (l_mat - m_mat)
    d1 = (l_mat + m_mat - 1.0) / (l_mat - m_mat)

  d0_mask_indices = jnp.triu_indices(l_max + 1, 1)
  d1_mask_indices = jnp.triu_indices(l_max + 1, 2)
  d_zeros = jnp.zeros((l_max + 1, l_max + 1))
  d0_mask = d_zeros.at[d0_mask_indices].set(d0[d0_mask_indices])
  d1_mask = d_zeros.at[d1_mask_indices].set(d1[d1_mask_indices])

  # Creates a 3D mask that contains 1s on the diagonal plane and 0s elsewhere.
  # i = jnp.arange(l_max + 1)[:, None, None]
  # j = jnp.arange(l_max + 1)[None, :, None]
  # k = jnp.arange(l_max + 1)[None, None, :]
  i, j, k = jnp.ogrid[:l_max + 1, :l_max + 1, :l_max + 1]
  mask = 1.0 * (i + j - k == 0)

  d0_mask_3d = jnp.einsum('jk,ijk->ijk', d0_mask, mask)
  d1_mask_3d = jnp.einsum('jk,ijk->ijk', d1_mask, mask)

  return (d0_mask_3d, d1_mask_3d)


@partial(jit, static_argnums=(2))
def _gen_derivatives(p: jnp.ndarray,
                     x: jnp.ndarray,
                     is_normalized: bool) -> jnp.ndarray:
  """Generates derivatives of associated Legendre functions of the first kind.

  Args:
    p: The 3D array containing the values of associated Legendre functions; the
      dimensions are in the sequence of order (m), degree (l), and evalution
      points.
    x: A vector of type `float32` or `float64` containing the sampled points.
    is_normalized: True if the associated Legendre functions are normalized.
  Returns:
    The 3D array representing the derivatives of associated Legendre functions
    of the first kind.
  """

  num_m, num_l, num_x = p.shape

  # p_{l-1}^m.
  p_m_lm1 = jnp.pad(p, ((0, 0), (1, 0), (0, 0)))[:, :num_l, :]

  # p_{l-1}^{m+2}.
  p_mp2_lm1 = jnp.pad(p_m_lm1, ((0, 2), (0, 0), (0, 0)))[2:num_m + 2, :, :]

  # p_{l-1}^{m-2}.
  p_mm2_lm1 = jnp.pad(p_m_lm1, ((2, 0), (0, 0), (0, 0)))[:num_m, :, :]

  # Derivative computation requires negative orders.
  if is_normalized:
    raise NotImplementedError(
        'Negative orders for normalization is not implemented yet.')
  else:
    if num_l > 1:
      l_vec = jnp.arange(1, num_l - 1)
      p_p1 = p[1, 1:num_l - 1, :]
      coeff = -1.0 / ((l_vec + 1) * l_vec)
      update_p_p1 = jnp.einsum('i,ij->ij', coeff, p_p1)
      p_mm2_lm1 = p_mm2_lm1.at[1, 2:num_l, :].set(update_p_p1)

    if num_l > 2:
      l_vec = jnp.arange(2, num_l - 1)
      p_p2 = p[2, 2:num_l - 1, :]
      coeff = 1.0 / ((l_vec + 2) * (l_vec + 1) * l_vec)
      update_p_p2 = jnp.einsum('i,ij->ij', coeff, p_p2)
      p_mm2_lm1 = p_mm2_lm1.at[0, 3:num_l, :].set(update_p_p2)

  m_mat, l_mat = jnp.mgrid[:num_m, :num_l]

  coeff_zeros = jnp.zeros((num_m, num_l))
  upper_0_indices = jnp.triu_indices(num_m, 0, num_l)
  zero_vec = jnp.zeros((num_l,))

  a0 = -0.5 / (m_mat - 1.0)
  a0_masked = coeff_zeros.at[upper_0_indices].set(a0[upper_0_indices])
  a0_masked = a0_masked.at[1, :].set(zero_vec)

  b0 = l_mat + m_mat
  c0 = a0 * (b0 - 2.0) * (b0 - 1.0)
  c0_masked = coeff_zeros.at[upper_0_indices].set(c0[upper_0_indices])
  c0_masked = c0_masked.at[1, :].set(zero_vec)

  # p_l^{m-1}.
  p_mm1_l = (jnp.einsum('ij,ijk->ijk', a0_masked, p_m_lm1) +
             jnp.einsum('ij,ijk->ijk', c0_masked, p_mm2_lm1))

  d0 = -0.5 / (m_mat + 1.0)
  d0_masked = coeff_zeros.at[upper_0_indices].set(d0[upper_0_indices])
  e0 = d0 * b0 * (b0 + 1.0)
  e0_masked = coeff_zeros.at[upper_0_indices].set(e0[upper_0_indices])

  # p_l^{m+1}.
  p_mp1_l = (jnp.einsum('ij,ijk->ijk', d0_masked, p_mp2_lm1) +
             jnp.einsum('ij,ijk->ijk', e0_masked, p_m_lm1))

  f0 = b0 * (l_mat - m_mat + 1.0) / 2.0
  f0_masked = coeff_zeros.at[upper_0_indices].set(f0[upper_0_indices])
  p_derivative = jnp.einsum('ij,ijk->ijk', f0_masked, p_mm1_l) - 0.5 * p_mp1_l

  # Special treatment of the singularity at m = 1.
  if num_m > 1:
    l_vec = jnp.arange(num_l)
    g0 = jnp.einsum('i,ij->ij', (l_vec + 1) * l_vec, p[0, :, :])
    if num_l > 2:
      g0 = g0 -  p[2, :, :]
    p_derivative_m0 = jnp.einsum('j,ij->ij', 0.5 / jnp.sqrt(1 - x * x), g0)
    p_derivative = p_derivative.at[1, :, :].set(p_derivative_m0)
    p_derivative = p_derivative.at[1, 0, :].set(jnp.zeros((num_x,)))

  return p_derivative


@partial(jit, static_argnums=(0, 2))
def _gen_associated_legendre(l_max: int,
                             x: jnp.ndarray,
                             is_normalized: bool) -> jnp.ndarray:
  r"""Computes associated Legendre functions (ALFs) of the first kind.

  The ALFs of the first kind are used in spherical harmonics. The spherical
  harmonic of degree `l` and order `m` can be written as
  `Y_l^m(θ, φ) = N_l^m * P_l^m(cos(θ)) * exp(i m φ)`, where `N_l^m` is the
  normalization factor and θ and φ are the colatitude and longitude,
  repectively. `N_l^m` is chosen in the way that the spherical harmonics form
  a set of orthonormal basis function of L^2(S^2). For the computational
  efficiency of spherical harmonics transform, the normalization factor is
  used in the computation of the ALFs. In addition, normalizing `P_l^m`
  avoids overflow/underflow and achieves better numerical stability. Three
  recurrence relations are used in the computation.

  Args:
    l_max: The maximum degree of the associated Legendre function. Both the
      degrees and orders are `[0, 1, 2, ..., l_max]`.
    x: A vector of type `float32`, `float64` containing the sampled points in
      spherical coordinates, at which the ALFs are computed; `x` is essentially
      `cos(θ)`. For the numerical integration used by the spherical harmonics
      transforms, `x` contains the quadrature points in the interval of
      `[-1, 1]`. There are several approaches to provide the quadrature points:
      Gauss-Legendre method (`scipy.special.roots_legendre`), Gauss-Chebyshev
      method (`scipy.special.roots_chebyu`), and Driscoll & Healy
      method (Driscoll, James R., and Dennis M. Healy. "Computing Fourier
      transforms and convolutions on the 2-sphere." Advances in applied
      mathematics 15, no. 2 (1994): 202-250.). The Gauss-Legendre quadrature
      points are nearly equal-spaced along θ and provide exact discrete
      orthogonality, (P^m)^T W P_m = I, where `T` represents the transpose
      operation, `W` is a diagonal matrix containing the quadrature weights,
      and `I` is the identity matrix. The Gauss-Chebyshev points are equally
      spaced, which only provide approximate discrete orthogonality. The
      Driscoll & Healy qudarture points are equally spaced and provide the
      exact discrete orthogonality. The number of sampling points is required to
      be twice as the number of frequency points (modes) in the Driscoll & Healy
      approach, which enables FFT and achieves a fast spherical harmonics
      transform.
    is_normalized: True if the associated Legendre functions are normalized.
      With normalization, `N_l^m` is applied such that the spherical harmonics
      form a set of orthonormal basis functions of L^2(S^2).

  Returns:
    The 3D array of shape `(l_max + 1, l_max + 1, len(x))` containing the values
    of the ALFs at `x`; the dimensions in the sequence of order, degree, and
    evalution points.
  """
  p = jnp.zeros((l_max + 1, l_max + 1, x.shape[0]))

  a_idx = jnp.arange(1, l_max + 1)
  b_idx = jnp.arange(l_max)
  if is_normalized:
    initial_value = 0.5 / jnp.sqrt(jnp.pi)  # The initial value p(0,0).
    f_a = jnp.cumprod(-1 * jnp.sqrt(1.0 + 0.5 / a_idx))
    f_b = jnp.sqrt(2.0 * b_idx + 3.0)
  else:
    initial_value = 1.0  # The initial value p(0,0).
    f_a = jnp.cumprod(1.0 - 2.0 * a_idx)
    f_b = 2.0 * b_idx + 1.0

  p = p.at[(0, 0)].set(initial_value)

  # Compute the diagonal entries p(l,l) with recurrence.
  y = jnp.cumprod(
      jnp.broadcast_to(jnp.sqrt(1.0 - x * x), (l_max, x.shape[0])),
      axis=0)
  p_diag = initial_value * jnp.einsum('i,ij->ij', f_a, y)
  diag_indices = jnp.diag_indices(l_max + 1)
  p = p.at[(diag_indices[0][1:], diag_indices[1][1:])].set(p_diag)

  # Compute the off-diagonal entries with recurrence.
  p_offdiag = jnp.einsum('ij,ij->ij',
                         jnp.einsum('i,j->ij', f_b, x),
                         p[jnp.diag_indices(l_max)])
  offdiag_indices = (diag_indices[0][:l_max], diag_indices[1][:l_max] + 1)
  p = p.at[offdiag_indices].set(p_offdiag)

  # Compute the remaining entries with recurrence.
  d0_mask_3d, d1_mask_3d = _gen_recurrence_mask(
      l_max, is_normalized=is_normalized)

  def body_fun(i, p_val):
    coeff_0 = d0_mask_3d[i]
    coeff_1 = d1_mask_3d[i]
    h = (jnp.einsum('ij,ijk->ijk',
                    coeff_0,
                    jnp.einsum(
                        'ijk,k->ijk', jnp.roll(p_val, shift=1, axis=1), x)) -
         jnp.einsum('ij,ijk->ijk', coeff_1, jnp.roll(p_val, shift=2, axis=1)))
    p_val = p_val + h
    return p_val

  # TODO(jakevdp): use some sort of fixed-point procedure here instead?
  p = p.astype(jnp.result_type(p, x, d0_mask_3d))
  if l_max > 1:
    p = lax.fori_loop(lower=2, upper=l_max+1, body_fun=body_fun, init_val=p)

  return p


def lpmn(m: int, n: int, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """The associated Legendre functions (ALFs) of the first kind.

  Args:
    m: The maximum order of the associated Legendre functions.
    n: The maximum degree of the associated Legendre function, often called
      `l` in describing ALFs. Both the degrees and orders are
      `[0, 1, 2, ..., l_max]`, where `l_max` denotes the maximum degree.
    z: A vector of type `float32` or `float64` containing the sampling
      points at which the ALFs are computed.

  Returns:
    A 2-tuple of 3D arrays of shape `(l_max + 1, l_max + 1, len(z))` containing
    the values and derivatives of the associated Legendre functions of the
    first kind. The return type matches the type of `z`.

  Raises:
    TypeError if elements of array `z` are not in (float32, float64).
    ValueError if array `z` is not 1D.
    NotImplementedError if `m!=n`.
  """
  dtype = lax.dtype(z)
  if dtype not in (jnp.float32, jnp.float64):
    raise TypeError(
        'z.dtype={} is not supported, see docstring for supported types.'
        .format(dtype))

  if z.ndim != 1:
    raise ValueError('z must be a 1D array.')

  m = core.concrete_or_error(int, m, 'Argument m of lpmn.')
  n = core.concrete_or_error(int, n, 'Argument n of lpmn.')

  if m != n:
    raise NotImplementedError('Computations for m!=n are not yet supported.')

  l_max = n
  is_normalized = False
  p_vals = _gen_associated_legendre(l_max, z, is_normalized)
  p_derivatives = _gen_derivatives(p_vals, z, is_normalized)

  return (p_vals, p_derivatives)


def lpmn_values(m: int, n: int, z: jnp.ndarray, is_normalized: bool) -> jnp.ndarray:
  r"""The associated Legendre functions (ALFs) of the first kind.

  Unlike `lpmn`, this function only computes the values of ALFs.
  The ALFs of the first kind can be used in spherical harmonics. The
  spherical harmonic of degree `l` and order `m` can be written as
  :math:`Y_l^m(\theta, \phi) = N_l^m * P_l^m(\cos \theta) * \exp(i m \phi)`,
  where :math:`N_l^m` is the normalization factor and θ and φ are the
  colatitude and longitude, repectively. :math:`N_l^m` is chosen in the
  way that the spherical harmonics form a set of orthonormal basis function
  of :math:`L^2(S^2)`. Normalizing :math:`P_l^m` avoids overflow/underflow
  and achieves better numerical stability.

  Args:
    m: The maximum order of the associated Legendre functions.
    n: The maximum degree of the associated Legendre function, often called
      `l` in describing ALFs. Both the degrees and orders are
      `[0, 1, 2, ..., l_max]`, where `l_max` denotes the maximum degree.
    z: A vector of type `float32` or `float64` containing the sampling
      points at which the ALFs are computed.
    is_normalized: True if the associated Legendre functions are normalized.
      With normalization, :math:`N_l^m` is applied such that the spherical
      harmonics form a set of orthonormal basis functions of :math:`L^2(S^2)`.

  Returns:
    A 3D array of shape `(l_max + 1, l_max + 1, len(z))` containing
    the values of the associated Legendre functions of the first kind. The
    return type matches the type of `z`.

  Raises:
    TypeError if elements of array `z` are not in (float32, float64).
    ValueError if array `z` is not 1D.
    NotImplementedError if `m!=n`.
  """
  dtype = lax.dtype(z)
  if dtype not in (jnp.float32, jnp.float64):
    raise TypeError(
        'z.dtype={} is not supported, see docstring for supported types.'
        .format(dtype))

  if z.ndim != 1:
    raise ValueError('z must be a 1D array.')

  m = core.concrete_or_error(int, m, 'Argument m of lpmn.')
  n = core.concrete_or_error(int, n, 'Argument n of lpmn.')

  if m != n:
    raise NotImplementedError('Computations for m!=n are not yet supported.')

  l_max = n

  return _gen_associated_legendre(l_max, z, is_normalized)



@partial(jit, static_argnums=(4,))
def _sph_harm(m: jnp.ndarray,
              n: jnp.ndarray,
              theta: jnp.ndarray,
              phi: jnp.ndarray,
              n_max: int) -> jnp.ndarray:
  """Computes the spherical harmonics."""

  cos_colatitude = jnp.cos(phi)

  legendre = _gen_associated_legendre(n_max, cos_colatitude, True)
  legendre_val = legendre.at[abs(m), n, jnp.arange(len(n))].get(mode="clip")

  angle = abs(m) * theta
  vandermonde = lax.complex(jnp.cos(angle), jnp.sin(angle))
  harmonics = lax.complex(legendre_val * jnp.real(vandermonde),
                          legendre_val * jnp.imag(vandermonde))

  # Negative order.
  harmonics = jnp.where(m < 0,
                        (-1.0)**abs(m) * jnp.conjugate(harmonics),
                        harmonics)

  return harmonics


def sph_harm(m: jnp.ndarray,
             n: jnp.ndarray,
             theta: jnp.ndarray,
             phi: jnp.ndarray,
             n_max: Optional[int] = None) -> jnp.ndarray:
  r"""Computes the spherical harmonics.

  The JAX version has one extra argument `n_max`, the maximum value in `n`.

  The spherical harmonic of degree `n` and order `m` can be written as
  :math:`Y_n^m(\theta, \phi) = N_n^m * P_n^m(\cos \phi) * \exp(i m \theta)`,
  where :math:`N_n^m = \sqrt{\frac{\left(2n+1\right) \left(n-m\right)!}
  {4 \pi \left(n+m\right)!}}` is the normalization factor and :math:`\phi` and
  :math:\theta` are the colatitude and longitude, repectively. :math:`N_n^m` is
  chosen in the way that the spherical harmonics form a set of orthonormal basis
  functions of :math:`L^2(S^2)`.

  Args:
    m: The order of the harmonic; must have `|m| <= n`. Return values for
      `|m| > n` ara undefined.
    n: The degree of the harmonic; must have `n >= 0`. The standard notation for
      degree in descriptions of spherical harmonics is `l (lower case L)`. We
      use `n` here to be consistent with `scipy.special.sph_harm`. Return
      values for `n < 0` are undefined.
    theta: The azimuthal (longitudinal) coordinate; must be in [0, 2*pi].
    phi: The polar (colatitudinal) coordinate; must be in [0, pi].
    n_max: The maximum degree `max(n)`. If the supplied `n_max` is not the true
      maximum value of `n`, the results are clipped to `n_max`. For example,
      `sph_harm(m=jnp.array([2]), n=jnp.array([10]), theta, phi, n_max=6)`
      acutually returns
      `sph_harm(m=jnp.array([2]), n=jnp.array([6]), theta, phi, n_max=6)`
  Returns:
    A 1D array containing the spherical harmonics at (m, n, theta, phi).
  """

  if jnp.isscalar(phi):
    phi = jnp.array([phi])

  if n_max is None:
    n_max = jnp.max(n)
  n_max = core.concrete_or_error(
      int, n_max, 'The `n_max` argument of `jnp.scipy.special.sph_harm` must '
      'be statically specified to use `sph_harm` within JAX transformations.')

  return _sph_harm(m, n, theta, phi, n_max)


# exponential integrals
# these algorithms are ported over from the files ei.c and expn.c in the Cephes mathematical library.
# https://fossies.org/dox/cephes-math-28/ei_8c_source.html
# https://fossies.org/dox/cephes-math-28/expn_8c_source.html


def _expint1(x):
  # 0 < x <= 2
  A = [
    -5.350447357812542947283e0,
    2.185049168816613393830e2,
    -4.176572384826693777058e3,
    5.541176756393557601232e4,
    -3.313381331178144034309e5,
    1.592627163384945414220e6,
  ]
  B = [
    1.0,
    -5.250547959112862969197e1,
    1.259616186786790571525e3,
    -1.756549581973534652631e4,
    1.493062117002725991967e5,
    -7.294949239640527645655e5,
    1.592627163384945429726e6,
  ]
  A, B = [jnp.array(U, dtype=x.dtype) for U in [A, B]]
  f = jnp.polyval(A, x) / jnp.polyval(B, x)
  return x * f + jnp.euler_gamma + jnp.log(x)


def _eval_expint_k(A, B, x):
  # helper function for all subsequent intervals
  A, B = [jnp.array(U, dtype=x.dtype) for U in [A, B]]
  one = _constant_like(x, 1.0)
  w = one / x
  f = jnp.polyval(A, w) / jnp.polyval(B, w)
  f = w * f + one
  return jnp.exp(x) * w * f


def _expint2(x):
  # 2 <= x < 4
  A = [
    1.981808503259689673238e-2,
    -1.271645625984917501326e0,
    -2.088160335681228318920e0,
    2.755544509187936721172e0,
    -4.409507048701600257171e-1,
    4.665623805935891391017e-2,
    -1.545042679673485262580e-3,
    7.059980605299617478514e-5,
  ]
  B = [
    1.0,
    1.476498670914921440652e0,
    5.629177174822436244827e-1,
    1.699017897879307263248e-1,
    2.291647179034212017463e-2,
    4.450150439728752875043e-3,
    1.727439612206521482874e-4,
    3.953167195549672482304e-5,
  ]
  return _eval_expint_k(A, B, x)


def _expint3(x):
  # 4 <= x <= 8
  A = [
    -1.373215375871208729803e0,
    -7.084559133740838761406e-1,
    1.580806855547941010501e0,
    -2.601500427425622944234e-1,
    2.994674694113713763365e-2,
    -1.038086040188744005513e-3,
    4.371064420753005429514e-5,
    2.141783679522602903795e-6,
  ]
  B = [
    1.0,
    8.585231423622028380768e-1,
    4.483285822873995129957e-1,
    7.687932158124475434091e-2,
    2.449868241021887685904e-2,
    8.832165941927796567926e-4,
    4.590952299511353531215e-4,
    -4.729848351866523044863e-6,
    2.665195537390710170105e-6,
  ]
  return _eval_expint_k(A, B, x)


def _expint4(x):
  # 8 <= x <= 16
  A = [
    -2.106934601691916512584e0,
    1.732733869664688041885e0,
    -2.423619178935841904839e-1,
    2.322724180937565842585e-2,
    2.372880440493179832059e-4,
    -8.343219561192552752335e-5,
    1.363408795605250394881e-5,
    -3.655412321999253963714e-7,
    1.464941733975961318456e-8,
    6.176407863710360207074e-10,
  ]
  B = [
    1.0,
    -2.298062239901678075778e-1,
    1.105077041474037862347e-1,
    -1.566542966630792353556e-2,
    2.761106850817352773874e-3,
    -2.089148012284048449115e-4,
    1.708528938807675304186e-5,
    -4.459311796356686423199e-7,
    1.394634930353847498145e-8,
    6.150865933977338354138e-10,
  ]
  return _eval_expint_k(A, B, x)


def _expint5(x):
  # 16 <= x <= 32
  A = [
    -2.458119367674020323359e-1,
    -1.483382253322077687183e-1,
    7.248291795735551591813e-2,
    -1.348315687380940523823e-2,
    1.342775069788636972294e-3,
    -7.942465637159712264564e-5,
    2.644179518984235952241e-6,
    -4.239473659313765177195e-8,
  ]
  B = [
    1.0,
    -1.044225908443871106315e-1,
    -2.676453128101402655055e-1,
    9.695000254621984627876e-2,
    -1.601745692712991078208e-2,
    1.496414899205908021882e-3,
    -8.462452563778485013756e-5,
    2.728938403476726394024e-6,
    -4.239462431819542051337e-8,
  ]
  return _eval_expint_k(A, B, x)


def _expint6(x):
  # 32 <= x <= 64
  A = [
    1.212561118105456670844e-1,
    -5.823133179043894485122e-1,
    2.348887314557016779211e-1,
    -3.040034318113248237280e-2,
    1.510082146865190661777e-3,
    -2.523137095499571377122e-5,
  ]
  B = [
    1.0,
    -1.002252150365854016662e0,
    2.928709694872224144953e-1,
    -3.337004338674007801307e-2,
    1.560544881127388842819e-3,
    -2.523137093603234562648e-5,
  ]
  return _eval_expint_k(A, B, x)


def _expint7(x):
  # x > 64
  A = [
    -7.657847078286127362028e-1,
    6.886192415566705051750e-1,
    -2.132598113545206124553e-1,
    3.346107552384193813594e-2,
    -3.076541477344756050249e-3,
    1.747119316454907477380e-4,
    -6.103711682274170530369e-6,
    1.218032765428652199087e-7,
    -1.086076102793290233007e-9,
  ]
  B = [
    1.0,
    -1.888802868662308731041e0,
    1.066691687211408896850e0,
    -2.751915982306380647738e-1,
    3.930852688233823569726e-2,
    -3.414684558602365085394e-3,
    1.866844370703555398195e-4,
    -6.345146083130515357861e-6,
    1.239754287483206878024e-7,
    -1.086076102793126632978e-9,
  ]
  return _eval_expint_k(A, B, x)


def _expi_pos(x):
  # x > 0
  _c = _constant_like
  conds = [(_c(x, 0) < x) & (x <= _c(x, 2))] + [
    (_c(x, 2 ** i) < x) & (x <= _c(x, 2 ** (i + 1))) for i in range(1, 6)
  ]
  return jnp.piecewise(
    x,
    conds,
    [_expint1, _expint2, _expint3, _expint4, _expint5, _expint6, _expint7],
  )


@_wraps(osp_special.expi)
@api.custom_jvp
@jit
def expi(x):
  (x,) = _promote_args_inexact("expi", x)
  ret = jnp.piecewise(x, [x < 0], [lambda x: -exp1(-x), _expi_pos])
  return ret


@expi.defjvp
@jit
def expi_jvp(primals, tangents):
  (x,) = primals
  (x_dot,) = tangents
  return expi(x), jnp.exp(x) / x * x_dot


def _expn1(n, x):
  # exponential integral En
  _c = _constant_like
  x = jnp.array(x)
  MACHEP = jnp.finfo(x.dtype).eps

  zero = _c(x, 0.0)
  one = _c(x, 1.0)
  psi = -jnp.euler_gamma - jnp.log(x)
  psi = lax.fori_loop(_c(n, 1), n, lambda i, psi: psi + one / i, psi)
  n1 = jnp.where(n == _c(n, 1), one + one, n)
  init = dict(
    x=x,
    z=-x,
    xk=zero,
    yk=one,
    pk=one - n,
    ans=jnp.where(n == _c(n, 1), zero, one / (one - n1)),
    t=jnp.inf,
  )

  def body(d):
    d["xk"] += one
    d["yk"] *= d["z"] / d["xk"]
    d["pk"] += one
    d["ans"] += jnp.where(d["pk"] != zero, d["yk"] / d["pk"], zero)
    d["t"] = jnp.where(d["ans"] != zero, abs(d["yk"] / d["ans"]), one)
    return d

  def cond(d):
    return (d["x"] > _c(d["x"], 0.0)) & (d["t"] > MACHEP)

  d = lax.while_loop(cond, body, init)
  t = n
  r = n - _c(n, 1)
  return d["z"] ** r * psi / jnp.exp(gammaln(t)) - d["ans"]


def _expn2(n, x):
  # x > 1.
  _c = _constant_like
  BIG = _c(x, 1.44115188075855872e17)
  MACHEP = jnp.finfo(BIG.dtype).eps  # ?
  zero = _c(x, 0.0)
  one = _c(x, 1.0)

  init = dict(
    k=_c(n, 1),
    pkm2=one,
    qkm2=x,
    pkm1=one,
    qkm1=x + n,
    ans=one / (x + n),
    t=_c(x, jnp.inf),
    r=zero,
    x=x,
  )

  def body(d):
    x = d["x"]
    d["k"] += _c(d["k"], 1)
    k = d["k"]
    odd = k % _c(k, 2) == _c(k, 1)
    yk = jnp.where(odd, one, x)
    xk = jnp.where(odd, n + (k - _c(k, 1)) / _c(k, 2), k / _c(k, 2))
    pk = d["pkm1"] * yk + d["pkm2"] * xk
    qk = d["qkm1"] * yk + d["qkm2"] * xk
    nz = qk != zero
    d["r"] = r = jnp.where(nz, pk / qk, d["r"])
    d["t"] = jnp.where(nz, abs((d["ans"] - r) / r), one)
    d["ans"] = jnp.where(nz, r, d["ans"])
    d["pkm2"] = d["pkm1"]
    d["pkm1"] = pk
    d["qkm2"] = d["qkm1"]
    d["qkm1"] = qk
    is_big = abs(pk) > BIG
    for s in "pq":
      for i in "12":
        key = s + "km" + i
        d[key] = jnp.where(is_big, d[key] / BIG, d[key])
    return d

  def cond(d):
    return (d["x"] > _c(d["k"], 0)) & (d["t"] > MACHEP)

  d = lax.while_loop(cond, body, init)
  return d["ans"] * jnp.exp(-x)


def _expn3(n, x):
  # n >= 5000
  _c = _constant_like
  one = _c(x, 1.0)
  xk = x + n
  yk = one / (xk * xk)
  t = n
  ans = yk * t * (_c(x, 6) * x * x - _c(x, 8) * t * x + t * t)
  ans = yk * (ans + t * (t - _c(x, 2) * x))
  ans = yk * (ans + t)
  return (ans + one) * jnp.exp(-x) / xk


@_wraps(osp_special.expn)
@partial(api.custom_jvp, nondiff_argnums=(0,))
@jnp.vectorize
@jit
def expn(n, x):
  n, x = _promote_args_inexact("expn", n, x)
  _c = _constant_like
  zero = _c(x, 0)
  one = _c(x, 1)
  conds = [
    (n < _c(n, 0)) | (x < zero),
    (x == zero) & (n < _c(n, 2)),
    (x == zero) & (n >= _c(n, 2)),
    (n == _c(n, 0)) & (x >= zero),
    (n >= _c(n, 5000)),
    (x > one),
  ]
  n1 = jnp.where(n == _c(n, 1), n + n, n)
  vals = [
    jnp.nan,
    jnp.inf,
    one / n1,  # prevent div by zero
    jnp.exp(-x) / x,
    partial(_expn3, n),
    partial(_expn2, n),
    partial(_expn1, n),
  ]
  ret = jnp.piecewise(x, conds, vals)
  return ret


@expn.defjvp
@jit
def expn_jvp(n, primals, tangents):
  (x,), (x_dot,) = primals, tangents
  return expn(n, x), lax.mul(
    lax.neg(x_dot), expn(lax.sub(n, _constant_like(n, 1)), x)
  )


@_wraps(osp_special.exp1)
def exp1(x):
  (x,) = _promote_args_inexact("exp1", x)
  return expn(1, x)
