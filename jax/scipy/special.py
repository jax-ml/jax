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


import math

import numpy as onp
import scipy.special as osp_special

from .. import lax
from .. import util
from ..api import custom_transforms, defjvp
from ..numpy import lax_numpy as np
from ..numpy.lax_numpy import (_wraps, asarray, _reduction_dims, _constant_like,
                               _promote_args_inexact)


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


@_wraps(osp_special.digamma, update_doc=False)
def digamma(x):
  x, = _promote_args_inexact("digamma", x)
  return lax.digamma(x)


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


@_wraps(osp_special.logit, update_doc=False)
@custom_transforms
def logit(x):
  x = asarray(x)
  return lax.log(lax.div(x, lax.sub(lax._const(x, 1), x)))
defjvp(logit, lambda g, ans, x: g / (x * (1 - x)))


@_wraps(osp_special.expit, update_doc=False)
@custom_transforms
def expit(x):
  x = asarray(x)
  one = lax._const(x, 1)
  return lax.div(one, lax.add(one, lax.exp(lax.neg(x))))
defjvp(expit, lambda g, ans, x: g * ans * (lax._const(ans, 1) - ans))


@_wraps(osp_special.logsumexp)
def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
  if b is not None or return_sign:
    raise NotImplementedError("Only implemented for b=None, return_sign=False")
  dims = _reduction_dims(a, axis)
  shape = util.subvals(onp.shape(a), zip(dims, (1,) * len(dims)))
  dimadd = lambda x: lax.reshape(x, shape)
  amax = lax.reduce(a, _constant_like(a, -onp.inf), lax.max, dims)
  amax = lax.select(lax.is_finite(amax), amax, lax.full_like(amax, 0))
  amax_singletons = dimadd(amax)
  out = lax.add(lax.log(lax.reduce(lax.exp(lax.sub(a, amax_singletons)),
                                   _constant_like(a, 0), lax.add, dims)), amax)
  return dimadd(out) if keepdims else out


@_wraps(osp_special.xlogy)
def xlogy(x, y):
  x, y = _promote_args_inexact("xlogy", x, y)
  return lax._safe_mul(x, lax.log(y))


@_wraps(osp_special.xlog1py, update_doc=False)
def xlog1py(x, y):
  x, y = _promote_args_inexact("xlog1py", x, y)
  return lax._safe_mul(x, lax.log1p(y))


@_wraps(osp_special.entr)
def entr(x):
  x, = _promote_args_inexact("entr", x)
  return lax.select(lax.lt(x, _constant_like(x, 0)),
                    lax.full_like(x, -onp.inf),
                    lax.neg(xlogy(x, x)))


@_wraps(osp_special.multigammaln, update_doc=False)
def multigammaln(a, d):
  a, = _promote_args_inexact("multigammaln", a)
  d = lax.convert_element_type(d, lax.dtype(a))
  constant = lax.mul(lax.mul(lax.mul(_constant_like(a, 0.25), d),
                             lax.sub(d, _constant_like(a, 1))),
                     lax.log(_constant_like(a, onp.pi)))
  res = np.sum(gammaln(np.expand_dims(a, axis=-1) -
                       lax.div(np.arange(d), _constant_like(a, 2))),
               axis=-1)
  return res + constant


# Normal distributions

# Functions "ndtr" [... is] derived from calculations made in:
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
_LOGNDTR_FLOAT64_LOWER = onp.array(-20, onp.float64)
_LOGNDTR_FLOAT32_LOWER = onp.array(-10, onp.float32)

# Upper bound values were chosen by examining for which values of 'x'
# Log[cdf(x)] is 0, after which point we need to use the approximation
# Log[cdf(x)] = Log[1 - cdf(-x)] approx -cdf(-x). We chose a value slightly
# conservative, meaning we use the approximation earlier than needed.
_LOGNDTR_FLOAT64_UPPER = onp.array(8, onp.float64)
_LOGNDTR_FLOAT32_UPPER = onp.array(5, onp.float32)


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
  x = np.asarray(x)
  dtype = lax.dtype(x)
  if dtype not in (np.float32, np.float64):
    raise TypeError(
        "x.dtype={} is not supported, see docstring for supported types."
        .format(dtype))
  return _ndtr(x)


def _ndtr(x):
  """Implements ndtr core logic."""
  dtype = lax.dtype(x).type
  half_sqrt_2 = dtype(0.5) * onp.sqrt(2., dtype=dtype)
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
    p: an array of floating-point type.

  Returns:
    an array with `dtype=p.dtype`.

  Raises:
    TypeError: if `p` is not floating-type.
  """
  p, = _promote_args_inexact("ndtri", p)
  return lax.erf_inv(np.asarray(p) * 2 - 1) * math.sqrt(2)


@custom_transforms
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

  x = np.asarray(x)
  dtype = lax.dtype(x)

  if dtype == np.float64:
    lower_segment = _LOGNDTR_FLOAT64_LOWER
    upper_segment = _LOGNDTR_FLOAT64_UPPER
  elif dtype == np.float32:
    lower_segment = _LOGNDTR_FLOAT32_LOWER
    upper_segment = _LOGNDTR_FLOAT32_UPPER
  else:
    raise TypeError("x.dtype={} is not supported.".format(onp.dtype(dtype)))

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
  return np.where(
      lax.gt(x, upper_segment),
      -_ndtr(-x),  # log(1-x) ~= -x, x << 1
      np.where(lax.gt(x, lower_segment),
                      lax.log(_ndtr(lax.max(x, lower_segment))),
                      _log_ndtr_lower(lax.min(x, lower_segment),
                                      series_order)))


def _log_ndtr_lower(x, series_order):
  """Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`."""
  dtype = lax.dtype(x).type
  x_2 = lax.square(x)
  # Log of the term multiplying (1 + sum)
  log_scale = -dtype(0.5) * x_2 - lax.log(-x) - dtype(0.5 * onp.log(2. * onp.pi))
  return log_scale + lax.log(_log_ndtr_asymptotic_series(x, series_order))


def _log_ndtr_asymptotic_series(x, series_order):
  """Calculates the asymptotic series used in log_ndtr."""
  dtype = lax.dtype(x).type
  if series_order <= 0:
    return onp.array(1, dtype)
  x_2 = lax.square(x)
  even_sum = np.zeros_like(x)
  odd_sum = np.zeros_like(x)
  x_2n = x_2  # Start with x^{2*1} = x^{2*n} with n = 1.
  for n in range(1, series_order + 1):
    y = onp.array(_double_factorial(2 * n - 1), dtype) / x_2n
    if n % 2:
      odd_sum += y
    else:
      even_sum += y
    x_2n *= x_2
  return dtype(1.) + even_sum - odd_sum


def _double_factorial(n):
  """The double factorial function for small Python integer `n`."""
  return onp.prod(onp.arange(n, 1, -2))


_norm_logpdf_constant = onp.log(onp.sqrt(2 * onp.pi))

def _norm_logpdf(x):
  neg_half = _constant_like(x, -0.5)
  log_normalizer = _constant_like(x, _norm_logpdf_constant)
  return lax.sub(lax.mul(neg_half, lax.square(x)), log_normalizer)

defjvp(log_ndtr,
       lambda g, ans, x: lax.mul(g, lax.exp(lax.sub(_norm_logpdf(x), ans))))

@_wraps(osp_special.i0e)
def i0e(x):
  return lax.bessel_i0e(x)

@_wraps(osp_special.i1e)
def i1e(x):
  return lax.bessel_i1e(x)
