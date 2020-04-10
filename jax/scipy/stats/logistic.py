# Copyright 2020 Google LLC
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

import scipy.stats as osp_stats
from jax.scipy.special import expit, logit

from ... import lax
from ...numpy.lax_numpy import _promote_args_inexact, _wraps


@_wraps(osp_stats.logistic.logpdf, update_doc=False)
def logpdf(x):
  return lax.neg(x) - 2. * lax.log1p(lax.exp(lax.neg(x)))

@_wraps(osp_stats.logistic.pdf, update_doc=False)
def pdf(x):
  return lax.exp(logpdf(x))

@_wraps(osp_stats.logistic.ppf, update_doc=False)
def ppf(x):
  return logit(x)

@_wraps(osp_stats.logistic.sf, update_doc=False)
def sf(x):
  return expit(lax.neg(x))

@_wraps(osp_stats.logistic.isf, update_doc=False)
def isf(x):
  return -logit(x)

@_wraps(osp_stats.logistic.cdf, update_doc=False)
def cdf(x):
  return expit(x)
