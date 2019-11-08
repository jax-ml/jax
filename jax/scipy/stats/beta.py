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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
import scipy.stats as osp_stats

from ... import lax
from ...numpy.lax_numpy import (_promote_args_like, _constant_like, _wraps,
                                where, inf, logical_or)
from ..special import gammaln


@_wraps(osp_stats.beta.logpdf, update_doc=False)
def logpdf(x, a, b, loc=0, scale=1):
  x, a, b, loc, scale = _promote_args_like(osp_stats.beta.logpdf, x, a, b, loc, scale)
  one = _constant_like(x, 1)
  shape_term_tmp = lax.add(gammaln(a), gammaln(b))
  shape_term = lax.sub(gammaln(lax.add(a, b)), shape_term_tmp)
  y = lax.div(lax.sub(x, loc), scale)
  log_linear_term = lax.add(lax.mul(lax.sub(a, one), lax.log(y)),
                            lax.mul(lax.sub(b, one), lax.log1p(lax.neg(y))))
  log_probs = lax.sub(lax.add(shape_term, log_linear_term), lax.log(scale))
  return where(logical_or(lax.gt(x, lax.add(loc, scale)),
                          lax.lt(x, loc)), -inf, log_probs)

@_wraps(osp_stats.beta.pdf, update_doc=False)
def pdf(x, a, b, loc=0, scale=1):
  return lax.exp(logpdf(x, a, b, loc, scale))

