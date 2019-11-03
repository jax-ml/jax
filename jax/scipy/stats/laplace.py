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
from ...numpy.lax_numpy import _promote_args_like, _constant_like, _wraps


@_wraps(osp_stats.laplace.logpdf, update_doc=False)
def logpdf(x, loc=0, scale=1):
  x, loc, scale = _promote_args_like(osp_stats.laplace.logpdf, x, loc, scale)
  two = _constant_like(x, 2)
  linear_term = lax.div(lax.abs(lax.sub(x, loc)), scale)
  return lax.neg(lax.add(linear_term, lax.log(lax.mul(two, scale))))

@_wraps(osp_stats.laplace.pdf, update_doc=False)
def pdf(x, loc=0, scale=1):
  return lax.exp(logpdf(x, loc, scale))

@_wraps(osp_stats.laplace.cdf, update_doc=False)
def cdf(x, loc=0, scale=1):
  x, loc, scale = _promote_args_like(osp_stats.laplace.cdf, x, loc, scale)
  half = _constant_like(x, 0.5)
  one = _constant_like(x, 1)
  zero = _constant_like(x, 0)
  diff = lax.div(lax.sub(x, loc), scale)
  return lax.select(lax.le(diff, zero),
                    lax.mul(half, lax.exp(diff)),
                    lax.sub(one, lax.mul(half, lax.exp(lax.neg(diff)))))
