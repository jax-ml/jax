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
from ...numpy.lax_numpy import _promote_args_like, _constant_like, _wraps, all, sum
from ..special import gammaln, xlogy


def _is_simplex(x):
    x_sum = sum(x, axis=-1)
    return all(x > 0, axis=-1) & (x_sum <= 1) & (x_sum > 1 - 1e-6)


@_wraps(osp_stats.dirichlet.logpdf)
def logpdf(x, alpha):
    args = (onp.ones((0,), lax.dtype(x)), onp.ones((1,), lax.dtype(alpha)))
    to_dtype = lax.dtype(osp_stats.dirichlet.logpdf(*args))
    x, alpha = [lax.convert_element_type(arg, to_dtype) for arg in (x, alpha)]
    one = _constant_like(x, 1)
    normalize_term = sum(gammaln(alpha), axis=-1) - gammaln(sum(alpha, axis=-1))
    log_probs = lax.sub(sum(xlogy(lax.sub(alpha, one), x), axis=-1), normalize_term)
    return where(_is_simplex(x), log_probs, -inf)


@_wraps(osp_stats.dirichlet.pdf)
def pdf(x, alpha):
  return lax.exp(logpdf(x, alpha))
