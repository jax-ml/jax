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
from ...numpy.lax_numpy import dot, subtract, einsum
from ...numpy.linalg import det, inv

@_wraps(osp_stats.multivariate_normal.logpdf, update_doc=False)
def logpdf(x, mean, cov):
  # TODO(mattjj): osp_stats.multivariate_normal.logpdf doesn't like being fed
  # empty-shape arrays, so we can't use _promote_args_like as written; consider
  # revising the dtype promotion logic here if it's an issue.
  # x, mean, cov = _promote_args_like(osp_stats.multivariate_normal.logpdf, x, mean, cov)
  x = x.astype(cov.dtype)
  mean = mean.astype(cov.dtype)
  two = _constant_like(x, 2)
  dim = _constant_like(x, mean.shape[0])
  det_sig = det(cov).astype(cov.dtype)
  log_normalizer = lax.log(lax.mul(lax.pow(_constant_like(x, 2 * onp.pi), dim), det_sig))
  x_shape = x.shape[:-1]
  if x_shape:
    x_2d = x.reshape((-1, mean.shape[0]))
    quadratic = einsum("ij,jk,ik->i", subtract(x_2d, mean), inv(cov),
                       subtract(x_2d, mean)).reshape(x_shape).astype(cov.dtype)
  else:
    quadratic = dot(dot(subtract(x, mean), inv(cov)), subtract(x, mean).T).astype(cov.dtype)
  return lax.div(lax.neg(lax.add(log_normalizer, quadratic)), two)

@_wraps(osp_stats.multivariate_normal.pdf, update_doc=False)
def pdf(x, mean, cov):
  return lax.exp(logpdf(x, mean, cov))
