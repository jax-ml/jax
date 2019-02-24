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
import scipy.special as osp_special

from .. import lax
from ..numpy.lax_numpy import _wraps, asarray, _reduction_dims, _constant_like


# need to create new functions because _wraps sets the __name__ attribute
gammaln = _wraps(osp_special.gammaln)(lambda x: lax.lgamma(x))
digamma = _wraps(osp_special.digamma)(lambda x: lax.digamma(x))
erf = _wraps(osp_special.erf)(lambda x: lax.erf(x))
erfc = _wraps(osp_special.erfc)(lambda x: lax.erfc(x))
erfinv = _wraps(osp_special.erfinv)(lambda x: lax.erf_inv(x))


@_wraps(osp_special.logit)
def logit(x):
  x = asarray(x)
  return lax.log(lax.div(x, lax.sub(lax._const(x, 1), x)))


@_wraps(osp_special.expit)
def expit(x):
  x = asarray(x)
  one = lax._const(x, 1)
  return lax.div(one, lax.add(one, lax.exp(lax.neg(x))))


@_wraps(osp_special.logsumexp)
def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
  if b is not None or return_sign:
    raise NotImplementedError("Only implemented for b=None, return_sign=False")
  dims = _reduction_dims(a, axis)
  shape = lax.subvals(onp.shape(a), zip(dims, (1,) * len(dims)))
  dimadd = lambda x: lax.reshape(x, shape)
  amax = lax.reduce(a, _constant_like(a, -onp.inf), lax.max, dims)
  amax_singletons = dimadd(amax)
  out = lax.add(lax.log(lax.reduce(lax.exp(lax.sub(a, amax_singletons)),
                                   _constant_like(a, 0), lax.add, dims)), amax)
  return dimadd(out) if keepdims else out