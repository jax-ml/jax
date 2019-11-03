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
from ...numpy import lax_numpy as np
from ..special import xlogy, xlog1py


@np._wraps(osp_stats.bernoulli.logpmf, update_doc=False)
def logpmf(k, p, loc=0):
  k, p, loc = np._promote_args_like(osp_stats.bernoulli.logpmf, k, p, loc)
  zero = np._constant_like(k, 0)
  one = np._constant_like(k, 1)
  x = lax.sub(k, loc)
  log_probs = xlogy(x, p) + xlog1py(lax.sub(one, x), -p)
  return np.where(np.logical_or(lax.lt(x, zero), lax.gt(x, one)),
                  -np.inf, log_probs)

@np._wraps(osp_stats.bernoulli.pmf, update_doc=False)
def pmf(k, p, loc=0):
  return np.exp(pmf(k, p, loc))
