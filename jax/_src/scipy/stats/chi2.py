# Copyright 2021 Google LLC
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
# limitations under the License


import scipy.stats as osp_stats

from jax import lax
from jax._src.numpy.util import _wraps
from jax._src.numpy.lax_numpy import _promote_args_inexact, _constant_like, where, inf


@_wraps(osp_stats.chi2.logpdf, update_doc=False)
def logpdf(x, df, loc=0, scale=1):
    x, df, loc, scale = _promote_args_inexact("chi2.logpdf", x, df, loc, scale)
    one = _constant_like(x, 1)
    two = _constant_like(x, 2)
    y = lax.div(lax.sub(x, loc), scale)
    df_on_two = lax.div(df, two)

    kernel = lax.sub(lax.mul(lax.sub(df_on_two, one), lax.log(y)), lax.div(y,two))

    nrml_cnst = lax.neg(lax.add(lax.lgamma(df_on_two),lax.div(lax.mul(lax.log(two), df),two)))

    log_probs = lax.add(lax.sub(nrml_cnst, lax.log(scale)), kernel)
    return where(lax.lt(x, loc), -inf, log_probs)

@_wraps(osp_stats.chi2.pdf, update_doc=False)
def pdf(x, df, loc=0, scale=1):
    return lax.exp(logpdf(x, df, loc, scale))
