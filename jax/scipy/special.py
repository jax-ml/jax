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

import scipy.special as osp_special

from .. import lax
from ..numpy.lax_numpy import _wraps


# need to create new functions because _wraps sets the __name__ attribute
gammaln = _wraps(osp_special.gammaln)(lambda x: lax.lgamma(x))
digamma = _wraps(osp_special.digamma)(lambda x: lax.digamma(x))
erf = _wraps(osp_special.erf)(lambda x: lax.erf(x))
erfc = _wraps(osp_special.erfc)(lambda x: lax.erfc(x))
erfinv = _wraps(osp_special.erfinv)(lambda x: lax.erf_inv(x))
