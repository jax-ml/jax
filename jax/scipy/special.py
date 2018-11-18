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


def _wraps(fun):
  """Like functools.wraps but works with numpy.ufuncs."""
  docstr = """
  LAX-backed implementation of {fun}. Corresponding Scipy docstring below.

  {np_doc}
  """.format(fun=fun.__name__, np_doc=fun.__doc__)
  def wrap(op):
    try:
      op.__name__ = fun.__name__
      op.__doc__ = docstr
    finally:
      return op
  return wrap


gammaln = _wraps(osp_special.gammaln)(lax.lgamma)
digamma = _wraps(osp_special.digamma)(lax.digamma)
erf = _wraps(osp_special.erf)(lax.erf)
erfc = _wraps(osp_special.erfc)(lax.erfc)
erfinv = _wraps(osp_special.erfinv)(lax.erf_inv)
