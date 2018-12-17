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
import warnings

from .. import lax_linalg
from .lax_numpy import _not_implemented
from .lax_numpy import _wraps
from . import lax_numpy as np
from ..util import get_module_functions

_EXPERIMENTAL_WARNING = "numpy.linalg support is experimental and may cause silent failures or wrong outputs"

_T = lambda x: np.swapaxes(x, -1, -2)

@_wraps(onp.linalg.cholesky)
def cholesky(a):
  warnings.warn(_EXPERIMENTAL_WARNING)
  return lax_linalg.cholesky(a)

@_wraps(onp.linalg.inv)
def inv(a):
  warnings.warn(_EXPERIMENTAL_WARNING)
  if np.ndim(a) < 2 or a.shape[-1] != a.shape[-2]:
    raise ValueError("Argument to inv must have shape [..., n, n], got {}."
      .format(np.shape(a)))
  q, r = qr(a)
  return lax_linalg.triangular_solve(r, _T(q), lower=False, left_side=True)


@_wraps(onp.linalg.qr)
def qr(a, mode="reduced"):
  warnings.warn(_EXPERIMENTAL_WARNING)
  if mode in ("reduced", "r", "full"):
    full_matrices = False
  elif mode == "complete":
    full_matrices = True
  else:
    raise ValueError("Unsupported QR decomposition mode '{}'".format(mode))
  q, r = lax_linalg.qr(a, full_matrices)
  if mode == "r":
    return r
  return q, r

for func in get_module_functions(onp.linalg):
  if func.__name__ not in globals():
    globals()[func.__name__] = _not_implemented(func)
