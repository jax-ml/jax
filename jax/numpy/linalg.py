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

from .. import lax_linalg
from .lax_numpy import _not_implemented
from .lax_numpy import _wraps
from .lax_numpy import IMPLEMENTED_FUNCS
from . import lax_numpy as np
from ..util import get_module_functions


dot = np.dot
matmul = np.matmul
trace = np.trace


@_wraps(onp.linalg.cholesky)
def cholesky(a):
  return lax_linalg.cholesky(a)


@_wraps(onp.linalg.qr)
def qr(a, mode="reduced"):
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


UNIMPLEMENTED_FUNCS = get_module_functions(onp.linalg) - set(IMPLEMENTED_FUNCS)
for func in UNIMPLEMENTED_FUNCS:
  if func.__name__ not in globals():
    globals()[func.__name__] = _not_implemented(func)
