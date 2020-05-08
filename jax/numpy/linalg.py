# Copyright 2020 Google LLC
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

from jax._src.numpy.linalg import (
  cholesky,
  cond,
  det,
  eig,
  eigh,
  eigvals,
  eigvalsh,
  inv,
  matrix_power,
  matrix_rank,
  multi_dot,
  norm,
  pinv,
  qr,
  slogdet,
  solve,
  svd,
  tensorinv,
  tensorsolve,
)

def _init():
  import numpy as np
  from jax.util import get_module_functions
  from jax._src.numpy.lax_numpy import _not_implemented
  for func in get_module_functions(np.linalg):
    if func.__name__ not in globals():
      globals()[func.__name__] = _not_implemented(func)

_init()
del _init