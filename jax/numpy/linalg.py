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

# flake8: noqa: F401

from jax._src.numpy.linalg import (
  cholesky,
  det,
  eig,
  eigh,
  eigvals,
  eigvalsh,
  inv,
  lstsq,
  matrix_power,
  matrix_rank,
  norm,
  pinv,
  qr,
  slogdet,
  solve,
  svd,
)
from jax._src.third_party.numpy.linalg import (
  cond,
  multi_dot,
  tensorinv,
  tensorsolve
)

# Module initialization is encapsulated in a function to avoid accidental
# namespace pollution.
_NOT_IMPLEMENTED = []
def _init():
  import numpy as np
  from jax._src.numpy import lax_numpy
  from jax._src import util
  # Builds a set of all unimplemented NumPy functions.
  for name, func in util.get_module_functions(np.linalg).items():
    if name not in globals():
      _NOT_IMPLEMENTED.append(name)
      globals()[name] = lax_numpy._not_implemented(func)

_init()
del _init
