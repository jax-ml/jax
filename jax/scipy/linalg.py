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

from jax._src.scipy.linalg import (
  block_diag as block_diag,
  cholesky as cholesky,
  cho_factor as cho_factor,
  cho_solve as cho_solve,
  det as det,
  eigh as eigh,
  eigh_tridiagonal as eigh_tridiagonal,
  expm as expm,
  expm_frechet as expm_frechet,
  inv as inv,
  lu as lu,
  lu_factor as lu_factor,
  lu_solve as lu_solve,
  polar as polar,
  qr as qr,
  solve as solve,
  solve_triangular as solve_triangular,
  svd as svd,
  tril as tril,
  triu as triu,
)

from jax._src.lax.polar import (
  polar_unitary as polar_unitary,
)
