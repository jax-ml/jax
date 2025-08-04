# Copyright 2020 The JAX Authors.
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

from jax._src.lax.linalg import (
  cholesky as cholesky,
  cholesky_p as cholesky_p,
  cholesky_update as cholesky_update,
  cholesky_update_p as cholesky_update_p,
  eig as eig,
  eig_p as eig_p,
  eigh as eigh,
  eigh_p as eigh_p,
  hessenberg as hessenberg,
  hessenberg_p as hessenberg_p,
  householder_product as householder_product,
  householder_product_p as householder_product_p,
  lu as lu,
  lu_p as lu_p,
  lu_pivots_to_permutation as lu_pivots_to_permutation,
  lu_pivots_to_permutation_p as lu_pivots_to_permutation_p,
  qr as qr,
  qr_p as qr_p,
  schur as schur,
  schur_p as schur_p,
  svd as svd,
  svd_p as svd_p,
  SvdAlgorithm as SvdAlgorithm,
  symmetric_product as symmetric_product,
  symmetric_product_p as symmetric_product_p,
  triangular_solve as triangular_solve,
  triangular_solve_p as triangular_solve_p,
  tridiagonal as tridiagonal,
  tridiagonal_p as tridiagonal_p,
  tridiagonal_solve as tridiagonal_solve,
  tridiagonal_solve_p as tridiagonal_solve_p,
)

from jax._src.tpu.linalg.qdwh import (
  qdwh as qdwh
)
