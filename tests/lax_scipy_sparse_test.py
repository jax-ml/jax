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

from functools import partial
from absl.testing import parameterized

import numpy as onp
import scipy.sparse.linalg as osp_sparse
from jax import lax
from jax import test_util as jtu
from jax.experimental import sparse

@parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}D_{}".format(dim, onp.dtype(dtype).name),
       "dim": dim, "dtype": dtype}
      for dim in [3, 5, 10]
      for dtype in [onp.float32, onp.float64]))
def test_linalg_sparse_cg(self, dim, dtype):
  def high_precision_dot(a, b):
    return lax.dot(a, b, precision=lax.Precision.HIGHEST)

  def build_and_solve(a, b):
    # intentionally non-linear in a and b
    matvec = partial(high_precision_dot, a)
    return sparse.cg(matvec, b)

  r = onp.random.RandomState(dim)
  square_mat = r.randn(dim, dim)
  a = onp.dot(square_mat, square_mat.T) + dim * onp.eye(dim)
  b = r.randn(dim)
  expected = osp_sparse.cg(a, b)
  actual = build_and_solve(a, b)
  self.assertAllClose(expected, actual, atol=1e-5, check_dtypes=True)
  jtu.check_grads(build_and_solve, (a, b), atol=1e-5, order=2, rtol=2e-3)
