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
from absl.testing import absltest
import numpy as onp
import scipy.sparse.linalg as osp_sparse

from jax import numpy as np
from jax import lax
from jax import test_util as jtu
from jax.experimental import sparse

float_types = [onp.float32, onp.float64]
complex_types = [onp.complex64, onp.complex128]
_T = lambda x: np.swapaxes(np.conj(x), -1, -2)


class LaxBackedScipyTests(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(4, 4), (7, 7), (200, 200), (1000, 1000)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  def test_linalg_sparse_cg(self, shape, dtype, rng_factory):
    def high_precision_dot(a, b):
      return lax.dot(a, b, precision=lax.Precision.HIGHEST)

    def build_and_solve(a, b, maxiter=None):
      # intentionally non-linear in a and b
      matvec = partial(high_precision_dot, a)
      return sparse.cg(matvec=matvec, b=b, maxiter=maxiter)

    def args_maker():
      rng = rng_factory()
      square_mat = rng(shape, dtype)
      b = rng((shape[0],), dtype)
      diag_mat = np.eye(N=shape[0], dtype=dtype) + 0.3
      spd_mat = np.matmul(np.matmul(square_mat, diag_mat), _T(square_mat))
      return spd_mat, b

    a, b = args_maker()
    expected = osp_sparse.cg(a, b)
    expected_1_step = osp_sparse.cg(A=a, b=b, maxiter=1)
    actual = build_and_solve(a, b)
    actual_1_step = build_and_solve(a, b, 1)
    self.assertAllClose(expected, actual, atol=1e-5, check_dtypes=True)
    self.assertAllClose(expected_1_step, actual_1_step, atol=1e-5, check_dtypes=True)
    jtu.check_grads(build_and_solve, (a, b), atol=1e-5, order=2, rtol=2e-3)

if __name__ == "__main__":
  absltest.main()