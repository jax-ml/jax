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

from functools import partial
from absl.testing import parameterized
from absl.testing import absltest

import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg
from jax import lax
from jax import test_util as jtu
from jax.experimental import sparse

float_types = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]
_H = lambda x: np.swapaxes(x, -1, -2).conj()


def dot_high_precision(a, b):
  return lax.dot(a, b, precision=lax.Precision.HIGHEST)


class LaxBackedScipyTests(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for shape in [(4, 4), (7, 7), (32, 32)]
      for dtype in float_types + complex_types
      for rng_factory in [jtu.rand_default]))
  def test_linalg_sparse_cg(self, shape, dtype, rng_factory):

    def scipy_cg(a, b, maxiter=None, tol=0.0, atol=0.0):
      x, _ = scipy.sparse.linalg.cg(
          a, b, maxiter=maxiter, tol=tol, atol=atol)
      return x

    def lax_cg(a, b, maxiter=None, tol=0.0, atol=0.0):
      matvec = lambda x: dot_high_precision(a, x)
      x, _ = sparse.cg(matvec=matvec, b=b, maxiter=maxiter, tol=tol, atol=atol)
      return x

    def args_maker():
      rng = rng_factory()
      b = rng((shape[0],), dtype)
      square_mat = np.eye(N=shape[0], dtype=dtype) + rng(shape, dtype)
      spd_mat = np.matmul(square_mat, _H(square_mat))
      return spd_mat, b

    self._CheckAgainstNumpy(
        partial(scipy_cg, maxiter=1),
        partial(lax_cg, maxiter=1),
        args_maker,
        check_dtypes=True)

    self._CheckAgainstNumpy(
        partial(scipy_cg, maxiter=3),
        partial(lax_cg, maxiter=3),
        args_maker,
        check_dtypes=True)

    self._CheckAgainstNumpy(
        partial(scipy_cg, atol=1e-6),
        partial(lax_cg, atol=1e-6),
        args_maker,
        check_dtypes=True,
        tol=1e-4)

    # TODO(shoyer): figure out why calculating gradients appears to crash XLA
    # rng = rng_factory()
    # a = rng(shape, dtype)
    # b = rng((shape[0],), dtype)
    # jtu.check_grads(
    #     lambda x, y: lax_cg(dot_high_precision(x, x), y),
    #     (a, b),
    #     order=2)


if __name__ == "__main__":
  absltest.main()
