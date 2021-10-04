# Copyright 2021 Google LLC
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
import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import config
from jax import dtypes
from jax.experimental import sparse
from jax.experimental.sparse.ops import _bcoo_nse
from jax import lax
from jax._src.lib import cusparse
from jax._src.lib import xla_bridge
from jax import jit
from jax._src import test_util as jtu
from jax import xla
import jax.numpy as jnp
from jax import jvp
import numpy as np
import scipy.sparse
config.parse_flags_with_absl()
FLAGS = config.FLAGS

MATMUL_TOL = {
  np.float32: 1E-5,
  np.float64: 1E-10,
  np.complex64: 1e-5,
  np.complex128: 1E-10,
}

all_dtypes = jtu.dtypes.integer + jtu.dtypes.floating + jtu.dtypes.complex


def rand_sparse(rng, nse=0.5, post=lambda x: x):
  def _rand_sparse(shape, dtype, nse=nse):
    rand = jtu.rand_default(rng)
    size = np.prod(shape)
    if 0 <= nse < 1:
      nse = nse * size
    nse = min(size, int(nse))
    M = rand(shape, dtype)
    indices = rng.choice(size, size - nse, replace=False)
    M.flat[indices] = 0
    return post(M)
  return _rand_sparse


class cuSparseTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_csr_todense(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)

    args = (M.data, M.indices, M.indptr)
    todense = lambda *args: sparse.csr_todense(*args, shape=M.shape)

    self.assertArraysEqual(M.toarray(), todense(*args))
    self.assertArraysEqual(M.toarray(), jit(todense)(*args))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_csr_fromdense(self, shape, dtype):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_csr = scipy.sparse.csr_matrix(M)

    nse = M_csr.nnz
    index_dtype = jnp.int32
    fromdense = lambda M: sparse.csr_fromdense(M, nse=nse, index_dtype=jnp.int32)

    data, indices, indptr = fromdense(M)
    self.assertArraysEqual(data, M_csr.data.astype(dtype))
    self.assertArraysEqual(indices, M_csr.indices.astype(index_dtype))
    self.assertArraysEqual(indptr, M_csr.indptr.astype(index_dtype))

    data, indices, indptr = jit(fromdense)(M)
    self.assertArraysEqual(data, M_csr.data.astype(dtype))
    self.assertArraysEqual(indices, M_csr.indices.astype(index_dtype))
    self.assertArraysEqual(indptr, M_csr.indptr.astype(index_dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_T={}".format(jtu.format_shape_dtype_string(shape, dtype), transpose),
       "shape": shape, "dtype": dtype, "transpose": transpose}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for transpose in [True, False]))
  def test_csr_matvec(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    v_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)
    v = v_rng(op(M).shape[1], dtype)

    args = (M.data, M.indices, M.indptr, v)
    matvec = lambda *args: sparse.csr_matvec(*args, shape=M.shape, transpose=transpose)

    self.assertAllClose(op(M) @ v, matvec(*args), rtol=MATMUL_TOL)
    self.assertAllClose(op(M) @ v, jit(matvec)(*args), rtol=MATMUL_TOL)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_T={}".format(jtu.format_shape_dtype_string(shape, dtype), transpose),
       "shape": shape, "dtype": dtype, "transpose": transpose}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for transpose in [True, False]))
  def test_csr_matmat(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    B_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)
    B = B_rng((op(M).shape[1], 4), dtype)

    args = (M.data, M.indices, M.indptr, B)
    matmat = lambda *args: sparse.csr_matmat(*args, shape=shape, transpose=transpose)

    self.assertAllClose(op(M) @ B, matmat(*args), rtol=MATMUL_TOL)
    self.assertAllClose(op(M) @ B, jit(matmat)(*args), rtol=MATMUL_TOL)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_coo_todense(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)

    args = (M.data, M.row, M.col)
    todense = lambda *args: sparse.coo_todense(*args, shape=M.shape)

    self.assertArraysEqual(M.toarray(), todense(*args))
    self.assertArraysEqual(M.toarray(), jit(todense)(*args))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_coo_fromdense(self, shape, dtype):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_coo = scipy.sparse.coo_matrix(M)

    nse = M_coo.nnz
    index_dtype = jnp.int32
    fromdense = lambda M: sparse.coo_fromdense(M, nse=nse, index_dtype=jnp.int32)

    data, row, col = fromdense(M)
    self.assertArraysEqual(data, M_coo.data.astype(dtype))
    self.assertArraysEqual(row, M_coo.row.astype(index_dtype))
    self.assertArraysEqual(col, M_coo.col.astype(index_dtype))

    data, indices, indptr = jit(fromdense)(M)
    self.assertArraysEqual(data, M_coo.data.astype(dtype))
    self.assertArraysEqual(row, M_coo.row.astype(index_dtype))
    self.assertArraysEqual(col, M_coo.col.astype(index_dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_T={}".format(jtu.format_shape_dtype_string(shape, dtype), transpose),
       "shape": shape, "dtype": dtype, "transpose": transpose}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for transpose in [True, False]))
  def test_coo_matvec(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    v_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)
    v = v_rng(op(M).shape[1], dtype)

    args = (M.data, M.row, M.col, v)
    matvec = lambda *args: sparse.coo_matvec(*args, shape=M.shape, transpose=transpose)

    self.assertAllClose(op(M) @ v, matvec(*args), rtol=MATMUL_TOL)
    self.assertAllClose(op(M) @ v, jit(matvec)(*args), rtol=MATMUL_TOL)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_T={}".format(jtu.format_shape_dtype_string(shape, dtype), transpose),
       "shape": shape, "dtype": dtype, "transpose": transpose}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for transpose in [True, False]))
  def test_coo_matmat(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    B_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)
    B = B_rng((op(M).shape[1], 4), dtype)

    args = (M.data, M.row, M.col, B)
    matmat = lambda *args: sparse.coo_matmat(*args, shape=shape, transpose=transpose)

    self.assertAllClose(op(M) @ B, matmat(*args), rtol=MATMUL_TOL)
    self.assertAllClose(op(M) @ B, jit(matmat)(*args), rtol=MATMUL_TOL)

    y, dy = jvp(lambda x: sparse.coo_matmat(M.data, M.row, M.col, x, shape=shape, transpose=transpose).sum(), (B, ), (jnp.ones_like(B), ))
    self.assertAllClose((op(M) @ B).sum(), y, rtol=MATMUL_TOL)

    y, dy = jvp(lambda x: sparse.coo_matmat(x, M.row, M.col, B, shape=shape, transpose=transpose).sum(), (M.data, ), (jnp.ones_like(M.data), ))
    self.assertAllClose((op(M) @ B).sum(), y, rtol=MATMUL_TOL)

  def test_coo_matmat_layout(self):
    # Regression test for https://github.com/google/jax/issues/7533
    d = jnp.array([1.0, 2.0, 3.0, 4.0])
    i = jnp.array([0, 0, 1, 2])
    j = jnp.array([0, 2, 0, 0])
    shape = (3, 3)

    x = jnp.arange(9).reshape(3, 3).astype(d.dtype)

    def f(x):
      return sparse.coo_matmat(d, i, j, x.T, shape=shape)

    result = f(x)
    result_jit = jit(f)(x)

    self.assertAllClose(result, result_jit)

  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  def test_gpu_translation_rule(self):
    version = xla_bridge.get_backend().platform_version
    cuda_version = None if version == "<unknown>" else int(version.split()[-1])
    if cuda_version is None or cuda_version < 11000:
      self.assertFalse(cusparse and cusparse.is_supported)
      self.assertNotIn(sparse.csr_todense_p, xla.backend_specific_translations["gpu"])
    else:
      self.assertTrue(cusparse and cusparse.is_supported)
      self.assertIn(sparse.csr_todense_p, xla.backend_specific_translations["gpu"])

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
         jtu.format_shape_dtype_string(shape, dtype), mat_type),
       "shape": shape, "dtype": dtype, "mat_type": mat_type}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for mat_type in ['csr', 'coo']))
  def test_extra_nse(self, shape, dtype, mat_type):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = (M != 0).sum() + 5
    fromdense = getattr(sparse, f"{mat_type}_fromdense")
    todense = getattr(sparse, f"{mat_type}_todense")
    args = fromdense(M, nse=nse, index_dtype=jnp.int32)
    M_out = todense(*args, shape=M.shape)
    self.assertArraysEqual(M, M_out)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_coo_todense_ad(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    data, row, col = sparse.coo_fromdense(M, nse=(M != 0).sum())
    f = lambda data: sparse.coo_todense(data, row, col, shape=M.shape)

    # Forward-mode
    primals, tangents = jax.jvp(f, [data], [jnp.ones_like(data)])
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(tangents, jnp.zeros_like(M).at[row, col].set(1))

    # Reverse-mode
    primals, vjp_fun = jax.vjp(f, data)
    data_out, = vjp_fun(primals)
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(data_out, data)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_coo_fromdense_ad(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    nse = (M != 0).sum()
    f = lambda M: sparse.coo_fromdense(M, nse=nse)

    # Forward-mode
    primals, tangents = jax.jvp(f, [M], [jnp.ones_like(M)])
    self.assertArraysEqual(primals[0], f(M)[0])
    self.assertArraysEqual(primals[1], f(M)[1])
    self.assertArraysEqual(primals[2], f(M)[2])
    self.assertArraysEqual(tangents[0], jnp.ones(nse, dtype=dtype))
    self.assertEqual(tangents[1].dtype, dtypes.float0)
    self.assertEqual(tangents[2].dtype, dtypes.float0)

    # Reverse-mode
    primals, vjp_fun = jax.vjp(f, M)
    M_out, = vjp_fun(primals)
    self.assertArraysEqual(primals[0], f(M)[0])
    self.assertArraysEqual(primals[1], f(M)[1])
    self.assertArraysEqual(primals[2], f(M)[2])
    self.assertArraysEqual(M_out, M)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
        jtu.format_shape_dtype_string(shape, dtype),
        jtu.format_shape_dtype_string(bshape, dtype)),
       "shape": shape, "dtype": dtype, "bshape": bshape}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for bshape in [shape[-1:] + s for s in [()]]  # TODO: matmul autodiff
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))  # TODO: other types

  def test_coo_matvec_ad(self, shape, dtype, bshape):
    tol = {np.float32: 1E-6, np.float64: 1E-13, np.complex64: 1E-6, np.complex128: 1E-13}

    rng = rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())

    M = rng(shape, dtype)
    data, row, col = sparse.coo_fromdense(M, nse=(M != 0).sum())
    x = rng_b(bshape, dtype)
    xdot = rng_b(bshape, dtype)

    # Forward-mode with respect to the vector
    f_dense = lambda x: M @ x
    f_sparse = lambda x: sparse.coo_matvec(data, row, col, x, shape=M.shape)
    v_sparse, t_sparse = jax.jvp(f_sparse, [x], [xdot])
    v_dense, t_dense = jax.jvp(f_dense, [x], [xdot])
    self.assertAllClose(v_sparse, v_dense, atol=tol, rtol=tol)
    self.assertAllClose(t_sparse, t_dense, atol=tol, rtol=tol)

    # Reverse-mode with respect to the vector
    primals_dense, vjp_dense = jax.vjp(f_dense, x)
    primals_sparse, vjp_sparse = jax.vjp(f_sparse, x)
    out_dense, = vjp_dense(primals_dense)
    out_sparse, = vjp_sparse(primals_sparse)
    self.assertAllClose(primals_dense[0], primals_sparse[0], atol=tol, rtol=tol)
    self.assertAllClose(out_dense, out_sparse, atol=tol, rtol=tol)

    # Forward-mode with respect to nonzero elements of the matrix
    f_sparse = lambda data: sparse.coo_matvec(data, row, col, x, shape=M.shape)
    f_dense = lambda data: sparse.coo_todense(data, row, col, shape=M.shape) @ x
    data = rng((len(data),), data.dtype)
    data_dot = rng((len(data),), data.dtype)
    v_sparse, t_sparse = jax.jvp(f_sparse, [data], [data_dot])
    v_dense, t_dense = jax.jvp(f_dense, [data], [data_dot])

    self.assertAllClose(v_sparse, v_dense, atol=tol, rtol=tol)
    self.assertAllClose(t_sparse, t_dense, atol=tol, rtol=tol)

    # Reverse-mode with respect to nonzero elements of the matrix
    primals_dense, vjp_dense = jax.vjp(f_dense, data)
    primals_sparse, vjp_sparse = jax.vjp(f_sparse, data)
    out_dense, = vjp_dense(primals_dense)
    out_sparse, = vjp_sparse(primals_sparse)
    self.assertAllClose(primals_dense[0], primals_sparse[0], atol=tol, rtol=tol)
    self.assertAllClose(out_dense, out_sparse, atol=tol, rtol=tol)

class BCOOTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_dense_round_trip(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    n_sparse = M.ndim - n_batch - n_dense
    nse = int(_bcoo_nse(M, n_batch=n_batch, n_dense=n_dense))
    data, indices = sparse.bcoo_fromdense(M, n_batch=n_batch, n_dense=n_dense)
    data_jit, indices_jit = jit(partial(sparse.bcoo_fromdense, nse=nse, n_batch=n_batch, n_dense=n_dense))(M)
    self.assertArraysEqual(data, data_jit)
    self.assertArraysEqual(indices, indices_jit)

    assert data.dtype == dtype
    assert data.shape == shape[:n_batch] + (nse,) + shape[n_batch + n_sparse:]
    assert indices.dtype == jnp.int32  # TODO: test passing this arg
    assert indices.shape == shape[:n_batch] + (nse, n_sparse)

    todense = partial(sparse.bcoo_todense, shape=shape)
    self.assertArraysEqual(M, todense(data, indices))
    self.assertArraysEqual(M, jit(todense)(data, indices))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_todense_ad(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    data, indices = sparse.bcoo_fromdense(M, n_batch=n_batch, n_dense=n_dense)

    todense = partial(sparse.bcoo_todense, indices=indices, shape=shape)
    j1 = jax.jacfwd(todense)(data)
    j2 = jax.jacrev(todense)(data)
    hess = jax.hessian(todense)(data)
    self.assertArraysAllClose(j1, j2)
    self.assertEqual(j1.shape, M.shape + data.shape)
    self.assertEqual(hess.shape, M.shape + 2 * data.shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_fromdense_ad(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = int(_bcoo_nse(M, n_batch=n_batch, n_dense=n_dense))

    def fromdense(M):
      return sparse.bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)[0]
    data = fromdense(M)

    j1 = jax.jacfwd(fromdense)(M)
    j2 = jax.jacrev(fromdense)(M)
    hess = jax.hessian(fromdense)(M)
    self.assertArraysAllClose(j1, j2)
    self.assertEqual(j1.shape, data.shape + M.shape)
    self.assertEqual(hess.shape, data.shape + 2 * M.shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_dense_round_trip_batched(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    n_sparse = M.ndim - n_batch - n_dense
    nse = int(_bcoo_nse(M, n_batch=n_batch, n_dense=n_dense))

    fromdense = partial(sparse.bcoo_fromdense, nse=nse, n_dense=n_dense)
    todense = partial(sparse.bcoo_todense, shape=shape[n_batch:])
    for i in range(n_batch):
      fromdense = jax.vmap(fromdense)
      todense = jax.vmap(todense)

    data, indices = fromdense(M)

    assert data.dtype == dtype
    assert data.shape == shape[:n_batch] + (nse,) + shape[n_batch + n_sparse:]
    assert indices.dtype == jnp.int32  # TODO: test passing this arg
    assert indices.shape == shape[:n_batch] + (nse, n_sparse)

    self.assertArraysEqual(M, todense(data, indices))
    self.assertArraysEqual(M, jit(todense)(data, indices))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_extract(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    data, indices = sparse.bcoo_fromdense(M)
    data2 = sparse.bcoo_extract(indices, M)
    self.assertArraysEqual(data, data2)
    data3 = jit(sparse.bcoo_extract)(indices, M)
    self.assertArraysEqual(data, data3)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_extract_ad(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    data, indices = sparse.bcoo_fromdense(M, n_batch=n_batch, n_dense=n_dense)

    extract = partial(sparse.bcoo_extract, indices)
    j1 = jax.jacfwd(extract)(M)
    j2 = jax.jacrev(extract)(M)
    hess = jax.hessian(extract)(M)
    self.assertArraysAllClose(j1, j2)
    self.assertEqual(j1.shape, data.shape + M.shape)
    self.assertEqual(hess.shape, data.shape + 2 * M.shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_transpose(self, shape, dtype, n_batch, n_dense):
    n_sparse = len(shape) - n_batch - n_dense
    rng = self.rng()
    sprng = rand_sparse(rng)
    M = sprng(shape, dtype)
    data, indices = sparse.bcoo_fromdense(M, n_batch=n_batch, n_dense=n_dense)

    permutation = np.concatenate([
      rng.permutation(range(n_batch)),
      rng.permutation(range(n_batch, n_batch + n_sparse)),
      rng.permutation(range(n_batch + n_sparse, len(shape)))]).astype(int)

    M_T = M.transpose(permutation)
    trans = partial(sparse.bcoo_transpose, shape=shape, permutation=permutation)
    self.assertArraysEqual(M_T, sparse.bcoo_todense(*trans(data, indices), shape=M_T.shape))
    self.assertArraysEqual(M_T, sparse.bcoo_todense(*jit(trans)(data, indices), shape=M_T.shape))

    # test batched
    def trans(M):
      return M.transpose([p - n_batch for p in permutation[n_batch:]])
    for _ in range(n_batch):
      trans = jax.vmap(trans)
    Msp = sparse.BCOO.fromdense(M, n_batch=n_batch, n_dense=n_dense)
    self.assertArraysEqual(trans(M), trans(Msp).todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_transpose_ad(self, shape, dtype, n_batch, n_dense):
    n_sparse = len(shape) - n_batch - n_dense
    rng = self.rng()
    sprng = rand_sparse(self.rng())

    M = sprng(shape, dtype)
    data, indices = sparse.bcoo_fromdense(M, n_batch=n_batch, n_dense=n_dense)

    permutation = np.concatenate([
      rng.permutation(range(n_batch)),
      rng.permutation(range(n_batch, n_batch + n_sparse)),
      rng.permutation(range(n_batch + n_sparse, len(shape)))]).astype(int)

    def f_sparse(data):
      return sparse.bcoo_transpose(data, indices, shape=shape, permutation=permutation)[0]

    jf_sparse = jax.jacfwd(f_sparse)(data)
    jr_sparse = jax.jacrev(f_sparse)(data)

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-3}

    # TODO(jakevdp) also test against dense version?
    self.assertAllClose(jf_sparse, jr_sparse, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(1, len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_todense_partial_batch(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    data, indices = sparse.bcoo_fromdense(M, n_batch=n_batch, n_dense=n_dense)

    M1 = sparse.bcoo_todense(data, indices[:1], shape=M.shape)
    M2 = sparse.bcoo_todense(data, jnp.stack(shape[0] * [indices[0]]), shape=M.shape)
    self.assertAllClose(M1, M2)

    M3 = sparse.bcoo_todense(data[:1], indices, shape=M.shape)
    M4 = sparse.bcoo_todense(jnp.stack(shape[0] * [data[0]]), indices, shape=M.shape)
    self.assertAllClose(M3, M4)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_lhs_contracting={}_rhs_contracting={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               lhs_contracting, rhs_contracting, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "lhs_contracting": lhs_contracting, "rhs_contracting": rhs_contracting,
       "n_dense": n_dense}
      for lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
          [(5,), (6,), [], []],
          [(5,), (5,), [0], [0]],
          [(5, 7), (5,), [0], [0]],
          [(7, 5), (5,), [1], [0]],
          [(3, 5), (2, 5), [1], [1]],
          [(5, 3), (5, 2), [0], [0]],
          [(5, 3, 2), (5, 2, 4), [0], [0]],
          [(5, 3, 2), (5, 2, 4), [0,2], [0,1]],
          [(5, 3, 2), (3, 5, 2, 4), [0,2], [1,2]],
          [(1, 2, 2, 3), (1, 2, 3, 1), [1], [1]],
          [(3, 2), (2, 4), [1], [0]],
      ]
      for n_dense in range(len(lhs_shape) - max(lhs_contracting, default=0))
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_dot_general_contract_only(self, lhs_shape, rhs_shape, dtype,
                                          lhs_contracting, rhs_contracting, n_dense):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())
    def args_maker():
      lhs = rng_sparse(lhs_shape, dtype)
      rhs = rng(rhs_shape, dtype)
      data, indices = sparse.bcoo_fromdense(lhs, n_dense=n_dense)
      return data, indices, lhs, rhs
    dimension_numbers = ((lhs_contracting, rhs_contracting), ([], []))

    def f_dense(data, indices, lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

    def f_sparse(data, indices, lhs, rhs):
      return sparse.bcoo_dot_general(data, indices, rhs,
                                         lhs_shape=lhs.shape,
                                         dimension_numbers=dimension_numbers)

    self._CheckAgainstNumpy(f_dense, f_sparse, args_maker)
    self._CheckAgainstNumpy(f_dense, jit(f_sparse), args_maker)
    # TODO(jakevdp): In rare cases, this fails python_should_be_executing check. Why?
    # self._CompileAndCheck(f_sparse, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_n_batch={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, n_batch, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "n_batch": n_batch, "n_dense": n_dense}
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 2, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 1),
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_dot_general_contract_and_batch(self, lhs_shape, rhs_shape, dtype,
                                               dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())
    def args_maker():
      lhs = rng_sparse(lhs_shape, dtype)
      rhs = rng(rhs_shape, dtype)
      data, indices = sparse.bcoo_fromdense(lhs, n_batch=n_batch, n_dense=n_dense)
      return data, indices, lhs, rhs

    def f_dense(data, indices, lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

    def f_sparse(data, indices, lhs, rhs):
      return sparse.bcoo_dot_general(data, indices, rhs,
                                         lhs_shape=lhs.shape,
                                         dimension_numbers=dimension_numbers)

    self._CheckAgainstNumpy(f_dense, f_sparse, args_maker)
    self._CheckAgainstNumpy(f_dense, jit(f_sparse), args_maker)
    # TODO(jakevdp): In rare cases, this fails python_should_be_executing check. Why?
    # self._CompileAndCheck(f_sparse, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_n_batch={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, n_batch, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "n_batch": n_batch, "n_dense": n_dense}
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((3, 2, 4), (3, 3, 2), (([1], [2]), ([0], [0])), 1, 0),
          ((3, 2, 4), (3, 3, 2), (([1], [2]), ([0], [0])), 2, 0),
          ((2, 3, 4), (3, 3, 2), (([0], [2]), ([1], [0])), 1, 0),
          ((2, 3, 4), (3, 3, 2), (([0], [2]), ([1], [0])), 2, 0),
          ((3, 4, 3, 2), (3, 4, 2, 4), (([3], [2]), ([0], [0])), 1, 0),
          ((3, 4, 3, 2), (3, 4, 2, 4), (([3], [2]), ([0, 1], [0, 1])), 2, 0),
          ((3, 4, 3, 2), (3, 4, 2, 4), (([3], [2]), ([0, 1], [0, 1])), 2, 1),
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_rdot_general_contract_and_batch(self, lhs_shape, rhs_shape, dtype,
                                                dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())
    def args_maker():
      lhs = rng(lhs_shape, dtype)
      rhs = rng_sparse(rhs_shape, dtype)
      data, indices = sparse.bcoo_fromdense(rhs, n_batch=n_batch, n_dense=n_dense)
      return data, indices, lhs, rhs

    def f_dense(data, indices, lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

    def f_sparse(data, indices, lhs, rhs):
      return sparse.bcoo_rdot_general(lhs, data, indices,
                                          rhs_shape=rhs.shape,
                                          dimension_numbers=dimension_numbers)

    self._CheckAgainstNumpy(f_dense, f_sparse, args_maker)
    self._CheckAgainstNumpy(f_dense, jit(f_sparse), args_maker)
    # TODO(jakevdp): In rare cases, this fails python_should_be_executing check. Why?
    # self._CompileAndCheck(f_sparse, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_n_batch={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, n_batch, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "n_batch": n_batch, "n_dense": n_dense}
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 2, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 1),
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_dot_general_partial_batch(self, lhs_shape, rhs_shape, dtype,
                                          dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())

    X = rng_sparse(lhs_shape, dtype)
    data, indices = sparse.bcoo_fromdense(X, n_batch=n_batch, n_dense=n_dense)
    Y = rng(rhs_shape, dtype)

    def f_dense(X, Y):
      return lax.dot_general(X, Y, dimension_numbers=dimension_numbers)

    def f_sparse(data, indices, Y):
      return sparse.bcoo_dot_general(data, indices, Y, lhs_shape=X.shape,
                                        dimension_numbers=dimension_numbers)

    for data, indices in itertools.product([data, data[:1]], [indices, indices[:1]]):
      X = sparse.bcoo_todense(data, indices, shape=X.shape)
      self.assertAllClose(f_dense(X, Y), f_sparse(data, indices, Y))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_n_batch={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, n_batch, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "n_batch": n_batch, "n_dense": n_dense}
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((4, 5), (5, 3), (([1], [0]), ([], [])), 0, 0),
          ((2, 4, 5), (2, 5, 3), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 0),
          # These require contraction over batch & dense dimensions
          # which is not yet implemented:
          # ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 2, 0),
          # ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 2, 0),
          # ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 1),
      ]
      for dtype in jtu.dtypes.floating))
  def test_bcoo_dot_general_ad(self, lhs_shape, rhs_shape, dtype,
                               dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())

    X = rng_sparse(lhs_shape, dtype)
    data, indices = sparse.bcoo_fromdense(X, n_batch=n_batch, n_dense=n_dense)
    Y = rng(rhs_shape, dtype)

    # gradient with respect to rhs
    def f_dense(Y):
      return lax.dot_general(X, Y, dimension_numbers=dimension_numbers)

    def f_sparse(Y):
      return sparse.bcoo_dot_general(data, indices, Y, lhs_shape=X.shape,
                                     dimension_numbers=dimension_numbers)

    jf_dense = jax.jacfwd(f_dense)(Y)
    jr_dense = jax.jacrev(f_dense)(Y)
    jf_sparse = jax.jacfwd(f_sparse)(Y)
    jr_sparse = jax.jacrev(f_sparse)(Y)

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-3}

    self.assertAllClose(jf_dense, jf_sparse, rtol=tol)
    self.assertAllClose(jr_dense, jr_sparse, rtol=tol)
    self.assertAllClose(jf_sparse, jr_sparse, rtol=tol)

    # gradient with respect to lhs
    def g_dense(X):
      return lax.dot_general(X, Y, dimension_numbers=dimension_numbers)

    def g_sparse(data):
      return sparse.bcoo_dot_general(data, indices, Y, lhs_shape=X.shape,
                                     dimension_numbers=dimension_numbers)

    jf_dense = jax.jacfwd(g_dense)(X)
    jr_dense = jax.jacrev(g_dense)(X)
    jf_sparse = jax.jacfwd(g_sparse)(data)
    jr_sparse = jax.jacrev(g_sparse)(data)

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-3}

    self.assertAllClose(jf_dense, jr_dense, rtol=tol)
    self.assertAllClose(jf_sparse, jr_sparse, rtol=tol)

    # Extract the sparse jacobian from the dense & compare.
    def extract(X):
      return sparse.bcoo_extract(indices, X)
    for i in range(g_dense(X).ndim):
      extract = jax.vmap(extract)
    self.assertAllClose(extract(jf_dense), jf_sparse, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_n_batch={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, n_batch, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "n_batch": n_batch, "n_dense": n_dense}
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 0, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 0, 1),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 0, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 1),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 0, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 1, 2),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 1),
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_dot_general_sampled(self, lhs_shape, rhs_shape, dtype, dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_default(self.rng())
    sprng = rand_sparse(self.rng())
    out_shape = lax.dot_general(
      jnp.zeros(lhs_shape), jnp.zeros(rhs_shape),
      dimension_numbers=dimension_numbers).shape

    args_maker = lambda: [
      rng(lhs_shape, dtype), rng(rhs_shape, dtype),
      sparse.BCOO.fromdense(sprng(out_shape, dtype),
                            n_batch=n_batch, n_dense=n_dense).indices]

    def dense_fun(lhs, rhs, indices):
      AB = lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)
      return sparse.bcoo_extract(indices, AB)
    def sparse_fun(lhs, rhs, indices):
      return sparse.bcoo_dot_general_sampled(
                lhs, rhs, indices, dimension_numbers=dimension_numbers)

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-3}

    self._CheckAgainstNumpy(dense_fun, sparse_fun, args_maker, tol=tol)
    # TODO: python_should_be_executing check occasionally fails... why?
    # self._CompileAndCheck(sparse_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_n_batch={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, n_batch, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "n_batch": n_batch, "n_dense": n_dense}
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 1),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 2, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 1),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 1),
      ]
      for dtype in jtu.dtypes.floating))
  def test_bcoo_dot_general_sampled_ad(self, lhs_shape, rhs_shape, dtype, dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_default(self.rng())
    sprng = rand_sparse(self.rng())
    out_shape = lax.dot_general(
      jnp.zeros(lhs_shape), jnp.zeros(rhs_shape),
      dimension_numbers=dimension_numbers).shape

    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    indices = sparse.BCOO.fromdense(sprng(out_shape, dtype),
                                    n_batch=n_batch, n_dense=n_dense).indices

    def dense_fun(lhs, rhs, indices):
      AB = lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)
      return sparse.bcoo_extract(indices, AB)
    def sparse_fun(lhs, rhs, indices):
      return sparse.bcoo_dot_general_sampled(
                lhs, rhs, indices, dimension_numbers=dimension_numbers)

    jf_dense = jax.jacfwd(dense_fun)(lhs, rhs, indices)
    jf_sparse = jax.jacfwd(sparse_fun)(lhs, rhs, indices)
    jr_dense = jax.jacrev(dense_fun)(lhs, rhs, indices)
    jr_sparse = jax.jacrev(sparse_fun)(lhs, rhs, indices)

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-3}

    self.assertAllClose(jf_sparse, jf_dense, atol=tol)
    self.assertAllClose(jr_sparse, jr_dense, atol=tol)
    self.assertAllClose(jf_sparse, jr_sparse, atol=tol)

  @unittest.skipIf(jtu.device_under_test() == "tpu", "TPU has insufficient precision")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}[n_batch={}]_{}[n_batch={}]_dims={}".format(
        jtu.format_shape_dtype_string(lhs_shape, dtype), lhs_n_batch,
        jtu.format_shape_dtype_string(rhs_shape, dtype), rhs_n_batch,
        dimension_numbers),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape,
       "lhs_n_batch": lhs_n_batch, "rhs_n_batch": rhs_n_batch,
       "dimension_numbers": dimension_numbers, "dtype": dtype}
      for lhs_shape, lhs_n_batch, rhs_shape, rhs_n_batch, dimension_numbers in [
          # (batched) vector-vector products
          ((5,), 0, (5,), 0, (([0], [0]), ([], []))),
          ((7,), 0, (7,), 0, (([0], [0]), ([], []))),
          ((5, 7), 1, (7,), 0, (([1], [0]), ([], []))),
          ((2, 3, 4), 2, (2, 4), 1, (([2], [1]), ([0], [0]))),
          ((2, 3, 4), 2, (2, 4), 1, (([2], [1]), ([], []))),
          ((2, 3, 4), 2, (3, 4), 1, (([2], [1]), ([1], [0]))),
          ((2, 3, 4), 2, (3, 4), 1, (([2], [1]), ([], []))),
          # (batched) matrix-vector products
          ((5, 7), 0, (7,), 0, (([1], [0]), ([], []))),
          ((2, 3, 4), 1, (4,), 0, (([2], [0]), ([], []))),
          ((2, 3, 4), 1, (2, 4), 1, (([2], [1]), ([0], [0]))),
          ((3, 2, 4), 1, (3, 4), 1, (([2], [1]), ([0], [0]))),
          ((2, 3, 4), 0, (2,), 0, (([0], [0]), ([], []))),
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_spdot_general(self, lhs_shape, lhs_n_batch, rhs_shape, rhs_n_batch, dtype, dimension_numbers):
    sprng = rand_sparse(self.rng())
    def args_maker():
      x = sprng(lhs_shape, dtype)
      y = sprng(rhs_shape, dtype)
      xsp = sparse.BCOO.fromdense(x, n_batch=lhs_n_batch)
      ysp = sparse.BCOO.fromdense(y, n_batch=rhs_n_batch)
      return x, y, xsp, ysp

    def f_dense(x, y, xsp, ysp):
      return lax.dot_general(x, y, dimension_numbers=dimension_numbers)

    def f_sparse(x, y, xsp, ysp):
      shape = sparse.ops._dot_general_validated_shape(x.shape, y.shape, dimension_numbers)
      data, indices = sparse.bcoo_spdot_general(xsp.data, xsp.indices, ysp.data, ysp.indices,
                                                lhs_shape=x.shape, rhs_shape=y.shape,
                                                dimension_numbers=dimension_numbers)
      return sparse.bcoo_todense(data, indices, shape=shape)

    self._CheckAgainstNumpy(f_dense, f_sparse, args_maker)
    # TODO(jakevdp): This occasionally fails python_should_be_executing check. Why?
    # self._CompileAndCheck(f_sparse, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_dedupe(self, shape, dtype, n_batch, n_dense):
    rng = self.rng()
    rng_sparse = rand_sparse(self.rng())
    M = sparse.BCOO.fromdense(rng_sparse(shape, dtype))
    for i, s in enumerate(shape[n_batch:len(shape) - n_dense]):
      M.indices = M.indices.at[..., i, :].set(rng.randint(0, s, size=M.indices.shape[-1]))
    M_dedup = M._dedupe()
    self.assertAllClose(M.todense(), M_dedup.todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}_axes={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense, axes),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense, "axes": axes}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)
      for naxes in range(len(shape))
      for axes in itertools.combinations(range(len(shape)), naxes)))
  def test_bcoo_reduce_sum(self, shape, dtype, n_batch, n_dense, axes):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    data, indices = sparse.bcoo_fromdense(M, n_batch=n_batch, n_dense=n_dense)
    data_out, indices_out, shape_out = sparse.bcoo_reduce_sum(data, indices, shape=shape, axes=axes)
    result_dense = M.sum(axes)
    result_sparse = sparse.bcoo_todense(data_out, indices_out, shape=shape_out)
    tol = {np.float32: 1E-6, np.float64: 1E-14}
    self.assertAllClose(result_dense, result_sparse, atol=tol, rtol=tol)

  @unittest.skipIf(jtu.device_under_test() == "tpu", "TPU has insufficient precision")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
        jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
        jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
      }
      for lhs_shape, rhs_shape in [[(3,), (3,)],
                                   [(3, 4), (4,)],
                                   [(4,), (4, 5)],
                                   [(3, 4), (4, 5)]]
      for lhs_dtype in all_dtypes
      for rhs_dtype in all_dtypes))
  def test_bcoo_matmul(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    lhs = jnp.array(rng(lhs_shape, lhs_dtype))
    rhs = jnp.array(rng(rhs_shape, rhs_dtype))

    out1 = lhs @ rhs
    out2 = sparse.BCOO.fromdense(lhs) @ rhs
    out3 = lhs @ sparse.BCOO.fromdense(rhs)

    tol = {np.float64: 1E-13, np.complex128: 1E-13,
           np.float32: 1E-6, np.complex64: 1E-6}
    self.assertAllClose(out1, out2, rtol=tol)
    self.assertAllClose(out1, out3, rtol=tol)

  def test_bcoo_vmap_shape(self, shape=(2, 3, 4, 5), dtype=np.float32):
    # This test checks that BCOO shape metadata interacts correctly with vmap.
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)

    def make_bcoo(M):
      return sparse.BCOO.fromdense(M, nse=np.prod(M.shape[:-1], dtype=int), n_dense=1)

    for _ in range(3):
      make_bcoo = jax.vmap(make_bcoo)
      Msp = make_bcoo(M)
      self.assertEqual(Msp.shape, M.shape)
      self.assertArraysEqual(Msp.todense(), M)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_unbatch(self, shape, dtype, n_batch, n_dense):
    rng_sparse = rand_sparse(self.rng())
    M1 = sparse.BCOO.fromdense(rng_sparse(shape, dtype), n_batch=n_batch, n_dense=n_dense)
    M2 = M1._unbatch()
    self.assertEqual(M2.n_batch, 0)
    self.assertEqual(M1.n_dense, M2.n_dense)
    self.assertEqual(M1.shape, M2.shape)
    self.assertEqual(M1.dtype, M2.dtype)
    self.assertArraysEqual(M1.todense(), M2.todense())


class SparseGradTest(jtu.JaxTestCase):
  def test_sparse_grad(self):
    rng_sparse = rand_sparse(self.rng())
    rng = jtu.rand_default(self.rng())

    y = rng(5, "float32")
    X = rng_sparse((10, 5), "float32")
    Xsp = sparse.BCOO.fromdense(X)

    def f(X, y):
      return jnp.sum(X @ y)

    grad_dense = jax.grad(f, argnums=0)(X, y)
    grad_sparse = sparse.grad(f, argnums=0)(Xsp, y)

    # extract sparse gradient from dense gradient
    indices = tuple(Xsp.indices.T)
    grad_sparse_from_dense = jnp.zeros_like(grad_dense).at[indices].set(grad_dense[indices])

    self.assertArraysEqual(grad_sparse.todense(), grad_sparse_from_dense)


class SparseObjectTest(jtu.JaxTestCase):
  def test_repr(self):
    M = sparse.BCOO.fromdense(jnp.arange(5, dtype='float32'))
    assert repr(M) == "BCOO(float32[5], nse=4)"

    M_invalid = sparse.BCOO(([], []), shape=100)
    assert repr(M_invalid) == "BCOO(<invalid>)"

  @parameterized.named_parameters(
    {"testcase_name": "_{}".format(Obj.__name__), "Obj": Obj}
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO])
  def test_attrs(self, Obj, shape=(5, 8), dtype=np.float16):
    rng = rand_sparse(self.rng(), post=Obj.fromdense)
    M = rng(shape, dtype)

    assert isinstance(M, Obj)
    assert M.shape == shape
    assert M.dtype == dtype
    assert M.nse == (M.todense() != 0).sum()
    assert M.data.dtype == dtype

    if isinstance(M, sparse.CSR):
      assert len(M.data) == len(M.indices)
      assert len(M.indptr) == M.shape[0] + 1
    elif isinstance(M, sparse.CSC):
      assert len(M.data) == len(M.indices)
      assert len(M.indptr) == M.shape[1] + 1
    elif isinstance(M, sparse.COO):
      assert len(M.data) == len(M.row) == len(M.col)
    elif isinstance(M, sparse.BCOO):
      assert M.data.shape[M.n_batch] == M.indices.shape[-2]
      assert M.indices.shape[-1] == M.n_sparse
    else:
      raise ValueError("Obj={Obj} not expected.")

  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
      {"testcase_name": "_{}_Obj={}".format(
        jtu.format_shape_dtype_string(shape, dtype), Obj.__name__),
       "shape": shape, "dtype": dtype, "Obj": Obj}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex)
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  def test_dense_round_trip(self, shape, dtype, Obj):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    Msparse = Obj.fromdense(M)
    self.assertArraysEqual(M, Msparse.todense())

  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
      {"testcase_name": "_{}_Obj={}".format(
        jtu.format_shape_dtype_string(shape, dtype), Obj.__name__),
       "shape": shape, "dtype": dtype, "Obj": Obj}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex)
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  def test_transpose(self, shape, dtype, Obj):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    Msparse = Obj.fromdense(M)
    self.assertArraysEqual(M.T, Msparse.T.todense())

  @unittest.skipIf(jtu.device_under_test() == "tpu", "TPU has insufficient precision")
  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
      {"testcase_name": "_{}_Obj={}_bshape={}".format(
        jtu.format_shape_dtype_string(shape, dtype), Obj.__name__, bshape),
       "shape": shape, "dtype": dtype, "Obj": Obj, "bshape": bshape}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for bshape in [shape[-1:] + s for s in [(), (3,), (4,)]]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex)
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  def test_matmul(self, shape, dtype, Obj, bshape):
    rng = rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())
    M = rng(shape, dtype)
    Msp = Obj.fromdense(M)
    x = rng_b(bshape, dtype)
    x = jnp.asarray(x)

    self.assertAllClose(M @ x, Msp @ x, rtol=MATMUL_TOL)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}({})".format(
        input_type.__name__,
        jtu.format_shape_dtype_string(shape, dtype)),
       "input_type": input_type, "shape": shape, "dtype": dtype}
      for input_type in [scipy.sparse.coo_matrix, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_from_scipy_sparse(self, input_type, shape, dtype):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_sparse = input_type(M)
    M_bcoo = sparse.BCOO.from_scipy_sparse(M_sparse)
    self.assertArraysEqual(M, M_bcoo.todense())

  def test_bcoo_methods(self):
    M = jnp.arange(12).reshape(3, 4)
    Msp = sparse.BCOO.fromdense(M)

    self.assertArraysEqual(-M, (-Msp).todense())

    self.assertArraysEqual(2 * M, (2 * Msp).todense())
    self.assertArraysEqual(M * 2, (Msp * 2).todense())

    self.assertArraysEqual(M + M, (Msp + Msp).todense())

    self.assertArraysEqual(M.sum(0), Msp.sum(0).todense())
    self.assertArraysEqual(M.sum(1), Msp.sum(1).todense())
    self.assertArraysEqual(M.sum(), Msp.sum())

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
