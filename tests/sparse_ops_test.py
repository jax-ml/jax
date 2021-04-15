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

import unittest

from absl.testing import absltest
from absl.testing import parameterized
from jax import config
from jax.experimental import sparse_ops
from jax.lib import xla_bridge
from jax import jit
from jax import test_util as jtu
from jax import xla
import jax.numpy as jnp

import numpy as np
from scipy import sparse

config.parse_flags_with_absl()
FLAGS = config.FLAGS


def rand_sparse(rng, nnz=0.1, post=lambda x: x):
  def _rand_sparse(shape, dtype, nnz=nnz):
    rand = jtu.rand_default(rng)
    size = np.prod(shape)
    if 0 <= nnz < 1:
      nnz = nnz * size
    nnz = min(size, int(nnz))
    M = rand(shape, dtype)
    indices = rng.choice(size, size - nnz, replace=False)
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
    rng = rand_sparse(self.rng(), post=sparse.csr_matrix)
    M = rng(shape, dtype)

    args = (M.data, M.indices, M.indptr)
    todense = lambda *args: sparse_ops.csr_todense(*args, shape=M.shape)

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
    M_csr = sparse.csr_matrix(M)

    nnz = M_csr.nnz
    index_dtype = jnp.int32
    fromdense = lambda M: sparse_ops.csr_fromdense(M, nnz=nnz, index_dtype=jnp.int32)

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
    rng = rand_sparse(self.rng(), post=sparse.csr_matrix)
    M = rng(shape, dtype)
    v = v_rng(op(M).shape[1], dtype)

    args = (M.data, M.indices, M.indptr, v)
    matvec = lambda *args: sparse_ops.csr_matvec(*args, shape=M.shape, transpose=transpose)

    self.assertAllClose(op(M) @ v, matvec(*args))
    self.assertAllClose(op(M) @ v, jit(matvec)(*args))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_T={}".format(jtu.format_shape_dtype_string(shape, dtype), transpose),
       "shape": shape, "dtype": dtype, "transpose": transpose}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for transpose in [True, False]))
  def test_csr_matmat(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    B_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=sparse.csr_matrix)
    M = rng(shape, dtype)
    B = B_rng((op(M).shape[1], 4), dtype)

    args = (M.data, M.indices, M.indptr, B)
    matmat = lambda *args: sparse_ops.csr_matmat(*args, shape=shape, transpose=transpose)

    self.assertAllClose(op(M) @ B, matmat(*args))
    self.assertAllClose(op(M) @ B, jit(matmat)(*args))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_coo_todense(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=sparse.coo_matrix)
    M = rng(shape, dtype)

    args = (M.data, M.row, M.col)
    todense = lambda *args: sparse_ops.coo_todense(*args, shape=M.shape)

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
    M_coo = sparse.coo_matrix(M)

    nnz = M_coo.nnz
    index_dtype = jnp.int32
    fromdense = lambda M: sparse_ops.coo_fromdense(M, nnz=nnz, index_dtype=jnp.int32)

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
    rng = rand_sparse(self.rng(), post=sparse.coo_matrix)
    M = rng(shape, dtype)
    v = v_rng(op(M).shape[1], dtype)

    args = (M.data, M.row, M.col, v)
    matvec = lambda *args: sparse_ops.coo_matvec(*args, shape=M.shape, transpose=transpose)

    self.assertAllClose(op(M) @ v, matvec(*args))
    self.assertAllClose(op(M) @ v, jit(matvec)(*args))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_T={}".format(jtu.format_shape_dtype_string(shape, dtype), transpose),
       "shape": shape, "dtype": dtype, "transpose": transpose}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for transpose in [True, False]))
  def test_coo_matmat(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    B_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=sparse.coo_matrix)
    M = rng(shape, dtype)
    B = B_rng((op(M).shape[1], 4), dtype)

    args = (M.data, M.row, M.col, B)
    matmat = lambda *args: sparse_ops.coo_matmat(*args, shape=shape, transpose=transpose)

    self.assertAllClose(op(M) @ B, matmat(*args))
    self.assertAllClose(op(M) @ B, jit(matmat)(*args))

  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  def test_gpu_translation_rule(self):
    version = xla_bridge.get_backend().platform_version
    cuda_version = None if version == "<unknown>" else int(version.split()[-1])
    if cuda_version is None or cuda_version < 11000:
      self.assertNotIn(sparse_ops.csr_todense_p, xla.backend_specific_translations["gpu"])
    else:
      self.assertIn(sparse_ops.csr_todense_p, xla.backend_specific_translations["gpu"])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
