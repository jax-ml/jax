# Copyright 2021 The JAX Authors.
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

from collections.abc import Iterable, Iterator, Sequence
import contextlib
from functools import partial
import itertools
import math
import operator
import platform
import random
import unittest
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.random
from jax import config
from jax import dtypes
from jax.experimental import sparse
from jax.experimental.sparse import coo as sparse_coo
from jax.experimental.sparse import csr as sparse_csr
from jax.experimental.sparse import bcoo as sparse_bcoo
from jax.experimental.sparse import bcsr as sparse_bcsr
from jax.experimental.sparse import util as sparse_util
from jax.experimental.sparse import test_util as sptu
from jax.experimental.sparse import _lowerings
from jax import lax
from jax._src import xla_bridge
from jax._src.lib import gpu_sparse
from jax._src.util import unzip2
from jax import jit
from jax import tree_util
from jax import vmap
from jax._src import test_util as jtu
from jax._src.lax.lax import remaining, DotDimensionNumbers
from jax.interpreters import mlir
import jax.numpy as jnp
from jax.util import split_list
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

GPU_LOWERING_ENABLED = gpu_sparse and (gpu_sparse.cuda_is_supported or
                                       gpu_sparse.rocm_is_supported)

COMPATIBLE_SHAPE_PAIRS = [
  [(), ()],
  [(), (1,)],
  [(3,), (1, 3)],
  [(3, 1), (3,)],
  [(6,), (2, 3)],
  [(3, 2), (6,)],
  [(2, 3), (1, 6)],
  [(2, 4), (4, 1, 2)],
  [(3, 4, 5), (2, 6, 5)],
  [(2,), (2,)]
]


class BatchedDotGeneralProperties(NamedTuple):
  lhs_shape: tuple[int, ...]
  rhs_shape: tuple[int, ...]
  n_batch: int
  n_dense: int
  dimension_numbers: DotDimensionNumbers


def _iter_subsets(s: Sequence) -> Iterable[tuple]:
  """Return an iterator over all subsets of a sequence s"""
  return itertools.chain.from_iterable(itertools.combinations(s, n) for n in range(len(s) + 1))


class SparseLayout(NamedTuple):
  n_batch: int
  n_dense: int
  n_sparse: int


def iter_sparse_layouts(shape: Sequence[int], min_n_batch=0) -> Iterator[SparseLayout]:
  for n_batch in range(min_n_batch, len(shape) + 1):
    for n_dense in range(len(shape) + 1 - n_batch):
      n_sparse = len(shape) - n_batch - n_dense
      yield SparseLayout(n_batch=n_batch, n_sparse=n_sparse, n_dense=n_dense)

def iter_bcsr_layouts(shape: Sequence[int], min_n_batch=0) -> Iterator[SparseLayout]:
  n_sparse = 2
  for n_batch in range(min_n_batch, len(shape) - 1):
    n_dense = len(shape) - n_sparse - n_batch
    yield SparseLayout(n_batch=n_batch, n_sparse=n_sparse, n_dense=n_dense)


def _generate_batched_dot_general_properties(
    shapes=((5,), (2, 3), (2, 3, 4), (2, 3, 4, 4)),
    sparse_format='bcoo') -> BatchedDotGeneralProperties:
  """Generator of properties for bcoo_dot_general tests."""
  rng = random.Random(0)

  if sparse_format not in ['bcoo', 'bcsr']:
    raise ValueError(f"Sparse format {sparse_format} not supported.")

  for shape in shapes:
    for layout in iter_sparse_layouts(shape):
      if sparse_format == "bcsr" and layout.n_sparse != 2:
        continue
      subsets = split_list(range(len(shape)), [layout.n_batch, layout.n_sparse])
      for batch_dims in _iter_subsets(range(layout.n_batch)):
        for contracting_dims in _iter_subsets(remaining(range(layout.n_batch + layout.n_sparse), batch_dims)):
          # We want coverage of permutations without generating hundreds of thousands of test cases;
          # we do this by deterministic pseudo-random sampling instead of iterating.
          rhs_permute = rng.sample(range(len(shape)), len(shape))
          lhs_permute = list(itertools.chain.from_iterable(
            rng.sample(subset, len(subset)) for subset in subsets))
          yield BatchedDotGeneralProperties(
            lhs_shape=tuple(shape[p] for p in lhs_permute),
            rhs_shape=tuple(shape[p] for p in rhs_permute),
            n_batch=layout.n_batch,
            n_dense=layout.n_dense,
            dimension_numbers=(
              ([lhs_permute.index(d) for d in contracting_dims], [rhs_permute.index(d) for d in contracting_dims]),
              ([lhs_permute.index(d) for d in batch_dims], [rhs_permute.index(d) for d in batch_dims])
            ),
          )


def _generate_bcoo_dot_general_sampled_properties(shapes=((5,), (2, 3), (2, 3, 4), (2, 3, 4, 4))) -> BatchedDotGeneralProperties:
  """Generator of properties for bcoo_dot_general_sampled tests."""
  rng = random.Random(0)

  for shape in shapes:
    for batch_dims in _iter_subsets(range(len(shape))):
      for contracting_dims in _iter_subsets(remaining(range(len(shape)), batch_dims)):
        # We want coverage of permutations without generating hundreds of thousands of test cases;
        # we do this by deterministic pseudo-random sampling instead of iterating.
        lhs_permute = rng.sample(range(len(shape)), len(shape))
        rhs_permute = rng.sample(range(len(shape)), len(shape))
        lhs_shape = tuple(shape[p] for p in lhs_permute)
        rhs_shape = tuple(shape[p] for p in rhs_permute)
        dimension_numbers = (
          ([lhs_permute.index(d) for d in contracting_dims], [rhs_permute.index(d) for d in contracting_dims]),
          ([lhs_permute.index(d) for d in batch_dims], [rhs_permute.index(d) for d in batch_dims])
        )
        out = jax.eval_shape(partial(lax.dot_general, dimension_numbers=dimension_numbers),
                             jax.ShapeDtypeStruct(lhs_shape, 'float32'), jax.ShapeDtypeStruct(rhs_shape, 'float32'))
        for layout in iter_sparse_layouts(out.shape):
          yield BatchedDotGeneralProperties(
            lhs_shape=lhs_shape, rhs_shape=rhs_shape,
            n_batch=layout.n_batch, n_dense=layout.n_dense,
            dimension_numbers=dimension_numbers)


all_dtypes = jtu.dtypes.integer + jtu.dtypes.floating + jtu.dtypes.complex


def rand_sparse(rng, nse=0.5, post=lambda x: x, rand_method=jtu.rand_default):
  def _rand_sparse(shape, dtype, nse=nse):
    rand = rand_method(rng)
    size = math.prod(shape)
    if 0 <= nse < 1:
      nse = nse * size
    nse = min(size, int(nse))
    M = rand(shape, dtype)
    indices = rng.choice(size, size - nse, replace=False)
    M.flat[indices] = 0
    return post(M)
  return _rand_sparse


def _is_required_cuda_version_satisfied(cuda_version):
  version = xla_bridge.get_backend().platform_version
  if version == "<unknown>" or version.split()[0] == "rocm":
    return False
  else:
    return int(version.split()[-1]) >= cuda_version


@unittest.skipIf(platform.system() == "Windows", "Test crashes on Windows")
class cuSparseTest(sptu.SparseTestCase):
  def gpu_dense_conversion_warning_context(self, dtype):
    if jtu.device_under_test() == "gpu" and np.issubdtype(dtype, np.integer):
      return self.assertWarns(sparse.CuSparseEfficiencyWarning)
    return contextlib.nullcontext()

  def gpu_matmul_dtype_warning_context(self, dtype):
    if jtu.device_under_test() == "gpu" and dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
      return self.assertWarns(sparse.CuSparseEfficiencyWarning)
    return contextlib.nullcontext()

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=all_dtypes,
  )
  def test_csr_todense(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)

    args = (M.data, M.indices, M.indptr)
    todense = lambda *args: sparse_csr._csr_todense(*args, shape=M.shape)

    with self.gpu_dense_conversion_warning_context(dtype):
      self.assertArraysEqual(M.toarray(), todense(*args))
      self.assertArraysEqual(M.toarray(), jit(todense)(*args))

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_csr_todense_ad(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    data, indices, indptr = sparse_csr._csr_fromdense(M, nse=(M != 0).sum())
    row, col = sparse_util._csr_to_coo(indices, indptr)
    f = lambda data: sparse_csr._csr_todense(data, indices, indptr, shape=M.shape)

    # Forward-mode
    primals, tangents = jax.jvp(f, [data], [jnp.ones_like(data)])
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(tangents, jnp.zeros_like(M).at[row, col].set(1))

    # Reverse-mode
    primals, vjp_fun = jax.vjp(f, data)
    data_out, = vjp_fun(primals)
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(data_out, data)

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_csr_fromdense_ad(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    nse = (M != 0).sum()
    f = lambda M: sparse_csr._csr_fromdense(M, nse=nse)

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

  @jtu.sample_product(
    [dict(shape=shape, bshape=bshape)
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for bshape in [shape[-1:] + s for s in [(), (1,), (3,)]]
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def test_csr_matmul_ad(self, shape, dtype, bshape):
    csr_matmul = sparse_csr._csr_matvec if len(bshape) == 1 else sparse_csr._csr_matmat
    tol = {np.float32: 2E-5, np.float64: 1E-12, np.complex64: 1E-5,
           np.complex128: 1E-12}

    rng = rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())

    M = rng(shape, dtype)
    data, indices, indptr = sparse_csr._csr_fromdense(M, nse=(M != 0).sum())
    x = rng_b(bshape, dtype)
    xdot = rng_b(bshape, dtype)

    # Forward-mode with respect to the vector
    f_dense = lambda x: M @ x
    f_sparse = lambda x: csr_matmul(data, indices, indptr, x, shape=M.shape)
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
    f_sparse = lambda data: csr_matmul(data, indices, indptr, x, shape=M.shape)
    f_dense = lambda data: sparse_csr._csr_todense(data, indices, indptr, shape=M.shape) @ x
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

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=all_dtypes,
  )
  def test_csr_fromdense(self, shape, dtype):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_csr = scipy.sparse.csr_matrix(M)

    nse = M_csr.nnz
    index_dtype = jnp.int32
    fromdense = lambda M: sparse_csr._csr_fromdense(M, nse=nse, index_dtype=jnp.int32)

    with self.gpu_dense_conversion_warning_context(dtype):
      data, indices, indptr = fromdense(M)
    self.assertArraysEqual(data, M_csr.data.astype(dtype))
    self.assertArraysEqual(indices, M_csr.indices.astype(index_dtype))
    self.assertArraysEqual(indptr, M_csr.indptr.astype(index_dtype))

    with self.gpu_dense_conversion_warning_context(dtype):
      data, indices, indptr = jit(fromdense)(M)
    self.assertArraysEqual(data, M_csr.data.astype(dtype))
    self.assertArraysEqual(indices, M_csr.indices.astype(index_dtype))
    self.assertArraysEqual(indptr, M_csr.indptr.astype(index_dtype))

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=all_dtypes,
    transpose=[True, False],
  )
  def test_csr_matvec(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    v_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)
    v = v_rng(op(M).shape[1], dtype)

    args = (M.data, M.indices, M.indptr, v)
    matvec = lambda *args: sparse_csr._csr_matvec(*args, shape=M.shape, transpose=transpose)

    with self.gpu_matmul_dtype_warning_context(dtype):
      self.assertAllClose(op(M) @ v, matvec(*args), rtol=MATMUL_TOL)
      self.assertAllClose(op(M) @ v, jit(matvec)(*args), rtol=MATMUL_TOL)

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=all_dtypes,
    transpose=[True, False],
  )
  def test_csr_matmat(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    B_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)
    B = B_rng((op(M).shape[1], 4), dtype)

    args = (M.data, M.indices, M.indptr, B)
    matmat = lambda *args: sparse_csr._csr_matmat(*args, shape=shape, transpose=transpose)

    with self.gpu_matmul_dtype_warning_context(dtype):
      self.assertAllClose(op(M) @ B, matmat(*args), rtol=MATMUL_TOL)
      self.assertAllClose(op(M) @ B, jit(matmat)(*args), rtol=MATMUL_TOL)

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=all_dtypes,
  )
  def test_coo_todense(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)

    args = (M.data, M.row, M.col)
    todense = lambda *args: sparse_coo._coo_todense(*args, spinfo=sparse_coo.COOInfo(shape=M.shape, rows_sorted=True))

    with self.gpu_dense_conversion_warning_context(dtype):
      self.assertArraysEqual(M.toarray(), todense(*args))
      self.assertArraysEqual(M.toarray(), jit(todense)(*args))

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=all_dtypes,
  )
  def test_coo_fromdense(self, shape, dtype):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_coo = scipy.sparse.coo_matrix(M)

    nse = M_coo.nnz
    index_dtype = jnp.int32
    fromdense = lambda M: sparse_coo._coo_fromdense(M, nse=nse, index_dtype=jnp.int32)

    with self.gpu_dense_conversion_warning_context(dtype):
      data, row, col = fromdense(M)
    self.assertArraysEqual(data, M_coo.data.astype(dtype))
    self.assertArraysEqual(row, M_coo.row.astype(index_dtype))
    self.assertArraysEqual(col, M_coo.col.astype(index_dtype))

    with self.gpu_dense_conversion_warning_context(dtype):
      data, row, col = jit(fromdense)(M)
    self.assertArraysEqual(data, M_coo.data.astype(dtype))
    self.assertArraysEqual(row, M_coo.row.astype(index_dtype))
    self.assertArraysEqual(col, M_coo.col.astype(index_dtype))

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=all_dtypes,
    transpose=[True, False],
  )
  def test_coo_matvec(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    v_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)
    v = v_rng(op(M).shape[1], dtype)

    args = (M.data, M.row, M.col, v)
    matvec = lambda *args: sparse_coo._coo_matvec(*args, spinfo=sparse_coo.COOInfo(shape=M.shape, rows_sorted=True), transpose=transpose)

    with self.gpu_matmul_dtype_warning_context(dtype):
      self.assertAllClose(op(M) @ v, matvec(*args), rtol=MATMUL_TOL)
      self.assertAllClose(op(M) @ v, jit(matvec)(*args), rtol=MATMUL_TOL)

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=all_dtypes,
    transpose=[True, False],
  )
  def test_coo_matmat(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    B_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)
    B = B_rng((op(M).shape[1], 4), dtype)

    args = (M.data, M.row, M.col, B)
    matmat = lambda *args: sparse_coo._coo_matmat(*args, spinfo=sparse_coo.COOInfo(shape=shape, rows_sorted=True), transpose=transpose)

    with self.gpu_matmul_dtype_warning_context(dtype):
      self.assertAllClose(op(M) @ B, matmat(*args), rtol=MATMUL_TOL)
      self.assertAllClose(op(M) @ B, jit(matmat)(*args), rtol=MATMUL_TOL)

  def test_coo_matmat_layout(self):
    # Regression test for https://github.com/google/jax/issues/7533
    d = jnp.array([1.0, 2.0, 3.0, 4.0])
    i = jnp.array([0, 0, 1, 2])
    j = jnp.array([0, 2, 0, 0])
    shape = (3, 3)

    x = jnp.arange(9).reshape(3, 3).astype(d.dtype)

    def f(x):
      return sparse_coo._coo_matmat(d, i, j, x.T, spinfo=sparse_coo.COOInfo(shape=shape, rows_sorted=True))

    result = f(x)
    result_jit = jit(f)(x)

    self.assertAllClose(result, result_jit)

  def test_coo_sorted_indices(self):
    rng = self.rng()
    sprng = rand_sparse(rng)

    mat = sparse.COO.fromdense(sprng((5, 6), np.float32))
    perm = rng.permutation(mat.nse)
    mat_unsorted = sparse.COO((mat.data[perm], mat.row[perm], mat.col[perm]), shape=mat.shape)
    mat_resorted = mat_unsorted._sort_indices()
    self.assertArraysEqual(mat.todense(), mat_resorted.todense())

  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  def test_coo_sorted_indices_gpu_lowerings(self):
    dtype = jnp.float32

    mat = jnp.arange(12, dtype=dtype).reshape(4, 3)

    mat_rows_sorted = sparse.COO.fromdense(mat)
    self.assertTrue(mat_rows_sorted._rows_sorted)
    self.assertFalse(mat_rows_sorted._cols_sorted)

    mat_cols_sorted = sparse.COO.fromdense(mat.T).T
    self.assertFalse(mat_cols_sorted._rows_sorted)
    self.assertTrue(mat_cols_sorted._cols_sorted)

    mat_unsorted = sparse.COO(mat_rows_sorted._bufs, shape=mat_rows_sorted.shape)
    self.assertFalse(mat_unsorted._rows_sorted)
    self.assertFalse(mat_unsorted._cols_sorted)

    self.assertArraysEqual(mat, mat_rows_sorted._sort_indices().todense())
    self.assertArraysEqual(mat, mat_cols_sorted._sort_indices().todense())
    self.assertArraysEqual(mat, mat_unsorted._sort_indices().todense())

    todense = jit(sparse.coo_todense)
    with self.assertNoWarnings():
      dense_rows_sorted = todense(mat_rows_sorted)
      dense_cols_sorted = todense(mat_cols_sorted)
      dense_unsorted = todense(mat_unsorted._sort_indices())
    with self.assertWarnsRegex(sparse.CuSparseEfficiencyWarning, "coo_todense GPU lowering requires matrices with sorted rows.*"):
      dense_unsorted_fallback = todense(mat_unsorted)
    self.assertArraysEqual(mat, dense_rows_sorted)
    self.assertArraysEqual(mat, dense_cols_sorted)
    self.assertArraysEqual(mat, dense_unsorted)
    self.assertArraysEqual(mat, dense_unsorted_fallback)

    rhs_vec = jnp.arange(3, dtype=dtype)
    matvec = jit(sparse.coo_matvec)
    matvec_expected = mat @ rhs_vec
    with self.assertNoWarnings():
      matvec_rows_sorted = matvec(mat_rows_sorted, rhs_vec)
      matvec_cols_sorted = matvec(mat_cols_sorted, rhs_vec)
      matvec_unsorted = matvec(mat_unsorted._sort_indices(), rhs_vec)
    with self.assertWarnsRegex(sparse.CuSparseEfficiencyWarning, "coo_matvec GPU lowering requires matrices with sorted rows.*"):
      matvec_unsorted_fallback = matvec(mat_unsorted, rhs_vec)
    self.assertArraysEqual(matvec_expected, matvec_rows_sorted)
    self.assertArraysEqual(matvec_expected, matvec_cols_sorted)
    self.assertArraysEqual(matvec_expected, matvec_unsorted)
    self.assertArraysEqual(matvec_expected, matvec_unsorted_fallback)

    rhs_mat = jnp.arange(6, dtype=dtype).reshape(3, 2)
    matmat = jit(sparse.coo_matmat)
    matmat_expected = mat @ rhs_mat
    with self.assertNoWarnings():
      matmat_rows_sorted = matmat(mat_rows_sorted, rhs_mat)
      matmat_cols_sorted = matmat(mat_cols_sorted, rhs_mat)
      matmat_unsorted = matmat(mat_unsorted._sort_indices(), rhs_mat)
    with self.assertWarnsRegex(sparse.CuSparseEfficiencyWarning, "coo_matmat GPU lowering requires matrices with sorted rows.*"):
      matmat_unsorted_fallback = matmat(mat_unsorted, rhs_mat)
    self.assertArraysEqual(matmat_expected, matmat_rows_sorted)
    self.assertArraysEqual(matmat_expected, matmat_cols_sorted)
    self.assertArraysEqual(matmat_expected, matmat_unsorted)
    self.assertArraysEqual(matmat_expected, matmat_unsorted_fallback)

  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  def test_gpu_translation_rule(self):
    version = xla_bridge.get_backend().platform_version
    if version.split()[0] != "rocm":
      cuda_version = None if version == "<unknown>" else int(
          version.split()[-1])
      if cuda_version is None or cuda_version < 11000:
        self.assertFalse(gpu_sparse and gpu_sparse.cuda_is_supported)
        self.assertNotIn(sparse.csr_todense_p,
                         mlir._platform_specific_lowerings["cuda"])
      else:
        self.assertTrue(gpu_sparse and gpu_sparse.cuda_is_supported)
        self.assertIn(sparse.csr_todense_p,
                      mlir._platform_specific_lowerings["cuda"])
    else:
      self.assertTrue(gpu_sparse and gpu_sparse.rocm_is_supported)
      self.assertIn(sparse.csr_todense_p,
                    mlir._platform_specific_lowerings["rocm"])

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_coo_todense_ad(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    data, row, col = sparse_coo._coo_fromdense(M, nse=(M != 0).sum())
    f = lambda data: sparse_coo._coo_todense(data, row, col, spinfo=sparse_coo.COOInfo(shape=M.shape, rows_sorted=True))

    # Forward-mode
    primals, tangents = jax.jvp(f, [data], [jnp.ones_like(data)])
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(tangents, jnp.zeros_like(M).at[row, col].set(1))

    # Reverse-mode
    primals, vjp_fun = jax.vjp(f, data)
    data_out, = vjp_fun(primals)
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(data_out, data)

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_coo_fromdense_ad(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    nse = (M != 0).sum()
    f = lambda M: sparse_coo._coo_fromdense(M, nse=nse)

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

  @jtu.sample_product(
    [dict(shape=shape, bshape=bshape)
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for bshape in [shape[-1:] + s for s in [(), (1,), (3,)]]
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def test_coo_matmul_ad(self, shape, dtype, bshape):
    coo_matmul = sparse_coo._coo_matvec if len(bshape) == 1 else sparse_coo._coo_matmat
    tol = {np.float32: 1E-5, np.float64: 1E-12, np.complex64: 1E-5, np.complex128: 1E-12}

    rng = rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())

    M = rng(shape, dtype)
    data, row, col = sparse_coo._coo_fromdense(M, nse=(M != 0).sum())
    x = rng_b(bshape, dtype)
    xdot = rng_b(bshape, dtype)
    spinfo = sparse_coo.COOInfo(shape=M.shape, rows_sorted=True)

    # Forward-mode with respect to the vector
    f_dense = lambda x: M @ x
    f_sparse = lambda x: coo_matmul(data, row, col, x, spinfo=spinfo)
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
    f_sparse = lambda data: coo_matmul(data, row, col, x, spinfo=spinfo)
    f_dense = lambda data: sparse_coo._coo_todense(data, row, col, spinfo=spinfo) @ x
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

  @jtu.sample_product(
    shape=[(4, 5), (3, 4), (5, 4)],
    dtype=_lowerings.SUPPORTED_DATA_DTYPES,
    transpose=[True, False],
  )
  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  def test_coo_spmv(self, shape, dtype, transpose):
    rng_sparse = rand_sparse(self.rng())
    rng_dense = jtu.rand_default(self.rng())

    mat = rng_sparse(shape, dtype)
    vec = rng_dense(shape[0] if transpose else shape[1], dtype)

    row, col = jnp.where(mat != 0)
    data = mat[row, col]

    expected = (mat.T if transpose else mat) @ vec
    actual = _lowerings.coo_spmv_p.bind(
        data, row.astype('int32'), col.astype('int32'), vec,
        transpose=transpose,
        shape=mat.shape)
    self.assertArraysAllClose(actual, expected)

  @jtu.sample_product(
    shape=[(4, 5), (3, 4), (5, 4)],
    dtype=_lowerings.SUPPORTED_DATA_DTYPES,
    transpose=[True, False],
  )
  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  def test_coo_spmm(self, shape, dtype, transpose):
    rng_sparse = rand_sparse(self.rng())
    rng_dense = jtu.rand_default(self.rng())

    mat = rng_sparse(shape, dtype)
    vec = rng_dense((shape[0] if transpose else shape[1], 3), dtype)

    row, col = jnp.where(mat != 0)
    data = mat[row, col]

    expected = (mat.T if transpose else mat) @ vec
    actual = _lowerings.coo_spmm_p.bind(
        data, row.astype('int32'), col.astype('int32'), vec,
        transpose=transpose,
        shape=mat.shape)
    self.assertArraysAllClose(actual, expected)

  @jtu.sample_product(
    shape=[(4, 5), (3, 4), (5, 4)],
    dtype=_lowerings.SUPPORTED_DATA_DTYPES,
    transpose=[True, False],
  )
  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  def test_csr_spmv(self, shape, dtype, transpose):
    rng_sparse = rand_sparse(self.rng())
    rng_dense = jtu.rand_default(self.rng())

    mat = rng_sparse(shape, dtype)
    data, indices, indptr = sparse_csr._csr_fromdense(mat, nse=(mat != 0).sum())
    vec = rng_dense(shape[0] if transpose else shape[1], dtype)

    expected = (mat.T if transpose else mat) @ vec
    actual = _lowerings.csr_spmv_p.bind(
        data, indices.astype('int32'), indptr.astype('int32'), vec,
        transpose=transpose,
        shape=mat.shape)
    self.assertArraysAllClose(actual, expected)

  @jtu.sample_product(
    shape=[(4, 5), (3, 4), (5, 4)],
    dtype=_lowerings.SUPPORTED_DATA_DTYPES,
    transpose=[True, False],
  )
  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  def test_csr_spmm(self, shape, dtype, transpose):
    rng_sparse = rand_sparse(self.rng())
    rng_dense = jtu.rand_default(self.rng())

    mat = rng_sparse(shape, dtype)
    data, indices, indptr = sparse_csr._csr_fromdense(mat, nse=(mat != 0).sum())
    vec = rng_dense((shape[0] if transpose else shape[1], 3), dtype)

    expected = (mat.T if transpose else mat) @ vec
    actual = _lowerings.csr_spmm_p.bind(
        data, indices.astype('int32'), indptr.astype('int32'), vec,
        transpose=transpose,
        shape=mat.shape)
    self.assertArraysAllClose(actual, expected)


@unittest.skipIf(platform.system() == "Windows", "Test crashes on Windows")
class BCOOTest(sptu.SparseTestCase):

  def gpu_matmul_warning_context(self, msg):
    if GPU_LOWERING_ENABLED and config.jax_bcoo_cusparse_lowering:
      return self.assertWarnsRegex(sparse.CuSparseEfficiencyWarning, msg)
    return contextlib.nullcontext()

  def test_repr(self):
    x = sparse.BCOO.fromdense(jnp.arange(5, dtype='float32'))
    self.assertEqual(repr(x), "BCOO(float32[5], nse=4)")

    y = sparse.BCOO.fromdense(jnp.arange(6, dtype='float32').reshape(2, 3), n_batch=1)
    self.assertEqual(repr(y), "BCOO(float32[2, 3], nse=3, n_batch=1)")

    y = sparse.BCOO.fromdense(jnp.arange(6, dtype='float32').reshape(2, 3), n_batch=1, n_dense=1)
    self.assertEqual(repr(y), "BCOO(float32[2, 3], nse=1, n_batch=1, n_dense=1)")

    M_invalid = sparse.BCOO.fromdense(jnp.arange(6, dtype='float32').reshape(2, 3))
    M_invalid.indices = jnp.array([])
    self.assertEqual(repr(M_invalid), "BCOO(<invalid>)")

    @jit
    def f(x):
      self.assertEqual(repr(x), "DynamicJaxprTracer[BCOO(float32[5], nse=4)]")
    f(x)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)],
    dtype=all_dtypes,
  )
  def test_empty(self, shape, dtype, n_batch, n_dense):
    M = sparse.empty(shape, dtype=dtype, n_batch=n_batch, n_dense=n_dense)
    self.assertIsInstance(M, sparse.BCOO)
    self.assertEqual(M.nse, 0)
    self.assertEqual(M.n_batch, n_batch)
    self.assertEqual(M.n_dense, n_dense)
    self.assertEqual(M.dtype, dtype)
    self.assertArraysEqual(M.todense(), jnp.empty(shape, dtype))

  @jtu.sample_product(
    [dict(n_batch=layout.n_batch, n_dense=layout.n_dense)
     for layout in iter_sparse_layouts((3, 3))],
    N=[3, 5],
    M=[None, 4],
    k=[-3, -1, 0, 2, 4],
    dtype=all_dtypes,
  )
  def test_eye(self, N, M, k, dtype, n_batch, n_dense):
    mat = sparse.eye(N, M, k, dtype=dtype, n_batch=n_batch, n_dense=n_dense)
    expected = jnp.eye(N, M, k, dtype=dtype)
    expected_nse = sparse.BCOO.fromdense(expected, n_batch=n_batch, n_dense=n_dense).nse

    self.assertIsInstance(mat, sparse.BCOO)
    self.assertEqual(mat.n_batch, n_batch)
    self.assertEqual(mat.n_dense, n_dense)
    self.assertEqual(mat.dtype, dtype)
    self.assertEqual(mat.nse, expected_nse)
    self.assertArraysEqual(mat.todense(), expected)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)],
    dtype=all_dtypes,
  )
  def test_bcoo_dense_round_trip(self, shape, dtype, n_batch, n_dense):
    n_sparse = len(shape) - n_batch - n_dense
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch, n_dense=n_dense)
    def round_trip(M):
      return sparse.BCOO.fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense).todense()
    args_maker = lambda: [M]
    ident = lambda x: x

    self._CheckAgainstNumpy(ident, round_trip, args_maker)
    self._CompileAndCheck(round_trip, args_maker)
    self._CheckBatchingSparse(ident, round_trip, args_maker, bdims=self._random_bdims(n_batch))
    if jnp.issubdtype(dtype, jnp.floating):
      # For n_sparse != 0, we can't use an identity because output zeros must not
      # be dependent on input zeros. This mimics the code in count_stored_elements().
      def expected(M):
        if n_sparse == 0: return M
        mask = (M != 0).any(range(M.ndim - n_dense, M.ndim), keepdims=True)
        return jnp.where(mask, M, 0)
      self._CheckGradsSparse(expected, round_trip, args_maker)

  def test_bcoo_fromdense_sorted_and_unique_indices(self):
    rng = self.rng()
    rng_sparse = rand_sparse(rng)
    mat = sparse.BCOO.fromdense(rng_sparse((5, 6), np.float32))
    perm = rng.permutation(mat.nse)
    mat_unsorted = sparse.BCOO((mat.data[perm], mat.indices[perm]),
                               shape=mat.shape,
                               unique_indices=mat.unique_indices)
    mat_resorted = mat_unsorted.sort_indices()
    with self.subTest('sorted indices'):
      self.assertArraysEqual(mat.indices, mat_resorted.indices)
      self.assertArraysEqual(mat.data, mat_resorted.data)

    with self.subTest('unique indices'):
      self.assertTrue(mat.unique_indices)
      self.assertTrue(mat_unsorted.unique_indices)
      self.assertTrue(mat_resorted.unique_indices)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
    assume_unique=[True, False, None]
  )
  def test_bcoo_extract(self, shape, dtype, n_batch, n_dense, assume_unique):
    rng = rand_sparse(self.rng())

    def args_maker():
      x = rng(shape, dtype)
      x_bcoo = sparse.bcoo_fromdense(x, n_batch=n_batch, n_dense=n_dense)
      # Unique indices are required for this test when assume_unique == True.
      self.assertTrue(x_bcoo.unique_indices)
      return x_bcoo, x

    dense_op = lambda _, x: x
    sparse_op = partial(sparse.bcoo_extract, assume_unique=assume_unique)

    self._CheckAgainstDense(dense_op, sparse_op, args_maker)
    self._CheckBatchingSparse(dense_op, sparse_op, args_maker, bdims=2 * self._random_bdims(n_batch))

  def test_bcoo_extract_duplicate_indices(self):
    data = jnp.array([1, 3, 9, 27, 81, 243])
    indices = jnp.array([[0], [5], [0], [3], [2], [3]])
    shape = (6,)
    mat = sparse.BCOO((data, indices), shape=shape).todense()

    data1 = sparse_bcoo._bcoo_extract(indices, mat, assume_unique=True)
    self.assertArraysEqual(data1, jnp.array([10, 3, 10, 270, 81, 270]))

    data2 = sparse_bcoo._bcoo_extract(indices, mat, assume_unique=False)
    self.assertArraysEqual(data2, jnp.array([10, 3, 0, 270, 81, 0]))

  def test_bcoo_extract_duplicate_indices_n_sparse_0(self):
    data = jnp.arange(6).reshape(3, 2)
    indices = jnp.empty((3, 2, 0), dtype=int)
    shape = (3,)
    mat = sparse.BCOO((data, indices), shape=shape).todense()

    data1 = sparse_bcoo._bcoo_extract(indices, mat, assume_unique=True)
    self.assertArraysEqual(data1, jnp.array([[1, 1], [5, 5], [9, 9]]))

    data2 = sparse_bcoo._bcoo_extract(indices, mat, assume_unique=False)
    self.assertArraysEqual(data2, jnp.array([[1, 0], [5, 0], [9, 0]]))

  def test_bcoo_extract_batching(self):
    # https://github.com/google/jax/issues/9431
    indices = jnp.zeros((4, 1, 1), dtype=int)
    mat = jnp.arange(4.).reshape((4, 1))

    # in_axes = (0, None)
    expected = jnp.vstack([sparse_bcoo._bcoo_extract(i, mat[0]) for i in indices])
    actual = vmap(sparse_bcoo._bcoo_extract, in_axes=(0, None))(indices, mat[0])
    self.assertArraysEqual(expected, actual)

    # in_axes = (None, 0)
    expected = jnp.vstack([sparse_bcoo._bcoo_extract(indices[0], m) for m in mat])
    actual = vmap(sparse_bcoo._bcoo_extract, in_axes=(None, 0))(indices[0], mat)
    self.assertArraysEqual(expected, actual)

    # in_axes = (0, 0)
    expected = jnp.vstack([sparse_bcoo._bcoo_extract(i, m) for i, m in zip(indices, mat)])
    actual = vmap(sparse_bcoo._bcoo_extract, in_axes=0)(indices, mat)
    self.assertArraysEqual(expected, actual)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)],
    dtype=jtu.dtypes.floating,
  )
  def test_bcoo_extract_ad(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)

    extract = partial(sparse_bcoo._bcoo_extract, indices)
    j1 = jax.jacfwd(extract)(M)
    j2 = jax.jacrev(extract)(M)
    hess = jax.hessian(extract)(M)
    self.assertArraysAllClose(j1, j2)
    self.assertEqual(j1.shape, data.shape + M.shape)
    self.assertEqual(hess.shape, data.shape + 2 * M.shape)

  def test_bcoo_extract_zero_nse(self):
    # Regression test for https://github.com/google/jax/issues/13653

    # (n_batch, n_sparse, n_dense) = (1, 0, 0), nse = 2
    args_maker = lambda: (jnp.zeros((3, 2, 0), dtype='int32'), jnp.arange(3))
    self._CompileAndCheck(sparse_bcoo._bcoo_extract, args_maker)

    # (n_batch, n_sparse, n_dense) = (0, 0, 1), nse = 2
    args_maker = lambda: (jnp.zeros((2, 0), dtype='int32'), jnp.arange(3))
    self._CompileAndCheck(sparse_bcoo._bcoo_extract, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)],
    dtype=jtu.dtypes.numeric,
  )
  def test_bcoo_transpose(self, shape, dtype, n_batch, n_dense):
    n_sparse = len(shape) - n_batch - n_dense
    rng = self.rng()
    sprng = sptu.rand_bcoo(rng, n_batch=n_batch, n_dense=n_dense)

    permutation = np.concatenate([
      rng.permutation(range(n_batch)),
      rng.permutation(range(n_batch, n_batch + n_sparse)),
      rng.permutation(range(n_batch + n_sparse, len(shape)))]).astype(int)

    args_maker = lambda: [sprng(shape, dtype)]
    dense_func = partial(lax.transpose, permutation=permutation)
    sparse_func = partial(sparse.bcoo_transpose, permutation=permutation)

    self._CheckAgainstDense(dense_func, sparse_func, args_maker)
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(dense_func, sparse_func, args_maker)
    self._CheckBatchingSparse(dense_func, sparse_func, args_maker, bdims=self._random_bdims(n_batch))

  def test_bcoo_transpose_indices_sorted(self):
    rng = self.rng()
    rng_sparse = rand_sparse(rng)
    n_batch, n_dense = 2, 2
    shape = (2, 3, 4, 5, 6, 7, 8)
    mat = sparse.BCOO.fromdense(rng_sparse(shape, np.float32),
                                n_dense=n_dense, n_batch=n_batch)

    permutations = (1, 0, 2, 3, 4, 6, 5)
    mat_T_indices_sorted = mat.transpose(axes=permutations)
    self.assertTrue(mat_T_indices_sorted.indices_sorted)

    permutations = (0, 1, 3, 2, 4, 5, 6)
    mat_T_indices_unsorted = mat.transpose(axes=permutations)
    self.assertFalse(mat_T_indices_unsorted.indices_sorted)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape, min_n_batch=1)
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcoo_todense_partial_batch(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)

    M1 = sparse_bcoo._bcoo_todense(data, indices[:1], spinfo=sparse_util.SparseInfo(M.shape))
    M2 = sparse_bcoo._bcoo_todense(data, jnp.stack(shape[0] * [indices[0]]), spinfo=sparse_util.SparseInfo(M.shape))
    self.assertAllClose(M1, M2)

    M3 = sparse_bcoo._bcoo_todense(data[:1], indices, spinfo=sparse_util.SparseInfo(M.shape))
    M4 = sparse_bcoo._bcoo_todense(jnp.stack(shape[0] * [data[0]]), indices, spinfo=sparse_util.SparseInfo(M.shape))
    self.assertAllClose(M3, M4)

  @jtu.sample_product(
    props=_generate_batched_dot_general_properties(),
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def test_bcoo_dot_general(self, dtype: np.dtype, props: BatchedDotGeneralProperties):
    rng = jtu.rand_default(self.rng())
    sprng = sptu.rand_bcoo(self.rng(), n_batch=props.n_batch, n_dense=props.n_dense)
    args_maker = lambda: [sprng(props.lhs_shape, dtype),
                          rng(props.rhs_shape, dtype)]
    dense_fun = partial(lax.dot_general, dimension_numbers=props.dimension_numbers)
    sparse_fun = partial(sparse.bcoo_dot_general, dimension_numbers=props.dimension_numbers)

    tol = {np.float64: 1E-12, np.complex128: 1E-12,
           np.float32: 1E-5, np.complex64: 1E-5}
    self._CheckAgainstDense(dense_fun, sparse_fun, args_maker, tol=tol)
    if jnp.issubdtype(dtype, jnp.floating) and props.n_dense == 0:
      # Dense dimensions not yet fully supported in reverse mode.
      modes = ['fwd'] if props.n_dense != 0 else ['fwd', 'rev']
      self._CheckGradsSparse(dense_fun, sparse_fun, args_maker, modes=modes, atol=tol, rtol=tol)
    self._CheckBatchingSparse(dense_fun, sparse_fun, args_maker, atol=tol, rtol=tol,
                              bdims=self._random_bdims(props.n_batch, len(props.rhs_shape)))

  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape,
          lhs_contracting=lhs_contracting, rhs_contracting=rhs_contracting)
      for lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
          [(5,), (5,), [0], [0]],
          [(5,), (5, 7), [0], [0]],
          [(5,), (7, 5), [0], [1]],
          [(5, 7), (5,), [0], [0]],
          [(7, 5), (5,), [1], [0]],
          [(3, 5), (2, 5), [1], [1]],
          [(3, 5), (5, 2), [1], [0]],
          [(5, 3), (2, 5), [0], [1]],
          [(5, 3), (5, 2), [0], [0]],
      ]
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def test_bcoo_dot_general_cusparse(
    self, lhs_shape, rhs_shape, dtype, lhs_contracting, rhs_contracting):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())
    def args_maker():
      lhs = rng_sparse(lhs_shape, dtype)
      rhs = rng(rhs_shape, dtype)
      nse = sparse.util._count_stored_elements(lhs, n_batch=0, n_dense=0)
      lhs_bcoo = sparse_bcoo.bcoo_fromdense(lhs, nse=nse, index_dtype=jnp.int32)
      return lhs_bcoo, lhs, rhs

    dimension_numbers = ((lhs_contracting, rhs_contracting), ([], []))

    def f_dense(lhs_bcoo, lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

    def f_sparse(lhs_bcoo, lhs, rhs):
      return sparse_bcoo.bcoo_dot_general(lhs_bcoo, rhs,
                                          dimension_numbers=dimension_numbers)

    self._CompileAndCheck(f_sparse, args_maker)
    self._CheckAgainstNumpy(f_dense, f_sparse, args_maker)

  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  @jtu.sample_product(
    [dict(n_batch=n_batch, lhs_shape=lhs_shape, rhs_shape=rhs_shape,
          lhs_contracting=lhs_contracting, rhs_contracting=rhs_contracting)
      for n_batch, lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
          [1, (1, 2, 3), (3, 2), [2], [0]],
          [1, (1, 3, 2), (3, 2), [1], [0]],
          [1, (1, 3, 2), (4, 3), [1], [1]],
          [1, (4, 2, 3), (3, 5), [2], [0]],
          [1, (4, 2, 3), (2, 5), [1], [0]],
          [1, (4, 2, 3), (5, 3), [2], [1]],
      ]
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def test_bcoo_batched_matmat_cusparse(
    self, n_batch, lhs_shape, rhs_shape, dtype, lhs_contracting,
    rhs_contracting):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())
    def args_maker():
      lhs = rng_sparse(lhs_shape, dtype)
      rhs = rng(rhs_shape, dtype)
      nse = sparse.util._count_stored_elements(lhs, n_batch=n_batch,
                                               n_dense=0)
      lhs_bcoo = sparse_bcoo.bcoo_fromdense(lhs, n_batch=n_batch, nse=nse,
                                            index_dtype=jnp.int32)
      return lhs_bcoo, lhs, rhs

    dimension_numbers = ((lhs_contracting, rhs_contracting), ([], []))

    def f_dense(lhs_bcoo, lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

    def f_sparse(lhs_bcoo, lhs, rhs):
      return sparse_bcoo.bcoo_dot_general(lhs_bcoo, rhs,
                                          dimension_numbers=dimension_numbers)

    # TODO(tianjianlu): In some cases, this fails python_should_be_executing.
    # self._CompileAndCheck(f_sparse, args_maker)
    self._CheckAgainstNumpy(f_dense, f_sparse, args_maker)

  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  @jtu.sample_product(
    [dict(n_batch=n_batch, lhs_shape=lhs_shape, rhs_shape=rhs_shape,
          lhs_contracting=lhs_contracting, rhs_contracting=rhs_contracting)
      for n_batch, lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
          [1, (1, 2, 3), (3), [2], [0]],
          [1, (1, 2), (3, 2), [1], [1]],
      ]
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcoo_batched_matmat_default_lowering(
    self, n_batch, lhs_shape, rhs_shape, dtype, lhs_contracting,
    rhs_contracting):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())
    lhs = rng_sparse(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    nse = sparse.util._count_stored_elements(lhs, n_batch=n_batch,
                                             n_dense=0)
    lhs_bcoo = sparse_bcoo.bcoo_fromdense(lhs, n_batch=n_batch, nse=nse,
                                            index_dtype=jnp.int32)
    dimension_numbers = ((lhs_contracting, rhs_contracting), ([], []))
    matmat_expected = lax.dot_general(lhs, rhs,
                                      dimension_numbers=dimension_numbers)
    sp_matmat = jit(partial(sparse_bcoo.bcoo_dot_general,
                            dimension_numbers=dimension_numbers))

    # TODO(jakevdp): uncomment once batching is supported again.
    # with self.gpu_matmul_warning_context(
    #     "bcoo_dot_general GPU lowering currently does not support this batch-mode computation.*"):
    matmat_default_lowering_fallback = sp_matmat(lhs_bcoo, rhs)
    self.assertArraysEqual(matmat_expected, matmat_default_lowering_fallback)

  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  def test_bcoo_dot_general_oob_and_unsorted_indices_cusparse(self):
    """Tests bcoo dot general with out-of-bound and unsorted indices."""

    rhs = jnp.ones((5, 3), dtype=jnp.float32)

    # It creates out-of-bound indices when nse > nnz.
    lhs_mat_dense = jnp.array([[1, 0, 2, 3, 0], [0, 0, 0, 4, 0]],
                              dtype=jnp.float32)
    lhs_mat_bcoo = sparse.BCOO.fromdense(lhs_mat_dense, nse=7)
    rng = self.rng()
    perm = rng.permutation(lhs_mat_bcoo.nse)
    lhs_mat_bcoo_unsorted = sparse.BCOO(
        (lhs_mat_bcoo.data[perm], lhs_mat_bcoo.indices[perm]),
        shape=lhs_mat_dense.shape)

    dimension_numbers_2d = (([1], [0]), ([], []))
    sp_matmat = jit(partial(sparse_bcoo.bcoo_dot_general,
                            dimension_numbers=dimension_numbers_2d))

    matmat_expected = lax.dot_general(lhs_mat_dense, rhs,
                                      dimension_numbers=dimension_numbers_2d)
    with self.subTest(msg="2D"):
      with self.gpu_matmul_warning_context(
          "bcoo_dot_general GPU lowering requires matrices with sorted indices*"):
        matmat_unsorted_fallback = sp_matmat(lhs_mat_bcoo_unsorted, rhs)
      self.assertArraysEqual(matmat_expected, matmat_unsorted_fallback)

    lhs_vec_dense = jnp.array([0, 1, 0, 2, 0], dtype=jnp.float32)
    lhs_vec_bcoo = sparse.BCOO.fromdense(lhs_vec_dense, nse=5)
    rng = self.rng()
    perm = rng.permutation(lhs_vec_bcoo.nse)
    lhs_vec_bcoo_unsorted = sparse.BCOO(
        (lhs_vec_bcoo.data[perm], lhs_vec_bcoo.indices[perm]),
        shape=lhs_vec_dense.shape, indices_sorted=False)

    dimension_numbers_1d = (([0], [0]), ([], []))
    sp_vecmat = jit(partial(sparse_bcoo.bcoo_dot_general,
                            dimension_numbers=dimension_numbers_1d))

    vecmat_expected = lax.dot_general(lhs_vec_dense, rhs,
                                      dimension_numbers=dimension_numbers_1d)

    with self.subTest(msg="1D"):
      with self.gpu_matmul_warning_context(
          "bcoo_dot_general GPU lowering requires matrices with sorted indices*"):
        vecmat_unsorted_fallback = sp_vecmat(lhs_vec_bcoo_unsorted, rhs)
      self.assertArraysEqual(vecmat_expected, vecmat_unsorted_fallback)

  @jtu.sample_product(
    props=_generate_batched_dot_general_properties(),
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def test_bcoo_rdot_general(self, dtype: np.dtype, props: BatchedDotGeneralProperties):
    rng = jtu.rand_default(self.rng())
    sprng = sptu.rand_bcoo(self.rng(), n_batch=props.n_batch, n_dense=props.n_dense)
    args_maker = lambda: [rng(props.rhs_shape, dtype),
                          sprng(props.lhs_shape, dtype)]
    dimension_numbers = tuple(d[::-1] for d in props.dimension_numbers)
    sparse_fun = partial(sparse.bcoo_dot_general, dimension_numbers=dimension_numbers)
    dense_fun = partial(lax.dot_general, dimension_numbers=dimension_numbers)

    tol = {np.float64: 1E-12, np.complex128: 1E-12,
           np.float32: 1E-5, np.complex64: 1E-5}
    self._CheckAgainstDense(dense_fun, sparse_fun, args_maker, tol=tol)
    if jnp.issubdtype(dtype, jnp.floating):
      # Dense dimensions not yet fully supported in reverse mode.
      modes = ['fwd'] if props.n_dense != 0 else ['fwd', 'rev']
      self._CheckGradsSparse(dense_fun, sparse_fun, args_maker, modes=modes, atol=tol, rtol=tol)

  @jtu.sample_product(
    [dict(n_batch=n_batch, n_dense=n_dense, lhs_shape=lhs_shape,
          rhs_shape=rhs_shape, dimension_numbers=dimension_numbers)
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 2, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 1),
      ]
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def test_bcoo_dot_general_partial_batch(self, lhs_shape, rhs_shape, dtype,
                                          dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())

    X = rng_sparse(lhs_shape, dtype)
    nse = sparse.util._count_stored_elements(X, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(X, nse=nse, n_batch=n_batch, n_dense=n_dense)
    Y = rng(rhs_shape, dtype)

    def f_dense(X, Y):
      return lax.dot_general(X, Y, dimension_numbers=dimension_numbers)

    def f_sparse(data, indices, Y):
      return sparse_bcoo._bcoo_dot_general(data, indices, Y, lhs_spinfo=sparse_util.SparseInfo(X.shape),
                                           dimension_numbers=dimension_numbers, preferred_element_type=None)

    for data, indices in itertools.product([data, data[:1]], [indices, indices[:1]]):
      X = sparse_bcoo._bcoo_todense(data, indices, spinfo=sparse_util.SparseInfo(X.shape))
      self.assertAllClose(f_dense(X, Y), f_sparse(data, indices, Y))

  @jtu.sample_product(
    props=_generate_bcoo_dot_general_sampled_properties(),
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_bcoo_dot_general_sampled(self, props, dtype):
    rng = jtu.rand_default(self.rng())
    sprng = sptu.rand_bcoo(self.rng(), n_batch=props.n_batch, n_dense=props.n_dense)
    out = jax.eval_shape(partial(lax.dot_general, dimension_numbers=props.dimension_numbers),
                         jax.ShapeDtypeStruct(props.lhs_shape, dtype),
                         jax.ShapeDtypeStruct(props.rhs_shape, dtype))
    args_maker = lambda: [rng(props.lhs_shape, dtype), rng(props.rhs_shape, dtype),
                          sprng(out.shape, dtype).indices]

    def dense_fun(lhs, rhs, indices):
      AB = lax.dot_general(lhs, rhs, dimension_numbers=props.dimension_numbers)
      return sparse_bcoo._bcoo_extract(indices, AB)
    def sparse_fun(lhs, rhs, indices):
      return sparse.bcoo_dot_general_sampled(
          lhs, rhs, indices, dimension_numbers=props.dimension_numbers)

    self._CheckAgainstDense(dense_fun, sparse_fun, args_maker)
    if jnp.issubdtype(dtype, jnp.floating):
      # Note: forward mode fails for some sparse layouts.
      # TODO(jakevdp) fix forward-mode autodiff & enable tests here.
      self._CheckGradsSparse(dense_fun, sparse_fun, args_maker, modes=['rev'], argnums=[0, 1])

  @jtu.sample_product(
    [{'xshape': xshape, 'yshape': yshape, 'lhs_contract': lhs_contract, 'rhs_contract': rhs_contract}
     for (xshape, yshape, lhs_contract, rhs_contract) in [
      [(4, 3), (4, 5), (0,), (0,)],
      [(3, 4), (4, 5), (1,), (0,)],
      [(4, 3), (5, 4), (0,), (1,)],
      [(3, 4), (5, 4), (1,), (1,)],
      [(3,), (3,), (), ()],
      [(3,), (5,), (), ()],
      [(5,), (3,), (), ()],
      [(5,), (5,), (), ()],
     ]],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
    n_batch=[0, 1, 2],
  )
  @jax.default_matmul_precision("float32")
  def test_bcoo_dot_general_sampled_fast_cases(
      self, xshape, yshape, lhs_contract, rhs_contract, n_batch, dtype):
    rng = jtu.rand_default(self.rng())
    sprng = sptu.rand_bcoo(self.rng(), n_batch=n_batch)
    dimension_numbers = ((lhs_contract, rhs_contract), ([], []))

    out_shape = jax.eval_shape(partial(lax.dot_general, dimension_numbers=dimension_numbers),
                               jax.ShapeDtypeStruct(xshape, dtype), jax.ShapeDtypeStruct(yshape, dtype))

    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype),
                          sprng(out_shape.shape, out_shape.dtype).indices]

    def f1(x, y, indices):
      mat_full = lax.dot_general(x, y, dimension_numbers=dimension_numbers)
      return sparse_bcoo._bcoo_extract(indices, mat_full)

    def f2(x, y, indices):
      return sparse.bcoo_dot_general_sampled(x, y, indices, dimension_numbers=dimension_numbers)

    self._CheckAgainstNumpy(f1, f2, args_maker, tol=MATMUL_TOL)
    self._CompileAndCheck(f2, args_maker, tol=MATMUL_TOL)

  @jtu.sample_product(
    [dict(n_batch=n_batch, n_dense=n_dense, lhs_shape=lhs_shape,
          rhs_shape=rhs_shape, dimension_numbers=dimension_numbers)
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
    ],
    dtype=jtu.dtypes.floating,
  )
  @jax.default_matmul_precision("float32")
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
      return sparse_bcoo._bcoo_extract(indices, AB)
    def sparse_fun(lhs, rhs, indices):
      return sparse.bcoo_dot_general_sampled(
                lhs, rhs, indices, dimension_numbers=dimension_numbers)

    jf_dense = jax.jacfwd(dense_fun)(lhs, rhs, indices)
    jf_sparse = jax.jacfwd(sparse_fun)(lhs, rhs, indices)
    jr_dense = jax.jacrev(dense_fun)(lhs, rhs, indices)
    jr_sparse = jax.jacrev(sparse_fun)(lhs, rhs, indices)

    self.assertAllClose(jf_sparse, jf_dense)
    self.assertAllClose(jr_sparse, jr_dense)
    self.assertAllClose(jf_sparse, jr_sparse)

  @jtu.sample_product(
    [dict(lhs_n_batch=lhs_n_batch, rhs_n_batch=rhs_n_batch, lhs_shape=lhs_shape,
          rhs_shape=rhs_shape, dimension_numbers=dimension_numbers)
      for lhs_shape, lhs_n_batch, rhs_shape, rhs_n_batch, dimension_numbers in [
          # (batched) outer products (no contraction)
          ((5,), 0, (6,), 0, (([], []), ([], []))),
          ((3, 5), 0, (2, 4), 0, (([], []), ([], []))),
          ((3, 5), 1, (3, 4), 1, (([], []), ([0], [0]))),
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
          # (batched) matrix-matrix products
          ((5, 7), 0, (7, 3), 0, (([1], [0]), ([], []))),
          ((2, 3, 4), 1, (4, 3), 0, (([2], [0]), ([], []))),
          ((2, 3, 4), 1, (2, 4, 3), 1, (([2], [1]), ([0], [0]))),
          # more general operations
          ((2, 3, 4, 3), 1, (2, 4, 3, 4), 1, (([2, 3], [1, 2]), ([0], [0]))),
          ((2, 3, 4, 3, 1), 2, (3, 2, 3, 4), 2, (([2, 3], [3, 2]), ([0, 1], [1, 0]))),
      ]
    ],
    swap=[True, False],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_bcoo_spdot_general(self, lhs_shape, lhs_n_batch, rhs_shape, rhs_n_batch, dtype, swap, dimension_numbers):
    if swap:
      dimension_numbers = tuple(d[::-1] for d in dimension_numbers)
      lhs_shape, rhs_shape = rhs_shape, lhs_shape
      lhs_n_batch, rhs_n_batch = rhs_n_batch, lhs_n_batch

    lhs_n_sparse = len(lhs_shape) - lhs_n_batch
    rhs_batch = dimension_numbers[1][1]
    lhs_contracting = dimension_numbers[0][0]
    should_error = (rhs_n_batch > len(rhs_batch) and lhs_n_sparse > len(lhs_contracting))

    sprng = sptu.rand_bcoo(self.rng())
    args_maker = lambda: [sprng(lhs_shape, dtype, n_batch=lhs_n_batch),
                          sprng(rhs_shape, dtype, n_batch=rhs_n_batch)]

    def f_dense(x, y):
      return lax.dot_general(x, y, dimension_numbers=dimension_numbers)

    def f_sparse(xsp, ysp):
      return sparse.bcoo_dot_general(xsp, ysp, dimension_numbers=dimension_numbers)

    if should_error:
      with self.assertRaisesRegex(ValueError, ".*cannot have unused batch dims on rhs with unused sparse dims on lhs."):
        f_sparse(*args_maker())
    else:
      tol = {"float32": 1E-5, "complex64": 1E-5, "float64": 1E-14, "complex128": 1E-14}
      self._CheckAgainstDense(f_dense, f_sparse, args_maker, tol=tol)
      self._CheckBatchingSparse(f_dense, f_sparse, args_maker, tol=tol)
      if jnp.issubdtype(dtype, jnp.floating):
        self._CheckGradsSparse(f_dense, f_sparse, args_maker, modes=['fwd'])

  @jtu.sample_product(
    lhs_shape=[(5,), (4, 5)],
    rhs_shape=[(5,), (5, 4)])
  @jax.default_matmul_precision("float32")
  def test_bcoo_spdot_general_nse(self, lhs_shape, rhs_shape):
    rng = sptu.rand_bcoo(self.rng())
    dtype = jnp.float32
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    out = lhs @ rhs

    expected_out = lhs.todense() @ rhs.todense()
    expected_nse = min(lhs.nse * rhs.nse, out.size)

    self.assertArraysAllClose(out.todense(), expected_out)
    self.assertEqual(out.nse, expected_nse)

  def test_bcoo_spdot_general_ad_bug(self):
    # Regression test for https://github.com/google/jax/issues/10163
    A_indices = jnp.array([[0, 1], [0, 2], [1, 1], [1, 2], [1, 0]])
    A_values = jnp.array([-2.0, 1.0, -1.0, 0.5, 2.0])
    A_shape = (2, 3)

    B_indices = jnp.array([[0, 2], [2, 1], [0, 3], [1, 3], [1, 0], [0, 0]])
    B_values = jnp.array([10.0, 100.0, 1000.0, -5.0, -50.0, -500.0])
    B_shape = (3, 4)

    def sp_sp_product(v1, v2):
        A = sparse.BCOO((v1, A_indices), shape=A_shape)
        B = sparse.BCOO((v2, B_indices), shape=B_shape)
        return (A @ B).todense()

    def sp_de_product(v1, v2):
        A = sparse.BCOO((v1, A_indices), shape=A_shape)
        B = sparse.BCOO((v2, B_indices), shape=B_shape).todense()
        return A @ B

    def de_de_product(v1, v2):
        sparse1 = sparse.BCOO((v1, A_indices), shape=A_shape).todense()
        dense2 = sparse.BCOO((v2, B_indices), shape=B_shape).todense()
        return sparse1 @ dense2

    sp_sp_jac = jax.jacfwd(sp_sp_product, argnums=1)(A_values, B_values)
    sp_de_jac = jax.jacfwd(sp_de_product, argnums=1)(A_values, B_values)
    de_de_jac = jax.jacfwd(de_de_product, argnums=1)(A_values, B_values)

    self.assertAllClose(sp_sp_jac, de_de_jac)
    self.assertAllClose(sp_de_jac, de_de_jac)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(), (5,), (5, 8), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)],
    dtype=jtu.dtypes.numeric,
  )
  def test_bcoo_slice(self, shape, dtype, n_batch, n_dense):
    rng = self.rng()
    sprng = sptu.rand_bcoo(rng, n_batch=n_batch, n_dense=n_dense)
    args_maker = lambda: [sprng(shape, dtype)]

    slices = rng.randint(0, np.array(shape) + 1, (2, len(shape))).T
    slices.sort(1)
    start_indices, limit_indices = unzip2(slices)
    strides = list(rng.randint(1, 4, len(shape)))
    kwds = dict(start_indices=start_indices, limit_indices=limit_indices, strides=strides)

    dense_func = partial(lax.slice, **kwds)
    sparse_func = partial(sparse.bcoo_slice, **kwds)

    self._CheckAgainstDense(dense_func, sparse_func, args_maker)
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(dense_func, sparse_func, args_maker)

    mat, = args_maker()
    out = sparse_func(mat)

    # Array layout is the same
    self.assertEqual(mat.n_batch, out.n_batch)
    self.assertEqual(mat.n_sparse, out.n_sparse)
    self.assertEqual(mat.n_dense, out.n_dense)

    # Unnecessary padding eliminated
    max_nse = math.prod(out.shape[out.n_batch: out.n_batch + out.n_sparse])
    self.assertLessEqual(out.nse, max_nse)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(), (5,), (5, 8), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)],
    dtype=jtu.dtypes.numeric,
  )
  def test_bcoo_dynamic_slice(self, shape, dtype, n_batch, n_dense):
    rng = self.rng()
    sprng = sptu.rand_bcoo(rng, n_batch=n_batch, n_dense=n_dense)
    args_maker = lambda: [sprng(shape, dtype)]

    rng = self.rng()
    # Note: test out-of-range start indices
    start_indices = rng.randint(-max(shape, default=0), max(shape, default=0), len(shape))
    slice_sizes = rng.randint(0, shape, len(shape))
    kwds = dict(start_indices=start_indices, slice_sizes=slice_sizes)
    dense_func = partial(lax.dynamic_slice, **kwds)
    sparse_func = partial(sparse.bcoo_dynamic_slice, **kwds)

    self._CheckAgainstDense(dense_func, sparse_func, args_maker)
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(dense_func, sparse_func, args_maker)

    mat, = args_maker()
    out = sparse_func(mat)

    # Array layout is the same
    self.assertEqual(mat.n_batch, out.n_batch)
    self.assertEqual(mat.n_sparse, out.n_sparse)
    self.assertEqual(mat.n_dense, out.n_dense)

    # Unnecessary padding eliminated
    max_nse = math.prod(out.shape[out.n_batch: out.n_batch + out.n_sparse])
    self.assertLessEqual(out.nse, max_nse)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=n_batch, n_dense=n_dense, idx=idx)
      for shape, idx in [
        [(5,), np.index_exp[:]],
        [(5,), np.index_exp[4]],
        [(5,), np.index_exp[::2]],
        [(5,), np.index_exp[1::2]],
        [(5,), 1],
        [(3, 4), np.index_exp[1]],
        [(3, 4), np.index_exp[1, 2]],
        [(3, 4), np.index_exp[np.array([1, 2])]],
        [(3, 4), np.index_exp[np.array([[1], [2]]), 0]],
        [(3, 4), np.index_exp[np.array([[1], [2]]), 1:]],
        [(3, 4), np.index_exp[np.array([True, False, True])]],
        [(3, 4), np.index_exp[:2, np.array([True, False, True, False])]],
        [(3, 4), np.index_exp[None, 0, np.array([[2]])]],
        [(3, 4, 5), np.index_exp[2]],
        [(3, 4, 5), np.index_exp[:, 2]]
      ]
      for n_batch in range(len(shape) + 1)
      for n_dense in [0]  # TODO(jakevdp): add tests with n_dense
    ],
    dtype=jtu.dtypes.numeric,
  )
  def test_bcoo_getitem(self, shape, dtype, n_batch, n_dense, idx):
    sprng = sptu.rand_bcoo(self.rng(), n_batch=n_batch, n_dense=n_dense)
    args_maker = lambda: [sprng(shape, dtype)]

    fun = lambda x: x[idx]

    self._CheckAgainstDense(fun, fun, args_maker)
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(fun, fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=n_batch, n_dense=n_dense)
     for shape in [(2,), (3, 4), (5, 6, 2)]
     for n_batch in range(len(shape) + 1)
     for n_dense in [0]  # TODO(jakevdp): add tests with n_dense
    ],
    dtype=jtu.dtypes.numeric,
  )
  def test_bcoo_iter(self, shape, dtype, n_batch, n_dense):
    sprng = rand_sparse(self.rng())
    args_maker = lambda: [sprng(shape, dtype)]

    self._CheckAgainstDense(list, list, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense, nse=nse)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)
      for nse in [None, math.prod(shape) - 1]
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
    remove_zeros=[True, False],
  )
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_bcoo_sum_duplicates(self, shape, dtype, n_batch, n_dense, nse, remove_zeros):
    sprng = sptu.rand_bcoo(self.rng(), n_batch=n_batch, n_dense=n_dense)

    def args_maker():
      # Create a matrix with duplicate indices
      M = sprng(shape, dtype)
      new_indices = jnp.concatenate([M.indices, M.indices], axis=n_batch)
      new_data = jnp.concatenate([M.data, M.data], axis=n_batch)
      return [sparse.BCOO((new_data, new_indices), shape=M.shape)]

    dense_fun = lambda x: x
    def sparse_fun(x):
      out = x.sum_duplicates(nse=nse, remove_zeros=remove_zeros)
      self.assertTrue(out.unique_indices)
      if nse:
        self.assertEqual(out.nse, nse)
      return out
    self._CheckAgainstDense(dense_fun, sparse_fun, args_maker, check_jit=(nse is not None))
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(dense_fun, sparse_fun, args_maker)
    if nse is not None:
      self._CheckBatchingSparse(dense_fun, sparse_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcoo_sort_indices(self, shape, dtype, n_batch, n_dense):
    rng_sparse = rand_sparse(self.rng(), rand_method=jtu.rand_some_zero)
    M = sparse.BCOO.fromdense(rng_sparse(shape, dtype), n_batch=n_batch, n_dense=n_dense)
    M.indices = M.indices[..., ::-1, :]
    M.indices_sorted = False

    M_sorted = M.sort_indices()
    self.assertArraysEqual(M.todense(), M_sorted.todense())
    self.assertEqual(M.unique_indices, M_sorted.unique_indices)
    self.assertEqual(True, M_sorted.indices_sorted)

    indices = M_sorted.indices
    if indices.size > 0:
      flatind = indices.reshape(-1, *indices.shape[-2:]).transpose(0, 2, 1)
      sorted = jax.vmap(jnp.lexsort)(flatind[:, ::-1])
      self.assertArraysEqual(sorted, lax.broadcasted_iota(sorted.dtype, sorted.shape, sorted.ndim - 1))

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape, min_n_batch=1)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcoo_sort_indices_batching(self, shape, dtype, n_batch, n_dense):
    rng_sparse = rand_sparse(self.rng(), rand_method=jtu.rand_some_zero)
    M = sparse.BCOO.fromdense(rng_sparse(shape, dtype), n_batch=n_batch, n_dense=n_dense)
    M.indices = M.indices[..., ::-1, :]
    M.indices_sorted = False

    identity = lambda M: M
    sort_ind = lambda M: M.sort_indices()
    for b in range(n_batch):
      identity = jax.vmap(identity, in_axes=b)
      sort_ind = jax.vmap(sort_ind, in_axes=b)
    M_sorted = sort_ind(M)
    M_expected = identity(M)
    self.assertArraysEqual(M_expected.todense(), M_sorted.todense())
    self.assertEqual(M.unique_indices, M_sorted.unique_indices)
    self.assertEqual(True, M_sorted.indices_sorted)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)],
    dtype=jtu.dtypes.floating,
  )
  def test_bcoo_sort_indices_ad(self, shape, dtype, n_batch, n_dense):
    rng_sparse = rand_sparse(self.rng(), rand_method=jtu.rand_some_zero)
    M = sparse.BCOO.fromdense(rng_sparse(shape, dtype), n_batch=n_batch, n_dense=n_dense)
    M.indices = M.indices[..., ::-1, :]

    def sort_indices(data):
      return sparse.BCOO((data, M.indices), shape=M.shape).sort_indices().data

    data_dot_fwd = jax.jacfwd(sort_indices)(M.data)
    data_dot_rev = jax.jacrev(sort_indices)(M.data)

    self.assertAllClose(data_dot_fwd, data_dot_rev)

  def test_bcoo_sort_indices_broadcasted(self):
    rng_index = jtu.rand_int(self.rng(), low=0, high=10)
    rng_data = jtu.rand_default(self.rng())

    # Construct matrix with three broadcasted batch dimensions.
    indices = rng_index((1, 3, 1, 10, 2), dtype='int32')
    data = rng_data((1, 1, 4, 10, 3), dtype='int32')
    shape = (2, 3, 4, 5, 4, 3)
    mat = sparse.BCOO((data, indices), shape=shape)

    indices_shape_out = indices.shape
    data_shape_out = (*map(max, indices.shape[:3], data.shape[:3]), *data.shape[3:])

    mat_sorted = sparse.bcoo_sort_indices(mat)
    assert mat_sorted.indices.shape == indices_shape_out
    assert mat_sorted.data.shape == data_shape_out
    self.assertArraysEqual(mat.todense(), mat_sorted.todense())

    mat_sorted_jit = jit(sparse.bcoo_sort_indices)(mat)
    assert mat_sorted_jit.indices.shape == indices_shape_out
    assert mat_sorted_jit.data.shape == data_shape_out
    self.assertArraysEqual(mat.todense(), mat_sorted_jit.todense())

  def test_bcoo_sum_duplicates_inferred_nse(self):
    x = sparse.BCOO.fromdense(jnp.diag(jnp.arange(4)))
    self.assertEqual(x.nse, 3)
    y = x + x.T
    self.assertEqual(y.nse, 6)
    y2 = y.sum_duplicates()
    self.assertEqual(y2.nse, 3)
    self.assertArraysEqual(y.todense(), y2.todense())

  def test_bcoo_sum_duplicates_remove_zeros(self):
    data = jnp.array([0, 1, 0, 0])
    indices = jnp.array([[0], [1], [2], [3]])
    x = sparse.BCOO((data, indices), shape=(4,))
    self.assertEqual(x.nse, 4)

    y1 = x.sum_duplicates(remove_zeros=True)
    self.assertArraysEqual(x.todense(), y1.todense())
    self.assertEqual(y1.nse, 1)

    y2 = x.sum_duplicates(remove_zeros=False)
    self.assertArraysEqual(x.todense(), y2.todense())
    self.assertEqual(y2.nse, x.nse)

  def test_bcoo_sum_duplicates_padding(self):
    # Regression test for https://github.com/google/jax/issues/8163
    size = 3
    data = jnp.array([1, 0, 0])
    indices = jnp.array([1, size, size])[:, None]
    x = sparse.BCOO((data, indices), shape=(3,))
    y = x.sum_duplicates(nse=x.nse)
    self.assertArraysEqual(x.todense(), y.todense())
    self.assertArraysEqual(x.indices, y.indices)
    self.assertArraysEqual(x.data, y.data)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense, axes=axes)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)
      for naxes in range(len(shape))
      for axes in itertools.combinations(range(len(shape)), naxes)
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcoo_reduce_sum(self, shape, dtype, n_batch, n_dense, axes):
    sprng = sptu.rand_bcoo(self.rng(), n_batch=n_batch, n_dense=n_dense)
    args_maker = lambda: [sprng(shape, dtype)]
    sparse_fun = partial(sparse.bcoo_reduce_sum, axes=axes)
    dense_fun = partial(lambda x: x.sum(axes))

    tol = {np.float64: 1E-14}
    self._CheckAgainstDense(dense_fun, sparse_fun, args_maker, tol=tol)
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(dense_fun, sparse_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, dimensions=dimensions, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape, dimensions in [
          [(1,), (0,)],
          [(1,), (-1,)],
          [(2, 1, 4), (1,)],
          [(2, 1, 3, 1), (1,)],
          [(2, 1, 3, 1), (1, 3)],
          [(2, 1, 3, 1), (3,)],
      ]
      for layout in iter_sparse_layouts(shape)],
    dtype=jtu.dtypes.numeric,
  )
  def test_bcoo_squeeze(self, shape, dtype, dimensions, n_batch, n_dense):
    sprng = sptu.rand_bcoo(self.rng(), n_batch=n_batch, n_dense=n_dense)
    args_maker = lambda: [sprng(shape, dtype)]
    dense_func = partial(lax.squeeze, dimensions=dimensions)
    sparse_func = partial(sparse.bcoo_squeeze, dimensions=dimensions)

    self._CheckAgainstDense(dense_func, sparse_func, args_maker)
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(dense_func, sparse_func, args_maker)

  @jtu.sample_product(
    [dict(batch_shapes=shapes, batch_perm=perm)
     for shapes in COMPATIBLE_SHAPE_PAIRS
     for perm in itertools.permutations(range(len(shapes[0])))],
    [dict(sparse_shapes=shapes, sparse_perm=perm)
     for shapes in COMPATIBLE_SHAPE_PAIRS
     for perm in itertools.permutations(range(len(shapes[0])))],
    [dict(dense_shapes=shapes, dense_perm=perm)
     for shapes in [[(),()]]  # TODO(jakevdp) add support for dense shapes
     for perm in itertools.permutations(range(len(shapes[0])))],
    dtype=jtu.dtypes.numeric
  )
  def test_bcoo_reshape(self, batch_shapes, sparse_shapes, dense_shapes,
                        batch_perm, sparse_perm, dense_perm, dtype):
    # Sparse reshapes cannot mix between sparse, dense, and batch dimensions.
    shape = (*batch_shapes[0], *sparse_shapes[0], *dense_shapes[0])
    new_sizes = (*batch_shapes[1], *sparse_shapes[1], *dense_shapes[1])
    n_batch = len(batch_shapes[0])
    n_sparse = len(sparse_shapes[0])
    n_dense = len(dense_shapes[0])
    dimensions = (
      *batch_perm,
      *(dim + n_batch for dim in sparse_perm),
      *(dim + n_batch + n_sparse for dim in dense_perm)
    )

    rng = sptu.rand_bcoo(self.rng(), n_batch=n_batch, n_dense=n_dense)
    args_maker = lambda: [rng(shape, dtype)]

    sparse_func = partial(sparse.bcoo_reshape, new_sizes=new_sizes, dimensions=dimensions)
    dense_func = partial(lax.reshape, new_sizes=new_sizes, dimensions=dimensions)

    self._CheckAgainstDense(dense_func, sparse_func, args_maker)
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(dense_func, sparse_func, args_maker)

  def test_bcoo_reshape_error(self):
    x = sparse.BCOO.fromdense(jnp.ones((2, 2, 3)), n_batch=1)
    with self.assertRaisesRegex(ValueError, ".*cannot mix batch and sparse dimensions.*"):
      x.reshape(3, 2, 2)
    y = sparse.BCOO((x.data[:1], x.indices), shape=x.shape)
    with self.assertRaisesRegex(NotImplementedError, "reshape of arrays with broadcasted batch dimensions."):
      y.reshape(2, 3, 2)

  @jtu.sample_product(
    [dict(shape=shape, dimensions=dimensions, n_batch=layout.n_batch, n_dense=layout.n_dense)
     for shape in [(3,), (3, 4), (3, 4, 5)]
     for dimensions in _iter_subsets(range(len(shape)))
     for layout in iter_sparse_layouts(shape)],
    dtype=jtu.dtypes.numeric,
  )
  def test_bcoo_rev(self, shape, dtype, n_batch, n_dense, dimensions):
    sprng = sptu.rand_bcoo(self.rng(), n_batch=n_batch, n_dense=n_dense)
    args_maker = lambda: [sprng(shape, dtype)]
    dense_func = partial(lax.rev, dimensions=dimensions)
    sparse_func = partial(sparse.bcoo_rev, dimensions=dimensions)

    self._CheckAgainstDense(dense_func, sparse_func, args_maker)
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(dense_func, sparse_func, args_maker)

  def test_bcsr_matmul_with_out_of_bounds_data(self):
    # Simple regression test of a failure mode for cuSparse.
    data = jnp.array([1, 2, 3, 4], dtype='float32')
    indices = jnp.array([0, 1, 2, 3])
    indptr = jnp.array([0, 1, 3, 3])
    M = sparse.BCSR((data, indices, indptr), shape=(3, 4))
    x = jnp.array([1, 2, 3, 4], dtype='float32')

    sparse_result = jax.jit(operator.matmul)(M, x)
    dense_result = jax.jit(operator.matmul)(M.todense(), x)
    self.assertAllClose(sparse_result, dense_result)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [[(3, 4), (4,)],
                                   [(3, 4), (4, 5)],
                                   [(3, 4), (2, 4, 5)]]
    ],
    lhs_dtype=all_dtypes,
    rhs_dtype=all_dtypes,
  )
  @jax.default_matmul_precision("float32")
  @jtu.ignore_warning(category=sparse.CuSparseEfficiencyWarning)
  def test_bcsr_matmul(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    # Note: currently, batch dimensions in matmul must correspond to batch
    # dimensions in the sparse representation.
    n_batch_lhs = max(0, len(lhs_shape) - 2)

    rng = jtu.rand_default(self.rng())
    sprng = sptu.rand_bcsr(self.rng())
    args_maker = lambda: [sprng(lhs_shape, lhs_dtype, n_batch=n_batch_lhs),
                          jnp.array(rng(rhs_shape, rhs_dtype))]

    tol = {np.float64: 1E-13, np.complex128: 1E-13,
           np.float32: 2E-6, np.complex64: 2E-6}

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstDense(operator.matmul, operator.matmul, args_maker, tol=tol)


  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [[(3,), (3,)],
                                   [(3, 4), (4,)],
                                   [(4,), (4, 5)],
                                   [(3, 4), (4, 5)],
                                   [(3, 4), (2, 4, 5)],
                                   [(2, 3, 4), (4, 5)],
                                   [(2, 3, 4), (2, 4, 5)]]
    ],
    lhs_dtype=all_dtypes,
    rhs_dtype=all_dtypes,
  )
  @jax.default_matmul_precision("float32")
  @jtu.ignore_warning(category=sparse.CuSparseEfficiencyWarning)
  def test_bcoo_matmul(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    if (jtu.device_under_test() == "gpu" and
        _is_required_cuda_version_satisfied(12000)):
      raise unittest.SkipTest("Triggers a bug in cuda-12 b/287344632")

    # Note: currently, batch dimensions in matmul must correspond to batch
    # dimensions in the sparse representation.
    n_batch_lhs = max(0, len(lhs_shape) - 2)
    n_batch_rhs = max(0, len(rhs_shape) - 2)

    rng = jtu.rand_default(self.rng())
    sprng = sptu.rand_bcoo(self.rng())
    args_maker_de_sp = lambda: [jnp.array(rng(lhs_shape, lhs_dtype)),
                                sprng(rhs_shape, rhs_dtype, n_batch=n_batch_rhs)]
    args_maker_sp_de = lambda: [sprng(lhs_shape, lhs_dtype, n_batch=n_batch_lhs),
                                jnp.array(rng(rhs_shape, rhs_dtype))]

    tol = {np.float64: 1E-13, np.complex128: 1E-13,
           np.float32: 1E-6, np.complex64: 1E-6}

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstDense(operator.matmul, operator.matmul, args_maker_de_sp, tol=tol)
      self._CheckAgainstDense(operator.matmul, operator.matmul, args_maker_sp_de, tol=tol)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, n_batch=layout.n_batch,
          n_dense=layout.n_dense)
      for lhs_shape, rhs_shape in [[(3,), ()], [(3,), (1,)], [(3,), (3,)],
                                   [(3, 4), ()], [(3, 4), (4,)], [(3, 4), (3, 1)], [(3, 4), (3, 4)],
                                   [(3, 4, 5), (4, 5)], [(3, 4, 5), (3, 1, 1)], [(3, 4, 5), (1, 4, 1)]]
      for layout in iter_sparse_layouts(lhs_shape)
    ],
    lhs_dtype=all_dtypes,
    rhs_dtype=all_dtypes,
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def test_bcoo_mul_dense(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, n_batch, n_dense):
    rng = jtu.rand_default(self.rng())
    sprng = sptu.rand_bcoo(self.rng(), n_batch=n_batch, n_dense=n_dense)

    args_maker_sp_de = lambda: [sprng(lhs_shape, lhs_dtype), jnp.array(rng(rhs_shape, rhs_dtype))]
    args_maker_de_sp = lambda: [jnp.array(rng(rhs_shape, rhs_dtype)), sprng(lhs_shape, lhs_dtype)]

    tol = {np.float64: 1E-13, np.complex128: 1E-13,
           np.float32: 1E-6, np.complex64: 1E-6}

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstDense(operator.mul, operator.mul, args_maker_de_sp, tol=tol)
      self._CheckAgainstDense(operator.mul, operator.mul, args_maker_sp_de, tol=tol)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, lhs_n_batch=lhs_n_batch,
          rhs_n_batch=rhs_n_batch, n_dense=n_dense)
      # TODO(jakevdp): add broadcasted shapes (from bcoo_mul_dense) once sparse-sparse mul
      # supports inputs of differing rank.
      for lhs_shape, rhs_shape in [[(3,), (1,)], [(3,), (3,)],
                                   [(3, 4), (1, 1)], [(3, 4), (1, 4)], [(3, 4), (3, 1)], [(3, 4), (3, 4)],
                                   [(3, 4, 5), (1, 4, 5)], [(3, 4, 5), (3, 1, 1)], [(3, 4, 5), (1, 4, 1)]]
      # TODO(jakevdp): add tests for batch & dense dimensions.
      for lhs_n_batch in range(len(lhs_shape) + 1)
      for rhs_n_batch in range(len(lhs_shape) + 1)
      for n_dense in range(len(lhs_shape) + 1 - max(lhs_n_batch, rhs_n_batch))
    ],
    lhs_dtype=all_dtypes,
    rhs_dtype=all_dtypes,
  )
  def test_bcoo_mul_sparse(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, lhs_n_batch, rhs_n_batch, n_dense):
    sprng = sptu.rand_bcoo(self.rng(), n_dense=n_dense)
    args_maker = lambda: [sprng(lhs_shape, lhs_dtype, n_batch=lhs_n_batch),
                          sprng(rhs_shape, rhs_dtype, n_batch=rhs_n_batch)]

    tol = {np.float64: 1E-13, np.complex128: 1E-13,
           np.float32: 1E-5, np.complex64: 1E-5}

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstDense(operator.mul, operator.mul, args_maker, tol=tol)

  def test_bcoo_mul_sparse_with_duplicates(self):
    # Regression test for https://github.com/google/jax/issues/8888
    indices = jnp.array([[0, 1, 0, 0, 1, 1],
                         [1, 0, 1, 2, 0, 2]]).T
    data = jnp.array([1, 2, 3, 4, 5, 6])
    mat = sparse.BCOO((data, indices), shape=(3, 3))
    self.assertArraysEqual((mat * mat).todense(), mat.todense() * mat.todense())

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
       for shape in [(), (3,), (3, 5), (3, 5, 4)]
       for layout in iter_sparse_layouts(shape)],
    dtype=all_dtypes,
  )
  def test_bcoo_broadcast_in_dim(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    x = jnp.array(rng(shape, dtype))
    xsp = sparse.BCOO.fromdense(x, n_batch=n_batch, n_dense=n_dense)

    self.assertEqual(xsp[None].n_batch, xsp.n_batch + 1)
    self.assertArraysEqual(xsp[None].todense(), x[None])

    if len(shape) >= 1:
      self.assertEqual(xsp[:, None].n_batch, xsp.n_batch if xsp.n_batch < 1 else xsp.n_batch + 1)
      self.assertArraysEqual(xsp[:, None].todense(), x[:, None])
      self.assertArraysEqual(xsp[:, None, None].todense(), x[:, None, None])
    if len(shape) >= 2:
      self.assertEqual(xsp[:, :, None].n_batch, xsp.n_batch if xsp.n_batch < 2 else xsp.n_batch + 1)
      self.assertArraysEqual(xsp[:, :, None].todense(), x[:, :, None])
      self.assertArraysEqual(xsp[:, None, :, None].todense(), x[:, None, :, None])

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense, dimension=dimension)
       for shape in [ (3,), (3, 5), (3, 5, 4)]
       for layout in iter_sparse_layouts(shape)
       for dimension in range(len(shape) - layout.n_dense)  # Concatenation of dense dimensions not implemented.
    ],
    dtype=all_dtypes,
  )
  def test_bcoo_concatenate(self, shape, dtype, n_batch, n_dense, dimension):
    sprng = sptu.rand_bcoo(self.rng(), n_batch=n_batch, n_dense=n_dense)
    args_maker = lambda: [[sprng(shape, dtype) for i in range(3)]]
    dense_func = partial(lax.concatenate, dimension=dimension)
    sparse_func = partial(sparse.bcoo_concatenate, dimension=dimension)

    self._CheckAgainstDense(dense_func, sparse_func, args_maker)
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(dense_func, sparse_func, args_maker)

  @jtu.sample_product(
    lhs_shape=[(1, 1, 5), (1, 1, 10)],
    rhs_shape=[(1, 1, 5), (1, 1, 10)],
    padding=['SAME', 'VALID', [(3, 3)]],
    dtype=jtu.dtypes.inexact,
    format=['sp-de', 'de-sp', 'sp-sp']
  )
  @jax.default_matmul_precision("float32")
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_bcoo_conv_general_dilated(self, lhs_shape, rhs_shape, dtype, padding, format):
    kwds = dict(window_strides=(1,), padding=padding)
    sparse_fun = partial(sparse.bcoo_conv_general_dilated, **kwds)
    dense_fun = partial(lax.conv_general_dilated, **kwds)
    sprng = sptu.rand_bcoo(self.rng(), n_batch=2, n_dense=0)
    rng = jtu.rand_default(self.rng())

    def args_maker():
      lhs = (sprng if format.startswith('sp') else rng)(lhs_shape, dtype)
      rhs = (sprng if format.endswith('sp') else rng)(rhs_shape, dtype)
      return lhs, rhs

    tol = {np.float32: 1E-5, np.complex64: 1E-5, np.float64: 1E-14, np.complex128: 1E-14}
    self._CheckAgainstDense(dense_fun, sparse_fun, args_maker, tol=tol)

  def test_bcoo_vmap_shape(self, shape=(2, 3, 4, 5), dtype=np.float32):
    # This test checks that BCOO shape metadata interacts correctly with vmap.
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)

    def make_bcoo(M):
      return sparse_bcoo._bcoo_fromdense(M, nse=math.prod(M.shape[:-1]), n_dense=1)

    todense = partial(sparse_bcoo._bcoo_todense, spinfo=sparse_util.SparseInfo(shape))

    for _ in range(3):
      make_bcoo = jax.vmap(make_bcoo)
      Msp_data, Msp_indices = make_bcoo(M)
      Msp_dense = todense(Msp_data, Msp_indices)
      self.assertEqual(Msp_dense.shape, M.shape)
      self.assertArraysEqual(Msp_dense, M)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense,
          n_batch_out=layout_out.n_batch, n_dense_out=layout_out.n_dense)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)
      for layout_out in iter_sparse_layouts(shape)
    ],
    dtype=jtu.dtypes.integer,
  )
  def test_bcoo_update_layout(self, shape, dtype, n_batch, n_batch_out, n_dense, n_dense_out):
    rng = rand_sparse(self.rng())
    mat = sparse.BCOO.fromdense(rng(shape, dtype), n_batch=n_batch, n_dense=n_dense)
    kwds = dict(n_batch=n_batch_out, n_dense=n_dense_out)
    # TODO(jakevdp): in case of length-0 or length-1 shapes errors/warnings will not be raised.
    if n_dense_out > n_dense or n_batch_out > n_batch:
      with self.assertRaises(sparse.SparseEfficiencyError):
        sparse.bcoo_update_layout(mat, **kwds)
      with self.assertRaises(sparse.SparseEfficiencyError):
        sparse.bcoo_update_layout(mat, **kwds, on_inefficient='error')
      with self.assertWarns(sparse.SparseEfficiencyWarning):
        sparse.bcoo_update_layout(mat, **kwds, on_inefficient='warn')
      kwds['on_inefficient'] = None
    mat_new = sparse.bcoo_update_layout(mat, **kwds)
    self.assertEqual(mat_new.n_batch, n_batch_out)
    self.assertEqual(mat_new.n_dense, n_dense_out)
    self.assertArraysEqual(mat.todense(), mat_new.todense())

  def test_bcoo_update_layout_method(self, shape=(2, 3, 4)):
    # simple test to make sure update_layout method properly forwards.
    rng = rand_sparse(self.rng())
    mat = sparse.BCOO.fromdense(rng((2, 3, 4), 'float32'), n_batch=1, n_dense=1)
    mat_new = mat.update_layout(n_batch=0, n_dense=0)
    self.assertEqual(mat_new.n_batch, 0)
    self.assertEqual(mat_new.n_dense, 0)
    self.assertArraysEqual(mat.todense(), mat_new.todense())

  def test_bcoo_bad_fillvals(self):
    # Extra values have 100 rather than zero. This lets us check that logic is
    # properly ignoring these indices.
    data = jnp.array([1, 2, 3, 100, 100])
    indices = jnp.array([1, 2, 3, 5, 5])[:, None]
    x_sp = sparse.BCOO((data, indices), shape=(5,))
    x_de = x_sp.todense()

    data = jnp.array([3, 2, 100, 100])
    indices = jnp.array([2, 3, 5, 5])[:, None]
    y_sp = sparse.BCOO((data, indices), shape=(5,))
    y_de = y_sp.todense()

    self.assertArraysEqual(x_de, jnp.array([0, 1, 2, 3, 0]))
    self.assertArraysEqual(y_de, jnp.array([0, 0, 3, 2, 0]))

    self.assertArraysEqual(x_sp.sum_duplicates().todense(), x_de)
    self.assertArraysEqual(y_sp.sum_duplicates().todense(), y_de)

    # reduce_sum:
    self.assertArraysEqual(x_sp.sum(), x_de.sum())

    # bcoo_dot_general
    self.assertArraysEqual(x_sp @ y_de, x_de @ y_de)

    # bcoo_rdot_general
    self.assertArraysEqual(x_de @ y_sp, x_de @ y_de)

    # bcoo_spdot_general
    self.assertArraysEqual((x_sp @ y_sp).todense(), x_de @ y_de)
    self.assertArraysEqual((y_sp @ x_sp).todense(), y_de @ x_de)


# TODO(tianjianlu): Unify the testing for BCOOTest and BCSRTest.
@unittest.skipIf(platform.system() == "Windows", "Test crashes on Windows")
class BCSRTest(sptu.SparseTestCase):

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_bcsr_layouts(shape)],
    dtype=all_dtypes,
  )
  def test_bcsr_dense_round_trip(self, shape, dtype, n_batch, n_dense):
    n_sparse = len(shape) - n_batch - n_dense
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch, n_dense=n_dense)
    def round_trip(M):
      return sparse.BCSR.fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense).todense()
    args_maker = lambda: [M]
    ident = lambda x: x

    self._CheckAgainstNumpy(ident, round_trip, args_maker)
    self._CompileAndCheck(round_trip, args_maker)
    self._CheckBatchingSparse(ident, round_trip, args_maker, bdims=self._random_bdims(n_batch))
    if jnp.issubdtype(dtype, jnp.floating):
      # For n_sparse != 0, we can't use an identity because output zeros must not
      # be dependent on input zeros. This mimics the code in count_stored_elements().
      def expected(M):
        if n_sparse == 0: return M
        mask = (M != 0).any(range(M.ndim - n_dense, M.ndim), keepdims=True)
        return jnp.where(mask, M, 0)
      self._CheckGradsSparse(expected, round_trip, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=n_batch)
      for shape in [(5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for n_batch in range(len(shape) - 1)
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcsr_bcoo_round_trip(self, shape, n_batch, dtype):
    n_sparse = 2
    n_dense = len(shape) - n_sparse - n_batch
    rng = self.rng()
    sprng = sptu.rand_bcsr(rng, n_batch=n_batch, n_dense=n_dense)

    M_bcsr = sprng(shape, dtype)
    self.assertIsInstance(M_bcsr, sparse.BCSR)

    M_dense = M_bcsr.todense()
    M_bcoo = M_bcsr.to_bcoo()
    self.assertIsInstance(M_bcoo, sparse.BCOO)
    self.assertAllClose(M_dense, M_bcoo.todense())

    M_bcsr2 = sparse.BCSR.from_bcoo(M_bcoo)
    self.assertAllClose(M_dense, M_bcsr2.todense())
    self.assertArraysEqual(M_bcsr.indptr, M_bcsr2.indptr)

    # TODO(jakevdp): This will only be true in general when M_bcsr.indices is sorted.
    # self.assertSparseArraysEquivalent(M_bcsr, M_bcsr2)

  @jtu.sample_product(
    [dict(shape=shape, n_batch=n_batch)
      for shape in [(5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for n_batch in range(len(shape) - 1)
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcsr_extract(self, shape, dtype, n_batch):
    n_dense = len(shape) - n_batch - 2
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices, indptr = sparse_bcsr._bcsr_fromdense(
        M, nse=nse, n_batch=n_batch, n_dense=n_dense)
    data2 = sparse.bcsr_extract(indices, indptr, M)
    self.assertArraysEqual(data, data2)
    args_maker_bcsr_extract = lambda: [indices, indptr, M]
    self._CompileAndCheck(sparse.bcsr_extract, args_maker_bcsr_extract)

  @jtu.sample_product(
    props=_generate_batched_dot_general_properties(
        shapes=((2, 3), (2, 3, 4), (2, 3, 4, 4)), sparse_format='bcsr'),
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def test_bcsr_dot_general(self, dtype: np.dtype, props: BatchedDotGeneralProperties):
    rng = jtu.rand_default(self.rng())
    sprng = sptu.rand_bcsr(self.rng(), n_batch=props.n_batch, n_dense=props.n_dense)
    args_maker = lambda: [sprng(props.lhs_shape, dtype),
                          rng(props.rhs_shape, dtype)]
    dense_fun = partial(lax.dot_general,
                        dimension_numbers=props.dimension_numbers)
    sparse_fun = partial(sparse.bcsr_dot_general,
                         dimension_numbers=props.dimension_numbers)

    tol = {np.float64: 1E-12, np.complex128: 1E-12,
           np.float32: 1E-5, np.complex64: 1E-5}

    self._CheckAgainstDense(dense_fun, sparse_fun, args_maker, tol=tol)
    if jnp.issubdtype(dtype, jnp.floating) and props.n_dense == 0:
      # Dense dimensions not yet fully supported in reverse mode.
      modes = ['fwd'] if props.n_dense != 0 else ['fwd', 'rev']
      self._CheckGradsSparse(dense_fun, sparse_fun, args_maker, modes=modes, atol=tol, rtol=tol)
    self._CheckBatchingSparse(dense_fun, sparse_fun, args_maker, atol=tol, rtol=tol,
                              bdims=self._random_bdims(props.n_batch, len(props.rhs_shape)))

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
       for shape in [(3, 5), (3, 5, 4)]
       for layout in iter_bcsr_layouts(shape)],
    dtype=all_dtypes,
  )
  def test_bcsr_broadcast_in_dim(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    x = jnp.array(rng(shape, dtype))
    xsp = sparse.BCSR.fromdense(x, n_batch=n_batch, n_dense=n_dense)

    self.assertEqual(xsp[None].n_batch, xsp.n_batch + 1)
    self.assertArraysEqual(xsp[None].todense(), x[None])

    if n_batch == 1:
      self.assertEqual(xsp[:, None].n_batch, xsp.n_batch + 1)
      self.assertArraysEqual(xsp[:, None].todense(), x[:, None])

  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense, dimension=dimension)
       for shape in [(3, 5), (3, 5, 4)]
       for layout in iter_sparse_layouts(shape)
       for dimension in range(len(shape) - layout.n_dense)  # Concatenation of dense dimensions not implemented.
    ],
    dtype=all_dtypes,
  )
  def test_bcsr_concatenate(self, shape, dtype, n_batch, n_dense, dimension):
    sprng = sptu.rand_bcoo(self.rng(), n_batch=n_batch, n_dense=n_dense)
    args_maker = lambda: [[sprng(shape, dtype) for i in range(3)]]
    dense_func = partial(lax.concatenate, dimension=dimension)
    sparse_func = partial(sparse.bcoo_concatenate, dimension=dimension)

    self._CheckAgainstDense(dense_func, sparse_func, args_maker)
    if jnp.issubdtype(dtype, jnp.floating):
      self._CheckGradsSparse(dense_func, sparse_func, args_maker)


class SparseGradTest(sptu.SparseTestCase):
  @jtu.sample_product(has_aux=[True, False])
  def test_sparse_value_and_grad(self, has_aux):
    rng_sparse = rand_sparse(self.rng())
    rng = jtu.rand_default(self.rng())

    y = rng(5, "float32")
    X = rng_sparse((10, 5), "float32")
    Xsp = sparse.BCOO.fromdense(X)

    def f(X, y):
      if has_aux:
        return jnp.sum(X @ y), {'X': X.shape, 'y': y.shape}
      return jnp.sum(X @ y)

    with self.subTest("wrt sparse"):
      val_de, grad_de = jax.value_and_grad(f, argnums=0, has_aux=has_aux)(X, y)
      val_sp, grad_sp = sparse.value_and_grad(f, argnums=0, has_aux=has_aux)(Xsp, y)
      self.assertIsInstance(grad_sp, sparse.BCOO)
      self.assertAllClose(val_de, val_sp)
      self.assertAllClose(grad_sp.data, sparse_bcoo._bcoo_extract(grad_sp.indices, grad_de))

    with self.subTest("wrt dense"):
      self.assertAllClose(jax.value_and_grad(f, argnums=1, has_aux=has_aux)(X, y),
                          sparse.value_and_grad(f, argnums=1, has_aux=has_aux)(Xsp, y))

  @jtu.sample_product(has_aux=[True, False])
  def test_sparse_grad(self, has_aux):
    rng_sparse = rand_sparse(self.rng())
    rng = jtu.rand_default(self.rng())

    y = rng(5, "float32")
    X = rng_sparse((10, 5), "float32")
    Xsp = sparse.BCOO.fromdense(X)

    def f(X, y):
      if has_aux:
        return jnp.sum(X @ y), {'X': X.shape, 'y': y.shape}
      return jnp.sum(X @ y)

    with self.subTest("wrt sparse"):
      grad_de = jax.grad(f, argnums=0, has_aux=has_aux)(X, y)
      grad_sp = sparse.grad(f, argnums=0, has_aux=has_aux)(Xsp, y)
      if has_aux:
        grad_de, aux_de = grad_de
        grad_sp, aux_sp = grad_sp
        self.assertAllClose(aux_de, aux_sp)
      self.assertIsInstance(grad_sp, sparse.BCOO)
      self.assertAllClose(grad_sp.data, sparse_bcoo._bcoo_extract(grad_sp.indices, grad_de))

    with self.subTest("wrt dense"):
      self.assertAllClose(jax.grad(f, argnums=1, has_aux=has_aux)(X, y),
                          sparse.grad(f, argnums=1, has_aux=has_aux)(Xsp, y))

  @jtu.sample_product(
    has_aux=[True, False],
    transform=['jacrev', 'jacfwd', 'jacobian']
  )
  @jax.default_matmul_precision("float32")
  def test_sparse_jacobian(self, has_aux, transform):
    jac_dense = getattr(jax, transform)
    jac_sparse = getattr(sparse, transform)

    rng_sparse = rand_sparse(self.rng())
    rng = jtu.rand_default(self.rng())

    y = rng(5, "float32")
    X = rng_sparse((10, 5), "float32")
    Xsp = sparse.BCOO.fromdense(X)

    def f(X, y):
      if has_aux:
        return X @ y, {'X': X.shape, 'y': y.shape}
      return X @ y

    with self.subTest("wrt sparse"):
      grad_de = jac_dense(f, argnums=0, has_aux=has_aux)(X, y)
      grad_sp = jac_sparse(f, argnums=0, has_aux=has_aux)(Xsp, y)
      if has_aux:
        grad_de, aux_de = grad_de
        grad_sp, aux_sp = grad_sp
        self.assertAllClose(aux_de, aux_sp)
      self.assertIsInstance(grad_sp, sparse.BCOO)
      self.assertAllClose(grad_sp.data, sparse_bcoo._bcoo_extract(grad_sp.indices, grad_de))

    with self.subTest("wrt dense"):
      rtol = 0.01 if jtu.device_under_test() == 'tpu' else None
      self.assertAllClose(jac_dense(f, argnums=1, has_aux=has_aux)(X, y),
                          jac_sparse(f, argnums=1, has_aux=has_aux)(Xsp, y), rtol=rtol)


class SparseObjectTest(sptu.SparseTestCase):
  @parameterized.named_parameters(
    {"testcase_name": f"_{cls.__name__}", "cls": cls}
    for cls in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO, sparse.BCSR])
  def test_pytree_flattening(self, cls):
    sparse_format = cls.__name__.lower()
    M = sparse.empty((2, 4), sparse_format=sparse_format)
    self.assertIsInstance(M, cls)
    buffers, tree = tree_util.tree_flatten(M)
    self.assertTrue(all([isinstance(buffer, jax.Array) for buffer in buffers]))
    M_out = tree_util.tree_unflatten(tree, buffers)
    self.assertEqual(M.dtype, M_out.dtype)
    self.assertEqual(M.shape, M_out.shape)
    self.assertEqual(M.nse, M_out.nse)

  @parameterized.named_parameters(
    {"testcase_name": f"_{cls.__name__}", "cls": cls}
    for cls in [sparse.BCOO, sparse.BCSR])
  def test_vmappable(self, cls):
    # Note: test should avoid dependence on batching rules of BCOO/BCSR primitives
    M = jnp.arange(24).reshape((2, 3, 4))
    Msp = cls.fromdense(M, n_batch=1)

    def from_elt(x):
      assert x.ndim == 2
      return sparse.empty(x.shape, x.dtype, sparse_format=cls.__name__.lower())

    with self.subTest('from_elt'):
      M_out = vmap(from_elt)(M)
      self.assertIsInstance(M_out, cls)
      self.assertEqual(M_out.n_batch, 1)
      self.assertEqual(M.shape, M_out.shape)

    def to_elt(x):
      assert x.ndim == 2
      assert x.n_sparse == 2
      return jnp.empty(x.shape, x.dtype)

    with self.subTest('to_elt'):
      M_out = vmap(to_elt)(Msp)
      self.assertIsInstance(M_out, jax.Array)
      self.assertEqual(Msp.shape, M_out.shape)

    with self.subTest('axis_None'):
      x, y = vmap(lambda *args: args, in_axes=(0, None), out_axes=(0, None))(Msp, Msp)
      self.assertIsInstance(x, cls)
      self.assertEqual(x.n_batch, 1)
      self.assertEqual(x.shape, Msp.shape)
      self.assertEqual(x._info, Msp._info)

      self.assertIsInstance(y, cls)
      self.assertEqual(y.n_batch, 1)
      self.assertEqual(y.shape, Msp.shape)
      self.assertEqual(y._info, Msp._info)

  @parameterized.named_parameters(
    {"testcase_name": f"_{cls.__name__}", "cls": cls}
    for cls in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO])
  def test_jit_lower(self, cls):
    sparse_format = cls.__name__.lower()
    M = sparse.empty((2, 4), sparse_format=sparse_format)
    self.assertIsInstance(M, cls)
    jax.jit(lambda x: x).lower(M)  # doesn't crash

  @parameterized.named_parameters(
    {"testcase_name": f"_{cls.__name__}{shape}", "cls": cls, "shape": shape}
    for cls in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]
    for shape in ([2, 5], [5, 3]))
  def test_empty(self, cls, shape):
    sparse_format = cls.__name__.lower()
    M = sparse.empty(shape, sparse_format=sparse_format)
    self.assertIsInstance(M, cls)
    self.assertEqual(M.nse, 0)
    self.assertArraysEqual(M.todense(), jnp.empty(shape))

  @parameterized.named_parameters(
    {"testcase_name": f"_{cls.__name__}{(N, M, k)}",
     "cls": cls, "N": N, "M": M, "k": k}
    for cls in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]
    for N in [2, 5]
    for M in [None, 3]
    for k in [-2, 0, 1])
  def test_eye(self, cls, N, M, k):
    sparse_format = cls.__name__.lower()
    func = partial(sparse.eye, N, M, k, sparse_format=sparse_format)
    expected = jnp.eye(N, M, k)
    expected_nse = jnp.count_nonzero(expected)

    mat = func()
    self.assertIsInstance(mat, cls)
    self.assertArraysEqual(mat.todense(), expected)
    self.assertEqual(mat.nse, expected_nse)

    mat_jit = jit(func)()
    self.assertIsInstance(mat_jit, cls)
    self.assertArraysEqual(mat_jit.todense(), expected)
    self.assertEqual(mat_jit.nse, expected_nse)

  @parameterized.named_parameters(
    {"testcase_name": f"{nse}_BCOO{shape}", "shape": shape, "nse": nse}
    for shape in ([2, 5], [5, 3])
    for nse in [0, 2])
  def test_empty_nse(self, shape, nse=2):
    M = sparse.empty(shape, nse=nse)
    self.assertEqual(M.nse, nse)
    self.assertArraysEqual(M.todense(), jnp.empty(shape))

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO])
  def test_block_until_ready(self, Obj, shape=(5, 8), dtype=np.float32):
    rng = rand_sparse(self.rng(), post=Obj.fromdense)
    M = rng(shape, dtype)
    self.assertEqual(M.shape, M.block_until_ready().shape)
    self.assertArraysEqual(M.data, M.block_until_ready().data)
    self.assertArraysEqual(M.todense(), M.block_until_ready().todense())

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [jnp.array, sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO])
  def test_todense(self, Obj, shape=(5, 8), dtype=np.float32):
    rng = rand_sparse(self.rng())
    M_dense = rng(shape, dtype)
    M = jnp.array(M_dense) if Obj is jnp.array else Obj.fromdense(M_dense)
    self.assertArraysEqual(sparse.todense(M), M_dense)
    self.assertArraysEqual(jit(sparse.todense)(M), M_dense)

  def test_todense_scalar(self):
    self.assertEqual(sparse.todense(1.0), 1.0)
    self.assertEqual(jit(sparse.todense)(1.0), 1.0)

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [jnp.array, sparse.BCOO])
  def test_todense_batching(self, Obj, shape=(5, 8), dtype=np.float32):
    rng = rand_sparse(self.rng())
    M_dense = rng(shape, dtype)
    if Obj is sparse.BCOO:
      M = sparse.BCOO.fromdense(M_dense, n_batch=1)
    else:
      M = jnp.asarray(M_dense)
    self.assertArraysEqual(vmap(sparse.todense)(M), M_dense)
    self.assertArraysEqual(jit(vmap(sparse.todense))(M), M_dense)

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [jnp.array, sparse.BCOO])
  def test_todense_ad(self, Obj, shape=(3,), dtype=np.float32):
    M_dense = jnp.array([1., 2., 3.])
    M = M_dense if Obj is jnp.array else Obj.fromdense(M_dense)
    bufs, tree = tree_util.tree_flatten(M)
    jac = jnp.eye(M.shape[0], dtype=M.dtype)
    jac1 = jax.jacfwd(lambda *bufs: sparse.todense_p.bind(*bufs, tree=tree))(*bufs)
    jac2 = jax.jacrev(lambda *bufs: sparse.todense_p.bind(*bufs, tree=tree))(*bufs)
    self.assertArraysEqual(jac1, jac2)
    self.assertArraysEqual(jac, jac2)

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO, sparse.BCSR])
  def test_attrs(self, Obj, shape=(5, 8), dtype=np.float32):
    rng = rand_sparse(self.rng(), post=Obj.fromdense)
    M = rng(shape, dtype)

    self.assertIsInstance(M, Obj)
    self.assertEqual(M.shape, shape)
    self.assertEqual(M.size, math.prod(shape))
    self.assertEqual(M.ndim, len(shape))
    self.assertEqual(M.dtype, dtype)
    self.assertEqual(M.nse, (M.todense() != 0).sum())
    self.assertEqual(M.data.dtype, dtype)
    self.assertEqual(len(M), M.shape[0])

    with self.assertRaises(TypeError):
      hash(M)

    if isinstance(M, sparse.CSR):
      self.assertEqual(len(M.data), len(M.indices))
      self.assertEqual(len(M.indptr), M.shape[0] + 1)
    elif isinstance(M, sparse.CSC):
      self.assertEqual(len(M.data), len(M.indices))
      self.assertEqual(len(M.indptr), M.shape[1] + 1)
    elif isinstance(M, sparse.COO):
      self.assertEqual(len(M.data), len(M.row))
      self.assertEqual(len(M.data), len(M.col))
    elif isinstance(M, sparse.BCOO):
      self.assertEqual(M.data.shape[M.n_batch], M.indices.shape[-2])
      self.assertEqual(M.indices.shape[-1], M.n_sparse)
    elif isinstance(M, sparse.BCSR):
      self.assertEqual(M.data.shape[M.n_batch], M.indices.shape[-1])
      self.assertEqual(M.indptr.shape[-1], M.shape[M.n_batch] + 1)
    else:
      raise ValueError(f"{Obj=} not expected.")

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      Obj=[Obj],
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=jtu.dtypes.floating + jtu.dtypes.complex,
    )
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  def test_dense_round_trip(self, shape, dtype, Obj):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    Msparse = Obj.fromdense(M)
    self.assertArraysEqual(M, Msparse.todense())

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      Obj=[Obj],
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=jtu.dtypes.floating + jtu.dtypes.complex,
    )
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  def test_transpose(self, shape, dtype, Obj):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    Msparse = Obj.fromdense(M)
    self.assertArraysEqual(M.T, Msparse.T.todense())

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(shape=shape, bshape=bshape)
       for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
       for bshape in [shape[-1:] + s for s in [(), (3,), (4,)]]
      ],
      Obj=[Obj],
      dtype=jtu.dtypes.floating + jtu.dtypes.complex,
    )
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  @jax.default_matmul_precision("float32")
  def test_matmul(self, shape, dtype, Obj, bshape):
    rng = rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())
    M = rng(shape, dtype)
    Msp = Obj.fromdense(M)

    # Test matching type
    x = rng_b(bshape, dtype)
    x = jnp.asarray(x)
    self.assertAllClose(M @ x, Msp @ x, rtol=MATMUL_TOL)

    # Test mismatched type
    x = rng_b(bshape, np.int32)
    x = jnp.asarray(x)
    with jax.numpy_dtype_promotion('standard'):
      self.assertAllClose(M @ x, Msp @ x, rtol=MATMUL_TOL)

  @jtu.sample_product(
    cls=[sparse.BCOO, sparse.BCSR],
    input_type=[scipy.sparse.coo_matrix, scipy.sparse.csr_matrix,
                scipy.sparse.csc_matrix],
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcoo_bcsr_from_scipy_sparse(self, cls, input_type, shape, dtype):
    """Test BCOO and BCSR from_scipy_sparse."""
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_scipy = input_type(M)
    M_jax = cls.from_scipy_sparse(M_scipy)
    self.assertArraysEqual(M, M_jax.todense())

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

    self.assertArraysEqual(M.astype(float), Msp.astype(float).todense())

  @jtu.sample_product(
    [dict(shape=shape, n_batch=n_batch)
      for shape in [(5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for n_batch in range(len(shape) - 1)
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcoo_to_bcsr_round_trip(self, shape, dtype, n_batch):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    n_dense = len(shape) - 2 - n_batch
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    _, bcoo_indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch,
                                                  n_dense=n_dense)

    bcoo_to_bcsr = partial(sparse_bcsr._bcoo_to_bcsr, shape=shape)

    args_maker_bcoo_to_bcsr = lambda: [bcoo_indices]
    self._CompileAndCheck(bcoo_to_bcsr, args_maker_bcoo_to_bcsr)

    bcsr_indices, indptr = bcoo_to_bcsr(bcoo_indices)

    self.assertEqual(bcsr_indices.dtype, jnp.int32)
    self.assertEqual(bcsr_indices.shape, shape[:n_batch] + (nse,))
    self.assertEqual(indptr.dtype, jnp.int32)
    self.assertEqual(indptr.shape, shape[:n_batch] + (shape[n_batch] + 1,))

    bcsr_to_bcoo = partial(sparse_bcsr._bcsr_to_bcoo, shape=shape)
    self.assertArraysEqual(bcoo_indices, bcsr_to_bcoo(bcsr_indices, indptr))
    args_maker_bcsr_to_bcoo = lambda: [bcsr_indices, indptr]
    self._CompileAndCheck(bcsr_to_bcoo, args_maker_bcsr_to_bcoo)


class SparseRandomTest(sptu.SparseTestCase):
  @jtu.sample_product(
    [dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for layout in iter_sparse_layouts(shape)],
    dtype=jtu.dtypes.floating,
    indices_dtype=jtu.dtypes.integer,
  )
  def test_random_bcoo(self, shape, dtype, indices_dtype, n_batch, n_dense):
    key = jax.random.PRNGKey(1701)
    with jax.legacy_prng_key('allow'):
      mat = sparse.random_bcoo(
          key, shape=shape, dtype=dtype, indices_dtype=indices_dtype,
          n_batch=n_batch, n_dense=n_dense)

    mat_dense = mat.todense()
    self.assertEqual(mat_dense.shape, shape)
    self.assertEqual(mat_dense.dtype, dtype)
    self.assertEqual(mat.indices.dtype, indices_dtype)

    n_sparse = len(shape) - n_batch - n_dense
    batch_shape, sparse_shape, dense_shape = split_list(shape, [n_batch, n_sparse])

    approx_expected_num_nonzero = (
      np.ceil(0.2 * math.prod(sparse_shape))
      * math.prod(batch_shape) * math.prod(dense_shape))
    num_nonzero = (mat_dense != 0).sum()
    self.assertAlmostEqual(int(num_nonzero), approx_expected_num_nonzero, delta=2)


class SparseSolverTest(sptu.SparseTestCase):
  @jtu.sample_product(
    size=[20, 50, 100],
    reorder=[0, 1, 2, 3],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @unittest.skipIf(jtu.device_under_test() == "tpu", "test requires CPU or GPU")
  @unittest.skipIf(jtu.device_under_test() == "cuda" and not GPU_LOWERING_ENABLED,
                   "test requires cusparse/cusolver")
  @jtu.skip_on_devices("rocm", "test n gpu requires cusolver")
  def test_sparse_qr_linear_solver(self, size, reorder, dtype):
    rng = rand_sparse(self.rng())
    a = rng((size, size), dtype)
    nse = (a != 0).sum()
    data, indices, indptr = sparse_csr._csr_fromdense(a, nse=nse)

    rng_k = jtu.rand_default(self.rng())
    b = rng_k([size], dtype)

    def args_maker():
      return data, indices, indptr, b

    tol = 1e-8
    def sparse_solve(data, indices, indptr, b):
      return sparse.linalg.spsolve(data, indices, indptr, b, tol, reorder)
    x = sparse_solve(data, indices, indptr, b)

    self.assertAllClose(a @ x, b, rtol=1e-2, atol=1e-3)
    self._CompileAndCheck(sparse_solve, args_maker)

  @jtu.sample_product(
    size=[10, 20, 50],
    dtype=jtu.dtypes.floating,
  )
  @unittest.skipIf(jtu.device_under_test() == "tpu", "test requires CPU or GPU")
  @unittest.skipIf(jtu.device_under_test() == "cuda" and not GPU_LOWERING_ENABLED,
                   "test requires cusparse/cusolver")
  @jtu.skip_on_devices("rocm", "test requires cusolver")
  def test_sparse_qr_linear_solver_grads(self, size, dtype):
    rng = rand_sparse(self.rng())
    a = rng((size, size), dtype)
    nse = (a != 0).sum()
    data, indices, indptr = sparse_csr._csr_fromdense(a, nse=nse)

    rng_k = jtu.rand_default(self.rng())
    b = rng_k([size], dtype)

    def sparse_solve(data, b, tol=1e-8):
      return sparse.linalg.spsolve(data, indices, indptr, b, tol=tol)

    jtu.check_grads(sparse_solve, (data, b), order=1, rtol=0.05, atol=0.05)


class SparseUtilTest(sptu.SparseTestCase):

  @jtu.sample_product(
    [dict(n_batch=n_batch, n_dense=n_dense, expected_nse=expected_nse)
      for n_batch, n_dense, expected_nse in
        [(0, 0, 4), (1, 0, 2), (0, 1, 2), (2, 0, 1), (1, 1, 1), (0, 2, 1)]
    ],
    dtype=all_dtypes,
  )
  def test_count_stored_elements(self, dtype, n_batch, n_dense, expected_nse):
    """Test counting nse."""
    mat = np.array([[1, 0, 2, 0], [0, 0, 0, 0], [0, 3, 0, 4]], dtype=dtype)
    actual_nse = sparse.util._count_stored_elements(
        mat, n_batch=n_batch, n_dense=n_dense)
    self.assertEqual(expected_nse, actual_nse)

  @jtu.sample_product(
    [dict(n_batch=n_batch, n_dense=n_dense)
     for n_batch in range(3)
     for n_dense in range(3 - n_batch)
    ],
    dtype=all_dtypes,
  )
  def test_count_stored_elements_empty(self, dtype, n_batch, n_dense):
    mat = np.empty((0, 4), dtype=dtype)
    actual_nse = sparse.util._count_stored_elements(
        mat, n_batch=n_batch, n_dense=n_dense)
    self.assertEqual(0, actual_nse)

  @jtu.sample_product(
    [dict(n_batch=n_batch, n_dense=n_dense, expected_nse=expected_nse)
      for n_batch, n_dense, expected_nse in
        [(0, 0, 14), (1, 0, np.array([6, 8])), (0, 1, 9),
         (2, 0, np.array([[3, 3], [4, 4]]))]
    ],
    dtype=all_dtypes
  )
  def test_count_stored_elements_per_batch(self, dtype, n_batch, n_dense,
                                           expected_nse):
    """Test counting nse."""
    mat = np.array([[[[1, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 3]],
                     [[0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 3]]],
                    [[[1, 0, 2, 0], [0, 0, 0, 0], [0, 3, 0, 4]],
                     [[0, 0, 0, 1], [0, 0, 2, 0], [3, 0, 0, 4]]]], dtype=dtype)
    actual_nse = sparse.util._count_stored_elements_per_batch(
        mat, n_batch=n_batch, n_dense=n_dense)
    self.assertArraysEqual(expected_nse, actual_nse, check_dtypes=False)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
