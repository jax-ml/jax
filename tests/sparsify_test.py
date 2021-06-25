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

import operator

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from jax import lax, partial
import jax.numpy as jnp
import jax.test_util as jtu
from jax.experimental.sparse import BCOO, sparsify
from jax.experimental.sparse.transform import (
  arrays_to_argspecs, argspecs_to_arrays, sparsify_raw, ArgSpec, SparseEnv)


class SparsifyTest(jtu.JaxTestCase):

  def assertBcooIdentical(self, x, y):
    self.assertIsInstance(x, BCOO)
    self.assertIsInstance(y, BCOO)
    self.assertEqual(x.shape, y.shape)
    self.assertArraysEqual(x.data, y.data)
    self.assertArraysEqual(x.indices, y.indices)

  def testArgSpec(self):
    X = jnp.arange(5)
    X_BCOO = BCOO.fromdense(X)

    args = (X, X_BCOO, X_BCOO)

    # Independent index
    spenv = SparseEnv()
    argspecs = arrays_to_argspecs(spenv, args)
    self.assertEqual(len(argspecs), len(args))
    self.assertEqual(spenv.size(), 5)
    self.assertEqual(argspecs,
        [ArgSpec(X.shape, 0, None), ArgSpec(X.shape, 1, 2), ArgSpec(X.shape, 3, 4)])

    args_out = argspecs_to_arrays(spenv, argspecs)
    self.assertEqual(len(args_out), len(args))
    self.assertArraysEqual(args[0], args_out[0])
    self.assertBcooIdentical(args[1], args_out[1])
    self.assertBcooIdentical(args[2], args_out[2])

    # Shared index
    argspecs = (ArgSpec(X.shape, 0, None), ArgSpec(X.shape, 1, 2), ArgSpec(X.shape, 3, 2))
    spenv = SparseEnv([X, X_BCOO.data, X_BCOO.indices, X_BCOO.data])

    args_out = argspecs_to_arrays(spenv, argspecs)
    self.assertEqual(len(args_out), len(args))
    self.assertArraysEqual(args[0], args_out[0])
    self.assertBcooIdentical(args[1], args_out[1])
    self.assertBcooIdentical(args[2], args_out[2])

  def testSparsify(self):
    M_dense = jnp.arange(24).reshape(4, 6)
    M_sparse = BCOO.fromdense(M_dense)
    v = jnp.arange(M_dense.shape[0])

    @sparsify
    def func(x, v):
      return -jnp.sin(jnp.pi * x).T @ (v + 1)

    result_dense = func(M_dense, v)
    result_sparse = func(M_sparse, v)

    self.assertAllClose(result_sparse, result_dense)

  def testSparseAdd(self):
    x = BCOO.fromdense(jnp.arange(5))
    y = BCOO.fromdense(2 * jnp.arange(5))

    # Distinct indices
    out = sparsify(operator.add)(x, y)
    self.assertEqual(out.nnz, 8)  # uses concatenation.
    self.assertArraysEqual(out.todense(), 3 * jnp.arange(5))

    # Shared indices – requires lower level call
    argspecs = [
      ArgSpec(x.shape, 1, 0),
      ArgSpec(y.shape, 2, 0)
    ]
    spenv = SparseEnv([x.indices, x.data, y.data])

    result = sparsify_raw(operator.add)(spenv, *argspecs)
    args_out, _ = result
    out, = argspecs_to_arrays(spenv, args_out)

    self.assertAllClose(out.todense(), x.todense() + y.todense())

  def testSparseMul(self):
    x = BCOO.fromdense(jnp.arange(5))
    y = BCOO.fromdense(2 * jnp.arange(5))

    # Scalar multiplication
    out = sparsify(operator.mul)(x, 2.5)
    self.assertArraysEqual(out.todense(), x.todense() * 2.5)

    # Shared indices – requires lower level call
    argspecs = [
      ArgSpec(x.shape, 1, 0),
      ArgSpec(y.shape, 2, 0)
    ]
    spenv = SparseEnv([x.indices, x.data, y.data])

    result = sparsify_raw(operator.mul)(spenv, *argspecs)
    args_out, _ = result
    out, = argspecs_to_arrays(spenv, args_out)

    self.assertAllClose(out.todense(), x.todense() * y.todense())

  def testSparseSum(self):
    x = jnp.arange(20).reshape(4, 5)
    xsp = BCOO.fromdense(x)

    def f(x):
      return x.sum(), x.sum(0), x.sum(1), x.sum((0, 1))

    result_dense = f(x)
    result_sparse = sparsify(f)(xsp)

    assert len(result_dense) == len(result_sparse)

    for res_dense, res_sparse in zip(result_dense, result_sparse):
      if isinstance(res_sparse, BCOO):
        res_sparse = res_sparse.todense()
      self.assertArraysAllClose(res_dense, res_sparse)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dimensions={}_nbatch={}, ndense={}".format(
          jtu.format_shape_dtype_string(shape, np.float32), dimensions, n_batch, n_dense),
       "shape": shape, "dimensions": dimensions, "n_batch": n_batch, "n_dense": n_dense}
      for shape, dimensions in [
          [(1,), (0,)],
          [(1,), (-1,)],
          [(2, 1, 4), (1,)],
          [(2, 1, 3, 1), (1,)],
          [(2, 1, 3, 1), (1, 3)],
          [(2, 1, 3, 1), (3,)],
      ]
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) - n_batch + 1)))
  def testSparseSqueeze(self, shape, dimensions, n_batch, n_dense):
    rng = jtu.rand_default(self.rng())

    M_dense = rng(shape, np.float32)
    M_sparse = BCOO.fromdense(M_dense, n_batch=n_batch, n_dense=n_dense)
    func = sparsify(partial(lax.squeeze, dimensions=dimensions))

    result_dense = func(M_dense)
    result_sparse = func(M_sparse).todense()

    self.assertAllClose(result_sparse, result_dense)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
