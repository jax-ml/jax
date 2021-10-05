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
import operator

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from jax import config, core, jit, lax
import jax.numpy as jnp
import jax._src.test_util as jtu
from jax.experimental.sparse import BCOO, sparsify
from jax.experimental.sparse.transform import (
  arrays_to_argspecs, argspecs_to_arrays, sparsify_raw, ArgSpec, SparseEnv)

config.parse_flags_with_absl()


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
        (ArgSpec(X.shape, 0, None), ArgSpec(X.shape, 1, 2), ArgSpec(X.shape, 3, 4)))

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

  def testUnitHandling(self):
    x = BCOO.fromdense(jnp.arange(5))
    f = jit(lambda x, y: x)
    result = sparsify(jit(f))(x, core.unit)
    self.assertBcooIdentical(result, x)

  def testDropvar(self):
    def inner(x):
      return x * 2, x * 3

    def f(x):
      _, y = jit(inner)(x)
      return y * 4

    x_dense = jnp.arange(5)
    x_sparse = BCOO.fromdense(x_dense)
    self.assertArraysEqual(sparsify(f)(x_sparse).todense(), f(x_dense))

  def testPytreeInput(self):
    f = sparsify(lambda x: x)
    args = (jnp.arange(4), BCOO.fromdense(jnp.arange(4)))
    out = f(args)
    self.assertLen(out, 2)
    self.assertArraysEqual(args[0], out[0])
    self.assertBcooIdentical(args[1], out[1])

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

  def testSparsifyWithConsts(self):
    M_dense = jnp.arange(24).reshape(4, 6)
    M_sparse = BCOO.fromdense(M_dense)

    @sparsify
    def func(x):
      return jit(lambda x: jnp.sum(x, 1))(x)

    result_dense = func(M_dense)
    result_sparse = func(M_sparse)

    self.assertAllClose(result_sparse.todense(), result_dense)

  def testSparseMatmul(self):
    X = jnp.arange(16).reshape(4, 4)
    Xsp = BCOO.fromdense(X)
    Y = jnp.ones(4)
    Ysp = BCOO.fromdense(Y)

    # dot_general
    result_sparse = sparsify(operator.matmul)(Xsp, Y)
    result_dense = operator.matmul(X, Y)
    self.assertAllClose(result_sparse, result_dense)

    # rdot_general
    result_sparse = sparsify(operator.matmul)(Y, Xsp)
    result_dense = operator.matmul(Y, X)
    self.assertAllClose(result_sparse, result_dense)

    # spdot_general
    result_sparse = sparsify(operator.matmul)(Xsp, Ysp)
    result_dense = operator.matmul(X, Y)
    self.assertAllClose(result_sparse.todense(), result_dense)

  def testSparseAdd(self):
    x = BCOO.fromdense(jnp.arange(5))
    y = BCOO.fromdense(2 * jnp.arange(5))

    # Distinct indices
    out = sparsify(operator.add)(x, y)
    self.assertEqual(out.nse, 8)  # uses concatenation.
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

  def testSparseWhileLoop(self):
    def cond_fun(params):
      i, A = params
      return i < 5

    def body_fun(params):
      i, A = params
      return i + 1, 2 * A

    def f(A):
      return lax.while_loop(cond_fun, body_fun, (0, A))

    A = jnp.arange(4)
    out_dense = f(A)

    Asp = BCOO.fromdense(A)
    out_sparse = sparsify(f)(Asp)

    self.assertEqual(len(out_dense), 2)
    self.assertEqual(len(out_sparse), 2)
    self.assertArraysEqual(out_dense[0], out_dense[0])
    self.assertArraysEqual(out_dense[1], out_sparse[1].todense())

  def testSparseWhileLoopDuplicateIndices(self):
    def cond_fun(params):
      i, A, B = params
      return i < 5

    def body_fun(params):
      i, A, B = params
      # TODO(jakevdp): track shared indices through while loop & use this
      #   version of the test, which requires shared indices in order for
      #   the nse of the result to remain the same.
      # return i + 1, A, A + B

      # This version is fine without shared indices, and tests that we're
      # flattening non-shared indices consistently.
      return i + 1, B, A

    def f(A):
      return lax.while_loop(cond_fun, body_fun, (0, A, A))

    A = jnp.arange(4).reshape((2, 2))
    out_dense = f(A)

    Asp = BCOO.fromdense(A)
    out_sparse = sparsify(f)(Asp)

    self.assertEqual(len(out_dense), 3)
    self.assertEqual(len(out_sparse), 3)
    self.assertArraysEqual(out_dense[0], out_dense[0])
    self.assertArraysEqual(out_dense[1], out_sparse[1].todense())
    self.assertArraysEqual(out_dense[2], out_sparse[2].todense())

  def testSparsifyDenseXlaCall(self):
    # Test handling of dense xla_call within jaxpr interpreter.
    out = sparsify(jit(lambda x: x + 1))(0.0)
    self.assertEqual(out, 1.0)

  def testSparsifySparseXlaCall(self):
    # Test sparse lowering of XLA call
    def func(M):
      return 2 * M

    M = jnp.arange(6).reshape(2, 3)
    Msp = BCOO.fromdense(M)

    out_dense = func(M)
    out_sparse = sparsify(jit(func))(Msp)
    self.assertArraysEqual(out_dense, out_sparse.todense())

  def testSparseForiLoop(self):
    def func(M, x):
      body_fun = lambda i, val: (M @ val) / M.shape[1]
      return lax.fori_loop(0, 2, body_fun, x)

    x = jnp.arange(5.0)
    M = jnp.arange(25).reshape(5, 5)
    M_bcoo = BCOO.fromdense(M)

    result_dense = func(M, x)
    result_sparse = sparsify(func)(M_bcoo, x)

    self.assertArraysAllClose(result_dense, result_sparse)

  def testSparseCondSimple(self):
    def func(x):
      return lax.cond(False, lambda x: x, lambda x: 2 * x, x)

    x = jnp.arange(5.0)
    result_dense = func(x)

    x_bcoo = BCOO.fromdense(x)
    result_sparse = sparsify(func)(x_bcoo)

    self.assertArraysAllClose(result_dense, result_sparse.todense())

  def testSparseCondMismatchError(self):
    @sparsify
    def func(x, y):
      return lax.cond(False, lambda x: x[0], lambda x: x[1], (x, y))

    x = jnp.arange(5.0)
    y = jnp.arange(5.0)

    x_bcoo = BCOO.fromdense(x)
    y_bcoo = BCOO.fromdense(y)

    func(x, y)  # No error
    func(x_bcoo, y_bcoo)  # No error

    with self.assertRaisesRegex(TypeError, "sparsified true_fun and false_fun output.*"):
      func(x_bcoo, y)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
